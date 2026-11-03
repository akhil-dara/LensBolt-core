//! Reading cells from a camera image: box-blurred sampling, pooled pilot alignment, per-block colour
//! models. This is the mechanism measured in tools/channel_pilot.py (docs/2026-10-27-*): one corner
//! homography cannot follow a rolling shutter plus a moving hand, so every 8x8 block of cells gets its
//! own sampling shift, chosen from the pilots of the block and its neighbours.

use super::colour::ColourModel;
use super::format::MODULE_PX;
use super::header::{FlatHeader, HeaderCopy};
use super::layout::{Chrome, FlatCell, FlatGrid, BLOCK, PILOT_LOCAL};
use crate::detect::{two_means_threshold, Homography};
use crate::image::RgbImage;
use crate::par::{par_map, par_rows_mut};

/// Box filter of side `k` (k = 1 returns a copy): what the cell sampler reads from, so that a sample
/// averages over the central part of a cell instead of hitting one camera pixel (and the screen's own
/// pixel grid) at a time.
pub fn box_blur(img: &RgbImage, k: u32) -> RgbImage {
    if k <= 1 { return img.clone(); }
    let (w, h) = (img.width as usize, img.height as usize);
    let stride = w + 1;
    let mut sum = vec![[0u32; 3]; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row = [0u32; 3];
        for x in 0..w {
            let p = img.get(x as u32, y as u32);
            for c in 0..3 { row[c] += p[c] as u32; }
            let above = sum[y * stride + x + 1];
            sum[(y + 1) * stride + x + 1] = [above[0] + row[0], above[1] + row[1], above[2] + row[2]];
        }
    }
    let r0 = (k / 2) as i64;
    let mut out = RgbImage::new(img.width, img.height);
    for y in 0..h as i64 {
        let (y0, y1) = ((y - r0).max(0) as usize, (y - r0 + k as i64).min(h as i64) as usize);
        for x in 0..w as i64 {
            let (x0, x1) = ((x - r0).max(0) as usize, (x - r0 + k as i64).min(w as i64) as usize);
            let n = ((y1 - y0) * (x1 - x0)) as u32;
            let (a, b, c, d) = (sum[y0 * stride + x0], sum[y0 * stride + x1], sum[y1 * stride + x0], sum[y1 * stride + x1]);
            let mut p = [0u8; 3];
            for ch in 0..3 { p[ch] = ((d[ch] + a[ch] - b[ch] - c[ch] + n / 2) / n) as u8; }
            out.set(x as u32, y as u32, p);
        }
    }
    out
}

#[derive(Clone, Copy, Debug)]
pub struct ReadParams {
    /// Alignment search reach in camera px (0 = no search).
    pub reach_px: f32,
    pub step_px: f32,
    /// Pilot costs are pooled over the blocks within this many blocks (the shift field is smooth).
    pub pool: u32,
    /// Box-filter side as a fraction of a cell's camera size.
    pub window: f32,
    /// Fit a colour model per block from the pooled pilots where it beats the global one.
    pub local_colour: bool,
    /// Widen the search with the sampling: the measured bend is up to ~2 screen px, so the reach becomes
    /// at least 2.5 x sampling camera px and the step at least sampling / 4.
    pub auto_reach: bool,
    /// Subtract neighbour-cell crosstalk: sample' = sample + alpha * (sample - mean of the 4 edge
    /// neighbours), alpha chosen per frame from `ALPHAS` by the held-out pilot residual.
    pub equalise: bool,
    /// Alignment passes: after the first, the colour model is refitted on the pilots at their block's
    /// chosen shift and the shift search is scored again (the unshifted fit is half a cell off at 2-3 px cells).
    pub passes: u32,
    /// Low two bits of the frame sequence the header announced: the pilot colours to start from.
    pub tag_hint: u8,
    /// Detect per block which frame (tag) its pilots show and re-read with those colours: a capture that
    /// straddles two displayed frames gets both halves right and tagged (`FlatReads::block_tag`).
    pub tag_search: bool,
    /// Black/white: classify each cell against a 2-means threshold of its own block's data cells (3x3 block
    /// neighbourhood) instead of the pilot colour model; at 1-2 px cells 4 pilots per block are too few.
    pub local_threshold: bool,
    /// Largest crosstalk factor tried (0.1 steps from 0).
    pub alpha_max: f32,
    /// Black/white: decide every cell against a threshold that depends on how many of its neighbours are
    /// white (pattern-dependent decision). Pilots, whose values are known, measure per neighbour class how
    /// bright a black and a white cell look; blur in linear light lifts a black cell among whites far more
    /// than it darkens a white one, which a single threshold or a linear equaliser cannot follow.
    pub pattern_decide: bool,
}
impl Default for ReadParams {
    fn default() -> Self { ReadParams { reach_px: 3.0, step_px: 0.5, pool: 1, window: 0.5, local_colour: true, auto_reach: true, equalise: true, passes: 2, tag_hint: 0, pattern_decide: true, tag_search: true, local_threshold: false, alpha_max: 0.5 } }
}
/// `FlatReads::block_tag` value of a block whose pilots fit no frame tag.
pub const TAG_UNKNOWN: u8 = 255;
/// A block whose cells barely spread (a covered patch: a notification over the sender, a thumb; or blown out):
/// unreadable, unlike `TAG_UNKNOWN` (its cells may still read: grey frames show some on clean captures). Its
/// cells are erasures; the stitcher leaves them missing for a later capture.
pub const TAG_FLAT: u8 = 254;
/// Whether `t` is a frame tag (0..4), not `TAG_UNKNOWN` or `TAG_FLAT`.
pub fn tag_known(t: u8) -> bool { t < 4 }
impl ReadParams {
    /// The reader settings measured best for a cell pitch on real phone captures (2026-10-27, OnePlus 12 at
    /// 2.2-2.5 camera px per screen px): 1 px cells want a finer alignment step (sampling / 8), wider pooling
    /// (3 blocks) and a 0.7-cell box window -- 991 -> 1324 of 1776 codewords on four captures. Explicit
    /// overrides (auto_reach off) are kept.
    pub fn tuned_for(self, pitch: u8) -> ReadParams {
        if pitch == 1 && self.auto_reach { ReadParams { step_px: 0.0, pool: 3, window: 0.7, alpha_max: 0.7, ..self } } else { self }
    }
}

/// Candidate crosstalk factors; 0 (no equalisation) wins ties.
pub const ALPHAS: [f32; 6] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5];

pub struct FlatReads { pub cols: u32, pub rows: u32, pub values: Vec<u8>, pub margins: Vec<u16>, pub shifts: Vec<(f32, f32)>, pub alpha: f32,
    /// Per 8x8 block: the frame-sequence bits (0..4) its pilots showed; one value on a clean capture, two
    /// on a capture that straddles a display refresh.
    pub block_tag: Vec<u8>,
    /// The share of the tile (rows of blocks) where another sender frame has more than a quarter of the light:
    /// an exposure across the sender's frame change blends two frames in every row (b/w luma reads; 0 otherwise).
    pub mix: f32,
    /// The 10-90% width, in camera px, of the chrome border's edges (the same on every frame: focus and motion
    /// blur, not a blend); 0 when not measured (`decode_luma` measures it).
    pub edge_px: f32 }
impl FlatReads {
    pub fn at(&self, col: u32, row: u32) -> (u8, u16) { let i = (row * self.cols + col) as usize; (self.values[i], self.margins[i]) }
    pub fn data(&self, grid: &FlatGrid, index: u32) -> (u8, u16) { let (c, r) = grid.data_position(index); self.at(c, r) }
}

fn sampling_at(h: &Homography, x: f32, y: f32) -> f32 {
    let (ax, ay) = h.apply(x, y); let (bx, by) = h.apply(x + 1.0, y);
    (bx - ax).hypot(by - ay)
}
/// Camera px per screen px at the grid centre.
pub fn sampling(h: &Homography, grid: &FlatGrid) -> f32 { sampling_at(h, grid.width_px() as f32 / 2.0, grid.height_px() as f32 / 2.0) }

fn luma(p: [u8; 3]) -> u8 { (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32 + 0.5) as u8 }

/// One sample at frame position (x, y) shifted by (dx, dy) camera px; None outside the image.
#[inline]
fn sample(src: &RgbImage, h: &Homography, x: f32, y: f32, dx: f32, dy: f32) -> Option<[u8; 3]> {
    let (cx, cy) = h.apply(x, y);
    let (cx, cy) = (cx + dx, cy + dy);
    if cx < 0.0 || cy < 0.0 || cx >= src.width as f32 || cy >= src.height as f32 { return None; }
    Some(src.sample(cx - 0.5, cy - 0.5))
}

fn header_blur(img: &RgbImage, chrome: &Chrome, h: &Homography) -> RgbImage {
    let s = sampling_at(h, chrome.width_px() as f32 / 2.0, chrome.height_px() as f32 / 2.0);
    box_blur(img, (MODULE_PX as f32 * s * 0.5).round().max(1.0) as u32)
}
fn header_bits_from(src: &RgbImage, chrome: &Chrome, h: &Homography, copy: u32) -> Vec<bool> {
    let vals: Vec<u8> = (0..Chrome::HEADER_CELLS).map(|i| {
        let (x, y) = chrome.header_cell_center(copy, i);
        sample(src, h, x, y, 0.0, 0.0).map(luma).unwrap_or(0)
    }).collect();
    let thr = two_means_threshold(&vals);
    vals.iter().map(|&v| (v > thr) ^ (copy == 1)).collect()
}
/// The 208 header cells of `copy` as bits: box-sampled luma thresholded at its two-means level.
pub fn read_header_bits(img: &RgbImage, chrome: &Chrome, h: &Homography, copy: u32) -> Vec<bool> {
    header_bits_from(&header_blur(img, chrome, h), chrome, h, copy)
}

/// `read_header` from a luma plane: each header cell is the mean of the same box the RGB path blurs over.
pub fn read_header_luma(y: &crate::image::GrayImage, chrome: &Chrome, h: &Homography) -> [Option<FlatHeader>; 2] {
    read_header_copies_luma(y, chrome, h).map(HeaderCopy::header)
}
/// `read_header_luma`, telling a header of another version from an unreadable copy.
pub fn read_header_copies_luma(y: &crate::image::GrayImage, chrome: &Chrome, h: &Homography) -> [HeaderCopy; 2] {
    let s = sampling_at(h, chrome.width_px() as f32 / 2.0, chrome.height_px() as f32 / 2.0);
    let k = (MODULE_PX as f32 * s * 0.5).round().max(1.0) as i32;
    // each cell's box summed from the pixels: a few hundred small boxes, against an integral image of the
    // whole frame (1.5 M u32 written per Full HD capture, ~1 ms on the tracked path)
    let (w, hh) = (y.width as i32, y.height as i32);
    let box_mean = |x0: i32, y0: i32, x1: i32, y1: i32| -> f32 {
        let (x0, y0, x1, y1) = (x0.clamp(0, w), y0.clamp(0, hh), x1.clamp(0, w), y1.clamp(0, hh));
        if x1 <= x0 || y1 <= y0 { return 0.0; }
        let total: u32 = (y0..y1).map(|r| y.data[(r * w + x0) as usize..(r * w + x1) as usize].iter().map(|&v| v as u32).sum::<u32>()).sum();
        total as f32 / ((x1 - x0) * (y1 - y0)) as f32
    };
    let bits = |copy: u32| -> Vec<bool> {
        let vals: Vec<u8> = (0..Chrome::HEADER_CELLS).map(|i| {
            let (x, yy) = chrome.header_cell_center(copy, i);
            let (cx, cy) = h.apply(x, yy);
            if cx < 0.0 || cy < 0.0 || cx >= y.width as f32 || cy >= y.height as f32 { return 0; }
            let (x0, y0) = ((cx - k as f32 / 2.0).round() as i32, (cy - k as f32 / 2.0).round() as i32);
            (box_mean(x0, y0, x0 + k, y0 + k) + 0.5) as u8
        }).collect();
        let thr = two_means_threshold(&vals);
        vals.iter().map(|&v| (v > thr) ^ (copy == 1)).collect()
    };
    [FlatHeader::copy_from_bits(&bits(0)), FlatHeader::copy_from_bits(&bits(1))]
}

/// Both header copies (a capture split across two displayed frames can show two different ones); the
/// box-blurred image is shared by both reads.
pub fn read_header(img: &RgbImage, chrome: &Chrome, h: &Homography) -> [Option<FlatHeader>; 2] {
    read_header_copies(img, chrome, h).map(HeaderCopy::header)
}
/// `read_header`, telling a header of another version from an unreadable copy.
pub fn read_header_copies(img: &RgbImage, chrome: &Chrome, h: &Homography) -> [HeaderCopy; 2] {
    let src = header_blur(img, chrome, h);
    [FlatHeader::copy_from_bits(&header_bits_from(&src, chrome, h, 0)), FlatHeader::copy_from_bits(&header_bits_from(&src, chrome, h, 1))]
}

#[derive(Clone)]
pub(crate) struct Pilot { pub(crate) cx: f32, pub(crate) cy: f32, pub(crate) colour: u8, pub(crate) block: usize, pub(crate) col: u32, pub(crate) row: u32 }   // camera px, grid cell

/// Frame tag per block from each block's cost under the 4 tags (`tagged[b]` = tagged pilots it has, < 2 =
/// no say): one tag everywhere, or one straight cut (rolling shutter) between two tags, then blocks whose
/// pilots fit no tag become `TAG_UNKNOWN`.
pub(crate) fn assign_tags(cost: &[[f32; 4]], tagged: &[u8], nbr: usize, nbc: usize) -> Vec<u8> {
    let nblocks = nbr * nbc;
    let mut block_tag = vec![0u8; nblocks];
    let row_cost: Vec<[f32; 4]> = (0..nbr).map(|by| { let mut a = [0f32; 4]; for bx in 0..nbc { for t in 0..4 { a[t] += cost[by * nbc + bx][t]; } } a }).collect();
    let col_cost: Vec<[f32; 4]> = (0..nbc).map(|bx| { let mut a = [0f32; 4]; for by in 0..nbr { for t in 0..4 { a[t] += cost[by * nbc + bx][t]; } } a }).collect();
    let best_of = |sum: [f32; 4]| -> (f32, u8) { let mut best = (f32::MAX, 0u8); for t in 0..4 { if sum[t] < best.0 { best = (sum[t], t as u8); } } best };
    let total = |v: &[[f32; 4]], lo: usize, hi: usize| -> [f32; 4] { let mut a = [0f32; 4]; for x in &v[lo..hi] { for t in 0..4 { a[t] += x[t]; } } a };
    let whole = best_of(total(&row_cost, 0, nbr));
    let mut best = (whole.0, None::<(bool, usize, u8, u8)>, whole.1);
    for (vertical, v, n) in [(false, &row_cost, nbr), (true, &col_cost, nbc)] {
        for cut in 1..n {
            let (c1, t1) = best_of(total(v, 0, cut)); let (c2, t2) = best_of(total(v, cut, n));
            // a cut must buy a clear saving on the side it moves: at least a third of what that side
            // costs under the single best tag (noise alone shaves a few percent)
            let (lo, hi) = (total(v, 0, cut), total(v, cut, n));
            let moved = if t1 == whole.1 { hi[whole.1 as usize] - c2 } else { lo[whole.1 as usize] - c1 };
            let side = if t1 == whole.1 { hi[whole.1 as usize] } else { lo[whole.1 as usize] };
            if t1 != t2 && side > 0.0 && moved >= 0.33 * side && c1 + c2 < 0.98 * best.0 { best = (c1 + c2, Some((vertical, cut, t1, t2)), t1); }
        }
    }
    for b in 0..nblocks {
        block_tag[b] = match best.1 { None => best.2, Some((vertical, cut, t1, t2)) => { let pos = if vertical { b % nbc } else { b / nbc }; if pos < cut { t1 } else { t2 } } };
    }
    // blocks whose pilots fit no tag (another format under this grid, glare, a finger) are unknown
    // (TAG_UNKNOWN): the stitcher keeps them out of every frame
    let mut mins: Vec<f32> = (0..nblocks).filter(|&b| tagged[b] >= 2).map(|b| cost[b].iter().cloned().fold(f32::MAX, f32::min)).collect();
    if mins.len() >= 8 {
        mins.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = mins[mins.len() / 2];
        let mut seconds: Vec<f32> = (0..nblocks).filter(|&b| tagged[b] >= 2).map(|b| { let mut c = cost[b]; c.sort_by(|a, b| a.partial_cmp(b).unwrap()); c[1] }).collect();
        seconds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let runner_up = seconds[seconds.len() / 2];
        for b in 0..nblocks {
            if tagged[b] < 2 { continue; }
            let mut c = cost[b]; c.sort_by(|a, b| a.partial_cmp(b).unwrap());
            // a readable block fits its tag far better than the runner-up (half the alphabet apart);
            // garbage fits every tag about equally badly, and badly in absolute terms
            if c[0] > 2.0 * median && c[0] >= 0.6 * c[1] && c[0] > 0.25 * runner_up { block_tag[b] = TAG_UNKNOWN; }
        }
    }
    block_tag
}

/// Read every data cell of `grid` from `img` through `h` (frame px -> image px).
pub fn read_cells(img: &RgbImage, grid: &FlatGrid, h: &Homography, params: &ReadParams) -> FlatReads {
    let t_start = std::env::var_os("LB_PROFILE").map(|_| std::time::Instant::now());
    let mark = |what: &str| { if let Some(t) = t_start { eprintln!("  [read_cells] {:>7.1} ms  {}", t.elapsed().as_secs_f64() * 1000.0, what); } };
    let (cols, rows) = (grid.cols(), grid.rows());
    let fmt = grid.format();
    let pal = fmt.alphabet.palette();
    let s = sampling(h, grid);
    let k = (fmt.pitch as f32 * s * params.window).round().max(1.0) as u32;
    let blurred;
    let src: &RgbImage = if k > 1 { blurred = box_blur(img, k); &blurred } else { img };
    let (bw, bh) = grid.blocks();
    let nblocks = (bw * bh) as usize;
    let mut pilots: Vec<Pilot> = grid.pilot_positions().iter().filter_map(|&(c, r)| {
        let FlatCell::Pilot { .. } = grid.cell(c, r) else { return None };
        let colour = grid.pilot_colour_at(c, r, params.tag_hint & 3);
        let (x, y) = grid.cell_center(c, r);
        let (cx, cy) = h.apply(x, y);
        Some(Pilot { cx, cy, colour, block: ((r / BLOCK) * bw + c / BLOCK) as usize, col: c, row: r })
    }).collect();
    // a pilot sample at a camera-px shift, projected once
    let sample_pilot = |p: &Pilot, dx: f32, dy: f32| -> Option<[u8; 3]> {
        let (cx, cy) = (p.cx + dx, p.cy + dy);
        if cx < 0.0 || cy < 0.0 || cx >= src.width as f32 || cy >= src.height as f32 { return None; }
        Some(src.sample(cx - 0.5, cy - 0.5))
    };
    let mut by_block: Vec<Vec<usize>> = vec![Vec::new(); nblocks];
    for (i, p) in pilots.iter().enumerate() { by_block[p.block].push(i); }
    // candidate shifts in camera px
    // auto reach: at least 2.5 x sampling; auto step: sampling / 4, or sampling / 8 when step_px is 0 (the 1 px tuning)
    let (reach, step) = if params.auto_reach && params.reach_px > 0.0 { (params.reach_px.max(2.5 * s), if params.step_px <= 0.0 { s / 8.0 } else { params.step_px.max(s / 4.0) }) } else { (params.reach_px, params.step_px) };
    // coarse-to-fine: the first pass searches the whole reach at a coarse step (at least sampling / 4); later
    // passes search the fine step only within a window of one coarse step around each block's previous shift
    let coarse_step = if step > 0.0 { step.max(s / 4.0) } else { step };
    let shift_grid = |st: f32| -> Vec<(f32, f32)> {
        if reach <= 0.0 || st <= 0.0 { return vec![(0.0, 0.0)]; }
        let n = (reach / st).round() as i32;
        (-n..=n).flat_map(|j| (-n..=n).map(move |i| (i as f32 * st, j as f32 * st))).collect()
    };
    let shifts_coarse = shift_grid(coarse_step);
    let shifts_fine = if step > 0.0 && step < coarse_step - 1e-6 { shift_grid(step) } else { shifts_coarse.clone() };
    let fine_window = coarse_step + step;
    // global colour model from every pilot at the unshifted position
    let fit = |samples: &[([u8; 3], [u8; 3])]| if fmt.alphabet.is_luma() { ColourModel::fit_luma(samples) } else { ColourModel::fit(samples) };
    let global_samples: Vec<([u8; 3], [u8; 3])> = pilots.iter().filter_map(|p| sample_pilot(p, 0.0, 0.0).map(|v| (v, pal[p.colour as usize]))).collect();
    let pool = params.pool as i64;
    let neighbours = |b: usize, reach: i64| -> Vec<usize> {
        let (bx, by) = ((b as u32 % bw) as i64, (b as u32 / bw) as i64);
        let mut v = Vec::new();
        for y in (by - reach).max(0)..=(by + reach).min(bh as i64 - 1) { for x in (bx - reach).max(0)..=(bx + reach).min(bw as i64 - 1) { v.push((y * bw as i64 + x) as usize); } }
        v
    };
    mark("blur + pilots + global fit");
    let mut global = fit(&global_samples).unwrap_or_else(ColourModel::identity);
    // one alignment pass: pilot costs per block and shift, pooled, settled, holes filled from neighbours
    let search = |pilots: &[Pilot], by_block: &[Vec<usize>], global: &ColourModel, shifts: &[(f32, f32)], window: Option<(&[(f32, f32)], f32)>| -> Vec<(f32, f32)> {
            // the shifts form a square grid (index j*m + i); a windowed pass only ever chooses, for a block,
            // a shift inside its own window, so costs are computed and pooled there only (a neighbour's cost
            // outside its window is the same 1e9 the dense search used)
            let m = (shifts.len() as f64).sqrt().round() as usize;
            debug_assert_eq!(m * m, shifts.len());
            let rect = |b: usize| -> (usize, usize, usize, usize) {
                let Some((centres, r)) = window else { return (0, m, 0, m) };
                let (cx, cy) = centres[b];
                let inside = |v: f32, c: f32| (v - c).abs() <= r + 1e-4;
                let is: Vec<usize> = (0..m).filter(|&i| inside(shifts[i].0, cx)).collect();
                let js: Vec<usize> = (0..m).filter(|&j| inside(shifts[j * m].1, cy)).collect();
                match (is.first(), is.last(), js.first(), js.last()) { (Some(&i0), Some(&i1), Some(&j0), Some(&j1)) => (i0, i1 + 1, j0, j1 + 1), _ => (0, 0, 0, 0) }
            };
            // pilot cost per block and shift of its window: (rect i0, i1, j0, j1 half-open, costs row-major)
            let costs: Vec<((usize, usize, usize, usize), Vec<f32>)> = par_map(nblocks, |b| {
                let rc = rect(b);
                let mut v = Vec::with_capacity((rc.1 - rc.0) * (rc.3 - rc.2));
                for j in rc.2..rc.3 { for i in rc.0..rc.1 {
                    let (dx, dy) = shifts[j * m + i];
                    v.push(by_block[b].iter().filter_map(|&k| {
                        let p = &pilots[k];
                        let s = sample_pilot(p, dx, dy)?;
                        let f = global.apply(s); let t = pal[p.colour as usize];
                        Some((0..3).map(|c| (f[c] - t[c] as f32).powi(2)).sum::<f32>())
                    }).sum());
                } }
                (rc, v)
            });
            let cost_at = |nb: usize, i: usize, j: usize| -> f32 {
                let ((i0, i1, j0, j1), ref v) = costs[nb];
                if i >= i0 && i < i1 && j >= j0 && j < j1 { v[(j - j0) * (i1 - i0) + (i - i0)] } else { 1e9 }
            };
            // shifts ordered by magnitude: among shifts of (nearly) equal cost the smallest wins, so a clean
            // frame whose pilots read the same anywhere inside their cells keeps a zero shift
            let mut order: Vec<usize> = (0..shifts.len()).collect();
            order.sort_by(|&a, &b| (shifts[a].0.hypot(shifts[a].1)).partial_cmp(&shifts[b].0.hypot(shifts[b].1)).unwrap());
            let mut rank = vec![0usize; shifts.len()];
            for (r, &i) in order.iter().enumerate() { rank[i] = r; }
            let chosen: Vec<Option<(f32, f32)>> = par_map(nblocks, |b| {
                let (i0, i1, j0, j1) = costs[b].0;
                let own = (i1 - i0) * (j1 - j0);
                if own == 0 { return None; }
                let nbs = neighbours(b, pool);
                let mut pooled;
                if window.is_none() {
                    // dense pass: every block's costs cover the whole grid in the same order
                    pooled = vec![0f32; own];
                    for &nb in &nbs { for (acc, c) in pooled.iter_mut().zip(&costs[nb].1) { *acc += c; } }
                } else {
                    pooled = Vec::with_capacity(own);
                    for j in j0..j1 { for i in i0..i1 { pooled.push(nbs.iter().map(|&nb| cost_at(nb, i, j)).sum::<f32>()); } }
                }
                let (mut lo, mut hi) = (f32::MAX, 0f32);
                for &c in &pooled { lo = lo.min(c); hi = hi.max(c); }
                // a block whose pilots see nothing (a finger, glare, no pilots at all) has a flat cost: it
                // inherits its neighbours' shift below instead of a random argmin. The median is over every shift:
                // when the window holds fewer than half of them the rest pool to >= 1e9, so the median does too
                let settled = if own * 2 <= shifts.len() { hi > 0.0 && lo < 0.7e9 } else {
                    let mut all: Vec<f32> = if own == shifts.len() { pooled.clone() } else { (0..shifts.len()).map(|k| nbs.iter().map(|&nb| cost_at(nb, k % m, k / m)).sum::<f32>()).collect() };
                    hi = all.iter().cloned().fold(hi, f32::max);
                    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    hi > 0.0 && lo < 0.7 * all[all.len() / 2]
                };
                if !settled { return None; }
                let tol = 0.01 * lo + 1.0;
                let mut best: Option<(usize, usize)> = None;
                for (k, &c) in pooled.iter().enumerate() {
                    if c > lo + tol { continue; }
                    let idx = (j0 + k / (i1 - i0)) * m + i0 + k % (i1 - i0);
                    if best.is_none_or(|(r, _)| rank[idx] < r) { best = Some((rank[idx], idx)); }
                }
                best.map(|(_, idx)| shifts[idx])
            });
            let mut block_shift_p: Vec<(f32, f32)> = chosen.iter().map(|c| c.unwrap_or((0.0, 0.0))).collect();
            let mut settled: Vec<bool> = chosen.iter().map(|c| c.is_some()).collect();
            for _ in 0..(bw + bh) as usize {
                let todo: Vec<usize> = (0..nblocks).filter(|&b| !settled[b]).collect();
                if todo.is_empty() { break; }
                let mut fills = Vec::new();
                for b in todo {
                    let (bx, by) = (b as u32 % bw, b as u32 / bw);
                    let mut acc = (0f32, 0f32, 0f32);
                    let mut push = |x: u32, y: u32| { let n = (y * bw + x) as usize; if settled[n] { acc.0 += block_shift_p[n].0; acc.1 += block_shift_p[n].1; acc.2 += 1.0; } };
                    if bx > 0 { push(bx - 1, by); } if bx + 1 < bw { push(bx + 1, by); } if by > 0 { push(bx, by - 1); } if by + 1 < bh { push(bx, by + 1); }
                    if acc.2 > 0.0 { fills.push((b, (acc.0 / acc.2, acc.1 / acc.2))); }
                }
                if fills.is_empty() { break; }
                for (b, sh) in fills { block_shift_p[b] = sh; settled[b] = true; }
            }
        block_shift_p
    };
    let refit = |pilots: &[Pilot], block_shift: &[(f32, f32)]| -> Option<ColourModel> {
        let aligned: Vec<([u8; 3], [u8; 3])> = pilots.iter().filter_map(|p| { let (dx, dy) = block_shift[p.block]; sample_pilot(p, dx, dy).map(|v| (v, pal[p.colour as usize])) }).collect();
        fit(&aligned)
    };
    // the coarse pass is anchored on the untagged pilots (0 and 2) only: in a capture straddling two frames
    // whose tags differ in both bits, the tagged pilots of the other frame carry the wrong colour, and at 1 px
    // b/w that leaves the right shift no better than a random one (measured: the other frame's half went
    // untagged and its cells were stitched into the wrong frame)
    let kind_of = |p: &Pilot| PILOT_LOCAL.iter().position(|&q| q == (p.row % BLOCK, p.col % BLOCK)).unwrap_or(0);
    let anchor: Vec<Pilot> = if params.tag_search { pilots.iter().filter(|p| kind_of(p) % 2 == 0).cloned().collect() } else { pilots.clone() };
    let mut by_block_anchor: Vec<Vec<usize>> = vec![Vec::new(); nblocks];
    for (i, p) in anchor.iter().enumerate() { by_block_anchor[p.block].push(i); }
    let mut block_shift = search(&anchor, &by_block_anchor, &global, &shifts_coarse, None);
    mark("alignment pass 1");
    // the fine passes too: a wrong-coloured pilot pulls its block toward a neighbouring cell of the expected
    // colour even inside the fine window; every pilot joins once the tags are known
    for _ in 1..params.passes.max(1) {
        if let Some(m) = refit(&anchor, &block_shift) { global = m; }
        let prev = block_shift.clone();
        block_shift = search(&anchor, &by_block_anchor, &global, &shifts_fine, Some((&prev, fine_window)));
    }
    mark("alignment passes done");
    // frame tags: for every block the cost of each of the 4 tags on its two tagged pilots (1 and 3), then
    // the cheapest of: one tag everywhere, a horizontal split, a vertical split (a rolling shutter cuts
    // the tile along one straight line)
    let (nbr, nbc) = (bh as usize, bw as usize);
    let mut block_tag = vec![params.tag_hint & 3; nblocks];
    if params.tag_search {
        let mut cost = vec![[0f32; 4]; nblocks];
        let mut tagged = vec![0u8; nblocks];
        let kind = |p: &Pilot| PILOT_LOCAL.iter().position(|&q| q == (p.row % BLOCK, p.col % BLOCK)).unwrap_or(0);
        // the colour model for the tag test comes from the untagged pilots (0 and 2) only: a wrong hint
        // mislabels the tagged ones and would poison a fit that includes them
        let untagged: Vec<([u8; 3], [u8; 3])> = pilots.iter().filter(|p| kind(p) % 2 == 0).filter_map(|p| { let (dx, dy) = block_shift[p.block]; sample_pilot(p, dx, dy).map(|v| (v, pal[p.colour as usize])) }).collect();
        let tag_model = fit(&untagged).unwrap_or(global);
        for p in &pilots {
            let kk = kind(p);
            if kk != 1 && kk != 3 { continue; }
            let (dx, dy) = block_shift[p.block];
            let Some(v) = sample_pilot(p, dx, dy) else { continue };
            let f = tag_model.apply(v);
            for t in 0..4u8 {
                let c = pal[grid.pilot_colour_at(p.col, p.row, t) as usize];
                cost[p.block][t as usize] += (0..3).map(|k| (f[k] - c[k] as f32).powi(2)).sum::<f32>();
            }
            tagged[p.block] += 1;
        }
        // a block missing one of its two tagged pilots (partial block rows/cols at the tile edge) has no say:
        // on real captures one noisy pilot per block was enough to invent a cut along the last block row
        for b in 0..nblocks { if tagged[b] < 2 { cost[b] = [0.0; 4]; } }
        block_tag = assign_tags(&cost, &tagged, nbr, nbc);
        for p in pilots.iter_mut() { let t = block_tag[p.block]; if t != TAG_UNKNOWN { p.colour = grid.pilot_colour_at(p.col, p.row, t); } }
        if let Some(m) = refit(&pilots, &block_shift) { global = m; }
        let prev = block_shift.clone();
        block_shift = search(&pilots, &by_block, &global, &shifts_fine, Some((&prev, fine_window)));
    }
    let pilots = pilots;
    mark("frame tags");
    // crosstalk equaliser: every pilot's own sample and the mean of its 4 edge neighbours at the block's
    // shift; alpha is chosen on the odd pilots after fitting the global model on the even ones
    let mean4 = |c: u32, r: u32, dx: f32, dy: f32| -> Option<[f32; 3]> {
        let mut acc = [0f32; 3]; let mut n = 0f32;
        let mut push = |cc: u32, rr: u32| { let (x, y) = grid.cell_center(cc, rr); if let Some(v) = sample(src, h, x, y, dx, dy) { for k in 0..3 { acc[k] += v[k] as f32; } n += 1.0; } };
        if c > 0 { push(c - 1, r); } if c + 1 < cols { push(c + 1, r); } if r > 0 { push(c, r - 1); } if r + 1 < rows { push(c, r + 1); }
        if n == 0.0 { None } else { Some([acc[0] / n, acc[1] / n, acc[2] / n]) }
    };
    let equalise = |own: [u8; 3], nb: Option<[f32; 3]>, alpha: f32| -> [u8; 3] {
        match nb { Some(m) if alpha != 0.0 => { let mut o = [0u8; 3]; for k in 0..3 { o[k] = (own[k] as f32 + alpha * (own[k] as f32 - m[k])).round().clamp(0.0, 255.0) as u8; } o } _ => own }
    };
    let pilot_raw: Vec<Option<([u8; 3], Option<[f32; 3]>)>> = pilots.iter().map(|p| {
        let (dx, dy) = block_shift[p.block];
        sample_pilot(p, dx, dy).map(|v| (v, mean4(p.col, p.row, dx, dy)))
    }).collect();
    // colour models for a given alpha: global on every pilot, per block fitted on the neighbouring
    // blocks' pilots and judged on the block's own (held out: a model judged on its own training pilots
    // would always win); returns the models and the held-out residual summed over the frame
    let models_for = |alpha: f32| -> (Vec<ColourModel>, f32) {
        let eq: Vec<Option<([u8; 3], [u8; 3])>> = pilots.iter().enumerate().map(|(i, p)| pilot_raw[i].map(|(v, nb)| (equalise(v, nb, alpha), pal[p.colour as usize]))).collect();
        let global = fit(&eq.iter().flatten().copied().collect::<Vec<_>>()).unwrap_or(global);
        let per_block: Vec<(ColourModel, f32)> = par_map(nblocks, |b| {
            let take = |nb: usize| by_block[nb].iter().filter_map(|&i| eq[i]).collect::<Vec<([u8; 3], [u8; 3])>>();
            let own = take(b);
            if own.len() < 2 { return (global, 0.0); }
            let g_res = global.residual(&own);
            if !params.local_colour { return (global, g_res * own.len() as f32); }
            let others: Vec<([u8; 3], [u8; 3])> = neighbours(b, pool.max(1)).into_iter().filter(|&nb| nb != b).flat_map(take).collect();
            match fit(&others) { Some(m) => { let r = m.residual(&own); if r < g_res { (m, r * own.len() as f32) } else { (global, g_res * own.len() as f32) } } None => (global, g_res * own.len() as f32) }
        });
        let total = per_block.iter().map(|x| x.1).sum();
        (per_block.into_iter().map(|x| x.0).collect(), total)
    };
    mark("equaliser samples");
    let (alpha, models) = if params.equalise {
        let n_alpha = ((params.alpha_max / 0.1).round() as usize).clamp(0, 12);
        let alphas: Vec<f32> = (0..=n_alpha).map(|i| i as f32 * 0.1).collect();
        let scored: Vec<(f32, Vec<ColourModel>, f32)> = alphas.iter().map(|&a| { let (m, r) = models_for(a); (a, m, r) }).collect();
        let best = scored.iter().map(|s| s.2).fold(f32::MAX, f32::min);
        // the smallest alpha within 2% of the best residual: a clean frame stays at 0
        let pick = scored.into_iter().find(|s| s.2 <= best * 1.02 + 1e-3).unwrap();
        (pick.0, pick.1)
    } else { let (m, _) = models_for(0.0); (0.0, m) };
    // every used data cell at its interpolated shift
    let cell_shift = |c: u32, r: u32| -> (f32, f32) {
        let u = ((c as f32 + 0.5) / BLOCK as f32 - 0.5).clamp(0.0, (bw - 1) as f32);
        let v = ((r as f32 + 0.5) / BLOCK as f32 - 0.5).clamp(0.0, (bh - 1) as f32);
        let (i, j) = ((u.floor() as u32).min(bw - 1), (v.floor() as u32).min(bh - 1));
        let (i1, j1) = ((i + 1).min(bw - 1), (j + 1).min(bh - 1));
        let (fu, fv) = (u - i as f32, v - j as f32);
        let at = |x: u32, y: u32| block_shift[(y * bw + x) as usize];
        let (a, b, c2, d) = (at(i, j), at(i1, j), at(i, j1), at(i1, j1));
        let top = (a.0 + (b.0 - a.0) * fu, a.1 + (b.1 - a.1) * fu);
        let bot = (c2.0 + (d.0 - c2.0) * fu, c2.1 + (d.1 - c2.1) * fu);
        (top.0 + (bot.0 - top.0) * fv, top.1 + (bot.1 - top.1) * fv)
    };
    mark("alpha + colour models");
    let used = grid.used_data_cells();
    let mut values = vec![0u8; (cols * rows) as usize];
    let mut margins = vec![0u16; (cols * rows) as usize];
    if params.local_threshold && pal.len() == 2 {
        // black/white: sample every cell's luma first, then threshold each block against the 2-means split of
        // the data cells in its 3x3 block neighbourhood (pilot-free, follows brightness gradients and blur)
        let lum: Vec<Vec<(u32, u32, f32)>> = par_map(used.div_ceil(4096), |chunk| {
            (chunk * 4096..((chunk + 1) * 4096).min(used)).filter_map(|i| {
                let (c, r) = grid.data_position(i as u32);
                let (x, y) = grid.cell_center(c, r);
                let (dx, dy) = cell_shift(c, r);
                let v = sample(src, h, x, y, dx, dy)?;
                let v = if alpha != 0.0 { equalise(v, mean4(c, r, dx, dy), alpha) } else { v };
                Some((c, r, luma(v) as f32))
            }).collect()
        });
        let mut lum_map = vec![f32::NAN; (cols * rows) as usize];
        for &(c, r, l) in lum.iter().flatten() { lum_map[(r * cols + c) as usize] = l; }
        let (nbr, nbc) = (bh as usize, bw as usize);
        let split: Vec<(f32, f32, f32)> = par_map(nblocks, |b| {
            let (bx, by) = ((b % nbc) as i64, (b / nbc) as i64);
            let mut vals: Vec<f32> = Vec::with_capacity(9 * 64);
            for y in (by - 1).max(0)..=(by + 1).min(nbr as i64 - 1) { for x in (bx - 1).max(0)..=(bx + 1).min(nbc as i64 - 1) {
                for r in (y as u32 * BLOCK)..((y as u32 + 1) * BLOCK).min(rows) { for c in (x as u32 * BLOCK)..((x as u32 + 1) * BLOCK).min(cols) {
                    let l = lum_map[(r * cols + c) as usize]; if !l.is_nan() { vals.push(l); }
                } }
            } }
            if vals.len() < 8 { return (f32::NAN, 0.0, 255.0); }
            let mut thr = vals.iter().sum::<f32>() / vals.len() as f32;
            let (mut lo, mut hi) = (0.0f32, 255.0f32);
            for _ in 0..6 {
                let (mut sl, mut nl, mut sh, mut nh) = (0.0f32, 0usize, 0.0f32, 0usize);
                for &v in &vals { if v <= thr { sl += v; nl += 1; } else { sh += v; nh += 1; } }
                if nl == 0 || nh == 0 { break; }
                lo = sl / nl as f32; hi = sh / nh as f32; thr = (lo + hi) / 2.0;
            }
            (thr, lo, hi)
        });
        let dark = if luma(pal[0]) <= luma(pal[1]) { 0u8 } else { 1u8 };
        for &(c, r, l) in lum.iter().flatten() {
            let b = ((r / BLOCK) * bw + c / BLOCK) as usize;
            let (thr, lo, hi) = split[b];
            let (thr, lo, hi) = if thr.is_nan() { (127.5, 0.0, 255.0) } else { (thr, lo, hi) };
            let i = (r * cols + c) as usize;
            values[i] = if l > thr { 1 - dark } else { dark };
            margins[i] = (((hi - lo) * (2.0 * l - lo - hi).abs()) / 4.0).clamp(0.0, 65535.0) as u16;
        }
    } else {
        // position of a fitted sample along the palette's black->white axis (0 = pal[0], 1 = pal[1]); b/w only
        let axis = |f: [f32; 3]| -> f32 {
            if pal.len() != 2 { return 0.0; }
            let (p0, p1) = (pal[0], pal[1]);
            let d: [f32; 3] = [p1[0] as f32 - p0[0] as f32, p1[1] as f32 - p0[1] as f32, p1[2] as f32 - p0[2] as f32];
            let l2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if l2 <= 0.0 { return 0.0; }
            ((f[0] - p0[0] as f32) * d[0] + (f[1] - p0[1] as f32) * d[1] + (f[2] - p0[2] as f32) * d[2]) / l2
        };
        let results: Vec<Vec<(u32, u32, u8, u16, f32)>> = par_map(used.div_ceil(4096), |chunk| {
            (chunk * 4096..((chunk + 1) * 4096).min(used)).map(|i| {
                let (c, r) = grid.data_position(i as u32);
                let (x, y) = grid.cell_center(c, r);
                let (dx, dy) = cell_shift(c, r);
                match sample(src, h, x, y, dx, dy) {
                    Some(v) => {
                        let v = if alpha != 0.0 { equalise(v, mean4(c, r, dx, dy), alpha) } else { v };
                        let m = &models[((r / BLOCK) * bw + c / BLOCK) as usize];
                        let (idx, mg) = m.classify(v, pal); (c, r, idx, mg, axis(m.apply(v)))
                    }
                    None => (c, r, 0, 0, f32::NAN),
                }
            }).collect()
        });
        let results: Vec<(u32, u32, u8, u16, f32)> = results.into_iter().flatten().collect();
        for &(c, r, v, m, _) in &results { let i = (r * cols + c) as usize; values[i] = v; margins[i] = m; }
        if pal.len() == 2 && params.pattern_decide {
            pattern_decide(grid, pal, &pilots, &results, &mut values, &mut margins, |p: &Pilot| {
                let (c, r) = (p.col, p.row);
                let (x, y) = grid.cell_center(c, r);
                let (dx, dy) = cell_shift(c, r);
                let v = sample(src, h, x, y, dx, dy)?;
                let v = if alpha != 0.0 { equalise(v, mean4(c, r, dx, dy), alpha) } else { v };
                Some(axis(models[p.block].apply(v)))
            });
        }
    }
    mark("cells classified");
    FlatReads { cols, rows, values, margins, shifts: block_shift, alpha, block_tag, mix: 0.0, edge_px: 0.0 }
}

/// Pattern-dependent black/white decision (see `ReadParams::pattern_decide`). `cells` are the data cells with
/// their first decision and their position `x` on the black->white axis; `pilot_x` gives a pilot's `x`.
/// Neighbour class of a cell = (white 4-neighbours) * 5 + (white diagonal neighbours), from the current
/// decisions (pilots: their known colour). Per class the pilots give the mean `x` of black and of white
/// cells; the threshold is their midpoint (a class with too few pilots falls back to its 4-neighbour count,
/// then to 0.5). Two rounds: the second sees the first round's corrected neighbours.
pub(crate) fn pattern_decide(grid: &FlatGrid, pal: &[[u8; 3]], pilots: &[Pilot], cells: &[(u32, u32, u8, u16, f32)], values: &mut [u8], margins: &mut [u16],
                  pilot_x: impl Fn(&Pilot) -> Option<f32> + Sync) {
    let (cols, rows) = (grid.cols() as usize, grid.rows() as usize);
    let white = white_of(pal);
    let (mut first, mut xg) = (vec![0u8; cols * rows], vec![f32::NAN; cols * rows]);
    for &(c, r, v, _, x) in cells {
        let i = r as usize * cols + c as usize;
        first[i] = (v == white) as u8;
        xg[i] = x;
    }
    pattern_decide_grid(grid, pal, pilots, &first, &xg, values, margins, pilot_x);
}

/// The palette level `pattern_decide` counts as white: the brighter of the two.
pub(crate) fn white_of(pal: &[[u8; 3]]) -> u8 {
    if pal[1].iter().map(|&v| v as u32).sum::<u32>() >= pal[0].iter().map(|&v| v as u32).sum::<u32>() { 1 } else { 0 }
}

/// `pattern_decide` on the grid: per grid cell (row-major, cols x rows) `first` = 1 where a data cell's first
/// decision is the white level (`white_of`), 0 elsewhere; `xg` = the data cell's `x` (NaN: unread, or not a
/// data cell: nothing to decide). Every cell with an `x` gets its value and margin; the others are left.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pattern_decide_grid(grid: &FlatGrid, pal: &[[u8; 3]], pilots: &[Pilot], first: &[u8], xg: &[f32], values: &mut [u8], margins: &mut [u16],
                  pilot_x: impl Fn(&Pilot) -> Option<f32> + Sync) {
    let (cols, rows) = (grid.cols() as usize, grid.rows() as usize);
    let t0 = std::env::var_os("LB_PROFILE").map(|_| std::time::Instant::now());
    let px: Vec<Option<f32>> = par_map(pilots.len(), |i| pilot_x(&pilots[i]));
    if let Some(t0) = t0 { eprintln!("    pattern: pilots {:.1} ms", t0.elapsed().as_secs_f64() * 1e3); }
    let white = white_of(pal);
    let l = { let d: Vec<f32> = (0..3).map(|k| pal[1][k] as f32 - pal[0][k] as f32).collect(); d.iter().map(|v| v * v).sum::<f32>() };
    // white map of the current decisions with a one-cell border (outside, cells that are neither pilots nor
    // data cells, count as not white): data cells their first decision, pilots their colour
    let pw = cols + 2;
    let mut wm = vec![0u8; pw * (rows + 2)];
    for (r, f) in first.chunks_exact(cols).enumerate() { wm[(r + 1) * pw + 1..][..cols].copy_from_slice(f); }
    for p in pilots { wm[(p.row as usize + 1) * pw + p.col as usize + 1] = (p.colour == white) as u8; }
    let mut cls = vec![0u8; cols * rows];
    for round in 0..2 {
        // every cell's class from the white map: (white 4-neighbours) * 5 + white diagonal neighbours, a row at
        // a time over the map's rows above, at and below
        for (r, out) in cls.chunks_exact_mut(cols).enumerate() {
            let (up, mid, dn) = (&wm[r * pw..][..pw], &wm[(r + 1) * pw..][..pw], &wm[(r + 2) * pw..][..pw]);
            for (c, o) in out.iter_mut().enumerate() {
                let n4 = mid[c] + mid[c + 2] + up[c + 1] + dn[c + 1];
                let nd = up[c] + up[c + 2] + dn[c] + dn[c + 2];
                *o = n4 * 5 + nd;
            }
        }
        // per class: sum and count of x for pilots of value 0 and 1
        let mut fine = vec![[(0f64, 0u32); 2]; 25];
        let mut coarse = vec![[(0f64, 0u32); 2]; 5];
        for (i, p) in pilots.iter().enumerate() {
            let Some(x) = px[i] else { continue };
            if x.is_nan() || p.colour > 1 { continue; }
            let k = cls[p.row as usize * cols + p.col as usize] as usize;
            let v = p.colour as usize;
            fine[k][v].0 += x as f64; fine[k][v].1 += 1;
            coarse[k / 5][v].0 += x as f64; coarse[k / 5][v].1 += 1;
        }
        let mid = |a: [(f64, u32); 2]| -> Option<f32> { if a[0].1 >= 30 && a[1].1 >= 30 { Some(((a[0].0 / a[0].1 as f64 + a[1].0 / a[1].1 as f64) / 2.0) as f32) } else { None } };
        let thr: [f32; 25] = std::array::from_fn(|k| mid(fine[k]).or_else(|| mid(coarse[k / 5])).unwrap_or(0.5));
        // every data cell with an x decided again against its class's threshold (each from the classes of
        // this round's map, so the order does not matter), in bands of rows
        let (xg, cls) = (&xg, &cls);
        par_rows_mut(values, margins, cols, |rr, vb, mb| {
            let base = rr.start * cols;
            for (j, (v, m)) in vb.iter_mut().zip(mb.iter_mut()).enumerate() {
                let x = xg[base + j];
                if x.is_nan() { continue; }
                let t = thr[cls[base + j] as usize];
                (*v, *m) = ((x > t) as u8, (l * (x - t).abs() / 2.0).clamp(0.0, 65535.0) as u16);
            }
        });
        if round == 0 {
            // the second round's map sees the corrected decisions
            for r in 0..rows {
                let (vr, xr, wr) = (&values[r * cols..][..cols], &xg[r * cols..][..cols], &mut wm[(r + 1) * pw + 1..][..cols]);
                for ((w, &v), &x) in wr.iter_mut().zip(vr).zip(xr) { if !x.is_nan() { *w = (v == white) as u8; } }
            }
        }
        if let Some(t0) = t0 { eprintln!("    pattern: round {round} done {:.1} ms", t0.elapsed().as_secs_f64() * 1e3); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::Homography;
    use crate::flat::{compose, render, Alphabet, CellFormat, FlatCell, FlatGrid, FlatHeader};
    use crate::header::FrameType;
    use crate::image::RgbImage;
    use crate::testutil::bend;
    use std::sync::Arc;

    fn frame(fmt: CellFormat) -> (Arc<FlatGrid>, FlatHeader, crate::flat::FlatFrame, RgbImage) {
        let g = Arc::new(FlatGrid::try_new(1024, 1024, fmt).unwrap());
        let h = FlatHeader { format: fmt, cols: g.cols() as u16, rows: g.rows() as u16, frame_type: FrameType::Data, transfer_id: 7, segment: 1, block_id: 99, frame_seq: 0 };
        let payload: Vec<u8> = (0..g.payload_bytes()).map(|i| ((i as u64 * 2654435761) >> 7) as u8).collect();
        let f = compose(g.clone(), h, &payload);
        let im = render(&f);
        (g, h, f, im)
    }
    fn wrong(g: &FlatGrid, f: &crate::flat::FlatFrame, reads: &FlatReads) -> (usize, usize) {
        let n = g.used_data_cells();
        let bad = (0..n as u32).filter(|&i| { let (c, r) = g.data_position(i); reads.at(c, r).0 != f.value(c, r) }).count();
        (bad, n)
    }
    #[test]
    fn clean_frame_reads_every_cell() {
        for fmt in [CellFormat { pitch: 4, alphabet: Alphabet::C8 }, CellFormat { pitch: 3, alphabet: Alphabet::C16 }, CellFormat { pitch: 5, alphabet: Alphabet::Bw }] {
            let (g, _, f, im) = frame(fmt);
            let reads = read_cells(&im, &g, &Homography::identity(), &ReadParams::default());
            let (bad, n) = wrong(&g, &f, &reads);
            assert_eq!(bad, 0, "{fmt:?}: {bad} of {n} wrong");
            assert!((0..n as u32).all(|i| { let (c, r) = g.data_position(i); reads.at(c, r).1 > 0 }));
            assert!((sampling(&Homography::identity(), &g) - 1.0).abs() < 1e-3);
        }
    }
    #[test]
    fn bent_frame_reads_through_alignment() {
        let (g, _, f, im) = frame(CellFormat { pitch: 4, alphabet: Alphabet::C8 });
        let bent = bend(&im, 2.5, 1);
        let aligned = read_cells(&bent, &g, &Homography::identity(), &ReadParams::default());
        let (bad, n) = wrong(&g, &f, &aligned);
        assert!(bad * 1000 < n * 5, "aligned: {bad} of {n} wrong");
        let fixed = read_cells(&bent, &g, &Homography::identity(), &ReadParams { reach_px: 0.0, ..ReadParams::default() });
        let (bad0, _) = wrong(&g, &f, &fixed);
        assert!(bad0 * 100 > n * 5, "without alignment only {bad0} of {n} wrong");
    }
    #[test]
    fn alignment_reach_scales_with_sampling() {
        // a close phone: 3 camera px per screen px, and a bend of 2.3 screen px = 7 camera px (more than half a cell)
        let (g, _, f, im) = frame(CellFormat { pitch: 4, alphabet: Alphabet::C8 });
        let f2c = Homography { m: [3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0] };
        let cam = bend(&im.warp(&f2c.inverse().unwrap(), 3072, 3072), 7.0, 9);
        let auto = read_cells(&cam, &g, &f2c, &ReadParams::default());
        let (bad, n) = wrong(&g, &f, &auto);
        assert!(bad * 1000 < n * 5, "auto reach: {bad} of {n} wrong");
        let fixed = read_cells(&cam, &g, &f2c, &ReadParams { auto_reach: false, ..ReadParams::default() });
        let (bad0, _) = wrong(&g, &f, &fixed);
        assert!(bad0 * 100 > n && bad0 > 4 * bad.max(1), "with a 3 px reach only {bad0} of {n} wrong (auto {bad})");
    }
    #[test]
    fn blocks_without_pilots_inherit_the_neighbourhood_shift() {
        let (g, _, f, im) = frame(CellFormat { pitch: 4, alphabet: Alphabet::C8 });
        let mut bent = bend(&im, 2.0, 3);
        bent.fill_rect(452, 452, 120, 120, [0, 0, 0]);
        let reads = read_cells(&bent, &g, &Homography::identity(), &ReadParams::default());
        let n = g.used_data_cells();
        let outside = (0..n as u32).filter(|&i| { let (c, r) = g.data_position(i); let (x, y) = g.cell_center(c, r); !(x > 448.0 && x < 576.0 && y > 448.0 && y < 576.0) });
        let (mut bad, mut tot) = (0, 0);
        for i in outside { let (c, r) = g.data_position(i); tot += 1; if reads.at(c, r).0 != f.value(c, r) { bad += 1; } }
        assert!(bad * 100 < tot, "{bad} of {tot} cells outside the blob wrong");
        // blocks wholly under the blob (cells x,y in 452..572 -> cols 111..141, rows 107..137 -> blocks 14..17 x 14..16)
        let (bw, _) = g.blocks();
        for by in 14..=16u32 { for bx in 14..=17u32 {
            let s = reads.shifts[(by * bw + bx) as usize];
            let nb: Vec<(f32, f32)> = [(bx - 1, by), (bx + 1, by), (bx, by - 1), (bx, by + 1)].iter().map(|&(x, y)| reads.shifts[(y * bw + x) as usize]).collect();
            let (mx, my) = (nb.iter().map(|p| p.0).sum::<f32>() / 4.0, nb.iter().map(|p| p.1).sum::<f32>() / 4.0);
            assert!((s.0 - mx).abs() <= 1.0 && (s.1 - my).abs() <= 1.0, "block ({bx},{by}) shift {s:?} vs neighbours ({mx},{my})");
        } }
    }
    #[test]
    fn header_reads_through_a_homography() {
        let (g, h, _, im) = frame(CellFormat { pitch: 4, alphabet: Alphabet::C8 });
        let (s, a) = (1.3f32, 10f32.to_radians());
        let map = |x: f32, y: f32| (s * (x * a.cos() - y * a.sin()) + 300.0, s * (x * a.sin() + y * a.cos()) + 40.0);
        let src = [(0.0, 0.0), (1024.0, 0.0), (1024.0, 1024.0), (0.0, 1024.0)];
        let dst = [map(0.0, 0.0), map(1024.0, 0.0), map(1024.0, 1024.0), map(0.0, 1024.0)];
        let f2c = Homography::from_points(src, dst).unwrap();
        let cam = im.warp(&f2c.inverse().unwrap(), 1700, 1600);
        let hs = read_header(&cam, g.chrome(), &f2c);
        assert_eq!(hs, [Some(h), Some(h)]);
        assert!((sampling(&f2c, &g) - 1.3).abs() < 0.01);
        let bits = read_header_bits(&cam, g.chrome(), &f2c, 0);
        assert_eq!(bits.len(), 208);
    }
    #[test]
    fn brightness_gradient_is_fitted_per_block() {
        // a webcam-like frame: 0.55x brightness at the left edge rising to 1.0x at the right, 16 colours
        let (g, _, f, im) = frame(CellFormat { pitch: 4, alphabet: Alphabet::C16 });
        let mut cast = im.clone();
        for y in 0..cast.height { for x in 0..cast.width {
            let gain = 0.55 + 0.45 * x as f32 / 1023.0;
            let p = cast.get(x, y);
            cast.set(x, y, [(p[0] as f32 * gain + 8.0).min(255.0) as u8, (p[1] as f32 * gain + 8.0).min(255.0) as u8, (p[2] as f32 * gain + 8.0).min(255.0) as u8]);
        } }
        let local = read_cells(&cast, &g, &Homography::identity(), &ReadParams::default());
        let (bad, n) = wrong(&g, &f, &local);
        assert!(bad * 1000 < n * 2, "per-block colour: {bad} of {n} wrong");
        let global = read_cells(&cast, &g, &Homography::identity(), &ReadParams { local_colour: false, ..ReadParams::default() });
        let (bad0, _) = wrong(&g, &f, &global);
        assert!(bad0 > 4 * bad.max(1) && bad0 * 100 > n, "one global model only {bad0} of {n} wrong");
    }
    #[test]
    fn header_survives_a_two_pixel_geometry_error_at_webcam_scale() {
        // 0.6 camera px per screen px: a header cell is 4.8 px; a 2 px error in the fitted geometry must not lose it
        let (g, h, _, im) = frame(CellFormat { pitch: 12, alphabet: Alphabet::C8 });
        let f2c = Homography { m: [0.6, 0.0, 100.0, 0.0, 0.6, 80.0, 0.0, 0.0, 1.0] };
        let cam = im.warp(&f2c.inverse().unwrap(), 900, 800);
        let off = Homography { m: [0.6, 0.0, 102.0, 0.0, 0.6, 78.5, 0.0, 0.0, 1.0] };
        let hs = read_header(&cam, g.chrome(), &off);
        assert!(hs.iter().any(|x| *x == Some(h)), "{hs:?}");
    }
    #[test]
    fn colour_cast_is_fitted_away() {
        let (g, _, f, im) = frame(CellFormat { pitch: 4, alphabet: Alphabet::C8 });
        let mut cast = im.clone();
        for p in cast.data.chunks_mut(3) { let gains = [0.8f32, 0.93, 1.0]; for k in 0..3 { p[k] = (p[k] as f32 * gains[k] + 20.0).min(255.0) as u8; } }
        let reads = read_cells(&cast, &g, &Homography::identity(), &ReadParams::default());
        let (bad, n) = wrong(&g, &f, &reads);
        assert_eq!(bad, 0, "{bad} of {n} wrong under a colour cast");
        let _ = FlatCell::Unused;
    }
    #[test]
    fn neighbour_crosstalk_is_equalised() {
        // a 5 px box blur on 4 px cells mixes a quarter of each neighbour into every sample
        for alpha in [Alphabet::C8, Alphabet::G4, Alphabet::C16] {
            let (g, _, f, im) = frame(CellFormat { pitch: 4, alphabet: alpha });
            let blurred = box_blur(&im, 5);
            let plain = read_cells(&blurred, &g, &Homography::identity(), &ReadParams { equalise: false, ..ReadParams::default() });
            let (bad0, n) = wrong(&g, &f, &plain);
            assert!(bad0 * 1000 > n * 5, "{alpha:?}: the blur must cost something, only {bad0} of {n} wrong");
            let eq = read_cells(&blurred, &g, &Homography::identity(), &ReadParams::default());
            let (bad, _) = wrong(&g, &f, &eq);
            assert!(bad * 3 < bad0 && bad * 1000 < n * 3, "{alpha:?}: equalised {bad} vs plain {bad0} of {n}");
        }
        // and a clean frame loses nothing to the equaliser
        let (g, _, f, im) = frame(CellFormat { pitch: 4, alphabet: Alphabet::C8 });
        let reads = read_cells(&im, &g, &Homography::identity(), &ReadParams::default());
        assert_eq!(wrong(&g, &f, &reads).0, 0);
        assert_eq!(reads.alpha, 0.0, "a clean frame picks no crosstalk factor");
    }
    #[test]
    fn pattern_decision_follows_linear_light_blur() {
        // a camera blurs light, not gamma-encoded values: a black cell among whites is lifted far more than a
        // white cell among blacks is darkened. The pattern-dependent decision must beat the single threshold.
        use crate::testutil::photograph_with_homography;
        let fmt = CellFormat { pitch: 1, alphabet: Alphabet::Bw };
        let g = Arc::new(FlatGrid::try_new(1024, 1024, fmt).unwrap());
        let h = FlatHeader { format: fmt, cols: g.cols() as u16, rows: g.rows() as u16, frame_type: FrameType::Data, transfer_id: 7, segment: 1, block_id: 99, frame_seq: 0 };
        let payload: Vec<u8> = (0..g.payload_bytes()).map(|i| ((i as u64 * 2654435761) >> 7) as u8).collect();
        let f = compose(g.clone(), h, &payload);
        let (cam, f2c) = photograph_with_homography(&render(&f), 2400, 2400, 4.0, 0, 5);
        // gaussian-ish blur (radius 2 camera px, ~0.9 screen px) in linear light, then back to gamma 2.2
        let (w, hh) = (cam.width as usize, cam.height as usize);
        let lin: Vec<f32> = cam.data.iter().map(|&v| (v as f32 / 255.0).powf(2.2)).collect();
        let k = [1.0f32, 4.0, 6.0, 4.0, 1.0];
        let mut tmp = vec![0f32; lin.len()];
        for y in 0..hh { for x in 0..w { for c in 0..3 { let mut acc = 0.0; let mut n = 0.0; for (i, kv) in k.iter().enumerate() { let xx = x as i32 + i as i32 - 2; if xx >= 0 && (xx as usize) < w { acc += kv * lin[(y * w + xx as usize) * 3 + c]; n += kv; } } tmp[(y * w + x) * 3 + c] = acc / n; } } }
        let mut out = cam.clone();
        for y in 0..hh { for x in 0..w { for c in 0..3 { let mut acc = 0.0; let mut n = 0.0; for (i, kv) in k.iter().enumerate() { let yy = y as i32 + i as i32 - 2; if yy >= 0 && (yy as usize) < hh { acc += kv * tmp[(yy as usize * w + x) * 3 + c]; n += kv; } } out.data[(y * w + x) * 3 + c] = ((acc / n).powf(1.0 / 2.2) * 255.0).round().clamp(0.0, 255.0) as u8; } } }
        let base = ReadParams::default().tuned_for(1);
        let (bad_off, n) = wrong(&g, &f, &read_cells(&out, &g, &f2c, &ReadParams { pattern_decide: false, ..base }));
        let (bad_on, _) = wrong(&g, &f, &read_cells(&out, &g, &f2c, &ReadParams { pattern_decide: true, ..base }));
        assert!(bad_off * 1000 > n, "the blur must hurt the single threshold ({bad_off} of {n})");
        assert!(bad_on * 2 < bad_off, "pattern decision {bad_on} vs single threshold {bad_off} of {n}");
    }
    #[test]
    fn noisy_one_px_capture_keeps_its_tags() {
        // a 1 px b/w frame photographed at ~2.3 camera px per screen px with camera noise: the data cells still
        // read (strong margins), so the frame tag must be known almost everywhere; every block judged
        // "unknown" from its two tagged pilots alone erases 60 data cells, and on real phone captures that
        // was the main reason stripes failed
        use crate::testutil::photograph_with_homography;
        let fmt = CellFormat { pitch: 1, alphabet: Alphabet::Bw };
        let g = Arc::new(FlatGrid::try_new(1024, 1024, fmt).unwrap());
        let h = FlatHeader { format: fmt, cols: g.cols() as u16, rows: g.rows() as u16, frame_type: FrameType::Data, transfer_id: 7, segment: 1, block_id: 99, frame_seq: 5 };
        let payload: Vec<u8> = (0..g.payload_bytes()).map(|i| ((i as u64 * 2654435761) >> 7) as u8).collect();
        let f = compose(g.clone(), h, &payload);
        let (cam, f2c) = photograph_with_homography(&render(&f), 2500, 2500, 6.0, 24, 3);
        let reads = read_cells(&cam, &g, &f2c, &ReadParams { tag_hint: h.frame_seq & 3, ..ReadParams::default() }.tuned_for(1));
        let (bad, n) = wrong(&g, &f, &reads);
        assert!(bad * 100 < n * 3, "the cells themselves read: {bad} of {n} wrong");
        let unknown = reads.block_tag.iter().filter(|&&t| t == TAG_UNKNOWN).count();
        let wrong_tag = reads.block_tag.iter().filter(|&&t| t != TAG_UNKNOWN && t != 1).count();
        assert_eq!(wrong_tag, 0, "blocks with a wrong tag");
        assert!(unknown * 1000 <= reads.block_tag.len() * 2, "{unknown} of {} blocks have an unknown tag", reads.block_tag.len());
    }
    fn two_frames(seq_a: u8, seq_b: u8) -> (Arc<FlatGrid>, crate::flat::FlatFrame, crate::flat::FlatFrame) {
        let fmt = CellFormat { pitch: 4, alphabet: Alphabet::C8 };
        let g = Arc::new(FlatGrid::try_new(1024, 1024, fmt).unwrap());
        let h = FlatHeader { format: fmt, cols: g.cols() as u16, rows: g.rows() as u16, frame_type: FrameType::Data, transfer_id: 7, segment: 1, block_id: 99, frame_seq: seq_a };
        let pa: Vec<u8> = (0..g.payload_bytes()).map(|i| ((i as u64 * 2654435761) >> 7) as u8).collect();
        let pb: Vec<u8> = (0..g.payload_bytes()).map(|i| ((i as u64 * 40503 + 17) >> 3) as u8).collect();
        (g.clone(), compose(g.clone(), h, &pa), compose(g, FlatHeader { frame_seq: seq_b, ..h }, &pb))
    }
    #[test]
    fn straddled_capture_tags_its_blocks_by_frame() {
        // top of frame A (seq 4, tag 0) joined to the bottom of frame B (seq 5, tag 1) at row 520 px
        let (g, a, b) = two_frames(4, 5);
        let (ia, ib) = (render(&a), render(&b));
        let mut cam = ia.clone();
        for y in 520..1024 { for x in 0..1024 { cam.set(x, y, ib.get(x, y)); } }
        let reads = read_cells(&cam, &g, &Homography::identity(), &ReadParams { tag_hint: 0, ..ReadParams::default() });
        let (bw, bh) = g.blocks();
        let (mut bad, mut n) = (0, 0);
        for i in 0..g.used_data_cells() as u32 {
            let (c, r) = g.data_position(i);
            let (_, y) = g.cell_center(c, r);
            let truth = if y < 520.0 { a.value(c, r) } else { b.value(c, r) };
            n += 1; if reads.at(c, r).0 != truth { bad += 1; }
        }
        assert!(bad * 1000 < n, "{bad} of {n} cells wrong across the straddle");
        for by in 0..bh { for bx in 0..bw {
            let (_, y0) = g.cell_top_left(bx * BLOCK, by * BLOCK);
            let (y0, y1) = (y0 as f32, (y0 + BLOCK * 4) as f32);
            let tag = reads.block_tag[(by * bw + bx) as usize];
            if tag == TAG_UNKNOWN { continue; }
            if y1 <= 520.0 { assert_eq!(tag, 0, "block ({bx},{by}) above the line (tags {:?})", { let mut h = std::collections::HashMap::new(); for &t in &reads.block_tag { *h.entry(t).or_insert(0) += 1; } h }); }
            else if y0 >= 520.0 { assert_eq!(tag, 1, "block ({bx},{by}) below the line"); }
        } }
    }
    #[test]
    fn wrong_tag_hint_is_corrected_on_a_whole_frame() {
        let (g, _, b) = two_frames(4, 7);   // frame B has tag 3: both tagged pilots shifted
        let ib = render(&b);
        let reads = read_cells(&ib, &g, &Homography::identity(), &ReadParams { tag_hint: 0, ..ReadParams::default() });
        let mut hist = std::collections::HashMap::new(); for &t in &reads.block_tag { *hist.entry(t).or_insert(0) += 1; }
        assert!(reads.block_tag.iter().all(|&t| t == 3 || t == TAG_UNKNOWN), "every block reads tag 3 (or no tag): {hist:?}");
        assert!(reads.block_tag.iter().filter(|&&t| t == TAG_UNKNOWN).count() <= 4, "{hist:?}");
        let bad = (0..g.used_data_cells() as u32).filter(|&i| { let (c, r) = g.data_position(i); reads.at(c, r).0 != b.value(c, r) }).count();
        assert_eq!(bad, 0);
    }
    #[test]
    fn black_white_reads_by_local_thresholds_without_pilots() {
        // 1 px b/w under a strong brightness gradient and a 2 px blur: thresholds from each block's own cells
        let (g, _, f, im) = { let fmt = CellFormat { pitch: 1, alphabet: Alphabet::Bw }; frame(fmt) };
        let mut cast = im.clone();
        for y in 0..cast.height { for x in 0..cast.width {
            let gain = 0.45 + 0.55 * x as f32 / 1023.0;
            let p = cast.get(x, y);
            cast.set(x, y, [(p[0] as f32 * gain + 10.0).min(255.0) as u8, (p[1] as f32 * gain + 10.0).min(255.0) as u8, (p[2] as f32 * gain + 10.0).min(255.0) as u8]);
        } }
        let f2c = Homography { m: [2.4, 0.0, 0.0, 0.0, 2.4, 0.0, 0.0, 0.0, 1.0] };
        let cam = box_blur(&cast.warp(&f2c.inverse().unwrap(), 2458, 2458), 3);
        let local = read_cells(&cam, &g, &f2c, &ReadParams::default());
        let (bad, n) = wrong(&g, &f, &local);
        let pilots_only = read_cells(&cam, &g, &f2c, &ReadParams { local_threshold: false, ..ReadParams::default() });
        let (bad0, _) = wrong(&g, &f, &pilots_only);
        assert!(bad * 1000 < n * 2, "local thresholds: {bad} of {n} wrong (pilot colour model: {bad0})");
        assert!(bad <= bad0, "local thresholds must not be worse than the pilot colour model ({bad} vs {bad0})");
    }
}
