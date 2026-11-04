//! Fast luma reader for the black/white formats (the design notes).
//! Same output as `read_cells` (values, margins, per-block shifts, frame tags), a different search: the luma
//! plane is box-blurred once, and the 8x8 blocks' sampling shifts come coarse to fine, all in parallel: the
//! tile's shift from every anchor block's costs over the full reach summed; sparse anchor blocks each search
//! a disk around it, those on their disk's edge or away from their neighbours follow their neighbours' median
//! until nothing changes, outliers take their neighbourhood's median; the field between anchors is
//! interpolated, every other block refines it with one narrow search, and a lattice refinement pooled over
//! neighbouring blocks' pilots sets every block's shift, instead of a dense 441-shift search over every block.
//! At 2 px and coarser b/w the pattern decision's doubtful cells are then decided again by a local equaliser
//! (`equalise_bw`), which keeps small and tilted tiles reading (docs/2026-10-31-fhd-2px-floor.md).

use super::layout::{Chrome, FlatCell, FlatGrid, BLOCK, CELL_DATA, CELL_PAST, CELL_PILOT, PILOT_LOCAL};
use super::read::{assign_tags, pattern_decide_grid, sampling, tag_known, FlatReads, Pilot, TAG_FLAT};
use crate::detect::Homography;
use crate::image::GrayImage;
use crate::par::{par_map, par_ranges};

/// Luma of each palette level (up to 8 for g8; entries past the alphabet unused).
type Lv = [f32; 8];

/// Box blur of side `k` (same box as `read::box_blur`: x - k/2 .. x - k/2 + k, clamped at the image edge, mean
/// rounded half up), computed only inside `roi` = (x0, y0, x1, y1); pixels outside stay 0. Row bands in
/// parallel; per band a running vertical sum of horizontal box sums, and the interior's constant box area
/// divided by an exact multiply-shift (a division per pixel was most of this function's time).
fn blur_roi(y: &GrayImage, k: u32, roi: (u32, u32, u32, u32)) -> GrayImage {
    if k <= 1 { return y.clone(); }
    let (w, h) = (y.width as usize, y.height as usize);
    let (x0, y0, x1, y1) = (roi.0 as usize, roi.1 as usize, roi.2 as usize, roi.3 as usize);
    let mut out = GrayImage::new(y.width, y.height);
    if x1 <= x0 || y1 <= y0 { return out; }
    let (rw, rh, k, r0) = (x1 - x0, y1 - y0, k as usize, (k / 2) as usize);
    // k = 2 (the 1 px format): the box of (x, y) is (x-1..x+1, y-1..y+1); interior pixels in one vectorisable
    // pass, the ROI's outer ring (where the box meets the image edge) by the general path below
    if k == 2 && x0 >= 1 && y0 >= 1 {
        let bands: Vec<(usize, Vec<u8>)> = par_ranges(rh, |range| {
            let start = range.start;
            let mut buf = vec![0u8; range.len() * rw];
            for (bi, oy) in range.enumerate() {
                let yy = y0 + oy;
                let (ra, rb) = (&y.data[(yy - 1) * w + x0 - 1..(yy - 1) * w + x1], &y.data[yy * w + x0 - 1..yy * w + x1]);
                let line = &mut buf[bi * rw..(bi + 1) * rw];
                for i in 0..rw { line[i] = ((ra[i] as u16 + ra[i + 1] as u16 + rb[i] as u16 + rb[i + 1] as u16 + 2) >> 2) as u8; }
            }
            (start, buf)
        });
        for (start, buf) in bands { for (bi, row) in buf.chunks(rw).enumerate() { let o = (y0 + start + bi) * w + x0; out.data[o..o + rw].copy_from_slice(row); } }
        return out;
    }
    let col = |x: usize| (x.saturating_sub(r0), (x + k - r0).min(w));
    let widths: Vec<u32> = (x0..x1).map(|x| { let (a, b) = col(x); (b - a) as u32 }).collect();
    // the box sums of one input row for every ROI column
    let hsum_row = |j: usize, out: &mut [u32]| {
        let row = &y.data[j * w..(j + 1) * w];
        let (a, b) = col(x0);
        let mut sum: u32 = row[a..b].iter().map(|&v| v as u32).sum();
        let (mut lo, mut hi) = (a, b);
        out[0] = sum;
        for (i, x) in (x0 + 1..x1).enumerate() {
            let (na, nb) = col(x);
            while lo < na { sum -= row[lo] as u32; lo += 1; }
            while hi < nb { sum += row[hi] as u32; hi += 1; }
            out[i + 1] = sum;
        }
    };
    let bands: Vec<(usize, Vec<u8>)> = par_ranges(rh, |range| {
        let start = range.start;
        let mut buf = vec![0u8; range.len() * rw];
        let (mut acc, mut tmp) = (vec![0u32; rw], vec![0u32; rw]);
        let mut win = (0usize, 0usize);
        for (bi, oy) in range.enumerate() {
            let yy = y0 + oy;
            let (a, b) = (yy.saturating_sub(r0), (yy + k - r0).min(h));
            if bi == 0 {
                for j in a..b { hsum_row(j, &mut tmp); for (s, t) in acc.iter_mut().zip(&tmp) { *s += t; } }
            } else {
                for j in win.0..a { hsum_row(j, &mut tmp); for (s, t) in acc.iter_mut().zip(&tmp) { *s -= t; } }
                for j in win.1..b { hsum_row(j, &mut tmp); for (s, t) in acc.iter_mut().zip(&tmp) { *s += t; } }
            }
            win = (a, b);
            let rows_n = (b - a) as u32;
            // exact (sum + n/2) / n for the interior area n = k * rows_n: sums stay far below 2^32 / n
            let n_int = k as u32 * rows_n;
            let m = ((1u64 << 32) / n_int as u64) + 1;
            let line = &mut buf[bi * rw..(bi + 1) * rw];
            for ((o, &sum), &wx) in line.iter_mut().zip(&acc).zip(&widths) {
                *o = if wx as usize == k { (((sum + n_int / 2) as u64 * m) >> 32) as u8 } else { let n = wx * rows_n; ((sum + n / 2) / n) as u8 };
            }
        }
        (start, buf)
    });
    for (start, buf) in bands {
        for (bi, row) in buf.chunks(rw).enumerate() { let o = (y0 + start + bi) * w + x0; out.data[o..o + rw].copy_from_slice(row); }
    }
    out
}

/// The integer scale (1, 2 or 3: high-DPI displays and VM viewers) and integer pixel offset of a tile captured
/// pixel-exactly (a screen capture of a window).
/// The marker geometry only gets within a pixel or so (measured: 0.4-0.5 px off, scale 0.9993), so the
/// nine integer offsets around it are tried and one is accepted only if the whole border timing pattern
/// (alternating white and black modules) reads exactly at the module centres.
fn pixel_exact_offset(y: &GrayImage, h: &Homography, grid: &FlatGrid) -> Option<(i64, i64, i64)> {
    let (w, hh) = (grid.width_px() as f32, grid.height_px() as f32);
    let (ox, oy) = h.apply(0.0, 0.0);
    let k = ((h.apply(w, 0.0).0 - ox) / w).round();
    if !(1.0..=3.0).contains(&k) { return None; }
    for (x, yy) in [(w, 0.0), (0.0, hh), (w, hh), (w / 2.0, hh / 2.0)] {
        let (cx, cy) = h.apply(x, yy);
        if (cx - k * x - ox).abs() > 1.5 * k || (cy - k * yy - oy).abs() > 1.5 * k { return None; }
    }
    let k_i = k as i64;
    let chrome = grid.chrome();
    let n = chrome.border_module_count();
    let (iw, ih) = (y.width as i64, y.height as i64);
    let (bx, by) = (ox.round() as i64, oy.round() as i64);
    for dy in -k_i..=k_i { for dx in -k_i..=k_i {
        let (tx, ty) = (bx + dx, by + dy);
        if tx < 0 || ty < 0 || tx + k_i * w as i64 > iw || ty + k_i * hh as i64 > ih { continue; }
        let at = |i: u32| { let (cx, cy) = chrome.border_module_center(i); y.data[(((cy * k) as i64 + ty) * iw + (cx * k) as i64 + tx) as usize] };
        let (mut lo, mut hi) = (255u8, 0u8);
        for i in 0..n { let v = at(i); lo = lo.min(v); hi = hi.max(v); }
        if hi < lo.saturating_add(64) { continue; }
        let mid = ((lo as u16 + hi as u16) / 2) as u8;
        let border_bad = (0..n).filter(|&i| (at(i) > mid) != crate::flat::Chrome::border_is_white(i)).count();
        if std::env::var_os("LB_DEBUG").is_some() { eprintln!("exact? k {k_i} offset ({tx},{ty}): border mismatches {border_bad} of {n}"); }
        // overlays on a real screen (a cursor, a status line, a notification) cover some border modules: up to
        // 8% may differ; the pilots below decide at cell resolution
        if border_bad * 100 > n as usize * 8 { continue; }
        // the border's 8 px modules pass at a one-pixel slip too: the untagged pilots (one cell each, values
        // independent of the frame tag) decide at cell resolution
        let fmt = grid.format();
        let (pal, n_lv) = (fmt.alphabet.palette(), fmt.alphabet.size());
        let (mut seen, mut good) = (0u32, 0u32);
        // a spread subsample of about 2000 pilots decides as well as all of them
        let stride = (grid.pilot_positions().len() / 2000).max(1);
        for &(c, r) in grid.pilot_positions().iter().step_by(stride) {
            let kind = PILOT_LOCAL.iter().position(|&q| q == (r % BLOCK, c % BLOCK)).unwrap_or(0);
            if kind % 2 != 0 { continue; }
            let (x, yy) = grid.cell_center(c, r);
            let v = y.data[(((yy * k).floor() as i64 + ty) * iw + (x * k).floor() as i64 + tx) as usize] as f32;
            let expect = grid.pilot_colour_at(c, r, 0) as usize;
            // the palette level nearest to the pixel, on the border's black..white scale
            let level = |k: usize| lo as f32 + (hi as f32 - lo as f32) * pal[k][0] as f32 / 255.0;
            let got = (0..n_lv).min_by(|&a, &b| (v - level(a)).abs().partial_cmp(&(v - level(b)).abs()).unwrap()).unwrap();
            seen += 1;
            if got == expect { good += 1; }
        }
        if std::env::var_os("LB_DEBUG").is_some() { eprintln!("exact? k {k_i} offset ({tx},{ty}): pilots {good} of {seen}"); }
        if seen > 0 && good * 100 >= seen * 98 { return Some((tx, ty, k_i)); }
    } }
    None
}

/// Pixel-exact read: every cell is its centre pixel; levels from the untagged pilots, frame tags from the
/// tagged ones, each cell the nearest level. No blur, no alignment search: the capture is the sender's pixels.
fn read_exact(y: &GrayImage, grid: &FlatGrid, off: (i64, i64, i64), tag_hint: u8) -> FlatReads {
    let fmt = grid.format();
    let (n_lv, pal) = (fmt.alphabet.size(), fmt.alphabet.palette());
    let (cols, rows) = (grid.cols(), grid.rows());
    let (bw, bh) = grid.blocks();
    let nblocks = (bw * bh) as usize;
    let (iw, ih) = (y.width as i64, y.height as i64);
    let kf = off.2 as f32;
    let px = |c: u32, r: u32| -> Option<u8> {
        let (x, yy) = grid.cell_center(c, r);
        let (ix, iy) = ((x * kf).floor() as i64 + off.0, (yy * kf).floor() as i64 + off.1);
        if ix < 0 || iy < 0 || ix >= iw || iy >= ih { None } else { Some(y.data[(iy * iw + ix) as usize]) }
    };
    let kind_of = |c: u32, r: u32| PILOT_LOCAL.iter().position(|&q| q == (r % BLOCK, c % BLOCK)).unwrap_or(0);
    // levels: mean pixel of each palette level among the untagged pilots (kinds 0 and 2)
    let mut acc = [(0u64, 0u32); 8];
    for &(c, r) in grid.pilot_positions() {
        if kind_of(c, r) % 2 != 0 { continue; }
        let Some(v) = px(c, r) else { continue };
        let k = grid.pilot_colour_at(c, r, tag_hint & 3) as usize;
        acc[k].0 += v as u64; acc[k].1 += 1;
    }
    let lv: [f32; 8] = std::array::from_fn(|k| if k < n_lv && acc[k].1 > 0 { acc[k].0 as f32 / acc[k].1 as f32 } else if k < n_lv { pal[k][0] as f32 } else { 0.0 });
    let nearest = |v: f32| -> (u8, u16) {
        let (mut best, mut d1, mut d2) = (0usize, f32::MAX, f32::MAX);
        for k in 0..n_lv { let d = (v - lv[k]).powi(2); if d < d1 { d2 = d1; d1 = d; best = k; } else if d < d2 { d2 = d; } }
        let scale = (255.0 / (lv[n_lv - 1] - lv[0]).abs().max(1.0)).powi(2);
        (best as u8, (3.0 * (d2 - d1) * scale / 4.0).clamp(0.0, 65535.0) as u16)
    };
    // frame tags from the tagged pilots (kinds 1 and 3), the same cut search as the camera path
    let mut tcost = vec![[0f32; 4]; nblocks];
    let mut tagged = vec![0u8; nblocks];
    for &(c, r) in grid.pilot_positions() {
        let kk = kind_of(c, r);
        if kk != 1 && kk != 3 { continue; }
        let Some(v) = px(c, r) else { continue };
        let b = ((r / BLOCK) * bw + c / BLOCK) as usize;
        for t in 0..4u8 { tcost[b][t as usize] += (v as f32 - lv[grid.pilot_colour_at(c, r, t) as usize]).powi(2); }
        tagged[b] += 1;
    }
    for b in 0..nblocks { if tagged[b] < 2 { tcost[b] = [0.0; 4]; } }
    let block_tag = assign_tags(&tcost, &tagged, bh as usize, bw as usize);
    let mut values = vec![0u8; (cols * rows) as usize];
    let mut margins = vec![0u16; (cols * rows) as usize];
    // a cell is one 8-bit pixel: its decision and margin come from a 256-entry table; positions are integer
    // steps along each row (cell centres at origin + c * pitch + pitch / 2)
    let lut: Vec<(u8, u16)> = (0..256).map(|v| nearest(v as f32)).collect();
    let (x0, _) = grid.cell_center(0, 0);
    let (x0, p) = ((x0 * kf).floor() as i64 + off.0, fmt.pitch as i64 * off.2);
    let rows_out: Vec<Vec<(u8, u16)>> = par_ranges(rows as usize, |rr| {
        let mut out = Vec::with_capacity(rr.len() * cols as usize);
        for r in rr {
            let (_, yy) = grid.cell_center(0, r as u32);
            let iy = (yy * kf).floor() as i64 + off.1;
            if iy < 0 || iy >= ih { out.extend(std::iter::repeat((0u8, 0u16)).take(cols as usize)); continue; }
            let row = &y.data[(iy * iw) as usize..((iy + 1) * iw) as usize];
            for c in 0..cols as i64 {
                let ix = x0 + c * p;
                out.push(if ix >= 0 && ix < iw { lut[row[ix as usize] as usize] } else { (0, 0) });
            }
        }
        out
    });
    for (i, (v, m)) in rows_out.into_iter().flatten().enumerate() { values[i] = v; margins[i] = m; }
    FlatReads { cols, rows, values, margins, shifts: vec![(0.0, 0.0); nblocks], alpha: 0.0, block_tag, mix: 0.0, edge_px: 0.0 }
}

/// The blurred plane at camera pixel (cx, cy) (pixel centres at half-integers), bilinear; None outside the image.
fn sample_at(src: &GrayImage, cx: f32, cy: f32) -> Option<f32> {
    if cx < 0.0 || cy < 0.0 || cx >= src.width as f32 || cy >= src.height as f32 { None } else { Some(src.sample(cx - 0.5, cy - 0.5)) }
}

/// One block's pilots in the pooled refinement's 7x7 window: entry j * 7 + i sums (sample - level)^2, and
/// counts the pilots, over the pilots `pts` = (x, y, level) moved by (off[i].0, off[j].1) that lie inside the
/// image, each sample as `sample_at` takes it. The window is a product of 7 offsets along each axis, so the
/// image bounds, the pixel cell and its weight are found once per axis position, and each pixel row's
/// horizontal blend once for the candidate rows that share it: the same arithmetic in the same order as
/// `sample_at`, so the same bits (locked by `pilot_window_is_sample_at`). A NaN position (only from a
/// degenerate homography) is skipped here where `sample_at` would clamp it to the image edge.
fn pilot_window(src: &GrayImage, pts: impl Iterator<Item = (f32, f32, f32)> + Clone, off: &[(f32, f32); 7]) -> [(f32, u32); 49] {
    let mut v = [(0f32, 0u32); 49];
    let (sw, sh) = (src.width as usize, src.height as usize);
    if sw < 2 || sh < 2 {
        // GrayImage::sample's general path
        for j in 0..7 { for i in 0..7 {
            let e = &mut v[j * 7 + i];
            for (x, y, t) in pts.clone() { if let Some(val) = sample_at(src, x + off[i].0, y + off[j].1) { e.0 += (val - t).powi(2); e.1 += 1; } }
        } }
        return v;
    }
    let (wf, hf) = (src.width as f32, src.height as f32);
    // (pixel cell, weight, inside the image) per axis position, as `sample_at` and `GrayImage::sample` find it
    let axis = |c: f32, lim: f32, n: usize| -> (usize, f32, bool) {
        let x = (c - 0.5).max(0.0).min(lim - 1.0);
        let x0 = (x as usize).min(n - 2);
        (x0, x - x0 as f32, c >= 0.0 && c < lim)
    };
    let px = &src.data;
    for (x, y, t) in pts {
        let xs: [(usize, f32, bool); 7] = std::array::from_fn(|i| axis(x + off[i].0, wf, sw));
        let ys: [(usize, f32, bool); 7] = std::array::from_fn(|j| axis(y + off[j].1, hf, sh));
        // the horizontal blends of pixel rows `row` and `row + 1` at the 7 columns
        let blend = |row: usize| -> [f32; 7] { let r = &px[row * sw..]; std::array::from_fn(|i| { let (x0, fx, _) = xs[i]; let (a, c) = (r[x0] as f32, r[x0 + 1] as f32); a + (c - a) * fx }) };
        let (mut row, mut top, mut bot) = (usize::MAX, [0f32; 7], [0f32; 7]);
        for (j, &(y0, fy, yin)) in ys.iter().enumerate() {
            if y0 != row {
                // the offsets grow with j, so the rows are non-decreasing in j: a row one below the last
                // reuses its bottom blend as the top (any other row blends both)
                if row != usize::MAX && y0 == row + 1 { top = bot; } else { top = blend(y0); }
                bot = blend(y0 + 1);
                row = y0;
            }
            if !yin { continue; }
            for i in 0..7 {
                if !xs[i].2 { continue; }
                let val = top[i] + (bot[i] - top[i]) * fy;
                let e = &mut v[j * 7 + i];
                e.0 += (val - t).powi(2); e.1 += 1;
            }
        }
    }
    v
}

/// A block whose samples spread less than this share of the median block's is covered or blown out
/// (`mark_flat_blocks`). A covered block spreads by the sensor's noise (a few levels against 40+); blur, small
/// cells and glare lower the spread of every block in their region alike, and stay well above it.
const FLAT_SPREAD: f32 = 0.2;

/// Blocks whose 64 cell samples barely spread become `TAG_FLAT`: a patch of the tile covered (a
/// notification over the sender, a thumb in front of the camera) or blown out. Real data always spreads (a
/// random mix of levels, blur or not); a covered patch is one flat level. In b/w the frame tag cannot tell:
/// two tagged pilots of two colours make four patterns, one per tag, so even a solid patch fits a tag exactly
/// and its confident, wrong cells cost Reed-Solomon two parity bytes each; as erasures they cost one.
/// Measured: 4% of a 2 px b/w tile covered read 46 of 111 codewords, 5% 21-32; now 109-111. (The four pilots
/// alone are too few: at 1.7 camera px per cell their swing flagged readable blocks.)
fn mark_flat_blocks(block_tag: &mut [u8], raw: &[f32], cols: u32, rows: u32, bw: u32) {
    let mut acc = vec![(0u32, 0f64, 0f64); block_tag.len()];
    for r in 0..rows {
        let row = &raw[(r * cols) as usize..((r + 1) * cols) as usize];
        let base = ((r / BLOCK) * bw) as usize;
        for (c, &v) in row.iter().enumerate() {
            if v.is_nan() { continue; }
            let a = &mut acc[base + c / BLOCK as usize];
            a.0 += 1; a.1 += v as f64; a.2 += (v as f64) * (v as f64);
        }
    }
    // a block mostly out of the picture is left as it is: its cells come with zero margins, suspects RS may
    // erase or keep (about half are right); marked missing they cost more (3 px with a corner cut off: 8 of 49
    // codewords as suspects, 0 as erasures, every codeword then over the erasure budget and not tried)
    let spread: Vec<Option<f32>> = acc.iter().map(|&(n, s, ss)| (n >= 32).then(|| { let m = s / n as f64; (ss / n as f64 - m * m).max(0.0).sqrt() as f32 })).collect();
    let mut known: Vec<f32> = spread.iter().flatten().copied().collect();
    if known.len() < 8 { return; }
    let mid = known.len() / 2;
    let median = *known.select_nth_unstable_by(mid, f32::total_cmp).1;
    if median <= 0.0 { return; }
    for (b, sd) in spread.iter().enumerate() {
        if sd.is_some_and(|sd| sd < FLAT_SPREAD * median) { block_tag[b] = TAG_FLAT; }
    }
}

/// Per column of a grid row, the inputs of `sample_row`: the homography's terms in x (m0 x, m3 x, m6 x for
/// the column's cell centre x), the shift at the block column to the left `s0` and the step `ds` to the
/// one on the right, and the column's weight `fu` between them.
struct RowCols<'a> { hx: [&'a [f32]; 3], fu: &'a [f32], s0x: &'a [f32], dsx: &'a [f32], s0y: &'a [f32], dsy: &'a [f32] }

/// `sample_row`'s per-column working arrays, kept across rows.
#[derive(Default)]
struct RowScratch { xs: Vec<f32>, ys: Vec<f32>, fxs: Vec<f32>, fys: Vec<f32>, ok: Vec<bool> }

/// One grid row's samples: column c at the camera position `h.apply(x_c, yy)` moved by `s0 + ds * fu`, as
/// `sample_at` takes it, NaN outside the image. The homography's x terms come per column and its y terms
/// once per row, so a cell costs its two divisions and one bilinear sample: the same operations in the same
/// order as `h.apply` and `sample_at`, so the same bits (locked by `sample_row_is_sample_at`). The pixel
/// cell is found by an exact whole-number step instead of a per-lane saturating cast, which does not
/// vectorise; images under 2 px a side, or too large for that step, go through `sample_at`.
fn sample_row(src: &GrayImage, hm: &[f32; 9], yy: f32, cols: &RowCols, scratch: &mut RowScratch, out: &mut [f32]) {
    let n = out.len();
    let (sw, sh) = (src.width as usize, src.height as usize);
    let (wf, hf) = (src.width as f32, src.height as f32);
    let (ay, by, wy) = (hm[1] * yy, hm[4] * yy, hm[7] * yy);
    // every slice cut to the row's length: no bounds checks, so the passes vectorise
    let (h0, h1, h2, fu, s0x, dsx, s0y, dsy) = (&cols.hx[0][..n], &cols.hx[1][..n], &cols.hx[2][..n], &cols.fu[..n], &cols.s0x[..n], &cols.dsx[..n], &cols.s0y[..n], &cols.dsy[..n]);
    if sw < 2 || sh < 2 || sw >= 1 << 23 || sh >= 1 << 23 {
        for (c, o) in out.iter_mut().enumerate() {
            let w = h2[c] + wy + hm[8];
            let w = if w.abs() < 1e-12 { 1e-12 } else { w };
            let (cx, cy) = ((h0[c] + ay + hm[2]) / w, (h1[c] + by + hm[5]) / w);
            *o = sample_at(src, cx + s0x[c] + dsx[c] * fu[c], cy + s0y[c] + dsy[c] * fu[c]).unwrap_or(f32::NAN);
        }
        return;
    }
    const TWO23: f32 = 8_388_608.0;
    let (xmax, ymax) = ((sw - 2) as f32, (sh - 2) as f32);
    for v in [&mut scratch.xs, &mut scratch.ys, &mut scratch.fxs, &mut scratch.fys] { v.resize(n, 0.0); }
    scratch.ok.resize(n, false);
    let (xs, ys, fxs, fys, ok) = (&mut scratch.xs[..n], &mut scratch.ys[..n], &mut scratch.fxs[..n], &mut scratch.fys[..n], &mut scratch.ok[..n]);
    for c in 0..n {
        let w = h2[c] + wy + hm[8];
        let w = if w.abs() < 1e-12 { 1e-12 } else { w };
        let (cx, cy) = ((h0[c] + ay + hm[2]) / w, (h1[c] + by + hm[5]) / w);
        let (px, py) = (cx + s0x[c] + dsx[c] * fu[c], cy + s0y[c] + dsy[c] * fu[c]);
        ok[c] = !(px < 0.0 || py < 0.0 || px >= wf || py >= hf);
        // GrayImage::sample's cell and weights. x, y lie in [0, side - 1] (below 2^23), where adding and
        // taking away 2^23 rounds to the nearest whole number exactly, one less when that is above:
        // the cell `as usize` gives
        let (x, y) = ((px - 0.5).max(0.0).min(wf - 1.0), (py - 0.5).max(0.0).min(hf - 1.0));
        let (rx, ry) = ((x + TWO23) - TWO23, (y + TWO23) - TWO23);
        let (x0, y0) = ((if rx > x { rx - 1.0 } else { rx }).min(xmax), (if ry > y { ry - 1.0 } else { ry }).min(ymax));
        (xs[c], ys[c], fxs[c], fys[c]) = (x0, y0, x - x0, y - y0);
    }
    let px = &src.data;
    for (((((o, &x0), &y0), &fx), &fy), &ok) in out.iter_mut().zip(&*xs).zip(&*ys).zip(&*fxs).zip(&*fys).zip(&*ok) {
        let i = y0 as i32 as usize * sw + x0 as i32 as usize;
        let d = &px[i..i + sw + 2];
        let (a, b, cc, e) = (d[0] as f32, d[1] as f32, d[sw] as f32, d[sw + 1] as f32);
        let (top, bot) = (a + (b - a) * fx, cc + (e - cc) * fx);
        *o = if ok { top + (bot - top) * fy } else { f32::NAN };
    }
}

// The b/w equaliser (`equalise_bw`). Measured on the degraded real captures (`tests/fhd2px_floor.rs`: 12
// Full HD 2 px captures area-averaged and tilted), codewords read cold (no tracking hint).
/// Side, in cells, of the regions the equaliser fits its own model in: at 1.3 px per cell straight on 91.8%
/// with regions of 24, 87.6% of 32, 85.9% of 40 (one model for the whole tile reads worse than the pattern
/// decision alone: how the camera mixes neighbouring cells changes over the tile).
const EQ_REGION: usize = 24;
/// Round 1 decides only the cells the pattern decision left within this distance of its threshold on the
/// black..white axis; the others keep its decision (at 1.3 px per cell straight on / 1.7 px at 30 degrees of
/// yaw: every cell 82.9 / 93.7%, within 0.5 86.2 / 93.8%, 0.35 89.6 / 95.6%, 0.25 91.8 / 95.9%, 0.15 87.9 /
/// 94.4%, 0.1 73.6 / 86.3%: where the pattern decision is sure, it is more often right than a region's linear
/// model).
const EQ_DOUBT: f32 = 0.25;
/// Round 2 decides again only the cells whose pattern changed in round 1 and those round 1 left within this
/// distance of their threshold (the refit moves thresholds by far less: within 0.2 points of deciding every
/// cell again; 0.3 the same, 0.08 within 0.5 points).
const EQ_RECHECK: f32 = 0.15;

/// A region's model is used only when its black and white predictions lie at least this share of the frame's
/// black..white span apart (before: 1 luma level, whatever the contrast).
const EQ_MIN_GAP: f32 = 0.05;
/// In place, the 512-point Walsh-Hadamard transforms of two interleaved tables (`v[2 p]`, `v[2 p + 1]`):
/// out[k] = sum over p of in[p] * (-1)^popcount(p & k) for each. Every stage on contiguous halves.
fn wht512x2(v: &mut [f32; 1024]) {
    let mut h = 2;
    while h < 1024 {
        for pair in v.chunks_exact_mut(2 * h) {
            let (a, b) = pair.split_at_mut(h);
            for (x, y) in a.iter_mut().zip(b.iter_mut()) { let (p, q) = (*x, *y); *x = p + q; *y = p - q; }
        }
        h *= 2;
    }
}

/// The solution of the 10 x 10 symmetric positive definite system `a x = y` (Cholesky); None when it is not
/// positive definite (too few or too alike samples).
fn solve10(a: &[[f64; 10]; 10], y: &[f64; 10]) -> Option<[f64; 10]> {
    let mut l = [[0f64; 10]; 10];
    for i in 0..10 {
        for j in 0..=i {
            let s: f64 = a[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j { if s <= 1e-6 * a[i][i].abs().max(1.0) { return None; } l[i][i] = s.sqrt(); } else { l[i][j] = s / l[j][j]; }
        }
    }
    let mut z = [0f64; 10];
    for i in 0..10 { z[i] = (y[i] - (0..i).map(|k| l[i][k] * z[k]).sum::<f64>()) / l[i][i]; }
    let mut x = [0f64; 10];
    for i in (0..10).rev() { x[i] = (z[i] - (i + 1..10).map(|k| l[k][i] * x[k]).sum::<f64>()) / l[i][i]; }
    Some(x)
}

/// The cells without a value (1), with a border of them one cell wide around the grid, and every cell's role in
/// the equaliser: bit 0 = it fits (it and its 8 neighbours have values, its sample is read), bit 1 = a data
/// cell to decide, bit 2 = a neighbour without a value. `nv` comes in with the grid's cells set; `raw` None:
/// every sample is read.
fn eq_roles(cols: usize, rows: usize, kinds: &[u8], nv: &[u8], raw: Option<&[f32]>) -> Vec<u8> {
    let pw = cols + 2;
    // whether its sample is read, then the rest (one pass per element type)
    let mut role: Vec<u8> = match raw { Some(raw) => raw.iter().map(|v| !v.is_nan() as u8).collect(), None => vec![1; cols * rows] };
    for r in 0..rows {
        let (up, mid, dn) = (&nv[r * pw..][..pw], &nv[(r + 1) * pw..][..pw], &nv[(r + 2) * pw..][..pw]);
        let (ul, uc, ur) = (&up[..cols], &up[1..cols + 1], &up[2..cols + 2]);
        let (ml, mc, mr) = (&mid[..cols], &mid[1..cols + 1], &mid[2..cols + 2]);
        let (dl, dc, dr) = (&dn[..cols], &dn[1..cols + 1], &dn[2..cols + 2]);
        let (kr, o) = (&kinds[r * cols..][..cols], &mut role[r * cols..][..cols]);
        for c in 0..cols {
            let free = ml[c] | mr[c] | ul[c] | uc[c] | ur[c] | dl[c] | dc[c] | dr[c];
            let read = o[c];
            o[c] = (read & (mc[c] ^ 1) & (free ^ 1)) | (read & (kr[c] == CELL_DATA) as u8) << 1 | free << 2;
        }
    }
    role
}

/// Black/white decisions again, from a local model of what the camera does to a cell: in every region of
/// `EQ_REGION` x `EQ_REGION` cells, the least-squares fit of each cell's sample `raw` as a constant plus a
/// weight for the cell's own value and one for each of its 8 neighbours' (+1 white, -1 black), over the
/// pilots (known) and the data cells (their current decisions); then a data cell takes the value its sample
/// fits better given its neighbours' current values, and its margin from how far the sample lies from the
/// threshold between them. Round 1 decides the cells the pattern decision was unsure of (`EQ_DOUBT`); round
/// 2 fits again on round 1's decisions and decides again the cells that can change (`EQ_RECHECK`). Codewords
/// read cold at 1.3 / 1.45 px per cell straight on and 1.5 px at 20 degrees of yaw: one round 74.7 / 99.2 /
/// 96.1%, two 91.8 / 99.9 / 98.8%, three 93.3 / 99.8 / 99.2% (each round after the first costs a quarter of
/// the first).
///
/// Below ~1.8 camera px per cell a cell's sample carries a good part of its neighbours (blur, and a sampling
/// point a little off the cell's centre where the alignment is a fraction of a pixel off), and how much of
/// which neighbour changes over the tile: the pattern decision's thresholds are the tile's and symmetric
/// (counts of white neighbours), this model is the region's and has a weight per neighbour, so it follows a
/// residual offset to one side or a level that changes over the tile.
///
/// `pilots`: per pilot cell its value as the white bit, None for a pilot of a block whose tag is unknown (no
/// value). Data cells past the payload are black; the payload's data cells start from `values` and `margins`
/// (the pattern decision's; palette indices, `white` the white level's), unread ones (NaN `raw`) have no
/// value, nor do marker zones. Cells of a region with too few samples to fit keep their decision.
fn equalise_bw(grid: &FlatGrid, raw: &[f32], pilots: &[(usize, Option<bool>)], white: u8, span: f32, values: &mut [u8], margins: &mut [u16]) {
    // the profile timer exists only when asked for (Instant::now panics in browser WebAssembly); its lines
    // are sub-stages of the reader's "equalise" mark
    let t0 = std::env::var_os("LB_PROFILE").map(|_| std::time::Instant::now());
    let mark = |what: &str| { if let Some(t0) = t0 { eprintln!("  [eq] {:>7.1} ms  {}", t0.elapsed().as_secs_f64() * 1000.0, what); } };
    let (cols, rows) = (grid.cols() as usize, grid.rows() as usize);
    let kinds = grid.cell_kinds();
    let n = cols * rows;
    // per cell, with a border of one cell around the grid (a cell's 8 neighbours are then plain offsets): its
    // white bit (0 where it has no value)
    let pw = cols + 2;
    let mut st = vec![0u8; pw * (rows + 2)];
    for r in 0..rows {
        let (o, x) = ((r + 1) * pw + 1, r * cols);
        for ((s, &k), &val) in st[o..o + cols].iter_mut().zip(&kinds[x..x + cols]).zip(&values[x..x + cols]) {
            *s = ((k == CELL_DATA) & (val == white)) as u8 | ((k == CELL_PAST) & (white == 0)) as u8;
        }
    }
    for &(i, b) in pilots { st[(i / cols + 1) * pw + i % cols + 1] = b.unwrap_or(false) as u8; }
    // which cells have no value, and the roles: the grid's own (built once per grid) when every sample is read
    // and every pilot's tag known, else this read's
    let nv_of = |unread: bool| -> Vec<u8> {
        let mut nv = vec![1u8; pw * (rows + 2)];
        for r in 0..rows {
            let (o, x) = ((r + 1) * pw + 1, r * cols);
            for ((v, &k), &y) in nv[o..o + cols].iter_mut().zip(&kinds[x..x + cols]).zip(&raw[x..x + cols]) {
                *v = !(((k == CELL_DATA) & !(unread & y.is_nan())) | (k == CELL_PAST) | (k == CELL_PILOT)) as u8;
            }
        }
        nv
    };
    let unread = raw.iter().fold(false, |a, v| a | v.is_nan());
    let untagged = pilots.iter().any(|p| p.1.is_none());
    let own;
    let (nv, role): (&[u8], &[u8]) = if !unread && !untagged {
        let (nv, role) = grid.eq_roles(|| { let nv = nv_of(false); let role = eq_roles(cols, rows, kinds, &nv, None); (nv, role) });
        (nv, role)
    } else {
        let mut nv = nv_of(true);
        for &(i, b) in pilots { if b.is_none() { nv[(i / cols + 1) * pw + i % cols + 1] = 1; } }
        let role = eq_roles(cols, rows, kinds, &nv, Some(raw));
        own = (nv, role);
        (&own.0, &own.1)
    };
    // the 9 cells of a pattern around padded index o, in the pattern's bit order: the cell, then its
    // neighbours on the left, right, above, below, above left, above right, below left, below right
    let at = [0isize, -1, 1, -(pw as isize), pw as isize, -(pw as isize) - 1, -(pw as isize) + 1, pw as isize - 1, pw as isize + 1];
    // every cell's pattern: its own white bit (bit 0) and its neighbours' (bits 1..=8, in `at`'s order)
    let pattern = |st: &[u8], o: usize| -> u16 { at.iter().enumerate().fold(0u16, |p, (k, &d)| p | (st[(o as isize + d) as usize] as u16) << k) };
    let mut pat = vec![0u16; n];
    crate::par::par_rows_mut(&mut pat, &mut vec![(); n], cols, |rr, pb, _| {
        for (r, o) in rr.zip(pb.chunks_exact_mut(cols)) {
            let (up, mid, dn) = (&st[r * pw..][..pw], &st[(r + 1) * pw..][..pw], &st[(r + 2) * pw..][..pw]);
            let (ul, uc, ur) = (&up[..cols], &up[1..cols + 1], &up[2..cols + 2]);
            let (ml, mc, mr) = (&mid[..cols], &mid[1..cols + 1], &mid[2..cols + 2]);
            let (dl, dc, dr) = (&dn[..cols], &dn[1..cols + 1], &dn[2..cols + 2]);
            for c in 0..cols {
                o[c] = mc[c] as u16 | (ml[c] as u16) << 1 | (mr[c] as u16) << 2 | (uc[c] as u16) << 3 | (dc[c] as u16) << 4
                    | (ul[c] as u16) << 5 | (ur[c] as u16) << 6 | (dl[c] as u16) << 7 | (dr[c] as u16) << 8;
            }
        }
    });
    mark("states + patterns");
    let (rx, ry) = (cols.div_ceil(EQ_REGION), rows.div_ceil(EQ_REGION));
    let region = |i: usize| (i / cols / EQ_REGION) * rx + (i % cols) / EQ_REGION;
    // per region the normal equations of the model's 10 weights (constant, own value, 8 neighbours) over its
    // fitting cells, from a histogram of their patterns (count and sample sum per pattern), whose
    // Walsh-Hadamard transform holds every sum they need (the features are +-1: a product of two is a parity
    // of the pattern)
    let mut eqs: Vec<([[f64; 10]; 10], [f64; 10])> = par_map(rx * ry, |q| {
        let (c0, c1) = ((q % rx) * EQ_REGION, ((q % rx + 1) * EQ_REGION).min(cols));
        let (r0, r1) = ((q / rx) * EQ_REGION, ((q / rx + 1) * EQ_REGION).min(rows));
        // counts and sample sums per pattern, interleaved (one transform for both); two tables, for the even
        // and the odd cells of a row (a cell's add then does not wait on its neighbour's when they share a
        // pattern)
        let (mut h, mut h2) = ([0f32; 1024], [0f32; 1024]);
        for r in r0..r1 {
            let (a, b) = (r * cols + c0, r * cols + c1);
            let (pr, xr, fr) = (&pat[a..b], &raw[a..b], &role[a..b]);
            // every cell adds, a cell that does not fit adds nothing (no branch to mispredict)
            let add = |h: &mut [f32; 1024], k: usize| { let f = fr[k] & 1; let p = 2 * (pr[k] as usize & 511); h[p] += f as f32; h[p + 1] += if f != 0 { xr[k] } else { 0.0 }; };
            let mut k = 0;
            while k + 1 < pr.len() { add(&mut h, k); add(&mut h2, k + 1); k += 2; }
            if k < pr.len() { add(&mut h, k); }
        }
        for (x, y) in h.iter_mut().zip(&h2) { *x += y; }
        wht512x2(&mut h);
        // feature j + 1 is s_j = +1 (bit j set) or -1: its sum over the cells is -H[1 << j], that of s_j s_k
        // (j != k) is H[(1 << j) | (1 << k)]
        let (cf, sf) = (|k: usize| h[2 * k] as f64, |k: usize| h[2 * k + 1] as f64);
        let mut a = [[0f64; 10]; 10];
        let mut y = [0f64; 10];
        a[0][0] = cf(0); y[0] = sf(0);
        for j in 0..9 {
            a[0][j + 1] = -cf(1 << j); a[j + 1][0] = a[0][j + 1];
            y[j + 1] = -sf(1 << j);
            for k in 0..9 { a[j + 1][k + 1] = if j == k { cf(0) } else { cf((1 << j) | (1 << k)) }; }
        }
        (a, y)
    });
    mark("histograms + transforms");
    // the margin scale of `pattern_decide_grid`: 3 * 255^2 * |x - t| / 2 with x on the black..white axis
    let l = 3.0 * 255.0 * 255.0f32;
    // per region, from its equations: the prediction without the cell's own term for each of the 256
    // neighbour patterns, and the margin scale; None where the region has too few samples (it keeps its
    // decisions)
    let tables = |eqs: &[([[f64; 10]; 10], [f64; 10])]| -> Vec<Option<([f32; 256], f32)>> { par_map(eqs.len(), |q| {
        let (a, y) = &eqs[q];
        // ten unknowns: a region needs several samples per weight
        if a[0][0] < 100.0 { return None; }
        let w = solve10(a, y)?;
        // a model in which white does not read clearly brighter than black has learnt nothing usable (its black
        // and white predictions 2 w1 apart: under EQ_MIN_GAP of the frame's black..white span)
        if 2.0 * w[1] <= (EQ_MIN_GAP * span.abs()) as f64 { return None; }
        // w0 + the sum of +-w_k over the neighbours (bit k of the pattern: neighbour k white), bit by bit
        let mut t = [0f32; 256];
        t[0] = (w[0] - w[2..].iter().sum::<f64>()) as f32;
        for k in 0..8 { let d = (2.0 * w[k + 2]) as f32; for q in 0..1usize << k { t[q | 1 << k] = t[q] + d; } }
        // black and white predictions lie 2 w1 apart: x - t = d / (2 w1) on the black..white axis
        Some((t, l / (4.0 * w[1] as f32)))
    }) };
    // data cell i (role f, pattern p) in table t: the white level where its sample lies above the prediction
    // without its own term, the margin from how far; a neighbour without a value (role bit 2) counts 0: the
    // mean of the predictions with it black and with it white (the prediction is linear in it)
    let black = 1 - white;
    let decide = |(t, scale): &([f32; 256], f32), i: usize, p: u16, f: u8| -> (u8, u16) {
        let set = (p >> 1) as usize & 255;
        let rest = if f & 4 == 0 { t[set] } else {
            let o = (i / cols + 1) * pw + i % cols + 1;
            let free = at[1..].iter().enumerate().fold(0usize, |a, (b, &d)| if nv[(o as isize + d) as usize] != 0 { a | 1 << b } else { a });
            (t[set & !free] + t[set | free]) / 2.0
        };
        let d = raw[i] - rest;
        (if d > 0.0 { white } else { black }, (scale * d.abs()).min(65535.0) as u16)
    };
    // round 1: the data cells the pattern decision left within EQ_DOUBT of its threshold, in their region's
    // model, given their neighbours' values before the round (from the patterns, so the order does not
    // matter); the cells whose decision changed, and those left within EQ_RECHECK of their threshold (only
    // such a cell, or one whose pattern changed, can change in round 2)
    let tab = tables(&eqs);
    mark("tables 1");
    let near = (l * EQ_RECHECK / 2.0) as u16;
    let mut changed: Vec<usize> = Vec::new();
    // the close cells' indices, written without a branch (every cell writes its index at the end of the list,
    // which moves on only for a close one) into a buffer kept by the thread: a fresh one per read would be
    // fresh memory to fault in
    thread_local! { static CLOSE: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) }; }
    let mut close = CLOSE.with(|c| std::mem::take(&mut *c.borrow_mut()));
    if close.len() < n + 1 { close.resize(n + 1, 0); }
    let mut nn = 0usize;
    let doubt = (l * EQ_DOUBT / 2.0) as u16;
    for r in 0..rows {
        for (q, table) in tab[(r / EQ_REGION) * rx..][..rx].iter().enumerate() {
            let Some((t, scale)) = table else { continue };
            let (a, b) = (r * cols + q * EQ_REGION, r * cols + ((q + 1) * EQ_REGION).min(cols));
            for (k, ((((v, m), &p), &x), &f)) in values[a..b].iter_mut().zip(&mut margins[a..b]).zip(&pat[a..b]).zip(&raw[a..b]).zip(&role[a..b]).enumerate() {
                // every cell computes, only the doubtful data cells take the result (no branch to mispredict)
                let go = (f & 2 != 0) & (*m < doubt);
                let set = (p >> 1) as usize & 255;
                let rest = if f & 4 == 0 { t[set] } else {
                    let o = (r + 1) * pw + q * EQ_REGION + k + 1;
                    let free = at[1..].iter().enumerate().fold(0usize, |a, (b, &d)| if nv[(o as isize + d) as usize] != 0 { a | 1 << b } else { a });
                    (t[set & !free] + t[set | free]) / 2.0
                };
                let d = x - rest;
                let w = d > 0.0;
                let nm = (scale * d.abs()).min(65535.0) as u16;
                *v = if go { if w { white } else { black } } else { *v };
                *m = if go { nm } else { *m };
                if go & (w as u16 != p & 1) { changed.push(a + k); }
                close[nn] = (a + k) as u32;
                nn += (go & (nm < near)) as usize;
            }
        }
    }
    mark("round 1");
    // round 2 fits again on round 1's decisions: only the patterns that changed (the changed cells' and their
    // neighbours') change the equations, each leaving them and coming back with its new bits
    for &i in &changed { st[(i / cols + 1) * pw + i % cols + 1] ^= 1; }
    // the cells whose pattern changed, once each
    let mut touched: Vec<usize> = Vec::with_capacity(changed.len() * 9);
    for &i in &changed {
        let (r, c) = (i / cols, i % cols);
        for rr in r.saturating_sub(1)..(r + 2).min(rows) { for cc in c.saturating_sub(1)..(c + 2).min(cols) { touched.push(rr * cols + cc); } }
    }
    touched.sort_unstable();
    touched.dedup();
    let feat = |p: u16| -> [f64; 10] { std::array::from_fn(|k| if k == 0 || p >> (k - 1) & 1 == 1 { 1.0 } else { -1.0 }) };
    for &j in &touched {
        let new = pattern(&st, (j / cols + 1) * pw + j % cols + 1);
        if role[j] & 1 != 0 {
            let (fo, fnw, v) = (feat(pat[j]), feat(new), raw[j] as f64);
            let (a, y) = &mut eqs[region(j)];
            // only the rows and columns of the features that changed
            let flip = (pat[j] ^ new) << 1;
            for p in (0..10).filter(|&p| flip >> p & 1 == 1) {
                y[p] += v * (fnw[p] - fo[p]);
                for q in 0..10 {
                    let d = fnw[p] * fnw[q] - fo[p] * fo[q];
                    a[p][q] += d;
                    if flip >> q & 1 == 0 { a[q][p] += d; }
                }
            }
        }
        pat[j] = new;
    }
    mark("refit");
    let tab = tables(&eqs);
    mark("tables 2");
    // the cells that can change, again in their region's new model (a cell on both lists is decided twice,
    // the same way; a cell whose pattern changed is decided whatever the pattern decision's margin: measured
    // a little better than keeping those to EQ_DOUBT)
    for i in close[..nn].iter().map(|&i| i as usize).chain(touched.iter().copied()) {
        let f = role[i];
        if f & 2 == 0 { continue; }
        let Some(t) = &tab[region(i)] else { continue };
        (values[i], margins[i]) = decide(t, i, pat[i], f);
    }
    mark("round 2");
    CLOSE.with(|c| *c.borrow_mut() = close);
}

/// The 10-90% width, in camera px, of the edges between the chrome border's alternating modules (the median over
/// the edges in view): focus and motion blur, not a blend of two sender frames (the border is the same on every
/// frame). Each edge's profile is read across it along the border, half a module each side; 0 when no edge is
/// in view.
pub fn edge_spread(y: &GrayImage, chrome: &Chrome, h: &Homography) -> f32 {
    let n = chrome.border_module_count();
    let half = super::format::MODULE_PX as f32 / 2.0;
    const K: usize = 17;
    let mut widths: Vec<f32> = Vec::new();
    for i in 0..n {
        let j = (i + 1) % n;
        let (a, b) = (chrome.border_module_center(i), chrome.border_module_center(j));
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let len = dx.hypot(dy);
        // neighbours along one side only, never round a corner
        if (len - 2.0 * half).abs() > 0.01 || (dx != 0.0 && dy != 0.0) { continue; }
        let (ux, uy) = (dx / len, dy / len);
        let (mx, my) = ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0);
        let mut prof = [0f32; K];
        let mut inside = true;
        for (k, v) in prof.iter_mut().enumerate() {
            let t = -half + 2.0 * half * k as f32 / (K - 1) as f32;
            let (cx, cy) = h.apply(mx + ux * t, my + uy * t);
            if cx < 1.0 || cy < 1.0 || cx > y.width as f32 - 2.0 || cy > y.height as f32 - 2.0 { inside = false; break; }
            *v = y.sample(cx - 0.5, cy - 0.5);
        }
        if !inside { continue; }
        // rising from black to white
        if Chrome::border_is_white(i) { prof.reverse(); }
        let (lo, hi) = ((prof[0] + prof[1]) / 2.0, (prof[K - 1] + prof[K - 2]) / 2.0);
        if hi - lo < 20.0 { continue; }
        let cross = |f: f32| -> Option<f32> {
            let t = lo + f * (hi - lo);
            (1..K).find(|&k| prof[k] >= t && prof[k - 1] < t).map(|k| (k - 1) as f32 + (t - prof[k - 1]) / (prof[k] - prof[k - 1]))
        };
        let (Some(a10), Some(a90)) = (cross(0.1), cross(0.9)) else { continue };
        // samples are 2 half / (K - 1) tile px apart; camera px per tile px along this edge
        let (p0, p1) = (h.apply(mx - ux * half, my - uy * half), h.apply(mx + ux * half, my + uy * half));
        let scale = (p1.0 - p0.0).hypot(p1.1 - p0.1) / (2.0 * half);
        widths.push((a90 - a10).max(0.0) * 2.0 * half / (K - 1) as f32 * scale);
    }
    if widths.is_empty() { return 0.0; }
    widths.sort_by(f32::total_cmp);
    widths[widths.len() / 2]
}

thread_local! { static LAST_ANCHOR_PASSES: std::cell::Cell<u32> = const { std::cell::Cell::new(0) }; }
/// The search-again passes the latest cold anchor field on this thread ran (diagnostics and tests).
pub fn last_anchor_passes() -> u32 { LAST_ANCHOR_PASSES.with(|p| p.get()) }

/// Read every data cell of a black/white `grid` from the luma plane `y` through `h`. `tag_hint` = the
/// header's frame sequence & 3; `prev_shifts` = the previous capture's per-block shifts (tracking).
/// `None` for alphabets this reader does not cover yet (the caller falls back to `read_cells`).
pub fn read_cells_luma(y: &GrayImage, grid: &FlatGrid, h: &Homography, tag_hint: u8, prev_shifts: Option<&[(f32, f32)]>) -> Option<FlatReads> {
    read_cells_luma_tracked(y, grid, h, tag_hint, prev_shifts, true, false).map(|r| r.0)
}

/// `read_cells_luma`, and whether a tracked read was read again from scratch and the re-read lost (Some(true):
/// the alignment check fired but the field was not the problem: glare, a finger). `reread`: allowed to re-read.
/// `precheck`: a capture two sender frames blend over (the tagged pilots, with the untagged ones agreeing with
/// the field: a field gone wrong reads the tagged ones like a blend too) comes back with its blend and no cells
/// (`FlatReads::values` empty), before the cell read.
pub(crate) fn read_cells_luma_tracked(y: &GrayImage, grid: &FlatGrid, h: &Homography, tag_hint: u8, prev_shifts: Option<&[(f32, f32)]>, reread: bool, precheck: bool) -> Option<(FlatReads, Option<bool>)> {
    let fmt = grid.format();
    if !fmt.alphabet.is_luma() { return None; }
    if let Some(off) = pixel_exact_offset(y, h, grid) { return Some((read_exact(y, grid, off, tag_hint), None)); }
    let (reads, off) = read_cells_luma_from(y, grid, h, tag_hint, prev_shifts, precheck);
    if reads.values.is_empty() { return Some((reads, None)); }
    // a tracked read whose untagged pilots disagree with their blocks' levels started from a field gone wrong
    // (the narrow tracking search cannot leave it): read again from scratch and keep the better of the two
    if reread && prev_shifts.is_some() && off > TRACK_OFF_MAX {
        let (again, off2) = read_cells_luma_from(y, grid, h, tag_hint, None, precheck);
        if again.values.is_empty() { return Some((again, Some(false))); }
        if off2 < off { return Some((again, Some(false))); }
        return Some((reads, Some(true)));
    }
    Some((reads, None))
}

/// A tracked read is read again from scratch when more than this share of its groups of 3 x 3 blocks have a
/// third or more of their untagged pilots (the same colour on every frame) nearer the other level of their
/// block. Measured: reads from a right field 0-0.69% (the 60 soft captures, blended ones included, the 40 of
/// test/phone_fhd2px_live, the degraded and glare gates of tests/fhd2px_floor.rs), 0-2.65% at 1 px
/// (test/phone_max1px_live); from the wrong field a manifest straddle left 11.8-12.5% (the straddle itself
/// 18-21%). A share of single pilots instead overlapped: 2.1-4.8% on degraded small soft tiles, 5.7% wrong.
/// Leaving out blocks whose levels span under half the frame's (so glare would not count) hides the wrong field:
/// its blocks' levels collapse too (11.8-21.5% -> at most 3.3%); glare stayed at 0.47% with them in.
const TRACK_OFF_MAX: f32 = 0.05;

/// The blend (`mix_of_rows`'s tighter count) from which a capture is not read before its cells, when the
/// border's edges are sharp (soft focus moves the tagged pilots between the levels too; the degraded soft
/// captures of tests/fhd2px_floor.rs read 94% yet count 1.0). Measured on the live, phase and soft phone
/// dumps (2026-11-03): clean captures 0.00-0.16, blends 0.38-0.85.
pub const BLEND_SKIP: f32 = 0.4;
/// A block row counts as blended (`FlatReads::mix`) when the other frame has more than a quarter of its light:
/// blends measure 0.49-0.59, clean captures 0.00-0.03 at 1.5-1.9 px per cell, a hard straddle cut under 0.1.
const MIX_ROW_OTHER: f32 = 0.25;
/// The tighter count for the pre-check: rows where the other frame has more than this share. A slow LCD panel
/// settles slowly, so captures carry 15-25% of the previous frame over the whole tile and read 2 of 96 while
/// the quarter rule says 0.06-0.36.
const MIX_ROW_OTHER_TIGHT: f32 = 0.18;
/// ... and only where the tagged pilots sit this much further from their level than the untagged ones of the
/// same rows: soft focus and motion blur move both alike (the degraded soft captures of tests/fhd2px_floor.rs
/// read 94% and counted 1.0 without this), a blend moves the tagged ones alone.
const MIX_ROW_OVER_BASE: f32 = 0.1;
/// Border edges narrower than this (10-90% width, camera px) are sharp: 1.48-1.53 on every phone capture
/// measured, sharp or blended; a 5 px box blur adds more than 1.5 (the app's Guidance.EDGE_SOFT).
pub const EDGE_SHARP_PX: f32 = 2.5;

/// The blend of a capture, per row of blocks: the tag-following pilots `tagged` (their colour follows the
/// frame tag's low bit) show the share of the light from the other frame as how far their sample `x` (0 black
/// .. 1 white) sits from the level the header's frame gives them (`white`). Smoothed over three rows: (the
/// share of rows over `MIX_ROW_OTHER`, the share over `MIX_ROW_OTHER_TIGHT`, the largest row share).
fn mix_of_rows(bh: usize, pilots: &[Pilot], tagged: impl Fn(&Pilot) -> bool, untagged: impl Fn(&Pilot) -> bool, x: impl Fn(&Pilot) -> Option<f32>, white: impl Fn(&Pilot) -> bool) -> (f32, f32, f32) {
    if bh == 0 { return (0.0, 0.0, 0.0); }
    // per row: (tagged other-share sum, count, untagged other-share sum, count)
    let mut row = vec![(0f32, 0u32, 0f32, 0u32); bh];
    for p in pilots {
        let (t, u) = (tagged(p), untagged(p));
        if !t && !u { continue; }
        let Some(x) = x(p) else { continue };
        let r = (p.row / BLOCK) as usize;
        if r >= bh { continue; }
        let other = if white(p) { 1.0 - x } else { x };
        let e = &mut row[r];
        if t { e.0 += other; e.1 += 1; } else { e.2 += other; e.3 += 1; }
    }
    let rows_seen = (0..bh).filter(|&j| row[j].1 > 0).count();
    let (mut mixed, mut tight, mut peak) = (0usize, 0usize, 0f32);
    for j in 0..bh {
        if row[j].1 == 0 { continue; }
        let (mut a, mut n, mut b, mut m) = (0f32, 0u32, 0f32, 0u32);
        for k in j.saturating_sub(1)..=(j + 1).min(bh - 1) { a += row[k].0; n += row[k].1; b += row[k].2; m += row[k].3; }
        if n == 0 { continue; }
        let w = a / n as f32;
        let o = w.min(1.0 - w);
        // the untagged pilots (the same on every frame) sit between the levels by blur and noise alone: a row
        // counts for the tight measure only when its tagged pilots sit clearly further out than they do
        let base = if m > 0 { let u = b / m as f32; u.min(1.0 - u) } else { 0.0 };
        peak = peak.max(o);
        if o > MIX_ROW_OTHER { mixed += 1; }
        if o > MIX_ROW_OTHER_TIGHT && o > base + MIX_ROW_OVER_BASE { tight += 1; }
    }
    (mixed as f32 / rows_seen.max(1) as f32, tight as f32 / rows_seen.max(1) as f32, peak)
}

/// `read_cells_luma` without the pixel-exact path, from `prev_shifts` when given; with the share of untagged
/// pilots nearer the other level of their block (an alignment check).
fn read_cells_luma_from(y: &GrayImage, grid: &FlatGrid, h: &Homography, tag_hint: u8, prev_shifts: Option<&[(f32, f32)]>, precheck: bool) -> (FlatReads, f32) {
    let fmt = grid.format();
    let n_lv = fmt.alphabet.size();
    // the profile timer exists only when asked for: Instant::now panics in browser WebAssembly
    let t0 = std::env::var_os("LB_PROFILE").map(|_| std::time::Instant::now());
    let mark = |what: &str| { if let Some(t0) = t0 { eprintln!("  [fast] {:>7.1} ms  {}", t0.elapsed().as_secs_f64() * 1000.0, what); } };
    let pal = fmt.alphabet.palette();
    let (cols, rows) = (grid.cols(), grid.rows());
    let s = sampling(h, grid);
    let cell = fmt.pitch as f32 * s;
    // box window: measured best 0.7 of a cell at 1 px, 0.5 above (ReadParams::tuned_for)
    let k = (cell * if fmt.pitch == 1 { 0.7 } else { 0.5 }).round().max(1.0) as u32;
    let reach = (2.5 * s).max(3.0);
    // the tile's bounding box in the image, widened by the search reach and the box
    let corners = [(0.0, 0.0), (grid.width_px() as f32, 0.0), (0.0, grid.height_px() as f32), (grid.width_px() as f32, grid.height_px() as f32)].map(|(a, b)| h.apply(a, b));
    let pad = reach + k as f32 + 2.0;
    let fold = |f: fn(f32, f32) -> f32, init: f32, sel: fn(&(f32, f32)) -> f32| corners.iter().map(sel).fold(init, f);
    let (bx0, bx1) = (fold(f32::min, f32::MAX, |c| c.0) - pad, fold(f32::max, f32::MIN, |c| c.0) + pad);
    let (by0, by1) = (fold(f32::min, f32::MAX, |c| c.1) - pad, fold(f32::max, f32::MIN, |c| c.1) + pad);
    let clampu = |v: f32, hi: u32| v.max(0.0).min(hi as f32) as u32;
    let roi = (clampu(bx0, y.width), clampu(by0, y.height), clampu(bx1, y.width), clampu(by1, y.height));
    let src = blur_roi(y, k, roi);
    mark("blur");
    let at = |cx: f32, cy: f32| -> Option<f32> { sample_at(&src, cx, cy) };

    let (bw, bh) = grid.blocks();
    let nblocks = (bw * bh) as usize;
    let kind_of = |c: u32, r: u32| PILOT_LOCAL.iter().position(|&q| q == (r % BLOCK, c % BLOCK)).unwrap_or(0);
    let mut pilots: Vec<Pilot> = grid.pilot_positions().iter().filter_map(|&(c, r)| {
        let FlatCell::Pilot { .. } = grid.cell(c, r) else { return None };
        let (x, yy) = grid.cell_center(c, r);
        let (cx, cy) = h.apply(x, yy);
        Some(Pilot { cx, cy, colour: grid.pilot_colour_at(c, r, tag_hint & 3), block: ((r / BLOCK) * bw + c / BLOCK) as usize, col: c, row: r })
    }).collect();
    // alignment uses the untagged pilots (kinds 0 and 2): their colour does not depend on the frame tag, so a
    // straddled capture aligns both halves (see read_cells)
    let mut anchor_by_block: Vec<Vec<usize>> = vec![Vec::new(); nblocks];
    for (i, p) in pilots.iter().enumerate() { if kind_of(p.col, p.row) % 2 == 0 { anchor_by_block[p.block].push(i); } }
    let neighbourhood = |b: usize| -> Vec<usize> {
        let (x, yy) = ((b as u32 % bw) as i64, (b as u32 / bw) as i64);
        let mut v = Vec::with_capacity(9);
        for j in (yy - 1).max(0)..=(yy + 1).min(bh as i64 - 1) { for i in (x - 1).max(0)..=(x + 1).min(bw as i64 - 1) { v.push((j * bw as i64 + i) as usize); } }
        v
    };
    // the luma of every palette level among the anchors at the given per-block shifts (unused entries 0)
    let levels = |shift: &[(f32, f32)]| -> Lv {
        let mut acc = [(0f64, 0u32); 8];
        for &i in anchor_by_block.iter().flatten() {
            let p = &pilots[i];
            let (dx, dy) = shift[p.block];
            if let Some(v) = at(p.cx + dx, p.cy + dy) { let a = &mut acc[p.colour as usize]; a.0 += v as f64; a.1 += 1; }
        }
        std::array::from_fn(|c| if c >= n_lv { 0.0 } else if acc[c].1 == 0 { pal[c][0] as f32 } else { (acc[c].0 / acc[c].1 as f64) as f32 })
    };
    // the alignment pilots of block b's 3x3 block neighbourhood as (x, y, level): a search's pooled cost at a
    // shift is their squared distance to their levels there (gathered once per search, not per candidate)
    let pooled = |b: usize, lv: &Lv| -> Vec<(f32, f32, f32)> {
        let (x, yy) = (b as u32 % bw, b as u32 / bw);
        let mut v = Vec::with_capacity(18);
        for j in yy.saturating_sub(1)..=(yy + 1).min(bh - 1) { for i in x.saturating_sub(1)..=(x + 1).min(bw - 1) {
            for &k in &anchor_by_block[(j * bw + i) as usize] { let p = &pilots[k]; v.push((p.cx, p.cy, lv[p.colour as usize])); }
        } }
        v
    };
    let (sw, sh) = (src.width as usize, src.height as usize);
    let best_around = |b: usize, centre: (f32, f32), step: f32, half: i32, lv: &Lv| -> Option<(f32, f32)> {
        let pts = pooled(b, lv);
        let m = (2 * half + 1) as usize;
        let d: Vec<f32> = (-half..=half).map(|i| i as f32 * step).collect();
        let (mut cost, mut cnt) = (vec![0f32; m * m], vec![0u32; m * m]);
        let (mut xs, mut ys) = (vec![(0usize, 0f32); m], vec![(0usize, 0f32); m]);
        for &(x, y, t) in &pts {
            // the candidates' pixel cells and weights once per pilot, as `at` finds them; a pilot whose window
            // leaves the image interior samples every candidate through `at`
            let mut inside = true;
            for i in 0..m {
                let (u, v) = ((x + (centre.0 + d[i])) - 0.5, (y + (centre.1 + d[i])) - 0.5);
                inside &= u >= 0.0 && v >= 0.0 && (u as usize) + 1 < sw && (v as usize) + 1 < sh;
                xs[i] = (u as usize, u - (u as usize) as f32); ys[i] = (v as usize, v - (v as usize) as f32);
            }
            for (j, &(y0, fy)) in ys.iter().enumerate() {
                for (i, &(x0, fx)) in xs.iter().enumerate() {
                    let v = if inside {
                        let (o, px) = (y0 * sw + x0, &src.data);
                        let (a, b, c, e) = (px[o] as f32, px[o + 1] as f32, px[o + sw] as f32, px[o + sw + 1] as f32);
                        let (top, bot) = (a + (b - a) * fx, c + (e - c) * fx);
                        top + (bot - top) * fy
                    } else { match at(x + (centre.0 + d[i]), y + (centre.1 + d[j])) { Some(v) => v, None => continue } };
                    cost[j * m + i] += (v - t).powi(2); cnt[j * m + i] += 1;
                }
            }
        }
        let mut best: Option<(f32, f32, f32)> = None;
        for j in 0..m { for i in 0..m {
            let n = cnt[j * m + i];
            if n == 0 { continue; }
            // ties go to the smaller move
            let (di, dj) = (i as i32 - half, j as i32 - half);
            let c = cost[j * m + i] / n as f32 + 1e-3 * ((di * di + dj * dj) as f32);
            if best.is_none_or(|bb| c < bb.2) { best = Some((centre.0 + d[i], centre.1 + d[j], c)); }
        } }
        best.map(|(dx, dy, _)| (dx, dy))
    };
    // starting levels from the tile's own pixels (2-means over a subsample of the blurred ROI): pilots at an
    // unknown shift would average black and white into the same grey and flatten every cost. Only the cold
    // anchor field needs them: a tracked capture starts from the previous field and takes its levels from the
    // pilots there (measured ~0.7 ms per Full HD 2 px capture: the sort and ten passes over 20 000 samples)
    let start_levels = || -> Lv {
        let mut px: Vec<f32> = Vec::new();
        let step = (((roi.2 - roi.0) * (roi.3 - roi.1)) as f32 / 20_000.0).sqrt().max(1.0) as usize;
        for yy in (roi.1 as usize..roi.3 as usize).step_by(step) { for xx in (roi.0 as usize..roi.2 as usize).step_by(step) { px.push(src.data[yy * src.width as usize + xx] as f32); } }
        // n-means (n = palette levels) started on the palette's spacing between the darkest and brightest
        // percent: a 2-means split of a 4-grey tile lands between levels and biases the alignment cost
        px.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (p_lo, p_hi) = (px[px.len() / 100], px[px.len() - 1 - px.len() / 100]);
        let mut lv: Lv = std::array::from_fn(|c| if c < n_lv { p_lo + (p_hi - p_lo) * pal[c][0] as f32 / 255.0 } else { 0.0 });
        for _ in 0..10 {
            let mut acc = [(0f32, 0u32); 8];
            for &v in &px {
                let k = (0..n_lv).min_by(|&a, &b| (v - lv[a]).abs().partial_cmp(&(v - lv[b]).abs()).unwrap()).unwrap();
                acc[k].0 += v; acc[k].1 += 1;
            }
            for k in 0..n_lv { if acc[k].1 > 0 { lv[k] = acc[k].0 / acc[k].1 as f32; } }
        }
        lv
    };
    let mut lv: Lv;
    let st1 = (cell / 4.0).clamp(0.2, 1.5);
    let st2 = st1 / 2.0;
    // coarse alignment, a robust field from independent searches (no search waits on a chain of others: each
    // anchor searches alone, each lattice block once from the anchors' field, so one bad region cannot carry
    // its error onward):
    // 1. the tile's shift: every anchor block's (every 4th block in each direction, and the last row and
    //    column) costs over the full reach, summed, and the best of the sum;
    // 2. each anchor searches a disk of 1.75 cells around that shift on its own, then refines with one 3x3
    //    search at st2; an anchor whose best lies on its disk's edge or away from its neighbours' median
    //    searches again around that median, until nothing changes (regional offsets beyond one disk);
    // 3. an anchor more than 1.5 st1 from the median of its 3x3 anchor neighbourhood in either axis (glare, a
    //    finger, a flat region) takes that median, one pass;
    // 4. every block's prediction is the bilinear interpolation of the anchor field;
    // 5. every other block in each direction refines its prediction with one 3x3 search at st1; the blocks in
    //    between take their solved neighbours' mean.
    // The pooled refinement below does the fine work for every block in parallel.
    let prev = prev_shifts.filter(|p| p.len() == nblocks);
    // tracking: the previous capture's field is the start (the anchor field and its smoothing are skipped)
    let mut shift = if let Some(p) = prev { p.to_vec() } else {
        lv = start_levels();
        let axis = |n: u32| -> Vec<u32> { let mut v: Vec<u32> = (0..n).step_by(4).collect(); if v.last() != Some(&(n - 1)) { v.push(n - 1); } v };
        let (ax, ay) = (axis(bw), axis(bh));
        let (aw, ah) = (ax.len(), ay.len());
        let anchor_block = |a: usize| (ay[a / aw] * bw + ax[a % aw]) as usize;
        // the untagged pilots repeat 4 cells along a row on frames whose tag bits are clear (each matches the
        // tagged pilot 4 cells to its left), so an anchor searches a disk of at most 1.75 cells around the
        // tile's shift: the repeat never lies inside one search (measured on real 1 px captures: anchors
        // searching the full 2.5-cell reach locked whole regions onto the repeat, 0.70% cell errors on capture
        // 1 against 0.40%)
        let rad = reach.min(1.75 * cell);
        // the search on whole camera pixels: a pilot's bilinear weights are then the same at every candidate, so
        // its costs are two passes over a patch of the blurred plane. Whole pixels are at most half a cell from
        // 2 camera px per cell up; below that (and for pilots too near the image edge) the search runs at st1.
        // The costs of every whole-pixel shift within r px of the whole-pixel centre (cx, cy) for block b's
        // pooled pilots, row-major over a (2r + 1)^2 grid, and how many pilots they sum
        let grid_at = |b: usize, (cx, cy): (f32, f32), r: usize| -> (Vec<f32>, u32) {
            let side = 2 * r + 1;
            let (mut acc, mut hrow, mut n) = (vec![0f32; side * side], vec![0f32; (side + 1) * side], 0u32);
            for (x, y, t) in pooled(b, &lv) {
                let (px, py) = (x - 0.5 + cx, y - 0.5 + cy);
                // only pilots whose every candidate lies inside the image: every candidate sums the same set
                if px < r as f32 || py < r as f32 || px as usize + r + 2 > sw || py as usize + r + 2 > sh { continue; }
                let (x0, y0) = (px as usize, py as usize);
                let (fx, fy) = (px - x0 as f32, py - y0 as f32);
                for (rj, line) in hrow.chunks_exact_mut(side).enumerate() {
                    let row = &src.data[(y0 - r + rj) * sw + x0 - r..][..side + 1];
                    for ((o, &a), &c) in line.iter_mut().zip(row).zip(&row[1..]) { *o = a as f32 + (c as f32 - a as f32) * fx; }
                }
                // candidate row j blends horizontal rows j and j + 1: one pass over the whole grid
                for ((o, &a), &c) in acc.iter_mut().zip(&hrow).zip(&hrow[side..]) { let v = a + (c - a) * fy - t; *o += v * v; }
                n += 1;
            }
            (acc, n)
        };
        // the best of such a grid within `lim` px of its centre (ties go to the smaller move from it)
        let best_in = |acc: &[f32], n: u32, r: usize, lim: f32| -> (f32, f32) {
            let side = 2 * r + 1;
            let mut best = (0f32, 0f32, f32::MAX);
            for (q, &c) in acc.iter().enumerate() {
                let (dx, dy) = ((q % side) as f32 - r as f32, (q / side) as f32 - r as f32);
                if dx * dx + dy * dy > lim * lim { continue; }
                let c = c / n as f32 + 1e-3 * (dx * dx + dy * dy);
                if c < best.2 { best = (dx, dy, c); }
            }
            (best.0, best.1)
        };
        let r = rad.ceil() as usize;
        // around a whole-pixel centre c; also whether the best lies in the outer pixel of the disk
        let whole = |b: usize, c: (f32, f32)| -> Option<((f32, f32), bool)> {
            let (cx, cy) = (c.0.round(), c.1.round());
            let (acc, n) = grid_at(b, (cx, cy), r);
            if n == 0 { return None; }
            let best = best_in(&acc, n, r, rad);
            Some(((cx + best.0, cy + best.1), best.0.hypot(best.1) > rad - 1.0))
        };
        // the disks' common centre, from the whole tile: every anchor's costs over the full reach around no
        // shift, summed (the grids share one layout), and the best shift of the sum. The repeat matches only
        // where a pilot's neighbour 4 cells along carries its colour, so the tile's thousands of pilots pick
        // the true shift over it; each disk then holds the true shift and not its repeat (measured on real
        // capture 0 read through its homography moved 2.3 cells along the rows: 9.01% cell errors with every
        // disk around no shift, 0.543% around the tile's shift; clean 0.526%; 2.5 ms)
        let rf = reach.ceil() as usize;
        let centre = {
            let grids = par_map(aw * ah, |a| grid_at(anchor_block(a), (0.0, 0.0), rf));
            let mut sum = vec![0f32; (2 * rf + 1) * (2 * rf + 1)];
            let mut n = 0u32;
            for (acc, k) in &grids { if *k > 0 { for (s, v) in sum.iter_mut().zip(acc) { *s += v; } n += k; } }
            if n == 0 { (0.0, 0.0) } else { best_in(&sum, n, rf, reach) }
        };
        let whole_px = cell >= 2.0;
        let full = (rad / st1).round() as i32;
        // straight from whole pixels to st2 (measured on real 1 px captures: a st1 search between them gave
        // 0.538, 0.395, 0.514, 0.585% cell errors against 0.527, 0.392, 0.515, 0.586%)
        let search = |a: usize, c: (f32, f32)| -> Option<((f32, f32), bool)> {
            let b = anchor_block(a);
            let (s, edge) = match if whole_px { whole(b, c) } else { None } { Some(w) => w, None => { let s = best_around(b, c, st1, full, &lv)?; (s, (s.0 - c.0).hypot(s.1 - c.1) > rad - st1) } };
            Some((best_around(b, s, st2, 1, &lv).unwrap_or(s), edge))
        };
        let med = |v: &mut Vec<f32>| -> f32 { v.sort_by(f32::total_cmp); let n = v.len(); if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 } };
        let med2 = |pts: &mut dyn Iterator<Item = (f32, f32)>| -> Option<(f32, f32)> { let (mut xs, mut ys): (Vec<f32>, Vec<f32>) = pts.unzip(); if xs.is_empty() { None } else { Some((med(&mut xs), med(&mut ys))) } };
        let around = |found: &[Option<((f32, f32), bool)>], a: usize| -> Option<(f32, f32)> {
            let (i, j) = (a % aw, a / aw);
            med2(&mut (j.saturating_sub(1)..=(j + 1).min(ah - 1)).flat_map(|jj| (i.saturating_sub(1)..=(i + 1).min(aw - 1)).filter_map(move |ii| found[jj * aw + ii].map(|f| f.0))))
        };
        // first every anchor searches around the tile's shift; then, until nothing changes (at most 12 passes),
        // every anchor whose best lay on its search's outer pixel or that sits more than 1.5 st1 from its 3x3
        // anchor neighbourhood's median searches again around that median. A regional offset beyond one search
        // from the tile's shift (a bent hand-held capture) is followed in steps from where it is small, and
        // every search stays within 1.75 cells of its neighbours' field, clear of the repeat (measured on real
        // capture 0 bent 3 cells at the centre: 2.61% cell errors with the first pass alone, 0.514% once
        // settled; clean 0.526%)
        let mut found: Vec<Option<((f32, f32), bool)>> = par_map(aw * ah, |a| search(a, centre));
        // ... and stops once the passes stop converging: from the third on, a pass that changes more than three
        // quarters as many anchors as the one before. A field that settles shrinks fast (the capture bent 3
        // cells: 100, 19, 11, 0 of 1056; clean ones 3-27 then 0); a capture too soft to read never agrees (160,
        // 93, 68, 59, 47 ... of 272 for all 12 passes, 50-65 ms of a 150-200 ms failed read on a phone sender
        // nearer than the lens focuses), and one or two anchors flipping between two answers forever cost a
        // pass each
        let mut changed_before = usize::MAX;
        let mut passes = 0u32;
        for pass in 1..12 {
            let prevf = &found;
            let again: Vec<Option<((f32, f32), bool)>> = par_map(aw * ah, |a| {
                let m = around(prevf, a);
                let stands = match (prevf[a], m) { (Some((s, false)), Some(m)) => (s.0 - m.0).abs() <= 1.5 * st1 && (s.1 - m.1).abs() <= 1.5 * st1, (Some((_, false)), None) => true, _ => false };
                match m { Some(m) if !stands => search(a, m).or(prevf[a]), _ => prevf[a] }
            });
            let changed = again.iter().zip(prevf).filter(|(x, y)| x.map(|v| v.0) != y.map(|v| v.0)).count();
            found = again;
            passes = pass;
            if changed == 0 || (pass >= 2 && changed * 4 > changed_before * 3) { break; }
            changed_before = changed;
        }
        LAST_ANCHOR_PASSES.with(|p| p.set(passes));
        let found: Vec<Option<(f32, f32)>> = found.into_iter().map(|f| f.map(|f| f.0)).collect();
        // robust rejection against the searches' own values: an anchor with nothing to read also takes its
        // neighbourhood's median; one with no readable neighbour at all, the median of every anchor (none: 0)
        let found = &found;
        let all = med2(&mut found.iter().flatten().copied()).unwrap_or((0.0, 0.0));
        let tol = 1.5 * st1;
        let (anchor, kept): (Vec<(f32, f32)>, Vec<bool>) = (0..aw * ah).map(|a| {
            let (i, j) = (a % aw, a / aw);
            let mut around = (j.saturating_sub(1)..=(j + 1).min(ah - 1)).flat_map(|jj| (i.saturating_sub(1)..=(i + 1).min(aw - 1)).filter_map(move |ii| found[jj * aw + ii]));
            let m = med2(&mut around).unwrap_or(all);
            match found[a] { Some(s) if (s.0 - m.0).abs() <= tol && (s.1 - m.1).abs() <= tol => (s, true), _ => (m, false) }
        }).unzip();
        // per block column (row): the anchor at or before it and the weight of the next one, by block index
        let span = |ax: &[u32], n: u32| -> Vec<(usize, f32)> { (0..n).map(|x| {
            if ax.len() < 2 { return (0, 0.0); }
            let i = ax.partition_point(|&a| a <= x).saturating_sub(1).min(ax.len() - 2);
            (i, (x - ax[i]) as f32 / (ax[i + 1] - ax[i]) as f32)
        }).collect() };
        let (spx, spy) = (span(&ax, bw), span(&ay, bh));
        let mut shift: Vec<(f32, f32)> = (0..nblocks).map(|b| {
            let ((i, u), (j, v)) = (spx[b % bw as usize], spy[b / bw as usize]);
            let (i1, j1) = ((i + 1).min(aw - 1), (j + 1).min(ah - 1));
            let (a, c, d, e) = (anchor[j * aw + i], anchor[j * aw + i1], anchor[j1 * aw + i], anchor[j1 * aw + i1]);
            let (top, bot) = ((a.0 + (c.0 - a.0) * u, a.1 + (c.1 - a.1) * u), (d.0 + (e.0 - d.0) * u, d.1 + (e.1 - d.1) * u));
            (top.0 + (bot.0 - top.0) * v, top.1 + (bot.1 - top.1) * v)
        }).collect();
        // anchors whose own search stood keep it (finer than a st1 step around itself); the rest of the lattice
        // refines its prediction
        let mut solved = vec![false; nblocks];
        for a in 0..aw * ah { if kept[a] { solved[anchor_block(a)] = true; } }
        let todo: Vec<usize> = (0..nblocks).filter(|&b| !solved[b] && (b as u32 % bw) % 2 == 0 && (b as u32 / bw) % 2 == 0).collect();
        let lat: Vec<(f32, f32)> = par_map(todo.len(), |q| { let b = todo[q]; best_around(b, shift[b], st1, 1, &lv).unwrap_or(shift[b]) });
        for (&b, s) in todo.iter().zip(lat) { shift[b] = s; solved[b] = true; }
        for b in 0..nblocks {
            if solved[b] { continue; }
            let (x, yy) = (b as u32 % bw, b as u32 / bw);
            let (mut sx, mut sy, mut m) = (0f32, 0f32, 0f32);
            for j in yy.saturating_sub(1)..=(yy + 1).min(bh - 1) { for i in x.saturating_sub(1)..=(x + 1).min(bw - 1) {
                let a = (j * bw + i) as usize; if solved[a] { sx += shift[a].0; sy += shift[a].1; m += 1.0; }
            } }
            if m > 0.0 { shift[b] = (sx / m, sy / m); }
        }
        shift
    };
    mark("anchor field");
    // levels at the aligned positions, then one parallel smoothing pass: every block re-searches around the
    // mean of its neighbourhood's shifts (one lattice search in a glare block can land off)
    lv = levels(&shift);
    if prev.is_none() { shift = par_map(nblocks, |b| {
        let nbh = neighbourhood(b);
        let m = nbh.len() as f32;
        let mean = (nbh.iter().map(|&n| shift[n].0).sum::<f32>() / m, nbh.iter().map(|&n| shift[n].1).sum::<f32>() / m);
        // the mean alone: the pooled refinement below searches (measured on real 1 px captures: the same
        // cell errors without a search here, 34 ms less)
        mean
    }); }
    mark("smooth");
    // frame tags from the tagged pilots (kinds 1 and 3)
    let mut tcost = vec![[0f32; 4]; nblocks];
    let mut tagged = vec![0u8; nblocks];
    for p in &pilots {
        let kk = kind_of(p.col, p.row);
        if kk != 1 && kk != 3 { continue; }
        let (dx, dy) = shift[p.block];
        let Some(v) = at(p.cx + dx, p.cy + dy) else { continue };
        for t in 0..4u8 { tcost[p.block][t as usize] += (v - lv[grid.pilot_colour_at(p.col, p.row, t) as usize]).powi(2); }
        tagged[p.block] += 1;
    }
    for b in 0..nblocks { if tagged[b] < 2 { tcost[b] = [0.0; 4]; } }
    let mut block_tag = assign_tags(&tcost, &tagged, bh as usize, bw as usize);
    for p in pilots.iter_mut() { let t = block_tag[p.block]; if tag_known(t) { p.colour = grid.pilot_colour_at(p.col, p.row, t); } }
    // refinement with every pilot (tags known) pooled over a wider block neighbourhood: 7x7 blocks at 1 px,
    // as read_cells' tuning (measured on real 1 px capture 0: the anchor field without this refinement 0.874%
    // cell errors, with it pooled over 3x3 blocks 0.631%, over 7x7 0.526%), 3x3 above; two rounds at a
    // halving step
    let mut all_by_block: Vec<Vec<usize>> = vec![Vec::new(); nblocks];
    for (i, p) in pilots.iter().enumerate() { if tag_known(block_tag[p.block]) { all_by_block[p.block].push(i); } }
    // 7x7 at 1 px: 5x5 measured +0.02 pt cell errors on the real captures for ~20 ms
    let pool: i64 = if fmt.pitch == 1 { 3 } else { 1 };
    // a 5x5 candidate window from 2 px up (neighbours a lattice step apart at most) measured on the degraded
    // Full HD 2 px captures: 1.5 px per cell soft, cold 93.7 -> 88.7% of codewords, tracked 99.3 -> 97.3%;
    // 1.45 px straight on cold 99.9 -> 99.4% (for ~1.4 ms of the 2.8 ms refinement): not taken
    // two rounds: one alone leaves 0.02-0.03 points more cell errors on the real 1 px captures
    // tracking needs only the fine round (the field moves little between captures)
    for round in (if prev.is_some() { 1 } else { 0 })..2 {
        let step = st2 / (1 << round) as f32;
        // every block's own pilots on a common lattice (multiples of `step`), in a 7x7 window around its
        // shift: the pooled cost of a shift is then a sum of the neighbours' entries, not a re-sampling
        let base: Vec<(i32, i32)> = shift.iter().map(|&(x, y)| ((x / step).round() as i32, (y / step).round() as i32)).collect();
        let own: Vec<[(f32, u32); 49]> = par_map(nblocks, |b| {
            let off: [(f32, f32); 7] = std::array::from_fn(|i| ((base[b].0 + i as i32 - 3) as f32 * step, (base[b].1 + i as i32 - 3) as f32 * step));
            pilot_window(&src, all_by_block[b].iter().map(|&k| { let p = &pilots[k]; (p.cx, p.cy, lv[p.colour as usize]) }), &off)
        });
        shift = par_map(nblocks, |b| {
            let centre = shift[b];
            let (x, yy) = ((b as u32 % bw) as i64, (b as u32 / bw) as i64);
            // the nine candidates' sums in one pass over the neighbours (each candidate still sums them in the
            // same order)
            let (mut cs, mut ns) = ([0f32; 9], [0u32; 9]);
            for ny in (yy - pool).max(0)..=(yy + pool).min(bh as i64 - 1) { for nx in (x - pool).max(0)..=(x + pool).min(bw as i64 - 1) {
                let nb = (ny * bw as i64 + nx) as usize;
                // only neighbours whose window holds all nine candidates: every candidate is summed
                // over the same set (a partial set biases the choice toward the fuller sums)
                let (di, dj) = (base[b].0 - base[nb].0, base[b].1 - base[nb].1);
                if di.abs() > 2 || dj.abs() > 2 { continue; }
                // candidate (i, j) of b is entry (i + di, j + dj) of nb's window
                let e = &own[nb];
                for j in 0..3 {
                    let row = &e[((j as i32 + dj + 2) * 7 + di + 2) as usize..][..3];
                    for (i, q) in row.iter().enumerate() { cs[j * 3 + i] += q.0; ns[j * 3 + i] += q.1; }
                }
            } }
            let mut best: Option<(f32, f32, f32)> = None;
            for j in -1..=1i32 { for i in -1..=1i32 {
                let (lx, ly) = (base[b].0 + i, base[b].1 + j);
                let q = ((j + 1) * 3 + i + 1) as usize;
                let (c, n) = (cs[q], ns[q]);
                if n == 0 { continue; }
                let c = c / n as f32 + 1e-3 * ((i * i + j * j) as f32);
                let (dx, dy) = (lx as f32 * step, ly as f32 * step);
                if best.is_none_or(|bb| c < bb.2) { best = Some((dx, dy, c)); }
            } }
            best.map(|(dx, dy, _)| (dx, dy)).unwrap_or(centre)
        });
    }
    mark("refine");
    // the blend pre-check: the kind-1 pilots (their colour follows the frame tag's low bit) sampled at the refined
    // shifts against the frame's levels, before any cell is read. A capture two sender frames share the light
    // over (`MIX_SKIP` and above: blends measure 0.49-0.59, clean captures 0.00-0.03, a hard straddle cut under
    // 0.1) cannot read: it goes back with its blend and no cells, and the stages after this one are not paid for
    // (a 60 fps camera's blended half of the captures costs it little)
    // per-block levels from the 3x3 neighbourhood's pilots (all kinds, tags known)
    // every block's pilots in pilot order, as ranges of one list (a counting sort, not a list per block)
    let (blk_start, blk_list) = {
        let mut start = vec![0usize; nblocks + 1];
        for p in &pilots { start[p.block + 1] += 1; }
        for b in 0..nblocks { start[b + 1] += start[b]; }
        let (mut fill, mut list) = (start.clone(), vec![0usize; pilots.len()]);
        for (i, p) in pilots.iter().enumerate() { list[fill[p.block]] = i; fill[p.block] += 1; }
        (start, list)
    };
    // the pilots of block b's 3x3 block neighbourhood, block by block in `neighbourhood`'s order
    let around = |b: usize, f: &mut dyn FnMut(usize)| {
        let (x, yy) = (b as u32 % bw, b as u32 / bw);
        for j in yy.saturating_sub(1)..=(yy + 1).min(bh - 1) { for i in x.saturating_sub(1)..=(x + 1).min(bw - 1) {
            let nb = (j * bw + i) as usize;
            for &k in &blk_list[blk_start[nb]..blk_start[nb + 1]] { f(k); }
        } }
    };
    // per block, the levels from its 3x3 neighbourhood's pilot samples `pv` (in pilot order); a level with fewer
    // than 3 pilots around keeps the frame's
    let block_levels = |pv: &[Option<f32>]| -> Vec<Lv> { par_map(nblocks, |b| {
        let mut acc = [(0f32, 0u32); 8];
        around(b, &mut |i| if let Some(v) = pv[i] { let c = pilots[i].colour as usize; if c < n_lv { acc[c].0 += v; acc[c].1 += 1; } });
        std::array::from_fn(|c| if c >= n_lv { 0.0 } else if acc[c].1 < 3 { lv[c] } else { acc[c].0 / acc[c].1 as f32 })
    }) };
    // a capture that straddles the manifest (blocks whose pilots fit no tag) is not pre-checked: its data half
    // reads on its own, the other half goes on to the colour path
    let known = block_tag.iter().filter(|&&t| tag_known(t)).count();
    if precheck && n_lv == 2 && known * 10 >= nblocks * 9 {
        let white = super::read::white_of(pal) as usize;
        // every pilot sampled at its block's shift, and the levels of its block's neighbourhood (lighting changes
        // over the tile: the frame's levels would read a dark corner as a blend)
        let pv: Vec<Option<f32>> = par_map(pilots.len(), |i| { let p = &pilots[i]; let (dx, dy) = shift[p.block]; at(p.cx + dx, p.cy + dy) });
        let blv = block_levels(&pv);
        let xs: Vec<Option<f32>> = pilots.iter().zip(&pv).map(|(p, v)| {
            if !tag_known(block_tag[p.block]) { return None; }
            let v = (*v)?;
            let l = blv[p.block];
            let span = l[white] - l[1 - white];
            if span.abs() < 1.0 { None } else { Some(((v - l[1 - white]) / span).clamp(0.0, 1.0)) }
        }).collect();
        // the untagged pilots (the same on every frame) nearer the other level: a field gone wrong, not a blend
        let (mut wrong, mut seen) = (0u32, 0u32);
        for (p, x) in pilots.iter().zip(&xs) {
            if kind_of(p.col, p.row) % 2 != 0 { continue; }
            let Some(x) = *x else { continue };
            seen += 1;
            if (x > 0.5) != (grid.pilot_colour_at(p.col, p.row, tag_hint & 3) as usize == white) { wrong += 1; }
        }
        let field_ok = seen >= 20 && (wrong as f32) < 0.2 * seen as f32;
        let base = pilots.as_ptr() as usize;
        let (mix_pre, tight_pre, peak_pre) = if !field_ok { (0.0, 0.0, 0.0) } else {
            mix_of_rows(bh as usize, &pilots, |p| kind_of(p.col, p.row) == 1, |p| kind_of(p.col, p.row) % 2 == 0, |p| xs[(p as *const Pilot as usize - base) / std::mem::size_of::<Pilot>()],
                |p| grid.pilot_colour_at(p.col, p.row, tag_hint & 3) as usize == white)
        };
        if std::env::var_os("LB_BLEND_LOG").is_some() { eprintln!("  [blend] mix {mix_pre:.3} tight {tight_pre:.3} peak {peak_pre:.3} field_ok {field_ok} wrong {wrong}/{seen}"); }
        if tight_pre >= BLEND_SKIP {
            mark("blend pre-check: skipped");
            return (FlatReads { cols, rows, values: Vec::new(), margins: Vec::new(), shifts: shift, alpha: 0.0, block_tag, mix: mix_pre, edge_px: 0.0 }, 0.0);
        }
    }

    // every cell sampled once at its interpolated shift (NaN outside the image), row by row (`sample_row`):
    // the shift's interpolation along v is done once per row and block column; the homography's terms in x
    // (and the block column's weight along u) once per column
    let hm = h.m;
    // per column: m0 x, m3 x, m6 x and the weight along u of the block column to the right; apart, so a row's
    // coordinates are one vectorisable pass
    let ncol = cols as usize;
    let (mut hx, mut fus) = ([vec![0f32; ncol], vec![0f32; ncol], vec![0f32; ncol]], vec![0f32; ncol]);
    // runs of columns between the same two block columns: (first column, end, left block, right block)
    let mut runs: Vec<(usize, usize, usize, usize)> = Vec::new();
    for c in 0..cols {
        let (x, _) = grid.cell_center(c, 0);
        let cu = c as usize;
        (hx[0][cu], hx[1][cu], hx[2][cu]) = (hm[0] * x, hm[3] * x, hm[6] * x);
        let u = ((c as f32 + 0.5) / BLOCK as f32 - 0.5).clamp(0.0, (bw - 1) as f32);
        let i = (u.floor() as u32).min(bw - 1);
        fus[cu] = u - i as f32;
        let (i0, i1) = (i as usize, (i + 1).min(bw - 1) as usize);
        match runs.last_mut() { Some(r) if r.2 == i0 && r.3 == i1 => r.1 = cu + 1, _ => runs.push((cu, cu + 1, i0, i1)) }
    }
    let parts: Vec<Vec<f32>> = par_ranges(rows as usize, |rr| {
        let mut out = vec![0f32; rr.len() * ncol];
        let mut row_shift = vec![(0f32, 0f32); bw as usize];
        // per column of the row: the left block's shift and the step to the right one's
        let (mut s0x, mut dsx, mut s0y, mut dsy) = (vec![0f32; ncol], vec![0f32; ncol], vec![0f32; ncol], vec![0f32; ncol]);
        let mut scratch = RowScratch::default();
        for (line, r) in out.chunks_exact_mut(ncol).zip(rr) {
            let r = r as u32;
            let v = ((r as f32 + 0.5) / BLOCK as f32 - 0.5).clamp(0.0, (bh - 1) as f32);
            let j = (v.floor() as u32).min(bh - 1);
            let (j1, fv) = ((j + 1).min(bh - 1), v - j as f32);
            for i in 0..bw { let (a, b) = (shift[(j * bw + i) as usize], shift[(j1 * bw + i) as usize]); row_shift[i as usize] = (a.0 + (b.0 - a.0) * fv, a.1 + (b.1 - a.1) * fv); }
            for &(c0, c1, i0, i1) in &runs {
                let (s0, s1) = (row_shift[i0], row_shift[i1]);
                s0x[c0..c1].fill(s0.0); dsx[c0..c1].fill(s1.0 - s0.0); s0y[c0..c1].fill(s0.1); dsy[c0..c1].fill(s1.1 - s0.1);
            }
            let (_, yy) = grid.cell_center(0, r);
            let cols_in = RowCols { hx: [&hx[0], &hx[1], &hx[2]], fu: &fus, s0x: &s0x, dsx: &dsx, s0y: &s0y, dsy: &dsy };
            sample_row(&src, &hm, yy, &cols_in, &mut scratch, line);
        }
        out
    });
    let raw: Vec<f32> = if parts.len() == 1 { parts.into_iter().next().unwrap() } else { parts.concat() };
    mark_flat_blocks(&mut block_tag, &raw, cols, rows, bw);
    let raw_at = |c: u32, r: u32| -> Option<f32> { let v = raw[(r * cols + c) as usize]; if v.is_nan() { None } else { Some(v) } };
    let pilot_v: Vec<Option<f32>> = par_map(pilots.len(), |i| raw_at(pilots[i].col, pilots[i].row));
    let block_lv: Vec<Lv> = block_levels(&pilot_v);
    // position on the black -> white axis between the block's own levels
    let axis_at = |c: u32, r: u32| -> Option<f32> {
        let v = raw_at(c, r)?;
        let l = block_lv[((r / BLOCK) * bw + c / BLOCK) as usize];
        let span = l[1] - l[0];
        Some(if span.abs() < 1.0 { 0.5 } else { (v - l[0]) / span })
    };
    mark("tags + levels");
    // the alignment check (b/w: grey levels lie closer, not measured): per group of 3 x 3 blocks, its untagged
    // pilots (kinds 0 and 2) nearer the other level of their block; `off` = the share of groups where a third
    // or more are (noise spreads thin over every group; a wrong field takes whole groups)
    let (gw, gh) = (bw.div_ceil(3) as usize, bh.div_ceil(3) as usize);
    let mut groups = vec![(0u32, 0u32); gw * gh];
    for (p, v) in pilots.iter().zip(&pilot_v).filter(|_| n_lv == 2) {
        if kind_of(p.col, p.row) % 2 != 0 || (p.colour as usize) >= n_lv { continue; }
        let Some(v) = *v else { continue };
        let l = block_lv[p.block];
        let near = (0..n_lv).min_by(|&a, &b| (v - l[a]).abs().total_cmp(&(v - l[b]).abs())).unwrap_or(0);
        let g = &mut groups[(p.block / bw as usize / 3) * gw + (p.block % bw as usize) / 3];
        g.0 += 1;
        g.1 += (near != p.colour as usize) as u32;
    }
    // the blend (b/w): pilots of kind 1 change colour between consecutive frames (the tag's low bit), so where
    // their sample sits between their block's levels is the share of the light from the header's frame; per row
    // of blocks, smoothed over three, the rows where the other frame has more than a quarter
    let mix = if n_lv != 2 { 0.0 } else {
        let white = super::read::white_of(pal) as usize;
        // the pilots' samples by index (pilot_v is in pilot order)
        let xs: Vec<Option<f32>> = pilots.iter().zip(&pilot_v).map(|(p, v)| {
            let v = (*v)?;
            let l = block_lv[p.block];
            let span = l[white] - l[1 - white];
            if span.abs() < 1.0 { None } else { Some(((v - l[1 - white]) / span).clamp(0.0, 1.0)) }
        }).collect();
        let base = pilots.as_ptr() as usize;
        mix_of_rows(bh as usize, &pilots, |p| kind_of(p.col, p.row) == 1, |p| kind_of(p.col, p.row) % 2 == 0, |p| xs[(p as *const Pilot as usize - base) / std::mem::size_of::<Pilot>()],
            |p| grid.pilot_colour_at(p.col, p.row, tag_hint & 3) as usize == white).0
    };
    let counted = groups.iter().filter(|g| g.0 >= 6).count();
    let off = groups.iter().filter(|g| g.0 >= 6 && 3 * g.1 >= g.0).count() as f32 / counted.max(1) as f32;
    let used = grid.used_data_cells();
    let mut values = vec![0u8; (cols * rows) as usize];
    let mut margins = vec![0u16; (cols * rows) as usize];
    if n_lv == 2 {
        // every used data cell's position x on the black -> white axis (`axis_at`; NaN: unread) and whether
        // its first decision (x > 0.5) is the white level, in place on the grid, in bands of rows: the pattern
        // decision reads both from the grid and decides every read cell again (its value and margin; an
        // unread cell keeps 0, 0), so the first decision's own value and margin are never kept. (A list of
        // cells, flattened from per-chunk lists and scattered, was most of this stage's time.)
        let white = super::read::white_of(pal);
        let ncell = (cols * rows) as usize;
        let (mut xg, mut first) = (vec![f32::NAN; ncell], vec![0u8; ncell]);
        crate::par::par_rows_mut(&mut xg, &mut first, ncol, |rr, xb, fb| {
            for (r, (xr, fr)) in rr.zip(xb.chunks_exact_mut(ncol).zip(fb.chunks_exact_mut(ncol))) {
                let r = r as u32;
                let raw_row = &raw[r as usize * ncol..][..ncol];
                let lv_row = &block_lv[(r / BLOCK * bw) as usize..][..bw as usize];
                for (c, ((x_out, f_out), &v)) in xr.iter_mut().zip(fr.iter_mut()).zip(raw_row).enumerate() {
                    let FlatCell::Data { index } = grid.cell(c as u32, r) else { continue };
                    if index as usize >= used { continue; }
                    let x = if v.is_nan() { f32::NAN } else {
                        let l = lv_row[c / BLOCK as usize];
                        let span = l[1] - l[0];
                        if span.abs() < 1.0 { 0.5 } else { (v - l[0]) / span }
                    };
                    *x_out = x;
                    *f_out = ((x > 0.5) as u8 == white) as u8;
                }
            }
        });
        mark("cells");
        // the thresholds are the whole tile's: a pilot whose block levels span less than half the frame's (glare,
        // a finger: black and white read alike) does not vote (measured on real capture 0 under a 480 px white
        // square at the centre: 0.847% cell errors outside it with every pilot voting, 0.577% without those,
        // against 0.560% for the clean read of the same cells)
        let span = lv[1] - lv[0];
        pattern_decide_grid(grid, pal, &pilots, &first, &xg, &mut values, &mut margins, |p: &Pilot| { let l = block_lv[p.block]; if (l[1] - l[0]) * span.signum() < 0.5 * span.abs() { None } else { axis_at(p.col, p.row) } });
        mark("pattern decision");
        // at every cell size from 2 px up: measured on real captures degraded to 2.3-3.6 camera px per cell (4
        // sources), without it 98.2-100% of codewords and 0.21-0.28% cells wrong, with it 100% and 0.05-0.08%
        if fmt.pitch >= 2 {
            // the pilots' values where their block's tag is known
            let known: Vec<(usize, Option<bool>)> = pilots.iter().map(|p| ((p.row * cols + p.col) as usize, tag_known(block_tag[p.block]).then_some(p.colour == white))).collect();
            equalise_bw(grid, &raw, &known, white, lv[1] - lv[0], &mut values, &mut margins);
            mark("equalise");
        }
    } else {
        // grey: neighbouring cells bleed into each other (blur), and grey levels are close: equalise every
        // sample, v + alpha * (v - mean of its 4 edge neighbours), with alpha chosen by how many pilots then
        // land on their own level; then the nearest of the block's (equalised) levels
        let raw_g = raw_at;
        let eq_at = |c: u32, r: u32, alpha: f32| -> Option<f32> {
            let v = raw_g(c, r)?;
            if alpha == 0.0 { return Some(v); }
            let (mut acc, mut n) = (0f32, 0f32);
            let mut push = |cc: u32, rr: u32| { if let Some(u) = raw_g(cc, rr) { acc += u; n += 1.0; } };
            if c > 0 { push(c - 1, r); } if c + 1 < cols { push(c + 1, r); } if r > 0 { push(c, r - 1); } if r + 1 < rows { push(c, r + 1); }
            Some(if n == 0.0 { v } else { v + alpha * (v - acc / n) })
        };
        let levels_for = |alpha: f32| -> (Vec<Option<f32>>, Vec<Lv>) {
            let pv: Vec<Option<f32>> = par_map(pilots.len(), |i| eq_at(pilots[i].col, pilots[i].row, alpha));
            // the frame's levels: the fallback for a block with too few pilots of a level around it
            let g: Lv = { let mut t = [(0f64, 0u32); 8]; for (i, v) in pv.iter().enumerate() { if let Some(v) = v { let c = pilots[i].colour as usize; if c < n_lv { t[c].0 += *v as f64; t[c].1 += 1; } } } std::array::from_fn(|c| if t[c].1 == 0 { lv[c] } else { (t[c].0 / t[c].1 as f64) as f32 }) };
            let blv: Vec<Lv> = par_map(nblocks, |b| {
                let mut acc = [(0f32, 0u32); 8];
                around(b, &mut |i| if let Some(v) = pv[i] { let c = pilots[i].colour as usize; if c < n_lv { acc[c].0 += v; acc[c].1 += 1; } });
                std::array::from_fn(|c| if c >= n_lv { 0.0 } else if acc[c].1 < 3 { g[c] } else { acc[c].0 / acc[c].1 as f32 })
            });
            (pv, blv)
        };
        let nearest = |v: f32, l: &Lv| -> (usize, f32, f32) {
            let (mut best, mut d1, mut d2) = (0usize, f32::MAX, f32::MAX);
            for k in 0..n_lv { let d = (v - l[k]).powi(2); if d < d1 { d2 = d1; d1 = d; best = k; } else if d < d2 { d2 = d; } }
            (best, d1, d2)
        };
        let pilot_errors = |alpha: f32| -> usize {
            let (pv, blv) = levels_for(alpha);
            pilots.iter().enumerate().filter(|(i, p)| pv[*i].is_some_and(|v| nearest(v, &blv[p.block]).0 != p.colour as usize)).count()
        };
        // coarse 0..0.8 in 0.2 steps, then the neighbours of the best at 0.1
        let mut best_alpha = (0f32, usize::MAX);
        for step in 0..=4 { let a = step as f32 * 0.2; let e = pilot_errors(a); if e < best_alpha.1 { best_alpha = (a, e); } }
        for a in [best_alpha.0 - 0.1, best_alpha.0 + 0.1] { if (0.0..=0.85).contains(&a) { let e = pilot_errors(a); if e < best_alpha.1 { best_alpha = (a, e); } } }
        let alpha = best_alpha.0;
        mark("grey alpha sweep");
        let (pilot_eq, block_lv) = levels_for(alpha);
        // every data cell: (grid index, block, equalised sample); NaN outside the image
        let cells: Vec<(usize, usize, f32)> = (0..used).map(|i| {
            let (c, r) = grid.data_position(i as u32);
            ((r * cols + c) as usize, ((r / BLOCK) * bw + c / BLOCK) as usize, eq_at(c, r, alpha).unwrap_or(f32::NAN))
        }).collect();
        let decide = |v: f32, l: &Lv, off: &[f32; 8]| -> (u8, u16) {
            let (mut best, mut d1, mut d2) = (0usize, f32::MAX, f32::MAX);
            for k in 0..n_lv { let d = (v - l[k] - off[k]).powi(2); if d < d1 { d2 = d1; d1 = d; best = k; } else if d < d2 { d2 = d; } }
            // margin in palette units as ColourModel::classify gives it (3 equal channels, (d2 - d1) / 4)
            let scale = (255.0 / (l[n_lv - 1] - l[0]).abs().max(1.0)).powi(2);
            (best as u8, (3.0 * (d2 - d1) * scale / 4.0).clamp(0.0, 65535.0) as u16)
        };
        let zero = [0f32; 8];
        for &(i, b, v) in &cells { if !v.is_nan() { (values[i], margins[i]) = decide(v, &block_lv[b], &zero); } }
        // pattern-dependent decision (the grey analogue of pattern_decide): blur pulls a cell toward its
        // neighbours, so a level reads differently among dark and among bright neighbours. Class = sum of
        // the 4 edge neighbours' levels (pilots: their known level); per class and level, the pilots give the
        // mean offset of their sample from their block's level; cells are decided again with those offsets.
        // Two rounds (the second sees the corrected neighbours).
        let n_class = 4 * (n_lv - 1) + 1;
        let mut known = vec![-1i8; (cols * rows) as usize];
        for p in &pilots { known[(p.row * cols + p.col) as usize] = p.colour as i8; }
        let mid = ((n_lv - 1) / 2) as u8;
        // a cell's level for the neighbour classes: pilots their known level, data cells their decision,
        // anything else (and the outside) the middle level
        let mut lvl = vec![mid; (cols * rows) as usize];
        // data cells past the payload carry level 0 (the codeword stream leaves them 0)
        for r in 0..rows { for c in 0..cols { if matches!(grid.cell(c, r), FlatCell::Data { .. }) { lvl[(r * cols + c) as usize] = 0; } } }
        for p in &pilots { lvl[(p.row * cols + p.col) as usize] = p.colour; }
        mark("grey cells");
        for _round in 0..2 {
            for &(i, _, _) in &cells { if known[i] < 0 { lvl[i] = values[i]; } }
            let lvl = &lvl;
            let class_of = |c: u32, r: u32| -> usize {
                let at = |cc: i64, rr: i64| -> usize { if cc < 0 || rr < 0 || cc >= cols as i64 || rr >= rows as i64 { mid as usize } else { lvl[(rr as u32 * cols + cc as u32) as usize] as usize } };
                let (c, r) = (c as i64, r as i64);
                at(c - 1, r) + at(c + 1, r) + at(c, r - 1) + at(c, r + 1)
            };
            let mut acc = vec![[(0f64, 0u32); 8]; n_class];
            for (i, p) in pilots.iter().enumerate() {
                let (Some(v), k) = (pilot_eq[i], p.colour as usize) else { continue };
                if k >= n_lv || !tag_known(block_tag[p.block]) { continue; }
                let e = &mut acc[class_of(p.col, p.row).min(n_class - 1)][k];
                e.0 += (v - block_lv[p.block][k]) as f64; e.1 += 1;
            }
            // a (class, level) seen by fewer than 12 pilots keeps no offset
            let off: Vec<[f32; 8]> = acc.iter().map(|row| std::array::from_fn(|k| if k < n_lv && row[k].1 >= 12 { (row[k].0 / row[k].1 as f64) as f32 } else { 0.0 })).collect();
            let next: Vec<(usize, u8, u16)> = par_map(cells.len().div_ceil(8192), |chunk| {
                (chunk * 8192..((chunk + 1) * 8192).min(cells.len())).filter_map(|j| {
                    let (i, b, v) = cells[j];
                    if v.is_nan() { return None; }
                    let (c, r) = ((i as u32) % cols, (i as u32) / cols);
                    let (val, m) = decide(v, &block_lv[b], &off[class_of(c, r).min(n_class - 1)]);
                    Some((i, val, m))
                }).collect::<Vec<_>>()
            }).into_iter().flatten().collect();
            for (i, val, m) in next { values[i] = val; margins[i] = m; }
        }
        mark("cells");
    }
    (FlatReads { cols, rows, values, margins, shifts: shift, alpha: 0.0, block_tag, mix, edge_px: 0.0 }, off)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::Homography;
    use crate::flat::{compose, render, Alphabet, CellFormat, FlatGrid, FlatHeader};
    use crate::header::FrameType;
    use crate::image::GrayImage;
    use std::sync::Arc;

    fn frame(fmt: CellFormat, seq: u8) -> (Arc<FlatGrid>, crate::flat::FlatFrame, GrayImage) {
        let g = Arc::new(FlatGrid::try_new(1024, 1024, fmt).unwrap());
        let h = FlatHeader { format: fmt, cols: g.cols() as u16, rows: g.rows() as u16, frame_type: FrameType::Data, transfer_id: 7, segment: 1, block_id: 99 + seq as u32, frame_seq: seq };
        let payload: Vec<u8> = (0..g.payload_bytes()).map(|i| (((i as u64 + seq as u64 * 7919) * 2654435761) >> 7) as u8).collect();
        let f = compose(g.clone(), h, &payload);
        let im = render(&f).to_gray();
        (g, f, im)
    }
    fn wrong(g: &FlatGrid, f: &crate::flat::FlatFrame, reads: &crate::flat::FlatReads) -> (usize, usize) {
        let n = g.used_data_cells();
        ((0..n as u32).filter(|&i| { let (c, r) = g.data_position(i); reads.at(c, r).0 != f.value(c, r) }).count(), n)
    }
    const BW3: CellFormat = CellFormat { pitch: 3, alphabet: Alphabet::Bw };

    #[test]
    fn fast_reader_reads_a_clean_frame_exactly() {
        let (g, f, im) = frame(BW3, 0);
        let r = read_cells_luma(&im, &g, &Homography::identity(), 0, None).expect("bw is supported");
        assert_eq!(wrong(&g, &f, &r).0, 0);
        assert!(r.block_tag.iter().all(|&t| t == 0));
    }
    #[test]
    fn fast_reader_follows_a_two_px_bend() {
        let (g, f, _) = frame(BW3, 0);
        let rgb = crate::testutil::bend(&render(&f), 2.0, 11);
        let r = read_cells_luma(&rgb.to_gray(), &g, &Homography::identity(), 0, None).unwrap();
        let (bad, n) = wrong(&g, &f, &r);
        assert!(bad * 500 < n, "{bad} of {n} wrong after a 2 px bend");
    }
    #[test]
    fn straddled_capture_shows_two_tags() {
        let (g, _, a) = frame(BW3, 0);
        let (_, _, b) = frame(BW3, 1);
        let mut cam = a.clone();
        let half = (cam.height / 2 * cam.width) as usize;
        cam.data[half..].copy_from_slice(&b.data[half..]);
        let r = read_cells_luma(&cam, &g, &Homography::identity(), 0, None).unwrap();
        let tags: std::collections::BTreeSet<u8> = r.block_tag.iter().copied().filter(|&t| t != crate::flat::read::TAG_UNKNOWN).collect();
        assert!(tags.contains(&0) && tags.contains(&1), "tags {tags:?}");
    }
    /// Tracking from a field gone wrong (the previous capture's bottom showed the manifest, a 12 px colour frame,
    /// read with this grid's blocks): the tracked read's narrow search cannot leave it, and every capture after
    /// handed it on (measured on the 60 soft captures: tracked 39 of 6456 codewords, cold 323). A tracked read
    /// whose untagged pilots disagree with their block's levels reads again from scratch.
    #[test]
    fn a_wrong_tracking_field_is_dropped() {
        let fmt = CellFormat { pitch: 2, alphabet: Alphabet::Bw };
        let (g, f, im) = frame(fmt, 0);
        let (cam, h) = camera(&im, 0.9, (1.3, 0.8), 4);
        let cold = read_cells_luma(&cam, &g, &h, 0, None).unwrap();
        let (bad_cold, n) = wrong(&g, &f, &cold);
        // the bottom half of the field 3 cells off, the rest right
        let (_, bh) = g.blocks();
        let nb = cold.shifts.len();
        let bad_field: Vec<(f32, f32)> = cold.shifts.iter().enumerate().map(|(b, &(x, y))| if b >= nb / 2 && (b / (nb / bh as usize)) as u32 >= bh / 2 { (x + 5.3, y - 4.6) } else { (x, y) }).collect();
        let tracked = read_cells_luma(&cam, &g, &h, 0, Some(&bad_field)).unwrap();
        let (bad_t, _) = wrong(&g, &f, &tracked);
        assert!(bad_t <= bad_cold + n / 2000, "tracked from a wrong field: {bad_t} of {n} cells wrong, cold {bad_cold}");
    }
    /// A capture over the sender's frame change with a long exposure: every row holds both frames, the header's
    /// from 90% of the light at the top to 10% at the bottom (the 2026-11-02 run: 88% to 6%). The read reports
    /// the share of the tile where the other frame has more than a quarter of the light (rows with the header's
    /// share between 0.25 and 0.75: 62% here); a clean capture and a hard cut (a short exposure) report ~0.
    #[test]
    fn a_blended_capture_reports_its_mix() {
        let (g, _, a) = frame(BW3, 1);
        let (_, _, b) = frame(BW3, 0);
        let mut cam = a.clone();
        let h = cam.height as f32;
        for y in 0..cam.height { for x in 0..cam.width {
            let o = (y * cam.width + x) as usize;
            let w = 0.9 - 0.8 * y as f32 / h;
            cam.data[o] = (w * a.data[o] as f32 + (1.0 - w) * b.data[o] as f32).round() as u8;
        } }
        // through a camera (not pixel-exact)
        let read = |im: &GrayImage| { let (c, h) = camera(im, 1.2, (1.3, 0.8), 2); read_cells_luma(&c, &g, &h, 1, None).unwrap() };
        let r = read(&cam);
        assert!((0.5..0.75).contains(&r.mix), "blended: mix {}", r.mix);
        let clean = read(&a);
        assert!(clean.mix < 0.05, "clean: mix {}", clean.mix);
        let mut cut = a.clone();
        let half = (cut.height / 2 * cut.width) as usize;
        cut.data[half..].copy_from_slice(&b.data[half..]);
        let rc = read(&cut);
        assert!(rc.mix < 0.1, "hard cut: mix {}", rc.mix);
    }
    /// Focus from the chrome's own edges (the border's alternating modules, the same on every frame, so a blend
    /// of two frames leaves them sharp): the 10-90% width of their edges in camera px grows with the lens's blur.
    #[test]
    fn edge_spread_tells_focus_from_the_border() {
        let fmt = CellFormat { pitch: 2, alphabet: Alphabet::Bw };
        let (g, _, im) = frame(fmt, 0);
        let (cam, h) = camera(&im, 0.9, (1.3, 0.8), 2);
        let sharp = edge_spread(&cam, g.chrome(), &h);
        // the same through a softer lens: a box blur of 5 camera px
        let mut soft = cam.clone();
        let (w, hh) = (cam.width as i64, cam.height as i64);
        for y in 0..hh { for x in 0..w {
            let mut acc = 0u32;
            for dy in -2..=2i64 { for dx in -2..=2i64 { acc += cam.data[((y + dy).clamp(0, hh - 1) * w + (x + dx).clamp(0, w - 1)) as usize] as u32; } }
            soft.data[(y * w + x) as usize] = (acc / 25) as u8;
        } }
        let blurred = edge_spread(&soft, g.chrome(), &h);
        assert!(sharp > 0.5 && sharp < 3.0, "sharp: {sharp}");
        assert!(blurred > sharp + 1.5, "blurred {blurred} against sharp {sharp}");
    }
    /// A finger over a fifth of the tile: its blocks fail the alignment check from any field, so a tracked read
    /// reads again from scratch, the re-read is no better, and the read says so (the decoder then backs off).
    #[test]
    fn a_finger_makes_the_reread_lose() {
        let fmt = CellFormat { pitch: 2, alphabet: Alphabet::Bw };
        let (g, _, im) = frame(fmt, 0);
        let (mut cam, h) = camera(&im, 0.9, (1.3, 0.8), 4);
        let w = cam.width as usize;
        for y in (cam.height as usize * 3 / 10)..(cam.height as usize * 6 / 10) { for x in (w * 3 / 10)..(w * 6 / 10) { cam.data[y * w + x] = 140; } }
        let cold = read_cells_luma(&cam, &g, &h, 0, None).unwrap();
        let (_, lost) = read_cells_luma_tracked(&cam, &g, &h, 0, Some(&cold.shifts), true, false).unwrap();
        assert_eq!(lost, Some(true), "the check fires and the re-read loses");
        let (_, none) = read_cells_luma_tracked(&cam, &g, &h, 0, Some(&cold.shifts), false, false).unwrap();
        assert_eq!(none, None, "backing off: no re-read");
    }
    /// `im` as a camera `s` of its px per screen px sees it, the tile moved by `off` camera px: each camera
    /// pixel the mean of the screen over its footprint (6x6 points), then the lens's blur (a [1 2 1] / 4
    /// kernel each way) and sensor noise (+-`noise` levels); the homography frame -> camera.
    fn camera(im: &GrayImage, s: f32, off: (f32, f32), noise: i32) -> (GrayImage, Homography) {
        let (w, h) = ((im.width as f32 * s) as u32 + 4, (im.height as f32 * s) as u32 + 4);
        let mut acc = vec![0f32; (w * h) as usize];
        for y in 0..h { for x in 0..w {
            let mut a = 0f32;
            for j in 0..6 { for i in 0..6 {
                let (sx, sy) = (((x as f32 + (i as f32 + 0.5) / 6.0) - off.0) / s, ((y as f32 + (j as f32 + 0.5) / 6.0) - off.1) / s);
                a += if sx < 0.0 || sy < 0.0 || sx >= im.width as f32 || sy >= im.height as f32 { 0.0 } else { im.data[sy as usize * im.width as usize + sx as usize] as f32 };
            } }
            acc[(y * w + x) as usize] = a / 36.0;
        } }
        let at = |x: i64, y: i64| acc[(y.clamp(0, h as i64 - 1) * w as i64 + x.clamp(0, w as i64 - 1)) as usize];
        let mut out = GrayImage::new(w, h);
        let mut seed = 0x9e3779b97f4a7c15u64;
        for y in 0..h as i64 { for x in 0..w as i64 {
            let mut v = 0f32;
            for (dy, ky) in [(-1, 1.0), (0, 2.0), (1, 1.0)] { for (dx, kx) in [(-1, 1.0), (0, 2.0), (1, 1.0)] { v += at(x + dx, y + dy) * ky * kx; } }
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            let n = (seed % (2 * noise as u64 + 1)) as i32 - noise;
            // a screen seen by a camera: black is not 0, white not 255
            out.data[(y * w as i64 + x) as usize] = (30.0 + v / 16.0 * 0.75 + n as f32).round().clamp(0.0, 255.0) as u8;
        } }
        (out, Homography { m: [s, 0.0, off.0, 0.0, s, off.1, 0.0, 0.0, 1.0] })
    }
    /// 2 px cells at 1.5 camera px a cell (a phone a little too far for them), through a soft lens: a camera
    /// pixel mixes neighbouring cells, unevenly with the sampling phase. Measured: 3537 of 226440 cells wrong
    /// (1.56%) with the pattern decision alone, 639 (0.28%) after the equaliser.
    #[test]
    fn small_two_px_cells_read_through_the_equaliser() {
        let fmt = CellFormat { pitch: 2, alphabet: Alphabet::Bw };
        let (g, f, im) = frame(fmt, 0);
        let (cam, h) = camera(&im, 0.75, (1.3, 0.8), 6);
        let r = read_cells_luma(&cam, &g, &h, 0, None).unwrap();
        let (bad, n) = wrong(&g, &f, &r);
        assert!(bad * 200 < n, "{bad} of {n} wrong at 1.5 camera px per cell");
    }
    /// The same 1.5 px tile under a brightness ramp (a lamp's glare across the screen: the contrast halves from
    /// one side to the other and the levels rise by 50 toward the bottom). Measured: 1581 of 226440 cells wrong
    /// (0.70%, against 0.28% without the ramp: the half-contrast side's noise); the equaliser fed every sample on
    /// its block's black..white axis instead (a level that changes inside a region taken out) read 1684, so the
    /// regions' constant is not what costs here and the model stays on the samples.
    #[test]
    fn small_two_px_cells_read_under_a_brightness_ramp() {
        let fmt = CellFormat { pitch: 2, alphabet: Alphabet::Bw };
        let (g, f, im) = frame(fmt, 0);
        let (mut cam, h) = camera(&im, 0.75, (1.3, 0.8), 6);
        let (w, hh) = (cam.width as f32, cam.height as f32);
        for y in 0..cam.height { for x in 0..cam.width {
            let o = (y * cam.width + x) as usize;
            cam.data[o] = (cam.data[o] as f32 * (0.5 + 0.5 * x as f32 / w) + 50.0 * y as f32 / hh).round().clamp(0.0, 255.0) as u8;
        } }
        let r = read_cells_luma(&cam, &g, &h, 0, None).unwrap();
        let (bad, n) = wrong(&g, &f, &r);
        assert!(bad * 100 < n, "{bad} of {n} wrong at 1.5 camera px per cell under a brightness ramp");
    }
    #[test]
    fn tile_partly_outside_gives_erasures() {
        let (g, _, im) = frame(BW3, 0);
        // the tile moved 100 px left out of the image: those cells cannot be read
        let h = Homography { m: [1.0, 0.0, -100.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] };
        let r = read_cells_luma(&im, &g, &h, 0, None).unwrap();
        let (c, row) = g.data_position(0);
        let (x, _) = g.cell_center(c, row);
        if x < 100.0 { assert_eq!(r.at(c, row).1, 0, "a cell outside the image must be an erasure"); }
        assert!(r.margins.iter().filter(|&&m| m == 0).count() > 0);
    }
    #[test]
    fn grey_formats_read_a_clean_frame_exactly() {
        for alphabet in [Alphabet::G4, Alphabet::G8] {
            let (g, f, im) = frame(CellFormat { pitch: 3, alphabet }, 0);
            let r = read_cells_luma(&im, &g, &Homography::identity(), 0, None).expect("grey is on the fast path");
            assert_eq!(wrong(&g, &f, &r).0, 0, "{alphabet:?}");
        }
    }
    /// The pooled refinement's window build against `sample_at`, sample by sample (the reads' bits rest on
    /// it): a noisy synthetic frame, pilots through a sheared homography and within 3 px of every edge (some
    /// just outside), lattice steps 0.05 to 0.75 around several lattice bases, one pilot at a time and all
    /// together; and the general path of an image 1 px high.
    #[test]
    fn pilot_window_is_sample_at() {
        let reference = |src: &GrayImage, pts: &[(f32, f32, f32)], off: &[(f32, f32); 7]| -> [(f32, u32); 49] {
            let mut v = [(0f32, 0u32); 49];
            for j in 0..7 { for i in 0..7 {
                let e = &mut v[j * 7 + i];
                for &(x, y, t) in pts { if let Some(val) = sample_at(src, x + off[i].0, y + off[j].1) { e.0 += (val - t).powi(2); e.1 += 1; } }
            } }
            v
        };
        let noisy = |w: u32, h: u32| { let mut im = GrayImage::new(w, h); for (i, p) in im.data.iter_mut().enumerate() { *p = ((i as u64 * 2654435761) >> 13) as u8; } im };
        let (w, h) = (97u32, 61u32);
        let im = noisy(w, h);
        // a sheared, slightly projective homography over a lattice of pilots
        let hm = Homography { m: [1.03, 0.21, 2.5, -0.17, 0.96, 9.25, 0.0004, -0.0003, 1.0] };
        let mut pts = Vec::new();
        for gy in 0..12 { for gx in 0..17 { let (x, y) = hm.apply(gx as f32 * 5.3 + 0.7, gy as f32 * 4.9 + 0.3); pts.push((x, y, ((gx * 7 + gy * 3) % 5) as f32 * 60.0)); } }
        // and along every edge, from 0.25 px outside to 2.75 px inside
        for k in 0..40 {
            let (f, d) = (k as f32 / 40.0, (k % 7) as f32 * 0.5 - 0.25);
            pts.extend([(d, f * h as f32, 30.0), (w as f32 - d, f * h as f32, 200.0), (f * w as f32, d, 90.0), (f * w as f32, h as f32 - d, 140.0)]);
        }
        let (mut partial, mut windows) = (0, 0);
        for step in [0.05f32, 0.125, 0.3, 0.5, 0.75] {
            for (bx, by) in [(0i32, 0i32), (3, -2), (-7, 5), (11, 11)] {
                let off: [(f32, f32); 7] = std::array::from_fn(|i| ((bx + i as i32 - 3) as f32 * step, (by + i as i32 - 3) as f32 * step));
                for one in pts.chunks(1).chain(std::iter::once(&pts[..])) {
                    let (got, want) = (pilot_window(&im, one.iter().copied(), &off), reference(&im, one, &off));
                    for q in 0..49 { assert_eq!((got[q].0.to_bits(), got[q].1), (want[q].0.to_bits(), want[q].1), "step {step}, base ({bx}, {by}), entry {q}, first pilot {:?}", one[0]); }
                    // a window that leaves the image: some of its entries count the pilot, some do not
                    if one.len() == 1 && got.iter().any(|e| e.1 == 0) && got.iter().any(|e| e.1 == 1) { partial += 1; }
                    windows += 1;
                }
            }
        }
        assert!(partial > 100, "only {partial} of {windows} windows crossed an image edge");
        // the general path (an image under 2 px a side)
        let thin = noisy(9, 1);
        for step in [0.05f32, 0.5, 0.75] {
            let off: [(f32, f32); 7] = std::array::from_fn(|i| ((i as f32 - 3.0) * step, (i as f32 - 3.0) * step * 0.25));
            let pts = [(0.3f32, 0.6f32, 10.0f32), (4.5, 0.5, 200.0), (8.9, 0.1, 99.0), (2.0, 0.9, 0.0)];
            let (got, want) = (pilot_window(&thin, pts.iter().copied(), &off), reference(&thin, &pts, &off));
            for q in 0..49 { assert_eq!((got[q].0.to_bits(), got[q].1), (want[q].0.to_bits(), want[q].1), "1 px high, step {step}, entry {q}"); }
        }
    }
    /// The per-row cell sampler against `at(h.apply(x, yy) + shift)` (`sample_at` of the moved position),
    /// sample by sample (the reads' bits rest on it): projective homographies with w varying along the row
    /// (one through w = 0 exactly), rows and columns off every edge of the image, exact positions through the
    /// identity (x.5 ties of the pixel cell, x = side - 1, just inside and just outside each edge), NaN and
    /// infinite positions (from the shift and from the homography), and the general path of images 1 px
    /// high and 1 px wide.
    #[test]
    fn sample_row_is_sample_at() {
        let noisy = |w: u32, h: u32| { let mut im = GrayImage::new(w, h); for (i, p) in im.data.iter_mut().enumerate() { *p = ((i as u64 * 2654435761) >> 13) as u8; } im };
        // (inside, outside, x.5 ties inside, x = side - 1 inside) over the checked samples
        let mut seen = (0usize, 0usize, 0usize, 0usize);
        let mut check = |im: &GrayImage, hm: &Homography, xs: &[f32], yy: f32, sh: &dyn Fn(usize) -> (f32, f32, f32, f32, f32)| {
            let n = xs.len();
            let hx: [Vec<f32>; 3] = [xs.iter().map(|&x| hm.m[0] * x).collect(), xs.iter().map(|&x| hm.m[3] * x).collect(), xs.iter().map(|&x| hm.m[6] * x).collect()];
            let (mut s0x, mut dsx, mut s0y, mut dsy, mut fu) = (vec![0f32; n], vec![0f32; n], vec![0f32; n], vec![0f32; n], vec![0f32; n]);
            for c in 0..n { (s0x[c], dsx[c], s0y[c], dsy[c], fu[c]) = sh(c); }
            let cols = RowCols { hx: [&hx[0], &hx[1], &hx[2]], fu: &fu, s0x: &s0x, dsx: &dsx, s0y: &s0y, dsy: &dsy };
            let (mut out, mut scratch) = (vec![0f32; n], RowScratch::default());
            // twice through the same scratch: a row reuses the previous row's working arrays
            for _ in 0..2 {
                sample_row(im, &hm.m, yy, &cols, &mut scratch, &mut out);
                for c in 0..n {
                    let (cx, cy) = hm.apply(xs[c], yy);
                    let (px, py) = (cx + s0x[c] + dsx[c] * fu[c], cy + s0y[c] + dsy[c] * fu[c]);
                    let want = sample_at(im, px, py).unwrap_or(f32::NAN);
                    assert_eq!(out[c].to_bits(), want.to_bits(), "{}x{} image, h {:?}, column {c} (x {}), yy {yy}: at ({px}, {py})", im.width, im.height, hm.m, xs[c]);
                    if want.is_nan() { seen.1 += 1; } else {
                        seen.0 += 1;
                        if (px - 0.5).fract() == 0.5 || (py - 0.5).fract() == 0.5 { seen.2 += 1; }
                        if px - 0.5 == im.width as f32 - 1.0 || py - 0.5 == im.height as f32 - 1.0 { seen.3 += 1; }
                    }
                }
            }
        };
        let im = noisy(97, 61);
        let wobble = |c: usize| -> (f32, f32, f32, f32, f32) { ((((c * 37) % 11) as f32 - 5.0) * 0.13, ((c % 5) as f32 - 2.0) * 0.07, (((c * 23) % 9) as f32 - 4.0) * 0.11, ((c % 3) as f32 - 1.0) * 0.05, ((c % 8) as f32 + 0.5) / 8.0) };
        // projective: w = 1 + 0.004 x - 0.003 y along and across the row; the tile reaches past every edge
        let proj = Homography { m: [1.03, 0.21, -6.5, -0.17, 0.96, -4.25, 0.004, -0.003, 1.0] };
        let cols: Vec<f32> = (-12..118).map(|c| c as f32 + 0.5).collect();
        for r in -10..80 { check(&im, &proj, &cols, r as f32 * 0.75 + 0.5, &wobble); }
        // w through 0: 1 - x / 16 is exactly 0 at x = 16 (the 1e-12 floor), huge either side of it
        let pole = Homography { m: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.0625, 0.0, 1.0] };
        let cols_p: Vec<f32> = (0..64).map(|c| c as f32 * 0.5).collect();
        for yy in [0.5f32, 10.25, 30.0, 60.5] { check(&im, &pole, &cols_p, yy, &wobble); }
        // and 0 / 0 there without the floor: the floor puts the cell at the origin (then moved by its shift)
        let pole0 = Homography { m: [1.0, 0.0, -16.0, 0.0, 1.0, 0.0, -0.0625, 0.0, 1.0] };
        check(&im, &pole0, &cols_p, 0.0, &|_| (1.25, 0.0, 1.75, 0.0, 0.0));
        // exact positions through the identity: cell centres moved by quarter pixels give every x.5 tie,
        // x = side - 1 and the positions just inside and just outside every edge
        let id = Homography::identity();
        let cols_i: Vec<f32> = (0..97).map(|c| c as f32 + 0.5).collect();
        let quarters = |c: usize| -> (f32, f32, f32, f32, f32) { let q = ((c % 9) as f32 - 4.0) * 0.25; (q, 0.0, ((c % 7) as f32 - 3.0) * 0.25, 0.0, 0.0) };
        for r in 0..61 { check(&im, &id, &cols_i, r as f32 + 0.5, &quarters); }
        for yy in [-0.25f32, 0.0, 0.25, 60.5, 60.75, 61.0, 61.25] { check(&im, &id, &cols_i, yy, &|_| (0.0, 0.0, 0.0, 0.0, 0.0)); }
        let edges: Vec<f32> = vec![-0.75, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 95.5, 96.0, 96.25, 96.5, 96.75, 97.0, 97.25];
        for yy in [0.5f32, 30.5, 60.5] { check(&im, &id, &edges, yy, &|_| (0.0, 0.0, 0.0, 0.0, 0.0)); }
        // NaN and infinite positions from the shift, and NaN from the homography
        let wild = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1e30, -1e30];
        for yy in [0.5f32, 30.5] {
            check(&im, &id, &cols_i[..20], yy, &|c| (wild[c % 5], 0.0, 0.0, 0.0, 0.0));
            check(&im, &id, &cols_i[..20], yy, &|c| (0.0, 0.0, wild[c % 5], 0.0, 0.0));
            check(&im, &id, &cols_i[..20], yy, &|c| (0.0, wild[c % 5], 0.0, 0.0, 0.5));
        }
        let nan_h = Homography { m: [1.0, 0.0, f32::NAN, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] };
        check(&im, &nan_h, &cols_i[..10], 5.5, &wobble);
        // the general path: images under 2 px a side
        for (w, h) in [(9u32, 1u32), (1, 7)] {
            let thin = noisy(w, h);
            let xs: Vec<f32> = (-2..12).map(|c| c as f32 * 0.75).collect();
            for yy in [-0.5f32, 0.0, 0.3, 0.5, 0.9, 1.0, 3.5, 7.25] { check(&thin, &proj, &xs, yy, &wobble); check(&thin, &id, &xs, yy, &quarters); }
        }
        let (inside, outside, ties, last) = seen;
        assert!(inside > 20_000 && outside > 2_000, "{inside} samples inside, {outside} outside");
        assert!(ties > 500 && last > 50, "{ties} x.5 ties, {last} at side - 1");
    }
    #[test]
    fn colour_formats_fall_back() {
        let (g, _, im) = frame(CellFormat { pitch: 4, alphabet: Alphabet::C8 }, 0);
        assert!(read_cells_luma(&im, &g, &Homography::identity(), 0, None).is_none());
    }
}
