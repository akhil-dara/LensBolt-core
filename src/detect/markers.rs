use crate::image::GrayImage;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MarkerSet { pub tl: (f32, f32), pub tr: (f32, f32), pub br: (f32, f32), pub bl: (f32, f32) }
impl MarkerSet { pub fn as_array(&self) -> [(f32, f32); 4] { [self.tl, self.tr, self.br, self.bl] } }

#[derive(Clone, Copy)]
struct Hit { x: f32, y: f32, unit: f32 }

/// 1:1:3:1:1 check with asymmetric tolerance: over-exposure and blur erode the black rings (bloom)
/// while whites keep their size, so the unit is estimated from the white runs and black runs are
/// accepted down to a quarter of it.
fn ratio_ok(runs: &[u32; 5]) -> Option<f32> {
    let total: u32 = runs.iter().sum();
    if total < 7 { return None; }
    let unit = (runs[0] as f32 + runs[2] as f32 / 3.0 + runs[4] as f32) / 3.0;
    if unit < 1.0 { return None; }
    let r = |i: usize| runs[i] as f32 / unit;
    let white_ok = |v: f32| (0.55..=1.7).contains(&v);
    if !white_ok(r(0)) || !white_ok(r(4)) { return None; }
    if !(2.2..=4.0).contains(&r(2)) { return None; }
    if !(0.12..=1.7).contains(&r(1)) || !(0.12..=1.7).contains(&r(3)) { return None; }
    if runs[1] < 2 || runs[3] < 2 { return None; }
    Some(unit)
}

/// Walk from (x, y) along one axis collecting run lengths: centre run (white), black, white.
/// Returns [centre_half, black, white] run lengths or None if the pattern breaks.
fn runs_from(bin: &[u8], w: u32, h: u32, x: i32, y: i32, dx: i32, dy: i32, max: i32) -> Option<[u32; 3]> {
    let at = |xx: i32, yy: i32| -> u8 { if xx < 0 || yy < 0 || xx >= w as i32 || yy >= h as i32 { 0 } else { bin[(yy as u32 * w + xx as u32) as usize] } };
    if at(x, y) == 0 { return None; }
    let mut runs = [0u32; 3];
    let (mut cx, mut cy) = (x, y);
    for idx in 0..3 {
        let want = if idx % 2 == 0 { 1 } else { 0 };
        let mut len = 0;
        while at(cx, cy) == want { len += 1; cx += dx; cy += dy; if len > max { return None; } }
        if len == 0 { return None; }
        runs[idx] = len as u32;
    }
    Some(runs)
}

/// Check the 1:1:3:1:1 pattern through (x,y) along (dx,dy) both ways; returns (centre offset along axis, unit).
fn check_axis(bin: &[u8], w: u32, h: u32, x: u32, y: u32, dx: i32, dy: i32, unit: f32) -> Option<(f32, f32)> {
    let max = (unit * 3.0) as i32 + 3;
    let a = runs_from(bin, w, h, x as i32, y as i32, -dx, -dy, max)?;
    let b = runs_from(bin, w, h, x as i32 + dx, y as i32 + dy, dx, dy, max)?;
    let total = [a[2], a[1], a[0] + b[0], b[1], b[2]];
    let u = ratio_ok(&total)?;
    // centre of the middle run, relative to (x,y): middle run spans [-a0+1, +b0]
    let start = -(a[0] as f32) + 1.0;
    let centre = start + (a[0] + b[0]) as f32 / 2.0 - 0.5;
    Some((centre, u))
}

/// Binarisation threshold by Otsu's method (maximum between-class variance). Unlike an iterated
/// two-means seeded at 128 it cannot get stuck when the whole screen is darker than mid-grey, which a
/// dim display or a conservative auto-exposure easily produces.
pub fn two_means_threshold(data: &[u8]) -> u8 { two_means_of(&histogram(data), data.len()) }

/// Counts of each grey level. Four interleaved tables: runs of equal pixels (a 1-2 px tile's black and
/// white) otherwise wait on the same counter.
fn histogram(data: &[u8]) -> [u64; 256] {
    let mut lanes = [[0u32; 256]; 4];
    let mut hist = [0u64; 256];
    // a lane counts at most u32::MAX in one part
    for part in data.chunks(u32::MAX as usize) {
        let quads = part.chunks_exact(4);
        for &v in quads.remainder() { hist[v as usize] += 1; }
        for q in quads { lanes[0][q[0] as usize] += 1; lanes[1][q[1] as usize] += 1; lanes[2][q[2] as usize] += 1; lanes[3][q[3] as usize] += 1; }
        for l in lanes.iter_mut() { for (h, c) in hist.iter_mut().zip(l.iter_mut()) { *h += *c as u64; *c = 0; } }
    }
    hist
}

/// Frames up to this many pixels (8.4 MP) give the gate their whole histogram: Full HD, the browser
/// receiver's 2560x1920 and the 8 MP 3264x2448 frames a phone's browser delivers (one core of the
/// development machine: the whole gate 1.7 ms at 2560x1920, 2.7 ms at 3264x2448). Measured on the Full HD
/// 2 px tier: a sampled threshold one level off the full frame's moves a codeword on two of 40 tracked
/// captures, so every frame up to this size is read whole; only the 9.4-12.6 MP camera frames sample.
const THRESHOLD_FULL_MAX: usize = 1 << 23;
/// Pixels the gate's threshold histogram samples on a larger frame (a 12.6 MP frame: every 4th pixel of
/// every 7th row). Measured on the 24 real 9.4-12.6 MP frames (8 at 1 px 4096x3072, 16 at 1 px 3072x3072):
/// 21 at the full-frame threshold, 3 one level off (`tests/gate_sample.rs` prints each).
const THRESHOLD_SAMPLE: usize = 1 << 19;

/// `two_means_threshold` of `gray`, from a strided sample on a frame over `THRESHOLD_FULL_MAX` pixels:
/// rows spread evenly over the frame, every `c`-th pixel along a row with the start shifted from row to
/// row (no column of a periodic pattern is favoured), about `THRESHOLD_SAMPLE` pixels. The row step is odd:
/// an even one samples rows of one parity only (a camera's 2x2 colour and chroma structure): with a step
/// of 6, 22 of those 24 frames came out one level low.
pub fn two_means_threshold_sampled(gray: &GrayImage) -> u8 {
    let (w, h) = (gray.width as usize, gray.height as usize);
    if w * h <= THRESHOLD_FULL_MAX { return two_means_threshold(&gray.data); }
    let c = (w / 1024).clamp(1, 4);
    let r = (w * h).div_ceil(c * THRESHOLD_SAMPLE) | 1;
    let mut hist = [0u64; 256];
    let mut n = 0usize;
    for (k, y) in (r / 2..h).step_by(r).enumerate() {
        let row = &gray.data[y * w..(y + 1) * w];
        for &v in row[k % c..].iter().step_by(c) { hist[v as usize] += 1; n += 1; }
    }
    two_means_of(&hist, n)
}

/// Otsu over a histogram of `count` pixels.
fn two_means_of(hist: &[u64; 256], count: usize) -> u8 {
    let total = count.max(1) as f64;
    let sum_all: f64 = hist.iter().enumerate().map(|(v, &n)| v as f64 * n as f64).sum();
    let (mut w_b, mut sum_b, mut best, mut best_t) = (0f64, 0f64, -1f64, 0usize);
    for t in 0..256 {
        w_b += hist[t] as f64;
        if w_b == 0.0 { continue; }
        let w_f = total - w_b;
        if w_f == 0.0 { break; }
        sum_b += t as f64 * hist[t] as f64;
        let m_b = sum_b / w_b;
        let m_f = (sum_all - sum_b) / w_f;
        let between = w_b * w_f * (m_b - m_f) * (m_b - m_f);
        if between > best { best = between; best_t = t; }
    }
    best_t as u8
}

/// Binarise against the local mean (window ~1/8 of the image, integral image). A screen seen at an angle,
/// or with vignetting, is far darker on one side: measured, the far-side marker whites read 86-99 against
/// a global Otsu threshold of 84 while the near side read 200+. A floor keeps black areas free of noise.
pub fn local_binarize(gray: &GrayImage) -> Vec<u8> { local_binarize_r(gray, (gray.width.max(gray.height) as usize / 16).max(8)) }
/// `local_binarize` with an explicit mean radius (a crop of a frame binarises with the frame's radius).
///
/// A u32 integral image whose sums wrap: a window's sum is the same modulo 2^32 and always fits (255 x the
/// window's area), so the wrapping differences are exact at any frame size, at half the memory of u64. A pixel
/// is above the window's floor mean exactly when `v * area > sum` (integers), so no division per pixel. Was
/// 14.9 ms of the 17.5 ms marker search on a 1080x1920 capture (one core); the same bits
/// (`local_binarize_matches_the_reference`).
pub fn local_binarize_r(gray: &GrayImage, r: usize) -> Vec<u8> {
    let (w, h) = (gray.width as usize, gray.height as usize);
    let stride = w + 1;
    let mut integral = vec![0u32; stride * (h + 1)];
    for y in 0..h {
        let src = &gray.data[y * w..(y + 1) * w];
        let (above, here) = integral[y * stride..(y + 2) * stride].split_at_mut(stride);
        let mut row = 0u32;
        for x in 0..w {
            row = row.wrapping_add(src[x] as u32);
            here[x + 1] = above[x + 1].wrapping_add(row);
        }
    }
    let integral = &integral;
    let mut out = vec![0u8; w * h];
    crate::par::par_chunks_rows(&mut out, w, |y, line| {
        let (y0, y1) = (y.saturating_sub(r), (y + r + 1).min(h));
        let (top, bot) = (&integral[y0 * stride..(y0 + 1) * stride], &integral[y1 * stride..(y1 + 1) * stride]);
        let src = &gray.data[y * w..(y + 1) * w];
        let rows_in = (y1 - y0) as u32;
        // the band of rows y0..y1 summed up to each column: a window's sum is the difference of two of these
        // (exact in u32: a window holds at most 255 x its area, and the area at most (2 r + 1)^2)
        let band: Vec<u32> = bot.iter().zip(top).map(|(&b, &t)| b.wrapping_sub(t)).collect();
        // v > floor(sum / area) exactly when v * area > sum; and above the floor of 24
        let edge = |x: usize| -> u8 {
            let (x0, x1) = (x.saturating_sub(r), (x + r + 1).min(w));
            let (sum, area) = (band[x1].wrapping_sub(band[x0]) as u64, rows_in as u64 * (x1 - x0) as u64);
            let v = src[x] as u64;
            (v > 24 && v * area > sum) as u8
        };
        // the columns whose window lies inside the frame: one area, no clamping (vectorised)
        let (lo, hi) = if w > 2 * r + 1 { (r, w - r - 1) } else { (w, w) };
        for x in 0..lo { line[x] = edge(x); }
        if hi > lo && (rows_in as u64) * (2 * r as u64 + 1) * 255 < u32::MAX as u64 {
            let area = rows_in * (2 * r as u32 + 1);
            let (left, right) = (&band[..], &band[2 * r + 1..]);
            for (x, (o, &v)) in line[lo..hi].iter_mut().zip(&src[lo..hi]).enumerate() {
                let sum = right[x].wrapping_sub(left[x]);
                let v = v as u32;
                *o = (v > 24 && v * area > sum) as u8;
            }
        } else {
            for x in lo..hi { line[x] = edge(x); }
        }
        for x in hi.max(lo)..w { line[x] = edge(x); }
    });
    out
}

/// All marker-like clusters: (x, y, unit, hits). Exposed for diagnostics.
pub fn find_marker_candidates(gray: &GrayImage) -> Vec<(f32, f32, f32, u32)> { find_marker_candidates_r(gray, (gray.width.max(gray.height) as usize / 16).max(8)) }
/// `find_marker_candidates` with an explicit binarisation radius (a window cut from a frame uses the frame's).
pub fn find_marker_candidates_r(gray: &GrayImage, radius: usize) -> Vec<(f32, f32, f32, u32)> {
    let small = gray;
    let (w, h) = (small.width, small.height);
    if w < 8 || h < 8 { return Vec::new(); }
    let bin = local_binarize_r(small, radius);
    let bin = &bin;
    let hits: Vec<Hit> = crate::par::par_ranges_px(h as usize, w as usize, |rows| {
    let mut hits: Vec<Hit> = Vec::new();
    for y in rows.start as u32..rows.end as u32 {
        let row = &bin[(y * w) as usize..((y + 1) * w) as usize];
        let mut runs: Vec<(u8, u32, u32)> = Vec::new();
        let mut start = 0u32;
        for x in 1..=w {
            if x == w || row[x as usize] != row[start as usize] { runs.push((row[start as usize], start, x - start)); start = x; }
        }
        for i in 0..runs.len().saturating_sub(4) {
            if runs[i].0 != 1 { continue; }
            let r = [runs[i].2, runs[i + 1].2, runs[i + 2].2, runs[i + 3].2, runs[i + 4].2];
            if let Some(unit) = ratio_ok(&r) {
                let cx = runs[i + 2].1 + runs[i + 2].2 / 2;
                if let Some((dyc, u)) = check_axis(&bin, w, h, cx, y, 0, 1, unit) {
                    let cy = y as f32 + dyc;
                    let cy_i = cy.round().clamp(0.0, h as f32 - 1.0) as u32;
                    if let Some((dxc, _)) = check_axis(&bin, w, h, cx, cy_i, 1, 0, unit) {
                        let cxf = cx as f32 + dxc;
                        hits.push(Hit { x: cxf + 0.5, y: cy + 0.5, unit: u });
                    }
                }
            }
        }
    }
    hits
    }).concat();
    let mut clusters: Vec<(f32, f32, f32, u32)> = Vec::new();
    for hit in &hits {
        if let Some(c) = clusters.iter_mut().find(|c| { let (cx, cy) = (c.0 / c.3 as f32, c.1 / c.3 as f32); ((cx - hit.x).powi(2) + (cy - hit.y).powi(2)).sqrt() < 1.5 * hit.unit }) {
            c.0 += hit.x; c.1 += hit.y; c.2 += hit.unit; c.3 += 1;
        } else { clusters.push((hit.x, hit.y, hit.unit, 1)); }
    }
    clusters.iter().map(|c| (c.0 / c.3 as f32, c.1 / c.3 as f32, c.2 / c.3 as f32, c.3)).collect()
}

pub fn find_markers(gray: &GrayImage) -> Option<MarkerSet> {
    let mut clusters = find_marker_candidates(gray);
    clusters.sort_by(|a, b| b.3.cmp(&a.3));
    let pts: Vec<(f32, f32)> = clusters.iter().filter(|c| c.3 >= 2).map(|c| (c.0, c.1)).collect();
    corner_set(&pts)
}

/// Candidates a corner search weighs at most (the strongest first): C(12, 4) = 495 quadrilaterals.
const CORNER_CANDIDATES: usize = 12;

/// The four tile corners among marker candidate points, strongest first (only the first `CORNER_CANDIDATES` count):
/// the four that span the largest convex quadrilateral, clockwise on screen (y down) from the one at the smallest
/// angle about their centre, so an upright tile comes out top-left, top-right, bottom-right, bottom-left. Which
/// corner is which is the callers' business (they try the four cyclic orders). Unlike the extremes of x + y and
/// x - y, this does not depend on the tile's roll: rolled 45 degrees its corners tie on both diagonals and those
/// extremes picked one marker twice (every capture of a diagonally held phone read nothing). The largest
/// quadrilateral has its corners on the points' convex hull, so only the hull's points are tried (a hull of h
/// points: C(h, 4) quadrilaterals, already in order; 12 points cost a few microseconds).
pub fn corner_set(pts: &[(f32, f32)]) -> Option<MarkerSet> {
    let pts = &pts[..pts.len().min(CORNER_CANDIDATES)];
    let hull = convex_hull_cw(pts);
    let h = hull.len();
    if h < 4 { return None; }
    let area = |q: &[(f32, f32); 4]| (0..4).map(|i| q[i].0 * q[(i + 1) % 4].1 - q[(i + 1) % 4].0 * q[i].1).sum::<f32>() / 2.0;
    let mut best: Option<(f32, [(f32, f32); 4])> = None;
    for a in 0..h { for b in a + 1..h { for c in b + 1..h { for d in c + 1..h {
        let q = [hull[a], hull[b], hull[c], hull[d]];
        let s = area(&q);
        if best.is_none_or(|(m, _)| s > m) { best = Some((s, q)); }
    } } } }
    let (s, mut q) = best?;
    if s <= 0.0 { return None; }
    // start at the smallest angle about the quadrilateral's centre (the top-left of an upright tile)
    let (cx, cy) = (q.iter().map(|p| p.0).sum::<f32>() / 4.0, q.iter().map(|p| p.1).sum::<f32>() / 4.0);
    let first = (0..4).min_by(|&i, &j| (q[i].1 - cy).atan2(q[i].0 - cx).total_cmp(&(q[j].1 - cy).atan2(q[j].0 - cx))).unwrap();
    q.rotate_left(first);
    Some(MarkerSet { tl: q[0], tr: q[1], br: q[2], bl: q[3] })
}

/// The convex hull of `pts`, clockwise on screen (y down; counter-clockwise in y-up terms), collinear points
/// dropped (Andrew's monotone chain).
fn convex_hull_cw(pts: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let mut p: Vec<(f32, f32)> = pts.to_vec();
    p.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    p.dedup();
    if p.len() < 3 { return p; }
    let cross = |o: (f32, f32), a: (f32, f32), b: (f32, f32)| (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0);
    let mut hull: Vec<(f32, f32)> = Vec::with_capacity(2 * p.len());
    for pass in 0..2 {
        let start = hull.len();
        let iter: Box<dyn Iterator<Item = &(f32, f32)>> = if pass == 0 { Box::new(p.iter()) } else { Box::new(p.iter().rev()) };
        for &q in iter {
            while hull.len() >= start + 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], q) <= 0.0 { hull.pop(); }
            hull.push(q);
        }
        hull.pop();
    }
    hull
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::flat::{Alphabet, CellFormat};
    use crate::testutil::synth_flat;

    #[test]
    fn finds_all_four_markers_within_two_pixels() {
        for (fmt, w, h) in [(CellFormat { pitch: 4, alphabet: Alphabet::C8 }, 1600, 1500), (CellFormat { pitch: 8, alphabet: Alphabet::Bw }, 1280, 1200), (CellFormat { pitch: 3, alphabet: Alphabet::C16 }, 2000, 1900)] {
            let (g, cam, h2c) = synth_flat(fmt, w, h, 12.0, 6);
            let found = find_markers(&cam.to_gray()).expect("markers");
            let expect = g.chrome().marker_centers().map(|(x, y)| h2c.apply(x, y));
            for (i, (f, e)) in found.as_array().iter().zip(expect.iter()).enumerate() {
                let d = ((f.0 - e.0).powi(2) + (f.1 - e.1).powi(2)).sqrt();
                assert!(d < 2.0, "{fmt:?} {w}x{h} marker {i}: found {f:?} expected {e:?} (d={d:.2})");
            }
        }
    }
    /// `local_binarize_r` as it was (a u64 integral image, a division per pixel): the reference the faster one
    /// must match bit for bit.
    fn local_binarize_reference(gray: &GrayImage, r: usize) -> Vec<u8> {
        let (w, h) = (gray.width as usize, gray.height as usize);
        let mut integral = vec![0u64; (w + 1) * (h + 1)];
        for y in 0..h { let mut row = 0u64; for x in 0..w { row += gray.data[y * w + x] as u64; integral[(y + 1) * (w + 1) + x + 1] = integral[y * (w + 1) + x + 1] + row; } }
        let mut out = Vec::with_capacity(w * h);
        for y in 0..h { let (y0, y1) = (y.saturating_sub(r), (y + r + 1).min(h)); for x in 0..w {
            let (x0, x1) = (x.saturating_sub(r), (x + r + 1).min(w));
            let sum = integral[y1 * (w + 1) + x1] + integral[y0 * (w + 1) + x0] - integral[y0 * (w + 1) + x1] - integral[y1 * (w + 1) + x0];
            let mean = sum / ((y1 - y0) * (x1 - x0)) as u64;
            out.push((gray.data[y * w + x] as u64 > mean.max(24)) as u8);
        } }
        out
    }
    #[test]
    fn local_binarize_matches_the_reference() {
        let mut st = 0x2545F4914F6CDD1Du64;
        for (w, h, r) in [(64u32, 48u32, 8usize), (333, 211, 20), (1080, 1920, 120), (17, 9, 30), (512, 512, 1)] {
            let mut g = crate::image::GrayImage::new(w, h);
            // a gradient (a screen at an angle), noise, saturated patches and dark corners
            for y in 0..h { for x in 0..w {
                st ^= st << 13; st ^= st >> 7; st ^= st << 17;
                let base = (x * 180 / w.max(1) + y * 60 / h.max(1)) as i32;
                let v = if (x / 37 + y / 23) % 7 == 0 { 255 } else if x < w / 9 && y < h / 9 { (st % 20) as i32 } else { base + (st % 41) as i32 - 20 };
                g.data[(y * w + x) as usize] = v.clamp(0, 255) as u8;
            } }
            assert!(local_binarize_r(&g, r) == local_binarize_reference(&g, r), "{w}x{h} r {r}");
        }
        let big = crate::image::GrayImage { width: 4096, height: 3072, data: vec![255u8; 4096 * 3072] };
        assert!(local_binarize_r(&big, 256) == local_binarize_reference(&big, 256), "a saturated 12.6 MP frame (the u32 integral wraps)");
    }
    #[test]
    fn corner_set_ignores_roll() {
        for deg in [0.0f32, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0, -45.0] {
            let a = deg.to_radians();
            let sq = [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].map(|(x, y)| (500.0 + 300.0 * (x * a.cos() - y * a.sin()), 400.0 + 300.0 * (x * a.sin() + y * a.cos())));
            // the markers among weaker inner clusters (data cells imitating a marker), strongest first
            let pts = [sq[2], sq[0], sq[3], sq[1], (520.0, 410.0), (430.0, 380.0), (600.0, 450.0)];
            let m = corner_set(&pts).unwrap_or_else(|| panic!("roll {deg}: no corners"));
            let got = m.as_array();
            let k = (0..4).find(|&k| got[0] == sq[k]).unwrap_or_else(|| panic!("roll {deg}: {got:?}"));
            for i in 0..4 { assert_eq!(got[i], sq[(k + i) % 4], "roll {deg}: clockwise order"); }
        }
        assert!(corner_set(&[(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]).is_none(), "collinear points are no tile");
    }
    #[test]
    fn blank_image_has_no_markers() {
        assert!(find_markers(&crate::image::GrayImage::new(640, 480)).is_none());
    }
}
