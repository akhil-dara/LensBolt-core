//! Sub-pixel homography refinement from the timing border.
//!
//! The four marker centres give a homography good to ~0.5 px, which is not enough for 5 px tiles seen
//! at ~4 camera px. The border's white cells are hundreds of known points; locating each one's
//! intensity centroid near its predicted position and solving a least-squares homography over all of
//! them (plus the markers) brings the error down to a small fraction of a pixel.
use super::Homography;
use crate::image::GrayImage;

/// A frame's timing border as the refinement sees it: alternating known-colour cells around the edge.
/// Implemented by `flat::Chrome` (8 px modules).
pub trait TimingBorder {
    fn border_count(&self) -> u32;
    /// Centre of border cell `i` in frame px.
    fn border_center(&self, i: u32) -> (f32, f32);
    fn border_is_white(&self, i: u32) -> bool;
    /// Side of a border cell in frame px.
    fn module_px(&self) -> f32;
    fn size_px(&self) -> (u32, u32);
}

impl<T: TimingBorder> TimingBorder for std::sync::Arc<T> {
    fn border_count(&self) -> u32 { (**self).border_count() }
    fn border_center(&self, i: u32) -> (f32, f32) { (**self).border_center(i) }
    fn border_is_white(&self, i: u32) -> bool { (**self).border_is_white(i) }
    fn module_px(&self) -> f32 { (**self).module_px() }
    fn size_px(&self) -> (u32, u32) { (**self).size_px() }
}

impl TimingBorder for crate::flat::Chrome {
    fn border_count(&self) -> u32 { self.border_module_count() }
    fn border_center(&self, i: u32) -> (f32, f32) { self.border_module_center(i) }
    fn border_is_white(&self, i: u32) -> bool { crate::flat::Chrome::border_is_white(i) }
    fn module_px(&self) -> f32 { crate::flat::MODULE_PX as f32 }
    fn size_px(&self) -> (u32, u32) { (self.width_px(), self.height_px()) }
}

/// Least-squares affine map src -> dst (6 parameters). Needs >= 3 non-collinear points. Used while the
/// known points still cover only a small corner of the frame, where a full homography would be poorly
/// conditioned in its perspective terms.
pub fn affine_least_squares(pairs: &[((f32, f32), (f32, f32))]) -> Option<Homography> {
    if pairs.len() < 3 { return None; }
    let n = pairs.len() as f64;
    let (mut mx, mut my) = (0f64, 0f64);
    for &((x, y), _) in pairs { mx += x as f64; my += y as f64; }
    let (mx, my) = (mx / n, my / n);
    // normal equations on centred source coordinates: [sxx sxy sx; sxy syy sy; sx sy n]
    let mut m = [[0f64; 3]; 3];
    let (mut bu, mut bv) = ([0f64; 3], [0f64; 3]);
    for &((x, y), (u, v)) in pairs {
        let r = [x as f64 - mx, y as f64 - my, 1.0];
        for i in 0..3 { for j in 0..3 { m[i][j] += r[i] * r[j]; } bu[i] += r[i] * u as f64; bv[i] += r[i] * v as f64; }
    }
    let solve = |b: [f64; 3]| -> Option<[f64; 3]> {
        let mut a = [[0f64; 4]; 3];
        for i in 0..3 { a[i][..3].copy_from_slice(&m[i]); a[i][3] = b[i]; }
        for col in 0..3 {
            let piv = (col..3).max_by(|&r1, &r2| a[r1][col].abs().partial_cmp(&a[r2][col].abs()).unwrap())?;
            if a[piv][col].abs() < 1e-9 { return None; }
            a.swap(col, piv);
            for r in 0..3 { if r != col { let f = a[r][col] / a[col][col]; for c in col..4 { a[r][c] -= f * a[col][c]; } } }
        }
        Some([a[0][3] / a[0][0], a[1][3] / a[1][1], a[2][3] / a[2][2]])
    };
    let (pu, pv) = (solve(bu)?, solve(bv)?);
    // u = pu0*(x-mx) + pu1*(y-my) + pu2
    Some(Homography { m: [pu[0] as f32, pu[1] as f32, (pu[2] - pu[0] * mx - pu[1] * my) as f32,
                          pv[0] as f32, pv[1] as f32, (pv[2] - pv[0] * mx - pv[1] * my) as f32, 0.0, 0.0, 1.0] })
}

/// Least-squares homography (DLT normal equations) mapping src -> dst. Needs >= 4 points.
pub fn homography_least_squares(pairs: &[((f32, f32), (f32, f32))]) -> Option<Homography> {
    if pairs.len() < 4 { return None; }
    // normalise for conditioning
    let n = pairs.len() as f64;
    let (mut sx, mut sy, mut dx, mut dy) = (0f64, 0f64, 0f64, 0f64);
    for &((x, y), (u, v)) in pairs { sx += x as f64; sy += y as f64; dx += u as f64; dy += v as f64; }
    let (sx, sy, dx, dy) = (sx / n, sy / n, dx / n, dy / n);
    let (mut ss, mut sd) = (0f64, 0f64);
    for &((x, y), (u, v)) in pairs { ss += ((x as f64 - sx).powi(2) + (y as f64 - sy).powi(2)).sqrt(); sd += ((u as f64 - dx).powi(2) + (v as f64 - dy).powi(2)).sqrt(); }
    let (ks, kd) = (n / ss.max(1e-9), n / sd.max(1e-9));
    let mut ata = [[0f64; 8]; 8];
    let mut atb = [0f64; 8];
    for &((x, y), (u, v)) in pairs {
        let (x, y) = ((x as f64 - sx) * ks, (y as f64 - sy) * ks);
        let (u, v) = ((u as f64 - dx) * kd, (v as f64 - dy) * kd);
        let rows = [([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y], u), ([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y], v)];
        for (r, b) in rows {
            for i in 0..8 { atb[i] += r[i] * b; for j in 0..8 { ata[i][j] += r[i] * r[j]; } }
        }
    }
    // solve ata h = atb
    let mut a = [[0f64; 9]; 8];
    for i in 0..8 { a[i][..8].copy_from_slice(&ata[i]); a[i][8] = atb[i]; }
    for col in 0..8 {
        let piv = (col..8).max_by(|&r1, &r2| a[r1][col].abs().partial_cmp(&a[r2][col].abs()).unwrap())?;
        if a[piv][col].abs() < 1e-12 { return None; }
        a.swap(col, piv);
        for r in 0..8 { if r != col { let f = a[r][col] / a[col][col]; if f != 0.0 { for c in col..9 { a[r][c] -= f * a[col][c]; } } } }
    }
    let mut hn = [0f64; 9];
    for i in 0..8 { hn[i] = a[i][8] / a[i][i]; }
    hn[8] = 1.0;
    // denormalise: H = Td^-1 * Hn * Ts, with Ts: p -> (p - s) * ks ; Td^-1: q -> q / kd + d
    let ts = [ks, 0.0, -sx * ks, 0.0, ks, -sy * ks, 0.0, 0.0, 1.0];
    let tdi = [1.0 / kd, 0.0, dx, 0.0, 1.0 / kd, dy, 0.0, 0.0, 1.0];
    let m = mul3(&tdi, &mul3(&hn, &ts));
    let mut out = [0f32; 9];
    for i in 0..9 { out[i] = (m[i] / m[8]) as f32; }
    Some(Homography { m: out })
}

fn mul3(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut o = [0f64; 9];
    for r in 0..3 { for c in 0..3 { for k in 0..3 { o[r * 3 + c] += a[r * 3 + k] * b[k * 3 + c]; } } }
    o
}

/// Intensity-weighted centroid of the bright pixels within radius `r` of (cx, cy). "Bright" is judged
/// inside the window (midpoint of its min and max), so a dim corner of the screen works as well as a
/// bright one; `thr` (the global threshold) only caps that midpoint so a bright window keeps its old cut.
fn bright_centroid(gray: &GrayImage, cx: f32, cy: f32, r: f32, thr: u8) -> Option<(f32, f32)> {
    let (x0, x1) = (((cx - r).floor().max(0.0)) as i32, ((cx + r).ceil().min(gray.width as f32 - 1.0)) as i32);
    let (y0, y1) = (((cy - r).floor().max(0.0)) as i32, ((cy + r).ceil().min(gray.height as f32 - 1.0)) as i32);
    if x1 <= x0 || y1 <= y0 { return None; }
    let (mut lo, mut hi) = (255u8, 0u8);
    for y in y0..=y1 { for x in x0..=x1 { let v = gray.get(x as u32, y as u32); lo = lo.min(v); hi = hi.max(v); } }
    if hi < lo.saturating_add(24) { return None; } // no contrast: not a white cell on black
    let thr = thr.min(((lo as u16 + hi as u16) / 2) as u8);
    let (mut sw, mut sx, mut sy) = (0f32, 0f32, 0f32);
    for y in y0..=y1 { for x in x0..=x1 {
        let v = gray.get(x as u32, y as u32);
        if v > thr {
            let w = (v - thr) as f32;
            sw += w; sx += w * (x as f32 + 0.5); sy += w * (y as f32 + 0.5);
        }
    } }
    if sw <= 0.0 { return None; }
    let (mx, my) = (sx / sw, sy / sw);
    if ((mx - cx).powi(2) + (my - cy).powi(2)).sqrt() > r { return None; }
    Some((mx, my))
}

/// The step a border's white cell must stand above its black neighbour by, for the steps `diffs` (white minus
/// black) seen in one capture: `STEP_SHARE` of their median size, between `STEP_MIN` and `STEP_MAX` levels. A
/// normal screen's step is 100+ levels and gets the full `STEP_MAX` (20, the fixed margin before); a dim or
/// washed-out one (black and white 20 levels apart: a panel at low brightness, glare lifting the blacks) failed
/// every check with a fixed 20 and read nothing although its cells read whole, and gets a margin scaled to its
/// own contrast. A wrong geometry's steps come in either sign: the share above the margin stays low.
pub fn step_margin(diffs: &mut [f32]) -> f32 {
    if diffs.is_empty() { return STEP_MAX; }
    for d in diffs.iter_mut() { *d = d.abs(); }
    let mid = diffs.len() / 2;
    let (_, med, _) = diffs.select_nth_unstable_by(mid, f32::total_cmp);
    (STEP_SHARE * *med).clamp(STEP_MIN, STEP_MAX)
}
/// `step_margin`'s share of the median step, and its bounds in grey levels: the floor stays clear of a camera's
/// noise (+-2-4 levels), the ceiling is the fixed margin every screen used before.
const STEP_SHARE: f32 = 0.4;
const STEP_MIN: f32 = 6.0;
const STEP_MAX: f32 = 20.0;

/// White minus black for border cell `i` (a white one) and its black neighbour through `h`; None when either
/// is out of view.
fn border_step(gray: &GrayImage, grid: &impl TimingBorder, h: &Homography, i: u32) -> Option<f32> {
    let n = grid.border_count();
    let j = if !grid.border_is_white((i + 1) % n) { (i + 1) % n } else { (i + n - 1) % n };
    let centre = |k: u32| { let (x, y) = grid.border_center(k); h.apply(x, y) };
    let ((wx, wy), (bx, by)) = (centre(i), centre(j));
    let inside = |x: f32, y: f32| x >= 1.0 && y >= 1.0 && x < gray.width as f32 - 2.0 && y < gray.height as f32 - 2.0;
    (inside(wx, wy) && inside(bx, by)).then(|| gray.sample(wx - 0.5, wy - 0.5) - gray.sample(bx - 0.5, by - 0.5))
}

/// Fit (homography or affine) with outlier trimming: points that disagree with the fit by more than
/// `tol` camera px are dropped and the fit repeated. Markers weigh four times but are trimmed like any
/// other point: under a rolling shutter one of them can belong to another frame, and the border then
/// outvotes it.
fn fit_trimmed(points: &[((f32, f32), (f32, f32))], markers: &[((f32, f32), (f32, f32))], full: bool, tol: f32) -> Option<Homography> {
    let mut pts: Vec<((f32, f32), (f32, f32))> = points.to_vec();
    let mut mks: Vec<((f32, f32), (f32, f32))> = markers.to_vec();
    let mut fit = None;
    for _ in 0..4 {
        let mut all = pts.clone();
        for _ in 0..4 { all.extend_from_slice(&mks); }
        let f = if full { homography_least_squares(&all) } else { affine_least_squares(&all) }?;
        let ok = |(a, b): &((f32, f32), (f32, f32))| { let (x, y) = f.apply(a.0, a.1); (x - b.0).hypot(y - b.1) <= tol };
        let before = pts.len() + mks.len();
        pts.retain(ok);
        if pts.len() >= 12 { mks.retain(ok); } // only enough border points may overrule a marker
        fit = Some(f);
        if pts.len() + mks.len() == before || pts.len() < 6 { break; }
    }
    fit
}

/// Refine `h` (frame px -> camera px) using the border's white cells and the marker centres.
/// `thr` is the binarisation threshold used for detection. Falls back to `h` if too few cells are found.
pub fn refine_homography(gray: &GrayImage, grid: &impl TimingBorder, h: &Homography, markers: &[((f32, f32), (f32, f32))], thr: u8) -> Homography {
    refine_homography_passes(gray, grid, h, markers, thr, 2)
}
/// `refine_homography` with `passes` rounds over the whole border.
pub fn refine_homography_passes(gray: &GrayImage, grid: &impl TimingBorder, h: &Homography, markers: &[((f32, f32), (f32, f32))], thr: u8, passes: usize) -> Homography {
    let mut cur = *h;
    for _ in 0..passes { match refine_pass(gray, grid, &cur, markers, thr, &|_| true) { Some(nh) => cur = nh, None => break } }
    cur
}

/// Reaches, as shares of the tile's longer side, over which `refine_homography_grow` widens the border it fits.
const GROW_REACH: [f32; 6] = [0.2, 0.32, 0.45, 0.6, 0.8, 1.1];

/// The fit from three found markers and a completed fourth (a parallelogram's corner, a guess that ignores the
/// perspective: 50 camera px off on a tile rolled 15 degrees with its corner out of the picture). Fitted over the
/// whole border at once it does not come in: the cells far from the found markers are predicted too far off to
/// match, so the fit learns only near the three corners and keeps the guessed perspective (5 px off even beside
/// them, a header module, so no header read). Grown instead: the border near the found markers first, where the
/// prediction holds, then wider, each fit predicting the next ring, then two rounds over everything in view.
pub fn refine_homography_grow(gray: &GrayImage, grid: &impl TimingBorder, h: &Homography, markers: &[((f32, f32), (f32, f32))], thr: u8) -> Homography {
    let (w_px, h_px) = grid.size_px();
    let side = w_px.max(h_px) as f32;
    let anchors: Vec<(f32, f32)> = markers.iter().map(|m| m.0).collect();
    let mut cur = *h;
    for share in GROW_REACH {
        let reach = share * side;
        let near = |c: (f32, f32)| anchors.iter().any(|a| (a.0 - c.0).hypot(a.1 - c.1) <= reach);
        if let Some(nh) = refine_pass(gray, grid, &cur, markers, thr, &near) { cur = nh; }
    }
    refine_homography_passes(gray, grid, &cur, markers, thr, 2)
}

/// One round of the border's fit through `cur`, over the white border cells `near` accepts (tile px): None when
/// the pitch is absurd (a wrong candidate) or too few cells match.
fn refine_pass(gray: &GrayImage, grid: &impl TimingBorder, cur: &Homography, markers: &[((f32, f32), (f32, f32))], thr: u8, near: &dyn Fn((f32, f32)) -> bool) -> Option<Homography> {
    let pitch = grid.module_px();
    let (w_px, h_px) = grid.size_px();
    // camera-space pitch near the frame centre
    let (cx, cy) = (w_px as f32 / 2.0, h_px as f32 / 2.0);
    let (ax, ay) = cur.apply(cx, cy); let (bx, by) = cur.apply(cx + pitch, cy);
    let cam_pitch = ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt();
    // a wrong grid/rotation candidate can give an absurd pitch; refuse rather than scan huge windows
    if !(1.0..=200.0).contains(&cam_pitch) { return None; }
    let r = (cam_pitch * 0.55).clamp(1.5, 64.0);
    let mut pairs: Vec<((f32, f32), (f32, f32))> = Vec::new();
    // a white border cell counts when it shows this grid's local pattern: its black neighbour clearly darker, by
    // a margin scaled to this capture's contrast. Where the border belongs to another frame (a rolling-shutter
    // split across grids) or to clutter, it usually is not, and the point stays out of the fit.
    let steps: Vec<(u32, f32)> = (0..grid.border_count()).filter(|&i| grid.border_is_white(i) && near(grid.border_center(i))).filter_map(|i| border_step(gray, grid, cur, i).map(|d| (i, d))).collect();
    let margin = step_margin(&mut steps.iter().map(|s| s.1).collect::<Vec<_>>());
    for &(i, step) in &steps {
        let centre = grid.border_center(i);
        let (px, py) = cur.apply(centre.0, centre.1);
        if step <= margin { continue; }
        if let Some(m) = bright_centroid(gray, px, py, r, thr) { pairs.push((centre, m)); }
    }
    if pairs.len() < 16 { return None; }
    // markers count several times (they are the most reliable points); outliers are trimmed
    fit_trimmed(&pairs, markers, true, (0.4 * cam_pitch).max(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn affine_least_squares_recovers_an_affine_map() {
        let h = Homography { m: [1.3, 0.2, 40.0, -0.15, 1.25, 12.0, 0.0, 0.0, 1.0] };
        let pairs: Vec<_> = (0..12).map(|i| { let p = ((i % 4) as f32 * 30.0, (i / 4) as f32 * 17.0); (p, h.apply(p.0, p.1)) }).collect();
        let a = affine_least_squares(&pairs).unwrap();
        for (p, q) in &pairs { let r = a.apply(p.0, p.1); assert!((r.0 - q.0).abs() < 1e-2 && (r.1 - q.1).abs() < 1e-2); }
    }
    #[test]
    fn least_squares_recovers_an_exact_homography() {
        let h = Homography::from_points([(0.0, 0.0), (100.0, 0.0), (100.0, 60.0), (0.0, 60.0)], [(5.0, 7.0), (210.0, 12.0), (205.0, 140.0), (8.0, 128.0)]).unwrap();
        let pairs: Vec<_> = (0..30).map(|i| { let p = ((i % 6) as f32 * 20.0, (i / 6) as f32 * 12.0); (p, h.apply(p.0, p.1)) }).collect();
        let ls = homography_least_squares(&pairs).unwrap();
        for (p, q) in &pairs { let r = ls.apply(p.0, p.1); assert!((r.0 - q.0).abs() < 1e-2 && (r.1 - q.1).abs() < 1e-2, "{p:?} -> {r:?} vs {q:?}"); }
    }
    #[test]
    fn refinement_beats_marker_only_homography() {
        use crate::detect::{find_markers, markers::two_means_threshold};
        use crate::flat::{Alphabet, CellFormat};
        use crate::testutil::synth_flat;
        let (g, cam, h2c) = synth_flat(CellFormat { pitch: 4, alphabet: Alphabet::C8 }, 1600, 1500, 10.0, 5);
        let gray = cam.to_gray();
        let m = find_markers(&gray).unwrap();
        let chrome = *g.chrome();
        // the markers alone already land within ~0.3 px here; perturb two of them by 1.5 px so the
        // border has something to correct (a real camera's marker centres are that noisy)
        let mut pts = m.as_array();
        pts[1].0 += 1.5; pts[2].1 -= 1.5;
        let coarse = Homography::from_points(chrome.marker_centers(), pts).unwrap();
        let pairs: Vec<_> = chrome.marker_centers().iter().zip(pts.iter()).map(|(a, b)| (*a, *b)).collect();
        let fine = refine_homography(&gray, &chrome, &coarse, &pairs, two_means_threshold(&gray.data));
        let err = |h: &Homography| -> f32 {
            let mut worst = 0f32;
            for (c, r) in [(50u32, 50u32), (160, 90), (230, 130), (100, 170), (200, 20)] {
                let (x, y) = g.cell_center(c, r); let (a, b) = h.apply(x, y); let (u, v) = h2c.apply(x, y);
                worst = worst.max(((a - u).powi(2) + (b - v).powi(2)).sqrt());
            }
            worst
        };
        let (e0, e1) = (err(&coarse), err(&fine));
        assert!(e1 < e0 * 0.6 && e1 < 0.5, "coarse {e0:.3} px, refined {e1:.3} px");
    }
}
