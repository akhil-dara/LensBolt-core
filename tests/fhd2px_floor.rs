//! Full HD 2 px b/w below its comfortable size: real captures (`test/phone_fhd2px_live/dump`, a OnePlus 12 at
//! 1440x1080 filming a 1024 tile of 2 px b/w at 1.76 camera px per cell; the 30 of its 40 captures that read
//! whole, so every cell's true value is known) degraded as a farther or tilted phone would see them: the
//! capture's pixels integrated again over a larger footprint (a farther camera integrates more screen per
//! sensor pixel), then a perspective (yaw, pitch) and a roll about the tile's centre.
//! - `fhd2px_reads_small_and_tilted_tiles`: the gate, on 8 of them and the 8 captures of the run the reader
//!   before the equaliser did not read whole (`HARD`; `MIN_SHARE`).
//! - `fhd2px_reads_under_glare`: the same at 1.6 px under a lamp's reflection (`GLARE`).
//! - `fhd2px_reads_through_motion`: smeared and sheared as a hand-held camera without stabilisation (`MOTION`).
//! - `measure` (ignored): the curve, `cargo test --release --test fhd2px_floor measure -- --ignored
//!   --nocapture` (docs/2026-10-31-fhd-2px-floor.md).
//! Without the captures (they are not in the repository: the gates run only on a machine that has them; keep a
//! copy of test/phone_fhd2px_live) the tests say so and pass; `LB_REQUIRE_CAPTURES=1` makes them fail instead.
use lensbolt_core::detect::Homography;
use lensbolt_core::flat::{compose, FlatDecoder, FlatGrid, FlatHint, FlatRead};
use lensbolt_core::image::GrayImage;

mod common;
use common::{captures, gate_dir};

const SEQ: &str = "test/phone_fhd2px_live/dump";
/// The camera's focal length in px of its 1440x1080 stream (a ~74 degree wide main camera).
const FOCAL_PX: f64 = 1000.0;

fn mul(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut o = [0f64; 9];
    for r in 0..3 { for c in 0..3 { o[r * 3 + c] = (0..3).map(|k| a[r * 3 + k] * b[k * 3 + c]).sum(); } }
    o
}
fn to_h(m: &[f64; 9]) -> Homography { Homography { m: std::array::from_fn(|i| (m[i] / m[8]) as f32) } }
fn of_h(h: &Homography) -> [f64; 9] { std::array::from_fn(|i| h.m[i] as f64) }

/// Camera px per screen px along each tile edge through `h` (top, right, bottom, left).
fn edges(h: &Homography, grid: &FlatGrid) -> [f32; 4] {
    let (w, hh) = (grid.width_px() as f32, grid.height_px() as f32);
    let along = |x0: f32, y0: f32, x1: f32, y1: f32| { let (a, b) = (h.apply(x0, y0), h.apply(x1, y1)); (a.0 - b.0).hypot(a.1 - b.1) / (x1 - x0).hypot(y1 - y0) };
    [along(0.0, 0.0, w, 0.0), along(w, 0.0, w, hh), along(0.0, hh, w, hh), along(0.0, 0.0, 0.0, hh)]
}
/// (mean, smallest) camera px per cell over the four edges.
fn cell_px(h: &Homography, grid: &FlatGrid) -> (f32, f32) {
    let e = edges(h, grid);
    let p = grid.format().pitch as f32;
    (e.iter().sum::<f32>() / 4.0 * p, e.iter().copied().fold(f32::MAX, f32::min) * p)
}

/// A degradation: the tile's mean camera px per cell, then yaw / pitch (perspective about the tile's centre,
/// on the optical axis) and roll, in degrees; `soft`: a blur after, the kernel [a, 1 - 2a, a] each way with a
/// = soft / 20 (a camera a little out of focus, or moving; soft 5 is [1 2 1] / 4).
#[derive(Clone, Copy, Debug)]
struct Deg { px: f32, yaw: f32, pitch: f32, roll: f32, soft: u8 }

/// The input -> output camera mapping of a degradation for a tile seen through `hc`: the tile's centre to the
/// output's centre, scaled by `s`, rolled, then seen in perspective.
fn mapping(hc: &Homography, grid: &FlatGrid, d: Deg, s: f64, out: (u32, u32)) -> [f64; 9] {
    let (cx, cy) = hc.apply(grid.width_px() as f32 / 2.0, grid.height_px() as f32 / 2.0);
    let t_in = [1.0, 0.0, -cx as f64, 0.0, 1.0, -cy as f64, 0.0, 0.0, 1.0];
    let sc = [s, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, 1.0];
    let (r, y, p) = ((d.roll as f64).to_radians(), (d.yaw as f64).to_radians(), (d.pitch as f64).to_radians());
    let roll = [r.cos(), -r.sin(), 0.0, r.sin(), r.cos(), 0.0, 0.0, 0.0, 1.0];
    // a plane turned by yaw about the vertical axis at distance f: x' = f x cos / (f + x sin)
    let yaw = [y.cos(), 0.0, 0.0, 0.0, 1.0, 0.0, y.sin() / FOCAL_PX, 0.0, 1.0];
    let pitch = [1.0, 0.0, 0.0, 0.0, p.cos(), 0.0, 0.0, p.sin() / FOCAL_PX, 1.0];
    let t_out = [1.0, 0.0, out.0 as f64 / 2.0, 0.0, 1.0, out.1 as f64 / 2.0, 0.0, 0.0, 1.0];
    // applied in order: scale, roll, yaw, pitch, then the move to the output's centre
    [sc, roll, yaw, pitch, t_out].iter().fold(t_in, |acc, m| mul(m, &acc))
}

/// The capture as a camera at the degradation would see it: each output pixel the mean of the input (constant
/// over each of its pixels, as the sensor integrated it) over the output pixel's footprint (6x6 points),
/// outside the input the input's median. The canvas is the camera's frame, or larger where the tile at that
/// size and tilt would not fit in it (the reader reads what is in view; framing is the guidance's job).
/// Returns the image and the input -> output mapping.
fn degrade(y: &GrayImage, hc: &Homography, grid: &FlatGrid, d: Deg) -> (GrayImage, [f64; 9]) {
    let out = (y.width, y.height);
    // the scale for the wanted mean cell size (the perspective changes the mean a little: settle it)
    let mut s = if d.px <= 0.0 { 1.0 } else { (d.px / cell_px(hc, grid).0) as f64 };
    // px 0: the scale stays 1 (only the move of the tile's centre to the output's centre)
    for _ in 0..if d.px <= 0.0 { 0 } else { 4 } {
        let f = mapping(hc, grid, d, s, out);
        let got = cell_px(&to_h(&mul(&f, &of_h(hc))), grid).0;
        s *= (d.px / got) as f64;
    }
    // a canvas that holds the whole tile and a margin
    let f0 = mapping(hc, grid, d, s, (0, 0));
    let (w, hh) = (grid.width_px() as f32, grid.height_px() as f32);
    let ext = [(0.0, 0.0), (w, 0.0), (0.0, hh), (w, hh)].iter().map(|&(x, yy)| { let (a, b) = hc.apply(x, yy); to_h(&f0).apply(a, b) }).fold((0f32, 0f32), |m, (a, b)| (m.0.max(a.abs()), m.1.max(b.abs())));
    let out = (out.0.max((2.0 * ext.0 + 160.0) as u32), out.1.max((2.0 * ext.1 + 160.0) as u32));
    let f = mapping(hc, grid, d, s, out);
    let inv = to_h(&f).inverse().unwrap();
    let mut sorted = y.data.clone();
    sorted.sort_unstable();
    let fill = sorted[sorted.len() / 2] as f32;
    let mut o = GrayImage::new(out.0, out.1);
    const K: usize = 6;
    for oy in 0..out.1 {
        for ox in 0..out.0 {
            let mut acc = 0f32;
            for j in 0..K { for i in 0..K {
                let (qx, qy) = (ox as f32 + (i as f32 + 0.5) / K as f32, oy as f32 + (j as f32 + 0.5) / K as f32);
                let (sx, sy) = inv.apply(qx, qy);
                acc += if sx < 0.0 || sy < 0.0 || sx >= y.width as f32 || sy >= y.height as f32 { fill } else { y.data[sy as usize * y.width as usize + sx as usize] as f32 };
            } }
            o.data[(oy * out.0 + ox) as usize] = (acc / (K * K) as f32).round().clamp(0.0, 255.0) as u8;
        }
    }
    if d.soft > 0 {
        let (w, h) = (o.width as i64, o.height as i64);
        let a = d.soft as f32 / 20.0;
        let k = [(-1i64, a), (0, 1.0 - 2.0 * a), (1, a)];
        let src = o.data.clone();
        let at = |x: i64, y: i64| src[(y.clamp(0, h - 1) * w + x.clamp(0, w - 1)) as usize] as f32;
        for y in 0..h { for x in 0..w {
            let mut v = 0f32;
            for (dy, ky) in k { for (dx, kx) in k { v += at(x + dx, y + dy) * ky * kx; } }
            o.data[(y * w + x) as usize] = v.round().clamp(0.0, 255.0) as u8;
        } }
    }
    (o, f)
}

/// The captures of `SEQ` (their millisecond stamps) that read whole cold before the equaliser, in capture
/// order: every curve and the gate start from these (the equaliser reads 38 of the 40 whole: every b/w one).
const SOURCES: &[u64] = &[
    45382350, 45382587, 45382616, 45382648, 45382696, 45382719, 45382746, 45382780, 45382822, 45382861,
    45382953, 45382987, 45383012, 45383053, 45383081, 45383111, 45383146, 45383180, 45383211, 45383256,
    45383291, 45383313, 45383343, 45383379, 45383414, 45383446, 45383488, 45383522, 45383547, 45383579,
];

/// A real capture read whole: its grid, header, homography, shifts, and the true value of every cell.
struct Source { stem: String, y: GrayImage, read: FlatRead, truth: Vec<u8> }

/// The b/w captures of `SEQ` the reader before the equaliser did not read whole cold (the five that came out
/// short in the tracked chain, 107-110 of 111, among them); the equaliser reads them whole, and the gate
/// degrades them with the others.
const HARD: &[u64] = &[45382283, 45382312, 45382385, 45382414, 45382444, 45382484, 45382521, 45382545];

fn sources(every: usize, max: usize) -> Vec<Source> { sources_of(SOURCES, every, max) }

fn sources_of(list: &[u64], every: usize, max: usize) -> Vec<Source> {
    let Some(dir) = gate_dir(SEQ) else { return Vec::new() };
    let dec = FlatDecoder::new();
    let mut out = Vec::new();
    let listed = captures(&dir).into_iter().filter(|c| c.stem.split('_').nth(1).and_then(|t| t.parse().ok()).is_some_and(|t: u64| list.contains(&t)));
    for cap in listed.step_by(every) {
        let y = cap.luma();
        let Ok(mut frs) = dec.decode_luma(&y, None) else { continue };
        let fr = frs.remove(0);
        let p = fr.payload();
        assert!(p.codewords_failed == 0 && p.codewords_missing == 0, "{}: listed as read whole", cap.stem);
        let truth = compose(fr.grid.clone(), fr.header, &p.bytes).values;
        out.push(Source { stem: cap.stem.clone(), y, read: fr, truth });
        if out.len() == max { break; }
    }
    out
}

/// One degraded read's outcome.
#[derive(Default, Debug, Clone)]
struct Outcome { ok: usize, cell_err: f32, conf_err: f32, align_med: f32, align_p95: f32, h_err: f32 }

fn block_centres(grid: &FlatGrid) -> Vec<(f32, f32)> {
    let (bw, bh) = grid.blocks();
    let p = grid.format().pitch as f32;
    let (x0, y0) = grid.cell_center(0, 0);
    (0..bw * bh).map(|b| { let (bx, by) = (b % bw, b / bw); (x0 + (bx as f32 * 8.0 + 3.5) * p, y0 + (by as f32 * 8.0 + 3.5) * p) }).collect()
}

/// `fr` against the truth: codewords, data cells wrong (all, and those with a margin above the erasure
/// margin), per-block alignment error against the true positions `pos` (camera px), the homography's worst
/// corner against the true one.
fn score(fr: &FlatRead, truth: &[u8], pos: &[(f32, f32)], h_true: &Homography) -> Outcome {
    let p = fr.payload();
    let g = &fr.grid;
    let n = g.used_data_cells();
    let (mut bad, mut conf) = (0usize, 0usize);
    for i in 0..n as u32 {
        let (c, r) = g.data_position(i);
        let (v, m) = fr.reads.at(c, r);
        if v != truth[(r * g.cols() + c) as usize] { bad += 1; if m >= lensbolt_core::flat::ERASURE_MARGIN { conf += 1; } }
    }
    let mut errs: Vec<f32> = block_centres(g).iter().zip(&fr.reads.shifts).zip(pos).map(|((&(x, y), &(dx, dy)), &(tx, ty))| {
        let (cx, cy) = fr.homography.apply(x, y);
        (cx + dx - tx).hypot(cy + dy - ty)
    }).collect();
    errs.sort_by(f32::total_cmp);
    let (w, hh) = (g.width_px() as f32, g.height_px() as f32);
    let h_err = [(0.0, 0.0), (w, 0.0), (0.0, hh), (w, hh)].iter().map(|&(x, y)| { let (a, b) = (fr.homography.apply(x, y), h_true.apply(x, y)); (a.0 - b.0).hypot(a.1 - b.1) }).fold(0f32, f32::max);
    Outcome { ok: p.codewords_ok, cell_err: bad as f32 / n as f32 * 100.0, conf_err: conf as f32 / n as f32 * 100.0, align_med: errs[errs.len() / 2], align_p95: errs[errs.len() * 95 / 100], h_err }
}

/// Measurement (prints the curve): `cargo test --release --test fhd2px_floor measure -- --ignored --nocapture`.
/// `LB_FLOOR_GRID` = "px,yaw,pitch,roll;..." overrides the grid; `LB_FLOOR_EVERY` / `LB_FLOOR_MAX` the sources.
#[test]
#[ignore]
fn measure() {
    let every = std::env::var("LB_FLOOR_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let max = std::env::var("LB_FLOOR_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
    let srcs = sources(every, max);
    if srcs.is_empty() { eprintln!("skip: no captures in {SEQ}"); return; }
    for s in &srcs { let (m, lo) = cell_px(&s.read.homography, &s.read.grid); eprintln!("source {}: {m:.2} px per cell (smallest edge {lo:.2})", s.stem); }
    let grid: Vec<Deg> = match std::env::var("LB_FLOOR_GRID") {
        Ok(g) => g.split(';').map(|t| { let v: Vec<f32> = t.split(',').map(|x| x.trim().parse().unwrap()).collect(); Deg { px: v[0], yaw: v[1], pitch: v[2], roll: v[3], soft: v.get(4).map_or(0, |&s| s as u8) } }).collect(),
        Err(_) => {
            let mut v: Vec<Deg> = [1.7, 1.6, 1.55, 1.5, 1.45, 1.4, 1.3, 1.2].iter().map(|&px| Deg { px, yaw: 0.0, pitch: 0.0, roll: 0.0, soft: 0 }).collect();
            for a in [10.0, 20.0, 30.0] { v.push(Deg { px: 1.7, yaw: a, pitch: 0.0, roll: 0.0, soft: 0 }); }
            for a in [10.0, 20.0, 30.0] { v.push(Deg { px: 1.7, yaw: 0.0, pitch: a, roll: 0.0, soft: 0 }); }
            for a in [5.0, 15.0] { v.push(Deg { px: 1.7, yaw: 0.0, pitch: 0.0, roll: a, soft: 0 }); }
            v.push(Deg { px: 1.6, yaw: 20.0, pitch: 0.0, roll: 0.0, soft: 0 });
            v.push(Deg { px: 1.5, yaw: 20.0, pitch: 0.0, roll: 0.0, soft: 0 });
            // a little out of focus: soft 2 at 1.7 / 1.5 / 1.4 px, soft 1 at 1.5 px
            for (px, soft) in [(1.7, 2), (1.5, 1), (1.5, 2), (1.4, 2)] { v.push(Deg { px, yaw: 0.0, pitch: 0.0, roll: 0.0, soft }); }
            v
        }
    };
    let dec = FlatDecoder::new();
    for d in grid {
        let (mut ok, mut all, mut fails) = (0usize, 0usize, std::collections::BTreeMap::<String, usize>::new());
        let (mut ok_t, mut all_t) = (0usize, 0usize);
        let (mut ok_o, mut cell_o) = (0usize, 0f32);
        let mut per_frame: Vec<usize> = Vec::new();
        let mut rows: Vec<Outcome> = Vec::new();
        let mut hint: Option<FlatHint> = None;
        let mut min_px = f32::MAX;
        for s in &srcs {
            let hc = s.read.homography;
            let (img, f) = degrade(&s.y, &hc, &s.read.grid, d);
            let h_true = to_h(&mul(&f, &of_h(&hc)));
            min_px = min_px.min(cell_px(&h_true, &s.read.grid).1);
            let fh = to_h(&f);
            let pos: Vec<(f32, f32)> = block_centres(&s.read.grid).iter().zip(&s.read.reads.shifts).map(|(&(x, y), &(dx, dy))| { let (cx, cy) = hc.apply(x, y); fh.apply(cx + dx, cy + dy) }).collect();
            all += 111;
            match dec.decode_luma(&img, None) {
                Ok(frs) => { let o = score(&frs[0], &s.truth, &pos, &h_true); ok += o.ok; per_frame.push(o.ok); rows.push(o); }
                Err(e) => { per_frame.push(0); *fails.entry(format!("{e:?}").chars().take(12).collect()).or_default() += 1; }
            }
            // tracked as on the phone: the previous degraded capture's read as the hint
            all_t += 111;
            match dec.decode_luma(&img, hint.as_ref()) {
                Ok(frs) => { ok_t += frs[0].payload().codewords_ok; hint = Some(frs[0].hint()); }
                Err(_) => { hint = None; }
            }
            // oracle: the true homography and the true shift field (tracking start), the reader's search after it
            let oshift: Vec<(f32, f32)> = block_centres(&s.read.grid).iter().zip(&pos).map(|(&(x, y), &(tx, ty))| { let (cx, cy) = h_true.apply(x, y); (tx - cx, ty - cy) }).collect();
            if let Some(reads) = lensbolt_core::flat::fast::read_cells_luma(&img, &s.read.grid, &h_true, s.read.header.frame_seq & 3, Some(&oshift)) {
                let fr = FlatRead { grid: s.read.grid.clone(), header: s.read.header, homography: h_true, reads, decode_ms: 0.0 };
                let o = score(&fr, &s.truth, &pos, &h_true);
                ok_o += o.ok; cell_o += o.cell_err;
            }
        }
        let n = rows.len().max(1) as f32;
        let mean = |f: fn(&Outcome) -> f32| rows.iter().map(f).sum::<f32>() / n;
        eprintln!("px {:.2} yaw {:>2} pitch {:>2} roll {:>2} soft {} (smallest edge {min_px:.2}): cold {ok}/{all} = {:.1}%  tracked {ok_t}/{all_t} = {:.1}%  oracle {ok_o}/{all} = {:.1}% (cells {:.2}%)  | decoded {} fails {fails:?} cells {:.2}% confident {:.2}% align med {:.2} p95 {:.2} px, H corner err {:.2} px",
            d.px, d.yaw, d.pitch, d.roll, d.soft, ok as f32 / all as f32 * 100.0, ok_t as f32 / all_t as f32 * 100.0, ok_o as f32 / all as f32 * 100.0, cell_o / srcs.len() as f32,
            rows.len(), mean(|o| o.cell_err), mean(|o| o.conf_err), mean(|o| o.align_med), mean(|o| o.align_p95), mean(|o| o.h_err));
        per_frame.sort_unstable();
        eprintln!("    cold codewords per frame (sorted): {per_frame:?}");
        if std::env::var_os("LB_FLOOR_ROWS").is_some() { for o in &rows { eprintln!("    {o:?}"); } }
    }
}

/// (codewords read cold, codewords read tracked, all) over `srcs` at `d`; tracked = the previous degraded
/// capture's read as the hint, as on the phone.
fn read_at(srcs: &[Source], d: Deg) -> (usize, usize, usize) {
    let dec = FlatDecoder::new();
    let (mut cold, mut tracked, mut all) = (0, 0, 0);
    let mut hint: Option<FlatHint> = None;
    for s in srcs {
        let (img, _) = degrade(&s.y, &s.read.homography, &s.read.grid, d);
        all += s.read.grid.codewords();
        if let Ok(frs) = dec.decode_luma(&img, None) { cold += frs[0].payload().codewords_ok; }
        match dec.decode_luma(&img, hint.as_ref()) {
            Ok(frs) => { tracked += frs[0].payload().codewords_ok; hint = Some(frs[0].hint()); }
            Err(_) => hint = None,
        }
    }
    (cold, tracked, all)
}

/// The gate on sixteen real captures (the first eight of `SOURCES` and the eight of `HARD`), degraded: the share of
/// codewords read, cold and tracked, at each size and tilt (`MIN_SHARE`). Without the captures it says so
/// and passes.
#[test]
fn fhd2px_reads_small_and_tilted_tiles() {
    let mut srcs = sources(1, 8);
    srcs.extend(sources_of(HARD, 1, HARD.len()));
    if srcs.is_empty() { eprintln!("skip: no captures in {SEQ}"); return; }
    let mut short = Vec::new();
    for &(d, min) in MIN_SHARE {
        let (cold, tracked, all) = read_at(&srcs, d);
        let (c, t) = (cold as f32 / all as f32, tracked as f32 / all as f32);
        eprintln!("{d:?}: cold {cold}/{all} = {:.1}%, tracked {tracked}/{all} = {:.1}% (at least {:.0}%)", c * 100.0, t * 100.0, min * 100.0);
        if c < min || t < min { short.push(format!("{d:?}: cold {:.1}% tracked {:.1}% < {:.0}%", c * 100.0, t * 100.0, min * 100.0)); }
    }
    assert!(short.is_empty(), "codewords short of the floor: {short:?}");
}

/// (degradation, smallest share of codewords read cold and tracked).
const MIN_SHARE: &[(Deg, f32)] = &[
    (Deg { px: 1.45, yaw: 0.0, pitch: 0.0, roll: 0.0, soft: 0 }, 0.95),
    (Deg { px: 1.6, yaw: 20.0, pitch: 0.0, roll: 0.0, soft: 0 }, 0.95),
    (Deg { px: 1.5, yaw: 0.0, pitch: 0.0, roll: 0.0, soft: 2 }, 0.9),
];

/// `img` under glare: a lamp's reflection, a soft disc of light added (amplitude `amp` at its centre, falling
/// off as a Gaussian of radius `r` camera px) a third of the way across the frame, clipped at 255.
fn glare(img: &GrayImage, amp: f32, r: f32) -> GrayImage {
    let mut o = img.clone();
    let (cx, cy) = (img.width as f32 * 0.4, img.height as f32 * 0.45);
    for y in 0..img.height { for x in 0..img.width {
        let d2 = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
        let i = (y * img.width + x) as usize;
        o.data[i] = (img.data[i] as f32 + amp * (-d2 / (2.0 * r * r)).exp()).round().min(255.0) as u8;
    } }
    o
}

/// The equaliser under glare (a 2 px tile at 1.6 camera px per cell with a lamp's reflection over part of it):
/// its regions fit their own levels, the pattern decision's pilots under the glare do not vote. Codewords cold /
/// tracked at least the share `GLARE` gives.
#[test]
fn fhd2px_reads_under_glare() {
    let srcs = sources(1, 8);
    if srcs.is_empty() { eprintln!("skip: no captures in {SEQ}"); return; }
    let dec = FlatDecoder::new();
    let d = Deg { px: 1.6, yaw: 0.0, pitch: 0.0, roll: 0.0, soft: 0 };
    let mut short = Vec::new();
    for &(amp, r, min) in GLARE {
        let (mut cold, mut tracked, mut all) = (0usize, 0usize, 0usize);
        let mut hint: Option<FlatHint> = None;
        for s in &srcs {
            let (img, _) = degrade(&s.y, &s.read.homography, &s.read.grid, d);
            let img = glare(&img, amp, r);
            all += s.read.grid.codewords();
            if let Ok(frs) = dec.decode_luma(&img, None) { cold += frs[0].payload().codewords_ok; }
            match dec.decode_luma(&img, hint.as_ref()) {
                Ok(frs) => { tracked += frs[0].payload().codewords_ok; hint = Some(frs[0].hint()); }
                Err(_) => hint = None,
            }
        }
        let (c, t) = (cold as f32 / all as f32, tracked as f32 / all as f32);
        eprintln!("glare +{amp} r {r}: cold {cold}/{all} = {:.1}%, tracked {tracked}/{all} = {:.1}%", c * 100.0, t * 100.0);
        if c < min || t < min { short.push(format!("glare +{amp} r {r}: cold {:.1}% tracked {:.1}% < {:.0}%", c * 100.0, t * 100.0, min * 100.0)); }
    }
    assert!(short.is_empty(), "{short:?}");
}

/// Glare (amplitude, radius camera px) and the smallest share of codewords read under it, cold and tracked.
/// Measured 2026-11-02 (8 sources at 1.6 px): +80 r 150 99.2 / 99.9% (the pattern decision alone 47.6 / 76.4%),
/// +120 r 100 70.8 / 86.5% (20.2 / 45.5%), +160 r 80 41.2 / 62.0% (11.8 / 26.1%); +160 r 120 clips a twelfth of
/// the tile white and reads 0.2% with or without the equaliser (past what the codewords correct: not gated).
/// The equaliser fed every sample on its block's black..white axis read 98.5 / 99.7, 73.8 / 85.9 and 47.2 /
/// 66.7%: no clear gain, so the model stays on the samples.
const GLARE: &[(f32, f32, f32)] = &[(80.0, 150.0, 0.95), (120.0, 100.0, 0.6), (160.0, 80.0, 0.3)];

/// `img` as a camera without stabilisation sees it while the hand moves: a linear smear of `smear` camera px
/// along x over the exposure (the mean of 9 samples across it), and a rolling shutter's shear: each row moved
/// along x in proportion to its height, `shear` camera px from the top of the frame to the bottom.
fn motion(img: &GrayImage, smear: f32, shear: f32) -> GrayImage {
    let mut o = img.clone();
    let (w, h) = (img.width as f32, img.height as f32);
    for y in 0..img.height { for x in 0..img.width {
        let base = x as f32 + 0.5 - shear * (y as f32 / h - 0.5);
        let mut acc = 0f32;
        for k in 0..9 {
            let sx = (base + smear * (k as f32 / 8.0 - 0.5)).clamp(0.0, w - 1.0);
            acc += img.sample(sx - 0.5, y as f32);
        }
        o.data[(y * img.width + x) as usize] = (acc / 9.0).round() as u8;
    } }
    o
}

/// Motion without stabilisation: real captures at 1.7 and 1.5 camera px per cell smeared 1-3 camera px and
/// sheared 1-2 px over the frame (`MOTION`); codewords cold / tracked at least the share given.
#[test]
fn fhd2px_reads_through_motion() {
    let srcs = sources(1, 8);
    if srcs.is_empty() { eprintln!("skip: no captures in {SEQ}"); return; }
    let dec = FlatDecoder::new();
    let mut short = Vec::new();
    for &(px, smear, shear, min) in MOTION {
        let d = Deg { px, yaw: 0.0, pitch: 0.0, roll: 0.0, soft: 0 };
        let (mut cold, mut tracked, mut all) = (0usize, 0usize, 0usize);
        let mut hint: Option<FlatHint> = None;
        for s in &srcs {
            let (img, _) = degrade(&s.y, &s.read.homography, &s.read.grid, d);
            let img = motion(&img, smear, shear);
            all += s.read.grid.codewords();
            if let Ok(frs) = dec.decode_luma(&img, None) { cold += frs[0].payload().codewords_ok; }
            match dec.decode_luma(&img, hint.as_ref()) {
                Ok(frs) => { tracked += frs[0].payload().codewords_ok; hint = Some(frs[0].hint()); }
                Err(_) => hint = None,
            }
        }
        let (c, t) = (cold as f32 / all as f32, tracked as f32 / all as f32);
        eprintln!("{px} px smear {smear} shear {shear}: cold {cold}/{all} = {:.1}%, tracked {tracked}/{all} = {:.1}%", c * 100.0, t * 100.0);
        if c < min || t < min { short.push(format!("{px} px smear {smear} shear {shear}: cold {:.1}% tracked {:.1}% < {:.0}%", c * 100.0, t * 100.0, min * 100.0)); }
    }
    assert!(short.is_empty(), "{short:?}");
}

/// (camera px per cell, smear px, shear px, smallest share of codewords read cold and tracked). Measured
/// 2026-11-02 (8 sources), cold / tracked: 1.7 px smear 1 100 / 100%, smear 2 47.3 / 70.6%, smear 3 0 / 0%;
/// shear 2 100 / 100%; smear 2 + shear 2 64.3 / 83.3%; 1.5 px smear 1 + shear 1 97.9 / 98.9%, smear 2 + shear
/// 1 0.1 / 1.2%. The shear costs nothing (a full perspective per capture and a shift per block follow it); a
/// smear of 2 camera px is the cliff. The hand moved the tile 1.2-1.45 px per 33 ms capture (median; 90th
/// percentile 2.4-2.7 px) in the 2026-10-30 and 2026-11-02 runs: with an exposure as long as the frame, that
/// is the smear (the app keeps auto exposure: a fixed short one needs the tile's levels to set it; the live set
/// read whole at that drift). Gated: what reads.
const MOTION: &[(f32, f32, f32, f32)] = &[(1.7, 1.0, 0.0, 0.95), (1.7, 0.0, 2.0, 0.95), (1.5, 1.0, 1.0, 0.9)];

/// The live hint's words read better on the benchmark: at 1.15 camera px per cell (a little soft) 2 px b/w reads
/// under half its codewords, the hint's prediction at that size offers 3 px b/w, and the same hold at 3 px (the
/// cells 1.5x larger: the same capture at 1.725 px per cell, the scaling the hint assumes) reads at least 95%.
/// Without the captures it says so and passes.
#[test]
fn hint_words_read_better_on_the_benchmark() {
    use lensbolt_core::tune::hint::coarser_bw;
    let srcs = sources(1, 8);
    if srcs.is_empty() { eprintln!("skip: no captures in {SEQ}"); return; }
    let at = |px: f32| Deg { px, yaw: 0.0, pitch: 0.0, roll: 0.0, soft: 2 };
    let (_, t2, all) = read_at(&srcs, at(1.15));
    let share2 = t2 as f32 / all as f32;
    eprintln!("2 px at 1.15 px per cell: tracked {t2}/{all} = {:.1}%", share2 * 100.0);
    assert!(share2 < 0.5, "2 px reads {share2} at 1.15 px per cell");
    assert_eq!(coarser_bw(2, 1.15), Some(3), "the hint offers 3 px b/w");
    let (_, t3, all) = read_at(&srcs, at(1.725));
    let share3 = t3 as f32 / all as f32;
    eprintln!("3 px at that hold (1.725 px per cell): tracked {t3}/{all} = {:.1}%", share3 * 100.0);
    assert!(share3 >= 0.95, "3 px reads {share3} at that hold");
}
