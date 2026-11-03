//! The sharpness gate (`FlatDecoder::gate`, the first stage of every decode) on real captures: it decides as
//! the full-frame score does, its threshold stays within 2 of the full-frame two-means value, and at 12.6 MP
//! it costs at most 10 ms on one core. Then synthetic frames sharp in only a part of the view: the gate must
//! not pass one the full-frame score rejects (it would cost a full decode), nor reject one it passes.
//! Fixtures: the three tiers of `phone_max1px.rs` and the 3072x3072 lab burst of `fast_ref.rs`; absent
//! fixtures are skipped with a message.
use lensbolt_core::detect::sharpness as sample;
use lensbolt_core::detect::{sharpness, two_means_threshold};
use lensbolt_core::flat::{DecodeError, FlatDecoder};
use lensbolt_core::image::{GrayImage, RgbImage};
use std::path::PathBuf;
use std::time::Instant;

mod common;
use common::{capture_dir, captures};

/// One measurement at a time: cargo runs a binary's tests in parallel.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

const TIERS: [&str; 3] = ["test/phone_max1px_live/dump", "test/phone_fhd2px_live/dump", "test/phone_fhd"];

/// The full-frame score the gate decided on before it sampled: the mean absolute 4-neighbour Laplacian over
/// the 4x box-downscaled frame (box means truncated).
fn reference_sharpness(gray: &GrayImage) -> f32 {
    let g = gray.downscale(4);
    if g.width < 3 || g.height < 3 { return 0.0; }
    let mut acc = 0u64; let mut n = 0u64;
    for y in 1..g.height - 1 { for x in 1..g.width - 1 {
        let c = g.get(x, y) as i32 * 4;
        let l = c - g.get(x - 1, y) as i32 - g.get(x + 1, y) as i32 - g.get(x, y - 1) as i32 - g.get(x, y + 1) as i32;
        acc += l.unsigned_abs() as u64; n += 1;
    } }
    acc as f32 / n as f32
}

/// The full-frame threshold the gate returned before it sampled: Otsu (maximum between-class variance) over
/// a one-table histogram of every pixel, written out here rather than calling the library's.
fn reference_threshold(data: &[u8]) -> u8 {
    let mut hist = [0u64; 256];
    for &v in data { hist[v as usize] += 1; }
    let total = data.len().max(1) as f64;
    let sum_all: f64 = hist.iter().enumerate().map(|(v, &n)| v as f64 * n as f64).sum();
    let (mut w_b, mut sum_b, mut best, mut best_t) = (0f64, 0f64, -1f64, 0usize);
    for t in 0..256 {
        w_b += hist[t] as f64;
        if w_b == 0.0 { continue; }
        let w_f = total - w_b;
        if w_f == 0.0 { break; }
        sum_b += t as f64 * hist[t] as f64;
        let (m_b, m_f) = (sum_b / w_b, (sum_all - sum_b) / w_f);
        let between = w_b * w_f * (m_b - m_f) * (m_b - m_f);
        if between > best { best = between; best_t = t; }
    }
    best_t as u8
}

/// The centre `w` x `h` of `g`: frames of the sizes a browser receiver gets, cut from a real capture.
fn crop(g: &GrayImage, w: u32, h: u32) -> GrayImage {
    let (x0, y0) = ((g.width - w) / 2, (g.height - h) / 2);
    let mut out = GrayImage::new(w, h);
    for y in 0..h { let o = ((y0 + y) * g.width + x0) as usize; out.data[(y * w) as usize..((y + 1) * w) as usize].copy_from_slice(&g.data[o..o + w as usize]); }
    out
}

/// Browser receiver frame sizes: its own camera request (2560x1920) and the 8 MP frames a phone's browser
/// delivers (portrait and landscape).
const BROWSER_SIZES: [(u32, u32); 3] = [(2560, 1920), (2448, 3264), (3264, 2448)];

/// Every stored real frame as the gate sees it: the Y plane (`decode_luma`) and the RGB frame's grey
/// (`decode_tracked`, `identify`); the 1 px captures also cut to the browser receiver's frame sizes.
fn frames() -> Vec<(String, GrayImage)> {
    let mut out = Vec::new();
    for rel in TIERS {
        let Some(dir) = capture_dir(rel) else { eprintln!("skip {rel}: no captures on this machine"); continue };
        for c in captures(&dir) {
            let y = c.luma();
            for (w, h) in BROWSER_SIZES {
                if w <= y.width && h <= y.height { out.push((format!("{} (y, centre {w}x{h})", c.stem), crop(&y, w, h))); }
            }
            out.push((format!("{} (y)", c.stem), y));
            out.push((format!("{} (rgb)", c.stem), c.rgb().to_gray()));
        }
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let burst = root.join("test/lab/burst_00_1793117880");
    let mut pngs: Vec<PathBuf> = std::fs::read_dir(&burst).map(|d| d.filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "png")).collect()).unwrap_or_default();
    pngs.sort();
    if pngs.is_empty() { eprintln!("skip {}: not stored", burst.display()); }
    for p in pngs { out.push((p.file_name().unwrap().to_string_lossy().into_owned(), RgbImage::load_png(&p).unwrap().to_gray())); }
    out
}

#[test]
fn gate_decides_as_the_full_frame_score_on_real_captures() {
    let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dec = FlatDecoder::new();
    let fr = frames();
    if fr.is_empty() { eprintln!("skip: no real captures on this machine"); return; }
    let mut worst = 0i32;
    for (name, g) in &fr {
        let s_ref = reference_sharpness(g);
        // the full-frame score itself is unchanged (the CLI and the receiver's failure report print it)
        assert_eq!(sharpness(g), s_ref, "{name}: full-frame sharpness changed");
        let t_ref = reference_threshold(&g.data);
        // the library's full-frame threshold (four count tables) is the reference's
        assert_eq!(two_means_threshold(&g.data), t_ref, "{name}: full-frame threshold changed");
        let want_ok = s_ref >= dec.min_sharpness;
        let got = dec.gate(g);
        match &got {
            Ok(t) => {
                let d = *t as i32 - t_ref as i32;
                worst = worst.max(d.abs());
                eprintln!("{name}: {}x{} sharpness {s_ref:.2} threshold {t_ref} -> {t} ({d:+})", g.width, g.height);
                assert!(d.abs() <= 2, "{name}: threshold {t} against the full-frame {t_ref}");
                // up to 8.4 MP (Full HD, the browser's 2560x1920 and 3264x2448) the gate counts every pixel
                if (g.width as usize) * (g.height as usize) <= 1 << 23 { assert_eq!(d, 0, "{name}: a frame up to 8.4 MP must take the whole-frame threshold"); }
            }
            Err(e) => eprintln!("{name}: {}x{} sharpness {s_ref:.2} -> {e:?}", g.width, g.height),
        }
        assert_eq!(got.is_ok(), want_ok, "{name}: full-frame sharpness {s_ref:.3} (gate {}) but the gate says {got:?}", dec.min_sharpness);
        if let Err(DecodeError::TooBlurry(s)) = got { assert!(s < dec.min_sharpness, "{name}: TooBlurry({s})"); }
    }
    eprintln!("{} frames: same decisions, threshold within {worst} of the full frame", fr.len());
}

/// Fastest of five gate calls on one frame, in ms.
fn gate_ms(dec: &FlatDecoder, g: &GrayImage) -> f64 {
    (0..5).map(|_| { let t = Instant::now(); let _ = std::hint::black_box(dec.gate(std::hint::black_box(g))); t.elapsed().as_secs_f64() * 1e3 })
        .fold(f64::MAX, f64::min)
}

#[test]
fn gate_costs_at_most_10_ms_at_12_6_mp() {
    let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("LB_THREADS", "1");
    let dec = FlatDecoder::new();
    let mut big: Option<f64> = None;
    for rel in TIERS {
        let Some(dir) = capture_dir(rel) else { eprintln!("skip {rel}: no captures on this machine"); continue };
        let caps = captures(&dir);
        let g = caps[0].luma();
        let ms = gate_ms(&dec, &g);
        eprintln!("gate {}x{} ({}): {ms:.2} ms, fastest of five", g.width, g.height, caps[0].stem);
        if g.width as u64 * g.height as u64 >= 4096 * 3072 {
            big = Some(ms);
            // the browser receiver's frame sizes, cut from the same capture (whole histogram)
            for (w, h) in BROWSER_SIZES {
                // the portrait 8 MP frame does not fit in a landscape capture: the landscape one has its cost
                if w > g.width || h > g.height { continue; }
                let c = crop(&g, w, h);
                eprintln!("gate {w}x{h} (centre of {}): {:.2} ms, fastest of five", caps[0].stem, gate_ms(&dec, &c));
            }
        }
    }
    match big {
        Some(ms) => assert!(ms <= 10.0, "gate {ms:.2} ms at 4096x3072 (budget 10 ms)"),
        None => eprintln!("skip: no 4096x3072 capture on this machine"),
    }
}

/// xorshift: a fixed pseudo-random byte stream.
fn noise(seed: &mut u64) -> u8 { *seed ^= *seed << 13; *seed ^= *seed >> 7; *seed ^= *seed << 17; (*seed >> 24) as u8 }

/// A 4096x3072 frame of flat grey with sharp random 1 px content in `rect` (x0, y0, x1, y1).
fn partly_sharp(rect: (u32, u32, u32, u32)) -> GrayImage {
    let (w, h) = (4096u32, 3072u32);
    let mut g = GrayImage::new(w, h);
    g.data.fill(128);
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    for y in rect.1..rect.3 { for x in rect.0..rect.2 { g.data[(y * w + x) as usize] = noise(&mut seed); } }
    g
}

/// Review focus: a capture sharp in only a part of the view (the tile half out of view, a defocused band)
/// must fail fast where the full-frame score fails, and pass where it passes. The gate's minimum is set
/// relative to each frame's full-frame score, from half to twice it. Sharp bands from 64 px (2% of the
/// frame height) to half the frame, at every 4 px offset through one 32 px sampling period, the tile in a
/// corner or a strip at an edge, a small patch in the middle.
#[test]
fn frame_sharp_in_part_decides_as_the_full_frame_score() {
    let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut regions: Vec<(u32, u32, u32, u32)> = Vec::new();
    for hgt in [64u32, 96, 192, 384, 768, 1536] { for off in (0..32).step_by(4) { let y0 = 1000 + off; regions.push((0, y0, 4096, y0 + hgt)); } }
    // the tile in a corner, the rest of the view out of focus or off screen
    regions.extend([(0, 0, 1024, 768), (3072, 2304, 4096, 3072), (0, 2560, 4096, 3072), (3584, 0, 4096, 3072), (2000, 1500, 2600, 1950)]);
    let mut worst = 0f32;
    for r in regions {
        let g = partly_sharp(r);
        let full = sharpness(&g);
        assert!(full > 0.0);
        let sampled = sample::sharpness_rows(&g, sample::sample_step(&g));
        worst = worst.max((sampled / full - 1.0).abs());
        // 15% either side: the full frame decides (inside the gate's margin); 30% and 2x: the sample does
        for (ratio, pass) in [(0.5f32, false), (0.7, false), (0.85, false), (1.15, true), (1.3, true), (2.0, true)] {
            let mut dec = FlatDecoder::new(); dec.min_sharpness = full / ratio;
            let got = dec.gate(&g);
            assert_eq!(got.is_ok(), pass, "region {r:?}: full-frame {full:.3}, sampled {sampled:.3}, gate minimum {:.3}: {got:?}", dec.min_sharpness);
        }
    }
    eprintln!("frames sharp in part: the row sample within {:.1}% of the full-frame score", worst * 100.0);
}
