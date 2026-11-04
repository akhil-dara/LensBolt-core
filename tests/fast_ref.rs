//! The fast luma reader against the reference reader on stored real captures: 1 px b/w, a phone at
//! 3072x3072 (test/lab/burst_00_1793117880), truth from the matching dumps (test/dump/burst_00_1793117880).
//! Reference (2026-10-29, read_cells): cell errors 0.457, 0.315, 0.464, 0.521 % for captures 0..3, 2004 ms
//! on one core.
use lensbolt_core::flat::FlatDecoder;
use lensbolt_core::image::RgbImage;
use std::path::PathBuf;

/// The timing assertions need the core to themselves: cargo runs a binary's tests in parallel.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn fast_reader_matches_reference_on_real_1px_captures() {
    let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    // single-core cost: the phone decodes one capture per worker, so per-capture CPU time is what counts
    std::env::set_var("LB_THREADS", "1");
    lensbolt_core::flat::header::ACCEPT_V7_HEADERS.store(true, std::sync::atomic::Ordering::Relaxed);
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let reference = [0.457f64, 0.315, 0.464, 0.521];
    // the fast reader's own errors before its coarse alignment became the robust anchor field (2026-10-29,
    // ~190 ms): the field may not cost more than 0.02 points (nor push a read past 220 ms: the next test)
    let fast_before = [0.522f64, 0.399, 0.519, 0.592];
    for (i, &ref_pct) in reference.iter().enumerate() {
        let cap = root.join(format!("test/lab/burst_00_1793117880/burst_{i:04}_3072x3072.png"));
        let dump = root.join(format!("test/dump/burst_00_1793117880/burst_{i:04}_3072x3072.bin"));
        if !cap.exists() || !dump.exists() { eprintln!("skip: {} not stored", cap.display()); return; }
        let img = RgbImage::load_png(&cap).unwrap();
        let b = std::fs::read(&dump).unwrap();
        let u32_at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let (cols, rows) = (u32_at(0), u32_at(4));
        let k = (cols * rows) as usize;
        let truth = &b[12 + 3 * k..12 + 4 * k];
        let dec = FlatDecoder::new();
        let (chrome, h, headers) = dec.identify(&img).unwrap();
        let hdr = headers.into_iter().flatten().next().unwrap();
        let grid = lensbolt_core::flat::FlatGrid::try_new(chrome.width_px(), chrome.height_px(), hdr.format).unwrap();
        let gray = img.to_gray();
        let r = lensbolt_core::flat::fast::read_cells_luma(&gray, &grid, &h, hdr.frame_seq & 3, None).expect("bw");
        let n = grid.used_data_cells();
        let err = (0..n).filter(|&j| { let (c, rr) = grid.data_position(j as u32); let q = (rr * cols + c) as usize; r.values[q] != truth[q] }).count();
        let pct = err as f64 * 100.0 / n as f64;
        eprintln!("capture {i}: fast {pct:.3}% (reference {ref_pct}%)");
        assert!(pct <= ref_pct + 0.1, "capture {i}: fast reader {pct:.3}% vs reference {ref_pct}%");
        assert!(pct <= fast_before[i] + 0.02, "capture {i}: fast reader {pct:.3}% vs {}% before the anchor field", fast_before[i]);
    }
}

/// Set in the child run that times the reads.
const TIMING_CHILD: &str = "LB_FAST_REF_TIMING_CHILD";
const READ_CAP_MS: f64 = 220.0;

/// The fast reader on the same 4 real captures reads each in at most 220 ms on one core: the fastest of
/// five reads, timed in a child process of this binary that runs only this test (in the full suite this
/// binary's other tests and the machine's state after the library tests pushed single in-process reads to
/// 285-289 ms against 180-195 ms run alone), and the fastest of up to three child runs, the next only
/// while a capture is over the cap.
#[test]
fn fast_reader_reads_real_1px_captures_within_220_ms() {
    const NAME: &str = "fast_reader_reads_real_1px_captures_within_220_ms";
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let cap = |i: usize| root.join(format!("test/lab/burst_00_1793117880/burst_{i:04}_3072x3072.png"));
    if (0..4).any(|i| !cap(i).exists()) { eprintln!("skip: {} not stored", cap(0).display()); return; }
    if std::env::var_os(TIMING_CHILD).is_some() {
        std::env::set_var("LB_THREADS", "1");
        lensbolt_core::flat::header::ACCEPT_V7_HEADERS.store(true, std::sync::atomic::Ordering::Relaxed);
        let dec = FlatDecoder::new();
        for i in 0..4 {
            let img = RgbImage::load_png(&cap(i)).unwrap();
            let (chrome, h, headers) = dec.identify(&img).unwrap();
            let hdr = headers.into_iter().flatten().next().unwrap();
            let grid = lensbolt_core::flat::FlatGrid::try_new(chrome.width_px(), chrome.height_px(), hdr.format).unwrap();
            let gray = img.to_gray();
            let mut ms = f64::MAX;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                assert!(lensbolt_core::flat::fast::read_cells_luma(&gray, &grid, &h, hdr.frame_seq & 3, None).is_some());
                ms = ms.min(t.elapsed().as_secs_f64() * 1000.0);
            }
            eprintln!("@@ read {i} {ms:.3}");
        }
        return;
    }
    let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut best = [f64::MAX; 4];
    for run in 1..=3 {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([NAME, "--exact", "--nocapture", "--test-threads=1"])
            .env(TIMING_CHILD, "1").env("LB_THREADS", "1").env_remove("LB_PROFILE").output().expect("the timing child run");
        let text = String::from_utf8_lossy(&out.stderr).into_owned();
        assert!(out.status.success(), "the timing child run failed:\n{text}");
        let mut seen = 0;
        for l in text.lines().filter_map(|l| l.strip_prefix("@@ read ")) {
            let (i, ms) = l.split_once(' ').unwrap();
            let (i, ms): (usize, f64) = (i.parse().unwrap(), ms.parse().unwrap());
            best[i] = best[i].min(ms);
            seen += 1;
        }
        assert_eq!(seen, 4, "the timing child run reported {seen} of 4 reads:\n{text}");
        eprintln!("run {run}: fastest reads {:.0?} ms", best);
        if best.iter().all(|&ms| ms <= READ_CAP_MS) { break; }
    }
    for (i, ms) in best.iter().enumerate() {
        assert!(*ms <= READ_CAP_MS, "capture {i}: fast reader took {ms:.0} ms on one core (fastest of up to 3 runs; cap {READ_CAP_MS} ms, reference 2004 ms)");
    }
}

/// Grey 4 on the fast path against the reference reader on realistic simulated photos (perspective, lens
/// blur, colour rendering, noise): 3 px at a Full HD camera's ~2.6 camera px per cell, 2 px at a close
/// phone's ~2.6. One core.
#[test]
fn fast_reader_matches_reference_on_grey4() {
    let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    use lensbolt_core::flat::{compose, render, Alphabet, CellFormat, FlatGrid, FlatHeader, ReadParams};
    use lensbolt_core::header::FrameType;
    use std::sync::Arc;
    std::env::set_var("LB_THREADS", "1");
    for (pitch, cam) in [(3u8, 1180u32), (2, 1750)] {
        let fmt = CellFormat { pitch, alphabet: Alphabet::G4 };
        let g = Arc::new(FlatGrid::try_new(1024, 1024, fmt).unwrap());
        let hdr = FlatHeader { format: fmt, cols: g.cols() as u16, rows: g.rows() as u16, frame_type: FrameType::Data, transfer_id: 3, segment: 1, block_id: 7, frame_seq: 0 };
        let payload: Vec<u8> = (0..g.payload_bytes()).map(|i| ((i as u64 * 2654435761) >> 9) as u8).collect();
        let f = compose(g.clone(), hdr, &payload);
        let photo = lensbolt_core::testutil::photograph_realistic(&render(&f), 1.0, cam, cam, 11);
        let dec = FlatDecoder::new();
        let (_, h, _) = dec.identify(&photo).expect("identify");
        let n = g.used_data_cells();
        let wrong = |v: &[u8]| (0..n).filter(|&i| { let (c, r) = g.data_position(i as u32); v[(r * g.cols() + c) as usize] != f.value(c, r) }).count() as f64 * 100.0 / n as f64;
        let t = std::time::Instant::now();
        let reference = lensbolt_core::flat::read_cells(&photo, &g, &h, &ReadParams::default().tuned_for(pitch));
        let ref_ms = t.elapsed().as_secs_f64() * 1000.0;
        let gray = photo.to_gray();
        let t = std::time::Instant::now();
        let fast = lensbolt_core::flat::fast::read_cells_luma(&gray, &g, &h, 0, None).expect("g4 on the fast path");
        let fast_ms = t.elapsed().as_secs_f64() * 1000.0;
        let (re, fe) = (wrong(&reference.values), wrong(&fast.values));
        eprintln!("{pitch}px g4: reference {re:.3}% in {ref_ms:.0} ms, fast {fe:.3}% in {fast_ms:.0} ms (sampling {:.2})", lensbolt_core::flat::sampling(&h, &g));
        assert!(fe <= re + 0.1, "{pitch}px g4: fast {fe:.3}% vs reference {re:.3}%");
        // RS parity 24 per 255 corrects 12 byte errors (4.7%); 2 bits per cell put ~4 cells in a byte, so the
        // cell error rate must stay near 1% for codewords to decode
        let target = if pitch == 2 { 1.5 } else { 0.8 };
        assert!(fe <= target, "{pitch}px g4: fast {fe:.3}% above the {target}% the code needs");
        assert!(fast_ms <= 0.45 * ref_ms, "{pitch}px g4: fast {fast_ms:.0} ms vs reference {ref_ms:.0} ms");
    }
}

/// Tracking across captures: capture i+1 read with capture i's block shifts as the starting field (the live
/// receiver's case: consecutive captures of a hand-held phone) is as accurate as a cold read and cheaper.
#[test]
fn tracked_read_of_the_next_capture_is_as_good_and_faster() {
    let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("LB_THREADS", "1");
    lensbolt_core::flat::header::ACCEPT_V7_HEADERS.store(true, std::sync::atomic::Ordering::Relaxed);
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let load = |i: usize| -> Option<(RgbImage, Vec<u8>)> {
        let cap = root.join(format!("test/lab/burst_00_1793117880/burst_{i:04}_3072x3072.png"));
        let dump = root.join(format!("test/dump/burst_00_1793117880/burst_{i:04}_3072x3072.bin"));
        if !cap.exists() || !dump.exists() { return None; }
        let b = std::fs::read(&dump).unwrap();
        let u32_at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let k = (u32_at(0) * u32_at(4)) as usize;
        Some((RgbImage::load_png(&cap).unwrap(), b[12 + 3 * k..12 + 4 * k].to_vec()))
    };
    let dec = FlatDecoder::new();
    let mut prev: Option<Vec<(f32, f32)>> = None;
    for i in 0..4 {
        let Some((img, truth)) = load(i) else { eprintln!("skip: captures not stored"); return };
        let (chrome, h, headers) = dec.identify(&img).unwrap();
        let hdr = headers.into_iter().flatten().next().unwrap();
        let grid = lensbolt_core::flat::FlatGrid::try_new(chrome.width_px(), chrome.height_px(), hdr.format).unwrap();
        let gray = img.to_gray();
        let n = grid.used_data_cells();
        let err = |v: &[u8]| (0..n).filter(|&j| { let (c, r) = grid.data_position(j as u32); let q = (r * grid.cols() + c) as usize; v[q] != truth[q] }).count() as f64 * 100.0 / n as f64;
        // the fastest of five reads each (the timing gates' rule): one read of ~60-100 ms moves by 10% or more
        // with the machine, and in the full suite the fastest of three tracked reads once came out at 73 ms
        // against 57-62 ms run alone (2026-10-30)
        let fastest = |p: Option<&[(f32, f32)]>| {
            let mut best = (f64::MAX, None);
            for _ in 0..5 {
                let t = std::time::Instant::now();
                let r = lensbolt_core::flat::fast::read_cells_luma(&gray, &grid, &h, hdr.frame_seq & 3, p).unwrap();
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                if ms < best.0 { best = (ms, Some(r)); }
            }
            (best.1.unwrap(), best.0)
        };
        let (cold, cold_ms) = fastest(None);
        if let Some(p) = &prev {
            let (warm, warm_ms) = fastest(Some(p));
            let (ce, we) = (err(&cold.values), err(&warm.values));
            eprintln!("capture {i}: cold {ce:.3}% {cold_ms:.0} ms, tracked {we:.3}% {warm_ms:.0} ms ({:.2}x)", warm_ms / cold_ms);
            assert!(we <= ce + 0.05, "capture {i}: tracked {we:.3}% vs cold {ce:.3}%");
            // tracking skips the anchor field, the smoothing and the first refinement round: 0.70-0.76x of a
            // cold read before the refinement's window and pooling got cheap (2026-10-30), 0.73-0.82x after
            // (it saves ~25 ms of a ~115 ms cold read, against ~65 of ~185 before), 0.67-0.79x after the levels and
            // pattern decision got cheap (~25 ms of ~85)
            assert!(warm_ms <= 0.9 * cold_ms, "capture {i}: tracked {warm_ms:.0} ms vs cold {cold_ms:.0} ms");
        }
        prev = Some(cold.shifts.clone());
    }
}

/// Screen capture (a sender in a VM or remote-desktop window, read by window sharing): the tile's pixels
/// arrive 1:1, offset inside a larger frame. The reader reads them exactly and fast.
#[test]
fn pixel_exact_capture_reads_every_cell_fast() {
    use lensbolt_core::flat::{compose, render, Alphabet, CellFormat, FlatGrid, FlatHeader};
    use lensbolt_core::header::FrameType;
    use std::sync::Arc;
    let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("LB_THREADS", "1");
    for alphabet in [Alphabet::Bw, Alphabet::G4] {
        let fmt = CellFormat { pitch: 1, alphabet };
        let g = Arc::new(FlatGrid::try_new(1024, 1024, fmt).unwrap());
        let hdr = FlatHeader { format: fmt, cols: g.cols() as u16, rows: g.rows() as u16, frame_type: FrameType::Data, transfer_id: 4, segment: 1, block_id: 9, frame_seq: 0 };
        let payload: Vec<u8> = (0..g.payload_bytes()).map(|i| ((i as u64 * 2654435761) >> 9) as u8).collect();
        let f = compose(g.clone(), hdr, &payload);
        let tile = render(&f);
        // the shared window: the tile at (70, 45) inside a 1164 x 1110 frame, and the same window captured at
        // a high-DPI display's 2x
        for k in [1u32, 2] {
        let mut shot = lensbolt_core::image::RgbImage::new(1164 * k, 1110 * k);
        for y in 0..1024 * k { for x in 0..1024 * k { shot.set(x + 70 * k, y + 45 * k, tile.get(x / k, y / k)); } }
        let gray = shot.to_gray();
        let dec = FlatDecoder::new();
        let (_, h, _) = dec.identify(&shot).expect("identify");
        let t = std::time::Instant::now();
        let r = lensbolt_core::flat::fast::read_cells_luma(&gray, &g, &h, 0, None).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let n = g.used_data_cells();
        let bad = (0..n).filter(|&i| { let (c, rr) = g.data_position(i as u32); r.at(c, rr).0 != f.value(c, rr) }).count();
        eprintln!("1px {alphabet:?} pixel-exact at {k}x: {bad} of {n} wrong in {ms:.1} ms");
        assert_eq!(bad, 0, "{alphabet:?} at {k}x: a pixel-exact capture reads every cell");
        assert!(ms <= 25.0, "{alphabet:?} at {k}x: pixel-exact read took {ms:.1} ms on one core");
        }
    }
}

/// A stored real 1 px capture and its truth (the cell values of the dump), None when not stored.
fn real_capture(i: usize) -> Option<(RgbImage, Vec<u8>)> {
    lensbolt_core::flat::header::ACCEPT_V7_HEADERS.store(true, std::sync::atomic::Ordering::Relaxed);
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let cap = root.join(format!("test/lab/burst_00_1793117880/burst_{i:04}_3072x3072.png"));
    let dump = root.join(format!("test/dump/burst_00_1793117880/burst_{i:04}_3072x3072.bin"));
    if !cap.exists() || !dump.exists() { return None; }
    let b = std::fs::read(&dump).unwrap();
    let u32_at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    let k = (u32_at(0) * u32_at(4)) as usize;
    Some((RgbImage::load_png(&cap).unwrap(), b[12 + 3 * k..12 + 4 * k].to_vec()))
}

/// Glare (or a finger) over the tile centre: the cells it covers are lost, the rest of the field is not. A
/// white 480 x 480 px square on real capture 0 where the tile centre lands (about 218 cells across: more than
/// any one search or neighbourhood); cells farther from the centre than the square's half-diagonal + 20 px
/// read within 0.05 points of the clean capture's read of the same cells.
#[test]
fn glare_at_the_tile_centre_stays_local() {
    let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("LB_THREADS", "1");
    let Some((img, truth)) = real_capture(0) else { eprintln!("skip: captures not stored"); return };
    let (chrome, h, headers) = FlatDecoder::new().identify(&img).unwrap();
    let hdr = headers.into_iter().flatten().next().unwrap();
    let grid = lensbolt_core::flat::FlatGrid::try_new(chrome.width_px(), chrome.height_px(), hdr.format).unwrap();
    let clean = img.to_gray();
    let mut gray = clean.clone();
    let (pcx, pcy) = h.apply(grid.width_px() as f32 / 2.0, grid.height_px() as f32 / 2.0);
    let side = 480u32;
    let (x0, y0) = ((pcx - side as f32 / 2.0).round() as u32, (pcy - side as f32 / 2.0).round() as u32);
    for y in y0..y0 + side { for x in x0..x0 + side { gray.data[(y * gray.width + x) as usize] = 255; } }
    let lim = side as f32 / 2.0 * std::f32::consts::SQRT_2 + 20.0;
    // the cells outside: wrong in the glare read, wrong in the clean read, seen
    let (mut seen, mut bad, mut bad_clean) = (0usize, 0usize, 0usize);
    let r = lensbolt_core::flat::fast::read_cells_luma(&gray, &grid, &h, hdr.frame_seq & 3, None).expect("bw");
    let c0 = lensbolt_core::flat::fast::read_cells_luma(&clean, &grid, &h, hdr.frame_seq & 3, None).expect("bw");
    for j in 0..grid.used_data_cells() {
        let (c, rr) = grid.data_position(j as u32);
        let (x, y) = grid.cell_center(c, rr);
        let (ix, iy) = h.apply(x, y);
        if (ix - pcx).hypot(iy - pcy) <= lim { continue; }
        let q = (rr * grid.cols() + c) as usize;
        seen += 1;
        bad += (r.values[q] != truth[q]) as usize;
        bad_clean += (c0.values[q] != truth[q]) as usize;
    }
    let (pct, pct_clean) = (bad as f64 * 100.0 / seen as f64, bad_clean as f64 * 100.0 / seen as f64);
    eprintln!("glare {side} px at the centre: {pct:.3}% wrong outside the patch against {pct_clean:.3}% clean ({bad} and {bad_clean} of {seen})");
    assert!(pct <= pct_clean + 0.05, "glare at the centre spread: {pct:.3}% outside the patch against {pct_clean:.3}% clean");
}

/// Most anchors unreadable (the bottom 60% of the capture a flat grey): the reader still returns reads and
/// does not panic.
#[test]
fn mostly_unreadable_capture_still_returns() {
    let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("LB_THREADS", "1");
    let Some((img, _)) = real_capture(0) else { eprintln!("skip: captures not stored"); return };
    let (chrome, h, headers) = FlatDecoder::new().identify(&img).unwrap();
    let hdr = headers.into_iter().flatten().next().unwrap();
    let grid = lensbolt_core::flat::FlatGrid::try_new(chrome.width_px(), chrome.height_px(), hdr.format).unwrap();
    let mut gray = img.to_gray();
    let from = (gray.height as usize * 4 / 10) * gray.width as usize;
    gray.data[from..].fill(128);
    assert!(lensbolt_core::flat::fast::read_cells_luma(&gray, &grid, &h, hdr.frame_seq & 3, None).is_some());
}

/// A capture too soft to read (a phone sender nearer than the lens focuses): its anchors never agree, and the
/// search-again loop ran all 12 passes, 50-65 ms of a 150-200 ms failed read (a phone reading a phone sender's
/// 2 px tile, 2026-11-04) where a good read takes about 30. It stops once the passes stop converging (every
/// pass changing more than three quarters as many anchors as the one before); the field it keeps is no worse.
#[test]
fn unreadable_capture_stops_searching_early() {
    let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("LB_THREADS", "1");
    let Some((img, _)) = real_capture(0) else { eprintln!("skip: captures not stored"); return };
    let (chrome, h, headers) = FlatDecoder::new().identify(&img).unwrap();
    let hdr = headers.into_iter().flatten().next().unwrap();
    let grid = lensbolt_core::flat::FlatGrid::try_new(chrome.width_px(), chrome.height_px(), hdr.format).unwrap();
    // three box blurs: the cells smear into each other, the border and markers stay
    let mut soft = img.clone();
    for _ in 0..3 { soft = lensbolt_core::testutil::box_blur(&soft); }
    let gray = soft.to_gray();
    let _ = lensbolt_core::flat::fast::read_cells_luma(&gray, &grid, &h, hdr.frame_seq & 3, None);
    let passes = lensbolt_core::flat::fast::last_anchor_passes();
    assert!(passes <= 5, "{passes} search-again passes on an unreadable capture");
}

/// A hand-held capture bends: a smooth displacement of the content peaking at 3 cells at the tile centre
/// and falling to none at the tile's edges and corners (real capture 0 resampled), beyond one anchor search
/// from the tile's shift at the centre. The field follows it: cell errors within 0.05 points of the clean
/// capture.
#[test]
fn regional_offset_beyond_one_search_is_recovered() {
    let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("LB_THREADS", "1");
    let Some((img, truth)) = real_capture(0) else { eprintln!("skip: captures not stored"); return };
    let dec = FlatDecoder::new();
    let err = |img: &RgbImage| -> f64 {
        let (chrome, h, headers) = dec.identify(img).unwrap();
        let hdr = headers.into_iter().flatten().next().unwrap();
        let grid = lensbolt_core::flat::FlatGrid::try_new(chrome.width_px(), chrome.height_px(), hdr.format).unwrap();
        let r = lensbolt_core::flat::fast::read_cells_luma(&img.to_gray(), &grid, &h, hdr.frame_seq & 3, None).expect("bw");
        let n = grid.used_data_cells();
        (0..n).filter(|&j| { let (c, rr) = grid.data_position(j as u32); let q = (rr * grid.cols() + c) as usize; r.values[q] != truth[q] }).count() as f64 * 100.0 / n as f64
    };
    let clean = err(&img);
    // the displacement: peak * (1 - u^2) * (1 - v^2) along one direction, (u, v) = position over the tile's
    // image box from its centre (-1 .. 1); content at p - D(p) moves to p
    let (chrome, h, headers) = dec.identify(&img).unwrap();
    let hdr = headers.into_iter().flatten().next().unwrap();
    let grid = lensbolt_core::flat::FlatGrid::try_new(chrome.width_px(), chrome.height_px(), hdr.format).unwrap();
    let (w, hh) = (grid.width_px() as f32, grid.height_px() as f32);
    let corners = [(0.0, 0.0), (w, 0.0), (0.0, hh), (w, hh)].map(|(x, y)| h.apply(x, y));
    let (x0, x1) = (corners.iter().map(|c| c.0).fold(f32::MAX, f32::min), corners.iter().map(|c| c.0).fold(f32::MIN, f32::max));
    let (y0, y1) = (corners.iter().map(|c| c.1).fold(f32::MAX, f32::min), corners.iter().map(|c| c.1).fold(f32::MIN, f32::max));
    let (cx, cy, rx, ry) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0, (x1 - x0) / 2.0, (y1 - y0) / 2.0);
    let cell = lensbolt_core::flat::sampling(&h, &grid) * hdr.format.pitch as f32;
    // along the tile's rows, where the pilots repeat every 4 cells: at the peak, the repeat of the true shift
    // lies 1 cell the other side of no shift, and the true shift more than one search from the tile's shift
    let (peak, dir) = (3.0 * cell, (1.0f32, 0.0f32));
    let mut warped = img.clone();
    for y in y0.max(0.0) as u32..(y1.ceil() as u32).min(img.height) { for x in x0.max(0.0) as u32..(x1.ceil() as u32).min(img.width) {
        let (u, v) = ((x as f32 - cx) / rx, (y as f32 - cy) / ry);
        let f = (1.0 - u * u).max(0.0) * (1.0 - v * v).max(0.0) * peak;
        warped.set(x, y, img.sample(x as f32 - f * dir.0, y as f32 - f * dir.1));
    } }
    let bent = err(&warped);
    eprintln!("regional offset of 3 cells: {bent:.3}% wrong against {clean:.3}% clean");
    assert!(bent <= clean + 0.05, "regional offset of 3 cells: {bent:.3}% wrong against {clean:.3}% clean");
}

/// A hand-held capture whose marker fit is off by a uniform 2 cells or more (real 1 px captures reach 1.72
/// cells per block): capture 0 read through its identified homography moved by a fixed offset. Along the
/// rows the pilots repeat every 4 cells, so the repeat of the true shift lies nearer to no shift than the
/// shift itself. Every read within 0.05 points of the clean read.
#[test]
fn uniform_offset_of_two_cells_is_recovered() {
    let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("LB_THREADS", "1");
    let Some((img, truth)) = real_capture(0) else { eprintln!("skip: captures not stored"); return };
    let (chrome, h, headers) = FlatDecoder::new().identify(&img).unwrap();
    let hdr = headers.into_iter().flatten().next().unwrap();
    let grid = lensbolt_core::flat::FlatGrid::try_new(chrome.width_px(), chrome.height_px(), hdr.format).unwrap();
    let gray = img.to_gray();
    let cell = lensbolt_core::flat::sampling(&h, &grid) * hdr.format.pitch as f32;
    let n = grid.used_data_cells();
    let err = |hh: &lensbolt_core::detect::Homography| -> f64 {
        let r = lensbolt_core::flat::fast::read_cells_luma(&gray, &grid, hh, hdr.frame_seq & 3, None).expect("bw");
        (0..n).filter(|&j| { let (c, rr) = grid.data_position(j as u32); let q = (rr * grid.cols() + c) as usize; r.values[q] != truth[q] }).count() as f64 * 100.0 / n as f64
    };
    let clean = err(&h);
    // the homography followed by a move of (dx, dy) image px
    let moved = |dx: f32, dy: f32| { let m = h.m; lensbolt_core::detect::Homography { m: [m[0] + dx * m[6], m[1] + dx * m[7], m[2] + dx * m[8], m[3] + dy * m[6], m[4] + dy * m[7], m[5] + dy * m[8], m[6], m[7], m[8]] } };
    let mut worst = Vec::new();
    for (ox, oy) in [(2.0f32, 0.0f32), (2.3, 0.0), (0.0, 2.0), (0.0, 2.3)] {
        let e = err(&moved(ox * cell, oy * cell));
        eprintln!("offset ({ox}, {oy}) cells: {e:.3}% wrong against {clean:.3}% clean");
        if e > clean + 0.05 { worst.push(format!("({ox}, {oy}): {e:.3}%")); }
    }
    assert!(worst.is_empty(), "uniform offsets lost the lock against {clean:.3}% clean: {}", worst.join(", "));
}
