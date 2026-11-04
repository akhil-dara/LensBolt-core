//! The live receiver (what the phone app runs): captures in from the camera thread, never blocking it;
//! a fixed pool decodes, one collector stitches and feeds the receiver; the file comes out SHA-verified.
use lensbolt_core::flat::live::{LiveConfig, LiveReceiver};
use lensbolt_core::flat::{render, FlatSender, FlatSenderConfig};
use lensbolt_core::image::RgbImage;
use lensbolt_core::testutil::{bend, pseudo_random_file};
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join("lensbolt_live"); std::fs::create_dir_all(&d).unwrap(); d.join(name)
}

/// RGB -> planar YUV 4:2:0 the way Android's YUV_420_888 hands it out (BT.601 full range, U/V interleaved
/// with pixel stride 2, rows padded).
fn to_yuv(img: &RgbImage, row_pad: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>, usize, usize) {
    let (w, h) = (img.width as usize, img.height as usize);
    let y_row = w + row_pad;
    let uv_row = w + row_pad; // NV12-style: U and V share rows, pixel stride 2
    let mut y = vec![0u8; y_row * h];
    let mut uv = vec![0u8; uv_row * h.div_ceil(2) + 1];
    for yy in 0..h { for xx in 0..w {
        let [r, g, b] = img.get(xx as u32, yy as u32).map(|v| v as f32);
        y[yy * y_row + xx] = (0.299 * r + 0.587 * g + 0.114 * b).round().clamp(0.0, 255.0) as u8;
        if yy % 2 == 0 && xx % 2 == 0 {
            let u = -0.168_736 * r - 0.331_264 * g + 0.5 * b + 128.0;
            let v = 0.5 * r - 0.418_688 * g - 0.081_312 * b + 128.0;
            uv[(yy / 2) * uv_row + xx] = u.round().clamp(0.0, 255.0) as u8;
            uv[(yy / 2) * uv_row + xx + 1] = v.round().clamp(0.0, 255.0) as u8;
        }
    } }
    let v = uv[1..].to_vec();
    (y, uv, v, y_row, uv_row)
}

#[test]
fn yuv420_converts_back_to_rgb() {
    let mut img = RgbImage::new(6, 4);
    img.fill_rect(0, 0, 2, 2, [255, 255, 255]);
    img.fill_rect(2, 0, 2, 2, [255, 0, 0]);
    img.fill_rect(4, 0, 2, 2, [0, 0, 255]);
    img.fill_rect(0, 2, 6, 2, [0, 200, 0]);
    let (y, u, v, y_row, uv_row) = to_yuv(&img, 3);
    let back = RgbImage::from_yuv420(6, 4, &y, y_row, &u, &v, uv_row, 2);
    for yy in 0..4 { for xx in 0..6 {
        let (a, b) = (img.get(xx, yy), back.get(xx, yy));
        for k in 0..3 { assert!((a[k] as i32 - b[k] as i32).abs() <= 3, "({xx},{yy}): {a:?} vs {b:?}"); }
    } }
}

#[test]
fn live_receiver_completes_a_transfer_from_yuv_captures() {
    let src = tmp("live.bin"); pseudo_random_file(&src, 150_000, 21).unwrap();
    let out = tmp("live_out"); let _ = std::fs::remove_dir_all(&out); std::fs::create_dir_all(&out).unwrap();
    let mut s = FlatSender::new(&src, FlatSenderConfig { manifest_every_frames: 10, transfer_id: 91, ..FlatSenderConfig::default() }, Some(false)).unwrap();
    let live = LiveReceiver::new(&out, LiveConfig { workers: 2 }).unwrap();
    let t0 = Instant::now();
    let mut n = 0u32;
    let mut prev: Option<RgbImage> = None;
    while live.status().complete.is_none() {
        assert!(t0.elapsed() < Duration::from_secs(240), "no completion: {:?}", live.status());
        let cur = render(&s.next_frame().unwrap());
        // every third capture straddles a refresh (top of the previous frame, bottom of this one)
        let mut cam = cur.clone();
        if let (Some(p), true) = (&prev, n % 3 == 1) { for y in 0..cam.height / 2 { for x in 0..cam.width { cam.set(x, y, p.get(x, y)); } } }
        let cam = bend(&cam, 2.0, n as u64);
        let (y, u, v, y_row, uv_row) = to_yuv(&cam, 16);
        let (w, h) = (cam.width, cam.height);
        // a camera does not wait: offer each capture once, and keep showing frames while the pool is busy
        let _ = live.submit(move || RgbImage::from_yuv420(w, h, &y, y_row, &u, &v, uv_row, 2));
        prev = Some(cur);
        n += 1;
        std::thread::sleep(Duration::from_millis(5));
    }
    let st = live.status();
    let path = st.complete.clone().unwrap();
    assert!(std::fs::read(&path).unwrap() == std::fs::read(&src).unwrap(), "output differs");
    assert!(st.decoded > 0 && st.received >= st.decoded, "{st:?}");
    assert_eq!(st.filename.as_deref(), Some("live.bin"));
    assert!(st.to_json().contains("\"complete\":"), "{}", st.to_json());
    // the last read's geometry for the app's guidance: every edge seen, cells some camera px across
    assert!(st.edges.iter().all(|&e| e > 0.2) && st.cell_px > 0.5 && st.cell_px < 40.0, "{st:?}");
    assert!(st.to_json().contains("\"cell_px\":") && st.to_json().contains("\"edges\":["), "{}", st.to_json());
    // how the data reads: the share of the tile two sender frames blend in (none here), and the border's edge width
    assert!(st.to_json().contains("\"mix\":0.00") && st.edge_px > 0.3 && st.edge_px < 6.0, "{}", st.to_json());
    // nothing is short once the file is in: no words to offer
    assert!(st.finish_words.is_none() && st.to_json().contains("\"finish_words\":null"), "{}", st.to_json());
    assert!(st.finish_parts.is_none() && st.to_json().contains("\"finish_parts\":null"), "{}", st.to_json());
    // a single file: no batch, an empty page
    assert!(st.batch.is_none() && st.to_json().contains("\"batch\":null"), "{}", st.to_json());
    assert_eq!(live.batch_page(0, 10), "{\"offset\":0,\"total\":0,\"files\":[]}");
    // the verified file's SHA-256, as the browser's status gives it (the app shows it without reading the file
    // again), and the data reads counted (the app's "locked on" counts reads, not its polls)
    let want = { use sha2::Digest; lensbolt_core::flat::live::hex(&sha2::Sha256::digest(std::fs::read(&src).unwrap())) };
    assert_eq!(st.file_sha256.map(|h| lensbolt_core::flat::live::hex(&h)), Some(want.clone()), "{st:?}");
    assert!(st.to_json().contains(&format!("\"sha256\":\"{want}\"")), "{}", st.to_json());
    assert!(st.data_reads > 0 && st.data_reads < st.decoded && st.to_json().contains(&format!("\"data_reads\":{}", st.data_reads)), "{}", st.to_json());
}

/// A batch through the phone's receiver: the status carries its counts, `batch_page` its files (read while the
/// collector keeps feeding), each saved into the batch's folder.
#[test]
fn live_receiver_pages_a_batch() {
    use lensbolt_core::file::BatchSource;
    let out = tmp("live_batch_out"); let _ = std::fs::remove_dir_all(&out); std::fs::create_dir_all(&out).unwrap();
    let files: Vec<(String, Vec<u8>)> = (0..4u64).map(|i| (format!("scan-{i}.pdf"), { let mut s = 0x99 + i; (0..15_000 + 9_000 * i as usize).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s as u8 }).collect() })).collect();
    let src = files.iter().map(|(n, d)| (n.clone(), BatchSource::Bytes(d.clone()))).collect();
    let mut s = FlatSender::from_batch("scans", src, FlatSenderConfig { manifest_every_frames: 10, ..FlatSenderConfig::default() }).unwrap();
    let live = LiveReceiver::new(&out, LiveConfig { workers: 2 }).unwrap();
    let t0 = Instant::now();
    let mut pages = 0;
    while live.status().complete.is_none() {
        assert!(t0.elapsed() < Duration::from_secs(240), "no completion: {:?}", live.status());
        let cam = render(&s.next_frame().unwrap());
        let _ = live.submit(move || cam);
        // the page is read from another thread than the collector's, while it works
        if live.batch_page(0, 2).contains("\"total\":4") { pages += 1; }
        std::thread::sleep(Duration::from_millis(5));
    }
    std::thread::sleep(Duration::from_millis(50));
    let st = live.status();
    let b = st.batch.clone().expect("a batch");
    assert_eq!((b.name.as_str(), b.total, b.received, b.failed, b.missing_files, b.error.as_deref()), ("scans", 4, 4, 0, None, None));
    assert!(st.to_json().contains("\"batch\":{\"name\":\"scans\",\"total\":4,\"received\":4,\"failed\":0,\"missing_files\":null,\"error\":null}"), "{}", st.to_json());
    let page = live.batch_page(1, 10);
    // a received file carries the SHA-256 it was verified against: the file's own
    let sha1 = { use sha2::Digest; lensbolt_core::flat::live::hex(&sha2::Sha256::digest(&files[1].1)) };
    assert!(page.starts_with(&format!("{{\"offset\":1,\"total\":4,\"files\":[{{\"name\":\"scan-1.pdf\",\"size\":24000,\"state\":\"received\",\"sha256\":\"{sha1}\"}}")), "{page}");
    assert_eq!(page.matches("\"state\":\"received\"").count(), 3, "{page}");
    // a batch has no single file's hash in the status: each file's is on its page
    assert!(st.file_sha256.is_none() && st.to_json().contains("\"sha256\":null"), "{}", st.to_json());
    assert!(pages > 0, "the page was never read while the list was known");
    let dir = st.complete.unwrap();
    for (name, data) in &files { assert!(std::fs::read(dir.join(name)).unwrap() == *data, "{name}"); }
}

#[test]
fn submit_never_blocks_when_the_pool_is_busy() {
    let out = tmp("busy_out"); let _ = std::fs::remove_dir_all(&out); std::fs::create_dir_all(&out).unwrap();
    let live = LiveReceiver::new(&out, LiveConfig { workers: 1 }).unwrap();
    let t0 = Instant::now();
    let mut accepted = 0;
    for _ in 0..20 {
        if live.submit(|| { std::thread::sleep(Duration::from_millis(200)); RgbImage::new(64, 64) }) { accepted += 1; }
    }
    assert!(t0.elapsed() < Duration::from_millis(150), "submit blocked for {:?}", t0.elapsed());
    assert!(accepted < 20, "every capture was queued: the camera would fall behind");
    assert!(live.status().dropped > 0);
    // the app skips copying the planes when no worker is free, and still counts the capture
    let before = live.status();
    live.skip();
    let after = live.status();
    assert_eq!((after.received, after.dropped), (before.received + 1, before.dropped + 1));
}

#[test]
fn y_plane_with_stride_is_copied_row_by_row() {
    use lensbolt_core::image::GrayImage;
    let (w, h, stride) = (10usize, 4usize, 16usize);
    let data: Vec<u8> = (0..stride * h).map(|i| (i % 251) as u8).collect();
    let g = GrayImage::from_plane(w as u32, h as u32, &data, stride);
    assert_eq!((g.width, g.height), (10, 4));
    for y in 0..h { for x in 0..w { assert_eq!(g.get(x as u32, y as u32), data[y * stride + x], "({x},{y})"); } }
}

#[test]
fn integral_box_mean_matches_direct_mean() {
    use lensbolt_core::image::GrayImage;
    let (w, h) = (37u32, 23u32);
    let mut seed = 5u64;
    let mut rnd = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) as u32 };
    let data: Vec<u8> = (0..w * h).map(|_| (rnd() % 256) as u8).collect();
    let g = GrayImage::from_plane(w, h, &data, w as usize);
    let ii = g.integral();
    for _ in 0..50 {
        let (x0, y0) = (rnd() as i32 % 45 - 4, rnd() as i32 % 30 - 4);
        let (x1, y1) = (x0 + 1 + rnd() as i32 % 9, y0 + 1 + rnd() as i32 % 9);
        // direct mean over the box clamped to the image
        let (cx0, cy0, cx1, cy1) = (x0.clamp(0, w as i32), y0.clamp(0, h as i32), x1.clamp(0, w as i32), y1.clamp(0, h as i32));
        if cx1 <= cx0 || cy1 <= cy0 { continue; }
        let mut s = 0u64; for y in cy0..cy1 { for x in cx0..cx1 { s += g.get(x as u32, y as u32) as u64; } }
        let want = s as f32 / ((cx1 - cx0) * (cy1 - cy0)) as f32;
        let got = ii.box_mean(x0, y0, x1, y1);
        assert!((got - want).abs() < 1e-3, "box ({x0},{y0})-({x1},{y1}): {got} vs {want}");
    }
}

#[test]
fn luma_header_matches_rgb_header() {
    use lensbolt_core::flat::read::read_header_luma;
    use lensbolt_core::flat::{read_header, FlatDecoder};
    let src = tmp("hdr.bin"); pseudo_random_file(&src, 30_000, 4).unwrap();
    let mut s = FlatSender::new(&src, FlatSenderConfig { manifest_every_frames: 10, transfer_id: 5, ..FlatSenderConfig::default() }, Some(false)).unwrap();
    let _ = s.next_frame().unwrap();
    let cam = bend(&render(&s.next_frame().unwrap()), 2.0, 3);
    let fr = FlatDecoder::new().decode(&cam).unwrap();
    let rgb = read_header(&cam, fr.grid.chrome(), &fr.homography);
    let luma = read_header_luma(&cam.to_gray(), fr.grid.chrome(), &fr.homography);
    assert_eq!(rgb, luma);
    assert!(luma[0].is_some());
}

#[test]
fn live_receiver_reads_manifest_and_bw_data() {
    use lensbolt_core::flat::live::Capture;
    use lensbolt_core::flat::{Alphabet, CellFormat};
    let src = tmp("live_bw.bin"); pseudo_random_file(&src, 120_000, 22).unwrap();
    let out = tmp("live_bw_out"); let _ = std::fs::remove_dir_all(&out); std::fs::create_dir_all(&out).unwrap();
    let cfg = FlatSenderConfig { manifest_every_frames: 10, transfer_id: 92, format: CellFormat { pitch: 3, alphabet: Alphabet::Bw }, ..FlatSenderConfig::default() };
    let mut s = FlatSender::new(&src, cfg, Some(false)).unwrap();
    let live = LiveReceiver::new(&out, LiveConfig { workers: 2 }).unwrap();
    let t0 = Instant::now();
    let mut n = 0u64;
    let mut progress_before_done = 0u64;
    while live.status().complete.is_none() {
        assert!(t0.elapsed() < Duration::from_secs(240), "no completion: {:?}", live.status());
        let st = live.status();
        if st.complete.is_none() { progress_before_done = progress_before_done.max(st.received_bytes); }
        let cam = bend(&render(&s.next_frame().unwrap()), 1.5, n);
        let y = cam.to_gray();
        let _ = live.submit_capture(move || Capture { y, rgb: Some(Box::new(move || cam)), origin: (0, 0) });
        n += 1;
        std::thread::sleep(Duration::from_millis(5));
    }
    let st = live.status();
    assert!(std::fs::read(st.complete.as_ref().unwrap()).unwrap() == std::fs::read(&src).unwrap(), "output differs");
    assert!(st.luma_ms > 0.0, "b/w data must take the luma path: {st:?}");
    assert!(st.colour_ms > 0.0, "the c8 manifest must take the colour path: {st:?}");
    // a file smaller than one 16 MB segment must still show progress while it arrives
    assert!(progress_before_done > 0, "no progress reported before completion: {st:?}");
    assert!(st.received_bytes <= st.stream_size, "{st:?}");
}

/// The codewords in the status are those of the latest decoded capture's frame, also when none of its packets
/// were new (the app's guidance never says "Move closer" while the codewords read: a stale count from the
/// manifest, 9 of 9, read as a tile reading well while the data frames read almost nothing).
#[test]
fn codewords_follow_the_latest_capture_even_when_nothing_new_decodes() {
    use lensbolt_core::flat::live::Capture;
    use lensbolt_core::flat::{Alphabet, CellFormat};
    use lensbolt_core::testutil::photograph;
    let src = tmp("live_cw.bin"); pseudo_random_file(&src, 40_000, 23).unwrap();
    let out = tmp("live_cw_out"); let _ = std::fs::remove_dir_all(&out); std::fs::create_dir_all(&out).unwrap();
    let live = LiveReceiver::new(&out, LiveConfig { workers: 2 }).unwrap();
    let settle = |n: u64| {
        let t0 = Instant::now();
        loop {
            let st = live.status();
            if st.decoded + st.failed + st.colour_skipped >= n { return st; }
            assert!(t0.elapsed() < Duration::from_secs(60), "captures never came back: {st:?}");
            std::thread::sleep(Duration::from_millis(1));
        }
    };
    let offer = |cam: RgbImage| { let y = cam.to_gray(); assert!(live.submit_capture(move || Capture { y, rgb: Some(Box::new(move || cam)), origin: (0, 0) })); };
    // the colour manifest, read in full
    let cfg = |pitch| FlatSenderConfig { manifest_every_frames: 10, transfer_id: 93, format: CellFormat { pitch, alphabet: Alphabet::Bw }, ..FlatSenderConfig::default() };
    let mut s = FlatSender::new(&src, cfg(1), Some(false)).unwrap();
    let manifest_img = render(&s.next_frame().unwrap());
    offer(manifest_img.clone());
    let st = settle(1);
    // the manifest (12 px colour) always reads whole: its codewords say nothing about how the data reads
    assert!(st.have_manifest && st.codewords == 0 && st.codewords_ok == 0, "a manifest read leaves no codewords: {st:?}");
    assert!(st.data_reads == 0 && st.to_json().contains("\"data_reads\":0") && st.to_json().contains("\"sha256\":null"), "a manifest read is not a data read: {}", st.to_json());
    let manifest = (st.codewords_ok, st.codewords);
    // 1 px b/w data photographed far too small: the header reads, the cells do not
    for k in 0..3u64 {
        let cam = photograph(&render(&s.next_frame().unwrap()), 1100, 1000, 3.0, 6, 5 + k);
        offer(cam);
        let st = settle(2 + k);
        assert_eq!(st.decoded, 2 + k, "the 1 px frame's header must read for this test: {st:?}");
        assert!(st.format.as_deref().is_some_and(|f| f.starts_with("1px bw")), "{st:?}");
        assert!(st.codewords > 100 && st.codewords_ok * 10 < st.codewords,
            "capture {k}: the 1 px frame's own codewords, nearly none read: {} of {} (the manifest read {manifest:?}): {st:?}", st.codewords_ok, st.codewords);
    }
    // the manifest again: the data read's codewords stay (guidance never sees 9 of 9 between 3 of 111)
    let before = live.status();
    offer(manifest_img);
    let st = settle(5);
    assert_eq!((st.codewords_ok, st.codewords), (before.codewords_ok, before.codewords), "a manifest read changed the codewords: {before:?} -> {st:?}");
    // the data reads: the three 1 px captures, not the two manifest reads
    assert_eq!((before.data_reads, st.data_reads), (3, 3), "{st:?}");
}

/// The app's status is JSON whatever the numbers: non-finite ones go out as 0.
#[test]
fn live_status_numbers_are_always_finite() {
    use lensbolt_core::flat::live::LiveStatus;
    let s = LiveStatus { decode_ms: f64::NAN, cell_px: f64::INFINITY, edges: [f64::NAN, 1.0, f64::NEG_INFINITY, 1.0], seconds: f64::INFINITY, mbit_per_s: f64::NAN, luma_ms: f64::INFINITY, colour_ms: f64::NAN, sender_fps: f64::INFINITY, ..LiveStatus::default() };
    let j = s.to_json();
    for bad in ["inf", "NaN"] { assert!(!j.contains(bad), "{bad} in the status: {j}"); }
    assert!(j.contains("\"decode_ms\":0.0") && j.contains("\"edges\":[0.00,1.00,0.00,1.00]") && j.contains("\"seconds\":0.000") && j.contains("\"sender_fps\":0.0"), "{j}");
}

/// The phone's list reads never wait for the receiver: while the collector is inside a feed (here a store write
/// held shut, as a 500 MB file being written would hold it), `try_batch_page` says busy at once (None) instead of
/// waiting, so the app's lock around the handle is never held across a feed; once the feed ends it reads the page.
#[test]
fn try_batch_page_is_busy_while_a_feed_holds_the_receiver() {
    use lensbolt_core::file::{BatchSource, PartStore, StoreFactory};
    use std::io;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};
    use std::sync::Arc;
    // a store whose writes wait while `shut` is set (at most 20 s: a failed test still ends); `inside` counts them
    #[derive(Clone, Default)]
    struct Gate { shut: Arc<AtomicBool>, inside: Arc<AtomicUsize> }
    struct Stores(Gate);
    struct Store { data: Vec<u8>, gate: Gate }
    impl PartStore for Store {
        fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
            let o = offset as usize;
            buf.copy_from_slice(self.data.get(o..o + buf.len()).ok_or_else(|| io::Error::other("read past the end"))?);
            Ok(())
        }
        fn write_at(&mut self, offset: u64, d: &[u8]) -> io::Result<()> {
            self.gate.inside.fetch_add(1, SeqCst);
            let t0 = Instant::now();
            while self.gate.shut.load(SeqCst) && t0.elapsed() < Duration::from_secs(20) { std::thread::sleep(Duration::from_millis(1)); }
            let o = offset as usize;
            if self.data.len() < o + d.len() { self.data.resize(o + d.len(), 0); }
            self.data[o..o + d.len()].copy_from_slice(d);
            self.gate.inside.fetch_sub(1, SeqCst);
            Ok(())
        }
        fn set_len(&mut self, len: u64) -> io::Result<()> { self.data.resize(len as usize, 0); Ok(()) }
    }
    impl StoreFactory for Stores {
        fn create(&mut self, _name: &str) -> io::Result<Box<dyn PartStore>> { Ok(Box::new(Store { data: Vec::new(), gate: self.0.clone() })) }
        fn remove(&mut self, _name: &str) {}
    }
    let files: Vec<(String, Vec<u8>)> = (0..3u64).map(|i| (format!("doc-{i}.bin"), { let mut s = 0x51 + i; (0..20_000 + 7_000 * i as usize).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s as u8 }).collect() })).collect();
    let src = files.iter().map(|(n, d)| (n.clone(), BatchSource::Bytes(d.clone()))).collect();
    let mut s = FlatSender::from_batch("docs", src, FlatSenderConfig { manifest_every_frames: 10, ..FlatSenderConfig::default() }).unwrap();
    let gate = Gate::default();
    gate.shut.store(true, SeqCst);
    let live = LiveReceiver::with_stores(Box::new(Stores(gate.clone())), LiveConfig { workers: 2 }).unwrap();
    // before any feed: a page at once (an empty list)
    assert_eq!(live.try_batch_page(0, 10).as_deref(), Some("{\"offset\":0,\"total\":0,\"files\":[]}"));
    let deadline = Instant::now() + Duration::from_secs(120);
    while gate.inside.load(SeqCst) == 0 {
        assert!(Instant::now() < deadline, "no feed ever reached the store: {:?}", live.status());
        let cam = render(&s.next_frame().unwrap());
        let _ = live.submit(move || cam);
        std::thread::sleep(Duration::from_millis(5));
    }
    // the collector is inside a feed, holding the receiver: every try answers busy, each at once
    let mut worst = Duration::ZERO;
    for _ in 0..20 {
        let t = Instant::now();
        let page = live.try_batch_page(0, 10);
        worst = worst.max(t.elapsed());
        assert!(page.is_none(), "a page while a feed holds the receiver: {page:?}");
    }
    assert!(gate.inside.load(SeqCst) > 0, "the feed ended during the probe: the test proved nothing");
    assert!(worst <= Duration::from_millis(50), "try_batch_page waited {worst:?}");
    // the feed goes on: the transfer completes and the page reads again
    gate.shut.store(false, SeqCst);
    while live.status().complete.is_none() {
        assert!(Instant::now() < deadline, "no completion: {:?}", live.status());
        let cam = render(&s.next_frame().unwrap());
        let _ = live.submit(move || cam);
        std::thread::sleep(Duration::from_millis(5));
    }
    let mut page = None;
    while page.is_none() { assert!(Instant::now() < deadline, "the page never read again"); page = live.try_batch_page(0, 10); std::thread::sleep(Duration::from_millis(5)); }
    let page = page.unwrap();
    assert!(page.starts_with("{\"offset\":0,\"total\":3,\"files\":[{\"name\":\"doc-0.bin\""), "{page}");
    assert_eq!(page.matches("\"state\":\"received\"").count(), 3, "{page}");
}

/// A full-sensor stream (a phone sender's 2 px cells read only at the sensor's resolution): while the tile is
/// tracked, only its region is copied (the receiver's `crop_rect`; a whole 12 MP plane costs 6 ms a capture on
/// the camera's thread, the region under 1 ms), and the worker reads it with the hint moved into the crop. The
/// tile jumps once mid-transfer, out of the crop: that capture fails, the next is copied whole and found again.
#[test]
fn live_receiver_reads_tracked_crops_of_a_large_frame() {
    use lensbolt_core::detect::Homography;
    use lensbolt_core::flat::live::Capture;
    use lensbolt_core::flat::{Alphabet, CellFormat};
    use lensbolt_core::image::GrayImage;
    let src = tmp("live_crop.bin"); pseudo_random_file(&src, 600_000, 24).unwrap();
    let out = tmp("live_crop_out"); let _ = std::fs::remove_dir_all(&out); std::fs::create_dir_all(&out).unwrap();
    let cfg = FlatSenderConfig { manifest_every_frames: 10, transfer_id: 94, format: CellFormat { pitch: 3, alphabet: Alphabet::Bw }, ..FlatSenderConfig::default() };
    let mut s = FlatSender::new(&src, cfg, Some(false)).unwrap();
    let live = LiveReceiver::new(&out, LiveConfig { workers: 2 }).unwrap();
    let (w, h) = (3000u32, 2250u32);
    let place = |im: &RgbImage, cx: f32, cy: f32| -> RgbImage {
        let s = 1200.0 / im.width as f32;
        let src = [(0.0, 0.0), (im.width as f32, 0.0), (im.width as f32, im.height as f32), (0.0, im.height as f32)];
        let dst = src.map(|p| (cx + s * (p.0 - im.width as f32 / 2.0), cy + s * (p.1 - im.height as f32 / 2.0)));
        im.warp(&Homography::from_points(src, dst).unwrap().inverse().unwrap(), w, h)
    };
    let t0 = Instant::now();
    let (mut n, mut cropped, mut whole) = (0u64, 0u64, 0u64);
    while live.status().complete.is_none() {
        assert!(t0.elapsed() < Duration::from_secs(240), "no completion: {:?}", live.status());
        // the tile jumps 700 px right after 30 captures
        let cx = if n < 30 { 1100.0 } else { 1800.0 };
        let cam = place(&render(&s.next_frame().unwrap()), cx, 1125.0);
        let y = cam.to_gray();
        let crop = if live.want_chroma() { None } else { live.crop_rect(w, h) };
        let ok = match crop {
            Some([x0, y0, x1, y1]) => {
                let mut data = Vec::with_capacity(((x1 - x0) * (y1 - y0)) as usize);
                for yy in y0..y1 { data.extend_from_slice(&y.data[(yy * w + x0) as usize..(yy * w + x1) as usize]); }
                let c = GrayImage { width: x1 - x0, height: y1 - y0, data };
                live.submit_capture(move || Capture { y: c, rgb: None, origin: (x0, y0) })
            }
            None => live.submit_capture(move || Capture { y, rgb: Some(Box::new(move || cam)), origin: (0, 0) }),
        };
        if ok { if crop.is_some() { cropped += 1 } else { whole += 1 } }
        n += 1;
        std::thread::sleep(Duration::from_millis(15));
    }
    let st = live.status();
    assert!(std::fs::read(st.complete.as_ref().unwrap()).unwrap() == std::fs::read(&src).unwrap(), "output differs");
    // whole: the start (no geometry yet), every 4th capture (its colour copy), the manifest's colour reads and the
    // capture after the jump; the rest cropped (43 of 66 when written)
    assert!(cropped > whole, "most captures cropped: {cropped} cropped, {whole} whole");
}
