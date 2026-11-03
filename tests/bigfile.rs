//! 1 GiB streamed round trip at the packet layer (no images). Run explicitly:
//!   cargo test --release --test bigfile -- --ignored --nocapture
//! Both sides run in this one process; watch peak memory in Task Manager / `/usr/bin/time -v`.
use lensbolt_core::file::{FeedOutcome, Receiver};
use lensbolt_core::flat::{FlatSender, FlatSenderConfig, MANIFEST_EVERY_FRAMES};
use lensbolt_core::testutil::pseudo_random_file;
use std::time::Instant;

#[test]
#[ignore]
fn one_gib_round_trip_streams_with_bounded_memory() {
    let base = std::env::temp_dir().join("lensbolt_bigfile");
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(base.join("out")).unwrap();
    let input = base.join("big.bin");
    let t0 = Instant::now();
    pseudo_random_file(&input, 1u64 << 30, 99).unwrap();
    eprintln!("generated 1 GiB in {:?}", t0.elapsed());
    let t1 = Instant::now();
    let mut s = FlatSender::new(&input, FlatSenderConfig { manifest_every_frames: 120, transfer_id: 5, ..FlatSenderConfig::default() }, None).unwrap();
    eprintln!("prepared (hash+sniff) in {:?}; segments={} frames/pass={}", t1.elapsed(), s.manifest().segment_count(), s.frames_per_pass());
    assert!(!s.manifest().compressed);
    assert_eq!(s.manifest().segment_count(), 64);
    let mut r = Receiver::new(&base.join("out")).unwrap();
    let t2 = Instant::now();
    let mut max_live = 0usize; let mut max_mem = 0usize; let mut done = None; let mut frames = 0u64;
    let mut seed = 11u64;
    while done.is_none() {
        let (h, ps) = s.next_packets().unwrap(); frames += 1;
        // 2% packet loss so the repair path is exercised at scale
        let kept: Vec<_> = ps.into_iter().filter(|_| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed % 50 != 0 }).collect();
        if let FeedOutcome::Complete(p) = r.feed_packets(&h, &kept).unwrap() { done = Some(p); }
        max_live = max_live.max(r.live_decoders()); max_mem = max_mem.max(r.live_memory_bytes());
        if frames % 5000 == 0 { let p = r.progress(); eprintln!("frames {frames}: segments {}/{} live decoders {}", p.segments_done, p.segments_total, r.live_decoders()); }
        assert!(frames < s.frames_per_pass() * 3, "did not complete within 3 passes");
    }
    eprintln!("transfer: {} frames in {:?} (one pass = {} frames); max open chunk decoders {max_live}, max decoder memory {:.1} MiB", frames, t2.elapsed(), s.frames_per_pass(), max_mem as f64 / 1048576.0);
    assert!(max_mem <= 100 << 20, "decoder memory bound violated: {} bytes", max_mem);
    let t3 = Instant::now();
    let out = done.unwrap();
    assert_eq!(std::fs::metadata(&out).unwrap().len(), 1u64 << 30);
    use std::io::Read;
    let (mut a, mut b) = (std::fs::File::open(&input).unwrap(), std::fs::File::open(&out).unwrap());
    let (mut ba, mut bb) = (vec![0u8; 8 << 20], vec![0u8; 8 << 20]);
    loop { let n = a.read(&mut ba).unwrap(); let m = b.read(&mut bb).unwrap(); assert_eq!(n, m); if n == 0 { break; } assert_eq!(ba[..n], bb[..n]); }
    eprintln!("verified byte-identical in {:?}", t3.elapsed());
    let _ = std::fs::remove_dir_all(&base);
}

/// Peak resident memory of this process (VmHWM, Linux), in bytes; 0 where /proc is unavailable.
fn peak_rss() -> u64 {
    std::fs::read_to_string("/proc/self/status").ok().and_then(|s| s.lines().find(|l| l.starts_with("VmHWM:"))
        .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())).map(|kb| kb * 1024).unwrap_or(0)
}

/// 1 GB through the frame format (frame-wide codewords, packets back to back) at the 20x configuration (1 px b/w, 1024 tile): every frame is composed into
/// cells, 0.3% of its data cells are flipped (the real channel reads 0.3-0.85%), 3% of frames are lost
/// whole, and the cells go through the real extractor, packet unpacking and the receiver. Rendering and
/// camera reading are skipped (covered by the frame tests); this checks that the format, the fountain and
/// the receiver scale to 1 GB with bounded memory, and SHA-256 at the end.
///   cargo test --release --test bigfile one_gb_through_frames -- --ignored --nocapture
#[test]
#[ignore]
fn one_gb_through_frames() {
    use lensbolt_core::flat::{extract_cells, Alphabet, CellFormat};
    use lensbolt_core::packet::unpack_frame;
    use sha2::{Digest, Sha256};
    let base = std::env::temp_dir().join("lensbolt_bigfile_frames");
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(base.join("out")).unwrap();
    let input = base.join("big.bin");
    let size = 1_000_000_000u64;
    pseudo_random_file(&input, size, 2026).unwrap();
    let rss0 = peak_rss();
    let t1 = Instant::now();
    let cfg = FlatSenderConfig { format: CellFormat { pitch: 1, alphabet: Alphabet::Bw }, manifest_every_frames: MANIFEST_EVERY_FRAMES, transfer_id: 77, ..FlatSenderConfig::default() };
    let mut s = FlatSender::new(&input, cfg, Some(false)).unwrap();
    eprintln!("prepared in {:?}; segments={} frames/pass={} packets/frame={}", t1.elapsed(), s.manifest().segment_count(), s.frames_per_pass(), s.grid().packets());
    let mut r = Receiver::new(&base.join("out")).unwrap();
    let t2 = Instant::now();
    let (mut seed, mut frames, mut lost, mut cw_failed) = (12345u64, 0u64, 0u64, 0usize);
    let mut rnd = move || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
    let mut done = None;
    let mut max_mem = 0usize;
    while done.is_none() {
        let f = s.next_frame().unwrap(); frames += 1;
        if rnd() % 100 < 3 { lost += 1; continue; }
        let g = f.grid.clone();
        let mut values = f.values;
        let n = g.used_data_cells() as u64;
        for _ in 0..n * 3 / 1000 { let (c, rr) = g.data_position((rnd() % n) as u32); values[(rr * g.cols() + c) as usize] ^= 1; }
        let margins = vec![u16::MAX; values.len()];
        let fp = extract_cells(&g, &values, &margins, None, None);
        cw_failed += fp.codewords_failed;
        if let FeedOutcome::Complete(p) = r.feed_packets(&f.header, &unpack_frame(&fp)).unwrap() { done = Some(p); }
        max_mem = max_mem.max(r.live_memory_bytes());
        if frames % 2000 == 0 { let p = r.progress(); eprintln!("frames {frames}: segments {}/{} peak RSS {:.0} MiB ({:?})", p.segments_done, p.segments_total, peak_rss() as f64 / 1048576.0, t2.elapsed()); }
        assert!(frames < s.frames_per_pass() * 3, "did not complete within 3 passes");
    }
    let secs_at_30 = frames as f64 / 30.0;
    eprintln!("complete: {frames} frames ({lost} lost, {cw_failed} codewords failed) in {:?}; one pass = {} frames; at 30 fps that is {:.0} s = {:.2} Mbit/s; decoder memory {:.1} MiB; peak RSS {:.0} MiB (before sender {:.0} MiB)",
        t2.elapsed(), s.frames_per_pass(), secs_at_30, size as f64 * 8.0 / secs_at_30 / 1e6, max_mem as f64 / 1048576.0, peak_rss() as f64 / 1048576.0, rss0 as f64 / 1048576.0);
    let out = done.unwrap();
    let hash = |p: &std::path::Path| { use std::io::Read; let mut h = Sha256::new(); let mut f = std::fs::File::open(p).unwrap(); let mut b = vec![0u8; 8 << 20]; loop { let n = f.read(&mut b).unwrap(); if n == 0 { break; } h.update(&b[..n]); } h.finalize() };
    assert_eq!(std::fs::metadata(&out).unwrap().len(), size);
    assert_eq!(hash(&input), hash(&out), "SHA-256 differs");
    eprintln!("SHA-256 identical");
    assert!(peak_rss() < 512 << 20, "peak RSS {} bytes", peak_rss());
    let _ = std::fs::remove_dir_all(&base);
}
