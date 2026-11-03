//! Whole transfers through the flat format: sender -> render -> decoder -> receiver.
use lensbolt_core::file::{FeedOutcome, Receiver};
use lensbolt_core::flat::{render, Alphabet, CellFormat, FlatDecoder, FlatSender, FlatSenderConfig, CAROUSEL, MANIFEST_FORMAT};
use lensbolt_core::header::FrameType;
use lensbolt_core::packet::unpack_frame;
use lensbolt_core::testutil::{bend, pseudo_random_file};
use std::path::PathBuf;

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join("lensbolt_flat_transfer"); std::fs::create_dir_all(&d).unwrap(); d.join(name)
}
fn cfg() -> FlatSenderConfig { FlatSenderConfig { manifest_every_frames: 10, transfer_id: 77, ..FlatSenderConfig::default() } }

#[test]
fn small_file_completes_through_render_and_decode() {
    let src = tmp("small.bin"); pseudo_random_file(&src, 200_000, 11).unwrap();
    let out = tmp("small_out"); let _ = std::fs::remove_dir_all(&out); std::fs::create_dir_all(&out).unwrap();
    let mut s = FlatSender::new(&src, cfg(), Some(false)).unwrap();
    assert_eq!(s.config().format, CellFormat { pitch: 4, alphabet: Alphabet::C8 });
    assert_eq!(s.grid().format(), s.config().format);
    assert_eq!(s.manifest_grid().format(), MANIFEST_FORMAT);
    let dec = FlatDecoder::new();
    let mut r = Receiver::new(&out).unwrap();
    let mut result = None;
    let budget = s.frames_per_pass() + 2;
    for n in 0..budget {
        let f = s.next_frame().unwrap();
        if n == 0 { assert_eq!(f.header.frame_type, FrameType::Manifest); assert_eq!(f.grid.format(), MANIFEST_FORMAT); }
        let im = render(&f);
        let read = dec.decode(&im).expect("decode");
        assert_eq!(read.header, f.header);
        let packets = unpack_frame(&read.payload());
        assert!(!packets.is_empty());
        if let FeedOutcome::Complete(p) = r.feed_packets(&read.header, &packets).unwrap() { result = Some(p); break; }
    }
    let p = result.expect("transfer did not complete within frames_per_pass + 2");
    assert!(std::fs::read(&p).unwrap() == std::fs::read(&src).unwrap(), "output differs");
    assert!(s.bytes_per_second(30.0) > 300_000.0, "{}", s.bytes_per_second(30.0));
    let pr = r.progress();
    assert!(pr.packets_new >= 196 && pr.packets_new <= pr.packets_accepted, "new {} accepted {}", pr.packets_new, pr.packets_accepted);
}

#[test]
fn completes_with_bend_and_splits() {
    let src = tmp("bent.bin"); pseudo_random_file(&src, 120_000, 12).unwrap();
    let out = tmp("bent_out"); let _ = std::fs::remove_dir_all(&out); std::fs::create_dir_all(&out).unwrap();
    let mut s = FlatSender::new(&src, cfg(), Some(false)).unwrap();
    let dec = FlatDecoder::new();
    let mut r = Receiver::new(&out).unwrap();
    let mut result = None;
    let budget = s.frames_per_pass() * 5 / 2 + 4;
    let mut prev = render(&s.next_frame().unwrap());
    for n in 0..budget {
        let cur = render(&s.next_frame().unwrap());
        // every third capture straddles the previous and the current displayed frame (not in step with the
        // manifest cadence: a manifest frame is one stripe, so a split manifest capture yields nothing)
        let split = n % 3 == 1;
        let mut cam = cur.clone();
        if split { let cut = cam.height / 2; for y in 0..cut { for x in 0..cam.width { cam.set(x, y, prev.get(x, y)); } } }
        let cam = bend(&cam, 2.5, n);
        let reads = dec.decode_all(&cam).unwrap_or_else(|e| panic!("capture {n} (split {split}): {e:?}"));
        for read in &reads {
            let packets = unpack_frame(&read.payload());
            if let FeedOutcome::Complete(p) = r.feed_packets(&read.header, &packets).unwrap() { result = Some(p); }
        }
        if result.is_some() { break; }
        prev = cur;
    }
    let p = result.expect("transfer did not complete");
    assert!(std::fs::read(&p).unwrap() == std::fs::read(&src).unwrap(), "output differs");
}

#[test]
fn calibration_carousel_cycles_formats() {
    let src = tmp("cal.bin"); pseudo_random_file(&src, 50_000, 13).unwrap();
    let mut s = FlatSender::new(&src, FlatSenderConfig { calibration: true, manifest_every_frames: 100, ..cfg() }, Some(false)).unwrap();
    assert_eq!(s.grids().len(), CAROUSEL.len() + 1);
    assert!(CAROUSEL.contains(&CellFormat { pitch: 2, alphabet: Alphabet::C16 }) && CAROUSEL.contains(&CellFormat { pitch: 2, alphabet: Alphabet::C8 }));
    // 2 px b/w (the default, Fast) is measured too, second after 1 px b/w
    assert!(CAROUSEL.contains(&CellFormat { pitch: 2, alphabet: Alphabet::Bw }));
    assert_eq!(CAROUSEL.len(), 18);
    let first = s.next_frame().unwrap();
    assert_eq!(first.header.frame_type, FrameType::Manifest);
    assert_eq!(first.grid.format(), MANIFEST_FORMAT);
    let mut seen = Vec::new();
    for _ in 0..CAROUSEL.len() { let f = s.next_frame().unwrap(); assert_eq!(f.header.frame_type, FrameType::Data); assert_eq!(f.header.format, f.grid.format()); seen.push(f.header.format); }
    assert_eq!(seen, CAROUSEL.to_vec());
    let again = s.next_frame().unwrap();
    assert_eq!(again.header.format, CAROUSEL[0]);
}

#[test]
fn rejects_sizes_without_a_chrome() {
    let src = tmp("bad.bin"); pseudo_random_file(&src, 1000, 14).unwrap();
    assert!(FlatSender::new(&src, FlatSenderConfig { width: 853, height: 480, ..cfg() }, Some(false)).is_err());
    assert!(FlatSender::new(&src, FlatSenderConfig { width: 256, height: 256, ..cfg() }, Some(false)).is_err(), "no room for a manifest stripe");
    assert!(FlatSender::from_bytes("x.bin", vec![1, 2, 3], cfg()).is_ok());
    assert!(FlatSender::new(&src, FlatSenderConfig { width: 568, height: 1024, ..cfg() }, Some(false)).is_err(), "no room for a header row");
    assert!(FlatSender::new(&src, FlatSenderConfig { width: 984, height: 1024, ..cfg() }, Some(false)).is_ok(), "the header wraps on narrower tiles");
    let s = FlatSender::new(&src, FlatSenderConfig { manifest_every_frames: 1, ..cfg() }, Some(false)).unwrap();
    assert_eq!(s.config().manifest_every_frames, 2, "a manifest every frame would never send data");
    assert!(FlatSender::new(&src, FlatSenderConfig { width: 1024, height: 768, ..cfg() }, Some(false)).is_ok(), "a laptop-height tile");
}


/// The live hint's restart: the user stops the sender, types the hint's words (3 px b/w) and presses Start. The same
/// file gets the same transfer id (from its content) and its packets do not depend on the format, so the receiver
/// keeps every packet it has: its bytes never drop, and it completes from the new sender in fewer frames than a
/// fresh receiver needs.
#[test]
fn restart_with_hint_words_keeps_packets() {
    let data: Vec<u8> = { let mut x = 0x2545F4914F6CDD1Du64; (0..3_000_000).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x as u8 }).collect() };
    let fast = FlatSenderConfig { format: CellFormat { pitch: 2, alphabet: Alphabet::Bw }, repair_offset: Some(11), ..FlatSenderConfig::default() };
    let reliable = FlatSenderConfig { format: CellFormat { pitch: 3, alphabet: Alphabet::Bw }, repair_offset: Some(5_000), ..FlatSenderConfig::default() };
    let mut a = FlatSender::from_bytes("big.bin", data.clone(), fast).unwrap();
    let mut rx = Receiver::in_memory();
    let mut have = 0u64;
    while rx.progress().packets_new * 1024 < data.len() as u64 * 4 / 10 {
        let (h, mut p) = a.next_packets().unwrap();
        // a poor hold: two packets of three lost
        let mut k = 0; p.retain(|q| { k += 1; q.id.segment == 0 || k % 3 == 0 });
        rx.feed_packets(&h, &p).unwrap();
        assert!(rx.progress().packets_new >= have); have = rx.progress().packets_new;
    }
    let b_cfg = reliable.clone();
    let mut b = FlatSender::from_bytes("big.bin", data.clone(), b_cfg).unwrap();
    assert_eq!(a.config().transfer_id, b.config().transfer_id, "the same file, the same transfer id");
    let mut frames_b = 0u64;
    let done = loop {
        let (h, p) = b.next_packets().unwrap();
        frames_b += 1;
        let out = rx.feed_packets(&h, &p).unwrap();
        assert!(rx.progress().packets_new >= have, "received bytes dropped"); have = rx.progress().packets_new;
        if let FeedOutcome::Complete(_) = out { break true; }
        if frames_b > 5_000 { break false; }
    };
    assert!(done);
    assert!(rx.take_file().unwrap().1 == data, "content differs");
    // a fresh receiver from the new sender alone
    let mut fresh = Receiver::in_memory();
    let mut c = FlatSender::from_bytes("big.bin", data.clone(), reliable).unwrap();
    let mut frames_fresh = 0u64;
    loop { let (h, p) = c.next_packets().unwrap(); frames_fresh += 1; if let FeedOutcome::Complete(_) = fresh.feed_packets(&h, &p).unwrap() { break; } assert!(frames_fresh < 5_000); }
    assert!(frames_b < frames_fresh, "after the restart {frames_b} frames, fresh {frames_fresh}");
}
