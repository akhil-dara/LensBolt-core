//! A receiver that sees a tuning sender: it names it (the status's "tune_sender"), never locks on it and never
//! creates a store for it, and receives a normal transfer started right after.
use lensbolt_core::file::{FeedOutcome, PartStore, Receiver, StoreFactory, TuneInfo};
use lensbolt_core::flat::session::RxSession;
use lensbolt_core::flat::{render, FlatSender, FlatSenderConfig};
use lensbolt_core::tune::schedule::SCHEDULE_VERSION;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn pseudo_random(len: usize, seed: u64) -> Vec<u8> {
    let mut x = seed | 1;
    (0..len).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x as u8 }).collect()
}
fn rgba(img: &lensbolt_core::image::RgbImage) -> Vec<u8> {
    let mut v = Vec::with_capacity(img.data.len() / 3 * 4);
    for p in img.data.chunks_exact(3) { v.extend_from_slice(&[p[0], p[1], p[2], 255]); }
    v
}

#[test]
fn tune_frames_never_lock_and_never_store() {
    let mut rx = Receiver::in_memory();
    let mut tune = FlatSender::tune(FlatSenderConfig::default(), true).unwrap();
    for _ in 0..200 { let (h, p) = tune.next_packets().unwrap(); rx.feed_packets(&h, &p).unwrap(); }
    let p = rx.progress();
    assert_eq!(p.tune_sender, Some(TuneInfo { schedule: SCHEDULE_VERSION, display_60: true }));
    assert!(!p.have_manifest && p.stream_size == 0 && p.disk_peak == 0 && p.filename.is_none(), "{p:?}");
    // a real transfer right after is received
    let data = pseudo_random(300_000, 7);
    let mut s = FlatSender::from_bytes("a.bin", data.clone(), FlatSenderConfig::default()).unwrap();
    let mut done = false;
    for _ in 0..2000 { let (h, p) = s.next_packets().unwrap(); if let FeedOutcome::Complete(_) = rx.feed_packets(&h, &p).unwrap() { done = true; break; } }
    assert!(done);
    assert_eq!(rx.take_file().unwrap().1, data);
    assert_eq!(rx.progress().tune_sender, None);
}

#[test]
fn tune_frames_after_the_manifest_are_ignored() {
    let mut rx = Receiver::in_memory();
    let mut tune = FlatSender::tune(FlatSenderConfig::default(), false).unwrap();
    // the first frame is the manifest: from it on, every frame of the tune transfer is ignored
    for n in 0..100 {
        let (h, p) = tune.next_packets().unwrap();
        assert_eq!(rx.feed_packets(&h, &p).unwrap(), FeedOutcome::Ignored, "frame {n}");
    }
    assert_eq!(rx.progress().tune_sender, Some(TuneInfo { schedule: SCHEDULE_VERSION, display_60: false }));
    assert_eq!(rx.live_decoders(), 0);
}

#[derive(Clone, Default)]
struct Counting { creates: Arc<AtomicUsize> }
struct Nothing;
impl PartStore for Nothing {
    fn read_at(&mut self, _: u64, _: &mut [u8]) -> std::io::Result<()> { Ok(()) }
    fn write_at(&mut self, _: u64, _: &[u8]) -> std::io::Result<()> { Ok(()) }
    fn set_len(&mut self, _: u64) -> std::io::Result<()> { Ok(()) }
}
impl StoreFactory for Counting {
    fn create(&mut self, _: &str) -> std::io::Result<Box<dyn PartStore>> { self.creates.fetch_add(1, Ordering::SeqCst); Ok(Box::new(Nothing)) }
    fn remove(&mut self, _: &str) {}
}

#[test]
fn a_store_backed_receiver_creates_no_store_for_tune() {
    let stores = Counting::default();
    let mut rx = Receiver::with_stores(Box::new(stores.clone()));
    let mut tune = FlatSender::tune(FlatSenderConfig::default(), true).unwrap();
    // start mid-loop: data frames first, then the manifest
    for _ in 0..5 { tune.next_packets().unwrap(); }
    for _ in 0..200 { let (h, p) = tune.next_packets().unwrap(); rx.feed_packets(&h, &p).unwrap(); }
    assert_eq!(stores.creates.load(Ordering::SeqCst), 0);
    assert!(rx.progress().tune_sender.is_some());
    assert!(!rx.progress().have_manifest);
}

#[test]
fn status_says_a_sender_is_tuning() {
    let mut rx = RxSession::new();
    let cfg = FlatSenderConfig { width: 1024, height: 640, ..FlatSenderConfig::default() };
    let mut tune = FlatSender::tune(cfg.clone(), true).unwrap();
    let mut j = String::new();
    for _ in 0..4 { let f = tune.next_frame().unwrap(); let img = render(&f); j = rx.push_rgba(&rgba(&img), img.width, img.height); }
    assert!(j.contains("\"tune_sender\":true"), "{j}");
    assert!(j.contains("\"have_manifest\":false") && j.contains("\"stream_size\":0"), "{j}");
    let data = pseudo_random(20_000, 3);
    let mut s = FlatSender::from_bytes("b.bin", data, cfg).unwrap();
    let mut saw_manifest = false;
    for _ in 0..6 {
        let f = s.next_frame().unwrap(); let img = render(&f); j = rx.push_rgba(&rgba(&img), img.width, img.height);
        if j.contains("\"have_manifest\":true") { saw_manifest = true; assert!(j.contains("\"tune_sender\":false"), "{j}"); }
    }
    assert!(saw_manifest, "{j}");
}

/// A real transfer that carries the tune id (an older sender, or one told to): its manifest frames are still read,
/// so its manifest clears the tune record and the file is received.
#[test]
fn a_real_manifest_under_the_tune_id_is_received() {
    use lensbolt_core::tune::schedule::tune_transfer_id;
    let mut rx = Receiver::in_memory();
    let mut tune = FlatSender::tune(FlatSenderConfig::default(), true).unwrap();
    for _ in 0..100 { let (h, p) = tune.next_packets().unwrap(); rx.feed_packets(&h, &p).unwrap(); }
    assert!(rx.progress().tune_sender.is_some());
    let data = pseudo_random(200_000, 11);
    let mut s = FlatSender::from_bytes("same-id.bin", data.clone(), FlatSenderConfig { transfer_id: tune_transfer_id(true), ..FlatSenderConfig::default() }).unwrap();
    let mut done = false;
    for _ in 0..2000 { let (h, p) = s.next_packets().unwrap(); if let FeedOutcome::Complete(_) = rx.feed_packets(&h, &p).unwrap() { done = true; break; } }
    assert!(done, "{:?}", rx.progress());
    assert_eq!(rx.take_file().unwrap().1, data);
    assert_eq!(rx.progress().tune_sender, None);
}

/// A tune manifest frame read only in part leaves no lock that keeps another transfer out.
#[test]
fn a_part_read_tune_manifest_does_not_block_a_transfer() {
    let mut rx = Receiver::in_memory();
    let mut tune = FlatSender::tune(FlatSenderConfig::default(), false).unwrap();
    for _ in 0..40 { let (h, p) = tune.next_packets().unwrap(); rx.feed_packets(&h, &p).unwrap(); }
    assert!(rx.progress().tune_sender.is_some());
    // the next tune manifest frame arrives with its packets lost (a partial read)
    loop {
        let (h, mut p) = tune.next_packets().unwrap();
        if h.frame_type == lensbolt_core::header::FrameType::Manifest { p.clear(); rx.feed_packets(&h, &p).unwrap(); break; }
    }
    let data = pseudo_random(150_000, 5);
    let mut s = FlatSender::from_bytes("next.bin", data.clone(), FlatSenderConfig::default()).unwrap();
    let mut done = false;
    for _ in 0..2000 { let (h, p) = s.next_packets().unwrap(); if let FeedOutcome::Complete(_) = rx.feed_packets(&h, &p).unwrap() { done = true; break; } }
    assert!(done, "{:?}", rx.progress());
    assert_eq!(rx.take_file().unwrap().1, data);
}
