//! What the camera thread copies: the Y plane always, U and V only when the receiver asks for them
//! (`LiveReceiver::want_chroma`). A b/w tile reads from Y alone; the colour (c8) manifest needs chroma, and
//! must still be read within a second of frames when most captures carry only Y.
use lensbolt_core::flat::live::{Capture, CopyPolicy, LiveConfig, LiveReceiver, LiveStatus};
use lensbolt_core::flat::{render, Alphabet, CellFormat, FlatSender, FlatSenderConfig, MANIFEST_EVERY_FRAMES};
use lensbolt_core::image::RgbImage;
use lensbolt_core::testutil::pseudo_random_file;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// One receiver at a time (each runs its own decode pool).
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
fn serial() -> std::sync::MutexGuard<'static, ()> { SERIAL.lock().unwrap_or_else(|e| e.into_inner()) }

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join("lensbolt_live_copy"); std::fs::create_dir_all(&d).unwrap(); d.join(name)
}
fn out_dir(name: &str) -> PathBuf { let o = tmp(name); let _ = std::fs::remove_dir_all(&o); std::fs::create_dir_all(&o).unwrap(); o }
fn ms(n: u64) -> Duration { Duration::from_millis(n) }

/// The phone's data format: b/w 2 px data, the colour c8 manifest every MANIFEST_EVERY_FRAMES (31) frames.
const BW2: CellFormat = CellFormat { pitch: 2, alphabet: Alphabet::Bw };
fn sender(src: &PathBuf, bytes: u64, seed: u64, id: u16) -> FlatSender { sender_in(src, bytes, seed, id, BW2) }
fn sender_in(src: &PathBuf, bytes: u64, seed: u64, id: u16, format: CellFormat) -> FlatSender {
    pseudo_random_file(src, bytes, seed).unwrap();
    let cfg = FlatSenderConfig { manifest_every_frames: MANIFEST_EVERY_FRAMES, transfer_id: id, format, ..FlatSenderConfig::default() };
    FlatSender::new(src, cfg, Some(false)).unwrap()
}

/// What the JNI layer does with one camera image: nothing copied when no worker is free (counted), else Y
/// always and U/V only when `chroma`. Returns whether the capture was accepted.
fn offer(live: &LiveReceiver, img: RgbImage, chroma: bool) -> bool { offer_at(live, img, chroma, Instant::now()) }
/// `offer` with the capture's time (a simulated camera clock).
fn offer_at(live: &LiveReceiver, img: RgbImage, chroma: bool, at: Instant) -> bool {
    if !live.has_room() { live.skip(); return false; }
    live.submit_capture_at(at, move || {
        let y = img.to_gray();
        Capture { y, rgb: if chroma { Some(Box::new(move || img)) } else { None } }
    })
}

/// Wait until every accepted capture has come back from the workers (the policy has seen it).
fn settle(live: &LiveReceiver, accepted: u64) -> LiveStatus {
    let t0 = Instant::now();
    loop {
        let st = live.status();
        if st.decoded + st.failed + st.colour_skipped >= accepted || st.complete.is_some() { return st; }
        assert!(t0.elapsed() < Duration::from_secs(60), "captures never came back: {st:?}");
        std::thread::sleep(ms(1));
    }
}

#[test]
fn chroma_copied_until_the_first_luma_decode() {
    let _g = serial();
    // the rule alone: while nothing reads (no tile, or failures) every capture copies chroma
    let t = Instant::now();
    let mut p = CopyPolicy::new();
    assert!(p.want_chroma(), "nothing read yet");
    for i in 1..=8 { p.observe(false, false, t + ms(33 * i)); assert!(p.want_chroma(), "capture {i}: no tile read yet"); }

    // the receiver: captures with no tile, then the sender from its first frame (the manifest), chroma as asked
    let src = tmp("first.bin");
    let mut s = sender(&src, 4_000_000, 41, 141);
    let live = LiveReceiver::new(&out_dir("first_out"), LiveConfig { workers: 2 }).unwrap();
    let mut accepted = 0;
    for i in 0..5 {
        assert!(live.want_chroma(), "blank capture {i}: no tile read yet");
        assert!(offer(&live, RgbImage::new(1024, 1024), live.want_chroma()));
        accepted += 1; settle(&live, accepted);
    }
    // the manifest is colour: read from chroma, and still no luma tile
    assert!(live.want_chroma());
    assert!(offer(&live, render(&s.next_frame().unwrap()), live.want_chroma()));
    accepted += 1;
    let st = settle(&live, accepted);
    assert!(st.have_manifest && st.decoded == 1, "{st:?}");
    assert!(live.want_chroma(), "no luma tile read yet: {st:?}");
    // the first b/w data frame reads on luma; from then on only the cadence copies chroma
    assert!(offer(&live, render(&s.next_frame().unwrap()), live.want_chroma()));
    accepted += 1;
    let st = settle(&live, accepted);
    assert!(st.decoded == 2 && st.luma_ms > 0.0, "{st:?}");
    let mut wants = Vec::new();
    for _ in 0..8 {
        let w = live.want_chroma(); wants.push(w);
        assert!(offer(&live, render(&s.next_frame().unwrap()), w));
        accepted += 1; settle(&live, accepted);
    }
    assert!(wants.iter().filter(|w| **w).count() <= 2, "chroma after the first luma decode: {wants:?}");
    let st = live.status();
    assert_eq!(st.decoded, 10, "every data frame read from Y alone: {st:?}");
}

#[test]
fn chroma_skipped_while_a_bw_tile_decodes() {
    let _g = serial();
    // the rule alone: luma reads with the manifest known copy chroma on the cadence only (1 in 4)
    let t = Instant::now();
    let mut p = CopyPolicy::new();
    p.observe(false, true, t);
    assert!(p.want_chroma(), "a luma read, but the manifest is not known yet");
    p.manifest_known(true);
    let mut wants = Vec::new();
    for i in 1..=40 { p.observe(false, true, t + ms(33 * i)); wants.push(p.want_chroma()); }
    assert_eq!(wants.iter().filter(|w| **w).count(), 10, "{wants:?}");
    // a colour header seen without chroma: the next capture copies it, and stops once luma reads again
    let t1 = t + ms(33 * 41);
    p.observe(true, false, t1);
    assert!(p.need_chroma(), "the last capture asked for colour");
    p.observe(false, true, t1 + ms(33));
    assert!(!p.need_chroma());
    // no luma read for more than a second (the tile lost): chroma on every capture until one reads
    p.observe(false, false, t1 + ms(33 + 900));
    assert!(!p.need_chroma(), "a luma read 0.9 s ago still vouches for b/w");
    p.observe(false, false, t1 + ms(33 + 1100));
    assert!(p.need_chroma(), "no luma read for 1.1 s");

    let src = tmp("bw.bin");
    let mut s = sender(&src, 6_000_000, 42, 142);
    let live = LiveReceiver::new(&out_dir("bw_out"), LiveConfig { workers: 2 }).unwrap();
    // the manifest and one data frame: the receiver knows the transfer and reads luma
    let mut accepted = 0;
    for _ in 0..2 { assert!(offer(&live, render(&s.next_frame().unwrap()), live.want_chroma())); accepted += 1; settle(&live, accepted); }
    let st = live.status();
    assert!(st.have_manifest && st.decoded == 2, "{st:?}");
    // a run of 88 frames (two manifests among them, which a Y-only capture cannot read): chroma on about 1 in 4
    let n = 88u64;
    let mut chroma = 0u64;
    for _ in 0..n {
        let w = live.want_chroma(); if w { chroma += 1; }
        assert!(offer(&live, render(&s.next_frame().unwrap()), w));
        accepted += 1; settle(&live, accepted);
    }
    let st = live.status();
    assert!(st.complete.is_none(), "the file must outlast the run: {st:?}");
    // the cadence (n/4), plus one capture after each manifest seen without chroma
    assert!(chroma <= n / 4 + 1 + 2, "{chroma} of {n} captures copied chroma: {st:?}");
    assert!(st.decoded >= 2 + n - 2, "b/w frames must read from Y alone: {st:?}");
    assert_eq!(st.failed, 0, "{st:?}");
}

/// Frames from one manifest to the next (31, ~1 s at 30 fps): a receiver joining anywhere is shown one within
/// this many frames.
const PERIOD: u64 = MANIFEST_EVERY_FRAMES as u64;
/// One replay of the sender at 30 fps through the receiver, starting `offset` frames into the stream, chroma
/// as the policy asks (or on every capture). Stops after one manifest period of frames unless `to_end`. The
/// camera's clock is simulated: capture `i` shows frame `i` and is offered at `i / 30` s, and each capture's
/// result is in before the next is offered, so a busy laptop drops no capture and two runs see the same
/// captures (the wall clock decides nothing). `late`: (chroma copies, accepted captures) after the first period
/// of frames. `paced_in` replays `bytes` of `format` data.
struct Run { manifest_in_period: bool, frames_to_complete: Option<u64>, chroma: u64, offered: u64, late: (u64, u64), status: LiveStatus }
fn paced(src: &PathBuf, offset: u64, policy: bool, to_end: bool, tag: &str) -> Run { paced_in(src, BW2, 2_500_000, offset, policy, to_end, tag) }
fn paced_in(src: &PathBuf, format: CellFormat, bytes: u64, offset: u64, policy: bool, to_end: bool, tag: &str) -> Run {
    let mut s = sender_in(src, bytes, 43, 143, format);
    for _ in 0..offset { s.next_frame().unwrap(); }
    let live = LiveReceiver::new(&out_dir(&format!("paced_{tag}")), LiveConfig { workers: 4 }).unwrap();
    let period = Duration::from_nanos(1_000_000_000 / 30);
    let deadline = Instant::now() + Duration::from_secs(600);
    let base = Instant::now();
    let (mut accepted, mut chroma, mut last_accepted, mut manifest_in_period, mut late) = (0u64, 0u64, 0u64, false, (0u64, 0u64));
    let mut i = 0u64;
    loop {
        assert!(Instant::now() < deadline, "no completion: {:?}", live.status());
        if i == PERIOD {
            // one manifest period of frames shown and read: the manifest must be known
            manifest_in_period = live.status().have_manifest;
            if !to_end { break; }
        }
        let img = render(&s.next_frame().unwrap());
        let w = !policy || live.want_chroma();
        assert!(offer_at(&live, img, w, base + period * i as u32), "capture {i}: the previous one is back, a worker is free: {:?}", live.status());
        accepted += 1; last_accepted = i; if w { chroma += 1; }
        if i >= PERIOD { late.1 += 1; if w { late.0 += 1; } }
        settle(&live, accepted);
        i += 1;
        if live.status().complete.is_some() { break; }
    }
    let status = live.status();
    if let Some(p) = &status.complete { assert!(std::fs::read(p).unwrap() == std::fs::read(src).unwrap(), "output differs"); }
    Run { manifest_in_period, frames_to_complete: status.complete.as_ref().map(|_| last_accepted + 1), chroma, offered: i, late, status }
}

#[test]
fn manifest_read_within_one_period_with_y_only_copies() {
    let _g = serial();
    let src = tmp("paced.bin");
    // joining the stream 1..4 frames in: the first manifest falls on each position of the chroma cadence
    let mut missed = Vec::new();
    for offset in 1..=4 {
        let r = paced(&src, offset, true, false, &format!("m{offset}"));
        println!("offset {offset}: manifest in one period {}, {} chroma copies of {}, colour skipped {}", r.manifest_in_period, r.chroma, r.offered, r.status.colour_skipped);
        if !r.manifest_in_period { missed.push((offset, r.chroma, r.status.colour_skipped)); }
    }
    assert!(missed.is_empty(), "manifest not known after one period of frames at (offset, chroma copies, colour skipped) {missed:?}");
    // the whole file with the policy and with chroma on every capture
    let with = paced(&src, 1, true, true, "policy");
    let all = paced(&src, 1, false, true, "all");
    let (fw, fa) = (with.frames_to_complete.expect("policy run completes"), all.frames_to_complete.expect("all-chroma run completes"));
    println!("frames to complete: policy {fw} ({} of {} captures copied chroma; after the first period {} of {}), all chroma {fa}", with.chroma, with.offered, with.late.0, with.late.1);
    assert!(fw * 100 <= fa * 105, "policy needed {fw} frames, all chroma {fa}: {:?}", with.status);
    // once the manifest is read: the cadence (1 in 4) plus one capture after each manifest seen without chroma
    let (c, n) = with.late;
    assert!(n > PERIOD && c <= n / 4 + 1 + n / PERIOD + 1, "after the first period {c} of {n} captures copied chroma: {:?}", with.status);
}

#[test]
fn colour_data_copies_chroma_on_every_capture() {
    let _g = serial();
    // colour data (4 px, 8 colours): no capture ever reads on luma, so the policy copies chroma on all of them
    // and the file completes on the same captures as with chroma always copied
    let src = tmp("colour.bin");
    let c8 = CellFormat { pitch: 4, alphabet: Alphabet::C8 };
    let with = paced_in(&src, c8, 800_000, 1, true, true, "c8_policy");
    let all = paced_in(&src, c8, 800_000, 1, false, true, "c8_all");
    let (fw, fa) = (with.frames_to_complete.expect("policy run completes"), all.frames_to_complete.expect("all-chroma run completes"));
    println!("c8 4px colour data: policy {fw} frames ({} of {} captures copied chroma, colour skipped {}, dropped {}), all chroma {fa} (dropped {})",
        with.chroma, with.offered, with.status.colour_skipped, with.status.dropped, all.status.dropped);
    assert_eq!(with.status.colour_skipped, 0, "{:?}", with.status);
    assert_eq!(with.chroma, with.offered - with.status.dropped, "every accepted capture copied chroma: {:?}", with.status);
    assert_eq!(with.status.failed, 0, "{:?}", with.status);
    assert_eq!((with.status.dropped, all.status.dropped), (0, 0));
    assert_eq!(fw, fa, "policy needed {fw} frames, all chroma {fa}: {:?}", with.status);
}

#[test]
fn a_colour_header_without_chroma_asks_until_the_camera_thread_copies() {
    let _g = serial();
    // a capture whose header asked for colour came back without chroma; before the camera thread asks again,
    // a later capture reads on luma. The request must still be there for the camera thread, once.
    let src = tmp("starved.bin");
    let mut s = sender(&src, 6_000_000, 44, 144);
    let live = LiveReceiver::new(&out_dir("starved_out"), LiveConfig { workers: 2 }).unwrap();
    let m = MANIFEST_EVERY_FRAMES as usize;
    let frames: Vec<_> = (0..m + 2).map(|_| s.next_frame().unwrap()).collect();
    // the manifest with chroma, a data frame on luma: the transfer is known and the tile b/w
    assert!(offer(&live, render(&frames[0]), true));
    settle(&live, 1);
    assert!(offer(&live, render(&frames[1]), false));
    let st = settle(&live, 2);
    assert!(st.have_manifest && st.decoded == 2 && st.luma_ms > 0.0, "{st:?}");
    // the next manifest (frame 31) with Y only, then a data frame read on luma
    assert!(offer(&live, render(&frames[m]), false));
    let st = settle(&live, 3);
    assert_eq!(st.colour_skipped, 1, "{st:?}");
    assert!(offer(&live, render(&frames[m + 1]), false));
    let st = settle(&live, 4);
    assert_eq!(st.decoded, 3, "{st:?}");
    // 4 captures accepted: not the cadence's turn; the policy alone no longer asks (the last read was luma)
    assert!(live.want_chroma(), "the colour request was lost before the camera thread saw it");
    assert!(!live.want_chroma(), "the request is taken once");
}
