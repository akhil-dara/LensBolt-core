//! Completion guarantee at the packet level: every transfer completes bit-exact within a bound of frames when
//! the camera loses whole frames (at random or in long bursts), captures only part of each frame, or joins
//! after the first pass. Bounds are in frames shown by the sender (2 px bw, 1024x1024, the sender's manifest
//! period, MANIFEST_EVERY_FRAMES).
//!     cargo test --release --test completion -- --nocapture
use lensbolt_core::file::batch::{Entry, Index};
use lensbolt_core::file::packets::PacketSource;
use lensbolt_core::file::receiver::{BatchStatus, FeedOutcome, FileStage, PartStore, Receiver, StoreFactory};
use lensbolt_core::file::{BatchInfo, BatchSource, Prepared};
use lensbolt_core::header::{FrameInfo, FrameType};
use lensbolt_core::finish::encode;
use lensbolt_core::flat::sender::{FlatSender, FlatSenderConfig, MANIFEST_EVERY_FRAMES};
use lensbolt_core::flat::{Alphabet, CellFormat};
use lensbolt_core::packet::{Packet, PacketId};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

fn xorshift(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }
fn data(len: usize) -> Vec<u8> { let mut s = 0x1234_5678u64; (0..len).map(|_| xorshift(&mut s) as u8).collect() }
fn sender(data: &[u8], pitch: u8) -> FlatSender {
    let cfg = FlatSenderConfig { width: 1024, height: 1024, format: CellFormat { pitch, alphabet: Alphabet::Bw }, manifest_every_frames: MANIFEST_EVERY_FRAMES, transfer_id: 9, ..FlatSenderConfig::default() };
    FlatSender::from_bytes("completion.bin", data.to_vec(), cfg).unwrap()
}
fn frames_per_pass(data: &[u8]) -> u64 { sender(data, 2).frames_per_pass() }

/// Frames before `start` are never seen; `drop` returns false to lose the whole frame and may remove packets
/// from it (a partial capture). Returns the frames shown at completion and the received bytes.
fn run(data: &[u8], pitch: u8, mut drop: impl FnMut(u64, &mut Vec<Packet>) -> bool, start: u64, max_frames: u64) -> Option<(u64, Vec<u8>)> {
    let mut tx = sender(data, pitch);
    let mut rx = Receiver::in_memory();
    for n in 0..max_frames {
        let (h, mut p) = tx.next_packets().unwrap();
        if n < start || !drop(n, &mut p) { continue; }
        if let FeedOutcome::Complete(_) = rx.feed_packets(&h, &p).unwrap() { return Some((n + 1, rx.take_file().expect("in-memory file").1)); }
    }
    None
}

/// Runs to 4x the bound so a miss still reports how far it was, then asserts bit-exact within the bound.
fn check(name: &str, data: &[u8], drop: impl FnMut(u64, &mut Vec<Packet>) -> bool, start: u64, bound: f64) {
    let got = run(data, 2, drop, start, (bound * 4.0) as u64 + 200);
    let frames = got.as_ref().map(|g| g.0);
    eprintln!("{name}: {} bytes, complete at {frames:?} frames, bound {bound:.0} ({} frames per pass)", data.len(), frames_per_pass(data));
    let (frames, bytes) = got.unwrap_or_else(|| panic!("{name}: never completed"));
    assert!(bytes == data, "{name}: received bytes differ from the input");
    assert!(frames as f64 <= bound, "{name}: {frames} frames > bound {bound:.0}");
}

const MB4: usize = 4_000_000;

#[test]
fn random_frame_loss_completes() {
    let d = data(MB4);
    let mut r = 99u64;
    check("random_frame_loss", &d, |_, _| xorshift(&mut r) % 1000 >= 300, 0, 1.25 * frames_per_pass(&d) as f64 / 0.7 + 60.0);
}

#[test]
fn long_bursts_of_loss_complete() {
    let d = data(MB4);
    check("long_bursts_of_loss", &d, |n, _| n % 50 >= 20, 0, 1.25 * frames_per_pass(&d) as f64 / 0.6 + 60.0);
}

#[test]
fn half_captured_frames_complete() {
    let d = data(MB4);
    let mut r = 7u64;
    // a random half of each frame's packets survives (partial Fisher-Yates, then keep the first half); an odd
    // count keeps its middle packet with probability 1/2: a manifest frame is one packet, and rounding it down
    // to none every time meant the manifest never arrived
    check("half_captured_frames", &d, |_, p| {
        let keep = p.len() / 2 + (p.len() % 2 == 1 && xorshift(&mut r) % 2 == 0) as usize;
        for i in 0..keep { let j = i + (xorshift(&mut r) % (p.len() - i) as u64) as usize; p.swap(i, j); }
        p.truncate(keep); true
    }, 0, 1.25 * frames_per_pass(&d) as f64 / 0.5 + 60.0);
}

#[test]
fn joining_after_the_first_pass_completes() {
    let d = data(MB4);
    let per_pass = frames_per_pass(&d);
    let start = per_pass + 7;
    check("joining_after_the_first_pass", &d, |_, _| true, start, start as f64 + 1.3 * per_pass as f64 + 60.0);
}

#[test]
fn joining_during_the_first_pass_completes_in_about_one_pass() {
    // the receiver starts 40% of the way into the first pass (the phone aimed after the sender started), then
    // sees every frame: it must need about one pass of frames from there on. Measured
    // (docs/2026-10-30-repair-schedule.md): segment by segment, the pass left the first segments wholly to the
    // blind repair rounds, 1.6 passes after joining; with every chunk's blocks spread over the whole pass, 1.00
    let d = data(40 * 1024 * 1024); // 3 segments: the first 40% of the pass is all of segment 1
    let per_pass = frames_per_pass(&d);
    let start = per_pass * 2 / 5;
    check("joining_during_the_first_pass", &d, |_, _| true, start, start as f64 + 1.1 * per_pass as f64 + 60.0);
}

#[test]
fn a_camera_at_exactly_half_the_sender_rate_completes() {
    // a camera phase-locked at half the sender's frame rate sees every other frame shown. With a manifest every
    // 30 frames (all even) the camera that sees the odd frames never saw one and never completed; the default
    // period is odd, so both phases see a manifest every other period. Bound: as for random loss at 50%
    let d = data(MB4);
    let cfg = FlatSenderConfig { format: CellFormat { pitch: 2, alphabet: Alphabet::Bw }, transfer_id: 9, ..FlatSenderConfig::default() };
    for phase in [1u64, 0] {
        let mut tx = FlatSender::from_bytes("half.bin", d.clone(), cfg.clone()).unwrap();
        let bound = 1.25 * tx.frames_per_pass() as f64 / 0.5 + 60.0;
        let mut rx = Receiver::in_memory();
        let mut done = None;
        for n in 0..(bound * 4.0) as u64 {
            let (h, p) = tx.next_packets().unwrap();
            if n % 2 != phase { continue; }
            if let FeedOutcome::Complete(_) = rx.feed_packets(&h, &p).unwrap() { done = Some(n + 1); break; }
        }
        eprintln!("half rate, phase {phase}: complete at {done:?} frames shown, bound {bound:.0} (manifest every {})", cfg.manifest_every_frames);
        let frames = done.unwrap_or_else(|| panic!("the camera that sees the {} frames never completed", if phase == 1 { "odd" } else { "even" }));
        assert!(rx.take_file().expect("in-memory file").1 == d, "phase {phase}: received bytes differ from the input");
        assert!(frames as f64 <= bound, "phase {phase}: {frames} frames > bound {bound:.0}");
    }
}

#[test]
fn a_lossless_transfer_completes_within_one_pass() {
    // the first pass still carries every block once, in full frames: a receiver that misses nothing is done by
    // the pass's last frame (throughput unchanged by the order of the pass)
    let d = data(40 * 1024 * 1024 + 4321);
    check("lossless", &d, |_, _| true, 0, frames_per_pass(&d) as f64);
}

#[test]
fn empty_file_completes() {
    check("empty_file", &[], |_, _| true, 0, 120.0);
}

#[test]
fn segment_boundary_plus_one_completes() {
    let d = data(16 * 1024 * 1024 + 1);
    let mut r = 31u64;
    check("segment_boundary_plus_one", &d, |_, _| xorshift(&mut r) % 1000 >= 100, 0, 1.25 * frames_per_pass(&d) as f64 / 0.9 + 60.0);
}

/// 4 segments; the first pass loses every packet of segment 2, later frames are lost at 30%. With `words`,
/// the finish words for segment 2 (unit 1) are typed once the pass has gone by. Returns the frames shown at
/// completion and the received bytes.
fn finish_arm(d: &[u8], words: bool) -> Option<(u64, Vec<u8>)> {
    let mut tx = sender(d, 2);
    let per_pass = tx.frames_per_pass();
    let mut rx = Receiver::in_memory();
    let mut r = 41u64;
    for n in 0..per_pass * 8 {
        if words && n == per_pass {
            // segments 1, 3 and 4 are in, 2 short: the unit the test withheld (the receiver's own list of short
            // units comes with the receiver's words)
            assert_eq!(rx.progress().segments_done, 3);
            let code = encode(tx.config().transfer_id, 4, &[1]).unwrap();
            assert_eq!(tx.finish(&code), Ok(vec![1]));
        }
        let (h, mut p) = tx.next_packets().unwrap();
        if n < per_pass {
            p.retain(|q| q.id.segment != 2);
            if p.is_empty() { continue; }
        } else if xorshift(&mut r) % 1000 < 300 { continue; }
        if let FeedOutcome::Complete(_) = rx.feed_packets(&h, &p).unwrap() { return Some((n + 1, rx.take_file().expect("in-memory file").1)); }
    }
    None
}

#[test]
fn finish_words_end_a_transfer_sooner() {
    let d = data(64 * 1024 * 1024);
    let (a, a_bytes) = finish_arm(&d, false).expect("normal schedule never completed");
    let (b, b_bytes) = finish_arm(&d, true).expect("finish words never completed");
    eprintln!("finish_words: normal schedule {a} frames, finish words {b} frames ({:.2}x), {} frames per pass", b as f64 / a as f64, frames_per_pass(&d));
    assert!(a_bytes == d, "normal schedule: received bytes differ from the input");
    assert!(b_bytes == d, "finish words: received bytes differ from the input");
    assert!(b as f64 <= 0.7 * a as f64, "finish words took {b} frames, more than 0.7x the normal schedule's {a}");
}

#[test]
fn same_file_same_transfer_id() {
    let first_id = |mut s: FlatSender| s.next_packets().unwrap().0.transfer_id;
    let id = |name: &str, d: &[u8]| first_id(FlatSender::from_bytes(name, d.to_vec(), FlatSenderConfig::default()).unwrap());
    let d = data(100_000);
    let a = id("report.pdf", &d);
    assert_eq!(a, id("report.pdf", &d), "the same file under the same name");
    assert_eq!(a & 1, 1, "the low bit is set");
    assert_ne!(a, id("report-final.pdf", &d), "the same bytes under another name");
    assert_ne!(a, id("report.pdf", &d[..99_999]), "other bytes under the same name");
    let (e1, e2) = (id("a.txt", &[]), id("b.txt", &[]));
    assert_ne!(e1, e2, "two empty files under different names");
    assert!(e1 & 1 == 1 && e2 & 1 == 1);
    // the stream is what the id names: a compressed stream of a file is another transfer than the file as-is
    let dir = std::env::temp_dir().join(format!("lensbolt-content-id-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("notes.txt");
    let text: Vec<u8> = b"the same line, again and again\n".iter().copied().cycle().take(200_000).collect();
    std::fs::write(&path, &text).unwrap();
    let on_disk = |compress: bool| first_id(FlatSender::new(&path, FlatSenderConfig::default(), Some(compress)).unwrap());
    let (plain, packed) = (on_disk(false), on_disk(true));
    assert_eq!(plain, id("notes.txt", &text), "the same bytes, name and preparation (as-is) on disk and in memory");
    assert_eq!(packed, on_disk(true), "the same bytes, name and preparation (compressed)");
    assert_ne!(plain, packed, "the same file compressed and not");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn finish_words_on_a_one_segment_file() {
    // 5.5 MiB: one segment of 6 chunks, the units of its finish words; the first pass loses every packet of
    // chunks 1 and 5, later frames are lost at 20%
    let d = data(5 * 1024 * 1024 + 512 * 1024);
    let mut tx = sender(&d, 2);
    let per_pass = tx.frames_per_pass();
    let mut rx = Receiver::in_memory();
    let mut r = 53u64;
    for n in 0..per_pass * 8 {
        if n == per_pass { assert_eq!(tx.finish(&encode(tx.config().transfer_id, 6, &[1, 5]).unwrap()), Ok(vec![1, 5])); }
        let (h, mut p) = tx.next_packets().unwrap();
        if n < per_pass {
            p.retain(|q| !(q.id.segment == 1 && (q.id.chunk == 1 || q.id.chunk == 5)));
            if p.is_empty() { continue; }
        } else {
            // only the named chunks until each has had its repair rounds (2 per block, at 20% loss the file completes first)
            assert!(!tx.focus_done(), "frame {n}: the sender left the words before the file completed");
            for q in p.iter().filter(|q| q.id.segment != 0) { assert!(q.id.segment == 1 && (q.id.chunk == 1 || q.id.chunk == 5), "frame {n}: {:?} after the words named chunks 1 and 5", q.id); }
            if xorshift(&mut r) % 1000 < 200 { continue; }
        }
        if let FeedOutcome::Complete(_) = rx.feed_packets(&h, &p).unwrap() {
            eprintln!("finish_words_on_a_one_segment_file: complete at {} frames ({per_pass} frames per pass)", n + 1);
            assert!(rx.take_file().expect("in-memory file").1 == d, "received bytes differ from the input");
            return;
        }
    }
    panic!("never completed");
}

/// The first `count` data packet ids a sender emits.
fn data_ids(tx: &mut FlatSender, count: usize) -> Vec<PacketId> {
    let mut ids = Vec::with_capacity(count);
    while ids.len() < count { ids.extend(tx.next_packets().unwrap().1.iter().filter(|q| q.id.segment != 0).map(|q| q.id)); }
    ids.truncate(count);
    ids
}

#[test]
fn resumed_sender_sends_fresh_repair_ids() {
    let d = data(16 * 1024 * 1024 + 1); // two segments: unit 0 is segment 1
    let focused = || {
        let mut tx = sender(&d, 2);
        tx.finish(&encode(tx.config().transfer_id, 2, &[0]).unwrap()).unwrap();
        let ids = data_ids(&mut tx, 200);
        for id in &ids { assert!(id.segment == 1 && id.pid >= 1024, "{id:?} is not a repair packet of segment 1"); }
        (tx, ids)
    };
    let (mut first, a) = focused();
    let (_, b) = focused();
    let (a, b): (HashSet<PacketId>, HashSet<PacketId>) = (a.into_iter().collect(), b.into_iter().collect());
    assert_eq!((a.len(), b.len()), (200, 200), "a sender repeated a packet id");
    assert!(a.is_disjoint(&b), "the restarted sender sent {} packet ids the first one had sent", a.intersection(&b).count());
    // "send everything" returns to the normal schedule: the systematic pass, from its start
    first.send_everything();
    let normal = data_ids(&mut first, 32);
    assert_eq!(normal, data_ids(&mut sender(&d, 2), 32));
}

#[test]
fn send_everything_resumes_the_normal_schedule() {
    // a twin that never focuses: after a pass and some repair rounds, one twin sends finish words for a while,
    // then everything again; its normal schedule picks up exactly where it left off
    let d = data(16 * 1024 * 1024 + 1);
    let (mut a, mut b) = (sender(&d, 2), sender(&d, 2));
    let ids = |s: &mut FlatSender| s.next_packets().unwrap().1.iter().map(|q| q.id).collect::<Vec<_>>();
    for _ in 0..a.frames_per_pass() + 100 { assert_eq!(ids(&mut a), ids(&mut b)); }
    // segment 1 (a full one: 2 x 1024 rounds of 16 packets) is still being sent after 50 frames; segment 2 is one
    // block, whose 2 rounds would be over at once
    a.finish(&encode(a.config().transfer_id, 2, &[0]).unwrap()).unwrap();
    for _ in 0..50 { a.next_packets().unwrap(); }
    assert!(a.focus().is_some() && !a.focus_done());
    a.send_everything();
    assert_eq!((a.focus(), a.focus_done()), (None, false));
    assert_eq!(data_ids(&mut a, 2000), data_ids(&mut b, 2000));
}

#[test]
fn senders_with_other_repair_offsets_send_other_ids() {
    // the browser has no clock in WebAssembly: the page passes the offset in
    let d = data(16 * 1024 * 1024 + 1);
    let focused = |offset: u32| {
        let cfg = FlatSenderConfig { format: CellFormat { pitch: 2, alphabet: Alphabet::Bw }, transfer_id: 9, repair_offset: Some(offset), ..FlatSenderConfig::default() };
        let mut tx = FlatSender::from_bytes("completion.bin", d.clone(), cfg).unwrap();
        tx.finish(&encode(9, 2, &[0]).unwrap()).unwrap();
        let ids: HashSet<PacketId> = data_ids(&mut tx, 200).into_iter().collect();
        assert_eq!(ids.len(), 200, "a sender repeated a packet id");
        assert!(ids.iter().all(|id| id.segment == 1 && id.pid >= 1024), "a packet that is not a repair packet of segment 1");
        ids
    };
    let (a, b) = (focused(0x1234_5678), focused(0x9ABC_DEF0));
    assert!(a.is_disjoint(&b), "{} packet ids in common", a.intersection(&b).count());
    assert_eq!(focused(0x1234_5678), a, "the same offset, the same ids");
}

/// 4 segments; the first pass loses every packet of segments 2, 3 and 4 (units 1-3), later frames are lost at 30%.
/// `code`: the units the words typed after the pass name (None: no words). Returns the frames shown at completion,
/// the frame the sender left focus by itself (if it did) and the received bytes.
fn short_three_arm(d: &[u8], code: Option<&[u32]>, max_passes: u64) -> (Option<u64>, Option<u64>, Vec<u8>) {
    let mut tx = sender(d, 2);
    let per_pass = tx.frames_per_pass();
    let mut rx = Receiver::in_memory();
    let mut r = 43u64;
    let mut left_focus = None;
    for n in 0..per_pass * max_passes {
        if n == per_pass {
            if let Some(units) = code {
                assert_eq!(rx.progress().segments_done, 1);
                assert_eq!(tx.finish(&encode(tx.config().transfer_id, 4, units).unwrap()).as_deref(), Ok(units));
            }
        }
        let (h, mut p) = tx.next_packets().unwrap();
        if code.is_some() && n >= per_pass && left_focus.is_none() && tx.focus_done() { left_focus = Some(n); }
        if n < per_pass {
            p.retain(|q| q.id.segment == 0 || q.id.segment == 1);
            if p.is_empty() { continue; }
        } else if xorshift(&mut r) % 1000 < 300 { continue; }
        if let FeedOutcome::Complete(_) = rx.feed_packets(&h, &p).unwrap() { return (Some(n + 1), left_focus, rx.take_file().expect("in-memory file").1); }
    }
    (None, left_focus, Vec::new())
}

#[test]
fn words_naming_only_some_short_parts_still_complete() {
    // a window code on a large file names only the short parts in its window; an old code, or a mistyped one the
    // check let through (1 in 64), names other parts than the short ones. The sender leaves the named parts by
    // itself once each has had its share of repair (`PacketSource::set_focus`) and sends everything again, so the
    // transfer completes without typing anything more. Here three parts are short and the words name only the first
    let d = data(64 * 1024 * 1024);
    let per_pass = frames_per_pass(&d);
    let (normal, _, normal_bytes) = short_three_arm(&d, None, 8);
    let normal = normal.expect("normal schedule never completed");
    let (got, left, bytes) = short_three_arm(&d, Some(&[1]), 8);
    eprintln!("words naming 1 of 3 short parts: complete at {got:?} frames, focus left at {left:?}; normal schedule {normal} frames; {per_pass} frames per pass");
    let got = got.expect("the transfer never completed after words naming only some short parts");
    assert!(normal_bytes == d && bytes == d, "received bytes differ from the input");
    // the named segment's rounds: 2 per block of its chunks, 2 x 16 MB of packets, half a pass of this file
    let left = left.expect("the sender never left focus by itself");
    assert!(left > per_pass && (left - per_pass) as f64 <= 0.5 * per_pass as f64 + 2.0, "focus left at {left}, the words typed at {per_pass}");
    // bound, from measurement: 8431 frames against the normal schedule's 6981 (+0.51 pass, the focus's half pass)
    let bound = normal as f64 + 0.6 * per_pass as f64;
    assert!(got as f64 <= bound, "{got} frames > bound {bound:.0} (normal {normal}, {per_pass} per pass)");
}

/// 40 files from 1 B to 3 MB, sizes spread geometrically, given largest first; every fifth one is text (its own
/// zstd frame in the stream), the rest noise.
fn forty_files() -> Vec<(String, Vec<u8>)> {
    (0..40u32).rev().map(|i| {
        let size = 3_000_000f64.powf(i as f64 / 39.0).round() as usize;
        if i % 5 == 4 { (format!("notes-{i:02}.txt"), b"one line of notes, and then another. ".iter().copied().cycle().take(size).collect()) }
        else { let mut s = 0x9E37_79B9u64 + i as u64; (format!("photo-{i:02}.jpg"), (0..size).map(|_| xorshift(&mut s) as u8).collect()) }
    }).collect()
}
fn as_sources(files: &[(String, Vec<u8>)]) -> Vec<(String, BatchSource)> { files.iter().map(|(n, d)| (n.clone(), BatchSource::Bytes(d.clone()))).collect() }
fn sha(b: &[u8]) -> [u8; 32] { Sha256::digest(b).into() }

#[test]
fn batch_completes_under_loss() {
    // 40 files of 1 B to 3 MB, an empty one and a second file under a name already used, every third read from a
    // path, as one batch; 20% of the frames lost: the stream completes within the random-loss bound, and every
    // file in it is bit-exact, as the receiver saves it: the index read from the stream (its hash checked),
    // each file at its offset, decompressed with the output capped at its size, checked against its own SHA-256
    let mut files = forty_files();
    files.push(("empty.txt".to_string(), Vec::new()));
    files.push(("photo-07.jpg".to_string(), data(777)));
    let dir = std::env::temp_dir().join(format!("lensbolt-batch-loss-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let sources = files.iter().enumerate().map(|(k, (n, d))| {
        if k % 3 != 0 { return (n.clone(), BatchSource::Bytes(d.clone())); }
        let path = dir.join(format!("{k}.src"));
        std::fs::write(&path, d).unwrap();
        (n.clone(), BatchSource::Path(path))
    }).collect();
    // what the index must list, in order: smallest first, equal sizes in the order given
    let mut sorted: Vec<&(String, Vec<u8>)> = files.iter().collect();
    sorted.sort_by_key(|f| f.1.len());
    let cfg = FlatSenderConfig { format: CellFormat { pitch: 2, alphabet: Alphabet::Bw }, ..FlatSenderConfig::default() };
    let mut tx = FlatSender::from_batch("forty", sources, cfg).unwrap();
    let per_pass = tx.frames_per_pass();
    let bound = 1.25 * per_pass as f64 / 0.8 + 60.0;
    let mut rx = Receiver::in_memory();
    let mut r = 0x5EED_u64;
    let mut done = None;
    for n in 0..(bound * 4.0) as u64 {
        let (h, p) = tx.next_packets().unwrap();
        if xorshift(&mut r) % 1000 < 200 { continue; }
        if let FeedOutcome::Complete(_) = rx.feed_packets(&h, &p).unwrap() { done = Some(n + 1); break; }
    }
    let m = tx.manifest().clone();
    eprintln!("batch_completes_under_loss: {} files, {} bytes of files, a stream of {} bytes; complete at {done:?} frames, bound {bound:.0} ({per_pass} frames per pass)",
        files.len(), files.iter().map(|f| f.1.len()).sum::<usize>(), m.stream_size);
    let frames = done.expect("the batch never completed");
    assert!(frames as f64 <= bound, "{frames} frames > bound {bound:.0}");
    assert!(rx.take_file().is_none(), "a batch is saved file by file, not as one stream");
    let index = tx.batch_index().expect("the index the sender sent");
    assert_eq!((index.entries.len(), index.stream_len()), (files.len(), m.stream_size));
    for w in index.entries.windows(2) { assert!(w[0].size <= w[1].size, "smallest first"); }
    let st = rx.batch_status().expect("a batch status");
    assert_eq!((st.name.as_str(), st.total as usize, st.received as usize, st.failed, st.error.clone()), ("forty", files.len(), files.len(), 0, None));
    let compressed = index.entries.iter().filter(|e| e.compressed).count();
    for (k, (e, (name, original))) in index.entries.iter().zip(sorted).enumerate() {
        assert_eq!(&e.name, name, "the file in this place");
        assert_eq!((st.files[k].size, &st.files[k].state), (e.size, &FileStage::Received), "{name}");
        let (saved_as, data) = rx.take_batch_file(k).unwrap_or_else(|| panic!("{name}: saved"));
        assert_eq!(saved_as, st.files[k].name);
        assert!(&data == original, "{}: bytes differ from the file sent", e.name);
    }
    eprintln!("batch_completes_under_loss: {compressed} of the 8 text files compressed");
    if cfg!(feature = "zstd") { assert!(compressed >= 4, "the larger text files travel compressed ({compressed} did)"); }
    let mut sevens: Vec<&str> = st.files.iter().map(|f| f.name.as_str()).filter(|n| n.starts_with("photo-07")).collect();
    sevens.sort();
    assert_eq!(sevens, ["photo-07 (2).jpg", "photo-07.jpg"], "both files named photo-07.jpg are kept");
    assert_eq!((st.files[0].name.as_str(), st.files[0].size), ("empty.txt", 0));
    drop(tx);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_batch_and_a_single_file_never_share_an_id() {
    let id = |s: FlatSender| s.config().transfer_id;
    let batch = |name: &str, files: &[(String, Vec<u8>)]| id(FlatSender::from_batch(name, as_sources(files), FlatSenderConfig::default()).unwrap());
    let one = vec![("report.pdf".to_string(), data(100_000))];
    let a = batch("report.pdf", &one);
    assert_eq!(a & 1, 1, "the low bit is set");
    assert_eq!(a, batch("report.pdf", &one), "the same batch again: the same id, so it resumes");
    assert_ne!(a, batch("reports", &one), "the same files under another batch name");
    let single = FlatSender::from_bytes("report.pdf", one[0].1.clone(), FlatSenderConfig::default()).unwrap();
    assert_ne!(a, id(single), "a batch of one file and the file alone");
    // the batch's own stream sent as one file under the batch's name: the same stream, another transfer
    let files = forty_files();
    let tx = FlatSender::from_batch("forty", as_sources(&files), FlatSenderConfig::default()).unwrap();
    let mut p = Prepared::batch("forty", as_sources(&files)).unwrap();
    let stream: Vec<u8> = (0..p.manifest.chunk_count(1)).flat_map(|c| p.read_chunk(1, c).unwrap()).collect();
    let twin = FlatSender::from_bytes("forty", stream, FlatSenderConfig::default()).unwrap();
    assert_eq!(twin.manifest().segment_sha256, tx.manifest().segment_sha256, "the same stream");
    assert_ne!(id(tx), id(twin), "a batch and a single file with the same bytes");
    // a file left out, or one more: another index, another id
    assert_ne!(batch("forty", &files[1..]), batch("forty", &files));
}

// ---- the receiver's side of a batch: per-file state, each file saved as it verifies ----

/// Stores held in memory and shared with the test: name -> bytes. `budget`: bytes the file stores may take in
/// all (the stream's `.lbpart` store does not count; a write past it fails as a full browser store does);
/// `refuse`: a name whose store fails every write with "no space left on device"; `written`: the most bytes
/// each name ever held (kept after a store is removed); `log`: every write to a file store (name, offset,
/// bytes); `part_reads`: reads of the stream's store so far, and `fail_read`: the one (counted from 0) that fails.
#[derive(Clone, Default)]
struct MemStores { files: Arc<Mutex<HashMap<String, Vec<u8>>>>, written: Arc<Mutex<HashMap<String, usize>>>, budget: Option<Arc<Mutex<u64>>>, refuse: Option<String>,
    log: Arc<Mutex<Vec<(String, u64, Vec<u8>)>>>, part_reads: Arc<Mutex<u64>>, fail_read: Option<u64> }
struct MemStore { name: String, s: MemStores }
impl PartStore for MemStore {
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        if self.name.ends_with(".lbpart") {
            let mut n = self.s.part_reads.lock().unwrap();
            *n += 1;
            if self.s.fail_read == Some(*n - 1) { return Err(std::io::Error::other("the store could not be read")); }
        }
        let files = self.s.files.lock().unwrap();
        let f = files.get(&self.name).ok_or_else(|| std::io::Error::other("no such store"))?;
        let (o, n) = (offset as usize, buf.len());
        if o + n > f.len() { return Err(std::io::Error::other("read past the end")); }
        buf.copy_from_slice(&f[o..o + n]);
        Ok(())
    }
    fn write_at(&mut self, offset: u64, data: &[u8]) -> std::io::Result<()> {
        if self.s.refuse.as_deref() == Some(self.name.as_str()) { return Err(std::io::Error::new(std::io::ErrorKind::StorageFull, "no space left on device")); }
        let end = offset as usize + data.len();
        let mut files = self.s.files.lock().unwrap();
        let f = files.get_mut(&self.name).ok_or_else(|| std::io::Error::other("no such store"))?;
        if let (Some(b), false) = (&self.s.budget, self.name.ends_with(".lbpart")) {
            let mut left = b.lock().unwrap();
            let grow = end.saturating_sub(f.len()) as u64;
            if grow > *left { return Err(std::io::Error::other("QuotaExceededError: the storage is full")); }
            *left -= grow;
        }
        if f.len() < end { f.resize(end, 0); }
        f[offset as usize..end].copy_from_slice(data);
        if !self.name.ends_with(".lbpart") { self.s.log.lock().unwrap().push((self.name.clone(), offset, data.to_vec())); }
        let mut w = self.s.written.lock().unwrap();
        let most = w.entry(self.name.clone()).or_insert(0);
        *most = (*most).max(f.len());
        Ok(())
    }
    fn set_len(&mut self, len: u64) -> std::io::Result<()> { self.s.files.lock().unwrap().get_mut(&self.name).ok_or_else(|| std::io::Error::other("no such store"))?.resize(len as usize, 0); Ok(()) }
}
impl StoreFactory for MemStores {
    fn create(&mut self, name: &str) -> std::io::Result<Box<dyn PartStore>> {
        self.files.lock().unwrap().insert(name.to_string(), Vec::new());
        Ok(Box::new(MemStore { name: name.to_string(), s: self.clone() }))
    }
    fn remove(&mut self, name: &str) { self.files.lock().unwrap().remove(name); }
}
impl MemStores {
    fn get(&self, name: &str) -> Option<Vec<u8>> { self.files.lock().unwrap().get(name).cloned() }
    /// Every write ever made under `name` held the file's own bytes at that place: no unverified byte carried it.
    fn only_true_bytes(&self, name: &str, data: &[u8]) -> bool {
        self.log.lock().unwrap().iter().filter(|w| w.0 == name).all(|(_, o, b)| data.get(*o as usize..*o as usize + b.len()) == Some(&b[..]))
    }
    fn names(&self) -> Vec<String> { let mut v: Vec<String> = self.files.lock().unwrap().keys().cloned().collect(); v.sort(); v }
}

fn noise(len: usize, seed: u64) -> Vec<u8> { let mut s = seed | 1; (0..len).map(|_| xorshift(&mut s) as u8).collect() }
fn batch_cfg() -> FlatSenderConfig { FlatSenderConfig { format: CellFormat { pitch: 2, alphabet: Alphabet::Bw }, ..FlatSenderConfig::default() } }
/// Feeds `frames` frames of `tx` to `rx`, data packets only when `keep` says so; returns the frame count at
/// completion, if it completed.
fn feed_batch(tx: &mut FlatSender, rx: &mut Receiver, frames: u64, mut keep: impl FnMut(&Packet) -> bool) -> Option<u64> {
    for n in 0..frames {
        let (h, mut p) = tx.next_packets().unwrap();
        p.retain(|p| p.id.segment == 0 || keep(p));
        if let FeedOutcome::Complete(_) = rx.feed_packets(&h, &p).unwrap() { return Some(n + 1); }
    }
    None
}
fn stage<'a>(st: &'a BatchStatus, name: &str) -> &'a FileStage { &st.files.iter().find(|f| f.name == name).unwrap_or_else(|| panic!("no file {name} in {:?}", st.files.iter().map(|f| &f.name).collect::<Vec<_>>())).state }

#[test]
fn batch_files_are_saved_as_they_verify() {
    // 30 small files and a 6 MB one: the small ones and the big one's start share the stream's first chunk.
    // Everything past the second chunk is withheld: the small files are saved (bytes as sent) before any
    // segment verified, while the big one shows its share received
    let mut files: Vec<(String, Vec<u8>)> = (0..30).map(|i| (format!("note-{i:02}.bin"), noise(1_000 + 997 * i, 100 + i as u64))).collect();
    files.push(("film.bin".into(), noise(6_000_000, 7)));
    let mut tx = FlatSender::from_batch("trip", as_sources(&files), batch_cfg()).unwrap();
    let stores = MemStores::default();
    let mut rx = Receiver::with_stores(Box::new(stores.clone()));
    let pass = tx.frames_per_pass();
    assert_eq!(feed_batch(&mut tx, &mut rx, pass + 40, |p| p.id.chunk < 2), None, "the withheld chunks keep it from completing");
    let st = rx.batch_status().expect("a batch status");
    assert_eq!((st.name.as_str(), st.total, st.received, st.failed, st.error.as_deref()), ("trip", 31, 30, 0, None));
    assert_eq!(rx.progress().segments_done, 0, "saved before their segment verified");
    for (name, data) in &files[..30] {
        assert_eq!(stage(&st, name), &FileStage::Received, "{name}");
        assert!(stores.get(name).as_ref() == Some(data), "{name}: saved bytes differ");
    }
    match stage(&st, "film.bin") {
        FileStage::Receiving(p) => assert!(*p > 0.2 && *p < 0.4, "film.bin at {p}"),
        other => panic!("film.bin: {other:?}"),
    }
    assert!(stores.get("film.bin").is_none(), "nothing of an unverified file is saved");
    // the rest arrives: the big one is saved too and the stream's part store is gone
    assert!(feed_batch(&mut tx, &mut rx, 3 * pass, |_| true).is_some(), "completes once everything is sent");
    let st = rx.batch_status().unwrap();
    assert_eq!((st.received, st.failed), (31, 0));
    assert!(stores.get("film.bin").as_ref() == Some(&files[30].1));
    assert_eq!(stores.names().len(), 31, "only the files are left: {:?}", stores.names());
}

#[test]
fn storage_full_fails_that_file_only() {
    // 20 incompressible files of 10 KB to 200 KB, smallest first; the store of the 6th refuses every write:
    // that file fails with the reason, the others before and after it are saved, the transfer completes
    let files: Vec<(String, Vec<u8>)> = (0..20).map(|i| (format!("photo-{i:02}.jpg"), noise(10_000 * (i + 1), 300 + i as u64))).collect();
    let run = |stores: MemStores| {
        let mut tx = FlatSender::from_batch("full", as_sources(&files), batch_cfg()).unwrap();
        let mut rx = Receiver::with_stores(Box::new(stores));
        let pass = tx.frames_per_pass();
        assert!(feed_batch(&mut tx, &mut rx, 3 * pass, |_| true).is_some(), "the transfer completes");
        rx.batch_status().unwrap()
    };
    let stores = MemStores { refuse: Some("photo-05.jpg".into()), ..MemStores::default() };
    let st = run(stores.clone());
    assert_eq!((st.received, st.failed), (19, 1));
    assert_eq!(stage(&st, "photo-05.jpg"), &FileStage::Failed("storage full".into()));
    assert!(stores.get("photo-05.jpg").is_none(), "the failed file's store is removed");
    for (name, data) in files.iter().filter(|f| f.0 != "photo-05.jpg") {
        assert_eq!(stage(&st, name), &FileStage::Received, "{name}");
        assert!(stores.get(name).as_ref() == Some(data), "{name}");
    }
    // storage that fills up: room for the first 7 files and half the 8th. The 8th and every later one fail
    // with the reason (each is larger than what is left); the 7 saved files stay as they are
    let room: usize = files[..7].iter().map(|f| f.1.len()).sum::<usize>() + files[7].1.len() / 2;
    let stores = MemStores { budget: Some(Arc::new(Mutex::new(room as u64))), ..MemStores::default() };
    let st = run(stores.clone());
    assert_eq!((st.received, st.failed), (7, 13));
    for (k, (name, data)) in files.iter().enumerate() {
        if k < 7 { assert_eq!(stage(&st, name), &FileStage::Received, "{name}"); assert!(stores.get(name).as_ref() == Some(data), "{name}"); }
        else { assert_eq!(stage(&st, name), &FileStage::Failed("storage full".into()), "{name}"); assert!(stores.get(name).is_none(), "{name}"); }
    }
}

#[test]
fn duplicate_names_are_kept_apart() {
    // names a case-insensitive or normalising file system (Windows, Android storage, macOS) takes for one file:
    // each kept, the later ones as "name (2).ext". Saved to disk, so this machine's file system checks it too
    let long = format!("{}.jpg", "x".repeat(251));
    let long2 = format!("{} (2).jpg", "x".repeat(247));
    let named: Vec<(&str, &str)> = vec![
        ("photo.jpg", "photo.jpg"), ("photo.jpg", "photo (2).jpg"), ("PHOTO.JPG", "PHOTO (3).JPG"),
        ("Caf\u{e9}.txt", "Caf\u{e9}.txt"), ("Cafe\u{301}.txt", "Cafe\u{301} (2).txt"),
        ("a:b.txt", "a_b.txt"), ("a_b.txt", "a_b (2).txt"),
        ("stra\u{df}e.txt", "stra\u{df}e.txt"), ("STRASSE.txt", "STRASSE (2).txt"),
        (&long, &long), (&long, &long2),
        ("photo (2).jpg", "photo (2) (2).jpg"),
        ("\u{ac00}.txt", "\u{ac00}.txt"), ("\u{1100}\u{1161}.txt", "\u{1100}\u{1161} (2).txt"),
        ("\u{304c}.txt", "\u{304c}.txt"), ("\u{304b}\u{3099}.txt", "\u{304b}\u{3099} (2).txt"),
        (".profile", ".profile"), ("profile", "profile (2)"),
    ];
    // equal sizes keep the order given
    let files: Vec<(String, Vec<u8>)> = named.iter().enumerate().map(|(k, (n, _))| (n.to_string(), noise(64, 500 + k as u64))).collect();
    let base = std::env::temp_dir().join(format!("lensbolt-batch-dups-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    let mut tx = FlatSender::from_batch("same names", as_sources(&files), batch_cfg()).unwrap();
    let mut rx = Receiver::new(&base).unwrap();
    let pass = tx.frames_per_pass();
    let mut done = None;
    for _ in 0..3 * pass { let (h, p) = tx.next_packets().unwrap(); if let FeedOutcome::Complete(path) = rx.feed_packets(&h, &p).unwrap() { done = Some(path); break; } }
    let dir = done.expect("completes");
    assert_eq!(dir, base.join("same names"), "a disk receiver saves a batch in a folder named after it");
    let st = rx.batch_status().unwrap();
    assert_eq!(st.files.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(), named.iter().map(|n| n.1).collect::<Vec<_>>());
    for (f, (_, data)) in st.files.iter().zip(&files) {
        assert_eq!(f.state, FileStage::Received, "{}", f.name);
        assert!(f.name.len() <= 255, "{} bytes", f.name.len());
        assert!(&std::fs::read(dir.join(&f.name)).unwrap() == data, "{}: bytes on disk", f.name);
    }
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), named.len(), "one file on disk per file sent");
    let left: Vec<_> = std::fs::read_dir(&base).unwrap().map(|e| e.unwrap().file_name()).collect();
    assert_eq!(left, vec![std::ffi::OsString::from("same names")], "the stream's part file is removed");
    std::fs::remove_dir_all(&base).unwrap();
    // a store-backed receiver keeps its stream in a store named after the batch: a file of that name gets another
    let files = vec![("same names.lbpart".to_string(), noise(10, 1)), ("x.bin".to_string(), noise(10, 2))];
    let mut tx = FlatSender::from_batch("same names", as_sources(&files), batch_cfg()).unwrap();
    let stores = MemStores::default();
    let mut rx = Receiver::with_stores(Box::new(stores.clone()));
    assert!(feed_batch(&mut tx, &mut rx, 3 * pass, |_| true).is_some());
    let st = rx.batch_status().unwrap();
    assert_eq!(st.files[0].name, "same names (2).lbpart");
    assert_eq!(stores.names(), vec!["same names (2).lbpart".to_string(), "x.bin".to_string()]);
}

#[test]
fn empty_file_in_a_batch() {
    // two empty files and a 3 MB one: the empty files hold no bytes of the stream, so they are received as
    // soon as the file list is (here while the 3 MB file is still short of its last chunks)
    let files = vec![("empty-a.txt".to_string(), Vec::new()), ("song.bin".to_string(), noise(3_000_000, 9)), ("empty-b.txt".to_string(), Vec::new())];
    let mut tx = FlatSender::from_batch("with empty", as_sources(&files), batch_cfg()).unwrap();
    let mut rx = Receiver::in_memory();
    let pass = tx.frames_per_pass();
    assert_eq!(feed_batch(&mut tx, &mut rx, pass + 40, |p| p.id.chunk == 0), None);
    let st = rx.batch_status().unwrap();
    assert_eq!((st.total, st.received), (3, 2));
    assert_eq!((stage(&st, "empty-a.txt"), stage(&st, "empty-b.txt")), (&FileStage::Received, &FileStage::Received));
    assert!(matches!(stage(&st, "song.bin"), FileStage::Receiving(_)), "{:?}", stage(&st, "song.bin"));
    assert!(feed_batch(&mut tx, &mut rx, 3 * pass, |_| true).is_some());
    let st = rx.batch_status().unwrap();
    assert_eq!((st.received, st.failed), (3, 0));
    // in memory: each file taken out once, by its place in the list (smallest first: the empty ones, then the song)
    assert_eq!(st.files.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(), ["empty-a.txt", "empty-b.txt", "song.bin"]);
    assert_eq!(rx.take_batch_file(0), Some(("empty-a.txt".to_string(), Vec::new())));
    assert_eq!(rx.take_batch_file(1), Some(("empty-b.txt".to_string(), Vec::new())));
    assert!(rx.take_batch_file(2) == Some(("song.bin".to_string(), files[1].1.clone())));
    assert_eq!(rx.take_batch_file(2), None, "taken once");
    assert!(rx.take_file().is_none(), "a batch is not one file");
    // a batch of one empty file: the stream is the file list alone
    let one = vec![("nothing.txt".to_string(), Vec::new())];
    let mut tx = FlatSender::from_batch("nothing", as_sources(&one), batch_cfg()).unwrap();
    let mut rx = Receiver::in_memory();
    assert!(feed_batch(&mut tx, &mut rx, pass, |_| true).is_some());
    let st = rx.batch_status().unwrap();
    assert_eq!((st.total, st.received, st.files[0].state.clone()), (1, 1, FileStage::Received));
    assert_eq!(rx.take_batch_file(0), Some(("nothing.txt".to_string(), Vec::new())));
}

#[test]
fn ten_thousand_files_batch() {
    // 10,000 small files (0 to 2 KB): every one received and saved, timed on the receiver's side
    let files: Vec<(String, Vec<u8>)> = (0..10_000u64).map(|i| (format!("file-{i:05}.bin"), noise(((i * 7919) % 2048) as usize, i + 1))).collect();
    let t0 = std::time::Instant::now();
    let mut tx = FlatSender::from_batch("ten thousand", as_sources(&files), batch_cfg()).unwrap();
    let prepare = t0.elapsed();
    let stores = MemStores::default();
    let mut rx = Receiver::with_stores(Box::new(stores.clone()));
    let (mut feeding, mut status, mut done) = (std::time::Duration::ZERO, std::time::Duration::ZERO, None);
    let pass = tx.frames_per_pass();
    for n in 0..pass + 40 {
        let (h, p) = tx.next_packets().unwrap();
        let t = std::time::Instant::now();
        let out = rx.feed_packets(&h, &p).unwrap();
        feeding += t.elapsed();
        if n % 30 == 0 { let t = std::time::Instant::now(); let _ = rx.batch_status(); status += t.elapsed(); }
        if let FeedOutcome::Complete(_) = out { done = Some(n + 1); break; }
    }
    let frames = done.expect("completes in one pass");
    let t = std::time::Instant::now();
    let st = rx.batch_status().expect("a batch status");
    let one_status = t.elapsed();
    eprintln!("ten_thousand_files_batch: {} bytes of files, a stream of {} bytes, {frames} frames; prepare {prepare:?}, receiver {feeding:?} in all, one status {one_status:?} ({status:?} for every 30th frame)",
        files.iter().map(|f| f.1.len()).sum::<usize>(), tx.manifest().stream_size);
    assert_eq!((st.total, st.received, st.failed), (10_000, 10_000, 0));
    assert!(st.files.iter().all(|f| f.state == FileStage::Received));
    for (name, data) in files.iter().step_by(97) { assert!(stores.get(name).as_ref() == Some(data), "{name}"); }
    assert_eq!(stores.names().len(), 10_000);
}

#[test]
fn the_file_list_comes_first() {
    // 10,000 files: a 400 KB file list, 391 blocks. They open the first pass (and are sent again in their places),
    // so the list is known within the first 5% of the pass (47% in the balanced order alone), the pass is longer
    // by those blocks only, and a receiver that joins at 30% of the pass still completes within about one pass
    let files: Vec<(String, Vec<u8>)> = (0..10_000u64).map(|i| (format!("file-{i:05}.bin"), noise(((i * 7919) % 2048) as usize, i + 1))).collect();
    let tx = FlatSender::from_batch("first", as_sources(&files), batch_cfg()).unwrap();
    let (pass, index_len) = (tx.frames_per_pass(), tx.manifest().batch.as_ref().unwrap().index_len);
    let stream_blocks: u64 = (1..=tx.manifest().segment_count() as u16).flat_map(|s| { let m = tx.manifest().clone(); (0..m.chunk_count(s)).map(move |c| lensbolt_core::file::chunk_blocks(m.chunk_len(s, c)) as u64) }).sum();
    let run = |start: u64| {
        let mut tx = FlatSender::from_batch("first", as_sources(&files), batch_cfg()).unwrap();
        let mut rx = Receiver::in_memory();
        let mut list = None;
        for n in 0..start + 3 * pass {
            let (h, p) = tx.next_packets().unwrap();
            if n < start { continue; }
            let out = rx.feed_packets(&h, &p).unwrap();
            if list.is_none() && rx.batch_status().is_some_and(|s| s.total > 0) { list = Some(n); }
            if let FeedOutcome::Complete(_) = out { return (n + 1 - start, list.unwrap()); }
        }
        panic!("never completed");
    };
    let (frames, list) = run(0);
    let late = run(pass * 3 / 10).0;
    eprintln!("the_file_list_comes_first: {pass} frames per pass, list known at frame {list} ({:.1}%), complete at {frames}; joining at 30%: {late} frames", 100.0 * list as f64 / pass as f64);
    assert!((list as f64) < 0.05 * pass as f64, "the list at frame {list} of {pass}");
    let packets = lensbolt_core::file::packets::PacketSource::new(Prepared::batch("first", as_sources(&files)).unwrap()).unwrap().data_packets_per_pass();
    assert_eq!(packets, stream_blocks + index_len.div_ceil(1024), "the pass: the stream's blocks and the list's once more");
    assert!(frames <= pass && (late as f64) <= 1.02 * pass as f64, "on time {frames}, joining at 30% {late}, {pass} per pass");
}

#[test]
#[ignore]
fn ten_thousand_files_to_disk_timing() {
    // the phone's receiver writes each file to disk: timed, 10,000 small files
    //     cargo test --release --test completion ten_thousand_files_to_disk -- --ignored --nocapture
    let files: Vec<(String, Vec<u8>)> = (0..10_000u64).map(|i| (format!("file-{i:05}.bin"), noise(((i * 7919) % 2048) as usize, i + 1))).collect();
    let base = std::env::temp_dir().join(format!("lensbolt-batch-10k-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    let mut tx = FlatSender::from_batch("ten thousand", as_sources(&files), batch_cfg()).unwrap();
    let mut rx = Receiver::new(&base).unwrap();
    let (mut feeding, mut dir) = (std::time::Duration::ZERO, None);
    for _ in 0..tx.frames_per_pass() + 40 {
        let (h, p) = tx.next_packets().unwrap();
        let t = std::time::Instant::now();
        let out = rx.feed_packets(&h, &p).unwrap();
        feeding += t.elapsed();
        if let FeedOutcome::Complete(d) = out { dir = Some(d); break; }
    }
    let dir = dir.expect("completes");
    eprintln!("ten_thousand_files_to_disk: receiver {feeding:?} in all");
    assert_eq!(rx.batch_status().unwrap().received, 10_000);
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 10_000);
    std::fs::remove_dir_all(&base).unwrap();
}

/// A packet of chunk (`seg`, `chunk`) with id `pid` and a payload no sender made: what a corrupted read that
/// passed its CRC would feed the fountain. Fed before the sender's own packet with that id, which is then a
/// duplicate.
fn forged(tx: &FlatSender, seg: u16, chunk: u8, pid: u16) -> (FrameInfo, Vec<Packet>) {
    let info = FrameInfo { frame_type: FrameType::Data, transfer_id: tx.config().transfer_id, segment: seg, block_id: 0 };
    (info, vec![Packet { id: PacketId { segment: seg, chunk, pid }, payload: [0xA5; 1024] }])
}

#[test]
fn a_file_decoded_from_bad_bytes_is_received_once_they_are_decoded_again() {
    // block 30 of the first chunk arrives forged, inside a 2 KB file: the chunk decodes with it, that file does
    // not match its hash and waits (never failed: its segment has not verified); the segment's hash then fails,
    // its chunks decode again from repair packets, and the file is saved. No write under its name ever held a
    // byte that is not the file's
    let mut files: Vec<(String, Vec<u8>)> = (0..60).map(|i| (format!("doc-{i:02}.bin"), noise(2_000 + 10 * i, 900 + i as u64))).collect();
    files.push(("big.bin".into(), noise(2_500_000, 77)));
    let mut tx = FlatSender::from_batch("forged", as_sources(&files), batch_cfg()).unwrap();
    let index = tx.batch_index().unwrap().clone();
    let victim = index.entries.iter().find(|e| e.offset <= 30 * 1024 && e.offset + e.stored_size > 30 * 1024).expect("a small file holds block 30").name.clone();
    assert!(files.iter().find(|f| f.0 == victim).unwrap().1.len() < 3_000, "{victim} is one of the small files");
    let stores = MemStores::default();
    let mut rx = Receiver::with_stores(Box::new(stores.clone()));
    let pass = tx.frames_per_pass();
    let (mut received_at, mut failed_at, mut done) = (None, None, false);
    for n in 0..4 * pass {
        let (h, p) = tx.next_packets().unwrap();
        if let FeedOutcome::Complete(_) = rx.feed_packets(&h, &p).unwrap() { done = true; }
        if n == 0 { let (fh, fp) = forged(&tx, 1, 0, 30); rx.feed_packets(&fh, &fp).unwrap(); }
        if let Some(st) = rx.batch_status() {
            if let Some(f) = st.files.iter().find(|f| f.name == victim) {
                assert!(!matches!(f.state, FileStage::Failed(_)), "frame {n}: {victim} failed: {:?}", f.state);
                if f.state == FileStage::Received { received_at.get_or_insert(n); }
            }
        }
        if rx.hash_failures() > 0 { failed_at.get_or_insert(n); }
        if done { break; }
    }
    assert!(done, "completes");
    let (received_at, failed_at) = (received_at.expect("saved"), failed_at.expect("the forged block made its segment fail"));
    assert!(received_at > failed_at, "{victim} waited for its bytes to decode again: saved at frame {received_at}, the segment failed at {failed_at}");
    let st = rx.batch_status().unwrap();
    assert_eq!((st.received as usize, st.failed), (files.len(), 0));
    let data = &files.iter().find(|f| f.0 == victim).unwrap().1;
    assert!(stores.get(&victim).as_ref() == Some(data));
    for (name, data) in &files { assert!(stores.only_true_bytes(name, data), "{name}: a write held bytes that are not the file's"); }
}

#[test]
fn a_file_list_decoded_from_bad_bytes_is_read_once_it_is_decoded_again() {
    // block 0 of the stream, the start of the file list, arrives forged: the list does not match its hash and
    // waits (no error: its segment has not verified); once the segment fails and decodes again, the list is read
    // and every file saved
    let files: Vec<(String, Vec<u8>)> = (0..30).map(|i| (format!("pic-{i:02}.jpg"), noise(30_000 + 100 * i, 50 + i as u64))).collect();
    let mut tx = FlatSender::from_batch("forged list", as_sources(&files), batch_cfg()).unwrap();
    let stores = MemStores::default();
    let mut rx = Receiver::with_stores(Box::new(stores.clone()));
    let pass = tx.frames_per_pass();
    let mut done = false;
    for n in 0..4 * pass {
        let (h, p) = tx.next_packets().unwrap();
        if let FeedOutcome::Complete(_) = rx.feed_packets(&h, &p).expect("no error: the list waits") { done = true; break; }
        if n == 0 { let (fh, fp) = forged(&tx, 1, 0, 0); rx.feed_packets(&fh, &fp).unwrap(); }
        if n == pass / 2 { assert_eq!(rx.batch_status().unwrap().total, 0, "the forged list is not read"); }
        assert_eq!(rx.batch_status().unwrap().error, None, "frame {n}");
    }
    assert!(done && rx.hash_failures() >= 1);
    let st = rx.batch_status().unwrap();
    assert_eq!((st.total as usize, st.received as usize, st.failed), (files.len(), files.len(), 0));
    for (name, data) in &files { assert!(stores.get(name).as_ref() == Some(data), "{name}"); }
}

#[test]
fn a_file_across_the_first_segment_boundary_is_saved_before_the_second_segment_verifies() {
    // 26 MB in 4 files: the third runs from 8.5 MB to 17 MB, over the 16 MiB boundary into the second segment's
    // first chunk. Withholding the second segment's other chunks: the first segment verifies, the second does
    // not, and the third file is saved all the same (its chunks decoded, its hash matched)
    let files = vec![("a.bin".to_string(), noise(1_000_000, 1)), ("b.bin".to_string(), noise(7_500_000, 2)), ("across.bin".to_string(), noise(8_500_000, 3)), ("d.bin".to_string(), noise(9_000_000, 4))];
    let mut tx = FlatSender::from_batch("across", as_sources(&files), batch_cfg()).unwrap();
    let e = tx.batch_index().unwrap().entries.iter().find(|e| e.name == "across.bin").unwrap().clone();
    let seg = lensbolt_core::file::SEGMENT_BYTES;
    assert!(e.offset < seg && e.offset + e.stored_size > seg && e.offset + e.stored_size < seg + (1 << 20), "{e:?}");
    let stores = MemStores::default();
    let mut rx = Receiver::with_stores(Box::new(stores.clone()));
    let pass = tx.frames_per_pass();
    assert_eq!(feed_batch(&mut tx, &mut rx, pass + 40, |p| p.id.segment == 1 || p.id.chunk == 0), None);
    let st = rx.batch_status().unwrap();
    assert_eq!(rx.progress().segments_done, 1, "only the first segment verified");
    assert_eq!((stage(&st, "a.bin"), stage(&st, "b.bin"), stage(&st, "across.bin")), (&FileStage::Received, &FileStage::Received, &FileStage::Received));
    assert!(matches!(stage(&st, "d.bin"), FileStage::Receiving(_)));
    assert!(stores.get("across.bin").as_ref() == Some(&files[2].1));
    let f = st.files.iter().find(|f| f.name == "across.bin").unwrap();
    assert_eq!((f.offset, f.stored_size), (e.offset, e.stored_size), "a file's place in the stream, for the finish words");
}

#[test]
fn a_failed_read_of_the_stream_is_tried_again() {
    // the stream's store fails one read: the first one, when the file list is read, then one in the frame that
    // completes the transfer. Either way the read is tried again (from the next chunk for the list, at once at
    // the end) and the transfer completes in the frame it would have, with every file
    let files: Vec<(String, Vec<u8>)> = (0..12).map(|i| (format!("f-{i:02}.bin"), noise(150_000 + 1_000 * i, 70 + i as u64))).collect();
    let run = |fail_read: Option<u64>| {
        let mut tx = FlatSender::from_batch("reads", as_sources(&files), batch_cfg()).unwrap();
        let stores = MemStores { fail_read, ..MemStores::default() };
        let mut rx = Receiver::with_stores(Box::new(stores.clone()));
        let (mut errors, mut reads_before, mut done, mut list_at) = (0, Vec::new(), None, None);
        for n in 0..3 * tx.frames_per_pass() {
            let (h, p) = tx.next_packets().unwrap();
            reads_before.push(*stores.part_reads.lock().unwrap());
            match rx.feed_packets(&h, &p) { Ok(FeedOutcome::Complete(_)) => { done = Some(n); break; } Ok(_) => {} Err(_) => errors += 1 }
            if list_at.is_none() && rx.batch_status().is_some_and(|s| s.total > 0) { list_at = Some(n); }
        }
        let st = rx.batch_status().unwrap();
        (done.expect("completes"), errors, reads_before, st.received, stores, list_at.unwrap_or(u64::MAX))
    };
    let (frame, errors, reads, received, _, list_at) = run(None);
    assert_eq!((errors, received), (0, 12));
    let last_feed_first_read = reads[frame as usize];
    for fail in [0, last_feed_first_read] {
        let (f, _, _, received, stores, list) = run(Some(fail));
        assert_eq!((f, received), (frame, 12), "read {fail} failed once");
        assert!(list <= list_at + 2, "read {fail} failed once: the file list known at frame {list}, at {list_at} without the failure");
        for (name, data) in &files { assert!(stores.get(name).as_ref() == Some(data), "{name}"); }
    }
}

#[test]
fn a_file_already_on_disk_is_never_replaced() {
    // something else writes "later.bin" into the batch's folder while the batch arrives: the batch's own
    // later.bin is saved as "later (2).bin" and the other file keeps its bytes
    let base = std::env::temp_dir().join(format!("lensbolt-batch-noreplace-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    let files = vec![("first.bin".to_string(), noise(5_000, 1)), ("later.bin".to_string(), noise(3_000_000, 2))];
    let mut tx = FlatSender::from_batch("kept", as_sources(&files), batch_cfg()).unwrap();
    let mut rx = Receiver::new(&base).unwrap();
    let pass = tx.frames_per_pass();
    assert_eq!(feed_batch(&mut tx, &mut rx, pass + 20, |p| p.id.chunk == 0), None);
    assert_eq!(stage(&rx.batch_status().unwrap(), "first.bin"), &FileStage::Received);
    std::fs::write(base.join("kept").join("later.bin"), b"not the batch's").unwrap();
    assert!(feed_batch(&mut tx, &mut rx, 3 * pass, |_| true).is_some());
    let st = rx.batch_status().unwrap();
    assert_eq!(st.files.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(), ["first.bin", "later (2).bin"]);
    assert_eq!(std::fs::read(base.join("kept").join("later.bin")).unwrap(), b"not the batch's");
    assert!(std::fs::read(base.join("kept").join("later (2).bin")).unwrap() == files[1].1);
    let mut left: Vec<_> = std::fs::read_dir(base.join("kept")).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    left.sort();
    assert_eq!(left, ["first.bin", "later (2).bin", "later.bin"], "no temporary file left");
    assert_eq!(std::fs::read_dir(&base).unwrap().count(), 1, "no part or temporary file beside the folder");
    std::fs::remove_dir_all(&base).unwrap();
}

#[test]
fn a_batch_name_of_255_bytes_fits_its_folder_and_its_part() {
    // the longest batch name (ASCII: NTFS counts UTF-16 units, ext4 bytes; here they agree): its part file
    // and a second folder of that name still fit in 255
    let name = format!("{}.zip", "n".repeat(251));
    assert_eq!(name.len(), 255);
    // (a verbatim path on Windows, past its 260-character limit on whole paths)
    let base = std::env::temp_dir().join(format!("lensbolt-batch-longname-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    let base = std::fs::canonicalize(&base).unwrap();
    std::fs::create_dir_all(base.join(&name)).unwrap();
    let files = vec![("x.bin".to_string(), noise(2_000_000, 3))];
    let mut tx = FlatSender::from_batch(&name, as_sources(&files), batch_cfg()).unwrap();
    let mut rx = Receiver::new(&base).unwrap();
    let mut seen = 0;
    let mut done = None;
    for _ in 0..3 * tx.frames_per_pass() {
        let (h, p) = tx.next_packets().unwrap();
        if let FeedOutcome::Complete(d) = rx.feed_packets(&h, &p).unwrap() { done = Some(d); break; }
        seen = seen.max(std::fs::read_dir(&base).unwrap().map(|e| e.unwrap().file_name().len()).max().unwrap_or(0));
    }
    let dir = done.expect("completes");
    let folder = dir.file_name().unwrap().to_str().unwrap().to_string();
    assert!(folder.len() <= 255 && folder.ends_with(" (2).zip") && folder != name, "{folder}");
    assert!(seen <= 255, "every name beside the folder within 255 bytes: {seen}");
    assert!(std::fs::read(dir.join("x.bin")).unwrap() == files[0].1);
    std::fs::remove_dir_all(&base).unwrap();
}

#[cfg(feature = "zstd")]
#[test]
fn a_zstd_frame_asking_for_a_window_beyond_its_size_is_refused() {
    // 300 bytes compressed as a stream with no size in its header, at level 19: the frame asks for an 8 MiB
    // window. The receiver lets a file's frame have a window of its size at most (2^10 to 2^27), so it is
    // refused (never 8 MiB of memory for 300 bytes); the same bytes compressed whole are saved
    use std::io::Write as _;
    let text = b"window ".repeat(43)[..300].to_vec();
    let mut enc = zstd::stream::Encoder::new(Vec::new(), 19).unwrap();
    enc.write_all(&text[..150]).unwrap(); enc.flush().unwrap(); enc.write_all(&text[150..]).unwrap();
    let wide = enc.finish().unwrap();
    assert_eq!(zstd::stream::decode_all(&wide[..]).unwrap(), text, "a valid frame");
    let whole = zstd::bulk::compress(&text, 19).unwrap();
    let mut src = crafted("windows", vec![("wide.txt", wide, 300, true, sha(&text)), ("whole.txt", whole, 300, true, sha(&text))]);
    let stores = MemStores::default();
    let mut rx = Receiver::with_stores(Box::new(stores.clone()));
    feed_crafted(&mut src, &mut rx);
    let st = rx.batch_status().unwrap();
    assert_eq!(stage(&st, "wide.txt"), &FileStage::Failed("cannot be unpacked".into()));
    assert_eq!(stage(&st, "whole.txt"), &FileStage::Received);
}

/// A batch built by hand (to send what a correct sender never would): `files` are (name, stored bytes, size,
/// compressed, SHA-256), laid out after the index as a sender lays them out. Returns its source and the
/// transfer id to send it under.
fn crafted(name: &str, files: Vec<(&str, Vec<u8>, u64, bool, [u8; 32])>) -> PacketSource { crafted_with(name, files, |_| {}) }
/// `crafted`, its batch section changed by `edit` before it is sent.
fn crafted_with(name: &str, files: Vec<(&str, Vec<u8>, u64, bool, [u8; 32])>, edit: impl FnOnce(&mut BatchInfo)) -> PacketSource {
    let mut index = Index { entries: files.iter().map(|(n, stored, size, z, h)| Entry { name: n.to_string(), size: *size, stored_size: stored.len() as u64, compressed: *z, offset: 0, sha256: *h }).collect() };
    let mut at = index.encoded_len() as u64;
    for e in &mut index.entries { e.offset = at; at += e.stored_size; }
    index.validate().expect("a valid index");
    let bytes = index.encode();
    let mut stream = bytes.clone();
    for f in &files { stream.extend_from_slice(&f.1); }
    let mut p = Prepared::from_bytes(name, stream).unwrap();
    let mut info = BatchInfo { name: name.to_string(), index_len: bytes.len() as u64, index_sha256: sha(&bytes) };
    edit(&mut info);
    p.manifest.batch = Some(info);
    PacketSource::new(p).unwrap()
}
/// Feeds `src` to `rx` as frames of 40 packets, every 10th a manifest frame, until it completes.
fn feed_crafted(src: &mut PacketSource, rx: &mut Receiver) {
    let id = 4321;
    for n in 0..20_000 {
        let manifest = n % 10 == 0;
        let packets: Vec<Packet> = (0..40).map(|_| if manifest { src.next_manifest_packet() } else { src.next_data_packet().unwrap() }).collect();
        let info = FrameInfo { frame_type: if manifest { FrameType::Manifest } else { FrameType::Data }, transfer_id: id, segment: packets[0].id.segment, block_id: if manifest { src.manifest_bytes_len() as u32 } else { 0 } };
        if let FeedOutcome::Complete(_) = rx.feed_packets(&info, &packets).unwrap() { return; }
    }
    panic!("the crafted batch never completed");
}

#[test]
fn a_file_whose_checksum_fails_is_failed_and_the_others_continue() {
    // the second file's listed SHA-256 is not its bytes': once its segment verified it is failed, with the
    // reason, and nothing of it is saved; the files around it are saved
    let (a, b, c) = (noise(5_000, 1), noise(6_000, 2), noise(7_000, 3));
    let mut src = crafted("checked", vec![("a.bin", a.clone(), 5_000, false, sha(&a)), ("b.bin", b.clone(), 6_000, false, sha(b"not b")), ("c.bin", c.clone(), 7_000, false, sha(&c))]);
    let stores = MemStores::default();
    let mut rx = Receiver::with_stores(Box::new(stores.clone()));
    feed_crafted(&mut src, &mut rx);
    let st = rx.batch_status().unwrap();
    assert_eq!((st.received, st.failed), (2, 1));
    assert_eq!(st.files[1].state, FileStage::Failed("checksum mismatch".into()));
    assert!(stores.get("b.bin").is_none());
    assert!(stores.get("a.bin") == Some(a) && stores.get("c.bin") == Some(c));
}

#[test]
fn a_file_list_that_does_not_match_its_checksum_is_refused() {
    // the manifest's hash of the file list is not the list's: once the segment holding it verified, the receiver
    // says so once (an error from that feed) and in its status, saves nothing, never reports the batch complete,
    // and drops the stream it holds
    let a = noise(5_000, 1);
    let mut src = crafted_with("refused", vec![("a.bin", a.clone(), 5_000, false, sha(&a))], |b| b.index_sha256 = [0; 32]);
    let stores = MemStores::default();
    let mut rx = Receiver::with_stores(Box::new(stores.clone()));
    let (mut errors, mut complete) = (Vec::new(), false);
    for n in 0..2_000 {
        let manifest = n % 10 == 0;
        let packets: Vec<Packet> = (0..40).map(|_| if manifest { src.next_manifest_packet() } else { src.next_data_packet().unwrap() }).collect();
        let info = FrameInfo { frame_type: if manifest { FrameType::Manifest } else { FrameType::Data }, transfer_id: 4321, segment: packets[0].id.segment, block_id: if manifest { src.manifest_bytes_len() as u32 } else { 0 } };
        match rx.feed_packets(&info, &packets) { Ok(FeedOutcome::Complete(_)) => { complete = true; break; } Ok(_) => {} Err(e) => errors.push(e.to_string()) }
    }
    assert!(!complete && !rx.is_complete(), "a batch whose file list was refused is never complete");
    assert_eq!(errors, ["the file list does not match its checksum"]);
    assert_eq!(rx.progress().segments_done, rx.progress().segments_total, "every segment verified");
    let st = rx.batch_status().unwrap();
    assert_eq!((st.total, st.received, st.error.as_deref()), (0, 0, Some("the file list does not match its checksum")));
    assert!(stores.names().is_empty(), "nothing saved, the part store removed: {:?}", stores.names());
}

#[test]
fn a_compressed_file_unpacks_no_further_than_its_size() {
    // "lensbolt in a browser " x 200 (4,400 bytes) as a 40-byte zstd frame (Python's compression.zstd, level
    // 19; its header states 4,400 bytes, its window as much). Listed with its true size it is saved (with ruzstd
    // too, in a build without zstd). Listed as 4,300 bytes it unpacks past its size: the output stops one byte
    // past 4,300 and the file fails. Listed as 100 bytes its window is past what 100 bytes need (2^10): the zstd
    // crate refuses it before unpacking anything (ruzstd, without a window limit, stops at the size as above).
    // A failed file never has a byte written for it
    let frame: Vec<u8> = vec![40,181,47,253,96,48,16,245,0,0,176,108,101,110,115,98,111,108,116,32,105,110,32,97,32,98,114,111,119,115,101,114,32,1,0,46,34,251,217,4];
    let text = b"lensbolt in a browser ".repeat(200);
    let mut src = crafted("packed", vec![("tiny.txt", frame.clone(), 100, true, sha(&text[..100])), ("short.txt", frame.clone(), 4_300, true, sha(&text[..4_300])), ("whole.txt", frame, 4_400, true, sha(&text))]);
    let stores = MemStores::default();
    let mut rx = Receiver::with_stores(Box::new(stores.clone()));
    feed_crafted(&mut src, &mut rx);
    let st = rx.batch_status().unwrap();
    assert_eq!(stage(&st, "whole.txt"), &FileStage::Received);
    assert_eq!(stores.get("whole.txt"), Some(text));
    assert_eq!(stage(&st, "short.txt"), &FileStage::Failed("does not unpack to its size".into()));
    let tiny = if cfg!(feature = "zstd") { "cannot be unpacked" } else { "does not unpack to its size" };
    assert_eq!(stage(&st, "tiny.txt"), &FileStage::Failed(tiny.into()));
    for name in ["tiny.txt", "short.txt"] {
        assert!(stores.get(name).is_none() && !stores.written.lock().unwrap().contains_key(name), "{name}: nothing written for a file that failed");
    }
}

#[test]
fn a_disk_receiver_saves_a_batch_in_its_folder() {
    // the phone's receiver: the files land in <out>/<batch name>/ as they verify (a compressed text among them),
    // the stream's part file is removed at the end; a folder of that name already there gets " (2)"
    let base = std::env::temp_dir().join(format!("lensbolt-batch-disk-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(base.join("holiday")).unwrap();
    let files = vec![("a.jpg".to_string(), noise(300_000, 4)), ("notes.txt".to_string(), b"a line of notes ".repeat(50_000)), ("b.jpg".to_string(), noise(2_500_000, 5))];
    let mut tx = FlatSender::from_batch("holiday", as_sources(&files), batch_cfg()).unwrap();
    let mut rx = Receiver::new(&base).unwrap();
    let pass = tx.frames_per_pass();
    let mut done = None;
    for _ in 0..3 * pass { let (h, p) = tx.next_packets().unwrap(); if let FeedOutcome::Complete(path) = rx.feed_packets(&h, &p).unwrap() { done = Some(path); break; } }
    let dir = done.expect("completes");
    assert_eq!(dir, base.join("holiday (2)"));
    for (name, data) in &files { assert!(&std::fs::read(dir.join(name)).unwrap() == data, "{name}"); }
    let mut left: Vec<_> = std::fs::read_dir(&base).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    left.sort();
    assert_eq!(left, ["holiday", "holiday (2)"], "no part or temporary file left");
    // the peak it reported: the stream and every file beside it
    assert_eq!(rx.progress().disk_peak, tx.manifest().stream_size + files.iter().map(|f| f.1.len() as u64).sum::<u64>());
    std::fs::remove_dir_all(&base).unwrap();
}

/// The files of `st` not yet received or failed: what the finish words must bring.
fn missing(st: &BatchStatus) -> Vec<usize> {
    st.files.iter().enumerate().filter(|(_, f)| !matches!(f.state, FileStage::Received | FileStage::Failed(_))).map(|(i, _)| i).collect()
}
/// The receiver's line under the finish words of a batch: the missing files they bring, two named, then "+n".
fn files_line(st: &BatchStatus, files: &[usize]) -> String {
    let names: Vec<&str> = files.iter().take(2).map(|&i| st.files[i].name.as_str()).collect();
    let more = if files.len() > 2 { format!(" +{}", files.len() - 2) } else { String::new() };
    format!("{} file{}: {}{more}", files.len(), if files.len() == 1 { "" } else { "s" }, names.join(", "))
}

#[test]
fn finish_words_for_a_batch_name_exactly_the_missing_files() {
    // 30 files over ten 1 MiB chunks of one segment: the units are chunks. Chunk 3 never arrives, so the files with
    // bytes in it (two of them across its edges) stay missing while every other file is saved. On every frame with
    // words: the words decode to the short unit, the files in it are exactly the files not received, the counts say
    // how many, and the line under the words names them
    let files: Vec<(String, Vec<u8>)> = (0..30).map(|i| (format!("photo-{i:02}.jpg"), noise(150_000 + 13_000 * i, 500 + i as u64))).collect();
    let mut tx = FlatSender::from_batch("trip", as_sources(&files), batch_cfg()).unwrap();
    let (m, pass) = (tx.manifest().clone(), tx.frames_per_pass());
    assert_eq!(m.segment_count(), 1);
    let mut rx = Receiver::in_memory();
    let mut with_words = 0;
    for _ in 0..pass + 60 {
        let (h, mut p) = tx.next_packets().unwrap();
        p.retain(|p| p.id.segment == 0 || p.id.chunk != 3);
        rx.feed_packets(&h, &p).unwrap();
        let Some(words) = rx.finish_words() else { assert_eq!(rx.batch_counts().and_then(|c| c.missing_files), None); continue };
        let units = lensbolt_core::finish::decode(h.transfer_id, m.finish_units(), &words).unwrap();
        assert_eq!(units, vec![3], "\"{words}\"");
        let st = rx.batch_status().unwrap();
        let named = st.files_in_units(&units);
        assert_eq!(named, missing(&st), "the files in the named unit are the missing ones");
        assert!(named.len() >= 3 && named.len() < 30, "{named:?}");
        let c = rx.batch_counts().unwrap();
        assert_eq!((c.name.as_str(), c.total, c.received, c.failed, c.missing_files, c.error), ("trip", 30, 30 - named.len() as u32, 0, Some(named.len() as u32), None));
        assert_eq!(rx.finish_parts(), Some(files_line(&st, &named)));
        with_words += 1;
    }
    assert!(with_words >= 30, "words on {with_words} frames");
}

#[test]
fn finish_words_over_segments_name_the_files_still_missing_in_them() {
    // 36 MB in 5 files, three segments: the units are segments. Segment 1 and the first chunk of the others arrive:
    // across.bin (over the first boundary) is saved, d.bin (inside segment 2) and e.bin (over the second boundary)
    // are not. The words name segments 2 and 3, which hold across.bin, d.bin and e.bin; the missing files are d and e
    // (from each file's state, not its segments). Then segment 2's chunks up to d.bin's end arrive: d.bin is saved
    // while segment 2 is still short, the words stay the same, and the missing files follow (e.bin alone)
    let files = vec![("a.bin".to_string(), noise(1_000_000, 1)), ("b.bin".to_string(), noise(7_500_000, 2)), ("across.bin".to_string(), noise(8_500_000, 3)),
                     ("d.bin".to_string(), noise(9_000_000, 4)), ("e.bin".to_string(), noise(9_500_000, 5))];
    let mut tx = FlatSender::from_batch("segments", as_sources(&files), batch_cfg()).unwrap();
    let (m, pass) = (tx.manifest().clone(), tx.frames_per_pass());
    assert_eq!(m.finish_units(), 3);
    let d = tx.batch_index().unwrap().entries.iter().find(|e| e.name == "d.bin").unwrap().clone();
    let seg = lensbolt_core::file::SEGMENT_BYTES;
    let d_last_chunk = ((d.offset + d.stored_size - 1 - seg) >> 20) as u8;
    assert!(d.offset > seg && d.offset + d.stored_size < 2 * seg && d_last_chunk < 15, "{d:?}");
    let (mut rx, id) = (Receiver::in_memory(), tx.config().transfer_id);
    let check = |rx: &Receiver, want_missing: &[&str]| {
        let words = rx.finish_words().expect("words once repair runs");
        let units = lensbolt_core::finish::decode(id, 3, &words).unwrap();
        assert_eq!(units, vec![1, 2]);
        let st = rx.batch_status().unwrap();
        let named: Vec<&str> = st.files_in_units(&units).into_iter().map(|i| st.files[i].name.as_str()).collect();
        assert_eq!(named, ["across.bin", "d.bin", "e.bin"], "the files with bytes in segments 2 and 3");
        let miss = missing(&st);
        assert_eq!(miss.iter().map(|&i| st.files[i].name.as_str()).collect::<Vec<_>>(), want_missing);
        assert!(st.files_in_units(&units).into_iter().filter(|i| miss.contains(i)).eq(miss.iter().copied()), "the missing files are the named ones not yet in");
        assert_eq!(rx.batch_counts().unwrap().missing_files, Some(want_missing.len() as u32));
        assert_eq!(rx.finish_parts(), Some(files_line(&st, &miss)));
        words
    };
    assert_eq!(feed_batch(&mut tx, &mut rx, pass + 40, |p| p.id.segment == 1 || p.id.chunk == 0), None);
    assert_eq!(rx.progress().segments_done, 1);
    let words = check(&rx, &["d.bin", "e.bin"]);
    let mut frames = 0;
    while rx.batch_status().unwrap().files.iter().find(|f| f.name == "d.bin").unwrap().state != FileStage::Received {
        frames += 1;
        assert!(frames < 3 * pass, "d.bin never arrived");
        feed_batch(&mut tx, &mut rx, 1, |p| p.id.segment == 1 || p.id.chunk == 0 || (p.id.segment == 2 && p.id.chunk <= d_last_chunk));
    }
    assert_eq!(rx.progress().segments_done, 1, "segment 2 still short");
    assert_eq!(check(&rx, &["e.bin"]), words, "the same words");
}

#[test]
fn files_map_to_units_by_overlap() {
    // units of 100 bytes; the file list is bytes 0..10. f0 shares unit 0 with the list; f1 and f4 are empty (in no
    // unit, even where they sit inside one); f2 runs over units 0, 1 and 2; f3 ends at the stream's end
    let f = |name: &str, offset: u64, size: u64| lensbolt_core::file::FileState { name: name.into(), size, offset, stored_size: size, state: FileStage::Waiting, sha256: None };
    let st = BatchStatus { name: "b".into(), total: 5, received: 0, failed: 0, error: None, unit_bytes: 100,
        files: vec![f("f0", 10, 50), f("f1", 60, 0), f("f2", 60, 190), f("f3", 250, 50), f("f4", 300, 0)] };
    assert_eq!(st.files_in_units(&[0]), vec![0, 2]);
    assert_eq!(st.files_in_units(&[1]), vec![2]);
    assert_eq!(st.files_in_units(&[2]), vec![2, 3]);
    assert_eq!(st.files_in_units(&[2, 0, 2, 1]), vec![0, 2, 3], "any order, repeats, each file once");
    assert_eq!(st.files_in_units(&[3]), Vec::<usize>::new(), "past the stream (and f4, empty at its end)");
    assert_eq!(st.files_in_units(&[]), Vec::<usize>::new());
    assert_eq!(st.files_in_units(&[u32::MAX]), Vec::<usize>::new(), "no overflow");
    // a compressed file: its stored bytes are what sits in the stream
    let st = BatchStatus { files: vec![lensbolt_core::file::FileState { stored_size: 40, ..f("z", 90, 1_000) }], ..st };
    assert_eq!((st.files_in_units(&[0]), st.files_in_units(&[1]), st.files_in_units(&[9])), (vec![0], vec![0], vec![]));
}

#[test]
fn batch_counts_do_not_walk_the_file_list() {
    // 10,000 files in one segment, its chunk 5 withheld: about 1,000 files missing. The counts are read on every
    // frame (as the status is): the missing files are counted once for the words, again only when the words or the
    // settled files change, never per read; a thousand reads of the counts and the line cost microseconds each
    let files: Vec<(String, Vec<u8>)> = (0..10_000u64).map(|i| (format!("file-{i:05}.bin"), noise(((i * 7919) % 2048) as usize, i + 1))).collect();
    let mut tx = FlatSender::from_batch("ten thousand", as_sources(&files), batch_cfg()).unwrap();
    let pass = tx.frames_per_pass();
    let mut rx = Receiver::in_memory();
    let mut scans_per_frame = Vec::new();
    for _ in 0..pass + 40 {
        let before = rx.batch_file_scans();
        feed_batch(&mut tx, &mut rx, 1, |p| p.id.chunk != 5);
        let _ = (rx.batch_counts(), rx.finish_parts());
        scans_per_frame.push(rx.batch_file_scans() - before);
    }
    let c = rx.batch_counts().unwrap();
    let st = rx.batch_status().unwrap();
    let missing = missing(&st).len() as u32;
    assert!(rx.finish_words().is_some() && missing > 500 && c.missing_files == Some(missing), "{c:?}, {missing} missing");
    let counted = scans_per_frame.iter().filter(|&&n| n > 0).count();
    assert!(counted <= 2, "the missing files were counted on {counted} frames: {:?}", scans_per_frame.iter().filter(|&&n| n > 0).collect::<Vec<_>>());
    assert!(scans_per_frame.iter().all(|&n| n <= missing as u64 + 2), "a count looks only at the named units' files");
    let (scans, t) = (rx.batch_file_scans(), std::time::Instant::now());
    for _ in 0..1_000 { assert_eq!(rx.batch_counts().unwrap().missing_files, Some(missing)); let _ = rx.finish_parts(); }
    let took = t.elapsed();
    eprintln!("batch_counts_do_not_walk_the_file_list: {missing} missing of 10,000; 1,000 reads of the counts and the line in {took:?}");
    assert_eq!(rx.batch_file_scans(), scans, "reads never count again");
    assert!(took < std::time::Duration::from_millis(100), "1,000 reads took {took:?}");
    // the rest arrives: words gone, nothing missing
    assert!(feed_batch(&mut tx, &mut rx, 3 * pass, |_| true).is_some());
    assert_eq!(rx.batch_counts().unwrap().missing_files, None);
}

// ---- the file list for a receiver that joins late ----

/// Four files, 9.4 MB in all (the phone's batch of 2026-10-30): 5.0, 2.6, 1.4 and 0.4 MB of noise.
fn four_files() -> Vec<(String, Vec<u8>)> {
    [("video.mp4", 5_000_000usize), ("scan.pdf", 2_600_000), ("photo.jpg", 1_400_000), ("notes.docx", 400_000)].iter().enumerate()
        .map(|(i, &(n, len))| (n.to_string(), noise(len, 0xF00D + i as u64))).collect()
}
fn ten_thousand_files() -> Vec<(String, Vec<u8>)> { (0..10_000u64).map(|i| (format!("file-{i:05}.bin"), noise(((i * 7919) % 2048) as usize, i + 1))).collect() }
/// A receiver that sees the frames of `tx` from frame `start` on, except those `lost` says: the frames from
/// `start` until it knows the file list (None for a single file), and until it completes.
fn join(mut tx: FlatSender, start: u64, max: u64, mut lost: impl FnMut(u64) -> bool) -> (Option<u64>, Option<u64>) {
    let mut rx = Receiver::in_memory();
    let mut list = None;
    for n in 0..start + max {
        let (h, p) = tx.next_packets().unwrap();
        if n < start || lost(n) { continue; }
        let out = rx.feed_packets(&h, &p).unwrap();
        if list.is_none() && rx.batch_counts().is_some_and(|c| c.total > 0) { list = Some(n + 1 - start); }
        if let FeedOutcome::Complete(_) = out { return (list, Some(n + 1 - start)); }
    }
    (list, None)
}

/// Frames after joining within which a receiver knows a small file list (two of them, one frame being the manifest's:
/// the list goes out again about once per manifest period, 31 frames, ~1 s at 30 fps; and the manifest itself
/// comes once per period). Measured: docs/2026-10-30-repair-schedule.md "The file list, repeated".
const LIST_WITHIN_FRAMES: u64 = 45;

#[test]
fn a_receiver_that_joins_late_lists_the_files_at_once() {
    // the phone started while the sender was already sending (2026-10-30): before, it waited for the list until its
    // chunk decoded, about when everything else did (397 frames after joining at 10-60% of the pass, 4 files).
    // The list's blocks now go out again regularly, the whole transfer long: joining half way through the first
    // pass, or in the repair rounds, the receiver lists the files within LIST_WITHIN_FRAMES, and still completes in
    // about one pass after joining; one that sees everything still completes within the pass
    for (name, files) in [("four", four_files()), ("forty", forty_files())] {
        let make = || FlatSender::from_batch(name, as_sources(&files), batch_cfg()).unwrap();
        let pass = make().frames_per_pass();
        for (when, start) in [("half way through the first pass", pass / 2), ("in the repair rounds", pass * 13 / 10)] {
            let (list, done) = join(make(), start, 4 * pass, |_| false);
            eprintln!("{name}: joining {when} (frame {start} of a {pass}-frame pass): list after {list:?} frames, complete after {done:?}");
            let list = list.unwrap_or_else(|| panic!("{name}, {when}: the list never came"));
            assert!(list <= LIST_WITHIN_FRAMES, "{name}, joining {when}: the list after {list} frames (bound {LIST_WITHIN_FRAMES})");
            assert!(done.is_some_and(|d| d as f64 <= 1.05 * pass as f64 + 10.0), "{name}, joining {when}: complete after {done:?} ({pass} per pass)");
        }
        let (_, done) = join(make(), 0, 2 * pass, |_| false);
        assert!(done.is_some_and(|d| d <= pass), "{name}: on time, complete after {done:?} of a {pass}-frame pass");
    }
}

#[test]
#[ignore]
fn file_list_for_late_joiners_measured() {
    // the frames until a receiver that joins late knows the file list, and until it completes, for the batches the
    // phone and the browser send (2 px b/w, 1024 x 1024); docs/2026-10-30-repair-schedule.md "The file list, repeated"
    //     cargo test --release --test completion file_list_for_late_joiners -- --ignored --nocapture
    let batches: Vec<(&str, Box<dyn Fn() -> FlatSender>)> = vec![
        ("4 files, 9.4 MB", Box::new(|| FlatSender::from_batch("four", as_sources(&four_files()), batch_cfg()).unwrap())),
        ("40 files", Box::new(|| FlatSender::from_batch("forty", as_sources(&forty_files()), batch_cfg()).unwrap())),
        ("10,000 files", Box::new(|| FlatSender::from_batch("ten thousand", as_sources(&ten_thousand_files()), batch_cfg()).unwrap())),
        ("one file, 9.4 MB", Box::new(|| FlatSender::from_bytes("one.bin", noise(9_400_000, 0xF00D), batch_cfg()).unwrap())),
    ];
    for (name, make) in &batches {
        let tx = make();
        let pass = tx.frames_per_pass();
        let blocks = tx.manifest().batch.as_ref().map_or(0, |b| b.index_len.div_ceil(1024));
        eprintln!("{name}: {pass} frames per pass, the list {blocks} blocks, {} packets per frame", tx.grid().packets());
        for (label, start) in [("from the start", 0), ("joins at 10%", pass / 10), ("joins at 30%", pass * 3 / 10), ("joins at 60%", pass * 6 / 10), ("joins in repair (1.3 passes)", pass * 13 / 10)] {
            let (list, done) = join(make(), start, 6 * pass, |_| false);
            eprintln!("  {label}: list after {list:?} frames, complete after {done:?}");
        }
        let mut total = 0;
        for seed in 1..=3u64 {
            let mut r = seed * 0x9E37_79B9;
            let (list, done) = join(make(), 0, 6 * pass, |_| xorshift(&mut r) % 1000 < 200);
            eprintln!("  20% of frames lost, seed {seed}: list after {list:?} frames, complete after {done:?}");
            total += done.unwrap();
        }
        eprintln!("  20% lost, mean of 3: {:.1}", total as f64 / 3.0);
    }
}
