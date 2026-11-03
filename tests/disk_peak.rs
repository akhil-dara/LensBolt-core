//! Peak disk use of the on-disk receiver (the phone's `cacheDir/rx`), measured for a 64 MB transfer: a poller
//! thread sums the lengths of every file in the output directory while the transfer runs, the finish included.
//! The receiver reports the peak it will need (`Progress::disk_peak`, `Manifest::receive_disk_peak`) as soon as
//! the manifest is in, so the app can check free space before the transfer starts rather than fail halfway.
//!   cargo test --release --test disk_peak -- --nocapture
use lensbolt_core::file::{FeedOutcome, Receiver};
use lensbolt_core::flat::{FlatSender, FlatSenderConfig};
use lensbolt_core::testutil::pseudo_random_file;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

const SIZE: u64 = 64 << 20;

fn dir_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir).map(|rd| rd.filter_map(|e| e.ok()?.metadata().ok()).filter(|m| m.is_file()).map(|m| m.len()).sum()).unwrap_or(0)
}

/// Runs one transfer of `input` into a fresh `out` dir; returns (measured peak bytes, reported disk_peak,
/// stream_size, file_size, output path).
fn run(base: &Path, input: &Path, force_compress: Option<bool>, id: u16) -> (u64, u64, u64, u64, PathBuf) {
    let out = base.join(format!("out{id}"));
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).unwrap();
    let cfg = FlatSenderConfig { manifest_every_frames: 60, transfer_id: id, ..FlatSenderConfig::default() };
    let mut s = FlatSender::new(input, cfg, force_compress).unwrap();
    let (stream, file) = (s.manifest().stream_size, s.manifest().file_size);
    assert_eq!(s.manifest().receive_disk_peak(), stream + if s.manifest().compressed { file } else { 0 });

    let peak = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let poller = {
        let (peak, stop, out) = (peak.clone(), stop.clone(), out.clone());
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) { peak.fetch_max(dir_bytes(&out), Ordering::Relaxed); std::thread::yield_now(); }
            peak.fetch_max(dir_bytes(&out), Ordering::Relaxed);
        })
    };
    let mut r = Receiver::new(&out).unwrap();
    let mut reported = 0u64; let mut frames = 0u64; let mut done = None;
    while done.is_none() {
        let (h, ps) = s.next_packets().unwrap(); frames += 1;
        if let FeedOutcome::Complete(p) = r.feed_packets(&h, &ps).unwrap() { done = Some(p); }
        let p = r.progress();
        // known from the manifest on, and constant
        if p.have_manifest { if reported == 0 { reported = p.disk_peak; } assert_eq!(p.disk_peak, reported); } else { assert_eq!(p.disk_peak, 0); }
        assert!(frames < s.frames_per_pass() * 3, "did not complete within 3 passes");
    }
    stop.store(true, Ordering::Relaxed);
    poller.join().unwrap();
    let path = done.unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().len(), SIZE);
    assert_eq!(dir_bytes(&out), SIZE, "only the finished file is left");
    (peak.load(Ordering::Relaxed), reported, stream, file, path)
}

#[test]
fn raw_transfer_peaks_at_the_stream_size() {
    let base = std::env::temp_dir().join("lensbolt_disk_peak_raw");
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    let input = base.join("random.bin");
    pseudo_random_file(&input, SIZE, 7).unwrap();
    let (peak, reported, stream, file, _) = run(&base, &input, Some(false), 41);
    eprintln!("raw 64 MB: stream {stream}, file {file}, measured peak {peak} ({:.4} x stream), reported {reported}", peak as f64 / stream as f64);
    assert_eq!((stream, file), (SIZE, SIZE));
    // the part file is sized to the stream at the manifest and renamed into place: never more than the stream
    assert_eq!(reported, stream);
    assert!(peak <= reported, "measured {peak} > reported {reported}");
    assert!(peak >= stream, "the poller saw the part file at its full length");
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn compressed_transfer_peaks_at_stream_plus_file() {
    let base = std::env::temp_dir().join("lensbolt_disk_peak_zstd");
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    let input = base.join("log.txt");
    {
        // text that zstd shrinks (a log): ~ 5-10x
        let mut f = std::io::BufWriter::new(std::fs::File::create(&input).unwrap());
        let (mut n, mut seed) = (0u64, 3u64);
        let mut line = String::new();
        while n < SIZE {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            line.clear();
            use std::fmt::Write as _;
            write!(line, "2026-10-30 12:{:02}:{:02} frame {} decoded {} ms codewords {}/83 ok\n", (seed >> 8) % 60, (seed >> 16) % 60, seed % 100_000, (seed >> 24) % 90, (seed >> 32) % 84).unwrap();
            let take = (line.len() as u64).min(SIZE - n) as usize;
            f.write_all(&line.as_bytes()[..take]).unwrap(); n += take as u64;
        }
        f.flush().unwrap();
    }
    let (peak, reported, stream, file, _) = run(&base, &input, None, 42);
    eprintln!("zstd 64 MB: stream {stream}, file {file}, measured peak {peak} ({:.3} x stream, {:.3} x file), reported {reported}", peak as f64 / stream as f64, peak as f64 / file as f64);
    assert!(stream < file / 2, "the log compresses: stream {stream}");
    // the part stays until the unpacked file is written and verified next to it
    assert_eq!(reported, stream + file);
    assert!(peak <= reported, "measured {peak} > reported {reported}");
    assert!(peak > file, "the poller saw the part and the unpacked file side by side: {peak}");
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn live_status_carries_the_disk_peak() {
    let st = lensbolt_core::flat::live::LiveStatus::default();
    assert!(st.to_json().contains("\"disk_peak\":0"), "{}", st.to_json());
}

fn tree_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir).map(|rd| rd.filter_map(|e| e.ok()).map(|e| match e.metadata() {
        Ok(m) if m.is_dir() => tree_bytes(&e.path()),
        Ok(m) => m.len(),
        Err(_) => 0,
    }).sum()).unwrap_or(0)
}

#[test]
fn batch_peaks_at_the_stream_plus_its_files() {
    // a batch on disk: the part file (the stream) stays until the end while each file is saved beside it, so the
    // peak is the stream plus the files. Before the file list is in, the receiver says the stream twice less the
    // list (exact when no file travels compressed); once it is in, the exact figure, which it keeps
    let base = std::env::temp_dir().join("lensbolt_disk_peak_batch");
    let _ = std::fs::remove_dir_all(&base);
    let out = base.join("out");
    std::fs::create_dir_all(&out).unwrap();
    let mut seed = 11u64;
    let mut noise = |n: usize| (0..n).map(|_| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed as u8 }).collect::<Vec<u8>>();
    let mut files: Vec<(String, lensbolt_core::file::BatchSource)> = (0..12).map(|i| (format!("photo-{i}.jpg"), lensbolt_core::file::BatchSource::Bytes(noise(1_500_000 + 250_000 * i)))).collect();
    files.push(("log.txt".into(), lensbolt_core::file::BatchSource::Bytes(b"a line of a log that packs well ".repeat(200_000))));
    let size_of = |f: &(String, lensbolt_core::file::BatchSource)| match &f.1 { lensbolt_core::file::BatchSource::Bytes(b) => b.len() as u64, _ => 0 };
    let (sizes, largest): (u64, u64) = (files.iter().map(size_of).sum(), files.iter().map(size_of).max().unwrap());
    let cfg = FlatSenderConfig { manifest_every_frames: 60, transfer_id: 43, ..FlatSenderConfig::default() };
    let mut s = FlatSender::from_batch("photos", files, cfg).unwrap();
    let (stream, index_len) = (s.manifest().stream_size, s.manifest().batch.as_ref().unwrap().index_len);
    assert_eq!(s.manifest().receive_disk_peak(), 2 * stream - index_len);
    let peak = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let poller = {
        let (peak, stop, out) = (peak.clone(), stop.clone(), out.clone());
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) { peak.fetch_max(tree_bytes(&out), Ordering::Relaxed); std::thread::yield_now(); }
            peak.fetch_max(tree_bytes(&out), Ordering::Relaxed);
        })
    };
    let mut r = Receiver::new(&out).unwrap();
    let mut done = None;
    let mut seen = Vec::new();
    for _ in 0..3 * s.frames_per_pass() {
        let (h, ps) = s.next_packets().unwrap();
        if let FeedOutcome::Complete(p) = r.feed_packets(&h, &ps).unwrap() { done = Some(p); }
        let p = r.progress().disk_peak;
        if seen.last() != Some(&p) { seen.push(p); }
        if done.is_some() { break; }
    }
    stop.store(true, Ordering::Relaxed);
    poller.join().unwrap();
    let dir = done.expect("completes within 3 passes");
    let peak = peak.load(Ordering::Relaxed);
    eprintln!("batch of 13 files ({sizes} bytes, a stream of {stream}): reported {seen:?}, measured peak {peak}");
    assert_eq!(*seen.last().unwrap(), stream + sizes, "the stream and every file");
    assert!(seen.len() <= 3, "0, maybe the estimate, then the exact figure: {seen:?}");
    // the poller lists the output folder, then the batch's folder: a file moved from the one to the other in
    // between is counted twice, so one file of slack (the largest) for the measurement, none in what is reported
    assert!(peak <= stream + sizes + largest, "measured {peak} > reported {} (+ {largest} for a file counted twice)", stream + sizes);
    assert!(peak > stream, "the poller saw saved files beside the part");
    assert_eq!(tree_bytes(&out), sizes, "only the files are left");
    assert_eq!(tree_bytes(&dir), sizes);
    let _ = std::fs::remove_dir_all(&base);
}
