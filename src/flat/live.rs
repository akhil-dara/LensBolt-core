//! Live receiver for a camera: captures in, file out. The camera thread offers each capture with `submit`,
//! which never blocks: a capture is queued only while a worker is free (otherwise dropped and counted; the
//! fountain code does not care which frames arrive). Workers decode with the geometry of the latest good
//! read as a hint; one collector thread owns the stitcher and feeds the receiver, and works outside the status
//! lock, which it takes only to copy in what it computed (the camera and the status never wait on stitching or the
//! disk; only `batch_page` waits for the receiver, between feeds, and `try_batch_page` never does). A decoder panic costs one capture, never the
//! session.

use super::decoder::{DecodeError, FlatDecoder, FlatHint, FlatRead};
use super::format::CellFormat;
use super::rate::RateEstimate;
use super::read::read_header;
use super::session::edge_sampling;
use super::stitch::Stitcher;
use crate::file::{BatchCounts, FeedOutcome, FileStage, Receiver, StoreFactory};
use crate::header::FrameType;
use crate::image::{GrayImage, RgbImage};
use crate::packet::unpack_frame;
use crate::tune::{HintSample, HintWatch, LiveHint, TuneMeter, TuneOptions, TuneSample};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

/// Every frame visible in one capture, one read per format. The tracked path reads one frame; a read whose
/// blocks carry two frame tags is a straddled capture, so the other header copy (maybe another format, a
/// manifest under a data frame) gets its own read. Without a hint, or when tracking fails: full search.
pub fn decode_capture(dec: &FlatDecoder, img: &RgbImage, hint: Option<&FlatHint>) -> Result<Vec<FlatRead>, DecodeError> {
    let frs = match (hint, dec.decode_tracked(img, hint)) {
        (Some(_), Ok(fr)) => {
            // two frame tags: a straddled capture (unknown and flat blocks are no frame's)
            let mut tags: Vec<u8> = fr.reads.block_tag.iter().copied().filter(|&t| super::read::tag_known(t)).collect(); tags.sort_unstable(); tags.dedup();
            let mut frs = vec![fr];
            if tags.len() > 1 {
                let (chrome, h) = (*frs[0].grid.chrome(), frs[0].homography);
                for hdr in read_header(img, &chrome, &h).into_iter().flatten() {
                    if (hdr.format, hdr.cols, hdr.rows) == (frs[0].header.format, frs[0].header.cols, frs[0].header.rows) { continue; }
                    if let Ok(other) = dec.read_frame(img, &chrome, &h, hdr) { frs.push(other); }
                }
            }
            frs
        }
        _ => dec.decode_all(img)?,
    };
    // same-format frames share a read (their block tags tell them apart in the stitcher)
    let mut seen: Vec<(CellFormat, u16, u16)> = Vec::new();
    Ok(frs.into_iter().filter(|fr| { let k = (fr.header.format, fr.header.cols, fr.header.rows); let new = !seen.contains(&k); if new { seen.push(k); } new }).collect())
}

/// Luma first (b/w and grey frames never build RGB), the RGB path when the header names a colour alphabet.
/// The bool says whether RGB was needed.
pub fn decode_luma_first(dec: &FlatDecoder, y: &GrayImage, rgb: impl FnOnce() -> RgbImage, hint: Option<&FlatHint>) -> Result<(Vec<FlatRead>, bool), DecodeError> {
    match dec.decode_luma(y, hint) {
        // the colour cells read where the luma pass found the tile (a second search of the whole frame cost more
        // than the read: a phone's manifest capture took 160 ms on one thread); the full RGB path when that fails
        Err(DecodeError::NeedsColour(_, Some(at))) => {
            let img = rgb();
            let mut frs: Vec<FlatRead> = Vec::new();
            for hdr in at.headers.into_iter().flatten() {
                if frs.iter().any(|r: &FlatRead| (r.header.format, r.header.cols, r.header.rows) == (hdr.format, hdr.cols, hdr.rows)) { continue; }
                if let Ok(fr) = dec.read_frame(&img, &at.chrome, &at.homography, hdr) { frs.push(fr); }
            }
            let frs = if frs.is_empty() { decode_capture(dec, &img, hint)? } else { frs };
            Ok((frs.into_iter().map(|mut r| { r.reads.edge_px = super::fast::edge_spread(y, r.grid.chrome(), &r.homography); r }).collect(), true))
        }
        // the border's edge width from the Y plane, as the luma path measures it
        Err(DecodeError::NeedsColour(_, None)) => decode_capture(dec, &rgb(), hint).map(|mut f| {
            for r in &mut f { r.reads.edge_px = super::fast::edge_spread(y, r.grid.chrome(), &r.homography); }
            (f, true)
        }),
        other => other.map(|f| (f, false)),
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LiveConfig { pub workers: usize }
impl Default for LiveConfig {
    /// Half the cores, each worker decoding on its own thread (`par::serial_here`): the captures go side by side.
    fn default() -> Self { LiveConfig { workers: (std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2) / 2).clamp(1, 4) } }
}

/// What the app shows. Rates are per wall second since the first capture that showed a tile.
#[derive(Clone, Debug, Default)]
pub struct LiveStatus {
    pub received: u64, pub dropped: u64, pub decoded: u64, pub failed: u64,
    pub last_error: Option<String>, pub decode_ms: f64,
    /// last read: "1px bw 1024x1024", and the codewords decoded and all of its frame so far
    /// (`Stitcher::last_codewords`), also when the capture added no new packet
    pub format: Option<String>, pub codewords_ok: usize, pub codewords: usize,
    /// clean luma data frames in a row stitched with under a fifth of their codewords, and the re-anchors they
    /// caused (the workers' shared geometry dropped: the next captures identify the tile from scratch). A tracked
    /// geometry gone wrong reads confidently and wrongly, so only the codewords show it (phone, 2026-11-02: 111 of
    /// 111 at 1.7 Mbit/s, then 3-5 of 111 for 35 s with clean captures and sharp edges)
    pub poor_frames: u32, pub reanchors: u64,
    /// last read's geometry: camera px per screen px along each tile edge (top, right, bottom, left; their
    /// spread is the perspective) and the smallest times the cell pitch, camera px per cell (how large the
    /// cells read); zero until a tile is read. The app's framing guidance reads both.
    pub edges: [f64; 4], pub cell_px: f64,
    /// how the last data capture read (b/w luma reads): the share of the tile where another sender frame has
    /// more than a quarter of the light (an exposure across the sender's frame change, `FlatReads::mix`), and the
    /// 10-90% width in camera px of the chrome border's edges (focus and motion blur, `FlatReads::edge_px`)
    pub mix: f64, pub edge_px: f64,
    pub have_manifest: bool, pub filename: Option<String>, pub bytes_done: u64, pub stream_size: u64,
    /// live progress: new (unique) 1 KB fountain blocks received, capped at the stream size (`bytes_done`
    /// grows only when a whole 16 MB segment verifies)
    pub received_bytes: u64,
    pub segments_done: u32, pub segments_total: u32,
    /// bytes the receiver will hold on disk at its peak (`Progress::disk_peak`); 0 until the manifest is in
    pub disk_peak: u64,
    pub seconds: f64, pub mbit_per_s: f64,
    /// decode time (moving average) of captures read on the luma path and of those that needed RGB
    pub luma_ms: f64, pub colour_ms: f64,
    /// captures whose header asked for colour when only the Y plane was copied (a manifest between chroma
    /// copies): not read, and not decode failures either
    pub colour_skipped: u64,
    /// the sender's new-frame rate from the frame numbers of decoded captures against their offer times
    /// (`RateEstimate`); 0.0 until known, once the frame number has stopped, and once nothing has decoded
    /// for 2 s
    pub sender_fps: f64,
    pub complete: Option<PathBuf>, pub receiver_error: Option<String>,
    /// the words to type on the sender once repair starts (`Receiver::finish_words`)
    pub finish_words: Option<String>,
    /// the parts those words name, "Parts 7, 12–15, 30" (`Receiver::finish_parts`)
    pub finish_parts: Option<String>,
    /// a batch's counts (`Receiver::batch_counts`); None for a single file. Its files: `LiveReceiver::batch_page`
    pub batch: Option<BatchCounts>,
    /// the verified file's SHA-256 once it is complete (`Receiver::file_sha256`); None for a batch (each file's is
    /// on its page) and before completion. The status's "sha256", as the browser's
    pub file_sha256: Option<[u8; 32]>,
    /// captures read that showed data alone (not the manifest): the browser's "data_reads". The app's "locked on"
    /// counts its reads by this, not its polls
    pub data_reads: u64,
    /// the last manifest read was the Tune carousel's (`Progress::tune_sender`): the sender is tuning, and this
    /// receiver holds no lock on it
    pub tune_sender: bool,
    /// Tune while it runs (`LiveReceiver::start_tune`): the meter's status JSON (`TuneMeter::status_json`), the
    /// status's "tune"; None (null) when not tuning
    pub tune: Option<String>,
    /// the live hint (`tune::hint`): safer b/w words or Tune, while reading poorly; None when not (and while tuning)
    pub hint: Option<LiveHint>,
}

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() { match c { '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"), c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)), c => o.push(c) } }
    o.push('"');
    o
}
fn json_opt(s: &Option<String>) -> String { s.as_deref().map(json_str).unwrap_or_else(|| "null".into()) }
/// Bytes as lower-case hex (a SHA-256: 64 digits).
pub fn hex(bytes: &[u8]) -> String { bytes.iter().map(|b| format!("{b:02x}")).collect() }
/// A number JSON can carry: a non-finite one (`inf`, `NaN` are not JSON) goes out as 0.
pub(crate) fn fin(v: f64) -> f64 { if v.is_finite() { v } else { 0.0 } }

/// A batch's counts as the status JSON carries them, `{"name","total","received","failed","missing_files","error"}`
/// (`BatchCounts`: `total` 0 while the file list is not in; `missing_files` null without finish words; `error` null
/// unless the file list was refused), or `null` for a single file. A few numbers: read every frame.
pub fn batch_json(b: Option<&BatchCounts>) -> String {
    let Some(b) = b else { return "null".into() };
    format!("{{\"name\":{},\"total\":{},\"received\":{},\"failed\":{},\"missing_files\":{},\"error\":{}}}",
        json_str(&b.name), b.total, b.received, b.failed, b.missing_files.map_or_else(|| "null".into(), |n| n.to_string()), json_opt(&b.error))
}

/// The most files one page of a batch's list holds (a page of 1,000 is ~80 KB of JSON).
pub const BATCH_PAGE_MAX: u32 = 1_000;

/// Files `offset .. offset + count` (at most `BATCH_PAGE_MAX`) of the batch `rx` receives, in stream order (smallest
/// first), as JSON: `{"offset","total","files":[{"name","size","state"}]}`. `state` is "waiting", "receiving" (with
/// "progress", 0 to 0.99, the share of its bytes in), "received" (with "sha256", the hex SHA-256 its bytes were verified
/// against) or "failed" (with "reason", in words for the screen).
/// `offset` as asked; `total` the files in the list (0 for a single file or before the list is in); fewer files at the
/// end of the list, none past it. Costs the page only.
pub fn batch_page_json(rx: &Receiver, offset: u32, count: u32) -> String {
    let total = rx.batch_counts().map_or(0, |b| b.total);
    let files: Vec<String> = rx.batch_files(offset as usize, count.min(BATCH_PAGE_MAX) as usize).iter().map(|f| {
        let state = match &f.state {
            FileStage::Waiting => "\"waiting\"".to_string(),
            FileStage::Receiving(p) => format!("\"receiving\",\"progress\":{:.3}", fin(*p as f64)),
            FileStage::Received => match f.sha256 { Some(h) => format!("\"received\",\"sha256\":\"{}\"", hex(&h)), None => "\"received\"".to_string() },
            FileStage::Failed(why) => format!("\"failed\",\"reason\":{}", json_str(why)),
        };
        format!("{{\"name\":{},\"size\":{},\"state\":{state}}}", json_str(&f.name), f.size)
    }).collect();
    format!("{{\"offset\":{offset},\"total\":{total},\"files\":[{}]}}", files.join(","))
}

impl LiveStatus {
    pub fn to_json(&self) -> String {
        let e = self.edges.map(fin);
        format!("{{\"received\":{},\"dropped\":{},\"reanchors\":{},\"decoded\":{},\"failed\":{},\"last_error\":{},\"decode_ms\":{:.1},\"format\":{},\"codewords_ok\":{},\"codewords\":{},\"edges\":[{:.2},{:.2},{:.2},{:.2}],\"cell_px\":{:.2},\"mix\":{:.2},\"edge_px\":{:.2},\"have_manifest\":{},\"filename\":{},\"bytes_done\":{},\"received_bytes\":{},\"stream_size\":{},\"segments_done\":{},\"segments_total\":{},\"disk_peak\":{},\"seconds\":{:.3},\"mbit_per_s\":{:.3},\"luma_ms\":{:.1},\"colour_ms\":{:.1},\"colour_skipped\":{},\"sender_fps\":{:.1},\"complete\":{},\"receiver_error\":{},\"finish_words\":{},\"finish_parts\":{},\"batch\":{},\"data_reads\":{},\"sha256\":{},\"tune_sender\":{},\"tune\":{},\"hint\":{}}}",
            self.received, self.dropped, self.reanchors, self.decoded, self.failed, json_opt(&self.last_error), fin(self.decode_ms), json_opt(&self.format),
            self.codewords_ok, self.codewords, e[0], e[1], e[2], e[3], fin(self.cell_px), fin(self.mix), fin(self.edge_px), self.have_manifest, json_opt(&self.filename), self.bytes_done, self.received_bytes, self.stream_size,
            self.segments_done, self.segments_total, self.disk_peak, fin(self.seconds), fin(self.mbit_per_s), fin(self.luma_ms), fin(self.colour_ms), self.colour_skipped, fin(self.sender_fps),
            json_opt(&self.complete.as_ref().map(|p| p.display().to_string().replace('\\', "/"))), json_opt(&self.receiver_error), json_opt(&self.finish_words), json_opt(&self.finish_parts), batch_json(self.batch.as_ref()),
            self.data_reads, self.file_sha256.map_or_else(|| "null".into(), |h| format!("\"{}\"", hex(&h))), self.tune_sender, self.tune.as_deref().unwrap_or("null"),
            self.hint.as_ref().map_or_else(|| "null".into(), |h| h.to_json()))
    }
}

/// One camera capture: the Y plane (the b/w fast path reads only this) and, built only when a frame turns out
/// to be colour (the 12 px c8 manifest, colour data), the RGB image. `rgb` is None when the camera thread
/// copied only Y (`LiveReceiver::want_chroma` was false): a colour header then costs the capture.
/// `origin`: where `y` sits in the camera's picture, (0, 0) for the whole picture; a crop (`LiveReceiver::crop_rect`)
/// carries its corner, and its geometry is moved back into the picture's coordinates after the read.
pub struct Capture { pub y: GrayImage, pub rgb: Option<Box<dyn FnOnce() -> RgbImage + Send>>, pub origin: (u32, u32) }
/// A capture to build and decode, with the time it was offered (the sender's rate is read against it).
type Job = (Instant, Box<dyn FnOnce() -> Capture + Send>);
/// A worker's result: a read (decode ms, whether it needed RGB), a failure, or a colour header without
/// chroma; each with the capture's offer time.
enum Done { Ok(Vec<FlatRead>, f64, bool, Instant), Failed(String, f64, Instant), NoChroma(Instant) }
impl Done { fn offered(&self) -> Instant { match self { Done::Ok(.., t) | Done::Failed(.., t) | Done::NoChroma(t) => *t } } }

/// Which planes the camera thread copies. Y always; U and V (half of the copy) only while the tile may need
/// them: the manifest (colour c8) not yet read, or no luma tile read in the last second, or the last capture's
/// header asked for colour and it had no chroma, or the cadence's turn (every 4th capture). The manifest
/// shows on one frame in 31 (`MANIFEST_EVERY_FRAMES`), so a cadence alone could miss it for seconds. With a
/// period of 30 the cadence of 4 aliased with it (sender and camera both at 30 fps: half the phases never
/// landed on it); 31 is coprime to 4, so every phase meets it, but only once in 4 x 31 frames (~4 s at 30 fps):
/// still too slow, hence the manifest rule. Once it is read a b/w transfer needs no colour.
/// Fed one observation per capture back from the workers.
#[derive(Clone, Debug, Default)]
pub struct CopyPolicy { seen: u64, now: Option<Instant>, last_luma: Option<Instant>, starved: bool, manifest: bool }
/// A luma read older than this no longer vouches for the tile being b/w.
const LUMA_FRESH: Duration = Duration::from_secs(1);
impl CopyPolicy {
    pub fn new() -> CopyPolicy { CopyPolicy::default() }
    /// Whether the receiver has read the transfer's manifest.
    pub fn manifest_known(&mut self, known: bool) { self.manifest = known; }
    /// One capture back from the workers: `needed_colour` = its header asked for colour and it had no chroma;
    /// `decoded_luma` = it was read on the luma path.
    pub fn observe(&mut self, needed_colour: bool, decoded_luma: bool, now: Instant) {
        // captures can come back out of order (several workers): the clock never goes back
        let now = self.now.map_or(now, |n| n.max(now));
        self.seen += 1;
        self.now = Some(now);
        self.starved = needed_colour;
        if decoded_luma { self.last_luma = Some(now); }
    }
    /// Chroma for a reason (not the cadence).
    pub fn need_chroma(&self) -> bool {
        let stale = match (self.last_luma, self.now) { (Some(l), Some(n)) => n.saturating_duration_since(l) > LUMA_FRESH, _ => true };
        !self.manifest || stale || self.starved
    }
    /// Copy U and V for the next capture.
    pub fn want_chroma(&self) -> bool { self.need_chroma() || chroma_turn(self.seen) }
}
/// The cadence: every 4th capture copies chroma whatever the policy says (`n` captures before this one).
fn chroma_turn(n: u64) -> bool { n % 4 == 3 }

pub struct LiveReceiver {
    work: Option<mpsc::SyncSender<Job>>,
    free: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    /// the collector's part of the status; it locks only to copy in what it has already computed
    status: Arc<Mutex<LiveStatus>>,
    /// the camera thread's counters, lock-free: `submit_capture` and `skip` never wait on the collector
    received: AtomicU64, dropped: AtomicU64,
    /// the collector's `CopyPolicy::need_chroma` after the latest capture, and the captures accepted so far
    /// (the cadence): `want_chroma` reads both without a lock
    chroma: Arc<AtomicBool>, accepted: AtomicU64,
    /// a capture's header asked for colour and it had no chroma: set by the collector, taken by the next
    /// `want_chroma` (a later luma read does not clear it before the camera thread has copied chroma once)
    starved: Arc<AtomicBool>,
    /// the receiver: the collector feeds it, `batch_page` reads a page of its file list
    rx: Arc<Mutex<Receiver>>,
    /// Tune: the command the collector takes with its next capture, and whether Tune runs (every capture then copies
    /// chroma: the colour segments need it); `epoch` is the collector's clock
    tune_cmd: Arc<Mutex<Option<TuneCmd>>>, tuning: AtomicBool, epoch: Instant,
    /// the workers' shared geometry (`crop_rect` reads it), and a cropped capture that failed: the next capture
    /// is taken whole (the tile may have moved out of the crop)
    hint: Arc<Mutex<Option<FlatHint>>>, whole_next: Arc<AtomicBool>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

/// Start or stop Tune (`LiveReceiver::start_tune`, `stop_tune`), for the collector; Start carries the time on the
/// collector's clock.
enum TuneCmd { Start(TuneOptions, f64), Stop }

/// The receiver, also after a panic while it was held (the collector catches it; the receiver's state stays usable).
fn lock(rx: &Mutex<Receiver>) -> std::sync::MutexGuard<'_, Receiver> { rx.lock().unwrap_or_else(|e| e.into_inner()) }

impl LiveReceiver {
    pub fn new(out_dir: &Path, cfg: LiveConfig) -> io::Result<LiveReceiver> { LiveReceiver::start(Receiver::new(out_dir)?, cfg) }
    /// `new` with the stream and the file kept in `stores` (`Receiver::with_stores`): `complete` names the file.
    pub fn with_stores(stores: Box<dyn StoreFactory>, cfg: LiveConfig) -> io::Result<LiveReceiver> { LiveReceiver::start(Receiver::with_stores(stores), cfg) }
    fn start(receiver: Receiver, cfg: LiveConfig) -> io::Result<LiveReceiver> {
        let rx = Arc::new(Mutex::new(receiver));
        let workers = cfg.workers.max(1);
        let dec = Arc::new(FlatDecoder::new());
        let (work_tx, work_rx) = mpsc::sync_channel::<Job>(workers);
        let work_rx = Arc::new(Mutex::new(work_rx));
        let (done_tx, done_rx) = mpsc::channel::<Done>();
        let hint: Arc<Mutex<Option<FlatHint>>> = Arc::new(Mutex::new(None));
        let free = Arc::new(AtomicUsize::new(workers));
        let whole_next = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let status = Arc::new(Mutex::new(LiveStatus::default()));
        let chroma = Arc::new(AtomicBool::new(CopyPolicy::new().need_chroma()));
        let starved = Arc::new(AtomicBool::new(false));
        let mut threads = Vec::new();
        for i in 0..workers {
            let (rx, tx, dec, hint, free, stop, whole) = (work_rx.clone(), done_tx.clone(), dec.clone(), hint.clone(), free.clone(), stop.clone(), whole_next.clone());
            threads.push(std::thread::Builder::new().name(format!("lb-decode-{i}")).spawn(move || { crate::par::serial_here(); loop {
                let job = { let Ok(guard) = rx.lock() else { break }; guard.recv() };
                let Ok((offered, job)) = job else { break };
                if stop.load(Ordering::Relaxed) { free.fetch_add(1, Ordering::AcqRel); continue; }
                let t = Instant::now();
                let h = hint.lock().map(|g| g.clone()).unwrap_or(None);
                // luma first (b/w frames never build RGB); a colour header falls back to the RGB path
                let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let cap = job();
                    // a crop: the hint moved into it, the reads moved back into the picture's coordinates
                    let (ox, oy) = (cap.origin.0 as f32, cap.origin.1 as f32);
                    let cropped = cap.origin != (0, 0);
                    let h = if cropped { h.map(|mut x| { x.homography = x.homography.translated(-ox, -oy); x }) } else { h };
                    let r = match cap.rgb {
                        Some(rgb) => decode_luma_first(&dec, &cap.y, rgb, h.as_ref()),
                        // only Y was copied: a colour header comes back as NeedsColour
                        None => dec.decode_luma(&cap.y, h.as_ref()).map(|f| (f, false)),
                    };
                    if !cropped { return r; }
                    // a crop that asked for colour lost nothing (the next colour copy is whole anyway); any other
                    // failure may be the tile leaving the crop
                    if matches!(&r, Err(e) if !matches!(e, DecodeError::NeedsColour(..))) { whole.store(true, Ordering::Release); }
                    r.map(|(mut frs, c)| { for f in &mut frs { f.homography = f.homography.translated(ox, oy); } (frs, c) })
                }));
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                let msg = match res {
                    Ok(Ok((frs, colour))) => { if let Ok(mut g) = hint.lock() { *g = Some(frs[0].hint()); } Done::Ok(frs, ms, colour, offered) }
                    Ok(Err(DecodeError::NeedsColour(..))) => Done::NoChroma(offered),
                    Ok(Err(e)) => Done::Failed(format!("{e:?}"), ms, offered),
                    Err(_) => Done::Failed("Panic".into(), ms, offered),
                };
                free.fetch_add(1, Ordering::AcqRel);
                if tx.send(msg).is_err() { break; }
            } })?);
        }
        drop(done_tx);
        let (st, stop_c, chroma_c, starved_c, receiver) = (status.clone(), stop.clone(), chroma.clone(), starved.clone(), rx.clone());
        let tune_cmd: Arc<Mutex<Option<TuneCmd>>> = Arc::new(Mutex::new(None));
        let tune_c = tune_cmd.clone();
        let hint_c = hint.clone();
        // the clock the sender's rate is measured on: started before any capture can be offered
        let epoch = Instant::now();
        threads.push(std::thread::Builder::new().name("lb-collect".into()).spawn(move || {
            let mut stitcher = Stitcher::new();
            let mut t_first: Option<Instant> = None;
            // the collector's own copy of its status fields: stitching, error correction, feeding the receiver
            // (disk writes, segment SHA-256) and its progress all happen outside the shared lock
            let mut s = LiveStatus::default();
            let mut policy = CopyPolicy::new();
            let mut rate = RateEstimate::new();
            let mut meter: Option<TuneMeter> = None;
            // the live hint, and the data format and cell size of the last capture that showed data alone (a manifest
            // read is the same tile at 12 px: it says nothing of how the data reads)
            let mut watch = HintWatch::new();
            let mut data_read: Option<((CellFormat, u32, u32), f64)> = None;
            while let Ok(msg) = done_rx.recv() {
                if stop_c.load(Ordering::Relaxed) { continue; }
                match tune_c.lock().ok().and_then(|mut c| c.take()) {
                    Some(TuneCmd::Start(opts, at)) => meter = Some(TuneMeter::new(opts, at)),
                    Some(TuneCmd::Stop) => meter = None,
                    None => {}
                }
                let (needed_colour, decoded_luma) = (matches!(msg, Done::NoChroma(_)), matches!(msg, Done::Ok(_, _, false, _)));
                // every capture moves the rate's clock on (it does not hold while nothing decodes); decoded
                // ones also add their frame number
                let offered = msg.offered();
                let t = offered.saturating_duration_since(epoch).as_secs_f64() * 1000.0;
                if let Done::Ok(frs, ..) = &msg {
                    rate.add(frs[0].header.frame_seq, t);
                    if frs.iter().all(|r| r.header.frame_type == FrameType::Data) {
                        let (h, g) = (&frs[0].header, &frs[0].grid);
                        let cell = edge_sampling(&frs[0]).iter().copied().fold(f32::MAX, f32::min) as f64 * h.format.pitch as f64;
                        data_read = Some(((h.format, g.width_px(), g.height_px()), cell));
                    }
                }
                s.sender_fps = rate.fps_at(t).unwrap_or(0.0);
                collect(&mut s, msg, &mut stitcher, &receiver, &mut t_first, &stop_c, meter.as_mut(), t, &hint_c);
                // the meter's work runs under the same guard as the decoders': a panic ends Tune, never the session
                s.tune = meter.as_ref().map(|m| std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| m.status_json(t)))
                    .unwrap_or_else(|_| "{\"state\":\"failed\",\"progress\":0.000,\"failure\":\"interrupted\",\"result\":null}".into()));
                // the hint watches normal transfers only
                s.hint = if meter.is_some() { watch = HintWatch::new(); None } else {
                    let hs = HintSample { t_ms: t, decoded: s.decoded, failed: s.failed, format: data_read.map(|d| d.0), cell_px: data_read.map_or(0.0, |d| d.1),
                        codewords_ok: s.codewords_ok as u64, codewords: s.codewords as u64, mix: s.mix, received_bytes: s.bytes_done.max(s.received_bytes), stream_size: s.stream_size,
                        complete: s.complete.is_some(), sender_fps: s.sender_fps };
                    watch.update(&hs).cloned()
                };
                policy.observe(needed_colour, decoded_luma, offered);
                policy.manifest_known(s.have_manifest);
                // before the status: a caller that sees this capture counted sees the policy that followed it
                if needed_colour { starved_c.store(true, Ordering::Release); }
                chroma_c.store(policy.need_chroma(), Ordering::Release);
                let snap = s.clone();
                let Ok(mut g) = st.lock() else { break };
                *g = snap;
            }
        })?);
        Ok(LiveReceiver { work: Some(work_tx), free, stop, status, chroma, starved, rx, received: AtomicU64::new(0), dropped: AtomicU64::new(0), accepted: AtomicU64::new(0), threads,
            tune_cmd, tuning: AtomicBool::new(false), epoch, hint, whole_next })
    }

    /// Offer a capture; `make` runs on a worker (convert the camera planes there, not on the camera thread).
    /// Returns false when every worker is busy or the transfer is complete: the capture is dropped.
    pub fn submit<F: FnOnce() -> RgbImage + Send + 'static>(&self, make: F) -> bool {
        self.submit_capture(move || { let img = make(); Capture { y: img.to_gray(), rgb: Some(Box::new(move || img)), origin: (0, 0) } })
    }
    /// `submit` for a camera that hands out a Y plane: `make` builds the capture on a worker.
    /// The offer time is the capture's time for the sender's frame-rate estimate and the copy policy.
    pub fn submit_capture<F: FnOnce() -> Capture + Send + 'static>(&self, make: F) -> bool { self.submit_capture_at(Instant::now(), make) }
    /// `submit_capture` with the capture's time given (a camera timestamp, or a test's simulated clock): the
    /// sender's rate and the copy policy's "no luma read for a second" are judged on these times. Times must
    /// not go back, and not before this receiver was made.
    pub fn submit_capture_at<F: FnOnce() -> Capture + Send + 'static>(&self, offered: Instant, make: F) -> bool {
        self.received.fetch_add(1, Ordering::AcqRel);
        // take a worker slot only if one is left: two submitters racing for the last slot cannot wrap `free`
        let accepted = !self.stop.load(Ordering::Relaxed) && self.free.fetch_update(Ordering::AcqRel, Ordering::Acquire, |f| f.checked_sub(1)).is_ok() && {
            match self.work.as_ref().map(|w| w.try_send((offered, Box::new(make)))) {
                Some(Ok(())) => true,
                _ => { self.free.fetch_add(1, Ordering::AcqRel); false }
            }
        };
        if accepted { self.accepted.fetch_add(1, Ordering::AcqRel); } else { self.dropped.fetch_add(1, Ordering::AcqRel); }
        accepted
    }
    /// The region of a `width` x `height` picture to copy for the next capture while the tile is tracked: the
    /// tile's corners through the latest geometry, a margin of an eighth of its size (at least 32 px) around,
    /// as [x0, y0, x1, y1). None (copy the whole picture): no geometry yet, the last cropped capture failed (taken
    /// once), the region would be most of the picture anyway, or a worker holds the geometry right now. A whole
    /// 12 MP plane costs 6 ms to copy on the camera's thread, the tile's region under 1 ms. A capture that also
    /// needs colour (`want_chroma`) is copied whole: crops are luma only.
    pub fn crop_rect(&self, width: u32, height: u32) -> Option<[u32; 4]> {
        if self.whole_next.swap(false, Ordering::AcqRel) { return None; }
        let hint = self.hint.try_lock().ok()?.clone()?;
        let c = hint.chrome;
        let (w, h) = (c.width_px() as f32, c.height_px() as f32);
        let pts = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)].map(|p| hint.homography.apply(p.0, p.1));
        let (x0, x1) = pts.iter().fold((f32::MAX, f32::MIN), |a, p| (a.0.min(p.0), a.1.max(p.0)));
        let (y0, y1) = pts.iter().fold((f32::MAX, f32::MIN), |a, p| (a.0.min(p.1), a.1.max(p.1)));
        if !(x0.is_finite() && x1.is_finite() && y0.is_finite() && y1.is_finite()) { return None; }
        let m = (0.125 * (x1 - x0).max(y1 - y0)).max(32.0);
        let r = [(x0 - m).max(0.0) as u32, (y0 - m).max(0.0) as u32, ((x1 + m).max(0.0) as u32).min(width), ((y1 + m).max(0.0) as u32).min(height)];
        if r[2] <= r[0] + 64 || r[3] <= r[1] + 64 { return None; }
        let area = (r[2] - r[0]) as u64 * (r[3] - r[1]) as u64;
        if area * 10 > width as u64 * height as u64 * 6 { return None; }
        Some(r)
    }
    /// Count a capture the caller dropped without offering it (no worker was free to take its planes).
    pub fn skip(&self) { self.received.fetch_add(1, Ordering::AcqRel); self.dropped.fetch_add(1, Ordering::AcqRel); }
    /// A worker is free: copy the camera planes only when this is true.
    pub fn has_room(&self) -> bool { self.free.load(Ordering::Acquire) > 0 }
    /// Copy the U and V planes for the next capture (the Y plane always): a colour header seen without chroma
    /// since the last call (this call takes it), the collector's policy after the latest capture, or the
    /// cadence's turn. Lock-free, for the camera thread.
    pub fn want_chroma(&self) -> bool {
        let starved = self.starved.swap(false, Ordering::AcqRel);
        starved || self.tuning.load(Ordering::Acquire) || self.chroma.load(Ordering::Acquire) || chroma_turn(self.accepted.load(Ordering::Acquire))
    }
    /// Starts Tune: from the next capture on, data reads are measured (`TuneMeter`), not received, and every
    /// capture copies chroma; the status's "tune" says where it stands. Starting again starts over.
    pub fn start_tune(&self, opts: TuneOptions) { self.start_tune_at(opts, Instant::now()) }
    /// `start_tune` at `at` (the captures' clock, as `submit_capture_at`).
    pub fn start_tune_at(&self, opts: TuneOptions, at: Instant) {
        let t = at.saturating_duration_since(self.epoch).as_secs_f64() * 1000.0;
        self.tuning.store(true, Ordering::Release);
        if let Ok(mut c) = self.tune_cmd.lock() { *c = Some(TuneCmd::Start(opts, t)); }
    }
    /// Back to receiving: the status's "tune" goes null with the next capture.
    pub fn stop_tune(&self) {
        self.tuning.store(false, Ordering::Release);
        if let Ok(mut c) = self.tune_cmd.lock() { *c = Some(TuneCmd::Stop); }
    }
    /// Files `offset .. offset + count` of the batch being received (`batch_page_json`; the status carries only the
    /// batch's counts). It waits for the collector's feed under way, which may write files: call it off the UI thread.
    pub fn batch_page(&self, offset: u32, count: u32) -> String { batch_page_json(&lock(&self.rx), offset, count) }
    /// `batch_page` without the wait: None (busy) while the collector is inside a feed, which may be writing a large
    /// file for seconds; the caller asks again later (the phone's list on its next tick). Never blocks, so a caller
    /// may hold its own lock around the handle while it asks.
    pub fn try_batch_page(&self, offset: u32, count: u32) -> Option<String> {
        match self.rx.try_lock() {
            Ok(g) => Some(batch_page_json(&g, offset, count)),
            Err(std::sync::TryLockError::Poisoned(e)) => Some(batch_page_json(&e.into_inner(), offset, count)),
            Err(std::sync::TryLockError::WouldBlock) => None,
        }
    }
    pub fn status(&self) -> LiveStatus {
        let mut s = self.status.lock().map(|s| s.clone()).unwrap_or_default();
        // dropped first: a capture is counted received before it can be counted dropped, so received >= dropped
        s.dropped = self.dropped.load(Ordering::Acquire);
        s.received = self.received.load(Ordering::Acquire);
        s
    }
}

/// One worker result into the collector's status copy `s`: counters and decode times, then (for a read) the
/// stitcher, the receiver and its progress. The caller publishes `s`.
/// `tune`: Tune runs: a capture's data reads go to the meter (at `t_ms`, the collector's clock), not the receiver.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn collect(s: &mut LiveStatus, msg: Done, stitcher: &mut Stitcher, receiver: &Mutex<Receiver>, t_first: &mut Option<Instant>, stop: &AtomicBool, mut tune: Option<&mut TuneMeter>, t_ms: f64, hint: &Mutex<Option<FlatHint>>) {
    let (ms, read) = match msg {
        Done::NoChroma(_) => { s.colour_skipped += 1; return; }
        Done::Failed(e, ms, _) => (ms, Err(e)),
        Done::Ok(frs, ms, colour, _) => (ms, Ok((frs, colour))),
    };
    s.decode_ms = if s.decoded + s.failed == 0 { ms } else { s.decode_ms * 0.8 + ms * 0.2 };
    let frs = match read {
        Err(e) => {
            s.failed += 1; s.last_error = Some(e);
            if let Some(m) = tune.as_deref_mut() {
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| m.capture(TuneSample { t_ms, read: None }))).is_err() { m.abort(); }
            }
            return;
        }
        Ok((frs, colour)) => {
            let avg = if colour { &mut s.colour_ms } else { &mut s.luma_ms };
            *avg = if *avg == 0.0 { ms } else { *avg * 0.8 + ms * 0.2 };
            frs
        }
    };
    s.decoded += 1;
    let t0 = *t_first.get_or_insert_with(Instant::now);
    let h = &frs[0].header;
    let luma = h.format.alphabet.is_luma();
    s.format = Some(format!("{}px {} {}x{}", h.format.pitch, h.format.alphabet.name(), frs[0].grid.width_px(), frs[0].grid.height_px()));
    let e = edge_sampling(&frs[0]);
    s.edges = e.map(|v| v as f64);
    s.cell_px = e.iter().copied().fold(f32::MAX, f32::min) as f64 * h.format.pitch as f64;
    // the blend and the edge width of a capture that showed data alone (the manifest is read on the colour path)
    if frs.iter().all(|r| r.header.frame_type == FrameType::Data) { (s.mix, s.edge_px) = (frs[0].reads.mix as f64, frs[0].reads.edge_px as f64); s.data_reads += 1; }
    // Tune: the data reads are measured, not received; manifests go on to the receiver
    let tuning = tune.is_some();
    let frs: Vec<FlatRead> = match tune {
        Some(m) => {
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| crate::tune::meter::feed_reads(m, &frs, t_ms))).is_err() { m.abort(); }
            frs.into_iter().filter(|r| r.header.frame_type == FrameType::Manifest).collect()
        }
        None => frs,
    };
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut done = None;
        if frs.is_empty() { return (None, None); }
        let frames = stitcher.add_capture(&frs);
        // this capture's frame, also when it added no packet (the payloads report only frames with new ones);
        // only a capture that showed data alone: the manifest (12 px colour) always reads whole and says
        // nothing about how the data reads (a straddled capture may end on the manifest's codewords)
        let last = (!tuning && frs.iter().all(|r| r.header.frame_type == FrameType::Data)).then(|| stitcher.last_codewords());
        // held for the feed only: stitching runs outside it
        let mut receiver = lock(receiver);
        for (header, fp) in frames {
            match receiver.feed_packets(&header, &unpack_frame(&fp)) {
                Ok(FeedOutcome::Complete(p)) => { done = Some(Ok(p)); break; }
                Ok(_) => {}
                Err(e) => { done = Some(Err(e.to_string())); }
            }
        }
        (last, done)
    }));
    let (last, done) = res.unwrap_or((None, Some(Err("receiver panic".into()))));
    if let Some((ok, total)) = last {
        (s.codewords_ok, s.codewords) = (ok, total);
        // two clean luma data frames in a row under a fifth of their codewords: the shared geometry is dropped (a
        // blend reads few for its own reason, mix 0.49-0.59 on the soft set; colour frames read partially by nature)
        s.poor_frames = if luma && total > 0 && ok * 5 < total && s.mix < 0.15 { s.poor_frames + 1 } else { 0 };
        if s.poor_frames >= 2 { s.poor_frames = 0; s.reanchors += 1; if let Ok(mut g) = hint.lock() { *g = None; } }
    }
    let receiver = lock(receiver);
    let p = receiver.progress();
    (s.have_manifest, s.filename, s.bytes_done, s.stream_size, s.segments_done, s.segments_total) =
        (p.have_manifest, p.filename.clone(), p.bytes_done, p.stream_size, p.segments_done, p.segments_total);
    s.disk_peak = p.disk_peak;
    s.tune_sender = p.tune_sender.is_some();
    s.finish_words = receiver.finish_words();
    s.finish_parts = receiver.finish_parts();
    s.batch = receiver.batch_counts();
    s.file_sha256 = receiver.file_sha256();
    s.received_bytes = if p.stream_size > 0 { (p.packets_new * crate::fountain::BLOCK_BYTES as u64).min(p.stream_size) } else { 0 };
    s.seconds = t0.elapsed().as_secs_f64();
    if s.seconds > 0.0 { s.mbit_per_s = s.bytes_done.max(s.received_bytes) as f64 * 8.0 / s.seconds / 1e6; }
    match done {
        // stop before the status says complete: a caller that sees `complete` gets no capture accepted
        Some(Ok(path)) => { stop.store(true, Ordering::Relaxed); s.complete = Some(path); }
        Some(Err(e)) => s.receiver_error = Some(e),
        None => {}
    }
}

impl Drop for LiveReceiver {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.work.take();
        for t in self.threads.drain(..) { let _ = t.join(); }
    }
}
