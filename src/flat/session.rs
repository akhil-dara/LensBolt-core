//! Single-threaded receive session for a browser (one WebAssembly instance in a worker): camera RGBA frames in,
//! a status line out, the verified file out. The same decode as the phone app (luma first, RGB only for colour
//! frames, the previous capture's geometry as the hint), the stitcher, and an in-memory receiver.
//!
//! The session is a reader (`FrameReader`: the decoder and its tracked geometry) and a collector (the stitcher, the
//! receiver, the status) in one. A browser with cores to spare runs readers in workers of their own and hands their
//! records (`FrameRead::to_bytes`) to one session's `push_read`: the same status as `push_i420_at`, byte for byte.

use super::decoder::{FlatDecoder, FlatHint, FlatRead};
use super::header::{FlatHeader, FLAT_HEADER_BYTES};
use super::layout::{Chrome, FlatGrid};
use super::live::{decode_luma_first, fin};
use super::rate::RateEstimate;
use super::read::FlatReads;
use super::stitch::Stitcher;
use crate::detect::Homography;
use crate::file::{FeedOutcome, Receiver, StoreFactory};
use crate::header::FrameType;
use crate::image::{GrayImage, RgbImage};
use crate::packet::unpack_frame;
use crate::tune::{HintSample, HintWatch, LiveHint, TuneMeter, TuneOptions, TuneSample};
use std::sync::Arc;

/// What a reader made of one camera frame: the collector's whole input (`RxSession::push_read`).
pub enum FrameRead {
    /// an empty frame (no width or height): counted, nothing else
    NoFrame,
    /// planes that do not fit the frame's size or layout
    BadFrame,
    /// no tile read, and why (`DecodeError` in words: the status's last miss)
    Failed(String),
    /// every frame the capture showed (`live::decode_luma_first`), and the sharpness gate's score of it
    Read { reads: Vec<FlatRead>, sharpness: f32 },
}

/// A frame reader: the decoder and the geometry of the latest frame it read (the next frame's hint). One per
/// worker; each tracks the frames it is given, and takes another reader's geometry with `set_hint`.
/// The share of a read's payload cells whose margin lies within the erasure margin (the cells the stitcher
/// would treat as doubtful); 0 for a read without payload cells.
fn weak_share(r: &FlatRead) -> f32 {
    let em = super::extract::erasure_margin();
    let (grid, cols) = (&r.grid, r.reads.cols);
    let n = grid.used_data_cells();
    if n == 0 || r.reads.margins.is_empty() { return 0.0; }
    let weak = (0..n as u32).filter(|&i| { let (c, rr) = grid.data_position(i); r.reads.margins[(rr * cols + c) as usize] < em }).count();
    weak as f32 / n as f32
}

pub struct FrameReader { dec: FlatDecoder, hint: Option<FlatHint>, poor: u32 }

impl Default for FrameReader { fn default() -> Self { Self::new() } }

/// A tracked read is poor when more than this share of its payload cells decided within the erasure margin
/// (a right read of a clean capture leaves ~0.1% there; a read from a wrong geometry leaves most cells there).
const POOR_WEAK_SHARE: f32 = 0.25;
/// Tracked reads that poor in a row before the reader drops its hint and identifies the tile again from
/// scratch (the narrow tracking search cannot leave a geometry gone wrong on its own).
const POOR_READS_RESET: u32 = 3;
/// Data frames in a row stitched with under a fifth of their codewords before the session clears the geometry.
const POOR_FRAMES_RESET: u32 = 2;

impl FrameReader {
    pub fn new() -> FrameReader { FrameReader { dec: FlatDecoder::new(), hint: None, poor: 0 } }
    /// One camera frame, RGBA as `getImageData` gives it.
    pub fn read_rgba(&mut self, rgba: &[u8], width: u32, height: u32) -> FrameRead {
        let n = width as usize * height as usize;
        if rgba.len() < n * 4 { return FrameRead::BadFrame; }
        if n == 0 { return FrameRead::NoFrame; }
        let gray = GrayImage { width, height, data: rgba.chunks_exact(4).take(n).map(|p| ((77 * p[0] as u32 + 150 * p[1] as u32 + 29 * p[2] as u32) >> 8) as u8).collect() };
        let rgb = || {
            let mut img = RgbImage::new(width, height);
            for (d, s) in img.data.chunks_exact_mut(3).zip(rgba.chunks_exact(4)) { d.copy_from_slice(&s[..3]); }
            img
        };
        self.read(&gray, rgb)
    }
    /// One camera frame as I420 planes in one buffer (`RxSession::push_i420_at`'s layout).
    pub fn read_i420(&mut self, buf: &[u8], width: u32, height: u32, layout: [usize; 6]) -> FrameRead {
        let (w, h) = (width as usize, height as usize);
        if w == 0 || h == 0 { return FrameRead::NoFrame; }
        let [yo, ys, uo, us, vo, vs] = layout;
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let fits = |off: usize, stride: usize, pw: usize, ph: usize| ph == 0 || (stride >= pw && stride.checked_mul(ph - 1).and_then(|n| n.checked_add(off)).and_then(|n| n.checked_add(pw)).is_some_and(|end| end <= buf.len()));
        if !fits(yo, ys, w, h) || !fits(uo, us, cw, ch) || !fits(vo, vs, cw, ch) || us != vs { return FrameRead::BadFrame; }
        let gray = GrayImage::from_plane(width, height, &buf[yo..], ys);
        let rgb = || RgbImage::from_yuv420(width, height, &buf[yo..], ys, &buf[uo..], &buf[vo..], us, 1);
        self.read(&gray, rgb)
    }
    fn read(&mut self, gray: &GrayImage, rgb: impl FnOnce() -> RgbImage) -> FrameRead {
        let tracked = self.hint.is_some();
        match decode_luma_first(&self.dec, gray, rgb, self.hint.as_ref()) {
            Ok((frs, _)) => {
                // a tracked reader whose reads keep coming out poor re-anchors: the next read is cold
                self.poor = if tracked && weak_share(&frs[0]) > POOR_WEAK_SHARE { self.poor + 1 } else { 0 };
                self.hint = if self.poor >= POOR_READS_RESET { self.poor = 0; None } else { Some(frs[0].hint()) };
                FrameRead::Read { reads: frs, sharpness: self.dec.last_gate_score() }
            }
            Err(e) => { self.poor = 0; FrameRead::Failed(format!("{e:?}")) }
        }
    }
    /// Forget the tracked geometry: the next read identifies the tile from scratch (the collector asks for it when
    /// the frames keep reading few codewords: a tracked geometry gone wrong reads confidently and wrongly, so the
    /// reader's own margins do not show it).
    pub fn clear_hint(&mut self) { self.hint = None; self.poor = 0; }
    /// The geometry this reader tracks (the tile's size, where it is, the read's sampling shifts) as bytes for
    /// another reader's `set_hint`; empty before its first read.
    pub fn hint_bytes(&self) -> Vec<u8> {
        let Some(h) = &self.hint else { return Vec::new() };
        let mut w = Wr(Vec::new());
        w.0.extend_from_slice(HINT_MAGIC);
        w.u32(h.chrome.width_px()); w.u32(h.chrome.height_px());
        for v in h.homography.m { w.f32(v); }
        match &h.shifts { Some(s) => { w.u8(1); w.pairs(s); } None => w.u8(0) }
        w.0
    }
    /// Another reader's geometry (`hint_bytes`) as this reader's hint for its next frame. False (and the hint as it
    /// was) for bytes that are not a hint.
    pub fn set_hint(&mut self, bytes: &[u8]) -> bool {
        let read = || -> Option<FlatHint> {
            let mut r = Rd(bytes);
            if r.take(HINT_MAGIC.len())? != HINT_MAGIC { return None; }
            let chrome = Chrome::new(r.u32()?, r.u32()?)?;
            let mut m = [0f32; 9];
            for v in &mut m { *v = r.f32()?; }
            let shifts = match r.u8()? { 0 => None, 1 => Some(Arc::new(r.pairs()?)), _ => return None };
            r.0.is_empty().then_some(FlatHint { chrome, homography: Homography { m }, shifts })
        };
        match read() { Some(h) => { self.hint = Some(h); true } None => false }
    }
}

/// The first bytes of a record and of a hint (and their layout's version).
const RECORD_MAGIC: &[u8] = b"LBr1";
const HINT_MAGIC: &[u8] = b"LBh1";

/// Little-endian writer of a record.
struct Wr(Vec<u8>);
impl Wr {
    fn u8(&mut self, v: u8) { self.0.push(v); }
    fn u32(&mut self, v: u32) { self.0.extend_from_slice(&v.to_le_bytes()); }
    fn f32(&mut self, v: f32) { self.0.extend_from_slice(&v.to_bits().to_le_bytes()); }
    fn bytes(&mut self, v: &[u8]) { self.u32(v.len() as u32); self.0.extend_from_slice(v); }
    fn pairs(&mut self, v: &[(f32, f32)]) { self.u32(v.len() as u32); for &(a, b) in v { self.f32(a); self.f32(b); } }
}
/// Its reader: None past the end.
struct Rd<'a>(&'a [u8]);
impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> { if self.0.len() < n { return None; } let (a, b) = self.0.split_at(n); self.0 = b; Some(a) }
    fn u8(&mut self) -> Option<u8> { Some(self.take(1)?[0]) }
    fn u32(&mut self) -> Option<u32> { Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?)) }
    fn f32(&mut self) -> Option<f32> { self.u32().map(f32::from_bits) }
    fn f64(&mut self) -> Option<f64> { Some(f64::from_bits(u64::from_le_bytes(self.take(8)?.try_into().ok()?))) }
    /// A count of `unit`-byte items that the bytes left can hold.
    fn len(&mut self, unit: usize) -> Option<usize> { let n = self.u32()? as usize; (n.checked_mul(unit)? <= self.0.len()).then_some(n) }
    fn bytes(&mut self) -> Option<Vec<u8>> { let n = self.len(1)?; Some(self.take(n)?.to_vec()) }
    fn pairs(&mut self) -> Option<Vec<(f32, f32)>> { let n = self.len(8)?; (0..n).map(|_| Some((self.f32()?, self.f32()?))).collect() }
}

impl FrameRead {
    /// The record a reader hands to the collector: everything `RxSession` takes from a read, exactly (the reads'
    /// cells, margins, block tags and geometry, the header, the blend and edge width, the gate's score). The read's
    /// sampling shifts stay with the reader (the collector never uses them; another reader gets them in a hint).
    /// Little-endian: "LBr1", the kind (0 no frame, 1 bad frame, 2 failed, 3 read), then for a failure its reason,
    /// for a read the sharpness and each frame read.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = Wr(Vec::new());
        w.0.extend_from_slice(RECORD_MAGIC);
        match self {
            FrameRead::NoFrame => w.u8(0),
            FrameRead::BadFrame => w.u8(1),
            FrameRead::Failed(e) => { w.u8(2); w.bytes(e.as_bytes()); }
            FrameRead::Read { reads, sharpness } => {
                w.u8(3); w.f32(*sharpness); w.u32(reads.len() as u32);
                for fr in reads {
                    let r = &fr.reads;
                    w.0.reserve(r.values.len() * 3 + r.block_tag.len() + 128);
                    w.u32(fr.grid.width_px()); w.u32(fr.grid.height_px());
                    w.0.extend_from_slice(&fr.header.pack());
                    for v in fr.homography.m { w.f32(v); }
                    w.0.extend_from_slice(&fr.decode_ms.to_bits().to_le_bytes());
                    w.u32(r.cols); w.u32(r.rows); w.f32(r.alpha); w.f32(r.mix); w.f32(r.edge_px);
                    w.bytes(&r.values);
                    w.u32(r.margins.len() as u32);
                    for m in &r.margins { w.0.extend_from_slice(&m.to_le_bytes()); }
                    w.bytes(&r.block_tag);
                }
            }
        }
        w.0
    }
    /// A record back (`to_bytes`); None for anything else: cut short, a byte too many, a grid that is not a
    /// format, cells that do not match their grid.
    pub fn from_bytes(bytes: &[u8]) -> Option<FrameRead> {
        let mut r = Rd(bytes);
        if r.take(RECORD_MAGIC.len())? != RECORD_MAGIC { return None; }
        let out = match r.u8()? {
            0 => FrameRead::NoFrame,
            1 => FrameRead::BadFrame,
            2 => FrameRead::Failed(String::from_utf8(r.bytes()?).ok()?),
            3 => {
                let sharpness = r.f32()?;
                let n = r.u32()? as usize;
                // a capture shows at most a few frames (one per format)
                if n == 0 || n > 8 { return None; }
                let mut reads = Vec::with_capacity(n);
                for _ in 0..n {
                    let (w, h) = (r.u32()?, r.u32()?);
                    let header = FlatHeader::unpack(r.take(FLAT_HEADER_BYTES)?.try_into().ok()?).ok()?;
                    let grid = FlatGrid::shared(w, h, header.format)?;
                    let mut m = [0f32; 9];
                    for v in &mut m { *v = r.f32()?; }
                    let decode_ms = r.f64()?;
                    let (cols, rows, alpha, mix, edge_px) = (r.u32()?, r.u32()?, r.f32()?, r.f32()?, r.f32()?);
                    let values = r.bytes()?;
                    let nm = r.len(2)?;
                    let margins: Vec<u16> = r.take(nm * 2)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                    let block_tag = r.bytes()?;
                    // the stitcher reads every data cell of the grid from these
                    let cells = cols as usize * rows as usize;
                    // a read the blend pre-check stopped before its cells carries none
                    let skipped = values.is_empty() && margins.is_empty();
                    if (cols, rows) != (grid.cols(), grid.rows()) || (!skipped && (values.len() != cells || margins.len() != cells)) { return None; }
                    reads.push(FlatRead { grid, header, homography: Homography { m }, decode_ms,
                        reads: FlatReads { cols, rows, values, margins, shifts: Vec::new(), alpha, block_tag, mix, edge_px } });
                }
                FrameRead::Read { reads, sharpness }
            }
            _ => return None,
        };
        r.0.is_empty().then_some(out)
    }
}

pub struct RxSession {
    reader: FrameReader, stitcher: Stitcher, rx: Receiver,
    frames: u64, decoded: u64, failed: u64, last_error: Option<String>,
    format: Option<String>, edges: Option<[f32; 4]>, pitch: u8,
    /// the last read's tile corners in the camera picture (top-left, top-right, bottom-right, bottom-left), and
    /// the codewords of the last frame it stitched (corrected, all): where the square is and how well it reads
    corners: Option<[f32; 8]>, codewords: (usize, usize),
    /// data frames in a row stitched with under a fifth of their codewords (`POOR_FRAMES_RESET` of them clear the
    /// reader's geometry), and whether readers elsewhere (the browser's worker pool) should clear theirs too
    poor_frames: u32, reanchors: u32,
    /// the last data read's format reads from luma (the tracked luma path is where a geometry can trap the reader)
    luma: bool,
    /// the sharpness gate's score of the last capture that showed data only (the manifest scores higher)
    sharpness: f32,
    /// the last data capture's blend and border edge width (`FlatReads::mix`, `FlatReads::edge_px`)
    mix: f32, edge_px: f32,
    /// captures that showed data alone (their codewords, blend and edge width are the status's): a receiver tells a
    /// new data read from a manifest read by it
    data_reads: u64,
    t_first: Option<f64>, t_last: f64, complete: bool, filename: Option<String>, receiver_error: Option<String>,
    /// the sender's frame rate: the decoded frames' numbers against the pushes' `now_ms` (as of the latest
    /// push, decoded or not)
    rate: RateEstimate,
    /// Tune, while it runs (`start_tune`): data reads go to the meter only, manifests to the receiver as before
    tune: Option<TuneMeter>,
    /// the live hint over this session's statuses (not while tuning), and the hint it shows
    watch: HintWatch, hint: Option<LiveHint>,
}

impl Default for RxSession { fn default() -> Self { Self::new() } }

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() { match c { '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"), c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)), c => o.push(c) } }
    o.push('"');
    o
}
fn json_opt(s: &Option<String>) -> String { s.as_deref().map(json_str).unwrap_or_else(|| "null".into()) }

/// Camera px per screen px along each tile edge (top, right, bottom, left): the framing guidance compares the
/// smallest, times the cell pitch, with the reader's floor (~2.45 camera px per cell for b/w).
pub(crate) fn edge_sampling(fr: &FlatRead) -> [f32; 4] {
    let (w, h) = (fr.grid.width_px() as f32, fr.grid.height_px() as f32);
    let hm = &fr.homography;
    let along = |x0: f32, y0: f32, x1: f32, y1: f32| { let (a, b) = (hm.apply(x0, y0), hm.apply(x1, y1)); (a.0 - b.0).hypot(a.1 - b.1) / (x1 - x0).hypot(y1 - y0) };
    [along(0.0, 0.0, w, 0.0), along(w, 0.0, w, h), along(0.0, h, w, h), along(0.0, 0.0, 0.0, h)]
}

/// The tile's corners in the camera picture: top-left, top-right, bottom-right, bottom-left, as x, y pairs.
pub(crate) fn tile_corners(fr: &FlatRead) -> [f32; 8] {
    let (w, h) = (fr.grid.width_px() as f32, fr.grid.height_px() as f32);
    let mut c = [0.0; 8];
    for (i, (x, y)) in [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)].into_iter().enumerate() { (c[2 * i], c[2 * i + 1]) = fr.homography.apply(x, y); }
    c
}

impl RxSession {
    pub fn new() -> RxSession {
        RxSession { reader: FrameReader::new(), stitcher: Stitcher::new(), rx: Receiver::in_memory(),
            frames: 0, decoded: 0, failed: 0, last_error: None, format: None, edges: None, pitch: 0, corners: None, codewords: (0, 0), poor_frames: 0, reanchors: 0, luma: false, sharpness: 0.0, mix: 0.0, edge_px: 0.0, data_reads: 0,
            t_first: None, t_last: 0.0, complete: false, filename: None, receiver_error: None, rate: RateEstimate::new(), tune: None, watch: HintWatch::new(), hint: None }
    }
    /// A session whose stream and file live in `stores` (the browser's OPFS): large files never sit in memory.
    pub fn with_stores(stores: Box<dyn StoreFactory>) -> RxSession { let mut s = RxSession::new(); s.rx = Receiver::with_stores(stores); s }
    /// (shown file name, store name) of a store-backed session's verified file.
    pub fn stored_file(&self) -> Option<(String, String)> { self.rx.stored_file() }
    pub fn is_complete(&self) -> bool { self.complete }
    /// The verified file (name, bytes), once.
    pub fn take_file(&mut self) -> Option<(String, Vec<u8>)> { self.rx.take_file() }
    /// One camera frame, RGBA as `getImageData` gives it, without a clock (no rate in the status).
    pub fn push_rgba(&mut self, rgba: &[u8], width: u32, height: u32) -> String { self.push_rgba_at(rgba, width, height, f64::NAN) }
    /// One camera frame at `now_ms` (any monotonic clock, e.g. `performance.now()`): returns the status JSON.
    pub fn push_rgba_at(&mut self, rgba: &[u8], width: u32, height: u32, now_ms: f64) -> String {
        if self.complete {
            // nothing more is read; a short buffer is still named
            if rgba.len() < width as usize * height as usize * 4 { self.last_error = Some("BadFrame".into()); }
            self.accept(FrameRead::NoFrame, now_ms, false);
        } else {
            let read = self.reader.read_rgba(rgba, width, height);
            self.accept(read, now_ms, false);
        }
        self.status_json()
    }
    /// One camera frame as I420 planes in one buffer (the browser's `VideoFrame.copyTo` layout): `layout` is
    /// [y offset, y stride, u offset, u stride, v offset, v stride] (U and V share a stride). Black/white frames are read from the Y plane
    /// alone (no RGBA readback); colour frames (the manifest) convert the planes only when they need it.
    pub fn push_i420_at(&mut self, buf: &[u8], width: u32, height: u32, layout: [usize; 6], now_ms: f64) -> String {
        let read = if self.complete { FrameRead::NoFrame } else { self.reader.read_i420(buf, width, height, layout) };
        self.accept(read, now_ms, false);
        self.status_json()
    }
    /// A reader's record (`FrameRead::to_bytes`, made by a `FrameReader` in another worker) of a frame taken at
    /// `now_ms`: the same status `push_i420_at` gives for that frame. Records may come out of order: one taken before
    /// the newest record so far adds its counts and packets, and leaves the last frame's fields (format, edges,
    /// corners, codewords, sharpness, blend, last miss) and the clock to the newer one. A record that is not one is
    /// counted as a frame and named in the last miss ("BadRecord").
    pub fn push_read(&mut self, record: &[u8], now_ms: f64) -> String {
        let late = now_ms < self.t_last;
        match FrameRead::from_bytes(record) {
            Some(read) => self.accept(read, now_ms, late),
            None => { self.accept(FrameRead::NoFrame, now_ms, late); self.last_error = Some("BadRecord".into()); }
        }
        self.status_json()
    }
    /// One frame's read into the session: counted, then (unless the file is complete) its failure, or its frames
    /// stitched and fed. `late`: a frame older than the newest taken, whose last-frame fields are not shown.
    fn accept(&mut self, read: FrameRead, now_ms: f64, late: bool) {
        // the profile timer exists only when asked for: Instant::now panics in browser WebAssembly
        let t0 = std::env::var_os("LB_PROFILE").map(|_| std::time::Instant::now());
        let mark = |what: &str| { if let Some(t0) = t0 { eprintln!("  [session] {:>7.1} ms  {}", t0.elapsed().as_secs_f64() * 1000.0, what); } };
        self.frames += 1;
        if !now_ms.is_nan() && !late { self.t_last = now_ms; }
        if self.complete { return; }
        match read {
            FrameRead::NoFrame => {}
            FrameRead::BadFrame => { if !late { self.last_error = Some("BadFrame".into()); } }
            FrameRead::Failed(e) => {
                self.failed += 1;
                if !late { self.last_error = Some(e); }
                let t = self.clock(now_ms);
                if let Some(m) = self.tune.as_mut() { m.capture(TuneSample { t_ms: t, read: None }); }
            }
            FrameRead::Read { reads: frs, sharpness } => {
                self.decoded += 1;
                if !late { self.last_error = None; }
                // the first decoded frame starts the clock (a late one, taken before it, starts it earlier)
                if !now_ms.is_nan() && (self.t_first.is_none() || late) { self.t_first = Some(self.t_first.map_or(now_ms, |t| t.min(now_ms))); }
                self.rate.add(frs[0].header.frame_seq, now_ms);
                // how the data reads (format, pitch, codewords, sharpness) comes from DATA reads only: the manifest (12 px
                // colour, one frame in 31) always reads large and whole. Where the square is (corners, edges: camera px
                // per screen px) comes from every read: the manifest is the same tile, so it is as good and fresher.
                let data = frs.iter().find(|r| r.header.frame_type == FrameType::Data);
                if !late {
                    if let Some(d) = data {
                        let f = d.header.format;
                        self.format = Some(format!("{}px {} {}x{}", f.pitch, f.alphabet.name(), d.grid.width_px(), d.grid.height_px()));
                        self.pitch = f.pitch;
                        self.luma = f.alphabet.is_luma();
                    }
                    let geo = data.unwrap_or(&frs[0]);
                    self.edges = Some(edge_sampling(geo));
                    self.corners = Some(tile_corners(geo));
                }
                mark("cells read");
                // a capture that showed data only (a straddled one may end on the manifest's codewords)
                let data_only = data.is_some() && frs.iter().all(|r| r.header.frame_type == FrameType::Data);
                let (mix, edge_px) = (frs[0].reads.mix, frs[0].reads.edge_px);
                // Tune: the data reads are measured, not received; manifests go on to the receiver (a tuning sender is
                // named in the status, and a normal sender's manifest is in once Tune stops)
                let t = self.clock(now_ms);
                let frs: Vec<FlatRead> = match self.tune.as_mut() {
                    Some(m) => { crate::tune::meter::feed_reads(m, &frs, t); frs.into_iter().filter(|r| r.header.frame_type == FrameType::Manifest).collect() }
                    None => frs,
                };
                let frames = if frs.is_empty() { Vec::new() } else { self.stitcher.add_capture(&frs) };
                mark("payload (stitch, RS)");
                if data_only {
                    if !late {
                        if self.tune.is_none() { self.codewords = self.stitcher.last_codewords(); }
                        // a frame that reads under a fifth of its codewords while the tile is seen: twice in a row and
                        // the geometry is dropped (phone, 2026-11-02: 25-35 s of reads at 2-7 of 96-111 codewords,
                        // clean captures, sharp edges, margins confident: the tracked field had gone wrong)
                        // a blended capture (two sender frames in one picture) reads few codewords for its own reason:
                        // only a clean capture counts (the soft set's blends measure 0.49-0.59, clean ones 0.00-0.03)
                        let (ok, all) = self.codewords;
                        // (colour frames read partial codewords by nature: the net watches the luma path only)
                        self.poor_frames = if self.luma && all > 0 && ok * 5 < all && self.mix < 0.15 { self.poor_frames + 1 } else { 0 };
                        if self.poor_frames >= POOR_FRAMES_RESET { self.poor_frames = 0; self.reader.clear_hint(); self.reanchors += 1; }
                        self.sharpness = sharpness;
                        (self.mix, self.edge_px) = (mix, edge_px);
                    }
                    self.data_reads += 1;
                }
                for (header, fp) in frames {
                    match self.rx.feed_packets(&header, &unpack_frame(&fp)) {
                        Ok(FeedOutcome::Complete(p)) => { self.complete = true; self.filename = p.file_name().map(|s| s.to_string_lossy().into_owned()); break; }
                        Ok(_) => {}
                        Err(e) => self.receiver_error = Some(e.to_string()),
                    }
                }
            }
        }
        mark("packets fed");
        // the live hint watches normal transfers, on frames in time order
        if self.tune.is_none() && !late {
            let (received, stream_size) = self.received();
            let cell_px = self.edges.map_or(0.0, |e| e.iter().copied().fold(f32::MAX, f32::min) as f64 * self.pitch as f64);
            let hs = HintSample { t_ms: self.clock(now_ms), decoded: self.decoded, failed: self.failed, format: self.format.as_deref().and_then(crate::tune::hint::parse_format),
                cell_px, codewords_ok: self.codewords.0 as u64, codewords: self.codewords.1 as u64, mix: self.mix as f64, received_bytes: received, stream_size,
                complete: self.complete, sender_fps: self.rate.fps_at(self.t_last).unwrap_or(0.0) };
            self.hint = self.watch.update(&hs).cloned();
        }
    }
    /// (bytes in so far, stream size) as the status says them.
    fn received(&self) -> (u64, u64) {
        let p = self.rx.progress();
        let received = if p.stream_size > 0 { (p.packets_new * crate::fountain::BLOCK_BYTES as u64).min(p.stream_size).max(p.bytes_done) } else { 0 };
        (if self.complete { p.stream_size.max(received) } else { received }, p.stream_size)
    }

    /// Starts Tune at `now_ms` (the pushes' clock): from the next frame on, data reads are measured (`TuneMeter`),
    /// not received; the status's "tune" says where it stands. Starting again starts over.
    pub fn start_tune(&mut self, opts: TuneOptions, now_ms: f64) { self.tune = Some(TuneMeter::new(opts, self.clock(now_ms))); self.watch = HintWatch::new(); self.hint = None; }
    /// Back to receiving ("tune" null).
    pub fn stop_tune(&mut self) { self.tune = None; }
    /// The clock of a push: its `now_ms`, or (pushed without one) the frame count at 30 fps.
    fn clock(&self, now_ms: f64) -> f64 { if now_ms.is_nan() { self.frames as f64 * 1000.0 / 30.0 } else { now_ms } }

    /// Files `offset .. offset + count` of the batch being received (`live::batch_page_json`).
    pub fn batch_page(&self, offset: u32, count: u32) -> String { super::live::batch_page_json(&self.rx, offset, count) }
    /// Files `offset .. offset + count` of the batch's list, as `Receiver::batch_files` gives them (no JSON).
    pub fn batch_files(&self, offset: usize, count: usize) -> Vec<crate::file::FileState> { self.rx.batch_files(offset, count) }
    /// File `i` of the batch an in-memory session saved (its name and bytes), once (`Receiver::take_batch_file`).
    pub fn take_batch_file(&mut self, i: usize) -> Option<(String, Vec<u8>)> { self.rx.take_batch_file(i) }

    pub fn status_json(&self) -> String {
        let p = self.rx.progress();
        let (done, _) = self.received();
        // every number finite (`inf` and `NaN` are not JSON): a non-finite one goes out as 0
        let f = |v: f32| fin(v as f64);
        let seconds = fin(match self.t_first { Some(t0) if self.t_last > t0 => (self.t_last - t0) / 1000.0, _ => 0.0 });
        let mbit = fin(if seconds > 0.0 { done as f64 * 8.0 / seconds / 1e6 } else { 0.0 });
        let edges = self.edges.map(|e| format!("[{:.2},{:.2},{:.2},{:.2}]", f(e[0]), f(e[1]), f(e[2]), f(e[3]))).unwrap_or_else(|| "null".into());
        let cell_px = self.edges.map(|e| f(e.iter().copied().fold(f32::MAX, f32::min) * self.pitch as f32)).unwrap_or(0.0);
        let corners = self.corners.map(|c| format!("[{}]", c.iter().map(|&v| format!("{:.1}", f(v))).collect::<Vec<_>>().join(","))).unwrap_or_else(|| "null".into());
        let (cw_ok, cw) = self.codewords;
        let sharp = f(self.sharpness);
        format!("{{\"frames\":{},\"reanchors\":{},\"decoded\":{},\"failed\":{},\"last_error\":{},\"format\":{},\"edges\":{edges},\"cell_px\":{cell_px:.2},\"have_manifest\":{},\"filename\":{},\"received_bytes\":{done},\"stream_size\":{},\"disk_peak\":{},\"seconds\":{seconds:.3},\"mbit_per_s\":{mbit:.3},\"sender_fps\":{:.1},\"complete\":{},\"receiver_error\":{},\"finish_words\":{},\"finish_parts\":{},\"codewords_ok\":{cw_ok},\"codewords\":{cw},\"sharpness\":{sharp:.2},\"mix\":{:.2},\"edge_px\":{:.2},\"data_reads\":{},\"corners\":{corners},\"batch\":{},\"sha256\":{},\"tune_sender\":{},\"tune\":{},\"hint\":{}}}",
            self.frames, self.reanchors, self.decoded, self.failed, json_opt(&self.last_error), json_opt(&self.format), p.have_manifest,
            json_opt(&p.filename.clone().or_else(|| self.filename.clone())), p.stream_size, p.disk_peak, fin(self.rate.fps_at(self.t_last).unwrap_or(0.0)), self.complete, json_opt(&self.receiver_error), json_opt(&self.rx.finish_words()), json_opt(&self.rx.finish_parts()), f(self.mix), f(self.edge_px), self.data_reads, super::live::batch_json(self.rx.batch_counts().as_ref()),
            self.rx.file_sha256().map_or_else(|| "null".into(), |h| format!("\"{}\"", super::live::hex(&h))), p.tune_sender.is_some(),
            self.tune.as_ref().map_or_else(|| "null".into(), |m| m.status_json(self.t_last)),
            if self.tune.is_some() { "null".into() } else { self.hint.as_ref().map_or_else(|| "null".into(), |h| h.to_json()) })
    }
}
