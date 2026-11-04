//! Camera image -> identified frame(s): markers, chrome homography, header copies, cells.

use std::sync::atomic::{AtomicU32, Ordering};
use super::extract::{extract_payload, FramePayload};
use super::header::{FlatHeader, HeaderCopy};
use super::layout::{Chrome, FlatGrid};
use super::read::{read_cells, read_header_copies, read_header_copies_luma, FlatReads, ReadParams};
use super::format::MODULE_PX;
use crate::detect::{find_marker_candidates, gate_sharpness, refine_homography, two_means_threshold_sampled, Homography, MarkerSet, TimingBorder};
use crate::image::{GrayImage, RgbImage};

#[derive(Debug, PartialEq)]
pub enum DecodeError { TooBlurry(f32), NoMarkers, NoHeader,
    /// A LensBolt header read whole (RS, CRC, magic) that names another format version (its version field): the
    /// sender and this receiver are different LensBolt versions. Never read further: the versions' packets
    /// differ (format 9: the repair packets of small chunks).
    OtherVersion(u8),
    /// The luma path found a frame of a colour alphabet (the 12 px c8 manifest, colour data): read it with
    /// the RGB path. With where it found it (`ColourAt`): the RGB read starts there (`decode_luma_first`).
    NeedsColour(FlatHeader, Option<Box<ColourAt>>) }

/// Where the luma path found a colour frame: its chrome, geometry and both header copies.
#[derive(Debug, PartialEq)]
pub struct ColourAt { pub chrome: Chrome, pub homography: Homography, pub headers: [Option<FlatHeader>; 2] }

/// Tile widths and heights a sender may draw (every pair with a chrome is tried; the standard tile first).
pub const FRAME_WIDTHS: [u32; 6] = [1024, 1280, 1536, 1920, 2048, 2560];
pub const FRAME_HEIGHTS: [u32; 7] = [1024, 768, 896, 640, 1080, 1152, 1440];
/// Kept for callers that enumerate sizes: the standard tile and the common full screens.
pub const FRAME_SIZES: [(u32, u32); 6] = [(1024, 1024), (1024, 768), (1024, 896), (1920, 1080), (2560, 1440), (1280, 720)];
/// Tiles matched to a camera's frame (16:9, about camera width / 2.45 cells across at 1 px): a Full HD camera
/// fills with ~784 x 440, 720p with ~520 x 296 would be below the header's minimum width.
pub const CAMERA_MATCHED_SIZES: [(u32, u32); 6] = [(784, 440), (768, 432), (800, 448), (832, 464), (896, 504), (640, 360)];
fn all_sizes() -> Vec<(u32, u32)> {
    let mut v: Vec<(u32, u32)> = FRAME_SIZES.to_vec();
    v.extend_from_slice(&CAMERA_MATCHED_SIZES);
    for &w in &FRAME_WIDTHS { for &h in &FRAME_HEIGHTS { if !v.contains(&(w, h)) { v.push((w, h)); } } }
    v
}
use std::sync::Arc;

pub struct FlatRead { pub grid: Arc<FlatGrid>, pub header: FlatHeader, pub homography: Homography, pub reads: FlatReads, pub decode_ms: f64 }
impl FlatRead {
    pub fn hint(&self) -> FlatHint { FlatHint { chrome: *self.grid.chrome(), homography: self.homography, shifts: Some(Arc::new(self.reads.shifts.clone())) } }
    pub fn payload(&self) -> FramePayload { extract_payload(&self.reads, &self.grid) }
}

/// Geometry of the previous capture: the screen barely moves between video frames.
#[derive(Clone, Debug)]
pub struct FlatHint { pub chrome: Chrome, pub homography: Homography,
    /// the read's per-block sampling shifts: the luma reader's starting field for the next capture of the
    /// same grid (measured on consecutive real 1 px captures: same cell errors, 25% less read time)
    pub shifts: Option<Arc<Vec<(f32, f32)>>> }

/// `gate_score`: the sharpness gate's score of the latest frame it scored (f32 bits), for a status line (the
/// browser receiver refocuses on it); with several threads decoding at once it is whichever scored last.
pub struct FlatDecoder { pub sizes: Vec<(u32, u32)>, pub min_sharpness: f32, pub params: ReadParams, gate_score: AtomicU32,
    /// tracked reads left without the alignment check's re-read, after two re-reads in a row that lost (the check
    /// fired on something the field was not to blame for: glare, a finger): `REREAD_BACKOFF` captures; and the
    /// re-reads lost in a row so far
    reread_backoff: AtomicU32, rereads_lost: AtomicU32,
    /// cold identifications run so far (the full marker search: the costly path; `cold_identifications`), and
    /// tracked captures in a row whose header failed under a geometry that checked out
    cold_ids: AtomicU32, tracked_misses: AtomicU32 }

/// Captures a tracked read goes without the re-read from scratch after two in a row that lost. One lost re-read
/// is not enough: a capture whose bottom shows the manifest loses its re-read, and the next capture needs one
/// (measured on the 60 soft captures: backing off after one lost re-read read 289 of 6456 codewords tracked,
/// against 330).
pub const REREAD_BACKOFF: u32 = 5;

impl Default for FlatDecoder { fn default() -> Self { Self::new() } }

fn now_ms() -> f64 {
    #[cfg(not(target_arch = "wasm32"))]
    { use std::time::{SystemTime, UNIX_EPOCH}; SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64() * 1000.0).unwrap_or(0.0) }
    #[cfg(target_arch = "wasm32")]
    { 0.0 }
}

/// Fraction of the (up to 14) white border modules nearest `anchor` that are clearly brighter than the
/// black module next to them through `h` (by `step_margin` of their steps); None if fewer than 6 are in view. Rejects wrong size/rotation
/// candidates for a few pixel reads.
fn probe_score(gray: &GrayImage, chrome: &Chrome, h: &Homography, anchor: (f32, f32)) -> Option<f32> {
    let n = chrome.border_count();
    let mut cand: Vec<(f32, u32)> = (0..n).filter(|&i| chrome.border_is_white(i) && !chrome.border_is_white((i + 1) % n))
        .map(|i| { let c = chrome.border_center(i); ((c.0 - anchor.0).hypot(c.1 - anchor.1), i) }).collect();
    cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut steps: Vec<f32> = Vec::with_capacity(14);
    for &(_, i) in cand.iter().take(40) {
        let (w, b) = (chrome.border_center(i), chrome.border_center((i + 1) % n));
        let ((wx, wy), (bx, by)) = (h.apply(w.0, w.1), h.apply(b.0, b.1));
        let inside = |x: f32, y: f32| x >= 1.0 && y >= 1.0 && x < gray.width as f32 - 2.0 && y < gray.height as f32 - 2.0;
        if !inside(wx, wy) || !inside(bx, by) { continue; }
        steps.push(gray.sample(wx - 0.5, wy - 0.5) - gray.sample(bx - 0.5, by - 0.5));
        if steps.len() == 14 { break; }
    }
    // a marker whose border runs off the image (the tile touching the frame edge) cannot vote
    if steps.len() < 6 { return None; }
    // clearly brighter: by a margin scaled to this capture's contrast (a dim screen's step is a few levels)
    let margin = crate::detect::refine::step_margin(&mut steps.clone());
    Some(steps.iter().filter(|&&d| d > margin).count() as f32 / steps.len() as f32)
}

/// The four corner markers among the candidate clusters. The 7-module markers with a 3-module centre
/// give many hit rows (about 3 per module of size), while random data cells imitate the run pattern
/// only on a couple of rows, so only clusters near the strongest ones compete for the corners.
fn pick_markers(gray: &GrayImage) -> Option<MarkerSet> {
    pick_from(&strongest_first(gray, None)).or_else(|| pick_from(&strongest_first(gray, Some(FINE_RADIUS))))
}
/// The binarisation radius of the second marker search (camera px), for a small tile lit on a dark screen (a
/// phone sender): the frame's window (1/16 of the frame, 90 px at 1440x1080) is mostly dark screen at the
/// tile's corners, its mean about that of the marker's blurred 1-module ring, so the ring reads white and no
/// marker is found (60 of 60 captures of a phone sender, `tests/phone_sender_tile.rs`). A window of a couple of
/// modules at 2 camera px per cell sees the ring against its own whites. Only when the frame's window finds
/// nothing: a big marker's 3-module centre is wider than this window and breaks up in it.
const FINE_RADIUS: usize = 12;
/// Frames of this many px and more look for markers on a half-size copy first (`identify_core`).
const COARSE_FROM_PX: u64 = 4_000_000;
/// `gray` at half the width and height, each pixel the mean of its 2 x 2 (candidate coordinates and units in it
/// are half those of `gray`: pixel i's centre i + 0.5 is 2i + 1 in `gray`).
fn half_size(gray: &GrayImage) -> GrayImage {
    let (w, h) = (gray.width / 2, gray.height / 2);
    let mut out = GrayImage::new(w, h);
    let src = gray.width as usize;
    crate::par::par_chunks_rows(&mut out.data, w as usize, |y, line| {
        let (a, b) = (&gray.data[2 * y * src..], &gray.data[(2 * y + 1) * src..]);
        for (x, o) in line.iter_mut().enumerate() { *o = ((a[2 * x] as u16 + a[2 * x + 1] as u16 + b[2 * x] as u16 + b[2 * x + 1] as u16 + 2) / 4) as u8; }
    });
    out
}
/// The marker candidates of `gray` (x, y, unit, hits), the strongest first: the frame's binarisation window, or
/// the one of `radius`.
fn strongest_first(gray: &GrayImage, radius: Option<usize>) -> Vec<(f32, f32, f32, u32)> {
    let mut cands = match radius { None => find_marker_candidates(gray), Some(r) => crate::detect::markers::find_marker_candidates_r(gray, r) };
    cands.sort_by(|a, b| b.3.cmp(&a.3));
    cands
}
/// `pick_markers` from candidates already found (strongest first).
fn pick_from(cands: &[(f32, f32, f32, u32)]) -> Option<MarkerSet> {
    if cands.len() < 4 { return None; }
    let floor = (cands[3].3 / 2).max(2);
    let pts: Vec<(f32, f32)> = cands.iter().filter(|c| c.3 >= floor).map(|c| (c.0, c.1)).collect();
    crate::detect::markers::corner_set(&pts)
}

/// What the candidates tried so far showed: one passed the border probe; a whole header of another version.
#[derive(Default)]
struct Seen { probed: bool, other: Option<u8> }

/// Candidates a three-marker search weighs (the strongest clusters).
const THREE_MARKER_CANDIDATES: usize = 4;

/// Corner sets from three markers, for a tile whose fourth corner is out of the picture or covered: from the
/// strongest candidate clusters, every three that span a triangle, each corner of it taken in turn as the one
/// between the other two (the parallelogram's fourth point completes it: a + c - b), clockwise, with which
/// corners were found. The geometry from a completed corner is a guess the border's fit corrects.
fn three_marker_sets(cands: &[(f32, f32, f32, u32)]) -> Vec<([(f32, f32); 4], [bool; 4])> {
    // markers give many hit rows; data cells imitating one give a couple
    let pts: Vec<(f32, f32)> = cands.iter().take(THREE_MARKER_CANDIDATES).filter(|c| c.3 >= 4).map(|c| (c.0, c.1)).collect();
    let mut out = Vec::new();
    let n = pts.len();
    for i in 0..n { for j in i + 1..n { for k in j + 1..n {
        let tri = [pts[i], pts[j], pts[k]];
        let area = ((tri[1].0 - tri[0].0) * (tri[2].1 - tri[0].1) - (tri[2].0 - tri[0].0) * (tri[1].1 - tri[0].1)).abs() / 2.0;
        let side = (0..3).map(|q| { let (a, b) = (tri[q], tri[(q + 1) % 3]); (a.0 - b.0).hypot(a.1 - b.1) }).fold(0f32, f32::max);
        if area < 0.2 * side * side { continue; } // nearly collinear: no tile
        for b in 0..3 {
            let (a, c, mid) = (tri[(b + 1) % 3], tri[(b + 2) % 3], tri[b]);
            let fourth = (a.0 + c.0 - mid.0, a.1 + c.1 - mid.1);
            let Some(set) = crate::detect::markers::corner_set(&[a, mid, c, fourth]) else { continue };
            let m = set.as_array();
            out.push((m, m.map(|p| p != fourth)));
        }
    } } }
    out
}

impl FlatDecoder {
    pub fn new() -> FlatDecoder {
        let sizes = all_sizes().into_iter().filter(|&(w, h)| Chrome::new(w, h).is_some()).collect();
        FlatDecoder { sizes, min_sharpness: 4.0, params: ReadParams::default(), gate_score: AtomicU32::new(0), reread_backoff: AtomicU32::new(0), rereads_lost: AtomicU32::new(0), cold_ids: AtomicU32::new(0), tracked_misses: AtomicU32::new(0) }
    }
    /// Cold identifications (full marker searches) run so far.
    pub fn cold_identifications(&self) -> u32 { self.cold_ids.load(Ordering::Relaxed) }
    /// The first stage of every decode: `TooBlurry` for a frame below `min_sharpness`, else the global
    /// two-means threshold the geometry binarises with. Both from a sparse sample of a large frame
    /// (`gate_sharpness`, `two_means_threshold_sampled`); the full-frame score decides a borderline frame,
    /// and a frame up to 8.4 MP takes its threshold from every pixel.
    pub fn gate(&self, gray: &GrayImage) -> Result<u8, DecodeError> {
        let s = gate_sharpness(gray, self.min_sharpness);
        self.gate_score.store(s.to_bits(), Ordering::Relaxed);
        if s < self.min_sharpness { return Err(DecodeError::TooBlurry(s)); }
        Ok(two_means_threshold_sampled(gray))
    }
    /// The gate's score of the latest frame it scored (0 before the first).
    pub fn last_gate_score(&self) -> f32 { f32::from_bits(self.gate_score.load(Ordering::Relaxed)) }
    /// Sharpness gate -> markers -> (size x rotation) candidates -> border probe -> refine -> header copies.
    pub fn identify(&self, img: &RgbImage) -> Result<(Chrome, Homography, [Option<FlatHeader>; 2]), DecodeError> {
        let gray = img.to_gray();
        let thr = self.gate(&gray)?;
        self.identify_with(img, &gray, thr)
    }
    fn identify_with(&self, img: &RgbImage, gray: &GrayImage, thr: u8) -> Result<(Chrome, Homography, [Option<FlatHeader>; 2]), DecodeError> {
        self.identify_core(gray, thr, &|c: &Chrome, h: &Homography| read_header_copies(img, c, h))
    }
    /// The first (size x rotation) candidate with a header of this version. None: `OtherVersion` when a candidate
    /// read a whole header of another version, else `NoHeader` when one passed the border probe, else `NoMarkers`.
    /// The four corner markers first; when they give nothing, three markers (`three_marker_sets`): a corner out of
    /// the picture or covered.
    fn identify_core(&self, gray: &GrayImage, thr: u8, read_hdr: &dyn Fn(&Chrome, &Homography) -> [HeaderCopy; 2]) -> Result<(Chrome, Homography, [Option<FlatHeader>; 2]), DecodeError> {
        self.cold_ids.fetch_add(1, Ordering::Relaxed);
        let mut seen = Seen::default();
        // a large frame (a full-sensor stream) looks on a half-size copy first: the marker search binarises and
        // scans every pixel (58 of the 85 ms of a 12 MP cold decode), and markers of 8 tile px modules stay
        // several px wide when halved; the full size is searched only when the copy finds nothing that reads
        let small = (gray.width as u64 * gray.height as u64 >= COARSE_FROM_PX).then(|| half_size(gray));
        let searches: Vec<(&GrayImage, f32, Option<usize>)> = match &small {
            Some(sm) => vec![(sm, 2.0, None), (sm, 2.0, Some(FINE_RADIUS)), (gray, 1.0, None), (gray, 1.0, Some(FINE_RADIUS))],
            None => vec![(gray, 1.0, None), (gray, 1.0, Some(FINE_RADIUS))],
        };
        // the frame's binarisation window, then a small one (FINE_RADIUS: a phone sender's tile on a dark screen)
        for (img, scale, radius) in searches {
            let cands: Vec<(f32, f32, f32, u32)> = strongest_first(img, radius).into_iter().map(|c| (c.0 * scale, c.1 * scale, c.2 * scale, c.3)).collect();
            if let Some(markers) = pick_from(&cands) {
                if let Some(r) = self.identify_from(gray, thr, markers.as_array(), [true; 4], read_hdr, &mut seen) { return Ok(r); }
            }
            for (m, real) in three_marker_sets(&cands) {
                if let Some(r) = self.identify_from(gray, thr, m, real, read_hdr, &mut seen) { return Ok(r); }
            }
        }
        Err(match (seen.other, seen.probed) { (Some(v), _) => DecodeError::OtherVersion(v), (None, true) => DecodeError::NoHeader, (None, false) => DecodeError::NoMarkers })
    }
    /// Every (size x rotation) candidate for corner points `m` (clockwise) of which `real[i]` were found (not
    /// completed): the border probe and the refinement use the real ones only, so a completed corner (a guess,
    /// out of the picture or under a thumb) neither vetoes a candidate nor pulls the fit.
    fn identify_from(&self, gray: &GrayImage, thr: u8, m: [(f32, f32); 4], real: [bool; 4], read_hdr: &dyn Fn(&Chrome, &Homography) -> [HeaderCopy; 2], seen: &mut Seen) -> Option<(Chrome, Homography, [Option<FlatHeader>; 2])> {
        let rot = |k: usize| -> ([(f32, f32); 4], [bool; 4]) { (std::array::from_fn(|i| m[(i + k) % 4]), std::array::from_fn(|i| real[(i + k) % 4])) };
        let dbg = std::env::var("LB_DEBUG").is_ok();
        if dbg { eprintln!("flat identify: markers {m:?} (found {real:?})"); }
        for &(w, h) in &self.sizes {
            let Some(chrome) = Chrome::new(w, h) else { continue };
            for k in 0..4 {
                let (pts, found) = rot(k);
                let Some(coarse) = Homography::from_points(chrome.marker_centers(), pts) else { continue };
                // every found marker whose border is in view must show the timing pattern, and at least two must be
                let scores: Vec<f32> = chrome.marker_centers().iter().zip(found).filter(|(_, f)| *f).filter_map(|(&p, _)| probe_score(gray, &chrome, &coarse, p)).collect();
                let score = scores.iter().copied().fold(1.0f32, f32::min);
                if dbg { eprintln!("  size {w}x{h} rot {:?}: probe {score:.2} over {} markers", pts[0], scores.len()); }
                if scores.len() < 2 || score < 0.64 { continue; }
                let pairs: Vec<_> = chrome.marker_centers().iter().zip(pts.iter()).zip(found).filter(|(_, f)| *f).map(|((a, b), _)| (*a, *b)).collect();
                // a completed corner starts far off: the border's fit grows out from the found markers
                let refined = if found.iter().all(|&f| f) { refine_homography(gray, &chrome, &coarse, &pairs, thr) } else { crate::detect::refine::refine_homography_grow(gray, &chrome, &coarse, &pairs, thr) };
                let copies = read_hdr(&chrome, &refined);
                let headers = copies.map(HeaderCopy::header);
                if dbg { eprintln!("    headers {copies:?}"); }
                if headers.iter().any(|h| h.is_some()) { return Some((chrome, refined, headers)); }
                seen.probed = true;
                if seen.other.is_none() { seen.other = copies.iter().find_map(|c| c.other_version()); }
            }
        }
        None
    }
    fn read_with(&self, img: &RgbImage, chrome: &Chrome, h: &Homography, header: FlatHeader, t0: f64) -> Result<FlatRead, DecodeError> {
        let grid = FlatGrid::shared(chrome.width_px(), chrome.height_px(), header.format).ok_or(DecodeError::NoHeader)?;
        if (grid.cols() as u16, grid.rows() as u16) != (header.cols, header.rows) { return Err(DecodeError::NoHeader); }
        let reads = read_cells(img, &grid, h, &ReadParams { tag_hint: header.frame_seq & 3, ..self.params }.tuned_for(header.format.pitch));
        Ok(FlatRead { grid, header, homography: *h, reads, decode_ms: now_ms() - t0 })
    }
    /// Read the cells of `img` for a header the caller already has (a second header copy showing another
    /// frame of another format in a straddled capture), with the geometry of an earlier read.
    pub fn read_frame(&self, img: &RgbImage, chrome: &Chrome, h: &Homography, header: FlatHeader) -> Result<FlatRead, DecodeError> { self.read_with(img, chrome, h, header, now_ms()) }
    /// Full pipeline for the first valid header copy.
    pub fn decode(&self, img: &RgbImage) -> Result<FlatRead, DecodeError> { self.decode_tracked(img, None) }
    /// Every distinct frame in the capture: a rolling shutter can show the top of one displayed frame and
    /// the bottom of the next, with one header copy each. Frames of the same format share one cell read.
    pub fn decode_all(&self, img: &RgbImage) -> Result<Vec<FlatRead>, DecodeError> {
        let t0 = now_ms();
        let (chrome, h, headers) = self.identify(img)?;
        let mut out: Vec<FlatRead> = Vec::new();
        for hdr in headers.into_iter().flatten() {
            if out.iter().any(|r| r.header == hdr) { continue; }
            if let Some(same) = out.iter().find(|r| r.header.format == hdr.format && (r.header.cols, r.header.rows) == (hdr.cols, hdr.rows)) {
                let reads = FlatReads { cols: same.reads.cols, rows: same.reads.rows, values: same.reads.values.clone(), margins: same.reads.margins.clone(), shifts: same.reads.shifts.clone(), alpha: same.reads.alpha, block_tag: same.reads.block_tag.clone(), mix: same.reads.mix, edge_px: same.reads.edge_px };
                out.push(FlatRead { grid: same.grid.clone(), header: hdr, homography: h, reads, decode_ms: now_ms() - t0 });
                continue;
            }
            if let Ok(r) = self.read_with(img, &chrome, &h, hdr, t0) { out.push(r); }
        }
        if out.is_empty() { Err(DecodeError::NoHeader) } else { Ok(out) }
    }
    /// The four corner markers of `chrome`, searched only in windows around where `hint` projects them (live
    /// video: the tile barely moves between captures). In `chrome.marker_centers()` order; None if any window
    /// shows no marker.
    fn markers_near(gray: &GrayImage, chrome: &Chrome, hint: &Homography) -> Option<[(f32, f32); 4]> {
        let centres = chrome.marker_centers();
        let mut out = [(0f32, 0f32); 4];
        for (k, &(mx, my)) in centres.iter().enumerate() {
            let (cx, cy) = hint.apply(mx, my);
            let (ax, ay) = hint.apply(mx + 1.0, my);
            let scale = (ax - cx).hypot(ay - cy);
            // a window three marker sizes wide (a marker is 7 modules)
            let half = (1.5 * 7.0 * MODULE_PX as f32 * scale).max(24.0);
            let (x0, y0) = ((cx - half).max(0.0) as u32, (cy - half).max(0.0) as u32);
            let (x1, y1) = (((cx + half) as u32).min(gray.width), ((cy + half) as u32).min(gray.height));
            if x1 <= x0 + 8 || y1 <= y0 + 8 { return None; }
            let (w, h) = (x1 - x0, y1 - y0);
            let mut win = GrayImage::new(w, h);
            for yy in 0..h { let o = ((y0 + yy) * gray.width + x0) as usize; win.data[(yy * w) as usize..((yy + 1) * w) as usize].copy_from_slice(&gray.data[o..o + w as usize]); }
            // the frame's binarisation window, then one of two marker modules (FINE_RADIUS: a tile on a dark screen)
            let strongest = |r: usize| crate::detect::markers::find_marker_candidates_r(&win, r).into_iter().max_by_key(|c| c.3).filter(|c| c.3 >= 4);
            let best = strongest((gray.width.max(gray.height) as usize / 16).max(8))
                .or_else(|| strongest(((2.0 * MODULE_PX as f32 * scale).round() as usize).max(8)))?;
            out[k] = (best.0 + x0 as f32, best.1 + y0 as f32);
        }
        Some(out)
    }
    /// The luma path (black/white and grey frames, the phone's Y plane): tracked geometry from `hint` (windowed marker
    /// search) with a full search as fallback, headers and b/w cells read from luma. A colour frame returns
    /// `NeedsColour` with its header; a second header copy of a colour format (a straddled manifest) is skipped.
    pub fn decode_luma(&self, y: &GrayImage, hint: Option<&FlatHint>) -> Result<Vec<FlatRead>, DecodeError> {
        let t0 = now_ms();
        let profile = std::env::var_os("LB_PROFILE").is_some();
        let mark = |what: &str| { if profile { eprintln!("  [decode_luma] {:>7.1} ms  {}", now_ms() - t0, what); } };
        let thr = self.gate(y)?;
        mark("gate");
        let rd = |c: &Chrome, h: &Homography| read_header_copies_luma(y, c, h);
        let tracked = hint.and_then(|hint| {
            let chrome = hint.chrome;
            let m = Self::markers_near(y, &chrome, &hint.homography)?;
            mark("  markers near");
            let coarse = Homography::from_points(chrome.marker_centers(), m)?;
            // the same border check a cold identification passes: a window can hold a marker-like cluster of data
            // cells when the real marker left it (the hold turned), and a geometry on such a corner still reads
            // the headers while most cells misread; its own shifts then become the next hint and the reader stays
            // trapped (phone, 2026-11-02: 25 s of reads at 2-7 of 96 codewords between full reads)
            let scores: Vec<f32> = chrome.marker_centers().iter().filter_map(|&p| probe_score(y, &chrome, &coarse, p)).collect();
            if scores.len() < 2 || scores.iter().copied().fold(1.0f32, f32::min) < 0.64 { return None; }
            let pairs: Vec<_> = chrome.marker_centers().iter().zip(m.iter()).map(|(a, b)| (*a, *b)).collect();
            let h = refine_homography(y, &chrome, &coarse, &pairs, thr);
            mark("  refine homography");
            let headers = rd(&chrome, &h).map(HeaderCopy::header);
            // markers where the hint put them and the border checked out: the tile is not lost, so an unreadable
            // header is a damaged one (a capture taken while the sender changed frames: 14 of 60 phone
            // captures). A cold search would find this geometry again and fail the same way, at 85 ms against
            // 31 ms for a read (4 threads)
            if headers.iter().any(|x| x.is_some()) { return Some(Ok((chrome, h, headers))); }
            // the fit through all four can miss the header where a fit from three reads it (a centre found a
            // little off: 4 of 60 phone captures read only through a cold search's three-marker fit): each
            // marker left out in turn, its corner completed (a + c - b) and the border grown from the others
            for k in 0..4 {
                let (a, b, c) = (m[(k + 1) % 4], m[(k + 2) % 4], m[(k + 3) % 4]);
                let mut m3 = m;
                m3[k] = (a.0 + c.0 - b.0, a.1 + c.1 - b.1);
                let Some(coarse) = Homography::from_points(chrome.marker_centers(), m3) else { continue };
                let pairs3: Vec<_> = pairs.iter().enumerate().filter(|(i, _)| *i != k).map(|(_, p)| *p).collect();
                let h3 = crate::detect::refine::refine_homography_grow(y, &chrome, &coarse, &pairs3, thr);
                let headers = rd(&chrome, &h3).map(HeaderCopy::header);
                if headers.iter().any(|x| x.is_some()) { return Some(Ok((chrome, h3, headers))); }
            }
            Some(Err(DecodeError::NoHeader))
        });
        mark(if matches!(tracked, Some(Ok(_))) { "tracked geometry" } else { "tracking skipped or failed" });
        // a tracked miss answers NoHeader at once, but a second in a row searches cold: one damaged capture
        // comes between two good ones; misses in a row may be a tracked geometry gone wrong
        let tracked = match tracked {
            Some(Ok(t)) => { self.tracked_misses.store(0, Ordering::Relaxed); Some(t) }
            Some(Err(e)) => { if self.tracked_misses.fetch_add(1, Ordering::Relaxed) == 0 { return Err(e); } self.tracked_misses.store(0, Ordering::Relaxed); None }
            None => None,
        };
        let (chrome, h, headers) = match tracked { Some(t) => t, None => self.identify_core(y, thr, &rd)? };
        mark("geometry + headers");
        let mut out: Vec<FlatRead> = Vec::new();
        for (i, hdr) in headers.into_iter().flatten().enumerate() {
            if out.iter().any(|r| r.header == hdr) { continue; }
            if !hdr.format.alphabet.is_luma() {
                if i == 0 || out.is_empty() { return Err(DecodeError::NeedsColour(hdr, Some(Box::new(ColourAt { chrome, homography: h, headers })))); }
                continue;
            }
            if let Some(same) = out.iter().find(|r| r.header.format == hdr.format && (r.header.cols, r.header.rows) == (hdr.cols, hdr.rows)) {
                let reads = FlatReads { cols: same.reads.cols, rows: same.reads.rows, values: same.reads.values.clone(), margins: same.reads.margins.clone(), shifts: same.reads.shifts.clone(), alpha: same.reads.alpha, block_tag: same.reads.block_tag.clone(), mix: same.reads.mix, edge_px: same.reads.edge_px };
                out.push(FlatRead { grid: same.grid.clone(), header: hdr, homography: h, reads, decode_ms: now_ms() - t0 });
                continue;
            }
            let grid = FlatGrid::shared(chrome.width_px(), chrome.height_px(), hdr.format).ok_or(DecodeError::NoHeader)?;
            if (grid.cols() as u16, grid.rows() as u16) != (hdr.cols, hdr.rows) { continue; }
            // tracking: the previous read's shift field when it was the same grid (the reader checks the size)
            let prev = hint.filter(|hh| hh.chrome == chrome).and_then(|hh| hh.shifts.as_deref()).map(|v| v.as_slice());
            let back = self.reread_backoff.load(Ordering::Relaxed);
            // the blend pre-check needs the border's edges sharp (a blend keeps them so; soft focus does not):
            // their width is measured before the read, and kept for the status
            let edge_px = super::fast::edge_spread(y, &chrome, &h);
            let sharp = edge_px > 0.0 && edge_px < super::fast::EDGE_SHARP_PX;
            let Some((mut reads, reread)) = super::fast::read_cells_luma_tracked(y, &grid, &h, hdr.frame_seq & 3, prev, back == 0, sharp) else { continue };
            match reread {
                Some(true) => if self.rereads_lost.fetch_add(1, Ordering::Relaxed) + 1 >= 2 {
                    self.rereads_lost.store(0, Ordering::Relaxed);
                    self.reread_backoff.store(REREAD_BACKOFF, Ordering::Relaxed);
                },
                Some(false) => self.rereads_lost.store(0, Ordering::Relaxed),
                None if prev.is_some() && back > 0 => self.reread_backoff.store(back - 1, Ordering::Relaxed),
                None => {}
            }
            reads.edge_px = edge_px;
            out.push(FlatRead { grid, header: hdr, homography: h, reads, decode_ms: now_ms() - t0 });
        }
        if out.is_empty() { Err(DecodeError::NoHeader) } else { Ok(out) }
    }
    /// Live video: refit the previous capture's geometry on the border and confirm it with the header;
    /// fall back to a full identification when that fails.
    pub fn decode_tracked(&self, img: &RgbImage, hint: Option<&FlatHint>) -> Result<FlatRead, DecodeError> {
        let t0 = now_ms();
        let gray = img.to_gray();
        let thr = self.gate(&gray)?;
        if let Some(hint) = hint {
            // the hint says which chrome size and which corner order; the geometry itself is re-anchored on
            // the markers of THIS capture (a hint refined frame after frame drifts: measured on a phone burst,
            // the same frames read 73/80 cold and 13/80 tracked). Without markers in view, refine the hint.
            let chrome = hint.chrome;
            let anchored = pick_markers(&gray).and_then(|m| {
                let m = m.as_array();
                let proj = chrome.marker_centers().map(|p| hint.homography.apply(p.0, p.1));
                let rotations = [[m[0], m[1], m[2], m[3]], [m[1], m[2], m[3], m[0]], [m[2], m[3], m[0], m[1]], [m[3], m[0], m[1], m[2]]];
                let rot = rotations.iter().min_by(|a, b| {
                    let d = |r: &[(f32, f32); 4]| r.iter().zip(proj.iter()).map(|(p, q)| (p.0 - q.0).hypot(p.1 - q.1)).sum::<f32>();
                    d(a).partial_cmp(&d(b)).unwrap()
                })?;
                let coarse = Homography::from_points(chrome.marker_centers(), *rot)?;
                let pairs: Vec<_> = chrome.marker_centers().iter().zip(rot.iter()).map(|(a, b)| (*a, *b)).collect();
                Some(refine_homography(&gray, &chrome, &coarse, &pairs, thr))
            });
            let h = anchored.unwrap_or_else(|| refine_homography(&gray, &chrome, &hint.homography, &[], thr));
            let headers = read_header_copies(img, &chrome, &h).map(HeaderCopy::header);
            if let Some(hdr) = headers.into_iter().flatten().next() {
                if let Ok(r) = self.read_with(img, &chrome, &h, hdr, t0) { return Ok(r); }
            }
        }
        let (chrome, h, headers) = self.identify_with(img, &gray, thr)?;
        let hdr = headers.into_iter().flatten().next().ok_or(DecodeError::NoHeader)?;
        self.read_with(img, &chrome, &h, hdr, t0)
    }
}
