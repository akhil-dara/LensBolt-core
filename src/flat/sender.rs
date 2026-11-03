//! Flat-format sender: one data configuration per transfer, manifests on the coarsest format, the lab
//! calibration carousel, and the Tune carousel (`FlatSender::tune`, `tune::schedule`).

use super::compose::{compose, FlatFrame};
use super::format::{Alphabet, CellFormat, MANIFEST_FORMAT};
use super::header::FlatHeader;
use super::layout::FlatGrid;
use crate::file::{content_transfer_id, BatchSource, Index, Manifest, PacketSource, Prepared};
use crate::finish::FinishError;
use crate::header::FrameType;
use crate::packet::{pack_frame, Packet, PACKET_PAYLOAD};
use std::io;
use std::path::Path;
use std::sync::Arc;

/// Configurations the calibration carousel cycles through (densest first); the receiver measures each
/// and the two words it shows pick one.
pub const CAROUSEL: [CellFormat; 18] = [
    CellFormat { pitch: 1, alphabet: Alphabet::Bw }, CellFormat { pitch: 2, alphabet: Alphabet::Bw }, CellFormat { pitch: 2, alphabet: Alphabet::C16 }, CellFormat { pitch: 2, alphabet: Alphabet::C8 }, CellFormat { pitch: 2, alphabet: Alphabet::G8 }, CellFormat { pitch: 3, alphabet: Alphabet::G8 },
    CellFormat { pitch: 3, alphabet: Alphabet::C16 }, CellFormat { pitch: 3, alphabet: Alphabet::C8 },
    CellFormat { pitch: 4, alphabet: Alphabet::C16 }, CellFormat { pitch: 4, alphabet: Alphabet::C8 },
    CellFormat { pitch: 5, alphabet: Alphabet::C16 }, CellFormat { pitch: 5, alphabet: Alphabet::C8 },
    CellFormat { pitch: 6, alphabet: Alphabet::C8 }, CellFormat { pitch: 8, alphabet: Alphabet::C8 }, CellFormat { pitch: 12, alphabet: Alphabet::C8 },
    CellFormat { pitch: 5, alphabet: Alphabet::Bw }, CellFormat { pitch: 6, alphabet: Alphabet::Bw }, CellFormat { pitch: 8, alphabet: Alphabet::Bw },
];

/// Frames from one manifest frame to the next, by default (every sender: the CLI, the browser page). Odd: a
/// camera phase-locked at exactly half the sender's rate sees only every other shown frame, and with an even
/// period the one that sees the odd frames never saw a manifest (it never completed); with an odd one both
/// phases see a manifest every other period (tests/completion.rs).
pub const MANIFEST_EVERY_FRAMES: u32 = 31;

/// `transfer_id` 0 (the default) takes the id from the file (`content_transfer_id`): the same file under the
/// same name always gets the same id, so finish words and resume work across restarts. `repair_offset`:
/// where finish-word repair rounds start; None draws it from the clock (WebAssembly has none: the page
/// passes its own).
#[derive(Clone, Debug)]
pub struct FlatSenderConfig { pub width: u32, pub height: u32, pub format: CellFormat, pub manifest_every_frames: u32, pub transfer_id: u16, pub calibration: bool, pub repair_offset: Option<u32> }
impl Default for FlatSenderConfig {
    fn default() -> Self {
        FlatSenderConfig { width: 1024, height: 1024, format: CellFormat { pitch: 4, alphabet: Alphabet::C8 }, manifest_every_frames: MANIFEST_EVERY_FRAMES, transfer_id: 0, calibration: false, repair_offset: None }
    }
}

/// `held`: manifest frames shown again while the page holds its first frame (`hold_manifest_frame`); they
/// take frame numbers but no place in the schedule, so later numbers are the schedule's plus `held`.
/// `tune`: Some(display_60) for the Tune carousel, whose data frames take the format their frame number names
/// (`tune::schedule::slot`; the grids are in `carousel`, one per format); `frame_fps` the shown rate of the frame
/// last returned (a tune sender's).
pub struct FlatSender { src: PacketSource, cfg: FlatSenderConfig, grid: Arc<FlatGrid>, manifest_grid: Arc<FlatGrid>, carousel: Vec<Arc<FlatGrid>>, frames: u64, held: u64, tune: Option<bool>, frame_fps: u32 }

impl FlatSender {
    pub fn new(path: &Path, cfg: FlatSenderConfig, force_compress: Option<bool>) -> io::Result<FlatSender> {
        FlatSender::from_prepared(Prepared::new(path, force_compress)?, cfg)
    }
    pub fn from_bytes(filename: &str, data: Vec<u8>, cfg: FlatSenderConfig) -> io::Result<FlatSender> {
        FlatSender::from_prepared(Prepared::from_bytes(filename, data)?, cfg)
    }
    /// A file in memory shared with the caller (`Prepared::from_shared`): the same frames as `from_bytes`, no copy.
    pub fn from_shared(filename: &str, data: Arc<Vec<u8>>, cfg: FlatSenderConfig) -> io::Result<FlatSender> {
        FlatSender::from_prepared(Prepared::from_shared(filename, data)?, cfg)
    }
    /// `from_shared` with the file's hashes computed as it was read (`Prepared::from_shared_hashed`): not read again.
    pub fn from_shared_hashed(filename: &str, data: Arc<Vec<u8>>, hashes: crate::file::FileHashes, cfg: FlatSenderConfig) -> io::Result<FlatSender> {
        FlatSender::from_prepared(Prepared::from_shared_hashed(filename, data, hashes)?, cfg)
    }
    /// Several files as one transfer (`Prepared::batch`: the file index, then the files, smallest first). The
    /// transfer id, unless `cfg` sets one, comes from the batch's manifest, which carries the index's SHA-256
    /// and the batch name: the same batch sent again resumes under the same id.
    pub fn from_batch(name: &str, files: Vec<(String, BatchSource)>, cfg: FlatSenderConfig) -> io::Result<FlatSender> {
        FlatSender::from_prepared(Prepared::batch(name, files)?, cfg)
    }
    /// The Tune carousel: the generated tune payload (`Prepared::tune`), each data frame in the format its frame
    /// number names (`tune::schedule::slot`), manifests every `cfg.manifest_every_frames` as for a file. `cfg`'s
    /// width, height, repair offset and manifest period are used; its format and calibration are not.
    /// `display_60`: the page measured a 60 or 120 Hz screen (the 60 fps segments are drawn at 60, else at 30).
    pub fn tune(cfg: FlatSenderConfig, display_60: bool) -> io::Result<FlatSender> {
        use crate::tune::schedule::SEGMENTS;
        let cfg = FlatSenderConfig { format: SEGMENTS[0].format, calibration: false, transfer_id: 0, ..cfg };
        let mut s = FlatSender::from_prepared(Prepared::tune(display_60)?, cfg)?;
        let mut grids: Vec<Arc<FlatGrid>> = Vec::new();
        for seg in SEGMENTS {
            if grids.iter().any(|g| g.format() == seg.format) { continue; }
            let g = FlatGrid::try_new(s.cfg.width, s.cfg.height, seg.format).map(Arc::new)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("{}x{} cannot hold the Tune carousel's {:?} frames", s.cfg.width, s.cfg.height, seg.format)))?;
            grids.push(g);
        }
        s.carousel = grids;
        s.tune = Some(display_60);
        Ok(s)
    }
    /// The rate the frame last returned (`next_frame`, `next_packets`, `hold_manifest_frame`) is shown at, for a
    /// tune sender: its segment's (`tune::schedule::shown_fps`), 30 for held frames. None for any other sender.
    pub fn tune_frame_fps(&self) -> Option<u32> { self.tune.map(|_| self.frame_fps) }
    fn from_prepared(prepared: Prepared, mut cfg: FlatSenderConfig) -> io::Result<FlatSender> {
        if cfg.manifest_every_frames == 0 { cfg.manifest_every_frames = MANIFEST_EVERY_FRAMES; }
        if cfg.manifest_every_frames < 2 { cfg.manifest_every_frames = 2; } // every frame a manifest would never send data
        if cfg.transfer_id == 0 { cfg.transfer_id = content_transfer_id(&prepared.manifest); }
        let grid_of = |f: CellFormat| FlatGrid::try_new(cfg.width, cfg.height, f).map(Arc::new)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("{}x{} cannot hold a {:?} frame (sides must be multiples of 8, >= 256 px, with room for one stripe)", cfg.width, cfg.height, f)));
        let grid = grid_of(cfg.format)?;
        // 12 px / 8 colours where it fits; the small camera-matched tiles take 8 or 6 px
        let manifest_grid = grid_of(super::format::manifest_format_for(cfg.width, cfg.height).unwrap_or(MANIFEST_FORMAT))?;
        let carousel = if cfg.calibration { CAROUSEL.iter().filter_map(|&f| FlatGrid::try_new(cfg.width, cfg.height, f).map(Arc::new)).collect() } else { Vec::new() };
        let mut src = PacketSource::new(prepared)?;
        if let Some(offset) = cfg.repair_offset { src.set_repair_offset(offset as u64); }
        // a batch's file list goes out again about once per manifest period, so it is timed in frames of this tile
        src.set_packets_per_frame(grid.packets());
        Ok(FlatSender { src, cfg, grid, manifest_grid, carousel, frames: 0, held: 0, tune: None, frame_fps: 30 })
    }
    pub fn manifest(&self) -> &Manifest { self.src.manifest() }
    /// A batch's file index as sent (names as the receiver writes them, offsets in the stream); None for one file.
    pub fn batch_index(&self) -> Option<&Index> { self.src.batch_index() }
    pub fn config(&self) -> &FlatSenderConfig { &self.cfg }
    pub fn frames_emitted(&self) -> u64 { self.frames }
    pub fn grid(&self) -> Arc<FlatGrid> { self.grid.clone() }
    pub fn manifest_grid(&self) -> Arc<FlatGrid> { self.manifest_grid.clone() }
    /// Every grid this sender draws: the manifest grid, then the data grid or the carousel.
    pub fn grids(&self) -> Vec<Arc<FlatGrid>> {
        let mut v = vec![self.manifest_grid.clone()];
        if self.carousel.is_empty() { v.push(self.grid.clone()); } else { v.extend(self.carousel.iter().cloned()); }
        v
    }
    /// What frame `n` is: (is_manifest, grid). Fixed and blind.
    fn frame_kind(&self, n: u64) -> (bool, Arc<FlatGrid>) {
        // an empty stream has no data blocks to send (a data frame would ask for a block of a 0-byte chunk):
        // the manifest alone completes it
        if n % self.cfg.manifest_every_frames as u64 == 0 || self.src.manifest().stream_size == 0 { return (true, self.manifest_grid.clone()); }
        if self.tune.is_some() {
            // the format the frame number names (the held frames took numbers before it)
            let f = crate::tune::schedule::slot(((n + self.held) & 0xFF) as u8).format;
            return (false, self.carousel.iter().find(|g| g.format() == f).expect("a grid per tune format").clone());
        }
        if !self.carousel.is_empty() {
            // data frames cycle the carousel in order, skipping the manifest slots
            let k = n - n / self.cfg.manifest_every_frames as u64 - 1;
            return (false, self.carousel[(k % self.carousel.len() as u64) as usize].clone());
        }
        (false, self.grid.clone())
    }
    /// Frames in one systematic pass (all data once, manifests interleaved).
    pub fn frames_per_pass(&self) -> u64 {
        let packets = self.src.data_packets_per_pass();
        let (mut sent, mut n) = (0u64, 0u64);
        loop {
            let (is_manifest, grid) = self.frame_kind(n);
            if !is_manifest { sent += grid.packets() as u64; }
            n += 1;
            if sent >= packets { return n; }
        }
    }
    /// File payload per second at `fps` display frames per second (packet payload only; manifest frames excluded).
    pub fn bytes_per_second(&self, fps: f64) -> f64 {
        let m = self.cfg.manifest_every_frames as f64;
        fps * (m - 1.0) / m * (self.grid.packets() * PACKET_PAYLOAD) as f64
    }
    /// Header and packets of the next frame (no cell composition): the transport layer on its own.
    pub fn next_packets(&mut self) -> io::Result<(FlatHeader, Vec<Packet>)> {
        let (header, packets, _) = self.next_frame_parts()?;
        Ok((header, packets))
    }
    fn next_frame_parts(&mut self) -> io::Result<(FlatHeader, Vec<Packet>, Arc<FlatGrid>)> {
        let n = self.frames; self.frames += 1;
        let (is_manifest, grid) = self.frame_kind(n);
        if let Some(d60) = self.tune {
            let seq = ((n + self.held) & 0xFF) as u8;
            self.frame_fps = crate::tune::schedule::shown_fps(crate::tune::schedule::slot(seq).fps, d60);
        }
        self.parts(is_manifest, grid, n + self.held)
    }
    fn parts(&mut self, is_manifest: bool, grid: Arc<FlatGrid>, seq: u64) -> io::Result<(FlatHeader, Vec<Packet>, Arc<FlatGrid>)> {
        let mut packets = Vec::with_capacity(grid.packets());
        for _ in 0..grid.packets() { packets.push(if is_manifest { self.src.next_manifest_packet() } else { self.src.next_data_packet()? }); }
        let first = packets[0].id;
        let header = FlatHeader {
            format: grid.format(), cols: grid.cols() as u16, rows: grid.rows() as u16,
            frame_type: if is_manifest { FrameType::Manifest } else { FrameType::Data },
            transfer_id: self.cfg.transfer_id,
            segment: first.segment,
            block_id: if is_manifest { self.src.manifest_bytes_len() as u32 } else { (first.chunk as u32) << 12 | first.pid as u32 },
            frame_seq: (seq & 0xFF) as u8,
        };
        Ok((header, packets, grid))
    }
    /// Applies finish words: from the next data packet on, only repair packets of the units `code` names
    /// (round-robin), manifests as before, until each has had its rounds (`focus_done`). A new code replaces the
    /// last one; an invalid one changes nothing.
    pub fn finish(&mut self, code: &str) -> Result<Vec<u32>, FinishError> {
        let units = crate::finish::decode(self.cfg.transfer_id, self.src.manifest().finish_units(), code)?;
        self.src.set_focus(Some(units.clone()));
        Ok(units)
    }
    /// What finish-word units `units` are, for the sender's screen: for a batch the files with bytes in them
    /// (`Index::files_in_units`, `files_text`: "3 files: a.jpg, b.pdf +1"; the receiver may have some of them
    /// already: these are the files in these parts), else, or when they hold no file, the parts ("3 parts: 7, 12, 30").
    pub fn units_text(&self, units: &[u32]) -> String {
        if let Some(idx) = self.batch_index() {
            let files = idx.files_in_units(self.manifest().finish_unit_bytes(), units);
            if !files.is_empty() {
                let names: Vec<&str> = files.iter().take(2).map(|&i| idx.entries[i].name.as_str()).collect();
                return crate::file::files_text(files.len(), &names);
            }
        }
        crate::finish::parts_text(units)
    }
    /// Back to the normal schedule, where it left off.
    pub fn send_everything(&mut self) { self.src.set_focus(None); }
    /// The units the finish words named, while they are being sent.
    pub fn focus(&self) -> Option<&[u32]> { self.src.focus() }
    /// True once the finish words' parts have all had their repair rounds and the sender went back to sending
    /// everything by itself (`PacketSource::focus_done`): new words from the receiver would finish faster.
    pub fn focus_done(&self) -> bool { self.src.focus_done() }
    /// The manifest again under the next frame number, for a sender that holds its first frame while the user
    /// aims (the countdown): each displayed frame is new to a receiver, which reads the sender's frame rate
    /// before data starts. The schedule does not move (`frames_emitted` stays); the numbers of the frames after
    /// it carry on from this one. Its packets are further fountain packets of the same manifest.
    pub fn hold_manifest_frame(&mut self) -> FlatFrame {
        let seq = self.frames + self.held;
        self.held += 1;
        self.frame_fps = 30;
        let grid = self.manifest_grid.clone();
        let (header, packets, grid) = self.parts(true, grid, seq).expect("manifest packets never fail");
        let payload = pack_frame(&packets, &grid);
        compose(grid, header, &payload)
    }
    pub fn next_frame(&mut self) -> io::Result<FlatFrame> {
        let (header, packets, grid) = self.next_frame_parts()?;
        let payload = pack_frame(&packets, &grid);
        Ok(compose(grid, header, &payload))
    }
}
