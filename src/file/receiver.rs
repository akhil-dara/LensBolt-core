use super::batch::{files_in_units, files_text, safe_name, Entry, Index};
use super::manifest::{chunk_blocks, BatchInfo, Manifest, TuneInfo};
use super::names::{with_suffix, UniqueNames};
use crate::fountain::{BlockStore, LtDecoder, BLOCK_BYTES};
use crate::header::{FrameMeta, FrameType};
use crate::packet::{unpack_frame, Packet};
use crate::flat::FramePayload;
use sha2::{Digest, Sha256};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq)]
/// `packets_accepted` counts every packet fed to a decoder (manifest and duplicates included);
/// `packets_new` only data packets the fountain had not seen before.
/// `disk_peak`: `Manifest::receive_disk_peak` once the manifest is in (for a batch, once its file list is in:
/// the stream plus every file's size), else 0.
pub struct Progress { pub have_manifest: bool, pub segments_done: u32, pub segments_total: u32, pub bytes_done: u64, pub stream_size: u64, pub packets_accepted: u64, pub packets_new: u64, pub packets_ignored: u64, pub filename: Option<String>, pub disk_peak: u64,
    /// the last manifest decoded was the Tune carousel's (`Manifest::tune`): the sender is tuning. The receiver
    /// holds no lock on it; cleared when another transfer's manifest is decoded
    pub tune_sender: Option<TuneInfo> }

#[derive(Debug, PartialEq)]
pub enum FeedOutcome { Ignored, Progress, ManifestReceived, SegmentDone(u16), Complete(PathBuf) }

/// Where one file of a batch stands. `Receiving(p)`: p in (0, 1), the share of its bytes in the stream decoded
/// so far. `Failed` carries the reason, in words for the screen ("storage full", "checksum mismatch", ...).
#[derive(Clone, Debug, PartialEq)]
pub enum FileStage { Waiting, Receiving(f32), Received, Failed(String) }

/// One file of a batch: `name` is the name it is saved under (the sender's, sanitised, with " (2)" when an
/// earlier file of the batch took it; the store's name for a store-backed receiver), `size` its size; `offset`
/// and `stored_size` where its bytes sit in the stream (what maps the finish words' units to files).
#[derive(Clone, Debug, PartialEq)]
pub struct FileState { pub name: String, pub size: u64, pub offset: u64, pub stored_size: u64, pub state: FileStage,
    /// the file's SHA-256 once it is received (its bytes verified against it), else None
    pub sha256: Option<[u8; 32]> }

/// A batch transfer as the receiver sees it. Until the file list (the index) has arrived and verified, `total`
/// is 0 and `files` is empty (a batch always lists at least one file). `error`: the file list was refused
/// (it did not match its checksum once its segments verified, or did not decode): no file can be saved.
/// `unit_bytes`: the bytes of the stream one finish-word unit spans (unit u is bytes u * unit_bytes ..
/// (u + 1) * unit_bytes): a segment when the stream has two or more, else a chunk (`Manifest::finish_units`).
#[derive(Clone, Debug, PartialEq)]
pub struct BatchStatus { pub name: String, pub total: u32, pub received: u32, pub failed: u32, pub files: Vec<FileState>, pub error: Option<String>, pub unit_bytes: u64 }

impl BatchStatus {
    /// The files (indices into `files`, ascending) with stored bytes in any of the finish-word units `units`
    /// (any order, repeats allowed; units past the stream name nothing): by overlap of each file's
    /// `offset .. offset + stored_size` with the unit's bytes. An empty file holds no bytes and is in no unit; the
    /// file list and the first files share unit 0. Whatever their state: the caller keeps those still missing.
    pub fn files_in_units(&self, units: &[u32]) -> Vec<usize> {
        files_in_units(self.files.len(), |i| (self.files[i].offset, self.files[i].stored_size), self.unit_bytes, units)
    }
}

/// A batch's counts for the status shown every frame: no file list walked, nothing as large as the list copied.
/// `missing_files`: the files the current finish words would bring (not yet received or failed, with bytes in the
/// units the words name); None while there are no words or no file list.
#[derive(Clone, Debug, PartialEq)]
pub struct BatchCounts { pub name: String, pub total: u32, pub received: u32, pub failed: u32, pub missing_files: Option<u32>, pub error: Option<String> }

struct Transfer {
    id: u16,
    manifest_len: usize, manifest_decoders: Vec<LtDecoder>, manifest: Option<Manifest>,
    part: Option<Part>, part_path: Option<PathBuf>,
    chunks: HashMap<(u16, u8), LtDecoder>,
    chunk_done: HashMap<u16, Vec<bool>>, segment_done: Vec<bool>,
    bytes_done: u64, accepted: u64, new_packets: u64, ignored: u64, hash_failures: u32,
    /// a repair packet (packet id >= its chunk's block count) has arrived: the sender's first pass is over
    repair_seen: bool,
    /// data packets that arrived before the manifest (chunk sizes unknown yet), replayed once it decodes
    pending: Vec<Packet>,
}

/// Data packets kept while waiting for the first manifest: ~6.7 s of 1 px frames at 30 fps (manifests come
/// every second), 17 MB.
pub const PENDING_MAX_PACKETS: usize = 16_384;

/// Random-access storage for one file of a transfer (a browser's Origin Private File System, say): the receiver
/// writes the reassembled stream into one store and a compressed transfer's file into a second, chunk by chunk,
/// so neither is ever whole in memory.
pub trait PartStore: Send {
    /// Read exactly `buf.len()` bytes at `offset`.
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()>;
    fn write_at(&mut self, offset: u64, data: &[u8]) -> io::Result<()>;
    fn set_len(&mut self, len: u64) -> io::Result<()>;
}
/// Creates named stores and removes them (the part of a compressed transfer once its file is out).
pub trait StoreFactory: Send {
    fn create(&mut self, name: &str) -> io::Result<Box<dyn PartStore>>;
    fn remove(&mut self, name: &str);
}

/// The reassembled stream: a part file on disk, a buffer in memory, or a store (see `Receiver::with_stores`).
enum Part { Disk(File), Mem(io::Cursor<Vec<u8>>), Ext { store: Box<dyn PartStore>, name: String, pos: u64, len: u64 } }
impl Read for Part {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        match self {
            Part::Disk(f) => f.read(b), Part::Mem(c) => c.read(b),
            Part::Ext { store, pos, len, .. } => {
                let n = (b.len() as u64).min(len.saturating_sub(*pos)) as usize;
                if n > 0 { store.read_at(*pos, &mut b[..n])?; *pos += n as u64; }
                Ok(n)
            }
        }
    }
}
impl Write for Part {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        match self {
            Part::Disk(f) => f.write(b), Part::Mem(c) => c.write(b),
            Part::Ext { store, pos, len, .. } => { store.write_at(*pos, b)?; *pos += b.len() as u64; *len = (*len).max(*pos); Ok(b.len()) }
        }
    }
    fn flush(&mut self) -> io::Result<()> { match self { Part::Disk(f) => f.flush(), Part::Mem(c) => c.flush(), Part::Ext { .. } => Ok(()) } }
}
impl Seek for Part {
    fn seek(&mut self, p: SeekFrom) -> io::Result<u64> {
        match self {
            Part::Disk(f) => f.seek(p), Part::Mem(c) => c.seek(p),
            Part::Ext { pos, len, .. } => {
                let to = match p { SeekFrom::Start(o) => o as i64, SeekFrom::End(o) => *len as i64 + o, SeekFrom::Current(o) => *pos as i64 + o };
                if to < 0 { return Err(io::Error::other("seek before the start")); }
                *pos = to as u64;
                Ok(*pos)
            }
        }
    }
}

/// Block store over the part file: block i of a chunk lives at `base + i*1024`. Reads are rare (only when a
/// repair packet touches an already-known block) and hit the OS page cache. A failed read or write (a full
/// disk, a browser store over its quota) is kept in `err` and the store does nothing more: the caller returns
/// it as the feed's error (a panic would trap a browser's WebAssembly instance for good).
struct FileStore<'a> { file: &'a mut Part, base: u64, len: usize, err: Option<io::Error> }
impl<'a> BlockStore for FileStore<'a> {
    fn read(&mut self, i: usize) -> [u8; BLOCK_BYTES] {
        let mut b = [0u8; BLOCK_BYTES];
        if self.err.is_some() { return b; }
        let n = (self.len - i * BLOCK_BYTES).min(BLOCK_BYTES);
        let r = self.file.seek(SeekFrom::Start(self.base + (i * BLOCK_BYTES) as u64)).and_then(|_| self.file.read_exact(&mut b[..n]));
        if let Err(e) = r { self.err = Some(e); }
        b
    }
    fn write(&mut self, i: usize, data: &[u8; BLOCK_BYTES]) {
        if self.err.is_some() { return; }
        let n = (self.len - i * BLOCK_BYTES).min(BLOCK_BYTES);
        let r = self.file.seek(SeekFrom::Start(self.base + (i * BLOCK_BYTES) as u64)).and_then(|_| self.file.write_all(&data[..n]));
        if let Err(e) = r { self.err = Some(e); }
    }
}

/// Receives packets of one transfer at a time, writes verified segments to a part file, and finishes
/// the file (decompress/rename + SHA-256) when every segment is in. Memory ~ live chunk decoders only.
pub struct Receiver { out_dir: PathBuf, t: Option<Transfer>, complete: bool, completed_id: Option<u16>, last_progress: Progress,
    /// in-memory receiver: no file system; the verified file waits in `taken` for `take_file`
    memory: bool, taken: Option<(String, Vec<u8>)>,
    /// store-backed receiver: the factory, and (shown name, store name) of the verified file once complete
    stores: Option<Box<dyn StoreFactory>>, stored: Option<(String, String)>,
    /// a batch transfer: its file list and each file's state, until another transfer starts
    batch: Option<Batch>,
    /// the Tune carousel's transfer id and tune section, once its manifest decoded: its frames are ignored (no
    /// lock, no part, no store) until another transfer's manifest decodes
    tune: Option<(u16, TuneInfo)> }

/// The sender's filename is untrusted: keep only a final path component, drop separators and control
/// characters, never produce `.`/`..`/empty.
pub fn safe_filename(name: &str) -> String {
    let last = name.rsplit(|c| c == '/' || c == '\\').next().unwrap_or("");
    let cleaned: String = last.chars().filter(|c| !c.is_control() && !matches!(c, ':' | '*' | '?' | '"' | '<' | '>' | '|')).collect();
    let cleaned = cleaned.trim().trim_matches('.').to_string();
    if cleaned.is_empty() { "file".to_string() } else { cleaned }
}

fn unique_path(dir: &Path, name: &str) -> PathBuf {
    let p = dir.join(name);
    if !p.exists() { return p; }
    let (stem, ext) = match name.rfind('.') { Some(i) if i > 0 => (&name[..i], &name[i..]), _ => (name, "") };
    for n in 2.. { let q = dir.join(format!("{stem} ({n}){ext}")); if !q.exists() { return q; } }
    unreachable!()
}

/// A streaming zstd decoder over `src`: the zstd crate where it is built in, ruzstd otherwise (WebAssembly).
/// `window_log`: the largest window a frame may ask for, 2^window_log bytes (the zstd crate; 27, its default, for
/// a single file; ruzstd grows its buffer only as output arrives, which callers cap).
fn decoder<'a, R: Read + 'a>(src: &'a mut R, window_log: u32) -> io::Result<Box<dyn Read + 'a>> {
    #[cfg(feature = "zstd")]
    { let mut d = zstd::stream::Decoder::new(src)?; d.window_log_max(window_log)?; Ok(Box::new(d)) }
    #[cfg(not(feature = "zstd"))]
    { let _ = window_log; Ok(Box::new(ruzstd::StreamingDecoder::new(src).map_err(|e| io::Error::other(e.to_string()))?)) }
}

/// Copies `r` into `sink` until its end or until more than `cap` bytes came: false when it passed `cap` (it stops
/// within one buffer of that). A transfer's file never unpacks past the size its manifest declares; a stream that
/// would (a hostile sender's zip bomb: a few KB of zstd for gigabytes) is refused before it fills memory or storage.
fn copy_capped(r: &mut dyn Read, cap: u64, mut sink: impl FnMut(&[u8]) -> io::Result<()>) -> io::Result<bool> {
    let mut buf = vec![0u8; 1 << 20];
    let mut total = 0u64;
    loop {
        let n = r.read(&mut buf)?;
        if n == 0 { return Ok(true); }
        total += n as u64;
        if total > cap { return Ok(false); }
        sink(&buf[..n])?;
    }
}

/// `decompress`, refused (an error) past `cap` bytes (`copy_capped`).
fn decompress_capped(stream: &[u8], cap: u64) -> io::Result<Vec<u8>> {
    let mut src = stream;
    let mut out = Vec::new();
    if copy_capped(&mut decoder(&mut src, 27)?, cap, |b| { out.extend_from_slice(b); Ok(()) })? { Ok(out) } else { Err(io::Error::other("unpacks past its declared size")) }
}

/// A whole zstd stream in memory: the zstd crate where it is built in, pure-Rust ruzstd otherwise (WebAssembly).
#[allow(dead_code)]
fn decompress(stream: &[u8]) -> io::Result<Vec<u8>> {
    #[cfg(feature = "zstd")]
    { zstd::stream::decode_all(stream) }
    #[cfg(not(feature = "zstd"))]
    {
        let mut src = stream;
        let mut dec = ruzstd::StreamingDecoder::new(&mut src).map_err(|e| io::Error::other(e.to_string()))?;
        let mut out = Vec::new();
        dec.read_to_end(&mut out)?;
        Ok(out)
    }
}

impl Receiver {
    pub fn new(out_dir: &Path) -> io::Result<Receiver> {
        std::fs::create_dir_all(out_dir)?;
        Ok(Receiver { out_dir: out_dir.to_path_buf(), t: None, complete: false, completed_id: None,
            last_progress: Progress { have_manifest: false, segments_done: 0, segments_total: 0, bytes_done: 0, stream_size: 0, packets_accepted: 0, packets_new: 0, packets_ignored: 0, filename: None, disk_peak: 0, tune_sender: None },
            memory: false, taken: None, stores: None, stored: None, batch: None, tune: None })
    }
    /// A receiver that keeps the stream in memory (a browser): `Complete` carries the file name only and the
    /// verified file is taken out with `take_file`.
    pub fn in_memory() -> Receiver {
        Receiver { out_dir: PathBuf::new(), t: None, complete: false, completed_id: None,
            last_progress: Progress { have_manifest: false, segments_done: 0, segments_total: 0, bytes_done: 0, stream_size: 0, packets_accepted: 0, packets_new: 0, packets_ignored: 0, filename: None, disk_peak: 0, tune_sender: None },
            memory: true, taken: None, stores: None, stored: None, batch: None, tune: None }
    }
    /// A receiver whose stream and file live in `stores` (a browser's OPFS): nothing whole in memory.
    /// `Complete` carries the shown file name; `stored_file` names the store holding the verified file.
    pub fn with_stores(stores: Box<dyn StoreFactory>) -> Receiver {
        let mut r = Receiver::in_memory();
        r.memory = false;
        r.stores = Some(stores);
        r
    }
    /// (shown file name, store name) of a store-backed receiver's verified file.
    pub fn stored_file(&self) -> Option<(String, String)> { self.stored.clone() }
    /// The verified file of an in-memory receiver (name, bytes), once.
    pub fn take_file(&mut self) -> Option<(String, Vec<u8>)> { self.taken.take() }
    pub fn is_complete(&self) -> bool { self.complete }
    /// The SHA-256 of the single file received (the file was verified against it): Some once complete, until another
    /// transfer starts; None for a batch (each file's is in `batch_files`).
    pub fn file_sha256(&self) -> Option<[u8; 32]> {
        if !self.complete || self.batch.is_some() { return None; }
        self.t.as_ref()?.manifest.as_ref().map(|m| m.file_sha256)
    }
    /// A batch's file list with each file's state (in stream order: smallest first), from its manifest on and
    /// after completion until another transfer starts; None for a single file or before the manifest. Each file
    /// is saved the moment its bytes verify: a disk receiver into `<out_dir>/<batch name>/` (" (2)" when that
    /// folder exists), a store-backed one into a store named as the file, an in-memory one for
    /// `take_batch_file`. Costs a pass over the list.
    pub fn batch_status(&self) -> Option<BatchStatus> { self.batch.as_ref().map(|b| b.status(self.t.as_ref())) }
    /// A batch's counts (`BatchCounts`), for a status read every frame: a few numbers and the batch's name. The
    /// missing files are counted again only when the finish words or the files settled (received or failed) change.
    pub fn batch_counts(&self) -> Option<BatchCounts> {
        let b = self.batch.as_ref()?;
        let missing_files = self.named_missing().map(|(n, _)| n);
        Some(BatchCounts { name: b.name.clone(), total: b.entries.len() as u32, received: b.received, failed: b.failed, missing_files, error: b.error.clone() })
    }
    /// Files `offset .. offset + count` of a batch's list (fewer at its end, none past it; none before the list is
    /// in), as `batch_status` gives them: a page of the list costs that page only.
    pub fn batch_files(&self, offset: usize, count: usize) -> Vec<FileState> {
        let Some(b) = self.batch.as_ref() else { return Vec::new() };
        let end = offset.saturating_add(count).min(b.entries.len());
        (offset.min(end)..end).map(|i| b.file_state(i, self.t.as_ref())).collect()
    }
    /// Files looked at so far to count the missing files (`BatchCounts::missing_files`): for tests, which check
    /// that a status read every frame does not walk the list.
    pub fn batch_file_scans(&self) -> u64 { self.batch.as_ref().map_or(0, |b| b.scans.get()) }
    /// The files the finish words would bring: how many, and the first two (indices). They are the files not yet
    /// received or failed with bytes in the units the words name (each file's own state: a file is saved as its
    /// chunks decode, so a short segment may hold saved files). Counted again only when the words, or the files
    /// settled, change; None without words, a file list, or when the list was refused.
    fn named_missing(&self) -> Option<(u32, Vec<usize>)> {
        let b = self.batch.as_ref()?;
        if b.entries.is_empty() || b.error.is_some() { return None; }
        let words = self.finish_words()?;
        let settled = b.received + b.failed;
        if let Some(c) = b.missing.borrow().as_ref().filter(|c| c.words == words && c.settled == settled) { return Some((c.count, c.first.clone())); }
        let t = self.t.as_ref()?;
        let m = t.manifest.as_ref()?;
        let units = crate::finish::decode(t.id, m.finish_units(), &words).ok()?;
        let named = files_in_units(b.entries.len(), |i| (b.entries[i].offset, b.entries[i].stored_size), unit_bytes(m), &units);
        b.scans.set(b.scans.get() + named.len() as u64);
        let missing: Vec<usize> = named.into_iter().filter(|&i| matches!(b.slots[i], Slot::Pending { .. })).collect();
        let c = Missing { words, settled, count: missing.len() as u32, first: missing.into_iter().take(2).collect() };
        let out = (c.count, c.first.clone());
        *b.missing.borrow_mut() = Some(c);
        Some(out)
    }
    /// File `i` of a batch an in-memory receiver saved (its name, its bytes), once.
    pub fn take_batch_file(&mut self, i: usize) -> Option<(String, Vec<u8>)> {
        let b = self.batch.as_mut()?;
        let data = b.mem.get_mut(i)?.take()?;
        Some((b.names[i].clone(), data))
    }
    pub fn live_decoders(&self) -> usize { self.t.as_ref().map(|t| t.chunks.len() + t.manifest_decoders.iter().filter(|d| !d.is_done()).count()).unwrap_or(0) }
    /// Approximate bytes held by all open chunk decoders (pending repair equations + bitmaps).
    pub fn live_memory_bytes(&self) -> usize { self.t.as_ref().map(|t| t.chunks.values().map(|d| d.memory_bytes()).sum()).unwrap_or(0) }
    pub fn hash_failures(&self) -> u32 { self.t.as_ref().map(|t| t.hash_failures).unwrap_or(0) }
    pub fn progress(&self) -> Progress {
        let tune_sender = self.tune.map(|(_, info)| info);
        match &self.t {
            None => Progress { tune_sender, ..self.last_progress.clone() },
            Some(t) => Progress {
                have_manifest: t.manifest.is_some(),
                segments_done: t.segment_done.iter().filter(|&&d| d).count() as u32,
                segments_total: t.manifest.as_ref().map(|m| m.segment_count()).unwrap_or(0),
                bytes_done: t.bytes_done, stream_size: t.manifest.as_ref().map(|m| m.stream_size).unwrap_or(0),
                packets_accepted: t.accepted, packets_new: t.new_packets, packets_ignored: t.ignored, filename: t.manifest.as_ref().map(|m| m.filename.clone()),
                disk_peak: match (t.manifest.as_ref(), &self.batch) { (Some(m), Some(b)) => b.disk_peak(m), (Some(m), None) => m.receive_disk_peak(), (None, _) => 0 },
                tune_sender,
            },
        }
    }
    /// The finish words to type on the sender: Some once the manifest is known, a repair packet has arrived
    /// (the first pass is over) and at least one unit (`Manifest::finish_units`) is complete while at least
    /// one is short; they name the short units (a window code only those in a ..= a + 32 - b, a the first
    /// short unit, b = max(1, ceil(log2 n))). A unit is complete when its segment verified (two or more
    /// segments) or its chunk decoded (one segment).
    pub fn finish_words(&self) -> Option<String> {
        let t = self.t.as_ref()?;
        let m = t.manifest.as_ref()?;
        if !t.repair_seen { return None; }
        let n = m.finish_units();
        let by_segment = m.segment_count() >= 2;
        let short: Vec<u32> = (0..n).filter(|&u| if by_segment { !t.segment_done[u as usize + 1] } else { !t.chunk_done[&1][u as usize] }).collect();
        // None when nothing or everything is short: encode names no code then
        crate::finish::encode(t.id, n, &short)
    }
    /// The parts the finish words name, for a line under them: "Parts 7, 12–15, 30" (`finish::parts_line`, from 1
    /// as the sender shows them); None whenever there are no words. A window code names only the short parts in its
    /// window, so this is what typing the words will send, not every short part.
    ///
    /// A batch whose file list is in names the files instead: those the words bring (`BatchCounts::missing_files`),
    /// two by name, "3 files: a.jpg, b.pdf +1" (a name over 40 characters shortened in the middle, its end kept); the
    /// parts line while the list is not in, or when every file in the named parts is already saved.
    pub fn finish_parts(&self) -> Option<String> {
        let t = self.t.as_ref()?;
        let n = t.manifest.as_ref()?.finish_units();
        let units = crate::finish::decode(t.id, n, &self.finish_words()?).ok()?;
        if let (Some(b), Some((count, first))) = (self.batch.as_ref(), self.named_missing()) {
            if count > 0 {
                let names: Vec<&str> = first.iter().map(|&i| b.names[i].as_str()).collect();
                return Some(files_text(count as usize, &names));
            }
        }
        Some(crate::finish::parts_line(&units))
    }
    /// Convenience: unpack_frame + feed_packets.
    pub fn feed<H: FrameMeta>(&mut self, header: &H, payload: &FramePayload) -> io::Result<FeedOutcome> {
        let packets = unpack_frame(payload);
        self.feed_packets(header, &packets)
    }

    /// `header` is any frame header (v1 tile or v2 flat): only its FrameInfo is used.
    pub fn feed_packets<H: FrameMeta>(&mut self, header: &H, packets: &[Packet]) -> io::Result<FeedOutcome> {
        let header = header.info();
        // a finished transfer keeps broadcasting; do not start receiving the same one again
        if self.completed_id == Some(header.transfer_id) { return Ok(FeedOutcome::Ignored); }
        if header.frame_type == FrameType::Probe { return Ok(FeedOutcome::Ignored); }
        // the Tune carousel, once its manifest is known: nothing to receive. Its manifest frames are still read: a real
        // transfer under the same id (an older sender) has its manifest read and received
        let tune_id = self.tune.map(|(id, _)| id);
        if tune_id == Some(header.transfer_id) && header.frame_type != FrameType::Manifest { return Ok(FeedOutcome::Ignored); }
        if self.complete { self.t = None; self.complete = false; }
        // a receiver locks on the first transfer it sees, data frame or manifest; while it has seen no manifest
        // frame of that transfer, a manifest of another one takes over (a lock taken from data frames alone may
        // be a stale broadcast); once it has, other transfers are ignored
        let fresh = |id: u16| Transfer { id, manifest_len: 0, manifest_decoders: Vec::new(), manifest: None, part: None, part_path: None,
            chunks: HashMap::new(), chunk_done: HashMap::new(), segment_done: Vec::new(), bytes_done: 0, accepted: 0, new_packets: 0, ignored: 0, hash_failures: 0, repair_seen: false, pending: Vec::new() };
        match &self.t {
            Some(t) if t.id != header.transfer_id => {
                // a tune manifest being read again holds no lock: another transfer's manifest takes over
                let held = (t.manifest_len > 0 || t.manifest.is_some()) && tune_id != Some(t.id);
                if held || header.frame_type != FrameType::Manifest { return Ok(FeedOutcome::Ignored); }
                self.t = Some(fresh(header.transfer_id));
                self.batch = None;
            }
            None => { self.t = Some(fresh(header.transfer_id)); self.batch = None; }
            _ => {}
        }
        let out_dir = self.out_dir.clone();
        let t = self.t.as_mut().unwrap();
        // a manifest header announces the manifest length (block_id); set its decoders up once
        if header.frame_type == FrameType::Manifest && t.manifest.is_none() {
            let len = header.block_id as usize;
            if len > 0 && len < (1 << 24) && len <= super::SEGMENT_BYTES as usize && t.manifest_len != len {
                t.manifest_len = len;
                let chunk = super::CHUNK_BYTES as usize;
                t.manifest_decoders = (0..len).step_by(chunk).map(|off| { let l = (len - off).min(chunk); LtDecoder::new(chunk_blocks(l), l) }).collect();
            }
        }
        // Route every packet by its own id, not by the frame header: under a rolling shutter one capture can
        // hold the stripes of two displayed frames (say the top of a manifest frame and the bottom of a data
        // frame). Packet CRCs and the per-segment SHA-256 keep a wrong guess harmless.
        let mut outcome = FeedOutcome::Progress;
        for p in packets.iter().filter(|p| p.id.segment == 0) {
            if t.manifest.is_some() || p.id.chunk as usize >= t.manifest_decoders.len() { t.ignored += 1; continue; }
            t.manifest_decoders[p.id.chunk as usize].add(p.id.pid as u32, &p.payload); t.accepted += 1;
        }
        if t.manifest.is_none() && !t.manifest_decoders.is_empty() && t.manifest_decoders.iter().all(|d| d.is_done()) {
            let decs = std::mem::take(&mut t.manifest_decoders);
            let mut bytes = Vec::with_capacity(t.manifest_len);
            for d in decs { bytes.extend(d.take().unwrap()); }
            match Manifest::decode(&bytes) {
                None => { t.manifest_len = 0; }
                Some(m) if m.tune.is_some() => {
                    // the Tune carousel: named in the progress, never locked on, no part or store created; a
                    // transfer started next is received as if nothing came before
                    self.tune = Some((t.id, m.tune.unwrap()));
                    self.t = None;
                    self.batch = None;
                    return Ok(FeedOutcome::Ignored);
                }
                Some(mut m) => {
                    self.tune = None;
                    // a batch's name names its folder: the batch rules (`safe_name`), as for its files
                    m.filename = match m.batch.as_mut() { Some(b) => { b.name = safe_name(&b.name); b.name.clone() } None => safe_filename(&m.filename) };
                    let (part, part_path) = if let Some(stores) = self.stores.as_mut() {
                        let name = fit(&m.filename, ".lbpart");
                        let mut store = stores.create(&name)?;
                        store.set_len(m.stream_size)?;
                        (Part::Ext { store, name, pos: 0, len: m.stream_size }, PathBuf::from(&m.filename))
                    } else if self.memory {
                        (Part::Mem(io::Cursor::new(vec![0u8; m.stream_size as usize])), PathBuf::from(&m.filename))
                    } else {
                        let part_path = out_dir.join(fit(&m.filename, ".lbpart"));
                        let f = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&part_path)?;
                        f.set_len(m.stream_size)?;
                        (Part::Disk(f), part_path)
                    };
                    t.segment_done = vec![false; m.segment_count() as usize + 1];
                    for seg in 1..=m.segment_count() as u16 { t.chunk_done.insert(seg, vec![false; m.chunk_count(seg) as usize]); }
                    let part_name = match &part { Part::Ext { name, .. } => Some(name.clone()), _ => None };
                    self.batch = m.batch.as_ref().map(|info| Batch::new(info, part_name));
                    t.part = Some(part); t.part_path = Some(part_path); t.manifest = Some(m);
                    outcome = FeedOutcome::ManifestReceived;
                }
            }
        }
        let mut data: Vec<&Packet> = packets.iter().filter(|p| p.id.segment != 0).collect();
        let Some(m) = t.manifest.clone() else {
            let room = PENDING_MAX_PACKETS.saturating_sub(t.pending.len());
            t.ignored += data.len().saturating_sub(room) as u64;
            t.pending.extend(data.into_iter().take(room).cloned());
            return Ok(outcome);
        };
        let pending = std::mem::take(&mut t.pending);
        if !pending.is_empty() { data.splice(0..0, pending.iter()); }
        // an empty file has no segments and its frames no data: done once the manifest is in
        if data.is_empty() && m.segment_count() > 0 { return Ok(outcome); }
        // a batch's files are saved as their chunks decode (each checked against its own SHA-256)
        let mut out = match self.stores.as_deref_mut() { Some(s) => Out::Stores(s), None if self.memory => Out::Mem, None => Out::Dir(&out_dir) };
        let mut batch_err: Option<io::Error> = None;
        for p in data {
            let (seg, chunk) = (p.id.segment, p.id.chunk);
            if seg > m.segment_count() as u16 || chunk >= m.chunk_count(seg) { t.ignored += 1; continue; }
            let len = m.chunk_len(seg, chunk);
            if p.id.pid as usize >= chunk_blocks(len) { t.repair_seen = true; }
            if t.segment_done[seg as usize] || t.chunk_done[&seg][chunk as usize] { t.ignored += 1; continue; }
            let dec = t.chunks.entry((seg, chunk)).or_insert_with(|| LtDecoder::new_external(chunk_blocks(len), len));
            t.accepted += 1;
            let (before, solved) = (dec.seen_count(), dec.solved_blocks());
            let (done, err) = {
                let mut store = FileStore { file: t.part.as_mut().unwrap(), base: m.chunk_offset(seg, chunk), len, err: None };
                let done = dec.add_with(p.id.pid as u32, &p.payload, &mut store);
                (done, store.err)
            };
            if dec.seen_count() > before { t.new_packets += 1; }
            let progressed = dec.solved_blocks() > solved;
            // the chunk's blocks on disk are no longer what its decoder believes: start the chunk over
            if let Some(e) = err { t.chunks.remove(&(seg, chunk)); return Err(e); }
            if !done && progressed {
                if let Some(b) = self.batch.as_mut() {
                    let have = Have { seg: &t.segment_done, chunks: &t.chunk_done, live: &t.chunks };
                    if let Err(e) = b.on_progress(&m, seg, chunk, &have, t.part.as_mut().unwrap(), &mut out) { batch_err.get_or_insert(e); }
                }
            }
            if done {
                t.chunks.remove(&(seg, chunk));
                t.chunk_done.get_mut(&seg).unwrap()[chunk as usize] = true;
                if let Some(b) = self.batch.as_mut() {
                    let have = Have { seg: &t.segment_done, chunks: &t.chunk_done, live: &t.chunks };
                    if let Err(e) = b.on_chunk(&m, seg, chunk, &have, t.part.as_mut().unwrap(), &mut out) { batch_err.get_or_insert(e); }
                }
                if t.chunk_done[&seg].iter().all(|&d| d) {
                    let f = t.part.as_mut().unwrap();
                    f.seek(SeekFrom::Start(m.chunk_offset(seg, 0)))?;
                    let mut h = Sha256::new(); let mut left = m.segment_len(seg); let mut buf = vec![0u8; 1 << 20];
                    while left > 0 { let n = (left as usize).min(buf.len()); f.read_exact(&mut buf[..n])?; h.update(&buf[..n]); left -= n as u64; }
                    if <[u8; 32]>::from(h.finalize()) == m.segment_sha256[seg as usize - 1] {
                        t.segment_done[seg as usize] = true; t.bytes_done += m.segment_len(seg);
                        outcome = FeedOutcome::SegmentDone(seg);
                        if let Some(b) = self.batch.as_mut() {
                            let have = Have { seg: &t.segment_done, chunks: &t.chunk_done, live: &t.chunks };
                            if let Err(e) = b.on_segment(&m, seg, &have, t.part.as_mut().unwrap(), &mut out) { batch_err.get_or_insert(e); }
                        }
                    } else {
                        t.hash_failures += 1;
                        for d in t.chunk_done.get_mut(&seg).unwrap() { *d = false; }
                        if let Some(b) = self.batch.as_mut() { b.on_segment_failed(seg); }
                    }
                }
            }
        }
        drop(out);
        let all_done = (1..=m.segment_count() as usize).all(|s| t.segment_done[s]);
        // a batch completes in the feed its last segment verifies, whatever went wrong earlier in it (a read that
        // failed is tried again while settling); a batch whose file list was refused never completes
        if all_done && self.batch.is_some() {
            return match self.finish_batch()? { Some(p) => Ok(FeedOutcome::Complete(p)), None => batch_err.map_or(Ok(outcome), Err) };
        }
        if let Some(e) = batch_err { return Err(e); }
        if all_done {
            let path = self.finish()?;
            return Ok(match path { Some(p) => FeedOutcome::Complete(p), None => FeedOutcome::Progress });
        }
        Ok(outcome)
    }

    /// A batch's segments all verified: every file has its verdict (saved or failed), the stream's part is
    /// removed. Returns the batch's folder (a disk receiver) or its name; None when the file list was refused (the
    /// part is removed, the batch never completes: its status carries the error). The stream's own SHA-256 is not
    /// checked again: every segment verified, and every file against its own hash.
    fn finish_batch(&mut self) -> io::Result<Option<PathBuf>> {
        let out_dir = self.out_dir.clone();
        let t = self.t.as_mut().unwrap();
        let m = t.manifest.clone().unwrap();
        let b = self.batch.as_mut().unwrap();
        let settled = {
            let mut out = match self.stores.as_deref_mut() { Some(s) => Out::Stores(s), None if self.memory => Out::Mem, None => Out::Dir(&out_dir) };
            let have = Have { seg: &t.segment_done, chunks: &t.chunk_done, live: &t.chunks };
            match t.part.as_mut() { Some(part) => b.settle(&m, &have, part, &mut out), None => Ok(()) }
        };
        if b.error.is_some() { remove_part(t, self.stores.as_mut()); return settled.map(|()| None); }
        settled?;
        remove_part(t, self.stores.as_mut());
        let path = if self.stores.is_none() && !self.memory {
            if b.dir.is_none() { b.dir = Some(new_folder(&out_dir, &b.name)?); }
            b.dir.clone().unwrap()
        } else { PathBuf::from(&b.name) };
        self.last_progress = self.progress();
        self.complete = true;
        self.completed_id = self.t.as_ref().map(|t| t.id);
        Ok(Some(path))
    }

    /// All segments verified: decompress or rename, verify the file hash. None => hash failed, state reset.
    fn finish(&mut self) -> io::Result<Option<PathBuf>> {
        let (m, part_path, part) = {
            let t = self.t.as_mut().unwrap();
            (t.manifest.clone().unwrap(), t.part_path.clone().unwrap(), t.part.take())
        };
        if let Some(Part::Mem(c)) = part {
            let stream = c.into_inner();
            // past its declared size: refused like a file whose hash fails (copy_capped)
            let data = if m.compressed { match decompress_capped(&stream, m.file_size) { Ok(d) => d, Err(_) => { self.t = None; return Ok(None); } } } else { stream };
            if <[u8; 32]>::from(Sha256::digest(&data)) != m.file_sha256 { self.t = None; return Ok(None); }
            self.taken = Some((m.filename.clone(), data));
            self.last_progress = self.progress();
            self.complete = true;
            self.completed_id = self.t.as_ref().map(|t| t.id);
            return Ok(Some(PathBuf::from(&m.filename)));
        }
        if let Some(Part::Ext { store, name, len, .. }) = part {
            let mut part = Part::Ext { store, name: name.clone(), pos: 0, len };
            let mut hasher = Sha256::new();
            let mut buf = vec![0u8; 1 << 20];
            let shown = if m.compressed {
                // decompress chunk by chunk from the part store into the file's store
                let stores = self.stores.as_mut().unwrap();
                let mut out = stores.create(&m.filename)?;
                let mut at = 0u64;
                let fits = {
                    let mut dec = decoder(&mut part, 27)?;
                    copy_capped(&mut dec, m.file_size, |b| { hasher.update(b); out.write_at(at, b)?; at += b.len() as u64; Ok(()) })?
                };
                out.set_len(at)?;
                drop(part);
                stores.remove(&name);
                // past its declared size: refused like a file whose hash fails
                if !fits { drop(out); stores.remove(&m.filename); self.t = None; return Ok(None); }
                m.filename.clone()
            } else {
                loop { let n = part.read(&mut buf)?; if n == 0 { break; } hasher.update(&buf[..n]); }
                name
            };
            if <[u8; 32]>::from(hasher.finalize()) != m.file_sha256 { self.t = None; return Ok(None); }
            self.stored = Some((m.filename.clone(), shown));
            self.last_progress = self.progress();
            self.complete = true;
            self.completed_id = self.t.as_ref().map(|t| t.id);
            return Ok(Some(PathBuf::from(&m.filename)));
        }
        let final_path = unique_path(&self.out_dir, &m.filename);
        let mut hasher = Sha256::new();
        #[cfg(not(feature = "zstd"))]
        if m.compressed { let _ = (&part_path, &mut hasher); return Err(io::Error::other("compressed transfer, but this build has no zstd")); }
        #[cfg(feature = "zstd")]
        if m.compressed {
            let tmp = self.out_dir.join(format!("{}.lbtmp", m.filename));
            {
                let mut dec = zstd::stream::Decoder::new(File::open(&part_path)?)?;
                let mut out = io::BufWriter::new(File::create(&tmp)?);
                let fits = copy_capped(&mut dec, m.file_size, |b| { hasher.update(b); out.write_all(b) })?;
                out.flush()?;
                // past its declared size: refused like a file whose hash fails
                if !fits { drop(out); let _ = std::fs::remove_file(&tmp); let _ = std::fs::remove_file(&part_path); self.t = None; return Ok(None); }
            }
            // verify before discarding the stream: a failure here means a bug, not a channel error
            if <[u8; 32]>::from(hasher.finalize()) != m.file_sha256 { let _ = std::fs::remove_file(&tmp); let _ = std::fs::remove_file(&part_path); self.t = None; return Ok(None); }
            std::fs::remove_file(&part_path)?;
            std::fs::rename(&tmp, &final_path)?;
        } else {
            {
                let mut f = File::open(&part_path)?; let mut buf = vec![0u8; 1 << 20];
                loop { let n = f.read(&mut buf)?; if n == 0 { break; } hasher.update(&buf[..n]); }
            }
            if <[u8; 32]>::from(hasher.finalize()) != m.file_sha256 { std::fs::remove_file(&part_path)?; self.t = None; return Ok(None); }
            std::fs::rename(&part_path, &final_path)?;
        }
        self.last_progress = self.progress();
        self.complete = true;
        self.completed_id = self.t.as_ref().map(|t| t.id);
        Ok(Some(final_path))
    }
}

/// A batch transfer's files: the file list once it verified (`entries`, in stream order), the name each file
/// is saved under, each file's slot, and where saved files go. Kept until another transfer starts, so the
/// list stays readable after completion.
struct Batch {
    name: String, index_len: u64, index_sha256: [u8; 32],
    /// the file list's bytes were all decoded without matching `index_sha256`: refused when its segments verify
    /// like that, read again when its chunks decode again (cleared when a segment holding it fails its hash)
    index_bad: bool,
    entries: Vec<Entry>, names: Vec<String>, slots: Vec<Slot>,
    /// hands out the names; kept, so that a name something else took on disk meanwhile gets the next number
    namer: UniqueNames,
    received: u32, failed: u32, total_size: u64,
    error: Option<String>,
    /// a store-backed receiver's stream store: a name no file may take
    part_name: Option<String>,
    /// a disk receiver's folder for the batch, made when the first file is saved
    dir: Option<PathBuf>,
    /// an in-memory receiver's saved files, until taken
    mem: Vec<Option<Vec<u8>>>,
    /// the files the finish words bring, as last counted (`Receiver::named_missing`), and the files looked at to count
    /// them so far
    missing: RefCell<Option<Missing>>, scans: Cell<u64>,
}

/// The missing files as counted for finish words `words` with `settled` files received or failed: how many, the
/// first two.
struct Missing { words: String, settled: u32, count: u32, first: Vec<usize> }

/// `Pending { bad }`: not saved yet; `bad` says why its bytes, all decoded, did not verify last time (they may
/// be decoded again: a final verdict only once the segments holding them verified).
#[derive(Clone)]
enum Slot { Pending { bad: Option<String> }, Received, Failed(String) }

/// Where a batch's files are saved: a folder on disk (`<out_dir>/<batch name>/`), memory, or the stores.
enum Out<'a> { Dir(&'a Path), Mem, Stores(&'a mut dyn StoreFactory) }

/// What of the stream is in: verified segments, decoded chunks, and the solved blocks of chunks under way.
struct Have<'a> { seg: &'a [bool], chunks: &'a HashMap<u16, Vec<bool>>, live: &'a HashMap<(u16, u8), LtDecoder> }
const CHUNKS_PER_SEG: u64 = super::CHUNKS_PER_SEGMENT as u64;
/// (segment, chunk) of the stream's chunk `g` (from 0).
fn chunk_at(g: u64) -> (u16, u8) { ((g / CHUNKS_PER_SEG + 1) as u16, (g % CHUNKS_PER_SEG) as u8) }
impl Have<'_> {
    fn chunk(&self, g: u64) -> bool {
        let (s, c) = chunk_at(g);
        self.seg.get(s as usize) == Some(&true) || self.chunks.get(&s).is_some_and(|v| v.get(c as usize) == Some(&true))
    }
    /// Every byte of `start .. start + len` decoded (an empty range: true): its chunks decoded, or in a chunk
    /// under way the blocks holding it solved (what lets the file list be read before its chunk is whole).
    fn done(&self, start: u64, len: u64) -> bool {
        if len == 0 { return true; }
        let (end, size) = (start + len, super::CHUNK_BYTES);
        (start / size..=(end - 1) / size).all(|g| self.chunk(g) || self.live.get(&chunk_at(g)).is_some_and(|d| {
            let (lo, hi) = (start.max(g * size) - g * size, end.min((g + 1) * size) - g * size);
            (lo / BLOCK_BYTES as u64..=(hi - 1) / BLOCK_BYTES as u64).all(|b| d.is_solved(b as usize))
        }))
    }
    /// Every segment holding bytes `start .. start + len` verified.
    fn verified(&self, start: u64, len: u64) -> bool { len == 0 || (start / super::SEGMENT_BYTES + 1..=(start + len - 1) / super::SEGMENT_BYTES + 1).all(|s| self.seg.get(s as usize) == Some(&true)) }
}

/// A file's bytes as they came out, or why they could not be saved.
enum Fail { Part(io::Error), Bad(&'static str), Write(io::Error) }
const CHECKSUM: &str = "checksum mismatch";
const UNPACK_SIZE: &str = "does not unpack to its size";
const UNPACK: &str = "cannot be unpacked";
/// The largest file verified in memory before it is written; a larger one is hashed as it is written to a
/// temporary file (disk), or read twice, once to verify and once to write (stores).
const SMALL_FILE: u64 = 1 << 20;

/// The reason a write failed, in words for the screen. A full store says so by its kind (a disk, Android) or
/// in its message (a browser's quota error).
fn write_failure(e: &io::Error) -> String {
    use io::ErrorKind::*;
    let msg = e.to_string();
    let low = msg.to_ascii_lowercase();
    let full = low.split(|c: char| !c.is_ascii_alphanumeric()).any(|w| w == "full" || w.starts_with("quota"));
    match e.kind() {
        StorageFull | QuotaExceeded => "storage full".into(),
        FileTooLarge => "too large for this storage".into(),
        PermissionDenied | ReadOnlyFilesystem => "storage not writable".into(),
        _ if full => "storage full".into(),
        _ => format!("write error: {msg}"),
    }
}

/// `name` and then `suffix`, in at most 255 bytes: the name cut at a character boundary.
fn fit(name: &str, suffix: &str) -> String {
    let mut n = (super::batch::MAX_NAME_BYTES - suffix.len()).min(name.len());
    while !name.is_char_boundary(n) { n -= 1; }
    format!("{}{suffix}", &name[..n])
}

/// A folder `<root>/<name>` that does not exist yet, made: `name`, else `with_suffix(name, 2, 3, ...)`.
fn new_folder(root: &Path, name: &str) -> io::Result<PathBuf> {
    for n in 1u32.. {
        let p = root.join(if n == 1 { name.to_owned() } else { with_suffix(name, n) });
        match std::fs::create_dir(&p) { Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue, r => return r.map(|()| p) }
    }
    unreachable!()
}

/// Reads through `r`, keeping its first error: a failed read of the stream's own store is the receiver's error,
/// not the file's.
struct Tracked<R> { r: R, err: Option<io::Error> }
impl<R: Read> Read for Tracked<R> {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        self.r.read(b).map_err(|e| { let copy = io::Error::new(e.kind(), e.to_string()); self.err.get_or_insert(e); copy })
    }
}

/// Where a file's bytes go while they are checked: a temporary file, a store, memory, or nowhere (a check only).
enum Sink { File(io::BufWriter<File>), Store(Box<dyn PartStore>), Mem(Vec<u8>), Null }
impl Sink {
    fn write(&mut self, at: u64, b: &[u8]) -> io::Result<()> {
        match self { Sink::File(w) => w.write_all(b), Sink::Store(s) => s.write_at(at, b), Sink::Mem(v) => { v.extend_from_slice(b); Ok(()) } Sink::Null => Ok(()) }
    }
}

/// The largest zstd window a file of `size` bytes needs, as a log (2^10 to 2^27): a frame that asks for more is
/// refused rather than given the memory (a sender's frame of a file states its size or fits it).
fn window_log(size: u64) -> u32 { (64 - size.max(1).saturating_sub(1).leading_zeros()).clamp(10, 27) }

/// Copies file `e` from the stream into `sink`: its stored bytes, unpacked when compressed with the output
/// capped at its size, hashed on the way.
fn copy_out(e: &Entry, part: &mut Part, sink: &mut Sink) -> Result<(), Fail> {
    part.seek(SeekFrom::Start(e.offset)).map_err(Fail::Part)?;
    let mut src = Tracked { r: (&mut *part).take(e.stored_size), err: None };
    let mut buf = vec![0u8; (e.size.saturating_add(1)).min(1 << 20) as usize];
    let mut hash = Sha256::new();
    let mut n = 0u64;
    let r = (|| {
        let rd: Box<dyn Read + '_> = if e.compressed { decoder(&mut src, window_log(e.size)).map_err(|_| Fail::Bad(UNPACK))? } else { Box::new(&mut src) };
        // one byte past the size is enough to know it unpacks to more
        let mut rd = rd.take(e.size + 1);
        loop {
            let k = match rd.read(&mut buf) { Ok(0) => return Ok(()), Ok(k) => k, Err(x) if x.kind() == io::ErrorKind::Interrupted => continue, Err(_) => return Err(Fail::Bad(UNPACK)) };
            if n + k as u64 > e.size { return Err(Fail::Bad(UNPACK_SIZE)); }
            hash.update(&buf[..k]);
            sink.write(n, &buf[..k]).map_err(Fail::Write)?;
            n += k as u64;
        }
    })();
    if let Some(x) = src.err.take() { return Err(Fail::Part(x)); }
    r?;
    if n != e.size { return Err(Fail::Bad(if e.compressed { UNPACK_SIZE } else { CHECKSUM })); }
    if <[u8; 32]>::from(hash.finalize()) != e.sha256 { return Err(Fail::Bad(CHECKSUM)); }
    Ok(())
}

/// Writes `data` into a new store `name` in pieces of at most 1 MiB; removes the store when that fails.
fn write_store(f: &mut dyn StoreFactory, name: &str, data: &[u8]) -> Result<(), Fail> {
    let mut s = f.create(name).map_err(Fail::Write)?;
    let r = data.chunks(1 << 20).enumerate().try_for_each(|(i, piece)| s.write_at((i << 20) as u64, piece)).and_then(|()| s.set_len(data.len() as u64));
    drop(s);
    if r.is_err() { f.remove(name); }
    r.map_err(Fail::Write)
}

impl Batch {
    fn new(info: &BatchInfo, part_name: Option<String>) -> Batch {
        Batch { name: info.name.clone(), index_len: info.index_len, index_sha256: info.index_sha256, index_bad: false,
            entries: Vec::new(), names: Vec::new(), slots: Vec::new(), namer: UniqueNames::new(), received: 0, failed: 0, total_size: 0,
            error: None, part_name, dir: None, mem: Vec::new(), missing: RefCell::new(None), scans: Cell::new(0) }
    }
    /// The stream, and beside it every file as it is saved (the part goes only at the end); before the file list
    /// is in, the manifest's estimate.
    fn disk_peak(&self, m: &Manifest) -> u64 { if self.entries.is_empty() { m.receive_disk_peak() } else { m.stream_size.saturating_add(self.total_size) } }
    fn refuse(&mut self, why: &str) -> io::Error { self.error = Some(why.to_string()); io::Error::new(io::ErrorKind::InvalidData, why.to_string()) }
    fn fail(&mut self, i: usize, why: String) { self.slots[i] = Slot::Failed(why); self.failed += 1; }
    fn index_in(&self, m: &Manifest, seg: u16, chunk: u8) -> bool { m.chunk_offset(seg, chunk) < self.index_len }

    /// A packet solved blocks of chunk (`seg`, `chunk`), not yet whole: the file list, when it lies there and all
    /// of it is in (it can be read long before its chunk is whole).
    fn on_progress(&mut self, m: &Manifest, seg: u16, chunk: u8, have: &Have, part: &mut Part, out: &mut Out) -> io::Result<()> {
        if self.error.is_some() || !self.entries.is_empty() || self.index_bad || !self.index_in(m, seg, chunk) { return Ok(()); }
        self.try_index(m, have, part, out)
    }
    /// Chunk (`seg`, `chunk`) decoded: while the file list is missing, it (from any chunk: a read that failed is
    /// tried again; a list that did not verify only when its own chunks decoded again), else the files with bytes
    /// in the chunk.
    fn on_chunk(&mut self, m: &Manifest, seg: u16, chunk: u8, have: &Have, part: &mut Part, out: &mut Out) -> io::Result<()> {
        if self.error.is_some() { return Ok(()); }
        if self.entries.is_empty() { return if !self.index_bad || self.index_in(m, seg, chunk) { self.try_index(m, have, part, out) } else { Ok(()) }; }
        let start = m.chunk_offset(seg, chunk);
        self.files_in(start, start + m.chunk_len(seg, chunk) as u64, true, have, part, out)
    }
    /// Segment `seg` failed its hash: its chunks decode again, the file list with them.
    fn on_segment_failed(&mut self, seg: u16) { if (seg as u64 - 1) * super::SEGMENT_BYTES < self.index_len { self.index_bad = false; } }
    /// Segment `seg` verified: what failed to verify in it is final now.
    fn on_segment(&mut self, m: &Manifest, seg: u16, have: &Have, part: &mut Part, out: &mut Out) -> io::Result<()> {
        if self.error.is_some() { return Ok(()); }
        if self.entries.is_empty() {
            if self.index_bad && have.verified(0, self.index_len) { return Err(self.refuse("the file list does not match its checksum")); }
            return self.try_index(m, have, part, out);
        }
        let start = (seg as u64 - 1) * super::SEGMENT_BYTES;
        self.files_in(start, start + m.segment_len(seg), false, have, part, out)
    }
    /// Every segment verified: the file list and every file have their verdict.
    fn settle(&mut self, m: &Manifest, have: &Have, part: &mut Part, out: &mut Out) -> io::Result<()> {
        if self.error.is_some() { return Ok(()); }
        if self.entries.is_empty() {
            self.try_index(m, have, part, out)?;
            if self.entries.is_empty() && self.error.is_none() { return Err(self.refuse("the file list does not match its checksum")); }
            return Ok(());
        }
        for i in 0..self.entries.len() { self.attempt(i, true, have, part, out)?; }
        Ok(())
    }
    /// The files with stored bytes in `start .. end` (files are in stream order, their ends never decrease).
    fn files_in(&mut self, start: u64, end: u64, recheck: bool, have: &Have, part: &mut Part, out: &mut Out) -> io::Result<()> {
        let first = self.entries.partition_point(|e| e.offset + e.stored_size <= start);
        for i in first..self.entries.len() {
            if self.entries[i].offset >= end { break; }
            if self.entries[i].stored_size > 0 { self.attempt(i, recheck, have, part, out)?; }
        }
        Ok(())
    }
    /// The file list's bytes are all decoded: checked against the manifest's hash, decoded, checked against the
    /// stream's size (in that order); then every file whose bytes are in already (empty files at once).
    fn try_index(&mut self, m: &Manifest, have: &Have, part: &mut Part, out: &mut Out) -> io::Result<()> {
        if !self.entries.is_empty() || self.error.is_some() || !have.done(0, self.index_len) { return Ok(()); }
        let mut bytes = vec![0u8; self.index_len as usize];
        part.seek(SeekFrom::Start(0))?;
        part.read_exact(&mut bytes)?;
        if <[u8; 32]>::from(Sha256::digest(&bytes)) != self.index_sha256 {
            if have.verified(0, self.index_len) { return Err(self.refuse("the file list does not match its checksum")); }
            self.index_bad = true;
            return Ok(());
        }
        let index = match Index::decode(&bytes) { Ok(i) => i, Err(e) => return Err(self.refuse(&format!("the file list cannot be read: {e}"))) };
        if index.stream_len() != m.stream_size { return Err(self.refuse("the file list does not match the transfer's size")); }
        if let Some(p) = &self.part_name { self.namer.reserve(p); }
        self.names = index.entries.iter().map(|e| self.namer.add(&e.name)).collect();
        self.total_size = index.entries.iter().fold(0u64, |a, e| a.saturating_add(e.size));
        self.slots = vec![Slot::Pending { bad: None }; index.entries.len()];
        if matches!(out, Out::Mem) { self.mem = vec![None; index.entries.len()]; }
        self.entries = index.entries;
        for i in 0..self.entries.len() { self.attempt(i, true, have, part, out)?; }
        Ok(())
    }
    /// File `i`, if pending and its bytes are all decoded: saved when they verify; failed when they do not and
    /// the segments holding them verified (`recheck`: read them again even when they did not verify last time).
    fn attempt(&mut self, i: usize, recheck: bool, have: &Have, part: &mut Part, out: &mut Out) -> io::Result<()> {
        let bad = match &self.slots[i] { Slot::Pending { bad } => bad.clone(), _ => return Ok(()) };
        let (off, len) = (self.entries[i].offset, self.entries[i].stored_size);
        if !have.done(off, len) { return Ok(()); }
        let verified = have.verified(off, len);
        if let Some(why) = bad {
            if verified { self.fail(i, why); return Ok(()); }
            if !recheck { return Ok(()); }
        }
        match self.save(i, part, out) {
            Ok(()) => { self.slots[i] = Slot::Received; self.received += 1; }
            Err(Fail::Part(e)) => return Err(e),
            Err(Fail::Bad(why)) => if verified { self.fail(i, why.into()) } else { self.slots[i] = Slot::Pending { bad: Some(why.into()) } },
            Err(Fail::Write(e)) => self.fail(i, write_failure(&e)),
        }
        Ok(())
    }
    /// Saves file `i`. Nothing unverified ever carries a file's name, and nothing already there is replaced:
    /// - a store: a file up to `SMALL_FILE` verified in memory, a larger one read once to verify, then written;
    /// - disk: written to a temporary file beside the batch's folder (from memory once verified, or hashed as it
    ///   is written), then linked in under its name (a hard link never replaces a file: a name taken meanwhile
    ///   gets the next number) and the temporary name removed; where hard links are not supported, moved in when
    ///   the name is free. A crash leaves at most the temporary file, never a short file under a real name;
    /// - memory: kept once verified.
    fn save(&mut self, i: usize, part: &mut Part, out: &mut Out) -> Result<(), Fail> {
        let size = self.entries[i].size;
        match out {
            Out::Mem => {
                let mut sink = Sink::Mem(Vec::new());
                copy_out(&self.entries[i], part, &mut sink)?;
                if let Sink::Mem(v) = sink { self.mem[i] = Some(v); }
                Ok(())
            }
            Out::Stores(f) => {
                let name = self.names[i].clone();
                if size <= SMALL_FILE {
                    let mut sink = Sink::Mem(Vec::with_capacity(size as usize));
                    copy_out(&self.entries[i], part, &mut sink)?;
                    let Sink::Mem(v) = sink else { unreachable!() };
                    return write_store(&mut **f, &name, &v);
                }
                copy_out(&self.entries[i], part, &mut Sink::Null)?;
                let mut sink = Sink::Store(f.create(&name).map_err(Fail::Write)?);
                let r = copy_out(&self.entries[i], part, &mut sink).and_then(|()| match &mut sink { Sink::Store(s) => s.set_len(size).map_err(Fail::Write), _ => Ok(()) });
                drop(sink);
                if r.is_err() { f.remove(&name); }
                r
            }
            Out::Dir(root) => {
                if self.dir.is_none() { self.dir = Some(new_folder(root, &self.name).map_err(Fail::Write)?); }
                let dir = self.dir.clone().unwrap();
                let tmp = root.join(fit(&dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(), ".lbtmp"));
                let r = (|| {
                    let mut w = io::BufWriter::new(File::create(&tmp).map_err(Fail::Write)?);
                    if size <= SMALL_FILE {
                        let mut sink = Sink::Mem(Vec::with_capacity(size as usize));
                        copy_out(&self.entries[i], part, &mut sink)?;
                        if let Sink::Mem(v) = sink { w.write_all(&v).map_err(Fail::Write)?; }
                    } else {
                        let mut sink = Sink::File(w);
                        copy_out(&self.entries[i], part, &mut sink)?;
                        let Sink::File(f) = sink else { unreachable!() };
                        w = f;
                    }
                    w.into_inner().map_err(|e| Fail::Write(e.into_error()))?;
                    self.place(i, &tmp, &dir)
                })();
                let _ = std::fs::remove_file(&tmp);
                r
            }
        }
    }
    /// Gives the verified temporary file `tmp` file `i`'s name in `dir`, never over another file.
    fn place(&mut self, i: usize, tmp: &Path, dir: &Path) -> Result<(), Fail> {
        loop {
            let target = dir.join(&self.names[i]);
            match std::fs::hard_link(tmp, &target) {
                Ok(()) => return Ok(()),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
                // no hard links on this storage (FAT, some shared storage): moved in, unless the name is taken
                Err(_) if !target.exists() => return std::fs::rename(tmp, &target).map_err(Fail::Write),
                Err(_) => {}
            }
            self.names[i] = self.namer.add(&self.entries[i].name);
        }
    }
    /// File `i` as the status shows it.
    fn file_state(&self, i: usize, t: Option<&Transfer>) -> FileState {
        let e = &self.entries[i];
        FileState { name: self.names[i].clone(), size: e.size, offset: e.offset, stored_size: e.stored_size, sha256: matches!(self.slots[i], Slot::Received).then_some(e.sha256), state: match &self.slots[i] {
            Slot::Received => FileStage::Received,
            Slot::Failed(why) => FileStage::Failed(why.clone()),
            Slot::Pending { .. } => match t.map_or(0.0, |t| decoded_share(t, e)) { p if p <= 0.0 => FileStage::Waiting, p => FileStage::Receiving(p.min(0.99) as f32) },
        } }
    }
    fn status(&self, t: Option<&Transfer>) -> BatchStatus {
        let files = (0..self.entries.len()).map(|i| self.file_state(i, t)).collect();
        BatchStatus { name: self.name.clone(), total: self.entries.len() as u32, received: self.received, failed: self.failed, files, error: self.error.clone(), unit_bytes: t.and_then(|t| t.manifest.as_ref()).map_or(0, unit_bytes) }
    }
}

/// The bytes one finish-word unit spans: a segment when the stream has two or more, else a chunk.
fn unit_bytes(m: &Manifest) -> u64 { m.finish_unit_bytes() }

/// The share of `e`'s stored bytes decoded so far: whole chunks, and the solved blocks of chunks under way.
fn decoded_share(t: &Transfer, e: &Entry) -> f64 {
    if e.stored_size == 0 { return 0.0; }
    let have = Have { seg: &t.segment_done, chunks: &t.chunk_done, live: &t.chunks };
    let (end, chunk) = (e.offset + e.stored_size, super::CHUNK_BYTES);
    let mut got = 0.0;
    for g in e.offset / chunk..=(end - 1) / chunk {
        let bytes = (((g + 1) * chunk).min(end) - (g * chunk).max(e.offset)) as f64;
        if have.chunk(g) { got += bytes; }
        else if let Some(d) = t.chunks.get(&chunk_at(g)) { got += bytes * d.solved_blocks() as f64 / d.k().max(1) as f64; }
    }
    got / e.stored_size as f64
}

/// Removes the stream's part (its file or its store).
fn remove_part(t: &mut Transfer, stores: Option<&mut Box<dyn StoreFactory>>) {
    match t.part.take() {
        Some(Part::Disk(f)) => { drop(f); if let Some(p) = &t.part_path { let _ = std::fs::remove_file(p); } }
        Some(Part::Ext { store, name, .. }) => { drop(store); if let Some(s) = stores { s.remove(&name); } }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // a stream that unpacks past the file size its manifest declares (a hostile sender's zip bomb: a few KB of
    // zstd for gigabytes): the copy stops one byte past the size, never filling memory or storage first
    #[cfg(feature = "zstd")]
    #[test]
    fn unpacking_stops_past_the_declared_size() {
        let big = vec![0u8; 8 << 20];
        let stream = zstd::bulk::compress(&big, 19).unwrap();
        assert!(stream.len() < 4096, "{}", stream.len());
        let mut seen = 0u64;
        let fits = copy_capped(&mut decoder(&mut &stream[..], 27).unwrap(), 1000, |b| { seen += b.len() as u64; Ok(()) }).unwrap();
        assert!(!fits && seen <= 1 << 20, "stopped after {seen} bytes");
        let mut all = 0u64;
        assert!(copy_capped(&mut decoder(&mut &stream[..], 27).unwrap(), big.len() as u64, |b| { all += b.len() as u64; Ok(()) }).unwrap());
        assert_eq!(all, big.len() as u64);
        assert!(decompress_capped(&stream, 1000).is_err());
        assert_eq!(decompress_capped(&stream, big.len() as u64).unwrap().len(), big.len());
    }
    use crate::flat::{Alphabet, CellFormat, FlatSender, FlatSenderConfig};
    use crate::header::FrameType;
    fn dirs(tag: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let base = std::env::temp_dir().join(format!("lensbolt_recv_{tag}"));
        let _ = std::fs::remove_dir_all(&base);
        let (i, o) = (base.join("in"), base.join("out"));
        std::fs::create_dir_all(&i).unwrap(); std::fs::create_dir_all(&o).unwrap();
        (i, o)
    }
    fn write_random(p: &std::path::Path, n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed; let d: Vec<u8> = (0..n).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s as u8 }).collect();
        std::fs::write(p, &d).unwrap(); d
    }
    fn cfg() -> FlatSenderConfig { FlatSenderConfig { width: 1024, height: 1024, format: CellFormat { pitch: 4, alphabet: Alphabet::C8 }, manifest_every_frames: 20, transfer_id: 9, calibration: false, repair_offset: None } }

    #[test]
    fn perfect_channel_completes_in_one_pass() {
        let (i, o) = dirs("perfect");
        let data = write_random(&i.join("a.bin"), 3_000_000, 1);
        let mut s = FlatSender::new(&i.join("a.bin"), cfg(), None).unwrap();
        let mut r = Receiver::new(&o).unwrap();
        let mut out = None;
        for n in 0..s.frames_per_pass() + 5 {
            let (h, ps) = s.next_packets().unwrap();
            match r.feed_packets(&h, &ps).unwrap() { FeedOutcome::Complete(p) => { out = Some(p); break; } FeedOutcome::ManifestReceived => assert_eq!(n, 0), _ => {} }
        }
        let p = out.expect("completed");
        assert_eq!(std::fs::read(&p).unwrap(), data);
        assert_eq!(p.file_name().unwrap(), "a.bin");
        assert!(r.is_complete());
        assert_eq!(r.live_decoders(), 0);
    }
    /// Test stores: one temp file per store name; counts the largest single write (no whole-file buffers).
    struct TempStores { dir: std::path::PathBuf, largest_write: std::sync::Arc<std::sync::atomic::AtomicUsize> }
    struct TempStore { f: File, largest_write: std::sync::Arc<std::sync::atomic::AtomicUsize> }
    impl PartStore for TempStore {
        fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> { self.f.seek(SeekFrom::Start(offset))?; self.f.read_exact(buf) }
        fn write_at(&mut self, offset: u64, data: &[u8]) -> io::Result<()> {
            self.largest_write.fetch_max(data.len(), std::sync::atomic::Ordering::Relaxed);
            self.f.seek(SeekFrom::Start(offset))?; self.f.write_all(data)
        }
        fn set_len(&mut self, len: u64) -> io::Result<()> { self.f.set_len(len) }
    }
    impl StoreFactory for TempStores {
        fn create(&mut self, name: &str) -> io::Result<Box<dyn PartStore>> {
            let f = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(self.dir.join(name))?;
            Ok(Box::new(TempStore { f, largest_write: self.largest_write.clone() }))
        }
        fn remove(&mut self, name: &str) { let _ = std::fs::remove_file(self.dir.join(name)); }
    }

    #[test]
    fn store_backed_receiver_writes_through_the_factory() {
        // the browser's OPFS: the stream and the decompressed file live in stores, never whole in memory
        for (name, data, compress) in [("big.bin", (0..2_500_000u32).map(|i| (i.wrapping_mul(2654435761) >> 11) as u8).collect::<Vec<u8>>(), Some(false)),
                                       ("log.txt", b"lensbolt streams to disk ".repeat(60_000), Some(true))] {
            let (i, o) = dirs(&format!("store_{name}"));
            std::fs::write(i.join(name), &data).unwrap();
            let largest = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let mut s = FlatSender::new(&i.join(name), cfg(), compress).unwrap();
            let mut r = Receiver::with_stores(Box::new(TempStores { dir: o.clone(), largest_write: largest.clone() }));
            let mut done = false;
            for _ in 0..2 * s.frames_per_pass() + 5 {
                let (h, ps) = s.next_packets().unwrap();
                if let FeedOutcome::Complete(p) = r.feed_packets(&h, &ps).unwrap() { assert_eq!(p.file_name().unwrap(), name); done = true; break; }
            }
            assert!(done, "{name}: did not complete");
            assert!(r.take_file().is_none(), "{name}: a store-backed receiver keeps nothing in memory");
            let (shown, store) = r.stored_file().expect("the store holding the verified file");
            assert_eq!(shown, name);
            assert!(std::fs::read(o.join(&store)).unwrap() == data, "{name}: stored content differs");
            assert!(largest.load(std::sync::atomic::Ordering::Relaxed) <= 1 << 20, "{name}: writes stay chunk-sized");
            if compress == Some(true) { assert!(!o.join(format!("{store}.part")).exists() || store.ends_with(".out"), "the compressed stream's part is removed"); }
        }
    }
    #[test]
    fn a_store_that_cannot_be_written_is_an_error_not_a_panic() {
        // OPFS when the phone is full: the write fails; the receiver says so (a panic would trap the browser's
        // WebAssembly instance for good)
        struct Full;
        impl PartStore for Full {
            fn read_at(&mut self, _: u64, buf: &mut [u8]) -> io::Result<()> { buf.fill(0); Ok(()) }
            fn write_at(&mut self, _: u64, _: &[u8]) -> io::Result<()> { Err(io::Error::other("storage is full")) }
            fn set_len(&mut self, _: u64) -> io::Result<()> { Ok(()) }
        }
        struct FullStores;
        impl StoreFactory for FullStores {
            fn create(&mut self, _: &str) -> io::Result<Box<dyn PartStore>> { Ok(Box::new(Full)) }
            fn remove(&mut self, _: &str) {}
        }
        let (i, _) = dirs("full_store");
        write_random(&i.join("f.bin"), 300_000, 5);
        let mut s = FlatSender::new(&i.join("f.bin"), cfg(), Some(false)).unwrap();
        let mut r = Receiver::with_stores(Box::new(FullStores));
        let mut err = None;
        for _ in 0..s.frames_per_pass() + 5 {
            let (h, ps) = s.next_packets().unwrap();
            if let Err(e) = r.feed_packets(&h, &ps) { err = Some(e); break; }
        }
        let e = err.expect("a failed store write must come back as an error");
        assert!(e.to_string().contains("storage is full"), "{e}");
    }
    #[test]
    fn decompress_reads_a_zstd_frame() {
        // "lensbolt in a browser " x 200 compressed by Python's compression.zstd (level 19): the in-memory path
        // must decode it with the zstd crate and, in builds without it (WebAssembly), with ruzstd
        let frame: [u8; 40] = [40,181,47,253,96,48,16,245,0,0,176,108,101,110,115,98,111,108,116,32,105,110,32,97,32,98,114,111,119,115,101,114,32,1,0,46,34,251,217,4];
        assert_eq!(decompress(&frame).unwrap(), b"lensbolt in a browser ".repeat(200));
    }
    #[test]
    fn in_memory_receiver_hands_out_the_file() {
        // the browser receiver has no file system: the part lives in memory and the verified file is taken out
        for (name, data, compress) in [("m.bin", (0..700_000u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect::<Vec<u8>>(), Some(false)),
                                       ("notes.txt", b"lensbolt in a browser ".repeat(20_000), Some(true))] {
            let (i, _) = dirs(&format!("mem_{name}"));
            std::fs::write(i.join(name), &data).unwrap();
            let mut s = FlatSender::new(&i.join(name), cfg(), compress).unwrap();
            let mut r = Receiver::in_memory();
            let mut done = false;
            for _ in 0..s.frames_per_pass() + 5 {
                let (h, ps) = s.next_packets().unwrap();
                if let FeedOutcome::Complete(p) = r.feed_packets(&h, &ps).unwrap() { assert_eq!(p.file_name().unwrap(), name); done = true; break; }
            }
            assert!(done, "{name}: did not complete");
            let (got_name, got) = r.take_file().expect("the verified file");
            assert_eq!(got_name, name);
            assert!(got == data, "{name}: content differs");
            assert!(r.take_file().is_none(), "taken once");
        }
    }
    #[test]
    fn packets_before_the_first_manifest_are_kept() {
        // a receiver that starts mid-stream sees data frames before its first manifest (one every 20 frames
        // here, 31 in practice: MANIFEST_EVERY_FRAMES): they are buffered and used once the manifest arrives, so a perfect channel
        // still completes within one pass of frames counted from the start of the stream
        let (i, o) = dirs("premanifest");
        let data = write_random(&i.join("c.bin"), 3_000_000, 3);
        let mut s = FlatSender::new(&i.join("c.bin"), cfg(), None).unwrap();
        let mut r = Receiver::new(&o).unwrap();
        let _ = s.next_packets().unwrap(); // frame 0, the first manifest, is missed
        let mut out = None;
        for _ in 1..s.frames_per_pass() + 1 {
            let (h, ps) = s.next_packets().unwrap();
            if let FeedOutcome::Complete(p) = r.feed_packets(&h, &ps).unwrap() { out = Some(p); break; }
        }
        let p = out.expect("completed within one pass despite starting after the manifest");
        assert_eq!(std::fs::read(&p).unwrap(), data);
    }
    #[test]
    fn lossy_channel_completes_with_repair_and_ignores_other_transfers() {
        let (i, o) = dirs("lossy");
        let data = write_random(&i.join("b.bin"), 20 * 1024 * 1024 + 777, 2);
        let mut s = FlatSender::new(&i.join("b.bin"), cfg(), None).unwrap();
        let mut other = FlatSender::new(&i.join("b.bin"), FlatSenderConfig { transfer_id: 77, ..cfg() }, None).unwrap();
        let mut r = Receiver::new(&o).unwrap();
        let mut seed = 5u64; let mut done = None; let mut frames = 0u64;
        while done.is_none() && frames < s.frames_per_pass() * 4 {
            frames += 1;
            let (h, ps) = s.next_packets().unwrap();
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            if h.frame_type == FrameType::Data && seed % 10 < 3 { continue; }
            let kept: Vec<_> = ps.into_iter().filter(|_| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed % 10 >= 2 }).collect();
            if let FeedOutcome::Complete(p) = r.feed_packets(&h, &kept).unwrap() { done = Some(p); }
            let (oh, ops) = other.next_packets().unwrap();
            // (once ours completes, the receiver may lock on to the next transfer it sees)
            if done.is_none() { assert_eq!(r.feed_packets(&oh, &ops).unwrap(), FeedOutcome::Ignored, "frame {frames}: the other transfer must be ignored"); }
        }
        let p = done.expect("completed under loss");
        assert_eq!(std::fs::read(&p).unwrap(), data);
        assert!(frames < s.frames_per_pass() * 3, "took {frames} frames");
        assert!(r.progress().packets_ignored > 0);
    }
    #[test]
    fn compressed_file_is_restored_and_name_collision_avoided() {
        let (i, o) = dirs("zstd");
        let text: Vec<u8> = b"compress me please ".iter().cycle().take(2_500_000).copied().collect();
        std::fs::write(i.join("notes.txt"), &text).unwrap();
        std::fs::write(o.join("notes.txt"), b"existing").unwrap();
        let mut s = FlatSender::new(&i.join("notes.txt"), cfg(), None).unwrap();
        assert!(s.manifest().compressed);
        let mut r = Receiver::new(&o).unwrap();
        let mut out = None;
        for _ in 0..s.frames_per_pass() + 5 { let (h, ps) = s.next_packets().unwrap(); if let FeedOutcome::Complete(p) = r.feed_packets(&h, &ps).unwrap() { out = Some(p); break; } }
        let p = out.unwrap();
        assert_eq!(p.file_name().unwrap(), "notes (2).txt");
        assert_eq!(std::fs::read(&p).unwrap(), text);
        assert_eq!(std::fs::read(o.join("notes.txt")).unwrap(), b"existing");
        assert!(!o.join("notes (2).txt.lbpart").exists());
    }
    #[test]
    fn hostile_filename_cannot_escape_out_dir() {
        assert_eq!(safe_filename("../../etc/passwd"), "passwd");
        assert_eq!(safe_filename("..\\..\\x\\startup.cmd"), "startup.cmd");
        assert_eq!(safe_filename("..."), "file");
        assert_eq!(safe_filename(""), "file");
        assert_eq!(safe_filename("a:b*c?.txt"), "abc.txt");
        assert_eq!(safe_filename("notes.txt"), "notes.txt");
    }
    #[test]
    fn completed_transfer_is_not_restarted_by_the_same_broadcast() {
        let (i, o) = dirs("restart");
        let data = write_random(&i.join("d.bin"), 300_000, 4);
        let mut s = FlatSender::new(&i.join("d.bin"), cfg(), None).unwrap();
        let mut r = Receiver::new(&o).unwrap();
        let mut done = false;
        for _ in 0..s.frames_per_pass() * 3 {
            let (h, ps) = s.next_packets().unwrap();
            match r.feed_packets(&h, &ps).unwrap() {
                FeedOutcome::Complete(_) => { assert!(!done, "completed twice"); done = true; }
                FeedOutcome::Ignored if done => {}
                other => assert!(!done, "after completion got {other:?}"),
            }
        }
        assert!(done);
        assert_eq!(std::fs::read(o.join("d.bin")).unwrap(), data);
        assert!(!o.join("d (2).bin").exists());
    }
    /// A 2 px b/w sender (the content transfer id) of `len` noise bytes: packets only, no rendering.
    fn bw2(name: &str, len: usize, seed: u64) -> FlatSender {
        let mut s = seed; let d: Vec<u8> = (0..len).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s as u8 }).collect();
        FlatSender::from_bytes(name, d, FlatSenderConfig { format: CellFormat { pitch: 2, alphabet: Alphabet::Bw }, ..FlatSenderConfig::default() }).unwrap()
    }
    fn is_repair(m: &Manifest, p: &Packet) -> bool { p.id.segment != 0 && p.id.pid as usize >= chunk_blocks(m.chunk_len(p.id.segment, p.id.chunk)) }
    /// Feeds frames `from..to` of `s` (after `edit`, which may drop or change packets and returns false to drop
    /// the frame) and checks the words after each fed frame: none before the first repair packet, and once they
    /// appear, on every later frame, naming exactly `want`. Returns (first frame with a repair packet, first
    /// frame with words, frames fed from then on).
    fn words_after_repair(s: &mut FlatSender, r: &mut Receiver, from: u64, to: u64, want: &[u32], mut edit: impl FnMut(u64, &FrameType, &mut Vec<Packet>) -> bool) -> (Option<u64>, Option<u64>, u64) {
        let (m, n_units) = (s.manifest().clone(), s.manifest().finish_units());
        let (mut repair, mut words, mut with_words) = (None, None, 0);
        for n in 0..to {
            let (h, mut ps) = s.next_packets().unwrap();
            if n < from || !edit(n, &h.frame_type, &mut ps) { continue; }
            if repair.is_none() && ps.iter().any(|p| is_repair(&m, p)) { repair = Some(n); }
            r.feed_packets(&h, &ps).unwrap();
            // the parts the words name, for the line under them (from 1, as the sender says them)
            assert_eq!(r.finish_parts(), r.finish_words().map(|_| crate::finish::parts_line(want)), "frame {n}");
            match r.finish_words() {
                None => assert!(words.is_none(), "frame {n}: the words went away ({:?})", r.progress()),
                Some(w) => {
                    assert!(repair.is_some(), "frame {n}: words \"{w}\" before any repair packet");
                    assert_eq!(crate::finish::decode(h.transfer_id, n_units, &w), Ok(want.to_vec()), "frame {n}: \"{w}\"");
                    words.get_or_insert(n);
                    with_words += 1;
                }
            }
        }
        (repair, words, with_words)
    }
    #[test]
    fn finish_words_name_the_short_chunk_of_a_one_segment_file() {
        // six chunk units; chunk 3's first-pass blocks never arrive. Chunk 5 (half a chunk) completes halfway
        // through the pass: complete and short units then, but no words before repair
        let mut s = bw2("one.bin", 5 * 1024 * 1024 + 512 * 1024, 11);
        let (m, pass) = (s.manifest().clone(), s.frames_per_pass());
        assert_eq!(m.finish_units(), 6);
        let mut r = Receiver::in_memory();
        let (repair, words, with) = words_after_repair(&mut s, &mut r, 0, pass + 10, &[3], |_, _, ps| { ps.retain(|p| !(p.id.chunk == 3 && p.id.segment == 1 && !is_repair(&m, p))); true });
        assert!(repair.is_some() && words == repair, "repair from frame {repair:?}, words from {words:?}");
        assert!(with >= 10, "words on {with} frames");
    }
    #[test]
    fn finish_words_name_a_segment_whose_hash_failed() {
        // segment 1's first-pass blocks are corrupted (one byte of block 5 of every chunk): its chunks all decode,
        // its hash fails and it starts again; segment 2 verifies. The words name segment 1 (unit 0) only
        let mut s = bw2("two.bin", 18 << 20, 12);
        let pass = s.frames_per_pass();
        let mut r = Receiver::in_memory();
        let (repair, words, with) = words_after_repair(&mut s, &mut r, 0, pass + 10, &[0], |_, _, ps| {
            for p in ps.iter_mut().filter(|p| p.id.segment == 1 && p.id.pid == 5) { p.payload[0] ^= 0xFF; }
            true
        });
        assert!(repair.is_some() && words == repair, "repair from frame {repair:?}, words from {words:?}");
        assert_eq!((r.hash_failures(), r.progress().segments_done), (1, 1));
        assert!(with >= 5, "words on {with} frames");
    }
    #[test]
    fn no_finish_words_while_nothing_is_complete() {
        // a receiver that joins after the first pass sees only repair frames (and manifests): repair, but no
        // complete unit to skip, so no words
        let mut s = bw2("late.bin", 18 << 20, 13);
        let pass = s.frames_per_pass();
        let mut r = Receiver::in_memory();
        let (repair, words, _) = words_after_repair(&mut s, &mut r, pass, pass + 70, &[], |_, _, _| true);
        assert!(repair.is_some() && words.is_none(), "repair from frame {repair:?}, words from {words:?}");
        let t = r.t.as_ref().unwrap();
        assert!(t.repair_seen && t.manifest.is_some(), "the test must reach repair with the manifest known");
        assert_eq!(r.progress().segments_done, 0);
    }
    #[test]
    fn repair_packets_before_the_manifest_count_once_it_arrives() {
        // every manifest frame is missed until repair has run for a while: the data (repair packets included)
        // waits in the pending buffer, no words without the manifest, then the right words at once
        let mut s = bw2("pending.bin", 5 * 1024 * 1024 + 512 * 1024, 14);
        let (m, pass) = (s.manifest().clone(), s.frames_per_pass());
        let mut r = Receiver::in_memory();
        let (mut repair_frames, mut manifest_at) = (0, None);
        let (repair, words, with) = words_after_repair(&mut s, &mut r, 0, pass + 70, &[3], |n, ft, ps| {
            if *ft == FrameType::Manifest { if repair_frames < 5 { return false; } manifest_at.get_or_insert(n); return true; }
            if ps.iter().any(|p| is_repair(&m, p)) { repair_frames += 1; }
            ps.retain(|p| !(p.id.chunk == 3 && !is_repair(&m, p)));
            true
        });
        let at = manifest_at.expect("a manifest frame after repair started");
        assert!(repair.is_some_and(|f| f < at), "repair from frame {repair:?}, the manifest at {at}");
        assert_eq!((words, with), (Some(at), pass + 70 - at), "words from the manifest frame on");
        assert_eq!(r.progress().segments_done, 0);
    }
    #[test]
    fn progress_reports_segments() {
        let (i, o) = dirs("progress");
        write_random(&i.join("c.bin"), 17 * 1024 * 1024, 3);
        let mut s = FlatSender::new(&i.join("c.bin"), cfg(), None).unwrap();
        let mut r = Receiver::new(&o).unwrap();
        assert!(!r.progress().have_manifest);
        let (h, ps) = s.next_packets().unwrap(); r.feed_packets(&h, &ps).unwrap();
        let pr = r.progress();
        assert_eq!((pr.have_manifest, pr.segments_total, pr.segments_done), (true, 2, 0));
        assert_eq!(pr.filename.as_deref(), Some("c.bin"));
        // segment 2's packets are withheld: the pass spreads both segments over its whole length, so with
        // everything received they would verify in the same last frames
        let mut seen_seg1 = false;
        for _ in 0..s.frames_per_pass() {
            let (h, mut ps) = s.next_packets().unwrap();
            ps.retain(|p| p.id.segment != 2);
            if r.feed_packets(&h, &ps).unwrap() == FeedOutcome::SegmentDone(1) { seen_seg1 = true; break; }
        }
        assert!(seen_seg1);
        assert_eq!(r.progress().segments_done, 1);
        assert_eq!(r.progress().bytes_done, 16 * 1024 * 1024);
    }
}
