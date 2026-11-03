use super::batch::{safe_name, varint_len, BatchError, Entry, Index, MAX_ENTRIES, MAX_INDEX_BYTES};
use super::manifest::{BatchInfo, Manifest, TuneInfo, SEGMENT_BYTES};
use super::sniff::{should_compress, SNIFF_BYTES};
use crate::fountain::BLOCK_BYTES;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Per-segment SHA-256 accumulator over the stream as it is produced.
struct SegHash { cur: Sha256, in_seg: u64, out: Vec<[u8; 32]>, total: u64 }
impl SegHash {
    fn new() -> Self { SegHash { cur: Sha256::new(), in_seg: 0, out: Vec::new(), total: 0 } }
    fn push(&mut self, mut d: &[u8]) {
        while !d.is_empty() {
            let room = (SEGMENT_BYTES - self.in_seg) as usize;
            let n = room.min(d.len());
            self.cur.update(&d[..n]); self.in_seg += n as u64; self.total += n as u64; d = &d[n..];
            if self.in_seg == SEGMENT_BYTES { self.out.push(std::mem::take(&mut self.cur).finalize().into()); self.in_seg = 0; }
        }
    }
    fn finish(mut self) -> (Vec<[u8; 32]>, u64) { if self.in_seg > 0 { self.out.push(self.cur.finalize().into()); } (self.out, self.total) }
}

/// A file's hashes as a manifest of it sent as-is carries them: the whole file's SHA-256 and each 16 MiB segment's,
/// and the bytes they cover (`FileHasher`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileHashes { pub file: [u8; 32], pub segments: Vec<[u8; 32]>, pub len: u64 }
/// The hashes of a file fed in pieces as it is read (the browser reads a file a few MiB at a time): one pass over the
/// bytes, the same values `Prepared::from_shared` computes. `Prepared::from_shared_hashed` takes them.
pub struct FileHasher { file: Sha256, seg: SegHash }
impl Default for FileHasher { fn default() -> Self { Self::new() } }
impl FileHasher {
    pub fn new() -> FileHasher { FileHasher { file: Sha256::new(), seg: SegHash::new() } }
    /// The next bytes of the file, in order.
    pub fn update(&mut self, d: &[u8]) { self.file.update(d); self.seg.push(d); }
    pub fn finish(self) -> FileHashes { let (segments, len) = self.seg.finish(); FileHashes { file: self.file.finalize().into(), segments, len } }
}

#[cfg(feature = "zstd")]
struct Tee<'a> { w: io::BufWriter<File>, seg: &'a mut SegHash }
#[cfg(feature = "zstd")]
impl<'a> Write for Tee<'a> {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> { self.seg.push(b); self.w.write_all(b)?; Ok(b.len()) }
    fn flush(&mut self) -> io::Result<()> { self.w.flush() }
}

/// A file prepared for sending: manifest + a seekable stream (the file itself, a zstd temp file, or a
/// byte buffer when there is no filesystem, as in a browser).
pub struct Prepared { pub manifest: Manifest, stream: Stream, pub temp_path: Option<PathBuf>, pub index: Option<Index> }

enum Stream { File(File), Memory(Vec<u8>), Shared(Arc<Vec<u8>>), Batch(Box<BatchStream>) }

impl Drop for Prepared { fn drop(&mut self) { if let Some(p) = &self.temp_path { let _ = std::fs::remove_file(p); } } }

impl Prepared {
    /// One streaming pass: SHA-256 of the file; sniff; if compressing, write zstd-3 to a temp file next to
    /// `path`; per-segment SHA-256 over the stream. Memory: one 1 MiB buffer.
    pub fn new(path: &Path, force_compress: Option<bool>) -> io::Result<Prepared> {
        let filename = path.file_name().and_then(|s| s.to_str()).unwrap_or("file").to_string();
        let file_size = std::fs::metadata(path)?.len();
        let mut f = File::open(path)?;
        let compress = match force_compress {
            Some(v) => v,
            None => { let mut s = vec![0u8; SNIFF_BYTES.min(file_size as usize)]; f.read_exact(&mut s)?; f.seek(SeekFrom::Start(0))?; should_compress(&s) }
        };
        let mut file_hash = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        let mut seg = SegHash::new();
        #[cfg(not(feature = "zstd"))]
        let compress = { let _ = compress; false };
        let (stream, temp_path) = if compress {
            let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(0);
            let tmp = path.with_extension(format!("lbz.{}.{:08x}.tmp", std::process::id(), nonce));
            let out = File::create(&tmp)?;
            {
                #[cfg(feature = "zstd")]
                {
                    let mut enc = zstd::stream::Encoder::new(Tee { w: io::BufWriter::new(out), seg: &mut seg }, 3)?;
                    loop { let n = f.read(&mut buf)?; if n == 0 { break; } file_hash.update(&buf[..n]); enc.write_all(&buf[..n])?; }
                    enc.finish()?.flush()?;
                }
                #[cfg(not(feature = "zstd"))]
                { let _ = (out, &mut seg, &mut file_hash, &mut buf, &mut f); }
            }
            (File::open(&tmp)?, Some(tmp))
        } else {
            loop { let n = f.read(&mut buf)?; if n == 0 { break; } file_hash.update(&buf[..n]); seg.push(&buf[..n]); }
            (File::open(path)?, None)
        };
        let (segment_sha256, stream_size) = seg.finish();
        if stream_size > u16::MAX as u64 * SEGMENT_BYTES { return Err(io::Error::other("file too large for one transfer")); }
        let manifest = Manifest { filename, file_size, stream_size, compressed: compress, file_sha256: file_hash.finalize().into(), segment_sha256, batch: None, tune: None };
        Ok(Prepared { manifest, stream: Stream::File(stream), temp_path, index: None })
    }
    /// A file held in memory (the browser sender): sent as-is, never compressed.
    pub fn from_bytes(filename: &str, data: Vec<u8>) -> io::Result<Prepared> {
        let manifest = memory_manifest(filename, &data)?;
        Ok(Prepared { manifest, stream: Stream::Memory(data), temp_path: None, index: None })
    }
    /// A file held in memory and shared (the browser sender keeps each chosen file once, and every sender made for it
    /// reads the same bytes): prepared and sent exactly as `from_bytes` would, without a copy.
    pub fn from_shared(filename: &str, data: Arc<Vec<u8>>) -> io::Result<Prepared> {
        let manifest = memory_manifest(filename, &data)?;
        Ok(Prepared { manifest, stream: Stream::Shared(data), temp_path: None, index: None })
    }
    /// `from_shared` with the file's hashes already computed as it was read (`FileHasher`): the bytes are not read
    /// again. Hashes of another length than the bytes are not trusted: computed from the bytes.
    pub fn from_shared_hashed(filename: &str, data: Arc<Vec<u8>>, hashes: FileHashes) -> io::Result<Prepared> {
        let file_size = data.len() as u64;
        let fits = hashes.len == file_size && hashes.segments.len() as u64 == file_size.div_ceil(SEGMENT_BYTES);
        if !fits { return Prepared::from_shared(filename, data); }
        if file_size > u16::MAX as u64 * SEGMENT_BYTES { return Err(io::Error::other("file too large for one transfer")); }
        let manifest = Manifest { filename: filename.to_string(), file_size, stream_size: file_size, compressed: false, file_sha256: hashes.file, segment_sha256: hashes.segments, batch: None, tune: None };
        Ok(Prepared { manifest, stream: Stream::Shared(data), temp_path: None, index: None })
    }
    /// The Tune carousel's transfer: the generated payload (`tune::schedule::tune_payload`, the same bytes on every
    /// sender, so the same transfer id), named "LensBolt tune", stored as it is, its manifest saying tune.
    /// `display_60`: the sender's screen runs at 60 or 120 Hz (the 60 fps segments are drawn at 60).
    pub fn tune(display_60: bool) -> io::Result<Prepared> {
        let data = crate::tune::schedule::tune_payload();
        let mut manifest = memory_manifest("LensBolt tune", &data)?;
        manifest.tune = Some(TuneInfo { schedule: crate::tune::schedule::SCHEDULE_VERSION, display_60 });
        Ok(Prepared { manifest, stream: Stream::Memory(data), temp_path: None, index: None })
    }
    pub fn read_chunk(&mut self, seg: u16, chunk: u8) -> io::Result<Vec<u8>> {
        let len = self.manifest.chunk_len(seg, chunk);
        let off = self.manifest.chunk_offset(seg, chunk) as usize;
        match &mut self.stream {
            Stream::Memory(d) => Ok(d[off..off + len].to_vec()),
            Stream::Shared(d) => Ok(d[off..off + len].to_vec()),
            Stream::File(f) => { let mut v = vec![0u8; len]; f.seek(SeekFrom::Start(off as u64))?; f.read_exact(&mut v)?; Ok(v) }
            Stream::Batch(b) => { let mut v = vec![0u8; len]; b.read_at(off as u64, &mut v)?; Ok(v) }
        }
    }
    /// XORs block `i` of a chunk into `out` (a short last block as if zero-padded): the fountain packets are
    /// built from the blocks they need, never from whole chunks.
    pub fn xor_block(&mut self, seg: u16, chunk: u8, i: usize, out: &mut [u8; BLOCK_BYTES]) -> io::Result<()> {
        let n = self.manifest.chunk_len(seg, chunk).saturating_sub(i * BLOCK_BYTES).min(BLOCK_BYTES);
        let off = self.manifest.chunk_offset(seg, chunk) + (i * BLOCK_BYTES) as u64;
        let mut buf = [0u8; BLOCK_BYTES];
        let block: &[u8] = match &mut self.stream {
            Stream::Memory(d) => &d[off as usize..off as usize + n],
            Stream::Shared(d) => &d[off as usize..off as usize + n],
            Stream::File(f) => { read_at(f, off, &mut buf[..n])?; &buf[..n] }
            Stream::Batch(b) => { b.read_at(off, &mut buf[..n])?; &buf[..n] }
        };
        for (o, b) in out.iter_mut().zip(block) { *o ^= *b; }
        Ok(())
    }
}

/// The manifest of one file held in memory, sent as-is (`from_bytes`, `from_shared`).
fn memory_manifest(filename: &str, data: &[u8]) -> io::Result<Manifest> {
    let file_size = data.len() as u64;
    if file_size > u16::MAX as u64 * SEGMENT_BYTES { return Err(io::Error::other("file too large for one transfer")); }
    let file_sha256: [u8; 32] = Sha256::digest(data).into();
    let mut seg = SegHash::new(); seg.push(data);
    let (segment_sha256, stream_size) = seg.finish();
    Ok(Manifest { filename: filename.to_string(), file_size, stream_size, compressed: false, file_sha256, segment_sha256, batch: None, tune: None })
}

/// Where a file of a batch comes from: a path (read when the batch is prepared, and again while it is sent if
/// it is large and sent as it is), bytes in memory, or bytes in memory shared with whoever holds them (the browser
/// keeps each chosen file once: every sender made for the batch reads the same bytes, none copies them; a file
/// worth compressing is the exception, its zstd frame is its own).
pub enum BatchSource { Path(PathBuf), Bytes(Vec<u8>), Shared(Arc<Vec<u8>>),
    /// `Shared` with the file's SHA-256 already computed (`FileHasher` as it was read): not hashed again
    Hashed(Arc<Vec<u8>>, [u8; 32]) }

/// Raw files from a path at least this large are read from their own file while sending; smaller ones (and
/// every compressed file) are copied into the spool, so a chunk of many small files opens none of them.
const IN_PLACE_MIN: u64 = 1 << 20;
/// Spool bytes held in memory before the spool moves to a temp file.
const SPOOL_IN_MEMORY: usize = 64 << 20;
/// Files read in place kept open at once: a folder of a couple of hundred photos stays open whole, under the
/// 256 descriptors macOS allows a process by default (Linux: 1024; Windows: no such limit). Past it see
/// `BatchStream::file`.
const OPEN_FILES: usize = 200;

/// A file's zstd frame is kept only when it saves at least 5% (`should_compress`'s measure on a sample): never
/// when it is no smaller, and a frame that saves less costs the receiver a decompression for little.
#[cfg(feature = "zstd")]
fn worth(stored: u64, size: u64) -> bool { (stored as f64) < size as f64 * 0.95 }

/// Fills `buf` from `f` unless the file ends first; the bytes read. Whole buffers, so a file and the same bytes
/// in memory reach the zstd encoder in the same pieces.
fn read_full(f: &mut File, buf: &mut [u8]) -> io::Result<usize> {
    let mut n = 0;
    while n < buf.len() { let k = f.read(&mut buf[n..])?; if k == 0 { break; } n += k; }
    Ok(n)
}

/// Files shorter than this are never compressed: a zstd frame's header and checksum leave next to nothing to
/// save, and skipping them keeps a batch of thousands of tiny files fast.
#[cfg(feature = "zstd")]
const MIN_COMPRESS: usize = 64;

/// Packs files into their stored bytes: one zstd context, reused for every file of a batch.
struct Packer {
    #[cfg(feature = "zstd")]
    bulk: zstd::bulk::Compressor<'static>,
}

impl Packer {
    fn new() -> io::Result<Packer> {
        Ok(Packer {
            #[cfg(feature = "zstd")]
            bulk: zstd::bulk::Compressor::new(3)?,
        })
    }

    /// The bytes a file of `data.len()` bytes takes in the stream, and whether that is its zstd frame. Under
    /// MIN_COMPRESS bytes: as it is. Up to SNIFF_BYTES: the whole file compressed at once (it is its own
    /// sample). Larger: the first SNIFF_BYTES decide (`should_compress`), then the whole file streams through
    /// the encoder in 1 MiB pieces, as from a path.
    fn pack(&mut self, data: Vec<u8>) -> io::Result<(Vec<u8>, bool)> {
        Ok(match self.compressed(&data)? { Some(c) => (c, true), None => (data, false) })
    }

    /// `data`'s zstd frame when that is the bytes to send (`pack`'s rule), else None.
    fn compressed(&mut self, data: &[u8]) -> io::Result<Option<Vec<u8>>> {
        #[cfg(feature = "zstd")]
        {
            let size = data.len() as u64;
            if data.len() < MIN_COMPRESS {
            } else if data.len() <= SNIFF_BYTES {
                if let Ok(c) = self.bulk.compress(data) { if worth(c.len() as u64, size) { return Ok(Some(c)); } }
            } else if should_compress(&data[..SNIFF_BYTES]) {
                let mut enc = zstd::stream::Encoder::new(Vec::new(), 3)?;
                for piece in data.chunks(1 << 20) { enc.write_all(piece)?; }
                let c = enc.finish()?;
                if worth(c.len() as u64, size) { return Ok(Some(c)); }
            }
        }
        #[cfg(not(feature = "zstd"))]
        let _ = data;
        Ok(None)
    }

    /// A file from a path: its size, SHA-256, stored size, whether compressed, and where its stored bytes are.
    /// The first SNIFF_BYTES + 1 bytes tell a small file (read whole, packed as bytes are) from a large one.
    fn pack_path(&mut self, path: &Path, spool: &mut Spool) -> io::Result<(u64, [u8; 32], u64, bool, Stored)> {
        let mut f = File::open(path)?;
        let mut head = Vec::new();
        (&mut f).take(SNIFF_BYTES as u64 + 1).read_to_end(&mut head)?;
        if head.len() <= SNIFF_BYTES {
            let (size, sha256) = (head.len() as u64, Sha256::digest(&head).into());
            let (stored, compressed) = self.pack(head)?;
            let stored_size = stored.len() as u64;
            let at = spool.len();
            let place = if size == 0 { Stored::Empty }
                else if compressed || size < IN_PLACE_MIN { spool.write_all(&stored)?; Stored::Spool(at) }
                else { Stored::InPlace(path.to_path_buf()) };
            return Ok((size, sha256, stored_size, compressed, place));
        }
        #[cfg(feature = "zstd")]
        let compress = should_compress(&head[..SNIFF_BYTES]);
        drop(head);
        f.seek(SeekFrom::Start(0))?;
        let mut buf = vec![0u8; 1 << 20];
        let (mut hasher, mut size) = (Sha256::new(), 0u64);
        #[cfg(feature = "zstd")]
        { if compress {
            let at = spool.len();
            let mut enc = zstd::stream::Encoder::new(&mut *spool, 3)?;
            loop { let n = read_full(&mut f, &mut buf)?; if n == 0 { break; } hasher.update(&buf[..n]); size += n as u64; enc.write_all(&buf[..n])?; }
            enc.finish()?;
            let stored_size = spool.len() - at;
            if worth(stored_size, size) { return Ok((size, hasher.finalize().into(), stored_size, true, Stored::Spool(at))); }
            spool.truncate(at)?;
            return Ok((size, hasher.finalize().into(), size, false, Stored::InPlace(path.to_path_buf())));
        } }
        loop { let n = read_full(&mut f, &mut buf)?; if n == 0 { break; } hasher.update(&buf[..n]); size += n as u64; }
        Ok((size, hasher.finalize().into(), size, false, Stored::InPlace(path.to_path_buf())))
    }
}

/// Where a file's stored bytes are while the batch is sent.
enum Stored { Empty, Mem(Vec<u8>), Shared(Arc<Vec<u8>>), Spool(u64), InPlace(PathBuf) }

/// Bytes of the stream held in memory: the batch's own (the index, bytes sources, zstd frames) or the caller's.
enum Buf { Own(Vec<u8>), Shared(Arc<Vec<u8>>) }
impl Buf { fn bytes(&self) -> &[u8] { match self { Buf::Own(b) => b, Buf::Shared(b) => b } } }

fn invalid(msg: String) -> io::Error { io::Error::new(io::ErrorKind::InvalidInput, msg) }
/// `e`, saying which file it is about.
fn about(path: &Path, e: io::Error) -> io::Error { io::Error::new(e.kind(), format!("{}: {e}", path.display())) }
fn changed(path: &Path) -> io::Error { io::Error::other(format!("{} changed while preparing", path.display())) }
fn index_too_long(len: u64) -> io::Error {
    invalid(format!("the file list of this batch takes {len} bytes, more than the {} MiB one transfer carries: send fewer files, or files with shorter names", MAX_INDEX_BYTES >> 20))
}

impl Prepared {
    /// Several files as one transfer: the stream is the file index (`batch::Index`), then the files back to back,
    /// smallest first (equal sizes in the order given), each its own zstd frame when that saves at least 5%.
    /// Names are what the receiver writes (`safe_name`); equal names stay (the receiver tells them apart). An
    /// empty `name` names the batch after the first file given ("photo.jpg +11 more"). The manifest's batch
    /// section carries the batch name and the index's length and SHA-256; `file_sha256` is the whole stream's.
    ///
    /// Refused (InvalidInput): no files, more than `MAX_ENTRIES`, an index over `MAX_INDEX_BYTES` (checked
    /// before any file is opened, and again once sizes are known), a stream longer than one transfer carries, a
    /// path that is not a regular file. Errors about a file name its path.
    /// Memory: bytes sources stay in memory as given (or as their smaller zstd frame); from paths, files under
    /// 1 MiB and compressed frames go to a spool (memory up to 64 MiB, then a temp file the system deletes when
    /// it is closed, even if the sender is killed), larger raw files are read from their own file while sending
    /// (up to OPEN_FILES kept open). Every path file is read twice: once here to hash and pack it, once to hash
    /// the segments of the stream laid out; a file read in place is hashed again then, and one that changed
    /// between the two is refused ("... changed while preparing").
    pub fn batch(name: &str, files: Vec<(String, BatchSource)>) -> io::Result<Prepared> {
        Prepared::batch_with(name, files, Limits::new(SPOOL_IN_MEMORY, OPEN_FILES))
    }

    fn batch_with(name: &str, files: Vec<(String, BatchSource)>, limits: Limits) -> io::Result<Prepared> {
        let count = files.len();
        if count == 0 { return Err(invalid("a batch needs at least one file".into())); }
        if count > MAX_ENTRIES { return Err(invalid(format!("{count} files are more than the {MAX_ENTRIES} one batch carries"))); }
        let mut files: Vec<(String, BatchSource)> = files.into_iter().map(|(n, s)| (safe_name(&n), s)).collect();
        // the shortest the index can be (every size one byte, nothing compressed), before any file is opened
        let floor = 5 + varint_len(count as u64) + files.iter().map(|(n, _)| varint_len(n.len() as u64) + n.len() + 1 + 1 + 32).sum::<usize>();
        if floor > MAX_INDEX_BYTES { return Err(index_too_long(floor as u64)); }
        let batch_name = safe_name(&if name.trim().is_empty() {
            if count == 1 { files[0].0.clone() } else { format!("{} +{} more", files[0].0, count - 1) }
        } else { name.to_string() });
        // smallest first; a stable sort keeps equal sizes in the order given
        let sizes = files.iter().map(|(_, s)| match s {
            BatchSource::Path(p) => {
                let m = std::fs::metadata(p).map_err(|e| about(p, e))?;
                if !m.is_file() { return Err(invalid(format!("{} is not a regular file", p.display()))); }
                Ok(m.len())
            }
            BatchSource::Bytes(b) => Ok(b.len() as u64),
            BatchSource::Shared(b) | BatchSource::Hashed(b, _) => Ok(b.len() as u64),
        }).collect::<io::Result<Vec<u64>>>()?;
        let mut order: Vec<usize> = (0..count).collect();
        order.sort_by_key(|&i| sizes[i]);
        let mut slots: Vec<Option<(String, BatchSource)>> = files.drain(..).map(Some).collect();
        // each file: hashed, packed, its stored bytes put where they are read from while sending
        let mut spool = Spool::new(limits.spool_mem);
        let mut packer = Packer::new()?;
        let mut entries = Vec::with_capacity(count);
        let mut stored = Vec::with_capacity(count);
        for i in order {
            let (name, source) = slots[i].take().expect("each file once");
            let (size, sha256, stored_size, compressed, place) = match source {
                BatchSource::Path(p) => packer.pack_path(&p, &mut spool).map_err(|e| about(&p, e))?,
                BatchSource::Bytes(data) => {
                    let (size, sha256) = (data.len() as u64, Sha256::digest(&data).into());
                    let (bytes, compressed) = packer.pack(data)?;
                    (size, sha256, bytes.len() as u64, compressed, if bytes.is_empty() { Stored::Empty } else { Stored::Mem(bytes) })
                }
                // read from the caller's bytes, unless its zstd frame is worth sending
                BatchSource::Shared(_) | BatchSource::Hashed(..) => {
                    let (data, sha256) = match source { BatchSource::Hashed(d, h) => (d, h), BatchSource::Shared(d) => { let h = Sha256::digest(&*d).into(); (d, h) } _ => unreachable!() };
                    let size = data.len() as u64;
                    match packer.compressed(&data)? {
                        Some(c) => (size, sha256, c.len() as u64, true, Stored::Mem(c)),
                        None => (size, sha256, size, false, if data.is_empty() { Stored::Empty } else { Stored::Shared(data) }),
                    }
                }
            };
            entries.push(Entry { name, size, stored_size, compressed, offset: 0, sha256 });
            stored.push(place);
        }
        spool.done()?;
        // laid out back to back after the index, then checked as a receiver would read it
        let mut index = Index { entries };
        let index_len = index.encoded_len() as u64;
        let mut at = index_len;
        for e in &mut index.entries { e.offset = at; at = at.saturating_add(e.stored_size); }
        index.validate().map_err(|e| match e {
            BatchError::IndexTooLong(n) => index_too_long(n),
            BatchError::PastStream { .. } => invalid("the files are too large for one transfer".into()),
            e => invalid(format!("the batch cannot be sent: {e}")),
        })?;
        let index_bytes = index.encode();
        let index_sha256: [u8; 32] = Sha256::digest(&index_bytes).into();
        let mut pieces = vec![Piece { start: 0, len: index_len, src: Src::Mem(0) }];
        let (mut bufs, mut paths, mut in_place_sha) = (vec![Buf::Own(index_bytes)], Vec::new(), Vec::new());
        for (e, place) in index.entries.iter().zip(stored) {
            let src = match place {
                Stored::Empty => continue,
                Stored::Mem(b) => { bufs.push(Buf::Own(b)); Src::Mem(bufs.len() - 1) }
                Stored::Shared(b) => { bufs.push(Buf::Shared(b)); Src::Mem(bufs.len() - 1) }
                Stored::Spool(at) => Src::Spool(at),
                Stored::InPlace(p) => { paths.push(p); in_place_sha.push(e.sha256); Src::File(paths.len() - 1) }
            };
            pieces.push(Piece { start: e.offset, len: e.stored_size, src });
        }
        let temp_path = spool.path.clone();
        let open = paths.iter().map(|_| None).collect();
        let mut stream = BatchStream { pieces, bufs, spool, paths, open, open_count: 0, open_cap: limits.open_files.max(1), last: 0, opens: 0 };
        #[cfg(test)]
        if let Some(touch) = limits.between_passes { touch(&stream.paths); }
        // the stream as sent: per-segment hashes and the whole stream's. Files read in place are hashed again on
        // the way: one that changed since it was hashed above (its index entry) is refused, not sent
        let stream_size = index.stream_len();
        let (mut seg, mut whole, mut buf) = (SegHash::new(), Sha256::new(), vec![0u8; 1 << 20]);
        for k in 0..stream.pieces.len() {
            let Piece { start, len, src } = stream.pieces[k];
            let mut again = matches!(src, Src::File(_)).then(Sha256::new);
            let mut off = 0u64;
            while off < len {
                let n = (len - off).min(buf.len() as u64) as usize;
                stream.read_at(start + off, &mut buf[..n]).map_err(|e| match src {
                    Src::File(f) if e.kind() == io::ErrorKind::UnexpectedEof => changed(&stream.paths[f]),
                    _ => e,
                })?;
                seg.push(&buf[..n]); whole.update(&buf[..n]);
                if let Some(h) = &mut again { h.update(&buf[..n]); }
                off += n as u64;
            }
            if let (Some(h), Src::File(f)) = (again, src) {
                if <[u8; 32]>::from(h.finalize()) != in_place_sha[f] { return Err(changed(&stream.paths[f])); }
            }
        }
        let (segment_sha256, total) = seg.finish();
        debug_assert_eq!(total, stream_size);
        let manifest = Manifest {
            filename: batch_name.clone(), file_size: stream_size, stream_size, compressed: false, file_sha256: whole.finalize().into(), segment_sha256,
            batch: Some(BatchInfo { name: batch_name, index_len, index_sha256 }), tune: None,
        };
        Ok(Prepared { manifest, stream: Stream::Batch(Box::new(stream)), temp_path, index: Some(index) })
    }
}

/// How much a batch holds in memory and open: the spool's bytes before it moves to a temp file, and the in-place
/// files kept open at once (small values in tests).
#[derive(Clone, Copy)]
struct Limits {
    spool_mem: usize, open_files: usize,
    /// tests: called with the in-place files between the two passes (to change one)
    #[cfg(test)]
    between_passes: Option<fn(&[PathBuf])>,
}
impl Limits {
    fn new(spool_mem: usize, open_files: usize) -> Limits {
        Limits { spool_mem, open_files, #[cfg(test)] between_passes: None }
    }
}

#[cfg(test)]
impl Prepared {
    /// In-place files opened so far (a batch; 0 otherwise).
    pub(crate) fn opens(&self) -> u64 { match &self.stream { Stream::Batch(b) => b.opens, _ => 0 } }
}

/// Bytes of the stream `start .. start + len`, from one place.
#[derive(Clone, Copy)]
struct Piece { start: u64, len: u64, src: Src }
/// `Mem(i)`: `BatchStream::bufs[i]` (the index, bytes sources and zstd frames; shared sources); `Spool(at)`: from `at` in the spool;
/// `File(i)`: `BatchStream::paths[i]` from its start.
#[derive(Clone, Copy)]
enum Src { Mem(usize), Spool(u64), File(usize) }

/// A batch stream: the pieces in stream order, back to back (empty files have none), and the places they are
/// read from; `open[i]`: `paths[i]` while it is open (at most `open_cap` at once), `last` the one read last,
/// `opens` how many times a file was opened.
struct BatchStream { pieces: Vec<Piece>, bufs: Vec<Buf>, spool: Spool, paths: Vec<PathBuf>, open: Vec<Option<File>>, open_count: usize, open_cap: usize, last: usize, opens: u64 }

impl BatchStream {
    /// Exactly `buf.len()` bytes of the stream from `off`, across pieces.
    fn read_at(&mut self, mut off: u64, mut buf: &mut [u8]) -> io::Result<()> {
        let mut i = self.pieces.partition_point(|p| p.start + p.len <= off);
        while !buf.is_empty() {
            let Piece { start, len, src } = *self.pieces.get(i).ok_or_else(|| io::Error::from(io::ErrorKind::UnexpectedEof))?;
            let within = off - start;
            let n = (len - within).min(buf.len() as u64) as usize;
            let (head, rest) = std::mem::take(&mut buf).split_at_mut(n);
            match src {
                Src::Mem(b) => head.copy_from_slice(&self.bufs[b].bytes()[within as usize..within as usize + n]),
                Src::Spool(at) => self.spool.read_at(at + within, head)?,
                Src::File(f) => read_at(self.file(f)?, within, head)?,
            }
            buf = rest; off += n as u64; i += 1;
        }
        Ok(())
    }
    /// `paths[i]`, open. Every in-place file stays open up to `open_cap`. Past it, the file read last makes room:
    /// the pass and the repair rounds visit the chunks in turn, so the files they read cycle, and under a cycle
    /// evicting the least recently used evicts exactly the file needed next (every switch to another file opens
    /// one); evicting the most recently used keeps `open_cap - 1` files open through every cycle, so about
    /// (files - open_cap + 1) / files of the switches open one.
    fn file(&mut self, i: usize) -> io::Result<&mut File> {
        if self.open[i].is_none() {
            let f = File::open(&self.paths[i])?;
            self.opens += 1;
            if self.open_count == self.open_cap { self.open[self.last] = None; } else { self.open_count += 1; }
            self.open[i] = Some(f);
        }
        self.last = i;
        Ok(self.open[i].as_mut().expect("just opened"))
    }
}

/// Stored bytes of small and compressed files from paths, appended while the batch is prepared: in memory up to
/// `cap` bytes, then in a temp file (deleted when the spool is dropped, on success or failure).
struct Spool { mem: Vec<u8>, w: Option<io::BufWriter<File>>, r: Option<File>, path: Option<PathBuf>, len: u64, cap: usize }

impl Spool {
    fn new(cap: usize) -> Spool { Spool { mem: Vec::new(), w: None, r: None, path: None, len: 0, cap } }
    fn len(&self) -> u64 { self.len }
    fn spill(&mut self) -> io::Result<()> {
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(0);
        let path = std::env::temp_dir().join(format!("lensbolt-batch.{}.{:08x}.{}.tmp", std::process::id(), nonce, NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
        let mut options = std::fs::OpenOptions::new();
        options.read(true).write(true).create_new(true);
        // gone when the last handle closes, however the process ends: Windows deletes it on close; elsewhere the
        // name is removed at once and the open handle keeps the bytes
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            const FILE_ATTRIBUTE_TEMPORARY: u32 = 0x100;
            const FILE_FLAG_DELETE_ON_CLOSE: u32 = 0x0400_0000;
            options.attributes(FILE_ATTRIBUTE_TEMPORARY).custom_flags(FILE_FLAG_DELETE_ON_CLOSE);
        }
        let f = options.open(&path)?;
        #[cfg(not(windows))]
        { let _ = std::fs::remove_file(&path); }
        self.path = Some(path);
        let mut w = io::BufWriter::with_capacity(1 << 20, f);
        w.write_all(&self.mem)?;
        self.mem = Vec::new();
        self.w = Some(w);
        Ok(())
    }
    /// Drops everything from `len` on (a zstd frame that saved too little).
    #[cfg(feature = "zstd")]
    fn truncate(&mut self, len: u64) -> io::Result<()> {
        match &mut self.w {
            Some(w) => { w.flush()?; w.get_mut().set_len(len)?; w.get_mut().seek(SeekFrom::Start(len))?; }
            None => self.mem.truncate(len as usize),
        }
        self.len = len;
        Ok(())
    }
    /// No more appends: the temp file, if any, is read from now on.
    fn done(&mut self) -> io::Result<()> {
        if let Some(w) = self.w.take() { self.r = Some(w.into_inner().map_err(|e| e.into_error())?); }
        Ok(())
    }
    fn read_at(&mut self, off: u64, buf: &mut [u8]) -> io::Result<()> {
        match &mut self.r {
            Some(f) => read_at(f, off, buf),
            None => { buf.copy_from_slice(&self.mem[off as usize..off as usize + buf.len()]); Ok(()) }
        }
    }
}

impl Write for Spool {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        if self.w.is_none() && self.mem.len() + b.len() > self.cap { self.spill()?; }
        match &mut self.w { Some(w) => w.write_all(b)?, None => self.mem.extend_from_slice(b) }
        self.len += b.len() as u64;
        Ok(b.len())
    }
    fn flush(&mut self) -> io::Result<()> { if let Some(w) = &mut self.w { w.flush()?; } Ok(()) }
}

impl Drop for Spool {
    fn drop(&mut self) {
        // the handles first (a file still open cannot be deleted everywhere)
        self.w = None; self.r = None;
        if let Some(p) = &self.path { let _ = std::fs::remove_file(p); }
    }
}

/// Exactly `buf.len()` bytes at `off`: one positioned read (no separate seek) where the platform has one.
fn read_at(f: &mut File, off: u64, buf: &mut [u8]) -> io::Result<()> {
    #[cfg(unix)]
    { use std::os::unix::fs::FileExt; f.read_exact_at(buf, off) }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;
        while done < buf.len() {
            let n = f.seek_read(&mut buf[done..], off + done as u64)?;
            if n == 0 { return Err(io::ErrorKind::UnexpectedEof.into()); }
            done += n;
        }
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    { f.seek(SeekFrom::Start(off))?; f.read_exact(buf) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    fn tmp(name: &str, data: &[u8]) -> std::path::PathBuf {
        let d = std::env::temp_dir().join("lensbolt_prepare"); std::fs::create_dir_all(&d).unwrap();
        let p = d.join(name); std::fs::write(&p, data).unwrap(); p
    }
    #[test]
    fn from_bytes_matches_the_file_path_preparation() {
        let data: Vec<u8> = (0..300_000u32).map(|i| (i * 7919 % 251) as u8).collect();
        let dir = std::env::temp_dir().join("lensbolt_prep_bytes"); std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("x.bin"); std::fs::write(&path, &data).unwrap();
        let mut a = Prepared::new(&path, Some(false)).unwrap();
        let mut b = Prepared::from_bytes("x.bin", data.clone()).unwrap();
        assert_eq!(a.manifest.encode(), b.manifest.encode());
        assert_eq!(a.read_chunk(1, 0).unwrap(), b.read_chunk(1, 0).unwrap());
        assert_eq!(b.read_chunk(1, 0).unwrap().len(), b.manifest.chunk_len(1, 0));
    }
    #[test]
    fn blocks_xor_the_same_from_a_file_and_from_bytes() {
        // 300,000 bytes: one chunk of 293 blocks, the last one 992 bytes (the rest of it counts as zeros)
        let data: Vec<u8> = (0..300_000u32).map(|i| (i * 7919 % 251) as u8).collect();
        let dir = std::env::temp_dir().join("lensbolt_prep_blocks"); std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("x.bin"); std::fs::write(&path, &data).unwrap();
        let (mut a, mut b) = (Prepared::new(&path, Some(false)).unwrap(), Prepared::from_bytes("x.bin", data.clone()).unwrap());
        for i in [0usize, 1, 150, 291, 292] {
            let (mut x, mut y) = ([7u8; BLOCK_BYTES], [7u8; BLOCK_BYTES]);
            a.xor_block(1, 0, i, &mut x).unwrap();
            b.xor_block(1, 0, i, &mut y).unwrap();
            let mut want = [7u8; BLOCK_BYTES];
            for (w, d) in want.iter_mut().zip(&data[i * BLOCK_BYTES..((i + 1) * BLOCK_BYTES).min(data.len())]) { *w ^= *d; }
            assert_eq!((x, y), (want, want), "block {i}");
        }
    }
    #[test]
    fn incompressible_file_is_sent_raw_with_correct_hashes() {
        let mut seed = 7u64; let data: Vec<u8> = (0..3_000_000).map(|_| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed as u8 }).collect();
        let p = tmp("raw.bin", &data);
        let mut pr = Prepared::new(&p, None).unwrap();
        let m = pr.manifest.clone();
        assert!(!m.compressed);
        assert_eq!((m.file_size, m.stream_size), (3_000_000, 3_000_000));
        assert_eq!(m.filename, "raw.bin");
        assert_eq!(m.file_sha256, <[u8; 32]>::from(Sha256::digest(&data)));
        assert_eq!(m.segment_count(), 1);
        assert_eq!(m.segment_sha256[0], <[u8; 32]>::from(Sha256::digest(&data)));
        assert_eq!(pr.read_chunk(1, 2).unwrap(), &data[2 * 1024 * 1024..3_000_000]);
    }
    #[test]
    #[cfg(feature = "zstd")]
    fn text_is_compressed_and_chunks_read_from_the_stream() {
        let data: Vec<u8> = b"lorem ipsum dolor sit amet ".iter().cycle().take(5_000_000).copied().collect();
        let p = tmp("text.txt", &data);
        let mut pr = Prepared::new(&p, None).unwrap();
        assert!(pr.manifest.compressed);
        assert!(pr.manifest.stream_size < 1_000_000);
        let mut stream = Vec::new();
        for c in 0..pr.manifest.chunk_count(1) { stream.extend(pr.read_chunk(1, c).unwrap()); }
        assert_eq!(stream.len() as u64, pr.manifest.stream_size);
        assert_eq!(zstd::bulk::decompress(&stream, data.len()).unwrap(), data);
        let temp = pr.temp_path.clone().unwrap();
        drop(pr);
        assert!(!temp.exists(), "temp stream must be deleted on drop");
    }

    fn hex(b: &[u8]) -> String { b.iter().map(|x| format!("{x:02x}")).collect() }
    fn noise(len: usize, mut seed: u64) -> Vec<u8> { (0..len).map(|_| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed as u8 }).collect() }
    fn text(len: usize) -> Vec<u8> { b"a line of text that repeats, more or less. ".iter().cycle().take(len).copied().collect() }
    fn sha(b: &[u8]) -> [u8; 32] { Sha256::digest(b).into() }
    /// The whole stream, chunk by chunk, as the packet source reads it.
    fn stream_of(p: &mut Prepared) -> Vec<u8> {
        let m = p.manifest.clone();
        let mut v = Vec::new();
        for s in 1..=m.segment_count() as u16 { for c in 0..m.chunk_count(s) { v.extend(p.read_chunk(s, c).unwrap()); } }
        v
    }
    /// A batch stream split as a receiver does it: the index from the manifest's length and hash, then each
    /// file's bytes at its offset, decompressed (output capped at the file's size) and checked against its hash.
    fn unpack(m: &Manifest, stream: &[u8]) -> (Index, Vec<Vec<u8>>) {
        let info = m.batch.as_ref().expect("a batch manifest");
        assert_eq!(stream.len() as u64, m.stream_size);
        let raw = &stream[..info.index_len as usize];
        assert_eq!(sha(raw), info.index_sha256, "the index hash in the manifest");
        let index = Index::decode(raw).expect("the index decodes");
        assert_eq!(index.stream_len(), m.stream_size, "the index ends where the stream ends");
        let files = index.entries.iter().map(|e| {
            let stored = &stream[e.offset as usize..(e.offset + e.stored_size) as usize];
            let data = if e.compressed {
                #[cfg(feature = "zstd")]
                { zstd::bulk::decompress(stored, e.size as usize).expect("a zstd frame no larger than the file") }
                #[cfg(not(feature = "zstd"))]
                { panic!("compressed without zstd") }
            } else { stored.to_vec() };
            assert_eq!((data.len() as u64, sha(&data)), (e.size, e.sha256), "{}: size and hash", e.name);
            data
        }).collect();
        (index, files)
    }

    #[test]
    fn single_file_manifest_is_unchanged() {
        // today's encoding, byte for byte (taken at 241d968, before batches): a batch never changes a single file
        const RAW: &str = "4c424d460100409c000000000000409c0000000000000a00676f6c64656e2e62696ef02c6a5b983198a34eb1069f751cc0cb4387f2dbda8b1ec864bc98a78a2c8e5701000000f02c6a5b983198a34eb1069f751cc0cb4387f2dbda8b1ec864bc98a78a2c8e57b8d0";
        const ZST: &str = "4c424d460101000080020000000000001001000000000c007661636174696f6e2e7a697001010101010101010101010101010101010101010101010101010101010101010200000002020202020202020202020202020202020202020202020202020202020202020303030303030303030303030303030303030303030303030303030303030303e234";
        const EMPTY: &str = "4c424d460100000000000000000000000000000000000900656d7074792e747874e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855000000005a89";
        let data: Vec<u8> = (0..40_000u32).map(|i| (i * 7919 % 251) as u8).collect();
        let p = Prepared::from_bytes("golden.bin", data.clone()).unwrap();
        assert_eq!(hex(&p.manifest.encode()), RAW);
        assert_eq!(super::super::content_transfer_id(&p.manifest), 0x3d35);
        let on_disk = Prepared::new(&tmp("golden.bin", &data), Some(false)).unwrap();
        assert_eq!(hex(&on_disk.manifest.encode()), RAW);
        let z = Manifest { filename: "vacation.zip".into(), file_size: 40 << 20, stream_size: 17 << 20, compressed: true, file_sha256: [1; 32], segment_sha256: vec![[2; 32], [3; 32]], batch: None, tune: None };
        assert_eq!(hex(&z.encode()), ZST);
        assert_eq!(super::super::content_transfer_id(&z), 0xece9);
        let e = Prepared::from_bytes("empty.txt", Vec::new()).unwrap();
        assert_eq!(hex(&e.manifest.encode()), EMPTY);
        assert_eq!(super::super::content_transfer_id(&e.manifest), 0x19bf);
    }

    #[test]
    fn stream_is_index_then_files_smallest_first() {
        let given: Vec<(&str, Vec<u8>)> = vec![
            ("big.bin", noise(300_000, 1)), ("tiny.txt", b"hi".to_vec()), ("text.txt", text(200_000)),
            ("same-a.bin", noise(5000, 2)), ("empty", Vec::new()), ("same-b.bin", noise(5000, 3)), ("mid.bin", noise(50_000, 4)),
        ];
        let files = given.iter().map(|(n, d)| (n.to_string(), BatchSource::Bytes(d.clone()))).collect();
        let mut p = Prepared::batch("My batch", files).unwrap();
        let m = p.manifest.clone();
        let stream = stream_of(&mut p);
        // the manifest: the batch name, the stream as the file (raw, hashed whole and by segment)
        let info = m.batch.clone().unwrap();
        assert_eq!((m.filename.as_str(), info.name.as_str()), ("My batch", "My batch"));
        assert_eq!((m.file_size, m.compressed, m.file_sha256), (m.stream_size, false, sha(&stream)));
        assert_eq!(m.segment_sha256, stream.chunks(SEGMENT_BYTES as usize).map(sha).collect::<Vec<_>>());
        assert_eq!(Manifest::decode(&m.encode()), Some(m.clone()));
        let (index, got) = unpack(&m, &stream);
        // smallest first, ties in the order given
        let names: Vec<&str> = index.entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["empty", "tiny.txt", "same-a.bin", "same-b.bin", "mid.bin", "text.txt", "big.bin"]);
        // the files back to back right after the index
        assert_eq!(index.entries[0].offset, info.index_len);
        for w in index.entries.windows(2) { assert_eq!(w[1].offset, w[0].offset + w[0].stored_size); }
        for (e, data) in index.entries.iter().zip(&got) { assert!(given.iter().any(|(n, d)| *n == e.name && d == data), "{}", e.name); }
        // each compressed on its own when that makes it smaller: the text; noise, 2 bytes and nothing as they are
        let packed: Vec<bool> = index.entries.iter().map(|e| e.compressed).collect();
        assert_eq!(packed, [false, false, false, false, false, cfg!(feature = "zstd"), false]);
        assert!(index.entries.iter().all(|e| !e.compressed || e.stored_size < e.size));
        assert_eq!(p.index.as_ref(), Some(&index), "the sender holds the index it sent");
    }

    #[test]
    fn batch_names_are_sanitised_and_duplicates_kept() {
        let names = ["a/b:c.txt", "photo.jpg", "photo.jpg", "CON", "trail. ", "..", "PHOTO.JPG"];
        let files = names.iter().enumerate().map(|(i, n)| (n.to_string(), BatchSource::Bytes(vec![i as u8; 10]))).collect();
        let mut p = Prepared::batch("../evil:batch", files).unwrap();
        let m = p.manifest.clone();
        // the names the receiver writes, on the sender too; equal names are left for the receiver to tell apart
        assert_eq!((m.filename.as_str(), m.batch.as_ref().unwrap().name.as_str()), (".._evil_batch", ".._evil_batch"));
        let (index, got) = unpack(&m, &stream_of(&mut p));
        let sent: Vec<&str> = index.entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(sent, ["a_b_c.txt", "photo.jpg", "photo.jpg", "_CON", "trail", "_", "PHOTO.JPG"]);
        assert_eq!(got, (0..7u8).map(|i| vec![i; 10]).collect::<Vec<_>>(), "equal sizes keep the order given");
        assert!(index.validate().is_ok());
    }

    #[test]
    fn empty_files_in_a_batch() {
        let files = vec![("a.txt".to_string(), BatchSource::Bytes(Vec::new())), ("b.txt".to_string(), BatchSource::Bytes(Vec::new()))];
        let mut p = Prepared::batch("nothing", files).unwrap();
        let m = p.manifest.clone();
        let info = m.batch.clone().unwrap();
        // the stream is the index alone; both files start (and end) where it ends
        assert_eq!(m.stream_size, info.index_len);
        assert_eq!(m.segment_count(), 1);
        let (index, got) = unpack(&m, &stream_of(&mut p));
        assert_eq!(index.entries.iter().map(|e| (e.offset, e.size, e.stored_size)).collect::<Vec<_>>(), [(info.index_len, 0, 0); 2]);
        assert_eq!(got, [Vec::<u8>::new(), Vec::new()]);
        // an empty file among others, from a path too
        let dir = std::env::temp_dir().join(format!("lensbolt_batch_empty_{}", std::process::id())); std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("zero"), b"").unwrap();
        let files = vec![("x.bin".to_string(), BatchSource::Bytes(noise(3000, 9))), ("zero".to_string(), BatchSource::Path(dir.join("zero")))];
        let mut p = Prepared::batch("mixed", files).unwrap();
        let m = p.manifest.clone();
        let base = m.batch.as_ref().unwrap().index_len;
        let (index, got) = unpack(&m, &stream_of(&mut p));
        assert_eq!(index.entries[0].name, "zero");
        assert_eq!((index.entries[0].offset, index.entries[1].offset), (base, base));
        assert_eq!(got[1], noise(3000, 9));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_batch_of_one_file() {
        let data = noise(70_000, 5);
        let mut p = Prepared::batch("", vec![("only.bin".to_string(), BatchSource::Bytes(data.clone()))]).unwrap();
        let m = p.manifest.clone();
        // no name given: the first file's
        assert_eq!(m.batch.as_ref().unwrap().name, "only.bin");
        let stream = stream_of(&mut p);
        let (index, got) = unpack(&m, &stream);
        assert_eq!((index.entries.len(), &got[0]), (1, &data));
        assert_eq!(&stream[m.batch.as_ref().unwrap().index_len as usize..], &data[..], "the stream is the index, then the file");
        // a batch of one is not the file alone: another manifest, another stream
        let single = Prepared::from_bytes("only.bin", data).unwrap();
        assert_ne!(m.encode(), single.manifest.encode());
        // several files and no name: the first given, and how many more
        let files = ["b.txt", "a.txt", "c.txt"].iter().map(|n| (n.to_string(), BatchSource::Bytes(vec![1; 4]))).collect();
        assert_eq!(Prepared::batch("  ", files).unwrap().manifest.batch.clone().unwrap().name, "b.txt +2 more");
    }

    #[test]
    fn an_index_above_the_limit_is_refused_at_prepare_time() {
        // 260,000 names of 255 bytes: an index of at least 75 MB, over MAX_INDEX_BYTES (64 MiB). The paths do not
        // exist: the batch is refused before any file is opened
        let name = format!("{}.jpg", "n".repeat(251));
        let files = (0..260_000).map(|i| (name.clone(), BatchSource::Path(PathBuf::from(format!("no-such-dir/{i}"))))).collect();
        let err = Prepared::batch("too many", files).err().expect("refused");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        let msg = err.to_string();
        assert!(msg.contains("file list") && msg.contains("64 MiB"), "{msg}");
        // and as many files with short names fit
        let files = (0..260_000).map(|_| ("n.jpg".to_string(), BatchSource::Bytes(Vec::new()))).collect();
        let p = Prepared::batch("fits", files).unwrap();
        assert!(p.manifest.batch.as_ref().unwrap().index_len <= super::super::batch::MAX_INDEX_BYTES as u64);
        // no files: nothing to send
        assert_eq!(Prepared::batch("none", Vec::new()).err().map(|e| e.kind()), Some(io::ErrorKind::InvalidInput));
    }

    #[test]
    fn path_and_bytes_sources_make_the_same_stream() {
        // small files (compressed, or raw under 1 MiB) go to the spool, here spilled to a temp file after 4 KiB;
        // large raw ones are read in place; large compressible ones stream through zstd into the spool
        let dir = std::env::temp_dir().join(format!("lensbolt_batch_paths_{}", std::process::id())); std::fs::create_dir_all(&dir).unwrap();
        let given: Vec<(&str, Vec<u8>)> = vec![
            ("notes.txt", text(30_000)), ("small.bin", noise(20_000, 6)), ("two-mib.bin", noise(2 << 20, 7)),
            ("big-text.log", text(5 << 20)), ("big.bin", noise(5 << 20, 8)), ("empty.txt", Vec::new()), ("one.txt", b"1".to_vec()),
        ];
        for (n, d) in &given { std::fs::write(dir.join(n), d).unwrap(); }
        let paths = given.iter().map(|(n, _)| (n.to_string(), BatchSource::Path(dir.join(n)))).collect();
        let bytes = given.iter().map(|(n, d)| (n.to_string(), BatchSource::Bytes(d.clone()))).collect();
        let mut a = Prepared::batch_with("set", paths, Limits::new(4096, OPEN_FILES)).unwrap();
        let mut b = Prepared::batch("set", bytes).unwrap();
        assert_eq!(a.manifest.encode(), b.manifest.encode());
        let spool = a.temp_path.clone();
        assert!(spool.is_some(), "the spool spilled to a temp file");
        // Windows deletes it when its handle closes; elsewhere its name is gone already
        assert_eq!(spool.as_ref().unwrap().exists(), cfg!(windows));
        assert!(b.temp_path.is_none(), "bytes never touch the file system");
        let stream = stream_of(&mut a);
        assert!(stream == stream_of(&mut b));
        let (index, got) = unpack(&a.manifest, &stream);
        for (e, data) in index.entries.iter().zip(&got) { assert!(given.iter().any(|(n, d)| *n == e.name && d == data), "{}", e.name); }
        let packed: Vec<(&str, bool)> = index.entries.iter().map(|e| (e.name.as_str(), e.compressed)).collect();
        let z = cfg!(feature = "zstd");
        assert_eq!(packed, [("empty.txt", false), ("one.txt", false), ("small.bin", false), ("notes.txt", z), ("two-mib.bin", false), ("big-text.log", z), ("big.bin", false)]);
        // blocks read the same from both (across file boundaries, from the spool, in place)
        let last = a.manifest.chunk_count(1) - 1;
        for (chunk, i) in [(0u8, 0usize), (0, 3), (1, 1023), (2, 500), (last, 0)] {
            let (mut x, mut y) = ([0u8; BLOCK_BYTES], [0u8; BLOCK_BYTES]);
            a.xor_block(1, chunk, i, &mut x).unwrap();
            b.xor_block(1, chunk, i, &mut y).unwrap();
            assert_eq!(x, y, "block {i} of chunk {chunk}");
        }
        drop(a);
        assert!(!spool.unwrap().exists(), "the spool is deleted with the batch");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn in_place_files_stay_open_through_the_sweep() {
        // raw files of 1 MiB and more are read in place while sending. The pass (and every repair round) visits the
        // chunks in turn, so the files it reads cycle: up to the cap every file stays open and none is opened again;
        // over it, evicting the least recently used would reopen a file at every switch from one file to the next
        let dir = std::env::temp_dir().join(format!("lensbolt_batch_open_{}", std::process::id())); std::fs::create_dir_all(&dir).unwrap();
        let files: Vec<PathBuf> = (0..12).map(|i| {
            let p = dir.join(format!("photo-{i:02}.jpg"));
            std::fs::write(&p, noise((1 << 20) + i * 40_000, 100 + i as u64)).unwrap();
            p
        }).collect();
        let pass = |cap: usize| -> (u64, u64, u64) {
            let sources = files.iter().map(|p| (p.file_name().unwrap().to_string_lossy().into_owned(), BatchSource::Path(p.clone()))).collect();
            let p = Prepared::batch_with("photos", sources, Limits::new(SPOOL_IN_MEMORY, cap)).unwrap();
            let (index, m) = (p.index.clone().unwrap(), p.manifest.clone());
            let index_len = m.batch.as_ref().unwrap().index_len;
            let mut src = super::super::packets::PacketSource::new(p).unwrap();
            let before = src.prepared().opens();
            let (mut switches, mut last) = (0u64, None);
            let packets = src.data_packets_per_pass();
            for _ in 0..packets {
                let id = src.next_data_packet().unwrap().id;
                let off = m.chunk_offset(id.segment, id.chunk) + id.pid as u64 * BLOCK_BYTES as u64;
                if off < index_len { continue; }
                let file = index.entries.partition_point(|e| e.offset + e.stored_size <= off);
                if last.is_some_and(|l| l != file) { switches += 1; }
                last = Some(file);
            }
            (src.prepared().opens() - before, switches, packets)
        };
        let (opens, _, _) = pass(16);
        assert_eq!(opens, 0, "12 files under a cap of 16: every one stays open from preparation on");
        let (opens, switches, packets) = pass(8);
        eprintln!("in_place_files_stay_open_through_the_sweep: 12 files, 8 open at most: {opens} opens in a pass of {packets} packets ({switches} switches between files)");
        assert!(switches > 1000, "the pass switches between files {switches} times");
        // keeping 7 of the 12 open through each cycle: about (12 - 8 + 1) / 12 of the switches open a file (measured:
        // 4658 opens for 12770 switches; evicting the least recently used instead, 6378)
        assert!(opens as f64 <= 5.0 / 12.0 * switches as f64, "{opens} opens for {switches} switches between files");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_file_changed_while_preparing_is_refused() {
        let dir = std::env::temp_dir().join(format!("lensbolt_batch_changed_{}", std::process::id())); std::fs::create_dir_all(&dir).unwrap();
        let big = dir.join("big.bin");
        let sources = || vec![("big.bin".to_string(), BatchSource::Path(dir.join("big.bin"))), ("small.txt".to_string(), BatchSource::Bytes(text(500)))];
        // one byte flipped, or the file cut short, between the hashing and the segment pass
        fn flip(paths: &[PathBuf]) { let mut d = std::fs::read(&paths[0]).unwrap(); d[700_000] ^= 1; std::fs::write(&paths[0], d).unwrap(); }
        fn cut(paths: &[PathBuf]) { std::fs::OpenOptions::new().write(true).open(&paths[0]).unwrap().set_len(1 << 20).unwrap(); }
        for touch in [flip as fn(&[PathBuf]), cut] {
            std::fs::write(&big, noise((1 << 20) + 5000, 11)).unwrap();
            let limits = Limits { between_passes: Some(touch), ..Limits::new(SPOOL_IN_MEMORY, OPEN_FILES) };
            let err = Prepared::batch_with("b", sources(), limits).err().expect("refused");
            assert!(err.to_string().ends_with("big.bin changed while preparing"), "{err}");
        }
        // untouched, it prepares
        std::fs::write(&big, noise((1 << 20) + 5000, 11)).unwrap();
        assert!(Prepared::batch_with("b", sources(), Limits::new(SPOOL_IN_MEMORY, OPEN_FILES)).is_ok());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn only_regular_files_and_errors_name_the_path() {
        let dir = std::env::temp_dir().join(format!("lensbolt_batch_regular_{}", std::process::id())); std::fs::create_dir_all(&dir).unwrap();
        let err = Prepared::batch("b", vec![("d".to_string(), BatchSource::Path(dir.clone()))]).err().expect("a directory is refused");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(err.to_string(), format!("{} is not a regular file", dir.display()));
        let missing = dir.join("missing.jpg");
        let err = Prepared::batch("b", vec![("m".to_string(), BatchSource::Path(missing.clone()))]).err().expect("a missing file is refused");
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
        assert!(err.to_string().starts_with(&format!("{}: ", missing.display())), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn tiny_files_are_never_compressed() {
        // under 64 bytes as they are, even when zstd would shave a byte or two; 64 and up when it saves 5%
        let files = vec![("a.txt".to_string(), BatchSource::Bytes(vec![b'a'; 63])), ("b.txt".to_string(), BatchSource::Bytes(vec![b'a'; 64]))];
        let p = Prepared::batch("t", files).unwrap();
        let packed: Vec<bool> = p.index.as_ref().unwrap().entries.iter().map(|e| e.compressed).collect();
        assert_eq!(packed, [false, cfg!(feature = "zstd")]);
    }

    #[test]
    #[ignore = "timing: cargo test --release --lib ten_thousand -- --ignored --nocapture"]
    fn ten_thousand_small_path_files_prepare() {
        // 10,000 files of 1 B to 4 KB (a third text), from paths and from bytes
        let dir = std::env::temp_dir().join(format!("lensbolt_batch_10k_{}", std::process::id())); std::fs::create_dir_all(&dir).unwrap();
        let given: Vec<(String, Vec<u8>)> = (0..10_000u64).map(|i| {
            let size = 1 + (i * 7919 % 4096) as usize;
            (format!("file-{i:05}.{}", if i % 3 == 0 { "txt" } else { "bin" }), if i % 3 == 0 { text(size) } else { noise(size, i + 1) })
        }).collect();
        for (n, d) in &given { std::fs::write(dir.join(n), d).unwrap(); }
        let t = std::time::Instant::now();
        let a = Prepared::batch("10k", given.iter().map(|(n, _)| (n.clone(), BatchSource::Path(dir.join(n)))).collect()).unwrap();
        let paths = t.elapsed();
        let t = std::time::Instant::now();
        let b = Prepared::batch("10k", given.iter().map(|(n, d)| (n.clone(), BatchSource::Bytes(d.clone()))).collect()).unwrap();
        let bytes = t.elapsed();
        assert_eq!(a.manifest.encode(), b.manifest.encode());
        let packed = a.index.as_ref().unwrap().entries.iter().filter(|e| e.compressed).count();
        eprintln!("10,000 small files ({} bytes, {packed} compressed, a stream of {} bytes): from paths {paths:?}, from bytes {bytes:?}",
            given.iter().map(|g| g.1.len()).sum::<usize>(), a.manifest.stream_size);
        drop(a);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Every chunk of the stream, in order.
    fn all_chunks(p: &mut Prepared) -> Vec<Vec<u8>> {
        let m = p.manifest.clone();
        let mut out = Vec::new();
        for seg in 1..=m.segment_count() as u16 { for c in 0..m.chunk_count(seg) { out.push(p.read_chunk(seg, c).unwrap()); } }
        out
    }
    #[test]
    fn shared_bytes_are_sent_as_owned_bytes_without_a_copy() {
        // 17 MB: two segments, a short last chunk
        let data = noise(17_000_000, 11);
        let shared = Arc::new(data.clone());
        let mut a = Prepared::from_bytes("x.bin", data.clone()).unwrap();
        let mut b = Prepared::from_shared("x.bin", shared.clone()).unwrap();
        assert_eq!(a.manifest.encode(), b.manifest.encode(), "the same manifest, byte for byte");
        assert!(all_chunks(&mut a) == all_chunks(&mut b), "the same stream");
        for (seg, c, i) in [(1u16, 0u8, 0usize), (1, 15, 1023), (2, 0, 7), (2, 0, 217)] {
            let (mut x, mut y) = ([3u8; BLOCK_BYTES], [3u8; BLOCK_BYTES]);
            a.xor_block(seg, c, i, &mut x).unwrap(); b.xor_block(seg, c, i, &mut y).unwrap();
            assert_eq!(x, y, "block {i} of chunk {c} of segment {seg}");
        }
        assert_eq!(Arc::strong_count(&shared), 2, "the prepared file holds the caller's bytes, not a copy");
        drop(b);
        assert_eq!(Arc::strong_count(&shared), 1);
        let empty = Arc::new(Vec::new());
        assert_eq!(Prepared::from_shared("e", empty.clone()).unwrap().manifest.encode(), Prepared::from_bytes("e", Vec::new()).unwrap().manifest.encode());
    }
    #[test]
    fn shared_batch_sources_are_sent_as_bytes_sources_without_a_copy() {
        // raw files, a 0-byte file, two of the same name, one that packs well (compressed where zstd is built in)
        let files: Vec<(String, Vec<u8>)> = vec![
            ("b.jpg".into(), noise(1_500_000, 1)), ("a.txt".into(), b"one line of a log ".repeat(20_000)), ("empty.dat".into(), Vec::new()),
            ("c.bin".into(), noise(70_000, 2)), ("c.bin".into(), noise(70_000, 3)), ("tiny".into(), vec![9u8; 5]),
        ];
        let shared: Vec<Arc<Vec<u8>>> = files.iter().map(|(_, d)| Arc::new(d.clone())).collect();
        let mut a = Prepared::batch("", files.iter().map(|(n, d)| (n.clone(), BatchSource::Bytes(d.clone()))).collect()).unwrap();
        let mut b = Prepared::batch("", files.iter().zip(&shared).map(|((n, _), d)| (n.clone(), BatchSource::Shared(d.clone()))).collect()).unwrap();
        assert_eq!(a.manifest.encode(), b.manifest.encode(), "the same manifest");
        assert_eq!(a.index, b.index, "the same file list");
        assert!(all_chunks(&mut a) == all_chunks(&mut b), "the same stream");
        let idx = b.index.clone().unwrap();
        for ((name, d), arc) in files.iter().zip(&shared) {
            let e = idx.entries.iter().find(|e| &e.name == name && e.size == d.len() as u64 && e.sha256 == <[u8; 32]>::from(Sha256::digest(d))).unwrap();
            // a raw non-empty file is read from the caller's bytes; an empty or compressed one keeps nothing of them
            let held = if d.is_empty() || e.compressed { 1 } else { 2 };
            assert_eq!(Arc::strong_count(arc), held, "{name} ({} B, compressed {})", d.len(), e.compressed);
        }
        drop(b);
        assert!(shared.iter().all(|a| Arc::strong_count(a) == 1));
    }
}
