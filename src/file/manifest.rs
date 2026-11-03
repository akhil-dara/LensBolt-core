use crate::crc16::crc16;

pub const SEGMENT_BYTES: u64 = 16 * 1024 * 1024;
pub const CHUNK_BYTES: u64 = 1024 * 1024;
pub const CHUNKS_PER_SEGMENT: u32 = 16;
pub const MANIFEST_MAGIC: &[u8; 4] = b"LBMF";
const MANIFEST_VERSION: u8 = 1;

/// What a manifest says about a single file or a batch. A single file: its name, its size, whether the stream
/// is its zstd frame, its SHA-256, and the SHA-256 of each 16 MiB segment of the stream. A batch (`batch` is
/// Some): the stream is `index || file_1 || file_2 || ...` (`batch::Index`; each file compressed on its own or
/// not), `filename` is the batch name too (decode refuses one that differs), `file_size` is the stream size, `compressed` is false and
/// `file_sha256` is the SHA-256 of the whole stream.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Manifest { pub filename: String, pub file_size: u64, pub stream_size: u64, pub compressed: bool, pub file_sha256: [u8; 32], pub segment_sha256: Vec<[u8; 32]>, pub batch: Option<BatchInfo>, pub tune: Option<TuneInfo> }

/// The tune section of a manifest: the sender is showing the Tune carousel (`tune::schedule`), not a file. Its
/// payload is generated (`tune::schedule::tune_payload`) and no receiver ever offers it to save. `schedule`: the
/// carousel's version; `display_60`: the sender measured a 60 or 120 Hz screen, so the 60 fps segments are drawn
/// at 60 (else at 30, and receivers ignore them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TuneInfo { pub schedule: u8, pub display_60: bool }

/// The batch section of a manifest: the batch name, and the length and SHA-256 of the file index at the
/// start of the stream (bytes `0 .. index_len`). A receiver can refuse an index longer than
/// `batch::MAX_INDEX_BYTES` before buffering any of it, and decodes it once its bytes match `index_sha256`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BatchInfo { pub name: String, pub index_len: u64, pub index_sha256: [u8; 32] }

/// Bit 0 of the manifest's flags byte: the stream is the file's zstd frame (a single file only).
const FLAG_COMPRESSED: u8 = 1;
/// Bit 1: a batch; the batch section follows the segment hashes. A build that predates batches reads the
/// flags byte as 0 or 1 only, so it refuses a batch manifest instead of taking the stream for one file.
const FLAG_BATCH: u8 = 2;
/// Bit 2: the Tune carousel (never with bits 0 or 1); a 2-byte tune section (schedule, flags: bit 0 `display_60`)
/// follows the segment hashes. Earlier builds refuse it, as they refuse a batch.
const FLAG_TUNE: u8 = 4;

/// Fountain blocks (k) for a chunk of `len` bytes.
pub fn chunk_blocks(len: usize) -> usize { (len + 1023) / 1024 }

impl Manifest {
    pub fn segment_count(&self) -> u32 { ((self.stream_size + SEGMENT_BYTES - 1) / SEGMENT_BYTES) as u32 }
    /// seg in 1..=segment_count
    pub fn segment_len(&self, seg: u16) -> u64 {
        let start = (seg as u64 - 1) * SEGMENT_BYTES;
        (self.stream_size - start).min(SEGMENT_BYTES)
    }
    pub fn chunk_count(&self, seg: u16) -> u8 { ((self.segment_len(seg) + CHUNK_BYTES - 1) / CHUNK_BYTES) as u8 }
    pub fn chunk_len(&self, seg: u16, chunk: u8) -> usize {
        let seg_len = self.segment_len(seg);
        let start = chunk as u64 * CHUNK_BYTES;
        (seg_len - start).min(CHUNK_BYTES) as usize
    }
    pub fn chunk_offset(&self, seg: u16, chunk: u8) -> u64 { (seg as u64 - 1) * SEGMENT_BYTES + chunk as u64 * CHUNK_BYTES }
    /// Bytes a receiver writing to disk holds at its peak for this transfer: the part file (sized to the stream
    /// when the manifest arrives) and, for a compressed transfer, the unpacked file written next to it before the
    /// part is removed (measured in tests/disk_peak.rs: exactly this for 64 MB raw and zstd transfers).
    ///
    /// A batch: the part file, and beside it each file as it is saved (the part goes once the last segment
    /// verified), so the stream plus the files' sizes. The manifest does not carry those sizes, so this is the
    /// stream twice less the file list, exact when no file travels compressed; the receiver reports the exact
    /// figure once the file list is in (`Progress::disk_peak`), which is larger when files travel compressed.
    pub fn receive_disk_peak(&self) -> u64 {
        match &self.batch {
            Some(b) => self.stream_size + self.stream_size.saturating_sub(b.index_len),
            None => self.stream_size + if self.compressed { self.file_size } else { 0 },
        }
    }
    /// The units a finish code names: the data segments (unit u = segment u + 1) when there are two or more,
    /// else the chunks of the one segment (unit u = chunk u); 0 for an empty stream.
    pub fn finish_units(&self) -> u32 {
        match self.segment_count() { 0 => 0, 1 => self.chunk_count(1) as u32, s => s }
    }
    /// The bytes of the stream one finish-word unit spans (unit u is bytes u * this .. (u + 1) * this): a segment
    /// when the stream has two or more, else a chunk.
    pub fn finish_unit_bytes(&self) -> u64 { if self.segment_count() >= 2 { SEGMENT_BYTES } else { CHUNK_BYTES } }

    pub fn encode(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(64 + self.filename.len() + 32 * self.segment_sha256.len());
        b.extend_from_slice(MANIFEST_MAGIC);
        b.push(MANIFEST_VERSION);
        b.push(if self.compressed { FLAG_COMPRESSED } else { 0 } | if self.batch.is_some() { FLAG_BATCH } else { 0 } | if self.tune.is_some() { FLAG_TUNE } else { 0 });
        b.extend_from_slice(&self.file_size.to_le_bytes());
        b.extend_from_slice(&self.stream_size.to_le_bytes());
        let name = self.filename.as_bytes();
        b.extend_from_slice(&(name.len() as u16).to_le_bytes());
        b.extend_from_slice(name);
        b.extend_from_slice(&self.file_sha256);
        b.extend_from_slice(&(self.segment_sha256.len() as u32).to_le_bytes());
        for h in &self.segment_sha256 { b.extend_from_slice(h); }
        // the batch section: index length u64 LE, index SHA-256, batch name length u16 LE, batch name
        if let Some(batch) = &self.batch {
            b.extend_from_slice(&batch.index_len.to_le_bytes());
            b.extend_from_slice(&batch.index_sha256);
            b.extend_from_slice(&(batch.name.len() as u16).to_le_bytes());
            b.extend_from_slice(batch.name.as_bytes());
        }
        // the tune section: schedule version, flags (bit 0 display_60)
        if let Some(t) = &self.tune { b.push(t.schedule); b.push(t.display_60 as u8); }
        let c = crc16(&b).to_be_bytes();
        b.extend_from_slice(&c);
        b
    }
    /// A manifest from untrusted bytes. A single file's manifest is what every earlier build wrote (flags 0 or
    /// 1, nothing after the segment hashes). A batch's (flags 2) is refused when its stream says it is one zstd
    /// frame, its file size is not its stream size, its index is empty, longer than the stream or longer than
    /// `batch::MAX_INDEX_BYTES` (so a receiver never buffers more), or its name is empty, over 255 bytes, not
    /// UTF-8 or not its `filename` (one name, whichever field a receiver reads).
    pub fn decode(bytes: &[u8]) -> Option<Manifest> {
        if bytes.len() < 4 + 1 + 1 + 8 + 8 + 2 + 32 + 4 + 2 { return None; }
        let (body, crc) = bytes.split_at(bytes.len() - 2);
        if u16::from_be_bytes([crc[0], crc[1]]) != crc16(body) { return None; }
        if &body[..4] != MANIFEST_MAGIC || body[4] != MANIFEST_VERSION { return None; }
        let (compressed, is_batch, is_tune) = match body[5] { 0 => (false, false, false), FLAG_COMPRESSED => (true, false, false), FLAG_BATCH => (false, true, false), FLAG_TUNE => (false, false, true), _ => return None };
        let mut p = 6usize;
        let rd8 = |p: &mut usize| -> Option<u64> { let v = u64::from_le_bytes(body.get(*p..*p + 8)?.try_into().ok()?); *p += 8; Some(v) };
        let rd_name = |p: &mut usize| -> Option<String> {
            let len = u16::from_le_bytes(body.get(*p..*p + 2)?.try_into().ok()?) as usize; *p += 2;
            let s = String::from_utf8(body.get(*p..*p + len)?.to_vec()).ok()?; *p += len;
            Some(s)
        };
        let file_size = rd8(&mut p)?; let stream_size = rd8(&mut p)?;
        let filename = rd_name(&mut p)?;
        let mut file_sha256 = [0u8; 32]; file_sha256.copy_from_slice(body.get(p..p + 32)?); p += 32;
        let seg_count = u32::from_le_bytes(body.get(p..p + 4)?.try_into().ok()?) as usize; p += 4;
        if seg_count > u16::MAX as usize || stream_size > u16::MAX as u64 * SEGMENT_BYTES { return None; }
        let hashes = body.get(p..p + 32 * seg_count)?; p += 32 * seg_count;
        let segment_sha256 = hashes.chunks(32).map(|c| { let mut h = [0u8; 32]; h.copy_from_slice(c); h }).collect();
        let batch = if is_batch {
            let index_len = rd8(&mut p)?;
            let mut index_sha256 = [0u8; 32]; index_sha256.copy_from_slice(body.get(p..p + 32)?); p += 32;
            let name = rd_name(&mut p)?;
            if file_size != stream_size || index_len == 0 || index_len > stream_size || index_len > super::batch::MAX_INDEX_BYTES as u64 { return None; }
            if name.is_empty() || name.len() > super::batch::MAX_NAME_BYTES || name != filename { return None; }
            Some(BatchInfo { name, index_len, index_sha256 })
        } else { None };
        let tune = if is_tune {
            let t = body.get(p..p + 2)?; p += 2;
            if t[1] & !1 != 0 { return None; }
            Some(TuneInfo { schedule: t[0], display_60: t[1] & 1 == 1 })
        } else { None };
        if p != body.len() { return None; }
        let m = Manifest { filename, file_size, stream_size, compressed, file_sha256, segment_sha256, batch, tune };
        if m.segment_count() as usize != seg_count { return None; }
        Some(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn m(stream: u64) -> Manifest {
        Manifest { filename: "vacation.zip".into(), file_size: stream + 5, stream_size: stream, compressed: true, file_sha256: [1; 32], segment_sha256: vec![[2; 32]; ((stream + SEGMENT_BYTES - 1) / SEGMENT_BYTES) as usize], batch: None, tune: None }
    }
    fn batch(stream: u64) -> Manifest {
        let info = BatchInfo { name: "Holiday +39 more".into(), index_len: 1234, index_sha256: [7; 32] };
        Manifest { filename: info.name.clone(), file_size: stream, compressed: false, batch: Some(info), ..m(stream) }
    }
    fn hex(s: &str) -> Vec<u8> { (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect() }
    #[test]
    fn geometry_of_segments_and_chunks() {
        let man = m(40 * 1024 * 1024 + 100);
        assert_eq!(man.segment_count(), 3);
        assert_eq!(man.segment_len(1), SEGMENT_BYTES);
        assert_eq!(man.segment_len(3), 8 * 1024 * 1024 + 100);
        assert_eq!(man.chunk_count(1), 16);
        assert_eq!(man.chunk_count(3), 9);
        assert_eq!(man.chunk_len(3, 8), 100);
        assert_eq!(man.chunk_len(1, 15), 1024 * 1024);
        assert_eq!(man.chunk_offset(2, 1), SEGMENT_BYTES + CHUNK_BYTES);
        assert_eq!(chunk_blocks(100), 1);
        assert_eq!(chunk_blocks(1024 * 1024), 1024);
        assert_eq!(chunk_blocks(1025), 2);
    }
    #[test]
    fn encode_decode_round_trip_and_corruption() {
        let man = m(1_000_000);
        let b = man.encode();
        assert_eq!(&b[..4], MANIFEST_MAGIC);
        assert_eq!(Manifest::decode(&b), Some(man.clone()));
        let mut bad = b.clone(); bad[20] ^= 1;
        assert_eq!(Manifest::decode(&bad), None);
        assert_eq!(Manifest::decode(&b[..b.len() - 1]), None);
    }
    #[test]
    fn one_tb_manifest_fits_in_a_segment() {
        let man = m(1u64 << 40);
        assert_eq!(man.segment_count(), 65536);
        assert!(man.encode().len() as u64 <= SEGMENT_BYTES);
    }

    #[test]
    fn batch_manifest_round_trips() {
        for stream in [1234u64, 1_000_000, 40 * 1024 * 1024 + 7] {
            let man = batch(stream);
            let b = man.encode();
            assert_eq!(b[5], FLAG_BATCH, "the flags byte says batch");
            assert_eq!(Manifest::decode(&b), Some(man.clone()));
            // the batch section: index length, index SHA-256, name length, name, after the segment hashes
            let name = b"Holiday +39 more";
            let tail = &b[b.len() - 2 - (8 + 32 + 2 + name.len())..b.len() - 2];
            assert_eq!(&tail[..8], &1234u64.to_le_bytes());
            assert_eq!(&tail[8..40], &[7u8; 32]);
            assert_eq!(&tail[40..42], &(name.len() as u16).to_le_bytes());
            assert_eq!(&tail[42..], name);
            assert_eq!(b.len(), Manifest { batch: None, ..man.clone() }.encode().len() + 8 + 32 + 2 + name.len());
            // every truncation and every flipped byte is refused (the CRC, then the fields)
            for n in 0..b.len() { assert_eq!(Manifest::decode(&b[..n]), None, "cut to {n}"); }
            for i in 0..b.len() { let mut bad = b.clone(); bad[i] ^= 0x41; assert_eq!(Manifest::decode(&bad), None, "byte {i} flipped"); }
        }
        // a name of 255 bytes, the longest
        let mut man = batch(5000);
        man.batch.as_mut().unwrap().name = "n".repeat(255); man.filename = "n".repeat(255);
        assert_eq!(Manifest::decode(&man.encode()), Some(man));
    }

    fn tune(display_60: bool) -> Manifest {
        Manifest { filename: "LensBolt tune".into(), file_size: 1 << 20, compressed: false, tune: Some(TuneInfo { schedule: 1, display_60 }), ..m(1 << 20) }
    }
    #[test]
    fn tune_manifest_round_trips() {
        for d60 in [false, true] {
            let man = tune(d60);
            let b = man.encode();
            assert_eq!(b[5], FLAG_TUNE, "the flags byte says tune");
            assert_eq!(&b[b.len() - 4..b.len() - 2], &[1, d60 as u8], "the tune section: schedule, display_60");
            assert_eq!(b.len(), Manifest { tune: None, ..man.clone() }.encode().len() + 2);
            assert_eq!(Manifest::decode(&b), Some(man.clone()));
            for n in 0..b.len() { assert_eq!(Manifest::decode(&b[..n]), None, "cut to {n}"); }
        }
    }
    #[test]
    fn tune_with_other_flags_is_refused() {
        let reseal = |mut b: Vec<u8>| { b.truncate(b.len() - 2); let c = crc16(&b).to_be_bytes(); b.extend_from_slice(&c); b };
        let good = tune(true).encode();
        for flags in [5u8, 6, 7, 12] { let mut bad = good.clone(); bad[5] = flags; assert_eq!(Manifest::decode(&reseal(bad)), None, "flags {flags}"); }
        // a tune section's flags byte with bits other than display_60
        let mut bad = good.clone(); let at = bad.len() - 3; bad[at] = 2;
        assert_eq!(Manifest::decode(&reseal(bad)), None, "tune flags 2");
        // a tune manifest without its section, and a compressed or batch one encoded with it
        let mut bad = good.clone(); bad.truncate(bad.len() - 4);
        assert_eq!(Manifest::decode(&reseal([bad, vec![0, 0]].concat())), None, "no tune section");
        assert_eq!(Manifest::decode(&Manifest { compressed: true, ..tune(false) }.encode()), None, "compressed tune");
    }
    #[test]
    fn old_format_manifest_without_a_batch_section_decodes() {
        // a single-file manifest as every earlier build wrote it (core/src/file/prepare.rs, the golden test)
        let b = hex("4c424d460100409c000000000000409c0000000000000a00676f6c64656e2e62696ef02c6a5b983198a34eb1069f751cc0cb4387f2dbda8b1ec864bc98a78a2c8e5701000000f02c6a5b983198a34eb1069f751cc0cb4387f2dbda8b1ec864bc98a78a2c8e57b8d0");
        let man = Manifest::decode(&b).expect("an old single-file manifest decodes");
        assert_eq!((man.filename.as_str(), man.file_size, man.stream_size, man.compressed, man.batch.clone()), ("golden.bin", 40_000, 40_000, false, None));
        assert_eq!(man.encode(), b, "and encodes back to the same bytes");
        let z = hex("4c424d460101000080020000000000001001000000000c007661636174696f6e2e7a697001010101010101010101010101010101010101010101010101010101010101010200000002020202020202020202020202020202020202020202020202020202020202020303030303030303030303030303030303030303030303030303030303030303e234");
        let man = Manifest::decode(&z).expect("an old compressed manifest decodes");
        assert!(man.compressed && man.batch.is_none());
        assert_eq!(man.encode(), z);
    }

    #[test]
    fn batch_section_is_checked() {
        let refused = |man: Manifest, why: &str| assert_eq!(Manifest::decode(&man.encode()), None, "{why}");
        let with = |f: &dyn Fn(&mut Manifest)| { let mut man = batch(1_000_000); f(&mut man); man };
        refused(with(&|m| m.compressed = true), "a batch whose stream says it is one zstd frame");
        refused(with(&|m| m.filename = "Holiday".into()), "a batch whose name is not its filename");
        refused(with(&|m| m.file_size += 1), "a batch whose size is not its stream's");
        refused(with(&|m| m.batch.as_mut().unwrap().index_len = 0), "an empty index");
        refused(with(&|m| m.batch.as_mut().unwrap().index_len = 1_000_001), "an index longer than the stream");
        let big = |m: &mut Manifest| { m.stream_size = 200 << 20; m.file_size = m.stream_size; m.segment_sha256 = vec![[2; 32]; 13]; };
        let mut ok = with(&big); ok.batch.as_mut().unwrap().index_len = super::super::batch::MAX_INDEX_BYTES as u64;
        assert_eq!(Manifest::decode(&ok.encode()), Some(ok.clone()), "an index of MAX_INDEX_BYTES");
        ok.batch.as_mut().unwrap().index_len += 1;
        refused(ok, "an index longer than MAX_INDEX_BYTES, refused before any of it is buffered");
        refused(with(&|m| m.batch.as_mut().unwrap().name = String::new()), "a batch without a name");
        refused(with(&|m| m.batch.as_mut().unwrap().name = "n".repeat(256)), "a batch name over 255 bytes");
        // not UTF-8, and bytes after the batch section, with a correct CRC
        let reseal = |mut b: Vec<u8>| { b.truncate(b.len() - 2); let c = crc16(&b).to_be_bytes(); b.extend_from_slice(&c); b };
        let good = batch(1_000_000).encode();
        let mut bad = good.clone(); let at = bad.len() - 3; bad[at] = 0xFF;
        assert_eq!(Manifest::decode(&reseal(bad)), None, "a name that is not UTF-8");
        let mut bad = good.clone(); bad.insert(bad.len() - 2, 0);
        assert_eq!(Manifest::decode(&reseal(bad)), None, "a byte after the batch section");
        // flags other than 0, 1 (a single file, raw or zstd) and 2 (a batch)
        for flags in [3u8, 4, 0x80, 0xFF] { let mut bad = good.clone(); bad[5] = flags; assert_eq!(Manifest::decode(&reseal(bad)), None, "flags {flags}"); }
        // a single-file manifest with a batch section's bytes after its hashes
        let mut bad = good.clone(); bad[5] = 0;
        assert_eq!(Manifest::decode(&reseal(bad)), None, "batch bytes without the batch flag");
    }
}
