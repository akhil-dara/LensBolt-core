//! Frame header for the flat format: carried in black/white 8 px chrome cells, so it is read before the
//! data pitch is known.

use super::format::CellFormat;
use crate::bits::{BitReader, BitWriter};
use crate::crc16::crc16;
use crate::header::{FrameInfo, FrameType, HeaderError, HEADER_MAGIC};
use crate::rs::RsCodec;

/// 5: flat cells, fixed 8 px chrome, pilots. 6: 10-bit cols/rows, manifests on 12 px / 8 colours. 7: frame
/// tags in pilots. 8: frame-wide codewords, packets back to back (the design notes). 9: the
/// fountain's repair packets for chunks of up to 128 blocks take each block with probability 1/2
/// (docs/2026-10-30-repair-schedule.md); frames are otherwise as in 8, but a receiver of one version must never
/// read the other's repair packets, so the version that every frame carries keeps them apart.
pub const FLAT_VERSION: u8 = 9;
/// 12 body bytes + CRC-16.
pub const FLAT_HEADER_BYTES: usize = 14;
/// Diagnostics only: also accept version-7 headers, so the stored real v7 captures (test/ref1px) can still be
/// located and their cells read (their payload is not v8 and never decodes).
pub static ACCEPT_V7_HEADERS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Diagnostics only: also accept version-8 headers, so the stored real v8 captures (test/phone_*) can still be
/// read cell by cell and codeword by codeword (v8 frames differ from v9 only in the repair packets of chunks of
/// up to 128 blocks, which a v9 receiver must never read). Set by the capture tests, never by a receiver.
pub static ACCEPT_V8_HEADERS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// RS parity over the 14 header bytes: 26 bytes = 208 bits = one header copy's cells.
pub const FLAT_HEADER_RS_PARITY: usize = 12;
pub const FLAT_HEADER_BITS: usize = (FLAT_HEADER_BYTES + FLAT_HEADER_RS_PARITY) * 8;
/// Erasures a soft header read gives Reed-Solomon at most (of the 12 parity bytes: two kept for an error).
pub const MAX_ERASED: usize = 10;
/// A cell this near the cut (in half the distance between the two levels) makes its byte a candidate erasure.
const WEAK_CELL: f32 = 0.5;

/// Each cell's confidence: its distance from the two-means cut in half the gap between the two levels'
/// means, positive for a 1 bit (copy 1, drawn inverted, negated).
fn soft_cells(vals: &[u8], inverted: bool) -> Vec<f32> {
    let thr = crate::detect::markers::two_means_threshold(vals);
    let mean = |hi: bool| { let v: Vec<f32> = vals.iter().filter(|&&x| (x > thr) == hi).map(|&x| x as f32).collect(); if v.is_empty() { thr as f32 } else { v.iter().sum::<f32>() / v.len() as f32 } };
    let half = ((mean(true) - mean(false)) / 2.0).max(1.0);
    vals.iter().map(|&v| { let c = (v as f32 - thr as f32 - 0.5) / half; if inverted { -c } else { c } }).collect()
}

/// One header copy as read: a header of this version, a LensBolt header of another version (read whole: RS,
/// CRC and magic; the number is its version field), or nothing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HeaderCopy { Header(FlatHeader), OtherVersion(u8), Unreadable }
impl HeaderCopy {
    pub fn header(self) -> Option<FlatHeader> { if let HeaderCopy::Header(h) = self { Some(h) } else { None } }
    pub fn other_version(self) -> Option<u8> { if let HeaderCopy::OtherVersion(v) = self { Some(v) } else { None } }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FlatHeader {
    pub format: CellFormat, pub cols: u16, pub rows: u16, pub frame_type: FrameType,
    pub transfer_id: u16, pub segment: u16, pub block_id: u32,
    /// Display frame counter (mod 256): consecutive frames differ by one, so a capture that straddles two
    /// frames can tell which blocks belong to which. Its low two bits colour two pilots per block.
    pub frame_seq: u8,
}

impl FlatHeader {
    /// magic(4) version(4) pitch index(3) alphabet(3) cols(10) rows(10) type(2) transfer(16) segment(12)
    /// block_id(24) frame_seq(8) = 96 bits, then CRC-16 big-endian.
    pub fn pack(&self) -> [u8; FLAT_HEADER_BYTES] {
        let mut w = BitWriter::new();
        w.write(HEADER_MAGIC as u32, 4);
        w.write(FLAT_VERSION as u32, 4);
        w.write(self.format.pitch_index().expect("pitch not in PITCHES") as u32, 3);
        w.write(self.format.alphabet.id() as u32, 3);
        w.write(self.cols as u32 & 0x3FF, 10);
        w.write(self.rows as u32 & 0x3FF, 10);
        w.write(self.frame_type as u32, 2);
        w.write(self.transfer_id as u32, 16);
        w.write(self.segment as u32 & 0xFFF, 12);
        w.write(self.block_id & 0xFF_FFFF, 24);
        w.write(self.frame_seq as u32, 8);
        let body = w.finish();
        debug_assert_eq!(body.len(), 12);
        let mut out = [0u8; FLAT_HEADER_BYTES];
        out[..12].copy_from_slice(&body);
        out[12..].copy_from_slice(&crc16(&body).to_be_bytes());
        out
    }
    pub fn unpack(bytes: &[u8; FLAT_HEADER_BYTES]) -> Result<FlatHeader, HeaderError> {
        let crc = u16::from_be_bytes([bytes[12], bytes[13]]);
        if crc != crc16(&bytes[..12]) { return Err(HeaderError::BadCrc); }
        let mut r = BitReader::new(&bytes[..12]);
        if r.read(4) as u8 != HEADER_MAGIC { return Err(HeaderError::BadMagic); }
        let v = r.read(4) as u8;
        let diag = |flag: &std::sync::atomic::AtomicBool| flag.load(std::sync::atomic::Ordering::Relaxed);
        if v != FLAT_VERSION && !(v == 7 && diag(&ACCEPT_V7_HEADERS)) && !(v == 8 && diag(&ACCEPT_V8_HEADERS)) { return Err(HeaderError::BadVersion); }
        let (pi, ai) = (r.read(3) as u8, r.read(3) as u8);
        let format = CellFormat::from_indices(pi, ai).ok_or(HeaderError::BadMode)?;
        let cols = r.read(10) as u16;
        let rows = r.read(10) as u16;
        let frame_type = match r.read(2) { 0 => FrameType::Data, 1 => FrameType::Manifest, 2 => FrameType::Probe, _ => return Err(HeaderError::BadType) };
        let transfer_id = r.read(16) as u16;
        let segment = r.read(12) as u16;
        let block_id = r.read(24);
        let frame_seq = r.read(8) as u8;
        Ok(FlatHeader { format, cols, rows, frame_type, transfer_id, segment, block_id, frame_seq })
    }
    pub fn info(&self) -> FrameInfo { FrameInfo { frame_type: self.frame_type, transfer_id: self.transfer_id, segment: self.segment, block_id: self.block_id } }
    /// The 208 header cells: RS(12) codeword of `pack()`, whitened, MSB first.
    pub fn to_bits(&self) -> Vec<bool> { FlatHeader::header_bits(&self.pack()) }
    /// The 208 header cells of any 14 header bytes (`to_bits` is this of `pack()`): a test draws another
    /// version's header with it.
    pub fn header_bits(bytes: &[u8; FLAT_HEADER_BYTES]) -> Vec<bool> {
        let mut cw = RsCodec::new(FLAT_HEADER_RS_PARITY).encode(bytes);
        crate::interleave::whiten(&mut cw);
        cw.iter().flat_map(|&b| (0..8).map(move |k| (b >> (7 - k)) & 1 == 1)).collect()
    }
    pub fn from_bits(bits: &[bool]) -> Option<FlatHeader> { FlatHeader::copy_from_bits(bits).header() }
    /// `from_bits`, telling a header of another version from an unreadable copy.
    /// A header copy from its cells' brightness (`inverted`: copy 1, drawn light on dark), soft: the bits as
    /// thresholded, then, when Reed-Solomon cannot mend those, again with the bytes whose weakest cell sits
    /// nearest the cut given as erasures (one parity byte each instead of two: up to [MAX_ERASED] of the 12).
    /// A phone camera at 2 camera px per cell reads many header cells grey near the cut (2026-11-03: a dozen
    /// of 208 on whole captures); the CRC, the magic and the version still guard every candidate.
    pub fn copy_from_values(vals: &[u8], inverted: bool) -> HeaderCopy { FlatHeader::from_soft(&soft_cells(vals, inverted)) }
    /// Both copies' cells as one: their confidences summed (copy 1 inverted), so a cell grey in one and clear in
    /// the other is decided. For a capture of one frame; two copies of different frames (a split capture)
    /// sum to nothing the CRC accepts, and only a whole header is returned.
    pub fn combined_from_values(v0: &[u8], v1: &[u8]) -> HeaderCopy {
        let (a, b) = (soft_cells(v0, false), soft_cells(v1, true));
        if a.len() != b.len() { return HeaderCopy::Unreadable; }
        let sum: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        match FlatHeader::from_soft(&sum) { HeaderCopy::Header(h) => HeaderCopy::Header(h), _ => HeaderCopy::Unreadable }
    }
    fn from_soft(cells: &[f32]) -> HeaderCopy {
        let bits: Vec<bool> = cells.iter().map(|&c| c > 0.0).collect();
        let hard = FlatHeader::copy_from_bits(&bits);
        if !matches!(hard, HeaderCopy::Unreadable) || cells.len() != FLAT_HEADER_BITS { return hard; }
        let mut cw: Vec<u8> = bits.chunks(8).map(|c| c.iter().fold(0u8, |acc, &b| (acc << 1) | b as u8)).collect();
        crate::interleave::whiten(&mut cw);
        // bytes by their weakest cell, the weakest first (whitening flips bits, not byte positions)
        let mut weak: Vec<(f32, usize)> = cells.chunks(8).enumerate().map(|(i, c)| (c.iter().fold(f32::MAX, |m, x| m.min(x.abs())), i)).collect();
        weak.sort_by(|a, b| a.0.total_cmp(&b.0));
        let rs = RsCodec::new(FLAT_HEADER_RS_PARITY);
        for n in (2..=MAX_ERASED).step_by(2) {
            let erased: Vec<usize> = weak.iter().take(n).filter(|w| w.0 < WEAK_CELL).map(|w| w.1).collect();
            if erased.is_empty() { break; }
            let mut fixed = cw.clone();
            if rs.decode(&mut fixed, &erased).is_ok() {
                let mut b = [0u8; FLAT_HEADER_BYTES];
                b.copy_from_slice(&fixed[..FLAT_HEADER_BYTES]);
                if let Ok(h) = FlatHeader::unpack(&b) { return HeaderCopy::Header(h); }
            }
            if erased.len() < n { break; }
        }
        hard
    }
    pub fn copy_from_bits(bits: &[bool]) -> HeaderCopy {
        if bits.len() != FLAT_HEADER_BITS { return HeaderCopy::Unreadable; }
        let mut cw: Vec<u8> = bits.chunks(8).map(|c| c.iter().fold(0u8, |acc, &b| (acc << 1) | b as u8)).collect();
        crate::interleave::whiten(&mut cw);
        if RsCodec::new(FLAT_HEADER_RS_PARITY).decode(&mut cw, &[]).is_err() { return HeaderCopy::Unreadable; }
        let mut b = [0u8; FLAT_HEADER_BYTES];
        b.copy_from_slice(&cw[..FLAT_HEADER_BYTES]);
        match FlatHeader::unpack(&b) {
            Ok(h) => HeaderCopy::Header(h),
            // unpack checks the CRC and the magic before the version: magic(4) version(4) is the first byte
            Err(HeaderError::BadVersion) => HeaderCopy::OtherVersion(b[0] & 0x0F),
            Err(_) => HeaderCopy::Unreadable,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flat::{Alphabet, CellFormat};
    use crate::header::{FrameInfo, FrameType, HeaderError};

    fn sample() -> FlatHeader {
        FlatHeader { format: CellFormat { pitch: 4, alphabet: Alphabet::C8 }, cols: 252, rows: 244, frame_type: FrameType::Data, transfer_id: 0xBEEF, segment: 513, block_id: 0x123456, frame_seq: 0 } }
    #[test]
    fn pack_unpack_round_trip() {
        let h = sample();
        let b = h.pack();
        assert_eq!(b.len(), FLAT_HEADER_BYTES);
        assert_eq!(b[0], 0xA9);
        assert_eq!(FlatHeader::unpack(&b), Ok(h));
        for ft in [FrameType::Data, FrameType::Manifest, FrameType::Probe] {
            for a in Alphabet::ALL { for p in crate::flat::PITCHES {
                let h = FlatHeader { frame_type: ft, format: CellFormat { pitch: p, alphabet: a }, ..sample() };
                assert_eq!(FlatHeader::unpack(&h.pack()), Ok(h));
            } }
        }
        assert_eq!(h.info(), FrameInfo { frame_type: FrameType::Data, transfer_id: 0xBEEF, segment: 513, block_id: 0x123456 });
    }
    #[test]
    fn cols_and_rows_up_to_1023_round_trip() {
        for (cols, rows) in [(634u16, 360u16), (1023, 1023), (252, 244)] {
            let h = FlatHeader { cols, rows, ..sample() };
            assert_eq!(FlatHeader::unpack(&h.pack()), Ok(h));
        }
    }
    #[test]
    fn bits_round_trip_survives_five_byte_errors() {
        let h = sample();
        let bits = h.to_bits();
        assert_eq!(bits.len(), 208);
        assert_eq!(FlatHeader::from_bits(&bits), Some(h));
        let mut damaged = bits.clone();
        for byte in [3usize, 7, 11, 15, 19] { for k in 0..8 { damaged[byte * 8 + k] = !damaged[byte * 8 + k]; } }
        assert_eq!(FlatHeader::from_bits(&damaged), Some(h));
        let mut worse = bits.clone();
        for byte in [1usize, 4, 7, 10, 13, 16] { for k in 0..8 { worse[byte * 8 + k] = !worse[byte * 8 + k]; } }
        let r = FlatHeader::from_bits(&worse);
        assert!(r.is_none() || r == Some(h));
        assert_eq!(FlatHeader::from_bits(&bits[..100]), None);
    }
    #[test]
    fn rejects_v1_header_bytes() {
        // a version-4 (tile format) header: magic 0xA, version 4, then anything, with a valid CRC
        let mut b = sample().pack();
        for old in [0xA4u8, 0xA7, 0xA8] {
            b[0] = old; // v1 tiles, v7 (stripe codewords: its cells mean something else), v8 (other repair packets)
            let c = crate::crc16::crc16(&b[..12]).to_be_bytes(); b[12] = c[0]; b[13] = c[1];
            assert_eq!(FlatHeader::unpack(&b), Err(HeaderError::BadVersion));
        }
        let mut c = sample().pack(); c[5] ^= 0x10;
        assert_eq!(FlatHeader::unpack(&c), Err(HeaderError::BadCrc));
    }
    #[test]
    fn a_header_copy_of_another_version_says_which() {
        // read whole (RS, CRC, magic) with another version: OtherVersion, not Unreadable; a bad CRC or magic is
        // Unreadable whatever its version field says
        let with = |first: u8, crc_ok: bool| {
            let mut b = sample().pack(); b[0] = first;
            let c = crate::crc16::crc16(&b[..12]).to_be_bytes(); b[12] = c[0]; b[13] = c[1] ^ (!crc_ok) as u8;
            FlatHeader::copy_from_bits(&FlatHeader::header_bits(&b))
        };
        for v in [4u8, 7, 8, 10, 15] { assert_eq!(with(0xA0 | v, true), HeaderCopy::OtherVersion(v)); }
        assert_eq!(with(0xA0 | FLAT_VERSION, true), HeaderCopy::Header(sample()));
        assert_eq!(with(0xA8, false), HeaderCopy::Unreadable);
        assert_eq!(with(0xB8, true), HeaderCopy::Unreadable);
        assert_eq!(FlatHeader::copy_from_bits(&[false; 100]), HeaderCopy::Unreadable);
    }
    #[test]
    fn frame_sequence_rides_in_the_pad_bits() {
        let h = FlatHeader { frame_seq: 137, ..sample() };
        assert_eq!(FlatHeader::unpack(&h.pack()).unwrap(), h);
        assert_ne!(h.pack(), sample().pack());
        assert_eq!(FLAT_VERSION, 9, "v9 repair packets: v8 frames must not decode");
    }

    /// Cell values of a header copy as a camera sees it: 1 bits at `hi`, 0 bits at `lo` (copy 1 drawn inverted).
    fn cell_values(h: &FlatHeader, inverted: bool) -> Vec<u8> {
        h.to_bits().iter().map(|&b| if b ^ inverted { 210 } else { 40 }).collect()
    }
    /// Bit `k` of byte `byte` pushed just past the cut, to the wrong side (a blurred cell between two others).
    fn smudge(v: &mut [u8], byte: usize, k: usize) { let i = byte * 8 + k; v[i] = if v[i] > 125 { 115 } else { 135 }; }

    // a phone camera's header at 2 camera px per cell (2026-11-03): many cells read grey near the cut, a
    // couple of wrong bits in several bytes; as erasures (the bytes whose weakest cell sits nearest the cut)
    // Reed-Solomon takes up to 12 bytes instead of 6
    #[test]
    fn copy_from_values_erases_the_weak_bytes() {
        let h = sample();
        let mut v = cell_values(&h, false);
        for byte in [0, 3, 5, 8, 11, 14, 17, 20, 23, 25] { smudge(&mut v, byte, byte % 8); }
        let hard: Vec<bool> = v.iter().map(|&x| x > 125).collect();
        assert_eq!(FlatHeader::copy_from_bits(&hard), HeaderCopy::Unreadable, "10 bytes wrong: past the 6 errors RS mends");
        assert_eq!(FlatHeader::copy_from_values(&v, false), HeaderCopy::Header(h));
        let mut w = cell_values(&h, true);
        for byte in [1, 4, 9, 12, 15, 18, 21, 24] { smudge(&mut w, byte, 3); }
        assert_eq!(FlatHeader::copy_from_values(&w, true), HeaderCopy::Header(h));
    }

    // the two copies show the same header (a capture of one frame): a cell grey in one is clear in the other,
    // so their confidences summed decide it; each alone has 13 weak bytes, past the 12 erasures RS takes
    #[test]
    fn combined_copies_read_what_neither_reads_alone() {
        let h = sample();
        let (mut a, mut b) = (cell_values(&h, false), cell_values(&h, true));
        for byte in 0..13 { smudge(&mut a, byte, 2); }
        for byte in 13..26 { smudge(&mut b, byte, 5); }
        assert_eq!(FlatHeader::copy_from_values(&a, false), HeaderCopy::Unreadable);
        assert_eq!(FlatHeader::copy_from_values(&b, true), HeaderCopy::Unreadable);
        assert_eq!(FlatHeader::combined_from_values(&a, &b), HeaderCopy::Header(h));
    }

    // two copies of different frames (a capture split between two): no combined header made up of both
    #[test]
    fn combined_copies_of_two_frames_read_nothing() {
        let (h, mut g) = (sample(), sample());
        g.block_id ^= 0x5A5A5A; g.frame_seq ^= 0x33;
        assert_eq!(FlatHeader::combined_from_values(&cell_values(&h, false), &cell_values(&g, true)), HeaderCopy::Unreadable);
    }
}
