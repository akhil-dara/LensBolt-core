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
}
