//! What one capture of the Tune carousel showed: the frame it read (its number, format, codewords decoded) and
//! where the tile sat (camera px per screen px along each edge, the corners, the blend, the edge width, the cells'
//! contrast), or that it read no tile.
use crate::flat::session::{edge_sampling, tile_corners};
use crate::flat::{CellFormat, FlatRead};
use crate::header::FrameType;
use crate::packet::PACKET_BYTES;

/// One frame read from a capture. `packets_ok` of `packets` (the grid's packet slots per frame) came out whole from
/// this capture alone: what the transfer receives (a packet spans 5-6 codewords and needs all of them). `codewords_ok`
/// of `codewords` decoded, for the details. A manifest read counts 0 of 0 (it says where the tile is, not how the
/// data reads). `edges`: camera px per screen px along top, right, bottom, left; `corners`: TL, TR, BR, BL as x, y
/// in the camera picture; `contrast`: the mean cell margin (the screen's flicker and the exposure move it).
#[derive(Clone, Debug, PartialEq)]
pub struct SampleRead { pub seq: u8, pub format: CellFormat, pub manifest: bool, pub packets_ok: u32, pub packets: u32, pub codewords_ok: u32, pub codewords: u32, pub tile: (u32, u32), pub edges: [f32; 4], pub corners: [f32; 8], pub mix: f32, pub edge_px: f32, pub contrast: f32, pub decode_ms: f64 }

/// Packet slots of a payload whose bytes are all valid (the `packet::unpack_frame` rule, without the CRC: a slot
/// of decoded codewords passes it).
pub fn packets_valid(valid: &[bool], slots: usize) -> u32 {
    (0..slots).filter(|&k| valid.get(k * PACKET_BYTES..(k + 1) * PACKET_BYTES).is_some_and(|v| v.iter().all(|&b| b))).count() as u32
}
/// The same from which codewords decoded: the payload is the codewords' data bytes back to back (`data_len` each),
/// so packet slot k needs the codewords its bytes fall in.
pub fn packets_from_codewords(ok: &[bool], data_len: usize, slots: usize) -> u32 {
    (0..slots).filter(|&k| {
        let (a, b) = (k * PACKET_BYTES / data_len, ((k + 1) * PACKET_BYTES - 1) / data_len);
        (a..=b).all(|c| ok.get(c).copied().unwrap_or(false))
    }).count() as u32
}

impl SampleRead {
    /// The sample of a read (a data read's payload is decoded here: its codewords).
    pub fn of(fr: &FlatRead) -> SampleRead {
        let manifest = fr.header.frame_type == FrameType::Manifest;
        let (ok, all, pk_ok, pk) = if manifest { (0, 0, 0, 0) } else {
            let p = fr.payload();
            let slots = fr.grid.packets();
            (p.codewords_ok as u32, fr.grid.codewords() as u32, packets_valid(&p.valid, slots), slots as u32)
        };
        let m = &fr.reads.margins;
        let contrast = if m.is_empty() { 0.0 } else { (m.iter().map(|&v| v as f64).sum::<f64>() / m.len() as f64) as f32 };
        SampleRead { seq: fr.header.frame_seq, format: fr.header.format, manifest, packets_ok: pk_ok, packets: pk, codewords_ok: ok, codewords: all,
            tile: (fr.grid.width_px(), fr.grid.height_px()), edges: edge_sampling(fr), corners: tile_corners(fr),
            mix: fr.reads.mix, edge_px: fr.reads.edge_px, contrast, decode_ms: fr.decode_ms }
    }
    /// The smallest edge's camera px per screen px.
    pub fn sampling(&self) -> f64 { self.edges.iter().copied().fold(f32::MAX, f32::min) as f64 }
    /// The largest edge over the smallest (1 for a square-on hold).
    pub fn perspective(&self) -> f64 {
        let lo = self.sampling();
        let hi = self.edges.iter().copied().fold(0.0f32, f32::max) as f64;
        if lo > 0.0 { hi / lo } else { 1.0 }
    }
    /// The share of the frame's packets this capture read whole (0 for a manifest).
    pub fn share(&self) -> f64 { if self.packets == 0 { 0.0 } else { self.packets_ok as f64 / self.packets as f64 } }
    /// The share of the frame's codewords this capture decoded (0 for a manifest).
    pub fn codeword_share(&self) -> f64 { if self.codewords == 0 { 0.0 } else { self.codewords_ok as f64 / self.codewords as f64 } }
}

/// One capture at `t_ms` (the receiver's clock): its read, or None when it read no tile.
#[derive(Clone, Debug, PartialEq)]
pub struct TuneSample { pub t_ms: f64, pub read: Option<SampleRead> }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_packet_needs_every_codeword_it_spans() {
        // 231-byte codewords (2 px b/w): packet 0 is bytes 0..1030, codewords 0..=4; packet 1 codewords 4..=8
        let mut ok = vec![true; 20];
        assert_eq!(packets_from_codewords(&ok, 231, 4), 4);
        ok[4] = false;
        assert_eq!(packets_from_codewords(&ok, 231, 4), 2, "codeword 4 is in packets 0 and 1");
        let mut valid = vec![true; 4 * PACKET_BYTES];
        valid[PACKET_BYTES + 7] = false;
        assert_eq!(packets_valid(&valid, 4), 3);
        assert_eq!(packets_valid(&valid[..3 * PACKET_BYTES], 4), 2, "a slot past the payload is not whole");
    }
}
