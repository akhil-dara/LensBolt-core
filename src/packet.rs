use crate::crc16::crc16;
use crate::flat::{FlatGrid, FramePayload};

pub const PACKET_PAYLOAD: usize = 1024;
pub const PACKET_BYTES: usize = 1030;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct PacketId { pub segment: u16, pub chunk: u8, pub pid: u16 }
impl PacketId {
    pub fn to_u32(self) -> u32 { (self.segment as u32) << 16 | ((self.chunk as u32) & 0xF) << 12 | (self.pid as u32 & 0xFFF) }
    pub fn from_u32(v: u32) -> PacketId { PacketId { segment: (v >> 16) as u16, chunk: ((v >> 12) & 0xF) as u8, pid: (v & 0xFFF) as u16 } }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Packet { pub id: PacketId, pub payload: [u8; PACKET_PAYLOAD] }
impl Packet {
    pub fn to_wire(&self) -> [u8; PACKET_BYTES] {
        let mut w = [0u8; PACKET_BYTES];
        w[..4].copy_from_slice(&self.id.to_u32().to_be_bytes());
        w[4..4 + PACKET_PAYLOAD].copy_from_slice(&self.payload);
        let c = crc16(&w[..4 + PACKET_PAYLOAD]).to_be_bytes();
        w[PACKET_BYTES - 2..].copy_from_slice(&c);
        w
    }
    pub fn from_wire(b: &[u8]) -> Option<Packet> {
        if b.len() != PACKET_BYTES { return None; }
        let c = u16::from_be_bytes([b[PACKET_BYTES - 2], b[PACKET_BYTES - 1]]);
        if c != crc16(&b[..PACKET_BYTES - 2]) { return None; }
        let id = PacketId::from_u32(u32::from_be_bytes([b[0], b[1], b[2], b[3]]));
        let mut payload = [0u8; PACKET_PAYLOAD];
        payload.copy_from_slice(&b[4..4 + PACKET_PAYLOAD]);
        Some(Packet { id, payload })
    }
}
/// Packets back to back over the frame payload (v8); the tail after the last packet is zero.
pub fn packets_per_frame(grid: &FlatGrid) -> usize { grid.packets() }
pub fn pack_frame(packets: &[Packet], grid: &FlatGrid) -> Vec<u8> {
    assert!(packets.len() <= packets_per_frame(grid));
    let mut out = vec![0u8; grid.payload_bytes()];
    for (k, p) in packets.iter().enumerate() { out[k * PACKET_BYTES..(k + 1) * PACKET_BYTES].copy_from_slice(&p.to_wire()); }
    out
}
/// Packets whose bytes are all valid and whose CRC passes; all-zero slots are padding.
pub fn unpack_frame(fp: &FramePayload) -> Vec<Packet> {
    let n = fp.bytes.len() / PACKET_BYTES;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let range = i * PACKET_BYTES..(i + 1) * PACKET_BYTES;
        if !fp.valid[range.clone()].iter().all(|&v| v) { continue; }
        let slot = &fp.bytes[range];
        if slot.iter().all(|&b| b == 0) { continue; }
        if let Some(p) = Packet::from_wire(slot) { out.push(p); }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn id_packs_into_u32() {
        let id = PacketId { segment: 0xBEEF, chunk: 0xA, pid: 0xFFF };
        assert_eq!(id.to_u32(), 0xBEEF_AFFF);
        assert_eq!(PacketId::from_u32(0xBEEF_AFFF), id);
    }
    #[test]
    fn wire_round_trip_and_crc() {
        let p = Packet { id: PacketId { segment: 1, chunk: 2, pid: 3 }, payload: [7u8; 1024] };
        let w = p.to_wire();
        assert_eq!(w.len(), 1030);
        assert_eq!(Packet::from_wire(&w), Some(p.clone()));
        let mut bad = w; bad[500] ^= 1;
        assert_eq!(Packet::from_wire(&bad), None);
        assert_eq!(Packet::from_wire(&w[..1029]), None);
    }
    #[test]
    fn frame_packing_and_unpacking() {
        let g = FlatGrid::try_new(1024, 1024, crate::flat::CellFormat { pitch: 4, alphabet: crate::flat::Alphabet::C8 }).unwrap();
        let n = packets_per_frame(&g);
        assert_eq!(n, g.packets());
        assert!(n >= 17, "1024 tile at 4 px should carry >= 17 packets, got {n}");
        let ps: Vec<Packet> = (0..n as u16).map(|i| Packet { id: PacketId { segment: 1, chunk: 0, pid: i }, payload: [i as u8; 1024] }).collect();
        let bytes = pack_frame(&ps, &g);
        assert_eq!(bytes.len(), g.payload_bytes());
        assert_eq!(&bytes[5 * PACKET_BYTES..5 * PACKET_BYTES + 4], &ps[5].id.to_u32().to_be_bytes(), "packets sit back to back");
        let mut fp = FramePayload { valid: vec![true; bytes.len()], bytes, codewords_ok: g.codewords(), codewords_failed: 0, codewords_missing: 0, coverage: 1.0 };
        assert_eq!(unpack_frame(&fp), ps);
        for v in &mut fp.valid[5 * PACKET_BYTES + 10..5 * PACKET_BYTES + 12] { *v = false; }
        let got = unpack_frame(&fp);
        assert_eq!(got.len(), n - 1);
        assert!(got.iter().all(|p| p.id.pid != 5));
    }
}
