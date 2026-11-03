//! Payload bytes -> cell values (format v8: frame-wide RS codewords, whitening, bit packing) plus the chrome.

use super::header::FlatHeader;
use super::layout::{FlatCell, FlatGrid};
use crate::interleave::{interleave, whiten};
use crate::rs::RsCodec;
use std::sync::Arc;

/// A composed frame: the palette index of every cell (row-major, cols x rows). Unused cells and data
/// cells beyond the codeword stream are 0.
pub struct FlatFrame { pub grid: Arc<FlatGrid>, pub header: FlatHeader, pub values: Vec<u8> }

impl FlatFrame {
    pub fn value(&self, col: u32, row: u32) -> u8 { self.values[(row * self.grid.cols() + col) as usize] }
}

/// Pack bytes into cells of `bits` bits, MSB first; the last cell is zero-padded.
pub fn stripe_cells(bytes: &[u8], bits: u32) -> Vec<u8> {
    let total = bytes.len() * 8;
    let n = total.div_ceil(bits as usize);
    let mut out = Vec::with_capacity(n);
    let mut pos = 0usize;
    for _ in 0..n {
        let mut v = 0u8;
        for _ in 0..bits {
            let bit = if pos < total { (bytes[pos / 8] >> (7 - pos % 8)) & 1 } else { 0 };
            v = (v << 1) | bit;
            pos += 1;
        }
        out.push(v);
    }
    out
}

/// Inverse of `stripe_cells`: the whole bytes the cells' bits form (padding bits dropped).
pub fn cells_to_stripe(cells: &[u8], bits: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(cells.len() * bits as usize / 8);
    // a shift register: cells enter at the bottom, whole bytes leave from the top
    let (mask, mut acc, mut held) = ((1u32 << bits) - 1, 0u32, 0u32);
    for &c in cells {
        acc = (acc << bits) | (c as u32 & mask);
        held += bits;
        if held >= 8 { held -= 8; out.push((acc >> held) as u8); acc &= (1 << held) - 1; }
    }
    out
}

/// The frame's whitened byte stream: payload split into N codewords (codeword c = payload bytes
/// [c*data_len, (c+1)*data_len) + parity), byte j of codeword c at stream position j*N + c.
pub fn frame_stream(grid: &FlatGrid, payload: &[u8]) -> Vec<u8> {
    assert_eq!(payload.len(), grid.payload_bytes(), "payload must fill the frame exactly");
    let fmt = grid.format();
    let rs = RsCodec::new(fmt.parity());
    let cws: Vec<Vec<u8>> = payload.chunks(fmt.data_len()).map(|d| rs.encode(d)).collect();
    debug_assert_eq!(cws.len(), grid.codewords());
    let mut stream = interleave(&cws);
    whiten(&mut stream);
    stream
}

pub fn compose(grid: Arc<FlatGrid>, header: FlatHeader, payload: &[u8]) -> FlatFrame {
    let stream = frame_stream(&grid, payload);
    let bits = grid.format().alphabet.bits();
    let idx = &grid.data_indices()[..grid.used_data_cells()];
    let mut values = vec![0u8; (grid.cols() * grid.rows()) as usize];
    if 8 % bits == 0 {
        // 1, 2 or 4 bits a cell: each stream byte unpacks straight into its cells' places (a 1 px frame's
        // million cells spent 2.8 ms striped into a list and then placed one position lookup at a time)
        let (per, mask) = ((8 / bits) as usize, (1u8 << bits) - 1);
        for (chunk, &b) in idx.chunks(per).zip(&stream) {
            for (s, &i) in chunk.iter().enumerate() { values[i as usize] = (b >> (8 - bits as usize * (s + 1))) & mask; }
        }
    } else {
        let cells = stripe_cells(&stream, bits);
        debug_assert_eq!(cells.len(), idx.len());
        for (&i, &v) in idx.iter().zip(&cells) { values[i as usize] = v; }
    }
    let tag = header.frame_seq & 3;
    for &(c, r) in grid.pilot_positions() {
        if let FlatCell::Pilot { .. } = grid.cell(c, r) { values[(r * grid.cols() + c) as usize] = grid.pilot_colour_at(c, r, tag); }
    }
    FlatFrame { grid, header, values }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flat::{Alphabet, CellFormat, FlatGrid, FlatCell, FlatHeader};
    use crate::header::FrameType;
    use crate::rs::RsCodec;
    use std::sync::Arc;

    pub(crate) fn hdr(g: &FlatGrid) -> FlatHeader {
        FlatHeader { format: g.format(), cols: g.cols() as u16, rows: g.rows() as u16, frame_type: FrameType::Data, transfer_id: 7, segment: 1, block_id: 99, frame_seq: 0 } }
    #[test]
    fn bit_packing_round_trips() {
        let bytes: Vec<u8> = (0..1275u32).map(|i| (i.wrapping_mul(2654435761u32) >> 13) as u8).collect();
        for bits in 1..=4u32 {
            let cells = stripe_cells(&bytes, bits);
            assert_eq!(cells.len(), 1275 * 8 / bits as usize);
            assert!(cells.iter().all(|&c| (c as u32) < (1 << bits)));
            assert_eq!(cells_to_stripe(&cells, bits), bytes);
        }
        assert_eq!(stripe_cells(&[0b1011_0001], 3), vec![0b101, 0b100, 0b01 << 1]); // 8 bits -> 2 full cells + 2 bits padded with a zero
    }
    /// compose as it was: striped cells placed one data position at a time (the reference the fused unpacking
    /// must match cell for cell)
    fn compose_reference(grid: &FlatGrid, header: FlatHeader, payload: &[u8]) -> Vec<u8> {
        let cells = stripe_cells(&frame_stream(grid, payload), grid.format().alphabet.bits());
        let mut values = vec![0u8; (grid.cols() * grid.rows()) as usize];
        for (i, &v) in cells.iter().enumerate() { let (c, r) = grid.data_position(i as u32); values[(r * grid.cols() + c) as usize] = v; }
        let tag = header.frame_seq & 3;
        for &(c, r) in grid.pilot_positions() { if let FlatCell::Pilot { .. } = grid.cell(c, r) { values[(r * grid.cols() + c) as usize] = grid.pilot_colour_at(c, r, tag); } }
        values
    }
    #[test]
    fn compose_matches_the_reference_for_every_alphabet() {
        for alphabet in Alphabet::ALL {
            for pitch in [1u8, 2, 3, 4] {
                let fmt = CellFormat { pitch, alphabet };
                let Some(g) = FlatGrid::try_new(1024, 1024, fmt) else { continue };
                let g = Arc::new(g);
                let payload: Vec<u8> = (0..g.payload_bytes()).map(|i| ((i as u64).wrapping_mul(2654435761) >> 11) as u8).collect();
                let mut h = hdr(&g);
                h.frame_seq = 5;
                assert!(compose(g.clone(), h, &payload).values == compose_reference(&g, h, &payload), "{fmt:?}");
            }
        }
    }
    #[test]
    fn compose_places_header_pilots_and_data() {
        let g = Arc::new(FlatGrid::try_new(1024, 1024, CellFormat { pitch: 4, alphabet: Alphabet::C8 }).unwrap());
        let payload: Vec<u8> = (0..g.payload_bytes()).map(|i| (i * 31 % 251) as u8).collect();
        let f = compose(g.clone(), hdr(&g), &payload);
        assert_eq!(f.values.len(), (g.cols() * g.rows()) as usize);
        let FlatCell::Pilot { colour } = g.cell(18, 14) else { panic!() };
        assert_eq!(f.value(18, 14), colour);
        // v8: codeword c holds payload bytes [c*223, (c+1)*223); byte j of codeword c is stream byte j*N + c
        let n = g.codewords();
        let rs = RsCodec::new(32);
        let cws: Vec<Vec<u8>> = payload.chunks(223).map(|d| rs.encode(d)).collect();
        assert_eq!(cws.len(), n);
        let mut stream = vec![0u8; n * 255];
        for (c, cw) in cws.iter().enumerate() { for (j, &v) in cw.iter().enumerate() { stream[j * n + c] = v; } }
        crate::interleave::whiten(&mut stream);
        assert_eq!(frame_stream(&g, &payload), stream);
        let cells = stripe_cells(&stream, 3);
        assert_eq!(cells.len(), g.used_data_cells());
        for i in [0u32, 1, 2, 3399, 3400, g.used_data_cells() as u32 - 1] { let (c, r) = g.data_position(i); assert_eq!(f.value(c, r), cells[i as usize], "cell {i}"); }
        // cells past the stream are 0
        for i in g.used_data_cells() as u32..g.data_cell_count() { let (c, r) = g.data_position(i); assert_eq!(f.value(c, r), 0); }
        assert_eq!(f.value(251, 243), 0); // unused cells are 0
    }
    #[test]
    #[should_panic]
    fn wrong_payload_length_panics() {
        let g = Arc::new(FlatGrid::try_new(1024, 1024, CellFormat { pitch: 4, alphabet: Alphabet::C8 }).unwrap());
        compose(g.clone(), hdr(&g), &[0u8; 10]);
    }
    #[test]
    fn frame_sequence_bits_colour_two_pilots_per_block() {
        use crate::flat::layout::{PILOT_LOCAL, BLOCK};
        let g = Arc::new(FlatGrid::try_new(1024, 1024, CellFormat { pitch: 4, alphabet: Alphabet::C8 }).unwrap());
        let h = FlatHeader { format: g.format(), cols: g.cols() as u16, rows: g.rows() as u16, frame_type: FrameType::Data, transfer_id: 7, segment: 1, block_id: 99, frame_seq: 0 };
        let payload = vec![0u8; g.payload_bytes()];
        let base = compose(g.clone(), h, &payload);
        for (seq, pilot_k) in [(1u8, 1usize), (2, 3), (5, 1)] {
            let other = compose(g.clone(), FlatHeader { frame_seq: seq, ..h }, &payload);
            let changed: Vec<(u32, u32)> = (0..g.rows()).flat_map(|r| (0..g.cols()).map(move |c| (c, r))).filter(|&(c, r)| base.value(c, r) != other.value(c, r)).collect();
            assert!(!changed.is_empty(), "seq {seq} must change pilot {pilot_k}");
            for &(c, r) in &changed {
                assert_eq!(PILOT_LOCAL.iter().position(|&q| q == (r % BLOCK, c % BLOCK)), Some(pilot_k), "seq {seq}: cell ({c},{r}) changed but is not pilot {pilot_k}");
                let d = (other.value(c, r) as i32 - base.value(c, r) as i32).rem_euclid(8);
                assert_eq!(d, 4, "pilot colour shifts by half the alphabet");
            }
            let pilots_k = g.pilot_positions().iter().filter(|&&(c, r)| PILOT_LOCAL[pilot_k] == (r % BLOCK, c % BLOCK)).count();
            assert_eq!(changed.len(), pilots_k, "every pilot {pilot_k} changes");
        }
        // seq 3 flips both; seq 4 flips neither
        assert_eq!(compose(g.clone(), FlatHeader { frame_seq: 4, ..h }, &payload).values, base.values);
    }
}
