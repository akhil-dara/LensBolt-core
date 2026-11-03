/// Slot index in the frame byte stream for byte `byte` (0..255) of codeword `cw` (0..n_cw).
pub fn slot_of(cw: usize, byte: usize, n_cw: usize) -> usize { byte * n_cw + cw }

/// XOR the slot stream with a fixed pseudo-random mask so a frame never degenerates into one symbol and
/// one colour (a sparse file, or a tiny manifest padded with zeros, would otherwise defeat the per-frame
/// colour calibration and the block alignment, which both need variety). Applying it twice undoes it.
pub fn whiten(slots: &mut [u8]) {
    let mut s: u64 = 0x4C45_4E53_424F_4C54; // "LENSBOLT"
    for chunk in slots.chunks_mut(8) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        let m = s.to_le_bytes();
        for (b, &k) in chunk.iter_mut().zip(m.iter()) { *b ^= k; }
    }
}

/// Stripe layout. The slot stream is a concatenation of stripes of `per` codewords (`per * 255` slots
/// each); inside a stripe, codeword bytes are interleaved and the first 2/3 of its slots are symbol bytes
/// (two 4-bit cells each) and the last 1/3 colour bytes (four 2-bit cells each). Cells are consumed in
/// data order, `per * 340` per stripe, so every stripe is a contiguous band of the frame.
pub const CELLS_PER_CODEWORD: usize = 340;
pub fn stripe_slot(cw: usize, byte: usize, per: usize) -> usize { (cw / per) * per * 255 + byte * per + cw % per }
/// (symbol slot, nibble shift) and (colour slot, bit shift) of data cell `i`.
pub fn cell_slots(i: usize, per: usize) -> ((usize, u32), (usize, u32)) {
    let sc = per * CELLS_PER_CODEWORD;
    let (s, j) = (i / sc, i % sc);
    let base = s * per * 255;
    ((base + j / 2, if j % 2 == 0 { 4 } else { 0 }), (base + per * 170 + j / 4, 6 - 2 * (j % 4) as u32))
}
pub fn stripe_interleave(codewords: &[Vec<u8>], per: usize) -> Vec<u8> {
    assert_eq!(codewords.len() % per, 0);
    let mut out = vec![0u8; 255 * codewords.len()];
    for (cw, data) in codewords.iter().enumerate() {
        assert_eq!(data.len(), 255);
        for (b, &v) in data.iter().enumerate() { out[stripe_slot(cw, b, per)] = v; }
    }
    out
}

pub fn interleave(codewords: &[Vec<u8>]) -> Vec<u8> {
    let n = codewords.len();
    let mut out = vec![0u8; 255 * n];
    for (cw, data) in codewords.iter().enumerate() {
        assert_eq!(data.len(), 255);
        for (b, &v) in data.iter().enumerate() { out[slot_of(cw, b, n)] = v; }
    }
    out
}
pub fn deinterleave(slots: &[u8], n_cw: usize) -> Vec<Vec<u8>> {
    assert_eq!(slots.len(), 255 * n_cw);
    (0..n_cw).map(|cw| (0..255).map(|b| slots[slot_of(cw, b, n_cw)]).collect()).collect()
}
/// First 2/3 of the bytes are the symbol stream, last 1/3 the colour stream.
pub fn split_streams(slots: &[u8]) -> (&[u8], &[u8]) {
    assert_eq!(slots.len() % 3, 0);
    slots.split_at(slots.len() * 2 / 3)
}
pub fn join_streams(symbols: &[u8], colors: &[u8]) -> Vec<u8> {
    let mut v = symbols.to_vec(); v.extend_from_slice(colors); v
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn interleave_is_a_bijection() {
        let cws: Vec<Vec<u8>> = (0..6).map(|c| (0..255).map(|b| (c * 37 + b) as u8).collect()).collect();
        let slots = interleave(&cws);
        assert_eq!(slots.len(), 255 * 6);
        assert_eq!(slots[slot_of(2, 0, 6)], cws[2][0]);
        assert_eq!(slots[slot_of(5, 254, 6)], cws[5][254]);
        assert_eq!(slot_of(1, 1, 6), 7);
        assert_eq!(deinterleave(&slots, 6), cws);
    }
    #[test]
    fn adjacent_codeword_bytes_are_far_apart() {
        assert_eq!(slot_of(0, 1, 165) - slot_of(0, 0, 165), 165);
    }
    #[test]
    fn whitening_is_an_involution_and_spreads_zeros() {
        let mut v = vec![0u8; 255 * 3];
        whiten(&mut v);
        let distinct = v.iter().collect::<std::collections::HashSet<_>>().len();
        assert!(distinct > 100, "mask too regular: {distinct} distinct bytes");
        whiten(&mut v);
        assert!(v.iter().all(|&b| b == 0));
    }
    #[test]
    fn stripe_layout_is_a_bijection_and_local() {
        let per = 5;
        let n = 3 * per;
        let mut seen = vec![false; n * 255];
        for cw in 0..n { for b in 0..255 { let s = stripe_slot(cw, b, per); assert!(!seen[s]); seen[s] = true; assert_eq!(s / (per * 255), cw / per); } }
        assert!(seen.iter().all(|&v| v));
        // every cell of a stripe reads only that stripe's slots, and the cells cover every slot bit once
        let mut bits = vec![0u8; n * 255];
        for i in 0..n * CELLS_PER_CODEWORD {
            let ((ss, sh), (cs, ch)) = cell_slots(i, per);
            assert_eq!(ss / (per * 255), i / (per * CELLS_PER_CODEWORD));
            assert_eq!(cs / (per * 255), i / (per * CELLS_PER_CODEWORD));
            assert_eq!(bits[ss] & (0xF << sh), 0); bits[ss] |= 0xF << sh;
            assert_eq!(bits[cs] & (3 << ch), 0); bits[cs] |= 3 << ch;
        }
        assert!(bits.iter().all(|&b| b == 0xFF));
    }
    #[test]
    fn stream_split_two_thirds_one_third() {
        let slots: Vec<u8> = (0..255 * 3).map(|i| i as u8).collect();
        let (s, c) = split_streams(&slots);
        assert_eq!((s.len(), c.len()), (510, 255));
        assert_eq!(join_streams(s, c), slots);
    }
}
