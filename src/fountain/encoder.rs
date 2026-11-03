use super::{neighbors, BLOCK_BYTES, MAX_BLOCKS};

pub struct LtEncoder { blocks: Vec<[u8; BLOCK_BYTES]> }

impl LtEncoder {
    /// `data.len()` <= MAX_BLOCKS*BLOCK_BYTES; last block zero-padded.
    pub fn new(data: &[u8]) -> LtEncoder {
        assert!(data.len() <= MAX_BLOCKS * BLOCK_BYTES && !data.is_empty());
        let blocks = data.chunks(BLOCK_BYTES).map(|c| { let mut b = [0u8; BLOCK_BYTES]; b[..c.len()].copy_from_slice(c); b }).collect();
        LtEncoder { blocks }
    }
    pub fn k(&self) -> usize { self.blocks.len() }
    pub fn packet(&self, pid: u32) -> [u8; BLOCK_BYTES] {
        let mut out = [0u8; BLOCK_BYTES];
        for i in neighbors(pid, self.k()) { xor_into(&mut out, &self.blocks[i as usize]); }
        out
    }
}

#[inline]
pub(crate) fn xor_into(dst: &mut [u8; BLOCK_BYTES], src: &[u8; BLOCK_BYTES]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) { *d ^= *s; }
}
