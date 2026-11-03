//! MSB-first bit packing.

pub struct BitWriter { buf: Vec<u8>, nbits: usize }
impl BitWriter {
    pub fn new() -> Self { BitWriter { buf: Vec::new(), nbits: 0 } }
    pub fn write(&mut self, value: u32, nbits: u32) {
        for i in (0..nbits).rev() {
            if self.nbits % 8 == 0 { self.buf.push(0); }
            let bit = ((value >> i) & 1) as u8;
            let idx = self.nbits / 8;
            self.buf[idx] |= bit << (7 - (self.nbits % 8));
            self.nbits += 1;
        }
    }
    pub fn bit_len(&self) -> usize { self.nbits }
    pub fn finish(self) -> Vec<u8> { self.buf }
}
impl Default for BitWriter { fn default() -> Self { Self::new() } }

pub struct BitReader<'a> { data: &'a [u8], pos: usize }
impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self { BitReader { data, pos: 0 } }
    pub fn read(&mut self, nbits: u32) -> u32 {
        let mut v = 0u32;
        for _ in 0..nbits {
            let bit = if self.pos / 8 < self.data.len() { (self.data[self.pos / 8] >> (7 - self.pos % 8)) & 1 } else { 0 };
            v = (v << 1) | bit as u32;
            self.pos += 1;
        }
        v
    }
    pub fn remaining_bits(&self) -> usize { (self.data.len() * 8).saturating_sub(self.pos) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn writes_msb_first_and_reads_back() {
        let mut w = BitWriter::new();
        w.write(0b101, 3); w.write(0xABCD, 16); w.write(1, 1); w.write(0, 4);
        assert_eq!(w.bit_len(), 24);
        let bytes = w.finish();
        assert_eq!(bytes, vec![0b1011_0101, 0b0111_1001, 0b1011_0000]);
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read(3), 0b101); assert_eq!(r.read(16), 0xABCD); assert_eq!(r.read(1), 1);
        assert_eq!(r.remaining_bits(), 4);
        assert_eq!(r.read(10), 0);
    }
    #[test]
    fn partial_last_byte_is_zero_padded() {
        let mut w = BitWriter::new(); w.write(0b11, 2);
        assert_eq!(w.finish(), vec![0b1100_0000]);
    }
}
