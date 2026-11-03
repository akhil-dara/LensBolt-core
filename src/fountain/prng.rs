pub struct Xorshift64(pub u64);
impl Xorshift64 {
    pub fn new(seed: u64) -> Self { Xorshift64(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed }) }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: u32) -> u32 { (((self.next_u64() >> 32) as u64 * n as u64) >> 32) as u32 }
}
