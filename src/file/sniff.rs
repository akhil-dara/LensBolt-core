pub const SNIFF_BYTES: usize = 4 * 1024 * 1024;

/// true if zstd-3 shrinks `sample` below 95%.
pub fn should_compress(sample: &[u8]) -> bool {
    if sample.len() < 1024 { return false; }
    #[cfg(feature = "zstd")]
    { match zstd::bulk::compress(sample, 3) { Ok(c) => (c.len() as f64) < sample.len() as f64 * 0.95, Err(_) => false } }
    #[cfg(not(feature = "zstd"))]
    { false }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn text_compresses_random_does_not() {
        let text: Vec<u8> = b"the quick brown fox jumps over the lazy dog. ".iter().cycle().take(200_000).copied().collect();
        assert!(should_compress(&text));
        let mut seed = 1u64; let rnd: Vec<u8> = (0..200_000).map(|_| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed as u8 }).collect();
        assert!(!should_compress(&rnd));
        assert!(!should_compress(&[]));
    }
}
