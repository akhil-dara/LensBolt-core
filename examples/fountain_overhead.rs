//! Fountain overhead: packets received until a chunk decodes, over K, with random packet loss.
//!     cargo run --release --example fountain_overhead
use lensbolt_core::fountain::decoder::LtDecoder;
use lensbolt_core::fountain::encoder::LtEncoder;
use lensbolt_core::fountain::BLOCK_BYTES;
fn main() {
    for &k in &[64usize, 256, 512] {
        let data: Vec<u8> = (0..k * BLOCK_BYTES).map(|i| (i * 31 % 251) as u8).collect();
        let enc = LtEncoder::new(&data);
        for &loss in &[0.0f64, 0.3, 0.5] {
            let mut ratios = Vec::new();
            for trial in 0..10u64 {
                let mut dec = LtDecoder::new(k, data.len());
                let mut s = trial.wrapping_mul(0x9E3779B97F4A7C15) | 1;
                let (mut pid, mut got) = (0u32, 0usize);
                while !dec.is_done() && pid < (k as u32) * 8 {
                    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                    if (s % 1000) as f64 >= loss * 1000.0 { dec.add(pid, &enc.packet(pid)); got += 1; }
                    pid += 1;
                }
                ratios.push(got as f64 / k as f64);
            }
            ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!("K {k:5} loss {loss:.1}: received/K median {:.3} worst {:.3}", ratios[ratios.len() / 2], ratios[ratios.len() - 1]);
        }
    }
}
