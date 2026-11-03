//! Sender frame production time across passes (first systematic pass, then repair rounds).
//!     cargo run --release --example sender_speed -- <pitch> <MB>
use lensbolt_core::flat::sender::{FlatSender, FlatSenderConfig, MANIFEST_EVERY_FRAMES};
use lensbolt_core::flat::{Alphabet, CellFormat};
fn main() {
    let a: Vec<f64> = std::env::args().skip(1).map(|v| v.parse().unwrap()).collect();
    let (pitch, mb) = (a.first().copied().unwrap_or(2.0) as u8, a.get(1).copied().unwrap_or(8.0));
    let mut s = 0x1234_5678u64;
    let data: Vec<u8> = (0..(mb * 1e6) as usize).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s as u8 }).collect();
    let cfg = FlatSenderConfig { width: 1024, height: 1024, format: CellFormat { pitch, alphabet: Alphabet::Bw }, manifest_every_frames: MANIFEST_EVERY_FRAMES, transfer_id: 9, ..FlatSenderConfig::default() };
    let mut tx = FlatSender::from_bytes("sim.bin", data, cfg).unwrap();
    let per = tx.frames_per_pass();
    println!("frames per pass {per}");
    for block in 0..8 {
        let t = std::time::Instant::now();
        for _ in 0..per / 2 { tx.next_frame().unwrap(); }
        println!("frames {:5}..{:5}: {:.1} ms per frame", block * per / 2, (block + 1) * per / 2, t.elapsed().as_secs_f64() * 1e3 / (per / 2) as f64);
    }
}
