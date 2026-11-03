//! Transport efficiency: frames the sender shows until the receiver completes, when the receiver sees only
//! some frames (a camera at 25 of 30 fps, plus random losses). Prints new-packet yield per second over time.
//! A fifth argument loses the same share of every frame's packets (the last ones: a region the camera
//! keeps missing, as with straddled captures).
//!     cargo run --release --example transport_sim -- <pitch> <MB> <keep_of_6> <loss%> [region_loss%]
use lensbolt_core::file::receiver::{FeedOutcome, Receiver};
use lensbolt_core::flat::sender::{FlatSender, FlatSenderConfig, MANIFEST_EVERY_FRAMES};
use lensbolt_core::flat::{Alphabet, CellFormat};
fn main() {
    let a: Vec<f64> = std::env::args().skip(1).map(|v| v.parse().unwrap()).collect();
    let (pitch, mb, keep6, loss) = (a.first().copied().unwrap_or(2.0) as u8, a.get(1).copied().unwrap_or(8.0), a.get(2).copied().unwrap_or(5.0) as u64, a.get(3).copied().unwrap_or(0.0));
    let mut s = 0x1234_5678u64;
    let data: Vec<u8> = (0..(mb * 1e6) as usize).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s as u8 }).collect();
    let cfg = FlatSenderConfig { width: 1024, height: 1024, format: CellFormat { pitch, alphabet: Alphabet::Bw }, manifest_every_frames: MANIFEST_EVERY_FRAMES, transfer_id: 9, ..FlatSenderConfig::default() };
    let region = a.get(4).copied().unwrap_or(0.0) / 100.0;
    let mut tx = FlatSender::from_bytes("sim.bin", data, cfg).unwrap();
    let per_pass = tx.frames_per_pass();
    let mut rx = Receiver::in_memory();
    let mut r = 99u64;
    let mut last_new = 0u64;
    for n in 0..200_000u64 {
        let (h, mut p) = tx.next_packets().unwrap();
        p.truncate(p.len() - (p.len() as f64 * region) as usize);
        r ^= r << 13; r ^= r >> 7; r ^= r << 17;
        let seen = n % 6 < keep6 && (r % 1000) as f64 >= loss * 10.0;
        if seen {
            if let Ok(FeedOutcome::Complete(_)) = rx.feed_packets(&h, &p) { println!("complete after {n} frames = {:.2} passes ({per_pass} frames per pass), {:.1} s at 30 fps", n as f64 / per_pass as f64, n as f64 / 30.0); return; }
        }
        if n % 150 == 149 {
            let pr = rx.progress();
            println!("  {:5.1} s  new packets/s {:6.0}  done {:.2} MB", (n + 1) as f64 / 30.0, (pr.packets_new - last_new) as f64 / 5.0, pr.bytes_done as f64 / 1e6);
            last_new = pr.packets_new;
        }
    }
}
