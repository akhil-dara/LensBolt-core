//! Packets per frame and Mbit/s at 30 fps for each format on a tile (a manifest every MANIFEST_EVERY_FRAMES frames).
//!     cargo run --release --example capacity -- 1024 1024
use lensbolt_core::flat::{Alphabet, CellFormat, FlatGrid, MANIFEST_EVERY_FRAMES, PITCHES};
fn main() {
    let a: Vec<u32> = std::env::args().skip(1).map(|v| v.parse().unwrap()).collect();
    let (w, h) = (a.first().copied().unwrap_or(1024), a.get(1).copied().unwrap_or(1024));
    for &p in &PITCHES { for al in Alphabet::ALL {
        let Some(g) = FlatGrid::try_new(w, h, CellFormat { pitch: p, alphabet: al }) else { continue };
        let m = MANIFEST_EVERY_FRAMES as f64;
        let mbit = g.packets() as f64 * 1024.0 * 8.0 * 30.0 * (m - 1.0) / m / 1e6;
        println!("{p}px {:4}: {:3} packets/frame  {:6.2} Mbit/s", al.name(), g.packets(), mbit);
    } }
}
