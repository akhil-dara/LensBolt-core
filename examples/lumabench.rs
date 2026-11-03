//! End-to-end luma decode time on stored captures (v7 headers accepted): cold (full search) vs tracked
//! (hint from the cold read). `LB_THREADS=1` for single-core times.
//!     cargo run --release --example lumabench -- CAPTURE.png [...]
use lensbolt_core::flat::FlatDecoder;
use lensbolt_core::image::RgbImage;
use std::time::Instant;
fn main() {
    lensbolt_core::flat::header::ACCEPT_V7_HEADERS.store(true, std::sync::atomic::Ordering::Relaxed);
    let dec = FlatDecoder::new();
    for f in std::env::args().skip(1) {
        let gray = RgbImage::load_png(std::path::Path::new(&f)).unwrap().to_gray();
        let t = Instant::now();
        let cold = dec.decode_luma(&gray, None);
        let cold_ms = t.elapsed().as_secs_f64() * 1e3;
        let Ok(cold) = cold else { println!("{f}: cold {:?}", cold.err()); continue };
        let hint = cold[0].hint();
        let t = Instant::now();
        let warm = dec.decode_luma(&gray, Some(&hint));
        let warm_ms = t.elapsed().as_secs_f64() * 1e3;
        println!("{f}: {}px cold {cold_ms:.0} ms, tracked {warm_ms:.0} ms ({})", cold[0].header.format.pitch, if warm.is_ok() { "ok" } else { "failed" });
    }
}
