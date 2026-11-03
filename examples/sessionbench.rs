//! The browser receiver's per-frame cost on one captured frame, pushed repeatedly through RxSession.
//! cargo run --release --example sessionbench -- frame.png [repeats]
use lensbolt_core::flat::session::RxSession;
use lensbolt_core::image::RgbImage;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let img = RgbImage::load_png(std::path::Path::new(&a[1])).unwrap();
    let n: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(8);
    let rgba: Vec<u8> = img.data.chunks_exact(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect();
    let mut s = RxSession::new();
    for i in 0..n {
        let t = std::time::Instant::now();
        let st = s.push_rgba_at(&rgba, img.width, img.height, i as f64 * 33.0);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let dec = st.contains("\"failed\":0");
        println!("frame {i}: {ms:.1} ms {}", if dec { "ok" } else { &st[..120.min(st.len())] });
    }
}
