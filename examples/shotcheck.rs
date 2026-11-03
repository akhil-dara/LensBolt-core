//! A shared-screen frame (PNG): identified geometry, whether it is pixel-exact, and the decode result.
use lensbolt_core::flat::FlatDecoder;
use lensbolt_core::image::RgbImage;
fn main() {
    let f = std::env::args().nth(1).unwrap();
    let img = RgbImage::load_png(std::path::Path::new(&f)).unwrap();
    let dec = FlatDecoder::new();
    let (chrome, h, hdr) = dec.identify(&img).unwrap();
    let (w, hh) = (chrome.width_px() as f32, chrome.height_px() as f32);
    println!("image {}x{} tile {}x{} header {:?}", img.width, img.height, w, hh, hdr[0].map(|x| x.format));
    for (x, y) in [(0.0, 0.0), (w, 0.0), (0.0, hh), (w, hh)] { let (cx, cy) = h.apply(x, y); println!("  ({x},{y}) -> ({cx:.2},{cy:.2})  delta ({:.2},{:.2})", cx - x, cy - y); }
    let gray = img.to_gray();
    let t = std::time::Instant::now();
    match dec.decode_luma(&gray, None) { Ok(frs) => { let p = frs[0].payload(); println!("decode_luma {:.0} ms: codewords {}/{}", t.elapsed().as_secs_f64() * 1e3, p.codewords_ok, p.codewords_ok + p.codewords_failed); } Err(e) => println!("decode_luma {e:?}") }
}
