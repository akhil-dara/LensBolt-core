//! Where a shared-screen frame's decode time goes: grid construction, the cell read, the payload.
use lensbolt_core::flat::{FlatDecoder, FlatGrid};
use lensbolt_core::image::RgbImage;
fn main() {
    let f = std::env::args().nth(1).unwrap();
    let img = RgbImage::load_png(std::path::Path::new(&f)).unwrap();
    let gray = img.to_gray();
    let dec = FlatDecoder::new();
    let (chrome, h, hdr) = dec.identify(&img).unwrap();
    let hdr = hdr[0].unwrap();
    let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1e3;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        let grid = FlatGrid::try_new(chrome.width_px(), chrome.height_px(), hdr.format).unwrap();
        let a = ms(t);
        let t = std::time::Instant::now();
        let r = lensbolt_core::flat::fast::read_cells_luma(&gray, &grid, &h, hdr.frame_seq & 3, None).unwrap();
        let b = ms(t);
        println!("grid {a:.1} ms  read {b:.1} ms  cells {}", r.values.len());
    }
}
