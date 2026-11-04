//! box_blur against box_blur_region over a whole 1440x1080 frame and over a 60% region.
use lensbolt_core::flat::read::{box_blur, box_blur_region};
use lensbolt_core::image::RgbImage;
fn main() {
    let mut img = RgbImage::new(1440, 1080);
    let mut st = 0x9E37_79B9u32;
    for p in img.data.iter_mut() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; *p = (st >> 24) as u8; }
    for k in [3u32, 4, 6] {
        let t = std::time::Instant::now(); for _ in 0..10 { std::hint::black_box(box_blur(&img, k)); } let a = t.elapsed().as_secs_f64() * 100.0;
        let t = std::time::Instant::now(); for _ in 0..10 { std::hint::black_box(box_blur_region(&img, k, [0, 0, 1440, 1080])); } let b = t.elapsed().as_secs_f64() * 100.0;
        let t = std::time::Instant::now(); for _ in 0..10 { std::hint::black_box(box_blur_region(&img, k, [200, 100, 1240, 980])); } let c = t.elapsed().as_secs_f64() * 100.0;
        println!("k {k}: whole {a:.1} ms, region=whole {b:.1} ms, 60% region {c:.1} ms");
    }
}
