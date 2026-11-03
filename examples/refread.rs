//! Reader regression on stored real captures: locate the tile (v7 headers accepted), read the cells, and
//! count errors against the truth of the matching dump.
//!   cargo run --release --example refread -- CAPTURE.png DUMP.bin [CAPTURE.png DUMP.bin ...]
use lensbolt_core::flat::FlatDecoder;
use lensbolt_core::image::RgbImage;

fn main() {
    lensbolt_core::flat::header::ACCEPT_V7_HEADERS.store(true, std::sync::atomic::Ordering::Relaxed);
    let args: Vec<String> = std::env::args().skip(1).collect();
    for pair in args.chunks(2) {
        let img = RgbImage::load_png(std::path::Path::new(&pair[0])).expect("png");
        let b = std::fs::read(&pair[1]).expect("dump");
        let u32_at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let (cols, rows) = (u32_at(0), u32_at(4));
        let k = (cols * rows) as usize;
        let truth = &b[12 + 3 * k..12 + 4 * k];
        let t = std::time::Instant::now();
        let r = FlatDecoder::new().decode(&img).expect("decode");
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let g = &r.grid;
        let t2 = std::time::Instant::now();
        let fp = lensbolt_core::flat::extract_payload(&r.reads, g);
        eprintln!("  [extract] {:.1} ms (v7 cells under the v8 layout: {} codewords fail, as they must)", t2.elapsed().as_secs_f64() * 1000.0, fp.codewords_failed);
        let t3 = std::time::Instant::now();
        let gray = img.to_gray();
        let _ = lensbolt_core::detect::sharpness(&gray);
        eprintln!("  [gray + sharpness] {:.1} ms", t3.elapsed().as_secs_f64() * 1000.0);
        let n = g.used_data_cells();
        let err = (0..n).filter(|&i| { let (c, rr) = g.data_position(i as u32); let j = (rr * cols + c) as usize; r.reads.values[j] != truth[j] }).count();
        let mut tags = std::collections::BTreeMap::new(); for &t in &r.reads.block_tag { *tags.entry(t).or_insert(0) += 1; }
        println!("{}: cell errors {err}/{n} = {:.3}%  tags {tags:?}  decode {ms:.0} ms  sampling {:.2} camera px per screen px", pair[0], err as f64 * 100.0 / n as f64, lensbolt_core::flat::sampling(&r.homography, g));
    }
}
