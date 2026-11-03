//! Fast luma reader on a simulated capture vs the displayed screen it came from (phone_sim --save):
//! cell error rate of the capture's read against an exact read of the screen, and codewords decoded.
//!     cargo run --release --example lumacmp -- CAP.png SCREEN.png
use lensbolt_core::detect::Homography;
use lensbolt_core::flat::FlatDecoder;
use lensbolt_core::image::RgbImage;
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let cap = RgbImage::load_png(std::path::Path::new(&a[0])).unwrap();
    let scr = RgbImage::load_png(std::path::Path::new(&a[1])).unwrap();
    let dec = FlatDecoder::new();
    let frs = match dec.decode_luma(&cap.to_gray(), None) { Ok(f) => f, Err(e) => { println!("decode_luma: {e:?}"); return; } };
    let fr = &frs[0];
    let truth = lensbolt_core::flat::fast::read_cells_luma(&scr.to_gray(), &fr.grid, &Homography::identity(), fr.header.frame_seq & 3, None).unwrap();
    let g = &fr.grid;
    let n = g.used_data_cells();
    let err = (0..n).filter(|&i| { let (c, r) = g.data_position(i as u32); fr.reads.at(c, r).0 != truth.at(c, r).0 }).count();
    let p = fr.payload();
    let mut tags = std::collections::BTreeMap::new(); for &t in &fr.reads.block_tag { *tags.entry(t).or_insert(0) += 1; }
    println!("{}: {}px {} cell errors {:.2}%  codewords {}/{}  tags {tags:?}  sampling {:.2}", a[0], g.format().pitch, g.format().alphabet.name(), err as f64 * 100.0 / n as f64, p.codewords_ok, p.codewords_ok + p.codewords_failed, lensbolt_core::flat::sampling(&fr.homography, g));
}
