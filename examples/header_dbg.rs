//! Why does a capture's header not read? Compares the bits read from CAPTURE.png (through the marker
//! homography) with the bits of the truth frame TRUTH.png (read at identity).
//!   cargo run --release --example header_dbg -- capture.png truth.png
use lensbolt_core::detect::{find_marker_candidates, refine_homography, two_means_threshold, Homography, MarkerSet};
use lensbolt_core::flat::{read_header_bits, Chrome, FlatHeader};
use lensbolt_core::image::RgbImage;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let cap = RgbImage::load_png(std::path::Path::new(&a[1])).unwrap();
    let truth = RgbImage::load_png(std::path::Path::new(&a[2])).unwrap();
    let chrome = Chrome::new(truth.width, truth.height).unwrap();
    let want: Vec<Vec<bool>> = (0..2).map(|c| read_header_bits(&truth, &chrome, &Homography::identity(), c)).collect();
    println!("truth headers: {:?}", want.iter().map(|b| FlatHeader::from_bits(b).map(|h| h.block_id)).collect::<Vec<_>>());
    let gray = cap.to_gray();
    let mut cands = find_marker_candidates(&gray);
    cands.sort_by(|a, b| b.3.cmp(&a.3));
    let pts: Vec<(f32, f32)> = cands.iter().take(4).map(|c| (c.0, c.1)).collect();
    let pick = |f: &dyn Fn(&(f32, f32)) -> f32| *pts.iter().max_by(|a, b| f(a).partial_cmp(&f(b)).unwrap()).unwrap();
    let m = MarkerSet { tl: pick(&|p| -(p.0 + p.1)), tr: pick(&|p| p.0 - p.1), br: pick(&|p| p.0 + p.1), bl: pick(&|p| p.1 - p.0) };
    let coarse = Homography::from_points(chrome.marker_centers(), m.as_array()).unwrap();
    let pairs: Vec<_> = chrome.marker_centers().iter().zip(m.as_array().iter()).map(|(a, b)| (*a, *b)).collect();
    let thr = two_means_threshold(&gray.data);
    for (name, h) in [("coarse", coarse), ("refined", refine_homography(&gray, &chrome, &coarse, &pairs, thr))] {
        for copy in 0..2u32 {
            let got = read_header_bits(&cap, &chrome, &h, copy);
            let wrong = got.iter().zip(&want[copy as usize]).filter(|(a, b)| a != b).count();
            let ones = got.iter().filter(|&&b| b).count();
            let first: String = got.iter().take(40).map(|&b| if b { '1' } else { '0' }).collect();
            let wfirst: String = want[copy as usize].iter().take(40).map(|&b| if b { '1' } else { '0' }).collect();
            println!("{name} copy {copy}: {wrong}/208 bits wrong, {ones} ones, decoded {:?}\n  got  {first}\n  want {wfirst}", FlatHeader::from_bits(&got).map(|h| h.block_id));
        }
        let (x, y) = chrome.header_cell_center(1, 0); let (cx, cy) = h.apply(x, y);
        println!("  copy 1 cell 0 at camera ({cx:.1}, {cy:.1}); module px {:.2}", lensbolt_core::flat::sampling(&h, &lensbolt_core::flat::FlatGrid::try_new(1024, 1024, lensbolt_core::flat::MANIFEST_FORMAT).unwrap()) * 8.0);
    }
}
