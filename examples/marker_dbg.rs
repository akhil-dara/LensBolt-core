//! Marker search stages on one phone Y plane: <file.y> <w> <h> <rowstride>
use lensbolt_core::detect::markers::{corner_set, find_marker_candidates, find_marker_candidates_r, find_markers, local_binarize};
use lensbolt_core::image::GrayImage;
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (w, h, rs): (usize, usize, usize) = (a[1].parse().unwrap(), a[2].parse().unwrap(), a[3].parse().unwrap());
    let raw = std::fs::read(&a[0]).unwrap();
    let mut data = Vec::with_capacity(w * h);
    for y in 0..h { data.extend_from_slice(&raw[y * rs..y * rs + w]); }
    let g = GrayImage { width: w as u32, height: h as u32, data };
    let bin = local_binarize(&g);
    println!("white share {:.3}", bin.iter().map(|&b| b as f64).sum::<f64>() / bin.len() as f64);
    let mut c = find_marker_candidates(&g);
    c.sort_by(|a, b| b.3.cmp(&a.3));
    for x in c.iter().take(15) { println!("cand x {:.0} y {:.0} unit {:.2} hits {}", x.0, x.1, x.2, x.3); }
    println!("markers {:?}", find_markers(&g).map(|m| m.as_array()));
    for r in [8usize, 12, 16, 24, 32, 48, 64] {
        let t = std::time::Instant::now();
        let mut c = find_marker_candidates_r(&g, r);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        c.sort_by(|a, b| b.3.cmp(&a.3));
        let pts: Vec<(f32, f32)> = c.iter().filter(|c| c.3 >= 2).map(|c| (c.0, c.1)).collect();
        let top: Vec<String> = c.iter().take(5).map(|x| format!("({:.0},{:.0} u{:.1} h{})", x.0, x.1, x.2, x.3)).collect();
        println!("r {r}: {:.1} ms, {} cands, set {:?}; top {}", ms, c.len(), corner_set(&pts).map(|m| m.as_array().map(|p| (p.0 as i32, p.1 as i32))), top.join(" "));
    }
    let rows: Vec<u8> = bin.iter().map(|&b| b * 255).collect();
    std::fs::write(a.get(4).cloned().unwrap_or("bin.raw".into()), rows).unwrap();
}
