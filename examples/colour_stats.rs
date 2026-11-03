//! Per true colour: mean and spread of the camera RGB at cell centres (truth = the rebuilt displayed frame).
//!     cargo run --release --example colour_stats -- CAPTURE.png TRUTH.png
use lensbolt_core::flat::FlatDecoder;
use lensbolt_core::image::RgbImage;
use std::path::Path;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let cam = RgbImage::load_png(Path::new(&a[1])).unwrap();
    let truth = RgbImage::load_png(Path::new(&a[2])).unwrap();
    let fr = FlatDecoder::new().decode(&cam).unwrap();
    let g = &fr.grid;
    let pal = g.format().alphabet.palette();
    let mut acc = vec![([0f64; 3], [0f64; 3], 0u32, vec![0u32; pal.len()]); pal.len()];
    for i in 0..g.used_data_cells() as u32 {
        let (c, r) = g.data_position(i);
        let (x, y) = g.cell_center(c, r);
        let t = truth.get(x as u32, y as u32);
        let k = pal.iter().position(|p| *p == t).unwrap();
        let (cx, cy) = fr.homography.apply(x, y);
        let v = cam.sample(cx - 0.5, cy - 0.5);
        let e = &mut acc[k];
        for j in 0..3 { e.0[j] += v[j] as f64; e.1[j] += (v[j] as f64).powi(2); }
        e.2 += 1; e.3[fr.reads.at(c, r).0 as usize] += 1;
    }
    // tint of the white cells over a coarse grid: R - B per region (moire shows as smooth bands)
    let (gx, gy) = (12usize, 12usize);
    let mut tint = vec![(0f64, 0u32); gx * gy];
    for i in 0..g.used_data_cells() as u32 {
        let (c, r) = g.data_position(i);
        let (x, y) = g.cell_center(c, r);
        if truth.get(x as u32, y as u32) != [255, 255, 255] { continue; }
        let (cx, cy) = fr.homography.apply(x, y);
        let v = cam.sample(cx - 0.5, cy - 0.5);
        let k = ((y as usize * gy) / g.height_px() as usize).min(gy - 1) * gx + ((x as usize * gx) / g.width_px() as usize).min(gx - 1);
        tint[k].0 += v[0] as f64 - v[2] as f64; tint[k].1 += 1;
    }
    println!("white cells R-B by region:");
    for j in 0..gy { println!("  {}", (0..gx).map(|i| { let t = tint[j * gx + i]; if t.1 == 0 { "   .".into() } else { format!("{:4.0}", t.0 / t.1 as f64) } }).collect::<Vec<String>>().join("")); }
    for (k, e) in acc.iter().enumerate() {
        let m = [0, 1, 2].map(|j| e.0[j] / e.2 as f64);
        let sd = [0, 1, 2].map(|j| (e.1[j] / e.2 as f64 - m[j] * m[j]).max(0.0).sqrt());
        println!("{:?} -> cam mean [{:.0} {:.0} {:.0}] sd [{:.0} {:.0} {:.0}]  read as {:?}", pal[k], m[0], m[1], m[2], sd[0], sd[1], sd[2], e.3);
    }
}
