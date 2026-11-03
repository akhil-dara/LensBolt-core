//! Decode raw phone frames dumped by the Android app (files/dump/f_<ms>_<W>x<H>_<yrow>_<uvrow>_<px>.{y,u,v}):
//! the same conversion and decode the phone runs, with per-frame codeword results and a PNG of each frame.
//! Version-8 headers are accepted too (diagnostics only): the stored dumps (test/phone_*) are of a v8 sender.
//!     cargo run --release --example yuv_diag -- DIR
use lensbolt_core::flat::FlatDecoder;
use lensbolt_core::image::RgbImage;
use std::path::Path;

fn main() {
    lensbolt_core::flat::header::ACCEPT_V8_HEADERS.store(true, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::args().nth(1).expect("dump dir");
    let mut stems: Vec<String> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter_map(|n| n.strip_suffix(".y").map(String::from)).collect();
    stems.sort();
    let dec = FlatDecoder::new();
    for stem in stems {
        let f: Vec<&str> = stem.split('_').collect();
        let (w, h) = { let d: Vec<u32> = f[2].split('x').map(|v| v.parse().unwrap()).collect(); (d[0], d[1]) };
        let (yr, ur, px): (usize, usize, usize) = (f[3].parse().unwrap(), f[4].parse().unwrap(), f[5].parse().unwrap());
        let rd = |e: &str| std::fs::read(Path::new(&dir).join(format!("{stem}.{e}"))).unwrap();
        let img = RgbImage::from_yuv420(w, h, &rd("y"), yr, &rd("u"), &rd("v"), ur, px);
        img.save_png(&Path::new(&dir).join(format!("{stem}.png"))).unwrap();
        let t = std::time::Instant::now();
        match dec.decode_all(&img) {
            Ok(frs) => for fr in frs {
                let p = fr.payload();
                let (tl, br) = (fr.homography.apply(0.0, 0.0), fr.homography.apply(fr.grid.width_px() as f32, fr.grid.height_px() as f32));
                println!("{stem}: {:?} {}px {} seq {} transfer {} block {} codewords ok {} failed {} missing {} | tile {:.0}x{:.0} cam px = {:.2} cam px/screen px | {:.0} ms",
                    fr.header.frame_type, fr.header.format.pitch, fr.header.format.alphabet.name(), fr.header.frame_seq, fr.header.transfer_id, fr.header.block_id, p.codewords_ok, p.codewords_failed, p.codewords_missing,
                    (br.0 - tl.0).abs(), (br.1 - tl.1).abs(), (br.0 - tl.0).abs() / fr.grid.width_px() as f32, t.elapsed().as_secs_f64() * 1000.0);
            },
            Err(e) => println!("{stem}: {e:?}"),
        }
    }
}
