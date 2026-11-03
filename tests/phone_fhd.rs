//! Real captures: a OnePlus 12 at Full HD (1440x1080 YUV_420_888, 0.85 camera px per screen px) filming a
//! 1024x1024 4 px c8 tile on a laptop screen, exposure already lowered (EV -3). Raw planes as the app hands
//! them to the native receiver (`test/phone_fhd/f_<ms>_<W>x<H>_<yrow>_<uvrow>_<pixel>.{y,u,v}`).
//! Before the measured-centre colour decision: 2, 3 and 44 of 83 codewords (cell errors 6.5%, 7.2%, 2.2%).
use lensbolt_core::flat::FlatDecoder;
use lensbolt_core::image::RgbImage;
use std::path::PathBuf;

fn frames() -> Vec<(String, RgbImage)> {
    // version-8 captures (tests/v8_captures.rs)
    lensbolt_core::flat::header::ACCEPT_V8_HEADERS.store(true, std::sync::atomic::Ordering::Relaxed);
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../test/phone_fhd");
    let mut stems: Vec<String> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter_map(|n| n.strip_suffix(".y").map(String::from)).collect();
    stems.sort();
    stems.into_iter().map(|stem| {
        let f: Vec<&str> = stem.split('_').collect();
        let d: Vec<u32> = f[2].split('x').map(|v| v.parse().unwrap()).collect();
        let (yr, ur, px): (usize, usize, usize) = (f[3].parse().unwrap(), f[4].parse().unwrap(), f[5].parse().unwrap());
        let rd = |e: &str| std::fs::read(dir.join(format!("{stem}.{e}"))).unwrap();
        let img = RgbImage::from_yuv420(d[0], d[1], &rd("y"), yr, &rd("u"), &rd("v"), ur, px);
        (stem, img)
    }).collect()
}

/// Target not met yet (2026-10-29). Three reader changes made it worse: pilot-measured colour centres
/// (1/0/35), bilinear chroma (0/0/1), frame-wide 8-means (1/0/27); per-block colour models are best, so the
/// colour response varies across the tile (moire of the LCD subpixels at 0.85 camera px per screen px is the
/// leading suspect). Run with --ignored while working on the Full HD colour path.
#[test]
#[ignore]
fn full_hd_phone_captures_decode_most_codewords() {
    let dec = FlatDecoder::new();
    let fr = frames();
    assert_eq!(fr.len(), 3);
    let mut report = Vec::new();
    for (stem, img) in &fr {
        let read = dec.decode(img).unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        let p = read.payload();
        report.push((stem.clone(), p.codewords_ok, p.codewords_ok + p.codewords_failed + p.codewords_missing));
    }
    // every capture must carry packets on its own: 70 of 83 codewords leave most 1 KB packets whole
    for (stem, ok, n) in &report { assert!(*ok * 100 >= *n * 84, "{stem}: {ok}/{n} codewords; all: {report:?}"); }
}
