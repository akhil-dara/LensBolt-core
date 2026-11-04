//! Real captures: a phone at Full HD (1440x1080, Y plane as the app hands it to the native receiver) reading
//! a phone sender, 3 px b/w 1024x1024 at 20 fps, the tile 720 px wide (2.1 camera px per cell) on
//! the phone's dark screen (`test/phone_sender_tile/f_<ms>_<W>x<H>_<yrow>_..y`, 2026-11-03).
//!
//! The marker's 1-module dark ring blurs to about 100 against whites of 215, and the binarisation window
//! (1/16 of the frame, 90 px) is mostly dark screen at the tile's corners (its mean about 103): the ring read
//! white and no marker was found, 60 of 60 captures; the transfer crawled at 0.04 Mbit/s.
use lensbolt_core::flat::FlatDecoder;
use lensbolt_core::image::GrayImage;

mod common;

/// The stored captures (not in the public repository: without them the test says so and passes, or fails under
/// `LB_REQUIRE_CAPTURES=1`).
fn frames() -> Vec<(String, GrayImage)> {
    let Some(dir) = common::gate_dir("test/phone_sender_tile") else { return Vec::new() };
    let mut stems: Vec<String> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter_map(|n| n.strip_suffix(".y").map(String::from)).collect();
    stems.sort();
    stems.into_iter().map(|stem| {
        let f: Vec<&str> = stem.split('_').collect();
        let d: Vec<usize> = f[2].split('x').map(|v| v.parse().unwrap()).collect();
        let rs: usize = f[3].parse().unwrap();
        let raw = std::fs::read(dir.join(format!("{stem}.y"))).unwrap();
        let mut data = Vec::with_capacity(d[0] * d[1]);
        for y in 0..d[1] { data.extend_from_slice(&raw[y * rs..y * rs + d[0]]); }
        (stem, GrayImage { width: d[0] as u32, height: d[1] as u32, data })
    }).collect()
}

/// Every capture finds the tile (a small binarisation window when the frame's finds no marker), and the
/// headers read soft (weak bytes as erasures, the two copies combined): 46 of the 60 captures of that run read,
/// 2170 of their 2174 codewords. One capture here still fails its header (both copies damaged).
#[test]
fn phone_sender_tile_on_a_dark_screen_is_found() {
    use lensbolt_core::flat::DecodeError;
    let dec = FlatDecoder::new();
    let mut whole = 0;
    for (stem, g) in frames() {
        match dec.decode_luma(&g, None) {
            Ok(reads) => {
                let (ok, n) = (reads[0].payload().codewords_ok, reads[0].grid.codewords());
                eprintln!("{stem}: {ok}/{n} codewords");
                if ok * 100 >= n * 95 { whole += 1; }
            }
            Err(e) => { eprintln!("{stem}: {e:?}"); assert!(!matches!(e, DecodeError::NoMarkers), "{stem}: no markers found"); }
        }
    }
    let n = frames().len();
    if n > 0 { assert!(whole >= 2, "{whole} of {n} captures read whole"); }
}

