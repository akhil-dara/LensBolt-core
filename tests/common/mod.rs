//! Real phone captures as the app hands them over: `f_<ms>_<W>x<H>_<yrow>_<uvrow>_<pixel>.{y,u,v}`.
#![allow(dead_code)]
use lensbolt_core::image::{GrayImage, RgbImage};
use std::path::{Path, PathBuf};

/// One capture's planes.
pub struct Capture { pub stem: String, pub w: u32, pub h: u32, pub y_row: usize, pub uv_row: usize, pub uv_pixel: usize, pub y: Vec<u8>, pub u: Vec<u8>, pub v: Vec<u8> }
impl Capture {
    pub fn luma(&self) -> GrayImage { GrayImage::from_plane(self.w, self.h, &self.y, self.y_row) }
    pub fn rgb(&self) -> RgbImage { RgbImage::from_yuv420(self.w, self.h, &self.y, self.y_row, &self.u, &self.v, self.uv_row, self.uv_pixel) }
}

/// `rel` (relative to the repository root) when it holds captures on this machine.
pub fn capture_dir(rel: &str) -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join(rel);
    let has = std::fs::read_dir(&dir).map(|d| d.filter_map(|e| e.ok()).any(|e| e.path().extension().is_some_and(|x| x == "y"))).unwrap_or(false);
    has.then_some(dir)
}

/// `LB_REQUIRE_CAPTURES=1`: a capture gate whose stored captures are missing fails instead of skipping (the captures
/// are not in the repository; a machine that keeps them sets it so a moved or deleted folder cannot pass unseen).
pub fn captures_required() -> bool { std::env::var("LB_REQUIRE_CAPTURES").is_ok_and(|v| v == "1") }

/// `capture_dir` for a gate: without the captures it says so and gives None, or fails under `LB_REQUIRE_CAPTURES=1`.
pub fn gate_dir(rel: &str) -> Option<PathBuf> {
    let dir = capture_dir(rel);
    if dir.is_none() {
        assert!(!captures_required(), "no captures in {rel} on this machine, and LB_REQUIRE_CAPTURES=1");
        eprintln!("skip: no captures in {rel} on this machine");
    }
    dir
}

/// The captures of `dir` in capture order. They are of a version-8 sender: loading them turns on the
/// diagnostics acceptance of v8 headers in this test process (tests/v8_captures.rs).
pub fn captures(dir: &Path) -> Vec<Capture> {
    lensbolt_core::flat::header::ACCEPT_V8_HEADERS.store(true, std::sync::atomic::Ordering::Relaxed);
    let mut stems: Vec<String> = std::fs::read_dir(dir).unwrap().filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter_map(|n| n.strip_suffix(".y").map(String::from)).collect();
    // capture order: the millisecond stamp after `f_` (not the string: the stamps are not zero-padded)
    stems.sort_by_key(|s| s.split('_').nth(1).and_then(|t| t.parse::<u64>().ok()).unwrap_or(u64::MAX));
    stems.into_iter().map(|stem| {
        let f: Vec<&str> = stem.split('_').collect();
        let d: Vec<u32> = f[2].split('x').map(|v| v.parse().unwrap()).collect();
        let rd = |e: &str| std::fs::read(dir.join(format!("{stem}.{e}"))).unwrap();
        Capture { w: d[0], h: d[1], y_row: f[3].parse().unwrap(), uv_row: f[4].parse().unwrap(), uv_pixel: f[5].parse().unwrap(),
            y: rd("y"), u: rd("u"), v: rd("v"), stem }
    }).collect()
}
