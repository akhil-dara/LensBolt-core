//! A sender tile drawn at a fractional scale with nearest-neighbour snapping (each 2 px cell becomes 2 or 3
//! screen px wide, as a browser draws a canvas with `image-rendering: pixelated` at 1.26x) against the same tile
//! drawn uniformly, photographed at the same camera px per cell: does the snapping cost reads?
//!   cargo run --release --example snapped_scale -- [scale] [cell_px...]
use lensbolt_core::detect::Homography;
use lensbolt_core::flat::{compose, render, Alphabet, CellFormat, FlatDecoder, FlatGrid, FlatHeader};
use lensbolt_core::header::FrameType;
use lensbolt_core::image::RgbImage;
use std::sync::Arc;

/// `im` scaled by `f` with nearest-neighbour sampling (pixel x of the result shows source pixel floor(x / f)).
fn snapped(im: &RgbImage, f: f32) -> RgbImage {
    let (w, h) = ((im.width as f32 * f).round() as u32, (im.height as f32 * f).round() as u32);
    let mut out = RgbImage::new(w, h);
    for y in 0..h { for x in 0..w {
        let (sx, sy) = (((x as f32 + 0.5) / f) as u32, ((y as f32 + 0.5) / f) as u32);
        out.set(x, y, im.get(sx.min(im.width - 1), sy.min(im.height - 1)));
    } }
    out
}

/// `screen` (a w x h picture of the sender's screen) photographed so one tile cell (2 tile px, `tile_px` screen px
/// per tile px) is `cell` camera px, rolled 2 degrees, centred in 1920x1440, then blurred once.
fn photograph(screen: &RgbImage, tile_px: f32, cell: f32, seed: u64) -> lensbolt_core::image::GrayImage {
    let s = cell / (2.0 * tile_px);
    let (cw, ch) = (1920.0f32, 1440.0f32);
    let r = 2f32.to_radians() + seed as f32 * 0.003;
    let (w, h) = (screen.width as f32, screen.height as f32);
    let map = |x: f32, y: f32| { let (u, v) = (x - w / 2.0, y - h / 2.0); (cw / 2.0 + s * (u * r.cos() - v * r.sin()), ch / 2.0 + s * (u * r.sin() + v * r.cos())) };
    let src = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)];
    let f2c = Homography::from_points(src, src.map(|p| map(p.0, p.1))).unwrap();
    lensbolt_core::testutil::box_blur(&screen.warp(&f2c.inverse().unwrap(), cw as u32, ch as u32)).to_gray()
}

fn main() {
    lensbolt_core::par::serial_here();
    let a: Vec<String> = std::env::args().skip(1).collect();
    let f: f32 = a.first().and_then(|v| v.parse().ok()).unwrap_or(1.26);
    let cells: Vec<f32> = if a.len() > 1 { a[1..].iter().filter_map(|v| v.parse().ok()).collect() } else { vec![1.2, 1.3, 1.4, 1.5, 1.7] };
    let fmt = CellFormat { pitch: 2, alphabet: Alphabet::Bw };
    let g = Arc::new(FlatGrid::try_new(1024, 1024, fmt).unwrap());
    let dec = FlatDecoder::new();
    for &cell in &cells {
        let (mut uni, mut snp, mut n) = (0usize, 0usize, 0usize);
        for k in 0..6u64 {
            let hd = FlatHeader { format: fmt, cols: g.cols() as u16, rows: g.rows() as u16, frame_type: FrameType::Data, transfer_id: 1, segment: 1, block_id: k as u32, frame_seq: k as u8 };
            let payload: Vec<u8> = (0..g.payload_bytes()).map(|i| ((i as u64 * 2654435761 + k * 77) >> 7) as u8).collect();
            let im = render(&compose(g.clone(), hd, &payload));
            let read = |gray: &lensbolt_core::image::GrayImage| dec.decode_luma(gray, None).map(|r| r[0].payload().codewords_ok).unwrap_or(0);
            // uniform: the 1x tile photographed at `cell`; snapped: the f-scaled tile photographed at the same `cell`
            uni += read(&photograph(&im, 1.0, cell, k));
            snp += read(&photograph(&snapped(&im, f), f, cell, k));
            n += g.codewords();
        }
        println!("{cell:.2} camera px per cell: uniform {uni}/{n} ({:.1}%), snapped at {f}x {snp}/{n} ({:.1}%)", 100.0 * uni as f64 / n as f64, 100.0 * snp as f64 / n as f64);
    }
}
