//! What a full-sensor frame costs: a 2 px b/w 1024 tile photographed into a w x h luma frame at `cell` camera px
//! per cell, decoded cold and tracked (luma path, one thread like a live worker), with stage timings under
//! LB_PROFILE.   cargo run --release --example bigframe -- [w] [h] [cell_px]
use lensbolt_core::flat::{compose, render, Alphabet, CellFormat, FlatDecoder, FlatGrid, FlatHeader};
use lensbolt_core::header::FrameType;
use lensbolt_core::detect::Homography;
use std::sync::Arc;
fn main() {
    lensbolt_core::par::serial_here();
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (w, h): (u32, u32) = (a.first().and_then(|v| v.parse().ok()).unwrap_or(4032), a.get(1).and_then(|v| v.parse().ok()).unwrap_or(3024));
    let cell: f32 = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(2.2);
    let fmt = CellFormat { pitch: 2, alphabet: Alphabet::Bw };
    let g = Arc::new(FlatGrid::try_new(1024, 1024, fmt).unwrap());
    let hd = FlatHeader { format: fmt, cols: g.cols() as u16, rows: g.rows() as u16, frame_type: FrameType::Data, transfer_id: 1, segment: 1, block_id: 0, frame_seq: 0 };
    let payload: Vec<u8> = (0..g.payload_bytes()).map(|i| (i * 7 + 3) as u8).collect();
    let im = render(&compose(g.clone(), hd, &payload));
    // the tile at `cell` px per 2 px cell, centred in a w x h picture, rolled 2 degrees; a touch of lens blur
    let tile = 1024.0 * cell / 2.0;
    let (cx, cy, s, r) = (w as f32 / 2.0, h as f32 / 2.0, tile / 1024.0, 2f32.to_radians());
    let map = |x: f32, y: f32| { let (u, v) = (x - 512.0, y - 512.0); (cx + s * (u * r.cos() - v * r.sin()), cy + s * (u * r.sin() + v * r.cos())) };
    let src = [(0.0, 0.0), (1024.0, 0.0), (1024.0, 1024.0), (0.0, 1024.0)];
    let f2c = Homography::from_points(src, src.map(|p| map(p.0, p.1))).unwrap();
    let cam = lensbolt_core::testutil::box_blur(&im.warp(&f2c.inverse().unwrap(), w, h));
    let gray = cam.to_gray();
    let dec = FlatDecoder::new();
    let t = std::time::Instant::now();
    let cold = dec.decode_luma(&gray, None);
    let tc = t.elapsed().as_secs_f64() * 1e3;
    let hint = cold.as_ref().ok().map(|r| r[0].hint());
    let t = std::time::Instant::now();
    let tracked = dec.decode_luma(&gray, hint.as_ref());
    let tt = t.elapsed().as_secs_f64() * 1e3;
    // the app's copies of a camera plane (row stride padded): the whole plane to a Vec, then the rows unpadded
    let stride = (w as usize + 63) / 64 * 64;
    let mut plane = vec![0u8; stride * h as usize];
    for y in 0..h as usize { plane[y * stride..y * stride + w as usize].copy_from_slice(&gray.data[y * w as usize..(y + 1) * w as usize]); }
    let t = std::time::Instant::now();
    let v = plane.to_vec();
    let g2 = lensbolt_core::image::GrayImage::from_plane(w, h, &v, stride);
    let tcopy = t.elapsed().as_secs_f64() * 1e3;
    std::hint::black_box(&g2);
    // only the tile's region (its corners through the hint, a margin of an eighth of its size) copied, the hint
    // moved into the crop
    if let Some(hint) = &hint {
        let c = hint.chrome;
        let pts = [(0.0, 0.0), (c.width_px() as f32, 0.0), (c.width_px() as f32, c.height_px() as f32), (0.0, c.height_px() as f32)].map(|p| hint.homography.apply(p.0, p.1));
        let (x0, x1) = pts.iter().fold((f32::MAX, f32::MIN), |a, p| (a.0.min(p.0), a.1.max(p.0)));
        let (y0, y1) = pts.iter().fold((f32::MAX, f32::MIN), |a, p| (a.0.min(p.1), a.1.max(p.1)));
        let m = 0.125 * (x1 - x0).max(y1 - y0);
        let (cx0, cy0) = ((x0 - m).max(0.0) as usize, (y0 - m).max(0.0) as usize);
        let (cx1, cy1) = (((x1 + m) as usize).min(w as usize), ((y1 + m) as usize).min(h as usize));
        let t = std::time::Instant::now();
        let mut crop = Vec::with_capacity((cx1 - cx0) * (cy1 - cy0));
        for y in cy0..cy1 { crop.extend_from_slice(&plane[y * stride + cx0..y * stride + cx1]); }
        let cg = lensbolt_core::image::GrayImage { width: (cx1 - cx0) as u32, height: (cy1 - cy0) as u32, data: crop };
        let tcrop = t.elapsed().as_secs_f64() * 1e3;
        let mut hh = hint.clone();
        let (ox, oy, mm) = (cx0 as f32, cy0 as f32, hint.homography.m);
        hh.homography.m = [mm[0] - ox * mm[6], mm[1] - ox * mm[7], mm[2] - ox * mm[8], mm[3] - oy * mm[6], mm[4] - oy * mm[7], mm[5] - oy * mm[8], mm[6], mm[7], mm[8]];
        let t = std::time::Instant::now();
        let r = dec.decode_luma(&cg, Some(&hh));
        let td = t.elapsed().as_secs_f64() * 1e3;
        println!("copies: whole plane {tcopy:.1} ms; crop {}x{} {tcrop:.1} ms, tracked on the crop {td:.1} ms {:?}", cg.width, cg.height, r.as_ref().map(|v| v[0].payload().codewords_ok).map_err(|e| format!("{e:?}")));
    }
    let cw = |r: &Result<Vec<lensbolt_core::flat::FlatRead>, lensbolt_core::flat::DecodeError>| r.as_ref().map(|v| format!("{}/{}", v[0].payload().codewords_ok, v[0].grid.codewords())).unwrap_or_else(|e| format!("{e:?}"));
    println!("{w}x{h}, tile {tile} px ({cell} px/cell): cold {tc:.1} ms {}, tracked {tt:.1} ms {}", cw(&cold), cw(&tracked));
}
