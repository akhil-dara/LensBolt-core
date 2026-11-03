//! Helpers shared by tests and the CLI: a synthetic camera and large pseudo-random files.
use crate::detect::Homography;
use crate::image::RgbImage;
use std::io::Write;
use std::path::Path;

/// "Photograph" a rendered frame: perspective (corner jitter), downscale to cam_w x cam_h, +-noise. Deterministic for a seed.
pub fn photograph(frame: &RgbImage, cam_w: u32, cam_h: u32, jitter: f32, noise: u8, seed: u64) -> RgbImage {
    let (w, h) = (frame.width as f32, frame.height as f32);
    let (mx, my) = (cam_w as f32 * 0.08, cam_h as f32 * 0.08);
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut jit = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * jitter };
    let dst = [(mx + jit(), my + jit()), (cam_w as f32 - mx + jit(), my + jit()), (cam_w as f32 - mx + jit(), cam_h as f32 - my + jit()), (mx + jit(), cam_h as f32 - my + jit())];
    let src = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)];
    let f2c = Homography::from_points(src, dst).expect("non-degenerate");
    let mut cam = frame.warp(&f2c.inverse().expect("invertible"), cam_w, cam_h);
    if noise > 0 {
        for p in cam.data.iter_mut() {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            let n = (s % (2 * noise as u64 + 1)) as i32 - noise as i32;
            *p = (*p as i32 + n).clamp(0, 255) as u8;
        }
    }
    cam
}

/// Same as `photograph` but also returns the frame->camera homography (for detection accuracy tests).
pub fn photograph_with_homography(frame: &RgbImage, cam_w: u32, cam_h: u32, jitter: f32, noise: u8, seed: u64) -> (RgbImage, Homography) {
    let (w, h) = (frame.width as f32, frame.height as f32);
    let (mx, my) = (cam_w as f32 * 0.08, cam_h as f32 * 0.08);
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut jit = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * jitter };
    let dst = [(mx + jit(), my + jit()), (cam_w as f32 - mx + jit(), my + jit()), (cam_w as f32 - mx + jit(), cam_h as f32 - my + jit()), (mx + jit(), cam_h as f32 - my + jit())];
    let src = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)];
    let f2c = Homography::from_points(src, dst).expect("non-degenerate");
    (photograph(frame, cam_w, cam_h, jitter, noise, seed), f2c)
}

/// Render a flat frame of `fmt` (1024x1024, pseudo-random payload) and photograph it: perspective,
/// downscale to cam_w x cam_h, slight noise. Returns the grid, the camera image and frame->camera map.
pub fn synth_flat(fmt: crate::flat::CellFormat, cam_w: u32, cam_h: u32, jitter: f32, noise: u8) -> (std::sync::Arc<crate::flat::FlatGrid>, RgbImage, Homography) {
    use crate::flat::{compose, render, FlatGrid, FlatHeader};
    let g = std::sync::Arc::new(FlatGrid::try_new(1024, 1024, fmt).expect("1024 tile"));
    let h = FlatHeader { format: fmt, cols: g.cols() as u16, rows: g.rows() as u16, frame_type: crate::header::FrameType::Data, transfer_id: 1, segment: 1, block_id: 0, frame_seq: 0 };
    let payload: Vec<u8> = (0..g.payload_bytes()).map(|i| (i * 13 + 5) as u8).collect();
    let im = render(&compose(g.clone(), h, &payload));
    let (cam, f2c) = photograph_with_homography(&im, cam_w, cam_h, jitter, noise, 1);
    (g, cam, f2c)
}

pub fn pseudo_random_file(path: &Path, bytes: u64, seed: u64) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut s = seed | 1; let mut buf = vec![0u8; 1 << 20]; let mut left = bytes;
    while left > 0 {
        let n = (left as usize).min(buf.len());
        for chunk in buf[..n].chunks_mut(8) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; let b = s.to_le_bytes(); chunk.copy_from_slice(&b[..chunk.len()]); }
        f.write_all(&buf[..n])?; left -= n as u64;
    }
    f.flush()
}

/// Camera colour rendering as seen on a real phone photo of a laptop screen (measured 2026-10-10):
/// warm white balance, strokes desaturated toward luma, mild gamma. `desat` 0..1 = fraction blended to luma.
pub fn degrade_colour(img: &RgbImage, gains: [f32; 3], desat: f32, gamma: f32) -> RgbImage {
    let mut out = img.clone();
    for p in out.data.chunks_mut(3) {
        let mut v = [p[0] as f32 * gains[0], p[1] as f32 * gains[1], p[2] as f32 * gains[2]];
        let l = 0.299 * v[0] + 0.587 * v[1] + 0.114 * v[2];
        for c in 0..3 { v[c] = v[c] * (1.0 - desat) + l * desat; v[c] = (v[c].clamp(0.0, 255.0) / 255.0).powf(1.0 / gamma) * 255.0; }
        for c in 0..3 { p[c] = v[c].round().clamp(0.0, 255.0) as u8; }
    }
    out
}

/// The frame as a screen shows it when a viewer scales it (e.g. the Photos app at 85%): bilinear resample.
pub fn display_scaled(frame: &RgbImage, scale: f32) -> RgbImage {
    let (w, h) = ((frame.width as f32 * scale).round() as u32, (frame.height as f32 * scale).round() as u32);
    let h_out_to_src = Homography { m: [1.0 / scale, 0.0, 0.0, 0.0, 1.0 / scale, 0.0, 0.0, 0.0, 1.0] };
    frame.warp(&h_out_to_src, w, h)
}

/// Realistic phone photo of a screen: optional display scaling, perspective, camera resolution, colour
/// rendering and noise. Defaults modelled on the first real test photos.
pub fn photograph_realistic(frame: &RgbImage, display_scale: f32, cam_w: u32, cam_h: u32, seed: u64) -> RgbImage {
    let shown = if (display_scale - 1.0).abs() > 1e-3 { display_scaled(frame, display_scale) } else { frame.clone() };
    let cam = photograph(&shown, cam_w, cam_h, 25.0, 0, seed);
    let cam = box_blur(&box_blur(&cam)); // two 3x3 passes ~ Gaussian sigma 1.1 px: lens softness + demosaic
    let coloured = degrade_colour(&cam, [1.0, 0.93, 0.80], 0.30, 1.15);
    // noise after colour rendering
    let mut out = coloured; let mut s = seed | 1;
    for p in out.data.iter_mut() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; let n = (s % 13) as i32 - 6; *p = (*p as i32 + n).clamp(0, 255) as u8; }
    out
}

/// 3x3 box blur (edges clamped).
pub fn box_blur(img: &RgbImage) -> RgbImage {
    let (w, h) = (img.width as i32, img.height as i32);
    let mut out = RgbImage::new(img.width, img.height);
    for y in 0..h { for x in 0..w {
        let mut acc = [0u32; 3];
        for dy in -1..=1 { for dx in -1..=1 {
            let p = img.get((x + dx).clamp(0, w - 1) as u32, (y + dy).clamp(0, h - 1) as u32);
            for c in 0..3 { acc[c] += p[c] as u32; }
        } }
        out.set(x as u32, y as u32, [(acc[0] / 9) as u8, (acc[1] / 9) as u8, (acc[2] / 9) as u8]);
    } }
    out
}

/// The geometry bend measured on handheld phone captures (2026-10-27): a rolling-shutter shear growing
/// linearly down the image plus a smooth random field (3x3 control points, bilinear), both of
/// `amplitude_px` camera px. Deterministic for a seed.
pub fn bend(img: &RgbImage, amplitude_px: f32, seed: u64) -> RgbImage {
    let (w, h) = (img.width, img.height);
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut rnd = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * amplitude_px };
    let shear = (rnd(), rnd());
    let ctrl: Vec<(f32, f32)> = (0..9).map(|_| (rnd(), rnd())).collect();
    let field = |x: f32, y: f32| -> (f32, f32) {
        let (u, v) = (x / (w as f32 - 1.0) * 2.0, y / (h as f32 - 1.0) * 2.0);
        let (i, j) = ((u.floor() as usize).min(1), (v.floor() as usize).min(1));
        let (fu, fv) = (u - i as f32, v - j as f32);
        let c = |a: usize, b: usize| ctrl[b * 3 + a];
        let lerp = |p: (f32, f32), q: (f32, f32), t: f32| (p.0 + (q.0 - p.0) * t, p.1 + (q.1 - p.1) * t);
        let top = lerp(c(i, j), c(i + 1, j), fu);
        let bot = lerp(c(i, j + 1), c(i + 1, j + 1), fu);
        let (fx, fy) = lerp(top, bot, fv);
        let t = y / (h as f32 - 1.0);
        (fx + shear.0 * t, fy + shear.1 * t)
    };
    let mut out = RgbImage::new(w, h);
    for y in 0..h { for x in 0..w {
        let (dx, dy) = field(x as f32, y as f32);
        out.set(x, y, img.sample(x as f32 + dx, y as f32 + dy));
    } }
    out
}
