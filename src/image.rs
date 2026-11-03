use std::io::{self, BufReader, BufWriter};
use std::path::Path;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RgbImage { pub width: u32, pub height: u32, pub data: Vec<u8> }

impl RgbImage {
    pub fn new(width: u32, height: u32) -> RgbImage { RgbImage { width, height, data: vec![0; (width * height * 3) as usize] } }
    #[inline] fn idx(&self, x: u32, y: u32) -> usize { ((y * self.width + x) * 3) as usize }
    pub fn get(&self, x: u32, y: u32) -> [u8; 3] { let i = self.idx(x, y); [self.data[i], self.data[i + 1], self.data[i + 2]] }
    pub fn set(&mut self, x: u32, y: u32, rgb: [u8; 3]) { let i = self.idx(x, y); self.data[i..i + 3].copy_from_slice(&rgb); }
    pub fn fill_rect(&mut self, x: u32, y: u32, w: u32, h: u32, rgb: [u8; 3]) {
        for yy in y..(y + h).min(self.height) { for xx in x..(x + w).min(self.width) { self.set(xx, yy, rgb); } }
    }
    #[inline]
    pub fn sample(&self, x: f32, y: f32) -> [u8; 3] {
        if self.width < 2 || self.height < 2 {
            let (x, y) = (x.clamp(0.0, self.width as f32 - 1.0), y.clamp(0.0, self.height as f32 - 1.0));
            let (x0, y0) = (x.floor() as u32, y.floor() as u32);
            let (x1, y1) = ((x0 + 1).min(self.width - 1), (y0 + 1).min(self.height - 1));
            let (fx, fy) = (x - x0 as f32, y - y0 as f32);
            let (a, b, c, d) = (self.get(x0, y0), self.get(x1, y0), self.get(x0, y1), self.get(x1, y1));
            let mut out = [0u8; 3];
            for k in 0..3 { let top = a[k] as f32 * (1.0 - fx) + b[k] as f32 * fx; let bot = c[k] as f32 * (1.0 - fx) + d[k] as f32 * fx; out[k] = (top * (1.0 - fy) + bot * fy).round() as u8; }
            return out;
        }
        let w = self.width as usize;
        let (x, y) = (x.max(0.0).min(self.width as f32 - 1.0), y.max(0.0).min(self.height as f32 - 1.0));
        let (x0, y0) = ((x as usize).min(w - 2), (y as usize).min(self.height as usize - 2));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let i = (y0 * w + x0) * 3;
        let d = &self.data[i..i + 3 * w + 6];
        let mut out = [0u8; 3];
        for k in 0..3 {
            let (a, b, c, e) = (d[k] as f32, d[3 + k] as f32, d[3 * w + k] as f32, d[3 * w + 3 + k] as f32);
            let top = a + (b - a) * fx;
            let bot = c + (e - c) * fx;
            out[k] = (top + (bot - top) * fy + 0.5) as u8;
        }
        out
    }
    /// Android `YUV_420_888` (BT.601 full range, as camera2 hands it out) -> RGB. `u`/`v` start at the plane's
    /// first sample; `uv_pixel` is 1 (planar) or 2 (interleaved). Rows are converted on all cores.
    #[allow(clippy::too_many_arguments)]
    pub fn from_yuv420(width: u32, height: u32, y: &[u8], y_row: usize, u: &[u8], v: &[u8], uv_row: usize, uv_pixel: usize) -> RgbImage {
        let (w, h) = (width as usize, height as usize);
        let mut out = RgbImage::new(width, height);
        if w == 0 || h == 0 { return out; }
        // LB_THREADS as the decoder (the phone's workers each convert their own capture on one thread)
        let threads = std::env::var("LB_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)).clamp(1, 8);
        let rows_per = h.div_ceil(threads);
        let convert = move |ci: usize, chunk: &mut [u8]| {
            for (ri, row) in chunk.chunks_exact_mut(w * 3).enumerate() {
                let yy = ci * rows_per + ri;
                let (yr, cr) = (&y[yy * y_row..], (yy / 2) * uv_row);
                for (x, px) in row.chunks_exact_mut(3).enumerate() {
                    let l = yr[x] as i32;
                    let c = cr + (x / 2) * uv_pixel;
                    let (cb, cr_) = (u.get(c).copied().unwrap_or(128) as i32 - 128, v.get(c).copied().unwrap_or(128) as i32 - 128);
                    // fixed point, 16 fractional bits: R = Y + 1.402 Cr, G = Y - 0.344 Cb - 0.714 Cr, B = Y + 1.772 Cb
                    px[0] = (l + ((91_881 * cr_ + 32_768) >> 16)).clamp(0, 255) as u8;
                    px[1] = (l - ((22_554 * cb + 46_802 * cr_ - 32_768) >> 16)).clamp(0, 255) as u8;
                    px[2] = (l + ((116_130 * cb + 32_768) >> 16)).clamp(0, 255) as u8;
                }
            }
        };
        // one thread runs here: browser WebAssembly cannot start threads
        if threads == 1 { convert(0, &mut out.data); return out; }
        std::thread::scope(|sc| {
            for (ci, chunk) in out.data.chunks_mut(rows_per * w * 3).enumerate() { sc.spawn(move || convert(ci, chunk)); }
        });
        out
    }
    pub fn to_gray(&self) -> GrayImage {
        let data = self.data.chunks(3).map(|p| ((77 * p[0] as u32 + 150 * p[1] as u32 + 29 * p[2] as u32) >> 8) as u8).collect();
        GrayImage { width: self.width, height: self.height, data }
    }
    /// Output pixel (x,y) = bilinear sample of `self` at `out_to_src.apply(x+0.5, y+0.5) - 0.5`. Outside -> black.
    pub fn warp(&self, out_to_src: &crate::detect::Homography, out_w: u32, out_h: u32) -> RgbImage {
        let mut out = RgbImage::new(out_w, out_h);
        for y in 0..out_h {
            for x in 0..out_w {
                let (sx, sy) = out_to_src.apply(x as f32 + 0.5, y as f32 + 0.5);
                let (sx, sy) = (sx - 0.5, sy - 0.5);
                if sx < -0.5 || sy < -0.5 || sx > self.width as f32 - 0.5 || sy > self.height as f32 - 0.5 { continue; }
                out.set(x, y, self.sample(sx, sy));
            }
        }
        out
    }
    pub fn save_png(&self, path: &Path) -> io::Result<()> {
        let f = BufWriter::new(std::fs::File::create(path)?);
        let mut enc = png::Encoder::new(f, self.width, self.height);
        enc.set_color(png::ColorType::Rgb); enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().map_err(io::Error::other)?;
        w.write_image_data(&self.data).map_err(io::Error::other)
    }
    pub fn load_png(path: &Path) -> io::Result<RgbImage> {
        let dec = png::Decoder::new(BufReader::new(std::fs::File::open(path)?));
        let mut reader = dec.read_info().map_err(io::Error::other)?;
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).map_err(io::Error::other)?;
        if info.bit_depth != png::BitDepth::Eight { return Err(io::Error::other("only 8-bit PNG supported")); }
        let data = match info.color_type {
            png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
            png::ColorType::Rgba => buf[..info.buffer_size()].chunks(4).flat_map(|p| [p[0], p[1], p[2]]).collect(),
            png::ColorType::Grayscale => buf[..info.buffer_size()].iter().flat_map(|&v| [v, v, v]).collect(),
            _ => return Err(io::Error::other("unsupported PNG colour type")),
        };
        Ok(RgbImage { width: info.width, height: info.height, data })
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GrayImage { pub width: u32, pub height: u32, pub data: Vec<u8> }

/// Summed-area table of a `GrayImage` (u32: images up to 16.8 M px of 255).
pub struct Integral { pub width: u32, pub height: u32, sum: Vec<u32> }
impl Integral {
    /// Mean over pixels x0..x1, y0..y1 (exclusive ends), clamped to the image; 0 for an empty box.
    #[inline]
    pub fn box_mean(&self, x0: i32, y0: i32, x1: i32, y1: i32) -> f32 {
        let (w, h) = (self.width as i32, self.height as i32);
        let (x0, y0, x1, y1) = (x0.clamp(0, w), y0.clamp(0, h), x1.clamp(0, w), y1.clamp(0, h));
        if x1 <= x0 || y1 <= y0 { return 0.0; }
        let s = (w + 1) as usize;
        let at = |x: i32, y: i32| self.sum[y as usize * s + x as usize] as i64;
        let total = at(x1, y1) - at(x0, y1) - at(x1, y0) + at(x0, y0);
        total as f32 / ((x1 - x0) * (y1 - y0)) as f32
    }
}

impl GrayImage {
    pub fn new(width: u32, height: u32) -> GrayImage { GrayImage { width, height, data: vec![0; (width * height) as usize] } }
    pub fn get(&self, x: u32, y: u32) -> u8 { self.data[(y * self.width + x) as usize] }
    /// A camera Y plane with row padding (`row_stride` >= width), copied row by row.
    pub fn from_plane(width: u32, height: u32, data: &[u8], row_stride: usize) -> GrayImage {
        let w = width as usize;
        let mut out = Vec::with_capacity(w * height as usize);
        for y in 0..height as usize { out.extend_from_slice(&data[y * row_stride..y * row_stride + w]); }
        GrayImage { width, height, data: out }
    }
    /// Summed-area table: any box mean in four reads.
    pub fn integral(&self) -> Integral {
        let (w, h) = (self.width as usize, self.height as usize);
        let stride = w + 1;
        let mut sum = vec![0u32; stride * (h + 1)];
        for y in 0..h {
            let mut row = 0u32;
            let (src, (above, cur)) = (&self.data[y * w..(y + 1) * w], sum.split_at_mut((y + 1) * stride));
            let above = &above[y * stride..];
            for x in 0..w { row += src[x] as u32; cur[x + 1] = above[x + 1] + row; }
        }
        Integral { width: self.width, height: self.height, sum }
    }
    /// Bilinear sample, clamped to the image. The decoder's hottest function (~125 calls per cell): the
    /// coordinates are clamped first, so truncation equals floor and the four reads are always in bounds.
    #[inline]
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let w = self.width as usize;
        if self.width < 2 || self.height < 2 { return self.sample_general(x, y); }
        // clamp the integer cell, not the float: `height - 1.0001` rounds to `height - 1` in f32 above ~2048
        let (x, y) = (x.max(0.0).min(self.width as f32 - 1.0), y.max(0.0).min(self.height as f32 - 1.0));
        let (x0, y0) = ((x as usize).min(w - 2), (y as usize).min(self.height as usize - 2));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let i = y0 * w + x0;
        let d = &self.data[i..i + w + 2];
        let (a, b, c, e) = (d[0] as f32, d[1] as f32, d[w] as f32, d[w + 1] as f32);
        let top = a + (b - a) * fx;
        let bot = c + (e - c) * fx;
        top + (bot - top) * fy
    }
    /// Clamped bilinear sample for any size (images narrower or shorter than 2 px).
    fn sample_general(&self, x: f32, y: f32) -> f32 {
        let (x, y) = (x.clamp(0.0, self.width as f32 - 1.0), y.clamp(0.0, self.height as f32 - 1.0));
        let (x0, y0) = (x.floor() as u32, y.floor() as u32);
        let (x1, y1) = ((x0 + 1).min(self.width - 1), (y0 + 1).min(self.height - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let top = self.get(x0, y0) as f32 * (1.0 - fx) + self.get(x1, y0) as f32 * fx;
        let bot = self.get(x0, y1) as f32 * (1.0 - fx) + self.get(x1, y1) as f32 * fx;
        top * (1.0 - fy) + bot * fy
    }
    pub fn downscale(&self, factor: u32) -> GrayImage {
        let (w, h) = (self.width / factor, self.height / factor);
        let mut out = GrayImage::new(w, h);
        for y in 0..h { for x in 0..w {
            let mut sum = 0u32;
            for dy in 0..factor { for dx in 0..factor { sum += self.get(x * factor + dx, y * factor + dy) as u32; } }
            out.data[(y * w + x) as usize] = (sum / (factor * factor)) as u8;
        } }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn set_get_fill_and_gray() {
        let mut im = RgbImage::new(4, 3);
        im.set(1, 2, [10, 20, 30]);
        assert_eq!(im.get(1, 2), [10, 20, 30]);
        im.fill_rect(2, 0, 2, 2, [255, 255, 255]);
        assert_eq!(im.get(3, 1), [255, 255, 255]);
        assert_eq!(im.get(3, 2), [0, 0, 0]);
        let g = im.to_gray();
        assert_eq!(g.get(3, 1), 255);
        assert_eq!(g.get(0, 0), 0);
    }
    #[test]
    fn sampling_at_the_far_edge_of_a_large_image_stays_in_bounds() {
        // f32 cannot hold `2268 - 1.0001`: it rounds to 2267, one row too far (found on a 4032x2268 photo)
        let g = GrayImage::new(4032, 2268);
        for (x, y) in [(4031.9, 2267.9), (4032.0, 2268.0), (1e9, 1e9), (4031.0, 2267.0)] { let _ = g.sample(x, y); }
        let im = RgbImage::new(4032, 2268);
        for (x, y) in [(4031.9, 2267.9), (4032.0, 2268.0), (1e9, 1e9)] { let _ = im.sample(x, y); }
    }
    #[test]
    fn bilinear_sample_interpolates() {
        let mut g = GrayImage::new(2, 1);
        g.data = vec![0, 200];
        assert_eq!(g.sample(0.5, 0.0), 100.0);
        assert_eq!(g.sample(-5.0, 9.0), 0.0);
        let mut im = RgbImage::new(2, 1);
        im.set(1, 0, [200, 100, 0]);
        assert_eq!(im.sample(0.5, 0.0), [100, 50, 0]);
    }
    #[test]
    fn downscale_box_averages() {
        let mut g = GrayImage::new(4, 2);
        g.data = vec![0, 100, 50, 50, 200, 100, 50, 50];
        let d = g.downscale(2);
        assert_eq!((d.width, d.height), (2, 1));
        assert_eq!(d.data, vec![100, 50]);
    }
    #[test]
    fn png_round_trip() {
        let mut im = RgbImage::new(5, 4);
        for y in 0..4 { for x in 0..5 { im.set(x, y, [x as u8 * 40, y as u8 * 60, 7]); } }
        let dir = std::env::temp_dir().join("lensbolt_png_test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.png");
        im.save_png(&p).unwrap();
        assert_eq!(RgbImage::load_png(&p).unwrap(), im);
    }
}
