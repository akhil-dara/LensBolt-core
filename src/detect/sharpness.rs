use crate::image::GrayImage;

/// Laplacian values the gate's sample aims at: a 12.6 MP frame (1022 x 766 interior 4x4 boxes) samples
/// every 8th box row, a Full HD frame (358 x 268) keeps the full grid.
const SAMPLE_TARGET: usize = 100_000;

/// Mean absolute Laplacian over a 4x-downscaled image; higher = sharper. The full frame: the gate's own
/// reference (`gate_sharpness`), and what the CLI and the receiver's failure report print.
pub fn sharpness(gray: &GrayImage) -> f32 { sharpness_rows(gray, 1) }

/// Box rows between the sampled Laplacian rows of `gray` (1 = the full grid).
pub fn sample_step(gray: &GrayImage) -> usize {
    let (gw, gh) = ((gray.width / 4) as usize, (gray.height / 4) as usize);
    if gw < 3 || gh < 3 { return 1; }
    ((gw - 2) * (gh - 2)).div_ceil(SAMPLE_TARGET).max(1)
}

/// `sharpness` from every `step`-th row of the 4x4 box grid (each with its rows above and below), the
/// rows spread evenly: the same statistic over a systematic sample of rows, so its scale is the
/// full-frame score's. `step` 1 is exactly `sharpness`. Reads the rows as slices, allocates three rows.
pub fn sharpness_rows(gray: &GrayImage, step: usize) -> f32 {
    let (w, gw, gh) = (gray.width as usize, (gray.width / 4) as usize, (gray.height / 4) as usize);
    if gw < 3 || gh < 3 { return 0.0; }
    let step = step.max(1);
    // box rows 1..=gh-2 carry a Laplacian; the sampled ones centred in their strides
    let first = 1 + ((gh - 2 - 1) % step) / 2;
    // row r of the box grid lives in slot r % 3: a centre row and its two neighbours never collide
    let mut rows = [vec![0u8; gw], vec![0u8; gw], vec![0u8; gw]];
    let mut have = [usize::MAX; 3];
    let mut cols = vec![0u16; 4 * gw];
    let (mut acc, mut n) = (0u64, 0u64);
    for cy in (first..gh - 1).step_by(step) {
        for r in cy - 1..=cy + 1 {
            if have[r % 3] == r { continue; }
            let px = |k: usize| &gray.data[(4 * r + k) * w..(4 * r + k) * w + 4 * gw];
            let (a, b, c, d) = (px(0), px(1), px(2), px(3));
            for x in 0..4 * gw { cols[x] = a[x] as u16 + b[x] as u16 + c[x] as u16 + d[x] as u16; }
            for (o, q) in rows[r % 3].iter_mut().zip(cols.chunks_exact(4)) { *o = ((q[0] + q[1] + q[2] + q[3]) / 16) as u8; }
            have[r % 3] = r;
        }
        let (up, mid, down) = (&rows[(cy - 1) % 3], &rows[cy % 3], &rows[(cy + 1) % 3]);
        let mut row = 0u32;
        for x in 1..gw - 1 {
            let l = mid[x] as i32 * 4 - mid[x - 1] as i32 - mid[x + 1] as i32 - up[x] as i32 - down[x] as i32;
            row += l.unsigned_abs();
        }
        acc += row as u64; n += (gw - 2) as u64;
    }
    acc as f32 / n as f32
}

/// Relative band around the gate's minimum inside which the gate scores the full frame instead of the
/// sample: a sample can then only decide differently from the full frame when its error exceeds 25%.
pub const GATE_MARGIN: f32 = 0.25;

/// The score the gate compares with `min`: the row sample's, or the full frame's when the sample lands
/// within `GATE_MARGIN` of `min`. Real captures score far from the gate (sharp 1-2 px tiles 45-70 against
/// a minimum of 4), so the full frame is scored only for a borderline capture, at a few ms.
pub fn gate_sharpness(gray: &GrayImage, min: f32) -> f32 {
    let step = sample_step(gray);
    let s = sharpness_rows(gray, step);
    if step > 1 && (s - min).abs() < GATE_MARGIN * min { sharpness(gray) } else { s }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::synth_flat;
    #[test]
    fn blurred_frame_scores_lower() {
        let (_, cam, _) = synth_flat(crate::flat::CellFormat { pitch: 4, alphabet: crate::flat::Alphabet::C8 }, 1600, 1500, 0.0, 0);
        let sharp = sharpness(&cam.to_gray());
        let g = cam.to_gray();
        let mut b = g.clone();
        for y in 2..g.height - 2 { for x in 2..g.width - 2 {
            let mut s = 0u32; for dy in 0..5 { for dx in 0..5 { s += g.get(x + dx - 2, y + dy - 2) as u32; } }
            b.data[(y * g.width + x) as usize] = (s / 25) as u8;
        } }
        let blurred = sharpness(&b);
        assert!(sharp > 6.0, "sharp {sharp}");
        assert!(blurred < sharp * 0.75, "sharp {sharp} blurred {blurred}");
        // the gate's sample ranks them the same way
        let (ss, sb) = (sharpness_rows(&g, 4), sharpness_rows(&b, 4));
        assert!(sb < ss * 0.75, "sampled: sharp {ss} blurred {sb}");
    }
    #[test]
    fn full_grid_rows_equal_the_downscaled_score() {
        // the score as defined: the Laplacian over `downscale(4)` (odd sizes leave pixels out)
        let (_, cam, _) = synth_flat(crate::flat::CellFormat { pitch: 2, alphabet: crate::flat::Alphabet::Bw }, 1203, 1001, 0.0, 0);
        let g = cam.to_gray();
        let d = g.downscale(4);
        let mut acc = 0u64; let mut n = 0u64;
        for y in 1..d.height - 1 { for x in 1..d.width - 1 {
            let l = d.get(x, y) as i32 * 4 - d.get(x - 1, y) as i32 - d.get(x + 1, y) as i32 - d.get(x, y - 1) as i32 - d.get(x, y + 1) as i32;
            acc += l.unsigned_abs() as u64; n += 1;
        } }
        assert_eq!(sharpness(&g), acc as f32 / n as f32);
        assert_eq!(sharpness(&GrayImage::new(11, 11)), 0.0);
        assert_eq!(sharpness(&GrayImage::new(8, 400)), 0.0);
    }
    #[test]
    fn step_keeps_small_frames_whole() {
        assert_eq!(sample_step(&GrayImage::new(1440, 1080)), 1);
        assert_eq!(sample_step(&GrayImage::new(4096, 3072)), 8);
        assert_eq!(sample_step(&GrayImage::new(640, 480)), 1);
    }
}
