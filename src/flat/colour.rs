//! Per-frame (and per-block) affine colour model: camera RGB -> palette RGB, fitted on the pilots.

/// palette_rgb = m * [r, g, b, 1].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColourModel { pub m: [[f32; 4]; 3] }

pub const MIN_FIT_SAMPLES: usize = 6;
const LUMA: [f32; 3] = [0.299, 0.587, 0.114];

/// Solve the symmetric 4x4 system a * x = b by Gaussian elimination with partial pivoting.
fn solve4(mut a: [[f64; 4]; 4], mut b: [f64; 4]) -> Option<[f64; 4]> {
    for col in 0..4 {
        let piv = (col..4).max_by(|&i, &j| a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap())?;
        if a[piv][col].abs() < 1e-6 { return None; }
        a.swap(col, piv); b.swap(col, piv);
        for r in col + 1..4 {
            let f = a[r][col] / a[col][col];
            for c in col..4 { a[r][c] -= f * a[col][c]; }
            b[r] -= f * b[col];
        }
    }
    let mut x = [0f64; 4];
    for r in (0..4).rev() {
        let s: f64 = (r + 1..4).map(|c| a[r][c] * x[c]).sum();
        x[r] = (b[r] - s) / a[r][r];
    }
    Some(x)
}

impl ColourModel {
    pub fn identity() -> ColourModel { ColourModel { m: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]] } }
    /// Least-squares affine fit of (camera, palette) pairs; None with fewer than `MIN_FIT_SAMPLES` or a
    /// degenerate set (all samples alike).
    pub fn fit(samples: &[([u8; 3], [u8; 3])]) -> Option<ColourModel> {
        if samples.len() < MIN_FIT_SAMPLES { return None; }
        let mut ata = [[0f64; 4]; 4];
        let mut atb = [[0f64; 4]; 3];
        for &(cam, pal) in samples {
            let v = [cam[0] as f64, cam[1] as f64, cam[2] as f64, 1.0];
            for i in 0..4 { for j in 0..4 { ata[i][j] += v[i] * v[j]; } for k in 0..3 { atb[k][i] += v[i] * pal[k] as f64; } }
        }
        // a sample set with no spread cannot determine the gains: the covariance must be non-trivial
        let n = samples.len() as f64;
        let spread: f64 = (0..3).map(|i| ata[i][i] / n - (ata[i][3] / n).powi(2)).sum();
        if spread < 1.0 { return None; }
        let mut m = [[0f32; 4]; 3];
        for k in 0..3 { let x = solve4(ata, atb[k])?; for i in 0..4 { m[k][i] = x[i] as f32; } }
        Some(ColourModel { m })
    }
    /// Gain + offset on luma only (luma-only alphabets): every output channel is the fitted luma.
    pub fn fit_luma(samples: &[([u8; 3], [u8; 3])]) -> Option<ColourModel> {
        if samples.len() < MIN_FIT_SAMPLES { return None; }
        let (mut sx, mut sy, mut sxx, mut sxy) = (0f64, 0f64, 0f64, 0f64);
        for &(cam, pal) in samples {
            let x = (0..3).map(|k| LUMA[k] as f64 * cam[k] as f64).sum::<f64>();
            let y = (0..3).map(|k| LUMA[k] as f64 * pal[k] as f64).sum::<f64>();
            sx += x; sy += y; sxx += x * x; sxy += x * y;
        }
        let n = samples.len() as f64;
        let den = n * sxx - sx * sx;
        if den.abs() < 1e-3 { return None; }
        let a = (n * sxy - sx * sy) / den;
        let b = (sy - a * sx) / n;
        let row = [(a * LUMA[0] as f64) as f32, (a * LUMA[1] as f64) as f32, (a * LUMA[2] as f64) as f32, b as f32];
        Some(ColourModel { m: [row; 3] })
    }
    pub fn apply(&self, rgb: [u8; 3]) -> [f32; 3] {
        let v = [rgb[0] as f32, rgb[1] as f32, rgb[2] as f32, 1.0];
        let mut out = [0f32; 3];
        for k in 0..3 { out[k] = self.m[k][0] * v[0] + self.m[k][1] * v[1] + self.m[k][2] * v[2] + self.m[k][3]; }
        out
    }
    /// Mean squared error (summed over channels) of the fitted samples.
    pub fn residual(&self, samples: &[([u8; 3], [u8; 3])]) -> f32 {
        if samples.is_empty() { return f32::MAX; }
        let s: f32 = samples.iter().map(|&(cam, pal)| { let f = self.apply(cam); (0..3).map(|k| (f[k] - pal[k] as f32).powi(2)).sum::<f32>() }).sum();
        s / samples.len() as f32
    }
    /// Nearest palette colour in the fitted space and the margin to the runner-up: (d2 - d1) / 4 of the
    /// squared distances, saturated to u16.
    pub fn classify(&self, rgb: [u8; 3], palette: &[[u8; 3]]) -> (u8, u16) {
        let f = self.apply(rgb);
        let (mut best, mut d1, mut d2) = (0u8, f32::MAX, f32::MAX);
        for (i, p) in palette.iter().enumerate() {
            let d = (0..3).map(|k| (f[k] - p[k] as f32).powi(2)).sum::<f32>();
            if d < d1 { d2 = d1; d1 = d; best = i as u8; } else if d < d2 { d2 = d; }
        }
        let margin = if d2 == f32::MAX { u16::MAX } else { ((d2 - d1) / 4.0).clamp(0.0, 65535.0) as u16 };
        (best, margin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flat::Alphabet;

    fn lcg(seed: &mut u64) -> u32 { *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (*seed >> 33) as u32 }
    fn distort(rgb: [u8; 3], seed: &mut u64) -> [u8; 3] {
        let (r, g, b) = (rgb[0] as f32, rgb[1] as f32, rgb[2] as f32);
        let mixed = [0.97 * r + 0.015 * g + 0.015 * b, 0.015 * r + 0.97 * g + 0.015 * b, 0.015 * r + 0.015 * g + 0.97 * b];
        let gains = [0.8, 0.93, 1.0]; let offs = [10.0, 12.0, 15.0];
        let mut out = [0u8; 3];
        for k in 0..3 { let noise = (lcg(seed) % 7) as f32 - 3.0; out[k] = (gains[k] * mixed[k] + offs[k] + noise).clamp(0.0, 255.0) as u8; }
        out
    }
    #[test]
    fn recovers_a_known_affine_distortion() {
        let pal = Alphabet::C16.palette();
        let mut seed = 7u64;
        let samples: Vec<([u8; 3], [u8; 3])> = (0..64).map(|_| { let c = pal[(lcg(&mut seed) % 16) as usize]; (distort(c, &mut seed), c) }).collect();
        let m = ColourModel::fit(&samples).expect("fit");
        assert!(m.residual(&samples) < 40.0, "residual {}", m.residual(&samples));
        for &(cam, truth) in &samples {
            let (idx, margin) = m.classify(cam, pal);
            assert_eq!(pal[idx as usize], truth);
            assert!(margin > 100);
        }
        assert_eq!(ColourModel::identity().apply([10, 20, 30]), [10.0, 20.0, 30.0]);
    }
    #[test]
    fn fit_needs_enough_samples() {
        let pal = Alphabet::C8.palette();
        let five: Vec<([u8; 3], [u8; 3])> = (0..5).map(|i| (pal[i], pal[i])).collect();
        assert!(ColourModel::fit(&five).is_none());
        let same: Vec<([u8; 3], [u8; 3])> = (0..10).map(|_| ([50, 50, 50], pal[0])).collect();
        assert!(ColourModel::fit(&same).is_none(), "singular input must not fit");
    }
    #[test]
    fn luma_fit_on_greys() {
        let pal = Alphabet::G4.palette();
        let samples: Vec<([u8; 3], [u8; 3])> = (0..12).map(|i| { let c = pal[i % 4]; let v = (0.6 * c[0] as f32 + 20.0) as u8; ([v, v, v], c) }).collect();
        let m = ColourModel::fit_luma(&samples).expect("luma fit");
        for &(cam, truth) in &samples { assert_eq!(pal[m.classify(cam, pal).0 as usize], truth); }
        // a tinted grey still classifies by its luma (0.6 * 85 + 20 = 71)
        assert_eq!(m.classify([60, 75, 80], pal).0, 1);
    }
    #[test]
    fn margin_is_small_near_a_boundary() {
        let pal = Alphabet::C8.palette();
        let m = ColourModel::identity();
        let (_, near) = m.classify([255, 0, 127], pal);
        assert!(near < 500, "margin {near}");
        let (idx, far) = m.classify([255, 0, 0], pal);
        assert_eq!(idx, 4); assert!(far > 3000, "margin {far}"); // runner-up is orange, 128 apart: 128^2 / 4
    }
}
