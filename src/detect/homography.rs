#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Homography { pub m: [f32; 9] }

impl Homography {
    pub fn identity() -> Homography { Homography { m: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] } }

    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        let m = &self.m;
        let w = m[6] * x + m[7] * y + m[8];
        let w = if w.abs() < 1e-12 { 1e-12 } else { w };
        ((m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w)
    }

    pub fn from_points(src: [(f32, f32); 4], dst: [(f32, f32); 4]) -> Option<Homography> {
        let mut a = [[0f64; 9]; 8];
        for i in 0..4 {
            let (x, y) = (src[i].0 as f64, src[i].1 as f64);
            let (u, v) = (dst[i].0 as f64, dst[i].1 as f64);
            a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
            a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
        }
        for col in 0..8 {
            let piv = (col..8).max_by(|&r1, &r2| a[r1][col].abs().partial_cmp(&a[r2][col].abs()).unwrap())?;
            if a[piv][col].abs() < 1e-9 { return None; }
            a.swap(col, piv);
            for r in 0..8 {
                if r == col { continue; }
                let f = a[r][col] / a[col][col];
                if f == 0.0 { continue; }
                for c in col..9 { a[r][c] -= f * a[col][c]; }
            }
        }
        let mut m = [0f32; 9];
        for i in 0..8 { m[i] = (a[i][8] / a[i][i]) as f32; }
        m[8] = 1.0;
        Some(Homography { m })
    }

    pub fn inverse(&self) -> Option<Homography> {
        let m: Vec<f64> = self.m.iter().map(|&v| v as f64).collect();
        let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
        if det.abs() < 1e-12 { return None; }
        let inv = [
            (m[4] * m[8] - m[5] * m[7]) / det, (m[2] * m[7] - m[1] * m[8]) / det, (m[1] * m[5] - m[2] * m[4]) / det,
            (m[5] * m[6] - m[3] * m[8]) / det, (m[0] * m[8] - m[2] * m[6]) / det, (m[2] * m[3] - m[0] * m[5]) / det,
            (m[3] * m[7] - m[4] * m[6]) / det, (m[1] * m[6] - m[0] * m[7]) / det, (m[0] * m[4] - m[1] * m[3]) / det,
        ];
        let s = inv[8];
        let mut out = [0f32; 9];
        for i in 0..9 { out[i] = (inv[i] / s) as f32; }
        Some(Homography { m: out })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn close(a: (f32, f32), b: (f32, f32)) -> bool { (a.0 - b.0).abs() < 1e-2 && (a.1 - b.1).abs() < 1e-2 }
    #[test]
    fn identity_maps_to_self() {
        let h = Homography::identity();
        assert!(close(h.apply(12.5, -3.0), (12.5, -3.0)));
    }
    #[test]
    fn four_points_map_exactly_and_invert() {
        let src = [(0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0)];
        let dst = [(10.0, 12.0), (205.0, 20.0), (190.0, 130.0), (5.0, 110.0)];
        let h = Homography::from_points(src, dst).unwrap();
        for i in 0..4 { assert!(close(h.apply(src[i].0, src[i].1), dst[i]), "pt {i}: {:?}", h.apply(src[i].0, src[i].1)); }
        let inv = h.inverse().unwrap();
        assert!(close(inv.apply(dst[2].0, dst[2].1), src[2]));
        let mid = h.apply(50.0, 25.0);
        assert!(close(inv.apply(mid.0, mid.1), (50.0, 25.0)));
    }
    #[test]
    fn degenerate_points_return_none() {
        let src = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)];
        assert!(Homography::from_points(src, src).is_none());
    }
    #[test]
    fn warp_identity_copies_image() {
        use crate::image::RgbImage;
        let mut im = RgbImage::new(4, 4);
        im.set(1, 2, [9, 8, 7]);
        let out = im.warp(&Homography::identity(), 4, 4);
        assert_eq!(out, im);
    }
    #[test]
    fn warp_scale_down_by_two() {
        use crate::image::RgbImage;
        let mut im = RgbImage::new(8, 8);
        im.fill_rect(0, 0, 4, 8, [200, 200, 200]);
        let h = Homography { m: [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0] };
        let out = im.warp(&h, 4, 4);
        assert_eq!(out.get(0, 0), [200, 200, 200]);
        assert_eq!(out.get(3, 3), [0, 0, 0]);
    }
}
