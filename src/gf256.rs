//! GF(2^8) with primitive polynomial 0x11D (x^8+x^4+x^3+x^2+1), generator 2.

const PRIM: u16 = 0x11D;

struct Tables { exp: [u8; 512], log: [u8; 256] }

static TABLES: std::sync::LazyLock<Tables> = std::sync::LazyLock::new(|| {
    let mut exp = [0u8; 512];
    let mut log = [0u8; 256];
    let mut x: u16 = 1;
    for i in 0..255 {
        exp[i] = x as u8;
        log[x as usize] = i as u8;
        x <<= 1;
        if x & 0x100 != 0 { x ^= PRIM; }
    }
    for i in 255..512 { exp[i] = exp[i - 255]; }
    Tables { exp, log }
});

pub fn exp(i: usize) -> u8 { TABLES.exp[i % 255] }
pub fn log(a: u8) -> u8 { debug_assert!(a != 0); TABLES.log[a as usize] }

pub fn mul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 { return 0; }
    let t = &*TABLES;
    t.exp[t.log[a as usize] as usize + t.log[b as usize] as usize]
}
pub fn inv(a: u8) -> u8 {
    assert!(a != 0, "gf256 inverse of zero");
    TABLES.exp[255 - TABLES.log[a as usize] as usize]
}
pub fn div(a: u8, b: u8) -> u8 { mul(a, inv(b)) }
pub fn pow(a: u8, n: usize) -> u8 {
    if n == 0 { return 1; }
    if a == 0 { return 0; }
    exp((log(a) as usize * n) % 255)
}

pub fn poly_add(a: &[u8], b: &[u8]) -> Vec<u8> {
    let n = a.len().max(b.len());
    (0..n).map(|i| a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0)).collect()
}
pub fn poly_scale(p: &[u8], s: u8) -> Vec<u8> { p.iter().map(|&c| mul(c, s)).collect() }
pub fn poly_mul(a: &[u8], b: &[u8]) -> Vec<u8> {
    if a.is_empty() || b.is_empty() { return vec![]; }
    let mut out = vec![0u8; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        if x == 0 { continue; }
        for (j, &y) in b.iter().enumerate() { out[i + j] ^= mul(x, y); }
    }
    out
}
/// Horner evaluation, coefficients low->high.
pub fn poly_eval(p: &[u8], x: u8) -> u8 {
    let mut acc = 0u8;
    for &c in p.iter().rev() { acc = mul(acc, x) ^ c; }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mul_inverse_round_trips() {
        for a in 1..=255u8 { assert_eq!(mul(a, inv(a)), 1, "a={a}"); }
    }
    #[test]
    fn exp_log_are_inverse() {
        for i in 0..255usize { assert_eq!(log(exp(i)) as usize, i); }
    }
    #[test]
    fn known_products() {
        assert_eq!(mul(2, 128), 0x1D);
        assert_eq!(mul(0, 77), 0);
        assert_eq!(mul(1, 77), 77);
    }
    #[test]
    fn poly_ops() {
        assert_eq!(poly_mul(&[1, 1], &[2, 1]), vec![2, 3, 1]);
        assert_eq!(poly_eval(&[2, 3, 1], 1), 0);
        assert_eq!(poly_eval(&[2, 3, 1], 2), 0);
        assert_eq!(poly_add(&[1, 2], &[1]), vec![0, 2]);
        assert_eq!(poly_scale(&[1, 2], 2), vec![2, 4]);
    }
}
