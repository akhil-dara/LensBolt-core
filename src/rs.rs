//! Reed–Solomon over GF(256), systematic, roots α^0..α^(parity-1).
//! Codeword index i  <->  coefficient of x^(n-1-i).
use crate::gf256 as gf;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum RsError { TooManyErrors, BadLength }

pub const SAFETY_MARGIN: usize = 2;

/// `synd_mul[j][v]` = v · α^j: the syndromes are Horner evaluations at fixed roots, one table lookup per
/// symbol instead of a general multiply (a clean codeword costs its syndromes and nothing else).
pub struct RsCodec { parity: usize, gen: Vec<u8>, synd_mul: Vec<[u8; 256]> }

impl RsCodec {
    pub fn new(parity: usize) -> RsCodec {
        assert!((2..=128).contains(&parity));
        let mut gen = vec![1u8];
        for i in 0..parity { gen = gf::poly_mul(&gen, &[gf::exp(i), 1]); }
        let synd_mul = (0..parity).map(|j| { let x = gf::exp(j); std::array::from_fn(|v| gf::mul(v as u8, x)) }).collect();
        RsCodec { parity, gen, synd_mul }
    }
    pub fn parity(&self) -> usize { self.parity }

    pub fn encode(&self, data: &[u8]) -> Vec<u8> {
        assert!(data.len() + self.parity <= 255);
        let mut out = data.to_vec();
        out.extend(std::iter::repeat(0u8).take(self.parity));
        let glen = self.gen.len();
        for i in 0..data.len() {
            let coef = out[i];
            if coef == 0 { continue; }
            for j in 1..glen {
                out[i + j] ^= gf::mul(self.gen[glen - 1 - j], coef);
            }
        }
        let mut cw = data.to_vec();
        cw.extend_from_slice(&out[data.len()..]);
        cw
    }

    fn syndromes(&self, cw: &[u8]) -> Vec<u8> {
        // the syndromes advance in groups of 16 (then 8, 4, 1) over the word, each group's accumulators in
        // registers: independent lookup chains the CPU overlaps (one syndrome at a time is a single dependent
        // chain, a table-lookup latency per step; all of them at once keep the accumulators in memory, a load
        // and a store per step)
        fn group<const L: usize>(cw: &[u8], t: &[[u8; 256]], out: &mut [u8]) {
            let t: &[[u8; 256]; L] = t.try_into().unwrap();
            let mut a = [0u8; L];
            for &c in cw { for q in 0..L { a[q] = t[q][a[q] as usize] ^ c; } }
            out.copy_from_slice(&a);
        }
        let mut acc = vec![0u8; self.parity];
        let mut j = 0;
        while j < self.parity {
            let left = self.parity - j;
            let l = if left >= 16 { 16 } else if left >= 8 { 8 } else if left >= 4 { 4 } else { 1 };
            let (t, out) = (&self.synd_mul[j..j + l], &mut acc[j..j + l]);
            match l { 16 => group::<16>(cw, t, out), 8 => group::<8>(cw, t, out), 4 => group::<4>(cw, t, out), _ => group::<1>(cw, t, out) }
            j += l;
        }
        acc
    }

    pub fn decode(&self, cw: &mut [u8], erasures: &[usize]) -> Result<usize, RsError> {
        let n = cw.len();
        if n > 255 || n <= self.parity { return Err(RsError::BadLength); }
        if erasures.iter().any(|&e| e >= n) { return Err(RsError::BadLength); }
        if erasures.len() + SAFETY_MARGIN > self.parity { return Err(RsError::TooManyErrors); }
        let (p, ne) = (self.parity, erasures.len());
        let synd = self.syndromes(cw);
        if synd.iter().all(|&s| s == 0) { return Ok(0); }
        let lt = &*LOGS;

        // Erasure locator Γ(x) = Π (1 + X_i x), X_i = α^(n-1-i) (log X_i = n-1-i <= 254), built in place
        let mut gamma = [0u8; 128];
        gamma[0] = 1;
        for (k, &e) in erasures.iter().enumerate() {
            let lx = (n - 1 - e) as u16;
            for j in (1..=k + 1).rev() { gamma[j] ^= lt.mul_log(gamma[j - 1], lx); }
        }
        let gamma = &gamma[..ne + 1];
        // Modified syndromes T(x) = S(x)Γ(x) mod x^parity
        let mut t = [0u8; 128];
        let ls: [u16; 128] = std::array::from_fn(|a| lt.log0[*synd.get(a).unwrap_or(&0) as usize]);
        for (b, &g) in gamma.iter().enumerate() {
            let lg = lt.log0[g as usize];
            for a in 0..p - b { t[a + b] ^= lt.exp0[(ls[a] + lg) as usize]; }
        }

        // Safety margin: never spend the last 2 parity symbols. At full budget the
        // miscorrection rate on hopeless words is ~1/150; the margin cuts it by 256^2.
        let budget = self.parity - erasures.len();
        let (lambda, l) = berlekamp_massey(&t[ne..p], budget);
        if 2 * l + erasures.len() + SAFETY_MARGIN > self.parity { return Err(RsError::TooManyErrors); }

        let psi = gf::poly_mul(&lambda, gamma);
        // the roots of Ψ = ΛΓ: a field has no zero divisors, so Ψ(X_i^-1) = 0 exactly where Γ or Λ is zero;
        // Γ's roots are the erasures themselves (X_i = X_e only for i = e, as n <= 255), so only Λ (degree
        // l, a few errors) is evaluated at every position, not Ψ (degree l + erasures)
        let mut erased = [false; 255];
        for &e in erasures { erased[e] = true; }
        // Λ(X_i^-1) at every position, Chien style: X_i^-1 = α^u with u = (i + 1 - n) mod 255, one more
        // step of α per position, so each term λ_k α^(k·u) advances its log by k
        let mut terms: Vec<(u16, u16)> = lambda.iter().enumerate().skip(1).filter(|&(_, &c)| c != 0)
            .map(|(k, &c)| (((lt.log0[c as usize] as usize + k * ((256 - n) % 255)) % 255) as u16, k as u16)).collect();
        let mut positions = Vec::new();
        for i in 0..n {
            let mut v = lambda[0];
            for (e, k) in terms.iter_mut() { v ^= lt.exp0[*e as usize]; *e += *k; if *e >= 255 { *e -= 255; } }
            if erased[i] || v == 0 { positions.push(i); }
        }
        let expected = l + erasures.len();
        if positions.len() != expected { return Err(RsError::TooManyErrors); }

        // Forney. Ω = SΨ mod x^parity = TΛ mod x^parity (T = SΓ mod x^parity): Λ has a few terms, Ψ dozens.
        // Ω and Ψ' are evaluated as sums of independent terms in the log domain (Horner is one dependent
        // chain of multiplies), each term c_k·x^k = α^(log c_k + k·log x)
        let mut omega = [0u8; 128];
        let ltt: [u16; 128] = std::array::from_fn(|k| lt.log0[t[k] as usize]);
        for (a, &c) in lambda.iter().enumerate().take(p) {
            let lc = lt.log0[c as usize];
            for k in a..p { omega[k] ^= lt.exp0[(ltt[k - a] + lc) as usize]; }
        }
        let lom: [u16; 128] = std::array::from_fn(|k| lt.log0[omega[k] as usize]);
        // Ψ'(x) = Σ_{k odd} ψ_k x^(k-1) (characteristic 2)
        let lpsi: Vec<u16> = psi.iter().skip(1).step_by(2).map(|&c| lt.log0[c as usize]).collect();
        let mut fixes: Vec<(usize, u8)> = Vec::with_capacity(positions.len());
        for &i in &positions {
            let lxi = (n - 1 - i) as u16;
            let u = (255 - lxi) % 255; // log X_i^-1
            let (mut num, mut e) = (0u8, 0u16);
            for &lc in &lom[..p] { num ^= lt.exp0[(lc + e) as usize]; e += u; if e >= 255 { e -= 255; } }
            let u2 = (2 * u) % 255;
            let (mut den, mut e) = (0u8, 0u16);
            for &lc in &lpsi { den ^= lt.exp0[(lc + e) as usize]; e += u2; if e >= 255 { e -= 255; } }
            if den == 0 { return Err(RsError::TooManyErrors); }
            // mag = X_i · num / den
            let mag = if num == 0 { 0 } else { gf::exp((lxi + lt.log0[num as usize] + 255 - lt.log0[den as usize]) as usize) };
            cw[i] ^= mag;
            fixes.push((i, mag));
        }
        // the corrected word's syndromes, by linearity: S_j + Σ mag_i X_i^j (the word itself is not re-read)
        let mut check = [0u8; 128];
        check[..p].copy_from_slice(&synd);
        for &(i, mag) in &fixes {
            if mag == 0 { continue; }
            let lxi = (n - 1 - i) as u16;
            let mut e = lt.log0[mag as usize];
            for s in &mut check[..p] { *s ^= lt.exp0[e as usize]; e += lxi; if e >= 255 { e -= 255; } }
        }
        if check[..p].iter().any(|&s| s != 0) { return Err(RsError::TooManyErrors); }
        Ok(positions.len())
    }
}

/// Log-domain tables with zero folded in, so a product needs no branch: `log0[0]` = `ZERO_LOG`, and
/// `exp0[a + b]` = α^(a+b) for logs a, b <= 254 (sums up to `MAX_LOG_SUM`) and 0 when either is `ZERO_LOG`
/// (any index from `ZERO_LOG` up, > `MAX_LOG_SUM`, to 2 · `ZERO_LOG`).
const ZERO_LOG: u16 = 512;
const MAX_LOG_SUM: usize = 2 * 254;
struct Logs { exp0: [u8; 2 * ZERO_LOG as usize + 1], log0: [u16; 256] }
impl Logs {
    /// v · α^lx (lx <= 254).
    #[inline]
    fn mul_log(&self, v: u8, lx: u16) -> u8 { self.exp0[(self.log0[v as usize] + lx) as usize] }
}
static LOGS: std::sync::LazyLock<Logs> = std::sync::LazyLock::new(|| {
    let mut exp0 = [0u8; 2 * ZERO_LOG as usize + 1];
    for (i, v) in exp0.iter_mut().enumerate().take(MAX_LOG_SUM + 1) { *v = gf::exp(i); }
    let log0 = std::array::from_fn(|v| if v == 0 { ZERO_LOG } else { gf::log(v as u8) as u16 });
    Logs { exp0, log0 }
});

/// Returns (Λ coefficients low->high, L = number of errors located).
fn berlekamp_massey(s: &[u8], max_len: usize) -> (Vec<u8>, usize) {
    // fixed arrays and log-domain products in place of a vector per step; the same steps and values. A
    // polynomial's length grows by at most one per step (c: <= step + 2), so s.len() <= 128 fits.
    let lt = &*LOGS;
    let (mut c, mut b, mut t) = ([0u8; 256], [0u8; 256], [0u8; 256]);
    c[0] = 1; b[0] = 1;
    let (mut clen, mut blen) = (1usize, 1usize);
    let mut l = 0usize;
    let mut m = 1usize;
    let mut bb = 1u8;
    for nn in 0..s.len().min(max_len) {
        let mut d = s[nn];
        for i in 1..=l.min(clen - 1) { d ^= lt.exp0[(lt.log0[c[i] as usize] + lt.log0[s[nn - i] as usize]) as usize]; }
        if d == 0 { m += 1; continue; }
        t[..clen].copy_from_slice(&c[..clen]);
        let tlen = clen;
        // c += x^m · (d / bb) · b
        let lcoef = (lt.log0[d as usize] + 255 - lt.log0[bb as usize]) % 255;
        for k in 0..blen { c[m + k] ^= lt.exp0[(lt.log0[b[k] as usize] + lcoef) as usize]; }
        clen = clen.max(m + blen);
        if 2 * l <= nn {
            l = nn + 1 - l; b[..tlen].copy_from_slice(&t[..tlen]); blen = tlen; bb = d; m = 1;
        } else { m += 1; }
    }
    (c[..clen.min(l + 1)].to_vec(), l)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample(n: usize) -> Vec<u8> { (0..n).map(|i| (i * 7 + 3) as u8).collect() }

    #[test]
    fn encode_then_clean_decode_is_identity() {
        let rs = RsCodec::new(20);
        let data = sample(235);
        let mut cw = rs.encode(&data);
        assert_eq!(cw.len(), 255);
        assert_eq!(rs.decode(&mut cw, &[]), Ok(0));
        assert_eq!(&cw[..235], &data[..]);
    }
    #[test]
    fn corrects_up_to_parity_over_two_random_errors() {
        let rs = RsCodec::new(20);
        let data = sample(235);
        let clean = rs.encode(&data);
        let mut cw = clean.clone();
        for k in 0..9 { cw[k * 23] ^= 0x5A ^ k as u8; }
        assert_eq!(rs.decode(&mut cw, &[]), Ok(9));
        assert_eq!(cw, clean);
    }
    #[test]
    fn ten_errors_exceed_the_safe_budget() {
        let rs = RsCodec::new(20);
        let clean = rs.encode(&sample(235));
        let mut cw = clean.clone();
        for k in 0..10 { cw[k * 20 + 1] ^= 0xFF; }
        assert_eq!(rs.decode(&mut cw, &[]), Err(RsError::TooManyErrors));
    }
    #[test]
    fn erasures_double_the_budget() {
        let rs = RsCodec::new(20);
        let clean = rs.encode(&sample(235));
        let mut cw = clean.clone();
        let erased: Vec<usize> = (0..10).map(|k| k * 11 + 2).collect();
        for &i in &erased { cw[i] = 0; }
        for k in 0..4 { cw[200 + k * 9] ^= 0x33; }
        assert_eq!(rs.decode(&mut cw, &erased), Ok(14));
        assert_eq!(cw, clean);
    }
    #[test]
    fn short_codewords_work() {
        let rs = RsCodec::new(14);
        let data = sample(14);
        let mut cw = rs.encode(&data);
        assert_eq!(cw.len(), 28);
        cw[0] ^= 1; cw[5] ^= 2; cw[27] ^= 4;
        assert_eq!(rs.decode(&mut cw, &[]), Ok(3));
        assert_eq!(&cw[..14], &data[..]);
    }
    #[test]
    fn massive_damage_is_rejected_never_miscorrected() {
        let rs = RsCodec::new(20);
        let clean = rs.encode(&sample(235));
        let mut seed = 12345u64;
        let mut miscorrections = 0;
        for trial in 0..300 {
            let mut cw = clean.clone();
            let mut damaged = Vec::new();
            for i in 0..255 { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; if seed % 100 < 60 { cw[i] = if i < 170 { 0 } else { 0x55 }; damaged.push(i); } }
            let erasures: Vec<usize> = damaged.iter().take(10).copied().collect();
            let r = rs.decode(&mut cw, &erasures);
            if r.is_ok() && cw != clean {
                miscorrections += 1;
                let reenc = rs.encode(&cw[..235]);
                let dist_rx = { let mut rx = clean.clone(); for &i in &damaged { rx[i] = if i < 170 { 0 } else { 0x55 }; } rx.iter().zip(cw.iter()).filter(|(a, b)| a != b).count() };
                eprintln!("trial {trial}: Ok({:?}) but wrong; is_codeword={} dist_from_received={} damaged={}", r, reenc == cw, dist_rx, damaged.len());
            }
        }
        assert_eq!(miscorrections, 0);
    }
    /// The decoder's contract over random words (seeded): every parity the formats use and the extremes,
    /// lengths from parity + 1 to 255 (both ends included), errors at random positions and erasures that
    /// cover some of them and some correct symbols. Within the budget (2 errors + erasures + 2 <= parity)
    /// the word comes back clean with the count of positions it fixed; beyond it, an error or a codeword
    /// within that many changes of what was received, never a silent non-codeword; a duplicate erasure is
    /// an error, an erasure past the word a bad length.
    #[test]
    fn decode_contract_on_random_words() {
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = move |m: usize| -> usize { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; (seed % m as u64) as usize };
        // (decoded within budget, beyond budget: rejected, beyond budget: decoded to another codeword)
        let (mut within, mut rejected, mut other) = (0usize, 0usize, 0usize);
        let distinct = |rnd: &mut dyn FnMut(usize) -> usize, k: usize, n: usize, avoid: &[usize]| -> Vec<usize> {
            let mut v: Vec<usize> = Vec::new();
            while v.len() < k { let p = rnd(n); if !v.contains(&p) && !avoid.contains(&p) { v.push(p); } }
            v
        };
        for &p in &[2usize, 3, 12, 24, 32, 40, 64, 127, 128] {
            let rs = RsCodec::new(p);
            for t in 0..400 {
                let n = match t { 0 => p + 1, 1 => 255, _ => p + 1 + rnd(255 - p) };
                let data: Vec<u8> = if t == 2 { vec![0; n - p] } else { (0..n - p).map(|_| rnd(256) as u8).collect() };
                let clean = rs.encode(&data);
                assert_eq!(rs.syndromes(&clean).iter().filter(|&&v| v != 0).count(), 0);
                // within budget: f erasures (some of them on correct symbols), e errors elsewhere
                let f = rnd((p - 2).min(n) + 1);
                let e = rnd((p - 2 - f) / 2 + 1).min(n - f);
                let erased = distinct(&mut rnd, f, n, &[]);
                let wrong = distinct(&mut rnd, e, n, &erased);
                let mut cw = clean.clone();
                for &i in &erased { if rnd(3) > 0 { cw[i] = rnd(256) as u8; } }
                for &i in &wrong { cw[i] ^= 1 + rnd(255) as u8; }
                let fixed = if cw == clean { 0 } else { e + f };
                assert_eq!(rs.decode(&mut cw, &erased), Ok(fixed), "parity {p} n {n}: {e} errors, {f} erasures");
                assert_eq!(cw, clean, "parity {p} n {n}: {e} errors, {f} erasures");
                within += 1;
                // beyond budget: more errors than the parity left after the erasures can locate
                let f = rnd((p - 2).min(n) + 1);
                let lo = (p - 2 - f) / 2 + 1;
                if lo > n - f { continue; }
                let e = lo + rnd(n - f - lo + 1);
                let erased = distinct(&mut rnd, f, n, &[]);
                let wrong = distinct(&mut rnd, e, n, &erased);
                let mut cw = clean.clone();
                for &i in &erased { cw[i] = rnd(256) as u8; }
                for &i in &wrong { cw[i] ^= 1 + rnd(255) as u8; }
                let received = cw.clone();
                match rs.decode(&mut cw, &erased) {
                    Err(_) => rejected += 1,
                    Ok(k) => {
                        assert!(rs.syndromes(&cw).iter().all(|&v| v == 0), "parity {p} n {n}: Ok({k}) is not a codeword");
                        let changed = cw.iter().zip(&received).filter(|(a, b)| a != b).count();
                        assert!(changed <= k, "parity {p} n {n}: Ok({k}) changed {changed} symbols");
                        other += 1;
                    }
                }
                // a duplicate erasure (on a word with errors), an erasure past the word
                if p >= 4 && n >= 2 {
                    let mut cw = clean.clone();
                    let i = rnd(n);
                    cw[i] ^= 0x5A;
                    let j = (i + 1) % n;
                    assert_eq!(rs.decode(&mut cw.clone(), &[i, j, j]), Err(RsError::TooManyErrors), "parity {p} n {n}: duplicate erasure");
                    assert_eq!(rs.decode(&mut cw, &[i, n]), Err(RsError::BadLength), "parity {p} n {n}: erasure past the word");
                }
            }
        }
        // erasures alone on a shortened word fill it back exactly
        let rs = RsCodec::new(24);
        let clean = rs.encode(&sample(40));
        let mut cw = clean.clone();
        let erased: Vec<usize> = (0..22).map(|k| k * 3).collect();
        for &i in &erased { cw[i] = 0; }
        assert_eq!(rs.decode(&mut cw, &erased), Ok(22));
        assert_eq!(cw, clean);
        assert!(within == 9 * 400 && rejected > 1000 && other < rejected, "within {within}, beyond: rejected {rejected}, decoded to another codeword {other}");
    }
    #[test]
    fn rejects_overlong() {
        let rs = RsCodec::new(20);
        let mut cw = vec![0u8; 256];
        assert_eq!(rs.decode(&mut cw, &[]), Err(RsError::BadLength));
    }
}
