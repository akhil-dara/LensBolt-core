//! Finish code: the shortest word code naming the units (segments, or chunks of a one-segment file) the
//! receiver is still missing, typed on the sender so it sends only those.
//!
//! Bits, MSB-first throughout: `[tag: 2][body][check: 6]`, then zero bits to whole words of 10 (word index =
//! value, first word first). The receiver builds every code that applies and shows the one with the fewest
//! words (ties: the lower tag); every code fits 4 words (40 bits).
//!
//! | tag | body |
//! |---|---|
//! | 0 set | gamma(k), then the colex rank of the sorted k-subset in ceil(log2 C(n,k)) bits |
//! | 1 ranges | gamma(r), then per range gamma(gap from the previous range's end + 1), gamma(length) |
//! | 2 map | n bits, bit i = unit i (only when n <= 32) |
//! | 3 window | the first short unit a in b = max(1, ceil(log2 n)) bits, then 32 - b bits for units a+1.. |
//!
//! gamma(v), v >= 1, is Elias gamma: floor(log2 v) zeros, then v in binary. The window code names only the
//! short units within the 32 - b after a (always 4 words); the receiver's next code names the rest.
use crate::crc16::crc16;
use crate::words::{word_index, WordError, WORDS};

#[derive(Debug, PartialEq, Eq)]
pub enum FinishError { Word(WordError), Check, Empty, Malformed }

/// The largest unit count: the manifest's segment limit. The window code still fits (b = 16, w = 16).
const MAX_UNITS: u32 = 65_535;
const MAX_BITS: usize = 40;
const SET: u8 = 0;
const RANGES: u8 = 1;
const MAP: u8 = 2;
const WINDOW: u8 = 3;

/// The words naming `short` (unit indices below `n`, any order) for transfer `transfer_id`; None when
/// nothing is short or everything is (then there is nothing to skip), and when n is outside 1 ..= 65,535
/// or a unit is not below n (never a panic: the caller just shows no words).
pub fn encode(transfer_id: u16, n: u32, short: &[u32]) -> Option<String> {
    if n == 0 || n > MAX_UNITS || short.iter().any(|&u| u >= n) { return None; }
    let mut s = short.to_vec();
    s.sort_unstable();
    s.dedup();
    if s.is_empty() || s.len() == n as usize { return None; }
    let bodies = [set_body(n, &s), ranges_body(&s), map_body(n, &s), Some(window_body(n, &s))];
    let (tag, body) = bodies.into_iter().enumerate().filter_map(|(t, b)| Some((t as u8, b?)))
        .filter(|(_, b)| b.len() + 8 <= MAX_BITS)
        .min_by_key(|(t, b)| ((b.len() + 8).div_ceil(10), *t))?;
    Some(to_words(transfer_id, n, tag, &body))
}

/// The parts `units` (indices from 0, sorted) name, as people number them: from 1, runs joined, the first eight
/// runs, then an ellipsis: "7, 12–15, 30".
pub fn parts_list(units: &[u32]) -> String {
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for &u in units {
        match runs.last_mut() { Some(r) if r.1 + 1 == u => r.1 = u, _ => runs.push((u, u)) }
    }
    let mut list: Vec<String> = runs.iter().take(8).map(|&(a, b)| if a == b { format!("{}", a + 1) } else { format!("{}\u{2013}{}", a + 1, b + 1) }).collect();
    if runs.len() > 8 { list.push("\u{2026}".into()); }
    list.join(", ")
}
/// What the sender says it sends: "3 parts: 7, 12–15, 30".
pub fn parts_text(units: &[u32]) -> String {
    format!("{} part{}: {}", units.len(), if units.len() == 1 { "" } else { "s" }, parts_list(units))
}
/// The receiver's line under its words: "Parts 7, 12–15, 30" ("Part 2" for one), the same numbers the sender shows.
pub fn parts_line(units: &[u32]) -> String {
    format!("Part{} {}", if units.len() == 1 { "" } else { "s" }, parts_list(units))
}

/// Tag, body and check, padded with zero bits to whole words.
fn to_words(transfer_id: u16, n: u32, tag: u8, body: &[bool]) -> String {
    let mut bits = Vec::with_capacity(MAX_BITS);
    put(&mut bits, tag as u128, 2);
    bits.extend_from_slice(body);
    put(&mut bits, check(transfer_id, n, tag, body) as u128, 6);
    bits.resize(bits.len().div_ceil(10) * 10, false);
    let words: Vec<&str> = bits.chunks(10).map(|w| WORDS[w.iter().fold(0, |v, &b| v << 1 | b as usize)]).collect();
    words.join(" ")
}

/// The sorted units `code` names for transfer `transfer_id` of `n` units. Words may be separated by any mix
/// of spaces, dashes and commas, in any case; each is matched by `word_index`.
pub fn decode(transfer_id: u16, n: u32, code: &str) -> Result<Vec<u32>, FinishError> {
    let typed: Vec<&str> = code.split(|c: char| c.is_whitespace() || c == '-' || c == ',').filter(|w| !w.is_empty()).collect();
    if typed.is_empty() { return Err(FinishError::Empty); }
    if typed.len() * 10 > MAX_BITS { return Err(FinishError::Malformed); }
    let mut bits = Vec::with_capacity(MAX_BITS);
    for w in typed { put(&mut bits, word_index(w).map_err(FinishError::Word)? as u128, 10); }
    decode_bits(transfer_id, n, &bits)
}

/// `decode` after the words are read: the code's bits, a whole number of words.
fn decode_bits(transfer_id: u16, n: u32, bits: &[bool]) -> Result<Vec<u32>, FinishError> {
    use FinishError::*;
    if bits.len() > MAX_BITS || n == 0 || n > MAX_UNITS { return Err(Malformed); }
    let mut r = Reader { bits, pos: 0 };
    let tag = r.take(2).ok_or(Malformed)? as u8;
    let units = match tag {
        SET => read_set(&mut r, n),
        RANGES => read_ranges(&mut r, n),
        MAP => if n <= 32 { r.take(n as usize).map(|m| from_map(m, n as usize, 0)) } else { None },
        WINDOW => {
            let b = addr_bits(n);
            r.take(b).zip(r.take(32 - b)).map(|(a, m)| [vec![a as u64], from_map(m, 32 - b, a as u64 + 1)].concat())
        }
        _ => None,
    }.ok_or(Malformed)?;
    let body = &bits[2..r.pos];
    let sum = r.take(6).ok_or(Malformed)? as u8;
    // the check is the next 6 bits, and the rest is padding: zero, and less than one word of it
    if bits.len() - r.pos >= 10 || bits[r.pos..].iter().any(|&b| b) { return Err(Malformed); }
    if sum != check(transfer_id, n, tag, body) { return Err(Check); }
    if units.is_empty() || units.iter().any(|&u| u >= n as u64) { return Err(Malformed); }
    Ok(units.into_iter().map(|u| u as u32).collect())
}

/// The 6-bit check: the low 6 bits of CRC-16 over transfer id (u16 LE), n (u32 LE), tag (u8), the body bits
/// packed MSB-first into bytes (zero bits to a whole byte), and the body's bit length (u32 LE). The length
/// is in because packing alone cannot tell a body from the same body with zero bits appended.
fn check(transfer_id: u16, n: u32, tag: u8, body: &[bool]) -> u8 {
    let mut d = Vec::with_capacity(16);
    d.extend_from_slice(&transfer_id.to_le_bytes());
    d.extend_from_slice(&n.to_le_bytes());
    d.push(tag);
    d.extend(body.chunks(8).map(|c| c.iter().enumerate().fold(0u8, |v, (i, &b)| v | (b as u8) << (7 - i))));
    d.extend_from_slice(&(body.len() as u32).to_le_bytes());
    (crc16(&d) & 0x3f) as u8
}

/// Appends the low `width` bits of `v`, MSB first.
fn put(bits: &mut Vec<bool>, v: u128, width: usize) {
    bits.extend((0..width).rev().map(|i| v >> i & 1 == 1));
}

fn put_gamma(bits: &mut Vec<bool>, v: u64) {
    let w = 64 - v.leading_zeros() as usize;
    bits.extend(std::iter::repeat(false).take(w - 1));
    put(bits, v as u128, w);
}

struct Reader<'a> { bits: &'a [bool], pos: usize }

impl Reader<'_> {
    fn take(&mut self, width: usize) -> Option<u128> {
        let b = self.bits.get(self.pos..self.pos + width)?;
        self.pos += width;
        Some(b.iter().fold(0, |v, &x| v << 1 | x as u128))
    }
    fn gamma(&mut self) -> Option<u64> {
        let z = self.bits[self.pos..].iter().take_while(|&&b| !b).count();
        self.pos += z;
        self.take(z + 1).map(|v| v as u64)
    }
}

/// Bits that hold any value below `c` (0 when `c` <= 1).
fn width_below(c: u128) -> usize { if c <= 1 { 0 } else { 128 - (c - 1).leading_zeros() as usize } }

/// Bits for a unit index: ceil(log2 n), at least 1.
fn addr_bits(n: u32) -> usize { width_below(n as u128).max(1) }

/// C(n, k), or None when it exceeds 2^64 (far beyond any body that fits 4 words).
fn binom(n: u64, k: u64) -> Option<u128> {
    if k > n { return Some(0); }
    let k = k.min(n - k);
    let mut c = 1u128;
    for i in 0..k {
        // c = C(n, i) exactly, so c * (n - i) divides by i + 1; C(n, i) only grows up to i = n / 2, and
        // c <= 2^64 times n - i < 2^64 cannot overflow
        c = c.checked_mul((n - i) as u128)? / (i + 1) as u128;
        if c > 1 << 64 { return None; }
    }
    Some(c)
}

/// Tag 0: gamma(k), then the colex rank (the sum of C(s_i, i + 1) over the sorted subset), below C(n, k).
fn set_body(n: u32, s: &[u32]) -> Option<Vec<bool>> {
    let k = s.len() as u64;
    let c = binom(n as u64, k)?;
    let mut b = Vec::new();
    put_gamma(&mut b, k);
    if b.len() + width_below(c) + 8 > MAX_BITS { return None; }
    // every term is below the rank, which is below C(n, k): none saturates
    let rank: u128 = s.iter().enumerate().map(|(i, &u)| binom(u as u64, i as u64 + 1).unwrap()).sum();
    put(&mut b, rank, width_below(c));
    Some(b)
}

fn read_set(r: &mut Reader, n: u32) -> Option<Vec<u64>> {
    let k = r.gamma()?;
    if k > n as u64 { return None; }
    let c = binom(n as u64, k)?;
    let mut rank = r.take(width_below(c))?;
    if rank >= c { return None; }
    // unrank: for i = k down to 1, the largest unit u below the last with C(u, i) <= rank. C(u, i) grows
    // with u from C(i - 1, i) = 0 (a saturated C counts as too large), so a binary search finds it.
    let mut v = vec![0u64; k as usize];
    let mut hi = n as u64;
    for i in (1..=k).rev() {
        let (mut lo, mut top) = (i - 1, hi - 1);
        while lo < top {
            let mid = (lo + top + 1) / 2;
            if binom(mid, i).is_some_and(|x| x <= rank) { lo = mid } else { top = mid - 1 }
        }
        rank -= binom(lo, i).unwrap();
        v[i as usize - 1] = lo;
        hi = lo;
    }
    Some(v)
}

/// Tag 1: the maximal runs of consecutive units; None once it is past 4 words.
fn ranges_body(s: &[u32]) -> Option<Vec<bool>> {
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for &u in s {
        match runs.last_mut() { Some((a, l)) if *a + *l == u => *l += 1, _ => runs.push((u, 1)) }
    }
    let mut b = Vec::new();
    put_gamma(&mut b, runs.len() as u64);
    let mut end = 0;
    for (a, l) in runs {
        put_gamma(&mut b, (a - end) as u64 + 1);
        put_gamma(&mut b, l as u64);
        end = a + l;
        if b.len() + 8 > MAX_BITS { return None; }
    }
    Some(b)
}

fn read_ranges(r: &mut Reader, n: u32) -> Option<Vec<u64>> {
    let runs = r.gamma()?;
    let (mut v, mut end) = (Vec::new(), 0u64);
    for _ in 0..runs {
        let a = end + r.gamma()? - 1;
        end = a + r.gamma()?;
        if end > n as u64 { return None; }
        v.extend(a..end);
    }
    Some(v)
}

/// Tag 2: one bit per unit, only for files of up to 32 units.
fn map_body(n: u32, s: &[u32]) -> Option<Vec<bool>> {
    if n > 32 { return None; }
    let mut b = vec![false; n as usize];
    for &u in s { b[u as usize] = true; }
    Some(b)
}

/// The units `first + i` for the set bits i of the `width`-bit map `m` (MSB = i = 0).
fn from_map(m: u128, width: usize, first: u64) -> Vec<u64> {
    (0..width).filter(|&i| m >> (width - 1 - i) & 1 == 1).map(|i| first + i as u64).collect()
}

/// Tag 3: the first short unit a, then a map of the 32 - b units after it (bits beyond n stay zero).
fn window_body(n: u32, s: &[u32]) -> Vec<bool> {
    let (b, a) = (addr_bits(n), s[0]);
    let mut v = Vec::with_capacity(32);
    put(&mut v, a as u128, b);
    v.extend((a + 1..=a + (32 - b) as u32).map(|u| s.binary_search(&u).is_ok()));
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    const ID: u16 = 0x4c5b;
    const NS: [u32; 7] = [1, 2, 5, 16, 32, 320, 2000];

    fn words(code: &str) -> usize { code.split(' ').count() }
    /// The tag: the top 2 bits of the first word.
    fn tag_of(code: &str) -> u8 { (word_index(code.split(' ').next().unwrap()).unwrap() >> 8) as u8 }
    /// Exactly what the code for `s` must name: all of `s`, or for a window code the units of `s` in
    /// a ..= a + w, a = s[0], w = 32 - max(1, ceil(log2 n)) (worked out apart from `addr_bits`).
    fn expected(n: u32, s: &[u32], code: &str) -> Vec<u32> {
        if tag_of(code) != WINDOW { return s.to_vec(); }
        let b = (32 - n.saturating_sub(1).leading_zeros()).max(1);
        s.iter().copied().filter(|&u| u <= s[0] + 32 - b).collect()
    }
    /// Encodes `s` (sorted), decodes it, and checks the decode names exactly `expected`.
    fn round_trip(n: u32, s: &[u32]) -> String {
        let code = encode(ID, n, s).unwrap_or_else(|| panic!("no code for n={n} {s:?}"));
        let got = decode(ID, n, &code).unwrap_or_else(|e| panic!("n={n} {s:?} {code}: {e:?}"));
        assert_eq!(got, expected(n, s, &code), "n={n} {s:?} {code}");
        code
    }
    /// The fewest words an exact code (set, ranges, map) needs, worked out apart from the encoder; None
    /// when none fits 40 bits and only a window code is left.
    fn exact_words(n: u32, s: &[u32]) -> Option<usize> {
        let gamma = |v: u64| 2 * (64 - v.leading_zeros() as usize) - 1;
        let (n64, k) = (n as u64, s.len() as u64);
        let mut c = Some(1u128);
        for i in 0..k.min(n64 - k) { c = c.and_then(|c| c.checked_mul((n64 - i) as u128)).map(|c| c / (i as u128 + 1)); }
        let set = c.map(|c| 8 + gamma(k) + if c <= 1 { 0 } else { 128 - (c - 1).leading_zeros() as usize });
        let mut runs: Vec<(u32, u32)> = Vec::new();
        for &u in s { match runs.last_mut() { Some(r) if r.0 + r.1 == u => r.1 += 1, _ => runs.push((u, 1)) } }
        let (mut ranges, mut end) = (8 + gamma(runs.len() as u64), 0);
        for &(a, l) in &runs { ranges += gamma((a - end) as u64 + 1) + gamma(l as u64); end = a + l; }
        let map = (n <= 32).then_some(8 + n as usize);
        [set, Some(ranges), map].into_iter().flatten().filter(|&b| b <= 40).min().map(|b| b.div_ceil(10))
    }
    /// An Ok decode is non-empty, sorted, duplicate-free and below n.
    fn in_range(n: u32, v: &[u32]) -> bool {
        !v.is_empty() && v.windows(2).all(|w| w[0] < w[1]) && v.iter().all(|&u| u < n)
    }
    /// Bits from a string of 0s and 1s.
    fn bits(s: &str) -> Vec<bool> { s.bytes().map(|c| c == b'1').collect() }
    /// The test patterns for `n` that name some but not all units: one unit, three scattered, a burst of 7,
    /// 17 scattered (n >= 32), every other unit (n <= 32).
    fn patterns(n: u32) -> Vec<Vec<u32>> {
        let mut v = vec![vec![n / 2]];
        if n >= 3 { v.push(vec![0, n / 2, n - 1]); }
        if n >= 8 { v.push((n / 3..n / 3 + 7).collect()); }
        if n >= 32 { v.push((0..17).map(|i| i * n / 17).collect()); }
        if n <= 32 { v.push((0..n).step_by(2).collect()); }
        v.retain(|s| s.len() < n as usize);
        v
    }
    fn rng(mut seed: u64) -> impl FnMut(u32) -> u32 {
        move |n| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; (seed % n as u64) as u32 }
    }

    #[test]
    fn parts_are_named_from_one_with_runs_joined() {
        assert_eq!(parts_list(&[6, 11, 12, 13, 14, 29]), "7, 12\u{2013}15, 30");
        assert_eq!(parts_text(&[6, 11, 12, 13, 14, 29]), "6 parts: 7, 12\u{2013}15, 30");
        assert_eq!(parts_text(&[1]), "1 part: 2");
        assert_eq!(parts_line(&[6, 11, 12, 13, 14, 29]), "Parts 7, 12\u{2013}15, 30");
        assert_eq!(parts_line(&[1]), "Part 2");
        assert_eq!(parts_line(&[1, 2]), "Parts 2\u{2013}3");
        // the first eight runs, then an ellipsis
        let every_other: Vec<u32> = (0..20).map(|i| i * 2).collect();
        assert_eq!(parts_list(&every_other), "1, 3, 5, 7, 9, 11, 13, 15, \u{2026}");
    }
    #[test]
    fn round_trips() {
        for n in NS { for s in patterns(n) { round_trip(n, &s); } }
        assert_eq!(decode(ID, 2, &round_trip(2, &[1])), Ok(vec![1]));
        assert_eq!(decode(ID, 2, &round_trip(2, &[0])), Ok(vec![0]));
        // any order and repeats in, sorted units out
        assert_eq!(decode(ID, 320, &encode(ID, 320, &[100, 5, 7, 5, 6]).unwrap()), Ok(vec![5, 6, 7, 100]));
    }

    #[test]
    fn large_files_use_window_codes() {
        // up to the manifest's 65,535 segments: b = 16, w = 16 at the top
        for n in [5000, 65_535] {
            for s in [vec![7], vec![10, 20, 25], vec![0, 100, n - 1], (0..17).map(|i| i * n / 17).collect(), vec![n - 3, n - 1]] {
                assert!(words(&round_trip(n, &s)) <= 4, "n={n} {s:?}");
            }
        }
        assert_eq!(tag_of(&encode(ID, 65_535, &[0, 100, 65_534]).unwrap()), WINDOW);
        assert_eq!(decode(ID, 65_535, &encode(ID, 65_535, &[40_000, 40_016, 40_017]).unwrap()), Ok(vec![40_000, 40_016]));
        // never a panic: no units, a unit beyond n, or n outside 1 ..= 65,535 give no words
        assert_eq!(encode(ID, 5000, &[5000]), None);
        assert_eq!(encode(ID, 5000, &[3, 9000]), None);
        assert_eq!(encode(ID, 70_000, &[1]), None);
        assert_eq!(encode(ID, 0, &[0]), None);
        assert_eq!(encode(ID, 0, &[]), None);
    }

    #[test]
    fn lengths_for_32_segments() {
        let len = |s: Vec<u32>| words(&encode(ID, 32, &s).unwrap());
        assert_eq!(len(vec![9]), 2);
        assert_eq!(len(vec![4, 21]), 2);
        assert_eq!(len(vec![3, 14, 27]), 3);
        assert_eq!(len((0..17).map(|i| i * 32 / 17).collect()), 4);
        assert!(len((0..6).chain(12..17).chain(25..31).collect()) <= 4);
    }

    #[test]
    fn never_more_than_four_words() {
        let mut next = rng(0x9e37_79b9_7f4a_7c15);
        for n in NS {
            for t in 0..200 {
                // scattered picks, a random density, or a few bursts
                let mut m = vec![false; n as usize];
                match t % 3 {
                    0 => for _ in 0..1 + next(24) { m[next(n) as usize] = true; },
                    1 => { let p = 1 + next(99); for x in m.iter_mut() { *x = next(100) < p; } }
                    _ => for _ in 0..1 + next(4) {
                        let a = next(n); for u in a..(a + 1 + next(n / 4 + 1)).min(n) { m[u as usize] = true; }
                    },
                }
                let s: Vec<u32> = (0..n).filter(|&u| m[u as usize]).collect();
                if s.is_empty() || s.len() == n as usize { assert_eq!(encode(ID, n, &s), None); continue; }
                let code = round_trip(n, &s);
                // the shortest exact code when one fits 4 words (a tie goes to it, the lower tag), else a window
                let exact = exact_words(n, &s);
                assert_eq!(words(&code), exact.unwrap_or(4), "n={n} {s:?} {code}");
                assert_eq!(tag_of(&code) == WINDOW, exact.is_none(), "n={n} {s:?} {code}");
            }
        }
    }

    #[test]
    fn window_codes_make_progress() {
        let n = 320;
        let mut left: Vec<u32> = (0..17).map(|i| i * 19 + (i * i) % 7).collect();
        assert!(encode(ID, n, &left).map(|c| words(&c)) == Some(4));
        for _ in 0..17 {
            if left.is_empty() { break; }
            let code = round_trip(n, &left);
            let got = decode(ID, n, &code).unwrap();
            assert_eq!(got[0], left[0], "{code}");
            left.retain(|u| !got.contains(u));
        }
        assert!(left.is_empty(), "left after 17 codes: {left:?}");
    }

    #[test]
    fn other_transfer_is_refused() {
        let code = encode(ID, 32, &[3, 14, 27]).unwrap();
        assert_eq!(decode(ID, 32, &code), Ok(vec![3, 14, 27]));
        assert_eq!(decode(ID + 2, 32, &code), Err(FinishError::Check));
    }

    #[test]
    fn empty_and_all_short_give_no_words() {
        assert_eq!(encode(ID, 32, &[]), None);
        assert_eq!(encode(ID, 32, &(0..32).collect::<Vec<_>>()), None);
        assert_eq!(encode(ID, 1, &[0]), None);
    }

    #[test]
    fn typing_is_forgiving_and_strict() {
        let s = [3, 14, 27];
        let code = encode(ID, 32, &s).unwrap();
        let loose = format!("  {}  ", code.to_uppercase().replace(' ', " -, "));
        assert_eq!(decode(ID, 32, &loose), Ok(s.to_vec()));
        assert_eq!(decode(ID, 32, ""), Err(FinishError::Empty));
        assert_eq!(decode(ID, 32, " -, "), Err(FinishError::Empty));
        assert!(matches!(decode(ID, 32, &format!("{code} zzzzzz")), Err(FinishError::Word(_))));
        // a trailing all-zero word is padding the code does not need: refused, not ignored
        assert_eq!(decode(ID, 32, &format!("{code} {}", crate::words::WORDS[0])), Err(FinishError::Malformed));
        assert_eq!(decode(ID, 70_000, &code), Err(FinishError::Malformed));
        assert_eq!(decode(ID, 0, &code), Err(FinishError::Malformed));
        assert_eq!(decode(ID, 32, &format!("{code} {code}")), Err(FinishError::Malformed));
    }

    #[test]
    fn decode_never_panics_and_stays_in_range() {
        // every 1- and 2-word code, as bits (the words themselves are word_index's)
        let mut ok = 0;
        for n in [1, 2, 3, 32, 33, 4096, 65_535] {
            for x in 0..1u32 << 20 {
                let b: Vec<bool> = (0..20).rev().map(|i| x >> i & 1 == 1).collect();
                for b in [&b[..], &b[10..]] {
                    if let Ok(v) = decode_bits(ID, n, b) { ok += 1; assert!(in_range(n, &v), "n={n} {x:#x} {v:?}"); }
                }
            }
        }
        assert!(ok > 0);
        // 1-4 typed words, biased to word 0 (long runs of zeros: huge gammas) and low words
        let mut next = rng(0x2545_f491_4f6c_dd1d);
        for n in [1, 2, 5, 32, 320, 2000, 65_535] {
            for _ in 0..20_000 {
                let t: Vec<&str> = (0..1 + next(4)).map(|_| WORDS[match next(4) { 0 => 0, 1 => next(8), _ => next(1024) } as usize]).collect();
                if let Ok(v) = decode(ID, n, &t.join(" ")) { assert!(in_range(n, &v), "n={n} {t:?} {v:?}"); }
            }
        }
        // codes with a valid check that name what cannot be: each refused as Malformed, beside a valid neighbour
        let m = Err(FinishError::Malformed);
        let at = |n: u32, tag: u8, body: &str| decode(ID, n, &to_words(ID, n, tag, &bits(body)));
        assert_eq!(at(5, SET, "0111001"), Ok(vec![2, 3, 4]), "rank 9 = C(5, 3) - 1: the last subset");
        assert_eq!(at(5, SET, "0111010"), m, "rank 10 = C(5, 3)");
        assert_eq!(at(5, SET, "0111111"), m, "rank 15");
        assert_eq!(at(2, SET, "011"), m, "k = 3 > n = 2");
        assert_eq!(at(5, RANGES, "100100010"), Ok(vec![3, 4]), "start 3, length 2");
        assert_eq!(at(5, RANGES, "100100011"), m, "start 3, length 3: past n");
        assert_eq!(at(33, MAP, "1111111111"), m, "map with n > 32");
        assert_eq!(at(5, MAP, "00000"), m, "map naming nothing");
        assert_eq!(at(5, MAP, "10001"), Ok(vec![0, 4]));
        let window = |a: &str, map: &str| format!("{a}{map:0<29}");
        assert_eq!(at(5, WINDOW, &window("100", "")), Ok(vec![4]));
        assert_eq!(at(5, WINDOW, &window("101", "")), m, "window a = n");
        assert_eq!(at(5, WINDOW, &window("111", "")), m, "window a > n");
        assert_eq!(at(5, WINDOW, &window("011", "1")), Ok(vec![3, 4]));
        assert_eq!(at(5, WINDOW, &window("011", "01")), m, "window bit beyond n");
        assert_eq!(at(65_535, WINDOW, &format!("{:016b}{:016b}", 65_533, 0x8000)), Ok(vec![65_533, 65_534]));
        assert_eq!(at(65_535, WINDOW, &format!("{:016b}{:016b}", 65_533, 0x4000)), m, "unit 65,535");
    }

    #[test]
    fn check_refuses_63_in_64() {
        // every other transfer id, and every one-word change to the code: about 1 in 64 passes the check
        let code = encode(ID, 32, &[3, 14, 27]).unwrap();
        let ids = (0..=u16::MAX).filter(|&i| i != ID && decode(i, 32, &code).is_ok()).count();
        assert!((65535 / 128..65535 / 32).contains(&ids), "{ids} other ids pass");
        let w: Vec<&str> = code.split(' ').collect();
        let (mut tried, mut passed) = (0, 0);
        for i in 0..w.len() {
            for x in crate::words::WORDS.iter().filter(|&&x| x != w[i]) {
                let mut t = w.clone();
                t[i] = x;
                tried += 1;
                passed += decode(ID, 32, &t.join(" ")).is_ok() as usize;
            }
        }
        assert!(passed * 32 < tried, "{passed} of {tried} changed codes pass");
    }
}
