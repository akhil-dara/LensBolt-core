use super::prng::Xorshift64;

pub const REPAIR_DEGREE_MIN: usize = 32;
pub const REPAIR_DEGREE_MAX: usize = 96;
/// Chunks of up to this many blocks: a repair packet takes each block with probability 1/2 (`neighbors`).
/// Larger chunks keep the 32..=96 band, whose degree (64 on average) is what a 128-block chunk's coin flips
/// give: the receiver's cost per packet never exceeds it.
pub const COIN_FLIP_MAX_K: usize = 128;

/// Degree of a repair packet of a chunk of more than `COIN_FLIP_MAX_K` blocks: 32..=96. With a GF(2)
/// eliminator on the receiver there is no need for peeling-friendly low degrees, and a high degree means a
/// packet is almost never useless even when most blocks are already known (a degree-2 soliton packet is
/// wasted with probability p^2 at known-fraction p; 0.9^32 ~ 3%). Not the degree of a smaller chunk's packets
/// (`neighbors` draws those by coin flips), so only `neighbors` calls it.
pub(crate) fn degree(rng: &mut Xorshift64, k: usize) -> usize {
    debug_assert!(k <= 1 || k > COIN_FLIP_MAX_K, "degree() is the band of chunks over {COIN_FLIP_MAX_K} blocks, not of {k}");
    if k <= 1 { return 1; }
    let hi = REPAIR_DEGREE_MAX.min(k);
    let lo = REPAIR_DEGREE_MIN.min(hi);
    lo + rng.below((hi - lo + 1) as u32) as usize
}

/// splitmix64's finaliser.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Source blocks XORed into packet `pid` of a chunk with `k` blocks. pid < k => [pid] (systematic).
///
/// Repair packets of a chunk of up to `COIN_FLIP_MAX_K` blocks (format 9): each block with probability 1/2,
/// drawn again while none is taken, so a uniformly random nonempty subset. Random GF(2) rows of density 1/2
/// are the best conditioned: whatever blocks are already held, the rows restricted to the missing ones are
/// uniform, and the eliminator needs ~1.6 packets beyond the missing count for every k (measured for every
/// k in 2..=1024: docs/2026-10-30-repair-schedule.md). Format 8 drew 32..=k of k blocks for k = 33..44, rows
/// nearly all ones and so nearly alike: 130 extra packets at k = 33. The seed is mixed first: the
/// generator's state is a GF(2)-linear function of its seed, and seeds that differ in the pid bits only gave
/// rows with linear structure (measured unmixed: 4-7 extra on average at k = 127, p99 85-154).
pub fn neighbors(pid: u32, k: usize) -> Vec<u32> {
    if (pid as usize) < k { return vec![pid]; }
    let seed = ((pid as u64) << 32) ^ (k as u64) ^ 0xC0DE_F0F0_1234_5678;
    if k <= COIN_FLIP_MAX_K {
        let mut rng = Xorshift64::new(mix(seed));
        loop {
            let mut out: Vec<u32> = Vec::with_capacity(k / 2 + 8);
            for base in (0..k).step_by(32) {
                let w = rng.next_u64() >> 32; // the generator's high bits
                for b in 0..32.min(k - base) { if w >> b & 1 == 1 { out.push((base + b) as u32); } }
            }
            if !out.is_empty() { return out; }
        }
    }
    let mut rng = Xorshift64::new(seed);
    let d = degree(&mut rng, k);
    let mut out: Vec<u32> = Vec::with_capacity(d);
    while out.len() < d {
        let i = rng.below(k as u32);
        if !out.contains(&i) { out.push(i); }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn systematic_packets_map_to_themselves() {
        for pid in 0..50u32 { assert_eq!(neighbors(pid, 50), vec![pid]); }
    }
    #[test]
    fn repair_packets_are_deterministic_distinct_and_in_range() {
        let a = neighbors(1234, 1000);
        assert_eq!(a, neighbors(1234, 1000));
        let mut s = a.clone(); s.sort(); s.dedup();
        assert_eq!(s.len(), a.len());
        assert!(a.iter().all(|&i| (i as usize) < 1000));
        assert!(!a.is_empty());
    }
    #[test]
    fn degree_is_dense_and_in_range() {
        let mut rng = Xorshift64::new(42);
        for _ in 0..10_000 { let d = degree(&mut rng, 1024); assert!((32..=96).contains(&d)); }
        assert_eq!(degree(&mut rng, 1), 1);
        // small chunks: different pids give different subsets
        let a = neighbors(10, 10); let b = neighbors(11, 10);
        assert_ne!(a, b);
    }

    /// Repair packets (from a random place in the repair ids) until the unknown blocks are determined: GF(2)
    /// rank of the packets' block sets restricted to the unknown blocks, which is when the eliminator solves.
    /// Returns the packets used beyond the number of unknown blocks.
    fn extra_packets(k: usize, known: &[bool], start: u32) -> usize {
        let need = known.iter().filter(|&&b| !b).count();
        let words = k.div_ceil(64);
        let mut basis: Vec<Option<[u64; 16]>> = vec![None; k];
        let (mut rank, mut used) = (0, 0usize);
        while rank < need {
            let pid = k as u32 + (start + used as u32) % (4096 - k as u32);
            used += 1;
            let mut row = [0u64; 16];
            for i in neighbors(pid, k) { let i = i as usize; if !known[i] { row[i / 64] ^= 1 << (i % 64); } }
            while let Some(c) = (0..words).find(|&w| row[w] != 0).map(|w| w * 64 + row[w].trailing_zeros() as usize) {
                match &basis[c] { None => { basis[c] = Some(row); rank += 1; break; } Some(b) => for w in 0..words { row[w] ^= b[w]; } }
            }
            assert!(used < 4 * k + 400, "k {k}: {used} repair packets and still short");
        }
        used - need
    }

    #[test]
    fn every_small_chunk_decodes_with_a_few_extra_packets() {
        // chunks of 2..=200 blocks (a file's short last chunk, a small file), none, 20% or half of their blocks
        // already held: mean extra packets <= 3 over 60 trials and none above 20 for every k, and over all of them
        // at most 1.5% needing 8 or more (random dense rows: P(extra >= e) ~ 2^-e). The 2026-10-30 rule (32..k of
        // k blocks for k = 33..44, nearly all of them) needed 130 extra on average at k = 33
        let mut s = 0x6f6f_u64;
        let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let (mut tail, mut trials) = (0, 0);
        for k in 2..=200usize {
            for held in [0.0, 0.2, 0.5] {
                let extra: Vec<usize> = (0..60).map(|_| {
                    let mut known = vec![false; k];
                    let mut idx: Vec<usize> = (0..k).collect();
                    let n = (held * k as f64).round() as usize;
                    for i in 0..n { let j = i + (next() % (k - i) as u64) as usize; idx.swap(i, j); known[idx[i]] = true; }
                    extra_packets(k, &known, (next() % (4096 - k as u64)) as u32)
                }).collect();
                let mean = extra.iter().sum::<usize>() as f64 / extra.len() as f64;
                let max = *extra.iter().max().unwrap();
                assert!(mean <= 3.0 && max <= 20, "k {k}, {:.0}% held: mean extra {mean:.2}, max {max}", held * 100.0);
                tail += extra.iter().filter(|&&e| e >= 8).count();
                trials += extra.len();
            }
        }
        assert!(tail as f64 <= 0.015 * trials as f64, "{tail} of {trials} trials needed 8 or more extra packets");
        eprintln!("{tail} of {trials} trials needed 8 or more extra packets");
    }

    #[test]
    fn the_decoder_completes_small_chunks_with_a_few_extra_packets() {
        // the real decoder (peeling + eliminator) on the sizes the old rule served worst, half the blocks held
        use crate::fountain::{LtDecoder, LtEncoder, BLOCK_BYTES};
        for k in [33usize, 34, 36, 40, 44] {
            let mut total = 0;
            for t in 0..12u64 {
                let mut s = 0x7a7a_0000 ^ (k as u64) << 8 ^ t;
                let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
                let data: Vec<u8> = (0..k * BLOCK_BYTES - 3).map(|_| next() as u8).collect();
                let enc = LtEncoder::new(&data);
                let mut dec = LtDecoder::new(k, data.len());
                for pid in (0..k as u32).filter(|p| p % 2 == 0) { dec.add(pid, &enc.packet(pid)); }
                let need = k - dec.solved_blocks();
                let (mut pid, mut used) = (k as u32 + (next() % 3000) as u32, 0);
                while !dec.add(pid, &enc.packet(pid)) { pid += 1; used += 1; assert!(used < 400, "k {k}: 400 repair packets"); }
                total += used + 1 - need;
                assert_eq!(dec.take(), Some(data));
            }
            assert!(total as f64 / 12.0 <= 3.0, "k {k}: mean extra {:.2}", total as f64 / 12.0);
        }
    }

    #[test]
    fn small_chunks_take_each_block_with_probability_one_half() {
        // up to COIN_FLIP_MAX_K blocks: degree k/2 on average (random dense rows: the best-conditioned over GF(2));
        // above it the 32..96 band (large chunks: the eliminator's cost grows with the degree)
        for k in [2usize, 7, 33, 64, 128] {
            let n = 4000u32;
            let mut per = vec![0u32; k];
            let mut deg = 0usize;
            for pid in k as u32..k as u32 + n {
                let nb = neighbors(pid, k);
                assert!(!nb.is_empty());
                let mut s = nb.clone(); s.sort_unstable(); s.dedup();
                assert_eq!(s.len(), nb.len(), "k {k}: repeated block");
                for &i in &nb { per[i as usize] += 1; }
                deg += nb.len();
            }
            // nonempty subsets uniformly: mean degree k/2 / (1 - 2^-k)
            let want = k as f64 / 2.0 / (1.0 - 0.5f64.powi(k as i32));
            assert!((deg as f64 / n as f64 - want).abs() < 0.05 * want + 0.05, "k {k}: mean degree {}", deg as f64 / n as f64);
            let p = 0.5 / (1.0 - 0.5f64.powi(k as i32));
            for (i, &c) in per.iter().enumerate() { assert!((c as f64 / n as f64 - p).abs() < 0.05, "k {k}: block {i} in {c} of {n}"); }
        }
        for k in [129usize, 500, 1024] { for pid in k as u32..k as u32 + 500 { assert!((32..=96).contains(&neighbors(pid, k).len())); } }
    }

    #[test]
    fn format_9_neighbour_sets_are_pinned() {
        // the neighbour sets are the wire format: a sender and a receiver of one FLAT_VERSION must draw the same
        // blocks for every packet. FNV-1a over (k, pid, the blocks in order) for every k in 1..=200 (both degree
        // rules, the coin flips up to COIN_FLIP_MAX_K and the 32..=96 band above) and a spread of pids: systematic
        // ones, the first repair ids, a stride through the rest and the last id, 4095
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut eat = |v: u32| for b in v.to_le_bytes() { h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3); };
        let mut sets = 0;
        for k in 1..=200usize {
            let mut pids: Vec<u32> = vec![0, k as u32 / 2, k as u32 - 1];
            pids.extend(k as u32..k as u32 + 40);
            pids.extend((k as u32 + 40..4095).step_by(61));
            pids.push(4095);
            for pid in pids {
                let nb = neighbors(pid, k);
                eat(k as u32); eat(pid); eat(nb.len() as u32);
                for i in nb { eat(i); }
                sets += 1;
            }
        }
        assert_eq!(sets, 21_867);
        assert_eq!(h, 0x6b55_e4ae_7d2b_57cd, "changing the neighbour sets changes the wire format: bump FLAT_VERSION (got {h:#018x})");
    }
}
