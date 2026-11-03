use super::encoder::xor_into;
use super::{neighbors, BLOCK_BYTES};
use std::collections::HashSet;

const WORDS: usize = 16; // 1024 bits

/// Where solved blocks live. The in-memory store is for small/short-lived chunks (the manifest);
/// a file-backed store keeps a 1 GB transfer's receiver memory at "pending repair equations" only.
pub trait BlockStore {
    fn read(&mut self, i: usize) -> [u8; BLOCK_BYTES];
    fn write(&mut self, i: usize, data: &[u8; BLOCK_BYTES]);
}

pub struct MemStore { blocks: Vec<Option<Box<[u8; BLOCK_BYTES]>>> }
impl MemStore {
    pub fn new(k: usize) -> MemStore { MemStore { blocks: vec![None; k] } }
}
impl BlockStore for MemStore {
    fn read(&mut self, i: usize) -> [u8; BLOCK_BYTES] { **self.blocks[i].as_ref().expect("block not solved") }
    fn write(&mut self, i: usize, data: &[u8; BLOCK_BYTES]) { self.blocks[i] = Some(Box::new(*data)); }
}

#[derive(Clone)]
struct Equation { bits: [u64; WORDS], data: [u8; BLOCK_BYTES], count: u32 }
impl Equation {
    fn has(&self, i: usize) -> bool { self.bits[i / 64] >> (i % 64) & 1 == 1 }
    fn clear(&mut self, i: usize) { self.bits[i / 64] &= !(1u64 << (i % 64)); self.count -= 1; }
    fn first(&self) -> Option<usize> { for (w, &b) in self.bits.iter().enumerate() { if b != 0 { return Some(w * 64 + b.trailing_zeros() as usize); } } None }
}

/// Fountain decoder for one chunk. Memory = pending equations (each ~1 KB) + a known-block bitmap;
/// solved block data is handed to the `BlockStore` immediately and never retained here.
pub struct LtDecoder {
    k: usize, data_len: usize,
    known: Vec<bool>, n_solved: usize,
    pending: Vec<Equation>, seen: HashSet<u32>, received: usize,
    mem: Option<MemStore>,
}

impl LtDecoder {
    /// Decoder with its own in-memory block store (use `add`/`take`).
    pub fn new(k: usize, data_len: usize) -> LtDecoder {
        let mut d = LtDecoder::new_external(k, data_len);
        d.mem = Some(MemStore::new(k));
        d
    }
    /// Decoder whose solved blocks go to a caller-provided store (use `add_with`).
    pub fn new_external(k: usize, data_len: usize) -> LtDecoder {
        assert!(k >= 1 && k <= 1024 && data_len <= k * BLOCK_BYTES);
        LtDecoder { k, data_len, known: vec![false; k], n_solved: 0, pending: Vec::new(), seen: HashSet::new(), received: 0, mem: None }
    }
    pub fn k(&self) -> usize { self.k }
    pub fn is_done(&self) -> bool { self.n_solved == self.k }
    pub fn solved_blocks(&self) -> usize { self.n_solved }
    /// Block `i` is solved (its bytes are in the store).
    pub fn is_solved(&self, i: usize) -> bool { self.known.get(i) == Some(&true) }
    pub fn received(&self) -> usize { self.received }
    /// Bytes held in memory by this decoder (pending equations + bookkeeping), excluding an internal MemStore.
    pub fn memory_bytes(&self) -> usize { self.pending.len() * std::mem::size_of::<Equation>() + self.k + self.seen.len() * 8 }

    /// Distinct packet ids received so far.
    pub fn seen_count(&self) -> usize { self.seen.len() }
    /// In-memory variant. Returns true once the chunk is fully decoded. Duplicate pids are ignored.
    pub fn add(&mut self, pid: u32, payload: &[u8; BLOCK_BYTES]) -> bool {
        let mut mem = self.mem.take().expect("use add_with on an external decoder");
        let r = self.add_with(pid, payload, &mut mem);
        self.mem = Some(mem);
        r
    }

    /// Feed one packet; solved blocks are written to `store`. Returns true once the chunk is complete.
    pub fn add_with(&mut self, pid: u32, payload: &[u8; BLOCK_BYTES], store: &mut dyn BlockStore) -> bool {
        if self.is_done() || !self.seen.insert(pid) { return self.is_done(); }
        self.received += 1;
        let mut eq = Equation { bits: [0; WORDS], data: *payload, count: 0 };
        for i in neighbors(pid, self.k) {
            let i = i as usize;
            if self.known[i] { let b = store.read(i); xor_into(&mut eq.data, &b); } else { eq.bits[i / 64] |= 1 << (i % 64); eq.count += 1; }
        }
        self.insert(eq, store);
        if !self.is_done() && self.received >= self.k && self.pending.len() >= self.k - self.n_solved { self.eliminate(store); }
        self.is_done()
    }

    fn insert(&mut self, eq: Equation, store: &mut dyn BlockStore) {
        if eq.count == 0 { return; }
        if eq.count == 1 { let i = eq.first().unwrap(); self.solve(i, eq.data, store); return; }
        self.pending.push(eq);
    }

    /// Record block i (to the store) and peel it out of every pending equation (iteratively).
    fn solve(&mut self, i: usize, data: [u8; BLOCK_BYTES], store: &mut dyn BlockStore) {
        let mut queue = vec![(i, data)];
        while let Some((i, data)) = queue.pop() {
            if self.known[i] { continue; }
            self.known[i] = true; self.n_solved += 1;
            store.write(i, &data);
            let mut j = 0;
            while j < self.pending.len() {
                if self.pending[j].has(i) {
                    xor_into(&mut self.pending[j].data, &data);
                    self.pending[j].clear(i);
                    if self.pending[j].count <= 1 {
                        let eq = self.pending.swap_remove(j);
                        if eq.count == 1 { queue.push((eq.first().unwrap(), eq.data)); }
                        continue;
                    }
                }
                j += 1;
            }
        }
    }

    /// Gaussian elimination over GF(2) on the pending equations; solves whatever becomes determined.
    fn eliminate(&mut self, store: &mut dyn BlockStore) {
        let mut rows = std::mem::take(&mut self.pending);
        let mut r = 0;
        for col in 0..self.k {
            if self.known[col] { continue; }
            let Some(p) = (r..rows.len()).find(|&i| rows[i].has(col)) else { continue };
            rows.swap(r, p);
            let pivot = rows[r].clone();
            for i in 0..rows.len() {
                if i != r && rows[i].has(col) {
                    for w in 0..WORDS { rows[i].bits[w] ^= pivot.bits[w]; }
                    rows[i].count = rows[i].bits.iter().map(|b| b.count_ones()).sum();
                    xor_into(&mut rows[i].data, &pivot.data);
                }
            }
            r += 1;
        }
        let mut solved_now = Vec::new();
        for eq in rows.into_iter() {
            if eq.count == 1 { solved_now.push((eq.first().unwrap(), eq.data)); } else if eq.count > 1 { self.pending.push(eq); }
        }
        for (i, d) in solved_now { self.solve(i, d, store); }
    }

    /// In-memory variant: the decoded bytes once done.
    pub fn take(self) -> Option<Vec<u8>> {
        if !self.is_done() { return None; }
        let mem = self.mem.expect("take() needs an in-memory decoder");
        let mut out = Vec::with_capacity(self.k * BLOCK_BYTES);
        for b in mem.blocks { out.extend_from_slice(&*b.unwrap()); }
        out.truncate(self.data_len);
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fountain::{LtEncoder, BLOCK_BYTES};
    fn data(n: usize) -> Vec<u8> { (0..n).map(|i| (i as u64 * 2654435761 >> 7) as u8).collect() }

    #[test]
    fn systematic_only_decodes_with_zero_overhead() {
        let d = data(1000 * BLOCK_BYTES + 17);
        let enc = LtEncoder::new(&d);
        assert_eq!(enc.k(), 1001);
        let mut dec = LtDecoder::new(enc.k(), d.len());
        for pid in 0..1001u32 { let done = dec.add(pid, &enc.packet(pid)); assert_eq!(done, pid == 1000); }
        assert_eq!(dec.take(), Some(d));
    }
    #[test]
    fn repair_only_decodes_with_small_overhead() {
        let d = data(600 * BLOCK_BYTES);
        let enc = LtEncoder::new(&d);
        let mut dec = LtDecoder::new(enc.k(), d.len());
        let mut pid = 600u32;
        while !dec.add(pid, &enc.packet(pid)) { pid += 1; assert!(pid < 600 + 600 + 18, "needed more than 3% overhead"); }
        assert_eq!(dec.take(), Some(d));
    }
    #[test]
    fn half_systematic_lost_then_repair() {
        let d = data(1024 * BLOCK_BYTES);
        let enc = LtEncoder::new(&d);
        let mut dec = LtDecoder::new(1024, d.len());
        for pid in (0..1024u32).filter(|p| p % 2 == 0) { dec.add(pid, &enc.packet(pid)); }
        assert_eq!(dec.solved_blocks(), 512);
        let mut pid = 1024u32; let mut n = 0;
        while !dec.add(pid, &enc.packet(pid)) { pid += 1; n += 1; assert!(n < 600); }
        assert!(n + 1 < 512 + 26, "repair packets used: {}", n + 1);
        assert_eq!(dec.take(), Some(d));
    }
    #[test]
    fn duplicates_and_garbage_pids_are_harmless() {
        let d = data(3 * BLOCK_BYTES);
        let enc = LtEncoder::new(&d);
        let mut dec = LtDecoder::new(3, d.len());
        dec.add(0, &enc.packet(0)); dec.add(0, &enc.packet(0));
        assert_eq!(dec.received(), 1);
        dec.add(1, &enc.packet(1));
        assert!(dec.add(2, &enc.packet(2)));
        assert!(dec.is_done());
    }
    #[test]
    fn tiny_chunk_one_block() {
        let d = data(10);
        let enc = LtEncoder::new(&d);
        let mut dec = LtDecoder::new(1, 10);
        assert!(dec.add(5, &enc.packet(5)));
        assert_eq!(dec.take(), Some(d));
    }
    #[test]
    fn external_store_receives_blocks_and_memory_stays_small() {
        struct VecStore(Vec<Option<[u8; BLOCK_BYTES]>>);
        impl BlockStore for VecStore {
            fn read(&mut self, i: usize) -> [u8; BLOCK_BYTES] { self.0[i].unwrap() }
            fn write(&mut self, i: usize, d: &[u8; BLOCK_BYTES]) { self.0[i] = Some(*d); }
        }
        let d = data(1024 * BLOCK_BYTES);
        let enc = LtEncoder::new(&d);
        let mut dec = LtDecoder::new_external(1024, d.len());
        let mut store = VecStore(vec![None; 1024]);
        for pid in 0..1024u32 { if pid % 33 != 0 { dec.add_with(pid, &enc.packet(pid), &mut store); } }
        assert!(dec.memory_bytes() < 64 * 1024, "memory after systematic pass: {}", dec.memory_bytes());
        let mut pid = 1024u32;
        while !dec.add_with(pid, &enc.packet(pid), &mut store) { pid += 1; assert!(pid < 1024 + 200); }
        let out: Vec<u8> = store.0.iter().flat_map(|b| b.unwrap()).collect();
        assert_eq!(out, d);
        assert!(dec.memory_bytes() < 16 * 1024, "no pending equations after completion");
    }
}
