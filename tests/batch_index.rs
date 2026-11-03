//! The batch index decoder's memory bound, measured: an allocator that counts this thread's live bytes, so the
//! peak while decoding can be held against the input's length (its own binary, so no other test shares it).
use lensbolt_core::file::batch::{Index, Entry, INDEX_MAGIC, INDEX_VERSION};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Counting;
thread_local! {
    static LIVE: Cell<usize> = const { Cell::new(0) };
    static PEAK: Cell<usize> = const { Cell::new(0) };
}
fn note(add: usize, sub: usize) {
    let _ = LIVE.try_with(|l| { let v = l.get() + add - sub.min(l.get() + add); l.set(v); let _ = PEAK.try_with(|p| p.set(p.get().max(v))); });
}
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 { note(l.size(), 0); System.alloc(l) }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) { note(0, l.size()); System.dealloc(p, l) }
    // both blocks are live while realloc copies: the new size counts first, the old one goes after
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 { note(n, 0); let q = System.realloc(p, l, n); note(0, l.size()); q }
}
#[global_allocator]
static A: Counting = Counting;

/// Peak bytes allocated above the starting level while decoding `b`.
fn peak(b: &[u8]) -> usize {
    let base = LIVE.with(|l| l.get());
    PEAK.with(|p| p.set(base));
    let r = std::hint::black_box(Index::decode(b));
    let top = PEAK.with(|p| p.get()) - base;
    drop(r);
    top
}
fn varint(mut v: u64, b: &mut Vec<u8>) {
    loop { let low = (v & 0x7f) as u8; v >>= 7; if v == 0 { b.push(low); return; } b.push(low | 0x80); }
}

#[test]
fn decoding_allocates_a_small_multiple_of_the_input() {
    // a million entries promised in 40 bytes, and the largest count in a few: nothing is reserved for them
    for (count, extra) in [(1_000_000u64, 40usize), (u64::MAX, 10), (999_999, 36 * 999_998)] {
        let mut b = INDEX_MAGIC.to_vec(); b.push(INDEX_VERSION); varint(count, &mut b); b.resize(b.len() + extra, 0);
        let p = peak(&b);
        assert!(Index::decode(&b).is_err());
        assert!(p <= 4096, "count {count} in {} bytes: peak {p} bytes", b.len());
    }
    // valid indexes, tiny names (the most entries per byte) and long ones (the most name per entry)
    for (n, name_len) in [(100_000usize, 1usize), (10_000, 255), (1, 1)] {
        let entries = (0..n).map(|i| Entry { name: "a".repeat(name_len), size: i as u64 % 100, stored_size: i as u64 % 100, compressed: false, offset: 0, sha256: [i as u8; 32] }).collect();
        let mut idx = Index { entries };
        let mut at = 0; for e in &mut idx.entries { e.offset = at; at += e.stored_size; }
        let b = idx.encode();
        let p = peak(&b);
        assert_eq!(Index::decode(&b).map(|i| i.entries.len()), Ok(n));
        eprintln!("{n} entries, names of {name_len}: {} bytes in, peak {p} bytes ({:.2}x)", b.len(), p as f64 / b.len() as f64);
        assert!(p <= 4 * b.len() + 4096, "{n} entries: peak {p} for {} bytes", b.len());
    }
    // random bytes behind a good header
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    for _ in 0..2_000 {
        let mut b = INDEX_MAGIC.to_vec(); b.push(INDEX_VERSION);
        let n = (s % 4096) as usize;
        for _ in 0..n { s ^= s << 13; s ^= s >> 7; s ^= s << 17; b.push(s as u8); }
        let p = peak(&b);
        assert!(p <= 4 * b.len() + 4096, "{} random bytes: peak {p}", b.len());
    }
}
