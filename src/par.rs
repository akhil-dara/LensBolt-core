//! Minimal data parallelism on scoped std threads (no dependency): the decoder's hot loops are
//! embarrassingly parallel over rows, blocks or codewords.

/// Worker threads: all cores, or `LB_THREADS` (benchmarks measure single-core cost with LB_THREADS=1).
fn threads() -> usize {
    static T: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var("LB_THREADS").ok().and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1)).clamp(1, 16))
}

/// Run `f(0..n)` on all cores over contiguous chunks; results in index order.
pub fn par_map<T: Send, F: Fn(usize) -> T + Sync>(n: usize, f: F) -> Vec<T> {
    let threads = threads();
    if threads == 1 || n < 2 { return (0..n).map(&f).collect(); }
    let chunk = n.div_ceil(threads);
    let f = &f;
    std::thread::scope(|sc| {
        let handles: Vec<_> = (0..n).step_by(chunk).map(|start| sc.spawn(move || (start..(start + chunk).min(n)).map(f).collect::<Vec<T>>())).collect();
        handles.into_iter().flat_map(|hd| hd.join().unwrap()).collect()
    })
}

/// Run `f(range)` over `n` items split into one contiguous range per core; results in range order.
pub fn par_ranges<T: Send, F: Fn(std::ops::Range<usize>) -> T + Sync>(n: usize, f: F) -> Vec<T> {
    let threads = threads();
    if threads == 1 || n < 64 { return vec![f(0..n)]; }
    let chunk = n.div_ceil(threads);
    let f = &f;
    std::thread::scope(|sc| {
        let handles: Vec<_> = (0..n).step_by(chunk).map(|start| sc.spawn(move || f(start..(start + chunk).min(n)))).collect();
        handles.into_iter().map(|hd| hd.join().unwrap()).collect()
    })
}

/// Run `f(rows, a_band, b_band)` over two row-major arrays of `width` items per row (`a` and `b` the same
/// number of rows), split into one contiguous band of rows per core: each call writes only its own rows.
pub fn par_rows_mut<A: Send, B: Send, F: Fn(std::ops::Range<usize>, &mut [A], &mut [B]) + Sync>(a: &mut [A], b: &mut [B], width: usize, f: F) {
    let rows = if width == 0 { 0 } else { a.len() / width };
    assert!(b.len() == rows * width && a.len() == rows * width, "par_rows_mut: arrays of {} and {} items are not {rows} rows of {width}", a.len(), b.len());
    let threads = threads();
    if threads == 1 || rows < 64 { f(0..rows, a, b); return; }
    let chunk = rows.div_ceil(threads);
    let f = &f;
    std::thread::scope(|sc| {
        for (k, (ab, bb)) in a.chunks_mut(chunk * width).zip(b.chunks_mut(chunk * width)).enumerate() {
            let start = k * chunk;
            sc.spawn(move || f(start..start + ab.len() / width, ab, bb));
        }
    });
}
