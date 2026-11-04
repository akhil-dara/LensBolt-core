//! Minimal data parallelism on scoped std threads (no dependency): the decoder's hot loops are
//! embarrassingly parallel over rows, blocks or codewords.

thread_local! { static SERIAL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }

/// From now on this thread runs every parallel helper on itself. For a thread that is one of several decoding
/// side by side (the live receiver's workers): its loops would only compete with the other workers, and each
/// thread start costs about 1 ms on Windows (40 phone captures tracked in 19.5 ms on one thread, 80 ms on 16).
pub fn serial_here() { SERIAL.with(|s| s.set(true)); }

/// Worker threads: all cores, or `LB_THREADS` (benchmarks measure single-core cost with LB_THREADS=1); 1 on a
/// thread marked `serial_here`.
fn threads() -> usize {
    if SERIAL.with(|s| s.get()) { return 1; }
    static T: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var("LB_THREADS").ok().and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1)).clamp(1, 16))
}

/// Work below this many items (pixels) runs on the calling thread: starting a thread per core costs more than
/// it saves on a marker window (115 x 115 px: 60 ms of thread starts per tracked capture against 0.6 ms of work).
pub const PAR_MIN_ITEMS: usize = 1 << 16;

/// `par_ranges` over the `rows` rows of a `rows` x `width` image: on the calling thread below [PAR_MIN_ITEMS] px.
pub fn par_ranges_px<T: Send, F: Fn(std::ops::Range<usize>) -> T + Sync>(rows: usize, width: usize, f: F) -> Vec<T> {
    if rows * width < PAR_MIN_ITEMS { return vec![f(0..rows)]; }
    par_ranges(rows, f)
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

/// Run `f(y, row)` for every row of a row-major array of `width` items per row, one contiguous band of rows per
/// core: each call writes only its own row.
pub fn par_chunks_rows<A: Send, F: Fn(usize, &mut [A]) + Sync>(a: &mut [A], width: usize, f: F) {
    let rows = if width == 0 { 0 } else { a.len() / width };
    let threads = threads();
    if threads == 1 || rows < 64 || a.len() < PAR_MIN_ITEMS { for (y, line) in a.chunks_mut(width.max(1)).enumerate() { f(y, line); } return; }
    let chunk = rows.div_ceil(threads);
    let f = &f;
    std::thread::scope(|sc| {
        for (k, band) in a.chunks_mut(chunk * width).enumerate() {
            sc.spawn(move || for (i, line) in band.chunks_mut(width).enumerate() { f(k * chunk + i, line); });
        }
    });
}

/// Run `f(rows, a_band, b_band)` over two row-major arrays of `width` items per row (`a` and `b` the same
/// number of rows), split into one contiguous band of rows per core: each call writes only its own rows.
pub fn par_rows_mut<A: Send, B: Send, F: Fn(std::ops::Range<usize>, &mut [A], &mut [B]) + Sync>(a: &mut [A], b: &mut [B], width: usize, f: F) {
    let rows = if width == 0 { 0 } else { a.len() / width };
    assert!(b.len() == rows * width && a.len() == rows * width, "par_rows_mut: arrays of {} and {} items are not {rows} rows of {width}", a.len(), b.len());
    let threads = threads();
    if threads == 1 || rows < 64 || a.len() < PAR_MIN_ITEMS { f(0..rows, a, b); return; }
    let chunk = rows.div_ceil(threads);
    let f = &f;
    std::thread::scope(|sc| {
        for (k, (ab, bb)) in a.chunks_mut(chunk * width).zip(b.chunks_mut(chunk * width)).enumerate() {
            let start = k * chunk;
            sc.spawn(move || f(start..start + ab.len() / width, ab, bb));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    // a marker window (115 x 115 px) is tiny work: a thread per core to start, twice per search and eight
    // searches per capture, cost 60 ms of a tracked decode against 0.6 ms on the calling thread (2026-11-03)
    #[test]
    fn small_work_stays_on_the_calling_thread() {
        let me = std::thread::current().id();
        let mut a = vec![0u8; 115 * 115];
        par_chunks_rows(&mut a, 115, |_, line| { assert_eq!(std::thread::current().id(), me); line[0] = 1; });
        let mut b = vec![0u8; 115 * 115];
        par_rows_mut(&mut a, &mut b, 115, |_, _, _| assert_eq!(std::thread::current().id(), me));
        let ids = par_ranges_px(115, 115, |_| std::thread::current().id());
        assert!(ids.iter().all(|&i| i == me));
    }

    // a live decode worker: the receiver already decodes one capture per worker side by side, and a thread
    // start costs about 1 ms on Windows (2026-11-03: 40 phone captures tracked in 19.5 ms on one thread, 80 ms
    // on 16), so a worker's decode runs its loops on its own thread
    #[test]
    fn a_serial_thread_runs_large_work_on_itself() {
        let me = std::thread::current().id();
        std::thread::scope(|s| { s.spawn(|| {
            serial_here();
            let them = std::thread::current().id();
            assert!(par_map(10_000, |_| std::thread::current().id()).iter().all(|&i| i == them));
            let mut a = vec![0u8; 1000 * 1000];
            par_chunks_rows(&mut a, 1000, |_, _| assert_eq!(std::thread::current().id(), them));
        }); });
        // other threads are untouched
        if threads() > 1 { assert!(par_map(10_000, |_| std::thread::current().id()).iter().any(|&i| i != me)); }
    }
}
