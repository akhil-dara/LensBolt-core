//! The live receiver's status lock covers counters only: while the collector is deep in a slow feed (a big
//! segment written and verified on a slow disk), the camera thread's `submit_capture` and the UI's `status()`
//! still answer at once.
use lensbolt_core::file::{PartStore, StoreFactory};
use lensbolt_core::flat::live::{Capture, LiveConfig, LiveReceiver};
use lensbolt_core::flat::{render, FlatSender, FlatSenderConfig};
use lensbolt_core::image::{GrayImage, RgbImage};
use lensbolt_core::testutil::pseudo_random_file;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::SeqCst};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A slow disk: while `slow` is set every write sleeps 200 ms; `inside` counts writes in progress.
#[derive(Clone, Default)]
struct Gate { slow: Arc<AtomicBool>, inside: Arc<AtomicUsize> }
struct SlowStores(Gate);
struct SlowStore { data: Vec<u8>, gate: Gate }
impl PartStore for SlowStore {
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let o = offset as usize;
        buf.copy_from_slice(self.data.get(o..o + buf.len()).ok_or_else(|| io::Error::other("read past the end"))?);
        Ok(())
    }
    fn write_at(&mut self, offset: u64, d: &[u8]) -> io::Result<()> {
        self.gate.inside.fetch_add(1, SeqCst);
        if self.gate.slow.load(SeqCst) { std::thread::sleep(Duration::from_millis(200)); }
        let o = offset as usize;
        if self.data.len() < o + d.len() { self.data.resize(o + d.len(), 0); }
        self.data[o..o + d.len()].copy_from_slice(d);
        self.gate.inside.fetch_sub(1, SeqCst);
        Ok(())
    }
    fn set_len(&mut self, len: u64) -> io::Result<()> { self.data.resize(len as usize, 0); Ok(()) }
}
impl StoreFactory for SlowStores {
    fn create(&mut self, _name: &str) -> io::Result<Box<dyn PartStore>> { Ok(Box::new(SlowStore { data: Vec::new(), gate: self.0.clone() })) }
    fn remove(&mut self, _name: &str) {}
}

#[test]
fn status_answers_while_the_receiver_is_slow() {
    let src = std::env::temp_dir().join("lensbolt_live_lock.bin");
    pseudo_random_file(&src, 200_000, 31).unwrap();
    let mut s = FlatSender::new(&src, FlatSenderConfig { manifest_every_frames: 10, transfer_id: 93, ..FlatSenderConfig::default() }, Some(false)).unwrap();
    let gate = Gate::default();
    gate.slow.store(true, SeqCst);
    let live = LiveReceiver::with_stores(Box::new(SlowStores(gate.clone())), LiveConfig { workers: 2 }).unwrap();
    let (calls, accepted) = (AtomicU64::new(0), AtomicU64::new(0));
    let offer = |make: Box<dyn FnOnce() -> Capture + Send>| { calls.fetch_add(1, SeqCst); if live.submit_capture(make) { accepted.fetch_add(1, SeqCst); } };
    let probe_done = AtomicBool::new(false);
    // a wait behind the receiver is a 200 ms write; 50 ms leaves room for Windows time slices
    let limit = Duration::from_millis(50);
    let deadline = Instant::now() + Duration::from_secs(120);
    // the sender's real frames, offered as workers free up, until the collector is feeding the slow store; this
    // loop is the only submitter until it stops, then the probe is
    while gate.inside.load(SeqCst) == 0 {
        assert!(Instant::now() < deadline, "no feed ever reached the store: {:?}", live.status());
        let img = render(&s.next_frame().unwrap());
        while !live.has_room() && gate.inside.load(SeqCst) == 0 {
            assert!(Instant::now() < deadline, "no worker came free: {:?}", live.status());
            std::thread::sleep(Duration::from_millis(1));
        }
        if gate.inside.load(SeqCst) > 0 { break; }
        offer(Box::new(move || Capture { y: img.to_gray(), rgb: Some(Box::new(move || img)) }));
    }
    let (worst_status, worst_submit, overlapped) = std::thread::scope(|sc| {
        // a watchdog: if the probe is stuck behind the collector, let the disk go fast after 5 s so the test ends
        sc.spawn(|| {
            let t0 = Instant::now();
            while !probe_done.load(SeqCst) && t0.elapsed() < Duration::from_secs(5) { std::thread::sleep(Duration::from_millis(1)); }
            gate.slow.store(false, SeqCst);
        });
        // the probe ends the watchdog's wait however it ends (a failed assert included)
        struct Finished<'a>(&'a AtomicBool);
        impl Drop for Finished<'_> { fn drop(&mut self) { self.0.store(true, SeqCst); } }
        let _finished = Finished(&probe_done);
        // the UI and the camera, while the collector is inside a slow write: 20 calls each, every one timed
        let (mut ws, mut wc) = (Duration::ZERO, Duration::ZERO);
        for _ in 0..20 {
            let t = Instant::now(); let _ = live.status(); ws = ws.max(t.elapsed());
            // a capture whose planes take a while to convert: the first few take the free workers, then
            // has_room is false and the rest are dropped; either way the call must return at once
            let t = Instant::now();
            offer(Box::new(|| { std::thread::sleep(Duration::from_millis(50)); let g = GrayImage::new(64, 64); Capture { y: g, rgb: Some(Box::new(|| RgbImage::new(64, 64))) } }));
            wc = wc.max(t.elapsed());
        }
        (ws, wc, gate.inside.load(SeqCst) > 0)
    });
    gate.slow.store(false, SeqCst);
    assert!(worst_status <= limit, "status() waited {worst_status:?} behind the receiver");
    assert!(worst_submit <= limit, "submit_capture waited {worst_submit:?} behind the receiver");
    assert!(overlapped, "the probe outlasted the slow write: it did not run while a feed was in progress");
    // the counters stay exact: every offer is received, and each one is either accepted or dropped
    let st = live.status();
    assert_eq!(st.received, calls.load(SeqCst), "{st:?}");
    assert_eq!(st.received, accepted.load(SeqCst) + st.dropped, "{st:?}");
    // and no capture is counted twice: each accepted one is read, failed, skipped for colour or still in flight
    assert!(st.received >= st.decoded + st.failed + st.colour_skipped + st.dropped, "{st:?}");
}
