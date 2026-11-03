//! The sender's frame rate, read from the frame numbers (`frame_seq`, mod 256) of decoded captures against
//! their capture times. The camera runs ~30 fps, the sender 15, 30 or 60; captures are lost (busy workers,
//! failed reads), arrive out of order (several workers) and the 8-bit number wraps. The estimate must stay
//! within 10% of the truth after 1.5 s and read 0 (not the old value) when the frame number stops.
use lensbolt_core::flat::live::{Capture, LiveConfig, LiveReceiver};
use lensbolt_core::flat::rate::RateEstimate;
use lensbolt_core::flat::session::RxSession;
use lensbolt_core::flat::{render, Alphabet, CellFormat, FlatSender, FlatSenderConfig, MANIFEST_EVERY_FRAMES};
use lensbolt_core::testutil::pseudo_random_file;
use std::path::PathBuf;
use std::time::{Duration, Instant};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Rng { Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1) }
    /// uniform in [0, 1)
    fn f(&mut self) -> f64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; (self.0 >> 11) as f64 / (1u64 << 53) as f64 }
}

const CAMERA_MS: f64 = 1000.0 / 30.0;

/// One decoded capture: the frame number it read and the time it was taken (as reported, with jitter).
#[derive(Clone, Copy, Debug)]
struct Cap { seq: u8, t: f64 }

/// A 30 fps camera looking at a sender whose frame number at true time `t` (ms) is `frame(t)`: captures
/// lost with probability `loss`, the reported time off the true one by up to +-5 ms. In time order.
fn camera(secs: f64, loss: f64, rng: &mut Rng, frame: impl Fn(f64) -> u64) -> Vec<Cap> { camera_every(CAMERA_MS, secs, loss, rng, frame) }
/// `camera` taking a capture every `period` ms.
fn camera_every(period: f64, secs: f64, loss: f64, rng: &mut Rng, frame: impl Fn(f64) -> u64) -> Vec<Cap> {
    // a monotonic clock that started long ago, and a random camera phase
    let t0 = 50_000.0 + rng.f() * period;
    let n = (secs * 1000.0 / period) as usize;
    let mut out = Vec::new();
    for k in 0..n {
        let t = t0 + k as f64 * period;
        let (jitter, lost) = ((rng.f() * 2.0 - 1.0) * 5.0, rng.f() < loss);
        if !lost { out.push(Cap { seq: (frame(t) & 0xFF) as u8, t: t + jitter }); }
    }
    out
}

/// A sender at a constant `fs` fps from a random phase and a random first frame number.
fn steady(fs: f64, rng: &mut Rng) -> impl Fn(f64) -> u64 {
    let (phase, first) = (rng.f() * 1000.0, (rng.f() * 256.0) as u64);
    move |t: f64| first + ((t + phase) * fs / 1000.0).floor() as u64
}

/// Feed `caps` in order; every reading once the newest capture is 1.5 s after the first must be within 10%
/// of `fs`, and none may exist while less than 0.5 s has been seen. Returns the worst relative error.
fn check(caps: &[Cap], fs: f64, what: &str) -> f64 {
    let mut est = RateEstimate::new();
    let t_first = caps.iter().map(|c| c.t).fold(f64::MAX, f64::min);
    let (mut newest, mut worst, mut readings) = (f64::MIN, 0.0f64, 0);
    for c in caps {
        est.add(c.seq, c.t);
        newest = newest.max(c.t);
        let got = est.fps();
        if newest - t_first < 500.0 { assert!(got.is_none(), "{what}: a reading after {:.0} ms: {got:?}", newest - t_first); }
        if newest - t_first >= 1500.0 {
            let f = got.unwrap_or_else(|| panic!("{what}: no reading {:.0} ms in", newest - t_first));
            let err = (f - fs).abs() / fs;
            assert!(err <= 0.10, "{what}: {f:.2} fps for {fs} at {:.0} ms", newest - t_first);
            worst = worst.max(err);
            readings += 1;
        }
    }
    assert!(readings > 0, "{what}: never checked");
    worst
}

#[test]
fn rate_within_ten_percent_at_15_30_60_fps_with_loss() {
    for fs in [15.0, 30.0, 60.0] {
        for loss in [0.0, 0.2, 0.5] {
            let mut worst = 0.0f64;
            for seed in 0..12 {
                let mut rng = Rng::new(seed * 7919 + (fs as u64) * 31 + (loss * 100.0) as u64);
                let f = steady(fs, &mut rng);
                let caps = camera(6.0, loss, &mut rng, f);
                worst = worst.max(check(&caps, fs, &format!("sender {fs} fps, loss {loss}, seed {seed}")));
            }
            println!("sender {fs:>4} fps, camera 30 fps, loss {:>3.0}%: worst error {:.2}%", loss * 100.0, worst * 100.0);
        }
    }
}

#[test]
fn rate_stays_right_across_the_frame_number_wrap() {
    // 60 fps for 10 s: 600 frames, the 8-bit number wraps twice
    for loss in [0.0, 0.2] {
        let mut worst = 0.0f64;
        for seed in 0..6 {
            let mut rng = Rng::new(1000 + seed);
            let f = steady(60.0, &mut rng);
            let caps = camera(10.0, loss, &mut rng, f);
            let span = caps.last().unwrap().t - caps[0].t;
            assert!(span * 60.0 / 1000.0 > 512.0, "{span} ms");
            worst = worst.max(check(&caps, 60.0, &format!("wrap, loss {loss}, seed {seed}")));
        }
        println!("wrap at 60 fps, loss {:>3.0}%: worst error {:.2}%", loss * 100.0, worst * 100.0);
    }
}

#[test]
fn rate_survives_captures_out_of_order() {
    // several workers: captures come back shuffled within 150 ms of each other
    for fs in [15.0, 30.0, 60.0] {
        let mut worst = 0.0f64;
        for seed in 0..8 {
            let mut rng = Rng::new(2000 + seed + fs as u64);
            let f = steady(fs, &mut rng);
            let caps = camera(6.0, 0.2, &mut rng, f);
            let t0 = caps[0].t;
            let mut shuffled = Vec::new();
            let mut i = 0;
            while i < caps.len() {
                let bucket = ((caps[i].t - t0) / 150.0).floor();
                let mut j = i;
                while j < caps.len() && ((caps[j].t - t0) / 150.0).floor() == bucket { j += 1; }
                let mut b = caps[i..j].to_vec();
                for k in (1..b.len()).rev() { let r = (rng.f() * (k + 1) as f64) as usize; b.swap(k, r); }
                shuffled.extend(b);
                i = j;
            }
            assert!(shuffled.windows(2).any(|w| w[1].t < w[0].t), "nothing reordered");
            worst = worst.max(check(&shuffled, fs, &format!("reordered, sender {fs} fps, seed {seed}")));
        }
        println!("reordered within 150 ms, sender {fs:>4} fps, loss 20%: worst error {:.2}%", worst * 100.0);
    }
}

#[test]
fn a_stalled_sender_reads_zero_then_the_new_rate() {
    // 30 fps for 3 s, the same frame for 3 s, then 15 fps (the sender paused and changed its rate)
    let mut worst_recovery = 0.0f64;
    let mut latest_zero = 0.0f64;
    for seed in 0..8 {
        let mut rng = Rng::new(3000 + seed);
        let (phase, first) = (rng.f() * 1000.0, (rng.f() * 256.0) as u64);
        let t_start = 50_000.0;
        let (stall, resume) = (t_start + 3000.0, t_start + 6000.0);
        let at = move |t: f64| -> u64 {
            let frozen = first + ((stall + phase) * 30.0 / 1000.0).floor() as u64;
            if t < stall { first + ((t + phase) * 30.0 / 1000.0).floor() as u64 }
            else if t < resume { frozen }
            else { frozen + 1 + ((t - resume) * 15.0 / 1000.0).floor() as u64 }
        };
        let caps = camera(10.0, 0.2, &mut rng, at);
        let frozen = (at(stall) & 0xFF) as u8;
        // the frozen frame first seen (the last new frame number), and the first capture after the resume
        let first_frozen = caps.iter().find(|c| c.seq == frozen && c.t > stall - 200.0).unwrap().t;
        let first_new = caps.iter().find(|c| c.t > resume - 50.0 && c.seq != frozen).unwrap().t;
        let mut est = RateEstimate::new();
        let mut zero_at = None;
        for c in &caps {
            est.add(c.seq, c.t);
            let got = est.fps();
            if c.t >= caps[0].t + 1500.0 && c.t < first_frozen {
                let f = got.unwrap();
                assert!((f - 30.0).abs() <= 3.0, "seed {seed}: {f:.2} before the stall");
            }
            if c.t >= first_frozen && c.t < first_new {
                // during the stall: falling, never above the old rate
                if let Some(f) = got { assert!(f <= 33.0, "seed {seed}: {f:.2} during the stall"); }
                if got == Some(0.0) && zero_at.is_none() { zero_at = Some(c.t - first_frozen); }
                if c.t >= first_frozen + 2000.0 + 10.0 { assert_eq!(got, Some(0.0), "seed {seed}: {:.0} ms into the stall", c.t - first_frozen); }
            }
            if c.t >= first_new {
                if let Some(f) = got { assert!(f <= 16.5, "seed {seed}: {f:.2} recovering to 15 fps"); }
                if c.t >= first_new + 1500.0 {
                    let f = got.unwrap_or_else(|| panic!("seed {seed}: no reading {:.0} ms after the resume", c.t - first_new));
                    let err = (f - 15.0).abs() / 15.0;
                    assert!(err <= 0.10, "seed {seed}: {f:.2} fps {:.0} ms after the resume", c.t - first_new);
                    worst_recovery = worst_recovery.max(err);
                }
            }
        }
        latest_zero = latest_zero.max(zero_at.expect("never read 0 in the stall"));
    }
    println!("stall: 0 within {latest_zero:.0} ms of the last new frame; 15 fps after the resume: worst error {:.2}% from 1.5 s on", worst_recovery * 100.0);
}

#[test]
fn rate_is_unknown_with_too_little_seen() {
    let mut est = RateEstimate::new();
    assert_eq!(est.fps(), None);
    // three frames, but only 0.2 s
    for (i, t) in [0.0, 100.0, 200.0].iter().enumerate() { est.add(i as u8, 1000.0 + t); }
    assert_eq!(est.fps(), None);
    // 0.6 s, but only two distinct frames
    let mut est = RateEstimate::new();
    for (seq, t) in [(5u8, 0.0), (5, 300.0), (6, 600.0)] { est.add(seq, 1000.0 + t); }
    assert_eq!(est.fps(), None);
    // not-a-number times are ignored
    est.add(7, f64::NAN);
    assert_eq!(est.fps(), None);
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join("lensbolt_sender_rate"); std::fs::create_dir_all(&d).unwrap(); d.join(name)
}

/// One real-time receiver at a time: the paced tests measure against the wall clock.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
fn serial() -> std::sync::MutexGuard<'static, ()> { SERIAL.lock().unwrap_or_else(|e| e.into_inner()) }

fn bw_sender(src: &PathBuf, bytes: u64, seed: u64, id: u16) -> FlatSender {
    pseudo_random_file(src, bytes, seed).unwrap();
    let cfg = FlatSenderConfig { manifest_every_frames: MANIFEST_EVERY_FRAMES, transfer_id: id, format: CellFormat { pitch: 2, alphabet: Alphabet::Bw }, ..FlatSenderConfig::default() };
    FlatSender::new(src, cfg, Some(false)).unwrap()
}

#[test]
fn live_status_carries_the_sender_rate() {
    let _g = serial();
    // a sender at 30 fps, every frame offered at 30 fps in real time (the capture time is the offer time)
    let src = tmp("live.bin");
    let mut s = bw_sender(&src, 6_000_000, 51, 151);
    let out = tmp("live_out"); let _ = std::fs::remove_dir_all(&out); std::fs::create_dir_all(&out).unwrap();
    let live = LiveReceiver::new(&out, LiveConfig { workers: 4 }).unwrap();
    assert_eq!(live.status().sender_fps, 0.0);
    assert!(live.status().to_json().contains("\"sender_fps\":0.0"), "{}", live.status().to_json());
    let period = Duration::from_nanos(1_000_000_000 / 30);
    let base = Instant::now();
    let mut accepted = 0u64;
    for i in 0..60u32 {
        let img = render(&s.next_frame().unwrap());
        if let Some(wait) = (base + period * i).checked_duration_since(Instant::now()) { std::thread::sleep(wait); }
        if live.has_room() && live.submit_capture(move || Capture { y: img.to_gray(), rgb: Some(Box::new(move || img)) }) { accepted += 1; }
    }
    let t0 = Instant::now();
    let st = loop {
        let st = live.status();
        if st.decoded + st.failed + st.colour_skipped >= accepted { break st; }
        assert!(t0.elapsed() < Duration::from_secs(30), "captures never came back: {st:?}");
        std::thread::sleep(Duration::from_millis(2));
    };
    assert!(st.complete.is_none(), "the file must outlast the run: {st:?}");
    let err = (st.sender_fps - 30.0).abs() / 30.0;
    println!("LiveReceiver, sender 30 fps offered at 30 fps for 2 s: sender_fps {:.2} ({:.2}% off), {} of 60 accepted, {} decoded", st.sender_fps, err * 100.0, accepted, st.decoded);
    assert!(err <= 0.10, "sender_fps {:.2}: {st:?}", st.sender_fps);
    assert!(st.to_json().contains(&format!("\"sender_fps\":{:.1}", st.sender_fps)), "{}", st.to_json());
}

/// The browser's value: `RxSession` timed by the page's clock (`now_ms`).
fn json_num(j: &str, key: &str) -> f64 {
    let at = j.find(&format!("\"{key}\":")).unwrap_or_else(|| panic!("no {key} in {j}")) + key.len() + 3;
    j[at..].split([',', '}']).next().unwrap().parse().unwrap()
}

#[test]
fn browser_status_carries_the_sender_rate() {
    // a sender at 60 fps seen by a 30 fps camera: every other frame, 2 s of captures
    let src = tmp("session.bin");
    let mut s = bw_sender(&src, 8_000_000, 52, 152);
    let mut rx = RxSession::new();
    assert_eq!(json_num(&rx.status_json(), "sender_fps"), 0.0);
    let mut last = String::new();
    for n in 0..60u32 {
        let f = s.next_frame().unwrap();
        s.next_frame().unwrap();
        let img = render(&f);
        let mut rgba = Vec::with_capacity(img.data.len() / 3 * 4);
        for p in img.data.chunks_exact(3) { rgba.extend_from_slice(&[p[0], p[1], p[2], 255]); }
        last = rx.push_rgba_at(&rgba, img.width, img.height, 9_000.0 + n as f64 * CAMERA_MS);
    }
    assert!(!rx.is_complete(), "{last}");
    let fps = json_num(&last, "sender_fps");
    println!("RxSession, sender 60 fps, camera 30 fps for 2 s: sender_fps {fps:.1} ({:.2}% off)", (fps - 60.0).abs() / 60.0 * 100.0);
    assert!((fps - 60.0).abs() <= 6.0, "{last}");
    // without a clock there is no rate
    let mut rx = RxSession::new();
    let img = render(&s.next_frame().unwrap());
    let mut rgba = Vec::new();
    for p in img.data.chunks_exact(3) { rgba.extend_from_slice(&[p[0], p[1], p[2], 255]); }
    assert_eq!(json_num(&rx.push_rgba(&rgba, img.width, img.height), "sender_fps"), 0.0);
}

#[test]
fn a_restarted_sender_reads_its_new_rate() {
    // the user changes the sender's rate: a 60 fps sender ending near frame number 200, 50, 240 or 250 stops,
    // and a new sender starts from frame number 0 at 30 fps, at once or after up to 1 s with the tile out of
    // view (some of these jumps look like a 60 fps sender carrying on)
    for old_end in [200u64, 50, 240, 250] {
        for gap in [0.0, 300.0, 650.0, 800.0, 1000.0] {
            let (high, late, err) = restart(old_end, gap, 8);
            println!("restart from ~{old_end} (60 fps) to 0 (30 fps), gap {gap:>4} ms: highest {high:.1} fps, highest from 0.5 s after {late:.1}, worst error against 30 from 1.5 s {:.2}%", err * 100.0);
        }
    }
}

#[test]
fn a_restarted_sender_from_any_frame_number_and_gap() {
    // every 5th last frame number of the old sender, gaps 0..2 s in 0.1 s steps
    let (mut high, mut late, mut err) = (0.0f64, 0.0f64, 0.0f64);
    for old_end in (0..256u64).step_by(5) {
        for gap in (0..=20).map(|g| g as f64 * 100.0) {
            let r = restart(old_end, gap, 4);
            (high, late, err) = (high.max(r.0), late.max(r.1), err.max(r.2));
        }
    }
    println!("restart from every 5th frame number, gaps 0..2 s: highest {high:.1} fps, highest from 0.5 s after {late:.1}, worst error against 30 from 1.5 s {:.2}%", err * 100.0);
}

/// A 60 fps sender whose last frame number is `old_end` stops; `gap_ms` later (nothing decoded meanwhile) a
/// new sender starts from frame number 0 at 30 fps. No reading above 66, none above 33 from 0.5 s after the
/// restart, within 10% of 30 from 1.5 s. Returns (highest reading, highest from 0.5 s, worst error from 1.5 s).
fn restart(old_end: u64, gap: f64, seeds: u64) -> (f64, f64, f64) {
    let (mut worst_high, mut worst_late, mut worst_err) = (0.0f64, 0.0f64, 0.0f64);
    for seed in 0..seeds {
        let mut rng = Rng::new(4000 + seed + old_end + gap as u64);
        let phase = rng.f() * 1000.0;
        let a_end = 50_000.0 + 3000.0;
        let restart = a_end + gap;
        // the old sender's frame number reaches `old_end` on its last frame
        let n_end = ((a_end - 1e-6 + phase) * 60.0 / 1000.0).floor() as u64;
        let first = (old_end + 256 * 1000 - n_end % 256) % 256;
        let at = move |t: f64| -> u64 {
            if t < a_end { first + ((t + phase) * 60.0 / 1000.0).floor() as u64 }
            else { ((t - restart).max(0.0) * 30.0 / 1000.0).floor() as u64 }
        };
        // nothing decodes during the gap (the tile out of view)
        let caps: Vec<Cap> = camera(9.0, 0.2, &mut rng, at).into_iter().filter(|c| c.t < a_end - 5.0 || c.t >= restart + 5.0).collect();
        let last_old = caps.iter().filter(|c| c.t < a_end).last().unwrap().seq;
        let first_new = caps.iter().find(|c| c.t >= restart).unwrap().t;
        let what = format!("old sender ending at {last_old} (60 fps), restart at 30 fps after {gap} ms, seed {seed}");
        let mut est = RateEstimate::new();
        for c in &caps {
            est.add(c.seq, c.t);
            let got = est.fps();
            if let Some(f) = got {
                assert!(f <= 66.0, "{what}: {f:.2} fps {:.0} ms after the restart", c.t - first_new);
                worst_high = worst_high.max(f);
            }
            if c.t >= first_new + 500.0 {
                let f = got.unwrap_or(0.0);
                assert!(f <= 33.0, "{what}: still {f:.2} fps {:.0} ms after the restart", c.t - first_new);
                worst_late = worst_late.max(f);
            }
            if c.t >= first_new + 1500.0 {
                let f = got.unwrap_or_else(|| panic!("{what}: no reading {:.0} ms after the restart", c.t - first_new));
                let err = (f - 30.0).abs() / 30.0;
                assert!(err <= 0.10, "{what}: {f:.2} fps {:.0} ms after the restart", c.t - first_new);
                worst_err = worst_err.max(err);
            }
        }
    }
    (worst_high, worst_late, worst_err)
}

#[test]
fn a_60_fps_sender_out_of_view_for_two_seconds_keeps_its_rate() {
    // 60 fps, nothing decoded for 2.1..2.5 s (a step of more than 127 frames), then the same sender again
    for gap in [2100.0, 2300.0, 2500.0] {
        let mut lowest = f64::MAX;
        for seed in 0..6 {
            let mut rng = Rng::new(5000 + seed + gap as u64);
            let f = steady(60.0, &mut rng);
            let caps = camera(8.0, 0.0, &mut rng, f);
            let (g0, t0) = (caps[0].t + 3000.0, caps[0].t);
            let caps: Vec<Cap> = caps.into_iter().filter(|c| c.t < g0 || c.t >= g0 + gap).collect();
            let mut est = RateEstimate::new();
            for c in &caps {
                est.add(c.seq, c.t);
                if c.t >= t0 + 1500.0 {
                    if c.t >= g0 + gap { lowest = lowest.min(est.fps().unwrap_or(0.0)); }
                    // right after the gap the window holds a capture or two: no reading is fine, a wrong one is not
                    if let Some(f) = est.fps() { assert!((f - 60.0).abs() <= 6.0, "gap {gap} ms, seed {seed}: {f:.2} fps {:.0} ms after the gap", c.t - g0 - gap); }
                }
            }
        }
        println!("60 fps out of view for {gap} ms: lowest reading after the gap {lowest:.1} (0 = no reading yet)");
    }
}

/// A blank camera picture: no tile, the decode fails.
fn blank() -> lensbolt_core::image::RgbImage { lensbolt_core::image::RgbImage::new(640, 480) }

#[test]
fn live_rate_falls_to_zero_while_nothing_decodes() {
    let _g = serial();
    // decoded captures at 30 fps for 2 s, then 3 s of captures that fail (the phone pointed away)
    let src = tmp("away.bin");
    let mut s = bw_sender(&src, 6_000_000, 53, 153);
    let out = tmp("away_out"); let _ = std::fs::remove_dir_all(&out); std::fs::create_dir_all(&out).unwrap();
    let live = LiveReceiver::new(&out, LiveConfig { workers: 4 }).unwrap();
    let period = Duration::from_nanos(1_000_000_000 / 30);
    let settle = |accepted: u64| {
        let t0 = Instant::now();
        loop {
            let st = live.status();
            if st.decoded + st.failed + st.colour_skipped >= accepted { return st; }
            assert!(t0.elapsed() < Duration::from_secs(30), "captures never came back: {st:?}");
            std::thread::sleep(Duration::from_millis(1));
        }
    };
    let base = Instant::now();
    let mut accepted = 0u64;
    for i in 0..60u32 {
        let img = render(&s.next_frame().unwrap());
        if let Some(wait) = (base + period * i).checked_duration_since(Instant::now()) { std::thread::sleep(wait); }
        if live.has_room() && live.submit_capture(move || Capture { y: img.to_gray(), rgb: Some(Box::new(move || img)) }) { accepted += 1; }
    }
    let st = settle(accepted);
    assert!((st.sender_fps - 30.0).abs() <= 3.0, "before: {st:?}");
    let away = base + period * 60;
    let mut zero_by = None;
    for i in 60..150u32 {
        if let Some(wait) = (base + period * i).checked_duration_since(Instant::now()) { std::thread::sleep(wait); }
        let at = Instant::now().saturating_duration_since(away).as_secs_f64() * 1000.0;
        let img = blank();
        if live.has_room() && live.submit_capture(move || Capture { y: img.to_gray(), rgb: Some(Box::new(move || img)) }) { accepted += 1; }
        let st = settle(accepted);
        if st.sender_fps == 0.0 && zero_by.is_none() { zero_by = Some(at); }
        if at >= 2100.0 { assert_eq!(st.sender_fps, 0.0, "{at:.0} ms of failed captures: {st:?}"); }
    }
    let st = live.status();
    assert!(st.failed >= 60, "{st:?}");
    println!("LiveReceiver: 0 after {:.0} ms of failed captures", zero_by.unwrap());
}

#[test]
fn browser_rate_falls_to_zero_while_nothing_decodes() {
    let src = tmp("away_session.bin");
    let mut s = bw_sender(&src, 8_000_000, 54, 154);
    let mut rx = RxSession::new();
    let rgba = |img: &lensbolt_core::image::RgbImage| { let mut v = Vec::with_capacity(img.data.len() / 3 * 4); for p in img.data.chunks_exact(3) { v.extend_from_slice(&[p[0], p[1], p[2], 255]); } v };
    let t = |n: u32| 9_000.0 + n as f64 * CAMERA_MS;
    let mut last = String::new();
    for n in 0..60u32 { let img = render(&s.next_frame().unwrap()); last = rx.push_rgba_at(&rgba(&img), img.width, img.height, t(n)); }
    assert!((json_num(&last, "sender_fps") - 30.0).abs() <= 3.0, "{last}");
    let away = t(60);
    let (img, mut zero_by) = (blank(), None);
    let px = rgba(&img);
    for n in 60..150u32 {
        last = rx.push_rgba_at(&px, img.width, img.height, t(n));
        let at = t(n) - away;
        if json_num(&last, "sender_fps") == 0.0 && zero_by.is_none() { zero_by = Some(at); }
        if at >= 2100.0 { assert_eq!(json_num(&last, "sender_fps"), 0.0, "{at:.0} ms of failed captures: {last}"); }
    }
    assert!(json_num(&last, "failed") >= 90.0, "{last}");
    println!("RxSession: 0 after {:.0} ms of failed captures", zero_by.unwrap());
}

#[test]
fn rate_at_120_fps_seen_at_30_and_10_captures_per_second() {
    // calibration can pick 120 fps; a camera (or a slow browser) decodes 30 or 10 captures a second, 20% lost
    for period in [CAMERA_MS, 100.0] {
        let (mut worst, mut readings, mut checked) = (0.0f64, 0u64, 0u64);
        for seed in 0..12 {
            let mut rng = Rng::new(6000 + seed + period as u64);
            let f = steady(120.0, &mut rng);
            let caps = camera_every(period, 6.0, 0.2, &mut rng, f);
            let mut est = RateEstimate::new();
            for c in &caps {
                est.add(c.seq, c.t);
                if c.t < caps[0].t + 1500.0 { continue; }
                checked += 1;
                if let Some(f) = est.fps() {
                    let err = (f - 120.0).abs() / 120.0;
                    assert!(err <= 0.10, "{:.0} captures/s, seed {seed}: {f:.2} fps at {:.0} ms", 1000.0 / period, c.t - caps[0].t);
                    worst = worst.max(err);
                    readings += 1;
                }
            }
        }
        let share = readings as f64 / checked as f64;
        println!("sender 120 fps, {:.0} captures/s, loss 20%: a reading on {:.1}% of captures from 1.5 s, worst error {:.2}%", 1000.0 / period, share * 100.0, worst * 100.0);
        assert!(share >= 0.90, "{:.0} captures/s: a reading on only {:.1}% of captures", 1000.0 / period, share * 100.0);
    }
}
