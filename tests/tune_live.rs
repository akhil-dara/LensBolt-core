//! Tune in both receivers: the browser's session (`RxSession`) and the app's live receiver (`LiveReceiver`) film
//! the sender's Tune carousel, frame by frame on its schedule, and finish within one loop from any start. The meter's
//! rules (restart, interrupted, a slow sender, a short tile, any anchor) are tested on synthetic samples in
//! core/src/tune/meter.rs; these run the whole path, rendered frames through the decoder.
use lensbolt_core::calib::{Config, FAST};
use lensbolt_core::flat::live::{Capture, LiveConfig, LiveReceiver};
use lensbolt_core::flat::session::RxSession;
use lensbolt_core::flat::{render, Alphabet, FlatSender, FlatSenderConfig};
use lensbolt_core::image::RgbImage;
use lensbolt_core::tune::schedule::loop_ms;
use lensbolt_core::tune::TuneOptions;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// One camera capture: its time and what it showed (None: no screen in the picture).
type Shot = (f64, Option<Arc<RgbImage>>);

/// A camera at `cam_fps` filming `sender` from `t0` for `ms`: each sender frame stays up for its schedule's time
/// times `slow`; `blank(t)`: the screen is out of the picture at t. The sender first skips `skip` frames.
fn film(sender: &mut FlatSender, skip: usize, cam_fps: f64, t0: f64, ms: f64, slow: f64, blank: &dyn Fn(f64) -> bool) -> Vec<Shot> {
    for _ in 0..skip { sender.next_packets().unwrap(); }
    let mut shots = Vec::new();
    let mut frame = sender.next_frame().unwrap();
    let mut frame_end = t0 + 1000.0 / sender.tune_frame_fps().unwrap_or(30) as f64 * slow;
    let mut img: Option<Arc<RgbImage>> = None;
    let mut k = 0;
    loop {
        let t = t0 + 3.0 + k as f64 * 1000.0 / cam_fps;
        if t >= t0 + ms { break; }
        while t >= frame_end {
            frame = sender.next_frame().unwrap();
            frame_end += 1000.0 / sender.tune_frame_fps().unwrap_or(30) as f64 * slow;
            img = None;
        }
        if blank(t) { shots.push((t, None)); } else {
            let i = img.get_or_insert_with(|| Arc::new(render(&frame))).clone();
            shots.push((t, Some(i)));
        }
        k += 1;
    }
    shots
}
fn rgba(img: &RgbImage) -> Vec<u8> {
    let mut v = Vec::with_capacity(img.data.len() / 3 * 4);
    for p in img.data.chunks_exact(3) { v.extend_from_slice(&[p[0], p[1], p[2], 255]); }
    v
}
fn grey() -> Arc<RgbImage> { let mut g = RgbImage::new(1024, 1024); g.fill_rect(0, 0, 1024, 1024, [128, 128, 128]); Arc::new(g) }
fn tune(w: u32, h: u32) -> FlatSender { FlatSender::tune(FlatSenderConfig { width: w, height: h, ..FlatSenderConfig::default() }, true).unwrap() }

/// The status's "tune" object.
fn tune_obj(j: &str) -> &str {
    let at = j.find("\"tune\":").unwrap_or_else(|| panic!("no tune in {j}")) + 7;
    let rest = &j[at..];
    if rest.starts_with("null") { return "null"; }
    let mut depth = 0;
    for (i, c) in rest.char_indices() {
        match c { '{' => depth += 1, '}' => { depth -= 1; if depth == 0 { return &rest[..=i]; } } _ => {} }
    }
    panic!("unbalanced {j}")
}
fn field<'a>(j: &'a str, key: &str) -> &'a str {
    let at = j.find(&format!("\"{key}\":")).unwrap_or_else(|| panic!("no {key} in {j}")) + key.len() + 3;
    let rest = &j[at..];
    let end = if let Some(r) = rest.strip_prefix('"') { r.find('"').unwrap() + 2 } else { rest.find([',', '}']).unwrap() };
    rest[..end].trim_matches('"')
}
fn state(j: &str) -> String { let t = tune_obj(j); if t == "null" { "null".into() } else { field(t, "state").to_string() } }

/// Pushes the shots into a session that starts Tune at the first one; the statuses.
fn session(shots: &[Shot], rx: &mut RxSession) -> Vec<String> {
    shots.iter().map(|(t, img)| { let i = img.clone().unwrap_or_else(grey); rx.push_rgba_at(&rgba(&i), i.width, i.height, *t) }).collect()
}

#[test]
fn perfect_camera_tunes_in_one_loop() {
    // a 1024x640 tile: the whole path at two thirds of the pixels
    let mut s = tune(1024, 640);
    let shots = film(&mut s, 100, 30.0, 1000.0, loop_ms(true) + 1000.0, 1.0, &|_| false);
    let mut rx = RxSession::new();
    rx.start_tune(TuneOptions::default(), 1000.0);
    let mut last = 0.0;
    let mut done_at = None;
    for (j, (t, _)) in session(&shots, &mut rx).iter().zip(&shots) {
        match state(j).as_str() {
            "measuring" => { let p: f64 = field(tune_obj(j), "progress").parse().unwrap(); assert!(p >= last, "{p} < {last}"); last = p; }
            "done" => { done_at.get_or_insert(*t); }
            "waiting" => assert!(done_at.is_none() && last == 0.0),
            other => panic!("{other}: {j}"),
        }
        assert!(j.contains("\"tune_sender\":") && !j.contains("\"received_bytes\":1"), "nothing is received while tuning: {j}");
    }
    let done_at = done_at.expect("done");
    assert!(done_at - shots[0].0 <= loop_ms(true) + 1000.0, "done after {} ms", done_at - shots[0].0);
    let j = rx.status_json();
    let r = tune_obj(&j);
    assert_eq!(Config::from_code(field(r, "words")), Ok(Config { pitch: 2, alphabet: Alphabet::G8, fps: 0 }), "{r}");
    assert_eq!(field(r, "sixty"), "receiver");
    assert!(j.contains("\"tune_sender\":true"), "{j}");
    // back to receiving: "tune" null
    rx.stop_tune();
    assert_eq!(tune_obj(&rx.status_json()), "null");
}

/// The result without the decode times (each receiver measures its own).
fn without_times(r: &str) -> String {
    r.split(',').filter(|kv| !kv.contains("decode_ms") && !kv.contains("edge_px_median") && !kv.contains("contrast_cv")).collect::<Vec<_>>().join(",")
}

#[test]
fn live_receiver_gives_the_same_result() {
    let mut s = tune(1024, 640);
    let shots = film(&mut s, 30, 30.0, 0.0, loop_ms(true) + 800.0, 1.0, &|_| false);
    // the browser's session on a thread of its own while the app's receiver takes the same captures
    let shots_c = shots.clone();
    let browser = std::thread::spawn(move || {
        let mut rx = RxSession::new();
        rx.start_tune(TuneOptions::default(), 0.0);
        session(&shots_c, &mut rx);
        without_times(tune_obj(&rx.status_json()))
    });

    let out = std::env::temp_dir().join("lensbolt_tune_live"); let _ = std::fs::remove_dir_all(&out);
    let live = LiveReceiver::new(&out, LiveConfig { workers: 1 }).unwrap();
    let base = Instant::now() + Duration::from_millis(5);
    live.start_tune_at(TuneOptions::default(), base);
    for (n, (t, img)) in shots.iter().enumerate() {
        let i = img.clone().unwrap_or_else(grey);
        while !live.has_room() { std::thread::sleep(Duration::from_millis(1)); }
        assert!(live.submit_capture_at(base + Duration::from_secs_f64(t / 1000.0), move || Capture { y: i.to_gray(), rgb: Some(Box::new(move || (*i).clone())) }));
        let t_wait = Instant::now();
        loop {
            let st = live.status();
            if st.decoded + st.failed + st.colour_skipped > n as u64 { break; }
            assert!(t_wait.elapsed() < Duration::from_secs(10), "capture {n} not back");
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    let st = live.status();
    let got = without_times(tune_obj(&st.to_json()).trim());
    let want = browser.join().unwrap();
    assert!(want.contains("\"state\":\"done\""), "{want}");
    assert_eq!(got, want);
    assert!(st.tune_sender);
}

#[test]
fn a_normal_sender_says_not_tuning() {
    let data: Vec<u8> = (0..200_000u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
    let mut file = FlatSender::from_bytes("a.bin", data, FlatSenderConfig { format: FAST.format(), ..FlatSenderConfig::default() }).unwrap();
    let mut rx = RxSession::new();
    rx.start_tune(TuneOptions::default(), 0.0);
    let mut j = String::new();
    for n in 0..6 { let img = render(&file.next_frame().unwrap()); j = rx.push_rgba_at(&rgba(&img), img.width, img.height, n as f64 * 33.3); }
    assert_eq!(state(&j), "not_tuning", "{j}");
    assert!(j.contains("\"received_bytes\":0"), "a file is not received while tuning: {j}");
    assert!(j.contains("\"hint\":null"), "no live hint while tuning: {j}");
    let mut s = tune(1024, 1024);
    let shots = film(&mut s, 0, 30.0, 300.0, 1200.0, 1.0, &|_| false);
    let js = session(&shots, &mut rx);
    assert_eq!(state(js.last().unwrap()), "measuring", "{}", js.last().unwrap());
}

#[test]
fn nothing_seen_after_fifteen_seconds() {
    let mut rx = RxSession::new();
    rx.start_tune(TuneOptions::default(), 0.0);
    let g = rgba(&grey());
    let mut j = String::new();
    for k in 0..=76 { j = rx.push_rgba_at(&g, 1024, 1024, k as f64 * 200.0); if k == 10 { assert_eq!(state(&j), "waiting", "{j}"); } }
    assert_eq!(state(&j), "failed", "{j}");
    assert_eq!(field(tune_obj(&j), "failure"), "nothing_seen");
}

#[test]
fn chroma_on_every_capture_while_tuning() {
    let out = std::env::temp_dir().join("lensbolt_tune_chroma"); let _ = std::fs::remove_dir_all(&out);
    let live = LiveReceiver::new(&out, LiveConfig { workers: 1 }).unwrap();
    live.start_tune(TuneOptions::default());
    for _ in 0..40 { assert!(live.want_chroma()); }
    live.stop_tune();
}
