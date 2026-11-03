//! The live hint replayed on recorded phone statuses (core/tests/data/status, one line a second from the app's
//! session logs): a clean run never hints, a run of blended frames offers Tune, a long poor stretch offers Tune and
//! clears once reading again; small cells offer coarser b/w words that keep the sender's rate.
use lensbolt_core::calib::Config;
use lensbolt_core::flat::{Alphabet, CellFormat};
use lensbolt_core::tune::hint::{bw_share, parse_format, HintSample, HintWatch, LiveHint};

/// The number (or null, false: 0) after `"key":` in `j`, None when the key is missing.
fn num(j: &str, key: &str) -> Option<f64> {
    let at = j.find(&format!("\"{key}\":"))? + key.len() + 3;
    let rest = &j[at..];
    let v = &rest[..rest.find([',', '}']).unwrap_or(rest.len())];
    Some(v.trim().parse().unwrap_or(0.0))
}
/// The string after `"key":`, None for null or missing.
fn text<'a>(j: &'a str, key: &str) -> Option<&'a str> {
    let at = j.find(&format!("\"{key}\":"))? + key.len() + 3;
    let rest = j[at..].strip_prefix('"')?;
    Some(&rest[..rest.find('"')?])
}
fn sample(line: &str) -> HintSample {
    let st = &line[line.find("\"status\":").expect("status")..];
    HintSample {
        t_ms: num(line, "wall_ms").unwrap(), decoded: num(st, "decoded").unwrap() as u64, failed: num(st, "failed").unwrap() as u64,
        format: text(st, "format").and_then(parse_format), cell_px: num(st, "cell_px").unwrap(), codewords_ok: num(st, "codewords_ok").unwrap() as u64,
        codewords: num(st, "codewords").unwrap() as u64, mix: num(st, "mix").unwrap_or(0.0), received_bytes: num(st, "received_bytes").unwrap() as u64,
        stream_size: num(st, "stream_size").unwrap() as u64, complete: text(st, "complete").is_some(), sender_fps: num(st, "sender_fps").unwrap_or(0.0),
    }
}
fn samples(name: &str) -> Vec<HintSample> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/status").join(name);
    std::fs::read_to_string(p).unwrap().lines().filter(|l| !l.trim().is_empty()).map(sample).collect()
}
fn replay_samples(ss: &[HintSample]) -> Vec<Option<LiveHint>> {
    let mut w = HintWatch::new();
    ss.iter().map(|s| w.update(s).cloned()).collect()
}
fn replay(name: &str) -> Vec<Option<LiveHint>> { replay_samples(&samples(name)) }

/// clean.jsonl up to its last second before completing (the tile in view), with the cells shrunk by `scale` and the
/// reads of a poor hold from tick 20 on: codewords 10/111 and the bytes coming in at a tenth of the rate.
fn shrunk(scale: f64, fps: Option<f64>) -> Vec<HintSample> {
    let mut ss = samples("clean.jsonl");
    ss.truncate(ss.iter().position(|s| s.complete).unwrap() - 1);
    let base = ss[20].received_bytes;
    for (i, s) in ss.iter_mut().enumerate().skip(20) {
        s.cell_px *= scale;
        if s.codewords > 0 { s.codewords_ok = 10; }
        s.received_bytes = base + (s.received_bytes - base) / 10;
        if let Some(f) = fps { s.sender_fps = f; }
        let _ = i;
    }
    ss
}

#[test]
fn clean_run_never_hints() { assert!(replay("clean.jsonl").iter().all(|h| h.is_none())); }

#[test]
fn blend_run_offers_tune() {
    let hs = replay("blend.jsonl");
    assert!(hs.iter().any(|h| matches!(h, Some(LiveHint::Tune { .. }))), "{hs:?}");
    assert!(!hs.iter().any(|h| matches!(h, Some(LiveHint::Words { .. }))));
}

#[test]
fn stretch_run_offers_tune_then_clears() {
    let ss = samples("stretch.jsonl");
    let hs = replay_samples(&ss);
    let first = hs.iter().position(|h| h.is_some()).expect("a hint during the stretch");
    assert!(matches!(hs[first], Some(LiveHint::Tune { .. })), "{:?}", hs[first]);
    // reading again: the first second from which every tick reads 111/111
    let back = (0..ss.len()).find(|&i| i > first && ss[i..ss.len().min(i + 8)].iter().all(|s| s.codewords_ok == 111)).expect("reads again");
    let cleared = (back..ss.len()).find(|&i| hs[i].is_none()).expect("cleared");
    assert!(ss[cleared].t_ms - ss[back].t_ms <= 6500.0, "cleared {} ms after reading again", ss[cleared].t_ms - ss[back].t_ms);
    assert!(hs[cleared..].iter().all(|h| h.is_none()));
}

#[test]
fn small_cells_offer_reliable() {
    let hs = replay_samples(&shrunk(0.68, None));
    let Some(Some(LiveHint::Words { words, mode, text, config })) = hs.last() else { panic!("{:?}", hs.last()) };
    assert_eq!(words, "sauce-minivan"); assert_eq!(mode, "Reliable");
    assert_eq!(text, "Stop sending, type sauce-minivan in Tune words, press Start: what has arrived is kept.");
    assert_eq!(*config, Config { pitch: 3, alphabet: Alphabet::Bw, fps: 0 });
    // the same reads without the small cells (the cells are not the cause): Tune
    let hs = replay_samples(&shrunk(1.0, None));
    assert!(matches!(hs.last(), Some(Some(LiveHint::Tune { .. }))), "{:?}", hs.last());
}

#[test]
fn a_60_fps_sender_keeps_its_rate() {
    let hs = replay_samples(&shrunk(0.68, Some(59.8)));
    let Some(Some(LiveHint::Words { words, .. })) = hs.last() else { panic!("{:?}", hs.last()) };
    assert_eq!(Config::from_code(words), Ok(Config { pitch: 3, alphabet: Alphabet::Bw, fps: 1 }));
}

#[test]
fn near_end_and_out_of_view_never_hint() {
    // out of view: decoded stops rising
    let mut ss = shrunk(0.68, None);
    let d = ss[20].decoded;
    for s in ss.iter_mut().skip(20) { s.decoded = d; }
    assert!(replay_samples(&ss).iter().all(|h| h.is_none()), "out of view");
    // complete
    let mut ss = shrunk(0.68, None);
    for s in ss.iter_mut().skip(20) { s.complete = true; }
    assert!(replay_samples(&ss).iter().all(|h| h.is_none()), "complete");
    // under 15 s left at the window's rate
    let mut ss = shrunk(0.68, None);
    for s in ss.iter_mut() { s.stream_size = s.received_bytes + 200_000; }
    assert!(replay_samples(&ss).iter().all(|h| h.is_none()), "near the end");
}

#[test]
fn coarsest_bw_and_colour_offer_tune_only() {
    for (f, scale) in [("5px bw 1024x1024", 0.6 * 5.0 / 2.0), ("4px c8 1024x1024", 1.0)] {
        let mut ss = shrunk(scale, None);
        // a coarser format carries less: its poor hold brings in less again (under the goodput gate)
        let base = ss[20].received_bytes;
        for s in ss.iter_mut() {
            if s.format.is_some_and(|g| g.0 == CellFormat { pitch: 2, alphabet: Alphabet::Bw }) { s.format = parse_format(f); }
            if s.received_bytes > base { s.received_bytes = base + (s.received_bytes - base) / 5; }
        }
        let hs = replay_samples(&ss);
        assert!(matches!(hs.last(), Some(Some(LiveHint::Tune { .. }))), "{f}: {:?}", hs.last());
        assert!(!hs.iter().any(|h| matches!(h, Some(LiveHint::Words { .. }))), "{f}");
    }
}

#[test]
fn bw_share_curve() { assert_eq!(bw_share(1.1), 0.0); assert_eq!(bw_share(1.25), 0.5); assert_eq!(bw_share(1.6), 1.0); }
