//! Time and correctness gate on real phone captures, through the phone's own path: the Y plane into
//! `decode_luma_first` (RGB only for a colour frame), tracking from the previous decoded capture, then the
//! capture's payload through a fresh `Stitcher`; one core, release, per capture the fastest of five passes.
//! - 1px tier: a phone at Camera: Max (4096x3072 YUV_420_888) filming a 1024 tile of 1 px b/w
//!   (`test/phone_max1px_live/dump`, 8 captures in capture order).
//! - Full HD 2px tier: the same phone at 1440x1080 filming a 1024 tile of 2 px b/w at 30 fps
//!   (`test/phone_fhd2px_live/dump`, 40 consecutive captures of a live transfer, the manifest among them).
//! - Full HD colour guard: 1440x1080 filming a 1024 tile of 4 px c8 (`test/phone_fhd`, 3 captures).
//!
//! Gate: every capture decodes to the codewords recorded below, and the median per-capture cost (Y copy +
//! decode + stitcher payload) over the tracked captures is within budget: `LB_BUDGET_MS` /
//! `LB_BUDGET_FHD2_MS` / `LB_BUDGET_FHD_MS`, default the idle measurement + 10%. Measured in a child run of this test (up to
//! three while over budget, the fastest counts), which also prints a per-stage breakdown (`LB_PROFILE`):
//! `cargo test --release --test phone_max1px -- --nocapture`.
//! The 1px and 2px fixtures are not in the repository: without them those tests say so and pass
//! (`LB_REQUIRE_CAPTURES=1`: they fail instead).
use lensbolt_core::flat::live::decode_luma_first;
use lensbolt_core::flat::{FlatDecoder, FlatHint, FlatRead, Stitcher};
use std::path::PathBuf;
use std::time::Instant;

mod common;
use common::{captures, Capture};

/// One measurement at a time: cargo runs a binary's tests in parallel.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
/// Set (to the test's name) in the child run that measures.
const CHILD: &str = "LB_GATE_CHILD";
const RUNS: usize = 5;
/// Child runs at most per gate (see `gate`).
const CHILD_RUNS: usize = 3;
const ALL_444: &[(usize, usize)] = &[(444, 444)];
/// The blend (`FlatReads::mix`) a clean capture stays under: the app's phase-step line (PhaseLock.MIX_HIGH).
/// Measured 2026-11-02: 0.00 on every b/w capture of the 1 px and Full HD 2 px tiers.
const MIX_CLEAN: f32 = 0.15;

struct Tier {
    test: &'static str,
    dir: &'static str,
    budget_env: &'static str,
    budget_ms: f64,
    /// Per capture in capture order, None = no decode: the stitcher's (codewords ok, codewords total) per
    /// frame that completed a packet (what the phone's receiver gets)...
    expected: &'static [Option<&'static [(usize, usize)]>],
    /// ...and each frame read's own (codewords ok, codewords total), packets complete or not.
    reads: &'static [Option<&'static [(usize, usize)]>],
}

/// Baseline 2026-10-29, development machine (i7-1255U laptop) on AC power while charging from 16%,
/// otherwise idle, one core: single child runs 188-235 ms, median 207 ms per capture (Y copy + decode ~170,
/// stitcher payload ~31); budget = 207 ms + 10%, against the fastest of up to three runs.
/// The gate on a sparse sample (2026-10-30; gate 13-22 -> 0.8-1.1 ms): back-to-back base/head medians
/// 201.3/192.8 and 212.2/185.4 ms; budget = 188 ms + 10%.
/// The fast reader's refinement cut (2026-10-30; refine 47.8-49.5 -> 14.7-15.1 ms, two commits): base/head
/// medians 179.1/161.6, 182.4/156.3 then 158.3/148.9, 158.2/147.8 ms; budget = 148 ms + 10%.
/// The fast reader's levels and pattern decision (2026-10-30; tags + levels 21.6 -> 8.1-8.5 ms, pattern
/// decision 20.6-21.6 -> 6.6-7.2 ms, three commits): base/head medians 138.8/130.0, 140.4/128.1, then
/// 128.7/125.6, 127.8/124.9, then 127.5/110.8, 127.4/110.8 ms; at the head 111.6, 110.9 ms; budget =
/// 111.3 ms + 10%.
/// The payload's error correction and the fast reader's cells stage (2026-10-30; extract codewords
/// 19.9-20.4 -> 5.4-6.1 ms, cells 15.9-17.5 -> 3.1-3.2 ms, five commits): final base/head medians
/// 111.7/80.0, 111.4/77.1 ms. Budget from the slowest head run + 10% (not the median: a review run on the
/// same machine read 87.5 ms): head runs 78.7, 79.6, 80.0, 77.1, 78.9, 78.9, 80.8, 80.7 and the review's
/// 87.5 ms; budget = 87.5 ms + 10%.
const MAX1PX: Tier = Tier {
    test: "phone_max1px_tracked_captures_within_budget",
    dir: "test/phone_max1px_live/dump",
    budget_env: "LB_BUDGET_MS",
    budget_ms: 96.3,
    // every capture, the first one cold: all 444 codewords
    expected: &[Some(ALL_444); 8],
    reads: &[Some(ALL_444); 8],
};

/// Baseline 2026-10-29, same machine and conditions: single child runs 112-134 ms, median 115 ms per
/// capture (Y copy + decode ~113, stitcher payload ~2); budget = 115 ms + 10%. The tile is colour (c8):
/// decode_luma's header read, then the RGB conversion and the colour reader. The codewords are today's
/// (the Full HD colour path is not at its target yet, see `phone_fhd.rs`): a change shows here; record the
/// new counts when it is an improvement.
const FHD: Tier = Tier {
    test: "phone_fhd_tracked_captures_within_budget",
    dir: "test/phone_fhd",
    budget_env: "LB_BUDGET_FHD_MS",
    budget_ms: 127.0,
    // the reads' own counts carry the accuracy signal. 2026-11-04: the colour read starts from the luma pass's
    // geometry (no second search of the frame: 120-230 -> 66-80 ms a capture); capture 3 tracked reads 40 of 83
    // where the RGB path's own refit read 44, and its stitched read now completes a packet
    expected: &[Some(&[]), Some(&[]), Some(&[(39, 83)])],
    reads: &[Some(&[(2, 83)]), Some(&[(3, 83)]), Some(&[(40, 83)])],
};

/// A 2 px b/w data frame read whole, and the manifest frame (c8, 12 px) read whole.
const W: Option<&[(usize, usize)]> = Some(&[(111, 111)]);
const M: Option<&[(usize, usize)]> = Some(&[(9, 9)]);

/// Baseline 2026-10-29, same machine on AC power (charging 41-44%), otherwise idle, one core: single child
/// runs 35-39 ms, median 37.5 ms per capture (Y copy + decode ~33.5, stitcher payload ~4); budget = 37.5 ms
/// + 10%, against the fastest of up to three runs.
/// The gate on a sparse sample (2026-10-30; gate 2.0-3.1 -> 0.7-1.0 ms): back-to-back base/head medians
/// 35.5/35.4 and 36.8/34.2 ms; budget = 35.4 ms + 10%.
/// The fast reader's refinement cut (2026-10-30; refine 8.3-8.5 -> 2.5 ms): base/head medians 32.9/27.3,
/// 31.6/27.5 then 27.5/27.1, 26.9/26.4 ms; budget = 26.8 ms + 10%.
/// The fast reader's levels and pattern decision (2026-10-30; tags + levels 5.5 -> 2.0 ms, pattern decision
/// 4.1 -> 1.5 ms): base/head medians 25.6/22.6, 25.5/23.4, then 23.1/22.4, 23.3/21.5, then 22.7/19.9,
/// 22.2/19.7 ms; at the head 19.7, 19.0 ms; budget = 19.35 ms + 10%.
/// The payload's error correction and the fast reader's cells stage (2026-10-30; extract codewords 1.7 ->
/// 0.7 ms, cells 2.9-3.3 -> 0.7 ms): final base/head medians 19.5/14.4, 19.0/14.3 ms. Budget from the slowest
/// head run + 10%: head runs 14.5, 14.4, 14.4, 14.3, 14.5, 15.0, 14.7, 14.9 and the review's 15.8 ms; budget
/// = 15.8 ms + 10%.
/// The fast reader's local equaliser (2026-10-31; every b/w capture whole, before 5 short of 111; degraded
/// captures: tests/fhd2px_floor.rs): back-to-back base/head medians 14.7/17.8, 15.4/17.5, 15.4/18.0,
/// 15.9/17.9, 15.7/18.7, 15.7/17.7 ms (the machine shared with other jobs; equalise 2.5 ms in the stage
/// profile). Budget from the slowest head run + 10%: 18.7 ms + 10%.
/// Speed Task 8 (tracking check, blend and edge width in the read; 2026-11-02, the machine never idle):
/// back-to-back base 4a6e3d5 / head medians 17.8/18.5, 20.9/17.9, 17.8/18.0, then 22.3/20.4, 19.4/19.7,
/// 19.6/20.5 ms (base and head alike). Budget from the slowest head run + 10%: 20.5 ms + 10%.
/// The budgets are from runs on a machine shared with other jobs: a loaded laptop reads 20-30% slower (2026-11-02:
/// 25.6, 24.5 then 20.3 ms in three child runs minutes apart), which the three child runs absorb only when the
/// load passes; re-measure idle before tightening a budget. The fixtures are not in the repository.
/// The tier the Full HD phones run: a phone at 1440x1080 filming a 1024 tile of 2 px b/w at 30 fps,
/// hand-held (`test/phone_fhd2px_live/dump`, 40 consecutive captures of a live transfer). Captures 19 and
/// 20 show the transfer's manifest frame (c8 at 12 px), which decode_luma hands to the colour path
/// (`NeedsColour`), as on the phone.
const FHD2: Tier = Tier {
    test: "phone_fhd2px_tracked_captures_within_budget",
    dir: "test/phone_fhd2px_live/dump",
    budget_env: "LB_BUDGET_FHD2_MS",
    budget_ms: 22.6,
    // every b/w capture whole since the local equaliser (2026-10-31; before: 107, 107, 110, 108 and 110 of 111
    // on captures 1, 2, 3, 6 and 8, reads 108, 108, 110, 109, 110)
    expected: &[W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, M, M, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    reads: &[W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, M, M, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
};

#[test]
fn phone_max1px_tracked_captures_within_budget() { gate(&MAX1PX); }

#[test]
fn phone_fhd2px_tracked_captures_within_budget() { gate(&FHD2); }

#[test]
fn phone_fhd_tracked_captures_within_budget() { gate(&FHD); }


fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 { f64::NAN } else if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

/// One phone decode as the worker runs it: the Y plane copied out of the camera buffer, luma first, RGB
/// (converted here) only for a colour frame.
fn decode(dec: &FlatDecoder, cap: &Capture, hint: Option<&FlatHint>) -> Option<Vec<FlatRead>> {
    let y = cap.luma();
    decode_luma_first(dec, &y, || cap.rgb(), hint).ok().map(|(frs, _)| frs)
}

/// The capture's payload as the phone's collector gets it: a fresh stitcher per capture (a long-lived one
/// skips codewords it already has), (codewords ok, codewords total) per frame that completed a packet.
fn stitch(frs: &[FlatRead]) -> Vec<(usize, usize)> {
    Stitcher::new().add_capture(frs).iter().map(|(_, p)| (p.codewords_ok, p.codewords_ok + p.codewords_failed + p.codewords_missing)).collect()
}

/// Per capture: None = no decode; Some(per frame that completed a packet: (codewords ok, total)).
type Codewords = Option<Vec<(usize, usize)>>;
/// One child run, per capture in capture order: stem, tracked, fastest decode ms, fastest payload ms,
/// stitcher codewords, read codewords.
type Timed = Vec<(String, bool, f64, f64, Codewords, Codewords)>;

/// `Codewords` on the child's `@@ timed` line: `-` no decode, `none` no frame, else `ok/all,ok/all`.
fn encode(c: &Codewords) -> String {
    match c {
        None => "-".to_string(),
        Some(v) if v.is_empty() => "none".to_string(),
        Some(v) => v.iter().map(|(a, b)| format!("{a}/{b}")).collect::<Vec<_>>().join(","),
    }
}
fn parse(s: &str) -> Codewords {
    match s {
        "-" => None,
        "none" => Some(Vec::new()),
        s => Some(s.split(',').map(|p| { let (a, b) = p.split_once('/').unwrap(); (a.parse().unwrap(), b.parse().unwrap()) }).collect()),
    }
}

/// Runs the measuring child once: its timed captures, and its stderr for the stage profile.
fn run_child(tier: &Tier) -> (Timed, String) {
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args([tier.test, "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD, tier.test).env("LB_THREADS", "1").env_remove("LB_PROFILE").output().expect("the measuring child run");
    let text = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "{}: the measuring child run failed:\n{text}", tier.test);
    let mut caps: Timed = Vec::new();
    for l in text.lines().filter_map(|l| l.strip_prefix("@@ timed ")) {
        let f: Vec<&str> = l.split_whitespace().collect();
        caps.push((f[0].to_string(), f[1] == "1", f[2].parse().unwrap(), f[3].parse().unwrap(), parse(f[4]), parse(f[5])));
    }
    (caps, text)
}

/// (captures, median decode, median payload, median per-capture total) over the tracked decoded captures.
fn medians(caps: &Timed) -> (usize, f64, f64, f64) {
    let (mut d, mut p, mut t) = (Vec::new(), Vec::new(), Vec::new());
    for (_, tracked, dm, pm, cw, _) in caps { if *tracked && cw.is_some() { d.push(*dm); p.push(*pm); t.push(dm + pm); } }
    (t.len(), median(&mut d), median(&mut p), median(&mut t))
}

/// The measurement runs in a fresh child process of this test binary (one test, one thread): measured on
/// the development machine (a hybrid-core laptop), a test thread that first waited for the other test's
/// lock ran every stage ~1.4x slower, while the same test run alone did not.
fn gate(tier: &Tier) {
    let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join(tier.dir);
    let has_captures = std::fs::read_dir(&dir).map(|d| d.filter_map(|e| e.ok()).any(|e| e.path().extension().is_some_and(|x| x == "y"))).unwrap_or(false);
    if !has_captures {
        assert!(!common::captures_required(), "{}: no captures in {} on this machine, and LB_REQUIRE_CAPTURES=1", tier.test, dir.display());
        eprintln!("skip {}: no captures in {} on this machine", tier.test, dir.display());
        return;
    }
    if std::env::var(CHILD).ok().as_deref() == Some(tier.test) { child(&captures(&dir)); return; }
    let budget = std::env::var(tier.budget_env).ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(tier.budget_ms);
    let want = |e: &[Option<&[(usize, usize)]>]| -> Vec<Codewords> { e.iter().map(|e| e.map(|v| v.to_vec())).collect() };
    let (expected, expected_reads) = (want(tier.expected), want(tier.reads));

    // up to CHILD_RUNS child runs, the next only while over budget (a healthy suite pays one): the gate
    // takes the fastest run, so a slow spell of the machine (single runs of one build spread up to ~20%
    // on the development machine) does not fail it, while a real slowdown fails every run
    let mut best: Option<(f64, Timed, String)> = None;
    for k in 1..=CHILD_RUNS {
        let (caps, text) = run_child(tier);
        let got: Vec<Codewords> = caps.iter().map(|c| c.4.clone()).collect();
        let got_reads: Vec<Codewords> = caps.iter().map(|c| c.5.clone()).collect();
        assert_eq!(got, expected, "{}: stitched codewords per capture changed", tier.test);
        assert_eq!(got_reads, expected_reads, "{}: read codewords per capture changed", tier.test);
        let (n, d, p, t) = medians(&caps);
        assert!(n > 0, "{}: no tracked capture decoded", tier.test);
        eprintln!("{}: run {k}: tracked median {t:.1} ms per capture (decode {d:.1} + payload {p:.1}) over {n} captures", tier.test);
        if best.as_ref().is_none_or(|b| t < b.0) { best = Some((t, caps, text)); }
        if t <= budget { break; }
    }
    let (t, caps, text) = best.unwrap();
    for (stem, tracked, d, p, cw, rcw) in &caps {
        eprintln!("{stem}: {} {d:>6.1} ms decode + {p:>5.1} ms payload, stitched {} read {}", if *tracked { "tracked" } else { "cold   " }, encode(cw), encode(rcw));
    }
    let (n, d, p, _) = medians(&caps);
    eprintln!("{}: tracked median {t:.1} ms per capture (decode {d:.1} + payload {p:.1}) over {n} captures, budget {budget:.1} ms", tier.test);
    eprintln!("{}: per-stage medians over the tracked captures (LB_PROFILE, fastest of {RUNS} passes):", tier.test);
    for (name, ms, k) in profile(&text) { eprintln!("  {ms:>7.1} ms  {name}{}", if k != n { format!(" ({k} of {n} captures)") } else { String::new() }); }
    assert!(t <= budget, "{}: tracked median {t:.1} ms per capture (fastest of {CHILD_RUNS} runs) over the {budget:.1} ms budget ({})", tier.test, tier.budget_env);
}

/// The child run. RUNS passes over the captures in order, the hint the last decoded capture's as on the
/// phone; per capture the fastest pass (passes rather than back-to-back repeats: a disturbance of a second
/// or two costs a few samples, not every sample of one capture). Then the same passes with `LB_PROFILE`
/// marks. Everything goes to stderr for the parent.
fn child(caps: &[Capture]) {
    // single-core cost: the phone decodes one capture per worker thread
    std::env::set_var("LB_THREADS", "1");
    let dec = FlatDecoder::new();
    let (mut got, mut reads): (Vec<Codewords>, Vec<Codewords>) = (Vec::new(), Vec::new());
    let mut tracked: Vec<bool> = Vec::new();
    let (mut best_dec, mut best_pay) = (vec![f64::MAX; caps.len()], vec![f64::MAX; caps.len()]);
    for pass in 0..RUNS {
        let mut hint: Option<FlatHint> = None;
        for (i, cap) in caps.iter().enumerate() {
            let t = Instant::now();
            let frs = decode(&dec, cap, hint.as_ref());
            best_dec[i] = best_dec[i].min(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            let cw: Codewords = frs.as_deref().map(stitch);
            best_pay[i] = best_pay[i].min(if frs.is_some() { t.elapsed().as_secs_f64() * 1e3 } else { 0.0 });
            if pass == 0 {
                // the reads' own codewords, once and untimed (the phone never decodes a read alone)
                reads.push(frs.as_ref().map(|frs| frs.iter().map(|f| f.payload()).map(|p| (p.codewords_ok, p.codewords_ok + p.codewords_failed + p.codewords_missing)).collect()));
                got.push(cw); tracked.push(hint.is_some());
                // these captures hold one sender frame each: the read's blend stays under the app's phase-step line
                for f in frs.iter().flatten().filter(|f| f.header.format.alphabet.size() == 2) {
                    assert!(f.reads.mix < MIX_CLEAN, "{}: mix {} on a clean capture", cap.stem, f.reads.mix);
                }
            } else { assert_eq!(cw, got[i], "{}: pass {pass} read differently", cap.stem); }
            if let Some(frs) = frs { hint = Some(frs[0].hint()); }
        }
    }
    for (i, cap) in caps.iter().enumerate() {
        eprintln!("@@ timed {} {} {:.3} {:.3} {} {}", cap.stem, tracked[i] as u8, best_dec[i], best_pay[i], encode(&got[i]), encode(&reads[i]));
    }
    std::env::set_var("LB_PROFILE", "1");
    for _ in 0..RUNS {
        let mut hint: Option<FlatHint> = None;
        for (i, cap) in caps.iter().enumerate() {
            eprintln!("@@ capture {i} tracked {}", hint.is_some() as u8);
            let t = Instant::now();
            let y = cap.luma();
            eprintln!("  [y] {:>7.1} ms  copy from plane", t.elapsed().as_secs_f64() * 1e3);
            let rgb = || { let t = Instant::now(); let img = cap.rgb(); eprintln!("  [rgb] {:>7.1} ms  from yuv", t.elapsed().as_secs_f64() * 1e3); img };
            let frs = decode_luma_first(&dec, &y, rgb, hint.as_ref()).ok().map(|f| f.0);
            eprintln!("  [decode] {:>7.1} ms  total", t.elapsed().as_secs_f64() * 1e3);
            if let Some(frs) = frs {
                let t = Instant::now();
                let _ = stitch(&frs);
                eprintln!("  [payload] {:>7.1} ms  total", t.elapsed().as_secs_f64() * 1e3);
                hint = Some(frs[0].hint());
            }
        }
    }
}

/// The child's stage marks (cumulative ms per `[tag]`) as per-stage times: the fastest pass per capture,
/// the median over the tracked captures. (stage, median ms, captures it appeared in), in the order a
/// tracked capture shows them.
fn profile(text: &str) -> Vec<(String, f64, usize)> {
    use std::collections::{BTreeMap, HashMap};
    let mut order: Vec<String> = Vec::new();
    // per (capture, stage): the fastest pass
    let mut best: BTreeMap<(usize, String), f64> = BTreeMap::new();
    let mut tracked: Vec<bool> = Vec::new();
    let mut cur: Option<usize> = None;
    let mut last: HashMap<String, f64> = HashMap::new();
    let mut run: HashMap<String, f64> = HashMap::new();
    let flush = |cur: Option<usize>, run: &mut HashMap<String, f64>, best: &mut BTreeMap<(usize, String), f64>| {
        let Some(c) = cur else { return };
        // decode time between the marks (setup after the last mark, stages that print none)
        if let Some(&total) = run.get("decode: total") {
            // "eq:" lines are sub-stages of "fast: equalise": not summed again
            let marked: f64 = run.iter().filter(|(k, _)| !["decode:", "payload:", "stitch:", "extract:", "eq:"].iter().any(|t| k.starts_with(t))).map(|(_, v)| v).sum();
            run.insert("decode: not marked".into(), total - marked);
        }
        for (k, v) in run.drain() { let e = best.entry((c, k)).or_insert(f64::MAX); *e = e.min(v); }
    };
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("@@ capture ") {
            flush(cur, &mut run, &mut best);
            let f: Vec<&str> = rest.split_whitespace().collect();
            let c: usize = f[0].parse().unwrap();
            if tracked.len() <= c { tracked.resize(c + 1, false); }
            tracked[c] = f[2] == "1";
            cur = Some(c);
            last.clear();
            continue;
        }
        let Some(c) = cur else { continue };
        let l = line.trim();
        let Some(close) = l.strip_prefix('[').and_then(|r| r.find(']')) else { continue };
        let tag = &l[1..close + 1];
        let rest = l.get(close + 2..).unwrap_or("").trim();
        let Some((num, what)) = rest.split_once(" ms") else { continue };
        let Ok(cum) = num.trim().parse::<f64>() else { continue };
        // a tag's marks count from its own start; a smaller value is a new call of that stage group
        let prev = last.get(tag).copied().filter(|&p| p <= cum).unwrap_or(0.0);
        last.insert(tag.to_string(), cum);
        let name = format!("{tag}: {}", what.trim());
        if tracked[c] && !order.contains(&name) {
            if name == "decode: total" { order.push("decode: not marked".into()); }
            order.push(name.clone());
        }
        *run.entry(name).or_insert(0.0) += cum - prev;
    }
    flush(cur, &mut run, &mut best);
    order.into_iter().map(|name| {
        let mut v: Vec<f64> = best.iter().filter(|((c, k), _)| *k == name && tracked[*c]).map(|(_, &ms)| ms).collect();
        let k = v.len();
        (name, median(&mut v), k)
    }).filter(|s| s.2 > 0).collect()
}


