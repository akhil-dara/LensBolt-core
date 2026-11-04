//! The "soft" captures of 2026-11-02 (`test/phone_fhd2px_soft/dump`: a phone at 1440x1080 filming a 1024
//! tile of 2 px b/w at 1.6-1.9 camera px per cell, sender 30 fps; 60 consecutive captures of a run that read 0-4
//! of 111 codewords for 90 s). They are not soft: every capture but three holds two sender frames blended over
//! the exposure, the header's frame from ~88% of the light at the top of the tile to ~6% at the bottom
//! (docs/2026-11-02-soft-frames.md).
//! The 60 captures at 3 ms (`test/phone_fhd2px_ramp3ms`) show the same ramp: the sender's panel mid-change.
//! - `tracking_survives_a_manifest_straddle`: the gate. A capture whose bottom shows the manifest (12 px
//!   colour) read with the 2 px grid leaves a wrong shift field; tracking from it once read 39 codewords where
//!   the same captures read cold give 323, every capture after it handing the field on.
//! - `the_blend_is_reported`: the read's blend (`mix`) high on the blended captures, low on clean ones
//!   (`test/phone_fhd2px_live`, `test/phone_fhd2px_phase`).
//! What the reader cannot do here: read the frames. A frame needs under ~0.5% of its cells wrong; at the measured
//! share of light per row its cells are 0.9% wrong where it has 0.88, 8.4% at 0.77, 18.6% at 0.58, and every row
//! of a frame is at under 0.85 in both captures that show it over most of the tile (docs/2026-11-02-soft-frames.md).
//! Without the captures (they are not in the repository) the tests say so and pass (`LB_REQUIRE_CAPTURES=1`: they fail).
use lensbolt_core::flat::live::decode_luma_first;
use lensbolt_core::flat::{FlatDecoder, FlatHint};

mod common;
use common::{captures, gate_dir};

/// The 60 captures at auto exposure, and 60 more at 3 ms and ISO 1600 with noise reduction and edge enhancement
/// off (`test/phone_fhd2px_ramp3ms/dump`): the same ramp, so it is the sender's panel mid-change, not the camera.
const SETS: &[&str] = &["test/phone_fhd2px_soft/dump", "test/phone_fhd2px_ramp3ms/dump"];

/// Per set: codewords cold and tracked (the hint from the latest capture that decoded, as on the phone), and the
/// b/w reads' blend (`FlatReads::mix`).
fn run(dir: &std::path::Path) -> (usize, usize, Vec<f32>) {
    let dec = FlatDecoder::new();
    let (mut cold, mut tracked) = (0usize, 0usize);
    let mut mixes = Vec::new();
    let mut hint: Option<FlatHint> = None;
    for cap in captures(dir) {
        let y = cap.luma();
        // a capture the blend pre-check stopped before its cells on either path (no values) counts for neither:
        // the comparison is between reads that were read (a blended capture's 2-4 codewords are what the
        // pre-check gives up on purpose)
        let c = decode_luma_first(&dec, &y, || cap.rgb(), None).ok();
        let t = decode_luma_first(&dec, &y, || cap.rgb(), hint.as_ref()).ok();
        if let Some((v, _)) = &c { if v[0].header.format.alphabet.size() == 2 { mixes.push(v[0].reads.mix); } }
        if let Some((v, _)) = &t { hint = Some(v[0].hint()); }
        let skipped = c.as_ref().is_some_and(|(v, _)| v[0].reads.values.is_empty()) || t.as_ref().is_some_and(|(v, _)| v[0].reads.values.is_empty());
        if skipped { continue; }
        if let Some((v, _)) = &c { cold += v[0].payload().codewords_ok; }
        if let Some((v, _)) = &t { tracked += v[0].payload().codewords_ok; }
    }
    (cold, tracked, mixes)
}

#[test]
fn tracking_survives_a_manifest_straddle() {
    let mut any = false;
    for rel in SETS {
        let Some(dir) = gate_dir(rel) else { continue };
        any = true;
        let (cold, tracked, _) = run(&dir);
        eprintln!("{rel}: codewords cold {cold}, tracked {tracked}");
        // at least one whole frame among the reads compared (the blended captures are skipped before their cells)
        assert!(cold >= 111, "{rel}: cold reads {cold}: the captures changed?");
        assert!(tracked * 100 >= cold * 95, "{rel}: tracked {tracked} against cold {cold}: a wrong tracking field was handed on");
    }
    if !any { eprintln!("skip: no captures"); }
}

/// Clean captures: the 40 of the 2026-10-30 run (`test/phone_fhd2px_live`) and the 60 of the 2026-11-02 run whose
/// camera started in the panel's clean phase (`test/phone_fhd2px_phase`, 3.52 Mbit/s, the phase lock never needed
/// to step). Measured: mix 0.00 on every b/w read but two of the second set (0.02, 0.03).
const CLEAN: &[&str] = &["test/phone_fhd2px_live/dump", "test/phone_fhd2px_phase/dump"];
/// The blend a clean capture stays under: the app's phase-step line (PhaseLock.MIX_HIGH).
const MIX_CLEAN: f32 = 0.15;

/// The blend is seen: the read reports it on the blended captures (the status's `mix`, which the app's phase
/// steps and both receivers' "two frames overlap" line act on), and not on a clean one.
#[test]
fn the_blend_is_reported() {
    for rel in SETS {
        let Some(dir) = gate_dir(rel) else { continue };
        let (_, _, mixes) = run(&dir);
        let high = mixes.iter().filter(|&&m| m >= 0.4).count();
        let clean = mixes.iter().filter(|&&m| m < 0.1).count();
        eprintln!("{rel}: {} b/w reads, mix >= 0.4 on {high}, under 0.1 on {clean}", mixes.len());
        assert!(high * 100 >= mixes.len() * 85, "{rel}: mix >= 0.4 on {high} of {}", mixes.len());
        // the one capture of each set that shows a single frame
        assert!(clean >= 1 && mixes.iter().filter(|&&m| m < 0.1).count() == clean, "{rel}: no clean capture reads under 0.1");
    }
    // ... and stays low where the captures are clean: the app never steps the phase on them
    for rel in CLEAN {
        let Some(dir) = gate_dir(rel) else { continue };
        let (cold, _, mixes) = run(&dir);
        let worst = mixes.iter().copied().fold(0f32, f32::max);
        eprintln!("{rel}: {} b/w reads, codewords cold {cold}, mix at most {worst:.2}", mixes.len());
        assert!(!mixes.is_empty() && worst < MIX_CLEAN, "{rel}: mix {worst} on a clean capture");
    }
}
