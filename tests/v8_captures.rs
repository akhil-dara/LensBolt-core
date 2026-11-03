//! The stored real phone captures (test/phone_*) are of a version-8 sender: this version's receiver rejects
//! them at their header, as `OtherVersion(8)` (the user is told the sender is another version), and reads
//! them only with the diagnostics flag the capture tests set (`ACCEPT_V8_HEADERS`, never set by a receiver).
//! One test in this binary: the flag is process-wide.
use lensbolt_core::flat::header::ACCEPT_V8_HEADERS;
use lensbolt_core::flat::live::decode_luma_first;
use lensbolt_core::flat::{DecodeError, FlatDecoder};
use std::sync::atomic::Ordering;

mod common;
use common::{capture_dir, captures};

#[test]
fn stored_v8_captures_read_only_with_the_diagnostics_flag() {
    let dec = FlatDecoder::new();
    let mut tiers = 0;
    for rel in ["test/phone_fhd", "test/phone_fhd2px_live/dump", "test/phone_max1px_live/dump"] {
        let Some(dir) = capture_dir(rel) else { eprintln!("skip {rel}: no captures on this machine"); continue };
        let caps = captures(&dir); // turns the flag on, as for every capture test
        tiers += 1;
        let read = |accept: bool| {
            ACCEPT_V8_HEADERS.store(accept, Ordering::Relaxed);
            caps.iter().map(|c| decode_luma_first(&dec, &c.luma(), || c.rgb(), None)).collect::<Vec<_>>()
        };
        let (off, on) = (read(false), read(true));
        let decoded = on.iter().filter(|r| r.is_ok()).count();
        assert!(decoded >= 1, "{rel}: no capture reads even with the flag");
        for (c, (a, b)) in caps.iter().zip(off.iter().zip(&on)) {
            if b.is_ok() { assert_eq!(a.as_ref().err(), Some(&DecodeError::OtherVersion(8)), "{rel} {}: a v8 capture read without the flag, or not reported as another version", c.stem); }
        }
        eprintln!("{rel}: {decoded} of {} captures read with the flag, none without", caps.len());
    }
    // the captures live outside the published crate: without them the test says so and passes
    if tiers == 0 { assert!(!common::captures_required(), "no v8 captures on this machine, and LB_REQUIRE_CAPTURES=1"); eprintln!("skip: no v8 captures on this machine"); }
}
