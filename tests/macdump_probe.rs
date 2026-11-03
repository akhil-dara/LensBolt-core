//! Probe: cold and tracked reads of dumped phone captures in LB_PROBE_DIR (codewords per capture); skipped without it.
mod common;
use lensbolt_core::flat::{FlatDecoder, FlatHint};
#[test]
fn probe() {
    let Some(dir) = std::env::var_os("LB_PROBE_DIR") else { return };
    let caps = common::captures(std::path::Path::new(&dir));
    let dec = FlatDecoder::new();
    let mut hint: Option<FlatHint> = None;
    for c in &caps {
        let y = c.luma();
        let cold = dec.decode_luma(&y, None).ok().map(|f| { let p = f[0].payload(); (p.codewords_ok, p.codewords_failed, p.codewords_missing) });
        let tracked = dec.decode_luma(&y, hint.as_ref());
        let t = match &tracked { Ok(f) => { let p = f[0].payload(); hint = Some(f[0].hint()); Some((p.codewords_ok, p.codewords_failed, p.codewords_missing, f[0].reads.mix, f[0].reads.edge_px)) } Err(_) => { hint = None; None } };
        let seq = tracked.as_ref().ok().map(|f| (f[0].header.frame_seq, f[0].header.block_id)); eprintln!("{}: seq {:?} cold {:?}  tracked {:?}", c.stem, seq, cold, t);
    }
}
