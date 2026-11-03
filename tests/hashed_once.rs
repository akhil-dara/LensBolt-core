//! The browser sender hashes each file once, while it is read (`FileHasher`, fed the pieces as they come): the
//! sender made at Start takes those hashes instead of reading the file again, and its manifest is exactly the one it
//! would have computed. Hashes that do not fit the bytes are not trusted (computed again).
use lensbolt_core::file::{BatchSource, FileHasher, Prepared};
use lensbolt_core::flat::{FlatSender, FlatSenderConfig};
use std::sync::Arc;

fn bytes(n: usize, seed: u64) -> Vec<u8> { let mut s = seed | 1; (0..n).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s as u8 }).collect() }

#[test]
fn hashes_from_the_pieces_give_the_same_manifest() {
    // empty, small, one byte past a 16 MiB segment, two segments and a bit
    for &n in &[0usize, 1, 70_001, (16 << 20) + 1, (33 << 20) + 5] {
        let data = Arc::new(bytes(n, n as u64 + 3));
        let mut h = FileHasher::new();
        let mut at = 0;
        for k in 1.. { if at >= n { break; } let end = (at + k * 999_983).min(n); h.update(&data[at..end]); at = end; }
        let hashes = h.finish();
        let want = Prepared::from_shared("f.bin", data.clone()).unwrap().manifest.clone();
        assert_eq!(hashes.file, want.file_sha256, "{n} bytes: the file's hash");
        let got = Prepared::from_shared_hashed("f.bin", data.clone(), hashes.clone()).unwrap().manifest.clone();
        assert_eq!(got, want, "{n} bytes: the manifest from the hashes differs");
        // a sender made from them sends the very frames
        let cfg = FlatSenderConfig { transfer_id: 0, ..FlatSenderConfig::default() };
        let (mut a, mut b) = (FlatSender::from_shared("f.bin", data.clone(), cfg.clone()).unwrap(), FlatSender::from_shared_hashed("f.bin", data.clone(), hashes, cfg).unwrap());
        for _ in 0..3 { assert_eq!(a.next_frame().unwrap().header, b.next_frame().unwrap().header, "{n} bytes"); }
    }
}

#[test]
fn hashes_that_do_not_fit_are_computed_again() {
    let data = Arc::new(bytes(50_000, 9));
    let mut h = FileHasher::new();
    h.update(&data[..40_000]);
    let short = h.finish();
    let got = Prepared::from_shared_hashed("f.bin", data.clone(), short).unwrap().manifest.clone();
    assert_eq!(got, Prepared::from_shared("f.bin", data).unwrap().manifest.clone(), "hashes of other bytes: computed from the bytes");
}

#[test]
fn a_batch_of_hashed_files_is_the_same_batch() {
    let files: Vec<(String, Arc<Vec<u8>>)> = (0..5).map(|i| (format!("p{i}.jpg"), Arc::new(bytes(10_000 + 30_000 * i, i as u64)))).collect();
    let plain = Prepared::batch("", files.iter().map(|(n, d)| (n.clone(), BatchSource::Shared(d.clone()))).collect()).unwrap();
    let hashed = Prepared::batch("", files.iter().map(|(n, d)| {
        let mut h = FileHasher::new(); h.update(d);
        (n.clone(), BatchSource::Hashed(d.clone(), h.finish().file))
    }).collect()).unwrap();
    assert_eq!(hashed.manifest, plain.manifest, "the batch's manifest");
    assert_eq!(hashed.index, plain.index, "the file list, each file's hash in it");
}
