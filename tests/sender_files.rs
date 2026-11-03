//! The browser sender's needs from the core: a file (or a batch) kept once in memory and shared by every sender made
//! for it, sent exactly as the owned bytes would be; and what finish words mean in files, on the sender's screen.
//!     cargo test --release --test sender_files
use lensbolt_core::file::{files_text, BatchSource, CHUNK_BYTES, SEGMENT_BYTES};
use lensbolt_core::finish::{encode, parts_text};
use lensbolt_core::flat::sender::{FlatSender, FlatSenderConfig};
use lensbolt_core::flat::{Alphabet, CellFormat};
use std::sync::Arc;

fn noise(len: usize, mut s: u64) -> Vec<u8> { (0..len).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s as u8 }).collect() }
fn cfg() -> FlatSenderConfig {
    FlatSenderConfig { width: 1024, height: 1024, format: CellFormat { pitch: 3, alphabet: Alphabet::Bw }, repair_offset: Some(77), ..FlatSenderConfig::default() }
}

#[test]
fn a_shared_file_sends_the_frames_owned_bytes_send() {
    let data = noise(2_500_000, 5);
    let shared = Arc::new(data.clone());
    let mut a = FlatSender::from_bytes("photo.jpg", data, cfg()).unwrap();
    let mut b = FlatSender::from_shared("photo.jpg", shared.clone(), cfg()).unwrap();
    assert_eq!(a.manifest().encode(), b.manifest().encode());
    assert_eq!(a.config().transfer_id, b.config().transfer_id, "the same transfer id: a reload resumes");
    for n in 0..80 {
        let (x, y) = (a.next_frame().unwrap(), b.next_frame().unwrap());
        assert!(x.header == y.header && x.values == y.values, "frame {n}");
    }
    // finish words too
    let units = a.manifest().finish_units();
    let code = encode(a.config().transfer_id, units, &[1]).unwrap();
    a.finish(&code).unwrap(); b.finish(&code).unwrap();
    for n in 0..40 { let (x, y) = (a.next_frame().unwrap(), b.next_frame().unwrap()); assert!(x.values == y.values, "repair frame {n}"); }
    assert_eq!(Arc::strong_count(&shared), 2, "held, not copied");
}

#[test]
fn finish_unit_bytes_follow_the_finish_units() {
    let one = FlatSender::from_bytes("a", noise(3_000_000, 1), cfg()).unwrap();
    assert_eq!((one.manifest().finish_units(), one.manifest().finish_unit_bytes()), (3, CHUNK_BYTES), "one segment: chunks");
    let two = FlatSender::from_bytes("b", noise(17_000_000, 2), cfg()).unwrap();
    assert_eq!((two.manifest().finish_units(), two.manifest().finish_unit_bytes()), (2, SEGMENT_BYTES), "two segments: segments");
}

#[test]
fn finish_words_name_the_files_in_their_parts_on_the_sender() {
    // 30 files of 60..390 KB, about 6.5 MB: one segment, finish units are its chunks
    let files: Vec<(String, Arc<Vec<u8>>)> = (0..30).map(|i| (format!("pic-{i:02}.jpg"), Arc::new(noise(60_000 + 11_000 * i, 100 + i as u64)))).collect();
    let mut tx = FlatSender::from_batch("", files.iter().map(|(n, d)| (n.clone(), BatchSource::Shared(d.clone()))).collect(), cfg()).unwrap();
    let (m, idx) = (tx.manifest().clone(), tx.batch_index().unwrap().clone());
    assert_eq!(m.finish_unit_bytes(), CHUNK_BYTES);
    for unit in [0u32, 3, 6] {
        let code = encode(tx.config().transfer_id, m.finish_units(), &[unit]).unwrap();
        let units = tx.finish(&code).unwrap();
        assert_eq!(units, vec![unit]);
        // the files whose stored bytes overlap the chunk, worked out here from the list itself
        let (lo, hi) = (unit as u64 * CHUNK_BYTES, (unit as u64 + 1) * CHUNK_BYTES);
        let want: Vec<usize> = idx.entries.iter().enumerate().filter(|(_, e)| e.stored_size > 0 && e.offset < hi && e.offset + e.stored_size > lo).map(|(i, _)| i).collect();
        assert!(!want.is_empty());
        assert_eq!(idx.files_in_units(m.finish_unit_bytes(), &units), want, "unit {unit}");
        let names: Vec<&str> = want.iter().take(2).map(|&i| idx.entries[i].name.as_str()).collect();
        assert_eq!(tx.units_text(&units), files_text(want.len(), &names), "unit {unit}");
        assert!(tx.units_text(&units).starts_with(&format!("{} file", want.len())), "{}", tx.units_text(&units));
    }
    // a unit naming no file (past the stream) falls back to the parts
    assert_eq!(tx.units_text(&[40]), parts_text(&[40]));
    // one file: the parts, as before
    let one = FlatSender::from_bytes("x.bin", noise(3_000_000, 9), cfg()).unwrap();
    assert_eq!(one.units_text(&[0, 2]), "2 parts: 1, 3");
}
