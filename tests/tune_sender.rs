//! The Tune carousel on the sender: a fixed 256-frame loop keyed by the frame number in every header, a
//! generated payload whose manifest says tune, and the rate each frame is shown at.
use lensbolt_core::file::TuneInfo;
use lensbolt_core::flat::{FlatSender, FlatSenderConfig, MANIFEST_EVERY_FRAMES};
use lensbolt_core::header::FrameType;
use lensbolt_core::tune::schedule::{shown_fps, slot, SCHEDULE_VERSION, TUNE_PAYLOAD_BYTES};

#[test]
fn tune_sender_follows_the_schedule() {
    let mut s = FlatSender::tune(FlatSenderConfig::default(), true).unwrap();
    for _ in 0..7 { s.hold_manifest_frame(); assert_eq!(s.tune_frame_fps(), Some(30)); }
    for n in 0..600u64 {
        let (h, _) = s.next_packets().unwrap();
        let seq = ((n + 7) & 0xFF) as u8;
        assert_eq!(h.frame_seq, seq);
        if n % MANIFEST_EVERY_FRAMES as u64 == 0 { assert_eq!(h.frame_type, FrameType::Manifest) }
        else { assert_eq!(h.frame_type, FrameType::Data); assert_eq!(h.format, slot(seq).format, "frame {n} seq {seq}") }
        assert_eq!(s.tune_frame_fps(), Some(shown_fps(slot(seq).fps, true)));
    }
    // the composed frames carry the same formats
    let f = s.next_frame().unwrap();
    assert_eq!(f.header.format, f.grid.format());
}

#[test]
fn tune_payload_is_fixed() {
    let a = FlatSender::tune(FlatSenderConfig::default(), false).unwrap();
    let b = FlatSender::tune(FlatSenderConfig { repair_offset: Some(99), ..FlatSenderConfig::default() }, false).unwrap();
    assert_eq!(a.config().transfer_id, b.config().transfer_id);
    assert_eq!(a.manifest().tune, Some(TuneInfo { schedule: SCHEDULE_VERSION, display_60: false }));
    assert_eq!(a.manifest().stream_size as usize, TUNE_PAYLOAD_BYTES);
    assert_eq!(a.manifest().filename, "LensBolt tune");
    assert!(!a.manifest().compressed && a.manifest().batch.is_none());
    // 60 fps segments drawn at 30 on a slower screen
    let mut s = FlatSender::tune(FlatSenderConfig::default(), false).unwrap();
    let mut rates = std::collections::BTreeSet::new();
    for _ in 0..256 { s.next_packets().unwrap(); rates.insert(s.tune_frame_fps().unwrap()); }
    assert_eq!(rates.into_iter().collect::<Vec<_>>(), vec![20, 30]);
    // a short tile (a window refit) still draws every segment
    let mut s = FlatSender::tune(FlatSenderConfig { width: 1024, height: 640, ..FlatSenderConfig::default() }, true).unwrap();
    for n in 0..256u64 { let (h, _) = s.next_packets().unwrap(); if h.frame_type == FrameType::Data { assert_eq!(h.format, slot((n & 0xFF) as u8).format); } }
}

#[test]
fn normal_senders_have_no_tune_rate() {
    let mut s = FlatSender::from_bytes("a.bin", vec![7u8; 50_000], FlatSenderConfig::default()).unwrap();
    assert_eq!(s.tune_frame_fps(), None);
    s.next_packets().unwrap(); s.hold_manifest_frame();
    assert_eq!(s.tune_frame_fps(), None);
    assert_eq!(s.manifest().tune, None);
}

/// A file whose content id would be the Tune carousel's (1 in about 16,000) gets another id, the same every time
/// (finish words and resume keep working), so a receiver never takes its frames for Tune.
#[test]
fn a_file_never_takes_the_tune_id() {
    use lensbolt_core::file::{content_hash_id, content_transfer_id, Prepared};
    use lensbolt_core::tune::schedule::{tune_display_60, tune_transfer_id};
    let tune = [tune_transfer_id(false), tune_transfer_id(true)];
    let (name, data) = (0u32..2_000_000).map(|i| (format!("c{i}.bin"), i.to_le_bytes().to_vec()))
        .find(|(n, d)| tune.contains(&content_hash_id(&Prepared::from_bytes(n, d.clone()).unwrap().manifest))).expect("a colliding file");
    let m = Prepared::from_bytes(&name, data.clone()).unwrap().manifest.clone();
    let id = content_transfer_id(&m);
    assert_eq!(tune_display_60(id), None, "{name}: {id:#06x}");
    assert_eq!(id, content_hash_id(&m) ^ 2);
    let a = FlatSender::from_bytes(&name, data.clone(), FlatSenderConfig::default()).unwrap();
    let b = FlatSender::from_bytes(&name, data, FlatSenderConfig { repair_offset: Some(9), ..FlatSenderConfig::default() }).unwrap();
    assert_eq!((a.config().transfer_id, b.config().transfer_id), (id, id));
    // the tune sender keeps its own
    assert_eq!(FlatSender::tune(FlatSenderConfig::default(), true).unwrap().config().transfer_id, tune[1]);
}
