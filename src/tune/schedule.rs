//! The Tune carousel: a fixed loop of 256 frames, one per value of the header's 8-bit frame number, so every
//! frame names its segment (format and rate) by its number alone. A receiver needs no clock to attribute a read,
//! can join at any point, and has seen everything after one loop. No wire change: the frames are ordinary data
//! frames of a generated payload whose manifest says tune (`file::TuneInfo`).
use crate::calib::FPS_OPTIONS;
use crate::flat::{Alphabet, CellFormat};

/// The carousel's version, carried in the tune manifest (so in the tune transfer id): a receiver measures only a
/// schedule it knows. 2: each segment in three stretches spread over the loop.
pub const SCHEDULE_VERSION: u8 = 2;
/// Frames in one loop: every frame number once.
pub const LOOP_FRAMES: usize = 256;
/// The generated payload's size: 1 MiB, one chunk.
pub const TUNE_PAYLOAD_BYTES: usize = 1 << 20;

/// One configuration the loop measures: `frames` frames in all of `format` shown at `FPS_OPTIONS[fps]`, in three
/// stretches (`STRETCHES`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment { pub format: CellFormat, pub fps: u8, pub frames: u16 }

const fn seg(pitch: u8, alphabet: Alphabet, fps: u8, frames: u16) -> Segment { Segment { format: CellFormat { pitch, alphabet }, fps, frames } }

/// The segments: the formats that matter at 1024 px at 30 fps (22 frames each), the two b/w candidates at 20 fps (3
/// refreshes per frame of a 60 Hz screen, 16 each), and at 60 fps (24 each; drawn at 30 on a slower screen, where
/// receivers ignore them). 8.27 s per loop on a 60 Hz sender, 9.07 s on a slower one.
pub const SEGMENTS: [Segment; 12] = [
    seg(2, Alphabet::Bw, 0, 22), seg(2, Alphabet::G8, 0, 22), seg(3, Alphabet::Bw, 0, 22), seg(3, Alphabet::G8, 0, 22),
    seg(3, Alphabet::C8, 0, 22), seg(4, Alphabet::C8, 0, 22), seg(5, Alphabet::Bw, 0, 22), seg(8, Alphabet::C8, 0, 22),
    seg(2, Alphabet::Bw, 2, 16), seg(3, Alphabet::Bw, 2, 16),
    seg(2, Alphabet::Bw, 1, 24), seg(3, Alphabet::Bw, 1, 24),
];

/// The loop in order, as (segment, frames): three thirds, each showing every segment once in `SEGMENTS` order, so a
/// hold that drifts slowly over the loop (a hand backing off) reaches every format, not only those shown last. Each
/// segment's frames split 8 + 7 + 7 (30 fps), 6 + 5 + 5 (20 fps), 8 + 8 + 8 (60 fps).
pub const STRETCHES: [(usize, u16); 36] = {
    let mut out = [(0usize, 0u16); 36];
    let mut third = 0;
    while third < 3 {
        let mut k = 0;
        while k < 12 {
            let frames = match (SEGMENTS[k].frames, third) { (22, 0) => 8, (22, _) => 7, (16, 0) => 6, (16, _) => 5, (f, _) => f / 3 };
            out[third * 12 + k] = (k, frames);
            k += 1;
        }
        third += 1;
    }
    out
};

/// What a frame number shows: its segment's index, format and rate (an `FPS_OPTIONS` index), and the stretch of the
/// loop it falls in (an index in `STRETCHES`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot { pub segment: usize, pub format: CellFormat, pub fps: u8, pub stretch: usize }

/// The first frame number of each stretch, and one past the last.
pub fn stretch_start(stretch: usize) -> usize { STRETCHES[..stretch.min(STRETCHES.len())].iter().map(|s| s.1 as usize).sum() }

/// The slot of frame number `seq`.
pub fn slot(seq: u8) -> Slot {
    let mut at = seq as usize;
    for (i, &(k, frames)) in STRETCHES.iter().enumerate() {
        if at < frames as usize { let s = SEGMENTS[k]; return Slot { segment: k, format: s.format, fps: s.fps, stretch: i }; }
        at -= frames as usize;
    }
    unreachable!("the stretches cover all 256 frame numbers")
}

/// The rate a segment is drawn at: its own, except 60 fps (and faster) on a sender whose screen is not 60 Hz,
/// drawn at 30.
pub fn shown_fps(fps: u8, display_60: bool) -> u32 {
    let f = FPS_OPTIONS[(fps as usize).min(FPS_OPTIONS.len() - 1)] as u32;
    if f > 30 && !display_60 { 30 } else { f }
}

/// How long frame `seq` stays on screen, in ms.
pub fn frame_ms(seq: u8, display_60: bool) -> f64 { 1000.0 / shown_fps(slot(seq).fps, display_60) as f64 }

/// One loop, in ms.
pub fn loop_ms(display_60: bool) -> f64 { (0..LOOP_FRAMES).map(|s| frame_ms(s as u8, display_60)).sum() }

/// The tune payload: `TUNE_PAYLOAD_BYTES` of xorshift64 from a fixed seed, the same on every sender (so the same
/// transfer id), and incompressible like most files.
pub fn tune_payload() -> Vec<u8> {
    let mut x = 0x4C42_5455_4E45_0001u64;
    let mut v = Vec::with_capacity(TUNE_PAYLOAD_BYTES);
    while v.len() < TUNE_PAYLOAD_BYTES {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        v.extend_from_slice(&x.to_le_bytes());
    }
    v.truncate(TUNE_PAYLOAD_BYTES);
    v
}

/// The Tune carousel's transfer id for a sender that draws the 60 fps segments at 60 (`display_60`) or at 30: the
/// content id of the tune manifest (`file::content_transfer_id`), the same on every sender since the payload is
/// fixed. A receiver in Tune knows a tune frame, and how its sender draws, by this id alone. Computed once.
pub fn tune_transfer_id(display_60: bool) -> u16 {
    static IDS: std::sync::OnceLock<[u16; 2]> = std::sync::OnceLock::new();
    let ids = IDS.get_or_init(|| [false, true].map(|d| crate::file::Prepared::tune(d).map(|p| crate::file::content_transfer_id(&p.manifest)).unwrap_or(0)));
    ids[display_60 as usize]
}
/// Whether frames of `transfer_id` are the Tune carousel's, and if so whether its 60 fps segments are drawn at 60.
pub fn tune_display_60(transfer_id: u16) -> Option<bool> {
    [false, true].into_iter().find(|&d| tune_transfer_id(d) == transfer_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flat::FlatGrid;
    #[test]
    fn one_loop_is_256_frames_within_budget() {
        assert_eq!(SEGMENTS.iter().map(|s| s.frames as usize).sum::<usize>(), LOOP_FRAMES);
        assert!((loop_ms(true) - 8266.7).abs() < 1.0 && (loop_ms(false) - 9066.7).abs() < 1.0, "{} {}", loop_ms(true), loop_ms(false));
        for h in [640, 768, 896, 1024] { for s in SEGMENTS { assert!(FlatGrid::try_new(1024, h, s.format).is_some(), "{s:?} at 1024x{h}"); } }
    }
    #[test]
    fn every_frame_number_names_its_segment() {
        assert_eq!(slot(0), Slot { segment: 0, format: CellFormat { pitch: 2, alphabet: Alphabet::Bw }, fps: 0, stretch: 0 });
        // the first third: 8 x 8 + 2 x 6 + 2 x 8 frames; 2 px b/w at 20 fps from 64, at 60 fps from 76
        assert_eq!((slot(64).segment, slot(64).fps), (8, 2));
        assert_eq!((slot(76).segment, slot(76).fps), (10, 1));
        assert_eq!(slot(92), Slot { segment: 0, format: CellFormat { pitch: 2, alphabet: Alphabet::Bw }, fps: 0, stretch: 12 });
        assert_eq!(slot(255), Slot { segment: 11, format: CellFormat { pitch: 3, alphabet: Alphabet::Bw }, fps: 1, stretch: 35 });
        assert_eq!(shown_fps(1, false), 30); assert_eq!(shown_fps(1, true), 60); assert_eq!(shown_fps(2, true), 20);
        assert_eq!(shown_fps(2, false), 20); assert_eq!(shown_fps(0, false), 30);
        for (i, &(k, _)) in STRETCHES.iter().enumerate() { assert_eq!(slot(stretch_start(i) as u8).segment, k); assert_eq!(slot(stretch_start(i) as u8).stretch, i); }
        assert_eq!(stretch_start(STRETCHES.len()), LOOP_FRAMES);
        // each segment's stretches add up to its frames, and are a third of the loop apart
        for (k, seg) in SEGMENTS.iter().enumerate() {
            let parts: Vec<usize> = (0..36).filter(|&i| STRETCHES[i].0 == k).collect();
            assert_eq!(parts.len(), 3);
            assert_eq!(parts.iter().map(|&i| STRETCHES[i].1).sum::<u16>(), seg.frames);
            assert!(stretch_start(parts[2]) - stretch_start(parts[0]) >= 160, "{k}: {parts:?}");
        }
        assert!((frame_ms(76, true) - 1000.0 / 60.0).abs() < 1e-9 && (frame_ms(64, false) - 50.0).abs() < 1e-9);
    }
    #[test]
    fn tune_ids_are_the_senders() {
        for d in [false, true] {
            let s = crate::flat::FlatSender::tune(crate::flat::FlatSenderConfig::default(), d).unwrap();
            assert_eq!(s.config().transfer_id, tune_transfer_id(d));
            assert_eq!(tune_display_60(tune_transfer_id(d)), Some(d));
        }
        assert_ne!(tune_transfer_id(false), tune_transfer_id(true));
        assert_eq!(tune_display_60(0x4c5b), None);
    }
    #[test]
    fn payload_is_fixed_and_full() {
        let p = tune_payload();
        assert_eq!(p.len(), TUNE_PAYLOAD_BYTES);
        assert_eq!(p, tune_payload());
        assert!(p.iter().filter(|&&b| b == 0).count() < TUNE_PAYLOAD_BYTES / 128);
    }
}
