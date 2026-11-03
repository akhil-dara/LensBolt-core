//! Flat-format frames photographed by a synthetic camera (perspective, bend, noise) and decoded.
use lensbolt_core::detect::Homography;
use lensbolt_core::flat::{compose, render, Alphabet, CellFormat, FlatDecoder, FlatFrame, FlatGrid, FlatHeader};
use lensbolt_core::header::FrameType;
use lensbolt_core::image::RgbImage;
use lensbolt_core::flat::DecodeError;
use lensbolt_core::testutil::bend;
use std::sync::Arc;

fn frame(fmt: CellFormat, block_id: u32) -> (Arc<FlatGrid>, FlatFrame, Vec<u8>) {
    let g = Arc::new(FlatGrid::try_new(1024, 1024, fmt).unwrap());
    let h = FlatHeader { format: fmt, cols: g.cols() as u16, rows: g.rows() as u16, frame_type: FrameType::Data, transfer_id: 0x1234, segment: 1, block_id, frame_seq: 0 };
    let payload: Vec<u8> = (0..g.payload_bytes()).map(|i| ((i as u64 * 2654435761 + block_id as u64 * 77) >> 7) as u8).collect();
    let f = compose(g.clone(), h, &payload);
    (g, f, payload)
}

/// Photograph `im` (1024x1024) into a cam_w x cam_h image: the tile is `tile_px` wide, centred, rotated by
/// `yaw_deg` with a touch of perspective; then bent and noised. Deterministic for a seed.
fn synth(im: &RgbImage, cam_w: u32, cam_h: u32, tile_px: f32, yaw_deg: f32, bend_px: f32, noise: u8, seed: u64) -> RgbImage {
    let s = tile_px / im.width as f32;
    let a = yaw_deg.to_radians();
    let (cx, cy) = (cam_w as f32 / 2.0, cam_h as f32 / 2.0);
    let map = |x: f32, y: f32| {
        let (u, v) = (x - im.width as f32 / 2.0, y - im.height as f32 / 2.0);
        let persp = 1.0 + 0.00004 * u; // the right side slightly farther away
        (cx + s * (u * a.cos() - v * a.sin()) / persp, cy + s * (u * a.sin() + v * a.cos()) / persp)
    };
    let (w, h) = (im.width as f32, im.height as f32);
    let src = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)];
    let dst = [map(0.0, 0.0), map(w, 0.0), map(w, h), map(0.0, h)];
    let f2c = Homography::from_points(src, dst).unwrap();
    let mut cam = im.warp(&f2c.inverse().unwrap(), cam_w, cam_h);
    if bend_px > 0.0 { cam = bend(&cam, bend_px, seed); }
    if noise > 0 {
        let mut st = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        for p in cam.data.iter_mut() { st ^= st << 13; st ^= st >> 7; st ^= st << 17; let n = (st % (2 * noise as u64 + 1)) as i32 - noise as i32; *p = (*p as i32 + n).clamp(0, 255) as u8; }
    }
    cam
}

#[test]
fn frontal_4k_decodes_everything() {
    let (g, f, payload) = frame(CellFormat { pitch: 4, alphabet: Alphabet::C8 }, 1);
    let cam = synth(&render(&f), 4032, 3024, 1024.0 * 1.85, 3.0, 2.5, 6, 1);
    let dec = FlatDecoder::new();
    let read = dec.decode(&cam).expect("decode");
    assert_eq!(read.header, f.header);
    assert_eq!((read.grid.cols(), read.grid.rows()), (g.cols(), g.rows()));
    let fp = read.payload();
    assert_eq!(fp.codewords_failed, 0, "{}/{} codewords", fp.codewords_ok, g.codewords());
    assert!(fp.bytes == payload, "payload differs");
}

#[test]
fn webcam_scale_decodes_bw() {
    let (g, f, payload) = frame(CellFormat { pitch: 5, alphabet: Alphabet::Bw }, 2);
    let cam = synth(&render(&f), 1920, 1080, 600.0, -4.0, 1.5, 4, 2);
    let read = FlatDecoder::new().decode(&cam).expect("decode");
    let fp = read.payload();
    assert_eq!(fp.codewords_failed, 0, "{}/{} codewords", fp.codewords_ok, g.codewords());
    assert!(fp.bytes == payload, "payload differs");
}

#[test]
fn rotated_capture_decodes() {
    let (_, f, payload) = frame(CellFormat { pitch: 6, alphabet: Alphabet::C8 }, 3);
    let cam = synth(&render(&f), 2000, 1500, 1024.0 * 1.4, 0.0, 1.0, 4, 3);
    let mut r180 = RgbImage::new(cam.width, cam.height);
    for y in 0..cam.height { for x in 0..cam.width { r180.set(cam.width - 1 - x, cam.height - 1 - y, cam.get(x, y)); } }
    let read = FlatDecoder::new().decode(&r180).expect("decode rotated");
    assert_eq!(read.header, f.header);
    let fp = read.payload();
    assert_eq!(fp.codewords_failed, 0);
    assert!(fp.bytes == payload, "payload differs");
}

#[test]
fn rotated_90_capture_decodes() {
    let (_, f, payload) = frame(CellFormat { pitch: 5, alphabet: Alphabet::C8 }, 4);
    let cam = synth(&render(&f), 2000, 1500, 1024.0 * 1.3, 0.0, 1.0, 4, 6);
    let mut r90 = RgbImage::new(cam.height, cam.width);
    for y in 0..cam.height { for x in 0..cam.width { r90.set(cam.height - 1 - y, x, cam.get(x, y)); } }
    let read = FlatDecoder::new().decode(&r90).expect("decode rotated 90");
    assert_eq!(read.header, f.header);
    let fp = read.payload();
    assert_eq!(fp.codewords_failed, 0);
    assert!(fp.bytes == payload, "payload differs");
}

#[test]
fn wide_tile_1024x768_decodes() {
    let fmt = CellFormat { pitch: 4, alphabet: Alphabet::C8 };
    let g = Arc::new(FlatGrid::try_new(1024, 768, fmt).unwrap());
    let h = FlatHeader { format: fmt, cols: g.cols() as u16, rows: g.rows() as u16, frame_type: FrameType::Data, transfer_id: 9, segment: 1, block_id: 7, frame_seq: 0 };
    let payload: Vec<u8> = (0..g.payload_bytes()).map(|i| ((i as u64 * 2654435761) >> 9) as u8).collect();
    let f = compose(g.clone(), h, &payload);
    let cam = synth(&render(&f), 3000, 2000, 1024.0 * 1.85, 2.0, 1.5, 4, 7);
    let read = FlatDecoder::new().decode(&cam).expect("decode 1024x768");
    assert_eq!(read.header, h);
    assert!(read.payload().bytes == payload, "payload differs");
}

#[test]
fn pitch_2_decodes_when_the_phone_is_close() {
    // the 20x configuration: 2 px cells at 2.8 camera px per screen px (the tile fills ~70% of a 4096 frame)
    for (fmt, min_packets) in [(CellFormat { pitch: 2, alphabet: Alphabet::C8 }, 64), (CellFormat { pitch: 2, alphabet: Alphabet::C16 }, 85)] {
        let (g, f, payload) = frame(fmt, 21);
        assert!(g.packets() >= min_packets);
        let cam = synth(&render(&f), 4096, 3072, 1024.0 * 2.8, 2.0, 4.0, 5, 8);
        let read = FlatDecoder::new().decode(&cam).unwrap_or_else(|e| panic!("{fmt:?}: {e:?}"));
        let fp = read.payload();
        assert_eq!(fp.codewords_failed, 0, "{fmt:?}: {}/{} codewords", fp.codewords_ok, g.codewords());
        assert!(fp.bytes == payload, "{fmt:?}: payload differs");
    }
}

#[test]
fn split_capture_yields_both_frames() {
    // v8: every codeword spans the whole frame, so neither half of a split capture decodes alone; the stitcher
    // keeps both halves and completes each frame when another capture brings the rest of it
    let fmt = CellFormat { pitch: 4, alphabet: Alphabet::C8 };
    let seq_frame = |block: u32, seq: u8| { let (g, f, p) = frame(fmt, block); let h = FlatHeader { frame_seq: seq, ..f.header }; (g.clone(), compose(g, h, &p), p) };
    let (g, fa, pa) = seq_frame(10, 10);
    let (_, fb, pb) = seq_frame(11, 11);
    let (ia, ib) = (render(&fa), render(&fb));
    let mut split = ia.clone();
    let cut = split.height * 55 / 100;
    for y in cut..split.height { for x in 0..split.width { split.set(x, y, ib.get(x, y)); } }
    let cam = synth(&split, 3000, 2400, 1024.0 * 1.85, 2.0, 1.5, 4, 4);
    let dec = FlatDecoder::new();
    let reads = dec.decode_all(&cam).expect("decode_all");
    assert_eq!(reads.len(), 2, "both header copies should be found");
    let mut ids: Vec<u32> = reads.iter().map(|r| r.header.block_id).collect(); ids.sort();
    assert_eq!(ids, vec![10, 11]);
    let mut st = lensbolt_core::flat::Stitcher::new();
    let first = st.add_capture(&reads[..1]);
    assert!(first.iter().all(|(_, p)| !p.valid.iter().any(|&v| v)), "half a frame decodes no packet");
    // the next captures show B whole, then A whole: each completes, with its own bytes
    let n = g.packets() * lensbolt_core::packet::PACKET_BYTES;
    for (img, want, seq) in [(&ib, &pb, 11u8), (&ia, &pa, 10u8)] {
        let r = dec.decode(&synth(img, 3000, 2400, 1024.0 * 1.85, 2.0, 1.5, 4, 9)).expect("clean capture");
        let out = st.add_capture(&[r]);
        let (_, p) = out.iter().find(|(h, _)| h.frame_seq == seq).expect("frame completes");
        assert!(p.valid[..n].iter().all(|&v| v), "frame {seq}: every packet reported");
        assert_eq!(&p.bytes[..n], &want[..n]);
    }
}

#[test]
fn tracked_decode_uses_the_hint() {
    let (_, f, payload) = frame(CellFormat { pitch: 4, alphabet: Alphabet::C8 }, 5);
    let cam = synth(&render(&f), 3000, 2400, 1024.0 * 1.85, 1.0, 1.5, 4, 5);
    let dec = FlatDecoder::new();
    let cold = dec.decode(&cam).expect("cold");
    let hint = cold.hint();
    let warm = dec.decode_tracked(&cam, Some(&hint)).expect("tracked");
    assert_eq!(warm.header, cold.header);
    assert!(warm.payload().bytes == payload, "payload differs");
    assert!(warm.decode_ms >= 0.0 && cold.decode_ms >= 0.0);
}

#[test]
fn tracked_decode_survives_a_moved_camera() {
    // the hand moves 60 px between captures: the hint's geometry is stale, the frame must still read fully
    let (_, f, payload) = frame(CellFormat { pitch: 4, alphabet: Alphabet::C8 }, 12);
    let im = render(&f);
    let cam0 = synth(&im, 3000, 2400, 1024.0 * 1.85, 1.0, 1.5, 4, 21);
    // the next capture: the phone came 8% closer, rolled 4 degrees and moved 60 px, with its own bend
    let cam1 = synth(&im, 3000, 2400, 1024.0 * 2.0, 5.0, 2.0, 4, 22);
    let dec = FlatDecoder::new();
    let cold = dec.decode(&cam0).expect("cold");
    let hint = cold.hint();
    let warm = dec.decode_tracked(&cam1, Some(&hint)).expect("tracked after a move");
    let fp = warm.payload();
    assert_eq!(fp.codewords_failed, 0, "{}/{} codewords after a 60 px move", fp.codewords_ok, fp.codewords_ok + fp.codewords_failed);
    assert!(fp.bytes == payload, "payload differs");
}

#[test]
fn too_blurry_is_reported() {
    let dec = FlatDecoder::new();
    assert_eq!(dec.decode(&RgbImage::new(640, 480)).err(), Some(DecodeError::TooBlurry(0.0)));
}


fn luma_payload(dec: &FlatDecoder, cam: &RgbImage, hint: Option<&lensbolt_core::flat::FlatHint>) -> Result<(lensbolt_core::flat::FlatRead, Vec<u8>), DecodeError> {
    let frs = dec.decode_luma(&cam.to_gray(), hint)?;
    let fr = frs.into_iter().next().unwrap();
    let p = fr.payload();
    Ok((fr, p.bytes))
}

#[test]
fn luma_decode_reads_bw_and_tracks_a_move() {
    let (_, f, payload) = frame(CellFormat { pitch: 3, alphabet: Alphabet::Bw }, 40);
    let im = render(&f);
    let cam0 = synth(&im, 3000, 2400, 1024.0 * 1.85, 1.0, 1.5, 4, 31);
    let cam1 = synth(&im, 3000, 2400, 1024.0 * 2.0, 5.0, 2.0, 4, 32);
    let dec = FlatDecoder::new();
    let (cold, bytes0) = luma_payload(&dec, &cam0, None).expect("cold luma decode");
    assert!(bytes0 == payload, "cold payload differs");
    let (_, bytes1) = luma_payload(&dec, &cam1, Some(&cold.hint())).expect("tracked luma decode after a move");
    assert!(bytes1 == payload, "tracked payload differs");
}

#[test]
fn luma_decode_handles_rotations() {
    let (_, f, payload) = frame(CellFormat { pitch: 3, alphabet: Alphabet::Bw }, 41);
    let im = render(&f);
    let dec = FlatDecoder::new();
    for yaw in [90.0f32, 180.0, 270.0] {
        let cam = synth(&im, 2600, 2600, 1024.0 * 1.85, yaw + 1.0, 1.0, 4, 33);
        let (_, bytes) = luma_payload(&dec, &cam, None).unwrap_or_else(|e| panic!("yaw {yaw}: {e:?}"));
        assert!(bytes == payload, "yaw {yaw}: payload differs");
    }
}

#[test]
fn colour_header_on_luma_path_asks_for_colour() {
    let (_, f, _) = frame(CellFormat { pitch: 4, alphabet: Alphabet::C8 }, 42);
    let cam = synth(&render(&f), 3000, 2400, 1024.0 * 1.85, 1.0, 1.0, 4, 34);
    match FlatDecoder::new().decode_luma(&cam.to_gray(), None) {
        Err(DecodeError::NeedsColour(h)) => assert_eq!(h.format.alphabet, Alphabet::C8),
        other => panic!("expected NeedsColour, got {:?}", other.map(|v| v.len())),
    }
}

fn fhd_frame(fmt: CellFormat, block_id: u32) -> (FlatFrame, Vec<u8>) {
    let g = Arc::new(FlatGrid::try_new(1920, 1080, fmt).unwrap());
    let h = FlatHeader { format: fmt, cols: g.cols() as u16, rows: g.rows() as u16, frame_type: FrameType::Data, transfer_id: 0x2345, segment: 1, block_id, frame_seq: 0 };
    let payload: Vec<u8> = (0..g.payload_bytes()).map(|i| ((i as u64 * 2654435761 + block_id as u64 * 31) >> 7) as u8).collect();
    (compose(g, h, &payload), payload)
}

#[test]
fn fhd_fullscreen_3px_bw_decodes_on_luma_path() {
    // a full-screen 1920x1080 tile filling a Full HD camera at 0.9 camera px per screen px (2.7 per cell)
    let (f, payload) = fhd_frame(CellFormat { pitch: 3, alphabet: Alphabet::Bw }, 50);
    let cam = synth(&render(&f), 1920, 1080, 1920.0 * 0.9, 0.5, 1.5, 4, 51);
    let frs = FlatDecoder::new().decode_luma(&cam.to_gray(), None).expect("full-screen FHD tile");
    assert_eq!((frs[0].grid.width_px(), frs[0].grid.height_px()), (1920, 1080));
    let p = frs[0].payload();
    assert!(p.bytes == payload, "payload differs: {}/{} codewords", p.codewords_ok, p.codewords_ok + p.codewords_failed);
}

#[test]
fn realistic_grey_frames_decode_every_codeword() {
    // the luma formats for ordinary cameras: 3 px at a Full HD camera's sampling, 2 px for a closer phone;
    // perspective, lens blur, colour rendering and noise (testutil::photograph_realistic)
    for (pitch, cam) in [(3u8, 1180u32), (2, 1750)] {
        let (_, f, payload) = frame(CellFormat { pitch, alphabet: Alphabet::G4 }, 60 + pitch as u32);
        let photo = lensbolt_core::testutil::photograph_realistic(&render(&f), 1.0, cam, cam, 61);
        let frs = FlatDecoder::new().decode_luma(&photo.to_gray(), None).unwrap_or_else(|e| panic!("{pitch}px g4: {e:?}"));
        let p = frs[0].payload();
        assert_eq!(p.codewords_failed + p.codewords_missing, 0, "{pitch}px g4: {}/{} codewords", p.codewords_ok, p.codewords_ok + p.codewords_failed + p.codewords_missing);
        assert!(p.bytes == payload, "{pitch}px g4: payload differs");
    }
}

#[test]
fn camera_matched_784x440_tile_at_1px_decodes() {
    // the Full HD camera's tile: ~784 x 440 cells at 1 px fill a 1920 x 1080 frame at ~2.45 camera px per cell
    let fmt = CellFormat { pitch: 1, alphabet: Alphabet::Bw };
    let g = Arc::new(FlatGrid::try_new(784, 440, fmt).expect("1 px grid in a 784x440 tile"));
    let h = FlatHeader { format: fmt, cols: g.cols() as u16, rows: g.rows() as u16, frame_type: FrameType::Data, transfer_id: 0x3456, segment: 1, block_id: 70, frame_seq: 0 };
    let payload: Vec<u8> = (0..g.payload_bytes()).map(|i| ((i as u64 * 2654435761) >> 7) as u8).collect();
    let f = compose(g.clone(), h, &payload);
    let cam = synth(&render(&f), 1960, 1100, 784.0 * 2.5, 0.3, 0.8, 3, 71);
    let frs = FlatDecoder::new().decode_luma(&cam.to_gray(), None).expect("decode");
    assert_eq!((frs[0].grid.width_px(), frs[0].grid.height_px()), (784, 440));
    let p = frs[0].payload();
    assert!(p.bytes == payload, "{}/{} codewords", p.codewords_ok, p.codewords_ok + p.codewords_failed);
}
