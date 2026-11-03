//! Format 9 changed the fountain's repair packets for chunks of up to 128 blocks (core/src/fountain/lt.rs): a
//! receiver that read the other version's repair packets would solve wrong blocks. The frame header's version
//! keeps the two apart: a version-8 sender's frames never reach a version-9 receiver's packets, and every
//! version-9 frame says 9, which a version-8 receiver (it accepts 8 only) rejects. Real v8 captures:
//! tests/v8_captures.rs.
use lensbolt_core::crc16::crc16;
use lensbolt_core::file::receiver::{FeedOutcome, Receiver};
use lensbolt_core::file::chunk_blocks;
use lensbolt_core::flat::render::render_chrome;
use lensbolt_core::flat::session::RxSession;
use lensbolt_core::flat::live::{decode_luma_first, Capture, LiveConfig, LiveReceiver};
use lensbolt_core::flat::{compose, render, Alphabet, CellFormat, DecodeError, FlatDecoder, FlatFrame, FlatHeader, FlatSender, FlatSenderConfig, FLAT_HEADER_BYTES, FLAT_VERSION};
use lensbolt_core::fountain::prng::Xorshift64;
use lensbolt_core::fountain::BLOCK_BYTES;
use lensbolt_core::header::FrameType;
use lensbolt_core::image::RgbImage;
use lensbolt_core::packet::{pack_frame, Packet};

/// Version 8's repair rule (2026-10-30, before this change): degree (k+1)/2..=k for k <= 32, else 32..=min(96, k).
fn v8_neighbors(pid: u32, k: usize) -> Vec<u32> {
    if (pid as usize) < k { return vec![pid]; }
    let mut rng = Xorshift64::new(((pid as u64) << 32) ^ (k as u64) ^ 0xC0DE_F0F0_1234_5678);
    let d = if k <= 1 { 1 } else if k <= 32 { let lo = (k + 1) / 2; lo + rng.below((k - lo + 1) as u32) as usize } else { 32 + rng.below((96.min(k) - 32 + 1) as u32) as usize };
    let mut out: Vec<u32> = Vec::with_capacity(d);
    while out.len() < d { let i = rng.below(k as u32); if !out.contains(&i) { out.push(i); } }
    out
}

/// The header bytes with another version in their version field (CRC recomputed).
fn with_version(h: &FlatHeader, v: u8) -> [u8; FLAT_HEADER_BYTES] {
    let mut b = h.pack();
    b[0] = (b[0] & 0xF0) | v;
    let c = crc16(&b[..12]).to_be_bytes();
    b[12] = c[0]; b[13] = c[1];
    b
}

/// What a version-8 receiver's header check accepts: a valid CRC, the magic, version 8.
fn v8_receiver_accepts(b: &[u8; FLAT_HEADER_BYTES]) -> bool {
    u16::from_be_bytes([b[12], b[13]]) == crc16(&b[..12]) && b[0] >> 4 == 0xA && b[0] & 0xF == 8
}

fn rgba(img: &RgbImage) -> Vec<u8> {
    let mut v = Vec::with_capacity(img.data.len() / 3 * 4);
    for p in img.data.chunks_exact(3) { v.extend_from_slice(&[p[0], p[1], p[2], 255]); }
    v
}

fn file() -> Vec<u8> {
    let mut s = 0xC0551_u64;
    (0..40 * 1024 - 5).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s as u8 }).collect() // one chunk of 40 blocks
}

fn cfg() -> FlatSenderConfig {
    FlatSenderConfig { format: CellFormat { pitch: 4, alphabet: Alphabet::Bw }, manifest_every_frames: 7, transfer_id: 0x5A5B, repair_offset: Some(11), ..FlatSenderConfig::default() }
}

/// The frames of a 40-block file as `v8` (or this version) would send them: frame, its packets, and whether the
/// camera sees it (the first pass loses every other data frame, so repair packets are needed).
fn frames(v8: bool, n: usize) -> Vec<(FlatFrame, Vec<Packet>, bool)> {
    let data = file();
    let mut s = FlatSender::from_bytes("cross.bin", data.clone(), cfg()).unwrap();
    let mut twin = FlatSender::from_bytes("cross.bin", data.clone(), cfg()).unwrap();
    let k = chunk_blocks(data.len());
    let (mut out, mut data_frames) = (Vec::new(), 0);
    for _ in 0..n {
        let f = s.next_frame().unwrap();
        let (_, mut packets) = twin.next_packets().unwrap();
        let mut seen = true;
        if f.header.frame_type == FrameType::Data {
            let systematic = packets.iter().all(|p| (p.id.pid as usize) < k);
            if systematic { data_frames += 1; seen = data_frames % 2 == 1; }
        }
        if v8 {
            for p in packets.iter_mut().filter(|p| p.id.segment != 0) {
                let mut payload = [0u8; BLOCK_BYTES];
                for i in v8_neighbors(p.id.pid as u32, k) {
                    for (j, b) in data.iter().skip(i as usize * BLOCK_BYTES).take(BLOCK_BYTES).enumerate() { payload[j] ^= b; }
                }
                p.payload = payload;
            }
        }
        let frame = compose(f.grid.clone(), f.header, &pack_frame(&packets, &f.grid));
        out.push((frame, packets, seen));
    }
    out
}

#[test]
fn a_v8_senders_frames_never_reach_a_v9_receiver() {
    let data = file();
    let n = 60;
    // control: this version's frames, rendered, complete the file
    let mut rx = RxSession::new();
    let mut done_at = None;
    for (i, (f, _, seen)) in frames(false, n).into_iter().enumerate() {
        if !seen { continue; }
        let img = render(&f);
        rx.push_rgba(&rgba(&img), img.width, img.height);
        if rx.is_complete() { done_at = Some(i); break; }
    }
    let done_at = done_at.expect("this version's frames never completed the file");
    assert!(rx.take_file().expect("file").1 == data, "this version: received bytes differ");
    // v8 frames: v8 repair packets under a version-8 header: every frame is rejected at its header, none decodes
    let mut rx = RxSession::new();
    let mut status = String::new();
    for (f, _, seen) in frames(true, n) {
        if !seen { continue; }
        let mut img = render(&f);
        render_chrome(&mut img, f.grid.chrome(), &FlatHeader::header_bits(&with_version(&f.header, 8)));
        status = rx.push_rgba(&rgba(&img), img.width, img.height);
        assert!(status.contains("\"decoded\":0,") && status.contains("\"last_error\":\"OtherVersion(8)\""), "a v8 frame decoded: {status}");
        assert!(!rx.is_complete());
    }
    assert!(status.contains("\"have_manifest\":false") && status.contains("\"received_bytes\":0"), "{status}");
    // why the gate matters: the same v8 packets read as this version's (no header check) solve wrong blocks;
    // the segment hash catches them, so even then no wrong file comes out, but no file comes out at all
    let mut rx = Receiver::in_memory();
    for (f, packets, seen) in frames(true, 4 * n) {
        if !seen { continue; }
        if let FeedOutcome::Complete(_) = rx.feed_packets(&f.header, &packets).unwrap() { panic!("v8 packets completed a v9 receiver"); }
    }
    assert!(rx.hash_failures() >= 1 && rx.take_file().is_none(), "v8 repair packets solved the right blocks: the test needs repair packets of a changed chunk size");
    eprintln!("v9 frames complete at frame {done_at}; v8 frames: none decoded; v8 packets without the gate: {} hash failures, no file", rx.hash_failures());
}

#[test]
fn every_v9_frame_says_version_9() {
    // a version-8 receiver checks the header's version first and accepts 8 only: every frame kind this sender
    // draws (data, manifest, held manifest, the calibration carousel) carries 9
    let data = file();
    let mut s = FlatSender::from_bytes("cross.bin", data.clone(), cfg()).unwrap();
    let mut cal = FlatSender::from_bytes("cross.bin", data, FlatSenderConfig { calibration: true, ..cfg() }).unwrap();
    let mut headers = vec![s.hold_manifest_frame().header];
    for _ in 0..40 { headers.push(s.next_frame().unwrap().header); headers.push(cal.next_frame().unwrap().header); }
    assert!(headers.iter().any(|h| h.frame_type == FrameType::Manifest) && headers.iter().any(|h| h.frame_type == FrameType::Data));
    for h in &headers {
        let b = h.pack();
        assert_eq!(b[0] & 0xF, 9, "{h:?}");
        assert!(!v8_receiver_accepts(&b), "{h:?}");
        assert_eq!(FlatHeader::unpack(&b), Ok(*h));
        // the copied v8 check accepts the same header carrying 8 (it is the check, not a check that rejects all)
        assert!(v8_receiver_accepts(&with_version(h, 8)));
        assert!(FlatHeader::unpack(&with_version(h, 8)).is_err(), "a v9 receiver took a v8 header");
    }
}

/// A data frame and a manifest frame of this sender drawn with `v` in their headers' version field.
fn frames_of_version(v: u8) -> Vec<RgbImage> {
    let mut s = FlatSender::from_bytes("cross.bin", file(), cfg()).unwrap();
    let mut out = Vec::new();
    for _ in 0..2 {
        let f = s.next_frame().unwrap();
        let mut img = render(&f);
        render_chrome(&mut img, f.grid.chrome(), &FlatHeader::header_bits(&with_version(&f.header, v)));
        out.push(img);
    }
    out
}

#[test]
fn a_frame_of_another_version_is_its_own_error() {
    // the header is read whole (RS, CRC, magic) and names another version: the decoder says so, not NoHeader,
    // on every path (RGB, luma, the live receiver's and the browser's), so the user can be told to update.
    // A version from before (8) and one from after (10) alike; this version's frames still decode
    let dec = FlatDecoder::new();
    for v in [8u8, 10] {
        for img in frames_of_version(v) {
            assert_eq!(dec.decode(&img).err(), Some(DecodeError::OtherVersion(v)));
            assert_eq!(dec.decode_all(&img).err(), Some(DecodeError::OtherVersion(v)));
            assert_eq!(decode_luma_first(&dec, &img.to_gray(), || img.clone(), None).err(), Some(DecodeError::OtherVersion(v)));
            let mut rx = RxSession::new();
            let st = rx.push_rgba(&rgba(&img), img.width, img.height);
            assert!(st.contains(&format!("\"last_error\":\"OtherVersion({v})\"")), "{st}");
        }
    }
    for img in frames_of_version(FLAT_VERSION) { assert!(dec.decode(&img).is_ok()); }
    // a frame with no readable header at all is still NoHeader
    let mut img = frames_of_version(FLAT_VERSION).remove(1);
    let f = FlatSender::from_bytes("cross.bin", file(), cfg()).unwrap().next_frame().unwrap();
    let blank = vec![false; FlatHeader::header_bits(&f.header.pack()).len()];
    render_chrome(&mut img, f.grid.chrome(), &blank);
    assert_eq!(dec.decode(&img).err(), Some(DecodeError::NoHeader));
    // the phone's live receiver reports it in its status
    let out = std::env::temp_dir().join("lensbolt_cross_version_live");
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).unwrap();
    let live = LiveReceiver::new(&out, LiveConfig { workers: 1 }).unwrap();
    let img = frames_of_version(8).remove(1);
    assert!(live.submit_capture(move || Capture { y: img.to_gray(), rgb: Some(Box::new(move || img)) }));
    let t0 = std::time::Instant::now();
    while live.status().failed == 0 { assert!(t0.elapsed().as_secs() < 30, "{:?}", live.status()); std::thread::sleep(std::time::Duration::from_millis(2)); }
    assert_eq!(live.status().last_error.as_deref(), Some("OtherVersion(8)"));
}

/// The manifest decoder of every build before Tune (frozen copy of `Manifest::decode` as of 2026-11-01): flags 0, 1
/// (one file, raw or zstd) and 2 (a batch) only, nothing after the last section. Such a receiver refuses a tune
/// manifest (flag 4): its manifest branch then keeps no manifest and creates no part, so nothing is written, and the
/// tune frames' data waits for a manifest that never decodes.
fn pre_tune_decode(bytes: &[u8]) -> Option<(String, u64)> {
    if bytes.len() < 4 + 1 + 1 + 8 + 8 + 2 + 32 + 4 + 2 { return None; }
    let (body, crc) = bytes.split_at(bytes.len() - 2);
    if u16::from_be_bytes([crc[0], crc[1]]) != crc16(body) { return None; }
    if &body[..4] != b"LBMF" || body[4] != 1 { return None; }
    let is_batch = match body[5] { 0 | 1 => false, 2 => true, _ => return None };
    let mut p = 6usize;
    let rd8 = |p: &mut usize| -> Option<u64> { let v = u64::from_le_bytes(body.get(*p..*p + 8)?.try_into().ok()?); *p += 8; Some(v) };
    let rd_name = |p: &mut usize| -> Option<String> {
        let len = u16::from_le_bytes(body.get(*p..*p + 2)?.try_into().ok()?) as usize; *p += 2;
        let s = String::from_utf8(body.get(*p..*p + len)?.to_vec()).ok()?; *p += len;
        Some(s)
    };
    let _file_size = rd8(&mut p)?; let stream_size = rd8(&mut p)?;
    let filename = rd_name(&mut p)?;
    body.get(p..p + 32)?; p += 32;
    let seg_count = u32::from_le_bytes(body.get(p..p + 4)?.try_into().ok()?) as usize; p += 4;
    body.get(p..p + 32 * seg_count)?; p += 32 * seg_count;
    if is_batch { rd8(&mut p)?; body.get(p..p + 32)?; p += 32; rd_name(&mut p)?; }
    (p == body.len()).then_some((filename, stream_size))
}

#[test]
fn a_build_before_tune_refuses_the_tune_manifest() {
    for d60 in [false, true] {
        let tune = FlatSender::tune(FlatSenderConfig::default(), d60).unwrap();
        let bytes = tune.manifest().encode();
        assert_eq!(bytes[5], 4, "the tune flag");
        assert_eq!(pre_tune_decode(&bytes), None, "display_60 {d60}");
    }
    // and still reads what it always read
    let file = FlatSender::from_bytes("a.bin", vec![1u8; 5000], FlatSenderConfig::default()).unwrap();
    assert_eq!(pre_tune_decode(&file.manifest().encode()), Some(("a.bin".to_string(), 5000)));
}
