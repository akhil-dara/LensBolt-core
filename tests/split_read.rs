//! The browser receiver split in two: readers (`FrameReader`, one per worker, each tracking the frames it sees) turn
//! a camera frame into a compact record, and one collector (`RxSession::push_read`) holds the file. Over the stored
//! real captures the split gives the very status the single session (`push_i420_at`) gives, byte for byte, so the
//! reader's tracked state and every per-frame field the status shows (codewords, sharpness, mix, edge width) travel
//! in the record. Without the captures (they are not in the repository) the rendered frames of a 2 px b/w tile and
//! its manifest check the split, and the capture tests say so and pass (`LB_REQUIRE_CAPTURES=1`: they fail).
mod common;
use common::Capture;
use lensbolt_core::flat::session::{FrameRead, FrameReader, RxSession};

/// A capture's planes as `VideoFrame.copyTo` hands out I420: the Y plane, then U and V each in a plane of its own
/// (the app's dumps interleave U and V, `uv_pixel` 2).
fn i420(c: &Capture) -> (Vec<u8>, [usize; 6]) {
    let (w, h) = (c.w as usize, c.h as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut buf = Vec::with_capacity(c.y_row * h + 2 * cw * ch);
    buf.extend_from_slice(&c.y[..c.y_row * (h - 1) + w]);
    buf.resize(c.y_row * h, 0);
    let uo = buf.len();
    for plane in [&c.u, &c.v] {
        for r in 0..ch { for x in 0..cw { buf.push(plane.get(r * c.uv_row + x * c.uv_pixel).copied().unwrap_or(128)); } }
    }
    (buf, [0, c.y_row, uo, cw, uo + cw * ch, cw])
}

/// Clean live captures (Full HD 2 px, its phase run, 1 px), and the soft and blended ones (two sender frames in a
/// capture: mix up to ~0.9, 0-4 of 111 codewords, failed reads), so every field of the record is exercised.
const SETS: [&str; 5] = ["test/phone_fhd2px_live/dump", "test/phone_fhd2px_phase/dump", "test/phone_max1px_live/dump", "test/phone_fhd2px_soft/dump", "test/phone_fhd2px_ramp3ms/dump"];

fn field<'a>(json: &'a str, key: &str) -> &'a str {
    let at = json.find(&format!("\"{key}\":")).unwrap_or_else(|| panic!("no {key} in {json}")) + key.len() + 3;
    let end = json[at..].find([',', '}']).unwrap();
    &json[at..at + end]
}

#[test]
fn reader_records_and_collector_give_the_single_session_status_byte_for_byte() {
    let mut sets = 0;
    // what the statuses showed over every set: a blend, a failed read, a capture that read only part of its codewords
    let (mut mix_max, mut failed_max, mut partial) = (0f64, 0usize, 0usize);
    for rel in SETS {
        let Some(dir) = common::gate_dir(rel) else { continue };
        sets += 1;
        let caps = common::captures(&dir);
        let (mut one, mut reader, mut collector) = (RxSession::new(), FrameReader::new(), RxSession::new());
        let (mut bytes, mut reads) = (0usize, 0usize);
        let mut last = String::new();
        // each capture as it is, and now and then the same capture spoilt: a lower part dark (a hand over the screen:
        // markers missing or a partial read), or flat grey (nothing to read: a failure and its reason)
        let frames = caps.iter().enumerate().flat_map(|(i, c)| {
            let (buf, l) = i420(c);
            let mut out = vec![(c, buf.clone(), l)];
            if i % 9 == 4 { let mut cut = buf.clone(); let from = c.y_row * (c.h as usize * 3 / 5); cut[from..c.y_row * c.h as usize].fill(0); out.push((c, cut, l)); }
            if i % 9 == 7 { let mut grey = buf; grey[..c.y_row * c.h as usize].fill(128); out.push((c, grey, l)); }
            out
        });
        for (i, (c, buf, l)) in frames.enumerate() {
            let t = 1000.0 + i as f64 * 33.3;
            let want = one.push_i420_at(&buf, c.w, c.h, l, t);
            let read = reader.read_i420(&buf, c.w, c.h, l);
            if matches!(read, FrameRead::Read { .. }) { reads += 1; }
            let record = read.to_bytes();
            bytes = bytes.max(record.len());
            let got = collector.push_read(&record, t);
            assert_eq!(got, want, "{rel}, capture {i} ({}): the split's status differs", c.stem);
            mix_max = mix_max.max(field(&got, "mix").parse().unwrap());
            failed_max = failed_max.max(field(&got, "failed").parse().unwrap());
            let (ok, all): (usize, usize) = (field(&got, "codewords_ok").parse().unwrap(), field(&got, "codewords").parse().unwrap());
            if all > 0 && ok < all { partial += 1; }
            last = got;
        }
        let decoded: usize = field(&last, "decoded").parse().unwrap();
        eprintln!("{rel}: {} captures (and spoilt copies), {decoded} decoded, {reads} read records, largest record {} KB; last status {last}", caps.len(), bytes / 1024);
        assert!(decoded > 0 && decoded == reads, "{rel}: {decoded} decoded, {reads} read records");
    }
    eprintln!("over every set: mix up to {mix_max:.2}, {failed_max} failed reads in one set, {partial} statuses with codewords missing");
    // with the blended sets present, the records carried a blend, failures and partial reads, all identical
    assert!(sets == 0 || failed_max > 0, "no failed read: the failure records were not compared");
    if SETS[3..].iter().any(|r| common::capture_dir(r).is_some()) {
        assert!(mix_max >= 0.3 && partial > 10, "the blended sets did not exercise mix ({mix_max}), failures ({failed_max}) or partial reads ({partial})");
    }
    if sets == 0 { eprintln!("no stored captures on this machine: the rendered frames' test checks the split"); }
}

/// A record survives the trip: encoding what a decoded record gives back is the same bytes (nothing is lost or
/// added on the way), and a record that is cut short or garbled is refused without a panic or a counted decode.
#[test]
fn records_round_trip_and_damaged_ones_are_refused() {
    let Some(dir) = common::gate_dir(SETS[0]) else { return };
    let caps = common::captures(&dir);
    let mut reader = FrameReader::new();
    let mut seen = 0;
    for c in caps.iter().take(12) {
        let (buf, l) = i420(c);
        let record = reader.read_i420(&buf, c.w, c.h, l).to_bytes();
        let back = FrameRead::from_bytes(&record).expect("a record the reader made is read back");
        assert_eq!(back.to_bytes(), record, "{}: the record changed on its way", c.stem);
        if let FrameRead::Read { reads, .. } = &back {
            // the sampling shifts are the reader's (they travel in a hint): the collector never uses them
            assert!(reads.iter().all(|r| r.reads.shifts.is_empty()), "{}: the record carries the shifts", c.stem);
            seen += 1;
            for cut in [0, 1, 5, record.len() / 2, record.len() - 1] { assert!(FrameRead::from_bytes(&record[..cut]).is_none(), "cut to {cut} bytes: accepted"); }
            let mut extra = record.clone(); extra.push(0);
            assert!(FrameRead::from_bytes(&extra).is_none(), "a byte past the end: accepted");
        }
    }
    assert!(seen > 0, "no capture read");
    let mut rx = RxSession::new();
    let s = rx.push_read(&[1, 2, 3], 10.0);
    assert!(field(&s, "last_error") == "\"BadRecord\"" && field(&s, "frames") == "1" && field(&s, "decoded") == "0" && field(&s, "failed") == "0", "{s}");
}

/// Records may reach the collector out of order (several readers): their packets count, but a record older than
/// the newest one taken leaves the "last frame" fields (codewords, sharpness, corners, last miss) to the newer one.
#[test]
fn a_late_record_counts_but_leaves_the_last_frame_fields() {
    let Some(dir) = common::gate_dir(SETS[0]) else { return };
    let caps = common::captures(&dir);
    let mut reader = FrameReader::new();
    let mut recs: Vec<Vec<u8>> = Vec::new();
    for c in caps.iter().take(10) {
        let (buf, l) = i420(c);
        let r = reader.read_i420(&buf, c.w, c.h, l);
        if matches!(r, FrameRead::Read { .. }) { recs.push(r.to_bytes()); }
        if recs.len() == 2 { break; }
    }
    assert_eq!(recs.len(), 2, "two reads among the first captures");
    let mut in_order = RxSession::new();
    in_order.push_read(&recs[0], 100.0);
    let want = in_order.push_read(&recs[1], 133.0);
    let mut swapped = RxSession::new();
    swapped.push_read(&recs[1], 133.0);
    let got = swapped.push_read(&recs[0], 100.0);
    for key in ["frames", "decoded", "failed", "received_bytes", "data_reads", "codewords_ok", "codewords", "sharpness", "mix", "edge_px", "corners", "format", "last_error", "seconds"] {
        assert_eq!(field(&got, key), field(&want, key), "{key}: in order {want}\nswapped {got}");
    }
    // a failure that arrives late is counted, its reason is not the last miss
    let mut late_miss = RxSession::new();
    late_miss.push_read(&recs[1], 133.0);
    let s = late_miss.push_read(&FrameRead::Failed("NoMarkers".into()).to_bytes(), 120.0);
    assert!(field(&s, "failed") == "1" && field(&s, "last_error") == "null", "{s}");
}

/// A read's record without its decode time (a clock reading, different on every run).
fn untimed(mut r: FrameRead) -> Vec<u8> {
    if let FrameRead::Read { reads, .. } = &mut r { for fr in reads { fr.decode_ms = 0.0; } }
    r.to_bytes()
}

/// A reader that lost the tile takes another reader's geometry as its hint: the hint survives the trip, and the
/// reader that took it reads the next capture exactly as the reader that tracked it itself.
#[test]
fn a_hint_from_another_reader_reads_like_its_own() {
    let Some(dir) = common::gate_dir(SETS[0]) else { return };
    let caps = common::captures(&dir);
    let mut a = FrameReader::new();
    assert!(a.hint_bytes().is_empty(), "a new reader has no hint");
    let mut checked = 0;
    for w in caps.windows(2) {
        let (b0, l0) = i420(&w[0]);
        if !matches!(a.read_i420(&b0, w[0].w, w[0].h, l0), FrameRead::Read { .. }) { continue; }
        let hint = a.hint_bytes();
        let mut b = FrameReader::new();
        assert!(b.set_hint(&hint), "the hint is taken");
        assert_eq!(b.hint_bytes(), hint, "the hint survives the trip");
        let (b1, l1) = i420(&w[1]);
        let (ra, rb) = (untimed(a.read_i420(&b1, w[1].w, w[1].h, l1)), untimed(b.read_i420(&b1, w[1].w, w[1].h, l1)));
        assert!(ra == rb, "{}: the reader with the handed-over hint read it differently", w[1].stem);
        checked += 1;
        if checked == 6 { break; }
    }
    assert!(checked > 0, "no pair of captures to check");
    let mut c = FrameReader::new();
    assert!(!c.set_hint(&[9, 9, 9]), "a damaged hint is refused");
    assert!(c.hint_bytes().is_empty());
}

/// A rendered frame as `VideoFrame.copyTo` hands out I420 (BT.601 full range), rows padded past the width.
fn i420_of(img: &lensbolt_core::image::RgbImage) -> (Vec<u8>, [usize; 6]) {
    let (w, h) = (img.width as usize, img.height as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let (ys, cs) = (w + 8, cw + 8);
    let (uo, vo) = (ys * h, ys * h + cs * ch);
    let mut buf = vec![0u8; vo + cs * ch];
    let px = |x: usize, y: usize| { let p = &img.data[(y * w + x) * 3..]; (p[0] as f32, p[1] as f32, p[2] as f32) };
    for y in 0..h { for x in 0..w { let (r, g, b) = px(x, y); buf[y * ys + x] = (0.299 * r + 0.587 * g + 0.114 * b).round().clamp(0.0, 255.0) as u8; } }
    for y in 0..ch { for x in 0..cw {
        let (r, g, b) = px((2 * x).min(w - 1), (2 * y).min(h - 1));
        buf[uo + y * cs + x] = (128.0 - 0.168_736 * r - 0.331_264 * g + 0.5 * b).round().clamp(0.0, 255.0) as u8;
        buf[vo + y * cs + x] = (128.0 + 0.5 * r - 0.418_688 * g - 0.081_312 * b).round().clamp(0.0, 255.0) as u8;
    } }
    (buf, [0, ys, uo, cs, vo, cs])
}

/// Without any stored capture (a fresh clone): the real sender's frames of a 2 px b/w tile, its manifest frame
/// among them, as I420. Each frame's record through the collector gives the status the single session gives,
/// byte for byte, up to the verified file; a damaged record is refused (counted, named, nothing decoded), and a
/// record that arrives late adds its packets and leaves the last frame's fields to the newer one.
#[test]
fn rendered_frames_through_the_split_give_the_single_session_status() {
    use lensbolt_core::flat::{render, Alphabet, CellFormat, FlatSender, FlatSenderConfig};
    let data: Vec<u8> = { let mut s = 0x2b_u64; (0..30_000).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s as u8 }).collect() };
    let cfg = FlatSenderConfig { manifest_every_frames: 6, transfer_id: 81, format: CellFormat { pitch: 2, alphabet: Alphabet::Bw }, ..FlatSenderConfig::default() };
    let mut tx = FlatSender::from_bytes("split.bin", data.clone(), cfg).unwrap();
    let frames: Vec<(Vec<u8>, [usize; 6], u32, u32)> = (0..40).map(|_| {
        let img = render(&tx.next_frame().unwrap());
        let (buf, l) = i420_of(&img);
        (buf, l, img.width, img.height)
    }).collect();

    let (mut one, mut reader, mut collector) = (RxSession::new(), FrameReader::new(), RxSession::new());
    let (mut records, mut last) = (Vec::new(), String::new());
    for (i, (buf, l, w, h)) in frames.iter().enumerate() {
        let t = 1000.0 + i as f64 * 33.3;
        let want = one.push_i420_at(buf, *w, *h, *l, t);
        let record = reader.read_i420(buf, *w, *h, *l).to_bytes();
        let got = collector.push_read(&record, t);
        assert_eq!(got, want, "frame {i}: the split's status differs");
        records.push(record);
        last = got;
        if one.is_complete() { break; }
    }
    assert!(one.is_complete() && collector.is_complete(), "not complete: {last}");
    assert!(field(&last, "have_manifest") == "true" && field(&last, "format").starts_with("\"2px bw"), "{last}");
    assert_eq!(collector.take_file().expect("the split's file"), ("split.bin".to_string(), data));
    let reads = records.iter().filter(|r| matches!(FrameRead::from_bytes(r), Some(FrameRead::Read { .. }))).count();
    assert!(field(&last, "decoded").parse::<usize>().unwrap() == reads && reads >= 2, "{reads} reads: {last}");

    // a damaged record (cut short, or a byte past its end) is counted and named, and decodes nothing
    let (first, second) = (&records[0], &records[1]);
    let mut rx = RxSession::new();
    rx.push_read(first, 100.0);
    for bad in [first[..first.len() / 2].to_vec(), { let mut e = second.clone(); e.push(7); e }] {
        let s = rx.push_read(&bad, 133.0);
        assert!(field(&s, "last_error") == "\"BadRecord\"" && field(&s, "decoded") == "1" && field(&s, "failed") == "0", "{s}");
    }
    assert_eq!(field(&rx.push_read(second, 166.0), "decoded"), "2", "a good record after a damaged one still counts");

    // late: the manifest's and a data frame's records swapped give the same counts and last-frame fields
    let mut in_order = RxSession::new();
    in_order.push_read(first, 100.0);
    let want = in_order.push_read(second, 133.0);
    let mut swapped = RxSession::new();
    swapped.push_read(second, 133.0);
    let got = swapped.push_read(first, 100.0);
    for key in ["frames", "decoded", "failed", "received_bytes", "data_reads", "codewords_ok", "codewords", "sharpness", "mix", "edge_px", "corners", "format", "have_manifest", "last_error", "seconds"] {
        assert_eq!(field(&got, key), field(&want, key), "{key}: in order {want}\nswapped {got}");
    }
}
