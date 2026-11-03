//! The browser receiver's engine: camera RGBA frames in (as getImageData hands them out), status out, the
//! verified file out. Single-threaded (one WebAssembly instance per worker).
use lensbolt_core::flat::session::RxSession;
use lensbolt_core::flat::{render, Alphabet, CellFormat, FlatSender, FlatSenderConfig};
use lensbolt_core::testutil::{bend, pseudo_random_file};

fn rgba(img: &lensbolt_core::image::RgbImage) -> Vec<u8> {
    let mut v = Vec::with_capacity(img.data.len() / 3 * 4);
    for p in img.data.chunks_exact(3) { v.extend_from_slice(&[p[0], p[1], p[2], 255]); }
    v
}

#[test]
fn session_receives_a_file_from_rgba_frames() {
    let d = std::env::temp_dir().join("lensbolt_session"); std::fs::create_dir_all(&d).unwrap();
    let src = d.join("web.bin"); pseudo_random_file(&src, 90_000, 31).unwrap();
    let cfg = FlatSenderConfig { manifest_every_frames: 10, transfer_id: 77, format: CellFormat { pitch: 3, alphabet: Alphabet::Bw }, ..FlatSenderConfig::default() };
    let mut s = FlatSender::new(&src, cfg, Some(false)).unwrap();
    let mut rx = RxSession::new();
    let mut last = String::new();
    for n in 0..60u64 {
        let cam = bend(&render(&s.next_frame().unwrap()), 1.5, n);
        last = rx.push_rgba(&rgba(&cam), cam.width, cam.height);
        if rx.is_complete() { break; }
    }
    assert!(rx.is_complete(), "not complete: {last}");
    assert!(last.contains("\"complete\":true") && last.contains("\"filename\":\"web.bin\""), "{last}");
    assert!(last.contains("\"batch\":null"), "a single file has no batch: {last}");
    // the verified file's SHA-256, for the Done screen (the receiver checked the file against it)
    let want = format!("\"sha256\":\"{}\"", sha_hex(&std::fs::read(&src).unwrap()));
    assert!(last.contains(&want), "the status names the file's SHA-256 ({want}): {last}");
    let (name, data) = rx.take_file().expect("file");
    assert_eq!(name, "web.bin");
    assert!(data == std::fs::read(&src).unwrap(), "content differs");
}

#[test]
fn session_reports_why_a_frame_did_not_decode() {
    let mut rx = RxSession::new();
    let status = rx.push_rgba(&vec![0u8; 64 * 48 * 4], 64, 48);
    assert!(status.contains("\"last_error\":\"TooBlurry"), "{status}");
    assert!(!rx.is_complete());
}

/// A JSON object's own keys (sorted): strings at depth 1 followed by ':' (string values may hold escapes).
fn keys(j: &str) -> Vec<String> {
    let (mut keys, mut depth, mut chars) = (Vec::new(), 0i32, j.chars().peekable());
    while let Some(c) = chars.next() {
        match c {
            '{' | '[' => depth += 1,
            '}' | ']' => depth -= 1,
            '"' => {
                let mut t = String::new();
                while let Some(c) = chars.next() { match c { '\\' => { chars.next(); } '"' => break, c => t.push(c) } }
                if depth == 1 && chars.peek() == Some(&':') { keys.push(t); }
            }
            _ => {}
        }
    }
    keys.sort();
    keys
}

/// The status JSON carries exactly these keys: the receiver's own measurements, nothing else.
#[test]
fn status_carries_exactly_its_own_fields() {
    let s = lensbolt_core::flat::session::RxSession::new();
    let j = s.status_json();
    let mut want = ["frames", "reanchors", "decoded", "failed", "last_error", "format", "edges", "cell_px", "have_manifest", "filename", "received_bytes", "stream_size", "disk_peak", "seconds", "mbit_per_s", "sender_fps", "complete", "receiver_error", "finish_words", "finish_parts", "codewords_ok", "codewords", "sharpness", "mix", "edge_px", "data_reads", "corners", "batch", "sha256", "tune_sender", "tune", "hint"].map(String::from).to_vec();
    want.sort();
    assert_eq!(keys(&j), want, "status JSON: {j}");
    assert!(j.contains("\"batch\":null"), "no batch before a manifest: {j}");
    assert!(j.contains("\"sha256\":null"), "no hash before a verified file: {j}");
}

/// A SHA-256 as the status and the batch page write it: 64 lower-case hex digits.
fn sha_hex(data: &[u8]) -> String { use sha2::{Digest, Sha256}; Sha256::digest(data).iter().map(|b| format!("{b:02x}")).collect() }

/// The text of the object or array that follows `"key":` in `j` (brackets balanced; no brackets inside its strings).
fn object<'a>(j: &'a str, key: &str) -> &'a str {
    let at = j.find(&format!("\"{key}\":")).unwrap_or_else(|| panic!("no {key} in {j}")) + key.len() + 3;
    let mut depth = 0;
    for (i, c) in j[at..].char_indices() {
        match c { '{' | '[' => depth += 1, '}' | ']' => { depth -= 1; if depth == 0 { return &j[at..at + i + 1]; } } _ => {} }
        if depth == 0 { break; }
    }
    panic!("{key} is not an object or array in {j}")
}
/// The files of a batch page, as (name, state) pairs.
fn page_files(j: &str) -> Vec<(String, String)> {
    object(j, "files").split("{\"name\":").skip(1).map(|f| (field(&format!("{{\"name\":{f}"), "name").to_string(), field(f, "state").to_string())).collect()
}

/// A batch through the camera: the status carries its counts (exactly these keys), the page its files with their
/// states. A single file's page is empty.
#[test]
fn a_batch_shows_its_counts_and_pages_its_files() {
    let files: Vec<(String, Vec<u8>)> = (0..3u64).map(|i| (format!("pic-{i}.jpg"), { let mut s = 0x77 + i; (0..20_000 + 7_000 * i as usize).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s as u8 }).collect() })).collect();
    let cfg = FlatSenderConfig { manifest_every_frames: 10, format: CellFormat { pitch: 3, alphabet: Alphabet::Bw }, ..FlatSenderConfig::default() };
    let src = files.iter().map(|(n, d)| (n.clone(), lensbolt_core::file::BatchSource::Bytes(d.clone()))).collect();
    let mut s = FlatSender::from_batch("holiday", src, cfg).unwrap();
    let mut rx = RxSession::new();
    let mut last = String::new();
    for _ in 0..80 {
        let cam = render(&s.next_frame().unwrap());
        last = rx.push_rgba(&rgba(&cam), cam.width, cam.height);
        if rx.is_complete() { break; }
    }
    assert!(rx.is_complete(), "not complete: {last}");
    let b = object(&last, "batch");
    let want: Vec<String> = { let mut w = ["name", "total", "received", "failed", "missing_files", "error"].map(String::from).to_vec(); w.sort(); w };
    assert_eq!(keys(b), want, "{b}");
    assert_eq!((field(b, "name"), field(b, "total"), field(b, "received"), field(b, "failed"), field(b, "missing_files"), field(b, "error")), ("holiday", "3", "3", "0", "null", "null"), "{b}");
    let page = rx.batch_page(0, 10);
    assert_eq!(keys(&page), ["files", "offset", "total"].map(String::from).to_vec(), "{page}");
    assert_eq!((field(&page, "offset"), field(&page, "total")), ("0", "3"), "{page}");
    let got = page_files(&page);
    assert_eq!(got, files.iter().map(|(n, _)| (n.clone(), "received".to_string())).collect::<Vec<_>>(), "{page}");
    let first = object(&page, "files");
    assert_eq!(keys(&first[1..first.find('}').unwrap() + 1]), ["name", "sha256", "size", "state"].map(String::from).to_vec(), "a received file: its hash, no progress, no reason: {page}");
    for (n, d) in &files { assert!(page.contains(&format!("\"name\":\"{n}\",\"size\":{},\"state\":\"received\",\"sha256\":\"{}\"", d.len(), sha_hex(d))), "{n}: its verified SHA-256 in the page: {page}"); }
    assert!(!last.contains("\"sha256\":\""), "a batch has no single file's hash in the status: {last}");
    assert!(page.contains("\"size\":20000,"), "{page}");
    // the list without JSON, and an in-memory session's files handed over once each (the browser's fallback)
    let states = rx.batch_files(0, 10);
    assert_eq!(states.iter().map(|f| (f.name.clone(), f.size)).collect::<Vec<_>>(), files.iter().map(|(n, d)| (n.clone(), d.len() as u64)).collect::<Vec<_>>());
    for (i, (n, d)) in files.iter().enumerate() {
        assert_eq!(rx.take_batch_file(i), Some((n.clone(), d.clone())), "file {i}");
        assert_eq!(rx.take_batch_file(i), None, "file {i} is handed over once");
    }
    // a single file: no batch
    let one = RxSession::new();
    assert_eq!(one.batch_page(0, 10), "{\"offset\":0,\"total\":0,\"files\":[]}");
}

/// Pages of a 10,000-file list: each file once, in order, at most `BATCH_PAGE_MAX` a page; a count of 0 and an offset
/// at or past the end give no files; a page shows a file still coming with its progress.
#[test]
fn batch_pages_cover_the_list_within_bounds() {
    use lensbolt_core::file::{BatchSource, FeedOutcome, Receiver};
    use lensbolt_core::flat::live::{batch_page_json, BATCH_PAGE_MAX};
    let names: Vec<String> = (0..10_000u64).map(|i| format!("file-{i:05}.bin")).collect();
    let src = names.iter().enumerate().map(|(i, n)| (n.clone(), BatchSource::Bytes(vec![i as u8; (i * 7919) % 2048]))).collect();
    let mut tx = FlatSender::from_batch("ten thousand", src, FlatSenderConfig { format: CellFormat { pitch: 2, alphabet: Alphabet::Bw }, ..FlatSenderConfig::default() }).unwrap();
    let mut rx = Receiver::in_memory();
    assert_eq!(batch_page_json(&rx, 0, 10), "{\"offset\":0,\"total\":0,\"files\":[]}", "nothing yet");
    let pass = tx.frames_per_pass();
    let mut part = None;
    for n in 0..2 * pass {
        let (h, p) = tx.next_packets().unwrap();
        if let FeedOutcome::Complete(_) = rx.feed_packets(&h, &p).unwrap() { break; }
        // halfway: some files in, some coming
        if n == pass / 2 { part = Some((0..10).map(|k| batch_page_json(&rx, k * BATCH_PAGE_MAX, BATCH_PAGE_MAX)).collect::<String>()); }
    }
    let part = part.expect("a page halfway through");
    let i = part.find("\"state\":\"receiving\"").expect("a file under way halfway through");
    assert!(part[i..].starts_with("\"state\":\"receiving\",\"progress\":0."), "{}", &part[i..i + 60]);
    assert!(part.contains("\"state\":\"received\""), "files in halfway through");
    let order: Vec<String> = rx.batch_status().unwrap().files.into_iter().map(|f| f.name).collect();
    let mut seen = Vec::new();
    for page in 0..11u32 {
        let j = batch_page_json(&rx, page * BATCH_PAGE_MAX, BATCH_PAGE_MAX);
        assert_eq!((field(&j, "offset"), field(&j, "total")), ((page * BATCH_PAGE_MAX).to_string().as_str(), "10000"));
        seen.extend(page_files(&j).into_iter().map(|(n, st)| { assert_eq!(st, "received", "{n}"); n }));
    }
    assert_eq!(seen, order, "every file once, in the list's order");
    assert_eq!(page_files(&batch_page_json(&rx, 0, 5_000)).len(), BATCH_PAGE_MAX as usize, "a page holds at most BATCH_PAGE_MAX");
    assert_eq!(page_files(&batch_page_json(&rx, 9_995, 10)).len(), 5, "the end of the list");
    for (offset, count) in [(0, 0), (10_000, 10), (10_001, 1), (u32::MAX, u32::MAX)] {
        let j = batch_page_json(&rx, offset, count);
        assert_eq!(j, format!("{{\"offset\":{offset},\"total\":10000,\"files\":[]}}"), "offset {offset}, count {count}");
    }
}

/// The value of a top-level key of the status JSON: a number, `null`, or a string's text (no escapes in
/// the values read here).
fn field<'a>(j: &'a str, key: &str) -> &'a str {
    let at = j.find(&format!("\"{key}\":")).unwrap_or_else(|| panic!("no {key} in {j}")) + key.len() + 3;
    let rest = &j[at..];
    if let Some(s) = rest.strip_prefix('"') { &s[..s.find('"').unwrap()] } else { &rest[..rest.find([',', '}']).unwrap()] }
}

#[test]
fn finish_words_appear_once_repair_starts() {
    // one segment of three chunk units, the last 100 blocks long. The pass spreads every chunk over its whole
    // length; its frames that carry no block of chunk 2 (a twin sender shows each frame's packets) never reach
    // the camera: chunk 2 completes, chunks 0 and 1 stay short. No words while the first pass runs (it may
    // still bring them), words naming chunks 0 and 1 from the first repair frame on, the same on the frames
    // after it
    let mut seed = 0x5eed_u64;
    let data: Vec<u8> = (0..(2 << 20) + 100 * 1024).map(|_| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed as u8 }).collect();
    // 4 px: a few packets per frame, so most frames carry no block of chunk 2
    let cfg = FlatSenderConfig { format: CellFormat { pitch: 4, alphabet: Alphabet::Bw }, ..FlatSenderConfig::default() };
    let mut s = FlatSender::from_bytes("finish.bin", data.clone(), cfg.clone()).unwrap();
    let mut twin = FlatSender::from_bytes("finish.bin", data, cfg).unwrap();
    let m = s.manifest().clone();
    assert_eq!((m.segment_count(), m.finish_units()), (1, 3));
    let pass = s.frames_per_pass();
    let mut rx = RxSession::new();
    let (mut id, mut fed, mut words, mut after) = (None, 0u64, None::<String>, 0);
    // chunk 2's blocks fed so far, and the first-pass frames fed once it was complete: the frames on which "no
    // words before repair" is really asked (a chunk is in, others are short)
    let (mut chunk2, mut asked) = (std::collections::HashSet::new(), 0);
    let t = std::time::Instant::now();
    for n in 0..pass + 40 {
        let f = s.next_frame().unwrap();
        let (_, packets) = twin.next_packets().unwrap();
        let h = f.header;
        let id = *id.get_or_insert(h.transfer_id);
        let (chunk, pid) = ((h.block_id >> 12) as u8, h.block_id & 0xFFF);
        let data = h.frame_type == lensbolt_core::header::FrameType::Data;
        let repair = data && pid as usize >= lensbolt_core::file::chunk_blocks(m.chunk_len(h.segment, chunk));
        if data && !repair && !packets.iter().any(|p| p.id.chunk == 2) { continue; } // withheld
        let img = render(&f);
        let status = rx.push_rgba(&rgba(&img), img.width, img.height);
        fed += 1;
        assert_eq!(field(&status, "decoded"), fed.to_string(), "frame {n} did not decode: {status}");
        chunk2.extend(packets.iter().filter(|p| p.id.segment == 1 && p.id.chunk == 2 && p.id.pid < 100).map(|p| p.id.pid));
        if !repair && chunk2.len() == 100 { asked += 1; }
        let w = field(&status, "finish_words");
        match &words {
            None if !repair => assert!(w == "null" && field(&status, "finish_parts") == "null", "frame {n} (first pass, {pass} frames): {status}"),
            None => {
                assert_ne!(w, "null", "frame {n}, the first repair frame: no words: {status}");
                assert_eq!(lensbolt_core::finish::decode(id, 3, w), Ok(vec![0, 1]), "\"{w}\"");
                assert_eq!(field(&status, "finish_parts"), "Parts 1\u{2013}2", "a single file's line is unchanged: {status}");
                assert!(status.contains("\"batch\":null"), "{status}");
                words = Some(w.to_string());
            }
            Some(first) => {
                assert_eq!(w, first, "frame {n}, {after} frames after the first repair frame: {status}");
                after += 1;
                if after == 5 { break; }
            }
        }
    }
    let w = words.expect("a repair frame within 40 frames of the pass");
    assert_eq!(after, 5, "5 frames after the first repair frame");
    assert!(asked >= 1, "no first-pass frame was fed after chunk 2 was complete: the rule was never asked");
    eprintln!("{fed} frames fed of a {pass}-frame pass in {:.1} s, {asked} of them in the pass with chunk 2 complete; words \"{w}\"", t.elapsed().as_secs_f64());
}

/// A camera picture as the browser's `VideoFrame.copyTo` hands it out: I420 planes in one buffer, each plane with
/// its own offset and row stride (here padded past the width, as real layouts may be). BT.601 full range.
fn i420(img: &lensbolt_core::image::RgbImage, pad: usize) -> (Vec<u8>, [usize; 6]) {
    let (w, h) = (img.width as usize, img.height as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let (ys, cs) = (w + pad, cw + pad);
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

/// The browser receiver's fast path: camera planes straight from the video frame (no RGBA readback), the colour
/// manifest frames included.
#[test]
fn session_receives_a_file_from_i420_frames() {
    let d = std::env::temp_dir().join("lensbolt_session_i420"); std::fs::create_dir_all(&d).unwrap();
    let src = d.join("planes.bin"); pseudo_random_file(&src, 90_000, 37).unwrap();
    let cfg = FlatSenderConfig { manifest_every_frames: 10, transfer_id: 79, format: CellFormat { pitch: 3, alphabet: Alphabet::Bw }, ..FlatSenderConfig::default() };
    let mut s = FlatSender::new(&src, cfg, Some(false)).unwrap();
    let mut rx = RxSession::new();
    let mut last = String::new();
    for n in 0..60u64 {
        let cam = bend(&render(&s.next_frame().unwrap()), 1.5, n);
        let (buf, l) = i420(&cam, 16);
        last = rx.push_i420_at(&buf, cam.width, cam.height, l, n as f64 * 33.3);
        if rx.is_complete() { break; }
    }
    assert!(rx.is_complete(), "not complete: {last}");
    let (name, data) = rx.take_file().expect("file");
    assert_eq!(name, "planes.bin");
    assert!(data == std::fs::read(&src).unwrap(), "content differs");
}

#[test]
fn i420_frame_too_short_for_its_layout_is_reported() {
    let mut rx = RxSession::new();
    let status = rx.push_i420_at(&vec![0u8; 100], 64, 48, [0, 64, 3072, 32, 3840, 32], 0.0);
    assert!(status.contains("\"last_error\":\"BadFrame\""), "{status}");
    // a layout whose numbers overflow is refused, not wrapped around
    let status = rx.push_i420_at(&vec![0u8; 100], 64, 48, [usize::MAX; 6], 0.0);
    assert!(status.contains("\"last_error\":\"BadFrame\""), "{status}");
}

/// What the page's camera control reads: where the square is in the picture (its corners, camera px: how much
/// of the picture it fills, how far zoom can go before it leaves), how well its codewords read (the guidance
/// never says "closer" while they do), and the sharpness gate's score of the latest frame (focus).
#[test]
fn status_gives_the_square_its_codewords_and_the_sharpness() {
    use lensbolt_core::testutil::photograph_with_homography;
    let d = std::env::temp_dir().join("lensbolt_session"); std::fs::create_dir_all(&d).unwrap();
    let src = d.join("geom.bin"); pseudo_random_file(&src, 40_000, 3).unwrap();
    let cfg = FlatSenderConfig { manifest_every_frames: 10, transfer_id: 78, format: CellFormat { pitch: 3, alphabet: Alphabet::Bw }, ..FlatSenderConfig::default() };
    let mut s = FlatSender::new(&src, cfg, Some(false)).unwrap();
    s.next_frame().unwrap();
    let frame = render(&s.next_frame().unwrap());
    let (cam, f2c) = photograph_with_homography(&frame, 2000, 1800, 6.0, 2, 4);
    let mut rx = RxSession::new();
    let j = rx.push_rgba(&rgba(&cam), cam.width, cam.height);
    let nums = |key: &str| -> Vec<f64> {
        let v = field(&j, key);
        let at = j.find(&format!("\"{key}\":[")).unwrap_or_else(|| panic!("{key} is {v} in {j}")) + key.len() + 4;
        j[at..at + j[at..].find(']').unwrap()].split(',').map(|x| x.parse().unwrap()).collect()
    };
    let corners = nums("corners");
    let (w, h) = (frame.width as f32, frame.height as f32);
    for (i, (x, y)) in [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)].into_iter().enumerate() {
        let want = f2c.apply(x, y);
        let got = (corners[2 * i] as f32, corners[2 * i + 1] as f32);
        assert!((got.0 - want.0).hypot(got.1 - want.1) < 3.0, "corner {i}: {got:?} against {want:?}: {j}");
    }
    let (ok, all): (u64, u64) = (field(&j, "codewords_ok").parse().unwrap(), field(&j, "codewords").parse().unwrap());
    assert!(all > 0 && ok == all, "a clean frame reads all its codewords: {j}");
    let sharp: f32 = field(&j, "sharpness").parse().unwrap();
    let gate = lensbolt_core::detect::gate_sharpness(&cam.to_gray(), 4.0);
    assert!(sharp > 4.0 && (sharp - gate).abs() <= gate * 0.02 + 0.01, "sharpness {sharp}, the gate's own score {gate}: {j}");
    // a frame with no square: the last data read's corners and sharpness stay (the status describes the square)
    let blank = rx.push_rgba(&vec![90u8; 640 * 480 * 4], 640, 480);
    assert_eq!(field(&blank, "sharpness"), field(&j, "sharpness"), "{blank}");
    assert!(blank.contains("\"corners\":["), "{blank}");
    let fresh = RxSession::new().status_json();
    assert!(fresh.contains("\"corners\":null") && fresh.contains("\"codewords\":0") && fresh.contains("\"sharpness\":0.00"), "{fresh}");
}

/// The codewords in the status are those of the latest capture's frame, also when none of them decoded (a frame
/// with nothing new used to leave the previous frame's count, e.g. the manifest's 9 of 9, in the status).
#[test]
fn codewords_follow_the_latest_frame_even_when_none_decode() {
    use lensbolt_core::testutil::photograph_with_homography;
    let d = std::env::temp_dir().join("lensbolt_session"); std::fs::create_dir_all(&d).unwrap();
    let src = d.join("cw.bin"); pseudo_random_file(&src, 40_000, 8).unwrap();
    let mut rx = RxSession::new();
    let mut good = FlatSender::new(&src, FlatSenderConfig { manifest_every_frames: 10, transfer_id: 79, format: CellFormat { pitch: 3, alphabet: Alphabet::Bw }, ..FlatSenderConfig::default() }, Some(false)).unwrap();
    good.next_frame().unwrap();
    let m = render(&good.next_frame().unwrap());
    let j = rx.push_rgba(&rgba(&m), m.width, m.height);
    let (ok, all): (u64, u64) = (field(&j, "codewords_ok").parse().unwrap(), field(&j, "codewords").parse().unwrap());
    assert!(all > 0 && ok == all, "{j}");
    // a 1 px tile photographed far too small: the header reads, the cells do not
    let mut fine = FlatSender::new(&src, FlatSenderConfig { manifest_every_frames: 10, transfer_id: 79, format: CellFormat { pitch: 1, alphabet: Alphabet::Bw }, ..FlatSenderConfig::default() }, Some(false)).unwrap();
    fine.next_frame().unwrap();
    let f = render(&fine.next_frame().unwrap());
    let (cam, _) = photograph_with_homography(&f, 1100, 1000, 3.0, 6, 5);
    let j = rx.push_rgba(&rgba(&cam), cam.width, cam.height);
    assert!(field(&j, "format").starts_with("1px bw"), "the 1 px frame must be found for this test: {j}");
    let (ok, all): (u64, u64) = (field(&j, "codewords_ok").parse().unwrap(), field(&j, "codewords").parse().unwrap());
    assert!(all > 100 && ok * 10 < all, "the 1 px frame's own codewords, nearly none read: {ok} of {all}: {j}");
}

/// The manifest (12 px colour, one frame in 31) says nothing about how the data reads: its reads leave the last data
/// read's format, geometry and codewords in the status, so a page never sees "9 of 9" flash between 3 of 111.
#[test]
fn manifest_reads_leave_the_data_read_in_the_status() {
    use lensbolt_core::testutil::photograph_with_homography;
    let d = std::env::temp_dir().join("lensbolt_session"); std::fs::create_dir_all(&d).unwrap();
    let src = d.join("mf.bin"); pseudo_random_file(&src, 40_000, 12).unwrap();
    let cfg = |pitch| FlatSenderConfig { manifest_every_frames: 10, transfer_id: 80, format: CellFormat { pitch, alphabet: Alphabet::Bw }, ..FlatSenderConfig::default() };
    let mut rx = RxSession::new();
    // before any data read: nothing about how the data reads, but the square's place (the manifest is the same tile)
    let mut s = FlatSender::new(&src, cfg(1), Some(false)).unwrap();
    let manifest = render(&s.next_frame().unwrap());
    let j0 = rx.push_rgba(&rgba(&manifest), manifest.width, manifest.height);
    assert!(field(&j0, "have_manifest") == "true" && field(&j0, "format") == "null" && field(&j0, "codewords") == "0" && field(&j0, "sharpness") == "0.00" && j0.contains("\"corners\":["), "{j0}");
    // a 1 px data frame photographed too small: found, nearly no codewords
    let (cam, _) = photograph_with_homography(&render(&s.next_frame().unwrap()), 1100, 1000, 3.0, 6, 5);
    let j1 = rx.push_rgba(&rgba(&cam), cam.width, cam.height);
    assert!(field(&j1, "format").starts_with("1px bw"), "{j1}");
    // the manifest again, read whole (and sharper, and elsewhere in the picture): the status keeps how the data
    // reads (format, codewords, sharpness) and takes the square's new place (corners, edges)
    let j2 = rx.push_rgba(&rgba(&manifest), manifest.width, manifest.height);
    assert_eq!(field(&j2, "decoded"), "3", "{j2}");
    let c2: Vec<f32> = { let i = j2.find("\"corners\":[").unwrap() + 11; j2[i..i + j2[i..].find(']').unwrap()].split(',').map(|v| v.parse().unwrap()).collect() };
    assert!(c2[0].abs() < 3.0 && c2[1].abs() < 3.0 && (c2[4] - manifest.width as f32).abs() < 3.0 && (c2[5] - manifest.height as f32).abs() < 3.0, "the manifest read's own corners: {j2}");
    assert_ne!(field(&j2, "cell_px"), field(&j1, "cell_px"), "cell size from the new geometry and the data pitch: {j2}");
    assert!(field(&j2, "cell_px").parse::<f32>().unwrap() > 0.9, "1 px data pitch times ~1 camera px per screen px: {j2}");
    for k in ["format", "codewords_ok", "codewords", "sharpness"] {
        // the key's value: up to the next ',' or '}' (an array up to its ']')
        let at = |j: &str| { let i = j.find(&format!("\"{k}\":")).unwrap(); let r = &j[i..]; let end = if r[k.len() + 3..].starts_with('[') { r.find(']').unwrap() + 1 } else { r.find([',', '}']).unwrap() }; r[..end].to_string() };
        assert_eq!(at(&j2), at(&j1), "{k} changed on a manifest read: {j1} -> {j2}");
    }
}

/// A store that cannot be written (the browser's OPFS when the phone is full): the session reports it as
/// `receiver_error` and keeps answering. A panic here traps the WebAssembly instance, and every later push throws.
#[test]
fn a_full_store_is_a_receiver_error_not_a_panic() {
    use lensbolt_core::file::{PartStore, StoreFactory};
    use std::io;
    struct Full;
    impl PartStore for Full {
        fn read_at(&mut self, _: u64, buf: &mut [u8]) -> io::Result<()> { buf.fill(0); Ok(()) }
        fn write_at(&mut self, _: u64, _: &[u8]) -> io::Result<()> { Err(io::Error::other("storage is full")) }
        fn set_len(&mut self, _: u64) -> io::Result<()> { Ok(()) }
    }
    struct FullStores;
    impl StoreFactory for FullStores {
        fn create(&mut self, _: &str) -> io::Result<Box<dyn PartStore>> { Ok(Box::new(Full)) }
        fn remove(&mut self, _: &str) {}
    }
    let d = std::env::temp_dir().join("lensbolt_session"); std::fs::create_dir_all(&d).unwrap();
    let src = d.join("full.bin"); pseudo_random_file(&src, 90_000, 41).unwrap();
    let cfg = FlatSenderConfig { manifest_every_frames: 10, transfer_id: 81, format: CellFormat { pitch: 3, alphabet: Alphabet::Bw }, ..FlatSenderConfig::default() };
    let mut s = FlatSender::new(&src, cfg, Some(false)).unwrap();
    let mut rx = RxSession::with_stores(Box::new(FullStores));
    let mut errors = 0;
    for _ in 0..12 {
        let cam = render(&s.next_frame().unwrap());
        let j = rx.push_rgba(&rgba(&cam), cam.width, cam.height);
        if field(&j, "receiver_error") != "null" { errors += 1; assert!(j.contains("storage is full"), "{j}"); }
    }
    assert!(errors > 0, "a store that cannot be written must show as receiver_error");
    assert!(!rx.is_complete());
}

/// The status is JSON whatever the clock does: a non-finite number goes out as 0, never as `inf` or `NaN`.
#[test]
fn status_numbers_are_always_finite() {
    let d = std::env::temp_dir().join("lensbolt_session"); std::fs::create_dir_all(&d).unwrap();
    let src = d.join("fin.bin"); pseudo_random_file(&src, 40_000, 42).unwrap();
    let cfg = FlatSenderConfig { manifest_every_frames: 10, transfer_id: 82, format: CellFormat { pitch: 3, alphabet: Alphabet::Bw }, ..FlatSenderConfig::default() };
    let mut s = FlatSender::new(&src, cfg, Some(false)).unwrap();
    let mut rx = RxSession::new();
    let mut j = String::new();
    for (n, t) in [0.0, 33.0, f64::INFINITY].into_iter().enumerate() {
        let cam = render(&s.next_frame().unwrap());
        j = rx.push_rgba_at(&rgba(&cam), cam.width, cam.height, t);
        assert!(field(&j, "decoded") == (n + 1).to_string(), "{j}");
    }
    for bad in ["inf", "NaN"] { assert!(!j.contains(bad), "{bad} in the status: {j}"); }
    assert_eq!(field(&j, "seconds"), "0.000", "{j}");
    // what the page compares with the free space once the manifest is in: the receiver's peak on disk
    let (peak, stream): (u64, u64) = (field(&j, "disk_peak").parse().unwrap(), field(&j, "stream_size").parse().unwrap());
    assert!(stream > 0 && peak >= stream, "{j}");
}
