//! The sender's countdown: while the user aims, the page holds the first frame (the manifest) on screen, and
//! every displayed frame of it carries the next frame number, so a receiver measures the sender's rate before
//! data starts. Holding never moves the data schedule, and the numbers carry on into the data frames.
use lensbolt_core::flat::session::RxSession;
use lensbolt_core::flat::{render, Alphabet, CellFormat, FlatSender, FlatSenderConfig};
use lensbolt_core::header::FrameType;
use lensbolt_core::testutil::pseudo_random_file;

fn cfg() -> FlatSenderConfig {
    FlatSenderConfig { manifest_every_frames: 10, transfer_id: 91, format: CellFormat { pitch: 3, alphabet: Alphabet::Bw }, repair_offset: Some(7), ..FlatSenderConfig::default() }
}
fn sender(name: &str, bytes: u64) -> FlatSender {
    let d = std::env::temp_dir().join("lensbolt_held"); std::fs::create_dir_all(&d).unwrap();
    let src = d.join(name); pseudo_random_file(&src, bytes, 5).unwrap();
    FlatSender::new(&src, cfg(), Some(false)).unwrap()
}
fn rgba(img: &lensbolt_core::image::RgbImage) -> Vec<u8> {
    let mut v = Vec::with_capacity(img.data.len() / 3 * 4);
    for p in img.data.chunks_exact(3) { v.extend_from_slice(&[p[0], p[1], p[2], 255]); }
    v
}
fn num(j: &str, key: &str) -> f64 {
    let at = j.find(&format!("\"{key}\":")).unwrap_or_else(|| panic!("no {key} in {j}")) + key.len() + 3;
    let rest = &j[at..];
    rest[..rest.find([',', '}']).unwrap()].parse().unwrap_or_else(|_| panic!("{key} in {j}"))
}

#[test]
fn held_frames_are_the_manifest_under_the_next_numbers() {
    let mut s = sender("held-a.bin", 60_000);
    let first = s.next_frame().unwrap();
    assert_eq!(first.header.frame_type, FrameType::Manifest);
    assert_eq!(first.header.frame_seq, 0);
    // 300 held frames: the numbers count on and wrap at 256, every one a manifest of the same transfer
    for i in 1..=300u32 {
        let f = s.hold_manifest_frame();
        assert_eq!(f.header.frame_type, FrameType::Manifest, "held frame {i}");
        assert_eq!(f.header.frame_seq, (i & 0xFF) as u8, "held frame {i}");
        assert_eq!((f.header.transfer_id, f.header.format), (first.header.transfer_id, first.header.format));
    }
    // holding is not sending: the schedule did not move (frames_emitted counts the schedule's frames)
    assert_eq!(s.frames_emitted(), 1);
    // the data frames carry on from the held numbers
    let next = s.next_frame().unwrap();
    assert_eq!(next.header.frame_type, FrameType::Data);
    assert_eq!(next.header.frame_seq, (301u32 & 0xFF) as u8);
}

#[test]
fn holding_leaves_the_data_schedule_as_it_was() {
    let (mut held, mut plain) = (sender("held-b.bin", 60_000), sender("held-b.bin", 60_000));
    held.next_frame().unwrap(); plain.next_frame().unwrap();
    for _ in 0..45 { held.hold_manifest_frame(); }
    for n in 0..30 {
        let (a, b) = (held.next_packets().unwrap(), plain.next_packets().unwrap());
        assert_eq!(a.0.frame_type, b.0.frame_type, "frame {n}");
        // the schedule's own manifest frames carry later fountain packets of the manifest (the held frames
        // took the earlier ones); the data frames are the same frames
        if a.0.frame_type == FrameType::Manifest { continue; }
        assert_eq!(a.0.block_id, b.0.block_id, "frame {n}");
        assert!(a.1.iter().zip(&b.1).all(|(x, y)| x.id == y.id && x.payload == y.payload), "frame {n}: packets differ");
    }
}

#[test]
fn a_receiver_reads_the_rate_from_the_countdown_and_then_the_file() {
    for fps in [30.0, 60.0] {
        let d = std::env::temp_dir().join("lensbolt_held"); std::fs::create_dir_all(&d).unwrap();
        let src = d.join("held-c.bin"); pseudo_random_file(&src, 40_000, 9).unwrap();
        let mut s = FlatSender::new(&src, cfg(), Some(false)).unwrap();
        let mut rx = RxSession::new();
        let period = 1000.0 / fps;
        // 2 s of countdown: the first frame, then held frames, one per displayed frame
        let shown = (2.0 * fps) as u32;
        let mut st = String::new();
        for i in 0..shown {
            let f = if i == 0 { s.next_frame().unwrap() } else { s.hold_manifest_frame() };
            let cam = render(&f);
            st = rx.push_rgba_at(&rgba(&cam), cam.width, cam.height, i as f64 * period);
        }
        let rate = num(&st, "sender_fps");
        assert!((rate - fps).abs() <= fps * 0.05, "{fps} fps countdown read as {rate}: {st}");
        assert!(st.contains("\"have_manifest\":true"), "{st}");
        assert_eq!(num(&st, "received_bytes"), 0.0, "a countdown carries no data: {st}");
        // then the data: the rate holds through the change from countdown to data (no restart), the file completes
        for i in shown..shown + 400 {
            let cam = render(&s.next_frame().unwrap());
            st = rx.push_rgba_at(&rgba(&cam), cam.width, cam.height, i as f64 * period);
            let r = num(&st, "sender_fps");
            assert!((r - fps).abs() <= fps * 0.05, "{fps} fps: frame {i} read {r}");
            if rx.is_complete() { break; }
        }
        assert!(rx.is_complete(), "{st}");
        assert!(!st.contains("\"receiver_error\":\""), "{st}");
        assert!(rx.take_file().unwrap().1 == std::fs::read(&src).unwrap());
    }
}
