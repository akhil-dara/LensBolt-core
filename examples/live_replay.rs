//! A phone dump folder through the live receiver at the camera's pace (30 fps, as the app offers captures):
//! captures dropped while every worker was busy, decoded, failed, the mean decode time and the packets.
//!   cargo run --release --example live_replay -- <dir> [workers] [rounds]
use lensbolt_core::flat::live::{Capture, LiveConfig, LiveReceiver};
use lensbolt_core::image::GrayImage;
use std::path::PathBuf;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let dir = PathBuf::from(&a[0]);
    // the stored phone dumps are of version 8
    lensbolt_core::flat::header::ACCEPT_V8_HEADERS.store(true, std::sync::atomic::Ordering::Relaxed);
    let workers: usize = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(LiveConfig::default().workers);
    let rounds: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(1);
    let mut stems: Vec<String> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter_map(|n| n.strip_suffix(".y").map(String::from)).collect();
    stems.sort_by_key(|s| s.split('_').nth(1).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0));
    let frames: Vec<GrayImage> = stems.iter().map(|stem| {
        let f: Vec<&str> = stem.split('_').collect();
        let d: Vec<usize> = f[2].split('x').map(|v| v.parse().unwrap()).collect();
        let rs: usize = f[3].parse().unwrap();
        let raw = std::fs::read(dir.join(format!("{stem}.y"))).unwrap();
        let mut data = Vec::with_capacity(d[0] * d[1]);
        for y in 0..d[1] { data.extend_from_slice(&raw[y * rs..y * rs + d[0]]); }
        GrayImage { width: d[0] as u32, height: d[1] as u32, data }
    }).collect();
    let out = std::env::temp_dir().join(format!("lb-live-replay-{}", std::process::id()));
    let live = LiveReceiver::new(&out, LiveConfig { workers }).unwrap();
    let start = std::time::Instant::now();
    let mut n = 0u32;
    for _ in 0..rounds { for g in &frames {
        let due = start + std::time::Duration::from_micros(33_333 * n as u64);
        if let Some(wait) = due.checked_duration_since(std::time::Instant::now()) { std::thread::sleep(wait); }
        let g = g.clone();
        if live.has_room() { live.submit_capture(move || Capture { y: g, rgb: None, origin: (0, 0) }); } else { live.skip(); }
        n += 1;
    } }
    std::thread::sleep(std::time::Duration::from_millis(500));
    let st = live.status();
    println!("workers {workers}: offered {n}, dropped {}, decoded {}, failed {}, decode {:.1} ms, codewords {}/{}, received bytes {}",
        st.dropped, st.decoded, st.failed, st.decode_ms, st.codewords_ok, st.codewords, st.received_bytes);
    drop(live);
    let _ = std::fs::remove_dir_all(&out);
}
