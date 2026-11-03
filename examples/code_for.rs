//! The two calibration words that select a format on the sender: `cargo run --example code_for -- 1 bw 30`
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let pitch: u8 = a.first().and_then(|v| v.parse().ok()).unwrap_or(1);
    let alphabet = lensbolt_core::flat::Alphabet::from_name(a.get(1).map(|s| s.as_str()).unwrap_or("bw")).expect("alphabet");
    let fps: u32 = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(30);
    let fi = lensbolt_core::calib::FPS_OPTIONS.iter().position(|&f| f as u32 == fps).expect("fps") as u8;
    println!("{}", lensbolt_core::calib::Config { pitch, alphabet, fps: fi }.to_code());
}
