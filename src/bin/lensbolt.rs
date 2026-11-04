use clap::{Parser, Subcommand, ValueEnum};
use lensbolt_core::calib::{Calibrator, Config};
use lensbolt_core::file::{FeedOutcome, Receiver};
use lensbolt_core::flat::{render_to, Alphabet, CellFormat, FlatDecoder, FlatGrid, FlatHint, FlatSender, FlatSenderConfig, MANIFEST_EVERY_FRAMES, PITCHES};
use lensbolt_core::image::RgbImage;
use lensbolt_core::packet::unpack_frame;
use lensbolt_core::flat::DecodeError;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "lensbolt", version, about = "LensBolt optical file transfer: encode files to frames, decode frames to files")]
struct Cli { #[command(subcommand)] cmd: Cmd }

#[derive(Clone, Copy, ValueEnum)] enum Compress { Auto, On, Off }

#[derive(Subcommand)]
enum Cmd {
    /// Encode FILE into PNG frames
    Encode { file: PathBuf, #[arg(short, long)] out: PathBuf, #[arg(long, default_value_t = 1024)] width: u32, #[arg(long, default_value_t = 1024)] height: u32,
             #[arg(long, default_value_t = 1.0)] passes: f64, #[arg(long, default_value_t = MANIFEST_EVERY_FRAMES)] manifest_every: u32,
             #[arg(long, value_enum, default_value_t = Compress::Auto)] compress: Compress,
             /// data cell pitch in screen px (1, 2, 3, 4, 5, 6, 8, 12)
             #[arg(long, default_value_t = 4)] pitch: u8,
             /// colour alphabet: bw, g4, g8, c4, c8, c16
             #[arg(long, default_value = "c8")] alphabet: String },
    /// Decode PNG frames (files or directories) into OUT
    Decode { inputs: Vec<PathBuf>, #[arg(short, long)] out: PathBuf },
    /// Diagnose why a photo does or does not decode: per-stage report
    Diag { image: PathBuf, #[arg(long)] truth: Option<PathBuf>,
        /// reader overrides for sweeps on real captures: alignment step (fraction of the sampling), pooling
        /// radius in blocks, largest crosstalk factor tried, box window (fraction of a cell), passes
        #[arg(long)] step_div: Option<f32>, #[arg(long)] pool: Option<u32>, #[arg(long)] alpha_max: Option<f32>, #[arg(long)] window: Option<f32>, #[arg(long)] passes: Option<u32>, #[arg(long)] no_local_threshold: bool, #[arg(long)] erasure_margin: Option<u16> },
    /// Streaming receiver: raw RGB frames on stdin (u32 LE width, u32 LE height, then w*h*3 bytes, repeated),
    /// decoded on a thread pool, JSON progress lines on stdout
    Recv { #[arg(short, long)] out: PathBuf, #[arg(long, default_value_t = 4)] threads: usize,
        /// frames arrive as BGR (OpenCV native) and are swapped here
        #[arg(long)] bgr: bool,
        /// calibration: measure the sender's opening carousel and print the suggested words
        #[arg(long)] calibrate: bool,
        /// camera frame rate, for the calibration suggestion
        #[arg(long, default_value_t = 30.0)] camera_fps: f64,
        /// never drop frames when the decoders fall behind (offline simulations)
        #[arg(long)] no_drop: bool },
    /// Live sender: render frames of FILE forever (fresh fountain packets every pass) as raw RGB on stdout:
    /// u32 LE width, u32 LE height, then width*height*3 bytes, repeated. Stops when stdout closes.
    Stream { file: PathBuf, #[arg(long, default_value_t = 1024)] width: u32, #[arg(long, default_value_t = 1024)] height: u32,
             #[arg(long, default_value_t = MANIFEST_EVERY_FRAMES)] manifest_every: u32, #[arg(long, value_enum, default_value_t = Compress::Auto)] compress: Compress,
             #[arg(long, default_value_t = 4)] pitch: u8, #[arg(long, default_value = "c8")] alphabet: String,
             /// opening calibration carousel (every configuration in turn) until the receiver's words are entered
             #[arg(long)] calibration: bool,
             /// the two words the receiver showed (sets pitch, alphabet and fps; overrides --pitch/--alphabet)
             #[arg(long)] code: Option<String> },
    /// Rebuild the exact frame a stream showed (for measuring real captures cell by cell): data frame of FILE
    /// whose first packet is BLOCK, with the stream's transfer id and frame sequence number, as a PNG
    Truth { file: PathBuf, #[arg(short, long)] out: PathBuf, #[arg(long)] block: u32, #[arg(long)] transfer: u16, #[arg(long)] seq: u8,
            #[arg(long, default_value_t = 1024)] width: u32, #[arg(long, default_value_t = 1024)] height: u32,
            #[arg(long, default_value_t = 1)] pitch: u8, #[arg(long, default_value = "bw")] alphabet: String },
    /// Show what a transfer of FILE would look like
    Info { file: PathBuf, #[arg(long, default_value_t = 1024)] width: u32, #[arg(long, default_value_t = 1024)] height: u32, #[arg(long, default_value_t = 4)] pitch: u8, #[arg(long, default_value = "c8")] alphabet: String },
}

fn compress_opt(c: Compress) -> Option<bool> { match c { Compress::Auto => None, Compress::On => Some(true), Compress::Off => Some(false) } }

fn format_of(pitch: u8, alphabet: &str) -> Result<CellFormat, String> {
    if !PITCHES.contains(&pitch) { return Err(format!("pitch {pitch} not in {PITCHES:?}")); }
    let alphabet = Alphabet::from_name(alphabet).ok_or_else(|| format!("alphabet {alphabet:?} not one of bw, g4, g8, c4, c8, c16"))?;
    Ok(CellFormat { pitch, alphabet })
}

fn main() {
    let cli = Cli::parse();
    let code = match cli.cmd {
        Cmd::Encode { file, out, width, height, passes, manifest_every, compress, pitch, alphabet } => encode(&file, &out, width, height, passes, manifest_every, compress_opt(compress), pitch, &alphabet),
        Cmd::Decode { inputs, out } => decode(&inputs, &out),
        Cmd::Info { file, width, height, pitch, alphabet } => info(&file, width, height, pitch, &alphabet),
        Cmd::Truth { file, out, block, transfer, seq, width, height, pitch, alphabet } => truth(&file, &out, block, transfer, seq, width, height, pitch, &alphabet),
        Cmd::Diag { image, truth, step_div, pool, alpha_max, window, passes, no_local_threshold, erasure_margin } => {
            if let Some(m) = erasure_margin { lensbolt_core::flat::extract::ERASURE_MARGIN_OVERRIDE.store(m, std::sync::atomic::Ordering::Relaxed); }
            diag(&image, truth.as_deref(), &DiagOpts { step_div, pool, alpha_max, window, passes, no_local_threshold })
        }
        Cmd::Recv { out, threads, bgr, no_drop, calibrate, camera_fps } => recv(&out, threads.max(1), bgr, no_drop, calibrate, camera_fps),
        Cmd::Stream { file, width, height, manifest_every, compress, pitch, alphabet, calibration, code } => stream(&file, width, height, manifest_every, compress_opt(compress), pitch, &alphabet, calibration, code.as_deref()),
    };
    std::process::exit(code);
}

#[allow(clippy::too_many_arguments)]
fn encode(file: &Path, out: &Path, width: u32, height: u32, passes: f64, manifest_every: u32, compress: Option<bool>, pitch: u8, alphabet: &str) -> i32 {
    if let Err(e) = std::fs::create_dir_all(out) { eprintln!("cannot create {}: {e}", out.display()); return 1; }
    let format = match format_of(pitch, alphabet) { Ok(f) => f, Err(e) => { eprintln!("{e}"); return 1; } };
    let cfg = FlatSenderConfig { width, height, format, manifest_every_frames: manifest_every, ..FlatSenderConfig::default() };
    let mut s = match FlatSender::new(file, cfg, compress) { Ok(s) => s, Err(e) => { eprintln!("prepare failed: {e}"); return 1; } };
    let m = s.manifest();
    eprintln!("{}: {} bytes, stream {} bytes ({}), {} segments, {} frames per pass", m.filename, m.file_size, m.stream_size, if m.compressed { "zstd" } else { "raw" }, m.segment_count(), s.frames_per_pass());
    let total = (passes * s.frames_per_pass() as f64).ceil() as u64;
    for i in 0..total {
        let frame = match s.next_frame() { Ok(f) => f, Err(e) => { eprintln!("frame {i}: {e}"); return 1; } };
        let img = render_to(&frame, width, height);
        if let Err(e) = img.save_png(&out.join(format!("frame_{i:06}.png"))) { eprintln!("write failed: {e}"); return 1; }
        if i % 50 == 0 { eprintln!("frame {i}/{total}"); }
    }
    eprintln!("wrote {total} frames to {}", out.display());
    0
}

/// The data frame whose first packet id is `block` (chunk << 12 | pid), rebuilt from the sender's packets
/// (a packet depends only on the file and its id; the transfer id and frame sequence come from the capture).
#[allow(clippy::too_many_arguments)]
fn truth(file: &Path, out: &Path, block: u32, transfer: u16, seq: u8, width: u32, height: u32, pitch: u8, alphabet: &str) -> i32 {
    use lensbolt_core::flat::{compose, FlatHeader};
    let format = match format_of(pitch, alphabet) { Ok(f) => f, Err(e) => { eprintln!("{e}"); return 1; } };
    let cfg = FlatSenderConfig { width, height, format, manifest_every_frames: 1 << 30, transfer_id: transfer, ..FlatSenderConfig::default() };
    let mut s = match FlatSender::new(file, cfg, None) { Ok(s) => s, Err(e) => { eprintln!("prepare failed: {e}"); return 1; } };
    let grid = s.grid();
    let key = |id: &lensbolt_core::packet::PacketId| (id.chunk as u32) << 12 | id.pid as u32;
    let mut have = std::collections::BTreeMap::new();
    let mut segment = 1;
    for _ in 0..100_000 {
        let (h, packets) = match s.next_packets() { Ok(x) => x, Err(e) => { eprintln!("{e}"); return 1; } };
        if h.frame_type != lensbolt_core::header::FrameType::Data { continue; }
        for p in packets { let k = key(&p.id); if k >= block && k < block + grid.packets() as u32 { segment = p.id.segment; have.insert(k, p); } }
        if have.len() == grid.packets() { break; }
        if packets_past(&have, block, grid.packets()) { break; }
    }
    if have.len() != grid.packets() { eprintln!("only {} of {} packets from {block}", have.len(), grid.packets()); return 1; }
    let packets: Vec<_> = have.into_values().collect();
    let header = FlatHeader { format, cols: grid.cols() as u16, rows: grid.rows() as u16, frame_type: lensbolt_core::header::FrameType::Data,
        transfer_id: transfer, segment, block_id: block, frame_seq: seq };
    let frame = compose(grid.clone(), header, &lensbolt_core::packet::pack_frame(&packets, &grid));
    if let Err(e) = render_to(&frame, width, height).save_png(out) { eprintln!("write failed: {e}"); return 1; }
    eprintln!("truth frame block {block} seq {seq} transfer {transfer} -> {}", out.display());
    0
}
fn packets_past<T>(_have: &std::collections::BTreeMap<u32, T>, _block: u32, _n: usize) -> bool { false }

fn collect_pngs(inputs: &[PathBuf]) -> Vec<PathBuf> {
    let mut v = Vec::new();
    for p in inputs {
        if p.is_dir() {
            let mut d: Vec<PathBuf> = std::fs::read_dir(p).into_iter().flatten().flatten().map(|e| e.path())
                .filter(|q| q.extension().map(|x| x.eq_ignore_ascii_case("png")).unwrap_or(false)).collect();
            d.sort(); v.extend(d);
        } else { v.push(p.clone()); }
    }
    v
}

fn decode(inputs: &[PathBuf], out: &Path) -> i32 {
    let files = collect_pngs(inputs);
    if files.is_empty() { eprintln!("no PNG inputs"); return 1; }
    let mut r = match Receiver::new(out) { Ok(r) => r, Err(e) => { eprintln!("{e}"); return 1; } };
    let dec = FlatDecoder::new();
    for f in &files {
        let img = match RgbImage::load_png(f) { Ok(i) => i, Err(e) => { eprintln!("{}: {e}", f.display()); continue; } };
        match dec.decode_all(&img) {
            Err(DecodeError::TooBlurry(s)) => eprintln!("{}: blurry ({s:.1})", f.display()),
            Err(e) => eprintln!("{}: {e:?}", f.display()),
            Ok(reads) => for (k, fr) in reads.iter().enumerate() {
                if reads[..k].iter().any(|o| o.header.format == fr.header.format && (o.header.cols, o.header.rows) == (fr.header.cols, fr.header.rows)) { continue; } // same cells, same packets
                let fp = fr.payload();
                let packets = unpack_frame(&fp);
                let res = match r.feed_packets(&fr.header, &packets) { Ok(r) => r, Err(e) => { eprintln!("  receiver error: {e}"); FeedOutcome::Ignored } };
                eprintln!("{}: {:?} {}px {} seg {} block {}: codewords ok {} / failed {}, {} packets: {:?}", f.display(), fr.header.frame_type, fr.header.format.pitch, fr.header.format.alphabet.name(), fr.header.segment, fr.header.block_id, fp.codewords_ok, fp.codewords_failed, packets.len(), res);
                if let FeedOutcome::Complete(p) = res { eprintln!("COMPLETE: {}", p.display()); return 0; }
            },
        }
    }
    let p = r.progress();
    eprintln!("incomplete: manifest {} segments {}/{} packets accepted {} ignored {}", p.have_manifest, p.segments_done, p.segments_total, p.packets_accepted, p.packets_ignored);
    2
}

fn info(file: &Path, width: u32, height: u32, pitch: u8, alphabet: &str) -> i32 {
    let format = match format_of(pitch, alphabet) { Ok(f) => f, Err(e) => { eprintln!("{e}"); return 1; } };
    let cfg = FlatSenderConfig { width, height, format, ..FlatSenderConfig::default() };
    let s = match FlatSender::new(file, cfg, None) { Ok(s) => s, Err(e) => { eprintln!("{e}"); return 1; } };
    let m = s.manifest();
    let g = s.grid();
    println!("file:        {} ({} bytes)", m.filename, m.file_size);
    println!("stream:      {} bytes, {}", m.stream_size, if m.compressed { "zstd-3" } else { "raw" });
    println!("segments:    {} x 16 MiB", m.segment_count());
    println!("frame:       {width}x{height}, {pitch} px {} cells: {}x{} cells, {} data cells, {} pilots, {} codewords = {} B payload = {} packets", alphabet, g.cols(), g.rows(), g.data_cell_count(), g.pilot_positions().len(), g.codewords(), g.payload_bytes(), g.packets());
    let fpp = s.frames_per_pass();
    println!("one pass:    {fpp} frames = {:.1} s @60fps, {:.1} s @30fps", fpp as f64 / 60.0, fpp as f64 / 30.0);
    let goodput30 = s.bytes_per_second(30.0) * 8.0 / 1e6;
    println!("goodput:     {:.2} Mbit/s @30fps at 100% yield, {:.2} Mbit/s @60fps", goodput30, goodput30 * 2.0);
    0
}

struct DiagOpts { step_div: Option<f32>, pool: Option<u32>, alpha_max: Option<f32>, window: Option<f32>, passes: Option<u32>, no_local_threshold: bool }
fn diag(image: &Path, truth: Option<&Path>, o: &DiagOpts) -> i32 {
    use lensbolt_core::detect::{find_marker_candidates, sharpness, two_means_threshold, Homography};
    use lensbolt_core::flat::{read_cells, ReadParams};
    let img = match RgbImage::load_png(image) { Ok(i) => i, Err(e) => { eprintln!("{e}"); return 1; } };
    let gray = img.to_gray();
    println!("image {}x{}  sharpness {:.2} (gate 4.0)  threshold {}", img.width, img.height, sharpness(&gray), two_means_threshold(&gray.data));
    let mut cands = find_marker_candidates(&gray);
    cands.sort_by(|a, b| b.3.cmp(&a.3));
    println!("marker candidates: {} clusters, top: {:?}", cands.len(), cands.iter().take(8).map(|c| (c.0 as i32, c.1 as i32, (c.2 * 10.0).round() / 10.0, c.3)).collect::<Vec<_>>());
    let dec = FlatDecoder::new();
    let (chrome, h, headers) = match dec.identify(&img) {
        Ok(x) => x,
        Err(e) => { println!("identify: {e:?}"); return 2; }
    };
    let s = lensbolt_core::flat::sampling(&h, &FlatGrid::try_new(chrome.width_px(), chrome.height_px(), lensbolt_core::flat::MANIFEST_FORMAT).unwrap());
    println!("chrome {}x{}: {:.2} camera px per screen px; header copies: {:?}", chrome.width_px(), chrome.height_px(), s, headers.iter().map(|x| x.map(|x| format!("{:?} transfer {} seq {} {}px {} seg {} block {}", x.frame_type, x.transfer_id, x.frame_seq, x.format.pitch, x.format.alphabet.name(), x.segment, x.block_id))).collect::<Vec<_>>());
    for hdr in headers.into_iter().flatten() {
        let Some(g) = FlatGrid::try_new(chrome.width_px(), chrome.height_px(), hdr.format) else { continue };
        let mut rp = ReadParams { tag_hint: hdr.frame_seq & 3, ..ReadParams::default() }.tuned_for(hdr.format.pitch);
        if let Some(d) = o.step_div { rp.step_px = s / d; rp.auto_reach = false; rp.reach_px = 2.5 * s; }
        if let Some(p) = o.pool { rp.pool = p; }
        if let Some(a) = o.alpha_max { rp.alpha_max = a; }
        if let Some(w) = o.window { rp.window = w; }
        if let Some(p) = o.passes { rp.passes = p; }
        if o.no_local_threshold { rp.local_threshold = false; }
        let reads = read_cells(&img, &g, &h, &rp);
        let (bw, bh) = g.blocks();
        println!("grid {}x{} cells, {} codewords, {} packets; block shift map (|shift| in camera px: '.'<0.75 ':'<1.5 '+'<2.5 '#'>=2.5):", g.cols(), g.rows(), g.codewords(), g.packets());
        for by in 0..bh { let row: String = (0..bw).map(|bx| { let (dx, dy) = reads.shifts[(by * bw + bx) as usize]; let m = dx.hypot(dy); if m < 0.75 { '.' } else if m < 1.5 { ':' } else if m < 2.5 { '+' } else { '#' } }).collect(); println!("    {row}"); }
        let mut tags = [0usize; 4]; for &t in &reads.block_tag { tags[(t & 3) as usize] += 1; }
        println!("  frame tags per block: {:?} (header says {}); crosstalk alpha {}", tags, hdr.frame_seq & 3, reads.alpha);
        if tags.iter().filter(|&&n| n > 0).count() > 1 {
            for by in 0..bh { let row: String = (0..bw).map(|bx| char::from(b'0' + (reads.block_tag[(by * bw + bx) as usize] & 3))).collect(); println!("    {row}"); }
        }
        let fp = lensbolt_core::flat::extract_payload(&reads, &g);
        let packets_ok = lensbolt_core::packet::unpack_frame(&fp).len();
        println!("  packets ok {packets_ok}/{} (v8 frame-wide codewords)", g.packets());
        let low = (0..g.used_data_cells() as u32).filter(|&i| reads.data(&g, i).1 < lensbolt_core::flat::ERASURE_MARGIN).count();
        println!("  codewords ok {} / failed {}; weak-margin cells {}/{}", fp.codewords_ok, fp.codewords_failed, low, g.used_data_cells());
        let reach = if rp.auto_reach && rp.reach_px > 0.0 { rp.reach_px.max(2.5 * s) } else { rp.reach_px };
        let clipped = (0..reads.shifts.len()).filter(|&b| { let (dx, dy) = reads.shifts[b]; dx.abs().max(dy.abs()) >= reach - 0.3 }).count();
        println!("  search reach {reach:.2} px; blocks at the reach {clipped}/{}", reads.shifts.len());
        if let Some(tp) = truth {
            let timg = RgbImage::load_png(tp).expect("truth png");
            let treads = read_cells(&timg, &g, &Homography::identity(), &ReadParams { reach_px: 0.0, local_colour: false, ..ReadParams::default() });
            let n = g.used_data_cells() as u32;
            let k = hdr.format.alphabet.size();
            let mut conf = vec![vec![0u32; k]; k];
            let mut err = 0;
            for i in 0..n { let (a, b) = (reads.data(&g, i).0, treads.data(&g, i).0); if a != b { err += 1; } conf[b as usize][a as usize] += 1; }
            println!("  vs truth: cell errors {err}/{n} ({:.2}%)", err as f32 * 100.0 / n as f32);
            // LB_DUMP=path: the whole cell grid (read value, margin, true value) and the data cell order, for
            // offline analysis of real error patterns (u32 cols, rows, n; u8 read[cols*rows]; u16 margin[..];
            // u8 truth[..]; u32 data index -> row*cols+col [n])
            if let Ok(path) = std::env::var("LB_DUMP") {
                let mut buf: Vec<u8> = Vec::new();
                for v in [g.cols(), g.rows(), n] { buf.extend_from_slice(&v.to_le_bytes()); }
                buf.extend_from_slice(&reads.values);
                for m in &reads.margins { buf.extend_from_slice(&m.to_le_bytes()); }
                buf.extend_from_slice(&treads.values);
                for i in 0..n { let (c, r) = g.data_position(i); buf.extend_from_slice(&(r * g.cols() + c).to_le_bytes()); }
                let _ = std::fs::write(&path, &buf);
                println!("  dumped cells to {path}");
            }
            println!("  confusion (rows = true, cols = read): {:?}", conf);
            let mut berr = vec![0u32; (bw * bh) as usize]; let mut btot = vec![0u32; (bw * bh) as usize];
            for i in 0..n { let (c, r) = g.data_position(i); let b = ((r / 8) * bw + c / 8) as usize; btot[b] += 1; if reads.at(c, r).0 != treads.at(c, r).0 { berr[b] += 1; } }
            println!("  block error map ('.'<2% ':'<5% '+'<10% '#'>=10%):");
            for by in 0..bh { let row: String = (0..bw).map(|bx| { let b = (by * bw + bx) as usize; if btot[b] == 0 { ' ' } else { let e = berr[b] as f32 / btot[b] as f32; if e < 0.02 { '.' } else if e < 0.05 { ':' } else if e < 0.10 { '+' } else { '#' } } }).collect(); println!("    {row}"); }
        }
    }
    0
}

fn recv(out: &Path, threads: usize, bgr: bool, no_drop: bool, calibrate: bool, camera_fps: f64) -> i32 {
    use std::io::{Read, Write};
    use std::sync::{mpsc, Arc, Mutex};
    use std::time::Instant;
    let mut cal = Calibrator::new();
    let mut r = match Receiver::new(out) { Ok(r) => r, Err(e) => { eprintln!("{e}"); return 1; } };
    let dec = Arc::new(FlatDecoder::new());
    // bounded work queue: if the decoders fall behind, frames are dropped (the fountain code does not care)
    let (work_tx, work_rx) = mpsc::sync_channel::<(u64, RgbImage)>(threads);
    let work_rx = Arc::new(Mutex::new(work_rx));
    type Decoded = Option<Vec<lensbolt_core::flat::FlatRead>>;
    let mut stitcher = lensbolt_core::flat::Stitcher::new();
    type Failure = (String, f32, usize); // error kind, sharpness, strong marker candidates
    let (res_tx, res_rx) = mpsc::channel::<(u64, Decoded, f64, Option<Failure>)>();
    // tracking: every worker starts from the geometry of the latest successful decode
    let hint: Arc<Mutex<Option<FlatHint>>> = Arc::new(Mutex::new(None));
    for _ in 0..threads {
        let (rx, tx, dec, hint) = (work_rx.clone(), res_tx.clone(), dec.clone(), hint.clone());
        std::thread::spawn(move || loop {
            let job = { let guard = rx.lock().unwrap(); guard.recv() };
            let Ok((seq, img)) = job else { break };
            let h = hint.lock().unwrap().clone();
            let t = Instant::now();
            // the phone app's path: luma first, RGB only for colour frames
            let res = lensbolt_core::flat::live::decode_luma_first(&dec, &img.to_gray(), || img.clone(), h.as_ref()).map(|(f, _)| f);
            match res {
                Ok(frs) => {
                    *hint.lock().unwrap() = Some(frs[0].hint());
                    let ms = t.elapsed().as_secs_f64() * 1000.0;
                    let _ = tx.send((seq, Some(frs), ms, None));
                }
                Err(e) => {
                    // why: blur, no markers, or markers without a readable header
                    let gray = img.to_gray();
                    let strong = lensbolt_core::detect::find_marker_candidates(&gray).iter().filter(|c| c.3 >= 4).count();
                    let kind = match e { DecodeError::TooBlurry(_) => "TooBlurry", DecodeError::NoMarkers => "NoMarkers", DecodeError::NoHeader => "NoHeader", DecodeError::OtherVersion(_) => "OtherVersion", DecodeError::NeedsColour(..) => "NeedsColour" };
                    let _ = tx.send((seq, None, t.elapsed().as_secs_f64() * 1000.0, Some((kind.to_string(), lensbolt_core::detect::sharpness(&gray), strong))));
                }
            }
        });
    }
    drop(res_tx);
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    let mut stdout = std::io::stdout();
    let t0 = Instant::now();
    let (mut received, mut dropped, mut decoded, mut failed) = (0u64, 0u64, 0u64, 0u64);
    // one JSON line per decoded frame; returns Some(exit code) when the transfer completes
    macro_rules! handle {
        ($res:expr, $ms:expr, $seq:expr, $why:expr) => {{
            let ms: f64 = $ms;
            let seq: u64 = $seq;
            let why: Option<Failure> = $why;
            match $res {
                None => {
                    failed += 1;
                    let (kind, sharp, markers) = why.unwrap_or(("Unknown".to_string(), 0.0, 0));
                    let _ = writeln!(stdout, "{{\"t\":{:.3},\"decoded\":{decoded},\"failed\":{failed},\"error\":\"{kind}\",\"sharpness\":{sharp:.1},\"markers\":{markers},\"seq\":{seq},\"ms\":{ms:.1}}}", t0.elapsed().as_secs_f64());
                    let _ = stdout.flush();
                    None
                }
                Some(frs) => {
                    decoded += 1;
                    let fr = &frs[0];
                    let size = (fr.grid.width_px(), fr.grid.height_px());
                    if calibrate {
                        let fp = fr.payload();
                        cal.observe(fr.header.format, &fp, ms);
                        if decoded % 6 == 0 {
                            if let Some(sug) = cal.suggest(size.0, size.1, camera_fps, threads as f64) {
                                let yields: Vec<String> = sug.yields.iter().map(|(f, y)| format!("\"{}{}\":{y:.2}", f.pitch, f.alphabet.name())).collect();
                                let _ = writeln!(stdout, "{{\"suggest\":{{\"code\":\"{}\",\"pitch\":{},\"alphabet\":\"{}\",\"fps\":{},\"expected_mbit\":{:.2},\"yields\":{{{}}},\"captures\":{}}}}}",
                                    sug.config.to_code(), sug.config.pitch, sug.config.alphabet.name(), sug.config.fps_value(), sug.expected_mbit, yields.join(","), cal.captures());
                            }
                        }
                    }
                    // the stitcher merges this capture's blocks into their frames (two when the capture straddles
                    // a refresh) and reports every stripe that completed now, once
                    let outs = stitcher.add_capture(&frs);
                    let mut code = None;
                    for (header, fp) in outs {
                        let packets = unpack_frame(&fp);
                        let outcome = r.feed_packets(&header, &packets).unwrap_or(FeedOutcome::Ignored);
                        let p = r.progress();
                        let done = if let FeedOutcome::Complete(path) = &outcome { Some(path.display().to_string()) } else { None };
                        let _ = writeln!(stdout, "{{\"t\":{:.3},\"received\":{received},\"dropped\":{dropped},\"decoded\":{decoded},\"failed\":{failed},\"type\":\"{:?}\",\"pitch\":{},\"alphabet\":\"{}\",\"cols\":{},\"block\":{},\"frame_seq\":{},\"codewords_ok\":{},\"codewords_failed\":{},\"codewords\":{},\"coverage\":{:.3},\"packets_ok\":{},\"packets_frame\":{},\"segments\":{},\"segments_total\":{},\"bytes_done\":{},\"packets\":{},\"packets_new\":{},\"seq\":{seq},\"ms\":{ms:.1},\"complete\":{}}}",
                            t0.elapsed().as_secs_f64(), header.frame_type, header.format.pitch, header.format.alphabet.name(), header.cols, header.block_id, header.frame_seq, fp.codewords_ok, fp.codewords_failed, fp.codewords_ok + fp.codewords_failed + fp.codewords_missing, fp.coverage, packets.len(), fp.bytes.len() / lensbolt_core::packet::PACKET_BYTES, p.segments_done, p.segments_total, p.bytes_done, p.packets_accepted, p.packets_new,
                            match &done { Some(d) => format!("\"{}\"", d.replace('\\', "/")), None => "null".into() });
                        let _ = stdout.flush();
                        if done.is_some() { code = Some(0); break; }
                    }
                    code
                }
            }
        }};
    }
    let mut hdr = [0u8; 8];
    loop {
        if input.read_exact(&mut hdr).is_err() { break; }
        let (w, h) = (u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]), u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]));
        if w == 0 || h == 0 || w > 8192 || h > 8192 { break; }
        let mut img = RgbImage::new(w, h);
        if input.read_exact(&mut img.data).is_err() { break; }
        if bgr { for px in img.data.chunks_exact_mut(3) { px.swap(0, 2); } }
        received += 1;
        cal.capture();
        if no_drop {
            if work_tx.send((received, img)).is_err() { break; }
        } else {
            match work_tx.try_send((received, img)) { Ok(()) => {}, Err(mpsc::TrySendError::Full(_)) => dropped += 1, Err(_) => break }
        }
        while let Ok((seq, res, ms, why)) = res_rx.try_recv() {
            if let Some(code) = handle!(res, ms, seq, why) { return code; }
        }
    }
    // EOF: let the in-flight decodes finish and account for them
    drop(work_tx);
    while let Ok((seq, res, ms, why)) = res_rx.recv() {
        if let Some(code) = handle!(res, ms, seq, why) { return code; }
    }
    let _ = writeln!(stdout, "{{\"t\":{:.3},\"received\":{received},\"dropped\":{dropped},\"decoded\":{decoded},\"failed\":{failed},\"complete\":null,\"end\":true}}", t0.elapsed().as_secs_f64());
    2
}

#[allow(clippy::too_many_arguments)]
fn stream(file: &Path, width: u32, height: u32, manifest_every: u32, compress: Option<bool>, pitch: u8, alphabet: &str, calibration: bool, code: Option<&str>) -> i32 {
    use std::io::Write;
    let mut format = match format_of(pitch, alphabet) { Ok(f) => f, Err(e) => { eprintln!("{e}"); return 1; } };
    let mut calibration = calibration;
    if let Some(code) = code {
        match Config::from_code(code) {
            Ok(c) => { format = c.format(); calibration = false; eprintln!("code {code}: pitch {} alphabet {} fps {}", c.pitch, c.alphabet.name(), c.fps_value()); }
            Err(e) => { eprintln!("code {code:?} not accepted: {e:?}"); return 1; }
        }
    }
    let cfg = FlatSenderConfig { width, height, format, manifest_every_frames: manifest_every, calibration, ..FlatSenderConfig::default() };
    let mut s = match FlatSender::new(file, cfg, compress) { Ok(s) => s, Err(e) => { eprintln!("prepare failed: {e}"); return 1; } };
    eprintln!("streaming {} ({} bytes), {} px {} cells, {} frames per pass{}", s.manifest().filename, s.manifest().file_size, format.pitch, format.alphabet.name(), s.frames_per_pass(), if calibration { format!(", calibration carousel of {} formats", s.grids().len() - 1) } else { String::new() });
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::with_capacity(1 << 20, stdout.lock());
    loop {
        let frame = match s.next_frame() { Ok(f) => f, Err(e) => { eprintln!("{e}"); return 1; } };
        let img = render_to(&frame, width, height);
        let ok = out.write_all(&img.width.to_le_bytes()).and_then(|_| out.write_all(&img.height.to_le_bytes()))
            .and_then(|_| out.write_all(&img.data)).and_then(|_| out.flush());
        if ok.is_err() { return 0; }
    }
}
