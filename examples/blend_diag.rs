//! Experiment (2026-11-02): reading captures that blend two consecutive sender frames (an exposure across the
//! sender's frame change) by cancelling a frame already known. Per capture: the header frame's share of the
//! light per block row (tagged pilots), every cell's sample on its block's black..white axis; frames read whole
//! by one capture are known; then, pass after pass, each unknown frame takes per cell the most confident of: its
//! captures where it dominates (share > 0.55), and its captures where the other frame is known, through a
//! region model (24 x 24 cells, least squares) of the sample from both frames' 3x3 values, their own values'
//! product (the camera's tone curve) and a share ramp; error correction decides.
//!   cargo run --release --example blend_diag -- <dir>   (LB_ONLY=<stem parts>, LB_JOINT_DBG=1, LB_NO_CANCEL=1)
//! Result on test/phone_fhd2px_soft: with frame 168 known (the one capture that reads whole), frame 167 (in two
//! captures: on top of 166, under 168) reads 0 of 111 codewords. Its own weight in the capture under 168 is
//! 0.08-0.49 of the black..white span from top to bottom against a residual of 0.06-0.12, so the middle of the
//! tile (both captures near half and half) stays at 1-2% cells wrong: past the codewords' reach. The reader
//! cannot undo a half-and-half blend from these captures; the exposure has to be short (docs/2026-11-02-soft-frames.md).
use lensbolt_core::flat::layout::PILOT_LOCAL;
use lensbolt_core::flat::{compose, extract_cells, FlatDecoder, FlatGrid, FlatHeader, FlatRead};
use lensbolt_core::image::GrayImage;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

struct Cap { stem: String, seq: u8, grid: Arc<FlatGrid>, read: FlatRead, x: Vec<f32>, w: Vec<f32> }

type Evidence = (Arc<FlatGrid>, FlatHeader, Vec<u8>, Vec<u16>, Vec<f32>);

fn main() {
    let dir = PathBuf::from(std::env::args().nth(1).expect("dir"));
    lensbolt_core::flat::header::ACCEPT_V8_HEADERS.store(true, std::sync::atomic::Ordering::Relaxed);
    let mut stems: Vec<String> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter_map(|n| n.strip_suffix(".y").map(String::from)).collect();
    stems.sort_by_key(|s| s.split('_').nth(1).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0));
    let dec = FlatDecoder::new();
    let mut caps: Vec<Cap> = Vec::new();
    for stem in &stems {
        let f: Vec<&str> = stem.split('_').collect();
        let d: Vec<u32> = f[2].split('x').map(|v| v.parse().unwrap()).collect();
        let y = std::fs::read(dir.join(format!("{stem}.y"))).unwrap();
        let luma = GrayImage::from_plane(d[0], d[1], &y, f[3].parse().unwrap());
        let Ok(frs) = dec.decode_luma(&luma, None) else { println!("{stem}: no luma read"); continue };
        let fr = frs.into_iter().next().unwrap();
        let g = fr.grid.clone();
        let (cols, rows) = (g.cols(), g.rows());
        let (bw, bh) = g.blocks();
        let at = |c: u32, r: u32| -> f32 {
            let b = ((r / 8) * bw + c / 8) as usize;
            let (x, yy) = g.cell_center(c, r);
            let (cx, cy) = fr.homography.apply(x, yy);
            let (dx, dy) = fr.reads.shifts[b];
            luma.sample(cx + dx - 0.5, cy + dy - 0.5)
        };
        // per block: levels from the untagged pilots of its 3x3 blocks; per block row: the header frame's share
        let mut lv = vec![[(0f32, 0u32); 2]; (bw * bh) as usize];
        let mut wrow = vec![(0f32, 0u32); bh as usize];
        let seq = fr.header.frame_seq & 3;
        let mut tagged: Vec<(u32, u32, f32)> = Vec::new();
        for &(c, r) in g.pilot_positions() {
            let k = PILOT_LOCAL.iter().position(|&q| q == (r % 8, c % 8)).unwrap();
            let v = at(c, r);
            if k % 2 == 0 {
                let col = g.pilot_colour_at(c, r, 0) as usize;
                for j in (r / 8).saturating_sub(1)..=(r / 8 + 1).min(bh - 1) { for i in (c / 8).saturating_sub(1)..=(c / 8 + 1).min(bw - 1) {
                    let e = &mut lv[(j * bw + i) as usize][col]; e.0 += v; e.1 += 1;
                } }
            } else if k == 1 { tagged.push((c, r, v)); }
        }
        let lvl = |c: u32, r: u32| { let l = lv[((r / 8) * bw + c / 8) as usize]; (l[0].0 / l[0].1.max(1) as f32, l[1].0 / l[1].1.max(1) as f32) };
        for &(c, r, v) in &tagged {
            let (l0, l1) = lvl(c, r);
            let x = (v - l0) / (l1 - l0);
            let e = &mut wrow[(r / 8) as usize];
            e.0 += if g.pilot_colour_at(c, r, seq) == 1 { x } else { 1.0 - x }; e.1 += 1;
        }
        // smoothed over 5 block rows
        let wr: Vec<f32> = (0..bh as i64).map(|j| { let (mut s, mut n) = (0f32, 0u32); for k in (j - 2).max(0)..=(j + 2).min(bh as i64 - 1) { s += wrow[k as usize].0; n += wrow[k as usize].1; } (s / n.max(1) as f32).clamp(0.0, 1.0) }).collect();
        let mut x = vec![f32::NAN; (cols * rows) as usize];
        let mut w = vec![1f32; (cols * rows) as usize];
        for r in 0..rows { for c in 0..cols {
            let (l0, l1) = lvl(c, r);
            x[(r * cols + c) as usize] = (at(c, r) - l0) / (l1 - l0);
            w[(r * cols + c) as usize] = wr[(r / 8) as usize];
        } }
        let profile: Vec<String> = wr.iter().step_by(6).map(|v| format!("{v:.2}")).collect();
        println!("{stem}: seq {} share {}", fr.header.frame_seq, profile.join(" "));
        caps.push(Cap { stem: stem.clone(), seq: fr.header.frame_seq, grid: g.clone(), read: fr, x, w });
    }
    // known frames: those read whole by one capture
    let mut known: HashMap<u8, Vec<u8>> = HashMap::new();
    for c in &caps {
        let p = c.read.payload();
        if p.codewords_failed + p.codewords_missing == 0 { known.insert(c.seq, compose(c.grid.clone(), c.read.header, &p.bytes).values); println!("known from {}: {}", c.stem, c.seq); }
    }
    // naive decisions against a known frame, per band of 6 block rows, for every capture showing it
    for c in &caps {
        let Some(t) = known.get(&c.seq) else { continue };
        let g = &c.grid;
        let nb = g.blocks().1.div_ceil(6) as usize;
        let (mut bad, mut all, mut badr) = (vec![0usize; nb], vec![0usize; nb], vec![0usize; nb]);
        for i in 0..g.used_data_cells() as u32 {
            let (cc, r) = g.data_position(i);
            let j = (r * g.cols() + cc) as usize;
            let k = (r / 48) as usize;
            all[k] += 1;
            if (c.x[j] > 0.5) as u8 != t[j] { bad[k] += 1; }
            if c.read.reads.values[j] != t[j] { badr[k] += 1; }
        }
        let line: Vec<String> = (0..nb).map(|k| format!("{:.1}/{:.1}", 100.0 * bad[k] as f64 / all[k].max(1) as f64, 100.0 * badr[k] as f64 / all[k].max(1) as f64)).collect();
        println!("{}: seq {} cells wrong % naive/reader by band: {}", c.stem, c.seq, line.join(" "));
    }
    let cancel_on = std::env::var_os("LB_NO_CANCEL").is_none();
    loop {
        let mut progress = false;
        // evidence per unknown frame: per cell the best (value, margin)
        let mut ev: HashMap<u8, Evidence> = HashMap::new();
        for c in &caps {
            let n = c.seq;
            if let Ok(o) = std::env::var("LB_ONLY") { if !o.split(',').any(|q| c.stem.contains(q)) { continue; } }
            // the other frame of a capture: the header's minus one (the bottom), when the share falls below a half
            if c.w.iter().cloned().fold(1f32, f32::min) >= 0.5 || c.grid.format().alphabet.bits() != 1 { continue; }
            let m = n.wrapping_sub(1);
            for (target, cancel, share_t) in [(n, m, true), (m, n, false)] {
                if known.contains_key(&target) { continue; }
                let kv = if cancel_on { known.get(&cancel) } else { None };
                let e = ev.entry(target).or_insert_with(|| { let k = (c.grid.cols() * c.grid.rows()) as usize; (c.grid.clone(), FlatHeader { frame_seq: target, ..c.read.header }, vec![0; k], vec![0; k], vec![0.0; k]) });
                let g = &c.grid;
                // first decisions: the known frame's share removed, or the sample alone
                let n_c = (g.cols() * g.rows()) as usize;
                let mut a0 = vec![0u8; n_c];
                for j in 0..n_c {
                    let (wt, x) = (if share_t { c.w[j] } else { 1.0 - c.w[j] }, c.x[j]);
                    if x.is_nan() { continue; }
                    a0[j] = match kv { Some(k) => (((x - (1.0 - wt) * k[j] as f32) / wt.max(0.05)) > 0.5) as u8, None => (x > 0.5) as u8 };
                }
                if std::env::var_os("LB_JOINT_DBG").is_some() { eprintln!("{} target {target} known {:?}", c.stem, kv.is_some()); }
                let (vals, conf) = joint(g, &c.x, a0, target & 3, kv.map(|k| k.as_slice()));
                if std::env::var_os("LB_JOINT_DBG").is_some() {
                    // per band: mean confidence, and agreement with what the evidence held before
                    let nb = g.blocks().1.div_ceil(6) as usize;
                    let (mut cf, mut agree, mut both, mut cnt) = (vec![0f32; nb], vec![0usize; nb], vec![0usize; nb], vec![0usize; nb]);
                    for i in 0..g.used_data_cells() as u32 {
                        let (cc, r) = g.data_position(i);
                        let j = (r * g.cols() + cc) as usize;
                        let k = (r / 48) as usize;
                        cf[k] += conf[j]; cnt[k] += 1;
                        if e.4[j] > 2.0 && conf[j] > 2.0 { both[k] += 1; if e.2[j] == vals[j] { agree[k] += 1; } }
                    }
                    let line: Vec<String> = (0..nb).map(|k| format!("{:.1}/{:.0}%", cf[k] / cnt[k].max(1) as f32, 100.0 * agree[k] as f32 / both[k].max(1) as f32)).collect();
                    eprintln!("  conf/agreement by band: {}", line.join(" "));
                }
                for i in 0..g.used_data_cells() as u32 {
                    let (cc, r) = g.data_position(i);
                    let j = (r * g.cols() + cc) as usize;
                    let wt = if share_t { c.w[j] } else { 1.0 - c.w[j] };
                    // without the other frame: only where the target dominates (elsewhere the decisions follow the other frame)
                    let conf_j = if kv.is_none() && wt < 0.55 { 0.0 } else { conf[j] };
                    if conf_j > e.4[j] { e.4[j] = conf_j; e.2[j] = vals[j]; e.3[j] = (conf_j * 3000.0).min(65535.0) as u16; }
                }
            }
        }
        let mut keys: Vec<u8> = ev.keys().copied().collect();
        keys.sort();
        let mut line = String::new();
        for s in keys {
            let (g, h, v, m, _) = &ev[&s];
            let p = extract_cells(g, v, m, None, None);
            line += &format!(" {s}:{}", p.codewords_ok);
            if p.codewords_ok == g.codewords() { known.insert(s, compose(g.clone(), *h, &p.bytes).values); progress = true; }
        }
        println!("pass: codewords per unknown frame{line}; known {}", known.len());
        if !progress { break; }
    }
}

/// Per region of 24 x 24 cells, the least-squares model of each cell's sample from the target frame's 3x3
/// values (+-1) and, when known, the other frame's 3x3; the target's data cells decided again from it, three
/// rounds. Returns the decisions and a confidence (distance from the threshold over the region's residual).
fn joint(g: &FlatGrid, x: &[f32], mut a: Vec<u8>, tag: u8, b: Option<&[u8]>) -> (Vec<u8>, Vec<f32>) {
    use lensbolt_core::flat::layout::{CELL_DATA, CELL_PAST, CELL_PILOT};
    let (cols, rows) = (g.cols() as i64, g.rows() as i64);
    let kinds = g.cell_kinds();
    for &(c, r) in g.pilot_positions() { a[(r * g.cols() + c) as usize] = g.pilot_colour_at(c, r, tag); }
    let nf = if b.is_some() { 22 } else { 10 };
    let off: [(i64, i64); 9] = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)];
    let val = |v: &[u8], c: i64, r: i64| -> f64 {
        if c < 0 || r < 0 || c >= cols || r >= rows { return 0.0; }
        let j = (r * cols + c) as usize;
        match kinds[j] { CELL_DATA | CELL_PILOT => if v[j] == 1 { 1.0 } else { -1.0 }, CELL_PAST => -1.0, _ => 0.0 }
    };
    let mut conf = vec![0f32; a.len()];
    const R: i64 = 24;
    for _round in 0..3 {
        let prev = a.clone();
        for ry in (0..rows).step_by(R as usize) { for rx in (0..cols).step_by(R as usize) {
            let feats = |c: i64, r: i64| -> [f64; 22] {
                let mut f = [0f64; 22];
                f[0] = 1.0;
                for (k, &(dx, dy)) in off.iter().enumerate() { f[1 + k] = val(&prev, c + dx, r + dy); if let Some(b) = b { f[10 + k] = val(b, c + dx, r + dy); } }
                // the camera's tone curve: a blend of light, not of pixel values (the two frames' own values interact)
                f[19] = f[1] * f[10];
                let t = ((r - ry) as f64 - 11.5) / 12.0;
                f[20] = f[1] * t; f[21] = f[10] * t;
                f
            };
            let mut ata = [[0f64; 22]; 22];
            let mut aty = [0f64; 22];
            let mut n = 0usize;
            for r in ry..(ry + R).min(rows) { for c in rx..(rx + R).min(cols) {
                let j = (r * cols + c) as usize;
                if x[j].is_nan() || !(kinds[j] == CELL_DATA || kinds[j] == CELL_PILOT) { continue; }
                let f = feats(c, r);
                for p in 0..nf { aty[p] += f[p] * x[j] as f64; for q in 0..nf { ata[p][q] += f[p] * f[q]; } }
                n += 1;
            } }
            if n < 60 { continue; }
            for p in 0..nf { ata[p][p] += 1e-3; }
            let Some(wv) = solve(&ata, &aty, nf) else { continue };
            if wv[1] <= 0.02 { continue; }
            // residual spread
            let mut ss = 0f64;
            for r in ry..(ry + R).min(rows) { for c in rx..(rx + R).min(cols) {
                let j = (r * cols + c) as usize;
                if x[j].is_nan() || !(kinds[j] == CELL_DATA || kinds[j] == CELL_PILOT) { continue; }
                let f = feats(c, r);
                let pr: f64 = (0..nf).map(|p| f[p] * wv[p]).sum();
                ss += (x[j] as f64 - pr).powi(2);
            } }
            let sd = (ss / n as f64).sqrt().max(1e-3);
            if std::env::var_os("LB_JOINT_DBG").is_some() && rx == 240 && _round == 2 { eprintln!("  region row {ry}: own {:.3} other-own {:.3} sd {sd:.3} n {n}", wv[1], if nf > 10 { wv[10] } else { 0.0 }); }
            for r in ry..(ry + R).min(rows) { for c in rx..(rx + R).min(cols) {
                let j = (r * cols + c) as usize;
                if x[j].is_nan() || kinds[j] != CELL_DATA { continue; }
                let f = feats(c, r);
                let rest: f64 = (0..nf).filter(|&p| p != 1 && p != 19 && p != 20).map(|p| f[p] * wv[p]).sum();
                // own +1: rest + w1 + w19 b; own -1: rest - w1 - w19 b
                let own = wv[1] + if nf > 19 { wv[19] * f[10] + wv[20] * ((r - ry) as f64 - 11.5) / 12.0 } else { 0.0 };
                let d = (x[j] as f64 - rest) * own.signum();
                a[j] = (d > 0.0) as u8;
                conf[j] = ((d.abs().min(2.0 * own.abs())) / sd) as f32;
            } }
        } }
    }
    (a, conf)
}

fn solve(a: &[[f64; 22]; 22], y: &[f64; 22], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![vec![0f64; n]; n];
    for i in 0..n { for j in 0..=i {
        let s: f64 = a[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
        if i == j { if s <= 0.0 { return None; } l[i][i] = s.sqrt(); } else { l[i][j] = s / l[j][j]; }
    } }
    let mut z = vec![0f64; n];
    for i in 0..n { z[i] = (y[i] - (0..i).map(|k| l[i][k] * z[k]).sum::<f64>()) / l[i][i]; }
    let mut w = vec![0f64; n];
    for i in (0..n).rev() { w[i] = (z[i] - (i + 1..n).map(|k| l[k][i] * w[k]).sum::<f64>()) / l[i][i]; }
    Some(w)
}
