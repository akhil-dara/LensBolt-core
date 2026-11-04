//! Diagnostic for a phone dump folder (f_<ms>_<W>x<H>_<yrow>_<uvrow>_<pixel>.{y,u,v}): every capture read cold
//! and tracked through the phone's own path (the Y plane into `decode_luma_first`, the hint from the latest
//! capture that decoded), then into one `Stitcher` as on the phone. Per capture: codewords cold / tracked, the
//! blend of two sender frames (`mix`) and the border's edge width (`edge`, camera px), the frames the capture
//! shows and what the stitcher completed.
//!   cargo run --release --example dump_diag -- <dir> [--rgb] [--only <stem part>] [--blend] [--motion] [--truth <stem>]
//! - `--rgb`: the RGB path (`decode_tracked`, the reference reader) instead.
//! - `--blend`: the header frame's share of the light per band of 6 block rows (from the tagged pilots of kind
//!   1, whose colour flips between consecutive frames), top to bottom.
//! - `--motion`: how far the tile moved since the previous decoded capture (the homography's largest corner
//!   move, camera px) and the residual shear the block shifts show over the tile's height beyond the
//!   homography (a rolling shutter during motion: x shift at the bottom minus at the top, camera px).
//! - `--truth <stem>`: that capture reads whole; every capture showing its frame prints its cells wrong per
//!   band of 6 block rows against it.
use lensbolt_core::detect::Homography;
use lensbolt_core::flat::live::decode_luma_first;
use lensbolt_core::flat::layout::PILOT_LOCAL;
use lensbolt_core::flat::{compose, DecodeError, FlatDecoder, FlatHint, FlatRead, Stitcher};
use lensbolt_core::image::{GrayImage, RgbImage};
use lensbolt_core::packet::PACKET_BYTES;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = PathBuf::from(&args[0]);
    let flag = |f: &str| args.iter().any(|a| a == f);
    let value = |f: &str| args.iter().position(|a| a == f).map(|i| args[i + 1].clone());
    let (rgb_path, blend, motion) = (flag("--rgb"), flag("--blend"), flag("--motion"));
    lensbolt_core::flat::header::ACCEPT_V8_HEADERS.store(true, std::sync::atomic::Ordering::Relaxed);
    let mut stems: Vec<String> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter_map(|n| n.strip_suffix(".y").map(String::from)).collect();
    stems.sort_by_key(|s| s.split('_').nth(1).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0));
    if let Some(o) = value("--only") { stems.retain(|s| s.contains(o.as_str())); }
    let load = |stem: &str| -> (Vec<u8>, Vec<u8>, Vec<u8>, [usize; 5]) {
        let f: Vec<&str> = stem.split('_').collect();
        let d: Vec<usize> = f[2].split('x').map(|v| v.parse().unwrap()).collect();
        let rd = |e: &str| std::fs::read(dir.join(format!("{stem}.{e}"))).unwrap();
        (rd("y"), rd("u"), rd("v"), [d[0], d[1], f[3].parse().unwrap(), f[4].parse().unwrap(), f[5].parse().unwrap()])
    };
    let dec = FlatDecoder::new();
    let truth: Option<(u8, Vec<u8>)> = value("--truth").map(|stem| {
        let (y, _, _, d) = load(&stem);
        let fr = dec.decode_luma(&GrayImage::from_plane(d[0] as u32, d[1] as u32, &y, d[2]), None).unwrap().remove(0);
        let p = fr.payload();
        assert_eq!(p.codewords_failed + p.codewords_missing, 0, "the truth capture must read whole");
        (fr.header.frame_seq, compose(fr.grid.clone(), fr.header, &p.bytes).values)
    });
    let mut hint: Option<FlatHint> = None;
    let mut prev_h: Option<Homography> = None;
    let mut t0: Option<u64> = None;
    let mut stitcher = Stitcher::new();
    let (mut sum_c, mut sum_t, mut all, mut packets) = (0usize, 0usize, 0usize, 0usize);
    for stem in &stems {
        let ms: u64 = stem.split('_').nth(1).unwrap().parse().unwrap();
        let (y, u, v, d) = load(stem);
        let rgb = || RgbImage::from_yuv420(d[0] as u32, d[1] as u32, &y, d[2], &u, &v, d[3], d[4]);
        let luma = GrayImage::from_plane(d[0] as u32, d[1] as u32, &y, d[2]);
        let dt = ms - *t0.get_or_insert(ms);
        let run = |h: Option<&FlatHint>| -> Result<Vec<FlatRead>, DecodeError> {
            if rgb_path { dec.decode_tracked(&rgb(), h).map(|r| vec![r]) } else { decode_luma_first(&dec, &luma, rgb, h).map(|(v, _)| v) }
        };
        let t = std::time::Instant::now();
        let cold = run(None);
        let tc = t.elapsed().as_secs_f64() * 1e3;
        let t = std::time::Instant::now();
        let tracked = run(hint.as_ref());
        let tt = t.elapsed().as_secs_f64() * 1e3;
        let show = |r: &Result<Vec<FlatRead>, DecodeError>, ms: f64, sum: &mut usize| match r {
            Ok(frs) => {
                let (fr, p) = (&frs[0], frs[0].payload());
                *sum += p.codewords_ok;
                format!("cw {:>3}/{:<3} mix {:.2} edge {:.2} {ms:>5.1} ms", p.codewords_ok, fr.grid.codewords(), fr.reads.mix, fr.reads.edge_px)
            }
            Err(e) => format!("{e:?} {ms:.1} ms"),
        };
        all += tracked.as_ref().or(cold.as_ref()).map(|f| f[0].grid.codewords()).unwrap_or(0);
        let (sc, st) = (show(&cold, tc, &mut sum_c), show(&tracked, tt, &mut sum_t));
        let mut stitched = String::new();
        if let Ok(frs) = &tracked {
            let out = stitcher.add_capture(frs);
            let pk: usize = out.iter().map(|(_, p)| p.valid.iter().filter(|&&v| v).count() / PACKET_BYTES).sum();
            packets += pk;
            let seqs: Vec<u8> = frs.iter().map(|f| f.header.frame_seq).collect();
            stitched = format!(" | frames {seqs:?} stitched {:?} packets +{pk}", stitcher.last_codewords());
        }
        println!("{stem} +{dt:>5} ms | cold {sc} | tracked {st}{stitched}");
        let Ok(frs) = cold.as_ref() else { hint = tracked.ok().map(|f| f[0].hint()).or(hint); continue };
        let fr = &frs[0];
        if blend && fr.header.format.alphabet.size() == 2 { println!("    share of frame {} by band of 6 block rows: {}", fr.header.frame_seq, shares(fr, &luma).iter().map(|w| format!("{w:.2}")).collect::<Vec<_>>().join(" ")); }
        if motion {
            let (w, h) = (fr.grid.width_px() as f32, fr.grid.height_px() as f32);
            let drift = prev_h.map(|p| [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)].iter().map(|&(x, yy)| { let (a, b) = (fr.homography.apply(x, yy), p.apply(x, yy)); (a.0 - b.0).hypot(a.1 - b.1) }).fold(0f32, f32::max));
            let (bw, bh) = fr.grid.blocks();
            // least squares of each block's x and y shift against its block row
            let (mut n, mut sr, mut srr, mut sx, mut srx, mut sy, mut sry) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
            for (b, &(dx, dy)) in fr.reads.shifts.iter().enumerate() {
                let r = (b as u32 / bw) as f64;
                n += 1.0; sr += r; srr += r * r; sx += dx as f64; srx += r * dx as f64; sy += dy as f64; sry += r * dy as f64;
            }
            let den = n * srr - sr * sr;
            let (gx, gy) = if den > 0.0 { ((n * srx - sr * sx) / den * bh as f64, (n * sry - sr * sy) / den * bh as f64) } else { (0.0, 0.0) };
            println!("    motion: drift {} shear-x {gx:+.2} shear-y {gy:+.2}", drift.map_or("-".into(), |d| format!("{d:.2}")));
            prev_h = Some(fr.homography);
        }
        if let Some((seq, t)) = &truth {
            for fr in frs.iter().filter(|f| f.header.frame_seq == *seq) {
                let g = &fr.grid;
                let nb = g.blocks().1.div_ceil(6) as usize;
                let (mut bad, mut n) = (vec![0usize; nb], vec![0usize; nb]);
                for i in 0..g.used_data_cells() as u32 {
                    let (c, r) = g.data_position(i);
                    let k = (r / 48) as usize;
                    n[k] += 1;
                    if fr.reads.at(c, r).0 != t[(r * g.cols() + c) as usize] { bad[k] += 1; }
                }
                println!("    frame {seq} cells wrong % by band of 6 block rows: {}", (0..nb).map(|k| format!("{:.1}", 100.0 * bad[k] as f64 / n[k].max(1) as f64)).collect::<Vec<_>>().join(" "));
            }
        }
        hint = tracked.ok().map(|f| f[0].hint()).or(hint);
    }
    println!("total: codewords cold {sum_c}/{all}, tracked {sum_t}/{all}; packets the stitcher completed {packets}");
}

/// The header frame's share of the light per band of 6 block rows: kind 1 pilots flip colour between
/// consecutive frames, so where their sample (at the read's alignment) sits between the band's untagged pilot
/// levels is that share.
fn shares(fr: &FlatRead, luma: &GrayImage) -> Vec<f32> {
    let g = &fr.grid;
    let (bw, bh) = g.blocks();
    let seq = fr.header.frame_seq & 3;
    let at = |c: u32, r: u32| -> f32 {
        let (x, yy) = g.cell_center(c, r);
        let (cx, cy) = fr.homography.apply(x, yy);
        let (dx, dy) = fr.reads.shifts[((r / 8) * bw + c / 8) as usize];
        luma.sample(cx + dx - 0.5, cy + dy - 0.5)
    };
    (0..bh).step_by(6).map(|b0| {
        let in_band = |r: u32| r / 8 >= b0 && r / 8 < b0 + 6;
        let kind = |c: u32, r: u32| PILOT_LOCAL.iter().position(|&q| q == (r % 8, c % 8)).unwrap();
        let mut lv = [(0f32, 0u32); 2];
        for &(c, r) in g.pilot_positions().iter().filter(|p| in_band(p.1) && kind(p.0, p.1) % 2 == 0) {
            let e = &mut lv[g.pilot_colour_at(c, r, 0) as usize]; e.0 += at(c, r); e.1 += 1;
        }
        let (l0, l1) = (lv[0].0 / lv[0].1.max(1) as f32, lv[1].0 / lv[1].1.max(1) as f32);
        let ws: Vec<f32> = g.pilot_positions().iter().filter(|p| in_band(p.1) && kind(p.0, p.1) == 1).map(|&(c, r)| {
            let x = (at(c, r) - l0) / (l1 - l0);
            if g.pilot_colour_at(c, r, seq) == 1 { x } else { 1.0 - x }
        }).collect();
        ws.iter().sum::<f32>() / ws.len().max(1) as f32
    }).collect()
}
