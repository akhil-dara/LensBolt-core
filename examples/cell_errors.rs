//! Where a phone dump's b/w cells go wrong: every capture that reads whole (so its true cells are known: the
//! payload composed again) counts its wrong cells by direction (symbol 0 read as 1 and the reverse), by margin
//! (how many fall under the erasure margin, against the right cells) and by how many of the cell's four
//! neighbours differ from it (an isolated cell: all four).
//!   cargo run --release --example cell_errors -- <dir> [v8]
use lensbolt_core::flat::extract::ERASURE_MARGIN;
use lensbolt_core::flat::{compose, FlatDecoder, FlatHint};
use lensbolt_core::image::GrayImage;
use std::path::PathBuf;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let dir = PathBuf::from(&a[0]);
    if a.iter().any(|x| x == "v8") { lensbolt_core::flat::header::ACCEPT_V8_HEADERS.store(true, std::sync::atomic::Ordering::Relaxed); }
    let mut stems: Vec<String> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter_map(|n| n.strip_suffix(".y").map(String::from)).collect();
    stems.sort_by_key(|s| s.split('_').nth(1).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0));
    let dec = FlatDecoder::new();
    let mut hint: Option<FlatHint> = None;
    // [truth][read] counts; margins under ERASURE_MARGIN for wrong and right cells; by differing neighbours 0..4
    let (mut conf, mut weak_wrong, mut weak_right) = ([[0u64; 2]; 2], 0u64, 0u64);
    let (mut by_nb, mut by_nb_wrong) = ([0u64; 5], [0u64; 5]);
    let mut whole = 0;
    // margin histograms (buckets of powers of two) by [truth][read]
    let mut hist = [[[0u64; 20]; 2]; 2];
    for stem in &stems {
        let f: Vec<&str> = stem.split('_').collect();
        let d: Vec<usize> = f[2].split('x').map(|v| v.parse().unwrap()).collect();
        let raw = std::fs::read(dir.join(format!("{stem}.y"))).unwrap();
        let g = GrayImage::from_plane(d[0] as u32, d[1] as u32, &raw, f[3].parse().unwrap());
        let Ok(reads) = dec.decode_luma(&g, hint.as_ref()) else { continue };
        hint = Some(reads[0].hint());
        for fr in &reads {
            if !fr.header.format.alphabet.is_luma() || fr.grid.format().pitch > 4 { continue; }
            let p = fr.payload();
            if p.codewords_failed + p.codewords_missing > 0 { continue; }
            whole += 1;
            let t = compose(fr.grid.clone(), fr.header, &p.bytes).values;
            let gr = &fr.grid;
            let cols = gr.cols();
            let tv = |c: u32, r: u32| t[(r * cols + c) as usize];
            for i in 0..gr.used_data_cells() as u32 {
                let (c, r) = gr.data_position(i);
                let (v, m) = fr.reads.at(c, r);
                let truth = tv(c, r) as usize & 1;
                conf[truth][v as usize & 1] += 1;
                hist[truth][v as usize & 1][(32 - (m as u32).leading_zeros()) as usize] += 1;
                let wrong = (v as usize & 1) != truth;
                if m < ERASURE_MARGIN { if wrong { weak_wrong += 1 } else { weak_right += 1 } }
                let mut nb = 0;
                for (dc, dr) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let (cc, rr) = (c as i32 + dc, r as i32 + dr);
                    if cc >= 0 && rr >= 0 && (cc as u32) < cols && (rr as u32) < gr.rows() && (tv(cc as u32, rr as u32) & 1) as usize != truth { nb += 1; }
                }
                by_nb[nb] += 1;
                if wrong { by_nb_wrong[nb] += 1; }
            }
        }
    }
    let wrong = conf[0][1] + conf[1][0];
    let all: u64 = conf.iter().flatten().sum();
    println!("{whole} whole reads, {all} cells, {wrong} wrong ({:.3}%)", 100.0 * wrong as f64 / all.max(1) as f64);
    println!("symbol 0 read as 1: {} of {} ({:.3}%); symbol 1 read as 0: {} of {} ({:.3}%)",
        conf[0][1], conf[0][0] + conf[0][1], 100.0 * conf[0][1] as f64 / (conf[0][0] + conf[0][1]).max(1) as f64,
        conf[1][0], conf[1][0] + conf[1][1], 100.0 * conf[1][0] as f64 / (conf[1][0] + conf[1][1]).max(1) as f64);
    println!("under the erasure margin: {weak_wrong} of {wrong} wrong cells ({:.1}%), {weak_right} right cells ({:.2}% of them)",
        100.0 * weak_wrong as f64 / wrong.max(1) as f64, 100.0 * weak_right as f64 / (all - wrong).max(1) as f64);
    println!("margin buckets (upper bound 2^k): k: right-as-0 / wrong-as-0 (truth 1) | right-as-1 / wrong-as-1 (truth 0)");
    for k in 0..20 { let (a, b, c, e) = (hist[0][0][k], hist[1][0][k], hist[1][1][k], hist[0][1][k]); if a + b + c + e > 0 { println!("  2^{k:<2}: {a:>9} / {b:>5} | {c:>9} / {e:>5}"); } }
    for k in 0..5 { println!("  {k} of 4 neighbours differ: {:>9} cells, {:>6} wrong ({:.3}%), {:.1}% of all wrong", by_nb[k], by_nb_wrong[k], 100.0 * by_nb_wrong[k] as f64 / by_nb[k].max(1) as f64, 100.0 * by_nb_wrong[k] as f64 / wrong.max(1) as f64); }
}
