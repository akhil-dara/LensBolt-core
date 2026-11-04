//! Cell reads -> payload bytes (format v8): the frame's data cells form one byte stream; un-whiten, gather
//! codeword c from stream bytes c, c+N, c+2N, ...; RS with erasures for missing cells and weak margins.

use super::compose::cells_to_stripe;
use super::layout::FlatGrid;
use super::read::FlatReads;
use crate::interleave::{slot_of, whiten};
use crate::rs::RsCodec;

/// Decoded frame payload: `grid.payload_bytes()` bytes, valid where their codeword decoded. Packets sit back
/// to back (`crate::packet::unpack_frame`). `codewords_missing`: codewords not tried because too many of their
/// cells are missing (a partial frame in the stitcher); `coverage`: fraction of used data cells present.
pub struct FramePayload { pub bytes: Vec<u8>, pub valid: Vec<bool>, pub codewords_ok: usize, pub codewords_failed: usize, pub codewords_missing: usize, pub coverage: f32 }

/// A cell whose runner-up colour is this close (see ColourModel::classify) marks its bytes as erasure
/// candidates.
pub const ERASURE_MARGIN: u16 = 600;
/// Diagnostic override of `ERASURE_MARGIN` (0 = use the constant); set by `lensbolt diag --erasure-margin`.
pub static ERASURE_MARGIN_OVERRIDE: std::sync::atomic::AtomicU16 = std::sync::atomic::AtomicU16::new(0);
pub fn erasure_margin() -> u16 { let o = ERASURE_MARGIN_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed); if o == 0 { ERASURE_MARGIN } else { o } }
/// Parity a codeword keeps beyond its missing-cell erasures (4 correctable errors): with every parity byte
/// spent on erasures RS would accept anything.
pub const SPARE_PARITY: usize = 8;

/// A b/w cell read white counts this much weaker when its byte's erasure rank is set. On screens black cells read
/// white far more often than the reverse, and with margins that look safe: on real phone captures (a laptop's 2
/// px tile, a phone sender's 3 px) 7-11x as many black cells read white as white read black, and of the white reads
/// with margins up to 4096, 8192 and 16384, 88%, 76% and 43% were wrong, against 6%, 3% and 0.7% of black reads;
/// the erasure margin flagged only 5-9% of the wrong cells (screens glow into their dark cells; ink, the
/// reverse). Ranked this way, real 2 px captures degraded to their floor read more codewords: 1.2 px 55.7 ->
/// 64.1% cold, 82.9 -> 88.4% tracked; 1.4 px out of focus 84.8 -> 89.4% cold; no frame read wrong. Flat from 8
/// to 128; the erasure count and the spare parity are unchanged.
pub const WHITE_DISTRUST: u16 = 1 << WHITE_DISTRUST_SHIFT;
const WHITE_DISTRUST_SHIFT: u32 = 5;

pub fn extract_payload(reads: &FlatReads, grid: &FlatGrid) -> FramePayload {
    // a read the blend pre-check stopped before its cells: every cell missing
    if reads.values.is_empty() {
        let n = (grid.cols() * grid.rows()) as usize;
        return extract_cells(grid, &vec![0; n], &vec![0; n], Some(&vec![false; n]), None);
    }
    // the cells of flat blocks (covered, blown out) are unreadable: missing, so their bytes are erasures (one
    // parity byte each) instead of confident errors (two)
    if reads.block_tag.contains(&super::read::TAG_FLAT) {
        let (bw, _) = grid.blocks();
        let present: Vec<bool> = (0..grid.rows()).flat_map(|r| (0..grid.cols()).map(move |c| (c, r)))
            .map(|(c, r)| reads.block_tag.get(((r / super::layout::BLOCK) * bw + c / super::layout::BLOCK) as usize) != Some(&super::read::TAG_FLAT)).collect();
        return extract_cells(grid, &reads.values, &reads.margins, Some(&present), None);
    }
    extract_cells(grid, &reads.values, &reads.margins, None, None)
}

/// The frame payload from per-cell values and margins (row-major, cols x rows). `present` marks the cells
/// seen at all (None = every cell); missing cells erase their bytes. Codewords with `done[c]` set cost no RS
/// work and stay invalid.
pub fn extract_cells(grid: &FlatGrid, values: &[u8], margins: &[u16], present: Option<&[bool]>, done: Option<&[bool]>) -> FramePayload {
    let fmt = grid.format();
    // the profile timer exists only when asked for: Instant::now panics in browser WebAssembly
    let t0 = std::env::var_os("LB_PROFILE").map(|_| std::time::Instant::now());
    let mark = |what: &str| { if let Some(t0) = t0 { eprintln!("  [extract] {:>7.1} ms  {}", t0.elapsed().as_secs_f64() * 1000.0, what); } };
    let bits = fmt.alphabet.bits() as usize;
    let white_shift = if bits == 1 { WHITE_DISTRUST_SHIFT } else { 0 };
    let (parity, data_len, n) = (fmt.parity(), fmt.data_len(), grid.codewords());
    let used = grid.used_data_cells();
    let stream_len = n * 255;
    let cols = grid.cols();
    let mut cells = Vec::with_capacity(used);
    let mut seen = 0usize;
    // per stream byte: the weakest margin of the cells its bits came from (0 if one of them is missing)
    let mut weak = vec![u16::MAX; stream_len];
    let mut missing = vec![false; stream_len];
    for i in 0..used {
        let (c, r) = grid.data_position(i as u32);
        let idx = (r * cols + c) as usize;
        cells.push(values[idx]);
        let here = present.map_or(true, |p| p[idx]);
        if here { seen += 1; }
        let (b0, b1) = ((i * bits) / 8, ((i + 1) * bits - 1) / 8);
        // b/w: a cell read white is far less sure than its margin says (WHITE_DISTRUST); branchless (black and
        // white cells come at random: a branch here cost 5 ms of an 11 ms 1 px frame)
        let m = margins[idx] >> (white_shift * (values[idx] as u32 & 1));
        for b in b0..=b1.min(stream_len - 1) {
            if here { weak[b] = weak[b].min(m); } else { missing[b] = true; }
        }
    }
    mark("cells gathered");
    let mut stream = cells_to_stripe(&cells, bits as u32);
    stream.truncate(stream_len);
    whiten(&mut stream);
    mark("stream packed");
    let rs = RsCodec::new(parity);
    let em = erasure_margin();
    // None: not tried (done or too much missing); Some(None): RS failed; Some(Some(cw)): decoded
    let decoded = crate::par::par_map(n, |c| -> Option<Option<Vec<u8>>> {
        if done.is_some_and(|d| d[c]) { return None; }
        let mut cw = vec![0u8; 255];
        let mut gone: Vec<usize> = Vec::new();
        let mut suspects: Vec<(u16, usize)> = Vec::new();
        for (j, v) in cw.iter_mut().enumerate() {
            let b = slot_of(c, j, n);
            *v = stream[b];
            if missing[b] { gone.push(j); } else if weak[b] < em { suspects.push((weak[b], j)); }
        }
        if gone.len() + SPARE_PARITY > parity { return None; }
        suspects.sort_unstable();
        let extra = (parity / 2).min(parity - SPARE_PARITY - gone.len());
        let mut erasures = gone.clone();
        erasures.extend(suspects.iter().take(extra).map(|&(_, j)| j));
        let original = cw.clone();
        let mut res = rs.decode(&mut cw, &erasures);
        if res.is_err() && erasures.len() > gone.len() { cw.copy_from_slice(&original); res = rs.decode(&mut cw, &gone); }
        Some(if res.is_ok() { Some(cw) } else { None })
    });
    mark("codewords decoded");
    let mut bytes = vec![0u8; n * data_len];
    let mut valid = vec![false; n * data_len];
    let (mut ok, mut failed, mut skipped) = (0, 0, 0);
    for (c, d) in decoded.iter().enumerate() {
        match d {
            Some(Some(cw)) => {
                ok += 1;
                bytes[c * data_len..(c + 1) * data_len].copy_from_slice(&cw[..data_len]);
                for v in &mut valid[c * data_len..(c + 1) * data_len] { *v = true; }
            }
            Some(None) => failed += 1,
            None => if !done.is_some_and(|d| d[c]) { skipped += 1; },
        }
    }
    FramePayload { bytes, valid, codewords_ok: ok, codewords_failed: failed, codewords_missing: skipped, coverage: seen as f32 / used.max(1) as f32 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::Homography;
    use crate::flat::{compose, read_cells, render, Alphabet, CellFormat, FlatGrid, FlatHeader, ReadParams};
    use crate::header::FrameType;
    use crate::image::RgbImage;
    use crate::testutil::bend;
    use std::sync::Arc;

    fn roundtrip(fmt: CellFormat, payload_of: impl Fn(usize) -> u8, damage: impl Fn(&mut RgbImage)) -> (Vec<u8>, FramePayload) {
        let g = Arc::new(FlatGrid::try_new(1024, 1024, fmt).unwrap());
        let h = FlatHeader { format: fmt, cols: g.cols() as u16, rows: g.rows() as u16, frame_type: FrameType::Data, transfer_id: 1, segment: 1, block_id: 0, frame_seq: 0 };
        let payload: Vec<u8> = (0..g.payload_bytes()).map(payload_of).collect();
        let mut im = render(&compose(g.clone(), h, &payload));
        damage(&mut im);
        let reads = read_cells(&im, &g, &Homography::identity(), &ReadParams::default());
        (payload, extract_payload(&reads, &g))
    }
    fn pseudo(i: usize) -> u8 { ((i as u64 * 2654435761) >> 7) as u8 }

    #[test]
    fn clean_frames_extract_exactly() {
        for fmt in [CellFormat { pitch: 4, alphabet: Alphabet::C8 }, CellFormat { pitch: 3, alphabet: Alphabet::C16 }, CellFormat { pitch: 8, alphabet: Alphabet::Bw }, CellFormat { pitch: 12, alphabet: Alphabet::C4 }] {
            let (payload, fp) = roundtrip(fmt, pseudo, |_| {});
            assert_eq!(fp.codewords_failed, 0, "{fmt:?}");
            assert_eq!(fp.bytes, payload, "{fmt:?}");
            assert!(fp.valid.iter().all(|&v| v));
            assert_eq!((fp.codewords_missing, fp.coverage), (0, 1.0));
        }
    }
    #[test]
    fn a_finger_sized_blob_is_corrected() {
        let (payload, fp) = roundtrip(CellFormat { pitch: 4, alphabet: Alphabet::C8 }, pseudo, |im| im.fill_rect(900, 500, 60, 60, [0, 0, 0]));
        assert_eq!(fp.codewords_failed, 0, "ok={} failed={}", fp.codewords_ok, fp.codewords_failed);
        assert_eq!(fp.bytes, payload);
    }
    #[test]
    fn a_band_of_wrong_cells_costs_no_codeword() {
        // v7 lost every stripe a burst crossed; a v8 codeword samples the whole frame, so a 12% band of garbage
        // is ~12% byte errors spread over every codeword, inside RS's reach at 1 px (parity 64)
        let (payload, fp) = roundtrip(CellFormat { pitch: 1, alphabet: Alphabet::Bw }, pseudo, |im| { let h = im.height; im.fill_rect(0, h * 40 / 100, im.width, h * 6 / 100, [255, 255, 255]); });
        assert_eq!(fp.codewords_failed, 0, "ok={} failed={}", fp.codewords_ok, fp.codewords_failed);
        assert_eq!(fp.bytes, payload);
    }
    // black cells read white with margins that pass as sure (the screen-capture error, WHITE_DISTRUST): about
    // 12 wrong bytes a codeword, about what RS mends alone, decode once the white reads rank as erasures (with
    // the white reads ranked like the black: 54 of 111 codewords)
    #[test]
    fn black_cells_read_white_rank_as_erasures() {
        let fmt = CellFormat { pitch: 2, alphabet: Alphabet::Bw };
        let g = Arc::new(FlatGrid::try_new(1024, 1024, fmt).unwrap());
        let h = FlatHeader { format: fmt, cols: g.cols() as u16, rows: g.rows() as u16, frame_type: FrameType::Data, transfer_id: 1, segment: 1, block_id: 0, frame_seq: 0 };
        let payload: Vec<u8> = (0..g.payload_bytes()).map(|i| (i * 31 + 7) as u8).collect();
        let mut values = compose(g.clone(), h, &payload).values;
        let mut margins = vec![60000u16; values.len()];
        let mut st = 0x2545_F491u32;
        for i in 0..g.used_data_cells() as u32 {
            let (c, r) = g.data_position(i);
            let idx = (r * g.cols() + c) as usize;
            st ^= st << 13; st ^= st >> 17; st ^= st << 5;
            // 1.2% of the black data cells read white with a margin of 8000 (sure-looking, under the white reads' usual)
            if values[idx] == 0 && st % 1000 < 12 { values[idx] = 1; margins[idx] = 8000; }
        }
        let p = extract_cells(&g, &values, &margins, None, None);
        assert!(p.codewords_ok * 100 >= g.codewords() * 90, "{} of {} codewords", p.codewords_ok, g.codewords());
        assert!(p.bytes.iter().zip(&payload).zip(&p.valid).all(|((a, b), &v)| !v || a == b), "a byte marked valid is wrong");
    }

    #[test]
    fn missing_cells_are_erasures() {
        // half the frame unseen: every codeword has ~half its bytes missing, beyond parity, so nothing is
        // tried; a quarter missing at parity 64 (1 px) is ~25% bytes: still beyond; 10% missing decodes
        let fmt = CellFormat { pitch: 1, alphabet: Alphabet::Bw };
        let g = Arc::new(FlatGrid::try_new(1024, 1024, fmt).unwrap());
        let h = FlatHeader { format: fmt, cols: g.cols() as u16, rows: g.rows() as u16, frame_type: FrameType::Data, transfer_id: 1, segment: 1, block_id: 0, frame_seq: 0 };
        let payload: Vec<u8> = (0..g.payload_bytes()).map(pseudo).collect();
        let f = compose(g.clone(), h, &payload);
        let margins = vec![u16::MAX; f.values.len()];
        let mut values = f.values.clone();
        // the unseen cells hold garbage: they must not matter
        for (i, v) in values.iter_mut().enumerate() { if i % 3 == 0 { *v ^= 1; } }
        let rows = g.rows() as usize; let cols = g.cols() as usize;
        let band = |lo: usize, hi: usize| -> Vec<bool> { (0..rows * cols).map(|i| { let r = i / cols; !(r >= lo && r < hi) }).collect() };
        let half = band(0, rows / 2);
        let fp = extract_cells(&g, &values, &margins, Some(&half), None);
        assert_eq!(fp.codewords_ok + fp.codewords_failed, 0, "no codeword is tried with half its bytes gone");
        assert_eq!(fp.codewords_missing, g.codewords());
        assert!((fp.coverage - 0.5).abs() < 0.05, "coverage {}", fp.coverage);
        // 10% missing, and the seen cells are clean: every codeword decodes from the erasures alone
        let clean = f.values.clone();
        let tenth = band(rows / 3, rows / 3 + rows / 10);
        let mut dirty = clean.clone();
        for (i, v) in dirty.iter_mut().enumerate() { if !tenth[i] { *v ^= 1; } }
        let fp = extract_cells(&g, &dirty, &margins, Some(&tenth), None);
        assert_eq!(fp.codewords_ok, g.codewords(), "missing {} failed {}", fp.codewords_missing, fp.codewords_failed);
        assert_eq!(fp.bytes, payload);
        // done codewords are skipped and stay invalid
        let mut done = vec![false; g.codewords()]; done[3] = true;
        let fp = extract_cells(&g, &clean, &margins, None, Some(&done));
        assert_eq!(fp.codewords_ok, g.codewords() - 1);
        assert!(!fp.valid[3 * 191] && fp.valid[4 * 191]);
    }
    #[test]
    fn bent_and_cast_frame_extracts() {
        let (payload, fp) = roundtrip(CellFormat { pitch: 4, alphabet: Alphabet::C8 }, pseudo, |im| {
            let b = bend(im, 2.5, 5);
            *im = b;
            for p in im.data.chunks_mut(3) { let gains = [0.8f32, 0.93, 1.0]; for k in 0..3 { p[k] = (p[k] as f32 * gains[k] + 20.0).min(255.0) as u8; } }
        });
        assert_eq!(fp.codewords_failed, 0, "ok={} failed={}", fp.codewords_ok, fp.codewords_failed);
        assert_eq!(fp.bytes, payload);
    }
    #[test]
    fn fit_is_well_conditioned_on_whitened_zero_payload() {
        let (payload, fp) = roundtrip(CellFormat { pitch: 4, alphabet: Alphabet::C16 }, |_| 0, |_| {});
        assert_eq!(fp.codewords_failed, 0);
        assert_eq!(fp.bytes, payload);
    }
}
