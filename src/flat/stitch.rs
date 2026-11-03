//! Mixed-capture stitching: a rolling-shutter capture that straddles a display refresh holds the top of
//! frame k and the bottom of frame k+1. Each block is tagged with its frame (see `FlatReads::block_tag`);
//! the stitcher keeps one cell buffer per recent displayed frame and, whenever a capture adds to it, decodes
//! the codewords not yet decoded with the still-missing cells as erasures (v8: every codeword spans the whole
//! frame, so a half-seen frame decodes nothing and the next capture completes it). Each packet is reported
//! once, when its last codeword decodes.

use super::decoder::FlatRead;
use super::extract::{extract_cells, FramePayload};
use super::header::FlatHeader;
use super::layout::{FlatGrid, BLOCK};
use super::read::FlatReads;
use crate::packet::PACKET_BYTES;
use std::collections::HashMap;
use std::sync::Arc;

struct Partial {
    grid: Arc<FlatGrid>, header: FlatHeader, values: Vec<u8>, margins: Vec<u16>, filled: Vec<bool>,
    /// decoded payload so far, codewords decoded, packets reported; `dirty`: cells changed since the last decode
    bytes: Vec<u8>, cw_done: Vec<bool>, packet_done: Vec<bool>, dirty: bool, last_seen: u64,
}

/// Frame buffer key: transfer, frame sequence and the grid it was read with (pitch, alphabet, cols, rows).
/// A capture read with frame k's grid also stores the blocks it tagged as k+1 under k's grid: right when
/// k+1 has the same format, a harmless dead entry when it does not (its real read arrives under its own key).
type Key = (u16, u8, u8, u8, u32, u32);

/// Frames keyed by `Key`; the oldest are dropped past `keep` entries. `last`: codewords decoded and all of the
/// frame the latest read showed (its own frame number), whether or not anything new decoded.
pub struct Stitcher { frames: HashMap<Key, Partial>, keep: usize, captures: u64, last: (usize, usize) }

impl Default for Stitcher { fn default() -> Self { Stitcher::new() } }

impl Stitcher {
    pub fn new() -> Stitcher { Stitcher { frames: HashMap::new(), keep: 12, captures: 0, last: (0, 0) } }
    /// (decoded, all) codewords of the frame the latest read showed, so far over all its captures: how well
    /// the camera reads it now, also when the capture added nothing (the payloads report only frames with new
    /// packets).
    pub fn last_codewords(&self) -> (usize, usize) { self.last }
    /// Merge a capture and return, for every frame it touched, the payload with `valid` set only on the
    /// packets that completed now (never a packet reported before).
    pub fn add(&mut self, read: &FlatRead) -> Vec<(FlatHeader, FramePayload)> { self.add_masked(read, None) }
    /// A read the blend pre-check stopped before its cells (`FlatReads::values` empty): nothing to stitch, and the
    /// frame counts as read with none of its codewords.
    fn skipped(&mut self, read: &FlatRead) -> bool {
        if !read.reads.values.is_empty() { return false; }
        self.last = (0, read.grid.codewords());
        true
    }

    /// All reads of one capture at once. Two reads of different formats mean the capture straddles a
    /// refresh between a frame of each: the read whose tag map shows the cut tells where the other
    /// frame's cells are, and the other read keeps only the blocks on that side (its own pilots cannot
    /// judge cells of a foreign format).
    pub fn add_capture(&mut self, reads: &[FlatRead]) -> Vec<(FlatHeader, FramePayload)> {
        let mut out = Vec::new();
        if reads.iter().any(|r| r.reads.values.is_empty()) { for r in reads { self.skipped(r); } return out; }
        if reads.len() < 2 { for r in reads { out.extend(self.add(r)); } return out; }
        for (i, r) in reads.iter().enumerate() {
            // the first other read with a clean cut defines the region that is NOT its own frame
            let mask = reads.iter().enumerate().filter(|&(j, _)| j != i).find_map(|(_, o)| cut_of(o).map(|c| (c, o.header.frame_seq)));
            match mask {
                Some(((vertical, px, own_first), _)) => {
                    // keep the blocks of `r` on the side that is not the other read's own frame
                    let keep = move |x: f32, y: f32| -> bool { let v = if vertical { x } else { y }; if own_first { v >= px } else { v < px } };
                    out.extend(self.add_masked(r, Some(&keep)));
                }
                None => out.extend(self.add(r)),
            }
        }
        out
    }

    fn add_masked(&mut self, read: &FlatRead, mask: Option<&dyn Fn(f32, f32) -> bool>) -> Vec<(FlatHeader, FramePayload)> {
        self.captures += 1;
        if self.skipped(read) { return Vec::new(); }
        // the profile timer exists only when asked for: Instant::now panics in browser WebAssembly
        let t0 = std::env::var_os("LB_PROFILE").map(|_| std::time::Instant::now());
        let mark = |what: &str| { if let Some(t0) = t0 { eprintln!("  [stitch] {:>7.1} ms  {}", t0.elapsed().as_secs_f64() * 1000.0, what); } };
        let hdr = read.header; let grid = &read.grid; let reads: &FlatReads = &read.reads;
        let (bw, _) = grid.blocks();
        let ht = hdr.frame_seq & 3;
        // which frame each block belongs to: tag difference 0 -> this frame, 1 -> next, 3 -> previous, 2 -> unknown
        // which frame each block belongs to, and whether its cells are to be trusted: tag difference 0 ->
        // this frame, 1 -> next, 3 -> previous, 2 -> unknown; a block whose pilots fit no tag (glare, a
        // finger, another format) stays with this frame at zero confidence so one bad block costs erasures,
        // not a stripe
        let frame_of = |b: usize| -> Option<(u8, bool)> {
            let t = *reads.block_tag.get(b)?;
            if t == super::read::TAG_UNKNOWN { return Some((hdr.frame_seq, true)); }
            match (t.wrapping_sub(ht)) & 3 { 0 => Some((hdr.frame_seq, false)), 1 => Some((hdr.frame_seq.wrapping_add(1), false)), 3 => Some((hdr.frame_seq.wrapping_sub(1), false)), _ => None }
        };
        // a block next to a block of another tag holds the refresh line somewhere inside: its cells are a
        // mix of two frames, kept only with zero confidence so a later clean read replaces them
        let (_, bh) = grid.blocks();
        let edge = |b: usize| -> bool {
            let (bx, by) = ((b % bw as usize) as i64, (b / bw as usize) as i64);
            let t = reads.block_tag.get(b).copied();
            [(bx - 1, by), (bx + 1, by), (bx, by - 1), (bx, by + 1)].iter().any(|&(x, y)| x >= 0 && y >= 0 && x < bw as i64 && y < bh as i64 && reads.block_tag.get((y * bw as i64 + x) as usize).copied() != t)
        };
        // per block, once: the frame it belongs to (a slot below) and whether its cells are to be trusted
        let nb = (bw * bh) as usize;
        let mut slots: Vec<(u8, Option<Partial>)> = Vec::new();
        let block: Vec<Option<(usize, bool)>> = (0..nb).map(|b| {
            let (seq, untrusted) = frame_of(b)?;
            let si = slots.iter().position(|s| s.0 == seq).unwrap_or_else(|| { slots.push((seq, None)); slots.len() - 1 });
            Some((si, untrusted || edge(b)))
        }).collect();
        let key_of = |seq: u8| -> Key { (hdr.transfer_id, seq, grid.format().pitch, grid.format().alphabet.id(), grid.cols(), grid.rows()) };
        let cols = grid.cols();
        // each frame's partial leaves the map for the merge (one lookup per frame, not per cell) and goes back after
        for i in 0..grid.used_data_cells() as u32 {
            let (c, r) = grid.data_position(i);
            let Some((si, mixed)) = block[((r / BLOCK) * bw + c / BLOCK) as usize] else { continue };
            if let Some(m) = mask { let (x, y) = grid.cell_center(c, r); if !m(x, y) { continue; } }
            let slot = &mut slots[si];
            let part = slot.1.get_or_insert_with(|| self.frames.remove(&key_of(slot.0)).unwrap_or_else(|| {
                let n = (grid.cols() * grid.rows()) as usize;
                Partial { grid: grid.clone(), header: FlatHeader { frame_seq: slot.0, ..hdr }, values: vec![0; n], margins: vec![0; n], filled: vec![false; n],
                    bytes: vec![0; grid.payload_bytes()], cw_done: vec![false; grid.codewords()], packet_done: vec![false; grid.packets()], dirty: false, last_seen: 0 }
            }));
            let (v, m) = reads.at(c, r);
            let m = if mixed { 0 } else { m };
            let idx = (r * cols + c) as usize;
            if !part.filled[idx] || m > part.margins[idx] { part.values[idx] = v; part.margins[idx] = m; part.filled[idx] = true; part.dirty = true; }
        }
        let mut touched: Vec<u8> = Vec::new();
        for (seq, part) in slots {
            let Some(mut part) = part else { continue };
            part.last_seen = self.captures;
            self.frames.insert(key_of(seq), part);
            touched.push(seq);
        }
        mark("cells merged");
        let mut out = Vec::new();
        // no block of this capture under its own frame number: none of that frame's codewords read
        let mut last = (0, grid.codewords());
        for seq in touched {
            let key: Key = (hdr.transfer_id, seq, grid.format().pitch, grid.format().alphabet.id(), grid.cols(), grid.rows());
            let part = self.frames.get_mut(&key).unwrap();
            let own = seq == hdr.frame_seq;
            if !part.dirty || part.cw_done.iter().all(|&d| d) {
                if own { last = (part.cw_done.iter().filter(|&&d| d).count(), part.cw_done.len()); }
                continue;
            }
            part.dirty = false;
            let g = part.grid.clone();
            let fresh = extract_cells(&g, &part.values, &part.margins, Some(&part.filled), Some(&part.cw_done));
            mark("codewords decoded");
            let dl = g.format().data_len();
            for c in 0..g.codewords() {
                if !part.cw_done[c] && fresh.valid[c * dl] {
                    part.cw_done[c] = true;
                    part.bytes[c * dl..(c + 1) * dl].copy_from_slice(&fresh.bytes[c * dl..(c + 1) * dl]);
                }
            }
            if own { last = (part.cw_done.iter().filter(|&&d| d).count(), g.codewords()); }
            // a packet is new when every codeword its bytes sit in has now decoded
            let mut valid = vec![false; part.bytes.len()];
            let mut any = false;
            for k in 0..g.packets() {
                if part.packet_done[k] { continue; }
                let (c0, c1) = (k * PACKET_BYTES / dl, ((k + 1) * PACKET_BYTES - 1) / dl);
                if (c0..=c1).all(|c| part.cw_done[c]) {
                    part.packet_done[k] = true; any = true;
                    for v in &mut valid[k * PACKET_BYTES..(k + 1) * PACKET_BYTES] { *v = true; }
                }
            }
            if any {
                let ok = part.cw_done.iter().filter(|&&d| d).count();
                out.push((part.header, FramePayload { bytes: part.bytes.clone(), valid, codewords_ok: ok, codewords_failed: g.codewords() - ok - fresh.codewords_missing, codewords_missing: fresh.codewords_missing, coverage: fresh.coverage }));
            }
        }
        self.last = last;
        // forget old frames
        if self.frames.len() > self.keep {
            let mut keys: Vec<(Key, u64)> = self.frames.iter().map(|(k, p)| (*k, p.last_seen)).collect();
            keys.sort_by_key(|x| x.1);
            for (k, _) in keys.into_iter().take(self.frames.len() - self.keep) { self.frames.remove(&k); }
        }
        out
    }
}

/// The cut a read's tag map shows: (vertical?, position in tile px, own-frame side first?) — None when the
/// map has no clean two-region split. "Own" means the header's own tag; anything else (another tag or
/// unknown) is the other side.
fn cut_of(read: &FlatRead) -> Option<(bool, f32, bool)> {
    let grid = &read.grid; let (bw, bh) = grid.blocks(); let own = read.header.frame_seq & 3;
    let tags = &read.reads.block_tag;
    if tags.len() != (bw * bh) as usize { return None; }
    let score = |vertical: bool| -> Option<(usize, bool, f32)> {
        let n = if vertical { bw } else { bh } as usize;
        // per row (or column): own blocks minus other blocks
        let bal: Vec<i32> = (0..n).map(|i| {
            let mut s = 0i32;
            for j in 0..(if vertical { bh } else { bw }) as usize {
                let b = if vertical { j * bw as usize + i } else { i * bw as usize + j };
                // unknown counts as the other side: a frame of another format under this grid reads as unknown
                match tags[b] { t if t == own => s += 1, _ => s -= 1 }
            }
            s
        }).collect();
        let total: i32 = bal.iter().sum();
        let mut best: Option<(usize, bool, f32)> = None;
        let mut prefix = 0i32;
        for cut in 1..n {
            prefix += bal[cut - 1];
            let (lo, hi) = (prefix, total - prefix);
            // own first: lo positive, hi negative; other first: the reverse
            for (own_first, q) in [(true, lo - hi), (false, hi - lo)] {
                let q = q as f32;
                if best.map_or(true, |b| q > b.2) { best = Some((cut, own_first, q)); }
            }
        }
        let (cut, own_first, q) = best?;
        // a clean split: the quality must beat the no-cut extremes clearly
        if q <= total.abs() as f32 + 2.0 { return None; }
        Some((cut, own_first, q))
    };
    let h = score(false); let v = score(true);
    let (vertical, (cut, own_first, _)) = match (h, v) { (Some(a), Some(b)) => if b.2 > a.2 { (true, b) } else { (false, a) }, (Some(a), None) => (false, a), (None, Some(b)) => (true, b), _ => return None };
    let (x, y) = grid.cell_top_left(if vertical { cut as u32 * BLOCK } else { 0 }, if vertical { 0 } else { cut as u32 * BLOCK });
    Some((vertical, if vertical { x as f32 } else { y as f32 }, own_first))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::Homography;
    use crate::flat::{compose, render, Alphabet, CellFormat, FlatGrid, FlatHeader};
    use crate::flat::read::{read_cells, ReadParams};
    use crate::header::FrameType;
    use crate::image::RgbImage;

    fn frame(g: &Arc<FlatGrid>, seq: u8, salt: u64) -> (crate::flat::FlatFrame, Vec<u8>) {
        let h = FlatHeader { format: g.format(), cols: g.cols() as u16, rows: g.rows() as u16, frame_type: FrameType::Data, transfer_id: 7, segment: 1, block_id: 99, frame_seq: seq };
        let payload: Vec<u8> = (0..g.payload_bytes()).map(|i| ((i as u64 * 2654435761 + salt * 7919) >> 7) as u8).collect();
        (compose(g.clone(), h, &payload), payload)
    }
    fn straddle(top: &RgbImage, bottom: &RgbImage, row: u32) -> RgbImage {
        let mut cam = top.clone();
        for y in row..cam.height { for x in 0..cam.width { cam.set(x, y, bottom.get(x, y)); } }
        cam
    }
    fn capture(g: &Arc<FlatGrid>, img: &RgbImage, hdr: FlatHeader) -> FlatRead {
        let reads = read_cells(img, g, &Homography::identity(), &ReadParams { tag_hint: hdr.frame_seq & 3, ..ReadParams::default() });
        FlatRead { grid: g.clone(), header: hdr, homography: Homography::identity(), reads, decode_ms: 0.0 }
    }

    #[test]
    fn the_middle_frame_is_completed_from_two_straddled_captures() {
        let g = Arc::new(FlatGrid::try_new(1024, 1024, CellFormat { pitch: 4, alphabet: Alphabet::C8 }).unwrap());
        let (a, _) = frame(&g, 4, 1); let (b, pb) = frame(&g, 5, 2); let (c, _) = frame(&g, 6, 3);
        let (ia, ib, ic) = (render(&a), render(&b), render(&c));
        let cap1 = capture(&g, &straddle(&ia, &ib, 400), a.header);   // A above 400, B below
        let cap2 = capture(&g, &straddle(&ib, &ic, 700), b.header);   // B above 700, C below
        let mut st = Stitcher::new();
        let packets_of = |out: &[(FlatHeader, FramePayload)]| -> usize { out.iter().filter(|(h, _)| h.frame_seq == 5).map(|(_, p)| p.valid.iter().filter(|&&v| v).count() / PACKET_BYTES).sum() };
        let out1 = st.add(&cap1);
        let b1 = packets_of(&out1);
        // v8: B's codewords each span the whole frame; capture 1 holds only B's bottom 60%, so it can give at
        // most a few packets of B, never all of them
        assert!(b1 < g.packets(), "capture 1 alone cannot complete B ({b1} packets)");
        let out2 = st.add(&cap2);
        let b2 = packets_of(&out2);
        assert_eq!(b1 + b2, g.packets(), "after capture 2 every packet of B has been reported exactly once");
        // and the bytes are B's
        for (h, p) in out1.iter().chain(out2.iter()).filter(|(h, _)| h.frame_seq == 5) {
            for (i, &v) in p.valid.iter().enumerate() { if v { assert_eq!(p.bytes[i], pb[i], "byte {i} of frame {}", h.frame_seq); } }
        }
        // a third add of the same capture reports nothing new
        assert!(st.add(&cap2).iter().all(|(_, p)| !p.valid.iter().any(|&v| v)));
    }
    #[test]
    fn one_px_frame_is_completed_from_two_halves() {
        // the 20x format: each half alone is beyond the codewords' erasure budget; together they decode whole
        let g = Arc::new(FlatGrid::try_new(1024, 1024, CellFormat { pitch: 1, alphabet: Alphabet::Bw }).unwrap());
        let (a, _) = frame(&g, 8, 1); let (b, pb) = frame(&g, 9, 2); let (c, _) = frame(&g, 10, 3);
        let (ia, ib, ic) = (render(&a), render(&b), render(&c));
        let mut st = Stitcher::new();
        let out1 = st.add(&capture(&g, &straddle(&ia, &ib, 512), a.header));
        assert!(out1.iter().all(|(h, p)| h.frame_seq != 9 || !p.valid.iter().any(|&v| v)), "half of B decodes nothing");
        // B (tag 1) over C (tag 2): the tags differ in both bits, the case that once left C's half untagged
        let out2 = st.add(&capture(&g, &straddle(&ib, &ic, 512), b.header));
        let got = out2.iter().find(|(h, _)| h.frame_seq == 9).expect("B completes");
        assert_eq!(got.1.valid.iter().filter(|&&v| v).count(), g.packets() * PACKET_BYTES);
        assert_eq!(&got.1.bytes[..g.packets() * PACKET_BYTES], &pb[..g.packets() * PACKET_BYTES]);
    }
    #[test]
    fn frames_of_different_formats_are_stitched_from_both_header_reads() {
        // capture 1: top of data frame A (seq 4, 4 px) over the bottom of manifest frame M (seq 5, 12 px). The
        // decoder reads it once per visible header (decode_all): with A's grid and with M's grid. Capture 2:
        // top of M (down to row 700) over the bottom of A again. M's packet must come out of M's two parts.
        let ga = Arc::new(FlatGrid::try_new(1024, 1024, CellFormat { pitch: 4, alphabet: Alphabet::C8 }).unwrap());
        let gm = Arc::new(FlatGrid::try_new(1024, 1024, crate::flat::MANIFEST_FORMAT).unwrap());
        let (a, _) = frame(&ga, 4, 1);
        let hm = FlatHeader { format: gm.format(), cols: gm.cols() as u16, rows: gm.rows() as u16, frame_type: FrameType::Manifest, transfer_id: 7, segment: 0, block_id: 500, frame_seq: 5 };
        let pm: Vec<u8> = (0..gm.payload_bytes()).map(|i| (i * 31 % 251) as u8).collect();
        let m = compose(gm.clone(), hm, &pm);
        let (ia, im) = (render(&a), render(&m));
        let img1 = straddle(&ia, &im, 600);
        let img2 = straddle(&im, &ia, 700);
        let mut st = Stitcher::new();
        let mut reported = Vec::new();
        for (img, _) in [(&img1, 0), (&img2, 0)] {
            let (ca, cm) = (capture(&ga, img, a.header), capture(&gm, img, hm));
            assert!(cut_of(&ca).is_some() || cut_of(&cm).is_some(), "one of the two reads shows the cut");
            reported.extend(st.add_capture(&[ca, cm]));
        }
        let got = reported.iter().find(|(h, p)| h.frame_seq == 5 && h.format == gm.format() && p.valid[..gm.packets() * PACKET_BYTES].iter().all(|&v| v)).expect("the manifest frame comes out whole");
        assert_eq!(&got.1.bytes[..gm.packets() * PACKET_BYTES], &pm[..gm.packets() * PACKET_BYTES]);
    }
}
