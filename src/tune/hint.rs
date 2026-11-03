//! The live hint: during a normal transfer, when the receiver has read poorly for `WINDOW_MS`, it offers safer b/w
//! words (coarser cells, when the cells are what holds it back), or Tune; never when it would change nothing.
//!
//! - A tick (about a second of status) is poor when the tile was in view (decoded rose by at least
//!   `IN_VIEW_PER_S` a second) and its codewords share was under `POOR_CODEWORDS`, or more than `POOR_SHARE` of its
//!   captures failed, or its cells were under the format's floor.
//! - The condition holds when every tick of the last `WINDOW_MS` is poor, the transfer is not complete, the time left
//!   at the window's rate is over `END_S`, and the window's goodput is under `TUNE_BELOW` x the format's nominal
//!   rate at the sender's fps.
//! - Words: the format is b/w on `CHAIN`, its cells (the window's lower quartile of camera px per cell) are small
//!   enough that the b/w curve (`bw_share`, from the 2 px floor benchmark) says it should not read whole, and the
//!   window is not mostly blended frames (bigger cells do not cure a blend): the first coarser pitch on the chain the
//!   curve says reads (`WORDS_MIN_SHARE`) and that would be at least `WORDS_GAIN` x faster. Its words keep the
//!   sender's rate. Otherwise Tune.
//! - A shown hint stays until the condition has been false for `WINDOW_MS` or the format changes.
use super::choose::grid_numbers;
use crate::calib::{fps_index, mode_words, Config};
use crate::flat::{Alphabet, CellFormat, MANIFEST_EVERY_FRAMES, MANIFEST_FORMAT};
use crate::packet::PACKET_PAYLOAD;
use std::collections::VecDeque;

/// Poor for this long: the hint shows; fine for this long: it goes.
pub const WINDOW_MS: f64 = 5000.0;
/// A tick with more failed captures than this share is poor.
pub const POOR_SHARE: f64 = 0.25;
/// A tick whose codewords share is under this is poor.
pub const POOR_CODEWORDS: f64 = 0.75;
/// The tile is in view when decoded captures rise by at least this many a second.
pub const IN_VIEW_PER_S: f64 = 3.0;
/// The hint shows only while the goodput is under this share of the format's nominal rate.
pub const TUNE_BELOW: f64 = 0.4;
/// Words are offered only for a format at least this much faster than the window's rate.
pub const WORDS_GAIN: f64 = 1.25;
/// The share of codewords the b/w curve must predict for the coarser cells.
pub const WORDS_MIN_SHARE: f64 = 0.95;
/// The b/w pitches the hint steps along: 2 (Fast) -> 3 (Reliable) -> 5.
pub const CHAIN: [u8; 3] = [2, 3, 5];
/// No hint when the transfer would end within this many seconds at the window's rate.
pub const END_S: f64 = 15.0;
/// The length of a tick.
const TICK_MS: f64 = 1000.0;

/// What the hint reads from one status: the receiver's counters and the last data read (its format "2px bw
/// 1024x1024" with the tile's size, camera px per cell, codewords, blend), the bytes in so far, the sender's rate.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HintSample { pub t_ms: f64, pub decoded: u64, pub failed: u64, pub format: Option<(CellFormat, u32, u32)>, pub cell_px: f64, pub codewords_ok: u64, pub codewords: u64, pub mix: f64, pub received_bytes: u64, pub stream_size: u64, pub complete: bool, pub sender_fps: f64 }

/// A status's format, "2px bw 1024x1024": (format, tile width, tile height).
pub fn parse_format(s: &str) -> Option<(CellFormat, u32, u32)> {
    let mut it = s.split_whitespace();
    let pitch = it.next()?.strip_suffix("px")?.parse().ok()?;
    let alphabet = Alphabet::from_name(it.next()?)?;
    let (w, h) = it.next()?.split_once('x')?;
    if it.next().is_some() { return None; }
    Some((CellFormat { pitch, alphabet }, w.parse().ok()?, h.parse().ok()?))
}

/// What the receiver offers: words to type on the sender (a safer format), or Tune.
#[derive(Clone, Debug, PartialEq)]
pub enum LiveHint { Words { config: Config, words: String, mode: String, text: String }, Tune { text: String } }

impl LiveHint {
    fn words(config: Config) -> LiveHint {
        let words = config.to_code();
        let text = format!("Stop sending, type {words} in Tune words, press Start: what has arrived is kept.");
        LiveHint::Words { config, mode: mode_words(&config), words, text }
    }
    fn tune() -> LiveHint { LiveHint::Tune { text: "Not reading well from here: press Tune on the sender, then Tune here.".into() } }
    /// `{"kind":"words","words","mode","text"}` or `{"kind":"tune","text"}`.
    pub fn to_json(&self) -> String {
        let q = |s: &str| { let mut o = String::from("\""); for c in s.chars() { match c { '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"), c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)), c => o.push(c) } } o.push('"'); o };
        match self {
            LiveHint::Words { words, mode, text, .. } => format!("{{\"kind\":\"words\",\"words\":{},\"mode\":{},\"text\":{}}}", q(words), q(mode), q(text)),
            LiveHint::Tune { text } => format!("{{\"kind\":\"tune\",\"text\":{}}}", q(text)),
        }
    }
}

/// The share of codewords a b/w frame reads at `camera_px_per_cell` (the smallest edge's), from the 2 px floor
/// benchmark (docs/2026-10-31-fhd-2px-floor.md, soft-focus rows where they exist, rounded down): 1.15 -> 0,
/// 1.25 -> 0.5, 1.37 -> 0.95, 1.47 -> 0.99, 1.57 -> 1, linear between, 0 below, 1 above.
pub fn bw_share(camera_px_per_cell: f64) -> f64 {
    const CURVE: [(f64, f64); 5] = [(1.15, 0.0), (1.25, 0.5), (1.37, 0.95), (1.47, 0.99), (1.57, 1.0)];
    let x = camera_px_per_cell;
    if !x.is_finite() || x <= CURVE[0].0 { return 0.0; }
    for w in CURVE.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        if x <= x1 { return y0 + (y1 - y0) * (x - x0) / (x1 - x0); }
    }
    1.0
}

/// The share of packets read whole when each codeword decodes with probability `codeword_share`: a packet spans
/// `PACKET_BYTES / data_len + 1` codewords on average (5.5 for b/w) and needs all of them.
pub fn packet_share(codeword_share: f64, format: CellFormat) -> f64 {
    codeword_share.clamp(0.0, 1.0).powf(crate::packet::PACKET_BYTES as f64 / format.data_len() as f64 + 1.0)
}

/// The camera px per cell a format needs (the guidance's floors): 1.45 for b/w at 2 px and coarser, else 2.45.
pub fn floor(pitch: u8, bw: bool) -> f64 { if bw && pitch >= 2 { 1.45 } else { 2.45 } }

/// The first coarser b/w pitch on `CHAIN` that the curve says reads (`WORDS_MIN_SHARE`) where `pitch` reads at
/// `cell_px_q1` camera px per cell (the cells grow by the pitches' ratio); None when the current cells already
/// read (they are not the cause) or nothing on the chain would.
pub fn coarser_bw(pitch: u8, cell_px_q1: f64) -> Option<u8> {
    if pitch == 0 || bw_share(cell_px_q1) >= WORDS_MIN_SHARE { return None; }
    CHAIN.iter().copied().filter(|&p| p > pitch).find(|&p| bw_share(cell_px_q1 * p as f64 / pitch as f64) >= WORDS_MIN_SHARE)
}

/// File bits a second `format` carries on a `w` x `h` tile at `fps` (manifest frames excluded).
fn nominal_bits(format: CellFormat, w: u32, h: u32, fps: f64) -> f64 {
    let packets = grid_numbers(w, h, format).map_or(0, |n| n.1);
    (packets * PACKET_PAYLOAD) as f64 * 8.0 * fps * (MANIFEST_EVERY_FRAMES - 1) as f64 / MANIFEST_EVERY_FRAMES as f64
}

/// About a second of status.
#[derive(Clone, Debug)]
struct Tick { t0: f64, t1: f64, poor: bool, mix: f64, cells: Vec<f64>, received0: u64, received1: u64, fps: f64 }

/// What a tick gathers from the statuses inside it.
#[derive(Default)]
struct Acc { cw: f64, cw_n: u32, mix: f64, mix_n: u32, cells: Vec<f64>, fps: f64, fps_n: u32 }

/// Watches the statuses of a normal transfer and holds the hint (`update`).
#[derive(Default)]
pub struct HintWatch { start: Option<HintSample>, acc: Acc, ticks: VecDeque<Tick>, format: Option<(CellFormat, u32, u32)>, shown: Option<LiveHint>, false_since: Option<f64> }

impl HintWatch {
    pub fn new() -> HintWatch { HintWatch::default() }
    /// One status (any rate: the app's every capture, the logs' every second): the hint to show, if any.
    pub fn update(&mut self, s: &HintSample) -> Option<&LiveHint> {
        // the data format (a manifest read says nothing of it); a new one starts over
        if let Some(f) = s.format.filter(|f| f.0 != MANIFEST_FORMAT) {
            if self.format.is_some_and(|g| g != f) { self.ticks.clear(); self.shown = None; self.false_since = None; self.acc = Acc::default(); self.start = None; }
            self.format = Some(f);
            if s.cell_px > 0.0 { self.acc.cells.push(s.cell_px); }
        }
        if s.codewords > 0 { self.acc.cw += s.codewords_ok as f64 / s.codewords as f64; self.acc.cw_n += 1; }
        self.acc.mix += s.mix; self.acc.mix_n += 1;
        if s.sender_fps > 0.0 { self.acc.fps += s.sender_fps; self.acc.fps_n += 1; }
        if s.complete { self.shown = None; }
        let Some(start) = self.start.clone() else { self.start = Some(s.clone()); return self.shown.as_ref() };
        if s.t_ms - start.t_ms < TICK_MS { return self.shown.as_ref(); }
        // close the tick
        let dt = (s.t_ms - start.t_ms) / 1000.0;
        let (dd, df) = (s.decoded.saturating_sub(start.decoded) as f64, s.failed.saturating_sub(start.failed) as f64);
        let acc = std::mem::take(&mut self.acc);
        let cw = (acc.cw_n > 0).then(|| acc.cw / acc.cw_n as f64);
        let cell_min = acc.cells.iter().copied().fold(f64::MAX, f64::min);
        let small = self.format.is_some_and(|(f, _, _)| acc.cells.len() > 0 && cell_min < floor(f.pitch, f.alphabet == Alphabet::Bw));
        let in_view = dd / dt >= IN_VIEW_PER_S;
        let failing = dd + df > 0.0 && df / (dd + df) > POOR_SHARE;
        let poor = in_view && (cw.is_some_and(|c| c < POOR_CODEWORDS) || failing || small);
        self.ticks.push_back(Tick { t0: start.t_ms, t1: s.t_ms, poor, mix: if acc.mix_n > 0 { acc.mix / acc.mix_n as f64 } else { 0.0 }, cells: acc.cells,
            received0: start.received_bytes, received1: s.received_bytes, fps: if acc.fps_n > 0 { acc.fps / acc.fps_n as f64 } else { 0.0 } });
        while self.ticks.front().is_some_and(|t| t.t1 < s.t_ms - 2.0 * WINDOW_MS) { self.ticks.pop_front(); }
        self.start = Some(s.clone());

        match self.offer(s) {
            Some(h) => { if self.shown.is_none() { self.shown = Some(h); } self.false_since = None; }
            None => if self.shown.is_some() {
                let since = *self.false_since.get_or_insert(s.t_ms);
                if s.t_ms - since >= WINDOW_MS { self.shown = None; self.false_since = None; }
            },
        }
        self.shown.as_ref()
    }

    /// The hint the last `WINDOW_MS` call for, if the condition holds.
    fn offer(&self, s: &HintSample) -> Option<LiveHint> {
        let (format, w, h) = self.format?;
        if s.complete { return None; }
        let win: Vec<&Tick> = self.ticks.iter().filter(|t| t.t1 > s.t_ms - WINDOW_MS + 1.0).collect();
        let span = win.iter().map(|t| t.t1 - t.t0).sum::<f64>();
        if win.is_empty() || span < WINDOW_MS - TICK_MS || !win.iter().all(|t| t.poor) { return None; }
        let (t0, t1) = (win[0].t0, win[win.len() - 1].t1);
        let rate = win[win.len() - 1].received1.saturating_sub(win[0].received0) as f64 / ((t1 - t0) / 1000.0).max(1e-3);
        // near the end at this rate: nothing
        if s.stream_size > 0 && rate > 0.0 && s.stream_size.saturating_sub(s.received_bytes) as f64 / rate <= END_S { return None; }
        let fps_v: Vec<f64> = win.iter().map(|t| t.fps).filter(|&f| f > 0.0).collect();
        let fps = if fps_v.is_empty() { 30.0 } else { fps_v.iter().sum::<f64>() / fps_v.len() as f64 };
        // reading at a good share of what this format carries: a hint would change nothing
        if rate * 8.0 >= TUNE_BELOW * nominal_bits(format, w, h, fps) { return None; }
        let blends = win.iter().filter(|t| t.mix > super::choose::BLEND).count() as f64 / win.len() as f64;
        let mut cells: Vec<f64> = win.iter().flat_map(|t| t.cells.iter().copied()).collect();
        cells.sort_by(|a, b| a.total_cmp(b));
        let q1 = cells.get(((cells.len().max(1) - 1) as f64 * 0.25) as usize).copied();
        if format.alphabet == Alphabet::Bw && CHAIN.contains(&format.pitch) && blends <= 0.5 {
            if let Some(p) = q1.and_then(|q| coarser_bw(format.pitch, q)) {
                let coarser = CellFormat { pitch: p, alphabet: Alphabet::Bw };
                // the transfer receives whole packets: the curve's codeword share as a packet share
                let predicted = nominal_bits(coarser, w, h, fps) * packet_share(bw_share(q1.unwrap() * p as f64 / format.pitch as f64), coarser);
                if predicted >= WORDS_GAIN * rate * 8.0 {
                    // the sender's rate, if a code can carry it
                    let rounded = [20u32, 30, 60].into_iter().find(|&r| (fps - r as f64).abs() <= 0.12 * r as f64);
                    let fps_i = rounded.and_then(fps_index).unwrap_or(0);
                    return Some(LiveHint::words(Config { pitch: p, alphabet: Alphabet::Bw, fps: fps_i }));
                }
            }
        }
        Some(LiveHint::tune())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn formats_parse() {
        assert_eq!(parse_format("2px bw 1024x1024"), Some((CellFormat { pitch: 2, alphabet: Alphabet::Bw }, 1024, 1024)));
        assert_eq!(parse_format("12px c8 1024x640"), Some((MANIFEST_FORMAT, 1024, 640)));
        assert_eq!(parse_format("2px zz 1024x1024"), None);
        assert_eq!(parse_format(""), None);
    }
    #[test]
    fn curve_and_chain() {
        assert_eq!(bw_share(1.1), 0.0); assert_eq!(bw_share(1.25), 0.5); assert_eq!(bw_share(1.6), 1.0);
        assert!((bw_share(1.31) - 0.725).abs() < 1e-9);
        assert_eq!(coarser_bw(2, 1.15), Some(3));
        assert_eq!(coarser_bw(2, 1.7), None, "cells that read are not the cause");
        assert_eq!(coarser_bw(2, 0.85), Some(5));
        assert_eq!(coarser_bw(5, 1.2), None);
        assert_eq!(floor(2, true), 1.45); assert_eq!(floor(4, false), 2.45);
        let bw2 = CellFormat { pitch: 2, alphabet: Alphabet::Bw };
        assert!((packet_share(0.95, bw2) - 0.95f64.powf(1030.0 / 231.0 + 1.0)).abs() < 1e-12);
        assert_eq!(packet_share(1.0, bw2), 1.0); assert_eq!(packet_share(0.0, bw2), 0.0);
    }
    #[test]
    fn hint_json() {
        let w = LiveHint::words(Config { pitch: 3, alphabet: Alphabet::Bw, fps: 0 });
        assert_eq!(w.to_json(), "{\"kind\":\"words\",\"words\":\"sauce-minivan\",\"mode\":\"Reliable\",\"text\":\"Stop sending, type sauce-minivan in Tune words, press Start: what has arrived is kept.\"}");
        assert_eq!(LiveHint::tune().to_json(), "{\"kind\":\"tune\",\"text\":\"Not reading well from here: press Tune on the sender, then Tune here.\"}");
    }
}
