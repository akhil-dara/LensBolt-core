//! The choice: from one loop of Tune samples, each segment's yield with the drift discount, the fastest
//! configuration that keeps reading from this hold, and one line of advice on the hold.
//!
//! Per segment (format and rate of the carousel):
//! - `full`: packets read whole over the distinct frames read (each frame's best capture), against the packets
//!   shown: the segment's data frames inside the measured window while the tile was in view (the meter's count), or
//!   its frames less the manifest slots (30 in 31) for a whole loop. Packets, not codewords: a packet spans 5-6
//!   codewords and the transfer receives only whole ones.
//! - `capture(subset)`: the mean per-capture share of packets read whole (a capture that read no tile counts 0).
//! - The worse hold: captures with sampling (the smallest edge's camera px per screen px) at or below the run's
//!   lower quartile, or perspective at or above its upper quartile (a clause counts only where the run spread).
//! - `safe = full x min(1, capture(worse) / capture(all))`: a format that reads only at the best moments of the
//!   hold scores low.
//! - Expected Mbit/s = `safe x shown fps x 30/31 x packets x PACKET_PAYLOAD x 8 / 1e6` for the tile read.
//!
//! The highest expected among the segments with `safe >= MIN_SAFE_YIELD` wins (60 fps segments only where the
//! receiver captures at 60 and the sender's screen drew them at 60); Fast stays unless the winner beats Fast's
//! expected by `SWITCH_MARGIN`, so a tuned choice is never below the default measured from the same hold.
use super::sample::{SampleRead, TuneSample};
use super::schedule::{shown_fps, Slot, SEGMENTS};
use crate::calib::{mode_words, Config, FAST, FPS_OPTIONS};
use crate::flat::{CellFormat, FlatGrid, MANIFEST_EVERY_FRAMES};
use crate::packet::PACKET_PAYLOAD;
use std::collections::HashMap;

/// A winner must beat Fast's expected rate by this much to replace it.
pub const SWITCH_MARGIN: f64 = 1.10;
/// The least safe yield a segment needs to be chosen.
pub const MIN_SAFE_YIELD: f64 = 0.5;
/// A capture subset "reads well" at this share of codewords (the advice's target).
pub const READS_WELL: f64 = 0.9;
/// Perspective (largest edge over smallest) under which a hold counts as square on.
pub const PERSPECTIVE_MIN: f64 = 1.05;
/// Corner motion (share of the tile side per 33.3 ms) under which a hold counts as steady.
pub const SHAKE_MIN: f64 = 0.005;
/// A border edge wider than this (camera px) on a close hold: the camera cannot focus that close.
pub const EDGE_SOFT: f64 = 2.5;
/// A data read with more than this blend shows two sender frames.
pub const BLEND: f64 = 0.15;

/// What the receiver can do: `high_speed_60`, a capture path that really delivers 60 fps (neither receiver has
/// one yet).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TuneOptions { pub high_speed_60: bool }

/// One segment as measured. `fps`: its `FPS_OPTIONS` index; `captures`: the captures counted for it (data reads
/// of its format and captures that read nothing); `frames_read`: distinct data frames read; `full` and `safe` in
/// packets; `codewords`: the codeword yield on the same frames, for the details.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlotStat { pub format: CellFormat, pub fps: u8, pub captures: u32, pub frames_read: u32, pub full: f64, pub safe: f64, pub expected_mbit: f64, pub decode_ms: f64, pub codewords: f64 }

/// How the hand held. Sampling: the smallest edge's camera px per screen px; perspective: largest edge over
/// smallest; shake: corner motion between consecutive reads as a share of the tile side per 33.3 ms; edge px:
/// the border's 10-90% width; contrast CV: the variation of the cells' mean margin from capture to capture (screen
/// flicker, exposure); blend share: data reads that show two sender frames.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HoldStats { pub sampling_q1: f64, pub sampling_median: f64, pub perspective_q3: f64, pub shake_q3: f64, pub edge_px_median: f64, pub contrast_cv: f64, pub blend_share: f64 }

/// Whether 60 fps was measured: `Measured`; not, because the receiver cannot capture at 60 (`Receiver`); not,
/// because the sender's screen drew the 60 fps segments at 30 (`Display`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sixty { Measured, Receiver, Display }

/// One change to the hold, if any would help: `unlocks` is the configuration it would let read well.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Advice { Keep, Closer { factor: f64, unlocks: Config }, Farther, SquareUp { unlocks: Config }, Steadier { unlocks: Config } }

/// Why Tune gave no choice: no tune frame read, the screen left the picture partway, or nothing read well.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TuneFailure { NothingSeen, Interrupted, Unreadable }

/// Tune's result: the choice and its expected rate, the advice, the runner-up, whether 60 fps was measured, and
/// what was measured (`captures_per_s`: captures that reached the readers per second; the camera's own rate and its
/// dropped captures are the platform's to show).
#[derive(Clone, Debug, PartialEq)]
pub struct TuneResult { pub choice: Config, pub expected_mbit: f64, pub advice: Advice, pub runner_up: Option<(Config, f64)>, pub sixty: Sixty, pub slots: Vec<SlotStat>, pub hold: HoldStats, pub captures_per_s: f64, pub captures: u32, pub seconds: f64 }

/// The safe yield: `full` discounted by how much worse the worse hold reads than the whole run.
pub fn safe_yield(full: f64, capture_all: f64, capture_worse: f64) -> f64 {
    if capture_all <= 0.0 { return 0.0; }
    full * (capture_worse / capture_all).min(1.0)
}

fn fin(v: f64) -> f64 { if v.is_finite() { v } else { 0.0 } }
/// A tile's codewords and packets per frame in `format` (None: the tile cannot hold it), built once per tile and
/// format (a grid of 1024 px at 2 px is a quarter of a million cells).
pub(crate) fn grid_numbers(width: u32, height: u32, format: CellFormat) -> Option<(usize, usize)> {
    type Numbers = HashMap<(u32, u32, CellFormat), Option<(usize, usize)>>;
    static CACHE: std::sync::Mutex<Option<Numbers>> = std::sync::Mutex::new(None);
    let key = (width, height, format);
    if let Some(n) = CACHE.lock().ok().and_then(|c| c.as_ref().and_then(|m| m.get(&key).copied())) { return n; }
    let n = FlatGrid::try_new(width, height, format).map(|g| (g.codewords(), g.packets()));
    if let Ok(mut c) = CACHE.lock() { c.get_or_insert_with(HashMap::new).insert(key, n); }
    n
}
/// The value at quantile `p` of `v` (sorted here): the element at floor((n - 1) p). 0 for none.
fn quantile(v: &[f64], p: f64) -> f64 {
    if v.is_empty() { return 0.0; }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s[((s.len() - 1) as f64 * p).floor() as usize]
}
fn mean(v: impl Iterator<Item = f64>) -> Option<f64> {
    let (mut sum, mut n) = (0.0, 0usize);
    for x in v { sum += x; n += 1; }
    (n > 0).then(|| sum / n as f64)
}
/// The share of data frames: the manifest takes one slot in `MANIFEST_EVERY_FRAMES`.
fn data_share() -> f64 { (MANIFEST_EVERY_FRAMES - 1) as f64 / MANIFEST_EVERY_FRAMES as f64 }
fn config_of(seg: usize) -> Config { let s = SEGMENTS[seg]; Config { pitch: s.format.pitch, alphabet: s.format.alphabet, fps: s.fps } }

/// One capture as the choice sees it: its segment, the share it read (0 for none), and the hold's geometry at
/// that moment (a capture that read nothing takes the nearest read's).
struct Cap<'a> { segment: usize, read: Option<&'a SampleRead>, counted: bool, share: f64, sampling: Option<f64>, perspective: Option<f64>, shake: Option<f64>, edge_px: Option<f64> }

/// Corner motion between two reads, as a share of the tile side per 33.3 ms.
fn motion(a: &SampleRead, b: &SampleRead, dt_ms: f64) -> Option<f64> {
    if !(dt_ms > 0.0 && dt_ms <= 500.0) { return None; }
    let (ca, cb) = (&a.corners, &b.corners);
    let disp = (0..4).map(|i| ((cb[2 * i] - ca[2 * i]) as f64).hypot((cb[2 * i + 1] - ca[2 * i + 1]) as f64)).sum::<f64>() / 4.0;
    let side = (0..4).map(|i| { let j = (i + 1) % 4; ((cb[2 * j] - cb[2 * i]) as f64).hypot((cb[2 * j + 1] - cb[2 * i + 1]) as f64) }).sum::<f64>() / 4.0;
    (side > 0.0).then(|| disp / side * (1000.0 / 30.0) / dt_ms)
}

/// The data frames of each segment a whole loop shows: its frames less the manifest slots.
pub fn loop_shown() -> Vec<f64> { SEGMENTS.iter().map(|s| s.frames as f64 * data_share()).collect() }

/// The choice over one window of samples, each with the slot the meter gave it. `shown`: per segment, the data
/// frames the window showed while the tile was in view (`loop_shown` for a whole loop).
pub fn evaluate(samples: &[(TuneSample, Slot)], opts: TuneOptions, display_60: bool, shown: &[f64]) -> Result<TuneResult, TuneFailure> {
    let mut order: Vec<usize> = (0..samples.len()).collect();
    order.sort_by(|&a, &b| samples[a].0.t_ms.total_cmp(&samples[b].0.t_ms));
    let data_reads: Vec<&SampleRead> = order.iter().filter_map(|&i| samples[i].0.read.as_ref()).filter(|r| !r.manifest).collect();
    let Some(tile) = data_reads.last().map(|r| r.tile) else { return Err(TuneFailure::NothingSeen) };

    // the hold, from every read (a manifest shows the same tile)
    let reads: Vec<(f64, &SampleRead)> = order.iter().filter_map(|&i| samples[i].0.read.as_ref().map(|r| (samples[i].0.t_ms, r))).collect();
    let mut shakes: Vec<Option<f64>> = vec![None; reads.len()];
    for k in 1..reads.len() { shakes[k] = motion(reads[k - 1].1, reads[k].1, reads[k].0 - reads[k - 1].0); }
    if reads.len() > 1 { shakes[0] = shakes[1]; }
    let sampling: Vec<f64> = reads.iter().map(|r| r.1.sampling()).collect();
    let persp: Vec<f64> = reads.iter().map(|r| r.1.perspective()).collect();
    let shake_v: Vec<f64> = shakes.iter().flatten().copied().collect();
    let (s_q1, s_med) = (quantile(&sampling, 0.25), quantile(&sampling, 0.5));
    let (p_med, p_q3) = (quantile(&persp, 0.5), quantile(&persp, 0.75));
    let shake_q3 = quantile(&shake_v, 0.75);
    // a clause of the worse hold counts only where the run spread (a fixed hold has no worse quarter)
    let (s_spread, p_spread) = (s_q1 < s_med, p_q3 > p_med);

    // each capture with its geometry: a capture without a read takes the nearest read's (in time)
    let read_t: Vec<f64> = reads.iter().map(|r| r.0).collect();
    let nearest = |t: f64| -> Option<usize> {
        if read_t.is_empty() { return None; }
        let i = read_t.partition_point(|&x| x < t);
        match (i.checked_sub(1), (i < read_t.len()).then_some(i)) {
            (Some(a), Some(b)) => Some(if t - read_t[a] <= read_t[b] - t { a } else { b }),
            (Some(a), None) => Some(a),
            (None, b) => b,
        }
    };
    let mut read_k = 0usize;
    let caps: Vec<Cap> = order.iter().map(|&i| {
        let (s, slot) = (&samples[i].0, samples[i].1);
        let k = match &s.read { Some(_) => { read_k += 1; Some(read_k - 1) } None => nearest(s.t_ms) };
        let counted = match &s.read { None => true, Some(r) => !r.manifest && r.format == slot.format };
        Cap { segment: slot.segment, read: s.read.as_ref(), counted, share: s.read.as_ref().map_or(0.0, |r| r.share()),
            sampling: k.map(|k| sampling[k]), perspective: k.map(|k| persp[k]), shake: k.and_then(|k| shakes[k]), edge_px: k.map(|k| reads[k].1.edge_px as f64) }
    }).collect();
    let worse = |c: &Cap| (s_spread && c.sampling.is_some_and(|s| s <= s_q1)) || (p_spread && c.perspective.is_some_and(|p| p >= p_q3));
    // the better half of the hold: the larger, squarer and steadier captures
    let shake_med = quantile(&shake_v, 0.5);
    let best_half = |c: &Cap| c.sampling.is_some_and(|s| s >= s_med) && c.perspective.is_some_and(|p| p <= p_med) && c.shake.is_none_or(|s| s <= shake_med);

    // per segment
    let mut slots = Vec::with_capacity(SEGMENTS.len());
    let mut grids: Vec<Option<(usize, usize)>> = Vec::with_capacity(SEGMENTS.len());
    for (i, seg) in SEGMENTS.iter().enumerate() {
        let grid = grid_numbers(tile.0, tile.1, seg.format);
        let mine: Vec<&Cap> = caps.iter().filter(|c| c.segment == i && c.counted).collect();
        // each distinct frame's best capture: packets whole, codewords decoded
        let mut best: HashMap<u8, (u32, u32)> = HashMap::new();
        for c in &mine { if let Some(r) = c.read { let e = best.entry(r.seq).or_insert((0, 0)); *e = (e.0.max(r.packets_ok), e.1.max(r.codewords_ok)); } }
        let frames = shown.get(i).copied().unwrap_or(0.0).max(best.len() as f64);
        let (full, codewords_full, expected) = match grid {
            Some((codewords, packets)) if frames > 0.0 => {
                let full = (best.values().map(|v| v.0 as f64).sum::<f64>() / (frames * packets as f64)).min(1.0);
                let cw = (best.values().map(|v| v.1 as f64).sum::<f64>() / (frames * codewords as f64)).min(1.0);
                (full, cw, shown_fps(seg.fps, display_60) as f64 * data_share() * (packets * PACKET_PAYLOAD) as f64 * 8.0 / 1e6)
            }
            _ => (0.0, 0.0, 0.0),
        };
        let all = mean(mine.iter().map(|c| c.share)).unwrap_or(0.0);
        let worse_y = mean(mine.iter().filter(|c| worse(c)).map(|c| c.share)).unwrap_or(all);
        let safe = safe_yield(full, all, worse_y);
        let decode_ms = mean(mine.iter().filter_map(|c| c.read.map(|r| r.decode_ms))).unwrap_or(0.0);
        slots.push(SlotStat { format: seg.format, fps: seg.fps, captures: mine.len() as u32, frames_read: best.len() as u32, full, safe, expected_mbit: safe * expected, decode_ms, codewords: codewords_full });
        grids.push(grid);
    }

    // the choice
    let sixty_ok = opts.high_speed_60 && display_60;
    let allowed = |i: usize| grids[i].is_some() && (FPS_OPTIONS[SEGMENTS[i].fps as usize] <= 30 || sixty_ok);
    let eligible: Vec<usize> = (0..SEGMENTS.len()).filter(|&i| allowed(i) && slots[i].safe >= MIN_SAFE_YIELD).collect();
    let by_rate = |a: &usize, b: &usize| slots[*a].expected_mbit.total_cmp(&slots[*b].expected_mbit);
    let Some(&best) = eligible.iter().max_by(|a, b| by_rate(a, b)) else { return Err(TuneFailure::Unreadable) };
    let fast = (0..SEGMENTS.len()).find(|&i| config_of(i) == FAST);
    let chosen = match fast {
        Some(f) if slots[f].safe > 0.0 && slots[best].expected_mbit < slots[f].expected_mbit * SWITCH_MARGIN => f,
        _ => best,
    };
    let choice = config_of(chosen);
    let runner_up = eligible.iter().filter(|&&i| config_of(i) != choice).max_by(|a, b| by_rate(a, b)).map(|&i| (config_of(i), slots[i].expected_mbit));
    let sixty = if !opts.high_speed_60 { Sixty::Receiver } else if !display_60 { Sixty::Display } else { Sixty::Measured };

    // the advice: the segment that would be fastest from the better half of this hold, and what held it back
    let rate_best_half = |i: usize| -> f64 {
        let y = mean(caps.iter().filter(|c| c.segment == i && c.counted && best_half(c)).map(|c| c.share)).unwrap_or(0.0);
        grids[i].map_or(0.0, |(_, packets)| packets as f64 * shown_fps(SEGMENTS[i].fps, display_60) as f64 * y)
    };
    let f = (0..SEGMENTS.len()).filter(|&i| allowed(i)).max_by(|&a, &b| rate_best_half(a).total_cmp(&rate_best_half(b)));
    let advice = match f {
        Some(f) if config_of(f) != choice => {
            let fc: Vec<&Cap> = caps.iter().filter(|c| c.segment == f && c.counted).collect();
            let f_best = mean(fc.iter().filter(|c| best_half(c)).map(|c| c.share)).unwrap_or(0.0);
            let unlocks = config_of(f);
            #[derive(Clone, Copy)] enum Cause { Small, Tilted, Shaky, Soft }
            let mut causes: Vec<(Cause, f64)> = Vec::new();
            let mut subset = |cause: Cause, keep: &dyn Fn(&Cap) -> bool| { if let Some(y) = mean(fc.iter().filter(|c| keep(c)).map(|c| c.share)) { causes.push((cause, y)); } };
            if s_spread { subset(Cause::Small, &|c| c.sampling.is_some_and(|s| s <= s_q1)); }
            if p_q3 >= PERSPECTIVE_MIN { subset(Cause::Tilted, &|c| c.perspective.is_some_and(|p| p >= p_q3)); }
            if shake_q3 >= SHAKE_MIN { subset(Cause::Shaky, &|c| c.shake.is_some_and(|s| s >= shake_q3)); }
            subset(Cause::Soft, &|c| c.edge_px.is_some_and(|e| e >= EDGE_SOFT) && c.sampling.is_some_and(|s| s >= 2.0 * s_q1));
            let lowest = causes.iter().fold(None::<(Cause, f64)>, |m, &(c, y)| match m { Some((_, my)) if my <= y => m, _ => Some((c, y)) });
            match lowest {
                Some((cause, y)) if y < 0.9 * f_best => match cause {
                    Cause::Small => {
                        // the smallest sampling (bins of 0.05) at which this format's captures read well
                        let mut bins: HashMap<i64, (f64, usize)> = HashMap::new();
                        for c in &fc { if let Some(s) = c.sampling { let e = bins.entry((s / 0.05 + 1e-4).floor() as i64).or_insert((0.0, 0)); e.0 += c.share; e.1 += 1; } }
                        let mut keys: Vec<i64> = bins.keys().copied().collect(); keys.sort_unstable();
                        let well = keys.iter().find(|k| { let (sum, n) = bins[k]; sum / n as f64 >= READS_WELL }).map(|&k| k as f64 * 0.05);
                        let most = fc.iter().filter_map(|c| c.sampling).fold(0.0f64, f64::max);
                        let factor = match well { Some(s) if s_q1 > 0.0 => s / s_q1, _ if s_q1 > 0.0 => (most / s_q1).max(1.5), _ => 1.5 };
                        Advice::Closer { factor, unlocks }
                    }
                    Cause::Tilted => Advice::SquareUp { unlocks },
                    Cause::Shaky => Advice::Steadier { unlocks },
                    Cause::Soft => Advice::Farther,
                },
                _ => Advice::Keep,
            }
        }
        _ => Advice::Keep,
    };

    // the hold, for the details
    let edge_v: Vec<f64> = reads.iter().map(|r| r.1.edge_px as f64).filter(|&e| e > 0.0).collect();
    let mut by_format: HashMap<CellFormat, Vec<f64>> = HashMap::new();
    for r in &data_reads { by_format.entry(r.format).or_default().push(r.contrast as f64); }
    let (mut cv_sum, mut cv_n) = (0.0, 0usize);
    for v in by_format.values().filter(|v| v.len() >= 3) {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        if m > 0.0 { let sd = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt(); cv_sum += sd / m * v.len() as f64; cv_n += v.len(); }
    }
    let hold = HoldStats {
        sampling_q1: s_q1, sampling_median: s_med, perspective_q3: p_q3, shake_q3, edge_px_median: quantile(&edge_v, 0.5),
        contrast_cv: if cv_n > 0 { cv_sum / cv_n as f64 } else { 0.0 },
        blend_share: data_reads.iter().filter(|r| r.mix as f64 > BLEND).count() as f64 / data_reads.len() as f64,
    };
    let (t0, t1) = (samples[order[0]].0.t_ms, samples[*order.last().unwrap()].0.t_ms);
    let seconds = fin((t1 - t0) / 1000.0).max(0.0);
    let captures = samples.len() as u32;
    let captures_per_s = if seconds > 0.0 { (captures as f64 - 1.0) / seconds } else { 0.0 };
    Ok(TuneResult { choice, expected_mbit: slots[chosen].expected_mbit, advice, runner_up, sixty, slots, hold, captures_per_s, captures, seconds })
}

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() { match c { '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"), c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)), c => o.push(c) } }
    o.push('"');
    o
}

impl TuneResult {
    /// The two words to type on the sender.
    pub fn words(&self) -> String { self.choice.to_code() }
    /// The choice in words ("Fast, 20 fps (3 refreshes per frame)").
    pub fn mode(&self) -> String { mode_words(&self.choice) }
    /// The advice in one line.
    pub fn advice_text(&self) -> String {
        match self.advice {
            Advice::Keep => "Keep this hold".into(),
            Advice::Closer { factor, unlocks } => {
                let m = mode_words(&unlocks);
                if factor <= 1.15 { format!("Hold a little closer for {m}") }
                else if factor <= 1.4 { format!("Hold about a hand closer for {m}") }
                else { format!("Hold closer, so the square fills most of the picture, for {m}") }
            }
            Advice::Farther => "Hold a little farther: the camera cannot focus this close".into(),
            Advice::SquareUp { unlocks } => format!("Square up to the screen for {}", mode_words(&unlocks)),
            Advice::Steadier { unlocks } => format!("Hold steadier (rest your elbows) for {}", mode_words(&unlocks)),
        }
    }
    /// The result as the receivers' screens read it (every number finite).
    pub fn to_json(&self) -> String {
        let advice = match self.advice { Advice::Keep => "keep", Advice::Closer { .. } => "closer", Advice::Farther => "farther", Advice::SquareUp { .. } => "square_up", Advice::Steadier { .. } => "steadier" };
        let runner = match &self.runner_up {
            Some((c, m)) => format!("{{\"words\":{},\"mode\":{},\"expected_mbit\":{:.3}}}", json_str(&c.to_code()), json_str(&mode_words(c)), fin(*m)),
            None => "null".into(),
        };
        let sixty = match self.sixty { Sixty::Measured => "measured", Sixty::Receiver => "receiver", Sixty::Display => "display" };
        let h = &self.hold;
        let hold = format!("{{\"sampling_q1\":{:.3},\"sampling_median\":{:.3},\"perspective_q3\":{:.3},\"shake_q3\":{:.4},\"edge_px_median\":{:.2},\"contrast_cv\":{:.3},\"blend_share\":{:.3}}}",
            fin(h.sampling_q1), fin(h.sampling_median), fin(h.perspective_q3), fin(h.shake_q3), fin(h.edge_px_median), fin(h.contrast_cv), fin(h.blend_share));
        let slots: Vec<String> = self.slots.iter().map(|s| format!("{{\"format\":\"{}px {}\",\"fps\":{},\"full\":{:.3},\"safe\":{:.3},\"expected_mbit\":{:.3},\"captures\":{},\"decode_ms\":{:.1},\"codewords\":{:.3}}}",
            s.format.pitch, s.format.alphabet.name(), FPS_OPTIONS[s.fps as usize], fin(s.full), fin(s.safe), fin(s.expected_mbit), s.captures, fin(s.decode_ms), fin(s.codewords))).collect();
        format!("{{\"words\":{},\"mode\":{},\"expected_mbit\":{:.3},\"advice\":\"{advice}\",\"advice_text\":{},\"runner_up\":{runner},\"sixty\":\"{sixty}\",\"captures_per_s\":{:.2},\"seconds\":{:.2},\"captures\":{},\"hold\":{hold},\"slots\":[{}]}}",
            json_str(&self.words()), json_str(&self.mode()), fin(self.expected_mbit), json_str(&self.advice_text()), fin(self.captures_per_s), fin(self.seconds), self.captures, slots.join(","))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calib::Calibrator;
    use crate::flat::{Alphabet, MANIFEST_FORMAT};
    use crate::tune::schedule::{frame_ms, loop_ms, slot};

    /// (codewords, packets) of a 1024 px tile in `f`.
    fn grid(f: CellFormat) -> (usize, usize) { grid_numbers(1024, 1024, f).unwrap() }
    /// The frame on screen `t` ms into a loop that started at frame 0.
    fn seq_at(t: f64, display_60: bool) -> u8 {
        let mut at = 0.0;
        for s in 0..=255u8 { at += frame_ms(s, display_60); if t < at { return s; } }
        255
    }
    /// A capture's (packets whole, codewords decoded) when each codeword decodes with probability `share`
    /// (independently, seeded by `seed`), as a frame's payload lays them out.
    fn codeword_model(f: CellFormat, share: f64, seed: u64) -> (u32, u32) {
        let (cw, packets) = grid(f);
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let ok: Vec<bool> = (0..cw).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; ((x >> 11) as f64 / (1u64 << 53) as f64) < share }).collect();
        (crate::tune::sample::packets_from_codewords(&ok, f.data_len(), packets), ok.iter().filter(|&&b| b).count() as u32)
    }
    /// One loop filmed by a camera at `cam_fps`: `f(slot, capture)` gives the hold (sampling, perspective), the
    /// share of packets read whole (< 0: no tile read) and the blend; `corners` moves the tile (shake).
    fn film(cam_fps: f64, display_60: bool, f: &dyn Fn(Slot, usize) -> (f32, f32, f32, f32), corners: &dyn Fn(usize) -> (f32, f32)) -> Calibrator {
        film_model(cam_fps, display_60, f, corners, false)
    }
    /// `film`, the share being of codewords (`codeword_model`) when `codewords`.
    fn film_model(cam_fps: f64, display_60: bool, f: &dyn Fn(Slot, usize) -> (f32, f32, f32, f32), corners: &dyn Fn(usize) -> (f32, f32), codewords: bool) -> Calibrator {
        let mut cal = Calibrator::new();
        let total = loop_ms(display_60);
        for k in 0.. {
            let t = 5.0 + k as f64 * 1000.0 / cam_fps;
            if t >= total { break; }
            let seq = seq_at(t, display_60);
            let sl = slot(seq);
            let (s, p, share, mix) = f(sl, k);
            let read = (share >= 0.0).then(|| {
                let manifest = seq as u32 % MANIFEST_EVERY_FRAMES == 0;
                let (cw, pk) = grid(sl.format);
                let (pk_ok, cw_ok) = if codewords { codeword_model(sl.format, share as f64, k as u64 + 1) }
                    else { ((share as f64 * pk as f64).round() as u32, (share as f64 * cw as f64).round() as u32) };
                let (dx, dy) = corners(k);
                let side = 1024.0 * s;
                let m = |v: u32| if manifest { 0 } else { v };
                SampleRead { seq, format: if manifest { MANIFEST_FORMAT } else { sl.format }, manifest,
                    packets_ok: m(pk_ok), packets: m(pk as u32), codewords_ok: m(cw_ok), codewords: m(cw as u32),
                    tile: (1024, 1024), edges: [s, s, s * p, s],
                    corners: [100.0 + dx, 100.0 + dy, 100.0 + side + dx, 100.0 + dy, 100.0 + side + dx, 100.0 + side + dy, 100.0 + dx, 100.0 + side + dy],
                    mix, edge_px: 1.5, contrast: 100.0 + (k % 3) as f32, decode_ms: 10.0 }
            });
            cal.observe_sample(&TuneSample { t_ms: t, read }, sl);
        }
        // the data frames each segment showed (the meter counts them the same way: the manifest every 31 frames)
        let mut shown = vec![0.0; SEGMENTS.len()];
        for seq in 0..=255u8 { if seq as u32 % MANIFEST_EVERY_FRAMES != 0 { shown[slot(seq).segment] += 1.0; } }
        cal.set_shown(shown);
        cal
    }
    fn run(f: impl Fn(Slot, usize) -> (f32, f32, f32, f32)) -> Calibrator { film(30.0, true, &f, &|_| (0.0, 0.0)) }
    fn bw(pitch: u8) -> CellFormat { CellFormat { pitch, alphabet: Alphabet::Bw } }
    fn stat(r: &TuneResult, c: Config) -> SlotStat { *r.slots.iter().find(|s| s.format == c.format() && s.fps == c.fps).unwrap() }

    #[test]
    fn perfect_reads_pick_the_densest_at_30() {
        let r = run(|_, _| (1.1, 1.0, 1.0, 0.0)).result(TuneOptions::default(), true).unwrap();
        assert_eq!(r.choice, Config { pitch: 2, alphabet: Alphabet::G8, fps: 0 });
        assert_eq!(r.sixty, Sixty::Receiver);
        assert_eq!(r.advice, Advice::Keep);
        let (_, packets) = grid(r.choice.format());
        let s = stat(&r, r.choice);
        assert!(s.safe > 0.95, "{s:?}");
        let want = s.safe * 30.0 * 30.0 / 31.0 * (packets * PACKET_PAYLOAD) as f64 * 8.0 / 1e6;
        assert!((r.expected_mbit - want).abs() < 0.01 * want, "{} vs {want}", r.expected_mbit);
        assert!((r.captures_per_s - 30.0).abs() < 0.5, "{}", r.captures_per_s);
        assert!(r.runner_up.is_some());
    }

    #[test]
    fn sixty_needs_the_receiver_and_the_sender_screen() {
        // a 60 fps camera reading every frame of the b/w segments (grey and colour read nothing)
        let only_bw = |sl: Slot, _: usize| (1.6f32, 1.0f32, if sl.format.alphabet == Alphabet::Bw { 1.0f32 } else { 0.0 }, 0.0f32);
        let cal = film(60.0, true, &only_bw, &|_| (0.0, 0.0));
        let r = cal.result(TuneOptions { high_speed_60: true }, true).unwrap();
        assert_eq!(r.choice, Config { fps: 1, ..FAST });
        assert_eq!(r.sixty, Sixty::Measured);
        assert_eq!(r.mode(), "Fast, 60 fps");
        // the receiver cannot capture at 60: Fast at 30
        let r = cal.result(TuneOptions { high_speed_60: false }, true).unwrap();
        assert_eq!((r.choice, r.sixty), (FAST, Sixty::Receiver));
        // the sender drew the 60 fps segments at 30
        let cal = film(60.0, false, &only_bw, &|_| (0.0, 0.0));
        let r = cal.result(TuneOptions { high_speed_60: true }, false).unwrap();
        assert_eq!((r.choice, r.sixty), (FAST, Sixty::Display));
    }

    #[test]
    fn blends_at_30_pick_20() {
        // 30 fps segments: 3 of 4 captures blend two frames and read nothing; 20 fps segments read whole
        let r = run(|sl, k| if sl.fps == 0 { if k % 4 != 0 { (1.6, 1.0, 0.0, 0.5) } else { (1.6, 1.0, 1.0, 0.0) } } else { (1.6, 1.0, 1.0, 0.0) })
            .result(TuneOptions::default(), true).unwrap();
        assert_eq!(r.choice, Config { pitch: 2, alphabet: Alphabet::Bw, fps: 2 });
        assert_eq!(r.mode(), "Fast, 20 fps (3 refreshes per frame)");
        assert!(r.hold.blend_share > 0.4, "{:?}", r.hold);
    }

    #[test]
    fn a_format_that_reads_only_at_the_best_moments_scores_low() {
        // sampling cycles 0.70..0.95 (q1 0.70); 2 px b/w reads at >= 0.85 only; 3 px and 5 px b/w always
        const CYCLE: [f32; 8] = [0.70, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.70];
        let r = run(|sl, k| {
            let s = CYCLE[k % CYCLE.len()];
            let share = match (sl.format.alphabet, sl.format.pitch) { (Alphabet::Bw, 2) => if s >= 0.85 { 1.0 } else { 0.0 }, (Alphabet::Bw, _) => 1.0, _ => 0.0 };
            (s, 1.0, share, 0.0)
        }).result(TuneOptions::default(), true).unwrap();
        assert_eq!(r.choice.format(), bw(3));
        assert_eq!(r.choice.fps, 0);
        assert!((r.hold.sampling_q1 - 0.70).abs() < 1e-6, "{:?}", r.hold);
        assert!(matches!(r.advice, Advice::Closer { unlocks: FAST, .. }), "{:?}", r.advice);
        assert_eq!(r.advice_text(), "Hold about a hand closer for Fast");
        // Fast read whole at the best moments: its full yield is high, its safe yield is not
        let f = stat(&r, FAST);
        assert!(f.full > 0.3 && f.safe < 0.05, "{f:?}");
    }

    #[test]
    fn a_tilted_hold_says_square_up() {
        // perspective cycles 1.00, 1.03, 1.10, 1.10 (q3 1.10); 2 px b/w fails at >= 1.09, 3 px b/w always reads
        const P: [f32; 4] = [1.00, 1.03, 1.10, 1.10];
        let r = run(|sl, k| {
            let p = P[k % 4];
            let share = match (sl.format.alphabet, sl.format.pitch) { (Alphabet::Bw, 2) => if p >= 1.09 { 0.0 } else { 1.0 }, (Alphabet::Bw, _) => 1.0, _ => 0.0 };
            (1.6, p, share, 0.0)
        }).result(TuneOptions::default(), true).unwrap();
        assert_eq!(r.choice.format(), bw(3));
        assert!((r.hold.perspective_q3 - 1.10).abs() < 1e-3, "{:?}", r.hold);
        assert!(matches!(r.advice, Advice::SquareUp { unlocks: FAST }), "{:?}", r.advice);
        assert_eq!(r.advice_text(), "Square up to the screen for Fast");
    }

    #[test]
    fn a_shaky_hold_says_steadier() {
        // the tile moves by 0..27 camera px between captures (back and forth); 2 px b/w fails when it moved 9 px or
        // more (7 captures in 10), 3 px b/w always reads
        let step = |k: usize| ((k * 7) % 10) as f32 * 3.0;
        let at = move |k: usize| ((0..=k).map(|j| if j % 2 == 0 { step(j) } else { -step(j) }).sum::<f32>(), 0.0);
        let f = move |sl: Slot, k: usize| {
            let share = match (sl.format.alphabet, sl.format.pitch) { (Alphabet::Bw, 2) => if step(k) >= 9.0 { 0.0 } else { 1.0 }, (Alphabet::Bw, _) => 1.0, _ => 0.0 };
            (1.0f32, 1.0f32, share, 0.0f32)
        };
        let r = film(30.0, true, &f, &at).result(TuneOptions::default(), true).unwrap();
        assert!(r.hold.shake_q3 >= SHAKE_MIN, "{:?}", r.hold);
        assert_eq!(r.choice.format(), bw(3), "{:?} {:?}", r.choice, r.slots);
        assert!(matches!(r.advice, Advice::Steadier { unlocks: FAST }), "{:?}", r.advice);
        assert_eq!(r.advice_text(), "Hold steadier (rest your elbows) for Fast");
    }

    #[test]
    fn fast_stays_unless_beaten_by_ten_percent() {
        // Fast reads 0.55 of every frame; 4 px c8 reads what makes it 5% (then 20%) faster than Fast
        let ratio = grid(CellFormat { pitch: 4, alphabet: Alphabet::C8 }).1 as f64 / grid(bw(2)).1 as f64;
        for (gain, want) in [(1.05, FAST), (1.2, Config { pitch: 4, alphabet: Alphabet::C8, fps: 0 })] {
            let share = (0.55 * gain / ratio) as f32;
            assert!(share > 0.5 && share <= 1.0, "{share}");
            let r = run(|sl, _| {
                let y = if sl.format == bw(2) && sl.fps == 0 { 0.55 } else if sl.format == (CellFormat { pitch: 4, alphabet: Alphabet::C8 }) { share } else { 0.0 };
                (1.6, 1.0, y, 0.0)
            }).result(TuneOptions::default(), true).unwrap();
            assert_eq!(r.choice, want, "gain {gain}: {:?}", r.slots);
        }
    }

    #[test]
    fn never_below_the_default() {
        let mut x = 0x9e3779b97f4a7c15u64;
        let mut rnd = move || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x >> 11) as f64 / (1u64 << 53) as f64 };
        let mut checked = 0;
        for _ in 0..200 {
            let level: Vec<f64> = (0..SEGMENTS.len()).map(|_| rnd()).collect();
            let noise: Vec<f64> = (0..400).map(|_| rnd()).collect();
            let s0 = 0.8 + rnd() as f32;
            let r = run(|sl, k| {
                let s = s0 * (0.85 + 0.3 * noise[k % 400] as f32);
                let y = (level[sl.segment] * (0.6 + 0.8 * noise[(k * 7) % 400])).min(1.0) as f32;
                (s, 1.0 + 0.1 * noise[(k * 3) % 400] as f32, if noise[(k * 11) % 400] < 0.05 { -1.0 } else { y }, 0.0)
            }).result(TuneOptions::default(), true);
            let Ok(r) = r else { continue };
            let f = stat(&r, FAST);
            if f.safe > 0.0 { assert!(r.expected_mbit >= f.expected_mbit, "{} < Fast's {}", r.expected_mbit, f.expected_mbit); checked += 1; }
        }
        assert!(checked > 50, "{checked}");
    }

    #[test]
    fn nothing_reads_well_is_a_failure() {
        assert_eq!(run(|_, _| (1.6, 1.0, 0.3, 0.0)).result(TuneOptions::default(), true), Err(TuneFailure::Unreadable));
        assert_eq!(run(|_, _| (1.6, 1.0, -1.0, 0.0)).result(TuneOptions::default(), true), Err(TuneFailure::NothingSeen));
        assert_eq!(Calibrator::new().result(TuneOptions::default(), true), Err(TuneFailure::NothingSeen));
    }

    #[test]
    fn result_json_has_its_fields() {
        let r = run(|sl, k| (1.2 + 0.1 * (k % 5) as f32, 1.0 + 0.02 * (k % 3) as f32, if sl.format.alphabet == Alphabet::Bw { 0.97 } else { 0.4 }, 0.0))
            .result(TuneOptions::default(), true).unwrap();
        let j = r.to_json();
        for k in ["\"words\":", "\"mode\":", "\"expected_mbit\":", "\"advice\":", "\"advice_text\":", "\"runner_up\":", "\"sixty\":\"receiver\"", "\"captures_per_s\":", "\"seconds\":", "\"captures\":",
                  "\"hold\":{\"sampling_q1\":", "\"sampling_median\":", "\"perspective_q3\":", "\"shake_q3\":", "\"edge_px_median\":", "\"contrast_cv\":", "\"blend_share\":", "\"slots\":[{\"format\":\"2px bw\",\"fps\":30,\"full\":"] {
            assert!(j.contains(k), "{k} in {j}");
        }
        assert_eq!(j.matches("\"format\":").count(), SEGMENTS.len());
        assert!(!j.contains("NaN") && !j.contains("inf"), "{j}");
        assert_eq!(Config::from_code(&r.words()), Ok(r.choice));
        assert!(j.starts_with('{') && j.ends_with('}'));
    }

    #[test]
    fn a_format_reading_most_codewords_is_not_chosen_over_a_whole_fast() {
        // 2 px g8 decodes 60% of its codewords (5-6 per packet: few packets whole), Fast every one
        let f = |sl: Slot, _: usize| (1.6f32, 1.0f32, if sl.format == bw(2) { 1.0f32 } else if sl.format.alphabet == Alphabet::G8 { 0.6 } else { 0.0 }, 0.0f32);
        let r = film_model(30.0, true, &f, &|_| (0.0, 0.0), true).result(TuneOptions::default(), true).unwrap();
        assert_eq!(r.choice, FAST, "{:?}", r.slots);
        let g8 = stat(&r, Config { pitch: 2, alphabet: Alphabet::G8, fps: 0 });
        assert!(g8.codewords > 0.5 && g8.full < 0.1, "{g8:?}");
    }

    #[test]
    fn expected_rate_is_the_transfer_goodput() {
        // each format's expected Mbit/s against a transfer of a file in that format whose frames lose the packets of
        // the codewords that did not decode (the same model, frame by frame), at 30 fps
        use crate::file::{FeedOutcome, Receiver};
        use crate::flat::{FlatSender, FlatSenderConfig};
        let share = |f: CellFormat| if f.alphabet == Alphabet::Bw { 0.995 } else { 0.98 };
        let cal = film_model(30.0, true, &|sl: Slot, _| (1.6, 1.0, share(sl.format) as f32, 0.0), &|_| (0.0, 0.0), true);
        let r = cal.result(TuneOptions::default(), true).unwrap();
        let data: Vec<u8> = { let mut x = 7u64; (0..14_000_000).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x as u8 }).collect() };
        for f in [bw(2), bw(3), CellFormat { pitch: 2, alphabet: Alphabet::G8 }, CellFormat { pitch: 4, alphabet: Alphabet::C8 }] {
            let mut tx = FlatSender::from_bytes("g.bin", data.clone(), FlatSenderConfig { format: f, ..FlatSenderConfig::default() }).unwrap();
            let mut rx = Receiver::in_memory();
            let (cw, _) = grid(f);
            let frames = 150u64;
            for n in 0..frames {
                let (h, mut p) = tx.next_packets().unwrap();
                if h.frame_type == crate::header::FrameType::Data {
                    let mut x = (n + 77).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
                    let ok: Vec<bool> = (0..cw).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; ((x >> 11) as f64 / (1u64 << 53) as f64) < share(f) }).collect();
                    let mut k = 0usize;
                    p.retain(|_| { let (a, b) = (k * crate::packet::PACKET_BYTES / f.data_len(), ((k + 1) * crate::packet::PACKET_BYTES - 1) / f.data_len()); k += 1; (a..=b).all(|c| ok[c]) });
                }
                assert!(!matches!(rx.feed_packets(&h, &p).unwrap(), FeedOutcome::Complete(_)));
            }
            let goodput = rx.progress().packets_new as f64 * PACKET_PAYLOAD as f64 * 8.0 / (frames as f64 / 30.0) / 1e6;
            let want = stat(&r, Config { pitch: f.pitch, alphabet: f.alphabet, fps: 0 }).expected_mbit;
            assert!((want - goodput).abs() <= 0.15 * goodput, "{f:?}: expected {want:.2} Mbit/s, the transfer {goodput:.2}");
        }
    }

    #[test]
    fn a_slow_drift_touches_every_format() {
        // the hand backs off across the loop: 1.0 down to 0.6 camera px per screen px; 2 px formats read only while it
        // is at 0.9 or more (the first quarter of the loop), 3 px and coarser always. Every format is shown in stretches
        // spread over the loop, so Fast is measured late in the loop too, and is not chosen
        let total = loop_ms(true);
        let f = move |sl: Slot, k: usize| {
            let s = 1.0 - 0.4 * ((5.0 + k as f64 * 1000.0 / 30.0) / total) as f32;
            (s, 1.0, if sl.format.pitch == 2 { if s >= 0.9 { 1.0 } else { 0.0 } } else { 1.0 }, 0.0)
        };
        let r = run(f).result(TuneOptions::default(), true).unwrap();
        let fast = stat(&r, FAST);
        assert!(fast.full < 0.5, "{fast:?}");
        assert!(r.choice.pitch >= 3 && r.choice.fps == 0, "{:?} {:?}", r.choice, r.slots);
    }

    #[test]
    fn safe_yield_formula() {
        assert_eq!(safe_yield(0.9, 0.9, 0.45), 0.45); assert_eq!(safe_yield(0.9, 0.9, 1.0), 0.9); assert_eq!(safe_yield(0.5, 0.0, 0.0), 0.0);
    }
}
