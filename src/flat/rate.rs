//! The sender's display rate, read from the frame numbers of decoded captures: every frame header carries the
//! sender's frame counter mod 256 (`FlatHeader::frame_seq`), and each displayed frame has the next number.
//! Captures come with the time they were taken (any monotonic clock, in ms). Over the last ~1.5 s of captures
//! the 8-bit numbers are unwrapped in time order and a straight line is fitted through (time, frame number):
//! its slope is the rate. Lost captures only thin the points; a camera faster than the sender sees a number
//! twice; captures that come back out of order (several workers) are put back in time order.
//! Which captures belong to the count: until the points give a rate, a step forward no faster than the
//! fastest sender (`max_fps`); once they do, a frame number near the line through the last 2.5 s of them
//! (within 2 frames plus 5% of the predicted step: a looser band lets a restart whose jump happens to land
//! near the prediction bend the rate). Anything else is left out, and two such captures in a row that agree
//! with each other are a new sender (it starts again from frame 0, as when the user changes its rate, or
//! carries on at another rate): the count starts again from them, for good. The stall clock is the last
//! change of the frame number itself, whichever sender showed it.

use crate::calib::FPS_OPTIONS;

/// The window the rate is measured over.
const WINDOW_MS: f64 = 1500.0;
/// No rate before this much time and this many distinct frames in the window.
const MIN_SPAN_MS: f64 = 500.0;
const MIN_FRAMES: usize = 3;
/// No new frame number for this long while captures still arrive: the sender has stopped (rate 0). No decoded
/// capture for this long: the rate is not known (the tile is out of view).
const STALL_MS: f64 = 2000.0;
/// Captures kept: the window, and enough to see a stall.
const KEEP_MS: f64 = 2500.0;
/// Around the line: this many frames, plus this share of the predicted step.
const NEAR_FRAMES: f64 = 2.0;
const NEAR_SHARE: f64 = 0.05;

/// The fastest sender believed before a rate is known: 1.25 x the fastest rate calibration can pick.
fn max_fps() -> f64 { 1.25 * FPS_OPTIONS.iter().copied().max().unwrap_or(120) as f64 }
/// The largest forward step a sender at `max_fps` makes in `dt_ms`.
fn max_step(dt_ms: f64) -> u32 { 1 + (dt_ms.max(0.0) * max_fps() / 1000.0).ceil().min(255.0) as u32 }

/// Points (time, unwrapped frame number) that give a rate: at least 0.5 s and 3 distinct frames.
fn enough(w: &[(f64, i64)], t_end: f64) -> bool {
    !w.is_empty() && t_end - w[0].0 >= MIN_SPAN_MS && 1 + w.windows(2).filter(|p| p[1].1 > p[0].1).count() >= MIN_FRAMES
}
/// Least-squares line through the points: (slope in frames per ms, mean time, mean frame number).
fn fit(w: &[(f64, i64)]) -> Option<(f64, f64, f64)> {
    let n = w.len() as f64;
    let (tm, um) = (w.iter().map(|p| p.0).sum::<f64>() / n, w.iter().map(|p| p.1 as f64).sum::<f64>() / n);
    let (mut num, mut den) = (0.0, 0.0);
    for &(t, u) in w { num += (t - tm) * (u as f64 - um); den += (t - tm) * (t - tm); }
    (den > 0.0).then(|| (num / den, tm, um))
}

/// Frame numbers of decoded captures against their capture times.
#[derive(Clone, Debug)]
pub struct RateEstimate {
    caps: Vec<(f64, u8)>,
    /// where the current count starts (a new sender seen): older captures are not counted again, so the
    /// decision stands when the captures that showed it have left the window
    since: f64,
}

impl Default for RateEstimate { fn default() -> Self { RateEstimate { caps: Vec::new(), since: f64::NEG_INFINITY } } }

impl RateEstimate {
    pub fn new() -> Self { RateEstimate::default() }
    /// One decoded capture: the frame number it read and when it was taken (ms). Captures may arrive out of
    /// order; one older than everything kept is ignored, as is a time that is not a number.
    pub fn add(&mut self, frame_seq: u8, t_ms: f64) {
        if !t_ms.is_finite() { return; }
        let newest = self.caps.last().map_or(t_ms, |c| c.0.max(t_ms));
        if t_ms < newest - KEEP_MS { return; }
        let at = self.caps.partition_point(|c| c.0 <= t_ms);
        self.caps.insert(at, (t_ms, frame_seq));
        let old = self.caps.partition_point(|c| c.0 < newest - KEEP_MS);
        self.caps.drain(..old);
        if let (_, Some(restart)) = self.count() { self.since = restart; }
    }
    /// New frames per second as of the newest decoded capture, or None until at least 0.5 s and 3 distinct
    /// frames are in the window. 0 once the frame number has not changed for 2 s (never the rate from before
    /// the stall).
    pub fn fps(&self) -> Option<f64> { self.caps.last().and_then(|c| self.fps_at(c.0)) }
    /// `fps` at `now_ms` (the latest capture, decoded or not): None once nothing has decoded for 2 s, so the
    /// rate does not hold while the tile is out of view.
    pub fn fps_at(&self, now_ms: f64) -> Option<f64> {
        let t_end = self.caps.last()?.0;
        if now_ms.is_finite() && now_ms - t_end >= STALL_MS { return None; }
        // the last time the frame number changed (whichever sender showed it)
        let mut changed = self.caps[0].0;
        for p in self.caps.windows(2) { if p[1].1 != p[0].1 { changed = p[1].0; } }
        if t_end - changed >= STALL_MS { return Some(0.0); }
        let (pts, _) = self.count();
        // the counted points of the last 1.5 s, spanning 0.5 s themselves (a capture left out does not stretch
        // a few old points into a rate)
        let w = &pts[pts.partition_point(|p| p.0 < t_end - WINDOW_MS)..];
        if !w.last().is_some_and(|l| enough(w, l.0)) { return None; }
        fit(w).map(|(r, _, _)| (r * 1000.0).max(0.0))
    }
    /// The count: unwrapped frame numbers of the captures that belong to it, in time order, and where it
    /// started again if a new sender showed up.
    fn count(&self) -> (Vec<(f64, i64)>, Option<f64>) {
        let (mut prev, mut rejected, mut u, mut restart) = (None::<(f64, u8)>, None::<(f64, u8)>, 0i64, None);
        let mut pts: Vec<(f64, i64)> = Vec::with_capacity(self.caps.len());
        for &(t, seq) in &self.caps[self.caps.partition_point(|c| c.0 < self.since)..] {
            let Some((tp, sp)) = prev else { prev = Some((t, seq)); pts.push((t, 0)); continue };
            let raw = seq.wrapping_sub(sp) as i64;
            // the line through everything kept of this count: after a gap of up to 2 s it still predicts
            let w = &pts[pts.partition_point(|p| p.0 < t - KEEP_MS)..];
            let line = if w.last().is_some_and(|l| enough(w, l.0)) { fit(w) } else { None };
            let step = match line {
                // the step (mod 256, forward) nearest the line's prediction, if near enough
                Some((r, tm, um)) => {
                    let predicted = um + r * (t - tm);
                    let k = ((predicted - u as f64 - raw as f64) / 256.0).round().max(0.0) as i64;
                    let step = raw + 256 * k;
                    let near = ((u + step) as f64 - predicted).abs() <= NEAR_FRAMES + NEAR_SHARE * r.max(0.0) * (t - tp);
                    near.then_some(step)
                }
                None => (raw as u32 <= max_step(t - tp)).then_some(raw),
            };
            match step {
                Some(step) => { rejected = None; u += step; prev = Some((t, seq)); pts.push((t, u)); }
                None => match rejected {
                    // two left-out captures in a row that agree: a new sender, counted from the first of them
                    Some((tr, sr)) if seq.wrapping_sub(sr) as u32 <= max_step(t - tr) => {
                        pts.clear();
                        pts.push((tr, 0));
                        restart = Some(tr);
                        u = seq.wrapping_sub(sr) as i64;
                        pts.push((t, u));
                        (prev, rejected) = (Some((t, seq)), None);
                    }
                    _ => rejected = Some((t, seq)),
                },
            }
        }
        (pts, restart)
    }
}
