//! The Tune meter: one loop of the carousel from any start, in both receivers' collectors.
//!
//! - Anchor: the first data read of a tune frame. A frame's position is `(seq - anchor seq) mod 256`.
//! - A capture that read no tile takes the slot of the latest read before it, advanced by the schedule's time
//!   since that read (`schedule::frame_ms`).
//! - Restart: a read more than `RESTART_BACK` frames behind the furthest position (or further ahead than the time
//!   since the last read allows), before most of a loop has passed: the sender started again (a refit, a new
//!   countdown). The samples before it are dropped and the meter re-anchors.
//! - Done: a read at position `CLOSE_FRAMES` or later once every segment has a capture, a read past the loop (the
//!   next loop's), or `CLOSE_SLACK` loops of time since the anchor. Then `Calibrator::result` over the window.
//! - Interrupted: a segment with no tile read of any kind within `GAP_MS` of its middle (the screen left the
//!   picture): no choice is built on the missing segments.
//! - Nothing seen: no anchor `TIMEOUT_MS` after the start. Not tuning: the receiver read a normal transfer's frames
//!   (shown while waiting; a tune frame later turns it back to waiting).
//!
//! Samples may arrive out of order (the browser's reader pool): everything is worked out on the samples sorted by
//! time.
use super::choose::{TuneFailure, TuneOptions, TuneResult};
use super::sample::TuneSample;
use super::schedule::{frame_ms, loop_ms, slot, Slot, LOOP_FRAMES, SEGMENTS};
use crate::calib::Calibrator;
use crate::file::TuneInfo;
use crate::flat::MANIFEST_EVERY_FRAMES;

/// No tune frame read this long after the start: nothing seen.
pub const TIMEOUT_MS: f64 = 15_000.0;
/// A read at this position (or later) closes the window, once every segment has a capture.
pub const CLOSE_FRAMES: usize = 240;
/// The window closes at the latest this many loops (of the sender's schedule) after the anchor.
pub const CLOSE_SLACK: f64 = 1.3;
/// A segment with no read within this of its middle: the screen left the picture.
pub const GAP_MS: f64 = 400.0;
/// A frame shown within this of a tile read counts as shown in view (a few failed captures in a row do not take it
/// out; the screen out of the picture does).
pub const IN_VIEW_MS: f64 = 200.0;
/// A read this many frames behind the furthest position: the sender started again.
pub const RESTART_BACK: usize = 8;
/// The window is settled this long after the read that closed it: a capture read late (another reader) still counts,
/// so the result does not depend on the order the reads came back in.
pub const SETTLE_MS: f64 = 300.0;

/// Where Tune stands. `Measuring(progress)`: 0 to 1 of the loop.
#[derive(Clone, Debug, PartialEq)]
pub enum TuneState { Waiting, NotTuning, Measuring(f64), Done(Box<TuneResult>), Failed(TuneFailure) }

pub struct TuneMeter {
    opts: TuneOptions, start_ms: f64,
    /// how the tuning sender draws its 60 fps segments; None until a tune frame was read
    display_60: Option<bool>,
    /// the receiver read a normal transfer's frames since the last tune frame
    normal: bool,
    samples: Vec<TuneSample>,
    /// a tune frame was read (a manifest counts): a run that never reads a data frame is unreadable, not unseen
    tune_reads: bool,
    /// the result once the window closed (or Tune failed): kept until Tune starts again
    last: Option<TuneState>,
}

/// How the window stands over the samples (sorted by time): where the run starts, the anchor, the furthest
/// position, and whether (and where) it closed.
struct Window { start: usize, anchor: Option<(usize, u8, f64)>, max_pos: usize, end: Option<usize>, by_time: bool, by_pos: bool }

impl TuneMeter {
    pub fn new(opts: TuneOptions, now_ms: f64) -> TuneMeter {
        TuneMeter { opts, start_ms: now_ms, display_60: None, normal: false, samples: Vec::new(), tune_reads: false, last: None }
    }
    /// No data frame read by the timeout: nothing seen, or (tune manifests read: the screen is there) unreadable.
    fn no_anchor(&self) -> TuneFailure { if self.tune_reads { TuneFailure::Unreadable } else { TuneFailure::NothingSeen } }
    /// Measuring could not go on (the collector caught a panic): Tune ends as interrupted, to be started again.
    pub fn abort(&mut self) { self.last = Some(TuneState::Failed(TuneFailure::Interrupted)); }
    /// What the receiver knows of the sender: `tune` for a tune frame read (its manifest's section, or what its
    /// transfer id says), `normal` for a normal transfer's frame read.
    pub fn sender(&mut self, tune: Option<TuneInfo>, normal: bool) {
        if self.last.is_some() { return; }
        if let Some(info) = tune {
            // a sender that now draws its 60 fps segments otherwise is another run
            if self.display_60.is_some_and(|d| d != info.display_60) { self.samples.clear(); }
            self.display_60 = Some(info.display_60);
            self.normal = false;
        } else if normal {
            // the sender shows a file now: whatever was measured is dropped, Tune waits for the carousel
            self.normal = true;
            self.display_60 = None;
            self.samples.clear();
        }
    }
    /// One capture: a tune frame read (the capture's data read, else its manifest read), or None for a capture that
    /// read no tile. Ignored before a tune frame was read and after the window closed.
    pub fn capture(&mut self, s: TuneSample) {
        if self.last.is_some() { return; }
        if self.display_60.is_none() || s.t_ms.is_nan() {
            if s.t_ms - self.start_ms >= TIMEOUT_MS && !self.normal { self.last = Some(TuneState::Failed(self.no_anchor())); }
            return;
        }
        if s.read.is_some() { self.tune_reads = true; }
        let at = self.samples.partition_point(|x| x.t_ms <= s.t_ms);
        self.samples.insert(at, s);
        let now = self.samples.last().map_or(self.start_ms, |x| x.t_ms);
        let w = self.window(now);
        // a restart: the samples before it are of another run
        if w.start > 0 { self.samples.drain(..w.start); }
        let w = if w.start > 0 { self.window(now) } else { w };
        if w.anchor.is_none() && now - self.start_ms >= TIMEOUT_MS { self.last = Some(TuneState::Failed(self.no_anchor())); return; }
        if self.settled(&w, now) { self.last = Some(self.close(&w)); }
    }
    /// The window closed and no late read can change it any more.
    fn settled(&self, w: &Window, now_ms: f64) -> bool {
        match w.end {
            Some(_) if w.by_time => true,
            Some(end) => end > 0 && now_ms - self.samples[end.min(self.samples.len()) - 1].t_ms >= SETTLE_MS,
            None => false,
        }
    }
    /// Where Tune stands at `now_ms`.
    pub fn state(&self, now_ms: f64) -> TuneState {
        if let Some(s) = &self.last { return s.clone(); }
        let Some(d60) = self.display_60 else {
            if self.normal { return TuneState::NotTuning; }
            return if now_ms - self.start_ms >= TIMEOUT_MS { TuneState::Failed(TuneFailure::NothingSeen) } else { TuneState::Waiting };
        };
        let w = self.window(now_ms);
        match w.anchor {
            None => if now_ms - self.start_ms >= TIMEOUT_MS { TuneState::Failed(self.no_anchor()) } else { TuneState::Waiting },
            Some((_, _, t_anchor)) => {
                if self.settled(&w, now_ms) { return self.close(&w); }
                let by_pos = w.max_pos as f64 / (LOOP_FRAMES - 1) as f64;
                let by_time = (now_ms - t_anchor) / (CLOSE_SLACK * loop_ms(d60));
                TuneState::Measuring(by_pos.max(by_time.min(by_pos + 0.05)).clamp(0.0, 0.99))
            }
        }
    }
    /// The state as the receivers' status carries it: `{"state","progress","failure","result"}`.
    pub fn status_json(&self, now_ms: f64) -> String {
        let (state, progress, failure, result) = match self.state(now_ms) {
            TuneState::Waiting => ("waiting", 0.0, None, None),
            TuneState::NotTuning => ("not_tuning", 0.0, None, None),
            TuneState::Measuring(p) => ("measuring", p, None, None),
            TuneState::Done(r) => ("done", 1.0, None, Some(r.to_json())),
            TuneState::Failed(f) => ("failed", 0.0, Some(match f { TuneFailure::NothingSeen => "nothing_seen", TuneFailure::Interrupted => "interrupted", TuneFailure::Unreadable => "unreadable" }), None),
        };
        let progress = if progress.is_finite() { progress } else { 0.0 };
        format!("{{\"state\":\"{state}\",\"progress\":{progress:.3},\"failure\":{},\"result\":{}}}",
            failure.map_or_else(|| "null".into(), |f| format!("\"{f}\"")), result.unwrap_or_else(|| "null".into()))
    }

    /// The window over the samples (sorted by time) as of `now_ms`.
    fn window(&self, now_ms: f64) -> Window {
        let d60 = self.display_60.unwrap_or(false);
        let lp = loop_ms(d60);
        let mut w = Window { start: 0, anchor: None, max_pos: 0, end: None, by_time: false, by_pos: false };
        let mut last: Option<(usize, f64)> = None;
        for (i, s) in self.samples.iter().enumerate() {
            let Some(r) = &s.read else { continue };
            let Some((_, aseq, t_anchor)) = w.anchor else {
                if !r.manifest { w.anchor = Some((i, r.seq, s.t_ms)); w.max_pos = 0; last = Some((0, s.t_ms)); }
                continue;
            };
            let pos = (r.seq.wrapping_sub(aseq)) as usize;
            let (lpos, lt) = last.unwrap_or((0, t_anchor));
            let back = pos + RESTART_BACK < w.max_pos;
            let ahead = pos > lpos.max(w.max_pos) + RESTART_BACK + ((s.t_ms - lt).max(0.0) / (1000.0 / 60.0)).ceil() as usize;
            if back && s.t_ms - t_anchor >= 0.9 * lp {
                // the next loop: the window is this loop's samples
                w.end = Some(i); break;
            }
            if back || ahead {
                w.start = i; w.max_pos = 0;
                w.anchor = (!r.manifest).then_some((i, r.seq, s.t_ms));
                last = w.anchor.map(|_| (0, s.t_ms));
                continue;
            }
            w.max_pos = w.max_pos.max(pos);
            last = Some((pos, s.t_ms));
            if pos >= CLOSE_FRAMES && self.covered(&w, i + 1) { w.end = Some(i + 1); w.by_pos = true; break; }
        }
        if w.end.is_none() {
            if let Some((_, _, t_anchor)) = w.anchor {
                if now_ms - t_anchor >= CLOSE_SLACK * lp { w.end = Some(self.samples.len()); w.by_time = true; }
            }
        }
        w
    }
    /// The window's samples (from the anchor to `end`) with their slots: a read's frame number names it, a capture
    /// that read nothing takes the latest read's, advanced by the schedule's time.
    fn slotted(&self, w: &Window, end: usize) -> Vec<(TuneSample, Slot)> {
        let d60 = self.display_60.unwrap_or(false);
        let Some((a, _, _)) = w.anchor else { return Vec::new() };
        let mut out = Vec::with_capacity(end.saturating_sub(a));
        let mut last: Option<(u8, f64)> = None;
        for s in &self.samples[a..end] {
            match &s.read {
                Some(r) => { last = Some((r.seq, s.t_ms)); out.push((s.clone(), slot(r.seq))); }
                None => {
                    let Some((mut seq, mut t)) = last else { continue };
                    for _ in 0..LOOP_FRAMES { let d = frame_ms(seq, d60); if t + d > s.t_ms { break; } t += d; seq = seq.wrapping_add(1); }
                    out.push((s.clone(), slot(seq)));
                }
            }
        }
        out
    }
    /// Every segment has a capture among the window's samples up to `end`.
    fn covered(&self, w: &Window, end: usize) -> bool {
        let mut seen = [false; SEGMENTS.len()];
        for (_, sl) in self.slotted(w, end) { seen[sl.segment] = true; }
        seen.iter().all(|&s| s)
    }
    /// The closed window's outcome: interrupted, or the choice.
    fn close(&self, w: &Window) -> TuneState {
        let d60 = self.display_60.unwrap_or(false);
        let (Some((a, aseq, _)), Some(end)) = (w.anchor, w.end) else { return TuneState::Failed(TuneFailure::NothingSeen) };
        // reads of the window by position (time order is position order within one run)
        let reads: Vec<(usize, f64)> = self.samples[a..end].iter().filter_map(|s| s.read.as_ref().map(|r| ((r.seq.wrapping_sub(aseq)) as usize, s.t_ms))).collect();
        // the positions the window was meant to cover: up to the furthest read when it closed on a position, the
        // whole loop when it closed on time or on the next loop
        let covered_to = if w.by_pos { w.max_pos } else { LOOP_FRAMES - 1 };
        let time_of = |p: f64| -> f64 {
            let i = reads.partition_point(|&(q, _)| (q as f64) < p);
            match (i.checked_sub(1).map(|k| reads[k]), reads.get(i).copied()) {
                (Some((p0, t0)), Some((p1, t1))) if p1 > p0 => t0 + (t1 - t0) * (p - p0 as f64) / (p1 - p0) as f64,
                (Some((p0, t0)), _) => {
                    // past the last read: the schedule's time
                    let (mut t, mut q) = (t0, p0);
                    while (q as f64) + 1.0 <= p { t += frame_ms(aseq.wrapping_add(q as u8), d60); q += 1; }
                    t + (p - q as f64) * frame_ms(aseq.wrapping_add(q as u8), d60)
                }
                (None, Some((_, t1))) => t1,
                (None, None) => f64::NAN,
            }
        };
        let near_read = |t: f64, within: f64| t.is_finite() && { let i = reads.partition_point(|&(_, x)| x < t); [i.checked_sub(1), Some(i)].into_iter().flatten().filter_map(|k| reads.get(k)).any(|&(_, x)| (x - t).abs() <= within) };
        // each stretch of the loop in the window (a run of positions of one segment): a tile read near its middle
        let mut p = 0;
        while p <= covered_to {
            let seg = slot(aseq.wrapping_add(p as u8)).segment;
            let mut q = p;
            while q < covered_to && slot(aseq.wrapping_add(q as u8 + 1)).segment == seg { q += 1; }
            if !near_read(time_of((p + q) as f64 / 2.0), GAP_MS) { return TuneState::Failed(TuneFailure::Interrupted); }
            p = q + 1;
        }
        // the data frames each segment showed inside the window while the tile was in view: manifest positions known from
        // a manifest read (every MANIFEST_EVERY_FRAMES frames), else 30 in 31
        let every = MANIFEST_EVERY_FRAMES as usize;
        let phase = self.samples[a..end].iter().find_map(|s| s.read.as_ref().filter(|r| r.manifest).map(|r| r.seq.wrapping_sub(aseq) as usize % every));
        let mut shown = vec![0.0; SEGMENTS.len()];
        for p in 0..=covered_to {
            if !near_read(time_of(p as f64), IN_VIEW_MS) { continue; }
            let data = match phase { Some(m) => if p % every == m { 0.0 } else { 1.0 }, None => (every - 1) as f64 / every as f64 };
            shown[slot(aseq.wrapping_add(p as u8)).segment] += data;
        }
        let mut cal = Calibrator::new();
        cal.set_shown(shown);
        for (s, sl) in self.slotted(w, end) { cal.observe_sample(&s, sl); }
        match cal.result(self.opts, d60) {
            Ok(r) => TuneState::Done(Box::new(r)),
            Err(f) => TuneState::Failed(f),
        }
    }
}

/// One capture's reads into the meter (both receivers' collectors): a tune frame (known by its transfer id,
/// `schedule::tune_display_60`) tells the meter the sender is tuning and how it draws, and the capture's data read
/// (else its manifest read, for where the tile is) becomes the sample; a normal transfer's frame says the sender
/// is not tuning.
pub fn feed_reads(meter: &mut TuneMeter, frs: &[crate::flat::FlatRead], t_ms: f64) {
    use crate::header::FrameType;
    let (mut data, mut manifest) = (None, None);
    for fr in frs {
        match super::schedule::tune_display_60(fr.header.transfer_id) {
            Some(display_60) => {
                meter.sender(Some(TuneInfo { schedule: super::schedule::SCHEDULE_VERSION, display_60 }), false);
                if fr.header.frame_type == FrameType::Data { data.get_or_insert(fr); } else { manifest.get_or_insert(fr); }
            }
            None => meter.sender(None, true),
        }
    }
    if let Some(fr) = data.or(manifest) { meter.capture(TuneSample { t_ms, read: Some(super::sample::SampleRead::of(fr)) }); }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calib::FPS_OPTIONS;
    use crate::flat::{Alphabet, CellFormat};
    use crate::tune::sample::SampleRead;

    fn read(seq: u8, ok: bool) -> SampleRead { read_on(seq, ok, (1024, 1024)) }
    fn read_on(seq: u8, ok: bool, tile: (u32, u32)) -> SampleRead {
        let f = slot(seq).format;
        let (cw, pk) = crate::tune::choose::grid_numbers(tile.0, tile.1, f).unwrap();
        let (cw, pk) = (cw as u32, pk as u32);
        SampleRead { seq, format: f, manifest: false, packets_ok: if ok { pk } else { 0 }, packets: pk, codewords_ok: if ok { cw } else { 0 }, codewords: cw, tile, edges: [1.6; 4],
            corners: [0.0, 0.0, 1600.0, 0.0, 1600.0, 1600.0, 0.0, 1600.0], mix: 0.0, edge_px: 1.5, contrast: 100.0, decode_ms: 5.0 }
    }
    const TUNE: Option<TuneInfo> = Some(TuneInfo { schedule: 1, display_60: true });

    /// A 30 fps camera on the schedule from `start_seq`, `n` captures, `gone(t)`: no tile at that time.
    fn feed(m: &mut TuneMeter, start_seq: u8, t0: f64, n: usize, gone: &dyn Fn(f64) -> bool) -> f64 {
        let (mut seq, mut frame_end) = (start_seq, t0 + frame_ms(start_seq, true));
        let mut t = t0;
        for _ in 0..n {
            while t >= frame_end { seq = seq.wrapping_add(1); frame_end += frame_ms(seq, true); }
            m.sender(TUNE, false);
            m.capture(TuneSample { t_ms: t, read: (!gone(t)).then(|| read(seq, true)) });
            t += 1000.0 / 30.0;
        }
        t
    }

    #[test]
    fn one_loop_from_a_late_start() {
        let mut m = TuneMeter::new(TuneOptions::default(), 0.0);
        assert_eq!(m.state(0.0), TuneState::Waiting);
        let mut last = 0.0;
        let (mut seq, mut frame_end, mut t) = (100u8, 500.0 + frame_ms(100, true), 500.0);
        let mut done_at = None;
        for _ in 0..400 {
            while t >= frame_end { seq = seq.wrapping_add(1); frame_end += frame_ms(seq, true); }
            m.sender(TUNE, false);
            m.capture(TuneSample { t_ms: t, read: Some(read(seq, true)) });
            match m.state(t) {
                TuneState::Measuring(p) => { assert!(p >= last, "{p} < {last}"); last = p; }
                TuneState::Done(_) => { done_at.get_or_insert(t); }
                s => panic!("{s:?}"),
            }
            t += 1000.0 / 30.0;
        }
        let done_at = done_at.expect("done");
        assert!(done_at - 500.0 <= loop_ms(true) + 1000.0, "{done_at}");
        let TuneState::Done(r) = m.state(t) else { panic!() };
        assert!(r.slots.iter().all(|s| s.captures > 0), "{:?}", r.slots);
        assert!(m.status_json(t).starts_with("{\"state\":\"done\",\"progress\":1.000,\"failure\":null,\"result\":{\"words\":"));
    }

    /// A sender's frames from schedule index `n0` (frame number n & 0xFF, a manifest every 31) filmed at 30 fps on
    /// a perfect hold for `ms`.
    fn sender_shots(n0: u64, ms: f64) -> Vec<TuneSample> { sender_shots_on(n0, ms, 1.0, (1024, 1024)) }
    /// `sender_shots` from a sender whose frames stay up `slow` times their schedule's time, on a `tile`.
    fn sender_shots_on(n0: u64, ms: f64, slow: f64, tile: (u32, u32)) -> Vec<TuneSample> {
        let (mut n, mut t_frame, mut out) = (n0, 0.0, Vec::new());
        let mut t = 4.0;
        while t < ms {
            while t >= t_frame + slow * frame_ms((n & 0xFF) as u8, true) { t_frame += slow * frame_ms((n & 0xFF) as u8, true); n += 1; }
            let seq = (n & 0xFF) as u8;
            let mut r = read_on(seq, true, tile);
            if n % MANIFEST_EVERY_FRAMES as u64 == 0 { r = SampleRead { manifest: true, format: crate::flat::MANIFEST_FORMAT, packets_ok: 0, packets: 0, codewords_ok: 0, codewords: 0, ..r }; }
            out.push(TuneSample { t_ms: t, read: Some(r) });
            t += 1000.0 / 30.0;
        }
        out
    }

    #[test]
    fn the_result_does_not_depend_on_where_the_loop_was_joined() {
        let mut first: Option<Box<TuneResult>> = None;
        for anchor in 0..256u64 {
            let mut m = TuneMeter::new(TuneOptions::default(), 0.0);
            for s in sender_shots(anchor + 256 * 3, loop_ms(true) + 1500.0) { m.sender(TUNE, false); m.capture(s); }
            let TuneState::Done(r) = m.state(loop_ms(true) + 1500.0) else { panic!("anchor {anchor}: {:?}", m.state(20_000.0)) };
            for s in &r.slots {
                if FPS_OPTIONS[s.fps as usize] <= 30 { assert!(s.full >= 0.97, "anchor {anchor}: {s:?}"); }
            }
            match &first {
                None => first = Some(r),
                Some(f) => {
                    assert_eq!(r.choice, f.choice, "anchor {anchor}");
                    // a 30 fps camera sees every other frame of a 60 fps stretch: a stretch cut by the window's ends
                    // is off by a frame either way
                    for (a, b) in r.slots.iter().zip(&f.slots) {
                        let tol = if FPS_OPTIONS[a.fps as usize] <= 30 { 0.03 } else { 0.15 };
                        assert!((a.full - b.full).abs() <= tol, "anchor {anchor}: {a:?} vs {b:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn a_slow_sender_still_completes() {
        // every frame shown 25% longer than its schedule (a busy laptop)
        let mut m = TuneMeter::new(TuneOptions::default(), 0.0);
        let ms = loop_ms(true) * 1.25 + 1500.0;
        for s in sender_shots_on(40, ms, 1.25, (1024, 1024)) { m.sender(TUNE, false); m.capture(s); }
        let TuneState::Done(r) = m.state(ms) else { panic!("{:?}", m.state(ms)) };
        assert!(r.slots.iter().all(|s| s.captures > 0), "{:?}", r.slots);
        assert!(r.slots.iter().filter(|s| FPS_OPTIONS[s.fps as usize] <= 30).all(|s| s.full >= 0.97), "{:?}", r.slots);
    }

    #[test]
    fn a_short_tile_tunes() {
        let mut m = TuneMeter::new(TuneOptions::default(), 0.0);
        let ms = loop_ms(true) + 1500.0;
        for s in sender_shots_on(100, ms, 1.0, (1024, 640)) { m.sender(TUNE, false); m.capture(s); }
        let TuneState::Done(r) = m.state(ms) else { panic!("{:?}", m.state(ms)) };
        assert_eq!(r.choice, crate::calib::Config { pitch: 2, alphabet: Alphabet::G8, fps: 0 });
        assert!(r.slots.iter().filter(|s| FPS_OPTIONS[s.fps as usize] <= 30).all(|s| s.full >= 0.97), "{:?}", r.slots);
    }

    #[test]
    fn manifests_without_data_are_unreadable() {
        // the screen is there (tune manifests read) but no data frame ever reads: unreadable, not nothing seen
        let mut m = TuneMeter::new(TuneOptions::default(), 0.0);
        let mut t = 0.0;
        while t < 15_200.0 {
            m.sender(TUNE, false);
            let r = SampleRead { manifest: true, format: crate::flat::MANIFEST_FORMAT, packets_ok: 0, packets: 0, codewords_ok: 0, codewords: 0, ..read(0, true) };
            m.capture(TuneSample { t_ms: t, read: if (t as u64 / 33) % 31 == 0 { Some(r) } else { None } });
            t += 1000.0 / 30.0;
        }
        assert_eq!(m.state(t), TuneState::Failed(TuneFailure::Unreadable));
        let mut m = TuneMeter::new(TuneOptions::default(), 0.0);
        m.abort();
        assert_eq!(m.state(0.0), TuneState::Failed(TuneFailure::Interrupted));
    }

    #[test]
    fn a_gap_mid_loop_is_interrupted() {
        let mut m = TuneMeter::new(TuneOptions::default(), 0.0);
        feed(&mut m, 0, 0.0, 400, &|t| (3000.0..4500.0).contains(&t));
        assert_eq!(m.state(20_000.0), TuneState::Failed(TuneFailure::Interrupted));
        assert!(m.status_json(20_000.0).contains("\"failure\":\"interrupted\""));
    }

    #[test]
    fn a_restart_reanchors() {
        let mut m = TuneMeter::new(TuneOptions::default(), 0.0);
        let t = feed(&mut m, 0, 0.0, 120, &|_| false);
        // the sender starts again: frame numbers from 0
        feed(&mut m, 0, t, 400, &|_| false);
        let TuneState::Done(r) = m.state(30_000.0) else { panic!("{:?}", m.state(30_000.0)) };
        assert!(r.seconds * 1000.0 < loop_ms(true) + 100.0, "{}", r.seconds);
        assert!(r.captures < 270, "{}", r.captures);
    }

    #[test]
    fn nothing_seen_and_not_tuning() {
        let mut m = TuneMeter::new(TuneOptions::default(), 0.0);
        m.capture(TuneSample { t_ms: 10_000.0, read: None });
        assert_eq!(m.state(10_000.0), TuneState::Waiting);
        m.sender(None, true);
        assert_eq!(m.state(16_000.0), TuneState::NotTuning);
        assert!(m.status_json(16_000.0).starts_with("{\"state\":\"not_tuning\""));
        m.sender(TUNE, false);
        assert_eq!(m.state(16_000.0), TuneState::Failed(TuneFailure::NothingSeen));
        let mut m = TuneMeter::new(TuneOptions::default(), 0.0);
        m.capture(TuneSample { t_ms: 15_100.0, read: None });
        assert_eq!(m.state(15_100.0), TuneState::Failed(TuneFailure::NothingSeen));
        assert_eq!(read(0, true).format, CellFormat { pitch: 2, alphabet: Alphabet::Bw });
    }

    #[test]
    fn out_of_order_samples_give_the_same_result() {
        let mut a = TuneMeter::new(TuneOptions::default(), 0.0);
        let mut b = TuneMeter::new(TuneOptions::default(), 0.0);
        let mut caps = Vec::new();
        let (mut seq, mut frame_end, mut t) = (40u8, frame_ms(40, true), 0.0);
        for k in 0..330 {
            while t >= frame_end { seq = seq.wrapping_add(1); frame_end += frame_ms(seq, true); }
            caps.push(TuneSample { t_ms: t, read: (k % 9 != 4).then(|| read(seq, k % 5 != 0)) });
            t += 1000.0 / 30.0;
        }
        for c in &caps { a.sender(TUNE, false); a.capture(c.clone()); }
        // pairs swapped, as two readers hand them back
        for p in caps.chunks(2) { for c in p.iter().rev() { b.sender(TUNE, false); b.capture(c.clone()); } }
        let (TuneState::Done(ra), TuneState::Done(rb)) = (a.state(t), b.state(t)) else { panic!("{:?} {:?}", a.state(t), b.state(t)) };
        assert_eq!(ra.to_json(), rb.to_json());
    }
}
