//! Tune: the receiver measures the sender's Tune carousel from where it is held and names, in two words, the
//! fastest format that keeps reading from that hold.
//!
//! - `schedule`: the carousel, a fixed 256-frame loop keyed by the frame number every header carries, and the
//!   generated payload that drives it.
//! - `sample`: what one capture showed (`SampleRead`, `TuneSample`).
//! - `choose`: per-segment yields with the drift discount, the choice, the hold advice (`TuneResult`).
//! - `meter`: one loop from any start, in both receivers' collectors (`TuneMeter`).
//! - `hint`: during a normal transfer, safer b/w words or Tune when reading poorly (`HintWatch`, `LiveHint`).
pub mod schedule;
pub mod sample;
pub mod choose;
pub mod meter;
pub mod hint;
pub use choose::{Advice, HoldStats, Sixty, SlotStat, TuneFailure, TuneOptions, TuneResult};
pub use sample::{SampleRead, TuneSample};
pub use meter::{TuneMeter, TuneState};
pub use hint::{HintSample, HintWatch, LiveHint};
