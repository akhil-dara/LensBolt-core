//! Calibration words: the receiver measures the sender's opening carousel and shows two checksummed
//! words that pick the data format and the display frame rate; the user types them into the sender.
//!
//! The words come from the finish-word list (`words::WORDS`, one kind of word code in the product) and are
//! typed the same way (whole words, any case, separated by spaces, dashes or commas; a single mistake still
//! reads as the word meant). A code is two words: `WORDS[CALIB_FIRST + config]`, then the 10-bit check of the
//! config. The first word is always from the list's last quarter, whose 10 bits begin with the finish code's tag
//! 3 (window): a window code is always four words, so a calibration code is never finish words for any file,
//! and a finish code of fewer than four words never starts with a calibration word. The two fields of the
//! sender can never take each other's words.
use crate::flat::{Alphabet, CellFormat, FlatGrid, PITCHES};
use crate::flat::FramePayload;
use crate::words::{word_index, WordError, WORDS};
use crate::tune::schedule::Slot;
use crate::tune::{TuneFailure, TuneOptions, TuneResult, TuneSample};
use std::collections::HashMap;

/// What a code carries: the data cell format (pitch and alphabet) and the display frame rate.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Config { pub pitch: u8, pub alphabet: Alphabet, pub fps: u8 }

/// Display frame rates a code can ask for (index = 2 bits). Index 2 is 20 fps: on a 60 Hz screen each frame
/// stays up for 3 refreshes, so a panel slow to change leaves a clean window of about 2/3 of the period (24 was
/// never offered by a sender or suggested by a receiver).
pub const FPS_OPTIONS: [u8; 4] = [30, 60, 20, 120];

/// The index in `FPS_OPTIONS` of a frame rate, None for a rate a code cannot carry.
pub fn fps_index(fps: u32) -> Option<u8> { FPS_OPTIONS.iter().position(|&f| f as u32 == fps).map(|i| i as u8) }

/// The sender page's named speeds and their words (all at 30 fps): Reliable 3 px, Fast 2 px, Fastest 1 px, b/w.
pub const PRESETS: [(&str, &str); 3] = [("Reliable", "sauce-minivan"), ("Fast", "satin-swift"), ("Fastest", "sardine-berry")];

/// The sender's default: Fast, 2 px black and white at 30 fps.
pub const FAST: Config = Config { pitch: 2, alphabet: Alphabet::Bw, fps: 0 };

/// An alphabet in words, as the pages name it.
fn alphabet_words(a: Alphabet) -> &'static str {
    match a { Alphabet::Bw => "black and white", Alphabet::G4 => "4 greys", Alphabet::G8 => "8 greys", Alphabet::C4 => "4 colours", Alphabet::C8 => "8 colours", Alphabet::C16 => "16 colours" }
}

/// A configuration in words for the screens: the preset whose words name the same pitch and alphabet ("Fast"),
/// else "3 px · 8 greys"; then the rate, unless it is 30 fps: ", 20 fps (3 refreshes per frame)", ", 60 fps".
pub fn mode_words(c: &Config) -> String {
    let preset = PRESETS.iter().find(|(_, w)| Config::from_code(w).map(|p| p.format() == c.format()).unwrap_or(false));
    let name = match preset { Some((n, _)) => n.to_string(), None => format!("{} px · {}", c.pitch, alphabet_words(c.alphabet)) };
    match c.fps_value() {
        30 => name,
        20 => format!("{name}, 20 fps (3 refreshes per frame)"),
        f => format!("{name}, {f} fps"),
    }
}

/// Why typed calibration words are refused. `WrongShape`: not two words. `Word`: a word not on the list, or
/// close to two. `NotCalibration`: the first word cannot start a calibration code (finish words, say), or the
/// two words name a byte that is no configuration.
/// `Checksum`: the second word does not match the first.
#[derive(Debug, PartialEq, Eq)]
pub enum CodeError { WrongShape, Word(WordError), NotCalibration, Checksum }

/// The first word's index is `CALIB_FIRST` + the configuration byte: the list's last quarter (top bits 11, the
/// finish code's window tag).
pub const CALIB_FIRST: usize = 768;

impl Config {
    pub fn fps_value(&self) -> u32 { FPS_OPTIONS[(self.fps as usize).min(FPS_OPTIONS.len() - 1)] as u32 }
    pub fn format(&self) -> CellFormat { CellFormat { pitch: self.pitch, alphabet: self.alphabet } }
    /// 8 bits: pitch index (3), alphabet id (3), fps index (2).
    fn pack(&self) -> u8 {
        let pi = self.format().pitch_index().unwrap_or(1);
        (pi & 7) | ((self.alphabet.id() & 7) << 3) | ((self.fps & 3) << 6)
    }
    fn unpack(b: u8) -> Config {
        let f = CellFormat::from_indices(b & 7, (b >> 3) & 7).unwrap_or(CellFormat { pitch: PITCHES[1], alphabet: Alphabet::C8 });
        Config { pitch: f.pitch, alphabet: f.alphabet, fps: (b >> 6) & 3 }
    }
    /// Two words, joined by a dash: the configuration's word, then its check word.
    pub fn to_code(&self) -> String {
        let b = self.pack();
        format!("{}-{}", WORDS[CALIB_FIRST + b as usize], WORDS[check(b) as usize])
    }
    /// Two words separated by spaces, dashes or commas, any case, each read by `words::word_index` (words run
    /// together are refused, as finish words are).
    pub fn from_code(code: &str) -> Result<Config, CodeError> {
        let typed: Vec<&str> = code.split(|c: char| c.is_whitespace() || c == '-' || c == ',').filter(|w| !w.is_empty()).collect();
        let [w1, w2] = typed[..] else { return Err(CodeError::WrongShape) };
        let a = word_index(w1).map_err(CodeError::Word)?;
        let c = word_index(w2).map_err(CodeError::Word)?;
        from_indices(a as usize, c as usize)
    }
}

/// The configuration two list words name, by their indices. A byte that names no configuration (an alphabet id
/// past the last alphabet) is refused, never read as another configuration.
fn from_indices(a: usize, c: usize) -> Result<Config, CodeError> {
    let b = a.checked_sub(CALIB_FIRST).filter(|&b| b < 256).ok_or(CodeError::NotCalibration)? as u8;
    if c != check(b) as usize { return Err(CodeError::Checksum); }
    let config = Config::unpack(b);
    if config.pack() != b { return Err(CodeError::NotCalibration); }
    Ok(config)
}

/// The check word's index: the low 10 bits of CRC-16 over the configuration byte and 0x4C. Distinct for all 256
/// bytes, so a wrong word is always caught, and swapping the two words is too (both tested).
fn check(b: u8) -> u16 { crate::crc16::crc16(&[b, 0x4C]) & 0x3ff }

/// What the receiver measured for one format of the opening carousel.
#[derive(Clone, Copy, Debug, Default)]
pub struct FormatStat { pub frames: u32, pub codewords_ok: u64, pub codewords_total: u64, pub decode_ms: f64 }

#[derive(Debug, Clone, PartialEq)]
pub struct Suggestion { pub config: Config, pub expected_mbit: f64, pub yields: Vec<(CellFormat, f64)> }

/// Accumulates per-format results over the calibration captures and chooses a configuration: by format alone
/// (`observe`, `suggest`: the lab carousel and the CLI), or from Tune samples with the hold (`observe_sample`,
/// `result`: `tune::choose`).
#[derive(Default)]
pub struct Calibrator { stats: HashMap<CellFormat, FormatStat>, captures: u32, samples: Vec<(TuneSample, Slot)>, shown: Option<Vec<f64>> }

impl Calibrator {
    pub fn new() -> Calibrator { Calibrator::default() }
    pub fn captures(&self) -> u32 { self.captures }
    /// One capture went by (decoded or not).
    pub fn capture(&mut self) { self.captures += 1; }
    /// A frame decoded from a capture: its format, its codeword outcome, the decode time.
    pub fn observe(&mut self, format: CellFormat, fp: &FramePayload, decode_ms: f64) {
        let s = self.stats.entry(format).or_default();
        s.frames += 1; s.codewords_ok += fp.codewords_ok as u64; s.codewords_total += (fp.codewords_ok + fp.codewords_failed) as u64;
        s.decode_ms += decode_ms;
    }
    /// One Tune capture, with the carousel slot the meter gave it (a read's frame number names it; a capture that
    /// read nothing takes the slot on screen at its time).
    pub fn observe_sample(&mut self, s: &TuneSample, slot: Slot) { self.samples.push((s.clone(), slot)); }
    /// Tune samples observed so far.
    pub fn samples(&self) -> usize { self.samples.len() }
    /// The Tune choice over the samples (`tune::choose::evaluate`). `display_60`: the sender drew the 60 fps
    /// segments at 60.
    pub fn result(&self, opts: TuneOptions, display_60: bool) -> Result<TuneResult, TuneFailure> {
        let shown = self.shown.clone().unwrap_or_else(crate::tune::choose::loop_shown);
        crate::tune::choose::evaluate(&self.samples, opts, display_60, &shown)
    }
    /// Per segment, the data frames the measured window showed while the tile was in view (the meter's count; a whole
    /// loop's when not given).
    pub fn set_shown(&mut self, shown: Vec<f64>) { self.shown = Some(shown); }
    /// Choose for a tile of `width` x `height`, a camera running at `camera_fps`, and a receiver that
    /// can spend `cores` cores on decoding. Yield of a format = codewords decoded per codeword shown
    /// (a capture never identified yields nothing). None until something was observed.
    pub fn suggest(&self, width: u32, height: u32, camera_fps: f64, cores: f64) -> Option<Suggestion> {
        if self.stats.is_empty() || self.captures == 0 { return None; }
        let kinds = self.stats.len() as f64;
        let shown = (self.captures as f64 / kinds).max(1.0);
        let mut yields = Vec::new();
        let mut best: Option<Suggestion> = None;
        for (&format, s) in &self.stats {
            let Some(grid) = FlatGrid::try_new(width, height, format) else { continue };
            let y = (s.codewords_ok as f64 / (shown * grid.codewords() as f64)).min(1.0);
            yields.push((format, y));
            let ms = if s.frames > 0 { s.decode_ms / s.frames as f64 } else { 0.0 };
            let capacity = if ms > 0.0 { 1000.0 * cores / ms } else { camera_fps };
            let limit = camera_fps.min(capacity);
            // the highest frame rate at or below what the camera and the decoder sustain
            let fps_i = (0..FPS_OPTIONS.len()).filter(|&i| FPS_OPTIONS[i] as f64 <= limit + 0.5).max_by_key(|&i| FPS_OPTIONS[i]).unwrap_or(2);
            let f = (FPS_OPTIONS[fps_i] as f64).min(limit);
            let mbit = y * grid.payload_bytes() as f64 * 8.0 * f / 1e6;
            if best.as_ref().map(|b| mbit > b.expected_mbit).unwrap_or(true) {
                best = Some(Suggestion { config: Config { pitch: format.pitch, alphabet: format.alphabet, fps: fps_i as u8 }, expected_mbit: mbit, yields: Vec::new() });
            }
        }
        yields.sort_by(|a, b| (a.0.pitch, a.0.alphabet.id()).cmp(&(b.0.pitch, b.0.alphabet.id())));
        best.map(|mut b| { b.yields = yields; b })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flat::{Alphabet, CellFormat, FlatGrid, PITCHES};
    use crate::words::{WordError, WORDS};
    // every configuration a code can carry (the 8-bit value: pitch index, alphabet id, fps index)
    fn configs() -> Vec<Config> {
        let mut v = Vec::new();
        for p in PITCHES { for a in Alphabet::ALL { for fps in 0..FPS_OPTIONS.len() as u8 { v.push(Config { pitch: p, alphabet: a, fps }); } } }
        v
    }
    fn split(code: &str) -> (&str, &str) { code.split_once('-').unwrap() }
    fn index(w: &str) -> usize { WORDS.iter().position(|&x| x == w).unwrap() }
    #[test]
    fn codes_are_two_words_of_the_finish_list() {
        for c in configs() {
            let code = c.to_code();
            let (w1, w2) = split(&code);
            // both words are list words; the first from the list's last quarter (the finish code's tag 3)
            assert!(index(w1) >= CALIB_FIRST, "{code}");
            assert!(WORDS.contains(&w2), "{code}");
        }
    }
    #[test]
    fn code_round_trips_every_format() {
        let mut seen = std::collections::HashSet::new();
        for c in configs() {
            let code = c.to_code();
            assert_eq!(Config::from_code(&code), Ok(c), "{code}");
            assert_eq!(Config::from_code(&code.to_uppercase().replace('-', " ")), Ok(c));
            assert_eq!(Config::from_code(&format!("  {}, {} ", split(&code).0, split(&code).1)), Ok(c));
            // one mistake in a word (a letter dropped) still reads as that word, as finish words do
            let (w1, w2) = split(&code);
            match Config::from_code(&format!("{}-{}", &w1[..w1.len() - 1], w2)) {
                Ok(x) => assert_eq!(x, c, "{code}"),
                Err(CodeError::Word(WordError::Ambiguous(_))) => {}   // close to two words: asked again, never guessed
                e => panic!("{code}: {e:?}"),
            }
            // words run together are not split (as finish words: splitting cannot keep zero false positives)
            assert_eq!(Config::from_code(&code.replace('-', "")), Err(CodeError::WrongShape), "joined {code}");
            assert!(seen.insert(code));
        }
        assert_eq!(seen.len(), PITCHES.len() * Alphabet::ALL.len() * FPS_OPTIONS.len());
        assert_eq!(Config { pitch: 4, alphabet: Alphabet::C8, fps: 0 }.fps_value(), 30);
        assert_eq!(Config { pitch: 4, alphabet: Alphabet::C8, fps: 0 }.format(), CellFormat { pitch: 4, alphabet: Alphabet::C8 });
    }
    #[test]
    fn any_wrong_or_swapped_word_is_refused() {
        // every configuration value, both words replaced by every other list word: never accepted
        for b in 0..=255u8 {
            let (a, c) = (CALIB_FIRST + b as usize, check(b) as usize);
            if Config::unpack(b).pack() == b { assert_eq!(from_indices(a, c).map(|x| x.pack()), Ok(b)); }
            for o in 0..WORDS.len() {
                if o != a { assert!(from_indices(o, c).is_err(), "first word {} for {b}", WORDS[o]); }
                if o != c { assert!(from_indices(a, o).is_err(), "second word {} for {b}", WORDS[o]); }
            }
            assert!(from_indices(c, a).is_err(), "swapped {b}");
        }
        let code = Config { pitch: 5, alphabet: Alphabet::C16, fps: 1 }.to_code();
        let (w1, w2) = split(&code);
        let other = WORDS[CALIB_FIRST..].iter().find(|&&w| w != w1 && w != w2).unwrap();
        assert_eq!(Config::from_code(&format!("{other}-{w2}")), Err(CodeError::Checksum));
        assert_eq!(Config::from_code(&format!("{w1}-{other}")), Err(CodeError::Checksum));
        assert_eq!(Config::from_code(&format!("{}-{w2}", WORDS[0])), Err(CodeError::NotCalibration));
        assert_eq!(Config::from_code(&format!("{w1}-xylophone")), Err(CodeError::Word(WordError::Unknown("xylophone".into()))));
        assert_eq!(Config::from_code(w1), Err(CodeError::WrongShape));
        assert_eq!(Config::from_code(&format!("{w1} {w2} {w2}")), Err(CodeError::WrongShape));
    }
    #[test]
    fn only_bytes_that_name_a_configuration_are_accepted() {
        // 8 pitches x 6 alphabets x 4 frame rates of the 256 bytes name a configuration; the others (alphabet ids 6
        // and 7) once read as another configuration's fallback: refused, with their check word right
        let mut ok = 0;
        for b in 0..=255u8 {
            let got = from_indices(CALIB_FIRST + b as usize, check(b) as usize);
            if Config::unpack(b).pack() == b { assert_eq!(got.map(|c| c.pack()), Ok(b)); ok += 1; }
            else { assert_eq!(got, Err(CodeError::NotCalibration), "byte {b:#04x}"); }
        }
        assert_eq!(ok, PITCHES.len() * Alphabet::ALL.len() * FPS_OPTIONS.len());
    }
    #[test]
    fn calibration_words_are_never_finish_words() {
        // a calibration code starts with tag 3 (window), whose code is always four words: two words of it are
        // malformed for every file, whatever its transfer id (the check is never reached)
        let ns: Vec<u32> = (1..=64).chain([100, 320, 2000, 4096, 65_535]).collect();
        for c in configs() {
            let code = c.to_code();
            for &n in &ns { for id in [0x0001u16, 0x4c5b, 0x8001, 0xffff] {
                assert_eq!(crate::finish::decode(id, n, &code), Err(crate::finish::FinishError::Malformed), "{code} n {n} id {id:#x}");
            } }
        }
    }
    #[test]
    fn finish_words_are_never_calibration_words() {
        // every finish code of every non-trivial loss pattern of files of up to 12 parts, and random patterns of
        // larger files, for several transfers: the Speed field refuses each one, and none starts like a
        // calibration code unless it is four words long
        let mut x = 0x9e3779b97f4a7c15u64;
        let mut rnd = move || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x };
        let mut checked = 0;
        for id in [0x0001u16, 0x4c5b, 0xa5a5 | 1, 0xffff] {
            let mut try_code = |n: u32, short: &[u32]| {
                let Some(code) = crate::finish::encode(id, n, short) else { return };
                let words: Vec<&str> = code.split(' ').collect();
                if words.len() < 4 { assert!(index(words[0]) < CALIB_FIRST, "{code}"); }
                assert!(Config::from_code(&code).is_err(), "finish words {code} (n {n}) read as calibration words");
                checked += 1;
            };
            for n in 1..=12u32 { for m in 1..(1u32 << n) - 1 { try_code(n, &(0..n).filter(|i| m >> i & 1 == 1).collect::<Vec<_>>()); } }
            for n in [16u32, 32, 320, 2000, 65_535] {
                for _ in 0..400 {
                    let k = 1 + rnd() % 20;
                    let s: Vec<u32> = (0..k).map(|_| (rnd() % n as u64) as u32).collect();
                    try_code(n, &s);
                }
            }
        }
        assert!(checked > 30_000, "{checked}");
    }
    #[test]
    fn fps_index_2_is_twenty() {
        assert_eq!(Config { pitch: 2, alphabet: Alphabet::Bw, fps: 2 }.fps_value(), 20);
        assert_eq!(fps_index(20), Some(2)); assert_eq!(fps_index(60), Some(1)); assert_eq!(fps_index(24), None);
        assert_eq!(fps_index(30), Some(0)); assert_eq!(fps_index(120), Some(3));
    }
    #[test]
    fn presets_are_the_pages_speeds() {
        let f = |w| Config::from_code(w).map(|c| (c.pitch, c.alphabet, c.fps_value()));
        assert_eq!(f("sauce-minivan"), Ok((3, Alphabet::Bw, 30)));
        assert_eq!(f("satin-swift"), Ok((2, Alphabet::Bw, 30)));
        assert_eq!(f("sardine-berry"), Ok((1, Alphabet::Bw, 30)));
        assert_eq!(FAST.to_code(), "satin-swift");
        for (name, words) in PRESETS { assert_eq!(mode_words(&Config::from_code(words).unwrap()), name); }
    }
    #[test]
    fn mode_words_name_presets_and_rates() {
        assert_eq!(mode_words(&FAST), "Fast");
        assert_eq!(mode_words(&Config { fps: 1, ..FAST }), "Fast, 60 fps");
        assert_eq!(mode_words(&Config { pitch: 3, alphabet: Alphabet::Bw, fps: 2 }), "Reliable, 20 fps (3 refreshes per frame)");
        assert_eq!(mode_words(&Config { pitch: 2, alphabet: Alphabet::G8, fps: 0 }), "2 px · 8 greys");
        assert_eq!(mode_words(&Config { pitch: 4, alphabet: Alphabet::C8, fps: 1 }), "4 px · 8 colours, 60 fps");
        assert_eq!(mode_words(&Config { pitch: 5, alphabet: Alphabet::C16, fps: 3 }), "5 px · 16 colours, 120 fps");
    }
    #[test]
    fn three_words_starting_with_a_calibration_word_are_neither() {
        // the rule reserved for a later longer code: told apart by its word count
        for c in configs() { for third in [WORDS[0], WORDS[500], WORDS[900]] {
            let code = format!("{} {third}", c.to_code().replace('-', " "));
            assert_eq!(Config::from_code(&code), Err(CodeError::WrongShape), "{code}");
            for n in [1u32, 12, 320, 65_535] { for id in [0x0001u16, 0x4c5b, 0xffff] {
                assert_eq!(crate::finish::decode(id, n, &code), Err(crate::finish::FinishError::Malformed), "{code}");
            } }
        } }
    }
    fn fp(ok: usize, failed: usize) -> FramePayload { FramePayload { bytes: Vec::new(), valid: Vec::new(), codewords_ok: ok, codewords_failed: failed, codewords_missing: 0, coverage: 1.0 } }
    #[test]
    fn suggests_the_best_measured_format() {
        let c8 = CellFormat { pitch: 4, alphabet: Alphabet::C8 };
        let c16 = CellFormat { pitch: 3, alphabet: Alphabet::C16 };
        let (g8, g16) = (FlatGrid::try_new(1024, 1024, c8).unwrap(), FlatGrid::try_new(1024, 1024, c16).unwrap());
        let mut cal = Calibrator::new();
        for _ in 0..6 {
            cal.capture(); cal.observe(c8, &fp(g8.codewords() * 9 / 10, g8.codewords() / 10), 20.0);
            cal.capture(); cal.observe(c16, &fp(g16.codewords() * 3 / 10, g16.codewords() * 7 / 10), 25.0); // 3 px carries 2.3x the bytes: 30% of it loses to 90% of 4 px
        }
        assert!(cal.suggest(1024, 1024, 30.0, 4.0).is_none() || cal.captures() >= 4);
        let s = cal.suggest(1024, 1024, 30.0, 4.0).unwrap();
        assert_eq!(s.config.format(), c8, "{s:?}");
        assert_eq!(s.config.fps_value(), 30);
        let expect = 0.9 * 30.0 * g8.payload_bytes() as f64 * 8.0 / 1e6;
        assert!((s.expected_mbit - expect).abs() < 0.05 * expect, "{} vs {expect}", s.expected_mbit);
        assert!(s.yields.iter().any(|&(f, y)| f == c8 && (y - 0.9).abs() < 0.02));
        // a slow decoder caps the frame rate: 1 core at 20 ms per capture cannot keep up with 60 fps
        let s = cal.suggest(1024, 1024, 60.0, 1.0).unwrap();
        assert_eq!(s.config.fps_value(), 30, "{s:?}");
        // a format never observed is never suggested; an empty calibrator suggests nothing
        assert!(Calibrator::new().suggest(1024, 1024, 30.0, 4.0).is_none());
    }
}
