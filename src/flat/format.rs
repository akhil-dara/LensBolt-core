//! Cell formats: the data pitch and the colour alphabet a transfer uses.

/// Side of a chrome module (border, marker and header cells) in screen px, whatever the data pitch.
pub const MODULE_PX: u32 = 8;
/// Data cell pitches a sender may use, in screen px (index = 3 bits in the header and the calibration
/// words). 1 px b/w is the 20x configuration for a phone held close (~3 camera px per screen px); 12 px
/// the floor for a far or low-resolution camera.
pub const PITCHES: [u8; 8] = [1, 2, 3, 4, 5, 6, 8, 12];

/// Measured on real cameras (docs/2026-10-27-*): the alphabets that separate. Never hue x luma in one cell.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Alphabet { Bw, G4, C4, C8, C16, G8 }

const BW: [[u8; 3]; 2] = [[0, 0, 0], [255, 255, 255]];
const GREY4: [[u8; 3]; 4] = [[0, 0, 0], [85, 85, 85], [170, 170, 170], [255, 255, 255]];
/// 8 greys: luma only, 3 bits; the smallest cells a 4:3 sensor can read carry it when chroma is gone.
const GREY8: [[u8; 3]; 8] = [[0, 0, 0], [36, 36, 36], [73, 73, 73], [109, 109, 109], [146, 146, 146], [182, 182, 182], [219, 219, 219], [255, 255, 255]];
const PAL4: [[u8; 3]; 4] = [[255, 255, 255], [0, 255, 255], [255, 0, 255], [255, 255, 0]];
const PAL8: [[u8; 3]; 8] = [[255, 255, 255], [0, 255, 255], [255, 0, 255], [255, 255, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 128, 0]];
/// The 8 colours and their half-intensity versions (measured 0.6% at 5 px on the phone).
const PAL16: [[u8; 3]; 16] = [
    [255, 255, 255], [0, 255, 255], [255, 0, 255], [255, 255, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 128, 0],
    [127, 127, 127], [0, 127, 127], [127, 0, 127], [127, 127, 0], [127, 0, 0], [0, 127, 0], [0, 0, 127], [127, 64, 0],
];

impl Alphabet {
    pub const ALL: [Alphabet; 6] = [Alphabet::Bw, Alphabet::G4, Alphabet::C4, Alphabet::C8, Alphabet::C16, Alphabet::G8];
    pub fn bits(self) -> u32 { match self { Alphabet::Bw => 1, Alphabet::G4 | Alphabet::C4 => 2, Alphabet::C8 | Alphabet::G8 => 3, Alphabet::C16 => 4 } }
    pub fn size(self) -> usize { 1 << self.bits() }
    /// RS parity per 255-byte codeword: more colours, more parity.
    pub fn parity(self) -> usize { match self { Alphabet::Bw | Alphabet::G4 | Alphabet::C4 => 24, Alphabet::C8 | Alphabet::G8 => 32, Alphabet::C16 => 40 } }
    pub fn data_len(self) -> usize { 255 - self.parity() }
    pub fn palette(self) -> &'static [[u8; 3]] {
        match self { Alphabet::Bw => &BW, Alphabet::G4 => &GREY4, Alphabet::C4 => &PAL4, Alphabet::C8 => &PAL8, Alphabet::C16 => &PAL16, Alphabet::G8 => &GREY8 }
    }
    pub fn id(self) -> u8 { match self { Alphabet::Bw => 0, Alphabet::G4 => 1, Alphabet::C4 => 2, Alphabet::C8 => 3, Alphabet::C16 => 4, Alphabet::G8 => 5 } }
    pub fn from_id(id: u8) -> Option<Alphabet> { Alphabet::ALL.get(id as usize).copied() }
    /// Luma-only alphabets are read from the grey level alone (a webcam reads them at far smaller cells).
    pub fn is_luma(self) -> bool { matches!(self, Alphabet::Bw | Alphabet::G4 | Alphabet::G8) }
    pub fn name(self) -> &'static str { match self { Alphabet::Bw => "bw", Alphabet::G4 => "g4", Alphabet::C4 => "c4", Alphabet::C8 => "c8", Alphabet::C16 => "c16", Alphabet::G8 => "g8" } }
    pub fn from_name(s: &str) -> Option<Alphabet> { Alphabet::ALL.iter().copied().find(|a| a.name() == s) }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct CellFormat { pub pitch: u8, pub alphabet: Alphabet }

impl CellFormat {
    /// RS parity bytes per 255-byte codeword: the alphabet's, except 1 px frames, whose b/w bytes span 8
    /// cells each (1.5% cell error is ~11% byte error) and get 64 (32 correctable bytes).
    pub fn parity(self) -> usize { if self.pitch == 1 { 64 } else { self.alphabet.parity() } }
    pub fn data_len(self) -> usize { 255 - self.parity() }
    pub fn pitch_index(self) -> Option<u8> { PITCHES.iter().position(|&p| p == self.pitch).map(|i| i as u8) }
    pub fn from_indices(pitch_index: u8, alphabet_id: u8) -> Option<CellFormat> {
        Some(CellFormat { pitch: *PITCHES.get(pitch_index as usize)?, alphabet: Alphabet::from_id(alphabet_id)? })
    }
}

/// Manifest frames: the coarsest configuration that still holds one packet in a 1024x768 tile (12 px / 4
/// colours does not); 12 px / 8 colours read at 1.1% on a webcam.
pub const MANIFEST_FORMAT: CellFormat = CellFormat { pitch: 12, alphabet: Alphabet::C8 };

/// The manifest format for a tile: 12 px / 8 colours where it fits (every tile >= 1024 x 768), else the largest
/// of 8 and 6 px that does (the small camera-matched tiles). The header names the format, so a receiver reads
/// any of them.
pub fn manifest_format_for(width_px: u32, height_px: u32) -> Option<CellFormat> {
    [12u8, 8, 6].iter().map(|&pitch| CellFormat { pitch, alphabet: Alphabet::C8 }).find(|&f| super::layout::FlatGrid::try_new(width_px, height_px, f).is_some())
}

#[cfg(test)]
mod tests {
    #[test]
    fn manifest_format_fits_every_tile() {
        assert_eq!(super::manifest_format_for(1024, 1024), Some(MANIFEST_FORMAT));
        let small = super::manifest_format_for(784, 440).expect("a manifest format for the camera-matched tile");
        assert_eq!(small.alphabet, Alphabet::C8);
        assert!(small.pitch < 12 && crate::flat::FlatGrid::try_new(784, 440, small).is_some());
    }
    use super::*;
    #[test]
    fn alphabet_numbers() {
        assert_eq!(Alphabet::C16.bits(), 4); assert_eq!(Alphabet::C8.bits(), 3); assert_eq!(Alphabet::Bw.bits(), 1);
        assert_eq!(Alphabet::C16.parity(), 40); assert_eq!(Alphabet::C8.parity(), 32); assert_eq!(Alphabet::G4.parity(), 24);
        assert_eq!(Alphabet::C16.palette().len(), 16); assert_eq!(Alphabet::C16.palette()[8], [127, 127, 127]);
        assert_eq!(Alphabet::C8.palette()[7], [255, 128, 0]); assert_eq!(Alphabet::G4.palette()[1], [85, 85, 85]);
        for a in [Alphabet::Bw, Alphabet::G4, Alphabet::C4, Alphabet::C8, Alphabet::C16] {
            assert_eq!(Alphabet::from_id(a.id()), Some(a));
            assert_eq!(a.palette().len(), a.size());
            assert_eq!(a.data_len(), 255 - a.parity());
            assert_eq!(Alphabet::from_name(a.name()), Some(a));
        }
        assert_eq!(Alphabet::from_id(6), None);
        assert!(Alphabet::Bw.is_luma() && Alphabet::G4.is_luma() && Alphabet::G8.is_luma() && !Alphabet::C4.is_luma());
        // 8 greys: 3 luma bits per cell, no chroma to blur -- the 2 px configuration a 4:3 sensor can read
        assert_eq!(Alphabet::G8.bits(), 3); assert_eq!(Alphabet::G8.parity(), 32); assert_eq!(Alphabet::G8.id(), 5);
        assert_eq!(Alphabet::G8.palette()[1], [36, 36, 36]); assert_eq!(Alphabet::G8.palette()[7], [255, 255, 255]);
    }
    #[test]
    fn stripe_sizes() {
        assert_eq!(CellFormat { pitch: 12, alphabet: Alphabet::C4 }.pitch_index(), Some(7));
        assert_eq!(CellFormat { pitch: 7, alphabet: Alphabet::C4 }.pitch_index(), None);
        assert_eq!(CellFormat::from_indices(3, 3), Some(CellFormat { pitch: 4, alphabet: Alphabet::C8 }));
        assert_eq!(CellFormat::from_indices(1, 4), Some(CellFormat { pitch: 2, alphabet: Alphabet::C16 }));
        assert_eq!(CellFormat::from_indices(0, 0), Some(CellFormat { pitch: 1, alphabet: Alphabet::Bw }));
        assert_eq!(CellFormat::from_indices(8, 3), None);
        assert_eq!(MANIFEST_FORMAT, CellFormat { pitch: 12, alphabet: Alphabet::C8 });
        assert_eq!(PITCHES, [1, 2, 3, 4, 5, 6, 8, 12]); assert_eq!(MODULE_PX, 8);
    }
    #[test]
    fn one_px_frames_carry_more_parity() {
        // a b/w byte spans 8 cells: 1.5% cell error is ~11% byte error, beyond 24 parity bytes; 1 px frames
        // use 64 (corrects 32 bytes)
        let one = CellFormat { pitch: 1, alphabet: Alphabet::Bw };
        let two = CellFormat { pitch: 2, alphabet: Alphabet::Bw };
        assert_eq!(one.parity(), 64); assert_eq!(one.data_len(), 191);
        assert_eq!(two.parity(), 24); assert_eq!(two.data_len(), 231);
    }
}
