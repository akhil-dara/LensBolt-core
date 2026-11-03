//! LensBolt v2 "flat cell" format: solid-colour cells of a calibrated pitch inside a fixed 8 px chrome
//! (border, markers, header), read through pooled pilot alignment and per-block colour models.
//! Design: the design notes

pub mod format;
pub mod layout;
pub mod header;
pub mod compose;
pub mod render;
pub mod colour;
pub mod read;
pub mod extract;
pub mod decoder;
pub mod sender;
pub mod stitch;
pub mod live;
pub mod rate;
pub mod fast;
pub mod session;

pub use format::{Alphabet, CellFormat, MANIFEST_FORMAT, MODULE_PX, PITCHES};
pub use layout::{Chrome, FlatCell, FlatGrid};
pub use header::{FlatHeader, HeaderCopy, FLAT_HEADER_BYTES, FLAT_HEADER_RS_PARITY, FLAT_VERSION};
pub use compose::{compose, cells_to_stripe, frame_stream, stripe_cells, FlatFrame};
pub use render::{render, render_rgba, render_to};
pub use colour::ColourModel;
pub use read::{read_cells, read_header, read_header_bits, sampling, FlatReads, ReadParams};
pub use extract::{extract_cells, extract_payload, FramePayload, ERASURE_MARGIN};
pub use decoder::{DecodeError, FlatDecoder, FlatHint, FlatRead, FRAME_SIZES};
pub use sender::{FlatSender, FlatSenderConfig, CAROUSEL, MANIFEST_EVERY_FRAMES};
pub use stitch::Stitcher;
pub use rate::RateEstimate;
