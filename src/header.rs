//! What every frame header carries for the transport, and the frame kinds.

pub const HEADER_MAGIC: u8 = 0xA;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FrameType { Data = 0, Manifest = 1, Probe = 2 }

#[derive(Debug, PartialEq, Eq)]
pub enum HeaderError { BadMagic, BadVersion, BadCrc, BadMode, BadType }

/// What the transport needs from a frame header.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FrameInfo { pub frame_type: FrameType, pub transfer_id: u16, pub segment: u16, pub block_id: u32 }

/// Anything the receiver can be fed from: a flat header or a bare FrameInfo.
pub trait FrameMeta { fn info(&self) -> FrameInfo; }
impl FrameMeta for FrameInfo { fn info(&self) -> FrameInfo { *self } }
impl FrameMeta for crate::flat::FlatHeader { fn info(&self) -> FrameInfo { crate::flat::FlatHeader::info(self) } }
