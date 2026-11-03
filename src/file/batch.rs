//! The batch file index: the first bytes of a batch stream (`index || file_1 || file_2 || ...`), naming every
//! file, its sizes, where its stored bytes sit and its SHA-256.
//!
//! Layout, version 1 (varint = unsigned LEB128, canonical: no redundant trailing zero groups, at most 64 bits):
//!
//! | field | bytes |
//! |---|---|
//! | magic `LBIX` | 4 |
//! | version (1) | 1 |
//! | entry count, 1 ..= 1,000,000 | varint |
//! | per entry, in stream order: | |
//! | - name length, 1 ..= 255 | varint |
//! | - name, UTF-8 | name length |
//! | - flags: bit 0 = compressed (zstd); the other bits 0 | 1 |
//! | - size (original bytes) | varint |
//! | - stored size (bytes in the stream), only when compressed | varint |
//! | - SHA-256 of the original bytes | 32 |
//!
//! Nothing follows the last entry. Offsets are not sent: the files sit back to back in index order right after
//! the index, so a file's offset is the index's length on the wire + the stored sizes before it, and overlaps,
//! duplicate offsets and gaps cannot be written down. The index carries no checksum of its own: the manifest
//! carries its SHA-256 and the receiver decodes it only once the segments holding it verified. A raw entry takes
//! its name + 35 bytes (sizes and name lengths under 128; a byte more per further 7 bits), a compressed one also
//! its stored size's varint. An index is at most `MAX_INDEX_BYTES` (64 MiB): a million entries fit when their
//! names average 30 bytes or less.
//!
//! Decoding takes the whole index as one slice (bounded, not streaming): the manifest hashes the whole index,
//! so nothing in it is trusted before all of it is in. The receiver checks what the index alone cannot: that
//! `Index::stream_len` is the manifest's stream size.
use std::fmt;

pub const INDEX_MAGIC: &[u8; 4] = b"LBIX";
pub const INDEX_VERSION: u8 = 1;
/// The most files in one batch.
pub const MAX_ENTRIES: usize = 1_000_000;
/// The longest name, in UTF-8 bytes (what ext4 and most file systems allow).
pub const MAX_NAME_BYTES: usize = 255;
/// The longest stream one transfer carries: 65,535 segments of 16 MiB (`Prepared` refuses longer ones).
pub const MAX_STREAM_BYTES: u64 = u16::MAX as u64 * super::SEGMENT_BYTES;
/// The fewest bytes an entry takes on the wire (a one-byte name, raw, size < 128): what bounds a count.
const MIN_ENTRY_BYTES: usize = 1 + 1 + 1 + 1 + 32;
const COMPRESSED: u8 = 1;
/// The longest index: 64 MiB, what a receiver holds in memory to decode it (`decode` refuses longer input; a
/// manifest naming a longer index can be refused before any of it is buffered).
pub const MAX_INDEX_BYTES: usize = 64 << 20;
/// The longest extension `safe_name` keeps when it shortens a name.
const MAX_KEPT_EXTENSION: usize = 16;

/// One file of a batch. `offset` is the file's position in the batch stream: the first file starts where the
/// index ends (its length as sent, which `decode` takes from the bytes and a sender from `encoded_len`).
/// `stored_size` is the bytes the file takes in the stream: `size` when sent raw, the zstd frame's length when
/// `compressed`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Entry { pub name: String, pub size: u64, pub stored_size: u64, pub compressed: bool, pub offset: u64, pub sha256: [u8; 32] }

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Index { pub entries: Vec<Entry> }

/// Why index bytes (or an index about to be sent) are refused. `entry` is the 0-based entry at fault.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum BatchError {
    /// The bytes end inside a field.
    Truncated,
    /// Bytes follow the last entry.
    TrailingBytes,
    /// An index longer than `MAX_INDEX_BYTES`.
    IndexTooLong(u64),
    BadMagic,
    /// A version this build does not read.
    BadVersion(u8),
    /// No entries, more than `MAX_ENTRIES`, or more than the remaining bytes could hold.
    BadCount(u64),
    /// A varint longer than it needs to be, or past 64 bits.
    BadVarint,
    NameEmpty { entry: u32 },
    NameTooLong { entry: u32, len: u64 },
    NameNotUtf8 { entry: u32 },
    /// `validate` only: the name is not what `safe_name` makes of it (decoding would change it).
    NameUnsafe { entry: u32 },
    /// Flag bits this version does not define.
    BadFlags { entry: u32 },
    /// A size past `MAX_STREAM_BYTES`, a compressed entry with no bytes, or (`validate`) a raw entry whose
    /// stored size is not its size.
    BadSize { entry: u32 },
    /// The file's bytes would end past the longest stream one transfer carries.
    PastStream { entry: u32 },
    /// `validate` only: the file starts before the previous one ends, or the first inside the index (a
    /// duplicate offset of a file with bytes is one).
    Overlap { entry: u32 },
    /// `validate` only: the file starts after the previous one ends, or the first after the index ends.
    Gap { entry: u32 },
}

impl fmt::Display for BatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use BatchError::*;
        match self {
            Truncated => write!(f, "the file list ends early"),
            TrailingBytes => write!(f, "bytes follow the file list"),
            IndexTooLong(n) => write!(f, "a file list of {n} bytes is longer than {MAX_INDEX_BYTES}"),
            BadMagic => write!(f, "not a file list"),
            BadVersion(v) => write!(f, "file list version {v} is not supported"),
            BadCount(n) => write!(f, "a file list of {n} files is not possible"),
            BadVarint => write!(f, "a malformed number in the file list"),
            NameEmpty { entry } => write!(f, "file {} has no name", entry + 1),
            NameTooLong { entry, len } => write!(f, "file {}'s name is {len} bytes (at most {MAX_NAME_BYTES})", entry + 1),
            NameNotUtf8 { entry } => write!(f, "file {}'s name is not UTF-8", entry + 1),
            NameUnsafe { entry } => write!(f, "file {}'s name holds characters a receiver replaces", entry + 1),
            BadFlags { entry } => write!(f, "file {} has unknown flags", entry + 1),
            BadSize { entry } => write!(f, "file {} has an impossible size", entry + 1),
            PastStream { entry } => write!(f, "file {} ends past the longest transfer", entry + 1),
            Overlap { entry } => write!(f, "file {} overlaps the file before it", entry + 1),
            Gap { entry } => write!(f, "file {} does not start where the file before it ends", entry + 1),
        }
    }
}
impl std::error::Error for BatchError {}

impl Index {
    /// The files (indices into `entries`, ascending, each once) with stored bytes in any of the finish-word units
    /// `units`, each `unit_bytes` of the stream (`Manifest::finish_unit_bytes`): what the sender names for finish
    /// words, by the same overlap as the receiver's `BatchStatus::files_in_units`. An empty file is in no unit; the
    /// index and the first files share unit 0; units past the stream name nothing.
    pub fn files_in_units(&self, unit_bytes: u64, units: &[u32]) -> Vec<usize> {
        files_in_units(self.entries.len(), |i| (self.entries[i].offset, self.entries[i].stored_size), unit_bytes, units)
    }

    /// The batch stream's length: where the last file ends.
    pub fn stream_len(&self) -> u64 { self.entries.last().map_or(0, |e| e.offset.saturating_add(e.stored_size)) }

    /// The index's bytes (of any `Index`; `validate` first so that `decode` gives it back unchanged). Offsets
    /// are not written (see the module comment), nor a raw entry's stored size.
    pub fn encode(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(self.encoded_len());
        b.extend_from_slice(INDEX_MAGIC);
        b.push(INDEX_VERSION);
        put_varint(&mut b, self.entries.len() as u64);
        for e in &self.entries {
            put_varint(&mut b, e.name.len() as u64);
            b.extend_from_slice(e.name.as_bytes());
            b.push(if e.compressed { COMPRESSED } else { 0 });
            put_varint(&mut b, e.size);
            if e.compressed { put_varint(&mut b, e.stored_size); }
            b.extend_from_slice(&e.sha256);
        }
        b
    }

    /// `encode().len()`, without encoding.
    pub fn encoded_len(&self) -> usize {
        4 + 1 + varint_len(self.entries.len() as u64) + self.entries.iter().map(|e| {
            varint_len(e.name.len() as u64) + e.name.len() + 1 + varint_len(e.size) + if e.compressed { varint_len(e.stored_size) } else { 0 } + 32
        }).sum::<usize>()
    }

    /// Reads index bytes from an untrusted sender: the whole index, as sent (its length is where the first file
    /// starts). Never panics; allocates at most a small multiple of `bytes.len()` (a count is checked against
    /// the bytes left before anything is reserved), and refuses more than `MAX_INDEX_BYTES` before reading.
    ///
    /// Refused: everything in `BatchError` that bytes can say (length, truncation, trailing bytes, magic, version,
    /// counts outside 1 ..= 1,000,000 or beyond what the bytes hold, overlong varints, names empty, over 255
    /// bytes or not UTF-8, unknown flags, sizes past the transfer limit, a compressed file with no bytes, files
    /// ending past the longest stream). Offsets are laid out back to back, so they never overlap.
    ///
    /// Sanitised, not refused: names pass through `safe_name`, since a real file on one system may carry a
    /// name another cannot write (a `:` from Linux on Windows) and the receiver writes files under these names.
    /// The returned names are therefore always safe to join to a directory; equal names are left for the
    /// receiver to tell apart. A renamed index can encode to another length: offsets stay those of the stream
    /// as sent, so to send it on, lay it out again (`validate` refuses it until then).
    pub fn decode(bytes: &[u8]) -> Result<Index, BatchError> {
        if bytes.len() > MAX_INDEX_BYTES { return Err(BatchError::IndexTooLong(bytes.len() as u64)); }
        let mut r = Reader { b: bytes, p: 0 };
        if r.take(4)? != INDEX_MAGIC { return Err(BatchError::BadMagic); }
        let version = r.byte()?;
        if version != INDEX_VERSION { return Err(BatchError::BadVersion(version)); }
        let count = r.varint()?;
        if count == 0 || count > MAX_ENTRIES as u64 || count > (r.left() / MIN_ENTRY_BYTES) as u64 { return Err(BatchError::BadCount(count)); }
        // the files follow the index in one stream, so they have what the index leaves of the longest one
        let (base, room) = (bytes.len() as u64, MAX_STREAM_BYTES.saturating_sub(bytes.len() as u64));
        let mut entries = Vec::with_capacity(count as usize);
        let mut data = 0u64;
        for entry in 0..count as u32 {
            let len = r.varint()?;
            if len == 0 { return Err(BatchError::NameEmpty { entry }); }
            if len > MAX_NAME_BYTES as u64 { return Err(BatchError::NameTooLong { entry, len }); }
            let raw = std::str::from_utf8(r.take(len as usize)?).map_err(|_| BatchError::NameNotUtf8 { entry })?;
            let flags = r.byte()?;
            if flags & !COMPRESSED != 0 { return Err(BatchError::BadFlags { entry }); }
            let compressed = flags & COMPRESSED != 0;
            let size = r.varint()?;
            let stored_size = if compressed { r.varint()? } else { size };
            check_sizes(entry, size, stored_size, compressed, data, room)?;
            let mut sha256 = [0u8; 32];
            sha256.copy_from_slice(r.take(32)?);
            entries.push(Entry { name: safe_name(raw), size, stored_size, compressed, offset: base + data, sha256 });
            data += stored_size;
        }
        if r.left() != 0 { return Err(BatchError::TrailingBytes); }
        Ok(Index { entries })
    }

    /// What a sender checks before `encode`: everything `decode` refuses, and also that offsets are back to
    /// back from `encoded_len()` (`Overlap`, `Gap`), raw stored sizes equal sizes and names are already
    /// `safe_name`'s, so that `decode(encode(index)) == index`.
    pub fn validate(&self) -> Result<(), BatchError> {
        let count = self.entries.len();
        if count == 0 || count > MAX_ENTRIES { return Err(BatchError::BadCount(count as u64)); }
        let len = self.encoded_len();
        if len > MAX_INDEX_BYTES { return Err(BatchError::IndexTooLong(len as u64)); }
        let (base, room) = (len as u64, MAX_STREAM_BYTES.saturating_sub(len as u64));
        let mut at = 0u64;
        for (i, e) in self.entries.iter().enumerate() {
            let entry = i as u32;
            if e.name.is_empty() { return Err(BatchError::NameEmpty { entry }); }
            if e.name.len() > MAX_NAME_BYTES { return Err(BatchError::NameTooLong { entry, len: e.name.len() as u64 }); }
            if safe_name(&e.name) != e.name { return Err(BatchError::NameUnsafe { entry }); }
            if !e.compressed && e.stored_size != e.size { return Err(BatchError::BadSize { entry }); }
            if e.offset < base + at { return Err(BatchError::Overlap { entry }); }
            if e.offset > base + at { return Err(BatchError::Gap { entry }); }
            check_sizes(entry, e.size, e.stored_size, e.compressed, at, room)?;
            at += e.stored_size;
        }
        Ok(())
    }
}

/// The size rules `decode` and `validate` share; `at`, the stored bytes before the file, is at most `room` (the
/// bytes the stream has after the index) when called.
fn check_sizes(entry: u32, size: u64, stored: u64, compressed: bool, at: u64, room: u64) -> Result<(), BatchError> {
    if size > MAX_STREAM_BYTES || (compressed && (size == 0 || stored == 0)) { return Err(BatchError::BadSize { entry }); }
    if stored > room - at { return Err(BatchError::PastStream { entry }); }
    Ok(())
}

struct Reader<'a> { b: &'a [u8], p: usize }
impl<'a> Reader<'a> {
    fn left(&self) -> usize { self.b.len() - self.p }
    fn take(&mut self, n: usize) -> Result<&'a [u8], BatchError> {
        if n > self.left() { return Err(BatchError::Truncated); }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn byte(&mut self) -> Result<u8, BatchError> { Ok(self.take(1)?[0]) }
    /// Canonical LEB128: at most 10 bytes, the 10th at most 1, and no final zero group after the first byte.
    fn varint(&mut self) -> Result<u64, BatchError> {
        let mut v = 0u64;
        for i in 0..10 {
            let byte = self.byte()?;
            if i == 9 && byte > 1 { return Err(BatchError::BadVarint); }
            v |= ((byte & 0x7f) as u64) << (7 * i);
            if byte & 0x80 == 0 {
                if i > 0 && byte == 0 { return Err(BatchError::BadVarint); }
                return Ok(v);
            }
        }
        Err(BatchError::BadVarint)
    }
}

fn put_varint(b: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 { b.push(v as u8 | 0x80); v >>= 7; }
    b.push(v as u8);
}
pub(crate) fn varint_len(v: u64) -> usize { (64 - v.max(1).leading_zeros() as usize).div_ceil(7) }

/// A character a receiver never writes into a name.
fn unsafe_char(c: char) -> bool {
    matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|'
        // bidirectional marks, embeddings, overrides and isolates; line and paragraph separators
        | '\u{061C}' | '\u{200E}' | '\u{200F}' | '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}' | '\u{2028}' | '\u{2029}')
        || c.is_control()
}

/// A Windows device name: the part before the first dot, trailing spaces aside, in any case.
fn device_name(name: &str) -> bool {
    let stem = name.split('.').next().unwrap_or("").trim_end_matches(' ').to_ascii_uppercase();
    match stem.as_str() {
        "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$" => true,
        _ => (stem.starts_with("COM") || stem.starts_with("LPT")) && {
            let mut rest = stem[3..].chars();
            matches!((rest.next(), rest.next()), (Some('0'..='9' | '\u{B9}' | '\u{B2}' | '\u{B3}'), None))
        },
    }
}

/// The name a receiver writes a file under, from the name the sender sent (decode applies it; a sender
/// applies it first to show the same name). In order: a name over 255 bytes is cut at a character boundary,
/// keeping its last extension when that is at most 16 bytes; path separators (`/`, `\`), the characters
/// Windows and Android's storage refuse (`: * ? " < > |`), control characters, line and paragraph separators
/// and the bidirectional controls that can disguise an extension become `_` (a zero-width joiner stays: emoji
/// need it); trailing dots and spaces are dropped (Windows drops them); a name left empty becomes `_`; a
/// Windows device name (`CON`, `nul.txt`, `COM1.log`, ...) gets a leading `_` (cut again if that passes 255
/// bytes). The result is at most 255 bytes, has no separator, is never `.` or `..`, and is its own safe name.
pub fn safe_name(raw: &str) -> String {
    let ends_badly = |s: &str| s.ends_with(['.', ' ']);
    // the common case, a name that is fine as it is: one scan and one copy
    if !raw.is_empty() && raw.len() <= MAX_NAME_BYTES && !raw.chars().any(unsafe_char) && !ends_badly(raw) && !device_name(raw) { return raw.to_owned(); }
    // cut first: what follows only shortens a name (one `_` per character), save the device prefix
    let mut s: String = cut(raw, MAX_NAME_BYTES).chars().map(|c| if unsafe_char(c) { '_' } else { c }).collect();
    s.truncate(s.trim_end_matches(['.', ' ']).len());
    if s.is_empty() { s.push('_'); }
    if device_name(&s) {
        s.insert(0, '_');
        // starting with `_`, it is no device name however it is cut
        if s.len() > MAX_NAME_BYTES { s = cut(&s, MAX_NAME_BYTES).into_owned(); s.truncate(s.trim_end_matches(['.', ' ']).len()); }
    }
    s
}

/// `name` in at most `max` bytes, cut at a character boundary: the stem is cut and the last extension kept
/// when it is at most `MAX_KEPT_EXTENSION` bytes (the dot included) and follows a non-empty stem.
fn cut(name: &str, max: usize) -> std::borrow::Cow<'_, str> {
    if name.len() <= max { return name.into(); }
    let floor = |n: usize| { let mut n = n.min(name.len()); while !name.is_char_boundary(n) { n -= 1; } n };
    match name.rfind('.') {
        Some(dot) if dot > 0 && name.len() - dot <= MAX_KEPT_EXTENSION => format!("{}{}", &name[..floor(max - (name.len() - dot))], &name[dot..]).into(),
        _ => name[..floor(max)].into(),
    }
}

/// The files (of `len`, in stream order: `span(i)` = (offset, stored size), ends never decreasing) with stored bytes in
/// any of `units` (each `unit_bytes` of the stream), ascending, each once. A binary search per unit, then only the
/// files in it.
pub(crate) fn files_in_units(len: usize, span: impl Fn(usize) -> (u64, u64), unit_bytes: u64, units: &[u32]) -> Vec<usize> {
    let mut units = units.to_vec();
    units.sort_unstable();
    units.dedup();
    let mut out: Vec<usize> = Vec::new();
    for u in units {
        let start = (u as u64).saturating_mul(unit_bytes);
        let end = start.saturating_add(unit_bytes);
        // the first file ending after the unit starts
        let (mut lo, mut hi) = (0, len);
        while lo < hi { let mid = (lo + hi) / 2; let (o, n) = span(mid); if o + n <= start { lo = mid + 1 } else { hi = mid } }
        for i in lo..len {
            let (o, n) = span(i);
            if o >= end { break; }
            // a file over several units comes first in the next one: once
            if n > 0 && out.last() != Some(&i) { out.push(i); }
        }
    }
    out
}

/// Files named for finish words, as both ends show them: "3 files: a.jpg, b.pdf +1": the count, the first two
/// names (`first`; any after two are ignored), a name over 40 characters shortened in the middle so its end (the
/// extension) shows, and how many more. "1 file: d.bin", "2 files: a, b".
pub fn files_text(count: usize, first: &[&str]) -> String {
    let names: Vec<String> = first.iter().take(2).map(|n| middle_cut(n, 40)).collect();
    let more = if count > 2 { format!(" +{}", count - 2) } else { String::new() };
    format!("{count} file{}: {}{more}", if count == 1 { "" } else { "s" }, names.join(", "))
}

/// `name` in at most `max` characters: shortened in the middle with an ellipsis, so its end (the extension) shows.
pub(crate) fn middle_cut(name: &str, max: usize) -> String {
    let n = name.chars().count();
    if n <= max { return name.to_string(); }
    let head = (max - 1) / 2;
    let tail = max - 1 - head;
    name.chars().take(head).chain(std::iter::once('\u{2026}')).chain(name.chars().skip(n - tail)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, size: u64, stored: Option<u64>, seed: u8) -> Entry {
        Entry { name: name.into(), size, stored_size: stored.unwrap_or(size), compressed: stored.is_some(), offset: 0, sha256: [seed; 32] }
    }
    /// Lays `entries` out back to back after the index, as a sender does.
    fn index(entries: Vec<Entry>) -> Index {
        let mut idx = Index { entries };
        let mut at = idx.encoded_len() as u64;
        for e in &mut idx.entries { e.offset = at; at += e.stored_size; }
        idx
    }
    /// The names exactly as `b` (an index `decode` accepted) carries them.
    fn wire_names(b: &[u8]) -> Vec<Vec<u8>> {
        fn v(b: &[u8], p: &mut usize) -> u64 {
            let (mut x, mut s) = (0u64, 0);
            loop { let c = b[*p]; *p += 1; x |= ((c & 0x7f) as u64) << s; s += 7; if c & 0x80 == 0 { return x; } }
        }
        let mut p = 5;
        let n = v(b, &mut p);
        (0..n).map(|_| {
            let len = v(b, &mut p) as usize;
            let name = b[p..p + len].to_vec(); p += len;
            let flags = b[p]; p += 1;
            v(b, &mut p); if flags & 1 == 1 { v(b, &mut p); }
            p += 32;
            name
        }).collect()
    }
    fn sample() -> Index {
        index(vec![
            entry("empty.txt", 0, None, 1),
            entry("also empty", 0, None, 2),
            entry("a.bin", 1, None, 3),
            entry("été — notes.md", 5_000, Some(1_234), 4),
            entry(&"n".repeat(255), 300, None, 5),
            entry("大きな写真.jpeg", 3 << 20, None, 6),
            entry("video.mp4", 40_000_000_000, None, 7),
        ])
    }
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
        fn below(&mut self, n: u64) -> u64 { self.next() % n }
    }
    fn varint(mut v: u64, b: &mut Vec<u8>) {
        loop { let low = (v & 0x7f) as u8; v >>= 7; if v == 0 { b.push(low); return; } b.push(low | 0x80); }
    }
    /// Raw index bytes, for what `encode` cannot write: (name bytes, flags, size, stored size).
    fn raw(entries: &[(&[u8], u8, u64, u64)]) -> Vec<u8> {
        let mut b = INDEX_MAGIC.to_vec(); b.push(INDEX_VERSION); varint(entries.len() as u64, &mut b);
        for (name, flags, size, stored) in entries {
            varint(name.len() as u64, &mut b); b.extend_from_slice(name); b.push(*flags); varint(*size, &mut b);
            if flags & 1 == 1 { varint(*stored, &mut b); }
            b.extend_from_slice(&[9; 32]);
        }
        b
    }

    #[test]
    fn round_trip() {
        let idx = sample();
        assert_eq!(idx.validate(), Ok(()));
        let b = idx.encode();
        assert_eq!(&b[..4], INDEX_MAGIC);
        assert_eq!(Index::decode(&b), Ok(idx.clone()));
        // offsets are positions in the batch stream: the first file starts where the index ends
        assert_eq!(idx.entries[0].offset, b.len() as u64);
        assert_eq!(idx.stream_len(), b.len() as u64 + 1 + 1_234 + 300 + (3 << 20) + 40_000_000_000);
        // compact: 36 bytes + the name per raw entry, plus the varint bytes past one for sizes and lengths
        let want: usize = idx.entries.iter().map(|e| {
            let vlen = |v: u64| (64 - v.max(1).leading_zeros() as usize).div_ceil(7);
            33 + vlen(e.name.len() as u64) + e.name.len() + vlen(e.size) + if e.compressed { vlen(e.stored_size) } else { 0 }
        }).sum();
        assert_eq!(b.len(), 4 + 1 + 1 + want);
        // a one-file batch
        let one = index(vec![entry("x", 3, None, 0)]);
        assert_eq!(Index::decode(&one.encode()), Ok(one));
    }

    #[test]
    fn the_layout_is_pinned() {
        let idx = index(vec![entry("a", 0, None, 0xaa), entry("bc", 300, Some(200), 0xbb)]);
        let mut want = b"LBIX\x01\x02".to_vec();
        want.extend_from_slice(b"\x01a\x00\x00"); want.extend_from_slice(&[0xaa; 32]);
        want.extend_from_slice(b"\x02bc\x01\xac\x02\xc8\x01"); want.extend_from_slice(&[0xbb; 32]);
        assert_eq!(idx.encode(), want);
        let got = Index::decode(&want).unwrap();
        assert_eq!((got.entries[0].offset, got.entries[1].offset, got.stream_len()), (want.len() as u64, want.len() as u64, want.len() as u64 + 200));
    }

    #[test]
    fn offsets_are_stream_positions_even_when_names_are_sanitised() {
        // a 3-byte bidi control becomes `_`, CON becomes _CON: the index as decoded would encode to another
        // length, but the files sit after the index as it was sent
        let b = raw(&[(b"a\xe2\x80\xaeb.txt", 0, 10, 0), (b"CON", 0, 20, 0), (b"x:y", 1, 500, 30), (b"z", 0, 7, 0)]);
        let idx = Index::decode(&b).unwrap();
        assert_eq!(idx.entries.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(), ["a_b.txt", "_CON", "x_y", "z"]);
        assert_ne!(idx.encoded_len(), b.len());
        let l = b.len() as u64;
        assert_eq!(idx.entries.iter().map(|e| e.offset).collect::<Vec<_>>(), [l, l + 10, l + 30, l + 60]);
        assert_eq!(idx.stream_len(), l + 67);
    }

    #[test]
    fn long_names_keep_their_extension() {
        let cjk = format!("{}.jpg", "写".repeat(100));
        let s = safe_name(&cjk);
        assert!(s.len() <= MAX_NAME_BYTES && s.ends_with(".jpg") && s.starts_with("写写"), "{} bytes: {s}", s.len());
        assert_eq!(s.len(), 83 * 3 + 4, "the stem is cut at a character boundary");
        // an extension over 16 bytes is not one worth keeping: the name is cut at the end
        let long_ext = format!("{}.{}", "a".repeat(250), "b".repeat(20));
        assert_eq!(safe_name(&long_ext), long_ext[..255]);
        // a cut that leaves a trailing dot or space drops it; one that leaves a device name prefixes it
        let spaced = format!("{}{}b", "a".repeat(240), " ".repeat(30));
        assert_eq!(safe_name(&spaced), "a".repeat(240));
        let dev = format!("CON{}x", " ".repeat(252));
        assert_eq!(safe_name(&dev), "_CON");
        let dev_ext = format!("con.{}", "x".repeat(251));
        let s = safe_name(&dev_ext);
        assert!(s.len() <= MAX_NAME_BYTES && s.starts_with("_con.x"), "{}", s.len());
    }

    #[test]
    fn safe_name_is_idempotent() {
        let pieces = ["a", "Z", ".", " ", "/", "\\", ":", "\u{0}", "\n", "\u{85}", "\u{202E}", "\u{2028}", "\u{200D}", "写", "é", "😀", "CON", "nul", "com1", "LPT9", ".jpg", "..", "_"];
        // a cut that exposes trailing spaces or dots in front of a device name
        for s in [format!("CON{}x", " ".repeat(252)), format!("nul{}y", ".".repeat(253)), format!("AUX{}z", ". ".repeat(130))] {
            let once = safe_name(&s);
            assert_eq!(safe_name(&once), once, "{s:?}");
        }
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        for _ in 0..20_000 {
            let mut s = String::new();
            let target = rng.below(800) as usize;
            while s.len() < target { s.push_str(pieces[rng.below(pieces.len() as u64) as usize]); }
            let once = safe_name(&s);
            assert_eq!(safe_name(&once), once, "{s:?}");
            assert!(!once.is_empty() && once.len() <= MAX_NAME_BYTES && once != "." && once != ".." && !once.contains(['/', '\\']), "{s:?} -> {once:?}");
        }
    }

    #[test]
    fn malformed_never_panics() {
        let good = sample().encode();
        // every truncation is refused (the count promises entries the bytes no longer hold)
        for n in 0..good.len() { assert!(Index::decode(&good[..n]).is_err(), "truncated to {n}"); }
        let mut long = good.clone(); long.push(0);
        assert_eq!(Index::decode(&long), Err(BatchError::TrailingBytes));
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let check = |b: &[u8]| {
            if let Ok(idx) = Index::decode(b) {
                // files back to back from the end of the index as sent; names safe as they are
                let mut at = b.len() as u64;
                for e in &idx.entries { assert_eq!(e.offset, at, "{b:?}"); at += e.stored_size; }
                assert_eq!(idx.stream_len(), at);
                assert!(idx.entries.iter().all(|e| safe_name(&e.name) == e.name && e.name != "." && e.name != ".." && !e.name.contains(['/', '\\'])));
                let renamed = wire_names(b).iter().zip(&idx.entries).any(|(w, e)| w != e.name.as_bytes());
                if !renamed {
                    // nothing renamed: the encoding is canonical, a second trip gives back the very bytes
                    assert_eq!(idx.encode(), b);
                    assert_eq!(idx.validate(), Ok(()), "{b:?}");
                } else {
                    // renamed: laid out again after its own (other) length, it is a valid index to send on
                    let again = index(idx.entries.clone());
                    assert_eq!(again.validate(), Ok(()), "{b:?}");
                    assert_eq!(Index::decode(&again.encode()).as_ref(), Ok(&again));
                }
            }
        };
        for _ in 0..20_000 {
            // flipped bits and bytes in a good index
            let mut b = good.clone();
            for _ in 0..1 + rng.below(4) { let i = rng.below(b.len() as u64) as usize; b[i] ^= 1 << rng.below(8); }
            if rng.below(4) == 0 { let i = rng.below(b.len() as u64) as usize; b[i] = rng.next() as u8; }
            check(&b);
            // random bytes, bare and behind a good header and count
            let n = rng.below(600) as usize;
            let tail: Vec<u8> = (0..n).map(|_| rng.next() as u8).collect();
            check(&tail);
            let mut h = INDEX_MAGIC.to_vec(); h.push(INDEX_VERSION); varint(1 + rng.below(8), &mut h); h.extend_from_slice(&tail);
            check(&h);
        }
    }

    #[test]
    fn ten_thousand_entries_parse_fast() {
        let mut rng = Rng(42);
        let idx = index((0..10_000u32).map(|i| {
            let name = match i % 4 { 0 => format!("IMG_2026103{}_{i:05}.jpg", i % 10), 1 => format!("notes {i} — été.txt"), 2 => format!("写真_{i}.png"), _ => format!("report-{i}.pdf") };
            let size = rng.below(1 << 20);
            let stored = if i % 3 == 0 && size > 0 { Some(1 + size / 2) } else { None };
            let mut e = entry(&name, size, stored, i as u8); e.sha256[..4].copy_from_slice(&i.to_le_bytes()); e
        }).collect());
        let b = idx.encode();
        // the fastest of five parses (the timing gates' rule)
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            let got = std::hint::black_box(Index::decode(std::hint::black_box(&b)));
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
            assert_eq!(got.as_ref().map(|i| i.entries.len()), Ok(10_000));
        }
        eprintln!("10,000 entries: {} bytes, decoded in {best:.3} ms (fastest of five)", b.len());
        assert_eq!(Index::decode(&b), Ok(idx));
        assert!(best <= 50.0, "10,000-entry index took {best:.1} ms (cap 50 ms)");
    }

    #[test]
    fn overlaps_and_duplicate_offsets_are_refused() {
        let mut idx = sample();
        idx.entries[3].offset -= 1;
        assert_eq!(idx.validate(), Err(BatchError::Overlap { entry: 3 }));
        // two files with bytes at one offset
        let mut dup = index(vec![entry("a", 10, None, 0), entry("b", 10, None, 0)]);
        dup.entries[1].offset = dup.entries[0].offset;
        assert_eq!(dup.validate(), Err(BatchError::Overlap { entry: 1 }));
        // empty files share their offset with the next file: that is no overlap
        let first = sample().encoded_len() as u64;
        assert_eq!(sample().entries[..4].iter().map(|e| e.offset).collect::<Vec<_>>(), [first, first, first, first + 1]);
        let mut gap = sample(); gap.entries[2].offset += 1;
        assert_eq!(gap.validate(), Err(BatchError::Gap { entry: 2 }));
        // the first file starts exactly where the index ends: not inside it, not after a gap
        let mut late = sample(); late.entries[0].offset += 5;
        assert_eq!(late.validate(), Err(BatchError::Gap { entry: 0 }));
        let mut early = sample(); early.entries[0].offset -= 1;
        assert_eq!(early.validate(), Err(BatchError::Overlap { entry: 0 }));
    }

    #[test]
    fn offsets_past_the_stream_are_refused() {
        // two files that each fit, but together end past the longest stream
        let half = MAX_STREAM_BYTES / 2 + 1;
        let b = raw(&[(b"a", 0, half, 0), (b"b", 0, half, 0)]);
        assert_eq!(Index::decode(&b), Err(BatchError::PastStream { entry: 1 }));
        // one file exactly as long as the stream leaves room for after the index is fine
        let fits = MAX_STREAM_BYTES - raw(&[(b"a", 0, MAX_STREAM_BYTES, 0)]).len() as u64;
        assert!(Index::decode(&raw(&[(b"a", 0, fits, 0)])).is_ok());
        assert_eq!(Index::decode(&raw(&[(b"a", 0, fits + 1, 0)])), Err(BatchError::PastStream { entry: 0 }));
        let idx = index(vec![entry("a", half, None, 0), entry("b", half, None, 0)]);
        assert_eq!(idx.validate(), Err(BatchError::PastStream { entry: 1 }));
        // a size past the transfer limit, even compressed into a few bytes
        assert_eq!(Index::decode(&raw(&[(b"a", 1, MAX_STREAM_BYTES + 1, 10)])), Err(BatchError::BadSize { entry: 0 }));
        assert_eq!(Index::decode(&raw(&[(b"a", 1, u64::MAX, 10)])), Err(BatchError::BadSize { entry: 0 }));
    }

    #[test]
    fn names_are_checked() {
        let long = vec![b'n'; 256];
        assert_eq!(Index::decode(&raw(&[(b"ok", 0, 1, 0), (&long, 0, 1, 0)])), Err(BatchError::NameTooLong { entry: 1, len: 256 }));
        assert!(Index::decode(&raw(&[(&long[..255], 0, 1, 0)])).is_ok());
        // a length that would not fit in memory is refused before anything is read
        let mut huge = INDEX_MAGIC.to_vec(); huge.push(INDEX_VERSION); varint(1, &mut huge); varint(u64::MAX, &mut huge); huge.extend_from_slice(&[0; 40]);
        assert_eq!(Index::decode(&huge), Err(BatchError::NameTooLong { entry: 0, len: u64::MAX }));
        assert_eq!(Index::decode(&raw(&[(b"\xff\xfe.txt", 0, 1, 0)])), Err(BatchError::NameNotUtf8 { entry: 0 }));
        assert_eq!(Index::decode(&raw(&[(b"caf\xc3", 0, 1, 0)])), Err(BatchError::NameNotUtf8 { entry: 0 }));
        // (alone, an entry with no name is shorter than any entry can be: the count is refused first)
        assert_eq!(Index::decode(&raw(&[(b"", 0, 1, 0)])), Err(BatchError::BadCount(1)));
        assert_eq!(Index::decode(&raw(&[(b"abc", 0, 1, 0), (b"", 0, 1, 0)])), Err(BatchError::NameEmpty { entry: 1 }));
        // (a name changes the index's length: lay the files out again after it)
        let renamed = |i: usize, name: &str| { let mut e = sample().entries; e[i].name = name.into(); index(e).validate() };
        assert_eq!(renamed(4, &"n".repeat(256)), Err(BatchError::NameTooLong { entry: 4, len: 256 }));
        assert_eq!(renamed(1, ""), Err(BatchError::NameEmpty { entry: 1 }));
        assert_eq!(renamed(2, "../a.bin"), Err(BatchError::NameUnsafe { entry: 2 }));
    }

    #[test]
    fn unsafe_names_are_sanitised() {
        for (raw_name, want) in [
            ("../../etc/passwd", ".._.._etc_passwd"),
            ("..\\..\\Windows\\win.ini", ".._.._Windows_win.ini"),
            ("/abs", "_abs"),
            ("..", "_"), (".", "_"), ("...", "_"), ("  ", "_"),
            ("report.txt. . ", "report.txt"),
            ("a:b*c?d\"e<f>g|h.txt", "a_b_c_d_e_f_g_h.txt"),
            ("tab\there\nnew\u{0}nul\u{7f}\u{85}.txt", "tab_here_new_nul__.txt"),
            ("invoice\u{202E}fdp.exe", "invoice_fdp.exe"),
            ("a\u{2066}b\u{200F}c", "a_b_c"),
            ("a\u{061C}b\u{2028}c\u{2029}d.txt", "a_b_c_d.txt"),
            ("family \u{1F468}\u{200D}\u{1F469}.png", "family \u{1F468}\u{200D}\u{1F469}.png"),
            ("CON", "_CON"), ("nul.txt", "_nul.txt"), ("Com1.log", "_Com1.log"), ("lpt9", "_lpt9"), ("AUX .md", "_AUX .md"),
            ("COM10.txt", "COM10.txt"), ("CONSOLE.txt", "CONSOLE.txt"), ("icon.png", "icon.png"),
            ("été — notes.md", "été — notes.md"), (".bashrc", ".bashrc"), ("  lead.txt", "  lead.txt"), ("写真.png", "写真.png"),
        ] {
            assert_eq!(safe_name(raw_name), want, "{raw_name:?}");
            assert_eq!(safe_name(want), want, "safe names stay as they are: {want:?}");
        }
        // decode applies it
        let b = raw(&[(b"../x:y.txt", 0, 1, 0), (b"CON", 0, 1, 0)]);
        assert_eq!(Index::decode(&b).unwrap().entries.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(), [".._x_y.txt", "_CON"]);
    }

    #[test]
    fn absurd_counts_and_bad_fields_are_refused() {
        let mut h = INDEX_MAGIC.to_vec(); h.push(INDEX_VERSION);
        let with_count = |n: u64, extra: usize| { let mut b = h.clone(); varint(n, &mut b); b.extend(std::iter::repeat_n(0, extra)); b };
        assert_eq!(Index::decode(&with_count(0, 0)), Err(BatchError::BadCount(0)));
        assert_eq!(Index::decode(&with_count(1_000_001, 1_000_001 * 36)), Err(BatchError::BadCount(1_000_001)));
        assert_eq!(Index::decode(&with_count(u64::MAX, 100)), Err(BatchError::BadCount(u64::MAX)));
        // a million entries promised in 40 bytes: refused before anything is reserved for them
        assert_eq!(Index::decode(&with_count(1_000_000, 40)), Err(BatchError::BadCount(1_000_000)));
        assert_eq!(Index::decode(&with_count(2, 71)), Err(BatchError::BadCount(2)));
        // an index past 64 MiB is refused before it is read
        let big = vec![0u8; MAX_INDEX_BYTES + 1];
        assert_eq!(Index::decode(&big), Err(BatchError::IndexTooLong(MAX_INDEX_BYTES as u64 + 1)));
        let fat = index(vec![entry(&"n".repeat(255), 1, None, 0); 240_000]);
        assert!(fat.encoded_len() > MAX_INDEX_BYTES);
        assert_eq!(fat.validate(), Err(BatchError::IndexTooLong(fat.encoded_len() as u64)));
        let mut b = raw(&[(b"a", 0, 1, 0)]); b[0] = b'X';
        assert_eq!(Index::decode(&b), Err(BatchError::BadMagic));
        let mut b = raw(&[(b"a", 0, 1, 0)]); b[4] = 2;
        assert_eq!(Index::decode(&b), Err(BatchError::BadVersion(2)));
        assert_eq!(Index::decode(&raw(&[(b"a", 2, 1, 0)])), Err(BatchError::BadFlags { entry: 0 }));
        assert_eq!(Index::decode(&raw(&[(b"a", 0x81, 1, 1)])), Err(BatchError::BadFlags { entry: 0 }));
        // compressed with nothing in or nothing out
        assert_eq!(Index::decode(&raw(&[(b"a", 1, 0, 9)])), Err(BatchError::BadSize { entry: 0 }));
        assert_eq!(Index::decode(&raw(&[(b"a", 1, 9, 0)])), Err(BatchError::BadSize { entry: 0 }));
        // varints: an overlong count (1 as two bytes) and one past 64 bits
        let mut b = h.clone(); b.extend_from_slice(&[0x81, 0x00]);
        assert_eq!(Index::decode(&b), Err(BatchError::BadVarint));
        let mut b = h.clone(); b.push(1); b.push(1); b.push(b'a'); b.push(0); b.extend_from_slice(&[0xff; 9]); b.push(0x02); b.extend_from_slice(&[0; 32]);
        assert_eq!(Index::decode(&b), Err(BatchError::BadVarint));
        let mut b = h.clone(); b.push(1); b.push(1); b.push(b'a'); b.push(0); b.extend_from_slice(&[0xff; 9]); b.push(0x01); b.extend_from_slice(&[0; 32]);
        assert_eq!(Index::decode(&b), Err(BatchError::BadSize { entry: 0 }), "u64::MAX itself reads, then is too big");
        // validate's own: a raw file whose stored size is not its size, an empty index, too many entries
        let mut idx = sample(); idx.entries[6].stored_size -= 1;
        assert_eq!(idx.validate(), Err(BatchError::BadSize { entry: 6 }));
        assert_eq!(Index { entries: vec![] }.validate(), Err(BatchError::BadCount(0)));
        let many = Index { entries: vec![entry("a", 0, None, 0); MAX_ENTRIES + 1] };
        assert_eq!(many.validate(), Err(BatchError::BadCount(MAX_ENTRIES as u64 + 1)));
        assert!(index(vec![entry("a", 0, None, 0); MAX_ENTRIES]).validate().is_ok());
    }

    #[test]
    fn the_index_names_the_files_in_units_by_overlap() {
        // units of 100 bytes; the index takes the stream's first bytes, then (offset, stored size):
        let mut idx = Index { entries: vec![entry("a", 100, None, 1), entry("empty", 0, None, 2), entry("b", 250, None, 3), entry("c", 10, None, 4), entry("z", 0, None, 5)] };
        let spans = [(50u64, 100u64), (150, 0), (150, 250), (400, 10), (410, 0)];
        for (e, (o, n)) in idx.entries.iter_mut().zip(spans) { e.offset = o; e.stored_size = n; }
        assert_eq!(idx.files_in_units(100, &[0]), vec![0], "the index shares unit 0 with the first file");
        assert_eq!(idx.files_in_units(100, &[1]), vec![0, 2], "a file across a unit edge is in both; an empty one in none");
        assert_eq!(idx.files_in_units(100, &[3, 2, 3]), vec![2], "units unsorted and repeated: each file once");
        assert_eq!(idx.files_in_units(100, &[4]), vec![3]);
        assert_eq!(idx.files_in_units(100, &[4, 1]), vec![0, 2, 3]);
        assert!(idx.files_in_units(100, &[5, 9, u32::MAX]).is_empty() && idx.files_in_units(100, &[]).is_empty(), "past the stream: nothing");
    }
    #[test]
    fn files_text_names_two_files_and_counts_the_rest() {
        assert_eq!(files_text(3, &["a.jpg", "b.pdf", "c.txt"]), "3 files: a.jpg, b.pdf +1");
        assert_eq!(files_text(1, &["d.bin"]), "1 file: d.bin");
        assert_eq!(files_text(2, &["a", "b"]), "2 files: a, b");
        assert_eq!(files_text(200, &["a", "b"]), "200 files: a, b +198");
        let long = format!("{}.jpeg", "holiday-photo-from-the-beach-".repeat(3));
        let shown = files_text(1, &[&long]);
        assert_eq!(shown.chars().count(), "1 file: ".len() + 40, "{shown}");
        assert!(shown.ends_with("beach-.jpeg") && shown.contains('\u{2026}'), "the end (the extension) stays: {shown}");
    }
}
