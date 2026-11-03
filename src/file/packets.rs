//! The blind packet schedule shared by every sender (spec §4.2): a systematic pass over every block of
//! every chunk, coverage-balanced (every stretch of the pass carries each chunk's blocks in proportion to
//! its size), then gap-prioritised repair rounds, with manifest packets on request. Never needs feedback
//! from the receiver; finish words typed on the sender (`set_focus`) narrow it to the units the receiver
//! is still missing. Measured against the segment-by-segment pass: docs/2026-10-30-repair-schedule.md.
use super::manifest::{chunk_blocks, Manifest};
use super::prepare::Prepared;
use crate::fountain::{neighbors, LtEncoder, BLOCK_BYTES};
use crate::packet::{Packet, PacketId};
use sha2::{Digest, Sha256};
use std::cmp::Ordering as Order;
use std::collections::BinaryHeap;
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Nanoseconds from the clock (wasm32-unknown-unknown has no clock: a constant there).
fn clock() -> u64 {
    #[cfg(not(target_arch = "wasm32"))]
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1);
    #[cfg(target_arch = "wasm32")]
    let t: u64 = 0x9E37_79B9_7F4A_7C15;
    t
}

/// A new transfer id from the clock, for a sender that wants another id on every run (senders use the
/// stream's own id, `content_transfer_id`, by default). A constant on wasm32-unknown-unknown (no clock).
pub fn fresh_transfer_id() -> u16 {
    let t = clock();
    ((t ^ (t >> 17)) as u16) | 1
}

/// The transfer id of a prepared stream: the first 16 bits (big-endian) of SHA-256 of the encoded manifest,
/// low bit set (never 0). The same file under the same name, prepared the same way, always gets the same id;
/// another name, and the same file compressed or not (another stream, another manifest), get another, so
/// a receiver never mixes two streams under one id. Every empty file differs by its name. A batch's manifest
/// carries its batch flag, its index's SHA-256 and its name, so the id hashes those too: a batch and a single
/// file (even one whose bytes are the batch's stream) hash different manifests, and the same batch sent again
/// gets the same id. (Ids are 15 bits: two different transfers share one with probability 2^-15.)
pub fn content_transfer_id(m: &Manifest) -> u16 {
    let id = content_hash_id(m);
    // a file never takes the Tune carousel's id (receivers in Tune know its frames by it): another odd id, the same
    // every time, so finish words and resume still work
    if m.tune.is_some() { return id; }
    let mut id = id;
    while crate::tune::schedule::tune_display_60(id).is_some() { id ^= 2; if crate::tune::schedule::tune_display_60(id).is_some() { id ^= 4; } }
    id
}
/// The manifest's id from its SHA-256 alone (`content_transfer_id` before it steps off the Tune carousel's ids).
pub fn content_hash_id(m: &Manifest) -> u16 {
    let d = Sha256::digest(m.encode());
    u16::from_be_bytes([d[0], d[1]]) | 1
}

/// Where a source's focus repair rounds start: seeded from the clock once per process, then a step of
/// 1899 rounds per source (~0.618 of a full chunk's 3072 repair ids), so a sender restarted in the same
/// process starts far from the ids the last ones sent, and one restarted later wherever the clock puts it.
fn fresh_repair_offset() -> u64 {
    static SEED: OnceLock<u64> = OnceLock::new();
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let seed = *SEED.get_or_init(|| {
        // splitmix64: the clock's low bits are coarse (a multiple of 100 ns on some systems)
        let mut z = clock().wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) >> 32
    });
    seed + NEXT.fetch_add(1, Ordering::Relaxed) * 1899
}

/// Where the chunks start in sweep `n` of `c` chunks (a sweep: every chunk's block `n` of the first pass, or
/// every chunk's repair round `n`): c * frac(n * 0.618...), a golden-ratio (Weyl) rotation. In a fixed order a
/// packet slot of the frame carries the same chunk in every frame whenever the chunk count shares a factor with
/// the packets per frame (41, 82, 123 or 164 chunks at 82 packets), so a part of the frame that is never read
/// (glare, a corner out of focus) starved those chunks: measured, 2x the frames at 41 and 123 chunks, never
/// complete at 82 and 164. Rotated so, no chunk count and frame size lock a slot to a chunk (an integer step
/// still did when the chunk count divides the packets per frame); docs/2026-10-30-repair-schedule.md.
fn sweep_start(n: u64, c: u32) -> u32 { ((n.wrapping_mul(0x9E37_79B9_7F4A_7C15) as u128 * c as u128) >> 64) as u32 }

/// A chunk's place in the first pass: its block `next` (of `k`) is due at (2 next + 1) / 2k of the pass; ties
/// (chunks of one size) by `key`, the chunk's position `idx` in the stream rotated by the sweep's start.
#[derive(PartialEq, Eq)]
struct Due { next: u16, k: u16, idx: u32, key: u32 }
impl Ord for Due {
    /// The earliest due first out of the (max-)heap: the greater `Due` is the earlier one.
    fn cmp(&self, o: &Due) -> Order {
        let (a, b) = ((2 * self.next as u32 + 1) * o.k as u32, (2 * o.next as u32 + 1) * self.k as u32);
        b.cmp(&a).then(o.key.cmp(&self.key)).then(o.idx.cmp(&self.idx))
    }
}
impl PartialOrd for Due { fn partial_cmp(&self, o: &Due) -> Option<Order> { Some(self.cmp(o)) } }

/// Repair rounds each named unit gets before the sender leaves finish words by itself, per block of the unit's
/// largest chunk (a segment unit's round is a packet of each of its chunks, a chunk unit's a packet of it). One round
/// per block is enough to decode the unit from nothing when every frame is read; FOCUS_ROUNDS_PER_BLOCK of them
/// cover a receiver that reads half the frames (an LT decode needs a few percent over k). Then the normal schedule
/// again, so words that name only some of the short units (a window code on a large file, an old code, a mistyped
/// one the check let through) never leave the rest waiting. What it costs when the words were wrong: the named
/// units' share of FOCUS_ROUNDS_PER_BLOCK passes. Measured (64 MB, 4 segments, 30% of frames lost after the
/// pass; tests/completion.rs): words for the one short segment finish in 0.69x the normal schedule's frames at 1
/// round per block (focus ends before the segment is in), 0.55x at 1.5 and at 2; words naming 1 of 3 short
/// segments cost 0.26, 0.40 and 0.51 of a pass over the normal schedule.
pub const FOCUS_ROUNDS_PER_BLOCK: u32 = 2;

/// A batch's file list goes out again, the whole transfer long, once every this many data frames (all its blocks,
/// one at a time, spread), so a receiver that joins at any moment lists the files about one manifest period later
/// (31 frames, ~1 s at 30 fps: it waits that long for the manifest anyway). Before, a receiver that missed the
/// list's first blocks waited for the list until its chunk decoded, about when everything else did (4 files, 9.4
/// MB: 397 frames after joining anywhere in the pass); measured in docs/2026-10-30-repair-schedule.md "The file
/// list, repeated".
pub const LIST_EVERY_FRAMES: u64 = 30;
/// The repeats take at most one packet in this many (2%): a longer list comes round more slowly.
const LIST_MIN_SPACING: u64 = 50;
/// Packets per data frame until the sender says (`set_packets_per_frame`): the 2 px b/w tile of 1024 x 1024.
const DEFAULT_PACKETS_PER_FRAME: usize = 24;

/// Finish words being sent: the units (sorted), the one whose round is next, the chunk within it, and the rounds
/// each unit still has before the sender leaves focus (`left`, by position in `units`; a unit at 0 is skipped).
struct Focus { units: Vec<u32>, pos: usize, chunk: u8, left: Vec<u32> }

pub struct PacketSource {
    prepared: Prepared,
    manifest_bytes: Vec<u8>, manifest_encoders: Vec<LtEncoder>, manifest_cursor: (u8, u16),
    /// the first pass still to send, and (segment, chunk) of each chunk by its position in the stream
    pass_order: BinaryHeap<Due>, chunks: Vec<(u16, u8)>,
    /// packets the first pass sends before its balanced order (a batch's file list), and how many went
    front: Vec<PacketId>, front_sent: usize,
    /// the list sent again: every `list_spacing` packets of the schedule (16.16 fixed point; 0: never), the next one
    /// due once `plain` (the schedule's packets after the list's first blocks, repeats not counted) reaches
    /// `list_next`, and the list block it is; the stream's blocks, and the packets of a data frame
    list_spacing: u64, list_next: u64, plain: u64, list_cursor: usize, blocks: u64, packets_per_frame: usize,
    /// repair packets of the normal rounds sent so far: round n = repair / chunks, every chunk once per round
    pass: u32, repair: u64,
    focus: Option<Focus>, focus_rounds: Vec<u32>, repair_offset: u64,
    /// the last finish words' units have all had their rounds and the normal schedule is back (`focus_done`)
    focus_done: bool,
}

impl PacketSource {
    pub fn new(prepared: Prepared) -> io::Result<PacketSource> {
        let manifest_bytes = prepared.manifest.encode();
        if manifest_bytes.len() >= 1 << 24 { return Err(io::Error::other("manifest too large for the header length field")); }
        let manifest_encoders = manifest_bytes.chunks(crate::file::CHUNK_BYTES as usize).map(LtEncoder::new).collect();
        let m = &prepared.manifest;
        let segs = m.segment_count() as usize;
        let units = m.finish_units() as usize;
        let chunks: Vec<(u16, u8)> = (1..=segs as u16).flat_map(|s| (0..m.chunk_count(s)).map(move |c| (s, c))).collect();
        let pass_order: BinaryHeap<Due> = chunks.iter().enumerate().map(|(idx, &(s, ch))| Due { next: 0, k: chunk_blocks(m.chunk_len(s, ch)) as u16, idx: idx as u32, key: idx as u32 }).collect();
        // a batch's file list first: the blocks holding it (systematic: block j is packet j of its chunk) open the
        // first pass, so a receiver lists the files at once (10,000 files, a 400 KB list: known at 4.4% of the
        // pass, at 47% in the balanced order), and they are sent again in their balanced places, so a receiver
        // that joins later is short of no more than in the balanced pass. The pass grows by the list's blocks
        // (4.6% for 10,000 files of 1 KB, nothing to see for photos). Sending its whole chunks first instead, and
        // not again, cost a receiver joining at 30% of the pass 71-74% more frames (equal-rate repair rounds fill
        // a chunk it missed whole only at one block per round); measured in docs/2026-10-30-repair-schedule.md.
        // A single file's pass is the balanced one, unchanged
        let mut front = Vec::new();
        if let Some(b) = &m.batch {
            for g in 0..=(b.index_len - 1) / crate::file::CHUNK_BYTES {
                let (segment, chunk) = chunks[g as usize];
                let held = (b.index_len - g * crate::file::CHUNK_BYTES).min(crate::file::CHUNK_BYTES);
                let n = held.div_ceil(BLOCK_BYTES as u64).min(chunk_blocks(m.chunk_len(segment, chunk)) as u64);
                front.extend((0..n).map(|pid| PacketId { segment, chunk, pid: pid as u16 }));
            }
        }
        let blocks = pass_order.iter().map(|d| d.k as u64).sum();
        let mut src = PacketSource {
            prepared, manifest_bytes, manifest_encoders, manifest_cursor: (0, 0), pass_order, chunks, front, front_sent: 0,
            list_spacing: 0, list_next: 0, plain: 0, list_cursor: 0, blocks, packets_per_frame: DEFAULT_PACKETS_PER_FRAME,
            pass: 0, repair: 0,
            focus: None, focus_rounds: vec![0; units], repair_offset: fresh_repair_offset(), focus_done: false,
        };
        src.set_packets_per_frame(DEFAULT_PACKETS_PER_FRAME);
        Ok(src)
    }
    /// The packets of one data frame (the sender's tile), which sets how often a batch's file list goes out again:
    /// all of it once every LIST_EVERY_FRAMES data frames, one block every `spacing` packets, at most 2% of them
    /// (LIST_MIN_SPACING). The spacing carries a fraction (0.618), so the repeats move through the slots of the frame
    /// (a part of the frame that is never read misses one now and then, not all of them), and no frame carries two
    /// list blocks unless a frame holds more than 50 packets. A list that would take more than half a pass to come
    /// round at 2% (its blocks a share of the stream above ~1%: many tiny files) is not repeated: a receiver that
    /// joins late decodes its chunk from the repair rounds before then, and the repeats would only cost. Call it
    /// before the first packet; a single file sends no list.
    pub fn set_packets_per_frame(&mut self, packets: usize) {
        self.packets_per_frame = packets.max(1);
        let l = self.front.len() as u64;
        let spacing = if l == 0 { 0 } else { (LIST_EVERY_FRAMES * self.packets_per_frame as u64).div_ceil(l).max(LIST_MIN_SPACING) };
        self.list_spacing = if spacing == 0 || l * spacing > (self.blocks + l) / 2 { 0 } else { spacing << 16 | 0x9E37 };
        self.list_next = self.list_spacing;
    }
    pub fn manifest(&self) -> &Manifest { &self.prepared.manifest }
    /// A batch's file index as sent; None for one file.
    pub fn batch_index(&self) -> Option<&super::batch::Index> { self.prepared.index.as_ref() }
    #[cfg(test)]
    pub(crate) fn prepared(&self) -> &Prepared { &self.prepared }
    /// Sends only repair packets of `units` (finish-code units, `Manifest::finish_units`), round-robin: a
    /// round per segment unit (one packet per chunk), a packet per chunk unit. The systematic pass is skipped
    /// (the receiver already has some packets; repair ids are new to it), and the rounds start at this
    /// source's own offset, so a restarted sender sends ids it has not sent before. Units out of range are
    /// dropped; None, or nothing left, is the normal schedule again, where it left off. Each unit gets
    /// FOCUS_ROUNDS_PER_BLOCK rounds per block of its largest chunk; once every one has had them the sender
    /// leaves focus by itself (`focus_done`) and sends everything again, where the normal schedule left off.
    pub fn set_focus(&mut self, units: Option<Vec<u32>>) {
        let n = self.prepared.manifest.finish_units();
        self.focus_done = false;
        self.focus = units.map(|mut u| { u.retain(|&x| x < n); u.sort_unstable(); u.dedup(); u })
            .filter(|u| !u.is_empty()).map(|units| {
                let left = units.iter().map(|&u| self.unit_blocks(u) * FOCUS_ROUNDS_PER_BLOCK).collect();
                Focus { units, pos: 0, chunk: 0, left }
            });
    }
    /// Blocks of the largest chunk of a finish unit (a segment unit's first chunk, or the chunk unit itself).
    fn unit_blocks(&self, u: u32) -> u32 {
        let m = &self.prepared.manifest;
        let (seg, chunk) = if m.segment_count() >= 2 { (u as u16 + 1, 0) } else { (1, u as u8) };
        chunk_blocks(m.chunk_len(seg, chunk)) as u32
    }
    /// True once the last finish words' units have all had their repair rounds and the sender has gone back to
    /// sending everything by itself; false while they are sent, after "send everything" and before any words.
    pub fn focus_done(&self) -> bool { self.focus_done }
    /// Replaces the clock-drawn start of the focus repair rounds (a sender without a clock passes its own).
    pub fn set_repair_offset(&mut self, offset: u64) { self.repair_offset = offset; }
    /// The units being sent in focus, if any.
    pub fn focus(&self) -> Option<&[u32]> { self.focus.as_ref().map(|f| f.units.as_slice()) }
    /// Length of the encoded manifest (what a manifest frame's header announces).
    pub fn manifest_bytes_len(&self) -> usize { self.manifest_bytes.len() }
    /// Packets in one systematic pass over the whole stream (a batch's: with its file list's blocks sent first, and
    /// again as often as `set_packets_per_frame` says).
    pub fn data_packets_per_pass(&self) -> u64 {
        let packets = self.blocks;
        // the list's repeats due by the pass's last packet (`next_data_id` checks one before each packet of the
        // schedule, and once more before the first repair packet)
        let repeats = if self.list_spacing == 0 { 0 } else { (packets << 16) / self.list_spacing };
        packets + self.front.len() as u64 + repeats
    }
    fn next_focus_id(&mut self) -> Option<PacketId> {
        let m = &self.prepared.manifest;
        let f = self.focus.as_mut()?;
        let u = f.units[f.pos];
        let by_segment = m.segment_count() >= 2;
        let (seg, chunk) = if by_segment { (u as u16 + 1, f.chunk) } else { (1, u as u8) };
        let k = chunk_blocks(m.chunk_len(seg, chunk)) as u64;
        let round = self.repair_offset + self.focus_rounds[u as usize] as u64;
        let pid = (k + round % (4096 - k)) as u16;
        f.chunk += 1;
        if !by_segment || f.chunk >= m.chunk_count(seg) {
            f.chunk = 0; self.focus_rounds[u as usize] += 1; f.left[f.pos] -= 1;
            // the next unit with rounds left; none: back to the normal schedule, by itself
            match (1..=f.units.len()).map(|i| (f.pos + i) % f.units.len()).find(|&p| f.left[p] > 0) {
                Some(p) => f.pos = p,
                None => { self.focus = None; self.focus_done = true; }
            }
        }
        Some(PacketId { segment: seg, chunk, pid })
    }
    fn next_data_id(&mut self) -> PacketId {
        if let Some(id) = self.next_focus_id() { return id; }
        let m = &self.prepared.manifest;
        if self.pass == 0 && self.front_sent < self.front.len() { self.front_sent += 1; return self.front[self.front_sent - 1]; }
        // the file list again (systematic ids: a receiver that has a block already drops it), first pass and repair
        // rounds alike; not during finish words (above), which name the list's unit while its chunk is short
        if self.list_spacing > 0 && self.plain << 16 >= self.list_next {
            self.list_next += self.list_spacing;
            let id = self.front[self.list_cursor];
            self.list_cursor = (self.list_cursor + 1) % self.front.len();
            return id;
        }
        self.plain += 1;
        if self.pass == 0 {
            // coverage-balanced: block j of a chunk of k blocks goes out at (2j+1)/2k of the pass, so every stretch
            // of the pass carries every chunk's blocks in proportion to its size. A receiver that joins late, or
            // loses a run of frames, is short by the same share in every chunk, which the blind repair rounds
            // (one packet per chunk per round) make up together; a segment-by-segment pass left the segments it
            // missed wholly to the rounds, ~1 pass more after a late join. Consecutive packets fall in
            // different chunks (a sweep's order, rotated: `sweep_start`): a lost frame costs each chunk a block or a few
            if let Some(mut d) = self.pass_order.pop() {
                let (segment, chunk) = self.chunks[d.idx as usize];
                let id = PacketId { segment, chunk, pid: d.next };
                d.next += 1;
                d.key = (d.idx + sweep_start(d.next as u64, self.chunks.len() as u32)) % self.chunks.len() as u32;
                if d.next < d.k { self.pass_order.push(d); }
                if self.pass_order.is_empty() { self.pass = 1; }
                return id;
            }
            self.pass = 1;
        }
        // repair rounds: every chunk once per round (equal rates: docs/2026-10-30-repair-schedule.md), round n's
        // packets the n-th repair id of each chunk, the round's order rotated as the pass's sweeps
        let c = self.chunks.len() as u32;
        let (round, i) = (self.repair / c as u64, (self.repair % c as u64) as u32);
        self.repair += 1;
        let (segment, chunk) = self.chunks[((i + sweep_start(round, c)) % c) as usize];
        let k = chunk_blocks(m.chunk_len(segment, chunk)) as u64;
        PacketId { segment, chunk, pid: (k + round % (4096 - k)) as u16 }
    }
    pub fn next_data_packet(&mut self) -> io::Result<Packet> {
        let id = self.next_data_id();
        // built from the stream's blocks: consecutive packets fall in different chunks, so no chunk is kept whole
        let k = chunk_blocks(self.prepared.manifest.chunk_len(id.segment, id.chunk));
        let mut payload = [0u8; BLOCK_BYTES];
        for i in neighbors(id.pid as u32, k) { self.prepared.xor_block(id.segment, id.chunk, i as usize, &mut payload)?; }
        Ok(Packet { id, payload })
    }
    pub fn next_manifest_packet(&mut self) -> Packet {
        let (chunk, pid) = self.manifest_cursor;
        let enc = &self.manifest_encoders[chunk as usize];
        let payload = enc.packet(pid as u32);
        let id = PacketId { segment: 0, chunk, pid };
        let (mut nchunk, mut npid) = (chunk, pid + 1);
        if npid as usize >= enc.k() * 2 || npid >= 4095 { npid = 0; nchunk = (chunk + 1) % self.manifest_encoders.len() as u8; }
        self.manifest_cursor = (nchunk, npid);
        Packet { id, payload }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file::{Prepared, CHUNKS_PER_SEGMENT};
    use std::collections::HashMap;

    fn source(bytes: usize) -> PacketSource {
        let mut s = 7u64;
        let data: Vec<u8> = (0..bytes).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s as u8 }).collect();
        PacketSource::new(Prepared::from_bytes("x.bin", data).unwrap()).unwrap()
    }
    #[test]
    fn systematic_pass_sends_every_block_once() {
        let mut src = source(40 * 1024 * 1024 + 12345); // 3 segments, the last one short
        let n = src.data_packets_per_pass();
        let mut seen = std::collections::HashSet::new();
        for _ in 0..n { let id = src.next_data_id(); assert!(id.pid < 1024 && seen.insert(id), "{id:?} repeated"); }
        assert_eq!(seen.len() as u64, n);
        let m = src.manifest().clone();
        for seg in 1..=m.segment_count() as u16 { for c in 0..m.chunk_count(seg) { for pid in 0..chunk_blocks(m.chunk_len(seg, c)) as u16 { assert!(seen.contains(&PacketId { segment: seg, chunk: c, pid })); } } }
    }
    #[test]
    fn a_lost_frame_costs_each_chunk_only_a_few_blocks() {
        // a frame carries 82 consecutive packets (1 px b/w): taken from one chunk, a lost frame leaves that chunk
        // 82 blocks short, and the blind repair rounds give one chunk ~1 packet per 12 frames: a long tail
        let mut src = source(32 * 1024 * 1024); // two full segments (a short last segment has fewer chunks to spread over)
        let n = src.data_packets_per_pass() as usize;
        let ids: Vec<PacketId> = (0..n).map(|_| src.next_data_id()).collect();
        for w in ids.chunks(82) {
            let mut per: HashMap<(u16, u8), usize> = HashMap::new();
            for id in w { *per.entry((id.segment, id.chunk)).or_insert(0) += 1; }
            let worst = per.values().max().copied().unwrap_or(0);
            assert!(worst <= 82usize.div_ceil(CHUNKS_PER_SEGMENT as usize) + 1, "a frame takes {worst} blocks of one chunk");
        }
    }
    #[test]
    fn every_stretch_of_the_pass_covers_every_chunk_by_its_size() {
        // 3 segments, the last one 9 chunks with a 13-block last chunk: after any number n of the pass's packets,
        // every chunk has had n * k / (blocks in the stream) of them, within 2 (a late join, or a run of lost
        // frames, leaves every chunk short by the same share)
        let mut src = source(40 * 1024 * 1024 + 12345);
        let m = src.manifest().clone();
        let total = src.data_packets_per_pass();
        let mut sent: HashMap<(u16, u8), u64> = HashMap::new();
        for n in 1..=total {
            let id = src.next_data_id();
            *sent.entry((id.segment, id.chunk)).or_insert(0) += 1;
            if n % 41 != 0 && n != total { continue; }
            for seg in 1..=m.segment_count() as u16 { for c in 0..m.chunk_count(seg) {
                let (k, got) = (chunk_blocks(m.chunk_len(seg, c)) as f64, *sent.get(&(seg, c)).unwrap_or(&0) as f64);
                assert!((got - n as f64 * k / total as f64).abs() <= 2.0, "after {n} packets chunk {seg}/{c} (k {k}) had {got}");
            } }
        }
        assert_eq!(src.next_data_id().pid, 1024, "repair rounds follow the pass");
    }
    #[test]
    fn every_packet_slot_carries_every_chunk() {
        // frames of P packets, files of C full chunks with a common factor (the fixed order put chunk s mod C in
        // slot s of every frame, pass and repair rounds alike: a part of the frame never read starved those
        // chunks). Over the pass, and over as many repair rounds, every (slot, chunk) pair carries its share
        // 1024 / P within a factor of two
        for (c, ps) in [(4usize, [2usize, 4, 8]), (6, [3, 4, 6]), (8, [2, 4, 8])] {
            for p in ps {
                let mut src = source(c << 20);
                assert_eq!(src.data_packets_per_pass(), 1024 * c as u64);
                for phase in ["pass", "repair rounds"] {
                    let mut n = vec![vec![0u32; c]; p];
                    for t in 0..1024 * c {
                        let id = src.next_data_id();
                        assert_eq!(id.pid >= 1024, phase != "pass");
                        n[t % p][id.chunk as usize] += 1;
                    }
                    let share = 1024.0 / p as f64;
                    for (slot, row) in n.iter().enumerate() { for (chunk, &got) in row.iter().enumerate() {
                        assert!((got as f64) > share / 2.0 && (got as f64) < share * 2.0, "{c} chunks, {p} packets per frame, {phase}: slot {slot} carried chunk {chunk} {got} times (share {share:.0})");
                    } }
                }
            }
        }
    }
    #[test]
    fn repair_rounds_send_every_chunk_once_per_round_in_order_of_ids() {
        // equal rates: round n is every chunk's n-th repair id, whatever the order within the round
        let mut src = source(40 * 1024 * 1024 + 12345);
        let m = src.manifest().clone();
        for _ in 0..src.data_packets_per_pass() { src.next_data_id(); }
        let chunks: Vec<(u16, u8)> = (1..=m.segment_count() as u16).flat_map(|s| (0..m.chunk_count(s)).map(move |c| (s, c))).collect();
        for round in 0..5u16 {
            let mut got: Vec<PacketId> = (0..chunks.len()).map(|_| src.next_data_id()).collect();
            got.sort_by_key(|id| (id.segment, id.chunk));
            let want: Vec<PacketId> = chunks.iter().map(|&(segment, chunk)| PacketId { segment, chunk, pid: chunk_blocks(m.chunk_len(segment, chunk)) as u16 + round }).collect();
            assert_eq!(got, want, "round {round}");
        }
    }
    #[test]
    fn focus_ends_by_itself_after_its_rounds() {
        // 3 segments, the last one 12345 bytes (13 blocks): segment 2 gets 2 x 1024 rounds of its 16 chunks, segment
        // 3 2 x 13 rounds of its one chunk (then the rest of segment 2's rounds alone), and the sender is back on the
        // normal schedule by itself
        let mut src = source(32 * 1024 * 1024 + 12345);
        let m = src.manifest().clone();
        let (r2, r3) = (FOCUS_ROUNDS_PER_BLOCK as usize * 1024, FOCUS_ROUNDS_PER_BLOCK as usize * 13);
        let total = r2 * CHUNKS_PER_SEGMENT as usize + r3;
        src.set_focus(Some(vec![2, 1]));
        assert!(!src.focus_done());
        let mut seen = std::collections::HashSet::new();
        let mut per = [0usize; 4];
        for i in 0..total {
            assert_eq!(src.focus(), Some(&[1u32, 2][..]), "packet {i}: still the named parts");
            let id = src.next_data_id();
            assert!(id.segment >= 2 && id.pid as usize >= chunk_blocks(m.chunk_len(id.segment, id.chunk)), "{id:?}");
            assert!(seen.insert(id), "{id:?} repeated");
            per[id.segment as usize] += 1;
        }
        assert_eq!((per[2], per[3]), (r2 * CHUNKS_PER_SEGMENT as usize, r3));
        assert!(src.focus_done() && src.focus().is_none(), "the sender left the words by itself");
        // the normal schedule, from its start: the first pass (no packet was taken from it)
        assert!(src.next_data_id().pid < 1024);
        // the same words again: the rounds carry on, so every repair id is new
        src.set_focus(Some(vec![1]));
        assert!(!src.focus_done());
        for _ in 0..200 { let id = src.next_data_id(); assert!(id.segment == 2 && seen.insert(id), "{id:?} sent before"); }
        src.set_focus(None);
        assert!(!src.focus_done() && src.focus().is_none());
    }
    /// A batch of `files` noise files of `size` bytes each (a file list of a block or a few), as a source with frames
    /// of `ppf` packets.
    fn batch_source(files: usize, size: usize, ppf: usize) -> PacketSource {
        let mut s = 11u64;
        let sources = (0..files).map(|i| (format!("file-{i:03}.bin"), crate::file::BatchSource::Bytes((0..size).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s as u8 }).collect()))).collect();
        let mut src = PacketSource::new(Prepared::batch("batch", sources).unwrap()).unwrap();
        src.set_packets_per_frame(ppf);
        src
    }
    /// The systematic ids of the file list (the blocks the first pass opens with).
    fn list_ids(src: &PacketSource) -> std::collections::HashSet<PacketId> {
        let b = src.manifest().batch.clone().unwrap();
        assert!(b.index_len <= crate::file::CHUNK_BYTES, "a list within the first chunk");
        (0..b.index_len.div_ceil(BLOCK_BYTES as u64) as u16).map(|pid| PacketId { segment: 1, chunk: 0, pid }).collect()
    }
    #[test]
    fn the_file_list_goes_out_again_through_the_whole_transfer() {
        // 3 files of 3 MB (a list of one block) and 60 of 150 KB (a few blocks), frames of 24 packets (2 px b/w): after
        // the list's blocks open the pass, every block of the list goes out again within LIST_EVERY_FRAMES data
        // frames, through the pass and the repair rounds alike, never two list blocks in one frame, and the repeats
        // take at most 2% of the packets; the pass (data_packets_per_pass) counts the repeats it holds
        for (files, size) in [(3usize, 3_000_000usize), (60, 150_000)] {
            let mut src = batch_source(files, size, 24);
            let list = list_ids(&src);
            let per_pass = src.data_packets_per_pass();
            let total = 3 * per_pass as usize;
            let ids: Vec<PacketId> = (0..total).map(|_| src.next_data_id()).collect();
            let pass_end = ids.iter().position(|id| id.pid >= 1024 || id.pid as usize >= chunk_blocks(src.manifest().chunk_len(id.segment, id.chunk))).unwrap();
            assert_eq!(pass_end as u64, per_pass, "{files} files: the pass is as long as it says");
            let repeats: Vec<usize> = ids.iter().enumerate().skip(list.len()).filter(|(_, id)| list.contains(id)).map(|(i, _)| i).collect();
            // in the balanced pass each list block also goes out once in its place
            let extra = repeats.len() - list.len();
            assert!((extra as f64) <= 0.02 * total as f64, "{files} files: {extra} repeats in {total} packets");
            let cycle = (LIST_EVERY_FRAMES * 24) as usize;
            for id in &list {
                let at: Vec<usize> = ids.iter().enumerate().filter(|(_, x)| *x == id).map(|(i, _)| i).collect();
                for w in at.windows(2) { assert!(w[1] - w[0] <= cycle + 24, "{files} files: {id:?} at {} then {}", w[0], w[1]); }
                assert!(total - at.last().unwrap() <= cycle + 24, "{files} files: {id:?} last at {} of {total}", at.last().unwrap());
            }
            // (the balanced pass sends block j of the list's chunk at (2j + 1) / 2048 of the pass: a few blocks close
            // together in its first 1%, its own order; the repeats come after)
            let spread: Vec<usize> = repeats.into_iter().filter(|&i| i as u64 > per_pass / 100).collect();
            for w in spread.windows(2) { assert!(w[1] - w[0] >= 24, "{files} files: list blocks at {} and {}", w[0], w[1]); }
        }
    }
    #[test]
    fn a_list_as_long_as_its_files_is_not_repeated() {
        // 2,000 files of 200 B: a list of about 100 blocks in a stream of about 500. At 2% of the packets its blocks
        // would come round once every 5,000 packets, ten passes, long after a receiver that joined late has decoded
        // the list's chunk from the repair rounds: such a list goes out first and in place only (nothing wasted)
        let mut src = batch_source(2000, 200, 24);
        let list = list_ids(&src);
        let per_pass = src.data_packets_per_pass() as usize;
        let ids: Vec<PacketId> = (0..3 * per_pass).map(|_| src.next_data_id()).collect();
        assert_eq!(ids.iter().filter(|id| list.contains(id)).count(), 2 * list.len(), "first and in place, no more");
    }
    #[test]
    fn finish_words_send_no_list_blocks() {
        // the words name the list's unit when the receiver lacks it (its chunk is not whole), so focus rounds carry
        // repair packets only; once the words have had their rounds, the list goes out again with everything else
        let mut src = batch_source(3, 3_000_000, 24);
        let list = list_ids(&src);
        for _ in 0..src.data_packets_per_pass() { src.next_data_id(); }
        src.set_focus(Some(vec![0, 5]));
        let mut n = 0;
        while src.focus().is_some() {
            let id = src.next_data_id();
            assert!(id.pid as usize >= chunk_blocks(src.manifest().chunk_len(id.segment, id.chunk)), "focus sent {id:?}");
            n += 1;
        }
        assert!(n > 1000 && src.focus_done());
        let after = (0..(LIST_EVERY_FRAMES as usize + 1) * 24).filter(|_| list.contains(&src.next_data_id())).count();
        assert!(after >= 1, "the list again after the words");
    }
    #[test]
    fn packets_are_the_fountain_packets_of_their_chunk() {
        // built from single blocks, every packet is what an encoder over its whole chunk gives: the first pass,
        // then repair rounds (degree 32-96, and the 13-block last chunk's own repair degrees)
        let bytes = 17 * 1024 * 1024 + 12345;
        let mut s = 7u64;
        let data: Vec<u8> = (0..bytes).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s as u8 }).collect();
        let mut src = PacketSource::new(Prepared::from_bytes("x.bin", data.clone()).unwrap()).unwrap();
        let m = src.manifest().clone();
        let mut enc: HashMap<(u16, u8), LtEncoder> = HashMap::new();
        for _ in 0..src.data_packets_per_pass() + 3 * 18 {
            let p = src.next_data_packet().unwrap();
            let e = enc.entry((p.id.segment, p.id.chunk)).or_insert_with(|| {
                let off = m.chunk_offset(p.id.segment, p.id.chunk) as usize;
                LtEncoder::new(&data[off..off + m.chunk_len(p.id.segment, p.id.chunk)])
            });
            assert!(p.payload == e.packet(p.id.pid as u32), "{:?}", p.id);
        }
    }
}
