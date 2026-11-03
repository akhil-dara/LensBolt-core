//! Frame layout: the pitch-independent chrome (border, markers, header) in 8 px modules, and the data
//! grid of a given pitch with its pilot cells and its frame-wide codewords (format v8).

use super::format::{Alphabet, CellFormat, MODULE_PX};

pub const MARKER_MODULES: u32 = 7;
/// Markers sit two modules in from the border (a black guard module between the border and the marker's
/// white outer ring, or the ring merges with white border modules and the run-ratio detector loses
/// it); the zone (guard + marker + inner guard) is 10 modules = 80 px square.
pub const MARKER_OFFSET: u32 = 2;
pub const ZONE_MODULES: u32 = MARKER_OFFSET + MARKER_MODULES + 1;
/// Header: from module row 1, module column 10 (between the top marker zones), 104 cells per row in two rows
/// on tiles >= 992 px wide (the v8 layout); narrower tiles (a tile matched to a Full HD camera at 1 px) fill
/// the rows between the zones and wrap onto more rows.
pub const HEADER_ROW0: u32 = 1;
pub const HEADER_COL0: u32 = ZONE_MODULES;
pub const HEADER_ROW_CELLS: u32 = 104;
pub const HEADER_ROWS: u32 = 2;
/// The shortest header row a tile may have (4 rows of 52 hold the 208 cells).
pub const MIN_HEADER_ROW_CELLS: u32 = 52;
/// Pilots per 8x8 block at these (local row, local col) positions.
pub const BLOCK: u32 = 8;
pub const PILOT_LOCAL: [(u32, u32); 4] = [(2, 2), (2, 6), (6, 2), (6, 6)];
pub const MIN_SIDE_PX: u32 = 256;
/// A header row needs zone + at least 52 cells + zone modules.
pub const MIN_WIDTH_PX: u32 = (HEADER_COL0 + MIN_HEADER_ROW_CELLS + ZONE_MODULES) * MODULE_PX;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FlatCell { Data { index: u32 }, Pilot { colour: u8 }, Unused }

/// The part of the frame that does not depend on the pitch.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Chrome { width_px: u32, height_px: u32 }

impl Chrome {
    pub const HEADER_CELLS: u32 = HEADER_ROW_CELLS * HEADER_ROWS;
    pub fn new(width_px: u32, height_px: u32) -> Option<Chrome> {
        if width_px % MODULE_PX != 0 || height_px % MODULE_PX != 0 || width_px < MIN_WIDTH_PX || height_px < MIN_SIDE_PX { return None; }
        Some(Chrome { width_px, height_px })
    }
    pub fn width_px(&self) -> u32 { self.width_px }
    pub fn height_px(&self) -> u32 { self.height_px }
    pub fn modules(&self) -> (u32, u32) { (self.width_px / MODULE_PX, self.height_px / MODULE_PX) }
    /// Header cells per row: 104, or the room between the top marker zones on a narrower tile.
    pub fn header_row_cells(&self) -> u32 { (self.modules().0 - 2 * ZONE_MODULES).min(HEADER_ROW_CELLS) }
    /// Header rows: 2 on tiles >= 992 px wide, more on narrower ones.
    pub fn header_rows(&self) -> u32 { Self::HEADER_CELLS.div_ceil(self.header_row_cells()) }
    fn module_center(mx: u32, my: u32) -> (f32, f32) {
        ((mx * MODULE_PX) as f32 + MODULE_PX as f32 / 2.0, (my * MODULE_PX) as f32 + MODULE_PX as f32 / 2.0)
    }
    /// TL, TR, BR, BL marker centres in screen px.
    pub fn marker_centers(&self) -> [(f32, f32); 4] {
        let (mw, mh) = self.modules();
        let m = MARKER_OFFSET + MARKER_MODULES / 2;
        [Self::module_center(m, m), Self::module_center(mw - 1 - m, m), Self::module_center(mw - 1 - m, mh - 1 - m), Self::module_center(m, mh - 1 - m)]
    }
    /// Border modules clockwise from the top-left corner (same walk as geometry::Grid::border_position).
    pub fn border_module_count(&self) -> u32 { let (mw, mh) = self.modules(); 2 * mw + 2 * (mh - 2) }
    pub fn border_module(&self, index: u32) -> (u32, u32) {
        let (c, r) = self.modules();
        if index < c { return (index, 0); }
        let i = index - c;
        if i < r - 1 { return (c - 1, 1 + i); }
        let i = i - (r - 1);
        if i < c - 1 { return (c - 2 - i, r - 1); }
        let i = i - (c - 1);
        (0, r - 2 - i)
    }
    pub fn border_module_center(&self, index: u32) -> (f32, f32) { let (mx, my) = self.border_module(index); Self::module_center(mx, my) }
    pub fn border_is_white(index: u32) -> bool { index % 2 == 0 }
    /// Module of header cell `index` of `copy` (0 top-left, 1 point-mirrored bottom-right).
    pub fn header_module(&self, copy: u32, index: u32) -> (u32, u32) {
        let n = self.header_row_cells();
        let (row, col) = (HEADER_ROW0 + index / n, HEADER_COL0 + index % n);
        if copy == 0 { (col, row) } else { let (mw, mh) = self.modules(); (mw - 1 - col, mh - 1 - row) }
    }
    pub fn header_cell_center(&self, copy: u32, index: u32) -> (f32, f32) { let (mx, my) = self.header_module(copy, index); Self::module_center(mx, my) }
    /// White unless the ring at distance 1 from the marker's edge.
    pub fn marker_module_paint(local_x: u32, local_y: u32) -> bool {
        let m = MARKER_MODULES - 1;
        local_x.min(m - local_x).min(local_y).min(m - local_y) != 1
    }
    /// If module (mx, my) belongs to a marker, its local coordinates inside that marker.
    pub fn marker_local(&self, mx: u32, my: u32) -> Option<(u32, u32)> {
        let (mw, mh) = self.modules();
        let local = |m: u32, n: u32| -> Option<u32> {
            if m >= MARKER_OFFSET && m < MARKER_OFFSET + MARKER_MODULES { Some(m - MARKER_OFFSET) }
            else if m + MARKER_OFFSET + MARKER_MODULES >= n && m + MARKER_OFFSET < n { Some(m + MARKER_OFFSET + MARKER_MODULES - n) }
            else { None }
        };
        Some((local(mx, mw)?, local(my, mh)?))
    }
    /// Data area: inside the border and the header rows, as (x0, y0, w, h) in px.
    pub fn data_rect(&self) -> (u32, u32, u32, u32) {
        let top = MODULE_PX * (1 + self.header_rows());
        (MODULE_PX, top, self.width_px - 2 * MODULE_PX, self.height_px - 2 * top)
    }
    /// Whether the rect intersects one of the four 72x72 px marker zones.
    pub fn in_marker_zone(&self, x: u32, y: u32, w: u32, h: u32) -> bool {
        let z = ZONE_MODULES * MODULE_PX;
        let xs = x < z || x + w > self.width_px - z;
        let ys = y < z || y + h > self.height_px - z;
        xs && ys
    }
}

/// Vertical data regions (about 100 cells wide) set the data cell order only: since v8 every codeword spans
/// the whole frame, the order no longer matters for decoding, and keeping it keeps real cell dumps valid.
pub fn region_count(cols: u32) -> u32 { ((cols + 50) / 100).max(1) }

pub struct FlatGrid {
    chrome: Chrome, format: CellFormat, cols: u32, rows: u32, origin_x: u32, origin_y: u32,
    cells: Vec<FlatCell>, data_positions: Vec<(u32, u32)>, pilot_positions: Vec<(u32, u32)>,
    data_cells: u32, regions: u32, codewords: usize,
    /// `cell_kinds`, built on first use
    kinds: std::sync::OnceLock<Vec<u8>>,
    /// the fast reader's equaliser roles of a read with every cell read (`eq_roles`), built on first use
    eq_roles: std::sync::OnceLock<(Vec<u8>, Vec<u8>)>,
}

/// `FlatGrid::cell_kinds`: a cell that is neither data nor a pilot (a marker zone)...
pub const CELL_NONE: u8 = 0;
/// ...a data cell that carries payload (index below `used_data_cells`)...
pub const CELL_DATA: u8 = 1;
/// ...a data cell past the payload (composed as level 0)...
pub const CELL_PAST: u8 = 2;
/// ...a pilot.
pub const CELL_PILOT: u8 = 3;

impl FlatGrid {
    /// The grid for a tile size and format, built once per process: building a 1 px grid (cell map, data and
    /// pilot positions) costs more than reading every cell of a pixel-exact capture. The last few are kept.
    pub fn shared(width_px: u32, height_px: u32, format: CellFormat) -> Option<std::sync::Arc<FlatGrid>> {
        type Cache = Vec<((u32, u32, CellFormat), std::sync::Arc<FlatGrid>)>;
        static CACHE: std::sync::Mutex<Cache> = std::sync::Mutex::new(Vec::new());
        let key = (width_px, height_px, format);
        if let Some((_, g)) = CACHE.lock().ok()?.iter().find(|(k, _)| *k == key) { return Some(g.clone()); }
        let g = std::sync::Arc::new(FlatGrid::try_new(width_px, height_px, format)?);
        let mut c = CACHE.lock().ok()?;
        if c.len() >= 4 { c.remove(0); }
        c.push((key, g.clone()));
        Some(g)
    }
    pub fn try_new(width_px: u32, height_px: u32, format: CellFormat) -> Option<FlatGrid> {
        let chrome = Chrome::new(width_px, height_px)?;
        if !super::format::PITCHES.contains(&format.pitch) { return None; }
        let p = format.pitch as u32;
        let (x0, y0, w, h) = chrome.data_rect();
        let (cols, rows) = (w / p, h / p);
        if cols == 0 || rows == 0 { return None; }
        let origin_x = x0 + (w % p) / 2;
        let origin_y = y0 + (h % p) / 2;
        let mut cells = vec![FlatCell::Unused; (cols * rows) as usize];
        let mut positions: Vec<(u32, u32)> = Vec::new();
        let mut pilot_positions = Vec::new();
        let k = format.alphabet;
        for r in 0..rows {
            for c in 0..cols {
                if chrome.in_marker_zone(origin_x + c * p, origin_y + r * p, p, p) { continue; }
                let (lr, lc) = (r % BLOCK, c % BLOCK);
                if let Some(kk) = PILOT_LOCAL.iter().position(|&q| q == (lr, lc)) {
                    cells[(r * cols + c) as usize] = FlatCell::Pilot { colour: FlatGrid::pilot_colour(r / BLOCK, c / BLOCK, kk as u32, k) };
                    pilot_positions.push((c, r));
                } else {
                    positions.push((c, r));
                }
            }
        }
        // data order: region by region in a snake (even regions top to bottom, odd ones bottom to top), rows
        // left to right inside a region
        let regions = region_count(cols);
        let mut by_region: Vec<Vec<(u32, u32)>> = vec![Vec::new(); regions as usize];
        for &(c, r) in &positions { by_region[(c * regions / cols) as usize].push((c, r)); }
        let mut data_positions: Vec<(u32, u32)> = Vec::with_capacity(positions.len());
        for (q, v) in by_region.iter_mut().enumerate() {
            if q % 2 == 1 { v.sort_by_key(|&(c, r)| (std::cmp::Reverse(r), c)); }
            data_positions.extend_from_slice(v);
        }
        for (i, &(c, r)) in data_positions.iter().enumerate() { cells[(r * cols + c) as usize] = FlatCell::Data { index: i as u32 }; }
        let data_cells = data_positions.len() as u32;
        // v8: the frame's data cells are one byte stream of whole codewords; a grid without room for one
        // packet is not a format
        let codewords = data_positions.len() * k.bits() as usize / 8 / 255;
        if codewords * format.data_len() < crate::packet::PACKET_BYTES { return None; }
        Some(FlatGrid { chrome, format, cols, rows, origin_x, origin_y, cells, data_positions, pilot_positions, data_cells, regions, codewords, kinds: std::sync::OnceLock::new(), eq_roles: std::sync::OnceLock::new() })
    }
    /// Every cell's kind (`CELL_NONE`, `CELL_DATA`, `CELL_PAST`, `CELL_PILOT`), row-major, built once per grid
    /// (the reads of a shared grid do not walk its cell map).
    pub fn cell_kinds(&self) -> &[u8] {
        self.kinds.get_or_init(|| {
            let used = self.used_data_cells() as u32;
            self.cells.iter().map(|c| match *c { FlatCell::Data { index } if index < used => CELL_DATA, FlatCell::Data { .. } => CELL_PAST, FlatCell::Pilot { .. } => CELL_PILOT, FlatCell::Unused => CELL_NONE }).collect()
        })
    }
    /// The fast reader's equaliser tables for a read in which every cell is read and every pilot's tag known
    /// (`fast::equalise_bw`): `build` makes them on the first call for this grid.
    pub(crate) fn eq_roles(&self, build: impl FnOnce() -> (Vec<u8>, Vec<u8>)) -> &(Vec<u8>, Vec<u8>) { self.eq_roles.get_or_init(build) }
    pub fn chrome(&self) -> &Chrome { &self.chrome }
    pub fn format(&self) -> CellFormat { self.format }
    pub fn width_px(&self) -> u32 { self.chrome.width_px }
    pub fn height_px(&self) -> u32 { self.chrome.height_px }
    pub fn cols(&self) -> u32 { self.cols }
    pub fn rows(&self) -> u32 { self.rows }
    pub fn cell(&self, col: u32, row: u32) -> FlatCell { self.cells[(row * self.cols + col) as usize] }
    pub fn cell_top_left(&self, col: u32, row: u32) -> (u32, u32) { let p = self.format.pitch as u32; (self.origin_x + col * p, self.origin_y + row * p) }
    pub fn cell_center(&self, col: u32, row: u32) -> (f32, f32) {
        let (x, y) = self.cell_top_left(col, row); let h = self.format.pitch as f32 / 2.0; (x as f32 + h, y as f32 + h)
    }
    pub fn data_cell_count(&self) -> u32 { self.data_cells }
    /// Data cells the codeword stream fills (the rest are 0).
    pub fn used_data_cells(&self) -> usize { (self.codewords * 255 * 8).div_ceil(self.format.alphabet.bits() as usize) }
    pub fn data_position(&self, index: u32) -> (u32, u32) { self.data_positions[index as usize] }
    pub fn pilot_positions(&self) -> &[(u32, u32)] { &self.pilot_positions }
    /// Pilot colours cycle through the alphabet so that neighbouring pilots differ and every colour is
    /// sampled across the frame.
    pub fn pilot_colour(bi: u32, bj: u32, k: u32, alphabet: Alphabet) -> u8 { ((7 * bi + 3 * bj + 5 * k) % alphabet.size() as u32) as u8 }
    /// Pilot colour on a given display frame: pilots 1 and 3 of every block carry the two low bits of the
    /// frame sequence (shifted by half the alphabet when set), so a straddled capture can tell its frames apart.
    pub fn pilot_colour_tagged(bi: u32, bj: u32, k: u32, alphabet: Alphabet, tag: u8) -> u8 {
        let base = FlatGrid::pilot_colour(bi, bj, k, alphabet) as u32;
        let bit = match k { 1 => tag & 1, 3 => (tag >> 1) & 1, _ => 0 } as u32;
        let size = alphabet.size() as u32;
        ((base + bit * (size / 2)) % size) as u8
    }
    /// The tagged colour of the pilot at (col, row) on a frame with sequence bits `tag`.
    pub fn pilot_colour_at(&self, col: u32, row: u32, tag: u8) -> u8 {
        let kk = PILOT_LOCAL.iter().position(|&q| q == (row % BLOCK, col % BLOCK)).unwrap_or(0) as u32;
        FlatGrid::pilot_colour_tagged(row / BLOCK, col / BLOCK, kk, self.format.alphabet, tag)
    }
    /// RS codewords per frame (N): stream byte b belongs to codeword b mod N.
    pub fn codewords(&self) -> usize { self.codewords }
    pub fn payload_bytes(&self) -> usize { self.codewords * self.format.data_len() }
    /// Packets per frame, packed contiguously over the payload (a packet may straddle two codewords).
    pub fn packets(&self) -> usize { self.payload_bytes() / crate::packet::PACKET_BYTES }
    pub fn regions(&self) -> u32 { self.regions }
    /// Alignment blocks (across, down).
    pub fn blocks(&self) -> (u32, u32) { (self.cols.div_ceil(BLOCK), self.rows.div_ceil(BLOCK)) }
}

#[cfg(test)]
mod tests {
    #[test]
    fn shared_grids_are_built_once() {
        use super::*;
        let f = CellFormat { pitch: 1, alphabet: Alphabet::Bw };
        let a = FlatGrid::shared(1024, 640, f).expect("1024x640 1px is a grid");
        let b = FlatGrid::shared(1024, 640, f).unwrap();
        assert!(std::sync::Arc::ptr_eq(&a, &b), "the second lookup builds the grid again");
        let c = FlatGrid::shared(1024, 640, CellFormat { pitch: 2, alphabet: Alphabet::Bw }).unwrap();
        assert!(!std::sync::Arc::ptr_eq(&a, &c));
        assert_eq!((a.cols(), a.rows()), (FlatGrid::try_new(1024, 640, f).unwrap().cols(), FlatGrid::try_new(1024, 640, f).unwrap().rows()));
    }
    #[test]
    fn narrow_tiles_wrap_the_header() {
        // a tile matched to a Full HD camera at 1 px: ~784 cells across; the header wraps onto more rows
        let c = Chrome::new(784, 440).expect("784x440 is a tile");
        let (mw, mh) = c.modules();
        let mut seen = std::collections::HashSet::new();
        for copy in 0..2 { for i in 0..Chrome::HEADER_CELLS {
            let (mx, my) = c.header_module(copy, i);
            assert!(mx < mw && my < mh, "copy {copy} cell {i} outside the tile");
            assert!(c.marker_local(mx, my).is_none() && !c.in_marker_zone(mx * MODULE_PX, my * MODULE_PX, MODULE_PX, MODULE_PX), "copy {copy} cell {i} in a marker zone");
            assert!(seen.insert((mx, my)), "copy {copy} cell {i} collides");
        } }
        let (_, top, _, h) = c.data_rect();
        let header_bottom = (0..Chrome::HEADER_CELLS).map(|i| c.header_module(0, i).1).max().unwrap();
        assert!(top >= (header_bottom + 1) * MODULE_PX && h > 0, "data starts below the header");
        // a 1024-wide tile keeps the v8 layout exactly
        let w = Chrome::new(1024, 1024).unwrap();
        assert_eq!((w.header_module(0, 0), w.header_module(0, 103), w.header_module(0, 104)), ((10, 1), (113, 1), (10, 2)));
        assert_eq!(w.data_rect().1, MODULE_PX * 3);
        assert!(Chrome::new(560, 320).is_none(), "too narrow for a header row of 52");
    }
    use super::*;
    use crate::flat::{Alphabet, CellFormat, MANIFEST_FORMAT};

    #[test]
    fn chrome_geometry_1024() {
        let c = Chrome::new(1024, 1024).unwrap();
        assert_eq!(c.modules(), (128, 128));
        assert_eq!(c.marker_centers()[0], (44.0, 44.0)); assert_eq!(c.marker_centers()[2], (980.0, 980.0));
        assert_eq!(c.border_module_count(), 2 * 128 + 2 * 126);
        assert_eq!(c.border_module_center(0), (4.0, 4.0)); assert_eq!(c.border_module_center(128), (1020.0, 12.0));
        assert!(Chrome::border_is_white(0) && !Chrome::border_is_white(1));
        assert_eq!(c.header_cell_center(0, 0), (84.0, 12.0)); assert_eq!(c.header_cell_center(0, 104), (84.0, 20.0));
        assert_eq!(c.header_cell_center(1, 0), (940.0, 1012.0));
        assert!(c.in_marker_zone(78, 78, 4, 4)); assert!(!c.in_marker_zone(80, 80, 4, 4)); assert!(c.in_marker_zone(942, 8, 4, 4));
        assert!(Chrome::marker_module_paint(0, 0) && !Chrome::marker_module_paint(1, 1) && Chrome::marker_module_paint(3, 3) && !Chrome::marker_module_paint(5, 2));
        assert_eq!(c.marker_local(2, 2), Some((0, 0))); assert_eq!(c.marker_local(125, 8), Some((6, 6))); assert_eq!(c.marker_local(1, 1), None); assert_eq!(c.marker_local(9, 9), None); assert_eq!(c.marker_local(0, 3), None);
        assert_eq!(c.data_rect(), (8, 24, 1008, 976));
        assert!(Chrome::new(853, 480).is_none()); assert!(Chrome::new(200, 200).is_none());
    }
    #[test]
    fn grid_1024_pitch4_c8() {
        let g = FlatGrid::try_new(1024, 1024, CellFormat { pitch: 4, alphabet: Alphabet::C8 }).unwrap();
        assert_eq!((g.cols(), g.rows()), (252, 244));
        assert_eq!(g.cell_top_left(0, 0), (8, 24));
        assert_eq!(g.cell_center(1, 1), (14.0, 30.0));
        assert_eq!(g.cell(0, 0), FlatCell::Unused); assert_eq!(g.cell(17, 13), FlatCell::Unused);
        assert_eq!(g.cell(18, 0), FlatCell::Data { index: 0 }); assert_eq!(g.cell(19, 0), FlatCell::Data { index: 1 });
        assert!(matches!(g.cell(18, 15), FlatCell::Data { .. }));
        assert_eq!(g.cell(18, 14), FlatCell::Pilot { colour: FlatGrid::pilot_colour(1, 2, 2, Alphabet::C8) }); // block (1,2), local (6,2) = k 2
        assert_eq!(g.cell(251, 243), FlatCell::Unused);
        // v8: the whole data area is one stream of codewords; packets fill the payload back to back
        let n = g.data_cell_count() as usize * 3 / 8 / 255;
        assert_eq!(g.codewords(), n); assert!(n >= 80, "{n} codewords (v7 had 80)");
        assert_eq!(g.payload_bytes(), n * 223); assert_eq!(g.packets(), n * 223 / 1030);
        assert!(g.packets() >= 17, "{} packets (v7 had 16)", g.packets());
        assert_eq!(g.used_data_cells(), (n * 255 * 8).div_ceil(3));
        assert!(g.data_cell_count() as usize >= g.used_data_cells());
        assert!(g.data_cell_count() as usize - g.used_data_cells() < 255 * 8 / 3, "at most one codeword of cells is left over");
        let pilots = (0..244).flat_map(|r| (0..252).map(move |c| (c, r))).filter(|&(c, r)| matches!(g.cell(c, r), FlatCell::Pilot { .. })).count();
        assert_eq!(g.pilot_positions().len(), pilots);
        assert!(pilots * 100 / (pilots + g.data_cell_count() as usize) >= 5);
        for i in 0..g.used_data_cells() as u32 { let (c, r) = g.data_position(i); assert_eq!(g.cell(c, r), FlatCell::Data { index: i }); }
        assert_eq!(g.regions(), 3);
        assert_eq!(g.blocks(), (32, 31));
        // region 1 (the middle columns, clear of the marker zones) is filled bottom-up: its first cell is on the last row
        let region0 = (0..g.data_cell_count()).filter(|&i| g.data_position(i).0 < 84).count() as u32;
        assert_eq!(g.data_position(region0).1, 243);
        assert_eq!(g.data_position(region0 - 1), (83, 243)); // region 0 ends at the bottom of its last column
    }
    #[test]
    fn grid_1024_pitch8_bw_and_pitch12_c4() {
        let g = FlatGrid::try_new(1024, 1024, CellFormat { pitch: 8, alphabet: Alphabet::Bw }).unwrap();
        assert_eq!((g.cols(), g.rows()), (126, 122)); assert!(g.packets() >= 1);
        let m = FlatGrid::try_new(1024, 1024, MANIFEST_FORMAT).unwrap();
        assert_eq!((m.cols(), m.rows()), (84, 81)); assert!(m.packets() >= 1);
        assert!(FlatGrid::try_new(256, 256, CellFormat { pitch: 12, alphabet: Alphabet::Bw }).is_none());
    }
    #[test]
    fn pitch_2_and_16_grids() {
        let g = FlatGrid::try_new(1024, 1024, CellFormat { pitch: 2, alphabet: Alphabet::C8 }).unwrap();
        assert_eq!((g.cols(), g.rows()), (504, 488));
        assert!(g.packets() >= 64, "{} packets", g.packets());
        let c16 = FlatGrid::try_new(1024, 1024, CellFormat { pitch: 2, alphabet: Alphabet::C16 }).unwrap();
        assert!(c16.packets() >= 85, "{} packets", c16.packets());
        let big = FlatGrid::try_new(1024, 1024, CellFormat { pitch: 12, alphabet: Alphabet::C8 }).unwrap();
        assert_eq!((big.cols(), big.rows()), (84, 81)); assert_eq!(big.packets(), 1);
        // 12 px b/w holds ~5800 cells = 2 codewords = 462 B: less than a packet, so not a format
        assert!(FlatGrid::try_new(1024, 1024, CellFormat { pitch: 12, alphabet: Alphabet::Bw }).is_none());
        assert!(FlatGrid::try_new(1024, 1024, CellFormat { pitch: 16, alphabet: Alphabet::Bw }).is_none());
    }
    #[test]
    fn sizes_not_multiple_of_eight_are_rejected() {
        assert!(FlatGrid::try_new(853, 480, CellFormat { pitch: 4, alphabet: Alphabet::C8 }).is_none());
        assert!(FlatGrid::try_new(1920, 1080, CellFormat { pitch: 5, alphabet: Alphabet::C16 }).is_some());
    }
    #[test]
    fn tiles_too_narrow_for_the_header_and_bad_pitches_are_rejected() {
        // the header wraps on narrow tiles; a row needs (10 + 52 + 10) modules = 576 px
        assert!(Chrome::new(568, 256).is_none(), "a header row needs (10 + 52 + 10) modules = 576 px");
        assert!(Chrome::new(576, 256).is_some());
        assert_eq!(Chrome::new(992, 256).unwrap().header_rows(), 2);
        assert!(Chrome::new(1024, 248).is_none());
        let f = |p| FlatGrid::try_new(1024, 1024, CellFormat { pitch: p, alphabet: Alphabet::C8 });
        assert!(f(7).is_none()); assert!(f(0).is_none()); assert!(f(4).is_some());
    }
    #[test]
    fn pilot_colours_cycle_and_differ_from_neighbours() {
        let cs: Vec<u8> = (0..4).map(|k| FlatGrid::pilot_colour(0, 0, k, Alphabet::C16)).collect();
        assert_eq!(cs, vec![0, 5, 10, 15]);
        assert_eq!(FlatGrid::pilot_colour(3, 1, 0, Alphabet::Bw), (21 + 3) % 2);
    }
    #[test]
    fn one_px_black_white_is_the_twenty_x_grid() {
        // 1 px b/w cells in the standard tile: ~1 M cells. 20x = 17.04 Mbit/s at 30 fps with a manifest every
        // 30th frame = 72 packets per data frame; v7 carried 74 (one per stripe), v8 packs them back to back.
        // The rate below is at the sender's manifest period now (MANIFEST_EVERY_FRAMES, 31)
        let g = FlatGrid::try_new(1024, 1024, CellFormat { pitch: 1, alphabet: Alphabet::Bw }).expect("pitch 1 is a format");
        assert!(g.codewords() >= 444, "{} codewords", g.codewords());
        assert!(g.packets() >= 82, "{} packets per frame", g.packets());
        let m = crate::flat::MANIFEST_EVERY_FRAMES as f64;
        let mbit = g.packets() as f64 * 1024.0 * 8.0 * 30.0 * (m - 1.0) / m / 1e6;
        assert!(mbit >= 18.7, "{mbit:.2} Mbit/s");
        assert_eq!(CellFormat { pitch: 1, alphabet: Alphabet::Bw }.pitch_index(), Some(0));
    }
}
