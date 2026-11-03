//! Draw a flat frame: chrome modules and solid-colour cells.

use super::compose::FlatFrame;
use super::format::MODULE_PX;
use super::layout::{Chrome, FlatCell};
use crate::image::RgbImage;

const WHITE: [u8; 3] = [255, 255, 255];
const BLACK: [u8; 3] = [0, 0, 0];

fn module(img: &mut RgbImage, mx: u32, my: u32, white: bool) {
    img.fill_rect(mx * MODULE_PX, my * MODULE_PX, MODULE_PX, MODULE_PX, if white { WHITE } else { BLACK });
}

/// Chrome only (border, markers, header): black background.
pub fn render_chrome(img: &mut RgbImage, chrome: &Chrome, header_bits: &[bool]) {
    let (mw, mh) = chrome.modules();
    for i in 0..chrome.border_module_count() {
        let (mx, my) = chrome.border_module(i);
        module(img, mx, my, Chrome::border_is_white(i));
    }
    for my in 0..mh { for mx in 0..mw {
        if let Some((lx, ly)) = chrome.marker_local(mx, my) { module(img, mx, my, Chrome::marker_module_paint(lx, ly)); }
    } }
    // copy 1 is point-mirrored AND bit-inverted: an upside-down capture then fails the header check on
    // the wrong rotation instead of reading the mirrored copy as if it were copy 0
    for copy in 0..2 {
        for (i, &b) in header_bits.iter().enumerate().take(Chrome::HEADER_CELLS as usize) {
            let (mx, my) = chrome.header_module(copy, i as u32);
            module(img, mx, my, b ^ (copy == 1));
        }
    }
}

pub fn render(frame: &FlatFrame) -> RgbImage {
    let g = &frame.grid;
    let mut img = RgbImage::new(g.width_px(), g.height_px());
    render_chrome(&mut img, g.chrome(), &frame.header.to_bits());
    let pal = g.format().alphabet.palette();
    let p = g.format().pitch as u32;
    let used = g.used_data_cells() as u32;
    for r in 0..g.rows() {
        for c in 0..g.cols() {
            // unused cells and data cells beyond the last stripe stay black: no false colour samples
            match g.cell(c, r) { FlatCell::Unused => continue, FlatCell::Data { index } if index >= used => continue, _ => {} }
            let (x, y) = g.cell_top_left(c, r);
            img.fill_rect(x, y, p, p, pal[frame.value(c, r) as usize]);
        }
    }
    img
}

/// `render` scaled to (width, height) with nearest-neighbour sampling (integer factors keep cells crisp).
pub fn render_to(frame: &FlatFrame, width: u32, height: u32) -> RgbImage {
    let src = render(frame);
    if (width, height) == (src.width, src.height) { return src; }
    let mut out = RgbImage::new(width, height);
    for y in 0..height {
        let sy = (y as u64 * src.height as u64 / height as u64) as u32;
        for x in 0..width {
            let sx = (x as u64 * src.width as u64 / width as u64) as u32;
            out.set(x, y, src.get(sx, sy));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flat::compose::compose;
    use crate::flat::{Alphabet, CellFormat, FlatGrid};
    use crate::header::FrameType;
    use crate::flat::FlatHeader;
    use std::sync::Arc;

    #[test]
    fn render_size_and_chrome() {
        let g = Arc::new(FlatGrid::try_new(1024, 1024, CellFormat { pitch: 4, alphabet: Alphabet::C8 }).unwrap());
        let h = FlatHeader { format: g.format(), cols: g.cols() as u16, rows: g.rows() as u16, frame_type: FrameType::Data, transfer_id: 7, segment: 1, block_id: 99, frame_seq: 0 };
        let payload: Vec<u8> = (0..g.payload_bytes()).map(|i| (i * 31 % 251) as u8).collect();
        let f = compose(g.clone(), h, &payload);
        let im = render(&f);
        assert_eq!((im.width, im.height), (1024, 1024));
        assert_eq!(im.get(4, 4), [255, 255, 255]); assert_eq!(im.get(12, 4), [0, 0, 0]);   // border modules alternate
        assert_eq!(im.get(1020, 12), [255, 255, 255]);                                     // border module 128 (even) is white
        assert_eq!(im.get(44, 44), [255, 255, 255]); assert_eq!(im.get(28, 28), [0, 0, 0]); // marker centre white, ring black
        assert_eq!(im.get(20, 20), [255, 255, 255]);                                       // marker outer ring is white: module (2,2)
        assert_eq!(im.get(12, 12), [0, 0, 0]);                                             // guard module between border and marker
        let bits = h.to_bits();
        for (i, &b) in bits.iter().enumerate().take(20) {
            let (x, y) = g.chrome().header_cell_center(0, i as u32);
            assert_eq!(im.get(x as u32, y as u32), if b { [255, 255, 255] } else { [0, 0, 0] }, "header cell {i}");
            let (x, y) = g.chrome().header_cell_center(1, i as u32);
            assert_eq!(im.get(x as u32, y as u32), if b { [0, 0, 0] } else { [255, 255, 255] }, "header copy 1 cell {i} is inverted");
        }
        let (x, y) = g.cell_center(18, 0);
        assert_eq!(im.get(x as u32, y as u32), Alphabet::C8.palette()[f.value(18, 0) as usize]);
        let (x, y) = g.cell_center(17, 13); // unused cell in the marker zone's inner guard ring
        assert_eq!(im.get(x as u32, y as u32), [0, 0, 0]);
        // data cells beyond the last stripe carry nothing and are drawn black, not palette[0] (white)
        let g1 = Arc::new(FlatGrid::try_new(1024, 1024, CellFormat { pitch: 12, alphabet: Alphabet::C8 }).unwrap());
        let h1 = FlatHeader { format: g1.format(), cols: g1.cols() as u16, rows: g1.rows() as u16, frame_type: FrameType::Data, transfer_id: 7, segment: 1, block_id: 1, frame_seq: 0 };
        let f1 = compose(g1.clone(), h1, &vec![0u8; g1.payload_bytes()]);
        assert!(g1.used_data_cells() < g1.data_cell_count() as usize);
        let (c, r) = g1.data_position(g1.data_cell_count() - 1);
        let (x, y) = g1.cell_center(c, r);
        assert_eq!(render(&f1).get(x as u32, y as u32), [0, 0, 0], "unused data cell must be black");
        let big = render_to(&f, 2048, 2048);
        assert_eq!((big.width, big.height), (2048, 2048));
        assert_eq!(big.get(88, 88), [255, 255, 255]);
    }
}
