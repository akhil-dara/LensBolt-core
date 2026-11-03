//! Replay measured real cell errors onto v8 frames: for each `lensbolt diag` LB_DUMP file (read, margin and
//! truth per cell of a real b/w capture), compose a v8 frame on the same grid, flip the cells the camera got
//! wrong, keep the measured margins, and decode with the real extractor.
//!   cargo run --release --example replay -- DUMP.bin [...]
//! A replay, not a capture: the error pattern is the real one, the displayed content is v8's.
use lensbolt_core::flat::{compose, extract_cells, Alphabet, CellFormat, FlatGrid, FlatHeader};
use lensbolt_core::header::FrameType;
use lensbolt_core::packet::PACKET_BYTES;
use std::sync::Arc;

fn main() {
    let (mut tot_cw, mut tot_ok, mut tot_pk, mut tot_pok) = (0usize, 0usize, 0usize, 0usize);
    let mut worst = 1.0f64;
    for path in std::env::args().skip(1) {
        let b = std::fs::read(&path).expect("read dump");
        let u32_at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let (cols, rows, n) = (u32_at(0), u32_at(4), u32_at(8) as usize);
        let k = (cols * rows) as usize;
        let read = &b[12..12 + k];
        let margin: Vec<u16> = (0..k).map(|i| u16::from_le_bytes([b[12 + k + 2 * i], b[13 + k + 2 * i]])).collect();
        let truth = &b[12 + 3 * k..12 + 4 * k];
        let idx_o = 12 + 4 * k;
        let fmt = CellFormat { pitch: 1, alphabet: Alphabet::Bw };
        let g = Arc::new(FlatGrid::try_new(cols + 16, rows + 48, fmt).expect("1 px grid"));
        assert_eq!((g.cols(), g.rows()), (cols, rows));
        for i in (0..n).step_by(997) { let (c, r) = g.data_position(i as u32); assert_eq!(u32_at(idx_o + 4 * i), r * cols + c, "data order differs at {i}"); }
        let h = FlatHeader { format: fmt, cols: cols as u16, rows: rows as u16, frame_type: FrameType::Data, transfer_id: 1, segment: 1, block_id: 0, frame_seq: 0 };
        let payload: Vec<u8> = (0..g.payload_bytes()).map(|i| ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56) as u8).collect();
        let f = compose(g.clone(), h, &payload);
        let errs: usize = (0..g.used_data_cells()).filter(|&i| { let (c, r) = g.data_position(i as u32); let j = (r * cols + c) as usize; read[j] != truth[j] }).count();
        let values: Vec<u8> = f.values.iter().enumerate().map(|(j, &v)| v ^ (read[j] != truth[j]) as u8).collect();
        let fp = extract_cells(&g, &values, &margin, None, None);
        let pk = g.packets();
        let pok = (0..pk).filter(|&p| fp.valid[p * PACKET_BYTES..(p + 1) * PACKET_BYTES].iter().all(|&v| v) && fp.bytes[p * PACKET_BYTES..(p + 1) * PACKET_BYTES] == payload[p * PACKET_BYTES..(p + 1) * PACKET_BYTES]).count();
        let wrong = (0..g.codewords()).filter(|&c| { let dl = fmt.data_len(); fp.valid[c * dl] && fp.bytes[c * dl..(c + 1) * dl] != payload[c * dl..(c + 1) * dl] }).count();
        println!("{}: cell errors {:.2}%  codewords ok {}/{}  packets ok {}/{}  miscorrected {}", std::path::Path::new(&path).file_name().unwrap().to_string_lossy(), errs as f64 * 100.0 / g.used_data_cells() as f64, fp.codewords_ok, g.codewords(), pok, pk, wrong);
        tot_cw += g.codewords(); tot_ok += fp.codewords_ok; tot_pk += pk; tot_pok += pok;
        worst = worst.min(pok as f64 / pk as f64);
    }
    println!("total: codewords ok {tot_ok}/{tot_cw} ({:.2}%), packets ok {tot_pok}/{tot_pk} ({:.2}%), worst capture {:.1}% of packets", tot_ok as f64 * 100.0 / tot_cw as f64, tot_pok as f64 * 100.0 / tot_pk as f64, worst * 100.0);
}
