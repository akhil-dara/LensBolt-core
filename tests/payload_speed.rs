//! Per-frame payload cost: a clean 1 px frame through the stitcher (cells merged into the frame, codewords
//! decoded, packets reported) on one core. At 30 fps this stage shares a 33 ms budget with the cell read.
use lensbolt_core::detect::Homography;
use lensbolt_core::flat::{compose, render, Alphabet, CellFormat, FlatDecoder, FlatGrid, FlatHeader, FlatRead, Stitcher};
use lensbolt_core::header::FrameType;
use lensbolt_core::packet::PACKET_BYTES;
use std::sync::Arc;

#[test]
fn a_clean_1px_frame_costs_little_to_unpack() {
    std::env::set_var("LB_THREADS", "1");
    let fmt = CellFormat { pitch: 1, alphabet: Alphabet::Bw };
    let g = Arc::new(FlatGrid::try_new(1024, 1024, fmt).unwrap());
    let reads: Vec<FlatRead> = (0..4u8).map(|seq| {
        let hdr = FlatHeader { format: fmt, cols: g.cols() as u16, rows: g.rows() as u16, frame_type: FrameType::Data, transfer_id: 4, segment: 1, block_id: 9 + seq as u32, frame_seq: seq };
        let payload: Vec<u8> = (0..g.payload_bytes()).map(|i| ((i as u64 * 2654435761 + seq as u64 * 977) >> 9) as u8).collect();
        let tile = render(&compose(g.clone(), hdr, &payload)).to_gray();
        let h = Homography::identity();
        let r = lensbolt_core::flat::fast::read_cells_luma(&tile, &g, &h, seq & 3, None).unwrap();
        FlatRead { grid: g.clone(), header: hdr, homography: h, reads: r, decode_ms: 0.0 }
    }).collect();
    let _ = FlatDecoder::new();
    let mut st = Stitcher::new();
    let mut best = f64::MAX;
    for r in &reads {
        let t = std::time::Instant::now();
        let out = st.add(r);
        best = best.min(t.elapsed().as_secs_f64() * 1000.0);
        let valid = out.iter().map(|(_, p)| p.valid.iter().filter(|&&v| v).count()).sum::<usize>();
        assert_eq!(valid, g.packets() * PACKET_BYTES, "frame {} reports every packet", r.header.frame_seq);
    }
    eprintln!("1px clean frame through the stitcher: {best:.1} ms");
    // 12.0 until 2026-11-02; 12.5 measured on a quiet run that day. Speed Task 9 timed the same test on the build
    // before its reader changes and after, in alternation: 12.2 / 13.2 and 12.7 / 11.7 ms — the spread is the run's,
    // not the code's (nothing but the stitcher runs inside the timed call). Budget 13.5 = measured + 8%.
    assert!(best <= 13.5, "unpacking a clean 1 px frame took {best:.1} ms on one core");
}
