# lensbolt-core

The LensBolt codec and protocol as a Rust crate, plus a CLI for testing with PNG frames.

```
cargo test --release                                            # unit + integration (synthetic camera)
cargo test --release --test bigfile -- --ignored --nocapture     # 1 GiB streamed round trip
cargo run --release --bin lensbolt -- info <FILE>                # transfer geometry and goodput estimate
cargo run --release --bin lensbolt -- encode <FILE> -o frames/ [--passes 1.2]
cargo run --release --bin lensbolt -- decode frames/ -o out/ [--fuse 2]
```

## What is implemented

| Layer | Module | Notes |
|---|---|---|
| Error correction | `rs`, `gf256` | RS(255) over GF(2^8) with erasure decoding; 2-symbol safety margin against miscorrection |
| Frame geometry | `geometry` | 16:9 (and square) grids, 7×7 finder markers inside a black guard ring, timing border with colour-calibration cells, 3× redundant 14-byte header |
| Symbols | `symbols` | 16 generated shapes per mode: 5×5 dense, 4×4 super-pixel robust (8 px tiles); smooth and distinct after blur; matched-filter decoding |
| Colour | `palette` | 4 colours on black, per-frame self-calibration from the border |
| Sender side | `frame::compose`, `frame::render` | interleaved codewords, separate symbol/colour bit streams |
| Receiver side | `detect`, `read` | finder-pattern scan, 4-point homography, least-squares refinement from the border, per-block sub-pixel alignment, multi-capture fusion, erasure-guided RS |
| Fountain | `fountain` | systematic LT with dense repair packets; peeling + GF(2) elimination; ~1% overhead |
| File layer | `file` | 16 MiB segments / 1 MiB chunks, manifest with SHA-256s, zstd sniff, blind gap-prioritised scheduler, receiver with per-segment verification |

## Measured (desktop, synthetic camera; real-photo numbers from the 2026-10-10 fixture)

- Dense 1080p frame: **35,475 B** payload (34 packets, RS parity 40); robust: 15,387 B (14 packets, parity 32).
- Realistic camera model (lens blur sigma ~1.1 px, warm white balance, 30% desaturation, +-6 noise):
  robust frames decode fully from 1920 px wide photos (0.84x), dense frames from 2560 px (1.1x) up.
- Real phone photos (4096 px, frame shown at 85% zoom by the viewer): symbol errors 0.7-0.9%, colour errors 3-4%.
- Decode time, single thread on this desktop: dense frame 0.32 s (4K photo 0.35 s), robust 0.11 s.
  The Android plan adds threads/NEON; the algorithmic cost is now ~0.3 s.

### First real-camera transfer (2026-10-13, Razer Kiyo 1080p webcam, 24-inch 1080p monitor)

960x540 frames shown at native pixels, camera ~1.37 photo px per frame px, calibrated baseline (brightness
for a white level of 210, fixed white balance, scanned focus). 70 frames (2.5 passes of a 150 KB file), two
captures per frame: **140/140 captures identified their frame, file recovered byte-identical.** Robust
frames decode 15/15 codewords at 0.1-0.5% symbol / 0.3-1.3% colour error; dense frames 20-32 of 39 codewords
(colour error ~4.5%, dominated by Bayer colour moire at this sampling; yellow strokes read as white).
Rig: `tools/kiyo_baseline.py`, `tools/live_align.py`, `tools/live_capture.py`.

### Real-time baseline (2026-10-13, Razer Kiyo, `tools/live_rx.py`)

Frames played on the monitor at a fixed rate while the camera stream is piped into `lensbolt recv`
(thread-pool decoder, raw frames over stdin). 960x540 frames at 1.37 camera px per frame px, manual
exposure -8 (auto exposure drifts and turns yellow into white: colour error 13% vs 1.5%).

| display fps | unique captures/s | dense codewords ok per identified frame | useful dense codewords/s | result (150 KB file) |
|---|---|---|---|---|
| 5  | 25 | 31.9 / 39 | 475 (0.82 Mbit/s) | complete, 7.8 s |
| 10 | 17 | 25.9 / 39 | 283 | complete, 5.7-6.8 s |
| 15 | 17 | 20.7 / 39 | 189 | complete, 16 s |
| 20 | 17 | 10.7 / 39 | 94  | not complete in 45 s |
| 30 | 17 |  6.1 / 39 | 22  | not complete in 45 s |

1 MB file at 10 fps: byte-identical after 30.4 s (305 displayed frames = 2.15 passes; 0.26 Mbit/s).
Of 437 unique captures, 125 carried no readable header, 47 dense frames had 0 clean codewords and 39
were partial: about half of all captures hit a frame transition. Cause: the Kiyo's rolling-shutter
readout (~30 ms) and its real frame rate (25 fps at exposure -8; MSMF pads to 30/60 with duplicates), so
any display change during readout splits the capture into the top of one frame and the bottom of the
next, and the whole-frame interleaving then loses both.

Goal check: this webcam caps the test at 960x540 frames and ~10 fps useful rate, so it cannot show the
8.5 Mbit/s target; `lensbolt info` puts the codec's own ceiling at 6.0 Mbit/s for 1080p dense at 30 fps
and 12 Mbit/s at 60 fps, 1440p roughly 1.8x those. Reaching 8.5 Mbit/s at 30 fps needs
1440p frames plus a receiver that survives frame transitions (band-local codewords, next).

### Flexible receiving (2026-10-25, protocol v3, Razer Kiyo, camera placed by hand, not aligned)

The camera is wherever it happens to be: full-screen 1920x1080 frames seen at ~0.67 camera px per frame
px with strong perspective (one side twice the size of the other), so full-size cells are unreadable.

- **Stripe/region layout**: codewords interleaved within one-packet stripes inside ~100-cell-wide
  vertical regions (snake order, no capacity loss). A rolling-shutter capture split across two frames
  keeps every stripe on either side (test: <= 2 per region lost of 33); a side crop keeps every region
  in view.
- **Partial views**: frame identified from any two markers (hypotheses pruned by marker size and a
  timing-border probe), grid grown along the visible border, off-picture cells are erasures; the header
  is repeated point-mirrored at the bottom-right. Tests: left 60/70%, right 55%, top/bottom 55-60% views.
- **Local thresholds** for markers and border centroids (the far side of an angled screen read 86-99
  against a global threshold of 84).
- **Alignment**: block offsets clamped below half a cell and seeded only from confident blocks (a black
  block had slipped a whole row by one cell).
- **Multi-scale carousel**: every 4th frame is coarse (half-size grid at 2x), manifests always coarse.

Real-time, 1 MB file, live sender (`lensbolt stream`), camera untouched:

| schedule | display | result |
|---|---|---|
| blind default (1 in 4 coarse) | 10 fps | complete, byte-identical, 89.6 s (only coarse frames readable here) |
| all coarse (what a calibration step would choose here) | 10 fps | complete, byte-identical, 23.9 s (0.33 Mbit/s) |
| all coarse | 20 fps | complete, 31.3 s (camera delivered only 12.6 unique fps) |

Coarse frames decoded 98.6% (dense) and 99.5% (robust) of codewords in real time at this placement.

### 1 GiB round trip (2026-10-10, desktop, packet layer, 2% random packet loss)

| Step | Result |
|---|---|
| Prepare (SHA-256 + sniff, 1 GiB) | 1.4 s |
| Transfer | 41,293 frames = 1.17 passes (one pass = 35,287 frames) |
| Receiver memory | 1024 open chunk decoders, **13.1 MiB** of decoder state (blocks go straight to the part file) |
| Verification | per-segment + whole-file SHA-256 OK, byte-identical in 0.6 s |
| Wall time (both sides in one process, no rendering) | 176 s |

One pass of 35,287 frames is **9.8 min at 60 fps, 19.6 min at 30 fps** for 1 GiB.
Paper goodput with the default 4:1 dense:robust interleave: 14.6 Mbit/s @60 fps, 7.3 Mbit/s @30 fps.
Dense-only frames: 8.4 Mbit/s @30 fps. Real-device numbers are the job of the bench plan.

## Licence and support

LensBolt is free software under the [GNU AGPL-3.0](LICENSE). If it helps you, you can support the work:
[Buy me a coffee](https://buymeacoffee.com/akhildara1l) · [GitHub Sponsors](https://github.com/sponsors/akhil-dara).

The project lives in three repositories: [lensbolt-core](https://github.com/akhil-dara/lensbolt-core) (the codec and
decoder, Rust), [lensbolt-web](https://github.com/akhil-dara/lensbolt-web) (the sender and receiver pages and the
site, lensbolt.com) and [lensbolt-android](https://github.com/akhil-dara/lensbolt-android) (the Android receiver).
