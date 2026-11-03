# lensbolt-core

**Files through light.** `lensbolt-core` is the codec, decoder and transfer protocol behind
[LensBolt](https://lensbolt.com): a sender draws a moving square of tiles on any screen, a camera reads it, and the
file is rebuilt on the receiving device — no cable, no network, no account, one way only.

This crate is the engine. The sender and receiver pages ([lensbolt-web](https://github.com/akhil-dara/lensbolt-web))
compile it to WebAssembly; the Android app ([lensbolt-android](https://github.com/akhil-dara/lensbolt-android))
links it through JNI. The same Rust runs on the desktop, in the browser and on the phone, so a frame that decodes in
one place decodes everywhere.

- **Any screen → any camera.** Laptops, monitors, projectors, phones; a phone's camera, a webcam, or a shared window.
- **Robust by design.** Reed–Solomon over GF(2⁸), a systematic LT fountain, per-block pilots and a local equaliser:
  a capture can be tilted, blurred, partly out of frame, split across two displayed frames or blended by a slow
  panel, and the receiver still takes what it can from it.
- **Verified end to end.** Every file carries its SHA-256; the receiver offers nothing it has not checked.
- **Fast on real hardware.** Measured on a laptop screen and a 2024 phone: 4.6–5.1 Mbit/s with the Android app,
  2.4 Mbit/s in a phone browser (the figures the apps display are measured, not estimated).

## How it works

```
file ──► segments ──► packets ──► LT fountain ──► RS(255) codewords ──► frames (tiles) ──► screen
                                                                                            │ light
camera ──► capture ──► markers · homography · pilots ──► cells ──► codewords ──► packets ──► file ✓ SHA-256
```

**Sender.** The file is split into 1 MiB chunks and 16 MiB segments and described by a manifest (names, sizes,
hashes). A systematic LT fountain turns each segment into an endless stream of packets, so the sender never needs
to know what the receiver missed; it just keeps drawing. Each frame holds frame-wide RS(255) codewords laid out in
a square tile of up to 1024×1024 px with four corner markers, a timing border, a header in two copies and pilot
cells in every 8×8 block.

**Receiver.** Every capture is identified from its markers and refined to a full homography on the border (tilt,
rotation, keystone), then a per-block shift field follows lens distortion and hand movement. Cells are read against
per-block levels; a local equaliser re-decides the doubtful ones from what the camera does to their neighbours. RS
decoding with erasure hints turns cells into bytes; the fountain decoder and the segment verifier turn bytes into the
file. Tracking carries the geometry from one capture to the next, and lets go of it the moment the reads say it is
wrong.

**Formats.** Cells of 1, 2, 3 … px in black/white, 4 or 8 greys, or 4 to 16 colours; the sender picks a format
for the screen and the receiver tells the user, in two words, which one reads best for their camera and distance.
A 12 px colour manifest frame is shown every 31 frames so a receiver can join late.

## Using the crate

```toml
[dependencies]
lensbolt-core = { git = "https://github.com/akhil-dara/lensbolt-core", tag = "v0.1.0", default-features = false }
```

Features: `zstd` (compression through the C library; on by default, off for WebAssembly where files are sent as
they are) and `cli` (the command-line tool).

```rust
use lensbolt_core::flat::sender::{FlatSender, FlatSenderConfig};
use lensbolt_core::flat::session::{FrameReader, RxSession};

// sending: frames to draw, one after another, forever
let mut tx = FlatSender::from_bytes("notes.txt", data, FlatSenderConfig::default())?;
let frame = tx.next_frame()?;         // a tile to draw at 30 fps (FlatFrame: pixels + its header)

// receiving: one reader per camera frame, one session that collects
let mut reader = FrameReader::new();
let read = reader.read_rgba(&pixels, width, height);
let mut rx = RxSession::new();
let status = rx.push_read(&read.to_bytes(), now_ms);   // JSON: progress, rate, guidance, SHA-256 when done
```

`FrameReader` is safe to run in several threads or workers; `hint_bytes`/`set_hint` share the tracked geometry
between them, and `RxSession` merges their reads in any order.

## Command line

```
cargo run --release -- info <FILE>                     # geometry and goodput for a screen
cargo run --release -- stream <FILE> --pitch 2 --alphabet bw   # live frames as raw RGB on stdout
cargo run --release -- encode <FILE> -o frames/        # PNG frames
cargo run --release -- decode frames/ -o out/          # PNGs or photos back to the file
cargo run --release -- diag capture.png                # what the decoder sees in one capture
cargo run --release -- recv -o out/                    # raw camera frames on stdin → file
```

## Layout

| Module | What it holds |
|---|---|
| `flat` | the tile format: layout, sender, fast luma reader, equaliser, decoder, stitcher, live session, tracking |
| `fountain` | systematic LT encoder and peeling/Gaussian decoder |
| `rs`, `gf256`, `crc16` | Reed–Solomon with erasures, the field, header CRC |
| `file` | manifest, chunks and segments, batch sending, SHA-256 verification, compression sniffing |
| `detect`, `image` | marker search, homography, sharpness gate, image types |
| `tune`, `words`, `finish`, `calib` | the calibration carousel, the two-word codes, finish words |

## Tests

```
cargo test --release
```

Unit and integration tests run on synthetic captures and need nothing else. The accuracy and timing gates on real
phone captures (`tests/phone_*.rs`, `tests/fhd2px_floor.rs`, `tests/fast_ref.rs`) skip when the capture sets are
not present; set `LB_REQUIRE_CAPTURES=1` to make them fail instead. `LB_PROFILE=1 LB_THREADS=1` prints a per-stage
profile of the decoder.

## Status

Protocol version 9. Pre-1.0: the frame format is stable within a minor version; the crate API may still move.
Issues and ideas: the tracker of whichever repository they belong to.

## Licence and support

LensBolt is free software under the [GNU AGPL-3.0](LICENSE). If it helps you, you can support the work:
[Buy me a coffee](https://buymeacoffee.com/akhildara1l) · [GitHub Sponsors](https://github.com/sponsors/akhil-dara).
