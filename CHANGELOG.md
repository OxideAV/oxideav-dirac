# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- ffmpeg interop hard-asserts the homogeneous core-syntax I+P chain
  (#135). The legacy `tests/ffmpeg_interop.rs::ffmpeg_decodes_our_inter_stream_translating_square`
  now drives `encode_core_intra_then_inter_stream` instead of the
  mixed HQ-intra + core-inter `encode_intra_then_inter_stream`, so the
  stream is single-profile (parse codes `0x0C` + `0x09`) and ffmpeg's
  `dirac` decoder no longer trips its profile-mismatch guard. Both this
  test and the round-2 close-out
  `tests/encoder_intra_core_roundtrip.rs::ffmpeg_decodes_our_core_intra_then_inter_stream`
  are now hard asserts (no soft-skip on ffmpeg rejection). Verified
  against ffmpeg 8.1: intra Y PSNR ≈ 52 dB, inter Y PSNR ≈ 19 dB
  cross-decoded on the translating-square 64x64 fixture.

### Added

- Dirac core-syntax **intra encoder** (round 2) — new `encoder_intra_core`
  module emits `0x0C` AC-coded intra reference pictures (single
  codeblock per subband, no per-codeblock quant offset, no custom quant
  matrix, qindex=0 near-lossless). The new
  `encode_core_intra_then_inter_stream` chains a `0x0C` intra reference
  with the round-1 `0x09` inter so the entire stream stays in one
  parse-code family. ffmpeg's `dirac` decoder no longer rejects the
  chain (the round-1 ffmpeg-interop soft-skip closes here): cross-
  decoded intra Y PSNR ≈ 52 dB on a translating-square 64x64 fixture.
  Self-roundtrip is bit-exact on flat pictures and ≥48 dB Y/U on the
  testsrc gradient. 6 new unit tests + 4 new integration tests
  (`tests/encoder_intra_core_roundtrip.rs`) including a hard ffmpeg
  cross-decode (no soft-skip).

- Dirac core-syntax **inter encoder** (round 1) — `encoder_inter` module
  emits 1-reference, integer-pel-MV, OBMC-only inter pictures (parse
  code `0x09`) with §11.2 picture-prediction parameters and §12.3
  block_motion_data carried through a new binary arithmetic encoder
  (`arith::ArithEncoder`). Self-roundtrip on a translating-square
  64x64 fixture clears 30 dB Y PSNR (vertical & horizontal motion);
  zero-motion is bit-exact (∞ dB). 14 new tests including a full
  encoder→decoder integration suite (`tests/encoder_inter_roundtrip.rs`)
  and a soft-skip ffmpeg cross-decode (waits on r2 core-syntax intra).
- `arith::ArithEncoder` — Annex B.2 binary arithmetic encoder mirror
  of the existing `ArithDecoder`, with E1/E2/E3 carry handling and a
  conservative termination that keeps the decoder's past-end-1
  defaults inside the final interval. Roundtrip-tested with the
  decoder on uniform, biased, and signed-int symbol streams.

## [0.0.3](https://github.com/OxideAV/oxideav-dirac/compare/v0.0.2...v0.0.3) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core

## [0.0.2](https://github.com/OxideAV/oxideav-dirac/compare/v0.0.1...v0.0.2) - 2026-04-24

### Other

- fix LD slice_y_length field width — ffmpeg-bit-exact
- ffmpeg interop — HQ lossless, LD accepted, profile/preset fit
- crate-wide fmt + clippy cleanup
- VC-2 LD (Low-Delay) intra-only encoder
- ffmpeg-interop test for 8-bit 4:2:2
- 10/12-bit output + frame-rate-aware timebase
- inter-picture decode with OBMC motion compensation
- dirac encoder: ffmpeg-compatible headers + self-roundtrip (step 6)
- dirac encoder: bitwriter + HQ sequence/picture emit (steps 2-5)
- dirac encoder: forward wavelet analysis (step 1/5)
- update lib docs to reflect core-syntax support
- unit test for core-syntax VLC coefficient decode
- §13.4 core-syntax coefficient unpacking (AC + VLC paths)
- update crate docs + intra-only capability
- full intra decode for VC-2 LD + HQ low-delay pictures
- inverse quantisation + default quant matrices
- add wavelet IDWT + subband primitives
