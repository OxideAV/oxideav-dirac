# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
