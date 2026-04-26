# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.4](https://github.com/OxideAV/oxideav-dirac/compare/v0.0.3...v0.0.4) - 2026-04-26

### Other

- adopt slim VideoFrame/AudioFrame shape
- encoder/decoder quality matrix — 13 new regression tests
- pin release-plz to patch-only bumps

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
