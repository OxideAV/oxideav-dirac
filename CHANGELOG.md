# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.4](https://github.com/OxideAV/oxideav-dirac/compare/v0.0.3...v0.0.4) - 2026-05-05

### Other

- encoder-side §11.3 wavelet residue (~+15 dB ffmpeg cross-decode)
- accept VC-2 LD parse code 0x88 (SD-Profile variant)
- docs_corpus driver compares frames in display order
- panic-safe i32/u32 sums in inter motion-data + OBMC paths
- integrate docs/video/dirac fixture corpus
- replace never-match regex with semver_check = false
- migrate to centralized OxideAV/.github reusable workflows
- inter encoder OBMC-aware ME refinement ([#186](https://github.com/OxideAV/oxideav-dirac/pull/186))
- inter encoder sub-pel ME ([#168](https://github.com/OxideAV/oxideav-dirac/pull/168))
- hard-assert ffmpeg I+P interop on homogeneous core-syntax chain
- core-syntax intra encoder (round 2) — 0x0C AC ref
- inter encoder (round 1) — 1-ref OBMC + arith encoder
- adopt slim VideoFrame/AudioFrame shape
- encoder/decoder quality matrix — 13 new regression tests
- pin release-plz to patch-only bumps

### Added

- Dirac inter encoder **wavelet residue** (§11.3 / §13.4) — the
  `encoder_inter` path now closes the prediction-error loop by
  computing `source - decoder_OBMC_reconstruction` in the spec's
  signed pre-output-offset domain, forward-DWT'ing the difference,
  dead-zone quantising per the new `ResidueParams` (LeGall 5/3 /
  depth 3 / qindex 0 by default), and emitting the §11.3
  `ZERO_RESIDUAL=false` flag plus per-component AC-coded subbands
  (single codeblock per subband, no per-codeblock quant offset, no
  custom quant matrix) the decoder adds back at §15.8.2. The new
  encoder-side helpers (`build_obmc_prediction`,
  `build_residue_plane`, `forward_and_quantise_residue`,
  `write_residue_component_subbands`) reuse `crate::obmc::motion_compensate`
  and the §13.4.4 context machinery so the residue's reconstruction
  matches the decoder symbol-for-symbol. New
  `InterEncoderParams::residue: Option<ResidueParams>` knob — `Some(...)`
  is the default (residue ON); set `None` to revert to the round-1
  ZERO_RESIDUAL=true path for direct ME-only A/B comparison. PSNR
  uplift on every synthetic fixture is dramatic: at qindex=0 with
  LeGall 5/3 the inter self-roundtrip is **bit-exact (∞ dB)** on
  `synthetic_translating_pair_64(4, 0)`, `(0, -4)`, `(0, 0)` and
  `synthetic_camera_pan_64(1, 0)` — the residue captures everything
  ME / OBMC couldn't reach. ffmpeg cross-decode jumps from
  **19.39 dB → 34.38 dB** (~+15 dB) on the `+4`-pel translating-square
  homogeneous-profile chain (`tests/ffmpeg_interop.rs::ffmpeg_cross_decodes_inter_residue_beats_no_residue`,
  hard-asserted). 7 new tests (4 unit + 3 self-roundtrip / cross-decode
  including 1 hard ffmpeg assert); all 182 dirac tests pass.
  Existing OBMC / sub-pel tests now explicitly set `residue: None`
  in the no-residue baseline so they keep measuring pure ME quality.

### Fixed (parse code recognition)

- VC-2 LD pictures with parse code `0x88` (and the rest of the
  `0x88`-family: `0x89`, `0x8C`, `0x8D`) now decode through the
  low-delay path. The decoder previously only recognised the `0xC8`
  family and rejected the `0x88` family with "unsupported core-syntax
  parse code" — but `0x88` is the parse code that real ffmpeg /
  libschroedinger `vc2enc` emits for VC-2 SD-Profile (the in-tree
  `corpus_vc2_low_delay_tiny_320x240` and `corpus_vc2_low_delay_3pics_320x240`
  fixtures, sliced from a vc2enc stream, both use it). Both LD
  variants share the Golomb-coded slice path so accepting both cost
  one extra mask. Per Dirac BBC §9.5.1 + VC-2 ST 2042-1 Table 10.1,
  `bit 6` distinguishes the two LD encodings (legacy vs AC-coded
  variant) but is irrelevant to the decode path. Both fixtures now
  report Y PSNR ≈ 49 dB instead of an unsupported-parse-code error.
  HQ (`0xE8` family) and core-syntax (`0x08` family) classification
  is unchanged. New `low_delay_profile_recognises_88_and_c8_families`
  unit test pins the matrix.

### Changed

- `tests/docs_corpus.rs` driver now drains every decoded `VideoFrame`
  first and sorts by `pts` (== `picture_number` when the packet has
  no explicit pts, which is the case throughout the corpus) before
  per-frame comparison against `expected.yuv`. The reference YUVs are
  in display order; the previous in-order draw paired our P frame
  against the reference's B and vice versa for `corpus_i_p_b_320x240`,
  reporting Y PSNR 19.68 dB (P-vs-B) and 7.25 dB (B-vs-P) — both off
  by content, neither measuring true per-picture quality. The sorted
  comparison reports the actual numbers: I 48.79 dB Y, B 7.31 dB Y
  (the bipred OBMC convention gap), P 47.96 dB Y. Aggregate Y goes
  from 11.78 → 12.08 dB; the move is small because the I dominates
  but the per-frame numbers are now meaningful.

### Fixed

- Decoder no longer panics on 2-ref bipred (B-picture, parse `0x0a`)
  bitstreams in debug builds. Three spec-modular i32/u32 sums in the
  motion-data path — `block_dc` (§13.4 DC residual + spatial pred),
  `block_vector` (§12.3.6 MV residual + median pred), and
  `decode_sb_splits` (§12.3.4 split residual + neighbour pred), plus
  the §15.8.2 OBMC accumulator in `obmc::block_mc` — used direct `+`
  / `+=` and aborted in debug when a malformed/unsupported stream
  produced a wrapped sum. They now use `wrapping_add` (matching the
  `subband.rs::reconstruct_subband` precedent for §13.5.3 wavelet
  prediction). The `corpus_i_p_b_320x240` ReportOnly fixture now
  decodes end-to-end (intra ≈ 48.79 dB Y, ref-1 inter ≈ 19.68 dB Y,
  bipred ≈ 7.25 dB Y — the bipred floor is the known
  ffmpeg-OBMC-convention gap, not the panic).

### Added

- Dirac inter encoder **OBMC-aware ME refinement** (#186, §15.8.6) —
  `encoder_inter` now follows the per-block sub-pel SAD search with
  `obmc_refine_passes` (default 2) full passes of OBMC-aware refinement:
  for each block, the 8 sub-pel neighbours of its current MV are scored
  via the same §15.8.5 weighted-sum reconstruction the decoder will
  perform, keeping whichever MV minimises the per-block SSE against
  the source. The `neighbour_sum` is built from the eight surrounding
  blocks' weighted predictions (re-using `obmc::spatial_wt`,
  `obmc::subpel_predict`, `obmc::interp2by2`) so the encoder's blend
  mirrors the decoder symbol-for-symbol. Self-roundtrip uplift on
  fixtures where motion straddles the OBMC overlap region is
  dramatic — `synthetic_translating_pair_64(2,-1)` jumps from
  **32.56 dB** (no-OBMC baseline) to **∞ dB** (bit-exact); the vertical
  `(0,-4)` translation goes 30.24 → 33.37 dB; the integer camera-pan
  goes 26.92 → 28.02 dB. ffmpeg cross-decode is structurally capped
  near 19-20 dB on these synthetic fixtures regardless of MV quality
  (independent ffmpeg-side OBMC convention difference, documented in
  `tests/ffmpeg_interop.rs`); the new `ffmpeg_obmc_aware_me_does_not_regress_cross_decode`
  test asserts no cross-decode regression vs the no-OBMC baseline.
  Setting `obmc_refine_passes = 0` reverts to the pre-#186 hard-block
  SAD output for direct A/B comparison. 4 new tests (2 unit + 1
  self-roundtrip A/B + 1 ffmpeg cross-decode floor); all 165 dirac
  tests pass.

- Dirac inter encoder **sub-pel ME** (#168) — `encoder_inter` now runs
  the integer-pel SAD search followed by per-level 8-neighbor gradient
  refinement to the configured `mv_precision` (default quarter-pel, also
  half- and eighth-pel supported). Sub-pel candidates are evaluated
  against the spec's §15.8.10 / §15.8.11 reference (8-tap half-pel
  filter + bilinear sub-half) so SAD evaluated at the encoder matches
  the eventual reconstruction error pixel-for-pixel. New
  `synthetic_camera_pan_64` fixture with a vertical-bar pattern panned
  by a configurable quarter-pel offset exposes the gain: self-roundtrip
  Y PSNR jumps from **26.92 dB** (integer-pel) to **52.04 dB**
  (quarter-pel) on a 1/4-pel pan. ffmpeg cross-decode improves modestly
  (~1 dB) — that gap will widen once OBMC overlap encoding lands
  (#169). 5 new tests (3 unit + 1 self-roundtrip + 1 ffmpeg
  cross-decode); all 161 dirac tests pass. `mv_precision` is also
  exposed via `InterEncoderParams` so callers can pick integer-pel
  (`0`), half- (`1`), quarter- (`2`, default), or eighth-pel (`3`).

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
