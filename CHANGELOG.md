# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **VC-2 LD picture-level rate-control picker** (round-131). The LD
  analogue of HQ's `with_slice_size_target`: given a target
  picture-byte budget, the picker derives `slice_bytes_numer /
  slice_bytes_denom` so the encoded picture payload lands within ±1
  byte of the target, then walks `qindex ∈ 0..=127` to pick the
  smallest qindex for which **every** slice's `luma_bits +
  chroma_bits` fits the `payload_bits = 8*slice_bytes - header_bits`
  budget without Funnel-truncation. SMPTE ST 2042-1 §13.5.3.2 (per-
  slice byte budget), §13.5.2 (per-slice qindex header), §13.5.4
  (quant-matrix indexing).
  - New public API surface in `oxideav_dirac::encoder`:
    - `pick_ld_picture_qindex(seq, params, y, u, v) -> u32`
    - `ld_picture_qindex_diagnostic(seq, params, y, u, v) -> (u32, i64)`
    - `ld_picture_payload_bytes(params) -> usize` (size predictor —
      independent of content & qindex; sole input is `params`)
    - `derive_ld_slice_bytes_for_target(base, target_bytes) ->
      Option<LdEncoderParams>` (one-shot fixed-point that converges in
      ≤4 passes; returns `None` when target can't fit header +
      2·N_slices minimum)
    - `encode_single_ld_intra_picture_with_size_target(...)`
    - `encode_single_ld_intra_stream_with_size_target(...)` — returns
      `(stream_bytes, chosen_qindex, adjusted_params)`
  - New test file `tests/encoder_ld_picture_qindex.rs` (8 tests, all
    passing). The headline acceptance test pins the three-budget
    behaviour: targets `[200, 1024, 4096]` bytes land at exact actual
    bytes `[200, 1024, 4096]` (0% miss, well under the ±10% tolerance)
    and pick qindexes `[127, 37, 6]` — the small budget escalates to
    the most-aggressive available quantiser while the large budget
    stays near lossless, satisfying the small > large monotonicity
    requirement.
  - Picture-byte size = `picture_header(4) + byte-aligned
    transform_parameters(numer-dependent) + slice_bytes_numer`. Every
    slice writes exactly `slice_bytes(sx, sy)` bytes regardless of
    qindex (Funnel-bounded 1-padding in `write_funnel_bounded`), so
    once `slice_bytes_numer` is fixed the picture-byte count is
    deterministic and the picker's only remaining job is quality
    maximisation under the budget.

### Fixed

- **§12.3.6.6 Case 4 unbiased-mean rounding** (round-128). The DC
  prediction for intra blocks inside inter pictures (the spec's
  `dc_prediction()` Case 4 — all three neighbours intra-coded) computed
  the mean of three neighbour DC values as `(a + b + c + 1) / 3` using
  Rust's `/`, which truncates toward zero. §6.4.3 defines `mean(S)` as
  `(Σ + n//2) // n` with `//` being **floor** division; for negative
  sums truncate and floor differ by exactly 1, so every intra block
  whose neighbours' DC values averaged negative was reconstructed with
  its DC value biased up by 1 LSB. The bug propagated through OBMC into
  a localised +1 region on the inter corpus fixtures (closing the
  ~1% pixel-gap on `i-then-p` and `i-p-b`).
  The intra-only subband DC predictor (`picture::mean3`) already used
  `div_euclid` since round-118; this round brings the inter motion-data
  DC predictor (`picture_inter::dc_prediction`) into line with it. New
  unit test `dc_prediction_uses_floor_unbiased_mean` pins the
  negative-sum, positive-sum and exact-multiple-of-3 cases. The r125
  commit message attributed the residual ~1% gap to "OBMC convention",
  but the actual root cause was upstream in DC-value decoding; OBMC and
  the §15.8.5 weighted-sum reconstruction are unchanged.
  Effect on the docs-corpus fixtures (every previously bit-exact intra
  fixture stays bit-exact):
  - `corpus_i_then_p_320x240` — **100.00% bit-exact** (was 99.23%).
    Promoted from `Tier::ReportOnly` to `Tier::BitExact`.
  - `corpus_i_p_b_320x240` — **100.00% bit-exact** (was 99.21%).
    Promoted to `Tier::BitExact`.
  - `corpus_interlaced_720x576_i_then_p_wavelet_5_3` — **99.68%**
    pixel-exact (frame 0 100.00%; frame 1 99.36%). Stays
    `Tier::ReportOnly`; the residual gap is concentrated in the
    LeGall-5,3 wavelet path on the inter picture and is the next
    closeable item.

### Added

- **§13.2.1 inter quant-offset on the decoder** (round-125). The 2008
  Dirac specification defines two reconstruction offsets in
  inverse-quantisation:
  - **Intra** — `(qf + 1) / 2` (the spec's `quant_factor(q).div_ceil(2)`,
    biasing toward the midpoint of the dead-zone interval).
  - **Inter** — `(qf * 3 + 4) / 8` (biasing toward zero because the
    inter residue distribution is sharply Laplacian).
  SMPTE ST 2042-1 (VC-2) later collapsed both into the intra formula
  because VC-2 is intra-only by construction; the previous `quant_offset`
  was therefore correct for every LD / HQ slice and every core-syntax
  intra picture but applied the intra formula to inter pictures too,
  leaving a ~1-LSB-times-most-coefficients reconstruction bias on every
  inter wavelet residue.
  New public API: `quant_offset_for(q, is_intra)` and
  `inverse_quant_for(qcoeff, q, is_intra)`. The existing
  `quant_offset(q)` / `inverse_quant(qcoeff, q)` are kept as thin shims
  with `is_intra = true` so existing callers (LD slices, internal
  encoder roundtrip asserts) need no change. The `is_intra` flag is
  threaded through `picture_core::core_transform_component`,
  `decode_subband_ac` and `decode_subband_vlc` (it was already a
  parameter of `core_transform_component`; the helpers now propagate
  it).
  At `q ∈ {0, 1}` intra and inter agree (`offset = 1` and `2`
  respectively), so every encoder self-roundtrip test — all of which
  run at qindex=0 by design — is unaffected. At higher `q` the inter
  reconstruction is strictly closer to zero than the intra
  reconstruction for every non-zero coefficient (invariant pinned by
  `intra_offset_dominates_inter_offset` and
  `inverse_quant_inter_pulls_toward_zero`); the reconstruction-interval
  invariant `3 ≤ offset + 2 < quant_factor` from the spec's note holds
  for both branches at every `q >= 2` (pinned by
  `inter_offset_satisfies_reconstruction_interval_for_q_ge_2`).
  Effect on the docs-corpus inter fixtures (every intra fixture was
  already bit-exact since round-118 and is unchanged):
  - `corpus_i_then_p_320x240` — P frame **67.50 dB Y** (was ≈48 dB),
    aggregate 99.23% pixel-exact, UV ∞ dB.
  - `corpus_i_p_b_320x240` — P + B frames **67.16 dB Y** aggregate
    (was 47.96 dB P, 7.31 dB B), 99.21% pixel-exact, UV ∞ dB.
  - `corpus_interlaced_720x576_i_then_p_wavelet_5_3` —
    **73.90 dB Y** + **54.70 dB UV** aggregate, 99.62% pixel-exact.
  The residual ~1% pixel gap on each is the OBMC convention edge case
  (decoder rounds the §15.8.5 weighted-sum reconstruction one LSB
  differently from ffmpeg on a small fraction of block edges), not the
  inverse-quant offset. Eight new unit tests in
  `quant::tests` (offset agreement at low `q`, intra-dominates-inter
  invariant, reconstruction-interval invariant, inverse_quant
  identity at low `q`, inter-pulls-toward-zero invariant) plus one
  integration unit in `picture_core::tests`
  (`vlc_inter_offset_applies_via_is_intra_false`, pinning the wiring
  with an end-to-end VLC subband decode at `qindex = 12` where intra
  and inter reconstructions differ by 1 LSB).


## [0.0.7](https://github.com/OxideAV/oxideav-dirac/compare/v0.0.6...v0.0.7) - 2026-05-24

### Other

- §13.5.4 per-slice adaptive qindex (HQ profile)
- emit §12.4.5.3 custom quantisation matrix (HQ + LD)
- document the VLC core-intra encoder in the crate-level rustdoc
- VLC (non-arithmetic) core-syntax intra encoder (0x4C)
- core-intra encoder all-zero codeblock skip (§13.4.3.3)
- cumulative codeblock quant offset fix + core-intra spatial partition encoder
- inter encoder: post-OBMC bipred mode-only refinement pass (round-95)
- inter encoder: bipred per-ref candidate-set widening to {int-pel, half-pel, sub-pel}
- inter encoder: post-OBMC second adaptive sub-pel-vs-int-pel pass for 1-ref path
- inter encoder: per-block adaptive sub-pel-vs-int-pel selection for 1-ref path

### Added

- VC-2 HQ **per-slice adaptive quantisation index** (§13.5.2 / §13.5.4,
  round-114). `EncoderParams` gains `slice_size_target: Option<u32>`
  (default `None`) plus a `with_slice_size_target(target)` builder. With
  `None` the encoder keeps the legacy constant-qindex behaviour — every
  slice writes `params.qindex` verbatim and the whole picture is
  quantised once. With `Some(target)` each slice independently picks the
  **smallest** qindex in `params.qindex..=127` for which **every**
  component's HQ length byte is `<= target` (i.e. each component's
  coefficient payload fits in `target * slice_size_scaler` bytes). This
  is the spec's intended HQ rate-control knob: a flat / low-energy slice
  keeps the floor qindex (lossless-ish), while a busy slice raises its
  own qindex just enough to fit, instead of relying on a generous
  `slice_size_scaler` and silently truncating the slice on the wire
  (§13.5.2 bounded reads otherwise zero the tail). The HQ profile
  applies no §13.5.1 DC prediction, so each slice's coefficients are
  independent and may be quantised at its own qindex without any
  cross-slice coupling; the decoder already reads `qindex = read_nbits(7)`
  per slice (§13.5.2) so **no decode-side change is needed**. The HQ
  encode path was refactored to keep the *unquantised* coefficient
  pyramids and quantise each slice's region on the fly at its chosen
  qindex (the now-unused whole-picture `quantise_pyramid` helper is
  removed); the `None` path is byte-for-byte identical to the previous
  output (all 229 prior tests unchanged). A new public
  `encoder::hq_slice_qindexes` returns the per-slice
  `(qindex, max_component_length_byte)` vector for introspection,
  drift-proof against the emit path because both call the same
  `choose_hq_slice_qindex` / `hq_component_length_byte` helpers. Six
  integration tests in `tests/encoder_slice_qindex.rs`:
  `hq_adaptive_qindex_roundtrips_under_tight_budget` (tight budget →
  decoder accepts the stream and every slice fits its budget),
  `hq_adaptive_qindex_is_non_vacuous_on_busy_content` (a flat-quadrant +
  busy-checker picture keeps ≥1 cheap slice at the floor AND escalates
  ≥1 busy slice above it), `hq_adaptive_qindex_keeps_floor_and_matches_constant_on_flat`
  (flat content stays at the floor everywhere → stream byte-identical to
  the constant-qindex stream + bit-exact round-trip),
  `hq_adaptive_qindex_generous_budget_equals_none` (a budget above every
  slice's floor length reduces the adaptive path to the constant path),
  `hq_adaptive_qindex_is_deterministic`, and
  `hq_adaptive_qindex_tighter_budget_never_lowers_qindex` (a smaller
  target can only push qindexes up, never down — pins the search
  direction).

### Fixed

- **Intra DC subband-prediction `mean` rounding** (§5.4 / §13.3,
  round-118). `picture::intra_dc_prediction` (decoder) and
  `encoder::forward_dc_prediction` (encoder) computed the 3-neighbour
  prediction as `(a + b + c) // 3`, omitting the spec's unbiased-`mean`
  rounding term `(n // 2) = 1`: §5.4 defines `mean(S)` as
  `(s0 + … + s_{n-1} + (n // 2)) // n`, i.e. `(a + b + c + 1) // 3` with
  floor division. The missing `+1` left every intra picture's level-0 LL
  band ~1 LSB off after the IDWT. With the fix, **all five intra-only
  docs-corpus fixtures now decode bit-exactly against the ffmpeg
  reference** — `i-only-tiny`, `vc2-low-delay-tiny`,
  `vc2-low-delay-3pics`, `interlaced-720x576-i-only` (depth-4 IDWT) and
  `chroma-422-720x576` (4:2:2) — up from ~48–52 dB PSNR; their
  `docs_corpus` cases are promoted from `ReportOnly` to `BitExact`. The
  inter fixtures' intra reference frames also tighten (overall PSNR
  +3–5 dB). Encoder and decoder are kept in lockstep so the
  forward/inverse DC prediction stays a bit-exact pair.

- **`quant::quant_factor` u32 overflow at high qindex** (§13.2.1,
  round-114). `quant_factor(q)` computed `1u32 << (q/4)` and then
  `4 * base` / `503_829 * base` / … in `u32`, which overflows for
  `q >= 124` (and lower for the `q % 4 != 0` branches). The function was
  previously only ever called with `qindex = 0` so the overflow was
  dormant; the new §13.5.4 per-slice adaptive search drives `q` up to
  the 7-bit maximum (127) on busy slices and triggered it. The
  arithmetic now runs in `u64` and saturates at `u32::MAX`. Saturation
  is behaviour-preserving: a `qf` that large forward-quantises every
  8-bit-derived coefficient to 0 (`4*|x|/qf == 0`) and `inverse_quant`
  reconstructs 0 from a 0 qcoeff regardless of `qf`, so neither encode
  nor decode observes the clamp.

### Added

- VC-2 low-delay **custom quantisation matrix on the encoder**
  (§12.4.5.3 / §11.3.5 `quant_matrix()`, round-111). Both
  `EncoderParams` (HQ) and `LdEncoderParams` (LD) gain a
  `custom_quant_matrix: bool` field (default `false`) and a
  `with_custom_quant_matrix(matrix)` builder. When the flag is set,
  `write_transform_parameters` / `write_ld_transform_parameters` emit
  `custom_quant_matrix = True` followed by the explicit per-subband
  entries in the spec's exact read order (`QMATRIX[0][LL]`, then
  `QMATRIX[level][HL/LH/HH]` for `level = 1..=dwt_depth`) instead of the
  `custom_quant_matrix = False` flag that makes the decoder reconstruct
  the Annex E.1 default via `set_quant_matrix()`. This closes the
  encode-side half of the quant-matrix syntax — the decoder
  (`picture::parse_transform_parameters`) has parsed the custom matrix
  since the VC-2 interop work, but the encoder previously hard-wired the
  flag to `False`, so a non-default `EncoderParams::quant_matrix` was
  silently unrecoverable on decode. The shared `write_quant_matrix`
  helper byte-for-byte mirrors the decoder's read loop. Three
  integration tests in `tests/encoder_matrix.rs`:
  `hq_custom_quant_matrix_framing_roundtrips_q0` (a non-default all-zero
  custom matrix bit-exact at qindex=0 — proves the extra `read_uint`
  entries stay in lockstep with the encoder's writes, no bitstream
  desync), `hq_custom_all_zero_matrix_differs_from_default_at_q8` (the
  same picture encoded with the all-zero custom matrix vs the default
  matrix at qindex=8 produces *different* bitstreams and *different*
  reconstructions — proving the matrix actually travels and drives the
  per-subband quantisers; the custom path self-round-trips at 53.5 dB Y
  PSNR), and `ld_custom_quant_matrix_roundtrips_q0` (the LD
  slice-parameters → quant-matrix read order stays aligned: all-zero
  custom matrix at qindex=0 reconstructs a smooth gradient bit-exact).
  Note §11.3.5 *requires* the custom flag for `dwt_depth > 4`, where the
  Annex E.1 default table is undefined.
- Dirac core-syntax **VLC (non-arithmetic) intra reference encoder**
  (parse code `0x4C`, round-108). `encode_core_intra_picture_vlc` /
  `encode_single_core_intra_stream_vlc` emit a core-syntax intra reference
  whose per-codeblock entropy uses plain interleaved exp-Golomb instead of
  the binary arithmetic coder — the encoder counterpart to the decoder's
  long-present-but-previously-encoder-unreachable `decode_subband_vlc`
  (only hand-built unit-test blocks reached it before). The whole-picture
  framing (§12.2 picture header → §9.6.1 RETD → §11.3 transform parameters
  → §13.4.1 transform data) is bit-identical to the AC `0x0C` path; only
  the §13.4.2.2 entropy primitives change: §13.4.3.3 `zero_flag` becomes a
  single raw bit (`read_boolb()`), §13.4.3.4 `codeblock_quant_offset()`
  becomes `read_sintb()`, and §13.4.4 `coeff_unpack` becomes `read_sintb()`
  per coefficient with **no** neighbourhood / parent / sign conditioning
  (those condition only the arithmetic coder). The shared codeblock walk —
  the §13.4.3.3 all-zero skip, the §13.4.3.2 by-reference running quantiser
  and the §13.4.3.4 differential offset — is reused verbatim from the AC
  path, so the partition/skip/mode-1 behaviour matches. `using_ac()` =
  `(0x4C & 0x48) == 0x08` → false routes the decoder to the VLC subband
  reader. Because the VLC path applies no entropy-coder rounding, at
  `qindex = 0` (LeGall 5/3 dead-zone identity) it is **strictly lossless**
  and bit-exact on the synthetic testsrc fixture — including the V-plane
  steep gradient where the AC path carries a long-tolerated ~1-LSB
  roughness. Seven integration tests in
  `tests/encoder_intra_core_roundtrip.rs`:
  `core_intra_vlc_stream_uses_parse_code_0x4c` (parse-info walk asserts the
  `0x4C` picture code), `core_intra_vlc_self_roundtrip_yuv420_synth_testsrc`
  + `core_intra_vlc_self_roundtrip_constant_frame_is_bit_exact` (bit-exact
  Y/U/V at q=0), `core_intra_vlc_beats_ac_on_v_gradient` (VLC V matches the
  source while AC V does not; VLC PSNR ≥ AC PSNR per plane),
  `core_intra_vlc_multi_codeblock_skip_roundtrips` (4×4 partition, empty
  codeblocks coded as VLC skips → strictly smaller than the single-codeblock
  VLC stream, still bit-exact) and
  `core_intra_vlc_mode1_skip_does_not_advance_quantiser` (mode-1 VLC
  differential offset; a skipped codeblock leaves the running quantiser
  unchanged).
- Dirac core-syntax intra encoder **all-zero codeblock skip**
  (§13.4.3.3 `zero_flag`, round-103). `encode_subband_ac` now codes an
  empty codeblock of a partitioned subband as a skip (`zero_flag = True`)
  instead of a non-skip block followed by an explicit run of zero
  coefficients — both compressing the stream and exercising the
  decoder's previously-unreached `decode_subband_ac` skip branch from a
  self-produced stream. The skip decision is taken on the *quantised*
  coefficients (a high running quantiser that zeroes a codeblock turns it
  into a skip). Per §13.4.3.2 a skipped codeblock emits no quant offset
  and does **not** advance the by-reference running quantiser
  (`quant_idx += codeblock_quant_offset()` lives inside the
  `if skipped == False` branch); the two former encoder phases
  (quantise-all-then-emit) are merged into a single left-to-right
  codeblock walk so the skip decision, the running quantiser and the
  emitted symbols stay self-consistent with the decoder's read order.
  `codeblock_offset` now indexes the **non-skipped** codeblock ordinal
  rather than the absolute codeblock index, so the first non-skipped
  codeblock of every subband still carries offset 0. New integration
  tests in `tests/encoder_intra_core_roundtrip.rs`:
  `core_intra_skip_flag_compresses_and_roundtrips` (skip-aware stream
  strictly smaller than the single-codeblock stream and bit-exact) and
  `core_intra_skip_does_not_advance_quantiser_mode1` (the first
  non-skipped codeblock runs at the base quantiser regardless of how many
  empty codeblocks preceded it). The round-100
  `core_intra_multi_codeblock_mode1_cumulative_quant_testsrc` floors are
  relaxed to 44 dB on U/V: with the gradient fixture's empty subband
  halves now skipping, the non-empty codeblocks run at a lower quantiser
  and — because dead-zone round-trip exactness is not monotonic in `q` —
  the per-plane PSNR shifts (still near-lossless, still far above the
  ~37 dB reset-per-codeblock-bug collapse the test guards against).

### Fixed

- **Cumulative codeblock quantiser offset** (§13.4.3.2, round-100). The
  core-syntax coefficient decoder reset the effective quantiser to
  `base_q + delta` at the *start of each codeblock*, contradicting the
  spec's `quant_idx += codeblock_quant_offset()` — the offset is
  differential and accumulates **by reference** across the codeblocks of
  a subband in raster order. Both the AC path (`decode_subband_ac`) and
  the VLC path (`decode_subband_vlc`) in `picture_core` now carry the
  running quantiser forward instead of recomputing it from the subband
  base each codeblock. The bug was previously dormant (the only encoder
  that emitted core-syntax intra used a single codeblock per subband, so
  `codeblock_quant_offset()` was never read); it surfaces with the
  round-100 spatial-partition encoder below. Pinned by the new unit test
  `picture_core::tests::vlc_codeblock_quant_offset_accumulates_across_codeblocks`
  (two 1×1 codeblocks, offsets `+4, +4`, asserting the second codeblock
  dequantises at the cumulative `q = 8`, recovering `6`, rather than the
  reset `q = 4` recovering `3`).

### Added

- Dirac core-syntax **intra spatial partition** (§11.3.3
  `codeblock_parameters`) on the encoder. `CoreIntraEncoderParams` gains
  `codeblocks: Option<Vec<(u32, u32)>>` (per-level `(cbx, cby)` grid;
  level-0 LL is always forced to a single codeblock so its §13.3 DC
  prediction is preserved) and `codeblock_mode: u32` (0 = single
  per-subband quantiser; 1 = per-codeblock differential quantiser
  offset). `write_core_transform_parameters` now emits
  `spatial_partition_flag = 1` + the per-level counts + the codeblock
  mode when a grid is supplied; `encode_subband_ac` walks the subband
  codeblock-by-codeblock (matching the decoder's
  `decode_subband_ac`/`decode_subband_vlc`), emitting the §13.4.3.3
  ZERO_BLOCK skip flag (never skipped — coefficients are always present)
  for partitioned subbands and, under `codeblock_mode == 1`, the
  §13.4.3.4 differential quant offset (`0` for the first codeblock of
  every subband, `+1` thereafter → a strictly increasing running
  quantiser) for **every** codeblock including the single-codeblock LL
  band, then quantises the codeblock's coefficients at the running
  quantiser before AC-coding them. The default surface
  (`CoreIntraEncoderParams::default_intra`) is unchanged
  (`codeblocks = None`, single codeblock per subband). New integration
  tests in `tests/encoder_intra_core_roundtrip.rs`: multi-codeblock
  mode-0 bit-exact (constant frame) + near-lossless (testsrc), and
  multi-codeblock mode-1 cumulative-quantiser roundtrips that exercise
  the decoder fix end-to-end (mode-1 testsrc reconstructs at ~54 dB Y
  with the fix; a reset-per-codeblock decoder collapses it to ~37 dB,
  below the 48 dB assertion floor).

- Dirac inter encoder **post-OBMC bipred mode-only refinement pass**
  (round-95): the 2-ref analogue of the 1-ref round-80 post-OBMC
  re-evaluation. After `bipred_select_modes` picks its
  `{mode, mv1, mv2}` decisions from the round-91 widened candidate
  set, the new pass re-scores each block's mode under the full
  §15.8.5 OBMC blend with the neighbour grid frozen at the
  selector's output, choosing from the strict-superset trial set
  `{ current, Ref1Only(mv1), Ref2Only(mv2), Ref1And2(mv1, mv2) }` at
  the SAME MV pair the selector chose. This closes the cost-function
  gap between the selector's SAD-against-source metric and the
  decoder's OBMC-SSE-against-source reconstruction cost without
  disturbing the MV grid (so smooth-motion sub-pel choices are
  preserved — the camera-pan ffmpeg cross-decode floor is unchanged
  at ~53 dB). New public function
  `oxideav_dirac::encoder_inter::bipred_post_obmc_refine_modes`
  (mirrors the shape of `inter_select_int_pel_per_block`). New
  helpers `bipred_block_ref_value` (per-pixel reference value under a
  `BipredBlock` decision; computes `(v1 + v2 + 1) >> 1` for
  `Ref1And2` per §15.8.5 at `ref1_wt = ref2_wt = 1`,
  `refs_wt_precision = 1`) and `build_neighbour_sum_bipred` (the
  bipred analogue of `build_neighbour_sum`, summing weighted
  neighbour reference values across all 8 OBMC neighbours).
  Wired into `encode_bipred_inter_picture` immediately after
  `bipred_select_modes` returns and before
  `encode_block_motion_data_bipred` emits the §12.3 block_motion_data
  stream — gated by the new
  `InterEncoderParams::bipred_post_obmc_refine` field (default
  `true`; set to `false` for A/B testing against the pre-round-95
  behaviour). Strict-superset invariant pinned by the new test
  `bipred_post_obmc_refine_monotonic_per_block_obmc_sse`: for every
  block on the qpel camera-pan triplet, the post-pass per-block OBMC
  SSE under the frozen neighbour grid is ≤ the pre-pass per-block
  OBMC SSE. Integration test
  `bipred_post_obmc_refine_does_not_regress_no_residue` (ME-only,
  no residue) reports +0.80 dB Y self-roundtrip PSNR on the
  camera-pan fixture (51.44 → 52.24 dB) and asserts the pass never
  regresses picture-level PSNR by more than ε. The 1-ref P-path
  remains unchanged; the bipred path's `obmc_refine_me` per-ref
  refinement remains explicitly disabled (per the round-91 design
  note — refining each reference's MV independently against the
  source breaks the blend invariant), and the wavelet residue loop
  closes any remaining prediction error.

- Dirac inter encoder **bipred per-ref candidate-set widening to
  `{int-pel, half-pel, sub-pel}`** (round-91). Mirrors the 1-ref
  P-path's `inter_select_int_pel_per_block` strict-superset
  correctness invariant (round-73 + round-80 post-OBMC pass), scaled
  to the 2-ref bipred path. Round-39 evaluated 2 per-ref candidates
  (`{sub-pel, int-pel}`) and 4 bipred combinations; round-91 widens
  this to 3 per-ref candidates (`{int-pel, half-pel, sub-pel}`) and
  up to 3 × 3 = 9 bipred combinations (de-duplicated when the
  half-pel snap coincides with one of its peers, or when
  `bipred_mv_precision < 2` makes the half-pel grid degenerate). Per
  ST 2042-1 §11.2.5 `mv_precision` already enumerates int / half /
  qpel / ⅛-pel — no decoder-side change is required; the widening
  simply lets the encoder pick the same per-block MV from a strict
  superset. Tie-bias preserves the round-39 ordering: int-pel ≥
  half-pel ≥ sub-pel under equal SAD (smaller §15.8.11 8-tap-filter
  contribution → less OBMC blend drift with neighbour blocks).
  New helper `round_mv_to_half_pel(mv, p)` (private): degenerates
  to `round_mv_to_int_pel` at `p == 0`, identity at `p == 1` (MV
  already on half-pel grid), snap-to-nearest-multiple-of-`2^(p-1)`
  with ties toward zero at `p >= 2`. The strict-superset invariant
  is pinned by the new test `bipred_widened_candidate_set_monotonic_per_block_sad`:
  for every block on the camera-pan triplet (qpel), the round-91
  selector's per-ref SAD and bipred SAD are both ≤ the inlined
  round-39 reference's SADs. Diagnostic test
  `bipred_widened_set_exercises_int_half_and_qpel_grids` confirms
  the new half-pel candidate is genuinely picked (240 half-pel MVs
  across three fixtures, joining 407 int-pel and 240 qpel — the
  widening is non-vacuous). A half-pel-favourable
  `camera_pan_64` triplet (anchors at qpel 0 and 4, midpoint at
  qpel 2 → exact half-pel MV) is added by
  `bipred_widened_set_half_pel_favourable_self_roundtrip_no_residue`,
  measuring +12.23 dB Y self-roundtrip uplift over the int-pel-only
  ceiling (30.36 → 42.59 dB, residue OFF). The pre-existing
  `ffmpeg_cross_decodes_camera_pan_bipred_with_subpel_gain` test
  shows the round-39 fixture is unchanged (52.53 dB → 52.53 dB) —
  the SAD landscape on cosine-shaped pan-by-1-pel content was
  already converged on the qpel grid, so the new half-pel
  candidate is dominated by the qpel pick and the selector
  rationally keeps it; the strict-superset invariant guarantees
  the unchanged outcome rather than a regression.

- Dirac inter encoder **post-OBMC second adaptive sub-pel-vs-integer-pel
  pass for the 1-ref (P-picture) path** (round-80). Mirrors the
  pre-OBMC selector landed in round-73, but runs `inter_select_int_pel_per_block`
  a **second** time **after** [`obmc_refine_me`] has converged on a
  refined sub-pel MV grid. Motivation: `obmc_refine_me`'s ±1
  sub-pel-unit-per-pass step can drift a block off the integer-pel
  anchor that the pre-OBMC selector chose — typically to help a
  neighbour's blend — and once the neighbour grid finishes converging
  the drifted block's integer-pel peer may again be the lower-OBMC-SSE
  candidate. The post-OBMC pass evaluates exactly that integer-pel
  peer against the converged sub-pel MV under the same §15.8.5
  weighted-blend reconstruction the decoder will run, picking
  whichever gives lower per-block OBMC SSE (ties biased to integer-pel,
  same as round-73). Strict superset of the input candidate set
  (`{ current, round_to_int_pel(current) }`) → cannot regress per-block
  OBMC SSE; the load-bearing monotonicity invariant
  `inter_select_int_pel_monotonic_per_block_obmc_sse` already pins
  the helper's contract.
  New `InterEncoderParams::inter_adaptive_int_pel_post_obmc: bool`
  knob, defaults `true`; set to `false` to disable for A/B regression
  testing against pre-round-80 behaviour. At `mv_precision == 0` the
  helper takes an early-return path. At `obmc_refine_passes == 0` the
  post-OBMC pass is *almost* a no-op (the pre-OBMC selector already
  ran on the same grid) — kept for tie-bias safety. On the synthetic
  `translate(+2,-1)` and `camera_pan_64(+1,0)` fixtures the post-OBMC
  selector is a no-op (37.15 dB and 52.04 dB Y self-roundtrip
  respectively before/after, identical to the pre-round-80 baseline)
  — sub-pel ME + pre-OBMC selector together already converge on the
  optimal MV grid. The win cases sit on real video content where
  OBMC refinement's neighbour-aware ±1 steps push integer-pel-snapped
  blocks off the integer anchor to favour a neighbour with sub-pel
  motion, after which the post-OBMC pass un-drifts them. 4 new tests
  (1 non-regression + 1 determinism + 1 bit-exact self-roundtrip + 1
  unit pinning that the selector's only mutation is `current ↦
  round_to_int_pel(current)`); all 202 dirac tests pass.

- Dirac inter encoder **per-block adaptive sub-pel-vs-integer-pel
  selection for the 1-ref (P-picture) path** (round-73). Mirrors the
  bipred adaptive precision landed in round-39 (`bipred_select_modes`).
  After [`subpel_search_me`] produces the refined sub-pel MV grid and
  **before** [`obmc_refine_me`] runs, the new
  `inter_select_int_pel_per_block` helper scores each block's MV under
  the §15.8.5 weighted-blend reconstruction at both its sub-pel-refined
  position AND the nearest integer-pel-rounded peer (`round_mv_to_int_pel`),
  keeping whichever gives lower per-block OBMC SSE against the source.
  Ties bias toward the integer-pel MV (smaller decoder-side filter
  contribution → less risk of 8-tap half-pel-filter smoothing leaking
  multi-LSB into neighbouring blocks' OBMC blends).
  New `InterEncoderParams::inter_adaptive_int_pel: bool` knob, defaults
  `true`; set to `false` to disable for A/B testing against the
  pre-round-73 behaviour. At `mv_precision == 0` (integer-pel ME) the
  helper takes an early-return path — every MV is already integer-pel,
  so rounding is the identity. The per-block min-of-two-SSEs design is
  a strict superset of the legacy candidate set → cannot regress
  per-block OBMC reconstruction cost (load-bearing invariant pinned by
  `inter_select_int_pel_monotonic_per_block_obmc_sse`). On the
  synthetic `translate(+2,-1)` / `camera_pan_64` fixtures sub-pel ME
  already converges to integer-pel for integer motion or pure-sub-pel
  for smooth motion, so the selector is a no-op there — the win cases
  sit on real video content with sharp text/occluders that cause
  sub-pel ME to bottom out at fractional offsets the OBMC blend
  doesn't reward. 4 new tests (3 unit + 1 self-roundtrip
  non-regression + 1 determinism); all 197 dirac tests pass.

## [0.0.6](https://github.com/OxideAV/oxideav-dirac/compare/v0.0.5...v0.0.6) - 2026-05-06

### Other

- drop stale REGISTRARS / with_all_features intra-doc links
- drop dead `linkme` dep
- bipred encoder: per-block adaptive sub-pel-vs-int-pel selection (qpel default, +4.4 dB on smooth motion)
- registry calls: rename make_decoder/make_encoder → first_decoder/first_encoder
- bipred encoder: integer-pel ME lifts ffmpeg cross-decode from 42 to ~50 dB
- auto-register via oxideav_core::register! macro (linkme distributed slice)
- unify entry point on register(&mut RuntimeContext) ([#502](https://github.com/OxideAV/oxideav-dirac/pull/502))

### Changed

- Bipred B-picture encoder (`encode_bipred_inter_picture`) now uses
  **integer-pel ME** by default (`bipred_mv_precision = 0`). Quarter-pel
  ME produced an ~8 dB cross-decode penalty because our half-pel
  interpolation convention differs from ffmpeg's at the 2-ref OBMC blend
  stage, accumulating across blocks. Integer-pel eliminates all
  convention differences; the wavelet residue then closes the
  prediction-error loop exactly. ffmpeg cross-decode PSNR on the
  complementary-bar fixture improves from ~42 dB to ~50 dB. The 1-ref
  (P-picture) path is unaffected and continues to default to
  quarter-pel (`mv_precision = 2`).

### Added

- `InterEncoderParams::bipred_mv_precision` field — controls MV
  precision for B-pictures independently of the 1-ref `mv_precision`
  knob. Default `0` (integer-pel). Set to `2` to restore the old
  quarter-pel behaviour at an ~8 dB cross-decode cost.

## [0.0.5](https://github.com/OxideAV/oxideav-dirac/compare/v0.0.4...v0.0.5) - 2026-05-05

### Other

- 2-ref bipred B-picture encoder + decoder fix (42 dB ffmpeg)

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

- Dirac inter encoder **2-reference bipred** (B-picture, parse code
  `0x0A`) — the `encoder_inter` module now emits 2-reference inter
  pictures alongside the existing 1-reference path. New
  `encode_bipred_inter_picture` writes two §9.6.1 reference deltas, the
  §11.2 picture-prediction parameters, the §12.3 `block_motion_data`
  with both `v1x/v1y` and `v2x/v2y` MV blocks, and (via the existing
  residue path) the §11.3 wavelet residue computed against the
  decoder's bipred OBMC reconstruction. Per-block decision search
  (`bipred_select_modes`) runs sub-pel ME against each reference
  independently and picks `Ref1Only` / `Ref2Only` / `Ref1And2` per
  block by minimising SAD against the source — bipred carries a small
  lambda-style penalty (`BIPRED_PENALTY = 64`) so blocks with no
  averaging benefit fall back to the simpler 1-ref modes. New stream
  wrapper `encode_core_intra_then_bipred_stream` chains two `0x0C`
  intra references with one `0x0A` B picture so the full sequence
  stays in one parse-code family. PSNR results on the new
  complementary-bar fixture (a horizontal bar visible only in ref1, a
  vertical bar only in ref2, B has both at half intensity — neither
  single-ref MV alone can reach the target):
  - **Self-roundtrip residue ON: ∞ dB (bit-exact)** at qindex = 0.
  - **Self-roundtrip residue OFF: 32.60 dB bipred vs 20.18 dB 1-ref**
    (+12 dB, hard-asserted A/B).
  - **ffmpeg cross-decode: 42.27 dB Y** on the bipred B picture
    (hard-asserted, vs 25 dB defensive floor).
  6 new tests in `tests/encoder_bipred_roundtrip.rs` (block-motion-data
  roundtrip, constant-frames smoke, mode-count diagnostic, no-residue
  A/B, residue-on self-roundtrip, ffmpeg cross-decode); all 188 dirac
  tests pass.

### Fixed

- Decoder bipred (parse code `0x0A`) was silently dropping the §12.3
  `block_ref_mode` ref2 bit on 2-reference non-global pictures.
  `decode_block_motion_data` derived `MotionCtx::num_refs` from
  `params.global2.is_some()` (which is only `true` when global motion
  is enabled), so `block_ref_mode` only read the ref1 bit and every
  block fell into `Ref1Only` — hiding the entire bipred path. The
  fix uses the `num_refs` already passed into `decode_block_motion_data`
  (which is parse-code-derived). The existing 1-ref decode path (parse
  code `0x09`, `num_refs = 1`) is unaffected; the existing global-2-ref
  fallback through `params.global2.is_some()` ends up giving the same
  answer for that case. The `corpus_i_p_b_320x240` ReportOnly fixture's
  bipred B-frame Y PSNR moves accordingly (was previously masked by the
  same OBMC-convention floor; this is the structural unblock).

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
