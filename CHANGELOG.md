# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **§11.2.6 global-motion model estimation** (round-386) — the
  affine / perspective generalisation of round-382's pan estimator;
  all encoder policy on the proven §11.2.6/§15.8.8 emission path.
  - `estimate_global_motion_config(.., GlobalMotionModel)` with three
    model families: `Pan` (the round-382 median fit), `Affine`
    (6-parameter least squares `v(x) ≈ A·x + b` on the committed ME
    grid with one trimmed refit against foreground outliers) and
    `Perspective` (alternating linearised fit of the §15.8.8
    `v = (1 − 2^−β·cᵀx)(2^−α·A·x + b)` model — c-fit with the affine
    part frozen, then an affine refit on the de-perspectived targets).
    Quantisation onto the integer parameterisation picks
    `2^zrs_exp ≥ 4·max(w, h)`, rejects a perspective vector that could
    zero the on-frame denominator, and runs a per-component local
    search (matrix entries ±2 steps, pan −2..+1) scored on the exact
    floor-rounded `global_mv` output — plain LS rounding systematically
    underestimates axis-aligned staircase slopes. Per-block gmode is
    decided by measured SAD (field vs own block MV through the same
    §15.8.10 sampling), tie to global. Measured on a whole-frame zoom:
    fraction 1.0, bit-exact self-roundtrip at `qindex = 0`, stream
    3569 → 2477 bytes (−31%) vs block motion.
  - `estimate_global_bipred_config` — two models per B-picture (one
    per reference, independent ME grids at `bipred_mv_precision`) with
    a conservative AND-rule gmode grid: a block goes global only when
    the field beats its ME MV against **both** references.
  - `InterEncoderParams::auto_global_motion: Option<AutoGlobalMotion>`
    (default `None`) — the sequence driver estimates a model per
    picture and applies it iff the fraction clears `min_fraction`,
    resolved before the residue qindex picker so rate control measures
    the exact stream it emits; `InterPictureRate` grows
    `global_fraction` / `global_applied` telemetry. Explicit
    `global_motion` configs always win; a `min_fraction > 1.0` run is
    telemetry-only and byte-identical to auto-off.
  - External-oracle results: an estimator-produced pan model
    cross-decodes **bit-exactly**; on a non-trivial matrix the oracle
    is characterised (and pinned byte-for-byte) as **pan-only** — it
    applies the §15.8.8 field evaluated at `(0, 0)` to every pixel,
    ignoring the per-pixel matrix term of the spec's
    `global_mv(ref, ref_num, x, y, c)` process, which our decoder
    implements per-pixel.
  - 48-case random-warp fuzz sweep (Pan/Affine/Perspective ×
    `mv_precision 0..=3`, warps applied with the decoder's own field
    arithmetic) + degenerate-input sweep (solid / pulse frames) over
    the estimator: clean 2-frame decode + deterministic encode on
    every fitted model.

### Fixed

- **`GlobalMotionConfig::pan_tilt_all` emitted the identity matrix**
  (round-386): §15.8.8 computes the displacement
  `v = (A·x + 2^ez·b)·m / 2^(ez+ep)` directly, so the identity `A`
  made the constructor's field the position-proportional stretch
  `(x + dx + 1, y + dy + 1)` instead of the documented constant pan.
  Self-roundtrips stayed bit-exact (the encoder mirrors the decoder),
  but the prediction was wrong-shaped and the residue absorbed it.
  Now the zero matrix, matching `estimate_global_pan_config`'s proven
  convention; regression-pinned at frame-corner extremes plus the wire
  round-trip.

- **§11.2.6 global-motion encoder path** (round-382) — closes the
  `docs/video/dirac/dirac-fixtures-and-traces.md` "Global motion
  compensation (`globalmc_flag=1`)" corpus gap on the **encode** side.
  The decoder already parsed global motion parameters (§11.2.6), the
  per-block global-mode flag (§12.3.3.2), and reconstructed via the
  §15.8.8 `global_mv` affine-perspective field; the inter encoder now
  emits streams that exercise all of it.
  - `write_global_motion_parameters` — the exact bitstream inverse of
    the decoder's `parse_global_motion_parameters`: the `pan_tilt` /
    `zoom_rotate_shear` / `perspective` omission-flag triple. Paired
    with `effective_global_params`, which canonicalises a caller's
    [`GlobalParams`] to the value the decoder reads back (a non-zero
    `persp_exp` with a zero perspective vector never reaches the wire,
    so the encoder builds its prediction from the effective value).
    Round-tripped by `global_motion_parameters_roundtrip` across every
    flag combination.
  - `InterEncoderParams::global_motion: Option<GlobalMotionConfig>` (a
    new `GlobalParams` pair + optional per-block gmode grid; default
    `None`). `write_picture_prediction_parameters` /
    `picture_prediction_params_from` gained a `num_refs` argument and
    now emit `using_global = true` + one `global_motion_parameters`
    block per reference. Pinned by
    `picture_prediction_parameters_global_roundtrips` (1-ref emits
    global1 only; 2-ref emits both).
  - **§12.3.3.2 per-block global-mode emission.** `encode_prediction_modes`
    now emits the block-global flag for every non-intra block under
    global motion, coded as a residual against the §12.3.6.4
    neighbour-majority prediction (`block_global_prediction_enc` +
    `propagate_gmode` mirror the decoder). Global blocks carry **no** MV
    residual — `encode_vector_elements` already skipped them, and now
    also carries each block's `gmode` in its MV-prediction context so the
    §12.3.6.1 neighbour median matches the decoder exactly (previously
    the local context always saw `gmode = false`). `encode_block_motion_data`
    / `encode_block_motion_data_bipred` gained a `gmode: Option<&[bool]>`
    grid; `resolve_gmode_grid` derives it from the config (whole-picture
    global by default). Pinned by
    `block_motion_data_global_flags_roundtrip`: a mixed global/block-motion
    16-block grid recovers every `gmode` flag through
    `decode_block_motion_data`, and the non-global blocks recover their
    MVs.
  - **End-to-end global-motion P-picture** (parse code `0x09`).
    `encode_inter_picture` threads the global config through the block
    motion data **and** the §11.3 residue: the OBMC prediction the
    encoder subtracts is built from a [`PictureMotionData`] carrying
    `gmode` blocks + `global1`, so it reconstructs the same §15.8.8
    `global_mv` affine field the decoder does.
    `tests/encoder_inter_roundtrip.rs::intra_then_inter_global_motion_p_picture_roundtrips`
    encodes a whole-picture global model (zero affine matrix ⇒ constant
    `pan_tilt + 1` translation) matching a +4 shift and decodes it
    through the full registered pipeline: intra anchor bit-exact,
    global-motion inter frame Y ≈ 35.8 dB / chroma ≈ 70 dB — the first
    time the §11.2.6/§12.3.3.2/§15.8.8 global-motion decode path is
    exercised by an oxideav-produced stream.
  - New `GlobalMotionConfig::pan_tilt_all(dx, dy)` convenience
    constructor (whole-picture pure-translation global model against
    ref1).
  - **End-to-end global-motion bipred B-picture** (parse code `0x0A`).
    `encode_bipred_inter_picture` threads the config the same way the
    1-ref path does: both references' `global_motion_parameters` go on
    the wire, `build_motion_from_bipred_grid` carries the gmode grid +
    `global1`/`global2` into the residue's OBMC prediction, and global
    blocks skip the MV residual for **both** references
    (v1x/v1y/v2x/v2y).
    `tests/encoder_bipred_roundtrip.rs::bipred_global_motion_b_picture_roundtrips`
    drives the complementary-bar fixture through a whole-picture
    zero-translation global model on both refs: anchors bit-exact, the
    global-motion B frame reconstructs **bit-exactly** (∞ dB) through
    the §15.8.5 two-ref blend + qindex-0 residue.
  - **Global-field variant coverage** (round-382, after the terminator
    fix below unblocked the all-global case): three more end-to-end
    P-picture tests in `tests/encoder_inter_roundtrip.rs` —
    `intra_then_inter_global_zoom_field_roundtrips` (identity matrix at
    `zrs_exp = 4` ⇒ a per-pixel 0..=4-pel affine ramp; ≥ 60 dB pins
    encoder/decoder §15.8.8 field agreement across the whole plane),
    `intra_then_inter_global_motion_qpel_roundtrips` (`mv_precision = 2`,
    the field in qpel units through the §15.8.10 sub-pel sampler), and
    `intra_then_inter_mixed_global_and_block_motion_roundtrips` (left
    half of the block grid global, right half block-motion ME with MV
    residuals — the §12.3.6.1 global-block exclusion from the MV median
    and the §12.3.6.4 gmode majority prediction across the boundary,
    ≥ 60 dB). Discovering that the first two plateaued at the exact
    no-residue PSNR while 15-of-16-global closed bit-exactly is what
    exposed the §B.2.7.1 terminator bug.
  - **Global-motion fuzz arm**
    (`tests/encoder_inter_fuzz_oracle.rs::global_motion_parameter_sweep_never_panics`):
    120 combinations of field shape (pure pan / zoom ramp / huge
    out-of-frame rotation-shear that lands every fetch on the §15.8.9
    edge clamp / active perspective with per-pixel sign-varying `m` /
    extreme `zrs_exp = 12` magnitudes) × per-block gmode grid
    (all-global, half, sparse 1-in-7, alternating) × `mv_precision
    {0, 2}` × residue {q0, q64, off}. Every combination encodes and
    round-trips to exactly 2 frames — no panic, no arith desync, no
    livelock.
  - **Sequence-driver integration**
    (`tests/encoder_inter_sequence_rate.rs::sequence_driver_threads_global_motion`):
    the multi-picture rate-controlled driver clones `inter_params` per
    picture, so a `GlobalMotionConfig` reaches every `0x09` unit — the
    test parses each emitted PPP back (`using_global` + the caller's
    `pan_tilt` on the wire on all 3 inter pictures), the chain decodes
    one frame per input, and every satisfiable PerPicture residue
    request stays within budget. The `inter_residue_bytes_at_qindex`
    estimator resolves the same gmode grid + effective global params the
    emitter uses (`build_inter_residue_pyramids`), so rate control
    measures the true global-motion residue.
  - **External-oracle cross-decode**
    (`tests/ffmpeg_interop.rs::ffmpeg_decodes_our_global_motion_p_stream_bit_exact`,
    gated on validator availability): a whole-picture global-motion
    1-ref P stream (core `0x0C` anchor + `0x09` inter, constant
    `(-4, 0)` field, no MV residuals on the wire) is accepted by the
    external oracle and the inter frame reconstructs **bit-exactly** on
    all three planes — independent validation of the §11.2.6 parameter
    emission, §12.3.3.2 flag coding, §15.8.8 field arithmetic, and the
    exact terminator in one stream.
  - **Global-motion estimation** — `estimate_global_pan_config` fits a
    pan/tilt model from the encoder's own ME grid (component-wise median
    MV = dominant translation `t`; zero affine matrix with
    `pan_tilt = t - (1, 1)` makes the §15.8.8 field the constant `t`)
    and marks exactly the blocks whose ME MV equals `t` as global — a
    pure re-labelling that provably cannot change the prediction, it
    only sheds the matching blocks' MV residuals from the wire. Returns
    the config + the matched fraction so the caller can gate on motion
    coherence. Unit-pinned
    (`estimate_global_pan_matches_dominant_translation`: whole-frame
    integer pan over a textured field → fraction 1.0, field == median at
    every probe, per-block gmode consistency) and end-to-end
    (`estimated_global_motion_roundtrips_and_matches_block_motion_quality`:
    the estimated-global stream decodes bit-exactly, identical quality
    to the block-motion encode, marginally smaller stream).

- **§11.3.3 spatial-partition (codeblock grid) for the inter-residue
  encoder** (round-370) — closes the lib-doc "still unsupported:
  per-codeblock partitioning for the residue path" gap. The §11.3
  wavelet-residue path now accepts an optional per-level
  `(codeblocks_x, codeblocks_y)` grid (`ResidueParams::codeblocks`) plus
  `ResidueParams::codeblock_mode`, the inter-residue analogue of the
  core-intra encoder's spatial partition. Each HL/LH/HH subband splits
  into a grid of codeblocks, each carrying a §13.4.3.3 `ZERO_BLOCK` skip
  flag and, under `codeblock_mode == 1`, a §13.4.3.4 differential
  quantiser offset; the running quantiser accumulates by reference only
  across non-skipped codeblocks (§13.4.3.2). The emitter
  (`write_residue_transform_parameters` now serialises the grid + mode;
  the new codeblock walk requantises each codeblock in place at its
  running quantiser) is a **byte-for-byte mirror** of the proven
  core-intra `encode_subband_ac` — a new lib unit test
  (`cb_residue_bytes_match_intra_core_byte_for_byte`) asserts the two
  emit identical AC bytes on an identical band + grid. The decoder reads
  it through the shared `picture_core::decode_subband` walk
  (`spatial_partition_flag` already round-tripped on the decode side).
  With reversible LeGall 5/3 at `qindex = 0` the residue round-trips
  bit-exactly whenever every codeblock is ≥ 4×4 samples. (*Update,
  round-382*: the sub-4×4 "near-lossless §B.2.7.1 AC-terminator
  roughness" caveat that originally qualified this entry was the
  terminator bug fixed below — sub-4×4 and 1×1-sample codeblocks are now
  bit-exact too, pinned by `codeblock_sub4x4_grid_q0_legall_bit_exact`.)
  - `tests/encoder_inter_residue_codeblocks.rs` (5 tests): mode-0 2×2
    grid bit-exact (Y/U/V); mode-1 running-quantiser lockstep; the
    partitioned stream differs from the single-codeblock stream yet
    reconstructs identically at q=0; a realistic per-level
    `[(1,1),(2,2),(2,2),(2,2)]` grid bit-exact (mode 0) + lockstep
    (mode 1).
  - The codeblock grid is wired into **both** inter residue emission
    sites — the 1-ref (`0x09`) path and the 2-ref bipred (`0x0A`) path —
    via the shared `emit_residue_components` dispatch.
    `tests/encoder_bipred_roundtrip.rs::bipred_with_codeblock_residue_recovers_b_frame`
    confirms the bipred B-frame round-trips bit-exactly through the
    per-level codeblock grid at qindex 0.
  - `tests/encoder_inter_fuzz_oracle.rs::residue_codeblock_grid_sweep_never_panics`
    — a 60-combination robustness sweep over codeblock grids (uniform
    2×2 / 4×4, a per-level split, an asymmetric `(4,1)` grid, and a
    pathologically fine `(8,8)` grid that drives 1×1-sample and
    empty codeblocks at the deepest levels) × both `codeblock_mode`
    values × qindex {0, 24, 96} (so the skip path fires as the quantiser
    zeroes codeblocks) × {LeGall 5/3, Haar}. Every combination decodes to
    exactly 2 frames with no panic and no arithmetic-coder desync.

### Fixed

- **§B.2.7.1 arithmetic-encoder terminator: spurious follow bit corrupted
  the tail of every AC block** (round-382). `ArithEncoder::finish()`
  injected an artificial `follow_bits += 1` before emitting the 16-bit
  disambiguation value `T = low + 0x4000`, inserting one `!b0` bit
  between `T`'s top bit and its 15-bit tail — shifting the tail one
  position right. With a loose final interval the corrupted value still
  landed inside `[low, low + range)` and everything decoded; with a
  tight interval (strongly-adapted contexts near the end of a block,
  e.g. long runs of one symbol) the final few symbols misdecoded. This
  was the root cause of every long-tolerated "§B.2.7.1 AC-terminator
  roughness":
  - the AC core-intra testsrc V plane (V ≈ 24 dB "1-LSB gradient
    roughness", the CHANGELOG followup "carry-resolved flush") is now
    **bit-exact on all three planes** —
    `core_intra_self_roundtrip_yuv420_synth_testsrc` upgraded from
    `psnr >= 22 dB` floors to `assert_eq!`;
  - `core_intra_vlc_beats_ac_on_v_gradient` (which pinned the VLC/AC
    *contrast*) rewritten as `core_intra_vlc_and_ac_agree_at_q0`: both
    entropy coders now reconstruct bit-exactly at qindex 0;
  - the bipred camera-pan ffmpeg cross-decode integer-pel baseline went
    from ~50 dB to **bit-exact (∞ dB)** —
    `ffmpeg_cross_decodes_camera_pan_bipred_with_subpel_gain`'s
    "qpel beats int by ≥ 2 dB" premise was vacuous against a lossless
    baseline and is rewritten as a ≥ 60 dB floor on both variants;
  - sub-4×4-sample (and 1×1-sample) codeblocks in the §11.3.3 spatial
    partition now round-trip **bit-exactly** at qindex 0 — new
    `codeblock_sub4x4_grid_q0_legall_bit_exact` drives an 8×8 grid on a
    depth-3 transform (1×1-sample codeblocks at the deepest levels) to
    exactness, retiring the round-370 "must be ≥ 4×4 samples" caveat;
  - an all-global 16×16 motion grid (512 heavily-biased prediction-mode
    symbols) decoded block (12, 15)'s rmode wrong — the failure that
    exposed the bug.
  Discovered via the round-382 global-motion work; minimal repro pinned
  by `arith::tests::terminator_biased_stream_decodes_every_symbol`
  (symbol 504 of the exact 512-symbol pattern misdecoded before the
  fix) plus a 200-case seeded random sweep
  (`terminator_random_streams_decode_every_symbol`) over length ×
  bias × context count. The termination is now the WNC protocol with
  the correctness argument spelled out in the comment: `T >= low`,
  `T + 1 <= low + range` (range > 0x4000 after renormalise), and
  `T <= 0xFFFF` from the inductive invariant `low + range <= 0x10000`,
  so the decoder's past-end 1-extension stays inside the final interval
  for every remaining symbol resolution.

### Changed

- **Inter-residue rate control now accounts for the §11.3.3 codeblock
  grid** (round-370). `inter_residue_bytes_at_qindex` (the byte-cost
  estimator behind `pick_inter_residue_qindex` /
  `inter_residue_qindex_diagnostic`) now dispatches through the same
  `emit_residue_components` the picture emitter uses, so the estimate
  reflects the codeblock grid header + per-codeblock skip flags +
  per-codeblock requantise when `ResidueParams.codeblocks` is `Some`
  (previously it always measured the flat single-codeblock layout, which
  would mis-size a codeblock-partitioned residue). New regression:
  `tests/encoder_inter_residue_rate.rs::picker_accounts_for_codeblock_grid`.
  The multi-picture inter sequence driver
  (`encode_inter_sequence_with_residue_target[_report]`) already threads
  `ResidueParams.codeblocks` + `codeblock_mode` through every inter
  picture (it clones the residue config and overrides only `qindex`), so
  codeblock residue now integrates end-to-end with the four-variant
  (PerPicture / Cbr / Vbv / VbvHysteresis) rate control — pinned by
  `tests/encoder_inter_sequence_rate.rs::sequence_driver_threads_codeblock_grid`.

- **High-bit-depth (10/12-bit) intra encode → decode round-trip
  coverage** (round-345) — closes the `docs/video/dirac/dirac-fixtures-
  and-traces.md` "bit depths > 8" corpus gap on the **decode** side. The
  decoder's §10.5.2 `video_depth`-parameterised reconstruction (§15.9
  clip + §15.10 output offset + 16-bit LE plane packing) was correct but
  untested past 8 bits, because every upstream fixture is 8-bit and the
  encoder API only accepted `&[u8]`.
  - New `&[u16]` encode entry points so deeper samples reach the forward
    DWT unclipped: `encode_hq_intra_picture_u16`,
    `encode_single_hq_intra_stream_u16`, `encode_ld_intra_picture_u16`,
    `encode_single_ld_intra_stream_u16`. The forward-transform helpers
    were refactored to share a sample-width-agnostic core
    (`forward_component_with` / `forward_component_ld_with`) and the
    slice-packing loops to take pre-built `i32` coefficient pyramids
    (`encode_hq_intra_picture_from_pyramids` /
    `encode_ld_intra_picture_from_pyramids`); the existing `&[u8]` paths
    delegate unchanged.
  - New full-range signal-range presets `SignalRange::PRESET_10BIT_FULL`
    / `PRESET_12BIT_FULL` (offset/excursion centred so a full
    `[0, 2^N-1]` component round-trips symmetrically; emitted as §10.3.8
    *custom* ranges, `preset_idx = 0`) plus depth-aware sequence builders
    `make_minimal_sequence_with_signal_range` /
    `make_minimal_sequence_ld_with_signal_range`.
  - `tests/encoder_high_bit_depth_roundtrip.rs` (7 tests): HQ 10-bit is
    bit-exact across all three chroma formats × six reversible wavelets;
    HQ 12-bit 4:2:0 is bit-exact (four wavelets); a flat mid-grey case
    isolates the §15.10 output offset; LD 10-bit 4:2:0 reconstructs at
    ≥50 dB through the §13.5.1 DC-prediction slice path; HQ 10-bit is
    bit-exact across every spec-allowed `dwt_depth` (1..=5, depth 5 via a
    custom matrix) and through the §12.4.4 asymmetric (horizontal-only)
    transform (`dwt_depth_ho > 0`). The deeper 10-bit coefficient
    magnitudes need a wider HQ `slice_size_scaler`, which the tests set.
- **§15.8.7 `pixel_pred` / §15.8.2 `motion_compensate` global-motion
  branch coverage** (round-337) — two tests close the lone untested arm
  of the OBMC prediction chain: the `block.gmode == true` path that
  switches the per-pixel fetch from the block's own `mv[ref_num - 1]` to
  the affine-perspective `global_mv(g, x, y)` field. Round-331 unit-tested
  `global_mv` as a pure function, but every existing
  `pixel_pred` / `block_mc` / `motion_compensate` test used
  `gmode: false`, so the §15.8.7 global arm and its propagation through
  `block_mc` → OBMC weighting → §15.8.2 accumulation was never executed.
  - `pixel_pred_global_mode_uses_global_mv_not_block_mv` drives an 8×8
    distinct-per-cell reference through `pixel_pred` with a constant
    `(2, 0)` global field (zero affine matrix + zero perspective collapses
    `global_mv` to a uniform translation) and a deliberately wrong block
    MV `(-4, -4)`; it asserts every interior fetch reads the reference two
    columns to the right (the global vector) and explicitly rejects the
    block-MV result.
  - `motion_compensate_global_mode_shifts_reference_uniformly` runs a full
    16×16 picture with every block `gmode = true` (and a bogus per-block
    MV) through §15.8.2; since the §15.8.6 overlap weights sum to 64, the
    zero-residue reconstruction reproduces the reference uniformly shifted
    by the constant global field, and the test pins both the exact shift
    and a non-identity displacement.
  Sourced exclusively from `docs/video/dirac/dirac-spec-latest.pdf`
  §15.8.7 / §15.8.8 / §15.8.2 / §15.8.6. No external library source, no
  web search. Library test count: 239 → 241 (+2).

- **§15.8.8 `global_mv` unit tests** (round-331) — four hand-computed
  cases lock in the affine-perspective global-motion vector field:
  pure pan/tilt (verifying the per-axis `b[0]`/`b[1]` translation),
  pure zoom (including a floor-shift on a negative vertical vector),
  perspective magnitude modulation (`m = 2^ep − c·x`), and the
  origin-degenerate case. Closes the lone untested branch in the
  motion-compensation chain.

- **Inter sequence driver — leaky-bucket VBV + VbvHysteresis residue
  rate control** (round-326) — `encoder_inter::InterRateControl` grows two
  variants beyond the round-320 `PerPicture` / `Cbr` pair, so the
  multi-picture inter sequence driver
  (`encode_inter_sequence_with_residue_target` / `_report`) now offers the
  same four rate-control strategies the HQ/LD intra
  `encode_*_sequence_with_size_target` drivers have:
  - `InterRateControl::Vbv { buffer_bytes }` — leaky-bucket: identical
    feedback to `Cbr` (`carry = Σ(actual − target)` residue bytes), but the
    savings end of the accumulator is clamped at `-buffer_bytes` after each
    picture, so a run of undershooting inter pictures cannot bank unbounded
    headroom. The next picture's request `target − carry` is bounded above
    by `target + buffer_bytes` (an instantaneous peak residue-size cap).
    Overshoot debt (`carry > 0`) is never clamped — debt repayment is
    mandatory. `buffer_bytes == 0` forfeits all savings (surplus stays
    `>= 0`, every request `<= target`).
  - `InterRateControl::VbvHysteresis { buffer_bytes, max_drain_per_picture }`
    — drain-rate-limited leaky-bucket: same bucket fill / forfeit semantics
    as `Vbv`, but the per-picture banked-savings spend is additionally
    clamped at `max_drain_per_picture`, so a full bucket is emptied
    gradually rather than in one step. `max_drain_per_picture >=
    buffer_bytes` collapses to byte-identical plain `Vbv`;
    `max_drain_per_picture == 0` zeros the spend.
  Pure encoder-side residue rate policy — any §13.4.4 qindex is a legal,
  decodable choice, so every picked stream round-trips through
  `DiracDecoder`. The §11.3 residue analogue of the intra whole-picture
  byte-budget VBV variants from r146/r149/r159, applied to the residue
  payload. Closes the lib.rs "leaky-bucket VBV / VbvHysteresis residue
  carry for inter" gap. New `running_surplus_bytes` telemetry reports the
  bucket-clamped accumulator. `tests/encoder_inter_sequence_rate.rs`
  grows 2 tests (4 → 6): `vbv_clamps_savings_and_caps_request` (savings
  clamp + `buffer_bytes = 0` forfeiture, each request `<= target +
  buffer_bytes`, whole-stream decode) and
  `vbv_hysteresis_limits_drain_and_collapses_to_vbv` (per-picture drain
  cap + byte-identical collapse to plain `Vbv` when `max_drain >=
  buffer_bytes`). Sourced from `docs/video/dirac/dirac-spec-latest.pdf`
  §11.3 / §13.4.4 (any qindex legal); the bucket policy is a pure
  encoder-side shaping choice. No external library source, no web search.

- **Multi-picture rate-controlled inter sequence driver** (round-320) —
  `encoder_inter::encode_inter_sequence_with_residue_target(sequence,
  intra_params, inter_params, frames, target_residue_bytes, mode)` and
  its telemetry companion `encode_inter_sequence_with_residue_target_report`
  drive the per-picture `pick_inter_residue_qindex` picker across a
  complete elementary stream: an HQ intra **reference** anchor (`0xEC`)
  followed by N 1-ref inter pictures (`0x09`), each referencing the
  anchor (the stream's only reference picture, since `0x09` inter
  pictures are non-reference). New `InterRateControl` enum
  (`PerPicture` / `Cbr`) controls the **§11.3 residue-payload byte
  budget**: `PerPicture` sizes each picture's residue to the bare target;
  `Cbr` carries a signed `Σ(actual − target)` accumulator into the next
  picture's request (positive = overshoot debt → tighten; negative =
  savings → loosen). Per-picture telemetry (`InterPictureRate`) reports
  requested vs. actual residue bytes, the chosen qindex, and the running
  surplus. The inter analogue of the HQ/LD intra
  `encode_*_sequence_with_size_target` drivers — closes the lib.rs
  "multi-picture rate-controlled inter sequence driver" gap (the
  per-picture picker existed; this wires the sequence-level carry).
  Anchor-only input yields a valid one-frame stream (empty report);
  `residue = None` emits a ZERO_RESIDUAL chain (zero residue bytes,
  qindex floor). New `tests/encoder_inter_sequence_rate.rs` (4 tests)
  pins: the full stream decodes to one frame per input picture; every
  fitting qindex's actual residue stays within budget; the CBR
  accumulator equals the running `Σ(actual − target)` and each request
  folds the prior carry; PerPicture requests are the bare target; and
  the anchor-only / residue-disabled degeneracies.

## [0.0.8](https://github.com/OxideAV/oxideav-dirac/compare/v0.0.7...v0.0.8) - 2026-06-15

### Other

- §11.3 inter-residue rate-control qindex picker (round-309)
- HQ §12.4.5.2 slice_prefix_bytes builder + round-trip test (round-306)
- VC-2 v3 asymmetric transform bit-exact through fragmented path (round-299)
- Annex D Table D.9 corrected Fidelity quant matrices (round-293)
- §12.4.5.3 + Annex D asymmetric default quant matrices (round-290)
- §12.4.4 asymmetric transform end-to-end decode + encode (round-282)
- §12.4.5.3 asymmetric quant_matrix parsing
- typed-validation lift on §12.4.4 wavelet_index_ho (round-266)
- VC-2 §15.4.1 asymmetric IDWT driver — idwt_with_ho (round-256)
- VC-2 §15.4.2 h_synthesis horizontal-only IDWT step (round-249)
- drop release-plz.toml — use release-plz defaults across the workspace
- VC-2 v3 FragmentedPictureDecoder — picture-level §14 driver (round-248)
- VC-2 v3 fragment-assembler robustness oracle (round-238)
- VC-2 v3 §14.5 fragmented_wavelet_transform DC kick (round-233)
- VC-2 v3 §14.3 + §14.4 fragmented-picture state machine (round-229)
- VC-2 v3 §14.2 fragment-header parser + §10.5.2 Table 5 predicate (round-223)
- VC-2 HQ + LD encoder dwt_depth axis coverage (round-218)
- VC-2 v3 asymmetric extended_transform_parameters emission + decoder-rejection tests (round-212)
- VC-2 v3 extended_transform_parameters emission (round-206)
- VC-2 v3 extended_transform_parameters parser (round-201)
- row-major slice driving in vh_synth / vh_analysis + DD9/7 bench coverage (round-195)
- VC-2 inter-encoder fuzz oracle (round-193)
- criterion harness for decode / encode / roundtrip (round-190)
- tune HQ fuzz-oracle slice grid + frame amplitude for debug builds
- VC-2 HQ + LD rate-control fuzz oracle (round-179)
- VC-2 LD/HQ malformed-input fuzz oracle + 4 robustness fixes (round-165)
- VC-2 drain-rate-hysteresis (VbvHysteresis) rate-control variant (round-159)
- per-picture running_surplus_bytes rate-control telemetry (round-152)
- VC-2 LD leaky-bucket (VBV) rate-control variant (round-149)
- VC-2 HQ leaky-bucket (VBV) rate-control variant (round-146)
- VC-2 HQ multi-picture rate-controlled sequence driver (round-141)
- VC-2 HQ picture-level rate-control picker (round-138)
- VC-2 LD multi-picture rate-controlled sequence driver (round-134)
- VC-2 LD picture-level rate-control picker
- §12.3.6.6 inter DC-prediction unbiased-mean floor rounding (round-128)
- §13.2.1 inter quant-offset (round-125)
- fix §5.4 intra DC-prediction unbiased-mean rounding

### Added

- **§11.3 inter-residue rate-control qindex picker** (round-309) —
  `encoder_inter::pick_inter_residue_qindex(sequence, params, cur_y/u/v,
  ref_y/u/v, target_residue_bytes)` and its `(qindex, actual_bytes)`
  companion `inter_residue_qindex_diagnostic` — the inter §11.3-residue
  analogue of the HQ/LD intra picture-qindex pickers
  (`encoder::pick_hq_picture_qindex` / `pick_ld_picture_qindex`). The
  picker runs the same motion estimation `encode_inter_picture` commits
  to the bitstream (integer-pel SAD + round-73/80 adaptive int-pel snaps
  + §15.8.6 OBMC refinement, now factored into a shared `inter_mv_grid`
  helper so the measured residue matches the eventually-emitted one),
  reconstructs the §15.8 OBMC prediction once, forward-transforms the
  residue once, then walks `qindex ∈ rp.qindex.min(127)..=127` and
  returns the **smallest** qindex whose serialised §11.3 residue payload
  (transform_parameters + the three length-prefixed AC-coded component
  subband streams + the ZERO_RESIDUAL=false flag) is `<=
  target_residue_bytes`; an unsatisfiable budget returns 127. Monotone
  in the budget — a smaller budget can only raise the chosen qindex. The
  forward DWT is split out of `forward_and_quantise_residue` into
  `forward_residue_pyramid` (unquantised) + `quantise_residue_pyramid`
  so the walk re-quantises one transform. Pure encoder-side rate policy:
  any qindex is a legal §13.4.4 choice, so every picked stream decodes.
  Closes the lib.rs "tunable rate-controlled residue qindex" gap for the
  1-ref inter path. New `tests/encoder_inter_residue_rate.rs` (4 tests)
  pins: the diagnostic's actual residue bytes fit the budget when a fit
  exists; budget-monotonicity across an 8-point sweep; the floor /
  qindex-127 degeneracies; the configured `ResidueParams.qindex` floor
  is respected; and every chosen qindex yields a stream the production
  decoder accepts to two frames. Crate-wide tests: 454 → 458 (+4).
- **HQ §12.4.5.2 `slice_prefix_bytes` encoder support** (round-306) —
  `EncoderParams::with_slice_prefix_bytes(n)` sets the SMPTE ST
  2042-1:2022 §12.4.5.2 `slice_prefix_bytes` count, so every HQ slice is
  emitted with `n` leading application-specific bytes (zero-filled by
  this encoder) ahead of its qindex byte (§13.5.4 `read_uint_lit(state,
  slice_prefix_bytes)`). The decoder already skipped the prefix run
  (§13.5.4 NOTE: a conforming decoder may skip the prefix contents); the
  field was previously unreachable from the public encoder API, so the
  round-trip went untested. New
  `encode_then_decode_hq_slice_prefix_bytes_bit_exact_q0` integration
  test pins bit-exact reconstruction at qindex 0 for `slice_prefix_bytes
  ∈ {1, 3, 7}` and the per-slice stream-growth invariant (the prefix
  adds exactly `prefix * slices_x * slices_y` payload bytes; the
  byte-aligned header's `slice_prefix_bytes` exp-Golomb code may add at
  most one byte). Crate-wide tests: 453 → 454 (+1).
- **VC-2 v3 §12.4.4 asymmetric transform through the fragmented path —
  bit-exact verification** (round-299) — round 282 wired the
  asymmetric (horizontal-only) layout into `FragmentedPictureDecoder`
  (`init_pyramid_ho` / `idwt_with_ho`) but never drove a genuinely
  asymmetric (`dwt_depth_ho > 0`) picture through SMPTE ST 2042-1:2022
  §14. The new `hq_q0_asymmetric_transform_bit_exact_vs_non_fragmented`
  integration test (HQ, LeGall5/3 vertical × Haar0 horizontal-only,
  `dwt_depth = 2`, `dwt_depth_ho = 1`, `wavelet_index_ho !=
  wavelet_index`) encodes via `EncoderParams::with_asymmetric_transform`
  and asserts both fragment shapes (all slices in one data fragment +
  one data fragment per slice) reconstruct byte-identically to the
  non-fragmented `decode_picture` reference at qindex=0. The HQ payload
  dissector helper is made asymmetric-aware: `HqDissectParams` carries
  `(major_version, ext)` and `locate_hq_tp_end` mirrors the encoder's
  emission of the §12.4.4 `extended_transform_parameters()` flag bits
  and the §12.4.5.3 asymmetric custom-matrix shape so the byte cursor
  lands exactly on slice 0. +1 test.

- **Annex D Table D.9 corrected Fidelity quantisation matrices**
  (round-293) — `QuantMatrix::suggested_custom_fidelity(dwt_depth,
  dwt_depth_ho)` transcribes SMPTE ST 2042-1:2022 Table D.9, the
  *corrected* quantisation matrix for the Fidelity wavelet
  (`wavelet_index == 5`). Table D.6 (the frozen default) carries a NOTE
  that its values "do not correctly compensate for differential power
  gain"; D.9 supplies the corrected values that an encoder MAY emit as
  a custom matrix (`custom_quant_matrix = True`, §12.4.5.3) and the
  decoder reads straight back through `QuantMatrix::parse_custom`. Same
  Annex D depth bounds as the defaults (`dwt_depth ≤ 4`, `dwt_depth_ho
  ≤ 4`, `dwt_depth + dwt_depth_ho ≤ 5`); the table is depth-invariant
  across the `dwt_depth` columns for each `dwt_depth_ho` block. Wired
  end-to-end through the existing `EncoderParams::with_custom_quant_matrix`
  emission path. +6 tests (5 `quant` unit cells incl. a differs-from-D.6
  cross-check + a Fidelity HQ encode→decode that pins the in-band
  emission is byte-distinct from the D.6 default and parses without
  slice desync).

- **Annex D asymmetric default quantisation matrices** (round-290) —
  `QuantMatrix::default_for_asymmetric(wavelet, wavelet_ho, dwt_depth,
  dwt_depth_ho)` transcribes SMPTE ST 2042-1:2022 Tables D.1–D.8: the
  seven `wavelet_index_ho == wavelet_index` filter defaults plus the
  Table D.8 Haar0 (vertical) / LeGall (horizontal-only) cross-default,
  for `dwt_depth ≤ 4`, `dwt_depth_ho ≤ 4`, `dwt_depth + dwt_depth_ho ≤
  5`. The §12.4.5.3 `set_quant_matrix()` default branch now looks the
  matrix up rather than rejecting: a v3 asymmetric stream with
  `custom_quant_matrix = False` decodes end-to-end (bit-exact at
  qindex 0). `PictureError::AsymmetricTransformUnsupported` now fires
  only when the `(filter, ho-filter, depth, ho)` combination has no
  Annex D default and no custom matrix is supplied — the spec then
  mandates a custom matrix. +8 tests (6 `quant` unit cells +
  cross-checks, 2 picture/encoder-roundtrip end-to-end); crate-wide
  438 → 446.
- **§12.4.4 asymmetric (horizontal-only) transform — end-to-end decode
  + encode** (round-282) — streams with `dwt_depth_ho > 0` now decode
  through the full chain: `TransformParameters` carries the typed
  `wavelet_ho` / `dwt_depth_ho`, the §13.2.2 / §13.2.3 pyramid and
  subband dimensions span all `dwt_depth_ho + dwt_depth` levels
  (`subband::init_pyramid_ho` / `subband_dims_ho` /
  `padded_component_dims_ho` — width pads to `2^(ho+depth)`, height
  only to `2^depth`), the §13.5.3 / §13.5.4 slice unpack walks the new
  shared `subband::slice_band_order` sequence (L, then H ×
  `dwt_depth_ho`, then HL/LH/HH triplets), `quant::slice_quantisers`
  implements the §13.5.5 asymmetric else-branch (H quantiser emitted
  in pyramid slot 3, where the H band lives, bridging the matrix's
  §12.4.5.3 slot-0 storage), and the IDWT routes through the
  round-256 `wavelet::idwt_with_ho` (§15.4.1 driver + §15.4.2
  `h_synthesis` levels). LD's §13.5.2 DC prediction targets the
  level-0 L band — the same `[0][0]` pyramid slot as LL, so the call
  shape is unchanged. The fragmented path (§14) is wired identically:
  `dwt_depth_ho` flows into the assembler, the pyramids/dims use the
  asymmetric layout, `finish()` runs `idwt_with_ho`, and the §14.4
  trailing DC kick now succeeds on asymmetric LD pictures
  (`AssemblerError::AsymmetricDcPredictionUnsupported` is retained
  for API compatibility but no longer raised).
  `PictureError::AsymmetricTransformUnsupported` now fires **only**
  for an asymmetric stream with `custom_quant_matrix = False` (the
  Annex D asymmetric default-matrix tables are not transcribed yet);
  a v3 stream with `wavelet_index_ho != wavelet_index` but
  `dwt_depth_ho == 0` decodes with the symmetric default per the
  §12.4.4 NOTE, and `dwt_depth_ho + dwt_depth > 6` rejects as
  `UnsupportedDwtDepth`. Encoder side:
  `EncoderParams::with_asymmetric_transform(wavelet_ho, dwt_depth_ho)`
  (and the `LdEncoderParams` twin) selects v3, installs the override
  plus an all-zero custom quant matrix in the asymmetric shape;
  `forward_component` / `forward_component_ld` pad per §13.2.3 and run
  `wavelet::dwt_with_ho`; the slice packers and `write_quant_matrix`
  follow the asymmetric layout. New `dwt_depth_ho()` / `wavelet_ho()`
  accessors expose the effective values on both param structs.
  Roundtrip pins: HQ asymmetric self-roundtrips bit-exact at qindex 0
  for `(dwt_depth, dwt_depth_ho, wavelet_ho)` ∈ {(3, 1, LeGall 5/3),
  (2, 2, Haar0), (2, 1, DD 9/7)} — the latter two with
  `wavelet_index_ho != wavelet_index`; LD asymmetric ho=1 ≥ 35 dB
  Y/U/V at qindex 0; the inert `wavelet_index_ho`-only override
  (`dwt_depth_ho == 0`) decodes identically to the symmetric stream
  on both profiles. Crate-wide tests 431 → 438 (+7).

- **§12.4.5.3 asymmetric `quant_matrix` parsing** (round-274) —
  `QuantMatrix::parse_custom(r, dwt_depth, dwt_depth_ho)` implements
  the SMPTE ST 2042-1:2022 §12.4.5.3 `custom_quant_matrix == True`
  branch for both the symmetric (`dwt_depth_ho == 0`) and the
  asymmetric (`dwt_depth_ho > 0`) layouts. The asymmetric body reads
  a single L (DC) band at level 0, then `dwt_depth_ho` single
  horizontal-only H bands, then `dwt_depth` HL/LH/HH triplets — the
  §13.2.1 subband ordering (total levels `dwt_depth_ho + dwt_depth +
  1`). The L and H bands occupy the index-0 ("low") slot of their
  level, mirroring the existing LL convention, so the
  `Vec<[u32; 4]>` storage is unchanged; a new `dwt_depth_ho` field
  on [`quant::QuantMatrix`] records the split (0 for every existing
  symmetric matrix). `parse_transform_parameters` was restructured so
  the §12.4.4 `extended_transform_parameters` block is parsed and its
  `dwt_depth_ho` captured *before* the §12.4.5.2 slice_parameters and
  §12.4.5.3 quant_matrix, and the `AsymmetricTransformUnsupported`
  rejection (still surfaced for the unwired §13.5 slice / §15.4.1
  IDWT path) is deferred until the whole transform-parameters block
  has been consumed in its correct, `dwt_depth_ho`-dependent shape.
  This keeps the reader bit-aligned through a v3 asymmetric header
  rather than bailing mid-block. The default (`set_quant_matrix`)
  asymmetric path stays deferred — §12.4.5.3 mandates
  `custom_quant_matrix == True` for the asymmetric cases this crate
  could reach, and the Annex-D asymmetric default tables are not yet
  transcribed. Six new unit tests: four pin `parse_custom`
  (symmetric-equals-legacy, asymmetric L/H/triplet layout, exact-uint
  consumption with a trailing sentinel, zero-depth degenerate) and
  two drive `parse_transform_parameters` end-to-end on a synthesised
  v3 HQ header (asymmetric → value-checked `AsymmetricTransformUnsupported`
  after a full parse; symmetric-v3 → full `TransformParameters` with
  the custom matrix intact, guarding the refactor against regression).
  Library test count: 211 → 217 (+6).
- **§12.4.4 `wavelet_index_ho` typed-validation lift** (round-266)
  — `parse_extended_transform_parameters` now resolves the parsed
  `wavelet_index_ho` to a typed [`wavelet::WaveletFilter`] at parse
  time, exposed on `ExtendedTransformParameters::wavelet_ho`
  alongside the existing raw `wavelet_index_ho: u32` (which is kept
  so the asymmetric rejection diagnostic can still echo the
  bitstream value verbatim). When `asym_transform_index_flag` is
  set, an out-of-range index (`> 6`, since §12.4.4.2 reuses the
  `wavelet_index` value-space) is now rejected upfront as
  `PictureError::UnknownWaveletIndex`, mirroring the symmetric
  `wavelet_index` validation in `parse_transform_parameters` —
  before this change, a bogus ho index could either silently match
  the default and be accepted, or surface as
  `AsymmetricTransformUnsupported { wavelet_index_ho: <bogus>, .. }`
  which conflated "valid asymmetric filter we don't yet implement"
  with "value the §12.4.4.2 table doesn't define". The
  asymmetric-rejection downstream in `parse_transform_parameters`
  is otherwise unchanged: any `ExtendedTransformParameters` whose
  `wavelet_index_ho != wavelet_index` or `dwt_depth_ho != 0` still
  fails out with `AsymmetricTransformUnsupported`, but now the
  embedded filter index is guaranteed to name a real
  [`wavelet::WaveletFilter`]. Two new unit tests pin the new
  surface: one drives a 1-byte payload that decodes to
  `wavelet_index_ho = 8` (the bit pattern `1 0 0 0 0 0 1 1` =
  `0x83`) and asserts `UnknownWaveletIndex(8)`; the other drives a
  valid-but-different asymmetric index (`wavelet_index_ho = 3
  (Haar0)` over a `wavelet = LeGall5_3` default) and asserts the
  parser returns the typed filter unmodified, leaving the
  asymmetric-vs-symmetric decision to the caller. The four
  pre-existing `extended_transform_parameters_*` tests were
  re-pinned to include the new `wavelet_ho` field in their
  expected-value structs; their bit patterns are unchanged.
  Library test count: 209 → 211 (+2 = the two new validation
  tests; the five pre-existing
  `extended_transform_parameters_*` tests already shipped in
  round-201).
- **VC-2 §15.4.1 asymmetric IDWT driver** (round-256) —
  `wavelet::idwt_with_ho(pyramid, filter_v, filter_ho, dwt_depth_ho)`
  implements the SMPTE ST 2042-1:2022 §15.4.1 `idwt(state,
  coeff_data)` process for the `state[dwt_depth_ho] > 0`
  branch: seed the DC band from `coeff_data[0][L]`, invoke
  the round-249 `h_synth` (with `state[wavelet_index_ho]`)
  `dwt_depth_ho` times across `coeff_data[n][H]`, then chain
  the existing `vh_synth` (with `state[wavelet_index]`) over
  the remaining symmetric levels. With `dwt_depth_ho == 0`
  the function is byte-identical to the pre-v3 `idwt` (the
  §12.4.4 NOTE invariant), and a dedicated three-filter ×
  three-depth test pins that equivalence. A forward
  companion `wavelet::dwt_with_ho` peels off the symmetric
  levels first then the horizontal-only levels (the §15.4.1
  inverse order), producing a pyramid laid out exactly as
  `idwt_with_ho` consumes. Four new unit tests cover:
  symmetric-case equivalence with `idwt`, a pure
  horizontal-only round-trip on a 16×4 picture (no
  `vh_synth` levels — isolates the `h_synth` chain), a
  combined `dwt_depth_ho > 0` ∧ `dwt_depth > 0` round-trip
  across five `(filter_v, filter_ho, depth, ho-depth)`
  combinations that all exercise §12.4.4.2 `wavelet_index_ho
  != wavelet_index` AND §12.4.4.3 `dwt_depth_ho > 0`
  simultaneously, and an output-dimension algebra test
  (`width = lw << (ho + d)`, `height = lh << d`). The
  picture decoder still surfaces
  `PictureError::AsymmetricTransformUnsupported` for
  asymmetric streams — wiring `idwt_with_ho` into
  `picture.rs` is a separate round-scope step because the
  §13 slice-unpacking layout for the L / H bands (§13.5.3
  `ld_slice` / §13.5.4 `hq_slice` asymmetric branches) is
  not yet implemented. Crate-wide tests 419 → 423 (+4).
- **VC-2 §15.4.2 horizontal-only synthesis** (round-249) —
  `wavelet::h_synth(l, h, filter)` implements the SMPTE ST
  2042-1:2022 §15.4.2 `h_synthesis(state, L_data, H_data)`
  building block: horizontal interleave of equal-shape `L`
  and `H` subbands into a `2*width × height` array, one
  1-D synthesis per row using `state[wavelet_index_ho]`,
  then the §15.4.2 step-4 accuracy-bit rounding right-shift.
  Companion `wavelet::h_analysis` is the exact inverse used
  by round-trip tests and any future encoder path. Four new
  unit tests pin the step's invariants: a LeGall (5,3) pure-DC
  `L` band reconstructs uniform output (the asymmetric mirror
  of the existing `vh_synth` DC tests); a Haar0 row-independent
  pattern flows through without vertical leak, with each
  even/odd output sample matched against the spec's two
  lifting stages by hand; an `h_analysis` → `h_synth`
  integer round-trip across **all seven** spec filters
  (Tables 16–22 — DD9/7, LeGall 5/3, DD13/7, Haar0, Haar1,
  Fidelity, Daubechies 9/7) on a non-square 12×5 picture;
  and an explicit width-doubling / height-preserving
  dimension contract test. This is the §15 building block
  needed to lift the §12.4.4 / §14.5 asymmetric-transform
  rejection paths off `dwt_depth_ho > 0` streams once the
  rest of the §15.4.1 IDWT driver (multi-stage h+vh
  composition) and §14.5 horizontal-only `dc_prediction`
  land. Crate-wide tests 415 → 419 (+4).
- **VC-2 v3 fragmented-picture decoder** (round-248) —
  `FragmentedPictureDecoder<'s>` in `src/fragment.rs`, the
  picture-level driver that ties the §14 `FragmentAssembler`
  state machine to the §13.5 LD / HQ slice coefficient
  decoders. `new(&SequenceHeader)` builds an empty driver;
  `on_setup_fragment(&ParseInfo, payload)` parses the §14.2
  setup header, runs the §12.4 transform-parameters block
  (LD profile derived from `0xCC`, HQ from `0xEC`), allocates
  the three component pyramids and pre-computes per-level
  subband dims; `on_data_fragment(&ParseInfo, payload)`
  parses the §14.2 12-byte data header, walks the §14.4
  raster `(slice_x, slice_y)` coordinates and dispatches
  each slice through the same `picture::decode_ld_slice` /
  `picture::decode_hq_slice` primitives the non-fragmented
  `decode_picture` path uses (now `pub(crate)` for the
  driver); `finish()` runs the §14.5 trailing DC kick
  (LD only, HQ skipped per spec), the §13.3 IDWT, and the
  §13.6 trim / clip / output offset, returning a
  `DecodedPicture` bit-exact-equivalent to running
  `decode_picture` on a non-fragmented version of the same
  picture. New `FragmentedPictureError` wraps the three
  error sources distinctly: `Header(FragmentError)`,
  `Assembler(AssemblerError)`, `Picture(PictureError)`;
  plus `UnsupportedParseCode(u8)` (only `0xCC` / `0xEC`
  accepted), `NoActivePicture` (data fragment or `finish`
  before any setup), and `PictureIncomplete {
  slices_received, slices_expected }` (`finish` before
  §14.4 picture-done). The driver is reusable across
  consecutive pictures. `tests/fragmented_picture_decoder.rs`
  (9 new integration tests) pins HQ q=0 bit-exact vs the
  non-fragmented reference for both the
  all-slices-in-one-data-fragment shape and the
  one-data-fragment-per-slice shape; the matching LD q=0
  pair (which exercises the §14.5 DC kick); `finish` rejection
  of an incomplete picture with the exact counters;
  data-fragment-before-setup rejection; non-fragment parse-code
  rejection; consecutive-pictures reuse on one driver.
  Crate-wide tests grow 406 → 415 (+9).
- **VC-2 v3 fragment-assembler robustness oracle**
  (round-238) — `tests/fragment_assembler_fuzz_oracle.rs` (9
  tests, +9 crate-wide tests 397 → 406). Drives a deterministic
  xorshift-seeded random walk of `(Setup, Transform, Data,
  DcKick)` operations across 5 000 steps × 8 seeds into a
  `FragmentAssembler` running in parallel with a reference state
  model that knows the SMPTE ST 2042-1:2022 §14.1 / §14.3 / §14.4
  / §14.5 sequencing rules; asserts the assembler and model agree
  on accept / reject for every transition, that on accept the
  emitted `coords` length matches `slice_count`, that each
  `(slice_x, slice_y)` matches the §14.4 pseudocode `raster =
  y_offset * slices_x + x_offset + s; (raster % slices_x, raster
  / slices_x)` computed in `u64`, and that
  `slices_received` / `fragmented_picture_done()` stay in lockstep
  with the model. Pathological-geometry sweep (`(slices_x,
  slices_y)` at zero, `u32::MAX`, and the overflowing
  `(0x1_0000, 0x1_0000)` product) pins the assembler's
  `saturating_mul` invariant for `expected_total` so
  `fragmented_picture_done()` stays deterministic and
  `SliceOverflow` fires correctly. Parse-code-mixing sweep
  exercises every cross-class pair (HQ setup → LD data and the
  mirror, all four §10.5.2 Table 5 fragment codes
  `0xC8`/`0xCC`/`0xE8`/`0xEC`) and confirms the
  `using_dc_prediction(parse_code) := (parse_code & 0x28) ==
  0x08` predicate is captured per class. §14.5 DC-kick edge
  cases pin: rejection before picture completion
  (`DcPredictionBeforePictureComplete`); HQ no-op even when
  asymmetric; LD-path success on synthetic LL subbands of varied
  shapes (1×1, 1×N, N×1, square, 8×8) including a seeded
  non-zero pattern that would surface arithmetic-overflow bugs
  in the §13.4 neighbour-mean prediction; LD-path rejection on
  `dwt_depth_ho ∈ {1, 2, 3, 4}` with
  `AsymmetricDcPredictionUnsupported`. `slice_coords` helper
  cross-checked against the §14.4 pseudocode for a Cartesian
  sweep including `slices_x = u32::MAX`, `x_offset = u16::MAX`,
  `y_offset = u16::MAX`, `s = u32::MAX` — pins the `u64`-widened
  arithmetic prevents `u32` overflow on pathological raster
  indices. Determinism test pins the random walk to its seed so
  any regression is reproducible by re-running the same seed.
  Companion to r165's decoder fuzz oracle, r179's encoder
  rate-control fuzz oracle, and r193's inter-encoder fuzz oracle
  — the four oracles cover the four largest stateful surfaces in
  the crate.
- **VC-2 v3 §14.5 fragmented_wavelet_transform trailing
  `dc_prediction(...)` kick** (round-233, SMPTE ST 2042-1:2022
  §14.5). Closes the picture-completion gap that r229's `FragmentAssembler`
  left open: the assembler now exposes a single
  `FragmentAssembler::fragmented_wavelet_transform_dc_prediction(components:
  &mut [&mut SubbandData])` entry point that runs the §13.4 raster
  neighbour-mean DC prediction in-place on each component's
  level-0 LL subband once `fragmented_picture_done()` returns
  true. The LD path (`0xC8` / `0xCC`, captured at setup time via
  the §10.5.2 Table 5 `using_dc_prediction` predicate) executes
  the kick; the HQ path (`0xE8` / `0xEC`) returns `Ok(())`
  without touching the subbands — matching the §14.4 / §14.5
  rule that the trailing kick is gated by
  `state[using_dc_prediction]`. Three new `AssemblerError`
  variants pin the precondition surface:
  - `DcPredictionBeforePictureComplete` — invocation before
    `state[fragmented_picture_done] = True` per §14.4 is rejected
    (matches the §14.5 placement after `fragmented_picture_done`
    transitions to true).
  - `AsymmetricDcPredictionUnsupported { dwt_depth_ho }` —
    `dwt_depth_ho > 0` selects the level-`dwt_depth_ho` L
    (low-pass-only) subband per §12.4.4.3 instead of the level-0
    LL subband; the L-subband path is not implemented yet, so the
    assembler returns the v3-asymmetric gap with the offending
    depth surfaced for diagnostics. Mirrors the non-fragmented v3
    decoder's `PictureError::AsymmetricTransformUnsupported`
    rejection so the v3 asymmetric surface is consistent across
    fragmented and non-fragmented paths.
- **6 new fragment-module unit tests** covering: the LD-path
  happy path (`0xCC` 2x2 picture seeded with `1`s walks linearly
  to `1,2,3` on the first row per §13.4); HQ-path no-op
  (`0xEC`-completed picture passes through unchanged); incomplete
  picture rejection; asymmetric `dwt_depth_ho > 0` rejection;
  single-component (luma-only) call; empty-components slice
  accepted on both LD and HQ paths. Crate-wide test count:
  391 → 397 (+6).
- **VC-2 v3 fragmented-picture state machine** (round-229, SMPTE ST
  2042-1:2022 §14.3 / §14.4). Builds on r223's fragment-header parser
  by adding the *reassembly* layer:
  - `FragmentAssembler` — a per-picture state machine that ingests
    parsed fragment headers in sequence. The driver calls
    `on_setup_fragment(&FragmentHeader, parse_code: u8)` for a setup
    fragment, then `on_transform_parameters(slices_x, slices_y,
    dwt_depth_ho)` once it has parsed the §12.4 `transform_parameters`
    that follow the setup-fragment header, then
    `on_data_fragment(&FragmentHeader, parse_code: u8)` for each
    data fragment in turn. Each call returns a `FragmentEvent`
    enum: `SetupAccepted` for setup fragments, `DataSlices { coords,
    picture_done }` for data fragments. `coords` is the §14.4 list
    of raster `(slice_x, slice_y)` coordinates for the slices
    carried by that fragment; `picture_done` flips to `true` on the
    fragment that completes the picture (matches §14.4
    `state[fragmented_picture_done] = True`).
  - The §14.4 raster-scan formula is exposed as a pure free
    function `slice_coords(s, x_offset, y_offset, slices_x) ->
    Option<(u32, u32)>` (`raster = y_offset * slices_x + x_offset +
    s`; coordinates are `(raster % slices_x, raster / slices_x)`;
    `None` on `slices_x == 0`). Computed in `u64` so a `u16::MAX`
    `y_offset` on a wide picture grid does not overflow.
  - §14.1 sequencing constraints enforced as `AssemblerError`
    variants: `UnexpectedDataFragment` (data fragment before setup
    or before transform parameters arrive), `PictureNumberMismatch
    { setup, data }` (data fragment's §14.2 picture_number doesn't
    match the setup's), `InconsistentParseCode { setup, data }` (LD
    setup followed by an HQ data fragment or vice versa; the §14.4
    `using_dc_prediction` predicate is captured from the setup's
    parse code and must hold for every fragment in the picture),
    `SliceOverflow { expected_total, slices_received, slice_count }`
    (a data fragment would push the cumulative slice count past
    `slices_x * slices_y` — §14.4 explicitly forbids omitted /
    repeated slices), `SetupBeforePreviousPictureComplete` (§14.1
    forbids consecutive setup fragments while a picture is still
    incomplete), `InvalidSliceGrid { slices_x, slices_y }`
    (transform parameters with a zero slice dimension).
  - Observability accessors on the assembler:
    `slices_x()`, `slices_y()`, `dwt_depth_ho()`,
    `using_dc_prediction()`, `picture_number()`,
    `slices_received()`, `fragmented_picture_done()`. The last
    matches the §14.4 `state[fragmented_picture_done]` flag that
    keys the trailing `dc_prediction(...)` kick on the LL (or L,
    when `dwt_depth_ho > 0`) subbands.
- **VC-2 v3 strict §10.5.2 Table 5 predicates on `ParseInfo`**:
  - `is_ld_v3()`: `(parse_code & 0xF8) == 0xC8`. Matches `0xC8`
    (LD picture) and `0xCC` (LD picture fragment).
  - `is_hq_v3()`: `(parse_code & 0xF8) == 0xE8`. Matches `0xE8`
    (HQ picture) and `0xEC` (HQ picture fragment).
  - `is_picture_v3()`: `(parse_code & 0x8C) == 0x88`. Matches only
    the *non-fragment* picture codes `0xC8` / `0xE8`; the v3
    dispatcher routes fragments via the pre-existing
    `is_fragment_parse_code()` predicate so the two predicates
    partition the picture-or-fragment space cleanly.
  - `using_dc_prediction()`: `(parse_code & 0x28) == 0x08`. True
    for the LD path (`0xC8` / `0xCC`); false for the HQ path
    (`0xE8` / `0xEC`). This is the §14.4 predicate that gates the
    trailing per-component `dc_prediction(...)` kick after a
    fragmented picture completes.
  The pre-existing BBC Dirac v2.2.3 predicates `is_picture` /
  `is_low_delay` use slightly different bit masks (broader, because
  Dirac's parse-code table assigns several codes that VC-2 has
  reserved). Both sets are kept side-by-side so a v3 dispatcher
  and a Dirac-spec dispatcher can each query the appropriate one
  for the active stream's `major_version`.
- **10 new fragment-module unit tests** covering: the v3 strict
  predicates (`is_ld_v3` / `is_hq_v3` / `is_picture_v3` /
  `using_dc_prediction`), the §14.4 `slice_coords` happy path,
  mid-row-straddling, the `slices_x == 0` rejection, the
  `u16::MAX` `y_offset` non-overflow, the `FragmentAssembler`
  setup-fragment acceptance (LD + HQ), the
  `UnexpectedDataFragment` rejection (no setup), the
  `UnexpectedDataFragment` rejection (no transform parameters),
  the `InvalidSliceGrid` rejection on zero slice dimensions, the
  single-data-fragment-completes-picture happy path, the
  multi-data-fragment progressive completion (§14.4 cumulative
  `slices_received`), the `PictureNumberMismatch` rejection, the
  `InconsistentParseCode` rejection (LD setup + HQ data), the
  `SliceOverflow` rejection, the
  `SetupBeforePreviousPictureComplete` rejection + recovery once
  the in-flight picture completes, and the `dwt_depth_ho`
  preservation across data-fragment ingestion.
- **2 new `tests/fragment_parser.rs` integration tests** that
  drive the assembler through the real `DataUnitIter` byte
  walker: one walks a synthetic
  `[seq_hdr][0xCC setup][0xCC data 2 slices @ (0, 0)][0xCC data 2
  slices @ (2, 0)][EOS]` stream (`slices_x = 4`, `slices_y = 1`)
  and pins the per-fragment `(slice_x, slice_y)` raster output +
  `picture_done` firing on the second data fragment + the §14.4
  `using_dc_prediction` flag holding for the trailing DC-pred
  kick; the second walks
  `[seq_hdr][0xCC setup pic=0][0xCC setup pic=1][EOS]` and pins
  the §14.1 `SetupBeforePreviousPictureComplete` rejection on the
  second setup.
- Crate-wide test count: 369 → 391 (+22).
- All material consulted: `docs/video/vc2/vc2-specification.pdf`
  (§10.5.2 Table 4 / Table 5, §14.1 / §14.2 / §14.3 / §14.4,
  §12.4.4.3 `dwt_depth_ho`, §13.5.6 slice grid; Annex A.3.4
  `read_uint_lit`). No external library source, no web search.

- **VC-2 v3 fragment-header parser** (round-223, SMPTE ST 2042-1:2022
  §14.2). New `src/fragment.rs` carries `FragmentHeader::parse` for
  the byte-aligned fragment header that immediately follows a v3
  fragment parse-info unit. Layout:
  - 4-byte `picture_number` (big-endian; must match across a setup
    fragment and its associated data fragments per §14.2),
  - 2-byte `fragment_data_length` (the spec marks this "undefined
    data for the purposes of this standard" — preserved for stream
    layout tooling),
  - 2-byte `fragment_slice_count` (0 = setup fragment carrying
    transform parameters, non-zero = data fragment carrying that
    many consecutive slices in raster order),
  - 2-byte `fragment_x_offset` + 2-byte `fragment_y_offset` (data
    fragments only — first slice's raster coordinates).
  A setup fragment is therefore 8 bytes, a data fragment 12 bytes,
  matching Annex A.3.4's definition of `read_uint_lit(state, n)` as
  `read_nbits(state, 8 * n)`.
- **`ParseInfo::is_fragment_parse_code()`** — VC-2 v3 §10.5.2 Table 5
  bit predicate `(parse_code & 0x0C) == 0x0C`. Matches the v3
  picture-fragment codes `0xCC` (LD) and `0xEC` (HQ). Doc-noted
  caveat: the same bit pattern is reused by the BBC Dirac v2.2.3
  Table 9.1 for core-syntax reference picture codes (`0x0C`, `0x0D`,
  `0x0E`, `0x4C`) and for the v2 LD intra reference `0xCC`, so the
  predicate is *necessary but not sufficient* for fragment
  recognition — the dispatcher must combine it with a
  `major_version == 3` check from the sequence header. The predicate
  is the syntactic foundation for fragment reassembly in a follow-up
  round (the `fragment_data(state)` slice-routing logic of §14.4
  and the `if (fragmented_picture_done) dc_prediction(...)` kick at
  the end).
- **10 fragment-module unit tests** (setup + data parsing,
  `Truncated { needed, available }` differentiation between the
  8-byte and 12-byte header widths, `u32::MAX` picture-number
  round-trip, the §14.2 "undefined data" tolerance on
  `fragment_data_length`, `header_size()` agreement with the
  variant, the §10.5.2 Table 5 bit predicate firing on `0xCC` /
  `0xEC` and *not* firing on `0x00` / `0x10` / `0x20` / `0x30` /
  `0xC8` / `0xE8`, and a separate test pinning the spec ambiguity
  where the same predicate also fires on the BBC Dirac reference
  picture codes).
- **3 integration tests in `tests/fragment_parser.rs`** that build a
  synthetic VC-2 stream `[seq_hdr][0xEC setup][EOS]` /
  `[seq_hdr][0xCC data][EOS]` / `[seq_hdr][0xCC setup][0xCC data]
  [EOS]`, walk it with `DataUnitIter`, and confirm the fragment
  unit's `payload` (the bytes strictly between two consecutive
  parse-info headers) decodes cleanly via
  `FragmentHeader::parse`. Pins that the parser's input contract
  matches the stream walker's output.
- Crate-wide test count: 356 → 369 (+13).
- All material consulted: `docs/video/vc2/vc2-specification.pdf`
  (§10.5.2, §14.2, Annex A.3.4) and
  `docs/video/dirac/dirac-spec-latest.pdf` (Table 9.1 for the
  ambiguity note). No external library source, no web search.

- **VC-2 HQ + LD encoder/decoder roundtrip `dwt_depth` axis coverage**
  (round-218). The project's own
  `docs/video/dirac/dirac-fixtures-and-traces.md` "Gaps" section calls
  out **wavelet depths other than 3 and 4** (the spec-allowed range
  is `1..=5` per §11.3) as a feature surface that no upstream-slice
  fixture exercises. Pre-r218, the self-roundtrip test surface lined
  up with that gap: `tests/encoder_*` drove `dwt_depth = 3` almost
  exclusively, with one HQ `depth = 2` spot-check in
  `tests/encoder_matrix.rs::hq_q0_lossless_at_dwt_depth_two` and the
  asymmetric `dwt_depth_ho` v3 negative paths in
  `tests/encoder_roundtrip.rs`. New `tests/encoder_dwt_depth_coverage.rs`
  (7 tests) self-roundtrips the HQ and LD encoders against the
  decoder at the previously-uncovered depths:
  - HQ at depth 1, 4, 5 bit-exact reconstruction at qindex=0
    (`hq_q0_bit_exact_at_dwt_depth_{one,four,five_custom_matrix}`),
    plus LD at depth 1 and 4 above the 35 dB roundtrip threshold
    (`ld_q0_psnr_over_35_at_dwt_depth_{one,four}`).
  - Depth 5 takes the §11.3.5 *custom* quantisation matrix path
    (the Annex E.1 default tables only cover depth `<= 4`); the test
    constructs an all-zero matrix sized `dwt_depth + 1` levels and
    sets `custom_quant_matrix = true` so it travels in-band via
    §12.4.5.3. At qindex=0 every per-subband effective quantiser
    collapses to 0 (dead-zone identity), so the reconstruction is
    still bit-exact.
  - Depth 4 and 5 loosen the slice grid to `(2, 2)` (slice luma dim
    32 = `2^5`) so the bottom of the pyramid still owns at least one
    LL sample per slice.
  - Two helper-shape tests (`slice_grid_helper_shape`,
    `zero_custom_quant_matrix_shape_is_correct`) pin the test-local
    slice-grid picker and the all-zero matrix builder so a future
    tightening keeps the depth-coverage tests aimed at the right
    geometry. Crate-wide test count: 349 → 356 (+7).
  Sourced exclusively from `docs/video/dirac/dirac-spec-latest.pdf`
  + `docs/video/dirac/dirac-fixtures-and-traces.md` (the documented
  Gaps list). No external implementation source consulted.
- **VC-2 v3 asymmetric `extended_transform_parameters()` encoder
  emission + decoder-rejection coverage** (round-212). Extends the
  round-206 symmetric-default emitter with an `extended_transform_override:
  Option<ExtendedTransformOverride>` field on both `EncoderParams` and
  `LdEncoderParams` (with matching `with_extended_transform_override()`
  builders). When set and `major_version >= 3`, the encoder writes
  `asym_transform_index_flag = (wavelet_index_ho != wavelet_index)`
  followed by the gated `wavelet_index_ho` as an interleaved exp-Golomb
  code per SMPTE ST 2042-1:2022 §12.4.4.2, then
  `asym_transform_flag = (dwt_depth_ho != 0)` and the gated
  `dwt_depth_ho` per §12.4.4.3. The override is **negative-testing-only**:
  the §13.5.5 horizontal-only IDWT is not yet implemented, so any
  non-default emission is rejected by our own decoder with
  `PictureError::AsymmetricTransformUnsupported`. Bitstream layout is
  spec-conformant, so a future asymmetric-IDWT decoder would consume
  the same bytes unchanged. Also makes `encoder::wavelet_index` public
  so the override callers can write `wavelet_index_ho` in terms of a
  `WaveletFilter` value rather than a raw integer constant. Four new
  integration tests in `tests/encoder_roundtrip.rs` cover the HQ × LD
  axis crossed with the two override axes:
  `encode_hq_v3_asym_wavelet_index_ho_rejected_by_decoder`,
  `encode_hq_v3_asym_dwt_depth_ho_rejected_by_decoder`,
  `encode_ld_v3_asym_wavelet_index_ho_rejected_by_decoder`,
  `encode_ld_v3_asym_dwt_depth_ho_rejected_by_decoder` — each asserts
  the override changes the bitstream relative to the symmetric-default
  v3 emission AND the decoder surfaces the
  `"v3 asymmetric transform unsupported"` substring through
  `core::Error::invalid`.
- **VC-2 v3 `extended_transform_parameters()` encoder emission** (round-206).
  Encoder-side companion to round-201's decoder-side parser. Both
  `EncoderParams` (HQ) and `LdEncoderParams` (LD) gain a new
  `major_version: u32` field (default `2`) plus a `with_major_version_3()`
  builder. When the field is `>= 3`, `write_transform_parameters()` /
  `write_ld_transform_parameters()` emit the two
  `extended_transform_parameters()` `read_bool()` flag bits
  (`asym_transform_index_flag` then `asym_transform_flag`) at their
  `False` symmetric default right after `dwt_depth`, per SMPTE ST
  2042-1:2022 §12.4.4. The §12.4.4 NOTE — "If `state[dwt_depth_ho]` is
  0 and `state[wavelet_index_ho]` is `state[wavelet_index]` then the
  inverse wavelet transform process (see 13.2) is identical to that
  defined in earlier versions of this specification" — guarantees the
  v3 path's reconstruction is pixel-identical to v2. Callers MUST also
  set `sequence.parse_parameters.version_major = 3` so the decoder
  dispatches into the v3 parsing branch. Asymmetric (non-default)
  emission is intentionally not exposed — the decoder rejects those
  streams with `PictureError::AsymmetricTransformUnsupported`.
  - Two new integration tests in `tests/encoder_roundtrip.rs`:
    - `encode_then_decode_hq_v3_symmetric_default_lossless_q0` — HQ
      qindex=0, 64x64 4:2:0 testsrc. Asserts the v3 stream is **not**
      byte-identical to the v2 stream (the two flag bits + bumped
      `version_major` exp-Golomb code must change the layout) AND that
      both streams decode to (a) bit-exact reproduction of the source
      pixels and (b) pixel-identical reconstructions to each other.
    - `encode_then_decode_ld_v3_symmetric_default_qindex0_psnr_over_35`
      — LD qindex=0, 4x4 slices × 128 bytes/slice, smooth gradient.
      Same v3/v2 stream-divergence + reconstruction-identity assertion
      plus the existing LD-side ≥ 35 dB Y/U/V PSNR threshold.
  - Crate-wide test count: 338 → 345 (+7) — five picture-module unit
    tests from round-201 (counted in the previous round's report) plus
    the two new integration tests added this round.

### Added

- **VC-2 v3 `extended_transform_parameters()` parser** (round-201).
  `picture::parse_transform_parameters` previously aborted with
  `PictureError::ExtendedTransformParams` as soon as it saw a
  `major_version >= 3` sequence header, regardless of whether the v3
  stream actually used any v3-only feature. The new
  `parse_extended_transform_parameters` helper implements the SMPTE ST
  2042-1:2022 §12.4.4 pseudocode bit-for-bit: two boolean flags
  (`asym_transform_index_flag`, `asym_transform_flag`), each gating an
  interleaved-exp-Golomb field (`wavelet_index_ho`, `dwt_depth_ho`)
  with defaults inherited from the enclosing `transform_parameters`
  (§12.4.4.2: defaults to `wavelet_index`; §12.4.4.3: defaults to 0).
  v3 streams that reduce to the symmetric default — per the §12.4.4
  NOTE, "If `state[dwt_depth_ho]` is 0 and `state[wavelet_index_ho]`
  is `state[wavelet_index]` then the inverse wavelet transform
  process (see 13.2) is identical to that defined in earlier versions
  of this specification" — now decode cleanly. Genuinely asymmetric
  v3 streams are rejected with the new, more specific
  `PictureError::AsymmetricTransformUnsupported { wavelet_index_ho,
  dwt_depth_ho }` carrying the parsed values for diagnostics. The
  `dwt_depth_ho > 6` ceiling is also enforced (same IDWT-pyramid
  bound as `dwt_depth`). 5 new unit tests pin each bit-level pattern
  (both-flags-off / index-only / depth-only / both-on / over-cap).
  Crate-wide test count: 160 → 165 (+5) in `picture::tests`.

- **DD9/7 wavelet bench coverage** (round-195). All three Criterion bench
  harnesses (`decode`, `encode`, `roundtrip`) grow a fourth scenario,
  `hq_intra_64x64/qindex=0/wavelet=dd9_7`, covering the
  Deslauriers-Dubuc 9/7 wavelet (`wavelet_index = 0`) — Dirac's *default*
  filter, which the original r190 rows omitted in favour of LeGall 5/3
  only. DD9/7's second lifting step is 4-tap vs. LeGall's 2-tap, so this
  row's IDWT / forward-DWT cost is the dominant per-frame work and is the
  right A/B fixture for future profile-driven wavelet tweaks.
  - Bench harness: 9 → 12 timed scenarios (4 per binary).
  - Measured (M2, --measurement-time 3): decode dd9_7 row at
    80.7 µs / frame (50.7 Melem/s) vs. 73.2 µs / 56.0 Melem/s for the
    LeGall companion at the same `qindex=0` setting (~10% heavier
    inverse-DWT kernel as expected); encode dd9_7 row at 85.0 µs vs.
    78.3 µs LeGall (~9% heavier); roundtrip dd9_7 at 166.0 µs vs.
    148.0 µs LeGall (~12% heavier joint cost).

### Changed

- **IDWT / forward-DWT row-major slice driving** (round-195).
  `wavelet::vh_synth` and `wavelet::vh_analysis` are rewritten to drive
  the row-major backing `Vec<i32>` directly instead of going through
  `SubbandData::{get, set}`:
  - Interleave / de-interleave loops pre-slice the source two rows + four
    destination rows (or the four input subband rows + two destination
    rows for `vh_synth`) once per output row-pair, eliminating the four
    bounds-checked `set` / `get` calls per output sample.
  - Vertical lifting pass keeps the `col_buf` scratch buffer reuse but
    gathers / scatters via raw indexing into the underlying `data` slice
    so the compiler's bounds-check elision applies.
  - Horizontal lifting pass and the step-4 accuracy-bit shift fold were
    already optimal (`row_mut` slice + single `data.iter_mut()`).
  - Bit-exactness preserved: all 14 wavelet unit tests pass, including
    the 7-filter × depth-{1,2,3} `dwt_idwt_roundtrip_all_filters_all_depths`
    invariant and the all-filter ffmpeg cross-decode interop tests
    (Dirac-default DD9/7, LeGall 5/3, DD13/7, Haar0/1, Fidelity,
    Daubechies 9/7). Crate-wide test count unchanged at 338, all green.
  - Measured (M2, --measurement-time 3, criterion 100-sample baseline
    re-set per round): decode q=32 row -1.15% (p < 0.05 — statistically
    significant); decode q=0, decode ld q=16, encode/roundtrip rows
    within noise. The optimisation is small in absolute terms because
    at 64x64 with `dwt_depth = 3` the IDWT is co-dominant with the
    entropy-coder path; future rounds running the new DD9/7 row (where
    the IDWT *is* the dominant cost) will surface bigger deltas from
    follow-on wavelet-loop tuning.

### Added

- **Inter-encoder fuzz oracle** (round-193). New
  `tests/encoder_inter_fuzz_oracle.rs` (9 tests) — the inter-path
  analogue of r179's intra-side `encoder_rate_control_fuzz_oracle.rs`.
  Sweeps the full `InterEncoderParams` + `ResidueParams` surface against
  pathological combinations and pathological input pixel surfaces,
  pinning that every accepted (`InterEncoderParams`,
  `InterInputPicture`, `InterInputPicture`) combination produces a
  non-empty bytestream that round-trips through the registry-backed
  decoder to exactly two video frames, with no panic / no debug-assert /
  no integer overflow / no livelock.
  - **Precision / OBMC / search-range sweep.** Diagonal walk
    `mv_precision == bipred_mv_precision ∈ 0..=3` (integer / half-pel /
    quarter-pel / eighth-pel) × `obmc_refine_passes ∈ {0, 2}` ×
    `mv_search_range ∈ {2, 16}` against both the translating-square and
    camera-pan synthetic pairs, plus two off-diagonal precision pairs
    to pin that the 1-ref and 2-ref precisions are independent.
  - **Residue wavelet / depth / qindex sweep.** All seven
    `WaveletFilter` variants × dwt_depth `{1, 2, 3, 4}` at a
    representative mid-quantiser (qindex=32), plus a qindex axis walk
    `{0, 8, 32, 64, 127}` at the default wavelet × depth, plus the
    `residue = None` legacy ZERO_RESIDUAL=true path. Linear-plus-linear
    walk so axis coverage doesn't blow up combinatorially.
  - **Adaptive-flag boolean sweep.** All 8 combinations of
    `inter_adaptive_int_pel` × `inter_adaptive_int_pel_post_obmc` ×
    `bipred_post_obmc_refine` against the camera-pan fixture.
  - **Pathological pixel inputs.** Zero-luma / 0xFF-luma /
    mid-grey / single-pixel-pulse intra-and-inter pairs — stresses the
    "no-energy" SAD landscape and "saturated-energy" extreme without
    livelocking the sub-pel refinement or overflowing the residue
    coefficient block.
  - **Same-frame zero-motion degenerate input** + **deterministic
    output** under default + residue-off paths + **`mv_search_range=0`
    extreme** + **`residue qindex=127` extreme**.
  - Crate-wide test count grows from 329 → 338 (+9). All tests pass
    debug-build under `--test-threads=2` in ~17 s end-to-end (the
    Cartesian sweep is the long pole; deliberately trimmed to the
    diagonal of the precision matrix so debug-build CI stays well
    under a minute). Encoder-side analogue of r165's decoder fuzz
    oracle + r179's intra rate-control fuzz oracle, completing the
    encoder-side fuzz coverage of both intra (r179) and inter (r193)
    code paths.

- **Criterion benchmark suite for Dirac / VC-2 hot paths** (round-190).
  New `benches/{decode,encode,roundtrip}.rs` (3 binaries, 9 timed
  scenarios) exercise the production
  `encode_single_hq_intra_stream` / `encode_single_ld_intra_stream` +
  registry-backed decoder path on a deterministic xorshift-synthesised
  64x64 4:2:0 YUV input. No committed fixture files; no third-party
  crate / binary in the timed region. Pairs with r165's decoder fuzz
  oracle + r179's encoder rate-control fuzz oracle to give the next
  encoder / decoder rounds a numerical A/B baseline for algorithm
  tweaks (intra DC prediction, codeblock quant-offset walk, slice-
  bytes derivation, rate-control picker, IDWT inner loop).
  - **decode**: 3 scenarios (HQ q=0, HQ q=32, LD q=16). Each iteration
    builds a fresh `CodecRegistry` + `first_decoder`, pushes one packet,
    pulls one frame. Throughput reported as Y pixels/s. Indicative
    numbers on the dev machine (release, `--quick` measurement):
    HQ q=0 ≈ 74 µs (55 Melem/s), HQ q=32 ≈ 56 µs (74 Melem/s), LD q=16
    ≈ 51 µs (80 Melem/s) — confirms the spec-expected ordering (LD's
    fixed-rate budget faster than HQ's variable slice bytes; HQ q=32
    faster than q=0 because most coefficients quantise to zero).
  - **encode**: same 3 scenarios on the encoder side. Indicative
    numbers: HQ q=0 ≈ 80 µs, HQ q=32 ≈ 59 µs, LD q=16 ≈ 65 µs.
  - **roundtrip**: same 3 scenarios end-to-end (encode + decode in
    one iteration). Indicative numbers: HQ q=0 ≈ 156 µs, HQ q=32 ≈
    115 µs, LD q=16 ≈ 114 µs — matches the sum of the encode + decode
    rows to within 1-2 µs, validating the harnesses are not stepping on
    each other.
  - All three harnesses share a single `synth_yuv420(width, height,
    seed)` helper (xorshift32 Marsaglia 13/17/5; identical formulation
    across files) so sibling-bench rows stay numerically comparable.
    Input synthesis is outside the timed region for `decode` and
    `encode` (Criterion `bench_with_input`); for `roundtrip` the
    encode + decode cost dominates so the input is reused across
    iterations via the same input tuple.

- **Encoder-side rate-control fuzz oracle for VC-2 HQ + LD profiles**
  (round-179). New `tests/encoder_rate_control_fuzz_oracle.rs` (13
  tests) exercises the four rate-control variants (`PerPicture`, `Cbr`,
  `Vbv`, `VbvHysteresis`) on both `encode_hq_sequence_with_size_target`
  and `encode_ld_sequence_with_size_target` through a Cartesian sweep of
  `target_bytes ∈ {0, 1, 100, 1_000, 10_000}` × `buffer_bytes ∈ {0, 1,
  100, 10_000}` × `max_drain_per_picture ∈ {0, 1, 100, 10_000}` —
  asserting that every combination produces a non-empty stream, every
  per-picture row's `actual_payload_bytes > 0`, the post-clamp VBV
  invariant `running_surplus_bytes ≤ buffer_bytes` holds, and the
  encoded stream round-trips through the `DiracDecoder` to the expected
  frame count without panic / debug-assert. The encoder-side analogue
  of round-165's decoder fuzz oracle, covering panic / livelock /
  integer-overflow classes on the rate-control surface.
  - Strict-generalisation invariants pinned across pathological-target
    sweeps: `Vbv { buffer_bytes: 0 }` ≡ `PerPicture` (r146/r149) and
    `VbvHysteresis { max_drain_per_picture: 0, .. }` ≡ `PerPicture`
    (r159) and `VbvHysteresis { buffer_bytes: B, max_drain: B }` ≡
    `Vbv { buffer_bytes: B }` (r159) are each asserted byte-identical
    on both HQ + LD across `target ∈ {200, 1_000, 10_000}`. A future
    encoder change that subtly breaks one of these documented
    degeneracies is caught by the oracle rather than by downstream
    consumers.
  - Pathological pixel inputs: all-zero / all-`0xFF` / mid-grey solid
    luma + single-pixel-pulse frames are each fed through every
    rate-control variant. Tests that the picker's `qindex ∈ floor..=127`
    walk handles "no AC energy" (q=0 fits trivially) and "maximally
    concentrated AC energy" (one coefficient at the extreme,
    background near-zero) without panic / livelock.
  - Empty input slice (`frames.len() == 0`) is pinned to produce an
    empty per-picture report + a non-empty stream containing just the
    sequence-header / end-of-sequence brackets, for every variant.
  - Cumulative surplus identity from the r152 documentation
    (`running_surplus_bytes == (i+1) × target − Σ actual_payload_bytes`)
    is pinned for `PerPicture` + `Cbr` on a 5-picture run, on both HQ
    and LD. The r146/r149/r159 VBV variants additionally clamp the
    accumulator at `buffer_bytes`, asserted by the per-row VBV
    invariant above.

### Documented

- **LD picture payload bytes are linear in the requested target**
  (round-179 oracle finding). `ld_picture_payload_bytes(params) ==
  header_bytes + params.slice_bytes_numer`, and
  `derive_ld_slice_bytes_for_target` sets `slice_bytes_numer ≈
  target_picture_bytes − header`, so a `target_bytes = u32::MAX`
  request to `encode_ld_sequence_with_size_target` allocates a
  multi-GiB picture buffer per picture and OOMs the test runner before
  the picker ever returns. This is the documented §13.5.3.2 LD
  contract — the per-slice budget is fixed-rate and every slice writes
  its full budget regardless of coefficient energy — not a defect. The
  fuzz oracle's `target_bytes` sweep caps at 10_000 bytes per picture
  to keep the test tractable without exercising new code paths.

- **Decoder-side robustness fuzz oracle for malformed VC-2 LD/HQ inputs**
  (round-165). New `tests/decoder_fuzz_oracle.rs` (14 tests) drives the
  `DiracDecoder` through a structured corpus of corrupted byte streams
  and asserts the decoder either decodes (gracefully degraded) or
  returns a clean `Error::{InvalidData, Unsupported, NeedMore, Eof}` —
  never panics, never integer-overflows, never livelocks. Coverage:
  - **Truncation walks** — every 7-byte-step prefix of valid HQ + LD
    streams (~280 cuts each).
  - **Single-byte mutation walks** — every 11-byte-step position in a
    valid HQ + LD stream flipped to `{0x00, 0xFF, !orig}` (~600
    mutations per profile).
  - **Pathological gibberish** — empty / all-zero / all-`0xFF` / BBCD-
    prefix-spam (64 KiB) buffers, the every-parse-code walk (all 256
    single-byte codes wrapped in a parse-info).
  - **Oversized parse-info offsets** — `next_parse_offset = u32::MAX`
    and `0xFFFF_FFF0` (near-wrap), confirming the stream walker's
    `saturating_add` + fallback-to-byte-search recovery.
  - **Hand-crafted invalid headers** — unknown `base_video_format`,
    out-of-range `dwt_depth=99` (rejected by
    `PictureError::UnsupportedDwtDepth` → `Error::InvalidData`).

### Fixed

- **`BitReader::read_uint` livelock on post-EOF interleaved exp-Golomb**
  (round-165 fuzz-oracle dependency). When the unbounded `BitReader`
  walks past end-of-data it returns `0` from `read_bit` forever — a
  naive `read_uint` would then loop indefinitely (no follow-bit ever
  becomes `1`). The unbounded reader is called by the sequence-header
  parser (§10) and core-syntax transform-parameter parser (§11.3.1),
  both of which would hang on a truncated stream. Added an EOF +
  31-iteration cap that returns the partial value when either fires.
  Validated by all 14 fuzz oracle tests passing in <1 s total
  (without the fix, the HQ truncation walk hangs at cut=14).

- **Bounded exp-Golomb `read_uintb` accumulator overflow** (round-165
  fuzz-oracle dependency). The `BoundedBitReader::read_uintb` (used
  inside arithmetic-coded blocks) and the per-picture `Funnel::read_uintb`
  (used by the §13.5.3 LD and §13.5.4 HQ slice decoders) both shift the
  exp-Golomb accumulator left without bound. On a well-formed stream the
  loop terminates in a handful of iterations, but on a truncated /
  corrupted slice whose every data bit reads as 0 the bounded reader's
  `bits_left / 2` upper bound is much larger than 31 — `value <<= 1`
  wraps to 0 and `value - 1` underflows (`attempt to subtract with
  overflow` in debug). Same 31-iteration cap applied to both, matching
  the unbounded fix above.

- **Signed exp-Golomb negate-overflow on `i32::MIN`** (round-165
  fuzz-oracle dependency). `read_sint{,b}` cast the unsigned magnitude
  to `i32` and then negate when the sign bit follows. A saturated
  magnitude of exactly `0x8000_0000` cast to `i32::MIN` makes the
  `-value` step overflow. Clamp to `i32::MAX` before the cast on all
  three readers (`BitReader::read_sint`, `BoundedBitReader::read_sintb`,
  `Funnel::read_sintb`).

- **`quant_factor` multiplication overflow on out-of-spec qindex**
  (round-165 fuzz-oracle dependency). `decode_hq_slice` reads qindex
  via `read_uint_lit(1)` (8-bit field) without masking to the 7-bit
  spec range, so a malformed HQ slice can deliver `qindex = 0xFF`.
  `quant_factor(255)` then walks `1u64 << (255/4) = 1u64 << 63` and
  `503_829 * (1<<63)` (and the other branch multipliers) overflows u64
  before the docstring's "saturate at `u32::MAX`" final step can fire.
  Clamp `q` to 127 (the spec maximum) inside `quant_factor` so the
  documented "valid for any q ≥ 0" contract holds without forcing every
  caller to pre-mask the field. Saturation behaviour is preserved for
  in-spec inputs — q == 127 was already the previous overflow ceiling.

- **VC-2 drain-rate-hysteresis (`VbvHysteresis`) rate-control variant**
  (round-159). Both `HqRateControl` and `LdRateControl` gain a fourth
  variant `VbvHysteresis { buffer_bytes, max_drain_per_picture }`,
  identical to r146/r149's `Vbv` on the bucket *fill* side (savings
  forfeited above `buffer_bytes`) but with the savings *spent* on any
  one picture additionally clamped at `max_drain_per_picture`. Plain
  `Vbv` lets a single picture drain the entire bucket in one step
  (request can equal `target + buffer_bytes` the moment the bucket
  fills); `VbvHysteresis` instead spreads the savings across multiple
  pictures by capping the instantaneous drain. The remaining savings
  stay in the bucket for the next picture, smoothing the cliff between
  "bucket full → one picture spends everything → bucket empty" and the
  zero-carry pictures that follow it.
  - Per-picture request bound: `target ≤ requested ≤ target +
    min(buffer_bytes, max_drain_per_picture)` on the savings side.
    The debt branch (carry > 0 on LD, carry < 0 on HQ q=127 edge)
    is unchanged from `Vbv` — debt repayment is mandatory, not
    rate-limited.
  - Strict generalisation invariants pinned by tests:
    `max_drain_per_picture == 0` ≡ `PerPicture` (no savings can be
    spent — byte-identical stream); `max_drain_per_picture >=
    buffer_bytes` ≡ `Vbv { buffer_bytes }` (drain cap inert — byte-
    identical stream). Matches the r146/r149 `Vbv` generalisation
    relationship one-for-one.
  - The post-encode carry clamp is the same as `Vbv` (savings forfeit
    at `buffer_bytes`), so the `running_surplus_bytes ≤ buffer_bytes`
    telemetry invariant from r152 holds verbatim. Pure encoder-side
    rate-shaping policy — any qindex-per-picture sequence the encoder
    produces remains spec-conformant under BBC Dirac Specification
    v2.2.3 §13.5.4 (HQ) and SMPTE ST 2042-1 §13.5.2 / §13.5.3.2 (LD).
  - 10 new tests across `tests/encoder_hq_sequence_rate.rs` (16 → 21)
    and `tests/encoder_ld_sequence_rate.rs` (14 → 19): zero-drain
    degeneracy → `PerPicture`; `drain >= buffer` degeneracy → `Vbv`
    (both `drain == buffer` and `drain == u32::MAX`); drain-cap upper
    bound on requested bytes; `surplus ≤ buffer_bytes` invariant
    preserved; determinism. Bitstream output for the existing `Vbv` /
    `Cbr` / `PerPicture` variants is byte-identical to r152.

- **Per-picture `running_surplus_bytes` rate-control telemetry**
  (round-152). Both `LdPictureRate` (LD driver) and `HqPictureRate`
  (HQ driver) gain a public `running_surplus_bytes: i64` field reported
  *after* any [`Vbv`] bucket clamp has been applied. Sign convention:
  **positive = cumulative savings** (future pictures may spend it),
  **negative = cumulative debt** (future pictures must pay it back).
  Computed identically across all three rate-control modes as the
  signed deviation of the ideal cumulative budget from the encoded
  cumulative bytes (`pictures_seen × target_bytes − Σ
  actual_payload_bytes`); modes differ only in whether the next
  picture's request *uses* the accumulator. Under VBV the bucket clamp
  additionally guarantees `running_surplus_bytes ≤ buffer_bytes` per
  row. Pure telemetry — the bitstream output is byte-identical to
  r149/r146, only the report struct grew a field.
  - 6 new tests across `tests/encoder_ld_sequence_rate.rs` (11 → 14)
    and `tests/encoder_hq_sequence_rate.rs` (13 → 16): per-mode
    cumulative-budget-minus-Σ-actual identity for `PerPicture` and
    `Cbr`, and the VBV bucket-cap `surplus ≤ buffer_bytes` invariant.

- **VC-2 LD leaky-bucket (VBV) rate-control variant** (round-149). The
  LD analogue of r146's HQ `Vbv`: a third `LdRateControl` strategy
  alongside r134's `PerPicture` and `Cbr`,
  `LdRateControl::Vbv { buffer_bytes }`. Same carry semantics as `Cbr`
  but the spendable undershoot (i.e. `max(-carry, 0)` of LD's signed
  accumulator) is clamped at `buffer_bytes`, so every per-picture
  request is capped at `target_bytes + buffer_bytes` (an instantaneous
  peak-size guarantee that unbounded `Cbr` lacks). Savings above
  `buffer_bytes` are forfeited rather than accumulated. Pure
  encoder-side rate-shaping policy; bitstream output remains
  spec-conformant under SMPTE ST 2042-1 §13.5.2 / §13.5.3.2 (any
  per-slice qindex / slice-bytes the encoder produces is legal).
  - Strict generalisation invariants both pinned by tests:
    `Vbv { buffer_bytes: 0 }` → byte-identical to `PerPicture`;
    `Vbv { buffer_bytes: u32::MAX }` → byte-identical to `Cbr` (the
    cap can never bite). Intermediate `buffer_bytes` trade peak-size
    cap against long-run average. Matches the r146 HQ generalisation
    relationship one-for-one.
  - 5 new tests in `tests/encoder_ld_sequence_rate.rs` (6 → 11 total):
    zero-buffer = PerPicture; infinite-buffer = Cbr; peak cap
    (`requested ≤ target + buffer_bytes`) on a 6-picture run with a
    `target = 600 B` / `buffer = 64 B` setup; positive smoothing
    on a 5-picture mid-range budget with `buffer = target/4`;
    determinism. Existing wrapper-matches-`_report` test extended
    to cover three Vbv `buffer_bytes` points (0, 128, u32::MAX). All
    decoded streams round-trip to N frames.

- **VC-2 HQ leaky-bucket (VBV) rate-control variant** (round-146). A
  third `HqRateControl` strategy alongside r141's `PerPicture` and `Cbr`:
  `HqRateControl::Vbv { buffer_bytes }`. Same carry semantics as `Cbr`
  but the savings the next picture may spend are clamped at
  `buffer_bytes`, so every per-picture request is capped at
  `target + buffer_bytes` (an instantaneous peak-size guarantee that
  unbounded `Cbr` lacks). Savings above `buffer_bytes` are forfeited
  rather than accumulated. Pure encoder-side rate-shaping policy;
  bitstream output remains spec-conformant under BBC Dirac Specification
  v2.2.3 §13.5.2 / §13.5.4 (any qindex-per-picture sequence the encoder
  produces is legal).
  - Strict generalisation invariants both pinned by tests:
    `Vbv { buffer_bytes: 0 }` → byte-identical to `PerPicture`;
    `Vbv { buffer_bytes: u32::MAX }` → byte-identical to `Cbr` on
    streams where no picture overshoots (the cap never bites).
    Intermediate `buffer_bytes` trade peak-size cap against long-run
    average.
  - 5 new tests in `tests/encoder_hq_sequence_rate.rs` (8 → 13 total):
    zero-buffer = PerPicture; infinite-buffer = Cbr; peak cap
    (`requested ≤ target + buffer_bytes`) on a 6-picture undershoot
    stream with `buffer = 128 B`; positive smoothing on a 5-picture
    mid-range budget with `buffer = target/4`; determinism. All decoded
    streams round-trip to N frames.

- **VC-2 HQ multi-picture rate-controlled sequence encoder** (round-141).
  A stream-level driver on top of round-138's per-picture
  `pick_hq_picture_qindex` primitive: given a sequence of YUV frames plus
  a target byte-budget, it sizes + qindex-picks + encodes each picture
  and emits a complete VC-2 HQ elementary stream (sequence header `0x00`
  + one HQ intra picture per frame alternating non-reference `0xE8` /
  reference `0xEC` + end-of-sequence `0x10`, with the `next_parse_offset`
  / `previous_parse_offset` chain wired up) that round-trips through
  `DiracDecoder` to one decoded frame per input frame. The HQ analogue
  of round-134's LD multi-picture driver. BBC Dirac Specification v2.2.3
  §13.5.2 (per-slice qindex header), §13.5.4 (`slice_quantisers(qindex)`),
  §9.6 / §10.4 (parse-info sequence framing).
  - New public API surface in `oxideav_dirac::encoder`:
    - `encode_hq_sequence_with_size_target(seq, base, frames,
      target_bytes, mode) -> Vec<u8>` — full HQ elementary stream.
    - `encode_hq_sequence_with_size_target_report(seq, base, frames,
      target_bytes, mode) -> (Vec<u8>, Vec<HqPictureRate>)` — same
      stream plus per-picture telemetry (requested vs. actual bytes,
      chosen qindex, written parse code).
    - `HqRateControl::{PerPicture, Cbr}` — `PerPicture` sizes every
      picture independently to `target_bytes` (no carry-over);
      `Cbr` carries each picture's signed residual (`target − actual`,
      monotone non-negative outside the q=127 floor edge case) into the
      next picture's request so undershoots become future spending
      power.
    - `HqPictureRate` — `{ picture_number, requested_bytes,
      actual_payload_bytes, qindex, parse_code }`. The picker's
      `actual_payload_bytes ≤ requested_bytes` contract is preserved
      end-to-end; `parse_code` is the byte actually written into the
      picture's parse-info (alternating `0xE8` / `0xEC`).
    - Driver clears `base.slice_size_target` before invoking the picker
      so the chosen picture-level qindex is the one written into every
      slice header — same wrapper invariant as
      `encode_single_hq_intra_stream_with_size_target` (round-138).
  - New test file `tests/encoder_hq_sequence_rate.rs` (8 tests, all
    passing). Headline acceptance: a **5-picture** 64x64 4:2:0 sequence
    with `slice_size_scaler = 2`, 4x4 slices, mid-frequency-cross
    fixtures, exercised under both `PerPicture` and `Cbr`:
    - **PerPicture** at `target = 1234 B` → actuals `[1221, 1223, 1185,
      1221, 1223]` (all ≤ target; picker never overshoots) at qindexes
      `[23, 24, 26, 27, 27]`. Decoded 5 frames all > 8 dB Y PSNR.
    - **Cbr** at `target = 1031 B` (N=5 → ideal 5155 B total) → per-pic
      `[1015, 1035, 1035, 1021, 1047]` at qindexes `[32, 32, 33, 32,
      31]`. CBR total **5153 B vs ideal 5155 B** (0.04% miss; PerPicture
      total 5035 B at 2.3% under) — CBR carry converges the running
      stream total to within 2 bytes of `N × target` by spending each
      picture's undershoot on the next picture's qindex.
    Other tests pin: parse-code alternation (`0xE8`/`0xEC`/`0xE8`/`0xEC`)
    matches `encode_hq_intra_multi_stream`; wrapper output is
    byte-identical to `_report` for both modes; determinism (same input
    → byte-identical stream); empty frame list yields a valid sh+eos
    stream that decodes to zero frames; CBR request is monotone
    non-decreasing across the sequence when every picture undershoots
    (carry-only-grows contract); generous budget (≥ 2× max q=0 ceiling
    across all frames) keeps every picture at qindex 0 and reconstructs
    > 25 dB Y PSNR.

- **VC-2 HQ picture-level rate-control picker** (round-138). The HQ
  analogue of round-131's `pick_ld_picture_qindex`: given a target
  picture-payload byte budget, walk `qindex ∈ params.qindex..=127` and
  return the smallest qindex for which the entire HQ picture's encoded
  payload — with that single qindex written into every slice header —
  fits within the budget. Unlike LD (deterministic picture bytes from
  `slice_bytes_numer/denom`), HQ length bytes track each slice's actual
  coefficient block size, so picture bytes shrink monotonically with
  rising qindex (the dead-zone forward quantiser drives more
  coefficients toward zero → fewer interleaved exp-Golomb bits per
  slice). The picker exploits that monotonicity to stop at the first
  qindex that fits. BBC Dirac Specification v2.2.3 §13.5.2 (per-slice
  qindex header) + §13.5.4 (`slice_quantisers(qindex)`).
  - New public API surface in `oxideav_dirac::encoder`:
    - `hq_picture_payload_bytes_at_qindex(seq, params, y, u, v, qindex)
      -> usize` — content-dependent exact picture-payload byte count at
      a uniform qindex (ignores `params.slice_size_target`).
    - `pick_hq_picture_qindex(seq, params, y, u, v, target_bytes)
      -> u32` — smallest qindex whose picture bytes ≤ target.
    - `hq_picture_qindex_diagnostic(seq, params, y, u, v, target_bytes)
      -> (u32, usize)` — picker plus actual picture bytes at the chosen
      qindex.
    - `encode_single_hq_intra_stream_with_size_target(seq, base,
      target_bytes, picture_number, y, u, v) -> (Vec<u8>, u32, usize)`
      — full elementary stream wrapper. Clears `slice_size_target` on
      the input params so the §13.5.4 per-slice search does not also
      fire; the chosen picture-level qindex is the one written into
      every slice header.
  - New test file `tests/encoder_hq_picture_qindex.rs` (9 tests, all
    passing). Headline acceptance: a 64×64 4:2:0 mid-frequency fixture
    with q=0 ceiling 1934 B and q=127 floor 839 B sees three budgets
    `[907, 1420, 1934]` land at actual picture bytes `[905, 1388,
    1934]` (all ≤ target, picker never overshoots) with chosen
    qindexes `[35, 16, 0]` — the small budget escalates to a
    mid-aggressive quantiser, the large budget stays lossless at q=0.
    Other tests pin: predictor matches actual encode at every qindex
    `[0, 1, 5, 16, 32, 64, 100, 127]`; picture bytes are monotonically
    non-increasing across `qindex ∈ 0..=127`; picker is monotone in
    target_bytes (smaller budget → equal-or-higher qindex); flat
    content keeps q=0 even under a tight budget; unfit-target case
    returns q=127 with the q=127 floor as actual bytes (graceful
    degradation, stream still decodes); determinism (same input →
    byte-identical stream); wrapper-ignores-`slice_size_target`
    invariant.

- **VC-2 LD multi-picture rate-controlled sequence encoder** (round-134).
  A stream-level driver on top of the round-131 per-picture
  `pick_ld_picture_qindex` primitive: given a sequence of YUV frames plus
  a target byte-budget, it sizes + qindex-picks + encodes each picture
  and emits a complete VC-2 LD elementary stream (sequence header `0x00`
  + one `0xC8` LD intra picture per frame + end-of-sequence `0x10`, with
  the `next_parse_offset` / `previous_parse_offset` chain wired up) that
  round-trips through `DiracDecoder` to one decoded frame per input frame.
  SMPTE ST 2042-1 §13.5.3.2 (slice byte budget), §13.5.2 (per-slice
  qindex header), §D.1.1 (LD parse-code restriction), §9.6 / §10.4
  (parse-info framing).
  - New public API surface in `oxideav_dirac::encoder`:
    - `encode_ld_sequence_with_size_target(seq, base, frames,
      target_bytes, mode) -> Vec<u8>`
    - `encode_ld_sequence_with_size_target_report(...) -> (Vec<u8>,
      Vec<LdPictureRate>)` — same stream plus per-picture telemetry
      (requested vs. actual bytes + chosen qindex)
    - `enum LdRateControl { PerPicture, Cbr }`
    - `struct LdPictureRate { picture_number, requested_bytes,
      actual_payload_bytes, qindex }`
  - `PerPicture` sizes every picture independently to `target_bytes`;
    `Cbr` carries each picture's signed byte over/undershoot into the
    next picture's budget via a running accumulator. Tiny CBR requests
    are clamped up to the smallest viable picture (header + 2·N_slices)
    rather than dropped, so a CBR run can never emit an invalid picture.
  - New test file `tests/encoder_ld_sequence_rate.rs` (6 tests, all
    passing). Headline acceptance: a 3-frame sequence at a fixed 1024-byte
    per-picture budget round-trips to 3 decoded frames each at exactly
    1024 bytes (0% miss, well under ±10%) with content-driven qindexes
    `[37, 35, 37]`; a 5-frame CBR run at `target=900` lands the stream
    total at exactly `5×900 = 4500` bytes (0% miss, under ±5%).
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
