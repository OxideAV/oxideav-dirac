# oxideav-dirac

Pure-Rust decoder foundation for **Dirac** (BBC wavelet video codec) and its
intra-only subset **SMPTE VC-2**. Dirac is specified in the BBC "Dirac
Specification" (v2.2.3, 2008) and covers both long-GOP motion-compensated
video and a low-delay intra-only mode. Zero C dependencies.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace) framework
but usable standalone.

## Status

This crate is an **early foundation** pass. Dirac is a large codec
(wavelet transforms + custom arithmetic coder + motion compensation);
the goal here is to land the bitstream framing and primitives first.

| Area                                | State                                                         |
|-------------------------------------|---------------------------------------------------------------|
| Parse Info header (`BBCD` + codes)  | Parse + walk next/previous offsets                            |
| Parse code taxonomy                 | Sequence-header / end-of-sequence / auxiliary / padding       |
|                                     | Intra / inter / reference / AC / low-delay predicates         |
| Bit reader (MSB-first)              | Implemented — `read_bit`, `read_nbits`, `read_uint_lit`. **Round-165**: `read_uint` EOF + 31-iteration cap so a post-EOF interleaved exp-Golomb (all-zero `read_bit`) returns the partial value instead of live-locking — caught by the new fuzz-oracle truncation walk on the sequence-header / core-transform-parameters parsers. |
| Exp-Golomb (interleaved)            | Unsigned + signed decoders (`read_uint` / `read_sint`)        |
| Sequence header                     | Parse parameters + base video format + source overrides       |
| Predefined video formats            | Full table (indices 0-20) with frame size, chroma, range etc. |
| Arithmetic decoder                  | Binary multi-context engine + probability LUT (Annex B)       |
| Wavelet transforms                  | Inverse DWT (all 7 filters) + §13.3 intra DC prediction wired |
| VC-2 / Dirac intra **decode**       | Full coefficient decode. **Bit-exact vs ffmpeg** on all 5 intra-only docs-corpus fixtures (LD 0x88, core-syntax 0x0C, 4:2:0 + 4:2:2, depth 3 + 4) |
| Dirac core-syntax inter **decode**  | OBMC motion-comp + §11.3 residue. **Round-125**: §13.2.1 inter quant-offset wired (`(qf*3+4)/8` for inter pictures, intra path unchanged; `0/1` corner cases agree); `inverse_quant_for(qcoeff, q, is_intra)` plumbed through `core_transform_component`/`decode_subband_ac`/`decode_subband_vlc`. **Round-128**: §12.3.6.6 Case 4 DC prediction for intra blocks inside inter pictures now uses the spec's floor `mean()` (`div_euclid`) instead of Rust's truncating `/`, eliminating a 1-LSB upward bias on negative neighbour-averages that propagated through OBMC into a localised ~1% pixel-gap region. Two inter ReportOnly fixtures (`i-then-p-320x240`, `i-p-b-320x240`) are now **bit-exact**; `interlaced-720x576-i-then-p-wavelet-5-3` at 99.68% pixel-exact (the residual gap is the LeGall-5,3 inter path). |
| Encoder — VC-2 HQ intra             | Implemented (8/10-bit, 4:2:0/4:2:2/4:4:4, 6 wavelets). **Round-111**: optional §12.4.5.3 custom quant matrix (`custom_quant_matrix` flag + `with_custom_quant_matrix`; default emits the Annex E.1 default). **Round-114**: optional §13.5.4 per-slice adaptive qindex (`slice_size_target` + `with_slice_size_target`; default `None` = legacy constant qindex). Each slice picks the smallest qindex ≥ floor whose every component fits the byte budget; cheap slices keep the floor (lossless-ish), busy slices escalate just enough to fit (no silent truncation). HQ has no §13.5.1 DC prediction so slices are independent; decoder already reads per-slice qindex. **Round-138**: picture-level constant-qindex rate-control picker (`encode_single_hq_intra_stream_with_size_target`, `pick_hq_picture_qindex`, `hq_picture_payload_bytes_at_qindex`, `hq_picture_qindex_diagnostic`) — the HQ analogue of r131's `pick_ld_picture_qindex`. Given a target picture-payload byte budget, walks `qindex ∈ floor..=127` and returns the smallest qindex whose constant-qindex picture bytes ≤ target (HQ length bytes track each slice's actual coefficient block size, so picture bytes shrink monotonically as qindex rises). The wrapper clears `slice_size_target` so the chosen qindex is the one actually written into every slice header; the two knobs stay independent. Three-budget acceptance test on a 64x64 4:2:0 fixture pins targets `[907, 1420, 1934]` bytes → actuals `[905, 1388, 1934]` (all ≤ target) → qindexes `[35, 16, 0]` — small budget escalates to mid-aggressive quantiser, large budget stays lossless at q=0. **Round-141**: HQ multi-picture rate-controlled sequence driver (`encode_hq_sequence_with_size_target`, `..._report`, `HqRateControl::{PerPicture, Cbr}`, `HqPictureRate`) — the HQ analogue of r134's LD driver. Encodes a slice of input frames into a full HQ elementary stream (seq header + alternating `0xE8`/`0xEC` HQ intra pictures + EOS) that round-trips to one frame each. `PerPicture` sizes every picture independently to `target_bytes` (picker never overshoots → actual ≤ target); `Cbr` carries each picture's undershoot residual (`target − actual`, monotone non-negative outside q=127) into the next picture's request so savings become future spending power. 5-picture run at `target=1234` → actuals `[1221, 1223, 1185, 1221, 1223]` (all ≤ target) at qindexes `[23, 24, 26, 27, 27]`; 5-picture CBR at `target=1031` (ideal `5×1031=5155`) → stream total **5153 B** (0.04% miss; PerPicture total 5035 B at 2.3% under) — CBR converges to within 2 bytes of `N × target` by spending undershoots on the next picture's qindex. **Round-146**: `HqRateControl::Vbv { buffer_bytes }` third variant — leaky-bucket variant of CBR; same carry semantics but the spendable savings are clamped at `buffer_bytes`, so every per-picture request is peak-capped at `target + buffer_bytes` (savings above the bucket are forfeited). Strict generalisation: `buffer_bytes == 0` ≡ PerPicture (byte-identical stream); `buffer_bytes == u32::MAX` ≡ Cbr on no-overshoot streams (byte-identical). Pure encoder-side rate-shaping policy — any qindex-per-picture sequence remains spec-conformant under §13.5.2/§13.5.4. 5 new tests pin the cap invariant + degeneracies + determinism. **Round-152**: `HqPictureRate.running_surplus_bytes: i64` telemetry — signed running surplus reported after the VBV bucket clamp ("positive = savings, future may spend; negative = debt, future must repay"), computed mode-agnostically as `pictures_seen × target − Σ actual_payload_bytes` per row. Identical addition on `LdPictureRate`. Bitstream output unchanged; pure observability on the same per-picture report. 3 new HQ tests pin the cumulative-budget-minus-Σ-actual identity (PerPicture + Cbr) and the VBV `surplus ≤ buffer_bytes` invariant. **Round-159**: `HqRateControl::VbvHysteresis { buffer_bytes, max_drain_per_picture }` fourth variant — drain-rate-limited variant of r146's `Vbv`. Same bucket fill / forfeit semantics (savings clamped at `buffer_bytes`) but the savings *spent* on any one picture are additionally capped at `max_drain_per_picture`, so a full bucket cannot be drained onto a single picture's request. Per-picture request: `target ≤ requested ≤ target + min(buffer_bytes, max_drain_per_picture)` on the savings side; the q=127 debt branch is unchanged. Strict generalisation: `max_drain_per_picture == 0` ≡ PerPicture (byte-identical); `max_drain_per_picture >= buffer_bytes` ≡ `Vbv` (drain cap inert, byte-identical). Pure encoder-side rate-shaping policy; bitstream remains §13.5.4-conformant. 5 new tests pin both degeneracies + drain cap + r152 `surplus ≤ buffer_bytes` preserved + determinism. |
| Encoder — VC-2 LD intra             | Implemented (ffmpeg-bit-exact at q=0). **Round-111**: same optional §12.4.5.3 custom quant matrix as the HQ path. **Round-131**: picture-level rate-control picker (`encode_single_ld_intra_stream_with_size_target`, `pick_ld_picture_qindex`, `derive_ld_slice_bytes_for_target`, `ld_picture_payload_bytes`) — the LD analogue of the HQ §13.5.4 `with_slice_size_target`. Given a target picture-byte budget, derives `slice_bytes_numer/denom` so the encoded picture lands within ±1 byte of target (4-pass fixed-point converges in 1-2 rounds; header growth is sub-linear in numer), then walks `qindex ∈ 0..=127` for the smallest qindex that keeps every slice's `luma_bits + chroma_bits ≤ payload_bits` (no Funnel-truncation). Three-budget acceptance test pins targets `[200, 1024, 4096]` bytes → actuals `[200, 1024, 4096]` (0% miss) → qindexes `[127, 37, 6]` — small budget escalates to max-aggressive quantiser, large budget stays near-lossless. **Round-134**: multi-picture rate-controlled sequence driver (`encode_ld_sequence_with_size_target`, `..._report`, `LdRateControl::{PerPicture, Cbr}`) — encodes a slice of input frames into a full LD elementary stream (seq header + one `0xC8` picture/frame + EOS) that round-trips to one frame each. `PerPicture` sizes every picture to `target_bytes`; `Cbr` carries each picture's byte over/undershoot into the next via a running accumulator (tiny requests clamped up to header + 2·N_slices, never dropped). 3-frame run at 1024 B/picture → 3 frames at exactly 1024 B each; 5-frame CBR at `target=900` → stream total exactly `5×900` (0% miss). **Round-149**: `LdRateControl::Vbv { buffer_bytes }` third variant — LD analogue of r146's HQ VBV; same carry semantics as `Cbr` but the spendable savings (LD's `max(-carry, 0)`) are clamped at `buffer_bytes`, so every per-picture request is peak-capped at `target + buffer_bytes` (savings above the bucket are forfeited). Strict generalisation: `buffer_bytes == 0` ≡ PerPicture (byte-identical stream); `buffer_bytes == u32::MAX` ≡ Cbr (cap never bites, byte-identical). Pure encoder-side rate-shaping policy — any qindex-per-slice sequence remains spec-conformant under SMPTE ST 2042-1 §13.5.2 / §13.5.3.2. 5 new tests pin the cap invariant + both degeneracies + determinism. **Round-152**: `LdPictureRate.running_surplus_bytes: i64` telemetry — signed running surplus reported after the VBV bucket clamp ("positive = savings, negative = debt"), mode-agnostic `pictures_seen × target − Σ actual_payload_bytes` per row. Bitstream output unchanged. 3 new LD tests pin the identity (PerPicture + Cbr) plus the VBV `surplus ≤ buffer_bytes` post-clamp invariant. **Round-159**: `LdRateControl::VbvHysteresis { buffer_bytes, max_drain_per_picture }` fourth variant — LD analogue of r159's HQ `VbvHysteresis`. Same bucket fill / forfeit semantics as r149's `Vbv` (savings clamped at `buffer_bytes` on the signed accumulator) but the savings *spent* on any one picture (LD's `max(-carry, 0)`) are additionally capped at `max_drain_per_picture`. The mandatory debt-repayment branch (`carry > 0`) is unchanged from `Vbv` — only the savings spend side is rate-limited. Per-picture request: `target ≤ requested ≤ target + min(buffer_bytes, max_drain_per_picture)` on the savings side; debt-branch request unchanged. Strict generalisation: `max_drain_per_picture == 0` ≡ PerPicture on smooth fixtures (byte-identical); `max_drain_per_picture >= buffer_bytes` ≡ `Vbv { buffer_bytes }` (drain cap inert, byte-identical). Pure encoder-side rate-shaping policy; bitstream remains §13.5.2/§13.5.3.2-conformant. 5 new tests pin both degeneracies + drain cap + r152 telemetry invariant preserved + determinism. **Round-179**: `tests/encoder_rate_control_fuzz_oracle.rs` — 13 tests / Cartesian sweep across all four LD + HQ rate-control variants (`PerPicture`/`Cbr`/`Vbv`/`VbvHysteresis`) × pathological `target_bytes` × `buffer_bytes` × `max_drain_per_picture` × pathological pixel inputs (solid / pulse) × empty-frame edge cases. Asserts no panic / no overflow / VBV bucket invariant `running_surplus_bytes ≤ buffer_bytes` post-clamp / `Vbv{buffer=0}` ≡ `PerPicture` byte-identical (r146/r149) / `VbvHysteresis{drain=0,..}` ≡ `PerPicture` + `VbvHysteresis{B,D≥B}` ≡ `Vbv{B}` (r159) / surplus identity `(i+1)·target − Σ actual` for non-VBV variants. Encoder-side analogue of r165's decoder oracle. **Round-193**: inter-side companion at `tests/encoder_inter_fuzz_oracle.rs` (9 tests) — precision × OBMC passes × search range diagonal sweep + residue wavelet/depth/qindex sweep + adaptive-flag boolean sweep + pathological pixel inputs (zero / 0xFF / mid-grey / single-pixel pulse) + same-frame zero-motion + `mv_search_range=0` extreme + `residue qindex=127` extreme + determinism on both residue-on and residue-off paths. Pins no panic / no debug-assert / no integer overflow / no livelock + clean 2-frame round-trip on every accepted (`InterEncoderParams`, `InterInputPicture`, `InterInputPicture`) combination. Crate-wide test count grows 329 → 338 (+9). |
| Encoder — Dirac core-syntax intra | AC-coded `0x0C` ref, q=0 near-lossless, ffmpeg-validated. **Round-100**: optional §11.3.3 spatial partition — per-level codeblock grid (`codeblocks`) + `codeblock_mode` 0 (single quantiser) / 1 (per-codeblock differential quantiser offset, accumulating by reference per §13.4.3.2). Decoder's cumulative-offset bug fixed to match. Default stays single-codeblock. **Round-103**: empty codeblocks of a partitioned subband are coded as §13.4.3.3 `zero_flag` skips (smaller stream + exercises the decoder's skip branch); a skip emits no quant offset and does not advance the running quantiser, so the encoder's two former phases merge into one skip-aware codeblock walk and `codeblock_offset` indexes non-skipped codeblocks. **Round-108**: VLC (non-arithmetic) intra reference encoder (`encode_*_vlc`, parse code `0x4C`) — the §13.4.2.2 plain-exp-Golomb counterpart to the decoder's `decode_subband_vlc` (previously encoder-unreachable). Same whole-picture framing + shared codeblock/skip/quant-offset walk as the AC path; only the entropy primitives change (`zero_flag`→raw bit, offset+coeffs→`read_sintb`, no contexts). Applies no entropy-coder rounding, so at q=0 it is strictly lossless — bit-exact on the testsrc V-plane gradient where the AC path keeps a ~1-LSB roughness. |
| Encoder — Dirac inter               | 1-ref P (`0x09`) **and 2-ref bipred B** (`0x0A`); sub-pel ME (qpel default for both P and B), OBMC-aware ME refinement (#186), per-block `Ref1Only` / `Ref2Only` / `Ref1And2` decision search for bipred, **per-block adaptive sub-pel-vs-integer-pel selection on BOTH the 1-ref P-path (round-73, `inter_adaptive_int_pel`) and the 2-ref bipred path** (round-39): each MV is scored at its sub-pel-refined position AND the nearest integer-pel peer under the OBMC blend (1-ref) or SAD (bipred); whichever is lower wins per block, ties biased to int-pel. **Round-80**: a SECOND adaptive int-pel pass runs on the 1-ref path **after** `obmc_refine_me` converges (`inter_adaptive_int_pel_post_obmc`, default `true`) — lets blocks that drifted to a sub-pel offset during OBMC refinement snap back to integer-pel when the post-refinement neighbour grid no longer rewards the drift; strict-superset candidate-set invariant pinned by `inter_select_int_pel_monotonic_per_block_obmc_sse`. **Round-91**: bipred per-ref candidate set widened from `{sub-pel, int-pel}` to the strict superset `{int-pel, half-pel, sub-pel}` (3 per-ref candidates × 3×3 = 9 bipred combinations, de-duplicated when grids coincide); auditor-equivalent to the 1-ref strict-superset invariant; pinned by `bipred_widened_candidate_set_monotonic_per_block_sad`. On a half-pel-favourable camera-pan triplet (anchors 1 luma pel apart, midpoint at 0.5 luma pel → exact half-pel MV) the widened selector lifts ME-only Y self-roundtrip by +12.23 dB over the int-pel-only baseline (30.36 → 42.59 dB); the pre-existing camera-pan-by-1-pel ffmpeg cross-decode is unchanged (52.53 dB) since the SAD landscape was already converged on the qpel grid. Strict per-block min-of-{2,3}-SADs invariant — never regresses vs round-39. **Round-95**: 2-ref bipred path gains a **post-OBMC mode-only refinement pass** (`bipred_post_obmc_refine`, default `true`) — the bipred analogue of the 1-ref round-80 post-OBMC re-evaluation. After `bipred_select_modes` picks its `{mode, mv1, mv2}` decisions from the round-91 widened candidate set, each block's mode is re-scored under the full §15.8.5 OBMC blend with the neighbour grid frozen at the selector's output, choosing from the strict-superset trial set `{current, Ref1Only(mv1), Ref2Only(mv2), Ref1And2(mv1, mv2)}` at the same MV pair. Closes the cost-function gap between the selector's SAD-against-source metric and the decoder's OBMC-SSE-against-source reconstruction cost without disturbing the MV grid (so smooth-motion sub-pel choices are preserved). Lifts ME-only camera-pan bipred self-roundtrip from 51.44 → 52.24 dB (+0.80 dB); ffmpeg cross-decode on the camera-pan-by-1-pel fixture is unchanged at ~53 dB; per-block OBMC SSE monotonicity invariant pinned by `bipred_post_obmc_refine_monotonic_per_block_obmc_sse`. **§11.3 wavelet residue** (LeGall 5/3 / depth 3 / qindex 0 default; `ResidueParams`-configurable). At qindex=0 inter self-roundtrip is bit-exact (∞ dB) on every synthetic translation / camera-pan / complementary-bar fixture; setting `residue: None` reverts to the round-1 ZERO_RESIDUAL=true path for ME-only A/B comparison. |
| Encoder — mixed I+P ffmpeg interop  | Homogeneous core-syntax `0x0C` + `0x09` chain — ffmpeg accepts end-to-end (~52 dB intra Y cross-decoded; **+15 dB inter cross-decode uplift** from residue: 19.39 dB → 34.38 dB on `+4`-pel translating-square). 3-picture I+I+B chain (`0x0C`+`0x0C`+`0x0A`) — ffmpeg cross-decodes the bipred B at **~50 dB Y** on the sharp-edge complementary-bar fixture (per-block adaptive picks integer-pel; previously was 42.27 dB at unconditional qpel) and at **~52 dB Y** (+4.4 dB vs integer-only ceiling) on the smooth-motion camera-pan fixture (per-block adaptive picks qpel). |

## Codec ID

- `"dirac"`. Register with `oxideav_dirac::register(&mut codecs)`.

## Scope

- **Source-of-truth**: the BBC Dirac Specification v2.2.3 (2008), covering
  Dirac proper and VC-2 (its intra-only level-1 subset).
- **No third-party code** is consulted — the implementation is derived
  from the specification pseudocode only. FFmpeg is used solely as an
  opaque oracle to generate test bitstreams.

## Fuzz oracles

The crate ships three encoder/decoder fuzz oracles:

- **`tests/decoder_fuzz_oracle.rs`** (round-165) — decoder robustness
  against malformed inputs (truncation walk, byte mutation,
  pathological gibberish, oversized parse-info offsets).
- **`tests/encoder_rate_control_fuzz_oracle.rs`** (round-179) — intra
  encoder rate-control Cartesian sweep across all four LD/HQ variants
  (`PerPicture`/`Cbr`/`Vbv`/`VbvHysteresis`) × pathological targets +
  pixel inputs.
- **`tests/encoder_inter_fuzz_oracle.rs`** (round-193) — inter encoder
  parameter-surface sweep (precision × OBMC passes × search range +
  residue wavelet/depth/qindex + adaptive flags + pathological pixel
  inputs + same-frame zero-motion + extreme params). Pins no panic / no
  debug-assert / no integer overflow / no livelock + 2-frame round-trip
  on every accepted (`InterEncoderParams`, `InterInputPicture`,
  `InterInputPicture`) combination.

## Benchmarks

Round 190 adds a Criterion bench suite for the decode / encode /
roundtrip hot paths. Each bench synthesises a deterministic
64x64 4:2:0 YUV input via xorshift32 (no committed fixture files; no
third-party crate / binary in the timed region) and drives the
production `encode_single_hq_intra_stream` /
`encode_single_ld_intra_stream` + registry-backed decoder. Throughput
is reported in input pixels per second.

```
cargo bench -p oxideav-dirac --bench decode
cargo bench -p oxideav-dirac --bench encode
cargo bench -p oxideav-dirac --bench roundtrip
```

Three scenarios per binary: HQ intra qindex 0 (near-lossless), HQ
intra qindex 32 (lossy / short slice payloads), LD intra qindex 16
(fixed-rate, most-timing-stable). The matching IDs across the three
binaries (`hq_intra_64x64/qindex=0`, `hq_intra_64x64/qindex=32`,
`ld_intra_64x64/qindex=16`) let future rounds A/B encoder / decoder
algorithm tweaks against a stable baseline. Pairs with r165's
decoder fuzz oracle + r179's encoder rate-control fuzz oracle.

Round 195 extends each binary with a fourth scenario covering the
**DD9/7** wavelet (`hq_intra_64x64/qindex=0/wavelet=dd9_7`) — Dirac's
default filter (`wavelet_index = 0`), which the original three rows
omitted in favour of LeGall 5/3 only. DD9/7's second lifting step is
4-tap vs. LeGall's 2-tap, so this row's IDWT / forward-DWT cost is the
dominant per-frame work and is the right A/B fixture for future
profile-driven `vh_synth` / `vh_analysis` tweaks. Same round also
re-shapes `vh_synth` / `vh_analysis` to drive their row-major backing
slice directly (interleave + de-interleave drop their per-element
`SubbandData::set` calls; vertical-pass gather / scatter use raw
indexing into `data` so bounds-check elision applies). All 14 wavelet
unit tests + the 7-filter × depth-{1,2,3} roundtrip test + all ffmpeg
cross-decode interop tests stay green; decode q=32 row improved
~1.15% (`p < 0.05`), other rows within noise — bit-exactness preserved.

## License

MIT — see [LICENSE](LICENSE).
