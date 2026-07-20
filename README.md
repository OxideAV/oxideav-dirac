# oxideav-dirac

[![CI](https://github.com/OxideAV/oxideav-dirac/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-dirac/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-dirac.svg)](https://crates.io/crates/oxideav-dirac) [![docs.rs](https://docs.rs/oxideav-dirac/badge.svg)](https://docs.rs/oxideav-dirac) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust codec for **Dirac** (BBC wavelet video codec) and its
intra-only subset **SMPTE VC-2**. Implemented clean-room from the BBC
"Dirac Specification" (v2.2.3, 2008) and SMPTE ST 2042-1:2022. Zero C
dependencies.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Status

Dirac is a large codec (wavelet transforms + custom arithmetic coder +
motion compensation). This crate covers the full intra path bit-exactly,
a working inter path, encoders for both VC-2 profiles and the Dirac
core syntax, and the VC-2 v3 fragmented-picture path. The docs
fixture corpus (12 fixtures, 8-bit through 16-bit deep colour) decodes
12/12 bit-exact.

### Decode

| Area | State |
|------|-------|
| Parse-info / data-unit framing | `BBCD` header, next/previous offsets, parse-code taxonomy + intra/inter/reference/AC/low-delay/v3 predicates |
| Bit reader + exp-Golomb | MSB-first reader; interleaved unsigned/signed exp-Golomb with EOF/iteration caps |
| Sequence header | Parse parameters + base video format + source overrides + §12.4.4 extended (asymmetric) transform parameters |
| Predefined video formats | Full table (indices 0-20): frame size, chroma, range, etc. |
| Arithmetic coder | Binary multi-context engine + probability LUT (Annex B); the encoder's §B.2.7.1 terminator is exact as of round-382 (a spurious follow bit used to corrupt the final symbols of tight-interval AC blocks — the source of every historical "AC-terminator roughness"), so AC-coded streams round-trip bit-exactly at `qindex = 0` |
| Wavelet transforms | Inverse DWT (all 7 filters) + §13.3 intra DC prediction + §15.4 asymmetric (horizontal-only) IDWT (`idwt_with_ho`) |
| VC-2 / Dirac intra **decode** | Full coefficient decode, bit-exact on the intra-only docs-corpus fixtures (LD + core-syntax, 4:2:0 + 4:2:2, depth 3 + 4). The §12.4.4 asymmetric transform decodes end-to-end (custom and Annex D default quant matrices) |
| High-bit-depth / deep-colour intra **decode** | §10.5.2 `video_depth`-parameterised reconstruction proven end-to-end at 10-bit (HQ, all 3 chromas × 6 reversible wavelets; all `dwt_depth` 1..=5; §12.4.4 asymmetric transform), 12-bit (HQ, all 3 chromas — native `Yuv*P12Le` since round-417) and **13-16-bit deep colour** (HQ, 16-bit across all 3 chromas × 6 wavelets, 13/14/15-bit MSB-aligned — all bit-exact) via full-range `&[u16]` encode → decode round-trips. Output surfaces: `Yuv*P` / `Yuv*P10Le` / `Yuv*P12Le` / all-bits-significant `Yuv*P16Le` per `decoder::output_format_for`; `DiracDecoder::output_pixel_format` + an exact-header `receive_arena_frame` expose the choice to callers |
| Dirac core-syntax inter **decode** | OBMC motion compensation + §11.3 residue — round-419 proves the `video_depth`-parameterised inter reconstruction end-to-end at **10-16-bit deep colour** (self-encoded P/B chains cross-decode bit-exactly, incl. codeblock-grid residue and global motion) — **reference-exact on every inter fixture in the docs corpus** (integer-pel `i-then-p` / `i-p-b`, and since round-408 the quarter-pel interlaced LeGall-5,3 fixture as well). Round-408 pinned §15.8.10's out-of-frame edge extension by black-box probe bitstreams: clamp the *integer-pel part* of the half-pel coordinate, keep the half-pel fraction (the pseudocode's raw coordinate clip drops it, ±1..3 LSB at overspilled edges). The `edge-mc-probes-320x240` corpus fixture guards the rule at every MV precision on all four edges; §15.8.10/§15.8.11 sub-pel primitives are also pinned against hand-computed spec values |
| Decode tracing | Env-gated `DIRAC_TRACE` / `DIRAC_TRACE_FILE` instrumentation emitting the docs trace-contract event vocabulary — `PARSE_UNIT`, `SEQUENCE`, `PICTURE`, `MOTION`, `MOTION_BLOCK`, `MOTION_GLOBAL`, and per-block `MOTION_MV` lines (final MV + predictor + residual + DC per reference), plus the opt-in `DIRAC_TRACE_MC` `MC_PLANE` SHA-256 of the pre-residual OBMC prediction plane — so decode divergences can be localised to a block or split MC-vs-residual without external tooling |
| VC-2 v3 fragmented pictures | §14.2 fragment headers, §14.3/§14.4 reassembly, §14.5 trailing DC kick, and the `FragmentedPictureDecoder` driver — bit-exact-equivalent to the non-fragmented path on LD and HQ, including the asymmetric transform and (round-417) 16-bit deep-colour pictures |

### Encode

| Area | State |
|------|-------|
| VC-2 HQ intra | 8-16-bit (deeper samples via the `&[u16]` `encode_single_hq_intra_stream_u16` entry + `PRESET_{10,12,14,16}BIT_FULL` / custom §10.3.8 signal ranges — q0 bit-exact through 16-bit), 4:2:0/4:2:2/4:4:4, 6 wavelets, all spec-allowed `dwt_depth` (1..=5), optional custom quant matrix, per-slice / picture-level rate control (PerPicture / CBR / VBV / VBV-hysteresis), asymmetric transform emission, slice-prefix bytes |
| VC-2 LD intra | Same axis + rate-control coverage as HQ; bit-exact at q=0 against the reference oracle |
| Dirac core-syntax intra | AC-coded (`0x0C`) + VLC (`0x4C`) encoders, both bit-exact at q=0, optional §11.3.3 spatial-partition codeblock grid with per-codeblock differential quantiser |
| Dirac inter | 1-ref P (`0x09`) and 2-ref bipred B (`0x0A`); sub-pel ME (qpel default), OBMC-aware ME refinement, per-block reference-mode + adaptive sub-pel/integer-pel selection, §11.3 wavelet residue (bit-exact self-roundtrip at q=0) with optional §11.3.3 spatial-partition codeblock grid + per-codeblock differential quantiser (bit-exact for every codeblock geometry incl. 1×1-sample; byte-for-byte mirror of the core-intra codeblock encoder; wired on both the 1-ref and bipred paths), **§11.2.6 global motion** (affine-perspective `GlobalParams` per reference, §12.3.3.2 per-block global flags — whole-picture or per-block grids — global blocks carry no MV residual and predict via the §15.8.8 `global_mv` field; end-to-end on P + bipred B + the sequence driver, 120-case fuzz sweep, external-oracle bit-exact cross-decode), **global-model estimation from the ME grid** (round-386: `estimate_global_motion_config` with Pan / Affine / Perspective model families — trimmed 6-parameter affine least squares plus an alternating linearised perspective fit, quantised onto the integer §15.8.8 parameterisation with exact-field local refinement, per-block gmode by measured SAD; 2-model `estimate_global_bipred_config` for B-pictures with a both-refs AND rule; −31% stream bytes on a whole-frame zoom at bit-exact q=0; estimated pan cross-decodes bit-exact through the external oracle, which is itself characterised as pan-only — it applies the §15.8.8 field evaluated at (0,0) to every pixel, ignoring the per-pixel matrix term our decoder implements), inter-residue rate-control qindex picker, **multi-picture inter sequence driver** (HQ intra anchor + N `0x09` pictures, with PerPicture / Cbr / leaky-bucket Vbv / drain-limited VbvHysteresis residue-byte rate control — the full four-variant set the HQ/LD intra drivers carry — plus round-386 per-picture **auto global motion**: `auto_global_motion` estimates a model per picture and applies it only past a global-fraction threshold, resolved before the qindex picker so rate control measures the true stream; per-picture fraction/applied telemetry in the report). **Deep colour (round-419)**: the whole inter pipeline is sample-generic (`InterSample`, sealed `u8`/`u16`) — 10-16-bit `&[u16]` P + bipred B encode (HQ-anchored and homogeneous core-syntax chains), the §11.3.3 codeblock grid, §11.2.6 global motion (explicit, estimated, and per-picture auto) and **all four residue rate-control variants** hold at deep depths, q0 bit-exact end-to-end through the crate's own decoder on synthetic planes *and* on chains anchored to the r417 deep fixture trio (16-bit Cbr lands within ~2.4% of the per-picture residue target on the pinned fixture sequence) |
| Mixed I+P/B interop | Homogeneous core-syntax chains cross-decode through the external oracle **bit-exactly** since round-408: the encoder emits §11.2.2 block parameters as custom literals (the oracle resolves preset *index* 1 to non-overlapped blocks) and codes zero-residue pictures as explicit all-zero bands (the oracle mis-reconstructs `ZERO_RESIDUAL=1` skip pictures). **Deep inter is self-validated only** (round-419 characterisation): the oracle accepts 10/12-bit preset-range inter chains but corrupts its own output for both frames (zeroed mid-picture stripe on the intra it decodes bit-exactly stand-alone; near-garbage inter frame), while the identical bytes decode bit-exactly through this crate's decoder — and it rejects the §10.3.8 custom ranges deeper streams need, so 13-16-bit inter has no external ground truth at all |
| VC-2 v3 fragmented pictures (**encode**) | Round-386 §14 fragment **emitter**: `fragment_picture_payload` splits any conformant LD/HQ picture payload into a `[setup][data…]` fragment sequence (slice boundaries recovered from the payload itself — §13.5.4 HQ length-byte walk / §13.5.3.2 LD closed-form widths), plus `encode_single_{hq,ld}_intra_fragmented_stream` v3 stream drivers; bit-exact reassembly through the crate's own `FragmentedPictureDecoder` across every chunk geometry |

## Codec ID

`"dirac"`. Register with `oxideav_dirac::register(&mut codecs)`.

## Scope & clean-room provenance

- **Source of truth**: the BBC Dirac Specification v2.2.3 (2008),
  covering Dirac proper and VC-2 (its intra-only subset), plus
  SMPTE ST 2042-1:2022 for the VC-2 v3 fragmented-picture syntax.
- **No third-party code** is consulted — the implementation is derived
  from the specification pseudocode only. An external encoder is used
  solely as an opaque oracle to generate test bitstreams.

## Robustness & benchmarks

The crate ships four fuzz oracles under `tests/`:

- `decoder_fuzz_oracle.rs` — decoder robustness against truncation, byte
  mutation, gibberish, and oversized parse-info offsets.
- `encoder_rate_control_fuzz_oracle.rs` — Cartesian sweep across all
  four LD/HQ rate-control variants × pathological targets and inputs.
- `encoder_inter_fuzz_oracle.rs` — inter-encoder parameter-surface
  sweep (precision, OBMC passes, search range, residue config, adaptive
  flags, global-motion fields/grids, pathological pixels; round-419
  adds deep-colour `u16` arms at 10/13/16-bit incl. saturated
  full-range pathological inputs and a 16-bit bipred precision axis).
- `fragment_assembler_fuzz_oracle.rs` — VC-2 v3 §14 `FragmentAssembler`
  state machine driven by a seeded random walk against a reference
  model.

A Criterion suite covers the decode / encode / roundtrip hot paths on a
deterministic 64×64 4:2:0 input (no committed fixtures, no third-party
crate in the timed region):

```
cargo bench -p oxideav-dirac --bench decode
cargo bench -p oxideav-dirac --bench encode
cargo bench -p oxideav-dirac --bench roundtrip
```

## License

MIT — see [LICENSE](LICENSE).
