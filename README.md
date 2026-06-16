# oxideav-dirac

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
core syntax, and the VC-2 v3 fragmented-picture path.

### Decode

| Area | State |
|------|-------|
| Parse-info / data-unit framing | `BBCD` header, next/previous offsets, parse-code taxonomy + intra/inter/reference/AC/low-delay/v3 predicates |
| Bit reader + exp-Golomb | MSB-first reader; interleaved unsigned/signed exp-Golomb with EOF/iteration caps |
| Sequence header | Parse parameters + base video format + source overrides + §12.4.4 extended (asymmetric) transform parameters |
| Predefined video formats | Full table (indices 0-20): frame size, chroma, range, etc. |
| Arithmetic decoder | Binary multi-context engine + probability LUT (Annex B) |
| Wavelet transforms | Inverse DWT (all 7 filters) + §13.3 intra DC prediction + §15.4 asymmetric (horizontal-only) IDWT (`idwt_with_ho`) |
| VC-2 / Dirac intra **decode** | Full coefficient decode, bit-exact on the intra-only docs-corpus fixtures (LD + core-syntax, 4:2:0 + 4:2:2, depth 3 + 4). The §12.4.4 asymmetric transform decodes end-to-end (custom and Annex D default quant matrices) |
| Dirac core-syntax inter **decode** | OBMC motion compensation + §11.3 residue; the two inter ReportOnly fixtures are bit-exact, interlaced LeGall-5,3 inter at ~99.7% pixel-exact |
| VC-2 v3 fragmented pictures | §14.2 fragment headers, §14.3/§14.4 reassembly, §14.5 trailing DC kick, and the `FragmentedPictureDecoder` driver — bit-exact-equivalent to the non-fragmented path on LD and HQ, including the asymmetric transform |

### Encode

| Area | State |
|------|-------|
| VC-2 HQ intra | 8/10-bit, 4:2:0/4:2:2/4:4:4, 6 wavelets, all spec-allowed `dwt_depth` (1..=5), optional custom quant matrix, per-slice / picture-level rate control (PerPicture / CBR / VBV / VBV-hysteresis), asymmetric transform emission, slice-prefix bytes |
| VC-2 LD intra | Same axis + rate-control coverage as HQ; bit-exact at q=0 against the reference oracle |
| Dirac core-syntax intra | AC-coded (`0x0C`) + VLC (`0x4C`) encoders, near-lossless at q=0, optional §11.3.3 spatial-partition codeblock grid with per-codeblock differential quantiser |
| Dirac inter | 1-ref P (`0x09`) and 2-ref bipred B (`0x0A`); sub-pel ME (qpel default), OBMC-aware ME refinement, per-block reference-mode + adaptive sub-pel/integer-pel selection, §11.3 wavelet residue (bit-exact self-roundtrip at q=0), inter-residue rate-control qindex picker, **multi-picture inter sequence driver** (HQ intra anchor + N `0x09` pictures, with PerPicture / Cbr / leaky-bucket Vbv / drain-limited VbvHysteresis residue-byte rate control — the full four-variant set the HQ/LD intra drivers carry) |
| Mixed I+P/B interop | Homogeneous core-syntax chains accepted end-to-end by the external oracle |

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
  flags, pathological pixels).
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
