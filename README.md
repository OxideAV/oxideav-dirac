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
| Bit reader (MSB-first)              | Implemented — `read_bit`, `read_nbits`, `read_uint_lit`       |
| Exp-Golomb (interleaved)            | Unsigned + signed decoders (`read_uint` / `read_sint`)        |
| Sequence header                     | Parse parameters + base video format + source overrides       |
| Predefined video formats            | Full table (indices 0-20) with frame size, chroma, range etc. |
| Arithmetic decoder                  | Binary multi-context engine + probability LUT (Annex B)       |
| Wavelet transforms                  | Not yet wired up                                              |
| VC-2 low-delay intra                | Parse-only (byte framing; no coefficient decode)              |
| Dirac core-syntax intra / inter     | Parse-only                                                    |
| Encoder — VC-2 HQ intra             | Implemented (8/10-bit, 4:2:0/4:2:2/4:4:4, 6 wavelets)         |
| Encoder — VC-2 LD intra             | Implemented (ffmpeg-bit-exact at q=0)                         |
| Encoder — Dirac core-syntax intra (r2) | AC-coded `0x0C` ref, single codeblock, q=0 near-lossless. ffmpeg-validated. |
| Encoder — Dirac inter               | 1-ref P (`0x09`) **and 2-ref bipred B** (`0x0A`); sub-pel ME (qpel default for both P and B), OBMC-aware ME refinement (#186), per-block `Ref1Only` / `Ref2Only` / `Ref1And2` decision search for bipred, **per-block adaptive sub-pel-vs-integer-pel selection** (round-39: each MV is scored at its sub-pel-refined position AND at the nearest integer-pel peer; whichever gives lower SAD wins per block, ties biased toward zero), **§11.3 wavelet residue** (LeGall 5/3 / depth 3 / qindex 0 default; `ResidueParams`-configurable). At qindex=0 inter self-roundtrip is bit-exact (∞ dB) on every synthetic translation / camera-pan / complementary-bar fixture; setting `residue: None` reverts to the round-1 ZERO_RESIDUAL=true path for ME-only A/B comparison. |
| Encoder — mixed I+P ffmpeg interop  | Homogeneous core-syntax `0x0C` + `0x09` chain — ffmpeg accepts end-to-end (~52 dB intra Y cross-decoded; **+15 dB inter cross-decode uplift** from residue: 19.39 dB → 34.38 dB on `+4`-pel translating-square). 3-picture I+I+B chain (`0x0C`+`0x0C`+`0x0A`) — ffmpeg cross-decodes the bipred B at **~50 dB Y** on the sharp-edge complementary-bar fixture (per-block adaptive picks integer-pel; previously was 42.27 dB at unconditional qpel) and at **~52 dB Y** (+4.4 dB vs integer-only ceiling) on the smooth-motion camera-pan fixture (per-block adaptive picks qpel). |

## Codec ID

- `"dirac"`. Register with `oxideav_dirac::register(&mut codecs)`.

## Scope

- **Source-of-truth**: the BBC Dirac Specification v2.2.3 (2008), covering
  Dirac proper and VC-2 (its intra-only level-1 subset).
- **No third-party code** is consulted — the implementation is derived
  from the specification pseudocode only. FFmpeg is used solely as an
  opaque oracle to generate test bitstreams.

## License

MIT — see [LICENSE](LICENSE).
