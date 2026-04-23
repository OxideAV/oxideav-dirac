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
| Encoder                             | Not implemented                                               |

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
