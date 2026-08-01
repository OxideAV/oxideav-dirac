//! Pure-Rust Dirac / VC-2 video codec.
//!
//! Dirac is a wavelet-based video codec developed by the BBC, specified
//! in the BBC "Dirac Specification" v2.2.3 (2008). SMPTE ST 2042-1
//! "VC-2" extends that syntax with the **Low Delay (LD)** and
//! **High Quality (HQ)** intra-only profiles used in broadcast
//! contribution links.
//!
//! Both profiles share:
//!
//! * A byte-aligned parse-info framing scheme (`BBCD` prefix + 1-byte
//!   parse code + next/previous offsets). See [`parse_info`].
//! * An MSB-first bit reader with interleaved exp-Golomb VLCs. See
//!   [`bits`].
//! * A sequence header with an Annex C preset table plus per-field
//!   overrides. See [`sequence`] and [`video_format`].
//! * A binary arithmetic coder (Annex B) — used only by core-syntax
//!   pictures. See [`arith`].
//! * An inverse discrete wavelet transform with seven lifting-based
//!   filters (Deslauriers-Dubuc, LeGall, Haar, Fidelity, Daubechies).
//!   See [`wavelet`].
//! * VC-2 inverse quantisation and default per-filter quantisation
//!   matrices (Annex E.1). See [`quant`].
//!
//! What works today:
//!
//! * **VC-2 LD intra pictures** (parse codes 0xC8 / 0xCC) — full
//!   coefficient unpack, intra DC prediction, IDWT, output offset.
//! * **VC-2 HQ intra pictures** (parse codes 0xE8 / 0xEC) — same,
//!   minus the DC prediction. A multi-frame 128x96 oracle-generated test-source
//!   clip decodes pixel-for-pixel-identically to the oracle.
//! * **Core-syntax intra pictures** (parse code 0x08 / 0x0C) — §13.4
//!   per-subband codeblock unpacking, with both the VLC and the
//!   arithmetic-coded paths (zero-parent / zero-neighbourhood /
//!   sign-prediction contexts per Table 13.1). End-to-end testing
//!   relies on a third-party Dirac encoder; the oracle only emits VC-2.
//! * **Core-syntax inter pictures** — §11.2 picture prediction
//!   parameters, §12.3 block motion data decode (superblock splits,
//!   block modes, reference-1 / reference-2 motion vectors, DC values
//!   for intra blocks, with spatial predictions from §12.3.6), and
//!   §15.8 overlapped block motion compensation (8-tap half-pel
//!   interpolation, bilinear sub-pel up to 1/8, the ramp spatial
//!   weighting window, affine/perspective global motion, and bi-
//!   directional reference weighting). The reference-picture buffer
//!   is maintained across pictures by the decoder front-end. As of
//!   writing, the oracle does not emit Dirac inter (only VC-2 intra) so
//!   end-to-end testing of this path relies on primitives-level unit
//!   tests until a third-party Dirac encoder is available.
//!
//! Output plumbing: the decoder front-end picks an oxideav-core
//! [`PixelFormat`](oxideav_core::PixelFormat) from the sequence
//! header's chroma format + luma bit depth
//! ([`decoder::output_format_for`]). 8-bit streams emit
//! `Yuv420P / Yuv422P / Yuv444P`; 9-10-bit streams emit
//! `Yuv*P10Le` (two bytes per sample, little-endian); 11-12-bit
//! streams emit `Yuv*P12Le`; deeper streams (13-16-bit — §10.3.8
//! custom signal ranges above 12 bits per component) emit the
//! all-bits-significant `Yuv*P16Le` deep-colour surface, MSB-aligned
//! for sub-16-bit depths. Frame `time_base` is derived from §10.3.5
//! `frame_rate_numer / frame_rate_denom`, and an incoming packet's
//! `pts` is carried through to the decoded frame (falling back to
//! the §12.2 picture_number when absent).
//!
//! Encoder coverage:
//!
//! * **VC-2 HQ intra** ([`encoder::encode_single_hq_intra_stream`]) —
//!   the bit-exact oracle-interop baseline (≥48 dB Y PSNR at q=0
//!   across LeGall / DD9-7 / DD13-7 / Haar / Fidelity / Daubechies).
//! * **VC-2 LD intra** ([`encoder::encode_single_ld_intra_stream`]) —
//!   oracle-validated with the Round 9 `slice_y_length` width fix.
//! * **Dirac core-syntax inter** ([`encoder_inter::encode_intra_then_inter_stream`]
//!   for 1-ref P, [`encoder_inter::encode_bipred_inter_picture`] +
//!   [`encoder_intra_core::encode_core_intra_then_bipred_stream`] for
//!   2-ref bipred B) — 1-ref non-reference inter (parse code `0x09`)
//!   and 2-ref bipred B (parse code `0x0A`) over integer-pel full-search
//!   SAD ME with per-level 8-neighbor sub-pel refinement (configurable
//!   `mv_precision`; quarter-pel is the default), preset-1 8x8 blocks,
//!   **§15.8.6 OBMC-aware ME refinement** (#186) that converges the
//!   per-block MV grid on the same weighted-sum reconstruction the
//!   decoder will perform, **per-block bipred decision search**
//!   ([`encoder_inter::bipred_select_modes`]) that picks `Ref1Only` /
//!   `Ref2Only` / `Ref1And2` per block by SAD against the source, and
//!   **§11.3 wavelet residue** (default LeGall 5/3 / depth 3 /
//!   qindex 0; configurable via [`encoder_inter::ResidueParams`]). The
//!   residue closes the prediction-error loop: at the default
//!   `qindex = 0` the inter self-roundtrip is bit-exact (∞ dB) on every
//!   synthetic translation fixture in the test suite (1-ref and 2-ref
//!   alike), and the homogeneous-profile oracle cross-decode lands at
//!   ~34 dB (1-ref `+4`-pel translating-square, +15 dB over no-residue)
//!   and **~42 dB** (bipred B on the complementary-bar fixture). Setting
//!   `residue: None` reverts to the round-1 ZERO_RESIDUAL=true behaviour
//!   for direct ME-only A/B comparison. Driven by the
//!   [`arith::ArithEncoder`] (Annex B.2 mirror of `ArithDecoder`).
//! * **Dirac core-syntax intra** ([`encoder_intra_core::encode_single_core_intra_stream`]
//!   and [`encoder_intra_core::encode_core_intra_then_inter_stream`])
//!   — round 2: AC-coded intra reference picture (parse code `0x0C`),
//!   single codeblock per subband, no per-codeblock quant offset, no
//!   custom quant matrix. Self-roundtrip is bit-exact on flat
//!   pictures and ≥48 dB Y/U on a testsrc gradient. Pairs with the
//!   round-1 inter encoder for a homogeneous-syntax 2-frame stream
//!   that the oracle's `dirac` decoder accepts end-to-end (cross-decoded
//!   intra Y PSNR ≈ 52 dB). A **VLC (non-arithmetic) variant**
//!   ([`encoder_intra_core::encode_core_intra_picture_vlc`] /
//!   [`encoder_intra_core::encode_single_core_intra_stream_vlc`],
//!   parse code `0x4C`) emits the same picture with §13.4.2.2 plain
//!   exp-Golomb entropy instead of the arithmetic coder — the encoder
//!   counterpart to the decoder's `decode_subband_vlc`. It shares the
//!   AC codeblock/skip/quant-offset walk and applies no entropy-coder
//!   rounding, so at `qindex = 0` it is strictly lossless (bit-exact on
//!   the testsrc V-plane gradient where the AC path keeps a ~1-LSB
//!   roughness).
//!
//! Inter-residue rate control: the §11.3 wavelet-residue qindex for the
//! 1-ref inter path can be picked against a residue-payload byte budget
//! via [`encoder_inter::pick_inter_residue_qindex`] (and the
//! `(qindex, actual_bytes)` companion
//! [`encoder_inter::inter_residue_qindex_diagnostic`]) — the inter
//! analogue of the HQ/LD intra picture-qindex pickers. The picker runs
//! the same motion estimation the emitter commits, reconstructs the OBMC
//! prediction + forward-transforms the residue once, then walks
//! `qindex ∈ floor..=127` for the smallest quantiser whose serialised
//! residue stream fits the budget.
//!
//! Inter sequence rate control: a multi-picture inter sequence driver
//! ([`encoder_inter::encode_inter_sequence_with_residue_target`] and its
//! `_report` companion) wires the per-picture
//! [`encoder_inter::pick_inter_residue_qindex`] picker across an HQ intra
//! anchor (`0xEC`) plus N 1-ref inter pictures (`0x09`) with a
//! [`encoder_inter::InterRateControl::PerPicture`] /
//! [`encoder_inter::InterRateControl::Cbr`] /
//! [`encoder_inter::InterRateControl::Vbv`] (leaky-bucket) /
//! [`encoder_inter::InterRateControl::VbvHysteresis`] (drain-rate-limited
//! leaky-bucket) **residue-byte** accumulator — the inter analogue of the
//! HQ/LD intra `encode_*_sequence_with_size_target` drivers, now with the
//! same four rate-control variants. The accumulator carries
//! `Σ(actual − target)` residue bytes between pictures; under the two VBV
//! variants the savings end of the accumulator is clamped at
//! `-buffer_bytes` so a run of undershooting inter pictures cannot bank
//! unbounded headroom (peak residue-size cap). Every inter picture
//! references the intra anchor (the stream's only reference picture, since
//! `0x09` pictures are non-reference) so the whole sequence round-trips
//! through [`decoder::DiracDecoder`].
//!
//! Inter-residue spatial partition: the §11.3 wavelet-residue path now
//! supports the §11.3.3 codeblock grid (an optional
//! [`encoder_inter::ResidueParams::codeblocks`] per-level
//! `(codeblocks_x, codeblocks_y)` plus
//! [`encoder_inter::ResidueParams::codeblock_mode`]), the inter-residue
//! analogue of the core-intra encoder's spatial partition. Each
//! HL/LH/HH subband splits into a grid of codeblocks carrying a
//! §13.4.3.3 `ZERO_BLOCK` skip flag and, under `codeblock_mode == 1`, a
//! §13.4.3.4 differential quantiser offset; the emitter is a
//! byte-for-byte mirror of the proven core-intra codeblock encoder (the
//! decoder reads it through the shared `picture_core::decode_subband`
//! walk). With reversible LeGall 5/3 at `qindex = 0` the residue
//! round-trips bit-exactly for every codeblock geometry — including
//! sub-4×4-sample and 1×1-sample codeblocks, whose final AC symbols
//! used to land on the flawed §B.2.7.1 terminator tail before the
//! round-382 `ArithEncoder::finish()` fix.

#![allow(clippy::needless_range_loop)]

// The stable public surface is intentionally small: the decode entry
// points (`decoder`), the registry hooks (`register`, `register_codecs`,
// `CODEC_ID_STR`), and the sequence/video-format types those signatures
// expose. Every other module below is internal wavelet / OBMC / entropy /
// encoder / fragment plumbing that is only `pub` so the crate's own
// tests, benches, fuzz oracles and examples can drive it directly; it is
// marked `#[doc(hidden)]` so cargo-semver-checks does not treat it as
// part of the stable API. `#[doc(hidden)]` changes documentation and
// semver visibility only — it is not a visibility change, so all callers
// keep compiling.

// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod arith;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod bits;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod bitwriter;
pub mod decoder;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod encoder;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod encoder_inter;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod encoder_intra_core;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod fragment;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod obmc;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod parse_info;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod picture;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod picture_core;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod picture_inter;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod quant;
// Stays visible: `SequenceHeader` appears in the stable
// `DiracDecoder::last_sequence` signature. Internal items inside are
// hidden individually — cargo-semver-checks does not credit re-exports
// out of a `#[doc(hidden)]` module.
pub mod sequence;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod stream;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod subband;
pub(crate) mod trace;
// Stays visible: `ChromaFormat` / `ScanFormat` / `SignalRange` are
// reachable through the stable `SequenceHeader`. Internal items inside
// are hidden individually.
pub mod video_format;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod wavelet;

use oxideav_core::{CodecCapabilities, CodecId, CodecTag, Confidence, ProbeContext};
use oxideav_core::{CodecInfo, CodecRegistry, RuntimeContext};

/// Canonical oxideav codec id.
pub const CODEC_ID_STR: &str = "dirac";

/// Probe confidence when the packet/header bytes carry core-syntax
/// (long-GOP era) picture evidence — inter pictures or arithmetic/VLC
/// core-syntax intra. This is the syntax family only this codec id
/// implements, so the claim is near-certain. Kept below `1.0` so an
/// exact-match future claim could still outrank it, and well above the
/// deliberately weak long-GOP claim a VC-2-profile registration makes
/// on the same shared legacy tags.
#[doc(hidden)]
pub const PROBE_CONFIDENCE_LONG_GOP: Confidence = 0.95;

/// Probe confidence when no packet or header bytes are available at
/// resolution time (the common stream-discovery case — the Matroska
/// registration for this codec carries no CodecPrivate at all). The
/// legacy container codes (`V_DIRAC`, `drac`, ObjectType `0xA4`) are
/// registered to Dirac itself, so absent contrary evidence this crate
/// is the default owner — but the middling value leaves room for a
/// claim with real bitstream evidence to win.
#[doc(hidden)]
pub const PROBE_CONFIDENCE_NO_EVIDENCE: Confidence = 0.5;

/// Probe confidence when the bytes show only intra-profile (LD / HQ /
/// fragmented) pictures — the VC-2 flavour of the shared syntax. This
/// crate decodes those streams end-to-end, so the claim stays nonzero
/// (it must win when no dedicated VC-2 registration is present), but
/// it is deliberately weak so a dedicated VC-2-profile registration
/// outranks it on its home turf.
#[doc(hidden)]
pub const PROBE_CONFIDENCE_INTRA_PROFILE: Confidence = 0.3;

/// Upper bound on parse-info hops the probe will follow. A probe runs
/// per candidate registration at stream-discovery time; evidence, if
/// present at all, is in the first few data units.
const PROBE_MAX_UNITS: usize = 64;

/// Confidence probe for the shared legacy container tags.
///
/// Matroska `V_DIRAC`, the MP4 sample entry `drac` and MP4
/// ObjectTypeIndication `0xA4` are registered container codes for
/// Dirac; VC-2 (the intra-only profile family of the same bitstream
/// syntax) has no container codes of its own and rides under them.
/// Both this crate and a dedicated VC-2-profile crate may therefore
/// claim the same tags; the registry resolves the contest by probe
/// strength:
///
/// * core-syntax picture evidence (parse-code bit 7 clear, bit 3 set —
///   inter pictures and AC/VLC core-syntax intra, the long-GOP era
///   syntax) → [`PROBE_CONFIDENCE_LONG_GOP`]: this crate wins;
/// * only intra-profile pictures (LD / HQ / fragments, parse-code bit
///   7 set) → [`PROBE_CONFIDENCE_INTRA_PROFILE`]: a dedicated VC-2
///   registration outranks us, but we still claim the stream when it
///   is absent;
/// * no packet/header bytes, or units without pictures →
///   [`PROBE_CONFIDENCE_NO_EVIDENCE`]: legacy-owner default;
/// * bytes present but no `BBCD` parse-info structure at all → `0.0`
///   (not this bitstream syntax).
// internal — exposed for tests; registered via `CodecInfo::probe`
#[doc(hidden)]
pub fn container_tag_probe(ctx: &ProbeContext) -> Confidence {
    // Prefer real packet bytes; fall back to the container-level
    // stream-format blob (some carriage formats copy the head of the
    // elementary stream there). Empty blobs count as "no evidence":
    // the Matroska registration explicitly has no initialization data.
    let data = match (ctx.packet, ctx.header) {
        (Some(p), _) if !p.is_empty() => p,
        (_, Some(h)) if !h.is_empty() => h,
        _ => return PROBE_CONFIDENCE_NO_EVIDENCE,
    };

    let mut pos = match parse_info::ParseInfo::find_next(data, 0) {
        Some(p) => p,
        None => return 0.0,
    };
    let mut saw_unit = false;
    let mut saw_intra_profile_picture = false;
    for _ in 0..PROBE_MAX_UNITS {
        let Some(pi) = parse_info::ParseInfo::parse(data, pos) else {
            break;
        };
        saw_unit = true;
        if pi.is_picture() {
            if pi.is_core_syntax() {
                // Long-GOP era syntax — decisive.
                return PROBE_CONFIDENCE_LONG_GOP;
            }
            saw_intra_profile_picture = true;
        }
        // Advance: trust a sane next_parse_offset, otherwise scan for
        // the next prefix (offsets may be zero on the last unit or on
        // streams written without back-patching).
        let next = pi.next_parse_offset as usize;
        pos = if next >= parse_info::ParseInfo::SIZE && pos + next <= data.len() {
            pos + next
        } else {
            match parse_info::ParseInfo::find_next(data, pos + 1) {
                Some(p) => p,
                None => break,
            }
        };
    }
    if saw_intra_profile_picture {
        PROBE_CONFIDENCE_INTRA_PROFILE
    } else if saw_unit {
        PROBE_CONFIDENCE_NO_EVIDENCE
    } else {
        0.0
    }
}

/// Register the Dirac decoder with a codec registry.
pub fn register_codecs(reg: &mut CodecRegistry) {
    // Core-syntax inter pictures with OBMC motion compensation are
    // implemented, so we no longer advertise `intra_only`.
    let caps = CodecCapabilities::video("dirac_sw")
        .with_lossy(true)
        .with_max_size(7680, 4320);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(decoder::make_decoder)
            // The registered legacy container codes for this bitstream
            // syntax; VC-2 has none of its own and rides under them
            // (the Matroska registry entry says so explicitly, and the
            // MP4RA registry has no VC-2 sample entry). Raw elementary
            // streams carry no tag; the container code is what the
            // registry matches on. The probe resolves the co-claim
            // contest with a VC-2-profile registration by evidence.
            .tags([
                CodecTag::fourcc(b"drac"),
                CodecTag::matroska("V_DIRAC"),
                CodecTag::mp4_object_type(0xA4),
            ])
            .probe(container_tag_probe),
    );
}

/// Unified registration entry point: install the Dirac codec factories
/// into the codec sub-registry of a [`RuntimeContext`].
///
/// This is the preferred entry point for new code — it matches the
/// convention every sibling crate now follows. Direct callers that need
/// only the codec sub-registry can keep using [`register_codecs`].
///
/// Also wired into [`oxideav_meta::register_all`] via the
/// [`oxideav_core::register!`] macro below.
pub fn register(ctx: &mut RuntimeContext) {
    register_codecs(&mut ctx.codecs);
}

oxideav_core::register!("dirac", register);

#[cfg(test)]
mod register_tests {
    use super::*;
    use oxideav_core::{CodecId, CodecParameters, RuntimeContext};

    #[test]
    fn register_via_runtime_context_installs_codec_factory() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let dec = ctx
            .codecs
            .first_decoder(&params)
            .expect("dirac decoder factory");
        assert_eq!(dec.codec_id().as_str(), CODEC_ID_STR);
    }
}
