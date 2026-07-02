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
//!   minus the DC prediction. A multi-frame 128x96 ffmpeg testsrc
//!   clip decodes pixel-for-pixel-identically to ffmpeg.
//! * **Core-syntax intra pictures** (parse code 0x08 / 0x0C) — §13.4
//!   per-subband codeblock unpacking, with both the VLC and the
//!   arithmetic-coded paths (zero-parent / zero-neighbourhood /
//!   sign-prediction contexts per Table 13.1). End-to-end testing
//!   relies on a third-party Dirac encoder; ffmpeg only emits VC-2.
//! * **Core-syntax inter pictures** — §11.2 picture prediction
//!   parameters, §12.3 block motion data decode (superblock splits,
//!   block modes, reference-1 / reference-2 motion vectors, DC values
//!   for intra blocks, with spatial predictions from §12.3.6), and
//!   §15.8 overlapped block motion compensation (8-tap half-pel
//!   interpolation, bilinear sub-pel up to 1/8, the ramp spatial
//!   weighting window, affine/perspective global motion, and bi-
//!   directional reference weighting). The reference-picture buffer
//!   is maintained across pictures by the decoder front-end. As of
//!   writing, ffmpeg does not emit Dirac inter (only VC-2 intra) so
//!   end-to-end testing of this path relies on primitives-level unit
//!   tests until a third-party Dirac encoder is available.
//!
//! Output plumbing: the decoder front-end picks an oxideav-core
//! [`PixelFormat`](oxideav_core::PixelFormat) from the sequence
//! header's chroma format + luma bit depth. 8-bit streams emit
//! `Yuv420P / Yuv422P / Yuv444P`; 9-10-bit streams emit
//! `Yuv*P10Le` (two bytes per sample, little-endian); 11-12-bit 4:2:0
//! emits `Yuv420P12Le`. Frame `time_base` is derived from §10.3.5
//! `frame_rate_numer / frame_rate_denom`, and an incoming packet's
//! `pts` is carried through to the decoded frame (falling back to
//! the §12.2 picture_number when absent).
//!
//! Encoder coverage:
//!
//! * **VC-2 HQ intra** ([`encoder::encode_single_hq_intra_stream`]) —
//!   the bit-exact ffmpeg interop baseline (≥48 dB Y PSNR at q=0
//!   across LeGall / DD9-7 / DD13-7 / Haar / Fidelity / Daubechies).
//! * **VC-2 LD intra** ([`encoder::encode_single_ld_intra_stream`]) —
//!   ffmpeg-validated with the Round 9 `slice_y_length` width fix.
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
//!   alike), and the homogeneous-profile ffmpeg cross-decode lands at
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
//!   that ffmpeg's `dirac` decoder accepts end-to-end (cross-decoded
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

pub mod arith;
pub mod bits;
pub mod bitwriter;
pub mod decoder;
pub mod encoder;
pub mod encoder_inter;
pub mod encoder_intra_core;
pub mod fragment;
pub mod obmc;
pub mod parse_info;
pub mod picture;
pub mod picture_core;
pub mod picture_inter;
pub mod quant;
pub mod sequence;
pub mod stream;
pub mod subband;
pub mod video_format;
pub mod wavelet;

use oxideav_core::{CodecCapabilities, CodecId, CodecTag};
use oxideav_core::{CodecInfo, CodecRegistry, RuntimeContext};

/// Canonical oxideav codec id.
pub const CODEC_ID_STR: &str = "dirac";

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
            // Dirac-in-MP4 / MKV / MXF uses the `drac` sample entry
            // FourCC. Raw elementary streams carry no FourCC on their
            // own; the container tag is what the registry matches on.
            .tag(CodecTag::fourcc(b"drac")),
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
