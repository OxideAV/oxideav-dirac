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
//!
//! Still unsupported (planned): inter pictures + OBMC motion
//! compensation (parsing scaffold exists), v3 extended transform
//! parameters (horizontal-only transforms).

#![allow(clippy::needless_range_loop)]

pub mod arith;
pub mod bits;
pub mod decoder;
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

use oxideav_codec::{CodecInfo, CodecRegistry};
use oxideav_core::{CodecCapabilities, CodecId, CodecTag};

/// Canonical oxideav codec id.
pub const CODEC_ID_STR: &str = "dirac";

/// Register the Dirac decoder (foundation only) with a codec registry.
pub fn register(reg: &mut CodecRegistry) {
    // intra_only = true reflects the current decoder's reach (VC-2 LD
    // and HQ intra). Flip back to false once inter pictures with
    // motion compensation land.
    let caps = CodecCapabilities::video("dirac_sw")
        .with_lossy(true)
        .with_intra_only(true)
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
