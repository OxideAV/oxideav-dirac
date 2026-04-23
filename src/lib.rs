//! Pure-Rust Dirac / VC-2 video codec foundation.
//!
//! Dirac is a wavelet-based video codec developed by the BBC, specified
//! in the BBC "Dirac Specification" v2.2.3 (2008). SMPTE VC-2 is its
//! intra-only subset (profile 2, level 1). Both rely on:
//!
//! * A byte-aligned parse-info framing scheme (`BBCD` prefix + 1-byte
//!   parse code + next/previous offsets). See [`parse_info`].
//! * An MSB-first bit reader with interleaved exp-Golomb VLCs. See
//!   [`bits`].
//! * For sequence metadata: an Annex C table of base video formats plus
//!   per-field overrides. See [`sequence`] and [`video_format`].
//! * A custom binary arithmetic coder (Annex B) for core-syntax
//!   pictures. See [`arith`].
//!
//! This crate currently implements the **foundation layer**: parse-info
//! framing, sequence headers, the bit reader, exp-Golomb codes, and the
//! arithmetic decoding engine. Picture coefficient decoding (wavelet
//! transforms, codeblock unpacking, motion compensation) is **future
//! work** — picture data units currently return `Error::Unsupported`.
//!
//! Feeding bytes into the decoder and reading back the most recently
//! parsed sequence header already lets higher layers (probe, job graph)
//! recognise and describe Dirac streams.

#![allow(clippy::needless_range_loop)]

pub mod arith;
pub mod bits;
pub mod decoder;
pub mod parse_info;
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
    let caps = CodecCapabilities::video("dirac_sw")
        .with_lossy(true)
        .with_intra_only(false)
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
