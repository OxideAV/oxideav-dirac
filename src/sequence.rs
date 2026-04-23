//! Sequence header (§10).
//!
//! A Dirac sequence header carries four logical blocks:
//!
//! 1. Parse parameters  — version, profile, level.
//! 2. Base video format — index into Annex C's predefined tables.
//! 3. Source parameters — per-field overrides on top of the base defaults.
//! 4. Picture coding mode — 0 = frames, 1 = fields.
//!
//! After parsing, `set_coding_parameters` derives the luma/chroma
//! picture dimensions and the luma/chroma sample bit depths from the
//! effective signal range (§10.5).

use crate::bits::BitReader;
use crate::video_format::{
    preset_frame_rate, preset_pixel_aspect_ratio, BaseVideoFormat, ChromaFormat, ScanFormat,
    SignalRange,
};

/// Parsed parse parameters (§10.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseParameters {
    pub version_major: u32,
    pub version_minor: u32,
    pub profile: u32,
    pub level: u32,
}

/// Aggregated source video parameters: the Annex C defaults with any
/// per-field overrides applied. Field names mirror the spec's
/// `video_params` map labels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoParams {
    pub frame_width: u32,
    pub frame_height: u32,
    pub chroma_format: ChromaFormat,
    pub source_sampling: ScanFormat,
    pub top_field_first: bool,
    pub frame_rate_numer: u32,
    pub frame_rate_denom: u32,
    pub pixel_aspect_ratio_numer: u32,
    pub pixel_aspect_ratio_denom: u32,
    pub clean_width: u32,
    pub clean_height: u32,
    pub clean_left_offset: u32,
    pub clean_top_offset: u32,
    pub signal_range: SignalRange,
}

impl From<BaseVideoFormat> for VideoParams {
    fn from(b: BaseVideoFormat) -> Self {
        Self {
            frame_width: b.frame_width,
            frame_height: b.frame_height,
            chroma_format: b.chroma_format,
            source_sampling: b.source_sampling,
            top_field_first: b.top_field_first,
            frame_rate_numer: b.frame_rate_numer,
            frame_rate_denom: b.frame_rate_denom,
            pixel_aspect_ratio_numer: b.pixel_aspect_ratio_numer,
            pixel_aspect_ratio_denom: b.pixel_aspect_ratio_denom,
            clean_width: b.clean_width,
            clean_height: b.clean_height,
            clean_left_offset: b.clean_left_offset,
            clean_top_offset: b.clean_top_offset,
            signal_range: b.signal_range,
        }
    }
}

/// Picture coding mode (§10.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PictureCodingMode {
    Frames,
    Fields,
}

impl PictureCodingMode {
    pub fn from_index(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Frames),
            1 => Some(Self::Fields),
            _ => None,
        }
    }
}

/// A fully parsed sequence header plus the derived decoder state from
/// `set_coding_parameters` (§10.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceHeader {
    pub parse_parameters: ParseParameters,
    pub base_video_format_index: u32,
    pub video_params: VideoParams,
    pub picture_coding_mode: PictureCodingMode,
    pub luma_width: u32,
    pub luma_height: u32,
    pub chroma_width: u32,
    pub chroma_height: u32,
    pub luma_depth: u32,
    pub chroma_depth: u32,
}

/// Error variants produced while parsing a sequence header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// `base_video_format` index is outside the table we know.
    UnknownBaseVideoFormat(u32),
    /// `chroma_format_index` wasn't one of 0 (4:4:4), 1 (4:2:2) or 2 (4:2:0).
    UnknownChromaFormat(u32),
    /// `source_sampling` was neither 0 (progressive) nor 1 (interlaced).
    UnknownScanFormat(u32),
    /// Preset frame-rate / aspect-ratio / signal-range index out of range.
    PresetOutOfRange {
        which: &'static str,
        index: u32,
    },
    /// `picture_coding_mode` was not 0 or 1.
    UnknownPictureCodingMode(u32),
}

impl core::fmt::Display for ParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnknownBaseVideoFormat(i) => write!(f, "unknown base_video_format {i}"),
            Self::UnknownChromaFormat(i) => write!(f, "unknown chroma_format_index {i}"),
            Self::UnknownScanFormat(i) => write!(f, "unknown source_sampling {i}"),
            Self::PresetOutOfRange { which, index } => {
                write!(f, "{which} preset index {index} out of range")
            }
            Self::UnknownPictureCodingMode(i) => write!(f, "unknown picture_coding_mode {i}"),
        }
    }
}

impl std::error::Error for ParseError {}

/// Parse an entire sequence header from `data` (not including the
/// leading parse info). The slice should start at the sequence-header
/// payload; any trailing bytes belonging to subsequent data units are
/// ignored.
pub fn parse_sequence_header(data: &[u8]) -> Result<SequenceHeader, ParseError> {
    let mut r = BitReader::new(data);

    // §10: byte_align() then parse_parameters(), base_video_format,
    // source_parameters, picture_coding_mode, set_coding_parameters.
    r.byte_align();

    let parse_parameters = ParseParameters {
        version_major: r.read_uint(),
        version_minor: r.read_uint(),
        profile: r.read_uint(),
        level: r.read_uint(),
    };

    let base_video_format_index = r.read_uint();
    let base = BaseVideoFormat::lookup(base_video_format_index)
        .ok_or(ParseError::UnknownBaseVideoFormat(base_video_format_index))?;

    let mut vp: VideoParams = base.into();

    // §10.3.2 frame_size
    if r.read_bool() {
        vp.frame_width = r.read_uint();
        vp.frame_height = r.read_uint();
    }

    // §10.3.3 chroma_sampling_format
    if r.read_bool() {
        let idx = r.read_uint();
        vp.chroma_format =
            ChromaFormat::from_index(idx).ok_or(ParseError::UnknownChromaFormat(idx))?;
    }

    // §10.3.4 scan_format (note: TOP_FIELD_FIRST cannot be overridden)
    if r.read_bool() {
        let idx = r.read_uint();
        vp.source_sampling = match idx {
            0 => ScanFormat::Progressive,
            1 => ScanFormat::Interlaced,
            other => return Err(ParseError::UnknownScanFormat(other)),
        };
    }

    // §10.3.5 frame_rate
    if r.read_bool() {
        let idx = r.read_uint();
        if idx == 0 {
            vp.frame_rate_numer = r.read_uint();
            vp.frame_rate_denom = r.read_uint();
        } else {
            let (n, d) = preset_frame_rate(idx).ok_or(ParseError::PresetOutOfRange {
                which: "frame_rate",
                index: idx,
            })?;
            vp.frame_rate_numer = n;
            vp.frame_rate_denom = d;
        }
    }

    // §10.3.6 pixel_aspect_ratio
    if r.read_bool() {
        let idx = r.read_uint();
        if idx == 0 {
            vp.pixel_aspect_ratio_numer = r.read_uint();
            vp.pixel_aspect_ratio_denom = r.read_uint();
        } else {
            let (n, d) = preset_pixel_aspect_ratio(idx).ok_or(ParseError::PresetOutOfRange {
                which: "pixel_aspect_ratio",
                index: idx,
            })?;
            vp.pixel_aspect_ratio_numer = n;
            vp.pixel_aspect_ratio_denom = d;
        }
    }

    // §10.3.7 clean_area
    if r.read_bool() {
        vp.clean_width = r.read_uint();
        vp.clean_height = r.read_uint();
        vp.clean_left_offset = r.read_uint();
        vp.clean_top_offset = r.read_uint();
    }

    // §10.3.8 signal_range
    if r.read_bool() {
        let idx = r.read_uint();
        if idx == 0 {
            vp.signal_range = SignalRange {
                luma_offset: r.read_uint(),
                luma_excursion: r.read_uint(),
                chroma_offset: r.read_uint(),
                chroma_excursion: r.read_uint(),
            };
        } else {
            vp.signal_range = SignalRange::preset(idx).ok_or(ParseError::PresetOutOfRange {
                which: "signal_range",
                index: idx,
            })?;
        }
    }

    // §10.3.9 colour_spec — we parse-and-forget; this crate doesn't
    // need the primaries / matrix / transfer to decode the bitstream,
    // and the enclosing muxer usually carries them too.
    if r.read_bool() {
        let idx = r.read_uint();
        // Custom: may override primaries / matrix / transfer individually.
        if idx == 0 {
            if r.read_bool() {
                let _ = r.read_uint(); // colour primaries
            }
            if r.read_bool() {
                let _ = r.read_uint(); // colour matrix
            }
            if r.read_bool() {
                let _ = r.read_uint(); // transfer function
            }
        }
    }

    // §10.4 picture_coding_mode
    let pcm_idx = r.read_uint();
    let picture_coding_mode = PictureCodingMode::from_index(pcm_idx)
        .ok_or(ParseError::UnknownPictureCodingMode(pcm_idx))?;

    // §10.5.1 picture_dimensions
    let luma_width = vp.frame_width;
    let mut luma_height = vp.frame_height;
    let mut chroma_width = luma_width;
    let mut chroma_height = luma_height;
    match vp.chroma_format {
        ChromaFormat::Yuv444 => {}
        ChromaFormat::Yuv422 => {
            chroma_width /= 2;
        }
        ChromaFormat::Yuv420 => {
            chroma_width /= 2;
            chroma_height /= 2;
        }
    }
    if picture_coding_mode == PictureCodingMode::Fields {
        luma_height /= 2;
        chroma_height /= 2;
    }

    // §10.5.2 video_depth: intlog2(excursion + 1).
    let luma_depth = intlog2_ceil(vp.signal_range.luma_excursion + 1);
    let chroma_depth = intlog2_ceil(vp.signal_range.chroma_excursion + 1);

    Ok(SequenceHeader {
        parse_parameters,
        base_video_format_index,
        video_params: vp,
        picture_coding_mode,
        luma_width,
        luma_height,
        chroma_width,
        chroma_height,
        luma_depth,
        chroma_depth,
    })
}

/// Dirac `intlog2`: smallest `m` with `2^(m-1) < n <= 2^m`. For n == 0
/// returns 0. The spec only calls this with `n > 0`.
fn intlog2_ceil(n: u32) -> u32 {
    if n <= 1 {
        return 0;
    }
    32 - (n - 1).leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the bit string for a minimal 4:2:0 sequence header with
    /// everything at defaults: version 2/2, profile 2, level 1, base
    /// format = 4 (CIF), every custom flag False, picture_coding_mode
    /// = 0. The exp-Golomb encoding of the small values all begin
    /// with the appropriate interleaved patterns.
    fn encode_simple_seq_header() -> Vec<u8> {
        let mut bits = String::new();
        // uint 2 -> "011"
        bits.push_str("011"); // version major 2
        bits.push_str("011"); // version minor 2
        bits.push_str("011"); // profile 2
        bits.push_str("001"); // level 1
        bits.push_str("00011"); // base_video_format 4 (CIF)
                                // all 8 "custom_*_flag" booleans False (frame_size, chroma,
                                // scan, frame_rate, par, clean, signal_range, colour_spec).
        bits.push_str("00000000");
        bits.push_str("1"); // picture_coding_mode = 0
                            // pad to byte
        while bits.len() % 8 != 0 {
            bits.push('0');
        }
        let mut out = Vec::new();
        for chunk in bits.as_bytes().chunks(8) {
            let mut b = 0u8;
            for c in chunk {
                b = (b << 1) | if *c == b'1' { 1 } else { 0 };
            }
            out.push(b);
        }
        out
    }

    #[test]
    fn cif_progressive_420_frames() {
        let data = encode_simple_seq_header();
        let sh = parse_sequence_header(&data).unwrap();
        assert_eq!(sh.parse_parameters.version_major, 2);
        assert_eq!(sh.parse_parameters.version_minor, 2);
        assert_eq!(sh.parse_parameters.profile, 2);
        assert_eq!(sh.parse_parameters.level, 1);
        assert_eq!(sh.base_video_format_index, 4);
        assert_eq!(sh.video_params.frame_width, 352);
        assert_eq!(sh.video_params.frame_height, 288);
        assert_eq!(sh.video_params.chroma_format, ChromaFormat::Yuv420);
        assert_eq!(sh.picture_coding_mode, PictureCodingMode::Frames);
        assert_eq!(sh.luma_width, 352);
        assert_eq!(sh.luma_height, 288);
        assert_eq!(sh.chroma_width, 176);
        assert_eq!(sh.chroma_height, 144);
        assert_eq!(sh.luma_depth, 8); // 255+1 -> intlog2 = 8
        assert_eq!(sh.chroma_depth, 8);
    }

    #[test]
    fn unknown_base_format_rejected() {
        // base format 30 (unknown) — encoded as 0000111101 in exp-Golomb
        // is too complex to hand-craft; use uint 99 which should
        // definitely fail.
        // For this test, just patch the CIF bytes to set base format
        // to an impossible value via a crafted stream.
        // Easier: parse from a buffer whose base_video_format ends
        // up being 30. uint 30 in interleaved exp-Golomb is 7 bits:
        // the binary for 31 is 11111, which is 5 bits, so K=4:
        // pattern: 0 1 0 1 0 1 0 1 1 1 -> follow/data pairs. Not
        // trivial; instead just confirm the error path is reachable
        // by piggy-backing on the CIF test above and manually constructing
        // a uint that decodes to an unknown index.
        let mut bits = String::new();
        bits.push_str("011"); // ver major 2
        bits.push_str("011"); // ver minor 2
        bits.push_str("011"); // profile 2
        bits.push_str("001"); // level 1
                              // uint for value 30: N+1 = 31 = 11111 -> K=4
                              // sequence: 0 x3 0 x2 0 x1 0 x0 1 = 0 1 0 1 0 1 0 1 1
        bits.push_str("010101011");
        while bits.len() % 8 != 0 {
            bits.push('0');
        }
        let mut out = Vec::new();
        for chunk in bits.as_bytes().chunks(8) {
            let mut b = 0u8;
            for c in chunk {
                b = (b << 1) | if *c == b'1' { 1 } else { 0 };
            }
            out.push(b);
        }
        let err = parse_sequence_header(&out).unwrap_err();
        assert!(matches!(err, ParseError::UnknownBaseVideoFormat(30)));
    }

    #[test]
    fn intlog2_values() {
        assert_eq!(intlog2_ceil(256), 8); // 255+1
        assert_eq!(intlog2_ceil(1024), 10); // 1023+1
        assert_eq!(intlog2_ceil(1), 0);
        assert_eq!(intlog2_ceil(2), 1);
        assert_eq!(intlog2_ceil(3), 2);
    }
}
