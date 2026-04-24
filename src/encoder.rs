//! VC-2 HQ-profile intra-only encoder.
//!
//! This mirrors the decoder path in [`crate::picture`] for the HQ
//! profile (parse codes `0xE8` — non-reference — and `0xEC` —
//! reference), writing a self-contained bitstream that the decoder in
//! the same crate reads back pixel-for-pixel identically at qindex 0.
//!
//! The encoder produces a byte-aligned elementary stream:
//!
//! ```text
//!   [parse info 13B]  sequence header
//!   [parse info 13B]  HQ intra picture
//!   [parse info 13B]  end of sequence
//! ```
//!
//! Stage breakdown per picture:
//!
//! 1. Pad each component up to a multiple of `2^dwt_depth` and subtract
//!    the `2^(depth-1)` midpoint to centre around zero (§15.10 inverse).
//! 2. Forward DWT (§15.6 analysis, see [`crate::wavelet::dwt`]).
//! 3. Quantise each subband with a per-(level, orientation) quantiser
//!    derived from the slice `qindex` and the default quantisation
//!    matrix (§13.3.1, §13.5.4).
//! 4. Pack per-slice: `slice_prefix_bytes` zero bytes, a 1-byte qindex,
//!    then for each of Y / U / V a length byte followed by a bounded
//!    exp-Golomb coefficient block whose byte-length is the length byte
//!    times `slice_size_scaler`.
//!
//! Only 8-bit 4:2:0 intra pictures with LeGall 5/3 + depth 3 are
//! emitted today — enough to round-trip a 64x64 testsrc through the
//! crate's own decoder.

use crate::bitwriter::BitWriter;
use crate::parse_info::BBCD;
use crate::quant::{slice_quantisers, QuantMatrix};
use crate::sequence::SequenceHeader;
use crate::subband::{padded_component_dims, subband_dims, Orient, SubbandData};
use crate::video_format::ChromaFormat;
use crate::wavelet::{dwt, WaveletFilter};

/// Parameters that drive picture encoding. Mirrors the subset of
/// `TransformParameters` the HQ-profile encoder actually needs.
#[derive(Debug, Clone)]
pub struct EncoderParams {
    pub wavelet: WaveletFilter,
    pub dwt_depth: u32,
    pub slices_x: u32,
    pub slices_y: u32,
    pub slice_prefix_bytes: u32,
    pub slice_size_scaler: u32,
    pub quant_matrix: QuantMatrix,
    /// Per-slice quantisation index (0..=127). 0 means lossless-ish —
    /// qf=4, offset=1, resulting in near-identity inverse quant for
    /// coefficients small compared to 4.
    pub qindex: u32,
}

impl EncoderParams {
    /// Default HQ intra parameters: LeGall 5/3 at depth 3, 4x4 slices,
    /// no prefix bytes, scaler 1, qindex 0. Good for the round-trip
    /// smoke test on small pictures.
    pub fn default_hq(wavelet: WaveletFilter, dwt_depth: u32) -> Self {
        let quant_matrix = QuantMatrix::default_for(wavelet, dwt_depth)
            .expect("default quant matrix exists for depth <= 4");
        Self {
            wavelet,
            dwt_depth,
            slices_x: 8,
            slices_y: 8,
            slice_prefix_bytes: 0,
            slice_size_scaler: 1,
            quant_matrix,
            qindex: 0,
        }
    }
}

/// Encode a single intra HQ picture. `y`, `u`, `v` hold one byte per
/// sample in row-major order, sized exactly to the component's luma /
/// chroma dimensions from `sequence`.
pub fn encode_hq_intra_picture(
    sequence: &SequenceHeader,
    params: &EncoderParams,
    picture_number: u32,
    y: &[u8],
    u: &[u8],
    v: &[u8],
) -> Vec<u8> {
    let mut w = BitWriter::new();

    // §12.2 picture_header: byte-align then 4-byte picture_number.
    w.byte_align();
    w.write_uint_lit(4, picture_number);

    // §12.3 wavelet_transform — for intra there's no zero_residual.
    w.byte_align();
    write_transform_parameters(&mut w, params);
    w.byte_align();

    // Per-component coefficient pyramids.
    let luma_w = sequence.luma_width;
    let luma_h = sequence.luma_height;
    let chroma_w = sequence.chroma_width;
    let chroma_h = sequence.chroma_height;

    let y_py = forward_component(y, luma_w, luma_h, sequence.luma_depth, params);
    let u_py = forward_component(u, chroma_w, chroma_h, sequence.chroma_depth, params);
    let v_py = forward_component(v, chroma_w, chroma_h, sequence.chroma_depth, params);

    // Quantise in place.
    let q_per_level = slice_quantisers(params.qindex, &params.quant_matrix);
    let y_qpy = quantise_pyramid(&y_py, &q_per_level);
    let u_qpy = quantise_pyramid(&u_py, &q_per_level);
    let v_qpy = quantise_pyramid(&v_py, &q_per_level);

    // Precompute per-level subband sizes for each component.
    let mut luma_dims: Vec<(usize, usize)> =
        Vec::with_capacity(params.dwt_depth as usize + 1);
    let mut chroma_dims: Vec<(usize, usize)> =
        Vec::with_capacity(params.dwt_depth as usize + 1);
    for level in 0..=params.dwt_depth {
        luma_dims.push(subband_dims(luma_w, luma_h, params.dwt_depth, level));
        chroma_dims.push(subband_dims(chroma_w, chroma_h, params.dwt_depth, level));
    }

    for sy in 0..params.slices_y {
        for sx in 0..params.slices_x {
            encode_hq_slice(
                &mut w,
                params,
                sx,
                sy,
                &y_qpy,
                &u_qpy,
                &v_qpy,
                &luma_dims,
                &chroma_dims,
            );
        }
    }

    w.finish()
}

fn write_transform_parameters(w: &mut BitWriter, params: &EncoderParams) {
    // wavelet_index, dwt_depth, slice parameters, quant matrix flag.
    w.write_uint(wavelet_index(params.wavelet));
    w.write_uint(params.dwt_depth);
    // §12.4.5.2 slice_parameters — HQ branch.
    w.write_uint(params.slices_x);
    w.write_uint(params.slices_y);
    w.write_uint(params.slice_prefix_bytes);
    w.write_uint(params.slice_size_scaler);
    // Default quant matrix — emit flag=0 so the decoder looks up
    // the per-filter table. If callers pass a custom matrix we could
    // emit flag=1 + the explicit values; for now the encoder always
    // uses defaults.
    w.write_bool(false);
}

fn wavelet_index(filter: WaveletFilter) -> u32 {
    match filter {
        WaveletFilter::DeslauriersDubuc9_7 => 0,
        WaveletFilter::LeGall5_3 => 1,
        WaveletFilter::DeslauriersDubuc13_7 => 2,
        WaveletFilter::Haar0 => 3,
        WaveletFilter::Haar1 => 4,
        WaveletFilter::Fidelity => 5,
        WaveletFilter::Daubechies9_7 => 6,
    }
}

/// Pad one component up to a multiple of `2^dwt_depth`, subtract the
/// depth midpoint, run the forward DWT and return the pyramid.
fn forward_component(
    plane: &[u8],
    comp_w: u32,
    comp_h: u32,
    depth: u32,
    params: &EncoderParams,
) -> Vec<[SubbandData; 4]> {
    let (pw, ph) = padded_component_dims(comp_w, comp_h, params.dwt_depth);
    let half: i32 = 1i32 << (depth - 1);
    let mut pic = SubbandData::new(pw, ph);
    for y in 0..ph {
        for x in 0..pw {
            let src_x = x.min(comp_w as usize - 1);
            let src_y = y.min(comp_h as usize - 1);
            let v = plane[src_y * comp_w as usize + src_x] as i32 - half;
            pic.set(y, x, v);
        }
    }
    dwt(&pic, params.wavelet, params.dwt_depth)
}

/// Quantise every coefficient `x` to `sign(x) * (|x| / qf)`. Level 0
/// uses only the LL entry; higher levels use HL/LH/HH.
fn quantise_pyramid(
    pyramid: &[[SubbandData; 4]],
    q_per_level: &[[u32; 4]],
) -> Vec<[SubbandData; 4]> {
    let mut out: Vec<[SubbandData; 4]> = Vec::with_capacity(pyramid.len());
    for (level, bands) in pyramid.iter().enumerate() {
        let mut level_out: [SubbandData; 4] = [
            SubbandData::new(bands[0].width, bands[0].height),
            SubbandData::new(bands[1].width, bands[1].height),
            SubbandData::new(bands[2].width, bands[2].height),
            SubbandData::new(bands[3].width, bands[3].height),
        ];
        for orient_idx in 0..4 {
            let src = &bands[orient_idx];
            if src.width == 0 || src.height == 0 {
                continue;
            }
            let q = q_per_level[level][orient_idx];
            let dst = &mut level_out[orient_idx];
            for y in 0..src.height {
                for x in 0..src.width {
                    let v = src.get(y, x);
                    dst.set(y, x, quantise_coeff(v, q));
                }
            }
        }
        out.push(level_out);
    }
    out
}

/// Forward quantisation (§13.3.1 inverse). The spec's inverse quant
/// formula is `x = sign(q) * (|q| * qf + offset + 2) / 4`. A matching
/// forward map that's exactly recovered when the qindex keeps qf small
/// is `q = sign(x) * ((|x| * 4) / qf)`. Rounding is floor-of-magnitude
/// to preserve the sign + zero-reconstruction invariants of the
/// dead-zone scheme.
fn quantise_coeff(x: i32, q: u32) -> i32 {
    if x == 0 {
        return 0;
    }
    let qf = crate::quant::quant_factor(q) as i64;
    // 4 * |x| / qf rounds magnitude toward zero; this is the standard
    // Dirac "dead-zone" forward quantiser. At q=0 (qf=4) it's a
    // no-op: 4*|x|/4 = |x|, and inverse_quant reconstructs |x| exactly.
    let mag = (x.unsigned_abs() as i64 * 4) / qf;
    if x < 0 {
        -(mag as i32)
    } else {
        mag as i32
    }
}

/// Slice-area coordinates within a subband (§13.5.6.2). Mirrors the
/// decoder's version.
fn slice_bounds(
    sx: u32,
    sy: u32,
    slices_x: u32,
    slices_y: u32,
    sub_w: usize,
    sub_h: usize,
) -> (usize, usize, usize, usize) {
    let left = (sub_w * sx as usize) / slices_x as usize;
    let right = (sub_w * (sx as usize + 1)) / slices_x as usize;
    let top = (sub_h * sy as usize) / slices_y as usize;
    let bottom = (sub_h * (sy as usize + 1)) / slices_y as usize;
    (left, right, top, bottom)
}

/// Encode one HQ slice (§13.5.4):
/// `slice_prefix_bytes | qindex | (len_y, Y-coeffs) | (len_u, U-coeffs) | (len_v, V-coeffs)`.
#[allow(clippy::too_many_arguments)]
fn encode_hq_slice(
    w: &mut BitWriter,
    params: &EncoderParams,
    sx: u32,
    sy: u32,
    y_qpy: &[[SubbandData; 4]],
    u_qpy: &[[SubbandData; 4]],
    v_qpy: &[[SubbandData; 4]],
    luma_dims: &[(usize, usize)],
    chroma_dims: &[(usize, usize)],
) {
    // Emit the slice-prefix zero bytes (application-defined).
    w.byte_align();
    for _ in 0..params.slice_prefix_bytes {
        w.write_uint_lit(1, 0);
    }
    // qindex is 1 byte.
    w.write_uint_lit(1, params.qindex);

    // Per-component: emit a 1-byte length and the coefficient stream,
    // padded out to `length * slice_size_scaler` bytes.
    for comp in 0..3 {
        let dims: &[(usize, usize)] = if comp == 0 { luma_dims } else { chroma_dims };
        let qpy: &[[SubbandData; 4]] = match comp {
            0 => y_qpy,
            1 => u_qpy,
            _ => v_qpy,
        };
        let comp_bytes = encode_hq_component(params, qpy, dims, sx, sy);
        let scaler = params.slice_size_scaler.max(1) as usize;
        // Round length up to a multiple of scaler; emit the byte
        // length divided by scaler as the length prefix.
        let padded_len = comp_bytes.len().div_ceil(scaler) * scaler;
        let length_byte = padded_len / scaler;
        debug_assert!(
            length_byte <= 255,
            "length byte overflow: {length_byte} bytes for slice ({sx},{sy}) comp {comp} — bump slice_size_scaler or use more slices",
        );
        let length_byte = length_byte as u32;
        w.write_uint_lit(1, length_byte);
        w.byte_align();
        w.write_bytes(&comp_bytes);
        // Pad to padded_len.
        for _ in comp_bytes.len()..padded_len {
            w.write_uint_lit(1, 0);
        }
    }
}

/// Serialise one component's coefficients for a single slice to a
/// fresh byte buffer, using bounded interleaved exp-Golomb + byte
/// alignment at the end.
fn encode_hq_component(
    params: &EncoderParams,
    qpy: &[[SubbandData; 4]],
    dims: &[(usize, usize)],
    sx: u32,
    sy: u32,
) -> Vec<u8> {
    let mut w = BitWriter::new();
    // Level 0 — LL only.
    write_slice_band(&mut w, 0, Orient::LL, sx, sy, params, qpy, dims);
    for level in 1..=params.dwt_depth {
        for orient in [Orient::HL, Orient::LH, Orient::HH] {
            write_slice_band(&mut w, level, orient, sx, sy, params, qpy, dims);
        }
    }
    w.byte_align();
    w.finish()
}

#[allow(clippy::too_many_arguments)]
fn write_slice_band(
    w: &mut BitWriter,
    level: u32,
    orient: Orient,
    sx: u32,
    sy: u32,
    params: &EncoderParams,
    qpy: &[[SubbandData; 4]],
    dims: &[(usize, usize)],
) {
    let (sub_w, sub_h) = dims[level as usize];
    let (left, right, top, bottom) =
        slice_bounds(sx, sy, params.slices_x, params.slices_y, sub_w, sub_h);
    let band = &qpy[level as usize][orient.as_index()];
    if band.width == 0 || band.height == 0 {
        return;
    }
    for y in top..bottom {
        for x in left..right {
            let v = band.get(y, x);
            w.write_sint(v);
        }
    }
}

// -----------------------------------------------------------------
//   Sequence header, parse info, and full stream emit.
// -----------------------------------------------------------------

/// Emit a sequence header payload (not including the parse info).
/// Mirrors [`crate::sequence::parse_sequence_header`] on the read side.
pub fn encode_sequence_header(sequence: &SequenceHeader) -> Vec<u8> {
    let mut w = BitWriter::new();
    // parse_parameters (§10.1).
    w.write_uint(sequence.parse_parameters.version_major);
    w.write_uint(sequence.parse_parameters.version_minor);
    w.write_uint(sequence.parse_parameters.profile);
    w.write_uint(sequence.parse_parameters.level);
    // base_video_format index — we pick 0 (custom) and override the
    // fields we care about below.
    w.write_uint(sequence.base_video_format_index);

    // Custom frame size always on so we can reconstruct exact dims.
    w.write_bool(true);
    w.write_uint(sequence.video_params.frame_width);
    w.write_uint(sequence.video_params.frame_height);

    // Chroma sampling format.
    w.write_bool(true);
    w.write_uint(sequence.video_params.chroma_format.to_index());

    // Scan format.
    w.write_bool(false); // rely on base default — progressive.
    // Frame rate: rely on default.
    w.write_bool(false);
    // Pixel aspect ratio: default.
    w.write_bool(false);
    // Clean area: default.
    w.write_bool(false);
    // Signal range: prefer a preset index when the params match (keeps
    // the header compact and matches what ffmpeg-compatible decoders
    // expect). Fall back to a fully custom (index 0) range when the
    // caller supplies something off-preset.
    w.write_bool(true);
    let sr = &sequence.video_params.signal_range;
    use crate::video_format::SignalRange;
    let preset_idx = if *sr == SignalRange::PRESET_8BIT_FULL {
        1
    } else if *sr == SignalRange::PRESET_8BIT_VIDEO {
        2
    } else if *sr == SignalRange::PRESET_10BIT_VIDEO {
        3
    } else if *sr == SignalRange::PRESET_12BIT_VIDEO {
        4
    } else {
        0
    };
    w.write_uint(preset_idx);
    if preset_idx == 0 {
        w.write_uint(sr.luma_offset);
        w.write_uint(sr.luma_excursion);
        w.write_uint(sr.chroma_offset);
        w.write_uint(sr.chroma_excursion);
    }
    // Colour spec: default.
    w.write_bool(false);

    // picture_coding_mode.
    let pcm_idx = match sequence.picture_coding_mode {
        crate::sequence::PictureCodingMode::Frames => 0,
        crate::sequence::PictureCodingMode::Fields => 1,
    };
    w.write_uint(pcm_idx);
    w.byte_align();
    w.finish()
}

/// Write a 13-byte parse info header into `out` at `out.len()`.
pub fn write_parse_info(
    out: &mut Vec<u8>,
    parse_code: u8,
    next_parse_offset: u32,
    previous_parse_offset: u32,
) {
    out.extend_from_slice(BBCD);
    out.push(parse_code);
    out.extend_from_slice(&next_parse_offset.to_be_bytes());
    out.extend_from_slice(&previous_parse_offset.to_be_bytes());
}

/// Encode one 64x64 (or other) 4:2:0 8-bit YUV frame as a complete
/// VC-2 HQ intra-only Dirac elementary stream, including sequence
/// header, a single non-reference HQ intra picture (parse code `0xE8`)
/// and an end-of-sequence marker.
pub fn encode_single_hq_intra_stream(
    sequence: &SequenceHeader,
    params: &EncoderParams,
    picture_number: u32,
    y: &[u8],
    u: &[u8],
    v: &[u8],
) -> Vec<u8> {
    // 1) Sequence header payload.
    let sh_payload = encode_sequence_header(sequence);
    // 2) Picture payload.
    let pic_payload =
        encode_hq_intra_picture(sequence, params, picture_number, y, u, v);

    // Lay out: [pi_sh][sh][pi_pic][pic][pi_eos].
    let pi_size = 13usize;
    let sh_unit_len = pi_size + sh_payload.len();
    let pic_unit_len = pi_size + pic_payload.len();
    // End-of-sequence parse info only.
    let eos_unit_len = pi_size;

    let mut out = Vec::with_capacity(sh_unit_len + pic_unit_len + eos_unit_len);
    write_parse_info(&mut out, 0x00, sh_unit_len as u32, 0);
    out.extend_from_slice(&sh_payload);
    write_parse_info(
        &mut out,
        0xE8, // HQ non-reference intra picture.
        pic_unit_len as u32,
        sh_unit_len as u32,
    );
    out.extend_from_slice(&pic_payload);
    write_parse_info(&mut out, 0x10, 0, pic_unit_len as u32);
    out
}

/// Build a minimal sequence header describing an `WxH` progressive
/// 4:2:0 8-bit stream (or 4:4:4 / 4:2:2 depending on `chroma`).
pub fn make_minimal_sequence(
    frame_width: u32,
    frame_height: u32,
    chroma: ChromaFormat,
) -> SequenceHeader {
    use crate::sequence::{ParseParameters, PictureCodingMode, VideoParams};
    use crate::video_format::{ScanFormat, SignalRange};
    let signal_range = SignalRange::PRESET_8BIT_FULL;
    let vp = VideoParams {
        frame_width,
        frame_height,
        chroma_format: chroma,
        source_sampling: ScanFormat::Progressive,
        top_field_first: false,
        frame_rate_numer: 25,
        frame_rate_denom: 1,
        pixel_aspect_ratio_numer: 1,
        pixel_aspect_ratio_denom: 1,
        clean_width: frame_width,
        clean_height: frame_height,
        clean_left_offset: 0,
        clean_top_offset: 0,
        signal_range,
    };
    let (chroma_w, chroma_h) = match chroma {
        ChromaFormat::Yuv444 => (frame_width, frame_height),
        ChromaFormat::Yuv422 => (frame_width / 2, frame_height),
        ChromaFormat::Yuv420 => (frame_width / 2, frame_height / 2),
    };
    SequenceHeader {
        parse_parameters: ParseParameters {
            version_major: 2,
            version_minor: 0,
            profile: 3, // VC-2 high quality
            level: 3,   // VC-2 sub-sampled HQ profile level
        },
        base_video_format_index: 0,
        video_params: vp,
        picture_coding_mode: PictureCodingMode::Frames,
        luma_width: frame_width,
        luma_height: frame_height,
        chroma_width: chroma_w,
        chroma_height: chroma_h,
        luma_depth: 8,
        chroma_depth: 8,
    }
}

/// Build a synthetic 64x64 4:2:0 YUV test pattern: a diagonal luma
/// gradient with a cross + a tinted chroma. Useful for the encoder
/// round-trip tests below.
pub fn synthetic_testsrc_64_yuv420() -> ([u8; 64 * 64], [u8; 32 * 32], [u8; 32 * 32]) {
    let mut y = [0u8; 64 * 64];
    let mut u = [0u8; 32 * 32];
    let mut v = [0u8; 32 * 32];
    for row in 0..64 {
        for col in 0..64 {
            let base = ((row + col) * 2) as u8;
            let pix = if row == 32 || col == 32 { 255u8 } else { base };
            y[row * 64 + col] = pix;
        }
    }
    for row in 0..32 {
        for col in 0..32 {
            u[row * 32 + col] = (col * 4) as u8;
            v[row * 32 + col] = (row * 4) as u8;
        }
    }
    (y, u, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitReader;
    use crate::sequence::parse_sequence_header;

    #[test]
    fn sequence_header_roundtrip_420() {
        let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
        let bytes = encode_sequence_header(&seq);
        let parsed = parse_sequence_header(&bytes).expect("parse");
        assert_eq!(parsed.video_params.frame_width, 64);
        assert_eq!(parsed.video_params.frame_height, 64);
        assert_eq!(parsed.video_params.chroma_format, ChromaFormat::Yuv420);
        assert_eq!(parsed.luma_width, 64);
        assert_eq!(parsed.chroma_width, 32);
        assert_eq!(parsed.luma_depth, 8);
        assert_eq!(parsed.chroma_depth, 8);
        assert_eq!(
            parsed.picture_coding_mode,
            crate::sequence::PictureCodingMode::Frames
        );
    }

    #[test]
    fn sequence_header_roundtrip_444() {
        let seq = make_minimal_sequence(32, 32, ChromaFormat::Yuv444);
        let bytes = encode_sequence_header(&seq);
        let parsed = parse_sequence_header(&bytes).expect("parse");
        assert_eq!(parsed.video_params.chroma_format, ChromaFormat::Yuv444);
        assert_eq!(parsed.chroma_width, 32);
        assert_eq!(parsed.chroma_height, 32);
    }

    #[test]
    fn parse_info_header_layout() {
        let mut out = Vec::new();
        write_parse_info(&mut out, 0xE8, 100, 13);
        assert_eq!(&out[..4], BBCD);
        assert_eq!(out[4], 0xE8);
        assert_eq!(&out[5..9], &100u32.to_be_bytes());
        assert_eq!(&out[9..13], &13u32.to_be_bytes());
    }

    #[test]
    fn quantise_identity_at_q0() {
        // At qindex 0 (qf=4), forward quant = x * 4 / 4 = x. The
        // inverse then recovers x exactly (inverse_quant * q=0).
        for &x in &[-17i32, -1, 0, 1, 5, 42, 255] {
            assert_eq!(quantise_coeff(x, 0), x);
            assert_eq!(crate::quant::inverse_quant(quantise_coeff(x, 0), 0), x);
        }
    }

    /// Smoke-test: the emitted stream should start with BBCD + a
    /// sequence header parse code, followed by an HQ intra parse code.
    #[test]
    fn emitted_stream_has_expected_parse_codes() {
        let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
        let params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
        let (y, u, v) = synthetic_testsrc_64_yuv420();
        let stream = encode_single_hq_intra_stream(&seq, &params, 0, &y, &u, &v);
        // First parse info.
        assert_eq!(&stream[..4], BBCD);
        assert_eq!(stream[4], 0x00);
        let sh_next = u32::from_be_bytes([stream[5], stream[6], stream[7], stream[8]]) as usize;
        assert!(sh_next > 13);
        // Next parse info must be at `sh_next`.
        assert_eq!(&stream[sh_next..sh_next + 4], BBCD);
        assert_eq!(stream[sh_next + 4], 0xE8);
    }

    /// The transform_parameters block we emit should parse back with
    /// the same values.
    #[test]
    fn transform_parameters_roundtrip() {
        let params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
        let mut w = BitWriter::new();
        write_transform_parameters(&mut w, &params);
        w.byte_align();
        let bytes = w.finish();

        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_uint(), 1); // LeGall index
        assert_eq!(r.read_uint(), 3); // dwt_depth
        assert_eq!(r.read_uint(), 8); // slices_x
        assert_eq!(r.read_uint(), 8); // slices_y
        assert_eq!(r.read_uint(), 0); // slice_prefix_bytes
        assert_eq!(r.read_uint(), 1); // slice_size_scaler
        assert!(!r.read_bool()); // custom quant flag off
    }
}
