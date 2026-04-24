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
    let mut luma_dims: Vec<(usize, usize)> = Vec::with_capacity(params.dwt_depth as usize + 1);
    let mut chroma_dims: Vec<(usize, usize)> = Vec::with_capacity(params.dwt_depth as usize + 1);
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
    let pic_payload = encode_hq_intra_picture(sequence, params, picture_number, y, u, v);

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

// -----------------------------------------------------------------
//   VC-2 LD (Low-Delay) profile — intra only.
//
//   LD slices carry a fixed byte budget derived from
//   `slice_bytes_numer / slice_bytes_denom` (§13.5.3.2). Each slice
//   layout is:
//     qindex                      — 7 raw bits
//     slice_y_length              — intlog2(total_bits - 7) raw bits
//     luma coefficients           — slice_y_length bits (Funnel-bounded
//                                   interleaved exp-Golomb)
//     chroma (U,V) coefficients   — remaining bits, interleaved per
//                                   coefficient position
//
//   On the decode side the Funnel returns `1` for any bit read past
//   the end of the budget, which terminates a bounded uintb read with
//   value `0`. The encoder mirrors that by right-padding each bounded
//   region with `1` bits — the decoder then reads a string of zeroes,
//   which is exactly what the encoder skipped emitting.
//
//   DC prediction (§13.4) is encoder-side subtractive: we subtract the
//   `mean3` of the neighbour (a, b, c) cells from each LL coefficient
//   before quantisation-serialisation. The decoder adds the same
//   predictor back in after inverse quantisation.
// -----------------------------------------------------------------

/// LD-profile parameters. Fixed per-slice byte budget (§13.5.3.2) is
/// `slice_bytes_numer / slice_bytes_denom` per slice, spread across all
/// `slices_x * slices_y` slices.
#[derive(Debug, Clone)]
pub struct LdEncoderParams {
    pub wavelet: WaveletFilter,
    pub dwt_depth: u32,
    pub slices_x: u32,
    pub slices_y: u32,
    /// Total-bytes numerator — actual bytes for slice (sx,sy) are
    /// distributed from this via `slice_bytes()`.
    pub slice_bytes_numer: u32,
    pub slice_bytes_denom: u32,
    pub quant_matrix: QuantMatrix,
    /// Per-slice qindex (0..=127). 0 is near-lossless for coefficients
    /// in the range where the dead-zone quantiser's forward rounding is
    /// exact.
    pub qindex: u32,
}

impl LdEncoderParams {
    /// Default LD parameters with a generous per-slice byte budget so
    /// coefficients fit without truncation on small round-trip test
    /// pictures. `bytes_per_slice` bytes per slice × `slices_x *
    /// slices_y` slices gives the total.
    pub fn default_ld(
        wavelet: WaveletFilter,
        dwt_depth: u32,
        slices_x: u32,
        slices_y: u32,
        bytes_per_slice: u32,
    ) -> Self {
        let quant_matrix = QuantMatrix::default_for(wavelet, dwt_depth)
            .expect("default quant matrix exists for depth <= 4");
        let total = bytes_per_slice * slices_x * slices_y;
        Self {
            wavelet,
            dwt_depth,
            slices_x,
            slices_y,
            slice_bytes_numer: total,
            slice_bytes_denom: slices_x * slices_y,
            quant_matrix,
            qindex: 0,
        }
    }
}

/// `intlog2(n)` matching the decoder-side helper in `picture.rs` —
/// §13.5.3.1 slice-length sizing.
fn intlog2(n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        32 - (n - 1).leading_zeros()
    }
}

/// `slice_bytes(sx, sy)` (§13.5.3.2). Mirrors the decoder helper.
fn slice_bytes(slices_x: u32, numer: u32, denom: u32, sx: u32, sy: u32) -> u32 {
    let slice_num = sy as u64 * slices_x as u64 + sx as u64;
    let a = ((slice_num + 1) * numer as u64) / denom as u64;
    let b = (slice_num * numer as u64) / denom as u64;
    (a - b) as u32
}

/// Forward counterpart to [`intra_dc_prediction`]: subtract from each
/// LL coefficient the `mean3` of its already-processed neighbours.
/// After this runs the decoder's `intra_dc_prediction` recovers the
/// originals exactly.
///
/// The prediction context is computed against the *original* band,
/// matching the decoder which uses the *reconstructed* band values as
/// it fills in each sample in raster order. When the decoder does
/// `cur.wrapping_add(prediction(already-reconstructed cells))`, we
/// want the post-subtraction sample to be `orig - prediction(orig of
/// already-reconstructed cells)`, i.e. we must use the *original*
/// values of the already-visited cells as the prediction context.
pub fn forward_dc_prediction(band: &mut SubbandData) {
    // Capture the raw coefficients first so predictions use the
    // pre-subtraction (= post-reconstruction) neighbours.
    let orig = band.data.clone();
    let w = band.width;
    for y in 0..band.height {
        for x in 0..w {
            let prediction: i32 = if x > 0 && y > 0 {
                let a = orig[y * w + (x - 1)];
                let b = orig[(y - 1) * w + (x - 1)];
                let c = orig[(y - 1) * w + x];
                let s = a as i64 + b as i64 + c as i64;
                (s / 3) as i32
            } else if x > 0 {
                orig[y * w + (x - 1)]
            } else if y > 0 {
                orig[(y - 1) * w]
            } else {
                0
            };
            let cur = band.get(y, x);
            band.set(y, x, cur.wrapping_sub(prediction));
        }
    }
}

/// Encode a single LD intra picture payload (no parse info).
///
/// Mirrors [`crate::picture::decode_low_delay_picture`] on the LD path:
/// picture number, transform parameters, then `slices_x * slices_y`
/// slices each of exactly `slice_bytes(sx, sy)` bytes.
pub fn encode_ld_intra_picture(
    sequence: &SequenceHeader,
    params: &LdEncoderParams,
    picture_number: u32,
    y: &[u8],
    u: &[u8],
    v: &[u8],
) -> Vec<u8> {
    let mut w = BitWriter::new();

    // §12.2 picture_header.
    w.byte_align();
    w.write_uint_lit(4, picture_number);

    // §12.3 wavelet_transform — intra, so no zero_residual flag.
    w.byte_align();
    write_ld_transform_parameters(&mut w, params);
    w.byte_align();

    // Per-component coefficient pyramids (forward DWT + forward
    // quantisation + forward DC-prediction subtraction on LL).
    let luma_w = sequence.luma_width;
    let luma_h = sequence.luma_height;
    let chroma_w = sequence.chroma_width;
    let chroma_h = sequence.chroma_height;

    let y_py = forward_component_ld(y, luma_w, luma_h, sequence.luma_depth, params);
    let u_py = forward_component_ld(u, chroma_w, chroma_h, sequence.chroma_depth, params);
    let v_py = forward_component_ld(v, chroma_w, chroma_h, sequence.chroma_depth, params);

    let q_per_level = slice_quantisers(params.qindex, &params.quant_matrix);
    let mut y_qpy = quantise_pyramid_ld(&y_py, &q_per_level);
    let mut u_qpy = quantise_pyramid_ld(&u_py, &q_per_level);
    let mut v_qpy = quantise_pyramid_ld(&v_py, &q_per_level);

    // Forward DC prediction on the LL band (level 0).
    forward_dc_prediction(&mut y_qpy[0][0]);
    forward_dc_prediction(&mut u_qpy[0][0]);
    forward_dc_prediction(&mut v_qpy[0][0]);

    let mut luma_dims: Vec<(usize, usize)> = Vec::with_capacity(params.dwt_depth as usize + 1);
    let mut chroma_dims: Vec<(usize, usize)> = Vec::with_capacity(params.dwt_depth as usize + 1);
    for level in 0..=params.dwt_depth {
        luma_dims.push(subband_dims(luma_w, luma_h, params.dwt_depth, level));
        chroma_dims.push(subband_dims(chroma_w, chroma_h, params.dwt_depth, level));
    }

    for sy in 0..params.slices_y {
        for sx in 0..params.slices_x {
            encode_ld_slice(
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

fn write_ld_transform_parameters(w: &mut BitWriter, params: &LdEncoderParams) {
    w.write_uint(wavelet_index(params.wavelet));
    w.write_uint(params.dwt_depth);
    // §12.4.5.2 slice_parameters — LD branch.
    w.write_uint(params.slices_x);
    w.write_uint(params.slices_y);
    w.write_uint(params.slice_bytes_numer);
    w.write_uint(params.slice_bytes_denom);
    // Default quant matrix — emit flag=0 so the decoder uses the
    // per-filter table.
    w.write_bool(false);
}

fn forward_component_ld(
    plane: &[u8],
    comp_w: u32,
    comp_h: u32,
    depth: u32,
    params: &LdEncoderParams,
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

fn quantise_pyramid_ld(
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

/// Slice-area bounds within a subband — same helper as the HQ path.
fn ld_slice_bounds(
    sx: u32,
    sy: u32,
    slices_x: u32,
    slices_y: u32,
    sub_w: usize,
    sub_h: usize,
) -> (usize, usize, usize, usize) {
    slice_bounds(sx, sy, slices_x, slices_y, sub_w, sub_h)
}

/// Encode one LD slice (§13.5.3.1).
#[allow(clippy::too_many_arguments)]
fn encode_ld_slice(
    w: &mut BitWriter,
    params: &LdEncoderParams,
    sx: u32,
    sy: u32,
    y_qpy: &[[SubbandData; 4]],
    u_qpy: &[[SubbandData; 4]],
    v_qpy: &[[SubbandData; 4]],
    luma_dims: &[(usize, usize)],
    chroma_dims: &[(usize, usize)],
) {
    let slice_n_bytes = slice_bytes(
        params.slices_x,
        params.slice_bytes_numer,
        params.slice_bytes_denom,
        sx,
        sy,
    );
    assert!(
        slice_n_bytes > 0,
        "LD slice budget must be > 0 (sx={sx}, sy={sy})",
    );
    let total_bits = 8u64 * slice_n_bytes as u64;
    let length_bits = intlog2(slice_n_bytes.saturating_mul(8).saturating_sub(7));
    let header_bits = 7u64 + length_bits as u64;
    assert!(
        total_bits > header_bits,
        "slice too small to hold header bits",
    );
    let payload_bits = total_bits - header_bits;

    // Serialise the luma block first into a temp BitWriter (tightly
    // packed — we'll right-pad with 1s to hit the chosen `slice_y_length`).
    let mut luma_tmp = BitWriter::new();
    write_ld_component(&mut luma_tmp, params, y_qpy, luma_dims, sx, sy, true);
    let luma_bits = luma_tmp.measured_bits();

    // Serialise the chroma interleaved stream.
    let mut chroma_tmp = BitWriter::new();
    write_ld_chroma_interleaved(&mut chroma_tmp, params, u_qpy, v_qpy, chroma_dims, sx, sy);
    let chroma_bits = chroma_tmp.measured_bits();

    // Choose slice_y_length — ideally `luma_bits`, but capped at
    // (payload_bits - chroma_bits) so chroma fits as well. If the raw
    // total exceeds the budget we let the luma portion take what it
    // can (the remainder is truncated / 1-padded on the decode side).
    // `saturating_sub` on the else-branch handles the pathological
    // case where chroma alone is oversized — luma then gets 0 bits
    // and chroma is truncated.
    let slice_y_length = if luma_bits + chroma_bits <= payload_bits {
        luma_bits
    } else {
        payload_bits.saturating_sub(chroma_bits)
    };
    let chroma_allot = payload_bits - slice_y_length;

    assert!(
        slice_y_length < (1u64 << length_bits),
        "slice_y_length {slice_y_length} overflows {length_bits} bits — pick a larger slice budget",
    );

    // Header: 7-bit qindex then `length_bits` bits of slice_y_length.
    w.write_nbits(7, params.qindex);
    w.write_nbits(length_bits, slice_y_length as u32);

    // Luma: write serialised bytes up to slice_y_length bits, padded
    // with 1s to exactly slice_y_length.
    write_funnel_bounded(w, luma_tmp, slice_y_length);

    // Chroma: same with `chroma_allot` bits.
    write_funnel_bounded(w, chroma_tmp, chroma_allot);
}

/// Bounded write: consume `src`, append its exact bit payload to `w`
/// and right-pad to `bits_budget` bits using 1-bits. A Funnel read on
/// the decoder side returns 1 past end, which aborts any in-progress
/// interleaved exp-Golomb uint read at value 0 — so 1-padding is
/// harmless even if it falls inside a partially-consumed coefficient.
///
/// If `src` is already longer than `bits_budget`, the tail is
/// truncated. This only ever happens when the caller guarantees
/// it's safe (lossless round-trip tests pre-size the slice budget).
fn write_funnel_bounded(w: &mut BitWriter, src: BitWriter, bits_budget: u64) {
    let src_bits = src.measured_bits();
    let bytes = src.finish();
    // Write up to min(src_bits, bits_budget) bits from `bytes`.
    let to_write = src_bits.min(bits_budget);
    let full_bytes = (to_write / 8) as usize;
    for byte in bytes.iter().take(full_bytes) {
        w.write_nbits(8, *byte as u32);
    }
    let mut bits_written = (full_bytes as u64) * 8;
    let leftover = to_write - bits_written;
    if leftover > 0 {
        let last_byte = bytes[full_bytes];
        // MSB-first packing — leftover bits sit in the high end of
        // `last_byte`. `finish()` right-padded the rest with zeros.
        for i in (8 - leftover as u32..8).rev() {
            w.write_bit(((last_byte >> i) as u32) & 1);
        }
        bits_written += leftover;
    }
    // Pad up to bits_budget with 1s.
    while bits_written < bits_budget {
        w.write_bit(1);
        bits_written += 1;
    }
}

fn write_ld_component(
    w: &mut BitWriter,
    params: &LdEncoderParams,
    qpy: &[[SubbandData; 4]],
    dims: &[(usize, usize)],
    sx: u32,
    sy: u32,
    _is_luma: bool,
) {
    // Level 0 — LL only.
    write_ld_slice_band(w, 0, Orient::LL, sx, sy, params, qpy, dims);
    for level in 1..=params.dwt_depth {
        for orient in [Orient::HL, Orient::LH, Orient::HH] {
            write_ld_slice_band(w, level, orient, sx, sy, params, qpy, dims);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn write_ld_chroma_interleaved(
    w: &mut BitWriter,
    params: &LdEncoderParams,
    u_qpy: &[[SubbandData; 4]],
    v_qpy: &[[SubbandData; 4]],
    chroma_dims: &[(usize, usize)],
    sx: u32,
    sy: u32,
) {
    write_ld_slice_chroma_pair(w, 0, Orient::LL, sx, sy, params, u_qpy, v_qpy, chroma_dims);
    for level in 1..=params.dwt_depth {
        for orient in [Orient::HL, Orient::LH, Orient::HH] {
            write_ld_slice_chroma_pair(w, level, orient, sx, sy, params, u_qpy, v_qpy, chroma_dims);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn write_ld_slice_band(
    w: &mut BitWriter,
    level: u32,
    orient: Orient,
    sx: u32,
    sy: u32,
    params: &LdEncoderParams,
    qpy: &[[SubbandData; 4]],
    dims: &[(usize, usize)],
) {
    let (sub_w, sub_h) = dims[level as usize];
    let (left, right, top, bottom) =
        ld_slice_bounds(sx, sy, params.slices_x, params.slices_y, sub_w, sub_h);
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

#[allow(clippy::too_many_arguments)]
fn write_ld_slice_chroma_pair(
    w: &mut BitWriter,
    level: u32,
    orient: Orient,
    sx: u32,
    sy: u32,
    params: &LdEncoderParams,
    u_qpy: &[[SubbandData; 4]],
    v_qpy: &[[SubbandData; 4]],
    chroma_dims: &[(usize, usize)],
) {
    let (sub_w, sub_h) = chroma_dims[level as usize];
    let (left, right, top, bottom) =
        ld_slice_bounds(sx, sy, params.slices_x, params.slices_y, sub_w, sub_h);
    let u_band = &u_qpy[level as usize][orient.as_index()];
    let v_band = &v_qpy[level as usize][orient.as_index()];
    if u_band.width == 0 || u_band.height == 0 {
        return;
    }
    for y in top..bottom {
        for x in left..right {
            w.write_sint(u_band.get(y, x));
            w.write_sint(v_band.get(y, x));
        }
    }
}

/// Encode one 8-bit 4:2:0 YUV frame as a VC-2 LD intra-only elementary
/// stream.
pub fn encode_single_ld_intra_stream(
    sequence: &SequenceHeader,
    params: &LdEncoderParams,
    picture_number: u32,
    y: &[u8],
    u: &[u8],
    v: &[u8],
) -> Vec<u8> {
    let sh_payload = encode_sequence_header(sequence);
    let pic_payload = encode_ld_intra_picture(sequence, params, picture_number, y, u, v);

    let pi_size = 13usize;
    let sh_unit_len = pi_size + sh_payload.len();
    let pic_unit_len = pi_size + pic_payload.len();
    let eos_unit_len = pi_size;

    let mut out = Vec::with_capacity(sh_unit_len + pic_unit_len + eos_unit_len);
    write_parse_info(&mut out, 0x00, sh_unit_len as u32, 0);
    out.extend_from_slice(&sh_payload);
    // 0xC8 — LD non-reference intra picture.
    write_parse_info(&mut out, 0xC8, pic_unit_len as u32, sh_unit_len as u32);
    out.extend_from_slice(&pic_payload);
    write_parse_info(&mut out, 0x10, 0, pic_unit_len as u32);
    out
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

    /// The LD transform_parameters block we emit should parse back
    /// with identical values via the LD-profile reader in `picture.rs`.
    #[test]
    fn ld_transform_parameters_roundtrip() {
        let params = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 128);
        let mut w = BitWriter::new();
        write_ld_transform_parameters(&mut w, &params);
        w.byte_align();
        let bytes = w.finish();

        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_uint(), 1); // LeGall index
        assert_eq!(r.read_uint(), 3); // dwt_depth
        assert_eq!(r.read_uint(), 4); // slices_x
        assert_eq!(r.read_uint(), 4); // slices_y
        assert_eq!(r.read_uint(), 128 * 16); // slice_bytes_numer
        assert_eq!(r.read_uint(), 16); // slice_bytes_denom (slices_x*slices_y)
        assert!(!r.read_bool()); // custom quant flag off
    }

    /// `forward_dc_prediction` subtracts the same neighbour mean the
    /// decoder adds back — so composing the two on any integer band
    /// is the identity.
    #[test]
    fn forward_dc_prediction_inverts_decoder_prediction() {
        let mut band = SubbandData::new(4, 4);
        let mut expected = SubbandData::new(4, 4);
        // Arbitrary test pattern.
        for y in 0..4 {
            for x in 0..4 {
                let v = (y as i32 * 7 + x as i32 * 3 - 5) * (if (x + y) % 2 == 0 { 1 } else { -1 });
                band.set(y, x, v);
                expected.set(y, x, v);
            }
        }
        forward_dc_prediction(&mut band);
        crate::picture::intra_dc_prediction(&mut band);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(
                    band.get(y, x),
                    expected.get(y, x),
                    "round-trip mismatch at ({y}, {x})"
                );
            }
        }
    }

    /// The emitted LD stream should start with a sequence-header parse
    /// code and then a 0xC8 LD non-reference intra picture.
    #[test]
    fn ld_stream_has_expected_parse_codes() {
        let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
        let params = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 128);
        // Use a smooth pattern so the 128-byte budget is sufficient.
        let mut y = [0u8; 64 * 64];
        let u = [128u8; 32 * 32];
        let v = [128u8; 32 * 32];
        for row in 0..64 {
            for col in 0..64 {
                y[row * 64 + col] = ((row + col) * 2) as u8;
            }
        }
        let stream = encode_single_ld_intra_stream(&seq, &params, 0, &y, &u, &v);
        assert_eq!(&stream[..4], BBCD);
        assert_eq!(stream[4], 0x00); // sequence header
        let sh_next = u32::from_be_bytes([stream[5], stream[6], stream[7], stream[8]]) as usize;
        assert!(sh_next > 13);
        assert_eq!(&stream[sh_next..sh_next + 4], BBCD);
        assert_eq!(stream[sh_next + 4], 0xC8); // LD non-ref intra
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
