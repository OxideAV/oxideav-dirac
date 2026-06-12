//! Picture decode for VC-2 (SMPTE ST 2042-1) and Dirac (BBC).
//!
//! Two low-delay profiles are supported:
//!
//! * **Low Delay (LD)** — Dirac §11.3 / VC-2 §13.5.3. Slices share a
//!   common rate-controlled `slice_bytes_numer / slice_bytes_denom`
//!   ratio; each slice packs qindex, a length prefix for luma, the
//!   luma coefficients, and then the chroma coefficients in the
//!   remaining bits. DC subband prediction is applied to each
//!   component's level-0 LL band.
//!
//! * **High Quality (HQ)** — VC-2 §13.5.4. Each slice is independent:
//!   `slice_prefix_bytes` application-specific bytes, then a 1-byte
//!   common quantisation index, then per-component
//!   `(length_byte * slice_size_scaler)` bounded-byte blocks carrying
//!   that component's coefficients. HQ pictures do **not** use DC
//!   prediction.
//!
//! Core-syntax (arithmetic-coded) intra pictures are also handled
//! through [`crate::picture_core`], and core-syntax inter pictures
//! are decoded via §12.3 block motion data plus §15.8 OBMC in
//! [`crate::picture_inter`] and [`crate::obmc`].

use crate::bits::BitReader;
use crate::parse_info::ParseInfo;
use crate::quant::{inverse_quant, slice_quantisers, QuantMatrix};
use crate::sequence::SequenceHeader;
use crate::subband::{init_pyramid_ho, slice_band_order, subband_dims_ho, Orient, SubbandData};
use crate::wavelet::{idwt, idwt_with_ho, WaveletFilter};

/// A single fully-decoded picture: Y / U / V plane arrays (one signed
/// int per pixel, non-negative after output offset), plus the picture
/// number assigned by the encoder.
#[derive(Debug, Clone)]
pub struct DecodedPicture {
    pub picture_number: u32,
    pub luma_width: usize,
    pub luma_height: usize,
    pub chroma_width: usize,
    pub chroma_height: usize,
    pub y: Vec<i32>,
    pub u: Vec<i32>,
    pub v: Vec<i32>,
    pub luma_depth: u32,
    pub chroma_depth: u32,
}

/// A decoded picture held in its **signed, clipped, pre-output-offset**
/// form, suitable for use as a reference in subsequent inter decodes
/// (§15.4). `y` / `u` / `v` values lie in the range
/// `[-2^(depth-1), 2^(depth-1) - 1]`.
#[derive(Debug, Clone)]
pub struct ReferencePicture {
    pub picture_number: u32,
    pub luma_width: usize,
    pub luma_height: usize,
    pub chroma_width: usize,
    pub chroma_height: usize,
    pub y: Vec<i32>,
    pub u: Vec<i32>,
    pub v: Vec<i32>,
}

/// Which flavour of low-delay profile this picture uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LowDelayProfile {
    /// Low Delay (Dirac §11.3 / VC-2 §13.5.3). Uses
    /// `slice_bytes_numer / slice_bytes_denom` rate control.
    LD,
    /// High Quality (VC-2 §13.5.4). Uses `slice_prefix_bytes` and
    /// `slice_size_scaler` with per-component length-prefixed bytes.
    HQ,
}

/// Wavelet transform parameters as they appear in the bitstream.
#[derive(Debug, Clone)]
pub struct TransformParameters {
    pub wavelet: WaveletFilter,
    pub dwt_depth: u32,
    /// §12.4.4.2 horizontal-only wavelet filter
    /// (`state[wavelet_index_ho]`). Defaults to `wavelet` on pre-v3
    /// streams and on v3 streams that leave
    /// `asym_transform_index_flag` at `False`. Only consulted by the
    /// IDWT when `dwt_depth_ho > 0` (the §15.4.1 horizontal-only loop
    /// runs zero iterations otherwise).
    pub wavelet_ho: WaveletFilter,
    /// §12.4.4.3 horizontal-only transform depth
    /// (`state[dwt_depth_ho]`). `0` for a symmetric transform; `> 0`
    /// adds that many §15.4.2 `h_synthesis` levels atop the
    /// `dwt_depth` symmetric levels.
    pub dwt_depth_ho: u32,
    pub slices_x: u32,
    pub slices_y: u32,
    /// Set only for LD.
    pub slice_bytes_numer: u32,
    /// Set only for LD.
    pub slice_bytes_denom: u32,
    /// Set only for HQ.
    pub slice_prefix_bytes: u32,
    /// Set only for HQ.
    pub slice_size_scaler: u32,
    pub quant_matrix: QuantMatrix,
    pub profile: LowDelayProfile,
}

/// Decode errors raised by this module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PictureError {
    Truncated(&'static str),
    CoreSyntaxNotImplemented,
    InterNotImplemented,
    UnknownWaveletIndex(u32),
    UnsupportedDwtDepth(u32),
    ZeroSliceBytes,
    /// Stream declares `major_version >= 3`, the
    /// `extended_transform_parameters()` block (§12.4.4) selects an
    /// asymmetric transform (`dwt_depth_ho > 0`), **and** the
    /// §12.4.5.3 `custom_quant_matrix` flag is `False` — the stream
    /// relies on the Annex D *asymmetric* default quantisation
    /// matrices, which this decoder has not transcribed yet.
    /// Custom-matrix asymmetric streams decode end-to-end through
    /// [`crate::wavelet::idwt_with_ho`]; only the default-matrix
    /// combination still rejects. `wavelet_index_ho` is the
    /// horizontal-only filter index and `dwt_depth_ho` the extra
    /// horizontal-only depth.
    AsymmetricTransformUnsupported {
        wavelet_index_ho: u32,
        dwt_depth_ho: u32,
    },
    /// Slice data wider than its declared byte length.
    SliceOverflow,
    /// Inter picture referenced a picture number not in the reference
    /// buffer. The decoder front-end is responsible for keeping the
    /// buffer up to date.
    MissingReference(u32),
}

impl core::fmt::Display for PictureError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated(w) => write!(f, "truncated picture payload: {w}"),
            Self::CoreSyntaxNotImplemented => write!(
                f,
                "core-syntax (arithmetic-coded) pictures not yet implemented"
            ),
            Self::InterNotImplemented => write!(f, "inter pictures not yet implemented"),
            Self::UnknownWaveletIndex(i) => write!(f, "unknown wavelet index {i}"),
            Self::UnsupportedDwtDepth(d) => write!(f, "DWT depth {d} not supported"),
            Self::ZeroSliceBytes => write!(f, "slice byte size evaluates to zero"),
            Self::AsymmetricTransformUnsupported {
                wavelet_index_ho,
                dwt_depth_ho,
            } => write!(
                f,
                "v3 asymmetric transform unsupported with the Annex D default quant matrix (wavelet_index_ho={wavelet_index_ho}, dwt_depth_ho={dwt_depth_ho}); custom-matrix asymmetric streams decode"
            ),
            Self::SliceOverflow => write!(f, "slice data overflows declared length"),
            Self::MissingReference(n) => {
                write!(f, "inter picture references missing picture number {n}")
            }
        }
    }
}

impl std::error::Error for PictureError {}

/// Number of bytes in LD slice `(sx, sy)` (§13.5.3.2).
pub fn slice_bytes(slices_x: u32, numer: u32, denom: u32, sx: u32, sy: u32) -> u32 {
    let slice_num = sy as u64 * slices_x as u64 + sx as u64;
    let a = ((slice_num + 1) * numer as u64) / denom as u64;
    let b = (slice_num * numer as u64) / denom as u64;
    (a - b) as u32
}

/// Slice-area coordinates within a subband (§13.5.6.2).
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

/// `intlog2(n)` (§6.4.3) — retained for the sizing test below. The LD
/// `slice_y_length` field now uses [`crate::encoder::ld_length_bits`]
/// (which matches the SMPTE VC-2 reference) rather than the Dirac-spec
/// `intlog2(8*slice_bytes - 7)` formula.
#[cfg(test)]
fn intlog2(n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        32 - (n - 1).leading_zeros()
    }
}

/// Inline bounded-read helper: pulls bits from an outer `BitReader`,
/// defaulting to 1 past end.
struct Funnel<'a, 'b> {
    reader: &'b mut BitReader<'a>,
    bits_left: u64,
}

impl<'a, 'b> Funnel<'a, 'b> {
    fn new(reader: &'b mut BitReader<'a>, bits_left: u64) -> Self {
        Self { reader, bits_left }
    }

    fn read_bitb(&mut self) -> u32 {
        if self.bits_left == 0 {
            return 1;
        }
        self.bits_left -= 1;
        self.reader.read_bit()
    }

    fn read_uintb(&mut self) -> u32 {
        let mut value: u32 = 1;
        // Cap shift depth at 31 so `value <<= 1` cannot overflow past
        // `u32::MAX`. A well-formed stream never gets close to this
        // (Dirac fields are small) but a malformed / truncated buffer
        // whose every follow bit is 0 would otherwise wrap `value` to
        // 0 and underflow on `value - 1` (debug panic / release wrap).
        // The bounded reader's "follow defaults to 1 at EOF" rule
        // already terminates the loop, but only after bits_left/2
        // iterations — which can exceed 31 on large slices. Saturate
        // here for safety.
        let mut depth = 0u32;
        while self.read_bitb() == 0 {
            if depth >= 31 {
                return value.saturating_sub(1);
            }
            value <<= 1;
            if self.read_bitb() == 1 {
                value += 1;
            }
            depth += 1;
        }
        value - 1
    }

    fn read_sintb(&mut self) -> i32 {
        // Clamp the unsigned magnitude to `i32::MAX` so the negate
        // never overflows. `read_uintb` is already capped at 31
        // shifts (i.e. value ≤ u32::MAX), but a value of exactly
        // `i32::MIN as u32 = 0x8000_0000` casts to `i32::MIN` and
        // `-i32::MIN` overflows. This only happens on adversarial /
        // truncated input — well-formed Dirac coefficients fit in
        // a handful of bits.
        let raw = self.read_uintb().min(i32::MAX as u32);
        let v = raw as i32;
        if v != 0 && self.read_bitb() == 1 {
            -v
        } else {
            v
        }
    }

    fn flush(mut self) {
        while self.bits_left > 0 {
            self.bits_left -= 1;
            let _ = self.reader.read_bit();
        }
    }
}

/// Full intra picture decode. The payload starts at the picture
/// number (byte-aligned, immediately after the parse info header).
pub fn decode_picture(
    payload: &[u8],
    parse_info: ParseInfo,
    sequence: &SequenceHeader,
) -> Result<DecodedPicture, PictureError> {
    decode_picture_with_refs(payload, parse_info, sequence, &[])
}

/// Like [`decode_picture`], but allows the caller to supply the
/// reference-picture buffer for inter decodes. Intra pictures ignore
/// the references.
pub fn decode_picture_with_refs(
    payload: &[u8],
    parse_info: ParseInfo,
    sequence: &SequenceHeader,
    references: &[ReferencePicture],
) -> Result<DecodedPicture, PictureError> {
    // Dispatch by profile / syntax family.
    if parse_info.is_low_delay() {
        return decode_low_delay_picture(payload, parse_info, sequence);
    }
    if parse_info.is_core_syntax() {
        return decode_core_syntax_picture(payload, parse_info, sequence, references);
    }
    Err(PictureError::CoreSyntaxNotImplemented)
}

/// Decode the profile (`HQ` / `LD`) implied by a low-delay picture
/// parse code, or `None` for an unrecognised pattern.
///
/// Parse code layout (Dirac BBC spec §9.5.1 + VC-2 ST 2042-1
/// Table 10.1):
/// * bit 7 set + bit 3 set → low-delay picture (`ParseInfo::is_low_delay`)
/// * bit 5 set → HQ profile (parse_code 0xE8 family)
/// * bit 5 clear → LD profile (parse_code 0x88 or 0xC8 family)
///
/// `bit 6` distinguishes the two LD encodings (0x88 = legacy /
/// VC-2 SD-Profile, 0xC8 = AC-coded variant), but both decode
/// through the same Golomb-coded slice path here so we accept both.
/// `bits 2-0` encode num_refs / reference flag and don't affect
/// profile dispatch.
pub(crate) fn low_delay_profile_for(parse_code: u8) -> Option<LowDelayProfile> {
    if (parse_code & 0xF8) == 0xE8 {
        Some(LowDelayProfile::HQ)
    } else if (parse_code & 0xB8) == 0x88 {
        Some(LowDelayProfile::LD)
    } else {
        None
    }
}

/// Full low-delay picture decode (LD or HQ profile).
fn decode_low_delay_picture(
    payload: &[u8],
    parse_info: ParseInfo,
    sequence: &SequenceHeader,
) -> Result<DecodedPicture, PictureError> {
    let profile = low_delay_profile_for(parse_info.parse_code)
        .ok_or(PictureError::CoreSyntaxNotImplemented)?;
    if parse_info.is_inter() {
        return Err(PictureError::InterNotImplemented);
    }

    let mut r = BitReader::new(payload);

    // §12.2 picture_header: byte-align then read a 4-byte picture_number.
    r.byte_align();
    if payload.len() < 4 {
        return Err(PictureError::Truncated("picture number"));
    }
    let picture_number = r.read_uint_lit(4);

    // §12.3 wavelet_transform: for intra pictures, no zero_residual flag.
    r.byte_align();
    let params =
        parse_transform_parameters(&mut r, profile, sequence.parse_parameters.version_major)?;

    r.byte_align();

    let luma_w = sequence.luma_width;
    let luma_h = sequence.luma_height;
    let chroma_w = sequence.chroma_width;
    let chroma_h = sequence.chroma_height;

    // Pyramid and per-level dims cover all `dwt_depth_ho + dwt_depth`
    // levels (§13.2.2 / §13.2.3); with `dwt_depth_ho == 0` this is the
    // legacy symmetric layout.
    let ho = params.dwt_depth_ho;
    let total_levels = ho + params.dwt_depth;
    let mut y_py = init_pyramid_ho(luma_w, luma_h, params.dwt_depth, ho);
    let mut u_py = init_pyramid_ho(chroma_w, chroma_h, params.dwt_depth, ho);
    let mut v_py = init_pyramid_ho(chroma_w, chroma_h, params.dwt_depth, ho);

    let mut luma_dims: Vec<(usize, usize)> = Vec::with_capacity(total_levels as usize + 1);
    let mut chroma_dims: Vec<(usize, usize)> = Vec::with_capacity(total_levels as usize + 1);
    for level in 0..=total_levels {
        luma_dims.push(subband_dims_ho(luma_w, luma_h, params.dwt_depth, ho, level));
        chroma_dims.push(subband_dims_ho(
            chroma_w,
            chroma_h,
            params.dwt_depth,
            ho,
            level,
        ));
    }

    for sy in 0..params.slices_y {
        for sx in 0..params.slices_x {
            match profile {
                LowDelayProfile::LD => decode_ld_slice(
                    &mut r,
                    &params,
                    &mut y_py,
                    &mut u_py,
                    &mut v_py,
                    sx,
                    sy,
                    &luma_dims,
                    &chroma_dims,
                )?,
                LowDelayProfile::HQ => decode_hq_slice(
                    &mut r,
                    &params,
                    &mut y_py,
                    &mut u_py,
                    &mut v_py,
                    sx,
                    sy,
                    &luma_dims,
                    &chroma_dims,
                )?,
            }
        }
    }

    // §13.4 / §13.5.2 DC prediction — LD only. HQ never applies it.
    // For the asymmetric layout the level-0 entry is the L band (same
    // slot 0), so the call shape is identical (§13.5.2 transform_data
    // else-branch: `dc_prediction(state[y_transform][0][L])`).
    if matches!(profile, LowDelayProfile::LD) {
        intra_dc_prediction(&mut y_py[0][0]);
        intra_dc_prediction(&mut u_py[0][0]);
        intra_dc_prediction(&mut v_py[0][0]);
    }

    // §15.4.1 IDWT. With `dwt_depth_ho == 0` `idwt_with_ho` is
    // byte-equivalent to the symmetric `idwt` (the horizontal-only
    // loop runs zero iterations and `wavelet_ho` is unused).
    let y_data = idwt_with_ho(&y_py, params.wavelet, params.wavelet_ho, ho);
    let u_data = idwt_with_ho(&u_py, params.wavelet, params.wavelet_ho, ho);
    let v_data = idwt_with_ho(&v_py, params.wavelet, params.wavelet_ho, ho);
    let y = trim_clip_offset(
        &y_data,
        luma_w as usize,
        luma_h as usize,
        sequence.luma_depth,
    );
    let u = trim_clip_offset(
        &u_data,
        chroma_w as usize,
        chroma_h as usize,
        sequence.chroma_depth,
    );
    let v = trim_clip_offset(
        &v_data,
        chroma_w as usize,
        chroma_h as usize,
        sequence.chroma_depth,
    );

    Ok(DecodedPicture {
        picture_number,
        luma_width: luma_w as usize,
        luma_height: luma_h as usize,
        chroma_width: chroma_w as usize,
        chroma_height: chroma_h as usize,
        y,
        u,
        v,
        luma_depth: sequence.luma_depth,
        chroma_depth: sequence.chroma_depth,
    })
}

/// Full core-syntax picture decode (AC or VLC, intra or inter).
fn decode_core_syntax_picture(
    payload: &[u8],
    parse_info: ParseInfo,
    sequence: &SequenceHeader,
    references: &[ReferencePicture],
) -> Result<DecodedPicture, PictureError> {
    let is_intra = parse_info.is_intra();
    let using_ac = parse_info.using_ac();

    let mut r = BitReader::new(payload);
    r.byte_align();
    if payload.len() < 4 {
        return Err(PictureError::Truncated("picture number"));
    }
    let picture_number = r.read_uint_lit(4);

    // Inter pictures: §9.6.1 reference deltas, §11.2 picture prediction
    // parameters + block motion data, then §11.3 wavelet residue
    // (optionally all-zero).
    let inter_ctx = if parse_info.is_inter() {
        let num_refs = parse_info.num_refs() as u32;
        let d1 = r.read_sint();
        let ref1_num = picture_number.wrapping_add(d1 as u32);
        let ref2_num = if num_refs == 2 {
            let d2 = r.read_sint();
            Some(picture_number.wrapping_add(d2 as u32))
        } else {
            None
        };
        if parse_info.is_reference() {
            let _retd = r.read_sint();
        }
        r.byte_align();
        let pred =
            crate::picture_inter::parse_picture_prediction_parameters(&mut r, sequence, num_refs)?;
        r.byte_align();
        let motion = crate::picture_inter::decode_block_motion_data(&mut r, &pred, num_refs)?;
        r.byte_align();
        Some(InterDecodeContext {
            ref1_num,
            ref2_num,
            pred,
            motion,
        })
    } else {
        if parse_info.is_reference() {
            let _retd = r.read_sint();
        }
        None
    };

    // §11.3 wavelet_transform. Inter pictures have a ZERO_RESIDUAL bool
    // gate; intra pictures unconditionally carry the residual.
    r.byte_align();
    let zero_residual = if parse_info.is_inter() {
        r.read_bool()
    } else {
        false
    };

    let luma_w = sequence.luma_width;
    let luma_h = sequence.luma_height;
    let chroma_w = sequence.chroma_width;
    let chroma_h = sequence.chroma_height;

    // Decode (or zero) the wavelet residue, returning cropped planes.
    let (mut y_plane, mut u_plane, mut v_plane) = if zero_residual {
        (
            vec![0i32; luma_w as usize * luma_h as usize],
            vec![0i32; chroma_w as usize * chroma_h as usize],
            vec![0i32; chroma_w as usize * chroma_h as usize],
        )
    } else {
        // §11.3.1 transform_parameters (core flavour).
        let w_idx = r.read_uint();
        let wavelet = crate::wavelet::WaveletFilter::from_index(w_idx)
            .ok_or(PictureError::UnknownWaveletIndex(w_idx))?;
        let dwt_depth = r.read_uint();
        if dwt_depth > 6 {
            return Err(PictureError::UnsupportedDwtDepth(dwt_depth));
        }
        // §11.3.3 codeblock_parameters.
        let (codeblocks, codeblock_mode) =
            crate::picture_core::parse_codeblock_parameters(&mut r, dwt_depth)?;
        let params = crate::picture_core::CoreTransformParameters {
            wavelet,
            dwt_depth,
            codeblocks,
            codeblock_mode,
            quant_matrix: None,
        };

        r.byte_align();
        let y_py = crate::picture_core::core_transform_component(
            &mut r, &params, luma_w, luma_h, using_ac, is_intra,
        )?;
        let u_py = crate::picture_core::core_transform_component(
            &mut r, &params, chroma_w, chroma_h, using_ac, is_intra,
        )?;
        let v_py = crate::picture_core::core_transform_component(
            &mut r, &params, chroma_w, chroma_h, using_ac, is_intra,
        )?;
        let y_data = idwt(&y_py, wavelet);
        let u_data = idwt(&u_py, wavelet);
        let v_data = idwt(&v_py, wavelet);
        (
            crop_to(&y_data, luma_w as usize, luma_h as usize),
            crop_to(&u_data, chroma_w as usize, chroma_h as usize),
            crop_to(&v_data, chroma_w as usize, chroma_h as usize),
        )
    };

    // §15.8: motion-compensate (inter only) or just clip (intra).
    if let Some(ctx) = inter_ctx {
        let ref1 = find_ref(references, ctx.ref1_num)
            .ok_or(PictureError::MissingReference(ctx.ref1_num))?;
        let ref2_pic = match ctx.ref2_num {
            Some(n) => Some(find_ref(references, n).ok_or(PictureError::MissingReference(n))?),
            None => None,
        };
        motion_compensate_all(
            &mut y_plane,
            &mut u_plane,
            &mut v_plane,
            sequence,
            &ctx.pred,
            &ctx.motion,
            ref1,
            ref2_pic,
        );
    } else {
        clip_plane(&mut y_plane, sequence.luma_depth);
        clip_plane(&mut u_plane, sequence.chroma_depth);
        clip_plane(&mut v_plane, sequence.chroma_depth);
    }

    // §15.10 offset output.
    let y = offset_plane(&y_plane, sequence.luma_depth);
    let u = offset_plane(&u_plane, sequence.chroma_depth);
    let v = offset_plane(&v_plane, sequence.chroma_depth);

    Ok(DecodedPicture {
        picture_number,
        luma_width: luma_w as usize,
        luma_height: luma_h as usize,
        chroma_width: chroma_w as usize,
        chroma_height: chroma_h as usize,
        y,
        u,
        v,
        luma_depth: sequence.luma_depth,
        chroma_depth: sequence.chroma_depth,
    })
}

/// Inter-picture working state assembled while parsing the
/// core-syntax header, held long enough for the motion-compensate
/// pass after the wavelet residue is decoded.
struct InterDecodeContext {
    ref1_num: u32,
    ref2_num: Option<u32>,
    pred: crate::picture_inter::PicturePredictionParams,
    motion: crate::picture_inter::PictureMotionData,
}

fn crop_to(big: &SubbandData, w: usize, h: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            out.push(big.get(y, x));
        }
    }
    out
}

fn clip_plane(plane: &mut [i32], depth: u32) {
    let half = if depth == 0 {
        1i32
    } else {
        1i32 << (depth - 1)
    };
    let lo = -half;
    let hi = half - 1;
    for v in plane.iter_mut() {
        *v = (*v).clamp(lo, hi);
    }
}

fn offset_plane(plane: &[i32], depth: u32) -> Vec<i32> {
    let half = if depth == 0 {
        1i32
    } else {
        1i32 << (depth - 1)
    };
    plane.iter().map(|v| v + half).collect()
}

fn find_ref(refs: &[ReferencePicture], n: u32) -> Option<&ReferencePicture> {
    refs.iter().find(|r| r.picture_number == n)
}

#[allow(clippy::too_many_arguments)]
fn motion_compensate_all(
    y: &mut [i32],
    u: &mut [i32],
    v: &mut [i32],
    sequence: &SequenceHeader,
    pred: &crate::picture_inter::PicturePredictionParams,
    motion: &crate::picture_inter::PictureMotionData,
    ref1: &ReferencePicture,
    ref2: Option<&ReferencePicture>,
) {
    let chroma_h_ratio = sequence.video_params.chroma_format.h_ratio();
    let chroma_v_ratio = sequence.video_params.chroma_format.v_ratio();
    mc_one(
        y,
        sequence.luma_width as usize,
        sequence.luma_height as usize,
        pred,
        motion,
        false,
        sequence.luma_depth,
        sequence.chroma_depth,
        chroma_h_ratio,
        chroma_v_ratio,
        (&ref1.y, ref1.luma_width, ref1.luma_height),
        ref2.map(|r| (r.y.as_slice(), r.luma_width, r.luma_height)),
    );
    mc_one(
        u,
        sequence.chroma_width as usize,
        sequence.chroma_height as usize,
        pred,
        motion,
        true,
        sequence.luma_depth,
        sequence.chroma_depth,
        chroma_h_ratio,
        chroma_v_ratio,
        (&ref1.u, ref1.chroma_width, ref1.chroma_height),
        ref2.map(|r| (r.u.as_slice(), r.chroma_width, r.chroma_height)),
    );
    mc_one(
        v,
        sequence.chroma_width as usize,
        sequence.chroma_height as usize,
        pred,
        motion,
        true,
        sequence.luma_depth,
        sequence.chroma_depth,
        chroma_h_ratio,
        chroma_v_ratio,
        (&ref1.v, ref1.chroma_width, ref1.chroma_height),
        ref2.map(|r| (r.v.as_slice(), r.chroma_width, r.chroma_height)),
    );
}

#[allow(clippy::too_many_arguments)]
fn mc_one(
    pic: &mut [i32],
    comp_w: usize,
    comp_h: usize,
    pred: &crate::picture_inter::PicturePredictionParams,
    motion: &crate::picture_inter::PictureMotionData,
    is_chroma: bool,
    luma_depth: u32,
    chroma_depth: u32,
    chroma_h_ratio: u32,
    chroma_v_ratio: u32,
    ref1: (&[i32], usize, usize),
    ref2: Option<(&[i32], usize, usize)>,
) {
    let (xblen, yblen, xbsep, ybsep) = if is_chroma {
        (
            (pred.luma_xblen / chroma_h_ratio) as usize,
            (pred.luma_yblen / chroma_v_ratio) as usize,
            (pred.luma_xbsep / chroma_h_ratio) as usize,
            (pred.luma_ybsep / chroma_v_ratio) as usize,
        )
    } else {
        (
            pred.luma_xblen as usize,
            pred.luma_yblen as usize,
            pred.luma_xbsep as usize,
            pred.luma_ybsep as usize,
        )
    };
    let params = crate::obmc::McParams {
        len_x: comp_w,
        len_y: comp_h,
        xblen,
        yblen,
        xbsep,
        ybsep,
        blocks_x: pred.blocks_x,
        blocks_y: pred.blocks_y,
        mv_precision: pred.mv_precision,
        is_chroma,
        chroma_h_ratio,
        chroma_v_ratio,
        refs_wt_precision: pred.refs_wt_precision,
        ref1_wt: pred.ref1_wt,
        ref2_wt: pred.ref2_wt,
        luma_depth,
        chroma_depth,
    };
    crate::obmc::motion_compensate(pic, &params, motion, Some(ref1), ref2);
}

/// Parsed §12.4.4 `extended_transform_parameters()` result. Defaults
/// (set by §12.4 just before the call) are `wavelet_index_ho =
/// wavelet_index` and `dwt_depth_ho = 0`; only the flag-gated reads
/// may override them.
///
/// `wavelet_index_ho` is the raw §12.4.4.2 spec value (kept so the
/// asymmetric rejection diagnostic can echo it verbatim);
/// `wavelet_ho` is the typed [`WaveletFilter`] the parser resolved it
/// to. The two are always in lock-step: a bogus index is surfaced as
/// [`PictureError::UnknownWaveletIndex`] at parse time, so any
/// `ExtendedTransformParameters` produced by this module carries a
/// known filter even on the asymmetric path. Callers that route the
/// asymmetric IDWT through [`crate::wavelet::idwt_with_ho`] can pass
/// `wavelet_ho` straight through without re-resolving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExtendedTransformParameters {
    pub wavelet_index_ho: u32,
    pub wavelet_ho: WaveletFilter,
    pub dwt_depth_ho: u32,
}

/// §12.4.4 `extended_transform_parameters()`. Reads two boolean
/// flags, each gating an `interleaved exp-Golomb` field. Defaults
/// per §12.4.4.2 / §12.4.4.3 are inherited from the enclosing
/// `transform_parameters` call (`wavelet_index_ho` defaults to the
/// already-decoded `wavelet_index`; `dwt_depth_ho` defaults to 0).
///
/// Only the syntax is implemented here — the caller decides whether
/// the (possibly non-default) values are usable. The parser must
/// always run on a v3 stream so that subsequent bits (slice
/// parameters, quant matrix) are correctly aligned.
///
/// When `asym_transform_index_flag` is set the parsed index is
/// validated against [`WaveletFilter::from_index`] before the
/// asymmetric-vs-default decision is made: an out-of-range value
/// (`> 6`) surfaces [`PictureError::UnknownWaveletIndex`] in the same
/// way the symmetric `wavelet_index` does in
/// [`parse_transform_parameters`], so the asymmetric rejection
/// downstream never has to disambiguate "valid asymmetric filter" from
/// "bogus filter index".
pub(crate) fn parse_extended_transform_parameters(
    r: &mut BitReader<'_>,
    wavelet_index_default: u32,
    wavelet_default: WaveletFilter,
) -> Result<ExtendedTransformParameters, PictureError> {
    let mut wavelet_index_ho = wavelet_index_default;
    let mut wavelet_ho = wavelet_default;
    let mut dwt_depth_ho: u32 = 0;
    let asym_transform_index_flag = r.read_bool();
    if asym_transform_index_flag {
        wavelet_index_ho = r.read_uint();
        wavelet_ho = WaveletFilter::from_index(wavelet_index_ho)
            .ok_or(PictureError::UnknownWaveletIndex(wavelet_index_ho))?;
    }
    let asym_transform_flag = r.read_bool();
    if asym_transform_flag {
        dwt_depth_ho = r.read_uint();
        // §12.4.4.3 bounds: `dwt_depth_ho` shares the same physical
        // ceiling as `dwt_depth` (the IDWT pyramid cannot exceed six
        // total levels in our implementation). Reject anything that
        // would walk us off the array of subbands.
        if dwt_depth_ho > 6 {
            return Err(PictureError::UnsupportedDwtDepth(dwt_depth_ho));
        }
    }
    Ok(ExtendedTransformParameters {
        wavelet_index_ho,
        wavelet_ho,
        dwt_depth_ho,
    })
}

/// §12.4 `transform_parameters`.
pub(crate) fn parse_transform_parameters(
    r: &mut BitReader<'_>,
    profile: LowDelayProfile,
    major_version: u32,
) -> Result<TransformParameters, PictureError> {
    let w_idx = r.read_uint();
    let wavelet =
        WaveletFilter::from_index(w_idx).ok_or(PictureError::UnknownWaveletIndex(w_idx))?;
    let dwt_depth = r.read_uint();
    if dwt_depth > 6 {
        return Err(PictureError::UnsupportedDwtDepth(dwt_depth));
    }
    // §12.4.4 `extended_transform_parameters()`. The block (v3 only)
    // can optionally select a horizontal-only wavelet filter and an
    // extra horizontal-only DWT depth, both of which feed the §13.5
    // slice unpack (asymmetric band order), the §13.2.3 subband
    // dimensions and the §15.4.1 IDWT driver. For pre-v3 streams the
    // block is absent and `dwt_depth_ho` is 0 (the §12.4.4 NOTE
    // invariant: with `dwt_depth_ho == 0` and `wavelet_index_ho ==
    // wavelet_index` the inverse transform is identical to earlier
    // versions).
    let (wavelet_index_ho, wavelet_ho, dwt_depth_ho) = if major_version >= 3 {
        let ext = parse_extended_transform_parameters(r, w_idx, wavelet)?;
        (ext.wavelet_index_ho, ext.wavelet_ho, ext.dwt_depth_ho)
    } else {
        (w_idx, wavelet, 0)
    };
    // §13.2.2 total level count: a single pyramid spans both the
    // horizontal-only and the 2-D levels. Keep the combined depth
    // within the same physical ceiling as `dwt_depth` itself.
    if dwt_depth_ho + dwt_depth > 6 {
        return Err(PictureError::UnsupportedDwtDepth(dwt_depth_ho + dwt_depth));
    }

    // §12.4.5.2 slice_parameters.
    let slices_x = r.read_uint();
    let slices_y = r.read_uint();
    let (slice_bytes_numer, slice_bytes_denom, slice_prefix_bytes, slice_size_scaler) =
        match profile {
            LowDelayProfile::LD => {
                let n = r.read_uint();
                let d = r.read_uint();
                (n, d.max(1), 0, 1)
            }
            LowDelayProfile::HQ => {
                let pfx = r.read_uint();
                let scl = r.read_uint().max(1);
                (0, 1, pfx, scl)
            }
        };

    // §12.4.5.3 quant_matrix. The custom branch is asymmetric-aware:
    // with `dwt_depth_ho > 0` it reads a single L (DC) band, then
    // `dwt_depth_ho` single H bands, then the `dwt_depth` HL/LH/HH
    // triplets (§13.2.1 ordering). With `dwt_depth_ho == 0` it is the
    // legacy symmetric read. The default (`set_quant_matrix`) branch
    // only covers the *symmetric* Annex D tables: a genuinely
    // asymmetric stream (`dwt_depth_ho > 0`) that relies on the
    // Annex D asymmetric defaults is rejected below until those tables
    // are transcribed. The parse stays bit-exact through the whole
    // transform-parameters block either way, so a caller that recovers
    // from the rejection sees a correctly aligned reader.
    //
    // A v3 stream with `wavelet_index_ho != wavelet_index` but
    // `dwt_depth_ho == 0` is decoded with the symmetric default
    // matrix: the §15.4.1 horizontal-only loop runs zero iterations,
    // so the filter override has no effect on the transform and the
    // quant-matrix layout is the symmetric one.
    let custom_flag = r.read_bool();
    let quant_matrix = if custom_flag {
        QuantMatrix::parse_custom(r, dwt_depth, dwt_depth_ho)
    } else if dwt_depth_ho != 0 {
        return Err(PictureError::AsymmetricTransformUnsupported {
            wavelet_index_ho,
            dwt_depth_ho,
        });
    } else {
        QuantMatrix::default_for(wavelet, dwt_depth)
            .ok_or(PictureError::UnsupportedDwtDepth(dwt_depth))?
    };

    Ok(TransformParameters {
        wavelet,
        dwt_depth,
        wavelet_ho,
        dwt_depth_ho,
        slices_x,
        slices_y,
        slice_bytes_numer,
        slice_bytes_denom,
        slice_prefix_bytes,
        slice_size_scaler,
        quant_matrix,
        profile,
    })
}

/// §13.5.3.1 LD slice decode.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_ld_slice(
    r: &mut BitReader<'_>,
    params: &TransformParameters,
    y_py: &mut [[SubbandData; 4]],
    u_py: &mut [[SubbandData; 4]],
    v_py: &mut [[SubbandData; 4]],
    sx: u32,
    sy: u32,
    luma_dims: &[(usize, usize)],
    chroma_dims: &[(usize, usize)],
) -> Result<(), PictureError> {
    let slice_n_bytes = slice_bytes(
        params.slices_x,
        params.slice_bytes_numer,
        params.slice_bytes_denom,
        sx,
        sy,
    );
    if slice_n_bytes == 0 {
        return Err(PictureError::ZeroSliceBytes);
    }
    let total_bits = 8u64 * slice_n_bytes as u64;
    let qindex = r.read_nbits(7);
    // See `crate::encoder::ld_length_bits` — ffmpeg / SMPTE VC-2 reads
    // `slice_y_length` as `floor(log2(total_bits)) + 1` raw bits.
    let length_bits = crate::encoder::ld_length_bits(slice_n_bytes);
    let slice_y_length = r.read_nbits(length_bits) as u64;
    let header_bits = 7u64 + length_bits as u64;
    if header_bits + slice_y_length > total_bits {
        return Err(PictureError::Truncated("slice header wider than slice"));
    }
    let chroma_bits = total_bits - header_bits - slice_y_length;

    let q_per_level = slice_quantisers(qindex, &params.quant_matrix);

    // §13.5.3 band order — symmetric and asymmetric layouts share
    // the `slice_band_order` sequence (L/LL first, then any
    // horizontal-only H bands, then the HL/LH/HH triplets).
    let order = slice_band_order(params.dwt_depth, params.dwt_depth_ho);
    {
        let mut f = Funnel::new(r, slice_y_length);
        for &(level, orient) in &order {
            decode_luma_band(
                &mut f,
                level,
                orient,
                sx,
                sy,
                params,
                y_py,
                luma_dims,
                &q_per_level,
            );
        }
        f.flush();
    }
    {
        let mut f = Funnel::new(r, chroma_bits);
        for &(level, orient) in &order {
            decode_chroma_pair(
                &mut f,
                level,
                orient,
                sx,
                sy,
                params,
                u_py,
                v_py,
                chroma_dims,
                &q_per_level,
            );
        }
        f.flush();
    }
    Ok(())
}

/// §13.5.4 HQ slice decode.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_hq_slice(
    r: &mut BitReader<'_>,
    params: &TransformParameters,
    y_py: &mut [[SubbandData; 4]],
    u_py: &mut [[SubbandData; 4]],
    v_py: &mut [[SubbandData; 4]],
    sx: u32,
    sy: u32,
    luma_dims: &[(usize, usize)],
    chroma_dims: &[(usize, usize)],
) -> Result<(), PictureError> {
    // Skip `slice_prefix_bytes` bytes of application-specific prefix.
    // `read_uint_lit` from the spec byte-aligns and consumes `n` bytes.
    if params.slice_prefix_bytes > 0 {
        for _ in 0..params.slice_prefix_bytes {
            let _ = r.read_uint_lit(1);
        }
    }
    // qindex: 1 byte (§13.5.4).
    let qindex = r.read_uint_lit(1);

    let q_per_level = slice_quantisers(qindex, &params.quant_matrix);

    // Per-component: length byte -> length = scaler * byte.
    // Bounded-read the `length` bytes' worth of bits into the
    // component's subbands. The luma component uses `y_py`, then
    // chroma C1 / C2 — each independently.
    for comp in 0..3 {
        let len_byte = r.read_uint_lit(1);
        let length_bytes = len_byte as u64 * params.slice_size_scaler as u64;
        let length_bits = length_bytes * 8;
        let mut f = Funnel::new(r, length_bits);
        let dims: &[(usize, usize)] = if comp == 0 { luma_dims } else { chroma_dims };
        let py: &mut [[SubbandData; 4]] = match comp {
            0 => y_py,
            1 => u_py,
            _ => v_py,
        };
        // §13.5.4 band order — shared symmetric/asymmetric sequence.
        for (level, orient) in slice_band_order(params.dwt_depth, params.dwt_depth_ho) {
            decode_hq_component_band(
                &mut f,
                level,
                orient,
                sx,
                sy,
                params,
                py,
                dims,
                &q_per_level,
            );
        }
        f.flush();
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_luma_band(
    f: &mut Funnel<'_, '_>,
    level: u32,
    orient: Orient,
    sx: u32,
    sy: u32,
    params: &TransformParameters,
    y_py: &mut [[SubbandData; 4]],
    luma_dims: &[(usize, usize)],
    q_per_level: &[[u32; 4]],
) {
    let (sub_w, sub_h) = luma_dims[level as usize];
    let (left, right, top, bottom) =
        slice_bounds(sx, sy, params.slices_x, params.slices_y, sub_w, sub_h);
    let q = q_per_level[level as usize][orient.as_index()];
    let band = &mut y_py[level as usize][orient.as_index()];
    for y in top..bottom {
        for x in left..right {
            let val = f.read_sintb();
            band.set(y, x, inverse_quant(val, q));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_chroma_pair(
    f: &mut Funnel<'_, '_>,
    level: u32,
    orient: Orient,
    sx: u32,
    sy: u32,
    params: &TransformParameters,
    u_py: &mut [[SubbandData; 4]],
    v_py: &mut [[SubbandData; 4]],
    chroma_dims: &[(usize, usize)],
    q_per_level: &[[u32; 4]],
) {
    let (sub_w, sub_h) = chroma_dims[level as usize];
    let (left, right, top, bottom) =
        slice_bounds(sx, sy, params.slices_x, params.slices_y, sub_w, sub_h);
    let q = q_per_level[level as usize][orient.as_index()];
    let u_band = &mut u_py[level as usize][orient.as_index()];
    let v_band = &mut v_py[level as usize][orient.as_index()];
    for y in top..bottom {
        for x in left..right {
            let uval = f.read_sintb();
            let vval = f.read_sintb();
            u_band.set(y, x, inverse_quant(uval, q));
            v_band.set(y, x, inverse_quant(vval, q));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_hq_component_band(
    f: &mut Funnel<'_, '_>,
    level: u32,
    orient: Orient,
    sx: u32,
    sy: u32,
    params: &TransformParameters,
    py: &mut [[SubbandData; 4]],
    dims: &[(usize, usize)],
    q_per_level: &[[u32; 4]],
) {
    let (sub_w, sub_h) = dims[level as usize];
    let (left, right, top, bottom) =
        slice_bounds(sx, sy, params.slices_x, params.slices_y, sub_w, sub_h);
    let q = q_per_level[level as usize][orient.as_index()];
    let band = &mut py[level as usize][orient.as_index()];
    for y in top..bottom {
        for x in left..right {
            let val = f.read_sintb();
            band.set(y, x, inverse_quant(val, q));
        }
    }
}

/// §13.4 DC subband prediction. In-place. Used by LD only.
pub fn intra_dc_prediction(band: &mut SubbandData) {
    for y in 0..band.height {
        for x in 0..band.width {
            let prediction: i32 = if x > 0 && y > 0 {
                let a = band.get(y, x - 1);
                let b = band.get(y - 1, x - 1);
                let c = band.get(y - 1, x);
                mean3(a, b, c)
            } else if x > 0 && y == 0 {
                band.get(0, x - 1)
            } else if x == 0 && y > 0 {
                band.get(y - 1, 0)
            } else {
                0
            };
            let cur = band.get(y, x);
            band.set(y, x, cur.wrapping_add(prediction));
        }
    }
}

fn mean3(a: i32, b: i32, c: i32) -> i32 {
    // §5.4 `mean(S)` for n values is `(s0 + .. + s_{n-1} + (n//2)) // n`,
    // the *unbiased* integer mean — note the `+ (n//2)` rounding term.
    // For n = 3 that is `(a + b + c + 1) // 3`. §1.3 defines `//` as
    // floor division (rounds toward -infinity), unlike Rust's `/`
    // which truncates toward zero; for a negative sum this matters:
    // `-7 // 3 = -3` but `-7 / 3 = -2` in Rust, so we use `div_euclid`.
    let s = a as i64 + b as i64 + c as i64 + 1;
    s.div_euclid(3) as i32
}

/// Crop the IDWT output to the real component size (§15.7), clip to
/// `[-2^(depth-1), 2^(depth-1) - 1]` (§15.9), and offset by
/// `2^(depth-1)` for non-negative output (§15.10).
pub(crate) fn trim_clip_offset(
    big: &SubbandData,
    out_w: usize,
    out_h: usize,
    depth: u32,
) -> Vec<i32> {
    let half = if depth == 0 {
        1i32
    } else {
        1i32 << (depth - 1)
    };
    let min = -half;
    let max = half - 1;
    let mut out = Vec::with_capacity(out_w * out_h);
    for y in 0..out_h {
        for x in 0..out_w {
            let v = big.get(y, x).clamp(min, max);
            out.push(v + half);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intra_dc_predict_first_row_col() {
        let mut b = SubbandData::new(3, 3);
        for y in 0..3 {
            for x in 0..3 {
                b.set(y, x, 1);
            }
        }
        intra_dc_prediction(&mut b);
        assert_eq!(b.get(0, 0), 1);
        assert_eq!(b.get(0, 1), 2);
        assert_eq!(b.get(0, 2), 3);
        assert_eq!(b.get(1, 0), 2);
        // §5.4 unbiased mean of the already-reconstructed neighbours
        // (left=2, top-left=1, top=2): (2 + 1 + 2 + 1) // 3 = 2, so the
        // reconstructed coefficient is 1 + 2 = 3. The `+ 1` is the
        // `(n // 2)` rounding term the spec's `mean()` carries.
        assert_eq!(b.get(1, 1), 1 + (2 + 1 + 2 + 1) / 3);
        assert_eq!(b.get(1, 1), 3);
    }

    /// `mean3` is the spec's `mean(a, b, c)` = `(a + b + c + 1) // 3`
    /// with floor division. Spot-check the rounding bias and the
    /// negative-sum floor behaviour that distinguishes it from Rust's
    /// truncating `/`.
    #[test]
    fn mean3_unbiased_floor() {
        // Exact multiple of 3: the +1 bias does not change the result.
        assert_eq!(mean3(1, 1, 1), 1);
        assert_eq!(mean3(2, 2, 2), 2);
        // sum = 5 -> (5 + 1) // 3 = 2 (round-to-nearest behaviour from
        // the +1 bias; truncating 5/3 would give 1).
        assert_eq!(mean3(2, 1, 2), 2);
        // sum = 4 -> (4 + 1) // 3 = 1.
        assert_eq!(mean3(2, 1, 1), 1);
        // Negative sum: (-7 + 1) // 3 = -6 // 3 = -2 (floor); a
        // truncating divide of -6/3 also gives -2 here, but -8 shows
        // the floor: (-8 + 1) // 3 = -7 // 3 = -3.
        assert_eq!(mean3(-3, -2, -2), -2);
        assert_eq!(mean3(-3, -3, -2), -3);
    }

    #[test]
    fn intlog2_sizing() {
        assert_eq!(intlog2(0), 0);
        assert_eq!(intlog2(1), 0);
        assert_eq!(intlog2(2), 1);
        assert_eq!(intlog2(3), 2);
        assert_eq!(intlog2(4), 2);
        assert_eq!(intlog2(5), 3);
    }

    #[test]
    fn slice_bytes_uniform() {
        for sy in 0..4 {
            for sx in 0..4 {
                assert_eq!(slice_bytes(4, 4000, 4, sx, sy), 1000);
            }
        }
    }

    #[test]
    fn slice_bytes_fractional_distributes_one_byte_off() {
        let values: Vec<u32> = (0..4).map(|sx| slice_bytes(4, 17, 4, sx, 0)).collect();
        assert_eq!(values.iter().sum::<u32>(), 17);
    }

    #[test]
    fn slice_bounds_partition_subband() {
        let (l, r, t, b) = slice_bounds(0, 0, 2, 2, 8, 8);
        assert_eq!((l, r, t, b), (0, 4, 0, 4));
        let (l, r, t, b) = slice_bounds(1, 1, 2, 2, 8, 8);
        assert_eq!((l, r, t, b), (4, 8, 4, 8));
    }

    /// LD parse codes: 0x88 (VC-2 SD-Profile / legacy), 0xC8 (AC-coded
    /// variant), and their reference siblings 0x8C / 0xCC. Real ffmpeg
    /// `vc2enc` output uses 0x88 — the in-tree `corpus_vc2_low_delay_*`
    /// fixtures sliced from a vc2enc-produced stream were rejected as
    /// "unsupported core-syntax parse code" until the `0x88` family
    /// was admitted. HQ is parse 0xE8 and must NOT be misclassified as
    /// LD; core-syntax parse 0x08 must remain unmatched here.
    #[test]
    fn low_delay_profile_recognises_88_and_c8_families() {
        for code in [0x88u8, 0x89, 0x8C, 0x8D] {
            assert_eq!(
                low_delay_profile_for(code),
                Some(LowDelayProfile::LD),
                "0x{code:02X} should be LD"
            );
        }
        for code in [0xC8u8, 0xC9, 0xCC, 0xCD] {
            assert_eq!(
                low_delay_profile_for(code),
                Some(LowDelayProfile::LD),
                "0x{code:02X} should be LD"
            );
        }
        for code in [0xE8u8, 0xE9, 0xEC, 0xED] {
            assert_eq!(
                low_delay_profile_for(code),
                Some(LowDelayProfile::HQ),
                "0x{code:02X} should be HQ"
            );
        }
        // Core-syntax (parse 0x08 / 0x0C / 0x0D / 0x0A) must NOT be
        // dispatched through the low-delay decoder.
        for code in [0x08u8, 0x0C, 0x0D, 0x0A, 0x09, 0x00, 0x10, 0x20, 0x30] {
            assert_eq!(
                low_delay_profile_for(code),
                None,
                "0x{code:02X} should not be LD/HQ"
            );
        }
    }

    #[test]
    fn trim_clip_offset_8bit() {
        let mut sd = SubbandData::new(4, 4);
        sd.set(0, 0, -200);
        sd.set(0, 1, 0);
        sd.set(0, 2, 100);
        sd.set(0, 3, 300);
        let out = trim_clip_offset(&sd, 4, 1, 8);
        assert_eq!(&out, &[0, 128, 228, 255]);
    }

    // ---------------------------------------------------------------
    // §12.4.4 extended_transform_parameters tests
    //
    // Bit stream layout per the VC-2 pseudocode (one `read_bool` per
    // flag, one interleaved-exp-Golomb `read_uint` per gated field).
    // Bits are packed MSB-first within a byte; we wrap the byte
    // payload in a fresh BitReader for each case.
    // ---------------------------------------------------------------

    fn parse_ext(
        default_widx: u32,
        bytes: &[u8],
    ) -> Result<ExtendedTransformParameters, PictureError> {
        let mut r = BitReader::new(bytes);
        let default_filter = WaveletFilter::from_index(default_widx)
            .expect("test default wavelet index must be 0..=6");
        parse_extended_transform_parameters(&mut r, default_widx, default_filter)
    }

    /// Both flags zero → defaults survive: `wavelet_index_ho` =
    /// `wavelet_index`, `wavelet_ho` = the symmetric typed filter,
    /// `dwt_depth_ho` = 0. Exactly two flag bits are consumed.
    #[test]
    fn extended_transform_parameters_both_flags_off() {
        // Byte 0b00xxxxxx — two leading 0 bits, rest is padding the
        // caller never looks at.
        let parsed = parse_ext(2, &[0b0000_0000]).unwrap();
        assert_eq!(
            parsed,
            ExtendedTransformParameters {
                wavelet_index_ho: 2,
                wavelet_ho: WaveletFilter::DeslauriersDubuc13_7,
                dwt_depth_ho: 0,
            }
        );
    }

    /// `asym_transform_index_flag = 1` overrides `wavelet_index_ho`.
    /// The follow `read_uint()` of value 5 is the five-bit interleaved
    /// exp-Golomb code `0 1 0 0 1` (follow/data/follow/data/term);
    /// then `asym_transform_flag = 0`. The typed `wavelet_ho` follows
    /// the raw index through [`WaveletFilter::from_index`].
    #[test]
    fn extended_transform_parameters_only_index_flag() {
        // Bit pattern (MSB first):
        //   1                          asym_transform_index_flag = 1
        //   0 1 0 0 1                  read_uint() = 5
        //   0                          asym_transform_flag = 0
        // 7 bits + 1 pad = 0b1010_0100 = 0xA4
        let parsed = parse_ext(0, &[0xA4]).unwrap();
        assert_eq!(
            parsed,
            ExtendedTransformParameters {
                wavelet_index_ho: 5,
                wavelet_ho: WaveletFilter::Fidelity,
                dwt_depth_ho: 0,
            }
        );
    }

    /// `asym_transform_flag = 1` overrides `dwt_depth_ho`. Pattern:
    /// flag1=0, flag2=1, then `read_uint() = 0` (one bit `1`). The
    /// `wavelet_ho` field carries the symmetric default.
    #[test]
    fn extended_transform_parameters_only_depth_flag() {
        // Bit pattern (MSB first):
        //   0                          asym_transform_index_flag = 0
        //   1                          asym_transform_flag = 1
        //   1                          read_uint() = 0
        // = 0b0110_0000 = 0x60
        let parsed = parse_ext(4, &[0x60]).unwrap();
        assert_eq!(
            parsed,
            ExtendedTransformParameters {
                wavelet_index_ho: 4,
                wavelet_ho: WaveletFilter::Haar1,
                dwt_depth_ho: 0,
            }
        );
    }

    /// Both flags set. Bits: flag1=1, `read_uint()`=3 (`00001`,
    /// 5 bits), flag2=1, `read_uint()`=1 (`001`, 3 bits). Total
    /// 10 bits = 2 bytes (6 trailing pad bits at zero are unread).
    /// `wavelet_ho` resolves to [`WaveletFilter::Haar0`].
    #[test]
    fn extended_transform_parameters_both_flags_on() {
        // 1 00001 1 001 = byte0 `1000_0110` = 0x86, byte1 `01xx_xxxx` = 0x40.
        let parsed = parse_ext(0, &[0x86, 0x40]).unwrap();
        assert_eq!(
            parsed,
            ExtendedTransformParameters {
                wavelet_index_ho: 3,
                wavelet_ho: WaveletFilter::Haar0,
                dwt_depth_ho: 1,
            }
        );
    }

    /// `dwt_depth_ho > 6` is rejected (the IDWT pyramid cannot grow
    /// that deep — same ceiling as `dwt_depth`).
    #[test]
    fn extended_transform_parameters_dwt_depth_ho_over_cap_rejected() {
        // flag1=0, flag2=1, then read_uint()=7 = 7-bit code
        // `0000001` (six follow-zeros then the terminator). Bits:
        // 0 1 0000001 = 9 bits → byte0 `0100_0000` = 0x40,
        // byte1 `1xxx_xxxx` = 0x80.
        let err = parse_ext(0, &[0x40, 0x80]).unwrap_err();
        assert!(matches!(err, PictureError::UnsupportedDwtDepth(7)));
    }

    /// `asym_transform_index_flag = 1` with an out-of-range
    /// `wavelet_index_ho` (`> 6`, since §12.4.4.2 reuses the
    /// `wavelet_index` value-space) is rejected as
    /// [`PictureError::UnknownWaveletIndex`] at parse time, mirroring
    /// the symmetric `wavelet_index` validation in
    /// [`parse_transform_parameters`]. This keeps the asymmetric
    /// rejection downstream free of the "valid filter or bogus index?"
    /// disambiguation — by the time the caller sees an
    /// `AsymmetricTransformUnsupported`, the embedded filter index is
    /// guaranteed to be a known [`WaveletFilter`].
    #[test]
    fn extended_transform_parameters_unknown_wavelet_index_ho_rejected() {
        // `read_uint` is interleaved exp-Golomb (`follow,data` pairs
        // terminated by a `1` follow-bit). Bit-level trace for
        // value = 8: start `value = 1`, then
        //   follow=0,data=0 → 2; follow=0,data=0 → 4;
        //   follow=0,data=1 → 9; terminator=1 → return 9 − 1 = 8.
        // That's the 7-bit sequence `0 0 0 0 0 1 1`. Prefixed with the
        // `asym_transform_index_flag = 1` bit gives the 8-bit byte
        // `1 0 0 0 0 0 1 1` = 0b1000_0011 = 0x83. The parser bails out
        // at `from_index(8)` before reading `asym_transform_flag`, so
        // a single byte is sufficient.
        let err = parse_ext(0, &[0x83]).unwrap_err();
        assert!(
            matches!(err, PictureError::UnknownWaveletIndex(8)),
            "expected UnknownWaveletIndex(8), got {err:?}"
        );
    }

    /// `asym_transform_index_flag = 1` with a valid-but-different
    /// `wavelet_index_ho` is **not** rejected by
    /// [`parse_extended_transform_parameters`] itself — the asymmetric
    /// vs symmetric decision belongs to the caller
    /// ([`parse_transform_parameters`] surfaces
    /// [`PictureError::AsymmetricTransformUnsupported`]). This test
    /// pins that the parser returns the typed filter unmodified.
    #[test]
    fn extended_transform_parameters_valid_asymmetric_index_passes_through() {
        // Bit pattern (MSB first):
        //   1                          asym_transform_index_flag = 1
        //   0 0 0 0 1                  read_uint() = 3 (Haar0)
        //   0                          asym_transform_flag = 0
        // 7 bits → 0b1000_0100 = 0x84 with 1 trailing pad bit.
        // Default is wavelet 1 = LeGall5_3.
        let parsed = parse_ext(1, &[0x84]).unwrap();
        assert_eq!(
            parsed,
            ExtendedTransformParameters {
                wavelet_index_ho: 3,
                wavelet_ho: WaveletFilter::Haar0,
                dwt_depth_ho: 0,
            }
        );
    }

    // ---------------------------------------------------------------
    // §12.4 parse_transform_parameters — asymmetric quant_matrix path
    // ---------------------------------------------------------------

    use crate::bitwriter::BitWriter;

    /// Build a full §12.4 `transform_parameters` byte payload with a
    /// custom quant_matrix. When `ho > 0` the extended block selects an
    /// asymmetric transform (`asym_transform_flag = 1`, the same
    /// `wavelet_index` reused as `wavelet_index_ho`) and the matrix is
    /// emitted in the §12.4.5.3 asymmetric shape; when `ho == 0` it is
    /// a symmetric v3 stream whose extended block reduces to the
    /// default.
    fn build_transform_params_hq(
        wavelet_index: u32,
        dwt_depth: u32,
        ho: u32,
        matrix_uints: &[u32],
    ) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.write_uint(wavelet_index); // wavelet_index
        w.write_uint(dwt_depth); // dwt_depth
                                 // extended_transform_parameters (§12.4.4):
        w.write_bool(false); // asym_transform_index_flag = 0 → ho index = default
        if ho > 0 {
            w.write_bool(true); // asym_transform_flag = 1
            w.write_uint(ho); // dwt_depth_ho
        } else {
            w.write_bool(false); // asym_transform_flag = 0
        }
        // slice_parameters (§12.4.5.2, HQ branch): slices_x, slices_y,
        // slice_prefix_bytes, slice_size_scaler.
        w.write_uint(1);
        w.write_uint(1);
        w.write_uint(0);
        w.write_uint(1);
        // quant_matrix (§12.4.5.3): custom flag then body.
        w.write_bool(true);
        for &v in matrix_uints {
            w.write_uint(v);
        }
        w.finish()
    }

    /// A v3 asymmetric stream with a custom quant matrix parses to a
    /// full `TransformParameters`: `dwt_depth_ho` and `wavelet_ho` are
    /// populated and the §12.4.5.3 matrix body is read in the
    /// asymmetric shape (L, then `ho` H bands, then `dwt_depth`
    /// triplets). If the parser read the matrix with the symmetric
    /// shape it would mis-align and consume the wrong number of uints,
    /// so the value-checked matrix pins the layout.
    #[test]
    fn parse_transform_parameters_asymmetric_custom_matrix_accepted() {
        // wavelet_index = 1 (LeGall5_3), dwt_depth = 1, ho = 2.
        // Matrix (asymmetric): L, H, H, then one HL/LH/HH triplet.
        let matrix = [7u32, 11, 13, 5, 6, 8];
        let bytes = build_transform_params_hq(1, 1, 2, &matrix);
        let mut r = BitReader::new(&bytes);
        let tp = parse_transform_parameters(&mut r, LowDelayProfile::HQ, 3).unwrap();
        assert_eq!(tp.wavelet, WaveletFilter::LeGall5_3);
        // asym_transform_index_flag was 0 → wavelet_ho defaults to the
        // 2-D filter (§12.4.4.2).
        assert_eq!(tp.wavelet_ho, WaveletFilter::LeGall5_3);
        assert_eq!(tp.dwt_depth, 1);
        assert_eq!(tp.dwt_depth_ho, 2);
        assert_eq!(tp.quant_matrix.dwt_depth_ho, 2);
        assert_eq!(tp.quant_matrix.levels[0], [7, 0, 0, 0]); // L
        assert_eq!(tp.quant_matrix.levels[1], [11, 0, 0, 0]); // H
        assert_eq!(tp.quant_matrix.levels[2], [13, 0, 0, 0]); // H
        assert_eq!(tp.quant_matrix.levels[3], [0, 5, 6, 8]); // HL/LH/HH
    }

    /// A v3 asymmetric stream that relies on the Annex D *default*
    /// quantisation matrices (`custom_quant_matrix = False`) is still
    /// rejected with `AsymmetricTransformUnsupported` — those tables
    /// are not transcribed yet. Only the default-matrix combination
    /// rejects; the custom-matrix stream above decodes.
    #[test]
    fn parse_transform_parameters_asymmetric_default_matrix_rejected() {
        let mut w = BitWriter::new();
        w.write_uint(1); // wavelet_index = LeGall5_3
        w.write_uint(1); // dwt_depth
        w.write_bool(false); // asym_transform_index_flag
        w.write_bool(true); // asym_transform_flag
        w.write_uint(2); // dwt_depth_ho
        w.write_uint(1); // slices_x
        w.write_uint(1); // slices_y
        w.write_uint(0); // slice_prefix_bytes
        w.write_uint(1); // slice_size_scaler
        w.write_bool(false); // custom_quant_matrix = False → Annex D
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        let err = parse_transform_parameters(&mut r, LowDelayProfile::HQ, 3).unwrap_err();
        assert_eq!(
            err,
            PictureError::AsymmetricTransformUnsupported {
                wavelet_index_ho: 1,
                dwt_depth_ho: 2,
            }
        );
    }

    /// A symmetric v3 stream (extended block present but reducing to
    /// the default, `ho == 0`) still parses to a full
    /// `TransformParameters` with the custom matrix intact — the
    /// deferred-rejection refactor must not regress the symmetric path.
    #[test]
    fn parse_transform_parameters_symmetric_v3_still_parses() {
        // wavelet_index = 1, dwt_depth = 1, ho = 0.
        // Symmetric matrix: LL, then one HL/LH/HH triplet.
        let matrix = [4u32, 2, 2, 0];
        let bytes = build_transform_params_hq(1, 1, 0, &matrix);
        let mut r = BitReader::new(&bytes);
        let tp = parse_transform_parameters(&mut r, LowDelayProfile::HQ, 3).unwrap();
        assert_eq!(tp.wavelet, WaveletFilter::LeGall5_3);
        assert_eq!(tp.dwt_depth, 1);
        assert_eq!(tp.quant_matrix.dwt_depth_ho, 0);
        assert_eq!(tp.quant_matrix.levels[0], [4, 0, 0, 0]);
        assert_eq!(tp.quant_matrix.levels[1], [0, 2, 2, 0]);
    }
}
