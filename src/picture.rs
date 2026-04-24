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
use crate::subband::{init_pyramid, padded_component_dims, subband_dims, Orient, SubbandData};
use crate::wavelet::{idwt, WaveletFilter};

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
    /// Extended transform parameters (version 3) aren't implemented —
    /// the fixture uses v2 so this never trips in practice.
    ExtendedTransformParams,
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
            Self::ExtendedTransformParams => {
                write!(f, "extended transform parameters (v3) not implemented")
            }
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

/// `intlog2(n)` for §13.5.3.1 slice-length sizing.
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
        while self.read_bitb() == 0 {
            value <<= 1;
            if self.read_bitb() == 1 {
                value += 1;
            }
        }
        value - 1
    }

    fn read_sintb(&mut self) -> i32 {
        let v = self.read_uintb() as i32;
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

/// Full low-delay picture decode (LD or HQ profile).
fn decode_low_delay_picture(
    payload: &[u8],
    parse_info: ParseInfo,
    sequence: &SequenceHeader,
) -> Result<DecodedPicture, PictureError> {
    let profile = if (parse_info.parse_code & 0xF8) == 0xE8 {
        LowDelayProfile::HQ
    } else if (parse_info.parse_code & 0xF8) == 0xC8 {
        LowDelayProfile::LD
    } else {
        return Err(PictureError::CoreSyntaxNotImplemented);
    };
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

    let mut y_py = init_pyramid(luma_w, luma_h, params.dwt_depth);
    let mut u_py = init_pyramid(chroma_w, chroma_h, params.dwt_depth);
    let mut v_py = init_pyramid(chroma_w, chroma_h, params.dwt_depth);

    let mut luma_dims: Vec<(usize, usize)> = Vec::with_capacity(params.dwt_depth as usize + 1);
    let mut chroma_dims: Vec<(usize, usize)> = Vec::with_capacity(params.dwt_depth as usize + 1);
    for level in 0..=params.dwt_depth {
        luma_dims.push(subband_dims(luma_w, luma_h, params.dwt_depth, level));
        chroma_dims.push(subband_dims(chroma_w, chroma_h, params.dwt_depth, level));
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

    // §13.4 DC prediction — LD only. HQ never applies it.
    if matches!(profile, LowDelayProfile::LD) {
        intra_dc_prediction(&mut y_py[0][0]);
        intra_dc_prediction(&mut u_py[0][0]);
        intra_dc_prediction(&mut v_py[0][0]);
    }

    let y_data = idwt(&y_py, params.wavelet);
    let u_data = idwt(&u_py, params.wavelet);
    let v_data = idwt(&v_py, params.wavelet);

    let (_luma_pw, _luma_ph) = padded_component_dims(luma_w, luma_h, params.dwt_depth);
    let (_chroma_pw, _chroma_ph) = padded_component_dims(chroma_w, chroma_h, params.dwt_depth);
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

/// §12.4 `transform_parameters`.
fn parse_transform_parameters(
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
    if major_version >= 3 {
        // §12.4.4 extended_transform_parameters. Not used by our fixture.
        return Err(PictureError::ExtendedTransformParams);
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

    // §12.4.5.3 quant_matrix.
    let custom_flag = r.read_bool();
    let quant_matrix = if custom_flag {
        let ll0 = r.read_uint();
        let mut levels: Vec<[u32; 4]> = Vec::with_capacity(dwt_depth as usize + 1);
        levels.push([ll0, 0, 0, 0]);
        for _ in 1..=dwt_depth {
            let hl = r.read_uint();
            let lh = r.read_uint();
            let hh = r.read_uint();
            levels.push([0, hl, lh, hh]);
        }
        QuantMatrix { dwt_depth, levels }
    } else {
        QuantMatrix::default_for(wavelet, dwt_depth)
            .ok_or(PictureError::UnsupportedDwtDepth(dwt_depth))?
    };

    Ok(TransformParameters {
        wavelet,
        dwt_depth,
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
fn decode_ld_slice(
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
    let length_bits = intlog2(slice_n_bytes.saturating_mul(8).saturating_sub(7));
    let slice_y_length = r.read_nbits(length_bits) as u64;
    let header_bits = 7u64 + length_bits as u64;
    if header_bits + slice_y_length > total_bits {
        return Err(PictureError::Truncated("slice header wider than slice"));
    }
    let chroma_bits = total_bits - header_bits - slice_y_length;

    let q_per_level = slice_quantisers(qindex, &params.quant_matrix);

    {
        let mut f = Funnel::new(r, slice_y_length);
        decode_luma_band(
            &mut f,
            0,
            Orient::LL,
            sx,
            sy,
            params,
            y_py,
            luma_dims,
            &q_per_level,
        );
        for level in 1..=params.dwt_depth {
            for orient in [Orient::HL, Orient::LH, Orient::HH] {
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
        }
        f.flush();
    }
    {
        let mut f = Funnel::new(r, chroma_bits);
        decode_chroma_pair(
            &mut f,
            0,
            Orient::LL,
            sx,
            sy,
            params,
            u_py,
            v_py,
            chroma_dims,
            &q_per_level,
        );
        for level in 1..=params.dwt_depth {
            for orient in [Orient::HL, Orient::LH, Orient::HH] {
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
        }
        f.flush();
    }
    Ok(())
}

/// §13.5.4 HQ slice decode.
#[allow(clippy::too_many_arguments)]
fn decode_hq_slice(
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
        decode_hq_component_band(
            &mut f,
            0,
            Orient::LL,
            sx,
            sy,
            params,
            py,
            dims,
            &q_per_level,
        );
        for level in 1..=params.dwt_depth {
            for orient in [Orient::HL, Orient::LH, Orient::HH] {
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
    let s = a as i64 + b as i64 + c as i64;
    (s / 3) as i32
}

/// Crop the IDWT output to the real component size (§15.7), clip to
/// `[-2^(depth-1), 2^(depth-1) - 1]` (§15.9), and offset by
/// `2^(depth-1)` for non-negative output (§15.10).
fn trim_clip_offset(big: &SubbandData, out_w: usize, out_h: usize, depth: u32) -> Vec<i32> {
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
        assert_eq!(b.get(1, 1), 1 + (2 + 1 + 2) / 3);
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
}
