//! Core-syntax picture decode (§11 / §13.4).
//!
//! The Dirac "core syntax" is the family of parse codes whose bit 7 is
//! clear (`0x08`, `0x0C`, `0x09`..`0x0B`, etc.). Unlike the VC-2
//! low-delay profiles, core-syntax pictures split each subband into
//! **codeblocks** and entropy-code the coefficients with either plain
//! VLC reads or the Dirac binary arithmetic coder (Annex A.4).
//!
//! This module implements:
//!
//! * §11.3.3 `codeblock_parameters()` — per-level codeblock counts +
//!   single/multi-quantiser mode.
//! * §13.4 coefficient unpacking — per-component, per-subband iteration
//!   with per-codeblock skip flag, optional quant offset, and
//!   coefficient entropy decode.
//! * §13.4.4 coefficient context selection — zero-parent /
//!   zero-neighbourhood / sign-prediction conditioning for the arith
//!   path, as laid out in Table 13.1.
//!
//! VLC (non-AC) and arithmetic (AC) paths share everything except the
//! entropy engine. Arithmetic-coded streams also carry the extra
//! `ZERO_BLOCK` and `Q_OFFSET*` contexts for the skipped-codeblock flag
//! and the per-codeblock quant-index delta.

use crate::arith::{ArithDecoder, ContextBank};
use crate::bits::BitReader;
use crate::picture::PictureError;
use crate::quant::{inverse_quant, QuantMatrix};
use crate::subband::{init_pyramid, subband_dims, Orient, SubbandData};
use crate::wavelet::WaveletFilter;

/// Core-syntax transform parameters (§11.3).
#[derive(Debug, Clone)]
pub struct CoreTransformParameters {
    pub wavelet: WaveletFilter,
    pub dwt_depth: u32,
    /// `CODEBLOCKS_X[level]` and `CODEBLOCKS_Y[level]` — number of
    /// codeblocks per axis at each level. If no spatial partition flag
    /// is set in the stream, all levels get `(1, 1)`.
    pub codeblocks: Vec<(u32, u32)>,
    /// `CODEBLOCK_MODE`: 0 = single quantiser, 1 = per-codeblock delta.
    pub codeblock_mode: u32,
    /// Either `custom_quant_matrix` (only in v3 extended params) or
    /// the defaulted matrix. Non-LD streams typically don't transmit
    /// a matrix here — the matrix parameter is only parsed for
    /// low-delay profiles in v2. `None` means "use raw quantisation
    /// without a weighting matrix".
    pub quant_matrix: Option<QuantMatrix>,
}

/// Context labels for §13.4.4.4 Table 13.1. These indices are also
/// used for the codeblock quant-offset (§13.4.3.4) and the codeblock
/// skip flag (§13.4.3.3).
#[allow(non_camel_case_types, dead_code)]
pub mod ctx {
    pub const SIGN_ZERO: usize = 0;
    pub const SIGN_POS: usize = 1;
    pub const SIGN_NEG: usize = 2;
    pub const ZPZN_F1: usize = 3;
    pub const ZPNN_F1: usize = 4;
    pub const ZP_F2: usize = 5;
    pub const ZP_F3: usize = 6;
    pub const ZP_F4: usize = 7;
    pub const ZP_F5: usize = 8;
    pub const ZP_F6_PLUS: usize = 9;
    pub const NPZN_F1: usize = 10;
    pub const NPNN_F1: usize = 11;
    pub const NP_F2: usize = 12;
    pub const NP_F3: usize = 13;
    pub const NP_F4: usize = 14;
    pub const NP_F5: usize = 15;
    pub const NP_F6_PLUS: usize = 16;
    pub const COEFF_DATA: usize = 17;
    pub const ZERO_BLOCK: usize = 18;
    pub const Q_OFFSET_FOLLOW: usize = 19;
    pub const Q_OFFSET_DATA: usize = 20;
    pub const Q_OFFSET_SIGN: usize = 21;
    pub const NUM_CONTEXTS: usize = 22;
}

/// Codeblock bounds within a subband (§13.4.3.1).
fn codeblock_bounds(
    cx: u32,
    cy: u32,
    codeblocks_x: u32,
    codeblocks_y: u32,
    band_w: usize,
    band_h: usize,
) -> (usize, usize, usize, usize) {
    let left = (band_w * cx as usize) / codeblocks_x as usize;
    let right = (band_w * (cx as usize + 1)) / codeblocks_x as usize;
    let top = (band_h * cy as usize) / codeblocks_y as usize;
    let bottom = (band_h * (cy as usize + 1)) / codeblocks_y as usize;
    (left, right, top, bottom)
}

/// Decode one core-syntax component (Y, C1 or C2), returning the
/// freshly unpacked subband pyramid (§13.4.1 `core_transform_data`).
///
/// On entry the outer `BitReader` is byte-aligned, pointing at the
/// first subband length code. On return it sits right after the
/// component's last subband, byte-aligned per §13.4.2.
#[allow(clippy::too_many_arguments)]
pub fn core_transform_component(
    r: &mut BitReader<'_>,
    params: &CoreTransformParameters,
    comp_width: u32,
    comp_height: u32,
    using_ac: bool,
    is_intra: bool,
) -> Result<Vec<[SubbandData; 4]>, PictureError> {
    let mut py = init_pyramid(comp_width, comp_height, params.dwt_depth);

    // Iterate LL at level 0 first, then HL/LH/HH at each higher level.
    decode_subband(
        r,
        &mut py,
        0,
        Orient::LL,
        comp_width,
        comp_height,
        params,
        using_ac,
    )?;
    for level in 1..=params.dwt_depth {
        for orient in [Orient::HL, Orient::LH, Orient::HH] {
            decode_subband(
                r,
                &mut py,
                level,
                orient,
                comp_width,
                comp_height,
                params,
                using_ac,
            )?;
        }
    }

    // §13.4.2 last line: intra DC prediction on the LL band.
    if is_intra {
        crate::picture::intra_dc_prediction(&mut py[0][0]);
    }

    Ok(py)
}

/// One subband's-worth of coefficient unpacking (§13.4.2).
#[allow(clippy::too_many_arguments)]
fn decode_subband(
    r: &mut BitReader<'_>,
    py: &mut [[SubbandData; 4]],
    level: u32,
    orient: Orient,
    comp_width: u32,
    comp_height: u32,
    params: &CoreTransformParameters,
    using_ac: bool,
) -> Result<(), PictureError> {
    r.byte_align();
    let length = r.read_uint();
    // Subband dims for indexing; data is already zeroed by init_pyramid.
    let (band_w, band_h) = subband_dims(comp_width, comp_height, params.dwt_depth, level);

    if length == 0 {
        r.byte_align();
        return Ok(());
    }

    let quant_index = r.read_uint();
    r.byte_align();

    // The subband's length bytes occupy `[byte_pos .. byte_pos+length]`
    // in the outer reader's buffer. Hand that region to the arith
    // engine (or BoundedBitReader for VLC); on return, skip the outer
    // reader past that region regardless of how many bits the inner
    // engine consumed.
    let start = r.byte_pos();
    let data = r.data();
    let available = data.len().saturating_sub(start);
    let effective_len = (length as usize).min(available);
    let block = &data[start..start + effective_len];

    if using_ac {
        decode_subband_ac(
            block,
            py,
            level,
            orient,
            band_w,
            band_h,
            params,
            quant_index,
        )?;
    } else {
        decode_subband_vlc(
            block,
            effective_len as u64 * 8,
            py,
            level,
            orient,
            band_w,
            band_h,
            params,
            quant_index,
        )?;
    }

    let new_pos = start + (length as usize);
    // Clamp to buffer length; if a broken stream over-reports `length`
    // we still avoid a panic.
    let new_pos = new_pos.min(data.len());
    r.skip_to(new_pos);
    Ok(())
}

/// §13.4.2.2 VLC path — plain bounded exp-Golomb.
#[allow(clippy::too_many_arguments)]
fn decode_subband_vlc(
    block: &[u8],
    total_bits: u64,
    py: &mut [[SubbandData; 4]],
    level: u32,
    orient: Orient,
    band_w: usize,
    band_h: usize,
    params: &CoreTransformParameters,
    quant_index: u32,
) -> Result<(), PictureError> {
    let mut r = crate::bits::BoundedBitReader::new(block, total_bits);
    let q_matrix_entry = params
        .quant_matrix
        .as_ref()
        .map(|m| m.get(level, orient))
        .unwrap_or(0);
    let base_q = quant_index.saturating_sub(q_matrix_entry);
    let (cbx, cby) = cb_counts(params, level);

    for cy in 0..cby {
        for cx in 0..cbx {
            let skipped = if cbx * cby == 1 {
                false
            } else {
                r.read_bitb() == 1
            };
            if skipped {
                continue;
            }
            let mut q = base_q;
            if params.codeblock_mode == 1 {
                let delta = r.read_sintb();
                let next = q as i32 + delta;
                q = if next < 0 { 0 } else { next as u32 };
            }
            let (left, right, top, bottom) =
                codeblock_bounds(cx, cy, cbx, cby, band_w, band_h);
            let band = &mut py[level as usize][orient.as_index()];
            for y in top..bottom {
                for x in left..right {
                    let qc = r.read_sintb();
                    band.set(y, x, inverse_quant(qc, q));
                }
            }
        }
    }
    // f lush_inputb: implicit — dropping the reader discards trailing bits.
    drop(r);
    Ok(())
}

/// §13.4.2.2 AC path — arithmetic-coded codeblocks.
#[allow(clippy::too_many_arguments)]
fn decode_subband_ac(
    block: &[u8],
    py: &mut [[SubbandData; 4]],
    level: u32,
    orient: Orient,
    band_w: usize,
    band_h: usize,
    params: &CoreTransformParameters,
    quant_index: u32,
) -> Result<(), PictureError> {
    let mut bank = ContextBank::new(ctx::NUM_CONTEXTS);
    let mut dec = ArithDecoder::new(block, block.len());

    let q_matrix_entry = params
        .quant_matrix
        .as_ref()
        .map(|m| m.get(level, orient))
        .unwrap_or(0);
    let base_q = quant_index.saturating_sub(q_matrix_entry);
    let (cbx, cby) = cb_counts(params, level);
    let single_cb = cbx * cby == 1;

    for cy in 0..cby {
        for cx in 0..cbx {
            let skipped = if single_cb {
                false
            } else {
                dec.read_bool(&mut bank, ctx::ZERO_BLOCK)
            };
            if skipped {
                continue;
            }
            let mut q = base_q;
            if params.codeblock_mode == 1 {
                // §13.4.3.4: signed quant offset coded against the
                // Q_OFFSET_* contexts (follow / data / sign).
                let delta = dec.read_sint(
                    &mut bank,
                    &[ctx::Q_OFFSET_FOLLOW],
                    ctx::Q_OFFSET_DATA,
                    ctx::Q_OFFSET_SIGN,
                );
                let next = q as i32 + delta;
                q = if next < 0 { 0 } else { next as u32 };
            }
            let (left, right, top, bottom) =
                codeblock_bounds(cx, cy, cbx, cby, band_w, band_h);
            // Split the parent (immutable) and current (mutable) bands
            // using a conditional to avoid aliasing. For level < 2
            // there's no parent — use `None`.
            let (parent_opt, band): (Option<&SubbandData>, &mut SubbandData) = if level >= 2 {
                let (lower, upper) = py.split_at_mut(level as usize);
                (
                    Some(&lower[level as usize - 1][orient.as_index()]),
                    &mut upper[0][orient.as_index()],
                )
            } else {
                (None, &mut py[level as usize][orient.as_index()])
            };
            for y in top..bottom {
                for x in left..right {
                    let parent_zero = match parent_opt {
                        Some(p) => p.get(y / 2, x / 2) == 0,
                        None => true,
                    };
                    let nhood_zero = zero_nhood(band, x, y);
                    let sign_pred = sign_predict(band, orient, x, y);
                    let (follow, data_ctx, sign_ctx) =
                        select_coeff_ctxs(parent_zero, nhood_zero, sign_pred);
                    let qc = dec.read_sint(&mut bank, follow, data_ctx, sign_ctx);
                    band.set(y, x, inverse_quant(qc, q));
                }
            }
        }
    }
    // f lush_inputb: no-op — decoder goes out of scope.
    drop(dec);
    Ok(())
}

fn cb_counts(params: &CoreTransformParameters, level: u32) -> (u32, u32) {
    let idx = level as usize;
    if idx < params.codeblocks.len() {
        let (x, y) = params.codeblocks[idx];
        (x.max(1), y.max(1))
    } else {
        (1, 1)
    }
}

/// §13.4.4.2 zero_nhood — are all of (y-1,x-1), (y,x-1), (y-1,x) zero?
/// Uses the slightly asymmetric edge rule from the spec: on the top row
/// or left column we only check the single available neighbour.
fn zero_nhood(band: &SubbandData, x: usize, y: usize) -> bool {
    if y > 0 && x > 0 {
        band.get(y - 1, x - 1) == 0 && band.get(y, x - 1) == 0 && band.get(y - 1, x) == 0
    } else if y > 0 && x == 0 {
        band.get(y - 1, 0) == 0
    } else if y == 0 && x > 0 {
        band.get(0, x - 1) == 0
    } else {
        true
    }
}

/// §13.4.4.3 sign_predict — depends on subband orientation.
/// Returns -1 / 0 / +1 to match the spec's sign(…) function.
fn sign_predict(band: &SubbandData, orient: Orient, x: usize, y: usize) -> i32 {
    match orient {
        Orient::HL if y > 0 => signum(band.get(y - 1, x)),
        Orient::LH if x > 0 => signum(band.get(y, x - 1)),
        _ => 0,
    }
}

fn signum(v: i32) -> i32 {
    if v > 0 {
        1
    } else if v < 0 {
        -1
    } else {
        0
    }
}

/// §13.4.4.4 Table 13.1. Returns `(follow_ctxs, data_ctx, sign_ctx)`.
fn select_coeff_ctxs(
    parent_zero: bool,
    nhood_zero: bool,
    sign_pred: i32,
) -> (&'static [usize], usize, usize) {
    // The first follow-bin label depends on parent and nhood; bins 2..6+
    // are shared per parent-class. Use static slices.
    static ZP_ZN: [usize; 6] = [
        ctx::ZPZN_F1,
        ctx::ZP_F2,
        ctx::ZP_F3,
        ctx::ZP_F4,
        ctx::ZP_F5,
        ctx::ZP_F6_PLUS,
    ];
    static ZP_NN: [usize; 6] = [
        ctx::ZPNN_F1,
        ctx::ZP_F2,
        ctx::ZP_F3,
        ctx::ZP_F4,
        ctx::ZP_F5,
        ctx::ZP_F6_PLUS,
    ];
    static NP_ZN: [usize; 6] = [
        ctx::NPZN_F1,
        ctx::NP_F2,
        ctx::NP_F3,
        ctx::NP_F4,
        ctx::NP_F5,
        ctx::NP_F6_PLUS,
    ];
    static NP_NN: [usize; 6] = [
        ctx::NPNN_F1,
        ctx::NP_F2,
        ctx::NP_F3,
        ctx::NP_F4,
        ctx::NP_F5,
        ctx::NP_F6_PLUS,
    ];
    let follow: &'static [usize] = match (parent_zero, nhood_zero) {
        (true, true) => &ZP_ZN,
        (true, false) => &ZP_NN,
        (false, true) => &NP_ZN,
        (false, false) => &NP_NN,
    };
    let sign_ctx = if sign_pred == 0 {
        ctx::SIGN_ZERO
    } else if sign_pred < 0 {
        ctx::SIGN_NEG
    } else {
        ctx::SIGN_POS
    };
    (follow, ctx::COEFF_DATA, sign_ctx)
}

/// §11.3.3 codeblock_parameters.
pub fn parse_codeblock_parameters(
    r: &mut BitReader<'_>,
    dwt_depth: u32,
) -> Result<(Vec<(u32, u32)>, u32), PictureError> {
    let mut cb: Vec<(u32, u32)> = Vec::with_capacity(dwt_depth as usize + 1);
    let mut codeblock_mode = 0u32;
    let spatial_partition_flag = r.read_bool();
    if spatial_partition_flag {
        for _level in 0..=dwt_depth {
            let x = r.read_uint().max(1);
            let y = r.read_uint().max(1);
            cb.push((x, y));
        }
        codeblock_mode = r.read_uint();
    } else {
        for _ in 0..=dwt_depth {
            cb.push((1, 1));
        }
    }
    Ok((cb, codeblock_mode))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctx_count_is_22() {
        assert_eq!(ctx::NUM_CONTEXTS, 22);
    }

    #[test]
    fn codeblock_bounds_partitions_subband_exactly() {
        // 32x16 subband, 4x2 codeblocks → each 8x8.
        let cases = [
            ((0, 0), (0, 8, 0, 8)),
            ((1, 0), (8, 16, 0, 8)),
            ((3, 1), (24, 32, 8, 16)),
        ];
        for ((cx, cy), expect) in cases {
            assert_eq!(codeblock_bounds(cx, cy, 4, 2, 32, 16), expect);
        }
    }

    #[test]
    fn select_ctxs_zp_zn_signpos_routes_first_bin() {
        let (follow, data_ctx, sign_ctx) = select_coeff_ctxs(true, true, 1);
        assert_eq!(follow[0], ctx::ZPZN_F1);
        assert_eq!(follow[5], ctx::ZP_F6_PLUS);
        assert_eq!(data_ctx, ctx::COEFF_DATA);
        assert_eq!(sign_ctx, ctx::SIGN_POS);
    }

    #[test]
    fn select_ctxs_np_nn_signneg() {
        let (follow, _data, sign) = select_coeff_ctxs(false, false, -1);
        assert_eq!(follow[0], ctx::NPNN_F1);
        assert_eq!(sign, ctx::SIGN_NEG);
    }

    #[test]
    fn zero_nhood_corners() {
        let mut b = SubbandData::new(3, 3);
        assert!(zero_nhood(&b, 0, 0));
        assert!(zero_nhood(&b, 1, 1));
        b.set(0, 0, 1);
        // Now (1,1)'s neighbour (0,0) is non-zero.
        assert!(!zero_nhood(&b, 1, 1));
        // But (0,1) only checks (0,0) on the top row.
        assert!(!zero_nhood(&b, 1, 0));
    }

    #[test]
    fn sign_predict_orient_sensitive() {
        let mut b = SubbandData::new(3, 3);
        b.set(0, 1, -5);
        // HL at (1,1): look above → -5.
        assert_eq!(sign_predict(&b, Orient::HL, 1, 1), -1);
        // LH at (1,1): look left → band[1][0] = 0.
        assert_eq!(sign_predict(&b, Orient::LH, 1, 1), 0);
        // HH at any position → 0.
        assert_eq!(sign_predict(&b, Orient::HH, 2, 2), 0);
    }

    /// End-to-end smoke test for the VLC path: encode a 2x2 subband's
    /// coefficients as signed exp-Golomb, feed them through
    /// `decode_subband_vlc` and check the recovered values (after
    /// inverse-quantisation with q=0, which is identity for |x| > 0).
    #[test]
    fn vlc_decode_recovers_coefficients() {
        // Bits for sint 2, 0, -1, 3  in exp-Golomb (signed):
        //   2  -> 0 1 1 0        (4 bits)
        //   0  -> 1               (1 bit)
        //   -1 -> 0 0 1 1        (4 bits)
        //   3  -> 0 0 0 0 1 0 0   (7 bits)
        // Concatenated: 0110 1 0011 0000100 = "0110100110000100"
        let bits = "0110100110000100";
        let mut byte_buf: Vec<u8> = Vec::new();
        let padded = {
            let mut s = bits.to_string();
            while s.len() % 8 != 0 {
                s.push('0');
            }
            s
        };
        for chunk in padded.as_bytes().chunks(8) {
            let mut b = 0u8;
            for c in chunk {
                b = (b << 1) | if *c == b'1' { 1 } else { 0 };
            }
            byte_buf.push(b);
        }

        let params = CoreTransformParameters {
            wavelet: WaveletFilter::LeGall5_3,
            dwt_depth: 1,
            codeblocks: vec![(1, 1), (1, 1)],
            codeblock_mode: 0,
            quant_matrix: None,
        };
        // Build a 2x2 LL band at level 0.
        let mut py = vec![
            [
                SubbandData::new(2, 2),
                SubbandData::new(0, 0),
                SubbandData::new(0, 0),
                SubbandData::new(0, 0),
            ],
            [
                SubbandData::new(0, 0),
                SubbandData::new(2, 2),
                SubbandData::new(2, 2),
                SubbandData::new(2, 2),
            ],
        ];
        let total_bits = (byte_buf.len() as u64) * 8;
        decode_subband_vlc(
            &byte_buf,
            total_bits,
            &mut py,
            0,
            Orient::LL,
            2,
            2,
            &params,
            0,
        )
        .unwrap();
        let ll = &py[0][0];
        assert_eq!(ll.get(0, 0), 2);
        assert_eq!(ll.get(0, 1), 0);
        assert_eq!(ll.get(1, 0), -1);
        assert_eq!(ll.get(1, 1), 3);
    }
}
