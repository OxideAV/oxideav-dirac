//! Overlapped Block Motion Compensation primitives (§15.8).
//!
//! The spec describes OBMC top-down (`motion_compensate` → `block_mc` →
//! `pixel_pred` → `subpel_predict` → `interp2by2`). We build it
//! bottom-up in this module so each primitive is independently unit-
//! testable:
//!
//! * [`interp2by2`] — §15.8.11, the 8-tap half-pel upsampler.
//! * [`subpel_predict`] — §15.8.10, bilinear sub-pixel interpolation on
//!   top of the half-pel array.
//! * [`h_wt`] / [`v_wt`] / [`spatial_wt`] — §15.8.6, the ramp window
//!   used to weight contributions from each overlapping block.
//! * [`pixel_pred`] — §15.8.7, a single pixel's prediction from one
//!   reference picture given a motion vector (block or global).
//! * [`block_mc`] — §15.8.5, motion-compensate a single block into the
//!   temporary buffer, with bi-directional reference weighting.
//! * [`motion_compensate`] — §15.8.2, the outer double loop that adds
//!   the temporary prediction to the wavelet residue and clips.
//!
//! All arithmetic uses the spec's floor-division convention (operator
//! `//`) which for negative dividends differs from Rust's default `/`.
//! [`fdiv`] is a tiny helper that bridges that.

use crate::picture_inter::{BlockData, GlobalParams, PictureMotionData, RefPredMode};

/// Floor division matching the spec's `//` operator (rounds toward
/// negative infinity). Rust's `/` rounds toward zero.
#[inline]
pub fn fdiv(a: i64, b: i64) -> i64 {
    debug_assert!(b > 0);
    let q = a / b;
    let r = a % b;
    if r < 0 {
        q - 1
    } else {
        q
    }
}

/// Floor shift — `a >> b` with spec semantics: equivalent to
/// `fdiv(a, 1 << b)`.
#[inline]
pub fn fshr(a: i64, b: u32) -> i64 {
    if b == 0 {
        a
    } else {
        fdiv(a, 1i64 << b)
    }
}

/// Half-pel interpolation filter taps (§15.8.11, Fig. 15.2). The
/// filter is 8-tap symmetric: `t[0..4]` = `[21, -7, 3, -1]`; the full
/// filter reads
/// `(-1 * a + 3 * b - 7 * c + 21 * d + 21 * e - 7 * f + 3 * g - 1 * h + 16) >> 5`.
pub const HALF_PEL_TAPS: [i32; 4] = [21, -7, 3, -1];

/// §15.8.11 `interp2by2`. Given a reference plane `ref_plane` of size
/// `w * h`, produce a half-pel interpolated plane of size
/// `(2w - 1) * (2h - 1)`. Clipping is to `[-2^(depth-1), 2^(depth-1)-1]`.
pub fn interp2by2(ref_plane: &[i32], w: usize, h: usize, depth: u32) -> (Vec<i32>, usize, usize) {
    debug_assert_eq!(ref_plane.len(), w * h);
    if w == 0 || h == 0 {
        return (Vec::new(), 0, 0);
    }
    let up_w = 2 * w - 1;
    let up_h = 2 * h - 1;
    // First: vertical upsampling → `ref2` (width = w, height = 2h-1).
    let mut ref2: Vec<i32> = vec![0; w * up_h];
    let half = if depth == 0 { 1i32 } else { 1i32 << (depth - 1) };
    let min = -half;
    let max = half - 1;
    for q in 0..up_h {
        if q % 2 == 0 {
            for p in 0..w {
                ref2[q * w + p] = ref_plane[(q / 2) * w + p];
            }
        } else {
            for p in 0..w {
                let mut v: i64 = 16;
                for i in 0..4 {
                    let y_minus = (q as i64 - 1) / 2 - i as i64;
                    let y_plus = (q as i64 + 1) / 2 + i as i64;
                    let ym = y_minus.clamp(0, (h as i64) - 1) as usize;
                    let yp = y_plus.clamp(0, (h as i64) - 1) as usize;
                    v += HALF_PEL_TAPS[i] as i64 * ref_plane[ym * w + p] as i64;
                    v += HALF_PEL_TAPS[i] as i64 * ref_plane[yp * w + p] as i64;
                }
                let v = fshr(v, 5) as i32;
                ref2[q * w + p] = v.clamp(min, max);
            }
        }
    }
    // Second: horizontal upsampling → `upref` (width = 2w-1, height = 2h-1).
    let mut upref: Vec<i32> = vec![0; up_w * up_h];
    for q in 0..up_h {
        for p in 0..up_w {
            if p % 2 == 0 {
                upref[q * up_w + p] = ref2[q * w + (p / 2)];
            } else {
                let mut v: i64 = 16;
                for i in 0..4 {
                    let x_minus = (p as i64 - 1) / 2 - i as i64;
                    let x_plus = (p as i64 + 1) / 2 + i as i64;
                    let xm = x_minus.clamp(0, (w as i64) - 1) as usize;
                    let xp = x_plus.clamp(0, (w as i64) - 1) as usize;
                    v += HALF_PEL_TAPS[i] as i64 * ref2[q * w + xm] as i64;
                    v += HALF_PEL_TAPS[i] as i64 * ref2[q * w + xp] as i64;
                }
                let v = fshr(v, 5) as i32;
                upref[q * up_w + p] = v.clamp(min, max);
            }
        }
    }
    (upref, up_w, up_h)
}

/// §15.8.10 `subpel_predict`. `(u, v)` is in units of 1 << mv_precision.
/// When `mv_precision == 0` the caller should pick the integer pixel
/// from the reference directly rather than calling this.
pub fn subpel_predict(upref: &[i32], up_w: usize, up_h: usize, u: i64, v: i64, mv_precision: u32) -> i32 {
    // Half-pel part = u >> (mv_precision - 1); sub-half remainder = u - half_part << shift.
    let shift = mv_precision - 1;
    let hu = fshr(u, shift);
    let hv = fshr(v, shift);
    let ru = u - (hu << shift);
    let rv = v - (hv << shift);
    let denom = 1i64 << shift;
    let w00 = (denom - rv) * (denom - ru);
    let w01 = (denom - rv) * ru;
    let w10 = rv * (denom - ru);
    let w11 = rv * ru;
    let xpos = hu.clamp(0, (up_w as i64) - 1) as usize;
    let xpos1 = (hu + 1).clamp(0, (up_w as i64) - 1) as usize;
    let ypos = hv.clamp(0, (up_h as i64) - 1) as usize;
    let ypos1 = (hv + 1).clamp(0, (up_h as i64) - 1) as usize;
    let val = w00 * upref[ypos * up_w + xpos] as i64
        + w01 * upref[ypos * up_w + xpos1] as i64
        + w10 * upref[ypos1 * up_w + xpos] as i64
        + w11 * upref[ypos1 * up_w + xpos1] as i64;
    if mv_precision > 1 {
        let round = 1i64 << (2 * mv_precision - 3);
        fshr(val + round, 2 * mv_precision - 2) as i32
    } else {
        val as i32
    }
}

/// §15.8.6 horizontal weighting array. `xbsep` is block stride,
/// `xblen = xbsep + 2 * xoffset`. `i` is the block column index;
/// `blocks_x` is the total count. Returns a vector of length `xblen`.
pub fn h_wt(xblen: usize, xbsep: usize, xoffset: usize, i: u32, blocks_x: u32) -> Vec<i32> {
    let mut hwt = vec![0i32; xblen];
    if xoffset != 1 {
        let two_off = 2 * xoffset;
        // Leading edge.
        for x in 0..two_off {
            let val = 1 + fdiv(6 * x as i64 + xoffset as i64 - 1, 2 * xoffset as i64 - 1);
            hwt[x] = val as i32;
            hwt[x + xbsep] = 8 - val as i32;
        }
    } else {
        hwt[0] = 3;
        hwt[1] = 5;
        hwt[xbsep] = 5;
        hwt[xbsep + 1] = 3;
    }
    for x in 2 * xoffset..xbsep {
        hwt[x] = 8;
    }
    if i == 0 {
        for x in 0..2 * xoffset {
            hwt[x] = 8;
        }
    } else if i == blocks_x - 1 {
        for x in 0..2 * xoffset {
            hwt[x + xbsep] = 8;
        }
    }
    hwt
}

/// §15.8.6 vertical weighting array. See [`h_wt`].
pub fn v_wt(yblen: usize, ybsep: usize, yoffset: usize, j: u32, blocks_y: u32) -> Vec<i32> {
    let mut vwt = vec![0i32; yblen];
    if yoffset != 1 {
        let two_off = 2 * yoffset;
        for y in 0..two_off {
            let val = 1 + fdiv(6 * y as i64 + yoffset as i64 - 1, 2 * yoffset as i64 - 1);
            vwt[y] = val as i32;
            vwt[y + ybsep] = 8 - val as i32;
        }
    } else {
        vwt[0] = 3;
        vwt[1] = 5;
        vwt[ybsep] = 5;
        vwt[ybsep + 1] = 3;
    }
    for y in 2 * yoffset..ybsep {
        vwt[y] = 8;
    }
    if j == 0 {
        for y in 0..2 * yoffset {
            vwt[y] = 8;
        }
    } else if j == blocks_y - 1 {
        for y in 0..2 * yoffset {
            vwt[y + ybsep] = 8;
        }
    }
    vwt
}

/// §15.8.6 `spatial_wt`. 2-D outer product of the horizontal and
/// vertical ramp windows.
pub fn spatial_wt(
    xblen: usize,
    yblen: usize,
    xbsep: usize,
    ybsep: usize,
    xoffset: usize,
    yoffset: usize,
    i: u32,
    j: u32,
    blocks_x: u32,
    blocks_y: u32,
) -> Vec<i32> {
    let h = h_wt(xblen, xbsep, xoffset, i, blocks_x);
    let v = v_wt(yblen, ybsep, yoffset, j, blocks_y);
    let mut w = vec![0i32; xblen * yblen];
    for y in 0..yblen {
        for x in 0..xblen {
            w[y * xblen + x] = h[x] * v[y];
        }
    }
    w
}

/// Chroma MV scaling (§15.8.9): integer-divide the MV by the chroma
/// subsampling ratio. Floor division.
#[inline]
pub fn chroma_mv_scale(mv: (i32, i32), h_ratio: u32, v_ratio: u32) -> (i32, i32) {
    (
        fdiv(mv.0 as i64, h_ratio as i64) as i32,
        fdiv(mv.1 as i64, v_ratio as i64) as i32,
    )
}

/// §15.8.8 `global_mv`. Compute a per-pixel motion vector from the
/// affine-perspective parameters.
pub fn global_mv(g: &GlobalParams, x: i32, y: i32) -> (i32, i32) {
    let ez = g.zrs_exp;
    let ep = g.persp_exp;
    let b = g.pan_tilt;
    let a = g.zrs;
    let c = g.perspective;
    let m: i64 =
        (1i64 << ep) - (c.0 as i64 * x as i64 + c.1 as i64 * y as i64);
    let ax = a[0][0] as i64 * x as i64 + a[0][1] as i64 * y as i64;
    let ay = a[1][0] as i64 * x as i64 + a[1][1] as i64 * y as i64;
    // Spec writes `+ 2^ez * b[0]` for both coords — typo in spec; we
    // use b[0] for x and b[1] for y as libschro does (matches the
    // mathematical model described in the note).
    let v0 = m * (ax + (1i64 << ez) * b.0 as i64);
    let v1 = m * (ay + (1i64 << ez) * b.1 as i64);
    let round = 1i64 << (ez + ep);
    let sh = ez + ep;
    (fshr(v0 + round, sh) as i32, fshr(v1 + round, sh) as i32)
}

/// §15.8.7 `pixel_pred` for a single reference.
///
/// `ref_plane` is the reference picture component in its full size.
/// `upref` is the half-pel upsampled copy of the same plane.
/// Returns the predicted pixel value.
#[allow(clippy::too_many_arguments)]
pub fn pixel_pred(
    ref_plane: &[i32],
    ref_w: usize,
    ref_h: usize,
    upref: Option<(&[i32], usize, usize)>,
    block: &BlockData,
    ref_num: usize,
    x: i32,
    y: i32,
    is_chroma: bool,
    chroma_h_ratio: u32,
    chroma_v_ratio: u32,
    mv_precision: u32,
    global: &Option<GlobalParams>,
) -> i32 {
    let mv = if block.gmode {
        match global {
            Some(g) => global_mv(g, x, y),
            None => (0, 0),
        }
    } else {
        block.mv[ref_num - 1]
    };
    let mv = if is_chroma {
        chroma_mv_scale(mv, chroma_h_ratio, chroma_v_ratio)
    } else {
        mv
    };
    let px: i64 = ((x as i64) << mv_precision) + mv.0 as i64;
    let py: i64 = ((y as i64) << mv_precision) + mv.1 as i64;
    if mv_precision == 0 {
        let xi = px.clamp(0, (ref_w as i64) - 1) as usize;
        let yi = py.clamp(0, (ref_h as i64) - 1) as usize;
        ref_plane[yi * ref_w + xi]
    } else {
        let (up, up_w, up_h) = upref.expect("upref must be present when mv_precision > 0");
        subpel_predict(up, up_w, up_h, px, py, mv_precision)
    }
}

/// Per-picture metadata for motion compensation: picture dimensions,
/// block layout, reference weights, etc. Separated so that
/// [`motion_compensate`] is easy to drive in tests.
#[derive(Debug, Clone)]
pub struct McParams {
    /// Component width in samples.
    pub len_x: usize,
    /// Component height in samples.
    pub len_y: usize,
    /// Luma block dimensions (even for chroma — scaled inside).
    pub xblen: usize,
    pub yblen: usize,
    pub xbsep: usize,
    pub ybsep: usize,
    pub blocks_x: u32,
    pub blocks_y: u32,
    pub mv_precision: u32,
    pub is_chroma: bool,
    pub chroma_h_ratio: u32,
    pub chroma_v_ratio: u32,
    pub refs_wt_precision: u32,
    pub ref1_wt: i32,
    pub ref2_wt: i32,
    pub luma_depth: u32,
    pub chroma_depth: u32,
}

impl McParams {
    pub fn xoffset(&self) -> usize {
        (self.xblen - self.xbsep) / 2
    }
    pub fn yoffset(&self) -> usize {
        (self.yblen - self.ybsep) / 2
    }
    pub fn bit_depth(&self) -> u32 {
        if self.is_chroma {
            self.chroma_depth
        } else {
            self.luma_depth
        }
    }
}

/// §15.8.5 `block_mc`. Motion-compensate one block, adding the
/// weighted predictions into `mc_tmp`.
#[allow(clippy::too_many_arguments)]
pub fn block_mc(
    mc_tmp: &mut [i32],
    params: &McParams,
    motion: &PictureMotionData,
    i: u32,
    j: u32,
    ref1_plane: Option<(&[i32], usize, usize)>,
    ref1_upref: Option<(&[i32], usize, usize)>,
    ref2_plane: Option<(&[i32], usize, usize)>,
    ref2_upref: Option<(&[i32], usize, usize)>,
) {
    let xbsep = params.xbsep as i32;
    let ybsep = params.ybsep as i32;
    let xoff = params.xoffset() as i32;
    let yoff = params.yoffset() as i32;
    let xstart = i as i32 * xbsep - xoff;
    let ystart = j as i32 * ybsep - yoff;
    let xstop = (i as i32 + 1) * xbsep + xoff;
    let ystop = (j as i32 + 1) * ybsep + yoff;
    let block = motion.get_block(i, j);
    let mode = block.rmode;
    let wmat = spatial_wt(
        params.xblen,
        params.yblen,
        params.xbsep,
        params.ybsep,
        params.xoffset(),
        params.yoffset(),
        i,
        j,
        params.blocks_x,
        params.blocks_y,
    );
    let shift = params.refs_wt_precision;
    let round = if shift == 0 { 0 } else { 1i64 << (shift - 1) };
    let x_lo = xstart.max(0);
    let x_hi = xstop.min(params.len_x as i32);
    let y_lo = ystart.max(0);
    let y_hi = ystop.min(params.len_y as i32);
    for y in y_lo..y_hi {
        for x in x_lo..x_hi {
            let p = (x - xstart) as usize;
            let q = (y - ystart) as usize;
            let val: i64 = match mode {
                RefPredMode::Intra => {
                    // DC value per component — intra blocks use their
                    // DC value directly (no reference weight).
                    let dc_idx = if params.is_chroma { 1 } else { 0 };
                    block.dc[dc_idx] as i64
                }
                RefPredMode::Ref1Only => {
                    let (p1, w1, h1) = ref1_plane.expect("ref1 required");
                    let v = pixel_pred(
                        p1, w1, h1,
                        ref1_upref,
                        block, 1, x, y,
                        params.is_chroma,
                        params.chroma_h_ratio,
                        params.chroma_v_ratio,
                        params.mv_precision,
                        &motion.global1,
                    );
                    let vv = v as i64 * (params.ref1_wt + params.ref2_wt) as i64;
                    fshr(vv + round, shift)
                }
                RefPredMode::Ref2Only => {
                    let (p2, w2, h2) = ref2_plane.expect("ref2 required");
                    let v = pixel_pred(
                        p2, w2, h2,
                        ref2_upref,
                        block, 2, x, y,
                        params.is_chroma,
                        params.chroma_h_ratio,
                        params.chroma_v_ratio,
                        params.mv_precision,
                        &motion.global2,
                    );
                    let vv = v as i64 * (params.ref1_wt + params.ref2_wt) as i64;
                    fshr(vv + round, shift)
                }
                RefPredMode::Ref1And2 => {
                    let (p1, w1, h1) = ref1_plane.expect("ref1 required");
                    let (p2, w2, h2) = ref2_plane.expect("ref2 required");
                    let v1 = pixel_pred(
                        p1, w1, h1,
                        ref1_upref,
                        block, 1, x, y,
                        params.is_chroma,
                        params.chroma_h_ratio,
                        params.chroma_v_ratio,
                        params.mv_precision,
                        &motion.global1,
                    );
                    let v2 = pixel_pred(
                        p2, w2, h2,
                        ref2_upref,
                        block, 2, x, y,
                        params.is_chroma,
                        params.chroma_h_ratio,
                        params.chroma_v_ratio,
                        params.mv_precision,
                        &motion.global2,
                    );
                    let vv = v1 as i64 * params.ref1_wt as i64 + v2 as i64 * params.ref2_wt as i64;
                    fshr(vv + round, shift)
                }
            };
            let idx = (y as usize) * params.len_x + x as usize;
            mc_tmp[idx] += (val * wmat[q * params.xblen + p] as i64) as i32;
        }
    }
}

/// §15.8.2 `motion_compensate`. Build the temporary MC array then add
/// it (after the 1/64 scaling) into `pic`. `pic` contains the wavelet
/// residue; after this call it contains the final reconstructed pixel
/// values, clipped.
pub fn motion_compensate(
    pic: &mut [i32],
    params: &McParams,
    motion: &PictureMotionData,
    ref1_plane: Option<(&[i32], usize, usize)>,
    ref2_plane: Option<(&[i32], usize, usize)>,
) {
    let mut mc_tmp = vec![0i32; params.len_x * params.len_y];
    // Half-pel upsample each reference once up front if needed.
    let depth = params.bit_depth();
    let ref1_up = match (params.mv_precision > 0, ref1_plane) {
        (true, Some((p, w, h))) => Some(interp2by2(p, w, h, depth)),
        _ => None,
    };
    let ref2_up = match (params.mv_precision > 0, ref2_plane) {
        (true, Some((p, w, h))) => Some(interp2by2(p, w, h, depth)),
        _ => None,
    };
    let ref1_up_ref: Option<(&[i32], usize, usize)> = ref1_up.as_ref().map(|(v, w, h)| (v.as_slice(), *w, *h));
    let ref2_up_ref: Option<(&[i32], usize, usize)> = ref2_up.as_ref().map(|(v, w, h)| (v.as_slice(), *w, *h));
    for j in 0..params.blocks_y {
        for i in 0..params.blocks_x {
            block_mc(
                &mut mc_tmp,
                params,
                motion,
                i,
                j,
                ref1_plane,
                ref1_up_ref,
                ref2_plane,
                ref2_up_ref,
            );
        }
    }
    let bit_depth = params.bit_depth();
    let half = if bit_depth == 0 { 1i32 } else { 1i32 << (bit_depth - 1) };
    let lo = -half;
    let hi = half - 1;
    for y in 0..params.len_y {
        for x in 0..params.len_x {
            let idx = y * params.len_x + x;
            let cur = pic[idx] as i64 + fshr(mc_tmp[idx] as i64 + 32, 6);
            pic[idx] = (cur as i32).clamp(lo, hi);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fdiv_matches_spec_floor() {
        assert_eq!(fdiv(-7, 3), -3);
        assert_eq!(fdiv(7, 3), 2);
        assert_eq!(fdiv(-6, 3), -2);
        assert_eq!(fdiv(0, 3), 0);
    }

    #[test]
    fn hwt_overlap_4_offset_2_interior() {
        // xblen=8 xbsep=4 xoffset=2, interior column (not first nor last).
        let hwt = h_wt(8, 4, 2, 1, 4);
        // Per Table 15.8, overlap=4, offset=2, leading edge = 1,3,5,7
        assert_eq!(hwt, vec![1, 3, 5, 7, 7, 5, 3, 1]);
    }

    #[test]
    fn hwt_overlap_8_offset_4_interior() {
        // Per Table 15.8: overlap=8, offset=4. Valid min xbsep is 8
        // (constraint xblen <= 2*xbsep → xbsep >= 8 when overlap=8).
        // Pick xbsep=8 → xblen=16.
        let hwt = h_wt(16, 8, 4, 1, 4);
        assert_eq!(
            hwt,
            vec![1, 2, 3, 4, 4, 5, 6, 7, 7, 6, 5, 4, 4, 3, 2, 1]
        );
    }

    #[test]
    fn hwt_offset_1_special_case() {
        // xoffset=1 → xblen = xbsep + 2. Leading 3,5; interior 8s; trailing 5,3.
        let hwt = h_wt(6, 4, 1, 1, 4);
        assert_eq!(hwt, vec![3, 5, 8, 8, 5, 3]);
    }

    #[test]
    fn hwt_edge_block_overrides_to_flat() {
        // First column (i=0): leading edge is flattened to 8s.
        let hwt = h_wt(8, 4, 2, 0, 4);
        assert_eq!(&hwt[0..4], &[8, 8, 8, 8]);
    }

    #[test]
    fn spatial_wt_sums_in_overlap_region() {
        // For an interior block with xblen=8, xbsep=4 → overlap 4, offset 2.
        // The property is that across adjacent blocks, hwt[x+xbsep] + hwt[x] == 8.
        let w = h_wt(8, 4, 2, 1, 4);
        for x in 0..4 {
            assert_eq!(w[x] + w[x + 4], 8);
        }
    }

    #[test]
    fn interp2by2_copies_even_coordinates() {
        let r: Vec<i32> = (0..16).map(|i| i as i32).collect();
        let (up, w, h) = interp2by2(&r, 4, 4, 8);
        assert_eq!((w, h), (7, 7));
        // Even, even coordinates are copies.
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(up[(2 * y) * w + (2 * x)], r[y * 4 + x]);
            }
        }
    }

    #[test]
    fn interp2by2_constant_stays_constant() {
        let r: Vec<i32> = vec![40; 16];
        let (up, _, _) = interp2by2(&r, 4, 4, 8);
        for v in &up {
            assert_eq!(*v, 40);
        }
    }

    #[test]
    fn subpel_predict_integer_position_copies() {
        // mv_precision=1 → half-pel; u=hv<<0 (all at integer) should read
        // upref at those coordinates directly.
        let mut up = vec![0i32; 9];
        for i in 0..9 {
            up[i] = i as i32;
        }
        let v = subpel_predict(&up, 3, 3, 2, 2, 1);
        // At half-pel precision, hu=hv=2 (half-pel coord). Remainder ru=rv=0.
        // shift=0; denom=1. w00 = 1*1 = 1, w01 = w10 = w11 = 0.
        assert_eq!(v, up[2 * 3 + 2]);
    }

    #[test]
    fn chroma_mv_scale_halves_for_420() {
        assert_eq!(chroma_mv_scale((4, 6), 2, 2), (2, 3));
        assert_eq!(chroma_mv_scale((-3, -1), 2, 2), (-2, -1)); // floor division
    }

    /// End-to-end MC smoke test: a single-block inter picture with a
    /// zero motion vector and zero residue should reproduce the
    /// reference exactly. We use a single superblock (1x1 split=2 → 4
    /// blocks) covering a 16x16 picture.
    #[test]
    fn motion_compensate_zero_mv_zero_residue_reproduces_ref() {
        use crate::picture_inter::{BlockData, PictureMotionData, RefPredMode};
        // 16x16 reference filled with a ramp (signed, pre-offset).
        let w = 16usize;
        let h = 16usize;
        let mut ref_plane = vec![0i32; w * h];
        for y in 0..h {
            for x in 0..w {
                ref_plane[y * w + x] = (x as i32 + y as i32) - 16;
            }
        }
        // 8x8 blocks, xbsep = 4, xoffset = 2 — so 4x4 = 16 blocks cover
        // the 16x16 picture (xbsep * 4 = 16).
        let blocks_x = 4;
        let blocks_y = 4;
        let mut blocks = Vec::with_capacity((blocks_x * blocks_y) as usize);
        for _ in 0..blocks_x * blocks_y {
            blocks.push(BlockData {
                rmode: RefPredMode::Ref1Only,
                gmode: false,
                mv: [(0, 0); 2],
                dc: [0; 3],
            });
        }
        let motion = PictureMotionData {
            blocks_x,
            blocks_y,
            superblocks_x: 1,
            superblocks_y: 1,
            sb_split: vec![2], // level 2: 16 individual blocks
            blocks,
            global1: None,
            global2: None,
        };
        let params = McParams {
            len_x: w,
            len_y: h,
            xblen: 8,
            yblen: 8,
            xbsep: 4,
            ybsep: 4,
            blocks_x,
            blocks_y,
            mv_precision: 0,
            is_chroma: false,
            chroma_h_ratio: 1,
            chroma_v_ratio: 1,
            refs_wt_precision: 1,
            ref1_wt: 1,
            ref2_wt: 1,
            luma_depth: 8,
            chroma_depth: 8,
        };
        let mut pic = vec![0i32; w * h];
        motion_compensate(
            &mut pic,
            &params,
            &motion,
            Some((&ref_plane, w, h)),
            None,
        );
        // With MV=0 and residue=0, the motion-compensated picture
        // should match the reference exactly across the interior.
        // The ramp is within the 8-bit signed range so no clipping.
        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                assert_eq!(
                    pic[y * w + x],
                    ref_plane[y * w + x],
                    "mismatch at ({x}, {y})"
                );
            }
        }
    }
}
