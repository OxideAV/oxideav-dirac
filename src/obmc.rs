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
    let half = if depth == 0 {
        1i32
    } else {
        1i32 << (depth - 1)
    };
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
pub fn subpel_predict(
    upref: &[i32],
    up_w: usize,
    up_h: usize,
    u: i64,
    v: i64,
    mv_precision: u32,
) -> i32 {
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
#[allow(clippy::too_many_arguments)]
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
    let m: i64 = (1i64 << ep) - (c.0 as i64 * x as i64 + c.1 as i64 * y as i64);
    let ax = a[0][0] as i64 * x as i64 + a[0][1] as i64 * y as i64;
    let ay = a[1][0] as i64 * x as i64 + a[1][1] as i64 * y as i64;
    // The §15.8.8 pseudocode literally writes `2^ez * b[0]` in both the
    // `v[0]` and `v[1]` lines, but `b` (PAN_TILT) is the two-component
    // translation vector of the section's own mathematical model
    // (`v = Ax + b`, with `b = (b[0], b[1])`): the horizontal component
    // takes `b[0]` and the vertical component takes `b[1]`. Reusing
    // `b[0]` for the vertical row is a transcription slip in the
    // pseudocode — we apply the per-axis translation the model requires.
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
    /// Component index for §15.8.5 intra-block DC lookup: `0` = luma (Y),
    /// `1` = first chroma (C1 / U), `2` = second chroma (C2 / V). The
    /// spec reads `state[BLOCK DATA][j][i][dc][c]` indexed by the full
    /// component `c`, so C1 and C2 must select **different** DC values —
    /// collapsing both chroma planes onto index 1 makes every intra
    /// block in the V plane predict from the U plane's DC.
    pub component: usize,
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
                    // §15.8.5: intra blocks use their per-component DC
                    // value directly (no reference weight). `component`
                    // is the full Y/C1/C2 index — C1 and C2 carry
                    // distinct DC values, so this must not collapse both
                    // chroma planes onto a single index.
                    block.dc[params.component] as i64
                }
                RefPredMode::Ref1Only => {
                    let (p1, w1, h1) = ref1_plane.expect("ref1 required");
                    let v = pixel_pred(
                        p1,
                        w1,
                        h1,
                        ref1_upref,
                        block,
                        1,
                        x,
                        y,
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
                        p2,
                        w2,
                        h2,
                        ref2_upref,
                        block,
                        2,
                        x,
                        y,
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
                        p1,
                        w1,
                        h1,
                        ref1_upref,
                        block,
                        1,
                        x,
                        y,
                        params.is_chroma,
                        params.chroma_h_ratio,
                        params.chroma_v_ratio,
                        params.mv_precision,
                        &motion.global1,
                    );
                    let v2 = pixel_pred(
                        p2,
                        w2,
                        h2,
                        ref2_upref,
                        block,
                        2,
                        x,
                        y,
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
            // §15.8.2 OBMC accumulation. The product `val * wmat[..]`
            // is bounded for valid bitstreams (DC ≤ 2^(depth-1), wmat ≤
            // 64, summed across at most 4 overlapping blocks → fits in
            // i32 by margin of ~2^10). On corrupt or unsupported
            // bitstreams `val` can come from a wrapped MV producing an
            // out-of-range pixel prediction; the cast already truncates
            // to i32 silently, but the `+=` itself panics in debug. Use
            // `wrapping_add` so the corrupt path produces wrong-but-
            // bounded output rather than an unrecoverable panic.
            let contrib = (val.saturating_mul(wmat[q * params.xblen + p] as i64)) as i32;
            mc_tmp[idx] = mc_tmp[idx].wrapping_add(contrib);
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
    let ref1_up_ref: Option<(&[i32], usize, usize)> =
        ref1_up.as_ref().map(|(v, w, h)| (v.as_slice(), *w, *h));
    let ref2_up_ref: Option<(&[i32], usize, usize)> =
        ref2_up.as_ref().map(|(v, w, h)| (v.as_slice(), *w, *h));
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
    let half = if bit_depth == 0 {
        1i32
    } else {
        1i32 << (bit_depth - 1)
    };
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
        assert_eq!(hwt, vec![1, 2, 3, 4, 4, 5, 6, 7, 7, 6, 5, 4, 4, 3, 2, 1]);
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
        let r: Vec<i32> = (0..16).collect();
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

    /// §15.8.11 half-pel interpolation on a non-constant single-column
    /// reference, with every value hand-computed from the spec's 8-tap
    /// filter `(-1,3,-7,21,21,-7,3,-1)` + `16` rounding + floor `>> 5`,
    /// clipped to `[-128, 127]`. Also exercises the edge extension: the
    /// top and bottom odd rows read past the array and must clamp to the
    /// nearest in-range sample. This is the sub-pel path the docs-corpus
    /// never validated bit-exactly (its only quarter-pel fixture is
    /// still `Tier::Bounded`), so it is pinned directly against the spec.
    #[test]
    fn interp2by2_halfpel_column_matches_hand_computed_spec() {
        // Single column [10, 20, 40, 80], depth 8.
        let r = vec![10i32, 20, 40, 80];
        let (up, up_w, up_h) = interp2by2(&r, 1, 4, 8);
        assert_eq!((up_w, up_h), (1, 7));
        // Even rows copy; odd rows are the filtered half-pel values:
        //   q=1 (rows 0,1): (16 +21*30 -7*50 +3*90 -1*90) >> 5 = 476>>5 = 14
        //   q=3 (rows 1,2): (16 +21*60 -7*90 +3*90 -1*90) >> 5 = 826>>5 = 25
        //   q=5 (rows 2,3): (16 +21*120 -7*100 +3*90 -1*90) >> 5 = 2016>>5 = 63
        assert_eq!(up, vec![10, 14, 20, 25, 40, 63, 80]);
    }

    /// §15.8.10 quarter-pel sub-pixel prediction: bilinear blend of the
    /// four nearest half-pel samples. Hand-computed for `mv_precision=2`
    /// (`shift=1`, `denom=2`, final `(val + 2) >> 2`) at the (1,1)
    /// quarter-pel offset, where all four weights are 1.
    #[test]
    fn subpel_predict_quarter_pel_matches_hand_computed_spec() {
        // 3x3 half-pel array.
        let up = vec![0i32, 10, 20, 30, 40, 50, 60, 70, 80];
        // u = v = 1 (quarter-pel units): hu=hv=0, ru=rv=1.
        //   w00=w01=w10=w11 = 1
        //   val = up[0]+up[1]+up[3]+up[4] = 0+10+30+40 = 80
        //   (80 + 2) >> 2 = 20
        assert_eq!(subpel_predict(&up, 3, 3, 1, 1, 2), 20);
        // u=3, v=0: hu=1, ru=1, hv=0, rv=0 → w00=2, w01=2, w10=w11=0.
        //   val = 2*up[1] + 2*up[2] = 2*10 + 2*20 = 60
        //   (60 + 2) >> 2 = 15
        assert_eq!(subpel_predict(&up, 3, 3, 3, 0, 2), 15);
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

    /// §15.8.8 `global_mv`: the affine-perspective vector field.
    ///
    /// Each expected value is computed by hand from the §15.8.8
    /// pseudocode (`m = 2^ep - c·x`; `v = m * (A·x + 2^ez * b)`; then
    /// `v = (v + 2^(ez+ep)) >> (ez+ep)`), with the per-axis pan/tilt
    /// translation `b = (b[0], b[1])` taken from the section's own
    /// `v = Ax + b` model rather than the pseudocode's duplicated
    /// `b[0]`.
    #[test]
    fn global_mv_pure_translation_uses_per_axis_pan_tilt() {
        // m = 1 (no perspective), zoom matrix A = 4*I with ez = 2 so the
        // affine part is unit-zoom (2^-ez * A = I). round = 2^2 = 4.
        let g = GlobalParams {
            pan_tilt: (8, -4),
            zrs: [[4, 0], [0, 4]],
            zrs_exp: 2,
            perspective: (0, 0),
            persp_exp: 0,
        };
        // (x, y) = (3, 5):
        //   m   = 2^0 - 0 = 1
        //   ax  = 4*3 + 0*5 = 12 ; ay = 0*3 + 4*5 = 20
        //   v0  = 1 * (12 + 4*8)   = 44 -> (44 + 4) >> 2 = 12
        //   v1  = 1 * (20 + 4*(-4)) = 4 -> ( 4 + 4) >> 2 =  2
        assert_eq!(global_mv(&g, 3, 5), (12, 2));
        // A regression that reused b[0] for the vertical row would give
        // v1 = (20 + 4*8 + 4) >> 2 = 14, which this case rejects.
    }

    #[test]
    fn global_mv_zoom_only_scales_position() {
        // Pure zoom: A = 8*I with ez = 2 -> effective zoom factor 2.
        // No pan/tilt, no perspective.
        let g = GlobalParams {
            pan_tilt: (0, 0),
            zrs: [[8, 0], [0, 8]],
            zrs_exp: 2,
            perspective: (0, 0),
            persp_exp: 0,
        };
        // (x, y) = (10, -6):
        //   m  = 1
        //   ax = 80 -> v0 = 80 -> (80 + 4) >> 2 = 21
        //   ay = -48 -> v1 = -48 -> (-48 + 4) >> 2 = floor(-44/4) = -11
        assert_eq!(global_mv(&g, 10, -6), (21, -11));
    }

    #[test]
    fn global_mv_perspective_modulates_magnitude() {
        // Perspective term: ep = 3 -> m = 2^3 - c·x. round = 2^(ez+ep).
        let g = GlobalParams {
            pan_tilt: (16, 0),
            zrs: [[0, 0], [0, 0]],
            zrs_exp: 1,
            perspective: (2, 0),
            persp_exp: 3,
        };
        // (x, y) = (2, 0):
        //   m  = 2^3 - (2*2 + 0*0) = 8 - 4 = 4
        //   ax = 0 ; v0 = 4 * (0 + 2^1 * 16) = 4 * 32 = 128
        //   v1 = 4 * (0 + 2^1 * 0)  = 0
        //   ez + ep = 4 -> round = 16
        //   v0 = (128 + 16) >> 4 = 144 >> 4 = 9
        //   v1 = (  0 + 16) >> 4 =  16 >> 4 = 1
        assert_eq!(global_mv(&g, 2, 0), (9, 1));
    }

    #[test]
    fn global_mv_origin_is_pure_pan_tilt() {
        // At (0, 0) the affine matrix contributes nothing, so the
        // result is the rounded, perspective-free pan/tilt vector.
        let g = GlobalParams {
            pan_tilt: (12, -20),
            zrs: [[4, 1], [-2, 4]],
            zrs_exp: 2,
            perspective: (5, -3),
            persp_exp: 0,
        };
        // (x, y) = (0, 0):
        //   m  = 2^0 - 0 = 1
        //   ax = 0 ; ay = 0
        //   v0 = 1 * (0 + 2^2 * 12)   = 48 -> (48 + 4) >> 2 = 13
        //   v1 = 1 * (0 + 2^2 * (-20)) = -80 -> (-80 + 4) >> 2 = -19
        assert_eq!(global_mv(&g, 0, 0), (13, -19));
    }

    /// §15.8.7 `pixel_pred` global-motion branch: when a block's
    /// `gmode` flag is set, the predictor must fetch from the reference
    /// using the per-pixel `global_mv(g, x, y)` vector (§15.8.8) and
    /// **ignore** the block's own `mv[ref_num - 1]`. This pins the
    /// `block.gmode` arm of `pixel_pred` that the existing
    /// `motion_compensate_*` tests (all `gmode: false`) never reach.
    #[test]
    fn pixel_pred_global_mode_uses_global_mv_not_block_mv() {
        use crate::picture_inter::{BlockData, GlobalParams, RefPredMode};
        // 8x8 reference: distinct value per cell so a wrong fetch is
        // visible. Stored signed (pre-offset), within 8-bit range.
        let w = 8usize;
        let h = 8usize;
        let mut ref_plane = vec![0i32; w * h];
        for y in 0..h {
            for x in 0..w {
                ref_plane[y * w + x] = (y as i32) * 8 + (x as i32) - 32;
            }
        }
        // Zero affine matrix + zero perspective → global_mv collapses to
        // a *constant* translation across the whole plane:
        //   m = 1; ax = ay = 0; v0 = 1*(0 + 2^ez*b0); round = 2^ez;
        //   v0 = (2^ez*b0 + 2^ez) >> ez = b0 + 1. Same for v1.
        // With pan_tilt = (1, -1) the field is exactly (2, 0) everywhere.
        let g = GlobalParams {
            pan_tilt: (1, -1),
            zrs: [[0, 0], [0, 0]],
            zrs_exp: 0,
            perspective: (0, 0),
            persp_exp: 0,
        };
        assert_eq!(global_mv(&g, 0, 0), (2, 0));
        assert_eq!(global_mv(&g, 5, 3), (2, 0), "field must be constant");
        let global = Some(g);
        // The block carries a *different* mv to prove gmode overrides it.
        let block = BlockData {
            rmode: RefPredMode::Ref1Only,
            gmode: true,
            mv: [(-4, -4); 2],
            dc: [0; 3],
        };
        // mv_precision = 0 → integer-pel fetch: pred = ref[(y+dy)*w +
        // (x+dx)] clamped. With the (2, 0) global field, every interior
        // column reads two pixels to the right of the reference.
        for y in 0..h {
            for x in 0..(w - 2) {
                let pred = pixel_pred(
                    &ref_plane, w, h, None, &block, 1, x as i32, y as i32, false, 1, 1, 0, &global,
                );
                assert_eq!(
                    pred,
                    ref_plane[y * w + (x + 2)],
                    "global fetch mismatch at ({x}, {y})"
                );
            }
        }
        // A regression that used block.mv (-4, -4) instead would clamp
        // to the reference's top-left corner for the whole interior; the
        // distinct-per-cell reference rejects that.
        let pred_via_global = pixel_pred(
            &ref_plane, w, h, None, &block, 1, 4, 4, false, 1, 1, 0, &global,
        );
        assert_ne!(pred_via_global, ref_plane[0], "must not use block.mv");
    }

    /// End-to-end §15.8.2 `motion_compensate` over a fully global-motion
    /// picture: every block has `gmode = true`, so the reconstructed
    /// picture (zero residue) is the reference uniformly shifted by the
    /// constant global field. Exercises the global branch through the
    /// full `block_mc` → OBMC-weighting → accumulation chain, which the
    /// `gmode: false` smoke test never covers.
    #[test]
    fn motion_compensate_global_mode_shifts_reference_uniformly() {
        use crate::picture_inter::{BlockData, GlobalParams, PictureMotionData, RefPredMode};
        let w = 16usize;
        let h = 16usize;
        // Smooth horizontal ramp: constant along rows is irrelevant; the
        // ramp varies with x so a +2 column shift is detectable. Values
        // stay inside the 8-bit signed range, so no clipping occurs.
        let mut ref_plane = vec![0i32; w * h];
        for y in 0..h {
            for x in 0..w {
                ref_plane[y * w + x] = (x as i32) - 8;
            }
        }
        // Constant (2, 0) global field (see the pixel_pred test).
        let g = GlobalParams {
            pan_tilt: (1, -1),
            zrs: [[0, 0], [0, 0]],
            zrs_exp: 0,
            perspective: (0, 0),
            persp_exp: 0,
        };
        assert_eq!(global_mv(&g, 7, 9), (2, 0));
        let blocks_x = 4;
        let blocks_y = 4;
        let mut blocks = Vec::with_capacity((blocks_x * blocks_y) as usize);
        for _ in 0..blocks_x * blocks_y {
            blocks.push(BlockData {
                rmode: RefPredMode::Ref1Only,
                gmode: true,
                // Bogus per-block mv; gmode must override it.
                mv: [(7, 7); 2],
                dc: [0; 3],
            });
        }
        let motion = PictureMotionData {
            blocks_x,
            blocks_y,
            superblocks_x: 1,
            superblocks_y: 1,
            sb_split: vec![2],
            blocks,
            global1: Some(g),
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
            component: 0,
            chroma_h_ratio: 1,
            chroma_v_ratio: 1,
            refs_wt_precision: 1,
            ref1_wt: 1,
            ref2_wt: 1,
            luma_depth: 8,
            chroma_depth: 8,
        };
        let mut pic = vec![0i32; w * h];
        motion_compensate(&mut pic, &params, &motion, Some((&ref_plane, w, h)), None);
        // Interior pixels: the OBMC weights across overlapping blocks sum
        // to 64, so a uniform-shift prediction reproduces exactly. Each
        // interior pixel equals the reference column two to the right.
        for y in 1..(h - 1) {
            for x in 1..(w - 3) {
                assert_eq!(
                    pic[y * w + x],
                    ref_plane[y * w + (x + 2)],
                    "global-shift mismatch at ({x}, {y})"
                );
            }
        }
        // The result must differ from a plain copy of the reference,
        // proving the global field actually displaced the fetch.
        assert_ne!(
            pic[8 * w + 5],
            ref_plane[8 * w + 5],
            "global motion produced no displacement"
        );
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
            component: 0,
            chroma_h_ratio: 1,
            chroma_v_ratio: 1,
            refs_wt_precision: 1,
            ref1_wt: 1,
            ref2_wt: 1,
            luma_depth: 8,
            chroma_depth: 8,
        };
        let mut pic = vec![0i32; w * h];
        motion_compensate(&mut pic, &params, &motion, Some((&ref_plane, w, h)), None);
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

    /// §15.8.5 intra-block DC is indexed by the **full** component
    /// `c` (Y / C1 / C2). An all-intra picture whose blocks carry three
    /// distinct DC values (one per component) must reconstruct each
    /// plane from *its own* DC. This pins the fix for the bug where
    /// `block_mc` collapsed both chroma planes onto DC index 1, so the
    /// C2 (V) plane predicted every intra block from the C1 (U) DC.
    #[test]
    fn intra_block_dc_is_selected_per_component() {
        use crate::picture_inter::{BlockData, PictureMotionData, RefPredMode};
        let w = 16usize;
        let h = 16usize;
        // Distinct DC per component, all inside the 8-bit signed range.
        let dc_y = 11i32;
        let dc_c1 = -20i32;
        let dc_c2 = 47i32;
        let blocks_x = 4;
        let blocks_y = 4;
        let mut blocks = Vec::with_capacity((blocks_x * blocks_y) as usize);
        for _ in 0..blocks_x * blocks_y {
            blocks.push(BlockData {
                rmode: RefPredMode::Intra,
                gmode: false,
                mv: [(0, 0); 2],
                dc: [dc_y, dc_c1, dc_c2],
            });
        }
        let motion = PictureMotionData {
            blocks_x,
            blocks_y,
            superblocks_x: 1,
            superblocks_y: 1,
            sb_split: vec![2],
            blocks,
            global1: None,
            global2: None,
        };
        // Reference planes are irrelevant for intra blocks, but a
        // reference must be supplied for the driver.
        let ref_plane = vec![0i32; w * h];
        for (component, want) in [(0usize, dc_y), (1, dc_c1), (2, dc_c2)] {
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
                is_chroma: component != 0,
                component,
                chroma_h_ratio: 1,
                chroma_v_ratio: 1,
                refs_wt_precision: 1,
                ref1_wt: 1,
                ref2_wt: 1,
                luma_depth: 8,
                chroma_depth: 8,
            };
            let mut pic = vec![0i32; w * h];
            motion_compensate(&mut pic, &params, &motion, Some((&ref_plane, w, h)), None);
            // Every interior pixel reconstructs to this component's DC:
            // the OBMC weights across overlapping all-intra blocks sum to
            // 64, and the residue is zero.
            for y in 1..(h - 1) {
                for x in 1..(w - 1) {
                    assert_eq!(
                        pic[y * w + x],
                        want,
                        "component {component}: pixel ({x},{y}) should reconstruct \
                         to its own DC {want}, got {}",
                        pic[y * w + x]
                    );
                }
            }
        }
    }
}
