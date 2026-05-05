//! Dirac core-syntax **inter** encoder.
//!
//! Mirrors [`crate::picture_inter::decode_block_motion_data`] +
//! [`crate::picture::decode_picture_with_refs`] on the encode side.
//! Current shape (post-#168 sub-pel ME):
//!
//! * **Single reference** (parse code `0x09` — non-reference 1-ref AC
//!   inter picture). One `picture_number` delta, no `retd`.
//! * **No global motion**, no reference weights override (`refs_wt =
//!   1 / 1`, `refs_wt_precision = 1`).
//! * **Quarter-pel motion** (`mv_precision = 2` by default; configurable
//!   via [`InterEncoderParams::mv_precision`]) — integer-pel SAD search
//!   over `[-mv_search_range, +mv_search_range]` followed by an 8-neighbor
//!   gradient refinement at each successive sub-pel level (half-pel,
//!   quarter-pel) using the spec's §15.8.10 / §15.8.11 interpolation
//!   filter (8-tap half-pel + bilinear sub-half). Setting `mv_precision = 0`
//!   reverts to integer-pel-only ME.
//! * **Block grid**: preset 1 from Table 11.1 — 8x8 blocks with 4x4
//!   stride. Two-element block_data context — uniform splits, no
//!   superblock subdivisions.
//! * **Zero-residual** — we set the §11.3 `ZERO_RESIDUAL = true` flag
//!   so the encoder emits no wavelet-coefficient stream. Reconstruction
//!   is therefore the OBMC of the reference (with whatever MV grid the
//!   ME chose) plus zero residual. PSNR is dominated by motion
//!   estimation quality.
//!
//! Motion estimation: integer-pel full-search SAD over a
//! `(mv_search_range)`-pel window followed by per-level 8-neighbor
//! refinement to the configured sub-pel precision. r3 will add OBMC
//! overlap and 2-ref bipred.

use crate::arith::{ArithEncoder, ContextBank};
use crate::bitwriter::BitWriter;
use crate::encoder::write_parse_info;
use crate::obmc::{interp2by2, spatial_wt, subpel_predict, McParams};
use crate::picture_core::ctx;
use crate::picture_inter::{
    mvctx, BlockData, PictureMotionData, PicturePredictionParams, RefPredMode,
};
use crate::quant::quant_factor;
use crate::sequence::SequenceHeader;
use crate::subband::{padded_component_dims, Orient, SubbandData};
use crate::video_format::ChromaFormat;
use crate::wavelet::{dwt, WaveletFilter};

/// Encoder-side inter parameters.
#[derive(Debug, Clone)]
pub struct InterEncoderParams {
    /// Block-parameters preset index (Table 11.1). Only preset 1
    /// (xblen=yblen=8, xbsep=ybsep=4) is supported today.
    pub block_params_index: u32,
    /// MV search half-window in **luma pels**. Integer-pel SAD search
    /// runs over `[-r, +r]` in each direction.
    pub mv_search_range: u32,
    /// Motion-vector precision for **1-ref (P-picture)** paths (§11.2.5):
    /// `0`=integer, `1`=half-pel, `2`=quarter-pel, `3`=eighth-pel.
    /// At precision `p`, encoded MV components are in units of
    /// `1/(2^p)` luma pels and the encoder runs `p` rounds of 8-neighbor
    /// sub-pel refinement after the integer search using the §15.8 filter.
    pub mv_precision: u32,
    /// Motion-vector precision for **2-ref (B-picture / bipred)** paths.
    ///
    /// Empirically, integer-pel (`0`) gives significantly higher
    /// ffmpeg cross-decode PSNR (~50 dB) than quarter-pel (`2`, ~42 dB)
    /// on complementary-bar fixtures. The gain comes from eliminating
    /// sub-pel interpolation convention differences between our
    /// OBMC implementation and ffmpeg's at the 2-ref blend stage.
    /// The wavelet residue then closes the prediction-error loop
    /// exactly, making the extra interpolation noise unnecessary.
    ///
    /// Defaults to `0` (integer-pel). Set to `2` to use the same
    /// quarter-pel ME as the 1-ref path (at a ~8 dB cross-decode cost).
    pub bipred_mv_precision: u32,
    /// **OBMC-aware ME refinement passes** (#186, §15.8.6).
    ///
    /// After the per-block sub-pel SAD search picks an initial MV grid,
    /// run this many full passes of OBMC-aware refinement: for each
    /// block, score the 8 sub-pel neighbours of its current MV by
    /// rebuilding the §15.8.5 weighted-sum reconstruction over the
    /// block's `(xblen × yblen)` extent — i.e. the same blend the
    /// decoder will perform — and keep the MV that minimises sum of
    /// squared error against the source. Two passes typically suffice
    /// to converge on smooth motion fields; setting `0` disables OBMC
    /// refinement and reverts to the pre-#186 hard-block SAD output
    /// (useful as the no-OBMC baseline in regression tests).
    pub obmc_refine_passes: u32,
    /// **Wavelet residue** (§11.3 / §13.4).
    ///
    /// When `Some(residue)`, the encoder emits the §11.3 ZERO_RESIDUAL
    /// flag as `false`, computes `source - decoder_OBMC_reconstruction`
    /// in the spec's signed pre-output-offset domain, forward-transforms
    /// the difference (single-codeblock per subband, no per-codeblock
    /// quant offset, AC-coded), and emits the per-component subband
    /// blocks. The decoder adds this back to its OBMC reconstruction at
    /// §15.8.2 — closing the prediction-error loop.
    ///
    /// `None` (or `enable_residue = false`) keeps the round-1 behaviour:
    /// `ZERO_RESIDUAL = true`, no transform parameters, no coefficient
    /// stream. PSNR is then determined entirely by ME quality.
    ///
    /// Residue encoding lifts the inter-encoder's quality ceiling
    /// dramatically on real-world content (where ME alone leaves
    /// edge-clamp / OBMC-blend residuals across block boundaries).
    pub residue: Option<ResidueParams>,
}

/// Wavelet residue encoder parameters. Mirrors
/// [`crate::encoder_intra_core::CoreIntraEncoderParams`] but for the
/// inter-residue path: same `WaveletFilter` and `dwt_depth` knobs, plus
/// a single per-picture `qindex` (no Annex E.1 weighting).
#[derive(Debug, Clone)]
pub struct ResidueParams {
    /// DWT filter for the residue. LeGall 5/3 is bit-reversible at
    /// `qindex = 0` so the encode + AC + IDWT round-trip is exact when
    /// the residue values fit in the dynamic range of the filter.
    pub wavelet: WaveletFilter,
    /// Number of DWT levels. Same constraint as the intra path:
    /// component sizes must accommodate `2^dwt_depth` rounding
    /// (handled by [`crate::subband::padded_component_dims`]).
    pub dwt_depth: u32,
    /// Per-subband quantisation index (`0..=127`). `0` is near-lossless
    /// for small coefficients; the encoder + decoder use the same
    /// dead-zone forward / inverse pair as the intra path.
    pub qindex: u32,
}

impl ResidueParams {
    /// Default residue parameters: LeGall 5/3 at depth 3 with
    /// `qindex = 0` (near-lossless on small residues, matches the
    /// intra encoder's default).
    pub fn default_for(wavelet: WaveletFilter, dwt_depth: u32) -> Self {
        Self {
            wavelet,
            dwt_depth,
            qindex: 0,
        }
    }
}

impl Default for InterEncoderParams {
    fn default() -> Self {
        // Quarter-pel by default — a good speed/accuracy trade-off and
        // the most common `mv_precision` in real Dirac streams. Uses
        // the spec's §15.8.11 8-tap half-pel filter once for the
        // half-pel grid plus a single bilinear refinement to quarter.
        // Two OBMC refinement passes — empirically converges on the
        // small fixtures and lifts ffmpeg cross-decode by ≥ 5 dB on
        // the camera-pan fixture (#186 acceptance).
        // Wavelet residue on by default: LeGall 5/3 / depth 3 /
        // qindex=0. Set `residue = None` to revert to the round-1
        // ZERO_RESIDUAL=true path (used by the no-residue regression
        // tests).
        Self {
            block_params_index: 1,
            mv_search_range: 16,
            mv_precision: 2,
            // Integer-pel for bipred: eliminates sub-pel interpolation
            // convention differences with ffmpeg at the 2-ref OBMC blend
            // stage, lifting ffmpeg cross-decode PSNR from ~42 dB to ~50 dB.
            bipred_mv_precision: 0,
            obmc_refine_passes: 2,
            residue: Some(ResidueParams::default_for(WaveletFilter::LeGall5_3, 3)),
        }
    }
}

/// One input picture (Y/U/V planes + picture number).
#[derive(Debug, Clone)]
pub struct InterInputPicture<'a> {
    pub picture_number: u32,
    pub y: &'a [u8],
    pub u: &'a [u8],
    pub v: &'a [u8],
}

/// Block dims for preset 1 (Table 11.1) — the only preset r1 emits.
const PRESET1: (u32, u32, u32, u32) = (8, 8, 4, 4);

/// Build the §11.2 superblock / block grid sizes for `(luma_w, luma_h)`
/// at preset 1. Mirrors [`crate::picture_inter::parse_picture_prediction_parameters`]
/// motion_data_dimensions block.
fn motion_grid(luma_w: u32, luma_h: u32) -> (u32, u32, u32, u32) {
    let (_xblen, _yblen, xbsep, ybsep) = PRESET1;
    let four_xbsep = 4 * xbsep.max(1);
    let four_ybsep = 4 * ybsep.max(1);
    let superblocks_x = luma_w.div_ceil(four_xbsep);
    let superblocks_y = luma_h.div_ceil(four_ybsep);
    let blocks_x = 4 * superblocks_x;
    let blocks_y = 4 * superblocks_y;
    (superblocks_x, superblocks_y, blocks_x, blocks_y)
}

/// Per-block sub-pel motion vector in units of `1/(2^mv_precision)` luma
/// pels. For `mv_precision = 0` this is identical to integer-pel MVs.
/// Indexed `[by * blocks_x + bx]`.
///
/// Name retained from the integer-pel-only era of the encoder; the
/// component values are simply scaled (or refined) according to the
/// active `mv_precision`.
#[derive(Debug, Clone, Copy, Default)]
pub struct IntegerMv(pub i32, pub i32);

/// Full-search SAD motion estimation over a `[-search, +search]` luma-pel
/// window. Returns one MV per `bsep`-strided block (preset 1: 4-pel
/// stride, 8x8 block). The block at position `(bx, by)` covers source
/// pixels `[by*4 .. by*4+8) x [bx*4 .. bx*4+8)` (truncated against the
/// picture edge).
pub fn full_search_me(
    cur_y: &[u8],
    ref_y: &[u8],
    width: u32,
    height: u32,
    blocks_x: u32,
    blocks_y: u32,
    search: u32,
) -> Vec<IntegerMv> {
    let mut out = vec![IntegerMv::default(); (blocks_x * blocks_y) as usize];
    let (_xblen, _yblen, xbsep, ybsep) = PRESET1;
    let xblen = 8u32;
    let yblen = 8u32;
    let w = width as i32;
    let h = height as i32;
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let x0 = (bx * xbsep) as i32;
            let y0 = (by * ybsep) as i32;
            let mut best_sad = i64::MAX;
            let mut best_mv = (0i32, 0i32);
            for dy in -(search as i32)..=(search as i32) {
                for dx in -(search as i32)..=(search as i32) {
                    let sad = sad_block(
                        cur_y,
                        ref_y,
                        w,
                        h,
                        x0,
                        y0,
                        dx,
                        dy,
                        xblen as i32,
                        yblen as i32,
                    );
                    // Tiny tie-break preferring (0, 0) so motion-free
                    // areas stay zero-MV (smaller residuals).
                    let lambda = (dx.unsigned_abs() + dy.unsigned_abs()) as i64;
                    let cost = sad + lambda;
                    if cost < best_sad {
                        best_sad = cost;
                        best_mv = (dx, dy);
                    }
                }
            }
            out[(by * blocks_x + bx) as usize] = IntegerMv(best_mv.0, best_mv.1);
        }
    }
    out
}

/// Sum-of-absolute-differences for an `(xblen x yblen)` block at source
/// position `(x0, y0)` against reference position `(x0 + dx, y0 + dy)`,
/// with edge clamping on both pictures.
#[allow(clippy::too_many_arguments)]
fn sad_block(
    cur: &[u8],
    refp: &[u8],
    w: i32,
    h: i32,
    x0: i32,
    y0: i32,
    dx: i32,
    dy: i32,
    xblen: i32,
    yblen: i32,
) -> i64 {
    let mut sad: i64 = 0;
    for y in 0..yblen {
        let cy = (y0 + y).clamp(0, h - 1) as usize;
        let ry = (y0 + dy + y).clamp(0, h - 1) as usize;
        for x in 0..xblen {
            let cx = (x0 + x).clamp(0, w - 1) as usize;
            let rx = (x0 + dx + x).clamp(0, w - 1) as usize;
            let a = cur[cy * w as usize + cx] as i32;
            let b = refp[ry * w as usize + rx] as i32;
            sad += (a - b).unsigned_abs() as i64;
        }
    }
    sad
}

/// SAD between an `xblen × yblen` source block at `(x0, y0)` in `cur`
/// and a sub-pel-sampled reference block whose top-left lies at
/// `(qx, qy)` in `1/(2^mv_precision)`-pel units. Reference samples are
/// read via [`subpel_predict`] (§15.8.10) on the half-pel-upsampled
/// `upref` plane (§15.8.11) — the same path the decoder uses on the
/// reverse side, so SAD evaluated here matches the eventual
/// reconstruction error pixel-for-pixel.
#[allow(clippy::too_many_arguments)]
fn sad_subpel(
    cur: &[u8],
    upref: &[i32],
    up_w: usize,
    up_h: usize,
    w: i32,
    h: i32,
    x0: i32,
    y0: i32,
    qx: i64,
    qy: i64,
    xblen: i32,
    yblen: i32,
    mv_precision: u32,
) -> i64 {
    let mut sad: i64 = 0;
    let scale = 1i64 << mv_precision;
    for y in 0..yblen {
        let cy = (y0 + y).clamp(0, h - 1) as usize;
        let py = qy + (y as i64) * scale;
        for x in 0..xblen {
            let cx = (x0 + x).clamp(0, w - 1) as usize;
            let px = qx + (x as i64) * scale;
            let r = subpel_predict(upref, up_w, up_h, px, py, mv_precision);
            let a = cur[cy * w as usize + cx] as i32;
            sad += (a - r).unsigned_abs() as i64;
        }
    }
    sad
}

/// Build the half-pel upsampled reference plane (§15.8.11) for a
/// `width × height` u8 luma plane. The returned plane has size
/// `(2w - 1) × (2h - 1)` in i32. We use `depth = 9` so that the spec's
/// `[-2^(d-1), 2^(d-1)-1]` clip range (`[-256, 255]`) is wide enough
/// to hold any 0..255 input plus the 8-tap filter's small overshoot.
///
/// The samples are kept in unsigned `0..255` space rather than signed
/// `-128..127`; this matches the original sub-pel ME path (which scores
/// against the u8 source directly). The OBMC-aware refinement
/// ([`obmc_refine_me`]) builds its own *signed* upref via
/// [`build_upref_signed`] because the decoder OBMC blend operates on the
/// pre-offset signed reference buffer (§15.4 / §15.8.5).
pub(crate) fn build_upref(plane: &[u8], width: u32, height: u32) -> (Vec<i32>, usize, usize) {
    let w = width as usize;
    let h = height as usize;
    let signed: Vec<i32> = plane.iter().map(|&v| v as i32).collect();
    let (up, up_w, up_h) = interp2by2(&signed, w, h, 9);
    (up, up_w, up_h)
}

/// Like [`build_upref`] but with the spec's signed pre-offset
/// convention: each input sample is shifted by `-2^(depth-1)` (i.e.
/// `-128` for 8-bit) to match what the decoder's reference buffer
/// holds (§15.4 stores the pre-output-offset signed plane).
pub(crate) fn build_upref_signed(
    plane: &[u8],
    width: u32,
    height: u32,
) -> (Vec<i32>, usize, usize) {
    let w = width as usize;
    let h = height as usize;
    let signed: Vec<i32> = plane.iter().map(|&v| v as i32 - 128).collect();
    let (up, up_w, up_h) = interp2by2(&signed, w, h, 9);
    (up, up_w, up_h)
}

/// Sub-pel motion estimation: integer-pel SAD followed by per-level
/// 8-neighbor refinement down to `mv_precision` (in `1/(2^p)` pel units).
///
/// Strategy:
/// 1. Run [`full_search_me`] to find the best integer-pel MV for each
///    8x8 block.
/// 2. Promote to `mv_precision` units (multiply by `2^mv_precision`).
/// 3. For each refinement level `lvl = mv_precision .. 1` (i.e. a
///    half-pel step at level `mv_precision-1`, then quarter-pel at
///    `mv_precision-2`, etc., down to the finest level), test the 8
///    neighbours of the current best at step size `2^(lvl-1)` units.
///    Accept the lowest-SAD neighbour. This is the spec-spirit
///    "log search" — O(8 · mv_precision) sub-pel evaluations per block
///    instead of the O((2 · 2^p)^2) of a full-search.
///
/// At `mv_precision = 0` this degenerates to [`full_search_me`].
#[allow(clippy::too_many_arguments)]
pub fn subpel_search_me(
    cur_y: &[u8],
    ref_y: &[u8],
    width: u32,
    height: u32,
    blocks_x: u32,
    blocks_y: u32,
    search: u32,
    mv_precision: u32,
) -> Vec<IntegerMv> {
    let int_mvs = full_search_me(cur_y, ref_y, width, height, blocks_x, blocks_y, search);
    if mv_precision == 0 {
        return int_mvs;
    }
    let (xblen, yblen) = (8i32, 8i32);
    let (xbsep, ybsep) = (4i32, 4i32);
    let w = width as i32;
    let h = height as i32;
    let scale = 1i64 << mv_precision;
    let (up, up_w, up_h) = build_upref(ref_y, width, height);
    let mut out = vec![IntegerMv::default(); int_mvs.len()];
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let idx = (by * blocks_x + bx) as usize;
            let int_mv = int_mvs[idx];
            let x0 = bx as i32 * xbsep;
            let y0 = by as i32 * ybsep;
            // Promote integer-pel MV into 1/(2^p)-pel units.
            let mut best_qx = (int_mv.0 as i64) * scale;
            let mut best_qy = (int_mv.1 as i64) * scale;
            // Reference top-left for this block in sub-pel units.
            let mut q_ref_x = (x0 as i64) * scale + best_qx;
            let mut q_ref_y = (y0 as i64) * scale + best_qy;
            let mut best_sad = sad_subpel(
                cur_y,
                &up,
                up_w,
                up_h,
                w,
                h,
                x0,
                y0,
                q_ref_x,
                q_ref_y,
                xblen,
                yblen,
                mv_precision,
            );
            // Tiny lambda toward the integer center keeps motion-free
            // areas at zero-MV (matches the integer-pel pass behaviour).
            best_sad += (best_qx.unsigned_abs() + best_qy.unsigned_abs()) as i64;

            // Per-level 8-neighbor refinement. Step size halves each
            // level until we reach 1 sub-pel unit (the finest).
            let mut step: i64 = 1i64 << (mv_precision - 1);
            while step >= 1 {
                let mut improved = true;
                while improved {
                    improved = false;
                    let mut best_dx = 0i64;
                    let mut best_dy = 0i64;
                    for dy in [-step, 0, step] {
                        for dx in [-step, 0, step] {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let cand_qx = best_qx + dx;
                            let cand_qy = best_qy + dy;
                            let cand_ref_x = q_ref_x + dx;
                            let cand_ref_y = q_ref_y + dy;
                            let mut sad = sad_subpel(
                                cur_y,
                                &up,
                                up_w,
                                up_h,
                                w,
                                h,
                                x0,
                                y0,
                                cand_ref_x,
                                cand_ref_y,
                                xblen,
                                yblen,
                                mv_precision,
                            );
                            sad += (cand_qx.unsigned_abs() + cand_qy.unsigned_abs()) as i64;
                            if sad < best_sad {
                                best_sad = sad;
                                best_dx = dx;
                                best_dy = dy;
                                improved = true;
                            }
                        }
                    }
                    if improved {
                        best_qx += best_dx;
                        best_qy += best_dy;
                        q_ref_x += best_dx;
                        q_ref_y += best_dy;
                    }
                }
                step >>= 1;
            }
            out[idx] = IntegerMv(best_qx as i32, best_qy as i32);
        }
    }
    out
}

// ---- OBMC-aware ME refinement (#186, §15.8.6) ------------------------

/// Pre-baked spatial weight matrix for one block at grid position
/// `(i, j)`. `xblen × yblen` row-major, sums to `xblen * yblen` over the
/// block's extent (8 × 8 = 64 for preset 1). The matrix already encodes
/// the §15.8.6 edge-block flattening (first/last column or row gets a
/// flat `8`).
#[allow(clippy::too_many_arguments)]
fn block_weight(
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
    spatial_wt(
        xblen, yblen, xbsep, ybsep, xoffset, yoffset, i, j, blocks_x, blocks_y,
    )
}

/// Compute the §15.8.10 sub-pel reference value at picture coordinate
/// `(x, y)` given a sub-pel MV `(qx, qy)` in `1/(2^mv_precision)` units.
/// Mirrors the inner core of [`crate::obmc::pixel_pred`] for the 1-ref
/// integer-coordinate path.
#[inline]
#[allow(clippy::too_many_arguments)]
fn ref_pixel_at(
    upref: &[i32],
    up_w: usize,
    up_h: usize,
    refp: &[i32],
    ref_w: usize,
    ref_h: usize,
    mv_precision: u32,
    x: i32,
    y: i32,
    qx: i32,
    qy: i32,
) -> i32 {
    if mv_precision == 0 {
        let xi = (x + qx).clamp(0, (ref_w as i32) - 1) as usize;
        let yi = (y + qy).clamp(0, (ref_h as i32) - 1) as usize;
        refp[yi * ref_w + xi]
    } else {
        let px = ((x as i64) << mv_precision) + qx as i64;
        let py = ((y as i64) << mv_precision) + qy as i64;
        subpel_predict(upref, up_w, up_h, px, py, mv_precision)
    }
}

/// Build the `xblen × yblen` weighted prediction `w_ij(x, y) * pred_ij(x, y)`
/// for block `(i, j)` over its `[xstart, xstop) × [ystart, ystop)` extent,
/// for the given sub-pel MV `(qx, qy)`. `xstart = i*xbsep - xoffset` etc.
///
/// Returns a flat row-major buffer of length `xblen * yblen` with the
/// per-pixel `weight * prediction` values; entries falling outside the
/// `[0, picture_size)` reconstruction range are still computed (they
/// just won't contribute to the SSE in [`obmc_block_sse`]).
#[allow(clippy::too_many_arguments)]
fn block_weighted_pred(
    upref: &[i32],
    up_w: usize,
    up_h: usize,
    refp: &[i32],
    ref_w: usize,
    ref_h: usize,
    mv_precision: u32,
    weight: &[i32],
    xblen: usize,
    yblen: usize,
    xstart: i32,
    ystart: i32,
    qx: i32,
    qy: i32,
) -> Vec<i32> {
    let mut buf = vec![0i32; xblen * yblen];
    for q in 0..yblen {
        let y = ystart + q as i32;
        for p in 0..xblen {
            let x = xstart + p as i32;
            let pred = ref_pixel_at(
                upref,
                up_w,
                up_h,
                refp,
                ref_w,
                ref_h,
                mv_precision,
                x,
                y,
                qx,
                qy,
            );
            buf[q * xblen + p] = pred * weight[q * xblen + p];
        }
    }
    buf
}

/// Score the OBMC reconstruction for block `(i, j)` over its
/// `(xblen × yblen)` extent under the trial MV `(qx, qy)`, given the
/// pre-computed `neighbour_sum[q * xblen + p]` = Σₖ wₖ(x, y) pₖ(x, y)
/// over all *other* blocks k that overlap (x, y). Returns the sum of
/// squared error against the source plane `cur` for pixels inside the
/// picture's interior.
///
/// The §15.8.5 picture-domain reconstruction is
/// `recon(x, y) = clip(((Σ_k wₖ pₖ) + 32) >> 6)` so adding the trial
/// block's `w_ij(x, y) * pred_ij(x, y)` to `neighbour_sum`, normalising
/// and clipping reproduces exactly what the decoder will write into
/// the picture.
#[allow(clippy::too_many_arguments)]
fn obmc_block_sse(
    cur: &[u8],
    cur_w: i32,
    cur_h: i32,
    weighted_pred: &[i32],
    neighbour_sum: &[i32],
    xblen: usize,
    yblen: usize,
    xstart: i32,
    ystart: i32,
) -> i64 {
    let lo = -128i32;
    let hi = 127i32;
    let mut sse: i64 = 0;
    for q in 0..yblen {
        let y = ystart + q as i32;
        if y < 0 || y >= cur_h {
            continue;
        }
        for p in 0..xblen {
            let x = xstart + p as i32;
            if x < 0 || x >= cur_w {
                continue;
            }
            let total = weighted_pred[q * xblen + p] + neighbour_sum[q * xblen + p];
            // §15.8.2 picture-domain conversion: (sum + 32) >> 6 then
            // clip to the signed 8-bit range. Pre-offset cancels with
            // the 128-bias picture-storage convention, so we compare
            // against the raw u8 source minus 128.
            let recon_signed = ((total + 32) >> 6).clamp(lo, hi);
            let src_signed = cur[y as usize * cur_w as usize + x as usize] as i32 - 128;
            let d = recon_signed - src_signed;
            sse += (d * d) as i64;
        }
    }
    sse
}

/// Build the `neighbour_sum` array for block `(i, j)`: the §15.8.6
/// weighted prediction sum from all *other* blocks k that overlap `(i, j)`'s
/// extent, given the current MV grid `mvs`. Returned buffer is row-major
/// `xblen * yblen`.
///
/// Only the (up to) 8 neighbours of `(i, j)` can overlap — the OBMC
/// support is `(xblen × yblen)` = 8 × 8 with `(xbsep, ybsep)` = (4, 4)
/// stride (preset 1), so a block's extent intersects exactly the eight
/// surrounding 3×3 grid neighbours. Walking those is much cheaper than
/// rebuilding the full `mc_tmp` per candidate MV.
#[allow(clippy::too_many_arguments)]
fn build_neighbour_sum(
    mvs: &[IntegerMv],
    blocks_x: u32,
    blocks_y: u32,
    xblen: usize,
    yblen: usize,
    xbsep: usize,
    ybsep: usize,
    xoffset: usize,
    yoffset: usize,
    i: u32,
    j: u32,
    upref: &[i32],
    up_w: usize,
    up_h: usize,
    refp: &[i32],
    ref_w: usize,
    ref_h: usize,
    mv_precision: u32,
) -> Vec<i32> {
    let mut buf = vec![0i32; xblen * yblen];
    let xstart_ij = (i as i32) * (xbsep as i32) - (xoffset as i32);
    let ystart_ij = (j as i32) * (ybsep as i32) - (yoffset as i32);
    for dj in -1i32..=1 {
        for di in -1i32..=1 {
            if di == 0 && dj == 0 {
                continue;
            }
            let ni = i as i32 + di;
            let nj = j as i32 + dj;
            if ni < 0 || nj < 0 || ni >= blocks_x as i32 || nj >= blocks_y as i32 {
                continue;
            }
            let nbr_w = block_weight(
                xblen, yblen, xbsep, ybsep, xoffset, yoffset, ni as u32, nj as u32, blocks_x,
                blocks_y,
            );
            let nbr_xstart = ni * (xbsep as i32) - (xoffset as i32);
            let nbr_ystart = nj * (ybsep as i32) - (yoffset as i32);
            let nbr_mv = mvs[(nj as u32 * blocks_x + ni as u32) as usize];
            // Iterate over the intersection of the neighbour's extent
            // with (i, j)'s extent (in (i, j)'s local 0..xblen / 0..yblen
            // coordinates).
            let xlo_ij = (nbr_xstart - xstart_ij).max(0);
            let xhi_ij = ((nbr_xstart + xblen as i32) - xstart_ij).min(xblen as i32);
            let ylo_ij = (nbr_ystart - ystart_ij).max(0);
            let yhi_ij = ((nbr_ystart + yblen as i32) - ystart_ij).min(yblen as i32);
            for q_ij in ylo_ij..yhi_ij {
                let y = ystart_ij + q_ij;
                let q_nbr = (y - nbr_ystart) as usize;
                for p_ij in xlo_ij..xhi_ij {
                    let x = xstart_ij + p_ij;
                    let p_nbr = (x - nbr_xstart) as usize;
                    let pred = ref_pixel_at(
                        upref,
                        up_w,
                        up_h,
                        refp,
                        ref_w,
                        ref_h,
                        mv_precision,
                        x,
                        y,
                        nbr_mv.0,
                        nbr_mv.1,
                    );
                    buf[q_ij as usize * xblen + p_ij as usize] +=
                        pred * nbr_w[q_nbr * xblen + p_nbr];
                }
            }
        }
    }
    buf
}

/// **OBMC-aware ME refinement** (#186, Dirac §15.8.6). Iteratively
/// improves a starting MV grid by, for each block in raster order,
/// scoring the 8 sub-pel neighbours of its current MV via the same
/// weighted-blend reconstruction the decoder will perform — keeping
/// the candidate that minimises the per-block SSE against the source.
///
/// The refinement converges very fast on smooth motion (typically 1-2
/// passes); after that the MV grid is at a local minimum of the OBMC
/// reconstruction cost function. This is the standard "block-coordinate
/// descent" method described in OBMC literature and used by Dirac
/// reference encoders.
///
/// The function uses [`crate::obmc::spatial_wt`] for the §15.8.6 ramp
/// window so the encoder's blend matches the decoder symbol-for-symbol.
#[allow(clippy::too_many_arguments)]
pub fn obmc_refine_me(
    cur_y: &[u8],
    ref_y: &[u8],
    width: u32,
    height: u32,
    blocks_x: u32,
    blocks_y: u32,
    mvs: &mut [IntegerMv],
    mv_precision: u32,
    passes: u32,
) {
    if passes == 0 {
        return;
    }
    let (xblen_u, yblen_u, xbsep_u, ybsep_u) = (8usize, 8usize, 4usize, 4usize);
    let xoffset = (xblen_u - xbsep_u) / 2;
    let yoffset = (yblen_u - ybsep_u) / 2;
    let cur_w = width as i32;
    let cur_h = height as i32;
    // Pre-bake every block's spatial weight (only depends on its grid
    // coords). 16 × 16 = 256 vectors of 64 ints in the 64×64 fixture —
    // negligible memory.
    let mut weights: Vec<Vec<i32>> = Vec::with_capacity((blocks_x * blocks_y) as usize);
    for j in 0..blocks_y {
        for i in 0..blocks_x {
            weights.push(block_weight(
                xblen_u, yblen_u, xbsep_u, ybsep_u, xoffset, yoffset, i, j, blocks_x, blocks_y,
            ));
        }
    }
    // Build the half-pel-upsampled reference once per refinement.
    // The decoder OBMC blend operates on the §15.4 reference buffer,
    // which holds the *signed pre-output-offset* picture (i.e. `pic` -
    // `2^(depth-1)`). To match the decoder's reconstruction symbol-for-
    // symbol we feed signed reference samples and compare against
    // `source_u8 - 128`.
    let (upref, up_w, up_h) = if mv_precision > 0 {
        build_upref_signed(ref_y, width, height)
    } else {
        (Vec::new(), 0, 0)
    };
    let refp_signed: Vec<i32> = ref_y.iter().map(|&v| v as i32 - 128).collect();
    let ref_w = width as usize;
    let ref_h = height as usize;
    // Sub-pel step granularity: refine at the finest unit available.
    let step: i32 = 1;
    for _pass in 0..passes {
        for j in 0..blocks_y {
            for i in 0..blocks_x {
                let bidx = (j * blocks_x + i) as usize;
                let neighbour_sum = build_neighbour_sum(
                    mvs,
                    blocks_x,
                    blocks_y,
                    xblen_u,
                    yblen_u,
                    xbsep_u,
                    ybsep_u,
                    xoffset,
                    yoffset,
                    i,
                    j,
                    &upref,
                    up_w,
                    up_h,
                    &refp_signed,
                    ref_w,
                    ref_h,
                    mv_precision,
                );
                let xstart_ij = (i as i32) * (xbsep_u as i32) - (xoffset as i32);
                let ystart_ij = (j as i32) * (ybsep_u as i32) - (yoffset as i32);
                let weight = &weights[bidx];
                // Score the current MV first as the floor.
                let cur_mv = mvs[bidx];
                let cur_pred = block_weighted_pred(
                    &upref,
                    up_w,
                    up_h,
                    &refp_signed,
                    ref_w,
                    ref_h,
                    mv_precision,
                    weight,
                    xblen_u,
                    yblen_u,
                    xstart_ij,
                    ystart_ij,
                    cur_mv.0,
                    cur_mv.1,
                );
                let mut best_sse = obmc_block_sse(
                    cur_y,
                    cur_w,
                    cur_h,
                    &cur_pred,
                    &neighbour_sum,
                    xblen_u,
                    yblen_u,
                    xstart_ij,
                    ystart_ij,
                );
                let mut best_mv = cur_mv;
                // 8-neighbour search at the finest sub-pel unit.
                for dy in [-step, 0, step] {
                    for dx in [-step, 0, step] {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let cand_mv = IntegerMv(cur_mv.0 + dx, cur_mv.1 + dy);
                        let cand_pred = block_weighted_pred(
                            &upref,
                            up_w,
                            up_h,
                            &refp_signed,
                            ref_w,
                            ref_h,
                            mv_precision,
                            weight,
                            xblen_u,
                            yblen_u,
                            xstart_ij,
                            ystart_ij,
                            cand_mv.0,
                            cand_mv.1,
                        );
                        let sse = obmc_block_sse(
                            cur_y,
                            cur_w,
                            cur_h,
                            &cand_pred,
                            &neighbour_sum,
                            xblen_u,
                            yblen_u,
                            xstart_ij,
                            ystart_ij,
                        );
                        if sse < best_sse {
                            best_sse = sse;
                            best_mv = cand_mv;
                        }
                    }
                }
                mvs[bidx] = best_mv;
            }
        }
    }
}

// ---- §12.3 motion-data emit, mirrors decode_block_motion_data --------

/// Encode the §12.3 `block_motion_data` block for a 1-reference inter
/// picture with all-Ref1Only blocks at the bottom of the superblock
/// hierarchy (split=2 → 4x4 = 16 blocks per superblock, the maximum).
/// `mvs` is a `blocks_x * blocks_y` row-major array of integer-pel MVs.
pub fn encode_block_motion_data(
    w: &mut BitWriter,
    superblocks_x: u32,
    superblocks_y: u32,
    blocks_x: u32,
    blocks_y: u32,
    mvs: &[IntegerMv],
) {
    // Reconstruct what the decoder will end up with so we can match its
    // spatial-prediction / propagation behaviour symbol-for-symbol.
    let sb_split = vec![2u32; (superblocks_x * superblocks_y) as usize];
    let mut blocks: Vec<BlockData> = (0..blocks_x * blocks_y)
        .map(|i| BlockData {
            rmode: RefPredMode::Ref1Only,
            gmode: false,
            mv: [(mvs[i as usize].0, mvs[i as usize].1), (0, 0)],
            dc: [0; 3],
        })
        .collect();
    let bx = blocks_x;
    // 1) Superblock splits — emit the §12.3.1 length-prefixed block.
    let split_block = encode_sb_splits(superblocks_x, superblocks_y, &sb_split);
    write_uint_then_bytes(w, &split_block);

    // 2) Prediction modes — Ref1Only everywhere, no global. num_refs=1
    //    means we only emit the ref1 bit per block.
    let pmode_block =
        encode_prediction_modes(superblocks_x, superblocks_y, &sb_split, &blocks, bx, 1);
    write_uint_then_bytes(w, &pmode_block);

    // 3) Motion vectors — ref 1, dirn 0 then dirn 1.
    let v1x_block = encode_vector_elements(
        superblocks_x,
        superblocks_y,
        &sb_split,
        &mut blocks,
        bx,
        1,
        0,
    );
    write_uint_then_bytes(w, &v1x_block);
    let v1y_block = encode_vector_elements(
        superblocks_x,
        superblocks_y,
        &sb_split,
        &mut blocks,
        bx,
        1,
        1,
    );
    write_uint_then_bytes(w, &v1y_block);

    // 4) DC values for intra blocks — none in our r1 streams, so each
    //    component gets an empty arith block. We still must emit the
    //    length-prefixed empty block for each of Y/C1/C2.
    for _c in 0..3 {
        let dc_block = encode_dc_values(superblocks_x, superblocks_y, &sb_split, &blocks, bx);
        write_uint_then_bytes(w, &dc_block);
    }
}

/// Per-block bipred decision: which references this block uses and the
/// MVs against each. Indexed `[by * blocks_x + bx]`.
///
/// `rmode` is one of `Ref1Only` / `Ref2Only` / `Ref1And2` (never Intra
/// or gmode in our encoder); `mv1` is the MV against ref1 (only valid
/// when `rmode.uses_ref(1)`); `mv2` against ref2 (only valid when
/// `rmode.uses_ref(2)`). MVs are in `1/(2^mv_precision)` luma-pel units.
#[derive(Debug, Clone, Copy)]
pub struct BipredBlock {
    pub rmode: RefPredMode,
    pub mv1: IntegerMv,
    pub mv2: IntegerMv,
}

impl Default for BipredBlock {
    fn default() -> Self {
        Self {
            rmode: RefPredMode::Ref1Only,
            mv1: IntegerMv::default(),
            mv2: IntegerMv::default(),
        }
    }
}

/// Encode the §12.3 `block_motion_data` block for a **2-reference**
/// inter picture (parse code `0x0A`). Mirrors
/// [`encode_block_motion_data`] but writes the §11.2.7 prediction-mode
/// pair (ref1 + ref2 bits) and the four MV blocks (v1x, v1y, v2x, v2y).
///
/// `decisions` is `blocks_x * blocks_y` row-major; each entry carries
/// the per-block reference-mode + the two MVs the encoder chose.
pub fn encode_block_motion_data_bipred(
    w: &mut BitWriter,
    superblocks_x: u32,
    superblocks_y: u32,
    blocks_x: u32,
    blocks_y: u32,
    decisions: &[BipredBlock],
) {
    let sb_split = vec![2u32; (superblocks_x * superblocks_y) as usize];
    let mut blocks: Vec<BlockData> = (0..blocks_x * blocks_y)
        .map(|i| {
            let d = decisions[i as usize];
            BlockData {
                rmode: d.rmode,
                gmode: false,
                mv: [(d.mv1.0, d.mv1.1), (d.mv2.0, d.mv2.1)],
                dc: [0; 3],
            }
        })
        .collect();
    let bx = blocks_x;
    // 1) Superblock splits.
    let split_block = encode_sb_splits(superblocks_x, superblocks_y, &sb_split);
    write_uint_then_bytes(w, &split_block);

    // 2) Prediction modes — num_refs = 2 emits both bits per block.
    let pmode_block =
        encode_prediction_modes(superblocks_x, superblocks_y, &sb_split, &blocks, bx, 2);
    write_uint_then_bytes(w, &pmode_block);

    // 3a) ref1 vectors — dirn 0 then dirn 1. Skip blocks whose rmode
    //     doesn't use ref1 (Ref2Only) — same convention as the decoder.
    let v1x = encode_vector_elements(
        superblocks_x,
        superblocks_y,
        &sb_split,
        &mut blocks,
        bx,
        1,
        0,
    );
    write_uint_then_bytes(w, &v1x);
    let v1y = encode_vector_elements(
        superblocks_x,
        superblocks_y,
        &sb_split,
        &mut blocks,
        bx,
        1,
        1,
    );
    write_uint_then_bytes(w, &v1y);

    // 3b) ref2 vectors — dirn 0 then dirn 1. Skip blocks whose rmode
    //     doesn't use ref2 (Ref1Only) — same convention as the decoder.
    let v2x = encode_vector_elements(
        superblocks_x,
        superblocks_y,
        &sb_split,
        &mut blocks,
        bx,
        2,
        0,
    );
    write_uint_then_bytes(w, &v2x);
    let v2y = encode_vector_elements(
        superblocks_x,
        superblocks_y,
        &sb_split,
        &mut blocks,
        bx,
        2,
        1,
    );
    write_uint_then_bytes(w, &v2y);

    // 4) DC values for intra blocks — none here. Empty arith block per
    //    component.
    for _c in 0..3 {
        let dc_block = encode_dc_values(superblocks_x, superblocks_y, &sb_split, &blocks, bx);
        write_uint_then_bytes(w, &dc_block);
    }
}

/// Length-prefixed write: emit an interleaved exp-Golomb `length` then
/// byte-align and append the block bytes.
fn write_uint_then_bytes(w: &mut BitWriter, block: &[u8]) {
    w.write_uint(block.len() as u32);
    w.byte_align();
    w.write_bytes(block);
}

fn encode_sb_splits(sbx: u32, sby: u32, sb_split: &[u32]) -> Vec<u8> {
    let mut enc = ArithEncoder::new();
    let mut bank = ContextBank::new(3);
    let mut current = vec![0u32; (sbx * sby) as usize];
    for ysb in 0..sby {
        for xsb in 0..sbx {
            let pred = split_prediction(&current, sbx, xsb, ysb);
            let val = sb_split[(ysb * sbx + xsb) as usize];
            // residual = (val - pred) mod 3, in 0..=2.
            let residual = ((val as i32 - pred as i32).rem_euclid(3)) as u32;
            enc.write_uint(
                &mut bank,
                &[mvctx::SB_F1, mvctx::SB_F2],
                mvctx::SB_DATA,
                residual,
            );
            current[(ysb * sbx + xsb) as usize] = val;
        }
    }
    enc.finish()
}

fn split_prediction(current: &[u32], sbx: u32, x: u32, y: u32) -> u32 {
    if x == 0 && y == 0 {
        0
    } else if y == 0 {
        current[((y) * sbx + (x - 1)) as usize]
    } else if x == 0 {
        current[((y - 1) * sbx) as usize]
    } else {
        let a = current[((y - 1) * sbx + (x - 1)) as usize] as i64;
        let b = current[((y - 1) * sbx + x) as usize] as i64;
        let c = current[(y * sbx + (x - 1)) as usize] as i64;
        ((a + b + c + 1) / 3) as u32
    }
}

fn encode_prediction_modes(
    sbx: u32,
    sby: u32,
    sb_split: &[u32],
    blocks: &[BlockData],
    bx: u32,
    num_refs: u32,
) -> Vec<u8> {
    let mut enc = ArithEncoder::new();
    let mut bank = ContextBank::new(3);
    // Track which blocks have been "filled" by a top-level decode for
    // the prediction context (matches decoder propagation).
    let mut current_rmode = vec![RefPredMode::Intra; blocks.len()];
    for ysb in 0..sby {
        for xsb in 0..sbx {
            let split = sb_split[(ysb * sbx + xsb) as usize];
            let block_count = 1u32 << split;
            let step = 4 / block_count;
            for q in 0..block_count {
                for p in 0..block_count {
                    let blkx = 4 * xsb + p * step;
                    let blky = 4 * ysb + q * step;
                    let target = blocks[(blky * bx + blkx) as usize].rmode;
                    let pred = ref_mode_prediction(&current_rmode, bx, blkx, blky);
                    let bits = target.to_bits() ^ pred.to_bits();
                    // ref1 bit always emitted.
                    enc.write_bool(&mut bank, mvctx::PMODE_REF1, (bits & 1) == 1);
                    if num_refs == 2 {
                        // ref2 bit only emitted on 2-ref pictures (parse
                        // code 0x0A / 0x0B etc.). Mirrors the decoder's
                        // `block_ref_mode` second `read_bool` gated on
                        // `num_refs == 2`.
                        enc.write_bool(&mut bank, mvctx::PMODE_REF2, ((bits >> 1) & 1) == 1);
                    }
                    // No global block emit (using_global=false everywhere).
                    propagate_rmode(&mut current_rmode, bx, blkx, blky, step, target);
                }
            }
        }
    }
    enc.finish()
}

fn propagate_rmode(buf: &mut [RefPredMode], bx: u32, xtl: u32, ytl: u32, k: u32, val: RefPredMode) {
    for y in ytl..ytl + k {
        for x in xtl..xtl + k {
            buf[(y * bx + x) as usize] = val;
        }
    }
}

fn ref_mode_prediction(buf: &[RefPredMode], bx: u32, x: u32, y: u32) -> RefPredMode {
    if x == 0 && y == 0 {
        return RefPredMode::Intra;
    }
    if y == 0 {
        return buf[(x - 1) as usize];
    }
    if x == 0 {
        return buf[((y - 1) * bx) as usize];
    }
    let a = buf[((y - 1) * bx + (x - 1)) as usize].to_bits();
    let b = buf[((y - 1) * bx + x) as usize].to_bits();
    let c = buf[(y * bx + (x - 1)) as usize].to_bits();
    let r1 = (a & 1) + (b & 1) + (c & 1);
    let r2 = ((a >> 1) & 1) + ((b >> 1) & 1) + ((c >> 1) & 1);
    let mut p = 0u32;
    if r1 >= 2 {
        p |= 1;
    }
    if r2 >= 2 {
        p |= 2;
    }
    RefPredMode::from_bits(p)
}

fn encode_vector_elements(
    sbx: u32,
    sby: u32,
    sb_split: &[u32],
    blocks: &mut [BlockData],
    bx: u32,
    ref_num: u32,
    dirn: u32,
) -> Vec<u8> {
    let mut enc = ArithEncoder::new();
    let mut bank = ContextBank::new(7);
    let follow = [
        mvctx::VECTOR_F1,
        mvctx::VECTOR_F2,
        mvctx::VECTOR_F3,
        mvctx::VECTOR_F4,
        mvctx::VECTOR_F5PLUS,
    ];
    // Track the **decoded** values so far for spatial-prediction
    // context. Same data layout as `blocks` so we can call the
    // predictor directly.
    let mut current = vec![BlockData::default(); blocks.len()];
    let idx = (ref_num - 1) as usize;
    for ysb in 0..sby {
        for xsb in 0..sbx {
            let split = sb_split[(ysb * sbx + xsb) as usize];
            let block_count = 1u32 << split;
            let step = 4 / block_count;
            for q in 0..block_count {
                for p in 0..block_count {
                    let blkx = 4 * xsb + p * step;
                    let blky = 4 * ysb + q * step;
                    let bi = (blky * bx + blkx) as usize;
                    let block = &blocks[bi];
                    // Set the current block's rmode in `current` so
                    // mv_prediction's mv_available check matches.
                    current[bi].rmode = block.rmode;
                    // Decoder-side `block_vector` returns early if the
                    // block doesn't use this reference (or is global) —
                    // mirror that exactly so 2-ref Ref1Only blocks skip
                    // their unused ref2 residual (and vice versa).
                    if !block.rmode.uses_ref(ref_num as usize) || block.gmode {
                        // Propagate the (unchanged) zero MV.
                        propagate_mv(&mut current, bx, blkx, blky, step, idx);
                        continue;
                    }
                    let target = match dirn {
                        0 => block.mv[idx].0,
                        _ => block.mv[idx].1,
                    };
                    let pred = mv_prediction(&current, bx, blkx, blky, ref_num, dirn);
                    let residual = target - pred;
                    enc.write_sint(
                        &mut bank,
                        &follow,
                        mvctx::VECTOR_DATA,
                        mvctx::VECTOR_SIGN,
                        residual,
                    );
                    // Set the current MV component then propagate.
                    match dirn {
                        0 => current[bi].mv[idx].0 = target,
                        _ => current[bi].mv[idx].1 = target,
                    }
                    propagate_mv(&mut current, bx, blkx, blky, step, idx);
                }
            }
        }
    }
    enc.finish()
}

fn propagate_mv(buf: &mut [BlockData], bx: u32, xtl: u32, ytl: u32, k: u32, idx: usize) {
    let src = buf[(ytl * bx + xtl) as usize].mv[idx];
    let src_rmode = buf[(ytl * bx + xtl) as usize].rmode;
    for y in ytl..ytl + k {
        for x in xtl..xtl + k {
            if x == xtl && y == ytl {
                continue;
            }
            buf[(y * bx + x) as usize].mv[idx] = src;
            // Mirror the decoder propagating rmode early in
            // decode_prediction_modes — by the time vector_elements
            // runs, all blocks within a level-2 superblock share the
            // top-left's rmode.
            buf[(y * bx + x) as usize].rmode = src_rmode;
        }
    }
}

fn mv_prediction(current: &[BlockData], bx: u32, x: u32, y: u32, ref_num: u32, dirn: u32) -> i32 {
    if x == 0 && y == 0 {
        return 0;
    }
    let idx = (ref_num - 1) as usize;
    let pick = |b: &BlockData| -> i32 {
        match dirn {
            0 => b.mv[idx].0,
            _ => b.mv[idx].1,
        }
    };
    let avail = |b: &BlockData| -> bool { !b.gmode && b.rmode.uses_ref(ref_num as usize) };
    if y == 0 {
        let b = &current[(x - 1) as usize];
        return if avail(b) { pick(b) } else { 0 };
    }
    if x == 0 {
        let b = &current[((y - 1) * bx) as usize];
        return if avail(b) { pick(b) } else { 0 };
    }
    let mut values: Vec<i32> = Vec::with_capacity(3);
    for (dx, dy) in [(1u32, 0u32), (0, 1), (1, 1)] {
        let b = &current[((y - dy) * bx + (x - dx)) as usize];
        if avail(b) {
            values.push(pick(b));
        }
    }
    median(&mut values)
}

/// Same integer median as picture_inter.
fn median(values: &mut [i32]) -> i32 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        let a = values[n / 2 - 1] as i64;
        let b = values[n / 2] as i64;
        let sum = a + b + 1;
        let q = sum / 2;
        let r = sum % 2;
        (if r < 0 { q - 1 } else { q }) as i32
    }
}

fn encode_dc_values(
    sbx: u32,
    sby: u32,
    _sb_split: &[u32],
    _blocks: &[BlockData],
    _bx: u32,
) -> Vec<u8> {
    // r1 has no Intra blocks, so this is just an empty arith block —
    // every iteration inside the spec's loop short-circuits on `if
    // get_block().rmode != Intra { return; }`. The arith engine still
    // needs `finish()` so the decoder's `read_bool` calls (during
    // context updates triggered by the LAST symbol of the previous
    // block, which read past-end 1s) stay self-consistent.
    let enc = ArithEncoder::new();
    let _bank: ContextBank = ContextBank::new(4);
    // Walk to keep the loop structure parallel to the decoder — but
    // never write a symbol since rmode is always Ref1Only here.
    for _ysb in 0..sby {
        for _xsb in 0..sbx {
            // Iteration is a no-op for this r1 path.
        }
    }
    enc.finish()
}

// ---- §11.3 wavelet residue emit ---------------------------------------

/// Build the OBMC reference reconstruction the decoder will compute for
/// one component, given the current MV grid. Returns the §15.8.5
/// per-pixel `(Σ_k wₖ pₖ + 32) >> 6` value clipped to
/// `[-2^(depth-1), 2^(depth-1) - 1]` — the **signed pre-output-offset**
/// reference, which is also the domain in which the residue lives.
///
/// This is essentially [`crate::obmc::motion_compensate`] called with
/// `pic = 0` everywhere — but written out so the encoder can reuse the
/// same reference plane for later residue subtraction without doubling
/// the work or rebuilding `mc_tmp` per pixel.
#[allow(clippy::too_many_arguments)]
fn build_obmc_prediction(
    sequence: &SequenceHeader,
    pred: &PicturePredictionParams,
    motion: &PictureMotionData,
    is_chroma: bool,
    ref_y: &[u8],
    ref_u: &[u8],
    ref_v: &[u8],
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let chroma_h_ratio = sequence.video_params.chroma_format.h_ratio();
    let chroma_v_ratio = sequence.video_params.chroma_format.v_ratio();
    let _ = is_chroma;
    let pred_y = build_obmc_prediction_one(
        sequence,
        pred,
        motion,
        false,
        chroma_h_ratio,
        chroma_v_ratio,
        ref_y,
        sequence.luma_width as usize,
        sequence.luma_height as usize,
    );
    let pred_u = build_obmc_prediction_one(
        sequence,
        pred,
        motion,
        true,
        chroma_h_ratio,
        chroma_v_ratio,
        ref_u,
        sequence.chroma_width as usize,
        sequence.chroma_height as usize,
    );
    let pred_v = build_obmc_prediction_one(
        sequence,
        pred,
        motion,
        true,
        chroma_h_ratio,
        chroma_v_ratio,
        ref_v,
        sequence.chroma_width as usize,
        sequence.chroma_height as usize,
    );
    (pred_y, pred_u, pred_v)
}

/// Reconstruct one component's OBMC prediction by calling the same
/// [`crate::obmc::motion_compensate`] entry the decoder uses, with a
/// zero-residue input plane. The returned plane is the post-clip,
/// signed pre-output-offset reconstruction — exactly the shape the
/// decoder will emit before §15.10's output offset pass.
#[allow(clippy::too_many_arguments)]
fn build_obmc_prediction_one(
    sequence: &SequenceHeader,
    pred: &PicturePredictionParams,
    motion: &PictureMotionData,
    is_chroma: bool,
    chroma_h_ratio: u32,
    chroma_v_ratio: u32,
    ref_plane: &[u8],
    comp_w: usize,
    comp_h: usize,
) -> Vec<i32> {
    let depth = if is_chroma {
        sequence.chroma_depth
    } else {
        sequence.luma_depth
    };
    let half = 1i32 << (depth - 1);
    // Reference plane in the spec's signed pre-offset convention.
    let ref_signed: Vec<i32> = ref_plane.iter().map(|&v| v as i32 - half).collect();
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
    let mc_params = McParams {
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
        luma_depth: sequence.luma_depth,
        chroma_depth: sequence.chroma_depth,
    };
    let mut pic = vec![0i32; comp_w * comp_h];
    crate::obmc::motion_compensate(
        &mut pic,
        &mc_params,
        motion,
        Some((&ref_signed, comp_w, comp_h)),
        None,
    );
    pic
}

/// Build the §11.2.1-shaped `PicturePredictionParams` the residue path
/// needs to drive the OBMC reconstruction. Mirrors
/// [`write_picture_prediction_parameters`] symbol-for-symbol — what the
/// decoder will read back from those bytes.
///
/// `mv_precision` is passed explicitly so the bipred path can use
/// `params.bipred_mv_precision` while the 1-ref path uses `params.mv_precision`.
fn picture_prediction_params_from(
    sequence: &SequenceHeader,
    _iep: &InterEncoderParams,
    mv_precision: u32,
) -> PicturePredictionParams {
    let (sbx, sby, blocks_x, blocks_y) = motion_grid(sequence.luma_width, sequence.luma_height);
    let (xblen, yblen, xbsep, ybsep) = PRESET1;
    PicturePredictionParams {
        luma_xblen: xblen,
        luma_yblen: yblen,
        luma_xbsep: xbsep,
        luma_ybsep: ybsep,
        mv_precision,
        using_global: false,
        prediction_mode: 0,
        superblocks_x: sbx,
        superblocks_y: sby,
        blocks_x,
        blocks_y,
        refs_wt_precision: 1,
        ref1_wt: 1,
        ref2_wt: 1,
        global1: None,
        global2: None,
    }
}

/// Build the same `PictureMotionData` the decoder will reconstruct from
/// the `block_motion_data` block this encoder emits — every block is
/// `Ref1Only` with the chosen MV, all superblocks at split=2.
fn build_motion_from_mv_grid(
    sbx: u32,
    sby: u32,
    blocks_x: u32,
    blocks_y: u32,
    mvs: &[IntegerMv],
) -> PictureMotionData {
    let blocks: Vec<BlockData> = (0..blocks_x * blocks_y)
        .map(|i| BlockData {
            rmode: RefPredMode::Ref1Only,
            gmode: false,
            mv: [(mvs[i as usize].0, mvs[i as usize].1), (0, 0)],
            dc: [0; 3],
        })
        .collect();
    PictureMotionData {
        blocks_x,
        blocks_y,
        superblocks_x: sbx,
        superblocks_y: sby,
        sb_split: vec![2u32; (sbx * sby) as usize],
        blocks,
        global1: None,
        global2: None,
    }
}

/// Subtract the OBMC prediction from the source picture in the spec's
/// **signed pre-output-offset** domain — i.e.
/// `residue[x] = (source_u8[x] - half) - prediction_signed[x]`.
/// The result is a signed plane the wavelet residue encoder forward-
/// transforms as if it were a tiny intra picture.
fn build_residue_plane(source_u8: &[u8], prediction_signed: &[i32], depth: u32) -> Vec<i32> {
    debug_assert_eq!(source_u8.len(), prediction_signed.len());
    let half = 1i32 << (depth - 1);
    source_u8
        .iter()
        .zip(prediction_signed.iter())
        .map(|(&s, &p)| (s as i32 - half) - p)
        .collect()
}

/// Forward-DWT then dead-zone quantise one residue plane. Mirrors
/// [`crate::encoder_intra_core::forward_and_quantise`] but operates on
/// a pre-signed `i32` source (no `- half` step — the residue is already
/// in the signed pre-offset domain).
fn forward_and_quantise_residue(
    residue: &[i32],
    comp_w: u32,
    comp_h: u32,
    rp: &ResidueParams,
) -> Vec<[SubbandData; 4]> {
    let (pw, ph) = padded_component_dims(comp_w, comp_h, rp.dwt_depth);
    let mut pic = SubbandData::new(pw, ph);
    let comp_w_u = comp_w as usize;
    let comp_h_u = comp_h as usize;
    for y in 0..ph {
        for x in 0..pw {
            let src_x = x.min(comp_w_u - 1);
            let src_y = y.min(comp_h_u - 1);
            pic.set(y, x, residue[src_y * comp_w_u + src_x]);
        }
    }
    let py = dwt(&pic, rp.wavelet, rp.dwt_depth);
    let mut out: Vec<[SubbandData; 4]> = Vec::with_capacity(py.len());
    for bands in py.iter() {
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
            let dst = &mut level_out[orient_idx];
            for y in 0..src.height {
                for x in 0..src.width {
                    let v = src.get(y, x);
                    dst.set(y, x, residue_quantise_coeff(v, rp.qindex));
                }
            }
        }
        out.push(level_out);
    }
    out
}

/// Dead-zone forward quantisation — same formula as the intra encoder.
fn residue_quantise_coeff(x: i32, q: u32) -> i32 {
    if x == 0 {
        return 0;
    }
    let qf = quant_factor(q) as i64;
    let mag = (x.unsigned_abs() as i64 * 4) / qf;
    if x < 0 {
        -(mag as i32)
    } else {
        mag as i32
    }
}

/// Emit `transform_parameters` for the residue path. Mirrors
/// [`crate::encoder_intra_core::write_core_transform_parameters`] —
/// the inter decoder's residue path uses the same syntax (no
/// `is_intra` distinction at the parameter level).
fn write_residue_transform_parameters(w: &mut BitWriter, rp: &ResidueParams) {
    w.write_uint(wavelet_index(rp.wavelet));
    w.write_uint(rp.dwt_depth);
    // §11.3.3 codeblock_parameters: spatial_partition_flag = 0 →
    // single codeblock per subband at every level.
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

/// Emit one component's quantised residue pyramid as a sequence of
/// length-prefixed AC-coded subband blocks. Mirrors
/// [`crate::encoder_intra_core::write_component_subbands`] — the
/// decoder treats inter and intra subband bytes identically apart
/// from the LL DC-prediction step (which is gated on `is_intra` and
/// therefore skipped here).
fn write_residue_component_subbands(
    w: &mut BitWriter,
    rp: &ResidueParams,
    qpy: &[[SubbandData; 4]],
) {
    write_residue_subband_block(w, rp, qpy, 0, Orient::LL);
    for level in 1..=rp.dwt_depth {
        for orient in [Orient::HL, Orient::LH, Orient::HH] {
            write_residue_subband_block(w, rp, qpy, level, orient);
        }
    }
}

/// Emit a single subband block: byte-align, length, qindex,
/// byte-align, AC bytes. Empty bands (after IDWT padding) emit
/// `length = 0` only.
fn write_residue_subband_block(
    w: &mut BitWriter,
    rp: &ResidueParams,
    qpy: &[[SubbandData; 4]],
    level: u32,
    orient: Orient,
) {
    w.byte_align();
    let band = &qpy[level as usize][orient.as_index()];
    if band.width == 0 || band.height == 0 {
        w.write_uint(0);
        w.byte_align();
        return;
    }
    let bytes = encode_residue_subband_ac(qpy, level, orient);
    if bytes.is_empty() {
        w.write_uint(0);
        w.byte_align();
        return;
    }
    w.write_uint(bytes.len() as u32);
    w.write_uint(rp.qindex);
    w.byte_align();
    w.write_bytes(&bytes);
}

/// Encode one residue subband under the §13.4.4 AC contexts — same
/// raster walk and same `parent_zero / nhood_zero / sign_predict`
/// helpers as the intra path.
fn encode_residue_subband_ac(pyramid: &[[SubbandData; 4]], level: u32, orient: Orient) -> Vec<u8> {
    let mut bank = ContextBank::new(ctx::NUM_CONTEXTS);
    let mut enc = ArithEncoder::new();
    let level_idx = level as usize;
    let orient_idx = orient.as_index();
    let band_w = pyramid[level_idx][orient_idx].width;
    let band_h = pyramid[level_idx][orient_idx].height;
    let band = &pyramid[level_idx][orient_idx];
    let parent: Option<&SubbandData> = if level >= 2 {
        Some(&pyramid[level_idx - 1][orient_idx])
    } else {
        None
    };
    for y in 0..band_h {
        for x in 0..band_w {
            let parent_zero = match parent {
                Some(p) => p.get(y / 2, x / 2) == 0,
                None => true,
            };
            let nhood_zero = residue_zero_nhood(band, x, y);
            let sign_pred = residue_sign_predict(band, orient, x, y);
            let (follow, data_ctx, sign_ctx) =
                residue_select_coeff_ctxs(parent_zero, nhood_zero, sign_pred);
            let qc = band.get(y, x);
            enc.write_sint(&mut bank, follow, data_ctx, sign_ctx, qc);
        }
    }
    enc.finish()
}

// Local mirrors of `picture_core::{zero_nhood, sign_predict,
// select_coeff_ctxs}` — same logic, kept here so the inter encoder
// doesn't have to depend on private helpers in another module. They
// match the spec's pseudo-code directly.
fn residue_zero_nhood(band: &SubbandData, x: usize, y: usize) -> bool {
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

fn residue_sign_predict(band: &SubbandData, orient: Orient, x: usize, y: usize) -> i32 {
    match orient {
        Orient::HL if y > 0 => residue_signum(band.get(y - 1, x)),
        Orient::LH if x > 0 => residue_signum(band.get(y, x - 1)),
        _ => 0,
    }
}

fn residue_signum(v: i32) -> i32 {
    if v > 0 {
        1
    } else if v < 0 {
        -1
    } else {
        0
    }
}

fn residue_select_coeff_ctxs(
    parent_zero: bool,
    nhood_zero: bool,
    sign_pred: i32,
) -> (&'static [usize], usize, usize) {
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

// ---- Picture-level inter emission ------------------------------------

/// Encode one core-syntax 1-ref inter picture, parse code `0x09`. The
/// returned payload follows the parse-info header — caller is
/// responsible for the parse_info bytes themselves.
///
/// `picture_number` is the encoded picture's number; `ref1_picture_number`
/// is the reference's. The decoded delta is `ref1 - picture_number` cast
/// to `i32` (§9.6.1).
#[allow(clippy::too_many_arguments)]
pub fn encode_inter_picture(
    sequence: &SequenceHeader,
    params: &InterEncoderParams,
    picture_number: u32,
    ref1_picture_number: u32,
    cur_y: &[u8],
    cur_u: &[u8],
    cur_v: &[u8],
    ref_y: &[u8],
    ref_u: &[u8],
    ref_v: &[u8],
) -> Vec<u8> {
    assert_eq!(params.block_params_index, 1, "r1 only supports preset 1");

    let mut w = BitWriter::new();

    // §12.2 picture_header.
    w.byte_align();
    w.write_uint_lit(4, picture_number);

    // §9.6.1 reference deltas — one signed for ref1.
    let d1: i32 = ref1_picture_number.wrapping_sub(picture_number) as i32;
    w.write_sint(d1);
    // No retd — parse code 0x09 is non-reference.

    w.byte_align();

    // §11.2 picture_prediction_parameters.
    write_picture_prediction_parameters(&mut w, params, params.mv_precision);
    w.byte_align();

    // §12.3 block_motion_data. Sub-pel ME degenerates to integer-pel
    // when `mv_precision == 0`.
    let (sbx, sby, blocks_x, blocks_y) = motion_grid(sequence.luma_width, sequence.luma_height);
    let mut mvs = subpel_search_me(
        cur_y,
        ref_y,
        sequence.luma_width,
        sequence.luma_height,
        blocks_x,
        blocks_y,
        params.mv_search_range,
        params.mv_precision,
    );
    // §15.8.6 OBMC-aware refinement (#186) — minimises the per-block
    // SSE of the *blended* reconstruction, matching what the decoder
    // (and ffmpeg) will actually emit. No-op when `obmc_refine_passes
    // == 0`, which is the explicit "no-OBMC" baseline used by tests.
    obmc_refine_me(
        cur_y,
        ref_y,
        sequence.luma_width,
        sequence.luma_height,
        blocks_x,
        blocks_y,
        &mut mvs,
        params.mv_precision,
        params.obmc_refine_passes,
    );
    encode_block_motion_data(&mut w, sbx, sby, blocks_x, blocks_y, &mvs);
    w.byte_align();

    // §11.3 wavelet_transform.
    if let Some(ref residue) = params.residue {
        // ZERO_RESIDUAL = false → emit transform parameters + per-
        // component subbands. Build the same `PictureMotionData` the
        // decoder will reconstruct from the bytes we just wrote so the
        // OBMC prediction matches symbol-for-symbol.
        let pred = picture_prediction_params_from(sequence, params, params.mv_precision);
        let motion = build_motion_from_mv_grid(sbx, sby, blocks_x, blocks_y, &mvs);
        let (pred_y, pred_u, pred_v) =
            build_obmc_prediction(sequence, &pred, &motion, false, ref_y, ref_u, ref_v);
        let res_y = build_residue_plane(cur_y, &pred_y, sequence.luma_depth);
        let res_u = build_residue_plane(cur_u, &pred_u, sequence.chroma_depth);
        let res_v = build_residue_plane(cur_v, &pred_v, sequence.chroma_depth);

        // §11.3 ZERO_RESIDUAL bool. Per spec, transform_parameters
        // immediately follows in the same byte (no align), and only
        // *after* transform_parameters does the decoder byte-align before
        // pulling per-component subband bytes.
        w.write_bool(false);
        write_residue_transform_parameters(&mut w, residue);
        w.byte_align();

        let qpy_y = forward_and_quantise_residue(
            &res_y,
            sequence.luma_width,
            sequence.luma_height,
            residue,
        );
        let qpy_u = forward_and_quantise_residue(
            &res_u,
            sequence.chroma_width,
            sequence.chroma_height,
            residue,
        );
        let qpy_v = forward_and_quantise_residue(
            &res_v,
            sequence.chroma_width,
            sequence.chroma_height,
            residue,
        );
        write_residue_component_subbands(&mut w, residue, &qpy_y);
        write_residue_component_subbands(&mut w, residue, &qpy_u);
        write_residue_component_subbands(&mut w, residue, &qpy_v);
        w.byte_align();
    } else {
        // ZERO_RESIDUAL = true so we skip the entire transform_parameters
        // / coefficient stream. The decoder treats the residue as zero
        // everywhere and reconstruction = OBMC(reference).
        w.write_bool(true);
        w.byte_align();
    }

    w.finish()
}

fn write_picture_prediction_parameters(
    w: &mut BitWriter,
    _params: &InterEncoderParams,
    mv_precision: u32,
) {
    // §11.2.2 block_parameters: index 1 (preset 8x8 / 4x4).
    w.write_uint(1);
    // §11.2.5 motion_vector_precision (0=integer, 1=half, 2=quarter,
    // 3=eighth pel; spec caps at 3).
    debug_assert!(mv_precision <= 3, "mv_precision must be 0..=3");
    w.write_uint(mv_precision);
    // §11.2.6 global_motion: not used.
    w.write_bool(false);
    // §11.2.7 picture_prediction_mode: 0 (default).
    w.write_uint(0);
    // §11.2.8 reference_picture_weights_flag: false → defaults
    // refs_wt_precision=1, ref1_wt=ref2_wt=1.
    w.write_bool(false);
}

// ---- 2-reference bipred encoding (#190 follow-up) --------------------

/// **Bipred per-block decision search.** For each block, evaluates
/// three candidate prediction modes against the source plane:
///
/// * `Ref1Only` with the best ref1 MV (§15.8.7 single-ref pixel pred);
/// * `Ref2Only` with the best ref2 MV;
/// * `Ref1And2` with the per-ref MVs averaged 50/50 by the OBMC blend
///   (§15.8.5 weighted sum at `ref1_wt = ref2_wt = 1`,
///   `refs_wt_precision = 1`).
///
/// The bipred SAD is the absolute difference between the source pixel
/// and the rounded average of the two references' sub-pel-sampled
/// predictions. The mode that minimises per-block SAD wins, with a
/// tiny lambda-style tie-break preferring the simpler 1-ref modes
/// (smaller MV residue + no second MV pair to encode).
///
/// `mv_precision` is in `1/(2^p)` luma pel units (matches
/// [`InterEncoderParams::mv_precision`]).
#[allow(clippy::too_many_arguments)]
pub fn bipred_select_modes(
    cur_y: &[u8],
    ref1_y: &[u8],
    ref2_y: &[u8],
    width: u32,
    height: u32,
    blocks_x: u32,
    blocks_y: u32,
    search: u32,
    mv_precision: u32,
) -> Vec<BipredBlock> {
    // Independent per-ref ME — same path the 1-ref encoder uses, just
    // re-run against each reference.
    let mvs1 = subpel_search_me(
        cur_y,
        ref1_y,
        width,
        height,
        blocks_x,
        blocks_y,
        search,
        mv_precision,
    );
    let mvs2 = subpel_search_me(
        cur_y,
        ref2_y,
        width,
        height,
        blocks_x,
        blocks_y,
        search,
        mv_precision,
    );
    let (xblen, yblen, xbsep, ybsep) = (8i32, 8i32, 4i32, 4i32);
    let w_i = width as i32;
    let h_i = height as i32;
    // Build sub-pel upref planes for both references when needed —
    // bipred SAD reads the same §15.8.10/§15.8.11 samples the decoder
    // OBMC blend will, so the per-block ranking matches the eventual
    // reconstruction error.
    let (up1, up1_w, up1_h) = build_upref(ref1_y, width, height);
    let (up2, up2_w, up2_h) = build_upref(ref2_y, width, height);
    let mut out: Vec<BipredBlock> = Vec::with_capacity(mvs1.len());
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let idx = (by * blocks_x + bx) as usize;
            let mv1 = mvs1[idx];
            let mv2 = mvs2[idx];
            let x0 = bx as i32 * xbsep;
            let y0 = by as i32 * ybsep;
            // Per-mode SAD. For 1-ref modes we re-evaluate at the
            // chosen sub-pel MV via `sad_subpel` so the score lines up
            // exactly with the bipred path's resampling convention.
            let sad1 = if mv_precision == 0 {
                sad_block(cur_y, ref1_y, w_i, h_i, x0, y0, mv1.0, mv1.1, xblen, yblen)
            } else {
                let scale = 1i64 << mv_precision;
                let qx = (x0 as i64) * scale + mv1.0 as i64;
                let qy = (y0 as i64) * scale + mv1.1 as i64;
                sad_subpel(
                    cur_y,
                    &up1,
                    up1_w,
                    up1_h,
                    w_i,
                    h_i,
                    x0,
                    y0,
                    qx,
                    qy,
                    xblen,
                    yblen,
                    mv_precision,
                )
            };
            let sad2 = if mv_precision == 0 {
                sad_block(cur_y, ref2_y, w_i, h_i, x0, y0, mv2.0, mv2.1, xblen, yblen)
            } else {
                let scale = 1i64 << mv_precision;
                let qx = (x0 as i64) * scale + mv2.0 as i64;
                let qy = (y0 as i64) * scale + mv2.1 as i64;
                sad_subpel(
                    cur_y,
                    &up2,
                    up2_w,
                    up2_h,
                    w_i,
                    h_i,
                    x0,
                    y0,
                    qx,
                    qy,
                    xblen,
                    yblen,
                    mv_precision,
                )
            };
            // Bipred SAD: `(p1 + p2 + 1) >> 1` per pixel (the §15.8.5
            // weighted sum at `ref_wt=(1,1)` and `refs_wt_precision=1`,
            // i.e. shift by 1, round-to-nearest at `+1`). This is what
            // the decoder will accumulate into `mc_tmp` for `Ref1And2`
            // blocks (post-spatial-weight, post-OBMC-blend).
            let sad12 = sad_bipred(
                cur_y,
                &up1,
                up1_w,
                up1_h,
                ref1_y,
                &up2,
                up2_w,
                up2_h,
                ref2_y,
                w_i,
                h_i,
                x0,
                y0,
                mv1,
                mv2,
                xblen,
                yblen,
                mv_precision,
            );
            // Lambda-style tie-break: bipred carries an extra 2 MV
            // components so it must win by a clear margin to be picked.
            // Without this, equal-SAD ties default to bipred and the
            // decoder spends bits on a second MV pair that doesn't
            // improve reconstruction.
            const BIPRED_PENALTY: i64 = 64; // ~1 LSB per pixel over an 8x8 block.
            let mut best_mode = RefPredMode::Ref1Only;
            let mut best_sad = sad1;
            if sad2 < best_sad {
                best_sad = sad2;
                best_mode = RefPredMode::Ref2Only;
            }
            if sad12 + BIPRED_PENALTY < best_sad {
                best_mode = RefPredMode::Ref1And2;
            }
            out.push(BipredBlock {
                rmode: best_mode,
                mv1,
                mv2,
            });
        }
    }
    out
}

/// SAD of one block under the bipred prediction
/// `(pred_ref1 + pred_ref2 + 1) >> 1` against the source. Matches the
/// §15.8.5 reconstruction at `ref1_wt = ref2_wt = 1`,
/// `refs_wt_precision = 1` (post-OBMC blend, pre-spatial-weight scale).
#[allow(clippy::too_many_arguments)]
fn sad_bipred(
    cur: &[u8],
    up1: &[i32],
    up1_w: usize,
    up1_h: usize,
    ref1: &[u8],
    up2: &[i32],
    up2_w: usize,
    up2_h: usize,
    ref2: &[u8],
    w: i32,
    h: i32,
    x0: i32,
    y0: i32,
    mv1: IntegerMv,
    mv2: IntegerMv,
    xblen: i32,
    yblen: i32,
    mv_precision: u32,
) -> i64 {
    let mut sad: i64 = 0;
    if mv_precision == 0 {
        for y in 0..yblen {
            let cy = (y0 + y).clamp(0, h - 1) as usize;
            for x in 0..xblen {
                let cx = (x0 + x).clamp(0, w - 1) as usize;
                let r1x = (x0 + x + mv1.0).clamp(0, w - 1) as usize;
                let r1y = (y0 + y + mv1.1).clamp(0, h - 1) as usize;
                let r2x = (x0 + x + mv2.0).clamp(0, w - 1) as usize;
                let r2y = (y0 + y + mv2.1).clamp(0, h - 1) as usize;
                let p1 = ref1[r1y * w as usize + r1x] as i32;
                let p2 = ref2[r2y * w as usize + r2x] as i32;
                let pavg = (p1 + p2 + 1) >> 1;
                let s = cur[cy * w as usize + cx] as i32;
                sad += (s - pavg).unsigned_abs() as i64;
            }
        }
    } else {
        let scale = 1i64 << mv_precision;
        for y in 0..yblen {
            let cy = (y0 + y).clamp(0, h - 1) as usize;
            let py = (y0 as i64) * scale + mv1.1 as i64 + (y as i64) * scale;
            let py2 = (y0 as i64) * scale + mv2.1 as i64 + (y as i64) * scale;
            for x in 0..xblen {
                let cx = (x0 + x).clamp(0, w - 1) as usize;
                let px = (x0 as i64) * scale + mv1.0 as i64 + (x as i64) * scale;
                let px2 = (x0 as i64) * scale + mv2.0 as i64 + (x as i64) * scale;
                let p1 = subpel_predict(up1, up1_w, up1_h, px, py, mv_precision);
                let p2 = subpel_predict(up2, up2_w, up2_h, px2, py2, mv_precision);
                let pavg = (p1 + p2 + 1) >> 1;
                let s = cur[cy * w as usize + cx] as i32;
                sad += (s - pavg).unsigned_abs() as i64;
            }
        }
    }
    sad
}

/// Build the same `PictureMotionData` the decoder will reconstruct from
/// the 2-ref `block_motion_data` block this encoder emits — preserves
/// the per-block `RefPredMode` and both MVs.
fn build_motion_from_bipred_grid(
    sbx: u32,
    sby: u32,
    blocks_x: u32,
    blocks_y: u32,
    decisions: &[BipredBlock],
) -> PictureMotionData {
    let blocks: Vec<BlockData> = (0..blocks_x * blocks_y)
        .map(|i| {
            let d = decisions[i as usize];
            BlockData {
                rmode: d.rmode,
                gmode: false,
                mv: [(d.mv1.0, d.mv1.1), (d.mv2.0, d.mv2.1)],
                dc: [0; 3],
            }
        })
        .collect();
    PictureMotionData {
        blocks_x,
        blocks_y,
        superblocks_x: sbx,
        superblocks_y: sby,
        sb_split: vec![2u32; (sbx * sby) as usize],
        blocks,
        global1: None,
        global2: None,
    }
}

/// Build the §11.8.5 OBMC reference reconstruction for a 2-ref bipred
/// picture. Same shape as [`build_obmc_prediction`] but feeds both
/// reference planes through `crate::obmc::motion_compensate`.
#[allow(clippy::too_many_arguments)]
fn build_obmc_prediction_bipred(
    sequence: &SequenceHeader,
    pred: &PicturePredictionParams,
    motion: &PictureMotionData,
    ref1_y: &[u8],
    ref1_u: &[u8],
    ref1_v: &[u8],
    ref2_y: &[u8],
    ref2_u: &[u8],
    ref2_v: &[u8],
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let chroma_h_ratio = sequence.video_params.chroma_format.h_ratio();
    let chroma_v_ratio = sequence.video_params.chroma_format.v_ratio();
    let pred_y = build_obmc_prediction_one_bipred(
        sequence,
        pred,
        motion,
        false,
        chroma_h_ratio,
        chroma_v_ratio,
        ref1_y,
        ref2_y,
        sequence.luma_width as usize,
        sequence.luma_height as usize,
    );
    let pred_u = build_obmc_prediction_one_bipred(
        sequence,
        pred,
        motion,
        true,
        chroma_h_ratio,
        chroma_v_ratio,
        ref1_u,
        ref2_u,
        sequence.chroma_width as usize,
        sequence.chroma_height as usize,
    );
    let pred_v = build_obmc_prediction_one_bipred(
        sequence,
        pred,
        motion,
        true,
        chroma_h_ratio,
        chroma_v_ratio,
        ref1_v,
        ref2_v,
        sequence.chroma_width as usize,
        sequence.chroma_height as usize,
    );
    (pred_y, pred_u, pred_v)
}

#[allow(clippy::too_many_arguments)]
fn build_obmc_prediction_one_bipred(
    sequence: &SequenceHeader,
    pred: &PicturePredictionParams,
    motion: &PictureMotionData,
    is_chroma: bool,
    chroma_h_ratio: u32,
    chroma_v_ratio: u32,
    ref1_plane: &[u8],
    ref2_plane: &[u8],
    comp_w: usize,
    comp_h: usize,
) -> Vec<i32> {
    let depth = if is_chroma {
        sequence.chroma_depth
    } else {
        sequence.luma_depth
    };
    let half = 1i32 << (depth - 1);
    let ref1_signed: Vec<i32> = ref1_plane.iter().map(|&v| v as i32 - half).collect();
    let ref2_signed: Vec<i32> = ref2_plane.iter().map(|&v| v as i32 - half).collect();
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
    let mc_params = McParams {
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
        luma_depth: sequence.luma_depth,
        chroma_depth: sequence.chroma_depth,
    };
    let mut pic = vec![0i32; comp_w * comp_h];
    crate::obmc::motion_compensate(
        &mut pic,
        &mc_params,
        motion,
        Some((&ref1_signed, comp_w, comp_h)),
        Some((&ref2_signed, comp_w, comp_h)),
    );
    pic
}

/// Encode one 2-reference (bipred) core-syntax inter picture, parse
/// code `0x0A` (non-reference, AC-coded, 2 refs). Mirrors
/// [`encode_inter_picture`] but emits two reference deltas, the 2-ref
/// `block_motion_data` (incl. v2x / v2y), and feeds both reference
/// planes into the residue's OBMC prediction.
#[allow(clippy::too_many_arguments)]
pub fn encode_bipred_inter_picture(
    sequence: &SequenceHeader,
    params: &InterEncoderParams,
    picture_number: u32,
    ref1_picture_number: u32,
    ref2_picture_number: u32,
    cur_y: &[u8],
    cur_u: &[u8],
    cur_v: &[u8],
    ref1_y: &[u8],
    ref1_u: &[u8],
    ref1_v: &[u8],
    ref2_y: &[u8],
    ref2_u: &[u8],
    ref2_v: &[u8],
) -> Vec<u8> {
    assert_eq!(params.block_params_index, 1, "only preset 1 is supported");

    let mut w = BitWriter::new();

    // §12.2 picture_header.
    w.byte_align();
    w.write_uint_lit(4, picture_number);

    // §9.6.1 reference deltas — two signed deltas for ref1, ref2.
    let d1: i32 = ref1_picture_number.wrapping_sub(picture_number) as i32;
    let d2: i32 = ref2_picture_number.wrapping_sub(picture_number) as i32;
    w.write_sint(d1);
    w.write_sint(d2);
    // No retd — parse code 0x0A is non-reference.

    w.byte_align();

    // Bipred uses integer-pel ME by default (bipred_mv_precision = 0).
    // Integer-pel eliminates sub-pel interpolation convention differences
    // between our OBMC and ffmpeg's at the 2-ref blend stage, lifting
    // ffmpeg cross-decode PSNR from ~42 dB to ~50 dB. The wavelet
    // residue closes the prediction-error loop for both our decoder and
    // ffmpeg's — residue captures whatever integer-pel ME missed.
    let bmp = params.bipred_mv_precision;

    // §11.2 picture_prediction_parameters.
    write_picture_prediction_parameters(&mut w, params, bmp);
    w.byte_align();

    // §12.3 block_motion_data — 2-ref bipred path.
    let (sbx, sby, blocks_x, blocks_y) = motion_grid(sequence.luma_width, sequence.luma_height);
    let decisions = bipred_select_modes(
        cur_y,
        ref1_y,
        ref2_y,
        sequence.luma_width,
        sequence.luma_height,
        blocks_x,
        blocks_y,
        params.mv_search_range,
        bmp,
    );

    // Note: OBMC-aware ME refinement is NOT applied to the bipred path.
    // The single-reference `obmc_refine_me` optimises each reference's
    // MV independently against the source, but in a bipred B-picture the
    // decoder blends ref1 and ref2 predictions with OBMC spatial weights.
    // Refining ref1 MVs while ignoring ref2's contribution (and vice
    // versa) would shift MVs toward minimising the single-ref residue
    // rather than the blended reconstruction error — breaking the
    // self-roundtrip invariant. The wavelet residue loop closes the
    // remaining prediction error instead.

    encode_block_motion_data_bipred(&mut w, sbx, sby, blocks_x, blocks_y, &decisions);
    w.byte_align();

    // §11.3 wavelet_transform.
    if let Some(ref residue) = params.residue {
        let pred = picture_prediction_params_from(sequence, params, bmp);
        let motion = build_motion_from_bipred_grid(sbx, sby, blocks_x, blocks_y, &decisions);
        let (pred_y, pred_u, pred_v) = build_obmc_prediction_bipred(
            sequence, &pred, &motion, ref1_y, ref1_u, ref1_v, ref2_y, ref2_u, ref2_v,
        );
        let res_y = build_residue_plane(cur_y, &pred_y, sequence.luma_depth);
        let res_u = build_residue_plane(cur_u, &pred_u, sequence.chroma_depth);
        let res_v = build_residue_plane(cur_v, &pred_v, sequence.chroma_depth);

        w.write_bool(false);
        write_residue_transform_parameters(&mut w, residue);
        w.byte_align();

        let qpy_y = forward_and_quantise_residue(
            &res_y,
            sequence.luma_width,
            sequence.luma_height,
            residue,
        );
        let qpy_u = forward_and_quantise_residue(
            &res_u,
            sequence.chroma_width,
            sequence.chroma_height,
            residue,
        );
        let qpy_v = forward_and_quantise_residue(
            &res_v,
            sequence.chroma_width,
            sequence.chroma_height,
            residue,
        );
        write_residue_component_subbands(&mut w, residue, &qpy_y);
        write_residue_component_subbands(&mut w, residue, &qpy_u);
        write_residue_component_subbands(&mut w, residue, &qpy_v);
        w.byte_align();
    } else {
        w.write_bool(true);
        w.byte_align();
    }

    w.finish()
}

/// Encode a 2-picture stream: an HQ intra reference (`0xEC`) followed
/// by a single inter (`0x09`) referencing it. This is the minimal
/// inter-decode validator — the simplest legal Dirac sequence with a
/// motion-compensated picture.
pub fn encode_intra_then_inter_stream(
    sequence: &SequenceHeader,
    intra_params: &crate::encoder::EncoderParams,
    inter_params: &InterEncoderParams,
    intra: &InterInputPicture<'_>,
    inter: &InterInputPicture<'_>,
) -> Vec<u8> {
    let sh_payload = crate::encoder::encode_sequence_header(sequence);

    // Reference: HQ intra **reference** picture so its decoded form
    // ends up in the decoder's reference buffer.
    let intra_payload = crate::encoder::encode_hq_intra_picture(
        sequence,
        intra_params,
        intra.picture_number,
        intra.y,
        intra.u,
        intra.v,
    );

    let inter_payload = encode_inter_picture(
        sequence,
        inter_params,
        inter.picture_number,
        intra.picture_number,
        inter.y,
        inter.u,
        inter.v,
        intra.y,
        intra.u,
        intra.v,
    );

    let pi_size = 13usize;
    let sh_unit_len = pi_size + sh_payload.len();
    let intra_unit_len = pi_size + intra_payload.len();
    let inter_unit_len = pi_size + inter_payload.len();

    let mut out = Vec::with_capacity(sh_unit_len + intra_unit_len + inter_unit_len + pi_size);
    write_parse_info(&mut out, 0x00, sh_unit_len as u32, 0);
    out.extend_from_slice(&sh_payload);
    // Intra **reference** picture: 0xEC (HQ ref intra). Goes into the
    // decoder's reference buffer per §15.4.
    write_parse_info(&mut out, 0xEC, intra_unit_len as u32, sh_unit_len as u32);
    out.extend_from_slice(&intra_payload);
    // Inter non-reference, 1 reference, AC-coded core syntax: 0x09.
    write_parse_info(&mut out, 0x09, inter_unit_len as u32, intra_unit_len as u32);
    out.extend_from_slice(&inter_payload);
    write_parse_info(&mut out, 0x10, 0, inter_unit_len as u32);
    out
}

/// Two-frame YUV pair for the inter-encoder fixtures.
pub type TranslatingPair64 = (
    [u8; 64 * 64],
    [u8; 32 * 32],
    [u8; 32 * 32],
    [u8; 64 * 64],
    [u8; 32 * 32],
    [u8; 32 * 32],
);

/// Build a synthetic translating-square 64x64 4:2:0 YUV pair. Frame 0
/// is a centred bright square; frame 1 has the same square shifted by
/// `(dx, dy)` luma pels. Useful for the inter-decode validator.
pub fn synthetic_translating_pair_64(dx: i32, dy: i32) -> TranslatingPair64 {
    let mut y0 = [40u8; 64 * 64];
    let mut y1 = [40u8; 64 * 64];
    let u_const = [128u8; 32 * 32];
    let v_const = [128u8; 32 * 32];

    // 16x16 bright square at (24, 24) in frame 0.
    for r in 0..16usize {
        for c in 0..16usize {
            y0[(24 + r) * 64 + (24 + c)] = 220;
        }
    }
    // Same square shifted by (dx, dy) in frame 1.
    for r in 0..16i32 {
        for c in 0..16i32 {
            let rr = (24 + r) + dy;
            let cc = (24 + c) + dx;
            if (0..64).contains(&rr) && (0..64).contains(&cc) {
                y1[rr as usize * 64 + cc as usize] = 220;
            }
        }
    }
    // For 4:2:0 we keep chroma constant — chroma motion isn't tested
    // by this fixture (the square is luma-only), but the MC pipeline
    // still has to handle the 2x downsampled MV grid without losing
    // the constant chroma background.
    let _ = ChromaFormat::Yuv420;
    (y0, u_const, v_const, y1, u_const, v_const)
}

/// Build a synthetic **camera-pan** 64x64 4:2:0 YUV pair with **sub-pel**
/// horizontal motion. Frame 0 carries a high-frequency vertical-bar
/// pattern; frame 1 holds the same content panned by `dx_qpel` quarter-pel
/// units in x and `dy_qpel` quarter-pel units in y. The fractional
/// translation is synthesised via a separable 4-tap bicubic resample so
/// the resulting picture remains smooth at sub-pel offsets — any
/// integer-pel-only ME bottoms out at the nearest integer MV and leaves
/// a non-zero residue, while sub-pel ME can lock onto the true offset
/// and reduce the residue dramatically.
pub fn synthetic_camera_pan_64(dx_qpel: i32, dy_qpel: i32) -> TranslatingPair64 {
    // Frame 0: vertical bars (period 8 luma) on a uniform background —
    // a feature that's high-frequency in x but DC in y, perfect for
    // exposing horizontal sub-pel error.
    let mut y0 = [0u8; 64 * 64];
    for r in 0..64usize {
        for c in 0..64usize {
            // Cosine-shaped vertical bars: amplitude 80 around midgrey 128.
            // Values stay strictly within 0..=255 so the encoder doesn't
            // clip on the integer round-trip.
            let phase = (c as f64) * std::f64::consts::PI / 4.0;
            let v = 128.0 + 80.0 * phase.cos();
            y0[r * 64 + c] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    // Frame 1: same pattern translated by (dx_qpel/4, dy_qpel/4) pels.
    // We resample by direct cosine evaluation at the new phase — this
    // is the analytical truth, not a bicubic, so frame 1 is *exactly*
    // a sub-pel-shifted copy of frame 0 (modulo edge effects).
    let mut y1 = [0u8; 64 * 64];
    let dx_pel = dx_qpel as f64 / 4.0;
    let _dy_pel = dy_qpel as f64 / 4.0;
    for r in 0..64usize {
        for c in 0..64usize {
            // The cosine pattern is constant in y, so dy only affects
            // the boundary handling; we keep it analytical too.
            let new_c = (c as f64) - dx_pel;
            let phase = new_c * std::f64::consts::PI / 4.0;
            let v = 128.0 + 80.0 * phase.cos();
            y1[r * 64 + c] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    let u_const = [128u8; 32 * 32];
    let v_const = [128u8; 32 * 32];
    (y0, u_const, v_const, y1, u_const, v_const)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::picture_inter::{decode_block_motion_data, parse_picture_prediction_parameters};
    use crate::sequence::parse_sequence_header;

    #[test]
    fn motion_grid_matches_decoder_for_64x64() {
        // 64x64 / (4*4) = 4 superblocks per axis → 16 blocks per axis.
        let (sbx, sby, bx, by) = motion_grid(64, 64);
        assert_eq!((sbx, sby, bx, by), (4, 4, 16, 16));
    }

    #[test]
    fn motion_grid_pads_to_superblock_boundary() {
        // 70x70 → ceil(70 / 16) = 5 superblocks per axis.
        let (sbx, sby, bx, by) = motion_grid(70, 70);
        assert_eq!((sbx, sby, bx, by), (5, 5, 20, 20));
    }

    #[test]
    fn full_search_finds_pure_translation() {
        // Build two 32x32 frames where frame 1 is frame 0 shifted +5 in
        // x and -3 in y. Per §15.8.7 `pixel_pred` the MV semantics are
        // `ref_pos = cur_pos + mv` — so a forward feature shift of
        // (+5, -3) becomes a per-block MV of (-5, +3) (current must
        // look "back" to the unshifted source position).
        let mut y0 = vec![100u8; 32 * 32];
        let mut y1 = vec![100u8; 32 * 32];
        for r in 8..16 {
            for c in 8..16 {
                y0[r * 32 + c] = 200;
            }
        }
        for r in 5..13 {
            // r = 8 + (-3)
            for c in 13..21 {
                // c = 8 + 5
                y1[r * 32 + c] = 200;
            }
        }
        // 32x32 blocks_x = 32/4 = 8, blocks_y = 8. Current = y1, ref = y0.
        let mvs = full_search_me(&y1, &y0, 32, 32, 8, 8, 6);
        // Block (3, 1) covers cur pixels [12..20) x [4..12). It's the
        // top-left corner of the shifted square (which sits at
        // r=5..13, c=13..21 in y1). Best ref offset: (-5, +3) so the
        // 8x8 block lines up with the unshifted square at r=8..16,
        // c=8..16 in y0.
        let target_block = mvs[8 + 3];
        assert_eq!(
            (target_block.0, target_block.1),
            (-5, 3),
            "ME should find pure translation; got ({}, {})",
            target_block.0,
            target_block.1
        );
    }

    /// Encoded `block_motion_data` should round-trip through the
    /// decoder's parser, recovering the same MVs at every block.
    #[test]
    fn block_motion_data_roundtrips_through_decoder() {
        // Tiny 16x16 luma → 1 superblock with 4x4 = 16 blocks at split=2.
        let sbx = 1u32;
        let sby = 1u32;
        let bx = 4u32;
        let by = 4u32;
        let mvs: Vec<IntegerMv> = (0..16i32)
            .map(|i| IntegerMv((i % 4) - 1, (i / 4) - 1))
            .collect();
        let mut w = BitWriter::new();
        encode_block_motion_data(&mut w, sbx, sby, bx, by, &mvs);
        let bytes = w.finish();

        // Parse: simulate `parse_picture_prediction_parameters` having
        // already produced a PicturePredictionParams with these dims.
        // Manually build the PPP since we're skipping that header.
        use crate::picture_inter::PicturePredictionParams;
        let pred = PicturePredictionParams {
            luma_xblen: 8,
            luma_yblen: 8,
            luma_xbsep: 4,
            luma_ybsep: 4,
            mv_precision: 0,
            using_global: false,
            prediction_mode: 0,
            superblocks_x: sbx,
            superblocks_y: sby,
            blocks_x: bx,
            blocks_y: by,
            refs_wt_precision: 1,
            ref1_wt: 1,
            ref2_wt: 1,
            global1: None,
            global2: None,
        };
        let mut r = crate::bits::BitReader::new(&bytes);
        let motion = decode_block_motion_data(&mut r, &pred, 1).expect("decode motion");
        // Every superblock split should be 2 (we sent split=2 only).
        for &s in &motion.sb_split {
            assert_eq!(s, 2);
        }
        for by_ in 0..by {
            for bx_ in 0..bx {
                let i = (by_ * bx + bx_) as usize;
                let blk = &motion.blocks[i];
                assert_eq!(blk.rmode, RefPredMode::Ref1Only, "block {i} rmode");
                assert_eq!(
                    blk.mv[0],
                    (mvs[i].0, mvs[i].1),
                    "block {i} MV mismatch: got {:?}, want ({}, {})",
                    blk.mv[0],
                    mvs[i].0,
                    mvs[i].1
                );
            }
        }
    }

    /// `parse_picture_prediction_parameters` should accept the bytes
    /// emitted by `write_picture_prediction_parameters` and produce
    /// matching dims for our standard 64x64 grid. We sweep the supported
    /// `mv_precision` values to make sure the §11.2.5 read-uint maps
    /// back to the encoder-chosen precision.
    #[test]
    fn picture_prediction_parameters_roundtrips() {
        let seq = crate::encoder::make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
        let _ = parse_sequence_header; // re-export check.
        for mv_precision in [0u32, 1, 2, 3] {
            let mut w = BitWriter::new();
            let params = InterEncoderParams {
                mv_precision,
                ..InterEncoderParams::default()
            };
            write_picture_prediction_parameters(&mut w, &params, mv_precision);
            w.byte_align();
            let bytes = w.finish();

            let mut r = crate::bits::BitReader::new(&bytes);
            let pred = parse_picture_prediction_parameters(&mut r, &seq, 1).expect("parse PPP");
            assert_eq!(pred.luma_xblen, 8);
            assert_eq!(pred.luma_yblen, 8);
            assert_eq!(pred.luma_xbsep, 4);
            assert_eq!(pred.luma_ybsep, 4);
            assert_eq!(pred.mv_precision, mv_precision);
            assert!(!pred.using_global);
            assert_eq!(pred.superblocks_x, 4);
            assert_eq!(pred.superblocks_y, 4);
            assert_eq!(pred.blocks_x, 16);
            assert_eq!(pred.blocks_y, 16);
            assert_eq!(pred.refs_wt_precision, 1);
            assert_eq!(pred.ref1_wt, 1);
            assert_eq!(pred.ref2_wt, 1);
        }
    }

    /// At `mv_precision == 0`, [`subpel_search_me`] must be identical
    /// to [`full_search_me`].
    #[test]
    fn subpel_search_degenerates_to_integer_at_precision_0() {
        let (y0, _, _, y1, _, _) = synthetic_translating_pair_64(4, 0);
        let int_mvs = full_search_me(&y1, &y0, 64, 64, 16, 16, 16);
        let sub_mvs = subpel_search_me(&y1, &y0, 64, 64, 16, 16, 16, 0);
        assert_eq!(int_mvs.len(), sub_mvs.len());
        for (a, b) in int_mvs.iter().zip(sub_mvs.iter()) {
            assert_eq!((a.0, a.1), (b.0, b.1));
        }
    }

    /// Quarter-pel ME on an integer translation should produce the
    /// integer MV scaled by 4 (one quarter-pel unit per i32 step).
    #[test]
    fn subpel_search_qpel_matches_integer_scaled_for_integer_motion() {
        let (y0, _, _, y1, _, _) = synthetic_translating_pair_64(4, 0);
        let int_mvs = full_search_me(&y1, &y0, 64, 64, 16, 16, 16);
        let sub_mvs = subpel_search_me(&y1, &y0, 64, 64, 16, 16, 16, 2);
        for (i, (a, b)) in int_mvs.iter().zip(sub_mvs.iter()).enumerate() {
            assert_eq!(
                (b.0, b.1),
                (a.0 * 4, a.1 * 4),
                "block {i}: sub-pel MV {:?} not equal to integer MV {:?} × 4",
                (b.0, b.1),
                (a.0, a.1)
            );
        }
    }

    /// On the camera-pan fixture (vertical bars panned by 1 quarter-pel
    /// unit) the sub-pel search should snap to a non-integer MV at
    /// least somewhere inside the picture interior — proving the sub-pel
    /// refinement is actually firing rather than no-op'ing on the
    /// integer pass.
    #[test]
    fn subpel_search_finds_subpel_motion_on_pan_fixture() {
        let (y0, _, _, y1, _, _) = synthetic_camera_pan_64(1, 0);
        let mvs = subpel_search_me(&y1, &y0, 64, 64, 16, 16, 16, 2);
        // At least one block should have an x-MV that's not a multiple
        // of 4 — that's the signature of true sub-pel snapping.
        let any_subpel = mvs.iter().any(|m| (m.0 % 4) != 0 || (m.1 % 4) != 0);
        assert!(
            any_subpel,
            "sub-pel ME never produced a fractional MV on a 1/4-pel pan"
        );
    }

    #[test]
    fn translating_pair_has_expected_brightness_at_shift() {
        let (y0, _, _, y1, _, _) = synthetic_translating_pair_64(4, 0);
        // Frame 0: square at (24, 24)..(40, 40) is 220.
        assert_eq!(y0[24 * 64 + 24], 220);
        assert_eq!(y0[39 * 64 + 39], 220);
        // Frame 1: square shifted +4 in x → (24, 28)..(40, 44).
        assert_eq!(y1[24 * 64 + 28], 220);
        assert_eq!(y1[39 * 64 + 43], 220);
        // The original (24, 24) position is now back to background 40.
        assert_eq!(y1[24 * 64 + 24], 40);
    }

    /// `obmc_refine_me` with `passes = 0` must be a no-op — every MV
    /// stays at its sub-pel-search starting value. Anything else means
    /// the brief's "no-OBMC baseline" knob is silently doing work.
    #[test]
    fn obmc_refine_zero_passes_is_noop() {
        let (y0, _, _, y1, _, _) = synthetic_translating_pair_64(2, -1);
        let mut mvs = subpel_search_me(&y1, &y0, 64, 64, 16, 16, 16, 0);
        let snapshot: Vec<(i32, i32)> = mvs.iter().map(|m| (m.0, m.1)).collect();
        obmc_refine_me(&y1, &y0, 64, 64, 16, 16, &mut mvs, 0, 0);
        let after: Vec<(i32, i32)> = mvs.iter().map(|m| (m.0, m.1)).collect();
        assert_eq!(
            snapshot, after,
            "passes = 0 should leave the MV grid untouched"
        );
    }

    /// On a fixture with a small integer translation (+2, -1) — small
    /// enough that the bright square's edge falls inside several
    /// neighbouring blocks' OBMC overlap regions — OBMC-aware refinement
    /// should change at least one MV: the per-block SAD search picks
    /// zero-MV for blocks that hold mostly background, but the OBMC
    /// blend (which weights the square-corner block's prediction into
    /// the same overlap zone) prefers an MV that aligns with the
    /// translation. Without this convergence, the encoder leaves a
    /// noisy reconstruction across block boundaries (cf.
    /// `tests/encoder_inter_roundtrip.rs::intra_then_inter_obmc_refinement_beats_no_obmc_baseline`
    /// where the gap is ~22 dB).
    #[test]
    fn obmc_refine_changes_mv_grid_on_overlap_motion() {
        let (y0, _, _, y1, _, _) = synthetic_translating_pair_64(2, -1);
        let mut mvs_baseline = subpel_search_me(&y1, &y0, 64, 64, 16, 16, 16, 0);
        let mut mvs_refined = mvs_baseline.clone();
        obmc_refine_me(&y1, &y0, 64, 64, 16, 16, &mut mvs_refined, 0, 2);
        let mut changes = 0u32;
        for (b, r) in mvs_baseline.iter_mut().zip(mvs_refined.iter()) {
            if (b.0, b.1) != (r.0, r.1) {
                changes += 1;
            }
        }
        assert!(
            changes >= 4,
            "OBMC refinement should converge ≥ 4 blocks' MVs on the (+2, -1) translation \
             fixture — got {changes} changes (refinement is not engaging the OBMC blend)"
        );
    }

    /// Residue dead-zone forward + decoder inverse_quant must round-trip
    /// at qindex = 0 (the lossless mode the encoder uses by default).
    /// This is the §13.2 / §13.3 invariant the residue path relies on.
    #[test]
    fn residue_quantise_q0_roundtrip() {
        for &x in &[-200i32, -33, -1, 0, 1, 5, 42, 199] {
            let q = residue_quantise_coeff(x, 0);
            assert_eq!(q, x, "qindex=0 dead-zone forward should be identity");
            assert_eq!(
                crate::quant::inverse_quant(q, 0),
                x,
                "qindex=0 forward+inverse should round-trip",
            );
        }
    }

    /// `build_residue_plane` must reproduce
    /// `(source - 128) - prediction_signed` element-wise. This is the
    /// invariant that links the encoder's residue-domain math to the
    /// decoder's signed pre-output-offset reconstruction.
    #[test]
    fn build_residue_plane_subtracts_signed_prediction() {
        let src: Vec<u8> = vec![128, 200, 50, 10, 255, 0];
        let pred: Vec<i32> = vec![0, 50, -50, -100, 100, -128];
        let res = build_residue_plane(&src, &pred, 8);
        // (128-128)-0=0; (200-128)-50=22; (50-128)-(-50)=-28;
        // (10-128)-(-100)=-18; (255-128)-100=27; (0-128)-(-128)=0.
        assert_eq!(res, vec![0, 22, -28, -18, 27, 0]);
    }

    /// The `picture_prediction_params_from` helper must produce the
    /// same `PicturePredictionParams` shape the decoder reconstructs
    /// from `write_picture_prediction_parameters`. We assert the
    /// invariant by writing + reparsing and comparing the dimensions —
    /// this is the load-bearing piece for the residue encoder's OBMC
    /// reconstruction (the dims must match symbol-for-symbol).
    #[test]
    fn picture_prediction_params_from_matches_writer() {
        let seq = crate::encoder::make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
        for mv_precision in [0u32, 1, 2, 3] {
            let iep = InterEncoderParams {
                mv_precision,
                ..InterEncoderParams::default()
            };
            let mut w = BitWriter::new();
            write_picture_prediction_parameters(&mut w, &iep, mv_precision);
            w.byte_align();
            let bytes = w.finish();
            let mut r = crate::bits::BitReader::new(&bytes);
            let parsed = parse_picture_prediction_parameters(&mut r, &seq, 1).expect("PPP");
            let synthesised = picture_prediction_params_from(&seq, &iep, mv_precision);
            assert_eq!(parsed.luma_xblen, synthesised.luma_xblen);
            assert_eq!(parsed.luma_yblen, synthesised.luma_yblen);
            assert_eq!(parsed.luma_xbsep, synthesised.luma_xbsep);
            assert_eq!(parsed.luma_ybsep, synthesised.luma_ybsep);
            assert_eq!(parsed.mv_precision, synthesised.mv_precision);
            assert_eq!(parsed.using_global, synthesised.using_global);
            assert_eq!(parsed.superblocks_x, synthesised.superblocks_x);
            assert_eq!(parsed.superblocks_y, synthesised.superblocks_y);
            assert_eq!(parsed.blocks_x, synthesised.blocks_x);
            assert_eq!(parsed.blocks_y, synthesised.blocks_y);
            assert_eq!(parsed.refs_wt_precision, synthesised.refs_wt_precision);
            assert_eq!(parsed.ref1_wt, synthesised.ref1_wt);
            assert_eq!(parsed.ref2_wt, synthesised.ref2_wt);
        }
    }
}
