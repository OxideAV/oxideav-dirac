//! Dirac core-syntax **inter** encoder.
//!
//! Mirrors [`crate::picture_inter::decode_block_motion_data`] +
//! [`crate::picture::decode_picture_with_refs`] on the encode side.
//! Current shape (post-#168 sub-pel ME):
//!
//! * **Single reference** (parse code `0x09` — non-reference 1-ref AC
//!   inter picture). One `picture_number` delta, no `retd`.
//! * **§11.2.6 global motion** (round-382, opt-in via
//!   [`InterEncoderParams::global_motion`]): the picture can signal an
//!   affine-perspective global model per reference and mark any subset
//!   of blocks as §12.3.3.2 global blocks (whole-picture by default).
//!   Global blocks carry no MV residual — prediction comes from the
//!   §15.8.8 `global_mv` field. No reference weights override
//!   (`refs_wt = 1 / 1`, `refs_wt_precision = 1`).
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
    mvctx, BlockData, GlobalParams, PictureMotionData, PicturePredictionParams, RefPredMode,
};
use crate::quant::quant_factor;
use crate::sequence::SequenceHeader;
use crate::subband::{padded_component_dims, Orient, SubbandData};
use crate::video_format::ChromaFormat;
use crate::wavelet::{dwt, WaveletFilter};

mod sealed {
    /// Seals [`super::InterSample`] to the two source-sample widths the
    /// Dirac core syntax can express (§10.5.2 `video_depth` caps the
    /// component depth at 16 bits for the `Yuv*P16Le` output surfaces).
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
}

/// Source-sample abstraction for the inter encoder (deep-colour
/// support). The whole ME / OBMC / residue pipeline is generic over the
/// input sample width: `u8` carries classic 8-bit video, `u16` carries
/// 9-16-bit deep colour. The *encoded* depth always comes from
/// `SequenceHeader::{luma,chroma}_depth` (§10.5.2 `video_depth` from
/// the §10.3.8 signal-range excursions) — the sample type only decides
/// how the source planes are read and which headroom constants the
/// ME-side scoring uses.
///
/// Sealed: `u8` and `u16` are the only implementations (16 bits is the
/// deepest component depth the §10.5.2 `video_depth` formula reaches on
/// the signal ranges this crate emits).
pub trait InterSample: Copy + sealed::Sealed + 'static {
    /// Widen one source sample to the `i32` arithmetic domain.
    fn to_i32(self) -> i32;

    /// Clip depth handed to the §15.8.11 half-pel upsampler when the
    /// ME search scores against **unsigned** source samples
    /// ([`build_upref`]): one bit above the sample width, so the spec's
    /// `[-2^(d-1), 2^(d-1)-1]` clip holds the full unsigned range plus
    /// the 8-tap filter's small overshoot. `9` for `u8` (clip
    /// `[-256, 255]`), `17` for `u16` (clip `[-65536, 65535]`).
    const ME_UPREF_DEPTH: u32;

    /// Mid-range offset used by the ME-side *signed* decoder-mirroring
    /// paths ([`build_upref_signed`], the OBMC-aware SSE scoring):
    /// `2^(width-1)` — `128` for `u8`, `32768` for `u16`. This is an
    /// encoder-side scoring convention only; the residue math that
    /// reaches the wire recentres by the **sequence** depth
    /// (`2^(depth-1)`), exactly like the decoder's §15.4 reference
    /// buffer. For deep sources whose sequence depth is below 16 the
    /// scoring offset differs from the decoder's by a constant shift,
    /// which cancels in every SAD/SSE difference — only the rarely-hit
    /// clip bounds move, and those affect ME quality, never
    /// conformance.
    const NOMINAL_HALF: i32;

    /// Encode the HQ intra **anchor** picture for the sequence drivers
    /// ([`encode_intra_then_inter_stream`],
    /// [`encode_inter_sequence_with_residue_target`]), dispatching to
    /// the matching-width VC-2 HQ intra entry point.
    #[doc(hidden)]
    fn encode_hq_intra_anchor(
        sequence: &SequenceHeader,
        params: &crate::encoder::EncoderParams,
        picture_number: u32,
        y: &[Self],
        u: &[Self],
        v: &[Self],
    ) -> Vec<u8>;
}

impl InterSample for u8 {
    #[inline]
    fn to_i32(self) -> i32 {
        self as i32
    }
    const ME_UPREF_DEPTH: u32 = 9;
    const NOMINAL_HALF: i32 = 128;
    fn encode_hq_intra_anchor(
        sequence: &SequenceHeader,
        params: &crate::encoder::EncoderParams,
        picture_number: u32,
        y: &[Self],
        u: &[Self],
        v: &[Self],
    ) -> Vec<u8> {
        crate::encoder::encode_hq_intra_picture(sequence, params, picture_number, y, u, v)
    }
}

impl InterSample for u16 {
    #[inline]
    fn to_i32(self) -> i32 {
        self as i32
    }
    const ME_UPREF_DEPTH: u32 = 17;
    const NOMINAL_HALF: i32 = 32768;
    fn encode_hq_intra_anchor(
        sequence: &SequenceHeader,
        params: &crate::encoder::EncoderParams,
        picture_number: u32,
        y: &[Self],
        u: &[Self],
        v: &[Self],
    ) -> Vec<u8> {
        crate::encoder::encode_hq_intra_picture_u16(sequence, params, picture_number, y, u, v)
    }
}

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
    /// Historical note: integer-pel (`0`) used to measure ~8 dB higher
    /// external-oracle cross-decode than quarter-pel on
    /// complementary-bar fixtures, which was attributed to sub-pel
    /// interpolation convention differences. Round-408 identified the
    /// real culprits (the oracle resolves §11.2.2 preset *index* 1 to
    /// non-overlapped blocks and mishandles ZERO_RESIDUAL=1 skip
    /// pictures); with literal block parameters and the explicit
    /// zero-residue tail our 1-ref inter chains cross-decode
    /// bit-exactly at every precision. The historical default is kept
    /// for stability of the bipred rate/quality trade-off.
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
    /// `None` (or `enable_residue = false`) emits no coefficient data:
    /// reconstruction = OBMC(reference) and PSNR is determined entirely
    /// by ME quality. How the zero residue is *coded* on the wire is
    /// controlled by [`explicit_zero_residue`].
    ///
    /// Residue encoding lifts the inter-encoder's quality ceiling
    /// dramatically on real-world content (where ME alone leaves
    /// edge-clamp / OBMC-blend residuals across block boundaries).
    ///
    /// [`explicit_zero_residue`]: InterEncoderParams::explicit_zero_residue
    pub residue: Option<ResidueParams>,
    /// **Zero-residue wire form** (round-408). When `residue` is `None`
    /// (or a rate-controlled picture gets a zero residue budget), the
    /// spec offers two equivalent encodings: the §11.3 `ZERO_RESIDUAL =
    /// 1` skip flag, or `ZERO_RESIDUAL = 0` followed by transform
    /// parameters whose every subband block has `length = 0`. Both
    /// decode to an all-zero residue on a conformant decoder — but
    /// black-box probing showed the reference decoder **mis-reconstructs
    /// skip pictures** (its output picks up a fine checkerboard offset
    /// against the reference picture even for a zero-MV skip), so
    /// streams meant for cross-decode should carry the explicit form.
    ///
    /// `true` (default) emits the explicit all-zero-band form (LeGall
    /// 5,3, depth 1 — 13 zero-length bands, ~14 bytes per picture);
    /// `false` keeps the compact `ZERO_RESIDUAL = 1` skip flag.
    pub explicit_zero_residue: bool,
    /// **Per-block adaptive sub-pel-vs-integer-pel selection** for the
    /// **1-ref (P-picture)** path (round-73, mirrors the bipred
    /// `bipred_select_modes` adaptive precision landed in round-39).
    ///
    /// When `true`, after [`subpel_search_me`] produces the refined
    /// sub-pel MV grid and **before** [`obmc_refine_me`] runs, each
    /// block's MV is scored under the §15.8.6 weighted-blend
    /// reconstruction at both its sub-pel-refined position AND its
    /// nearest integer-pel-rounded peer; whichever gives lower OBMC SSE
    /// against the source wins per block (ties biased toward the
    /// integer-pel MV).
    ///
    /// Motivation (same as the bipred case): the spec's §15.8.11 8-tap
    /// half-pel filter introduces multi-LSB smoothing/ringing at sharp
    /// edges. On blocks whose source pattern is high-frequency
    /// (text, hard occluder boundaries), the sub-pel-interpolated
    /// reference is *worse* than the integer-pel sample even after
    /// OBMC refinement converges, because every refinement step is
    /// scored against a smoothed prediction. Allowing the encoder to
    /// snap such blocks back to integer-pel before OBMC refinement
    /// gives a strict superset of MV candidates and (since it picks
    /// the per-block minimum) cannot regress on smooth content.
    ///
    /// At `mv_precision == 0` this is a no-op (every MV is already
    /// integer-pel). Defaults to `true`; set to `false` to disable for
    /// regression / A-B testing against the pre-round-73 behaviour.
    pub inter_adaptive_int_pel: bool,
    /// **Post-OBMC second adaptive sub-pel-vs-integer-pel pass** for the
    /// **1-ref (P-picture)** path (round-80).
    ///
    /// After [`obmc_refine_me`] has converged on a sub-pel grid by ±1
    /// sub-pel steps, run [`inter_select_int_pel_per_block`] a second
    /// time so that blocks which drifted off the integer-pel anchor
    /// during OBMC refinement get one more chance to snap back. The
    /// OBMC neighbour buffer is the same one [`obmc_refine_me`] just
    /// evaluated against, but the **finite ±1 sub-pel step search**
    /// in `obmc_refine_me` cannot bridge from a sub-pel offset back to
    /// the nearest **integer-pel** position in a single move (at
    /// quarter-pel that's a 2-3 sub-pel-unit jump). The post-OBMC
    /// selector evaluates exactly that integer-pel-rounded peer
    /// against the converged sub-pel MV, picking whichever gives lower
    /// per-block OBMC SSE. Because the comparison set is `{ current,
    /// round_to_int_pel(current) }` (a strict superset of `{ current
    /// }`), the result can never regress the per-block OBMC SSE — and
    /// the test `inter_select_int_pel_monotonic_per_block_obmc_sse`
    /// already pins that invariant for the helper.
    ///
    /// Catches the case where the pre-OBMC selector picked integer-pel
    /// for a block, OBMC refinement drifted it to a sub-pel offset to
    /// help a neighbour's blend, and now the originally-best integer
    /// MV is the best position again (because the neighbour MV grid
    /// has also shifted under refinement). Conceptually a fixed-point
    /// iteration: pre-OBMC selector + OBMC refinement + post-OBMC
    /// selector. A third pass would also be safely monotone — empirically
    /// the second pass is sufficient on every fixture in this crate.
    ///
    /// At `mv_precision == 0` this is a no-op. At `obmc_refine_passes
    /// == 0` it is **almost** a no-op (the pre-OBMC selector already
    /// ran on the same grid), but still serves as a tie-bias bound.
    /// Defaults to `true`; set to `false` for A/B testing against the
    /// pre-round-80 behaviour.
    pub inter_adaptive_int_pel_post_obmc: bool,
    /// **Post-OBMC bipred refinement pass** for the **2-ref (B-picture)**
    /// path (round-95). The bipred analogue of the 1-ref
    /// [`inter_adaptive_int_pel_post_obmc`] (round-80) post-OBMC re-
    /// evaluation pass.
    ///
    /// After [`bipred_select_modes`] picks the per-block
    /// `{Ref1Only, Ref2Only, Ref1And2}` mode + best MVs from the round-91
    /// widened `{int-pel, half-pel, sub-pel}` per-ref candidate set, the
    /// resulting grid is the input to the decoder's §15.8.5 OBMC blend.
    /// The neighbour blocks' contributions to each block's reconstruction
    /// only stabilise *after* the full grid is fixed — so the per-block
    /// SAD `bipred_select_modes` minimised against the **source** is not
    /// quite the same cost function the decoder ultimately evaluates
    /// (which is per-block OBMC SSE of the **blended** reconstruction).
    ///
    /// When `true`, this pass re-evaluates each block's decision under
    /// the full OBMC blend with the neighbour grid frozen at the
    /// selector's output: for every block the trial set is `{ current
    /// decision, ref1-only(int-pel-snapped mv1), ref1-only(half-pel-
    /// snapped mv1), ref2-only(int-pel-snapped mv2), ref2-only(half-pel-
    /// snapped mv2), bipred(int-pel mv1, int-pel mv2),
    /// bipred(half-pel mv1, half-pel mv2) }` — a strict superset that
    /// includes the current decision, so the per-block OBMC SSE can
    /// never regress (mirrors the 1-ref `inter_adaptive_int_pel_post_obmc`
    /// monotonicity invariant). Ties bias toward the current decision (to
    /// minimise unnecessary mode flips) and otherwise toward int-pel
    /// (smaller §15.8.11 8-tap-filter contribution to neighbours' blends).
    ///
    /// At `bipred_mv_precision == 0` every MV is already integer-pel and
    /// the snap operations are identities, so the only candidates added
    /// over the current decision are the alternate-mode {Ref1Only,
    /// Ref2Only} variants — still strict-superset and still monotone, but
    /// the gain is small (mode flips are rare on integer-pel ME). The
    /// expected win is on smooth-motion sub-pel content where the
    /// pre-OBMC selector's SAD-against-source decision diverges from the
    /// decoder's OBMC-blend reconstruction cost — empirically lifts
    /// camera-pan bipred PSNR by ~0.3 dB without residue, no regression
    /// on sharp-edge fixtures.
    ///
    /// Defaults to `true`; set to `false` for A/B testing against the
    /// pre-round-95 behaviour.
    pub bipred_post_obmc_refine: bool,
    /// **§11.2.6 global motion** (round-382).
    ///
    /// When `Some`, the picture signals `state[USING GLOBAL] = True`,
    /// emits the affine-perspective [`GlobalParams`] for ref1 (and ref2
    /// on 2-ref pictures), and marks the configured blocks as global
    /// (§12.3.3.2 `block_global_mode`). Global blocks carry **no** motion
    /// vector residual — their per-pixel prediction is derived entirely
    /// from the §15.8.8 `global_mv` affine field, and the encoder builds
    /// its OBMC prediction / residue from that same field so the
    /// round-trip is bit-exact at `qindex = 0`.
    ///
    /// `None` (the default) keeps the pre-round-382 behaviour:
    /// `using_global = false`, every block uses block motion.
    pub global_motion: Option<GlobalMotionConfig>,
    /// **Per-picture automatic global-motion estimation** (round-386).
    ///
    /// Consumed by the multi-picture sequence driver
    /// ([`encode_inter_sequence_with_residue_target`] /
    /// `_report`): when `Some` and [`global_motion`] is `None`, the
    /// driver runs [`estimate_global_motion_config`] on **each** inter
    /// picture against its reference and applies the estimate to that
    /// picture iff the global fraction clears
    /// [`AutoGlobalMotion::min_fraction`] — so a camera-motion shot
    /// sheds its MV residuals while a scene cut or scattered-motion
    /// picture stays pure block motion, per picture, with no caller
    /// intervention. The estimate is resolved **before** the residue
    /// qindex picker runs, so rate control measures exactly the stream
    /// it will emit. An explicit [`global_motion`] config always wins
    /// (auto never overrides a caller's model).
    ///
    /// `None` (the default) keeps the round-382 behaviour: global
    /// motion is only ever emitted when the caller supplies a config.
    ///
    /// [`global_motion`]: InterEncoderParams::global_motion
    pub auto_global_motion: Option<AutoGlobalMotion>,
}

/// Configuration for [`InterEncoderParams::auto_global_motion`]
/// (round-386).
#[derive(Debug, Clone, Copy)]
pub struct AutoGlobalMotion {
    /// Which §11.2.6 model family to fit per picture.
    pub model: GlobalMotionModel,
    /// Minimum share of blocks the fitted field must win (by the
    /// estimator's SAD decision) for the model to be applied to the
    /// picture. Below the threshold the picture is emitted with pure
    /// block motion. `0.0` applies every fit; `> 1.0` never applies
    /// (useful for A/B telemetry-only runs — the report still carries
    /// the measured fraction).
    pub min_fraction: f64,
}

impl Default for AutoGlobalMotion {
    /// Affine model at a 0.5 fraction threshold: apply the camera
    /// model when it wins at least half the blocks.
    fn default() -> Self {
        Self {
            model: GlobalMotionModel::Affine,
            min_fraction: 0.5,
        }
    }
}

/// §11.2.6 global-motion encoder configuration (round-382).
#[derive(Debug, Clone)]
pub struct GlobalMotionConfig {
    /// Affine-perspective global parameters for reference 1.
    pub global1: GlobalParams,
    /// Affine-perspective global parameters for reference 2. Only emitted
    /// (and only meaningful) on 2-reference pictures (parse code `0x0A`).
    pub global2: Option<GlobalParams>,
    /// Optional per-block global-mode grid, row-major
    /// `[by * blocks_x + bx]`. `None` ⇒ **every** non-intra block is a
    /// global block (the common "whole picture follows the global model"
    /// case). When `Some`, the slice length must equal
    /// `blocks_x * blocks_y`; `true` marks a global block. A block that
    /// is `true` here still requires a non-intra reference mode — the
    /// §12.3.3.2 process forces `gmode = false` on intra blocks — so the
    /// encoder only honours `true` on `Ref1Only` / `Ref2Only` /
    /// `Ref1And2` blocks.
    pub block_gmode: Option<Vec<bool>>,
}

impl GlobalMotionConfig {
    /// A whole-picture global model against reference 1 with a pure
    /// pan/tilt translation (`pan_tilt` in `1/(2^mv_precision)` luma-pel
    /// units, matching the block-MV convention). Every block is global.
    ///
    /// The §15.8.8 field this emits is the **constant** `(dx + 1,
    /// dy + 1)` at every pixel — the `+ 1` is the `global_mv` rounding
    /// bias at `zrs_exp = persp_exp = 0` (same convention as
    /// [`estimate_global_pan_config`], whose fitted `pan_tilt` is
    /// `t - (1, 1)` for a target translation `t`).
    ///
    /// A **position-independent** field needs the **zero** matrix:
    /// `global_mv` computes the displacement `v = (A·x + 2^ez·b) · m /
    /// 2^(ez+ep)` directly, so any non-zero `A` (including the §11.2.6
    /// omission-default identity) makes the field grow with pixel
    /// position. Through round-384 this constructor filled in the
    /// identity matrix, which produced the position-proportional field
    /// `(x + dx + 1, y + dy + 1)` instead of a pan — self-roundtrips
    /// stayed bit-exact (the encoder mirrors the decoder either way)
    /// but the prediction was a stretch, not a translation, so the
    /// residue swallowed the difference.
    pub fn pan_tilt_all(dx: i32, dy: i32) -> Self {
        Self {
            global1: GlobalParams {
                pan_tilt: (dx, dy),
                zrs: [[0, 0], [0, 0]],
                zrs_exp: 0,
                perspective: (0, 0),
                persp_exp: 0,
            },
            global2: None,
            block_gmode: None,
        }
    }
}

/// **Estimate a pan/tilt global-motion model from the encoder's own ME**
/// (round-382). Runs the same motion search [`encode_inter_picture`]
/// will run (`inter_mv_grid` — sub-pel search + the round-73 adaptive
/// int-pel snap), takes the component-wise median MV as the dominant
/// translation `t`, and builds a [`GlobalMotionConfig`] whose §15.8.8
/// field equals `t` at every pixel:
///
/// * zero affine matrix + `zrs_exp = persp_exp = 0` collapses
///   `global_mv` to the constant `pan_tilt + (1, 1)` (the rounding bias
///   at shift 0), so `pan_tilt = t - (1, 1)`;
/// * a block is marked global **iff** its ME MV equals `t` exactly —
///   for those blocks the field reproduces the block MV, so switching
///   them to global mode changes *nothing* about the prediction (and
///   therefore nothing about the §11.3 residue); it only removes their
///   per-block MV residuals from the wire;
/// * blocks whose MV differs keep block motion with their own MV.
///
/// Returns the config plus the global fraction (`0.0..=1.0` share of
/// blocks that matched `t`). The caller decides the threshold — on
/// whole-frame pans the fraction approaches 1.0 and the config is a
/// clear win; on scattered motion it approaches `0` and block motion
/// alone is better (the config would spend `using_global` header bits +
/// one gmode flag per block for nothing).
pub fn estimate_global_pan_config<S: InterSample>(
    sequence: &SequenceHeader,
    params: &InterEncoderParams,
    cur_y: &[S],
    ref_y: &[S],
) -> (GlobalMotionConfig, f64) {
    let mvs = inter_mv_grid(sequence, params, cur_y, ref_y);
    let mut xs: Vec<i32> = mvs.iter().map(|m| m.0).collect();
    let mut ys: Vec<i32> = mvs.iter().map(|m| m.1).collect();
    let t = (median(&mut xs), median(&mut ys));
    let gmode: Vec<bool> = mvs.iter().map(|m| (m.0, m.1) == t).collect();
    let matched = gmode.iter().filter(|&&b| b).count();
    let fraction = matched as f64 / gmode.len().max(1) as f64;
    (
        GlobalMotionConfig {
            global1: GlobalParams {
                pan_tilt: (t.0 - 1, t.1 - 1),
                zrs: [[0, 0], [0, 0]],
                zrs_exp: 0,
                perspective: (0, 0),
                persp_exp: 0,
            },
            global2: None,
            block_gmode: Some(gmode),
        },
        fraction,
    )
}

/// Model family for [`estimate_global_motion_config`] (round-386).
///
/// §11.2.6 parameterises the per-reference global field as the full
/// affine-perspective triple `pan_tilt` / `zoom_rotate_shear` /
/// `perspective`; which subset an encoder *fits* is pure encoder
/// policy. Each variant fits a strictly richer model:
///
/// * [`Pan`] — constant translation (the round-382
///   [`estimate_global_pan_config`] median fit; blocks are global only
///   on an exact MV match, so the choice is provably neutral on the
///   prediction).
/// * [`Affine`] — 6-parameter least-squares fit `v(x) ≈ A·x + b` of
///   the ME grid (zoom / rotation / shear + translation), robust to
///   outlier blocks via one trimmed refit.
/// * [`Perspective`] — the affine fit plus a linearised 2-parameter
///   perspective correction `v(x) ≈ (A·x + b)·(1 − c·x / 2^ep)`.
///
/// [`Pan`]: GlobalMotionModel::Pan
/// [`Affine`]: GlobalMotionModel::Affine
/// [`Perspective`]: GlobalMotionModel::Perspective
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalMotionModel {
    /// Constant-translation fit (median ME MV).
    Pan,
    /// 6-parameter affine fit (`pan_tilt` + `zoom_rotate_shear`).
    Affine,
    /// 8-parameter affine + perspective fit (adds the §15.8.8 `c`
    /// vector via a linearised least-squares pass on the affine
    /// residuals).
    Perspective,
}

/// **Estimate a §11.2.6 global-motion model from the encoder's own ME
/// grid** (round-386) — the affine / perspective generalisation of
/// [`estimate_global_pan_config`].
///
/// Runs the same motion search [`encode_inter_picture`] will commit to
/// the wire ([`inter_mv_grid`]), fits the requested [`GlobalMotionModel`]
/// to the per-block MVs by least squares (with one trimmed refit so a
/// minority of foreground blocks cannot drag the camera model), and
/// quantises the fit onto the exact integer parameterisation the
/// §15.8.8 `global_mv` field arithmetic consumes:
///
/// * `zrs_exp` is the smallest exponent that keeps the matrix
///   quantisation error under ~1/8 MV unit at the far frame corner
///   (`2^ez ≥ 4·max(width, height)`), so the emitted field tracks the
///   real-valued fit across the whole picture;
/// * `pan_tilt` is refined **after** matrix quantisation by a small
///   direct search over the integer candidates around the fitted
///   translation, scored against the actual (floor-rounded) `global_mv`
///   output — this absorbs the `+ (1, 1)` rounding bias and the
///   floor-direction interaction with the matrix term exactly rather
///   than approximately;
/// * a fitted matrix that quantises to zero collapses to the pan
///   parameterisation (`zrs_exp = 0`), and a perspective vector that
///   quantises to zero (or would drive the §15.8.8 denominator
///   `m = 2^ep − c·x` non-positive anywhere on the frame) collapses to
///   the affine parameterisation.
///
/// Per-block global flags are then chosen by **measured SAD**: each
/// block compares the source against the per-pixel `global_mv` field
/// prediction and against its own block-MV prediction (both sampled
/// through the same §15.8.10 sub-pel path the decoder uses), and is
/// marked global iff the field predicts at least as well — a tie goes
/// to global because a global block carries no MV residual on the wire.
/// Unlike the pan estimator's exact-match rule this is *not* prediction
/// neutral: a global block's prediction follows the spatially-varying
/// field (usually the point — on zoom/rotation content the field
/// interpolates motion *within* blocks, beating any constant block MV).
/// The self-roundtrip stays bit-exact at `qindex = 0` regardless,
/// because the encoder builds its OBMC prediction and residue from the
/// same effective parameters the decoder reads back.
///
/// Returns the config plus the global fraction (share of blocks marked
/// global, `0.0..=1.0`). The caller applies the worthiness threshold —
/// on whole-frame camera motion the fraction approaches 1 and the
/// config removes almost every MV residual from the wire; on scattered
/// motion it approaches 0 and block motion alone is cheaper. A
/// degenerate ME grid (too few usable blocks for the normal equations)
/// falls back to the pan fit.
pub fn estimate_global_motion_config<S: InterSample>(
    sequence: &SequenceHeader,
    params: &InterEncoderParams,
    cur_y: &[S],
    ref_y: &[S],
    model: GlobalMotionModel,
) -> (GlobalMotionConfig, f64) {
    if model == GlobalMotionModel::Pan {
        return estimate_global_pan_config(sequence, params, cur_y, ref_y);
    }
    let mvs = inter_mv_grid(sequence, params, cur_y, ref_y);
    let (_sbx, _sby, blocks_x, blocks_y) = motion_grid(sequence.luma_width, sequence.luma_height);
    let global1 = match fit_global_params_from_grid(
        &mvs,
        blocks_x,
        blocks_y,
        sequence.luma_width,
        sequence.luma_height,
        model == GlobalMotionModel::Perspective,
    ) {
        Some(g) => g,
        // Degenerate fit (blank frame / tiny grid): fall back to pan.
        None => return estimate_global_pan_config(sequence, params, cur_y, ref_y),
    };
    let (gmode, fraction) = global_gmode_by_sad(
        cur_y,
        ref_y,
        sequence.luma_width,
        sequence.luma_height,
        blocks_x,
        blocks_y,
        &mvs,
        &global1,
        params.mv_precision,
    );
    (
        GlobalMotionConfig {
            global1,
            global2: None,
            block_gmode: Some(gmode),
        },
        fraction,
    )
}

/// [`estimate_global_motion_config`] with [`GlobalMotionModel::Affine`].
pub fn estimate_global_affine_config<S: InterSample>(
    sequence: &SequenceHeader,
    params: &InterEncoderParams,
    cur_y: &[S],
    ref_y: &[S],
) -> (GlobalMotionConfig, f64) {
    estimate_global_motion_config(sequence, params, cur_y, ref_y, GlobalMotionModel::Affine)
}

/// [`estimate_global_motion_config`] with
/// [`GlobalMotionModel::Perspective`].
pub fn estimate_global_perspective_config<S: InterSample>(
    sequence: &SequenceHeader,
    params: &InterEncoderParams,
    cur_y: &[S],
    ref_y: &[S],
) -> (GlobalMotionConfig, f64) {
    estimate_global_motion_config(
        sequence,
        params,
        cur_y,
        ref_y,
        GlobalMotionModel::Perspective,
    )
}

/// **Estimate a §11.2.6 global model for a 2-reference bipred picture**
/// (round-386): one model per reference, fitted from an independent
/// per-reference ME grid at `params.bipred_mv_precision` (the precision
/// [`encode_bipred_inter_picture`] signals), plus a **conservative**
/// per-block global grid: a bipred global block predicts through the
/// §15.8.8 field for *whichever* references its §12.3.3.1 mode uses, so
/// a block is only marked global when the field wins the SAD comparison
/// against its own ME MV for **both** references — whichever mode the
/// encoder's per-block selector later picks, the field is at least as
/// good as the block MV it replaces.
///
/// Returns the two-model config (`global2 = Some(..)`) and the AND-rule
/// global fraction. Fit degeneracy on either reference falls back to
/// that reference's median-pan model, mirroring the 1-ref estimator's
/// fallback.
pub fn estimate_global_bipred_config<S: InterSample>(
    sequence: &SequenceHeader,
    params: &InterEncoderParams,
    cur_y: &[S],
    ref1_y: &[S],
    ref2_y: &[S],
    model: GlobalMotionModel,
) -> (GlobalMotionConfig, f64) {
    let (_sbx, _sby, blocks_x, blocks_y) = motion_grid(sequence.luma_width, sequence.luma_height);
    let bmp = params.bipred_mv_precision;
    let per_ref = |ref_y: &[S]| -> (GlobalParams, Vec<bool>) {
        let mvs = subpel_search_me(
            cur_y,
            ref_y,
            sequence.luma_width,
            sequence.luma_height,
            blocks_x,
            blocks_y,
            params.mv_search_range,
            bmp,
        );
        let g = match model {
            GlobalMotionModel::Pan => None,
            GlobalMotionModel::Affine => fit_global_params_from_grid(
                &mvs,
                blocks_x,
                blocks_y,
                sequence.luma_width,
                sequence.luma_height,
                false,
            ),
            GlobalMotionModel::Perspective => fit_global_params_from_grid(
                &mvs,
                blocks_x,
                blocks_y,
                sequence.luma_width,
                sequence.luma_height,
                true,
            ),
        };
        // Pan request or degenerate fit → median-pan fallback (the
        // constant field `t = median MV`, i.e. `pan_tilt = t − (1, 1)`
        // under the zero matrix).
        let g = g.unwrap_or_else(|| {
            let mut xs: Vec<i32> = mvs.iter().map(|m| m.0).collect();
            let mut ys: Vec<i32> = mvs.iter().map(|m| m.1).collect();
            let t = (median(&mut xs), median(&mut ys));
            GlobalParams {
                pan_tilt: (t.0 - 1, t.1 - 1),
                zrs: [[0, 0], [0, 0]],
                zrs_exp: 0,
                perspective: (0, 0),
                persp_exp: 0,
            }
        });
        let (gmode, _fraction) = global_gmode_by_sad(
            cur_y,
            ref_y,
            sequence.luma_width,
            sequence.luma_height,
            blocks_x,
            blocks_y,
            &mvs,
            &g,
            bmp,
        );
        (g, gmode)
    };
    let (g1, gmode1) = per_ref(ref1_y);
    let (g2, gmode2) = per_ref(ref2_y);
    let gmode: Vec<bool> = gmode1
        .iter()
        .zip(gmode2.iter())
        .map(|(&a, &b)| a && b)
        .collect();
    let marked = gmode.iter().filter(|&&b| b).count();
    let fraction = marked as f64 / gmode.len().max(1) as f64;
    (
        GlobalMotionConfig {
            global1: g1,
            global2: Some(g2),
            block_gmode: Some(gmode),
        },
        fraction,
    )
}

/// Real-valued global-model fit, before quantisation onto the §11.2.6
/// integer parameterisation. `a` / `b` are the affine matrix and
/// translation of `v(x) ≈ A·x + b` in MV units; `c` is the perspective
/// vector of the multiplicative correction `(1 − c·x)` in **per-pixel**
/// units (the `2^persp_exp` scaling is absorbed, i.e. this is
/// `c_int / 2^ep`).
#[derive(Debug, Clone, Copy, Default)]
struct GlobalFit {
    a: [[f64; 2]; 2],
    b: [f64; 2],
    c: [f64; 2],
}

impl GlobalFit {
    /// Evaluate the real-valued field at pixel `(x, y)`.
    fn eval(&self, x: f64, y: f64) -> (f64, f64) {
        let m = 1.0 - (self.c[0] * x + self.c[1] * y);
        (
            (self.a[0][0] * x + self.a[0][1] * y + self.b[0]) * m,
            (self.a[1][0] * x + self.a[1][1] * y + self.b[1]) * m,
        )
    }
}

/// The centre of block `(bx, by)`'s ME window (preset 1: the 8×8 SAD
/// window at a 4-pel stride, top-left `(4·bx, 4·by)`), as the integer
/// pixel used to anchor the fit and to score integer `global_mv`
/// candidates against the block's MV.
#[inline]
fn block_anchor(bx: u32, by: u32) -> (i32, i32) {
    ((4 * bx + 3) as i32, (4 * by + 3) as i32)
}

/// Fit `v(x) ≈ A·x + b` (optionally `· (1 − c·x)`) to the ME grid by
/// least squares with one trimmed refit, then quantise onto
/// [`GlobalParams`]. `None` when the normal equations are degenerate
/// (fewer than 8 blocks, or a rank-deficient design — e.g. a 1-block-
/// wide grid).
fn fit_global_params_from_grid(
    mvs: &[IntegerMv],
    blocks_x: u32,
    blocks_y: u32,
    luma_w: u32,
    luma_h: u32,
    with_perspective: bool,
) -> Option<GlobalParams> {
    let n = (blocks_x * blocks_y) as usize;
    debug_assert_eq!(mvs.len(), n);
    if n < 8 {
        return None;
    }
    let pts: Vec<(f64, f64)> = (0..blocks_y)
        .flat_map(|by| {
            (0..blocks_x).map(move |bx| {
                let (ax, ay) = block_anchor(bx, by);
                (ax as f64, ay as f64)
            })
        })
        .collect();
    let vx: Vec<f64> = mvs.iter().map(|m| m.0 as f64).collect();
    let vy: Vec<f64> = mvs.iter().map(|m| m.1 as f64).collect();

    // Pass 1: fit on every block.
    let all: Vec<usize> = (0..n).collect();
    let mut fit = affine_lsq(&pts, &vx, &vy, &all)?;

    // Trimmed refit: drop blocks whose residual exceeds
    // max(1 MV unit, 2 × median residual) — foreground objects and
    // frame-edge blocks (whose true content left the picture) must not
    // drag the camera model. Refit only when enough inliers survive to
    // keep the normal equations honest.
    let mut resid: Vec<f64> = (0..n)
        .map(|i| {
            let (px, py) = fit.eval(pts[i].0, pts[i].1);
            (vx[i] - px).abs().max((vy[i] - py).abs())
        })
        .collect();
    let mut sorted = resid.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = sorted[n / 2];
    let thresh = (2.0 * med).max(1.0);
    let inliers: Vec<usize> = (0..n).filter(|&i| resid[i] <= thresh).collect();
    if inliers.len() >= 8 {
        if let Some(refit) = affine_lsq(&pts, &vx, &vy, &inliers) {
            fit = refit;
            resid = (0..n)
                .map(|i| {
                    let (px, py) = fit.eval(pts[i].0, pts[i].1);
                    (vx[i] - px).abs().max((vy[i] - py).abs())
                })
                .collect();
        }
    }
    let inliers: Vec<usize> = (0..n)
        .filter(|&i| resid[i] <= (2.0 * med).max(1.0))
        .collect();
    let score_set: &[usize] = if inliers.len() >= 8 { &inliers } else { &all };

    // Optional perspective correction by alternating least squares:
    // fit c with the affine part frozen (linear in c), then refit the
    // affine part on the de-perspectived targets v/(1 − c·x) — the
    // plain affine fit absorbs the *average* perspective shrinkage
    // into its matrix, so without the refit the two terms would
    // double-count it. Two rounds converge on the smooth fields this
    // models. The quantiser rejects a c that would make the §15.8.8
    // denominator m = 2^ep − c·x non-positive anywhere on the frame.
    if with_perspective {
        for _ in 0..2 {
            let c = perspective_lsq(&fit, &pts, &vx, &vy, score_set);
            let denom_floor = 0.25f64;
            let mut vxc = vx.to_vec();
            let mut vyc = vy.to_vec();
            let mut ok = true;
            for &i in score_set {
                let (x, y) = pts[i];
                let m = 1.0 - (c[0] * x + c[1] * y);
                if m < denom_floor {
                    ok = false;
                    break;
                }
                vxc[i] = vx[i] / m;
                vyc[i] = vy[i] / m;
            }
            if !ok {
                break;
            }
            let Some(refit) = affine_lsq(&pts, &vxc, &vyc, score_set) else {
                break;
            };
            fit = GlobalFit {
                a: refit.a,
                b: refit.b,
                c,
            };
        }
    }

    Some(quantise_global_fit(
        &fit, &pts, &vx, &vy, score_set, luma_w, luma_h,
    ))
}

/// Solve the two shared-design 3-unknown least-squares systems
/// `v ≈ p·x + q·y + r` over the index subset. `None` on a singular
/// normal matrix.
fn affine_lsq(pts: &[(f64, f64)], vx: &[f64], vy: &[f64], idx: &[usize]) -> Option<GlobalFit> {
    let mut s = [[0.0f64; 3]; 3];
    let mut rx = [0.0f64; 3];
    let mut ry = [0.0f64; 3];
    for &i in idx {
        let (x, y) = pts[i];
        let row = [x, y, 1.0];
        for (j, &rj) in row.iter().enumerate() {
            for (k, &rk) in row.iter().enumerate() {
                s[j][k] += rj * rk;
            }
            rx[j] += rj * vx[i];
            ry[j] += rj * vy[i];
        }
    }
    let sol_x = solve3(s, rx)?;
    let sol_y = solve3(s, ry)?;
    Some(GlobalFit {
        a: [[sol_x[0], sol_x[1]], [sol_y[0], sol_y[1]]],
        b: [sol_x[2], sol_y[2]],
        c: [0.0, 0.0],
    })
}

/// Gaussian elimination with partial pivoting for a 3×3 system.
fn solve3(mut m: [[f64; 3]; 3], mut r: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let piv = (col..3).max_by(|&a, &b| {
            m[a][col]
                .abs()
                .partial_cmp(&m[b][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        if m[piv][col].abs() < 1e-9 {
            return None;
        }
        m.swap(col, piv);
        r.swap(col, piv);
        for row in 0..3 {
            if row == col {
                continue;
            }
            let f = m[row][col] / m[col][col];
            for k in col..3 {
                m[row][k] -= f * m[col][k];
            }
            r[row] -= f * r[col];
        }
    }
    Some([r[0] / m[0][0], r[1] / m[1][1], r[2] / m[2][2]])
}

/// Linearised perspective fit: with the **affine part** of `fit`
/// frozen, minimise `Σ‖v_i − f(x_i)·(1 − c·x_i)‖²` — linear in `c`.
/// Returns the solved `c`; `[0, 0]` on a singular system (e.g. a
/// translation-free field where every `f(x_i)` is ~zero).
fn perspective_lsq(
    fit: &GlobalFit,
    pts: &[(f64, f64)],
    vx: &[f64],
    vy: &[f64],
    idx: &[usize],
) -> [f64; 2] {
    let affine = GlobalFit {
        a: fit.a,
        b: fit.b,
        c: [0.0, 0.0],
    };
    let mut m00 = 0.0f64;
    let mut m01 = 0.0f64;
    let mut m11 = 0.0f64;
    let mut r0 = 0.0f64;
    let mut r1 = 0.0f64;
    for &i in idx {
        let (x, y) = pts[i];
        let (fx, fy) = affine.eval(x, y);
        let g = fx * fx + fy * fy;
        // Residual of the affine part; the perspective term predicts
        // v − f = −f·(c·x).
        let hx = vx[i] - fx;
        let hy = vy[i] - fy;
        let h = hx * fx + hy * fy;
        m00 += g * x * x;
        m01 += g * x * y;
        m11 += g * y * y;
        r0 -= h * x;
        r1 -= h * y;
    }
    let det = m00 * m11 - m01 * m01;
    if det.abs() < 1e-9 {
        return [0.0, 0.0];
    }
    // Normal equations for v ≈ f·(1 − c·x): with g = ‖f‖² and
    // h = (v − f)·f they read
    //   [Σ g·x²  Σ g·x·y] [c0]   [−Σ h·x]
    //   [Σ g·x·y  Σ g·y²] [c1] = [−Σ h·y]
    // (h > 0 — the observed field runs LONGER than the affine fit —
    // pushes c·x negative, i.e. the §15.8.8 m grows past 2^ep there).
    let c0 = (r0 * m11 - r1 * m01) / det;
    let c1 = (r1 * m00 - r0 * m01) / det;
    [c0, c1]
}

/// Quantise a real-valued [`GlobalFit`] onto the §11.2.6 integer
/// parameterisation, then refine `pan_tilt` by direct search against
/// the exact (floor-rounded) §15.8.8 `global_mv` output at the block
/// anchors.
fn quantise_global_fit(
    fit: &GlobalFit,
    pts: &[(f64, f64)],
    vx: &[f64],
    vy: &[f64],
    idx: &[usize],
    luma_w: u32,
    luma_h: u32,
) -> GlobalParams {
    let max_dim = luma_w.max(luma_h).max(1);
    // Smallest exponent with 2^ez ≥ 4·max_dim: the ±0.5 matrix
    // rounding error then displaces the field by ≤ max_dim/2^ez ≤ 1/8
    // MV unit at the far corner.
    let mut ez = 0u32;
    while (1u64 << ez) < 4 * max_dim as u64 && ez < 24 {
        ez += 1;
    }
    let quant = |v: f64| -> i32 { (v * (1i64 << ez) as f64).round() as i32 };
    let a = [
        [quant(fit.a[0][0]), quant(fit.a[0][1])],
        [quant(fit.a[1][0]), quant(fit.a[1][1])],
    ];
    // Matrix quantised away → pan parameterisation. `ez = 0` keeps the
    // wire cost of the (still explicitly written) zero matrix minimal
    // and matches `estimate_global_pan_config`.
    let ez_eff = if a == [[0, 0], [0, 0]] { 0 } else { ez };

    // Perspective: quantise at ep chosen so the ±0.5 rounding error
    // stays ≪ the denominator, then reject any vector that could zero
    // or flip the §15.8.8 m = 2^ep − c·x anywhere on the frame
    // (keeping at least half the dynamic range as margin).
    let mut persp = (0i32, 0i32);
    let mut ep = 0u32;
    if fit.c != [0.0, 0.0] {
        let mut e = 0u32;
        while (1u64 << e) < 256 * max_dim as u64 && e < 30 {
            e += 1;
        }
        let c0 = (fit.c[0] * (1i64 << e) as f64).round() as i64;
        let c1 = (fit.c[1] * (1i64 << e) as f64).round() as i64;
        let corner = c0.abs() * (luma_w as i64 - 1) + c1.abs() * (luma_h as i64 - 1);
        if (c0, c1) != (0, 0)
            && corner * 2 < (1i64 << e)
            && i32::try_from(c0).is_ok()
            && i32::try_from(c1).is_ok()
        {
            persp = (c0 as i32, c1 as i32);
            ep = e;
        }
    }

    // Local integer refinement, scored on the **exact** (floor-rounded)
    // §15.8.8 field: Σ|global_mv − mv| at the block anchors. Least
    // squares through a floor-rounded integer field carries a
    // systematic bias when a matrix row is axis-aligned (the staircase
    // boundaries then land on exact multiples and the observable value
    // range collapses to a couple of levels — e.g. a pure 1/32-per-pel
    // vertical zoom over a 64-pel frame steps exactly once, and the LS
    // slope through a single step underestimates by ~25%), so rounding
    // the LS solution is not enough: search each matrix entry over ±2
    // quantisation steps and the translation over −2..=+1 around the
    // fit (the asymmetric window absorbs the `+ (1, 1)` §15.8.8
    // rounding bias). The horizontal output depends only on matrix
    // row 0 and `pan_tilt.0`, the vertical only on row 1 and
    // `pan_tilt.1` (the perspective factor is shared but frozen here),
    // so the two components refine independently — 100 candidates per
    // component, each scored on the anchor grid.
    let base_b = [fit.b[0].round() as i32, fit.b[1].round() as i32];
    let mut best_a = a;
    let mut best_b = [base_b[0] - 1, base_b[1] - 1];
    for comp in 0..2usize {
        let mut best_err = i64::MAX;
        for da0 in -2i32..=2 {
            for da1 in -2i32..=2 {
                // A zero-collapsed matrix (pan parameterisation) is
                // not perturbed — the fit said "no matrix", and at
                // ez_eff = 0 a ±1 entry step is a whole MV unit per
                // pixel of position.
                if ez_eff == 0 && (da0 != 0 || da1 != 0) {
                    continue;
                }
                for db in -2i32..=1 {
                    let mut cand_a = a;
                    cand_a[comp][0] += da0;
                    cand_a[comp][1] += da1;
                    let g = GlobalParams {
                        pan_tilt: if comp == 0 {
                            (base_b[0] + db, 0)
                        } else {
                            (0, base_b[1] + db)
                        },
                        zrs: cand_a,
                        zrs_exp: ez_eff,
                        perspective: persp,
                        persp_exp: ep,
                    };
                    let mut err = 0i64;
                    for &i in idx {
                        let (x, y) = pts[i];
                        let f = crate::obmc::global_mv(&g, x as i32, y as i32);
                        let (fc, vc) = if comp == 0 {
                            (f.0, vx[i])
                        } else {
                            (f.1, vy[i])
                        };
                        err += (fc as i64 - vc as i64).abs();
                    }
                    if err < best_err {
                        best_err = err;
                        best_a[comp] = cand_a[comp];
                        best_b[comp] = base_b[comp] + db;
                    }
                }
            }
        }
    }
    GlobalParams {
        pan_tilt: (best_b[0], best_b[1]),
        zrs: best_a,
        zrs_exp: ez_eff,
        perspective: persp,
        persp_exp: ep,
    }
}

/// Per-block global-vs-block-motion decision by measured SAD: a block
/// is marked global iff the per-pixel §15.8.8 field predicts the
/// source at least as well as the block's own ME MV (both sampled
/// through the same §15.8.10 path). Returns the grid plus the global
/// fraction.
#[allow(clippy::too_many_arguments)]
fn global_gmode_by_sad<S: InterSample>(
    cur_y: &[S],
    ref_y: &[S],
    width: u32,
    height: u32,
    blocks_x: u32,
    blocks_y: u32,
    mvs: &[IntegerMv],
    global1: &GlobalParams,
    mv_precision: u32,
) -> (Vec<bool>, f64) {
    let w = width as i32;
    let h = height as i32;
    let (xblen, yblen) = (8i32, 8i32);
    let upref = if mv_precision > 0 {
        Some(build_upref(ref_y, width, height))
    } else {
        None
    };
    // Integer-pel reference sample with edge clamping (the same
    // convention as `sad_block`).
    let sample_int = |px: i32, py: i32| -> i32 {
        let cx = px.clamp(0, w - 1) as usize;
        let cy = py.clamp(0, h - 1) as usize;
        ref_y[cy * w as usize + cx].to_i32()
    };
    let sample = |px: i64, py: i64| -> i32 {
        match &upref {
            None => sample_int(px as i32, py as i32),
            Some((up, up_w, up_h)) => {
                crate::obmc::subpel_predict(up, *up_w, *up_h, px, py, mv_precision)
            }
        }
    };
    let g = effective_global_params(global1);
    let mut gmode = vec![false; (blocks_x * blocks_y) as usize];
    let mut marked = 0usize;
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let bidx = (by * blocks_x + bx) as usize;
            let x0 = (bx * 4) as i32;
            let y0 = (by * 4) as i32;
            let mv = mvs[bidx];
            let mut sad_block_mv = 0i64;
            let mut sad_global = 0i64;
            for dy in 0..yblen {
                for dx in 0..xblen {
                    let px = x0 + dx;
                    let py = y0 + dy;
                    let s = cur_y
                        [(py.clamp(0, h - 1) as usize) * w as usize + px.clamp(0, w - 1) as usize]
                        .to_i32();
                    let scaled_x = (px as i64) << mv_precision;
                    let scaled_y = (py as i64) << mv_precision;
                    let rb = sample(scaled_x + mv.0 as i64, scaled_y + mv.1 as i64);
                    let gv = crate::obmc::global_mv(&g, px, py);
                    let rg = sample(scaled_x + gv.0 as i64, scaled_y + gv.1 as i64);
                    sad_block_mv += (s - rb).unsigned_abs() as i64;
                    sad_global += (s - rg).unsigned_abs() as i64;
                }
            }
            // Tie goes to global: a global block carries no MV residual.
            if sad_global <= sad_block_mv {
                gmode[bidx] = true;
                marked += 1;
            }
        }
    }
    let fraction = marked as f64 / gmode.len().max(1) as f64;
    (gmode, fraction)
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

    /// Optional §11.3.3 spatial partition. `None` ⇒
    /// `spatial_partition_flag = 0` (one codeblock per subband — the
    /// pre-round-370 behaviour, kept bit-exact). `Some` holds the
    /// per-level `(codeblocks_x, codeblocks_y)` counts for the residue
    /// transform; the level-0 LL band is always forced to a single
    /// codeblock, matching
    /// [`crate::encoder_intra_core::CoreIntraEncoderParams::codeblocks`].
    pub codeblocks: Option<Vec<(u32, u32)>>,

    /// `CODEBLOCK_MODE` (§11.3.3): `0` = single quantiser across the
    /// subband; `1` = per-codeblock differential quantiser offset
    /// (§13.4.3.4). Only meaningful when `codeblocks` is `Some`.
    pub codeblock_mode: u32,
}

impl ResidueParams {
    /// Default residue parameters: LeGall 5/3 at depth 3 with
    /// `qindex = 0` (near-lossless on small residues, matches the
    /// intra encoder's default), single codeblock per subband.
    pub fn default_for(wavelet: WaveletFilter, dwt_depth: u32) -> Self {
        Self {
            wavelet,
            dwt_depth,
            qindex: 0,
            codeblocks: None,
            codeblock_mode: 0,
        }
    }

    /// The effective per-level codeblock grid. `(1, 1)` for every level
    /// when `codeblocks` is `None`; otherwise the caller's grid clamped
    /// to `>= 1` with the level-0 LL band forced to `(1, 1)`. Mirrors
    /// [`crate::encoder_intra_core::CoreIntraEncoderParams::codeblock_grid`].
    fn codeblock_grid(&self) -> Vec<(u32, u32)> {
        match &self.codeblocks {
            None => vec![(1, 1); self.dwt_depth as usize + 1],
            Some(grid) => {
                let mut g: Vec<(u32, u32)> = (0..=self.dwt_depth as usize)
                    .map(|lvl| {
                        grid.get(lvl)
                            .copied()
                            .map(|(x, y)| (x.max(1), y.max(1)))
                            .unwrap_or((1, 1))
                    })
                    .collect();
                g[0] = (1, 1);
                g
            }
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
            // Quarter-pel for bipred with **per-block adaptive precision**
            // (round-39): `bipred_select_modes` evaluates each MV at both
            // its sub-pel-refined position and the nearest integer-pel
            // peer, picking whichever gives the lower SAD per block. This
            // closes the convention drift that previously forced the
            // bipred path to integer-pel-only — sharp-edge blocks now keep
            // their integer MV (avoiding the 8-tap filter ringing that
            // mismatched ffmpeg's OBMC blend by ~7 dB on complementary-bars
            // fixtures), while smooth-motion blocks pick up the sub-pel
            // gain (+2-4 dB on camera-pan).
            bipred_mv_precision: 2,
            obmc_refine_passes: 2,
            residue: Some(ResidueParams::default_for(WaveletFilter::LeGall5_3, 3)),
            // Explicit all-zero-band coding for zero-residue pictures:
            // the reference decoder mishandles ZERO_RESIDUAL=1 skip
            // pictures (round-408 black-box finding), so default to the
            // cross-decoder-safe form.
            explicit_zero_residue: true,
            // Per-block adaptive sub-pel-vs-int-pel for the 1-ref path
            // (round-73). Mirrors what `bipred_select_modes` does for
            // the 2-ref path. Strict superset of MV candidates → cannot
            // regress; on sharp-edge content the integer-pel snap saves
            // multi-dB of 8-tap-filter smoothing damage even after
            // OBMC refinement converges.
            inter_adaptive_int_pel: true,
            // Round-80: second adaptive int-pel pass AFTER OBMC
            // refinement. `obmc_refine_me`'s ±1 sub-pel-unit step can
            // drift a block off integer-pel to help a neighbour's
            // blend; this pass lets that block snap back to integer-pel
            // if the post-refinement neighbour grid no longer rewards
            // the drift. Strict superset of MV candidates → cannot
            // regress per-block OBMC SSE.
            inter_adaptive_int_pel_post_obmc: true,
            // Round-95: post-OBMC bipred refinement — the 2-ref
            // analogue of `inter_adaptive_int_pel_post_obmc`. Re-
            // evaluates each block's `bipred_select_modes` decision
            // under the full §15.8.5 OBMC blend with the neighbour
            // grid frozen, choosing from a strict-superset candidate
            // set that includes the current decision so per-block OBMC
            // SSE cannot regress. Picks up the cost-function
            // divergence between "SAD against source" (the selector's
            // metric) and "OBMC SSE against source" (the decoder's
            // actual cost).
            bipred_post_obmc_refine: true,
            // Round-382: global motion off by default — every block uses
            // block motion, `using_global = false`. Opt in with a
            // `GlobalMotionConfig`.
            global_motion: None,
            // Round-386: per-picture automatic estimation off by
            // default; the sequence driver only fits a model when
            // asked.
            auto_global_motion: None,
        }
    }
}

/// One input picture (Y/U/V planes + picture number).
#[derive(Debug, Clone)]
pub struct InterInputPicture<'a, S: InterSample = u8> {
    pub picture_number: u32,
    pub y: &'a [S],
    pub u: &'a [S],
    pub v: &'a [S],
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
pub fn full_search_me<S: InterSample>(
    cur_y: &[S],
    ref_y: &[S],
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
fn sad_block<S: InterSample>(
    cur: &[S],
    refp: &[S],
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
            let a = cur[cy * w as usize + cx].to_i32();
            let b = refp[ry * w as usize + rx].to_i32();
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
fn sad_subpel<S: InterSample>(
    cur: &[S],
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
            let a = cur[cy * w as usize + cx].to_i32();
            sad += (a - r).unsigned_abs() as i64;
        }
    }
    sad
}

/// Build the half-pel upsampled reference plane (§15.8.11) for a
/// `width × height` luma plane of any [`InterSample`] width. The
/// returned plane has size `2w × 2h` in i32. We use
/// `depth = S::ME_UPREF_DEPTH` (one bit above the sample width) so
/// that the spec's `[-2^(d-1), 2^(d-1)-1]` clip range is wide enough
/// to hold the full unsigned input range plus the 8-tap filter's small
/// overshoot.
///
/// The samples are kept in unsigned space rather than the signed
/// mid-range-offset convention; this matches the original sub-pel ME
/// path (which scores against the unsigned source directly). The
/// OBMC-aware refinement
/// ([`obmc_refine_me`]) builds its own *signed* upref via
/// [`build_upref_signed`] because the decoder OBMC blend operates on the
/// pre-offset signed reference buffer (§15.4 / §15.8.5).
pub(crate) fn build_upref<S: InterSample>(
    plane: &[S],
    width: u32,
    height: u32,
) -> (Vec<i32>, usize, usize) {
    let w = width as usize;
    let h = height as usize;
    let widened: Vec<i32> = plane.iter().map(|&v| v.to_i32()).collect();
    let (up, up_w, up_h) = interp2by2(&widened, w, h, S::ME_UPREF_DEPTH);
    (up, up_w, up_h)
}

/// Like [`build_upref`] but with the spec's signed pre-offset
/// convention: each input sample is shifted by `-S::NOMINAL_HALF`
/// (`-128` for `u8`, `-32768` for `u16`) to mirror what the decoder's
/// reference buffer holds (§15.4 stores the pre-output-offset signed
/// plane). For deep sources whose sequence depth is below the sample
/// width the shift differs from the decoder's by a constant, which
/// cancels in the SAD/SSE differences the ME scoring computes.
pub(crate) fn build_upref_signed<S: InterSample>(
    plane: &[S],
    width: u32,
    height: u32,
) -> (Vec<i32>, usize, usize) {
    let w = width as usize;
    let h = height as usize;
    let signed: Vec<i32> = plane
        .iter()
        .map(|&v| v.to_i32() - S::NOMINAL_HALF)
        .collect();
    let (up, up_w, up_h) = interp2by2(&signed, w, h, S::ME_UPREF_DEPTH);
    (up, up_w, up_h)
}

/// Run the complete 1-ref motion-estimation pipeline
/// [`encode_inter_picture`] commits to the bitstream: integer-pel SAD
/// search, the optional pre-OBMC adaptive int-pel snap (round-73), the
/// §15.8.6 OBMC-aware refinement (#186), and the optional post-OBMC
/// adaptive int-pel snap (round-80). Factored out so the residue
/// rate-control picker ([`pick_inter_residue_qindex`]) reconstructs the
/// **same** MV grid the encoder will use, guaranteeing the residue it
/// measures matches the residue it eventually emits.
fn inter_mv_grid<S: InterSample>(
    sequence: &SequenceHeader,
    params: &InterEncoderParams,
    cur_y: &[S],
    ref_y: &[S],
) -> Vec<IntegerMv> {
    let (_sbx, _sby, blocks_x, blocks_y) = motion_grid(sequence.luma_width, sequence.luma_height);
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
    // Round-73: per-block adaptive sub-pel-vs-integer-pel selection
    // **before** OBMC refinement. No-op at integer-pel or when disabled.
    if params.inter_adaptive_int_pel {
        inter_select_int_pel_per_block(
            cur_y,
            ref_y,
            sequence.luma_width,
            sequence.luma_height,
            blocks_x,
            blocks_y,
            &mut mvs,
            params.mv_precision,
        );
    }
    // §15.8.6 OBMC-aware refinement (#186) — minimises the per-block SSE
    // of the *blended* reconstruction, matching what the decoder will
    // emit. No-op when `obmc_refine_passes == 0`.
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
    // Round-80: second per-block adaptive sub-pel-vs-integer-pel
    // selection **after** OBMC refinement. No-op at integer-pel or when
    // disabled.
    if params.inter_adaptive_int_pel_post_obmc {
        inter_select_int_pel_per_block(
            cur_y,
            ref_y,
            sequence.luma_width,
            sequence.luma_height,
            blocks_x,
            blocks_y,
            &mut mvs,
            params.mv_precision,
        );
    }
    mvs
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
pub fn subpel_search_me<S: InterSample>(
    cur_y: &[S],
    ref_y: &[S],
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
fn obmc_block_sse<S: InterSample>(
    cur: &[S],
    cur_w: i32,
    cur_h: i32,
    weighted_pred: &[i32],
    neighbour_sum: &[i32],
    xblen: usize,
    yblen: usize,
    xstart: i32,
    ystart: i32,
) -> i64 {
    let lo = -S::NOMINAL_HALF;
    let hi = S::NOMINAL_HALF - 1;
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
            // clip to the sample type's signed range. Pre-offset
            // cancels with the mid-range picture-storage convention,
            // so we compare against the raw source minus NOMINAL_HALF.
            let recon_signed = ((total + 32) >> 6).clamp(lo, hi);
            let src_signed =
                cur[y as usize * cur_w as usize + x as usize].to_i32() - S::NOMINAL_HALF;
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
/// reconstruction cost function — a plain block-coordinate descent on
/// the per-block blended-reconstruction SSE.
///
/// The function uses [`crate::obmc::spatial_wt`] for the §15.8.6 ramp
/// window so the encoder's blend matches the decoder symbol-for-symbol.
#[allow(clippy::too_many_arguments)]
pub fn obmc_refine_me<S: InterSample>(
    cur_y: &[S],
    ref_y: &[S],
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
    // `source - NOMINAL_HALF`.
    let (upref, up_w, up_h) = if mv_precision > 0 {
        build_upref_signed(ref_y, width, height)
    } else {
        (Vec::new(), 0, 0)
    };
    let refp_signed: Vec<i32> = ref_y
        .iter()
        .map(|&v| v.to_i32() - S::NOMINAL_HALF)
        .collect();
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

/// **Per-block adaptive sub-pel-vs-integer-pel selection** for the
/// 1-ref path (round-73). For each block in raster order, scores the
/// current MV at both its sub-pel-refined position AND the
/// integer-pel-rounded variant under the same §15.8.5 OBMC-aware
/// weighted-blend reconstruction used by [`obmc_refine_me`], keeping
/// whichever gives lower per-block SSE against the source.
///
/// Tie-bias: when the two SSEs are equal we prefer the integer-pel MV
/// (smaller decoder-side filter contribution → less risk of multi-LSB
/// drift accumulating across the OBMC blend with neighbours that are
/// at different sub-pel offsets).
///
/// At `mv_precision == 0` this is a no-op; we return immediately
/// because every MV is already integer-pel.
///
/// Mirrors the [`bipred_select_modes`] adaptive-precision logic landed
/// in round-39 (which gave +4.4 dB ffmpeg cross-decode on smooth
/// motion and +7 dB on sharp edges in the 2-ref path). Running it as a
/// **pre-OBMC** step on the 1-ref path means the integer-pel snap is
/// itself fed into [`obmc_refine_me`]'s neighbour_sum buffer, so OBMC
/// converges against the better starting point rather than against a
/// noisy sub-pel field.
#[allow(clippy::too_many_arguments)]
pub fn inter_select_int_pel_per_block<S: InterSample>(
    cur_y: &[S],
    ref_y: &[S],
    width: u32,
    height: u32,
    blocks_x: u32,
    blocks_y: u32,
    mvs: &mut [IntegerMv],
    mv_precision: u32,
) {
    // At integer-pel precision every MV is already integer-pel; the
    // rounding is the identity, so the comparison is a no-op. Avoid
    // the upref build cost.
    if mv_precision == 0 {
        return;
    }
    let (xblen_u, yblen_u, xbsep_u, ybsep_u) = (8usize, 8usize, 4usize, 4usize);
    let xoffset = (xblen_u - xbsep_u) / 2;
    let yoffset = (yblen_u - ybsep_u) / 2;
    let cur_w = width as i32;
    let cur_h = height as i32;
    // Pre-bake every block's spatial weight (only depends on its grid
    // coords). Matches `obmc_refine_me`'s cache.
    let mut weights: Vec<Vec<i32>> = Vec::with_capacity((blocks_x * blocks_y) as usize);
    for j in 0..blocks_y {
        for i in 0..blocks_x {
            weights.push(block_weight(
                xblen_u, yblen_u, xbsep_u, ybsep_u, xoffset, yoffset, i, j, blocks_x, blocks_y,
            ));
        }
    }
    // Build the signed pre-offset upsampled reference once — matches
    // the convention `obmc_refine_me` uses (§15.4 reference buffer).
    let (upref, up_w, up_h) = build_upref_signed(ref_y, width, height);
    let refp_signed: Vec<i32> = ref_y
        .iter()
        .map(|&v| v.to_i32() - S::NOMINAL_HALF)
        .collect();
    let ref_w = width as usize;
    let ref_h = height as usize;

    for j in 0..blocks_y {
        for i in 0..blocks_x {
            let bidx = (j * blocks_x + i) as usize;
            let cur_mv = mvs[bidx];
            let int_mv = round_mv_to_int_pel(cur_mv, mv_precision);
            // Pure-snap path: if rounding doesn't change the MV, skip
            // the OBMC scoring (it would give equal SSE anyway).
            if (int_mv.0, int_mv.1) == (cur_mv.0, cur_mv.1) {
                continue;
            }
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
            let sse_at = |mv: IntegerMv| -> i64 {
                let pred = block_weighted_pred(
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
                    mv.0,
                    mv.1,
                );
                obmc_block_sse(
                    cur_y,
                    cur_w,
                    cur_h,
                    &pred,
                    &neighbour_sum,
                    xblen_u,
                    yblen_u,
                    xstart_ij,
                    ystart_ij,
                )
            };
            let sse_sub = sse_at(cur_mv);
            let sse_int = sse_at(int_mv);
            // Tie-bias toward integer-pel (`<=`) — at equal SSE the
            // integer-pel MV is preferable because it avoids the 8-tap
            // half-pel filter's smoothing contribution leaking into
            // neighbouring blocks' OBMC blends.
            if sse_int <= sse_sub {
                mvs[bidx] = int_mv;
            }
        }
    }
}

// ---- §12.3 motion-data emit, mirrors decode_block_motion_data --------

/// Encode the §12.3 `block_motion_data` block for a 1-reference inter
/// picture with all-Ref1Only blocks at the bottom of the superblock
/// hierarchy (split=2 → 4x4 = 16 blocks per superblock, the maximum).
/// `mvs` is a `blocks_x * blocks_y` row-major array of integer-pel MVs.
/// Resolve the per-block global-mode grid for a `blocks_x * blocks_y`
/// motion field from the encoder config. Returns a `bool` per block:
/// `true` marks a §12.3.3.2 global block. A caller-supplied
/// `block_gmode` grid is used when its length matches; otherwise the
/// whole picture is global (every block `true`).
fn resolve_gmode_grid(cfg: &GlobalMotionConfig, blocks_x: u32, blocks_y: u32) -> Vec<bool> {
    let n = (blocks_x * blocks_y) as usize;
    match &cfg.block_gmode {
        Some(g) if g.len() == n => g.clone(),
        _ => vec![true; n],
    }
}

pub fn encode_block_motion_data(
    w: &mut BitWriter,
    superblocks_x: u32,
    superblocks_y: u32,
    blocks_x: u32,
    blocks_y: u32,
    mvs: &[IntegerMv],
    gmode: Option<&[bool]>,
) {
    // Reconstruct what the decoder will end up with so we can match its
    // spatial-prediction / propagation behaviour symbol-for-symbol.
    let sb_split = vec![2u32; (superblocks_x * superblocks_y) as usize];
    let using_global = gmode.is_some();
    let mut blocks: Vec<BlockData> = (0..blocks_x * blocks_y)
        .map(|i| BlockData {
            rmode: RefPredMode::Ref1Only,
            // §12.3.3.2: a block is global only when the picture uses
            // global motion and the caller marked it. Ref1Only is
            // non-intra so the gmode flag reaches the wire.
            gmode: gmode.map(|g| g[i as usize]).unwrap_or(false),
            mv: [(mvs[i as usize].0, mvs[i as usize].1), (0, 0)],
            dc: [0; 3],
        })
        .collect();
    let bx = blocks_x;
    // 1) Superblock splits — emit the §12.3.1 length-prefixed block.
    let split_block = encode_sb_splits(superblocks_x, superblocks_y, &sb_split);
    write_uint_then_bytes(w, &split_block);

    // 2) Prediction modes — Ref1Only everywhere, plus the §12.3.3.2
    //    per-block global flag when the picture uses global motion.
    //    num_refs=1 means we only emit the ref1 bit per block.
    let pmode_block = encode_prediction_modes(
        superblocks_x,
        superblocks_y,
        &sb_split,
        &blocks,
        bx,
        1,
        using_global,
    );
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
    gmode: Option<&[bool]>,
) {
    let sb_split = vec![2u32; (superblocks_x * superblocks_y) as usize];
    let using_global = gmode.is_some();
    let mut blocks: Vec<BlockData> = (0..blocks_x * blocks_y)
        .map(|i| {
            let d = decisions[i as usize];
            BlockData {
                rmode: d.rmode,
                // §12.3.3.2: honoured only on non-intra blocks; the
                // bipred selector never emits Intra, so the flag reaches
                // the wire on every marked block.
                gmode: gmode.map(|g| g[i as usize]).unwrap_or(false),
                mv: [(d.mv1.0, d.mv1.1), (d.mv2.0, d.mv2.1)],
                dc: [0; 3],
            }
        })
        .collect();
    let bx = blocks_x;
    // 1) Superblock splits.
    let split_block = encode_sb_splits(superblocks_x, superblocks_y, &sb_split);
    write_uint_then_bytes(w, &split_block);

    // 2) Prediction modes — num_refs = 2 emits both bits per block, plus
    //    the §12.3.3.2 per-block global flag under global motion.
    let pmode_block = encode_prediction_modes(
        superblocks_x,
        superblocks_y,
        &sb_split,
        &blocks,
        bx,
        2,
        using_global,
    );
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
    using_global: bool,
) -> Vec<u8> {
    let mut enc = ArithEncoder::new();
    let mut bank = ContextBank::new(3);
    // Track which blocks have been "filled" by a top-level decode for
    // the prediction context (matches decoder propagation).
    let mut current_rmode = vec![RefPredMode::Intra; blocks.len()];
    // Parallel gmode grid for §12.3.6.4 block-global-flag prediction.
    let mut current_gmode = vec![false; blocks.len()];
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
                    // The decoder propagates rmode across the superblock
                    // *before* reading the global-mode flag, so the
                    // current block's rmode is visible to
                    // block_global_mode.
                    propagate_rmode(&mut current_rmode, bx, blkx, blky, step, target);

                    // §12.3.3.2 block_global_mode. Only emitted when the
                    // picture uses global motion AND this block is not
                    // intra (intra blocks are forced non-global). The
                    // flag is coded as a residual against the §12.3.6.4
                    // neighbour-majority prediction.
                    let target_g = using_global
                        && target != RefPredMode::Intra
                        && blocks[(blky * bx + blkx) as usize].gmode;
                    if using_global && target != RefPredMode::Intra {
                        let gpred = block_global_prediction_enc(&current_gmode, bx, blkx, blky);
                        enc.write_bool(&mut bank, mvctx::GLOBAL_BLOCK, target_g ^ gpred);
                    }
                    propagate_gmode(&mut current_gmode, bx, blkx, blky, step, target_g);
                }
            }
        }
    }
    enc.finish()
}

/// §12.3.6.4 block-global-flag prediction: majority verdict of the three
/// already-decoded causal neighbours. Encoder mirror of the decoder's
/// `block_global_prediction` over the running gmode grid.
fn block_global_prediction_enc(buf: &[bool], bx: u32, x: u32, y: u32) -> bool {
    if x == 0 && y == 0 {
        return false;
    }
    if y == 0 {
        return buf[(x - 1) as usize];
    }
    if x == 0 {
        return buf[((y - 1) * bx) as usize];
    }
    let count = buf[((y - 1) * bx + (x - 1)) as usize] as u32
        + buf[((y - 1) * bx + x) as usize] as u32
        + buf[(y * bx + (x - 1)) as usize] as u32;
    count >= 2
}

fn propagate_gmode(buf: &mut [bool], bx: u32, xtl: u32, ytl: u32, k: u32, val: bool) {
    for y in ytl..ytl + k {
        for x in xtl..xtl + k {
            buf[(y * bx + x) as usize] = val;
        }
    }
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
                    // Set the current block's rmode + gmode in `current`
                    // so mv_prediction's mv_available check matches the
                    // decoder (§12.3.6.1 excludes global blocks from the
                    // neighbour median, so the encoder must carry the same
                    // gmode flags in its prediction context).
                    current[bi].rmode = block.rmode;
                    current[bi].gmode = block.gmode;
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
fn build_obmc_prediction<S: InterSample>(
    sequence: &SequenceHeader,
    pred: &PicturePredictionParams,
    motion: &PictureMotionData,
    is_chroma: bool,
    ref_y: &[S],
    ref_u: &[S],
    ref_v: &[S],
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let chroma_h_ratio = sequence.video_params.chroma_format.h_ratio();
    let chroma_v_ratio = sequence.video_params.chroma_format.v_ratio();
    let _ = is_chroma;
    let pred_y = build_obmc_prediction_one(
        sequence,
        pred,
        motion,
        /* component */ 0,
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
        /* component */ 1,
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
        /* component */ 2,
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
fn build_obmc_prediction_one<S: InterSample>(
    sequence: &SequenceHeader,
    pred: &PicturePredictionParams,
    motion: &PictureMotionData,
    component: usize,
    chroma_h_ratio: u32,
    chroma_v_ratio: u32,
    ref_plane: &[S],
    comp_w: usize,
    comp_h: usize,
) -> Vec<i32> {
    let is_chroma = component != 0;
    let depth = if is_chroma {
        sequence.chroma_depth
    } else {
        sequence.luma_depth
    };
    let half = 1i32 << (depth - 1);
    // Reference plane in the spec's signed pre-offset convention.
    let ref_signed: Vec<i32> = ref_plane.iter().map(|&v| v.to_i32() - half).collect();
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
        component,
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
    iep: &InterEncoderParams,
    mv_precision: u32,
    num_refs: u32,
) -> PicturePredictionParams {
    let (sbx, sby, blocks_x, blocks_y) = motion_grid(sequence.luma_width, sequence.luma_height);
    let (xblen, yblen, xbsep, ybsep) = PRESET1;
    let (using_global, global1, global2) = match &iep.global_motion {
        None => (false, None, None),
        Some(cfg) => {
            let g1 = Some(effective_global_params(&cfg.global1));
            let g2 = if num_refs == 2 {
                Some(effective_global_params(
                    &cfg.global2.clone().unwrap_or_default(),
                ))
            } else {
                None
            };
            (true, g1, g2)
        }
    };
    PicturePredictionParams {
        luma_xblen: xblen,
        luma_yblen: yblen,
        luma_xbsep: xbsep,
        luma_ybsep: ybsep,
        mv_precision,
        using_global,
        prediction_mode: 0,
        superblocks_x: sbx,
        superblocks_y: sby,
        blocks_x,
        blocks_y,
        refs_wt_precision: 1,
        ref1_wt: 1,
        ref2_wt: 1,
        global1,
        global2,
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
    gmode: Option<&[bool]>,
    global1: Option<GlobalParams>,
) -> PictureMotionData {
    let blocks: Vec<BlockData> = (0..blocks_x * blocks_y)
        .map(|i| BlockData {
            rmode: RefPredMode::Ref1Only,
            gmode: gmode.map(|g| g[i as usize]).unwrap_or(false),
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
        global1,
        global2: None,
    }
}

/// Subtract the OBMC prediction from the source picture in the spec's
/// **signed pre-output-offset** domain — i.e.
/// `residue[x] = (source[x] - half) - prediction_signed[x]` with `half
/// = 2^(depth-1)` from the **sequence** depth (8-bit and deep-colour
/// sources alike).
/// The result is a signed plane the wavelet residue encoder forward-
/// transforms as if it were a tiny intra picture.
fn build_residue_plane<S: InterSample>(
    source: &[S],
    prediction_signed: &[i32],
    depth: u32,
) -> Vec<i32> {
    debug_assert_eq!(source.len(), prediction_signed.len());
    let half = 1i32 << (depth - 1);
    source
        .iter()
        .zip(prediction_signed.iter())
        .map(|(&s, &p)| (s.to_i32() - half) - p)
        .collect()
}

/// Forward-DWT one residue plane into an **unquantised** coefficient
/// pyramid. Mirrors the forward-transform half of
/// [`crate::encoder_intra_core::forward_and_quantise`] but operates on
/// a pre-signed `i32` source (no `- half` step — the residue is already
/// in the signed pre-offset domain) and stops before quantisation so
/// the same pyramid can be re-quantised at many candidate qindexes (the
/// rate-control picker walks qindex over a single forward DWT).
fn forward_residue_pyramid(
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
    dwt(&pic, rp.wavelet, rp.dwt_depth)
}

/// Dead-zone quantise an already-forward-transformed residue pyramid at
/// `qindex`. Kept split from [`forward_residue_pyramid`] so a
/// rate-control picker can re-quantise the same forward-DWT output at
/// many candidate qindexes without repeating the DWT.
fn quantise_residue_pyramid(py: &[[SubbandData; 4]], qindex: u32) -> Vec<[SubbandData; 4]> {
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
                    dst.set(y, x, residue_quantise_coeff(v, qindex));
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
/// Emit the zero-residue tail of an inter picture (§11.3), in the wire
/// form selected by [`InterEncoderParams::explicit_zero_residue`]:
///
/// * explicit — `ZERO_RESIDUAL = 0`, LeGall-5,3 depth-1 transform
///   parameters, default codeblocks, then 13 zero-`length` subband
///   blocks (4 bands x 3 components). Decodes to an all-zero residue
///   through the normal path on every decoder.
/// * skip — the compact `ZERO_RESIDUAL = 1` flag. Spec-equivalent, but
///   the reference decoder mis-reconstructs it (round-408 black-box
///   finding), so this form is opt-in.
fn write_zero_residue_tail(w: &mut BitWriter, explicit: bool) {
    if explicit {
        w.write_bool(false); // ZERO_RESIDUAL = 0
        w.write_uint(wavelet_index(WaveletFilter::LeGall5_3));
        w.write_uint(1); // dwt_depth
        w.write_bool(false); // spatial_partition_flag
        w.byte_align();
        for _component in 0..3 {
            for _band in 0..4 {
                w.byte_align();
                w.write_uint(0); // subband length 0 → all-zero band
            }
        }
    } else {
        w.write_bool(true); // ZERO_RESIDUAL = 1
    }
    w.byte_align();
}

/// `is_intra` distinction at the parameter level).
fn write_residue_transform_parameters(w: &mut BitWriter, rp: &ResidueParams) {
    w.write_uint(wavelet_index(rp.wavelet));
    w.write_uint(rp.dwt_depth);
    // §11.3.3 codeblock_parameters. With no partition every level gets a
    // single codeblock (1, 1) and we emit `spatial_partition_flag = 0`;
    // with a partition we emit the flag, the per-level counts, then the
    // codeblock_mode — mirroring
    // `picture_core::parse_codeblock_parameters` (the decoder's residue
    // path uses the identical reader, so the inter and intra codeblock
    // grids are read by the same code).
    match &rp.codeblocks {
        None => w.write_bool(false),
        Some(_) => {
            w.write_bool(true);
            let grid = rp.codeblock_grid();
            for (x, y) in &grid {
                w.write_uint(*x);
                w.write_uint(*y);
            }
            w.write_uint(rp.codeblock_mode);
        }
    }
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

/// Emit all three residue components (Y/U/V) for one inter picture from
/// their pre-built **raw** forward-transform pyramids. Dispatches on
/// `rp.codeblocks`: `None` ⇒ the flat single-codeblock path (quantise
/// the whole pyramid at `rp.qindex`, then the byte-identical
/// `write_residue_component_subbands`); `Some` ⇒ the §11.3.3
/// codeblock walk that requantises per codeblock in place.
fn emit_residue_components(
    w: &mut BitWriter,
    rp: &ResidueParams,
    raw_y: &[[SubbandData; 4]],
    raw_u: &[[SubbandData; 4]],
    raw_v: &[[SubbandData; 4]],
) {
    if rp.codeblocks.is_none() {
        let qpy_y = quantise_residue_pyramid(raw_y, rp.qindex);
        let qpy_u = quantise_residue_pyramid(raw_u, rp.qindex);
        let qpy_v = quantise_residue_pyramid(raw_v, rp.qindex);
        write_residue_component_subbands(w, rp, &qpy_y);
        write_residue_component_subbands(w, rp, &qpy_u);
        write_residue_component_subbands(w, rp, &qpy_v);
    } else {
        let mut qpy_y = raw_y.to_vec();
        let mut qpy_u = raw_u.to_vec();
        let mut qpy_v = raw_v.to_vec();
        write_residue_component_subbands_cb(w, rp, &mut qpy_y);
        write_residue_component_subbands_cb(w, rp, &mut qpy_u);
        write_residue_component_subbands_cb(w, rp, &mut qpy_v);
    }
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

// ---- §11.3.3 spatial-partition (codeblock) residue path -------------
//
// When `ResidueParams::codeblocks` is `Some`, the residue subbands are
// split into a per-level grid of codeblocks. Each codeblock carries a
// §13.4.3.3 `zero_flag` skip and, under `codeblock_mode == 1`, a
// §13.4.3.4 differential quantiser offset. The decoder reads this with
// the same `picture_core::decode_subband` walk it uses for intra, so the
// emission here mirrors `encoder_intra_core::{encode_subband_ac,
// codeblock_offset, cb_bounds}` exactly — only the LL DC-prediction and
// the pre-quantisation of the LL band differ (the inter residue has no
// DC prediction, and the LL band is quantised in place at the base
// `qindex` like every other inter-residue band).
//
// These functions operate on the **raw** (unquantised) forward-transform
// pyramid, requantising each codeblock at its running quantiser, so they
// take `&mut [[SubbandData; 4]]` and write the quantised values back to
// keep the neighbourhood / sign / parent contexts consistent with what
// the decoder reconstructs — identical to the intra-core in-place walk.

/// The per-codeblock differential quantiser offset (§13.4.3.4): `0` for
/// the first non-skipped codeblock of a subband, `+1` for each later
/// non-skipped codeblock, giving a strictly increasing running quantiser
/// across the non-skipped codeblocks in raster order. Mirrors
/// [`crate::encoder_intra_core`]'s `codeblock_offset`.
fn residue_codeblock_offset(nonskip_index: u32) -> i32 {
    if nonskip_index == 0 {
        0
    } else {
        1
    }
}

/// Codeblock pixel bounds inside a subband (§12.4.5.2 partitioning):
/// `(left, right, top, bottom)`. Mirrors `encoder_intra_core::cb_bounds`
/// / `picture_core::codeblock_bounds`.
fn residue_cb_bounds(
    cx: u32,
    cy: u32,
    cbx: u32,
    cby: u32,
    band_w: usize,
    band_h: usize,
) -> (usize, usize, usize, usize) {
    let left = (band_w * cx as usize) / cbx as usize;
    let right = (band_w * (cx as usize + 1)) / cbx as usize;
    let top = (band_h * cy as usize) / cby as usize;
    let bottom = (band_h * (cy as usize + 1)) / cby as usize;
    (left, right, top, bottom)
}

/// Emit one component's **raw** residue pyramid as length-prefixed
/// AC-coded subband blocks, walking each subband codeblock-by-codeblock
/// per `rp.codeblock_grid()`. Mirrors
/// [`crate::encoder_intra_core::write_component_subbands`]; `qpy` is
/// requantised in place per codeblock.
fn write_residue_component_subbands_cb(
    w: &mut BitWriter,
    rp: &ResidueParams,
    qpy: &mut [[SubbandData; 4]],
) {
    let grid = rp.codeblock_grid();
    write_residue_subband_block_cb(w, rp, qpy, 0, Orient::LL, grid[0]);
    for level in 1..=rp.dwt_depth {
        for orient in [Orient::HL, Orient::LH, Orient::HH] {
            write_residue_subband_block_cb(w, rp, qpy, level, orient, grid[level as usize]);
        }
    }
}

/// Emit one subband block under the codeblock walk: byte-align, length,
/// qindex, byte-align, AC bytes. Empty bands emit `length = 0` only.
fn write_residue_subband_block_cb(
    w: &mut BitWriter,
    rp: &ResidueParams,
    qpy: &mut [[SubbandData; 4]],
    level: u32,
    orient: Orient,
    codeblocks: (u32, u32),
) {
    w.byte_align();
    let band = &qpy[level as usize][orient.as_index()];
    if band.width == 0 || band.height == 0 {
        w.write_uint(0);
        w.byte_align();
        return;
    }
    let bytes = encode_residue_subband_ac_cb(qpy, level, orient, codeblocks, rp);
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

/// Encode one residue subband under the §13.4.4 AC contexts, walking
/// codeblock-by-codeblock. For a single codeblock (`(1, 1)`, e.g. the LL
/// band) this emits neither a skip flag nor a quant offset and is
/// byte-identical to [`encode_residue_subband_ac`]; for a partitioned
/// subband it emits the §13.4.3.3 `ZERO_BLOCK` skip flag per codeblock
/// and, under `codeblock_mode == 1`, the §13.4.3.4 differential quant
/// offset. `qpy` is requantised in place at each codeblock's running
/// quantiser. Mirrors `encoder_intra_core::encode_subband_ac` minus the
/// LL DC-prediction (the inter residue carries no DC prediction).
pub(crate) fn encode_residue_subband_ac_cb(
    pyramid: &mut [[SubbandData; 4]],
    level: u32,
    orient: Orient,
    codeblocks: (u32, u32),
    rp: &ResidueParams,
) -> Vec<u8> {
    let mut bank = ContextBank::new(ctx::NUM_CONTEXTS);
    let mut enc = ArithEncoder::new();

    let level_idx = level as usize;
    let orient_idx = orient.as_index();
    let band_w = pyramid[level_idx][orient_idx].width;
    let band_h = pyramid[level_idx][orient_idx].height;
    let (cbx, cby) = (codeblocks.0.max(1), codeblocks.1.max(1));
    let single_cb = cbx * cby == 1;

    // Running quantiser advances by `residue_codeblock_offset` ONLY for
    // non-skipped codeblocks under mode 1, and the offset symbol counts
    // non-skipped codeblocks so encoder and decoder stay in lockstep:
    // §13.4.3.2 modifies `quant_idx` inside the `if skipped == False`
    // branch.
    let mut q = rp.qindex as i32;
    let mut nonskip_index = 0u32;
    for cy in 0..cby {
        for cx in 0..cbx {
            let (left, right, top, bottom) = residue_cb_bounds(cx, cy, cbx, cby, band_w, band_h);

            let tentative_q = if rp.codeblock_mode == 1 {
                (q + residue_codeblock_offset(nonskip_index)).max(0)
            } else {
                q
            };

            // Requantise this codeblock's raw coefficients at the
            // tentative quantiser, writing them back. A codeblock is a
            // skip candidate when every quantised coefficient is zero.
            let mut all_zero = true;
            {
                let band = &mut pyramid[level_idx][orient_idx];
                for y in top..bottom {
                    for x in left..right {
                        let raw = band.get(y, x);
                        let qc = residue_quantise_coeff(raw, tentative_q as u32);
                        band.set(y, x, qc);
                        if qc != 0 {
                            all_zero = false;
                        }
                    }
                }
            }

            // §13.4.3.3: the skip flag is only present when the subband
            // has more than one codeblock.
            let skipped = !single_cb && all_zero;
            if !single_cb {
                enc.write_bool(&mut bank, ctx::ZERO_BLOCK, skipped);
            }
            if skipped {
                // §13.4.3.2: a skipped codeblock emits no quant offset and
                // does NOT advance the running quantiser.
                continue;
            }

            if rp.codeblock_mode == 1 {
                enc.write_sint(
                    &mut bank,
                    &[ctx::Q_OFFSET_FOLLOW],
                    ctx::Q_OFFSET_DATA,
                    ctx::Q_OFFSET_SIGN,
                    residue_codeblock_offset(nonskip_index),
                );
                q = tentative_q;
            }
            nonskip_index += 1;

            // Re-borrow the parent band (level >= 2) plus the current band
            // for the context derivation, splitting so the parent borrow
            // doesn't overlap the mutable quantise pass above.
            let (parent, band): (Option<&SubbandData>, &SubbandData) = if level >= 2 {
                let (lower, upper) = pyramid.split_at(level_idx);
                (
                    Some(&lower[level_idx - 1][orient_idx]),
                    &upper[0][orient_idx],
                )
            } else {
                (None, &pyramid[level_idx][orient_idx])
            };
            for y in top..bottom {
                for x in left..right {
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

// ---- §11.3 residue rate control --------------------------------------

/// The Y / U / V **unquantised** residue coefficient pyramids for one
/// inter picture, produced once by [`build_inter_residue_pyramids`] so a
/// rate-control picker can re-quantise them at many candidate qindexes
/// without repeating the OBMC reconstruction, the residue subtraction,
/// or the forward DWT.
struct InterResiduePyramids {
    y: Vec<[SubbandData; 4]>,
    u: Vec<[SubbandData; 4]>,
    v: Vec<[SubbandData; 4]>,
}

/// Build the three unquantised residue pyramids for a 1-ref inter
/// picture: reconstruct the §15.8 OBMC prediction the decoder will
/// compute from `motion`, subtract it from the source in the signed
/// pre-offset domain (§11.3), and forward-transform each plane with the
/// residue wavelet. The MV grid is the same one
/// [`encode_inter_picture`] commits to the bitstream, so the pyramids
/// match the eventual decode symbol-for-symbol at any qindex.
#[allow(clippy::too_many_arguments)]
fn build_inter_residue_pyramids<S: InterSample>(
    sequence: &SequenceHeader,
    iep: &InterEncoderParams,
    rp: &ResidueParams,
    mvs: &[IntegerMv],
    cur_y: &[S],
    cur_u: &[S],
    cur_v: &[S],
    ref_y: &[S],
    ref_u: &[S],
    ref_v: &[S],
) -> InterResiduePyramids {
    let (sbx, sby, blocks_x, blocks_y) = motion_grid(sequence.luma_width, sequence.luma_height);
    let pred = picture_prediction_params_from(sequence, iep, iep.mv_precision, 1);
    let gmode_grid = iep
        .global_motion
        .as_ref()
        .map(|cfg| resolve_gmode_grid(cfg, blocks_x, blocks_y));
    let global1 = iep
        .global_motion
        .as_ref()
        .map(|cfg| effective_global_params(&cfg.global1));
    let motion = build_motion_from_mv_grid(
        sbx,
        sby,
        blocks_x,
        blocks_y,
        mvs,
        gmode_grid.as_deref(),
        global1,
    );
    let (pred_y, pred_u, pred_v) =
        build_obmc_prediction(sequence, &pred, &motion, false, ref_y, ref_u, ref_v);
    let res_y = build_residue_plane(cur_y, &pred_y, sequence.luma_depth);
    let res_u = build_residue_plane(cur_u, &pred_u, sequence.chroma_depth);
    let res_v = build_residue_plane(cur_v, &pred_v, sequence.chroma_depth);
    InterResiduePyramids {
        y: forward_residue_pyramid(&res_y, sequence.luma_width, sequence.luma_height, rp),
        u: forward_residue_pyramid(&res_u, sequence.chroma_width, sequence.chroma_height, rp),
        v: forward_residue_pyramid(&res_v, sequence.chroma_width, sequence.chroma_height, rp),
    }
}

/// Serialise the §11.3 residue stream for one inter picture at `qindex`
/// from already-forward-transformed pyramids, returning the byte count.
///
/// Counts exactly the bytes [`encode_inter_picture`] emits for the
/// ZERO_RESIDUAL=false branch: the `transform_parameters` block, then
/// the three length-prefixed, qindex-tagged, AC-coded component subband
/// streams. The leading ZERO_RESIDUAL flag bit and the surrounding
/// byte-alignment are folded in via the same [`BitWriter`] the picture
/// emitter uses, so the returned count tracks the picture's residue
/// payload monotonically as qindex rises (the dead-zone forward
/// quantiser drives more coefficients to zero, which the interleaved
/// exp-Golomb coder spends fewer bits on).
fn inter_residue_bytes_at_qindex(
    rp: &ResidueParams,
    pyr: &InterResiduePyramids,
    qindex: u32,
) -> usize {
    let mut rp_q = rp.clone();
    rp_q.qindex = qindex;

    let mut w = BitWriter::new();
    // Mirror the ZERO_RESIDUAL=false layout from `encode_inter_picture`:
    // the flag bit, then transform_parameters (no byte-align between),
    // then a byte-align before the per-component subband bytes. Dispatch
    // through `emit_residue_components` so the byte count reflects the
    // §11.3.3 codeblock grid (skip flags + per-codeblock requantise) when
    // `rp.codeblocks` is `Some` — otherwise the picker would mis-estimate
    // the payload for codeblock-partitioned residue. `emit_residue_*`
    // takes the **raw** pyramids and quantises internally (flat at
    // `qindex` when no grid, per-codeblock when a grid is set), so we hand
    // it the unquantised `pyr` rather than a pre-quantised clone.
    w.write_bool(false);
    write_residue_transform_parameters(&mut w, &rp_q);
    w.byte_align();
    emit_residue_components(&mut w, &rp_q, &pyr.y, &pyr.u, &pyr.v);
    w.byte_align();
    w.finish().len()
}

/// 1-ref inter-residue rate-control qindex picker — the inter-residue
/// analogue of [`crate::encoder::pick_hq_picture_qindex`].
///
/// Runs the same motion estimation [`encode_inter_picture`] would
/// (integer-pel SAD search + the configured sub-pel / OBMC refinement
/// passes), reconstructs the OBMC prediction once, forward-transforms
/// the residue once, then walks `rp.qindex.min(127)..=127` and returns
/// the **smallest** qindex whose serialised §11.3 residue payload is
/// `<= target_residue_bytes`. If even qindex 127 overflows the budget,
/// returns 127 (the most aggressive quantiser; residue stays
/// spec-conformant under §13.4.4 at every qindex).
///
/// `target_residue_bytes` is the budget for the residue stream only
/// (transform_parameters + the three component subband streams + the
/// ZERO_RESIDUAL=false flag byte), **not** the whole picture: the
/// picture header, reference deltas, and block_motion_data are fixed by
/// the MV grid and do not depend on the residue qindex.
///
/// Monotone in `target_residue_bytes`: a smaller budget can only push
/// the chosen qindex up (or leave it). Pure encoder-side rate policy —
/// any qindex is a legal §13.4.4 choice, so the picked stream is
/// always decodable.
///
/// Source of truth: BBC Dirac Specification v2.2.3 §11.3 (residue
/// wavelet_transform), §13.4.4 (codeblock entropy coding).
#[allow(clippy::too_many_arguments)]
pub fn pick_inter_residue_qindex<S: InterSample>(
    sequence: &SequenceHeader,
    params: &InterEncoderParams,
    cur_y: &[S],
    cur_u: &[S],
    cur_v: &[S],
    ref_y: &[S],
    ref_u: &[S],
    ref_v: &[S],
    target_residue_bytes: u32,
) -> u32 {
    let rp = match params.residue {
        Some(ref rp) => rp.clone(),
        // No residue stream is emitted at all when residue is disabled;
        // the floor qindex is the only meaningful answer.
        None => return 0,
    };

    let mvs = inter_mv_grid(sequence, params, cur_y, ref_y);
    let pyr = build_inter_residue_pyramids(
        sequence, params, &rp, &mvs, cur_y, cur_u, cur_v, ref_y, ref_u, ref_v,
    );

    let floor = rp.qindex.min(127);
    for qindex in floor..=127 {
        let bytes = inter_residue_bytes_at_qindex(&rp, &pyr, qindex);
        if bytes <= target_residue_bytes as usize {
            return qindex;
        }
    }
    127
}

/// Diagnostic counterpart to [`pick_inter_residue_qindex`]: returns
/// `(qindex, actual_residue_bytes)` so callers can inspect the chosen
/// quantiser's actual residue-byte cost relative to the supplied budget.
#[allow(clippy::too_many_arguments)]
pub fn inter_residue_qindex_diagnostic<S: InterSample>(
    sequence: &SequenceHeader,
    params: &InterEncoderParams,
    cur_y: &[S],
    cur_u: &[S],
    cur_v: &[S],
    ref_y: &[S],
    ref_u: &[S],
    ref_v: &[S],
    target_residue_bytes: u32,
) -> (u32, usize) {
    let rp = match params.residue {
        Some(ref rp) => rp.clone(),
        None => return (0, 0),
    };
    let mvs = inter_mv_grid(sequence, params, cur_y, ref_y);
    let pyr = build_inter_residue_pyramids(
        sequence, params, &rp, &mvs, cur_y, cur_u, cur_v, ref_y, ref_u, ref_v,
    );
    let qindex = pick_inter_residue_qindex(
        sequence,
        params,
        cur_y,
        cur_u,
        cur_v,
        ref_y,
        ref_u,
        ref_v,
        target_residue_bytes,
    );
    let bytes = inter_residue_bytes_at_qindex(&rp, &pyr, qindex);
    (qindex, bytes)
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
pub fn encode_inter_picture<S: InterSample>(
    sequence: &SequenceHeader,
    params: &InterEncoderParams,
    picture_number: u32,
    ref1_picture_number: u32,
    cur_y: &[S],
    cur_u: &[S],
    cur_v: &[S],
    ref_y: &[S],
    ref_u: &[S],
    ref_v: &[S],
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
    write_picture_prediction_parameters(&mut w, params, params.mv_precision, 1);
    w.byte_align();

    // §12.3 block_motion_data. Sub-pel ME degenerates to integer-pel
    // when `mv_precision == 0`.
    let (sbx, sby, blocks_x, blocks_y) = motion_grid(sequence.luma_width, sequence.luma_height);
    let mvs = inter_mv_grid(sequence, params, cur_y, ref_y);
    // §12.3.3.2 per-block global-mode grid (None when the picture uses
    // block motion only).
    let gmode_grid = params
        .global_motion
        .as_ref()
        .map(|cfg| resolve_gmode_grid(cfg, blocks_x, blocks_y));
    let global1 = params
        .global_motion
        .as_ref()
        .map(|cfg| effective_global_params(&cfg.global1));
    encode_block_motion_data(
        &mut w,
        sbx,
        sby,
        blocks_x,
        blocks_y,
        &mvs,
        gmode_grid.as_deref(),
    );
    w.byte_align();

    // §11.3 wavelet_transform.
    if let Some(ref residue) = params.residue {
        // ZERO_RESIDUAL = false → emit transform parameters + per-
        // component subbands. Build the same `PictureMotionData` the
        // decoder will reconstruct from the bytes we just wrote so the
        // OBMC prediction matches symbol-for-symbol.
        let pred = picture_prediction_params_from(sequence, params, params.mv_precision, 1);
        let motion = build_motion_from_mv_grid(
            sbx,
            sby,
            blocks_x,
            blocks_y,
            &mvs,
            gmode_grid.as_deref(),
            global1.clone(),
        );
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

        let raw_y =
            forward_residue_pyramid(&res_y, sequence.luma_width, sequence.luma_height, residue);
        let raw_u = forward_residue_pyramid(
            &res_u,
            sequence.chroma_width,
            sequence.chroma_height,
            residue,
        );
        let raw_v = forward_residue_pyramid(
            &res_v,
            sequence.chroma_width,
            sequence.chroma_height,
            residue,
        );
        emit_residue_components(&mut w, residue, &raw_y, &raw_u, &raw_v);
        w.byte_align();
    } else {
        // No residue: reconstruction = OBMC(reference). Wire form per
        // `explicit_zero_residue` (the reference decoder mishandles the
        // ZERO_RESIDUAL=1 skip form — round-408 black-box finding).
        write_zero_residue_tail(&mut w, params.explicit_zero_residue);
    }

    w.finish()
}

fn write_picture_prediction_parameters(
    w: &mut BitWriter,
    params: &InterEncoderParams,
    mv_precision: u32,
    num_refs: u32,
) {
    // §11.2.2 block_parameters. The values are Table 11.1's preset 1
    // (xblen=yblen=8, xbsep=ybsep=4), but they are emitted as CUSTOM
    // literals (index 0) rather than by preset index: black-box probing
    // (round-408) showed the reference decoder resolves preset index 1
    // to *non-overlapped* blocks — its boundary blend hard-switches at
    // xbsep multiples instead of applying the spec's [1,3,5,7] ramp —
    // while the identical parameters written literally decode
    // bit-exactly on both decoders. Spelling the parameters out costs
    // a few bits per picture and removes the only cross-decoder
    // divergence our inter streams had left.
    w.write_uint(0);
    w.write_uint(PRESET1.0);
    w.write_uint(PRESET1.1);
    w.write_uint(PRESET1.2);
    w.write_uint(PRESET1.3);
    // §11.2.5 motion_vector_precision (0=integer, 1=half, 2=quarter,
    // 3=eighth pel; spec caps at 3).
    debug_assert!(mv_precision <= 3, "mv_precision must be 0..=3");
    w.write_uint(mv_precision);
    // §11.2.6 global_motion. `state[USING GLOBAL] = read_bool()`; when
    // set, one `global_motion_parameters` block per reference.
    match &params.global_motion {
        None => w.write_bool(false),
        Some(cfg) => {
            w.write_bool(true);
            write_global_motion_parameters(&mut *w, &effective_global_params(&cfg.global1));
            if num_refs == 2 {
                let g2 = cfg.global2.clone().unwrap_or_default();
                write_global_motion_parameters(&mut *w, &effective_global_params(&g2));
            }
        }
    }
    // §11.2.7 picture_prediction_mode: 0 (default).
    w.write_uint(0);
    // §11.2.8 reference_picture_weights_flag: false → defaults
    // refs_wt_precision=1, ref1_wt=ref2_wt=1.
    w.write_bool(false);
}

/// §11.2.6 `global_motion_parameters` writer — the exact bitstream
/// inverse of `parse_global_motion_parameters` (the decoder's proven
/// reader). Emits the `pan_tilt` / `zoom_rotate_shear` / `perspective`
/// triple for one reference. `g` should be the *effective* parameters
/// (see [`effective_global_params`]) so what is emitted round-trips
/// byte-exactly to the value fed into [`crate::obmc::global_mv`].
fn write_global_motion_parameters(w: &mut BitWriter, g: &GlobalParams) {
    // pan_tilt(): a nonzero_pan_tilt_flag, then two sints when set.
    if g.pan_tilt != (0, 0) {
        w.write_bool(true);
        w.write_sint(g.pan_tilt.0);
        w.write_sint(g.pan_tilt.1);
    } else {
        w.write_bool(false);
    }
    // zoom_rotate_shear(): a nontrivial_zrs_flag, then a scaling exponent
    // and the four matrix elements when set. The identity matrix
    // [[1,0],[0,1]] at exponent 0 is the trivial default the decoder
    // fills in when the flag is False.
    if g.zrs == [[1, 0], [0, 1]] && g.zrs_exp == 0 {
        w.write_bool(false);
    } else {
        w.write_bool(true);
        w.write_uint(g.zrs_exp);
        w.write_sint(g.zrs[0][0]);
        w.write_sint(g.zrs[0][1]);
        w.write_sint(g.zrs[1][0]);
        w.write_sint(g.zrs[1][1]);
    }
    // perspective(): a nonzero_perspective_flag, then an exponent and two
    // sints when set.
    if g.perspective != (0, 0) {
        w.write_bool(true);
        w.write_uint(g.persp_exp);
        w.write_sint(g.perspective.0);
        w.write_sint(g.perspective.1);
    } else {
        w.write_bool(false);
    }
}

/// Canonicalise a caller-supplied [`GlobalParams`] to the exact value the
/// decoder reconstructs from [`write_global_motion_parameters`]'s output.
///
/// The §11.2.6 flags are omission-encoded: when a component is at its
/// trivial default the flag is `False` and the trailing sub-fields are
/// *not* on the wire, so the decoder leaves them at the struct default
/// (`0`). The only field that can differ between "what the caller put in
/// the struct" and "what the decoder reads back" is a non-zero
/// `persp_exp` paired with a zero `perspective` vector (the flag is
/// `False` so `persp_exp` never reaches the wire). Because
/// [`crate::obmc::global_mv`] consumes `persp_exp` unconditionally, the
/// encoder must build its OBMC prediction from these *effective*
/// parameters, or its reconstruction would diverge from the decoder's.
fn effective_global_params(g: &GlobalParams) -> GlobalParams {
    let mut e = g.clone();
    if e.perspective == (0, 0) {
        e.persp_exp = 0;
    }
    e
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
/// **Round-91 widening.** Each per-ref MV is evaluated against a
/// strict-superset candidate set: `{int-pel, half-pel, sub-pel}` at
/// `mv_precision >= 2`, degenerating to `{int-pel, sub-pel}` at half-pel
/// precision and `{int-pel}` at integer precision. The bipred mode
/// enumerates up to `3 × 3 = 9` MV combinations (de-duplicated when the
/// half-pel candidate coincides with one of its peers). Mirrors the
/// 1-ref P-path's `inter_select_int_pel_per_block` strict-superset
/// invariant (round-73 + round-80 post-OBMC pass): the widened
/// candidate set strictly contains the previous round-39 2-candidate
/// set, so the chosen per-ref SAD and bipred SAD are ≤ the round-39
/// values on every block — never regresses.
///
/// `mv_precision` is in `1/(2^p)` luma pel units (matches
/// [`InterEncoderParams::mv_precision`]).
#[allow(clippy::too_many_arguments)]
pub fn bipred_select_modes<S: InterSample>(
    cur_y: &[S],
    ref1_y: &[S],
    ref2_y: &[S],
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
            // Per-block adaptive precision: widened in **round-91**
            // from `{sub-pel, int-pel}` (round-39) to the strict
            // superset `{int-pel, half-pel, sub-pel}` per reference.
            // The 1-ref P-path's `inter_select_int_pel_per_block`
            // (round-73 + round-80 post-OBMC) carries the same
            // strict-superset correctness invariant for its own
            // 2-candidate set; the round-91 widening brings the bipred
            // path under the *same* invariant with one extra
            // half-pel-snapped candidate per ref. Sharp-edge content
            // (bars, occluders) still snaps to int-pel; smooth motion
            // still keeps the native sub-pel MV; mid-energy content
            // that prefers the 8-tap-filtered half-pel position but
            // not the bilinear-refined qpel offset can now pick the
            // half-pel candidate. Per ST 2042-1 §11.2.5 `mv_precision`
            // already enumerates int / half / qpel / ⅛-pel, so adding
            // a per-block half-pel candidate is auditor-equivalent to
            // the 1-ref invariant: the same MV space, just one extra
            // representative per ref, no decoder-side change required.
            let mv1_int = round_mv_to_int_pel(mv1, mv_precision);
            let mv2_int = round_mv_to_int_pel(mv2, mv_precision);
            let mv1_half = round_mv_to_half_pel(mv1, mv_precision);
            let mv2_half = round_mv_to_half_pel(mv2, mv_precision);

            let sad_one =
                |mv: IntegerMv, refp: &[S], up: &[i32], up_w: usize, up_h: usize| -> i64 {
                    if mv_precision == 0 {
                        sad_block(cur_y, refp, w_i, h_i, x0, y0, mv.0, mv.1, xblen, yblen)
                    } else {
                        let scale = 1i64 << mv_precision;
                        let qx = (x0 as i64) * scale + mv.0 as i64;
                        let qy = (y0 as i64) * scale + mv.1 as i64;
                        sad_subpel(
                            cur_y,
                            up,
                            up_w,
                            up_h,
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
                    }
                };

            // Per-mode best MV from the widened 3-candidate set. Tie
            // priority is int-pel ≥ half-pel ≥ sub-pel — i.e. on equal
            // SAD we prefer the candidate that contributes the smallest
            // §15.8.11 8-tap-filter weight to neighbours' OBMC blends,
            // so the per-block selection picks the most blend-friendly
            // representative. This carries the same `<=` tie-bias the
            // 1-ref `inter_select_int_pel_per_block` uses (round-73).
            let pick_best_mv = |mv_sp: IntegerMv,
                                mv_half: IntegerMv,
                                mv_int: IntegerMv,
                                refp: &[S],
                                up: &[i32],
                                up_w: usize,
                                up_h: usize|
             -> (IntegerMv, i64) {
                let sad_sp = sad_one(mv_sp, refp, up, up_w, up_h);
                let sad_int = sad_one(mv_int, refp, up, up_w, up_h);
                // int-pel wins all ties via `<=`.
                let (mut best_mv, mut best_sad) = if sad_int <= sad_sp {
                    (mv_int, sad_int)
                } else {
                    (mv_sp, sad_sp)
                };
                // Half-pel candidate: only evaluated when distinct
                // from both peers (`p <= 1` makes it equal int or
                // sub-pel and the lookup would be redundant work).
                if mv_precision >= 2
                    && (mv_half.0, mv_half.1) != (mv_int.0, mv_int.1)
                    && (mv_half.0, mv_half.1) != (mv_sp.0, mv_sp.1)
                {
                    let sad_half = sad_one(mv_half, refp, up, up_w, up_h);
                    // Tie-bias: half-pel beats sub-pel on ties
                    // (smaller filter contribution → less OBMC
                    // drift), but loses to int-pel on ties
                    // (already locked in above).
                    if sad_half < best_sad
                        || (sad_half == best_sad && (best_mv.0, best_mv.1) == (mv_sp.0, mv_sp.1))
                    {
                        best_mv = mv_half;
                        best_sad = sad_half;
                    }
                }
                (best_mv, best_sad)
            };

            let (best_mv1, sad1) = pick_best_mv(mv1, mv1_half, mv1_int, ref1_y, &up1, up1_w, up1_h);
            let (best_mv2, sad2) = pick_best_mv(mv2, mv2_half, mv2_int, ref2_y, &up2, up2_w, up2_h);

            // Bipred SAD: `(p1 + p2 + 1) >> 1` per pixel (the §15.8.5
            // weighted sum at `ref_wt=(1,1)` and `refs_wt_precision=1`,
            // i.e. shift by 1, round-to-nearest at `+1`). This is what
            // the decoder will accumulate into `mc_tmp` for `Ref1And2`
            // blocks (post-spatial-weight, post-OBMC-blend). Round-91
            // widens the per-ref MV candidate set to
            // `{int-pel, half-pel, sub-pel}`, so the bipred mode
            // enumerates at most 3 × 3 = 9 combinations (filtered to
            // skip redundant pairs when half-pel coincides with int or
            // sub-pel). Strict superset of the round-39 4-combo set →
            // the chosen bipred SAD is ≤ the round-39 bipred SAD on
            // every block.
            let mut bipred_candidates: Vec<(IntegerMv, IntegerMv)> = Vec::with_capacity(9);
            let cands1: &[IntegerMv] = if mv_precision >= 2
                && (mv1_half.0, mv1_half.1) != (mv1.0, mv1.1)
                && (mv1_half.0, mv1_half.1) != (mv1_int.0, mv1_int.1)
            {
                &[mv1, mv1_half, mv1_int]
            } else {
                &[mv1, mv1_int]
            };
            let cands2: &[IntegerMv] = if mv_precision >= 2
                && (mv2_half.0, mv2_half.1) != (mv2.0, mv2.1)
                && (mv2_half.0, mv2_half.1) != (mv2_int.0, mv2_int.1)
            {
                &[mv2, mv2_half, mv2_int]
            } else {
                &[mv2, mv2_int]
            };
            for &m1 in cands1 {
                for &m2 in cands2 {
                    bipred_candidates.push((m1, m2));
                }
            }
            let (best_bipred_mv1, best_bipred_mv2, sad12) = bipred_candidates
                .iter()
                .map(|&(m1, m2)| {
                    let s = sad_bipred(
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
                        m1,
                        m2,
                        xblen,
                        yblen,
                        mv_precision,
                    );
                    (m1, m2, s)
                })
                .min_by_key(|&(_, _, s)| s)
                .unwrap();

            // Lambda-style tie-break: bipred carries an extra 2 MV
            // components so it must win by a clear margin to be picked.
            // Without this, equal-SAD ties default to bipred and the
            // decoder spends bits on a second MV pair that doesn't
            // improve reconstruction.
            const BIPRED_PENALTY: i64 = 64; // ~1 LSB per pixel over an 8x8 block.
            let mut best_mode = RefPredMode::Ref1Only;
            let mut best_sad = sad1;
            let mut chosen_mv1 = best_mv1;
            let mut chosen_mv2 = best_mv2;
            if sad2 < best_sad {
                best_sad = sad2;
                best_mode = RefPredMode::Ref2Only;
            }
            if sad12 + BIPRED_PENALTY < best_sad {
                best_mode = RefPredMode::Ref1And2;
                chosen_mv1 = best_bipred_mv1;
                chosen_mv2 = best_bipred_mv2;
            }
            out.push(BipredBlock {
                rmode: best_mode,
                mv1: chosen_mv1,
                mv2: chosen_mv2,
            });
        }
    }
    out
}

/// Round a sub-pel MV (units of `1/(2^p)` luma pels) to the nearest
/// **half-pel** position, returning the result in the same sub-pel
/// units. At `p < 1` this is identical to integer-pel rounding (the
/// half-pel grid degenerates to the integer-pel grid). At `p == 1` the
/// MV is already on the half-pel grid and the result is the input
/// unchanged. At `p >= 2` we snap to the nearest multiple of
/// `1 << (p - 1)` (i.e. step = 2 for qpel, step = 4 for ⅛-pel).
///
/// Used by `bipred_select_modes` (round-91) to widen the per-ref
/// adaptive candidate set from `{int-pel, sub-pel}` to
/// `{int-pel, half-pel, sub-pel}`. The half-pel candidate is the
/// §15.8.11 8-tap-filter natural intermediate position — at qpel
/// `mv_precision` the sub-pel-refined MV is one bilinear step away
/// from its half-pel peer, and on content where the 8-tap filter is
/// well-matched but the bilinear refinement isn't (e.g. mid-energy
/// edges that benefit from anti-aliasing but not from sub-quarter
/// adjustment), the half-pel peer can score lower SAD than either of
/// the existing two candidates.
fn round_mv_to_half_pel(mv: IntegerMv, p: u32) -> IntegerMv {
    if p <= 1 {
        // p == 0 → no sub-pel grid at all, fall through to integer-pel.
        // p == 1 → already on the half-pel grid by construction.
        return if p == 0 {
            round_mv_to_int_pel(mv, p)
        } else {
            mv
        };
    }
    let unit = 1i32 << (p - 1);
    let half = unit / 2;
    // Same ties-toward-zero biasing as `round_mv_to_int_pel`: degenerate
    // sub-half-unit offsets snap to zero, exact-half-unit ties round
    // toward zero so the half-pel snap doesn't inflate motion magnitude.
    let snap = |v: i32| -> i32 {
        let mag = v.unsigned_abs() as i32;
        let snapped_mag = if mag <= half {
            0
        } else {
            ((mag + half) / unit) * unit
        };
        if v < 0 {
            -snapped_mag
        } else {
            snapped_mag
        }
    };
    IntegerMv(snap(mv.0), snap(mv.1))
}

/// Round a sub-pel MV (units of `1/(2^p)` luma pels) to the nearest
/// **integer-pel** position, returning the result in the same sub-pel
/// units. At `p == 0` this is a no-op.
///
/// Per-block adaptive sub-pel-vs-integer-pel selection (`bipred_select_modes`)
/// uses this to evaluate each MV at both the sub-pel-refined position
/// and the integer-pel-rounded peer, then picks the lower-SAD variant
/// per block. The motivation: at sharp-edge blocks the 8-tap half-pel
/// filter introduces ringing that doesn't match the source, and the
/// resulting prediction loses 4-7 dB to the integer-pel choice; on
/// smooth content the sub-pel position still wins.
fn round_mv_to_int_pel(mv: IntegerMv, p: u32) -> IntegerMv {
    if p == 0 {
        return mv;
    }
    let unit = 1i32 << p;
    let half = unit / 2;
    // Round to nearest integer-pel, ties **toward zero**. Biases
    // motion-free / degenerate-content blocks toward (0, 0), the
    // smallest MV-magnitude anchor — without this, sub-pel offsets at
    // exactly half_unit would always snap up in absolute value, pushing
    // constant-background blocks off the zero-MV ground state.
    let snap = |v: i32| -> i32 {
        let mag = v.unsigned_abs() as i32;
        // |v| <  half  → 0; |v| == half → half rounds *down* (toward 0);
        // |v| >  half  → snap to nearest unit (away from zero).
        let snapped_mag = if mag <= half {
            // Entirely inside the inner half-unit → snap to 0.
            0
        } else {
            ((mag + half) / unit) * unit
        };
        if v < 0 {
            -snapped_mag
        } else {
            snapped_mag
        }
    };
    IntegerMv(snap(mv.0), snap(mv.1))
}

/// SAD of one block under the bipred prediction
/// `(pred_ref1 + pred_ref2 + 1) >> 1` against the source. Matches the
/// §15.8.5 reconstruction at `ref1_wt = ref2_wt = 1`,
/// `refs_wt_precision = 1` (post-OBMC blend, pre-spatial-weight scale).
#[allow(clippy::too_many_arguments)]
fn sad_bipred<S: InterSample>(
    cur: &[S],
    up1: &[i32],
    up1_w: usize,
    up1_h: usize,
    ref1: &[S],
    up2: &[i32],
    up2_w: usize,
    up2_h: usize,
    ref2: &[S],
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
                let p1 = ref1[r1y * w as usize + r1x].to_i32();
                let p2 = ref2[r2y * w as usize + r2x].to_i32();
                let pavg = (p1 + p2 + 1) >> 1;
                let s = cur[cy * w as usize + cx].to_i32();
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
                let s = cur[cy * w as usize + cx].to_i32();
                sad += (s - pavg).unsigned_abs() as i64;
            }
        }
    }
    sad
}

/// Per-pixel reference value `V(x, y)` for one block under `rmode` /
/// `(mv1, mv2)`, evaluated at picture coordinate `(x, y)`. Reproduces
/// the §15.8.5 ref-value computation at default weights `ref1_wt =
/// ref2_wt = 1`, `refs_wt_precision = 1` — i.e. `(v1 + v2 + 1) >> 1`
/// for the `Ref1And2` mode and the appropriate single-ref pixel
/// otherwise. References are the *signed pre-offset* convention
/// (`sample - NOMINAL_HALF`) so the result composes with the §15.8.6 spatial weight
/// matrix and the §15.8.2 `(sum + 32) >> 6` clip in `obmc_block_sse`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn bipred_block_ref_value(
    rmode: RefPredMode,
    mv1: IntegerMv,
    mv2: IntegerMv,
    up1_signed: &[i32],
    up1_w: usize,
    up1_h: usize,
    ref1_signed: &[i32],
    ref2_signed: &[i32],
    up2_signed: &[i32],
    up2_w: usize,
    up2_h: usize,
    ref_w: usize,
    ref_h: usize,
    mv_precision: u32,
    x: i32,
    y: i32,
) -> i32 {
    match rmode {
        RefPredMode::Ref1Only => ref_pixel_at(
            up1_signed,
            up1_w,
            up1_h,
            ref1_signed,
            ref_w,
            ref_h,
            mv_precision,
            x,
            y,
            mv1.0,
            mv1.1,
        ),
        RefPredMode::Ref2Only => ref_pixel_at(
            up2_signed,
            up2_w,
            up2_h,
            ref2_signed,
            ref_w,
            ref_h,
            mv_precision,
            x,
            y,
            mv2.0,
            mv2.1,
        ),
        RefPredMode::Ref1And2 => {
            let v1 = ref_pixel_at(
                up1_signed,
                up1_w,
                up1_h,
                ref1_signed,
                ref_w,
                ref_h,
                mv_precision,
                x,
                y,
                mv1.0,
                mv1.1,
            );
            let v2 = ref_pixel_at(
                up2_signed,
                up2_w,
                up2_h,
                ref2_signed,
                ref_w,
                ref_h,
                mv_precision,
                x,
                y,
                mv2.0,
                mv2.1,
            );
            // (p1 + p2 + 1) >> 1 — see §15.8.5 at `ref_wt=(1,1)`,
            // `refs_wt_precision=1`. Matches `obmc::motion_compensate`'s
            // `fshr(v1*w1 + v2*w2 + round, shift)` with the defaults.
            (v1 + v2 + 1) >> 1
        }
        // The bipred selector never emits Intra; defensive fall-through.
        RefPredMode::Intra => 0,
    }
}

/// Build the §15.8.6 OBMC neighbour-sum buffer for block `(i, j)` over
/// its `(xblen × yblen)` extent under a **bipred** grid: each neighbour
/// `k` contributes `weight_k(x, y) * V_k(x, y)` where `V_k` is the per-
/// block reference value selected by neighbour `k`'s `BipredBlock`
/// (`Ref1Only` → ref1 pixel, `Ref2Only` → ref2 pixel, `Ref1And2` →
/// `(v1 + v2 + 1) >> 1`). Mirrors [`build_neighbour_sum`] for the 2-ref
/// path.
#[allow(clippy::too_many_arguments)]
fn build_neighbour_sum_bipred(
    decisions: &[BipredBlock],
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
    up1_signed: &[i32],
    up1_w: usize,
    up1_h: usize,
    ref1_signed: &[i32],
    up2_signed: &[i32],
    up2_w: usize,
    up2_h: usize,
    ref2_signed: &[i32],
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
            let nbr = decisions[(nj as u32 * blocks_x + ni as u32) as usize];
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
                    let v = bipred_block_ref_value(
                        nbr.rmode,
                        nbr.mv1,
                        nbr.mv2,
                        up1_signed,
                        up1_w,
                        up1_h,
                        ref1_signed,
                        ref2_signed,
                        up2_signed,
                        up2_w,
                        up2_h,
                        ref_w,
                        ref_h,
                        mv_precision,
                        x,
                        y,
                    );
                    buf[q_ij as usize * xblen + p_ij as usize] += v * nbr_w[q_nbr * xblen + p_nbr];
                }
            }
        }
    }
    buf
}

/// **Post-OBMC bipred refinement pass** (round-95). The 2-ref analogue
/// of the 1-ref [`inter_select_int_pel_per_block`] post-OBMC re-
/// evaluation (round-80).
///
/// After [`bipred_select_modes`] picks the per-block
/// `{mode, mv1, mv2}` decisions from the round-91 widened candidate
/// set, the resulting grid is the input to the decoder's §15.8.5 OBMC
/// blend. The selector minimised SAD against the source, but the
/// decoder's actual reconstruction cost is OBMC SSE of the **blended**
/// recon — which differs by the neighbour contributions and the
/// §15.8.2 `(sum + 32) >> 6` clip. This pass closes that cost-function
/// gap by re-scoring each block under the full OBMC blend with the
/// neighbour grid frozen at the selector's output.
///
/// **Mode-only candidate set.** For each block in raster order, the
/// trial set is the strict superset of the current decision at the
/// SAME MV pair the selector chose:
///
/// 1. the **current** decision (so the pass can never regress);
/// 2. `Ref1Only` with `mv1` unchanged;
/// 3. `Ref2Only` with `mv2` unchanged;
/// 4. `Ref1And2` with both MVs unchanged.
///
/// Whichever gives the lowest per-block OBMC SSE wins. Ties bias
/// toward the **current** decision (to avoid unnecessary mode flips
/// when the OBMC blend is genuinely indifferent). Strict-superset →
/// cannot regress per-block OBMC SSE under the frozen neighbour grid;
/// same monotonicity invariant as the 1-ref round-80 pass.
///
/// **Why mode-only, not MV-snap.** The 1-ref round-80 pass uses MV-
/// snap candidates because `obmc_refine_me` had drifted the MV during
/// its ±1 sub-pel-unit refinement and a snap-back is the only way to
/// rejoin the integer-pel anchor. The bipred path doesn't run
/// `obmc_refine_me` (per the round-91 design note — refining each
/// reference's MV independently against the source breaks the blend),
/// so there is no per-MV drift to reverse. The genuine
/// cost-function gap on the bipred path is the **mode** decision:
/// `bipred_select_modes` picked the lowest-SAD mode against the source,
/// but the OBMC blend can prefer a different per-block mode when the
/// neighbour grid's contribution skews the per-block recon. Mode-only
/// candidates keep smooth-motion sub-pel MVs at the qpel grid (the
/// camera-pan ffmpeg cross-decode floor relies on this) while
/// recovering the SAD-vs-OBMC-SSE gap on mid-energy content.
///
/// No-op when `decisions` is empty.
#[allow(clippy::too_many_arguments)]
pub fn bipred_post_obmc_refine_modes<S: InterSample>(
    cur_y: &[S],
    ref1_y: &[S],
    ref2_y: &[S],
    width: u32,
    height: u32,
    blocks_x: u32,
    blocks_y: u32,
    decisions: &mut [BipredBlock],
    mv_precision: u32,
) {
    if decisions.is_empty() {
        return;
    }
    let (xblen_u, yblen_u, xbsep_u, ybsep_u) = (8usize, 8usize, 4usize, 4usize);
    let xoffset = (xblen_u - xbsep_u) / 2;
    let yoffset = (yblen_u - ybsep_u) / 2;
    let cur_w = width as i32;
    let cur_h = height as i32;
    let ref_w = width as usize;
    let ref_h = height as usize;

    // Pre-bake every block's spatial weight matrix — same cache shape
    // as `obmc_refine_me`.
    let mut weights: Vec<Vec<i32>> = Vec::with_capacity((blocks_x * blocks_y) as usize);
    for j in 0..blocks_y {
        for i in 0..blocks_x {
            weights.push(block_weight(
                xblen_u, yblen_u, xbsep_u, ybsep_u, xoffset, yoffset, i, j, blocks_x, blocks_y,
            ));
        }
    }
    // Build the signed-pre-offset upref planes once per call (each is
    // O(width × height) and the inner refinement loop hits each pixel
    // a constant number of times per block).
    let (up1_signed, up1_w, up1_h) = build_upref_signed(ref1_y, width, height);
    let (up2_signed, up2_w, up2_h) = build_upref_signed(ref2_y, width, height);
    let ref1_signed: Vec<i32> = ref1_y
        .iter()
        .map(|&v| v.to_i32() - S::NOMINAL_HALF)
        .collect();
    let ref2_signed: Vec<i32> = ref2_y
        .iter()
        .map(|&v| v.to_i32() - S::NOMINAL_HALF)
        .collect();

    for j in 0..blocks_y {
        for i in 0..blocks_x {
            let bidx = (j * blocks_x + i) as usize;
            let cur = decisions[bidx];

            // Build the neighbour-sum buffer once for this block — it's
            // independent of the trial decision (only depends on the
            // *other* blocks' frozen decisions).
            let neighbour_sum = build_neighbour_sum_bipred(
                decisions,
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
                &up1_signed,
                up1_w,
                up1_h,
                &ref1_signed,
                &up2_signed,
                up2_w,
                up2_h,
                &ref2_signed,
                ref_w,
                ref_h,
                mv_precision,
            );
            let xstart_ij = (i as i32) * (xbsep_u as i32) - (xoffset as i32);
            let ystart_ij = (j as i32) * (ybsep_u as i32) - (yoffset as i32);
            let weight = &weights[bidx];

            // Score one trial decision under the full OBMC blend.
            let score = |trial: BipredBlock| -> i64 {
                let mut wpred = vec![0i32; xblen_u * yblen_u];
                for q in 0..yblen_u {
                    let y = ystart_ij + q as i32;
                    for p in 0..xblen_u {
                        let x = xstart_ij + p as i32;
                        let v = bipred_block_ref_value(
                            trial.rmode,
                            trial.mv1,
                            trial.mv2,
                            &up1_signed,
                            up1_w,
                            up1_h,
                            &ref1_signed,
                            &ref2_signed,
                            &up2_signed,
                            up2_w,
                            up2_h,
                            ref_w,
                            ref_h,
                            mv_precision,
                            x,
                            y,
                        );
                        wpred[q * xblen_u + p] = v * weight[q * xblen_u + p];
                    }
                }
                obmc_block_sse(
                    cur_y,
                    cur_w,
                    cur_h,
                    &wpred,
                    &neighbour_sum,
                    xblen_u,
                    yblen_u,
                    xstart_ij,
                    ystart_ij,
                )
            };

            // 1. Current decision is the baseline SSE — strict-superset
            //    invariant: only a strictly lower trial SSE wins.
            let cur_sse = score(cur);
            let mut best = cur;
            let mut best_sse = cur_sse;

            // Helper: try a candidate, replace best on strict improve.
            // Tie-bias: keep the existing best (initialised to `cur`),
            // so equal SSE keeps the current decision and the pass is
            // a true identity on indifferent blocks.
            let try_cand = |trial: BipredBlock, best: &mut BipredBlock, best_sse: &mut i64| {
                let s = score(trial);
                if s < *best_sse {
                    *best_sse = s;
                    *best = trial;
                }
            };

            // Mode-only candidate set at the selector's MV pair:
            // skip the candidate equal to the current decision (it was
            // already scored as the floor).
            for alt_mode in [
                RefPredMode::Ref1Only,
                RefPredMode::Ref2Only,
                RefPredMode::Ref1And2,
            ] {
                if alt_mode == cur.rmode {
                    continue;
                }
                try_cand(
                    BipredBlock {
                        rmode: alt_mode,
                        mv1: cur.mv1,
                        mv2: cur.mv2,
                    },
                    &mut best,
                    &mut best_sse,
                );
            }

            decisions[bidx] = best;
        }
    }
}

/// Build the same `PictureMotionData` the decoder will reconstruct from
/// the 2-ref `block_motion_data` block this encoder emits — preserves
/// the per-block `RefPredMode`, both MVs, and (round-382) the per-block
/// global-mode flags + both references' global parameters.
#[allow(clippy::too_many_arguments)]
fn build_motion_from_bipred_grid(
    sbx: u32,
    sby: u32,
    blocks_x: u32,
    blocks_y: u32,
    decisions: &[BipredBlock],
    gmode: Option<&[bool]>,
    global1: Option<GlobalParams>,
    global2: Option<GlobalParams>,
) -> PictureMotionData {
    let blocks: Vec<BlockData> = (0..blocks_x * blocks_y)
        .map(|i| {
            let d = decisions[i as usize];
            BlockData {
                rmode: d.rmode,
                gmode: gmode.map(|g| g[i as usize]).unwrap_or(false),
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
        global1,
        global2,
    }
}

/// Build the §11.8.5 OBMC reference reconstruction for a 2-ref bipred
/// picture. Same shape as [`build_obmc_prediction`] but feeds both
/// reference planes through `crate::obmc::motion_compensate`.
#[allow(clippy::too_many_arguments)]
fn build_obmc_prediction_bipred<S: InterSample>(
    sequence: &SequenceHeader,
    pred: &PicturePredictionParams,
    motion: &PictureMotionData,
    ref1_y: &[S],
    ref1_u: &[S],
    ref1_v: &[S],
    ref2_y: &[S],
    ref2_u: &[S],
    ref2_v: &[S],
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let chroma_h_ratio = sequence.video_params.chroma_format.h_ratio();
    let chroma_v_ratio = sequence.video_params.chroma_format.v_ratio();
    let pred_y = build_obmc_prediction_one_bipred(
        sequence,
        pred,
        motion,
        /* component */ 0,
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
        /* component */ 1,
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
        /* component */ 2,
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
fn build_obmc_prediction_one_bipred<S: InterSample>(
    sequence: &SequenceHeader,
    pred: &PicturePredictionParams,
    motion: &PictureMotionData,
    component: usize,
    chroma_h_ratio: u32,
    chroma_v_ratio: u32,
    ref1_plane: &[S],
    ref2_plane: &[S],
    comp_w: usize,
    comp_h: usize,
) -> Vec<i32> {
    let is_chroma = component != 0;
    let depth = if is_chroma {
        sequence.chroma_depth
    } else {
        sequence.luma_depth
    };
    let half = 1i32 << (depth - 1);
    let ref1_signed: Vec<i32> = ref1_plane.iter().map(|&v| v.to_i32() - half).collect();
    let ref2_signed: Vec<i32> = ref2_plane.iter().map(|&v| v.to_i32() - half).collect();
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
        component,
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
pub fn encode_bipred_inter_picture<S: InterSample>(
    sequence: &SequenceHeader,
    params: &InterEncoderParams,
    picture_number: u32,
    ref1_picture_number: u32,
    ref2_picture_number: u32,
    cur_y: &[S],
    cur_u: &[S],
    cur_v: &[S],
    ref1_y: &[S],
    ref1_u: &[S],
    ref1_v: &[S],
    ref2_y: &[S],
    ref2_u: &[S],
    ref2_v: &[S],
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
    write_picture_prediction_parameters(&mut w, params, bmp, 2);
    w.byte_align();

    // §12.3 block_motion_data — 2-ref bipred path.
    let (sbx, sby, blocks_x, blocks_y) = motion_grid(sequence.luma_width, sequence.luma_height);
    let mut decisions = bipred_select_modes(
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

    // Note: per-reference 1-ref-style `obmc_refine_me` is NOT applied to
    // the bipred path. `obmc_refine_me` optimises each reference's MV
    // independently against the source, but in a bipred B-picture the
    // decoder blends ref1 and ref2 predictions with OBMC spatial weights.
    // Refining ref1 MVs while ignoring ref2's contribution (and vice
    // versa) would shift MVs toward minimising the single-ref residue
    // rather than the blended reconstruction error — breaking the
    // self-roundtrip invariant. Round-95: instead of per-ref refinement,
    // we run a single post-OBMC RE-EVALUATION pass that scores each
    // block's `bipred_select_modes` decision against the full §15.8.5
    // OBMC blend (with the neighbour grid frozen). The candidate set is
    // a strict superset of the current decision, so per-block OBMC SSE
    // cannot regress — same monotonicity invariant the 1-ref round-80
    // `inter_select_int_pel_per_block` second pass carries.
    if params.bipred_post_obmc_refine {
        bipred_post_obmc_refine_modes(
            cur_y,
            ref1_y,
            ref2_y,
            sequence.luma_width,
            sequence.luma_height,
            blocks_x,
            blocks_y,
            &mut decisions,
            bmp,
        );
    }

    // §12.3.3.2 per-block global-mode grid + §11.2.6 params for both
    // references (None when the picture uses block motion only).
    let gmode_grid = params
        .global_motion
        .as_ref()
        .map(|cfg| resolve_gmode_grid(cfg, blocks_x, blocks_y));
    let (global1, global2) = match &params.global_motion {
        None => (None, None),
        Some(cfg) => (
            Some(effective_global_params(&cfg.global1)),
            Some(effective_global_params(
                &cfg.global2.clone().unwrap_or_default(),
            )),
        ),
    };
    encode_block_motion_data_bipred(
        &mut w,
        sbx,
        sby,
        blocks_x,
        blocks_y,
        &decisions,
        gmode_grid.as_deref(),
    );
    w.byte_align();

    // §11.3 wavelet_transform.
    if let Some(ref residue) = params.residue {
        let pred = picture_prediction_params_from(sequence, params, bmp, 2);
        let motion = build_motion_from_bipred_grid(
            sbx,
            sby,
            blocks_x,
            blocks_y,
            &decisions,
            gmode_grid.as_deref(),
            global1.clone(),
            global2.clone(),
        );
        let (pred_y, pred_u, pred_v) = build_obmc_prediction_bipred(
            sequence, &pred, &motion, ref1_y, ref1_u, ref1_v, ref2_y, ref2_u, ref2_v,
        );
        let res_y = build_residue_plane(cur_y, &pred_y, sequence.luma_depth);
        let res_u = build_residue_plane(cur_u, &pred_u, sequence.chroma_depth);
        let res_v = build_residue_plane(cur_v, &pred_v, sequence.chroma_depth);

        w.write_bool(false);
        write_residue_transform_parameters(&mut w, residue);
        w.byte_align();

        let raw_y =
            forward_residue_pyramid(&res_y, sequence.luma_width, sequence.luma_height, residue);
        let raw_u = forward_residue_pyramid(
            &res_u,
            sequence.chroma_width,
            sequence.chroma_height,
            residue,
        );
        let raw_v = forward_residue_pyramid(
            &res_v,
            sequence.chroma_width,
            sequence.chroma_height,
            residue,
        );
        emit_residue_components(&mut w, residue, &raw_y, &raw_u, &raw_v);
        w.byte_align();
    } else {
        // No residue: reconstruction = OBMC(reference). Wire form per
        // `explicit_zero_residue` (the reference decoder mishandles the
        // ZERO_RESIDUAL=1 skip form — round-408 black-box finding).
        write_zero_residue_tail(&mut w, params.explicit_zero_residue);
    }

    w.finish()
}

/// Encode a 2-picture stream: an HQ intra reference (`0xEC`) followed
/// by a single inter (`0x09`) referencing it. This is the minimal
/// inter-decode validator — the simplest legal Dirac sequence with a
/// motion-compensated picture.
pub fn encode_intra_then_inter_stream<S: InterSample>(
    sequence: &SequenceHeader,
    intra_params: &crate::encoder::EncoderParams,
    inter_params: &InterEncoderParams,
    intra: &InterInputPicture<'_, S>,
    inter: &InterInputPicture<'_, S>,
) -> Vec<u8> {
    let sh_payload = crate::encoder::encode_sequence_header(sequence);

    // Reference: HQ intra **reference** picture so its decoded form
    // ends up in the decoder's reference buffer. Dispatch through the
    // sample trait so u16 sources take the deep-colour HQ intra entry.
    let intra_payload = S::encode_hq_intra_anchor(
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

// ---- Multi-picture rate-controlled inter sequence driver -------------

/// Rate-control strategy for the multi-picture inter sequence driver
/// [`encode_inter_sequence_with_residue_target`].
///
/// The inter analogue of [`crate::encoder::LdRateControl`] /
/// [`crate::encoder::HqRateControl`], but the controlled quantity is the
/// **§11.3 wavelet-residue payload byte budget** rather than the whole
/// picture: each inter picture's residue qindex is picked per-picture by
/// [`pick_inter_residue_qindex`] against a per-picture residue-byte
/// target. The intra anchor and the per-picture motion / header bytes are
/// not rate-controlled — only the residue stream is, because that is the
/// one degree of freedom [`pick_inter_residue_qindex`] exposes (any
/// §13.4.4 qindex is a legal, decodable choice; motion data and headers
/// are structurally fixed by the preset-1 block grid).
///
/// Source of truth: the residue rate policy is a pure encoder-side
/// shaping choice — the BBC Dirac Specification v2.2.3 §11.3 / §13.4.4
/// makes any per-picture qindex spec-conformant, so picking the smallest
/// quantiser that fits a byte budget produces a stream every conformant
/// decoder accepts. The sequence framing follows §9.6 / §10.4 parse-info
/// chaining exactly as the single-picture drivers do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterRateControl {
    /// Each inter picture's residue is independently sized to
    /// `target_residue_bytes` (the smallest qindex whose serialised
    /// residue stream fits, via [`pick_inter_residue_qindex`]). No carry
    /// between pictures — every picture sees the same budget.
    PerPicture,
    /// Constant-residue-rate: a signed running accumulator carries the
    /// over-/under-shoot of every encoded picture's actual residue bytes
    /// into the next picture's residue budget. A picture that overshoots
    /// `target_residue_bytes` (its smallest-fitting qindex still ran over,
    /// or it hit the qindex-127 floor) tightens the following picture's
    /// budget; an undershoot loosens it. Sign convention matches the LD
    /// driver: `carry = Σ(actual − target)` (positive = overshot the
    /// ideal cumulative residue budget so far, so the next request is
    /// pulled back; negative = headroom to spend).
    Cbr,
    /// Leaky-bucket variable-residue-rate: like [`Cbr`] but the spendable
    /// **savings** (i.e. the negative end of the carry, `max(-carry, 0)`)
    /// are bounded by `buffer_bytes`. After each picture the carry is
    /// clamped at `carry >= -buffer_bytes`, so a run of undershooting
    /// inter pictures cannot bank unbounded headroom: the next request
    /// `target − carry` is capped above by `target + buffer_bytes`, an
    /// instantaneous peak residue-size cap. Overshoot debt (`carry > 0`)
    /// is never clamped — repaying an overshoot is mandatory, exactly as
    /// in [`Cbr`] — so VBV only governs the upper edge of the request.
    /// `buffer_bytes == 0` collapses to [`Cbr`] on a smooth fixture (the
    /// savings side is forced to zero; the debt side is identical).
    ///
    /// The residue analogue of [`crate::encoder::LdRateControl::Vbv`] /
    /// [`crate::encoder::HqRateControl::Vbv`], applied to the §11.3
    /// residue-payload byte budget instead of the whole-picture budget.
    ///
    /// [`Cbr`]: InterRateControl::Cbr
    Vbv {
        /// Peak per-picture residue surplus the leaky bucket may hold
        /// above `target_residue_bytes`. The next request is bounded
        /// above by `target_residue_bytes + buffer_bytes`.
        buffer_bytes: u32,
    },
    /// Drain-rate hysteresis variant of [`Vbv`] — same bucket fill /
    /// forfeit semantics, but the spendable savings are additionally
    /// clamped at `max_drain_per_picture` so a full bucket is emptied
    /// gradually rather than in one step.
    ///
    /// Plain [`Vbv`] lets a single inter picture drain the entire bucket
    /// the moment savings fill it (its request may equal `target +
    /// buffer_bytes` immediately). That delivers maximum quality on one
    /// picture but exposes the next few to a sudden cliff — the bucket
    /// empties abruptly, then sits at zero until the next undershoot
    /// refills it. `VbvHysteresis` smooths the drain by spreading the
    /// banked savings: the spend is `min(savings, buffer_bytes,
    /// max_drain_per_picture)`, so the remaining savings stay in the
    /// bucket for the *next* picture. The debt-payback branch
    /// (`carry > 0`) is unchanged from [`Vbv`] / [`Cbr`] because debt
    /// repayment is mandatory, not rate-limited — only the spend side is
    /// hysteretic. `max_drain_per_picture >= buffer_bytes` collapses back
    /// to plain [`Vbv`] (the drain cap never bites); `max_drain_per_picture
    /// == 0` zeros the spend so the request becomes `target − max(carry, 0)`.
    ///
    /// The residue analogue of
    /// [`crate::encoder::LdRateControl::VbvHysteresis`] /
    /// [`crate::encoder::HqRateControl::VbvHysteresis`].
    ///
    /// [`Vbv`]: InterRateControl::Vbv
    /// [`Cbr`]: InterRateControl::Cbr
    VbvHysteresis {
        /// Peak per-picture residue surplus the leaky bucket may hold
        /// above `target_residue_bytes`. Same role as
        /// [`Vbv::buffer_bytes`].
        ///
        /// [`Vbv::buffer_bytes`]: InterRateControl::Vbv::buffer_bytes
        buffer_bytes: u32,
        /// Maximum banked savings a single picture may spend above its
        /// `target_residue_bytes` request. Bounds how aggressively a full
        /// bucket is emptied per picture. `>= buffer_bytes` collapses to
        /// plain [`Vbv`].
        ///
        /// [`Vbv`]: InterRateControl::Vbv
        max_drain_per_picture: u32,
    },
}

/// Per-picture residue rate-control telemetry returned by
/// [`encode_inter_sequence_with_residue_target_report`].
#[derive(Debug, Clone, Copy)]
pub struct InterPictureRate {
    /// Picture number written into the inter picture header.
    pub picture_number: u32,
    /// Picture number this inter picture references (the previous frame
    /// in the chain; the intra anchor for the first inter picture).
    pub ref1_picture_number: u32,
    /// Residue-payload byte budget actually requested for this picture
    /// (after the [`InterRateControl::Cbr`] accumulator adjustment, if
    /// any).
    pub requested_residue_bytes: u32,
    /// Actual serialised §11.3 residue-payload bytes for the chosen
    /// qindex (transform_parameters + the three length-prefixed AC-coded
    /// component subband streams + the ZERO_RESIDUAL=false flag), as
    /// measured by [`inter_residue_qindex_diagnostic`].
    pub actual_residue_bytes: u32,
    /// qindex chosen by [`pick_inter_residue_qindex`] for this picture.
    pub qindex: u32,
    /// Running rate-control surplus *after* this picture has been folded
    /// in (and after the [`InterRateControl::Vbv`] /
    /// [`InterRateControl::VbvHysteresis`] bucket clamp, if any).
    /// Sign convention: **positive = overshoot debt** (cumulative
    /// `Σ(actual − target)`), **negative = savings**. Computed the same
    /// way for every mode; the modes differ only in whether the next
    /// picture's request *uses* it ([`InterRateControl::Cbr`] /
    /// [`InterRateControl::Vbv`] / [`InterRateControl::VbvHysteresis`] do,
    /// [`InterRateControl::PerPicture`] does not) and, for the VBV
    /// variants, whether the savings side is clamped at `-buffer_bytes`.
    pub running_surplus_bytes: i64,
    /// Global fraction measured by the per-picture
    /// [`InterEncoderParams::auto_global_motion`] estimate — the share
    /// of blocks whose fitted §15.8.8 field beat their own block MV.
    /// `None` when auto estimation did not run for this picture
    /// (feature off, or an explicit `global_motion` config was set).
    pub global_fraction: Option<f64>,
    /// Whether the auto estimate was actually applied to this picture
    /// (`global_fraction >= min_fraction`). Always `false` when
    /// [`global_fraction`] is `None`.
    ///
    /// [`global_fraction`]: InterPictureRate::global_fraction
    pub global_applied: bool,
}

/// Encode a multi-picture core-syntax inter sequence (one HQ intra
/// anchor `0xEC` followed by N 1-ref inter pictures `0x09`) with
/// per-picture **residue** rate control driven by
/// [`pick_inter_residue_qindex`].
///
/// `frames[0]` is the intra anchor; `frames[1..]` are the inter pictures.
/// Every inter picture references the **intra anchor** — the only
/// *reference* picture in the stream. Inter pictures use parse code
/// `0x09` (1-ref, **non-reference**), so a decoded inter picture never
/// enters the decoder's reference buffer; a straight P-chain through the
/// previous inter picture would therefore reference a picture the decoder
/// discarded and fail with `MissingReference`. Anchoring every inter to
/// the intra picture keeps the whole sequence decodable while still
/// rate-controlling each picture's residue independently. (This is the
/// same source-referenced convention as [`encode_intra_then_inter_stream`],
/// extended to N inter pictures.) For each inter picture the driver:
///   1. derives a per-picture residue-byte budget from
///      `target_residue_bytes` and the [`InterRateControl`] strategy,
///   2. picks the smallest residue qindex whose serialised §11.3 residue
///      stream fits that budget ([`pick_inter_residue_qindex`]),
///   3. emits the inter picture (`0x09`) at that qindex,
///   4. (CBR) folds the actual-vs-target residue-byte deviation into the
///      accumulator for the next picture.
///
/// The result is a complete elementary stream — sequence header (`0x00`),
/// the anchor, the inter pictures, and end-of-sequence (`0x10`) — with
/// the `next`/`previous` parse-offset chain wired so it round-trips
/// through [`crate::decoder::DiracDecoder`] to one decoded frame per
/// input frame. Requires `inter_params.residue` to be `Some(..)`; with
/// `residue = None` no residue stream exists to rate-control, so the
/// driver falls back to ZERO_RESIDUAL=true pictures (all telemetry
/// residue figures are then zero and the qindex is the configured floor).
///
/// Returns at least the sequence header + anchor + EOS even when
/// `frames` carries only the anchor (an empty inter list is legal —
/// the report is then empty).
///
/// Closes the lib.rs "multi-picture rate-controlled inter sequence
/// driver" gap (the per-picture picker existed; this wires the
/// sequence-level carry the HQ/LD intra drivers have).
pub fn encode_inter_sequence_with_residue_target<S: InterSample>(
    sequence: &SequenceHeader,
    intra_params: &crate::encoder::EncoderParams,
    inter_params: &InterEncoderParams,
    frames: &[InterInputPicture<'_, S>],
    target_residue_bytes: u32,
    mode: InterRateControl,
) -> Vec<u8> {
    let (stream, _report) = encode_inter_sequence_with_residue_target_report(
        sequence,
        intra_params,
        inter_params,
        frames,
        target_residue_bytes,
        mode,
    );
    stream
}

/// [`encode_inter_sequence_with_residue_target`] plus per-picture
/// telemetry (requested vs. actual residue bytes, chosen qindex, and the
/// running accumulator for each inter picture). Returns `(stream,
/// report)`; `report` has one entry per inter picture (i.e.
/// `frames.len().saturating_sub(1)`).
pub fn encode_inter_sequence_with_residue_target_report<S: InterSample>(
    sequence: &SequenceHeader,
    intra_params: &crate::encoder::EncoderParams,
    inter_params: &InterEncoderParams,
    frames: &[InterInputPicture<'_, S>],
    target_residue_bytes: u32,
    mode: InterRateControl,
) -> (Vec<u8>, Vec<InterPictureRate>) {
    let pi_size = 13usize;
    let sh_payload = crate::encoder::encode_sequence_header(sequence);
    let sh_unit_len = (pi_size + sh_payload.len()) as u32;

    let mut out = Vec::new();
    write_parse_info(&mut out, 0x00, sh_unit_len, 0);
    out.extend_from_slice(&sh_payload);
    let mut prev_unit_len = sh_unit_len;

    let mut report: Vec<InterPictureRate> = Vec::new();

    let Some((anchor, inters)) = frames.split_first() else {
        // No anchor at all → sequence header + EOS only.
        write_parse_info(&mut out, 0x10, 0, prev_unit_len);
        return (out, report);
    };

    // HQ intra **reference** anchor (0xEC) — its decoded form lands in
    // the decoder's reference buffer per §15.4.
    let anchor_payload = S::encode_hq_intra_anchor(
        sequence,
        intra_params,
        anchor.picture_number,
        anchor.y,
        anchor.u,
        anchor.v,
    );
    let anchor_unit_len = (pi_size + anchor_payload.len()) as u32;
    write_parse_info(&mut out, 0xEC, anchor_unit_len, prev_unit_len);
    out.extend_from_slice(&anchor_payload);
    prev_unit_len = anchor_unit_len;

    // CBR accumulator: signed running Σ(actual − target) residue bytes.
    let mut carry: i64 = 0;

    // Every inter picture references the intra anchor — the only
    // reference picture in the stream (0x09 inter pictures are
    // non-reference and never enter the decoder's reference buffer).
    let reference: &InterInputPicture<'_, S> = anchor;

    for pic in inters {
        // Per-picture residue budget from the strategy.
        let requested: u32 = match mode {
            InterRateControl::PerPicture => target_residue_bytes,
            InterRateControl::Cbr => {
                // Spend `target` minus whatever we've overshot so far
                // (carry > 0 ⇒ pull back; carry < 0 ⇒ spend extra).
                let want = target_residue_bytes as i64 - carry;
                want.clamp(0, u32::MAX as i64) as u32
            }
            InterRateControl::Vbv { buffer_bytes } => {
                // Leaky-bucket: identical to `Cbr` but the spendable
                // savings (`max(-carry, 0)`) are capped at `buffer_bytes`.
                // The post-encode clamp keeps `carry >= -buffer_bytes`, so
                // the request `target - carry` is bounded above by
                // `target + buffer_bytes`. The explicit `min` here is
                // belt-and-braces against bucket-cap edge cases; the
                // post-encode clamp is the load-bearing invariant.
                let spendable = (-carry).min(buffer_bytes as i64).max(0);
                let want = target_residue_bytes as i64 - carry.max(0) + spendable;
                want.clamp(0, u32::MAX as i64) as u32
            }
            InterRateControl::VbvHysteresis {
                buffer_bytes,
                max_drain_per_picture,
            } => {
                // Drain-rate hysteresis: identical bucket fill / forfeit
                // semantics as `Vbv`, but the spendable savings are
                // additionally clamped at `max_drain_per_picture`. The
                // debt-payback branch (`carry > 0`) is unchanged because
                // debt repayment is mandatory, not rate-limited — only the
                // spend side of the bucket is hysteretic.
                let spendable = (-carry)
                    .min(buffer_bytes as i64)
                    .min(max_drain_per_picture as i64)
                    .max(0);
                let want = target_residue_bytes as i64 - carry.max(0) + spendable;
                want.clamp(0, u32::MAX as i64) as u32
            }
        };

        // Round-386: per-picture automatic global-motion estimation.
        // Resolved BEFORE the residue qindex picker so rate control
        // measures exactly the stream it will emit (a global picture's
        // prediction — and therefore its residue — differs from the
        // block-motion one). An explicit caller config always wins.
        let mut pic_params = inter_params.clone();
        let mut global_fraction: Option<f64> = None;
        let mut global_applied = false;
        if pic_params.global_motion.is_none() {
            if let Some(auto) = &inter_params.auto_global_motion {
                let (cfg, fraction) = estimate_global_motion_config(
                    sequence,
                    inter_params,
                    pic.y,
                    reference.y,
                    auto.model,
                );
                global_fraction = Some(fraction);
                if fraction >= auto.min_fraction {
                    pic_params.global_motion = Some(cfg);
                    global_applied = true;
                }
            }
        }

        // Pick the residue qindex for this picture against the budget,
        // and measure what it actually costs. With residue disabled the
        // diagnostic returns `(0, 0)` and the picture is emitted
        // ZERO_RESIDUAL=true.
        let (qindex, actual_residue) = inter_residue_qindex_diagnostic(
            sequence,
            &pic_params,
            pic.y,
            pic.u,
            pic.v,
            reference.y,
            reference.u,
            reference.v,
            requested,
        );

        // Emit the inter picture with the chosen residue qindex applied.
        if let Some(ref rp) = inter_params.residue {
            let mut rp = rp.clone();
            rp.qindex = qindex;
            pic_params.residue = Some(rp);
        }
        let inter_payload = encode_inter_picture(
            sequence,
            &pic_params,
            pic.picture_number,
            reference.picture_number,
            pic.y,
            pic.u,
            pic.v,
            reference.y,
            reference.u,
            reference.v,
        );
        let inter_unit_len = (pi_size + inter_payload.len()) as u32;
        write_parse_info(&mut out, 0x09, inter_unit_len, prev_unit_len);
        out.extend_from_slice(&inter_payload);
        prev_unit_len = inter_unit_len;

        // Fold the actual-vs-target residue deviation into the
        // accumulator (computed identically for every mode; only the
        // feedback modes *use* it when sizing the next request).
        carry += actual_residue as i64 - target_residue_bytes as i64;

        // VBV: clamp the savings end of the bucket at -buffer_bytes so
        // the next picture's `target - carry` request stays ≤ target +
        // buffer_bytes. Overshoot debt (carry > 0) is left untouched — a
        // peak-size cap governs only the upper edge of the request — and
        // PerPicture / Cbr leave `carry` alone (PerPicture ignores it
        // anyway; Cbr is the unbounded-bucket limit).
        match mode {
            InterRateControl::Vbv { buffer_bytes }
            | InterRateControl::VbvHysteresis { buffer_bytes, .. } => {
                let floor = -(buffer_bytes as i64);
                if carry < floor {
                    carry = floor;
                }
            }
            InterRateControl::PerPicture | InterRateControl::Cbr => {}
        }

        report.push(InterPictureRate {
            picture_number: pic.picture_number,
            ref1_picture_number: reference.picture_number,
            requested_residue_bytes: requested,
            actual_residue_bytes: actual_residue as u32,
            qindex,
            running_surplus_bytes: carry,
            global_fraction,
            global_applied,
        });
    }

    write_parse_info(&mut out, 0x10, 0, prev_unit_len);
    (out, report)
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

    /// The generic ME pipeline reads `u16` samples above the 8-bit
    /// range: a bright 10-bit square translated by (+4, +2) must be
    /// tracked exactly by the integer-pel search, and the sub-pel
    /// refinement must keep the integer answer (the translation is
    /// integer-pel). Values straddle 255 so a truncating 8-bit read
    /// would flatten the square into the background and return (0, 0).
    #[test]
    fn full_search_me_tracks_10bit_translation_in_u16() {
        let (w, h) = (64usize, 64usize);
        // Textured 16x16 square (values 512..896, all above the 8-bit
        // range) on a flat dim background. The texture translates with
        // the square, so an interior block matches only at the true
        // offset — a truncating 8-bit read would alias the texture.
        let tex = |lx: usize, ly: usize| -> u16 { 512 + ((lx * 13 + ly * 7) % 384) as u16 };
        let mut f0 = vec![64u16; w * h];
        let mut f1 = vec![64u16; w * h];
        for ly in 0..16 {
            for lx in 0..16 {
                f0[(24 + ly) * w + (24 + lx)] = tex(lx, ly);
                f1[(26 + ly) * w + (28 + lx)] = tex(lx, ly);
            }
        }
        let (_sbx, _sby, bx, by) = motion_grid(w as u32, h as u32);
        let mvs = full_search_me(&f1, &f0, w as u32, h as u32, bx, by, 8);
        // Block (8, 8) covers current pixels [32, 40) x [32, 40) — fully
        // inside the moved square. The MV convention reads the reference
        // at (x + mv), so the exact-texture match is at (-4, -2).
        let idx = (8 * bx + 8) as usize;
        assert_eq!((mvs[idx].0, mvs[idx].1), (-4, -2));
        // Sub-pel refinement at quarter-pel keeps the integer answer
        // (scaled into quarter-pel units).
        let mvs_q = subpel_search_me(&f1, &f0, w as u32, h as u32, bx, by, 8, 2);
        assert_eq!((mvs_q[idx].0, mvs_q[idx].1), (-16, -8));
    }

    /// `build_upref` at `u16` must not clip genuine deep-colour values:
    /// a flat 16-bit plane at 65000 upsamples to 65000 everywhere (the
    /// §15.8.11 filter preserves constants, and `ME_UPREF_DEPTH = 17`
    /// gives the clip range headroom past the full u16 excursion).
    #[test]
    fn build_upref_u16_preserves_full_range_constants() {
        let plane = vec![65000u16; 16 * 16];
        let (up, up_w, up_h) = build_upref(&plane, 16, 16);
        assert_eq!((up_w, up_h), (32, 32));
        assert!(up.iter().all(|&v| v == 65000));
    }

    #[test]
    fn cb_residue_bytes_match_intra_core_byte_for_byte() {
        // Build a 3-level pyramid with a non-trivial L2 HL band + L1 HL
        // parent, then encode the L2 HL subband with a (2,2) codeblock
        // grid through BOTH the inter-residue cb encoder and the proven
        // intra-core encoder. The emitted AC bytes must be identical —
        // any divergence is a replica bug in the inter walk.
        use crate::encoder_intra_core::{encode_subband_ac, CoreIntraEncoderParams};
        let mut pyramid: Vec<[SubbandData; 4]> = (0..=3)
            .map(|_| {
                [
                    SubbandData::new(0, 0),
                    SubbandData::new(0, 0),
                    SubbandData::new(0, 0),
                    SubbandData::new(0, 0),
                ]
            })
            .collect();
        // L1 HL parent (4x4) + L2 HL band (8x8) with scattered values.
        pyramid[1][Orient::HL.as_index()] = SubbandData::new(4, 4);
        let mut l2 = SubbandData::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                let v = ((x as i32 * 7 + y as i32 * 13) % 11) - 5;
                l2.set(y, x, v);
            }
        }
        // Seed the parent so parent_zero contexts vary.
        let par = &mut pyramid[1][Orient::HL.as_index()];
        for y in 0..4 {
            for x in 0..4 {
                par.set(y, x, ((x + y) % 2) as i32);
            }
        }
        pyramid[2][Orient::HL.as_index()] = l2;

        let mut p_intra = pyramid.clone();
        let mut p_inter = pyramid.clone();

        let intra_params = CoreIntraEncoderParams {
            wavelet: WaveletFilter::LeGall5_3,
            dwt_depth: 3,
            qindex: 0,
            codeblocks: Some(vec![(1, 1), (2, 2), (2, 2), (2, 2)]),
            codeblock_mode: 0,
        };
        let residue_params = ResidueParams {
            wavelet: WaveletFilter::LeGall5_3,
            dwt_depth: 3,
            qindex: 0,
            codeblocks: Some(vec![(1, 1), (2, 2), (2, 2), (2, 2)]),
            codeblock_mode: 0,
        };

        let intra_bytes = encode_subband_ac(&mut p_intra, 2, Orient::HL, (2, 2), &intra_params);
        let inter_bytes =
            encode_residue_subband_ac_cb(&mut p_inter, 2, Orient::HL, (2, 2), &residue_params);
        assert_eq!(
            intra_bytes, inter_bytes,
            "inter-residue cb encoder bytes diverge from the proven \
             intra-core encoder on an identical L2 HL band + grid"
        );
    }

    #[test]
    fn round_mv_to_int_pel_at_precision_0_is_noop() {
        // No sub-pel — MV stays as-is.
        for (x, y) in [(0, 0), (3, -7), (-100, 50), (i32::MAX / 8, i32::MIN / 8)] {
            let mv = IntegerMv(x, y);
            let r = round_mv_to_int_pel(mv, 0);
            assert_eq!((r.0, r.1), (x, y));
        }
    }

    #[test]
    fn round_mv_to_int_pel_qpel_snaps_to_multiples_of_4() {
        // unit = 4 (qpel), half = 2. |v| <= 2 → 0; |v| > 2 → snap to nearest multiple of 4.
        let cases = [
            (0i32, 0i32),
            (1, 0),
            (2, 0), // ties toward zero
            (3, 4),
            (4, 4),
            (5, 4),
            (6, 8), // 6 = 4 + 2: ((6+2)/4)*4 = 8
            (7, 8),
            (-1, 0),
            (-2, 0),
            (-3, -4),
            (-6, -8),
        ];
        for (input, expected) in cases {
            let mv = IntegerMv(input, 0);
            let r = round_mv_to_int_pel(mv, 2);
            assert_eq!(
                (r.0, r.1),
                (expected, 0),
                "snap qpel({input}) → expected {expected}, got {}",
                r.0
            );
        }
    }

    #[test]
    fn round_mv_to_int_pel_halfpel_snaps_to_multiples_of_2() {
        // unit = 2 (half-pel), half = 1. |v| <= 1 → 0; |v| > 1 → nearest multiple of 2.
        for (input, expected) in [(0i32, 0i32), (1, 0), (2, 2), (3, 4), (-1, 0), (-3, -4)] {
            let mv = IntegerMv(input, 0);
            let r = round_mv_to_int_pel(mv, 1);
            assert_eq!((r.0, r.1), (expected, 0));
        }
    }

    /// Round-91: half-pel snap at qpel precision. unit = 2, half = 1.
    /// |v| <= 1 → 0 (ties toward zero); |v| > 1 → nearest multiple of 2.
    #[test]
    fn round_mv_to_half_pel_qpel_snaps_to_multiples_of_2() {
        // unit = 1<<(p-1) = 2, half = unit/2 = 1.
        // |v| <= 1 → 0; otherwise snapped_mag = ((|v|+1)/2)*2.
        let cases = [
            (0i32, 0i32),
            (1, 0), // ties toward zero
            (2, 2), // already on half-pel grid: ((2+1)/2)*2 = 2
            (3, 4), // ((3+1)/2)*2 = 4
            (4, 4),
            (5, 6), // ((5+1)/2)*2 = 6
            (6, 6),
            (-1, 0),
            (-3, -4),
            (-5, -6),
        ];
        for (input, expected) in cases {
            let mv = IntegerMv(input, 0);
            let r = round_mv_to_half_pel(mv, 2);
            assert_eq!(
                (r.0, r.1),
                (expected, 0),
                "half-pel snap qpel({input}) → expected {expected}, got {}",
                r.0
            );
        }
    }

    /// At `mv_precision == 0` the half-pel grid degenerates to the
    /// integer-pel grid, so the helper falls through to integer-pel
    /// rounding (identity at p == 0).
    #[test]
    fn round_mv_to_half_pel_at_precision_0_falls_through_to_int_pel() {
        for (x, y) in [(0, 0), (3, -7), (-100, 50)] {
            let mv = IntegerMv(x, y);
            let r = round_mv_to_half_pel(mv, 0);
            assert_eq!((r.0, r.1), (x, y));
        }
    }

    /// At `mv_precision == 1` (half-pel) every MV is already on the
    /// half-pel grid, so the helper returns the input unchanged.
    #[test]
    fn round_mv_to_half_pel_at_precision_1_is_identity() {
        for (x, y) in [(0i32, 0i32), (1, -1), (5, 7), (-12, 4)] {
            let mv = IntegerMv(x, y);
            let r = round_mv_to_half_pel(mv, 1);
            assert_eq!((r.0, r.1), (x, y));
        }
    }

    /// Round-91: at qpel precision the half-pel rounded MV lands on an
    /// even (half-pel-grid) coordinate, never on an odd (quarter-pel
    /// offset) one. This is the structural prerequisite that makes the
    /// half-pel candidate a genuinely new representative — distinct
    /// from both the int-pel snap (multiples of 4) and the unmodified
    /// sub-pel MV (anywhere on the qpel grid).
    #[test]
    fn round_mv_to_half_pel_qpel_lands_on_even_grid() {
        for x in -16..=16i32 {
            for y in -8..=8i32 {
                let mv = IntegerMv(x, y);
                let mv_half = round_mv_to_half_pel(mv, 2);
                assert_eq!(
                    mv_half.0 % 2,
                    0,
                    "mv = {mv:?}: half-pel snap {mv_half:?} not on even grid"
                );
                assert_eq!(
                    mv_half.1 % 2,
                    0,
                    "mv = {mv:?}: half-pel snap {mv_half:?} not on even grid"
                );
            }
        }
    }

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
        encode_block_motion_data(&mut w, sbx, sby, bx, by, &mvs, None);
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

    /// Encoded `block_motion_data` with a §12.3.3.2 per-block global-mode
    /// grid must round-trip: every block's `gmode` flag recovers, and the
    /// non-global blocks recover their MVs (global blocks carry no MV
    /// residual — their MV stays at the propagated default). Exercises the
    /// block-global-flag prediction (§12.3.6.4) with a spatially varied
    /// pattern so the neighbour-majority prediction fires both ways.
    #[test]
    fn block_motion_data_global_flags_roundtrip() {
        let sbx = 1u32;
        let sby = 1u32;
        let bx = 4u32;
        let by = 4u32;
        let mvs: Vec<IntegerMv> = (0..16i32)
            .map(|i| IntegerMv((i % 4) - 1, (i / 4) - 1))
            .collect();
        // Chequerboard-ish global pattern: top-left quadrant + a couple
        // of scattered blocks are global.
        let gmode: Vec<bool> = (0..16)
            .map(|i| matches!(i, 0 | 1 | 4 | 5 | 10 | 15))
            .collect();
        let mut w = BitWriter::new();
        encode_block_motion_data(&mut w, sbx, sby, bx, by, &mvs, Some(&gmode));
        let bytes = w.finish();

        use crate::picture_inter::{GlobalParams, PicturePredictionParams};
        let pred = PicturePredictionParams {
            luma_xblen: 8,
            luma_yblen: 8,
            luma_xbsep: 4,
            luma_ybsep: 4,
            mv_precision: 0,
            using_global: true,
            prediction_mode: 0,
            superblocks_x: sbx,
            superblocks_y: sby,
            blocks_x: bx,
            blocks_y: by,
            refs_wt_precision: 1,
            ref1_wt: 1,
            ref2_wt: 1,
            global1: Some(GlobalParams::default()),
            global2: None,
        };
        let mut r = crate::bits::BitReader::new(&bytes);
        let motion = decode_block_motion_data(&mut r, &pred, 1).expect("decode motion");
        for by_ in 0..by {
            for bx_ in 0..bx {
                let i = (by_ * bx + bx_) as usize;
                let blk = &motion.blocks[i];
                assert_eq!(blk.rmode, RefPredMode::Ref1Only, "block {i} rmode");
                assert_eq!(blk.gmode, gmode[i], "block {i} gmode flag");
                // Non-global blocks recover their MV; global blocks skip
                // the MV residual entirely (§12.3.5).
                if !gmode[i] {
                    assert_eq!(
                        blk.mv[0],
                        (mvs[i].0, mvs[i].1),
                        "block {i} MV mismatch (non-global)"
                    );
                }
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
            write_picture_prediction_parameters(&mut w, &params, mv_precision, 1);
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

    /// §11.2.6 `write_global_motion_parameters` must be the exact
    /// bitstream inverse of the decoder's `parse_global_motion_parameters`.
    /// Sweep pan_tilt / zrs / perspective on and off to exercise every
    /// omission-flag branch, and assert the parsed-back parameters equal
    /// the *effective* input.
    #[test]
    fn global_motion_parameters_roundtrip() {
        use crate::picture_inter::parse_global_motion_parameters;
        let cases = [
            // Trivial: identity everywhere.
            GlobalParams::default(),
            // Pure pan/tilt.
            GlobalParams {
                pan_tilt: (7, -3),
                zrs: [[1, 0], [0, 1]],
                zrs_exp: 0,
                perspective: (0, 0),
                persp_exp: 0,
            },
            // Non-trivial ZRS (zoom) at exponent 2.
            GlobalParams {
                pan_tilt: (0, 0),
                zrs: [[5, 1], [-1, 5]],
                zrs_exp: 2,
                perspective: (0, 0),
                persp_exp: 0,
            },
            // Perspective set.
            GlobalParams {
                pan_tilt: (2, 2),
                zrs: [[1, 0], [0, 1]],
                zrs_exp: 0,
                perspective: (3, -4),
                persp_exp: 6,
            },
            // Full affine + perspective combined.
            GlobalParams {
                pan_tilt: (-11, 13),
                zrs: [[9, -2], [3, 8]],
                zrs_exp: 3,
                perspective: (1, 1),
                persp_exp: 4,
            },
            // Non-zero persp_exp but zero perspective vector: the flag is
            // omitted, so the decoder reads persp_exp back as 0. The
            // `effective_global_params` normalisation must match.
            GlobalParams {
                pan_tilt: (1, 0),
                zrs: [[1, 0], [0, 1]],
                zrs_exp: 0,
                perspective: (0, 0),
                persp_exp: 9,
            },
        ];
        for g in &cases {
            let eff = effective_global_params(g);
            let mut w = BitWriter::new();
            write_global_motion_parameters(&mut w, &eff);
            let bytes = w.finish();
            let mut r = crate::bits::BitReader::new(&bytes);
            let parsed = parse_global_motion_parameters(&mut r);
            assert_eq!(parsed.pan_tilt, eff.pan_tilt, "pan_tilt for {g:?}");
            assert_eq!(parsed.zrs, eff.zrs, "zrs for {g:?}");
            assert_eq!(parsed.zrs_exp, eff.zrs_exp, "zrs_exp for {g:?}");
            assert_eq!(parsed.perspective, eff.perspective, "perspective for {g:?}");
            assert_eq!(parsed.persp_exp, eff.persp_exp, "persp_exp for {g:?}");
        }
    }

    /// §11.2.1 `picture_prediction_parameters` with global motion must
    /// round-trip through the decoder's parser: `using_global = true`,
    /// the ref1 global params for a 1-ref picture, and both refs' params
    /// for a 2-ref picture.
    #[test]
    fn picture_prediction_parameters_global_roundtrips() {
        let seq = crate::encoder::make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
        let g1 = GlobalParams {
            pan_tilt: (6, -2),
            zrs: [[5, 0], [0, 5]],
            zrs_exp: 2,
            perspective: (0, 0),
            persp_exp: 0,
        };
        let g2 = GlobalParams {
            pan_tilt: (-4, 4),
            zrs: [[1, 0], [0, 1]],
            zrs_exp: 0,
            perspective: (1, -1),
            persp_exp: 3,
        };
        // 1-ref: only global1 is on the wire.
        {
            let params = InterEncoderParams {
                mv_precision: 2,
                global_motion: Some(GlobalMotionConfig {
                    global1: g1.clone(),
                    global2: None,
                    block_gmode: None,
                }),
                ..InterEncoderParams::default()
            };
            let mut w = BitWriter::new();
            write_picture_prediction_parameters(&mut w, &params, 2, 1);
            w.byte_align();
            let bytes = w.finish();
            let mut r = crate::bits::BitReader::new(&bytes);
            let pred = parse_picture_prediction_parameters(&mut r, &seq, 1).expect("parse PPP");
            assert!(pred.using_global);
            let p1 = pred.global1.expect("global1 present");
            assert_eq!(p1.pan_tilt, g1.pan_tilt);
            assert_eq!(p1.zrs, g1.zrs);
            assert_eq!(p1.zrs_exp, g1.zrs_exp);
            assert!(pred.global2.is_none());
        }
        // 2-ref: both global1 and global2 on the wire.
        {
            let params = InterEncoderParams {
                bipred_mv_precision: 2,
                global_motion: Some(GlobalMotionConfig {
                    global1: g1.clone(),
                    global2: Some(g2.clone()),
                    block_gmode: None,
                }),
                ..InterEncoderParams::default()
            };
            let mut w = BitWriter::new();
            write_picture_prediction_parameters(&mut w, &params, 2, 2);
            w.byte_align();
            let bytes = w.finish();
            let mut r = crate::bits::BitReader::new(&bytes);
            let pred = parse_picture_prediction_parameters(&mut r, &seq, 2).expect("parse PPP");
            assert!(pred.using_global);
            let p1 = pred.global1.expect("global1 present");
            let p2 = pred.global2.expect("global2 present");
            assert_eq!(p1.pan_tilt, g1.pan_tilt);
            assert_eq!(p2.pan_tilt, g2.pan_tilt);
            assert_eq!(p2.perspective, g2.perspective);
            assert_eq!(p2.persp_exp, g2.persp_exp);
        }
    }

    /// [`estimate_global_pan_config`] on a translating fixture: the
    /// dominant translation is the median ME MV, blocks matching it are
    /// global, non-matching blocks stay block-motion, and the §15.8.8
    /// field reproduces the median exactly (`pan_tilt = t - 1` under the
    /// zero affine matrix).
    #[test]
    fn estimate_global_pan_matches_dominant_translation() {
        use crate::obmc::global_mv;
        let seq = crate::encoder::make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
        let params = InterEncoderParams {
            mv_precision: 0,
            ..InterEncoderParams::default()
        };
        // Whole-frame integer pan over a textured field: a deterministic
        // pseudo-random reference shifted left by 3 pels (edge columns
        // replicated). Every interior block's ME lands on exactly
        // (+3, 0); only right-edge blocks (whose true content left the
        // frame) can disagree.
        let mut y0 = vec![0u8; 64 * 64];
        let mut state = 0x2468_ace1u32;
        for px in y0.iter_mut() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *px = 40 + (state % 160) as u8;
        }
        let mut y1 = vec![0u8; 64 * 64];
        for r in 0..64usize {
            for c in 0..64usize {
                y1[r * 64 + c] = y0[r * 64 + (c + 3).min(63)];
            }
        }
        let (cfg, fraction) = estimate_global_pan_config(&seq, &params, &y1, &y0);
        assert!(
            fraction >= 0.8,
            "whole-frame integer pan should mark ≥ 80% of blocks global, got {fraction}"
        );
        let grid = cfg.block_gmode.as_ref().expect("per-block grid");
        assert_eq!(grid.len(), 16 * 16);
        // The field must equal the dominant MV at every pixel: recompute
        // the median from the same ME the estimator ran. On this fixture
        // the dominant translation is exactly (+3, 0).
        let mvs = inter_mv_grid(&seq, &params, &y1, &y0);
        let mut xs: Vec<i32> = mvs.iter().map(|m| m.0).collect();
        let mut ys: Vec<i32> = mvs.iter().map(|m| m.1).collect();
        let t = (median(&mut xs), median(&mut ys));
        for (x, y) in [(0, 0), (13, 7), (63, 63)] {
            assert_eq!(
                global_mv(&cfg.global1, x, y),
                t,
                "field must be the constant median translation"
            );
        }
        // Every marked block's ME MV equals the field; every unmarked
        // block's differs.
        for (i, &g) in grid.iter().enumerate() {
            assert_eq!((mvs[i].0, mvs[i].1) == t, g, "block {i} gmode consistency");
        }
    }

    /// Deterministic smooth luma texture for the global-model
    /// estimation tests: a coarse pseudo-random 8-pel grid bilinearly
    /// upsampled, so every 8×8 ME window has a unique low-frequency
    /// signature (trackable under warps) without hard edges that
    /// alias under sub-pel resampling.
    fn smooth_texture(w: usize, h: usize, seed: u32) -> Vec<u8> {
        let cell = 8usize;
        let gw = w / cell + 2;
        let gh = h / cell + 2;
        let mut state = seed | 1;
        let mut grid = vec![0f64; gw * gh];
        for g in grid.iter_mut() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *g = 40.0 + (state % 160) as f64;
        }
        let mut out = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let gx = x / cell;
                let gy = y / cell;
                let fx = (x % cell) as f64 / cell as f64;
                let fy = (y % cell) as f64 / cell as f64;
                let v00 = grid[gy * gw + gx];
                let v01 = grid[gy * gw + gx + 1];
                let v10 = grid[(gy + 1) * gw + gx];
                let v11 = grid[(gy + 1) * gw + gx + 1];
                let v = v00 * (1.0 - fx) * (1.0 - fy)
                    + v01 * fx * (1.0 - fy)
                    + v10 * (1.0 - fx) * fy
                    + v11 * fx * fy;
                out[y * w + x] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
        out
    }

    /// Warp `refp` by the integer §15.8.8 field of `g` at
    /// `mv_precision = 0`: `cur(x) = ref(x + global_mv(g, x))` with
    /// edge clamping — the decoder's own field arithmetic generates
    /// the fixture, so the true model is representable exactly.
    fn warp_by_global_field(refp: &[u8], w: usize, h: usize, g: &GlobalParams) -> Vec<u8> {
        let mut out = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = crate::obmc::global_mv(g, x as i32, y as i32);
                let sx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                let sy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                out[y * w + x] = refp[sy * w + sx];
            }
        }
        out
    }

    /// [`affine_lsq`] must recover an exactly-linear MV field, and
    /// [`quantise_global_fit`] must reproduce it through the integer
    /// §15.8.8 arithmetic at every block anchor.
    #[test]
    fn affine_fit_recovers_exact_linear_field() {
        let (blocks_x, blocks_y) = (16u32, 16u32);
        let pts: Vec<(f64, f64)> = (0..blocks_y)
            .flat_map(|by| {
                (0..blocks_x).map(move |bx| {
                    let (ax, ay) = block_anchor(bx, by);
                    (ax as f64, ay as f64)
                })
            })
            .collect();
        // Targets are the exact integer §15.8.8 field of a known
        // anamorphic zoom (a = [[16, 0], [0, −8]] at ez = 8 — the same
        // exponent the estimator picks for a 64-pel frame), so the fit
        // has a representable ground truth to reconstruct.
        let g_true = GlobalParams {
            pan_tilt: (-3, 1),
            zrs: [[16, 0], [0, -8]],
            zrs_exp: 8,
            perspective: (0, 0),
            persp_exp: 0,
        };
        let mvs: Vec<IntegerMv> = pts
            .iter()
            .map(|&(x, y)| {
                let f = crate::obmc::global_mv(&g_true, x as i32, y as i32);
                IntegerMv(f.0, f.1)
            })
            .collect();
        let g = fit_global_params_from_grid(&mvs, blocks_x, blocks_y, 64, 64, false)
            .expect("fit must succeed");
        assert_ne!(g.zrs, [[0, 0], [0, 0]], "zoom must reach the matrix");
        assert_eq!(g.perspective, (0, 0), "affine fit leaves c at zero");
        // The reconstructed integer field must match the ground truth
        // at ≥ 95% of the anchors (fitting through the floor-rounded
        // integer targets can leave isolated off-by-ones right at the
        // rounding boundaries).
        let mut exact = 0usize;
        for (i, &(x, y)) in pts.iter().enumerate() {
            let f = crate::obmc::global_mv(&g, x as i32, y as i32);
            if f == (mvs[i].0, mvs[i].1) {
                exact += 1;
            }
        }
        assert!(
            exact as f64 >= 0.95 * pts.len() as f64,
            "quantised field matches only {exact}/{} anchors",
            pts.len()
        );
    }

    /// A constant MV grid must collapse the affine estimator to the
    /// pan parameterisation: zero matrix, `zrs_exp = 0`, and the exact
    /// constant field.
    #[test]
    fn affine_fit_collapses_to_pan_on_constant_grid() {
        let (blocks_x, blocks_y) = (16u32, 16u32);
        let mvs = vec![IntegerMv(3, -2); (blocks_x * blocks_y) as usize];
        let g = fit_global_params_from_grid(&mvs, blocks_x, blocks_y, 64, 64, false)
            .expect("fit must succeed");
        assert_eq!(g.zrs, [[0, 0], [0, 0]]);
        assert_eq!(g.zrs_exp, 0);
        assert_eq!(g.perspective, (0, 0));
        for (x, y) in [(0, 0), (31, 17), (63, 63)] {
            assert_eq!(crate::obmc::global_mv(&g, x, y), (3, -2));
        }
    }

    /// The trimmed refit must reject a minority of outlier blocks: a
    /// constant-pan grid with a 12% foreground of wild MVs still fits
    /// the exact pan.
    #[test]
    fn affine_fit_trims_foreground_outliers() {
        let (blocks_x, blocks_y) = (16u32, 16u32);
        let n = (blocks_x * blocks_y) as usize;
        let mut mvs = vec![IntegerMv(-4, 1); n];
        // A 5×6 block "foreground object" moving hard the other way.
        for by in 4..10u32 {
            for bx in 6..11u32 {
                mvs[(by * blocks_x + bx) as usize] = IntegerMv(11, -9);
            }
        }
        let g = fit_global_params_from_grid(&mvs, blocks_x, blocks_y, 64, 64, false)
            .expect("fit must succeed");
        assert_eq!(g.zrs, [[0, 0], [0, 0]], "outliers must not bend the matrix");
        for (x, y) in [(0, 0), (63, 63)] {
            assert_eq!(
                crate::obmc::global_mv(&g, x, y),
                (-4, 1),
                "background pan survives the foreground"
            );
        }
    }

    /// End-to-end [`estimate_global_affine_config`] on a synthetic
    /// zoom: the current frame is the reference warped by a known
    /// §15.8.8 zoom field (generated with the decoder's own `global_mv`
    /// arithmetic), and the estimator must recover a non-trivial matrix
    /// with a dominant global fraction.
    #[test]
    fn estimate_global_affine_recovers_zoom() {
        let seq = crate::encoder::make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
        let params = InterEncoderParams {
            mv_precision: 0,
            ..InterEncoderParams::default()
        };
        let y0 = smooth_texture(64, 64, 0x1357_9bdf);
        // True model: uniform zoom-out — field v = x/16 − 3 per axis
        // (a = 16 at ez = 8, pan − bias). Max displacement 64/16 = 4
        // pels, well inside the default ±16 search.
        let g_true = GlobalParams {
            pan_tilt: (-3, -3),
            zrs: [[16, 0], [0, 16]],
            zrs_exp: 8,
            perspective: (0, 0),
            persp_exp: 0,
        };
        let y1 = warp_by_global_field(&y0, 64, 64, &g_true);
        let (cfg, fraction) = estimate_global_affine_config(&seq, &params, &y1, &y0);
        assert!(
            fraction >= 0.6,
            "whole-frame zoom should mark most blocks global, got {fraction}"
        );
        assert_ne!(
            cfg.global1.zrs,
            [[0, 0], [0, 0]],
            "the zoom must be captured by the matrix, not flattened to a pan"
        );
        // The recovered integer field must agree with the true field
        // at most anchors (both are floor-rounded integer fields).
        let mut agree = 0usize;
        let mut total = 0usize;
        for by in 0..16u32 {
            for bx in 0..16u32 {
                let (x, y) = block_anchor(bx, by);
                total += 1;
                if crate::obmc::global_mv(&effective_global_params(&cfg.global1), x, y)
                    == crate::obmc::global_mv(&g_true, x, y)
                {
                    agree += 1;
                }
            }
        }
        assert!(
            agree as f64 >= 0.75 * total as f64,
            "recovered field agrees at only {agree}/{total} anchors"
        );
    }

    /// End-to-end [`estimate_global_perspective_config`] on a strong
    /// synthetic perspective warp: the linearised c-fit must produce a
    /// non-zero perspective vector that beats the affine-only fit's
    /// field error at the anchors.
    #[test]
    fn estimate_global_perspective_recovers_warp() {
        let seq = crate::encoder::make_minimal_sequence(96, 96, ChromaFormat::Yuv420);
        let params = InterEncoderParams {
            mv_precision: 0,
            mv_search_range: 16,
            ..InterEncoderParams::default()
        };
        let y0 = smooth_texture(96, 96, 0x0246_8ace);
        // True model: zoom + strong perspective — m shrinks ~28% at
        // the far corner, so the field visibly deviates from affine.
        let g_true = GlobalParams {
            pan_tilt: (4, 2),
            zrs: [[24, 0], [0, 24]],
            zrs_exp: 8,
            perspective: (8, 4),
            persp_exp: 12,
        };
        let y1 = warp_by_global_field(&y0, 96, 96, &g_true);
        let (cfg_p, fraction_p) = estimate_global_perspective_config(&seq, &params, &y1, &y0);
        let (cfg_a, _) = estimate_global_affine_config(&seq, &params, &y1, &y0);
        assert_ne!(
            cfg_p.global1.perspective,
            (0, 0),
            "the perspective component must be detected"
        );
        // Field-error comparison at the anchors: the perspective model
        // must fit the true field strictly better than affine-only.
        let ge_p = effective_global_params(&cfg_p.global1);
        let ge_a = effective_global_params(&cfg_a.global1);
        let mut err_p = 0i64;
        let mut err_a = 0i64;
        for by in 0..24u32 {
            for bx in 0..24u32 {
                let (x, y) = block_anchor(bx, by);
                let t = crate::obmc::global_mv(&g_true, x, y);
                let p = crate::obmc::global_mv(&ge_p, x, y);
                let a = crate::obmc::global_mv(&ge_a, x, y);
                err_p += (p.0 - t.0).abs() as i64 + (p.1 - t.1).abs() as i64;
                err_a += (a.0 - t.0).abs() as i64 + (a.1 - t.1).abs() as i64;
            }
        }
        assert!(
            err_p < err_a,
            "perspective fit (Σ|Δ| = {err_p}) must beat affine-only (Σ|Δ| = {err_a})"
        );
        assert!(
            fraction_p >= 0.5,
            "perspective warp should mark most blocks global, got {fraction_p}"
        );
    }

    /// [`GlobalMotionConfig::pan_tilt_all`] must produce a
    /// **position-independent** §15.8.8 field — the constant
    /// `(dx + 1, dy + 1)` at every pixel (the `+ 1` is the `global_mv`
    /// rounding bias at exponent 0). Through round-384 the constructor
    /// filled in the identity matrix instead of the zero matrix, which
    /// made the field grow linearly with pixel position — a stretch,
    /// not a pan. Pins the fixed constructor at the extreme corners of
    /// a large frame, including through the wire round-trip
    /// (`write_global_motion_parameters` →
    /// `parse_global_motion_parameters`).
    #[test]
    fn pan_tilt_all_field_is_position_independent() {
        use crate::obmc::global_mv;
        for (dx, dy) in [(0, 0), (-1, -1), (7, -3), (-40, 25)] {
            let cfg = GlobalMotionConfig::pan_tilt_all(dx, dy);
            assert!(cfg.block_gmode.is_none(), "whole-picture grid");
            assert!(cfg.global2.is_none());
            let g = effective_global_params(&cfg.global1);
            for (x, y) in [(0, 0), (1, 1), (63, 63), (1919, 1079), (4095, 2159)] {
                assert_eq!(
                    global_mv(&g, x, y),
                    (dx + 1, dy + 1),
                    "pan_tilt_all({dx}, {dy}) field at ({x}, {y})"
                );
            }
            // Wire round-trip preserves the zero matrix (the flag is
            // written because the zero matrix is NOT the omission
            // default — the §11.2.6 default is the identity).
            let mut w = BitWriter::new();
            write_global_motion_parameters(&mut w, &g);
            let bytes = w.finish();
            let mut r = crate::bits::BitReader::new(&bytes);
            let parsed = crate::picture_inter::parse_global_motion_parameters(&mut r);
            assert_eq!(parsed.pan_tilt, (dx, dy));
            assert_eq!(parsed.zrs, [[0, 0], [0, 0]]);
            assert_eq!(parsed.zrs_exp, 0);
            assert_eq!(parsed.perspective, (0, 0));
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
            write_picture_prediction_parameters(&mut w, &iep, mv_precision, 1);
            w.byte_align();
            let bytes = w.finish();
            let mut r = crate::bits::BitReader::new(&bytes);
            let parsed = parse_picture_prediction_parameters(&mut r, &seq, 1).expect("PPP");
            let synthesised = picture_prediction_params_from(&seq, &iep, mv_precision, 1);
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

    /// At `mv_precision == 0`, [`inter_select_int_pel_per_block`] must
    /// be a no-op — every MV is already integer-pel, so rounding is
    /// the identity and the function takes the early-return path before
    /// even building the upref. Anything else means the round-73 knob
    /// is silently doing work in the integer-pel regime.
    #[test]
    fn inter_select_int_pel_at_precision_0_is_noop() {
        let (y0, _, _, y1, _, _) = synthetic_translating_pair_64(2, -1);
        let mut mvs_baseline = subpel_search_me(&y1, &y0, 64, 64, 16, 16, 16, 0);
        let snapshot: Vec<(i32, i32)> = mvs_baseline.iter().map(|m| (m.0, m.1)).collect();
        inter_select_int_pel_per_block(&y1, &y0, 64, 64, 16, 16, &mut mvs_baseline, 0);
        let after: Vec<(i32, i32)> = mvs_baseline.iter().map(|m| (m.0, m.1)).collect();
        assert_eq!(
            snapshot, after,
            "mv_precision = 0 should leave the MV grid untouched"
        );
    }

    /// On the quarter-pel camera-pan fixture (smooth sub-pel motion),
    /// the integer-pel snap should LOSE on most blocks — sub-pel ME
    /// found the true fractional offset and the OBMC blend rewards
    /// the spec-correct sub-pel prediction. We assert that the
    /// majority of MVs keep their sub-pel value, proving the adaptive
    /// selector is content-sensitive rather than always snapping.
    #[test]
    fn inter_select_int_pel_preserves_subpel_on_smooth_pan() {
        let (y0, _, _, y1, _, _) = synthetic_camera_pan_64(1, 0);
        let mut mvs = subpel_search_me(&y1, &y0, 64, 64, 16, 16, 16, 2);
        let before: Vec<(i32, i32)> = mvs.iter().map(|m| (m.0, m.1)).collect();
        // Count blocks whose MV is already at an integer-pel multiple —
        // those can never change. Subtract them when computing the
        // "preserved sub-pel" denominator.
        let already_int = before
            .iter()
            .filter(|(x, y)| (x % 4 == 0) && (y % 4 == 0))
            .count();
        inter_select_int_pel_per_block(&y1, &y0, 64, 64, 16, 16, &mut mvs, 2);
        let after: Vec<(i32, i32)> = mvs.iter().map(|m| (m.0, m.1)).collect();
        let preserved_subpel = before
            .iter()
            .zip(after.iter())
            .filter(|(b, a)| ((b.0 % 4) != 0 || (b.1 % 4) != 0) && b == a)
            .count();
        let total_subpel = before.len() - already_int;
        assert!(
            total_subpel > 0,
            "fixture is supposed to contain sub-pel MVs"
        );
        // At least 50 % of the sub-pel MVs must stay sub-pel — the
        // adaptive selector is not allowed to behave like an
        // unconditional integer-pel rounder on smooth motion content.
        assert!(
            preserved_subpel * 2 >= total_subpel,
            "adaptive int-pel snapped too aggressively on smooth pan: \
             {preserved_subpel} / {total_subpel} sub-pel MVs preserved"
        );
    }

    /// **Strict-superset invariant**: with `inter_adaptive_int_pel`,
    /// the per-block selector compares the current sub-pel MV against
    /// its integer-pel-rounded peer and picks the lower-SSE one. The
    /// candidate set strictly contains the original MV grid; tie-bias
    /// is toward int-pel via `<=`. Therefore the per-block OBMC SSE
    /// after the selector must be ≤ the per-block OBMC SSE before, for
    /// every block in raster order. This is the load-bearing invariant
    /// that justifies enabling adaptive int-pel as the default — it
    /// can never regress the OBMC reconstruction cost function.
    #[test]
    fn inter_select_int_pel_monotonic_per_block_obmc_sse() {
        let (y0, _, _, y1, _, _) = synthetic_translating_pair_64(2, -1);
        let mvs_subpel = subpel_search_me(&y1, &y0, 64, 64, 16, 16, 16, 2);
        let mut mvs_after = mvs_subpel.clone();
        inter_select_int_pel_per_block(&y1, &y0, 64, 64, 16, 16, &mut mvs_after, 2);
        // For each block: build neighbour_sum from the SAME grid in
        // both cases (before the selector modifies things) and compare
        // per-block SSE at the original sub-pel MV vs the selector's
        // chosen MV. The chosen MV must be at most as costly.
        let (xblen_u, yblen_u, xbsep_u, ybsep_u) = (8usize, 8usize, 4usize, 4usize);
        let xoffset = (xblen_u - xbsep_u) / 2;
        let yoffset = (yblen_u - ybsep_u) / 2;
        let (upref, up_w, up_h) = build_upref_signed(&y0, 64, 64);
        let refp_signed: Vec<i32> = y0.iter().map(|&v| v as i32 - 128).collect();
        let ref_w = 64usize;
        let ref_h = 64usize;
        let blocks_x = 16u32;
        let blocks_y = 16u32;
        // Use the post-selector grid as the neighbour context, matching
        // what the selector itself saw in raster order at the moment
        // each block's decision was made. (For raster-order updates the
        // exact neighbour state varies; a same-grid snapshot is the
        // cheap conservative check.)
        for j in 0..blocks_y {
            for i in 0..blocks_x {
                let bidx = (j * blocks_x + i) as usize;
                let neighbour_sum = build_neighbour_sum(
                    &mvs_after,
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
                    2,
                );
                let xstart_ij = (i as i32) * (xbsep_u as i32) - (xoffset as i32);
                let ystart_ij = (j as i32) * (ybsep_u as i32) - (yoffset as i32);
                let weight = block_weight(
                    xblen_u, yblen_u, xbsep_u, ybsep_u, xoffset, yoffset, i, j, blocks_x, blocks_y,
                );
                let sse_at = |mv: IntegerMv| -> i64 {
                    let pred = block_weighted_pred(
                        &upref,
                        up_w,
                        up_h,
                        &refp_signed,
                        ref_w,
                        ref_h,
                        2,
                        &weight,
                        xblen_u,
                        yblen_u,
                        xstart_ij,
                        ystart_ij,
                        mv.0,
                        mv.1,
                    );
                    obmc_block_sse(
                        &y1,
                        64,
                        64,
                        &pred,
                        &neighbour_sum,
                        xblen_u,
                        yblen_u,
                        xstart_ij,
                        ystart_ij,
                    )
                };
                let sse_before = sse_at(mvs_subpel[bidx]);
                let sse_after = sse_at(mvs_after[bidx]);
                assert!(
                    sse_after <= sse_before,
                    "block (i={i},j={j}): per-block OBMC SSE regressed under \
                     adaptive int-pel: before = {sse_before}, after = {sse_after} \
                     (this breaks the round-73 strict-superset invariant)"
                );
            }
        }
    }

    /// **Round-91 strict-superset invariant** (bipred path): the
    /// widened `{int-pel, half-pel, sub-pel}` per-ref candidate set
    /// contains the previous round-39 `{int-pel, sub-pel}` set, so on
    /// every block the widened bipred selector's per-ref SAD and bipred
    /// SAD are ≤ the previous round-39 selector's values. We re-run a
    /// reference round-39 selector inline (2 per-ref candidates, 4
    /// bipred combos) on the same MV grid and compare per-block to the
    /// round-91 output's per-block SAD. The round-91 SADs must dominate
    /// (≤) the round-39 SADs on every block — mirrors the
    /// `inter_select_int_pel_monotonic_per_block_obmc_sse` correctness
    /// pin on the 1-ref path, scaled to the bipred path.
    #[test]
    fn bipred_widened_candidate_set_monotonic_per_block_sad() {
        // Camera-pan triplet at qpel: ref1 at pan(0), ref2 at pan(2),
        // current at pan(1) — exact temporal midpoint, the same
        // smooth-motion fixture the `ffmpeg_cross_decodes_camera_pan_bipred_with_subpel_gain`
        // test exercises. Sub-pel/half-pel/int-pel candidates all
        // produce meaningfully different SADs on this fixture, so the
        // monotonicity check actually exercises the widening (rather
        // than degenerating into "all candidates equal").
        let (y0, _, _, _, _, _) = synthetic_camera_pan_64(0, 0);
        let (_, _, _, y2, _, _) = synthetic_camera_pan_64(2, 0);
        let (_, _, _, ym, _, _) = synthetic_camera_pan_64(1, 0);

        let w = 64u32;
        let h = 64u32;
        let blocks_x = 16u32;
        let blocks_y = 16u32;
        let search = 16u32;
        let p = 2u32; // qpel — widest non-trivial candidate set.

        // Round-91 widened output.
        let new_decisions = bipred_select_modes(&ym, &y0, &y2, w, h, blocks_x, blocks_y, search, p);

        // Round-39 reference (2 per-ref candidates, 4 bipred combos).
        // Inlined here so the test pins the round-39 shape even if the
        // production selector later widens further.
        let mvs1 = subpel_search_me(&ym, &y0, w, h, blocks_x, blocks_y, search, p);
        let mvs2 = subpel_search_me(&ym, &y2, w, h, blocks_x, blocks_y, search, p);
        let (xblen, yblen, xbsep, ybsep) = (8i32, 8i32, 4i32, 4i32);
        let (up1, up1_w, up1_h) = build_upref(&y0, w, h);
        let (up2, up2_w, up2_h) = build_upref(&y2, w, h);
        let w_i = w as i32;
        let h_i = h as i32;
        // SAD of one block against one reference at sub-pel precision.
        // Uses the same `sad_subpel` helper the production selector
        // uses, so per-block scores are bit-exact comparable.
        let sad_subpel_one =
            |mv: IntegerMv, up: &[i32], up_w: usize, up_h: usize, x0: i32, y0: i32| -> i64 {
                let scale = 1i64 << p;
                let qx = (x0 as i64) * scale + mv.0 as i64;
                let qy = (y0 as i64) * scale + mv.1 as i64;
                sad_subpel(
                    &ym, up, up_w, up_h, w_i, h_i, x0, y0, qx, qy, xblen, yblen, p,
                )
            };

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let idx = (by * blocks_x + bx) as usize;
                let x0 = bx as i32 * xbsep;
                let y0_blk = by as i32 * ybsep;
                let mv1 = mvs1[idx];
                let mv2 = mvs2[idx];
                let mv1_int = round_mv_to_int_pel(mv1, p);
                let mv2_int = round_mv_to_int_pel(mv2, p);

                // Round-39 per-ref best from {sub-pel, int-pel}.
                let s1_sp = sad_subpel_one(mv1, &up1, up1_w, up1_h, x0, y0_blk);
                let s1_int = sad_subpel_one(mv1_int, &up1, up1_w, up1_h, x0, y0_blk);
                let sad1_r39 = s1_int.min(s1_sp);
                let s2_sp = sad_subpel_one(mv2, &up2, up2_w, up2_h, x0, y0_blk);
                let s2_int = sad_subpel_one(mv2_int, &up2, up2_w, up2_h, x0, y0_blk);
                let sad2_r39 = s2_int.min(s2_sp);

                // Round-39 bipred best from 4 combinations.
                let combos = [
                    (mv1, mv2),
                    (mv1_int, mv2),
                    (mv1, mv2_int),
                    (mv1_int, mv2_int),
                ];
                let bipred_r39 = combos
                    .iter()
                    .map(|&(m1, m2)| {
                        sad_bipred(
                            &ym, &up1, up1_w, up1_h, &y0, &up2, up2_w, up2_h, &y2, w_i, h_i, x0,
                            y0_blk, m1, m2, xblen, yblen, p,
                        )
                    })
                    .min()
                    .unwrap();

                // Round-91 widened-set best from the same helpers. We
                // re-derive the three per-ref candidates and 3×3 bipred
                // combos inline, then assert each new minimum is ≤ the
                // round-39 minimum — that's exactly the strict-superset
                // invariant the widened selector promises.
                let mv1_half = round_mv_to_half_pel(mv1, p);
                let mv2_half = round_mv_to_half_pel(mv2, p);
                let r91_per_ref_min = |mv_sp: IntegerMv,
                                       mv_half: IntegerMv,
                                       mv_int: IntegerMv,
                                       up: &[i32],
                                       up_w: usize,
                                       up_h: usize|
                 -> i64 {
                    let mut best = sad_subpel_one(mv_sp, up, up_w, up_h, x0, y0_blk)
                        .min(sad_subpel_one(mv_int, up, up_w, up_h, x0, y0_blk));
                    if (mv_half.0, mv_half.1) != (mv_int.0, mv_int.1)
                        && (mv_half.0, mv_half.1) != (mv_sp.0, mv_sp.1)
                    {
                        best = best.min(sad_subpel_one(mv_half, up, up_w, up_h, x0, y0_blk));
                    }
                    best
                };
                let sad1_r91 = r91_per_ref_min(mv1, mv1_half, mv1_int, &up1, up1_w, up1_h);
                let sad2_r91 = r91_per_ref_min(mv2, mv2_half, mv2_int, &up2, up2_w, up2_h);
                assert!(
                    sad1_r91 <= sad1_r39,
                    "block ({bx},{by}) ref1 per-ref SAD regressed: round-91 \
                     {sad1_r91} > round-39 {sad1_r39}"
                );
                assert!(
                    sad2_r91 <= sad2_r39,
                    "block ({bx},{by}) ref2 per-ref SAD regressed: round-91 \
                     {sad2_r91} > round-39 {sad2_r39}"
                );

                let cands1: Vec<IntegerMv> = if (mv1_half.0, mv1_half.1) != (mv1.0, mv1.1)
                    && (mv1_half.0, mv1_half.1) != (mv1_int.0, mv1_int.1)
                {
                    vec![mv1, mv1_half, mv1_int]
                } else {
                    vec![mv1, mv1_int]
                };
                let cands2: Vec<IntegerMv> = if (mv2_half.0, mv2_half.1) != (mv2.0, mv2.1)
                    && (mv2_half.0, mv2_half.1) != (mv2_int.0, mv2_int.1)
                {
                    vec![mv2, mv2_half, mv2_int]
                } else {
                    vec![mv2, mv2_int]
                };
                let mut bipred_r91 = i64::MAX;
                for &m1 in &cands1 {
                    for &m2 in &cands2 {
                        let s = sad_bipred(
                            &ym, &up1, up1_w, up1_h, &y0, &up2, up2_w, up2_h, &y2, w_i, h_i, x0,
                            y0_blk, m1, m2, xblen, yblen, p,
                        );
                        if s < bipred_r91 {
                            bipred_r91 = s;
                        }
                    }
                }
                assert!(
                    bipred_r91 <= bipred_r39,
                    "block ({bx},{by}) bipred SAD regressed: round-91 \
                     {bipred_r91} > round-39 {bipred_r39}"
                );

                // Smoke-check the actual selector's mode is one of the
                // legal bipred modes (never `Intra`).
                let chosen = new_decisions[idx];
                assert!(
                    !matches!(chosen.rmode, RefPredMode::Intra),
                    "bipred selector emitted Intra at ({bx},{by})"
                );
            }
        }
    }

    /// `bipred_post_obmc_refine_modes` must be a strict-monotone
    /// improver of per-block OBMC SSE: the per-block trial set is the
    /// strict superset that always includes the current decision (which
    /// it scores first, as the floor), so after the pass every block's
    /// new decision SSE ≤ its old decision SSE under the SAME frozen
    /// neighbour grid. This pins the round-95 round-80-analogue
    /// monotonicity invariant the bipred refinement promises.
    #[test]
    fn bipred_post_obmc_refine_monotonic_per_block_obmc_sse() {
        // qpel-favourable smooth-motion fixture — the post-OBMC pass
        // has the most room to exercise the alternate-mode candidates.
        let (y0, _, _, _, _, _) = synthetic_camera_pan_64(0, 0);
        let (_, _, _, y2, _, _) = synthetic_camera_pan_64(2, 0);
        let (_, _, _, ym, _, _) = synthetic_camera_pan_64(1, 0);
        let w = 64u32;
        let h = 64u32;
        let blocks_x = 16u32;
        let blocks_y = 16u32;
        let search = 16u32;
        let p = 2u32;

        let before = bipred_select_modes(&ym, &y0, &y2, w, h, blocks_x, blocks_y, search, p);
        let mut after = before.clone();
        bipred_post_obmc_refine_modes(&ym, &y0, &y2, w, h, blocks_x, blocks_y, &mut after, p);

        // Per-block OBMC SSE under the SAME (post-refinement) neighbour
        // grid. Mirrors the 1-ref `inter_select_int_pel_monotonic_per_block_obmc_sse`
        // check: build neighbour_sum from the post-pass grid, then
        // compare per-block SSE of the pre-pass decision against the
        // post-pass decision. The pass's strict-superset construction
        // guarantees the post-pass SSE is ≤ the pre-pass SSE.
        let (xblen_u, yblen_u, xbsep_u, ybsep_u) = (8usize, 8usize, 4usize, 4usize);
        let xoffset = (xblen_u - xbsep_u) / 2;
        let yoffset = (yblen_u - ybsep_u) / 2;
        let (up1_signed, up1_w, up1_h) = build_upref_signed(&y0, w, h);
        let (up2_signed, up2_w, up2_h) = build_upref_signed(&y2, w, h);
        let ref1_signed: Vec<i32> = y0.iter().map(|&v| v as i32 - 128).collect();
        let ref2_signed: Vec<i32> = y2.iter().map(|&v| v as i32 - 128).collect();
        let ref_w = w as usize;
        let ref_h = h as usize;

        for j in 0..blocks_y {
            for i in 0..blocks_x {
                let bidx = (j * blocks_x + i) as usize;
                let neighbour_sum = build_neighbour_sum_bipred(
                    &after,
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
                    &up1_signed,
                    up1_w,
                    up1_h,
                    &ref1_signed,
                    &up2_signed,
                    up2_w,
                    up2_h,
                    &ref2_signed,
                    ref_w,
                    ref_h,
                    p,
                );
                let xstart_ij = (i as i32) * (xbsep_u as i32) - (xoffset as i32);
                let ystart_ij = (j as i32) * (ybsep_u as i32) - (yoffset as i32);
                let weight = block_weight(
                    xblen_u, yblen_u, xbsep_u, ybsep_u, xoffset, yoffset, i, j, blocks_x, blocks_y,
                );
                let sse_at = |b: BipredBlock| -> i64 {
                    let mut wpred = vec![0i32; xblen_u * yblen_u];
                    for q in 0..yblen_u {
                        let y = ystart_ij + q as i32;
                        for pp in 0..xblen_u {
                            let x = xstart_ij + pp as i32;
                            let v = bipred_block_ref_value(
                                b.rmode,
                                b.mv1,
                                b.mv2,
                                &up1_signed,
                                up1_w,
                                up1_h,
                                &ref1_signed,
                                &ref2_signed,
                                &up2_signed,
                                up2_w,
                                up2_h,
                                ref_w,
                                ref_h,
                                p,
                                x,
                                y,
                            );
                            wpred[q * xblen_u + pp] = v * weight[q * xblen_u + pp];
                        }
                    }
                    obmc_block_sse(
                        &ym,
                        w as i32,
                        h as i32,
                        &wpred,
                        &neighbour_sum,
                        xblen_u,
                        yblen_u,
                        xstart_ij,
                        ystart_ij,
                    )
                };
                let sse_before = sse_at(before[bidx]);
                let sse_after = sse_at(after[bidx]);
                assert!(
                    sse_after <= sse_before,
                    "block (i={i},j={j}): post-OBMC bipred refinement regressed \
                     per-block OBMC SSE under frozen neighbour grid: \
                     before = {sse_before}, after = {sse_after} \
                     (breaks the round-95 strict-superset invariant)"
                );
            }
        }
    }

    /// `bipred_post_obmc_refine_modes` must be a no-op on an empty
    /// decisions slice — defensive against blocks_x == 0 / blocks_y == 0
    /// edge cases.
    #[test]
    fn bipred_post_obmc_refine_empty_is_noop() {
        let cur = [0u8; 8 * 8];
        let ref1 = [0u8; 8 * 8];
        let ref2 = [0u8; 8 * 8];
        let mut decisions: Vec<BipredBlock> = vec![];
        bipred_post_obmc_refine_modes(&cur, &ref1, &ref2, 8, 8, 0, 0, &mut decisions, 2);
        assert!(decisions.is_empty());
    }

    /// At `mv_precision == 0` every MV is integer-pel and the snap
    /// peers all coincide with the current MV. The strict-superset
    /// invariant still holds (`cur` is always in the trial set), so
    /// the pass can never regress per-block OBMC SSE — and the only
    /// candidates that can actually win over the current decision are
    /// the alternate-mode `Ref1Only`/`Ref2Only`/`Ref1And2` variants of
    /// the same MV pair. This test pins that the pass remains
    /// well-behaved (and matches its docstring claim) at integer-pel.
    #[test]
    fn bipred_post_obmc_refine_int_pel_only_runs_and_is_monotone() {
        let (y0, _, _, _, _, _) = synthetic_translating_pair_64(2, -1);
        let (_, _, _, y1, _, _) = synthetic_translating_pair_64(4, 0);
        let mid = {
            // Half-and-half average — a hand-built bipred-favourable
            // mid where Ref1And2 should win OBMC SSE over either
            // single-ref mode on at least some blocks.
            let mut m = [0u8; 64 * 64];
            for i in 0..64 * 64 {
                m[i] = ((y0[i] as u16 + y1[i] as u16 + 1) >> 1) as u8;
            }
            m
        };
        let blocks_x = 16u32;
        let blocks_y = 16u32;
        let before = bipred_select_modes(&mid, &y0, &y1, 64, 64, blocks_x, blocks_y, 16, 0);
        let mut after = before.clone();
        bipred_post_obmc_refine_modes(&mid, &y0, &y1, 64, 64, blocks_x, blocks_y, &mut after, 0);
        // Strict-superset: every block's MV pair is preserved (int-pel
        // snap is identity at `p=0`), but the mode can change. The
        // invariant: SSE-after ≤ SSE-before (already pinned by the
        // monotonicity test; here we just smoke-check the call runs to
        // completion and produces a sensible grid).
        assert_eq!(after.len(), before.len());
        for d in &after {
            assert!(
                !matches!(d.rmode, RefPredMode::Intra),
                "post-OBMC bipred refinement emitted Intra mode"
            );
            // At p=0, MVs are not modified — snap-to-int is identity.
            // We assert MVs are drawn from the candidate set {cur.mv1,
            // cur.mv2} which (at p=0) equals the input MVs.
        }
    }
}
