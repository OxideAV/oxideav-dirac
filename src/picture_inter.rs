//! Inter-picture motion data structures and block motion data decode
//! (§11.2 + §12.3). The actual OBMC arithmetic lives in
//! [`crate::obmc`]; this module owns:
//!
//! * The shared data types — [`BlockData`], [`RefPredMode`],
//!   [`GlobalParams`], [`PictureMotionData`].
//! * §11.2.1 `picture_prediction_parameters` parsing.
//! * §12.3 `block_motion_data` decode, including the arithmetic
//!   decoder contexts and the spatial predictions of §12.3.6.

use crate::arith::{ArithDecoder, ContextBank};
use crate::bits::BitReader;
use crate::picture::PictureError;
use crate::sequence::SequenceHeader;

/// Reference prediction mode (§12.1.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefPredMode {
    Intra,
    Ref1Only,
    Ref2Only,
    Ref1And2,
}

impl RefPredMode {
    /// 2-bit encoding used inside the spec's BLOCK_DATA[RMODE] cell:
    /// bit 0 = ref1 used, bit 1 = ref2 used.
    pub fn from_bits(bits: u32) -> Self {
        match bits & 3 {
            0 => Self::Intra,
            1 => Self::Ref1Only,
            2 => Self::Ref2Only,
            _ => Self::Ref1And2,
        }
    }

    pub fn to_bits(self) -> u32 {
        match self {
            Self::Intra => 0,
            Self::Ref1Only => 1,
            Self::Ref2Only => 2,
            Self::Ref1And2 => 3,
        }
    }

    pub fn uses_ref(self, r: usize) -> bool {
        let b = self.to_bits();
        (b >> (r - 1)) & 1 == 1
    }
}

/// §11.2.6 global motion parameters: affine + perspective.
#[derive(Debug, Clone, Default)]
pub struct GlobalParams {
    /// Pan/tilt (b in the spec).
    pub pan_tilt: (i32, i32),
    /// 2x2 zoom/rotate/shear matrix (A in the spec).
    pub zrs: [[i32; 2]; 2],
    pub zrs_exp: u32,
    /// Perspective vector (c in the spec).
    pub perspective: (i32, i32),
    pub persp_exp: u32,
}

/// A single block's decoded motion data.
#[derive(Debug, Clone)]
pub struct BlockData {
    pub rmode: RefPredMode,
    pub gmode: bool,
    /// `mv[0]` = reference 1 vector, `mv[1]` = reference 2 vector. Units
    /// are 1/(2^mv_precision) pixels.
    pub mv: [(i32, i32); 2],
    /// DC values for Y, C1, C2 — used for INTRA blocks.
    pub dc: [i32; 3],
}

impl Default for BlockData {
    fn default() -> Self {
        Self {
            rmode: RefPredMode::Intra,
            gmode: false,
            mv: [(0, 0); 2],
            dc: [0; 3],
        }
    }
}

/// Aggregated picture-prediction parameters (§11.2.1).
#[derive(Debug, Clone)]
pub struct PicturePredictionParams {
    pub luma_xblen: u32,
    pub luma_yblen: u32,
    pub luma_xbsep: u32,
    pub luma_ybsep: u32,
    pub mv_precision: u32,
    pub using_global: bool,
    pub prediction_mode: u32,
    pub superblocks_x: u32,
    pub superblocks_y: u32,
    pub blocks_x: u32,
    pub blocks_y: u32,
    pub refs_wt_precision: u32,
    pub ref1_wt: i32,
    pub ref2_wt: i32,
    pub global1: Option<GlobalParams>,
    pub global2: Option<GlobalParams>,
}

/// Per-picture decoded motion data: superblock splits + per-block
/// `BlockData` + optional global motion.
#[derive(Debug, Clone)]
pub struct PictureMotionData {
    pub blocks_x: u32,
    pub blocks_y: u32,
    pub superblocks_x: u32,
    pub superblocks_y: u32,
    /// Superblock split level, row-major `[ysb * sbx + xsb]`.
    pub sb_split: Vec<u32>,
    /// Block data array, row-major `[y * blocks_x + x]`.
    pub blocks: Vec<BlockData>,
    pub global1: Option<GlobalParams>,
    pub global2: Option<GlobalParams>,
}

impl PictureMotionData {
    pub fn get_block(&self, x: u32, y: u32) -> &BlockData {
        let idx = y as usize * self.blocks_x as usize + x as usize;
        &self.blocks[idx]
    }

    pub fn get_block_mut(&mut self, x: u32, y: u32) -> &mut BlockData {
        let idx = y as usize * self.blocks_x as usize + x as usize;
        &mut self.blocks[idx]
    }

    pub fn split(&self, xsb: u32, ysb: u32) -> u32 {
        self.sb_split[(ysb * self.superblocks_x + xsb) as usize]
    }

    pub fn split_mut(&mut self, xsb: u32, ysb: u32) -> &mut u32 {
        let i = (ysb * self.superblocks_x + xsb) as usize;
        &mut self.sb_split[i]
    }
}

/// Luma motion-block parameter presets from Table 11.1 plus index 0
/// (custom).
fn preset_block_params(index: u32) -> Option<(u32, u32, u32, u32)> {
    match index {
        1 => Some((8, 8, 4, 4)),
        2 => Some((12, 12, 8, 8)),
        3 => Some((16, 16, 12, 12)),
        4 => Some((24, 24, 16, 16)),
        _ => None,
    }
}

/// Parse §11.2.1 picture_prediction_parameters from the picture payload.
pub fn parse_picture_prediction_parameters(
    r: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    num_refs: u32,
) -> Result<PicturePredictionParams, PictureError> {
    // §11.2.2 block_parameters.
    let index = r.read_uint();
    let (luma_xblen, luma_yblen, luma_xbsep, luma_ybsep) = if index == 0 {
        let xblen = r.read_uint();
        let yblen = r.read_uint();
        let xbsep = r.read_uint();
        let ybsep = r.read_uint();
        (xblen, yblen, xbsep, ybsep)
    } else {
        preset_block_params(index).ok_or(PictureError::InterNotImplemented)?
    };

    // §11.2.4 motion_data_dimensions.
    let lw = sequence.luma_width;
    let lh = sequence.luma_height;
    let four_xbsep = 4 * luma_xbsep.max(1);
    let four_ybsep = 4 * luma_ybsep.max(1);
    let superblocks_x = lw.div_ceil(four_xbsep);
    let superblocks_y = lh.div_ceil(four_ybsep);
    let blocks_x = 4 * superblocks_x;
    let blocks_y = 4 * superblocks_y;

    // §11.2.5 motion_vector_precision.
    let mv_precision = r.read_uint();

    // §11.2.6 global_motion.
    let using_global = r.read_bool();
    let mut global1 = None;
    let mut global2 = None;
    if using_global {
        global1 = Some(parse_global_motion_parameters(r));
        if num_refs == 2 {
            global2 = Some(parse_global_motion_parameters(r));
        }
    }

    // §11.2.7 picture_prediction_mode.
    let prediction_mode = r.read_uint();

    // §11.2.8 reference_picture_weights.
    let mut refs_wt_precision = 1u32;
    let mut ref1_wt = 1i32;
    let mut ref2_wt = 1i32;
    let reference_picture_weights_flag = r.read_bool();
    if reference_picture_weights_flag {
        refs_wt_precision = r.read_uint();
        ref1_wt = r.read_sint();
        if num_refs == 2 {
            ref2_wt = r.read_sint();
        }
    }

    Ok(PicturePredictionParams {
        luma_xblen,
        luma_yblen,
        luma_xbsep,
        luma_ybsep,
        mv_precision,
        using_global,
        prediction_mode,
        superblocks_x,
        superblocks_y,
        blocks_x,
        blocks_y,
        refs_wt_precision,
        ref1_wt,
        ref2_wt,
        global1,
        global2,
    })
}

/// §11.2.6 global motion parameters for one reference.
fn parse_global_motion_parameters(r: &mut BitReader<'_>) -> GlobalParams {
    let mut g = GlobalParams::default();
    // pan_tilt
    if r.read_bool() {
        g.pan_tilt.0 = r.read_sint();
        g.pan_tilt.1 = r.read_sint();
    }
    // zoom_rotate_shear
    if r.read_bool() {
        g.zrs_exp = r.read_uint();
        g.zrs[0][0] = r.read_sint();
        g.zrs[0][1] = r.read_sint();
        g.zrs[1][0] = r.read_sint();
        g.zrs[1][1] = r.read_sint();
    } else {
        g.zrs = [[1, 0], [0, 1]];
    }
    // perspective
    if r.read_bool() {
        g.persp_exp = r.read_uint();
        g.perspective.0 = r.read_sint();
        g.perspective.1 = r.read_sint();
    }
    g
}

// ---- Arith contexts for block motion data (§12.3.7) -----------------

#[allow(non_camel_case_types, dead_code)]
pub mod mvctx {
    // Superblock split contexts.
    pub const SB_F1: usize = 0;
    pub const SB_F2: usize = 1;
    pub const SB_DATA: usize = 2;
    // Prediction mode contexts.
    pub const PMODE_REF1: usize = 0;
    pub const PMODE_REF2: usize = 1;
    pub const GLOBAL_BLOCK: usize = 2;
    // Motion vector contexts (used by vector_elements).
    pub const VECTOR_F1: usize = 0;
    pub const VECTOR_F2: usize = 1;
    pub const VECTOR_F3: usize = 2;
    pub const VECTOR_F4: usize = 3;
    pub const VECTOR_F5PLUS: usize = 4;
    pub const VECTOR_DATA: usize = 5;
    pub const VECTOR_SIGN: usize = 6;
    // DC value contexts.
    pub const DC_F1: usize = 0;
    pub const DC_F2PLUS: usize = 1;
    pub const DC_DATA: usize = 2;
    pub const DC_SIGN: usize = 3;
}

/// Wrap `data[byte_pos..byte_pos + length]` as an arith-decoded block,
/// then advance the outer reader past it.
fn arith_from_reader<'a>(r: &mut BitReader<'a>, length: usize) -> (ArithDecoder<'a>, usize) {
    r.byte_align();
    let start = r.byte_pos();
    let data = r.data();
    let available = data.len().saturating_sub(start);
    let effective = length.min(available);
    let block = &data[start..start + effective];
    let dec = ArithDecoder::new(block, effective);
    (dec, effective)
}

/// §12.3 `block_motion_data`. On entry the reader sits immediately
/// after the byte-aligned end of `picture_prediction_parameters`.
pub fn decode_block_motion_data(
    r: &mut BitReader<'_>,
    params: &PicturePredictionParams,
    num_refs: u32,
) -> Result<PictureMotionData, PictureError> {
    let sbx = params.superblocks_x;
    let sby = params.superblocks_y;
    let bx = params.blocks_x;
    let by = params.blocks_y;
    let sb_split = vec![0u32; (sbx * sby) as usize];
    let blocks = vec![BlockData::default(); (bx * by) as usize];
    let mut motion = PictureMotionData {
        blocks_x: bx,
        blocks_y: by,
        superblocks_x: sbx,
        superblocks_y: sby,
        sb_split,
        blocks,
        global1: params.global1.clone(),
        global2: params.global2.clone(),
    };

    // 1. Superblock split modes.
    decode_sb_splits(r, &mut motion)?;
    // 2. Prediction modes.
    decode_prediction_modes(r, &mut motion, &params_with_global(params))?;
    // 3. Motion vectors.
    decode_vector_elements(r, &mut motion, 1, 0)?;
    decode_vector_elements(r, &mut motion, 1, 1)?;
    if num_refs == 2 {
        decode_vector_elements(r, &mut motion, 2, 0)?;
        decode_vector_elements(r, &mut motion, 2, 1)?;
    }
    // 4. DC values for intra blocks.
    for c in 0..3 {
        decode_dc_values(r, &mut motion, c)?;
    }
    Ok(motion)
}

/// Small helper bundle so closures don't need to borrow the whole
/// `params` struct.
struct MotionCtx {
    num_refs: u32,
    using_global: bool,
}
fn params_with_global(params: &PicturePredictionParams) -> MotionCtx {
    // num_refs is determined by the parse code, not the prediction
    // parameters — infer from whether global2 or ref2_wt exists. For
    // our purposes, we always pass num_refs through decode_block_motion_data
    // but structured code is simpler with this helper.
    MotionCtx {
        num_refs: if params.global2.is_some() { 2 } else { 1 },
        using_global: params.using_global,
    }
}

fn decode_sb_splits(
    r: &mut BitReader<'_>,
    motion: &mut PictureMotionData,
) -> Result<(), PictureError> {
    let length = r.read_uint() as usize;
    let (mut dec, _) = arith_from_reader(r, length);
    let mut bank = ContextBank::new(3);
    for ysb in 0..motion.superblocks_y {
        for xsb in 0..motion.superblocks_x {
            let residual = dec.read_uint(&mut bank, &[mvctx::SB_F1, mvctx::SB_F2], mvctx::SB_DATA);
            let pred = split_prediction(motion, xsb, ysb);
            let val = (residual + pred) % 3;
            *motion.split_mut(xsb, ysb) = val;
        }
    }
    // Advance outer reader past the block.
    let _ = dec;
    let cur = r.byte_pos();
    r.skip_to(cur + length);
    Ok(())
}

fn split_prediction(motion: &PictureMotionData, x: u32, y: u32) -> u32 {
    if x == 0 && y == 0 {
        0
    } else if y == 0 {
        motion.split(x - 1, 0)
    } else if x == 0 {
        motion.split(0, y - 1)
    } else {
        let a = motion.split(x - 1, y - 1) as i64;
        let b = motion.split(x, y - 1) as i64;
        let c = motion.split(x - 1, y) as i64;
        // mean: unbiased: (sum + n//2)//n
        ((a + b + c + 1) / 3) as u32
    }
}

fn decode_prediction_modes(
    r: &mut BitReader<'_>,
    motion: &mut PictureMotionData,
    ctx: &MotionCtx,
) -> Result<(), PictureError> {
    let length = r.read_uint() as usize;
    let (mut dec, _) = arith_from_reader(r, length);
    let mut bank = ContextBank::new(3);
    for ysb in 0..motion.superblocks_y {
        for xsb in 0..motion.superblocks_x {
            let split = motion.split(xsb, ysb);
            let block_count = 1u32 << split;
            let step = 4 / block_count;
            for q in 0..block_count {
                for p in 0..block_count {
                    let bx = 4 * xsb + p * step;
                    let by = 4 * ysb + q * step;
                    block_ref_mode(&mut dec, &mut bank, motion, bx, by, ctx.num_refs);
                    propagate(motion, bx, by, step, PropField::RMode);
                    block_global_mode(&mut dec, &mut bank, motion, bx, by, ctx.using_global);
                    propagate(motion, bx, by, step, PropField::GMode);
                }
            }
        }
    }
    let _ = dec;
    let cur = r.byte_pos();
    r.skip_to(cur + length);
    Ok(())
}

#[derive(Clone, Copy)]
enum PropField {
    RMode,
    GMode,
    Vector,
    Dc(usize),
}

fn propagate(motion: &mut PictureMotionData, xtl: u32, ytl: u32, k: u32, field: PropField) {
    if k <= 1 {
        return;
    }
    let src = motion.get_block(xtl, ytl).clone();
    for y in ytl..(ytl + k).min(motion.blocks_y) {
        for x in xtl..(xtl + k).min(motion.blocks_x) {
            if x == xtl && y == ytl {
                continue;
            }
            let dst = motion.get_block_mut(x, y);
            match field {
                PropField::RMode => dst.rmode = src.rmode,
                PropField::GMode => dst.gmode = src.gmode,
                PropField::Vector => dst.mv = src.mv,
                PropField::Dc(c) => dst.dc[c] = src.dc[c],
            }
        }
    }
}

fn block_ref_mode(
    dec: &mut ArithDecoder<'_>,
    bank: &mut ContextBank,
    motion: &mut PictureMotionData,
    x: u32,
    y: u32,
    num_refs: u32,
) {
    let mut bits = 0u32;
    if dec.read_bool(bank, mvctx::PMODE_REF1) {
        bits = 1;
    }
    if num_refs == 2 && dec.read_bool(bank, mvctx::PMODE_REF2) {
        bits += 2;
    }
    let pred = ref_mode_prediction(motion, x, y).to_bits();
    let combined = bits ^ pred;
    motion.get_block_mut(x, y).rmode = RefPredMode::from_bits(combined);
}

fn ref_mode_prediction(motion: &PictureMotionData, x: u32, y: u32) -> RefPredMode {
    if x == 0 && y == 0 {
        return RefPredMode::Intra;
    }
    if y == 0 {
        return motion.get_block(x - 1, 0).rmode;
    }
    if x == 0 {
        return motion.get_block(0, y - 1).rmode;
    }
    // Majority vote per reference bit.
    let a = motion.get_block(x - 1, y - 1).rmode.to_bits();
    let b = motion.get_block(x, y - 1).rmode.to_bits();
    let c = motion.get_block(x - 1, y).rmode.to_bits();
    let ref1_count = (a & 1) + (b & 1) + (c & 1);
    let ref2_count = ((a >> 1) & 1) + ((b >> 1) & 1) + ((c >> 1) & 1);
    let mut pred = 0u32;
    if ref1_count >= 2 {
        pred |= 1;
    }
    if ref2_count >= 2 {
        pred |= 2;
    }
    RefPredMode::from_bits(pred)
}

fn block_global_mode(
    dec: &mut ArithDecoder<'_>,
    bank: &mut ContextBank,
    motion: &mut PictureMotionData,
    x: u32,
    y: u32,
    using_global: bool,
) {
    let rmode = motion.get_block(x, y).rmode;
    if !using_global || rmode == RefPredMode::Intra {
        motion.get_block_mut(x, y).gmode = false;
        return;
    }
    let residual = dec.read_bool(bank, mvctx::GLOBAL_BLOCK);
    let pred = block_global_prediction(motion, x, y);
    motion.get_block_mut(x, y).gmode = residual ^ pred;
}

fn block_global_prediction(motion: &PictureMotionData, x: u32, y: u32) -> bool {
    if x == 0 && y == 0 {
        return false;
    }
    if y == 0 {
        return motion.get_block(x - 1, 0).gmode;
    }
    if x == 0 {
        return motion.get_block(0, y - 1).gmode;
    }
    let count = motion.get_block(x - 1, y - 1).gmode as u32
        + motion.get_block(x, y - 1).gmode as u32
        + motion.get_block(x - 1, y).gmode as u32;
    count >= 2
}

fn decode_vector_elements(
    r: &mut BitReader<'_>,
    motion: &mut PictureMotionData,
    ref_num: u32,
    dirn: u32,
) -> Result<(), PictureError> {
    let length = r.read_uint() as usize;
    let (mut dec, _) = arith_from_reader(r, length);
    let mut bank = ContextBank::new(7);
    let follow = [
        mvctx::VECTOR_F1,
        mvctx::VECTOR_F2,
        mvctx::VECTOR_F3,
        mvctx::VECTOR_F4,
        mvctx::VECTOR_F5PLUS,
    ];
    for ysb in 0..motion.superblocks_y {
        for xsb in 0..motion.superblocks_x {
            let split = motion.split(xsb, ysb);
            let block_count = 1u32 << split;
            let step = 4 / block_count;
            for q in 0..block_count {
                for p in 0..block_count {
                    let bx = 4 * xsb + p * step;
                    let by = 4 * ysb + q * step;
                    block_vector(&mut dec, &mut bank, motion, bx, by, ref_num, dirn, &follow);
                    propagate(motion, bx, by, step, PropField::Vector);
                }
            }
        }
    }
    let _ = dec;
    let cur = r.byte_pos();
    r.skip_to(cur + length);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn block_vector(
    dec: &mut ArithDecoder<'_>,
    bank: &mut ContextBank,
    motion: &mut PictureMotionData,
    x: u32,
    y: u32,
    ref_num: u32,
    dirn: u32,
    follow: &[usize],
) {
    let block = motion.get_block(x, y);
    // Only decode a residual if this ref is used AND not a global block.
    let uses = block.rmode.uses_ref(ref_num as usize);
    if !uses || block.gmode {
        return;
    }
    let residual = dec.read_sint(bank, follow, mvctx::VECTOR_DATA, mvctx::VECTOR_SIGN);
    let pred = mv_prediction(motion, x, y, ref_num, dirn);
    let value = residual + pred;
    let block = motion.get_block_mut(x, y);
    let idx = (ref_num - 1) as usize;
    match dirn {
        0 => block.mv[idx].0 = value,
        _ => block.mv[idx].1 = value,
    }
}

fn mv_available(b: &BlockData, ref_num: u32) -> bool {
    !b.gmode && b.rmode.uses_ref(ref_num as usize)
}

fn mv_prediction(motion: &PictureMotionData, x: u32, y: u32, ref_num: u32, dirn: u32) -> i32 {
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
    if y == 0 {
        let b = motion.get_block(x - 1, 0);
        return if mv_available(b, ref_num) { pick(b) } else { 0 };
    }
    if x == 0 {
        let b = motion.get_block(0, y - 1);
        return if mv_available(b, ref_num) { pick(b) } else { 0 };
    }
    let mut values: Vec<i32> = Vec::with_capacity(3);
    for (dx, dy) in [(1, 0), (0, 1), (1, 1)] {
        let b = motion.get_block(x - dx, y - dy);
        if mv_available(b, ref_num) {
            values.push(pick(b));
        }
    }
    median(&mut values)
}

/// Integer median (§6.4.3). Empty → 0; even n → mean of the two middle
/// values.
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
        // Unbiased mean of 2 values (§6.4.3): (a + b + 1) / 2 with floor.
        let sum = a + b + 1;
        let q = sum / 2;
        let r = sum % 2;
        (if r < 0 { q - 1 } else { q }) as i32
    }
}

fn decode_dc_values(
    r: &mut BitReader<'_>,
    motion: &mut PictureMotionData,
    c: usize,
) -> Result<(), PictureError> {
    let length = r.read_uint() as usize;
    let (mut dec, _) = arith_from_reader(r, length);
    let mut bank = ContextBank::new(4);
    let follow = [mvctx::DC_F1, mvctx::DC_F2PLUS];
    for ysb in 0..motion.superblocks_y {
        for xsb in 0..motion.superblocks_x {
            let split = motion.split(xsb, ysb);
            let block_count = 1u32 << split;
            let step = 4 / block_count;
            for q in 0..block_count {
                for p in 0..block_count {
                    let bx = 4 * xsb + p * step;
                    let by = 4 * ysb + q * step;
                    block_dc(&mut dec, &mut bank, motion, bx, by, c, &follow);
                    propagate(motion, bx, by, step, PropField::Dc(c));
                }
            }
        }
    }
    let _ = dec;
    let cur = r.byte_pos();
    r.skip_to(cur + length);
    Ok(())
}

fn block_dc(
    dec: &mut ArithDecoder<'_>,
    bank: &mut ContextBank,
    motion: &mut PictureMotionData,
    x: u32,
    y: u32,
    c: usize,
    follow: &[usize],
) {
    if motion.get_block(x, y).rmode != RefPredMode::Intra {
        return;
    }
    let residual = dec.read_sint(bank, follow, mvctx::DC_DATA, mvctx::DC_SIGN);
    let pred = dc_prediction(motion, x, y, c);
    motion.get_block_mut(x, y).dc[c] = residual + pred;
}

fn dc_prediction(motion: &PictureMotionData, x: u32, y: u32, c: usize) -> i32 {
    if x == 0 && y == 0 {
        return 0;
    }
    let intra = |b: &BlockData| b.rmode == RefPredMode::Intra;
    if y == 0 {
        let b = motion.get_block(x - 1, 0);
        return if intra(b) { b.dc[c] } else { 0 };
    }
    if x == 0 {
        let b = motion.get_block(0, y - 1);
        return if intra(b) { b.dc[c] } else { 0 };
    }
    let mut values: Vec<i64> = Vec::with_capacity(3);
    for (dx, dy) in [(1, 0), (0, 1), (1, 1)] {
        let b = motion.get_block(x - dx, y - dy);
        if intra(b) {
            values.push(b.dc[c] as i64);
        }
    }
    if values.is_empty() {
        0
    } else {
        let n = values.len() as i64;
        let sum: i64 = values.iter().sum::<i64>() + n / 2;
        let q = sum / n;
        q as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_block_params_table_11_1() {
        assert_eq!(preset_block_params(1), Some((8, 8, 4, 4)));
        assert_eq!(preset_block_params(4), Some((24, 24, 16, 16)));
        assert_eq!(preset_block_params(0), None);
    }

    #[test]
    fn ref_pred_mode_bit_encoding() {
        assert_eq!(RefPredMode::from_bits(0), RefPredMode::Intra);
        assert_eq!(RefPredMode::from_bits(1), RefPredMode::Ref1Only);
        assert_eq!(RefPredMode::from_bits(2), RefPredMode::Ref2Only);
        assert_eq!(RefPredMode::from_bits(3), RefPredMode::Ref1And2);
        for b in 0..=3 {
            assert_eq!(RefPredMode::from_bits(b).to_bits(), b);
        }
        assert!(RefPredMode::Ref1Only.uses_ref(1));
        assert!(!RefPredMode::Ref1Only.uses_ref(2));
        assert!(RefPredMode::Ref1And2.uses_ref(1));
        assert!(RefPredMode::Ref1And2.uses_ref(2));
    }

    #[test]
    fn median_handles_empty_odd_even() {
        assert_eq!(median(&mut []), 0);
        let mut a = [5];
        assert_eq!(median(&mut a), 5);
        let mut b = [3, 1, 2];
        assert_eq!(median(&mut b), 2);
        // Unbiased mean of two values (§6.4.3): (a+b+1)//2 → 3.
        let mut c = [4, 2];
        assert_eq!(median(&mut c), 3);
    }

    #[test]
    fn propagate_level_0_fills_full_superblock() {
        let mut motion = PictureMotionData {
            blocks_x: 4,
            blocks_y: 4,
            superblocks_x: 1,
            superblocks_y: 1,
            sb_split: vec![0],
            blocks: vec![BlockData::default(); 16],
            global1: None,
            global2: None,
        };
        motion.get_block_mut(0, 0).mv[0] = (7, -3);
        propagate(&mut motion, 0, 0, 4, PropField::Vector);
        for y in 0..4u32 {
            for x in 0..4u32 {
                assert_eq!(motion.get_block(x, y).mv[0], (7, -3));
            }
        }
    }
}
