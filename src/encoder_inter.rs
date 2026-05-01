//! Dirac core-syntax **inter** encoder (round 1).
//!
//! Mirrors [`crate::picture_inter::decode_block_motion_data`] +
//! [`crate::picture::decode_picture_with_refs`] on the encode side. The
//! r1 design is intentionally narrow:
//!
//! * **Single reference** (parse code `0x09` — non-reference 1-ref AC
//!   inter picture). One `picture_number` delta, no `retd`.
//! * **No global motion**, no reference weights override (`refs_wt =
//!   1 / 1`, `refs_wt_precision = 1`).
//! * **Integer-pel motion** (`mv_precision = 0`) — chroma MVs scale
//!   directly via floor-division, no half-pel interpolation. Decoder's
//!   `pixel_pred` short-circuits the upref path here, dramatically
//!   simplifying the encoder/decoder agreement.
//! * **Block grid**: preset 1 from Table 11.1 — 8x8 blocks with 4x4
//!   stride. Two-element block_data context — uniform splits, no
//!   superblock subdivisions.
//! * **Zero-residual** — we set the §11.3 `ZERO_RESIDUAL = true` flag
//!   so the encoder emits no wavelet-coefficient stream. Reconstruction
//!   is therefore the OBMC of the reference (with whatever MV grid the
//!   ME chose) plus zero residual. PSNR is dominated by motion
//!   estimation quality.
//!
//! Motion estimation: full-search SAD over a `(mv_search_range)`-pel
//! window, integer pel only, per `bsep`-stride 8x8 block. r2 will add
//! sub-pel and OBMC overlap.

use crate::arith::{ArithEncoder, ContextBank};
use crate::bitwriter::BitWriter;
use crate::encoder::write_parse_info;
use crate::picture_inter::{mvctx, BlockData, RefPredMode};
use crate::sequence::SequenceHeader;
use crate::video_format::ChromaFormat;

/// Encoder-side inter parameters. Keep the surface small for r1.
#[derive(Debug, Clone)]
pub struct InterEncoderParams {
    /// Block-parameters preset index (Table 11.1). r1 only supports
    /// preset 1 (xblen=yblen=8, xbsep=ybsep=4).
    pub block_params_index: u32,
    /// MV search half-window in **luma pels**. SAD over `[-r, +r]` in
    /// each direction. Larger = slower but better.
    pub mv_search_range: u32,
}

impl Default for InterEncoderParams {
    fn default() -> Self {
        Self {
            block_params_index: 1,
            mv_search_range: 16,
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

/// Per-block integer-pel motion vector. Indexed `[by * blocks_x + bx]`.
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

    // 2) Prediction modes — Ref1Only everywhere, no global.
    let pmode_block = encode_prediction_modes(superblocks_x, superblocks_y, &sb_split, &blocks, bx);
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
                    // num_refs = 1 → only emit ref1 bit.
                    enc.write_bool(&mut bank, mvctx::PMODE_REF1, (bits & 1) == 1);
                    // No num_refs==2 path; no global block emit either
                    // (using_global=false everywhere in r1).
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
                    if block.rmode == RefPredMode::Intra || block.gmode {
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
    let _ = (cur_u, cur_v, ref_u, ref_v); // reserved for future chroma ME

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
    write_picture_prediction_parameters(&mut w, params);
    w.byte_align();

    // §12.3 block_motion_data.
    let (sbx, sby, blocks_x, blocks_y) = motion_grid(sequence.luma_width, sequence.luma_height);
    let mvs = full_search_me(
        cur_y,
        ref_y,
        sequence.luma_width,
        sequence.luma_height,
        blocks_x,
        blocks_y,
        params.mv_search_range,
    );
    encode_block_motion_data(&mut w, sbx, sby, blocks_x, blocks_y, &mvs);
    w.byte_align();

    // §11.3 wavelet_transform — emit zero_residual = true so we skip
    // the entire transform_parameters / coefficient stream. The
    // decoder will treat the residue as zero everywhere.
    w.write_bool(true);
    w.byte_align();

    w.finish()
}

fn write_picture_prediction_parameters(w: &mut BitWriter, _params: &InterEncoderParams) {
    // §11.2.2 block_parameters: index 1 (preset 8x8 / 4x4).
    w.write_uint(1);
    // §11.2.5 motion_vector_precision: 0 (integer-pel only in r1).
    w.write_uint(0);
    // §11.2.6 global_motion: not used.
    w.write_bool(false);
    // §11.2.7 picture_prediction_mode: 0 (default).
    w.write_uint(0);
    // §11.2.8 reference_picture_weights_flag: false → defaults
    // refs_wt_precision=1, ref1_wt=ref2_wt=1.
    w.write_bool(false);
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
    /// matching dims for our standard 64x64 grid.
    #[test]
    fn picture_prediction_parameters_roundtrips() {
        let seq = crate::encoder::make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
        let mut w = BitWriter::new();
        let params = InterEncoderParams::default();
        write_picture_prediction_parameters(&mut w, &params);
        w.byte_align();
        let bytes = w.finish();
        let _ = parse_sequence_header; // re-export check.

        let mut r = crate::bits::BitReader::new(&bytes);
        let pred = parse_picture_prediction_parameters(&mut r, &seq, 1).expect("parse PPP");
        assert_eq!(pred.luma_xblen, 8);
        assert_eq!(pred.luma_yblen, 8);
        assert_eq!(pred.luma_xbsep, 4);
        assert_eq!(pred.luma_ybsep, 4);
        assert_eq!(pred.mv_precision, 0);
        assert!(!pred.using_global);
        assert_eq!(pred.superblocks_x, 4);
        assert_eq!(pred.superblocks_y, 4);
        assert_eq!(pred.blocks_x, 16);
        assert_eq!(pred.blocks_y, 16);
        assert_eq!(pred.refs_wt_precision, 1);
        assert_eq!(pred.ref1_wt, 1);
        assert_eq!(pred.ref2_wt, 1);
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
}
