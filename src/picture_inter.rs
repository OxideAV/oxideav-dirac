//! Inter-picture decode scaffold (§11.2 + §15.8).
//!
//! Status: **incomplete**. This module parses the picture prediction
//! parameter block and the block motion data header far enough to keep
//! the stream walker in sync, but it does not yet implement OBMC
//! motion compensation (§15.8). Any inter picture currently returns
//! `PictureError::InterNotImplemented` after a best-effort scan of the
//! motion-data header — the downstream `DiracDecoder` then propagates
//! that as an `Error::unsupported`.
//!
//! The scope of a full inter implementation is large:
//!
//! * §12.3 block motion data — superblock splitting, per-block
//!   prediction-mode + motion-vector entropy decode with the
//!   spatial-prediction context maps in §12.3.7.
//! * §15.8.x OBMC — overlapped block motion compensation with the
//!   spatial weighting matrix, sub-pixel prediction and optional
//!   global motion.
//! * Reference-picture bookkeeping — maintain a buffer of decoded
//!   reference frames keyed by `picture_number`, expire on retired
//!   picture codes.
//!
//! The parsing done here is enough to probe the "does this look like a
//! plausible inter picture?" question in tests and to keep us from
//! mis-framing the bitstream when an inter picture appears.

use crate::bits::BitReader;
use crate::picture::PictureError;
use crate::sequence::SequenceHeader;

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

/// Aggregated picture-prediction parameters (§11.2.1). Fields not
/// needed by the scaffold are discarded during parse.
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
}

/// Parse §11.2.1 picture_prediction_parameters from the picture payload.
/// `num_refs` comes from the parse code (1 or 2).
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
    let superblocks_x = (lw + 4 * luma_xbsep - 1) / (4 * luma_xbsep.max(1));
    let superblocks_y = (lh + 4 * luma_ybsep - 1) / (4 * luma_ybsep.max(1));
    let blocks_x = 4 * superblocks_x;
    let blocks_y = 4 * superblocks_y;

    // §11.2.5 motion_vector_precision.
    let mv_precision = r.read_uint();

    // §11.2.6 global_motion.
    let using_global = r.read_bool();
    if using_global {
        // Each side parses three variable-length blocks (pan, matrix,
        // perspective). For now we only need to advance the reader so
        // it stays in sync — if the picture is flagged inter we bail
        // out before motion compensation anyway.
        for _ in 0..num_refs.min(2) {
            parse_global_motion_parameters(r);
        }
    }

    // §11.2.7 picture_prediction_mode — a uint. Currently only mode 0
    // is defined in the spec but future extensions may widen it.
    let prediction_mode = r.read_uint();

    // §11.2.8 reference_picture_weights — reference-frame weighted
    // prediction. A single bool gates the weights block.
    let reference_picture_weights_flag = r.read_bool();
    if reference_picture_weights_flag {
        let _precision = r.read_uint();
        let _w1 = r.read_sint();
        if num_refs == 2 {
            let _w2 = r.read_sint();
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
    })
}

/// §11.2.6 — three sub-blocks of global motion parameters.
fn parse_global_motion_parameters(r: &mut BitReader<'_>) {
    // pan_tilt
    if r.read_bool() {
        let _ = r.read_sint();
        let _ = r.read_sint();
    }
    // matrix
    if r.read_bool() {
        let _precision = r.read_uint();
        let _ = r.read_sint();
        let _ = r.read_sint();
        let _ = r.read_sint();
        let _ = r.read_sint();
    }
    // perspective
    if r.read_bool() {
        let _precision = r.read_uint();
        let _ = r.read_sint();
        let _ = r.read_sint();
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
}
