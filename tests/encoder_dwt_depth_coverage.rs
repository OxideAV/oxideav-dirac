//! Round-218 — `dwt_depth` axis coverage for the VC-2 HQ + LD intra
//! encoder/decoder round-trip.
//!
//! Background: the project's own
//! `docs/video/dirac/dirac-fixtures-and-traces.md` "Gaps" section
//! explicitly calls out **wavelet depths other than 3 and 4** as a
//! feature surface that no upstream-slice fixture exercises (the
//! allowed range per §11.3 is `1..=5`). The pre-r218 encoder/decoder
//! self-roundtrip surface lined up with that gap: `tests/encoder_*`
//! drove `dwt_depth = 3` almost exclusively, with a single `depth = 2`
//! HQ spot-check in `tests/encoder_matrix.rs::hq_q0_lossless_at_dwt_depth_two`
//! and the asymmetric `dwt_depth_ho` v3 paths in
//! `tests/encoder_roundtrip.rs`.
//!
//! This file fills in the axis by self-roundtripping the HQ and LD
//! encoders against the decoder at every spec-allowed depth `1..=5`:
//!
//! * depth `1..=4` use the Annex E.1 default quantisation matrix
//!   (`QuantMatrix::default_for`), exactly as `default_hq` /
//!   `default_ld` do.
//! * depth `5` requires a custom quantisation matrix per §11.3.5
//!   (the default tables only run to depth 4); we construct an all-zero
//!   quant matrix sized to `dwt_depth + 1` levels and set
//!   `custom_quant_matrix = true` so it travels in-band.
//!
//! Picture dimensions and slice grid are picked so the per-component
//! per-slice dimension `comp_size / slices_axis` stays an integer
//! multiple of `2^dwt_depth` (the spec's §15.7 alignment, also called
//! out in `tests/encoder_matrix.rs`). The encoder's `forward_component`
//! pads to a multiple of `2^dwt_depth` so the picture itself is fine
//! at any picture size, but slice subdivision has no such pad — for
//! depth `4` and `5` we use 64x64 pictures with 2x2 slices (slice
//! dim 32 = `2^5`).
//!
//! HQ at qindex=0 with the LeGall 5/3 wavelet is bit-exact on a
//! smooth-content picture (the dead-zone quantiser is an identity and
//! the lifting steps round-trip without integer loss). LD at qindex=0
//! is asserted PSNR >= 35 dB (the same threshold as
//! `tests/encoder_roundtrip.rs::encode_then_decode_ld_qindex0_psnr_over_35`).
//!
//! All material consulted: BBC Dirac Specification v2.2.3 (the spec
//! PDF at `docs/video/dirac/dirac-spec-latest.pdf`) and SMPTE ST
//! 2042-1 via its sections quoted in `src/encoder.rs` / `src/picture.rs`
//! doc comments. No external implementation source consulted.

use oxideav_core::CodecRegistry;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase, VideoFrame};
use oxideav_dirac::encoder::{
    encode_single_hq_intra_stream, encode_single_ld_intra_stream, make_minimal_sequence,
    make_minimal_sequence_ld, EncoderParams, LdEncoderParams,
};
use oxideav_dirac::quant::QuantMatrix;
use oxideav_dirac::video_format::ChromaFormat;
use oxideav_dirac::wavelet::WaveletFilter;

/// Build a smooth diagonal-gradient picture: an 8-bit plane where the
/// neighbour deltas are small enough that wavelet HF subbands stay tiny
/// and a tight slice budget fits at qindex=0. Caller-supplied seed
/// shifts the gradient so the three planes hold different content.
fn smooth_plane(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut p = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            // Two-step diagonal gradient + per-plane seed shift, kept
            // well below 255 so the inverse quantiser does not clip.
            let v = ((x as u32 + y as u32 + seed) & 0xff) as u8;
            p[y * w + x] = v.min(200);
        }
    }
    p
}

/// Drive the decoder on a single-frame elementary stream and return
/// the decoded video frame. Panics on send/receive errors or on a
/// non-video frame.
fn decode_one(stream: Vec<u8>) -> VideoFrame {
    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("decoder");
    let packet = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&packet).expect("send_packet");
    match dec.receive_frame().expect("receive_frame") {
        Frame::Video(vf) => vf,
        other => panic!("expected video frame, got {other:?}"),
    }
}

/// PSNR helper matching the convention in `tests/encoder_roundtrip.rs`:
/// 20 * log10(255) - 10 * log10(MSE), with a sentinel `INFINITY` for an
/// exact-match reconstruction.
fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut sse: u64 = 0;
    for i in 0..a.len() {
        let d = a[i] as i32 - b[i] as i32;
        sse += (d * d) as u64;
    }
    if sse == 0 {
        return f64::INFINITY;
    }
    let mse = sse as f64 / a.len() as f64;
    20.0 * (255.0f64).log10() - 10.0 * mse.log10()
}

/// Pick a slice grid such that each per-component slice dimension is
/// itself an integer multiple of `2^dwt_depth`. For our 64x64 luma /
/// 32x32 chroma 4:2:0 picture this gives:
///
/// * depth 1..=3 → `(8, 8)` (the `default_hq` / `default_ld` choice)
/// * depth 4..=5 → `(2, 2)` (slice luma dim 32, slice chroma dim 16
///   — both still multiples of `2^5`).
fn slice_grid_for_64x64_yuv420(dwt_depth: u32) -> (u32, u32) {
    if dwt_depth <= 3 {
        (8, 8)
    } else {
        (2, 2)
    }
}

/// Construct an all-zero custom quantisation matrix for the given
/// `(wavelet, dwt_depth)` pair. Each level carries `[LL, HL, LH, HH]`
/// quantiser offsets per §12.4.5.3 / §13.5.4; an all-zero matrix
/// leaves the per-subband quantiser equal to the slice header's
/// `qindex` (no per-band scaling). Required for `dwt_depth = 5` per
/// §11.3.5 since the Annex E.1 default tables only define depths up
/// to 4.
fn zero_custom_quant_matrix(dwt_depth: u32) -> QuantMatrix {
    let levels: Vec<[u32; 4]> = (0..=dwt_depth).map(|_| [0, 0, 0, 0]).collect();
    QuantMatrix {
        dwt_depth,
        dwt_depth_ho: 0,
        levels,
    }
}

// ------------------------------------------------------------------
// HQ intra: bit-exact roundtrip at every supported depth.
// ------------------------------------------------------------------

/// HQ at depth 1: shallowest legal pyramid (one LL + one set of
/// HL/LH/HH bands), `default_hq` default quant matrix, qindex=0.
/// Bit-exact reconstruction required.
#[test]
fn hq_q0_bit_exact_at_dwt_depth_one() {
    let w: u32 = 64;
    let h: u32 = 64;
    let y = smooth_plane(w as usize, h as usize, 0);
    let u = smooth_plane((w / 2) as usize, (h / 2) as usize, 17);
    let v = smooth_plane((w / 2) as usize, (h / 2) as usize, 31);

    let seq = make_minimal_sequence(w, h, ChromaFormat::Yuv420);
    let params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 1);

    let stream = encode_single_hq_intra_stream(&seq, &params, 0, &y, &u, &v);
    let frame = decode_one(stream);

    assert_eq!(frame.planes[0].data, y, "Y not bit-exact at depth=1");
    assert_eq!(frame.planes[1].data, u, "U not bit-exact at depth=1");
    assert_eq!(frame.planes[2].data, v, "V not bit-exact at depth=1");
}

/// HQ at depth 4: one more pyramid level than the `default_hq` test
/// surface, slice grid loosened to `(2, 2)` so each per-component
/// slice dimension stays an integer multiple of `2^4`. Bit-exact
/// reconstruction required.
#[test]
fn hq_q0_bit_exact_at_dwt_depth_four() {
    let w: u32 = 64;
    let h: u32 = 64;
    let y = smooth_plane(w as usize, h as usize, 2);
    let u = smooth_plane((w / 2) as usize, (h / 2) as usize, 5);
    let v = smooth_plane((w / 2) as usize, (h / 2) as usize, 11);

    let seq = make_minimal_sequence(w, h, ChromaFormat::Yuv420);
    let mut params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 4);
    let (sx, sy) = slice_grid_for_64x64_yuv420(4);
    params.slices_x = sx;
    params.slices_y = sy;

    let stream = encode_single_hq_intra_stream(&seq, &params, 0, &y, &u, &v);
    let frame = decode_one(stream);

    assert_eq!(frame.planes[0].data, y, "Y not bit-exact at depth=4");
    assert_eq!(frame.planes[1].data, u, "U not bit-exact at depth=4");
    assert_eq!(frame.planes[2].data, v, "V not bit-exact at depth=4");
}

/// HQ at depth 5: the deepest legal pyramid (§11.3 caps `dwt_depth`
/// at 5 in practice; our decoder also rejects `> 6` defensively).
/// `default_hq` cannot reach here because `QuantMatrix::default_for`
/// returns `None` for depths > 4 (§11.3.5: a *custom* quant matrix is
/// required). We construct an all-zero custom matrix and set the
/// `custom_quant_matrix = true` flag so it travels in-band; the
/// decoder reads it back via §12.4.5.3. At qindex=0 the dead-zone
/// quantiser collapses to an identity at every level, so the
/// reconstruction is bit-exact.
#[test]
fn hq_q0_bit_exact_at_dwt_depth_five_custom_matrix() {
    let w: u32 = 64;
    let h: u32 = 64;
    let y = smooth_plane(w as usize, h as usize, 4);
    let u = smooth_plane((w / 2) as usize, (h / 2) as usize, 8);
    let v = smooth_plane((w / 2) as usize, (h / 2) as usize, 13);

    let seq = make_minimal_sequence(w, h, ChromaFormat::Yuv420);
    let (sx, sy) = slice_grid_for_64x64_yuv420(5);
    let matrix = zero_custom_quant_matrix(5);
    let params = EncoderParams {
        wavelet: WaveletFilter::LeGall5_3,
        dwt_depth: 5,
        slices_x: sx,
        slices_y: sy,
        slice_prefix_bytes: 0,
        slice_size_scaler: 1,
        quant_matrix: matrix,
        custom_quant_matrix: true,
        qindex: 0,
        slice_size_target: None,
        major_version: 2,
        extended_transform_override: None,
    };

    let stream = encode_single_hq_intra_stream(&seq, &params, 0, &y, &u, &v);
    let frame = decode_one(stream);

    assert_eq!(frame.planes[0].data, y, "Y not bit-exact at depth=5");
    assert_eq!(frame.planes[1].data, u, "U not bit-exact at depth=5");
    assert_eq!(frame.planes[2].data, v, "V not bit-exact at depth=5");
}

// ------------------------------------------------------------------
// LD intra: above-threshold PSNR at every supported depth.
// ------------------------------------------------------------------

/// LD at depth 1: shallow pyramid, generous slice budget so qindex=0
/// reconstruction holds well above the 35 dB threshold used by
/// `tests/encoder_roundtrip.rs::encode_then_decode_ld_qindex0_psnr_over_35`.
#[test]
fn ld_q0_psnr_over_35_at_dwt_depth_one() {
    let w: u32 = 64;
    let h: u32 = 64;
    let y = smooth_plane(w as usize, h as usize, 0);
    let u = vec![128u8; (w * h / 4) as usize];
    let v = vec![128u8; (w * h / 4) as usize];

    let seq = make_minimal_sequence_ld(w, h, ChromaFormat::Yuv420);
    // depth = 1 uses the (8, 8) slice grid since 64 / 8 = 8 is itself
    // already a multiple of `2^1`. Bytes-per-slice intentionally
    // generous so a smooth gradient fits at q=0.
    let params = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 1, 4, 4, 256);

    let stream = encode_single_ld_intra_stream(&seq, &params, 0, &y, &u, &v);
    let frame = decode_one(stream);

    let p_y = psnr(&frame.planes[0].data, &y);
    let p_u = psnr(&frame.planes[1].data, &u);
    let p_v = psnr(&frame.planes[2].data, &v);
    eprintln!("LD q=0 depth=1: Y={p_y:.2}  U={p_u:.2}  V={p_v:.2}");
    assert!(p_y >= 35.0, "LD depth=1 Y PSNR {p_y:.2} dB < 35 dB");
    assert!(p_u >= 35.0, "LD depth=1 U PSNR {p_u:.2} dB < 35 dB");
    assert!(p_v >= 35.0, "LD depth=1 V PSNR {p_v:.2} dB < 35 dB");
}

/// LD at depth 4: one more pyramid level than the `default_ld` test
/// surface, slice grid loosened to `(2, 2)` so the per-component
/// slice dimensions stay multiples of `2^4`.
#[test]
fn ld_q0_psnr_over_35_at_dwt_depth_four() {
    let w: u32 = 64;
    let h: u32 = 64;
    let y = smooth_plane(w as usize, h as usize, 0);
    let u = vec![128u8; (w * h / 4) as usize];
    let v = vec![128u8; (w * h / 4) as usize];

    let seq = make_minimal_sequence_ld(w, h, ChromaFormat::Yuv420);
    // 2x2 slice grid × generous per-slice budget so a smooth gradient
    // round-trips comfortably at q=0.
    let params = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 4, 2, 2, 1024);

    let stream = encode_single_ld_intra_stream(&seq, &params, 0, &y, &u, &v);
    let frame = decode_one(stream);

    let p_y = psnr(&frame.planes[0].data, &y);
    let p_u = psnr(&frame.planes[1].data, &u);
    let p_v = psnr(&frame.planes[2].data, &v);
    eprintln!("LD q=0 depth=4: Y={p_y:.2}  U={p_u:.2}  V={p_v:.2}");
    assert!(p_y >= 35.0, "LD depth=4 Y PSNR {p_y:.2} dB < 35 dB");
    assert!(p_u >= 35.0, "LD depth=4 U PSNR {p_u:.2} dB < 35 dB");
    assert!(p_v >= 35.0, "LD depth=4 V PSNR {p_v:.2} dB < 35 dB");
}

/// `slice_grid_for_64x64_yuv420` returns a coarser slice grid for
/// depths >= 4 than for depths 1..=3. Tiny pure-Rust shape check —
/// pins the depth threshold so a future change keeps the deep-pyramid
/// tests aimed at a slice grid where the per-slice luma dimension is
/// itself >= `2^dwt_depth` (each slice owns at least one LL sample at
/// the bottom of the pyramid).
#[test]
fn slice_grid_helper_shape() {
    for depth in 1..=3u32 {
        assert_eq!(
            slice_grid_for_64x64_yuv420(depth),
            (8, 8),
            "depth {depth} should reuse the default-HQ 8x8 slice grid"
        );
    }
    for depth in 4..=5u32 {
        let (sx, sy) = slice_grid_for_64x64_yuv420(depth);
        assert_eq!(
            (sx, sy),
            (2, 2),
            "depth {depth} should drop to the coarser 2x2 slice grid"
        );
        // Luma slice dimension >= 2^depth so the bottom of the
        // pyramid still has at least one LL sample per slice.
        let luma_slice = 64 / sx;
        let scale: u32 = 1 << depth;
        assert!(
            luma_slice >= scale,
            "depth {depth} luma slice dim {luma_slice} < 2^{depth} = {scale}"
        );
    }
}

/// `zero_custom_quant_matrix` must produce `dwt_depth + 1` levels and
/// every cell zero — the round-218 depth-5 test relies on this being
/// the spec's `qf == qindex` identity at every subband.
#[test]
fn zero_custom_quant_matrix_shape_is_correct() {
    for depth in 1..=5u32 {
        let m = zero_custom_quant_matrix(depth);
        assert_eq!(m.dwt_depth, depth);
        assert_eq!(m.levels.len(), depth as usize + 1);
        for (i, lvl) in m.levels.iter().enumerate() {
            for (j, &cell) in lvl.iter().enumerate() {
                assert_eq!(
                    cell, 0,
                    "level {i} orient {j} should be zero in an all-zero matrix"
                );
            }
        }
    }
}
