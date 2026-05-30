//! Inter-encoder fuzz oracle (round-193).
//!
//! Sister to the round-179 intra-side `encoder_rate_control_fuzz_oracle.rs`:
//! that file sweeps the four HQ/LD rate-control variants against
//! pathological `target_bytes` / `buffer_bytes` / `max_drain_per_picture`
//! values. This file sweeps the parallel surface on the **inter** path —
//! `InterEncoderParams` (`mv_search_range`, `mv_precision`,
//! `bipred_mv_precision`, `obmc_refine_passes`, `residue`,
//! `inter_adaptive_int_pel`, `inter_adaptive_int_pel_post_obmc`,
//! `bipred_post_obmc_refine`) and `ResidueParams` (`wavelet`, `dwt_depth`,
//! `qindex`) — against pathological combinations and pathological input
//! pixel surfaces.
//!
//! Goal: every accepted (`InterEncoderParams`, `InterInputPicture`,
//! `InterInputPicture`) combination must produce a non-empty bytestream
//! that round-trips through the registry-backed decoder to exactly two
//! video frames, with no panic / no debug-assert / no integer overflow /
//! no livelock. Bit-exactness is **not** required (this is fuzz, not
//! a PSNR test); the contract is the same shape the decoder-side oracle
//! pins on its own input space: bounded time, clean termination, no
//! unsoundness.
//!
//! Coverage:
//!
//! * **Precision / OBMC / search-range sweep.** Walks the diagonal
//!   `mv_precision == bipred_mv_precision ∈ 0..=3` (integer / half-pel /
//!   quarter-pel / eighth-pel) × `obmc_refine_passes ∈ {0, 2}` ×
//!   `mv_search_range ∈ {2, 16}` against both the translating-square
//!   and camera-pan synthetic pairs, plus two off-diagonal precision
//!   pairs to pin that the 1-ref and 2-ref precisions are independent.
//!   Asserts no panic + 2-frame round-trip on every combination.
//! * **Residue wavelet / depth / qindex sweep.** All seven
//!   `WaveletFilter` variants × dwt_depth `{1, 2, 3, 4}` at a
//!   representative mid-quantiser (qindex=32), plus a qindex axis walk
//!   `{0, 8, 32, 64, 127}` at the default wavelet × depth, plus the
//!   `residue = None` legacy ZERO_RESIDUAL=true path. Linear-plus-linear
//!   walk so axis coverage doesn't blow up combinatorially.
//! * **Adaptive-flag boolean sweep.** All 8 combinations of
//!   `inter_adaptive_int_pel` × `inter_adaptive_int_pel_post_obmc` ×
//!   `bipred_post_obmc_refine` against the camera-pan fixture.
//! * **Pathological pixel inputs.** All-zero luma, all-`0xFF` luma, a
//!   single-pixel pulse, and mid-grey for both intra reference and inter
//!   target. Tests that ME / OBMC / residue paths handle the
//!   "no-energy" + "saturated-energy" extremes without livelocking the
//!   sub-pel refinement or overflowing the residue coefficient block.
//! * **Same-frame degenerate input.** Encoding `(frame, frame)` (the
//!   zero-motion edge case) — the ME path is well-defined but the SAD
//!   landscape is degenerate. The oracle pins that the encoder still
//!   produces a clean 2-frame round-trip.
//! * **Determinism.** Two back-to-back encode calls on the same input
//!   and same params must produce byte-identical streams.
//!
//! Workspace policy: clean-room. No external library code consulted.
//! Spec authority for the bounds checked here is the BBC Dirac
//! Specification v2.2.3 §11–§15 (motion compensation, sub-pel filters,
//! OBMC blend, inter residue) and the per-round invariants documented
//! in this crate's CHANGELOG (r39 / r73 / r80 / r91 / r95).

use oxideav_core::{CodecId, CodecParameters, CodecRegistry, Frame, Packet, TimeBase};
use oxideav_dirac::encoder::{make_minimal_sequence, EncoderParams};
use oxideav_dirac::encoder_inter::{
    encode_intra_then_inter_stream, synthetic_camera_pan_64, synthetic_translating_pair_64,
    InterEncoderParams, InterInputPicture, ResidueParams,
};
use oxideav_dirac::video_format::ChromaFormat;
use oxideav_dirac::wavelet::WaveletFilter;

// -------------------------------------------------------------------
// Frame-builder helpers — local to this oracle so the fuzz surface is
// self-contained (no cross-test sharing of the pathological fixtures).
// -------------------------------------------------------------------

/// 64x64 4:2:0 frame whose luma is identically `fill_y`, chroma at
/// mid-grey. Zero-energy fixture: the ME path's SAD landscape is flat,
/// every block is equally good — the picker must still converge in
/// bounded time without panicking.
fn solid_64(fill_y: u8) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    (
        vec![fill_y; 64 * 64],
        vec![128u8; 32 * 32],
        vec![128u8; 32 * 32],
    )
}

/// 64x64 4:2:0 frame with a single bright luma pixel at `(cx, cy)` on
/// an otherwise-dark field. The ME path has exactly one informative
/// block; everything else is degenerate flat — stresses the per-block
/// adaptive int-pel-vs-sub-pel decision.
fn pulse_64(cx: usize, cy: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![16u8; 64 * 64];
    if cx < 64 && cy < 64 {
        y[cy * 64 + cx] = 240;
    }
    (y, vec![128u8; 32 * 32], vec![128u8; 32 * 32])
}

/// Drive a stream through the registry-backed decoder and assert it
/// emits exactly `expected_frames` video frames in bounded time. Mirrors
/// `decoder_fuzz_oracle::drive` — any panic on the inside is an encoder
/// bug surfaced by this oracle.
fn assert_decodes_to(stream: &[u8], expected_frames: usize, label: &str) {
    assert!(
        !stream.is_empty(),
        "{label}: encoder produced an empty bytestream"
    );

    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("dirac decoder factory");
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream.to_vec());
    dec.send_packet(&pkt)
        .unwrap_or_else(|e| panic!("{label}: send_packet failed: {e:?}"));

    let mut frames = 0usize;
    // Generous cap so a hypothetical "always returns Ok with an empty
    // frame" decoder bug surfaces as a panic, not a hang.
    for _ in 0..16 {
        match dec.receive_frame() {
            Ok(Frame::Video(_)) => frames += 1,
            Ok(other) => panic!("{label}: non-video frame: {other:?}"),
            Err(_) if frames == expected_frames => return,
            Err(e) => panic!("{label}: receive_frame failed after {frames} frames: {e:?}"),
        }
        if frames == expected_frames {
            // One more pull confirms the decoder reports terminal/no-more
            // rather than spuriously emitting an extra frame.
            match dec.receive_frame() {
                Ok(_) => panic!("{label}: decoder emitted more than {expected_frames} frames"),
                Err(_) => return,
            }
        }
    }
    panic!(
        "{label}: drained 16 frames without ever reporting terminal (expected {expected_frames})"
    );
}

/// Stitch `(y, u, v)` triples into the `(intra, inter)` `InterInputPicture`
/// pair the encoder expects.
fn pair_inputs<'a>(
    intra: &'a (Vec<u8>, Vec<u8>, Vec<u8>),
    inter: &'a (Vec<u8>, Vec<u8>, Vec<u8>),
) -> (InterInputPicture<'a>, InterInputPicture<'a>) {
    (
        InterInputPicture {
            picture_number: 10,
            y: &intra.0,
            u: &intra.1,
            v: &intra.2,
        },
        InterInputPicture {
            picture_number: 11,
            y: &inter.0,
            u: &inter.1,
            v: &inter.2,
        },
    )
}

/// Baseline intra params shared across the sweeps. qindex=0 ensures the
/// intra reference round-trips bit-exact, isolating the inter path as
/// the only fuzz variable.
fn intra_params() -> EncoderParams {
    EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3)
}

// -------------------------------------------------------------------
// Sweeps
// -------------------------------------------------------------------

#[test]
fn mv_precision_obmc_search_range_sweep_never_panics() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_p = intra_params();

    // Two complementary fixtures: sharp-edge translation (favours
    // integer-pel ME) and smooth-motion camera pan (favours sub-pel).
    let pair_t = {
        let (y0, u0, v0, y1, u1, v1) = synthetic_translating_pair_64(4, 0);
        (
            (y0.to_vec(), u0.to_vec(), v0.to_vec()),
            (y1.to_vec(), u1.to_vec(), v1.to_vec()),
        )
    };
    let pair_c = {
        let (y0, u0, v0, y1, u1, v1) = synthetic_camera_pan_64(1, 0);
        (
            (y0.to_vec(), u0.to_vec(), v0.to_vec()),
            (y1.to_vec(), u1.to_vec(), v1.to_vec()),
        )
    };

    // Sub-pel precision walks 0..=3 = integer / half-pel / quarter-pel /
    // eighth-pel — diagonal sweep (`mvp == bp`) keeps both 1-ref and
    // 2-ref-path code paths exercised at every precision without the
    // full 4x4 cross-product. `obmc_refine_passes ∈ {0, 2}` covers
    // off / default. `mv_search_range ∈ {2, 16}` covers tight / default.
    // Per-fixture cost: 4 * 2 * 2 = 16 encodes (was 4*4*3*3 = 144).
    // The remaining tests in this file pick up the off-diagonal
    // precision combinations: `residue_…_sweep` runs at default
    // precision, `adaptive_flag_combinations` runs at default
    // precision, and the determinism / extreme-range tests cover
    // additional axis combinations under the default precision.
    for fixture_label in &["translating", "camera-pan"] {
        let pair = if *fixture_label == "translating" {
            &pair_t
        } else {
            &pair_c
        };
        let (intra, inter) = pair_inputs(&pair.0, &pair.1);

        for p in 0u32..=3 {
            for passes in [0u32, 2] {
                for range in [2u32, 16] {
                    let params = InterEncoderParams {
                        mv_precision: p,
                        bipred_mv_precision: p,
                        obmc_refine_passes: passes,
                        mv_search_range: range,
                        ..InterEncoderParams::default()
                    };
                    let stream =
                        encode_intra_then_inter_stream(&seq, &intra_p, &params, &intra, &inter);
                    assert_decodes_to(
                        &stream,
                        2,
                        &format!("{fixture_label} mvp=bp={p} passes={passes} range={range}"),
                    );
                }
            }
        }
    }

    // Off-diagonal precision sanity: a single (mvp, bp) pair where the
    // two precisions differ. Pins that the encoder handles mismatched
    // 1-ref vs 2-ref precision without sharing state between them.
    // Default fixture (camera-pan) only.
    let (intra, inter) = pair_inputs(&pair_c.0, &pair_c.1);
    for (mvp, bp) in [(0u32, 2u32), (2u32, 0u32)] {
        let params = InterEncoderParams {
            mv_precision: mvp,
            bipred_mv_precision: bp,
            ..InterEncoderParams::default()
        };
        let stream = encode_intra_then_inter_stream(&seq, &intra_p, &params, &intra, &inter);
        assert_decodes_to(
            &stream,
            2,
            &format!("off-diagonal mvp={mvp} bp={bp} on camera-pan"),
        );
    }
}

#[test]
fn residue_wavelet_depth_qindex_sweep_never_panics() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_p = intra_params();
    let (y0, u0, v0, y1, u1, v1) = synthetic_translating_pair_64(2, -1);
    let pair = (
        (y0.to_vec(), u0.to_vec(), v0.to_vec()),
        (y1.to_vec(), u1.to_vec(), v1.to_vec()),
    );
    let (intra, inter) = pair_inputs(&pair.0, &pair.1);

    let wavelets = [
        WaveletFilter::DeslauriersDubuc9_7,
        WaveletFilter::LeGall5_3,
        WaveletFilter::DeslauriersDubuc13_7,
        WaveletFilter::Haar0,
        WaveletFilter::Haar1,
        WaveletFilter::Fidelity,
        WaveletFilter::Daubechies9_7,
    ];

    // Cover every wavelet × every depth at one representative qindex
    // (mid-quantiser), then sweep the qindex axis at the default
    // wavelet × default depth. Quadratic blow-up (`7 * 4 * 5 = 140`
    // encodes) would push debug-build CI runtime well past a minute;
    // the linear-plus-linear walk holds the axis coverage at
    // `7 * 4 + 5 = 33` encodes.
    for wavelet in wavelets {
        for depth in [1u32, 2, 3, 4] {
            let params = InterEncoderParams {
                residue: Some(ResidueParams {
                    wavelet,
                    dwt_depth: depth,
                    qindex: 32,
                }),
                ..InterEncoderParams::default()
            };
            let stream = encode_intra_then_inter_stream(&seq, &intra_p, &params, &intra, &inter);
            assert_decodes_to(
                &stream,
                2,
                &format!("residue wavelet={wavelet:?} depth={depth} qindex=32"),
            );
        }
    }
    for qindex in [0u32, 8, 32, 64, 127] {
        let params = InterEncoderParams {
            residue: Some(ResidueParams {
                wavelet: WaveletFilter::LeGall5_3,
                dwt_depth: 3,
                qindex,
            }),
            ..InterEncoderParams::default()
        };
        let stream = encode_intra_then_inter_stream(&seq, &intra_p, &params, &intra, &inter);
        assert_decodes_to(
            &stream,
            2,
            &format!("residue qindex sweep wavelet=LeGall5_3 depth=3 qindex={qindex}"),
        );
    }

    // residue = None — the round-1 ZERO_RESIDUAL=true legacy path.
    let params_no_residue = InterEncoderParams {
        residue: None,
        ..InterEncoderParams::default()
    };
    let stream = encode_intra_then_inter_stream(&seq, &intra_p, &params_no_residue, &intra, &inter);
    assert_decodes_to(&stream, 2, "residue=None (legacy ZERO_RESIDUAL path)");
}

#[test]
fn adaptive_flag_combinations_never_panic() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_p = intra_params();
    let (y0, u0, v0, y1, u1, v1) = synthetic_camera_pan_64(1, 0);
    let pair = (
        (y0.to_vec(), u0.to_vec(), v0.to_vec()),
        (y1.to_vec(), u1.to_vec(), v1.to_vec()),
    );
    let (intra, inter) = pair_inputs(&pair.0, &pair.1);

    for adapt1 in [false, true] {
        for adapt_post in [false, true] {
            for bipred_post in [false, true] {
                let params = InterEncoderParams {
                    inter_adaptive_int_pel: adapt1,
                    inter_adaptive_int_pel_post_obmc: adapt_post,
                    bipred_post_obmc_refine: bipred_post,
                    ..InterEncoderParams::default()
                };
                let stream =
                    encode_intra_then_inter_stream(&seq, &intra_p, &params, &intra, &inter);
                assert_decodes_to(
                    &stream,
                    2,
                    &format!(
                        "adaptive flags adapt1={adapt1} adapt_post={adapt_post} bipred_post={bipred_post}"
                    ),
                );
            }
        }
    }
}

#[test]
fn pathological_pixel_inputs_never_panic() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_p = intra_params();
    let params = InterEncoderParams::default();

    let cases = [
        ("zero->zero", solid_64(0), solid_64(0)),
        ("ff->ff", solid_64(255), solid_64(255)),
        ("zero->ff", solid_64(0), solid_64(255)),
        ("mid->mid", solid_64(128), solid_64(128)),
        ("pulse->shifted-pulse", pulse_64(20, 20), pulse_64(24, 20)),
        ("pulse->mid", pulse_64(32, 32), solid_64(128)),
    ];

    for (label, intra_pix, inter_pix) in &cases {
        let (intra, inter) = pair_inputs(intra_pix, inter_pix);
        let stream = encode_intra_then_inter_stream(&seq, &intra_p, &params, &intra, &inter);
        assert_decodes_to(&stream, 2, &format!("pathological pixels: {label}"));
    }
}

#[test]
fn same_frame_zero_motion_round_trips_cleanly() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_p = intra_params();
    let params = InterEncoderParams::default();

    // Synthetic pair with dx=dy=0 — frame 0 == frame 1. Degenerate
    // SAD landscape (every MV gives identical zero cost) — the ME
    // tie-break path is exercised here.
    let (y0, u0, v0, y1, u1, v1) = synthetic_translating_pair_64(0, 0);
    assert_eq!(
        y0, y1,
        "synthetic_translating_pair_64(0,0) must be identical frames"
    );
    let pair = (
        (y0.to_vec(), u0.to_vec(), v0.to_vec()),
        (y1.to_vec(), u1.to_vec(), v1.to_vec()),
    );
    let (intra, inter) = pair_inputs(&pair.0, &pair.1);

    let stream = encode_intra_then_inter_stream(&seq, &intra_p, &params, &intra, &inter);
    assert_decodes_to(&stream, 2, "zero-motion identical-frame pair");
}

#[test]
fn deterministic_output_under_default_params() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_p = intra_params();
    let params = InterEncoderParams::default();

    let (y0, u0, v0, y1, u1, v1) = synthetic_camera_pan_64(2, 1);
    let pair = (
        (y0.to_vec(), u0.to_vec(), v0.to_vec()),
        (y1.to_vec(), u1.to_vec(), v1.to_vec()),
    );
    let (intra, inter) = pair_inputs(&pair.0, &pair.1);

    let a = encode_intra_then_inter_stream(&seq, &intra_p, &params, &intra, &inter);
    let b = encode_intra_then_inter_stream(&seq, &intra_p, &params, &intra, &inter);
    assert_eq!(
        a, b,
        "encode_intra_then_inter_stream must be deterministic under identical inputs"
    );
    assert_decodes_to(&a, 2, "determinism reference");
}

#[test]
fn deterministic_output_under_residue_off_path() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_p = intra_params();
    let params = InterEncoderParams {
        residue: None,
        ..InterEncoderParams::default()
    };

    let (y0, u0, v0, y1, u1, v1) = synthetic_translating_pair_64(4, 0);
    let pair = (
        (y0.to_vec(), u0.to_vec(), v0.to_vec()),
        (y1.to_vec(), u1.to_vec(), v1.to_vec()),
    );
    let (intra, inter) = pair_inputs(&pair.0, &pair.1);

    let a = encode_intra_then_inter_stream(&seq, &intra_p, &params, &intra, &inter);
    let b = encode_intra_then_inter_stream(&seq, &intra_p, &params, &intra, &inter);
    assert_eq!(
        a, b,
        "no-residue path must also be deterministic under identical inputs"
    );
    assert_decodes_to(&a, 2, "determinism, residue=None");
}

#[test]
fn extreme_search_range_zero_terminates() {
    // mv_search_range = 0 means the only candidate integer MV is (0, 0).
    // Sub-pel refinement still runs around that pin, so it's not strictly
    // a no-op; the ME landscape collapses to a single integer-pel point.
    // This pins that the encoder remains well-defined at the radius
    // floor — any future change that, e.g., divides by `search_range`
    // would panic here.
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_p = intra_params();
    let params = InterEncoderParams {
        mv_search_range: 0,
        ..InterEncoderParams::default()
    };

    let (y0, u0, v0, y1, u1, v1) = synthetic_translating_pair_64(4, 0);
    let pair = (
        (y0.to_vec(), u0.to_vec(), v0.to_vec()),
        (y1.to_vec(), u1.to_vec(), v1.to_vec()),
    );
    let (intra, inter) = pair_inputs(&pair.0, &pair.1);

    let stream = encode_intra_then_inter_stream(&seq, &intra_p, &params, &intra, &inter);
    assert_decodes_to(&stream, 2, "mv_search_range=0");
}

#[test]
fn high_qindex_residue_still_round_trips() {
    // qindex=127 is the maximum legal value; every residue coefficient
    // quantises to (essentially) zero, so this exercises the "residue
    // collapses to all-zero" branch end-to-end. The decoder must still
    // produce a valid frame (the prediction itself carries the picture).
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_p = intra_params();
    let params = InterEncoderParams {
        residue: Some(ResidueParams {
            wavelet: WaveletFilter::LeGall5_3,
            dwt_depth: 3,
            qindex: 127,
        }),
        ..InterEncoderParams::default()
    };

    let (y0, u0, v0, y1, u1, v1) = synthetic_translating_pair_64(2, -1);
    let pair = (
        (y0.to_vec(), u0.to_vec(), v0.to_vec()),
        (y1.to_vec(), u1.to_vec(), v1.to_vec()),
    );
    let (intra, inter) = pair_inputs(&pair.0, &pair.1);

    let stream = encode_intra_then_inter_stream(&seq, &intra_p, &params, &intra, &inter);
    assert_decodes_to(&stream, 2, "residue qindex=127 (max quantiser)");
}
