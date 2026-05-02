//! Inter-encoder end-to-end validator.
//!
//! Encode a 2-picture stream — HQ intra reference + 1-ref core-syntax
//! inter — through `oxideav_dirac::encoder_inter::encode_intra_then_inter_stream`,
//! pipe through our own decoder, and measure the inter frame's PSNR
//! against the input. Round 1 targets a *much* better PSNR than
//! all-intra would give at the same residue (zero) for any real motion,
//! since the inter encoder transmits actual MVs.

use oxideav_core::CodecRegistry;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_dirac::encoder::{make_minimal_sequence, EncoderParams};
use oxideav_dirac::encoder_inter::{
    encode_intra_then_inter_stream, synthetic_camera_pan_64, synthetic_translating_pair_64,
    InterEncoderParams, InterInputPicture,
};
use oxideav_dirac::video_format::ChromaFormat;
use oxideav_dirac::wavelet::WaveletFilter;

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

/// Drive 2 pictures through the encoder/decoder loop. The intra
/// reference round-trips bit-exact (qindex=0); the inter frame's PSNR
/// against ground truth measures how well the OBMC predictions
/// reconstruct the translating square (with zero wavelet residue).
#[test]
fn intra_then_inter_translate_4_pixels_yields_high_psnr() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter_params = InterEncoderParams::default();

    // Frame 0: square at (24, 24), Frame 1: square at (24, 28).
    let (y0, u0, v0, y1, u1, v1) = synthetic_translating_pair_64(4, 0);
    let intra = InterInputPicture {
        picture_number: 10,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let inter = InterInputPicture {
        picture_number: 11,
        y: &y1,
        u: &u1,
        v: &v1,
    };
    let stream = encode_intra_then_inter_stream(&seq, &intra_params, &inter_params, &intra, &inter);

    let mut reg = CodecRegistry::new();
    oxideav_dirac::register(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.make_decoder(&cp).expect("make decoder");
    let packet = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&packet).expect("send_packet");

    // First frame: intra — should round-trip bit-exact at qindex 0.
    let frame0 = match dec.receive_frame().expect("intra frame") {
        Frame::Video(vf) => vf,
        other => panic!("expected video frame, got {other:?}"),
    };
    assert_eq!(
        frame0.planes[0].data,
        y0.to_vec(),
        "intra Y plane should be bit-exact at qindex 0"
    );

    // Second frame: inter — measure PSNR against the **encoder input**.
    let frame1 = match dec.receive_frame().expect("inter frame") {
        Frame::Video(vf) => vf,
        other => panic!("expected video frame, got {other:?}"),
    };
    assert_eq!(frame1.planes[0].stride, 64);
    assert_eq!(frame1.planes[0].data.len() / frame1.planes[0].stride, 64);

    let psnr_y = psnr(&frame1.planes[0].data, &y1);
    let psnr_u = psnr(&frame1.planes[1].data, &u1);
    let psnr_v = psnr(&frame1.planes[2].data, &v1);
    eprintln!("inter frame PSNR: Y={psnr_y:.2} dB  U={psnr_u:.2} dB  V={psnr_v:.2} dB");

    // Translating-square w/ pure horizontal +4 shift, integer-pel MVs,
    // 8x8 blocks with 4-pel stride: ME should find the exact MV per
    // block; OBMC overlap then produces a near-bit-exact reconstruction
    // for blocks fully inside the picture. Edge blocks suffer from
    // edge clamping in the reference. Target: > 30 dB on Y.
    assert!(
        psnr_y > 30.0,
        "inter Y PSNR {psnr_y:.2} dB below 30 dB target"
    );
    // Chroma is constant in this fixture — perfect copy expected.
    assert!(
        psnr_u >= 40.0,
        "inter U PSNR {psnr_u:.2} dB below 40 dB threshold (chroma is constant!)"
    );
    assert!(
        psnr_v >= 40.0,
        "inter V PSNR {psnr_v:.2} dB below 40 dB threshold (chroma is constant!)"
    );
}

/// A second, slightly harder fixture: vertical translation. Confirms
/// the encoder isn't accidentally emitting only horizontal MVs (it
/// would still pass the +4 horizontal test if vertical handling were
/// silently broken).
#[test]
fn intra_then_inter_translate_vertical_yields_high_psnr() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter_params = InterEncoderParams::default();

    let (y0, u0, v0, y1, u1, v1) = synthetic_translating_pair_64(0, -4);
    let intra = InterInputPicture {
        picture_number: 0,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let inter = InterInputPicture {
        picture_number: 1,
        y: &y1,
        u: &u1,
        v: &v1,
    };
    let stream = encode_intra_then_inter_stream(&seq, &intra_params, &inter_params, &intra, &inter);

    let mut reg = CodecRegistry::new();
    oxideav_dirac::register(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.make_decoder(&cp).expect("make decoder");
    let packet = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&packet).expect("send_packet");

    let _intra_frame = dec.receive_frame().expect("intra frame");
    let inter_frame = match dec.receive_frame().expect("inter frame") {
        Frame::Video(vf) => vf,
        other => panic!("expected video, got {other:?}"),
    };
    let psnr_y = psnr(&inter_frame.planes[0].data, &y1);
    eprintln!("inter (vertical -4) PSNR Y: {psnr_y:.2} dB");
    assert!(
        psnr_y > 30.0,
        "inter Y PSNR {psnr_y:.2} dB below 30 dB target on vertical motion"
    );
}

/// **Sub-pel ME PSNR uplift** (#168). On a camera-pan fixture with
/// 1/4-pel horizontal motion (vertical-bar pattern smoothly translated
/// by 0.25 luma pels between frames), integer-pel ME bottoms out at the
/// nearest integer MV and leaves substantial residue. Quarter-pel ME
/// can lock onto the true motion and dramatically reduce that error.
///
/// We measure both modes back-to-back. Recording the integer-pel PSNR
/// as a floor: if quarter-pel doesn't beat it by a clear margin, the
/// sub-pel refinement isn't actually contributing.
#[test]
fn intra_then_inter_camera_pan_subpel_beats_integer() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);

    // 1/4-pel horizontal pan: one quarter-pel unit in x, none in y.
    let (y0, u0, v0, y1, u1, v1) = synthetic_camera_pan_64(1, 0);
    let intra = InterInputPicture {
        picture_number: 0,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let inter = InterInputPicture {
        picture_number: 1,
        y: &y1,
        u: &u1,
        v: &v1,
    };

    let decode_psnr = |inter_params: &InterEncoderParams| -> f64 {
        let stream =
            encode_intra_then_inter_stream(&seq, &intra_params, inter_params, &intra, &inter);
        let mut reg = CodecRegistry::new();
        oxideav_dirac::register(&mut reg);
        let cp = CodecParameters::video(CodecId::new("dirac"));
        let mut dec = reg.make_decoder(&cp).expect("make decoder");
        let packet = Packet::new(0, TimeBase::new(1, 25), stream);
        dec.send_packet(&packet).expect("send_packet");
        let _intra_frame = dec.receive_frame().expect("intra frame");
        let inter_frame = match dec.receive_frame().expect("inter frame") {
            Frame::Video(vf) => vf,
            other => panic!("expected video, got {other:?}"),
        };
        psnr(&inter_frame.planes[0].data, &y1)
    };

    // Integer-pel ME baseline.
    let int_params = InterEncoderParams {
        mv_precision: 0,
        ..InterEncoderParams::default()
    };
    let psnr_int = decode_psnr(&int_params);

    // Quarter-pel ME — the new path.
    let qpel_params = InterEncoderParams {
        mv_precision: 2,
        ..InterEncoderParams::default()
    };
    let psnr_qpel = decode_psnr(&qpel_params);

    eprintln!(
        "camera-pan 1/4 pel: integer-pel PSNR Y = {psnr_int:.2} dB, \
         quarter-pel PSNR Y = {psnr_qpel:.2} dB"
    );
    // Quarter-pel must beat integer by at least 5 dB on a fixture with
    // explicit sub-pel motion. Anything less means the sub-pel
    // refinement isn't actually finding fractional MVs (or the OBMC
    // path is throwing the gain away).
    assert!(
        psnr_qpel >= psnr_int + 5.0,
        "quarter-pel ME PSNR {psnr_qpel:.2} dB did not beat \
         integer-pel {psnr_int:.2} dB by ≥ 5 dB on the 1/4-pel pan \
         fixture (#168 acceptance: sub-pel ME must measurably reduce \
         residue on sub-pel motion)"
    );
}

/// Sanity: an inter frame with **no motion** (frame 1 = frame 0) should
/// reconstruct nearly bit-exactly via OBMC, since every block's best
/// MV is (0, 0) and the OBMC of an unshifted reference is the
/// reference itself (modulo edge effects).
#[test]
fn intra_then_inter_zero_motion_is_near_lossless() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter_params = InterEncoderParams::default();

    let (y0, u0, v0, ..) = synthetic_translating_pair_64(0, 0);
    let intra = InterInputPicture {
        picture_number: 0,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let inter = InterInputPicture {
        picture_number: 1,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let stream = encode_intra_then_inter_stream(&seq, &intra_params, &inter_params, &intra, &inter);

    let mut reg = CodecRegistry::new();
    oxideav_dirac::register(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.make_decoder(&cp).expect("make decoder");
    let packet = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&packet).expect("send_packet");

    let _intra_frame = dec.receive_frame().expect("intra frame");
    let inter_frame = match dec.receive_frame().expect("inter frame") {
        Frame::Video(vf) => vf,
        other => panic!("expected video, got {other:?}"),
    };
    let psnr_y = psnr(&inter_frame.planes[0].data, &y0);
    eprintln!("inter (zero-motion) PSNR Y: {psnr_y:.2} dB");
    // Without residual, OBMC at the edges still smooths the boundary —
    // not exactly bit-exact but should comfortably exceed 35 dB.
    assert!(
        psnr_y > 35.0,
        "zero-motion inter Y PSNR {psnr_y:.2} dB below 35 dB threshold"
    );
}

/// **OBMC-aware ME PSNR uplift** (#186). The encoder's §15.8.6
/// OBMC-aware ME refinement (`obmc_refine_passes`) iteratively
/// improves a starting MV grid by, for each block, scoring the 8 sub-pel
/// neighbours of its current MV via the *blended* per-block reconstruction
/// the decoder will perform — keeping whichever MV minimises the
/// per-block SSE against the source.
///
/// On translating-square fixtures with motion that intersects the OBMC
/// overlap region, the refinement converges on an MV grid the per-block
/// SAD search misses, lifting self-roundtrip Y PSNR by ≥ 5 dB. Setting
/// `obmc_refine_passes = 0` reverts to the pre-#186 hard-block SAD
/// output and is the explicit "no-OBMC" baseline this test measures
/// against.
#[test]
fn intra_then_inter_obmc_refinement_beats_no_obmc_baseline() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);

    // Translation by (+2, -1) luma pels — small enough that the
    // bright square's edge falls inside several blocks' OBMC overlap
    // regions, exposing the hard-block SAD's blind spot. Per-block
    // SAD picks zero-MV for nearby background blocks while the block
    // hosting the square's corner picks the full -2/+1 — leaving an
    // overlap region with two competing MVs and noisy reconstruction.
    // OBMC-aware refinement converges all four neighbours on the same
    // MV and reproduces the translation cleanly.
    let (y0, u0, v0, y1, u1, v1) = synthetic_translating_pair_64(2, -1);
    let intra = InterInputPicture {
        picture_number: 0,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let inter = InterInputPicture {
        picture_number: 1,
        y: &y1,
        u: &u1,
        v: &v1,
    };

    let decode_psnr = |inter_params: &InterEncoderParams| -> f64 {
        let stream =
            encode_intra_then_inter_stream(&seq, &intra_params, inter_params, &intra, &inter);
        let mut reg = CodecRegistry::new();
        oxideav_dirac::register(&mut reg);
        let cp = CodecParameters::video(CodecId::new("dirac"));
        let mut dec = reg.make_decoder(&cp).expect("make decoder");
        let packet = Packet::new(0, TimeBase::new(1, 25), stream);
        dec.send_packet(&packet).expect("send_packet");
        let _intra_frame = dec.receive_frame().expect("intra frame");
        let inter_frame = match dec.receive_frame().expect("inter frame") {
            Frame::Video(vf) => vf,
            other => panic!("expected video, got {other:?}"),
        };
        psnr(&inter_frame.planes[0].data, &y1)
    };

    // Baseline: integer-pel ME, OBMC refinement disabled.
    let no_obmc_params = InterEncoderParams {
        mv_precision: 0,
        obmc_refine_passes: 0,
        ..InterEncoderParams::default()
    };
    let psnr_no_obmc = decode_psnr(&no_obmc_params);

    // OBMC-aware: same precision, refinement on (defaults to 2 passes).
    let obmc_params = InterEncoderParams {
        mv_precision: 0,
        obmc_refine_passes: 2,
        ..InterEncoderParams::default()
    };
    let psnr_obmc = decode_psnr(&obmc_params);

    eprintln!(
        "translate(+2,-1) self-roundtrip: no-OBMC PSNR Y = {psnr_no_obmc:.2} dB, \
         OBMC-aware PSNR Y = {psnr_obmc:.2} dB"
    );
    // #186 acceptance: OBMC-aware refinement must beat the no-OBMC
    // baseline by at least 5 dB on a fixture whose motion straddles
    // the OBMC overlap region. The actual gap on translate(+2,-1) is
    // ~22 dB (32 dB → inf dB on this fixture); 5 dB is the margin
    // the brief calls for.
    assert!(
        psnr_obmc >= psnr_no_obmc + 5.0,
        "OBMC-aware ME PSNR {psnr_obmc:.2} dB did not beat no-OBMC \
         baseline {psnr_no_obmc:.2} dB by ≥ 5 dB on the translate(+2,-1) \
         fixture (#186 acceptance: encoder-side OBMC blend must converge \
         the per-block MV grid the §15.8.5 weighted-sum reconstructor \
         actually reads)"
    );
}
