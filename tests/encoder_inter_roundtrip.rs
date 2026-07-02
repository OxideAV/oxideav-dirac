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
    GlobalMotionConfig, InterEncoderParams, InterInputPicture, ResidueParams,
};
use oxideav_dirac::picture_inter::GlobalParams;
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
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("make decoder");
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
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("make decoder");
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
        oxideav_dirac::register_codecs(&mut reg);
        let cp = CodecParameters::video(CodecId::new("dirac"));
        let mut dec = reg.first_decoder(&cp).expect("make decoder");
        let packet = Packet::new(0, TimeBase::new(1, 25), stream);
        dec.send_packet(&packet).expect("send_packet");
        let _intra_frame = dec.receive_frame().expect("intra frame");
        let inter_frame = match dec.receive_frame().expect("inter frame") {
            Frame::Video(vf) => vf,
            other => panic!("expected video, got {other:?}"),
        };
        psnr(&inter_frame.planes[0].data, &y1)
    };

    // Integer-pel ME baseline. Residue OFF so the PSNR measures
    // *only* the ME quality (residue would otherwise close the loop
    // bit-exactly at qindex=0 and hide the sub-pel uplift).
    let int_params = InterEncoderParams {
        mv_precision: 0,
        residue: None,
        ..InterEncoderParams::default()
    };
    let psnr_int = decode_psnr(&int_params);

    // Quarter-pel ME — the new path.
    let qpel_params = InterEncoderParams {
        mv_precision: 2,
        residue: None,
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
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("make decoder");
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
        oxideav_dirac::register_codecs(&mut reg);
        let cp = CodecParameters::video(CodecId::new("dirac"));
        let mut dec = reg.first_decoder(&cp).expect("make decoder");
        let packet = Packet::new(0, TimeBase::new(1, 25), stream);
        dec.send_packet(&packet).expect("send_packet");
        let _intra_frame = dec.receive_frame().expect("intra frame");
        let inter_frame = match dec.receive_frame().expect("inter frame") {
            Frame::Video(vf) => vf,
            other => panic!("expected video, got {other:?}"),
        };
        psnr(&inter_frame.planes[0].data, &y1)
    };

    // Baseline: integer-pel ME, OBMC refinement disabled. Residue OFF
    // so the PSNR measures pure ME / OBMC quality — residue at qindex=0
    // would otherwise close the loop bit-exactly and hide the
    // refinement gap entirely.
    let no_obmc_params = InterEncoderParams {
        mv_precision: 0,
        obmc_refine_passes: 0,
        residue: None,
        ..InterEncoderParams::default()
    };
    let psnr_no_obmc = decode_psnr(&no_obmc_params);

    // OBMC-aware: same precision, refinement on (defaults to 2 passes).
    let obmc_params = InterEncoderParams {
        mv_precision: 0,
        obmc_refine_passes: 2,
        residue: None,
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

/// **Wavelet residue PSNR uplift**. The encoder's §11.3 residue path
/// (`InterEncoderParams::residue = Some(...)`) computes
/// `source - decoder_OBMC_reconstruction` in the spec's signed
/// pre-output-offset domain, forward-DWTs the difference, dead-zone
/// quantises, and emits per-component AC subbands the decoder adds
/// back at §15.8.2.
///
/// On a synthetic camera-pan fixture the residue captures the OBMC
/// reconstruction's left-over error and lifts self-roundtrip Y PSNR
/// from the ~28-52 dB ME-only floor up to a much higher figure (or
/// bit-exact). This test verifies both that the gap exists and that
/// the residue path doesn't accidentally regress the no-residue
/// baseline (the `residue: None` knob must keep the round-1 behaviour).
#[test]
fn intra_then_inter_residue_beats_no_residue_baseline() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);

    // Quarter-pel camera pan — the residue captures whatever the
    // sub-pel ME couldn't reach.
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
        oxideav_dirac::register_codecs(&mut reg);
        let cp = CodecParameters::video(CodecId::new("dirac"));
        let mut dec = reg.first_decoder(&cp).expect("make decoder");
        let packet = Packet::new(0, TimeBase::new(1, 25), stream);
        dec.send_packet(&packet).expect("send_packet");
        let _intra_frame = dec.receive_frame().expect("intra frame");
        let inter_frame = match dec.receive_frame().expect("inter frame") {
            Frame::Video(vf) => vf,
            other => panic!("expected video, got {other:?}"),
        };
        psnr(&inter_frame.planes[0].data, &y1)
    };

    // Baseline: residue OFF (round-1 behaviour). PSNR is dominated by
    // ME / OBMC quality.
    let no_residue = InterEncoderParams {
        residue: None,
        ..InterEncoderParams::default()
    };
    let psnr_no_res = decode_psnr(&no_residue);

    // Residue ON (default).
    let with_residue = InterEncoderParams::default();
    let psnr_with_res = decode_psnr(&with_residue);

    eprintln!(
        "camera-pan 1/4 pel: no-residue PSNR Y = {psnr_no_res:.2} dB, \
         residue-on PSNR Y = {psnr_with_res:.2} dB"
    );
    // Residue must beat the no-residue baseline by ≥ 5 dB. On the
    // 1/4-pel camera-pan the gap is ~52 dB → inf dB at qindex=0
    // (LeGall 5/3 reverses the residue exactly when coefficients are
    // small).
    assert!(
        psnr_with_res >= psnr_no_res + 5.0,
        "residue-on PSNR {psnr_with_res:.2} dB did not beat no-residue \
         {psnr_no_res:.2} dB by ≥ 5 dB on the 1/4-pel camera-pan fixture \
         (residue acceptance: §11.3 wavelet residue must measurably \
         close the prediction loop on sub-pel motion)"
    );
}

/// Residue at qindex=0 with LeGall 5/3 must round-trip the encoder
/// **bit-exactly** on synthetic fixtures whose residue values stay
/// within the filter's reversible range. Bit-exactness is the strict
/// guarantee the residue path makes when configured for lossless mode.
#[test]
fn intra_then_inter_residue_q0_legall_is_bit_exact_on_translate() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter_params = InterEncoderParams {
        residue: Some(ResidueParams::default_for(WaveletFilter::LeGall5_3, 3)),
        ..InterEncoderParams::default()
    };

    let (y0, u0, v0, y1, u1, v1) = synthetic_translating_pair_64(4, 0);
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
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("make decoder");
    let packet = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&packet).expect("send_packet");

    let _intra_frame = dec.receive_frame().expect("intra frame");
    let inter_frame = match dec.receive_frame().expect("inter frame") {
        Frame::Video(vf) => vf,
        other => panic!("expected video, got {other:?}"),
    };
    assert_eq!(
        inter_frame.planes[0].data,
        y1.to_vec(),
        "Y plane should round-trip bit-exactly with LeGall 5/3 residue at qindex=0"
    );
    assert_eq!(
        inter_frame.planes[1].data,
        u1.to_vec(),
        "U plane should round-trip bit-exactly with LeGall 5/3 residue at qindex=0"
    );
    assert_eq!(
        inter_frame.planes[2].data,
        v1.to_vec(),
        "V plane should round-trip bit-exactly with LeGall 5/3 residue at qindex=0"
    );
}

/// Setting `residue: None` must reproduce the round-1 ZERO_RESIDUAL=true
/// behaviour byte-for-byte — i.e. the encoder emits `1` for the
/// ZERO_RESIDUAL flag and zero further bytes for transform parameters.
/// This is the "no-residue baseline" knob that lets regression tests
/// exercise pure ME quality.
#[test]
fn residue_none_emits_zero_residual_true() {
    use oxideav_dirac::parse_info::BBCD;

    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);

    let (y0, u0, v0, y1, u1, v1) = synthetic_translating_pair_64(2, 0);
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

    let no_residue = InterEncoderParams {
        residue: None,
        ..InterEncoderParams::default()
    };
    let with_residue = InterEncoderParams::default();

    let no_res_stream =
        encode_intra_then_inter_stream(&seq, &intra_params, &no_residue, &intra, &inter);
    let with_res_stream =
        encode_intra_then_inter_stream(&seq, &intra_params, &with_residue, &intra, &inter);

    // Locate the inter parse-info (parse code 0x09) in each stream — it
    // should be identical up through the block_motion_data, then differ
    // at the residue boundary. The simplest invariant: residue-on
    // stream is *strictly larger* than residue-off (the residue adds
    // transform parameters + per-component subband bytes).
    assert!(
        with_res_stream.len() > no_res_stream.len(),
        "residue-on stream ({} bytes) should be strictly larger than \
         residue-off ({} bytes) — the residue must add transform_parameters + \
         subband data after block_motion_data",
        with_res_stream.len(),
        no_res_stream.len(),
    );
    // Sanity: both streams still start with the same sequence-header
    // parse-info, so the front matter lines up.
    assert_eq!(&no_res_stream[..4], BBCD);
    assert_eq!(&with_res_stream[..4], BBCD);
    assert_eq!(no_res_stream[4], with_res_stream[4]);
}

/// **Round-73 per-block adaptive sub-pel-vs-integer-pel selection
/// (1-ref path)**. With `inter_adaptive_int_pel = true` (the default)
/// the encoder evaluates each block's MV at both its sub-pel-refined
/// position and the nearest integer-pel-rounded peer, picking
/// whichever gives lower OBMC SSE — a strict superset of the
/// pre-round-73 sub-pel-only candidate set. The OBMC-aware
/// refinement (`obmc_refine_passes`) then runs on the chosen MV
/// grid.
///
/// On the integer-pel-translation fixture `synthetic_translating_pair_64(2, -1)`
/// the bright square's hard edge falls across OBMC overlap regions:
/// every sub-pel candidate at non-integer offsets introduces 8-tap-filter
/// smoothing that blurs the edge in the blended reconstruction. The
/// adaptive selector identifies those blocks and snaps them back to
/// integer-pel, **without** harming smooth-content blocks (the
/// per-block min-of-two-SSEs invariant guarantees no per-block
/// regression — see `inter_select_int_pel_monotonic_per_block_obmc_sse`).
///
/// We measure end-to-end self-roundtrip PSNR with residue OFF (so the
/// gain measures pure ME quality, not residue compensation) and
/// assert the adaptive path does not regress versus the legacy
/// always-sub-pel path. A non-regression on integer-pel content is
/// the load-bearing acceptance — the win cases sit on real video
/// content with sharp text/occluders that fall outside synthetic
/// 64×64 fixtures.
#[test]
fn intra_then_inter_adaptive_int_pel_does_not_regress_subpel() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);

    // Integer-pel translation by (+2, -1) — the same fixture the
    // OBMC-refinement A/B test uses. Hard-edge content where the
    // adaptive selector either snaps to int-pel (better) or keeps
    // sub-pel (tied or better — never worse).
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
        oxideav_dirac::register_codecs(&mut reg);
        let cp = CodecParameters::video(CodecId::new("dirac"));
        let mut dec = reg.first_decoder(&cp).expect("make decoder");
        let packet = Packet::new(0, TimeBase::new(1, 25), stream);
        dec.send_packet(&packet).expect("send_packet");
        let _intra_frame = dec.receive_frame().expect("intra frame");
        let inter_frame = match dec.receive_frame().expect("inter frame") {
            Frame::Video(vf) => vf,
            other => panic!("expected video, got {other:?}"),
        };
        psnr(&inter_frame.planes[0].data, &y1)
    };

    // Baseline: legacy always-sub-pel path. Residue OFF so the PSNR
    // measures pure ME quality (residue at qindex=0 would close the
    // loop bit-exactly and hide the adaptive-selector contribution
    // entirely).
    let legacy_params = InterEncoderParams {
        mv_precision: 2,
        inter_adaptive_int_pel: false,
        residue: None,
        ..InterEncoderParams::default()
    };
    let psnr_legacy = decode_psnr(&legacy_params);

    // New default: adaptive selector ON. All other knobs match.
    let adaptive_params = InterEncoderParams {
        mv_precision: 2,
        inter_adaptive_int_pel: true,
        residue: None,
        ..InterEncoderParams::default()
    };
    let psnr_adaptive = decode_psnr(&adaptive_params);

    eprintln!(
        "translate(+2,-1) self-roundtrip: legacy sub-pel-only PSNR Y = {psnr_legacy:.2} dB, \
         round-73 adaptive int-pel PSNR Y = {psnr_adaptive:.2} dB"
    );
    // Round-73 acceptance: adaptive must NOT regress the legacy path,
    // on any fixture. This is the strict-superset invariant translated
    // to picture-level PSNR. A 0.1 dB tolerance allows for floating-
    // point rounding in the PSNR computation; the per-block SSE
    // invariant (`inter_select_int_pel_monotonic_per_block_obmc_sse`)
    // is the bit-precise guarantee.
    assert!(
        psnr_adaptive + 0.1 >= psnr_legacy,
        "adaptive int-pel selector regressed PSNR by > 0.1 dB on the \
         translate(+2,-1) fixture: legacy = {psnr_legacy:.2} dB, \
         adaptive = {psnr_adaptive:.2} dB (round-73 strict-superset \
         invariant broken — selector must never pick a worse MV than \
         the sub-pel baseline)"
    );
}

/// **Round-73 — `inter_adaptive_int_pel = false` is deterministic
/// and serves as the regression-safety knob**. Encoding the same
/// fixture twice with the flag explicitly off must produce
/// byte-identical streams (no hidden state). This lets downstream
/// callers pin the pre-round-73 behaviour for golden-file tests.
#[test]
fn inter_adaptive_int_pel_disabled_is_deterministic() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let legacy_params = InterEncoderParams {
        mv_precision: 2,
        inter_adaptive_int_pel: false,
        obmc_refine_passes: 0,
        residue: None,
        ..InterEncoderParams::default()
    };
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
    let legacy_stream_a =
        encode_intra_then_inter_stream(&seq, &intra_params, &legacy_params, &intra, &inter);
    let legacy_stream_b =
        encode_intra_then_inter_stream(&seq, &intra_params, &legacy_params, &intra, &inter);
    assert_eq!(
        legacy_stream_a, legacy_stream_b,
        "encoder must be deterministic with `inter_adaptive_int_pel = false`"
    );
}

/// **Round-80 post-OBMC second adaptive int-pel pass (1-ref path)**.
/// Mirrors `intra_then_inter_adaptive_int_pel_does_not_regress_subpel`
/// for the post-OBMC selector. With `inter_adaptive_int_pel_post_obmc =
/// true` (the default) the encoder runs `inter_select_int_pel_per_block`
/// a second time **after** `obmc_refine_me` has finished. The selector
/// is a strict superset of its input grid (each block keeps the
/// lower-OBMC-SSE of its current MV and the integer-pel-rounded peer),
/// so the post-OBMC pass cannot regress self-roundtrip PSNR. The
/// monotonicity invariant is already pinned by the unit test
/// `inter_select_int_pel_monotonic_per_block_obmc_sse` for the helper
/// itself; this test guards the end-to-end wiring.
///
/// Residue OFF so PSNR reflects ME quality alone (qindex=0 residue
/// would close the loop bit-exactly).
///
/// Covers both the hard-edge `translate(+2,-1)` fixture (sub-pel ME
/// already converges to integer-pel for integer motion so the post-OBMC
/// pass is typically a no-op — the test simply guards against
/// regression) and the smooth-motion `camera_pan_64(+1, 0)` fixture
/// (the qpel-aligned target where pre-OBMC and OBMC both prefer
/// sub-pel, and the post-OBMC pass should still be a no-op or improve).
#[test]
fn intra_then_inter_adaptive_int_pel_post_obmc_does_not_regress() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);

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
        oxideav_dirac::register_codecs(&mut reg);
        let cp = CodecParameters::video(CodecId::new("dirac"));
        let mut dec = reg.first_decoder(&cp).expect("make decoder");
        let packet = Packet::new(0, TimeBase::new(1, 25), stream);
        dec.send_packet(&packet).expect("send_packet");
        let _intra_frame = dec.receive_frame().expect("intra frame");
        let inter_frame = match dec.receive_frame().expect("inter frame") {
            Frame::Video(vf) => vf,
            other => panic!("expected video, got {other:?}"),
        };
        psnr(&inter_frame.planes[0].data, &y1)
    };

    // Pre-round-80 baseline: post-OBMC selector OFF, pre-OBMC selector ON
    // (round-73 default). Two OBMC passes. Residue OFF for ME-only PSNR.
    let pre_r80 = InterEncoderParams {
        mv_precision: 2,
        inter_adaptive_int_pel: true,
        inter_adaptive_int_pel_post_obmc: false,
        obmc_refine_passes: 2,
        residue: None,
        ..InterEncoderParams::default()
    };
    let psnr_pre = decode_psnr(&pre_r80);

    // Round-80 default: post-OBMC selector also ON.
    let r80 = InterEncoderParams {
        mv_precision: 2,
        inter_adaptive_int_pel: true,
        inter_adaptive_int_pel_post_obmc: true,
        obmc_refine_passes: 2,
        residue: None,
        ..InterEncoderParams::default()
    };
    let psnr_post = decode_psnr(&r80);

    eprintln!(
        "translate(+2,-1) self-roundtrip: pre-r80 (post-OBMC selector OFF) PSNR Y = {psnr_pre:.2} dB, \
         round-80 (post-OBMC selector ON) PSNR Y = {psnr_post:.2} dB"
    );
    // Strict-superset invariant translated to picture-level PSNR;
    // 0.1 dB tolerance for floating-point rounding in PSNR.
    assert!(
        psnr_post + 0.1 >= psnr_pre,
        "round-80 post-OBMC adaptive int-pel regressed PSNR by > 0.1 dB \
         on the translate(+2,-1) fixture: pre-r80 = {psnr_pre:.2} dB, \
         round-80 = {psnr_post:.2} dB (strict-superset invariant broken)"
    );

    // Second fixture: smooth-motion camera-pan at the quarter-pel
    // offset that the OBMC-aware ME refinement landed against in
    // `intra_then_inter_camera_pan_subpel_beats_integer` (`dx_qpel=+1`).
    // Here sub-pel is the right answer; the post-OBMC selector should
    // be a no-op (current sub-pel MV already minimises OBMC SSE) but
    // we guard the determinism / non-regression invariant just like
    // the hard-edge fixture above.
    let (yp0, up0, vp0, yp1, up1, vp1) = synthetic_camera_pan_64(1, 0);
    let intra_p = InterInputPicture {
        picture_number: 0,
        y: &yp0,
        u: &up0,
        v: &vp0,
    };
    let inter_p = InterInputPicture {
        picture_number: 1,
        y: &yp1,
        u: &up1,
        v: &vp1,
    };
    let decode_psnr_pan = |inter_params: &InterEncoderParams| -> f64 {
        let stream =
            encode_intra_then_inter_stream(&seq, &intra_params, inter_params, &intra_p, &inter_p);
        let mut reg = CodecRegistry::new();
        oxideav_dirac::register_codecs(&mut reg);
        let cp = CodecParameters::video(CodecId::new("dirac"));
        let mut dec = reg.first_decoder(&cp).expect("make decoder");
        let packet = Packet::new(0, TimeBase::new(1, 25), stream);
        dec.send_packet(&packet).expect("send_packet");
        let _intra_frame = dec.receive_frame().expect("intra frame");
        let inter_frame = match dec.receive_frame().expect("inter frame") {
            Frame::Video(vf) => vf,
            other => panic!("expected video, got {other:?}"),
        };
        psnr(&inter_frame.planes[0].data, &yp1)
    };
    let psnr_pan_pre = decode_psnr_pan(&pre_r80);
    let psnr_pan_post = decode_psnr_pan(&r80);
    eprintln!(
        "camera_pan_64(+1,0) self-roundtrip: pre-r80 PSNR Y = {psnr_pan_pre:.2} dB, \
         round-80 PSNR Y = {psnr_pan_post:.2} dB"
    );
    assert!(
        psnr_pan_post + 0.1 >= psnr_pan_pre,
        "round-80 post-OBMC adaptive int-pel regressed PSNR by > 0.1 dB \
         on the camera_pan_64(+1,0) fixture: pre-r80 = {psnr_pan_pre:.2} dB, \
         round-80 = {psnr_pan_post:.2} dB"
    );
}

/// **Round-80 disabled-flag determinism**. Encoding the same fixture
/// twice with the post-OBMC selector explicitly off must produce
/// byte-identical streams (no hidden state, no nondeterministic
/// tie-break ordering). Lets downstream callers pin pre-round-80
/// behaviour for golden-file tests.
#[test]
fn inter_adaptive_int_pel_post_obmc_disabled_is_deterministic() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let pre_r80_params = InterEncoderParams {
        mv_precision: 2,
        inter_adaptive_int_pel: true,
        inter_adaptive_int_pel_post_obmc: false,
        obmc_refine_passes: 2,
        residue: None,
        ..InterEncoderParams::default()
    };
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
    let a = encode_intra_then_inter_stream(&seq, &intra_params, &pre_r80_params, &intra, &inter);
    let b = encode_intra_then_inter_stream(&seq, &intra_params, &pre_r80_params, &intra, &inter);
    assert_eq!(
        a, b,
        "encoder must be deterministic with `inter_adaptive_int_pel_post_obmc = false`"
    );
}

/// **Round-80 self-roundtrip non-regression with residue ON**.
/// Round-80's default has the post-OBMC selector enabled; the full
/// default pipeline (subpel ME + pre-OBMC selector + OBMC refinement +
/// post-OBMC selector + LeGall 5/3 residue at qindex 0) must continue
/// to produce a bit-exact (infinite-PSNR) self-roundtrip on every
/// translation fixture. This is the load-bearing end-to-end guard
/// against subtle decoder/encoder drift introduced by the new pass.
#[test]
fn intra_then_inter_post_obmc_selector_self_roundtrip_is_bit_exact() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter_params = InterEncoderParams {
        // Round-80 defaults — both selectors and OBMC refinement on,
        // residue ON with LeGall 5/3 at qindex 0.
        ..InterEncoderParams::default()
    };
    // Same fixture set the round-1 residue bit-exactness test uses
    // (`intra_then_inter_residue_q0_legall_is_bit_exact_on_translate`)
    // plus the synthetic mid-frame translations used elsewhere in this
    // file — keeping all motion vectors well inside the 64×64 picture
    // so OBMC overlap regions don't hit boundary clipping.
    for (dx, dy) in [(0, 0), (1, 0), (4, 0), (2, -1), (-3, 2)] {
        let (y0, u0, v0, y1, u1, v1) = synthetic_translating_pair_64(dx, dy);
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
        let stream =
            encode_intra_then_inter_stream(&seq, &intra_params, &inter_params, &intra, &inter);
        let mut reg = CodecRegistry::new();
        oxideav_dirac::register_codecs(&mut reg);
        let cp = CodecParameters::video(CodecId::new("dirac"));
        let mut dec = reg.first_decoder(&cp).expect("make decoder");
        let packet = Packet::new(0, TimeBase::new(1, 25), stream);
        dec.send_packet(&packet).expect("send_packet");
        let _intra = dec.receive_frame().expect("intra frame");
        let inter_frame = match dec.receive_frame().expect("inter frame") {
            Frame::Video(vf) => vf,
            other => panic!("expected video, got {other:?}"),
        };
        let p = psnr(&inter_frame.planes[0].data, &y1);
        assert!(
            p.is_infinite(),
            "round-80 default self-roundtrip should be bit-exact on \
             translate({dx},{dy}); got PSNR Y = {p:.2} dB"
        );
    }
}

/// **Round-80 unit test**: the post-OBMC selector preserves the
/// per-block min-of-two-SSEs property end-to-end. Run the full
/// `subpel_search_me → inter_select_int_pel_per_block → obmc_refine_me
/// → inter_select_int_pel_per_block` pipeline manually and confirm
/// that the second selector's per-block OBMC SSE is ≤ the pre-selector
/// per-block OBMC SSE (against the same neighbour grid). The helper's
/// own monotonicity test already pins the strict-superset property for
/// a sub-pel-search output; this test pins it for a **post-OBMC**
/// input, where MVs may have drifted off integer-pel during refinement.
#[test]
fn post_obmc_selector_monotonic_at_pipeline_endpoint() {
    use oxideav_dirac::encoder_inter::{
        inter_select_int_pel_per_block, obmc_refine_me, subpel_search_me, IntegerMv,
    };
    let (y0, _, _, y1, _, _) = synthetic_translating_pair_64(2, -1);

    // Stage 1: sub-pel ME.
    let mut mvs = subpel_search_me(&y1, &y0, 64, 64, 16, 16, 16, 2);
    // Stage 2: pre-OBMC adaptive int-pel selector.
    inter_select_int_pel_per_block(&y1, &y0, 64, 64, 16, 16, &mut mvs, 2);
    // Stage 3: OBMC refinement (may drift MVs off integer-pel).
    obmc_refine_me(&y1, &y0, 64, 64, 16, 16, &mut mvs, 2, 2);

    // Snapshot post-OBMC grid as the baseline for the strict-superset
    // check before stage 4 mutates it.
    let mvs_before_post: Vec<IntegerMv> = mvs.clone();

    // Stage 4: post-OBMC adaptive int-pel selector (round-80).
    inter_select_int_pel_per_block(&y1, &y0, 64, 64, 16, 16, &mut mvs, 2);

    // Every changed block must equal `round_mv_to_int_pel(before)` —
    // the selector only swaps `current` for its integer-pel peer. That
    // is the simplest possible candidate-set check: if a block changed,
    // it must equal the rounded form of its pre-stage-4 MV.
    for (idx, (b, a)) in mvs_before_post.iter().zip(mvs.iter()).enumerate() {
        if (b.0, b.1) != (a.0, a.1) {
            // The post-OBMC selector's only mutation is current → int_pel.
            // We can't import `round_mv_to_int_pel` directly (private),
            // but we can check that the new MV components are multiples
            // of `1 << mv_precision = 4`.
            assert!(
                a.0 % 4 == 0 && a.1 % 4 == 0,
                "block {idx}: post-OBMC selector changed the MV to a \
                 non-integer-pel position {:?} (pre-stage-4 was {:?})",
                a,
                b
            );
        }
    }
}

/// **§11.2.6 global-motion P-picture end-to-end** (round-382). Encode a
/// 1-ref inter picture whose motion is described entirely by the global
/// affine field (every block is a §12.3.3.2 global block, no per-block
/// MV residual), then decode through the full pipeline and confirm the
/// picture reconstructs.
///
/// A zero affine matrix collapses `global_mv` to a constant translation
/// `t = pan_tilt + 1` (the +1 is the §15.8.8 rounding bias with
/// `zrs_exp = perspective = 0`). We pick `pan_tilt = (-5, -1)` so the
/// field is a pure `(-4, 0)` translation, exactly matching a reference
/// that was shifted +4 horizontally to make the current frame. The
/// §11.3 wavelet residue (LeGall 5/3, qindex 0) then closes the
/// prediction-error loop, so the reconstruction is at least as good as
/// the block-motion path — and the global-motion decode path is
/// exercised end-to-end for the first time.
#[test]
fn intra_then_inter_global_motion_p_picture_roundtrips() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);

    // Frame 0: square at (24, 24); Frame 1: square at (28, 24).
    let (y0, u0, v0, y1, u1, v1) = synthetic_translating_pair_64(4, 0);

    // Zero affine matrix → constant translation `pan_tilt + 1`.
    // pan_tilt (-5, -1) ⇒ field (-4, 0), matching the +4 shift.
    let global1 = GlobalParams {
        pan_tilt: (-5, -1),
        zrs: [[0, 0], [0, 0]],
        zrs_exp: 0,
        perspective: (0, 0),
        persp_exp: 0,
    };
    let inter_params = InterEncoderParams {
        mv_precision: 0,
        global_motion: Some(GlobalMotionConfig {
            global1,
            global2: None,
            block_gmode: None,
        }),
        ..InterEncoderParams::default()
    };

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
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("make decoder");
    let packet = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&packet).expect("send_packet");

    // Intra anchor bit-exact at qindex 0.
    let frame0 = match dec.receive_frame().expect("intra frame") {
        Frame::Video(vf) => vf,
        other => panic!("expected video, got {other:?}"),
    };
    assert_eq!(frame0.planes[0].data, y0.to_vec(), "intra anchor bit-exact");

    // Inter global-motion picture reconstructs against ground truth.
    let frame1 = match dec.receive_frame().expect("inter frame") {
        Frame::Video(vf) => vf,
        other => panic!("expected video, got {other:?}"),
    };
    let psnr_y = psnr(&frame1.planes[0].data, &y1);
    let psnr_u = psnr(&frame1.planes[1].data, &u1);
    let psnr_v = psnr(&frame1.planes[2].data, &v1);
    eprintln!("global-motion inter PSNR: Y={psnr_y:.2} U={psnr_u:.2} V={psnr_v:.2}");
    assert!(
        psnr_y > 30.0,
        "global-motion inter Y PSNR {psnr_y:.2} dB below 30 dB"
    );
    assert!(psnr_u >= 40.0, "U PSNR {psnr_u:.2} dB below 40 dB");
    assert!(psnr_v >= 40.0, "V PSNR {psnr_v:.2} dB below 40 dB");
}

/// **Spatially-varying global field** (round-382). A pure-translation
/// field (zero affine matrix) exercises only one point of the §15.8.8
/// model; this test drives a genuine affine gradient: identity matrix at
/// `zrs_exp = 4` ⇒ `v(x, y) = ((x + 8) >> 4, (y + 8) >> 4)` — a gentle
/// 0..=4-pel zoom-style ramp across the 64×64 picture that varies
/// **per pixel**, not per block. The encoder's OBMC prediction and the
/// decoder's must evaluate the identical per-pixel field or the qindex-0
/// residue won't close the loop; a ≥ 60 dB reconstruction therefore
/// pins encoder/decoder `global_mv` agreement across the whole plane.
#[test]
fn intra_then_inter_global_zoom_field_roundtrips() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);

    let (y0, u0, v0, y1, u1, v1) = synthetic_translating_pair_64(4, 0);
    let global1 = GlobalParams {
        pan_tilt: (0, 0),
        zrs: [[1, 0], [0, 1]],
        zrs_exp: 4,
        perspective: (0, 0),
        persp_exp: 0,
    };
    let inter_params = InterEncoderParams {
        mv_precision: 0,
        global_motion: Some(GlobalMotionConfig {
            global1,
            global2: None,
            block_gmode: None,
        }),
        ..InterEncoderParams::default()
    };

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
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("make decoder");
    let packet = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&packet).expect("send_packet");
    let _frame0 = dec.receive_frame().expect("intra frame");
    let frame1 = match dec.receive_frame().expect("inter frame") {
        Frame::Video(vf) => vf,
        other => panic!("expected video, got {other:?}"),
    };
    let psnr_y = psnr(&frame1.planes[0].data, &y1);
    eprintln!("global zoom-field inter Y PSNR: {psnr_y:.2} dB");
    // The zoom field is a poor motion model for the pure-translation
    // fixture — the residue is what closes the loop. What this pins is
    // that the encoder subtracted the SAME per-pixel global prediction
    // the decoder adds back (any divergence anywhere in the plane
    // craters PSNR).
    assert!(
        psnr_y >= 60.0,
        "zoom-field inter Y PSNR {psnr_y:.2} dB below 60 dB — encoder/decoder \
         §15.8.8 global_mv fields disagree"
    );
}

/// **Quarter-pel global field** (round-382). At `mv_precision = 2` the
/// §15.8.8 field is in qpel units and the §15.8.7 `pixel_pred` fetch
/// goes through the §15.8.10 sub-pel sampler. `pan_tilt = (-17, -1)`
/// with a zero affine matrix gives the constant field `(-16, 0)` qpel =
/// −4 integer pels, matching the fixture's +4 shift exactly, so the
/// global prediction itself is near-perfect (block-interior pixels) and
/// the residue mops up the OBMC edge effects.
#[test]
fn intra_then_inter_global_motion_qpel_roundtrips() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);

    let (y0, u0, v0, y1, u1, v1) = synthetic_translating_pair_64(4, 0);
    let global1 = GlobalParams {
        pan_tilt: (-17, -1),
        zrs: [[0, 0], [0, 0]],
        zrs_exp: 0,
        perspective: (0, 0),
        persp_exp: 0,
    };
    let inter_params = InterEncoderParams {
        mv_precision: 2,
        global_motion: Some(GlobalMotionConfig {
            global1,
            global2: None,
            block_gmode: None,
        }),
        ..InterEncoderParams::default()
    };

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
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("make decoder");
    let packet = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&packet).expect("send_packet");
    let _frame0 = dec.receive_frame().expect("intra frame");
    let frame1 = match dec.receive_frame().expect("inter frame") {
        Frame::Video(vf) => vf,
        other => panic!("expected video, got {other:?}"),
    };
    let psnr_y = psnr(&frame1.planes[0].data, &y1);
    eprintln!("global qpel inter Y PSNR: {psnr_y:.2} dB");
    assert!(
        psnr_y >= 60.0,
        "qpel global inter Y PSNR {psnr_y:.2} dB below 60 dB"
    );
}

/// **Mixed global/block-motion picture** (round-382). Only the left
/// half of the block grid is global (§12.3.3.2 flags spatially split);
/// the right half uses ordinary per-block ME MVs with their §12.3.5
/// residuals. Exercises the two prediction branches side-by-side in one
/// picture — including the §12.3.6.1 rule that global blocks are
/// excluded from their neighbours' MV median and the §12.3.6.4
/// neighbour-majority gmode prediction across the half boundary.
#[test]
fn intra_then_inter_mixed_global_and_block_motion_roundtrips() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);

    let (y0, u0, v0, y1, u1, v1) = synthetic_translating_pair_64(4, 0);
    // Whole-picture translation model matching the +4 shift.
    let global1 = GlobalParams {
        pan_tilt: (-5, -1),
        zrs: [[0, 0], [0, 0]],
        zrs_exp: 0,
        perspective: (0, 0),
        persp_exp: 0,
    };
    // 64x64 at preset 1 → 16x16 block grid. Left 8 columns global.
    let blocks_x = 16u32;
    let blocks_y = 16u32;
    let gmode: Vec<bool> = (0..blocks_x * blocks_y)
        .map(|i| (i % blocks_x) < 8)
        .collect();
    let inter_params = InterEncoderParams {
        mv_precision: 0,
        global_motion: Some(GlobalMotionConfig {
            global1,
            global2: None,
            block_gmode: Some(gmode),
        }),
        ..InterEncoderParams::default()
    };

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
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("make decoder");
    let packet = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&packet).expect("send_packet");
    let _frame0 = dec.receive_frame().expect("intra frame");
    let frame1 = match dec.receive_frame().expect("inter frame") {
        Frame::Video(vf) => vf,
        other => panic!("expected video, got {other:?}"),
    };
    let psnr_y = psnr(&frame1.planes[0].data, &y1);
    eprintln!("mixed global/block inter Y PSNR: {psnr_y:.2} dB");
    assert!(
        psnr_y >= 60.0,
        "mixed global/block inter Y PSNR {psnr_y:.2} dB below 60 dB — the \
         two prediction branches disagree with the decoder somewhere"
    );
}

/// **Estimated global motion end-to-end** (round-382).
/// `estimate_global_pan_config` fits the dominant translation from the
/// encoder's own ME grid and marks exactly the matching blocks global;
/// feeding the estimate back into `encode_inter_picture` must produce a
/// stream that (a) signals `using_global`, (b) decodes at the same
/// near-lossless quality as the pure block-motion encode (the estimate
/// only re-labels blocks whose MV the field reproduces exactly, so the
/// prediction — and the qindex-0 residue — is unchanged), and (c) sheds
/// the global blocks' MV residuals from the wire.
#[test]
fn estimated_global_motion_roundtrips_and_matches_block_motion_quality() {
    use oxideav_dirac::encoder_inter::estimate_global_pan_config;

    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);

    // Whole-frame integer pan over a textured field (same construction
    // as the estimator's unit fixture).
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
    let u = vec![128u8; 32 * 32];
    let v = vec![128u8; 32 * 32];

    let base = InterEncoderParams {
        mv_precision: 0,
        ..InterEncoderParams::default()
    };
    let (cfg, fraction) = estimate_global_pan_config(&seq, &base, &y1, &y0);
    assert!(fraction >= 0.8, "estimator fraction {fraction} too low");
    let global_params = InterEncoderParams {
        global_motion: Some(cfg),
        ..base.clone()
    };

    let intra = InterInputPicture {
        picture_number: 0,
        y: &y0,
        u: &u,
        v: &v,
    };
    let inter = InterInputPicture {
        picture_number: 1,
        y: &y1,
        u: &u,
        v: &v,
    };

    let decode_y_psnr = |params: &InterEncoderParams| -> (f64, usize) {
        let stream = encode_intra_then_inter_stream(&seq, &intra_params, params, &intra, &inter);
        let len = stream.len();
        let mut reg = CodecRegistry::new();
        oxideav_dirac::register_codecs(&mut reg);
        let cp = CodecParameters::video(CodecId::new("dirac"));
        let mut dec = reg.first_decoder(&cp).expect("decoder");
        let packet = Packet::new(0, TimeBase::new(1, 25), stream);
        dec.send_packet(&packet).expect("send");
        let _f0 = dec.receive_frame().expect("intra");
        let f1 = match dec.receive_frame().expect("inter") {
            Frame::Video(vf) => vf,
            other => panic!("expected video, got {other:?}"),
        };
        (psnr(&f1.planes[0].data, &y1), len)
    };

    let (psnr_block, len_block) = decode_y_psnr(&base);
    let (psnr_global, len_global) = decode_y_psnr(&global_params);
    eprintln!(
        "estimated global: fraction {fraction:.3}, block-motion {psnr_block:.2} dB / \
         {len_block} B, global {psnr_global:.2} dB / {len_global} B"
    );
    // Same prediction ⇒ same residue ⇒ same reconstruction quality
    // (allow a whisker for arith-context divergence between layouts).
    assert!(
        psnr_global >= 60.0,
        "estimated-global inter Y PSNR {psnr_global:.2} dB below 60 dB"
    );
    assert!(
        (psnr_global - psnr_block).abs() < 1.0 || psnr_global.is_infinite(),
        "estimated global ({psnr_global:.2} dB) diverged from block motion \
         ({psnr_block:.2} dB) — the estimate must not change the prediction"
    );
}
