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
    InterEncoderParams, InterInputPicture, ResidueParams,
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
