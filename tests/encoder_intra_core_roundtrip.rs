//! Core-syntax intra encoder end-to-end validators.
//!
//! Two stream shapes are exercised:
//!
//! * **Intra-only** (`encode_single_core_intra_stream`) — confirm the
//!   core-syntax intra reference round-trips through our own decoder
//!   at near-lossless quality (qindex = 0).
//!
//! * **Intra + inter** (`encode_core_intra_then_inter_stream`) — the
//!   homogeneous-syntax replacement for `encode_intra_then_inter_stream`
//!   in `encoder_inter.rs`. With both pictures in the same parse-code
//!   family ffmpeg's `dirac` decoder no longer trips its
//!   profile-mismatch guard. Locally we still validate the inter PSNR
//!   stays above the brief's 30 dB target.

use oxideav_core::CodecRegistry;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_dirac::encoder::{make_minimal_sequence, synthetic_testsrc_64_yuv420};
use oxideav_dirac::encoder_inter::{
    synthetic_translating_pair_64, InterEncoderParams, InterInputPicture,
};
use oxideav_dirac::encoder_intra_core::{
    encode_core_intra_then_inter_stream, encode_single_core_intra_stream, CoreIntraEncoderParams,
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

#[test]
fn core_intra_self_roundtrip_yuv420_synth_testsrc() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let params = CoreIntraEncoderParams::default_intra(WaveletFilter::LeGall5_3, 3);
    let (y, u, v) = synthetic_testsrc_64_yuv420();
    let stream = encode_single_core_intra_stream(&seq, &params, 0, &y, &u, &v);

    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.make_decoder(&cp).expect("decoder");
    let packet = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&packet).expect("send_packet");
    let frame = dec.receive_frame().expect("receive_frame");
    let vf = match frame {
        Frame::Video(v) => v,
        other => panic!("expected video frame, got {other:?}"),
    };

    let py = psnr(&vf.planes[0].data, &y);
    let pu = psnr(&vf.planes[1].data, &u);
    let pv = psnr(&vf.planes[2].data, &v);
    eprintln!("core-intra self-roundtrip: Y={py:.2} U={pu:.2} V={pv:.2}");
    // Y / U on this fixture round-trip bit-exactly; V hits a 1-LSB
    // edge-coefficient quantisation roughness on the steep gradient.
    assert!(py >= 48.0);
    assert!(pu >= 48.0);
    assert!(pv >= 40.0);
}

/// Intra-only with a flat picture — the IDWT collapses to a single LL
/// DC coefficient per component, so the round-trip should be perfect
/// across all three planes.
#[test]
fn core_intra_self_roundtrip_constant_frame_is_bit_exact() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let params = CoreIntraEncoderParams::default_intra(WaveletFilter::LeGall5_3, 3);
    let y = [123u8; 64 * 64];
    let u = [200u8; 32 * 32];
    let v = [55u8; 32 * 32];
    let stream = encode_single_core_intra_stream(&seq, &params, 0, &y, &u, &v);

    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.make_decoder(&cp).expect("decoder");
    let packet = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&packet).expect("send_packet");
    let frame = dec.receive_frame().expect("receive_frame");
    let vf = match frame {
        Frame::Video(v) => v,
        other => panic!("expected video frame, got {other:?}"),
    };
    assert_eq!(vf.planes[0].data, y.to_vec(), "Y plane bit-exact");
    assert_eq!(vf.planes[1].data, u.to_vec(), "U plane bit-exact");
    assert_eq!(vf.planes[2].data, v.to_vec(), "V plane bit-exact");
}

/// 3-frame stream: core-syntax intra reference followed by 2 inter
/// pictures (same reference). Validates the parse-code chain and the
/// reference-picture buffer survives across two inter decodes.
#[test]
fn core_intra_then_two_inter_chain_decodes_each_frame() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = CoreIntraEncoderParams::default_intra(WaveletFilter::LeGall5_3, 3);
    let inter_params = InterEncoderParams::default();
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
    let stream =
        encode_core_intra_then_inter_stream(&seq, &intra_params, &inter_params, &intra, &inter);

    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.make_decoder(&cp).expect("decoder");
    let packet = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&packet).expect("send_packet");

    // First frame: intra — should be near bit-exact (this fixture is
    // mostly flat so even V holds at infinity).
    let f0 = match dec.receive_frame().expect("intra") {
        Frame::Video(v) => v,
        other => panic!("expected video, got {other:?}"),
    };
    assert_eq!(f0.planes[0].data, y0.to_vec(), "intra Y bit-exact");

    // Second frame: inter — same brief target (≥ 30 dB Y) as the
    // existing HQ-intra+core-inter chain.
    let f1 = match dec.receive_frame().expect("inter") {
        Frame::Video(v) => v,
        other => panic!("expected video, got {other:?}"),
    };
    let py = psnr(&f1.planes[0].data, &y1);
    eprintln!("core-intra+core-inter: Y={py:.2} dB");
    assert!(
        py >= 30.0,
        "core-intra→core-inter Y PSNR {py:.2} dB below 30 dB"
    );
}

/// ffmpeg cross-decode: feed a homogeneous core-syntax stream
/// (intra `0x0C` + inter `0x09`) to ffmpeg's `dirac` decoder. With
/// both pictures in the same parse-code family the decoder no longer
/// rejects on a profile-mismatch — this is the close-out for the
/// round-1 soft-skip rationale in
/// `tests/ffmpeg_interop.rs::ffmpeg_decodes_our_inter_stream_translating_square`.
///
/// **Round 2 (this task / #135) is a hard assertion**, not a soft skip.
/// ffmpeg 8.1 (verified locally) accepts the homogeneous `0x0C` + `0x09`
/// chain end-to-end and decodes both pictures. The intra Y PSNR sits at
/// ~52 dB (qindex = 0 + LeGall 5/3 dead-zone identity), the inter Y
/// PSNR around ~19 dB on the translating-square fixture (cross-decode
/// is below the brief's ≥ 30 dB self-roundtrip floor because ffmpeg's
/// inter path differs from ours in OBMC overlap weighting + half-pel
/// reference filtering — both follow-up items). The assert floor
/// here is set deliberately low to absorb ffmpeg-version drift while
/// still catching framing / linkage regressions: anything above 15 dB
/// means the MV grid, parse-code framing and intra reference all
/// reached ffmpeg coherently.
#[test]
fn ffmpeg_decodes_our_core_intra_then_inter_stream() {
    fn ffmpeg_available() -> bool {
        std::process::Command::new("ffmpeg")
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }
    if !ffmpeg_available() {
        eprintln!("ffmpeg not available; skipping core-intra interop test");
        return;
    }

    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = CoreIntraEncoderParams::default_intra(WaveletFilter::LeGall5_3, 3);
    let inter_params = InterEncoderParams::default();
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
    let stream =
        encode_core_intra_then_inter_stream(&seq, &intra_params, &inter_params, &intra, &inter);

    let tmpdir = std::env::temp_dir();
    let drc = tmpdir.join("oxideav_dirac_interop_core_intra_inter.drc");
    let yuv = tmpdir.join("oxideav_dirac_interop_core_intra_inter.yuv");
    std::fs::write(&drc, &stream).expect("write drc");

    let status = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "dirac",
            "-i",
        ])
        .arg(&drc)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&yuv)
        .status()
        .expect("run ffmpeg");
    // No soft skip — task #135 acceptance criterion is that ffmpeg
    // accepts our homogeneous-profile stream cleanly. If this fails
    // we want a loud failure so the regression is caught immediately.
    assert!(
        status.success(),
        "ffmpeg rejected our core-intra+inter stream — see {drc:?}; \
         this is a regression: task #135 acceptance is that ffmpeg \
         decodes the homogeneous 0x0C + 0x09 chain end-to-end"
    );

    let out = std::fs::read(&yuv).expect("read ffmpeg yuv");
    let frame_size = 64 * 64 + 2 * 32 * 32;
    assert!(
        out.len() >= 2 * frame_size,
        "ffmpeg produced {} bytes; expected at least {} (2 frames)",
        out.len(),
        2 * frame_size
    );

    let intra_yuv = &out[..frame_size];
    let inter_yuv = &out[frame_size..2 * frame_size];

    // Intra reference round-trip — qindex = 0 so this should be
    // effectively lossless on Y and U for the translating-square
    // fixture (V is constant). The brief's ≥ 30 dB floor applies
    // here; in practice we land ~52 dB on a 2026-04 ffmpeg build.
    let intra_y_psnr = psnr(&intra_yuv[..64 * 64], &y0);
    eprintln!("ffmpeg core-intra Y PSNR: {intra_y_psnr:.2} dB");
    assert!(
        intra_y_psnr >= 30.0,
        "ffmpeg failed to recover intra Y above 30 dB"
    );

    // Inter — translating square, integer-pel ME. The brief's ≥ 30 dB
    // target applies to **self-roundtrip**; ffmpeg's inter decoder
    // differs from ours in a couple of places (OBMC overlap weighting,
    // half-pel reference interpolation) so the cross-decode floor
    // sits ~10 dB lower in r2. That's a follow-up item alongside
    // OBMC overlap reduction at the encoder. The asserts here just
    // confirm the chain doesn't collapse to noise — anything above
    // 15 dB means the MV grid, parse-code framing and intra reference
    // all reached ffmpeg coherently.
    let inter_y_psnr = psnr(&inter_yuv[..64 * 64], &y1);
    eprintln!("ffmpeg core-inter Y PSNR: {inter_y_psnr:.2} dB");
    assert!(
        inter_y_psnr >= 15.0,
        "ffmpeg core-inter Y PSNR {inter_y_psnr:.2} dB below 15 dB — \
         framing or reference linkage broke"
    );
}
