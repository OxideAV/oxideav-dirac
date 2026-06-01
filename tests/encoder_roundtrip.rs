//! Encoder self-roundtrip: encode a synthetic 64x64 4:2:0 YUV frame as
//! a VC-2 HQ intra-only elementary stream with our encoder, feed it
//! through our own decoder, and compare reconstructed pixels to the
//! original.
//!
//! At qindex=0 the dead-zone quantiser is an identity (qf=4: `q = 4*x/4 = x`,
//! inverse `x = (|x| * 4 + 1 + 2) / 4 = x`), and the forward/inverse
//! wavelets round-trip bit-exactly. So we expect bit-exact reproduction
//! of the input plane on the smallest test pattern.

use oxideav_core::CodecRegistry;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_dirac::encoder::{
    encode_single_hq_intra_stream, encode_single_ld_intra_stream, make_minimal_sequence,
    make_minimal_sequence_ld, synthetic_testsrc_64_yuv420, EncoderParams, LdEncoderParams,
};
use oxideav_dirac::video_format::ChromaFormat;
use oxideav_dirac::wavelet::WaveletFilter;

/// Drive the full encoder -> decoder loop at qindex=0. Input plane
/// values must come back bit-exact.
#[test]
fn encode_then_decode_lossless_q0_64x64_yuv420() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let (y, u, v) = synthetic_testsrc_64_yuv420();
    let stream = encode_single_hq_intra_stream(&seq, &params, 0, &y, &u, &v);

    // Pipe through the decoder.
    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("decoder");
    let packet = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&packet).expect("send_packet");
    let frame = dec.receive_frame().expect("receive_frame");
    let v_frame = match frame {
        Frame::Video(vf) => vf,
        other => panic!("expected video frame, got {other:?}"),
    };
    // Yuv420P: Y plane stride == width, data.len()/stride == height.
    assert_eq!(v_frame.planes.len(), 3);
    assert_eq!(v_frame.planes[0].stride, 64);
    assert_eq!(v_frame.planes[0].data.len() / v_frame.planes[0].stride, 64);

    // Bit-exact comparison per plane.
    assert_eq!(v_frame.planes[0].data, y.to_vec(), "Y plane mismatch");
    assert_eq!(v_frame.planes[1].data, u.to_vec(), "U plane mismatch");
    assert_eq!(v_frame.planes[2].data, v.to_vec(), "V plane mismatch");
}

/// Roundtrip PSNR helper: return the y-plane PSNR (dB) of the decoder
/// output against the original. At qindex=0 this should be infinite;
/// at moderate qindex it quantifies how lossy we are.
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
fn encode_then_decode_lossy_q8_psnr_is_reasonable() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let mut params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    params.qindex = 8;
    // Use a flatter, less-pathological pattern for the lossy test so
    // the default slice-size scaler keeps length bytes under 256.
    let mut y_flat = [0u8; 64 * 64];
    let mut u_flat = [0u8; 32 * 32];
    let mut v_flat = [0u8; 32 * 32];
    for row in 0..64 {
        for col in 0..64 {
            y_flat[row * 64 + col] = ((row + col) * 2) as u8;
        }
    }
    for row in 0..32 {
        for col in 0..32 {
            u_flat[row * 32 + col] = 128u8.wrapping_add((col as i8) as u8);
            v_flat[row * 32 + col] = 128u8.wrapping_add((row as i8) as u8);
        }
    }
    let stream = encode_single_hq_intra_stream(&seq, &params, 0, &y_flat, &u_flat, &v_flat);
    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("decoder");
    let packet = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&packet).expect("send_packet");
    let frame = match dec.receive_frame().expect("receive_frame") {
        Frame::Video(v) => v,
        other => panic!("expected video, got {other:?}"),
    };
    let psnr_y = psnr(&frame.planes[0].data, &y_flat);
    eprintln!("PSNR (Y, qindex=8): {psnr_y:.2} dB");
    assert!(
        psnr_y > 25.0,
        "q8 PSNR {psnr_y} should be at least 25 dB on a smooth test pattern"
    );
}

/// VC-2 LD (Low-Delay) intra-only: encode → decode with a 128-byte
/// per-slice budget at qindex=0 (near-lossless) on a 64x64 4:2:0 test
/// pattern, and check that the PSNR is comfortably above the 35 dB
/// round-trip threshold.
#[test]
fn encode_then_decode_ld_qindex0_psnr_over_35() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    // 4x4 slices, 128 bytes/slice → 2048 bytes picture budget. With a
    // smooth gradient this fits comfortably at q=0.
    let params = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 128);
    // Smooth gradient so a tight slice budget fits at qindex=0.
    let mut y_flat = [0u8; 64 * 64];
    let mut u_flat = [128u8; 32 * 32];
    let mut v_flat = [128u8; 32 * 32];
    for row in 0..64 {
        for col in 0..64 {
            y_flat[row * 64 + col] = ((row + col) * 2) as u8;
        }
    }
    for row in 0..32 {
        for col in 0..32 {
            u_flat[row * 32 + col] = 128u8.wrapping_add((col as i8 / 2) as u8);
            v_flat[row * 32 + col] = 128u8.wrapping_add((row as i8 / 2) as u8);
        }
    }
    let stream = encode_single_ld_intra_stream(&seq, &params, 0, &y_flat, &u_flat, &v_flat);

    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("decoder");
    let packet = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&packet).expect("send_packet");
    let frame = match dec.receive_frame().expect("receive_frame") {
        Frame::Video(v) => v,
        other => panic!("expected video, got {other:?}"),
    };
    // Yuv420P: Y plane stride == width, data.len()/stride == height.
    assert_eq!(frame.planes[0].stride, 64);
    assert_eq!(frame.planes[0].data.len() / frame.planes[0].stride, 64);

    let psnr_y = psnr(&frame.planes[0].data, &y_flat);
    let psnr_u = psnr(&frame.planes[1].data, &u_flat);
    let psnr_v = psnr(&frame.planes[2].data, &v_flat);
    eprintln!("LD q0 PSNR:  Y={psnr_y:.2} dB  U={psnr_u:.2} dB  V={psnr_v:.2} dB");
    assert!(
        psnr_y >= 35.0,
        "LD Y PSNR {psnr_y:.2} dB below the 35 dB round-trip target"
    );
    assert!(
        psnr_u >= 35.0,
        "LD U PSNR {psnr_u:.2} dB below the 35 dB round-trip target"
    );
    assert!(
        psnr_v >= 35.0,
        "LD V PSNR {psnr_v:.2} dB below the 35 dB round-trip target"
    );
}

// ---------------------------------------------------------------------
// VC-2 v3 (SMPTE ST 2042-1:2022 §12.4.4) self-roundtrip coverage.
//
// Round-201 added the decoder-side `parse_extended_transform_parameters`
// helper plus the new `PictureError::AsymmetricTransformUnsupported`
// rejection path. The two tests below close the loop on the *encoder*
// side: with `EncoderParams::major_version = 3` /
// `LdEncoderParams::major_version = 3` and a matching sequence-header
// `version_major = 3`, the encoder emits the symmetric-default
// `extended_transform_parameters()` flag pair (per the §12.4.4 NOTE)
// and the resulting bitstream decodes back through our own decoder to
// bit-exact pixels (HQ at qindex=0) / above-threshold PSNR (LD at
// qindex=0). v2 (default) and v3 (symmetric) streams must produce
// pixel-identical reconstructions because the §12.4.4 NOTE guarantees
// the IDWT is the same in both cases.
// ---------------------------------------------------------------------

/// HQ intra, `major_version = 3` (symmetric `extended_transform_parameters`
/// defaults), qindex=0: bit-exact roundtrip AND pixel-identical to the
/// equivalent v2 stream.
#[test]
fn encode_then_decode_hq_v3_symmetric_default_lossless_q0() {
    let mut seq_v3 = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    seq_v3.parse_parameters.version_major = 3;
    let params_v3 = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3).with_major_version_3();

    let seq_v2 = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let params_v2 = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);

    let (y, u, v) = synthetic_testsrc_64_yuv420();
    let stream_v3 = encode_single_hq_intra_stream(&seq_v3, &params_v3, 0, &y, &u, &v);
    let stream_v2 = encode_single_hq_intra_stream(&seq_v2, &params_v2, 0, &y, &u, &v);

    // The v3 stream MUST differ from v2 byte-wise (two extra `False`
    // flag bits inside `transform_parameters()` + the bumped
    // `version_major` exp-Golomb code in the sequence header), but
    // both must decode to the identical reconstruction.
    assert_ne!(
        stream_v3, stream_v2,
        "v3 stream should not be byte-identical to v2 — the two flag bits + bumped version exp-Golomb must change the bitstream"
    );

    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));

    let mut dec_v3 = reg.first_decoder(&cp).expect("decoder");
    dec_v3
        .send_packet(&Packet::new(0, TimeBase::new(1, 25), stream_v3))
        .expect("send v3");
    let frame_v3 = match dec_v3.receive_frame().expect("recv v3") {
        Frame::Video(vf) => vf,
        other => panic!("expected video, got {other:?}"),
    };
    let mut dec_v2 = reg.first_decoder(&cp).expect("decoder");
    dec_v2
        .send_packet(&Packet::new(0, TimeBase::new(1, 25), stream_v2))
        .expect("send v2");
    let frame_v2 = match dec_v2.receive_frame().expect("recv v2") {
        Frame::Video(vf) => vf,
        other => panic!("expected video, got {other:?}"),
    };

    // Bit-exact against the input at qindex=0 on the v3 path.
    assert_eq!(frame_v3.planes[0].data, y.to_vec(), "v3 Y mismatch");
    assert_eq!(frame_v3.planes[1].data, u.to_vec(), "v3 U mismatch");
    assert_eq!(frame_v3.planes[2].data, v.to_vec(), "v3 V mismatch");

    // And byte-identical to the v2 reconstruction (per the §12.4.4
    // NOTE, the IDWT is identical when ho-defaults are in force).
    assert_eq!(
        frame_v3.planes[0].data, frame_v2.planes[0].data,
        "v3/v2 Y reconstructions diverge despite symmetric-default ho params"
    );
    assert_eq!(frame_v3.planes[1].data, frame_v2.planes[1].data);
    assert_eq!(frame_v3.planes[2].data, frame_v2.planes[2].data);
}

/// LD intra, `major_version = 3` (symmetric `extended_transform_parameters`
/// defaults), qindex=0: PSNR above the 35 dB round-trip threshold AND
/// pixel-identical to the equivalent v2 stream.
#[test]
fn encode_then_decode_ld_v3_symmetric_default_qindex0_psnr_over_35() {
    let mut seq_v3 = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    seq_v3.parse_parameters.version_major = 3;
    let params_v3 =
        LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 128).with_major_version_3();

    let seq_v2 = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let params_v2 = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 128);

    // Smooth gradient — the LD slice budget fits comfortably at q=0.
    let mut y_flat = [0u8; 64 * 64];
    let mut u_flat = [128u8; 32 * 32];
    let mut v_flat = [128u8; 32 * 32];
    for row in 0..64 {
        for col in 0..64 {
            y_flat[row * 64 + col] = ((row + col) * 2) as u8;
        }
    }
    for row in 0..32 {
        for col in 0..32 {
            u_flat[row * 32 + col] = 128u8.wrapping_add((col as i8 / 2) as u8);
            v_flat[row * 32 + col] = 128u8.wrapping_add((row as i8 / 2) as u8);
        }
    }

    let stream_v3 =
        encode_single_ld_intra_stream(&seq_v3, &params_v3, 0, &y_flat, &u_flat, &v_flat);
    let stream_v2 =
        encode_single_ld_intra_stream(&seq_v2, &params_v2, 0, &y_flat, &u_flat, &v_flat);

    assert_ne!(
        stream_v3, stream_v2,
        "v3 LD stream should not be byte-identical to v2 — flag bits + version differ"
    );

    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));

    let mut dec_v3 = reg.first_decoder(&cp).expect("decoder");
    dec_v3
        .send_packet(&Packet::new(0, TimeBase::new(1, 25), stream_v3))
        .expect("send v3");
    let frame_v3 = match dec_v3.receive_frame().expect("recv v3") {
        Frame::Video(vf) => vf,
        other => panic!("expected video, got {other:?}"),
    };
    let mut dec_v2 = reg.first_decoder(&cp).expect("decoder");
    dec_v2
        .send_packet(&Packet::new(0, TimeBase::new(1, 25), stream_v2))
        .expect("send v2");
    let frame_v2 = match dec_v2.receive_frame().expect("recv v2") {
        Frame::Video(vf) => vf,
        other => panic!("expected video, got {other:?}"),
    };

    let psnr_y = psnr(&frame_v3.planes[0].data, &y_flat);
    let psnr_u = psnr(&frame_v3.planes[1].data, &u_flat);
    let psnr_v = psnr(&frame_v3.planes[2].data, &v_flat);
    eprintln!("LD v3 q0 PSNR:  Y={psnr_y:.2} dB  U={psnr_u:.2} dB  V={psnr_v:.2} dB");
    assert!(
        psnr_y >= 35.0,
        "LD v3 Y PSNR {psnr_y:.2} dB below the 35 dB round-trip target"
    );
    assert!(psnr_u >= 35.0, "LD v3 U PSNR {psnr_u:.2} dB below 35 dB");
    assert!(psnr_v >= 35.0, "LD v3 V PSNR {psnr_v:.2} dB below 35 dB");

    // Pixel-identical to the v2 reconstruction.
    assert_eq!(
        frame_v3.planes[0].data, frame_v2.planes[0].data,
        "LD v3 / v2 Y reconstructions diverge despite symmetric-default ho params"
    );
    assert_eq!(frame_v3.planes[1].data, frame_v2.planes[1].data);
    assert_eq!(frame_v3.planes[2].data, frame_v2.planes[2].data);
}
