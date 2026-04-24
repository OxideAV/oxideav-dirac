//! Encoder self-roundtrip: encode a synthetic 64x64 4:2:0 YUV frame as
//! a VC-2 HQ intra-only elementary stream with our encoder, feed it
//! through our own decoder, and compare reconstructed pixels to the
//! original.
//!
//! At qindex=0 the dead-zone quantiser is an identity (qf=4: `q = 4*x/4 = x`,
//! inverse `x = (|x| * 4 + 1 + 2) / 4 = x`), and the forward/inverse
//! wavelets round-trip bit-exactly. So we expect bit-exact reproduction
//! of the input plane on the smallest test pattern.

use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, PixelFormat, TimeBase};
use oxideav_dirac::encoder::{
    encode_single_hq_intra_stream, encode_single_ld_intra_stream, make_minimal_sequence,
    synthetic_testsrc_64_yuv420, EncoderParams, LdEncoderParams,
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
    oxideav_dirac::register(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.make_decoder(&cp).expect("decoder");
    let packet = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&packet).expect("send_packet");
    let frame = dec.receive_frame().expect("receive_frame");
    let v_frame = match frame {
        Frame::Video(vf) => vf,
        other => panic!("expected video frame, got {other:?}"),
    };
    assert_eq!(v_frame.width, 64);
    assert_eq!(v_frame.height, 64);
    assert_eq!(v_frame.format, PixelFormat::Yuv420P);
    assert_eq!(v_frame.planes.len(), 3);

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
    oxideav_dirac::register(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.make_decoder(&cp).expect("decoder");
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
    oxideav_dirac::register(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.make_decoder(&cp).expect("decoder");
    let packet = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&packet).expect("send_packet");
    let frame = match dec.receive_frame().expect("receive_frame") {
        Frame::Video(v) => v,
        other => panic!("expected video, got {other:?}"),
    };
    assert_eq!(frame.width, 64);
    assert_eq!(frame.height, 64);
    assert_eq!(frame.format, PixelFormat::Yuv420P);

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
