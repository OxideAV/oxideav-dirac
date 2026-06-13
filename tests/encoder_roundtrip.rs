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
    make_minimal_sequence_ld, synthetic_testsrc_64_yuv420, EncoderParams,
    ExtendedTransformOverride, LdEncoderParams,
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

// ---------------------------------------------------------------------
// §12.4.4 asymmetric (non-default) extended_transform_parameters —
// encoder-side emission, decoder-side end-to-end decode.
//
// SMPTE ST 2042-1:2022 §12.4.4.2 lets a v3 stream set
// `wavelet_index_ho != wavelet_index` (raising asym_transform_index_flag)
// and §12.4.4.3 lets it set `dwt_depth_ho != 0` (raising
// asym_transform_flag). With `dwt_depth_ho > 0` the decode chain runs
// the §13.5 asymmetric slice unpack, the §13.5.5 asymmetric slice
// quantisers and the §15.4.1 / §15.4.2 horizontal-only synthesis
// levels; with `dwt_depth_ho == 0` the override is inert (the §12.4.4
// NOTE shortcut). Asymmetric streams using the Annex D *default*
// quantisation matrices (`custom_quant_matrix = False`, Tables D.1–D.8)
// now decode too; only a `(filter, ho-filter, depth, ho)` combination
// with no Annex D default and no custom matrix still rejects.
// ---------------------------------------------------------------------

const ASYM_ERR_SUBSTR: &str = "no Annex D default quant matrix";

/// Decode `stream` through the registry decoder and return the frame.
fn decode_video_frame(stream: Vec<u8>) -> oxideav_core::VideoFrame {
    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("decoder");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), stream))
        .expect("send_packet");
    match dec.receive_frame().expect("receive_frame") {
        Frame::Video(vf) => vf,
        other => panic!("expected video frame, got {other:?}"),
    }
}

/// HQ v3 with `asym_transform_index_flag = 1` (wavelet_index_ho !=
/// wavelet_index) but `dwt_depth_ho = 0` must (a) differ from the
/// symmetric-default v3 stream and (b) decode to the identical
/// reconstruction — the §15.4.1 horizontal-only loop runs zero
/// iterations, so the filter override is inert per the §12.4.4 NOTE.
#[test]
fn encode_hq_v3_asym_wavelet_index_ho_depth0_decodes_identically() {
    let mut seq_v3 = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    seq_v3.parse_parameters.version_major = 3;
    let params_sym = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3).with_major_version_3();
    // `wavelet_index_ho = 0` (DD9_7 differs from LeGall5_3 = 1) flips
    // `asym_transform_index_flag` to True and emits the override value
    // as an interleaved exp-Golomb code. `dwt_depth_ho = 0` keeps the
    // second flag at its default, so the transform itself is unchanged.
    let params_asym =
        params_sym
            .clone()
            .with_extended_transform_override(ExtendedTransformOverride {
                wavelet_index_ho: 0,
                dwt_depth_ho: 0,
            });

    let (y, u, v) = synthetic_testsrc_64_yuv420();
    let stream_sym = encode_single_hq_intra_stream(&seq_v3, &params_sym, 0, &y, &u, &v);
    let stream_asym = encode_single_hq_intra_stream(&seq_v3, &params_asym, 0, &y, &u, &v);

    assert_ne!(
        stream_sym, stream_asym,
        "asymmetric override should change the v3 bitstream (asym_transform_index_flag + exp-Golomb wavelet_index_ho)"
    );

    let frame = decode_video_frame(stream_asym);
    assert_eq!(frame.planes[0].data, y.to_vec(), "Y plane mismatch");
    assert_eq!(frame.planes[1].data, u.to_vec(), "U plane mismatch");
    assert_eq!(frame.planes[2].data, v.to_vec(), "V plane mismatch");
}

/// HQ v3 asymmetric transform (`dwt_depth_ho > 0`) end-to-end: the
/// encoder runs the horizontal-only forward analysis + asymmetric
/// slice packing, and the decoder reverses the whole chain — §13.5.4
/// asymmetric slice unpack → §13.5.5 asymmetric quantisers → §15.4.1
/// IDWT with `dwt_depth_ho` §15.4.2 `h_synthesis` levels — to
/// bit-exact pixels at qindex 0. Covers ho ∈ {1, 2} and a
/// horizontal-only filter that differs from the 2-D filter.
#[test]
fn encode_hq_v3_asym_dwt_depth_ho_lossless_roundtrip() {
    let (y, u, v) = synthetic_testsrc_64_yuv420();
    for (dwt_depth, ho, wavelet_ho) in [
        (3u32, 1u32, WaveletFilter::LeGall5_3), // wavelet_index_ho == wavelet_index
        (2, 2, WaveletFilter::Haar0),           // differing horizontal-only filter
        (2, 1, WaveletFilter::DeslauriersDubuc9_7),
    ] {
        let mut seq_v3 = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
        seq_v3.parse_parameters.version_major = 3;
        let params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, dwt_depth)
            .with_asymmetric_transform(wavelet_ho, ho);
        let stream = encode_single_hq_intra_stream(&seq_v3, &params, 0, &y, &u, &v);
        let frame = decode_video_frame(stream);
        assert_eq!(
            frame.planes[0].data,
            y.to_vec(),
            "Y plane mismatch (depth={dwt_depth}, ho={ho}, wavelet_ho={wavelet_ho:?})"
        );
        assert_eq!(
            frame.planes[1].data,
            u.to_vec(),
            "U plane mismatch (depth={dwt_depth}, ho={ho}, wavelet_ho={wavelet_ho:?})"
        );
        assert_eq!(
            frame.planes[2].data,
            v.to_vec(),
            "V plane mismatch (depth={dwt_depth}, ho={ho}, wavelet_ho={wavelet_ho:?})"
        );
    }
}

/// HQ v3 asymmetric stream relying on the Annex D *default*
/// quantisation matrices (`custom_quant_matrix = False`, Table D.2)
/// now decodes end-to-end. The encoder emits `custom_quant_matrix =
/// False`; the decoder looks the matrix up from Annex D rather than
/// reading it in-band. At qindex 0 every band's effective quantiser is
/// 0 regardless of the matrix values (§13.5.5 clamps `qindex - entry`
/// to 0), so the reconstruction is bit-exact. Covers ho ∈ {1, 2}
/// (sum ≤ 5) and a differing horizontal-only filter (Table D.8's
/// Haar0/LeGall cross-default).
#[test]
fn encode_hq_v3_asym_default_matrix_decodes() {
    let (y, u, v) = synthetic_testsrc_64_yuv420();
    for (wavelet, dwt_depth, wavelet_ho, ho) in [
        (
            WaveletFilter::LeGall5_3,
            3u32,
            WaveletFilter::LeGall5_3,
            1u32,
        ), // Table D.2
        (WaveletFilter::LeGall5_3, 2, WaveletFilter::LeGall5_3, 2), // Table D.2
        (WaveletFilter::Haar0, 2, WaveletFilter::LeGall5_3, 1),     // Table D.8 cross-default
    ] {
        let mut seq_v3 = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
        seq_v3.parse_parameters.version_major = 3;
        // Fully-wired asymmetric setup (shape-consistent quantiser
        // bookkeeping), then flip `custom_quant_matrix` to False so the
        // wire carries the `custom_quant_matrix = False` flag — the
        // Annex D default-matrix asymmetric stream.
        let mut params =
            EncoderParams::default_hq(wavelet, dwt_depth).with_asymmetric_transform(wavelet_ho, ho);
        params.custom_quant_matrix = false;

        let stream = encode_single_hq_intra_stream(&seq_v3, &params, 0, &y, &u, &v);
        let frame = decode_video_frame(stream);
        assert_eq!(
            frame.planes[0].data,
            y.to_vec(),
            "Y plane mismatch (wavelet={wavelet:?}, depth={dwt_depth}, wavelet_ho={wavelet_ho:?}, ho={ho})"
        );
        assert_eq!(frame.planes[1].data, u.to_vec(), "U plane mismatch");
        assert_eq!(frame.planes[2].data, v.to_vec(), "V plane mismatch");
    }
}

/// An asymmetric stream whose `(filter, ho-filter, depth, ho)`
/// combination has no Annex D default (here `dwt_depth + dwt_depth_ho
/// > 5`) and supplies no custom matrix is still rejected: the spec
/// requires a custom matrix there (§12.4.5.3). Pins the remaining
/// `AsymmetricTransformUnsupported` arm.
#[test]
fn encode_hq_v3_asym_off_table_default_matrix_rejected_by_decoder() {
    let mut seq_v3 = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    seq_v3.parse_parameters.version_major = 3;
    // dwt_depth = 3 + dwt_depth_ho = 3 → sum 6 > 5: no Annex D default.
    let mut params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3)
        .with_asymmetric_transform(WaveletFilter::LeGall5_3, 3);
    params.custom_quant_matrix = false;

    let (y, u, v) = synthetic_testsrc_64_yuv420();
    let stream = encode_single_hq_intra_stream(&seq_v3, &params, 0, &y, &u, &v);

    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("decoder");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), stream))
        .expect("send asym stream");
    let err = dec
        .receive_frame()
        .expect_err("off-table default-matrix asymmetric v3 stream must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains(ASYM_ERR_SUBSTR),
        "expected `{ASYM_ERR_SUBSTR}` in error, got: {msg}"
    );
}

/// LD v3 mirror of the HQ wavelet-index-ho-with-depth-0 test: the LD
/// `write_ld_transform_parameters` path emits the identical
/// `extended_transform_parameters()` block; with `dwt_depth_ho = 0`
/// the override is inert and the stream decodes to the same
/// reconstruction as the symmetric-default v3 stream.
#[test]
fn encode_ld_v3_asym_wavelet_index_ho_depth0_decodes_identically() {
    let mut seq_v3 = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    seq_v3.parse_parameters.version_major = 3;
    let params_sym =
        LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 128).with_major_version_3();
    let params_asym =
        params_sym
            .clone()
            .with_extended_transform_override(ExtendedTransformOverride {
                wavelet_index_ho: 0, // DD9_7 != LeGall5_3
                dwt_depth_ho: 0,
            });

    let mut y_flat = [0u8; 64 * 64];
    let u_flat = [128u8; 32 * 32];
    let v_flat = [128u8; 32 * 32];
    for row in 0..64 {
        for col in 0..64 {
            y_flat[row * 64 + col] = ((row + col) * 2) as u8;
        }
    }

    let stream_sym =
        encode_single_ld_intra_stream(&seq_v3, &params_sym, 0, &y_flat, &u_flat, &v_flat);
    let stream_asym =
        encode_single_ld_intra_stream(&seq_v3, &params_asym, 0, &y_flat, &u_flat, &v_flat);

    assert_ne!(
        stream_sym, stream_asym,
        "LD asymmetric override should change the v3 bitstream"
    );

    let frame_sym = decode_video_frame(stream_sym);
    let frame_asym = decode_video_frame(stream_asym);
    for plane in 0..3 {
        assert_eq!(
            frame_asym.planes[plane].data, frame_sym.planes[plane].data,
            "LD plane {plane} diverges despite dwt_depth_ho == 0 (inert filter override)"
        );
    }
}

/// LD v3 asymmetric transform end-to-end: encode with
/// `dwt_depth_ho = 1` (custom zero quant matrix, §12.4.5.3 asymmetric
/// shape) at qindex 0 with a generous slice budget, decode through the
/// §13.5.3 asymmetric slice unpack + §15.4.1 horizontal-only IDWT, and
/// check the reconstruction clears the same 35 dB threshold as the
/// symmetric LD roundtrip (LD slices truncate to budget, so the LD
/// contract is PSNR, not bit-exactness).
#[test]
fn encode_ld_v3_asym_dwt_depth_ho_roundtrip_psnr_over_35() {
    let mut seq_v3 = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    seq_v3.parse_parameters.version_major = 3;
    let params = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 128)
        .with_asymmetric_transform(WaveletFilter::LeGall5_3, 1);

    let mut y_flat = [0u8; 64 * 64];
    let u_flat = [128u8; 32 * 32];
    let v_flat = [128u8; 32 * 32];
    for row in 0..64 {
        for col in 0..64 {
            y_flat[row * 64 + col] = ((row + col) * 2) as u8;
        }
    }

    let stream = encode_single_ld_intra_stream(&seq_v3, &params, 0, &y_flat, &u_flat, &v_flat);
    let frame = decode_video_frame(stream);
    let psnr_y = psnr(&frame.planes[0].data, &y_flat);
    let psnr_u = psnr(&frame.planes[1].data, &u_flat);
    let psnr_v = psnr(&frame.planes[2].data, &v_flat);
    eprintln!("LD v3 asym ho=1 q0 PSNR:  Y={psnr_y:.2} dB  U={psnr_u:.2} dB  V={psnr_v:.2} dB");
    assert!(
        psnr_y >= 35.0,
        "LD asym Y PSNR {psnr_y:.2} dB below the 35 dB round-trip target"
    );
    assert!(psnr_u >= 35.0, "LD asym U PSNR {psnr_u:.2} dB below 35 dB");
    assert!(psnr_v >= 35.0, "LD asym V PSNR {psnr_v:.2} dB below 35 dB");
}
