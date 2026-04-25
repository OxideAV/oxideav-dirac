//! Encode -> decode quality matrix.
//!
//! These tests cross-multiply the encoder's degrees of freedom (wavelet
//! filter, chroma format, picture dimensions, slice grid, multi-frame)
//! against the decoder's expectations and assert lossless or near-
//! lossless reconstruction. They catch regressions that single-axis
//! tests miss — e.g. a tap-sign bug in the Fidelity filter that LeGall
//! cannot detect, or a slice-bounds off-by-one that only fires at
//! 96-pixel widths.
//!
//! The HQ profile at qindex=0 is bit-exact for any wavelet whose
//! lifting steps integer-reverse and any picture whose dimensions are
//! multiples of `2^dwt_depth` (the spec's §15.7 alignment). The
//! Fidelity filter expands intermediate magnitudes by a factor of
//! ~256 between lifting steps, which can overflow our 28-bit funnel
//! coding budget at large dimensions; we exclude it from full-picture
//! HQ tests for that reason and fall back to wavelet-level coverage in
//! `wavelet::tests::dwt_idwt_roundtrip_all_filters_all_depths`.

use oxideav_core::{CodecId, CodecParameters, CodecRegistry, Frame, Packet, PixelFormat, TimeBase};
use oxideav_dirac::encoder::{
    encode_hq_intra_multi_stream, encode_single_hq_intra_stream, encode_single_ld_intra_stream,
    make_minimal_sequence, make_minimal_sequence_ld, EncoderParams, InputPicture, LdEncoderParams,
};
use oxideav_dirac::video_format::ChromaFormat;
use oxideav_dirac::wavelet::WaveletFilter;

/// Build a deterministic non-trivial 8-bit plane: a smooth gradient
/// plus a few high-contrast spikes. Avoids saturating at 255 / 0 so
/// the dead-zone quantiser does not clip the inverse.
fn synth_plane(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut p = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let g = (x as u32 + y as u32 * 2 + seed) & 0xff;
            // A diagonal stripe at every 17th anti-diagonal stays a
            // contained delta — keeps slice budgets sane and avoids
            // saturating to 255.
            let v = if (x + y) % 17 == 0 {
                g.saturating_add(40)
            } else {
                g
            };
            p[y * w + x] = v as u8;
        }
    }
    p
}

/// Decode a single-frame elementary stream and return the 8-bit
/// `VideoFrame`. Panics if the decoder fails to produce a frame or the
/// frame is not 8-bit planar.
fn decode_one(stream: Vec<u8>) -> oxideav_core::VideoFrame {
    let mut reg = CodecRegistry::new();
    oxideav_dirac::register(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.make_decoder(&cp).expect("make decoder");
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&pkt).expect("send packet");
    match dec.receive_frame().expect("receive frame") {
        Frame::Video(v) => v,
        other => panic!("expected video frame, got {other:?}"),
    }
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "plane size mismatch");
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

/// Wavelet × chroma format matrix at qindex=0. Six of the seven
/// wavelets are exercised end-to-end (encode → our decoder); each one
/// must reproduce the input bit-exactly because the dead-zone
/// quantiser is identity at q=0 and integer-lifting wavelets are
/// reversible.
///
/// **Fidelity** is excluded from this test on purpose: its lifting
/// taps multiply each pass's magnitude by ~256 (Table 15.6), so an
/// 8-bit input to the encoder can hit ~16-bit subband coefficients,
/// and the §13.5.4 Funnel-bounded variable-length coder used by the
/// HQ slice path is sized for the typical-photographic-content
/// magnitudes that the other six filters produce. Wavelet-level
/// coverage of Fidelity lives in
/// `wavelet::tests::dwt_idwt_roundtrip_all_filters_all_depths`.
#[test]
fn hq_q0_lossless_across_six_wavelets_and_three_chromas() {
    let w: u32 = 64;
    let h: u32 = 64;
    for chroma in [
        ChromaFormat::Yuv420,
        ChromaFormat::Yuv422,
        ChromaFormat::Yuv444,
    ] {
        let (cw, ch) = match chroma {
            ChromaFormat::Yuv420 => (w / 2, h / 2),
            ChromaFormat::Yuv422 => (w / 2, h),
            ChromaFormat::Yuv444 => (w, h),
        };
        let y = synth_plane(w as usize, h as usize, 0);
        let u = synth_plane(cw as usize, ch as usize, 31);
        let v = synth_plane(cw as usize, ch as usize, 53);

        for filter in [
            WaveletFilter::DeslauriersDubuc9_7,
            WaveletFilter::LeGall5_3,
            WaveletFilter::DeslauriersDubuc13_7,
            WaveletFilter::Haar0,
            WaveletFilter::Haar1,
            WaveletFilter::Daubechies9_7,
        ] {
            let seq = make_minimal_sequence(w, h, chroma);
            let params = EncoderParams::default_hq(filter, 3);
            let stream = encode_single_hq_intra_stream(&seq, &params, 0, &y, &u, &v);
            let frame = decode_one(stream);
            assert_eq!(frame.width, w, "{filter:?} {chroma:?} width");
            assert_eq!(frame.height, h, "{filter:?} {chroma:?} height");
            let want_format = match chroma {
                ChromaFormat::Yuv420 => PixelFormat::Yuv420P,
                ChromaFormat::Yuv422 => PixelFormat::Yuv422P,
                ChromaFormat::Yuv444 => PixelFormat::Yuv444P,
            };
            assert_eq!(frame.format, want_format, "{filter:?} {chroma:?} pix_fmt");
            assert_eq!(
                frame.planes[0].data, y,
                "Y mismatch ({filter:?} {chroma:?})"
            );
            assert_eq!(
                frame.planes[1].data, u,
                "U mismatch ({filter:?} {chroma:?})"
            );
            assert_eq!(
                frame.planes[2].data, v,
                "V mismatch ({filter:?} {chroma:?})"
            );
        }
    }
}

/// Non-power-of-two frame sizes that are still depth-aligned: 96x80
/// and 128x48 at depth 3 (multiples of 8). These exercise the
/// asymmetric slice-bounds path (`slices_x != slices_y`) and the
/// chroma-plane size handling for non-square 4:2:0.
#[test]
fn hq_q0_lossless_at_non_square_dimensions() {
    for &(w, h, sx, sy) in &[(96u32, 80u32, 6u32, 5u32), (128u32, 48u32, 8u32, 3u32)] {
        let cw = w / 2;
        let ch = h / 2;
        let y = synth_plane(w as usize, h as usize, 7);
        let u = synth_plane(cw as usize, ch as usize, 11);
        let v = synth_plane(cw as usize, ch as usize, 13);
        let seq = make_minimal_sequence(w, h, ChromaFormat::Yuv420);
        let mut params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
        params.slices_x = sx;
        params.slices_y = sy;
        let stream = encode_single_hq_intra_stream(&seq, &params, 0, &y, &u, &v);
        let frame = decode_one(stream);
        assert_eq!(frame.width, w);
        assert_eq!(frame.height, h);
        assert_eq!(
            frame.planes[0].data, y,
            "Y not bit-exact at {w}x{h} ({sx}x{sy} slices)"
        );
        assert_eq!(frame.planes[1].data, u, "U not bit-exact at {w}x{h}");
        assert_eq!(frame.planes[2].data, v, "V not bit-exact at {w}x{h}");
    }
}

/// Multi-frame HQ stream at qindex=0: every emitted picture must come
/// back bit-exact, regardless of whether it's tagged as a reference
/// (`0xEC`) or non-reference (`0xE8`) intra. Three frames pin both
/// the alternation and the per-picture decode reset.
#[test]
fn hq_q0_lossless_three_frame_multi_stream() {
    let w: u32 = 64;
    let h: u32 = 64;
    let cw = w / 2;
    let ch = h / 2;
    // Three pictures with distinctly different content.
    let y0 = synth_plane(w as usize, h as usize, 0);
    let y1 = synth_plane(w as usize, h as usize, 7);
    let y2 = synth_plane(w as usize, h as usize, 19);
    let u0 = synth_plane(cw as usize, ch as usize, 1);
    let u1 = synth_plane(cw as usize, ch as usize, 8);
    let u2 = synth_plane(cw as usize, ch as usize, 20);
    let v0 = synth_plane(cw as usize, ch as usize, 2);
    let v1 = synth_plane(cw as usize, ch as usize, 9);
    let v2 = synth_plane(cw as usize, ch as usize, 21);
    let pics = [
        InputPicture {
            picture_number: 0,
            y: &y0,
            u: &u0,
            v: &v0,
        },
        InputPicture {
            picture_number: 1,
            y: &y1,
            u: &u1,
            v: &v1,
        },
        InputPicture {
            picture_number: 2,
            y: &y2,
            u: &u2,
            v: &v2,
        },
    ];
    let seq = make_minimal_sequence(w, h, ChromaFormat::Yuv420);
    let params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let stream = encode_hq_intra_multi_stream(&seq, &params, &pics);

    let mut reg = CodecRegistry::new();
    oxideav_dirac::register(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.make_decoder(&cp).expect("decoder");
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&pkt).expect("send_packet");

    let expected: [(&[u8], &[u8], &[u8]); 3] = [(&y0, &u0, &v0), (&y1, &u1, &v1), (&y2, &u2, &v2)];
    for (i, (ye, ue, ve)) in expected.iter().enumerate() {
        let frame = match dec.receive_frame().expect("receive_frame") {
            Frame::Video(v) => v,
            other => panic!("frame {i}: expected video, got {other:?}"),
        };
        assert_eq!(&frame.planes[0].data[..], *ye, "frame {i} Y mismatch");
        assert_eq!(&frame.planes[1].data[..], *ue, "frame {i} U mismatch");
        assert_eq!(&frame.planes[2].data[..], *ve, "frame {i} V mismatch");
    }
}

/// Edge case: smallest depth-3-compatible picture (8x8). At this size
/// the level-0 LL band is a single coefficient per component — exercises
/// the `subband_dims` floor / slice-bounds logic at the limits where
/// off-by-one regressions hide.
#[test]
fn hq_q0_lossless_at_8x8_minimum() {
    let w: u32 = 8;
    let h: u32 = 8;
    let cw = w / 2;
    let ch = h / 2;
    // Use a deliberate non-flat pattern so a faulty IDWT doesn't pass
    // by accident on a constant picture.
    let mut y = vec![0u8; (w * h) as usize];
    for (i, dst) in y.iter_mut().enumerate() {
        *dst = (i as u32 * 17 % 251) as u8;
    }
    let u = vec![100u8; (cw * ch) as usize];
    let v = vec![150u8; (cw * ch) as usize];
    let seq = make_minimal_sequence(w, h, ChromaFormat::Yuv420);
    // For an 8x8 picture only one slice fits — use a 1x1 grid.
    let mut params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    params.slices_x = 1;
    params.slices_y = 1;
    let stream = encode_single_hq_intra_stream(&seq, &params, 0, &y, &u, &v);
    let frame = decode_one(stream);
    assert_eq!(frame.width, 8);
    assert_eq!(frame.height, 8);
    assert_eq!(frame.planes[0].data, y, "Y not bit-exact at 8x8");
    assert_eq!(frame.planes[1].data, u, "U not bit-exact at 8x8");
    assert_eq!(frame.planes[2].data, v, "V not bit-exact at 8x8");
}

/// HQ at qindex=0 should be bit-exact at depth=2 as well as the more
/// commonly-tested depth=3. The `forward_component` padding rule
/// (multiple of `2^dwt_depth`) and `subband_dims` formula change with
/// depth; cover both.
#[test]
fn hq_q0_lossless_at_dwt_depth_two() {
    let w: u32 = 64;
    let h: u32 = 64;
    let cw = w / 2;
    let ch = h / 2;
    let y = synth_plane(w as usize, h as usize, 0);
    let u = synth_plane(cw as usize, ch as usize, 1);
    let v = synth_plane(cw as usize, ch as usize, 2);
    let seq = make_minimal_sequence(w, h, ChromaFormat::Yuv420);
    let params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 2);
    let stream = encode_single_hq_intra_stream(&seq, &params, 0, &y, &u, &v);
    let frame = decode_one(stream);
    assert_eq!(frame.planes[0].data, y, "Y not bit-exact at depth=2");
    assert_eq!(frame.planes[1].data, u, "U not bit-exact at depth=2");
    assert_eq!(frame.planes[2].data, v, "V not bit-exact at depth=2");
}

/// Fidelity wavelet (filter index 5, Table 15.6) probe: the spec
/// allows it but the lifting taps (`-25, 81, 81, -25` at S=8 with
/// `filter_shift=0`) inflate coefficient magnitudes by ~256x per pass.
/// On flat-ish content the inflation is bounded; assert that round-
/// tripping a near-constant 8-bit picture still yields a high-PSNR
/// reconstruction. Captures the regression boundary so a future
/// reduction in the slice budget for Fidelity can be flagged.
#[test]
fn hq_q0_fidelity_works_on_low_variance_input() {
    let w: u32 = 64;
    let h: u32 = 64;
    let cw = w / 2;
    let ch = h / 2;
    // Low-variance Y: a 16-step gradient. The HF subbands stay small.
    let mut y = vec![0u8; (w * h) as usize];
    for row in 0..h as usize {
        for col in 0..w as usize {
            y[row * w as usize + col] = 100u8 + ((row + col) as u8 / 8);
        }
    }
    let u = vec![128u8; (cw * ch) as usize];
    let v = vec![128u8; (cw * ch) as usize];
    let seq = make_minimal_sequence(w, h, ChromaFormat::Yuv420);
    let params = EncoderParams::default_hq(WaveletFilter::Fidelity, 3);
    let stream = encode_single_hq_intra_stream(&seq, &params, 0, &y, &u, &v);
    let frame = decode_one(stream);
    let p_y = psnr(&frame.planes[0].data, &y);
    eprintln!("HQ q=0 Fidelity (low-variance): Y PSNR = {p_y:.2} dB");
    assert!(
        p_y > 35.0,
        "Fidelity Y PSNR {p_y:.2} dB < 35 dB on low-variance gradient"
    );
}

/// LD profile across two wavelets at qindex=0: with a generous
/// per-slice budget the LD path should also achieve high-PSNR
/// reconstruction. Cover LeGall (the default) and Haar0 (a
/// fundamentally different shift convention — `filter_shift=0`).
#[test]
fn ld_q0_high_psnr_across_two_wavelets() {
    let w: u32 = 64;
    let h: u32 = 64;
    let cw = w / 2;
    let ch = h / 2;
    // Smooth content: a tight LD slice budget can encode this losslessly.
    let mut y = vec![0u8; (w * h) as usize];
    for row in 0..h as usize {
        for col in 0..w as usize {
            y[row * w as usize + col] = ((row + col) * 2) as u8;
        }
    }
    let u = vec![128u8; (cw * ch) as usize];
    let v = vec![128u8; (cw * ch) as usize];
    let seq = make_minimal_sequence_ld(w, h, ChromaFormat::Yuv420);
    for filter in [WaveletFilter::LeGall5_3, WaveletFilter::Haar0] {
        // Generous 256 bytes per slice on a 4x4 grid = 4 KiB / picture.
        let params = LdEncoderParams::default_ld(filter, 3, 4, 4, 256);
        let stream = encode_single_ld_intra_stream(&seq, &params, 0, &y, &u, &v);
        let frame = decode_one(stream);
        let p_y = psnr(&frame.planes[0].data, &y);
        eprintln!("LD q=0 {filter:?}: Y PSNR = {p_y:.2} dB");
        assert!(
            p_y > 35.0,
            "{filter:?}: LD Y PSNR {p_y:.2} dB < 35 dB on a smooth gradient"
        );
    }
}

/// HQ at moderate quantisation: a smooth gradient should survive
/// qindex=12 with PSNR comfortably above 30 dB across all six
/// supported wavelets. Catches a quant-matrix lookup bug or a
/// per-subband-quantiser regression that only fires when the qindex
/// is not zero.
#[test]
fn hq_q12_psnr_above_30db_across_six_wavelets() {
    let w: u32 = 64;
    let h: u32 = 64;
    let cw = w / 2;
    let ch = h / 2;
    // A smooth gradient — quantisation noise should be tiny.
    let mut y = vec![0u8; (w * h) as usize];
    for row in 0..h as usize {
        for col in 0..w as usize {
            y[row * w as usize + col] = ((row + col) * 2) as u8;
        }
    }
    let u = vec![128u8; (cw * ch) as usize];
    let v = vec![128u8; (cw * ch) as usize];
    let seq = make_minimal_sequence(w, h, ChromaFormat::Yuv420);
    for filter in [
        WaveletFilter::DeslauriersDubuc9_7,
        WaveletFilter::LeGall5_3,
        WaveletFilter::DeslauriersDubuc13_7,
        WaveletFilter::Haar0,
        WaveletFilter::Haar1,
        WaveletFilter::Daubechies9_7,
    ] {
        let mut params = EncoderParams::default_hq(filter, 3);
        params.qindex = 12;
        let stream = encode_single_hq_intra_stream(&seq, &params, 0, &y, &u, &v);
        let frame = decode_one(stream);
        let p_y = psnr(&frame.planes[0].data, &y);
        eprintln!("HQ q=12 {filter:?}: Y PSNR = {p_y:.2} dB");
        assert!(
            p_y > 30.0,
            "{filter:?}: Y PSNR {p_y:.2} dB < 30 dB at qindex=12 on a smooth gradient"
        );
    }
}
