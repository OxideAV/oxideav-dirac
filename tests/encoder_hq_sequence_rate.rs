//! VC-2 HQ multi-picture rate-controlled sequence encoder.
//!
//! r141 builds a stream-level driver on top of the r138 per-picture
//! `pick_hq_picture_qindex` primitive: given a sequence of YUV frames
//! plus a per-picture (or CBR) target byte-budget, it encodes each
//! picture with the auto-qindex picker and emits a complete VC-2 HQ
//! elementary stream (sequence header + per-picture parse-info + picture
//! data + end-of-sequence) that round-trips through the decoder.
//!
//! Two rate-control modes:
//!   * `PerPicture` — every picture independently sized to `target_bytes`,
//!     no carry-over. The picker never overshoots, so each picture's
//!     actual bytes ≤ target (within the q=127 floor edge case).
//!   * `Cbr` — a signed carry accumulator adds each picture's
//!     undershoot to the next picture's request, letting subsequent
//!     pictures spend the surplus.
//!
//! Source of truth: BBC Dirac Specification v2.2.3 §13.5.2 (per-slice
//! qindex header), §13.5.4 (`slice_quantisers(qindex)`), §9.6 / §10.4
//! (parse-info sequence framing). Spec PDF in
//! `docs/video/dirac/dirac-spec-latest.pdf` only.

use oxideav_core::{CodecId, CodecParameters, CodecRegistry, Frame, Packet, TimeBase};
use oxideav_dirac::encoder::{
    encode_hq_sequence_with_size_target, encode_hq_sequence_with_size_target_report,
    hq_picture_payload_bytes_at_qindex, make_minimal_sequence, EncoderParams, HqRateControl,
    InputPicture,
};
use oxideav_dirac::video_format::ChromaFormat;
use oxideav_dirac::wavelet::WaveletFilter;

/// One 64x64 4:2:0 frame with mid-frequency content whose phase varies
/// with `seed`. Enough energy that q=0 produces a multi-hundred-byte
/// picture but a tight budget can still escalate the picker.
fn frame_64(seed: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; 64 * 64];
    let mut u = vec![128u8; 32 * 32];
    let mut v = vec![128u8; 32 * 32];
    for row in 0..64 {
        for col in 0..64 {
            let base = ((((row + col) as u32 * 3) + seed * 5) & 0xFF) as u8;
            // A seed-dependent cross adds high-pass energy.
            let cross = if (row == ((31 + seed as usize) & 63) && (16..48).contains(&col))
                || (col == ((31 + seed as usize) & 63) && (16..48).contains(&row))
            {
                220
            } else {
                base
            };
            y[row * 64 + col] = cross;
        }
    }
    for row in 0..32 {
        for col in 0..32 {
            u[row * 32 + col] = 128u8.wrapping_add(((col as i32 - 16) * 2) as u8);
            v[row * 32 + col] = 128u8.wrapping_add(((row as i32 - 16) * 2) as u8);
        }
    }
    (y, u, v)
}

/// Decode every picture in `stream` by repeated `receive_frame` until
/// the decoder reports EOF / no more frames.
fn decode_all(stream: Vec<u8>) -> Vec<oxideav_core::VideoFrame> {
    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("make decoder");
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&pkt).expect("send packet");

    let mut frames = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => frames.push(v),
            Ok(_) => break,
            Err(_) => break,
        }
    }
    frames
}

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

fn base_hq_params() -> EncoderParams {
    let mut p = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    p.slices_x = 4;
    p.slices_y = 4;
    // `slice_size_scaler = 2` doubles the per-slice length-byte
    // capacity from 255 to 510 bytes per component. The 4x4 slice grid
    // over a 64x64 picture at q=127 with the high-energy synthetic
    // crosses below can otherwise push a single slice past the 255 B
    // §13.5.4 length-byte cap. The picker stops at qindex 127 for
    // unfit targets; bumping the scaler keeps the q=127 floor encode
    // valid for these fixtures.
    p.slice_size_scaler = 2;
    p
}

/// THE r141 acceptance test (PerPicture leg). A 5-picture HQ sequence
/// at a fixed per-picture budget:
///   (a) round-trips to exactly 5 decoded frames,
///   (b) every picture's actual payload bytes ≤ its target (the picker
///       never overshoots; equality is rare but allowed), and
///   (c) every decoded frame reconstructs its source at non-trivial
///       PSNR.
#[test]
fn hq_sequence_per_picture_five_frames_within_budget_each() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let sources: Vec<_> = (0..5).map(|s| frame_64(s * 4 + 1)).collect();
    let frames: Vec<InputPicture> = sources
        .iter()
        .enumerate()
        .map(|(i, t)| InputPicture {
            picture_number: i as u32,
            y: &t.0,
            u: &t.1,
            v: &t.2,
        })
        .collect();

    let base = base_hq_params();

    // Choose a budget that sits between q=0 and q=127 for typical
    // content so the picker actually exercises (rather than always
    // returning 0). Reference the first frame's q=0 ceiling and pick
    // ~60% of it.
    let q0 = hq_picture_payload_bytes_at_qindex(
        &seq,
        &base,
        &sources[0].0,
        &sources[0].1,
        &sources[0].2,
        0,
    ) as u32;
    let target = (q0 * 3 / 5).max(64);

    let (stream, report) = encode_hq_sequence_with_size_target_report(
        &seq,
        &base,
        &frames,
        target,
        HqRateControl::PerPicture,
    );

    assert_eq!(report.len(), 5, "expected one rate report per frame");

    // (b) each picture ≤ target (unless q=127 forced overshoot).
    for r in &report {
        if r.qindex < 127 {
            assert!(
                r.actual_payload_bytes <= target,
                "picture {} actual {} > target {target} but qindex {} < 127 — picker should have escalated",
                r.picture_number,
                r.actual_payload_bytes,
                r.qindex,
            );
        }
        assert!(r.qindex <= 127);
        // Parse code alternates 0xE8 / 0xEC.
        let expected = if r.picture_number % 2 == 0 {
            0xE8
        } else {
            0xEC
        };
        assert_eq!(
            r.parse_code, expected,
            "picture {} parse code mismatch",
            r.picture_number
        );
    }

    // (a) round-trips to 5 frames.
    let decoded = decode_all(stream);
    assert_eq!(
        decoded.len(),
        5,
        "5-frame HQ sequence must decode to 5 frames, got {}",
        decoded.len()
    );

    // (c) each frame is a valid reconstruction of its source.
    for (i, frame) in decoded.iter().enumerate() {
        let p = psnr(&frame.planes[0].data, &sources[i].0);
        assert!(
            p.is_finite() && p > 8.0,
            "frame {i}: Y PSNR {p:.2} dB — decode produced garbage",
        );
    }

    eprintln!(
        "HQ per-picture: target={target} q0_pic0={q0} actuals={:?} qindexes={:?}",
        report
            .iter()
            .map(|r| r.actual_payload_bytes)
            .collect::<Vec<_>>(),
        report.iter().map(|r| r.qindex).collect::<Vec<_>>(),
    );
}

/// THE r141 acceptance test (CBR leg). With the carry-over accumulator,
/// every picture's request is `target + carry` where `carry` is the
/// running sum of previous undershoots. The HQ picker never overshoots,
/// so on a budget tight enough to escalate qindex on the first picture,
/// the CBR total payload is at least as close to `N * target` as the
/// PerPicture total (and typically closer, because undershoot savings
/// get spent rather than discarded).
#[test]
fn hq_sequence_cbr_total_no_worse_than_per_picture() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let sources: Vec<_> = (0..5).map(|s| frame_64(s * 7 + 3)).collect();
    let frames: Vec<InputPicture> = sources
        .iter()
        .enumerate()
        .map(|(i, t)| InputPicture {
            picture_number: i as u32,
            y: &t.0,
            u: &t.1,
            v: &t.2,
        })
        .collect();

    let base = base_hq_params();
    let q0 = hq_picture_payload_bytes_at_qindex(
        &seq,
        &base,
        &sources[0].0,
        &sources[0].1,
        &sources[0].2,
        0,
    ) as u32;
    let target = (q0 / 2).max(64); // tight: forces picker escalation
    let n = frames.len() as i64;
    let ideal = n * target as i64;

    let (_, pp) = encode_hq_sequence_with_size_target_report(
        &seq,
        &base,
        &frames,
        target,
        HqRateControl::PerPicture,
    );
    let (cbr_stream, cbr) = encode_hq_sequence_with_size_target_report(
        &seq,
        &base,
        &frames,
        target,
        HqRateControl::Cbr,
    );

    let pp_total: i64 = pp.iter().map(|r| r.actual_payload_bytes as i64).sum();
    let cbr_total: i64 = cbr.iter().map(|r| r.actual_payload_bytes as i64).sum();

    // PerPicture: every picture ≤ target (unless q=127), so total ≤ N*target.
    // CBR: the carry-over spends undershoots, lifting the running total
    // toward N*target. The CBR total must be at least the PerPicture
    // total (carry only adds budget; never removes it).
    assert!(
        cbr_total >= pp_total,
        "CBR total {cbr_total} should be ≥ PerPicture total {pp_total} \
         (CBR carry only grows the per-picture budget; ideal {ideal})",
    );

    // The CBR stream still decodes to N frames.
    let decoded = decode_all(cbr_stream);
    assert_eq!(
        decoded.len(),
        frames.len(),
        "CBR stream must decode to N frames"
    );

    eprintln!(
        "HQ CBR: N={n} target={target} ideal={ideal} pp_total={pp_total} cbr_total={cbr_total} \
         per-pic={:?} qindexes={:?}",
        cbr.iter()
            .map(|r| r.actual_payload_bytes)
            .collect::<Vec<_>>(),
        cbr.iter().map(|r| r.qindex).collect::<Vec<_>>(),
    );
}

/// The thin `encode_hq_sequence_with_size_target` wrapper must produce a
/// byte-identical stream to the `_report` variant it delegates to.
#[test]
fn hq_sequence_wrapper_matches_report_variant() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let f0 = frame_64(1);
    let f1 = frame_64(2);
    let f2 = frame_64(3);
    let frames = [
        InputPicture {
            picture_number: 0,
            y: &f0.0,
            u: &f0.1,
            v: &f0.2,
        },
        InputPicture {
            picture_number: 1,
            y: &f1.0,
            u: &f1.1,
            v: &f1.2,
        },
        InputPicture {
            picture_number: 2,
            y: &f2.0,
            u: &f2.1,
            v: &f2.2,
        },
    ];
    let base = base_hq_params();

    for mode in [HqRateControl::PerPicture, HqRateControl::Cbr] {
        let plain = encode_hq_sequence_with_size_target(&seq, &base, &frames, 1024, mode);
        let (with_report, _) =
            encode_hq_sequence_with_size_target_report(&seq, &base, &frames, 1024, mode);
        assert_eq!(
            plain, with_report,
            "wrapper diverged from _report for {mode:?}"
        );
    }
}

/// Determinism: same inputs always produce the same stream + report.
#[test]
fn hq_sequence_is_deterministic() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let f0 = frame_64(0);
    let f1 = frame_64(4);
    let f2 = frame_64(8);
    let frames = [
        InputPicture {
            picture_number: 0,
            y: &f0.0,
            u: &f0.1,
            v: &f0.2,
        },
        InputPicture {
            picture_number: 1,
            y: &f1.0,
            u: &f1.1,
            v: &f1.2,
        },
        InputPicture {
            picture_number: 2,
            y: &f2.0,
            u: &f2.1,
            v: &f2.2,
        },
    ];
    let base = base_hq_params();

    let a = encode_hq_sequence_with_size_target(&seq, &base, &frames, 1024, HqRateControl::Cbr);
    let b = encode_hq_sequence_with_size_target(&seq, &base, &frames, 1024, HqRateControl::Cbr);
    assert_eq!(a, b, "CBR sequence encode must be deterministic");
}

/// An empty frame list yields a valid sequence-header + end-of-sequence
/// stream with no pictures — and decodes to zero frames without error.
#[test]
fn hq_sequence_empty_frames_is_valid() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let base = base_hq_params();
    let (stream, report) = encode_hq_sequence_with_size_target_report(
        &seq,
        &base,
        &[],
        1024,
        HqRateControl::PerPicture,
    );
    assert!(report.is_empty());
    let decoded = decode_all(stream);
    assert!(decoded.is_empty(), "empty sequence must decode to 0 frames");
}

/// Parse codes alternate `0xE8` (non-reference) on even indices and
/// `0xEC` (reference) on odd indices — same alternation as
/// `encode_hq_intra_multi_stream`. The report's `parse_code` field
/// reflects the byte actually written into the picture's parse-info.
#[test]
fn hq_sequence_alternates_parse_codes() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let sources: Vec<_> = (0..4).map(|s| frame_64(s + 1)).collect();
    let frames: Vec<InputPicture> = sources
        .iter()
        .enumerate()
        .map(|(i, t)| InputPicture {
            picture_number: i as u32,
            y: &t.0,
            u: &t.1,
            v: &t.2,
        })
        .collect();
    let base = base_hq_params();

    let (_, report) = encode_hq_sequence_with_size_target_report(
        &seq,
        &base,
        &frames,
        4096,
        HqRateControl::PerPicture,
    );
    let codes: Vec<u8> = report.iter().map(|r| r.parse_code).collect();
    assert_eq!(codes, vec![0xE8, 0xEC, 0xE8, 0xEC]);
}

/// On a generous per-picture budget every picture lands at qindex 0
/// (its q=0 picture bytes already fit) and the decoded frame matches
/// the source at the codec's intrinsic q=0 fidelity (multi-decibel,
/// content-dependent but well above the recognisability floor).
#[test]
fn hq_sequence_generous_budget_keeps_qindex_zero() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let sources: Vec<_> = (0..3).map(|s| frame_64(s * 2)).collect();
    let frames: Vec<InputPicture> = sources
        .iter()
        .enumerate()
        .map(|(i, t)| InputPicture {
            picture_number: i as u32,
            y: &t.0,
            u: &t.1,
            v: &t.2,
        })
        .collect();
    let base = base_hq_params();

    // Compute the *largest* per-picture q=0 ceiling across all frames
    // and use 2× that as budget — every picture's q=0 picture bytes
    // are far below it.
    let max_q0 = sources
        .iter()
        .map(|s| hq_picture_payload_bytes_at_qindex(&seq, &base, &s.0, &s.1, &s.2, 0))
        .max()
        .unwrap() as u32;
    let generous = max_q0 * 2;

    let (stream, report) = encode_hq_sequence_with_size_target_report(
        &seq,
        &base,
        &frames,
        generous,
        HqRateControl::PerPicture,
    );

    for r in &report {
        assert_eq!(
            r.qindex, 0,
            "generous budget {generous} ≥ 2*max q=0 ({max_q0}) must keep qindex 0; got {} for picture {}",
            r.qindex, r.picture_number,
        );
    }

    let decoded = decode_all(stream);
    assert_eq!(decoded.len(), frames.len());
    for (i, frame) in decoded.iter().enumerate() {
        let p = psnr(&frame.planes[0].data, &sources[i].0);
        // q=0 on this fixture clears ~30 dB Y PSNR (multi-decibel
        // reconstruction; not bit-exact because the dead-zone forward
        // quantiser still rounds magnitudes at qf=4).
        assert!(
            p > 25.0,
            "frame {i} at q=0 budget should reconstruct well; Y PSNR {p:.2} dB",
        );
    }
}

/// CBR carry surplus from undershoot grows the next picture's request:
/// with a budget every picture's q=0 ceiling fits under, every picture
/// undershoots, so the requested budget is monotonically NON-DECREASING
/// across the sequence (carry adds to target each picture). Pins the
/// CBR carry semantics at the function-contract level: the request
/// trend reflects "undershoot becomes future spending power."
#[test]
fn hq_sequence_cbr_requested_bytes_monotone_non_decreasing_under_undershoot() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let sources: Vec<_> = (0..5).map(|s| frame_64(s * 11)).collect();
    let frames: Vec<InputPicture> = sources
        .iter()
        .enumerate()
        .map(|(i, t)| InputPicture {
            picture_number: i as u32,
            y: &t.0,
            u: &t.1,
            v: &t.2,
        })
        .collect();
    let base = base_hq_params();

    // Pick a target that every picture's q=0 ceiling stays under, so
    // every picture undershoots and the carry monotonically grows.
    let max_q0 = sources
        .iter()
        .map(|s| hq_picture_payload_bytes_at_qindex(&seq, &base, &s.0, &s.1, &s.2, 0))
        .max()
        .unwrap() as u32;
    let target = max_q0 + 256; // strictly above every q=0 ceiling

    let (_, cbr) = encode_hq_sequence_with_size_target_report(
        &seq,
        &base,
        &frames,
        target,
        HqRateControl::Cbr,
    );

    let mut prev_req = 0u32;
    for (i, r) in cbr.iter().enumerate() {
        assert!(
            r.requested_bytes >= target,
            "picture {i}: CBR requested {} < target {target} — carry went negative even though every picture undershot",
            r.requested_bytes,
        );
        if i > 0 {
            assert!(
                r.requested_bytes >= prev_req,
                "picture {i}: CBR requested {} < previous {} — carry shrank despite undershoot",
                r.requested_bytes,
                prev_req,
            );
        }
        prev_req = r.requested_bytes;
    }
}
