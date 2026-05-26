//! VC-2 LD multi-picture rate-controlled sequence encoder.
//!
//! r134 builds a stream-level driver on top of the r131 per-picture
//! `pick_ld_picture_qindex` primitive: given a sequence of YUV frames
//! plus a per-picture (or CBR) target byte-budget, it encodes each
//! picture with the auto-qindex picker and emits a complete VC-2 LD
//! elementary stream (sequence header + per-picture parse-info + picture
//! data + end-of-sequence) that round-trips through the decoder.
//!
//! Three rate-control modes:
//!   * `PerPicture` — every picture independently sized to `target_bytes`
//!     (±10% each), no carry-over.
//!   * `Cbr` — a signed accumulator carries each picture's byte
//!     over/undershoot into the next picture's budget so the stream total
//!     lands within ±5% of `N * target_bytes`.
//!   * `Vbv { buffer_bytes }` (r149) — leaky-bucket variant of `Cbr`:
//!     identical carry behaviour but the spendable savings are clamped
//!     at `buffer_bytes`, so every per-picture request is capped at
//!     `target + buffer_bytes` (peak-size guarantee). `buffer_bytes ==
//!     0` collapses to `PerPicture`; an effectively infinite
//!     `buffer_bytes` coincides with `Cbr`. The LD analogue of r146's
//!     `HqRateControl::Vbv`.
//!
//! Source of truth: SMPTE ST 2042-1 §13.5.3.2 (slice byte budget),
//! §13.5.2 (per-slice qindex header), §D.1.1 (LD parse-code restriction),
//! §9.6 / §10.4 (parse-info framing). Spec PDF in
//! `docs/video/dirac/dirac-spec-latest.pdf` only.

use oxideav_core::{CodecId, CodecParameters, CodecRegistry, Frame, Packet, TimeBase};
use oxideav_dirac::encoder::{
    encode_ld_sequence_with_size_target, encode_ld_sequence_with_size_target_report,
    make_minimal_sequence_ld, InputPicture, LdEncoderParams, LdRateControl,
};
use oxideav_dirac::video_format::ChromaFormat;
use oxideav_dirac::wavelet::WaveletFilter;

/// One 64x64 4:2:0 frame whose luma content varies with `seed` so the
/// three frames in a sequence are genuinely distinct (different qindex
/// pressure per frame). Chroma carries a mild gradient.
fn frame_64(seed: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; 64 * 64];
    let mut u = vec![128u8; 32 * 32];
    let mut v = vec![128u8; 32 * 32];
    for row in 0..64 {
        for col in 0..64 {
            // Diagonal gradient + a seed-dependent phase so each frame
            // differs but all stay smooth enough to be codeable.
            y[row * 64 + col] = ((((row + col) as u32 * 4) + seed * 7) & 0xFF) as u8;
        }
    }
    for row in 0..32 {
        for col in 0..32 {
            u[row * 32 + col] = 128u8.wrapping_add(((col as i32 / 2) - 8) as u8);
            v[row * 32 + col] = 128u8.wrapping_add(((row as i32 / 2) - 8) as u8);
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
    // Pull frames until the decoder stops producing them. The decoder
    // returns an error (or a non-video frame) once the pending queue is
    // drained; we stop on the first failure.
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

/// THE r134 acceptance test (PerPicture leg). A 3-frame sequence at a
/// fixed per-picture budget:
///   (a) round-trips to exactly 3 decoded frames,
///   (b) each decoded frame's source picture lands within ±10% of the
///       per-picture target on actual payload bytes, and
///   (c) every decoded frame is a valid (finite, non-garbage) picture.
#[test]
fn ld_sequence_per_picture_three_frames_within_10pct_each() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let f0 = frame_64(0);
    let f1 = frame_64(3);
    let f2 = frame_64(9);
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

    let base = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 32);
    let target = 1024u32;

    let (stream, report) = encode_ld_sequence_with_size_target_report(
        &seq,
        &base,
        &frames,
        target,
        LdRateControl::PerPicture,
    );

    assert_eq!(report.len(), 3, "expected one rate report per frame");

    // (b) each picture within ±10% of the per-picture target.
    let lo = (target as f64 * 0.9).floor() as i64;
    let hi = (target as f64 * 1.1).ceil() as i64;
    for r in &report {
        assert!(
            (r.actual_payload_bytes as i64) >= lo && (r.actual_payload_bytes as i64) <= hi,
            "picture {} actual {} outside ±10% of {target} ({lo}..={hi}); qindex={}",
            r.picture_number,
            r.actual_payload_bytes,
            r.qindex,
        );
        assert!(r.qindex <= 127);
    }

    // (a) round-trips to 3 frames.
    let decoded = decode_all(stream);
    assert_eq!(
        decoded.len(),
        3,
        "3-frame LD sequence must decode to 3 frames, got {}",
        decoded.len()
    );

    // (c) each frame is a valid reconstruction of its source.
    let sources = [&f0, &f1, &f2];
    for (i, frame) in decoded.iter().enumerate() {
        let p = psnr(&frame.planes[0].data, &sources[i].0);
        assert!(
            p.is_finite() && p > 6.0,
            "frame {i}: Y PSNR {p:.2} dB — decode produced garbage",
        );
    }

    eprintln!(
        "LD per-picture: target={target} actuals={:?} qindexes={:?}",
        report
            .iter()
            .map(|r| r.actual_payload_bytes)
            .collect::<Vec<_>>(),
        report.iter().map(|r| r.qindex).collect::<Vec<_>>(),
    );
}

/// THE r134 acceptance test (CBR leg). With the carry-over accumulator,
/// the stream *total* (every picture payload summed) lands within ±5% of
/// `N * target_bytes` — tighter than the ±10% per-picture tolerance,
/// because the accumulator pays back each picture's ±1-byte residual.
#[test]
fn ld_sequence_cbr_total_within_5pct_of_n_times_target() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let f0 = frame_64(0);
    let f1 = frame_64(5);
    let f2 = frame_64(11);
    let f3 = frame_64(17);
    let f4 = frame_64(23);
    let triples = [&f0, &f1, &f2, &f3, &f4];
    let frames: Vec<InputPicture> = triples
        .iter()
        .enumerate()
        .map(|(i, t)| InputPicture {
            picture_number: i as u32,
            y: &t.0,
            u: &t.1,
            v: &t.2,
        })
        .collect();

    let base = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 32);
    let target = 900u32;
    let n = frames.len() as i64;

    let (stream, report) = encode_ld_sequence_with_size_target_report(
        &seq,
        &base,
        &frames,
        target,
        LdRateControl::Cbr,
    );

    let total_payload: i64 = report.iter().map(|r| r.actual_payload_bytes as i64).sum();
    let ideal = n * target as i64;
    let lo = (ideal as f64 * 0.95).floor() as i64;
    let hi = (ideal as f64 * 1.05).ceil() as i64;
    assert!(
        total_payload >= lo && total_payload <= hi,
        "CBR total payload {total_payload} outside ±5% of N*target {ideal} ({lo}..={hi}); per-pic={:?}",
        report.iter().map(|r| r.actual_payload_bytes).collect::<Vec<_>>(),
    );

    // Still round-trips to N frames.
    let decoded = decode_all(stream);
    assert_eq!(
        decoded.len(),
        frames.len(),
        "CBR stream must decode to N frames"
    );

    eprintln!(
        "LD CBR: N={n} target={target} ideal={ideal} total={total_payload} per-pic={:?} qindexes={:?}",
        report.iter().map(|r| r.actual_payload_bytes).collect::<Vec<_>>(),
        report.iter().map(|r| r.qindex).collect::<Vec<_>>(),
    );
}

/// The thin `encode_ld_sequence_with_size_target` wrapper must produce a
/// byte-identical stream to the `_report` variant it delegates to.
#[test]
fn ld_sequence_wrapper_matches_report_variant() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let f0 = frame_64(1);
    let f1 = frame_64(2);
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
    ];
    let base = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 32);

    for mode in [
        LdRateControl::PerPicture,
        LdRateControl::Cbr,
        LdRateControl::Vbv { buffer_bytes: 0 },
        LdRateControl::Vbv { buffer_bytes: 128 },
        LdRateControl::Vbv {
            buffer_bytes: u32::MAX,
        },
    ] {
        let plain = encode_ld_sequence_with_size_target(&seq, &base, &frames, 1024, mode);
        let (with_report, _) =
            encode_ld_sequence_with_size_target_report(&seq, &base, &frames, 1024, mode);
        assert_eq!(
            plain, with_report,
            "wrapper diverged from _report for {mode:?}"
        );
    }
}

/// Determinism: same inputs always produce the same stream + report.
#[test]
fn ld_sequence_is_deterministic() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
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
    let base = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 32);

    let a = encode_ld_sequence_with_size_target(&seq, &base, &frames, 1024, LdRateControl::Cbr);
    let b = encode_ld_sequence_with_size_target(&seq, &base, &frames, 1024, LdRateControl::Cbr);
    assert_eq!(a, b, "CBR sequence encode must be deterministic");
}

/// An empty frame list yields a valid sequence-header + end-of-sequence
/// stream with no pictures — and decodes to zero frames without error.
#[test]
fn ld_sequence_empty_frames_is_valid() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let base = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 32);
    let (stream, report) = encode_ld_sequence_with_size_target_report(
        &seq,
        &base,
        &[],
        1024,
        LdRateControl::PerPicture,
    );
    assert!(report.is_empty());
    let decoded = decode_all(stream);
    assert!(decoded.is_empty(), "empty sequence must decode to 0 frames");
}

/// CBR drives the *cumulative* total tighter than PerPicture when budgets
/// would otherwise drift: across many frames the CBR total deviation must
/// be no worse than the PerPicture total deviation from the ideal.
#[test]
fn ld_sequence_cbr_total_no_worse_than_per_picture() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let triples: Vec<_> = (0..6).map(|s| frame_64(s * 3)).collect();
    let frames: Vec<InputPicture> = triples
        .iter()
        .enumerate()
        .map(|(i, t)| InputPicture {
            picture_number: i as u32,
            y: &t.0,
            u: &t.1,
            v: &t.2,
        })
        .collect();
    let base = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 32);
    let target = 777u32; // non-round target maximises per-picture residual
    let ideal = frames.len() as i64 * target as i64;

    let (_, pp) = encode_ld_sequence_with_size_target_report(
        &seq,
        &base,
        &frames,
        target,
        LdRateControl::PerPicture,
    );
    let (_, cbr) = encode_ld_sequence_with_size_target_report(
        &seq,
        &base,
        &frames,
        target,
        LdRateControl::Cbr,
    );

    let pp_total: i64 = pp.iter().map(|r| r.actual_payload_bytes as i64).sum();
    let cbr_total: i64 = cbr.iter().map(|r| r.actual_payload_bytes as i64).sum();

    assert!(
        (cbr_total - ideal).abs() <= (pp_total - ideal).abs(),
        "CBR total deviation {} should be <= PerPicture deviation {} (ideal {ideal}, cbr {cbr_total}, pp {pp_total})",
        (cbr_total - ideal).abs(),
        (pp_total - ideal).abs(),
    );
}

// -- r149: VBV (leaky-bucket) LdRateControl variant --
//
// `LdRateControl::Vbv { buffer_bytes }` is the third LD strategy:
// identical to `Cbr` in its carry semantics, BUT the spendable savings
// (`max(-carry, 0)`) are clamped at `buffer_bytes`, so every per-picture
// request is capped at `target + buffer_bytes` — an instantaneous peak
// size guarantee that uncapped `Cbr` lacks. Four corner cases pin the
// contract: `buffer_bytes == 0` must collapse to `PerPicture`; a very
// large `buffer_bytes` must agree with `Cbr` when every picture
// undershoots (the cap never bites); the peak-cap promise (`requested
// ≤ target + buffer_bytes`) must hold at every intermediate
// `buffer_bytes`; and re-runs must be deterministic.

/// `Vbv { buffer_bytes: 0 }` degenerates to `PerPicture`: the leaky
/// bucket holds zero spendable savings, so the request collapses to
/// `target_bytes` on every picture and the emitted streams are
/// byte-identical. Note: LD's `Cbr` accumulator can also pull the
/// request *below* target when a picture overshoots its deterministic
/// budget (carry-debt branch); the zero-buffer Vbv preserves that
/// debt-payback branch — only the spendable-savings (undershoot)
/// branch is gated by the bucket. On the smooth fixtures here the
/// LD picker hits its ±1-byte residual within the budget, so the
/// debt branch is dormant and Vbv {0} == PerPicture is a clean
/// equality.
#[test]
fn ld_sequence_vbv_zero_buffer_equals_per_picture() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let triples: Vec<_> = (0..4).map(|s| frame_64(s * 3 + 2)).collect();
    let frames: Vec<InputPicture> = triples
        .iter()
        .enumerate()
        .map(|(i, t)| InputPicture {
            picture_number: i as u32,
            y: &t.0,
            u: &t.1,
            v: &t.2,
        })
        .collect();
    let base = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 32);
    let target = 1024u32;

    let (pp_stream, pp_report) = encode_ld_sequence_with_size_target_report(
        &seq,
        &base,
        &frames,
        target,
        LdRateControl::PerPicture,
    );
    let (vbv_stream, vbv_report) = encode_ld_sequence_with_size_target_report(
        &seq,
        &base,
        &frames,
        target,
        LdRateControl::Vbv { buffer_bytes: 0 },
    );

    assert_eq!(
        pp_stream, vbv_stream,
        "Vbv {{ buffer_bytes: 0 }} must produce a byte-identical stream to PerPicture",
    );
    let pp_q: Vec<_> = pp_report.iter().map(|r| r.qindex).collect();
    let vbv_q: Vec<_> = vbv_report.iter().map(|r| r.qindex).collect();
    assert_eq!(pp_q, vbv_q, "qindex sequences must match");
    let pp_req: Vec<_> = pp_report.iter().map(|r| r.requested_bytes).collect();
    let vbv_req: Vec<_> = vbv_report.iter().map(|r| r.requested_bytes).collect();
    assert_eq!(
        pp_req, vbv_req,
        "every VBV request must equal target (zero-buffer carry is unspendable)",
    );
    for r in &vbv_req {
        assert_eq!(*r, target);
    }
}

/// Peak-cap invariant: every Vbv requested bytes ≤ `target +
/// buffer_bytes`. The bucket clamp guarantees the picker is never
/// asked for more than `target + buffer_bytes` on any picture,
/// regardless of how deep the undershoots from earlier pictures
/// accumulated. Uses a tiny `min_budget`-aware buffer so the cap
/// actually bites within the test run.
#[test]
fn ld_sequence_vbv_request_capped_at_target_plus_buffer() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let triples: Vec<_> = (0..6).map(|s| frame_64(s * 7 + 3)).collect();
    let frames: Vec<InputPicture> = triples
        .iter()
        .enumerate()
        .map(|(i, t)| InputPicture {
            picture_number: i as u32,
            y: &t.0,
            u: &t.1,
            v: &t.2,
        })
        .collect();
    let base = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 32);
    // Small target + small bucket; with LD's tight residuals the carry
    // is mostly noise but the cap must hold deterministically.
    let target = 600u32;
    let buffer = 64u32;
    let cap = target + buffer;

    let (stream, report) = encode_ld_sequence_with_size_target_report(
        &seq,
        &base,
        &frames,
        target,
        LdRateControl::Vbv {
            buffer_bytes: buffer,
        },
    );

    for r in &report {
        assert!(
            r.requested_bytes <= cap,
            "picture {} VBV request {} exceeded peak cap target+buffer={} (target={target}, buffer={buffer})",
            r.picture_number,
            r.requested_bytes,
            cap,
        );
    }

    // Stream is still well-formed.
    let decoded = decode_all(stream);
    assert_eq!(
        decoded.len(),
        frames.len(),
        "VBV stream must decode to N frames",
    );

    eprintln!(
        "LD VBV cap test: target={target} buffer={buffer} cap={cap} requests={:?} actuals={:?}",
        report.iter().map(|r| r.requested_bytes).collect::<Vec<_>>(),
        report
            .iter()
            .map(|r| r.actual_payload_bytes)
            .collect::<Vec<_>>(),
    );
}

/// `Vbv { buffer_bytes: u32::MAX }` (effectively infinite bucket)
/// agrees with `Cbr`: the bucket cap can never bite (savings would
/// have to exceed 4 GB to leak), so the leaky-bucket and the uncapped
/// CBR produce byte-identical streams over any reasonable test run.
/// Pins the "Vbv is the bucket-capped strict generalisation of Cbr"
/// invariant on the LD side, matching r146's HQ guarantee.
#[test]
fn ld_sequence_vbv_infinite_buffer_equals_cbr() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let triples: Vec<_> = (0..5).map(|s| frame_64(s * 11 + 4)).collect();
    let frames: Vec<InputPicture> = triples
        .iter()
        .enumerate()
        .map(|(i, t)| InputPicture {
            picture_number: i as u32,
            y: &t.0,
            u: &t.1,
            v: &t.2,
        })
        .collect();
    let base = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 32);
    let target = 900u32;

    let cbr_stream =
        encode_ld_sequence_with_size_target(&seq, &base, &frames, target, LdRateControl::Cbr);
    let vbv_stream = encode_ld_sequence_with_size_target(
        &seq,
        &base,
        &frames,
        target,
        LdRateControl::Vbv {
            buffer_bytes: u32::MAX,
        },
    );
    assert_eq!(
        cbr_stream, vbv_stream,
        "Vbv with u32::MAX bucket must coincide with Cbr (cap never bites)",
    );
}

/// VBV positive-use case: a mid-range budget with a modest bucket
/// produces a valid, decodable stream and obeys the peak cap on every
/// per-picture request. Pins the "leaky-bucket smoothing produces a
/// usable encode" end-to-end behaviour.
#[test]
fn ld_sequence_vbv_smooths_with_bounded_peak() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let triples: Vec<_> = (0..5).map(|s| frame_64(s * 13 + 1)).collect();
    let frames: Vec<InputPicture> = triples
        .iter()
        .enumerate()
        .map(|(i, t)| InputPicture {
            picture_number: i as u32,
            y: &t.0,
            u: &t.1,
            v: &t.2,
        })
        .collect();
    let base = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 32);
    let target = 900u32;
    let buffer = target / 4;
    let cap = target + buffer;

    let (stream, report) = encode_ld_sequence_with_size_target_report(
        &seq,
        &base,
        &frames,
        target,
        LdRateControl::Vbv {
            buffer_bytes: buffer,
        },
    );

    for r in &report {
        assert!(
            r.requested_bytes <= cap,
            "picture {} VBV requested {} exceeded peak cap {cap}",
            r.picture_number,
            r.requested_bytes,
        );
    }

    let decoded = decode_all(stream);
    assert_eq!(
        decoded.len(),
        frames.len(),
        "VBV stream must decode to N frames",
    );
}

/// Determinism: the same VBV inputs always produce the same stream.
#[test]
fn ld_sequence_vbv_is_deterministic() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let f0 = frame_64(2);
    let f1 = frame_64(8);
    let f2 = frame_64(32);
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
    let base = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 32);
    let a = encode_ld_sequence_with_size_target(
        &seq,
        &base,
        &frames,
        1024,
        LdRateControl::Vbv { buffer_bytes: 128 },
    );
    let b = encode_ld_sequence_with_size_target(
        &seq,
        &base,
        &frames,
        1024,
        LdRateControl::Vbv { buffer_bytes: 128 },
    );
    assert_eq!(a, b, "VBV encode must be deterministic");
}

// ---------------------------------------------------------------------------
// running_surplus_bytes telemetry (r152)
//
// Every `LdPictureRate` carries a signed `running_surplus_bytes` reported
// AFTER any VBV bucket clamp. Convention:
//   * positive = cumulative savings (future pictures may spend it),
//   * negative = cumulative debt (future pictures must pay it back).
// Computed identically across modes: `pictures_seen × target − Σ actual`
// folded picture-by-picture, with VBV additionally clamping savings at
// `buffer_bytes`. Modes differ only in whether the next request uses it.
// ---------------------------------------------------------------------------

/// PerPicture: the accumulator is still tracked as an observable signed
/// `target × seen − Σ actual` running sum (mode-agnostic), but the
/// request derivation ignores it — the *reported* surplus must match
/// the cumulative budget-minus-actual sum, not be silently zero.
#[test]
fn ld_sequence_per_picture_running_surplus_matches_cumulative_budget_minus_actual() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let triples: Vec<_> = (0..5).map(|s| frame_64(s * 11 + 1)).collect();
    let frames: Vec<InputPicture> = triples
        .iter()
        .enumerate()
        .map(|(i, t)| InputPicture {
            picture_number: i as u32,
            y: &t.0,
            u: &t.1,
            v: &t.2,
        })
        .collect();
    let base = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 32);
    let target: u32 = 900;

    let (_stream, report) = encode_ld_sequence_with_size_target_report(
        &seq,
        &base,
        &frames,
        target,
        LdRateControl::PerPicture,
    );

    let mut cum_actual: i64 = 0;
    for (k, r) in report.iter().enumerate() {
        cum_actual += r.actual_payload_bytes as i64;
        let cum_budget = (k as i64 + 1) * target as i64;
        let expected_surplus = cum_budget - cum_actual;
        assert_eq!(
            r.running_surplus_bytes, expected_surplus,
            "PerPicture: picture {} surplus mismatch — got {}, expected (k+1)*target - Σactual = {}",
            r.picture_number, r.running_surplus_bytes, expected_surplus,
        );
    }
}

/// Cbr: the reported `running_surplus_bytes` after picture `k`
/// equals `(k+1) × target − Σ actual_payload_bytes[0..=k]` — i.e. the
/// signed deviation of the ideal cumulative budget from the
/// actually-encoded cumulative bytes, in "positive = savings" sign.
/// Pins the explicit accumulator semantics across the stream.
#[test]
fn ld_sequence_cbr_running_surplus_matches_cumulative_budget_minus_actual() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let triples: Vec<_> = (0..5).map(|s| frame_64(s * 13 + 7)).collect();
    let frames: Vec<InputPicture> = triples
        .iter()
        .enumerate()
        .map(|(i, t)| InputPicture {
            picture_number: i as u32,
            y: &t.0,
            u: &t.1,
            v: &t.2,
        })
        .collect();
    let base = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 32);
    let target: u32 = 950;

    let (_stream, report) = encode_ld_sequence_with_size_target_report(
        &seq,
        &base,
        &frames,
        target,
        LdRateControl::Cbr,
    );

    let mut cum_actual: i64 = 0;
    for (k, r) in report.iter().enumerate() {
        cum_actual += r.actual_payload_bytes as i64;
        let cum_budget = (k as i64 + 1) * target as i64;
        let expected_surplus = cum_budget - cum_actual;
        assert_eq!(
            r.running_surplus_bytes, expected_surplus,
            "Cbr: picture {} surplus mismatch — got {}, expected (k+1)*target - Σactual = {}",
            r.picture_number, r.running_surplus_bytes, expected_surplus,
        );
    }
}

/// VBV: the bucket clamp guarantees `running_surplus_bytes ≤
/// buffer_bytes` for every reported row (savings above the bucket are
/// forfeited). The lower edge — debt — is not clamped; only the
/// savings end of the bucket is bounded.
#[test]
fn ld_sequence_vbv_running_surplus_bounded_above_by_buffer_bytes() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let triples: Vec<_> = (0..6).map(|s| frame_64(s * 5 + 2)).collect();
    let frames: Vec<InputPicture> = triples
        .iter()
        .enumerate()
        .map(|(i, t)| InputPicture {
            picture_number: i as u32,
            y: &t.0,
            u: &t.1,
            v: &t.2,
        })
        .collect();
    let base = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 32);
    // Pick a target generous enough that the LD encoder undershoots
    // most pictures so the savings end of the bucket actually fills.
    let target: u32 = 2000;
    let buffer: u32 = 128;

    let (_stream, report) = encode_ld_sequence_with_size_target_report(
        &seq,
        &base,
        &frames,
        target,
        LdRateControl::Vbv {
            buffer_bytes: buffer,
        },
    );

    for r in &report {
        assert!(
            r.running_surplus_bytes <= buffer as i64,
            "VBV: picture {} surplus {} exceeds bucket cap {} after clamp",
            r.picture_number,
            r.running_surplus_bytes,
            buffer,
        );
    }
}
