//! VC-2 HQ picture-level rate-control: target-byte picker.
//!
//! r138 adds the HQ analogue of LD's r131 [`pick_ld_picture_qindex`]:
//! given a target *picture-payload* byte budget, pick the smallest
//! `qindex` for which the entire HQ picture's encoded payload (with that
//! single qindex written into every slice header) fits the budget.
//!
//! Unlike LD — where the picture-byte count is a deterministic function
//! of `slice_bytes_numer/denom` and independent of qindex — the HQ
//! profile lets each slice's length byte track its actual coefficient
//! block size, so picture bytes shrink monotonically as qindex grows.
//! The picker walks `qindex ∈ 0..=127` and stops at the first one whose
//! constant-qindex picture bytes fit.
//!
//! Source of truth: BBC Dirac Specification v2.2.3 §13.5.2 (per-slice
//! qindex header) + §13.5.4 (`slice_quantisers(qindex)`).
//!
//! Wall: spec PDF in `docs/video/dirac/dirac-spec-latest.pdf` only.

use oxideav_core::{CodecId, CodecParameters, CodecRegistry, Frame, Packet, TimeBase};
use oxideav_dirac::encoder::{
    encode_single_hq_intra_stream_with_size_target, hq_picture_payload_bytes_at_qindex,
    hq_picture_qindex_diagnostic, make_minimal_sequence, pick_hq_picture_qindex, EncoderParams,
};
use oxideav_dirac::video_format::ChromaFormat;
use oxideav_dirac::wavelet::WaveletFilter;

/// 64x64 4:2:0 test pattern with mid-frequency content: enough energy
/// that q=0 produces a multi-hundred-byte picture but a tight budget can
/// still escalate the picker to a non-zero qindex.
fn test_triple_64() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; 64 * 64];
    let mut u = vec![128u8; 32 * 32];
    let mut v = vec![128u8; 32 * 32];
    for row in 0..64 {
        for col in 0..64 {
            // Diagonal gradient + a checker cross in the centre raises
            // high-pass energy without saturating.
            let base = (((row + col) as u32 * 3) & 0xFF) as u8;
            let cross = if (row == 31 && (16..48).contains(&col))
                || (col == 31 && (16..48).contains(&row))
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

fn decode_one(stream: Vec<u8>) -> oxideav_core::VideoFrame {
    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("make decoder");
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&pkt).expect("send packet");
    match dec.receive_frame().expect("receive frame") {
        Frame::Video(v) => v,
        other => panic!("expected video frame, got {other:?}"),
    }
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

/// `hq_picture_payload_bytes_at_qindex` matches the actual encoded
/// picture payload length when the encoder is driven with the same
/// `qindex` and `slice_size_target = None` — the predictor is exact, not
/// a heuristic.
#[test]
fn hq_picture_payload_bytes_matches_actual_encode() {
    use oxideav_dirac::encoder::encode_hq_intra_picture;
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let (y, u, v) = test_triple_64();

    for &qindex in &[0u32, 1, 5, 16, 32, 64, 100, 127] {
        let mut params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
        params.slices_x = 4;
        params.slices_y = 4;
        params.qindex = qindex;
        params.slice_size_target = None;

        let predicted = hq_picture_payload_bytes_at_qindex(&seq, &params, &y, &u, &v, qindex);
        let actual = encode_hq_intra_picture(&seq, &params, 0, &y, &u, &v).len();
        assert_eq!(
            actual, predicted,
            "qindex={qindex}: predicted {predicted} != actual {actual}",
        );
    }
}

/// HQ picture bytes are monotonically NON-INCREASING in qindex: a higher
/// qindex (more aggressive dead-zone quantiser) can never produce a
/// larger encoded picture than a lower one. Pins the search-monotonicity
/// the picker relies on.
#[test]
fn hq_picture_bytes_monotone_decreasing_in_qindex() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let (y, u, v) = test_triple_64();
    let mut params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    params.slices_x = 4;
    params.slices_y = 4;

    let mut prev: Option<usize> = None;
    for qindex in 0u32..=127 {
        let bytes = hq_picture_payload_bytes_at_qindex(&seq, &params, &y, &u, &v, qindex);
        if let Some(p) = prev {
            assert!(
                bytes <= p,
                "qindex {qindex} produced {bytes} B, larger than qindex {} ({p} B) — non-monotone",
                qindex - 1,
            );
        }
        prev = Some(bytes);
    }
}

/// THE round-138 acceptance test. Three target picture-byte budgets
/// (small / medium / large) must each:
///   (a) decode end-to-end,
///   (b) land WITHIN the budget — the picker promises ≤ target_bytes
///       (never overshoots; may undershoot when even q=0 fits a generous
///       budget), and
///   (c) the small target must pick a qindex strictly greater than the
///       large target (tight budget → aggressive quantiser).
#[test]
fn hq_picture_qindex_picker_hits_three_budgets() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let (y, u, v) = test_triple_64();

    let mut base = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    base.slices_x = 4;
    base.slices_y = 4;

    // Probe the q=0 and q=127 endpoints so the test picks targets that
    // actually exercise the search range on this content.
    let max_bytes = hq_picture_payload_bytes_at_qindex(&seq, &base, &y, &u, &v, 0);
    let min_bytes = hq_picture_payload_bytes_at_qindex(&seq, &base, &y, &u, &v, 127);
    assert!(
        max_bytes > min_bytes,
        "test fixture is too flat: q=0 picture {max_bytes} B not larger than q=127 {min_bytes} B",
    );

    // Tight budget: just above q=127 floor → picker must escalate well
    // above q=0.
    let tight = min_bytes + ((max_bytes - min_bytes) / 16);
    // Loose budget: just under q=0 ceiling → picker should keep low q.
    let loose = max_bytes;
    let middle = (tight + loose) / 2;

    let targets = [tight as u32, middle as u32, loose as u32];
    let mut chosen_qindexes: Vec<u32> = Vec::new();
    let mut actuals: Vec<usize> = Vec::new();

    for &target in &targets {
        let (stream, qindex, actual_picture) =
            encode_single_hq_intra_stream_with_size_target(&seq, &base, target, 0, &y, &u, &v);

        // Picker promises ≤ target. Equality is achievable only when
        // some quantised picture happens to land exactly on the target.
        assert!(
            actual_picture <= target as usize || qindex == 127,
            "target {target}: actual {actual_picture} > target and qindex={qindex} < 127 (picker should have escalated further)",
        );
        actuals.push(actual_picture);
        chosen_qindexes.push(qindex);

        // The stream must decode end-to-end.
        let frame = decode_one(stream);
        let p_y = psnr(&frame.planes[0].data, &y);
        // p_y == INF is bit-exact reconstruction (best possible); finite
        // values must clear the floor for "recognisable picture".
        assert!(
            p_y > 8.0,
            "target {target}: Y PSNR {p_y:.2} dB — decode produced garbage",
        );
    }

    eprintln!(
        "HQ picture-qindex picker: targets={targets:?} actuals={actuals:?} qindexes={chosen_qindexes:?} (q=0 ceiling {max_bytes} B, q=127 floor {min_bytes} B)",
    );

    // Small target ⇒ qindex strictly higher than large target.
    assert!(
        chosen_qindexes[0] > chosen_qindexes[2],
        "small target qindex {} should exceed large target qindex {} (tighter budget = aggressive quantiser); all = {chosen_qindexes:?}",
        chosen_qindexes[0], chosen_qindexes[2],
    );
}

/// The picker is monotone in `target_bytes`: a smaller target can only
/// push the chosen qindex up (or leave it) — never down. Pins the search
/// direction at the function-contract level.
#[test]
fn hq_picture_qindex_picker_is_monotone_in_target() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let (y, u, v) = test_triple_64();
    let mut base = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    base.slices_x = 4;
    base.slices_y = 4;

    let big = hq_picture_payload_bytes_at_qindex(&seq, &base, &y, &u, &v, 0) as u32;
    let small = (hq_picture_payload_bytes_at_qindex(&seq, &base, &y, &u, &v, 64) as u32).max(1);

    let q_big = pick_hq_picture_qindex(&seq, &base, &y, &u, &v, big);
    let q_small = pick_hq_picture_qindex(&seq, &base, &y, &u, &v, small);
    assert!(
        q_small >= q_big,
        "smaller budget {small} (q={q_small}) should pick qindex ≥ larger budget {big} (q={q_big})",
    );
}

/// On flat content even a tight target fits at q=0 — every slice's
/// coefficients are tiny / mostly zero, so the picker returns q=0.
#[test]
fn hq_picture_qindex_picker_keeps_zero_on_flat_content() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let y = vec![128u8; 64 * 64];
    let u = vec![128u8; 32 * 32];
    let v = vec![128u8; 32 * 32];

    let mut base = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    base.slices_x = 4;
    base.slices_y = 4;

    // q=0 baseline byte count on flat content.
    let flat_q0 = hq_picture_payload_bytes_at_qindex(&seq, &base, &y, &u, &v, 0);
    // A budget of exactly q=0's size must keep qindex 0 (nothing forces
    // escalation; q=0 already fits).
    let q = pick_hq_picture_qindex(&seq, &base, &y, &u, &v, flat_q0 as u32);
    assert_eq!(
        q, 0,
        "flat content fits at q=0 ({flat_q0} B) under budget = flat_q0 → picker must keep q=0",
    );

    // The stream decodes; flat content reconstructs ~bit-exact at q=0.
    let (stream, qindex, _bytes) =
        encode_single_hq_intra_stream_with_size_target(&seq, &base, flat_q0 as u32, 0, &y, &u, &v);
    assert_eq!(qindex, 0);
    let frame = decode_one(stream);
    let p_y = psnr(&frame.planes[0].data, &y);
    assert!(
        p_y > 40.0,
        "flat content should decode near-lossless at q=0 (Y PSNR {p_y:.2} dB)",
    );
}

/// Determinism: same inputs always produce the same `(stream, qindex,
/// actual_bytes)` triple. The picker has no hidden ordering / state.
#[test]
fn hq_picture_qindex_picker_is_deterministic() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let (y, u, v) = test_triple_64();
    let mut base = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    base.slices_x = 4;
    base.slices_y = 4;

    let target = (hq_picture_payload_bytes_at_qindex(&seq, &base, &y, &u, &v, 16) as u32).max(64);

    let (s1, q1, b1) =
        encode_single_hq_intra_stream_with_size_target(&seq, &base, target, 0, &y, &u, &v);
    let (s2, q2, b2) =
        encode_single_hq_intra_stream_with_size_target(&seq, &base, target, 0, &y, &u, &v);

    assert_eq!(s1, s2, "picker must produce byte-identical streams");
    assert_eq!(q1, q2, "picker must produce identical qindex");
    assert_eq!(b1, b2, "picker must report identical picture bytes");
}

/// Even-smaller-than-min target: q=127 picture bytes is the floor. A
/// budget strictly below that floor cannot be satisfied; the picker
/// returns 127 (the most aggressive available quantiser) and the actual
/// picture bytes will overshoot the budget. The picker never panics or
/// produces an invalid stream.
#[test]
fn hq_picture_qindex_picker_returns_127_on_unfit_target() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let (y, u, v) = test_triple_64();
    let mut base = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    base.slices_x = 4;
    base.slices_y = 4;

    // q=127 picture bytes is the smallest the picker can emit.
    let floor = hq_picture_payload_bytes_at_qindex(&seq, &base, &y, &u, &v, 127);
    assert!(floor > 0);

    // Target = floor - 1 (or 1 if floor == 1) is strictly below the floor.
    let unfit = (floor as i64 - 1).max(1) as u32;
    let (stream, qindex, actual) =
        encode_single_hq_intra_stream_with_size_target(&seq, &base, unfit, 0, &y, &u, &v);
    assert_eq!(qindex, 127, "unfit target {unfit} must pick q=127 (floor)");
    assert!(
        actual >= floor,
        "unfit target must report at least the q=127 floor of {floor} B (got {actual})"
    );
    // Stream still decodes — graceful degradation, not encode failure.
    let _ = decode_one(stream);
}

/// Diagnostic mirrors the picker: `(qindex, actual_bytes)` from
/// `hq_picture_qindex_diagnostic` equals
/// `(pick_hq_picture_qindex(...), hq_picture_payload_bytes_at_qindex(..., q))`.
#[test]
fn hq_picture_qindex_diagnostic_mirrors_picker() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let (y, u, v) = test_triple_64();
    let mut base = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    base.slices_x = 4;
    base.slices_y = 4;

    let target = (hq_picture_payload_bytes_at_qindex(&seq, &base, &y, &u, &v, 8) as u32).max(64);

    let q1 = pick_hq_picture_qindex(&seq, &base, &y, &u, &v, target);
    let (q2, bytes2) = hq_picture_qindex_diagnostic(&seq, &base, &y, &u, &v, target);
    assert_eq!(q1, q2);
    let expected_bytes = hq_picture_payload_bytes_at_qindex(&seq, &base, &y, &u, &v, q1);
    assert_eq!(bytes2, expected_bytes);
}

/// `slice_size_target` on the input `base` must be IGNORED by the
/// stream wrapper — the picker uses a single picture-level qindex over
/// every slice and the §13.5.4 per-slice search must not also fire.
/// Pins that the wrapper clears the per-slice knob before invoking the
/// picker, so the chosen qindex actually corresponds to the encoded
/// stream's slice qindex bytes.
#[test]
fn hq_picture_qindex_wrapper_ignores_slice_size_target() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let (y, u, v) = test_triple_64();
    let mut with_slice_target = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    with_slice_target.slices_x = 4;
    with_slice_target.slices_y = 4;
    with_slice_target = with_slice_target.with_slice_size_target(10);

    let mut without = with_slice_target.clone();
    without.slice_size_target = None;

    // Pick the same target_bytes from a generous budget on both.
    let target = (hq_picture_payload_bytes_at_qindex(&seq, &without, &y, &u, &v, 0) as u32) * 2;
    let (stream_a, q_a, _) = encode_single_hq_intra_stream_with_size_target(
        &seq,
        &with_slice_target,
        target,
        0,
        &y,
        &u,
        &v,
    );
    let (stream_b, q_b, _) =
        encode_single_hq_intra_stream_with_size_target(&seq, &without, target, 0, &y, &u, &v);

    assert_eq!(
        q_a, q_b,
        "wrapper must clear slice_size_target before picking"
    );
    assert_eq!(
        stream_a, stream_b,
        "wrapper-emitted streams must be byte-identical regardless of caller's slice_size_target setting",
    );
}
