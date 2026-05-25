//! VC-2 LD picture-level rate-control: target-byte picker.
//!
//! r131 adds the LD analogue of HQ's `with_slice_size_target`. LD has a
//! fixed per-slice byte budget (§13.5.3.2) and a single picture-level
//! quantiser, so the picker:
//!   1. derives `slice_bytes_numer / denom` from `target_picture_bytes`
//!      so the encoded picture payload lands within ±1 byte of the
//!      target, and
//!   2. picks the **smallest** qindex (best quality) for which every
//!      slice's coefficient payload fits its `payload_bits` budget
//!      without Funnel-truncation.
//!
//! Source of truth: SMPTE ST 2042-1 §13.5.3.2 (per-slice byte budget),
//! §13.5.2 (per-slice qindex header), §13.5.4 (quantisation matrix
//! indexing by qindex).
//!
//! Wall: spec PDF in `docs/video/dirac/dirac-spec-latest.pdf` only.

use oxideav_core::{CodecId, CodecParameters, CodecRegistry, Frame, Packet, TimeBase};
use oxideav_dirac::encoder::{
    encode_single_ld_intra_stream, encode_single_ld_intra_stream_with_size_target,
    ld_picture_payload_bytes, ld_picture_qindex_diagnostic, make_minimal_sequence_ld,
    pick_ld_picture_qindex, LdEncoderParams,
};
use oxideav_dirac::video_format::ChromaFormat;
use oxideav_dirac::wavelet::WaveletFilter;

/// 64x64 4:2:0 test pattern: a mid-frequency diagonal gradient on Y plus
/// flat chroma. Smooth enough that q=0 fits a generous budget bit-exact;
/// rich enough that a tight budget needs a non-zero qindex.
fn test_triple_64() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; 64 * 64];
    let mut u = vec![128u8; 32 * 32];
    let mut v = vec![128u8; 32 * 32];
    for row in 0..64 {
        for col in 0..64 {
            y[row * 64 + col] = (((row + col) as u32 * 4) & 0xFF) as u8;
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

/// `ld_picture_payload_bytes` matches the actual length of
/// `encode_ld_intra_picture` for any input — it's a pure property of
/// the params, independent of input samples or qindex. Pins the
/// invariant the picker relies on for `derive_ld_slice_bytes_for_target`.
#[test]
fn ld_picture_payload_bytes_matches_actual() {
    use oxideav_dirac::encoder::encode_ld_intra_picture;
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let (y, u, v) = test_triple_64();
    for &bps in &[16u32, 32, 64, 128, 256] {
        for qindex in [0u32, 5, 16, 32, 64, 100, 127] {
            let mut params = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, bps);
            params.qindex = qindex;
            let predicted = ld_picture_payload_bytes(&params);
            let actual = encode_ld_intra_picture(&seq, &params, 0, &y, &u, &v).len();
            assert_eq!(
                actual, predicted,
                "bps={bps} qindex={qindex}: predicted {predicted} != actual {actual}",
            );
        }
    }
}

/// THE round-131 acceptance test. Three target picture-byte budgets
/// (small / medium / large) must each:
///   (a) decode end-to-end,
///   (b) land within ±10% of the target on actual picture bytes, and
///   (c) the small target must pick a strictly higher qindex than the
///       large target (lower target → tighter budget → must quantise
///       more aggressively to fit).
#[test]
fn ld_picture_qindex_picker_hits_three_budgets_within_10pct() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let (y, u, v) = test_triple_64();

    // 4x4 = 16 slices. Minimum picture bytes is roughly header (~15 B)
    // + 2 * 16 slices = ~50 B; the three targets sit well above that.
    let base = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 32);

    let targets = [200u32, 1024, 4096]; // small / medium / large
    let mut chosen_qindexes: Vec<u32> = Vec::new();
    let mut actuals: Vec<usize> = Vec::new();

    for &target in &targets {
        let (stream, qindex, adjusted) =
            encode_single_ld_intra_stream_with_size_target(&seq, &base, target, 0, &y, &u, &v)
                .unwrap_or_else(|| panic!("size target {target} did not yield a viable params"));

        // Actual picture payload bytes (the slice we sized to `target`).
        // The full stream additionally carries seq header + 3 parse infos.
        let actual_picture = ld_picture_payload_bytes(&adjusted);
        actuals.push(actual_picture);

        // ±10% tolerance on the picture payload bytes.
        let lo = (target as f64 * 0.9).floor() as i64;
        let hi = (target as f64 * 1.1).ceil() as i64;
        assert!(
            (actual_picture as i64) >= lo && (actual_picture as i64) <= hi,
            "target {target}: actual_picture_bytes {actual_picture} outside ±10% ({lo}..={hi}) — qindex={qindex} numer={} denom={}",
            adjusted.slice_bytes_numer, adjusted.slice_bytes_denom,
        );
        assert!(qindex <= 127);
        chosen_qindexes.push(qindex);

        // The stream itself must decode (a tight budget may make the
        // reconstruction lossy, but never invalid).
        let frame = decode_one(stream);
        let p_y = psnr(&frame.planes[0].data, &y);
        assert!(
            p_y.is_finite() && p_y > 6.0,
            "target {target}: Y PSNR {p_y:.2} dB — decode produced garbage",
        );
    }

    eprintln!(
        "LD picture-qindex picker: targets={:?} actuals={:?} qindexes={:?}",
        targets, actuals, chosen_qindexes,
    );

    // Small target ⇒ strictly higher qindex than large target.
    assert!(
        chosen_qindexes[0] > chosen_qindexes[2],
        "small target qindex {} should exceed large target qindex {} (tighter budget = aggressive quantiser); all = {:?}",
        chosen_qindexes[0], chosen_qindexes[2], chosen_qindexes,
    );
}

/// The picker is monotone: a smaller picture-byte target can only push
/// the chosen qindex up (or leave it) — never down. Pins the search
/// direction (lower budget = lower payload_bits = some qindexes that
/// previously fit no longer do, so the picker must escalate).
#[test]
fn ld_picture_qindex_picker_is_monotone_in_target() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let (y, u, v) = test_triple_64();
    let base = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 32);

    let big = 4096u32;
    let small = 256u32;

    let (_, q_big, _) =
        encode_single_ld_intra_stream_with_size_target(&seq, &base, big, 0, &y, &u, &v).unwrap();
    let (_, q_small, _) =
        encode_single_ld_intra_stream_with_size_target(&seq, &base, small, 0, &y, &u, &v).unwrap();

    assert!(
        q_small >= q_big,
        "smaller budget {small} (q={q_small}) should pick qindex ≥ larger budget {big} (q={q_big})",
    );
}

/// On flat content even a tight target fits at q=0 (the LL DC plus a
/// few cheap zero high-pass coefficients consume just a few bytes per
/// slice). The picker should return q=0 in that case — even a tight
/// budget with no real content is over-budget.
#[test]
fn ld_picture_qindex_picker_keeps_zero_on_flat_content() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    // Perfectly flat picture: every slice's quantised coefficients are
    // tiny / mostly zero, so q=0 fits even a tight budget.
    let y = vec![128u8; 64 * 64];
    let u = vec![128u8; 32 * 32];
    let v = vec![128u8; 32 * 32];

    let base = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 32);
    let target = 1024u32;
    let (stream, qindex, adjusted) =
        encode_single_ld_intra_stream_with_size_target(&seq, &base, target, 0, &y, &u, &v)
            .expect("flat content should fit any reasonable target at q=0");
    assert_eq!(qindex, 0, "flat content must keep floor qindex 0");

    // ±10% on picture bytes.
    let actual = ld_picture_payload_bytes(&adjusted);
    let lo = (target as f64 * 0.9).floor() as i64;
    let hi = (target as f64 * 1.1).ceil() as i64;
    assert!((actual as i64) >= lo && (actual as i64) <= hi);

    // Stream decodes; flat content should reconstruct ~bit-exact at q=0.
    let frame = decode_one(stream);
    let p_y = psnr(&frame.planes[0].data, &y);
    assert!(
        p_y > 40.0,
        "flat content should decode near-lossless at q=0 (Y PSNR {p_y:.2} dB)",
    );
}

/// Determinism: same inputs always produce the same `(stream, qindex,
/// adjusted_params)` triple. The picker has no hidden ordering / state.
#[test]
fn ld_picture_qindex_picker_is_deterministic() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let (y, u, v) = test_triple_64();
    let base = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 32);
    let target = 1024u32;

    let (s1, q1, _) =
        encode_single_ld_intra_stream_with_size_target(&seq, &base, target, 0, &y, &u, &v).unwrap();
    let (s2, q2, _) =
        encode_single_ld_intra_stream_with_size_target(&seq, &base, target, 0, &y, &u, &v).unwrap();

    assert_eq!(s1, s2, "picker must produce byte-identical streams");
    assert_eq!(q1, q2, "picker must produce identical qindex");
}

/// Very small target: too small to fit picture header + 2*N_slices
/// minimum. Picker must return `None` rather than panicking or producing
/// an invalid stream.
#[test]
fn ld_picture_qindex_picker_returns_none_on_unfit_target() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let (y, u, v) = test_triple_64();
    let base = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 32);

    // 4x4 = 16 slices. Each slice needs >= 2 bytes; header ~15 B; so
    // anything under ~50 B can't be encoded. Pick 5 — clearly too small.
    assert!(
        encode_single_ld_intra_stream_with_size_target(&seq, &base, 5, 0, &y, &u, &v).is_none(),
        "5-byte target should be rejected",
    );
}

/// Picker exposed to the diagnostic API returns the same qindex as the
/// stream encoder uses, with a non-negative overflow measure that's
/// zero when the qindex actually fits and positive only when even q=127
/// truncates. Pins that the diagnostic mirrors the encode-path picker.
#[test]
fn ld_picture_qindex_diagnostic_mirrors_picker() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let (y, u, v) = test_triple_64();
    let mut params = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 64);
    params.qindex = 0;

    let q1 = pick_ld_picture_qindex(&seq, &params, &y, &u, &v);
    let (q2, overflow) = ld_picture_qindex_diagnostic(&seq, &params, &y, &u, &v);
    assert_eq!(
        q1, q2,
        "diagnostic picker disagreed with the encode-path picker"
    );

    // If picker returned anything below 127 the chosen qindex must fit
    // — i.e. overflow == 0. Only q=127 is allowed to carry residual
    // overflow (the gracefully-truncated case).
    if q1 < 127 {
        assert_eq!(
            overflow, 0,
            "picker returned q={q1} < 127 but reported overflow {overflow} > 0",
        );
    }
}

/// Cross-check: a `target_picture_bytes` derived from a known
/// `default_ld(bps)` and encoded via the legacy
/// `encode_single_ld_intra_stream` should land at the same byte count.
/// Pins that the picker's slice-bytes derivation is consistent with the
/// straight `default_ld(bps)` constructor.
#[test]
fn ld_picture_qindex_picker_matches_direct_default_ld_at_same_target() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let (y, u, v) = test_triple_64();

    // 4*4 slices * 64 bytes/slice = 1024 slice bytes; picture payload
    // bytes is then 1024 + header (~15 B).
    let direct = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 64);
    let direct_stream = encode_single_ld_intra_stream(&seq, &direct, 0, &y, &u, &v);
    let direct_picture_bytes = ld_picture_payload_bytes(&direct);

    // Now ask the picker for the same picture-payload byte target.
    let base = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 32);
    let (picked_stream, _qindex, adjusted) = encode_single_ld_intra_stream_with_size_target(
        &seq,
        &base,
        direct_picture_bytes as u32,
        0,
        &y,
        &u,
        &v,
    )
    .expect("picker should accept the target");

    assert_eq!(
        ld_picture_payload_bytes(&adjusted),
        direct_picture_bytes,
        "picker did not converge on the same picture-byte count as default_ld",
    );
    // The streams may differ in qindex (and therefore reconstruction
    // quality) but never in slice_bytes_numer/denom for the same target.
    assert_eq!(adjusted.slice_bytes_numer, direct.slice_bytes_numer);
    assert_eq!(adjusted.slice_bytes_denom, direct.slice_bytes_denom);

    // Both streams must decode end-to-end.
    let _ = decode_one(direct_stream);
    let _ = decode_one(picked_stream);
}
