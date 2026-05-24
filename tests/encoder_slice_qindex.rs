//! §13.5.4 per-slice adaptive-qindex search (HQ profile).
//!
//! The HQ encoder can now pick each slice's `qindex` independently
//! (`EncoderParams::slice_size_target = Some(target)`) so that every
//! component's coefficient payload fits within `target` length-byte
//! units. A flat / low-energy slice keeps the floor qindex
//! (lossless-ish); a busy slice raises its own qindex just enough to
//! fit, instead of relying on a generous `slice_size_scaler` and
//! silently truncating the slice on the wire. The HQ profile applies no
//! §13.5.1 DC prediction, so slices are independent and may carry
//! different qindexes without any cross-slice coupling — the decoder
//! already reads a per-slice qindex (§13.5.2) so no decode-side change
//! is needed.
//!
//! Source of truth: BBC Dirac Specification v2.2.3 §13.5.2 (`slice()`
//! reads `qindex = read_nbits(7)` per slice) + §13.5.4
//! (`slice_quantisers(qindex)`).

use oxideav_core::{CodecId, CodecParameters, CodecRegistry, Frame, Packet, TimeBase};
use oxideav_dirac::encoder::{
    encode_single_hq_intra_stream, hq_slice_qindexes, make_minimal_sequence, EncoderParams,
};
use oxideav_dirac::video_format::ChromaFormat;
use oxideav_dirac::wavelet::WaveletFilter;

/// A high-spatial-energy luma/chroma plane whose energy varies sharply
/// from region to region: the top-left quadrant is a flat mid-grey
/// (cheap to code), while the rest carries a high-frequency checker that
/// produces large high-pass coefficients (expensive). This guarantees
/// the per-slice budget bites in some slices but not others.
fn varied_energy_plane(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut p = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            // Flat low-energy region in the top-left quadrant.
            let flat = x < w / 2 && y < h / 2;
            let v = if flat {
                128u32
            } else {
                // High-frequency 1-pel checker around mid-grey with a
                // strong amplitude → big HL/LH/HH coefficients.
                let checker = if (x + y + seed as usize) % 2 == 0 {
                    200
                } else {
                    40
                };
                checker as u32
            };
            p[y * w + x] = v as u8;
        }
    }
    p
}

/// A perfectly flat plane — every slice's coefficients are tiny, so the
/// adaptive search keeps the floor qindex for all of them.
fn flat_plane(w: usize, h: usize, value: u8) -> Vec<u8> {
    vec![value; w * h]
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

/// Build the standard 64x64 4:2:0 test triple from `varied_energy_plane`.
fn varied_triple() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (64usize, 64usize);
    let y = varied_energy_plane(w, h, 0);
    let u = varied_energy_plane(w / 2, h / 2, 1);
    let v = varied_energy_plane(w / 2, h / 2, 3);
    (y, u, v)
}

/// With a tight per-slice byte budget the adaptive search must:
///   1. produce a stream the decoder accepts end-to-end, and
///   2. drive every slice's reported max-component length down to the
///      budget (achievable here because qindex 127 zeroes everything).
#[test]
fn hq_adaptive_qindex_roundtrips_under_tight_budget() {
    let (w, h) = (64u32, 64u32);
    let (y, u, v) = varied_triple();
    let seq = make_minimal_sequence(w, h, ChromaFormat::Yuv420);

    // 10 length-byte units per component per slice. The flat top-left
    // quadrant's slices fit at the floor qindex (their q=0 length is 8),
    // while the busy checker quadrants overflow and escalate.
    let target = 10u32;
    let params =
        EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3).with_slice_size_target(target);

    // Every slice must fit the budget at its chosen qindex.
    let per_slice = hq_slice_qindexes(&seq, &params, &y, &u, &v);
    assert_eq!(
        per_slice.len(),
        (params.slices_x * params.slices_y) as usize
    );
    for (i, (q, max_len)) in per_slice.iter().enumerate() {
        assert!(
            *max_len <= target as usize,
            "slice {i}: max component length {max_len} > target {target} at qindex {q}",
        );
        assert!(*q <= 127, "slice {i}: qindex {q} out of range");
    }

    // The stream decodes; reconstruction is finite (the budget forces
    // lossy coding, so we only require a sane PSNR, not bit-exactness).
    let stream = encode_single_hq_intra_stream(&seq, &params, 0, &y, &u, &v);
    let frame = decode_one(stream);
    let p_y = psnr(&frame.planes[0].data, &y);
    assert!(
        p_y.is_finite() && p_y > 10.0,
        "Y PSNR {p_y:.2} dB too low under tight budget",
    );
}

/// Non-vacuity: on content with a flat quadrant and a busy remainder,
/// the adaptive search picks the floor qindex for at least one (cheap)
/// slice AND a strictly-higher qindex for at least one (busy) slice.
/// A bug that ignored the per-slice search would produce a constant
/// qindex vector and fail this.
#[test]
fn hq_adaptive_qindex_is_non_vacuous_on_busy_content() {
    let (w, h) = (64u32, 64u32);
    let (y, u, v) = varied_triple();
    let seq = make_minimal_sequence(w, h, ChromaFormat::Yuv420);

    let params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3).with_slice_size_target(10);
    let per_slice = hq_slice_qindexes(&seq, &params, &y, &u, &v);

    let floor = params.qindex;
    let at_floor = per_slice.iter().filter(|(q, _)| *q == floor).count();
    let above_floor = per_slice.iter().filter(|(q, _)| *q > floor).count();
    assert!(
        at_floor > 0,
        "expected at least one cheap slice to stay at the floor qindex {floor}; qs = {:?}",
        per_slice.iter().map(|(q, _)| *q).collect::<Vec<_>>(),
    );
    assert!(
        above_floor > 0,
        "expected at least one busy slice to raise its qindex above {floor}; qs = {:?}",
        per_slice.iter().map(|(q, _)| *q).collect::<Vec<_>>(),
    );
}

/// On perfectly flat content every slice fits the budget at the floor
/// qindex (0), so the adaptive search leaves the qindex vector all-zero
/// and the emitted stream is byte-identical to the constant-qindex
/// (`slice_size_target = None`) stream — and round-trips bit-exact.
#[test]
fn hq_adaptive_qindex_keeps_floor_and_matches_constant_on_flat() {
    let (w, h) = (64u32, 64u32);
    let y = flat_plane(w as usize, h as usize, 100);
    let u = flat_plane((w / 2) as usize, (h / 2) as usize, 110);
    let v = flat_plane((w / 2) as usize, (h / 2) as usize, 120);
    let seq = make_minimal_sequence(w, h, ChromaFormat::Yuv420);

    // Flat content's per-slice q=0 length is 10 bytes (a single nonzero
    // LL DC coefficient plus many cheap zero high-pass coefficients), so
    // a budget of 12 fits every slice at the floor.
    let constant = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let adaptive = constant.clone().with_slice_size_target(12);

    // All slices stay at the floor qindex (0) on flat content.
    let per_slice = hq_slice_qindexes(&seq, &adaptive, &y, &u, &v);
    assert!(
        per_slice.iter().all(|(q, _)| *q == 0),
        "flat content should keep floor qindex 0 everywhere; qs = {:?}",
        per_slice.iter().map(|(q, _)| *q).collect::<Vec<_>>(),
    );

    // Byte-identical to the constant-qindex stream.
    let s_const = encode_single_hq_intra_stream(&seq, &constant, 0, &y, &u, &v);
    let s_adapt = encode_single_hq_intra_stream(&seq, &adaptive, 0, &y, &u, &v);
    assert_eq!(
        s_const, s_adapt,
        "flat-content adaptive stream must equal the constant-qindex stream",
    );

    // And both round-trip bit-exact at qindex 0.
    let frame = decode_one(s_adapt);
    assert_eq!(frame.planes[0].data, y, "Y not bit-exact on flat content");
    assert_eq!(frame.planes[1].data, u, "U not bit-exact on flat content");
    assert_eq!(frame.planes[2].data, v, "V not bit-exact on flat content");
}

/// A generous budget that every slice fits at the floor reduces the
/// adaptive path to the constant-qindex path: the stream is byte-equal
/// to `slice_size_target = None`. This pins that the only effect of the
/// search is qindex *escalation* when a slice overflows — never a
/// regression on content that already fits.
#[test]
fn hq_adaptive_qindex_generous_budget_equals_none() {
    let (w, h) = (64u32, 64u32);
    let (y, u, v) = varied_triple();
    let seq = make_minimal_sequence(w, h, ChromaFormat::Yuv420);

    // Use a large scaler so even busy slices fit at qindex 0 within a
    // single length byte, then set the target above every slice length.
    let mut none = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    none.slices_x = 4;
    none.slices_y = 4;
    none.slice_size_scaler = 8;

    // Find the worst-case length at the floor so we can pick a target
    // strictly above it.
    let floor_lengths = hq_slice_qindexes(&seq, &none, &y, &u, &v);
    let worst = floor_lengths.iter().map(|(_, l)| *l).max().unwrap();

    let adaptive = none.clone().with_slice_size_target((worst + 4) as u32);
    // Every slice stays at the floor (everything fits the generous budget).
    let per_slice = hq_slice_qindexes(&seq, &adaptive, &y, &u, &v);
    assert!(
        per_slice.iter().all(|(q, _)| *q == none.qindex),
        "generous budget should keep the floor qindex everywhere",
    );

    let s_none = encode_single_hq_intra_stream(&seq, &none, 0, &y, &u, &v);
    let s_adapt = encode_single_hq_intra_stream(&seq, &adaptive, 0, &y, &u, &v);
    assert_eq!(
        s_none, s_adapt,
        "generous-budget adaptive stream must equal the constant-qindex stream",
    );
}

/// Determinism: the same inputs and params always produce a
/// byte-identical stream (the search has no hidden state / ordering
/// nondeterminism).
#[test]
fn hq_adaptive_qindex_is_deterministic() {
    let (w, h) = (64u32, 64u32);
    let (y, u, v) = varied_triple();
    let seq = make_minimal_sequence(w, h, ChromaFormat::Yuv420);
    let params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3).with_slice_size_target(10);

    let s1 = encode_single_hq_intra_stream(&seq, &params, 0, &y, &u, &v);
    let s2 = encode_single_hq_intra_stream(&seq, &params, 0, &y, &u, &v);
    assert_eq!(s1, s2, "adaptive-qindex encode must be deterministic");

    let q1 = hq_slice_qindexes(&seq, &params, &y, &u, &v);
    let q2 = hq_slice_qindexes(&seq, &params, &y, &u, &v);
    assert_eq!(q1, q2, "per-slice qindex vector must be deterministic");
}

/// A monotonicity property of the search itself: a *smaller* target can
/// only push qindexes *up* (or leave them), never down — the search
/// floor is the same and a tighter budget rejects more low-qindex
/// candidates. Pins that the search direction is correct.
#[test]
fn hq_adaptive_qindex_tighter_budget_never_lowers_qindex() {
    let (w, h) = (64u32, 64u32);
    let (y, u, v) = varied_triple();
    let seq = make_minimal_sequence(w, h, ChromaFormat::Yuv420);

    let loose = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3).with_slice_size_target(20);
    let tight = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3).with_slice_size_target(10);

    let q_loose = hq_slice_qindexes(&seq, &loose, &y, &u, &v);
    let q_tight = hq_slice_qindexes(&seq, &tight, &y, &u, &v);
    assert_eq!(q_loose.len(), q_tight.len());
    for (i, ((ql, _), (qt, _))) in q_loose.iter().zip(q_tight.iter()).enumerate() {
        assert!(
            qt >= ql,
            "slice {i}: tighter budget lowered qindex ({qt} < {ql})",
        );
    }
}
