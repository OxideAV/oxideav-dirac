//! Inter-residue rate-control qindex picker validator.
//!
//! Exercises `oxideav_dirac::encoder_inter::pick_inter_residue_qindex`
//! and its `inter_residue_qindex_diagnostic` companion — the inter
//! §11.3-residue analogue of the HQ/LD intra picture-qindex pickers.
//! Given a target residue-payload byte budget, the picker walks
//! `qindex ∈ floor..=127` and returns the smallest quantiser whose
//! serialised residue stream fits, mirroring how `pick_hq_picture_qindex`
//! sizes an intra picture.
//!
//! The tests pin: (1) the diagnostic's actual residue bytes never exceed
//! the budget when a fit exists; (2) monotonicity (a smaller budget can
//! only raise the chosen qindex); (3) the floor / 127 degeneracies; and
//! (4) that every chosen qindex yields a stream the production decoder
//! accepts to exactly two frames.

use oxideav_core::CodecRegistry;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_dirac::encoder::{make_minimal_sequence, EncoderParams};
use oxideav_dirac::encoder_inter::{
    encode_intra_then_inter_stream, inter_residue_qindex_diagnostic, pick_inter_residue_qindex,
    synthetic_camera_pan_64, InterEncoderParams, InterInputPicture, ResidueParams,
};
use oxideav_dirac::sequence::SequenceHeader;
use oxideav_dirac::video_format::ChromaFormat;
use oxideav_dirac::wavelet::WaveletFilter;

/// `(sequence, frame-0 Y/U/V, frame-1 Y/U/V)` for the 64x64 4:2:0
/// camera-pan fixture.
type Fixture = (
    SequenceHeader,
    [u8; 64 * 64],
    [u8; 32 * 32],
    [u8; 32 * 32],
    [u8; 64 * 64],
    [u8; 32 * 32],
    [u8; 32 * 32],
);

/// Camera-pan fixture: the OBMC prediction leaves a real (non-zero)
/// residual, so the qindex walk actually changes the residue byte count.
fn fixture() -> Fixture {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let (y0, u0, v0, y1, u1, v1) = synthetic_camera_pan_64(1, 0);
    (seq, y0, u0, v0, y1, u1, v1)
}

/// Encode the 2-picture intra+inter stream with `residue.qindex` set to
/// `qindex`, decode it, and return the decoded inter Y plane length (a
/// proxy for "the stream decoded cleanly to two frames").
fn encode_decode_at(
    seq: &SequenceHeader,
    base: &InterEncoderParams,
    qindex: u32,
    fix: &Fixture,
) -> usize {
    let mut params = base.clone();
    let mut rp = params.residue.clone().expect("residue configured");
    rp.qindex = qindex;
    params.residue = Some(rp);

    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let (_, y0, u0, v0, y1, u1, v1) = fix;
    let intra = InterInputPicture {
        picture_number: 10,
        y: y0,
        u: u0,
        v: v0,
    };
    let inter = InterInputPicture {
        picture_number: 11,
        y: y1,
        u: u1,
        v: v1,
    };
    let stream = encode_intra_then_inter_stream(seq, &intra_params, &params, &intra, &inter);

    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("make decoder");
    let packet = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&packet).expect("send_packet");

    let _frame0 = dec.receive_frame().expect("intra frame");
    let frame1 = match dec.receive_frame().expect("inter frame") {
        Frame::Video(vf) => vf,
        other => panic!("expected video frame, got {other:?}"),
    };
    frame1.planes[0].data.len()
}

/// The chosen qindex's actual residue payload must fit the budget for
/// every achievable target, and the resulting stream must decode.
#[test]
fn picked_qindex_fits_budget_and_decodes() {
    let fix = fixture();
    let (seq, y0, u0, v0, y1, u1, v1) = &fix;
    let params = InterEncoderParams::default(); // residue on by default.

    // Establish the floor (qindex=0) cost so the budgets straddle it.
    let (_q_floor, floor_bytes) =
        inter_residue_qindex_diagnostic(seq, &params, y1, u1, v1, y0, u0, v0, u32::MAX);
    assert!(floor_bytes > 0, "camera-pan residue must be non-empty");

    // Three budgets below the floor cost force escalating qindexes.
    for target in [
        (floor_bytes * 3 / 4) as u32,
        (floor_bytes / 2) as u32,
        (floor_bytes / 4).max(1) as u32,
    ] {
        let (q, actual) =
            inter_residue_qindex_diagnostic(seq, &params, y1, u1, v1, y0, u0, v0, target);
        // If a fit exists at any qindex ≤ 127 the actual must be ≤ target;
        // a budget that not even q=127 can hit returns 127 (actual may
        // exceed target, which the picker documents as "best effort").
        if q < 127 {
            assert!(
                actual <= target as usize,
                "q={q} actual={actual} must fit target={target}"
            );
        }
        // Whatever qindex was chosen, the stream must decode to 2 frames.
        let ylen = encode_decode_at(seq, &params, q, &fix);
        assert_eq!(ylen, 64 * 64, "decoded inter Y plane size");
    }
}

/// Monotone in the budget: a smaller residue budget can only raise the
/// chosen qindex (or leave it).
#[test]
fn picker_is_monotone_in_budget() {
    let fix = fixture();
    let (seq, y0, u0, v0, y1, u1, v1) = &fix;
    let params = InterEncoderParams::default();

    let (_q, floor_bytes) =
        inter_residue_qindex_diagnostic(seq, &params, y1, u1, v1, y0, u0, v0, u32::MAX);

    let mut prev_q = 0u32;
    // Sweep budgets from generous down to tight.
    for denom in [1u32, 2, 3, 4, 6, 8, 16, 64] {
        let target = (floor_bytes as u32 / denom).max(1);
        let q = pick_inter_residue_qindex(seq, &params, y1, u1, v1, y0, u0, v0, target);
        assert!(
            q >= prev_q,
            "qindex must not drop as budget tightens: prev={prev_q} q={q} (denom={denom})"
        );
        prev_q = q;
    }
}

/// A budget at or above the floor cost keeps the floor qindex; a budget
/// of 1 byte forces qindex 127 (the most aggressive quantiser).
#[test]
fn floor_and_ceiling_degeneracies() {
    let fix = fixture();
    let (seq, y0, u0, v0, y1, u1, v1) = &fix;
    let params = InterEncoderParams::default(); // residue floor qindex = 0.

    // A generous budget keeps qindex at the floor (0).
    let q_generous = pick_inter_residue_qindex(seq, &params, y1, u1, v1, y0, u0, v0, u32::MAX);
    assert_eq!(q_generous, 0, "generous budget keeps the floor qindex");

    // A 1-byte budget is unsatisfiable; the picker returns 127.
    let q_tight = pick_inter_residue_qindex(seq, &params, y1, u1, v1, y0, u0, v0, 1);
    assert_eq!(q_tight, 127, "unsatisfiable budget returns the max qindex");

    // Both extremes still decode.
    assert_eq!(encode_decode_at(seq, &params, q_generous, &fix), 64 * 64);
    assert_eq!(encode_decode_at(seq, &params, q_tight, &fix), 64 * 64);
}

/// A non-zero residue floor (`ResidueParams.qindex`) is honoured: the
/// picker never returns a qindex below the configured floor even when a
/// generous budget would otherwise allow qindex 0.
#[test]
fn floor_qindex_is_respected() {
    let fix = fixture();
    let (seq, y0, u0, v0, y1, u1, v1) = &fix;
    let mut rp = ResidueParams::default_for(WaveletFilter::LeGall5_3, 3);
    rp.qindex = 20; // floor.
    let params = InterEncoderParams {
        residue: Some(rp),
        ..InterEncoderParams::default()
    };

    let q = pick_inter_residue_qindex(seq, &params, y1, u1, v1, y0, u0, v0, u32::MAX);
    assert!(q >= 20, "picker must not go below the residue floor: q={q}");
    assert_eq!(encode_decode_at(seq, &params, q, &fix), 64 * 64);
}
