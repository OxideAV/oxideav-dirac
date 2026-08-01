//! Bipred (2-ref) residue rate-control qindex picker validator —
//! `pick_bipred_residue_qindex` / `bipred_residue_qindex_diagnostic`,
//! the 2-ref analogues of the proven 1-ref picker. Pins:
//!
//!   1. a generous budget keeps the floor qindex and reports a
//!      non-empty residue;
//!   2. the picker is monotone in the budget;
//!   3. a satisfiable budget's chosen qindex actually fits it;
//!   4. the residue floor (`ResidueParams::qindex`) is respected;
//!   5. `residue = None` degenerates to `(0, 0)`.

use oxideav_dirac::encoder::make_minimal_sequence;
use oxideav_dirac::encoder_inter::{
    bipred_residue_qindex_diagnostic, pick_bipred_residue_qindex, InterEncoderParams,
};
use oxideav_dirac::video_format::ChromaFormat;

struct Frame64 {
    y: [u8; 64 * 64],
    u: [u8; 32 * 32],
    v: [u8; 32 * 32],
}

/// A 16x16 textured square at `(x, 24)` — textured so the blended
/// prediction leaves a real residue to rate-control.
fn square_frame(x: usize) -> Frame64 {
    let mut fy = [40u8; 64 * 64];
    let mut fu = [110u8; 32 * 32];
    let mut fv = [140u8; 32 * 32];
    for r in 0..16 {
        for c in 0..16 {
            fy[(24 + r) * 64 + (x + c)] = 160 + ((r * 7 + c * 5) % 64) as u8;
        }
    }
    for r in 0..8 {
        for c in 0..8 {
            fu[(12 + r) * 32 + (x / 2 + c)] = 90;
            fv[(12 + r) * 32 + (x / 2 + c)] = 160;
        }
    }
    Frame64 {
        y: fy,
        u: fu,
        v: fv,
    }
}

/// (ref1, cur, ref2): the square marches 16 → 20 → 24, so the middle
/// frame predicts best from a blend of both neighbours.
fn fixture() -> (Frame64, Frame64, Frame64) {
    (square_frame(16), square_frame(20), square_frame(24))
}

#[test]
fn generous_budget_keeps_floor_and_reports_nonempty_residue() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let params = InterEncoderParams::default();
    let (r1, cur, r2) = fixture();

    let (q, bytes) = bipred_residue_qindex_diagnostic(
        &seq,
        &params,
        &cur.y,
        &cur.u,
        &cur.v,
        &r1.y,
        &r1.u,
        &r1.v,
        &r2.y,
        &r2.u,
        &r2.v,
        u32::MAX,
    );
    assert_eq!(q, 0, "generous budget keeps the floor qindex");
    assert!(bytes > 0, "textured bipred residue must be non-empty");
}

#[test]
fn picker_is_monotone_in_budget() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let params = InterEncoderParams::default();
    let (r1, cur, r2) = fixture();

    let (_q0, full) = bipred_residue_qindex_diagnostic(
        &seq,
        &params,
        &cur.y,
        &cur.u,
        &cur.v,
        &r1.y,
        &r1.u,
        &r1.v,
        &r2.y,
        &r2.u,
        &r2.v,
        u32::MAX,
    );
    let mut prev_q = 0u32;
    for div in [1u32, 2, 4, 8, 16, 64] {
        let target = ((full as u32) / div).max(1);
        let q = pick_bipred_residue_qindex(
            &seq, &params, &cur.y, &cur.u, &cur.v, &r1.y, &r1.u, &r1.v, &r2.y, &r2.u, &r2.v, target,
        );
        assert!(
            q >= prev_q,
            "qindex must not fall as the budget tightens: {q} < {prev_q} at /{div}"
        );
        prev_q = q;
    }
    assert!(prev_q > 0, "a /64 budget must escalate the qindex");
}

#[test]
fn satisfiable_budget_actually_fits() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let params = InterEncoderParams::default();
    let (r1, cur, r2) = fixture();

    let (_q, full) = bipred_residue_qindex_diagnostic(
        &seq,
        &params,
        &cur.y,
        &cur.u,
        &cur.v,
        &r1.y,
        &r1.u,
        &r1.v,
        &r2.y,
        &r2.u,
        &r2.v,
        u32::MAX,
    );
    let target = ((full as u32) / 3).max(1);
    let (q, bytes) = bipred_residue_qindex_diagnostic(
        &seq, &params, &cur.y, &cur.u, &cur.v, &r1.y, &r1.u, &r1.v, &r2.y, &r2.u, &r2.v, target,
    );
    if q < 127 {
        assert!(
            bytes <= target as usize,
            "chosen qindex {q} reports {bytes} bytes over the {target}-byte budget"
        );
    }
}

#[test]
fn residue_floor_is_respected() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let mut params = InterEncoderParams::default();
    if let Some(rp) = params.residue.as_mut() {
        rp.qindex = 20;
    }
    let (r1, cur, r2) = fixture();

    let q = pick_bipred_residue_qindex(
        &seq,
        &params,
        &cur.y,
        &cur.u,
        &cur.v,
        &r1.y,
        &r1.u,
        &r1.v,
        &r2.y,
        &r2.u,
        &r2.v,
        u32::MAX,
    );
    assert!(q >= 20, "picker must not go below the residue floor: q={q}");
}

#[test]
fn disabled_residue_degenerates_to_zero() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let params = InterEncoderParams {
        residue: None,
        ..Default::default()
    };
    let (r1, cur, r2) = fixture();

    let (q, bytes) = bipred_residue_qindex_diagnostic(
        &seq, &params, &cur.y, &cur.u, &cur.v, &r1.y, &r1.u, &r1.v, &r2.y, &r2.u, &r2.v, 100,
    );
    assert_eq!((q, bytes), (0, 0));
}
