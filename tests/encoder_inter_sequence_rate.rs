//! Multi-picture rate-controlled inter sequence driver validator.
//!
//! Exercises
//! `oxideav_dirac::encoder_inter::encode_inter_sequence_with_residue_target`
//! and its `_report` companion — the sequence-level wiring that drives
//! the per-picture `pick_inter_residue_qindex` picker across an HQ intra
//! anchor (`0xEC`) followed by N 1-ref inter pictures (`0x09`), with a
//! `PerPicture` / `Cbr` / `Vbv` / `VbvHysteresis` residue-byte
//! accumulator. The single-picture picker is validated separately in
//! `encoder_inter_residue_rate.rs`; this file pins the multi-picture
//! driver:
//!
//!   1. the full stream decodes to one frame per input picture;
//!   2. every report entry's actual residue bytes fit its requested
//!      budget when a fit exists (q < 127);
//!   3. the CBR accumulator equals the running `Σ(actual − target)`;
//!   4. `PerPicture` requests are the bare target every picture, while
//!      `Cbr` requests track the carry;
//!   5. `Vbv` clamps the banked savings at `-buffer_bytes` (peak
//!      residue-size cap) and `VbvHysteresis` additionally rate-limits
//!      the per-picture drain (collapsing to plain `Vbv` when
//!      `max_drain_per_picture >= buffer_bytes`);
//!   6. anchor-only and `residue = None` degeneracies behave.

use oxideav_core::CodecRegistry;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_dirac::encoder::{make_minimal_sequence, EncoderParams};
use oxideav_dirac::encoder_inter::{
    encode_inter_sequence_with_residue_target, encode_inter_sequence_with_residue_target_report,
    inter_residue_qindex_diagnostic, synthetic_camera_pan_64, InterEncoderParams,
    InterInputPicture, InterRateControl, ResidueParams,
};
use oxideav_dirac::sequence::SequenceHeader;
use oxideav_dirac::video_format::ChromaFormat;
use oxideav_dirac::wavelet::WaveletFilter;

/// Owned 64x64 4:2:0 frame planes (so the borrowed `InterInputPicture`
/// slices outlive the encode call).
struct Frame64 {
    y: [u8; 64 * 64],
    u: [u8; 32 * 32],
    v: [u8; 32 * 32],
}

/// Build a sequence header plus four progressively-panned 64x64 frames:
/// a flat-ish anchor (pan 0) then three camera-pans at increasing offset
/// so each inter picture carries a real (and growing) residue.
fn fixture() -> (SequenceHeader, Vec<Frame64>) {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let mut frames = Vec::new();
    // Anchor: pan 0 (the un-shifted reference pattern).
    let (y0, u0, v0, _, _, _) = synthetic_camera_pan_64(0, 0);
    frames.push(Frame64 {
        y: y0,
        u: u0,
        v: v0,
    });
    // Three inter frames at growing sub-pel pan against frame 0.
    for dx in [1i32, 3, 5] {
        let (_, _, _, y1, u1, v1) = synthetic_camera_pan_64(dx, 0);
        frames.push(Frame64 {
            y: y1,
            u: u1,
            v: v1,
        });
    }
    (seq, frames)
}

fn input_pictures(frames: &[Frame64]) -> Vec<InterInputPicture<'_>> {
    frames
        .iter()
        .enumerate()
        .map(|(i, f)| InterInputPicture {
            picture_number: 10 + i as u32,
            y: &f.y,
            u: &f.u,
            v: &f.v,
        })
        .collect()
}

/// Decode a complete elementary stream and return the number of decoded
/// video frames (asserting each is a video frame of the expected size).
fn decode_frame_count(stream: Vec<u8>) -> usize {
    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("make decoder");
    let packet = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&packet).expect("send_packet");

    let mut count = 0;
    while let Ok(frame) = dec.receive_frame() {
        match frame {
            Frame::Video(vf) => {
                assert_eq!(vf.planes[0].data.len(), 64 * 64, "decoded Y plane size");
                count += 1;
            }
            other => panic!("expected video frame, got {other:?}"),
        }
    }
    count
}

/// PerPicture: every report entry requests exactly `target`, the full
/// stream decodes to one frame per picture, and each fitting qindex's
/// actual residue is within budget.
#[test]
fn per_picture_decodes_and_fits_budget() {
    let (seq, frames) = fixture();
    let pics = input_pictures(&frames);
    let intra = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter = InterEncoderParams::default(); // residue on.

    // Establish the q=0 residue cost of the largest-pan inter against
    // the anchor, so the target straddles it (forces some escalation).
    let (_q, floor_bytes) = inter_residue_qindex_diagnostic(
        &seq,
        &inter,
        &frames[3].y,
        &frames[3].u,
        &frames[3].v,
        &frames[0].y,
        &frames[0].u,
        &frames[0].v,
        u32::MAX,
    );
    assert!(floor_bytes > 0, "camera-pan residue must be non-empty");
    let target = (floor_bytes / 2).max(1) as u32;

    let (stream, report) = encode_inter_sequence_with_residue_target_report(
        &seq,
        &intra,
        &inter,
        &pics,
        target,
        InterRateControl::PerPicture,
    );

    // One report entry per inter picture (anchor excluded).
    assert_eq!(report.len(), pics.len() - 1);
    for r in &report {
        assert_eq!(r.requested_residue_bytes, target, "PerPicture request");
        // Every inter references the anchor (picture 10).
        assert_eq!(r.ref1_picture_number, 10);
        if r.qindex < 127 {
            assert!(
                r.actual_residue_bytes <= target,
                "fitting qindex must stay within budget: {r:?}"
            );
        }
    }

    // Whole stream decodes to one frame per input picture.
    assert_eq!(decode_frame_count(stream), pics.len());

    // The convenience wrapper returns byte-identical output.
    let plain = encode_inter_sequence_with_residue_target(
        &seq,
        &intra,
        &inter,
        &pics,
        target,
        InterRateControl::PerPicture,
    );
    let (stream2, _) = encode_inter_sequence_with_residue_target_report(
        &seq,
        &intra,
        &inter,
        &pics,
        target,
        InterRateControl::PerPicture,
    );
    assert_eq!(plain, stream2, "wrapper matches _report stream");
}

/// CBR: the running accumulator equals Σ(actual − target), and each
/// request after the first reflects the prior carry.
#[test]
fn cbr_accumulator_tracks_running_deviation() {
    let (seq, frames) = fixture();
    let pics = input_pictures(&frames);
    let intra = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter = InterEncoderParams::default();

    let (_q, floor_bytes) = inter_residue_qindex_diagnostic(
        &seq,
        &inter,
        &frames[3].y,
        &frames[3].u,
        &frames[3].v,
        &frames[0].y,
        &frames[0].u,
        &frames[0].v,
        u32::MAX,
    );
    let target = (floor_bytes / 2).max(1) as u32;

    let (stream, report) = encode_inter_sequence_with_residue_target_report(
        &seq,
        &intra,
        &inter,
        &pics,
        target,
        InterRateControl::Cbr,
    );

    // Reconstruct the accumulator from the actual bytes and verify it
    // matches the reported running surplus at every step.
    let mut carry: i64 = 0;
    for (i, r) in report.iter().enumerate() {
        // Request = target − carry-so-far (clamped at 0).
        let expected_req = (target as i64 - carry).max(0) as u32;
        assert_eq!(
            r.requested_residue_bytes, expected_req,
            "Cbr request #{i} should fold prior carry"
        );
        carry += r.actual_residue_bytes as i64 - target as i64;
        assert_eq!(
            r.running_surplus_bytes, carry,
            "running surplus #{i} = Σ(actual − target)"
        );
    }

    assert_eq!(decode_frame_count(stream), pics.len());
}

/// VBV: the savings end of the accumulator is clamped at `-buffer_bytes`,
/// so the next picture's request never exceeds `target + buffer_bytes`.
/// With `buffer_bytes = 0` the savings side is forfeited entirely, so the
/// running surplus is always `>= 0` (debt is still tracked) and every
/// request is `<= target`. The whole stream still decodes.
#[test]
fn vbv_clamps_savings_and_caps_request() {
    let (seq, frames) = fixture();
    let pics = input_pictures(&frames);
    let intra = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter = InterEncoderParams::default();

    let (_q, floor_bytes) = inter_residue_qindex_diagnostic(
        &seq,
        &inter,
        &frames[3].y,
        &frames[3].u,
        &frames[3].v,
        &frames[0].y,
        &frames[0].u,
        &frames[0].v,
        u32::MAX,
    );
    let target = (floor_bytes / 2).max(1) as u32;
    let buffer_bytes = (floor_bytes / 8).max(1) as u32;

    let (stream, report) = encode_inter_sequence_with_residue_target_report(
        &seq,
        &intra,
        &inter,
        &pics,
        target,
        InterRateControl::Vbv { buffer_bytes },
    );

    // Reconstruct the clamped accumulator and verify the reported surplus
    // and each request match, with the savings clamp applied.
    let mut carry: i64 = 0;
    for (i, r) in report.iter().enumerate() {
        let spendable = (-carry).min(buffer_bytes as i64).max(0);
        let want = target as i64 - carry.max(0) + spendable;
        let expected_req = want.clamp(0, u32::MAX as i64) as u32;
        assert_eq!(
            r.requested_residue_bytes, expected_req,
            "Vbv request #{i} folds carry with savings clamp"
        );
        // The request can never exceed target + buffer_bytes.
        assert!(
            r.requested_residue_bytes <= target + buffer_bytes,
            "Vbv request #{i} bounded by target+buffer"
        );
        carry += r.actual_residue_bytes as i64 - target as i64;
        if carry < -(buffer_bytes as i64) {
            carry = -(buffer_bytes as i64);
        }
        assert_eq!(
            r.running_surplus_bytes, carry,
            "Vbv running surplus #{i} clamped at -buffer_bytes"
        );
        assert!(
            r.running_surplus_bytes >= -(buffer_bytes as i64),
            "savings never exceed buffer_bytes"
        );
    }

    assert_eq!(decode_frame_count(stream), pics.len());

    // buffer_bytes == 0: savings forfeited, surplus stays >= 0, requests
    // never exceed target.
    let (stream0, report0) = encode_inter_sequence_with_residue_target_report(
        &seq,
        &intra,
        &inter,
        &pics,
        target,
        InterRateControl::Vbv { buffer_bytes: 0 },
    );
    for r in &report0 {
        assert!(
            r.running_surplus_bytes >= 0,
            "buffer_bytes=0 forfeits all savings: {r:?}"
        );
        assert!(
            r.requested_residue_bytes <= target,
            "buffer_bytes=0 request never exceeds target: {r:?}"
        );
    }
    assert_eq!(decode_frame_count(stream0), pics.len());
}

/// VbvHysteresis: the spent savings per picture are additionally clamped
/// at `max_drain_per_picture`, while the bucket fill / debt semantics
/// match plain Vbv. A small drain cap means the request's savings-spend
/// never exceeds `max_drain_per_picture`; a `max_drain >= buffer_bytes`
/// cap collapses to byte-identical plain Vbv output.
#[test]
fn vbv_hysteresis_limits_drain_and_collapses_to_vbv() {
    let (seq, frames) = fixture();
    let pics = input_pictures(&frames);
    let intra = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter = InterEncoderParams::default();

    let (_q, floor_bytes) = inter_residue_qindex_diagnostic(
        &seq,
        &inter,
        &frames[3].y,
        &frames[3].u,
        &frames[3].v,
        &frames[0].y,
        &frames[0].u,
        &frames[0].v,
        u32::MAX,
    );
    let target = (floor_bytes / 2).max(1) as u32;
    let buffer_bytes = (floor_bytes / 4).max(2) as u32;
    let max_drain = (buffer_bytes / 2).max(1);

    let (stream, report) = encode_inter_sequence_with_residue_target_report(
        &seq,
        &intra,
        &inter,
        &pics,
        target,
        InterRateControl::VbvHysteresis {
            buffer_bytes,
            max_drain_per_picture: max_drain,
        },
    );

    // Reconstruct with the drain-limited spend and verify each request.
    let mut carry: i64 = 0;
    for (i, r) in report.iter().enumerate() {
        let spendable = (-carry)
            .min(buffer_bytes as i64)
            .min(max_drain as i64)
            .max(0);
        let want = target as i64 - carry.max(0) + spendable;
        let expected_req = want.clamp(0, u32::MAX as i64) as u32;
        assert_eq!(
            r.requested_residue_bytes, expected_req,
            "VbvHysteresis request #{i} drain-limited"
        );
        // The savings spent above target are capped at max_drain.
        let savings_spent = r.requested_residue_bytes as i64 - target as i64;
        if savings_spent > 0 {
            assert!(
                savings_spent <= max_drain as i64,
                "drain per picture #{i} bounded by max_drain"
            );
        }
        carry += r.actual_residue_bytes as i64 - target as i64;
        if carry < -(buffer_bytes as i64) {
            carry = -(buffer_bytes as i64);
        }
        assert_eq!(r.running_surplus_bytes, carry);
    }
    assert_eq!(decode_frame_count(stream), pics.len());

    // max_drain_per_picture >= buffer_bytes collapses to plain Vbv: the
    // drain cap never bites, so the streams are byte-identical.
    let collapsed = encode_inter_sequence_with_residue_target(
        &seq,
        &intra,
        &inter,
        &pics,
        target,
        InterRateControl::VbvHysteresis {
            buffer_bytes,
            max_drain_per_picture: buffer_bytes,
        },
    );
    let plain_vbv = encode_inter_sequence_with_residue_target(
        &seq,
        &intra,
        &inter,
        &pics,
        target,
        InterRateControl::Vbv { buffer_bytes },
    );
    assert_eq!(
        collapsed, plain_vbv,
        "max_drain >= buffer_bytes ≡ plain Vbv"
    );
}

/// Anchor-only input → sequence header + anchor + EOS, no inter
/// pictures, empty report, decodes to one frame.
#[test]
fn anchor_only_is_a_valid_one_frame_stream() {
    let (seq, frames) = fixture();
    let pics = vec![InterInputPicture {
        picture_number: 10,
        y: &frames[0].y,
        u: &frames[0].u,
        v: &frames[0].v,
    }];
    let intra = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter = InterEncoderParams::default();

    let (stream, report) = encode_inter_sequence_with_residue_target_report(
        &seq,
        &intra,
        &inter,
        &pics,
        64,
        InterRateControl::Cbr,
    );
    assert!(report.is_empty(), "no inter pictures → empty report");
    assert_eq!(decode_frame_count(stream), 1);
}

/// `residue = None` → ZERO_RESIDUAL pictures: every report entry has
/// zero residue bytes and qindex 0, and the stream still decodes to one
/// frame per picture.
#[test]
fn residue_disabled_emits_zero_residual_chain() {
    let (seq, frames) = fixture();
    let pics = input_pictures(&frames);
    let intra = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter = InterEncoderParams {
        residue: None,
        ..InterEncoderParams::default()
    };

    let (stream, report) = encode_inter_sequence_with_residue_target_report(
        &seq,
        &intra,
        &inter,
        &pics,
        128,
        InterRateControl::Cbr,
    );
    assert_eq!(report.len(), pics.len() - 1);
    for r in &report {
        assert_eq!(r.actual_residue_bytes, 0, "no residue stream emitted");
        assert_eq!(r.qindex, 0, "qindex floor when residue disabled");
    }
    assert_eq!(decode_frame_count(stream), pics.len());
}

/// The multi-picture inter sequence driver carries the §11.3.3 codeblock
/// grid through unchanged: the per-picture `inter_params.residue` (grid +
/// mode) is cloned for every inter picture and only its `qindex` is
/// overridden by the rate-control pick. With a per-level
/// `[(1,1),(2,2),(2,2),(2,2)]` grid (every codeblock >= 4x4 samples), a
/// CBR-targeted multi-picture sequence must (a) emit one report entry per
/// inter picture whose fitting qindex stays within budget, (b) keep the
/// Σ(actual − target) accumulator consistent, and (c) decode to one frame
/// per input picture — proving the codeblock residue integrates with the
/// full sequence driver + codeblock-aware rate control end-to-end.
#[test]
fn sequence_driver_threads_codeblock_grid() {
    let (seq, frames) = fixture();
    let pics = input_pictures(&frames);
    let intra = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);

    let mut rp = ResidueParams::default_for(WaveletFilter::LeGall5_3, 3);
    rp.codeblocks = Some(vec![(1, 1), (2, 2), (2, 2), (2, 2)]);
    rp.codeblock_mode = 0;
    let inter = InterEncoderParams {
        residue: Some(rp),
        ..InterEncoderParams::default()
    };

    let (_q, floor_bytes) = inter_residue_qindex_diagnostic(
        &seq,
        &inter,
        &frames[3].y,
        &frames[3].u,
        &frames[3].v,
        &frames[0].y,
        &frames[0].u,
        &frames[0].v,
        u32::MAX,
    );
    assert!(floor_bytes > 0, "codeblock residue must be non-empty");
    let target = (floor_bytes / 2).max(1) as u32;

    let (stream, report) = encode_inter_sequence_with_residue_target_report(
        &seq,
        &intra,
        &inter,
        &pics,
        target,
        InterRateControl::Cbr,
    );

    assert_eq!(report.len(), pics.len() - 1);
    let mut carry: i64 = 0;
    for r in &report {
        assert_eq!(r.ref1_picture_number, 10, "inter references the anchor");
        if r.qindex < 127 {
            assert!(
                r.actual_residue_bytes <= r.requested_residue_bytes,
                "fitting qindex must stay within the codeblock-residue budget: {r:?}"
            );
        }
        carry += r.actual_residue_bytes as i64 - target as i64;
    }
    let _ = carry;

    assert_eq!(
        decode_frame_count(stream),
        pics.len(),
        "codeblock-residue sequence must decode to one frame per picture"
    );

    // The convenience wrapper returns byte-identical output.
    let plain = encode_inter_sequence_with_residue_target(
        &seq,
        &intra,
        &inter,
        &pics,
        target,
        InterRateControl::Cbr,
    );
    let (stream2, _) = encode_inter_sequence_with_residue_target_report(
        &seq,
        &intra,
        &inter,
        &pics,
        target,
        InterRateControl::Cbr,
    );
    assert_eq!(
        plain, stream2,
        "wrapper matches _report stream (codeblock grid)"
    );
}

/// **§11.2.6 global motion threads through the sequence driver**
/// (round-382). The driver clones `inter_params` per picture (overriding
/// only the residue qindex), so a `GlobalMotionConfig` must reach every
/// `0x09` picture — the emitted PPPs signal `using_global` with the
/// caller's field, all blocks go global, and the whole rate-controlled
/// chain still decodes to one frame per input with the residue budget
/// respected.
#[test]
fn sequence_driver_threads_global_motion() {
    use oxideav_dirac::encoder_inter::GlobalMotionConfig;

    let (seq, frames) = fixture();
    let inputs = input_pictures(&frames);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter_params = InterEncoderParams {
        mv_precision: 0,
        residue: Some(ResidueParams::default_for(WaveletFilter::LeGall5_3, 3)),
        // Zero-translation global field: pan_tilt (-1, -1) ⇒ (0, 0)
        // after the §15.8.8 rounding bias. The camera-pan content is
        // absorbed by the residue; what's pinned is the plumbing.
        global_motion: Some(GlobalMotionConfig::pan_tilt_all(-1, -1)),
        ..InterEncoderParams::default()
    };

    let target = 900u32;
    let (stream, report) = encode_inter_sequence_with_residue_target_report(
        &seq,
        &intra_params,
        &inter_params,
        &inputs,
        target,
        InterRateControl::PerPicture,
    );

    // Every inter picture's PPP signals using_global with our field.
    use oxideav_dirac::bits::BitReader;
    use oxideav_dirac::parse_info::ParseInfo;
    use oxideav_dirac::picture_inter::parse_picture_prediction_parameters;
    let mut pos = 0usize;
    let mut inter_units = 0usize;
    while let Some(pi) = ParseInfo::parse(&stream, pos) {
        if pi.parse_code == 0x09 {
            let payload = &stream[pos + 13..pos + pi.next_parse_offset as usize];
            let mut r = BitReader::new(payload);
            r.byte_align();
            let _pic_num = r.read_uint_lit(4);
            let _d1 = r.read_sint();
            r.byte_align();
            let pred = parse_picture_prediction_parameters(&mut r, &seq, 1).expect("PPP");
            assert!(pred.using_global, "inter picture must signal using_global");
            let g1 = pred.global1.expect("global1 present");
            assert_eq!(g1.pan_tilt, (-1, -1), "pan_tilt reaches the wire");
            inter_units += 1;
        }
        if pi.next_parse_offset == 0 {
            break;
        }
        pos += pi.next_parse_offset as usize;
        if pos >= stream.len() {
            break;
        }
    }
    assert_eq!(inter_units, 3, "three 0x09 pictures in the chain");

    // The whole stream still decodes to one frame per input picture.
    assert_eq!(decode_frame_count(stream), inputs.len());

    // Rate control still functions: every satisfiable request fits.
    assert_eq!(report.len(), 3);
    for e in &report {
        assert_eq!(e.requested_residue_bytes, target, "PerPicture request");
        if e.qindex < 127 {
            assert!(
                e.actual_residue_bytes <= e.requested_residue_bytes,
                "pic {}: residue {} bytes over budget {}",
                e.picture_number,
                e.actual_residue_bytes,
                e.requested_residue_bytes
            );
        }
    }
}

/// **Round-386: per-picture automatic global-motion estimation in the
/// sequence driver.** With `auto_global_motion` set (affine model, 0.5
/// threshold) on the camera-pan fixture, every inter picture must:
/// signal `using_global` on the wire, carry the measured fraction +
/// applied flag in its report entry, and the whole chain must still
/// decode to one frame per input with rate control intact.
#[test]
fn sequence_driver_auto_global_estimates_per_picture() {
    use oxideav_dirac::encoder_inter::{AutoGlobalMotion, GlobalMotionModel};

    let (seq, frames) = fixture();
    let inputs = input_pictures(&frames);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter_params = InterEncoderParams {
        mv_precision: 0,
        residue: Some(ResidueParams::default_for(WaveletFilter::LeGall5_3, 3)),
        auto_global_motion: Some(AutoGlobalMotion {
            model: GlobalMotionModel::Affine,
            min_fraction: 0.5,
        }),
        ..InterEncoderParams::default()
    };

    let target = 900u32;
    let (stream, report) = encode_inter_sequence_with_residue_target_report(
        &seq,
        &intra_params,
        &inter_params,
        &inputs,
        target,
        InterRateControl::PerPicture,
    );

    // Telemetry: the estimate ran on every picture; the whole-frame pan
    // fixture must clear the 0.5 threshold on each.
    assert_eq!(report.len(), 3);
    for e in &report {
        let f = e
            .global_fraction
            .expect("auto estimation must run on every picture");
        assert!(
            f >= 0.5,
            "pic {}: pan fixture fraction {f} below threshold",
            e.picture_number
        );
        assert!(
            e.global_applied,
            "pic {}: estimate not applied",
            e.picture_number
        );
    }

    // Wire: every 0x09 picture signals using_global.
    use oxideav_dirac::bits::BitReader;
    use oxideav_dirac::parse_info::ParseInfo;
    use oxideav_dirac::picture_inter::parse_picture_prediction_parameters;
    let mut pos = 0usize;
    let mut inter_units = 0usize;
    while let Some(pi) = ParseInfo::parse(&stream, pos) {
        if pi.parse_code == 0x09 {
            let payload = &stream[pos + 13..pos + pi.next_parse_offset as usize];
            let mut r = BitReader::new(payload);
            r.byte_align();
            let _pic_num = r.read_uint_lit(4);
            let _d1 = r.read_sint();
            r.byte_align();
            let pred = parse_picture_prediction_parameters(&mut r, &seq, 1).expect("PPP");
            assert!(
                pred.using_global,
                "auto-applied picture must signal using_global"
            );
            assert!(pred.global1.is_some(), "global1 must be on the wire");
            inter_units += 1;
        }
        if pi.next_parse_offset == 0 {
            break;
        }
        pos += pi.next_parse_offset as usize;
        if pos >= stream.len() {
            break;
        }
    }
    assert_eq!(inter_units, 3, "three 0x09 pictures in the chain");

    // The stream still decodes to one frame per input picture.
    assert_eq!(decode_frame_count(stream), inputs.len());

    // Rate control still functions: every satisfiable request fits.
    for e in &report {
        assert_eq!(e.requested_residue_bytes, target);
        if e.qindex < 127 {
            assert!(e.actual_residue_bytes <= e.requested_residue_bytes);
        }
    }
}

/// **Round-386: telemetry-only auto-global run.** With a threshold
/// above 1.0 the estimate can never be applied: the report still
/// carries the measured fraction per picture, but no picture signals
/// `using_global` and the emitted stream is byte-identical to a run
/// with auto estimation off — the estimator must be observation-only
/// below threshold.
#[test]
fn sequence_driver_auto_global_telemetry_only_leaves_stream_untouched() {
    use oxideav_dirac::encoder_inter::{AutoGlobalMotion, GlobalMotionModel};

    let (seq, frames) = fixture();
    let inputs = input_pictures(&frames);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let base = InterEncoderParams {
        mv_precision: 0,
        residue: Some(ResidueParams::default_for(WaveletFilter::LeGall5_3, 3)),
        ..InterEncoderParams::default()
    };
    let telemetry = InterEncoderParams {
        auto_global_motion: Some(AutoGlobalMotion {
            model: GlobalMotionModel::Affine,
            min_fraction: 1.01,
        }),
        ..base.clone()
    };

    let target = 900u32;
    let (stream_base, report_base) = encode_inter_sequence_with_residue_target_report(
        &seq,
        &intra_params,
        &base,
        &inputs,
        target,
        InterRateControl::Cbr,
    );
    let (stream_tel, report_tel) = encode_inter_sequence_with_residue_target_report(
        &seq,
        &intra_params,
        &telemetry,
        &inputs,
        target,
        InterRateControl::Cbr,
    );

    assert_eq!(
        stream_base, stream_tel,
        "telemetry-only auto-global must not change a single byte"
    );
    for (b, t) in report_base.iter().zip(report_tel.iter()) {
        assert!(b.global_fraction.is_none(), "auto off ⇒ no fraction");
        assert!(!b.global_applied);
        assert!(
            t.global_fraction.is_some(),
            "telemetry run must still measure the fraction"
        );
        assert!(!t.global_applied, "1.01 threshold can never be met");
        assert_eq!(b.qindex, t.qindex, "rate decisions must be identical");
        assert_eq!(b.actual_residue_bytes, t.actual_residue_bytes);
    }
}

/// **Round-386: explicit config wins over auto.** When both
/// `global_motion` and `auto_global_motion` are set, the caller's
/// explicit model reaches the wire unchanged and the report shows the
/// estimator never ran.
#[test]
fn sequence_driver_explicit_global_beats_auto() {
    use oxideav_dirac::encoder_inter::{AutoGlobalMotion, GlobalMotionConfig};

    let (seq, frames) = fixture();
    let inputs = input_pictures(&frames);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter_params = InterEncoderParams {
        mv_precision: 0,
        residue: Some(ResidueParams::default_for(WaveletFilter::LeGall5_3, 3)),
        global_motion: Some(GlobalMotionConfig::pan_tilt_all(-1, -1)),
        auto_global_motion: Some(AutoGlobalMotion::default()),
        ..InterEncoderParams::default()
    };

    let (stream, report) = encode_inter_sequence_with_residue_target_report(
        &seq,
        &intra_params,
        &inter_params,
        &inputs,
        900,
        InterRateControl::PerPicture,
    );

    for e in &report {
        assert!(
            e.global_fraction.is_none(),
            "explicit config must suppress the estimator"
        );
        assert!(!e.global_applied);
    }

    // The explicit pan_tilt reaches every 0x09 PPP unchanged.
    use oxideav_dirac::bits::BitReader;
    use oxideav_dirac::parse_info::ParseInfo;
    use oxideav_dirac::picture_inter::parse_picture_prediction_parameters;
    let mut pos = 0usize;
    let mut inter_units = 0usize;
    while let Some(pi) = ParseInfo::parse(&stream, pos) {
        if pi.parse_code == 0x09 {
            let payload = &stream[pos + 13..pos + pi.next_parse_offset as usize];
            let mut r = BitReader::new(payload);
            r.byte_align();
            let _pic_num = r.read_uint_lit(4);
            let _d1 = r.read_sint();
            r.byte_align();
            let pred = parse_picture_prediction_parameters(&mut r, &seq, 1).expect("PPP");
            assert!(pred.using_global);
            assert_eq!(
                pred.global1.expect("global1").pan_tilt,
                (-1, -1),
                "caller's explicit pan_tilt must win over auto"
            );
            inter_units += 1;
        }
        if pi.next_parse_offset == 0 {
            break;
        }
        pos += pi.next_parse_offset as usize;
        if pos >= stream.len() {
            break;
        }
    }
    assert_eq!(inter_units, 3);
}
