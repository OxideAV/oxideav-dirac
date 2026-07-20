//! Round-419 — deep-colour (u16) multi-picture **rate-controlled**
//! inter sequences.
//!
//! The 8-bit driver invariants live in `encoder_inter_sequence_rate.rs`;
//! this suite re-pins them on 10- and 16-bit `&[u16]` sources through
//! the sample-generic `encode_inter_sequence_with_residue_target[_report]`
//! driver (HQ deep intra anchor `0xEC` + N 1-ref `0x09` pictures):
//!
//!   1. every stream decodes to one frame per input picture on the
//!      deep output surfaces, and with a budget at/above the q0 floor
//!      the whole sequence is **bit-exact** against the u16 sources;
//!   2. `PerPicture` requests the bare target every picture;
//!   3. the `Cbr` accumulator equals the running `Σ(actual − target)`
//!      and shapes the next request to `target − carry`;
//!   4. `Vbv` clamps banked savings at `-buffer_bytes` (peak
//!      residue-size cap on the request);
//!   5. `VbvHysteresis` additionally rate-limits the per-picture drain
//!      and collapses to plain `Vbv` when the drain cap can't bite;
//!   6. a tight budget escalates qindex (monotonically vs. a loose
//!      budget) and the stream still decodes.
//!
//! Wall: BBC Dirac spec §10.3.8/§10.5.2 (deep signal ranges), §11.3 /
//! §13.4.4 (residue + qindex legality) from
//! `docs/video/dirac/dirac-spec-latest.pdf`. Rate policy is pure
//! encoder-side shaping. No external source, no web.

use oxideav_core::{CodecId, CodecParameters, CodecRegistry, Frame, Packet, TimeBase, VideoFrame};
use oxideav_dirac::encoder::{make_minimal_sequence_with_signal_range, EncoderParams};
use oxideav_dirac::encoder_inter::{
    encode_inter_sequence_with_residue_target, encode_inter_sequence_with_residue_target_report,
    inter_residue_qindex_diagnostic, InterEncoderParams, InterInputPicture, InterPictureRate,
    InterRateControl,
};
use oxideav_dirac::sequence::SequenceHeader;
use oxideav_dirac::video_format::{ChromaFormat, SignalRange};
use oxideav_dirac::wavelet::WaveletFilter;

/// Owned deep 64x64 4:2:0 frame planes.
struct DeepFrame {
    y: Vec<u16>,
    u: Vec<u16>,
    v: Vec<u16>,
}

/// Deterministic deep texture sampled on an infinite lattice so a pan
/// is a true translation with fresh content entering at the edge.
fn tex(x: i64, y: i64, depth: u32) -> u16 {
    let max = (1u64 << depth) - 1;
    let mix = (x + 1024) as u64 * 17 + (y + 1024) as u64 * 31;
    ((mix * 2654435761) % (max + 1)) as u16
}

/// Anchor (pan 0) + three inter frames at growing integer pans, all at
/// `depth` bits. Chroma pans with luma at half rate (4:2:0).
fn fixture(depth: u32, sr: SignalRange) -> (SequenceHeader, Vec<DeepFrame>) {
    let seq = make_minimal_sequence_with_signal_range(64, 64, ChromaFormat::Yuv420, sr);
    let mut frames = Vec::new();
    for dx in [0i64, 1, 3, 5] {
        let y: Vec<u16> = (0..64 * 64)
            .map(|i| tex(i % 64 + dx, i / 64, depth))
            .collect();
        let u: Vec<u16> = (0..32 * 32)
            .map(|i| tex(i % 32 + dx / 2 + 4096, i / 32, depth))
            .collect();
        let v: Vec<u16> = (0..32 * 32)
            .map(|i| tex(i % 32 + dx / 2 + 8192, i / 32, depth))
            .collect();
        frames.push(DeepFrame { y, u, v });
    }
    (seq, frames)
}

fn input_pictures(frames: &[DeepFrame]) -> Vec<InterInputPicture<'_, u16>> {
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

fn decode_frames(stream: Vec<u8>) -> Vec<VideoFrame> {
    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("make decoder");
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&pkt).expect("send_packet");
    let mut out = Vec::new();
    while let Ok(frame) = dec.receive_frame() {
        match frame {
            Frame::Video(v) => out.push(v),
            other => panic!("expected video frame, got {other:?}"),
        }
    }
    out
}

fn plane_as_u16(data: &[u8]) -> Vec<u16> {
    data.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// q0 residue cost of the largest-pan picture against the anchor — the
/// budget floor the tests straddle.
fn q0_floor_bytes(seq: &SequenceHeader, inter: &InterEncoderParams, frames: &[DeepFrame]) -> u32 {
    let (_q, bytes) = inter_residue_qindex_diagnostic(
        seq,
        inter,
        &frames[3].y,
        &frames[3].u,
        &frames[3].v,
        &frames[0].y,
        &frames[0].u,
        &frames[0].v,
        u32::MAX,
    );
    bytes as u32
}

/// Shared per-entry sanity: a fitting qindex's actual bytes are within
/// the requested budget; qindex 127 marks "budget unreachable".
fn assert_entries_fit(report: &[InterPictureRate]) {
    for e in report {
        if e.qindex < 127 {
            assert!(
                e.actual_residue_bytes <= e.requested_residue_bytes,
                "picture {}: actual {} exceeds requested {} at qindex {}",
                e.picture_number,
                e.actual_residue_bytes,
                e.requested_residue_bytes,
                e.qindex
            );
        }
    }
}

/// PerPicture at 10-bit: every entry requests the bare target; with the
/// budget at the q0 floor every picture stays at qindex 0 and the whole
/// decoded sequence is bit-exact against the u16 sources.
#[test]
fn per_picture_10bit_loose_budget_bit_exact() {
    let (seq, frames) = fixture(10, SignalRange::PRESET_10BIT_FULL);
    let pics = input_pictures(&frames);
    let intra = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter = InterEncoderParams::default();
    let target = q0_floor_bytes(&seq, &inter, &frames);

    let (stream, report) = encode_inter_sequence_with_residue_target_report(
        &seq,
        &intra,
        &inter,
        &pics,
        target,
        InterRateControl::PerPicture,
    );
    assert_eq!(report.len(), 3);
    for e in &report {
        assert_eq!(e.requested_residue_bytes, target, "PerPicture bare target");
        assert_eq!(e.qindex, 0, "q0 floor budget must not escalate");
    }
    assert_entries_fit(&report);

    let decoded = decode_frames(stream);
    assert_eq!(decoded.len(), 4, "one frame per input picture");
    for (i, (df, sf)) in decoded.iter().zip(frames.iter()).enumerate() {
        assert_eq!(
            plane_as_u16(&df.planes[0].data),
            sf.y,
            "frame {i} Y bit-exact at q0"
        );
        assert_eq!(
            plane_as_u16(&df.planes[1].data),
            sf.u,
            "frame {i} U bit-exact at q0"
        );
        assert_eq!(
            plane_as_u16(&df.planes[2].data),
            sf.v,
            "frame {i} V bit-exact at q0"
        );
    }
}

/// Cbr at 16-bit: the accumulator is the running `Σ(actual − target)`
/// and each request is `clamp(target − carry_before, 0..)`. The final
/// surplus magnitude stays under one picture's target (long-run rate
/// adherence) on this fixture.
#[test]
fn cbr_16bit_accumulator_and_requests_track_carry() {
    let (seq, frames) = fixture(16, SignalRange::PRESET_16BIT_FULL);
    let pics = input_pictures(&frames);
    let intra = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter = InterEncoderParams::default();
    // Straddle the floor: roughly the mid-pan picture's cost.
    let target = q0_floor_bytes(&seq, &inter, &frames) * 3 / 4;

    let (stream, report) = encode_inter_sequence_with_residue_target_report(
        &seq,
        &intra,
        &inter,
        &pics,
        target,
        InterRateControl::Cbr,
    );
    assert_eq!(report.len(), 3);
    assert_entries_fit(&report);

    let mut carry: i64 = 0;
    for e in &report {
        let expect_request = (target as i64 - carry).clamp(0, u32::MAX as i64) as u32;
        assert_eq!(
            e.requested_residue_bytes, expect_request,
            "picture {}: Cbr request must be target − carry",
            e.picture_number
        );
        carry += e.actual_residue_bytes as i64 - target as i64;
        assert_eq!(
            e.running_surplus_bytes, carry,
            "picture {}: running surplus must be Σ(actual − target)",
            e.picture_number
        );
    }
    let final_dev = report.last().unwrap().running_surplus_bytes.unsigned_abs();
    eprintln!("16-bit Cbr: target {target} B/picture, final |Σ(actual − target)| = {final_dev} B");
    assert!(
        final_dev <= target as u64,
        "16-bit Cbr long-run deviation {final_dev} exceeds one picture target {target}"
    );

    assert_eq!(decode_frames(stream).len(), 4);
}

/// Vbv at 16-bit: the savings side of the bucket is clamped at
/// `-buffer_bytes`, so every request is capped at `target + buffer`.
#[test]
fn vbv_16bit_clamps_savings_and_caps_requests() {
    let (seq, frames) = fixture(16, SignalRange::PRESET_16BIT_FULL);
    let pics = input_pictures(&frames);
    let intra = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter = InterEncoderParams::default();
    // Generous target so early pictures bank savings; small bucket so
    // the clamp actually bites.
    let target = q0_floor_bytes(&seq, &inter, &frames);
    let buffer = target / 8;

    let (stream, report) = encode_inter_sequence_with_residue_target_report(
        &seq,
        &intra,
        &inter,
        &pics,
        target,
        InterRateControl::Vbv {
            buffer_bytes: buffer,
        },
    );
    assert_eq!(report.len(), 3);
    assert_entries_fit(&report);
    for e in &report {
        assert!(
            e.running_surplus_bytes >= -(buffer as i64),
            "picture {}: savings {} deeper than -buffer {}",
            e.picture_number,
            e.running_surplus_bytes,
            buffer
        );
        assert!(
            e.requested_residue_bytes <= target + buffer,
            "picture {}: request {} exceeds target + buffer {}",
            e.picture_number,
            e.requested_residue_bytes,
            target + buffer
        );
    }
    assert_eq!(decode_frames(stream).len(), 4);
}

/// VbvHysteresis at 10-bit: the per-picture drain cap bounds every
/// request at `target + min(buffer, drain)`; with `drain >= buffer` the
/// stream is byte-identical to plain Vbv (the cap can't bite).
#[test]
fn vbv_hysteresis_10bit_drain_cap_and_vbv_collapse() {
    let (seq, frames) = fixture(10, SignalRange::PRESET_10BIT_FULL);
    let pics = input_pictures(&frames);
    let intra = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter = InterEncoderParams::default();
    let target = q0_floor_bytes(&seq, &inter, &frames);
    let buffer = target / 4;
    let drain = buffer / 4;

    let (stream, report) = encode_inter_sequence_with_residue_target_report(
        &seq,
        &intra,
        &inter,
        &pics,
        target,
        InterRateControl::VbvHysteresis {
            buffer_bytes: buffer,
            max_drain_per_picture: drain,
        },
    );
    assert_eq!(report.len(), 3);
    assert_entries_fit(&report);
    for e in &report {
        assert!(
            e.requested_residue_bytes <= target + drain,
            "picture {}: request {} exceeds target + drain cap {}",
            e.picture_number,
            e.requested_residue_bytes,
            target + drain
        );
    }
    assert_eq!(decode_frames(stream).len(), 4);

    // drain >= buffer ⇒ identical stream to plain Vbv.
    let hyst_wide = encode_inter_sequence_with_residue_target(
        &seq,
        &intra,
        &inter,
        &pics,
        target,
        InterRateControl::VbvHysteresis {
            buffer_bytes: buffer,
            max_drain_per_picture: buffer,
        },
    );
    let vbv = encode_inter_sequence_with_residue_target(
        &seq,
        &intra,
        &inter,
        &pics,
        target,
        InterRateControl::Vbv {
            buffer_bytes: buffer,
        },
    );
    assert_eq!(
        hyst_wide, vbv,
        "VbvHysteresis with drain >= buffer must collapse to plain Vbv"
    );
}

/// A tight budget at 16-bit escalates the per-picture qindex relative
/// to a loose budget (monotone rate control) and the stream still
/// decodes to one frame per picture on the 16-bit output surface.
#[test]
fn tight_budget_16bit_escalates_qindex_and_decodes() {
    let (seq, frames) = fixture(16, SignalRange::PRESET_16BIT_FULL);
    let pics = input_pictures(&frames);
    let intra = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter = InterEncoderParams::default();
    let floor = q0_floor_bytes(&seq, &inter, &frames);

    let (loose_stream, loose) = encode_inter_sequence_with_residue_target_report(
        &seq,
        &intra,
        &inter,
        &pics,
        floor,
        InterRateControl::PerPicture,
    );
    let (tight_stream, tight) = encode_inter_sequence_with_residue_target_report(
        &seq,
        &intra,
        &inter,
        &pics,
        floor / 4,
        InterRateControl::PerPicture,
    );
    assert_eq!(loose.len(), 3);
    assert_eq!(tight.len(), 3);
    let mut escalated = false;
    for (l, t) in loose.iter().zip(tight.iter()) {
        assert!(
            t.qindex >= l.qindex,
            "picture {}: tight-budget qindex {} below loose-budget {}",
            t.picture_number,
            t.qindex,
            l.qindex
        );
        if t.qindex > l.qindex {
            escalated = true;
        }
        // Tighter budget can only shrink the residue payload.
        assert!(
            t.actual_residue_bytes <= l.actual_residue_bytes,
            "picture {}: tight-budget residue grew",
            t.picture_number
        );
    }
    assert!(
        escalated,
        "quartered budget never escalated qindex — fixture too easy to pin rate control"
    );
    assert!(
        tight_stream.len() < loose_stream.len(),
        "tight-budget stream should be smaller"
    );
    assert_eq!(decode_frames(tight_stream).len(), 4);
}
