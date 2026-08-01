//! P-chain sequence driver validator.
//!
//! Exercises `encode_inter_p_chain_with_residue_target` and its
//! `_report` companion — one HQ intra anchor (`0xEC`) followed by N
//! 1-ref **reference** inter pictures (`0x0D`), each referencing the
//! previous picture in the chain, encoded closed-loop (the encoder
//! decodes its own emission and references the reconstruction, exactly
//! as the decoder will). Pins:
//!
//!   1. the stream's parse-code chain is `0x00, 0xEC, 0x0D…, 0x10`;
//!   2. every inter picture references the *previous* picture, not the
//!      anchor;
//!   3. at residue qindex 0 (reversible wavelet) the whole chain
//!      round-trips bit-exactly through the crate's decoder;
//!   4. lossy budgets stay decodable with bounded per-picture error
//!      (closed loop — no drift accumulation down the chain);
//!   5. the CBR accumulator matches the running `Σ(actual − target)`
//!      and the VBV bucket clamps banked savings at `-buffer_bytes`;
//!   6. the anchor-only degeneracy behaves.

use oxideav_core::CodecRegistry;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_dirac::encoder::{make_minimal_sequence, EncoderParams};
use oxideav_dirac::encoder_inter::{
    encode_inter_p_chain_with_residue_target, encode_inter_p_chain_with_residue_target_report,
    inter_residue_qindex_diagnostic, InterEncoderParams, InterInputPicture, InterRateControl,
};
use oxideav_dirac::parse_info::ParseInfo;
use oxideav_dirac::sequence::SequenceHeader;
use oxideav_dirac::video_format::ChromaFormat;
use oxideav_dirac::wavelet::WaveletFilter;

struct Frame64 {
    y: [u8; 64 * 64],
    u: [u8; 32 * 32],
    v: [u8; 32 * 32],
}

/// A 16x16 bright square on a flat background, at `(x, y)` luma pels.
fn square_frame(x: usize, y: usize) -> Frame64 {
    let mut fy = [40u8; 64 * 64];
    let mut fu = [110u8; 32 * 32];
    let mut fv = [140u8; 32 * 32];
    for r in 0..16 {
        for c in 0..16 {
            fy[(y + r) * 64 + (x + c)] = 200;
        }
    }
    for r in 0..8 {
        for c in 0..8 {
            fu[(y / 2 + r) * 32 + (x / 2 + c)] = 90;
            fv[(y / 2 + r) * 32 + (x / 2 + c)] = 160;
        }
    }
    Frame64 {
        y: fy,
        u: fu,
        v: fv,
    }
}

/// Anchor plus three frames with the square marching +4 pels right —
/// each frame's best reference is its immediate predecessor.
fn fixture() -> (SequenceHeader, Vec<Frame64>) {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let frames = (0..4).map(|i| square_frame(16 + 4 * i, 24)).collect();
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

fn decode_video_frames(stream: Vec<u8>) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("make decoder");
    let packet = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&packet).expect("send_packet");

    let mut frames = Vec::new();
    while let Ok(Frame::Video(vf)) = dec.receive_frame() {
        frames.push((
            vf.planes[0].data.clone(),
            vf.planes[1].data.clone(),
            vf.planes[2].data.clone(),
        ));
    }
    frames
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let sse: u64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = x as i64 - y as i64;
            (d * d) as u64
        })
        .sum();
    if sse == 0 {
        return f64::INFINITY;
    }
    let mse = sse as f64 / a.len() as f64;
    20.0 * (255.0f64).log10() - 10.0 * mse.log10()
}

fn parse_code_chain(stream: &[u8]) -> Vec<u8> {
    let mut codes = Vec::new();
    let mut pos = 0usize;
    while let Some(pi) = ParseInfo::parse(stream, pos) {
        codes.push(pi.parse_code);
        if pi.next_parse_offset == 0 {
            break;
        }
        pos += pi.next_parse_offset as usize;
    }
    codes
}

#[test]
fn p_chain_emits_reference_inter_parse_codes() {
    let (seq, frames) = fixture();
    let pics = input_pictures(&frames);
    let intra = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter = InterEncoderParams::default();

    let stream = encode_inter_p_chain_with_residue_target(
        &seq,
        &intra,
        &inter,
        &pics,
        u32::MAX,
        InterRateControl::PerPicture,
    );
    assert_eq!(
        parse_code_chain(&stream),
        vec![0x00, 0xEC, 0x0D, 0x0D, 0x0D, 0x10],
        "sequence header, HQ intra reference, three 0x0D reference P, EOS"
    );
}

#[test]
fn p_chain_references_previous_picture_not_anchor() {
    let (seq, frames) = fixture();
    let pics = input_pictures(&frames);
    let intra = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter = InterEncoderParams::default();

    let (_stream, report) = encode_inter_p_chain_with_residue_target_report(
        &seq,
        &intra,
        &inter,
        &pics,
        u32::MAX,
        InterRateControl::PerPicture,
    );
    assert_eq!(report.len(), 3);
    let refs: Vec<u32> = report.iter().map(|r| r.ref1_picture_number).collect();
    assert_eq!(
        refs,
        vec![10, 11, 12],
        "each P must reference its immediate predecessor"
    );
    let nums: Vec<u32> = report.iter().map(|r| r.picture_number).collect();
    assert_eq!(nums, vec![11, 12, 13]);
}

#[test]
fn p_chain_q0_round_trips_bit_exact() {
    let (seq, frames) = fixture();
    let pics = input_pictures(&frames);
    let intra = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter = InterEncoderParams::default(); // residue on, qindex 0.

    let stream = encode_inter_p_chain_with_residue_target(
        &seq,
        &intra,
        &inter,
        &pics,
        u32::MAX,
        InterRateControl::PerPicture,
    );
    let decoded = decode_video_frames(stream);
    assert_eq!(decoded.len(), frames.len());
    for (i, ((dy, du, dv), f)) in decoded.iter().zip(&frames).enumerate() {
        assert_eq!(dy.as_slice(), &f.y[..], "frame {i} Y bit-exact at q0");
        assert_eq!(du.as_slice(), &f.u[..], "frame {i} U bit-exact at q0");
        assert_eq!(dv.as_slice(), &f.v[..], "frame {i} V bit-exact at q0");
    }
}

#[test]
fn p_chain_lossy_budget_decodes_without_drift() {
    let (seq, frames) = fixture();
    let pics = input_pictures(&frames);
    let intra = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter = InterEncoderParams::default();

    // Cost of the first P at q0, then squeeze to force escalation.
    let (_q, q0_bytes) = inter_residue_qindex_diagnostic(
        &seq,
        &inter,
        &frames[1].y,
        &frames[1].u,
        &frames[1].v,
        &frames[0].y,
        &frames[0].u,
        &frames[0].v,
        u32::MAX,
    );
    assert!(q0_bytes > 0);
    let target = (q0_bytes / 4).max(1) as u32;

    let (stream, report) = encode_inter_p_chain_with_residue_target_report(
        &seq,
        &intra,
        &inter,
        &pics,
        target,
        InterRateControl::PerPicture,
    );
    assert!(
        report.iter().any(|r| r.qindex > 0),
        "quarter budget must escalate at least one picture's qindex"
    );

    let decoded = decode_video_frames(stream);
    assert_eq!(decoded.len(), frames.len());
    let mut psnrs = Vec::new();
    for (i, ((dy, _, _), f)) in decoded.iter().zip(&frames).enumerate() {
        let p = psnr(dy, &f.y);
        eprintln!("frame {i}: Y PSNR {p:.2} dB");
        assert!(p > 28.0, "frame {i} Y PSNR {p:.2} dB under lossy budget");
        psnrs.push(p);
    }
    // Closed loop: the tail of the chain must stay in the same quality
    // regime as its head — per-picture quantisation error is allowed,
    // open-loop drift accumulation is not.
    let first = psnrs[1].min(psnrs[0]);
    let last = *psnrs.last().unwrap();
    assert!(
        first - last < 8.0,
        "chain tail fell {delta:.2} dB below its head — drift",
        delta = first - last
    );
}

#[test]
fn p_chain_cbr_accumulator_matches_running_sum() {
    let (seq, frames) = fixture();
    let pics = input_pictures(&frames);
    let intra = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter = InterEncoderParams::default();

    let (_q, q0_bytes) = inter_residue_qindex_diagnostic(
        &seq,
        &inter,
        &frames[1].y,
        &frames[1].u,
        &frames[1].v,
        &frames[0].y,
        &frames[0].u,
        &frames[0].v,
        u32::MAX,
    );
    let target = (q0_bytes / 2).max(1) as u32;

    let (_stream, report) = encode_inter_p_chain_with_residue_target_report(
        &seq,
        &intra,
        &inter,
        &pics,
        target,
        InterRateControl::Cbr,
    );
    let mut expect = 0i64;
    for (i, r) in report.iter().enumerate() {
        // Request tracks the carry: target - carry (clamped at 0).
        let want = (target as i64 - expect).clamp(0, u32::MAX as i64) as u32;
        assert_eq!(r.requested_residue_bytes, want, "picture {i} request");
        expect += r.actual_residue_bytes as i64 - target as i64;
        assert_eq!(r.running_surplus_bytes, expect, "picture {i} surplus");
    }
}

#[test]
fn p_chain_vbv_clamps_banked_savings() {
    let (seq, frames) = fixture();
    let pics = input_pictures(&frames);
    let intra = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter = InterEncoderParams::default();

    // Generous target so pictures undershoot and bank savings.
    let buffer_bytes = 64u32;
    let (_stream, report) = encode_inter_p_chain_with_residue_target_report(
        &seq,
        &intra,
        &inter,
        &pics,
        50_000,
        InterRateControl::Vbv { buffer_bytes },
    );
    for (i, r) in report.iter().enumerate() {
        assert!(
            r.running_surplus_bytes >= -(buffer_bytes as i64),
            "picture {i}: savings {surplus} below -buffer_bytes",
            surplus = r.running_surplus_bytes
        );
    }
}

#[test]
fn p_chain_anchor_only_degenerates_cleanly() {
    let (seq, frames) = fixture();
    let pics = input_pictures(&frames[..1]);
    let intra = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter = InterEncoderParams::default();

    let (stream, report) = encode_inter_p_chain_with_residue_target_report(
        &seq,
        &intra,
        &inter,
        &pics,
        1000,
        InterRateControl::Cbr,
    );
    assert!(report.is_empty());
    assert_eq!(parse_code_chain(&stream), vec![0x00, 0xEC, 0x10]);
    let decoded = decode_video_frames(stream);
    assert_eq!(decoded.len(), 1);
    assert_eq!(decoded[0].0.as_slice(), &frames[0].y[..]);
}
