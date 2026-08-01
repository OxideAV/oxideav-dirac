//! Deep-colour (u16) coverage of the reference-P-chain and GOP
//! sequence drivers — the round-419 deep inter pipeline extended to the
//! chained / GOP shapes:
//!
//!   1. a 10-bit and a 16-bit closed-loop P-chain (`0xEC` + `0x0D…`)
//!      round-trip **bit-exactly** at residue qindex 0;
//!   2. a 16-bit I/P/B GOP (`0x0D` + `0x0A`) round-trips bit-exactly
//!      at q0 through the coded→display mapping;
//!   3. the CBR accumulator law holds on a deep GOP report spanning P
//!      and B pictures;
//!   4. a tight deep P-chain budget escalates qindex and still
//!      decodes.
//!
//! Wall: BBC Dirac spec §10.3.8/§10.5.2 (deep signal ranges), §11.3 /
//! §13.4.4 (residue + qindex legality) from
//! `docs/video/dirac/dirac-spec-latest.pdf`. No external source, no
//! web.

use oxideav_core::{CodecId, CodecParameters, CodecRegistry, Frame, Packet, TimeBase, VideoFrame};
use oxideav_dirac::encoder::{make_minimal_sequence_with_signal_range, EncoderParams};
use oxideav_dirac::encoder_inter::{
    encode_inter_gop_with_residue_target, encode_inter_gop_with_residue_target_report,
    encode_inter_p_chain_with_residue_target, encode_inter_p_chain_with_residue_target_report,
    inter_residue_qindex_diagnostic, GopPictureKind, GopStructure, InterEncoderParams,
    InterInputPicture, InterRateControl,
};
use oxideav_dirac::sequence::SequenceHeader;
use oxideav_dirac::video_format::{ChromaFormat, SignalRange};
use oxideav_dirac::wavelet::WaveletFilter;

struct DeepFrame {
    y: Vec<u16>,
    u: Vec<u16>,
    v: Vec<u16>,
}

/// Deterministic deep texture on an infinite lattice (pan = true
/// translation with fresh edge content).
fn tex(x: i64, y: i64, depth: u32) -> u16 {
    let max = (1u64 << depth) - 1;
    let mix = (x + 1024) as u64 * 17 + (y + 1024) as u64 * 31;
    ((mix * 2654435761) % (max + 1)) as u16
}

/// `n` display-order frames panning +1 luma pel per frame at `depth`.
fn fixture(n: usize, depth: u32, sr: SignalRange) -> (SequenceHeader, Vec<DeepFrame>) {
    let seq = make_minimal_sequence_with_signal_range(64, 64, ChromaFormat::Yuv420, sr);
    let mut frames = Vec::new();
    for i in 0..n as i64 {
        let y: Vec<u16> = (0..64 * 64)
            .map(|p| tex(p % 64 + i, p / 64, depth))
            .collect();
        let u: Vec<u16> = (0..32 * 32)
            .map(|p| tex(p % 32 + i / 2 + 4096, p / 32, depth))
            .collect();
        let v: Vec<u16> = (0..32 * 32)
            .map(|p| tex(p % 32 + i / 2 + 8192, p / 32, depth))
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
    while let Ok(Frame::Video(v)) = dec.receive_frame() {
        out.push(v);
    }
    out
}

fn plane_as_u16(data: &[u8]) -> Vec<u16> {
    data.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn assert_frame_bit_exact(vf: &VideoFrame, f: &DeepFrame, label: &str) {
    assert_eq!(plane_as_u16(&vf.planes[0].data), f.y, "{label} Y");
    assert_eq!(plane_as_u16(&vf.planes[1].data), f.u, "{label} U");
    assert_eq!(plane_as_u16(&vf.planes[2].data), f.v, "{label} V");
}

#[test]
fn deep_p_chain_q0_bit_exact_10bit() {
    let (seq, frames) = fixture(4, 10, SignalRange::PRESET_10BIT_FULL);
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
    let decoded = decode_frames(stream);
    assert_eq!(decoded.len(), frames.len());
    for (i, (vf, f)) in decoded.iter().zip(&frames).enumerate() {
        assert_frame_bit_exact(vf, f, &format!("10-bit P-chain frame {i}"));
    }
}

#[test]
fn deep_p_chain_q0_bit_exact_16bit() {
    let (seq, frames) = fixture(4, 16, SignalRange::PRESET_16BIT_FULL);
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
    let decoded = decode_frames(stream);
    assert_eq!(decoded.len(), frames.len());
    for (i, (vf, f)) in decoded.iter().zip(&frames).enumerate() {
        assert_frame_bit_exact(vf, f, &format!("16-bit P-chain frame {i}"));
    }
}

#[test]
fn deep_gop_q0_bit_exact_16bit() {
    let (seq, frames) = fixture(5, 16, SignalRange::PRESET_16BIT_FULL);
    let pics = input_pictures(&frames);
    let intra = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter = InterEncoderParams::default();

    let stream = encode_inter_gop_with_residue_target(
        &seq,
        &intra,
        &inter,
        &pics,
        GopStructure { b_between_refs: 1 },
        u32::MAX,
        InterRateControl::PerPicture,
    );
    let decoded = decode_frames(stream);
    assert_eq!(decoded.len(), frames.len());
    // Coded order for n=5, s=1: anchor(0), P(2), B(1), P(4), B(3).
    let order = [0usize, 2, 1, 4, 3];
    for (coded_i, &display_i) in order.iter().enumerate() {
        assert_frame_bit_exact(
            &decoded[coded_i],
            &frames[display_i],
            &format!("16-bit GOP display frame {display_i}"),
        );
    }
}

#[test]
fn deep_gop_cbr_accumulator_law_10bit() {
    let (seq, frames) = fixture(5, 10, SignalRange::PRESET_10BIT_FULL);
    let pics = input_pictures(&frames);
    let intra = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter = InterEncoderParams::default();

    let target = 800u32;
    let (_stream, report) = encode_inter_gop_with_residue_target_report(
        &seq,
        &intra,
        &inter,
        &pics,
        GopStructure { b_between_refs: 1 },
        target,
        InterRateControl::Cbr,
    );
    assert!(report.iter().any(|r| r.kind == GopPictureKind::ReferenceP));
    assert!(report.iter().any(|r| r.kind == GopPictureKind::BipredB));
    let mut expect = 0i64;
    for (i, r) in report.iter().enumerate() {
        let want = (target as i64 - expect).clamp(0, u32::MAX as i64) as u32;
        assert_eq!(r.requested_residue_bytes, want, "deep coded picture {i}");
        expect += r.actual_residue_bytes as i64 - target as i64;
        assert_eq!(r.running_surplus_bytes, expect, "deep coded picture {i}");
    }
}

#[test]
fn deep_p_chain_tight_budget_escalates_and_decodes_16bit() {
    let (seq, frames) = fixture(4, 16, SignalRange::PRESET_16BIT_FULL);
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
    assert!(q0_bytes > 0);
    let target = (q0_bytes / 8).max(1) as u32;

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
        "an eighth-budget must escalate qindex on a deep chain"
    );
    let decoded = decode_frames(stream);
    assert_eq!(decoded.len(), frames.len());
    for vf in &decoded {
        assert_eq!(vf.planes[0].data.len(), 64 * 64 * 2, "P16 Y plane bytes");
    }
}
