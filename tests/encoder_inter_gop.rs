//! GOP-structured (I / P / B) sequence driver validator —
//! `encode_inter_gop_with_residue_target` and its `_report` companion.
//! Pins:
//!
//!   1. the coded-order parse-code chain for full groups
//!      (`0x00, 0xEC, [0x0D, 0x0A…]…, 0x10`) and for a ragged tail
//!      (`0x09` trailing pictures);
//!   2. reference wiring: each P references the previous reference,
//!      each B references its two enclosing references;
//!   3. at residue qindex 0 the whole GOP round-trips bit-exactly
//!      (frames matched through the coded→display mapping);
//!   4. `b_between_refs = 0` degenerates to the pure P-chain shape;
//!   5. one rate-control accumulator spans P and B pictures alike
//!      (CBR running-sum law over the mixed coded-order report);
//!   6. a lossy budget stays decodable with bounded error.

use oxideav_core::CodecRegistry;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_dirac::encoder::{make_minimal_sequence, EncoderParams};
use oxideav_dirac::encoder_inter::{
    encode_inter_gop_with_residue_target, encode_inter_gop_with_residue_target_report,
    GopPictureKind, GopStructure, InterEncoderParams, InterInputPicture, InterRateControl,
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

/// A 16x16 bright square at `(x, 24)` luma pels.
fn square_frame(x: usize) -> Frame64 {
    let mut fy = [40u8; 64 * 64];
    let mut fu = [110u8; 32 * 32];
    let mut fv = [140u8; 32 * 32];
    for r in 0..16 {
        for c in 0..16 {
            fy[(24 + r) * 64 + (x + c)] = 200;
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

/// `n` display-order frames with the square marching +2 pels per frame.
fn fixture(n: usize) -> (SequenceHeader, Vec<Frame64>) {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let frames = (0..n).map(|i| square_frame(12 + 2 * i)).collect();
    (seq, frames)
}

fn input_pictures(frames: &[Frame64]) -> Vec<InterInputPicture<'_>> {
    frames
        .iter()
        .enumerate()
        .map(|(i, f)| InterInputPicture {
            picture_number: 100 + i as u32,
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

/// Display index of each coded-order picture for `n` frames and GOP
/// spacing `s` (mirrors the driver's coded-order emission: anchor,
/// then per group the reference P before its Bs, then trailing).
fn coded_to_display(n: usize, s: usize) -> Vec<usize> {
    let mut order = vec![0usize];
    let group = s + 1;
    let mut idx = 1usize;
    while idx < n {
        if n - idx >= group {
            order.push(idx + group - 1); // the reference P
            for b in 0..group - 1 {
                order.push(idx + b); // its Bs, display order
            }
            idx += group;
        } else {
            order.push(idx); // trailing 0x09
            idx += 1;
        }
    }
    order
}

#[test]
fn gop_emits_expected_coded_order_parse_codes() {
    let (seq, frames) = fixture(7); // anchor + 3 full [B, P] groups
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
    assert_eq!(
        parse_code_chain(&stream),
        vec![0x00, 0xEC, 0x0D, 0x0A, 0x0D, 0x0A, 0x0D, 0x0A, 0x10],
        "I, then three [P-before-its-B] groups in coded order"
    );
}

#[test]
fn gop_ragged_tail_uses_non_reference_p() {
    let (seq, frames) = fixture(5); // anchor + [B,B,P] + 1 trailing
    let pics = input_pictures(&frames);
    let intra = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter = InterEncoderParams::default();

    let (stream, report) = encode_inter_gop_with_residue_target_report(
        &seq,
        &intra,
        &inter,
        &pics,
        GopStructure { b_between_refs: 2 },
        u32::MAX,
        InterRateControl::PerPicture,
    );
    assert_eq!(
        parse_code_chain(&stream),
        vec![0x00, 0xEC, 0x0D, 0x0A, 0x0A, 0x09, 0x10],
    );
    let kinds: Vec<GopPictureKind> = report.iter().map(|r| r.kind).collect();
    assert_eq!(
        kinds,
        vec![
            GopPictureKind::ReferenceP,
            GopPictureKind::BipredB,
            GopPictureKind::BipredB,
            GopPictureKind::TrailingP,
        ]
    );
    // Trailing P references the last reference (picture 103).
    assert_eq!(report[3].ref1_picture_number, 103);
    assert_eq!(report[3].ref2_picture_number, None);
}

#[test]
fn gop_reference_wiring_is_correct() {
    let (seq, frames) = fixture(7);
    let pics = input_pictures(&frames);
    let intra = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter = InterEncoderParams::default();

    let (_stream, report) = encode_inter_gop_with_residue_target_report(
        &seq,
        &intra,
        &inter,
        &pics,
        GopStructure { b_between_refs: 1 },
        u32::MAX,
        InterRateControl::PerPicture,
    );
    // Coded order: P102(ref 100), B101(100→102), P104(ref 102),
    // B103(102→104), P106(ref 104), B105(104→106).
    let want: Vec<(GopPictureKind, u32, u32, Option<u32>)> = vec![
        (GopPictureKind::ReferenceP, 102, 100, None),
        (GopPictureKind::BipredB, 101, 100, Some(102)),
        (GopPictureKind::ReferenceP, 104, 102, None),
        (GopPictureKind::BipredB, 103, 102, Some(104)),
        (GopPictureKind::ReferenceP, 106, 104, None),
        (GopPictureKind::BipredB, 105, 104, Some(106)),
    ];
    let got: Vec<(GopPictureKind, u32, u32, Option<u32>)> = report
        .iter()
        .map(|r| {
            (
                r.kind,
                r.picture_number,
                r.ref1_picture_number,
                r.ref2_picture_number,
            )
        })
        .collect();
    assert_eq!(got, want);
}

#[test]
fn gop_q0_round_trips_bit_exact_through_coded_order() {
    let (seq, frames) = fixture(7);
    let pics = input_pictures(&frames);
    let intra = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter = InterEncoderParams::default(); // residue on, qindex 0.

    let stream = encode_inter_gop_with_residue_target(
        &seq,
        &intra,
        &inter,
        &pics,
        GopStructure { b_between_refs: 1 },
        u32::MAX,
        InterRateControl::PerPicture,
    );
    let decoded = decode_video_frames(stream);
    assert_eq!(decoded.len(), frames.len());
    let order = coded_to_display(frames.len(), 1);
    for (coded_i, &display_i) in order.iter().enumerate() {
        let (dy, du, dv) = &decoded[coded_i];
        let f = &frames[display_i];
        assert_eq!(
            dy.as_slice(),
            &f.y[..],
            "coded frame {coded_i} (display {display_i}) Y bit-exact at q0"
        );
        assert_eq!(du.as_slice(), &f.u[..], "display {display_i} U");
        assert_eq!(dv.as_slice(), &f.v[..], "display {display_i} V");
    }
}

#[test]
fn gop_zero_bs_degenerates_to_p_chain_shape() {
    let (seq, frames) = fixture(4);
    let pics = input_pictures(&frames);
    let intra = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter = InterEncoderParams::default();

    let stream = encode_inter_gop_with_residue_target(
        &seq,
        &intra,
        &inter,
        &pics,
        GopStructure { b_between_refs: 0 },
        u32::MAX,
        InterRateControl::PerPicture,
    );
    assert_eq!(
        parse_code_chain(&stream),
        vec![0x00, 0xEC, 0x0D, 0x0D, 0x0D, 0x10],
    );
}

#[test]
fn gop_cbr_accumulator_spans_p_and_b_pictures() {
    let (seq, frames) = fixture(7);
    let pics = input_pictures(&frames);
    let intra = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter = InterEncoderParams::default();

    let target = 600u32;
    let (_stream, report) = encode_inter_gop_with_residue_target_report(
        &seq,
        &intra,
        &inter,
        &pics,
        GopStructure { b_between_refs: 1 },
        target,
        InterRateControl::Cbr,
    );
    let mut expect = 0i64;
    for (i, r) in report.iter().enumerate() {
        let want = (target as i64 - expect).clamp(0, u32::MAX as i64) as u32;
        assert_eq!(
            r.requested_residue_bytes, want,
            "coded picture {i} ({:?}) request",
            r.kind
        );
        expect += r.actual_residue_bytes as i64 - target as i64;
        assert_eq!(r.running_surplus_bytes, expect, "coded picture {i} surplus");
    }
    // The mixed report must actually contain both kinds.
    assert!(report.iter().any(|r| r.kind == GopPictureKind::ReferenceP));
    assert!(report.iter().any(|r| r.kind == GopPictureKind::BipredB));
}

#[test]
fn gop_lossy_budget_stays_decodable() {
    let (seq, frames) = fixture(7);
    let pics = input_pictures(&frames);
    let intra = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter = InterEncoderParams::default();

    let (stream, report) = encode_inter_gop_with_residue_target_report(
        &seq,
        &intra,
        &inter,
        &pics,
        GopStructure { b_between_refs: 1 },
        150,
        InterRateControl::PerPicture,
    );
    assert!(
        report.iter().any(|r| r.qindex > 0),
        "a 150-byte budget must escalate some qindex"
    );
    let decoded = decode_video_frames(stream);
    assert_eq!(decoded.len(), frames.len());
    let order = coded_to_display(frames.len(), 1);
    for (coded_i, &display_i) in order.iter().enumerate() {
        let p = psnr(&decoded[coded_i].0, &frames[display_i].y);
        eprintln!("display frame {display_i}: Y PSNR {p:.2} dB");
        assert!(p > 28.0, "display frame {display_i} Y PSNR {p:.2} dB");
    }
}
