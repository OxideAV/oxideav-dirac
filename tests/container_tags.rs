//! Shared legacy container-tag co-claim.
//!
//! Matroska `V_DIRAC`, the MP4 sample entry `drac` and MP4
//! ObjectTypeIndication `0xA4` are the registered container codes for
//! this bitstream syntax. VC-2 (the intra-only profile family) has no
//! container codes of its own and rides under them, so both this crate
//! and a dedicated VC-2-profile crate claim the same tags; the codec
//! registry resolves the contest by probe strength. The contract:
//!
//! * long-GOP (core-syntax) evidence → this crate answers with a
//!   strong claim (`0.95`), strictly above the deliberately weak
//!   long-GOP claim (`0.25`) the VC-2-profile side makes;
//! * intra-profile-only evidence → this crate answers weakly (`0.3`)
//!   so a dedicated VC-2 registration outranks it, but still nonzero
//!   because this crate decodes those streams end-to-end;
//! * no evidence → middling legacy-owner default (`0.5`);
//! * bytes that are not this syntax at all → `0.0`.

use oxideav_core::{CodecRegistry, CodecTag, ProbeContext};
use oxideav_dirac::encoder::{encode_single_hq_intra_stream, make_minimal_sequence, EncoderParams};
use oxideav_dirac::encoder_inter::{
    encode_intra_then_inter_stream, synthetic_translating_pair_64, InterEncoderParams,
    InterInputPicture,
};
use oxideav_dirac::video_format::ChromaFormat;
use oxideav_dirac::wavelet::WaveletFilter;
use oxideav_dirac::{
    container_tag_probe, PROBE_CONFIDENCE_INTRA_PROFILE, PROBE_CONFIDENCE_LONG_GOP,
    PROBE_CONFIDENCE_NO_EVIDENCE,
};

/// The declared strength of the VC-2-profile side's long-GOP claim on
/// the same shared tags. Our long-GOP claim must stay strictly above
/// it or contested long-GOP streams stop resolving to this crate.
const VC2_SIDE_LONG_GOP_CLAIM: f32 = 0.25;

fn registry() -> CodecRegistry {
    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    reg
}

fn long_gop_stream() -> Vec<u8> {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter_params = InterEncoderParams::default();
    let (y0, u0, v0, y1, u1, v1) = synthetic_translating_pair_64(4, 0);
    let intra = InterInputPicture {
        picture_number: 0,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let inter = InterInputPicture {
        picture_number: 1,
        y: &y1,
        u: &u1,
        v: &v1,
    };
    encode_intra_then_inter_stream(&seq, &intra_params, &inter_params, &intra, &inter)
}

fn intra_profile_stream() -> Vec<u8> {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let y = vec![64u8; 64 * 64];
    let u = vec![128u8; 32 * 32];
    let v = vec![128u8; 32 * 32];
    encode_single_hq_intra_stream(&seq, &params, 0, &y, &u, &v)
}

#[test]
fn all_three_legacy_tags_are_registered() {
    let reg = registry();
    let want = [
        CodecTag::fourcc(b"drac"),
        CodecTag::matroska("V_DIRAC"),
        CodecTag::mp4_object_type(0xA4),
    ];
    for tag in &want {
        assert!(
            reg.all_tag_registrations()
                .any(|(t, id)| t == tag && id.as_str() == "dirac"),
            "tag {tag} not registered to dirac"
        );
    }
}

#[test]
fn each_tag_resolves_to_dirac_without_evidence() {
    let reg = registry();
    for tag in [
        CodecTag::fourcc(b"drac"),
        CodecTag::matroska("V_DIRAC"),
        CodecTag::mp4_object_type(0xA4),
    ] {
        let ctx = ProbeContext::new(&tag);
        let id = reg.resolve_tag_ref(&ctx).expect("tag resolves");
        assert_eq!(id.as_str(), "dirac");
    }
}

#[test]
fn long_gop_packet_probes_strong_and_beats_vc2_side_claim() {
    let stream = long_gop_stream();
    let tag = CodecTag::matroska("V_DIRAC");
    let ctx = ProbeContext::new(&tag).packet(&stream);
    let conf = container_tag_probe(&ctx);
    assert_eq!(conf, PROBE_CONFIDENCE_LONG_GOP);
    assert!(
        conf > VC2_SIDE_LONG_GOP_CLAIM,
        "long-GOP claim {conf} must outrank the VC-2 side's {VC2_SIDE_LONG_GOP_CLAIM}"
    );
}

#[test]
fn long_gop_evidence_wins_through_registry_resolution() {
    let reg = registry();
    let stream = long_gop_stream();
    let tag = CodecTag::matroska("V_DIRAC");
    let ctx = ProbeContext::new(&tag).packet(&stream);
    let id = reg.resolve_tag_ref(&ctx).expect("resolves");
    assert_eq!(id.as_str(), "dirac");
}

#[test]
fn intra_profile_packet_probes_weak_but_nonzero() {
    let stream = intra_profile_stream();
    let tag = CodecTag::matroska("V_DIRAC");
    let ctx = ProbeContext::new(&tag).packet(&stream);
    let conf = container_tag_probe(&ctx);
    assert_eq!(conf, PROBE_CONFIDENCE_INTRA_PROFILE);
    assert!(conf > 0.0, "intra-profile fallback claim must stay nonzero");
    assert!(
        conf < PROBE_CONFIDENCE_LONG_GOP,
        "intra-profile claim must rank below the long-GOP claim"
    );
    // A dedicated intra-profile registration answering with anything
    // stronger than this value wins the contest — that is the point.
    assert!(conf < PROBE_CONFIDENCE_NO_EVIDENCE);
}

#[test]
fn evidence_from_header_blob_counts_like_packet_bytes() {
    let stream = long_gop_stream();
    let tag = CodecTag::fourcc(b"drac");
    let ctx = ProbeContext::new(&tag).header(&stream);
    assert_eq!(container_tag_probe(&ctx), PROBE_CONFIDENCE_LONG_GOP);
}

#[test]
fn no_evidence_probes_middling_default() {
    let tag = CodecTag::matroska("V_DIRAC");
    let ctx = ProbeContext::new(&tag);
    assert_eq!(container_tag_probe(&ctx), PROBE_CONFIDENCE_NO_EVIDENCE);
    // The Matroska registration carries no CodecPrivate; an empty blob
    // is the same as no blob.
    let empty: &[u8] = &[];
    let ctx = ProbeContext::new(&tag).header(empty);
    assert_eq!(container_tag_probe(&ctx), PROBE_CONFIDENCE_NO_EVIDENCE);
}

#[test]
fn non_dirac_bytes_probe_zero() {
    let tag = CodecTag::matroska("V_DIRAC");
    let garbage = vec![0xA5u8; 512];
    let ctx = ProbeContext::new(&tag).packet(&garbage);
    assert_eq!(container_tag_probe(&ctx), 0.0);
}

#[test]
fn sequence_header_only_units_probe_no_evidence_default() {
    // A stream head cut before the first picture: parse-info for a
    // sequence header, then end-of-sequence. Structurally this syntax,
    // but profile-undecided.
    let stream = intra_profile_stream();
    // The head layout is [pi_sh(13)][sh payload]…; keep only the first
    // unit by truncating at the picture's parse-info and appending an
    // end-of-sequence header.
    let first =
        oxideav_dirac::parse_info::ParseInfo::parse(&stream, 0).expect("sequence header unit");
    let mut head = stream[..first.next_parse_offset as usize].to_vec();
    let prev = head.len() as u32;
    head.extend_from_slice(b"BBCD");
    head.push(0x10);
    head.extend_from_slice(&0u32.to_be_bytes());
    head.extend_from_slice(&prev.to_be_bytes());
    let tag = CodecTag::matroska("V_DIRAC");
    let ctx = ProbeContext::new(&tag).packet(&head);
    assert_eq!(container_tag_probe(&ctx), PROBE_CONFIDENCE_NO_EVIDENCE);
}

#[test]
fn truncated_long_gop_head_still_probes_by_scanning() {
    // Cut a long-GOP stream mid-picture: offsets past the cut are
    // unusable, but the units before it still classify.
    let stream = long_gop_stream();
    // Find the inter picture's parse-info offset by walking units.
    let mut pos = 0usize;
    let mut inter_at = None;
    while let Some(pi) = oxideav_dirac::parse_info::ParseInfo::parse(&stream, pos) {
        if pi.is_inter() {
            inter_at = Some(pos);
            break;
        }
        if pi.next_parse_offset == 0 {
            break;
        }
        pos += pi.next_parse_offset as usize;
    }
    let inter_at = inter_at.expect("stream contains an inter picture");
    // Keep the inter parse-info header plus a few payload bytes only.
    let cut = &stream[..(inter_at + 20).min(stream.len())];
    let tag = CodecTag::matroska("V_DIRAC");
    let ctx = ProbeContext::new(&tag).packet(cut);
    assert_eq!(container_tag_probe(&ctx), PROBE_CONFIDENCE_LONG_GOP);
}
