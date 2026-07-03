//! §14 fragment EMITTER integration tests (round-386).
//!
//! Round 248 landed the decode side of VC-2 v3 fragmented pictures
//! (`FragmentAssembler` + `FragmentedPictureDecoder`); this file pins
//! the round-386 **encode** side: `fragment_picture_payload` (split any
//! conformant LD/HQ picture payload into a `[setup][data…]` §14
//! fragment sequence) and the two stream drivers
//! `encode_single_{hq,ld}_intra_fragmented_stream`.
//!
//! The oracle for every test is the crate's own proven decode path:
//! the same picture delivered non-fragmented must decode bit-exactly
//! equal to the emitter's fragment sequence pushed through
//! `FragmentedPictureDecoder` — across fragment sizes (all slices in
//! one data fragment, one slice per fragment, ragged chunks) and both
//! profiles (the LD path additionally exercises the §14.5 trailing
//! DC-prediction kick).
//!
//! Material consulted: SMPTE ST 2042-1:2022 §10.5.1/§10.5.2 (parse
//! info + Table 5 parse codes), §12.2/§12.4 (picture layout), §13.5.3.2
//! / §13.5.4 (LD/HQ slice widths), §14.1–§14.5 (fragment sequencing,
//! header, reassembly, DC kick). No external implementation consulted.

use oxideav_dirac::encoder::{
    encode_single_hq_intra_stream, encode_single_ld_intra_stream, make_minimal_sequence,
    make_minimal_sequence_ld, EncoderParams, LdEncoderParams,
};
use oxideav_dirac::fragment::{
    encode_single_hq_intra_fragmented_stream, encode_single_ld_intra_fragmented_stream,
    fragment_picture_payload, FragmentEmitError, FragmentUnit, FragmentedPictureDecoder,
};
use oxideav_dirac::parse_info::ParseInfo;
use oxideav_dirac::picture::{decode_picture, DecodedPicture};
use oxideav_dirac::sequence::{parse_sequence_header, SequenceHeader};
use oxideav_dirac::stream::DataUnitIter;
use oxideav_dirac::video_format::ChromaFormat;
use oxideav_dirac::wavelet::WaveletFilter;

fn smooth_plane(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h);
    for yy in 0..h {
        for xx in 0..w {
            let t = (xx as u32).wrapping_mul(7).wrapping_add((yy as u32) * 13) + seed;
            v.push((40 + (t % 160)) as u8);
        }
    }
    v
}

/// Decode a non-fragmented single-picture stream to its reference
/// `DecodedPicture` + sequence header.
fn reference_decode(stream: &[u8]) -> (SequenceHeader, DecodedPicture) {
    let units: Vec<_> = DataUnitIter::new(stream).collect();
    let seq = parse_sequence_header(units[0].payload).expect("sequence header parses");
    let pic = &units[1];
    let reference = decode_picture(pic.payload, pic.parse_info, &seq).expect("reference decode");
    (seq, reference)
}

/// Drive an emitter fragment-unit sequence through
/// `FragmentedPictureDecoder` and return the reassembled picture.
fn decode_fragments(seq: &SequenceHeader, units: &[FragmentUnit]) -> DecodedPicture {
    let mut dec = FragmentedPictureDecoder::new(seq);
    let mut prev = 0u32;
    for (i, unit) in units.iter().enumerate() {
        let pi = ParseInfo {
            parse_code: unit.parse_code,
            next_parse_offset: (ParseInfo::SIZE + unit.payload.len()) as u32,
            previous_parse_offset: prev,
        };
        prev = pi.next_parse_offset;
        if i == 0 {
            dec.on_setup_fragment(&pi, &unit.payload).expect("setup");
        } else {
            dec.on_data_fragment(&pi, &unit.payload).expect("data");
        }
    }
    assert!(
        dec.assembler().fragmented_picture_done(),
        "emitter fragment sequence must complete the picture"
    );
    dec.finish().expect("finish")
}

fn assert_pictures_equal(a: &DecodedPicture, b: &DecodedPicture, label: &str) {
    assert_eq!(
        a.picture_number, b.picture_number,
        "{label}: picture number"
    );
    assert_eq!(a.y, b.y, "{label}: Y");
    assert_eq!(a.u, b.u, "{label}: U");
    assert_eq!(a.v, b.v, "{label}: V");
}

/// HQ payload fragmented at several chunk sizes — every shape must
/// reassemble bit-exactly to the non-fragmented reference.
#[test]
fn hq_emitter_all_chunk_sizes_bit_exact() {
    let (w, h) = (16u32, 16u32);
    let y = smooth_plane(16, 16, 1);
    let u = smooth_plane(8, 8, 5);
    let v = smooth_plane(8, 8, 9);
    let seq = make_minimal_sequence(w, h, ChromaFormat::Yuv420);
    let params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 2);
    let stream = encode_single_hq_intra_stream(&seq, &params, 7, &y, &u, &v);
    let (seq_parsed, reference) = reference_decode(&stream);

    let units_all: Vec<_> = DataUnitIter::new(&stream).collect();
    let payload = units_all[1].payload;
    let total_slices = (params.slices_x * params.slices_y) as usize;

    for chunk in [1u32, 2, 3, total_slices as u32, u32::MAX] {
        let units = fragment_picture_payload(payload, 0xE8, 2, chunk).expect("fragment HQ payload");
        // setup + ceil(total/chunk) data fragments.
        let expect_data = total_slices.div_ceil(chunk.min(total_slices as u32) as usize);
        assert_eq!(units.len(), 1 + expect_data, "chunk {chunk}: unit count");
        assert!(units.iter().all(|f| f.parse_code == 0xEC));
        let got = decode_fragments(&seq_parsed, &units);
        assert_pictures_equal(&got, &reference, &format!("HQ chunk {chunk}"));
    }
}

/// LD payload fragmented at several chunk sizes — pins the §13.5.3.2
/// closed-form slice widths in the emitter's walk plus the §14.5 DC
/// kick on reassembly.
#[test]
fn ld_emitter_all_chunk_sizes_bit_exact() {
    let (w, h) = (16u32, 16u32);
    let y = smooth_plane(16, 16, 3);
    let u = smooth_plane(8, 8, 7);
    let v = smooth_plane(8, 8, 11);
    let seq = make_minimal_sequence_ld(w, h, ChromaFormat::Yuv420);
    let params = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 2, 4, 4, 64);
    let stream = encode_single_ld_intra_stream(&seq, &params, 9, &y, &u, &v);
    let (seq_parsed, reference) = reference_decode(&stream);

    let units_all: Vec<_> = DataUnitIter::new(&stream).collect();
    let payload = units_all[1].payload;
    let total_slices = (params.slices_x * params.slices_y) as usize;

    for chunk in [1u32, 2, 5, total_slices as u32] {
        let units = fragment_picture_payload(payload, 0xC8, 2, chunk).expect("fragment LD payload");
        assert!(units.iter().all(|f| f.parse_code == 0xCC));
        let got = decode_fragments(&seq_parsed, &units);
        assert_pictures_equal(&got, &reference, &format!("LD chunk {chunk}"));
    }
}

/// Data fragments must carry the §14.2 raster offset of their first
/// slice: with chunk = slices_x each fragment is exactly one slice row
/// and `y_offset` must walk 0, 1, 2, … with `x_offset = 0`.
#[test]
fn emitter_raster_offsets_walk_rows() {
    let (w, h) = (16u32, 16u32);
    let y = smooth_plane(16, 16, 21);
    let u = smooth_plane(8, 8, 23);
    let v = smooth_plane(8, 8, 27);
    let seq = make_minimal_sequence(w, h, ChromaFormat::Yuv420);
    let params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 2);
    let stream = encode_single_hq_intra_stream(&seq, &params, 1, &y, &u, &v);
    let units_all: Vec<_> = DataUnitIter::new(&stream).collect();
    let payload = units_all[1].payload;

    let units =
        fragment_picture_payload(payload, 0xE8, 2, params.slices_x).expect("fragment HQ payload");
    assert_eq!(units.len(), 1 + params.slices_y as usize);
    for (row, frag) in units[1..].iter().enumerate() {
        let p = &frag.payload;
        let slice_count = u16::from_be_bytes([p[6], p[7]]);
        let x_off = u16::from_be_bytes([p[8], p[9]]);
        let y_off = u16::from_be_bytes([p[10], p[11]]);
        assert_eq!(slice_count as u32, params.slices_x, "row {row} count");
        assert_eq!(x_off, 0, "row {row} x_offset");
        assert_eq!(y_off as usize, row, "row {row} y_offset");
    }
}

/// The full HQ v3 stream driver: sequence header carries
/// `version_major = 3`, the parse-offset chain is wired, every
/// fragment unit is `0xEC`, and the reassembled picture equals the
/// non-fragmented v3 encode of the same input.
#[test]
fn hq_fragmented_stream_driver_end_to_end() {
    let (w, h) = (16u32, 16u32);
    let y = smooth_plane(16, 16, 31);
    let u = smooth_plane(8, 8, 33);
    let v = smooth_plane(8, 8, 37);
    let seq = make_minimal_sequence(w, h, ChromaFormat::Yuv420);
    let params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 2).with_major_version_3();

    let stream = encode_single_hq_intra_fragmented_stream(&seq, &params, 11, &y, &u, &v, 2);

    // Reference: the same picture non-fragmented, same v3 params.
    let ref_stream = encode_single_hq_intra_stream(&seq, &params, 11, &y, &u, &v);
    // The reference stream's own sequence header is v2-shaped unless
    // the caller bumped it; decode the payload against the v3 header
    // the fragmented driver emits so both sides parse the §12.4.4
    // flags identically.
    let frag_units: Vec<_> = DataUnitIter::new(&stream).collect();
    let seq3 = parse_sequence_header(frag_units[0].payload).expect("v3 sequence header");
    assert_eq!(seq3.parse_parameters.version_major, 3);

    let ref_units: Vec<_> = DataUnitIter::new(&ref_stream).collect();
    let reference = decode_picture(ref_units[1].payload, ref_units[1].parse_info, &seq3)
        .expect("v3 non-fragmented reference decode");

    // Walk the parse-info chain: [0x00][0xEC…][0x10], offsets wired.
    let mut pos = 0usize;
    let mut codes = Vec::new();
    let mut prev_len = 0u32;
    while let Some(pi) = ParseInfo::parse(&stream, pos) {
        codes.push(pi.parse_code);
        assert_eq!(
            pi.previous_parse_offset, prev_len,
            "previous_parse_offset chain at {pos}"
        );
        if pi.next_parse_offset == 0 {
            break;
        }
        prev_len = pi.next_parse_offset;
        pos += pi.next_parse_offset as usize;
    }
    assert_eq!(codes[0], 0x00);
    assert_eq!(*codes.last().unwrap(), 0x10);
    assert!(codes[1..codes.len() - 1].iter().all(|&c| c == 0xEC));
    assert!(codes.len() >= 4, "setup + at least one data fragment");

    // Reassemble through the production fragmented decoder.
    let mut dec = FragmentedPictureDecoder::new(&seq3);
    for unit in DataUnitIter::new(&stream) {
        match unit.parse_info.parse_code {
            0x00 | 0x10 => {}
            0xEC => {
                if dec.transform_parameters().is_none() {
                    dec.on_setup_fragment(&unit.parse_info, unit.payload)
                        .expect("setup");
                } else {
                    dec.on_data_fragment(&unit.parse_info, unit.payload)
                        .expect("data");
                }
            }
            other => panic!("unexpected parse code {other:#04X}"),
        }
    }
    let got = dec.finish().expect("finish");
    assert_pictures_equal(&got, &reference, "HQ v3 stream driver");
}

/// LD v3 stream driver end-to-end (adds the §14.5 DC kick).
#[test]
fn ld_fragmented_stream_driver_end_to_end() {
    let (w, h) = (16u32, 16u32);
    let y = smooth_plane(16, 16, 41);
    let u = smooth_plane(8, 8, 43);
    let v = smooth_plane(8, 8, 47);
    let seq = make_minimal_sequence_ld(w, h, ChromaFormat::Yuv420);
    let mut params = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 2, 4, 4, 64);
    params.major_version = 3;

    let stream = encode_single_ld_intra_fragmented_stream(&seq, &params, 13, &y, &u, &v, 3);
    let frag_units: Vec<_> = DataUnitIter::new(&stream).collect();
    let seq3 = parse_sequence_header(frag_units[0].payload).expect("v3 sequence header");
    assert_eq!(seq3.parse_parameters.version_major, 3);

    let ref_stream = encode_single_ld_intra_stream(&seq, &params, 13, &y, &u, &v);
    let ref_units: Vec<_> = DataUnitIter::new(&ref_stream).collect();
    let reference = decode_picture(ref_units[1].payload, ref_units[1].parse_info, &seq3)
        .expect("v3 non-fragmented reference decode");

    let mut dec = FragmentedPictureDecoder::new(&seq3);
    for unit in DataUnitIter::new(&stream) {
        match unit.parse_info.parse_code {
            0x00 | 0x10 => {}
            0xCC => {
                if dec.transform_parameters().is_none() {
                    dec.on_setup_fragment(&unit.parse_info, unit.payload)
                        .expect("setup");
                } else {
                    dec.on_data_fragment(&unit.parse_info, unit.payload)
                        .expect("data");
                }
            }
            other => panic!("unexpected parse code {other:#04X}"),
        }
    }
    let got = dec.finish().expect("finish");
    assert_pictures_equal(&got, &reference, "LD v3 stream driver");
}

/// Error surface: zero chunk size, non-picture parse codes (including
/// an already-fragment code), and truncated payloads must be rejected
/// with the right variants — never a panic.
#[test]
fn emitter_error_surface() {
    let (w, h) = (16u32, 16u32);
    let y = smooth_plane(16, 16, 51);
    let u = smooth_plane(8, 8, 53);
    let v = smooth_plane(8, 8, 57);
    let seq = make_minimal_sequence(w, h, ChromaFormat::Yuv420);
    let params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 2);
    let stream = encode_single_hq_intra_stream(&seq, &params, 3, &y, &u, &v);
    let units_all: Vec<_> = DataUnitIter::new(&stream).collect();
    let payload = units_all[1].payload;

    assert_eq!(
        fragment_picture_payload(payload, 0xE8, 2, 0).unwrap_err(),
        FragmentEmitError::ZeroSlicesPerFragment
    );
    assert_eq!(
        fragment_picture_payload(payload, 0x00, 2, 1).unwrap_err(),
        FragmentEmitError::NotAPictureParseCode(0x00)
    );
    assert_eq!(
        fragment_picture_payload(payload, 0xEC, 2, 1).unwrap_err(),
        FragmentEmitError::NotAPictureParseCode(0xEC)
    );
    assert_eq!(
        fragment_picture_payload(&payload[..2], 0xE8, 2, 1).unwrap_err(),
        FragmentEmitError::Truncated
    );
    // Cut mid-slice: keep the TP but drop the payload tail.
    let cut = payload.len() - 3;
    assert_eq!(
        fragment_picture_payload(&payload[..cut], 0xE8, 2, 1).unwrap_err(),
        FragmentEmitError::Truncated
    );
}
