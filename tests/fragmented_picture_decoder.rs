//! VC-2 v3 fragmented-picture decoder (`FragmentedPictureDecoder`)
//! integration tests (SMPTE ST 2042-1:2022 §14).
//!
//! Round 248 lands the picture-level driver that ties the §14 fragment
//! state machine (`src/fragment.rs`) to the §13.5 LD / HQ slice
//! coefficient decoders in `src/picture.rs`. The driver takes a
//! sequence header, an LD or HQ fragment-stream, and reproduces the
//! same `DecodedPicture` the non-fragmented path
//! ([`oxideav_dirac::picture::decode_picture`]) yields when that
//! picture is delivered as one unit.
//!
//! The tests synthesise both an LD and an HQ picture via the
//! `oxideav_dirac::encoder::*` round-trip path, decode it the
//! non-fragmented way to capture a reference [`DecodedPicture`], then
//! repackage the encoder's picture-payload byte stream into a
//! `[setup fragment][data fragments…]` shape and feed the result to
//! [`FragmentedPictureDecoder`]. Equivalence with the reference picture
//! is bit-exact at qindex=0 (both LD and HQ).
//!
//! All material consulted:
//! * `docs/video/vc2/vc2-specification.pdf` §10.5.1 (parse-info header),
//!   §10.5.2 (parse codes / Table 4 / Table 5), §13.5.3.2 (LD slice
//!   layout / `slice_bytes`), §13.5.4 (HQ slice layout / length-byte +
//!   scaler), §14.1 (fragment sequencing), §14.2 (fragment header),
//!   §14.3 / §14.4 (state machine + raster coordinates), §14.5
//!   (fragmented-picture trailing DC kick).
//!
//! No external library source / web search consulted. No third-party
//! crate. Black-box: the test exercises the public API of
//! `oxideav_dirac` only.

use oxideav_dirac::bits::BitReader;
use oxideav_dirac::encoder::{
    encode_single_hq_intra_stream, encode_single_ld_intra_stream, make_minimal_sequence,
    make_minimal_sequence_ld, wavelet_index, EncoderParams, LdEncoderParams,
};
use oxideav_dirac::fragment::{
    AssemblerError, FragmentAssembler, FragmentEvent, FragmentHeader, FragmentKind,
    FragmentedPictureDecoder, FragmentedPictureError,
};
use oxideav_dirac::parse_info::ParseInfo;
use oxideav_dirac::picture::{
    decode_picture, slice_bytes as ld_slice_bytes, DecodedPicture, LowDelayProfile,
    TransformParameters,
};
use oxideav_dirac::sequence::parse_sequence_header;
use oxideav_dirac::stream::DataUnitIter;
use oxideav_dirac::video_format::ChromaFormat;
use oxideav_dirac::wavelet::WaveletFilter;

// ----------------------------------------------------------------------
// Helpers: build a fragment payload.
// ----------------------------------------------------------------------

fn setup_fragment_payload(picture_number: u32, transform_params_bytes: &[u8]) -> Vec<u8> {
    let mut p = Vec::with_capacity(8 + transform_params_bytes.len());
    p.extend_from_slice(&picture_number.to_be_bytes());
    // §14.2 `fragment_data_length` — explicitly "contains undefined
    // data and does not contribute to decoding"; pin to zero.
    p.extend_from_slice(&0u16.to_be_bytes());
    // §14.2 `fragment_slice_count` = 0 marks the setup fragment.
    p.extend_from_slice(&0u16.to_be_bytes());
    p.extend_from_slice(transform_params_bytes);
    p
}

fn data_fragment_payload(
    picture_number: u32,
    slice_count: u16,
    x_offset: u16,
    y_offset: u16,
    slice_bytes: &[u8],
) -> Vec<u8> {
    let mut p = Vec::with_capacity(12 + slice_bytes.len());
    p.extend_from_slice(&picture_number.to_be_bytes());
    p.extend_from_slice(&0u16.to_be_bytes());
    p.extend_from_slice(&slice_count.to_be_bytes());
    p.extend_from_slice(&x_offset.to_be_bytes());
    p.extend_from_slice(&y_offset.to_be_bytes());
    p.extend_from_slice(slice_bytes);
    p
}

// ----------------------------------------------------------------------
// Helpers: dissect an encoded HQ / LD picture payload into byte ranges.
// ----------------------------------------------------------------------

/// Return `(picture_number, transform_params_bytes_start..end, Vec<(slice_start, slice_end)>)`
/// for an HQ picture payload encoded by [`encode_single_hq_intra_stream`].
///
/// The picture payload layout (§12.2 + §12.4 + §13.5.4) is:
///
/// 1. 4-byte big-endian `picture_number` (byte-aligned per §12.2).
/// 2. Byte-aligned `transform_parameters()` (§12.4).
/// 3. `slices_x * slices_y` HQ slices. Per §13.5.4 each slice is a
///    sequence of byte-aligned reads: `slice_prefix_bytes` of opaque
///    prefix, then a 1-byte `qindex`, then per-component a 1-byte
///    length-byte followed by `length_byte * slice_size_scaler`
///    coefficient bytes. The slice is byte-aligned by construction so
///    the next slice starts on a byte boundary.
fn dissect_hq_picture_payload(
    payload: &[u8],
    params: &HqDissectParams,
) -> (u32, std::ops::Range<usize>, Vec<std::ops::Range<usize>>) {
    let picture_number = u32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let tp_start = 4;

    // Run a bit reader past the `transform_parameters()` block to learn
    // its byte length (we already know the encoder's settings — we
    // pre-compute the expected layout from those settings).
    let tp_end = locate_hq_tp_end(payload, params);

    let mut slices = Vec::with_capacity((params.slices_x * params.slices_y) as usize);
    let mut p = tp_end;
    for _ in 0..(params.slices_x * params.slices_y) {
        let start = p;
        p += params.slice_prefix_bytes as usize;
        // qindex byte.
        p += 1;
        for _ in 0..3 {
            let len_byte = payload[p];
            p += 1;
            p += (len_byte as usize) * params.slice_size_scaler as usize;
        }
        slices.push(start..p);
    }
    (picture_number, tp_start..tp_end, slices)
}

#[derive(Clone, Copy)]
struct HqDissectParams {
    slices_x: u32,
    slices_y: u32,
    slice_prefix_bytes: u32,
    slice_size_scaler: u32,
    /// §12 `major_version`. When `>= 3` the `transform_parameters()`
    /// block carries the §12.4.4 `extended_transform_parameters()`
    /// flag bits right after `dwt_depth`.
    major_version: u32,
    /// §12.4.4 `extended_transform_parameters()` override mirrored from
    /// the encoder: `(asym_transform_index_flag emitted, wavelet_index_ho,
    /// dwt_depth_ho)`. `None` means the v3 stream emitted both flags
    /// `False` (the symmetric-default form). Only consulted when
    /// `major_version >= 3`.
    ext: Option<HqExtParams>,
}

#[derive(Clone, Copy)]
struct HqExtParams {
    /// The primary `wavelet_index` — needed to decide whether
    /// `asym_transform_index_flag` was emitted as `True` (encoder logic:
    /// `wavelet_index_ho != wavelet_index`).
    wavelet_index: u32,
    /// §12.4.4.2 `wavelet_index_ho`.
    wavelet_index_ho: u32,
    /// §12.4.4.3 `dwt_depth_ho`.
    dwt_depth_ho: u32,
}

impl HqDissectParams {
    fn symmetric(
        slices_x: u32,
        slices_y: u32,
        slice_prefix_bytes: u32,
        slice_size_scaler: u32,
    ) -> Self {
        Self {
            slices_x,
            slices_y,
            slice_prefix_bytes,
            slice_size_scaler,
            major_version: 2,
            ext: None,
        }
    }
}

/// Run a `BitReader` to the end of the `transform_parameters()` block
/// in an HQ picture payload, returning the byte position immediately
/// after the block. The encoder byte-aligns at the entry to the block
/// and at the entry to slice 0, so the returned position is also the
/// start of slice 0.
fn locate_hq_tp_end(payload: &[u8], _params: &HqDissectParams) -> usize {
    // We re-parse with the same helper the production decoder uses —
    // that means reusing `parse_transform_parameters` indirectly via a
    // call to the production decoder on the synthetic picture. Cheaper
    // shortcut: a BitReader walk that mirrors the encoder's emission
    // order. We choose the cheap shortcut so this helper doesn't
    // depend on `parse_transform_parameters` being public.
    //
    // The encoder emits (§12.4 / §13.5.4):
    //   wavelet_index   (uint, exp-Golomb)
    //   dwt_depth       (uint)
    //   slices_x        (uint)
    //   slices_y        (uint)
    //   slice_prefix_bytes (uint)
    //   slice_size_scaler  (uint)
    //   custom_quant_matrix (bool, one bit)
    //   [if true: ll0 + (dwt_depth*3) per-band offsets each `uint`.]
    //   byte_align     (back up to the next byte boundary)
    let mut r = BitReader::new(&payload[4..]);
    let _w = r.read_uint();
    let dwt_depth = r.read_uint();
    // §12.4.4 `extended_transform_parameters()` — v3 only. Mirror the
    // encoder's `write_transform_parameters` emission so the byte cursor
    // lands exactly where the slice data begins.
    let mut dwt_depth_ho = 0u32;
    if _params.major_version >= 3 {
        match _params.ext {
            None => {
                let _asym_idx = r.read_bool(); // asym_transform_index_flag
                let _asym_depth = r.read_bool(); // asym_transform_flag
            }
            Some(ext) => {
                let asym_idx = r.read_bool();
                if asym_idx {
                    let _w_ho = r.read_uint(); // wavelet_index_ho
                }
                let asym_depth = r.read_bool();
                if asym_depth {
                    dwt_depth_ho = r.read_uint();
                }
                debug_assert_eq!(asym_idx, ext.wavelet_index_ho != ext.wavelet_index);
                debug_assert_eq!(asym_depth, ext.dwt_depth_ho != 0);
                debug_assert_eq!(dwt_depth_ho, if asym_depth { ext.dwt_depth_ho } else { 0 });
            }
        }
    }
    let _sx = r.read_uint();
    let _sy = r.read_uint();
    let _pfx = r.read_uint();
    let _scl = r.read_uint();
    let custom = r.read_bool();
    if custom {
        // Asymmetric custom matrix (§12.4.5.3): the level-0 L entry, one
        // H entry per level `1..=dwt_depth_ho`, then HL/LH/HH triplets
        // for the `dwt_depth` symmetric levels. With `dwt_depth_ho == 0`
        // this reduces to the symmetric `ll0 + dwt_depth*3` shape.
        let _ll0 = r.read_uint();
        for _ in 0..dwt_depth_ho {
            let _h = r.read_uint();
        }
        for _ in 0..dwt_depth {
            let _hl = r.read_uint();
            let _lh = r.read_uint();
            let _hh = r.read_uint();
        }
    }
    r.byte_align();
    4 + r.byte_pos()
}

/// LD analogue of [`dissect_hq_picture_payload`]. Each LD slice has a
/// fixed byte width given by [`ld_slice_bytes`] (= §13.5.3.2
/// `slice_bytes`); no need to peek into the slice's bitstream.
fn dissect_ld_picture_payload(
    payload: &[u8],
    params: &LdDissectParams,
) -> (u32, std::ops::Range<usize>, Vec<std::ops::Range<usize>>) {
    let picture_number = u32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let tp_start = 4;
    let tp_end = locate_ld_tp_end(payload, params);
    let mut slices = Vec::with_capacity((params.slices_x * params.slices_y) as usize);
    let mut p = tp_end;
    for sy in 0..params.slices_y {
        for sx in 0..params.slices_x {
            let bytes = ld_slice_bytes(
                params.slices_x,
                params.slice_bytes_numer,
                params.slice_bytes_denom,
                sx,
                sy,
            );
            let start = p;
            p += bytes as usize;
            slices.push(start..p);
        }
    }
    (picture_number, tp_start..tp_end, slices)
}

#[derive(Clone, Copy)]
struct LdDissectParams {
    slices_x: u32,
    slices_y: u32,
    slice_bytes_numer: u32,
    slice_bytes_denom: u32,
}

fn locate_ld_tp_end(payload: &[u8], _params: &LdDissectParams) -> usize {
    let mut r = BitReader::new(&payload[4..]);
    let _w = r.read_uint();
    let dwt_depth = r.read_uint();
    let _sx = r.read_uint();
    let _sy = r.read_uint();
    let _numer = r.read_uint();
    let _denom = r.read_uint();
    let custom = r.read_bool();
    if custom {
        let _ll0 = r.read_uint();
        for _ in 0..dwt_depth {
            let _hl = r.read_uint();
            let _lh = r.read_uint();
            let _hh = r.read_uint();
        }
    }
    r.byte_align();
    4 + r.byte_pos()
}

// ----------------------------------------------------------------------
// Decode the picture the non-fragmented way to capture a reference
// `DecodedPicture` we can compare against.
// ----------------------------------------------------------------------

struct EncodedRound<'a> {
    sequence: oxideav_dirac::sequence::SequenceHeader,
    picture_parse_info: ParseInfo,
    picture_payload: &'a [u8],
    reference: DecodedPicture,
}

fn parse_and_decode<'a>(stream: &'a [u8]) -> EncodedRound<'a> {
    let units: Vec<_> = DataUnitIter::new(stream).collect();
    // [seq][picture][EOS].
    let seq_hdr = parse_sequence_header(units[0].payload).expect("sequence header parses");
    let pic = &units[1];
    let reference =
        decode_picture(pic.payload, pic.parse_info, &seq_hdr).expect("non-fragmented decode");
    EncodedRound {
        sequence: seq_hdr,
        picture_parse_info: pic.parse_info,
        picture_payload: pic.payload,
        reference,
    }
}

// ----------------------------------------------------------------------
// Tests.
// ----------------------------------------------------------------------

fn smooth_plane(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let s = (x as u32)
                .wrapping_add((y as u32).wrapping_mul(13))
                .wrapping_add(seed);
            v.push((s & 0x7F) as u8 + 32);
        }
    }
    v
}

/// HQ qindex=0 picture round-trips bit-exactly through the
/// fragmented-picture decoder when packaged as `[setup][1 data
/// fragment carrying all 4 slices]`. The same one-fragment shape v3
/// streams use when the encoder doesn't bother to fragment but emits
/// a 0xEC parse code (Table 4 picture-fragment HQ) anyway.
#[test]
fn hq_q0_single_data_fragment_bit_exact_vs_non_fragmented() {
    let w: u32 = 16;
    let h: u32 = 16;
    let y = smooth_plane(w as usize, h as usize, 0);
    let u = smooth_plane((w / 2) as usize, (h / 2) as usize, 5);
    let v = smooth_plane((w / 2) as usize, (h / 2) as usize, 11);

    let seq = make_minimal_sequence(w, h, ChromaFormat::Yuv420);
    let params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 2);
    let stream = encode_single_hq_intra_stream(&seq, &params, 7, &y, &u, &v);
    let round = parse_and_decode(&stream);

    let dissect_params = HqDissectParams::symmetric(
        params.slices_x,
        params.slices_y,
        params.slice_prefix_bytes,
        params.slice_size_scaler,
    );
    let (picture_number, tp_range, slices) =
        dissect_hq_picture_payload(round.picture_payload, &dissect_params);
    assert_eq!(picture_number, 7);
    let total_slices = (params.slices_x * params.slices_y) as usize;
    assert_eq!(slices.len(), total_slices);

    // Build [setup payload][one data fragment payload carrying every slice].
    let tp_bytes = &round.picture_payload[tp_range.clone()];
    let setup_payload = setup_fragment_payload(picture_number, tp_bytes);

    let slice_region_start = slices[0].start;
    let slice_region_end = slices.last().unwrap().end;
    let all_slice_bytes = &round.picture_payload[slice_region_start..slice_region_end];
    let data_payload =
        data_fragment_payload(picture_number, total_slices as u16, 0, 0, all_slice_bytes);

    let setup_pi = ParseInfo {
        parse_code: 0xEC, // HQ picture fragment.
        next_parse_offset: (ParseInfo::SIZE + setup_payload.len()) as u32,
        previous_parse_offset: 0,
    };
    let data_pi = ParseInfo {
        parse_code: 0xEC,
        next_parse_offset: (ParseInfo::SIZE + data_payload.len()) as u32,
        previous_parse_offset: setup_pi.next_parse_offset,
    };

    let mut dec = FragmentedPictureDecoder::new(&round.sequence);
    dec.on_setup_fragment(&setup_pi, &setup_payload)
        .expect("setup");
    assert_eq!(dec.picture_number(), 7);
    let tp = dec.transform_parameters().expect("tp captured");
    assert_eq!(tp.profile, LowDelayProfile::HQ);
    assert_eq!(tp.slices_x, params.slices_x);
    assert_eq!(tp.slices_y, params.slices_y);
    assert!(
        !dec.assembler().fragmented_picture_done(),
        "picture not yet complete after setup"
    );

    dec.on_data_fragment(&data_pi, &data_payload).expect("data");
    assert!(
        dec.assembler().fragmented_picture_done(),
        "picture complete after one all-slice data fragment"
    );

    let frag_pic = dec.finish().expect("finish");
    assert_eq!(frag_pic.y, round.reference.y, "Y mismatch");
    assert_eq!(frag_pic.u, round.reference.u, "U mismatch");
    assert_eq!(frag_pic.v, round.reference.v, "V mismatch");
    assert_eq!(frag_pic.picture_number, round.reference.picture_number);
}

/// Same HQ picture but split into one data fragment per slice — the
/// adversarial v3 shape where every slice is its own data unit. Each
/// fragment carries `slice_count = 1` and the raster `(x_offset,
/// y_offset)` matches the slice's grid position.
#[test]
fn hq_q0_one_data_fragment_per_slice_bit_exact_vs_non_fragmented() {
    let w: u32 = 16;
    let h: u32 = 16;
    let y = smooth_plane(w as usize, h as usize, 3);
    let u = smooth_plane((w / 2) as usize, (h / 2) as usize, 9);
    let v = smooth_plane((w / 2) as usize, (h / 2) as usize, 17);

    let seq = make_minimal_sequence(w, h, ChromaFormat::Yuv420);
    let params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 2);
    let stream = encode_single_hq_intra_stream(&seq, &params, 42, &y, &u, &v);
    let round = parse_and_decode(&stream);

    let dissect_params = HqDissectParams::symmetric(
        params.slices_x,
        params.slices_y,
        params.slice_prefix_bytes,
        params.slice_size_scaler,
    );
    let (picture_number, tp_range, slices) =
        dissect_hq_picture_payload(round.picture_payload, &dissect_params);
    assert_eq!(picture_number, 42);

    let tp_bytes = &round.picture_payload[tp_range.clone()];
    let setup_payload = setup_fragment_payload(picture_number, tp_bytes);
    let setup_pi = ParseInfo {
        parse_code: 0xEC,
        next_parse_offset: (ParseInfo::SIZE + setup_payload.len()) as u32,
        previous_parse_offset: 0,
    };

    let mut dec = FragmentedPictureDecoder::new(&round.sequence);
    dec.on_setup_fragment(&setup_pi, &setup_payload)
        .expect("setup");

    let mut idx = 0;
    for sy in 0..params.slices_y {
        for sx in 0..params.slices_x {
            let slice_bytes = &round.picture_payload[slices[idx].clone()];
            let payload =
                data_fragment_payload(picture_number, 1, sx as u16, sy as u16, slice_bytes);
            let pi = ParseInfo {
                parse_code: 0xEC,
                next_parse_offset: (ParseInfo::SIZE + payload.len()) as u32,
                previous_parse_offset: 0,
            };
            dec.on_data_fragment(&pi, &payload).expect("data");
            idx += 1;
        }
    }
    assert!(dec.assembler().fragmented_picture_done());

    let frag_pic = dec.finish().expect("finish");
    assert_eq!(frag_pic.y, round.reference.y, "Y mismatch");
    assert_eq!(frag_pic.u, round.reference.u, "U mismatch");
    assert_eq!(frag_pic.v, round.reference.v, "V mismatch");
}

/// LD path with all slices in one data fragment. LD adds the §14.5
/// trailing `dc_prediction(...)` kick (the HQ path doesn't); pinning
/// LD round-trip bit-exactness against the non-fragmented LD decoder
/// proves the kick is being applied in the right place.
#[test]
fn ld_q0_all_slices_in_one_fragment_bit_exact_vs_non_fragmented() {
    let w: u32 = 16;
    let h: u32 = 16;
    let y = smooth_plane(w as usize, h as usize, 1);
    let u = vec![128u8; (w * h / 4) as usize];
    let v = vec![128u8; (w * h / 4) as usize];

    let seq = make_minimal_sequence_ld(w, h, ChromaFormat::Yuv420);
    let params = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 2, 4, 4, 64);
    let stream = encode_single_ld_intra_stream(&seq, &params, 13, &y, &u, &v);
    let round = parse_and_decode(&stream);
    assert!(round.picture_parse_info.is_low_delay());

    let dissect_params = LdDissectParams {
        slices_x: params.slices_x,
        slices_y: params.slices_y,
        slice_bytes_numer: params.slice_bytes_numer,
        slice_bytes_denom: params.slice_bytes_denom,
    };
    let (picture_number, tp_range, slices) =
        dissect_ld_picture_payload(round.picture_payload, &dissect_params);
    assert_eq!(picture_number, 13);

    let tp_bytes = &round.picture_payload[tp_range.clone()];
    let setup_payload = setup_fragment_payload(picture_number, tp_bytes);

    let slice_region_start = slices[0].start;
    let slice_region_end = slices.last().unwrap().end;
    let all_slice_bytes = &round.picture_payload[slice_region_start..slice_region_end];
    let total_slices = (params.slices_x * params.slices_y) as u16;
    let data_payload = data_fragment_payload(picture_number, total_slices, 0, 0, all_slice_bytes);

    let setup_pi = ParseInfo {
        parse_code: 0xCC, // LD picture fragment.
        next_parse_offset: (ParseInfo::SIZE + setup_payload.len()) as u32,
        previous_parse_offset: 0,
    };
    let data_pi = ParseInfo {
        parse_code: 0xCC,
        next_parse_offset: (ParseInfo::SIZE + data_payload.len()) as u32,
        previous_parse_offset: setup_pi.next_parse_offset,
    };

    let mut dec = FragmentedPictureDecoder::new(&round.sequence);
    dec.on_setup_fragment(&setup_pi, &setup_payload)
        .expect("setup");
    let tp = dec.transform_parameters().expect("tp captured");
    assert_eq!(tp.profile, LowDelayProfile::LD);
    dec.on_data_fragment(&data_pi, &data_payload).expect("data");
    assert!(dec.assembler().fragmented_picture_done());

    let frag_pic = dec.finish().expect("finish");
    assert_eq!(frag_pic.y, round.reference.y, "Y mismatch (LD)");
    assert_eq!(frag_pic.u, round.reference.u, "U mismatch (LD)");
    assert_eq!(frag_pic.v, round.reference.v, "V mismatch (LD)");
}

/// LD with one fragment per slice. Forces every slice to round-trip
/// through `on_data_fragment` independently — the §14.5 DC-prediction
/// kick runs once at `finish()` regardless.
#[test]
fn ld_q0_one_data_fragment_per_slice_bit_exact_vs_non_fragmented() {
    let w: u32 = 16;
    let h: u32 = 16;
    let y = smooth_plane(w as usize, h as usize, 7);
    let u = vec![128u8; (w * h / 4) as usize];
    let v = vec![128u8; (w * h / 4) as usize];

    let seq = make_minimal_sequence_ld(w, h, ChromaFormat::Yuv420);
    let params = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 2, 4, 4, 64);
    let stream = encode_single_ld_intra_stream(&seq, &params, 99, &y, &u, &v);
    let round = parse_and_decode(&stream);

    let dissect_params = LdDissectParams {
        slices_x: params.slices_x,
        slices_y: params.slices_y,
        slice_bytes_numer: params.slice_bytes_numer,
        slice_bytes_denom: params.slice_bytes_denom,
    };
    let (picture_number, tp_range, slices) =
        dissect_ld_picture_payload(round.picture_payload, &dissect_params);

    let tp_bytes = &round.picture_payload[tp_range.clone()];
    let setup_payload = setup_fragment_payload(picture_number, tp_bytes);
    let setup_pi = ParseInfo {
        parse_code: 0xCC,
        next_parse_offset: (ParseInfo::SIZE + setup_payload.len()) as u32,
        previous_parse_offset: 0,
    };

    let mut dec = FragmentedPictureDecoder::new(&round.sequence);
    dec.on_setup_fragment(&setup_pi, &setup_payload)
        .expect("setup");
    let mut idx = 0;
    for sy in 0..params.slices_y {
        for sx in 0..params.slices_x {
            let slice_bytes = &round.picture_payload[slices[idx].clone()];
            let payload =
                data_fragment_payload(picture_number, 1, sx as u16, sy as u16, slice_bytes);
            let pi = ParseInfo {
                parse_code: 0xCC,
                next_parse_offset: (ParseInfo::SIZE + payload.len()) as u32,
                previous_parse_offset: 0,
            };
            dec.on_data_fragment(&pi, &payload).expect("data");
            idx += 1;
        }
    }
    let frag_pic = dec.finish().expect("finish");
    assert_eq!(frag_pic.y, round.reference.y, "Y mismatch (LD per-slice)");
    assert_eq!(frag_pic.u, round.reference.u, "U mismatch (LD per-slice)");
    assert_eq!(frag_pic.v, round.reference.v, "V mismatch (LD per-slice)");
}

/// `finish()` rejects when not every slice has been ingested. The
/// helper exposes the gap via the new `PictureIncomplete` variant
/// with the running counters in payload.
#[test]
fn finish_rejects_incomplete_picture() {
    let w: u32 = 16;
    let h: u32 = 16;
    let y = smooth_plane(w as usize, h as usize, 0);
    let u = smooth_plane((w / 2) as usize, (h / 2) as usize, 0);
    let v = smooth_plane((w / 2) as usize, (h / 2) as usize, 0);
    let seq = make_minimal_sequence(w, h, ChromaFormat::Yuv420);
    let params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 2);
    let stream = encode_single_hq_intra_stream(&seq, &params, 0, &y, &u, &v);
    let round = parse_and_decode(&stream);
    let dissect_params = HqDissectParams::symmetric(
        params.slices_x,
        params.slices_y,
        params.slice_prefix_bytes,
        params.slice_size_scaler,
    );
    let (picture_number, tp_range, slices) =
        dissect_hq_picture_payload(round.picture_payload, &dissect_params);
    let tp_bytes = &round.picture_payload[tp_range.clone()];
    let setup_payload = setup_fragment_payload(picture_number, tp_bytes);
    let setup_pi = ParseInfo {
        parse_code: 0xEC,
        next_parse_offset: (ParseInfo::SIZE + setup_payload.len()) as u32,
        previous_parse_offset: 0,
    };
    let mut dec = FragmentedPictureDecoder::new(&round.sequence);
    dec.on_setup_fragment(&setup_pi, &setup_payload)
        .expect("setup");

    // Send only the first slice — the others remain outstanding.
    let slice_bytes = &round.picture_payload[slices[0].clone()];
    let payload = data_fragment_payload(picture_number, 1, 0, 0, slice_bytes);
    let pi = ParseInfo {
        parse_code: 0xEC,
        next_parse_offset: (ParseInfo::SIZE + payload.len()) as u32,
        previous_parse_offset: 0,
    };
    dec.on_data_fragment(&pi, &payload).expect("data");
    assert!(!dec.assembler().fragmented_picture_done());

    let err = dec.finish().unwrap_err();
    match err {
        FragmentedPictureError::PictureIncomplete {
            slices_received,
            slices_expected,
        } => {
            assert_eq!(slices_received, 1);
            assert_eq!(slices_expected, params.slices_x * params.slices_y);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

/// A data fragment delivered before the setup fragment is rejected via
/// the `Assembler` error wrapper — the assembler's §14.1 sequencing
/// surfaces through the picture decoder unchanged.
#[test]
fn data_fragment_before_setup_rejected() {
    let seq = make_minimal_sequence(16, 16, ChromaFormat::Yuv420);
    let mut dec = FragmentedPictureDecoder::new(&seq);
    let payload = data_fragment_payload(0, 1, 0, 0, &[0u8; 4]);
    let pi = ParseInfo {
        parse_code: 0xEC,
        next_parse_offset: (ParseInfo::SIZE + payload.len()) as u32,
        previous_parse_offset: 0,
    };
    let err = dec.on_data_fragment(&pi, &payload).unwrap_err();
    assert!(matches!(err, FragmentedPictureError::NoActivePicture));
}

/// A setup fragment with a non-fragment parse code (e.g. `0xE8` HQ
/// picture, not `0xEC` HQ fragment) is rejected. The decoder only
/// accepts the two §10.5.2 Table 4 fragment codes.
#[test]
fn unsupported_parse_code_rejected() {
    let seq = make_minimal_sequence(16, 16, ChromaFormat::Yuv420);
    let mut dec = FragmentedPictureDecoder::new(&seq);
    let payload = setup_fragment_payload(0, &[]);
    let pi = ParseInfo {
        parse_code: 0x00,
        next_parse_offset: (ParseInfo::SIZE + payload.len()) as u32,
        previous_parse_offset: 0,
    };
    let err = dec.on_setup_fragment(&pi, &payload).unwrap_err();
    assert!(matches!(
        err,
        FragmentedPictureError::UnsupportedParseCode(0x00)
    ));
}

/// Two pictures back-to-back: the decoder is reusable. After
/// `finish()` returns the first picture, the next `on_setup_fragment`
/// is accepted and the second picture decodes to its own reference.
#[test]
fn two_consecutive_pictures_through_one_decoder() {
    let w: u32 = 16;
    let h: u32 = 16;
    let seq = make_minimal_sequence(w, h, ChromaFormat::Yuv420);
    let params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 2);
    let dissect_params = HqDissectParams::symmetric(
        params.slices_x,
        params.slices_y,
        params.slice_prefix_bytes,
        params.slice_size_scaler,
    );

    let mut dec = FragmentedPictureDecoder::new(&seq);

    for (picture_number, seed_y, seed_u, seed_v) in [(0u32, 0, 5, 11), (1u32, 17, 33, 41)] {
        let y = smooth_plane(w as usize, h as usize, seed_y);
        let u = smooth_plane((w / 2) as usize, (h / 2) as usize, seed_u);
        let v = smooth_plane((w / 2) as usize, (h / 2) as usize, seed_v);
        let stream = encode_single_hq_intra_stream(&seq, &params, picture_number, &y, &u, &v);
        let round = parse_and_decode(&stream);

        let (pn, tp_range, slices) =
            dissect_hq_picture_payload(round.picture_payload, &dissect_params);
        assert_eq!(pn, picture_number);
        let tp_bytes = &round.picture_payload[tp_range.clone()];
        let setup_payload = setup_fragment_payload(pn, tp_bytes);
        let slice_region_start = slices[0].start;
        let slice_region_end = slices.last().unwrap().end;
        let all_slice_bytes = &round.picture_payload[slice_region_start..slice_region_end];
        let data_payload = data_fragment_payload(
            pn,
            (params.slices_x * params.slices_y) as u16,
            0,
            0,
            all_slice_bytes,
        );

        let setup_pi = ParseInfo {
            parse_code: 0xEC,
            next_parse_offset: (ParseInfo::SIZE + setup_payload.len()) as u32,
            previous_parse_offset: 0,
        };
        let data_pi = ParseInfo {
            parse_code: 0xEC,
            next_parse_offset: (ParseInfo::SIZE + data_payload.len()) as u32,
            previous_parse_offset: 0,
        };
        dec.on_setup_fragment(&setup_pi, &setup_payload)
            .expect("setup");
        dec.on_data_fragment(&data_pi, &data_payload).expect("data");
        let frag_pic = dec.finish().expect("finish");
        assert_eq!(frag_pic.picture_number, picture_number);
        assert_eq!(frag_pic.y, round.reference.y);
        assert_eq!(frag_pic.u, round.reference.u);
        assert_eq!(frag_pic.v, round.reference.v);
    }
}

/// VC-2 v3 §12.4.4 **asymmetric (horizontal-only) transform** through
/// the fragmented-picture path. The §13.2.2 / §13.2.3 asymmetric
/// pyramid layout, the §15.4.1 + §15.4.2 `idwt_with_ho`, and the
/// §13.5.3 / §13.5.4 asymmetric slice unpack were wired into the
/// non-fragmented decoder in round 282 and into the
/// `FragmentedPictureDecoder` (`init_pyramid_ho` / `idwt_with_ho`) at
/// the same time, but no test had yet driven a genuinely asymmetric
/// (`dwt_depth_ho > 0`) picture through the §14 fragmented path. This
/// pins that the fragmented driver reproduces the non-fragmented
/// asymmetric reconstruction bit-exactly at qindex=0, for both the
/// all-slices-in-one-fragment shape and the one-fragment-per-slice
/// shape.
///
/// The encoder is configured via
/// [`EncoderParams::with_asymmetric_transform`] (which selects v3,
/// installs the §12.4.4 override and the §12.4.5.3 all-zero asymmetric
/// custom quant matrix), exactly as the round-282 non-fragmented
/// asymmetric round-trip tests in `encoder_roundtrip.rs`. The dissect
/// helper is told the same `(major_version, dwt_depth, ext)` so it
/// walks past the `extended_transform_parameters()` flag bits and the
/// asymmetric-shaped custom matrix to land on slice 0.
///
/// Material: `docs/video/vc2/vc2-specification.pdf` §12.4.4
/// (extended_transform_parameters), §12.4.5.3 (asymmetric quant
/// matrix), §13.2.2 / §13.2.3 (asymmetric pyramid / subband dims),
/// §13.5.4 (HQ slice), §14.2 / §14.4 / §14.5 (fragment header / state
/// machine / trailing kick), §15.4.1 / §15.4.2 (asymmetric IDWT).
#[test]
fn hq_q0_asymmetric_transform_bit_exact_vs_non_fragmented() {
    let w: u32 = 32;
    let h: u32 = 16;
    let wavelet = WaveletFilter::LeGall5_3;
    let wavelet_ho = WaveletFilter::Haar0; // wavelet_index_ho != wavelet_index
    let dwt_depth: u32 = 2;
    let dwt_depth_ho: u32 = 1;

    let y = smooth_plane(w as usize, h as usize, 2);
    let u = smooth_plane((w / 2) as usize, (h / 2) as usize, 7);
    let v = smooth_plane((w / 2) as usize, (h / 2) as usize, 13);

    let mut seq = make_minimal_sequence(w, h, ChromaFormat::Yuv420);
    seq.parse_parameters.version_major = 3;
    let params = EncoderParams::default_hq(wavelet, dwt_depth)
        .with_asymmetric_transform(wavelet_ho, dwt_depth_ho);

    let stream = encode_single_hq_intra_stream(&seq, &params, 21, &y, &u, &v);
    let round = parse_and_decode(&stream);

    let dissect_params = HqDissectParams {
        slices_x: params.slices_x,
        slices_y: params.slices_y,
        slice_prefix_bytes: params.slice_prefix_bytes,
        slice_size_scaler: params.slice_size_scaler,
        major_version: 3,
        ext: Some(HqExtParams {
            wavelet_index: wavelet_index(wavelet),
            wavelet_index_ho: wavelet_index(wavelet_ho),
            dwt_depth_ho,
        }),
    };
    let (picture_number, tp_range, slices) =
        dissect_hq_picture_payload(round.picture_payload, &dissect_params);
    assert_eq!(picture_number, 21);
    let total_slices = (params.slices_x * params.slices_y) as usize;
    assert_eq!(slices.len(), total_slices);

    let tp_bytes = &round.picture_payload[tp_range.clone()];

    // Shape A: all slices in one data fragment.
    {
        let setup_payload = setup_fragment_payload(picture_number, tp_bytes);
        let slice_region_start = slices[0].start;
        let slice_region_end = slices.last().unwrap().end;
        let all_slice_bytes = &round.picture_payload[slice_region_start..slice_region_end];
        let data_payload =
            data_fragment_payload(picture_number, total_slices as u16, 0, 0, all_slice_bytes);

        let setup_pi = ParseInfo {
            parse_code: 0xEC,
            next_parse_offset: (ParseInfo::SIZE + setup_payload.len()) as u32,
            previous_parse_offset: 0,
        };
        let data_pi = ParseInfo {
            parse_code: 0xEC,
            next_parse_offset: (ParseInfo::SIZE + data_payload.len()) as u32,
            previous_parse_offset: 0,
        };

        let mut dec = FragmentedPictureDecoder::new(&round.sequence);
        dec.on_setup_fragment(&setup_pi, &setup_payload)
            .expect("setup");
        let tp = dec.transform_parameters().expect("tp captured");
        assert_eq!(
            tp.dwt_depth_ho, dwt_depth_ho,
            "asymmetric ho depth captured"
        );
        dec.on_data_fragment(&data_pi, &data_payload).expect("data");
        assert!(dec.assembler().fragmented_picture_done());
        let frag_pic = dec.finish().expect("finish");
        assert_eq!(
            frag_pic.y, round.reference.y,
            "Y mismatch (single fragment)"
        );
        assert_eq!(
            frag_pic.u, round.reference.u,
            "U mismatch (single fragment)"
        );
        assert_eq!(
            frag_pic.v, round.reference.v,
            "V mismatch (single fragment)"
        );
    }

    // Shape B: one data fragment per slice.
    {
        let setup_payload = setup_fragment_payload(picture_number, tp_bytes);
        let setup_pi = ParseInfo {
            parse_code: 0xEC,
            next_parse_offset: (ParseInfo::SIZE + setup_payload.len()) as u32,
            previous_parse_offset: 0,
        };
        let mut dec = FragmentedPictureDecoder::new(&round.sequence);
        dec.on_setup_fragment(&setup_pi, &setup_payload)
            .expect("setup");

        let mut idx = 0;
        for sy in 0..params.slices_y {
            for sx in 0..params.slices_x {
                let slice_bytes = &round.picture_payload[slices[idx].clone()];
                let payload =
                    data_fragment_payload(picture_number, 1, sx as u16, sy as u16, slice_bytes);
                let pi = ParseInfo {
                    parse_code: 0xEC,
                    next_parse_offset: (ParseInfo::SIZE + payload.len()) as u32,
                    previous_parse_offset: 0,
                };
                dec.on_data_fragment(&pi, &payload).expect("data");
                idx += 1;
            }
        }
        assert!(dec.assembler().fragmented_picture_done());
        let frag_pic = dec.finish().expect("finish");
        assert_eq!(frag_pic.y, round.reference.y, "Y mismatch (per-slice)");
        assert_eq!(frag_pic.u, round.reference.u, "U mismatch (per-slice)");
        assert_eq!(frag_pic.v, round.reference.v, "V mismatch (per-slice)");
    }
}

/// Sanity: the imported `FragmentAssembler` / `FragmentEvent` /
/// `FragmentHeader` / `FragmentKind` / `AssemblerError` names are
/// the ones the `oxideav_dirac::fragment` module exports. (Catches
/// accidental rename in a future refactor that would silently move
/// this integration test off the public path.)
#[test]
#[allow(unused_variables)]
fn assembler_public_path_resolves() {
    let _a: FragmentAssembler = FragmentAssembler::new();
    let _e: Option<FragmentEvent> = None;
    let _h: Option<FragmentHeader> = None;
    let _k: Option<FragmentKind> = None;
    let _x: Option<AssemblerError> = None;
    let _tp: Option<TransformParameters> = None;
}

/// Round-417 deep colour through the §14 fragment path: a **16-bit**
/// full-range HQ picture (the `&[u16]` encoder + §10.3.8 index-0
/// custom signal range) split into one data fragment per slice must
/// reassemble to the same `DecodedPicture` the non-fragmented decode
/// yields — the fragment state machine carries slice *bytes* and is
/// depth-agnostic, but nothing proved that end-to-end above 8 bits
/// until now. Deep coefficients need a wider slice length-byte unit
/// (`slice_size_scaler = 32`), which also exercises the dissector's
/// scaler-based slice walk at a non-default scaler.
#[test]
fn hq_q0_16bit_one_data_fragment_per_slice_bit_exact_vs_non_fragmented() {
    use oxideav_dirac::encoder::{
        encode_single_hq_intra_stream_u16, make_minimal_sequence_with_signal_range,
    };
    use oxideav_dirac::video_format::SignalRange;

    let w: u32 = 16;
    let h: u32 = 16;
    let deep_plane = |w: usize, h: usize, seed: u64| -> Vec<u16> {
        let mut v = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let mix = x as u64 * 17 + y as u64 * 31 + seed * 7;
                v.push(((mix * 2654435761) % 65536) as u16);
            }
        }
        v
    };
    let y = deep_plane(w as usize, h as usize, 3);
    let u = deep_plane((w / 2) as usize, (h / 2) as usize, 9);
    let v = deep_plane((w / 2) as usize, (h / 2) as usize, 17);

    let seq = make_minimal_sequence_with_signal_range(
        w,
        h,
        ChromaFormat::Yuv420,
        SignalRange {
            luma_offset: 32768,
            luma_excursion: 65535,
            chroma_offset: 32768,
            chroma_excursion: 65535,
        },
    );
    assert_eq!(seq.luma_depth, 16);
    let mut params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 2);
    params.slice_size_scaler = 32;
    let stream = encode_single_hq_intra_stream_u16(&seq, &params, 43, &y, &u, &v);
    let round = parse_and_decode(&stream);

    let dissect_params = HqDissectParams::symmetric(
        params.slices_x,
        params.slices_y,
        params.slice_prefix_bytes,
        params.slice_size_scaler,
    );
    let (picture_number, tp_range, slices) =
        dissect_hq_picture_payload(round.picture_payload, &dissect_params);
    assert_eq!(picture_number, 43);

    let tp_bytes = &round.picture_payload[tp_range.clone()];
    let setup_payload = setup_fragment_payload(picture_number, tp_bytes);
    let setup_pi = ParseInfo {
        parse_code: 0xEC,
        next_parse_offset: (ParseInfo::SIZE + setup_payload.len()) as u32,
        previous_parse_offset: 0,
    };

    let mut dec = FragmentedPictureDecoder::new(&round.sequence);
    dec.on_setup_fragment(&setup_pi, &setup_payload)
        .expect("setup");

    let mut idx = 0;
    for sy in 0..params.slices_y {
        for sx in 0..params.slices_x {
            let slice_bytes = &round.picture_payload[slices[idx].clone()];
            let payload =
                data_fragment_payload(picture_number, 1, sx as u16, sy as u16, slice_bytes);
            let pi = ParseInfo {
                parse_code: 0xEC,
                next_parse_offset: (ParseInfo::SIZE + payload.len()) as u32,
                previous_parse_offset: 0,
            };
            dec.on_data_fragment(&pi, &payload).expect("data");
            idx += 1;
        }
    }
    assert!(dec.assembler().fragmented_picture_done());

    let frag_pic = dec.finish().expect("finish");
    assert_eq!(frag_pic.y, round.reference.y, "Y mismatch");
    assert_eq!(frag_pic.u, round.reference.u, "U mismatch");
    assert_eq!(frag_pic.v, round.reference.v, "V mismatch");
    // And the reference itself is the lossless source (§15.10 offset
    // already applied → samples are the unsigned input verbatim).
    let want_y: Vec<i32> = y.iter().map(|&s| s as i32).collect();
    assert_eq!(frag_pic.y, want_y, "16-bit Y not lossless");
}
