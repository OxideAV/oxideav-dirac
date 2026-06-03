//! VC-2 v3 fragment-header parser integration tests
//! (SMPTE ST 2042-1:2022 §14.2).
//!
//! Round 223 — the crate gained `src/fragment.rs` carrying the §14.2
//! fragment header parser plus the §10.5.2 Table 5
//! `is_fragment(parse_code) := (parse_code & 0x0C) == 0x0C`
//! predicate. These tests exercise the parser through
//! [`crate::stream::DataUnitIter`] (the same byte-walker the decoder
//! front-end uses) so the wire-format contract is pinned from the
//! caller's vantage.
//!
//! All material consulted for this test file:
//! * `docs/video/vc2/vc2-specification.pdf` §10.5.2 (parse codes),
//!   §10.5.1 (parse-info header layout), §14.2 (fragment header
//!   layout), Annex A.3.4 (`read_uint_lit` is `read_nbits(8 * n)`).
//! * `docs/video/dirac/dirac-spec-latest.pdf` Table 9.1 (the BBC
//!   spec's `0xCC` / `0x0C` parse code definitions — relevant for the
//!   ambiguity note in `src/fragment.rs`).
//!
//! No external library source was consulted, no web search, no
//! third-party crate.

use oxideav_dirac::fragment::{
    AssemblerError, FragmentAssembler, FragmentEvent, FragmentHeader, FragmentKind,
};
use oxideav_dirac::parse_info::ParseInfo;
use oxideav_dirac::stream::DataUnitIter;

/// Build a 13-byte parse info header (§10.5.1: BBCD prefix +
/// parse_code byte + next/prev offsets as big-endian u32).
fn parse_info(parse_code: u8, next_off: u32, prev_off: u32) -> [u8; 13] {
    let mut b = [0u8; 13];
    b[..4].copy_from_slice(b"BBCD");
    b[4] = parse_code;
    b[5..9].copy_from_slice(&next_off.to_be_bytes());
    b[9..13].copy_from_slice(&prev_off.to_be_bytes());
    b
}

/// Build a §14.2 setup fragment payload: 4-byte picture number, 2-byte
/// `fragment_data_length`, 2-byte slice count (forced to zero). Total
/// 8 bytes — caller appends transform_parameters payload if needed.
fn setup_payload(picture_number: u32, fragment_data_length: u16) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&picture_number.to_be_bytes());
    p.extend_from_slice(&fragment_data_length.to_be_bytes());
    p.extend_from_slice(&0u16.to_be_bytes());
    p
}

/// Build a §14.2 data fragment payload: 4 + 2 + 2 + 2 + 2 = 12 bytes.
fn data_payload(
    picture_number: u32,
    fragment_data_length: u16,
    slice_count: u16,
    x_offset: u16,
    y_offset: u16,
) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&picture_number.to_be_bytes());
    p.extend_from_slice(&fragment_data_length.to_be_bytes());
    p.extend_from_slice(&slice_count.to_be_bytes());
    p.extend_from_slice(&x_offset.to_be_bytes());
    p.extend_from_slice(&y_offset.to_be_bytes());
    p
}

/// End-to-end: build a tiny three-unit stream containing a sequence
/// header followed by a setup fragment (parse code `0xEC`, HQ
/// fragment) and an EOS, walk it with `DataUnitIter`, and confirm the
/// fragment unit's `payload` decodes via
/// [`FragmentHeader::parse`]. Pins that the parser consumes exactly
/// the bytes the stream walker emits.
#[test]
fn stream_walker_yields_fragment_unit_decodes_setup() {
    let mut buf = Vec::new();
    // Sequence header (parse code 0x00) with an empty body — its
    // next_parse_offset is exactly the 13-byte header size.
    buf.extend_from_slice(&parse_info(0x00, 13, 0));
    // Setup fragment (parse code 0xEC = HQ picture fragment in v3).
    let frag = setup_payload(42, 0x1234);
    let frag_unit_len = (ParseInfo::SIZE + frag.len()) as u32;
    buf.extend_from_slice(&parse_info(0xEC, frag_unit_len, 13));
    buf.extend_from_slice(&frag);
    // End of sequence.
    buf.extend_from_slice(&parse_info(0x10, 0, frag_unit_len));

    let units: Vec<_> = DataUnitIter::new(&buf).collect();
    assert_eq!(units.len(), 3, "three units (seq hdr, fragment, EOS)");
    assert!(units[0].parse_info.is_seq_header());

    let frag_unit = &units[1];
    assert!(
        frag_unit.parse_info.is_fragment_parse_code(),
        "0xEC matches the §10.5.2 Table 5 fragment predicate"
    );
    assert_eq!(frag_unit.parse_info.parse_code, 0xEC);

    let parsed = FragmentHeader::parse(frag_unit.payload).expect("setup fragment");
    assert_eq!(parsed.picture_number, 42);
    assert_eq!(parsed.fragment_data_length, 0x1234);
    assert_eq!(parsed.kind, FragmentKind::Setup);
    assert_eq!(parsed.header_size(), 8);
    assert_eq!(frag_unit.payload.len(), parsed.header_size());

    assert!(units[2].parse_info.is_end_of_sequence());
}

/// Same end-to-end shape but with a §14.2 data fragment carrying a
/// non-zero slice count plus the raster `(x, y)` offset of its first
/// slice. Pins the 12-byte data-header parse.
#[test]
fn stream_walker_yields_fragment_unit_decodes_data() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&parse_info(0x00, 13, 0));

    let frag = data_payload(42, 0xABCD, 5, 7, 3);
    let frag_unit_len = (ParseInfo::SIZE + frag.len()) as u32;
    // 0xCC is LD picture fragment in v3.
    buf.extend_from_slice(&parse_info(0xCC, frag_unit_len, 13));
    buf.extend_from_slice(&frag);

    buf.extend_from_slice(&parse_info(0x10, 0, frag_unit_len));

    let units: Vec<_> = DataUnitIter::new(&buf).collect();
    assert_eq!(units.len(), 3);

    let frag_unit = &units[1];
    assert!(frag_unit.parse_info.is_fragment_parse_code());
    assert_eq!(frag_unit.parse_info.parse_code, 0xCC);

    let parsed = FragmentHeader::parse(frag_unit.payload).expect("data fragment");
    assert_eq!(parsed.picture_number, 42);
    assert_eq!(parsed.fragment_data_length, 0xABCD);
    assert_eq!(
        parsed.kind,
        FragmentKind::Data {
            slice_count: 5,
            x_offset: 7,
            y_offset: 3,
        }
    );
    assert_eq!(parsed.header_size(), 12);
    assert_eq!(frag_unit.payload.len(), parsed.header_size());
}

/// §14.1: each data fragment carries at least one slice. A setup
/// fragment's slice count is zero, and the parser correctly assigns
/// the `Setup` variant on that basis. Build two consecutive fragment
/// units (setup + data), walk them through the stream iterator, and
/// confirm both decode and that their picture numbers match (the
/// spec requires the data fragment to carry the same picture number
/// as its associated setup fragment).
#[test]
fn setup_then_data_fragments_share_picture_number() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&parse_info(0x00, 13, 0));

    let pic_num = 1234;
    let setup = setup_payload(pic_num, 0);
    let setup_unit_len = (ParseInfo::SIZE + setup.len()) as u32;
    buf.extend_from_slice(&parse_info(0xCC, setup_unit_len, 13));
    buf.extend_from_slice(&setup);

    let data = data_payload(pic_num, 0, 1, 0, 0);
    let data_unit_len = (ParseInfo::SIZE + data.len()) as u32;
    buf.extend_from_slice(&parse_info(0xCC, data_unit_len, setup_unit_len));
    buf.extend_from_slice(&data);

    buf.extend_from_slice(&parse_info(0x10, 0, data_unit_len));

    let units: Vec<_> = DataUnitIter::new(&buf).collect();
    assert_eq!(units.len(), 4);

    let setup_hdr = FragmentHeader::parse(units[1].payload).expect("setup");
    let data_hdr = FragmentHeader::parse(units[2].payload).expect("data");
    assert_eq!(setup_hdr.kind, FragmentKind::Setup);
    assert!(matches!(data_hdr.kind, FragmentKind::Data { .. }));
    assert_eq!(setup_hdr.picture_number, pic_num);
    assert_eq!(data_hdr.picture_number, pic_num);
}

/// Round-229 §14.3 / §14.4 driver: walk a synthetic stream
/// `[seq_hdr][0xCC setup][0xCC data 2 slices @(0,0)][0xCC data 2
/// slices @(2,0)][EOS]` through `DataUnitIter`, feed each fragment
/// header to a `FragmentAssembler`, and confirm:
/// * the setup event fires `FragmentEvent::SetupAccepted`,
/// * each data event emits the raster `(slice_x, slice_y)` per
///   §14.4,
/// * the final data event fires `picture_done == true`,
/// * the assembler's `fragmented_picture_done()` flips to true
///   after the final ingest.
///
/// Picture geometry: `slices_x = 4`, `slices_y = 1`, so all 4
/// slices live on row 0 in raster order.
#[test]
fn assembler_drives_through_stream_walker_setup_plus_two_data() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&parse_info(0x00, 13, 0));

    let pic_num = 11;
    let setup = setup_payload(pic_num, 0);
    let setup_unit_len = (ParseInfo::SIZE + setup.len()) as u32;
    buf.extend_from_slice(&parse_info(0xCC, setup_unit_len, 13));
    buf.extend_from_slice(&setup);

    let data1 = data_payload(pic_num, 0, 2, 0, 0);
    let data1_unit_len = (ParseInfo::SIZE + data1.len()) as u32;
    buf.extend_from_slice(&parse_info(0xCC, data1_unit_len, setup_unit_len));
    buf.extend_from_slice(&data1);

    let data2 = data_payload(pic_num, 0, 2, 2, 0);
    let data2_unit_len = (ParseInfo::SIZE + data2.len()) as u32;
    buf.extend_from_slice(&parse_info(0xCC, data2_unit_len, data1_unit_len));
    buf.extend_from_slice(&data2);

    buf.extend_from_slice(&parse_info(0x10, 0, data2_unit_len));

    let units: Vec<_> = DataUnitIter::new(&buf).collect();
    assert_eq!(units.len(), 5, "seq_hdr + setup + 2 data + EOS");

    let mut asm = FragmentAssembler::new();
    let setup_event = {
        let hdr = FragmentHeader::parse(units[1].payload).expect("setup");
        asm.on_setup_fragment(&hdr, units[1].parse_info.parse_code)
            .expect("setup accepted")
    };
    assert_eq!(setup_event, FragmentEvent::SetupAccepted);
    asm.on_transform_parameters(4, 1, 0)
        .expect("4x1 slice grid accepted");

    let data1_event = {
        let hdr = FragmentHeader::parse(units[2].payload).expect("data1");
        asm.on_data_fragment(&hdr, units[2].parse_info.parse_code)
            .expect("data1 accepted")
    };
    match data1_event {
        FragmentEvent::DataSlices {
            coords,
            picture_done,
        } => {
            assert_eq!(coords, vec![(0, 0), (1, 0)]);
            assert!(!picture_done, "still 2 slices to go");
        }
        _ => panic!("expected DataSlices"),
    }
    assert!(!asm.fragmented_picture_done());

    let data2_event = {
        let hdr = FragmentHeader::parse(units[3].payload).expect("data2");
        asm.on_data_fragment(&hdr, units[3].parse_info.parse_code)
            .expect("data2 accepted")
    };
    match data2_event {
        FragmentEvent::DataSlices {
            coords,
            picture_done,
        } => {
            assert_eq!(coords, vec![(2, 0), (3, 0)]);
            assert!(picture_done, "final data fragment completes picture");
        }
        _ => panic!("expected DataSlices"),
    }
    assert!(asm.fragmented_picture_done());
    assert!(asm.using_dc_prediction(), "0xCC LD path → DC pred kick");
    assert_eq!(asm.dwt_depth_ho(), 0, "symmetric transform default");
    assert_eq!(asm.picture_number(), pic_num);

    assert!(units[4].parse_info.is_end_of_sequence());
}

/// §14.1 sequencing constraint: a second setup fragment arriving
/// before the previous fragmented picture completes is rejected.
/// Drive a stream `[seq_hdr][0xCC setup pic=0][0xCC setup pic=1]
/// [EOS]` through the assembler — the second setup transition must
/// surface `SetupBeforePreviousPictureComplete`. Pinned via the
/// stream walker so this matches the real ingest path.
#[test]
fn assembler_rejects_consecutive_setup_fragments_through_stream() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&parse_info(0x00, 13, 0));

    let setup0 = setup_payload(0, 0);
    let setup0_unit_len = (ParseInfo::SIZE + setup0.len()) as u32;
    buf.extend_from_slice(&parse_info(0xCC, setup0_unit_len, 13));
    buf.extend_from_slice(&setup0);

    let setup1 = setup_payload(1, 0);
    let setup1_unit_len = (ParseInfo::SIZE + setup1.len()) as u32;
    buf.extend_from_slice(&parse_info(0xCC, setup1_unit_len, setup0_unit_len));
    buf.extend_from_slice(&setup1);

    buf.extend_from_slice(&parse_info(0x10, 0, setup1_unit_len));

    let units: Vec<_> = DataUnitIter::new(&buf).collect();
    let mut asm = FragmentAssembler::new();
    let h0 = FragmentHeader::parse(units[1].payload).unwrap();
    asm.on_setup_fragment(&h0, units[1].parse_info.parse_code)
        .unwrap();
    asm.on_transform_parameters(2, 1, 0).unwrap();
    // Second setup before any data fragment → §14.1 violation.
    let h1 = FragmentHeader::parse(units[2].payload).unwrap();
    let err = asm
        .on_setup_fragment(&h1, units[2].parse_info.parse_code)
        .unwrap_err();
    assert_eq!(err, AssemblerError::SetupBeforePreviousPictureComplete);
}
