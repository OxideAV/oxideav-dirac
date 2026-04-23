//! Integration tests against a real Dirac elementary stream produced
//! by ffmpeg. These cover:
//!
//! * Walking the parse-info framing and spotting sequence headers.
//! * Parsing the sequence header and extracting usable frame
//!   dimensions + chroma format.
//! * Driving the public `Decoder` trait far enough to parse headers
//!   (pictures still return `Unsupported`).

use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecId, CodecParameters, Error, Packet, TimeBase};
use oxideav_dirac::parse_info::ParseInfo;
use oxideav_dirac::sequence::{parse_sequence_header, PictureCodingMode};
use oxideav_dirac::stream::DataUnitIter;
use oxideav_dirac::video_format::ChromaFormat;

const TINY: &[u8] = include_bytes!("fixtures/tiny_1f.drc");

#[test]
fn first_unit_is_sequence_header() {
    let first = DataUnitIter::new(TINY).next().expect("at least one unit");
    assert!(
        first.parse_info.is_seq_header(),
        "first data unit should be a sequence header, got parse code {:#x}",
        first.parse_info.parse_code
    );
}

#[test]
fn walker_finds_multiple_units() {
    let n = DataUnitIter::new(TINY).count();
    assert!(n >= 2, "expected at least seq_header + picture; got {n}");
}

#[test]
fn sequence_header_describes_a_64x64_stream() {
    // Our ffmpeg fixture was generated with `-f lavfi testsrc=size=64x64`
    // at 25fps, progressive frames. The sequence header should
    // reproduce those numbers, modulo any padding ffmpeg chose.
    let sh_unit = DataUnitIter::new(TINY)
        .find(|u| u.parse_info.is_seq_header())
        .expect("sequence header");
    let sh = parse_sequence_header(sh_unit.payload).expect("parse");
    // 64x64 is obviously not in the Annex C tables, so it must arrive
    // via the `custom_dimensions_flag` override on top of whichever
    // base format ffmpeg picked. Either way the final frame size
    // must be what we asked for.
    assert_eq!(
        (sh.video_params.frame_width, sh.video_params.frame_height),
        (64, 64)
    );
    assert_eq!(sh.picture_coding_mode, PictureCodingMode::Frames);
    // Chroma format is encoder's choice; just require it's one of the
    // three Dirac-supported sampling grids and that the computed
    // picture dimensions are consistent with the subsampling ratios.
    let cf = sh.video_params.chroma_format;
    assert!(matches!(
        cf,
        ChromaFormat::Yuv420 | ChromaFormat::Yuv422 | ChromaFormat::Yuv444
    ));
    assert_eq!(sh.chroma_width, sh.luma_width / cf.h_ratio());
    assert_eq!(sh.chroma_height, sh.luma_height / cf.v_ratio());
}

#[test]
fn parse_info_offsets_link_forward() {
    // The first parse info's next_parse_offset should land on another
    // BBCD prefix if non-zero.
    let pi = ParseInfo::parse(TINY, 0).expect("first parse info");
    if pi.next_parse_offset != 0 {
        let next = pi.next_parse_offset as usize;
        assert!(
            ParseInfo::has_prefix_at(TINY, next),
            "next_parse_offset {next} does not land on a BBCD prefix"
        );
    }
}

#[test]
fn decoder_accepts_packets_and_reports_unsupported_picture() {
    let mut reg = CodecRegistry::new();
    oxideav_dirac::register(&mut reg);
    let params = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.make_decoder(&params).expect("decoder");
    let packet = Packet::new(0, TimeBase::new(1, 25), TINY.to_vec());
    dec.send_packet(&packet).expect("send_packet");
    // Foundation pass: picture decode returns Unsupported. We're
    // validating that we get there cleanly, not a crash.
    match dec.receive_frame() {
        Err(Error::Unsupported(msg)) => {
            assert!(msg.contains("picture decode"), "unexpected msg: {msg}");
        }
        Err(Error::NeedMore) => {
            // Also acceptable — means we haven't hit a picture yet.
        }
        Ok(_) => panic!("picture decode unexpectedly succeeded"),
        Err(other) => panic!("unexpected error: {other:?}"),
    }
}
