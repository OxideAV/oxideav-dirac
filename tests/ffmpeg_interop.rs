//! Integration tests against a real Dirac elementary stream produced
//! by ffmpeg. These cover:
//!
//! * Walking the parse-info framing and spotting sequence headers.
//! * Parsing the sequence header and extracting usable frame
//!   dimensions + chroma format.
//! * Driving the public `Decoder` trait all the way to a decoded
//!   picture (HQ-profile VC-2 intra pictures produce a `VideoFrame`).

use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, PixelFormat, TimeBase};
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
    let sh_unit = DataUnitIter::new(TINY)
        .find(|u| u.parse_info.is_seq_header())
        .expect("sequence header");
    let sh = parse_sequence_header(sh_unit.payload).expect("parse");
    assert_eq!(
        (sh.video_params.frame_width, sh.video_params.frame_height),
        (64, 64)
    );
    assert_eq!(sh.picture_coding_mode, PictureCodingMode::Frames);
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
    let pi = ParseInfo::parse(TINY, 0).expect("first parse info");
    if pi.next_parse_offset != 0 {
        let next = pi.next_parse_offset as usize;
        assert!(
            ParseInfo::has_prefix_at(TINY, next),
            "next_parse_offset {next} does not land on a BBCD prefix"
        );
    }
}

/// Drive the public `Decoder` trait: feed the whole stream, then pull
/// the first frame. With the HQ-profile intra path implemented, this
/// should produce a `VideoFrame` with 64x64 luma — the testsrc
/// pattern that ffmpeg's `lavfi testsrc=size=64x64` generated.
#[test]
fn decoder_produces_first_frame_from_hq_vc2_intra() {
    let mut reg = CodecRegistry::new();
    oxideav_dirac::register(&mut reg);
    let params = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.make_decoder(&params).expect("decoder");
    let packet = Packet::new(0, TimeBase::new(1, 25), TINY.to_vec());
    dec.send_packet(&packet).expect("send_packet");
    let frame = dec.receive_frame().expect("receive_frame");
    match frame {
        Frame::Video(v) => {
            assert_eq!(v.width, 64);
            assert_eq!(v.height, 64);
            // Fixture is yuv444p, 8-bit.
            assert_eq!(v.format, PixelFormat::Yuv444P);
            assert_eq!(v.planes.len(), 3);
            assert_eq!(v.planes[0].data.len(), 64 * 64);
            assert_eq!(v.planes[1].data.len(), 64 * 64);
            assert_eq!(v.planes[2].data.len(), 64 * 64);
            // testsrc's Y plane covers a wide range of intensities;
            // as a sanity check we expect both dark and bright pixels.
            let y = &v.planes[0].data;
            let min = *y.iter().min().unwrap();
            let max = *y.iter().max().unwrap();
            assert!(
                max - min > 50,
                "expected a varied Y plane (min={min}, max={max}) — \
                 looks like the decoded picture is flat / corrupt",
            );
        }
        other => panic!("expected a video frame, got {other:?}"),
    }
}
