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

/// Shell out to ffmpeg (if present) and emit a tiny 10-bit 4:2:0
/// VC-2 stream, then feed it through our decoder and check we now
/// produce a `Yuv420P10Le` frame rather than demoting to 8 bits. The
/// test is skipped cleanly when ffmpeg isn't available so CI doesn't
/// need it pre-installed.
fn ffmpeg_available() -> bool {
    std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Build a tiny VC-2 Dirac elementary stream with the given
/// `pix_fmt`. Returns true on success, false if ffmpeg isn't
/// available or the shell-out fails — callers skip cleanly in
/// that case.
fn build_vc2_stream(dest: &std::path::Path, pix_fmt: &str) -> bool {
    if !ffmpeg_available() {
        return false;
    }
    // `-f dirac` raw elementary stream; single-frame 64x64, shallow
    // wavelet depth so the slice buffers fit in the decoder's
    // default plumbing.
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:duration=0.04:rate=25",
            "-pix_fmt",
            pix_fmt,
            "-c:v",
            "vc2",
            "-wavelet_depth",
            "2",
            "-frames:v",
            "1",
            "-f",
            "dirac",
        ])
        .arg(dest)
        .status();
    matches!(status, Ok(s) if s.success())
}

fn build_10bit_stream(dest: &std::path::Path) -> bool {
    build_vc2_stream(dest, "yuv420p10le")
}

#[test]
fn ffmpeg_10bit_yuv420_produces_yuv420p10le_frame() {
    let tmp = std::env::temp_dir().join("oxideav_dirac_10bit_64x64.drc");
    if !build_10bit_stream(&tmp) {
        eprintln!("ffmpeg not available or failed to build 10-bit fixture; skipping");
        return;
    }
    let data = std::fs::read(&tmp).expect("read generated fixture");

    let mut reg = CodecRegistry::new();
    oxideav_dirac::register(&mut reg);
    let params = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.make_decoder(&params).expect("decoder");
    let packet = oxideav_core::Packet::new(0, oxideav_core::TimeBase::new(1, 25), data);
    dec.send_packet(&packet).expect("send_packet");
    let frame = dec.receive_frame().expect("receive_frame");
    match frame {
        Frame::Video(v) => {
            assert_eq!(v.width, 64);
            assert_eq!(v.height, 64);
            // The fixture is 4:2:0 + 10-bit — the decoder should emit
            // the proper LE 16-bit planar format, not 8-bit demotion.
            assert_eq!(v.format, PixelFormat::Yuv420P10Le);
            assert_eq!(v.planes.len(), 3);
            // Two bytes per luma sample, width=64 rows:
            assert_eq!(v.planes[0].stride, 128);
            assert_eq!(v.planes[0].data.len(), 64 * 64 * 2);
            // Chroma: 32x32 * 2 bytes per sample.
            assert_eq!(v.planes[1].stride, 64);
            assert_eq!(v.planes[1].data.len(), 32 * 32 * 2);

            // Decoder derives the timebase from the sequence header's
            // frame rate (§10.3.5); it must at minimum be sensible.
            let tb = v.time_base.as_rational();
            assert!(tb.num > 0 && tb.den > 0);

            // Sanity: the first sample should vary across the picture.
            // testsrc is a vivid pattern; a flat / garbled output
            // would sit near a single value.
            let mut min: u16 = u16::MAX;
            let mut max: u16 = 0;
            for chunk in v.planes[0].data.chunks_exact(2) {
                let s = u16::from_le_bytes([chunk[0], chunk[1]]);
                if s < min {
                    min = s;
                }
                if s > max {
                    max = s;
                }
            }
            assert!(
                max as i32 - min as i32 > 50,
                "expected a varied 10-bit Y plane (min={min}, max={max})"
            );
        }
        other => panic!("expected a video frame, got {other:?}"),
    }
}

/// 8-bit 4:2:2 ffmpeg-interop baseline. The checked-in `tiny_1f.drc`
/// fixture is 8-bit 4:4:4, and the round-1 highbitdepth test covers
/// 10-bit 4:2:0 — this test plugs the 8-bit 4:2:2 gap so all three
/// chroma geometries (§10.4) have baseline conformance coverage.
#[test]
fn ffmpeg_8bit_yuv422_produces_yuv422p_frame() {
    let tmp = std::env::temp_dir().join("oxideav_dirac_8bit_422_64x64.drc");
    if !build_vc2_stream(&tmp, "yuv422p") {
        eprintln!("ffmpeg not available or failed to build 8-bit 4:2:2 fixture; skipping");
        return;
    }
    let data = std::fs::read(&tmp).expect("read generated fixture");

    let mut reg = CodecRegistry::new();
    oxideav_dirac::register(&mut reg);
    let params = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.make_decoder(&params).expect("decoder");
    let packet = oxideav_core::Packet::new(0, oxideav_core::TimeBase::new(1, 25), data);
    dec.send_packet(&packet).expect("send_packet");
    let frame = dec.receive_frame().expect("receive_frame");
    match frame {
        Frame::Video(v) => {
            assert_eq!(v.width, 64);
            assert_eq!(v.height, 64);
            // 8-bit 4:2:2 — one byte per sample, horizontally
            // sub-sampled chroma (§10.4 Table 10.3).
            assert_eq!(v.format, PixelFormat::Yuv422P);
            assert_eq!(v.planes.len(), 3);
            assert_eq!(v.planes[0].stride, 64);
            assert_eq!(v.planes[0].data.len(), 64 * 64);
            // Chroma: 32x64, 1 byte per sample.
            assert_eq!(v.planes[1].stride, 32);
            assert_eq!(v.planes[1].data.len(), 32 * 64);
            assert_eq!(v.planes[2].stride, 32);
            assert_eq!(v.planes[2].data.len(), 32 * 64);

            let tb = v.time_base.as_rational();
            assert!(tb.num > 0 && tb.den > 0);

            // Same testsrc sanity check as the 8-bit 4:4:4 fixture:
            // a flat / garbled Y plane would sit near a single value.
            let y = &v.planes[0].data;
            let min = *y.iter().min().unwrap();
            let max = *y.iter().max().unwrap();
            assert!(
                max - min > 50,
                "expected a varied 8-bit Y plane (min={min}, max={max})"
            );
        }
        other => panic!("expected a video frame, got {other:?}"),
    }
}
