//! Round-417 — decoder output-format surface tests.
//!
//! The decoder front-end picks the oxideav-core `PixelFormat` from the
//! sequence header's §10.3.3 chroma format + §10.5.2 luma depth
//! (`decoder::output_format_for`). This suite pins the two public ways
//! that choice is surfaced to callers:
//!
//! * [`DiracDecoder::output_pixel_format`] — the format the decoder
//!   will emit for the current sequence, queryable after the first
//!   `send_packet`.
//! * [`Decoder::receive_arena_frame`] — the arena-backed frame path,
//!   whose `FrameHeader` must carry the real picture width / height
//!   and the true output format (the trait-default implementation can
//!   only guess from plane counts).
//!
//! All streams are self-encoded through the crate's own HQ intra
//! encoder (8-bit `&[u8]` and deep `&[u16]` entry points), so the test
//! is fully self-contained. Wall: BBC Dirac spec §10.3.3 / §10.3.8 /
//! §10.5 from `docs/video/dirac/dirac-spec-latest.pdf`; no external
//! source, no web.

use oxideav_core::{CodecId, Decoder, Frame, Packet, PixelFormat, TimeBase};
use oxideav_dirac::decoder::DiracDecoder;
use oxideav_dirac::encoder::{
    encode_single_hq_intra_stream, encode_single_hq_intra_stream_u16, make_minimal_sequence,
    make_minimal_sequence_with_signal_range, EncoderParams,
};
use oxideav_dirac::video_format::{ChromaFormat, SignalRange};
use oxideav_dirac::wavelet::WaveletFilter;

fn chroma_dims(w: u32, h: u32, chroma: ChromaFormat) -> (u32, u32) {
    match chroma {
        ChromaFormat::Yuv420 => (w / 2, h / 2),
        ChromaFormat::Yuv422 => (w / 2, h),
        ChromaFormat::Yuv444 => (w, h),
    }
}

/// Deterministic ramp covering `[0, 2^depth - 1]`.
fn ramp_u16(w: usize, h: usize, depth: u32, seed: u32) -> Vec<u16> {
    let max = (1u64 << depth) - 1;
    (0..w * h)
        .map(|i| {
            let x = (i % w) as u64;
            let y = (i / w) as u64;
            ((x * 131 + y * 17 + seed as u64 * 7) % (max + 1)) as u16
        })
        .collect()
}

fn ramp_u8(w: usize, h: usize, seed: u32) -> Vec<u8> {
    (0..w * h)
        .map(|i| ((i as u32).wrapping_mul(31).wrapping_add(seed * 7) % 256) as u8)
        .collect()
}

/// Encode one 8-bit HQ intra picture at the given chroma sampling.
fn encode_8bit(chroma: ChromaFormat) -> Vec<u8> {
    let (w, h) = (64u32, 64u32);
    let (cw, ch) = chroma_dims(w, h, chroma);
    let seq = make_minimal_sequence(w, h, chroma);
    let params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let y = ramp_u8(w as usize, h as usize, 1);
    let u = ramp_u8(cw as usize, ch as usize, 2);
    let v = ramp_u8(cw as usize, ch as usize, 3);
    encode_single_hq_intra_stream(&seq, &params, 0, &y, &u, &v)
}

/// Encode one deep HQ intra picture with a full-range custom signal
/// range at the given depth (§10.3.8 `index == 0`, offset `2^(d-1)`,
/// excursion `2^d - 1`).
fn encode_deep(chroma: ChromaFormat, depth: u32) -> Vec<u8> {
    let (w, h) = (64u32, 64u32);
    let (cw, ch) = chroma_dims(w, h, chroma);
    let sr = SignalRange {
        luma_offset: 1 << (depth - 1),
        luma_excursion: ((1u64 << depth) - 1) as u32,
        chroma_offset: 1 << (depth - 1),
        chroma_excursion: ((1u64 << depth) - 1) as u32,
    };
    let seq = make_minimal_sequence_with_signal_range(w, h, chroma, sr);
    let mut params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    // Deep coefficients overflow a scaler-1 HQ slice length byte;
    // widen the length-byte unit (same trick the 10-bit roundtrip
    // suite uses).
    params.slice_size_scaler = 16;
    let y = ramp_u16(w as usize, h as usize, depth, 1);
    let u = ramp_u16(cw as usize, ch as usize, depth, 2);
    let v = ramp_u16(cw as usize, ch as usize, depth, 3);
    encode_single_hq_intra_stream_u16(&seq, &params, 0, &y, &u, &v)
}

/// `output_pixel_format` is `None` before any sequence header and
/// reports the depth-and-chroma-derived format immediately after
/// `send_packet` — before the first frame is pulled.
#[test]
fn output_pixel_format_tracks_sequence_header() {
    let mut dec = DiracDecoder::new(CodecId::new("dirac"));
    assert_eq!(dec.output_pixel_format(), None);
    let stream = encode_8bit(ChromaFormat::Yuv420);
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), stream))
        .expect("send");
    assert_eq!(dec.output_pixel_format(), Some(PixelFormat::Yuv420P));
}

/// The format matrix across self-encodable chroma × depth pairs, via
/// the live decoder rather than the pure helper.
#[test]
fn output_pixel_format_matrix_via_live_streams() {
    let cases: [(ChromaFormat, u32, PixelFormat); 6] = [
        (ChromaFormat::Yuv422, 10, PixelFormat::Yuv422P10Le),
        (ChromaFormat::Yuv444, 12, PixelFormat::Yuv444P12Le),
        (ChromaFormat::Yuv420, 13, PixelFormat::Yuv420P16Le),
        (ChromaFormat::Yuv422, 14, PixelFormat::Yuv422P16Le),
        (ChromaFormat::Yuv444, 15, PixelFormat::Yuv444P16Le),
        (ChromaFormat::Yuv420, 16, PixelFormat::Yuv420P16Le),
    ];
    for (chroma, depth, want) in cases {
        let stream = encode_deep(chroma, depth);
        let mut dec = DiracDecoder::new(CodecId::new("dirac"));
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), stream))
            .expect("send");
        assert_eq!(
            dec.output_pixel_format(),
            Some(want),
            "chroma {chroma:?} depth {depth}"
        );
    }
}

/// `receive_arena_frame` must produce a header with the real picture
/// geometry and format for an 8-bit stream (64×64 4:2:2 → Yuv422P),
/// with per-plane byte counts matching the packed layout.
#[test]
fn arena_frame_header_is_exact_for_8bit_422() {
    let stream = encode_8bit(ChromaFormat::Yuv422);
    let mut dec = DiracDecoder::new(CodecId::new("dirac"));
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), stream))
        .expect("send");
    let af = dec.receive_arena_frame().expect("arena frame");
    let hdr = af.header();
    assert_eq!((hdr.width, hdr.height), (64, 64));
    assert_eq!(hdr.pixel_format, PixelFormat::Yuv422P);
    assert_eq!(af.plane_count(), 3);
    assert_eq!(af.plane(0).expect("Y").len(), 64 * 64);
    assert_eq!(af.plane(1).expect("U").len(), 32 * 64);
    assert_eq!(af.plane(2).expect("V").len(), 32 * 64);
}

/// The deep-colour arena path: a 16-bit 4:2:0 stream must surface
/// `Yuv420P16Le` with two bytes per sample, and the plane bytes must
/// match the `receive_frame` copy bit-for-bit (same packing code).
#[test]
fn arena_frame_header_is_exact_for_16bit_420() {
    let stream = encode_deep(ChromaFormat::Yuv420, 16);

    let mut dec = DiracDecoder::new(CodecId::new("dirac"));
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), stream.clone()))
        .expect("send");
    let af = dec.receive_arena_frame().expect("arena frame");
    let hdr = af.header();
    assert_eq!((hdr.width, hdr.height), (64, 64));
    assert_eq!(hdr.pixel_format, PixelFormat::Yuv420P16Le);
    assert_eq!(af.plane(0).expect("Y").len(), 64 * 64 * 2);
    assert_eq!(af.plane(1).expect("U").len(), 32 * 32 * 2);

    let mut dec2 = DiracDecoder::new(CodecId::new("dirac"));
    dec2.send_packet(&Packet::new(0, TimeBase::new(1, 25), stream))
        .expect("send");
    let vf = match dec2.receive_frame().expect("frame") {
        Frame::Video(v) => v,
        other => panic!("expected video frame, got {other:?}"),
    };
    for i in 0..3 {
        assert_eq!(
            af.plane(i).expect("plane"),
            &vf.planes[i].data[..],
            "plane {i} bytes must match the receive_frame packing"
        );
    }
}
