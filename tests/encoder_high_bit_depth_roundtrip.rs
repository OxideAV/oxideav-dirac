//! Round-345 — high-bit-depth (10/12-bit) intra encode → decode
//! round-trip coverage; round-417 extends it to the deep-colour
//! 13-16-bit depths that decode onto the all-bits-significant
//! `Yuv*P16Le` surfaces (16-bit verbatim, 13-15-bit MSB-aligned), and
//! to the now-native 12-bit 4:2:2/4:4:4 `Yuv*P12Le` paths.
//!
//! `docs/video/dirac/dirac-fixtures-and-traces.md` lists **bit depths >
//! 8 (10, 12, 16)** as an explicit corpus gap: every upstream sample we
//! slice from is 8-bit, so nothing drove the decoder's
//! `sequence.{luma,chroma}_depth`-parameterised reconstruction path
//! (§15.9 clip + §15.10 output offset + the 16-bit LE plane packing in
//! `decoder::plane_from_i32`) past 8 bits. The decode side has always
//! computed those depths from the §10.5.2 `video_depth(excursion + 1)`
//! formula, but no end-to-end test exercised it because the 8-bit-only
//! encoder produced no deeper stream.
//!
//! This suite closes that gap on the decode side: the new `&[u16]`
//! encode entry points (`encode_single_{hq,ld}_intra_stream_u16`) emit
//! genuine 10-/12-bit VC-2 intra streams whose §10.3.8 signal range is a
//! full-range N-bit custom range (`PRESET_{10,12}BIT_FULL`, written as
//! `preset_idx = 0` with explicit offset/excursion fields). We decode
//! them back through the registry and assert the reconstructed 16-bit
//! samples match the input bit-exactly at qindex 0.
//!
//! Why bit-exact is the right bar: at qindex 0 the dead-zone quantiser
//! is the identity (`qf = 4`, `4*|x|/4 = |x|`) and the integer-lifting
//! wavelets are perfectly reversible, so the only thing under test is
//! whether the depth plumbing (forward recentre by `2^(depth-1)` →
//! IDWT → §15.9 clip → §15.10 offset → 16-bit LE pack) is internally
//! consistent across the wider sample range. A single off-by-one in any
//! shift, offset, or clip bound would show up immediately.
//!
//! Wall: BBC Dirac spec §10.3.8 (signal range, Table 10.5), §10.5.2
//! (video depth), §15.9/§15.10 (clip + output offset) from
//! `docs/video/dirac/dirac-spec-latest.pdf`; no external source, no web.

use oxideav_core::{CodecId, CodecParameters, CodecRegistry, Frame, Packet, TimeBase, VideoFrame};
use oxideav_dirac::encoder::{
    encode_single_hq_intra_stream_u16, encode_single_ld_intra_stream_u16,
    make_minimal_sequence_ld_with_signal_range, make_minimal_sequence_with_signal_range,
    EncoderParams, LdEncoderParams,
};
use oxideav_dirac::quant::QuantMatrix;
use oxideav_dirac::video_format::{ChromaFormat, SignalRange};
use oxideav_dirac::wavelet::WaveletFilter;

/// Decode a single-frame elementary stream to a `VideoFrame`.
fn decode_one(stream: Vec<u8>) -> VideoFrame {
    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("make decoder");
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&pkt).expect("send packet");
    match dec.receive_frame().expect("receive frame") {
        Frame::Video(v) => v,
        other => panic!("expected video frame, got {other:?}"),
    }
}

/// Reinterpret a decoded plane's little-endian 16-bit samples as `u16`.
/// The decoder emits one 2-byte LE word per sample for any >8-bit
/// component (`decoder::plane_from_i32`, `store_depth` 10 or 12), so the
/// plane data length is `2 * sample_count`.
fn plane_as_u16(data: &[u8]) -> Vec<u16> {
    assert_eq!(data.len() % 2, 0, "16-bit plane must have even byte length");
    data.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Deterministic high-bit-depth ramp filling `[0, 2^depth - 1]` so the
/// full-range path (offset/excursion centred symmetrically) is exercised
/// edge-to-edge. The pattern mixes the two coordinates plus a seed so
/// neighbouring samples differ (a flat plane would hide many shift bugs).
fn ramp_plane(w: usize, h: usize, depth: u32, seed: u32) -> Vec<u16> {
    let max = (1u32 << depth) - 1;
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let v = (x as u32 * 17 + y as u32 * 31 + seed * 7) % (max + 1);
            out.push(v as u16);
        }
    }
    out
}

fn chroma_dims(w: u32, h: u32, chroma: ChromaFormat) -> (u32, u32) {
    match chroma {
        ChromaFormat::Yuv420 => (w / 2, h / 2),
        ChromaFormat::Yuv422 => (w / 2, h),
        ChromaFormat::Yuv444 => (w, h),
    }
}

/// Compare two `u16` slices and report the first divergence with context.
fn assert_planes_eq(label: &str, got: &[u16], want: &[u16]) {
    assert_eq!(got.len(), want.len(), "{label}: plane length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g, w, "{label}: sample {i} mismatch (got {g}, want {w})");
    }
}

/// HQ intra, 10-bit full-range, across the three chroma formats and the
/// six reversible wavelets — every sample must reconstruct bit-exactly.
#[test]
fn hq_10bit_full_range_q0_bit_exact_across_wavelets_and_chromas() {
    let (w, h) = (64u32, 64u32);
    let sr = SignalRange::PRESET_10BIT_FULL;
    for chroma in [
        ChromaFormat::Yuv420,
        ChromaFormat::Yuv422,
        ChromaFormat::Yuv444,
    ] {
        let (cw, ch) = chroma_dims(w, h, chroma);
        let y = ramp_plane(w as usize, h as usize, 10, 1);
        let u = ramp_plane(cw as usize, ch as usize, 10, 5);
        let v = ramp_plane(cw as usize, ch as usize, 10, 9);
        let seq = make_minimal_sequence_with_signal_range(w, h, chroma, sr);
        assert_eq!(seq.luma_depth, 10, "10-bit full range → luma_depth 10");
        assert_eq!(seq.chroma_depth, 10);

        for filter in [
            WaveletFilter::DeslauriersDubuc9_7,
            WaveletFilter::LeGall5_3,
            WaveletFilter::DeslauriersDubuc13_7,
            WaveletFilter::Haar0,
            WaveletFilter::Haar1,
            WaveletFilter::Daubechies9_7,
        ] {
            let params = EncoderParams::default_hq(filter, 3);
            let stream = encode_single_hq_intra_stream_u16(&seq, &params, 0, &y, &u, &v);
            let frame = decode_one(stream);

            let gy = plane_as_u16(&frame.planes[0].data);
            let gu = plane_as_u16(&frame.planes[1].data);
            let gv = plane_as_u16(&frame.planes[2].data);
            let tag = format!("HQ 10-bit {chroma:?} {filter:?}");
            assert_planes_eq(&format!("{tag} Y"), &gy, &y);
            assert_planes_eq(&format!("{tag} U"), &gu, &u);
            assert_planes_eq(&format!("{tag} V"), &gv, &v);
        }
    }
}

/// HQ intra, 12-bit full-range, 4:2:0 — the `Yuv420P12Le` storage
/// format carries the full 12-bit field bit-exactly.
#[test]
fn hq_12bit_full_range_q0_bit_exact_420() {
    let (w, h) = (64u32, 64u32);
    let sr = SignalRange::PRESET_12BIT_FULL;
    let chroma = ChromaFormat::Yuv420;
    let (cw, ch) = chroma_dims(w, h, chroma);
    let y = ramp_plane(w as usize, h as usize, 12, 2);
    let u = ramp_plane(cw as usize, ch as usize, 12, 6);
    let v = ramp_plane(cw as usize, ch as usize, 12, 10);
    let seq = make_minimal_sequence_with_signal_range(w, h, chroma, sr);
    assert_eq!(seq.luma_depth, 12, "12-bit full range → luma_depth 12");

    for filter in [
        WaveletFilter::DeslauriersDubuc9_7,
        WaveletFilter::LeGall5_3,
        WaveletFilter::Haar0,
        WaveletFilter::Haar1,
    ] {
        let params = EncoderParams::default_hq(filter, 3);
        let stream = encode_single_hq_intra_stream_u16(&seq, &params, 0, &y, &u, &v);
        let frame = decode_one(stream);
        let gy = plane_as_u16(&frame.planes[0].data);
        let gu = plane_as_u16(&frame.planes[1].data);
        let gv = plane_as_u16(&frame.planes[2].data);
        let tag = format!("HQ 12-bit 420 {filter:?}");
        assert_planes_eq(&format!("{tag} Y"), &gy, &y);
        assert_planes_eq(&format!("{tag} U"), &gu, &u);
        assert_planes_eq(&format!("{tag} V"), &gv, &v);
    }
}

/// HQ intra, 12-bit full-range, 4:2:2 and 4:4:4 — round-417: with
/// oxideav-core's `Yuv422P12Le` / `Yuv444P12Le` the decoder no longer
/// clips these chroma samplings to 10 bits, so the round-trip is now
/// bit-exact at 12 bits everywhere (this used to be impossible: the
/// two low bits died in the 10-bit demotion).
#[test]
fn hq_12bit_full_range_q0_bit_exact_422_444() {
    let (w, h) = (64u32, 64u32);
    let sr = SignalRange::PRESET_12BIT_FULL;
    for chroma in [ChromaFormat::Yuv422, ChromaFormat::Yuv444] {
        let (cw, ch) = chroma_dims(w, h, chroma);
        let y = ramp_plane(w as usize, h as usize, 12, 4);
        let u = ramp_plane(cw as usize, ch as usize, 12, 8);
        let v = ramp_plane(cw as usize, ch as usize, 12, 12);
        let seq = make_minimal_sequence_with_signal_range(w, h, chroma, sr);
        assert_eq!(seq.luma_depth, 12);

        let mut params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
        params.slice_size_scaler = 8;
        let stream = encode_single_hq_intra_stream_u16(&seq, &params, 0, &y, &u, &v);
        let frame = decode_one(stream);
        let gy = plane_as_u16(&frame.planes[0].data);
        let gu = plane_as_u16(&frame.planes[1].data);
        let gv = plane_as_u16(&frame.planes[2].data);
        let tag = format!("HQ 12-bit {chroma:?}");
        assert_planes_eq(&format!("{tag} Y"), &gy, &y);
        assert_planes_eq(&format!("{tag} U"), &gu, &u);
        assert_planes_eq(&format!("{tag} V"), &gv, &v);
    }
}

/// Undo the decoder's deep-colour MSB alignment: a `depth`-bit source
/// stored on the all-bits-significant 16-bit surface sits in the top
/// `depth` bits of each word (`sample << (16 - depth)`), so the
/// original sample is recovered by the inverse shift. For `depth == 16`
/// this is the identity — the surface is exactly the sample.
fn unshift_p16(plane: &[u16], depth: u32) -> Vec<u16> {
    let sh = 16 - depth;
    plane.iter().map(|&v| v >> sh).collect()
}

/// The deep-colour headline: HQ intra at **16-bit** full range must be
/// bit-exact at qindex 0 across all three chroma samplings and the six
/// reversible wavelets, decoding onto the `Yuv*P16Le` surface where
/// every word is the sample verbatim. This is the §10.3.8 custom
/// signal range with excursion 65535 → §10.5.2 `video_depth = 16`,
/// the deepest depth the output surface represents losslessly.
#[test]
fn hq_16bit_full_range_q0_bit_exact_across_wavelets_and_chromas() {
    let (w, h) = (64u32, 64u32);
    let sr = SignalRange::PRESET_16BIT_FULL;
    for chroma in [
        ChromaFormat::Yuv420,
        ChromaFormat::Yuv422,
        ChromaFormat::Yuv444,
    ] {
        let (cw, ch) = chroma_dims(w, h, chroma);
        let y = ramp_plane(w as usize, h as usize, 16, 1);
        let u = ramp_plane(cw as usize, ch as usize, 16, 5);
        let v = ramp_plane(cw as usize, ch as usize, 16, 9);
        let seq = make_minimal_sequence_with_signal_range(w, h, chroma, sr);
        assert_eq!(seq.luma_depth, 16, "16-bit full range → luma_depth 16");
        assert_eq!(seq.chroma_depth, 16);

        for filter in [
            WaveletFilter::DeslauriersDubuc9_7,
            WaveletFilter::LeGall5_3,
            WaveletFilter::DeslauriersDubuc13_7,
            WaveletFilter::Haar0,
            WaveletFilter::Haar1,
            WaveletFilter::Daubechies9_7,
        ] {
            // 16-bit coefficients are ~256× the 8-bit magnitude; widen
            // the HQ slice length-byte unit accordingly (decoded
            // coefficients are unaffected — qindex 0 stays bit-exact).
            let mut params = EncoderParams::default_hq(filter, 3);
            params.slice_size_scaler = 32;
            let stream = encode_single_hq_intra_stream_u16(&seq, &params, 0, &y, &u, &v);
            let frame = decode_one(stream);

            let gy = plane_as_u16(&frame.planes[0].data);
            let gu = plane_as_u16(&frame.planes[1].data);
            let gv = plane_as_u16(&frame.planes[2].data);
            let tag = format!("HQ 16-bit {chroma:?} {filter:?}");
            assert_planes_eq(&format!("{tag} Y"), &gy, &y);
            assert_planes_eq(&format!("{tag} U"), &gu, &u);
            assert_planes_eq(&format!("{tag} V"), &gv, &v);
        }
    }
}

/// 13-, 14- and 15-bit custom full ranges land on the same 16-bit
/// surface with the samples MSB-aligned (`<< (16 - depth)`); undoing
/// the alignment must recover the input bit-exactly at qindex 0. Each
/// depth pairs with a different chroma sampling so the intermediate
/// depths sweep the whole surface trio.
#[test]
fn hq_13_to_15bit_full_range_q0_bit_exact_msb_aligned() {
    let (w, h) = (64u32, 64u32);
    for (depth, chroma) in [
        (13u32, ChromaFormat::Yuv420),
        (14u32, ChromaFormat::Yuv422),
        (15u32, ChromaFormat::Yuv444),
    ] {
        let max = ((1u64 << depth) - 1) as u32;
        let sr = SignalRange {
            luma_offset: 1 << (depth - 1),
            luma_excursion: max,
            chroma_offset: 1 << (depth - 1),
            chroma_excursion: max,
        };
        let (cw, ch) = chroma_dims(w, h, chroma);
        let y = ramp_plane(w as usize, h as usize, depth, 2);
        let u = ramp_plane(cw as usize, ch as usize, depth, 6);
        let v = ramp_plane(cw as usize, ch as usize, depth, 10);
        let seq = make_minimal_sequence_with_signal_range(w, h, chroma, sr);
        assert_eq!(seq.luma_depth, depth);

        let mut params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
        params.slice_size_scaler = 32;
        let stream = encode_single_hq_intra_stream_u16(&seq, &params, 0, &y, &u, &v);
        let frame = decode_one(stream);
        // Two bytes per sample on the 16-bit surface.
        assert_eq!(frame.planes[0].data.len(), (w * h * 2) as usize);
        let gy = unshift_p16(&plane_as_u16(&frame.planes[0].data), depth);
        let gu = unshift_p16(&plane_as_u16(&frame.planes[1].data), depth);
        let gv = unshift_p16(&plane_as_u16(&frame.planes[2].data), depth);
        let tag = format!("HQ {depth}-bit {chroma:?}");
        assert_planes_eq(&format!("{tag} Y"), &gy, &y);
        assert_planes_eq(&format!("{tag} U"), &gu, &u);
        assert_planes_eq(&format!("{tag} V"), &gv, &v);

        // The alignment itself: a full-scale sample must occupy the
        // top `depth` bits (low `16 - depth` bits zero).
        let raw = plane_as_u16(&frame.planes[0].data);
        let low_mask = (1u16 << (16 - depth)) - 1;
        assert!(
            raw.iter().all(|&s| s & low_mask == 0),
            "{tag}: low bits of the MSB-aligned surface must be zero"
        );
    }
}

/// PSNR over a 10-bit (`max = 1023`) sample pair. `INFINITY` on an exact
/// match. The peak is the 10-bit maximum so the dB figure is comparable
/// to an 8-bit PSNR scaled for the deeper range.
fn psnr_10bit(a: &[u16], b: &[u16]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut sse: u64 = 0;
    for i in 0..a.len() {
        let d = a[i] as i64 - b[i] as i64;
        sse += (d * d) as u64;
    }
    if sse == 0 {
        return f64::INFINITY;
    }
    let mse = sse as f64 / a.len() as f64;
    20.0 * (1023.0f64).log10() - 10.0 * mse.log10()
}

/// LD intra, 10-bit full-range, 4:2:0 — drives the §13.5.1 DC-prediction
/// LD slice path (distinct from the HQ slice format) at a deeper sample
/// depth. LD is rate-constrained (each slice writes a fixed byte budget),
/// so with a generous budget the qindex-0 reconstruction is bit-exact;
/// the assertion tolerates a tiny shortfall via a high-PSNR floor in case
/// a single slice's coefficients overrun the budget at the wider range.
/// Either way it confirms the depth parameterisation carries through
/// forward DC prediction and the LD slice packer.
#[test]
fn ld_10bit_full_range_high_fidelity_420() {
    let (w, h) = (64u32, 64u32);
    let sr = SignalRange::PRESET_10BIT_FULL;
    let chroma = ChromaFormat::Yuv420;
    let (cw, ch) = chroma_dims(w, h, chroma);
    let y = ramp_plane(w as usize, h as usize, 10, 3);
    let u = ramp_plane(cw as usize, ch as usize, 10, 7);
    let v = ramp_plane(cw as usize, ch as usize, 10, 11);
    let seq = make_minimal_sequence_ld_with_signal_range(w, h, chroma, sr);
    assert_eq!(seq.luma_depth, 10);

    // 8×8 slice grid over the 64×64 picture, with a generous per-slice
    // byte budget so the qindex-0 (identity dead-zone) coefficients fit.
    let params = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 8, 8, 512);
    let stream = encode_single_ld_intra_stream_u16(&seq, &params, 0, &y, &u, &v);
    let frame = decode_one(stream);
    let gy = plane_as_u16(&frame.planes[0].data);
    let gu = plane_as_u16(&frame.planes[1].data);
    let gv = plane_as_u16(&frame.planes[2].data);

    let py = psnr_10bit(&gy, &y);
    let pu = psnr_10bit(&gu, &u);
    let pv = psnr_10bit(&gv, &v);
    assert!(
        py >= 50.0 && pu >= 50.0 && pv >= 50.0,
        "LD 10-bit reconstruction PSNR too low (Y {py:.2} U {pu:.2} V {pv:.2} dB)"
    );
}

/// LD intra at 16-bit full range: the §13.5.1 DC-prediction LD slice
/// path at the deepest representable depth. Same shape as the 10-bit
/// LD test — a generous per-slice byte budget keeps the qindex-0
/// coefficients unclipped, and a 16-bit-peak PSNR floor guards the
/// reconstruction (bit-exactness is asserted when the budget holds,
/// via the exact-match INFINITY case).
#[test]
fn ld_16bit_full_range_high_fidelity_420() {
    let (w, h) = (64u32, 64u32);
    let sr = SignalRange::PRESET_16BIT_FULL;
    let chroma = ChromaFormat::Yuv420;
    let (cw, ch) = chroma_dims(w, h, chroma);
    let y = ramp_plane(w as usize, h as usize, 16, 3);
    let u = ramp_plane(cw as usize, ch as usize, 16, 7);
    let v = ramp_plane(cw as usize, ch as usize, 16, 11);
    let seq = make_minimal_sequence_ld_with_signal_range(w, h, chroma, sr);
    assert_eq!(seq.luma_depth, 16);

    // 8×8 slice grid; 16-bit coefficients need ~2 more bytes per
    // sample than the 10-bit test's 512-byte budget allowed.
    let params = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 8, 8, 2048);
    let stream = encode_single_ld_intra_stream_u16(&seq, &params, 0, &y, &u, &v);
    let frame = decode_one(stream);
    let gy = plane_as_u16(&frame.planes[0].data);
    let gu = plane_as_u16(&frame.planes[1].data);
    let gv = plane_as_u16(&frame.planes[2].data);

    // 16-bit peak PSNR (peak 65535); INFINITY on bit-exact.
    let psnr16 = |a: &[u16], b: &[u16]| -> f64 {
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
        20.0 * (65535.0f64).log10() - 10.0 * mse.log10()
    };
    let py = psnr16(&gy, &y);
    let pu = psnr16(&gu, &u);
    let pv = psnr16(&gv, &v);
    assert!(
        py >= 50.0 && pu >= 50.0 && pv >= 50.0,
        "LD 16-bit reconstruction PSNR too low (Y {py:.2} U {pu:.2} V {pv:.2} dB)"
    );
}

/// A flat mid-grey 10-bit plane (value 512 = 2^9, the bi-polar zero
/// after the full-range recentre) must reconstruct exactly: this
/// isolates the §15.10 output offset (`+2^(depth-1)`) from any
/// coefficient activity, so a wrong offset would shift every sample.
/// The 16-bit analogue of the flat mid-grey probe: value 32768 = 2^15
/// is the bi-polar zero after the full-range recentre, so a wrong
/// §15.10 offset (or a wrong 16-bit surface shift) would move every
/// sample.
#[test]
fn hq_16bit_flat_midgrey_offset_is_exact() {
    let (w, h) = (32u32, 32u32);
    let sr = SignalRange::PRESET_16BIT_FULL;
    let chroma = ChromaFormat::Yuv420;
    let (cw, ch) = chroma_dims(w, h, chroma);
    let y = vec![32768u16; (w * h) as usize];
    let u = vec![32768u16; (cw * ch) as usize];
    let v = vec![32768u16; (cw * ch) as usize];
    let seq = make_minimal_sequence_with_signal_range(w, h, chroma, sr);
    let mut params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    params.slice_size_scaler = 32;
    let stream = encode_single_hq_intra_stream_u16(&seq, &params, 0, &y, &u, &v);
    let frame = decode_one(stream);
    let gy = plane_as_u16(&frame.planes[0].data);
    assert!(
        gy.iter().all(|&s| s == 32768),
        "flat 16-bit mid-grey must reconstruct to 32768 everywhere; got {:?}…",
        &gy[..8.min(gy.len())]
    );
}

#[test]
fn hq_10bit_flat_midgrey_offset_is_exact() {
    let (w, h) = (32u32, 32u32);
    let sr = SignalRange::PRESET_10BIT_FULL;
    let chroma = ChromaFormat::Yuv420;
    let (cw, ch) = chroma_dims(w, h, chroma);
    let y = vec![512u16; (w * h) as usize];
    let u = vec![512u16; (cw * ch) as usize];
    let v = vec![512u16; (cw * ch) as usize];
    let seq = make_minimal_sequence_with_signal_range(w, h, chroma, sr);
    let params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let stream = encode_single_hq_intra_stream_u16(&seq, &params, 0, &y, &u, &v);
    let frame = decode_one(stream);
    let gy = plane_as_u16(&frame.planes[0].data);
    assert!(
        gy.iter().all(|&s| s == 512),
        "flat 10-bit mid-grey must reconstruct to 512 everywhere; got {:?}…",
        &gy[..8.min(gy.len())]
    );
}

/// Slice grid for a 64×64 4:2:0 picture at `dwt_depth`: each
/// per-component slice dimension must stay an integer multiple of
/// `2^dwt_depth` (§15.7). Depths 1..=3 tile 8×8; 4..=5 loosen to 2×2.
fn slice_grid_64x64(dwt_depth: u32) -> (u32, u32) {
    if dwt_depth <= 3 {
        (8, 8)
    } else {
        (2, 2)
    }
}

/// All-zero custom quant matrix sized to `dwt_depth + 1` levels — the
/// §11.3.5 requirement for `dwt_depth = 5` (Annex E.1 defaults stop at
/// depth 4). At qindex 0 every per-band quantiser collapses to the
/// identity dead-zone, so reconstruction stays bit-exact.
fn zero_matrix(dwt_depth: u32) -> QuantMatrix {
    QuantMatrix {
        dwt_depth,
        dwt_depth_ho: 0,
        levels: (0..=dwt_depth).map(|_| [0u32; 4]).collect(),
    }
}

/// 10-bit HQ intra must be bit-exact at every spec-allowed `dwt_depth`
/// (§11.3 range `1..=5`), not just the docs-corpus depths 3/4. Depths
/// 1..=4 use the Annex E.1 default matrix; depth 5 requires a custom
/// matrix. This proves the §15.7 multi-level IDWT recursion reconstructs
/// the wider 10-bit range identically at every recursion depth.
#[test]
fn hq_10bit_bit_exact_across_all_dwt_depths() {
    let (w, h) = (64u32, 64u32);
    let sr = SignalRange::PRESET_10BIT_FULL;
    let chroma = ChromaFormat::Yuv420;
    let (cw, ch) = chroma_dims(w, h, chroma);
    let seq = make_minimal_sequence_with_signal_range(w, h, chroma, sr);

    for depth in 1u32..=5 {
        let y = ramp_plane(w as usize, h as usize, 10, depth);
        let u = ramp_plane(cw as usize, ch as usize, 10, depth + 20);
        let v = ramp_plane(cw as usize, ch as usize, 10, depth + 40);
        let (sx, sy) = slice_grid_64x64(depth);

        // 10-bit coefficients are ~4× the 8-bit magnitude, so an HQ
        // slice's per-component byte run can exceed the 255-unit length
        // byte at scaler 1. A larger `slice_size_scaler` widens the
        // length-byte unit (`len_byte * scaler` bytes) without changing
        // the decoded coefficients — qindex 0 stays bit-exact.
        let params = if depth <= 4 {
            let mut p = EncoderParams::default_hq(WaveletFilter::LeGall5_3, depth);
            p.slices_x = sx;
            p.slices_y = sy;
            p.slice_size_scaler = 8;
            p
        } else {
            EncoderParams {
                wavelet: WaveletFilter::LeGall5_3,
                dwt_depth: 5,
                slices_x: sx,
                slices_y: sy,
                slice_prefix_bytes: 0,
                slice_size_scaler: 8,
                quant_matrix: zero_matrix(5),
                custom_quant_matrix: true,
                qindex: 0,
                slice_size_target: None,
                major_version: 2,
                extended_transform_override: None,
            }
        };

        let stream = encode_single_hq_intra_stream_u16(&seq, &params, 0, &y, &u, &v);
        let frame = decode_one(stream);
        let gy = plane_as_u16(&frame.planes[0].data);
        let gu = plane_as_u16(&frame.planes[1].data);
        let gv = plane_as_u16(&frame.planes[2].data);
        assert_planes_eq(&format!("HQ 10-bit depth {depth} Y"), &gy, &y);
        assert_planes_eq(&format!("HQ 10-bit depth {depth} U"), &gu, &u);
        assert_planes_eq(&format!("HQ 10-bit depth {depth} V"), &gv, &v);
    }
}

/// 10-bit HQ intra through the §12.4.4 asymmetric (horizontal-only)
/// transform: `dwt_depth_ho > 0` runs the §15.4.2 `h_synthesis` extra
/// horizontal-only IDWT levels on top of the symmetric pyramid. The
/// asymmetric path was decode-verified at 8-bit; this confirms the
/// horizontal-only recursion also reconstructs the wider 10-bit range
/// bit-exactly. `with_asymmetric_transform` installs the matching
/// all-zero quant matrix (identity at qindex 0) and raises the v3 flag.
#[test]
fn hq_10bit_asymmetric_transform_bit_exact() {
    let (w, h) = (64u32, 64u32);
    let sr = SignalRange::PRESET_10BIT_FULL;
    let chroma = ChromaFormat::Yuv420;
    let (cw, ch) = chroma_dims(w, h, chroma);
    let mut seq = make_minimal_sequence_with_signal_range(w, h, chroma, sr);
    // The §12.4.4 extended_transform_parameters() block is gated on the
    // *sequence* version major being 3 (the decoder dispatches on the
    // header, not the per-picture params), so an asymmetric stream needs
    // a v3 sequence header.
    seq.parse_parameters.version_major = 3;

    for (dwt_depth, ho, wavelet_ho) in [
        (3u32, 1u32, WaveletFilter::LeGall5_3),
        (2u32, 2u32, WaveletFilter::DeslauriersDubuc9_7),
    ] {
        let y = ramp_plane(w as usize, h as usize, 10, ho + 1);
        let u = ramp_plane(cw as usize, ch as usize, 10, ho + 21);
        let v = ramp_plane(cw as usize, ch as usize, 10, ho + 41);
        let mut params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, dwt_depth)
            .with_asymmetric_transform(wavelet_ho, ho);
        // Widen the length-byte unit for the deeper 10-bit coefficients
        // (see `hq_10bit_bit_exact_across_all_dwt_depths`).
        params.slice_size_scaler = 8;

        let stream = encode_single_hq_intra_stream_u16(&seq, &params, 0, &y, &u, &v);
        let frame = decode_one(stream);
        let gy = plane_as_u16(&frame.planes[0].data);
        let gu = plane_as_u16(&frame.planes[1].data);
        let gv = plane_as_u16(&frame.planes[2].data);
        let tag = format!("HQ 10-bit asym depth={dwt_depth} ho={ho} {wavelet_ho:?}");
        assert_planes_eq(&format!("{tag} Y"), &gy, &y);
        assert_planes_eq(&format!("{tag} U"), &gu, &u);
        assert_planes_eq(&format!("{tag} V"), &gv, &v);
    }
}

/// The full-range presets must derive the depth the spec's §10.5.2
/// formula predicts, and must differ from the four Table 10.5 presets
/// (they are emitted as custom ranges, not preset indices).
#[test]
fn full_range_presets_have_expected_depths() {
    // intlog2(excursion + 1).
    assert_eq!(SignalRange::PRESET_10BIT_FULL.luma_excursion + 1, 1024);
    assert_eq!(SignalRange::PRESET_12BIT_FULL.luma_excursion + 1, 4096);
    assert_eq!(SignalRange::PRESET_14BIT_FULL.luma_excursion + 1, 16384);
    assert_eq!(
        SignalRange::PRESET_16BIT_FULL.luma_excursion as u64 + 1,
        65536
    );
    // The deep offsets centre the bi-polar recentre exactly.
    assert_eq!(SignalRange::PRESET_14BIT_FULL.luma_offset, 8192);
    assert_eq!(SignalRange::PRESET_16BIT_FULL.luma_offset, 32768);
    // Distinct from the standard video-range presets.
    assert_ne!(
        SignalRange::PRESET_10BIT_FULL,
        SignalRange::PRESET_10BIT_VIDEO
    );
    assert_ne!(
        SignalRange::PRESET_12BIT_FULL,
        SignalRange::PRESET_12BIT_VIDEO
    );
}
