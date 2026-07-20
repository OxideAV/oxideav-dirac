//! Round-419 — deep-colour (u16) **inter** encode → decode round-trips.
//!
//! The round-417 deep-colour work proved the 13-16-bit intra chain
//! (q0 bit-exact `&[u16]` encode → decode). This suite extends the bar
//! to the **inter** paths: 1-ref P (`0x09`) and 2-ref bipred B (`0x0A`)
//! pictures whose sources are 10/12/16-bit `&[u16]` planes, driven
//! through the generic sample-width inter pipeline
//! (`encoder_inter::InterSample`) and decoded back through the crate's
//! own registry decoder.
//!
//! Why bit-exact is the right bar: at `qindex = 0` the residue
//! dead-zone quantiser is the identity and the LeGall 5/3 lifting is
//! perfectly reversible, so `source = OBMC(decoded reference) +
//! residue` closes exactly — for **any** motion field the ME picks —
//! provided every depth-parameterised step (signed recentre by
//! `2^(depth-1)`, §15.8.11 upconversion clip, §15.8.5 blend clip,
//! §15.9/§15.10 output) is consistent between encoder and decoder at
//! the deeper sample widths. A single off-by-one in any of those
//! surfaces as a plane mismatch here.
//!
//! Wall: BBC Dirac spec §10.3.8 / §10.5.2 (signal range → video
//! depth), §11.2-§11.3 (inter pictures + wavelet residue), §15.8
//! (motion compensation) from `docs/video/dirac/dirac-spec-latest.pdf`.
//! No external source, no web.

use oxideav_core::{CodecId, CodecParameters, CodecRegistry, Frame, Packet, TimeBase, VideoFrame};
use oxideav_dirac::encoder::{make_minimal_sequence_with_signal_range, EncoderParams};
use oxideav_dirac::encoder_inter::{
    encode_intra_then_inter_stream, estimate_global_motion_config, GlobalMotionConfig,
    GlobalMotionModel, InterEncoderParams, InterInputPicture, ResidueParams,
};
use oxideav_dirac::encoder_intra_core::{
    encode_core_intra_then_bipred_stream, encode_core_intra_then_inter_stream,
    CoreIntraEncoderParams,
};
use oxideav_dirac::picture_inter::GlobalParams;
use oxideav_dirac::video_format::{ChromaFormat, SignalRange};
use oxideav_dirac::wavelet::WaveletFilter;

/// Decode a whole elementary stream through the registry decoder and
/// return the frames in decode order.
fn decode_stream(stream: Vec<u8>) -> Vec<VideoFrame> {
    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("make decoder");
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&pkt).expect("send_packet");
    let mut out = Vec::new();
    while let Ok(frame) = dec.receive_frame() {
        match frame {
            Frame::Video(v) => out.push(v),
            other => panic!("expected video frame, got {other:?}"),
        }
    }
    out
}

/// Reinterpret a decoded plane's little-endian 16-bit samples as `u16`.
fn plane_as_u16(data: &[u8]) -> Vec<u16> {
    assert_eq!(data.len() % 2, 0, "16-bit plane must have even byte length");
    data.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn chroma_dims(w: usize, h: usize, chroma: ChromaFormat) -> (usize, usize) {
    match chroma {
        ChromaFormat::Yuv420 => (w / 2, h / 2),
        ChromaFormat::Yuv422 => (w / 2, h),
        ChromaFormat::Yuv444 => (w, h),
    }
}

/// Compare a decoded 16-bit plane against the `u16` source with a
/// first-divergence report.
fn assert_plane_eq(label: &str, got: &[u8], want: &[u16]) {
    let got = plane_as_u16(got);
    assert_eq!(got.len(), want.len(), "{label}: plane length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g, w, "{label}: sample {i} mismatch (got {g}, want {w})");
    }
}

/// Two-frame deep translating scene: a textured 16×16 square (values in
/// the upper half of the depth range) over a smooth dim gradient
/// background; frame 1 translates the square by `(dx, dy)` pels. The
/// square texture translates with it, so ME must genuinely track deep
/// sample values.
#[allow(clippy::type_complexity)]
fn deep_translating_pair(
    w: usize,
    h: usize,
    chroma: ChromaFormat,
    depth: u32,
    dx: usize,
    dy: usize,
) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
    let max = (1u32 << depth) - 1;
    let bg = |x: usize, y: usize| -> u16 { ((x as u32 * 3 + y as u32 * 5) % (max / 8 + 1)) as u16 };
    let tex = |lx: usize, ly: usize| -> u16 {
        (max / 2 + ((lx as u32 * 13 + ly as u32 * 7) * 97) % (max / 2 + 1)) as u16
    };
    let mut y0: Vec<u16> = (0..w * h).map(|i| bg(i % w, i / w)).collect();
    let mut y1 = y0.clone();
    for ly in 0..16 {
        for lx in 0..16 {
            y0[(20 + ly) * w + (20 + lx)] = tex(lx, ly);
            y1[(20 + dy + ly) * w + (20 + dx + lx)] = tex(lx, ly);
        }
    }
    let (cw, ch) = chroma_dims(w, h, chroma);
    let mid = 1u16 << (depth - 1);
    let u0 = vec![mid; cw * ch];
    let v0 = vec![mid.wrapping_add(max as u16 / 5); cw * ch];
    (y0, u0.clone(), v0.clone(), y1, u0, v0)
}

/// HQ-anchored 1-ref P chain (`0xEC` + `0x09`), residue at qindex 0 —
/// both decoded frames must be **bit-exact** against the `u16` sources.
fn assert_p_chain_bit_exact(depth: u32, sr: SignalRange, chroma: ChromaFormat) {
    let (w, h) = (64usize, 64usize);
    let seq = make_minimal_sequence_with_signal_range(w as u32, h as u32, chroma, sr);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter_params = InterEncoderParams::default(); // residue ON, qindex 0

    let (y0, u0, v0, y1, u1, v1) = deep_translating_pair(w, h, chroma, depth, 4, 2);
    let intra = InterInputPicture {
        picture_number: 10,
        y: &y0[..],
        u: &u0[..],
        v: &v0[..],
    };
    let inter = InterInputPicture {
        picture_number: 11,
        y: &y1[..],
        u: &u1[..],
        v: &v1[..],
    };
    let stream = encode_intra_then_inter_stream(&seq, &intra_params, &inter_params, &intra, &inter);
    let frames = decode_stream(stream);
    assert_eq!(frames.len(), 2, "expected intra + inter frames");

    assert_plane_eq(
        &format!("{depth}-bit intra Y"),
        &frames[0].planes[0].data,
        &y0,
    );
    assert_plane_eq(
        &format!("{depth}-bit intra U"),
        &frames[0].planes[1].data,
        &u0,
    );
    assert_plane_eq(
        &format!("{depth}-bit intra V"),
        &frames[0].planes[2].data,
        &v0,
    );
    assert_plane_eq(
        &format!("{depth}-bit inter Y"),
        &frames[1].planes[0].data,
        &y1,
    );
    assert_plane_eq(
        &format!("{depth}-bit inter U"),
        &frames[1].planes[1].data,
        &u1,
    );
    assert_plane_eq(
        &format!("{depth}-bit inter V"),
        &frames[1].planes[2].data,
        &v1,
    );
}

#[test]
fn p_chain_10bit_420_q0_bit_exact() {
    assert_p_chain_bit_exact(10, SignalRange::PRESET_10BIT_FULL, ChromaFormat::Yuv420);
}

#[test]
fn p_chain_12bit_422_q0_bit_exact() {
    assert_p_chain_bit_exact(12, SignalRange::PRESET_12BIT_FULL, ChromaFormat::Yuv422);
}

#[test]
fn p_chain_16bit_444_q0_bit_exact() {
    assert_p_chain_bit_exact(16, SignalRange::PRESET_16BIT_FULL, ChromaFormat::Yuv444);
}

/// Homogeneous core-syntax P chain (`0x0C` + `0x09`) — the anchor goes
/// through the deep core-syntax AC intra entry instead of the HQ one.
#[test]
fn core_syntax_p_chain_16bit_420_q0_bit_exact() {
    let (w, h) = (64usize, 64usize);
    let chroma = ChromaFormat::Yuv420;
    let seq = make_minimal_sequence_with_signal_range(
        w as u32,
        h as u32,
        chroma,
        SignalRange::PRESET_16BIT_FULL,
    );
    let intra_params = CoreIntraEncoderParams::default_intra(WaveletFilter::LeGall5_3, 3);
    let inter_params = InterEncoderParams::default();

    let (y0, u0, v0, y1, u1, v1) = deep_translating_pair(w, h, chroma, 16, 4, 2);
    let intra = InterInputPicture {
        picture_number: 0,
        y: &y0[..],
        u: &u0[..],
        v: &v0[..],
    };
    let inter = InterInputPicture {
        picture_number: 1,
        y: &y1[..],
        u: &u1[..],
        v: &v1[..],
    };
    let stream =
        encode_core_intra_then_inter_stream(&seq, &intra_params, &inter_params, &intra, &inter);
    let frames = decode_stream(stream);
    assert_eq!(frames.len(), 2);
    assert_plane_eq("core 16-bit intra Y", &frames[0].planes[0].data, &y0);
    assert_plane_eq("core 16-bit inter Y", &frames[1].planes[0].data, &y1);
    assert_plane_eq("core 16-bit inter U", &frames[1].planes[1].data, &u1);
    assert_plane_eq("core 16-bit inter V", &frames[1].planes[2].data, &v1);
}

/// Deep complementary-occluder triplet for the bipred path: a
/// horizontal bright bar only in ref-A, a vertical one only in ref-B,
/// and both at the §15.8.5 `(p1 + p2 + 1) >> 1` half-intensity in the
/// B picture — so `Ref1And2` blocks are genuinely exercised at deep
/// sample values (either single-ref prediction is wrong by ~2^(d-2)).
#[allow(clippy::type_complexity)]
fn deep_bipred_triplet(w: usize, h: usize, depth: u32) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
    let max = (1u32 << depth) - 1;
    let bg = (max / 6) as u16;
    let bright = (max - max / 8) as u16;
    let half = ((bright as u32 + bg as u32 + 1) >> 1) as u16;
    let mut ya = vec![bg; w * h];
    let mut yb = vec![bg; w * h];
    let mut ym = vec![bg; w * h];
    for r in 30..34 {
        for c in 0..w {
            ya[r * w + c] = bright;
            ym[r * w + c] = half;
        }
    }
    for c in 30..34 {
        for r in 0..h {
            yb[r * w + c] = bright;
            ym[r * w + c] = if (30..34).contains(&r) { bright } else { half };
        }
    }
    let mid = vec![1u16 << (depth - 1); (w / 2) * (h / 2)];
    (ya, yb, ym, mid)
}

/// Core-syntax bipred chain (`0x0C` + `0x0C` + `0x0A`) at deep depths,
/// residue at qindex 0 — all three decoded frames bit-exact.
fn assert_bipred_chain_bit_exact(depth: u32, sr: SignalRange) {
    let (w, h) = (64usize, 64usize);
    let chroma = ChromaFormat::Yuv420;
    let seq = make_minimal_sequence_with_signal_range(w as u32, h as u32, chroma, sr);
    let intra_params = CoreIntraEncoderParams::default_intra(WaveletFilter::LeGall5_3, 3);
    let inter_params = InterEncoderParams::default();

    let (ya, yb, ym, c) = deep_bipred_triplet(w, h, depth);
    let intra_a = InterInputPicture {
        picture_number: 0,
        y: &ya[..],
        u: &c[..],
        v: &c[..],
    };
    let intra_b = InterInputPicture {
        picture_number: 2,
        y: &yb[..],
        u: &c[..],
        v: &c[..],
    };
    let bipred = InterInputPicture {
        picture_number: 1,
        y: &ym[..],
        u: &c[..],
        v: &c[..],
    };
    let stream = encode_core_intra_then_bipred_stream(
        &seq,
        &intra_params,
        &inter_params,
        &intra_a,
        &intra_b,
        &bipred,
    );
    let frames = decode_stream(stream);
    assert_eq!(frames.len(), 3, "expected intra-A, intra-B, bipred B");
    assert_plane_eq(
        &format!("{depth}-bit intra-A Y"),
        &frames[0].planes[0].data,
        &ya,
    );
    assert_plane_eq(
        &format!("{depth}-bit intra-B Y"),
        &frames[1].planes[0].data,
        &yb,
    );
    assert_plane_eq(
        &format!("{depth}-bit bipred Y"),
        &frames[2].planes[0].data,
        &ym,
    );
    assert_plane_eq(
        &format!("{depth}-bit bipred U"),
        &frames[2].planes[1].data,
        &c,
    );
    assert_plane_eq(
        &format!("{depth}-bit bipred V"),
        &frames[2].planes[2].data,
        &c,
    );
}

#[test]
fn bipred_chain_10bit_q0_bit_exact() {
    assert_bipred_chain_bit_exact(10, SignalRange::PRESET_10BIT_FULL);
}

#[test]
fn bipred_chain_16bit_q0_bit_exact() {
    assert_bipred_chain_bit_exact(16, SignalRange::PRESET_16BIT_FULL);
}

// ---- Deep codeblock-grid residue + global motion (round-419) ---------

/// §11.3.3 codeblock spatial partition on a deep P picture: 16-bit
/// source, mode-0 codeblock grid (2×2 per detail level), residue at
/// qindex 0 — still bit-exact (the codeblock walk is a byte-for-byte
/// mirror of the intra-core encoder at any sample depth).
#[test]
fn p_chain_16bit_codeblock_residue_q0_bit_exact() {
    let (w, h) = (64usize, 64usize);
    let chroma = ChromaFormat::Yuv420;
    let seq = make_minimal_sequence_with_signal_range(
        w as u32,
        h as u32,
        chroma,
        SignalRange::PRESET_16BIT_FULL,
    );
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let mut rp = ResidueParams::default_for(WaveletFilter::LeGall5_3, 3);
    rp.codeblocks = Some(vec![(1, 1), (2, 2), (2, 2), (2, 2)]);
    rp.codeblock_mode = 0;
    let inter_params = InterEncoderParams {
        residue: Some(rp),
        ..InterEncoderParams::default()
    };

    let (y0, u0, v0, y1, u1, v1) = deep_translating_pair(w, h, chroma, 16, 4, 2);
    let intra = InterInputPicture {
        picture_number: 0,
        y: &y0[..],
        u: &u0[..],
        v: &v0[..],
    };
    let inter = InterInputPicture {
        picture_number: 1,
        y: &y1[..],
        u: &u1[..],
        v: &v1[..],
    };
    let stream = encode_intra_then_inter_stream(&seq, &intra_params, &inter_params, &intra, &inter);
    let frames = decode_stream(stream);
    assert_eq!(frames.len(), 2);
    assert_plane_eq("cb 16-bit inter Y", &frames[1].planes[0].data, &y1);
    assert_plane_eq("cb 16-bit inter U", &frames[1].planes[1].data, &u1);
    assert_plane_eq("cb 16-bit inter V", &frames[1].planes[2].data, &v1);
}

/// Whole-frame deep pan: sample an infinite deterministic texture at a
/// `(dx, dy)` offset so frame 1 is a true translation of frame 0 with
/// fresh content entering at the edges.
fn deep_pan_pair(
    w: usize,
    h: usize,
    depth: u32,
    dx: i64,
    dy: i64,
) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
    let max = (1u64 << depth) - 1;
    let tex = |x: i64, y: i64| -> u16 {
        let mix = (x + 512) as u64 * 17 + (y + 512) as u64 * 31;
        ((mix * 2654435761) % (max + 1)) as u16
    };
    let mut y0 = Vec::with_capacity(w * h);
    let mut y1 = Vec::with_capacity(w * h);
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            y0.push(tex(x, y));
            y1.push(tex(x + dx, y + dy));
        }
    }
    let mid = vec![1u16 << (depth - 1); (w / 2) * (h / 2)];
    (y0, y1, mid.clone(), mid)
}

/// §11.2.6 global motion on a deep P picture: whole-picture pan field
/// at 10-bit, integer-pel, residue qindex 0 — bit-exact through the
/// decoder's §15.8.8 `global_mv` path at deep sample values.
#[test]
fn p_chain_10bit_whole_picture_global_pan_q0_bit_exact() {
    let (w, h) = (64usize, 64usize);
    let chroma = ChromaFormat::Yuv420;
    let seq = make_minimal_sequence_with_signal_range(
        w as u32,
        h as u32,
        chroma,
        SignalRange::PRESET_10BIT_FULL,
    );
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    // Zero affine matrix → constant field `pan_tilt + (1, 1)`.
    // pan_tilt (3, -1) ⇒ field (4, 0): prediction reads ref at (x+4, y),
    // matching a source that panned by sampling `tex(x + 4, y)`.
    let inter_params = InterEncoderParams {
        mv_precision: 0,
        global_motion: Some(GlobalMotionConfig {
            global1: GlobalParams {
                pan_tilt: (3, -1),
                zrs: [[0, 0], [0, 0]],
                zrs_exp: 0,
                perspective: (0, 0),
                persp_exp: 0,
            },
            global2: None,
            block_gmode: None, // whole picture global
        }),
        ..InterEncoderParams::default()
    };

    let (y0, y1, c0, c1) = deep_pan_pair(w, h, 10, 4, 0);
    let intra = InterInputPicture {
        picture_number: 0,
        y: &y0[..],
        u: &c0[..],
        v: &c0[..],
    };
    let inter = InterInputPicture {
        picture_number: 1,
        y: &y1[..],
        u: &c1[..],
        v: &c1[..],
    };
    let stream = encode_intra_then_inter_stream(&seq, &intra_params, &inter_params, &intra, &inter);
    let frames = decode_stream(stream);
    assert_eq!(frames.len(), 2);
    assert_plane_eq("global 10-bit inter Y", &frames[1].planes[0].data, &y1);
    assert_plane_eq("global 10-bit inter U", &frames[1].planes[1].data, &c1);
    assert_plane_eq("global 10-bit inter V", &frames[1].planes[2].data, &c1);
}

/// Bipred with a §11.2.6 global model per reference at 16-bit: ref-A
/// pans by (+4, 0), ref-B by (-4, 0) relative to the B picture, each
/// reference carries its own pan field, every block global — bit-exact
/// through the 2-ref global path at deep sample values.
#[test]
fn bipred_16bit_global_both_refs_q0_bit_exact() {
    let (w, h) = (64usize, 64usize);
    let seq = make_minimal_sequence_with_signal_range(
        w as u32,
        h as u32,
        ChromaFormat::Yuv420,
        SignalRange::PRESET_16BIT_FULL,
    );
    let intra_params = CoreIntraEncoderParams::default_intra(WaveletFilter::LeGall5_3, 3);
    // B picture at pan offset 0; ref-A sampled at -4 (so the field
    // mapping B → A reads ref at (x - 4) ⇒ pan_tilt (-5, -1)); ref-B
    // sampled at +4 (field (4, 0) ⇒ pan_tilt (3, -1)).
    let (yb_pic, ya, c0, _c1) = deep_pan_pair(w, h, 16, -4, 0);
    let (_, yb, _, _) = deep_pan_pair(w, h, 16, 4, 0);
    let pan = |dx: i32| GlobalParams {
        pan_tilt: (dx - 1, -1),
        zrs: [[0, 0], [0, 0]],
        zrs_exp: 0,
        perspective: (0, 0),
        persp_exp: 0,
    };
    let inter_params = InterEncoderParams {
        bipred_mv_precision: 0,
        global_motion: Some(GlobalMotionConfig {
            global1: pan(-4),
            global2: Some(pan(4)),
            block_gmode: None,
        }),
        ..InterEncoderParams::default()
    };
    let intra_a = InterInputPicture {
        picture_number: 0,
        y: &ya[..],
        u: &c0[..],
        v: &c0[..],
    };
    let intra_b = InterInputPicture {
        picture_number: 2,
        y: &yb[..],
        u: &c0[..],
        v: &c0[..],
    };
    let bipred = InterInputPicture {
        picture_number: 1,
        y: &yb_pic[..],
        u: &c0[..],
        v: &c0[..],
    };
    let stream = encode_core_intra_then_bipred_stream(
        &seq,
        &intra_params,
        &inter_params,
        &intra_a,
        &intra_b,
        &bipred,
    );
    let frames = decode_stream(stream);
    assert_eq!(frames.len(), 3);
    assert_plane_eq(
        "global bipred 16-bit B Y",
        &frames[2].planes[0].data,
        &yb_pic,
    );
}

/// Global-model **estimation** from a deep ME grid: on a whole-frame
/// 12-bit pan the pan estimator must mark a large majority of blocks
/// global, and applying the estimated config must still round-trip
/// bit-exactly (the encoder mirrors its own effective parameters).
#[test]
fn estimate_global_pan_12bit_high_fraction_and_bit_exact() {
    let (w, h) = (64usize, 64usize);
    let seq = make_minimal_sequence_with_signal_range(
        w as u32,
        h as u32,
        ChromaFormat::Yuv420,
        SignalRange::PRESET_12BIT_FULL,
    );
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let base_params = InterEncoderParams {
        mv_precision: 0,
        ..InterEncoderParams::default()
    };
    let (y0, y1, c0, c1) = deep_pan_pair(w, h, 12, 4, 0);
    let (cfg, fraction) =
        estimate_global_motion_config(&seq, &base_params, &y1, &y0, GlobalMotionModel::Pan);
    assert!(
        fraction >= 0.8,
        "pan estimator marked only {fraction:.2} of blocks global on a whole-frame 12-bit pan"
    );
    let inter_params = InterEncoderParams {
        global_motion: Some(cfg),
        ..base_params
    };
    let intra = InterInputPicture {
        picture_number: 0,
        y: &y0[..],
        u: &c0[..],
        v: &c0[..],
    };
    let inter = InterInputPicture {
        picture_number: 1,
        y: &y1[..],
        u: &c1[..],
        v: &c1[..],
    };
    let stream = encode_intra_then_inter_stream(&seq, &intra_params, &inter_params, &intra, &inter);
    let frames = decode_stream(stream);
    assert_eq!(frames.len(), 2);
    assert_plane_eq(
        "estimated-pan 12-bit inter Y",
        &frames[1].planes[0].data,
        &y1,
    );
}
