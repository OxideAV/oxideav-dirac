//! §11.3.3 spatial-partition (codeblock) coverage for the inter-residue
//! encoder.
//!
//! The inter-residue path historically emitted `spatial_partition_flag =
//! 0` (one codeblock per subband). Round-370 adds the optional per-level
//! codeblock grid (mirroring the intra-core encoder's §11.3.3 support):
//! each HL/LH/HH subband splits into a grid of codeblocks, each carrying
//! a §13.4.3.3 `ZERO_BLOCK` skip flag and, under `codeblock_mode == 1`, a
//! §13.4.3.4 differential quantiser offset. The decoder reads this with
//! the same `picture_core::decode_subband` walk it uses for the intra
//! residue, so these tests verify the encode->decode loop end-to-end.
//!
//! The inter-residue codeblock encoder is a faithful mirror of the proven
//! intra-core encoder — an internal unit test
//! (`encoder_inter::tests::cb_residue_bytes_match_intra_core_byte_for_byte`)
//! asserts the two emit byte-identical AC streams on an identical band +
//! grid. Like the intra-core path, with reversible LeGall 5/3 at
//! `qindex = 0` the residue round-trips **bit-exactly** for every
//! codeblock geometry. (Before round-382's §B.2.7.1 terminator fix,
//! sub-4x4-sample codeblocks carried a "near-lossless roughness" on
//! their final AC symbols — that was the spurious-follow-bit terminator
//! bug in `ArithEncoder::finish()`, since removed;
//! `codeblock_sub4x4_grid_q0_legall_bit_exact` pins the exactness.)
//!
//! Source of truth: BBC Dirac Specification v2.2.3 §11.3.3
//! (codeblock_parameters), §13.4.3 (codeblock skip + quant offset),
//! §13.4.4 (codeblock entropy coding).

use oxideav_core::CodecRegistry;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_dirac::encoder::{make_minimal_sequence, EncoderParams};
use oxideav_dirac::encoder_inter::{
    encode_intra_then_inter_stream, synthetic_translating_pair_64, InterEncoderParams,
    InterInputPicture, ResidueParams,
};
use oxideav_dirac::video_format::ChromaFormat;
use oxideav_dirac::wavelet::WaveletFilter;

/// Decode a 2-picture (intra + 1-ref inter) stream and return the inter
/// frame's Y/U/V planes.
fn decode_inter_planes(stream: Vec<u8>) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("make decoder");
    let packet = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&packet).expect("send_packet");

    let _intra_frame = dec.receive_frame().expect("intra frame");
    let inter_frame = match dec.receive_frame().expect("inter frame") {
        Frame::Video(vf) => vf,
        other => panic!("expected video, got {other:?}"),
    };
    (
        inter_frame.planes[0].data.clone(),
        inter_frame.planes[1].data.clone(),
        inter_frame.planes[2].data.clone(),
    )
}

/// A residue config with a uniform `(cbx, cby)` codeblock grid on every
/// HL/LH/HH level (the LL band is always forced to a single codeblock by
/// the encoder).
fn residue_with_grid(
    wavelet: WaveletFilter,
    depth: u32,
    cbx: u32,
    cby: u32,
    mode: u32,
) -> ResidueParams {
    let mut rp = ResidueParams::default_for(wavelet, depth);
    rp.codeblocks = Some(vec![(cbx, cby); depth as usize + 1]);
    rp.codeblock_mode = mode;
    rp
}

fn plane_psnr(a: &[u8], b: &[u8]) -> f64 {
    let mut sse: u64 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = *x as i32 - *y as i32;
        sse += (d * d) as u64;
    }
    if sse == 0 {
        f64::INFINITY
    } else {
        let mse = sse as f64 / a.len() as f64;
        20.0 * (255.0f64).log10() - 10.0 * mse.log10()
    }
}

#[allow(clippy::type_complexity)]
fn pics() -> (
    [u8; 4096],
    [u8; 1024],
    [u8; 1024],
    [u8; 4096],
    [u8; 1024],
    [u8; 1024],
) {
    synthetic_translating_pair_64(4, 0)
}

/// Codeblock-partitioned residue (mode 0) at qindex=0 with LeGall 5/3.
/// On a depth-3 transform a 2x2 grid keeps every codeblock at least 4x4
/// samples (smallest level-3 band is 8x8 -> 4x4 codeblocks), so the
/// reversible residue round-trips the inter frame **bit-exactly** — the
/// codeblock skip / running-quantiser bookkeeping reconstructs
/// identically to the single-codeblock path.
#[test]
fn codeblock_mode0_q0_legall_bit_exact() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter_params = InterEncoderParams {
        residue: Some(residue_with_grid(WaveletFilter::LeGall5_3, 3, 2, 2, 0)),
        ..InterEncoderParams::default()
    };

    let (y0, u0, v0, y1, u1, v1) = pics();
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
    let stream = encode_intra_then_inter_stream(&seq, &intra_params, &inter_params, &intra, &inter);
    let (dy, du, dv) = decode_inter_planes(stream);

    assert_eq!(dy, y1.to_vec(), "Y plane not bit-exact (mode 0, 2x2 grid)");
    assert_eq!(du, u1.to_vec(), "U plane not bit-exact (mode 0, 2x2 grid)");
    assert_eq!(dv, v1.to_vec(), "V plane not bit-exact (mode 0, 2x2 grid)");
}

/// Codeblock mode 1 (per-codeblock differential quantiser offset) at
/// base qindex=0 must decode in lockstep. Mode 1 walks the
/// running-quantiser accumulator (§13.4.3.2) inside the non-skip branch;
/// the decoder applies the same accumulator. The per-codeblock `+1`
/// offset quantises later codeblocks harder, so this is not lossless —
/// but it must decode to a high-PSNR approximation (no desync).
#[test]
fn codeblock_mode1_q0_legall_decodes_high_psnr() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let inter_params = InterEncoderParams {
        residue: Some(residue_with_grid(WaveletFilter::LeGall5_3, 3, 2, 2, 1)),
        ..InterEncoderParams::default()
    };

    let (y0, u0, v0, y1, u1, v1) = pics();
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
    let stream = encode_intra_then_inter_stream(&seq, &intra_params, &inter_params, &intra, &inter);
    let (dy, _du, _dv) = decode_inter_planes(stream);

    let psnr = plane_psnr(&dy, &y1);
    assert!(
        psnr >= 40.0,
        "mode-1 codeblock residue Y PSNR {psnr:.2} dB below 40 dB — the \
         per-codeblock running quantiser decode must stay in lockstep"
    );
}

/// The codeblock-partitioned stream must differ from the
/// single-codeblock stream — confirming `spatial_partition_flag = 1`
/// plus the per-level grid + mode are actually emitted (more header
/// bytes) — while both still reconstruct the same bit-exact frame at q=0
/// mode 0 (the skip flags don't change the lossless reconstruction).
#[test]
fn codeblock_grid_changes_stream_but_not_reconstruction_at_q0() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);

    let flat = InterEncoderParams {
        residue: Some(ResidueParams::default_for(WaveletFilter::LeGall5_3, 3)),
        ..InterEncoderParams::default()
    };
    let partitioned = InterEncoderParams {
        residue: Some(residue_with_grid(WaveletFilter::LeGall5_3, 3, 2, 2, 0)),
        ..InterEncoderParams::default()
    };

    let (y0, u0, v0, y1, u1, v1) = pics();
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

    let flat_stream = encode_intra_then_inter_stream(&seq, &intra_params, &flat, &intra, &inter);
    let part_stream =
        encode_intra_then_inter_stream(&seq, &intra_params, &partitioned, &intra, &inter);

    assert_ne!(
        flat_stream, part_stream,
        "partitioned stream must differ from single-codeblock stream \
         (spatial_partition_flag + grid + mode emitted)"
    );

    let (fy, fu, fv) = decode_inter_planes(flat_stream);
    let (py, pu, pv) = decode_inter_planes(part_stream);
    assert_eq!(fy, py, "Y reconstruction must match flat path at q=0");
    assert_eq!(fu, pu, "U reconstruction must match flat path at q=0");
    assert_eq!(fv, pv, "V reconstruction must match flat path at q=0");
    assert_eq!(py, y1.to_vec(), "partitioned Y must be bit-exact at q=0");
}

/// A non-uniform per-level grid that keeps every codeblock at least 4x4
/// samples: the level-0 LL stays single, the finer levels split into a
/// 2x2 grid (level-1 32x32 -> 16x16, level-2 16x16 -> 8x8, level-3 8x8
/// -> 4x4). This is the realistic VC-2/Dirac shape — split the
/// high-energy detail levels only — and must round-trip bit-exactly at
/// q=0 mode 0 on the reversible filter.
#[test]
fn codeblock_per_level_grid_bit_exact() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let mut rp = ResidueParams::default_for(WaveletFilter::LeGall5_3, 3);
    rp.codeblocks = Some(vec![(1, 1), (2, 2), (2, 2), (2, 2)]);
    rp.codeblock_mode = 0;
    let inter_params = InterEncoderParams {
        residue: Some(rp),
        ..InterEncoderParams::default()
    };

    let (y0, u0, v0, y1, u1, v1) = pics();
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
    let stream = encode_intra_then_inter_stream(&seq, &intra_params, &inter_params, &intra, &inter);
    let (dy, du, dv) = decode_inter_planes(stream);

    assert_eq!(dy, y1.to_vec(), "Y not bit-exact (per-level grid)");
    assert_eq!(du, u1.to_vec(), "U not bit-exact (per-level grid)");
    assert_eq!(dv, v1.to_vec(), "V not bit-exact (per-level grid)");
}

/// Mode-1 differential quantiser with the per-level grid must also decode
/// in lockstep (no desync) and stay high-PSNR. Exercises the
/// running-quantiser accumulator across a multi-level codeblock grid.
#[test]
fn codeblock_per_level_grid_mode1_lockstep() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let mut rp = ResidueParams::default_for(WaveletFilter::LeGall5_3, 3);
    rp.codeblocks = Some(vec![(1, 1), (2, 2), (2, 2), (2, 2)]);
    rp.codeblock_mode = 1;
    let inter_params = InterEncoderParams {
        residue: Some(rp),
        ..InterEncoderParams::default()
    };

    let (y0, u0, v0, y1, u1, v1) = pics();
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
    let stream = encode_intra_then_inter_stream(&seq, &intra_params, &inter_params, &intra, &inter);
    let (dy, _, _) = decode_inter_planes(stream);

    let psnr = plane_psnr(&dy, &y1);
    assert!(
        psnr >= 40.0,
        "per-level mode-1 grid Y PSNR {psnr:.2} dB below 40 dB — \
         running-quantiser accumulator decode must stay in lockstep"
    );
}

/// **Sub-4×4-sample codeblocks are bit-exact since the round-382
/// §B.2.7.1 terminator fix.** A pathologically fine 8×8 grid on a
/// depth-3 transform of a 64×64 picture drives 1×1-sample codeblocks at
/// the deepest levels — the exact shape whose final AC symbols used to
/// land on the corrupted terminator tail and decode "near-lossless"
/// instead of exact. With `ArithEncoder::finish()` fixed, the qindex-0
/// LeGall 5/3 residue round-trips bit-exactly even on this grid, so the
/// old "every codeblock must be ≥ 4×4 samples" caveat is gone.
#[test]
fn codeblock_sub4x4_grid_q0_legall_bit_exact() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let mut rp = ResidueParams::default_for(WaveletFilter::LeGall5_3, 3);
    // 8×8 codeblocks per subband at every level: level-1 luma subbands
    // are 8×8 coefficients → 1×1-sample codeblocks; chroma is half that,
    // driving empty codeblocks too.
    rp.codeblocks = Some(vec![(1, 1), (8, 8), (8, 8), (8, 8)]);
    rp.codeblock_mode = 0;
    let inter_params = InterEncoderParams {
        residue: Some(rp),
        ..InterEncoderParams::default()
    };

    let (y0, u0, v0, y1, u1, v1) = pics();
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
    let stream = encode_intra_then_inter_stream(&seq, &intra_params, &inter_params, &intra, &inter);
    let (dy, du, dv) = decode_inter_planes(stream);

    assert_eq!(dy, y1.to_vec(), "Y plane not bit-exact (mode 0, 8x8 grid)");
    assert_eq!(du, u1.to_vec(), "U plane not bit-exact (mode 0, 8x8 grid)");
    assert_eq!(dv, v1.to_vec(), "V plane not bit-exact (mode 0, 8x8 grid)");
}
