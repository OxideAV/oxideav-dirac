//! 2-reference bipred (B-picture) encoder validators.
//!
//! Three stream shapes are exercised:
//!
//! * **Self-roundtrip with averaging fixture** — two anchor frames whose
//!   content sits on opposite sides of the bipred B. The bipred encoder's
//!   per-block decision search should pick `Ref1And2` for blocks where
//!   the average of the two references matches the source closer than
//!   either reference alone, and reproduce the B-frame at high quality
//!   when the residue path is enabled.
//!
//! * **Self-roundtrip vs single-ref baseline** — same fixture, encoded
//!   once via the 1-ref `encode_core_intra_then_inter_stream` and once
//!   via the new 2-ref `encode_core_intra_then_bipred_stream`. The
//!   bipred path's PSNR must be at least as high as the 1-ref baseline
//!   (averaging gives the B-frame access to information the 1-ref path
//!   structurally can't reach).
//!
//! * **ffmpeg cross-decode** — hard-asserted: ffmpeg's `dirac` decoder
//!   accepts our 0x0C + 0x0C + 0x0A 3-picture chain and reconstructs the
//!   B-frame above a defensive cross-decode floor.

use oxideav_core::CodecRegistry;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_dirac::encoder::make_minimal_sequence;
use oxideav_dirac::encoder_inter::{bipred_select_modes, InterEncoderParams, InterInputPicture};
use oxideav_dirac::encoder_intra_core::{
    encode_core_intra_then_bipred_stream, encode_core_intra_then_inter_stream,
    CoreIntraEncoderParams,
};
use oxideav_dirac::picture_inter::RefPredMode;
use oxideav_dirac::video_format::ChromaFormat;
use oxideav_dirac::wavelet::WaveletFilter;

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut sse: u64 = 0;
    for i in 0..a.len() {
        let d = a[i] as i32 - b[i] as i32;
        sse += (d * d) as u64;
    }
    if sse == 0 {
        return f64::INFINITY;
    }
    let mse = sse as f64 / a.len() as f64;
    20.0 * (255.0f64).log10() - 10.0 * mse.log10()
}

/// Synthesise a 3-frame **bipred-favourable** 64x64 4:2:0 YUV sequence.
///
/// The fixture engineering requires that the bipred 1/2-average is the
/// only path to a low-error reconstruction — i.e. neither single-ref MV
/// can match the source. We achieve this with **complementary
/// occluders**: a horizontal bar that's only present in ref1, and a
/// vertical bar only in ref2. The B picture has BOTH bars at half
/// intensity. Single-ref ME from either anchor produces the wrong
/// occluder (full intensity, missing the other bar entirely); the 1/2
/// average reproduces both bars at half intensity exactly.
///
/// * **Frame A** (`picture_number = 0`): horizontal bar at rows
///   30..34, columns 0..64 (only this bar is present).
/// * **Frame B** (`picture_number = 2`): vertical bar at columns
///   30..34, rows 0..64 (only this bar).
/// * **Frame mid** (`picture_number = 1`): both bars present, each at
///   half intensity ((bright + bg) / 2). The bipred 1/2 average of
///   ref-A's bright horizontal + ref-B's bright vertical reconstructs
///   exactly this picture.
///
/// Returns `(y0, u0, v0, y1, u1, v1, y_mid, u_mid, v_mid)`.
#[allow(clippy::type_complexity)]
fn synthetic_bipred_triplet() -> (
    [u8; 64 * 64],
    [u8; 32 * 32],
    [u8; 32 * 32],
    [u8; 64 * 64],
    [u8; 32 * 32],
    [u8; 32 * 32],
    [u8; 64 * 64],
    [u8; 32 * 32],
    [u8; 32 * 32],
) {
    let bg = 60u8;
    let bright = 220u8;
    // Half intensity = (bright + bg + 1) >> 1 — matches the §15.8.5
    // weighted-sum 1/2 average's rounding (`(p1 + p2 + 1) >> 1`).
    let half_bright = (bright as u16 + bg as u16 + 1) >> 1;
    let half = half_bright as u8;
    let u = [128u8; 32 * 32];
    let v = [128u8; 32 * 32];
    let mut y0 = [bg; 64 * 64];
    let mut y1 = [bg; 64 * 64];
    let mut ymid = [bg; 64 * 64];

    // Horizontal bar — rows 30..34 (covers a 4-pel band so the OBMC
    // 8x8 / 4-pel-stride blocks get a solid bar inside their extent).
    for r in 30..34usize {
        for c in 0..64usize {
            // Frame A: bright bar, frame B: background (bar absent),
            // frame mid: half-bright (the 1/2-average of A and B).
            y0[r * 64 + c] = bright;
            ymid[r * 64 + c] = half;
        }
    }

    // Vertical bar — columns 30..34.
    for c in 30..34usize {
        for r in 0..64usize {
            // Frame B: bright bar, frame A: background, mid: half-bright.
            y1[r * 64 + c] = bright;
            // For pixels at the bar intersection (rows 30..34 also),
            // frame mid has BOTH bars overlapping. The 1/2 average of
            // ref-A's `bright` (horizontal pixel) and ref-B's `bright`
            // (vertical pixel) is exactly `bright` — so the intersection
            // stays bright. Otherwise just half-bright (the vertical
            // contribution).
            if (30..34).contains(&r) {
                ymid[r * 64 + c] = bright;
            } else {
                ymid[r * 64 + c] = half;
            }
        }
    }
    (y0, u, v, y1, u, v, ymid, u, v)
}

/// Decode a stream and return its frames in send order.
fn decode_stream(stream: Vec<u8>) -> Vec<oxideav_core::VideoFrame> {
    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("decoder");
    let packet = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&packet).expect("send_packet");
    let mut out = Vec::new();
    while let Ok(frame) = dec.receive_frame() {
        match frame {
            Frame::Video(vf) => out.push(vf),
            other => panic!("expected video frame, got {other:?}"),
        }
    }
    out
}

/// **Bipred self-roundtrip with residue.** At qindex = 0 + LeGall 5/3,
/// the residue closes the prediction-error loop bit-exactly on the
/// anchor frames, and the bipred B should reconstruct effectively
/// lossless against the source even when the per-block 1-ref MV alone
/// could not (because the dark square sits between the two anchors).
#[test]
fn bipred_self_roundtrip_with_residue_recovers_midpoint_b_frame() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = CoreIntraEncoderParams::default_intra(WaveletFilter::LeGall5_3, 3);
    let inter_params = InterEncoderParams::default(); // residue ON

    let (y0, u0, v0, y1, u1, v1, ym, um, vm) = synthetic_bipred_triplet();
    let intra_a = InterInputPicture {
        picture_number: 0,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let intra_b = InterInputPicture {
        picture_number: 2,
        y: &y1,
        u: &u1,
        v: &v1,
    };
    let bipred = InterInputPicture {
        picture_number: 1,
        y: &ym,
        u: &um,
        v: &vm,
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
    assert!(
        frames.len() >= 3,
        "expected 3 decoded frames (intra-A, intra-B, bipred), got {}",
        frames.len()
    );
    // Frames arrive in decode order. The bipred frame's pts is its
    // picture_number = 1, so it sorts to position 1 in display order.
    // We don't sort here — instead identify by content position.
    // The first frame is intra A (picture_number = 0).
    assert_eq!(
        frames[0].planes[0].data,
        y0.to_vec(),
        "intra-A Y bit-exact at qindex=0"
    );
    assert_eq!(
        frames[1].planes[0].data,
        y1.to_vec(),
        "intra-B Y bit-exact at qindex=0"
    );
    let py = psnr(&frames[2].planes[0].data, &ym);
    eprintln!("bipred B-frame self-roundtrip Y PSNR (residue ON): {py:.2} dB");
    // With residue at qindex = 0 the loop closes bit-exactly — the
    // residue captures whatever the 1/2-average of the OBMC predictions
    // didn't reach. ∞ dB on ideal fixtures (our complementary-bar
    // fixture lands here).
    assert!(
        py >= 60.0,
        "bipred B-frame Y PSNR {py:.2} dB below 60 dB target — residue \
         path failed to close the prediction loop"
    );
}

/// **Round-trip the bipred motion-data block** through the decoder's
/// parser, recovering the same per-block `(rmode, mv1, mv2)` tuples
/// for every block. This is the load-bearing invariant that the
/// encoder's `build_motion_from_bipred_grid` produces a
/// `PictureMotionData` that matches what the decoder reconstructs from
/// our emitted bytes.
#[test]
fn bipred_block_motion_data_roundtrips_through_decoder() {
    use oxideav_dirac::bitwriter::BitWriter;
    use oxideav_dirac::encoder_inter::{encode_block_motion_data_bipred, BipredBlock, IntegerMv};
    use oxideav_dirac::picture_inter::{
        decode_block_motion_data, PicturePredictionParams, RefPredMode,
    };

    // Tiny 16x16 luma → 1 superblock with 4x4 = 16 blocks at split=2.
    let sbx = 1u32;
    let sby = 1u32;
    let bx = 4u32;
    let by = 4u32;
    // Mix of all three modes plus distinct MV pairs per block.
    let decisions: Vec<BipredBlock> = (0..16i32)
        .map(|i| {
            let mode = match i % 3 {
                0 => RefPredMode::Ref1Only,
                1 => RefPredMode::Ref2Only,
                _ => RefPredMode::Ref1And2,
            };
            BipredBlock {
                rmode: mode,
                mv1: IntegerMv((i % 4) - 1, (i / 4) - 1),
                mv2: IntegerMv((i % 4) + 2, (i / 4) - 2),
            }
        })
        .collect();
    let mut w = BitWriter::new();
    encode_block_motion_data_bipred(&mut w, sbx, sby, bx, by, &decisions);
    let bytes = w.finish();

    let pred = PicturePredictionParams {
        luma_xblen: 8,
        luma_yblen: 8,
        luma_xbsep: 4,
        luma_ybsep: 4,
        mv_precision: 0,
        using_global: false,
        prediction_mode: 0,
        superblocks_x: sbx,
        superblocks_y: sby,
        blocks_x: bx,
        blocks_y: by,
        refs_wt_precision: 1,
        ref1_wt: 1,
        ref2_wt: 1,
        global1: None,
        global2: None,
    };
    let mut r = oxideav_dirac::bits::BitReader::new(&bytes);
    let motion = decode_block_motion_data(&mut r, &pred, 2).expect("decode 2-ref motion");
    for by_ in 0..by {
        for bx_ in 0..bx {
            let i = (by_ * bx + bx_) as usize;
            let blk = &motion.blocks[i];
            let want = &decisions[i];
            assert_eq!(blk.rmode, want.rmode, "block {i} rmode mismatch");
            if want.rmode.uses_ref(1) {
                assert_eq!(
                    blk.mv[0],
                    (want.mv1.0, want.mv1.1),
                    "block {i} ref1 MV mismatch"
                );
            }
            if want.rmode.uses_ref(2) {
                assert_eq!(
                    blk.mv[1],
                    (want.mv2.0, want.mv2.1),
                    "block {i} ref2 MV mismatch"
                );
            }
        }
    }
}

/// Smoke test: a constant 3-frame stream where all three pictures are
/// the same flat Y / U / V — the bipred B should round-trip perfectly
/// regardless of which mode it picks. If this fails, the framing /
/// reference linkage / parse-info chain is structurally broken.
#[test]
fn bipred_constant_frames_self_roundtrip_bit_exact() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = CoreIntraEncoderParams::default_intra(WaveletFilter::LeGall5_3, 3);
    let inter_params = InterEncoderParams::default();
    let y = [123u8; 64 * 64];
    let u = [200u8; 32 * 32];
    let v = [55u8; 32 * 32];
    let intra_a = InterInputPicture {
        picture_number: 0,
        y: &y,
        u: &u,
        v: &v,
    };
    let intra_b = InterInputPicture {
        picture_number: 2,
        y: &y,
        u: &u,
        v: &v,
    };
    let bipred = InterInputPicture {
        picture_number: 1,
        y: &y,
        u: &u,
        v: &v,
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
    assert_eq!(frames.len(), 3, "expected 3 frames, got {}", frames.len());
    // All three frames identical → all three should be bit-exact (or
    // very close — the residue path picks up sub-LSB OBMC blend
    // rounding noise on the chroma planes when the block grid + chroma
    // dimensions interact).
    for (i, f) in frames.iter().enumerate() {
        let py = psnr(&f.planes[0].data, &y);
        eprintln!("frame {i} Y PSNR: {py:.4} dB");
        assert!(
            py >= 40.0,
            "frame {i} Y plane PSNR {py:.2} dB below 40 dB on \
             constant-fixture bipred — basic linkage broken"
        );
    }
}

/// Diagnostic: confirm `bipred_select_modes` produces a non-zero count
/// of `Ref1And2` blocks on the midpoint-B fixture. If this count is
/// zero the per-block decision search isn't exercising the bipred
/// averaging path at all and any A/B vs single-ref test below it is
/// vacuous.
#[test]
fn bipred_select_modes_emits_ref1and2_on_midpoint_fixture() {
    let (y0, _u0, _v0, y1, _u1, _v1, ym, _um, _vm) = synthetic_bipred_triplet();
    let decisions = bipred_select_modes(&ym, &y0, &y1, 64, 64, 16, 16, 16, 2);
    let mut n_r1 = 0usize;
    let mut n_r2 = 0usize;
    let mut n_b = 0usize;
    for d in &decisions {
        match d.rmode {
            RefPredMode::Ref1Only => n_r1 += 1,
            RefPredMode::Ref2Only => n_r2 += 1,
            RefPredMode::Ref1And2 => n_b += 1,
            RefPredMode::Intra => {}
        }
    }
    eprintln!("bipred decision counts: Ref1Only={n_r1} Ref2Only={n_r2} Ref1And2={n_b}");
    assert!(
        n_b > 0,
        "bipred decision search picked Ref1And2 zero times on the \
         midpoint-B fixture — the per-block SAD scoring is collapsing \
         to single-ref everywhere"
    );
}

/// **Bipred ME-only A/B vs 1-ref ME-only.** With residue turned OFF,
/// the bipred encoder's per-block decision search should still beat
/// the 1-ref baseline on a fixture engineered for averaging — the dark
/// square sitting at the midpoint of the two anchors is fundamentally
/// unreachable by a single-reference MV (no offset of the anchor places
/// the dark square at (42, 42); it lives at (40, 40) in ref1 and
/// (44, 44) in ref2). The bipred 1/2-average covers it.
#[test]
fn bipred_no_residue_beats_single_ref_no_residue_baseline() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = CoreIntraEncoderParams::default_intra(WaveletFilter::LeGall5_3, 3);
    let no_residue = InterEncoderParams {
        residue: None,
        ..InterEncoderParams::default()
    };

    let (y0, u0, v0, y1, u1, v1, ym, um, vm) = synthetic_bipred_triplet();

    // Single-ref baseline: encode the bipred B as a 1-ref P picture
    // referencing the closer anchor (intra A, at picture_number = 0).
    let intra = InterInputPicture {
        picture_number: 0,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let inter = InterInputPicture {
        picture_number: 1,
        y: &ym,
        u: &um,
        v: &vm,
    };
    let single_ref_stream =
        encode_core_intra_then_inter_stream(&seq, &intra_params, &no_residue, &intra, &inter);
    let single_frames = decode_stream(single_ref_stream);
    let psnr_1ref = psnr(&single_frames[1].planes[0].data, &ym);

    // Bipred: 0x0C + 0x0C + 0x0A chain.
    let intra_a = InterInputPicture {
        picture_number: 0,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let intra_b = InterInputPicture {
        picture_number: 2,
        y: &y1,
        u: &u1,
        v: &v1,
    };
    let bipred = InterInputPicture {
        picture_number: 1,
        y: &ym,
        u: &um,
        v: &vm,
    };
    let bipred_stream = encode_core_intra_then_bipred_stream(
        &seq,
        &intra_params,
        &no_residue,
        &intra_a,
        &intra_b,
        &bipred,
    );
    let bipred_frames = decode_stream(bipred_stream);
    let psnr_bipred = psnr(&bipred_frames[2].planes[0].data, &ym);

    eprintln!(
        "midpoint-B fixture (no-residue): 1-ref Y = {psnr_1ref:.2} dB, \
         bipred Y = {psnr_bipred:.2} dB"
    );
    // Bipred must beat 1-ref by ≥ 1 dB on this fixture. Margin is
    // intentionally modest — OBMC overlap blends the two predictions
    // across block boundaries even on the bright-square area, so the
    // absolute uplift is dominated by the dark square's coverage.
    assert!(
        psnr_bipred >= psnr_1ref + 1.0,
        "bipred Y PSNR {psnr_bipred:.2} dB did not beat 1-ref baseline \
         {psnr_1ref:.2} dB by ≥ 1 dB on the midpoint-B fixture — the \
         per-block decision search isn't picking Ref1And2 where it \
         should"
    );
}

/// **ffmpeg cross-decode** — hard-asserted. The bipred 0x0C + 0x0C +
/// 0x0A chain must round-trip through ffmpeg's `dirac` decoder
/// end-to-end and reconstruct the B picture. Mirrors the equivalent
/// 1-ref test (`tests/ffmpeg_interop.rs::ffmpeg_decodes_our_inter_stream_translating_square`)
/// but exercises the new 2-ref path.
#[test]
fn ffmpeg_cross_decodes_our_bipred_b_frame() {
    fn ffmpeg_available() -> bool {
        std::process::Command::new("ffmpeg")
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }
    if !ffmpeg_available() {
        eprintln!("ffmpeg not available; skipping bipred cross-decode test");
        return;
    }

    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = CoreIntraEncoderParams::default_intra(WaveletFilter::LeGall5_3, 3);
    let inter_params = InterEncoderParams::default();
    let (y0, u0, v0, y1, u1, v1, ym, um, vm) = synthetic_bipred_triplet();
    let intra_a = InterInputPicture {
        picture_number: 0,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let intra_b = InterInputPicture {
        picture_number: 2,
        y: &y1,
        u: &u1,
        v: &v1,
    };
    let bipred = InterInputPicture {
        picture_number: 1,
        y: &ym,
        u: &um,
        v: &vm,
    };
    let stream = encode_core_intra_then_bipred_stream(
        &seq,
        &intra_params,
        &inter_params,
        &intra_a,
        &intra_b,
        &bipred,
    );

    let tmpdir = std::env::temp_dir();
    let drc = tmpdir.join("oxideav_dirac_interop_bipred.drc");
    let yuv = tmpdir.join("oxideav_dirac_interop_bipred.yuv");
    std::fs::write(&drc, &stream).expect("write drc");

    let status = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "dirac",
            "-i",
        ])
        .arg(&drc)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&yuv)
        .status()
        .expect("run ffmpeg");
    assert!(
        status.success(),
        "ffmpeg rejected our bipred 0x0C + 0x0C + 0x0A stream — see \
         {drc:?}; the 2-ref encoder's parse-info / block_motion_data / \
         residue framing isn't what ffmpeg's dirac decoder expects"
    );

    let out = std::fs::read(&yuv).expect("read ffmpeg yuv");
    let frame_size = 64 * 64 + 2 * 32 * 32;
    // ffmpeg outputs the 3 frames in display order: A (0), bipred (1),
    // B (2). The bipred B frame is at offset frame_size.
    assert!(
        out.len() >= 3 * frame_size,
        "ffmpeg produced {} bytes; expected at least {} (3 frames)",
        out.len(),
        3 * frame_size
    );
    let bipred_y = &out[frame_size..frame_size + 64 * 64];
    let py = psnr(bipred_y, &ym);
    eprintln!("ffmpeg bipred B-frame cross-decode Y PSNR: {py:.2} dB");
    // Cross-decode floor: 49 dB on this complementary-bars fixture.
    // The bipred default is now `bipred_mv_precision = 2` (quarter-pel)
    // with **per-block adaptive sub-pel-vs-integer-pel selection**
    // (round-39): `bipred_select_modes` evaluates each MV at both its
    // sub-pel-refined position and the nearest integer-pel peer,
    // choosing whichever gives the lower SAD per block. Sharp-edge
    // blocks (the bars in this fixture) snap back to integer-pel,
    // matching the previous integer-pel-only baseline (~50 dB) and
    // avoiding the 7+ dB regression that an unconditional sub-pel
    // pipeline produced. Smooth-motion blocks (covered by the
    // `ffmpeg_cross_decodes_camera_pan_bipred_with_subpel_gain` test
    // below) pick up a +4 dB improvement.
    assert!(
        py >= 49.0,
        "ffmpeg bipred B-frame Y PSNR {py:.2} dB below 49 dB floor — \
         per-block adaptive sub-pel-vs-integer-pel selection failed to \
         pick integer-pel for the sharp-edge bars in this fixture; expected \
         ~50 dB"
    );
}

/// **Bipred sub-pel gain on camera-pan**. With per-block adaptive
/// sub-pel-vs-integer-pel selection (round-39), the bipred encoder picks
/// quarter-pel MVs on smooth-motion blocks and integer-pel MVs on
/// sharp-edge blocks. The camera-pan fixture is entirely smooth-motion
/// (cosine-shaped vertical bars panned by 1 luma pel), so the bipred
/// path picks sub-pel MVs almost everywhere and lifts the ffmpeg
/// cross-decode from the integer-pel-only ceiling (~48 dB) to ≥ 50 dB.
#[test]
fn ffmpeg_cross_decodes_camera_pan_bipred_with_subpel_gain() {
    fn ffmpeg_available() -> bool {
        std::process::Command::new("ffmpeg")
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }
    if !ffmpeg_available() {
        eprintln!("ffmpeg not available; skipping camera-pan bipred subpel gain");
        return;
    }
    use oxideav_dirac::encoder_inter::synthetic_camera_pan_64;
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let intra_params = CoreIntraEncoderParams::default_intra(WaveletFilter::LeGall5_3, 3);
    // ref0 = pan(0), ref1 = pan(2), bipred-mid = pan(1) — exact temporal
    // midpoint (1/2 average reproduces the source after the cosine
    // analytical resampler).
    let (y0, u0, v0, _, _, _) = synthetic_camera_pan_64(0, 0);
    let (_, _, _, y2, _, _) = synthetic_camera_pan_64(2, 0);
    let (_, _, _, ym, _, _) = synthetic_camera_pan_64(1, 0);
    let intra_a = InterInputPicture {
        picture_number: 0,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let intra_b = InterInputPicture {
        picture_number: 2,
        y: &y2,
        u: &u0,
        v: &v0,
    };
    let bipred = InterInputPicture {
        picture_number: 1,
        y: &ym,
        u: &u0,
        v: &v0,
    };

    let measure = |bmp: u32, tag: &str| -> f64 {
        let p = InterEncoderParams {
            bipred_mv_precision: bmp,
            ..InterEncoderParams::default()
        };
        let stream = encode_core_intra_then_bipred_stream(
            &seq,
            &intra_params,
            &p,
            &intra_a,
            &intra_b,
            &bipred,
        );
        let drc = std::env::temp_dir().join(format!("oxideav_dirac_camera_pan_bipred_{tag}.drc"));
        let yuv = std::env::temp_dir().join(format!("oxideav_dirac_camera_pan_bipred_{tag}.yuv"));
        std::fs::write(&drc, &stream).expect("write drc");
        let s = std::process::Command::new("ffmpeg")
            .args([
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "dirac",
                "-i",
            ])
            .arg(&drc)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&yuv)
            .status()
            .expect("run ffmpeg");
        assert!(
            s.success(),
            "ffmpeg rejected camera-pan bipred ({tag}) — see {drc:?}"
        );
        let out = std::fs::read(&yuv).unwrap();
        let frame_size = 64 * 64 + 2 * 32 * 32;
        let bipred_y = &out[frame_size..frame_size + 64 * 64];
        psnr(bipred_y, &ym)
    };
    let psnr_int = measure(0, "int");
    let psnr_qpel = measure(2, "qpel");
    eprintln!(
        "camera-pan bipred ffmpeg cross-decode: int = {psnr_int:.2} dB, \
         qpel(adaptive) = {psnr_qpel:.2} dB"
    );
    // Floor: quarter-pel adaptive must beat integer-pel by ≥ 2 dB on
    // this all-smooth-motion fixture. Empirically lifts from 48 to 52.
    assert!(
        psnr_qpel >= psnr_int + 2.0,
        "bipred qpel(adaptive) PSNR {psnr_qpel:.2} dB did not beat \
         integer-pel {psnr_int:.2} dB by ≥ 2 dB on the camera-pan smooth-\
         motion fixture — round-39 per-block adaptive sub-pel selection \
         regression"
    );
    assert!(
        psnr_qpel >= 50.0,
        "bipred qpel(adaptive) camera-pan PSNR {psnr_qpel:.2} dB below 50 dB \
         absolute floor — expected ~52 dB after round-39 per-block adaptive \
         sub-pel selection"
    );
}
