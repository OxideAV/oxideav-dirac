//! Encoder-side rate-control fuzz oracle for VC-2 HQ + LD profiles
//! (round-179).
//!
//! The decoder-side [`tests/decoder_fuzz_oracle.rs`] (round-165) drives
//! the `DiracDecoder` through corrupted byte streams and asserts it
//! never panics, never integer-overflows, never livelocks. This file is
//! its encoder-side analogue: the four rate-control variants
//! (`PerPicture`, `Cbr`, `Vbv`, `VbvHysteresis`) on the HQ and LD
//! sequence drivers form a small state machine whose inputs are
//! `target_bytes`, `buffer_bytes`, `max_drain_per_picture`, and a
//! per-picture pixel surface — all four of which are user-supplied and
//! all four of which have already proved capable of producing
//! integer-overflow / debug-assert panics on out-of-band values (see
//! the changelog around r165 for `quant_factor` clamping). This oracle
//! pins the rate-control surface against the same panic-/livelock-/
//! invariant-violation classes.
//!
//! Coverage:
//!
//! * **Cartesian sweep over rate-control variants + extreme target /
//!   buffer values.** Every (HQ/LD, mode, target, buffer, max_drain)
//!   combination encodes a 3-picture sequence and asserts:
//!     1. no panic / no debug-assert;
//!     2. each picture's `actual_payload_bytes > 0`;
//!     3. the stream is non-empty;
//!     4. it round-trips through the decoder to exactly 3 frames.
//! * **Vbv bucket invariant.** Under `Vbv { buffer_bytes }` and
//!   `VbvHysteresis { buffer_bytes, .. }`, every per-picture row's
//!   `running_surplus_bytes ≤ buffer_bytes` after the post-encode
//!   clamp (the contract documented on `HqPictureRate` / `LdPictureRate`).
//! * **Strict-generalisation invariants.** `Vbv { buffer_bytes: 0 }`,
//!   `VbvHysteresis { max_drain_per_picture: 0, .. }`, and
//!   `VbvHysteresis { buffer_bytes: B, max_drain_per_picture: B }` are
//!   each pinned byte-identical to their documented degenerate forms
//!   (`PerPicture`, `PerPicture`, and `Vbv { buffer_bytes: B }`
//!   respectively). These are documented r146/r149/r159 invariants;
//!   the oracle pins them across pathological-target sweeps so a future
//!   encoder change that subtly breaks one is caught here, not in
//!   production.
//! * **Pathological pixel inputs.** All-zero luma, all-`0xFF` luma,
//!   uniform-mid-grey, and tiny single-pixel-pulse frames — each fed
//!   through every variant. Tests that the picker handles "no
//!   coefficient energy" (q=0 fits trivially) and "minimum coefficient
//!   energy" (every coefficient at the extreme) without livelocking
//!   the `qindex ∈ floor..=127` walk.
//! * **Extreme integer inputs.** `target_bytes = 0`, `target_bytes = 1`,
//!   `target_bytes ∈ {100, 1_000, 10_000}` — each must produce a
//!   well-formed stream (target clamped up at the picker by the existing
//!   `1`-bytes floor on each request) without integer overflow. The
//!   sweep deliberately caps at 10_000 bytes per picture rather than
//!   `u32::MAX` because the LD path's per-picture allocation grows with
//!   the requested budget (`ld_picture_payload_bytes = header +
//!   slice_bytes_numer`); a multi-GiB request would OOM the test runner
//!   without testing any new code path that 10_000 doesn't already.
//! * **Empty / single-picture sequences.** Confirms the drivers handle
//!   the degenerate frame-count edge cases without dividing-by-zero
//!   the running surplus or wrapping the per-picture parse-info chain.
//!
//! Workspace policy: clean-room. No external library code consulted.
//! Spec authority for the bounds checked here is the BBC Dirac
//! Specification v2.2.3 §13.5.2 / §13.5.3.2 / §13.5.4 plus the r146 /
//! r149 / r152 / r159 documented invariants in this crate's CHANGELOG.

use oxideav_core::{CodecId, CodecParameters, CodecRegistry, Frame, Packet, TimeBase};
use oxideav_dirac::encoder::{
    encode_hq_sequence_with_size_target, encode_hq_sequence_with_size_target_report,
    encode_ld_sequence_with_size_target, encode_ld_sequence_with_size_target_report,
    make_minimal_sequence, make_minimal_sequence_ld, EncoderParams, HqRateControl, InputPicture,
    LdEncoderParams, LdRateControl,
};
use oxideav_dirac::video_format::ChromaFormat;
use oxideav_dirac::wavelet::WaveletFilter;

// -------------------------------------------------------------------
// Frame-builder helpers
// -------------------------------------------------------------------

/// One 64x64 4:2:0 frame seeded so successive frames differ. Mid-range
/// luma energy + chroma mid-grey so every variant has measurable cost
/// to quantise.
fn frame_64(seed: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; 64 * 64];
    let u = vec![128u8; 32 * 32];
    let v = vec![128u8; 32 * 32];
    for row in 0..64 {
        for col in 0..64 {
            y[row * 64 + col] = ((((row + col) as u32 * 5) + seed * 11) & 0xFF) as u8;
        }
    }
    (y, u, v)
}

/// 64x64 4:2:0 frame whose luma is identically `fill`. The picker hits
/// the lowest possible cost at q=0 on this input — the DWT carries no
/// AC energy, every band collapses to a single low-magnitude DC value,
/// and the picture lands at near-floor bytes regardless of qindex.
fn frame_64_solid(fill_y: u8) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    (
        vec![fill_y; 64 * 64],
        vec![128u8; 32 * 32],
        vec![128u8; 32 * 32],
    )
}

/// 64x64 frame with a single bright pixel at (cx, cy). Concentrated
/// high-frequency content; tests the picker against a coefficient
/// landscape that's the opposite of `frame_64_solid` — one extreme
/// coefficient and a near-zero background.
fn frame_64_pulse(cx: usize, cy: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![16u8; 64 * 64];
    y[cy * 64 + cx] = 255;
    (y, vec![128u8; 32 * 32], vec![128u8; 32 * 32])
}

/// Build N seeded 64x64 input pictures into owned planes. Returned as
/// a parallel `(planes, refs)` pair so callers can borrow `&[u8]`s out
/// of `planes` into `InputPicture<'_>`s in `refs`.
fn seeded_frames(n: usize) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    (0..n).map(|i| frame_64(i as u32)).collect()
}

/// Bundle owned planes into `InputPicture<'_>` slice. The lifetime
/// borrowing requires the caller to keep `planes` alive across the use.
fn as_inputs(planes: &[(Vec<u8>, Vec<u8>, Vec<u8>)]) -> Vec<InputPicture<'_>> {
    planes
        .iter()
        .enumerate()
        .map(|(i, (y, u, v))| InputPicture {
            picture_number: i as u32,
            y,
            u,
            v,
        })
        .collect()
}

// -------------------------------------------------------------------
// Decode-round-trip harness (panic-trap)
// -------------------------------------------------------------------

/// Drive a freshly-built dirac decoder over a stream, counting frames.
/// A clean error after the expected frame count is fine — we only care
/// that the decoder neither panics nor stops short of `expected_frames`.
fn decode_count(stream: &[u8]) -> usize {
    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("dirac decoder factory");
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream.to_vec());
    dec.send_packet(&pkt).expect("send_packet on fuzz stream");
    let _ = dec.flush();
    let mut frames = 0usize;
    // Loop ceiling well above the fixtures we encode here so a
    // hypothetical livelock surfaces as an explicit panic.
    for _ in 0..64 {
        match dec.receive_frame() {
            Ok(Frame::Video(_)) => frames += 1,
            Ok(_) => break,
            Err(_) => break,
        }
    }
    frames
}

// -------------------------------------------------------------------
// HQ — Cartesian sweep over rate-control variants × extreme targets
// -------------------------------------------------------------------

fn hq_default_params() -> EncoderParams {
    let mut p = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    // A small 2x2 slice grid keeps each picture under a few KiB so the
    // sweep stays fast.
    p.slices_x = 2;
    p.slices_y = 2;
    p
}

/// Every (mode, target_bytes) combination encodes the same 3-picture
/// fixture and must:
///   * produce a non-empty stream,
///   * have `actual_payload_bytes > 0` on every per-picture row,
///   * round-trip through the decoder to exactly 3 frames,
///   * never panic on the inside.
#[test]
fn hq_rate_control_sweep_never_panics() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let params = hq_default_params();
    let planes = seeded_frames(3);
    let inputs = as_inputs(&planes);

    // Pathological + reasonable targets. We deliberately cap at 10_000
    // bytes (≈ 2.5 KiB per slice on a 2x2 grid) because the LD path's
    // per-picture allocation is proportional to the requested budget
    // (`ld_picture_payload_bytes = header + slice_bytes_numer`) and a
    // `target = u32::MAX` request escalates to multi-GiB allocations
    // that OOM the test runner — see the docs-gap note in the round-179
    // CHANGELOG entry. HQ does NOT share this amplification (its picker
    // walks a fixed qindex range against an in-memory DWT pyramid that
    // does not scale with target), but we cap both sweeps at the same
    // budget for symmetry.
    let targets = [0u32, 1, 100, 1_000, 10_000];
    let buffers = [0u32, 1, 100, 10_000];
    let drains = [0u32, 1, 100, 10_000];

    // Track encoded shapes — handy for debugging if a panic ever fires.
    for &target in &targets {
        // PerPicture / Cbr: target only.
        for mode in [HqRateControl::PerPicture, HqRateControl::Cbr] {
            let (stream, report) =
                encode_hq_sequence_with_size_target_report(&seq, &params, &inputs, target, mode);
            assert!(
                !stream.is_empty(),
                "HQ {mode:?} target={target} empty stream"
            );
            assert_eq!(report.len(), 3);
            for row in &report {
                assert!(
                    row.actual_payload_bytes > 0,
                    "HQ {mode:?} target={target} zero payload"
                );
            }
            // Round-trip through the decoder.
            assert_eq!(
                decode_count(&stream),
                3,
                "HQ {mode:?} target={target} decode mismatch"
            );
        }
        // Vbv: target × buffer.
        for &buffer_bytes in &buffers {
            let mode = HqRateControl::Vbv { buffer_bytes };
            let (stream, report) =
                encode_hq_sequence_with_size_target_report(&seq, &params, &inputs, target, mode);
            assert!(!stream.is_empty());
            assert_eq!(report.len(), 3);
            for row in &report {
                assert!(row.actual_payload_bytes > 0);
                assert!(
                    row.running_surplus_bytes <= buffer_bytes as i64,
                    "HQ Vbv post-clamp invariant violated: surplus={} buffer={}",
                    row.running_surplus_bytes,
                    buffer_bytes
                );
            }
            assert_eq!(decode_count(&stream), 3);
        }
        // VbvHysteresis: target × buffer × drain (sample sparsely to
        // keep the sweep total under a few seconds).
        for &buffer_bytes in &buffers {
            for &max_drain_per_picture in &drains {
                let mode = HqRateControl::VbvHysteresis {
                    buffer_bytes,
                    max_drain_per_picture,
                };
                let (stream, report) = encode_hq_sequence_with_size_target_report(
                    &seq, &params, &inputs, target, mode,
                );
                assert!(!stream.is_empty());
                assert_eq!(report.len(), 3);
                for row in &report {
                    assert!(row.actual_payload_bytes > 0);
                    // Same post-clamp invariant as plain Vbv — the
                    // bucket fill rule is identical, only the spend
                    // rule changes.
                    assert!(
                        row.running_surplus_bytes <= buffer_bytes as i64,
                        "HQ VbvHysteresis post-clamp invariant violated: \
                         surplus={} buffer={} drain={}",
                        row.running_surplus_bytes,
                        buffer_bytes,
                        max_drain_per_picture,
                    );
                }
                assert_eq!(decode_count(&stream), 3);
            }
        }
    }
}

// -------------------------------------------------------------------
// LD — Cartesian sweep over rate-control variants × extreme targets
// -------------------------------------------------------------------

fn ld_default_params() -> LdEncoderParams {
    // 2x2 slices × 64 bytes/slice = 256 bytes total budget — plenty for
    // 64x64 fixtures at qindex=0.
    LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 2, 2, 64)
}

#[test]
fn ld_rate_control_sweep_never_panics() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let params = ld_default_params();
    let planes = seeded_frames(3);
    let inputs = as_inputs(&planes);

    // See the HQ sweep above for the rationale on the 10_000-byte
    // ceiling: LD picture payload bytes are `header + slice_bytes_numer`
    // so a `target = u32::MAX` request escalates to a multi-GiB
    // allocation. The cap keeps the sweep tractable while still
    // covering every variant × target-shape combination meaningfully.
    let targets = [0u32, 1, 100, 1_000, 10_000];
    let buffers = [0u32, 1, 100, 10_000];
    let drains = [0u32, 1, 100, 10_000];

    for &target in &targets {
        for mode in [LdRateControl::PerPicture, LdRateControl::Cbr] {
            let (stream, report) =
                encode_ld_sequence_with_size_target_report(&seq, &params, &inputs, target, mode);
            assert!(
                !stream.is_empty(),
                "LD {mode:?} target={target} empty stream"
            );
            assert_eq!(report.len(), 3);
            for row in &report {
                assert!(
                    row.actual_payload_bytes > 0,
                    "LD {mode:?} target={target} zero payload"
                );
            }
            assert_eq!(
                decode_count(&stream),
                3,
                "LD {mode:?} target={target} decode mismatch"
            );
        }
        for &buffer_bytes in &buffers {
            let mode = LdRateControl::Vbv { buffer_bytes };
            let (stream, report) =
                encode_ld_sequence_with_size_target_report(&seq, &params, &inputs, target, mode);
            assert!(!stream.is_empty());
            assert_eq!(report.len(), 3);
            for row in &report {
                assert!(row.actual_payload_bytes > 0);
                assert!(
                    row.running_surplus_bytes <= buffer_bytes as i64,
                    "LD Vbv post-clamp invariant violated: surplus={} buffer={}",
                    row.running_surplus_bytes,
                    buffer_bytes
                );
            }
            assert_eq!(decode_count(&stream), 3);
        }
        for &buffer_bytes in &buffers {
            for &max_drain_per_picture in &drains {
                let mode = LdRateControl::VbvHysteresis {
                    buffer_bytes,
                    max_drain_per_picture,
                };
                let (stream, report) = encode_ld_sequence_with_size_target_report(
                    &seq, &params, &inputs, target, mode,
                );
                assert!(!stream.is_empty());
                assert_eq!(report.len(), 3);
                for row in &report {
                    assert!(row.actual_payload_bytes > 0);
                    assert!(
                        row.running_surplus_bytes <= buffer_bytes as i64,
                        "LD VbvHysteresis post-clamp invariant violated: \
                         surplus={} buffer={} drain={}",
                        row.running_surplus_bytes,
                        buffer_bytes,
                        max_drain_per_picture,
                    );
                }
                assert_eq!(decode_count(&stream), 3);
            }
        }
    }
}

// -------------------------------------------------------------------
// Strict-generalisation invariants
// -------------------------------------------------------------------

/// HQ: `Vbv { buffer_bytes: 0 }` produces a byte-identical stream to
/// `PerPicture` (documented in r146 — the bucket cap forbids spending
/// any saved bytes, so the request equals `target` every picture, just
/// like PerPicture). Pinned across a range of targets so a future
/// encoder change that subtly breaks the equivalence is caught here.
#[test]
fn hq_vbv_zero_buffer_equiv_perpicture() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let params = hq_default_params();
    let planes = seeded_frames(4);
    let inputs = as_inputs(&planes);
    for &target in &[200u32, 1_000, 10_000] {
        let baseline = encode_hq_sequence_with_size_target(
            &seq,
            &params,
            &inputs,
            target,
            HqRateControl::PerPicture,
        );
        let vbv_zero = encode_hq_sequence_with_size_target(
            &seq,
            &params,
            &inputs,
            target,
            HqRateControl::Vbv { buffer_bytes: 0 },
        );
        assert_eq!(
            baseline, vbv_zero,
            "HQ Vbv{{buffer=0}} must be byte-identical to PerPicture (target={target})"
        );
    }
}

/// HQ: `VbvHysteresis { max_drain_per_picture: 0, .. }` ≡ `PerPicture`.
/// r159 invariant: a zero drain rate forbids the picture from spending
/// any saved bytes regardless of bucket fill, so the per-picture
/// request collapses to `target` every picture.
#[test]
fn hq_vbv_hysteresis_zero_drain_equiv_perpicture() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let params = hq_default_params();
    let planes = seeded_frames(4);
    let inputs = as_inputs(&planes);
    for &target in &[200u32, 1_000, 10_000] {
        let baseline = encode_hq_sequence_with_size_target(
            &seq,
            &params,
            &inputs,
            target,
            HqRateControl::PerPicture,
        );
        for &buffer_bytes in &[0u32, 100, 1_000_000] {
            let hys = encode_hq_sequence_with_size_target(
                &seq,
                &params,
                &inputs,
                target,
                HqRateControl::VbvHysteresis {
                    buffer_bytes,
                    max_drain_per_picture: 0,
                },
            );
            assert_eq!(
                baseline, hys,
                "HQ VbvHysteresis{{drain=0, buffer={buffer_bytes}}} ≢ PerPicture (target={target})"
            );
        }
    }
}

/// HQ: `VbvHysteresis { buffer_bytes: B, max_drain_per_picture: D }`
/// with `D >= B` is byte-identical to plain `Vbv { buffer_bytes: B }`.
/// r159: when the drain cap is at least as large as the bucket, the
/// `min(carry, buffer, drain)` collapses to `min(carry, buffer)` —
/// exactly the Vbv spend rule.
#[test]
fn hq_vbv_hysteresis_drain_ge_buffer_equiv_vbv() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let params = hq_default_params();
    let planes = seeded_frames(4);
    let inputs = as_inputs(&planes);
    for &target in &[200u32, 1_000, 10_000] {
        for &buffer_bytes in &[10u32, 100, 1_000] {
            let plain_vbv = encode_hq_sequence_with_size_target(
                &seq,
                &params,
                &inputs,
                target,
                HqRateControl::Vbv { buffer_bytes },
            );
            // drain == buffer: borderline-inert.
            let hys_eq = encode_hq_sequence_with_size_target(
                &seq,
                &params,
                &inputs,
                target,
                HqRateControl::VbvHysteresis {
                    buffer_bytes,
                    max_drain_per_picture: buffer_bytes,
                },
            );
            // drain > buffer: cap definitely inert.
            let hys_gt = encode_hq_sequence_with_size_target(
                &seq,
                &params,
                &inputs,
                target,
                HqRateControl::VbvHysteresis {
                    buffer_bytes,
                    max_drain_per_picture: buffer_bytes.saturating_add(1_000),
                },
            );
            assert_eq!(plain_vbv, hys_eq);
            assert_eq!(plain_vbv, hys_gt);
        }
    }
}

/// LD: same `Vbv { buffer_bytes: 0 }` ≡ `PerPicture` invariant as HQ
/// (r149) — pin it across a few targets.
#[test]
fn ld_vbv_zero_buffer_equiv_perpicture() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let params = ld_default_params();
    let planes = seeded_frames(4);
    let inputs = as_inputs(&planes);
    for &target in &[400u32, 1_000, 10_000] {
        let baseline = encode_ld_sequence_with_size_target(
            &seq,
            &params,
            &inputs,
            target,
            LdRateControl::PerPicture,
        );
        let vbv_zero = encode_ld_sequence_with_size_target(
            &seq,
            &params,
            &inputs,
            target,
            LdRateControl::Vbv { buffer_bytes: 0 },
        );
        assert_eq!(
            baseline, vbv_zero,
            "LD Vbv{{buffer=0}} must be byte-identical to PerPicture (target={target})"
        );
    }
}

/// LD: `VbvHysteresis { max_drain_per_picture: 0, .. }` ≡ `PerPicture`
/// (r159).
#[test]
fn ld_vbv_hysteresis_zero_drain_equiv_perpicture() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let params = ld_default_params();
    let planes = seeded_frames(4);
    let inputs = as_inputs(&planes);
    for &target in &[400u32, 1_000, 10_000] {
        let baseline = encode_ld_sequence_with_size_target(
            &seq,
            &params,
            &inputs,
            target,
            LdRateControl::PerPicture,
        );
        for &buffer_bytes in &[0u32, 100, 1_000_000] {
            let hys = encode_ld_sequence_with_size_target(
                &seq,
                &params,
                &inputs,
                target,
                LdRateControl::VbvHysteresis {
                    buffer_bytes,
                    max_drain_per_picture: 0,
                },
            );
            assert_eq!(
                baseline, hys,
                "LD VbvHysteresis{{drain=0, buffer={buffer_bytes}}} ≢ PerPicture (target={target})"
            );
        }
    }
}

// -------------------------------------------------------------------
// Pathological pixel inputs
// -------------------------------------------------------------------

/// All-zero / all-0xFF / mid-grey / single-pulse luma frames through
/// every HQ variant. The picker's `qindex ∈ floor..=127` walk must
/// terminate, even on "no AC energy" (q=0 trivially fits) and "extreme
/// concentrated AC energy" (one coefficient dominates).
#[test]
fn hq_pathological_pixels_never_panic() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let params = hq_default_params();
    let pix_planes: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = vec![
        frame_64_solid(0),
        frame_64_solid(128),
        frame_64_solid(255),
        frame_64_pulse(0, 0),
        frame_64_pulse(31, 31),
        frame_64_pulse(63, 63),
    ];
    for plane in &pix_planes {
        let one = std::slice::from_ref(plane);
        let inputs = as_inputs(one);
        // Sweep a handful of modes per pixel input — the all-pairs
        // (modes × targets) coverage is in the Cartesian sweep above;
        // here we just confirm pathological *pixel* surfaces don't
        // perturb any individual variant.
        for mode in [
            HqRateControl::PerPicture,
            HqRateControl::Cbr,
            HqRateControl::Vbv { buffer_bytes: 100 },
            HqRateControl::VbvHysteresis {
                buffer_bytes: 100,
                max_drain_per_picture: 50,
            },
        ] {
            let stream = encode_hq_sequence_with_size_target(&seq, &params, &inputs, 1_000, mode);
            assert!(
                !stream.is_empty(),
                "HQ pathological pixel {mode:?} empty stream"
            );
            assert_eq!(decode_count(&stream), 1);
        }
    }
}

/// LD analogue of the pathological-pixel pass.
#[test]
fn ld_pathological_pixels_never_panic() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let params = ld_default_params();
    let pix_planes: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = vec![
        frame_64_solid(0),
        frame_64_solid(128),
        frame_64_solid(255),
        frame_64_pulse(0, 0),
        frame_64_pulse(31, 31),
        frame_64_pulse(63, 63),
    ];
    for plane in &pix_planes {
        let one = std::slice::from_ref(plane);
        let inputs = as_inputs(one);
        for mode in [
            LdRateControl::PerPicture,
            LdRateControl::Cbr,
            LdRateControl::Vbv { buffer_bytes: 100 },
            LdRateControl::VbvHysteresis {
                buffer_bytes: 100,
                max_drain_per_picture: 50,
            },
        ] {
            let stream = encode_ld_sequence_with_size_target(&seq, &params, &inputs, 1_000, mode);
            assert!(
                !stream.is_empty(),
                "LD pathological pixel {mode:?} empty stream"
            );
            assert_eq!(decode_count(&stream), 1);
        }
    }
}

// -------------------------------------------------------------------
// Empty / single-picture sequences
// -------------------------------------------------------------------

/// HQ with zero input pictures emits a well-formed `seq-header + EOS`
/// container (no picture units) and decodes to zero frames. The report
/// must be empty (no rows produced).
#[test]
fn hq_empty_frames_sequence() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let params = hq_default_params();
    let frames: Vec<InputPicture<'_>> = Vec::new();
    for mode in [
        HqRateControl::PerPicture,
        HqRateControl::Cbr,
        HqRateControl::Vbv { buffer_bytes: 100 },
        HqRateControl::VbvHysteresis {
            buffer_bytes: 100,
            max_drain_per_picture: 50,
        },
    ] {
        let (stream, report) =
            encode_hq_sequence_with_size_target_report(&seq, &params, &frames, 1_000, mode);
        assert!(
            report.is_empty(),
            "HQ {mode:?} empty input expected empty report"
        );
        // Stream still contains the sequence header + EOS bracket.
        assert!(
            !stream.is_empty(),
            "HQ {mode:?} empty input stream missing seq-header"
        );
    }
}

/// LD analogue of the empty-sequence pass.
#[test]
fn ld_empty_frames_sequence() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let params = ld_default_params();
    let frames: Vec<InputPicture<'_>> = Vec::new();
    for mode in [
        LdRateControl::PerPicture,
        LdRateControl::Cbr,
        LdRateControl::Vbv { buffer_bytes: 100 },
        LdRateControl::VbvHysteresis {
            buffer_bytes: 100,
            max_drain_per_picture: 50,
        },
    ] {
        let (stream, report) =
            encode_ld_sequence_with_size_target_report(&seq, &params, &frames, 1_000, mode);
        assert!(report.is_empty());
        assert!(!stream.is_empty());
    }
}

// -------------------------------------------------------------------
// Cumulative surplus identity
// -------------------------------------------------------------------

/// Mode-agnostic identity from the r152 documentation: the i-th row's
/// `running_surplus_bytes`, *before* the VBV bucket clamp, equals
/// `(i+1) * target_bytes - Σ_{j ≤ i} actual_payload_bytes[j]`.
///
/// PerPicture and Cbr modes don't clamp the running surplus, so the
/// reported value should match the cumulative-budget identity exactly
/// regardless of how target / qindex interact. We pin both modes here
/// for HQ + LD on a 5-picture run with reasonable target sizing.
#[test]
fn hq_surplus_identity_perpicture_cbr() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let params = hq_default_params();
    let planes = seeded_frames(5);
    let inputs = as_inputs(&planes);
    let target = 1_500u32;
    for mode in [HqRateControl::PerPicture, HqRateControl::Cbr] {
        let (_stream, report) =
            encode_hq_sequence_with_size_target_report(&seq, &params, &inputs, target, mode);
        let mut sum_actual: i64 = 0;
        for (i, row) in report.iter().enumerate() {
            sum_actual += row.actual_payload_bytes as i64;
            let expected_surplus = (i as i64 + 1) * target as i64 - sum_actual;
            assert_eq!(
                row.running_surplus_bytes, expected_surplus,
                "HQ {mode:?} surplus identity broken at picture {i}: \
                 reported={} expected={} (target={target}, sum_actual={})",
                row.running_surplus_bytes, expected_surplus, sum_actual,
            );
        }
    }
}

/// LD analogue of the surplus-identity pin.
#[test]
fn ld_surplus_identity_perpicture_cbr() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let params = ld_default_params();
    let planes = seeded_frames(5);
    let inputs = as_inputs(&planes);
    let target = 1_500u32;
    for mode in [LdRateControl::PerPicture, LdRateControl::Cbr] {
        let (_stream, report) =
            encode_ld_sequence_with_size_target_report(&seq, &params, &inputs, target, mode);
        let mut sum_actual: i64 = 0;
        for (i, row) in report.iter().enumerate() {
            sum_actual += row.actual_payload_bytes as i64;
            let expected_surplus = (i as i64 + 1) * target as i64 - sum_actual;
            assert_eq!(
                row.running_surplus_bytes, expected_surplus,
                "LD {mode:?} surplus identity broken at picture {i}",
            );
        }
    }
}
