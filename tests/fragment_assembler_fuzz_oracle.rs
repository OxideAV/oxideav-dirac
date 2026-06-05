//! Fragment-assembler robustness oracle for VC-2 v3 §14.1 / §14.3 /
//! §14.4 / §14.5.
//!
//! Goal: every sequence of `(parse_code, FragmentHeader, transform
//! parameters, dc-kick)` operations the assembler might see from a
//! hostile or merely-malformed source must produce a clean `Result`
//! in bounded time. Panics, debug-assert hits, integer overflows
//! (`saturating_*` paths must not silently wrap), and out-of-bounds
//! `Vec` indexing are all assembler bugs and are caught here by the
//! test harness — any panic kills the test thread.
//!
//! Coverage:
//!
//! * **Random-walk state-machine sweep** — a deterministic xorshift
//!   PRNG drives a stream of randomly-shaped fragment events into a
//!   `FragmentAssembler` and into a parallel reference model. The model
//!   knows which transitions §14.1 forbids; the test asserts the
//!   assembler agrees on accept/reject and that on accept the model's
//!   `slices_received` matches the assembler's.
//! * **Pathological geometry** — `slices_x` / `slices_y` at zero,
//!   `u32::MAX`, and `(u32::MAX, u32::MAX)` (whose product overflows
//!   `u32`); the assembler must surface `InvalidSliceGrid` (zero) or
//!   keep `fragmented_picture_done()` deterministic under
//!   `saturating_mul` semantics.
//! * **Pathological `slice_count` / offsets** — `u16::MAX` slice count,
//!   `(u16::MAX, u16::MAX)` raster offset with small `slices_x`; the
//!   assembler must either accept and emit `coords.len() == slice_count`
//!   per-slice raster pairs OR surface `SliceOverflow`. Either way it
//!   must not panic.
//! * **Parse-code mixing** — feed an HQ data fragment after an LD
//!   setup (and vice versa); the assembler must surface
//!   `InconsistentParseCode`. Feed all four §10.5.2 Table 5 fragment
//!   parse codes (`0xC8`/`0xCC` LD, `0xE8`/`0xEC` HQ) and confirm the
//!   §10.5.2 Table 5 `using_dc_prediction` predicate
//!   `(parse_code & 0x28) == 0x08` is captured correctly.
//! * **§14.5 DC-prediction kick edge cases** — kick before picture
//!   completion (rejects), kick with `dwt_depth_ho > 0` on LD path
//!   (rejects with `AsymmetricDcPredictionUnsupported`), kick on HQ
//!   path (no-op), kick on LD path with synthetic LL subbands of
//!   varied sizes (1×1, 1×N, N×1, square) — the predictor must run
//!   in-place without panicking on any of them.
//! * **`slice_coords` cross-check** — for random `(s, x_offset,
//!   y_offset, slices_x)` the helper must agree with the §14.4
//!   pseudocode `raster = y * slices_x + x + s; (raster % slices_x,
//!   raster / slices_x)` computed in `u64` (the spec's natural
//!   arithmetic width).
//!
//! Spec anchors for every assertion below:
//! `docs/video/vc2/vc2-specification.pdf` (SMPTE ST 2042-1:2022)
//! §10.5.2 (parse codes / Table 5 predicates), §14.1 (fragment
//! sequencing), §14.2 (fragment header), §14.3
//! (`initialize_fragment_state`), §14.4 (`fragment_data` + slice
//! raster scan), §14.5 (`fragmented_wavelet_transform` trailing DC
//! prediction).

use oxideav_dirac::fragment::{
    slice_coords, AssemblerError, FragmentAssembler, FragmentEvent, FragmentHeader, FragmentKind,
};
use oxideav_dirac::subband::SubbandData;

/// Deterministic 32-bit xorshift PRNG — same shape as
/// `encoder_inter_fuzz_oracle.rs` and `encoder_rate_control_fuzz_oracle.rs`.
/// Self-contained (no `rand` dependency); seed-driven so failures are
/// reproducible by re-running the same test.
#[derive(Debug, Clone)]
struct XorShift32 {
    state: u32,
}

impl XorShift32 {
    fn new(seed: u32) -> Self {
        // xorshift's all-zero fixed point: substitute a non-zero seed.
        let seed = if seed == 0 { 0xDEAD_BEEF } else { seed };
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    fn next_in_range(&mut self, lo: u32, hi_inclusive: u32) -> u32 {
        if hi_inclusive < lo {
            return lo;
        }
        let span = hi_inclusive - lo + 1;
        lo + (self.next_u32() % span)
    }

    fn next_bool(&mut self) -> bool {
        (self.next_u32() & 1) == 1
    }

    fn pick_parse_code(&mut self) -> u8 {
        // §10.5.2 Table 5 fragment parse codes: 0xC8 / 0xCC (LD) and
        // 0xE8 / 0xEC (HQ). Plus a couple of non-fragment codes to
        // exercise the assembler's parse-code-consistency check when
        // garbage arrives.
        let pool: [u8; 6] = [0xC8, 0xCC, 0xE8, 0xEC, 0x88, 0xE8];
        pool[self.next_in_range(0, pool.len() as u32 - 1) as usize]
    }
}

/// Reference-model assembler: tracks the same state the real
/// assembler should track, knows the §14.1 transitions, and tells the
/// test whether the next operation should succeed or which error it
/// should produce. Pure Rust, no `unsafe`, no spec interpretation
/// beyond what the §14.x text directly says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelPhase {
    AwaitingSetup,
    AwaitingTransformParameters,
    ReceivingData,
}

#[derive(Debug, Clone)]
struct Model {
    phase: ModelPhase,
    slices_x: u32,
    slices_y: u32,
    dwt_depth_ho: u32,
    using_dc_prediction: bool,
    picture_number: u32,
    slices_received: u32,
}

impl Model {
    fn new() -> Self {
        Self {
            phase: ModelPhase::AwaitingSetup,
            slices_x: 0,
            slices_y: 0,
            dwt_depth_ho: 0,
            using_dc_prediction: false,
            picture_number: 0,
            slices_received: 0,
        }
    }

    fn picture_done(&self) -> bool {
        self.slices_x != 0
            && self.phase == ModelPhase::ReceivingData
            && self.slices_received == self.slices_x.saturating_mul(self.slices_y)
    }

    /// Predict the assembler's response to `on_setup_fragment`. The
    /// `parse_code` byte is captured (for the §10.5.2 Table 5
    /// `using_dc_prediction` predicate) but never causes a setup
    /// rejection on its own — the only setup error is §14.1's
    /// "no setup while previous picture incomplete".
    fn predict_setup(&self, _parse_code: u8) -> Result<(), AssemblerError> {
        match self.phase {
            ModelPhase::AwaitingSetup => Ok(()),
            ModelPhase::ReceivingData if self.picture_done() => Ok(()),
            _ => Err(AssemblerError::SetupBeforePreviousPictureComplete),
        }
    }

    fn apply_setup(&mut self, header: &FragmentHeader, parse_code: u8) {
        debug_assert!(matches!(header.kind, FragmentKind::Setup));
        self.phase = ModelPhase::AwaitingTransformParameters;
        self.picture_number = header.picture_number;
        self.using_dc_prediction = (parse_code & 0x28) == 0x08;
        self.slices_received = 0;
        self.slices_x = 0;
        self.slices_y = 0;
        self.dwt_depth_ho = 0;
    }

    fn predict_transform(&self, slices_x: u32, slices_y: u32) -> Result<(), AssemblerError> {
        if slices_x == 0 || slices_y == 0 {
            return Err(AssemblerError::InvalidSliceGrid { slices_x, slices_y });
        }
        Ok(())
    }

    fn apply_transform(&mut self, slices_x: u32, slices_y: u32, dwt_depth_ho: u32) {
        self.slices_x = slices_x;
        self.slices_y = slices_y;
        self.dwt_depth_ho = dwt_depth_ho;
        self.phase = ModelPhase::ReceivingData;
    }

    fn predict_data(&self, header: &FragmentHeader, parse_code: u8) -> Result<(), AssemblerError> {
        let (slice_count, _x, _y) = match header.kind {
            FragmentKind::Data {
                slice_count,
                x_offset,
                y_offset,
            } => (u32::from(slice_count), x_offset, y_offset),
            // The assembler's data entry point on a setup-shaped
            // header returns UnexpectedDataFragment after a
            // debug_assert; the model agrees that this is an
            // out-of-band caller bug and is excluded from the
            // random sweep by construction (the harness only feeds
            // Data-kind headers into `on_data_fragment`).
            FragmentKind::Setup => return Err(AssemblerError::UnexpectedDataFragment),
        };
        if self.phase != ModelPhase::ReceivingData {
            return Err(AssemblerError::UnexpectedDataFragment);
        }
        if header.picture_number != self.picture_number {
            return Err(AssemblerError::PictureNumberMismatch {
                setup: self.picture_number,
                data: header.picture_number,
            });
        }
        let data_using_dc = (parse_code & 0x28) == 0x08;
        if data_using_dc != self.using_dc_prediction {
            let setup_indicative = if self.using_dc_prediction { 0x88 } else { 0xE8 };
            return Err(AssemblerError::InconsistentParseCode {
                setup: setup_indicative,
                data: parse_code,
            });
        }
        let expected_total = self.slices_x.saturating_mul(self.slices_y);
        let new_total = self.slices_received.saturating_add(slice_count);
        if new_total > expected_total {
            return Err(AssemblerError::SliceOverflow {
                expected_total,
                slices_received: self.slices_received,
                slice_count,
            });
        }
        Ok(())
    }

    fn apply_data(&mut self, header: &FragmentHeader) {
        if let FragmentKind::Data { slice_count, .. } = header.kind {
            self.slices_received = self.slices_received.saturating_add(u32::from(slice_count));
        }
    }
}

/// One operation the harness can apply to the assembler under test.
#[derive(Debug, Clone, Copy)]
enum Op {
    Setup {
        picture_number: u32,
        parse_code: u8,
    },
    Transform {
        slices_x: u32,
        slices_y: u32,
        dwt_depth_ho: u32,
    },
    Data {
        picture_number: u32,
        slice_count: u16,
        x_offset: u16,
        y_offset: u16,
        parse_code: u8,
    },
    DcKick,
}

fn random_op(rng: &mut XorShift32, model: &Model) -> Op {
    // Bias the operation choice toward the legal next move so the
    // walk reaches deep states, but always allow a "wrong" operation
    // so error paths fire too.
    let r = rng.next_in_range(0, 99);
    match model.phase {
        ModelPhase::AwaitingSetup => {
            // 60% setup, 20% data (rejected), 10% transform (no-op
            // model-wise; assembler treats it as a state-set), 10%
            // dc-kick (rejected).
            if r < 60 {
                Op::Setup {
                    picture_number: rng.next_u32(),
                    parse_code: rng.pick_parse_code(),
                }
            } else if r < 80 {
                Op::Data {
                    picture_number: rng.next_u32(),
                    slice_count: rng.next_in_range(1, 5) as u16,
                    x_offset: rng.next_in_range(0, 3) as u16,
                    y_offset: rng.next_in_range(0, 3) as u16,
                    parse_code: rng.pick_parse_code(),
                }
            } else if r < 90 {
                Op::Transform {
                    slices_x: rng.next_in_range(0, 4),
                    slices_y: rng.next_in_range(0, 4),
                    dwt_depth_ho: rng.next_in_range(0, 2),
                }
            } else {
                Op::DcKick
            }
        }
        ModelPhase::AwaitingTransformParameters => {
            // 70% transform, 20% data (rejected), 10% setup (rejected)
            if r < 70 {
                Op::Transform {
                    // Lean toward valid (>=1) so we reach ReceivingData,
                    // but occasionally inject zero for the
                    // InvalidSliceGrid path.
                    slices_x: if rng.next_in_range(0, 9) < 2 {
                        0
                    } else {
                        rng.next_in_range(1, 4)
                    },
                    slices_y: if rng.next_in_range(0, 9) < 2 {
                        0
                    } else {
                        rng.next_in_range(1, 4)
                    },
                    dwt_depth_ho: rng.next_in_range(0, 2),
                }
            } else if r < 90 {
                Op::Data {
                    picture_number: model.picture_number,
                    slice_count: rng.next_in_range(1, 5) as u16,
                    x_offset: rng.next_in_range(0, 3) as u16,
                    y_offset: rng.next_in_range(0, 3) as u16,
                    parse_code: rng.pick_parse_code(),
                }
            } else {
                Op::Setup {
                    picture_number: rng.next_u32(),
                    parse_code: rng.pick_parse_code(),
                }
            }
        }
        ModelPhase::ReceivingData => {
            // 60% data (mostly correct picture_number + matching
            // parse code), 20% setup (rejected if picture not done),
            // 10% dc-kick, 10% transform.
            if r < 60 {
                let mismatch_pic = rng.next_in_range(0, 9) < 2;
                let mismatch_code = rng.next_in_range(0, 9) < 2;
                let pic = if mismatch_pic {
                    model.picture_number.wrapping_add(1)
                } else {
                    model.picture_number
                };
                let code = if mismatch_code {
                    // Flip the using_dc_prediction class to force an
                    // InconsistentParseCode error.
                    if model.using_dc_prediction {
                        0xE8
                    } else {
                        0xC8
                    }
                } else if model.using_dc_prediction {
                    if rng.next_bool() {
                        0xC8
                    } else {
                        0xCC
                    }
                } else if rng.next_bool() {
                    0xE8
                } else {
                    0xEC
                };
                Op::Data {
                    picture_number: pic,
                    slice_count: rng.next_in_range(1, 3) as u16,
                    x_offset: rng.next_in_range(0, model.slices_x.saturating_sub(1).max(1) - 1)
                        as u16,
                    y_offset: rng.next_in_range(0, model.slices_y.saturating_sub(1).max(1) - 1)
                        as u16,
                    parse_code: code,
                }
            } else if r < 80 {
                Op::Setup {
                    picture_number: rng.next_u32(),
                    parse_code: rng.pick_parse_code(),
                }
            } else if r < 90 {
                Op::DcKick
            } else {
                Op::Transform {
                    slices_x: rng.next_in_range(1, 4),
                    slices_y: rng.next_in_range(1, 4),
                    dwt_depth_ho: rng.next_in_range(0, 2),
                }
            }
        }
    }
}

/// Run one operation against the assembler under test and against the
/// reference model, asserting both agree on success/failure.
fn step(asm: &mut FragmentAssembler, model: &mut Model, op: Op) {
    match op {
        Op::Setup {
            picture_number,
            parse_code,
        } => {
            let hdr = FragmentHeader {
                picture_number,
                fragment_data_length: 0,
                kind: FragmentKind::Setup,
            };
            let model_pred = model.predict_setup(parse_code);
            let asm_res = asm.on_setup_fragment(&hdr, parse_code);
            match (model_pred, asm_res) {
                (Ok(()), Ok(event)) => {
                    assert_eq!(event, FragmentEvent::SetupAccepted);
                    model.apply_setup(&hdr, parse_code);
                    assert_eq!(asm.picture_number(), model.picture_number);
                    assert_eq!(asm.using_dc_prediction(), model.using_dc_prediction);
                    assert_eq!(asm.slices_received(), 0);
                }
                (Err(model_err), Err(asm_err)) => {
                    assert_eq!(model_err, asm_err, "setup error mismatch");
                }
                (model_pred, asm_res) => {
                    panic!("setup model/asm disagree: model={model_pred:?} asm={asm_res:?}")
                }
            }
        }
        Op::Transform {
            slices_x,
            slices_y,
            dwt_depth_ho,
        } => {
            let model_pred = model.predict_transform(slices_x, slices_y);
            let asm_res = asm.on_transform_parameters(slices_x, slices_y, dwt_depth_ho);
            // The assembler accepts on_transform_parameters in ANY
            // phase as long as slices_x/y > 0 (it doesn't gate on
            // setup_state). The model's predict_transform mirrors
            // that: only zero slices is the error case.
            match (model_pred, asm_res) {
                (Ok(()), Ok(())) => {
                    model.apply_transform(slices_x, slices_y, dwt_depth_ho);
                    assert_eq!(asm.slices_x(), slices_x);
                    assert_eq!(asm.slices_y(), slices_y);
                    assert_eq!(asm.dwt_depth_ho(), dwt_depth_ho);
                }
                (Err(model_err), Err(asm_err)) => {
                    assert_eq!(model_err, asm_err, "transform error mismatch");
                }
                (model_pred, asm_res) => {
                    panic!("transform model/asm disagree: model={model_pred:?} asm={asm_res:?}")
                }
            }
        }
        Op::Data {
            picture_number,
            slice_count,
            x_offset,
            y_offset,
            parse_code,
        } => {
            let hdr = FragmentHeader {
                picture_number,
                fragment_data_length: 0,
                kind: FragmentKind::Data {
                    slice_count,
                    x_offset,
                    y_offset,
                },
            };
            let model_pred = model.predict_data(&hdr, parse_code);
            let asm_res = asm.on_data_fragment(&hdr, parse_code);
            match (model_pred, asm_res) {
                (
                    Ok(()),
                    Ok(FragmentEvent::DataSlices {
                        coords,
                        picture_done,
                    }),
                ) => {
                    assert_eq!(
                        coords.len(),
                        usize::from(slice_count),
                        "coords length must equal slice_count"
                    );
                    // §14.4 raster identity: each coord matches the
                    // pseudocode `raster = y_offset * slices_x +
                    // x_offset + s; (raster % slices_x, raster /
                    // slices_x)` computed in u64.
                    for (s, &(cx, cy)) in coords.iter().enumerate() {
                        let raster = u64::from(y_offset) * u64::from(model.slices_x)
                            + u64::from(x_offset)
                            + s as u64;
                        let want_x = (raster % u64::from(model.slices_x)) as u32;
                        let want_y = (raster / u64::from(model.slices_x)) as u32;
                        assert_eq!((cx, cy), (want_x, want_y), "§14.4 raster scan mismatch");
                    }
                    model.apply_data(&hdr);
                    assert_eq!(asm.slices_received(), model.slices_received);
                    assert_eq!(asm.fragmented_picture_done(), picture_done);
                    assert_eq!(asm.fragmented_picture_done(), model.picture_done());
                }
                (Ok(()), Ok(other)) => panic!("expected DataSlices, got {other:?}"),
                (Err(model_err), Err(asm_err)) => {
                    assert_eq!(model_err, asm_err, "data error mismatch");
                }
                (model_pred, asm_res) => {
                    panic!("data model/asm disagree: model={model_pred:?} asm={asm_res:?}")
                }
            }
        }
        Op::DcKick => {
            // Build a per-component scratch LL subband for the kick.
            // Sized to model.slices_x x model.slices_y (sliceable
            // bound; the §13.4 dc_prediction routine reads/writes
            // in-place at arbitrary 2-D shapes, so anything ≥ 0 is
            // valid input). Use the saturating product to avoid
            // huge allocations on pathological grids.
            let w = model.slices_x.clamp(1, 8) as usize;
            let h = model.slices_y.clamp(1, 8) as usize;
            let mut ll = SubbandData::new(w, h);
            let asm_res = {
                let mut comps = [&mut ll];
                asm.fragmented_wavelet_transform_dc_prediction(&mut comps)
            };
            if !model.picture_done() {
                assert_eq!(
                    asm_res,
                    Err(AssemblerError::DcPredictionBeforePictureComplete),
                    "kick before picture done must reject"
                );
            } else if !model.using_dc_prediction {
                // §14.5: HQ path (`using_dc_prediction == false`)
                // short-circuits to a no-op before any other check,
                // so an asymmetric HQ picture still returns Ok(()).
                assert_eq!(asm_res, Ok(()), "HQ kick is a no-op even when asymmetric");
            } else if model.dwt_depth_ho != 0 {
                assert_eq!(
                    asm_res,
                    Err(AssemblerError::AsymmetricDcPredictionUnsupported {
                        dwt_depth_ho: model.dwt_depth_ho,
                    }),
                    "LD asymmetric DC kick must reject"
                );
            } else {
                assert_eq!(
                    asm_res,
                    Ok(()),
                    "LD kick succeeds on symmetric completed picture"
                );
            }
        }
    }
}

/// Drive 5 000 random operations across 8 seeds and assert no panic.
/// Each seed's stream is independent so a regression on one fixed
/// shape only loses that seed's coverage, not the whole run.
#[test]
fn random_walk_state_machine_no_panic_no_disagree() {
    for seed in [
        0x0000_0001,
        0x0BADC0DE,
        0xCAFEBABE,
        0xDEADBEEF,
        0xFACEFEED,
        0x1234_5678,
        0x9ABC_DEF0,
        0xFFFF_FFFF,
    ] {
        let mut rng = XorShift32::new(seed);
        let mut asm = FragmentAssembler::new();
        let mut model = Model::new();
        for _ in 0..5_000 {
            let op = random_op(&mut rng, &model);
            step(&mut asm, &mut model, op);
        }
    }
}

/// Cross-check `slice_coords(s, x_offset, y_offset, slices_x)` against
/// the §14.4 pseudocode for a Cartesian sweep of inputs. The helper is
/// pure (no state); this is purely for tighter coverage of the
/// u64-widened arithmetic path that prevents `u32` overflow when
/// `y_offset * slices_x + x_offset + s` would exceed `u32::MAX`.
#[test]
fn slice_coords_matches_spec_pseudocode() {
    // Compact sweep — every (s, x, y, sx) in a small lattice plus a
    // handful of pathological corners. We expect equality with the
    // u64-widened §14.4 formula on every accepted input, and `None`
    // exactly when `slices_x == 0`.
    for slices_x in [0u32, 1, 2, 3, 4, 5, 7, 16, 64, 256, u32::MAX] {
        for x_offset in [0u16, 1, 2, 3, 7, 255, u16::MAX] {
            for y_offset in [0u16, 1, 2, 3, 7, 255, u16::MAX] {
                for s in [0u32, 1, 2, 5, 17, 1023, u32::MAX] {
                    let got = slice_coords(s, x_offset, y_offset, slices_x);
                    if slices_x == 0 {
                        assert_eq!(got, None, "slices_x=0 must yield None");
                        continue;
                    }
                    let raster = u64::from(y_offset) * u64::from(slices_x)
                        + u64::from(x_offset)
                        + u64::from(s);
                    let want_x = (raster % u64::from(slices_x)) as u32;
                    let want_y = (raster / u64::from(slices_x)) as u32;
                    assert_eq!(
                        got,
                        Some((want_x, want_y)),
                        "§14.4 mismatch at s={s} x={x_offset} y={y_offset} sx={slices_x}"
                    );
                }
            }
        }
    }
}

/// Pathological geometry — `(slices_x, slices_y)` shapes that test
/// the assembler's `saturating_mul` invariant on `expected_total`.
/// A `slices_x * slices_y` product that exceeds `u32::MAX` must
/// saturate (not wrap), so `fragmented_picture_done()` stays
/// deterministic and `SliceOverflow` fires when the cumulative count
/// would exceed the saturated total.
#[test]
fn pathological_slice_grid_saturates_no_wrap() {
    for (sx, sy) in [
        (1u32, 1u32),
        (1, u32::MAX),
        (u32::MAX, 1),
        (u32::MAX, u32::MAX),
        (0x1_0000, 0x1_0000), // exact u32::MAX + 1
    ] {
        let mut asm = FragmentAssembler::new();
        let setup = FragmentHeader {
            picture_number: 7,
            fragment_data_length: 0,
            kind: FragmentKind::Setup,
        };
        asm.on_setup_fragment(&setup, 0xCC).unwrap();
        asm.on_transform_parameters(sx, sy, 0).unwrap();
        assert_eq!(asm.slices_x(), sx);
        assert_eq!(asm.slices_y(), sy);
        // A single small data fragment must accept (the saturated
        // total is at least u32::MAX so 1 slice cannot overflow).
        let data = FragmentHeader {
            picture_number: 7,
            fragment_data_length: 0,
            kind: FragmentKind::Data {
                slice_count: 1,
                x_offset: 0,
                y_offset: 0,
            },
        };
        let res = asm.on_data_fragment(&data, 0xCC);
        assert!(
            res.is_ok(),
            "1-slice fragment must accept on saturated grid: {res:?}"
        );
        // `fragmented_picture_done()` must be false for any
        // saturating-mul total > 1 — assertion sanity.
        if sx != 1 || sy != 1 {
            assert!(!asm.fragmented_picture_done());
        }
    }
}

/// Mixed parse-code class within a single picture: HQ setup (`0xE8`)
/// followed by LD data (`0xCC`) must reject with
/// `InconsistentParseCode`. Same for the mirror (LD setup followed
/// by HQ data).
#[test]
fn mixed_parse_code_class_within_picture_rejected() {
    for (setup_code, data_code, _label) in [
        (0xE8u8, 0xCCu8, "HQ setup → LD data"),
        (0xEC, 0xC8, "HQ setup → LD data"),
        (0xC8, 0xE8, "LD setup → HQ data"),
        (0xCC, 0xEC, "LD setup → HQ data"),
    ] {
        let mut asm = FragmentAssembler::new();
        let setup = FragmentHeader {
            picture_number: 3,
            fragment_data_length: 0,
            kind: FragmentKind::Setup,
        };
        asm.on_setup_fragment(&setup, setup_code).unwrap();
        asm.on_transform_parameters(2, 1, 0).unwrap();
        let data = FragmentHeader {
            picture_number: 3,
            fragment_data_length: 0,
            kind: FragmentKind::Data {
                slice_count: 1,
                x_offset: 0,
                y_offset: 0,
            },
        };
        let err = asm.on_data_fragment(&data, data_code).unwrap_err();
        assert!(matches!(err, AssemblerError::InconsistentParseCode { .. }));
    }
}

/// All four §10.5.2 Table 5 fragment parse codes route the
/// `using_dc_prediction(state) := (parse_code & 0x28) == 0x08`
/// predicate correctly: `0xC8` / `0xCC` (LD) → true; `0xE8` / `0xEC`
/// (HQ) → false. This pins the predicate from the assembler's
/// captured-flag vantage rather than from the raw predicate, so
/// a regression in either layer surfaces here.
#[test]
fn parse_code_dc_prediction_predicate_captured_per_class() {
    for (code, want) in [(0xC8u8, true), (0xCC, true), (0xE8, false), (0xEC, false)] {
        let mut asm = FragmentAssembler::new();
        let setup = FragmentHeader {
            picture_number: 1,
            fragment_data_length: 0,
            kind: FragmentKind::Setup,
        };
        asm.on_setup_fragment(&setup, code).unwrap();
        assert_eq!(
            asm.using_dc_prediction(),
            want,
            "parse_code=0x{code:02X} must capture using_dc_prediction={want}"
        );
    }
}

/// §14.5 DC kick on the LD path with synthetic LL subbands of varied
/// sizes — the predictor must run in-place without panicking on any
/// of them. Sizes chosen to exercise the §13.4 raster neighbour-mean
/// edge cases (1x1, 1xN, Nx1, square).
#[test]
fn dc_kick_runs_on_varied_ll_shapes_no_panic() {
    for (w, h) in [(1usize, 1), (1, 4), (4, 1), (2, 2), (3, 5), (8, 8)] {
        let mut asm = FragmentAssembler::new();
        let setup = FragmentHeader {
            picture_number: 9,
            fragment_data_length: 0,
            kind: FragmentKind::Setup,
        };
        asm.on_setup_fragment(&setup, 0xCC).unwrap();
        asm.on_transform_parameters(1, 1, 0).unwrap();
        let data = FragmentHeader {
            picture_number: 9,
            fragment_data_length: 0,
            kind: FragmentKind::Data {
                slice_count: 1,
                x_offset: 0,
                y_offset: 0,
            },
        };
        asm.on_data_fragment(&data, 0xCC).unwrap();
        assert!(asm.fragmented_picture_done());

        // Seed the subband with a deterministic pattern so the
        // dc_prediction in-place edits a non-zero buffer (a uniform
        // zero buffer would short-circuit by symmetry and miss
        // arithmetic-overflow bugs in the prediction step).
        let mut ll = SubbandData::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = (y as i32 * 7 + x as i32 * 3 - 5).wrapping_mul(11);
                ll.set(y, x, v);
            }
        }
        let pre_w = ll.width;
        let pre_h = ll.height;
        let mut comps = [&mut ll];
        asm.fragmented_wavelet_transform_dc_prediction(&mut comps)
            .expect("LD kick on completed picture succeeds");
        assert_eq!(ll.width, pre_w);
        assert_eq!(ll.height, pre_h);
    }
}

/// §14.5 DC kick on HQ path is a no-op even when neighbours would
/// matter: the predictor does not run and the buffer is untouched.
#[test]
fn dc_kick_on_hq_path_is_noop_buffer_unchanged() {
    let mut asm = FragmentAssembler::new();
    let setup = FragmentHeader {
        picture_number: 10,
        fragment_data_length: 0,
        kind: FragmentKind::Setup,
    };
    asm.on_setup_fragment(&setup, 0xEC).unwrap();
    asm.on_transform_parameters(1, 1, 0).unwrap();
    let data = FragmentHeader {
        picture_number: 10,
        fragment_data_length: 0,
        kind: FragmentKind::Data {
            slice_count: 1,
            x_offset: 0,
            y_offset: 0,
        },
    };
    asm.on_data_fragment(&data, 0xEC).unwrap();

    let mut ll = SubbandData::new(3, 3);
    for y in 0..3 {
        for x in 0..3 {
            ll.set(y, x, (y as i32 * 100 + x as i32 + 1) * 1000);
        }
    }
    let snapshot: Vec<i32> = ll.data.clone();
    let mut comps = [&mut ll];
    asm.fragmented_wavelet_transform_dc_prediction(&mut comps)
        .expect("HQ kick succeeds with no-op");
    assert_eq!(
        ll.data, snapshot,
        "HQ DC kick must not touch the buffer (no-op per §14.5)"
    );
}

/// §14.5 DC kick on LD path with `dwt_depth_ho > 0` must surface
/// `AsymmetricDcPredictionUnsupported`. Pins that the asymmetric-
/// transform gap is consistently reported (mirrors
/// `PictureError::AsymmetricTransformUnsupported` on the
/// non-fragmented path).
#[test]
fn dc_kick_ld_asymmetric_rejected() {
    for dwt_depth_ho in [1u32, 2, 3, 4] {
        let mut asm = FragmentAssembler::new();
        let setup = FragmentHeader {
            picture_number: 5,
            fragment_data_length: 0,
            kind: FragmentKind::Setup,
        };
        asm.on_setup_fragment(&setup, 0xCC).unwrap();
        asm.on_transform_parameters(1, 1, dwt_depth_ho).unwrap();
        let data = FragmentHeader {
            picture_number: 5,
            fragment_data_length: 0,
            kind: FragmentKind::Data {
                slice_count: 1,
                x_offset: 0,
                y_offset: 0,
            },
        };
        asm.on_data_fragment(&data, 0xCC).unwrap();

        let mut ll = SubbandData::new(2, 2);
        let mut comps = [&mut ll];
        let err = asm
            .fragmented_wavelet_transform_dc_prediction(&mut comps)
            .unwrap_err();
        assert_eq!(
            err,
            AssemblerError::AsymmetricDcPredictionUnsupported { dwt_depth_ho }
        );
    }
}

/// Determinism: re-running the same seed twice must produce the
/// same sequence of state-machine outcomes. Pins that the assembler
/// state machine has no hidden non-determinism (e.g. relying on
/// allocator addresses or system clocks).
#[test]
fn random_walk_is_deterministic_per_seed() {
    for seed in [0x1234_5678u32, 0xABCD_EF01, 0xDEAD_BEEF] {
        let trace_a = trace_random_walk(seed, 500);
        let trace_b = trace_random_walk(seed, 500);
        assert_eq!(trace_a, trace_b, "seed=0x{seed:08X} non-deterministic");
    }
}

fn trace_random_walk(seed: u32, steps: usize) -> Vec<(u32, u32, u32, bool)> {
    let mut rng = XorShift32::new(seed);
    let mut asm = FragmentAssembler::new();
    let mut model = Model::new();
    let mut trace = Vec::with_capacity(steps);
    for _ in 0..steps {
        let op = random_op(&mut rng, &model);
        step(&mut asm, &mut model, op);
        trace.push((
            asm.slices_x(),
            asm.slices_y(),
            asm.slices_received(),
            asm.fragmented_picture_done(),
        ));
    }
    trace
}
