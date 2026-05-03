//! Integration tests against the docs/video/dirac/ fixture corpus.
//!
//! Each fixture under `../../docs/video/dirac/fixtures/<name>/` ships an
//! `input.drc` (raw Dirac elementary stream — `BBCD` parse-info chain),
//! an `expected.yuv` byte-for-byte ground truth produced by an
//! instrumented FFmpeg `diracdec` decoder, a `notes.md` describing the
//! bitstream feature focus, and a `trace.txt` (or `trace.txt.gz` when
//! large) capturing the per-step PARSE_UNIT / SEQUENCE / PICTURE /
//! CODE_BLOCK / MOTION events the reference decoder emitted. The
//! `expected.yuv` is the ground-truth pixel target.
//!
//! This driver decodes every fixture through the in-tree
//! [`DiracDecoder`] and reports the per-fixture pixel-match rate +
//! per-plane PSNR against the expected YUV.
//!
//! Acceptance:
//! * `Tier::BitExact` — must round-trip exactly. Failure = CI red.
//! * `Tier::ReportOnly` — divergence is logged but the test does NOT
//!   fail. All fixtures land here at first commit; promote individual
//!   cases to `BitExact` as the in-tree Dirac decoder closes the
//!   underlying gap (LD profile, OBMC inter, 4:2:2, interlaced,
//!   per-picture wavelet switch, …).
//!
//! Workspace policy: NO external library code (libschroedinger,
//! libdirac, libavcodec, …) was consulted while writing this driver.
//! Spec authority: SMPTE ST 2042-1 / -2 (VC-2) and the Dirac BBC
//! specification. The `trace.txt` files are an aid for the human
//! implementer when localising divergences — the driver references
//! the trace path in the `eprintln!` header so a failing run prints a
//! clickable pointer.

use std::fs;
use std::path::PathBuf;

use oxideav_core::{CodecId, Decoder, Error, Frame, Packet, TimeBase};
use oxideav_dirac::decoder::DiracDecoder;

/// Locate `docs/video/dirac/fixtures/<name>/`. Tests run with CWD set
/// to the crate root, so we walk two levels up to reach the workspace
/// root and then into `docs/`.
fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from("../../docs/video/dirac/fixtures").join(name)
}

/// Per-frame decode result against the per-frame slice of
/// `expected.yuv`. Counters are in bytes — the corpus is 8-bit
/// throughout, and our 8-bit decode path produces one byte per
/// sample with stride==width, so byte == sample comparisons line up.
#[derive(Default)]
struct FrameDiff {
    y_total: usize,
    y_exact: usize,
    y_max: i32,
    y_sse: u64,
    uv_total: usize,
    uv_exact: usize,
    uv_max: i32,
    uv_sse: u64,
}

impl FrameDiff {
    fn pct(&self) -> f64 {
        let exact = self.y_exact + self.uv_exact;
        let total = self.y_total + self.uv_total;
        if total == 0 {
            0.0
        } else {
            exact as f64 / total as f64 * 100.0
        }
    }

    fn merge(&mut self, other: &FrameDiff) {
        self.y_total += other.y_total;
        self.y_exact += other.y_exact;
        self.y_max = self.y_max.max(other.y_max);
        self.y_sse += other.y_sse;
        self.uv_total += other.uv_total;
        self.uv_exact += other.uv_exact;
        self.uv_max = self.uv_max.max(other.uv_max);
        self.uv_sse += other.uv_sse;
    }

    /// PSNR in dB for the luma plane against the 8-bit reference.
    /// Returns `f64::INFINITY` when the plane is bit-exact.
    fn y_psnr(&self) -> f64 {
        psnr(self.y_sse, self.y_total)
    }

    /// PSNR in dB for the combined chroma planes against the 8-bit
    /// reference. Returns `f64::INFINITY` when the planes are
    /// bit-exact.
    fn uv_psnr(&self) -> f64 {
        psnr(self.uv_sse, self.uv_total)
    }
}

fn psnr(sse: u64, n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    if sse == 0 {
        return f64::INFINITY;
    }
    let mse = sse as f64 / n as f64;
    10.0 * (255.0_f64 * 255.0 / mse).log10()
}

/// Compare a single 8-bit plane of our output against the reference.
/// The reference is in row-packed natural order (stride == width); our
/// 8-bit decoder produces the same shape. Returns `(n, exact, max,
/// sse)`.
fn diff_plane(our: &[u8], refp: &[u8]) -> (usize, usize, i32, u64) {
    let mut ex = 0usize;
    let mut max = 0i32;
    let mut sse: u64 = 0;
    let n = our.len().min(refp.len());
    for i in 0..n {
        let d = (our[i] as i32 - refp[i] as i32).abs();
        if d == 0 {
            ex += 1;
        }
        if d > max {
            max = d;
        }
        sse += (d as u64) * (d as u64);
    }
    (n, ex, max, sse)
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // BitExact unused at first commit; promote fixtures over time.
enum Tier {
    /// Must decode bit-exactly. Test fails on any divergence.
    BitExact,
    /// Decode is permitted to diverge from the reference; the
    /// per-fixture stats are logged but the test does not fail.
    /// Promote to `BitExact` once the underlying gap is closed.
    ReportOnly,
}

#[derive(Clone, Copy, Debug)]
enum Subsampling {
    /// 4:2:0 — chroma planes are half-width × half-height of luma.
    Yuv420,
    /// 4:2:2 — chroma planes are half-width × full-height of luma.
    Yuv422,
}

impl Subsampling {
    fn chroma_dims(&self, w: usize, h: usize) -> (usize, usize) {
        match self {
            Subsampling::Yuv420 => (w.div_ceil(2), h.div_ceil(2)),
            Subsampling::Yuv422 => (w.div_ceil(2), h),
        }
    }
    fn frame_bytes(&self, w: usize, h: usize) -> usize {
        let (cw, ch) = self.chroma_dims(w, h);
        w * h + 2 * cw * ch
    }
}

struct CorpusCase {
    name: &'static str,
    width: usize,
    height: usize,
    /// Number of frames stored back-to-back in `expected.yuv`. Note
    /// that for inter fixtures the reference decoder's display order
    /// may differ from our decode order — see the per-fixture comment.
    n_frames: usize,
    sub: Subsampling,
    tier: Tier,
}

struct DecodeReport {
    per_frame: Vec<Result<FrameDiff, String>>,
    visible_produced: usize,
    /// Top-level error (if `send_packet` or `receive_frame` returned
    /// something non-`NeedMore`). Recorded for the report; does NOT
    /// stop subsequent drains.
    fatal: Option<String>,
}

fn decode_fixture(case: &CorpusCase) -> Option<DecodeReport> {
    let dir = fixture_dir(case.name);
    let drc_path = dir.join("input.drc");
    let yuv_path = dir.join("expected.yuv");
    let trace_path = dir.join("trace.txt");
    let trace_gz_path = dir.join("trace.txt.gz");
    let drc = match fs::read(&drc_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "skip {}: missing {} ({e}). docs/ corpus is in the workspace \
                 umbrella repo; the standalone crate checkout has no fixtures.",
                case.name,
                drc_path.display()
            );
            return None;
        }
    };
    let yuv_ref = match fs::read(&yuv_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip {}: missing {} ({e})", case.name, yuv_path.display());
            return None;
        }
    };
    let trace_for_log = if trace_path.exists() {
        trace_path.display().to_string()
    } else {
        trace_gz_path.display().to_string()
    };
    eprintln!(
        "fixture {}: drc={} bytes, expected.yuv={} bytes, trace={}",
        case.name,
        drc.len(),
        yuv_ref.len(),
        trace_for_log,
    );

    let frame_size = case.sub.frame_bytes(case.width, case.height);
    assert_eq!(
        yuv_ref.len(),
        case.n_frames * frame_size,
        "fixture {} expected.yuv size mismatch (have {} bytes, expected {} = {} frames * {})",
        case.name,
        yuv_ref.len(),
        case.n_frames * frame_size,
        case.n_frames,
        frame_size,
    );

    // Dirac elementary stream — feed the entire .drc as a single
    // packet. The decoder's `scan()` walks the parse-info chain,
    // queues every picture data unit, and produces one VideoFrame per
    // `receive_frame` call.
    let mut dec = DiracDecoder::new(CodecId::new(oxideav_dirac::CODEC_ID_STR));
    let pkt = Packet::new(0, TimeBase::new(1, 25), drc);
    let mut fatal: Option<String> = None;
    if let Err(e) = dec.send_packet(&pkt) {
        fatal = Some(format!("send_packet: {e:?}"));
    }
    let _ = dec.flush();

    let (cw, ch) = case.sub.chroma_dims(case.width, case.height);
    let y_size = case.width * case.height;
    let uv_size = cw * ch;

    let mut visible_idx = 0usize;
    let mut per_frame: Vec<Result<FrameDiff, String>> = Vec::with_capacity(case.n_frames);
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => {
                if visible_idx >= case.n_frames {
                    visible_idx += 1;
                    continue;
                }
                let ref_off = visible_idx * frame_size;
                let ref_y = &yuv_ref[ref_off..ref_off + y_size];
                let ref_u = &yuv_ref[ref_off + y_size..ref_off + y_size + uv_size];
                let ref_v = &yuv_ref[ref_off + y_size + uv_size..ref_off + frame_size];

                if vf.planes.len() < 3 {
                    per_frame.push(Err(format!(
                        "visible {visible_idx}: decoder produced {} planes, expected 3",
                        vf.planes.len()
                    )));
                    visible_idx += 1;
                    continue;
                }
                let our_y = vf.planes[0].data.as_slice();
                let our_u = vf.planes[1].data.as_slice();
                let our_v = vf.planes[2].data.as_slice();
                if our_y.len() != y_size || our_u.len() != uv_size || our_v.len() != uv_size {
                    per_frame.push(Err(format!(
                        "visible {visible_idx}: plane size mismatch \
                         (Y {} U {} V {} expected {} {} {})",
                        our_y.len(),
                        our_u.len(),
                        our_v.len(),
                        y_size,
                        uv_size,
                        uv_size
                    )));
                    visible_idx += 1;
                    continue;
                }
                let (yt, ye, ym, ys) = diff_plane(our_y, ref_y);
                let (ut, ue, um, us) = diff_plane(our_u, ref_u);
                let (vt, ve, vm, vs) = diff_plane(our_v, ref_v);
                per_frame.push(Ok(FrameDiff {
                    y_total: yt,
                    y_exact: ye,
                    y_max: ym,
                    y_sse: ys,
                    uv_total: ut + vt,
                    uv_exact: ue + ve,
                    uv_max: um.max(vm),
                    uv_sse: us + vs,
                }));
                visible_idx += 1;
            }
            Ok(_) => continue,
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => {
                let msg = format!("visible {visible_idx}: receive_frame: {e:?}");
                per_frame.push(Err(msg.clone()));
                if fatal.is_none() {
                    fatal = Some(msg);
                }
                break;
            }
        }
    }

    Some(DecodeReport {
        per_frame,
        visible_produced: visible_idx,
        fatal,
    })
}

/// Pretty-print + tier-aware assertion.
fn evaluate(case: &CorpusCase) {
    let report = match decode_fixture(case) {
        Some(r) => r,
        None => return, // missing fixture — already logged
    };

    let mut agg = FrameDiff::default();
    let mut errors: Vec<String> = Vec::new();
    for (i, r) in report.per_frame.iter().enumerate() {
        match r {
            Ok(d) => {
                eprintln!(
                    "  frame {i}: Y {}/{} exact (max {}, PSNR {:.2} dB), \
                     UV {}/{} exact (max {}, PSNR {:.2} dB), pct={:.2}%",
                    d.y_exact,
                    d.y_total,
                    d.y_max,
                    d.y_psnr(),
                    d.uv_exact,
                    d.uv_total,
                    d.uv_max,
                    d.uv_psnr(),
                    d.pct(),
                );
                agg.merge(d);
            }
            Err(e) => {
                eprintln!("  frame {i}: ERROR {e}");
                errors.push(format!("frame {i}: {e}"));
            }
        }
    }

    let pct = agg.pct();
    eprintln!(
        "[{:?}] {}: aggregate {}/{} exact ({pct:.2}%), \
         Y max {} PSNR {:.2} dB, UV max {} PSNR {:.2} dB, \
         visible_produced={}/{}{}",
        case.tier,
        case.name,
        agg.y_exact + agg.uv_exact,
        agg.y_total + agg.uv_total,
        agg.y_max,
        agg.y_psnr(),
        agg.uv_max,
        agg.uv_psnr(),
        report.visible_produced,
        case.n_frames,
        match &report.fatal {
            Some(f) => format!(", first_fatal=\"{f}\""),
            None => String::new(),
        }
    );

    match case.tier {
        Tier::BitExact => {
            assert!(
                errors.is_empty(),
                "{}: {} frame errors prevented bit-exact comparison: {:?}",
                case.name,
                errors.len(),
                errors
            );
            assert_eq!(
                agg.y_exact + agg.uv_exact,
                agg.y_total + agg.uv_total,
                "{}: not bit-exact (Y max {} PSNR {:.2} dB, UV max {} PSNR {:.2} dB; {:.4}% match)",
                case.name,
                agg.y_max,
                agg.y_psnr(),
                agg.uv_max,
                agg.uv_psnr(),
                pct
            );
        }
        Tier::ReportOnly => {
            // Don't fail. The eprintln output above is the report.
            // TODO(dirac-corpus): tighten to BitExact once the
            // underlying decoder gap is closed.
            let _ = pct;
        }
    }
}

// ---------------------------------------------------------------------------
// Per-fixture tests
// ---------------------------------------------------------------------------
//
// All fixtures start as `Tier::ReportOnly`. As the in-tree Dirac
// decoder closes the relevant gap (LD-profile slice decode, OBMC inter,
// per-picture wavelet selection, 4:2:2 chroma, interlaced TFF picture
// reconstruction), individual cases should be promoted to `BitExact`.
//
// Trace files (referenced in `evaluate()` via the `eprintln!` header)
// live alongside each fixture and capture the per-step `PARSE_UNIT` /
// `SEQUENCE` / `PICTURE` / `CODE_BLOCK` / `MOTION` events the reference
// decoder emitted on the bitstream — useful for diffing against our
// own decoder trace if/when divergence localisation is needed.

/// Smallest fixture: one main-profile intra picture (parse 0x0c) at
/// 320×240 4:2:0 8-bit, DD 9,7 wavelet, depth 3.
/// Trace: docs/video/dirac/fixtures/i-only-tiny-320x240/trace.txt.gz
#[test]
fn corpus_i_only_tiny_320x240() {
    evaluate(&CorpusCase {
        name: "i-only-tiny-320x240",
        width: 320,
        height: 240,
        n_frames: 1,
        sub: Subsampling::Yuv420,
        tier: Tier::ReportOnly,
    });
}

/// Intra (parse 0x0c) followed by inter-with-1-ref (parse 0x0d).
/// Exercises the OBMC motion-block decode against a single reference.
/// Reference YUV is in display order (decode order = display order
/// for this slice).
/// Trace: docs/video/dirac/fixtures/i-then-p-320x240/trace.txt.gz
#[test]
fn corpus_i_then_p_320x240() {
    evaluate(&CorpusCase {
        name: "i-then-p-320x240",
        width: 320,
        height: 240,
        n_frames: 2,
        sub: Subsampling::Yuv420,
        tier: Tier::ReportOnly,
    });
}

/// Intra + 1-ref inter + bidirectional inter (parse 0x0a, num_refs=2).
/// The 0x0a picture is in stream-order position 2 but reorders for
/// display — ffmpeg outputs three frames in display order in the
/// reference YUV. Our decoder currently produces frames in decode
/// order; under ReportOnly this surfaces as a per-frame mismatch,
/// recorded but not asserted.
/// Trace: docs/video/dirac/fixtures/i-p-b-320x240/trace.txt.gz
#[test]
fn corpus_i_p_b_320x240() {
    evaluate(&CorpusCase {
        name: "i-p-b-320x240",
        width: 320,
        height: 240,
        n_frames: 3,
        sub: Subsampling::Yuv420,
        tier: Tier::ReportOnly,
    });
}

/// VC-2 low-delay profile (parse 0x88), 20×15 slice grid, 320×240
/// 4:2:0. The LD path is intra-only and requires the
/// `decode_lowdelay_slice` Golomb-coded path (no `CODE_BLOCK` events
/// in the trace).
/// Trace: docs/video/dirac/fixtures/vc2-low-delay-tiny-320x240/trace.txt
#[test]
fn corpus_vc2_low_delay_tiny_320x240() {
    evaluate(&CorpusCase {
        name: "vc2-low-delay-tiny-320x240",
        width: 320,
        height: 240,
        n_frames: 1,
        sub: Subsampling::Yuv420,
        tier: Tier::ReportOnly,
    });
}

/// Three back-to-back VC-2 LD pictures (LD profile is intra-only, so
/// no reorder). Useful for verifying state-reset between pictures —
/// a decoder that incorrectly retains slice-grid coefficients across
/// pictures will produce wrong output starting with picture #1.
/// Trace: docs/video/dirac/fixtures/vc2-low-delay-3pics-320x240/trace.txt
#[test]
fn corpus_vc2_low_delay_3pics_320x240() {
    evaluate(&CorpusCase {
        name: "vc2-low-delay-3pics-320x240",
        width: 320,
        height: 240,
        n_frames: 3,
        sub: Subsampling::Yuv420,
        tier: Tier::ReportOnly,
    });
}

/// SD-PAL interlaced TFF: single intra picture (parse 0x0c), 720×576
/// 4:2:0 8-bit, DD 9,7 wavelet, depth 4 (one more level than the
/// 320×240 fixtures — exercises the IDWT recursion-depth handling).
/// Trace: docs/video/dirac/fixtures/interlaced-720x576-i-only/trace.txt.gz
#[test]
fn corpus_interlaced_720x576_i_only() {
    evaluate(&CorpusCase {
        name: "interlaced-720x576-i-only",
        width: 720,
        height: 576,
        n_frames: 1,
        sub: Subsampling::Yuv420,
        tier: Tier::ReportOnly,
    });
}

/// Per-picture wavelet selection: intra at DD 9,7 (`wavelet_idx=0`)
/// followed by inter at LeGall 5,3 (`wavelet_idx=1`). The only
/// fixture in the corpus that exercises a wavelet other than DD 9,7.
/// Trace: docs/video/dirac/fixtures/interlaced-720x576-i-then-p-wavelet-5-3/trace.txt.gz
#[test]
fn corpus_interlaced_720x576_i_then_p_wavelet_5_3() {
    evaluate(&CorpusCase {
        name: "interlaced-720x576-i-then-p-wavelet-5-3",
        width: 720,
        height: 576,
        n_frames: 2,
        sub: Subsampling::Yuv420,
        tier: Tier::ReportOnly,
    });
}

/// 4:2:2 chroma subsampling: 720×576 SD-PAL interlaced, single intra
/// picture, DD 9,7 wavelet depth 4. The only fixture with
/// `chroma_format=1` — chroma planes decode at 360×576 (full-vertical,
/// half-horizontal). Reference YUV is `yuv422p` (829440 B = 720*576 +
/// 2*360*576).
/// Trace: docs/video/dirac/fixtures/chroma-422-720x576-i-only/trace.txt.gz
#[test]
fn corpus_chroma_422_720x576_i_only() {
    evaluate(&CorpusCase {
        name: "chroma-422-720x576-i-only",
        width: 720,
        height: 576,
        n_frames: 1,
        sub: Subsampling::Yuv422,
        tier: Tier::ReportOnly,
    });
}
