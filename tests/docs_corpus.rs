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
    /// Decode must match the reference to at least `min_match_pct` of
    /// samples, with per-plane maxima no larger than `max_luma` /
    /// `max_chroma`. A regression that reopens a closed sub-gap (e.g.
    /// the §15.8.5 per-component intra-DC fix) trips this even though
    /// the fixture is not yet fully bit-exact.
    Bounded {
        min_match_pct: f64,
        max_luma: i32,
        max_chroma: i32,
    },
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
    /// Bytes per stored sample, in both our decoder's output planes
    /// and `expected.yuv`: 1 for the 8-bit fixtures, 2 for the
    /// deep-colour fixtures (LE 16-bit words — `Yuv420P10/12/16Le`
    /// packing on our side, the matching rawvideo layout in the
    /// reference dump). The byte-level comparison is depth-agnostic
    /// for the `BitExact` tier (equal samples ⇔ equal bytes); the
    /// PSNR / max-error report lines are byte-based and only
    /// meaningful for 1-byte fixtures.
    bytes_per_sample: usize,
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

    let frame_size = case.sub.frame_bytes(case.width, case.height) * case.bytes_per_sample;
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
    let y_size = case.width * case.height * case.bytes_per_sample;
    let uv_size = cw * ch * case.bytes_per_sample;

    // Drain every available `VideoFrame` first so we can sort them by
    // pts (== picture_number when the packet has no explicit pts, which
    // is the case here). The reference YUV is in **display order** —
    // for fixtures with B-pictures (e.g. `i-p-b-320x240`) decode order
    // differs from display order, and a naive in-order comparison
    // would line up our P frame against the reference's B and vice
    // versa, dragging both PSNRs down with off-content differences
    // unrelated to the per-picture decode quality. Sorting by pts
    // restores display order so the PSNR numbers measure what they
    // claim to.
    let mut decoded: Vec<oxideav_core::VideoFrame> = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => decoded.push(vf),
            Ok(_) => continue,
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => {
                let msg = format!("after {} frames: receive_frame: {e:?}", decoded.len());
                if fatal.is_none() {
                    fatal = Some(msg);
                }
                break;
            }
        }
    }
    decoded.sort_by_key(|f| f.pts.unwrap_or(0));
    let visible_idx_total = decoded.len();

    let mut per_frame: Vec<Result<FrameDiff, String>> = Vec::with_capacity(case.n_frames);
    for (visible_idx, vf) in decoded.into_iter().enumerate() {
        if visible_idx >= case.n_frames {
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
    }
    let visible_idx = visible_idx_total;

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
        Tier::Bounded {
            min_match_pct,
            max_luma,
            max_chroma,
        } => {
            assert!(
                pct >= min_match_pct,
                "{}: match rate {:.4}% fell below the {:.4}% floor \
                 (regression?) — Y max {} PSNR {:.2} dB, UV max {} PSNR {:.2} dB",
                case.name,
                pct,
                min_match_pct,
                agg.y_max,
                agg.y_psnr(),
                agg.uv_max,
                agg.uv_psnr(),
            );
            assert!(
                agg.y_max <= max_luma,
                "{}: luma max error {} exceeds bound {}",
                case.name,
                agg.y_max,
                max_luma,
            );
            assert!(
                agg.uv_max <= max_chroma,
                "{}: chroma max error {} exceeds bound {}",
                case.name,
                agg.uv_max,
                max_chroma,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Per-fixture tests
// ---------------------------------------------------------------------------
//
// As of round-408 **every fixture in the corpus decodes bit-exactly**
// against the reference and is pinned at `Tier::BitExact`: the five
// intra-only cases (`i-only-tiny`, `vc2-low-delay-tiny`,
// `vc2-low-delay-3pics`, `interlaced-720x576-i-only`,
// `chroma-422-720x576`), the two integer-pel inter cases (`i-then-p`,
// `i-p-b`) and — since the round-408 integer-part-clamp edge-extension
// fix in the §15.8.10 sub-pel fetch — the quarter-pel
// `interlaced-720x576-i-then-p-wavelet-5-3` case as well. Round-417
// added the three deep-colour HQ fixtures (`hq-10bit-preset3`,
// `hq-12bit-preset4`, `hq-16bit-fullrange`), bit-exact from staging.
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
        // Bit-exact since round-118: the §5.4 unbiased-mean (`+1`)
        // rounding in intra DC subband prediction was corrected.
        tier: Tier::BitExact,
        bytes_per_sample: 1,
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
        // Bit-exact since round-128: the §12.3.6.6 Case 4 unbiased-mean
        // (`div_euclid` floor) fix for intra-block DC prediction inside
        // inter pictures closed the residual ~1% pixel gap left after
        // round-125's §13.2.1 inter quant-offset fix.
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

/// Intra + 1-ref inter + bidirectional inter (parse 0x0a, num_refs=2).
/// The 0x0a picture is in stream-order position 2 but reorders for
/// display — ffmpeg outputs three frames in display order in the
/// reference YUV. Our decoder produces frames in decode order; the
/// `decode_fixture` driver sorts them by `pts` (== `picture_number`
/// when packets carry no explicit pts) before per-frame comparison
/// so the PSNR numbers measure each picture against its true display
/// counterpart.
/// Trace: docs/video/dirac/fixtures/i-p-b-320x240/trace.txt.gz
#[test]
fn corpus_i_p_b_320x240() {
    evaluate(&CorpusCase {
        name: "i-p-b-320x240",
        width: 320,
        height: 240,
        n_frames: 3,
        sub: Subsampling::Yuv420,
        // Bit-exact since round-128: the §12.3.6.6 Case 4 unbiased-mean
        // (`div_euclid` floor) fix for intra-block DC prediction inside
        // inter pictures. Both the P frame and the B frame now match
        // ffmpeg byte-for-byte across all three pictures (I, P, B).
        tier: Tier::BitExact,
        bytes_per_sample: 1,
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
        // Bit-exact since round-118 (intra DC prediction `mean` fix).
        tier: Tier::BitExact,
        bytes_per_sample: 1,
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
        // Bit-exact since round-118 (intra DC prediction `mean` fix).
        tier: Tier::BitExact,
        bytes_per_sample: 1,
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
        // Bit-exact since round-118 (intra DC prediction `mean` fix);
        // also confirms the depth-4 IDWT recursion is correct.
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

/// Per-picture wavelet selection: intra at DD 9,7 (`wavelet_idx=0`)
/// followed by inter at LeGall 5,3 (`wavelet_idx=1`). The only
/// fixture in the corpus that exercises a wavelet other than DD 9,7 —
/// and, more importantly for motion compensation, the only one whose
/// inter picture uses **quarter-pel** motion (`mv_precision = 2`); the
/// two bit-exact 320×240 inter fixtures both use integer-pel motion
/// (`mv_precision = 0`), which bypasses the §15.8.10/§15.8.11 sub-pel
/// interpolation entirely.
///
/// Round-404 closed the dominant divergence (per-component intra-block
/// DC, §15.8.5), leaving ~0.09% on the top/right picture-boundary
/// strips. Round-408 closed that remainder: §15.8.10's out-of-frame
/// sub-pel fetch clamps the **integer-pel part** of the half-pel
/// coordinate while preserving its half-pel fraction (fetching the
/// nearest *filtered* edge row/column), rather than clipping the raw
/// half-pel coordinate as the pseudocode reads. The rule was pinned by
/// black-box uniform-MV probe pictures against the reference decoder
/// (see `oxideav_dirac::obmc::subpel_predict`). This fixture — whose P
/// picture pans up-right, overspilling exactly those two edges — has
/// been bit-exact since.
/// Trace: docs/video/dirac/fixtures/interlaced-720x576-i-then-p-wavelet-5-3/trace.txt.gz
#[test]
fn corpus_interlaced_720x576_i_then_p_wavelet_5_3() {
    evaluate(&CorpusCase {
        name: "interlaced-720x576-i-then-p-wavelet-5-3",
        width: 720,
        height: 576,
        n_frames: 2,
        sub: Subsampling::Yuv420,
        // Bit-exact since round-408 (integer-part-clamp edge extension
        // in the §15.8.10 sub-pel fetch).
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

/// Out-of-frame sub-pel MC probes: the `i-then-p-320x240` intra anchor
/// followed by four hand-crafted inter pictures, each with a *uniform*
/// MV grid whose blocks overspill the picture edges (half-pel (-3,-3)
/// top+left; quarter-pel (7,7) bottom+right; eighth-pel (-11,5); and a
/// far-overreach half-pel (0,-33)) and an explicit all-zero residual.
/// The reference YUV is the reference decoder's black-box output, so
/// the fixture pins §15.8.10's edge-extension rule for fetches beyond
/// the reference bounds: clamp the **integer-pel part** of the
/// half-pel coordinate and preserve its half-pel fraction (see
/// `oxideav_dirac::obmc::subpel_predict`). A regression back to the
/// pseudocode's raw half-pel coordinate clip flips edge samples by
/// ±1..3 LSB and trips this immediately.
///
/// The probes code the zero residual explicitly (`ZERO_RESIDUAL=0`,
/// all bands `length=0`) because the reference decoder mis-reconstructs
/// skip pictures (`zero_res=1`) — see the fixture's `notes.md`.
#[test]
fn corpus_edge_mc_probes_320x240() {
    evaluate(&CorpusCase {
        name: "edge-mc-probes-320x240",
        width: 320,
        height: 240,
        n_frames: 5,
        sub: Subsampling::Yuv420,
        // Bit-exact from the day the fixture was staged (round-408, the
        // same round the integer-part-clamp edge extension landed).
        tier: Tier::BitExact,
        bytes_per_sample: 1,
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
        // Bit-exact since round-118 (intra DC prediction `mean` fix);
        // also confirms 4:2:2 chroma-plane decode is correct.
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

/// Round-417 deep-colour fixture #10: HQ-profile (parse 0xE8) 10-bit
/// intra at 64×64 4:2:0, Table 10.5 signal-range preset 3. The
/// `expected.yuv` ground truth is the *reference decoder's* output on
/// an `oxideav-dirac`-encoded stream (black-box cross-validation) and
/// equals the analytic source pattern exactly; planes are LE 16-bit
/// words (low 10 bits — the `Yuv420P10Le` packing on both sides).
#[test]
fn corpus_hq_10bit_preset3_64x64() {
    evaluate(&CorpusCase {
        name: "hq-10bit-preset3-64x64",
        width: 64,
        height: 64,
        n_frames: 1,
        sub: Subsampling::Yuv420,
        // Bit-exact from the day the fixture was staged (round-417).
        tier: Tier::BitExact,
        bytes_per_sample: 2,
    });
}

/// Round-417 deep-colour fixture #11: HQ-profile 12-bit intra,
/// Table 10.5 signal-range preset 4 — the deepest depth the reference
/// decoder validates. Reference-cross-decoded ground truth,
/// `Yuv420P12Le` packing.
#[test]
fn corpus_hq_12bit_preset4_64x64() {
    evaluate(&CorpusCase {
        name: "hq-12bit-preset4-64x64",
        width: 64,
        height: 64,
        n_frames: 1,
        sub: Subsampling::Yuv420,
        // Bit-exact from the day the fixture was staged (round-417).
        tier: Tier::BitExact,
        bytes_per_sample: 2,
    });
}

/// Round-417 deep-colour fixture #12: HQ-profile **16-bit** intra via
/// a §10.3.8 custom (index 0) signal range — the only fixture
/// exercising the index-0 wire format and the all-bits-significant
/// `Yuv420P16Le` output surface. No external validator accepts
/// index-0 ranges (see the fixture's `notes.md`), so the ground truth
/// is the analytically-derivable source pattern (verified equal at
/// staging time): `((x*17 + y*31 + seed*7) * 2654435761) mod 2^16`,
/// Y/U/V seeds 1/5/9.
#[test]
fn corpus_hq_16bit_fullrange_64x64() {
    evaluate(&CorpusCase {
        name: "hq-16bit-fullrange-64x64",
        width: 64,
        height: 64,
        n_frames: 1,
        sub: Subsampling::Yuv420,
        // Bit-exact by construction (q0 lossless; expected == pattern).
        tier: Tier::BitExact,
        bytes_per_sample: 2,
    });
}
