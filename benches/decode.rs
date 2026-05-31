// Parallel-array index loops are idiomatic in codec / bench code; skip
// the lint (mirrors the crate root's `#![allow(clippy::needless_range_loop)]`).
#![allow(clippy::needless_range_loop)]

//! Criterion benchmarks for the Dirac / VC-2 decoder hot paths.
//!
//! Round 190 (depth-mode benchmarks): the dirac crate is feature-
//! complete enough on both the decoder (~95%) and encoder (~95%) sides
//! that further coverage rounds run into the saturated-crate guidance
//! ("oldest-modified first → fuzz / bench / profile"). r165 landed the
//! decoder-side malformed-input fuzz oracle; r179 landed the encoder-
//! side rate-control fuzz oracle; this round adds the first criterion
//! bench harness covering all three hot paths the future encoder /
//! decoder rounds will A/B their algorithm tweaks against (intra DC
//! prediction, codeblock quant-offset walk, slice-bytes derivation,
//! rate-control picker).
//!
//! Round 195 (depth-mode benchmarks, follow-up): the original three
//! decoder rows covered only LeGall 5/3 (`wavelet_index = 1`). This
//! round adds a fourth row covering Deslauriers-Dubuc 9/7
//! (`wavelet_index = 0`) — the Dirac *default* filter — at HQ q=0.
//! DD9/7 has a 4-tap second lifting step (vs. LeGall's 2-tap), so its
//! per-row lifting cost in the IDWT is ~2x LeGall's. Adding this row
//! makes the bench harness sensitive to wavelet-specific work the
//! LeGall-only rows wouldn't catch, and gives future profile-driven
//! IDWT tweaks a fixture where the lifting cost is the dominant
//! per-frame work rather than co-dominant with the entropy-coder path.
//!
//! Each scenario synthesises a deterministic YUV input plane via an
//! xorshift32 PRNG, encodes it **once** in a Criterion setup step with
//! the production `oxideav_dirac::encoder::encode_single_*_intra_stream`
//! entry points, and then iterates the decoder on the resulting byte
//! buffer. The encode work is *outside* the timed region — only the
//! per-packet `send_packet -> receive_frame` cost is measured. The
//! Throughput is reported in **input pixels per second** (W*H per
//! iteration) so the runs compare meaningfully across resolutions.
//!
//! Scenarios:
//!
//!   - **decode_hq_intra_64x64_q0**: 64x64 4:2:0 single HQ intra
//!     picture at qindex 0 (near-lossless). The smallest realistic HQ
//!     frame our encoder emits — exercises the §13.5.2 slice-header
//!     walk + Annex E.1 default quant matrix + LeGall-5/3 inverse DWT
//!     in their default configuration.
//!   - **decode_hq_intra_64x64_q32**: same fixture at qindex 32.
//!     Lossy regime — drops most of the high-frequency coefficient
//!     bits, so the slice payloads are shorter and the decoder spends
//!     a higher fraction of time on the sync-walk / parse-info path
//!     vs. the inverse-quant + IDWT path. Useful as the A/B partner
//!     for the q=0 row when tuning the inverse-quant inner loop.
//!   - **decode_ld_intra_64x64_q16**: 64x64 4:2:0 single VC-2 LD intra
//!     picture at qindex 16 (fixed-rate slice budget, mid-quantiser).
//!     The LD profile's coefficient-bytes-per-slice budget is fixed
//!     regardless of energy so this row's timing is the most stable
//!     of the three under PRNG-seed variation — useful as the
//!     comparison baseline when adding new LD-specific work.
//!   - **decode_hq_intra_64x64_q0_dd9_7** (round-195): 64x64 4:2:0 HQ
//!     intra at qindex 0 using the **DD9/7** wavelet
//!     (`wavelet_index = 0`, Dirac's default). Same fixture as the
//!     `qindex=0` row above but with a heavier lifting kernel — the
//!     IDWT inner-loop cost is the differentiator.
//!
//! Run with:
//!     cargo bench -p oxideav-dirac --bench decode

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_core::{CodecId, CodecParameters, CodecRegistry, Frame, Packet, TimeBase};
use oxideav_dirac::encoder::{
    encode_single_hq_intra_stream, encode_single_ld_intra_stream, make_minimal_sequence,
    make_minimal_sequence_ld, EncoderParams, LdEncoderParams,
};
use oxideav_dirac::video_format::ChromaFormat;
use oxideav_dirac::wavelet::WaveletFilter;

/// 32-bit xorshift PRNG (Marsaglia 13/17/5). Used purely for bench
/// input determinism — never inside the timed region.
fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Build a `width x height` 4:2:0 YUV triple seeded by `seed`. The Y
/// plane carries a diagonal-gradient base plus per-pixel noise; the
/// U/V planes carry a smoother gradient (mid-range chroma).
fn synth_yuv420(width: usize, height: usize, seed: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut state = seed;
    let mut y = vec![0u8; width * height];
    for row in 0..height {
        for col in 0..width {
            let base = (((row + col) as u32 * 3) & 0xFF) as u8;
            let n = (xorshift32(&mut state) >> 24) as u8;
            y[row * width + col] = base.wrapping_add(n / 16);
        }
    }
    let cw = width / 2;
    let ch = height / 2;
    let mut u = vec![128u8; cw * ch];
    let mut v = vec![128u8; cw * ch];
    for row in 0..ch {
        for col in 0..cw {
            u[row * cw + col] = 128u8.wrapping_add(((col as i32 - cw as i32 / 2) * 2) as u8);
            v[row * cw + col] = 128u8.wrapping_add(((row as i32 - ch as i32 / 2) * 2) as u8);
        }
    }
    (y, u, v)
}

/// Construct a HQ intra stream for `(width, height)` 4:2:0 at the
/// given qindex with the LeGall 5/3 wavelet. Encoder work happens here
/// — *outside* the bench closure.
fn build_hq_stream(width: u32, height: u32, qindex: u32, seed: u32) -> Vec<u8> {
    build_hq_stream_with_wavelet(width, height, qindex, seed, WaveletFilter::LeGall5_3)
}

/// Construct a HQ intra stream for `(width, height)` 4:2:0 at the
/// given qindex with `wavelet`. Round 195 adds the wavelet parameter so
/// the `decode_hq_intra_64x64_q0_dd9_7` row can swap in DD9/7 without
/// duplicating the rest of the encode harness.
fn build_hq_stream_with_wavelet(
    width: u32,
    height: u32,
    qindex: u32,
    seed: u32,
    wavelet: WaveletFilter,
) -> Vec<u8> {
    let seq = make_minimal_sequence(width, height, ChromaFormat::Yuv420);
    let mut params = EncoderParams::default_hq(wavelet, 3);
    params.qindex = qindex;
    let (y, u, v) = synth_yuv420(width as usize, height as usize, seed);
    encode_single_hq_intra_stream(&seq, &params, 0, &y, &u, &v)
}

/// Construct a LD intra stream for `(width, height)` 4:2:0 at the
/// given qindex. LD's per-slice byte budget is fixed up-front; we use
/// a 4x4 slice grid at 64 B/slice (`= 1024 B / picture` of coefficient
/// payload) which is comfortably above what synth_yuv420 needs at the
/// mid-quantiser sweet spot.
fn build_ld_stream(width: u32, height: u32, qindex: u32, seed: u32) -> Vec<u8> {
    let seq = make_minimal_sequence_ld(width, height, ChromaFormat::Yuv420);
    let mut params = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 64);
    params.qindex = qindex;
    let (y, u, v) = synth_yuv420(width as usize, height as usize, seed);
    encode_single_ld_intra_stream(&seq, &params, 0, &y, &u, &v)
}

/// One iteration of the timed region: build a registry-backed decoder,
/// push the prebuilt stream as a single packet, and pull one frame.
/// The registry construction is cheap and intentionally included — it
/// is the work an embedding application performs per-stream.
fn decode_one_iteration(stream: &[u8]) {
    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("decoder factory");
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream.to_vec());
    dec.send_packet(&pkt).expect("send_packet");
    let frame = dec.receive_frame().expect("receive_frame");
    // Sanity guard so optimisation cannot elide the work — but keep
    // it cheap (a tag check, no plane sum / hash).
    if let Frame::Video(v) = frame {
        std::hint::black_box(v.planes.len());
    } else {
        panic!("expected video frame");
    }
}

fn bench_decode(c: &mut Criterion) {
    let pixels_64x64: u64 = 64 * 64;

    let mut g = c.benchmark_group("dirac_decode");
    g.throughput(Throughput::Elements(pixels_64x64));

    // HQ q=0 — near-lossless slice payloads (longer coefficient
    // blocks → IDWT dominates).
    let hq_q0 = build_hq_stream(64, 64, 0, 0xDEAD_BEEF);
    g.bench_with_input(
        BenchmarkId::new("hq_intra_64x64", "qindex=0"),
        &hq_q0,
        |b, s| b.iter(|| decode_one_iteration(s)),
    );

    // HQ q=32 — short coefficient blocks (most coefs quantised to
    // zero); parse-info / sync walk dominates.
    let hq_q32 = build_hq_stream(64, 64, 32, 0xDEAD_BEEF);
    g.bench_with_input(
        BenchmarkId::new("hq_intra_64x64", "qindex=32"),
        &hq_q32,
        |b, s| b.iter(|| decode_one_iteration(s)),
    );

    // LD q=16 — fixed-rate slice budget; timing is the most stable
    // across PRNG seeds.
    let ld_q16 = build_ld_stream(64, 64, 16, 0xDEAD_BEEF);
    g.bench_with_input(
        BenchmarkId::new("ld_intra_64x64", "qindex=16"),
        &ld_q16,
        |b, s| b.iter(|| decode_one_iteration(s)),
    );

    // HQ q=0 with the DD9/7 wavelet (Dirac default; `wavelet_index = 0`).
    // Heavier lifting kernel than LeGall (4-tap second step vs. 2-tap)
    // makes the IDWT inner loop the dominant per-frame cost.
    let hq_q0_dd9_7 =
        build_hq_stream_with_wavelet(64, 64, 0, 0xDEAD_BEEF, WaveletFilter::DeslauriersDubuc9_7);
    g.bench_with_input(
        BenchmarkId::new("hq_intra_64x64", "qindex=0/wavelet=dd9_7"),
        &hq_q0_dd9_7,
        |b, s| b.iter(|| decode_one_iteration(s)),
    );

    g.finish();
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
