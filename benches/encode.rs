// Parallel-array index loops are idiomatic in codec / bench code; skip
// the lint (mirrors the crate root's `#![allow(clippy::needless_range_loop)]`).
#![allow(clippy::needless_range_loop)]

//! Criterion benchmarks for the Dirac / VC-2 encoder hot paths.
//!
//! Round 190 (depth-mode benchmarks): companion to `decode.rs`. Each
//! scenario synthesises a `width x height` 4:2:0 YUV input plane via
//! an xorshift32 PRNG in a setup step (so the timed region sees a
//! freshly built input — but the input synthesis itself is *outside*
//! the closure), then drives the production
//! `oxideav_dirac::encoder::encode_single_*_intra_stream` entry point
//! and measures one stream emission per iteration.
//!
//! Throughput is reported in **output pixels per second** (W*H per
//! iteration) so the runs compare meaningfully across resolutions and
//! against the matching `decode.rs` rows.
//!
//! Scenarios:
//!
//!   - **encode_hq_intra_64x64_q0**: 64x64 4:2:0 HQ intra at qindex 0
//!     (lossless DZ-quantiser). Exercises every coefficient in the
//!     §13.5.2 slice walk — the longest per-slice path the HQ encoder
//!     ever takes.
//!   - **encode_hq_intra_64x64_q32**: same fixture at qindex 32.
//!     Most coefficients quantise to zero; the per-slice path is
//!     dominated by the `quant_factor` / signed exp-Golomb writer
//!     loop rather than the IDWT inverse, complementary to q=0.
//!   - **encode_ld_intra_64x64_q16**: 64x64 4:2:0 LD intra at qindex
//!     16. LD's fixed-rate slice budget makes the per-slice path
//!     length independent of energy — the most timing-stable row
//!     for A/B comparisons of LD-specific work.
//!
//! Run with:
//!     cargo bench -p oxideav-dirac --bench encode

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_dirac::encoder::{
    encode_single_hq_intra_stream, encode_single_ld_intra_stream, make_minimal_sequence,
    make_minimal_sequence_ld, EncoderParams, LdEncoderParams,
};
use oxideav_dirac::sequence::SequenceHeader;
use oxideav_dirac::video_format::ChromaFormat;
use oxideav_dirac::wavelet::WaveletFilter;

/// 32-bit xorshift PRNG (Marsaglia 13/17/5).
fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Synthesise a `width x height` 4:2:0 YUV triple. Identical
/// formulation to `decode.rs::synth_yuv420` — sibling benches stay
/// comparable.
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

fn bench_encode(c: &mut Criterion) {
    let pixels_64x64: u64 = 64 * 64;

    let mut g = c.benchmark_group("dirac_encode");
    g.throughput(Throughput::Elements(pixels_64x64));

    // HQ encoder, qindex 0 — every coefficient survives the dead-zone
    // quantiser, so the per-slice byte cost is maximal.
    let seq_hq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let (y, u, v) = synth_yuv420(64, 64, 0xDEAD_BEEF);
    let mut params_hq_q0 = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    params_hq_q0.qindex = 0;
    g.bench_with_input(
        BenchmarkId::new("hq_intra_64x64", "qindex=0"),
        &(
            seq_hq.clone(),
            params_hq_q0,
            y.clone(),
            u.clone(),
            v.clone(),
        ),
        |b, (seq, params, y, u, v)| {
            b.iter(|| {
                let s = encode_single_hq_intra_stream(seq, params, 0, y, u, v);
                std::hint::black_box(s.len());
            })
        },
    );

    // HQ encoder, qindex 32 — most coefficients quantise to zero,
    // slice byte cost shrinks markedly; isolates the entropy-coder
    // path.
    let mut params_hq_q32 = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    params_hq_q32.qindex = 32;
    g.bench_with_input(
        BenchmarkId::new("hq_intra_64x64", "qindex=32"),
        &(
            seq_hq.clone(),
            params_hq_q32,
            y.clone(),
            u.clone(),
            v.clone(),
        ),
        |b, (seq, params, y, u, v): &(SequenceHeader, EncoderParams, Vec<u8>, Vec<u8>, Vec<u8>)| {
            b.iter(|| {
                let s = encode_single_hq_intra_stream(seq, params, 0, y, u, v);
                std::hint::black_box(s.len());
            })
        },
    );

    // LD encoder, qindex 16. 4x4 slice grid at 64 B / slice = 1024 B
    // coefficient payload; comfortably above what synth_yuv420 needs
    // at mid-quantiser.
    let seq_ld = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let mut params_ld_q16 = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 64);
    params_ld_q16.qindex = 16;
    g.bench_with_input(
        BenchmarkId::new("ld_intra_64x64", "qindex=16"),
        &(seq_ld, params_ld_q16, y, u, v),
        |b, (seq, params, y, u, v): &(SequenceHeader, LdEncoderParams, Vec<u8>, Vec<u8>, Vec<u8>)| {
            b.iter(|| {
                let s = encode_single_ld_intra_stream(seq, params, 0, y, u, v);
                std::hint::black_box(s.len());
            })
        },
    );

    g.finish();
}

criterion_group!(benches, bench_encode);
criterion_main!(benches);
