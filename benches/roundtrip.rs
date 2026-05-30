// Parallel-array index loops are idiomatic in codec / bench code; skip
// the lint (mirrors the crate root's `#![allow(clippy::needless_range_loop)]`).
#![allow(clippy::needless_range_loop)]

//! Criterion benchmarks for the Dirac / VC-2 encode -> decode roundtrip
//! hot paths.
//!
//! Round 190 (depth-mode benchmarks): companion to `decode.rs` and
//! `encode.rs`. Each scenario synthesises a 64x64 4:2:0 YUV input plane
//! via an xorshift32 PRNG, then per-iteration encodes it and decodes
//! it back through the production
//! `oxideav_dirac::encoder::encode_single_*_intra_stream` +
//! `register_codecs` / `first_decoder` path. The roundtrip row is the
//! one a downstream rate-control / qindex-picker tweak shifts in both
//! directions at once, so it gets its own harness instead of summing
//! the encode + decode rows visually.
//!
//! Throughput is reported in **roundtrip pixels per second** (W*H per
//! iteration). One roundtrip == one input frame encoded and decoded
//! end-to-end.
//!
//! Scenarios:
//!
//!   - **roundtrip_hq_intra_64x64_q0**: 64x64 4:2:0 HQ intra at qindex
//!     0 — the lossless-DZ-quantiser roundtrip every encoder /
//!     decoder bit-exact-cross-check test in the crate already
//!     depends on. Captures the joint cost.
//!   - **roundtrip_hq_intra_64x64_q32**: same fixture at qindex 32.
//!     Lossy regime — slice payloads shrink, so both the encoder's
//!     entropy-writer loop and the decoder's parse-info walk
//!     dominate over the IDWT inverse.
//!   - **roundtrip_ld_intra_64x64_q16**: 64x64 4:2:0 LD intra at
//!     qindex 16, 4x4 slices, 64 B/slice. LD's fixed-rate budget
//!     makes the roundtrip timing the most stable of the three under
//!     PRNG-seed variation.
//!
//! Run with:
//!     cargo bench -p oxideav-dirac --bench roundtrip

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_core::{CodecId, CodecParameters, CodecRegistry, Frame, Packet, TimeBase};
use oxideav_dirac::encoder::{
    encode_single_hq_intra_stream, encode_single_ld_intra_stream, make_minimal_sequence,
    make_minimal_sequence_ld, EncoderParams, LdEncoderParams,
};
use oxideav_dirac::sequence::SequenceHeader;
use oxideav_dirac::video_format::ChromaFormat;
use oxideav_dirac::wavelet::WaveletFilter;

fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

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

fn roundtrip_hq_once(seq: &SequenceHeader, params: &EncoderParams, y: &[u8], u: &[u8], v: &[u8]) {
    let stream = encode_single_hq_intra_stream(seq, params, 0, y, u, v);
    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("decoder factory");
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&pkt).expect("send_packet");
    let frame = dec.receive_frame().expect("receive_frame");
    if let Frame::Video(vf) = frame {
        std::hint::black_box(vf.planes.len());
    } else {
        panic!("expected video frame");
    }
}

fn roundtrip_ld_once(seq: &SequenceHeader, params: &LdEncoderParams, y: &[u8], u: &[u8], v: &[u8]) {
    let stream = encode_single_ld_intra_stream(seq, params, 0, y, u, v);
    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    let mut dec = reg.first_decoder(&cp).expect("decoder factory");
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream);
    dec.send_packet(&pkt).expect("send_packet");
    let frame = dec.receive_frame().expect("receive_frame");
    if let Frame::Video(vf) = frame {
        std::hint::black_box(vf.planes.len());
    } else {
        panic!("expected video frame");
    }
}

fn bench_roundtrip(c: &mut Criterion) {
    let pixels_64x64: u64 = 64 * 64;

    let mut g = c.benchmark_group("dirac_roundtrip");
    g.throughput(Throughput::Elements(pixels_64x64));

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
        |b, (seq, params, y, u, v)| b.iter(|| roundtrip_hq_once(seq, params, y, u, v)),
    );

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
            b.iter(|| roundtrip_hq_once(seq, params, y, u, v))
        },
    );

    let seq_ld = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let mut params_ld_q16 = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 64);
    params_ld_q16.qindex = 16;
    g.bench_with_input(
        BenchmarkId::new("ld_intra_64x64", "qindex=16"),
        &(seq_ld, params_ld_q16, y, u, v),
        |b,
         (seq, params, y, u, v): &(
            SequenceHeader,
            LdEncoderParams,
            Vec<u8>,
            Vec<u8>,
            Vec<u8>,
        )| { b.iter(|| roundtrip_ld_once(seq, params, y, u, v)) },
    );

    g.finish();
}

criterion_group!(benches, bench_roundtrip);
criterion_main!(benches);
