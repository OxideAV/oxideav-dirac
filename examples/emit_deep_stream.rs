//! Emit a single-frame deep-colour (9-16-bit) VC-2 HQ intra-only
//! Dirac elementary stream. By default the §10.3.8 signal range is a
//! full-range custom range (`preset_idx = 0`, offset `2^(d-1)`,
//! excursion `2^d - 1`), so the §10.5.2 `video_depth` equals the
//! requested depth. With the optional `video` range argument the
//! Table 10.5 *video* preset for the depth is used instead (index 3
//! at 10-bit, index 4 at 12-bit) — presets are the only ranges some
//! black-box validators accept, so the `video` mode is what
//! cross-validated fixtures are generated from. Useful for black-box
//! cross-decode probing and for staging deep-colour fixtures.
//!
//! Usage: `emit_deep_stream <depth 9..=16> [chroma 420|422|444] [path] [full|video]`
//!
//! The synthetic picture is the same deterministic full-range pattern
//! the crate's high-bit-depth round-trip tests use
//! (`(x*17 + y*31 + seed*7) * 2654435761 mod 2^d`; Y/U/V seeds 1/5/9),
//! so `expected` pixels are reproducible from this source alone.

use std::io::Write;

use oxideav_dirac::encoder::{
    encode_single_hq_intra_stream_u16, make_minimal_sequence_with_signal_range, EncoderParams,
};
use oxideav_dirac::video_format::{ChromaFormat, SignalRange};
use oxideav_dirac::wavelet::WaveletFilter;

fn ramp_plane(w: usize, h: usize, depth: u32, seed: u32) -> Vec<u16> {
    let max = (1u64 << depth) - 1;
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let mix = x as u64 * 17 + y as u64 * 31 + seed as u64 * 7;
            out.push(((mix * 2654435761) % (max + 1)) as u16);
        }
    }
    out
}

fn main() {
    let depth: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .filter(|d| (9..=16).contains(d))
        .expect("usage: emit_deep_stream <depth 9..=16> [chroma 420|422|444] [path]");
    let chroma = match std::env::args().nth(2).as_deref() {
        None | Some("420") => ChromaFormat::Yuv420,
        Some("422") => ChromaFormat::Yuv422,
        Some("444") => ChromaFormat::Yuv444,
        Some(other) => panic!("unknown chroma {other}; expected 420, 422 or 444"),
    };

    let (w, h) = (64u32, 64u32);
    let (cw, ch) = match chroma {
        ChromaFormat::Yuv420 => (w / 2, h / 2),
        ChromaFormat::Yuv422 => (w / 2, h),
        ChromaFormat::Yuv444 => (w, h),
    };
    let max = ((1u64 << depth) - 1) as u32;
    let sr = match std::env::args().nth(4).as_deref() {
        None | Some("full") => SignalRange {
            luma_offset: 1 << (depth - 1),
            luma_excursion: max,
            chroma_offset: 1 << (depth - 1),
            chroma_excursion: max,
        },
        Some("video") => match depth {
            10 => SignalRange::PRESET_10BIT_VIDEO,
            12 => SignalRange::PRESET_12BIT_VIDEO,
            other => panic!("no Table 10.5 video preset at depth {other}; use 10 or 12"),
        },
        Some(other) => panic!("unknown range mode {other}; expected full or video"),
    };
    let seq = make_minimal_sequence_with_signal_range(w, h, chroma, sr);
    let mut params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    params.slice_size_scaler = 32;
    let y = ramp_plane(w as usize, h as usize, depth, 1);
    let u = ramp_plane(cw as usize, ch as usize, depth, 5);
    let v = ramp_plane(cw as usize, ch as usize, depth, 9);
    let stream = encode_single_hq_intra_stream_u16(&seq, &params, 0, &y, &u, &v);

    match std::env::args().nth(3) {
        Some(path) => {
            std::fs::write(&path, &stream).expect("write file");
            eprintln!("Wrote {} bytes to {}", stream.len(), path);
        }
        None => {
            std::io::stdout().write_all(&stream).expect("write stdout");
        }
    }
}
