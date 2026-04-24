//! Emit a single-frame VC-2 HQ intra-only Dirac elementary stream to
//! stdout or a file. Useful for feeding `ffmpeg -c:v dirac` and
//! checking third-party interop.

use std::io::Write;

use oxideav_dirac::encoder::{
    encode_single_hq_intra_stream, make_minimal_sequence, synthetic_testsrc_64_yuv420,
    EncoderParams,
};
use oxideav_dirac::video_format::ChromaFormat;
use oxideav_dirac::wavelet::WaveletFilter;

fn main() {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let (y, u, v) = synthetic_testsrc_64_yuv420();
    let stream = encode_single_hq_intra_stream(&seq, &params, 0, &y, &u, &v);
    let target = std::env::args().nth(1);
    match target {
        Some(path) => {
            std::fs::write(&path, &stream).expect("write file");
            eprintln!(
                "Wrote {} bytes to {}",
                stream.len(),
                path
            );
        }
        None => {
            std::io::stdout().write_all(&stream).expect("write stdout");
        }
    }
}
