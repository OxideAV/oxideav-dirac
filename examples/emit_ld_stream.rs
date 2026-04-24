use oxideav_dirac::encoder::{
    encode_single_ld_intra_stream, make_minimal_sequence_ld, synthetic_testsrc_64_yuv420,
    LdEncoderParams,
};
use oxideav_dirac::video_format::ChromaFormat;
use oxideav_dirac::wavelet::WaveletFilter;
use std::io::Write;

fn main() {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    let params = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 128);
    let (y, u, v) = synthetic_testsrc_64_yuv420();
    let stream = encode_single_ld_intra_stream(&seq, &params, 0, &y, &u, &v);
    let target = std::env::args().nth(1);
    match target {
        Some(path) => {
            std::fs::write(&path, &stream).expect("write file");
            eprintln!("Wrote {} bytes to {}", stream.len(), path);
        }
        None => {
            std::io::stdout().write_all(&stream).expect("write stdout");
        }
    }
}
