//! Env-gated decode tracing.
//!
//! Implements the instrumented-decode trace contract from
//! `docs/video/dirac/dirac-fixtures-and-traces.md`: one tab-separated
//! `key=value` line per event, chronological decode order, activated by
//! the `DIRAC_TRACE` environment variable (written to
//! `DIRAC_TRACE_FILE`, or stderr when unset). The optional
//! motion-compensated prediction-plane checksum (`MC_PLANE`) is
//! additionally gated on `DIRAC_TRACE_MC`.
//!
//! Emitted events:
//!
//! | Event | Site | Contract section |
//! | --- | --- | --- |
//! | `PARSE_UNIT` | decoder scan loop | base vocabulary |
//! | `SEQUENCE` | after sequence-header parse | base vocabulary |
//! | `PICTURE` | after picture header + transform params | base vocabulary |
//! | `MOTION` | start of block-motion-data decode | base vocabulary |
//! | `MOTION_BLOCK` | per superblock (raster order) | base vocabulary |
//! | `MOTION_GLOBAL` | per reference when `globalmc_flag` | MV trace-spec |
//! | `MOTION_MV` | per decoded prediction block | MV trace-spec |
//! | `MC_PLANE` | per component, OBMC output pre-residual | MV trace-spec |
//!
//! The `MOTION_MV` stream is emitted after the picture's block motion
//! data has fully decoded (the wire carries splits, modes, vectors and
//! DCs as separate length-prefixed arith blocks), walking superblocks
//! in raster order and each superblock's coded blocks in raster/step
//! order — the same per-block order the contract specifies.
//!
//! Everything in this module is inert unless `DIRAC_TRACE` is set to a
//! non-empty value not starting with `0`: the fast path is a single
//! cached boolean load.

use std::fs::File;
use std::io::Write;
use std::sync::{Mutex, OnceLock};

enum Sink {
    File(Mutex<File>),
    Stderr,
}

struct TraceState {
    enabled: bool,
    mc_enabled: bool,
    sink: Option<Sink>,
}

fn state() -> &'static TraceState {
    static STATE: OnceLock<TraceState> = OnceLock::new();
    STATE.get_or_init(|| {
        let en = std::env::var("DIRAC_TRACE").ok();
        let enabled = matches!(&en, Some(v) if !v.is_empty() && !v.starts_with('0'));
        if !enabled {
            return TraceState {
                enabled: false,
                mc_enabled: false,
                sink: None,
            };
        }
        let mc = std::env::var("DIRAC_TRACE_MC").ok();
        let mc_enabled = matches!(&mc, Some(v) if !v.is_empty() && !v.starts_with('0'));
        let sink = match std::env::var("DIRAC_TRACE_FILE") {
            Ok(path) if !path.is_empty() => match File::create(&path) {
                Ok(f) => Some(Sink::File(Mutex::new(f))),
                Err(_) => Some(Sink::Stderr),
            },
            _ => Some(Sink::Stderr),
        };
        TraceState {
            enabled: true,
            mc_enabled,
            sink,
        }
    })
}

/// Whether tracing is active (`DIRAC_TRACE` set).
#[inline]
pub(crate) fn enabled() -> bool {
    state().enabled
}

/// Whether the `MC_PLANE` prediction-plane checksum is also active
/// (`DIRAC_TRACE` *and* `DIRAC_TRACE_MC` set).
#[inline]
pub(crate) fn mc_enabled() -> bool {
    let s = state();
    s.enabled && s.mc_enabled
}

/// Write one already-formatted event line (no trailing newline).
pub(crate) fn emit(line: &str) {
    let s = state();
    if !s.enabled {
        return;
    }
    match &s.sink {
        Some(Sink::File(f)) => {
            if let Ok(mut f) = f.lock() {
                let _ = writeln!(f, "{line}");
            }
        }
        Some(Sink::Stderr) | None => {
            eprintln!("{line}");
        }
    }
}

// ---------------------------------------------------------------------------
// Per-picture trace context (picture number for MC_PLANE lines).
// ---------------------------------------------------------------------------

std::thread_local! {
    static PIC_NUM: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Record the picture number currently being decoded on this thread so
/// downstream `MC_PLANE` emissions can tag their lines.
pub(crate) fn set_picture_number(n: u32) {
    PIC_NUM.with(|c| c.set(n));
}

pub(crate) fn picture_number() -> u32 {
    PIC_NUM.with(|c| c.get())
}

// ---------------------------------------------------------------------------
// Line formatters (pure — unit-tested without touching the global sink).
// ---------------------------------------------------------------------------

/// `PARSE_UNIT` — one per Dirac parse unit found by the scanner.
pub(crate) fn format_parse_unit(offset: usize, parse_code: u8, next: u32, prev: u32) -> String {
    format!(
        "PARSE_UNIT\toffset={offset}\tparse_code=0x{parse_code:02x}\tnext_offset={next}\tprev_offset={prev}\tsize={next}"
    )
}

/// `MOTION` — once per inter picture, before the superblock walk.
pub(crate) fn format_motion(sbw: u32, sbh: u32, num_refs: u32) -> String {
    format!(
        "MOTION\tsbwidth={sbw}\tsbheight={sbh}\tblwidth={}\tblheight={}\tnum_refs={num_refs}",
        4 * sbw,
        4 * sbh
    )
}

/// `MOTION_BLOCK` — once per superblock.
pub(crate) fn format_motion_block(sb_x: u32, sb_y: u32, split: u32) -> String {
    format!(
        "MOTION_BLOCK\tsb_x={sb_x}\tsb_y={sb_y}\tsplit={split}\tblkcnt={}\tstep={}",
        1u32 << split,
        4u32 >> split
    )
}

/// `MOTION_GLOBAL` — once per reference when the picture uses global
/// motion (§11.2.6).
pub(crate) fn format_motion_global(
    ref_idx: u32,
    g: &crate::picture_inter::GlobalParams,
    mv_precision: u32,
) -> String {
    format!(
        "MOTION_GLOBAL\tref={ref_idx}\tuse_global=1\tpan={},{}\tzrs={},{},{},{}\tzrs_exp={}\tperspective={},{}\tpersp_exp={}\tmv_precision={mv_precision}",
        g.pan_tilt.0,
        g.pan_tilt.1,
        g.zrs[0][0],
        g.zrs[0][1],
        g.zrs[1][0],
        g.zrs[1][1],
        g.zrs_exp,
        g.perspective.0,
        g.perspective.1,
        g.persp_exp,
    )
}

/// Captured predictor/residual pair for one coded MV component pair.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MvSide {
    /// Whether this reference's vector was actually decoded from the
    /// wire for this block (false for intra / unused-ref / global).
    pub coded: bool,
    pub pred: (i32, i32),
    pub res: (i32, i32),
}

/// `MOTION_MV` — one line per decoded prediction block. `mvN` /
/// `mvN_pred` / `mvN_res` appear only for references the block uses;
/// predictor and residual are omitted for global blocks, whose vector
/// comes from the affine field rather than a coded residual.
#[allow(clippy::too_many_arguments)]
pub(crate) fn format_motion_mv(
    sb_x: u32,
    sb_y: u32,
    bx: u32,
    by: u32,
    block: &crate::picture_inter::BlockData,
    sides: &[MvSide; 2],
) -> String {
    let refmask = block.rmode.to_bits();
    let glob = block.gmode as u32;
    let mut line = format!(
        "MOTION_MV\tsb_x={sb_x}\tsb_y={sb_y}\tbx={bx}\tby={by}\tref={refmask}\tglob0={glob}\tglob1={glob}"
    );
    for r in 0..2usize {
        if refmask >> r & 1 == 0 {
            continue;
        }
        let (dx, dy) = block.mv[r];
        line.push_str(&format!("\tmv{r}={dx},{dy}"));
        if sides[r].coded {
            let (px, py) = sides[r].pred;
            let (rx, ry) = sides[r].res;
            line.push_str(&format!("\tmv{r}_pred={px},{py}\tmv{r}_res={rx},{ry}"));
        }
    }
    line.push_str(&format!(
        "\tdc={},{},{}",
        block.dc[0], block.dc[1], block.dc[2]
    ));
    line
}

/// `MC_PLANE` — SHA-256 of the pre-clip, pre-residual OBMC prediction
/// plane, samples as little-endian `i16` in row-major order.
pub(crate) fn format_mc_plane(
    pic_num: u32,
    plane: usize,
    width: usize,
    height: usize,
    samples: &[i16],
) -> String {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for s in samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    let digest = sha256_hex(&bytes);
    format!(
        "MC_PLANE\tpic_num={pic_num}\tplane={plane}\twidth={width}\theight={height}\tstride={width}\tsha256={digest}"
    )
}

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4) — self-contained so the trace stays dependency-free.
// ---------------------------------------------------------------------------

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA-256 of `data`, lowercase hex (FIPS 180-4).
pub(crate) fn sha256_hex(data: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    // Padding: 0x80, zeros, 64-bit big-endian bit length.
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    let mut w = [0u32; 64];
    for chunk in msg.chunks_exact(64) {
        for (i, word) in w.iter_mut().take(16).enumerate() {
            *word = u32::from_be_bytes(chunk[4 * i..4 * i + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut out = String::with_capacity(64);
    for word in h {
        out.push_str(&format!("{word:08x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::picture_inter::{BlockData, RefPredMode};

    /// FIPS 180-4 test vectors.
    #[test]
    fn sha256_fips_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    /// The MOTION_MV line carries the fields the trace contract
    /// specifies: block-grid keys, reference mask, global flags, per-ref
    /// final MV plus predictor/residual for coded refs, and the DC
    /// triple.
    #[test]
    fn motion_mv_line_matches_contract_shape() {
        let block = BlockData {
            rmode: RefPredMode::Ref1Only,
            gmode: false,
            mv: [(9, -4), (0, 0)],
            dc: [0, 0, 0],
        };
        let sides = [
            MvSide {
                coded: true,
                pred: (7, -3),
                res: (2, -1),
            },
            MvSide::default(),
        ];
        let line = format_motion_mv(3, 2, 13, 8, &block, &sides);
        assert_eq!(
            line,
            "MOTION_MV\tsb_x=3\tsb_y=2\tbx=13\tby=8\tref=1\tglob0=0\tglob1=0\tmv0=9,-4\tmv0_pred=7,-3\tmv0_res=2,-1\tdc=0,0,0"
        );
    }

    /// Intra blocks (ref == 0) carry no MV keys — only the DC triple.
    #[test]
    fn motion_mv_intra_block_has_only_dc() {
        let block = BlockData {
            rmode: RefPredMode::Intra,
            gmode: false,
            mv: [(0, 0); 2],
            dc: [12, -3, 7],
        };
        let sides = [MvSide::default(), MvSide::default()];
        let line = format_motion_mv(0, 0, 0, 0, &block, &sides);
        assert_eq!(
            line,
            "MOTION_MV\tsb_x=0\tsb_y=0\tbx=0\tby=0\tref=0\tglob0=0\tglob1=0\tdc=12,-3,7"
        );
    }

    /// Global blocks report their final (affine-derived) MV but omit the
    /// predictor/residual pair, per the contract.
    #[test]
    fn motion_mv_global_block_omits_pred_res() {
        let block = BlockData {
            rmode: RefPredMode::Ref1And2,
            gmode: true,
            mv: [(4, 4), (-2, 0)],
            dc: [0, 0, 0],
        };
        let sides = [MvSide::default(), MvSide::default()];
        let line = format_motion_mv(1, 1, 5, 6, &block, &sides);
        assert_eq!(
            line,
            "MOTION_MV\tsb_x=1\tsb_y=1\tbx=5\tby=6\tref=3\tglob0=1\tglob1=1\tmv0=4,4\tmv1=-2,0\tdc=0,0,0"
        );
    }

    #[test]
    fn motion_and_block_lines_derive_counts() {
        assert_eq!(
            format_motion(23, 18, 1),
            "MOTION\tsbwidth=23\tsbheight=18\tblwidth=92\tblheight=72\tnum_refs=1"
        );
        assert_eq!(
            format_motion_block(4, 7, 2),
            "MOTION_BLOCK\tsb_x=4\tsb_y=7\tsplit=2\tblkcnt=4\tstep=1"
        );
    }

    #[test]
    fn mc_plane_line_hashes_le_samples() {
        // Two samples 0x0102, -1 → bytes 02 01 ff ff.
        let line = format_mc_plane(3, 0, 2, 1, &[0x0102, -1]);
        let expect = sha256_hex(&[0x02, 0x01, 0xff, 0xff]);
        assert_eq!(
            line,
            format!("MC_PLANE\tpic_num=3\tplane=0\twidth=2\theight=1\tstride=2\tsha256={expect}")
        );
    }
}
