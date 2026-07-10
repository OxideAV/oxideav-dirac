//! Inverse wavelet transform (§15.6).
//!
//! A Dirac picture is built from wavelet subbands arranged in a pyramid
//! of 1 + 3 * DWT_DEPTH subbands (§13.1). The decoder reverses the
//! forward transform by iteratively combining the LL band at level
//! `n − 1` with the three high-pass bands HL/LH/HH at level `n` into a
//! new LL band that's twice as wide and twice as tall.
//!
//! Each 2-D synthesis stage:
//!
//! 1. **Interleave** the four sub-arrays into one (§15.6.1 step 2).
//! 2. Apply a **1-D vertical** synthesis on every column (§15.6.2).
//! 3. Apply a **1-D horizontal** synthesis on every row.
//! 4. Optionally right-shift the result by `filtershift()` bits to drop
//!    accuracy bits.
//!
//! The 1-D synthesis is a sequence of reversible **lifting** filters
//! (types 1-4, Annex G). Four filter types differ in which parity gets
//! updated (even vs. odd index) and whether the update is add or
//! subtract. For integer reversibility the spec is very particular
//! about the rounding offset and edge-extension rules, so we mirror
//! §15.6.2 exactly.
//!
//! Tables 15.1-15.7 define seven wavelet filters; here we implement
//! the three most common (DD9_7, LeGall 5/3, Haar) plus DD13_7, the
//! two Haar variants, Fidelity, and Daubechies 9/7. They share the
//! [`WaveletFilter::lifting_steps`] / [`WaveletFilter::filter_shift`]
//! table that drives [`one_d_synthesis`].

use crate::subband::SubbandData;

/// One of the seven Dirac-defined wavelet filter presets (Table 11.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveletFilter {
    /// 0 — Deslauriers-Dubuc (9,7). The Dirac default.
    DeslauriersDubuc9_7,
    /// 1 — LeGall (5,3). The JPEG 2000 reversible filter.
    LeGall5_3,
    /// 2 — Deslauriers-Dubuc (13,7).
    DeslauriersDubuc13_7,
    /// 3 — Haar with no shift.
    Haar0,
    /// 4 — Haar with single shift per level.
    Haar1,
    /// 5 — Fidelity filter.
    Fidelity,
    /// 6 — Daubechies (9,7) integer approximation.
    Daubechies9_7,
}

impl WaveletFilter {
    /// Map the `state[WAVELET INDEX]` value (0..=6) to a filter. Returns
    /// `None` on out-of-range indices.
    pub fn from_index(idx: u32) -> Option<Self> {
        Some(match idx {
            0 => Self::DeslauriersDubuc9_7,
            1 => Self::LeGall5_3,
            2 => Self::DeslauriersDubuc13_7,
            3 => Self::Haar0,
            4 => Self::Haar1,
            5 => Self::Fidelity,
            6 => Self::Daubechies9_7,
            _ => return None,
        })
    }

    /// Inverse of [`WaveletFilter::from_index`]: the
    /// `state[WAVELET INDEX]` wire value (0..=6) for this filter.
    pub fn to_index(self) -> u32 {
        match self {
            Self::DeslauriersDubuc9_7 => 0,
            Self::LeGall5_3 => 1,
            Self::DeslauriersDubuc13_7 => 2,
            Self::Haar0 => 3,
            Self::Haar1 => 4,
            Self::Fidelity => 5,
            Self::Daubechies9_7 => 6,
        }
    }

    /// `filter_shift()` (Tables 15.1-15.7): the number of bits to right
    /// shift after each 2-D synthesis pass.
    pub fn filter_shift(self) -> u32 {
        match self {
            Self::DeslauriersDubuc9_7 => 1,
            Self::LeGall5_3 => 1,
            Self::DeslauriersDubuc13_7 => 1,
            Self::Haar0 => 0,
            Self::Haar1 => 1,
            Self::Fidelity => 0,
            Self::Daubechies9_7 => 1,
        }
    }

    /// The ordered sequence of 1-D lifting steps for this filter.
    pub fn lifting_steps(self) -> &'static [LiftingStep] {
        match self {
            // Table 15.1 DD9_7:
            //  1. Type 2, S=2, L=2, D= 0, taps=[ 1,  1]
            //  2. Type 3, S=4, L=4, D=-1, taps=[-1,  9, 9, -1]
            Self::DeslauriersDubuc9_7 => &[
                LiftingStep {
                    kind: LiftKind::Type2,
                    shift: 2,
                    d: 0,
                    taps: &[1, 1],
                },
                LiftingStep {
                    kind: LiftKind::Type3,
                    shift: 4,
                    d: -1,
                    taps: &[-1, 9, 9, -1],
                },
            ],
            // Table 15.2 LeGall 5/3:
            //  1. Type 2, S=2, L=2, D=0, taps=[1, 1]
            //  2. Type 3, S=1, L=2, D=0, taps=[1, 1]
            Self::LeGall5_3 => &[
                LiftingStep {
                    kind: LiftKind::Type2,
                    shift: 2,
                    d: 0,
                    taps: &[1, 1],
                },
                LiftingStep {
                    kind: LiftKind::Type3,
                    shift: 1,
                    d: 0,
                    taps: &[1, 1],
                },
            ],
            // Table 15.3 DD13_7:
            //  1. Type 2, S=5, L=4, D=-1, taps=[-1, 9, 9, -1]
            //  2. Type 3, S=4, L=4, D=-1, taps=[-1, 9, 9, -1]
            Self::DeslauriersDubuc13_7 => &[
                LiftingStep {
                    kind: LiftKind::Type2,
                    shift: 5,
                    d: -1,
                    taps: &[-1, 9, 9, -1],
                },
                LiftingStep {
                    kind: LiftKind::Type3,
                    shift: 4,
                    d: -1,
                    taps: &[-1, 9, 9, -1],
                },
            ],
            // Table 15.4 Haar no shift:
            //  1. Type 2, S=1, L=1, D=1, taps=[1]
            //  2. Type 3, S=0, L=1, D=0, taps=[1]
            Self::Haar0 | Self::Haar1 => &[
                LiftingStep {
                    kind: LiftKind::Type2,
                    shift: 1,
                    d: 1,
                    taps: &[1],
                },
                LiftingStep {
                    kind: LiftKind::Type3,
                    shift: 0,
                    d: 0,
                    taps: &[1],
                },
            ],
            // Table 15.6 Fidelity:
            //  1. Type 3, S=8, L=8, D=-3, taps=[-2,10,-25,81,81,-25,10,-2]
            //  2. Type 2, S=8, L=8, D=-3, taps=[-8,21,-46,161,161,-46,21,-8]
            Self::Fidelity => &[
                LiftingStep {
                    kind: LiftKind::Type3,
                    shift: 8,
                    d: -3,
                    taps: &[-2, 10, -25, 81, 81, -25, 10, -2],
                },
                LiftingStep {
                    kind: LiftKind::Type2,
                    shift: 8,
                    d: -3,
                    taps: &[-8, 21, -46, 161, 161, -46, 21, -8],
                },
            ],
            // Table 15.7 Daubechies (9,7) integer approx:
            //  1. Type 2, S=12, L=2, D=0, taps=[1817, 1817]
            //  2. Type 4, S=12, L=2, D=0, taps=[3616, 3616]
            //  3. Type 1, S=12, L=2, D=0, taps=[ 217,  217]
            //  4. Type 3, S=12, L=2, D=0, taps=[6497, 6497]
            Self::Daubechies9_7 => &[
                LiftingStep {
                    kind: LiftKind::Type2,
                    shift: 12,
                    d: 0,
                    taps: &[1817, 1817],
                },
                LiftingStep {
                    kind: LiftKind::Type4,
                    shift: 12,
                    d: 0,
                    taps: &[3616, 3616],
                },
                LiftingStep {
                    kind: LiftKind::Type1,
                    shift: 12,
                    d: 0,
                    taps: &[217, 217],
                },
                LiftingStep {
                    kind: LiftKind::Type3,
                    shift: 12,
                    d: 0,
                    taps: &[6497, 6497],
                },
            ],
        }
    }
}

/// One of the four lifting filter kinds (§15.6.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiftKind {
    /// `A[2n] += (sum(t[i]*A[2(n+i)+1]) + (1<<(s-1))) >> s`
    Type1,
    /// `A[2n] -= (sum(t[i]*A[2(n+i)+1]) + (1<<(s-1))) >> s`
    Type2,
    /// `A[2n+1] += (sum(t[i]*A[2(n+i)]) + (1<<(s-1))) >> s`
    Type3,
    /// `A[2n+1] -= (sum(t[i]*A[2(n+i)]) + (1<<(s-1))) >> s`
    Type4,
}

/// A single integer lifting step (Tables 15.1-15.7).
#[derive(Debug, Clone, Copy)]
pub struct LiftingStep {
    pub kind: LiftKind,
    /// Right-shift after the weighted sum. `shift == 0` disables the
    /// `+ (1 << (s-1))` rounding term in Type 1 too (§15.6.2).
    pub shift: u32,
    /// Index offset `D` into `taps` (§15.6.2, Annex G).
    pub d: i32,
    /// Tap coefficients.
    pub taps: &'static [i32],
}

/// Apply a 1-D lifting filter to `a` in place.
///
/// Spec §15.6.2. For each half-length `n = 0..length/2`:
///
/// * Types 1, 2 update the **even** sample at index `2n`, summing
///   `taps[i-D] * A[2(n+i)-1]` with positions clamped between 1 and
///   `length-1`.
/// * Types 3, 4 update the **odd** sample at index `2n+1`, summing
///   `taps[i-D] * A[2(n+i)]` with positions clamped between 0 and
///   `length-2`.
///
/// Positions are clamped by `min(pos, ...)` then `max(pos, ...)` — the
/// spec's edge-extension rule.
fn apply_lift(a: &mut [i32], step: &LiftingStep) {
    let length = a.len();
    if length < 2 {
        return;
    }
    let half = length / 2;
    let rounding: i32 = if step.shift > 0 {
        1 << (step.shift - 1)
    } else {
        0
    };
    let d_i32 = step.d;
    let taps = step.taps;
    match step.kind {
        LiftKind::Type1 | LiftKind::Type2 => {
            // pos = 2*(n+i) - 1, clamped to [1, length-1]
            for n in 0..half {
                let mut sum: i32 = 0;
                for (ti, &tap) in taps.iter().enumerate() {
                    let i = d_i32 + ti as i32;
                    let pos = 2 * (n as i32 + i) - 1;
                    let pos = pos.min(length as i32 - 1).max(1) as usize;
                    sum = sum.wrapping_add(tap.wrapping_mul(a[pos]));
                }
                let delta = (sum.wrapping_add(rounding)) >> step.shift;
                let idx = 2 * n;
                if matches!(step.kind, LiftKind::Type1) {
                    a[idx] = a[idx].wrapping_add(delta);
                } else {
                    a[idx] = a[idx].wrapping_sub(delta);
                }
            }
        }
        LiftKind::Type3 | LiftKind::Type4 => {
            // pos = 2*(n+i), clamped to [0, length-2]
            for n in 0..half {
                let mut sum: i32 = 0;
                for (ti, &tap) in taps.iter().enumerate() {
                    let i = d_i32 + ti as i32;
                    let pos = 2 * (n as i32 + i);
                    let pos = pos.min(length as i32 - 2).max(0) as usize;
                    sum = sum.wrapping_add(tap.wrapping_mul(a[pos]));
                }
                let delta = (sum.wrapping_add(rounding)) >> step.shift;
                let idx = 2 * n + 1;
                if matches!(step.kind, LiftKind::Type3) {
                    a[idx] = a[idx].wrapping_add(delta);
                } else {
                    a[idx] = a[idx].wrapping_sub(delta);
                }
            }
        }
    }
}

/// One-dimensional synthesis: apply every lifting step of `filter` in
/// order to the row / column `a` (§15.6.2, final paragraph).
pub fn one_d_synthesis(a: &mut [i32], filter: WaveletFilter) {
    for step in filter.lifting_steps() {
        apply_lift(a, step);
    }
}

/// Interleave and 2-D synthesise four equal-sized subbands `ll`, `hl`,
/// `lh`, `hh` into a new `2w x 2h` array (§15.6.1).
///
/// The four inputs must all share the same dimensions. Returns a
/// `SubbandData` whose width / height are exactly doubled. The lifting
/// runs first down every column, then across every row. Finally, if
/// `filter_shift() > 0`, the entire array is rounded and right-shifted.
///
/// Round 195 profile-driven optimisation: the body is rewritten to drive
/// the row-major backing `Vec<i32>` directly instead of going through
/// `SubbandData::{get,set}`. The interleave loop pre-slices the input
/// rows once per output row-pair, eliminating four bounds-checks per
/// output sample. The vertical lifting pass uses a scratch buffer so the
/// `one_d_synthesis` inner loop sees a contiguous slice — but the
/// row-major scatter back into `synth.data` is now a tight indexed write
/// instead of a per-element `set()` call. The post-shift fold collapses
/// into a single `data` iter. Decoder hot path (the `idwt` driver runs
/// this `dwt_depth` times per picture, per component) and encoder hot
/// path (`encode_*_intra_stream` runs the matching `vh_analysis`).
/// Comprehensive bit-exactness is guarded by the seven-filter ×
/// depth-{1,2,3} `dwt_idwt_roundtrip_all_filters_all_depths` test that
/// already lives in this module.
pub fn vh_synth(
    ll: &SubbandData,
    hl: &SubbandData,
    lh: &SubbandData,
    hh: &SubbandData,
    filter: WaveletFilter,
) -> SubbandData {
    debug_assert_eq!(ll.width, hl.width);
    debug_assert_eq!(ll.width, lh.width);
    debug_assert_eq!(ll.width, hh.width);
    debug_assert_eq!(ll.height, hl.height);
    debug_assert_eq!(ll.height, lh.height);
    debug_assert_eq!(ll.height, hh.height);
    let w = ll.width;
    let h = ll.height;
    let out_w = 2 * w;
    let out_h = 2 * h;
    let mut synth = SubbandData::new(out_w, out_h);
    // Step 2: interleave. Drive the row-major backing slices directly
    // so the compiler can elide the per-element bounds checks `set()`
    // would issue.
    {
        let dst = &mut synth.data[..];
        let ll_d = &ll.data[..];
        let hl_d = &hl.data[..];
        let lh_d = &lh.data[..];
        let hh_d = &hh.data[..];
        for y in 0..h {
            let in_off = y * w;
            let out_off = (2 * y) * out_w;
            let ll_row = &ll_d[in_off..in_off + w];
            let hl_row = &hl_d[in_off..in_off + w];
            let lh_row = &lh_d[in_off..in_off + w];
            let hh_row = &hh_d[in_off..in_off + w];
            let two_rows = &mut dst[out_off..out_off + 2 * out_w];
            let (dst_even, dst_odd) = two_rows.split_at_mut(out_w);
            for x in 0..w {
                dst_even[2 * x] = ll_row[x];
                dst_even[2 * x + 1] = hl_row[x];
                dst_odd[2 * x] = lh_row[x];
                dst_odd[2 * x + 1] = hh_row[x];
            }
        }
    }
    // Step 3a: vertical synthesis — one lifting pass per column.
    // `col_buf` is reused across columns. Gather / scatter use indexed
    // raw access into `synth.data` so the compiler sees the
    // monotonic-stride pattern.
    let mut col_buf = vec![0i32; out_h];
    {
        let data = &mut synth.data[..];
        for x in 0..out_w {
            for y in 0..out_h {
                col_buf[y] = data[y * out_w + x];
            }
            one_d_synthesis(&mut col_buf, filter);
            for y in 0..out_h {
                data[y * out_w + x] = col_buf[y];
            }
        }
    }
    // Step 3b: horizontal synthesis — one lifting pass per row, in
    // place on a contiguous slice (already optimal).
    for y in 0..out_h {
        let row = synth.row_mut(y);
        one_d_synthesis(row, filter);
    }
    // Step 4: accuracy-bit shift.
    let shift = filter.filter_shift();
    if shift > 0 {
        let round = 1i32 << (shift - 1);
        for v in synth.data.iter_mut() {
            *v = v.wrapping_add(round) >> shift;
        }
    }
    synth
}

/// Full picture IDWT: start from the level-0 LL band, iteratively
/// combine with each level's HL/LH/HH bands to produce a picture-sized
/// coefficient array (§15.6, `idwt_synthesis`).
///
/// `pyramid[0][0]` is the level-0 LL (DC); `pyramid[level][orient]` for
/// level >= 1 gives the HL/LH/HH bands at that level. `orient` maps
/// 0 => HL, 1 => LH, 2 => HH (see [`crate::subband::Orient`]).
pub fn idwt(pyramid: &[[SubbandData; 4]], filter: WaveletFilter) -> SubbandData {
    debug_assert!(!pyramid.is_empty());
    let mut ll = pyramid[0][0].clone();
    for level_bands in pyramid.iter().skip(1) {
        ll = vh_synth(
            &ll,
            &level_bands[1],
            &level_bands[2],
            &level_bands[3],
            filter,
        );
    }
    ll
}

/// Asymmetric / horizontal-only IDWT driver — SMPTE ST 2042-1:2022
/// §15.4.1 `idwt(state, coeff_data)` for the
/// `state[dwt_depth_ho] > 0` branch.
///
/// The v3 stream-syntax (`major_version >= 3`) allows the §12.4.4
/// `extended_transform_parameters()` block to select a
/// **horizontal-only** filter (`state[wavelet_index_ho]`) and an extra
/// horizontal-only depth (`state[dwt_depth_ho]`) on top of the regular
/// symmetric `state[dwt_depth]` 2-D levels. The §15.4.1 process then:
///
/// 1. Starts the DC band from `coeff_data[0][L]` (a 1-D low-pass band,
///    not `[0][LL]`).
/// 2. For `n = 1..=dwt_depth_ho`: invokes [`h_synth`] with
///    `state[wavelet_index_ho]` against `coeff_data[n][H]` — each step
///    doubles only the width.
/// 3. For `n = dwt_depth_ho + 1..=dwt_depth_ho + dwt_depth`: invokes
///    [`vh_synth`] with `state[wavelet_index]` against
///    `coeff_data[n][HL/LH/HH]` — each step doubles both axes.
///
/// `pyramid` follows the §13.2.2 subband layout:
///
/// * `pyramid[0][0]` carries the level-0 **L** band (used as the
///   horizontal-only DC seed).
/// * `pyramid[n][3]` carries the **H** band at level `n` for
///   `1 <= n <= dwt_depth_ho`. (Slot index 3 is reused to avoid
///   reshaping the existing `[SubbandData; 4]` quartet — the other
///   slots at horizontal-only levels are unused placeholders.)
/// * `pyramid[n][1..=3]` carries HL / LH / HH at level `n` for
///   `dwt_depth_ho + 1 <= n <= dwt_depth_ho + dwt_depth`.
///
/// When `dwt_depth_ho == 0` the function is byte-equivalent to
/// [`idwt`] called with `filter_v` — `pyramid[0][0]` is then the
/// classic LL band and `filter_ho` is unused (the §15.4.1 first
/// `for` loop runs zero iterations).
///
/// `filter_v` corresponds to `state[wavelet_index]` (drives the 2-D
/// `vh_synthesis`); `filter_ho` corresponds to
/// `state[wavelet_index_ho]` (drives the 1-D horizontal-only
/// `h_synthesis`). Per §12.4.4.2, the two indices can differ even when
/// neither is the §12.4.4 NOTE "symmetric default".
pub fn idwt_with_ho(
    pyramid: &[[SubbandData; 4]],
    filter_v: WaveletFilter,
    filter_ho: WaveletFilter,
    dwt_depth_ho: u32,
) -> SubbandData {
    debug_assert!(!pyramid.is_empty());
    let ho = dwt_depth_ho as usize;
    debug_assert!(
        pyramid.len() > ho,
        "pyramid is shorter than dwt_depth_ho asks for"
    );
    let mut dc = pyramid[0][0].clone();
    // §15.4.1 horizontal-only loop: n = 1..=dwt_depth_ho. The level-n
    // entry's H band lives in slot 3 (see doc-comment above for why we
    // reuse that slot rather than changing the pyramid quartet shape).
    for n in 1..=ho {
        dc = h_synth(&dc, &pyramid[n][3], filter_ho);
    }
    // §15.4.1 vertical+horizontal loop: n = dwt_depth_ho + 1..=
    // dwt_depth_ho + dwt_depth. Iterating over the remaining pyramid
    // entries reaches exactly that range.
    for level_bands in pyramid.iter().skip(ho + 1) {
        dc = vh_synth(
            &dc,
            &level_bands[1],
            &level_bands[2],
            &level_bands[3],
            filter_v,
        );
    }
    dc
}

/// One-dimensional **horizontal-only** synthesis stage — SMPTE ST
/// 2042-1:2022 §15.4.2 `h_synthesis(state, L_data, H_data)`.
///
/// The horizontal-only IDWT step combines two equal-shape subbands — a
/// low-pass `L` and a high-pass `H` — into an output array that is
/// **twice as wide and the same height**. Both inputs must share the
/// same `(width, height)`.
///
/// The pseudocode is:
///
/// 1. `synth = new_array(height(L_data), 2 * width(L_data))`
///    (§15.4.2 step 1).
/// 2. For each row `y` and column `x`:
///    `synth[y][2*x]     = L_data[y][x]`
///    `synth[y][2*x + 1] = H_data[y][x]`
///    (§15.4.2 step 2 — horizontal interleave).
/// 3. For each row `y`: `oned_synthesis(row(synth, y), state[wavelet_index_ho])`
///    (§15.4.2 step 3 — one-dimensional synthesis along the rows only;
///    columns are untouched).
/// 4. If `filter_bit_shift(state[wavelet_index_ho]) > 0`, rounding
///    right-shift every sample by that many bits (§15.4.2 step 4).
///
/// The `filter` argument carries the §12.4.4 horizontal-only filter
/// (`state[wavelet_index_ho]` — Tables 16-22) — independent of the
/// fully-2-D `state[wavelet_index]` consumed by [`vh_synth`]. Same
/// seven-filter palette in either role.
///
/// This is the §15.4.2 building block needed to support the §12.4.4
/// asymmetric (horizontal-only) transform path: when
/// `state[dwt_depth_ho] > 0`, the §15.4.1 IDWT runs `dwt_depth_ho`
/// invocations of `h_synthesis` (working from `coeff_data[0][L]` /
/// `coeff_data[n][H]`) before switching to [`vh_synth`] for the
/// remaining `state[dwt_depth]` levels.
pub fn h_synth(l: &SubbandData, h: &SubbandData, filter: WaveletFilter) -> SubbandData {
    debug_assert_eq!(l.width, h.width);
    debug_assert_eq!(l.height, h.height);
    let in_w = l.width;
    let in_h = l.height;
    let out_w = 2 * in_w;
    let out_h = in_h;
    let mut synth = SubbandData::new(out_w, out_h);
    // Step 2: horizontal interleave — even columns from L, odd from H.
    {
        let dst = &mut synth.data[..];
        let l_d = &l.data[..];
        let h_d = &h.data[..];
        for y in 0..in_h {
            let in_off = y * in_w;
            let out_off = y * out_w;
            let l_row = &l_d[in_off..in_off + in_w];
            let h_row = &h_d[in_off..in_off + in_w];
            let dst_row = &mut dst[out_off..out_off + out_w];
            for x in 0..in_w {
                dst_row[2 * x] = l_row[x];
                dst_row[2 * x + 1] = h_row[x];
            }
        }
    }
    // Step 3: 1-D synthesis along every row — already a contiguous slice.
    // No vertical synthesis: that's the whole point of the horizontal-only
    // path (§12.4.4.3).
    for y in 0..out_h {
        let row = synth.row_mut(y);
        one_d_synthesis(row, filter);
    }
    // Step 4: accuracy-bit rounding shift.
    let shift = filter.filter_shift();
    if shift > 0 {
        let round = 1i32 << (shift - 1);
        for v in synth.data.iter_mut() {
            *v = v.wrapping_add(round) >> shift;
        }
    }
    synth
}

/// Forward horizontal-only analysis — exact inverse of [`h_synth`].
///
/// Splits a `2w × h` picture-shaped array into two equal-shape `w × h`
/// subbands `(L, H)`. Used by encoder paths that want to produce a
/// horizontal-only stage, and by round-trip tests that need to confirm
/// [`h_synth`] is integer-reversible for every spec filter.
///
/// Mirrors the four-step structure of [`vh_analysis`] without the
/// vertical pass:
///
/// 1. Left-shift every sample by `filter_shift()` (undo the accuracy-bit
///    drop §15.4.2 step 4 inserted).
/// 2. One-dimensional analysis along every row.
/// 3. De-interleave: column `2x` → `L[y][x]`, column `2x + 1` → `H[y][x]`.
pub fn h_analysis(picture: &SubbandData, filter: WaveletFilter) -> (SubbandData, SubbandData) {
    debug_assert!(picture.width % 2 == 0);
    let out_w = picture.width;
    let out_h = picture.height;
    let mut work = picture.clone();
    // Step 1: undo the accuracy-bit shift.
    let shift = filter.filter_shift();
    if shift > 0 {
        for v in work.data.iter_mut() {
            *v = v.wrapping_shl(shift);
        }
    }
    // Step 2: horizontal analysis per row.
    for y in 0..out_h {
        let row = work.row_mut(y);
        one_d_analysis(row, filter);
    }
    // Step 3: de-interleave.
    let half_w = out_w / 2;
    let mut l = SubbandData::new(half_w, out_h);
    let mut h = SubbandData::new(half_w, out_h);
    {
        let src = &work.data[..];
        let l_d = &mut l.data[..];
        let h_d = &mut h.data[..];
        for y in 0..out_h {
            let src_off = y * out_w;
            let dst_off = y * half_w;
            let src_row = &src[src_off..src_off + out_w];
            let l_row = &mut l_d[dst_off..dst_off + half_w];
            let h_row = &mut h_d[dst_off..dst_off + half_w];
            for x in 0..half_w {
                l_row[x] = src_row[2 * x];
                h_row[x] = src_row[2 * x + 1];
            }
        }
    }
    (l, h)
}

// --------------------------------------------------------------------
//   Forward (analysis) transform — the exact inverse of the synthesis
//   path used by the decoder above. This is what the encoder runs on
//   each picture component before quantising the coefficients.
// --------------------------------------------------------------------

/// Invert a single lifting step: the synthesis applies `A[even] -= d`
/// (Type 2) and `A[odd] += d` (Type 3) using the tap-weighted sum of
/// neighbouring samples. The analysis does the opposite — Type 2
/// becomes Type 1 (add), Type 3 becomes Type 4 (subtract), and the
/// step is executed in reverse order.
fn apply_inverse_lift(a: &mut [i32], step: &LiftingStep) {
    let length = a.len();
    if length < 2 {
        return;
    }
    let half = length / 2;
    let rounding: i32 = if step.shift > 0 {
        1 << (step.shift - 1)
    } else {
        0
    };
    let d_i32 = step.d;
    let taps = step.taps;
    match step.kind {
        // Synthesis Type 1 adds delta to even; analysis subtracts it.
        // Synthesis Type 2 subtracts delta from even; analysis adds it.
        LiftKind::Type1 | LiftKind::Type2 => {
            for n in 0..half {
                let mut sum: i32 = 0;
                for (ti, &tap) in taps.iter().enumerate() {
                    let i = d_i32 + ti as i32;
                    let pos = 2 * (n as i32 + i) - 1;
                    let pos = pos.min(length as i32 - 1).max(1) as usize;
                    sum = sum.wrapping_add(tap.wrapping_mul(a[pos]));
                }
                let delta = (sum.wrapping_add(rounding)) >> step.shift;
                let idx = 2 * n;
                // Opposite sign vs. apply_lift.
                if matches!(step.kind, LiftKind::Type1) {
                    a[idx] = a[idx].wrapping_sub(delta);
                } else {
                    a[idx] = a[idx].wrapping_add(delta);
                }
            }
        }
        // Synthesis Type 3 adds delta to odd; analysis subtracts.
        // Synthesis Type 4 subtracts delta from odd; analysis adds.
        LiftKind::Type3 | LiftKind::Type4 => {
            for n in 0..half {
                let mut sum: i32 = 0;
                for (ti, &tap) in taps.iter().enumerate() {
                    let i = d_i32 + ti as i32;
                    let pos = 2 * (n as i32 + i);
                    let pos = pos.min(length as i32 - 2).max(0) as usize;
                    sum = sum.wrapping_add(tap.wrapping_mul(a[pos]));
                }
                let delta = (sum.wrapping_add(rounding)) >> step.shift;
                let idx = 2 * n + 1;
                if matches!(step.kind, LiftKind::Type3) {
                    a[idx] = a[idx].wrapping_sub(delta);
                } else {
                    a[idx] = a[idx].wrapping_add(delta);
                }
            }
        }
    }
}

/// One-dimensional analysis: apply every lifting step of `filter` in
/// reverse order with inverted sign. This is the exact inverse of
/// [`one_d_synthesis`].
pub fn one_d_analysis(a: &mut [i32], filter: WaveletFilter) {
    for step in filter.lifting_steps().iter().rev() {
        apply_inverse_lift(a, step);
    }
}

/// Forward 2-D analysis: split a `2w x 2h` picture-shaped array into
/// four equal-sized LL / HL / LH / HH subbands.
///
/// This is the inverse of [`vh_synth`]. The steps run in reverse order:
///
/// 1. Left-shift by `filter_shift()` (cancelling the synthesis's
///    accuracy-bit drop).
/// 2. Horizontal analysis — one 1-D analysis per row.
/// 3. Vertical analysis — one 1-D analysis per column.
/// 4. De-interleave: samples at (2y, 2x), (2y, 2x+1), (2y+1, 2x),
///    (2y+1, 2x+1) become LL, HL, LH, HH respectively.
///
/// The returned tuple is `(ll, hl, lh, hh)`.
pub fn vh_analysis(
    picture: &SubbandData,
    filter: WaveletFilter,
) -> (SubbandData, SubbandData, SubbandData, SubbandData) {
    debug_assert!(picture.width % 2 == 0);
    debug_assert!(picture.height % 2 == 0);
    let out_w = picture.width;
    let out_h = picture.height;
    let mut work = picture.clone();
    // Step 1: undo the accuracy-bit shift.
    let shift = filter.filter_shift();
    if shift > 0 {
        for v in work.data.iter_mut() {
            *v = v.wrapping_shl(shift);
        }
    }
    // Step 2: horizontal analysis (in place on contiguous rows).
    for y in 0..out_h {
        let row = work.row_mut(y);
        one_d_analysis(row, filter);
    }
    // Step 3: vertical analysis. Round 195 profile-driven optimisation:
    // gather / scatter use indexed raw access into `work.data` instead
    // of `get` / `set` so the bounds-check elision applies.
    let mut col_buf = vec![0i32; out_h];
    {
        let data = &mut work.data[..];
        for x in 0..out_w {
            for y in 0..out_h {
                col_buf[y] = data[y * out_w + x];
            }
            one_d_analysis(&mut col_buf, filter);
            for y in 0..out_h {
                data[y * out_w + x] = col_buf[y];
            }
        }
    }
    // Step 4: de-interleave. Round 195 profile-driven optimisation:
    // pre-slice the source two rows + four destination rows once per
    // output-row index, dropping the four bounds-checked `get` +
    // four `set` calls per output sample.
    let half_w = out_w / 2;
    let half_h = out_h / 2;
    let mut ll = SubbandData::new(half_w, half_h);
    let mut hl = SubbandData::new(half_w, half_h);
    let mut lh = SubbandData::new(half_w, half_h);
    let mut hh = SubbandData::new(half_w, half_h);
    {
        let src = &work.data[..];
        let ll_d = &mut ll.data[..];
        let hl_d = &mut hl.data[..];
        let lh_d = &mut lh.data[..];
        let hh_d = &mut hh.data[..];
        for y in 0..half_h {
            let src_off = (2 * y) * out_w;
            let two_rows = &src[src_off..src_off + 2 * out_w];
            let (src_even, src_odd) = two_rows.split_at(out_w);
            let dst_off = y * half_w;
            let ll_row = &mut ll_d[dst_off..dst_off + half_w];
            let hl_row = &mut hl_d[dst_off..dst_off + half_w];
            let lh_row = &mut lh_d[dst_off..dst_off + half_w];
            let hh_row = &mut hh_d[dst_off..dst_off + half_w];
            for x in 0..half_w {
                ll_row[x] = src_even[2 * x];
                hl_row[x] = src_even[2 * x + 1];
                lh_row[x] = src_odd[2 * x];
                hh_row[x] = src_odd[2 * x + 1];
            }
        }
    }
    (ll, hl, lh, hh)
}

/// Full picture DWT: given a picture-sized coefficient array (already
/// padded up to a multiple of `2^dwt_depth` per §15.7 and trimmed with
/// the `2^(depth-1)` offset subtracted), iteratively decompose it into
/// a pyramid of subbands matching the layout [`idwt`] expects.
///
/// The returned pyramid has the same shape as [`crate::subband::init_pyramid`]:
/// `pyramid[0][0]` holds the level-0 LL band, and `pyramid[level][1..=3]`
/// for `level >= 1` holds HL / LH / HH. Level 0's HL/LH/HH entries and
/// every level's LL slot for `level >= 1` are empty placeholders.
pub fn dwt(picture: &SubbandData, filter: WaveletFilter, dwt_depth: u32) -> Vec<[SubbandData; 4]> {
    let mut pyramid: Vec<[SubbandData; 4]> = Vec::with_capacity(dwt_depth as usize + 1);
    for _ in 0..=dwt_depth {
        pyramid.push([
            SubbandData::new(0, 0),
            SubbandData::new(0, 0),
            SubbandData::new(0, 0),
            SubbandData::new(0, 0),
        ]);
    }
    // Start from the full-resolution signal and repeatedly split its LL
    // sub-sample down by two until we've peeled off `dwt_depth` detail
    // levels.
    let mut ll = picture.clone();
    for level in (1..=dwt_depth).rev() {
        let (new_ll, hl, lh, hh) = vh_analysis(&ll, filter);
        pyramid[level as usize][1] = hl;
        pyramid[level as usize][2] = lh;
        pyramid[level as usize][3] = hh;
        ll = new_ll;
    }
    pyramid[0][0] = ll;
    pyramid
}

/// Forward analysis companion to [`idwt_with_ho`] — peels off
/// `dwt_depth` symmetric levels first (matching the inverse order of
/// the §15.4.1 synthesis driver) then `dwt_depth_ho` horizontal-only
/// levels, producing a pyramid laid out exactly as
/// [`idwt_with_ho`] expects.
///
/// `picture` must already be padded to width that's a multiple of
/// `2^(dwt_depth + dwt_depth_ho)` and height that's a multiple of
/// `2^dwt_depth` (per §13.2.3 — the horizontal-only levels do not
/// shrink the height). Used by round-trip tests; the production
/// encoder side of the asymmetric path is gated separately on
/// `EncoderParams::extended_transform_override` (see r212).
pub fn dwt_with_ho(
    picture: &SubbandData,
    filter_v: WaveletFilter,
    filter_ho: WaveletFilter,
    dwt_depth: u32,
    dwt_depth_ho: u32,
) -> Vec<[SubbandData; 4]> {
    let total = (dwt_depth + dwt_depth_ho) as usize;
    let mut pyramid: Vec<[SubbandData; 4]> = Vec::with_capacity(total + 1);
    for _ in 0..=total {
        pyramid.push([
            SubbandData::new(0, 0),
            SubbandData::new(0, 0),
            SubbandData::new(0, 0),
            SubbandData::new(0, 0),
        ]);
    }
    // Inverse of §15.4.1: peel off the outer `vh_analysis` levels first
    // (they are the last to be applied by the synthesis driver).
    let mut dc = picture.clone();
    for level in (1..=dwt_depth).rev() {
        let abs_level = (dwt_depth_ho + level) as usize;
        let (new_dc, hl, lh, hh) = vh_analysis(&dc, filter_v);
        pyramid[abs_level][1] = hl;
        pyramid[abs_level][2] = lh;
        pyramid[abs_level][3] = hh;
        dc = new_dc;
    }
    // Then peel off the horizontal-only levels. The level-n H band
    // lands in slot 3 to keep the pyramid quartet shape unchanged
    // (mirrors the slot assignment [`idwt_with_ho`] reads).
    for level in (1..=dwt_depth_ho).rev() {
        let (new_dc, h) = h_analysis(&dc, filter_ho);
        pyramid[level as usize][3] = h;
        dc = new_dc;
    }
    pyramid[0][0] = dc;
    pyramid
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A round-trip test: apply the forward version of the Haar lifting
    /// (the inverse of the spec's synthesis) and then the synthesis.
    /// We model a full-scale DC input and expect the synthesis of the
    /// resulting sub-sample structure to reproduce it.
    #[test]
    fn haar_one_d_round_trip_constant_signal() {
        // A constant array at some value v should be invariant under
        // a Haar forward+inverse: we approximate by constructing the
        // LL-only ("coefficient") picture and synthesising it.
        // For Haar no-shift: forward transform on constant c would
        // yield LL = c, HL=LH=HH=0. The 1-D synthesis receives
        // [c, 0, c, 0, ...] (interleaved) and should reconstruct
        // [c, c, c, c, ...].
        let mut a = vec![5, 0, 5, 0, 5, 0, 5, 0];
        one_d_synthesis(&mut a, WaveletFilter::Haar0);
        // After lifting-step 1 (Type 2, s=1, taps=[1]): pos = 2n+1 -> 0
        //   A[2n] -= (A[2n+1] + 1) >> 1
        //   for n=0: A[0] -= (A[1]+1)>>1 = 0 => A[0] = 5
        // After step 2 (Type 3, s=0, taps=[1]): A[2n+1] += A[2n]
        //   => A[1] = 5, A[3] = 5, ...
        for v in &a {
            assert_eq!(*v, 5, "expected constant 5, got {a:?}");
        }
    }

    /// LeGall (5,3) round-trip from a single-level 2-D synthesis of a
    /// constant picture. LL band carries the DC; high-pass bands are
    /// zero. The synthesis should reconstruct the original uniform
    /// intensity (after filter_shift right shift by 1 — the encoder
    /// compensates by left-shifting by 1 first; we mimic that by
    /// encoding the DC as 2x the target value).
    #[test]
    fn legall_vh_synth_from_dc_reconstructs_uniform() {
        let w = 4;
        let h = 4;
        let mut ll = SubbandData::new(w, h);
        // Pre-shift the DC up by filter_shift = 1.
        for y in 0..h {
            for x in 0..w {
                ll.set(y, x, 42 << 1);
            }
        }
        let zeros = SubbandData::new(w, h);
        let out = vh_synth(&ll, &zeros, &zeros, &zeros, WaveletFilter::LeGall5_3);
        for y in 0..out.height {
            for x in 0..out.width {
                assert_eq!(out.get(y, x), 42, "({x},{y}) got {}", out.get(y, x));
            }
        }
    }

    /// DD9_7 — same invariant as the LeGall test: a pure DC in the LL
    /// band with zero high-pass bands should reconstruct the DC value.
    #[test]
    fn dd97_vh_synth_from_dc_reconstructs_uniform() {
        let w = 8;
        let h = 8;
        let mut ll = SubbandData::new(w, h);
        for y in 0..h {
            for x in 0..w {
                ll.set(y, x, 17 << 1);
            }
        }
        let zeros = SubbandData::new(w, h);
        let out = vh_synth(
            &ll,
            &zeros,
            &zeros,
            &zeros,
            WaveletFilter::DeslauriersDubuc9_7,
        );
        for y in 0..out.height {
            for x in 0..out.width {
                assert_eq!(out.get(y, x), 17, "({x},{y}) got {}", out.get(y, x));
            }
        }
    }

    #[test]
    fn filter_index_mapping() {
        assert_eq!(
            WaveletFilter::from_index(0),
            Some(WaveletFilter::DeslauriersDubuc9_7)
        );
        assert_eq!(WaveletFilter::from_index(1), Some(WaveletFilter::LeGall5_3));
        assert_eq!(WaveletFilter::from_index(3), Some(WaveletFilter::Haar0));
        assert_eq!(WaveletFilter::from_index(4), Some(WaveletFilter::Haar1));
        assert_eq!(
            WaveletFilter::from_index(6),
            Some(WaveletFilter::Daubechies9_7)
        );
        assert_eq!(WaveletFilter::from_index(7), None);
    }

    #[test]
    fn filter_shifts() {
        assert_eq!(WaveletFilter::Haar0.filter_shift(), 0);
        assert_eq!(WaveletFilter::Haar1.filter_shift(), 1);
        assert_eq!(WaveletFilter::LeGall5_3.filter_shift(), 1);
        assert_eq!(WaveletFilter::Fidelity.filter_shift(), 0);
    }

    /// Sanity check: encoding `one_d_analysis` followed by
    /// `one_d_synthesis` reproduces the original sequence exactly. The
    /// spec's lifting steps are integer-reversible by design.
    #[test]
    fn one_d_legall_analysis_synthesis_roundtrip() {
        let original: Vec<i32> = vec![10, -7, 42, 31, -5, 100, 128, -200, 17, 19, 22, 0];
        let mut work = original.clone();
        one_d_analysis(&mut work, WaveletFilter::LeGall5_3);
        one_d_synthesis(&mut work, WaveletFilter::LeGall5_3);
        assert_eq!(work, original);
    }

    #[test]
    fn one_d_dd97_analysis_synthesis_roundtrip() {
        let original: Vec<i32> = vec![10, -7, 42, 31, -5, 100, 128, -200, 17, 19, 22, 0];
        let mut work = original.clone();
        one_d_analysis(&mut work, WaveletFilter::DeslauriersDubuc9_7);
        one_d_synthesis(&mut work, WaveletFilter::DeslauriersDubuc9_7);
        assert_eq!(work, original);
    }

    /// vh_analysis / vh_synth round trip for LeGall. Build a non-trivial
    /// picture, analyse it, then re-synthesise and check byte-for-byte
    /// equality.
    #[test]
    fn vh_legall_analysis_synthesis_roundtrip() {
        let w = 8;
        let h = 6;
        let mut pic = SubbandData::new(w, h);
        for y in 0..h {
            for x in 0..w {
                // A non-trivial pattern: diagonal gradient + sparse spikes.
                let mut v = (x as i32 * 3 - y as i32 * 5) * 4;
                if (x + y) % 7 == 0 {
                    v += 200;
                }
                pic.set(y, x, v);
            }
        }
        let (ll, hl, lh, hh) = vh_analysis(&pic, WaveletFilter::LeGall5_3);
        let back = vh_synth(&ll, &hl, &lh, &hh, WaveletFilter::LeGall5_3);
        assert_eq!(back.width, w);
        assert_eq!(back.height, h);
        for y in 0..h {
            for x in 0..w {
                assert_eq!(back.get(y, x), pic.get(y, x), "({x},{y})");
            }
        }
    }

    #[test]
    fn dwt_idwt_roundtrip_legall_depth3() {
        // 32x32 picture, depth 3 => subband sizes 4,4,8,16 per level.
        let w = 32;
        let h = 32;
        let mut pic = SubbandData::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = ((x as i32) ^ (y as i32 * 3)) * 2 - 32;
                pic.set(y, x, v);
            }
        }
        let pyramid = dwt(&pic, WaveletFilter::LeGall5_3, 3);
        let back = idwt(&pyramid, WaveletFilter::LeGall5_3);
        for y in 0..h {
            for x in 0..w {
                assert_eq!(
                    back.get(y, x),
                    pic.get(y, x),
                    "dwt/idwt roundtrip failed at ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn dwt_idwt_roundtrip_dd97_depth2() {
        let w = 16;
        let h = 16;
        let mut pic = SubbandData::new(w, h);
        for y in 0..h {
            for x in 0..w {
                pic.set(y, x, (x as i32 * 7 - y as i32 * 3) % 101);
            }
        }
        let pyramid = dwt(&pic, WaveletFilter::DeslauriersDubuc9_7, 2);
        let back = idwt(&pyramid, WaveletFilter::DeslauriersDubuc9_7);
        for y in 0..h {
            for x in 0..w {
                assert_eq!(back.get(y, x), pic.get(y, x));
            }
        }
    }

    #[test]
    fn dwt_idwt_roundtrip_haar0_depth2() {
        let w = 8;
        let h = 8;
        let mut pic = SubbandData::new(w, h);
        for y in 0..h {
            for x in 0..w {
                pic.set(y, x, (x + y * w) as i32);
            }
        }
        let pyramid = dwt(&pic, WaveletFilter::Haar0, 2);
        let back = idwt(&pyramid, WaveletFilter::Haar0);
        for y in 0..h {
            for x in 0..w {
                assert_eq!(back.get(y, x), pic.get(y, x));
            }
        }
    }

    /// Exhaustive `dwt` / `idwt` round-trip across **all seven** spec
    /// wavelet filters (Tables 15.1-15.7) for `dwt_depth` 1..=3 on a
    /// non-trivial 32x32 picture. Every filter is integer-reversible
    /// by construction (Annex G lifting), so the reconstructed picture
    /// must be bit-exact regardless of which filter was used.
    ///
    /// Catches regressions in filter step ordering, tap signs, or
    /// `filter_shift()` shift counts that bit-exactly cancel for
    /// LeGall / DD9_7 (the only filters previously covered) but break
    /// the heavier filters.
    #[test]
    fn dwt_idwt_roundtrip_all_filters_all_depths() {
        let w = 32;
        let h = 32;
        let mut pic = SubbandData::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let mut v = (x as i32 * 3 - y as i32 * 5) * 2;
                if (x + 2 * y) % 11 == 0 {
                    v += 64;
                }
                pic.set(y, x, v);
            }
        }
        for filter in [
            WaveletFilter::DeslauriersDubuc9_7,
            WaveletFilter::LeGall5_3,
            WaveletFilter::DeslauriersDubuc13_7,
            WaveletFilter::Haar0,
            WaveletFilter::Haar1,
            WaveletFilter::Fidelity,
            WaveletFilter::Daubechies9_7,
        ] {
            for depth in 1..=3u32 {
                let pyramid = dwt(&pic, filter, depth);
                let back = idwt(&pyramid, filter);
                for y in 0..h {
                    for x in 0..w {
                        assert_eq!(
                            back.get(y, x),
                            pic.get(y, x),
                            "filter {filter:?} depth {depth} mismatch at ({x},{y})"
                        );
                    }
                }
            }
        }
    }

    /// Round-trip on a non-square picture (40x24) at depth 3 — exercises
    /// the asymmetric column / row handling in `vh_synth` / `vh_analysis`
    /// that a square test cannot. 40 and 24 are both multiples of 8 =
    /// 2^depth, the spec's §15.7 alignment requirement.
    #[test]
    fn dwt_idwt_roundtrip_non_square_legall_depth3() {
        let w = 40;
        let h = 24;
        let mut pic = SubbandData::new(w, h);
        for y in 0..h {
            for x in 0..w {
                pic.set(y, x, (x as i32 * 7 + y as i32 * 11) % 257 - 128);
            }
        }
        let pyramid = dwt(&pic, WaveletFilter::LeGall5_3, 3);
        let back = idwt(&pyramid, WaveletFilter::LeGall5_3);
        assert_eq!(back.width, w);
        assert_eq!(back.height, h);
        for y in 0..h {
            for x in 0..w {
                assert_eq!(back.get(y, x), pic.get(y, x), "({x},{y})");
            }
        }
    }

    // -----------------------------------------------------------------
    //   §15.4.2 h_synthesis tests — horizontal-only IDWT step.
    // -----------------------------------------------------------------

    /// §15.4.2 sanity check: feeding `h_synth` a pure-DC L band (with
    /// the accuracy-bit pre-shift) and a zero H band reproduces a
    /// uniform output at the original DC value, exactly like the
    /// symmetric DC test above. LeGall (5,3) `filter_shift() == 1`, so
    /// the L coefficient is pre-shifted by 1.
    #[test]
    fn h_synth_from_dc_reconstructs_uniform_legall() {
        let w = 6;
        let h = 4;
        let mut l = SubbandData::new(w, h);
        for y in 0..h {
            for x in 0..w {
                // Pre-shift the DC up by filter_shift = 1 so the step-4
                // rounding right-shift restores the original value.
                l.set(y, x, 23 << 1);
            }
        }
        let zeros = SubbandData::new(w, h);
        let out = h_synth(&l, &zeros, WaveletFilter::LeGall5_3);
        assert_eq!(out.width, 2 * w);
        assert_eq!(out.height, h);
        for y in 0..out.height {
            for x in 0..out.width {
                assert_eq!(out.get(y, x), 23, "({x},{y}) got {}", out.get(y, x));
            }
        }
    }

    /// `h_synth` only touches rows: a row-pattern in L should land in
    /// the even columns of the output regardless of `y`, and the H band
    /// should not bleed into columns of a different row.
    #[test]
    fn h_synth_keeps_rows_independent_haar0() {
        // Haar0 has filter_shift == 0 and a trivial lifting kernel — the
        // post-synthesis even/odd interleaving recovery is the simplest
        // to reason about, so a row-distinguishing pattern flows through
        // cleanly.
        let w = 4;
        let h = 3;
        let mut l = SubbandData::new(w, h);
        let mut hh = SubbandData::new(w, h);
        // Distinct row constants so any vertical leak is visible.
        for y in 0..h {
            for x in 0..w {
                l.set(y, x, 100 + y as i32);
                hh.set(y, x, 7 + y as i32 * 11);
            }
        }
        let out = h_synth(&l, &hh, WaveletFilter::Haar0);
        // Haar0 step 1 (Type 2, s=1, taps=[1], D=1):
        //   A[2n] -= (A[2n+1] + 1) >> 1
        // Haar0 step 2 (Type 3, s=0, taps=[1], D=0):
        //   A[2n+1] += A[2n]
        // filter_shift = 0, so no step-4 right-shift.
        // For row y, interleaved input is [L_y, H_y, L_y, H_y, ...]
        // After step 1: even[n] = L_y - ((H_y + 1) >> 1)
        // After step 2: odd[n]  = H_y + (L_y - ((H_y + 1) >> 1))
        for y in 0..h {
            let lv = 100 + y as i32;
            let hv = 7 + y as i32 * 11;
            let even_expected = lv - ((hv + 1) >> 1);
            let odd_expected = hv + even_expected;
            for x in 0..2 * w {
                let expected = if x % 2 == 0 {
                    even_expected
                } else {
                    odd_expected
                };
                assert_eq!(
                    out.get(y, x),
                    expected,
                    "row {y} col {x}: got {} want {}",
                    out.get(y, x),
                    expected
                );
            }
        }
    }

    /// `h_analysis` ↔ `h_synth` integer round-trip across all seven
    /// spec filters and a non-square shape. The §15.4.2 process is
    /// integer-reversible by construction (Annex F lifting); a single
    /// horizontal stage must reproduce the input exactly.
    #[test]
    fn h_synth_h_analysis_roundtrip_all_filters() {
        // 12-wide is a multiple of 2 so de-interleave is exact; 5 rows
        // so a square-symmetric bug would be visible.
        let w = 12;
        let h = 5;
        let mut pic = SubbandData::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let mut v = x as i32 * 9 - y as i32 * 13;
                if (x + 3 * y) % 4 == 0 {
                    v += 50;
                }
                pic.set(y, x, v);
            }
        }
        for filter in [
            WaveletFilter::DeslauriersDubuc9_7,
            WaveletFilter::LeGall5_3,
            WaveletFilter::DeslauriersDubuc13_7,
            WaveletFilter::Haar0,
            WaveletFilter::Haar1,
            WaveletFilter::Fidelity,
            WaveletFilter::Daubechies9_7,
        ] {
            let (l, hh) = h_analysis(&pic, filter);
            assert_eq!(l.width, w / 2);
            assert_eq!(l.height, h);
            assert_eq!(hh.width, w / 2);
            assert_eq!(hh.height, h);
            let back = h_synth(&l, &hh, filter);
            assert_eq!(back.width, w);
            assert_eq!(back.height, h);
            for y in 0..h {
                for x in 0..w {
                    assert_eq!(
                        back.get(y, x),
                        pic.get(y, x),
                        "filter {filter:?} mismatch at ({x},{y})"
                    );
                }
            }
        }
    }

    /// Output dimensions: §15.4.2 mandates `width(synth) == 2 *
    /// width(L_data)` and `height(synth) == height(L_data)`. Pin this
    /// invariant explicitly so a future refactor that drops the
    /// vertical-untouched property fails loudly.
    #[test]
    fn h_synth_doubles_width_keeps_height() {
        let l = SubbandData::new(7, 3);
        let h = SubbandData::new(7, 3);
        let out = h_synth(&l, &h, WaveletFilter::LeGall5_3);
        assert_eq!(out.width, 14);
        assert_eq!(out.height, 3);
    }

    // -----------------------------------------------------------------
    //   §15.4.1 idwt_with_ho — asymmetric / horizontal-only driver.
    // -----------------------------------------------------------------

    /// With `dwt_depth_ho == 0` the §15.4.1 driver collapses to the
    /// pre-v3 [`idwt`]. Build a symmetric pyramid via [`dwt`], run it
    /// through both drivers, and confirm byte-identical pictures —
    /// this pins the §12.4.4 NOTE invariant ("the inverse wavelet
    /// transform process is identical to that defined in earlier
    /// versions of this specification" when the asymmetric block is
    /// at its symmetric default) at the wavelet layer.
    #[test]
    fn idwt_with_ho_equals_idwt_when_ho_depth_zero() {
        let w = 16;
        let h = 16;
        let mut pic = SubbandData::new(w, h);
        for y in 0..h {
            for x in 0..w {
                pic.set(y, x, (x as i32 * 5 - y as i32 * 3 + 7) & 0x7f);
            }
        }
        for filter in [
            WaveletFilter::LeGall5_3,
            WaveletFilter::Haar1,
            WaveletFilter::DeslauriersDubuc9_7,
        ] {
            for depth in 1..=3u32 {
                let pyramid = dwt(&pic, filter, depth);
                let baseline = idwt(&pyramid, filter);
                // filter_ho is unused at dwt_depth_ho = 0 — pick a
                // deliberately different filter to prove that.
                let ho_drive = idwt_with_ho(&pyramid, filter, WaveletFilter::Fidelity, 0);
                assert_eq!(baseline.width, ho_drive.width);
                assert_eq!(baseline.height, ho_drive.height);
                for y in 0..h {
                    for x in 0..w {
                        assert_eq!(
                            baseline.get(y, x),
                            ho_drive.get(y, x),
                            "filter {filter:?} depth {depth} at ({x},{y})"
                        );
                    }
                }
            }
        }
    }

    /// Pure horizontal-only path: `dwt_depth = 0`, `dwt_depth_ho > 0`.
    /// Build a 16x4 picture, peel off two horizontal-only levels with
    /// [`dwt_with_ho`], and confirm [`idwt_with_ho`] reconstructs it
    /// bit-exactly. With no symmetric 2-D levels the second §15.4.1
    /// loop runs zero iterations, so the test isolates the
    /// `h_synth`-chain step.
    #[test]
    fn idwt_with_ho_roundtrip_pure_horizontal() {
        let w = 16;
        let h = 4;
        let mut pic = SubbandData::new(w, h);
        for y in 0..h {
            for x in 0..w {
                pic.set(y, x, x as i32 * 11 + y as i32 * 23 - 50);
            }
        }
        for filter_ho in [
            WaveletFilter::LeGall5_3,
            WaveletFilter::Haar0,
            WaveletFilter::DeslauriersDubuc13_7,
        ] {
            let pyramid = dwt_with_ho(
                &pic,
                WaveletFilter::Haar0,
                filter_ho,
                /* dwt_depth */ 0,
                /* dwt_depth_ho */ 2,
            );
            // Level-0 L should be 4 wide (16 / 2^2), height unchanged
            // at 4 — confirms the §13.2.3 horizontal-only band
            // dimensions before driving the synthesis.
            assert_eq!(pyramid[0][0].width, 4);
            assert_eq!(pyramid[0][0].height, 4);
            assert_eq!(pyramid[1][3].width, 4);
            assert_eq!(pyramid[2][3].width, 8);
            let back = idwt_with_ho(&pyramid, WaveletFilter::Haar0, filter_ho, 2);
            assert_eq!(back.width, w);
            assert_eq!(back.height, h);
            for y in 0..h {
                for x in 0..w {
                    assert_eq!(
                        back.get(y, x),
                        pic.get(y, x),
                        "filter_ho {filter_ho:?} at ({x},{y})"
                    );
                }
            }
        }
    }

    /// Combined path: `dwt_depth_ho > 0` AND `dwt_depth > 0`, with
    /// `wavelet_index_ho` deliberately different from `wavelet_index`.
    /// Round-trips through `dwt_with_ho` / `idwt_with_ho` on a non-
    /// square 32x8 picture across a representative selection of filter
    /// pairs and depth pairs. The picture width must be divisible by
    /// `2^(dwt_depth + dwt_depth_ho)` and the height by `2^dwt_depth`
    /// (§13.2.3) — 32 = 2^5 and 8 = 2^3 cover every (depth, ho) up to
    /// (3, 2).
    #[test]
    fn idwt_with_ho_roundtrip_mixed_asymmetric() {
        let w = 32;
        let h = 8;
        let mut pic = SubbandData::new(w, h);
        for y in 0..h {
            for x in 0..w {
                pic.set(y, x, (x as i32 * 7 + y as i32 * 13) % 251 - 100);
            }
        }
        // (filter_v, filter_ho, dwt_depth, dwt_depth_ho) tuples — pick
        // pairs that exercise asym_transform_index_flag = True (v != ho)
        // and asym_transform_flag = True (ho > 0) together, i.e. both
        // §12.4.4.2 and §12.4.4.3 simultaneously.
        let cases: &[(WaveletFilter, WaveletFilter, u32, u32)] = &[
            (WaveletFilter::LeGall5_3, WaveletFilter::Haar1, 2, 1),
            (
                WaveletFilter::DeslauriersDubuc9_7,
                WaveletFilter::LeGall5_3,
                2,
                2,
            ),
            (WaveletFilter::Haar1, WaveletFilter::Haar0, 1, 2),
            (
                WaveletFilter::Fidelity,
                WaveletFilter::DeslauriersDubuc13_7,
                1,
                1,
            ),
            (WaveletFilter::Daubechies9_7, WaveletFilter::LeGall5_3, 3, 2),
        ];
        for &(fv, fho, dwt_depth, dwt_depth_ho) in cases {
            let pyramid = dwt_with_ho(&pic, fv, fho, dwt_depth, dwt_depth_ho);
            let back = idwt_with_ho(&pyramid, fv, fho, dwt_depth_ho);
            assert_eq!(back.width, w);
            assert_eq!(back.height, h);
            for y in 0..h {
                for x in 0..w {
                    assert_eq!(
                        back.get(y, x),
                        pic.get(y, x),
                        "fv {fv:?} fho {fho:?} depth {dwt_depth} ho {dwt_depth_ho} at ({x},{y})"
                    );
                }
            }
        }
    }

    /// The §15.4.1 driver returns a picture whose width is twice the
    /// level-0 L band's width per `h_synth` step, then twice that per
    /// `vh_synth` step. Pin the algebra: starting from an L band of
    /// width `lw`, after `ho` horizontal-only stages + `d` symmetric
    /// stages, the output width is `lw << (ho + d)`. Height grows by
    /// `<< d` only.
    #[test]
    fn idwt_with_ho_output_dimensions_scale_correctly() {
        // L band 3x5; ho=2, d=2 → width = 3 << 4 = 48, height = 5 << 2
        // = 20.
        let lw = 3;
        let lh = 5;
        let ho: u32 = 2;
        let d: u32 = 2;
        let total = (ho + d) as usize;
        let mut pyramid: Vec<[SubbandData; 4]> = Vec::with_capacity(total + 1);
        pyramid.push([
            SubbandData::new(lw, lh),
            SubbandData::new(0, 0),
            SubbandData::new(0, 0),
            SubbandData::new(0, 0),
        ]);
        // Horizontal-only levels: H slot doubles width each step,
        // height stays at lh.
        let mut cur_w = lw;
        for _ in 1..=ho as usize {
            pyramid.push([
                SubbandData::new(0, 0),
                SubbandData::new(0, 0),
                SubbandData::new(0, 0),
                SubbandData::new(cur_w, lh),
            ]);
            cur_w *= 2;
        }
        // Symmetric levels: HL/LH/HH all share the current shape,
        // doubling both axes per step.
        let mut cur_h = lh;
        for _ in 1..=d as usize {
            pyramid.push([
                SubbandData::new(0, 0),
                SubbandData::new(cur_w, cur_h),
                SubbandData::new(cur_w, cur_h),
                SubbandData::new(cur_w, cur_h),
            ]);
            cur_w *= 2;
            cur_h *= 2;
        }
        let out = idwt_with_ho(
            &pyramid,
            WaveletFilter::LeGall5_3,
            WaveletFilter::LeGall5_3,
            ho,
        );
        assert_eq!(out.width, lw << (ho + d));
        assert_eq!(out.height, lh << d);
    }
}
