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
    // Step 2: interleave.
    for y in 0..h {
        for x in 0..w {
            synth.set(2 * y, 2 * x, ll.get(y, x));
            synth.set(2 * y, 2 * x + 1, hl.get(y, x));
            synth.set(2 * y + 1, 2 * x, lh.get(y, x));
            synth.set(2 * y + 1, 2 * x + 1, hh.get(y, x));
        }
    }
    // Step 3a: vertical synthesis — one lifting pass per column.
    let mut col_buf = vec![0i32; out_h];
    for x in 0..out_w {
        for y in 0..out_h {
            col_buf[y] = synth.get(y, x);
        }
        one_d_synthesis(&mut col_buf, filter);
        for y in 0..out_h {
            synth.set(y, x, col_buf[y]);
        }
    }
    // Step 3b: horizontal synthesis — one lifting pass per row.
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
    // Step 2: horizontal analysis.
    for y in 0..out_h {
        let row = work.row_mut(y);
        one_d_analysis(row, filter);
    }
    // Step 3: vertical analysis.
    let mut col_buf = vec![0i32; out_h];
    for x in 0..out_w {
        for y in 0..out_h {
            col_buf[y] = work.get(y, x);
        }
        one_d_analysis(&mut col_buf, filter);
        for y in 0..out_h {
            work.set(y, x, col_buf[y]);
        }
    }
    // Step 4: de-interleave.
    let half_w = out_w / 2;
    let half_h = out_h / 2;
    let mut ll = SubbandData::new(half_w, half_h);
    let mut hl = SubbandData::new(half_w, half_h);
    let mut lh = SubbandData::new(half_w, half_h);
    let mut hh = SubbandData::new(half_w, half_h);
    for y in 0..half_h {
        for x in 0..half_w {
            ll.set(y, x, work.get(2 * y, 2 * x));
            hl.set(y, x, work.get(2 * y, 2 * x + 1));
            lh.set(y, x, work.get(2 * y + 1, 2 * x));
            hh.set(y, x, work.get(2 * y + 1, 2 * x + 1));
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
pub fn dwt(
    picture: &SubbandData,
    filter: WaveletFilter,
    dwt_depth: u32,
) -> Vec<[SubbandData; 4]> {
    let mut pyramid: Vec<[SubbandData; 4]> =
        Vec::with_capacity(dwt_depth as usize + 1);
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
        let out = vh_synth(
            &ll,
            &zeros,
            &zeros,
            &zeros,
            WaveletFilter::LeGall5_3,
        );
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
                assert_eq!(
                    out.get(y, x),
                    17,
                    "({x},{y}) got {}",
                    out.get(y, x)
                );
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
        assert_eq!(WaveletFilter::from_index(6), Some(WaveletFilter::Daubechies9_7));
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
        let original: Vec<i32> =
            vec![10, -7, 42, 31, -5, 100, 128, -200, 17, 19, 22, 0];
        let mut work = original.clone();
        one_d_analysis(&mut work, WaveletFilter::LeGall5_3);
        one_d_synthesis(&mut work, WaveletFilter::LeGall5_3);
        assert_eq!(work, original);
    }

    #[test]
    fn one_d_dd97_analysis_synthesis_roundtrip() {
        let original: Vec<i32> =
            vec![10, -7, 42, 31, -5, 100, 128, -200, 17, 19, 22, 0];
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
        let (ll, hl, lh, hh) =
            vh_analysis(&pic, WaveletFilter::LeGall5_3);
        let back = vh_synth(
            &ll,
            &hl,
            &lh,
            &hh,
            WaveletFilter::LeGall5_3,
        );
        assert_eq!(back.width, w);
        assert_eq!(back.height, h);
        for y in 0..h {
            for x in 0..w {
                assert_eq!(
                    back.get(y, x),
                    pic.get(y, x),
                    "({x},{y})"
                );
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
        let pyramid =
            dwt(&pic, WaveletFilter::DeslauriersDubuc9_7, 2);
        let back =
            idwt(&pyramid, WaveletFilter::DeslauriersDubuc9_7);
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
}
