//! Inverse quantisation (§13.2) and default quantisation matrices
//! (Annex E.1).
//!
//! Dirac quantises wavelet coefficients with a "dead-zone" scheme:
//!
//! ```text
//!   qcoeff = sign(x) * (|x| / qf)
//! ```
//!
//! and then adds a reconstruction offset on decode:
//!
//! ```text
//!   x  = sign(qcoeff) * ((|qcoeff| * qf + offset + 2) / 4)
//! ```
//!
//! The `qf` / `offset` pair depends on `q % 4` (§13.2.1) and, for the
//! offset, on whether the picture is intra or inter.
//!
//! Low-delay slices apply a per-subband quantisation matrix on top of
//! the slice's global quantisation index (§13.5.4 `slice_quantisers`).
//! Annex E.1 gives the default matrix for each combination of wavelet
//! filter and transform depth up to depth 4.

use crate::subband::Orient;
use crate::wavelet::WaveletFilter;

/// `quant_factor(q)` (§13.2.1). Valid for any `q ≥ 0`.
///
/// The §13.5.4 per-slice adaptive search drives `q` up toward the 7-bit
/// maximum (127) on busy slices, where `2^(q/4)` exceeds 32 bits; the
/// arithmetic is therefore done in `u64` and the result saturated at
/// `u32::MAX`. Saturation is behaviour-preserving: a `qf` that large
/// forward-quantises every 8-bit-derived coefficient to 0
/// (`4*|x|/qf == 0`), and [`inverse_quant`] reconstructs 0 from a 0
/// qcoeff regardless of `qf`, so neither encode nor decode observes the
/// clamp. Without it, `q >= 124` (or lower for the `q%4 != 0` branches)
/// would overflow.
pub fn quant_factor(q: u32) -> u32 {
    let base: u64 = 1u64 << (q / 4);
    let raw = match q % 4 {
        0 => 4 * base,
        1 => (503_829 * base + 52_958) / 105_917,
        2 => (665_857 * base + 58_854) / 117_708,
        3 => (440_253 * base + 32_722) / 65_444,
        _ => unreachable!(),
    };
    raw.min(u32::MAX as u64) as u32
}

/// `quant_offset(q)` for intra pictures, per the 2008 Dirac spec
/// §13.2.1. The spec also enforces the `0/1` special cases so that the
/// reconstruction-interval invariant `3 ≤ offset + 2 < quant_factor`
/// holds at the very low indices where the formula would otherwise
/// underflow. SMPTE ST 2042-1 (VC-2) collapses intra and inter into the
/// single intra formula — and VC-2 itself is intra-only by construction
/// — so this function is the correct offset for every VC-2 LD / HQ
/// slice as well as for every core-syntax intra picture.
pub fn quant_offset(q: u32) -> u32 {
    quant_offset_for(q, /* is_intra */ true)
}

/// `quant_offset(q)` per §13.2.1 with the intra / inter split.
///
/// The 2008 Dirac specification defines two reconstruction offsets:
///
/// * **Intra** — approximately `qf / 2`. The reconstruction point sits
///   in the middle of the dead-zone interval, giving an unbiased
///   estimator for the original magnitude.
/// * **Inter** — approximately `3 * qf / 8`. The reconstruction point
///   sits closer to zero because the inter residue distribution is
///   strongly Laplacian-shaped: most coefficients quantise to zero and
///   the non-zero ones are typically near the dead-zone boundary, so a
///   smaller offset reduces the average reconstruction error.
///
/// Both branches share the `q == 0` special case (`offset = 1`); the
/// intra branch additionally special-cases `q == 1` (`offset = 2`). The
/// inter branch has no `q == 1` carve-out because the
/// `(qf * 3 + 4) / 8` formula already evaluates to `2` at `q == 1`
/// (`qf == 5` → `(5*3 + 4) / 8 == 2`), which satisfies the same
/// reconstruction-interval invariant.
///
/// At `q == 0` intra and inter agree (`offset = 1`); the formulas only
/// diverge for `q >= 1`. The encoder's forward dead-zone step (`x =
/// sign(x) * (|x| / qf)`) is independent of the offset, so a change in
/// `is_intra` only affects the decoder side.
pub fn quant_offset_for(q: u32, is_intra: bool) -> u32 {
    if q == 0 {
        return 1;
    }
    if is_intra {
        if q == 1 {
            2
        } else {
            quant_factor(q).div_ceil(2)
        }
    } else {
        // 2008 spec §13.2.1: offset = (quant_factor(q) * 3 + 4) // 8.
        // Cast to u64 so the `* 3` step doesn't overflow when `qf` is
        // near `u32::MAX` (the §13.5.4 per-slice adaptive search drives
        // `q` up to 127); the result fits back into u32 because the
        // intra branch's `(qf + 1) / 2` is a larger upper bound and
        // already u32-valued.
        let qf = quant_factor(q) as u64;
        ((qf * 3 + 4) / 8) as u32
    }
}

/// `inverse_quant(qcoeff, q)` for intra pictures (§13.2 / §13.3.1).
///
/// Kept as a thin shim over [`inverse_quant_for`] with `is_intra = true`
/// so existing callers (LD slices, intra-only paths, internal encoder
/// roundtrip asserts) need no change. Use [`inverse_quant_for`]
/// explicitly when the caller knows whether the picture is intra or
/// inter.
pub fn inverse_quant(qcoeff: i32, q: u32) -> i32 {
    inverse_quant_for(qcoeff, q, /* is_intra */ true)
}

/// `inverse_quant(qcoeff, q)` (§13.2 / §13.3.1) parameterised by the
/// picture's intra / inter flag.
///
/// Pseudocode (§13.2):
///
/// ```text
///   if magnitude != 0:
///       magnitude *= quant_factor(q)
///       magnitude += quant_offset(q)   # intra or inter per §13.2.1
///       magnitude += 2
///       magnitude  = magnitude // 4
///   return sign(qcoeff) * magnitude
/// ```
pub fn inverse_quant_for(qcoeff: i32, q: u32, is_intra: bool) -> i32 {
    if qcoeff == 0 {
        return 0;
    }
    let mag = qcoeff.unsigned_abs() as u64;
    let qf = quant_factor(q) as u64;
    let off = quant_offset_for(q, is_intra) as u64;
    let mag = (mag * qf + off + 2) / 4;
    if qcoeff < 0 {
        -(mag as i32)
    } else {
        mag as i32
    }
}

/// The per-subband quantisation matrix used for low-delay slice
/// decoding (§13.5.4). Indexed by `(level, orient)`; level 0 carries
/// only LL, all other levels carry HL/LH/HH.
#[derive(Debug, Clone)]
pub struct QuantMatrix {
    pub dwt_depth: u32,
    /// `levels[level][orient_index]`. `orient_index` matches
    /// [`Orient::as_index`], so LL=0, HL=1, LH=2, HH=3.
    pub levels: Vec<[u32; 4]>,
}

impl QuantMatrix {
    /// Look up the default quantisation matrix for a given wavelet
    /// filter / depth combination (Annex E.1, Tables E.1-E.7). Returns
    /// `None` if the depth is >4 (§11.3.5 requires a custom matrix in
    /// that case).
    pub fn default_for(filter: WaveletFilter, dwt_depth: u32) -> Option<Self> {
        if dwt_depth > 4 {
            return None;
        }
        // Each entry in the tables below is [LL, HL, LH, HH] per level.
        // When the spec shows a dash we use 0 — such entries are never
        // read (level 0 uses LL only; levels >= 1 skip LL).
        //
        // The tables list columns by `dwt_depth`; we pick the matching
        // column for each level 0..=depth.
        let (ll_row, hl_row): (&[u32], &[(u32, u32, u32)]) = match filter {
            WaveletFilter::DeslauriersDubuc9_7 => (
                // LL(level=0) by depth 0..=4:
                &[0, 5, 5, 5, 5],
                // HL,LH,HH per level 1..=4 (each column = depth):
                // For a given depth D, the triplet at row `level` is:
                //   level=1 → (3,3,0) if D>=1
                //   level=2 → (4,4,1) if D>=2
                //   level=3 → (5,5,2) if D>=3
                //   level=4 → (6,6,3) if D==4
                &[(3, 3, 0), (4, 4, 1), (5, 5, 2), (6, 6, 3)],
            ),
            WaveletFilter::LeGall5_3 => (
                &[0, 4, 4, 4, 4],
                &[(2, 2, 0), (4, 4, 2), (5, 5, 3), (7, 7, 5)],
            ),
            WaveletFilter::DeslauriersDubuc13_7 => (
                &[0, 5, 5, 5, 5],
                &[(3, 3, 0), (4, 4, 1), (5, 5, 2), (6, 6, 3)],
            ),
            WaveletFilter::Haar0 => (
                // depth: 0,1,2,3,4 → LL = 0, 8, 12, 16, 20
                &[0, 8, 12, 16, 20],
                // For depth D, the triplet at each "level" l (1..=D) is
                // complicated by the pattern (see Table E.4): level 1 at
                // depth D has value (4 + 4 * (D-1),  4 + 4 * (D-1), max(0, 4 * (D-1)))
                // level 2 at depth D: (4 + 4 * (D-2), …)
                // Effectively the HL/LH pair is 4 * (D + 1 - l) and HH is
                // 4 * (D - l) but clamped to >=0.
                // We encode this depth-sensitive pattern in a closure below.
                &[],
            ),
            WaveletFilter::Haar1 => (
                &[0, 8, 8, 8, 8],
                &[(4, 4, 0), (4, 4, 0), (4, 4, 0), (4, 4, 0)],
            ),
            WaveletFilter::Fidelity => (
                &[0, 0, 0, 0, 0],
                &[(4, 4, 8), (8, 8, 12), (13, 13, 17), (17, 17, 21)],
            ),
            WaveletFilter::Daubechies9_7 => (
                &[0, 3, 3, 3, 3],
                &[(1, 1, 0), (4, 4, 2), (6, 6, 5), (9, 9, 7)],
            ),
        };

        let mut levels: Vec<[u32; 4]> = Vec::with_capacity(dwt_depth as usize + 1);
        let ll = ll_row[dwt_depth as usize];
        levels.push([ll, 0, 0, 0]);
        match filter {
            WaveletFilter::Haar0 => {
                // Table E.4 is depth-dependent; compute each level from
                // closed form values. For depth D and level l (l>=1):
                //   HL=LH = max(0, 4 * (D + 1 - l))
                //   HH    = max(0, 4 * (D - l))
                for l in 1..=dwt_depth {
                    let dp1 = dwt_depth as i32 - l as i32 + 1;
                    let d = dwt_depth as i32 - l as i32;
                    let hl = (4 * dp1).max(0) as u32;
                    let hh = (4 * d).max(0) as u32;
                    levels.push([0, hl, hl, hh]);
                }
            }
            _ => {
                for (i, &(hl, lh, hh)) in hl_row.iter().take(dwt_depth as usize).enumerate() {
                    let _level = i + 1;
                    levels.push([0, hl, lh, hh]);
                }
            }
        }
        Some(Self { dwt_depth, levels })
    }

    /// Safe getter. Returns 0 when `(level, orient)` is out of range.
    pub fn get(&self, level: u32, orient: Orient) -> u32 {
        let l = level as usize;
        if l >= self.levels.len() {
            return 0;
        }
        self.levels[l][orient.as_index()]
    }
}

/// `slice_quantisers` (§13.5.4): subtract each subband's quantisation
/// matrix entry from `qindex`, clamping to 0. Returns the per-level
/// `[LL, HL, LH, HH]` effective quantisers.
pub fn slice_quantisers(qindex: u32, qmatrix: &QuantMatrix) -> Vec<[u32; 4]> {
    let mut out: Vec<[u32; 4]> = Vec::with_capacity(qmatrix.levels.len());
    for (level, triplet) in qmatrix.levels.iter().enumerate() {
        let q_ll = if level == 0 {
            qindex.saturating_sub(triplet[0])
        } else {
            0
        };
        let q_hl = qindex.saturating_sub(triplet[1]);
        let q_lh = qindex.saturating_sub(triplet[2]);
        let q_hh = qindex.saturating_sub(triplet[3]);
        out.push([q_ll, q_hl, q_lh, q_hh]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_factor_spot_checks() {
        // q=0 → 4
        assert_eq!(quant_factor(0), 4);
        // q=4 → 8 (2^(4/4)=2, q%4=0 so 4 * 2 = 8)
        assert_eq!(quant_factor(4), 8);
        // q=8 → 16
        assert_eq!(quant_factor(8), 16);
    }

    #[test]
    fn inverse_quant_identity_on_zero() {
        assert_eq!(inverse_quant(0, 0), 0);
        assert_eq!(inverse_quant(0, 5), 0);
    }

    #[test]
    fn inverse_quant_trivial_q0() {
        // With q=0, qf=4, off=1: mag = (|x|*4 + 1 + 2)/4 = |x| (rounded).
        assert_eq!(inverse_quant(5, 0), 5);
        assert_eq!(inverse_quant(-3, 0), -3);
    }

    #[test]
    fn default_quant_matrix_legall_depth3() {
        // Table E.2, column "3":
        //   LL0 = 4
        //   HL/LH/HH at levels 1..=3: (2,2,0), (4,4,2), (5,5,3)
        let qm = QuantMatrix::default_for(WaveletFilter::LeGall5_3, 3).unwrap();
        assert_eq!(qm.levels.len(), 4);
        assert_eq!(qm.levels[0], [4, 0, 0, 0]);
        assert_eq!(qm.levels[1], [0, 2, 2, 0]);
        assert_eq!(qm.levels[2], [0, 4, 4, 2]);
        assert_eq!(qm.levels[3], [0, 5, 5, 3]);
    }

    #[test]
    fn default_quant_matrix_haar_no_shift_depth4() {
        let qm = QuantMatrix::default_for(WaveletFilter::Haar0, 4).unwrap();
        // Table E.4 column "4": LL=20, then triplets:
        //   level 1: 16,16,12
        //   level 2: 12,12,8
        //   level 3: 8,8,4
        //   level 4: 4,4,0
        assert_eq!(qm.levels[0], [20, 0, 0, 0]);
        assert_eq!(qm.levels[1], [0, 16, 16, 12]);
        assert_eq!(qm.levels[2], [0, 12, 12, 8]);
        assert_eq!(qm.levels[3], [0, 8, 8, 4]);
        assert_eq!(qm.levels[4], [0, 4, 4, 0]);
    }

    #[test]
    fn slice_quantisers_clip_to_zero() {
        let qm = QuantMatrix::default_for(WaveletFilter::LeGall5_3, 2).unwrap();
        let q = slice_quantisers(3, &qm);
        // qindex=3, LL0=4 → clipped to 0.
        assert_eq!(q[0][0], 0);
        // Level 1 HL matrix entry = 2 → 3 - 2 = 1.
        assert_eq!(q[1][1], 1);
        assert_eq!(q[1][2], 1);
        // Level 1 HH matrix entry = 0 → 3.
        assert_eq!(q[1][3], 3);
    }

    #[test]
    fn dwt_depth_greater_than_four_rejected() {
        assert!(QuantMatrix::default_for(WaveletFilter::LeGall5_3, 5).is_none());
    }

    // ---- §13.2.1 intra vs inter quant_offset ---------------------------

    /// At `q == 0` the spec collapses both branches to `offset = 1`.
    /// Pinned because the rest of the inter test surface relies on the
    /// intra and inter decoders being byte-for-byte identical when the
    /// per-slice quantiser is zero (the default for the encoder's
    /// residue path, and where every encoder-decoder self-roundtrip
    /// test currently exercises the wire).
    #[test]
    fn quant_offset_intra_inter_agree_at_q_zero() {
        assert_eq!(quant_offset_for(0, true), 1);
        assert_eq!(quant_offset_for(0, false), 1);
        // The unparameterised alias is the intra path.
        assert_eq!(quant_offset(0), 1);
    }

    /// `q == 1` is the corner the spec carves out: intra hard-codes
    /// `offset = 2`; the inter formula `(qf * 3 + 4) / 8` already
    /// evaluates to `2` there (`qf == 5` → `(5*3 + 4)/8 == 2`) so no
    /// explicit carve-out is needed for the inter branch.
    #[test]
    fn quant_offset_intra_inter_agree_at_q_one() {
        assert_eq!(quant_offset_for(1, true), 2);
        assert_eq!(quant_offset_for(1, false), 2);
        assert_eq!(quant_offset(1), 2);
    }

    /// Spot-check the §13.2.1 inter formula at `q == 2`, `q == 4`,
    /// `q == 8` against direct evaluation of `(qf * 3 + 4) // 8`. These
    /// are the small-q regime that every realistic inter residue
    /// stream lives in.
    #[test]
    fn quant_offset_inter_formula_spot_checks() {
        // q=2 → qf = (665857*1 + 58854)/117708 = 6; inter = (6*3+4)/8 = 22/8 = 2.
        assert_eq!(quant_factor(2), 6);
        assert_eq!(quant_offset_for(2, false), 2);
        // q=4 → qf = 8; inter = (8*3+4)/8 = 28/8 = 3.
        assert_eq!(quant_factor(4), 8);
        assert_eq!(quant_offset_for(4, false), 3);
        // q=8 → qf = 16; inter = (16*3+4)/8 = 52/8 = 6.
        assert_eq!(quant_factor(8), 16);
        assert_eq!(quant_offset_for(8, false), 6);
        // q=12 → qf = 32; inter = (32*3+4)/8 = 100/8 = 12.
        assert_eq!(quant_factor(12), 32);
        assert_eq!(quant_offset_for(12, false), 12);
    }

    /// At every quant index that's reachable by the 7-bit per-slice
    /// search (0..=127) the intra offset is `>= ` the inter offset
    /// (intra picks the midpoint of the dead-zone; inter biases toward
    /// zero). Equal only at `q ∈ {0, 1}`; strictly greater everywhere
    /// else. This invariant is what lets the existing intra-only test
    /// surface keep its assertions: switching a call site from intra
    /// to inter can only decrease the reconstructed magnitude, never
    /// increase it.
    #[test]
    fn intra_offset_dominates_inter_offset() {
        for q in 0..=127u32 {
            let oi = quant_offset_for(q, true);
            let on = quant_offset_for(q, false);
            assert!(
                oi >= on,
                "q={q}: intra offset {oi} should be >= inter offset {on}"
            );
            if q >= 2 {
                assert!(
                    oi > on,
                    "q={q}: intra offset {oi} should be strictly > inter offset {on}"
                );
            }
        }
    }

    /// The reconstruction-interval invariant from the spec's note in
    /// §13.2.1: `3 ≤ offset + 2 < quant_factor` must hold for every
    /// `q >= 2` (the `0/1` carve-outs deliberately violate the
    /// `< quant_factor` half — they sit at the boundary). Pinned for
    /// both branches because either branch crossing it would break
    /// the "inverse_quant then re-quantise" idempotence the spec
    /// guarantees.
    #[test]
    fn inter_offset_satisfies_reconstruction_interval_for_q_ge_2() {
        for q in 2..=127u32 {
            let qf = quant_factor(q);
            let oi = quant_offset_for(q, true);
            let on = quant_offset_for(q, false);
            assert!(
                oi + 2 >= 3 && oi + 2 < qf,
                "q={q}: intra offset {oi} fails 3 <= o+2 < qf={qf}"
            );
            assert!(
                on + 2 >= 3 && on + 2 < qf,
                "q={q}: inter offset {on} fails 3 <= o+2 < qf={qf}"
            );
        }
    }

    /// At `q == 0` and `q == 1` the intra and inter inverse-quant
    /// outputs are bit-identical for every signed coefficient, so the
    /// encoder's self-roundtrip tests (which all run at qindex=0) are
    /// unaffected by routing through the inter branch.
    #[test]
    fn inverse_quant_intra_inter_identical_at_low_q() {
        for q in 0..=1u32 {
            for x in [-7i32, -3, -1, 0, 1, 3, 7, 42, -42] {
                assert_eq!(
                    inverse_quant_for(x, q, true),
                    inverse_quant_for(x, q, false),
                    "q={q}, x={x}: intra and inter disagree"
                );
            }
        }
        // Sanity: the no-arg alias matches the intra branch.
        assert_eq!(inverse_quant(5, 0), inverse_quant_for(5, 0, true));
        assert_eq!(inverse_quant(-7, 1), inverse_quant_for(-7, 1, true));
    }

    /// At higher `q` the inter reconstruction is strictly closer to
    /// zero than the intra reconstruction for every non-zero
    /// coefficient. Pinned at a handful of `(x, q)` pairs to lock the
    /// behaviour change visible to a calling decoder.
    #[test]
    fn inverse_quant_inter_pulls_toward_zero() {
        for &q in &[2u32, 4, 8, 16, 32] {
            for &x in &[-5i32, -1, 1, 5] {
                let v_intra = inverse_quant_for(x, q, true);
                let v_inter = inverse_quant_for(x, q, false);
                // Same sign as `x` (sign comes from the qcoeff, not the
                // offset).
                assert!(v_intra.signum() == x.signum() || v_intra == 0);
                assert!(v_inter.signum() == x.signum() || v_inter == 0);
                // Inter magnitude <= intra magnitude.
                assert!(
                    v_inter.unsigned_abs() <= v_intra.unsigned_abs(),
                    "q={q}, x={x}: inter |{v_inter}| > intra |{v_intra}|"
                );
            }
        }
    }
}
