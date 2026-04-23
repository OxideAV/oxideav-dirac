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
pub fn quant_factor(q: u32) -> u32 {
    let base: u32 = 1u32 << (q / 4);
    match q % 4 {
        0 => 4 * base,
        1 => (503_829 * base + 52_958) / 105_917,
        2 => (665_857 * base + 58_854) / 117_708,
        3 => (440_253 * base + 32_722) / 65_444,
        _ => unreachable!(),
    }
}

/// `quant_offset(q)` per VC-2 §13.3.2 (SMPTE ST 2042-1:2022). The
/// 2008 Dirac spec had an intra / inter split; VC-2 collapsed both
/// branches into a single formula. Since this crate currently
/// supports only intra pictures the distinction is moot either way.
pub fn quant_offset(q: u32) -> u32 {
    if q == 0 {
        1
    } else if q == 1 {
        2
    } else {
        (quant_factor(q) + 1) / 2
    }
}

/// `inverse_quant(qcoeff, q)` (§13.3.1).
pub fn inverse_quant(qcoeff: i32, q: u32) -> i32 {
    if qcoeff == 0 {
        return 0;
    }
    let mag = qcoeff.unsigned_abs() as u64;
    let qf = quant_factor(q) as u64;
    let off = quant_offset(q) as u64;
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
                for (i, &(hl, lh, hh)) in
                    hl_row.iter().take(dwt_depth as usize).enumerate()
                {
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
}
