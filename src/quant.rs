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

use crate::bits::BitReader;
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
    // The qindex field is 7-bit (§13.5.4 — read as `read_nbits(7)` for LD
    // or `read_uint_lit(1) & 0x7F` semantically for HQ), so the spec
    // upper bound is q == 127. A malformed bitstream can deliver q == 255
    // through the unmasked `read_uint_lit(1)` path (`decode_hq_slice`)
    // which would walk `1u64 << (255/4) = 1u64 << 63` and then overflow
    // the per-branch multiplications (e.g. `503_829 * (1<<63)`). Clamp
    // here so the function honours its "valid for any q ≥ 0" docstring
    // without needing every caller to pre-validate the field. Clamping
    // at 127 keeps the saturation behaviour identical for in-spec inputs
    // and at most produces a `u32::MAX` `qf` for out-of-range qindex
    // values — which then forward-quantises every coefficient to 0,
    // exactly the "saturation is behaviour-preserving" property the
    // docstring relies on.
    let q = q.min(127);
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
/// decoding (§13.5.4). Indexed by `(level, orient)`.
///
/// For a **symmetric** transform (`dwt_depth_ho == 0`) level 0 carries
/// only LL and every higher level carries an HL/LH/HH triplet.
///
/// For an **asymmetric** (horizontal-only) transform
/// (`dwt_depth_ho > 0`, SMPTE ST 2042-1:2022 §12.4.5.3 / §13.2.1) the
/// total number of levels is `dwt_depth_ho + dwt_depth`. Level 0 then
/// carries only the L (DC) band, levels `1..=dwt_depth_ho` carry a
/// single horizontal-only H band, and the remaining
/// `dwt_depth_ho+1 ..= dwt_depth_ho+dwt_depth` levels each carry an
/// HL/LH/HH triplet. Both the single L band and the single H band live
/// in the index-0 ("low") slot of their level, mirroring the LL
/// convention, so the storage layout is unchanged.
#[derive(Debug, Clone)]
pub struct QuantMatrix {
    pub dwt_depth: u32,
    /// Horizontal-only transform depth (§12.4.4.3). `0` for a
    /// symmetric transform; the first `dwt_depth_ho` non-DC levels are
    /// then horizontal-only H bands rather than HL/LH/HH triplets.
    pub dwt_depth_ho: u32,
    /// `levels[level][orient_index]`. `orient_index` matches
    /// [`Orient::as_index`], so LL=0, HL=1, LH=2, HH=3. For asymmetric
    /// levels the single L / H band uses the index-0 slot.
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
        Some(Self {
            dwt_depth,
            dwt_depth_ho: 0,
            levels,
        })
    }

    /// Look up the default *asymmetric* (horizontal-only) quantisation
    /// matrix for a `(wavelet_index, wavelet_index_ho, dwt_depth,
    /// dwt_depth_ho)` combination, per SMPTE ST 2042-1:2022 Annex D
    /// (Tables D.1–D.8, the `set_quant_matrix()` referenced from
    /// §12.4.5.3).
    ///
    /// Annex D defines defaults for two sets of cases:
    ///
    /// 1. `wavelet_index_ho == wavelet_index` (Tables D.1–D.7, one per
    ///    filter index 0..=6), with `dwt_depth <= 4`, `dwt_depth_ho <= 4`
    ///    and `dwt_depth + dwt_depth_ho <= 5`.
    /// 2. `wavelet_index == 3` (Haar, no shift) and
    ///    `wavelet_index_ho == 1` (LeGall) — Table D.8 — under the same
    ///    depth bounds.
    ///
    /// Any combination outside those tables returns `None`: the spec
    /// then requires a custom matrix (§12.4.5.3).
    ///
    /// With `dwt_depth_ho == 0` this reduces to the symmetric Annex D
    /// defaults and is consistent with [`Self::default_for`]; callers
    /// that already have a symmetric stream may use either.
    ///
    /// The returned [`QuantMatrix`] follows the same storage convention
    /// as [`Self::parse_custom`]: level 0 holds the L (or LL) value in
    /// slot 0, levels `1..=dwt_depth_ho` hold the single H band in slot
    /// 0, and the remaining `dwt_depth` levels hold the HL/LH/HH triplet
    /// in slots 1..=3.
    pub fn default_for_asymmetric(
        wavelet: WaveletFilter,
        wavelet_ho: WaveletFilter,
        dwt_depth: u32,
        dwt_depth_ho: u32,
    ) -> Option<Self> {
        // Annex D depth bounds (the "Combined values ... not present in
        // the tables ... shall require a custom matrix" clause).
        if dwt_depth > 4 || dwt_depth_ho > 4 || dwt_depth + dwt_depth_ho > 5 {
            return None;
        }
        let d = dwt_depth as usize;
        let ho = dwt_depth_ho as usize;

        // Build the level list from a (l0, h_bands, triplets) shape that
        // is constant across the `dwt_depth` column (Tables D.1, D.2,
        // D.3, D.6, D.7). `l0` is the level-0 L/LL value; `h_bands` the
        // per-level H values (levels 1..=ho); `triplets` the bottom-up
        // HL/LH/HH triplets, of which the first `dwt_depth` apply.
        fn build(
            l0: u32,
            h_bands: &[u32],
            triplets: &[(u32, u32, u32)],
            d: usize,
        ) -> Vec<[u32; 4]> {
            let mut levels: Vec<[u32; 4]> = Vec::with_capacity(1 + h_bands.len() + d);
            levels.push([l0, 0, 0, 0]);
            for &h in h_bands {
                levels.push([h, 0, 0, 0]);
            }
            for &(hl, lh, hh) in triplets.iter().take(d) {
                levels.push([0, hl, lh, hh]);
            }
            levels
        }

        // For the depth-INVARIANT tables (D.1/2/3/6/7), each `ho` block
        // names the level-0 L value, the H-band values for levels
        // 1..=ho, and the full triplet list (the first `dwt_depth` of
        // which are emitted).
        type DepthInvariant = (u32, &'static [u32], &'static [(u32, u32, u32)]);

        let depth_invariant: Option<DepthInvariant> = match (wavelet, wavelet_ho) {
            // Table D.1 — Deslauriers-Dubuc (9,7), index 0.
            (WaveletFilter::DeslauriersDubuc9_7, WaveletFilter::DeslauriersDubuc9_7) => {
                Some(match ho {
                    0 => (5, &[], &[(3, 3, 0), (4, 4, 1), (5, 5, 2), (6, 6, 3)]),
                    1 => (3, &[0], &[(3, 3, 0), (4, 4, 1), (5, 5, 2), (6, 6, 3)]),
                    2 => (3, &[0, 3], &[(5, 5, 3), (6, 6, 4), (7, 7, 5)]),
                    3 => (3, &[0, 3, 5], &[(8, 8, 5), (9, 9, 6)]),
                    _ => (3, &[0, 3, 5, 8], &[(10, 10, 8)]),
                })
            }
            // Table D.2 — LeGall (5,3), index 1.
            (WaveletFilter::LeGall5_3, WaveletFilter::LeGall5_3) => Some(match ho {
                0 => (4, &[], &[(2, 2, 0), (4, 4, 2), (5, 5, 3), (7, 7, 5)]),
                1 => (2, &[0], &[(3, 3, 1), (4, 4, 2), (6, 6, 4), (8, 8, 6)]),
                2 => (2, &[0, 3], &[(6, 6, 4), (7, 7, 5), (9, 9, 7)]),
                3 => (2, &[0, 3, 6], &[(8, 8, 6), (10, 10, 8)]),
                _ => (2, &[0, 3, 6, 8], &[(11, 11, 9)]),
            }),
            // Table D.3 — Deslauriers-Dubuc (13,7), index 2.
            (WaveletFilter::DeslauriersDubuc13_7, WaveletFilter::DeslauriersDubuc13_7) => {
                Some(match ho {
                    0 => (5, &[], &[(3, 3, 0), (4, 4, 1), (5, 5, 2), (6, 6, 3)]),
                    1 => (3, &[0], &[(3, 3, 0), (4, 4, 1), (5, 5, 2), (6, 6, 3)]),
                    2 => (3, &[0, 3], &[(5, 5, 2), (6, 6, 4), (7, 7, 5)]),
                    3 => (3, &[0, 3, 5], &[(8, 8, 5), (9, 9, 6)]),
                    _ => (3, &[0, 3, 5, 8], &[(10, 10, 8)]),
                })
            }
            // Table D.6 — Fidelity, index 5.
            (WaveletFilter::Fidelity, WaveletFilter::Fidelity) => Some(match ho {
                0 => (0, &[], &[(4, 4, 8), (8, 8, 12), (13, 13, 17), (17, 17, 21)]),
                1 => (
                    0,
                    &[4],
                    &[(6, 6, 10), (11, 11, 15), (15, 15, 19), (19, 19, 23)],
                ),
                2 => (0, &[4, 6], &[(8, 8, 12), (13, 13, 17), (17, 17, 21)]),
                3 => (0, &[4, 6, 8], &[(11, 11, 15), (15, 15, 19)]),
                _ => (0, &[4, 6, 8, 11], &[(13, 13, 17)]),
            }),
            // Table D.7 — Daubechies (9,7), index 6.
            (WaveletFilter::Daubechies9_7, WaveletFilter::Daubechies9_7) => Some(match ho {
                0 => (3, &[], &[(1, 1, 0), (4, 4, 2), (6, 6, 5), (9, 9, 7)]),
                1 => (1, &[0], &[(3, 3, 2), (6, 6, 4), (8, 8, 7), (11, 11, 9)]),
                2 => (1, &[0, 3], &[(6, 6, 5), (9, 9, 8), (11, 11, 10)]),
                3 => (1, &[0, 3, 6], &[(10, 10, 8), (12, 12, 11)]),
                _ => (1, &[0, 3, 6, 10], &[(13, 13, 12)]),
            }),
            _ => None,
        };

        if let Some((l0, h_bands, triplets)) = depth_invariant {
            return Some(Self {
                dwt_depth,
                dwt_depth_ho,
                levels: build(l0, h_bands, triplets, d),
            });
        }

        // Depth-DEPENDENT tables (D.4 Haar0, D.5 Haar1, D.8 Haar0/LeGall):
        // the L/H band values change between the `dwt_depth` columns
        // because the Haar lifting carries a per-level shift. These are
        // transcribed cell-by-cell, keyed `[ho][depth]` → full level
        // list. An empty list marks an invalid `(ho, depth)` cell.
        let levels = match (wavelet, wavelet_ho) {
            (WaveletFilter::Haar0, WaveletFilter::Haar0) => haar0_levels(ho, d)?,
            (WaveletFilter::Haar1, WaveletFilter::Haar1) => haar1_levels(ho, d)?,
            (WaveletFilter::Haar0, WaveletFilter::LeGall5_3) => haar0_legall_levels(ho, d)?,
            _ => return None,
        };
        Some(Self {
            dwt_depth,
            dwt_depth_ho,
            levels,
        })
    }

    /// Parse a custom quantisation matrix (SMPTE ST 2042-1:2022
    /// §12.4.5.3, the `custom_quant_matrix == True` branch), supporting
    /// both the symmetric (`dwt_depth_ho == 0`) and asymmetric
    /// (`dwt_depth_ho > 0`) level layouts.
    ///
    /// The caller has already consumed the `custom_quant_matrix` flag
    /// and `wavelet_index` / `dwt_depth` / `dwt_depth_ho`; this reads
    /// the matrix body in stream order:
    ///
    /// * Level 0 — one `read_uint`: LL if symmetric, otherwise L.
    /// * Levels `1..=dwt_depth_ho` (asymmetric only) — one `read_uint`
    ///   each: the horizontal-only H band.
    /// * Levels `dwt_depth_ho+1 ..= dwt_depth_ho+dwt_depth` — three
    ///   `read_uint`s each, in HL, LH, HH order.
    ///
    /// With `dwt_depth_ho == 0` this is bit-for-bit the legacy
    /// symmetric read (LL then `dwt_depth` HL/LH/HH triplets).
    pub fn parse_custom(r: &mut BitReader<'_>, dwt_depth: u32, dwt_depth_ho: u32) -> Self {
        let total = (dwt_depth_ho + dwt_depth) as usize + 1;
        let mut levels: Vec<[u32; 4]> = Vec::with_capacity(total);
        if dwt_depth_ho == 0 {
            // Symmetric: level 0 is LL (index 0).
            let ll = r.read_uint();
            levels.push([ll, 0, 0, 0]);
        } else {
            // Asymmetric: level 0 is the single L (DC) band, stored in
            // the index-0 slot; levels 1..=dwt_depth_ho are single H
            // bands, also in the index-0 slot.
            let l = r.read_uint();
            levels.push([l, 0, 0, 0]);
            for _ in 1..=dwt_depth_ho {
                let h = r.read_uint();
                levels.push([h, 0, 0, 0]);
            }
        }
        // Remaining levels carry an HL/LH/HH triplet, regardless of the
        // symmetric/asymmetric distinction.
        for _ in 0..dwt_depth {
            let hl = r.read_uint();
            let lh = r.read_uint();
            let hh = r.read_uint();
            levels.push([0, hl, lh, hh]);
        }
        Self {
            dwt_depth,
            dwt_depth_ho,
            levels,
        }
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

/// Assemble a [`QuantMatrix::levels`] list from the explicit per-cell
/// shape of a depth-dependent Annex D table (D.4 / D.5 / D.8): the
/// level-0 L value, the per-level H-band values (levels 1..=ho), and
/// the bottom-up HL/LH/HH triplets (levels ho+1..=ho+depth).
/// One depth-dependent Annex D cell: the level-0 L value, the per-level
/// H-band values (levels `1..=dwt_depth_ho`), and the bottom-up
/// HL/LH/HH triplets (levels `dwt_depth_ho+1..=dwt_depth_ho+dwt_depth`).
type AnnexDCell = (u32, &'static [u32], &'static [(u32, u32, u32)]);

fn assemble_levels(l0: u32, h_bands: &[u32], triplets: &[(u32, u32, u32)]) -> Vec<[u32; 4]> {
    let mut levels: Vec<[u32; 4]> = Vec::with_capacity(1 + h_bands.len() + triplets.len());
    levels.push([l0, 0, 0, 0]);
    for &h in h_bands {
        levels.push([h, 0, 0, 0]);
    }
    for &(hl, lh, hh) in triplets {
        levels.push([0, hl, lh, hh]);
    }
    levels
}

/// SMPTE ST 2042-1:2022 Table D.4 — default matrices for
/// `wavelet_index == 3` and `wavelet_index_ho == 3` (Haar, no shift).
/// Transcribed per `(dwt_depth_ho, dwt_depth)` cell; `None` for cells
/// outside the table (those require a custom matrix).
fn haar0_levels(ho: usize, d: usize) -> Option<Vec<[u32; 4]>> {
    // For Haar0, level 0 = 4*(ho_offset) ... the L/H bands change with
    // depth, so the cells are transcribed verbatim rather than derived.
    let (l0, h, t): AnnexDCell = match (ho, d) {
        (0, 1) => (8, &[], &[(4, 4, 0)]),
        (0, 2) => (12, &[], &[(8, 8, 4), (4, 4, 0)]),
        (0, 3) => (16, &[], &[(12, 12, 8), (8, 8, 4), (4, 4, 0)]),
        (0, 4) => (20, &[], &[(16, 16, 12), (12, 12, 8), (8, 8, 4), (4, 4, 0)]),
        (1, 0) => (4, &[0], &[]),
        (1, 1) => (10, &[6], &[(4, 4, 0)]),
        (1, 2) => (14, &[10], &[(8, 8, 4), (4, 4, 0)]),
        (1, 3) => (18, &[14], &[(12, 12, 8), (8, 8, 4), (4, 4, 0)]),
        (1, 4) => (
            22,
            &[18],
            &[(16, 16, 12), (12, 12, 8), (8, 8, 4), (4, 4, 0)],
        ),
        (2, 0) => (6, &[2, 0], &[]),
        (2, 1) => (12, &[8, 6], &[(4, 4, 0)]),
        (2, 2) => (16, &[12, 10], &[(8, 8, 4), (4, 4, 0)]),
        (2, 3) => (20, &[16, 14], &[(12, 12, 8), (8, 8, 4), (4, 4, 0)]),
        (3, 0) => (8, &[4, 2, 0], &[]),
        (3, 1) => (14, &[10, 8, 6], &[(4, 4, 0)]),
        (3, 2) => (18, &[14, 12, 10], &[(8, 8, 4), (4, 4, 0)]),
        (4, 0) => (10, &[6, 4, 2, 0], &[]),
        (4, 1) => (16, &[12, 10, 8, 6], &[(4, 4, 0)]),
        _ => return None,
    };
    Some(assemble_levels(l0, h, t))
}

/// SMPTE ST 2042-1:2022 Table D.5 — default matrices for
/// `wavelet_index == 4` and `wavelet_index_ho == 4` (Haar, single shift
/// per level).
fn haar1_levels(ho: usize, d: usize) -> Option<Vec<[u32; 4]>> {
    let (l0, h, t): AnnexDCell = match (ho, d) {
        (0, 1) => (8, &[], &[(4, 4, 0)]),
        (0, 2) => (8, &[], &[(4, 4, 0), (4, 4, 0)]),
        (0, 3) => (8, &[], &[(4, 4, 0), (4, 4, 0), (4, 4, 0)]),
        (0, 4) => (8, &[], &[(4, 4, 0), (4, 4, 0), (4, 4, 0), (4, 4, 0)]),
        (1, 0) => (4, &[0], &[]),
        (1, 1) => (6, &[2], &[(4, 4, 0)]),
        (1, 2) => (6, &[2], &[(4, 4, 0), (4, 4, 0)]),
        (1, 3) => (6, &[2], &[(4, 4, 0), (4, 4, 0), (4, 4, 0)]),
        (1, 4) => (6, &[2], &[(4, 4, 0), (4, 4, 0), (4, 4, 0), (4, 4, 0)]),
        (2, 0) => (4, &[0, 2], &[]),
        (2, 1) => (4, &[0, 2], &[(4, 4, 0)]),
        (2, 2) => (4, &[0, 2], &[(4, 4, 0), (4, 4, 0)]),
        (2, 3) => (4, &[0, 2], &[(4, 4, 0), (4, 4, 0), (4, 4, 0)]),
        (3, 0) => (4, &[0, 2, 4], &[]),
        (3, 1) => (4, &[0, 2, 4], &[(6, 6, 2)]),
        (3, 2) => (4, &[0, 2, 4], &[(6, 6, 2), (6, 6, 2)]),
        (4, 0) => (4, &[0, 2, 4, 6], &[]),
        (4, 1) => (4, &[0, 2, 4, 6], &[(8, 8, 4)]),
        _ => return None,
    };
    Some(assemble_levels(l0, h, t))
}

/// SMPTE ST 2042-1:2022 Table D.8 — default matrices for
/// `wavelet_index == 3` (Haar, no shift) and `wavelet_index_ho == 1`
/// (LeGall). The single cross-filter default the spec defines.
fn haar0_legall_levels(ho: usize, d: usize) -> Option<Vec<[u32; 4]>> {
    let (l0, h, t): AnnexDCell = match (ho, d) {
        (0, 1) => (6, &[], &[(4, 2, 0)]),
        (0, 2) => (6, &[], &[(4, 2, 0), (5, 3, 1)]),
        (0, 3) => (6, &[], &[(4, 2, 0), (5, 3, 1), (6, 4, 2)]),
        (0, 4) => (6, &[], &[(4, 2, 0), (5, 3, 1), (6, 4, 2), (6, 5, 2)]),
        (1, 0) => (2, &[0], &[]),
        (1, 1) => (3, &[1], &[(4, 2, 0)]),
        (1, 2) => (3, &[1], &[(4, 2, 0), (5, 3, 1)]),
        (1, 3) => (3, &[1], &[(4, 2, 0), (5, 3, 1), (6, 4, 2)]),
        (1, 4) => (3, &[1], &[(4, 2, 0), (5, 3, 1), (6, 4, 2), (6, 5, 2)]),
        (2, 0) => (2, &[0, 3], &[]),
        (2, 1) => (2, &[0, 3], &[(6, 4, 2)]),
        (2, 2) => (2, &[0, 3], &[(6, 4, 2), (6, 5, 2)]),
        (2, 3) => (2, &[0, 3], &[(6, 4, 2), (6, 5, 2), (7, 5, 3)]),
        (3, 0) => (2, &[0, 3, 6], &[]),
        (3, 1) => (2, &[0, 3, 6], &[(8, 7, 4)]),
        (3, 2) => (2, &[0, 3, 6], &[(8, 7, 4), (9, 7, 5)]),
        (4, 0) => (2, &[0, 3, 6, 8], &[]),
        (4, 1) => (2, &[0, 3, 6, 8], &[(11, 9, 7)]),
        _ => return None,
    };
    Some(assemble_levels(l0, h, t))
}

/// `slice_quantizers` (§13.5.5): subtract each subband's quantisation
/// matrix entry from `qindex`, clamping to 0. Returns the per-level
/// `[LL, HL, LH, HH]` effective quantisers, in **pyramid-slot layout**
/// ([`Orient::as_index`]):
///
/// * Symmetric (`dwt_depth_ho == 0`): level 0 carries the LL quantiser
///   in slot 0; every higher level carries HL/LH/HH in slots 1..=3.
/// * Asymmetric (`dwt_depth_ho > 0`, §13.5.5 else-branch): level 0
///   carries the L quantiser in slot 0; levels `1..=dwt_depth_ho`
///   carry the single H-band quantiser in **slot 3** — the slot the
///   horizontal-only H band occupies in the coefficient pyramid (see
///   [`crate::subband::init_pyramid_ho`] /
///   [`crate::wavelet::idwt_with_ho`]) — so a single `(level, slot)`
///   pair indexes both the band and its quantiser. Note the *matrix*
///   stores the H entry in its index-0 slot (the §12.4.5.3
///   stream-order convention documented on [`QuantMatrix::levels`]);
///   this function performs the slot bridge.
pub fn slice_quantisers(qindex: u32, qmatrix: &QuantMatrix) -> Vec<[u32; 4]> {
    let ho = qmatrix.dwt_depth_ho as usize;
    let mut out: Vec<[u32; 4]> = Vec::with_capacity(qmatrix.levels.len());
    for (level, triplet) in qmatrix.levels.iter().enumerate() {
        if level == 0 {
            // LL (symmetric) / L (asymmetric) — slot 0 either way.
            out.push([qindex.saturating_sub(triplet[0]), 0, 0, 0]);
        } else if level <= ho {
            // Horizontal-only H band: matrix slot 0 → pyramid slot 3.
            out.push([0, 0, 0, qindex.saturating_sub(triplet[0])]);
        } else {
            let q_hl = qindex.saturating_sub(triplet[1]);
            let q_lh = qindex.saturating_sub(triplet[2]);
            let q_hh = qindex.saturating_sub(triplet[3]);
            out.push([0, q_hl, q_lh, q_hh]);
        }
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

    /// §13.5.5 asymmetric branch: the L quantiser stays in slot 0 and
    /// the per-level H quantiser is emitted in pyramid slot 3 (where
    /// the H band itself lives), bridging the matrix's slot-0 storage.
    #[test]
    fn slice_quantisers_asymmetric_slot_bridge() {
        // dwt_depth_ho = 2, dwt_depth = 1:
        //   level 0: L  = 7, level 1: H = 3, level 2: H = 1,
        //   level 3: HL,LH,HH = 5,6,8.
        let qm = QuantMatrix {
            dwt_depth: 1,
            dwt_depth_ho: 2,
            levels: vec![[7, 0, 0, 0], [3, 0, 0, 0], [1, 0, 0, 0], [0, 5, 6, 8]],
        };
        let q = slice_quantisers(10, &qm);
        assert_eq!(q[0], [3, 0, 0, 0]); // L: 10 - 7
        assert_eq!(q[1], [0, 0, 0, 7]); // H: 10 - 3, in slot 3
        assert_eq!(q[2], [0, 0, 0, 9]); // H: 10 - 1, in slot 3
        assert_eq!(q[3], [0, 5, 4, 2]); // HL/LH/HH: 10 - {5,6,8}
    }

    #[test]
    fn dwt_depth_greater_than_four_rejected() {
        assert!(QuantMatrix::default_for(WaveletFilter::LeGall5_3, 5).is_none());
    }

    // ---- Annex D asymmetric default matrices ---------------------------

    /// With `dwt_depth_ho == 0` and `wavelet_index_ho == wavelet_index`
    /// the Annex D lookup must agree with the legacy symmetric
    /// `default_for` across every filter and every depth 1..=4 (the
    /// §12.4.4 NOTE invariant — v3 reduces to v2 at the defaults).
    #[test]
    fn default_for_asymmetric_symmetric_case_matches_default_for() {
        let filters = [
            WaveletFilter::DeslauriersDubuc9_7,
            WaveletFilter::LeGall5_3,
            WaveletFilter::DeslauriersDubuc13_7,
            WaveletFilter::Haar0,
            WaveletFilter::Haar1,
            WaveletFilter::Fidelity,
            WaveletFilter::Daubechies9_7,
        ];
        for f in filters {
            for depth in 1..=4 {
                let sym = QuantMatrix::default_for(f, depth).unwrap();
                let asym = QuantMatrix::default_for_asymmetric(f, f, depth, 0).unwrap();
                assert_eq!(
                    sym.levels, asym.levels,
                    "filter {f:?} depth {depth}: symmetric vs Annex D ho=0 mismatch"
                );
                assert_eq!(asym.dwt_depth_ho, 0);
            }
        }
    }

    /// Table D.2 (LeGall) `dwt_depth_ho = 2`, `dwt_depth = 1`: L=2,
    /// H bands 0 and 3, then the level-3 triplet `6, 6, 4`.
    #[test]
    fn default_for_asymmetric_legall_d2_ho2_depth1() {
        let qm = QuantMatrix::default_for_asymmetric(
            WaveletFilter::LeGall5_3,
            WaveletFilter::LeGall5_3,
            1,
            2,
        )
        .unwrap();
        assert_eq!(qm.dwt_depth, 1);
        assert_eq!(qm.dwt_depth_ho, 2);
        assert_eq!(qm.levels.len(), 4);
        assert_eq!(qm.levels[0], [2, 0, 0, 0]); // L
        assert_eq!(qm.levels[1], [0, 0, 0, 0]); // H = 0
        assert_eq!(qm.levels[2], [3, 0, 0, 0]); // H = 3
        assert_eq!(qm.levels[3], [0, 6, 6, 4]); // HL, LH, HH
    }

    /// Table D.4 (Haar0) `dwt_depth_ho = 2`, `dwt_depth = 3`: the
    /// depth-dependent L/H band values plus three triplets bottom-up.
    #[test]
    fn default_for_asymmetric_haar0_d4_ho2_depth3() {
        let qm =
            QuantMatrix::default_for_asymmetric(WaveletFilter::Haar0, WaveletFilter::Haar0, 3, 2)
                .unwrap();
        assert_eq!(qm.levels.len(), 6);
        assert_eq!(qm.levels[0], [20, 0, 0, 0]); // L
        assert_eq!(qm.levels[1], [16, 0, 0, 0]); // H
        assert_eq!(qm.levels[2], [14, 0, 0, 0]); // H
        assert_eq!(qm.levels[3], [0, 12, 12, 8]);
        assert_eq!(qm.levels[4], [0, 8, 8, 4]);
        assert_eq!(qm.levels[5], [0, 4, 4, 0]);
    }

    /// Table D.8 — the cross-filter default: `wavelet_index == 3`
    /// (Haar0 vertical) and `wavelet_index_ho == 1` (LeGall horizontal),
    /// `dwt_depth_ho = 1`, `dwt_depth = 2`.
    #[test]
    fn default_for_asymmetric_haar0_legall_d8_ho1_depth2() {
        let qm = QuantMatrix::default_for_asymmetric(
            WaveletFilter::Haar0,
            WaveletFilter::LeGall5_3,
            2,
            1,
        )
        .unwrap();
        assert_eq!(qm.levels.len(), 4);
        assert_eq!(qm.levels[0], [3, 0, 0, 0]); // L
        assert_eq!(qm.levels[1], [1, 0, 0, 0]); // H
        assert_eq!(qm.levels[2], [0, 4, 2, 0]); // HL, LH, HH
        assert_eq!(qm.levels[3], [0, 5, 3, 1]);
    }

    /// Table D.7 (Daubechies (9,7)) `dwt_depth_ho = 4`, `dwt_depth = 1`
    /// — the deepest horizontal-only column the table defines.
    #[test]
    fn default_for_asymmetric_daubechies_d7_ho4_depth1() {
        let qm = QuantMatrix::default_for_asymmetric(
            WaveletFilter::Daubechies9_7,
            WaveletFilter::Daubechies9_7,
            1,
            4,
        )
        .unwrap();
        assert_eq!(qm.levels.len(), 6);
        assert_eq!(qm.levels[0], [1, 0, 0, 0]); // L
        assert_eq!(qm.levels[1], [0, 0, 0, 0]); // H
        assert_eq!(qm.levels[2], [3, 0, 0, 0]); // H
        assert_eq!(qm.levels[3], [6, 0, 0, 0]); // H
        assert_eq!(qm.levels[4], [10, 0, 0, 0]); // H
        assert_eq!(qm.levels[5], [0, 13, 13, 12]);
    }

    /// Annex D depth bounds: `dwt_depth + dwt_depth_ho > 5`, or either
    /// axis > 4, or a non-defined filter pair, returns `None` (the spec
    /// then requires a custom matrix).
    #[test]
    fn default_for_asymmetric_off_table_is_none() {
        // sum > 5.
        assert!(QuantMatrix::default_for_asymmetric(
            WaveletFilter::LeGall5_3,
            WaveletFilter::LeGall5_3,
            3,
            3
        )
        .is_none());
        // dwt_depth_ho > 4.
        assert!(QuantMatrix::default_for_asymmetric(
            WaveletFilter::LeGall5_3,
            WaveletFilter::LeGall5_3,
            0,
            5
        )
        .is_none());
        // Undefined cross-filter pair (LeGall vertical + DD9,7 ho).
        assert!(QuantMatrix::default_for_asymmetric(
            WaveletFilter::LeGall5_3,
            WaveletFilter::DeslauriersDubuc9_7,
            1,
            1
        )
        .is_none());
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

    // ---- §12.4.5.3 custom quant_matrix parsing ----

    /// Encode a sequence of `read_uint` values into a byte buffer the
    /// way the bitstream packs them, then hand back a reader cursored
    /// at the start. Used to drive [`QuantMatrix::parse_custom`].
    fn reader_for(values: &[u32]) -> (Vec<u8>, usize) {
        let mut w = crate::bitwriter::BitWriter::new();
        for &v in values {
            w.write_uint(v);
        }
        // Trailing align byte so the reader always has a full byte to
        // consume; `byte_pos` after parsing tells us how far we read.
        let bytes = w.finish();
        let n = values.len();
        (bytes, n)
    }

    /// With `dwt_depth_ho == 0` the custom parser is bit-for-bit the
    /// legacy symmetric layout: LL at level 0, then `dwt_depth`
    /// HL/LH/HH triplets. Mirrors the inline read the picture parser
    /// used before the asymmetric lift.
    #[test]
    fn parse_custom_symmetric_matches_legacy_layout() {
        // depth 2: LL=4, then (HL,LH,HH) for levels 1 and 2.
        let vals = [4u32, 2, 2, 0, 5, 5, 3];
        let (bytes, _) = reader_for(&vals);
        let mut r = BitReader::new(&bytes);
        let qm = QuantMatrix::parse_custom(&mut r, 2, 0);
        assert_eq!(qm.dwt_depth, 2);
        assert_eq!(qm.dwt_depth_ho, 0);
        assert_eq!(qm.levels.len(), 3);
        assert_eq!(qm.levels[0], [4, 0, 0, 0]);
        assert_eq!(qm.levels[1], [0, 2, 2, 0]);
        assert_eq!(qm.levels[2], [0, 5, 5, 3]);
        // Lookups resolve through the `Orient` enum.
        assert_eq!(qm.get(0, Orient::LL), 4);
        assert_eq!(qm.get(2, Orient::HL), 5);
        assert_eq!(qm.get(2, Orient::HH), 3);
    }

    /// §12.4.5.3 asymmetric layout (`dwt_depth_ho > 0`): level 0 is a
    /// single L (DC) band, levels `1..=dwt_depth_ho` are single H
    /// bands, then `dwt_depth` HL/LH/HH triplets. Total levels =
    /// `dwt_depth_ho + dwt_depth + 1`. Both the L and the H bands live
    /// in the index-0 slot (§13.2.1).
    #[test]
    fn parse_custom_asymmetric_layout() {
        // dwt_depth_ho = 2, dwt_depth = 1.
        //   level 0: L  = 7
        //   level 1: H  = 11
        //   level 2: H  = 13
        //   level 3: HL,LH,HH = 5,6,8
        let vals = [7u32, 11, 13, 5, 6, 8];
        let (bytes, _) = reader_for(&vals);
        let mut r = BitReader::new(&bytes);
        let qm = QuantMatrix::parse_custom(&mut r, 1, 2);
        assert_eq!(qm.dwt_depth, 1);
        assert_eq!(qm.dwt_depth_ho, 2);
        assert_eq!(qm.levels.len(), 4);
        // L (DC) and the two H bands all use the index-0 ("low") slot.
        assert_eq!(qm.levels[0], [7, 0, 0, 0]);
        assert_eq!(qm.levels[1], [11, 0, 0, 0]);
        assert_eq!(qm.levels[2], [13, 0, 0, 0]);
        assert_eq!(qm.levels[3], [0, 5, 6, 8]);
    }

    /// The asymmetric read consumes exactly `1 + dwt_depth_ho +
    /// 3*dwt_depth` uints and leaves the reader byte-aligned at the
    /// expected boundary — pinning that the parser neither over- nor
    /// under-reads, which is what keeps the subsequent slice data
    /// aligned in a real stream.
    #[test]
    fn parse_custom_asymmetric_consumes_exact_uints() {
        // Pad with a sentinel after the matrix; assert we can still read it.
        let matrix = [3u32, 9, 4, 7, 2];
        let sentinel = 42u32;
        let mut all: Vec<u32> = matrix.to_vec();
        all.push(sentinel);
        let (bytes, _) = reader_for(&all);
        let mut r = BitReader::new(&bytes);
        // dwt_depth_ho = 1, dwt_depth = 1 → 1 + 1 + 3 = 5 uints.
        let qm = QuantMatrix::parse_custom(&mut r, 1, 1);
        assert_eq!(qm.levels[0], [3, 0, 0, 0]); // L
        assert_eq!(qm.levels[1], [9, 0, 0, 0]); // H
        assert_eq!(qm.levels[2], [0, 4, 7, 2]); // HL,LH,HH
                                                // The very next uint must be the sentinel: proves the matrix
                                                // read stopped at the right bit.
        assert_eq!(r.read_uint(), sentinel);
    }

    /// A zero-depth, zero-ho matrix is a single LL entry and nothing
    /// else — the degenerate boundary.
    #[test]
    fn parse_custom_zero_depth() {
        let (bytes, _) = reader_for(&[6u32]);
        let mut r = BitReader::new(&bytes);
        let qm = QuantMatrix::parse_custom(&mut r, 0, 0);
        assert_eq!(qm.levels.len(), 1);
        assert_eq!(qm.levels[0], [6, 0, 0, 0]);
    }
}
