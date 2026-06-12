//! Subband data arrays (§13.1).
//!
//! A wavelet subband is a 2-D array of signed-integer coefficients.
//! Coefficients occupy positions `(y, x)` with `0 ≤ x < width` and
//! `0 ≤ y < height`; the decoder fills these in-place via the
//! coefficient unpacking process (§13.4, §13.5). After unpacking, the
//! IDWT (§15.6) combines the subband pyramid into a single picture-
//! sized coefficient array that's eventually clipped and offset into
//! pixel values.

/// Subband orientation labels (§13.1). `LL` is DC-only and appears
/// exclusively at level 0; the remaining levels each carry an HL, LH
/// and HH triplet in that order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orient {
    /// Low-pass both directions — the DC band.
    LL,
    /// High-pass horizontal, low-pass vertical — "vertical" edges.
    HL,
    /// Low-pass horizontal, high-pass vertical — "horizontal" edges.
    LH,
    /// High-pass both directions — diagonal detail.
    HH,
}

impl Orient {
    /// Index used inside `[SubbandData; 4]` tables: 0=LL, 1=HL, 2=LH,
    /// 3=HH. Matches `pyramid[level][orient]` in [`crate::wavelet`].
    pub fn as_index(self) -> usize {
        match self {
            Self::LL => 0,
            Self::HL => 1,
            Self::LH => 2,
            Self::HH => 3,
        }
    }
}

/// A 2-D signed-integer subband array (§13.1.1 `initialise_wavelet_data`).
#[derive(Debug, Clone)]
pub struct SubbandData {
    pub width: usize,
    pub height: usize,
    /// Row-major; element at `(y, x)` lives at `data[y * width + x]`.
    pub data: Vec<i32>,
}

impl SubbandData {
    /// Create a `width × height` subband with all coefficients zeroed.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0i32; width * height],
        }
    }

    /// Immutable access. Panics if `(y, x)` is out of range — the
    /// spec's `band[y][x]`.
    #[inline]
    pub fn get(&self, y: usize, x: usize) -> i32 {
        self.data[y * self.width + x]
    }

    /// Mutable set — the spec's `band[y][x] = v`.
    #[inline]
    pub fn set(&mut self, y: usize, x: usize, v: i32) {
        self.data[y * self.width + x] = v;
    }

    /// Clear every coefficient to zero (`zero_subband_data`, §13.4.2.1).
    pub fn clear(&mut self) {
        for v in self.data.iter_mut() {
            *v = 0;
        }
    }

    /// A mutable slice spanning row `y`.
    pub fn row_mut(&mut self, y: usize) -> &mut [i32] {
        let w = self.width;
        &mut self.data[y * w..y * w + w]
    }
}

/// Per-component subband dimensions (§13.1.2). Returns `(width,
/// height)` for the subband at the given `level` — with level 0 being
/// the DC band and levels `1..=dwt_depth` each carrying a HL/LH/HH
/// triplet. The picture component dimensions are padded up to a
/// multiple of `2^dwt_depth` before computing subband sizes.
pub fn subband_dims(
    comp_width: u32,
    comp_height: u32,
    dwt_depth: u32,
    level: u32,
) -> (usize, usize) {
    subband_dims_ho(comp_width, comp_height, dwt_depth, 0, level)
}

/// Per-component subband dimensions for a (possibly asymmetric)
/// transform — SMPTE ST 2042-1:2022 §13.2.3 `subband_width` /
/// `subband_height`.
///
/// The component width is padded up to a multiple of
/// `2^(dwt_depth_ho + dwt_depth)`; the height only to a multiple of
/// `2^dwt_depth` (horizontal-only levels never halve the height).
/// Then, per the §13.2.3 pseudocode:
///
/// * width — level 0: `pw / 2^(dwt_depth_ho + dwt_depth)`; level
///   `n >= 1`: `pw / 2^(dwt_depth_ho + dwt_depth - n + 1)`.
/// * height — level `n <= dwt_depth_ho` (including 0):
///   `ph / 2^dwt_depth`; level `n > dwt_depth_ho`:
///   `ph / 2^(dwt_depth_ho + dwt_depth - n + 1)`.
///
/// With `dwt_depth_ho == 0` this reduces exactly to the symmetric
/// [`subband_dims`].
pub fn subband_dims_ho(
    comp_width: u32,
    comp_height: u32,
    dwt_depth: u32,
    dwt_depth_ho: u32,
    level: u32,
) -> (usize, usize) {
    let (pw, ph) = padded_component_dims_ho(comp_width, comp_height, dwt_depth, dwt_depth_ho);
    let total = dwt_depth_ho + dwt_depth;
    let w = if level == 0 {
        pw >> total
    } else {
        pw >> (total - level + 1)
    };
    let h = if level <= dwt_depth_ho {
        ph >> dwt_depth
    } else {
        ph >> (total - level + 1)
    };
    (w, h)
}

/// Luma / chroma padded component dimensions `(padded_width,
/// padded_height)` — the dimensions of the final IDWT output before
/// trimming back to the real picture size (§15.7).
pub fn padded_component_dims(comp_width: u32, comp_height: u32, dwt_depth: u32) -> (usize, usize) {
    padded_component_dims_ho(comp_width, comp_height, dwt_depth, 0)
}

/// Asymmetric-aware padded component dimensions (§13.2.3): the width
/// is padded up to a multiple of `2^(dwt_depth_ho + dwt_depth)` (every
/// transform level halves the width) while the height is only padded
/// to a multiple of `2^dwt_depth` (the `dwt_depth_ho` horizontal-only
/// levels leave the height untouched).
pub fn padded_component_dims_ho(
    comp_width: u32,
    comp_height: u32,
    dwt_depth: u32,
    dwt_depth_ho: u32,
) -> (usize, usize) {
    let scale_w: u32 = 1 << (dwt_depth_ho + dwt_depth);
    let scale_h: u32 = 1 << dwt_depth;
    let pw = scale_w * comp_width.div_ceil(scale_w);
    let ph = scale_h * comp_height.div_ceil(scale_h);
    (pw as usize, ph as usize)
}

/// Build a freshly-zeroed 4-band entry for level 0 (LL only; HL/LH/HH
/// are empty placeholders) and for each subsequent level 1..=dwt_depth
/// (where only HL/LH/HH carry data). This matches `initialise_wavelet_data`.
pub fn init_pyramid(comp_width: u32, comp_height: u32, dwt_depth: u32) -> Vec<[SubbandData; 4]> {
    init_pyramid_ho(comp_width, comp_height, dwt_depth, 0)
}

/// Asymmetric-aware pyramid initialisation (§13.2.2
/// `initialize_wavelet_data`). Total level count is
/// `dwt_depth_ho + dwt_depth + 1`:
///
/// * level 0 — the DC band (`LL` symmetric / `L` asymmetric) in slot 0;
/// * levels `1..=dwt_depth_ho` — a single horizontal-only `H` band,
///   stored in slot 3 (the slot [`crate::wavelet::idwt_with_ho`]
///   reads — see its doc-comment for why the quartet shape is kept);
/// * levels `dwt_depth_ho+1..=dwt_depth_ho+dwt_depth` — the usual
///   HL/LH/HH triplet in slots 1..=3.
///
/// With `dwt_depth_ho == 0` this is exactly [`init_pyramid`].
pub fn init_pyramid_ho(
    comp_width: u32,
    comp_height: u32,
    dwt_depth: u32,
    dwt_depth_ho: u32,
) -> Vec<[SubbandData; 4]> {
    let total = dwt_depth_ho + dwt_depth;
    let mut levels: Vec<[SubbandData; 4]> = Vec::with_capacity(total as usize + 1);
    // Level 0: the DC band (LL or L) lives in slot 0.
    let (w0, h0) = subband_dims_ho(comp_width, comp_height, dwt_depth, dwt_depth_ho, 0);
    levels.push([
        SubbandData::new(w0, h0),
        SubbandData::new(0, 0),
        SubbandData::new(0, 0),
        SubbandData::new(0, 0),
    ]);
    // Levels 1..=dwt_depth_ho: single horizontal-only H band (slot 3).
    for level in 1..=dwt_depth_ho {
        let (w, h) = subband_dims_ho(comp_width, comp_height, dwt_depth, dwt_depth_ho, level);
        levels.push([
            SubbandData::new(0, 0),
            SubbandData::new(0, 0),
            SubbandData::new(0, 0),
            SubbandData::new(w, h), // H
        ]);
    }
    // Levels dwt_depth_ho+1..=total: HL/LH/HH triplet.
    for level in dwt_depth_ho + 1..=total {
        let (w, h) = subband_dims_ho(comp_width, comp_height, dwt_depth, dwt_depth_ho, level);
        levels.push([
            SubbandData::new(0, 0), // LL unused above level 0
            SubbandData::new(w, h), // HL
            SubbandData::new(w, h), // LH
            SubbandData::new(w, h), // HH
        ]);
    }
    levels
}

/// The §13.5.3 / §13.5.4 slice-band iteration order shared by the
/// slice packers and unpackers: `(level, slot)` pairs where `slot` is
/// the pyramid quartet index ([`Orient::as_index`] layout) the band
/// occupies.
///
/// * Symmetric (`dwt_depth_ho == 0`): `(0, LL)` then
///   `(level, HL/LH/HH)` for `level = 1..=dwt_depth`.
/// * Asymmetric (`dwt_depth_ho > 0`): `(0, L)` — slot 0, like LL —
///   then `(level, H)` for `level = 1..=dwt_depth_ho` — the single
///   horizontal-only band, stored in slot 3 per [`init_pyramid_ho`] —
///   then `(level, HL/LH/HH)` for
///   `level = dwt_depth_ho+1..=dwt_depth_ho+dwt_depth`.
///
/// [`crate::quant::slice_quantisers`] emits its per-level quantisers
/// in the same slot layout, so a single `(level, orient)` pair indexes
/// both the coefficient band and its effective quantiser.
pub fn slice_band_order(dwt_depth: u32, dwt_depth_ho: u32) -> Vec<(u32, Orient)> {
    let total = dwt_depth_ho + dwt_depth;
    let mut order = Vec::with_capacity(1 + dwt_depth_ho as usize + 3 * dwt_depth as usize);
    // Level 0: LL (symmetric) / L (asymmetric) — slot 0 either way.
    order.push((0, Orient::LL));
    // Horizontal-only H bands — slot 3 (see init_pyramid_ho).
    for level in 1..=dwt_depth_ho {
        order.push((level, Orient::HH));
    }
    for level in dwt_depth_ho + 1..=total {
        for orient in [Orient::HL, Orient::LH, Orient::HH] {
            order.push((level, orient));
        }
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subband_dims_exact_divisor() {
        // depth = 3, 64x64 component: padded = 64x64.
        // Level 0: 64/8 = 8.
        // Level 1: 64/8 = 8.
        // Level 2: 64/4 = 16.
        // Level 3: 64/2 = 32.
        assert_eq!(subband_dims(64, 64, 3, 0), (8, 8));
        assert_eq!(subband_dims(64, 64, 3, 1), (8, 8));
        assert_eq!(subband_dims(64, 64, 3, 2), (16, 16));
        assert_eq!(subband_dims(64, 64, 3, 3), (32, 32));
    }

    #[test]
    fn subband_dims_padded_to_multiple() {
        // 63x63 with depth 3 → padded to 64x64.
        assert_eq!(padded_component_dims(63, 63, 3), (64, 64));
        assert_eq!(subband_dims(63, 63, 3, 3), (32, 32));
    }

    #[test]
    fn init_pyramid_shape() {
        let pyramid = init_pyramid(64, 64, 3);
        assert_eq!(pyramid.len(), 4);
        assert_eq!((pyramid[0][0].width, pyramid[0][0].height), (8, 8));
        assert_eq!((pyramid[1][1].width, pyramid[1][1].height), (8, 8));
        assert_eq!((pyramid[2][1].width, pyramid[2][1].height), (16, 16));
        assert_eq!((pyramid[3][1].width, pyramid[3][1].height), (32, 32));
    }

    /// §13.2.3 asymmetric dims: width pads to `2^(ho+depth)`, height
    /// only to `2^depth`; horizontal-only levels keep the full
    /// `ph / 2^depth` height.
    #[test]
    fn subband_dims_ho_asymmetric() {
        // depth = 1, ho = 2, 64x64: pw = 64 (multiple of 8), ph = 64
        // (multiple of 2).
        // Level 0 (L):  64/8 = 8 wide, 64/2 = 32 tall.
        // Level 1 (H):  64/8 = 8 wide, 32 tall.
        // Level 2 (H):  64/4 = 16 wide, 32 tall.
        // Level 3 (HL/LH/HH): 64/2 = 32 wide, 64/2 = 32 tall.
        assert_eq!(subband_dims_ho(64, 64, 1, 2, 0), (8, 32));
        assert_eq!(subband_dims_ho(64, 64, 1, 2, 1), (8, 32));
        assert_eq!(subband_dims_ho(64, 64, 1, 2, 2), (16, 32));
        assert_eq!(subband_dims_ho(64, 64, 1, 2, 3), (32, 32));
        // Non-multiple input: 60x60, depth = 2, ho = 1 → pw multiple
        // of 8 = 64, ph multiple of 4 = 60.
        assert_eq!(padded_component_dims_ho(60, 60, 2, 1), (64, 60));
        assert_eq!(subband_dims_ho(60, 60, 2, 1, 0), (8, 15));
        assert_eq!(subband_dims_ho(60, 60, 2, 1, 1), (8, 15));
        assert_eq!(subband_dims_ho(60, 60, 2, 1, 2), (16, 15));
        assert_eq!(subband_dims_ho(60, 60, 2, 1, 3), (32, 30));
    }

    /// `subband_dims_ho` with `ho == 0` is bit-for-bit the symmetric
    /// helper.
    #[test]
    fn subband_dims_ho_zero_matches_symmetric() {
        for level in 0..=3 {
            assert_eq!(
                subband_dims_ho(63, 63, 3, 0, level),
                subband_dims(63, 63, 3, level)
            );
        }
    }

    /// Asymmetric pyramid layout: level 0 slot 0, H levels slot 3,
    /// 2-D levels slots 1..=3.
    #[test]
    fn init_pyramid_ho_shape() {
        let py = init_pyramid_ho(64, 64, 1, 2);
        assert_eq!(py.len(), 4);
        assert_eq!((py[0][0].width, py[0][0].height), (8, 32));
        // H levels: slots 0..=2 empty, slot 3 carries the band.
        for level in 1..=2usize {
            assert_eq!(py[level][0].width, 0);
            assert_eq!(py[level][1].width, 0);
            assert_eq!(py[level][2].width, 0);
            assert!(py[level][3].width > 0);
        }
        assert_eq!((py[1][3].width, py[1][3].height), (8, 32));
        assert_eq!((py[2][3].width, py[2][3].height), (16, 32));
        // 2-D level: slots 1..=3 carry HL/LH/HH.
        assert_eq!((py[3][1].width, py[3][1].height), (32, 32));
        assert_eq!((py[3][2].width, py[3][2].height), (32, 32));
        assert_eq!((py[3][3].width, py[3][3].height), (32, 32));
        assert_eq!(py[3][0].width, 0);
    }

    /// §13.5.3 / §13.5.4 band order: L, H × ho, then HL/LH/HH triplets.
    #[test]
    fn slice_band_order_layouts() {
        assert_eq!(
            slice_band_order(2, 0),
            vec![
                (0, Orient::LL),
                (1, Orient::HL),
                (1, Orient::LH),
                (1, Orient::HH),
                (2, Orient::HL),
                (2, Orient::LH),
                (2, Orient::HH),
            ]
        );
        assert_eq!(
            slice_band_order(1, 2),
            vec![
                (0, Orient::LL), // L — slot 0
                (1, Orient::HH), // H — slot 3
                (2, Orient::HH), // H — slot 3
                (3, Orient::HL),
                (3, Orient::LH),
                (3, Orient::HH),
            ]
        );
    }

    #[test]
    fn subband_row_access() {
        let mut sd = SubbandData::new(4, 3);
        sd.set(1, 2, 42);
        assert_eq!(sd.get(1, 2), 42);
        let row = sd.row_mut(1);
        assert_eq!(row, &[0, 0, 42, 0]);
        row[0] = -7;
        assert_eq!(sd.get(1, 0), -7);
    }
}
