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
    let scale: u32 = 1 << dwt_depth;
    let pw = scale * comp_width.div_ceil(scale);
    let ph = scale * comp_height.div_ceil(scale);
    if level == 0 {
        let d = 1u32 << dwt_depth;
        ((pw / d) as usize, (ph / d) as usize)
    } else {
        let d = 1u32 << (dwt_depth - level + 1);
        ((pw / d) as usize, (ph / d) as usize)
    }
}

/// Luma / chroma padded component dimensions `(padded_width,
/// padded_height)` — the dimensions of the final IDWT output before
/// trimming back to the real picture size (§15.7).
pub fn padded_component_dims(comp_width: u32, comp_height: u32, dwt_depth: u32) -> (usize, usize) {
    let scale: u32 = 1 << dwt_depth;
    let pw = scale * comp_width.div_ceil(scale);
    let ph = scale * comp_height.div_ceil(scale);
    (pw as usize, ph as usize)
}

/// Build a freshly-zeroed 4-band entry for level 0 (LL only; HL/LH/HH
/// are empty placeholders) and for each subsequent level 1..=dwt_depth
/// (where only HL/LH/HH carry data). This matches `initialise_wavelet_data`.
pub fn init_pyramid(comp_width: u32, comp_height: u32, dwt_depth: u32) -> Vec<[SubbandData; 4]> {
    let mut levels: Vec<[SubbandData; 4]> = Vec::with_capacity(dwt_depth as usize + 1);
    // Level 0: LL only.
    let (w0, h0) = subband_dims(comp_width, comp_height, dwt_depth, 0);
    levels.push([
        SubbandData::new(w0, h0),
        SubbandData::new(0, 0),
        SubbandData::new(0, 0),
        SubbandData::new(0, 0),
    ]);
    for level in 1..=dwt_depth {
        let (w, h) = subband_dims(comp_width, comp_height, dwt_depth, level);
        levels.push([
            SubbandData::new(0, 0), // LL unused above level 0
            SubbandData::new(w, h), // HL
            SubbandData::new(w, h), // LH
            SubbandData::new(w, h), // HH
        ]);
    }
    levels
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
