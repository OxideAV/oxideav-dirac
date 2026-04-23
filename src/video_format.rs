//! Predefined video format defaults (Annex C).
//!
//! When a sequence header carries a `base_video_format` index, these
//! tables provide the starting values for all source parameters. The
//! stream may override individual fields (e.g. custom frame size) via
//! the per-field flags in §10.3.

/// Chroma sampling format (§10.3.3, Table 10.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaFormat {
    /// 0 — 4:4:4.
    Yuv444,
    /// 1 — 4:2:2.
    Yuv422,
    /// 2 — 4:2:0.
    Yuv420,
}

impl ChromaFormat {
    pub fn from_index(idx: u32) -> Option<Self> {
        match idx {
            0 => Some(Self::Yuv444),
            1 => Some(Self::Yuv422),
            2 => Some(Self::Yuv420),
            _ => None,
        }
    }

    pub fn to_index(self) -> u32 {
        match self {
            Self::Yuv444 => 0,
            Self::Yuv422 => 1,
            Self::Yuv420 => 2,
        }
    }

    /// Horizontal chroma subsampling factor (luma_w / chroma_w).
    pub fn h_ratio(self) -> u32 {
        match self {
            Self::Yuv444 => 1,
            Self::Yuv422 | Self::Yuv420 => 2,
        }
    }

    /// Vertical chroma subsampling factor (luma_h / chroma_h).
    pub fn v_ratio(self) -> u32 {
        match self {
            Self::Yuv444 | Self::Yuv422 => 1,
            Self::Yuv420 => 2,
        }
    }
}

/// Source scan format (§10.3.4). `Progressive = 0`, `Interlaced = 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanFormat {
    Progressive,
    Interlaced,
}

/// Signal range preset (§10.3.8, Table 10.5). Custom ranges are
/// supported too; those carry (offset, excursion) pairs directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignalRange {
    pub luma_offset: u32,
    pub luma_excursion: u32,
    pub chroma_offset: u32,
    pub chroma_excursion: u32,
}

impl SignalRange {
    pub const PRESET_8BIT_FULL: Self = Self {
        luma_offset: 0,
        luma_excursion: 255,
        chroma_offset: 128,
        chroma_excursion: 255,
    };
    pub const PRESET_8BIT_VIDEO: Self = Self {
        luma_offset: 16,
        luma_excursion: 219,
        chroma_offset: 128,
        chroma_excursion: 224,
    };
    pub const PRESET_10BIT_VIDEO: Self = Self {
        luma_offset: 64,
        luma_excursion: 876,
        chroma_offset: 512,
        chroma_excursion: 896,
    };
    pub const PRESET_12BIT_VIDEO: Self = Self {
        luma_offset: 256,
        luma_excursion: 3504,
        chroma_offset: 2048,
        chroma_excursion: 3584,
    };

    pub fn preset(index: u32) -> Option<Self> {
        match index {
            1 => Some(Self::PRESET_8BIT_FULL),
            2 => Some(Self::PRESET_8BIT_VIDEO),
            3 => Some(Self::PRESET_10BIT_VIDEO),
            4 => Some(Self::PRESET_12BIT_VIDEO),
            _ => None,
        }
    }
}

/// Preset frame rate (§10.3.5, Table 10.3). Indices 1..=10.
pub fn preset_frame_rate(index: u32) -> Option<(u32, u32)> {
    match index {
        1 => Some((24000, 1001)),
        2 => Some((24, 1)),
        3 => Some((25, 1)),
        4 => Some((30000, 1001)),
        5 => Some((30, 1)),
        6 => Some((50, 1)),
        7 => Some((60000, 1001)),
        8 => Some((60, 1)),
        9 => Some((15000, 1001)),
        10 => Some((25, 2)),
        _ => None,
    }
}

/// Preset pixel aspect ratio (§10.3.6, Table 10.4). Indices 1..=6.
pub fn preset_pixel_aspect_ratio(index: u32) -> Option<(u32, u32)> {
    match index {
        1 => Some((1, 1)),
        2 => Some((10, 11)),
        3 => Some((12, 11)),
        4 => Some((40, 33)),
        5 => Some((16, 11)),
        6 => Some((4, 3)),
        _ => None,
    }
}

/// Base video format defaults (Annex C, Tables C.1-C.3). Only the
/// fields that are required to decode the bitstream are stored here;
/// colour spec / clean-area metadata is carried along but not used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BaseVideoFormat {
    pub name: &'static str,
    pub frame_width: u32,
    pub frame_height: u32,
    pub chroma_format: ChromaFormat,
    pub source_sampling: ScanFormat,
    pub top_field_first: bool,
    pub frame_rate_numer: u32,
    pub frame_rate_denom: u32,
    pub pixel_aspect_ratio_numer: u32,
    pub pixel_aspect_ratio_denom: u32,
    pub clean_width: u32,
    pub clean_height: u32,
    pub clean_left_offset: u32,
    pub clean_top_offset: u32,
    pub signal_range: SignalRange,
}

impl BaseVideoFormat {
    /// Look up the defaults for `base_video_format` (indices 0..=20).
    ///
    /// The "custom" format (index 0) is returned as a 640x480 4:2:0
    /// stub — as per the spec's note, virtually every field will be
    /// overridden before decode.
    pub fn lookup(index: u32) -> Option<Self> {
        Some(match index {
            0 => Self {
                name: "Custom Format",
                frame_width: 640,
                frame_height: 480,
                chroma_format: ChromaFormat::Yuv420,
                source_sampling: ScanFormat::Progressive,
                top_field_first: false,
                frame_rate_numer: 24000,
                frame_rate_denom: 1001,
                pixel_aspect_ratio_numer: 1,
                pixel_aspect_ratio_denom: 1,
                clean_width: 640,
                clean_height: 480,
                clean_left_offset: 0,
                clean_top_offset: 0,
                signal_range: SignalRange::PRESET_8BIT_FULL,
            },
            1 => Self {
                name: "QSIF525",
                frame_width: 176,
                frame_height: 120,
                chroma_format: ChromaFormat::Yuv420,
                source_sampling: ScanFormat::Progressive,
                top_field_first: false,
                frame_rate_numer: 15000,
                frame_rate_denom: 1001,
                pixel_aspect_ratio_numer: 10,
                pixel_aspect_ratio_denom: 11,
                clean_width: 176,
                clean_height: 120,
                clean_left_offset: 0,
                clean_top_offset: 0,
                signal_range: SignalRange::PRESET_8BIT_FULL,
            },
            2 => Self {
                name: "QCIF",
                frame_width: 176,
                frame_height: 144,
                chroma_format: ChromaFormat::Yuv420,
                source_sampling: ScanFormat::Progressive,
                top_field_first: true,
                frame_rate_numer: 25,
                frame_rate_denom: 2,
                pixel_aspect_ratio_numer: 12,
                pixel_aspect_ratio_denom: 11,
                clean_width: 176,
                clean_height: 144,
                clean_left_offset: 0,
                clean_top_offset: 0,
                signal_range: SignalRange::PRESET_8BIT_FULL,
            },
            3 => Self {
                name: "SIF525",
                frame_width: 352,
                frame_height: 240,
                chroma_format: ChromaFormat::Yuv420,
                source_sampling: ScanFormat::Progressive,
                top_field_first: false,
                frame_rate_numer: 15000,
                frame_rate_denom: 1001,
                pixel_aspect_ratio_numer: 10,
                pixel_aspect_ratio_denom: 11,
                clean_width: 352,
                clean_height: 240,
                clean_left_offset: 0,
                clean_top_offset: 0,
                signal_range: SignalRange::PRESET_8BIT_FULL,
            },
            4 => Self {
                name: "CIF",
                frame_width: 352,
                frame_height: 288,
                chroma_format: ChromaFormat::Yuv420,
                source_sampling: ScanFormat::Progressive,
                top_field_first: true,
                frame_rate_numer: 25,
                frame_rate_denom: 2,
                pixel_aspect_ratio_numer: 12,
                pixel_aspect_ratio_denom: 11,
                clean_width: 352,
                clean_height: 288,
                clean_left_offset: 0,
                clean_top_offset: 0,
                signal_range: SignalRange::PRESET_8BIT_FULL,
            },
            5 => Self {
                name: "4SIF525",
                frame_width: 704,
                frame_height: 480,
                chroma_format: ChromaFormat::Yuv420,
                source_sampling: ScanFormat::Progressive,
                top_field_first: false,
                frame_rate_numer: 15000,
                frame_rate_denom: 1001,
                pixel_aspect_ratio_numer: 10,
                pixel_aspect_ratio_denom: 11,
                clean_width: 704,
                clean_height: 480,
                clean_left_offset: 0,
                clean_top_offset: 0,
                signal_range: SignalRange::PRESET_8BIT_FULL,
            },
            6 => Self {
                name: "4CIF",
                frame_width: 704,
                frame_height: 576,
                chroma_format: ChromaFormat::Yuv420,
                source_sampling: ScanFormat::Progressive,
                top_field_first: true,
                frame_rate_numer: 25,
                frame_rate_denom: 2,
                pixel_aspect_ratio_numer: 12,
                pixel_aspect_ratio_denom: 11,
                clean_width: 704,
                clean_height: 576,
                clean_left_offset: 0,
                clean_top_offset: 0,
                signal_range: SignalRange::PRESET_8BIT_FULL,
            },
            7 => Self {
                name: "SD 480I-60",
                frame_width: 720,
                frame_height: 480,
                chroma_format: ChromaFormat::Yuv422,
                source_sampling: ScanFormat::Interlaced,
                top_field_first: false,
                frame_rate_numer: 30000,
                frame_rate_denom: 1001,
                pixel_aspect_ratio_numer: 10,
                pixel_aspect_ratio_denom: 11,
                clean_width: 704,
                clean_height: 480,
                clean_left_offset: 8,
                clean_top_offset: 0,
                signal_range: SignalRange::PRESET_10BIT_VIDEO,
            },
            8 => Self {
                name: "SD 576I-50",
                frame_width: 720,
                frame_height: 576,
                chroma_format: ChromaFormat::Yuv422,
                source_sampling: ScanFormat::Interlaced,
                top_field_first: true,
                frame_rate_numer: 25,
                frame_rate_denom: 1,
                pixel_aspect_ratio_numer: 12,
                pixel_aspect_ratio_denom: 11,
                clean_width: 704,
                clean_height: 576,
                clean_left_offset: 8,
                clean_top_offset: 0,
                signal_range: SignalRange::PRESET_10BIT_VIDEO,
            },
            9 => Self {
                name: "HD 720P-60",
                frame_width: 1280,
                frame_height: 720,
                chroma_format: ChromaFormat::Yuv422,
                source_sampling: ScanFormat::Progressive,
                top_field_first: true,
                frame_rate_numer: 60000,
                frame_rate_denom: 1001,
                pixel_aspect_ratio_numer: 1,
                pixel_aspect_ratio_denom: 1,
                clean_width: 1280,
                clean_height: 720,
                clean_left_offset: 0,
                clean_top_offset: 0,
                signal_range: SignalRange::PRESET_10BIT_VIDEO,
            },
            10 => Self {
                name: "HD 720P-50",
                frame_width: 1280,
                frame_height: 720,
                chroma_format: ChromaFormat::Yuv422,
                source_sampling: ScanFormat::Progressive,
                top_field_first: false,
                frame_rate_numer: 50,
                frame_rate_denom: 1,
                pixel_aspect_ratio_numer: 1,
                pixel_aspect_ratio_denom: 1,
                clean_width: 1280,
                clean_height: 720,
                clean_left_offset: 0,
                clean_top_offset: 0,
                signal_range: SignalRange::PRESET_10BIT_VIDEO,
            },
            11 => Self {
                name: "HD 1080I-60",
                frame_width: 1920,
                frame_height: 1080,
                chroma_format: ChromaFormat::Yuv422,
                source_sampling: ScanFormat::Interlaced,
                top_field_first: true,
                frame_rate_numer: 30000,
                frame_rate_denom: 1001,
                pixel_aspect_ratio_numer: 1,
                pixel_aspect_ratio_denom: 1,
                clean_width: 1920,
                clean_height: 1080,
                clean_left_offset: 0,
                clean_top_offset: 0,
                signal_range: SignalRange::PRESET_10BIT_VIDEO,
            },
            12 => Self {
                name: "HD 1080I-50",
                frame_width: 1920,
                frame_height: 1080,
                chroma_format: ChromaFormat::Yuv422,
                source_sampling: ScanFormat::Interlaced,
                top_field_first: true,
                frame_rate_numer: 25,
                frame_rate_denom: 1,
                pixel_aspect_ratio_numer: 1,
                pixel_aspect_ratio_denom: 1,
                clean_width: 1920,
                clean_height: 1080,
                clean_left_offset: 0,
                clean_top_offset: 0,
                signal_range: SignalRange::PRESET_10BIT_VIDEO,
            },
            13 => Self {
                name: "HD 1080P-60",
                frame_width: 1920,
                frame_height: 1080,
                chroma_format: ChromaFormat::Yuv422,
                source_sampling: ScanFormat::Progressive,
                top_field_first: false,
                frame_rate_numer: 60000,
                frame_rate_denom: 1001,
                pixel_aspect_ratio_numer: 1,
                pixel_aspect_ratio_denom: 1,
                clean_width: 1920,
                clean_height: 1080,
                clean_left_offset: 0,
                clean_top_offset: 0,
                signal_range: SignalRange::PRESET_10BIT_VIDEO,
            },
            14 => Self {
                name: "HD 1080P-50",
                frame_width: 1920,
                frame_height: 1080,
                chroma_format: ChromaFormat::Yuv422,
                source_sampling: ScanFormat::Progressive,
                top_field_first: false,
                frame_rate_numer: 50,
                frame_rate_denom: 1,
                pixel_aspect_ratio_numer: 1,
                pixel_aspect_ratio_denom: 1,
                clean_width: 1920,
                clean_height: 1080,
                clean_left_offset: 0,
                clean_top_offset: 0,
                signal_range: SignalRange::PRESET_10BIT_VIDEO,
            },
            15 => Self {
                name: "DC 2K-24",
                frame_width: 2048,
                frame_height: 1080,
                chroma_format: ChromaFormat::Yuv444,
                source_sampling: ScanFormat::Progressive,
                top_field_first: true,
                frame_rate_numer: 24,
                frame_rate_denom: 1,
                pixel_aspect_ratio_numer: 1,
                pixel_aspect_ratio_denom: 1,
                clean_width: 2048,
                clean_height: 1080,
                clean_left_offset: 0,
                clean_top_offset: 0,
                signal_range: SignalRange::PRESET_12BIT_VIDEO,
            },
            16 => Self {
                name: "DC 4K-24",
                frame_width: 4096,
                frame_height: 2160,
                chroma_format: ChromaFormat::Yuv444,
                source_sampling: ScanFormat::Progressive,
                top_field_first: true,
                frame_rate_numer: 24,
                frame_rate_denom: 1,
                pixel_aspect_ratio_numer: 1,
                pixel_aspect_ratio_denom: 1,
                clean_width: 4096,
                clean_height: 2160,
                clean_left_offset: 0,
                clean_top_offset: 0,
                signal_range: SignalRange::PRESET_12BIT_VIDEO,
            },
            17 => Self {
                name: "UHDTV 4K-60",
                frame_width: 3840,
                frame_height: 2160,
                chroma_format: ChromaFormat::Yuv422,
                source_sampling: ScanFormat::Progressive,
                top_field_first: false,
                frame_rate_numer: 60000,
                frame_rate_denom: 1001,
                pixel_aspect_ratio_numer: 1,
                pixel_aspect_ratio_denom: 1,
                clean_width: 3840,
                clean_height: 2160,
                clean_left_offset: 0,
                clean_top_offset: 0,
                signal_range: SignalRange::PRESET_10BIT_VIDEO,
            },
            18 => Self {
                name: "UHDTV 4K-50",
                frame_width: 3840,
                frame_height: 2160,
                chroma_format: ChromaFormat::Yuv422,
                source_sampling: ScanFormat::Progressive,
                top_field_first: true,
                frame_rate_numer: 50,
                frame_rate_denom: 1,
                pixel_aspect_ratio_numer: 1,
                pixel_aspect_ratio_denom: 1,
                clean_width: 3840,
                clean_height: 2160,
                clean_left_offset: 0,
                clean_top_offset: 0,
                signal_range: SignalRange::PRESET_10BIT_VIDEO,
            },
            19 => Self {
                name: "UHDTV 8K-60",
                frame_width: 7680,
                frame_height: 4320,
                chroma_format: ChromaFormat::Yuv422,
                source_sampling: ScanFormat::Progressive,
                top_field_first: false,
                frame_rate_numer: 60000,
                frame_rate_denom: 1001,
                pixel_aspect_ratio_numer: 1,
                pixel_aspect_ratio_denom: 1,
                clean_width: 7680,
                clean_height: 4320,
                clean_left_offset: 0,
                clean_top_offset: 0,
                signal_range: SignalRange::PRESET_10BIT_VIDEO,
            },
            20 => Self {
                name: "UHDTV 8K-50",
                frame_width: 7680,
                frame_height: 4320,
                chroma_format: ChromaFormat::Yuv422,
                source_sampling: ScanFormat::Progressive,
                top_field_first: true,
                frame_rate_numer: 50,
                frame_rate_denom: 1,
                pixel_aspect_ratio_numer: 1,
                pixel_aspect_ratio_denom: 1,
                clean_width: 7680,
                clean_height: 4320,
                clean_left_offset: 0,
                clean_top_offset: 0,
                signal_range: SignalRange::PRESET_10BIT_VIDEO,
            },
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cif_defaults() {
        let f = BaseVideoFormat::lookup(4).unwrap();
        assert_eq!((f.frame_width, f.frame_height), (352, 288));
        assert_eq!(f.chroma_format, ChromaFormat::Yuv420);
        assert_eq!(f.source_sampling, ScanFormat::Progressive);
    }

    #[test]
    fn chroma_subsample_ratios() {
        assert_eq!(ChromaFormat::Yuv444.h_ratio(), 1);
        assert_eq!(ChromaFormat::Yuv444.v_ratio(), 1);
        assert_eq!(ChromaFormat::Yuv422.h_ratio(), 2);
        assert_eq!(ChromaFormat::Yuv422.v_ratio(), 1);
        assert_eq!(ChromaFormat::Yuv420.h_ratio(), 2);
        assert_eq!(ChromaFormat::Yuv420.v_ratio(), 2);
    }

    #[test]
    fn signal_range_presets() {
        assert_eq!(
            SignalRange::preset(2).unwrap(),
            SignalRange::PRESET_8BIT_VIDEO
        );
        assert!(SignalRange::preset(99).is_none());
    }
}
