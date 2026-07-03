//! VC-2 v3 fragment-header parser (SMPTE ST 2042-1:2022 §14.2).
//!
//! Encoded picture data in a VC-2 v3 stream can optionally be broken
//! into smaller **fragment** data units. Each fragmented picture is
//! coded as one **setup fragment** (carrying the picture's transform
//! parameters and zero slices) followed by one or more **data
//! fragments** (each carrying a non-zero slice count, in raster order).
//!
//! Fragments are signalled by the v3 picture-fragment parse codes
//! `0xCC` (Low Delay Picture Fragment) and `0xEC` (High Quality Picture
//! Fragment) in Table 4 of §10.5.2. The v3 §10.5.2 Table 5 predicate
//! `is_fragment(state) := (parse_code & 0x0C) == 0x0C` keys fragment
//! recognition off the parse-code bit pattern alone.
//!
//! NOTE on `0xCC` / `0xEC` ambiguity with earlier specs: the BBC Dirac
//! v2.2.3 spec (`docs/video/dirac/dirac-spec-latest.pdf`, Table 9.1)
//! used `0xCC` for "Low-delay Intra Reference Picture" and `0xEC` is
//! unmentioned. The VC-2 v3 spec (SMPTE ST 2042-1:2022) reassigns both
//! codes to picture fragments. Whether a given byte means
//! "intra reference" or "fragment" depends on the active stream
//! version. This module is the v3 syntactic layer; it parses the
//! fragment header but does not gate on the stream's `major_version`
//! itself. The dispatcher in [`crate::decoder`] / [`crate::picture`] is
//! responsible for that version-aware routing decision.
//!
//! # Fragment header layout (§14.2)
//!
//! The spec's pseudocode is:
//!
//! ```text
//! fragment_header(state):
//!     state[picture_number] = read_uint_lit(state, 4)
//!     state[fragment_data_length] = read_uint_lit(state, 2)
//!     state[fragment_slice_count] = read_uint_lit(state, 2)
//!     if (state[fragment_slice_count] != 0):
//!         state[fragment_x_offset] = read_uint_lit(state, 2)
//!         state[fragment_y_offset] = read_uint_lit(state, 2)
//! ```
//!
//! Annex A.3.4 defines `read_uint_lit(state, n)` as `read_nbits(state,
//! 8 * n)`, i.e. an `n`-byte unsigned integer literal read big-endian.
//! The fragment header therefore has a **fixed byte layout**:
//!
//! | Field                  | Bytes | Type | Notes                              |
//! |------------------------|-------|------|------------------------------------|
//! | `picture_number`       | 4     | u32  | Must match across setup + data     |
//! |                        |       |      | fragments for the same picture.    |
//! | `fragment_data_length` | 2     | u16  | "Contains undefined data" per      |
//! |                        |       |      | §14.2; carried for stream layout   |
//! |                        |       |      | tooling and not used by decode.    |
//! | `fragment_slice_count` | 2     | u16  | 0 = setup fragment (transform      |
//! |                        |       |      | parameters follow). Non-zero =     |
//! |                        |       |      | data fragment with that many       |
//! |                        |       |      | slices.                            |
//! | `fragment_x_offset`    | 2     | u16  | Data fragments only. First slice's |
//! |                        |       |      | x coordinate (raster scan).        |
//! | `fragment_y_offset`    | 2     | u16  | Data fragments only.               |
//!
//! A setup fragment is therefore **8 bytes** of header; a data fragment
//! is **12 bytes**. The fragment header is byte-aligned by construction
//! (it immediately follows the byte-aligned 13-byte parse info header
//! per §10.5.1) so no extra `byte_align` is needed here.
//!
//! Per §14.1, between a setup fragment and its associated data
//! fragments the spec forbids further setup fragments or unfragmented
//! picture units until the picture is complete. This module does not
//! enforce that sequencing constraint — it is purely a per-unit
//! header parser; sequencing belongs to the dispatcher.

use crate::bits::BitReader;
use crate::parse_info::ParseInfo;
use crate::picture::{
    decode_hq_slice, decode_ld_slice, intra_dc_prediction, low_delay_profile_for,
    parse_transform_parameters, trim_clip_offset, DecodedPicture, LowDelayProfile, PictureError,
    TransformParameters,
};
use crate::sequence::SequenceHeader;
use crate::subband::{init_pyramid_ho, subband_dims_ho, SubbandData};
use crate::wavelet::idwt_with_ho;

/// A picture fragment header, parsed per §14.2.
///
/// The variant pins whether the fragment is a setup fragment (carrying
/// transform parameters and zero slices) or a data fragment (carrying
/// at least one slice and the raster offset of the first one).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FragmentHeader {
    /// §14.2 `picture_number`. Increments by one for each successive
    /// setup fragment (i.e. for each new picture) and wraps at
    /// 2^32 - 1; data fragments share the picture number of their
    /// associated setup fragment.
    pub picture_number: u32,
    /// §14.2 `fragment_data_length`. The spec explicitly states this
    /// field "contains undefined data for the purposes of this standard
    /// and does not contribute to the decoding process." Preserved here
    /// for round-trip tooling and stream-layout introspection only.
    pub fragment_data_length: u16,
    /// Either `Setup` (fragment carrying transform parameters; slice
    /// count is zero per §14.1) or `Data { ... }` (fragment carrying
    /// `slice_count` consecutive slices starting at `(x_offset,
    /// y_offset)` in the raster scan).
    pub kind: FragmentKind,
}

/// Distinguishes the two §14.1 fragment categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FragmentKind {
    /// A setup fragment: `fragment_slice_count == 0`, carries the
    /// picture's transform parameters (§12.4) immediately after the
    /// 8-byte header and must precede every associated data fragment.
    Setup,
    /// A data fragment: `fragment_slice_count > 0`, carries that many
    /// consecutive slices in raster scan starting at `(x_offset,
    /// y_offset)`.
    Data {
        /// Number of slices carried by this fragment. Always > 0.
        slice_count: u16,
        /// First slice's x coordinate (column) in raster scan.
        x_offset: u16,
        /// First slice's y coordinate (row) in raster scan.
        y_offset: u16,
    },
}

/// Errors raised by the fragment-header parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FragmentError {
    /// Payload is shorter than the minimum 8-byte setup-fragment
    /// header. The `needed` field is the byte count the parser
    /// expected to read (8 for a setup fragment, 12 for a data
    /// fragment after the slice-count field has been peeked).
    Truncated { needed: usize, available: usize },
}

impl core::fmt::Display for FragmentError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated { needed, available } => write!(
                f,
                "fragment header truncated: needed {needed} bytes, have {available}"
            ),
        }
    }
}

impl std::error::Error for FragmentError {}

impl FragmentHeader {
    /// Minimum byte width of a fragment header: 4 + 2 + 2 = 8 bytes
    /// (the setup-fragment case where `fragment_slice_count == 0`).
    pub const MIN_SIZE: usize = 8;

    /// Byte width of a data-fragment header: 4 + 2 + 2 + 2 + 2 = 12
    /// bytes (the case where `fragment_slice_count != 0`).
    pub const DATA_SIZE: usize = 12;

    /// Parse a fragment header from the start of `payload`. The
    /// payload is the parse-info-relative byte slice (i.e. the bytes
    /// strictly after the 13-byte parse info header), exactly as
    /// produced by [`crate::stream::DataUnitIter`].
    ///
    /// Returns the parsed header on success. Returns
    /// [`FragmentError::Truncated`] if the slice is shorter than the
    /// 8-byte setup-fragment minimum, or shorter than the 12-byte data
    /// fragment width once a non-zero slice count has been read.
    pub fn parse(payload: &[u8]) -> Result<Self, FragmentError> {
        if payload.len() < Self::MIN_SIZE {
            return Err(FragmentError::Truncated {
                needed: Self::MIN_SIZE,
                available: payload.len(),
            });
        }
        let picture_number = u32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]);
        let fragment_data_length = u16::from_be_bytes([payload[4], payload[5]]);
        let fragment_slice_count = u16::from_be_bytes([payload[6], payload[7]]);
        let kind = if fragment_slice_count == 0 {
            FragmentKind::Setup
        } else {
            if payload.len() < Self::DATA_SIZE {
                return Err(FragmentError::Truncated {
                    needed: Self::DATA_SIZE,
                    available: payload.len(),
                });
            }
            let x_offset = u16::from_be_bytes([payload[8], payload[9]]);
            let y_offset = u16::from_be_bytes([payload[10], payload[11]]);
            FragmentKind::Data {
                slice_count: fragment_slice_count,
                x_offset,
                y_offset,
            }
        };
        Ok(Self {
            picture_number,
            fragment_data_length,
            kind,
        })
    }

    /// Byte width of this header — 8 if [`FragmentKind::Setup`], 12
    /// otherwise. Useful when the caller needs to skip past the header
    /// to reach the following transform-parameters or slice payload.
    pub fn header_size(&self) -> usize {
        match self.kind {
            FragmentKind::Setup => Self::MIN_SIZE,
            FragmentKind::Data { .. } => Self::DATA_SIZE,
        }
    }
}

/// VC-2 v3 §14.4 raster-scan slice coordinate computation.
///
/// Given a data fragment's `(x_offset, y_offset)` (the §14.2 raster
/// coordinate of the fragment's first slice) and the picture's
/// `slices_x` (slice columns), return the `(slice_x, slice_y)` of the
/// `s`-th slice carried by this fragment (`s = 0..slice_count`).
///
/// The §14.4 pseudocode is:
///
/// ```text
/// slice_x = (state[fragment_y_offset] * state[slices_x] +
///            state[fragment_x_offset] + s) % state[slices_x]
/// slice_y = (state[fragment_y_offset] * state[slices_x] +
///            state[fragment_x_offset] + s) // state[slices_x]
/// ```
///
/// That is, the raster index of the s-th slice is
/// `y_offset * slices_x + x_offset + s`, then split back into
/// `(col, row)` modulo `slices_x`. Slices are emitted in raster scan
/// starting at slice (0, 0) per §14.2; a data fragment may straddle a
/// row boundary so the modulo / integer-division pair is what handles
/// the wrap.
///
/// Returns `None` if `slices_x == 0` (an out-of-spec picture geometry —
/// the picture would carry no slices at all).
pub fn slice_coords(s: u32, x_offset: u16, y_offset: u16, slices_x: u32) -> Option<(u32, u32)> {
    if slices_x == 0 {
        return None;
    }
    let raster = u64::from(y_offset) * u64::from(slices_x) + u64::from(x_offset) + u64::from(s);
    let slice_x = (raster % u64::from(slices_x)) as u32;
    let slice_y = (raster / u64::from(slices_x)) as u32;
    Some((slice_x, slice_y))
}

/// VC-2 v3 §14.3 / §14.4 fragmented-picture assembler.
///
/// One [`FragmentAssembler`] tracks the reconstruction state of a
/// single fragmented picture. The driver feeds it a sequence of
/// `(parsed) FragmentHeader`s and the assembler returns either:
///
/// * `FragmentEvent::SetupAccepted` — the setup fragment for a new
///   picture; the caller now parses `transform_parameters` (§12.4)
///   from the bytes after the fragment header and calls
///   [`FragmentAssembler::on_transform_parameters`].
/// * `FragmentEvent::DataSlices { coords, picture_done }` — the data
///   fragment's slices, each at a raster `(slice_x, slice_y)` per
///   §14.4. When `picture_done` is `true`, the caller is responsible
///   for the §14.4 trailing `dc_prediction` kick on the LL (or L)
///   subbands per [`FragmentAssembler::using_dc_prediction`] /
///   [`FragmentAssembler::dwt_depth_ho`].
///
/// The assembler enforces the §14.1 sequencing constraint that data
/// fragments must follow a setup fragment, all share the setup's
/// picture number, no further setup fragments arrive until the
/// picture is complete, and exactly `slices_x * slices_y` slices are
/// received in total.
#[derive(Debug, Clone)]
pub struct FragmentAssembler {
    /// `slices_x` from §13.5.6 (transform parameters → slices). Set
    /// after the setup fragment's `transform_parameters` parse.
    slices_x: u32,
    /// `slices_y` from §13.5.6.
    slices_y: u32,
    /// `dwt_depth_ho` from §12.4.4.3 (defaults to 0 on a symmetric
    /// transform). Captured at setup-fragment time for the §14.4
    /// trailing `dc_prediction` LL-vs-L subband selection.
    dwt_depth_ho: u32,
    /// `using_dc_prediction(state)` cache (§10.5.2 Table 5;
    /// `(parse_code & 0x28) == 0x08`). Captured from the setup
    /// fragment's parse code.
    using_dc_prediction: bool,
    /// Setup-fragment picture number — every data fragment for this
    /// picture must carry the same value (§14.2).
    picture_number: u32,
    /// `state[fragment_slices_received]` (§14.3) — count of slices
    /// fed in across all data fragments for this picture so far.
    slices_received: u32,
    /// Internal phase tracker — has the setup fragment's transform
    /// parameters been ingested yet?
    setup_state: SetupState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SetupState {
    /// `FragmentAssembler` exists but no fragment has been ingested
    /// yet (the assembler's `slices_x` / `slices_y` are placeholder
    /// zeros; the driver is expected to call
    /// [`FragmentAssembler::on_setup_fragment`] first).
    AwaitingSetup,
    /// Setup fragment was accepted but its transform parameters have
    /// not yet been ingested via
    /// [`FragmentAssembler::on_transform_parameters`]. Data fragments
    /// arriving in this phase are a §14.1 violation.
    AwaitingTransformParameters,
    /// Setup is complete (`slices_x` / `slices_y` known); data
    /// fragments may now be ingested.
    ReceivingData,
}

/// Driver-visible result of feeding one fragment header into the
/// assembler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FragmentEvent {
    /// The setup fragment was accepted; the driver now parses the
    /// `transform_parameters` (§12.4) immediately following the
    /// fragment header and feeds the resulting
    /// `(slices_x, slices_y, dwt_depth_ho)` triple back via
    /// [`FragmentAssembler::on_transform_parameters`].
    SetupAccepted,
    /// The data fragment delivered `coords.len()` slices, each at
    /// the listed raster `(slice_x, slice_y)` coordinate.
    /// `picture_done` is `true` when the cumulative
    /// `state[fragment_slices_received]` has reached
    /// `slices_x * slices_y` (§14.4) — the caller then runs the
    /// trailing DC-prediction kick on the LL (or L) subbands per
    /// §14.4.
    DataSlices {
        /// Per-slice raster coordinates in the order they appear in
        /// this fragment.
        coords: Vec<(u32, u32)>,
        /// Picture completion flag (§14.4
        /// `state[fragmented_picture_done] = True`).
        picture_done: bool,
    },
}

/// Errors raised by the fragmented-picture state machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssemblerError {
    /// A data fragment arrived without a preceding setup fragment, or
    /// arrived after the setup fragment but before its transform
    /// parameters were ingested. Either is a §14.1 sequencing
    /// violation.
    UnexpectedDataFragment,
    /// A setup fragment arrived while a previous fragmented picture
    /// was still incomplete (§14.1 forbids this).
    SetupBeforePreviousPictureComplete,
    /// A data fragment's `picture_number` did not match the active
    /// setup fragment's (§14.2 requires equality).
    PictureNumberMismatch { setup: u32, data: u32 },
    /// `using_dc_prediction(parse_code)` returned `true` on a setup
    /// fragment but later returned `false` on its associated data
    /// fragment, or vice versa. The §14.4 DC-prediction kick is keyed
    /// off the picture's parse code, so the parse code must be
    /// consistent across all fragments of the same picture.
    InconsistentParseCode { setup: u8, data: u8 },
    /// The data fragment would push the cumulative slice count past
    /// the picture's `slices_x * slices_y` total. §14.4 explicitly
    /// forbids this ("Slices shall not be omitted or repeated").
    SliceOverflow {
        expected_total: u32,
        slices_received: u32,
        slice_count: u32,
    },
    /// `slices_x == 0` — an out-of-spec transform parameter set.
    /// §13.5.6 requires at least one slice column.
    InvalidSliceGrid { slices_x: u32, slices_y: u32 },
    /// The §14.5 trailing `dc_prediction(...)` kick was requested
    /// before every fragment for the active picture had been
    /// ingested (i.e. before `fragmented_picture_done()` would
    /// return `true`). §14.5 keys the kick off the §14.4
    /// `state[fragmented_picture_done]` flag so a partial picture
    /// must not invoke the trailing prediction step.
    DcPredictionBeforePictureComplete,
    /// Historical variant — **no longer raised**. Earlier rounds
    /// rejected the §14.4 trailing kick on asymmetric pictures
    /// (`dwt_depth_ho > 0`, §12.4.4.3); the §14.4 else-branch in
    /// fact targets the level-0 **L** subband, which lives in the
    /// same `[0][0]` pyramid slot as the symmetric LL band, so the
    /// kick is identical and now always succeeds. The variant is
    /// retained so downstream `match` arms keep compiling.
    AsymmetricDcPredictionUnsupported { dwt_depth_ho: u32 },
}

impl core::fmt::Display for AssemblerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnexpectedDataFragment => write!(
                f,
                "data fragment received before its setup fragment / \
                 transform parameters"
            ),
            Self::SetupBeforePreviousPictureComplete => write!(
                f,
                "setup fragment received while the previous \
                 fragmented picture was still incomplete"
            ),
            Self::PictureNumberMismatch { setup, data } => write!(
                f,
                "fragment picture_number mismatch: setup {setup}, \
                 data {data}"
            ),
            Self::InconsistentParseCode { setup, data } => write!(
                f,
                "fragment parse code mismatch within one picture: \
                 setup 0x{setup:02X}, data 0x{data:02X}"
            ),
            Self::SliceOverflow {
                expected_total,
                slices_received,
                slice_count,
            } => write!(
                f,
                "fragment slice count {slice_count} would push the \
                 cumulative slice count past the picture total \
                 ({slices_received} received, {expected_total} \
                 expected)"
            ),
            Self::InvalidSliceGrid { slices_x, slices_y } => write!(
                f,
                "invalid slice grid: slices_x={slices_x}, \
                 slices_y={slices_y}"
            ),
            Self::DcPredictionBeforePictureComplete => write!(
                f,
                "§14.5 dc_prediction requested before \
                 fragmented_picture_done"
            ),
            Self::AsymmetricDcPredictionUnsupported { dwt_depth_ho } => write!(
                f,
                "§14.5 trailing dc_prediction on asymmetric \
                 transform (dwt_depth_ho={dwt_depth_ho}) not yet \
                 implemented"
            ),
        }
    }
}

impl std::error::Error for AssemblerError {}

impl Default for FragmentAssembler {
    fn default() -> Self {
        Self::new()
    }
}

impl FragmentAssembler {
    /// Build an empty assembler, awaiting its first setup fragment.
    pub fn new() -> Self {
        Self {
            slices_x: 0,
            slices_y: 0,
            dwt_depth_ho: 0,
            using_dc_prediction: false,
            picture_number: 0,
            slices_received: 0,
            setup_state: SetupState::AwaitingSetup,
        }
    }

    /// `state[slices_x]` (§13.5.6) as captured at setup-fragment time.
    /// Zero before the first setup fragment is ingested.
    pub fn slices_x(&self) -> u32 {
        self.slices_x
    }

    /// `state[slices_y]` (§13.5.6).
    pub fn slices_y(&self) -> u32 {
        self.slices_y
    }

    /// `state[dwt_depth_ho]` (§12.4.4.3) as captured at setup-fragment
    /// time; used by the §14.4 trailing DC-prediction kick to choose
    /// between the LL and L subbands.
    pub fn dwt_depth_ho(&self) -> u32 {
        self.dwt_depth_ho
    }

    /// `using_dc_prediction(state)` per §10.5.2 Table 5; captured
    /// from the setup fragment's parse code.
    pub fn using_dc_prediction(&self) -> bool {
        self.using_dc_prediction
    }

    /// Active picture's `picture_number` per §14.2 — zero before the
    /// first setup fragment is ingested.
    pub fn picture_number(&self) -> u32 {
        self.picture_number
    }

    /// Cumulative `state[fragment_slices_received]` per §14.3.
    pub fn slices_received(&self) -> u32 {
        self.slices_received
    }

    /// `state[fragmented_picture_done]` per §14.4 — `true` once
    /// `slices_received == slices_x * slices_y` for the active
    /// picture, ready for the trailing `dc_prediction` kick.
    pub fn fragmented_picture_done(&self) -> bool {
        self.slices_x != 0
            && self.slices_received == self.slices_x.saturating_mul(self.slices_y)
            && self.setup_state == SetupState::ReceivingData
    }

    /// Ingest a setup fragment header (§14.1 "fragment_slice_count
    /// == 0"). The driver passes the setup fragment's parse code so
    /// the §10.5.2 Table 5 `using_dc_prediction` predicate can be
    /// captured for the §14.4 trailing kick.
    ///
    /// Returns [`FragmentEvent::SetupAccepted`] on success. The
    /// caller then parses the immediately-following
    /// `transform_parameters` payload and calls
    /// [`Self::on_transform_parameters`].
    ///
    /// Errors:
    /// * `SetupBeforePreviousPictureComplete` if the previous
    ///   picture's `fragmented_picture_done` had not yet fired.
    pub fn on_setup_fragment(
        &mut self,
        header: &FragmentHeader,
        parse_code: u8,
    ) -> Result<FragmentEvent, AssemblerError> {
        debug_assert!(matches!(header.kind, FragmentKind::Setup));
        // §14.1: "A setup fragment shall not be followed by any
        // further setup fragments ... until the fragmented picture is
        // complete." Allow a setup only if we are at AwaitingSetup
        // (first picture) or the previous picture completed.
        match self.setup_state {
            SetupState::AwaitingSetup => {}
            SetupState::ReceivingData if self.fragmented_picture_done() => {}
            _ => return Err(AssemblerError::SetupBeforePreviousPictureComplete),
        }
        // Reset picture-scope state. `slices_x` / `slices_y` /
        // `dwt_depth_ho` carry over from the previous setup fragment
        // and stay placeholder until the transform parameters arrive
        // for the new one.
        self.picture_number = header.picture_number;
        self.using_dc_prediction = (parse_code & 0x28) == 0x08;
        self.slices_received = 0;
        self.slices_x = 0;
        self.slices_y = 0;
        self.dwt_depth_ho = 0;
        self.setup_state = SetupState::AwaitingTransformParameters;
        Ok(FragmentEvent::SetupAccepted)
    }

    /// Ingest the transform parameters parsed from the bytes
    /// immediately after the setup fragment header (§12.4). The
    /// caller passes the resulting `(slices_x, slices_y,
    /// dwt_depth_ho)` triple; the assembler stores them on
    /// `state[slices_x]` / `state[slices_y]` / `state[dwt_depth_ho]`
    /// and transitions to the `ReceivingData` phase ready to accept
    /// data fragments.
    pub fn on_transform_parameters(
        &mut self,
        slices_x: u32,
        slices_y: u32,
        dwt_depth_ho: u32,
    ) -> Result<(), AssemblerError> {
        if slices_x == 0 || slices_y == 0 {
            return Err(AssemblerError::InvalidSliceGrid { slices_x, slices_y });
        }
        self.slices_x = slices_x;
        self.slices_y = slices_y;
        self.dwt_depth_ho = dwt_depth_ho;
        self.setup_state = SetupState::ReceivingData;
        Ok(())
    }

    /// Ingest a data fragment header (§14.1 "fragment_slice_count
    /// greater than zero") and emit its per-slice raster
    /// `(slice_x, slice_y)` coordinates per §14.4. The driver passes
    /// the data fragment's parse code so the assembler can pin
    /// parse-code consistency against the active setup fragment.
    pub fn on_data_fragment(
        &mut self,
        header: &FragmentHeader,
        parse_code: u8,
    ) -> Result<FragmentEvent, AssemblerError> {
        let (slice_count, x_offset, y_offset) = match header.kind {
            FragmentKind::Data {
                slice_count,
                x_offset,
                y_offset,
            } => (slice_count, x_offset, y_offset),
            FragmentKind::Setup => {
                debug_assert!(false, "data fragment expected, got setup");
                return Err(AssemblerError::UnexpectedDataFragment);
            }
        };
        if self.setup_state != SetupState::ReceivingData {
            return Err(AssemblerError::UnexpectedDataFragment);
        }
        if header.picture_number != self.picture_number {
            return Err(AssemblerError::PictureNumberMismatch {
                setup: self.picture_number,
                data: header.picture_number,
            });
        }
        let data_using_dc = (parse_code & 0x28) == 0x08;
        if data_using_dc != self.using_dc_prediction {
            // Reconstruct an indicative setup-code byte for the error
            // payload (the assembler only stores the predicate
            // outcome, not the original byte). Use a canonical
            // representative: 0x88 == LD picture, 0xE8 == HQ picture.
            let setup_indicative = if self.using_dc_prediction { 0x88 } else { 0xE8 };
            return Err(AssemblerError::InconsistentParseCode {
                setup: setup_indicative,
                data: parse_code,
            });
        }
        // §14.4 picture total — guard against overflow.
        let expected_total = self.slices_x.saturating_mul(self.slices_y);
        let new_total = self.slices_received.saturating_add(u32::from(slice_count));
        if new_total > expected_total {
            return Err(AssemblerError::SliceOverflow {
                expected_total,
                slices_received: self.slices_received,
                slice_count: u32::from(slice_count),
            });
        }
        // §14.4 raster coordinate sweep.
        let mut coords = Vec::with_capacity(usize::from(slice_count));
        for s in 0..u32::from(slice_count) {
            let (sx, sy) = slice_coords(s, x_offset, y_offset, self.slices_x).ok_or(
                AssemblerError::InvalidSliceGrid {
                    slices_x: self.slices_x,
                    slices_y: self.slices_y,
                },
            )?;
            coords.push((sx, sy));
        }
        self.slices_received = new_total;
        let picture_done = self.slices_received == expected_total;
        Ok(FragmentEvent::DataSlices {
            coords,
            picture_done,
        })
    }

    /// VC-2 v3 §14.5 trailing `dc_prediction(...)` kick.
    ///
    /// Once `fragmented_picture_done()` returns `true` (i.e. all
    /// `slices_x * slices_y` slices for the active picture have been
    /// ingested via [`Self::on_data_fragment`]), the v3 §14.5
    /// `fragmented_wavelet_transform()` step runs `dc_prediction(...)`
    /// on **each component's** level-0 LL subband — but only when
    /// `using_dc_prediction()` is true (the LD path, parse codes
    /// `0xC8` / `0xCC`). For an HQ picture (`0xE8` / `0xEC`) the kick
    /// is skipped entirely; the assembler returns `Ok(())` without
    /// touching the subbands.
    ///
    /// The §13.4 raster prediction step is the exact same routine
    /// the non-fragmented LD path (`0xC8`) runs after coefficient
    /// unpack but before the IDWT — the v3 design choice for
    /// fragmented pictures is to defer it to picture-completion
    /// time because the prediction reads from already-reconstructed
    /// neighbours in raster order, and that order is only fully
    /// determined once every slice has been delivered.
    ///
    /// `components` is the per-component slice of mutable level-0 LL
    /// subbands the caller has assembled from the fragmented slice
    /// data. The §13.4 routine ([`intra_dc_prediction`]) runs
    /// in-place on each. Order is irrelevant: each component's LL
    /// subband is predicted independently (no inter-component
    /// signalling in the §13.4 routine).
    ///
    /// Returns:
    /// * `Ok(())` after a successful kick (LD path) or a successful
    ///   no-op (HQ path). An asymmetric picture (`dwt_depth_ho > 0`,
    ///   §12.4.4.3) succeeds too: the §14.4 else-branch targets each
    ///   component's level-0 **L** subband, which occupies the same
    ///   `[0][0]` pyramid slot as the symmetric LL band, so the
    ///   caller-supplied bands and the §13.4 routine are identical.
    /// * `Err(DcPredictionBeforePictureComplete)` if any data
    ///   fragment for the active picture is still outstanding.
    ///
    /// On success, the assembler does NOT auto-reset; the next setup
    /// fragment is accepted because `fragmented_picture_done()`
    /// continues to return `true` until [`Self::on_setup_fragment`]
    /// is called for a new picture.
    pub fn fragmented_wavelet_transform_dc_prediction(
        &self,
        components: &mut [&mut SubbandData],
    ) -> Result<(), AssemblerError> {
        if !self.fragmented_picture_done() {
            return Err(AssemblerError::DcPredictionBeforePictureComplete);
        }
        if !self.using_dc_prediction {
            // §14.5: the HQ path (`using_dc_prediction == false`)
            // skips the trailing `dc_prediction(...)` kick entirely.
            return Ok(());
        }
        // §14.4 `fragment_data`: with `dwt_depth_ho == 0` the kick
        // targets each component's level-0 LL subband; with
        // `dwt_depth_ho > 0` it targets the level-0 **L** subband —
        // which occupies the same pyramid slot (`[0][0]`), so the
        // caller passes the same band either way and the §13.4
        // routine is identical.
        for ll in components.iter_mut() {
            intra_dc_prediction(ll);
        }
        Ok(())
    }
}

impl ParseInfo {
    /// VC-2 v3 §10.5.2 Table 5 `is_fragment` predicate:
    /// `(parse_code & 0x0C) == 0x0C`.
    ///
    /// This matches the bit pattern that v3 streams use to mark
    /// picture-fragment data units (`0xCC` LD, `0xEC` HQ). In a v3
    /// stream context, a parse-info header for which this returns
    /// `true` is followed by a fragment header (parsed by
    /// [`FragmentHeader::parse`]) rather than a full picture unit.
    ///
    /// IMPORTANT: in earlier stream-syntax versions the bit pattern
    /// was used for different purposes. The BBC Dirac v2.2.3 spec
    /// reuses `0xCC` for "Low-delay Intra Reference Picture" and the
    /// VC-2 v2 spec did not define fragments at all. This predicate
    /// alone therefore does NOT identify a fragment unambiguously —
    /// the caller must combine it with a check that the active stream
    /// declares VC-2 v3 syntax (per the sequence header's
    /// `major_version` field; see [`crate::sequence`]).
    pub fn is_fragment_parse_code(&self) -> bool {
        (self.parse_code & 0x0C) == 0x0C
    }

    /// VC-2 v3 §10.5.2 Table 5 `is_ld(state) := (parse_code & 0xF8)
    /// == 0xC8`. Matches both `0xC8` (LD picture) and `0xCC` (LD
    /// picture fragment).
    ///
    /// This is the strict VC-2 v3 LD predicate. The pre-existing
    /// [`ParseInfo::is_low_delay`] uses the broader BBC Dirac
    /// v2.2.3 bit mask (`(parse_code & 0x88) == 0x88`); both are
    /// kept so a v3 dispatcher and a Dirac-spec dispatcher can
    /// query the appropriate one for the active stream version.
    pub fn is_ld_v3(&self) -> bool {
        (self.parse_code & 0xF8) == 0xC8
    }

    /// VC-2 v3 §10.5.2 Table 5 `is_hq(state) := (parse_code & 0xF8)
    /// == 0xE8`. Matches both `0xE8` (HQ picture) and `0xEC` (HQ
    /// picture fragment).
    pub fn is_hq_v3(&self) -> bool {
        (self.parse_code & 0xF8) == 0xE8
    }

    /// VC-2 v3 §10.5.2 Table 5 `is_picture(state) := (parse_code &
    /// 0x8C) == 0x88`. Matches the two non-fragment picture codes
    /// `0xC8` (LD) and `0xE8` (HQ); does NOT match the picture
    /// fragment codes `0xCC` / `0xEC` (those are routed via
    /// [`ParseInfo::is_fragment_parse_code`]).
    ///
    /// The pre-existing [`ParseInfo::is_picture`] uses the broader
    /// BBC Dirac v2.2.3 bit mask `(parse_code & 0x08) == 0x08`,
    /// which intentionally subsumes both pictures and fragments.
    /// Keep both: the v3 dispatcher wants the narrower
    /// "picture-only" version so fragment routing stays exclusive.
    pub fn is_picture_v3(&self) -> bool {
        (self.parse_code & 0x8C) == 0x88
    }

    /// VC-2 v3 §10.5.2 Table 5 `using_dc_prediction(state) :=
    /// (parse_code & 0x28) == 0x08`. True for LD pictures and LD
    /// fragments (`0xC8` / `0xCC`); false for HQ pictures and HQ
    /// fragments (`0xE8` / `0xEC`).
    ///
    /// This is the key §14.4 predicate: after a fragmented
    /// picture's slices are all received, the LD path runs a
    /// trailing `dc_prediction(...)` on the LL (or L) subbands;
    /// the HQ path does not.
    pub fn using_dc_prediction(&self) -> bool {
        (self.parse_code & 0x28) == 0x08
    }
}

/// Errors raised by [`FragmentedPictureDecoder`] when fragments are
/// fed in.
///
/// Sequencing errors (data-fragment-before-setup, setup-before-prior-
/// picture-complete, parse-code-flip, slice overflow, …) and the
/// asymmetric-transform gap are routed through [`AssemblerError`]; the
/// payload-level errors (truncated header bytes, malformed
/// transform-parameters, slice-coefficient overflow) are routed through
/// [`PictureError`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FragmentedPictureError {
    /// The fragment header itself failed to parse (the per-fragment
    /// parse-info-relative payload was shorter than the §14.2 minimum
    /// 8- or 12-byte header layout).
    Header(FragmentError),
    /// The fragment assembler's §14.1 / §14.3 / §14.4 sequencing or
    /// §14.5 trailing-kick contract was violated.
    Assembler(AssemblerError),
    /// The fragment payload (transform_parameters or slice bytes)
    /// failed to parse — same error surface as the non-fragmented
    /// [`crate::picture::decode_picture`] path.
    Picture(PictureError),
    /// A setup fragment arrived for a non-LD / non-HQ parse code. The
    /// only valid fragment parse codes are `0xCC` (LD fragment) and
    /// `0xEC` (HQ fragment); other §10.5.2 Table 4 codes are not
    /// fragments and the dispatcher should not feed them here.
    UnsupportedParseCode(u8),
    /// A data fragment / `finish` arrived before any setup fragment
    /// has been ingested — the decoder has no transform parameters
    /// or pyramid state to operate on yet.
    NoActivePicture,
    /// `finish()` was called while at least one §14.4 data slice was
    /// still outstanding for the active picture (i.e.
    /// `fragmented_picture_done() == false`).
    PictureIncomplete {
        slices_received: u32,
        slices_expected: u32,
    },
}

impl core::fmt::Display for FragmentedPictureError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Header(e) => write!(f, "{e}"),
            Self::Assembler(e) => write!(f, "{e}"),
            Self::Picture(e) => write!(f, "{e}"),
            Self::UnsupportedParseCode(c) => write!(
                f,
                "unsupported fragment parse code 0x{c:02X} (expected 0xCC LD or 0xEC HQ)"
            ),
            Self::NoActivePicture => write!(
                f,
                "fragment arrived before a setup fragment established a picture"
            ),
            Self::PictureIncomplete {
                slices_received,
                slices_expected,
            } => write!(
                f,
                "finish() called with {slices_received}/{slices_expected} slices ingested"
            ),
        }
    }
}

impl std::error::Error for FragmentedPictureError {}

impl From<FragmentError> for FragmentedPictureError {
    fn from(e: FragmentError) -> Self {
        Self::Header(e)
    }
}

impl From<AssemblerError> for FragmentedPictureError {
    fn from(e: AssemblerError) -> Self {
        Self::Assembler(e)
    }
}

impl From<PictureError> for FragmentedPictureError {
    fn from(e: PictureError) -> Self {
        Self::Picture(e)
    }
}

/// VC-2 v3 fragmented-picture decoder.
///
/// `FragmentedPictureDecoder` is the §14 driver that wraps a
/// [`FragmentAssembler`] together with the per-picture decoding state
/// (`TransformParameters`, the three component pyramids, and the
/// per-level dimension caches) and bridges the syntactic
/// [`FragmentEvent`] stream to the coefficient-decode primitives in
/// [`crate::picture`].
///
/// A driver feeds setup and data fragments in §14.1 order:
///
/// ```text
/// dec = FragmentedPictureDecoder::new(&sequence);
/// for each fragmented data-unit (parse_info, payload):
///     if parse_info.is_setup_fragment():
///         dec.on_setup_fragment(parse_info, payload)?;
///     else if parse_info.is_data_fragment():
///         dec.on_data_fragment(parse_info, payload)?;
/// let picture = dec.finish()?;
/// ```
///
/// `on_setup_fragment` parses the §14.2 fragment header (`Setup` kind)
/// then immediately runs the §12.4 transform-parameters block on the
/// bytes that follow, allocates a fresh per-component pyramid sized for
/// the sequence header's frame geometry, and transitions the assembler
/// to `ReceivingData`. `on_data_fragment` parses the §14.2 fragment
/// header (`Data` kind), walks the §14.4 raster `(slice_x, slice_y)`
/// coordinates, and per-slice calls the same `decode_ld_slice` /
/// `decode_hq_slice` primitives the non-fragmented
/// [`crate::picture::decode_picture`] path uses — the byte boundary
/// between slices comes from the same §13.5.3 / §13.5.4 rate-control
/// rules so a multi-slice data fragment is read as a contiguous bit
/// stream straddling those boundaries.
///
/// On the LD path, `finish()` runs the §14.5 trailing
/// `dc_prediction(...)` kick on each component's level-0 LL subband
/// before the IDWT; the HQ path skips the kick (§14.5 explicitly).
/// Both paths then run the §13.3 inverse wavelet transform, the §13.6
/// trim / clip / output offset, and return the
/// [`crate::picture::DecodedPicture`] — bit-exact-equivalent to running
/// [`crate::picture::decode_picture`] on a non-fragmented version of
/// the same picture.
///
/// The decoder is reusable: once `finish()` returns successfully its
/// assembler is in `ReceivingData` with `fragmented_picture_done()`
/// still true, so the next `on_setup_fragment` call is accepted per
/// §14.1 and starts the next picture cleanly.
#[derive(Debug, Clone)]
pub struct FragmentedPictureDecoder<'s> {
    sequence: &'s SequenceHeader,
    assembler: FragmentAssembler,
    /// Active picture's transform parameters (None before the first
    /// `on_setup_fragment` call).
    params: Option<TransformParameters>,
    /// Active picture's profile cached from the setup parse code.
    profile: Option<LowDelayProfile>,
    /// Picture number from the latest setup fragment.
    picture_number: u32,
    /// Per-component pyramids holding the dequantised coefficients.
    /// Sized by `init_pyramid` at setup-fragment time.
    y_py: Vec<[SubbandData; 4]>,
    u_py: Vec<[SubbandData; 4]>,
    v_py: Vec<[SubbandData; 4]>,
    /// Per-level `(width, height)` of every component's subband
    /// pyramid. Computed once per setup fragment and reused on every
    /// slice decode.
    luma_dims: Vec<(usize, usize)>,
    chroma_dims: Vec<(usize, usize)>,
}

impl<'s> FragmentedPictureDecoder<'s> {
    /// Build an empty decoder rooted at the sequence header that
    /// preceded the fragmented picture(s). The header carries the
    /// luma / chroma resolution and bit depth needed by the per-slice
    /// pyramid sizing and the §13.6 output-offset step.
    pub fn new(sequence: &'s SequenceHeader) -> Self {
        Self {
            sequence,
            assembler: FragmentAssembler::new(),
            params: None,
            profile: None,
            picture_number: 0,
            y_py: Vec::new(),
            u_py: Vec::new(),
            v_py: Vec::new(),
            luma_dims: Vec::new(),
            chroma_dims: Vec::new(),
        }
    }

    /// Borrow the underlying §14.3 assembler — useful for tests / for
    /// callers that want to introspect `fragmented_picture_done()` /
    /// `slices_received()` mid-picture.
    pub fn assembler(&self) -> &FragmentAssembler {
        &self.assembler
    }

    /// Active picture's `(slices_x, slices_y, dwt_depth)` cached from
    /// the latest setup-fragment transform parameters. Returns `None`
    /// before the first setup fragment is ingested.
    pub fn transform_parameters(&self) -> Option<&TransformParameters> {
        self.params.as_ref()
    }

    /// Active picture's `picture_number` per §14.2. Zero before the
    /// first setup fragment is ingested.
    pub fn picture_number(&self) -> u32 {
        self.picture_number
    }

    /// Ingest a §14.1 setup fragment.
    ///
    /// `payload` is the parse-info-relative byte slice — the bytes
    /// immediately after the 13-byte parse-info header, exactly as
    /// produced by [`crate::stream::DataUnitIter`]. The first 8 bytes
    /// are the §14.2 setup fragment header (`fragment_slice_count ==
    /// 0`); the remaining bytes carry the §12.4 transform-parameters
    /// block (byte-aligned).
    ///
    /// On success the assembler is in `ReceivingData` phase and the
    /// per-component pyramids are allocated; the next call must be
    /// [`Self::on_data_fragment`].
    pub fn on_setup_fragment(
        &mut self,
        parse_info: &ParseInfo,
        payload: &[u8],
    ) -> Result<(), FragmentedPictureError> {
        // §10.5.2 Table 4: only 0xCC (LD fragment) and 0xEC (HQ
        // fragment) are picture-fragment parse codes.
        let profile = low_delay_profile_for(parse_info.parse_code).ok_or(
            FragmentedPictureError::UnsupportedParseCode(parse_info.parse_code),
        )?;
        let header = FragmentHeader::parse(payload)?;
        if !matches!(header.kind, FragmentKind::Setup) {
            return Err(FragmentedPictureError::Assembler(
                AssemblerError::UnexpectedDataFragment,
            ));
        }
        self.assembler
            .on_setup_fragment(&header, parse_info.parse_code)?;

        // The transform_parameters block lives in the bytes after the
        // 8-byte setup header.
        let tp_bytes = &payload[FragmentHeader::MIN_SIZE..];
        let mut r = BitReader::new(tp_bytes);
        // Per §12.4 the block is byte-aligned at this entry; the
        // setup header is itself byte-aligned (8 bytes) so we start
        // at bit 0 of the trailing region — `byte_align` here is a
        // no-op but keeps the call shape consistent with the
        // non-fragmented path.
        r.byte_align();
        let params = parse_transform_parameters(
            &mut r,
            profile,
            self.sequence.parse_parameters.version_major,
        )?;
        self.assembler.on_transform_parameters(
            params.slices_x,
            params.slices_y,
            // §12.4.4.3 `dwt_depth_ho` — 0 for v2 streams and for v3
            // streams in the symmetric default; > 0 selects the
            // asymmetric (horizontal-only) layout, which the slice
            // unpack / pyramid / IDWT below all follow.
            params.dwt_depth_ho,
        )?;

        // Allocate the three component pyramids and pre-compute per-
        // level subband dims (every slice call needs both). Both are
        // asymmetric-aware (§13.2.2 / §13.2.3): the pyramid spans
        // `dwt_depth_ho + dwt_depth` levels.
        let luma_w = self.sequence.luma_width;
        let luma_h = self.sequence.luma_height;
        let chroma_w = self.sequence.chroma_width;
        let chroma_h = self.sequence.chroma_height;
        let ho = params.dwt_depth_ho;
        let total_levels = ho + params.dwt_depth;
        self.y_py = init_pyramid_ho(luma_w, luma_h, params.dwt_depth, ho);
        self.u_py = init_pyramid_ho(chroma_w, chroma_h, params.dwt_depth, ho);
        self.v_py = init_pyramid_ho(chroma_w, chroma_h, params.dwt_depth, ho);
        self.luma_dims = Vec::with_capacity(total_levels as usize + 1);
        self.chroma_dims = Vec::with_capacity(total_levels as usize + 1);
        for level in 0..=total_levels {
            self.luma_dims
                .push(subband_dims_ho(luma_w, luma_h, params.dwt_depth, ho, level));
            self.chroma_dims.push(subband_dims_ho(
                chroma_w,
                chroma_h,
                params.dwt_depth,
                ho,
                level,
            ));
        }
        self.picture_number = header.picture_number;
        self.profile = Some(profile);
        self.params = Some(params);
        Ok(())
    }

    /// Ingest a §14.1 data fragment.
    ///
    /// `payload` is the parse-info-relative byte slice. The first
    /// 12 bytes are the §14.2 data-fragment header
    /// (`fragment_slice_count > 0` plus `(x_offset, y_offset)`); the
    /// remaining bytes carry `slice_count` consecutive §13.5.3.2 LD
    /// or §13.5.4 HQ slices, byte-aligned at the entry to slice 0 and
    /// re-aligned at slice boundaries by the same rules as the
    /// non-fragmented path.
    pub fn on_data_fragment(
        &mut self,
        parse_info: &ParseInfo,
        payload: &[u8],
    ) -> Result<(), FragmentedPictureError> {
        if self.params.is_none() || self.profile.is_none() {
            return Err(FragmentedPictureError::NoActivePicture);
        }
        let header = FragmentHeader::parse(payload)?;
        if !matches!(header.kind, FragmentKind::Data { .. }) {
            return Err(FragmentedPictureError::Assembler(
                AssemblerError::UnexpectedDataFragment,
            ));
        }
        let event = self
            .assembler
            .on_data_fragment(&header, parse_info.parse_code)?;
        let coords = match event {
            FragmentEvent::DataSlices { coords, .. } => coords,
            // `on_data_fragment` only ever returns `DataSlices`; this
            // branch exists to keep the match exhaustive.
            FragmentEvent::SetupAccepted => {
                return Err(FragmentedPictureError::Assembler(
                    AssemblerError::UnexpectedDataFragment,
                ));
            }
        };

        // The slice payload starts immediately after the 12-byte data
        // fragment header.
        let slice_bytes = &payload[FragmentHeader::DATA_SIZE..];
        let mut r = BitReader::new(slice_bytes);
        // §13.5.3 / §13.5.4 entry: byte-aligned at the start of slice 0.
        r.byte_align();
        let profile = self.profile.expect("profile set above");
        let params = self.params.as_ref().expect("params set above");
        for (slice_x, slice_y) in coords {
            match profile {
                LowDelayProfile::LD => {
                    decode_ld_slice(
                        &mut r,
                        params,
                        &mut self.y_py,
                        &mut self.u_py,
                        &mut self.v_py,
                        slice_x,
                        slice_y,
                        &self.luma_dims,
                        &self.chroma_dims,
                    )?;
                }
                LowDelayProfile::HQ => {
                    decode_hq_slice(
                        &mut r,
                        params,
                        &mut self.y_py,
                        &mut self.u_py,
                        &mut self.v_py,
                        slice_x,
                        slice_y,
                        &self.luma_dims,
                        &self.chroma_dims,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Finalise the active picture once every §14.4 data slice has
    /// been ingested.
    ///
    /// Runs:
    /// 1. The §14.5 trailing `dc_prediction(...)` kick on each
    ///    component's level-0 LL subband (LD path only; HQ is a no-op).
    /// 2. The §13.3 inverse wavelet transform on every component.
    /// 3. The §13.6 trim / clip / output-offset step that maps signed
    ///    coefficients to the final `[0, 2^depth - 1]` sample range.
    ///
    /// Returns the resulting [`DecodedPicture`].
    ///
    /// Errors:
    /// * [`FragmentedPictureError::NoActivePicture`] if no setup
    ///   fragment has been ingested.
    /// * [`FragmentedPictureError::PictureIncomplete`] if at least
    ///   one §14.4 data slice is still outstanding.
    pub fn finish(&mut self) -> Result<DecodedPicture, FragmentedPictureError> {
        let params = self
            .params
            .as_ref()
            .ok_or(FragmentedPictureError::NoActivePicture)?;
        let profile = self
            .profile
            .ok_or(FragmentedPictureError::NoActivePicture)?;
        if !self.assembler.fragmented_picture_done() {
            return Err(FragmentedPictureError::PictureIncomplete {
                slices_received: self.assembler.slices_received(),
                slices_expected: self
                    .assembler
                    .slices_x()
                    .saturating_mul(self.assembler.slices_y()),
            });
        }

        // §14.5 trailing DC-prediction kick. The non-fragmented LD
        // path (`decode_low_delay_picture`) runs this inline after the
        // slice loop; the fragmented LD path defers it to picture
        // completion per §14.5. HQ skips entirely.
        if matches!(profile, LowDelayProfile::LD) {
            self.assembler
                .fragmented_wavelet_transform_dc_prediction(&mut [
                    &mut self.y_py[0][0],
                    &mut self.u_py[0][0],
                    &mut self.v_py[0][0],
                ])?;
        }

        // §15.4.1 IDWT — with `dwt_depth_ho == 0` `idwt_with_ho` is
        // byte-equivalent to the symmetric `idwt`.
        let y_data = idwt_with_ho(
            &self.y_py,
            params.wavelet,
            params.wavelet_ho,
            params.dwt_depth_ho,
        );
        let u_data = idwt_with_ho(
            &self.u_py,
            params.wavelet,
            params.wavelet_ho,
            params.dwt_depth_ho,
        );
        let v_data = idwt_with_ho(
            &self.v_py,
            params.wavelet,
            params.wavelet_ho,
            params.dwt_depth_ho,
        );

        let luma_w = self.sequence.luma_width as usize;
        let luma_h = self.sequence.luma_height as usize;
        let chroma_w = self.sequence.chroma_width as usize;
        let chroma_h = self.sequence.chroma_height as usize;
        let y = trim_clip_offset(&y_data, luma_w, luma_h, self.sequence.luma_depth);
        let u = trim_clip_offset(&u_data, chroma_w, chroma_h, self.sequence.chroma_depth);
        let v = trim_clip_offset(&v_data, chroma_w, chroma_h, self.sequence.chroma_depth);

        Ok(DecodedPicture {
            picture_number: self.picture_number,
            luma_width: luma_w,
            luma_height: luma_h,
            chroma_width: chroma_w,
            chroma_height: chroma_h,
            y,
            u,
            v,
            luma_depth: self.sequence.luma_depth,
            chroma_depth: self.sequence.chroma_depth,
        })
    }
}

// ----------------------------------------------------------------------
// §14 fragment EMITTER (round-386)
// ----------------------------------------------------------------------

/// One emitted §14 fragment data unit, ready for §10.5.1 parse-info
/// framing: the fragment parse code plus the complete fragment payload
/// (§14.2 fragment header + transform-parameter bytes for the setup
/// fragment, or slice bytes for a data fragment).
#[derive(Debug, Clone)]
pub struct FragmentUnit {
    /// `0xEC` (HQ) or `0xCC` (LD): the source picture's parse code with
    /// bit 2 set (Table 5, `is_fragment := (parse_code & 0x0C) == 0x0C`).
    pub parse_code: u8,
    /// Complete fragment payload following the 13-byte parse-info
    /// header.
    pub payload: Vec<u8>,
}

/// Errors raised by [`fragment_picture_payload`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FragmentEmitError {
    /// The parse code is not an LD (`0xC8`-shape) or HQ (`0xE8`-shape)
    /// picture code.
    NotAPictureParseCode(u8),
    /// `max_slices_per_fragment` must be at least 1.
    ZeroSlicesPerFragment,
    /// The payload's `transform_parameters()` block failed to parse.
    TransformParameters(PictureError),
    /// The slice walk ran off the end of the payload — the payload is
    /// shorter than its own transform parameters declare.
    Truncated,
    /// A §14.2 fragment-header field (`slice_count`, `x_offset`,
    /// `y_offset` — all 16-bit) cannot represent the requested split.
    FieldOverflow,
}

impl core::fmt::Display for FragmentEmitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotAPictureParseCode(pc) => {
                write!(f, "parse code {pc:#04X} is not an LD/HQ picture")
            }
            Self::ZeroSlicesPerFragment => write!(f, "max_slices_per_fragment must be >= 1"),
            Self::TransformParameters(e) => write!(f, "transform parameters: {e}"),
            Self::Truncated => write!(f, "picture payload truncated during slice walk"),
            Self::FieldOverflow => {
                write!(
                    f,
                    "fragment header field overflow (16-bit slice_count / offsets)"
                )
            }
        }
    }
}

impl std::error::Error for FragmentEmitError {}

/// **Split a non-fragmented LD / HQ picture payload into a §14 fragment
/// sequence** (round-386) — the encode-side inverse of
/// [`FragmentedPictureDecoder`].
///
/// `payload` is the picture payload as emitted by the crate's LD / HQ
/// picture encoders (§12.2 picture number + byte-aligned
/// `transform_parameters()` + byte-aligned slices);
/// `picture_parse_code` is the code the picture would carry
/// non-fragmented (`0xE8` / `0xC8` shapes); `major_version` is the
/// version the payload's transform parameters were encoded with (v3
/// payloads carry the §12.4.4 extended-transform flags). The result is
/// one setup fragment carrying the transform-parameter bytes followed
/// by data fragments of at most `max_slices_per_fragment` consecutive
/// raster-order slices, each stamped with the §14.2 raster offset of
/// its first slice.
///
/// The §14.2 `fragment_data_length` field "contains undefined data and
/// does not contribute to decoding"; this emitter fills it with the
/// fragment's trailing byte count when it fits in 16 bits and `0`
/// otherwise.
///
/// Slice boundaries are recovered from the payload itself: the HQ walk
/// follows §13.5.4 (prefix bytes, qindex byte, three
/// length-byte-scaled component blocks per slice); the LD widths are
/// the closed-form §13.5.3.2 `slice_bytes`. No encoder-internal state
/// is consulted, so any conformant payload — including one produced by
/// a different encoder — fragments correctly.
pub fn fragment_picture_payload(
    payload: &[u8],
    picture_parse_code: u8,
    major_version: u32,
    max_slices_per_fragment: u32,
) -> Result<Vec<FragmentUnit>, FragmentEmitError> {
    if max_slices_per_fragment == 0 {
        return Err(FragmentEmitError::ZeroSlicesPerFragment);
    }
    let profile = if (picture_parse_code & 0xF8) == 0xE8 {
        LowDelayProfile::HQ
    } else if (picture_parse_code & 0xF8) == 0xC8 {
        LowDelayProfile::LD
    } else {
        return Err(FragmentEmitError::NotAPictureParseCode(picture_parse_code));
    };
    if (picture_parse_code & 0x0C) == 0x0C {
        // Already a fragment code — fragmenting a fragment is a caller
        // bug, not a picture.
        return Err(FragmentEmitError::NotAPictureParseCode(picture_parse_code));
    }
    if payload.len() < 4 {
        return Err(FragmentEmitError::Truncated);
    }
    let picture_number = u32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]);

    // Transform parameters: parse with the production reader, then
    // byte-align — slice 0 starts at the next byte boundary.
    let mut r = BitReader::new(&payload[4..]);
    let tp = parse_transform_parameters(&mut r, profile, major_version)
        .map_err(FragmentEmitError::TransformParameters)?;
    r.byte_align();
    let tp_end = 4 + r.byte_pos();
    if tp_end > payload.len() {
        return Err(FragmentEmitError::Truncated);
    }

    // Raster-order slice byte ranges.
    let total_slices = (tp.slices_x as usize) * (tp.slices_y as usize);
    let mut slice_ranges: Vec<core::ops::Range<usize>> = Vec::with_capacity(total_slices);
    let mut p = tp_end;
    match profile {
        LowDelayProfile::HQ => {
            let scaler = tp.slice_size_scaler.max(1) as usize;
            for _ in 0..total_slices {
                let start = p;
                p = p
                    .checked_add(tp.slice_prefix_bytes as usize + 1)
                    .ok_or(FragmentEmitError::Truncated)?;
                for _ in 0..3 {
                    let len_byte = *payload.get(p).ok_or(FragmentEmitError::Truncated)? as usize;
                    p = p
                        .checked_add(1 + len_byte * scaler)
                        .ok_or(FragmentEmitError::Truncated)?;
                }
                if p > payload.len() {
                    return Err(FragmentEmitError::Truncated);
                }
                slice_ranges.push(start..p);
            }
        }
        LowDelayProfile::LD => {
            for sy in 0..tp.slices_y {
                for sx in 0..tp.slices_x {
                    let w = crate::picture::slice_bytes(
                        tp.slices_x,
                        tp.slice_bytes_numer,
                        tp.slice_bytes_denom,
                        sx,
                        sy,
                    ) as usize;
                    let start = p;
                    p = p.checked_add(w).ok_or(FragmentEmitError::Truncated)?;
                    if p > payload.len() {
                        return Err(FragmentEmitError::Truncated);
                    }
                    slice_ranges.push(start..p);
                }
            }
        }
    }

    // §14.2 field-width guards: slice_count and the raster offsets are
    // 16-bit. (A grid wider/taller than 65535 slices cannot be
    // addressed; a fragment cannot carry more than 65535 slices.)
    if tp.slices_x > u16::MAX as u32 + 1 || tp.slices_y > u16::MAX as u32 + 1 {
        return Err(FragmentEmitError::FieldOverflow);
    }

    let frag_code = picture_parse_code | 0x04;
    let data_len_field = |n: usize| -> [u8; 2] { u16::try_from(n).unwrap_or(0).to_be_bytes() };

    let mut out: Vec<FragmentUnit> = Vec::with_capacity(1 + total_slices);

    // Setup fragment: picture number + data length + slice_count 0 +
    // the transform-parameter bytes.
    let tp_bytes = &payload[4..tp_end];
    let mut setup = Vec::with_capacity(8 + tp_bytes.len());
    setup.extend_from_slice(&picture_number.to_be_bytes());
    setup.extend_from_slice(&data_len_field(tp_bytes.len()));
    setup.extend_from_slice(&0u16.to_be_bytes());
    setup.extend_from_slice(tp_bytes);
    out.push(FragmentUnit {
        parse_code: frag_code,
        payload: setup,
    });

    // Data fragments: consecutive raster-order chunks.
    let chunk = max_slices_per_fragment as usize;
    let mut s = 0usize;
    while s < total_slices {
        let n = chunk.min(total_slices - s);
        let count = u16::try_from(n).map_err(|_| FragmentEmitError::FieldOverflow)?;
        let x_off = (s as u32) % tp.slices_x;
        let y_off = (s as u32) / tp.slices_x;
        let (x_off, y_off) = (
            u16::try_from(x_off).map_err(|_| FragmentEmitError::FieldOverflow)?,
            u16::try_from(y_off).map_err(|_| FragmentEmitError::FieldOverflow)?,
        );
        let start = slice_ranges[s].start;
        let end = slice_ranges[s + n - 1].end;
        let bytes = &payload[start..end];
        let mut data = Vec::with_capacity(12 + bytes.len());
        data.extend_from_slice(&picture_number.to_be_bytes());
        data.extend_from_slice(&data_len_field(bytes.len()));
        data.extend_from_slice(&count.to_be_bytes());
        data.extend_from_slice(&x_off.to_be_bytes());
        data.extend_from_slice(&y_off.to_be_bytes());
        data.extend_from_slice(bytes);
        out.push(FragmentUnit {
            parse_code: frag_code,
            payload: data,
        });
        s += n;
    }
    Ok(out)
}

/// Frame a fragment-unit sequence into a complete elementary stream:
/// sequence header (`0x00`), the units, end-of-sequence (`0x10`), with
/// the §10.5.1 `next`/`previous` parse-offset chain wired.
fn frame_fragment_stream(sequence: &SequenceHeader, units: &[FragmentUnit]) -> Vec<u8> {
    let sh_payload = crate::encoder::encode_sequence_header(sequence);
    let pi_size = ParseInfo::SIZE;
    let mut out = Vec::new();
    let sh_unit_len = (pi_size + sh_payload.len()) as u32;
    crate::encoder::write_parse_info(&mut out, 0x00, sh_unit_len, 0);
    out.extend_from_slice(&sh_payload);
    let mut prev = sh_unit_len;
    for unit in units {
        let unit_len = (pi_size + unit.payload.len()) as u32;
        crate::encoder::write_parse_info(&mut out, unit.parse_code, unit_len, prev);
        out.extend_from_slice(&unit.payload);
        prev = unit_len;
    }
    crate::encoder::write_parse_info(&mut out, 0x10, 0, prev);
    out
}

/// **Encode a complete fragmented single-picture HQ intra stream**
/// (round-386): a `major_version = 3` sequence header (`0x00`), a
/// setup fragment + data fragments of at most
/// `max_slices_per_fragment` slices (all `0xEC`), and end-of-sequence
/// (`0x10`), with the §10.5.1 `next`/`previous` parse-offset chain
/// wired. The fragmented counterpart of
/// [`crate::encoder::encode_single_hq_intra_stream`].
///
/// `params.major_version` must be `>= 3` (fragmented pictures are VC-2
/// v3 syntax, and the picture's transform parameters must carry the
/// §12.4.4 extended flags the v3 sequence header promises) — use
/// [`crate::encoder::EncoderParams::with_major_version_3`]. The
/// emitted sequence header is the caller's with
/// `parse_parameters.version_major` forced to 3 so the stream is
/// self-consistent.
#[allow(clippy::too_many_arguments)]
pub fn encode_single_hq_intra_fragmented_stream(
    sequence: &SequenceHeader,
    params: &crate::encoder::EncoderParams,
    picture_number: u32,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    max_slices_per_fragment: u32,
) -> Vec<u8> {
    assert!(
        params.major_version >= 3,
        "fragmented pictures are v3 syntax; use EncoderParams::with_major_version_3()"
    );
    let mut seq3 = sequence.clone();
    seq3.parse_parameters.version_major = 3;

    let payload = crate::encoder::encode_hq_intra_picture(&seq3, params, picture_number, y, u, v);
    let units = fragment_picture_payload(
        &payload,
        0xE8,
        params.major_version,
        max_slices_per_fragment,
    )
    .expect("freshly encoded HQ payload must fragment cleanly");

    frame_fragment_stream(&seq3, &units)
}

/// **Encode a complete fragmented single-picture LD intra stream**
/// (round-386) — the LD (`0xCC`) counterpart of
/// [`encode_single_hq_intra_fragmented_stream`], and the fragmented
/// counterpart of [`crate::encoder::encode_single_ld_intra_stream`].
/// Same contract: `params.major_version` must be `>= 3`; the emitted
/// sequence header carries `version_major = 3`. The LD path is the one
/// with the §14.5 trailing DC-prediction kick on the decode side, so a
/// bit-exact round-trip through [`FragmentedPictureDecoder`] exercises
/// the full v3 LD pipeline.
#[allow(clippy::too_many_arguments)]
pub fn encode_single_ld_intra_fragmented_stream(
    sequence: &SequenceHeader,
    params: &crate::encoder::LdEncoderParams,
    picture_number: u32,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    max_slices_per_fragment: u32,
) -> Vec<u8> {
    assert!(
        params.major_version >= 3,
        "fragmented pictures are v3 syntax; set LdEncoderParams::major_version = 3"
    );
    let mut seq3 = sequence.clone();
    seq3.parse_parameters.version_major = 3;

    let payload = crate::encoder::encode_ld_intra_picture(&seq3, params, picture_number, y, u, v);
    let units = fragment_picture_payload(
        &payload,
        0xC8,
        params.major_version,
        max_slices_per_fragment,
    )
    .expect("freshly encoded LD payload must fragment cleanly");

    frame_fragment_stream(&seq3, &units)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_setup_payload(picture_number: u32, fragment_data_length: u16) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&picture_number.to_be_bytes());
        payload.extend_from_slice(&fragment_data_length.to_be_bytes());
        payload.extend_from_slice(&0u16.to_be_bytes()); // slice_count = 0
        payload
    }

    fn build_data_payload(
        picture_number: u32,
        fragment_data_length: u16,
        slice_count: u16,
        x_offset: u16,
        y_offset: u16,
    ) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&picture_number.to_be_bytes());
        payload.extend_from_slice(&fragment_data_length.to_be_bytes());
        payload.extend_from_slice(&slice_count.to_be_bytes());
        payload.extend_from_slice(&x_offset.to_be_bytes());
        payload.extend_from_slice(&y_offset.to_be_bytes());
        payload
    }

    /// §14.2 setup fragment: `fragment_slice_count == 0`. Header is
    /// exactly 8 bytes and decodes to `FragmentKind::Setup`.
    #[test]
    fn parse_setup_fragment_zero_slice_count() {
        let payload = build_setup_payload(0x1234_5678, 0xABCD);
        assert_eq!(payload.len(), 8);
        let parsed = FragmentHeader::parse(&payload).expect("setup fragment");
        assert_eq!(parsed.picture_number, 0x1234_5678);
        assert_eq!(parsed.fragment_data_length, 0xABCD);
        assert_eq!(parsed.kind, FragmentKind::Setup);
        assert_eq!(parsed.header_size(), 8);
    }

    /// §14.2 data fragment: `fragment_slice_count > 0`. Header is 12
    /// bytes and the `(x_offset, y_offset)` of the first slice is
    /// recovered.
    #[test]
    fn parse_data_fragment_nonzero_slice_count() {
        let payload = build_data_payload(0x0000_0007, 0x0040, 3, 1, 2);
        assert_eq!(payload.len(), 12);
        let parsed = FragmentHeader::parse(&payload).expect("data fragment");
        assert_eq!(parsed.picture_number, 7);
        assert_eq!(parsed.fragment_data_length, 0x0040);
        assert_eq!(
            parsed.kind,
            FragmentKind::Data {
                slice_count: 3,
                x_offset: 1,
                y_offset: 2,
            }
        );
        assert_eq!(parsed.header_size(), 12);
    }

    /// Payload shorter than the 8-byte setup minimum is rejected with
    /// a clear `needed` / `available` distinction.
    #[test]
    fn truncated_below_min_size_rejected() {
        let payload = [0u8; 7];
        let err = FragmentHeader::parse(&payload).unwrap_err();
        assert_eq!(
            err,
            FragmentError::Truncated {
                needed: 8,
                available: 7,
            }
        );
    }

    /// Slice count > 0 demands the full 12-byte data header. An 8-byte
    /// payload that begins with a non-zero slice count must report the
    /// 12-byte requirement, not the 8-byte minimum (so the caller can
    /// distinguish "header missing" from "data fragment offsets
    /// missing").
    #[test]
    fn truncated_data_fragment_reports_data_size() {
        let mut payload = [0u8; 8];
        // slice_count at bytes 6..8.
        payload[6] = 0x00;
        payload[7] = 0x05; // 5 slices
        let err = FragmentHeader::parse(&payload).unwrap_err();
        assert_eq!(
            err,
            FragmentError::Truncated {
                needed: 12,
                available: 8,
            }
        );
    }

    /// Trailing bytes after the header are ignored — the parser
    /// consumes exactly `header_size()` bytes and the caller is free
    /// to feed in transform-parameters / slice data behind it.
    #[test]
    fn trailing_bytes_are_ignored() {
        let mut payload = build_setup_payload(1, 0);
        payload.extend_from_slice(b"transform-params-go-here");
        let parsed = FragmentHeader::parse(&payload).expect("setup");
        assert_eq!(parsed.kind, FragmentKind::Setup);
        assert_eq!(parsed.header_size(), 8);
        // The slice past the header is the caller's to interpret —
        // assert it survived round-trip unchanged.
        assert_eq!(&payload[8..], b"transform-params-go-here");
    }

    /// §14.2 picture numbers are u32 and wrap at 2^32-1; the parser
    /// must round-trip the maximum value cleanly.
    #[test]
    fn parses_picture_number_u32_max() {
        let payload = build_setup_payload(u32::MAX, 0);
        let parsed = FragmentHeader::parse(&payload).expect("setup");
        assert_eq!(parsed.picture_number, u32::MAX);
    }

    /// `fragment_data_length` is "undefined data" per §14.2 — the
    /// parser must accept any value and preserve it for callers that
    /// need it for stream-layout tooling.
    #[test]
    fn fragment_data_length_round_trips_arbitrary_value() {
        let payload = build_setup_payload(0, 0xFFFF);
        let parsed = FragmentHeader::parse(&payload).expect("setup");
        assert_eq!(parsed.fragment_data_length, 0xFFFF);
    }

    /// `header_size()` returns 8 for setup, 12 for data. Pinned so a
    /// later round can dispatch on it without re-checking the variant.
    #[test]
    fn header_size_matches_kind() {
        let setup = FragmentHeader {
            picture_number: 0,
            fragment_data_length: 0,
            kind: FragmentKind::Setup,
        };
        assert_eq!(setup.header_size(), 8);
        let data = FragmentHeader {
            picture_number: 0,
            fragment_data_length: 0,
            kind: FragmentKind::Data {
                slice_count: 1,
                x_offset: 0,
                y_offset: 0,
            },
        };
        assert_eq!(data.header_size(), 12);
    }

    /// VC-2 v3 §10.5.2 Table 5 predicate `is_fragment(state) ==
    /// ((parse_code & 0x0C) == 0x0C)`. Pin the picture-fragment parse
    /// codes (0xCC LD, 0xEC HQ) on the true side and the
    /// non-fragment generic / picture codes on the false side.
    #[test]
    fn is_fragment_predicate_matches_v3_picture_fragments() {
        fn parse_info(parse_code: u8) -> ParseInfo {
            ParseInfo {
                parse_code,
                next_parse_offset: 0,
                previous_parse_offset: 0,
            }
        }
        // v3 picture-fragment parse codes per §10.5.2 Table 4.
        assert!(
            parse_info(0xCC).is_fragment_parse_code(),
            "0xCC = LD fragment"
        );
        assert!(
            parse_info(0xEC).is_fragment_parse_code(),
            "0xEC = HQ fragment"
        );
        // Generic and non-fragment picture codes per Table 4: bit
        // pattern `& 0x0C == 0x0C` is false for them.
        for code in [0x00u8, 0x10, 0x20, 0x30, 0xC8, 0xE8] {
            assert!(
                !parse_info(code).is_fragment_parse_code(),
                "0x{code:02X} should NOT match the v3 fragment predicate"
            );
        }
    }

    /// The predicate is a pure bit test. In the BBC Dirac v2.2.3 spec
    /// the same bit pattern is reused for core-syntax reference
    /// pictures (`0x0C`, `0x0D`, `0x0E`, `0x4C`, ...) and the v3
    /// fragment-code LD intra-reference reuse of `0xCC` — pin that
    /// the predicate fires on all of them so callers cannot misread
    /// it as "VC-2 v3 fragment only". Disambiguation belongs to the
    /// dispatcher's version check.
    #[test]
    fn is_fragment_predicate_also_fires_on_dirac_reference_picture_codes() {
        fn parse_info(parse_code: u8) -> ParseInfo {
            ParseInfo {
                parse_code,
                next_parse_offset: 0,
                previous_parse_offset: 0,
            }
        }
        // Dirac core-syntax reference picture codes (Table 9.1 of the
        // BBC spec): the bit pattern intentionally overlaps the v3
        // fragment bit pattern.
        for code in [0x0Cu8, 0x0D, 0x0E, 0x4C, 0xCC] {
            assert!(
                parse_info(code).is_fragment_parse_code(),
                "0x{code:02X} matches v3 (parse_code & 0x0C) == 0x0C predicate"
            );
        }
    }

    fn pi(parse_code: u8) -> ParseInfo {
        ParseInfo {
            parse_code,
            next_parse_offset: 0,
            previous_parse_offset: 0,
        }
    }

    /// VC-2 v3 §10.5.2 Table 5 `is_ld(state) := (parse_code & 0xF8)
    /// == 0xC8`. Fires on `0xC8` (LD picture) and `0xCC` (LD picture
    /// fragment); does not fire on `0xE8` / `0xEC` / non-picture
    /// codes.
    #[test]
    fn is_ld_v3_predicate_matches_only_ld_codes() {
        assert!(pi(0xC8).is_ld_v3());
        assert!(pi(0xCC).is_ld_v3());
        for code in [0x00u8, 0x10, 0x20, 0x30, 0xE8, 0xEC] {
            assert!(
                !pi(code).is_ld_v3(),
                "0x{code:02X} should NOT match v3 is_ld predicate"
            );
        }
    }

    /// VC-2 v3 §10.5.2 Table 5 `is_hq(state) := (parse_code & 0xF8)
    /// == 0xE8`. Fires on `0xE8` (HQ picture) and `0xEC` (HQ picture
    /// fragment).
    #[test]
    fn is_hq_v3_predicate_matches_only_hq_codes() {
        assert!(pi(0xE8).is_hq_v3());
        assert!(pi(0xEC).is_hq_v3());
        for code in [0x00u8, 0x10, 0x20, 0x30, 0xC8, 0xCC] {
            assert!(
                !pi(code).is_hq_v3(),
                "0x{code:02X} should NOT match v3 is_hq predicate"
            );
        }
    }

    /// VC-2 v3 §10.5.2 Table 5 `is_picture(state) := (parse_code &
    /// 0x8C) == 0x88`. Pure VC-2 v3: only `0xC8` and `0xE8`. The two
    /// fragment codes `0xCC` / `0xEC` deliberately fail this
    /// predicate so the v3 dispatcher routes them via
    /// `is_fragment_parse_code` instead.
    #[test]
    fn is_picture_v3_predicate_excludes_fragments() {
        assert!(pi(0xC8).is_picture_v3());
        assert!(pi(0xE8).is_picture_v3());
        assert!(
            !pi(0xCC).is_picture_v3(),
            "0xCC is a fragment, not a picture, under v3 routing"
        );
        assert!(
            !pi(0xEC).is_picture_v3(),
            "0xEC is a fragment, not a picture, under v3 routing"
        );
        for code in [0x00u8, 0x10, 0x20, 0x30] {
            assert!(!pi(code).is_picture_v3());
        }
    }

    /// VC-2 v3 §10.5.2 Table 5 `using_dc_prediction(state) :=
    /// (parse_code & 0x28) == 0x08`. True for both the LD picture
    /// (`0xC8`) and LD fragment (`0xCC`) codes; false for the HQ
    /// equivalents (`0xE8` / `0xEC`).
    #[test]
    fn using_dc_prediction_predicate_matches_only_ld_path() {
        assert!(pi(0xC8).using_dc_prediction(), "0xC8 = LD picture");
        assert!(pi(0xCC).using_dc_prediction(), "0xCC = LD fragment");
        assert!(!pi(0xE8).using_dc_prediction(), "0xE8 = HQ picture");
        assert!(!pi(0xEC).using_dc_prediction(), "0xEC = HQ fragment");
    }

    /// §14.4: `slice_coords(s, x_offset, y_offset, slices_x)` is the
    /// raster-scan slice index split into `(col, row)`. Single
    /// fragment carrying the whole picture from `(0, 0)`.
    #[test]
    fn slice_coords_top_left_walk_raster_order() {
        // 3-col, 2-row picture (slices_x = 3): raster indices 0..5
        // map to (0,0) (1,0) (2,0) (0,1) (1,1) (2,1).
        assert_eq!(slice_coords(0, 0, 0, 3), Some((0, 0)));
        assert_eq!(slice_coords(1, 0, 0, 3), Some((1, 0)));
        assert_eq!(slice_coords(2, 0, 0, 3), Some((2, 0)));
        assert_eq!(slice_coords(3, 0, 0, 3), Some((0, 1)));
        assert_eq!(slice_coords(4, 0, 0, 3), Some((1, 1)));
        assert_eq!(slice_coords(5, 0, 0, 3), Some((2, 1)));
    }

    /// §14.4: a data fragment that starts mid-row (`x_offset != 0`)
    /// can straddle the row boundary; the modulo / integer-division
    /// pair wraps `slice_x` back to 0 and increments `slice_y`.
    #[test]
    fn slice_coords_fragment_straddling_row_boundary() {
        // slices_x = 4; fragment starts at (3, 0) and carries 3
        // slices. Raster indices 3, 4, 5 → (3, 0), (0, 1), (1, 1).
        assert_eq!(slice_coords(0, 3, 0, 4), Some((3, 0)));
        assert_eq!(slice_coords(1, 3, 0, 4), Some((0, 1)));
        assert_eq!(slice_coords(2, 3, 0, 4), Some((1, 1)));
    }

    /// §14.4: `slices_x == 0` is rejected — a 0-column picture is
    /// out of spec (§13.5.6 requires ≥ 1 slice per dimension).
    #[test]
    fn slice_coords_rejects_zero_slices_x() {
        assert_eq!(slice_coords(0, 0, 0, 0), None);
        assert_eq!(slice_coords(10, 5, 7, 0), None);
    }

    /// §14.4: high-resolution stream stress — 256 slice columns,
    /// fragment starting at the bottom-right of the picture. Verify
    /// the `u64` arithmetic doesn't overflow on `u16::MAX` offsets.
    #[test]
    fn slice_coords_extreme_offsets_no_overflow() {
        // slices_x = 256, y_offset = u16::MAX → first raster index =
        // 65535 * 256 = 16776960. No overflow expected from
        // `u32::MAX` either: the formula upcasts via `u64`.
        let (sx, sy) = slice_coords(0, 0, u16::MAX, 256).unwrap();
        assert_eq!((sx, sy), (0, u32::from(u16::MAX)));
    }

    fn setup_hdr(picture_number: u32) -> FragmentHeader {
        FragmentHeader {
            picture_number,
            fragment_data_length: 0,
            kind: FragmentKind::Setup,
        }
    }

    fn data_hdr(
        picture_number: u32,
        slice_count: u16,
        x_offset: u16,
        y_offset: u16,
    ) -> FragmentHeader {
        FragmentHeader {
            picture_number,
            fragment_data_length: 0,
            kind: FragmentKind::Data {
                slice_count,
                x_offset,
                y_offset,
            },
        }
    }

    /// §14.1 / §14.3: a brand-new assembler accepts a setup fragment.
    /// The first transition stores the parse-code-derived
    /// `using_dc_prediction` flag and primes the assembler to await
    /// the transform parameters.
    #[test]
    fn assembler_accepts_first_setup_fragment_ld() {
        let mut asm = FragmentAssembler::new();
        let ev = asm.on_setup_fragment(&setup_hdr(7), 0xCC).unwrap();
        assert_eq!(ev, FragmentEvent::SetupAccepted);
        assert_eq!(asm.picture_number(), 7);
        assert!(asm.using_dc_prediction(), "0xCC LD fragment → DC pred");
        assert!(!asm.fragmented_picture_done());
        assert_eq!(asm.slices_received(), 0);
    }

    /// HQ counterpart — `0xEC` parse code yields
    /// `using_dc_prediction == false` so the §14.4 trailing kick is
    /// skipped on the HQ path.
    #[test]
    fn assembler_accepts_first_setup_fragment_hq() {
        let mut asm = FragmentAssembler::new();
        asm.on_setup_fragment(&setup_hdr(0), 0xEC).unwrap();
        assert!(!asm.using_dc_prediction(), "0xEC HQ fragment → no DC pred");
    }

    /// §14.1: data fragment before any setup is rejected as
    /// `UnexpectedDataFragment`.
    #[test]
    fn assembler_rejects_data_fragment_without_setup() {
        let mut asm = FragmentAssembler::new();
        let err = asm
            .on_data_fragment(&data_hdr(0, 1, 0, 0), 0xCC)
            .unwrap_err();
        assert_eq!(err, AssemblerError::UnexpectedDataFragment);
    }

    /// §14.1: data fragment after setup but before transform
    /// parameters is rejected — the assembler can't compute raster
    /// coordinates without `slices_x`.
    #[test]
    fn assembler_rejects_data_fragment_before_transform_parameters() {
        let mut asm = FragmentAssembler::new();
        asm.on_setup_fragment(&setup_hdr(0), 0xCC).unwrap();
        // No on_transform_parameters() call here.
        let err = asm
            .on_data_fragment(&data_hdr(0, 1, 0, 0), 0xCC)
            .unwrap_err();
        assert_eq!(err, AssemblerError::UnexpectedDataFragment);
    }

    /// §13.5.6: `slices_x == 0` from transform parameters is
    /// rejected.
    #[test]
    fn assembler_rejects_zero_slices_x_transform_parameters() {
        let mut asm = FragmentAssembler::new();
        asm.on_setup_fragment(&setup_hdr(0), 0xCC).unwrap();
        let err = asm.on_transform_parameters(0, 4, 0).unwrap_err();
        assert_eq!(
            err,
            AssemblerError::InvalidSliceGrid {
                slices_x: 0,
                slices_y: 4,
            }
        );
    }

    /// §14.4 happy path: a single data fragment carrying the whole
    /// 2x2 picture starting at (0, 0). The assembler emits the four
    /// raster coordinates in order, fires `picture_done` on the
    /// final slice, and `fragmented_picture_done()` then returns
    /// true.
    #[test]
    fn assembler_single_data_fragment_completes_picture() {
        let mut asm = FragmentAssembler::new();
        asm.on_setup_fragment(&setup_hdr(0), 0xCC).unwrap();
        asm.on_transform_parameters(2, 2, 0).unwrap();
        let ev = asm.on_data_fragment(&data_hdr(0, 4, 0, 0), 0xCC).unwrap();
        match ev {
            FragmentEvent::DataSlices {
                coords,
                picture_done,
            } => {
                assert_eq!(coords, vec![(0, 0), (1, 0), (0, 1), (1, 1)]);
                assert!(picture_done);
            }
            _ => panic!("expected DataSlices event"),
        }
        assert!(asm.fragmented_picture_done());
        assert_eq!(asm.slices_received(), 4);
    }

    /// §14.4: data fragments may be split arbitrarily. Drive a 3x2
    /// picture with three data fragments: 2 slices from (0, 0),
    /// 2 slices from (2, 0), 2 slices from (1, 1). Each fragment
    /// advances `slices_received`; only the final fragment fires
    /// `picture_done`.
    #[test]
    fn assembler_multiple_data_fragments_advance_progressively() {
        let mut asm = FragmentAssembler::new();
        asm.on_setup_fragment(&setup_hdr(5), 0xCC).unwrap();
        asm.on_transform_parameters(3, 2, 0).unwrap();

        let ev1 = asm.on_data_fragment(&data_hdr(5, 2, 0, 0), 0xCC).unwrap();
        match ev1 {
            FragmentEvent::DataSlices {
                coords,
                picture_done,
            } => {
                assert_eq!(coords, vec![(0, 0), (1, 0)]);
                assert!(!picture_done);
            }
            _ => panic!("expected DataSlices"),
        }
        assert_eq!(asm.slices_received(), 2);

        let ev2 = asm.on_data_fragment(&data_hdr(5, 2, 2, 0), 0xCC).unwrap();
        match ev2 {
            FragmentEvent::DataSlices {
                coords,
                picture_done,
            } => {
                // Raster start = 0*3 + 2 = 2, then 3 → (2,0) (0,1).
                assert_eq!(coords, vec![(2, 0), (0, 1)]);
                assert!(!picture_done);
            }
            _ => panic!("expected DataSlices"),
        }
        assert_eq!(asm.slices_received(), 4);

        let ev3 = asm.on_data_fragment(&data_hdr(5, 2, 1, 1), 0xCC).unwrap();
        match ev3 {
            FragmentEvent::DataSlices {
                coords,
                picture_done,
            } => {
                // Raster start = 1*3 + 1 = 4, then 5 → (1,1) (2,1).
                assert_eq!(coords, vec![(1, 1), (2, 1)]);
                assert!(picture_done);
            }
            _ => panic!("expected DataSlices"),
        }
        assert!(asm.fragmented_picture_done());
    }

    /// §14.2: data-fragment picture_number must match the setup
    /// fragment's. A mismatch is rejected with the setup vs data
    /// values surfaced in the error payload.
    #[test]
    fn assembler_rejects_mismatched_picture_number_on_data_fragment() {
        let mut asm = FragmentAssembler::new();
        asm.on_setup_fragment(&setup_hdr(42), 0xCC).unwrap();
        asm.on_transform_parameters(2, 1, 0).unwrap();
        let err = asm
            .on_data_fragment(&data_hdr(43, 1, 0, 0), 0xCC)
            .unwrap_err();
        assert_eq!(
            err,
            AssemblerError::PictureNumberMismatch {
                setup: 42,
                data: 43,
            }
        );
    }

    /// The §14.4 `using_dc_prediction` outcome is captured from the
    /// setup-fragment parse code; a data fragment whose parse code
    /// would flip the predicate is rejected (the §14.4 trailing
    /// kick is keyed off the captured predicate). Setup `0xCC` (LD,
    /// dc=true) + data `0xEC` (HQ, dc=false) is the regression
    /// fixture.
    #[test]
    fn assembler_rejects_inconsistent_parse_code_within_picture() {
        let mut asm = FragmentAssembler::new();
        asm.on_setup_fragment(&setup_hdr(1), 0xCC).unwrap();
        asm.on_transform_parameters(2, 1, 0).unwrap();
        let err = asm
            .on_data_fragment(&data_hdr(1, 1, 0, 0), 0xEC)
            .unwrap_err();
        assert!(matches!(
            err,
            AssemblerError::InconsistentParseCode {
                setup: _,
                data: 0xEC
            }
        ));
    }

    /// §14.4 ("Slices shall not be omitted or repeated") — a data
    /// fragment whose slice count would push the cumulative total
    /// past `slices_x * slices_y` is rejected with the deficit
    /// surfaced.
    #[test]
    fn assembler_rejects_slice_overflow() {
        let mut asm = FragmentAssembler::new();
        asm.on_setup_fragment(&setup_hdr(0), 0xCC).unwrap();
        asm.on_transform_parameters(2, 1, 0).unwrap();
        // 2 slices total; first fragment consumes 1, second tries 2.
        asm.on_data_fragment(&data_hdr(0, 1, 0, 0), 0xCC).unwrap();
        let err = asm
            .on_data_fragment(&data_hdr(0, 2, 1, 0), 0xCC)
            .unwrap_err();
        assert_eq!(
            err,
            AssemblerError::SliceOverflow {
                expected_total: 2,
                slices_received: 1,
                slice_count: 2,
            }
        );
    }

    /// §14.1: a setup fragment while a previous fragmented picture
    /// is still incomplete is rejected. Once the previous picture
    /// completes, the next setup is accepted (starts a new picture).
    #[test]
    fn assembler_rejects_setup_before_previous_picture_complete() {
        let mut asm = FragmentAssembler::new();
        asm.on_setup_fragment(&setup_hdr(0), 0xCC).unwrap();
        asm.on_transform_parameters(2, 1, 0).unwrap();
        // Receive only 1 of the 2 slices.
        asm.on_data_fragment(&data_hdr(0, 1, 0, 0), 0xCC).unwrap();
        assert!(!asm.fragmented_picture_done());
        let err = asm.on_setup_fragment(&setup_hdr(1), 0xCC).unwrap_err();
        assert_eq!(err, AssemblerError::SetupBeforePreviousPictureComplete);
        // Finish the picture; next setup is now accepted.
        asm.on_data_fragment(&data_hdr(0, 1, 1, 0), 0xCC).unwrap();
        assert!(asm.fragmented_picture_done());
        let ev = asm.on_setup_fragment(&setup_hdr(1), 0xCC).unwrap();
        assert_eq!(ev, FragmentEvent::SetupAccepted);
        assert_eq!(asm.picture_number(), 1);
        assert_eq!(asm.slices_received(), 0, "new picture resets slice counter");
    }

    /// §12.4.4.3 `dwt_depth_ho` carries through the assembler from
    /// transform parameters to the §14.4 trailing kick (it picks LL
    /// vs L subbands for the DC prediction). Pin that the value is
    /// preserved across the data-fragment ingest sequence.
    #[test]
    fn assembler_preserves_dwt_depth_ho_across_data_fragments() {
        let mut asm = FragmentAssembler::new();
        asm.on_setup_fragment(&setup_hdr(0), 0xCC).unwrap();
        asm.on_transform_parameters(2, 2, 3).unwrap();
        assert_eq!(asm.dwt_depth_ho(), 3);
        asm.on_data_fragment(&data_hdr(0, 4, 0, 0), 0xCC).unwrap();
        assert_eq!(asm.dwt_depth_ho(), 3);
        assert!(asm.fragmented_picture_done());
    }

    /// Convenience: drive the assembler through setup → transform
    /// parameters (symmetric, `dwt_depth_ho = 0`) → a single data
    /// fragment that delivers the whole `slices_x * slices_y` picture,
    /// using the supplied parse code (`0xCC` LD / `0xEC` HQ). Returns
    /// the primed assembler ready for the §14.5 trailing kick.
    fn drive_to_completion(parse_code: u8, slices_x: u32, slices_y: u32) -> FragmentAssembler {
        let total = (slices_x * slices_y) as u16;
        let mut asm = FragmentAssembler::new();
        asm.on_setup_fragment(&setup_hdr(0), parse_code).unwrap();
        asm.on_transform_parameters(slices_x, slices_y, 0).unwrap();
        asm.on_data_fragment(&data_hdr(0, total, 0, 0), parse_code)
            .unwrap();
        assert!(asm.fragmented_picture_done());
        asm
    }

    /// Build a 3x3 LL subband whose every coefficient is `1`. After
    /// the §13.4 / §14.5 trailing `dc_prediction` kick on the LD
    /// (`0xCC`) path, the reconstructed values walk diagonally per
    /// the spec's neighbour-mean prediction (see the existing
    /// `intra_dc_predict_first_row_col` test in `picture.rs`).
    fn flat_ll_band(width: usize, height: usize, value: i32) -> SubbandData {
        let mut b = SubbandData::new(width, height);
        for y in 0..height {
            for x in 0..width {
                b.set(y, x, value);
            }
        }
        b
    }

    /// §14.5 happy path on the LD (`0xCC`) path: after the picture
    /// is complete, the trailing `dc_prediction(...)` kick runs and
    /// applies §13.4 raster neighbour-mean prediction to every
    /// component's level-0 LL subband. Compare against the
    /// `intra_dc_predict_first_row_col` reference in `picture.rs`:
    /// the first row walks linearly (each cell adds the value of
    /// its left neighbour, which is also `1`), so the second cell
    /// becomes 2, the third 3, and so on.
    #[test]
    fn fragmented_dc_prediction_runs_on_ld_path() {
        let asm = drive_to_completion(0xCC, 2, 2);
        let mut y_ll = flat_ll_band(3, 3, 1);
        let mut u_ll = flat_ll_band(3, 3, 1);
        let mut v_ll = flat_ll_band(3, 3, 1);
        asm.fragmented_wavelet_transform_dc_prediction(&mut [&mut y_ll, &mut u_ll, &mut v_ll])
            .unwrap();
        // First row walks linearly from the seed `1`: 1, 2, 3 per
        // §13.4 left-neighbour prediction on the topmost row.
        for band in [&y_ll, &u_ll, &v_ll] {
            assert_eq!(band.get(0, 0), 1, "(0,0) carries the seed");
            assert_eq!(band.get(0, 1), 2, "(0,1) = seed + left");
            assert_eq!(band.get(0, 2), 3, "(0,2) = prev row1 + left");
        }
    }

    /// §14.5 / §10.5.2 Table 5: on the HQ (`0xEC`) path the
    /// trailing `dc_prediction(...)` is skipped entirely — the LL
    /// subbands survive the kick unchanged.
    #[test]
    fn fragmented_dc_prediction_skipped_on_hq_path() {
        let asm = drive_to_completion(0xEC, 2, 2);
        let mut y_ll = flat_ll_band(3, 3, 7);
        let original = y_ll.data.clone();
        asm.fragmented_wavelet_transform_dc_prediction(&mut [&mut y_ll])
            .unwrap();
        assert_eq!(
            y_ll.data, original,
            "HQ (0xEC) skips the §14.5 dc_prediction kick"
        );
    }

    /// §14.5 keys off `state[fragmented_picture_done]` per §14.4 —
    /// invoking the trailing kick before the picture is fully
    /// assembled is rejected.
    #[test]
    fn fragmented_dc_prediction_rejects_incomplete_picture() {
        let mut asm = FragmentAssembler::new();
        asm.on_setup_fragment(&setup_hdr(0), 0xCC).unwrap();
        asm.on_transform_parameters(2, 2, 0).unwrap();
        // Deliver only 2 of the 4 slices.
        asm.on_data_fragment(&data_hdr(0, 2, 0, 0), 0xCC).unwrap();
        assert!(!asm.fragmented_picture_done());
        let mut y_ll = flat_ll_band(3, 3, 1);
        let err = asm
            .fragmented_wavelet_transform_dc_prediction(&mut [&mut y_ll])
            .unwrap_err();
        assert_eq!(err, AssemblerError::DcPredictionBeforePictureComplete);
    }

    /// §14.4 / §12.4.4.3: when `dwt_depth_ho > 0`, the trailing
    /// prediction targets the level-0 **L** subband — the §14.4
    /// else-branch is `dc_prediction(state[y_transform][0][L])` —
    /// which occupies the same `[0][0]` pyramid slot as the symmetric
    /// LL band. The kick therefore succeeds on an asymmetric picture
    /// and runs the identical §13.4 routine on the caller-supplied
    /// band. (Earlier rounds rejected this case with
    /// `AsymmetricDcPredictionUnsupported`; that variant is no longer
    /// raised.)
    #[test]
    fn fragmented_dc_prediction_accepts_asymmetric_transform() {
        let mut asm = FragmentAssembler::new();
        asm.on_setup_fragment(&setup_hdr(0), 0xCC).unwrap();
        asm.on_transform_parameters(2, 2, 2).unwrap();
        asm.on_data_fragment(&data_hdr(0, 4, 0, 0), 0xCC).unwrap();
        assert!(asm.fragmented_picture_done());
        // Same seed pattern as the symmetric single-component test:
        // the asymmetric kick must produce the identical §13.4 result.
        let mut y_l = flat_ll_band(2, 2, 0);
        y_l.set(0, 0, 4);
        y_l.set(0, 1, 1);
        y_l.set(1, 0, 1);
        y_l.set(1, 1, 1);
        asm.fragmented_wavelet_transform_dc_prediction(&mut [&mut y_l])
            .unwrap();
        assert_eq!(y_l.get(0, 0), 4);
        assert_eq!(y_l.get(0, 1), 5);
        assert_eq!(y_l.get(1, 0), 5);
        assert_eq!(y_l.get(1, 1), 6);
    }

    /// §14.5 on a single-component (luma-only) call site: passing a
    /// one-element slice runs the kick on just that LL subband.
    /// Pinned so callers that drive YUV components one at a time do
    /// not have to construct a 3-element slice.
    #[test]
    fn fragmented_dc_prediction_accepts_single_component() {
        let asm = drive_to_completion(0xCC, 1, 1);
        let mut y_ll = flat_ll_band(2, 2, 0);
        // Pre-fill so the prediction has something to differentiate.
        y_ll.set(0, 0, 4);
        y_ll.set(0, 1, 1);
        y_ll.set(1, 0, 1);
        y_ll.set(1, 1, 1);
        asm.fragmented_wavelet_transform_dc_prediction(&mut [&mut y_ll])
            .unwrap();
        // (0,0) keeps its seed; (0,1) = 1 + left(4) = 5; (1,0) = 1 +
        // top(4) = 5; (1,1) = 1 + mean3(left=5, top-left=4, top=5) =
        // 1 + (5+4+5+1)/3 = 1 + 5 = 6.
        assert_eq!(y_ll.get(0, 0), 4);
        assert_eq!(y_ll.get(0, 1), 5);
        assert_eq!(y_ll.get(1, 0), 5);
        assert_eq!(y_ll.get(1, 1), 6);
    }

    /// §14.5 on an HQ picture: the kick is a no-op, but an empty
    /// `components` slice is still accepted — the caller may pass an
    /// empty slice on the HQ path simply because the kick will be
    /// skipped anyway. Pin that the empty-slice case does not panic
    /// on either path.
    #[test]
    fn fragmented_dc_prediction_accepts_empty_components() {
        let asm_hq = drive_to_completion(0xEC, 1, 1);
        asm_hq
            .fragmented_wavelet_transform_dc_prediction(&mut [])
            .unwrap();
        let asm_ld = drive_to_completion(0xCC, 1, 1);
        // LD path with no components: kick fires but iterates over
        // nothing, so still `Ok(())`.
        asm_ld
            .fragmented_wavelet_transform_dc_prediction(&mut [])
            .unwrap();
    }
}
