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

use crate::parse_info::ParseInfo;

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
}
