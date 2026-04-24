//! Dirac Parse Info headers (§9.4 / §9.6).
//!
//! Every data unit in a Dirac sequence is preceded by a 13-byte parse
//! info header:
//!
//! ```text
//!   +0..+4  : BBCD prefix            (0x42 0x42 0x43 0x44)
//!   +4      : parse code             (see Table 9.1)
//!   +5..+9  : next parse offset      (uint32 BE, 0 = last unit)
//!   +9..+13 : previous parse offset  (uint32 BE, 0 = first unit)
//! ```
//!
//! The parse code is a single byte whose bits encode the data-unit
//! kind, the number of motion-compensation references, whether
//! arithmetic coding is used, etc. (Section 9.6.2 "rationale").
//! Rather than sprinkle magic numbers throughout the parser, the
//! predicates below mirror the spec's pseudo-code names 1-for-1.

/// The 4-byte `BBCD` ISO/IEC-646 prefix that marks every parse info
/// header.
pub const BBCD: &[u8; 4] = b"BBCD";

/// A parsed 13-byte parse info header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseInfo {
    /// The parse code byte (see Table 9.1).
    pub parse_code: u8,
    /// Offset from this header's first byte to the next header's first
    /// byte, in bytes. Zero means "this is the last header in the
    /// sequence".
    pub next_parse_offset: u32,
    /// Offset from this header's first byte to the previous header's
    /// first byte, in bytes. Zero means "this is the first header in
    /// the sequence".
    pub previous_parse_offset: u32,
}

impl ParseInfo {
    /// Parse info headers are exactly 13 bytes.
    pub const SIZE: usize = 13;

    /// Does `data[pos..]` start with the BBCD prefix?
    pub fn has_prefix_at(data: &[u8], pos: usize) -> bool {
        data.len() >= pos + 4 && &data[pos..pos + 4] == BBCD
    }

    /// Parse a 13-byte parse info header from `data[pos..]`. Returns
    /// `None` if the slice is too short or the prefix mismatches.
    pub fn parse(data: &[u8], pos: usize) -> Option<Self> {
        if data.len() < pos + Self::SIZE {
            return None;
        }
        if !Self::has_prefix_at(data, pos) {
            return None;
        }
        let parse_code = data[pos + 4];
        let next_parse_offset =
            u32::from_be_bytes([data[pos + 5], data[pos + 6], data[pos + 7], data[pos + 8]]);
        let previous_parse_offset = u32::from_be_bytes([
            data[pos + 9],
            data[pos + 10],
            data[pos + 11],
            data[pos + 12],
        ]);
        Some(Self {
            parse_code,
            next_parse_offset,
            previous_parse_offset,
        })
    }

    /// Scan forward from `pos` in search of the next BBCD prefix.
    /// Returns the byte offset of the prefix's first byte, or `None`
    /// if no prefix is found before end-of-data.
    pub fn find_next(data: &[u8], pos: usize) -> Option<usize> {
        let end = data.len().saturating_sub(3);
        let mut i = pos;
        while i < end {
            if data[i] == b'B' && data[i + 1] == b'B' && data[i + 2] == b'C' && data[i + 3] == b'D'
            {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    // ---- Parse code predicates (§9.6.1) ------------------------------

    /// 0x00 — sequence header.
    pub fn is_seq_header(&self) -> bool {
        self.parse_code == 0x00
    }

    /// 0x10 — end of sequence.
    pub fn is_end_of_sequence(&self) -> bool {
        self.parse_code == 0x10
    }

    /// `(code & 0xF8) == 0x20` — auxiliary data.
    pub fn is_auxiliary_data(&self) -> bool {
        (self.parse_code & 0xF8) == 0x20
    }

    /// 0x30 — padding data.
    pub fn is_padding(&self) -> bool {
        self.parse_code == 0x30
    }

    /// A picture of any flavour (core / low-delay, intra / inter).
    pub fn is_picture(&self) -> bool {
        (self.parse_code & 0x08) == 0x08
    }

    /// Low-delay VC-2 picture (MSB set).
    pub fn is_low_delay(&self) -> bool {
        (self.parse_code & 0x88) == 0x88
    }

    /// Core-syntax picture (bit 7 clear, bit 3 set).
    pub fn is_core_syntax(&self) -> bool {
        (self.parse_code & 0x88) == 0x08
    }

    /// True if this picture uses arithmetic coding (core syntax only).
    pub fn using_ac(&self) -> bool {
        (self.parse_code & 0x48) == 0x08
    }

    /// A reference picture (bits 2,3 set).
    pub fn is_reference(&self) -> bool {
        (self.parse_code & 0x0C) == 0x0C
    }

    /// A non-reference picture (bit 3 set but bit 2 clear).
    pub fn is_non_reference(&self) -> bool {
        (self.parse_code & 0x0C) == 0x08
    }

    /// Number of motion-compensation references (0..=2).
    pub fn num_refs(&self) -> u8 {
        self.parse_code & 0x03
    }

    /// Intra picture: picture with zero references.
    pub fn is_intra(&self) -> bool {
        self.is_picture() && self.num_refs() == 0
    }

    /// Inter picture: picture with at least one reference.
    pub fn is_inter(&self) -> bool {
        self.is_picture() && self.num_refs() > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hdr(code: u8, next: u32, prev: u32) -> [u8; 13] {
        let mut buf = [0u8; 13];
        buf[..4].copy_from_slice(BBCD);
        buf[4] = code;
        buf[5..9].copy_from_slice(&next.to_be_bytes());
        buf[9..13].copy_from_slice(&prev.to_be_bytes());
        buf
    }

    #[test]
    fn parse_sequence_header_code() {
        let buf = hdr(0x00, 42, 0);
        let pi = ParseInfo::parse(&buf, 0).unwrap();
        assert_eq!(pi.parse_code, 0x00);
        assert_eq!(pi.next_parse_offset, 42);
        assert_eq!(pi.previous_parse_offset, 0);
        assert!(pi.is_seq_header());
        assert!(!pi.is_picture());
    }

    #[test]
    fn predicates_for_intra_ac_reference_picture() {
        // 0x0C — Intra Reference Picture (arithmetic coding).
        let pi = ParseInfo::parse(&hdr(0x0C, 0, 0), 0).unwrap();
        assert!(pi.is_picture());
        assert!(pi.is_core_syntax());
        assert!(pi.using_ac());
        assert!(pi.is_reference());
        assert!(!pi.is_non_reference());
        assert_eq!(pi.num_refs(), 0);
        assert!(pi.is_intra());
        assert!(!pi.is_inter());
    }

    #[test]
    fn predicates_for_inter_non_reference_two_refs() {
        // 0x0A — Inter Non-Reference Picture (arith coding, 2 refs).
        let pi = ParseInfo::parse(&hdr(0x0A, 0, 0), 0).unwrap();
        assert!(pi.is_picture());
        assert!(pi.is_core_syntax());
        assert!(pi.using_ac());
        assert!(!pi.is_reference());
        assert!(pi.is_non_reference());
        assert_eq!(pi.num_refs(), 2);
        assert!(pi.is_inter());
    }

    #[test]
    fn predicates_for_low_delay_intra() {
        // 0xCC — low-delay intra reference, 0xC8 — low-delay non-ref.
        let a = ParseInfo::parse(&hdr(0xCC, 0, 0), 0).unwrap();
        let b = ParseInfo::parse(&hdr(0xC8, 0, 0), 0).unwrap();
        assert!(a.is_low_delay());
        assert!(b.is_low_delay());
        assert!(!a.is_core_syntax());
        assert!(!a.using_ac());
        assert!(a.is_intra());
        assert!(b.is_intra());
    }

    #[test]
    fn find_next_scans_for_prefix() {
        let mut buf = vec![0u8; 32];
        let header = hdr(0x00, 0, 0);
        let n = (buf.len() - 20).min(header.len());
        buf[20..20 + n].copy_from_slice(&header[..n]);
        assert_eq!(ParseInfo::find_next(&buf, 0), Some(20));
    }

    #[test]
    fn rejects_bad_prefix() {
        let mut buf = hdr(0x00, 0, 0);
        buf[0] = b'X';
        assert!(ParseInfo::parse(&buf, 0).is_none());
    }
}
