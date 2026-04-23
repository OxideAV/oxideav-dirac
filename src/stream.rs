//! Stream walker: turn a Dirac byte stream into a sequence of
//! `(parse_info, payload)` pairs.
//!
//! A Dirac stream (§9.2) is a concatenation of *sequences*, where each
//! sequence is an alternating list of 13-byte parse info headers and
//! variable-length data units. Every data unit is preceded by exactly
//! one parse info header, and `next_parse_offset` points from the
//! current header's first byte to the next header's first byte (so a
//! header with no following unit has `next_parse_offset == 0`).
//!
//! This walker is deliberately permissive: it scans the stream
//! byte-by-byte looking for the BBCD prefix and emits every parse info
//! it finds. If a header's `next_parse_offset` is non-zero and
//! consistent with the bytes in the buffer, we use it as a fast jump;
//! otherwise we fall back to a byte-by-byte search for the next BBCD.
//! This keeps us robust against muxers that drop or rewrite offsets.

use crate::parse_info::ParseInfo;

/// A single data unit found in the stream, with its parse info and the
/// bytes between this header and the next (exclusive of either header).
#[derive(Debug, Clone)]
pub struct DataUnit<'a> {
    pub parse_info: ParseInfo,
    /// Offset (within the original buffer) of this unit's parse info.
    pub pi_offset: usize,
    /// Payload slice: bytes strictly between this parse info and the
    /// next one (or end-of-stream).
    pub payload: &'a [u8],
}

/// Iterator yielding `DataUnit`s from a Dirac byte stream.
pub struct DataUnitIter<'a> {
    data: &'a [u8],
    pos: usize,
    done: bool,
}

impl<'a> DataUnitIter<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        // Find the first BBCD; skip any leading junk.
        let pos = ParseInfo::find_next(data, 0).unwrap_or(data.len());
        Self {
            data,
            pos,
            done: pos >= data.len(),
        }
    }
}

impl<'a> Iterator for DataUnitIter<'a> {
    type Item = DataUnit<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let pi = ParseInfo::parse(self.data, self.pos)?;
        let pi_offset = self.pos;
        let payload_start = pi_offset + ParseInfo::SIZE;

        // Try next_parse_offset first. A value of 0 means "last unit".
        let next_pi_pos = if pi.next_parse_offset == 0 {
            None
        } else {
            let candidate = pi_offset.saturating_add(pi.next_parse_offset as usize);
            if candidate >= self.data.len() || !ParseInfo::has_prefix_at(self.data, candidate) {
                // Offset lied — fall back to byte search.
                ParseInfo::find_next(self.data, payload_start)
            } else {
                Some(candidate)
            }
        };

        let payload_end = next_pi_pos.unwrap_or(self.data.len());
        let payload_end = payload_end.min(self.data.len()).max(payload_start);
        let payload = &self.data[payload_start..payload_end];

        self.done = next_pi_pos.is_none();
        if let Some(n) = next_pi_pos {
            self.pos = n;
        }

        Some(DataUnit {
            parse_info: pi,
            pi_offset,
            payload,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_info::BBCD;

    fn make_header(code: u8, next_off: u32, prev_off: u32) -> [u8; 13] {
        let mut b = [0u8; 13];
        b[..4].copy_from_slice(BBCD);
        b[4] = code;
        b[5..9].copy_from_slice(&next_off.to_be_bytes());
        b[9..13].copy_from_slice(&prev_off.to_be_bytes());
        b
    }

    #[test]
    fn empty_stream_is_empty_iter() {
        let data = [];
        assert_eq!(DataUnitIter::new(&data).count(), 0);
    }

    #[test]
    fn walk_two_units_with_valid_offsets() {
        // Layout: [seq header 13b][payload 4b][pic 13b][payload 0]
        let mut buf = Vec::new();
        let seq = make_header(0x00, 17, 0); // 13 + 4 = 17
        buf.extend_from_slice(&seq);
        buf.extend_from_slice(b"abcd");
        let pic = make_header(0xCC, 0, 17);
        buf.extend_from_slice(&pic);

        let units: Vec<_> = DataUnitIter::new(&buf).collect();
        assert_eq!(units.len(), 2);
        assert!(units[0].parse_info.is_seq_header());
        assert_eq!(units[0].payload, b"abcd");
        assert!(units[1].parse_info.is_low_delay());
        assert_eq!(units[1].payload, b"");
    }

    #[test]
    fn broken_offsets_fall_back_to_byte_search() {
        // First header says next_parse_offset = 99999, which overflows
        // the buffer — walker must fall back to scanning.
        let mut buf = Vec::new();
        buf.extend_from_slice(&make_header(0x00, 99_999, 0));
        buf.extend_from_slice(b"xx"); // padding bytes
        buf.extend_from_slice(&make_header(0x10, 0, 0)); // end of seq
        let units: Vec<_> = DataUnitIter::new(&buf).collect();
        assert_eq!(units.len(), 2);
        assert_eq!(units[0].payload, b"xx");
        assert!(units[1].parse_info.is_end_of_sequence());
    }

    #[test]
    fn leading_junk_is_skipped() {
        let mut buf = Vec::from(&b"garbageBBCD"[..3]); // partial BBCD
        buf.extend_from_slice(&make_header(0x10, 0, 0));
        let units: Vec<_> = DataUnitIter::new(&buf).collect();
        assert_eq!(units.len(), 1);
        assert!(units[0].parse_info.is_end_of_sequence());
    }
}
