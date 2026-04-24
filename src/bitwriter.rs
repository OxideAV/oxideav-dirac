//! MSB-first bit writer + interleaved exp-Golomb VLC emission.
//!
//! This is the encoder-side counterpart to [`crate::bits::BitReader`].
//! It buffers bits MSB-first into a `Vec<u8>`: call `write_bit()` for
//! one bit at a time, `write_uint()` to emit an interleaved exp-Golomb
//! unsigned value (Annex A.3.2), or `write_uint_lit(n, v)` to emit a
//! byte-aligned `n`-byte unsigned integer literal (Annex A.2.3).
//!
//! `byte_align()` discards any partial byte by padding the rest of the
//! current byte with zero bits — matching the spec's byte-alignment
//! rule used between the sequence-header / transform-parameter blocks.

/// MSB-first bit writer into a growing byte buffer.
pub struct BitWriter {
    buf: Vec<u8>,
    /// Partial byte being assembled, left-justified (MSB first).
    current: u8,
    /// Number of bits already packed into `current`, 0..=7.
    n_bits: u8,
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            current: 0,
            n_bits: 0,
        }
    }

    /// Total whole bytes committed + 1 if a partial byte is in progress.
    pub fn byte_len(&self) -> usize {
        self.buf.len() + if self.n_bits > 0 { 1 } else { 0 }
    }

    /// Current committed byte length (partial byte NOT counted). Used
    /// when the caller needs the byte-aligned position for offsets.
    pub fn aligned_byte_pos(&self) -> usize {
        debug_assert!(self.n_bits == 0, "aligned_byte_pos while misaligned");
        self.buf.len()
    }

    /// Flush any partial byte into the buffer and return the full byte
    /// vec. Bits in the partial byte are left-justified (MSB first) so
    /// that read-back with [`crate::bits::BitReader`] recovers the
    /// exact bits that were written. Consumes self so a stray write
    /// after finish is impossible.
    pub fn finish(mut self) -> Vec<u8> {
        if self.n_bits > 0 {
            // Pad with zero bits on the right so the already-written
            // bits occupy the high end of the byte.
            let shift = 8 - self.n_bits;
            self.buf.push(self.current << shift);
            self.current = 0;
            self.n_bits = 0;
        }
        self.buf
    }

    /// Pad the current byte out with zero bits to a byte boundary.
    pub fn byte_align(&mut self) {
        if self.n_bits > 0 {
            let shift = 8 - self.n_bits;
            self.buf.push(self.current << shift);
            self.current = 0;
            self.n_bits = 0;
        }
    }

    /// Append a single bit (0 or 1). MSB-first within the output byte.
    pub fn write_bit(&mut self, bit: u32) {
        debug_assert!(bit <= 1);
        self.current = (self.current << 1) | (bit as u8 & 1);
        self.n_bits += 1;
        if self.n_bits == 8 {
            self.buf.push(self.current);
            self.current = 0;
            self.n_bits = 0;
        }
    }

    /// Append a boolean — 0 or 1.
    pub fn write_bool(&mut self, b: bool) {
        self.write_bit(if b { 1 } else { 0 });
    }

    /// Append `n` raw bits of `value`, MSB-first.
    pub fn write_nbits(&mut self, n: u32, value: u32) {
        for i in (0..n).rev() {
            self.write_bit((value >> i) & 1);
        }
    }

    /// Byte-align and append `n` bytes of `value`, big-endian.
    pub fn write_uint_lit(&mut self, n: u32, value: u32) {
        self.byte_align();
        self.write_nbits(8 * n, value);
    }

    /// Write raw bytes at the current byte-aligned position.
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        self.byte_align();
        self.buf.extend_from_slice(bytes);
    }

    /// Encode `value` as an interleaved exp-Golomb unsigned integer
    /// (Annex A.3.2). The code's length is `2k+1` bits, where `k`
    /// is the position of the most-significant bit of `value + 1`.
    pub fn write_uint(&mut self, value: u32) {
        // Bits of (value + 1) in MSB-first order form the magnitude;
        // the interleaved encoding emits each bit after a leading 0
        // "follow" bit, then terminates with a 1 follow bit.
        //
        // E.g. value=0 -> value+1=1 (just the MSB) -> encode as "1".
        //      value=2 -> value+1=3 (bits "11")    -> "0 1 1" -> "011".
        let n = value.wrapping_add(1);
        let bit_len = 32 - n.leading_zeros();
        // The top bit of `n` is always 1 and isn't transmitted as data;
        // the (bit_len - 1) lower bits are each preceded by a `0`
        // "follow" bit, then a final `1` follow bit terminates.
        for i in (0..bit_len - 1).rev() {
            self.write_bit(0);
            self.write_bit((n >> i) & 1);
        }
        self.write_bit(1);
    }

    /// Encode a signed interleaved exp-Golomb integer (Annex A.3.3).
    pub fn write_sint(&mut self, value: i32) {
        let mag = value.unsigned_abs();
        self.write_uint(mag);
        if mag != 0 {
            self.write_bit(if value < 0 { 1 } else { 0 });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitReader;

    #[test]
    fn write_bit_packs_msb_first() {
        let mut w = BitWriter::new();
        w.write_bit(1);
        w.write_bit(0);
        w.write_bit(1);
        w.write_bit(1);
        w.write_bit(0);
        w.write_bit(1);
        w.write_bit(0);
        w.write_bit(0);
        let out = w.finish();
        assert_eq!(out, vec![0b1011_0100]);
    }

    #[test]
    fn byte_align_pads_with_zeros() {
        let mut w = BitWriter::new();
        w.write_bit(1);
        w.write_bit(0);
        w.byte_align();
        w.write_nbits(8, 0x7A);
        let out = w.finish();
        assert_eq!(out, vec![0b1000_0000, 0x7A]);
    }

    #[test]
    fn exp_golomb_write_matches_read() {
        // Spot-check against Table A.1 via roundtrip.
        for &v in &[0u32, 1, 2, 3, 4, 5, 6, 7, 100, 1023, 1_000_000] {
            let mut w = BitWriter::new();
            w.write_uint(v);
            let out = w.finish();
            let mut r = BitReader::new(&out);
            assert_eq!(r.read_uint(), v, "uint {v}");
        }
    }

    #[test]
    fn exp_golomb_signed_roundtrip() {
        for &v in &[0i32, 1, -1, 2, -2, 12345, -12345, 1_000_000, -1_000_000] {
            let mut w = BitWriter::new();
            w.write_sint(v);
            let out = w.finish();
            let mut r = BitReader::new(&out);
            assert_eq!(r.read_sint(), v, "sint {v}");
        }
    }

    #[test]
    fn exp_golomb_bit_pattern_spot_check() {
        // value=0 -> "1" -> 0b1000_0000
        let mut w = BitWriter::new();
        w.write_uint(0);
        assert_eq!(w.finish(), vec![0b1000_0000]);
        // value=2 -> "011" -> 0b0110_0000
        let mut w = BitWriter::new();
        w.write_uint(2);
        assert_eq!(w.finish(), vec![0b0110_0000]);
    }
}
