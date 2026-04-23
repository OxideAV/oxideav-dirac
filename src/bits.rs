//! Dirac bit-reader and variable-length-code primitives.
//!
//! The Dirac specification accesses the compressed stream byte-by-byte,
//! MSB-first within each byte (Annex A.1). The decoder keeps a
//! "current byte" plus a `next_bit` index from 7 (msb) down to 0 (lsb);
//! once `next_bit` wraps below zero, the next byte is fetched and the
//! index resets to 7. `byte_align()` discards any partially-read byte so
//! the next read starts on a byte boundary.
//!
//! The interleaved exp-Golomb coding in Annex A.3 is built on top of
//! this bit accessor: follow bits alternate with data bits, with a
//! `1` follow bit terminating the code. The unsigned magnitude `N`
//! satisfies `N + 1 = 1 xK-1 xK-2 … x0` in binary, where the data bits
//! are the `xi`.
//!
//! A "bounded block" variant (`BoundedBitReader`) tracks `bits_left`
//! across a fixed-size block: reads past the end return 1, which makes
//! exp-Golomb default to zero, matching Annex A.3.1.

/// Unbounded MSB-first bit reader over a borrowed byte slice.
///
/// This mirrors the state variables `CURRENT_BYTE` / `NEXT_BIT` from
/// Annex A.1. `next_bit` ranges from 7 (msb) down to 0 (lsb); when it
/// dips below zero we load the next byte.
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Position of the next byte to load; after `read_byte()` the byte at
    /// index `byte_pos - 1` is the one now in `current_byte`.
    byte_pos: usize,
    current_byte: u8,
    /// 7..=0 while bits remain in `current_byte`; -1 triggers a reload.
    next_bit: i8,
    /// `True` once we've walked past the end of the slice. Further reads
    /// return 0 to keep the spec's bounded-read semantics safe — most
    /// Dirac parsers terminate well before that.
    eof: bool,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            current_byte: 0,
            next_bit: -1,
            eof: false,
        }
    }

    /// Position (in bytes) of the next byte yet to be loaded into
    /// `current_byte`. After `byte_align()` this equals the cursor in
    /// the stream.
    pub fn byte_pos(&self) -> usize {
        self.byte_pos
    }

    /// Number of bits remaining in the slice from the cursor's point of
    /// view. Zero once the reader has exhausted the input.
    pub fn bits_remaining(&self) -> usize {
        if self.eof {
            return 0;
        }
        let full_bytes = self.data.len().saturating_sub(self.byte_pos);
        let leftover = if self.next_bit >= 0 {
            (self.next_bit + 1) as usize
        } else {
            0
        };
        full_bytes * 8 + leftover
    }

    /// Read the next byte into `current_byte` and reset `next_bit` to 7.
    /// Sets the `eof` flag if the slice is exhausted.
    fn load_byte(&mut self) {
        if self.byte_pos >= self.data.len() {
            self.eof = true;
            self.current_byte = 0;
        } else {
            self.current_byte = self.data[self.byte_pos];
            self.byte_pos += 1;
        }
        self.next_bit = 7;
    }

    /// Read a single bit (MSB-first within the current byte). Returns 0
    /// after EOF. Matches `read_bit()` / Annex A.1.2.
    pub fn read_bit(&mut self) -> u32 {
        if self.next_bit < 0 {
            self.load_byte();
        }
        let bit = ((self.current_byte >> (self.next_bit as u8)) & 1) as u32;
        self.next_bit -= 1;
        bit
    }

    /// Read a boolean (Annex A.2.1): `read_bit() == 1`.
    pub fn read_bool(&mut self) -> bool {
        self.read_bit() == 1
    }

    /// Read an `n`-bit unsigned literal (Annex A.2.2), MSB-first.
    pub fn read_nbits(&mut self, n: u32) -> u32 {
        let mut val = 0u32;
        for _ in 0..n {
            val = (val << 1) | self.read_bit();
        }
        val
    }

    /// Read an `n`-byte unsigned integer literal (Annex A.2.3). This
    /// byte-aligns first.
    pub fn read_uint_lit(&mut self, n: u32) -> u32 {
        self.byte_align();
        self.read_nbits(8 * n)
    }

    /// Discard the rest of the current byte (Annex A.1.3).
    pub fn byte_align(&mut self) {
        if self.next_bit != 7 {
            // A full fresh byte has `next_bit == 7`; anything else means
            // we're partway through and need to load the next byte at
            // the next read.
            self.next_bit = -1;
        }
    }

    /// Decode an unsigned interleaved exp-Golomb value (Annex A.3.2).
    ///
    /// The code is a sequence of (follow, data) pairs terminated by a
    /// `1` follow bit. If the first follow bit is `1`, the value is 0.
    pub fn read_uint(&mut self) -> u32 {
        let mut value: u32 = 1;
        while self.read_bit() == 0 {
            value <<= 1;
            if self.read_bit() == 1 {
                value += 1;
            }
        }
        value - 1
    }

    /// Decode a signed interleaved exp-Golomb value (Annex A.3.3).
    pub fn read_sint(&mut self) -> i32 {
        let value = self.read_uint() as i32;
        if value != 0 && self.read_bit() == 1 {
            -value
        } else {
            value
        }
    }
}

/// Bounded bit reader used for arithmetic-coded blocks and for VLC
/// coding inside a fixed-size data unit (Annex A.3.1). Once `bits_left`
/// reaches zero, `read_bitb()` returns 1 by default so that any VLC
/// decoded beyond the block boundary falls out to zero.
pub struct BoundedBitReader<'a> {
    inner: BitReader<'a>,
    bits_left: u64,
}

impl<'a> BoundedBitReader<'a> {
    pub fn new(data: &'a [u8], bits_available: u64) -> Self {
        Self {
            inner: BitReader::new(data),
            bits_left: bits_available,
        }
    }

    pub fn bits_left(&self) -> u64 {
        self.bits_left
    }

    /// Spec's `read_bitb()`: default to 1 once the block is exhausted.
    pub fn read_bitb(&mut self) -> u32 {
        if self.bits_left == 0 {
            return 1;
        }
        self.bits_left -= 1;
        self.inner.read_bit()
    }

    /// Raw (unbounded) bit read, used by the arithmetic engine after
    /// init to consume the block byte-by-byte without defaulting.
    pub fn read_bit_raw(&mut self) -> u32 {
        if self.bits_left == 0 {
            return 0;
        }
        self.bits_left -= 1;
        self.inner.read_bit()
    }

    /// Spec's `flush_inputb()`: drop remaining block bits.
    pub fn flush(&mut self) {
        while self.bits_left > 0 {
            self.bits_left -= 1;
            let _ = self.inner.read_bit();
        }
    }

    pub fn read_uintb(&mut self) -> u32 {
        let mut value: u32 = 1;
        while self.read_bitb() == 0 {
            value <<= 1;
            if self.read_bitb() == 1 {
                value += 1;
            }
        }
        value - 1
    }

    pub fn read_sintb(&mut self) -> i32 {
        let value = self.read_uintb() as i32;
        if value != 0 && self.read_bitb() == 1 {
            -value
        } else {
            value
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a byte stream from a bit string like "10011011" (MSB
    /// first, padded with zeros to the next byte).
    fn bits(s: &str) -> Vec<u8> {
        let mut out = Vec::new();
        let mut cur = 0u8;
        let mut n = 0u8;
        for c in s.chars() {
            if c == ' ' {
                continue;
            }
            cur = (cur << 1) | if c == '1' { 1 } else { 0 };
            n += 1;
            if n == 8 {
                out.push(cur);
                cur = 0;
                n = 0;
            }
        }
        if n > 0 {
            out.push(cur << (8 - n));
        }
        out
    }

    #[test]
    fn read_bit_msb_first() {
        let data = [0b1011_0100];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bit(), 1);
        assert_eq!(r.read_bit(), 0);
        assert_eq!(r.read_bit(), 1);
        assert_eq!(r.read_bit(), 1);
        assert_eq!(r.read_bit(), 0);
        assert_eq!(r.read_bit(), 1);
        assert_eq!(r.read_bit(), 0);
        assert_eq!(r.read_bit(), 0);
    }

    #[test]
    fn exp_golomb_table_a1() {
        // From Table A.1 in the spec.
        let cases: &[(&str, u32)] = &[
            ("1", 0),
            ("001", 1),
            ("011", 2),
            ("00001", 3),
            ("00011", 4),
            ("01001", 5),
            ("01011", 6),
            ("0000001", 7),
            ("0000011", 8),
            ("0001001", 9),
        ];
        for (s, v) in cases {
            let data = bits(s);
            let mut r = BitReader::new(&data);
            assert_eq!(r.read_uint(), *v, "bits {s}");
        }
    }

    #[test]
    fn signed_exp_golomb_examples() {
        // From the spec's signed table.
        let cases: &[(&str, i32)] = &[
            ("000111", -4),
            ("0111", -2),
            ("0011", -1),
            ("1", 0),
            ("0010", 1),
            ("0110", 2),
            ("000010", 3),
        ];
        for (s, v) in cases {
            let data = bits(s);
            let mut r = BitReader::new(&data);
            assert_eq!(r.read_sint(), *v, "bits {s}");
        }
    }

    #[test]
    fn byte_align_skips_partial() {
        // Read two bits, align, read an 8-bit literal.
        let data = [0b1010_0000, 0x7A];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bit(), 1);
        assert_eq!(r.read_bit(), 0);
        r.byte_align();
        assert_eq!(r.read_nbits(8), 0x7A);
    }

    #[test]
    fn bounded_reader_defaults_to_one_past_block() {
        let data = [0b1000_0000];
        let mut r = BoundedBitReader::new(&data, 1);
        assert_eq!(r.read_bitb(), 1);
        assert_eq!(r.bits_left(), 0);
        // Past-end reads must return 1 (Dirac's "default to 1" rule).
        assert_eq!(r.read_bitb(), 1);
        assert_eq!(r.read_bitb(), 1);
        // And thus exp-Golomb decodes to 0.
        assert_eq!(r.read_uintb(), 0);
    }
}
