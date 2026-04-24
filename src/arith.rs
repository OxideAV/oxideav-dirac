//! Dirac binary arithmetic decoder (Annex B).
//!
//! Dirac uses a custom adaptive binary arithmetic coder with 16-bit
//! precision on `LOW` / `RANGE`, a 16-bit `CODE` register, and
//! per-context 16-bit probability-of-zero estimates. Unlike CABAC,
//! Dirac's coder is driven by a fixed 256-entry lookup table
//! (`PROB_LUT`) that maps the first 8 MSBs of the current probability
//! to an increment / decrement step.
//!
//! The decoder reads input from a bounded bit stream (Annex A.3.1) so
//! that past-end reads return 1 by default. The bit-stream size is
//! known in advance (Dirac carries a byte length in front of every
//! arithmetic-coded block).

/// Contex-probability lookup table. Indexed by the top 8 bits of a
/// 16-bit context probability. `PROB_LUT[i]` is the **decrement** added
/// to / subtracted from the probability when a 1 or 0 symbol is coded,
/// respectively. Values copied verbatim from Annex B Table B.1.
pub const PROB_LUT: [u16; 256] = [
    0, 2, 5, 8, 11, 15, 20, 24, 29, 35, 41, 47, 53, 60, 67, 74, 82, 89, 97, 106, 114, 123, 132,
    141, 150, 160, 170, 180, 190, 201, 211, 222, 233, 244, 256, 267, 279, 291, 303, 315, 327, 340,
    353, 366, 379, 392, 405, 419, 433, 447, 461, 475, 489, 504, 518, 533, 548, 563, 578, 593, 609,
    624, 640, 656, 672, 688, 705, 721, 738, 754, 771, 788, 805, 822, 840, 857, 875, 892, 910, 928,
    946, 964, 983, 1001, 1020, 1038, 1057, 1076, 1095, 1114, 1133, 1153, 1172, 1192, 1211, 1231,
    1251, 1271, 1291, 1311, 1332, 1352, 1373, 1393, 1414, 1435, 1456, 1477, 1498, 1520, 1541, 1562,
    1584, 1606, 1628, 1649, 1671, 1694, 1716, 1738, 1760, 1783, 1806, 1828, 1851, 1874, 1897, 1920,
    1935, 1942, 1949, 1955, 1961, 1968, 1974, 1980, 1985, 1991, 1996, 2001, 2006, 2011, 2016, 2021,
    2025, 2029, 2033, 2037, 2040, 2044, 2047, 2050, 2053, 2056, 2058, 2061, 2063, 2065, 2066, 2068,
    2069, 2070, 2071, 2072, 2072, 2072, 2072, 2072, 2072, 2071, 2070, 2069, 2068, 2066, 2065, 2063,
    2060, 2058, 2055, 2052, 2049, 2045, 2042, 2038, 2033, 2029, 2024, 2019, 2013, 2008, 2002, 1996,
    1989, 1982, 1975, 1968, 1960, 1952, 1943, 1934, 1925, 1916, 1906, 1896, 1885, 1874, 1863, 1851,
    1839, 1827, 1814, 1800, 1786, 1772, 1757, 1742, 1727, 1710, 1694, 1676, 1659, 1640, 1622, 1602,
    1582, 1561, 1540, 1518, 1495, 1471, 1447, 1422, 1396, 1369, 1341, 1312, 1282, 1251, 1219, 1186,
    1151, 1114, 1077, 1037, 995, 952, 906, 857, 805, 750, 690, 625, 553, 471, 376, 255,
];

/// A context probability is stored as a u16 whose value is the
/// probability of a 0 bit in the stream, scaled so that `0x8000` ≈ ½.
/// Contexts never hit 0 or 0xFFFF exactly — the update process clamps
/// away from the extremes, as B.2.1 notes.
pub type Probability = u16;

/// Each context in Dirac is identified by a label. We keep contexts in
/// a `Vec<Probability>` and look them up by index; decoders declare a
/// label->index mapping at init time.
#[derive(Debug, Clone)]
pub struct ContextBank {
    probs: Vec<Probability>,
}

impl ContextBank {
    /// Build a fresh context bank with `num_contexts` entries, each
    /// initialised to 0x8000 (½).
    pub fn new(num_contexts: usize) -> Self {
        Self {
            probs: vec![0x8000; num_contexts],
        }
    }

    pub fn len(&self) -> usize {
        self.probs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.probs.is_empty()
    }

    pub fn get(&self, ctx: usize) -> Probability {
        self.probs[ctx]
    }

    /// Update context `ctx` given that the symbol just decoded was
    /// `value` (Annex B.2.6). Decrements the prob-of-zero on a 1 bit,
    /// increments it on a 0 bit, by `PROB_LUT[prob0 >> 8]` (or
    /// `PROB_LUT[255 - (prob0 >> 8)]` for the positive update).
    pub fn update(&mut self, ctx: usize, value: bool) {
        let p = self.probs[ctx];
        let idx = (p >> 8) as usize;
        let p = p as i32;
        let new_p = if value {
            p - PROB_LUT[idx] as i32
        } else {
            p + PROB_LUT[255 - idx] as i32
        };
        // Clamp into [1, 0xFFFF]. The table is tuned so that neither
        // endpoint is reached in practice, but we guard against it.
        let clamped = new_p.clamp(1, 0xFFFF);
        self.probs[ctx] = clamped as u16;
    }
}

/// The Dirac arithmetic decoder. Operates on a byte-aligned compressed
/// block of known size. The block is fed in full at construction; past
/// the end, reads default to 1 (Annex A.3.1), which the decoder can
/// still consume even though no meaningful data remains.
///
/// Internally we use the "code_minus_low" efficient-implementation form
/// (Annex B.2.7.1) — tracking only `code - low` avoids one subtract per
/// boolean decode.
pub struct ArithDecoder<'a> {
    data: &'a [u8],
    /// Index of the next byte to consume into `code_minus_low`.
    byte_pos: usize,
    /// Remaining bits available in the block (Annex A `BITS_LEFT`).
    bits_left: u64,
    /// Shift register for the current byte being bit-consumed by
    /// `renormalise`. We read one byte at a time, then shift bits in
    /// MSB-first.
    current_byte: u8,
    /// Bit index within `current_byte`, 7..=0; -1 triggers a reload.
    next_bit: i8,

    range: u32,
    code_minus_low: u32,
}

impl<'a> ArithDecoder<'a> {
    /// Create and initialise a new decoder for a byte-aligned block of
    /// `length` bytes living at the start of `data`. Any bytes beyond
    /// `length` are ignored.
    pub fn new(data: &'a [u8], length: usize) -> Self {
        let eff = length.min(data.len());
        let mut dec = Self {
            data: &data[..eff],
            byte_pos: 0,
            bits_left: (eff as u64) * 8,
            current_byte: 0,
            next_bit: -1,
            range: 0xFFFF,
            code_minus_low: 0,
        };
        // B.2.2: clock 16 bits into CODE.
        for _ in 0..16 {
            dec.code_minus_low = (dec.code_minus_low << 1) | dec.read_bitb();
        }
        dec
    }

    /// Number of bytes consumed from the block so far (advances as
    /// `renormalise()` pulls fresh bits).
    pub fn byte_pos(&self) -> usize {
        self.byte_pos
    }

    /// Remaining unused bits in the block (defaults to 1 past the end).
    pub fn bits_left(&self) -> u64 {
        self.bits_left
    }

    /// Read one bit from the bounded block; returns 1 once the block is
    /// exhausted. MSB-first within each byte.
    fn read_bitb(&mut self) -> u32 {
        if self.bits_left == 0 {
            return 1;
        }
        self.bits_left -= 1;
        if self.next_bit < 0 {
            self.current_byte = self.data.get(self.byte_pos).copied().unwrap_or(0);
            self.byte_pos += 1;
            self.next_bit = 7;
        }
        let bit = ((self.current_byte >> (self.next_bit as u8)) & 1) as u32;
        self.next_bit -= 1;
        bit
    }

    /// B.2.5 renormalise (efficient form, B.2.7.1).
    fn renormalise(&mut self) {
        self.code_minus_low = (self.code_minus_low << 1) & 0xFFFF;
        self.range <<= 1;
        self.code_minus_low |= self.read_bitb();
    }

    /// Decode one boolean symbol against the probability in `bank[ctx]`
    /// (Annex B.2.4). Updates the context.
    pub fn read_bool(&mut self, bank: &mut ContextBank, ctx: usize) -> bool {
        let prob_zero = bank.get(ctx) as u32;
        let range_times_prob = (self.range * prob_zero) >> 16;
        let value = self.code_minus_low >= range_times_prob;
        if value {
            self.code_minus_low -= range_times_prob;
            self.range -= range_times_prob;
        } else {
            self.range = range_times_prob;
        }
        bank.update(ctx, value);
        while self.range <= 0x4000 {
            self.renormalise();
        }
        value
    }

    /// Decode an unsigned integer (Annex A.4.3.2) using the exp-Golomb
    /// binarisation but driving each bit through the arithmetic engine.
    ///
    /// `follow_ctxs` gives a list of contexts for the follow bits; the
    /// last entry is re-used for any further follow bits past the
    /// list's length. `data_ctx` is used for every data bit.
    pub fn read_uint(
        &mut self,
        bank: &mut ContextBank,
        follow_ctxs: &[usize],
        data_ctx: usize,
    ) -> u32 {
        let mut value: u32 = 1;
        let mut index = 0usize;
        loop {
            let ctx_pos = index.min(follow_ctxs.len() - 1);
            if self.read_bool(bank, follow_ctxs[ctx_pos]) {
                break;
            }
            value <<= 1;
            if self.read_bool(bank, data_ctx) {
                value += 1;
            }
            index += 1;
        }
        value - 1
    }

    /// Decode a signed integer (Annex A.4.3.3): magnitude + sign.
    pub fn read_sint(
        &mut self,
        bank: &mut ContextBank,
        follow_ctxs: &[usize],
        data_ctx: usize,
        sign_ctx: usize,
    ) -> i32 {
        let mag = self.read_uint(bank, follow_ctxs, data_ctx) as i32;
        if mag != 0 && self.read_bool(bank, sign_ctx) {
            -mag
        } else {
            mag
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_init_is_half() {
        let bank = ContextBank::new(4);
        assert_eq!(bank.get(0), 0x8000);
        assert_eq!(bank.get(3), 0x8000);
    }

    #[test]
    fn context_update_pushes_towards_biased_value() {
        let mut bank = ContextBank::new(1);
        // Coding a long run of 1s should pull prob-of-zero down.
        for _ in 0..50 {
            bank.update(0, true);
        }
        assert!(bank.get(0) < 0x4000);
        // Coding a long run of 0s in a fresh context should pull it up.
        let mut bank = ContextBank::new(1);
        for _ in 0..50 {
            bank.update(0, false);
        }
        assert!(bank.get(0) > 0xC000);
    }

    #[test]
    fn prob_lut_endpoints() {
        assert_eq!(PROB_LUT[0], 0);
        assert_eq!(PROB_LUT[255], 255);
        // Peak of the curve (2072) appears multiple times mid-table.
        assert!(PROB_LUT.contains(&2072));
        // Values rise monotonically from index 0 up to the peak region
        // and then fall monotonically; sanity-check both halves.
        assert!(PROB_LUT[64] > PROB_LUT[0]);
        assert!(PROB_LUT[128] > PROB_LUT[64]);
        assert!(PROB_LUT[200] < PROB_LUT[128]);
    }

    #[test]
    fn arith_decode_progresses_on_empty_past_end() {
        // With an empty block, read_bitb() returns 1 forever; confirm
        // that read_bool produces a stable stream of booleans without
        // panicking.
        let buf: [u8; 0] = [];
        let mut dec = ArithDecoder::new(&buf, 0);
        let mut bank = ContextBank::new(1);
        for _ in 0..16 {
            let _ = dec.read_bool(&mut bank, 0);
        }
    }
}
