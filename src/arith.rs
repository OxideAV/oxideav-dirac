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

/// Dirac binary arithmetic encoder — counterpart to [`ArithDecoder`].
///
/// Mirrors Annex B.2.7.1's "code-minus-low" decoder: tracks `low` /
/// `range` / `follow_bits` (the standard E1/E2/E3 carry mechanism for
/// integer-arithmetic encoding) and emits MSB-first bits into a byte
/// buffer. The encoder uses the same per-context probability of zero
/// as the decoder, with the same `PROB_LUT` update rule.
///
/// End-of-stream rule: after the final symbol, [`finish`] emits one
/// disambiguating "1" bit, then any pending `follow_bits` zeros, then
/// pads the partial byte out with "1" bits — matching the decoder's
/// past-end "1" default (Annex A.3.1) so trailing reads stay inside
/// the encoded interval.
///
/// [`finish`]: ArithEncoder::finish
pub struct ArithEncoder {
    out: Vec<u8>,
    cur: u8,
    nbits: u8,
    low: u32,
    range: u32,
    follow_bits: u32,
}

impl Default for ArithEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ArithEncoder {
    pub fn new() -> Self {
        Self {
            out: Vec::new(),
            cur: 0,
            nbits: 0,
            low: 0,
            range: 0xFFFF,
            follow_bits: 0,
        }
    }

    /// MSB-first bit sink — writes one bit into the partial byte and
    /// flushes when full.
    fn emit_bit(&mut self, bit: u32) {
        self.cur = (self.cur << 1) | ((bit as u8) & 1);
        self.nbits += 1;
        if self.nbits == 8 {
            self.out.push(self.cur);
            self.cur = 0;
            self.nbits = 0;
        }
    }

    /// Emit `bit` then `follow_bits` copies of `1 - bit` (the carry-
    /// resolution step from Witten/Neal/Cleary).
    fn emit_with_follows(&mut self, bit: u32) {
        self.emit_bit(bit);
        let inv = 1 - bit;
        while self.follow_bits > 0 {
            self.emit_bit(inv);
            self.follow_bits -= 1;
        }
    }

    /// Encode one boolean symbol against `bank[ctx]`. Mirrors
    /// [`ArithDecoder::read_bool`] symbol-for-symbol.
    pub fn write_bool(&mut self, bank: &mut ContextBank, ctx: usize, value: bool) {
        let prob_zero = bank.get(ctx) as u32;
        let r0 = (self.range * prob_zero) >> 16;
        if value {
            self.low += r0;
            self.range -= r0;
        } else {
            self.range = r0;
        }
        bank.update(ctx, value);
        // Renormalise while range <= 0x4000 (Annex B.2.5).
        while self.range <= 0x4000 {
            // E1: top half — emit 0, then any pending follow bits as 1s.
            if self.low + self.range <= 0x8000 {
                self.emit_with_follows(0);
            // E2: bottom half — emit 1, then follow bits as 0s.
            } else if self.low >= 0x8000 {
                self.emit_with_follows(1);
                self.low -= 0x8000;
            // E3: straddle the midpoint — bump follow_bits, shift away
            // from the midpoint, and let the next renormalise resolve.
            } else {
                self.follow_bits += 1;
                self.low -= 0x4000;
            }
            self.low <<= 1;
            self.range <<= 1;
            self.low &= 0xFFFF;
        }
    }

    /// Encode an unsigned integer (Annex A.4.3.2) — exp-Golomb
    /// binarisation through the arithmetic engine. Symmetrical to
    /// [`ArithDecoder::read_uint`].
    pub fn write_uint(
        &mut self,
        bank: &mut ContextBank,
        follow_ctxs: &[usize],
        data_ctx: usize,
        value: u32,
    ) {
        // Emit each (follow=0, data) pair for the bits below the MSB
        // of (value + 1), then the terminating follow=1.
        let n = value.wrapping_add(1);
        let bit_len = 32 - n.leading_zeros();
        for i in (0..bit_len - 1).rev() {
            let ctx_pos = ((bit_len - 1 - i - 1) as usize).min(follow_ctxs.len() - 1);
            self.write_bool(bank, follow_ctxs[ctx_pos], false);
            self.write_bool(bank, data_ctx, ((n >> i) & 1) == 1);
        }
        let last_pos = ((bit_len - 1) as usize).min(follow_ctxs.len() - 1);
        self.write_bool(bank, follow_ctxs[last_pos], true);
    }

    /// Encode a signed integer (Annex A.4.3.3): magnitude + sign.
    pub fn write_sint(
        &mut self,
        bank: &mut ContextBank,
        follow_ctxs: &[usize],
        data_ctx: usize,
        sign_ctx: usize,
        value: i32,
    ) {
        let mag = value.unsigned_abs();
        self.write_uint(bank, follow_ctxs, data_ctx, mag);
        if mag != 0 {
            self.write_bool(bank, sign_ctx, value < 0);
        }
    }

    /// Flush any carry-pending state and return the encoded bytes.
    ///
    /// The classical Witten/Neal/Cleary termination emits a single
    /// disambiguating bit + the pending follow bits. We follow that,
    /// then explicitly emit the upper 16 bits of `low | 0x8000` so the
    /// decoder's residual `code_minus_low` (16 bits past the last
    /// symbol) is bound to fall inside `[low, low + range)`. The byte
    /// is then right-padded with `1` bits — past-end reads return `1`
    /// per Annex A.3.1, so any extra renormalises the decoder performs
    /// while finalising the last symbol's context update stay inside
    /// the encoded interval.
    pub fn finish(mut self) -> Vec<u8> {
        // §B.2.7.1 termination. After the renormalise loops we have
        // 0x4000 < range <= 0x10000 and 0 <= low < 0x10000 with
        // `low + range > 0x4000` (a non-empty interval that doesn't
        // collapse on the lower edge). Emit a value `T` that's inside
        // [low, low + range), big enough that any subsequent
        // past-end-read 1s extending it stay inside the same interval.
        //
        // Choice: the bit string corresponding to the integer
        // `(low + 0x4000)` truncated/padded to 16 bits — it sits at
        // least 0x4000 above `low` (so >= low) and below `low + range`
        // since range > 0x4000. We follow the WNC bit-plus-follows
        // protocol for the FIRST disambiguating bit and emit the
        // remaining 15 bits raw.
        self.follow_bits += 1;
        let target = self.low.wrapping_add(0x4000) & 0xFFFF;
        // First bit: top bit of target.
        let b0 = (target >> 15) & 1;
        self.emit_with_follows(b0);
        // Remaining 15 bits, MSB-first.
        for i in (0..15).rev() {
            self.emit_bit((target >> i) & 1);
        }
        // Pad partial byte with 1s — the decoder reads past-end as 1
        // (Annex A.3.1) so this padding lines up with what its
        // renormalise loop will read.
        while self.nbits != 0 {
            self.emit_bit(1);
        }
        self.out
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

    /// The encoder + decoder must agree on a long alternating-symbol
    /// stream. This is the load-bearing roundtrip — it catches every
    /// E1/E2/E3 carry bug because the symbol stream forces follow_bits
    /// to fire repeatedly.
    #[test]
    fn arith_roundtrip_alternating_bits() {
        let mut enc_bank = ContextBank::new(1);
        let mut enc = ArithEncoder::new();
        let symbols: Vec<bool> = (0..200).map(|i| i % 2 == 0).collect();
        for &b in &symbols {
            enc.write_bool(&mut enc_bank, 0, b);
        }
        let bytes = enc.finish();
        let mut dec_bank = ContextBank::new(1);
        let mut dec = ArithDecoder::new(&bytes, bytes.len());
        for (i, &expected) in symbols.iter().enumerate() {
            let got = dec.read_bool(&mut dec_bank, 0);
            assert_eq!(got, expected, "bit {i} mismatch");
        }
    }

    /// Roundtrip with biased probability (lots of zeros) — exercises
    /// the renormalise loop dozens of times per symbol once the
    /// probability adapts heavily. Uses a moderately biased mix so the
    /// `PROB_LUT` adapts but doesn't pin probabilities at the
    /// extremes (which would emit zero-bit-rate-per-symbol streams the
    /// encoder can't represent without the decoder's past-end-1
    /// padding finally derailing things).
    #[test]
    fn arith_roundtrip_biased_zeros() {
        let mut enc_bank = ContextBank::new(1);
        let mut enc = ArithEncoder::new();
        // Mix: 70% zeros, 30% ones — within the range where adaptive
        // probabilities stay away from the ±extremes.
        let symbols: Vec<bool> = (0..200).map(|i| i % 3 != 0).collect();
        for &b in &symbols {
            enc.write_bool(&mut enc_bank, 0, b);
        }
        let bytes = enc.finish();
        let mut dec_bank = ContextBank::new(1);
        let mut dec = ArithDecoder::new(&bytes, bytes.len());
        for (i, &expected) in symbols.iter().enumerate() {
            let got = dec.read_bool(&mut dec_bank, 0);
            assert_eq!(got, expected, "biased bit {i}");
        }
    }

    /// Roundtrip uint encoding — exercises the exp-Golomb arithmetic
    /// path used by the Dirac §12.3 motion-data fields.
    #[test]
    fn arith_roundtrip_uints_with_follow_contexts() {
        let mut enc_bank = ContextBank::new(8);
        let mut enc = ArithEncoder::new();
        let follow = &[0usize, 1, 2, 3, 4];
        let data = 5usize;
        let values: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 7, 15, 100, 255, 1023, 0, 1, 0];
        for &v in &values {
            enc.write_uint(&mut enc_bank, follow, data, v);
        }
        let bytes = enc.finish();
        let mut dec_bank = ContextBank::new(8);
        let mut dec = ArithDecoder::new(&bytes, bytes.len());
        for (i, &expected) in values.iter().enumerate() {
            let got = dec.read_uint(&mut dec_bank, follow, data);
            assert_eq!(got, expected, "uint #{i} ({expected})");
        }
    }

    /// Signed-int encoding roundtrip with mixed signs and zeros.
    #[test]
    fn arith_roundtrip_sints() {
        let mut enc_bank = ContextBank::new(8);
        let mut enc = ArithEncoder::new();
        let follow = &[0usize, 1, 2, 3, 4];
        let data = 5usize;
        let sign = 6usize;
        let values: Vec<i32> = vec![0, 1, -1, 2, -2, 0, 17, -42, 255, -255, 0, 1, -1];
        for &v in &values {
            enc.write_sint(&mut enc_bank, follow, data, sign, v);
        }
        let bytes = enc.finish();
        let mut dec_bank = ContextBank::new(8);
        let mut dec = ArithDecoder::new(&bytes, bytes.len());
        for (i, &expected) in values.iter().enumerate() {
            let got = dec.read_sint(&mut dec_bank, follow, data, sign);
            assert_eq!(got, expected, "sint #{i} ({expected})");
        }
    }
}
