//! Dirac decoder front-end.
//!
//! State machine:
//! 1. Concatenate every incoming packet into a growing buffer.
//! 2. On each `receive_frame`, walk the buffer via
//!    [`crate::stream::DataUnitIter`].
//! 3. Sequence headers are parsed and cached.
//! 4. Intra pictures (LD, HQ, or core-syntax) are decoded to a
//!    `VideoFrame`.
//! 5. Core-syntax inter pictures are decoded by driving
//!    [`crate::picture::decode_picture_with_refs`] with the decoder's
//!    own reference-picture buffer (§15.4). Reference pictures are
//!    admitted on successful decode, oldest evicted when the buffer
//!    fills.

use oxideav_core::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Result, TimeBase, VideoFrame,
    VideoPlane,
};

use crate::picture::{decode_picture_with_refs, DecodedPicture, PictureError, ReferencePicture};
use crate::sequence::{parse_sequence_header, SequenceHeader};
use crate::stream::DataUnitIter;
use crate::video_format::ChromaFormat;

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(DiracDecoder::new(params.codec_id.clone())))
}

/// Decoder scaffold. See module docs.
pub struct DiracDecoder {
    codec_id: CodecId,
    buffer: Vec<u8>,
    last_sequence: Option<SequenceHeader>,
    /// Picture data units pending decode; as `receive_frame` is called
    /// we pop the front one.
    pending: std::collections::VecDeque<Vec<u8>>,
    /// Parse codes for each pending payload.
    pending_codes: std::collections::VecDeque<u8>,
    /// `pts` of the most recent packet that appended the byte range
    /// covering this picture. Dirac packets are typically one data
    /// unit; on the assumption that the container split the stream at
    /// data-unit boundaries we hand the packet's `pts` to the next
    /// frame pulled out.
    pending_pts: std::collections::VecDeque<Option<i64>>,
    /// `time_base` paired with each `pending_pts` entry.
    pending_time_base: std::collections::VecDeque<TimeBase>,
    /// PTS + time_base carried by the most recent `send_packet` call,
    /// so `scan()` (which runs after the append) can tag any newly
    /// discovered data units with the right metadata.
    last_packet_pts: Option<i64>,
    last_packet_time_base: TimeBase,
    eof: bool,
    /// How far into `buffer` we've already scanned; used so we don't
    /// re-parse units after calling `scan()` repeatedly.
    scan_cursor: usize,
    /// §15.4 reference picture buffer. Populated with reference
    /// pictures (parse code has bits 2,3 set) after each successful
    /// decode, and drained in FIFO order when full.
    reference_buffer: Vec<ReferencePicture>,
    /// Upper bound on the reference buffer — Annex D specifies this
    /// per profile / level; conservative default 4 covers all profiles
    /// we currently decode.
    max_ref_buffer: usize,
}

impl DiracDecoder {
    pub fn new(codec_id: CodecId) -> Self {
        Self {
            codec_id,
            buffer: Vec::new(),
            last_sequence: None,
            pending: std::collections::VecDeque::new(),
            pending_codes: std::collections::VecDeque::new(),
            pending_pts: std::collections::VecDeque::new(),
            pending_time_base: std::collections::VecDeque::new(),
            last_packet_pts: None,
            last_packet_time_base: TimeBase::new(1, 25),
            eof: false,
            scan_cursor: 0,
            reference_buffer: Vec::new(),
            max_ref_buffer: 4,
        }
    }

    /// The most recently parsed sequence header, if any. Tests and
    /// higher-level tooling (the CLI probe) can consult this after
    /// feeding a few packets in.
    pub fn last_sequence(&self) -> Option<&SequenceHeader> {
        self.last_sequence.as_ref()
    }

    /// Walk any new bytes appended to the buffer. We remember how far
    /// we've walked so subsequent calls don't reprocess old units.
    fn scan(&mut self) -> Result<()> {
        let start = self.scan_cursor;
        // Snapshot each unit (pi + payload bytes) before processing so
        // we can mutate self after the walker's borrow ends.
        let snap: Vec<(u8, usize, Vec<u8>)> = DataUnitIter::new(&self.buffer[start..])
            .map(|u| (u.parse_info.parse_code, u.pi_offset, u.payload.to_vec()))
            .collect();
        for (parse_code, pi_offset, payload) in snap {
            let pi = crate::parse_info::ParseInfo {
                parse_code,
                next_parse_offset: 0,
                previous_parse_offset: 0,
            };
            if pi.is_seq_header() {
                match parse_sequence_header(&payload) {
                    Ok(sh) => {
                        self.last_sequence = Some(sh);
                    }
                    Err(e) => {
                        return Err(Error::invalid(format!("dirac: bad sequence header: {e}")));
                    }
                }
            } else if pi.is_picture() {
                self.pending.push_back(payload.clone());
                self.pending_codes.push_back(parse_code);
                self.pending_pts.push_back(self.last_packet_pts);
                self.pending_time_base.push_back(self.last_packet_time_base);
            }
            let payload_end = start + pi_offset + 13 + payload.len();
            self.scan_cursor = payload_end.max(self.scan_cursor);
        }
        Ok(())
    }

    fn try_decode_next(&mut self) -> Result<Option<VideoFrame>> {
        loop {
            let payload = match self.pending.front() {
                Some(p) => p.clone(),
                None => return Ok(None),
            };
            let code = self.pending_codes.front().copied().unwrap_or(0);
            let seq = match self.last_sequence.as_ref() {
                Some(s) => s.clone(),
                None => {
                    // Drop pictures that arrive before any seq header.
                    self.pending.pop_front();
                    self.pending_codes.pop_front();
                    self.pending_pts.pop_front();
                    self.pending_time_base.pop_front();
                    continue;
                }
            };
            let pi = crate::parse_info::ParseInfo {
                parse_code: code,
                next_parse_offset: 0,
                previous_parse_offset: 0,
            };
            match decode_picture_with_refs(&payload, pi, &seq, &self.reference_buffer) {
                Ok(pic) => {
                    self.pending.pop_front();
                    self.pending_codes.pop_front();
                    let pkt_pts = self.pending_pts.pop_front().flatten();
                    let pkt_tb = self
                        .pending_time_base
                        .pop_front()
                        .unwrap_or_else(|| TimeBase::new(1, 25));
                    // §15.4 add to reference buffer if this is a
                    // reference picture.
                    if pi.is_reference() {
                        self.push_reference(&pic, &seq);
                    }
                    // Prefer the sequence header's frame rate (§10.3.5)
                    // as the timebase, falling back to the container's
                    // when the header didn't override it to a real rate.
                    // If the packet arrived with an explicit pts, carry
                    // it; otherwise derive from picture_number.
                    let tb = time_base_from_frame_rate(&seq);
                    let effective_tb = if seq.video_params.frame_rate_numer > 0 {
                        tb
                    } else {
                        pkt_tb
                    };
                    let effective_pts = pkt_pts.or(Some(pic.picture_number as i64));
                    return Ok(Some(decoded_to_video_frame(
                        &pic,
                        &seq,
                        effective_pts,
                        effective_tb,
                    )));
                }
                Err(PictureError::MissingReference(_)) => {
                    // The reference buffer hasn't caught up to this
                    // inter picture — skip and continue.
                    self.pending.pop_front();
                    self.pending_codes.pop_front();
                    self.pending_pts.pop_front();
                    self.pending_time_base.pop_front();
                    continue;
                }
                Err(PictureError::InterNotImplemented) => {
                    // Should no longer happen, but preserve the skip
                    // behaviour so partial bitstreams don't break.
                    self.pending.pop_front();
                    self.pending_codes.pop_front();
                    self.pending_pts.pop_front();
                    self.pending_time_base.pop_front();
                    continue;
                }
                Err(PictureError::CoreSyntaxNotImplemented) => {
                    return Err(Error::unsupported(
                        "dirac decoder: unsupported core-syntax parse code",
                    ));
                }
                Err(e) => {
                    return Err(Error::invalid(format!("dirac: picture decode: {e}")));
                }
            }
        }
    }

    fn push_reference(&mut self, pic: &DecodedPicture, seq: &SequenceHeader) {
        // Store a **pre-output-offset, clipped** copy: the decoded
        // picture we produce has already been offset for output, so we
        // subtract the offset here. The `i32` payload stays in
        // `[-2^(depth-1), 2^(depth-1) - 1]`.
        let luma_half = if seq.luma_depth == 0 {
            0
        } else {
            1i32 << (seq.luma_depth - 1)
        };
        let chroma_half = if seq.chroma_depth == 0 {
            0
        } else {
            1i32 << (seq.chroma_depth - 1)
        };
        let y: Vec<i32> = pic.y.iter().map(|v| v - luma_half).collect();
        let u: Vec<i32> = pic.u.iter().map(|v| v - chroma_half).collect();
        let v: Vec<i32> = pic.v.iter().map(|v| v - chroma_half).collect();
        let rp = ReferencePicture {
            picture_number: pic.picture_number,
            luma_width: pic.luma_width,
            luma_height: pic.luma_height,
            chroma_width: pic.chroma_width,
            chroma_height: pic.chroma_height,
            y,
            u,
            v,
        };
        self.reference_buffer.push(rp);
        while self.reference_buffer.len() > self.max_ref_buffer {
            self.reference_buffer.remove(0);
        }
    }
}

impl Decoder for DiracDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        // Stash the packet's metadata so `scan()` can label any newly
        // discovered picture data units with this `pts` / `time_base`.
        self.last_packet_pts = packet.pts;
        self.last_packet_time_base = packet.time_base;
        self.buffer.extend_from_slice(&packet.data);
        self.scan()?;
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if self.last_sequence.is_none() && self.pending.is_empty() {
            return Err(Error::NeedMore);
        }
        match self.try_decode_next()? {
            Some(vf) => Ok(Frame::Video(vf)),
            None => {
                if self.eof {
                    Err(Error::Eof)
                } else {
                    Err(Error::NeedMore)
                }
            }
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        self.scan()
    }
}

/// Map a decoded Dirac picture (Y/U/V as `Vec<i32>` 0..2^depth) into
/// an oxideav-core `VideoFrame`.
///
/// * 8-bit components use the packed `Yuv*P` formats.
/// * 9- and 10-bit components use the little-endian 16-bit
///   `Yuv*P10Le` formats (the stored sample is in the low `depth` bits
///   of each 16-bit word, following the oxideav-core convention).
/// * 11- and 12-bit 4:2:0 components use `Yuv420P12Le`. 11/12-bit
///   4:2:2 / 4:4:4 fall back to `Yuv*P10Le` and clip to 10 bits for
///   now — oxideav-core has no `Yuv422P12Le` / `Yuv444P12Le` variants
///   at the moment, so clipping keeps us lossless below ~2^10.
///
/// §15.10 already pre-offsets each sample by `2^(bit_depth-1)` to make
/// it non-negative, so `pic.y / u / v` values are in `[0, 2^depth - 1]`.
/// This function only repackages them for the downstream buffer.
fn decoded_to_video_frame(
    pic: &DecodedPicture,
    seq: &SequenceHeader,
    pts: Option<i64>,
    time_base: TimeBase,
) -> VideoFrame {
    // Pick the storage format from the chroma sampling and the luma
    // bit-depth. We assume luma_depth == chroma_depth in practice for
    // the formats oxideav-core exposes today; when they disagree we
    // conservatively key off luma, as that's the visible component.
    let (format, store_depth) = match (seq.video_params.chroma_format, pic.luma_depth) {
        (ChromaFormat::Yuv420, d) if d <= 8 => (PixelFormat::Yuv420P, 8),
        (ChromaFormat::Yuv422, d) if d <= 8 => (PixelFormat::Yuv422P, 8),
        (ChromaFormat::Yuv444, d) if d <= 8 => (PixelFormat::Yuv444P, 8),
        (ChromaFormat::Yuv420, d) if d <= 10 => (PixelFormat::Yuv420P10Le, 10),
        (ChromaFormat::Yuv422, d) if d <= 10 => (PixelFormat::Yuv422P10Le, 10),
        (ChromaFormat::Yuv444, d) if d <= 10 => (PixelFormat::Yuv444P10Le, 10),
        // 12-bit path (only 4:2:0 exists in core today). For 4:2:2 /
        // 4:4:4 above 10 bits we clip to 10; it's still better than
        // the old 8-bit demotion.
        (ChromaFormat::Yuv420, _) => (PixelFormat::Yuv420P12Le, 12),
        (ChromaFormat::Yuv422, _) => (PixelFormat::Yuv422P10Le, 10),
        (ChromaFormat::Yuv444, _) => (PixelFormat::Yuv444P10Le, 10),
    };
    let y = plane_from_i32(&pic.y, pic.luma_width, pic.luma_depth, store_depth);
    let u = plane_from_i32(&pic.u, pic.chroma_width, pic.chroma_depth, store_depth);
    let v = plane_from_i32(&pic.v, pic.chroma_width, pic.chroma_depth, store_depth);
    VideoFrame {
        format,
        width: pic.luma_width as u32,
        height: pic.luma_height as u32,
        pts,
        time_base,
        planes: vec![y, u, v],
    }
}

/// Repack a signed-int sample vector (already offset into
/// `[0, 2^source_depth - 1]` per §15.10) into the byte buffer of a
/// `VideoPlane` at the target storage depth.
///
/// * `store_depth == 8` writes one byte per sample, right-shifting
///   high-bit-depth samples so the top 8 bits survive.
/// * `store_depth == 10` or `12` writes two bytes per sample in
///   little-endian order, with the sample in the low `store_depth`
///   bits of each 16-bit word (oxideav-core convention; see
///   [`PixelFormat::Yuv420P10Le`] docs).
///
/// The `stride` we return is the byte stride of one row — `width` for
/// 8-bit formats and `2 * width` for 10/12-bit formats, since each
/// sample occupies two bytes.
fn plane_from_i32(values: &[i32], width: usize, source_depth: u32, store_depth: u32) -> VideoPlane {
    match store_depth {
        8 => {
            // Source might be >8 bits; right-shift the excess so the
            // top bits survive.
            let shift = source_depth.saturating_sub(8);
            let max_src = if source_depth == 0 {
                0
            } else {
                ((1u64 << source_depth) - 1) as i32
            };
            let mut data = Vec::with_capacity(values.len());
            for &v in values {
                let clamped = v.clamp(0, max_src);
                data.push((clamped >> shift) as u8);
            }
            VideoPlane {
                stride: width,
                data,
            }
        }
        10 | 12 => {
            // Store one 16-bit LE sample per coefficient, masking to
            // `store_depth` bits. If the source is wider than the
            // storage width, right-shift by the difference (e.g.
            // 12-bit source into a 10-bit plane — clips the two low
            // bits). If source is narrower, left-shift the sample
            // into the top of the field, as is conventional for
            // mixing unlike-precision samples (e.g. 8-bit input into
            // a 10-bit plane becomes val << 2).
            let max_store = (1u32 << store_depth) - 1;
            let mut data = Vec::with_capacity(values.len() * 2);
            let (lshift, rshift) = if store_depth >= source_depth {
                (store_depth - source_depth, 0u32)
            } else {
                (0u32, source_depth - store_depth)
            };
            for &v in values {
                let u = v.max(0) as u32;
                let shifted = if rshift > 0 { u >> rshift } else { u << lshift };
                let masked = (shifted & max_store) as u16;
                data.extend_from_slice(&masked.to_le_bytes());
            }
            VideoPlane {
                stride: width * 2,
                data,
            }
        }
        _ => unreachable!("unexpected store_depth {store_depth}"),
    }
}

/// Derive a `TimeBase` whose `den / num` equals the sequence header's
/// frame rate (§10.3.5). Used when the container-level timebase in the
/// incoming packet is either unset or the trivial "1/25" default we
/// can't trust.
fn time_base_from_frame_rate(seq: &SequenceHeader) -> TimeBase {
    // §10.3.5: frame rate = NUMER / DENOM. A ticks-per-frame clock with
    // that frame rate uses num = DENOM, den = NUMER (so each tick = one
    // picture).
    let n = seq.video_params.frame_rate_numer.max(1) as i64;
    let d = seq.video_params.frame_rate_denom.max(1) as i64;
    TimeBase::new(d, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::{ParseParameters, PictureCodingMode, VideoParams};
    use crate::video_format::{ChromaFormat, ScanFormat, SignalRange};

    /// 8-bit samples are written one byte per coefficient with the
    /// stride equal to the plane width.
    #[test]
    fn plane_from_i32_8bit_passthrough() {
        let values = [0i32, 127, 128, 255];
        let p = plane_from_i32(&values, 4, 8, 8);
        assert_eq!(p.stride, 4);
        assert_eq!(p.data, vec![0, 127, 128, 255]);
    }

    /// An 8-bit source rendered into a 10-bit plane is left-shifted by
    /// 2 so the MSBs line up (§15.10 leaves the sample in the low
    /// `bit_depth` bits — the convention for mixing with a deeper
    /// storage plane is to shift the whole field up).
    #[test]
    fn plane_from_i32_8bit_to_10bit_left_shifts() {
        let values = [0i32, 255];
        let p = plane_from_i32(&values, 2, 8, 10);
        assert_eq!(p.stride, 4); // 2 bytes per 16-bit sample
                                 // 0u16 -> 0x00, 0x00
                                 // 255 << 2 = 1020 = 0x03FC -> 0xFC, 0x03
        assert_eq!(p.data, vec![0x00, 0x00, 0xFC, 0x03]);
    }

    /// A native 10-bit source passes straight through, one little-endian
    /// 16-bit word per sample, masked to 10 bits.
    #[test]
    fn plane_from_i32_10bit_native_packs_le16() {
        let values = [0i32, 1023, 512, 1];
        let p = plane_from_i32(&values, 4, 10, 10);
        assert_eq!(p.stride, 8);
        assert_eq!(p.data, vec![0x00, 0x00, 0xFF, 0x03, 0x00, 0x02, 0x01, 0x00]);
    }

    /// A 12-bit source clipped down into a 10-bit plane right-shifts
    /// by 2 (drops the two least-significant bits).
    #[test]
    fn plane_from_i32_12bit_to_10bit_right_shifts() {
        let values = [0i32, 4095, 1024];
        let p = plane_from_i32(&values, 3, 12, 10);
        assert_eq!(p.stride, 6);
        // 4095 >> 2 = 1023 = 0x3FF -> 0xFF, 0x03
        // 1024 >> 2 =  256 = 0x100 -> 0x00, 0x01
        assert_eq!(p.data, vec![0x00, 0x00, 0xFF, 0x03, 0x00, 0x01]);
    }

    /// 12-bit storage writes two bytes per sample with the sample
    /// masked to 12 bits.
    #[test]
    fn plane_from_i32_12bit_native_packs_le16() {
        let values = [0i32, 4095, 2048];
        let p = plane_from_i32(&values, 3, 12, 12);
        assert_eq!(p.stride, 6);
        assert_eq!(p.data, vec![0x00, 0x00, 0xFF, 0x0F, 0x00, 0x08]);
    }

    /// Out-of-range negative values are clamped to zero, not
    /// reinterpreted as large positives. §15.9 already clips upstream,
    /// but the output path must be defensive.
    #[test]
    fn plane_from_i32_clamps_negative_to_zero() {
        let values = [-4i32, -1, 0];
        let p = plane_from_i32(&values, 3, 10, 10);
        assert_eq!(p.stride, 6);
        assert_eq!(p.data, vec![0, 0, 0, 0, 0, 0]);
    }

    fn fake_sequence(rate_n: u32, rate_d: u32, depth: u32) -> SequenceHeader {
        SequenceHeader {
            parse_parameters: ParseParameters {
                version_major: 2,
                version_minor: 2,
                profile: 3,
                level: 0,
            },
            base_video_format_index: 0,
            video_params: VideoParams {
                frame_width: 64,
                frame_height: 64,
                chroma_format: ChromaFormat::Yuv420,
                source_sampling: ScanFormat::Progressive,
                top_field_first: true,
                frame_rate_numer: rate_n,
                frame_rate_denom: rate_d,
                pixel_aspect_ratio_numer: 1,
                pixel_aspect_ratio_denom: 1,
                clean_width: 64,
                clean_height: 64,
                clean_left_offset: 0,
                clean_top_offset: 0,
                signal_range: SignalRange {
                    luma_offset: 0,
                    luma_excursion: (1u32 << depth) - 1,
                    chroma_offset: 1u32 << (depth - 1),
                    chroma_excursion: (1u32 << depth) - 1,
                },
            },
            picture_coding_mode: PictureCodingMode::Frames,
            luma_width: 64,
            luma_height: 64,
            chroma_width: 32,
            chroma_height: 32,
            luma_depth: depth,
            chroma_depth: depth,
        }
    }

    /// §10.3.5 table 10.3 entry 3: 25 / 1 fps. The decoder's timebase
    /// is the inverse: 1 tick per frame => num=1, den=25.
    #[test]
    fn time_base_matches_frame_rate_25fps() {
        let seq = fake_sequence(25, 1, 8);
        let tb = time_base_from_frame_rate(&seq);
        assert_eq!(tb.as_rational().num, 1);
        assert_eq!(tb.as_rational().den, 25);
    }

    /// §10.3.5 table 10.3 entry 1: 24000/1001 fps NTSC-film rate. The
    /// inverse — the picture-tick duration — is 1001/24000.
    #[test]
    fn time_base_matches_frame_rate_ntsc_film() {
        let seq = fake_sequence(24000, 1001, 8);
        let tb = time_base_from_frame_rate(&seq);
        assert_eq!(tb.as_rational().num, 1001);
        assert_eq!(tb.as_rational().den, 24000);
    }

    /// Zero numerator would produce a degenerate tick — the helper
    /// clamps it upwards to 1 so the returned timebase is still valid.
    #[test]
    fn time_base_zero_rate_gets_safe_default() {
        let seq = fake_sequence(0, 0, 8);
        let tb = time_base_from_frame_rate(&seq);
        assert_eq!(tb.as_rational().num, 1);
        assert_eq!(tb.as_rational().den, 1);
    }

    /// A 10-bit 4:2:2 stream should emit a Yuv422P10Le frame with
    /// stride `2 * width` per plane and a time_base lifted from the
    /// sequence header (50 fps here).
    #[test]
    fn decoded_to_video_frame_10bit_422_picks_p10le() {
        let mut seq = fake_sequence(50, 1, 10);
        seq.video_params.chroma_format = ChromaFormat::Yuv422;
        seq.chroma_width = 32;
        seq.chroma_height = 64;
        let pic = DecodedPicture {
            picture_number: 7,
            luma_width: 64,
            luma_height: 64,
            chroma_width: 32,
            chroma_height: 64,
            y: vec![512; 64 * 64],
            u: vec![256; 32 * 64],
            v: vec![768; 32 * 64],
            luma_depth: 10,
            chroma_depth: 10,
        };
        let tb = time_base_from_frame_rate(&seq);
        let frame = decoded_to_video_frame(&pic, &seq, Some(42), tb);
        assert_eq!(frame.format, PixelFormat::Yuv422P10Le);
        assert_eq!(frame.pts, Some(42));
        assert_eq!(frame.time_base.as_rational().num, 1);
        assert_eq!(frame.time_base.as_rational().den, 50);
        // Each sample occupies two bytes.
        assert_eq!(frame.planes[0].stride, 128);
        assert_eq!(frame.planes[0].data.len(), 64 * 64 * 2);
        assert_eq!(frame.planes[1].stride, 64);
        assert_eq!(frame.planes[1].data.len(), 32 * 64 * 2);
        // First Y sample is 512 = 0x200 -> 0x00, 0x02 in little-endian.
        assert_eq!(frame.planes[0].data[0], 0x00);
        assert_eq!(frame.planes[0].data[1], 0x02);
    }

    /// A 12-bit 4:2:0 stream picks Yuv420P12Le; samples are packed as
    /// 12-bit LE. oxideav-core has no 12-bit 4:2:2 / 4:4:4, so those
    /// paths clip down to 10 — covered separately.
    #[test]
    fn decoded_to_video_frame_12bit_420_picks_p12le() {
        let seq = fake_sequence(25, 1, 12);
        let pic = DecodedPicture {
            picture_number: 0,
            luma_width: 64,
            luma_height: 64,
            chroma_width: 32,
            chroma_height: 32,
            y: vec![2048; 64 * 64],
            u: vec![1024; 32 * 32],
            v: vec![3072; 32 * 32],
            luma_depth: 12,
            chroma_depth: 12,
        };
        let tb = time_base_from_frame_rate(&seq);
        let frame = decoded_to_video_frame(&pic, &seq, None, tb);
        assert_eq!(frame.format, PixelFormat::Yuv420P12Le);
        assert_eq!(frame.planes[0].stride, 128);
        // 2048 = 0x800 -> 0x00, 0x08 in little-endian.
        assert_eq!(frame.planes[0].data[0], 0x00);
        assert_eq!(frame.planes[0].data[1], 0x08);
    }

    /// A pure 8-bit stream emits an 8-bit planar frame with
    /// stride == width (one byte per sample) — a regression guard so
    /// the existing ffmpeg_interop decoder_produces_first_frame test
    /// keeps agreeing with our choice of PixelFormat::Yuv444P there.
    #[test]
    fn decoded_to_video_frame_8bit_preserves_byte_stride() {
        let mut seq = fake_sequence(25, 1, 8);
        seq.video_params.chroma_format = ChromaFormat::Yuv444;
        seq.chroma_width = 64;
        seq.chroma_height = 64;
        let pic = DecodedPicture {
            picture_number: 0,
            luma_width: 64,
            luma_height: 64,
            chroma_width: 64,
            chroma_height: 64,
            y: vec![128; 64 * 64],
            u: vec![64; 64 * 64],
            v: vec![200; 64 * 64],
            luma_depth: 8,
            chroma_depth: 8,
        };
        let tb = time_base_from_frame_rate(&seq);
        let frame = decoded_to_video_frame(&pic, &seq, None, tb);
        assert_eq!(frame.format, PixelFormat::Yuv444P);
        assert_eq!(frame.planes[0].stride, 64);
        assert_eq!(frame.planes[0].data.len(), 64 * 64);
        assert_eq!(frame.planes[0].data[0], 128);
    }
}
