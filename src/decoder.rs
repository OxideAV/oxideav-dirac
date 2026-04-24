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

use oxideav_codec::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Result,
    VideoFrame, VideoPlane,
};

use crate::picture::{
    decode_picture_with_refs, DecodedPicture, PictureError, ReferencePicture,
};
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
                        return Err(Error::invalid(format!(
                            "dirac: bad sequence header: {e}"
                        )));
                    }
                }
            } else if pi.is_picture() {
                self.pending.push_back(payload.clone());
                self.pending_codes.push_back(parse_code);
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
                    // §15.4 add to reference buffer if this is a
                    // reference picture.
                    if pi.is_reference() {
                        self.push_reference(&pic, &seq);
                    }
                    return Ok(Some(decoded_to_video_frame(&pic, &seq)));
                }
                Err(PictureError::MissingReference(_)) => {
                    // The reference buffer hasn't caught up to this
                    // inter picture — skip and continue.
                    self.pending.pop_front();
                    self.pending_codes.pop_front();
                    continue;
                }
                Err(PictureError::InterNotImplemented) => {
                    // Should no longer happen, but preserve the skip
                    // behaviour so partial bitstreams don't break.
                    self.pending.pop_front();
                    self.pending_codes.pop_front();
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
/// an oxideav-core `VideoFrame`. 8-bit depths use `Yuv*P` formats;
/// deeper bit depths (10/12-bit video) fall back to packing the low
/// byte of each sample into 8-bit planes for now — a follow-up will
/// produce `Yuv420P10Le` etc.
fn decoded_to_video_frame(pic: &DecodedPicture, seq: &SequenceHeader) -> VideoFrame {
    let format = match (seq.video_params.chroma_format, pic.luma_depth) {
        (ChromaFormat::Yuv420, d) if d <= 8 => PixelFormat::Yuv420P,
        (ChromaFormat::Yuv422, d) if d <= 8 => PixelFormat::Yuv422P,
        (ChromaFormat::Yuv444, d) if d <= 8 => PixelFormat::Yuv444P,
        // Higher bit depths: demote to 8 bits for now — better than
        // returning Unsupported, and lets the test rig look at the
        // actual decoded pattern.
        (ChromaFormat::Yuv420, _) => PixelFormat::Yuv420P,
        (ChromaFormat::Yuv422, _) => PixelFormat::Yuv422P,
        (ChromaFormat::Yuv444, _) => PixelFormat::Yuv444P,
    };
    let y = plane_from_i32(&pic.y, pic.luma_width, pic.luma_depth);
    let u = plane_from_i32(&pic.u, pic.chroma_width, pic.chroma_depth);
    let v = plane_from_i32(&pic.v, pic.chroma_width, pic.chroma_depth);
    VideoFrame {
        format,
        width: pic.luma_width as u32,
        height: pic.luma_height as u32,
        pts: None,
        time_base: oxideav_core::TimeBase::new(1, 25),
        planes: vec![y, u, v],
    }
}

fn plane_from_i32(values: &[i32], stride: usize, depth: u32) -> VideoPlane {
    // For depths > 8 we shift to fit into one byte per sample so the
    // downstream VideoFrame can still be 8-bit. 8-bit data pipes
    // through as-is.
    let shift = depth.saturating_sub(8);
    let mut data = Vec::with_capacity(values.len());
    for &v in values {
        let clamped = v.max(0).min(((1i64 << depth) - 1) as i32);
        let byte = (clamped >> shift) as u8;
        data.push(byte);
    }
    VideoPlane {
        stride,
        data,
    }
}
