//! Dirac decoder front-end (foundation pass).
//!
//! At this stage the decoder parses parse-info framing and sequence
//! headers but does **not** produce decoded pixels — the wavelet
//! transform and coefficient unpacking are future work. Any picture
//! data unit currently returns `Error::Unsupported` so callers can
//! treat this crate as "probe + parse" while the heavy lifting lands
//! in subsequent commits.
//!
//! The decoder's state machine:
//! 1. Concatenate every incoming packet into a growing buffer.
//! 2. On each `receive_frame`, walk the buffer via
//!    [`crate::stream::DataUnitIter`].
//! 3. Sequence headers are parsed and cached.
//! 4. Pictures return `Unsupported` with a descriptive message naming
//!    which feature is missing (wavelet transform, arithmetic
//!    coefficient decode, motion compensation, …).

use oxideav_codec::Decoder;
use oxideav_core::{CodecId, CodecParameters, Error, Frame, Packet, Result};

use crate::sequence::{parse_sequence_header, SequenceHeader};
use crate::stream::DataUnitIter;

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(DiracDecoder::new(params.codec_id.clone())))
}

/// Minimal Dirac decoder scaffold. See module docs for the current
/// capability surface.
pub struct DiracDecoder {
    codec_id: CodecId,
    buffer: Vec<u8>,
    last_sequence: Option<SequenceHeader>,
    eof: bool,
}

impl DiracDecoder {
    pub fn new(codec_id: CodecId) -> Self {
        Self {
            codec_id,
            buffer: Vec::new(),
            last_sequence: None,
            eof: false,
        }
    }

    /// The most recently parsed sequence header, if any. Tests and
    /// higher-level tooling (the CLI probe) can consult this after
    /// feeding a few packets in.
    pub fn last_sequence(&self) -> Option<&SequenceHeader> {
        self.last_sequence.as_ref()
    }

    /// Walk the accumulated buffer, updating `last_sequence` for every
    /// sequence header seen. Returns `Err` on a malformed sequence
    /// header; returns `Ok(())` with no output on pictures (currently
    /// unimplemented).
    fn scan(&mut self) -> Result<()> {
        // Walk in-place; we don't consume bytes yet because we don't
        // produce frames. A future commit will advance the buffer
        // cursor past fully-decoded units.
        for unit in DataUnitIter::new(&self.buffer) {
            if unit.parse_info.is_seq_header() {
                match parse_sequence_header(unit.payload) {
                    Ok(sh) => {
                        self.last_sequence = Some(sh);
                    }
                    Err(e) => {
                        return Err(Error::invalid(format!(
                            "dirac: bad sequence header: {e}"
                        )));
                    }
                }
            }
        }
        Ok(())
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
        // Pictures are not yet decoded. Defer with a descriptive error;
        // higher layers (CLI probe, job graph) can detect this and skip
        // past it rather than aborting.
        if self.last_sequence.is_none() {
            return Err(Error::NeedMore);
        }
        Err(Error::unsupported(
            "dirac decoder: picture decode not yet implemented \
             (foundation pass — parse-info + sequence header only)",
        ))
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        self.scan()
    }
}
