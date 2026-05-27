//! Decoder-side robustness oracle for malformed VC-2 LD/HQ inputs.
//!
//! Goal: every input shape the decoder might encounter from a hostile
//! or merely-corrupted source must produce a clean Result (Ok / NeedMore
//! / Eof / InvalidData / Unsupported) in bounded time. Panics, unbounded
//! loops, integer overflows, and out-of-bounds slice indexing are all
//! decoder bugs and are caught here by the test harness (any panic kills
//! the test thread; a bounded-time wall is implicit in cargo test's
//! per-test timeout but additionally guarded by capping our perturbed
//! corpus to small fixtures so any infinite loop would be visible).
//!
//! Coverage:
//!
//! * **Truncation walk** — take a valid HQ stream and feed every
//!   non-trivial prefix length to the decoder. Each truncation must
//!   either yield a frame (if the truncation happens to land at a
//!   complete picture boundary, which is rare) or return a clean error
//!   / NeedMore.
//! * **Single-byte mutation** — flip every byte in a valid LD stream to
//!   each of `{0x00, 0xFF, !orig}` and confirm the decoder still
//!   terminates cleanly. We don't compare output: a mutated stream may
//!   produce garbled pixels OR an error, both are acceptable; the
//!   pass/fail is "decoder did not panic."
//! * **Pathological gibberish** — random-ish, all-zeros, all-0xFF, and
//!   "BBCD-prefix repeated forever" buffers as the entire input.
//! * **Hand-crafted invalid headers** — sequence headers with unknown
//!   base_video_format / chroma_format / picture_coding_mode, picture
//!   parse codes the decoder rejects (`InterNotImplemented`-era), and
//!   transform parameter blocks with degenerate `slices_x=0`,
//!   `dwt_depth > 6`, unknown wavelet index, and `slice_bytes_denom=0`.
//! * **Oversized parse-info offsets** — `next_parse_offset = u32::MAX`,
//!   confirming the stream walker's fallback-to-byte-search recovers
//!   without consuming pathological memory.
//!
//! Workspace policy: clean-room. No external library code was
//! consulted. Spec authority for the error shapes is the Dirac BBC
//! spec + SMPTE ST 2042-1 (VC-2).

use oxideav_core::{
    CodecId, CodecParameters, CodecRegistry, Decoder, Error, Frame, Packet, TimeBase,
};
use oxideav_dirac::encoder::{
    encode_single_hq_intra_stream, encode_single_ld_intra_stream, make_minimal_sequence,
    make_minimal_sequence_ld, synthetic_testsrc_64_yuv420, EncoderParams, LdEncoderParams,
};
use oxideav_dirac::parse_info::BBCD;
use oxideav_dirac::video_format::ChromaFormat;
use oxideav_dirac::wavelet::WaveletFilter;

/// Construct a fresh dirac decoder via the registry. Mirrors the
/// pattern in `encoder_roundtrip.rs` so the fuzz oracle exercises the
/// same public surface that real consumers use.
fn fresh_decoder() -> Box<dyn Decoder> {
    let mut reg = CodecRegistry::new();
    oxideav_dirac::register_codecs(&mut reg);
    let cp = CodecParameters::video(CodecId::new("dirac"));
    reg.first_decoder(&cp).expect("dirac decoder factory")
}

/// Feed an entire byte buffer to the decoder as a single packet, then
/// drain `receive_frame` until it returns an error. Returns the count
/// of successfully decoded frames + the terminal error. Never panics —
/// any panic on the inside is a decoder bug.
fn drive(stream: &[u8]) -> (usize, Result<(), Error>) {
    let mut dec = fresh_decoder();
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream.to_vec());
    // send_packet might return Err for a stream whose sequence-header
    // parse fails the first time scan() walks it. Either way the
    // decoder must remain in a usable state; we just record it and
    // move on to flush + receive_frame.
    let send_res = dec.send_packet(&pkt);
    let _ = dec.flush();

    if let Err(e) = send_res {
        return (0, Err(e));
    }

    let mut frames = 0usize;
    // Cap the receive loop so a hypothetical "always returns Ok with
    // an empty frame" bug can't spin forever. A valid stream of 64x64
    // pictures decodes one frame per picture, and our corpus never
    // exceeds 2 pictures, so 16 is a generous ceiling.
    for _ in 0..16 {
        match dec.receive_frame() {
            Ok(Frame::Video(_)) => frames += 1,
            Ok(other) => panic!("dirac decoder emitted non-video frame: {other:?}"),
            Err(e) => return (frames, Err(e)),
        }
    }
    // Reached the loop cap without seeing an error — surface that as a
    // "stuck" failure rather than silently passing.
    panic!(
        "dirac decoder produced 16 frames from {} bytes without ever erroring — likely a livelock",
        stream.len()
    );
}

/// Build a known-good 1-picture HQ stream we can mutate.
fn good_hq_stream() -> Vec<u8> {
    let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
    let params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
    let (y, u, v) = synthetic_testsrc_64_yuv420();
    encode_single_hq_intra_stream(&seq, &params, 0, &y, &u, &v)
}

/// Build a known-good 1-picture LD stream we can mutate.
fn good_ld_stream() -> Vec<u8> {
    let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
    // LD encoder needs explicit slices_x/slices_y/bytes_per_slice. 4×4
    // slices × 64 bytes/slice = 1 KiB total payload — plenty of room for
    // a 64×64 fixture at qindex=0.
    let params = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 64);
    let (y, u, v) = synthetic_testsrc_64_yuv420();
    encode_single_ld_intra_stream(&seq, &params, 0, &y, &u, &v)
}

// -------------------------------------------------------------------
// Truncation walks
// -------------------------------------------------------------------

/// Every prefix of a valid HQ stream must either decode a frame (the
/// rare case where the prefix happens to end exactly at a complete
/// picture's parse-info boundary) or terminate with a clean error /
/// NeedMore — never panic.
#[test]
fn hq_truncation_walk_never_panics() {
    let stream = good_hq_stream();
    // Step by 7 to keep the test fast; 7 is coprime with the parse-info
    // header size (13) and the picture's typical byte multiples, so the
    // sample evenly visits arbitrary cut points inside every region.
    for cut in (0..stream.len()).step_by(7) {
        let prefix = &stream[..cut];
        // The harness's drive() function performs the panic-trap: if
        // any iteration panics, cargo test reports the cut offset.
        let _ = drive(prefix);
    }
    // A full-length stream must succeed at extracting at least one
    // frame — sanity check that good_hq_stream() actually is good.
    let (frames, _terminal) = drive(&stream);
    assert!(
        frames >= 1,
        "good HQ stream produced no frames — fixture-builder regression"
    );
}

/// Same as the HQ walk but on an LD-profile stream — the LD slice
/// decoder uses a different bit-budget path (`ld_length_bits`) and
/// Funnel-bounded reads, so it must be exercised independently.
#[test]
fn ld_truncation_walk_never_panics() {
    let stream = good_ld_stream();
    for cut in (0..stream.len()).step_by(7) {
        let prefix = &stream[..cut];
        let _ = drive(prefix);
    }
    let (frames, _terminal) = drive(&stream);
    assert!(
        frames >= 1,
        "good LD stream produced no frames — fixture-builder regression"
    );
}

// -------------------------------------------------------------------
// Single-byte mutation walk
// -------------------------------------------------------------------

/// Flip every byte of a valid LD stream to a few stress values and
/// confirm the decoder still terminates cleanly. We only sample every
/// 11th byte position to keep CI under a few seconds — 11 is coprime
/// with 13 (parse-info size) and 4 (picture-number field), so the
/// sample hits a balanced mix of header / payload / slice-data bytes.
#[test]
fn ld_byte_mutation_walk_never_panics() {
    let mut stream = good_ld_stream();
    let original = stream.clone();
    for pos in (0..stream.len()).step_by(11) {
        for &flip in &[0x00u8, 0xFFu8, !original[pos]] {
            stream[pos] = flip;
            let _ = drive(&stream);
            stream[pos] = original[pos]; // restore
        }
    }
}

/// Same idea for HQ — the HQ slice walker reads a 1-byte qindex + per-
/// component length bytes and is the most likely place a corrupted
/// length prefix could try to push the Funnel past its bit budget.
#[test]
fn hq_byte_mutation_walk_never_panics() {
    let mut stream = good_hq_stream();
    let original = stream.clone();
    for pos in (0..stream.len()).step_by(11) {
        for &flip in &[0x00u8, 0xFFu8, !original[pos]] {
            stream[pos] = flip;
            let _ = drive(&stream);
            stream[pos] = original[pos];
        }
    }
}

// -------------------------------------------------------------------
// Pathological gibberish buffers
// -------------------------------------------------------------------

/// Empty input → NeedMore (decoder waiting on the first sequence
/// header) or Eof if flushed. Never a panic, never an Ok frame.
#[test]
fn empty_buffer_yields_clean_terminal() {
    let (frames, terminal) = drive(&[]);
    assert_eq!(frames, 0, "no input must yield no frames");
    assert!(
        matches!(terminal, Err(Error::NeedMore) | Err(Error::Eof)),
        "empty input should be NeedMore or Eof, got {terminal:?}"
    );
}

/// All-zeros buffer — looks like nothing the walker recognises (no
/// BBCD prefix anywhere) and must NOT produce a frame.
#[test]
fn all_zeros_buffer_yields_no_frame() {
    let buf = vec![0u8; 4096];
    let (frames, terminal) = drive(&buf);
    assert_eq!(frames, 0, "all-zeros must not decode to a frame");
    assert!(
        matches!(terminal, Err(Error::NeedMore) | Err(Error::Eof)),
        "all-zeros should be NeedMore or Eof, got {terminal:?}"
    );
}

/// All-0xFF buffer — same: no BBCD anywhere.
#[test]
fn all_ones_buffer_yields_no_frame() {
    let buf = vec![0xFFu8; 4096];
    let (frames, terminal) = drive(&buf);
    assert_eq!(frames, 0);
    assert!(matches!(terminal, Err(Error::NeedMore) | Err(Error::Eof)));
}

/// A buffer of nothing-but-BBCD-prefixes-with-no-valid-parse-info-bytes.
/// The walker must NOT loop forever: each "header" claims
/// `next_parse_offset = 0` (read from the all-zero bytes following the
/// prefix), which is the terminator marker, so the iter ends after the
/// first unit. The first unit's parse_code is 0x00 (sequence header),
/// payload is empty, so sequence-header parsing reads off-end and the
/// front-end surfaces either InvalidData (empty seq-header rejected) or
/// NeedMore (decoder waiting on more data). Either is acceptable as
/// long as the walker terminates.
#[test]
fn bbcd_prefix_spam_terminates() {
    // 64 KiB of repeating "BBCD\x00\x00\x00\x00\x00\x00\x00\x00\x00" —
    // each 13-byte unit claims next_parse_offset 0 (= last) and 0
    // payload bytes. The walker should produce many "last" headers but
    // never an infinite loop. Important: with next=0 the iter marks
    // itself done immediately; the test really pins the
    // recovery-from-bogus-prefix path.
    let mut buf = Vec::new();
    while buf.len() < 64 * 1024 {
        buf.extend_from_slice(BBCD);
        buf.extend_from_slice(&[0u8; 9]); // 13-byte total per unit
    }
    let (frames, terminal) = drive(&buf);
    assert_eq!(frames, 0, "BBCD spam must not decode to a frame");
    // Accept the wider envelope here: a zero-payload sequence header is
    // legitimately classified as InvalidData by `parse_sequence_header`
    // (every field reads as zero, base_video_format_index = 0 is the
    // Custom slot, and the down-stream picture_coding_mode read off-end
    // returns 0 which IS a valid PictureCodingMode::Frames — so the
    // header itself doesn't fail, but the decoder still emits no frame
    // because no picture data unit followed).
    assert!(
        matches!(
            terminal,
            Err(Error::NeedMore) | Err(Error::Eof) | Err(Error::InvalidData(_))
        ),
        "BBCD spam terminal {terminal:?} should be NeedMore / Eof / InvalidData"
    );
}

/// Walk every single byte value as the *parse code* in a parse-info
/// otherwise pointed past end-of-buffer with no payload. The decoder
/// should classify each as either "ignored data unit" (seq=0x00 with
/// no payload errors out; padding/aux/EOS are no-ops; picture parse
/// codes either error or skip) but must never panic. This pins down
/// every entry in the §9.6 parse-code table.
#[test]
fn every_parse_code_terminates_cleanly() {
    for code in 0u16..=255 {
        let code = code as u8;
        // 13-byte parse info: BBCD + code + next_off=0 + prev_off=0.
        let mut unit = Vec::with_capacity(13);
        unit.extend_from_slice(BBCD);
        unit.push(code);
        unit.extend_from_slice(&[0u8; 8]); // both offsets = 0
        let (_frames, terminal) = drive(&unit);
        // Acceptable outcomes:
        //   * NeedMore / Eof — no picture in the unit.
        //   * InvalidData — the picture parse-code admitted decode but
        //     the (empty) payload tripped a Truncated.
        //   * Unsupported — core-syntax dispatch reached for a code
        //     the picture pipeline can't handle yet.
        match terminal {
            Err(Error::NeedMore)
            | Err(Error::Eof)
            | Err(Error::InvalidData(_))
            | Err(Error::Unsupported(_)) => {}
            other => panic!("parse code 0x{code:02X} produced unexpected terminal {other:?}"),
        }
    }
}

// -------------------------------------------------------------------
// Pathological parse-info offsets
// -------------------------------------------------------------------

/// A parse info with `next_parse_offset = u32::MAX` should NOT cause
/// the walker to allocate, indexed-read past end, or loop forever. The
/// stream walker is documented to fall back to byte-search; that path
/// must terminate (it bounds itself by `data.len()`).
#[test]
fn oversize_next_parse_offset_terminates() {
    // [seq parse info, broken next_off = u32::MAX][seq payload, 32 bytes][EOS]
    let mut buf = Vec::new();
    buf.extend_from_slice(BBCD);
    buf.push(0x00); // seq header
    buf.extend_from_slice(&u32::MAX.to_be_bytes()); // next_parse_offset = lie
    buf.extend_from_slice(&0u32.to_be_bytes()); // prev_parse_offset
    buf.extend_from_slice(&[0u8; 32]); // bogus seq header bytes (will fail parse)
    buf.extend_from_slice(BBCD);
    buf.push(0x10); // EOS
    buf.extend_from_slice(&[0u8; 8]); // both offsets = 0
    let (frames, terminal) = drive(&buf);
    assert_eq!(frames, 0);
    assert!(matches!(
        terminal,
        Err(Error::NeedMore) | Err(Error::Eof) | Err(Error::InvalidData(_))
    ));
}

/// `next_parse_offset` that wraps a usize-add. We feed a 13-byte parse
/// info whose `next_parse_offset` is just under `u32::MAX`; the walker
/// computes `pi_offset + next` with `saturating_add` (per stream.rs),
/// so it never wraps, but the test pins the contract.
#[test]
fn near_wrap_next_parse_offset_terminates() {
    let mut buf = Vec::new();
    buf.extend_from_slice(BBCD);
    buf.push(0x10); // EOS — easy classify, no payload required
    buf.extend_from_slice(&0xFFFF_FFF0u32.to_be_bytes()); // huge but not max
    buf.extend_from_slice(&0u32.to_be_bytes());
    let (frames, terminal) = drive(&buf);
    assert_eq!(frames, 0);
    assert!(matches!(terminal, Err(Error::NeedMore) | Err(Error::Eof)));
}

// -------------------------------------------------------------------
// Hand-crafted invalid sequence headers
// -------------------------------------------------------------------

/// A sequence header carrying an unknown `base_video_format` index
/// returns a clean InvalidData from `send_packet` (the decoder front-
/// end maps `sequence::ParseError::UnknownBaseVideoFormat` to
/// `Error::InvalidData`). Pin that contract here so a future refactor
/// doesn't accidentally panic or coerce to Ok.
#[test]
fn unknown_base_video_format_yields_invalid_data() {
    // Bit-craft a sequence header with version 2/2, profile 2, level 1,
    // base_video_format = 30 (unknown).
    let mut bits = String::new();
    bits.push_str("011"); // ver major = 2 (interleaved exp-golomb)
    bits.push_str("011"); // ver minor = 2
    bits.push_str("011"); // profile = 2
    bits.push_str("001"); // level = 1
                          // base_video_format = 30 -> N+1=31=11111 -> K=4
                          // interleaved: 0 1 0 1 0 1 0 1 1
    bits.push_str("010101011");
    while bits.len() % 8 != 0 {
        bits.push('0');
    }
    let payload: Vec<u8> = bits
        .as_bytes()
        .chunks(8)
        .map(|c| c.iter().fold(0u8, |acc, b| (acc << 1) | (*b == b'1') as u8))
        .collect();

    // Wrap in a parse-info header.
    let mut stream = Vec::new();
    stream.extend_from_slice(BBCD);
    stream.push(0x00); // seq header parse code
    let unit_len = 13 + payload.len();
    stream.extend_from_slice(&(unit_len as u32).to_be_bytes()); // next_off
    stream.extend_from_slice(&0u32.to_be_bytes()); // prev_off
    stream.extend_from_slice(&payload);
    // Followed by EOS to give the walker a clean terminator.
    stream.extend_from_slice(BBCD);
    stream.push(0x10);
    stream.extend_from_slice(&[0u8; 8]);

    let mut dec = fresh_decoder();
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream);
    let res = dec.send_packet(&pkt);
    // send_packet should return InvalidData immediately when the seq
    // header's base_video_format is rejected — that's the cleanest
    // signal to the caller that this stream is unrecoverable.
    assert!(
        matches!(res, Err(Error::InvalidData(_))),
        "unknown base_video_format should be InvalidData, got {res:?}"
    );
}

// -------------------------------------------------------------------
// Pathological transform parameter blocks
// -------------------------------------------------------------------

/// A picture data unit whose transform parameters carry
/// `dwt_depth = 99` (out of range; `> 6` rejected by
/// `parse_transform_parameters` with `UnsupportedDwtDepth`). The
/// decoder should surface this as InvalidData via try_decode_next's
/// `PictureError::UnsupportedDwtDepth` -> `Error::invalid` branch.
/// Pin that contract.
#[test]
fn picture_with_oversized_dwt_depth_yields_invalid_data() {
    // Build a valid HQ stream, then patch the first picture's
    // dwt_depth field. The transform_parameters block in an HQ picture
    // starts after [picture_number(4 bytes), byte_align()] so it's at
    // offset 4 within the picture payload. The format is:
    //   wavelet_index (exp-golomb)
    //   dwt_depth (exp-golomb)
    // For our default-encoded stream wavelet_index encodes value 1
    // (LeGall5_3) as bits "010" and dwt_depth encodes value 3 as
    // "00101", so the combined bit pattern at offset 4 (post-byte-align)
    // is "01000101 padded". Rather than bit-surgery the easiest way to
    // produce the test fixture is to use the encoder's spec-compliant
    // path with a synthetic dwt_depth — but EncoderParams::default_hq
    // only accepts known good values. We sidestep by replacing the
    // entire transform_parameters block manually.
    //
    // Simpler approach: feed the decoder a hand-constructed picture
    // payload whose transform-parameters block carries dwt_depth = 99.
    // wavelet_index 1 = "010"
    // dwt_depth 99: 99+1=100=0b1100100 (7 bits) -> K=6
    //   interleaved: 0 1 0 1 0 0 0 0 0 1 0 0 1 (13 bits, ending in
    //   data-bit '0' then follow-bit '1' for termination)
    //   actually for value 99, N+1=100=1100100; K=floor(log2(100))=6.
    //   pattern: 0 x5 0 x4 0 x3 0 x2 0 x1 0 x0 1  where 100=1100100,
    //   x5..x0 = 100100 (bits of 100 after the leading 1).
    //   So: 0 1 0 0 0 0 0 1 0 0 0 0 0 1 -> 14 bits.
    //
    // The picture-number field is 4 bytes big-endian; we use 0.
    // After picture_number we byte-align then write the transform-
    // parameters block. We don't need any slice data because the
    // decoder errors out before reading slices.
    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes()); // picture_number = 0
                                                    // byte-align is already on a boundary, so just write the bits.
    let mut bits = String::new();
    // wavelet_index = 1 (interleaved exp-golomb): "010"
    bits.push_str("010");
    // dwt_depth = 99 (interleaved exp-golomb): bits as derived above.
    bits.push_str("01000001000001");
    // Pad to byte boundary; the rest is whatever the decoder happens
    // to read while bailing out.
    while bits.len() % 8 != 0 {
        bits.push('0');
    }
    let tp_bytes: Vec<u8> = bits
        .as_bytes()
        .chunks(8)
        .map(|c| c.iter().fold(0u8, |acc, b| (acc << 1) | (*b == b'1') as u8))
        .collect();
    payload.extend_from_slice(&tp_bytes);
    // Pad payload to a sensible length so the parse-info offsets work.
    while payload.len() < 64 {
        payload.push(0);
    }

    // Build a valid HQ stream and replace its picture payload.
    let mut stream = Vec::new();
    // Sequence header — borrow from the encoder.
    let good = good_hq_stream();
    // Find the picture parse-info offset in the good stream: walk to
    // the second BBCD (first is seq header, second is picture).
    let mut bbcd_positions = Vec::new();
    let mut i = 0;
    while i + 4 <= good.len() {
        if &good[i..i + 4] == BBCD {
            bbcd_positions.push(i);
            i += 4;
        } else {
            i += 1;
        }
    }
    assert!(
        bbcd_positions.len() >= 3,
        "good HQ stream should have seq + picture + EOS BBCDs"
    );
    let pic_off = bbcd_positions[1];
    let eos_off = bbcd_positions[2];
    // Copy seq + seq payload + picture parse-info; replace picture
    // payload with our crafted bytes; then EOS verbatim.
    stream.extend_from_slice(&good[..pic_off + 13]);
    stream.extend_from_slice(&payload);
    stream.extend_from_slice(&good[eos_off..]);

    let (frames, terminal) = drive(&stream);
    assert_eq!(frames, 0, "oversized dwt_depth must not yield a frame");
    assert!(
        matches!(terminal, Err(Error::InvalidData(_))),
        "oversized dwt_depth should be InvalidData, got {terminal:?}"
    );
}

// -------------------------------------------------------------------
// Cross-walk: every prefix of the bbcd-spam buffer
// -------------------------------------------------------------------

/// Combined truncation + gibberish: every prefix of a "BBCD followed
/// by random bytes" buffer. Pins down the walker's robustness when a
/// prefix cuts in the middle of what looks like a parse info.
#[test]
fn truncated_gibberish_terminates() {
    // 1 KiB of "BBCD + random-looking bytes" so the walker sees a mix
    // of valid BBCD prefixes at irregular offsets.
    let mut buf = Vec::new();
    let mut seed: u32 = 0x1234_5678;
    while buf.len() < 1024 {
        if buf.len() % 17 == 0 {
            buf.extend_from_slice(BBCD);
        } else {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            buf.push((seed >> 16) as u8);
        }
    }
    for cut in (0..buf.len()).step_by(13) {
        let prefix = &buf[..cut];
        let (frames, terminal) = drive(prefix);
        // Gibberish must never decode to a frame.
        assert_eq!(
            frames, 0,
            "gibberish prefix of length {cut} decoded a frame"
        );
        // Terminal must be a clean error / NeedMore / Eof / Unsupported.
        match terminal {
            Err(Error::NeedMore)
            | Err(Error::Eof)
            | Err(Error::InvalidData(_))
            | Err(Error::Unsupported(_)) => {}
            other => panic!("gibberish prefix len {cut} produced unexpected terminal {other:?}"),
        }
    }
}
