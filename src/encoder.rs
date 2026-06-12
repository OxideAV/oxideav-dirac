//! VC-2 HQ-profile intra-only encoder.
//!
//! This mirrors the decoder path in [`crate::picture`] for the HQ
//! profile (parse codes `0xE8` — non-reference — and `0xEC` —
//! reference), writing a self-contained bitstream that the decoder in
//! the same crate reads back pixel-for-pixel identically at qindex 0.
//!
//! The encoder produces a byte-aligned elementary stream:
//!
//! ```text
//!   [parse info 13B]  sequence header
//!   [parse info 13B]  HQ intra picture
//!   [parse info 13B]  end of sequence
//! ```
//!
//! Stage breakdown per picture:
//!
//! 1. Pad each component up to a multiple of `2^dwt_depth` and subtract
//!    the `2^(depth-1)` midpoint to centre around zero (§15.10 inverse).
//! 2. Forward DWT (§15.6 analysis, see [`crate::wavelet::dwt`]).
//! 3. Quantise each subband with a per-(level, orientation) quantiser
//!    derived from the slice `qindex` and the default quantisation
//!    matrix (§13.3.1, §13.5.4).
//! 4. Pack per-slice: `slice_prefix_bytes` zero bytes, a 1-byte qindex,
//!    then for each of Y / U / V a length byte followed by a bounded
//!    exp-Golomb coefficient block whose byte-length is the length byte
//!    times `slice_size_scaler`.
//!
//! Only 8-bit 4:2:0 intra pictures with LeGall 5/3 + depth 3 are
//! emitted today — enough to round-trip a 64x64 testsrc through the
//! crate's own decoder.

use crate::bitwriter::BitWriter;
use crate::parse_info::BBCD;
use crate::quant::{slice_quantisers, QuantMatrix};
use crate::sequence::SequenceHeader;
use crate::subband::{
    padded_component_dims_ho, slice_band_order, subband_dims_ho, Orient, SubbandData,
};
use crate::video_format::ChromaFormat;
use crate::wavelet::{dwt, dwt_with_ho, WaveletFilter};

/// Parameters that drive picture encoding. Mirrors the subset of
/// `TransformParameters` the HQ-profile encoder actually needs.
#[derive(Debug, Clone)]
pub struct EncoderParams {
    pub wavelet: WaveletFilter,
    pub dwt_depth: u32,
    pub slices_x: u32,
    pub slices_y: u32,
    pub slice_prefix_bytes: u32,
    pub slice_size_scaler: u32,
    pub quant_matrix: QuantMatrix,
    /// When `true`, emit `quant_matrix.levels` explicitly into the
    /// stream as a §12.4.5.3 / §11.3.5 *custom* quantisation matrix
    /// (`custom_quant_matrix = True`) instead of the
    /// `custom_quant_matrix = False` flag that makes the decoder look up
    /// the Annex E.1 default. The encoder already quantises against
    /// `quant_matrix` regardless; this flag only controls whether the
    /// matrix travels in-band, so a non-default `quant_matrix` is
    /// recoverable on the decode side. Defaults to `false`. Note that
    /// §11.3.5 *requires* `True` whenever `dwt_depth > 4` (the default
    /// table is undefined there); the [`Self::with_custom_quant_matrix`]
    /// helper sets the flag for that case.
    pub custom_quant_matrix: bool,
    /// Per-slice quantisation index (0..=127). 0 means lossless-ish —
    /// qf=4, offset=1, resulting in near-identity inverse quant for
    /// coefficients small compared to 4.
    ///
    /// When [`Self::slice_size_target`] is `None` this exact value is
    /// written into every slice header and used to quantise the whole
    /// picture (the constant-qindex behaviour). When it is `Some(_)` it
    /// becomes the *floor* of the per-slice adaptive search (§13.5.4):
    /// each slice never uses a qindex below this value.
    pub qindex: u32,
    /// Per-slice byte-budget target for the §13.5.4 adaptive-qindex
    /// search. `None` (default) keeps the constant-`qindex` behaviour —
    /// every slice writes `qindex` verbatim. `Some(target)` makes each
    /// slice independently pick the *smallest* qindex in
    /// `qindex..=127` for which **every** component's HQ length byte is
    /// `<= target` (i.e. each component's coefficient payload fits in
    /// `target * slice_size_scaler` bytes). This is the spec's intended
    /// HQ rate-control knob — a slice with little energy keeps the low
    /// floor qindex (lossless-ish), while a busy slice raises its own
    /// qindex just enough to fit, instead of relying on a generous
    /// `slice_size_scaler` and silently truncating. The HQ profile
    /// applies no §13.5.1 DC prediction, so each slice's coefficients
    /// are independent and may be quantised at its own qindex without
    /// any cross-slice coupling. If even qindex 127 overflows the
    /// target the search keeps 127 (the length byte is still bounded by
    /// the 255 cap the `debug_assert!` in [`encode_hq_slice`] guards).
    pub slice_size_target: Option<u32>,
    /// Bitstream major version selector for the `transform_parameters()`
    /// block. `2` (default) writes the v2 syntax — no
    /// `extended_transform_parameters()` block follows the
    /// `wavelet_index` / `dwt_depth` pair. `3` writes the v3 syntax
    /// (SMPTE ST 2042-1:2022 §12.4.4): after `dwt_depth` we emit two
    /// `read_bool()` flag bits at their `False` default plus the
    /// resulting (empty) gated fields — i.e. the §12.4.4 NOTE
    /// "symmetric-default" form, where `wavelet_index_ho =
    /// wavelet_index` and `dwt_depth_ho = 0` and the IDWT therefore
    /// matches the v2 process verbatim. Callers MUST also set the
    /// sequence header's `parse_parameters.version_major` to `3` so the
    /// decoder dispatches into [`crate::picture::parse_extended_transform_parameters`].
    /// Asymmetric (non-default) emission is configured via
    /// [`Self::extended_transform_override`].
    pub major_version: u32,
    /// Optional override for the `extended_transform_parameters()`
    /// block (SMPTE ST 2042-1:2022 §12.4.4). `None` (default) keeps the
    /// symmetric default per the §12.4.4 NOTE: both `read_bool()` flag
    /// bits are emitted as `False` and the gated `wavelet_index_ho` /
    /// `dwt_depth_ho` fields are omitted, so the IDWT is identical to
    /// the v2 process. `Some(_)` instead emits the asymmetric form:
    /// `asym_transform_index_flag = (wavelet_index_ho != wavelet_index)`
    /// and `asym_transform_flag = (dwt_depth_ho != 0)`, with each
    /// non-default value written as an interleaved exp-Golomb code.
    ///
    /// Only consulted when [`Self::major_version`] is `>= 3`. With
    /// `dwt_depth_ho > 0` the whole encode pipeline follows the
    /// asymmetric layout: the forward transform runs `dwt_depth_ho`
    /// horizontal-only analysis levels beneath the `dwt_depth` 2-D
    /// levels ([`crate::wavelet::dwt_with_ho`]), the slice packers
    /// emit the §13.5.4 asymmetric band order (L, then H ×
    /// `dwt_depth_ho`, then HL/LH/HH triplets), and the §12.4.5.3
    /// quant matrix is written in the asymmetric shape. The caller
    /// must then set [`Self::custom_quant_matrix`] and supply a
    /// [`QuantMatrix`] whose `dwt_depth_ho` matches (the Annex D
    /// asymmetric default tables are not transcribed, so the decode
    /// side rejects a default-matrix asymmetric stream) — the
    /// [`Self::with_asymmetric_transform`] helper sets all of this
    /// up in one call.
    pub extended_transform_override: Option<ExtendedTransformOverride>,
}

/// Override values for the §12.4.4 `extended_transform_parameters()`
/// block, used only when [`EncoderParams::extended_transform_override`]
/// or [`LdEncoderParams::extended_transform_override`] is `Some`.
/// See those fields for the emission semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtendedTransformOverride {
    /// `wavelet_index_ho` (§12.4.4.2). Set equal to the primary
    /// `wavelet_index` to leave `asym_transform_index_flag` at `False`
    /// (the gated field is then omitted from the stream); set to a
    /// different valid wavelet index to emit `asym_transform_index_flag
    /// = True` followed by this value as an interleaved exp-Golomb code.
    pub wavelet_index_ho: u32,
    /// `dwt_depth_ho` (§12.4.4.3). Set to `0` to leave
    /// `asym_transform_flag` at `False` (the gated field is then
    /// omitted); set to a positive value to emit `asym_transform_flag =
    /// True` followed by this value as an interleaved exp-Golomb code.
    pub dwt_depth_ho: u32,
}

impl EncoderParams {
    /// Default HQ intra parameters: LeGall 5/3 at depth 3, 4x4 slices,
    /// no prefix bytes, scaler 1, qindex 0. Good for the round-trip
    /// smoke test on small pictures.
    pub fn default_hq(wavelet: WaveletFilter, dwt_depth: u32) -> Self {
        let quant_matrix = QuantMatrix::default_for(wavelet, dwt_depth)
            .expect("default quant matrix exists for depth <= 4");
        Self {
            wavelet,
            dwt_depth,
            slices_x: 8,
            slices_y: 8,
            slice_prefix_bytes: 0,
            slice_size_scaler: 1,
            quant_matrix,
            custom_quant_matrix: false,
            qindex: 0,
            slice_size_target: None,
            major_version: 2,
            extended_transform_override: None,
        }
    }

    /// Select the v3 `transform_parameters()` syntax (SMPTE ST
    /// 2042-1:2022 §12.4.4). The emitted block adds the two
    /// `extended_transform_parameters()` flag bits at their `False`
    /// default — i.e. the §12.4.4 NOTE "symmetric-default" form — so
    /// the IDWT remains identical to the v2 process. The caller must
    /// also set the sequence header's
    /// `parse_parameters.version_major` to `3` for the decoder to
    /// dispatch into the v3 parsing branch. See
    /// [`Self::major_version`] for the full semantics.
    pub fn with_major_version_3(mut self) -> Self {
        self.major_version = 3;
        self
    }

    /// Override the §12.4.4 `extended_transform_parameters()` emission
    /// with explicit asymmetric values. Only takes effect when
    /// [`Self::major_version`] is `>= 3`. See
    /// [`Self::extended_transform_override`] for the full semantics —
    /// with `dwt_depth_ho > 0` the caller must also install a matching
    /// custom asymmetric quant matrix; prefer
    /// [`Self::with_asymmetric_transform`], which does both.
    pub fn with_extended_transform_override(
        mut self,
        override_: ExtendedTransformOverride,
    ) -> Self {
        self.extended_transform_override = Some(override_);
        self
    }

    /// Configure a fully-wired §12.4.4 asymmetric (horizontal-only)
    /// transform: selects the v3 syntax, installs the
    /// `extended_transform_parameters()` override for `(wavelet_ho,
    /// dwt_depth_ho)`, and replaces the quantisation matrix with an
    /// all-zero **custom** matrix in the §12.4.5.3 asymmetric shape
    /// (`1 + dwt_depth_ho + 3 * dwt_depth` entries) — required because
    /// the Annex D asymmetric default tables are not transcribed. An
    /// all-zero matrix leaves every subband at the slice qindex
    /// (§13.5.5 subtracts matrix entries from `qindex`), so qindex 0
    /// stays lossless. The caller must still set the sequence header's
    /// `parse_parameters.version_major` to `3`.
    pub fn with_asymmetric_transform(
        mut self,
        wavelet_ho: WaveletFilter,
        dwt_depth_ho: u32,
    ) -> Self {
        self.major_version = 3;
        self.extended_transform_override = Some(ExtendedTransformOverride {
            wavelet_index_ho: wavelet_index(wavelet_ho),
            dwt_depth_ho,
        });
        self.quant_matrix = zero_quant_matrix(self.dwt_depth, dwt_depth_ho);
        self.custom_quant_matrix = true;
        self
    }

    /// Replace the quantisation matrix with `matrix` and set
    /// `custom_quant_matrix = true` so the decoder reads it back in-band
    /// (§12.4.5.3) rather than recomputing the Annex E.1 default. The
    /// caller is responsible for `matrix.dwt_depth == self.dwt_depth`.
    pub fn with_custom_quant_matrix(mut self, matrix: QuantMatrix) -> Self {
        self.quant_matrix = matrix;
        self.custom_quant_matrix = true;
        self
    }

    /// Enable the §13.5.4 per-slice adaptive-qindex search with a
    /// per-component byte budget of `target` length-byte units (i.e.
    /// `target * slice_size_scaler` bytes per component). See
    /// [`Self::slice_size_target`] for the full semantics. The current
    /// [`Self::qindex`] becomes the search floor.
    pub fn with_slice_size_target(mut self, target: u32) -> Self {
        self.slice_size_target = Some(target);
        self
    }

    /// Effective §12.4.4.3 `dwt_depth_ho`: the override value when the
    /// v3 syntax is selected, otherwise 0 (pre-v3 streams have no
    /// `extended_transform_parameters()` block).
    pub fn dwt_depth_ho(&self) -> u32 {
        if self.major_version >= 3 {
            self.extended_transform_override
                .map_or(0, |e| e.dwt_depth_ho)
        } else {
            0
        }
    }

    /// Effective §12.4.4.2 horizontal-only wavelet filter: the
    /// override's `wavelet_index_ho` resolved through
    /// [`WaveletFilter::from_index`], defaulting to [`Self::wavelet`]
    /// when no override is active (the §12.4.4 default) or the index
    /// is out of range.
    pub fn wavelet_ho(&self) -> WaveletFilter {
        if self.major_version >= 3 {
            self.extended_transform_override
                .and_then(|e| WaveletFilter::from_index(e.wavelet_index_ho))
                .unwrap_or(self.wavelet)
        } else {
            self.wavelet
        }
    }
}

/// All-zero custom quantisation matrix in the §12.4.5.3 shape for
/// `(dwt_depth, dwt_depth_ho)` — `1 + dwt_depth_ho + 3 * dwt_depth`
/// in-band entries, all zero, so every subband's effective quantiser
/// equals the slice qindex (§13.5.5).
fn zero_quant_matrix(dwt_depth: u32, dwt_depth_ho: u32) -> QuantMatrix {
    let total = (dwt_depth_ho + dwt_depth) as usize + 1;
    QuantMatrix {
        dwt_depth,
        dwt_depth_ho,
        levels: vec![[0u32; 4]; total],
    }
}

/// Encode a single intra HQ picture. `y`, `u`, `v` hold one byte per
/// sample in row-major order, sized exactly to the component's luma /
/// chroma dimensions from `sequence`.
pub fn encode_hq_intra_picture(
    sequence: &SequenceHeader,
    params: &EncoderParams,
    picture_number: u32,
    y: &[u8],
    u: &[u8],
    v: &[u8],
) -> Vec<u8> {
    let mut w = BitWriter::new();

    // §12.2 picture_header: byte-align then 4-byte picture_number.
    w.byte_align();
    w.write_uint_lit(4, picture_number);

    // §12.3 wavelet_transform — for intra there's no zero_residual.
    w.byte_align();
    write_transform_parameters(&mut w, params);
    w.byte_align();

    // Per-component coefficient pyramids.
    let luma_w = sequence.luma_width;
    let luma_h = sequence.luma_height;
    let chroma_w = sequence.chroma_width;
    let chroma_h = sequence.chroma_height;

    // Keep the *unquantised* coefficient pyramids — each slice may pick
    // its own qindex (§13.5.4) and quantise its own region on the fly.
    let y_py = forward_component(y, luma_w, luma_h, sequence.luma_depth, params);
    let u_py = forward_component(u, chroma_w, chroma_h, sequence.chroma_depth, params);
    let v_py = forward_component(v, chroma_w, chroma_h, sequence.chroma_depth, params);

    // Precompute per-level subband sizes for each component.
    let luma_dims = component_dims(luma_w, luma_h, params.dwt_depth, params.dwt_depth_ho());
    let chroma_dims = component_dims(chroma_w, chroma_h, params.dwt_depth, params.dwt_depth_ho());

    for sy in 0..params.slices_y {
        for sx in 0..params.slices_x {
            // §13.5.4: choose this slice's qindex. With no byte target
            // the floor `params.qindex` is used verbatim (the legacy
            // constant-qindex path). With a target, search upward for
            // the smallest qindex whose component lengths all fit.
            let qindex = match params.slice_size_target {
                None => params.qindex,
                Some(target) => choose_hq_slice_qindex(
                    params,
                    sx,
                    sy,
                    &y_py,
                    &u_py,
                    &v_py,
                    &luma_dims,
                    &chroma_dims,
                    target,
                ),
            };
            encode_hq_slice(
                &mut w,
                params,
                qindex,
                sx,
                sy,
                &y_py,
                &u_py,
                &v_py,
                &luma_dims,
                &chroma_dims,
            );
        }
    }

    w.finish()
}

/// Length byte (in `slice_size_scaler` units) one HQ component would
/// occupy if this slice's coefficients were quantised at `qindex`.
/// Mirrors the padding arithmetic in [`encode_hq_slice`].
fn hq_component_length_byte(
    params: &EncoderParams,
    qindex: u32,
    comp_py: &[[SubbandData; 4]],
    dims: &[(usize, usize)],
    sx: u32,
    sy: u32,
) -> usize {
    let q_per_level = slice_quantisers(qindex, &params.quant_matrix);
    let comp_bytes = encode_hq_component(params, &q_per_level, comp_py, dims, sx, sy);
    let scaler = params.slice_size_scaler.max(1) as usize;
    comp_bytes.len().div_ceil(scaler)
}

/// §13.5.4 adaptive per-slice quantiser search (HQ profile). Returns the
/// smallest `qindex` in `params.qindex..=127` for which **every**
/// component's HQ length byte is `<= target`. If even qindex 127 cannot
/// fit, returns 127 (the most aggressive quantiser available); the HQ
/// length-byte 255 cap is still enforced downstream by the
/// `debug_assert!` in [`encode_hq_slice`]. The HQ profile applies no
/// §13.5.1 DC prediction so each slice is independent — quantising one
/// slice at a higher qindex never affects another.
#[allow(clippy::too_many_arguments)]
fn choose_hq_slice_qindex(
    params: &EncoderParams,
    sx: u32,
    sy: u32,
    y_py: &[[SubbandData; 4]],
    u_py: &[[SubbandData; 4]],
    v_py: &[[SubbandData; 4]],
    luma_dims: &[(usize, usize)],
    chroma_dims: &[(usize, usize)],
    target: u32,
) -> u32 {
    let target = target as usize;
    let floor = params.qindex.min(127);
    for qindex in floor..=127 {
        let ly = hq_component_length_byte(params, qindex, y_py, luma_dims, sx, sy);
        let lu = hq_component_length_byte(params, qindex, u_py, chroma_dims, sx, sy);
        let lv = hq_component_length_byte(params, qindex, v_py, chroma_dims, sx, sy);
        if ly <= target && lu <= target && lv <= target {
            return qindex;
        }
    }
    127
}

/// Per-slice diagnostic for the §13.5.4 adaptive-qindex search. Returns
/// one `(qindex, max_component_length_byte)` pair per slice, in
/// raster slice order (`sy` outer, `sx` inner) — exactly the order
/// [`encode_hq_intra_picture`] emits slices. `qindex` is the value the
/// encoder would write into that slice's header; `max_component_length_byte`
/// is the largest of the three components' length bytes at that qindex
/// (i.e. the value the [`EncoderParams::slice_size_target`] budget is
/// compared against). With [`EncoderParams::slice_size_target`] = `None`
/// every pair carries `params.qindex` and the resulting length. This is
/// drift-proof against the emit path because both call the same
/// [`choose_hq_slice_qindex`] / [`hq_component_length_byte`] helpers.
pub fn hq_slice_qindexes(
    sequence: &SequenceHeader,
    params: &EncoderParams,
    y: &[u8],
    u: &[u8],
    v: &[u8],
) -> Vec<(u32, usize)> {
    let luma_w = sequence.luma_width;
    let luma_h = sequence.luma_height;
    let chroma_w = sequence.chroma_width;
    let chroma_h = sequence.chroma_height;

    let y_py = forward_component(y, luma_w, luma_h, sequence.luma_depth, params);
    let u_py = forward_component(u, chroma_w, chroma_h, sequence.chroma_depth, params);
    let v_py = forward_component(v, chroma_w, chroma_h, sequence.chroma_depth, params);

    let luma_dims = component_dims(luma_w, luma_h, params.dwt_depth, params.dwt_depth_ho());
    let chroma_dims = component_dims(chroma_w, chroma_h, params.dwt_depth, params.dwt_depth_ho());

    let mut out = Vec::with_capacity((params.slices_x * params.slices_y) as usize);
    for sy in 0..params.slices_y {
        for sx in 0..params.slices_x {
            let qindex = match params.slice_size_target {
                None => params.qindex,
                Some(target) => choose_hq_slice_qindex(
                    params,
                    sx,
                    sy,
                    &y_py,
                    &u_py,
                    &v_py,
                    &luma_dims,
                    &chroma_dims,
                    target,
                ),
            };
            let ly = hq_component_length_byte(params, qindex, &y_py, &luma_dims, sx, sy);
            let lu = hq_component_length_byte(params, qindex, &u_py, &chroma_dims, sx, sy);
            let lv = hq_component_length_byte(params, qindex, &v_py, &chroma_dims, sx, sy);
            out.push((qindex, ly.max(lu).max(lv)));
        }
    }
    out
}

/// Encoded payload byte count of a single VC-2 HQ intra picture when
/// every slice is forced to share `qindex`, ignoring
/// `params.slice_size_target` (i.e. the constant-qindex picture-byte
/// count at the given quantiser). Unlike LD's
/// [`ld_picture_payload_bytes`] this is **content-dependent** — each HQ
/// slice's length byte tracks the slice's actual coefficient block size,
/// not a fixed budget — so the function performs a full forward DWT,
/// quantises at `qindex`, and serialises every slice.
///
/// Mirrors [`encode_hq_intra_picture`] exactly for the content-dependent
/// per-slice size, but with the per-slice [`choose_hq_slice_qindex`]
/// search replaced by the supplied `qindex` for every slice. Used by
/// [`pick_hq_picture_qindex`] to walk qindex space against a picture-byte
/// budget.
///
/// Source of truth: BBC Dirac Specification v2.2.3 §13.5.2 (per-slice
/// qindex header) + §13.5.4 (`slice_quantisers(qindex)`).
pub fn hq_picture_payload_bytes_at_qindex(
    sequence: &SequenceHeader,
    params: &EncoderParams,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    qindex: u32,
) -> usize {
    let luma_w = sequence.luma_width;
    let luma_h = sequence.luma_height;
    let chroma_w = sequence.chroma_width;
    let chroma_h = sequence.chroma_height;

    let y_py = forward_component(y, luma_w, luma_h, sequence.luma_depth, params);
    let u_py = forward_component(u, chroma_w, chroma_h, sequence.chroma_depth, params);
    let v_py = forward_component(v, chroma_w, chroma_h, sequence.chroma_depth, params);

    let luma_dims = component_dims(luma_w, luma_h, params.dwt_depth, params.dwt_depth_ho());
    let chroma_dims = component_dims(chroma_w, chroma_h, params.dwt_depth, params.dwt_depth_ho());

    let mut w = BitWriter::new();
    // §12.2 picture_header: byte-align then 4-byte picture_number.
    w.byte_align();
    w.write_uint_lit(4, 0); // value does not affect size
                            // §12.3 wavelet_transform.
    w.byte_align();
    write_transform_parameters(&mut w, params);
    w.byte_align();
    for sy in 0..params.slices_y {
        for sx in 0..params.slices_x {
            encode_hq_slice(
                &mut w,
                params,
                qindex,
                sx,
                sy,
                &y_py,
                &u_py,
                &v_py,
                &luma_dims,
                &chroma_dims,
            );
        }
    }
    w.finish().len()
}

/// VC-2 HQ picture-level qindex picker — the HQ analogue of
/// [`pick_ld_picture_qindex`].
///
/// Given a target *picture-payload* byte budget (i.e. the bytes between
/// the picture's parse-info and the next parse-info, equal to
/// [`hq_picture_payload_bytes_at_qindex`]'s return), pick the **smallest**
/// `qindex` in `params.qindex.min(127)..=127` for which the constant-qindex
/// picture size ≤ `target_bytes`. If even qindex 127 overflows, return 127
/// (the most aggressive quantiser available; the §13.5.4 length-byte cap
/// of 255 still applies via the `debug_assert!` in [`encode_hq_slice`]).
///
/// `params.slice_size_target` is **ignored** by this picker — it forces a
/// single picture-level qindex onto every slice, mirroring LD's
/// per-picture rate model. Callers that want the §13.5.4 per-slice
/// adaptive search should keep using `with_slice_size_target` directly;
/// the two knobs are independent rate-control strategies.
///
/// Monotone in `target_bytes`: a smaller budget can only push the chosen
/// qindex up (or leave it). HQ picture bytes shrink monotonically with
/// increasing qindex because the dead-zone forward quantiser
/// ([`quantise_coeff`]) drives more coefficients toward zero, which the
/// interleaved exp-Golomb coder encodes with fewer bits.
///
/// Source of truth: BBC Dirac Specification v2.2.3 §13.5.2, §13.5.4.
pub fn pick_hq_picture_qindex(
    sequence: &SequenceHeader,
    params: &EncoderParams,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    target_bytes: u32,
) -> u32 {
    let luma_w = sequence.luma_width;
    let luma_h = sequence.luma_height;
    let chroma_w = sequence.chroma_width;
    let chroma_h = sequence.chroma_height;

    // Forward DWT once — every qindex re-quantises from the same
    // unquantised coefficient pyramid.
    let y_py = forward_component(y, luma_w, luma_h, sequence.luma_depth, params);
    let u_py = forward_component(u, chroma_w, chroma_h, sequence.chroma_depth, params);
    let v_py = forward_component(v, chroma_w, chroma_h, sequence.chroma_depth, params);

    let luma_dims = component_dims(luma_w, luma_h, params.dwt_depth, params.dwt_depth_ho());
    let chroma_dims = component_dims(chroma_w, chroma_h, params.dwt_depth, params.dwt_depth_ho());

    let floor = params.qindex.min(127);
    for qindex in floor..=127 {
        let bytes = hq_picture_bytes_inner(
            params,
            qindex,
            &y_py,
            &u_py,
            &v_py,
            &luma_dims,
            &chroma_dims,
        );
        if bytes <= target_bytes as usize {
            return qindex;
        }
    }
    127
}

/// Inner serialisation helper shared by [`pick_hq_picture_qindex`] and
/// [`hq_picture_qindex_diagnostic`]: serialises the whole picture at the
/// given qindex from already-computed pyramids and returns the byte count.
#[allow(clippy::too_many_arguments)]
fn hq_picture_bytes_inner(
    params: &EncoderParams,
    qindex: u32,
    y_py: &[[SubbandData; 4]],
    u_py: &[[SubbandData; 4]],
    v_py: &[[SubbandData; 4]],
    luma_dims: &[(usize, usize)],
    chroma_dims: &[(usize, usize)],
) -> usize {
    let mut w = BitWriter::new();
    w.byte_align();
    w.write_uint_lit(4, 0);
    w.byte_align();
    write_transform_parameters(&mut w, params);
    w.byte_align();
    for sy in 0..params.slices_y {
        for sx in 0..params.slices_x {
            encode_hq_slice(
                &mut w,
                params,
                qindex,
                sx,
                sy,
                y_py,
                u_py,
                v_py,
                luma_dims,
                chroma_dims,
            );
        }
    }
    w.finish().len()
}

/// Diagnostic counterpart to [`pick_hq_picture_qindex`]: returns
/// `(qindex, actual_picture_bytes)` so callers can inspect the chosen
/// quantiser's actual picture-byte cost relative to the supplied budget.
/// `actual_picture_bytes` is exactly [`hq_picture_payload_bytes_at_qindex`]
/// at the returned qindex.
pub fn hq_picture_qindex_diagnostic(
    sequence: &SequenceHeader,
    params: &EncoderParams,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    target_bytes: u32,
) -> (u32, usize) {
    let qindex = pick_hq_picture_qindex(sequence, params, y, u, v, target_bytes);
    let bytes = hq_picture_payload_bytes_at_qindex(sequence, params, y, u, v, qindex);
    (qindex, bytes)
}

/// Encode a single VC-2 HQ intra-only Dirac elementary stream sized to a
/// per-picture byte budget — the HQ analogue of
/// [`encode_single_ld_intra_stream_with_size_target`].
///
/// The picture-level qindex is picked by [`pick_hq_picture_qindex`] (a
/// single quantiser for every slice; `base.slice_size_target` is
/// intentionally cleared so the per-slice §13.5.4 search does not also
/// fire on top). The returned tuple is `(stream, chosen_qindex,
/// actual_picture_payload_bytes)`. `actual_picture_payload_bytes` is the
/// bytes between the picture's parse-info and the next parse-info — i.e.
/// the same `target_bytes` denominator the picker compared against.
///
/// The full stream additionally carries the sequence-header unit, the
/// picture's 13-byte parse-info, and the 13-byte end-of-sequence
/// parse-info — so `stream.len() = sh_unit_len + 13 +
/// actual_picture_payload_bytes + 13`.
pub fn encode_single_hq_intra_stream_with_size_target(
    sequence: &SequenceHeader,
    base: &EncoderParams,
    target_bytes: u32,
    picture_number: u32,
    y: &[u8],
    u: &[u8],
    v: &[u8],
) -> (Vec<u8>, u32, usize) {
    // The picker is a single picture-level qindex over the whole picture
    // — disable the per-slice §13.5.4 search so the chosen qindex is the
    // one that actually gets written into every slice header.
    let mut params = base.clone();
    params.slice_size_target = None;

    let qindex = pick_hq_picture_qindex(sequence, &params, y, u, v, target_bytes);
    params.qindex = qindex;

    let stream = encode_single_hq_intra_stream(sequence, &params, picture_number, y, u, v);
    let actual_picture_bytes =
        hq_picture_payload_bytes_at_qindex(sequence, &params, y, u, v, qindex);
    (stream, qindex, actual_picture_bytes)
}

fn write_transform_parameters(w: &mut BitWriter, params: &EncoderParams) {
    // wavelet_index, dwt_depth, [extended_transform_parameters], slice
    // parameters, quant matrix flag.
    let w_idx = wavelet_index(params.wavelet);
    w.write_uint(w_idx);
    w.write_uint(params.dwt_depth);
    // §12.4.4 `extended_transform_parameters()` — v3 only.
    //   - Default (`extended_transform_override == None`): emit the two
    //     `False` flag bits so the §12.4.4 NOTE applies and the IDWT is
    //     identical to v2.
    //   - Override `Some({wavelet_index_ho, dwt_depth_ho})`: emit
    //     `asym_transform_index_flag = (wavelet_index_ho != w_idx)` and
    //     `asym_transform_flag = (dwt_depth_ho != 0)`, each followed by
    //     the corresponding interleaved exp-Golomb value when the flag
    //     is `True` — see `EncoderParams::extended_transform_override`
    //     for the full asymmetric-pipeline semantics.
    if params.major_version >= 3 {
        match params.extended_transform_override {
            None => {
                w.write_bool(false); // asym_transform_index_flag
                w.write_bool(false); // asym_transform_flag
            }
            Some(ext) => {
                let asym_idx = ext.wavelet_index_ho != w_idx;
                w.write_bool(asym_idx);
                if asym_idx {
                    w.write_uint(ext.wavelet_index_ho);
                }
                let asym_depth = ext.dwt_depth_ho != 0;
                w.write_bool(asym_depth);
                if asym_depth {
                    w.write_uint(ext.dwt_depth_ho);
                }
            }
        }
    }
    // §12.4.5.2 slice_parameters — HQ branch.
    w.write_uint(params.slices_x);
    w.write_uint(params.slices_y);
    w.write_uint(params.slice_prefix_bytes);
    w.write_uint(params.slice_size_scaler);
    // §12.4.5.3 quant_matrix: emit flag=0 (default Annex E.1 table) or,
    // when `custom_quant_matrix` is set, flag=1 plus the explicit
    // per-subband entries. A custom matrix must be shaped for the
    // emitted `dwt_depth_ho`, or the decoder reads it mis-aligned.
    debug_assert!(
        !params.custom_quant_matrix || params.quant_matrix.dwt_depth_ho == params.dwt_depth_ho(),
        "custom quant matrix dwt_depth_ho ({}) must match the emitted §12.4.4.3 dwt_depth_ho ({})",
        params.quant_matrix.dwt_depth_ho,
        params.dwt_depth_ho(),
    );
    write_quant_matrix(w, params.custom_quant_matrix, &params.quant_matrix);
}

/// Emit the §12.4.5.3 (low-delay) / §11.3.5 `quant_matrix()` syntax.
///
/// When `custom == false` we write `custom_quant_matrix = False`; the
/// decoder then reconstructs the Annex E.1 default for the picture's
/// wavelet / depth via `set_quant_matrix()`. When `custom == true` we
/// write `custom_quant_matrix = True` followed by the explicit values
/// in the spec's exact read order — symmetric
/// (`matrix.dwt_depth_ho == 0`):
///
/// ```text
///   state[QMATRIX][0][LL] = read_uint()
///   for level = 1 to state[DWT_DEPTH]:
///     state[QMATRIX][level][HL] = read_uint()
///     state[QMATRIX][level][LH] = read_uint()
///     state[QMATRIX][level][HH] = read_uint()
/// ```
///
/// or asymmetric (`matrix.dwt_depth_ho > 0`): the level-0 **L** entry,
/// one **H** entry per level `1..=dwt_depth_ho`, then the HL/LH/HH
/// triplets for levels `dwt_depth_ho+1..=dwt_depth_ho+dwt_depth`.
///
/// `matrix.levels[level]` is `[LL, HL, LH, HH]` ([`Orient::as_index`]
/// ordering) with the single L / H entries in slot 0 (the
/// [`QuantMatrix::parse_custom`] storage convention), so the emission
/// is bit-exactly the order the decoder reads in
/// `picture::parse_transform_parameters`.
fn write_quant_matrix(w: &mut BitWriter, custom: bool, matrix: &QuantMatrix) {
    if !custom {
        w.write_bool(false);
        return;
    }
    w.write_bool(true);
    // Level 0: LL (symmetric) / L (asymmetric) — slot 0 either way.
    w.write_uint(matrix.levels[0][0]);
    // Asymmetric horizontal-only levels: one H entry each (slot 0).
    let ho = matrix.dwt_depth_ho as usize;
    for level in 1..=ho {
        w.write_uint(matrix.levels[level][0]);
    }
    // Levels dwt_depth_ho+1..=dwt_depth_ho+dwt_depth: HL, LH, HH.
    for level in ho + 1..=ho + matrix.dwt_depth as usize {
        let triplet = matrix.levels[level];
        w.write_uint(triplet[1]); // HL
        w.write_uint(triplet[2]); // LH
        w.write_uint(triplet[3]); // HH
    }
}

/// VC-2 §12.4.2 wavelet filter index. Public so that test code building
/// `extended_transform_parameters()` overrides can specify
/// `wavelet_index_ho` in terms of a `WaveletFilter` rather than a raw
/// integer constant.
pub fn wavelet_index(filter: WaveletFilter) -> u32 {
    match filter {
        WaveletFilter::DeslauriersDubuc9_7 => 0,
        WaveletFilter::LeGall5_3 => 1,
        WaveletFilter::DeslauriersDubuc13_7 => 2,
        WaveletFilter::Haar0 => 3,
        WaveletFilter::Haar1 => 4,
        WaveletFilter::Fidelity => 5,
        WaveletFilter::Daubechies9_7 => 6,
    }
}

/// Per-level subband dims for one component, asymmetric-aware
/// (§13.2.3): one `(width, height)` entry per level
/// `0..=dwt_depth_ho + dwt_depth`.
fn component_dims(
    comp_w: u32,
    comp_h: u32,
    dwt_depth: u32,
    dwt_depth_ho: u32,
) -> Vec<(usize, usize)> {
    (0..=dwt_depth_ho + dwt_depth)
        .map(|level| subband_dims_ho(comp_w, comp_h, dwt_depth, dwt_depth_ho, level))
        .collect()
}

/// Pad one component (width up to a multiple of
/// `2^(dwt_depth_ho + dwt_depth)`, height up to a multiple of
/// `2^dwt_depth` — §13.2.3), subtract the depth midpoint, run the
/// forward DWT and return the pyramid. With an active asymmetric
/// override the transform is [`dwt_with_ho`] (the §15.4.1 inverse:
/// `dwt_depth` 2-D levels atop `dwt_depth_ho` horizontal-only levels).
fn forward_component(
    plane: &[u8],
    comp_w: u32,
    comp_h: u32,
    depth: u32,
    params: &EncoderParams,
) -> Vec<[SubbandData; 4]> {
    let ho = params.dwt_depth_ho();
    let (pw, ph) = padded_component_dims_ho(comp_w, comp_h, params.dwt_depth, ho);
    let half: i32 = 1i32 << (depth - 1);
    let mut pic = SubbandData::new(pw, ph);
    for y in 0..ph {
        for x in 0..pw {
            let src_x = x.min(comp_w as usize - 1);
            let src_y = y.min(comp_h as usize - 1);
            let v = plane[src_y * comp_w as usize + src_x] as i32 - half;
            pic.set(y, x, v);
        }
    }
    if ho == 0 {
        dwt(&pic, params.wavelet, params.dwt_depth)
    } else {
        dwt_with_ho(
            &pic,
            params.wavelet,
            params.wavelet_ho(),
            params.dwt_depth,
            ho,
        )
    }
}

/// Forward quantisation (§13.3.1 inverse). The spec's inverse quant
/// formula is `x = sign(q) * (|q| * qf + offset + 2) / 4`. A matching
/// forward map that's exactly recovered when the qindex keeps qf small
/// is `q = sign(x) * ((|x| * 4) / qf)`. Rounding is floor-of-magnitude
/// to preserve the sign + zero-reconstruction invariants of the
/// dead-zone scheme.
fn quantise_coeff(x: i32, q: u32) -> i32 {
    if x == 0 {
        return 0;
    }
    let qf = crate::quant::quant_factor(q) as i64;
    // 4 * |x| / qf rounds magnitude toward zero; this is the standard
    // Dirac "dead-zone" forward quantiser. At q=0 (qf=4) it's a
    // no-op: 4*|x|/4 = |x|, and inverse_quant reconstructs |x| exactly.
    let mag = (x.unsigned_abs() as i64 * 4) / qf;
    if x < 0 {
        -(mag as i32)
    } else {
        mag as i32
    }
}

/// Slice-area coordinates within a subband (§13.5.6.2). Mirrors the
/// decoder's version.
fn slice_bounds(
    sx: u32,
    sy: u32,
    slices_x: u32,
    slices_y: u32,
    sub_w: usize,
    sub_h: usize,
) -> (usize, usize, usize, usize) {
    let left = (sub_w * sx as usize) / slices_x as usize;
    let right = (sub_w * (sx as usize + 1)) / slices_x as usize;
    let top = (sub_h * sy as usize) / slices_y as usize;
    let bottom = (sub_h * (sy as usize + 1)) / slices_y as usize;
    (left, right, top, bottom)
}

/// Encode one HQ slice (§13.5.4):
/// `slice_prefix_bytes | qindex | (len_y, Y-coeffs) | (len_u, U-coeffs) | (len_v, V-coeffs)`.
///
/// `qindex` is the quantiser this slice writes into its 1-byte header
/// and uses to quantise its own coefficient region. The three component
/// pyramids are the *unquantised* forward-DWT coefficients; this slice
/// quantises only its `(sx, sy)` region per [`slice_quantisers`].
#[allow(clippy::too_many_arguments)]
fn encode_hq_slice(
    w: &mut BitWriter,
    params: &EncoderParams,
    qindex: u32,
    sx: u32,
    sy: u32,
    y_py: &[[SubbandData; 4]],
    u_py: &[[SubbandData; 4]],
    v_py: &[[SubbandData; 4]],
    luma_dims: &[(usize, usize)],
    chroma_dims: &[(usize, usize)],
) {
    // Emit the slice-prefix zero bytes (application-defined).
    w.byte_align();
    for _ in 0..params.slice_prefix_bytes {
        w.write_uint_lit(1, 0);
    }
    // qindex is 1 byte (§13.5.2).
    w.write_uint_lit(1, qindex);

    // Per-subband effective quantisers for this slice's qindex.
    let q_per_level = slice_quantisers(qindex, &params.quant_matrix);

    // Per-component: emit a 1-byte length and the coefficient stream,
    // padded out to `length * slice_size_scaler` bytes.
    for comp in 0..3 {
        let dims: &[(usize, usize)] = if comp == 0 { luma_dims } else { chroma_dims };
        let py: &[[SubbandData; 4]] = match comp {
            0 => y_py,
            1 => u_py,
            _ => v_py,
        };
        let comp_bytes = encode_hq_component(params, &q_per_level, py, dims, sx, sy);
        let scaler = params.slice_size_scaler.max(1) as usize;
        // Round length up to a multiple of scaler; emit the byte
        // length divided by scaler as the length prefix.
        let padded_len = comp_bytes.len().div_ceil(scaler) * scaler;
        let length_byte = padded_len / scaler;
        debug_assert!(
            length_byte <= 255,
            "length byte overflow: {length_byte} bytes for slice ({sx},{sy}) comp {comp} — bump slice_size_scaler or use more slices",
        );
        let length_byte = length_byte as u32;
        w.write_uint_lit(1, length_byte);
        w.byte_align();
        w.write_bytes(&comp_bytes);
        // Pad to padded_len.
        for _ in comp_bytes.len()..padded_len {
            w.write_uint_lit(1, 0);
        }
    }
}

/// Serialise one component's coefficients for a single slice to a
/// fresh byte buffer, using bounded interleaved exp-Golomb + byte
/// alignment at the end. `q_per_level` holds the per-subband effective
/// quantisers for this slice (from [`slice_quantisers`]); the pyramid
/// `py` is *unquantised* and each coefficient is quantised on the fly.
fn encode_hq_component(
    params: &EncoderParams,
    q_per_level: &[[u32; 4]],
    py: &[[SubbandData; 4]],
    dims: &[(usize, usize)],
    sx: u32,
    sy: u32,
) -> Vec<u8> {
    let mut w = BitWriter::new();
    // §13.5.4 band order — shared symmetric/asymmetric sequence
    // (L/LL, any horizontal-only H bands, then HL/LH/HH triplets).
    for (level, orient) in slice_band_order(params.dwt_depth, params.dwt_depth_ho()) {
        write_slice_band(&mut w, level, orient, sx, sy, params, q_per_level, py, dims);
    }
    w.byte_align();
    w.finish()
}

#[allow(clippy::too_many_arguments)]
fn write_slice_band(
    w: &mut BitWriter,
    level: u32,
    orient: Orient,
    sx: u32,
    sy: u32,
    params: &EncoderParams,
    q_per_level: &[[u32; 4]],
    py: &[[SubbandData; 4]],
    dims: &[(usize, usize)],
) {
    let (sub_w, sub_h) = dims[level as usize];
    let (left, right, top, bottom) =
        slice_bounds(sx, sy, params.slices_x, params.slices_y, sub_w, sub_h);
    let band = &py[level as usize][orient.as_index()];
    if band.width == 0 || band.height == 0 {
        return;
    }
    let q = q_per_level[level as usize][orient.as_index()];
    for y in top..bottom {
        for x in left..right {
            let v = quantise_coeff(band.get(y, x), q);
            w.write_sint(v);
        }
    }
}

// -----------------------------------------------------------------
//   Sequence header, parse info, and full stream emit.
// -----------------------------------------------------------------

/// Emit a sequence header payload (not including the parse info).
/// Mirrors [`crate::sequence::parse_sequence_header`] on the read side.
///
/// For each overridable field, we only set the custom flag if the
/// caller's value differs from the preset's default. This keeps
/// Annex C preset-conformant headers (Level 1, §D.2.3) compact and
/// flag-free while still supporting ad-hoc customisations when the
/// caller picks `base_video_format_index = 0` (Custom).
pub fn encode_sequence_header(sequence: &SequenceHeader) -> Vec<u8> {
    use crate::video_format::{BaseVideoFormat, SignalRange};
    let mut w = BitWriter::new();
    // parse_parameters (§10.1).
    w.write_uint(sequence.parse_parameters.version_major);
    w.write_uint(sequence.parse_parameters.version_minor);
    w.write_uint(sequence.parse_parameters.profile);
    w.write_uint(sequence.parse_parameters.level);
    // base_video_format index — 0 (Custom) or 1..=20 (Annex C preset).
    w.write_uint(sequence.base_video_format_index);

    // Pull the preset defaults. For index 0 (Custom) lookup gives a
    // 640x480 stub; for a real preset this matches the spec exactly.
    let base = BaseVideoFormat::lookup(sequence.base_video_format_index)
        .unwrap_or_else(|| BaseVideoFormat::lookup(0).unwrap());
    let vp = &sequence.video_params;

    // Custom frame size — only emit the override if dims differ.
    let dims_match = vp.frame_width == base.frame_width && vp.frame_height == base.frame_height;
    w.write_bool(!dims_match);
    if !dims_match {
        w.write_uint(vp.frame_width);
        w.write_uint(vp.frame_height);
    }

    // Chroma sampling format.
    let chroma_match = vp.chroma_format == base.chroma_format;
    w.write_bool(!chroma_match);
    if !chroma_match {
        w.write_uint(vp.chroma_format.to_index());
    }

    // Scan format — we always carry the base default (progressive).
    let scan_match = vp.source_sampling == base.source_sampling;
    w.write_bool(!scan_match);
    if !scan_match {
        let idx = match vp.source_sampling {
            crate::video_format::ScanFormat::Progressive => 0,
            crate::video_format::ScanFormat::Interlaced => 1,
        };
        w.write_uint(idx);
    }

    // Frame rate — omit when the base preset already matches.
    let fr_match = vp.frame_rate_numer == base.frame_rate_numer
        && vp.frame_rate_denom == base.frame_rate_denom;
    w.write_bool(!fr_match);
    if !fr_match {
        w.write_uint(0); // custom: emit explicit numer/denom.
        w.write_uint(vp.frame_rate_numer);
        w.write_uint(vp.frame_rate_denom);
    }

    // Pixel aspect ratio.
    let par_match = vp.pixel_aspect_ratio_numer == base.pixel_aspect_ratio_numer
        && vp.pixel_aspect_ratio_denom == base.pixel_aspect_ratio_denom;
    w.write_bool(!par_match);
    if !par_match {
        w.write_uint(0);
        w.write_uint(vp.pixel_aspect_ratio_numer);
        w.write_uint(vp.pixel_aspect_ratio_denom);
    }

    // Clean area.
    let clean_match = vp.clean_width == base.clean_width
        && vp.clean_height == base.clean_height
        && vp.clean_left_offset == base.clean_left_offset
        && vp.clean_top_offset == base.clean_top_offset;
    w.write_bool(!clean_match);
    if !clean_match {
        w.write_uint(vp.clean_width);
        w.write_uint(vp.clean_height);
        w.write_uint(vp.clean_left_offset);
        w.write_uint(vp.clean_top_offset);
    }

    // Signal range — emit a preset index when it matches, else custom.
    let sr_match = vp.signal_range == base.signal_range;
    w.write_bool(!sr_match);
    if !sr_match {
        let preset_idx = if vp.signal_range == SignalRange::PRESET_8BIT_FULL {
            1
        } else if vp.signal_range == SignalRange::PRESET_8BIT_VIDEO {
            2
        } else if vp.signal_range == SignalRange::PRESET_10BIT_VIDEO {
            3
        } else if vp.signal_range == SignalRange::PRESET_12BIT_VIDEO {
            4
        } else {
            0
        };
        w.write_uint(preset_idx);
        if preset_idx == 0 {
            w.write_uint(vp.signal_range.luma_offset);
            w.write_uint(vp.signal_range.luma_excursion);
            w.write_uint(vp.signal_range.chroma_offset);
            w.write_uint(vp.signal_range.chroma_excursion);
        }
    }

    // Colour spec — always default.
    w.write_bool(false);

    // picture_coding_mode.
    let pcm_idx = match sequence.picture_coding_mode {
        crate::sequence::PictureCodingMode::Frames => 0,
        crate::sequence::PictureCodingMode::Fields => 1,
    };
    w.write_uint(pcm_idx);
    w.byte_align();
    w.finish()
}

/// Write a 13-byte parse info header into `out` at `out.len()`.
pub fn write_parse_info(
    out: &mut Vec<u8>,
    parse_code: u8,
    next_parse_offset: u32,
    previous_parse_offset: u32,
) {
    out.extend_from_slice(BBCD);
    out.push(parse_code);
    out.extend_from_slice(&next_parse_offset.to_be_bytes());
    out.extend_from_slice(&previous_parse_offset.to_be_bytes());
}

/// Encode one 64x64 (or other) 4:2:0 8-bit YUV frame as a complete
/// VC-2 HQ intra-only Dirac elementary stream, including sequence
/// header, a single non-reference HQ intra picture (parse code `0xE8`)
/// and an end-of-sequence marker.
pub fn encode_single_hq_intra_stream(
    sequence: &SequenceHeader,
    params: &EncoderParams,
    picture_number: u32,
    y: &[u8],
    u: &[u8],
    v: &[u8],
) -> Vec<u8> {
    // 1) Sequence header payload.
    let sh_payload = encode_sequence_header(sequence);
    // 2) Picture payload.
    let pic_payload = encode_hq_intra_picture(sequence, params, picture_number, y, u, v);

    // Lay out: [pi_sh][sh][pi_pic][pic][pi_eos].
    let pi_size = 13usize;
    let sh_unit_len = pi_size + sh_payload.len();
    let pic_unit_len = pi_size + pic_payload.len();
    // End-of-sequence parse info only.
    let eos_unit_len = pi_size;

    let mut out = Vec::with_capacity(sh_unit_len + pic_unit_len + eos_unit_len);
    write_parse_info(&mut out, 0x00, sh_unit_len as u32, 0);
    out.extend_from_slice(&sh_payload);
    write_parse_info(
        &mut out,
        0xE8, // HQ non-reference intra picture.
        pic_unit_len as u32,
        sh_unit_len as u32,
    );
    out.extend_from_slice(&pic_payload);
    write_parse_info(&mut out, 0x10, 0, pic_unit_len as u32);
    out
}

/// Build a minimal sequence header describing an `WxH` progressive
/// 4:2:0 8-bit stream (or 4:4:4 / 4:2:2 depending on `chroma`).
///
/// Produces an HQ-profile (profile=3) sequence header — suitable for
/// streams emitted via [`encode_single_hq_intra_stream`]. For LD
/// streams, call [`make_minimal_sequence_ld`] (or patch `profile` to
/// 0 — §D.1.1) so the parse parameters match the payload syntax.
///
/// `base_video_format_index` is 0 (Custom) with the width / height /
/// chroma / signal-range fields carried explicitly so the header can
/// describe any `(W, H, chroma)`. Level is set to 0 (RESERVED per
/// §D.2) because §D.2.3 Level 1 forbids all custom flags and our
/// sequences override at least dimensions; level 128 is Main-LongGOP
/// only. Applications that need a conformant level should use
/// [`make_preset_sequence`] with an Annex C index.
pub fn make_minimal_sequence(
    frame_width: u32,
    frame_height: u32,
    chroma: ChromaFormat,
) -> SequenceHeader {
    make_minimal_sequence_for(frame_width, frame_height, chroma, 3)
}

/// Same as [`make_minimal_sequence`] but with `profile = 0` — the
/// Low Delay VC-2 profile (§D.1.1). Matches streams emitted by
/// [`encode_single_ld_intra_stream`].
pub fn make_minimal_sequence_ld(
    frame_width: u32,
    frame_height: u32,
    chroma: ChromaFormat,
) -> SequenceHeader {
    make_minimal_sequence_for(frame_width, frame_height, chroma, 0)
}

fn make_minimal_sequence_for(
    frame_width: u32,
    frame_height: u32,
    chroma: ChromaFormat,
    profile: u32,
) -> SequenceHeader {
    use crate::sequence::{ParseParameters, PictureCodingMode, VideoParams};
    use crate::video_format::{ScanFormat, SignalRange};
    let signal_range = SignalRange::PRESET_8BIT_FULL;
    let vp = VideoParams {
        frame_width,
        frame_height,
        chroma_format: chroma,
        source_sampling: ScanFormat::Progressive,
        top_field_first: false,
        frame_rate_numer: 25,
        frame_rate_denom: 1,
        pixel_aspect_ratio_numer: 1,
        pixel_aspect_ratio_denom: 1,
        clean_width: frame_width,
        clean_height: frame_height,
        clean_left_offset: 0,
        clean_top_offset: 0,
        signal_range,
    };
    let (chroma_w, chroma_h) = match chroma {
        ChromaFormat::Yuv444 => (frame_width, frame_height),
        ChromaFormat::Yuv422 => (frame_width / 2, frame_height),
        ChromaFormat::Yuv420 => (frame_width / 2, frame_height / 2),
    };
    SequenceHeader {
        parse_parameters: ParseParameters {
            version_major: 2,
            version_minor: 0,
            profile,
            level: 0, // RESERVED — streams with overridden dims don't
                      // fit Level 1 (§D.2.3) or Level 128 (Main-LongGOP only).
        },
        base_video_format_index: 0,
        video_params: vp,
        picture_coding_mode: PictureCodingMode::Frames,
        luma_width: frame_width,
        luma_height: frame_height,
        chroma_width: chroma_w,
        chroma_height: chroma_h,
        luma_depth: 8,
        chroma_depth: 8,
    }
}

/// Build a sequence header that fits an Annex C preset exactly
/// (base_video_format indices 1..=20) with no custom-field overrides.
/// This is the path required by §D.2.3 Level 1: every `custom_*_flag`
/// stays False and the stream declares `level = 1`.
///
/// `profile` should be 0 for LD (§D.1.1) or 3 for VC-2 HQ.
/// Returns `None` if `preset_index` is not in `1..=20`.
pub fn make_preset_sequence(preset_index: u32, profile: u32) -> Option<SequenceHeader> {
    use crate::sequence::{ParseParameters, PictureCodingMode, VideoParams};
    use crate::video_format::BaseVideoFormat;
    let base = BaseVideoFormat::lookup(preset_index)?;
    if !(1..=20).contains(&preset_index) {
        return None;
    }
    let vp: VideoParams = base.into();
    let (chroma_w, chroma_h) = match vp.chroma_format {
        ChromaFormat::Yuv444 => (vp.frame_width, vp.frame_height),
        ChromaFormat::Yuv422 => (vp.frame_width / 2, vp.frame_height),
        ChromaFormat::Yuv420 => (vp.frame_width / 2, vp.frame_height / 2),
    };
    let luma_depth = intlog2_ceil_u32(vp.signal_range.luma_excursion + 1);
    let chroma_depth = intlog2_ceil_u32(vp.signal_range.chroma_excursion + 1);
    let luma_w = vp.frame_width;
    let luma_h = vp.frame_height;
    Some(SequenceHeader {
        parse_parameters: ParseParameters {
            version_major: 2,
            version_minor: 0,
            profile,
            level: 1, // VC-2 default level.
        },
        base_video_format_index: preset_index,
        video_params: vp,
        picture_coding_mode: PictureCodingMode::Frames,
        luma_width: luma_w,
        luma_height: luma_h,
        chroma_width: chroma_w,
        chroma_height: chroma_h,
        luma_depth,
        chroma_depth,
    })
}

fn intlog2_ceil_u32(n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        32 - (n - 1).leading_zeros()
    }
}

// -----------------------------------------------------------------
//   VC-2 LD (Low-Delay) profile — intra only.
//
//   LD slices carry a fixed byte budget derived from
//   `slice_bytes_numer / slice_bytes_denom` (§13.5.3.2). Each slice
//   layout is:
//     qindex                      — 7 raw bits
//     slice_y_length              — intlog2(total_bits - 7) raw bits
//     luma coefficients           — slice_y_length bits (Funnel-bounded
//                                   interleaved exp-Golomb)
//     chroma (U,V) coefficients   — remaining bits, interleaved per
//                                   coefficient position
//
//   On the decode side the Funnel returns `1` for any bit read past
//   the end of the budget, which terminates a bounded uintb read with
//   value `0`. The encoder mirrors that by right-padding each bounded
//   region with `1` bits — the decoder then reads a string of zeroes,
//   which is exactly what the encoder skipped emitting.
//
//   DC prediction (§13.4) is encoder-side subtractive: we subtract the
//   `mean3` of the neighbour (a, b, c) cells from each LL coefficient
//   before quantisation-serialisation. The decoder adds the same
//   predictor back in after inverse quantisation.
// -----------------------------------------------------------------

/// LD-profile parameters. Fixed per-slice byte budget (§13.5.3.2) is
/// `slice_bytes_numer / slice_bytes_denom` per slice, spread across all
/// `slices_x * slices_y` slices.
#[derive(Debug, Clone)]
pub struct LdEncoderParams {
    pub wavelet: WaveletFilter,
    pub dwt_depth: u32,
    pub slices_x: u32,
    pub slices_y: u32,
    /// Total-bytes numerator — actual bytes for slice (sx,sy) are
    /// distributed from this via `slice_bytes()`.
    pub slice_bytes_numer: u32,
    pub slice_bytes_denom: u32,
    pub quant_matrix: QuantMatrix,
    /// When `true`, emit `quant_matrix.levels` as a §12.4.5.3 *custom*
    /// quantisation matrix (`custom_quant_matrix = True`) instead of the
    /// `False` flag that makes the decoder use the Annex E.1 default. See
    /// [`EncoderParams::custom_quant_matrix`] for the full rationale.
    pub custom_quant_matrix: bool,
    /// Per-slice qindex (0..=127). 0 is near-lossless for coefficients
    /// in the range where the dead-zone quantiser's forward rounding is
    /// exact.
    pub qindex: u32,
    /// Bitstream major version selector for the `transform_parameters()`
    /// block. `2` (default) writes the v2 syntax. `3` writes the v3
    /// syntax (SMPTE ST 2042-1:2022 §12.4.4) — the
    /// `extended_transform_parameters()` flag pair both at `False`
    /// (symmetric-default form per the §12.4.4 NOTE). See
    /// [`EncoderParams::major_version`] for the full semantics; the LD
    /// path emits the same two flag bits in the same position relative
    /// to `dwt_depth`.
    pub major_version: u32,
    /// LD counterpart to [`EncoderParams::extended_transform_override`].
    /// Only consulted when [`Self::major_version`] is `>= 3`. See
    /// [`EncoderParams::extended_transform_override`] for the full
    /// semantics — the LD path emits exactly the same bit-level
    /// `extended_transform_parameters()` block in the same position
    /// relative to `dwt_depth`.
    pub extended_transform_override: Option<ExtendedTransformOverride>,
}

impl LdEncoderParams {
    /// Default LD parameters with a generous per-slice byte budget so
    /// coefficients fit without truncation on small round-trip test
    /// pictures. `bytes_per_slice` bytes per slice × `slices_x *
    /// slices_y` slices gives the total.
    pub fn default_ld(
        wavelet: WaveletFilter,
        dwt_depth: u32,
        slices_x: u32,
        slices_y: u32,
        bytes_per_slice: u32,
    ) -> Self {
        let quant_matrix = QuantMatrix::default_for(wavelet, dwt_depth)
            .expect("default quant matrix exists for depth <= 4");
        let total = bytes_per_slice * slices_x * slices_y;
        Self {
            wavelet,
            dwt_depth,
            slices_x,
            slices_y,
            slice_bytes_numer: total,
            slice_bytes_denom: slices_x * slices_y,
            quant_matrix,
            custom_quant_matrix: false,
            qindex: 0,
            major_version: 2,
            extended_transform_override: None,
        }
    }

    /// Replace the quantisation matrix with `matrix` and set
    /// `custom_quant_matrix = true` so it travels in-band (§12.4.5.3).
    pub fn with_custom_quant_matrix(mut self, matrix: QuantMatrix) -> Self {
        self.quant_matrix = matrix;
        self.custom_quant_matrix = true;
        self
    }

    /// Select the v3 `transform_parameters()` syntax (SMPTE ST
    /// 2042-1:2022 §12.4.4). The LD path emits the same two
    /// `extended_transform_parameters()` `False` flag bits as the HQ
    /// counterpart; see [`EncoderParams::with_major_version_3`] for
    /// the full description.
    pub fn with_major_version_3(mut self) -> Self {
        self.major_version = 3;
        self
    }

    /// Override the §12.4.4 `extended_transform_parameters()` emission
    /// with explicit asymmetric values. Only takes effect when
    /// [`Self::major_version`] is `>= 3`. LD counterpart to
    /// [`EncoderParams::with_extended_transform_override`] — see that
    /// method for the full semantics; with `dwt_depth_ho > 0` prefer
    /// [`Self::with_asymmetric_transform`], which also installs the
    /// matching custom asymmetric quant matrix.
    pub fn with_extended_transform_override(
        mut self,
        override_: ExtendedTransformOverride,
    ) -> Self {
        self.extended_transform_override = Some(override_);
        self
    }

    /// Configure a fully-wired §12.4.4 asymmetric (horizontal-only)
    /// transform on the LD path. LD counterpart to
    /// [`EncoderParams::with_asymmetric_transform`] — selects v3,
    /// installs the override and an all-zero custom quant matrix in
    /// the §12.4.5.3 asymmetric shape. The caller must still set the
    /// sequence header's `parse_parameters.version_major` to `3`.
    pub fn with_asymmetric_transform(
        mut self,
        wavelet_ho: WaveletFilter,
        dwt_depth_ho: u32,
    ) -> Self {
        self.major_version = 3;
        self.extended_transform_override = Some(ExtendedTransformOverride {
            wavelet_index_ho: wavelet_index(wavelet_ho),
            dwt_depth_ho,
        });
        self.quant_matrix = zero_quant_matrix(self.dwt_depth, dwt_depth_ho);
        self.custom_quant_matrix = true;
        self
    }

    /// Effective §12.4.4.3 `dwt_depth_ho` — see
    /// [`EncoderParams::dwt_depth_ho`].
    pub fn dwt_depth_ho(&self) -> u32 {
        if self.major_version >= 3 {
            self.extended_transform_override
                .map_or(0, |e| e.dwt_depth_ho)
        } else {
            0
        }
    }

    /// Effective §12.4.4.2 horizontal-only wavelet filter — see
    /// [`EncoderParams::wavelet_ho`].
    pub fn wavelet_ho(&self) -> WaveletFilter {
        if self.major_version >= 3 {
            self.extended_transform_override
                .and_then(|e| WaveletFilter::from_index(e.wavelet_index_ho))
                .unwrap_or(self.wavelet)
        } else {
            self.wavelet
        }
    }
}

/// Number of raw bits used by the slice-header `slice_y_length` field
/// (§13.5.2).
///
/// The VC-2 / SMPTE ST 2042-1 reference — and ffmpeg's `vc2` decoder —
/// size this field as `floor(log2(total_bits)) + 1` where
/// `total_bits = 8 * slice_bytes`, i.e. the minimum number of bits
/// required to represent any value in `[0, total_bits - 1]`. This
/// agrees with the Dirac formula `intlog2(8*slice_bytes - 7)` whenever
/// `total_bits` is not an exact power of two, and is one bit larger
/// otherwise. Using the Dirac formula at power-of-two slice sizes
/// (e.g. `slice_bytes ∈ {32, 64, 128, 256}`) causes a single-bit shift
/// in the luma coefficient stream that ffmpeg interprets as garbage —
/// the root cause of the Round 8 LD interop regression.
pub(crate) fn ld_length_bits(slice_bytes: u32) -> u32 {
    let total_bits = slice_bytes.saturating_mul(8).max(1);
    32 - total_bits.leading_zeros()
}

/// `slice_bytes(sx, sy)` (§13.5.3.2). Mirrors the decoder helper.
fn slice_bytes(slices_x: u32, numer: u32, denom: u32, sx: u32, sy: u32) -> u32 {
    let slice_num = sy as u64 * slices_x as u64 + sx as u64;
    let a = ((slice_num + 1) * numer as u64) / denom as u64;
    let b = (slice_num * numer as u64) / denom as u64;
    (a - b) as u32
}

/// Forward counterpart to [`intra_dc_prediction`]: subtract from each
/// LL coefficient the `mean3` of its already-processed neighbours.
/// After this runs the decoder's `intra_dc_prediction` recovers the
/// originals exactly.
///
/// The prediction context is computed against the *original* band,
/// matching the decoder which uses the *reconstructed* band values as
/// it fills in each sample in raster order. When the decoder does
/// `cur.wrapping_add(prediction(already-reconstructed cells))`, we
/// want the post-subtraction sample to be `orig - prediction(orig of
/// already-reconstructed cells)`, i.e. we must use the *original*
/// values of the already-visited cells as the prediction context.
pub fn forward_dc_prediction(band: &mut SubbandData) {
    // Capture the raw coefficients first so predictions use the
    // pre-subtraction (= post-reconstruction) neighbours.
    let orig = band.data.clone();
    let w = band.width;
    for y in 0..band.height {
        for x in 0..w {
            let prediction: i32 = if x > 0 && y > 0 {
                let a = orig[y * w + (x - 1)];
                let b = orig[(y - 1) * w + (x - 1)];
                let c = orig[(y - 1) * w + x];
                // §5.4 unbiased mean `(a + b + c + 1) // 3` (floor div),
                // mirroring the decoder's `mean3` exactly.
                let s = a as i64 + b as i64 + c as i64 + 1;
                s.div_euclid(3) as i32
            } else if x > 0 {
                orig[y * w + (x - 1)]
            } else if y > 0 {
                orig[(y - 1) * w]
            } else {
                0
            };
            let cur = band.get(y, x);
            band.set(y, x, cur.wrapping_sub(prediction));
        }
    }
}

/// Encode a single LD intra picture payload (no parse info).
///
/// Mirrors [`crate::picture::decode_low_delay_picture`] on the LD path:
/// picture number, transform parameters, then `slices_x * slices_y`
/// slices each of exactly `slice_bytes(sx, sy)` bytes.
pub fn encode_ld_intra_picture(
    sequence: &SequenceHeader,
    params: &LdEncoderParams,
    picture_number: u32,
    y: &[u8],
    u: &[u8],
    v: &[u8],
) -> Vec<u8> {
    let mut w = BitWriter::new();

    // §12.2 picture_header.
    w.byte_align();
    w.write_uint_lit(4, picture_number);

    // §12.3 wavelet_transform — intra, so no zero_residual flag.
    w.byte_align();
    write_ld_transform_parameters(&mut w, params);
    w.byte_align();

    // Per-component coefficient pyramids (forward DWT + forward
    // quantisation + forward DC-prediction subtraction on LL).
    let luma_w = sequence.luma_width;
    let luma_h = sequence.luma_height;
    let chroma_w = sequence.chroma_width;
    let chroma_h = sequence.chroma_height;

    let y_py = forward_component_ld(y, luma_w, luma_h, sequence.luma_depth, params);
    let u_py = forward_component_ld(u, chroma_w, chroma_h, sequence.chroma_depth, params);
    let v_py = forward_component_ld(v, chroma_w, chroma_h, sequence.chroma_depth, params);

    let q_per_level = slice_quantisers(params.qindex, &params.quant_matrix);
    let mut y_qpy = quantise_pyramid_ld(&y_py, &q_per_level);
    let mut u_qpy = quantise_pyramid_ld(&u_py, &q_per_level);
    let mut v_qpy = quantise_pyramid_ld(&v_py, &q_per_level);

    // Forward DC prediction on the LL band (level 0).
    forward_dc_prediction(&mut y_qpy[0][0]);
    forward_dc_prediction(&mut u_qpy[0][0]);
    forward_dc_prediction(&mut v_qpy[0][0]);

    let luma_dims = component_dims(luma_w, luma_h, params.dwt_depth, params.dwt_depth_ho());
    let chroma_dims = component_dims(chroma_w, chroma_h, params.dwt_depth, params.dwt_depth_ho());

    for sy in 0..params.slices_y {
        for sx in 0..params.slices_x {
            encode_ld_slice(
                &mut w,
                params,
                sx,
                sy,
                &y_qpy,
                &u_qpy,
                &v_qpy,
                &luma_dims,
                &chroma_dims,
            );
        }
    }

    w.finish()
}

/// Number of bytes a complete LD picture payload occupies when
/// `params` is used verbatim. Independent of the input samples and of
/// `params.qindex` — every LD slice writes exactly `slice_bytes(sx,sy)`
/// bytes (the Funnel-bounded 1-padding in [`write_funnel_bounded`] keeps
/// each slice's emitted byte count equal to its budget regardless of
/// quantiser or content). Picture-byte count = `picture_header(4) +
/// byte-aligned transform_parameters + slice_bytes_numer`.
///
/// This is the LD analogue of "every HQ slice's emitted length byte ×
/// `slice_size_scaler` equals its serialised coefficient block": both
/// profiles cap their picture size at the configured budget, but LD
/// caps every slice deterministically (the §13.5.3.2 fixed-rate path)
/// while HQ caps only by `slice_size_scaler × length_byte` and so can
/// be content-dependent unless §13.5.4 escalates the qindex.
pub fn ld_picture_payload_bytes(params: &LdEncoderParams) -> usize {
    // 4-byte picture_number after the leading byte_align in
    // `encode_ld_intra_picture` (which is a no-op at offset 0).
    let mut w = BitWriter::new();
    w.byte_align();
    w.write_uint_lit(4, 0); // picture_number value doesn't change the size
    w.byte_align();
    write_ld_transform_parameters(&mut w, params);
    w.byte_align();
    let header_bytes = w.finish().len();
    header_bytes + params.slice_bytes_numer as usize
}

/// VC-2 LD §13.5.4 picture-level qindex picker.
///
/// LD does NOT carry a per-slice qindex on the wire in the way HQ does
/// (it does — each LD slice header still holds `qindex = read_nbits(7)`,
/// but the SMPTE ST 2042-1 §13.5.3 rate-control model is *per-picture*:
/// every slice in an LD picture writes the same qindex, and the picture
/// budget is `slice_bytes_numer` plus a fixed header. With that model,
/// the rate-control knob is `qindex` itself — pick the **lowest** qindex
/// (best quality) for which the coefficient payload of every slice fits
/// into its `payload_bits` budget without truncation.
///
/// Returns the smallest `qindex ∈ 0..=127` for which **every** slice
/// satisfies `luma_bits + chroma_bits <= payload_bits`. If even q=127
/// overflows on some slice the picker returns 127 (the most aggressive
/// quantiser available; the encoder will simply truncate / 1-pad on
/// those slices, the decoder reads past-end zeros — i.e. a graceful
/// degradation rather than an encode failure).
///
/// Picture byte count is independent of qindex once `params.slice_bytes_numer
/// / slice_bytes_denom` are set — every LD slice always emits its full
/// budget. So the picker's job is purely to **maximise quality** subject
/// to the picture-budget constraint that the caller has baked into
/// `slice_bytes_numer`. Pair with [`derive_ld_slice_bytes_for_target`]
/// to first size `slice_bytes_numer` to a target picture-byte budget.
pub fn pick_ld_picture_qindex(
    sequence: &SequenceHeader,
    params: &LdEncoderParams,
    y: &[u8],
    u: &[u8],
    v: &[u8],
) -> u32 {
    let luma_w = sequence.luma_width;
    let luma_h = sequence.luma_height;
    let chroma_w = sequence.chroma_width;
    let chroma_h = sequence.chroma_height;

    // Unquantised forward DWT, once. Quantisation is the only step that
    // varies with qindex.
    let y_py = forward_component_ld(y, luma_w, luma_h, sequence.luma_depth, params);
    let u_py = forward_component_ld(u, chroma_w, chroma_h, sequence.chroma_depth, params);
    let v_py = forward_component_ld(v, chroma_w, chroma_h, sequence.chroma_depth, params);

    let luma_dims = component_dims(luma_w, luma_h, params.dwt_depth, params.dwt_depth_ho());
    let chroma_dims = component_dims(chroma_w, chroma_h, params.dwt_depth, params.dwt_depth_ho());

    let floor = params.qindex.min(127);
    for qindex in floor..=127 {
        if ld_picture_fits(
            params,
            qindex,
            &y_py,
            &u_py,
            &v_py,
            &luma_dims,
            &chroma_dims,
        ) {
            return qindex;
        }
    }
    127
}

/// Per-slice diagnostic for the LD picture-qindex picker. Returns
/// `(qindex, max_slice_bit_overflow)` where `qindex` is the value the
/// encoder would write into every slice header (a single picture-level
/// value, mirroring [`pick_ld_picture_qindex`]) and
/// `max_slice_bit_overflow` is the largest `luma_bits + chroma_bits -
/// payload_bits` across all slices at that qindex (0 if all slices fit;
/// positive when some slice's content would be Funnel-truncated). This
/// is the LD analogue of [`hq_slice_qindexes`].
pub fn ld_picture_qindex_diagnostic(
    sequence: &SequenceHeader,
    params: &LdEncoderParams,
    y: &[u8],
    u: &[u8],
    v: &[u8],
) -> (u32, i64) {
    let qindex = pick_ld_picture_qindex(sequence, params, y, u, v);
    let luma_w = sequence.luma_width;
    let luma_h = sequence.luma_height;
    let chroma_w = sequence.chroma_width;
    let chroma_h = sequence.chroma_height;

    let y_py = forward_component_ld(y, luma_w, luma_h, sequence.luma_depth, params);
    let u_py = forward_component_ld(u, chroma_w, chroma_h, sequence.chroma_depth, params);
    let v_py = forward_component_ld(v, chroma_w, chroma_h, sequence.chroma_depth, params);

    let luma_dims = component_dims(luma_w, luma_h, params.dwt_depth, params.dwt_depth_ho());
    let chroma_dims = component_dims(chroma_w, chroma_h, params.dwt_depth, params.dwt_depth_ho());

    let q_per_level = slice_quantisers(qindex, &params.quant_matrix);
    let mut y_qpy = quantise_pyramid_ld(&y_py, &q_per_level);
    let mut u_qpy = quantise_pyramid_ld(&u_py, &q_per_level);
    let mut v_qpy = quantise_pyramid_ld(&v_py, &q_per_level);
    forward_dc_prediction(&mut y_qpy[0][0]);
    forward_dc_prediction(&mut u_qpy[0][0]);
    forward_dc_prediction(&mut v_qpy[0][0]);

    let mut worst: i64 = 0;
    for sy in 0..params.slices_y {
        for sx in 0..params.slices_x {
            let slice_n_bytes = slice_bytes(
                params.slices_x,
                params.slice_bytes_numer,
                params.slice_bytes_denom,
                sx,
                sy,
            );
            let total_bits = 8u64 * slice_n_bytes as u64;
            let length_bits = ld_length_bits(slice_n_bytes);
            let header_bits = 7u64 + length_bits as u64;
            let payload_bits = total_bits.saturating_sub(header_bits) as i64;

            let mut luma_tmp = BitWriter::new();
            write_ld_component(&mut luma_tmp, params, &y_qpy, &luma_dims, sx, sy, true);
            let luma_bits = luma_tmp.measured_bits() as i64;
            let mut chroma_tmp = BitWriter::new();
            write_ld_chroma_interleaved(
                &mut chroma_tmp,
                params,
                &u_qpy,
                &v_qpy,
                &chroma_dims,
                sx,
                sy,
            );
            let chroma_bits = chroma_tmp.measured_bits() as i64;
            let overflow = (luma_bits + chroma_bits) - payload_bits;
            if overflow > worst {
                worst = overflow;
            }
        }
    }
    (qindex, worst)
}

/// True iff every slice's `luma_bits + chroma_bits` fits within
/// `payload_bits` at the given `qindex` — i.e. the LD picture survives
/// encoding at this qindex without any Funnel-truncation. Helper for
/// [`pick_ld_picture_qindex`].
#[allow(clippy::too_many_arguments)]
fn ld_picture_fits(
    params: &LdEncoderParams,
    qindex: u32,
    y_py: &[[SubbandData; 4]],
    u_py: &[[SubbandData; 4]],
    v_py: &[[SubbandData; 4]],
    luma_dims: &[(usize, usize)],
    chroma_dims: &[(usize, usize)],
) -> bool {
    let q_per_level = slice_quantisers(qindex, &params.quant_matrix);
    let mut y_qpy = quantise_pyramid_ld(y_py, &q_per_level);
    let mut u_qpy = quantise_pyramid_ld(u_py, &q_per_level);
    let mut v_qpy = quantise_pyramid_ld(v_py, &q_per_level);
    forward_dc_prediction(&mut y_qpy[0][0]);
    forward_dc_prediction(&mut u_qpy[0][0]);
    forward_dc_prediction(&mut v_qpy[0][0]);

    for sy in 0..params.slices_y {
        for sx in 0..params.slices_x {
            let slice_n_bytes = slice_bytes(
                params.slices_x,
                params.slice_bytes_numer,
                params.slice_bytes_denom,
                sx,
                sy,
            );
            if slice_n_bytes == 0 {
                return false;
            }
            let total_bits = 8u64 * slice_n_bytes as u64;
            let length_bits = ld_length_bits(slice_n_bytes);
            let header_bits = 7u64 + length_bits as u64;
            if total_bits <= header_bits {
                return false;
            }
            let payload_bits = total_bits - header_bits;

            let mut luma_tmp = BitWriter::new();
            write_ld_component(&mut luma_tmp, params, &y_qpy, luma_dims, sx, sy, true);
            let luma_bits = luma_tmp.measured_bits();
            let mut chroma_tmp = BitWriter::new();
            write_ld_chroma_interleaved(
                &mut chroma_tmp,
                params,
                &u_qpy,
                &v_qpy,
                chroma_dims,
                sx,
                sy,
            );
            let chroma_bits = chroma_tmp.measured_bits();

            if luma_bits + chroma_bits > payload_bits {
                return false;
            }
        }
    }
    true
}

/// Derive `slice_bytes_numer` so that an LD picture encoded with the
/// returned [`LdEncoderParams`] occupies ≈ `target_picture_bytes`. The
/// returned params are `base` cloned with `slice_bytes_numer` updated
/// and `slice_bytes_denom = slices_x * slices_y` (so each slice gets
/// exactly `slice_bytes_numer / (slices_x * slices_y)` bytes — uniform
/// per-slice budget, the standard §13.5.3.2 setup for a single-rate
/// picture).
///
/// Picture-byte size = `ld_picture_payload_bytes(adjusted)` which is
/// header (picture_number + transform_parameters, both byte-aligned)
/// plus `slice_bytes_numer`. We solve `header(numer) + numer ==
/// target_picture_bytes` for `numer` by one fixed-point pass — the
/// header size depends on `numer` (interleaved exp-Golomb), but only
/// very weakly (a 32-bit numer needs at most ~13 bytes versus an 8-bit
/// numer's ~3 bytes), so a single iteration suffices in practice.
///
/// Returns `None` if `target_picture_bytes` is too small to fit the
/// header (i.e. the resulting `slice_bytes_numer` would be ≤ 0 or each
/// slice would not even hold its 7-bit qindex + length-bits header).
/// Each slice must hold at minimum 2 bytes (qindex + 1-bit length_bits
/// plus a single coefficient bit), so the minimum picture target is
/// `header + 2 * slices_x * slices_y`.
pub fn derive_ld_slice_bytes_for_target(
    base: &LdEncoderParams,
    target_picture_bytes: u32,
) -> Option<LdEncoderParams> {
    let n_slices = base.slices_x.checked_mul(base.slices_y)?;
    if n_slices == 0 {
        return None;
    }
    // Initial guess: assume header is small (~10 bytes).
    let mut numer = target_picture_bytes.saturating_sub(10);
    // Iterate at most a handful of times — each round refines the
    // header-size estimate. Two rounds is enough since header growth
    // is sub-linear in `numer`.
    for _ in 0..4 {
        if numer == 0 {
            return None;
        }
        let candidate = LdEncoderParams {
            slice_bytes_numer: numer,
            slice_bytes_denom: n_slices,
            ..base.clone()
        };
        let actual = ld_picture_payload_bytes(&candidate);
        if actual == target_picture_bytes as usize {
            // Bytes-per-slice must be > 0 and large enough to hold the
            // slice header. With uniform denominator the per-slice
            // budget is `numer / n_slices` which is `>= 2` once
            // target_picture_bytes exceeds `header + 2*n_slices`.
            let bps = numer / n_slices;
            if bps < 2 {
                return None;
            }
            return Some(candidate);
        }
        // Adjust numer by the byte gap. Picture-bytes = header + numer
        // and header is monotone-non-decreasing in numer, so adjusting
        // by the gap converges quickly.
        let gap = target_picture_bytes as i64 - actual as i64;
        let new_numer = numer as i64 + gap;
        if new_numer <= 0 {
            return None;
        }
        // Clamp to keep monotone progress.
        if new_numer as u32 == numer {
            // Hit a fixed point on either side — accept this candidate
            // if it's within 1 byte of the target.
            if (actual as i64 - target_picture_bytes as i64).abs() <= 1 {
                let bps = numer / n_slices;
                if bps < 2 {
                    return None;
                }
                return Some(candidate);
            }
            return None;
        }
        numer = new_numer as u32;
    }
    // Final accept after iteration cap: emit whatever we converged to.
    let candidate = LdEncoderParams {
        slice_bytes_numer: numer,
        slice_bytes_denom: n_slices,
        ..base.clone()
    };
    let bps = numer / n_slices;
    if bps < 2 {
        return None;
    }
    Some(candidate)
}

/// Encode a single LD intra picture targeting `target_picture_bytes`
/// for the picture **payload** (= what [`encode_ld_intra_picture`]
/// returns; excludes the 13-byte parse-info wrapper).
///
/// The picker:
///   1. Derives `slice_bytes_numer / denom` so the encoded picture
///      payload lands within ±1 byte of `target_picture_bytes` (see
///      [`derive_ld_slice_bytes_for_target`]). The per-slice budget is
///      uniform — `slice_bytes_numer / (slices_x * slices_y)` bytes per
///      slice.
///   2. Calls [`pick_ld_picture_qindex`] to choose the **lowest**
///      `qindex ∈ base.qindex..=127` for which every slice's content
///      fits without Funnel-truncation. Lower qindex = higher quality;
///      a high target picks q=0, a low target may need q=127.
///
/// Returns `Some((bytes, qindex, adjusted_params))` on success, `None`
/// when `target_picture_bytes` is too small for the picture header +
/// `2 * slices_x * slices_y` minimum slice bytes (i.e. the budget can
/// not physically fit the LD picture structure regardless of qindex).
///
/// Source of truth: SMPTE ST 2042-1 §13.5.3.2 (slice byte budget),
/// §13.5.2 (per-slice qindex header), §13.5.4 (quantisation matrix
/// indexing by qindex).
pub fn encode_single_ld_intra_picture_with_size_target(
    sequence: &SequenceHeader,
    base: &LdEncoderParams,
    target_picture_bytes: u32,
    picture_number: u32,
    y: &[u8],
    u: &[u8],
    v: &[u8],
) -> Option<(Vec<u8>, u32, LdEncoderParams)> {
    let mut adjusted = derive_ld_slice_bytes_for_target(base, target_picture_bytes)?;
    let qindex = pick_ld_picture_qindex(sequence, &adjusted, y, u, v);
    adjusted.qindex = qindex;
    let bytes = encode_ld_intra_picture(sequence, &adjusted, picture_number, y, u, v);
    Some((bytes, qindex, adjusted))
}

/// Stream-level analogue of
/// [`encode_single_ld_intra_picture_with_size_target`]: same picker, but
/// returns a full elementary stream (sequence header + 1 LD picture +
/// end-of-sequence). The `target_picture_bytes` argument still refers
/// to the LD picture *payload* (excludes the 13-byte parse-info
/// wrapper); the returned stream's total byte count is therefore
/// `sh_unit_len + 13 + target_picture_bytes(±1) + 13`.
pub fn encode_single_ld_intra_stream_with_size_target(
    sequence: &SequenceHeader,
    base: &LdEncoderParams,
    target_picture_bytes: u32,
    picture_number: u32,
    y: &[u8],
    u: &[u8],
    v: &[u8],
) -> Option<(Vec<u8>, u32, LdEncoderParams)> {
    let (pic_payload, qindex, adjusted) = encode_single_ld_intra_picture_with_size_target(
        sequence,
        base,
        target_picture_bytes,
        picture_number,
        y,
        u,
        v,
    )?;

    let sh_payload = encode_sequence_header(sequence);
    let pi_size = 13usize;
    let sh_unit_len = pi_size + sh_payload.len();
    let pic_unit_len = pi_size + pic_payload.len();

    let mut out = Vec::with_capacity(sh_unit_len + pic_unit_len + pi_size);
    write_parse_info(&mut out, 0x00, sh_unit_len as u32, 0);
    out.extend_from_slice(&sh_payload);
    write_parse_info(&mut out, 0xC8, pic_unit_len as u32, sh_unit_len as u32);
    out.extend_from_slice(&pic_payload);
    write_parse_info(&mut out, 0x10, 0, pic_unit_len as u32);
    Some((out, qindex, adjusted))
}

/// Rate-control strategy for [`encode_ld_sequence_with_size_target`].
///
/// LD picture bytes are deterministic once `slice_bytes_numer` is fixed
/// (every slice writes its full budget; see [`ld_picture_payload_bytes`]),
/// so the only freedom is *how* the per-picture byte budget is derived
/// from the caller's `target_bytes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LdRateControl {
    /// Each picture is independently sized to exactly `target_bytes`
    /// (±1 byte from [`derive_ld_slice_bytes_for_target`]). The stream
    /// total tracks `N * target_bytes` plus the fixed sequence-header /
    /// parse-info overhead — there is no carry-over between pictures.
    PerPicture,
    /// Constant-bit-rate: a running accumulator carries the signed byte
    /// over-/under-shoot of every encoded picture into the next picture's
    /// budget. Because LD picture bytes are deterministic, the actual
    /// over/undershoot is purely the ±1-byte
    /// [`derive_ld_slice_bytes_for_target`] residual plus any clamping at
    /// the minimum picture size — the accumulator drives the running
    /// stream total to within a byte or two of `N * target_bytes`.
    Cbr,
    /// Leaky-bucket variable-bitrate: like [`Cbr`] but the spendable
    /// undershoot is **clamped at `buffer_bytes`** so the per-picture
    /// request never exceeds `target_bytes + buffer_bytes` — an
    /// instantaneous peak cap. Savings above `buffer_bytes` are
    /// forfeited rather than carried forward. This bounds the
    /// worst-case single-picture size (useful for transport / buffer-
    /// occupancy guarantees) while still letting short bursts of
    /// undershoot subsidise a higher-quality neighbour.
    ///
    /// LD's signed accumulator convention is `carry = sum(actual −
    /// target)` (positive = overshot the ideal so far). Because the
    /// spendable savings the next picture may use are `max(−carry, 0)`,
    /// the bucket clamp `carry.max(-buffer_bytes as i64)` after each
    /// encode caps the per-picture request at `target + buffer_bytes`.
    ///
    /// `buffer_bytes == 0` degenerates to [`PerPicture`] (no carry can
    /// ever be spent, so every request equals `target_bytes`); an
    /// effectively infinite `buffer_bytes` degenerates to [`Cbr`].
    /// Intermediate values trade peak-size cap against long-run
    /// average. The LD analogue of
    /// [`HqRateControl::Vbv`] from r146.
    ///
    /// Source of truth: SMPTE ST 2042-1 §13.5.2 / §13.5.3.2 specify the
    /// per-slice qindex and slice-bytes derivation; the leaky-bucket
    /// policy itself is a pure encoder-side rate-shaping choice not
    /// dictated by the bitstream spec (any per-slice qindex / slice-
    /// bytes the encoder produces is spec-conformant).
    ///
    /// [`Cbr`]: LdRateControl::Cbr
    /// [`PerPicture`]: LdRateControl::PerPicture
    /// [`HqRateControl::Vbv`]: HqRateControl::Vbv
    Vbv {
        /// Peak per-picture surplus the leaky bucket may hold above
        /// `target_bytes` — i.e. the maximum extra bytes a single
        /// picture's request can borrow from saved undershoots.
        buffer_bytes: u32,
    },
    /// Drain-rate hysteresis variant of [`Vbv`] — same bucket fill /
    /// forfeit semantics, but the savings *spent* on any one picture's
    /// request are additionally clamped at `max_drain_per_picture`.
    /// The LD analogue of [`HqRateControl::VbvHysteresis`].
    ///
    /// Plain [`Vbv`] lets a single picture drain the entire bucket
    /// (its request can equal `target + buffer_bytes` the moment
    /// savings fill the bucket). That delivers maximum quality on
    /// whichever picture wins the lottery but produces a sudden
    /// cliff — the bucket empties abruptly, then sits at zero until
    /// the next undershoot refills it. `VbvHysteresis` smooths the
    /// drain so the instantaneous request bump is capped at
    /// `max_drain_per_picture` even if the bucket is full; the
    /// remaining savings stay in the bucket for the *next* picture.
    ///
    /// LD's signed accumulator convention is `carry = sum(actual −
    /// target)` (positive = overshot the ideal so far, i.e. *debt*;
    /// negative = saved budget, i.e. *credit*). The spendable savings
    /// the next picture may use are `max(−carry, 0)`. Under
    /// `VbvHysteresis` we additionally clamp that spend at
    /// `max_drain_per_picture`, so the per-picture request becomes
    /// `target + max(0, min(−carry, buffer_bytes, max_drain_per_picture))`
    /// in the savings branch (`carry < 0`), and the debt branch
    /// (`carry > 0`) collapses the request toward `target − carry` —
    /// identical to `Vbv` because debt repayment is mandatory, not
    /// rate-limited.
    ///
    /// Strict generalisation invariants:
    ///   * `max_drain_per_picture == 0` ≡ [`PerPicture`] on streams
    ///     where the LD picker hits its target (no debt branch, no
    ///     spendable savings → byte-identical),
    ///   * `max_drain_per_picture >= buffer_bytes` ≡ [`Vbv`] (the drain
    ///     cap never bites because the bucket cap already does;
    ///     byte-identical).
    ///
    /// Pure encoder-side rate-shaping policy; the bitstream output is
    /// spec-conformant under SMPTE ST 2042-1 §13.5.2 / §13.5.3.2 (any
    /// per-slice qindex / slice-bytes the encoder produces is legal).
    ///
    /// [`Vbv`]: LdRateControl::Vbv
    /// [`PerPicture`]: LdRateControl::PerPicture
    /// [`HqRateControl::VbvHysteresis`]: HqRateControl::VbvHysteresis
    VbvHysteresis {
        /// Peak per-picture surplus the leaky bucket may hold above
        /// `target_bytes`. Same role as [`Vbv::buffer_bytes`].
        ///
        /// [`Vbv::buffer_bytes`]: LdRateControl::Vbv::buffer_bytes
        buffer_bytes: u32,
        /// Maximum savings that may be drained into a single picture's
        /// request. Bounds how aggressively a full bucket is emptied
        /// onto one neighbour — `0` means no savings can be spent
        /// (collapses to [`PerPicture`] when the picker hits target);
        /// values `≥ buffer_bytes` make the drain cap inert and
        /// collapse back to plain [`Vbv`].
        ///
        /// [`PerPicture`]: LdRateControl::PerPicture
        /// [`Vbv`]: LdRateControl::Vbv
        max_drain_per_picture: u32,
    },
}

/// Per-picture rate-control telemetry returned by
/// [`encode_ld_sequence_with_size_target_report`].
#[derive(Debug, Clone, Copy)]
pub struct LdPictureRate {
    /// Picture number written into the picture header.
    pub picture_number: u32,
    /// Byte budget actually requested for this picture (after the CBR
    /// accumulator adjustment, if any).
    pub requested_bytes: u32,
    /// Actual encoded picture *payload* bytes (excludes the 13-byte
    /// parse-info wrapper). Equals [`ld_picture_payload_bytes`] of the
    /// adjusted params.
    pub actual_payload_bytes: u32,
    /// qindex chosen by [`pick_ld_picture_qindex`] for this picture.
    pub qindex: u32,
    /// Running rate-control surplus *after* this picture has been
    /// encoded and the [`LdRateControl::Vbv`] / [`LdRateControl::VbvHysteresis`]
    /// bucket clamp (if any) has been applied. Sign convention:
    /// **positive = savings** (cumulative undershoot — future pictures
    /// may spend it), **negative = debt** (cumulative overshoot —
    /// future pictures must pay it back).
    ///
    /// Computed identically for every rate-control mode as the signed
    /// deviation of the ideal cumulative budget from the encoded
    /// cumulative bytes (`pictures_seen × target_bytes − Σ
    /// actual_payload_bytes`). The modes differ only in whether the
    /// next picture's request *uses* the accumulator
    /// ([`LdRateControl::Cbr`] / [`LdRateControl::Vbv`] /
    /// [`LdRateControl::VbvHysteresis`] do,
    /// [`LdRateControl::PerPicture`] does not) and how much of it any
    /// one picture may *spend* (Vbv: up to `buffer_bytes`;
    /// VbvHysteresis: additionally capped at `max_drain_per_picture`).
    /// Under VBV / VbvHysteresis the bucket clamp guarantees
    /// `running_surplus_bytes ≤ buffer_bytes` once a picture has been
    /// folded in.
    pub running_surplus_bytes: i64,
}

/// Encode a multi-picture VC-2 LD intra-only sequence with per-picture
/// rate control driven by [`pick_ld_picture_qindex`].
///
/// For every input frame the driver:
///   1. derives a per-picture byte budget from `target_bytes` and the
///      chosen [`LdRateControl`] strategy,
///   2. sizes `slice_bytes_numer / denom` so the encoded picture payload
///      lands within ±1 byte of that budget
///      ([`derive_ld_slice_bytes_for_target`]),
///   3. picks the lowest qindex for which every slice fits without
///      Funnel-truncation ([`pick_ld_picture_qindex`]), and
///   4. emits the LD picture with parse code `0xC8` (§D.1.1 — LD permits
///      only non-reference intra pictures).
///
/// The result is a complete elementary stream: sequence header (`0x00`)
/// plus one LD picture per frame plus end-of-sequence (`0x10`), with the
/// `next_parse_offset` / `previous_parse_offset` chain wired up so it
/// round-trips through [`crate::decoder::DiracDecoder`] to one decoded
/// frame per input frame.
///
/// Each frame's `(picture_number, y, u, v)` is taken from the supplied
/// [`InputPicture`] slice. `base` supplies the wavelet / depth / slice
/// grid / quant matrix; its `slice_bytes_numer / denom / qindex` are
/// overridden per-picture by the rate controller.
///
/// Pictures whose adjusted budget is too small to hold the LD picture
/// structure (header + `2 * slices_x * slices_y` minimum slice bytes,
/// per [`derive_ld_slice_bytes_for_target`]) are clamped up to the
/// smallest viable budget rather than dropped — a CBR run can therefore
/// never produce an invalid picture, and the accumulator absorbs the
/// clamp as an overshoot it pays back on the following pictures.
///
/// Source of truth: SMPTE ST 2042-1 §13.5.3.2 (slice byte budget),
/// §13.5.2 (per-slice qindex header), §D.1.1 (LD parse-code restriction),
/// §9.6 / §10.4 (parse-info sequence framing).
pub fn encode_ld_sequence_with_size_target(
    sequence: &SequenceHeader,
    base: &LdEncoderParams,
    frames: &[InputPicture<'_>],
    target_bytes: u32,
    mode: LdRateControl,
) -> Vec<u8> {
    let (stream, _report) =
        encode_ld_sequence_with_size_target_report(sequence, base, frames, target_bytes, mode);
    stream
}

/// [`encode_ld_sequence_with_size_target`] plus per-picture telemetry
/// (requested vs. actual bytes and the chosen qindex for each picture).
/// Returns `(stream, report)`.
pub fn encode_ld_sequence_with_size_target_report(
    sequence: &SequenceHeader,
    base: &LdEncoderParams,
    frames: &[InputPicture<'_>],
    target_bytes: u32,
    mode: LdRateControl,
) -> (Vec<u8>, Vec<LdPictureRate>) {
    let sh_payload = encode_sequence_header(sequence);
    let pi_size = 13usize;
    let sh_unit_len = pi_size + sh_payload.len();

    let mut out = Vec::new();
    write_parse_info(&mut out, 0x00, sh_unit_len as u32, 0);
    out.extend_from_slice(&sh_payload);

    let mut report: Vec<LdPictureRate> = Vec::with_capacity(frames.len());
    let mut prev_len = sh_unit_len as u32;
    let mut last_pic_len: Option<u32> = None;

    // CBR accumulator: signed running difference between the bytes we
    // have *requested* so far and `pictures_seen * target_bytes`. A
    // positive accumulator means we have overshot the ideal cumulative
    // budget, so the next picture's request is reduced; negative means
    // we have headroom to spend.
    let mut carry: i64 = 0;

    // Smallest viable per-picture budget for this slice grid (header +
    // 2 bytes per slice). Used to clamp tiny CBR requests rather than
    // dropping the picture.
    let n_slices = (base.slices_x.max(1) * base.slices_y.max(1)) as i64;
    let header_floor = {
        let probe = LdEncoderParams {
            slice_bytes_numer: 2 * n_slices as u32,
            slice_bytes_denom: n_slices as u32,
            ..base.clone()
        };
        ld_picture_payload_bytes(&probe) as i64
    };
    let min_budget = header_floor.max(2 * n_slices);

    for pic in frames {
        let requested: u32 = match mode {
            LdRateControl::PerPicture => target_bytes,
            LdRateControl::Cbr => {
                // Spend `target_bytes` minus whatever we've overshot so
                // far (carry > 0 ⇒ pull back; carry < 0 ⇒ spend extra).
                let want = target_bytes as i64 - carry;
                want.clamp(min_budget, u32::MAX as i64) as u32
            }
            LdRateControl::Vbv { buffer_bytes } => {
                // Leaky-bucket: identical to `Cbr` but the spendable
                // savings (i.e. `max(-carry, 0)`) are capped at
                // `buffer_bytes`. The post-encode update clamps
                // `carry >= -buffer_bytes`, so the request `target -
                // carry` is bounded above by `target + buffer_bytes` —
                // an instantaneous peak-size cap. `buffer_bytes == 0`
                // forces `spendable == 0` so the request collapses to
                // `target_bytes` (modulo the carry-debt branch when
                // `carry > 0`, which still tightens the request just
                // like `Cbr`). The explicit `min` on `spendable` here
                // is belt-and-braces against floating bucket-cap edge
                // cases; the post-encode clamp is the load-bearing
                // invariant.
                let spendable = (-carry).min(buffer_bytes as i64).max(0);
                let want = target_bytes as i64 - carry.max(0) + spendable;
                want.clamp(min_budget, u32::MAX as i64) as u32
            }
            LdRateControl::VbvHysteresis {
                buffer_bytes,
                max_drain_per_picture,
            } => {
                // Drain-rate hysteresis: identical bucket fill / forfeit
                // semantics as `Vbv`, but the spendable savings are
                // additionally clamped at `max_drain_per_picture`. The
                // debt-payback branch (`carry > 0`) is unchanged from
                // `Vbv` because debt repayment is mandatory, not
                // rate-limited — only the *spend* side of the bucket
                // is hysteretic. `max_drain_per_picture == 0` zeros
                // `spendable`, so the request collapses to
                // `target - max(carry, 0)` — same as the carry-debt
                // branch of every other VBV variant, and identical to
                // `PerPicture` whenever the LD picker hits target
                // (the picker's residual ±1-byte deviation means
                // `carry` stays at zero on smooth fixtures, so
                // `target - carry.max(0) == target`).
                let spendable = (-carry)
                    .min(buffer_bytes as i64)
                    .min(max_drain_per_picture as i64)
                    .max(0);
                let want = target_bytes as i64 - carry.max(0) + spendable;
                want.clamp(min_budget, u32::MAX as i64) as u32
            }
        };

        // Size + qindex-pick + encode this picture. On the rare clamp
        // where even `requested` can't fit, step the budget up until
        // `derive_ld_slice_bytes_for_target` accepts it.
        let mut budget = requested.max(min_budget as u32);
        let (pic_payload, qindex, adjusted) = loop {
            match encode_single_ld_intra_picture_with_size_target(
                sequence,
                base,
                budget,
                pic.picture_number,
                pic.y,
                pic.u,
                pic.v,
            ) {
                Some(triple) => break triple,
                None => {
                    // Grow the budget toward viability. The header is
                    // tiny so a single +n_slices step almost always
                    // suffices; loop guards against pathological grids.
                    budget = budget.saturating_add(n_slices.max(1) as u32 + 2);
                }
            }
        };

        let actual_payload = ld_picture_payload_bytes(&adjusted) as i64;
        // CBR feedback: the accumulator tracks deviation of the actual
        // requested-then-encoded payload from the ideal `target_bytes`.
        carry += actual_payload - target_bytes as i64;
        // VBV: clamp the savings end of the bucket at -buffer_bytes so
        // the next picture's `target - carry` request is ≤ target +
        // buffer_bytes. Overshoot debt (carry > 0) is left untouched —
        // a peak-size cap only governs the upper edge of the request,
        // not the lower edge — and PerPicture / Cbr leave `carry`
        // alone (PerPicture ignores `carry` in its request anyway).
        match mode {
            LdRateControl::Vbv { buffer_bytes }
            | LdRateControl::VbvHysteresis { buffer_bytes, .. } => {
                let floor = -(buffer_bytes as i64);
                if carry < floor {
                    carry = floor;
                }
            }
            LdRateControl::PerPicture | LdRateControl::Cbr => {}
        }

        // Telemetry: report the running surplus *after* the VBV clamp
        // applied above, in the "positive = savings, negative = debt"
        // convention. The LD accumulator's internal sign is the
        // mirror — `carry > 0` is overshoot debt and `carry < 0` is
        // savings — so we negate it on the way out.
        let running_surplus_bytes = -carry;

        report.push(LdPictureRate {
            picture_number: pic.picture_number,
            requested_bytes: budget,
            actual_payload_bytes: actual_payload as u32,
            qindex,
            running_surplus_bytes,
        });

        let pic_unit_len = (pi_size + pic_payload.len()) as u32;
        write_parse_info(&mut out, 0xC8, pic_unit_len, prev_len);
        out.extend_from_slice(&pic_payload);
        prev_len = pic_unit_len;
        last_pic_len = Some(pic_unit_len);
    }

    write_parse_info(
        &mut out,
        0x10,
        0,
        last_pic_len.unwrap_or(sh_unit_len as u32),
    );
    (out, report)
}

fn write_ld_transform_parameters(w: &mut BitWriter, params: &LdEncoderParams) {
    let w_idx = wavelet_index(params.wavelet);
    w.write_uint(w_idx);
    w.write_uint(params.dwt_depth);
    // §12.4.4 `extended_transform_parameters()` — v3 only. See
    // `write_transform_parameters` for the HQ-side rationale; the LD
    // path emits the same two flag bits (and gated values when the
    // override is set) in the same position relative to `dwt_depth`.
    if params.major_version >= 3 {
        match params.extended_transform_override {
            None => {
                w.write_bool(false); // asym_transform_index_flag
                w.write_bool(false); // asym_transform_flag
            }
            Some(ext) => {
                let asym_idx = ext.wavelet_index_ho != w_idx;
                w.write_bool(asym_idx);
                if asym_idx {
                    w.write_uint(ext.wavelet_index_ho);
                }
                let asym_depth = ext.dwt_depth_ho != 0;
                w.write_bool(asym_depth);
                if asym_depth {
                    w.write_uint(ext.dwt_depth_ho);
                }
            }
        }
    }
    // §12.4.5.2 slice_parameters — LD branch.
    w.write_uint(params.slices_x);
    w.write_uint(params.slices_y);
    w.write_uint(params.slice_bytes_numer);
    w.write_uint(params.slice_bytes_denom);
    // §12.4.5.3 quant_matrix: default (flag=0) or explicit (flag=1).
    // A custom matrix must be shaped for the emitted `dwt_depth_ho`.
    debug_assert!(
        !params.custom_quant_matrix || params.quant_matrix.dwt_depth_ho == params.dwt_depth_ho(),
        "custom quant matrix dwt_depth_ho ({}) must match the emitted §12.4.4.3 dwt_depth_ho ({})",
        params.quant_matrix.dwt_depth_ho,
        params.dwt_depth_ho(),
    );
    write_quant_matrix(w, params.custom_quant_matrix, &params.quant_matrix);
}

fn forward_component_ld(
    plane: &[u8],
    comp_w: u32,
    comp_h: u32,
    depth: u32,
    params: &LdEncoderParams,
) -> Vec<[SubbandData; 4]> {
    let ho = params.dwt_depth_ho();
    let (pw, ph) = padded_component_dims_ho(comp_w, comp_h, params.dwt_depth, ho);
    let half: i32 = 1i32 << (depth - 1);
    let mut pic = SubbandData::new(pw, ph);
    for y in 0..ph {
        for x in 0..pw {
            let src_x = x.min(comp_w as usize - 1);
            let src_y = y.min(comp_h as usize - 1);
            let v = plane[src_y * comp_w as usize + src_x] as i32 - half;
            pic.set(y, x, v);
        }
    }
    if ho == 0 {
        dwt(&pic, params.wavelet, params.dwt_depth)
    } else {
        dwt_with_ho(
            &pic,
            params.wavelet,
            params.wavelet_ho(),
            params.dwt_depth,
            ho,
        )
    }
}

fn quantise_pyramid_ld(
    pyramid: &[[SubbandData; 4]],
    q_per_level: &[[u32; 4]],
) -> Vec<[SubbandData; 4]> {
    let mut out: Vec<[SubbandData; 4]> = Vec::with_capacity(pyramid.len());
    for (level, bands) in pyramid.iter().enumerate() {
        let mut level_out: [SubbandData; 4] = [
            SubbandData::new(bands[0].width, bands[0].height),
            SubbandData::new(bands[1].width, bands[1].height),
            SubbandData::new(bands[2].width, bands[2].height),
            SubbandData::new(bands[3].width, bands[3].height),
        ];
        for orient_idx in 0..4 {
            let src = &bands[orient_idx];
            if src.width == 0 || src.height == 0 {
                continue;
            }
            let q = q_per_level[level][orient_idx];
            let dst = &mut level_out[orient_idx];
            for y in 0..src.height {
                for x in 0..src.width {
                    let v = src.get(y, x);
                    dst.set(y, x, quantise_coeff(v, q));
                }
            }
        }
        out.push(level_out);
    }
    out
}

/// Slice-area bounds within a subband — same helper as the HQ path.
fn ld_slice_bounds(
    sx: u32,
    sy: u32,
    slices_x: u32,
    slices_y: u32,
    sub_w: usize,
    sub_h: usize,
) -> (usize, usize, usize, usize) {
    slice_bounds(sx, sy, slices_x, slices_y, sub_w, sub_h)
}

/// Encode one LD slice (§13.5.3.1).
#[allow(clippy::too_many_arguments)]
fn encode_ld_slice(
    w: &mut BitWriter,
    params: &LdEncoderParams,
    sx: u32,
    sy: u32,
    y_qpy: &[[SubbandData; 4]],
    u_qpy: &[[SubbandData; 4]],
    v_qpy: &[[SubbandData; 4]],
    luma_dims: &[(usize, usize)],
    chroma_dims: &[(usize, usize)],
) {
    let slice_n_bytes = slice_bytes(
        params.slices_x,
        params.slice_bytes_numer,
        params.slice_bytes_denom,
        sx,
        sy,
    );
    assert!(
        slice_n_bytes > 0,
        "LD slice budget must be > 0 (sx={sx}, sy={sy})",
    );
    let total_bits = 8u64 * slice_n_bytes as u64;
    // §13.5.2 specifies `length_bits = intlog2(8*slice_bytes - 7)` with the
    // Dirac `intlog2` convention `2^(m-1) < n ≤ 2^m` (§6.4.3). In practice
    // ffmpeg's VC-2 decoder — and the SMPTE ST 2042-1 reference — sizes
    // this field as `floor(log2(total_bits)) + 1` (the minimum bits
    // needed to represent `0..total_bits-1`), which agrees with the
    // Dirac formula except when `total_bits` is an exact power of 2
    // (e.g. slice_bytes = 32 / 64 / 128 / 256), where Dirac's formula
    // under-sizes the field by one bit and shifts every subsequent
    // coefficient by that same bit — the ~12 dB interop gap from Round 8.
    let length_bits = ld_length_bits(slice_n_bytes);
    let header_bits = 7u64 + length_bits as u64;
    assert!(
        total_bits > header_bits,
        "slice too small to hold header bits",
    );
    let payload_bits = total_bits - header_bits;

    // Serialise the luma block first into a temp BitWriter (tightly
    // packed — we'll right-pad with 1s to hit the chosen `slice_y_length`).
    let mut luma_tmp = BitWriter::new();
    write_ld_component(&mut luma_tmp, params, y_qpy, luma_dims, sx, sy, true);
    let luma_bits = luma_tmp.measured_bits();

    // Serialise the chroma interleaved stream.
    let mut chroma_tmp = BitWriter::new();
    write_ld_chroma_interleaved(&mut chroma_tmp, params, u_qpy, v_qpy, chroma_dims, sx, sy);
    let chroma_bits = chroma_tmp.measured_bits();

    // Choose slice_y_length — ideally `luma_bits`, but capped at
    // (payload_bits - chroma_bits) so chroma fits as well. If the raw
    // total exceeds the budget we let the luma portion take what it
    // can (the remainder is truncated / 1-padded on the decode side).
    // `saturating_sub` on the else-branch handles the pathological
    // case where chroma alone is oversized — luma then gets 0 bits
    // and chroma is truncated.
    let slice_y_length = if luma_bits + chroma_bits <= payload_bits {
        luma_bits
    } else {
        payload_bits.saturating_sub(chroma_bits)
    };
    let chroma_allot = payload_bits - slice_y_length;

    assert!(
        slice_y_length < (1u64 << length_bits),
        "slice_y_length {slice_y_length} overflows {length_bits} bits — pick a larger slice budget",
    );

    // Header: 7-bit qindex then `length_bits` bits of slice_y_length.
    w.write_nbits(7, params.qindex);
    w.write_nbits(length_bits, slice_y_length as u32);

    // Luma: write serialised bytes up to slice_y_length bits, padded
    // with 1s to exactly slice_y_length.
    write_funnel_bounded(w, luma_tmp, slice_y_length);

    // Chroma: same with `chroma_allot` bits.
    write_funnel_bounded(w, chroma_tmp, chroma_allot);
}

/// Bounded write: consume `src`, append its exact bit payload to `w`
/// and right-pad to `bits_budget` bits using 1-bits. A Funnel read on
/// the decoder side returns 1 past end, which aborts any in-progress
/// interleaved exp-Golomb uint read at value 0 — so 1-padding is
/// harmless even if it falls inside a partially-consumed coefficient.
///
/// If `src` is already longer than `bits_budget`, the tail is
/// truncated. This only ever happens when the caller guarantees
/// it's safe (lossless round-trip tests pre-size the slice budget).
fn write_funnel_bounded(w: &mut BitWriter, src: BitWriter, bits_budget: u64) {
    let src_bits = src.measured_bits();
    let bytes = src.finish();
    // Write up to min(src_bits, bits_budget) bits from `bytes`.
    let to_write = src_bits.min(bits_budget);
    let full_bytes = (to_write / 8) as usize;
    for byte in bytes.iter().take(full_bytes) {
        w.write_nbits(8, *byte as u32);
    }
    let mut bits_written = (full_bytes as u64) * 8;
    let leftover = to_write - bits_written;
    if leftover > 0 {
        let last_byte = bytes[full_bytes];
        // MSB-first packing — leftover bits sit in the high end of
        // `last_byte`. `finish()` right-padded the rest with zeros.
        for i in (8 - leftover as u32..8).rev() {
            w.write_bit(((last_byte >> i) as u32) & 1);
        }
        bits_written += leftover;
    }
    // Pad up to bits_budget with 1s.
    while bits_written < bits_budget {
        w.write_bit(1);
        bits_written += 1;
    }
}

fn write_ld_component(
    w: &mut BitWriter,
    params: &LdEncoderParams,
    qpy: &[[SubbandData; 4]],
    dims: &[(usize, usize)],
    sx: u32,
    sy: u32,
    _is_luma: bool,
) {
    // §13.5.3 band order — shared symmetric/asymmetric sequence.
    for (level, orient) in slice_band_order(params.dwt_depth, params.dwt_depth_ho()) {
        write_ld_slice_band(w, level, orient, sx, sy, params, qpy, dims);
    }
}

#[allow(clippy::too_many_arguments)]
fn write_ld_chroma_interleaved(
    w: &mut BitWriter,
    params: &LdEncoderParams,
    u_qpy: &[[SubbandData; 4]],
    v_qpy: &[[SubbandData; 4]],
    chroma_dims: &[(usize, usize)],
    sx: u32,
    sy: u32,
) {
    // §13.5.3 band order — shared symmetric/asymmetric sequence.
    for (level, orient) in slice_band_order(params.dwt_depth, params.dwt_depth_ho()) {
        write_ld_slice_chroma_pair(w, level, orient, sx, sy, params, u_qpy, v_qpy, chroma_dims);
    }
}

#[allow(clippy::too_many_arguments)]
fn write_ld_slice_band(
    w: &mut BitWriter,
    level: u32,
    orient: Orient,
    sx: u32,
    sy: u32,
    params: &LdEncoderParams,
    qpy: &[[SubbandData; 4]],
    dims: &[(usize, usize)],
) {
    let (sub_w, sub_h) = dims[level as usize];
    let (left, right, top, bottom) =
        ld_slice_bounds(sx, sy, params.slices_x, params.slices_y, sub_w, sub_h);
    let band = &qpy[level as usize][orient.as_index()];
    if band.width == 0 || band.height == 0 {
        return;
    }
    for y in top..bottom {
        for x in left..right {
            let v = band.get(y, x);
            w.write_sint(v);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn write_ld_slice_chroma_pair(
    w: &mut BitWriter,
    level: u32,
    orient: Orient,
    sx: u32,
    sy: u32,
    params: &LdEncoderParams,
    u_qpy: &[[SubbandData; 4]],
    v_qpy: &[[SubbandData; 4]],
    chroma_dims: &[(usize, usize)],
) {
    let (sub_w, sub_h) = chroma_dims[level as usize];
    let (left, right, top, bottom) =
        ld_slice_bounds(sx, sy, params.slices_x, params.slices_y, sub_w, sub_h);
    let u_band = &u_qpy[level as usize][orient.as_index()];
    let v_band = &v_qpy[level as usize][orient.as_index()];
    if u_band.width == 0 || u_band.height == 0 {
        return;
    }
    for y in top..bottom {
        for x in left..right {
            w.write_sint(u_band.get(y, x));
            w.write_sint(v_band.get(y, x));
        }
    }
}

/// Encode one 8-bit 4:2:0 YUV frame as a VC-2 LD intra-only elementary
/// stream.
///
/// LD profile (§D.1.1) uses parse code `0xC8` (Intra Non-Reference
/// Picture) exclusively — `0xCC` from Table 9.1 is *syntactically*
/// permitted in the low-delay family, but §D.1.1 restricts compliant
/// LD sequences to non-reference intra pictures. Callers needing a
/// multi-picture sequence should use
/// [`encode_ld_intra_multi_stream`].
pub fn encode_single_ld_intra_stream(
    sequence: &SequenceHeader,
    params: &LdEncoderParams,
    picture_number: u32,
    y: &[u8],
    u: &[u8],
    v: &[u8],
) -> Vec<u8> {
    let sh_payload = encode_sequence_header(sequence);
    let pic_payload = encode_ld_intra_picture(sequence, params, picture_number, y, u, v);

    let pi_size = 13usize;
    let sh_unit_len = pi_size + sh_payload.len();
    let pic_unit_len = pi_size + pic_payload.len();
    let eos_unit_len = pi_size;

    let mut out = Vec::with_capacity(sh_unit_len + pic_unit_len + eos_unit_len);
    write_parse_info(&mut out, 0x00, sh_unit_len as u32, 0);
    out.extend_from_slice(&sh_payload);
    // 0xC8 — LD non-reference intra picture.
    write_parse_info(&mut out, 0xC8, pic_unit_len as u32, sh_unit_len as u32);
    out.extend_from_slice(&pic_payload);
    write_parse_info(&mut out, 0x10, 0, pic_unit_len as u32);
    out
}

/// One input picture: Y/U/V planes + picture number.
#[derive(Debug, Clone)]
pub struct InputPicture<'a> {
    pub picture_number: u32,
    pub y: &'a [u8],
    pub u: &'a [u8],
    pub v: &'a [u8],
}

/// Encode a multi-picture VC-2 LD intra-only elementary stream. Every
/// picture uses parse code `0xC8` (Intra Non-Reference) per §D.1.1 —
/// LD profile does not permit reference pictures.
pub fn encode_ld_intra_multi_stream(
    sequence: &SequenceHeader,
    params: &LdEncoderParams,
    pictures: &[InputPicture<'_>],
) -> Vec<u8> {
    let sh_payload = encode_sequence_header(sequence);
    let pi_size = 13usize;
    let sh_unit_len = pi_size + sh_payload.len();

    let mut out = Vec::new();
    write_parse_info(&mut out, 0x00, sh_unit_len as u32, 0);
    out.extend_from_slice(&sh_payload);

    let mut prev_len = sh_unit_len as u32;
    let mut last_pic_len: Option<u32> = None;
    for pic in pictures {
        let pic_payload =
            encode_ld_intra_picture(sequence, params, pic.picture_number, pic.y, pic.u, pic.v);
        let pic_unit_len = (pi_size + pic_payload.len()) as u32;
        write_parse_info(&mut out, 0xC8, pic_unit_len, prev_len);
        out.extend_from_slice(&pic_payload);
        prev_len = pic_unit_len;
        last_pic_len = Some(pic_unit_len);
    }
    // End of sequence — `prev` is the last picture's unit length (or
    // the seq-header if there were no pictures).
    write_parse_info(
        &mut out,
        0x10,
        0,
        last_pic_len.unwrap_or(sh_unit_len as u32),
    );
    out
}

/// Encode a multi-picture VC-2 HQ intra-only elementary stream,
/// alternating reference (`0xEC`) and non-reference (`0xE8`) parse
/// codes. Every second picture is emitted as a reference picture —
/// that covers the "reference-intra alternation" niche without
/// introducing any inter-picture dependencies.
pub fn encode_hq_intra_multi_stream(
    sequence: &SequenceHeader,
    params: &EncoderParams,
    pictures: &[InputPicture<'_>],
) -> Vec<u8> {
    let sh_payload = encode_sequence_header(sequence);
    let pi_size = 13usize;
    let sh_unit_len = pi_size + sh_payload.len();

    let mut out = Vec::new();
    write_parse_info(&mut out, 0x00, sh_unit_len as u32, 0);
    out.extend_from_slice(&sh_payload);

    let mut prev_len = sh_unit_len as u32;
    let mut last_pic_len: Option<u32> = None;
    for (i, pic) in pictures.iter().enumerate() {
        let pic_payload =
            encode_hq_intra_picture(sequence, params, pic.picture_number, pic.y, pic.u, pic.v);
        let pic_unit_len = (pi_size + pic_payload.len()) as u32;
        // Alternate: even index → non-ref (0xE8), odd → ref (0xEC).
        let parse_code = if i % 2 == 0 { 0xE8 } else { 0xEC };
        write_parse_info(&mut out, parse_code, pic_unit_len, prev_len);
        out.extend_from_slice(&pic_payload);
        prev_len = pic_unit_len;
        last_pic_len = Some(pic_unit_len);
    }
    write_parse_info(
        &mut out,
        0x10,
        0,
        last_pic_len.unwrap_or(sh_unit_len as u32),
    );
    out
}

/// Rate-control strategy for [`encode_hq_sequence_with_size_target`].
///
/// Unlike LD — where the picture-byte count is a deterministic function
/// of `slice_bytes_numer/denom` and constant across qindex — the HQ
/// profile lets each slice's length byte track its actual coefficient
/// block size, so picture bytes shrink monotonically with rising qindex
/// (the dead-zone forward quantiser drives more coefficients toward
/// zero → fewer interleaved exp-Golomb bits per slice). The picker
/// ([`pick_hq_picture_qindex`]) walks `qindex ∈ floor..=127` and stops
/// at the first one whose constant-qindex picture bytes ≤ target — so
/// it **never overshoots** the requested budget but may undershoot
/// when even q=0 already fits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HqRateControl {
    /// Each picture is independently sized to `target_bytes`. No
    /// carry-over between pictures: the per-picture residual (target
    /// minus actual bytes, always ≥ 0 because the picker never
    /// overshoots) is lost.
    PerPicture,
    /// Constant-bit-rate: a running signed accumulator carries each
    /// picture's residual (target − actual; positive when the picture
    /// undershot, signalling spare budget) into the next picture's
    /// request. The next request becomes `target_bytes + carry` — extra
    /// headroom when previous pictures undershot, no debt to pay back
    /// (the picker never overshoots a request, so `carry` is monotone
    /// non-negative across the stream). Drives the running stream
    /// total upward toward `N * target_bytes` as long as the budget is
    /// reachable; once a picture's budget exceeds the q=0 ceiling for
    /// that picture, the carry continues to grow without effect (no
    /// further savings to spend).
    Cbr,
    /// Leaky-bucket variable-bitrate: like [`Cbr`] but the carry is
    /// **clamped at `buffer_bytes`** so the per-picture request never
    /// exceeds `target_bytes + buffer_bytes` — an instantaneous peak
    /// cap. Savings above `buffer_bytes` are forfeited rather than
    /// accumulated. This bounds the worst-case single-picture size
    /// (useful for transport / buffer-occupancy guarantees) while still
    /// letting short bursts of undershoot fund a higher-quality
    /// neighbour.
    ///
    /// Per-picture request: `min(target + carry, target + buffer_bytes)`.
    /// After encoding, `carry = min(carry + target - actual,
    /// buffer_bytes)` (still non-negative thanks to the picker's
    /// no-overshoot guarantee, modulo the q=127 edge case).
    ///
    /// `buffer_bytes == 0` degenerates to [`PerPicture`]; an effectively
    /// infinite `buffer_bytes` degenerates to [`Cbr`]. Intermediate
    /// values trade peak-size cap against long-run average.
    ///
    /// Source of truth: BBC Dirac Specification v2.2.3 §13.5.4 specifies
    /// the slice-qindex range; the leaky-bucket policy itself is a pure
    /// encoder-side rate-shaping choice not dictated by the bitstream
    /// spec (any qindex-per-picture sequence the encoder produces is
    /// spec-conformant).
    ///
    /// [`Cbr`]: HqRateControl::Cbr
    /// [`PerPicture`]: HqRateControl::PerPicture
    Vbv {
        /// Peak per-picture surplus the leaky bucket may hold above
        /// `target_bytes` — i.e. the maximum extra bytes a single
        /// picture's request can borrow from saved undershoots.
        buffer_bytes: u32,
    },
    /// Drain-rate hysteresis variant of [`Vbv`] — same bucket fill /
    /// forfeit semantics, but the savings *spent* on any one picture
    /// are additionally clamped at `max_drain_per_picture`. The bucket
    /// can still hold up to `buffer_bytes` of savings; what changes is
    /// how *fast* those savings flow out to a single picture's request.
    ///
    /// Plain [`Vbv`] permits a single picture to drain the entire
    /// bucket in one step (its request may equal `target + buffer_bytes`
    /// the moment savings fill the bucket). That delivers maximum
    /// quality on whichever picture wins the lottery but exposes the
    /// next several pictures to a sudden cliff — the bucket empties
    /// abruptly, then the carry stays at zero until the next undershoot
    /// refills it. `VbvHysteresis` smooths the drain by spreading
    /// savings across multiple pictures: each picture's request is
    /// `min(target + carry, target + buffer_bytes, target +
    /// max_drain_per_picture)`, so even with a full bucket the
    /// instantaneous bump is capped at `max_drain_per_picture` and the
    /// remaining savings stay in the bucket for the *next* picture.
    ///
    /// Per-picture request: `min(target + carry, target + buffer_bytes,
    /// target + max_drain_per_picture)`. The post-encode carry update is
    /// identical to [`Vbv`] — savings forfeit at `buffer_bytes`, no
    /// drain limit on the *fill* side (only on the *spend* side).
    ///
    /// Strict generalisation invariants:
    ///   * `max_drain_per_picture == 0` ≡ [`PerPicture`] (no savings can
    ///     ever be spent, byte-identical stream),
    ///   * `max_drain_per_picture >= buffer_bytes` ≡ [`Vbv {
    ///     buffer_bytes }`] (the drain cap never bites because the
    ///     bucket cap already does; byte-identical stream),
    ///   * `max_drain_per_picture == buffer_bytes == 0` ≡ [`PerPicture`]
    ///     by both routes.
    ///
    /// Pure encoder-side rate-shaping policy; the bitstream output is
    /// spec-conformant under BBC Dirac Specification v2.2.3 §13.5.4
    /// (any per-picture qindex the encoder produces is legal).
    ///
    /// [`Vbv`]: HqRateControl::Vbv
    /// [`PerPicture`]: HqRateControl::PerPicture
    VbvHysteresis {
        /// Peak per-picture surplus the leaky bucket may hold above
        /// `target_bytes`. Same role as [`Vbv::buffer_bytes`].
        ///
        /// [`Vbv::buffer_bytes`]: HqRateControl::Vbv::buffer_bytes
        buffer_bytes: u32,
        /// Maximum savings that may be drained into a single picture's
        /// request. Bounds how aggressively a full bucket is emptied
        /// onto one neighbour — `0` means no savings can be spent
        /// (collapses to [`PerPicture`]); values `≥ buffer_bytes` make
        /// the drain cap inert and collapse back to plain [`Vbv`].
        ///
        /// [`PerPicture`]: HqRateControl::PerPicture
        /// [`Vbv`]: HqRateControl::Vbv
        max_drain_per_picture: u32,
    },
}

/// Per-picture rate-control telemetry returned by
/// [`encode_hq_sequence_with_size_target_report`].
#[derive(Debug, Clone, Copy)]
pub struct HqPictureRate {
    /// Picture number written into the picture header.
    pub picture_number: u32,
    /// Byte budget actually requested for this picture (after the CBR
    /// carry-over, if any).
    pub requested_bytes: u32,
    /// Actual encoded HQ picture *payload* bytes (excludes the 13-byte
    /// parse-info wrapper). Equals
    /// [`hq_picture_payload_bytes_at_qindex`] at the chosen qindex.
    pub actual_payload_bytes: u32,
    /// qindex chosen by [`pick_hq_picture_qindex`] for this picture
    /// (`0..=127`). The picker promises `actual_payload_bytes ≤
    /// requested_bytes` unless `qindex == 127` (in which case the
    /// q=127 floor still overshoots; graceful degradation).
    pub qindex: u32,
    /// Parse code written for this picture: `0xE8` (HQ non-reference
    /// intra) on even indices, `0xEC` (HQ reference intra) on odd
    /// indices. Matches [`encode_hq_intra_multi_stream`]'s alternation.
    pub parse_code: u8,
    /// Running rate-control surplus *after* this picture has been
    /// encoded and the [`HqRateControl::Vbv`] / [`HqRateControl::VbvHysteresis`]
    /// bucket clamp (if any) has been applied. Sign convention:
    /// **positive = savings** (cumulative undershoot — future pictures
    /// may spend it), **negative = debt** (cumulative overshoot —
    /// future pictures must pay it back).
    ///
    /// Computed identically for every rate-control mode as the signed
    /// deviation of the ideal cumulative budget from the encoded
    /// cumulative bytes (`pictures_seen × target_bytes − Σ
    /// actual_payload_bytes`). The modes differ only in whether the
    /// next picture's request *uses* the accumulator
    /// ([`HqRateControl::Cbr`] / [`HqRateControl::Vbv`] /
    /// [`HqRateControl::VbvHysteresis`] do,
    /// [`HqRateControl::PerPicture`] does not) and how much of it any
    /// one picture may *spend* (Vbv: up to `buffer_bytes`;
    /// VbvHysteresis: additionally capped at `max_drain_per_picture`).
    /// Because the HQ picker never overshoots unless `qindex == 127`,
    /// this value is monotone non-decreasing in the usual case; the
    /// q=127 floor edge case is the only way the per-picture
    /// contribution can be negative.
    /// Under VBV / VbvHysteresis the bucket clamp additionally guarantees
    /// `running_surplus_bytes ≤ buffer_bytes` once a picture has been
    /// folded in.
    pub running_surplus_bytes: i64,
}

/// Encode a multi-picture VC-2 HQ intra-only sequence with per-picture
/// rate control driven by [`pick_hq_picture_qindex`].
///
/// For every input frame the driver:
///   1. derives a per-picture byte budget from `target_bytes` and the
///      chosen [`HqRateControl`] strategy,
///   2. picks the smallest picture-level qindex whose constant-qindex
///      HQ picture payload ≤ budget ([`pick_hq_picture_qindex`]; HQ
///      picture bytes are monotone non-increasing in qindex), and
///   3. emits the HQ picture with parse code `0xE8` (non-reference)
///      on even indices and `0xEC` (reference) on odd indices —
///      same alternation as [`encode_hq_intra_multi_stream`].
///
/// The result is a complete elementary stream: sequence header (`0x00`)
/// plus one HQ picture per frame plus end-of-sequence (`0x10`), with
/// the `next_parse_offset` / `previous_parse_offset` chain wired so it
/// round-trips through [`crate::decoder::DiracDecoder`] to one decoded
/// frame per input frame.
///
/// The picker **never overshoots** the requested budget on the budget
/// it was given (the §13.5.4 length-byte cap is the only edge case,
/// guarded by `debug_assert!` in [`encode_hq_slice`]); if even q=127
/// cannot fit the budget the picker returns 127 and the actual bytes
/// land at the q=127 floor for that picture's content — a graceful
/// degradation rather than an encode failure.
///
/// `base.slice_size_target` is intentionally cleared before invoking
/// the picker so the picture-level qindex is the one actually written
/// into every slice header — matching
/// [`encode_single_hq_intra_stream_with_size_target`].
///
/// Source of truth: BBC Dirac Specification v2.2.3 §13.5.2 (per-slice
/// qindex header), §13.5.4 (`slice_quantisers(qindex)`), §9.6 / §10.4
/// (parse-info sequence framing).
pub fn encode_hq_sequence_with_size_target(
    sequence: &SequenceHeader,
    base: &EncoderParams,
    frames: &[InputPicture<'_>],
    target_bytes: u32,
    mode: HqRateControl,
) -> Vec<u8> {
    let (stream, _report) =
        encode_hq_sequence_with_size_target_report(sequence, base, frames, target_bytes, mode);
    stream
}

/// [`encode_hq_sequence_with_size_target`] plus per-picture telemetry
/// (requested vs. actual bytes, chosen qindex, parse code).
/// Returns `(stream, report)`.
pub fn encode_hq_sequence_with_size_target_report(
    sequence: &SequenceHeader,
    base: &EncoderParams,
    frames: &[InputPicture<'_>],
    target_bytes: u32,
    mode: HqRateControl,
) -> (Vec<u8>, Vec<HqPictureRate>) {
    // Mirror the per-stream wrapper: clear the per-slice §13.5.4 search
    // so the picture-level qindex is the one written into every slice
    // header.
    let mut params = base.clone();
    params.slice_size_target = None;

    let sh_payload = encode_sequence_header(sequence);
    let pi_size = 13usize;
    let sh_unit_len = pi_size + sh_payload.len();

    let mut out = Vec::new();
    write_parse_info(&mut out, 0x00, sh_unit_len as u32, 0);
    out.extend_from_slice(&sh_payload);

    let mut report: Vec<HqPictureRate> = Vec::with_capacity(frames.len());
    let mut prev_len = sh_unit_len as u32;
    let mut last_pic_len: Option<u32> = None;

    // CBR carry: signed running surplus from previous pictures'
    // undershoots. The HQ picker never overshoots, so `carry` is
    // monotone non-negative — every CBR request equals `target_bytes +
    // carry`, growing the budget the picture is allowed to use until
    // either q=0 fits (and the picture undershoots further) or the
    // chosen qindex reaches 0 (further headroom is wasted).
    let mut carry: i64 = 0;

    for (i, pic) in frames.iter().enumerate() {
        let requested: u32 = match mode {
            HqRateControl::PerPicture => target_bytes,
            HqRateControl::Cbr => {
                // Grow the per-picture budget by accumulated surplus.
                // Clamp at u32::MAX so degenerate huge carries do not
                // wrap the requested-bytes field.
                let want = target_bytes as i64 + carry;
                want.clamp(1, u32::MAX as i64) as u32
            }
            HqRateControl::Vbv { buffer_bytes } => {
                // Leaky-bucket: identical to Cbr but the carry that may
                // be spent on this picture is capped at `buffer_bytes`,
                // so the per-picture request never exceeds
                // `target + buffer_bytes` — an instantaneous peak cap.
                // `carry` is already kept ≤ `buffer_bytes` at the
                // post-encode update below, so a simple `min` here
                // suffices; the explicit clamp guards against the
                // degenerate `buffer_bytes == 0` case (request == target).
                let spendable = carry.min(buffer_bytes as i64).max(0);
                let want = target_bytes as i64 + spendable;
                want.clamp(1, u32::MAX as i64) as u32
            }
            HqRateControl::VbvHysteresis {
                buffer_bytes,
                max_drain_per_picture,
            } => {
                // Drain-rate hysteresis: same bucket fill / forfeit
                // semantics as `Vbv` (post-encode carry clamp at
                // `buffer_bytes`), but the per-picture spend is
                // additionally capped at `max_drain_per_picture`.
                // Spendable = min(carry, buffer_bytes, max_drain_per_picture).
                // `max_drain_per_picture == 0` zeroes the spend and
                // collapses to `PerPicture`; `max_drain_per_picture >=
                // buffer_bytes` makes the drain cap inert and reduces
                // to plain `Vbv`. Either way the post-encode carry
                // update below clamps at `buffer_bytes` exactly as
                // `Vbv` does, so the bucket-fill state machine is
                // bit-identical between the two variants.
                let spendable = carry
                    .min(buffer_bytes as i64)
                    .min(max_drain_per_picture as i64)
                    .max(0);
                let want = target_bytes as i64 + spendable;
                want.clamp(1, u32::MAX as i64) as u32
            }
        };

        // Re-pick + re-encode at the (possibly carry-adjusted) budget.
        // Each picture's params keep `slice_size_target = None` and
        // get its own picked qindex written into every slice.
        let mut pic_params = params.clone();
        let qindex = pick_hq_picture_qindex(sequence, &pic_params, pic.y, pic.u, pic.v, requested);
        pic_params.qindex = qindex;

        let pic_payload = encode_hq_intra_picture(
            sequence,
            &pic_params,
            pic.picture_number,
            pic.y,
            pic.u,
            pic.v,
        );
        let actual_payload = pic_payload.len() as i64;

        // CBR / VBV feedback: target minus actual. The HQ picker never
        // overshoots `requested`, so for `target ≤ requested` (CBR is
        // monotone non-decreasing) the contribution is non-negative
        // unless qindex == 127 and even the floor overshoots target —
        // in which case `carry` may go briefly negative, naturally
        // tightening the next picture's budget.
        carry += target_bytes as i64 - actual_payload;
        // VBV / VbvHysteresis: clamp the running bucket at
        // `buffer_bytes`; savings above the bucket size are forfeited.
        // PerPicture / Cbr leave `carry` untouched (PerPicture's
        // request ignores carry anyway). VbvHysteresis uses the same
        // bucket cap as Vbv — only the per-picture *spend* is rate-
        // limited, the *fill* is identical.
        match mode {
            HqRateControl::Vbv { buffer_bytes }
            | HqRateControl::VbvHysteresis { buffer_bytes, .. } => {
                carry = carry.min(buffer_bytes as i64);
            }
            HqRateControl::PerPicture | HqRateControl::Cbr => {}
        }

        // Alternate parse codes — even index → non-reference (0xE8),
        // odd index → reference (0xEC). Matches the LD driver's
        // single-code choice (LD permits only non-ref per §D.1.1) but
        // exercises both HQ intra codes across the sequence.
        let parse_code = if i % 2 == 0 { 0xE8 } else { 0xEC };

        // Telemetry: report the running surplus *after* the VBV clamp
        // applied above. The HQ accumulator already follows the
        // "positive = savings, negative = debt" convention directly
        // (`carry += target - actual`), so no flip is needed.
        let running_surplus_bytes = carry;

        report.push(HqPictureRate {
            picture_number: pic.picture_number,
            requested_bytes: requested,
            actual_payload_bytes: actual_payload as u32,
            qindex,
            parse_code,
            running_surplus_bytes,
        });

        let pic_unit_len = (pi_size + pic_payload.len()) as u32;
        write_parse_info(&mut out, parse_code, pic_unit_len, prev_len);
        out.extend_from_slice(&pic_payload);
        prev_len = pic_unit_len;
        last_pic_len = Some(pic_unit_len);
    }

    write_parse_info(
        &mut out,
        0x10,
        0,
        last_pic_len.unwrap_or(sh_unit_len as u32),
    );
    (out, report)
}

/// Build a synthetic 64x64 4:2:0 YUV test pattern: a diagonal luma
/// gradient with a cross + a tinted chroma. Useful for the encoder
/// round-trip tests below.
pub fn synthetic_testsrc_64_yuv420() -> ([u8; 64 * 64], [u8; 32 * 32], [u8; 32 * 32]) {
    let mut y = [0u8; 64 * 64];
    let mut u = [0u8; 32 * 32];
    let mut v = [0u8; 32 * 32];
    for row in 0..64 {
        for col in 0..64 {
            let base = ((row + col) * 2) as u8;
            let pix = if row == 32 || col == 32 { 255u8 } else { base };
            y[row * 64 + col] = pix;
        }
    }
    for row in 0..32 {
        for col in 0..32 {
            u[row * 32 + col] = (col * 4) as u8;
            v[row * 32 + col] = (row * 4) as u8;
        }
    }
    (y, u, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitReader;
    use crate::sequence::parse_sequence_header;

    #[test]
    fn sequence_header_roundtrip_420() {
        let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
        let bytes = encode_sequence_header(&seq);
        let parsed = parse_sequence_header(&bytes).expect("parse");
        assert_eq!(parsed.video_params.frame_width, 64);
        assert_eq!(parsed.video_params.frame_height, 64);
        assert_eq!(parsed.video_params.chroma_format, ChromaFormat::Yuv420);
        assert_eq!(parsed.luma_width, 64);
        assert_eq!(parsed.chroma_width, 32);
        assert_eq!(parsed.luma_depth, 8);
        assert_eq!(parsed.chroma_depth, 8);
        assert_eq!(
            parsed.picture_coding_mode,
            crate::sequence::PictureCodingMode::Frames
        );
    }

    #[test]
    fn sequence_header_roundtrip_444() {
        let seq = make_minimal_sequence(32, 32, ChromaFormat::Yuv444);
        let bytes = encode_sequence_header(&seq);
        let parsed = parse_sequence_header(&bytes).expect("parse");
        assert_eq!(parsed.video_params.chroma_format, ChromaFormat::Yuv444);
        assert_eq!(parsed.chroma_width, 32);
        assert_eq!(parsed.chroma_height, 32);
    }

    /// `make_minimal_sequence_ld` must set profile=0 per §D.1.1 so
    /// conformance tools recognise the stream as Low Delay, not HQ.
    #[test]
    fn ld_minimal_sequence_sets_profile_zero() {
        let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
        assert_eq!(seq.parse_parameters.profile, 0, "LD profile per §D.1.1");
        // Round-trip — parse what we emit and confirm profile stays 0.
        let bytes = encode_sequence_header(&seq);
        let parsed = parse_sequence_header(&bytes).expect("parse");
        assert_eq!(parsed.parse_parameters.profile, 0);
    }

    /// `make_preset_sequence` produces an Annex C-conformant header:
    /// base_video_format carries the preset index and every custom
    /// flag stays clear (§D.2.3 Level 1).
    #[test]
    fn preset_sequence_is_level1_clean() {
        // QCIF = index 2 (§C.1 Table C.1), LD profile.
        let seq = make_preset_sequence(2, 0).expect("QCIF preset");
        assert_eq!(seq.parse_parameters.profile, 0);
        assert_eq!(seq.parse_parameters.level, 1);
        assert_eq!(seq.base_video_format_index, 2);
        assert_eq!(seq.video_params.frame_width, 176);
        assert_eq!(seq.video_params.frame_height, 144);
        assert_eq!(seq.video_params.chroma_format, ChromaFormat::Yuv420);

        // Parse back — the stream should decode to the same values.
        let bytes = encode_sequence_header(&seq);
        let parsed = parse_sequence_header(&bytes).expect("parse");
        assert_eq!(parsed.base_video_format_index, 2);
        assert_eq!(parsed.video_params.frame_width, 176);
        assert_eq!(parsed.video_params.frame_height, 144);
        assert_eq!(parsed.parse_parameters.level, 1);
    }

    /// `make_preset_sequence` rejects out-of-range indices (0 = Custom
    /// is handled by [`make_minimal_sequence`], not here).
    #[test]
    fn preset_sequence_rejects_bad_index() {
        assert!(make_preset_sequence(0, 0).is_none());
        assert!(make_preset_sequence(21, 0).is_none());
        assert!(make_preset_sequence(1, 0).is_some());
        assert!(make_preset_sequence(20, 0).is_some());
    }

    /// Multi-picture HQ stream alternates 0xE8 / 0xEC for the niche
    /// of intra-only reference + non-reference picture interleaving.
    #[test]
    fn hq_multi_stream_alternates_parse_codes() {
        let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
        let params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
        let (y, u, v) = synthetic_testsrc_64_yuv420();
        let pics = [
            InputPicture {
                picture_number: 0,
                y: &y,
                u: &u,
                v: &v,
            },
            InputPicture {
                picture_number: 1,
                y: &y,
                u: &u,
                v: &v,
            },
            InputPicture {
                picture_number: 2,
                y: &y,
                u: &u,
                v: &v,
            },
        ];
        let stream = encode_hq_intra_multi_stream(&seq, &params, &pics);
        // Walk parse-info units to confirm: seq_hdr, then 0xE8, 0xEC, 0xE8, EOS.
        let mut codes: Vec<u8> = Vec::new();
        let mut off = 0usize;
        while off + 13 <= stream.len() {
            assert_eq!(&stream[off..off + 4], BBCD, "BBCD prefix at {off}");
            let pc = stream[off + 4];
            codes.push(pc);
            let next = u32::from_be_bytes([
                stream[off + 5],
                stream[off + 6],
                stream[off + 7],
                stream[off + 8],
            ]);
            if next == 0 {
                break;
            }
            off += next as usize;
        }
        assert_eq!(codes, vec![0x00, 0xE8, 0xEC, 0xE8, 0x10]);
    }

    /// Multi-picture LD stream only uses 0xC8 (§D.1.1 — reference
    /// pictures are not permitted in Low Delay profile sequences).
    #[test]
    fn ld_multi_stream_stays_non_reference() {
        let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
        let params = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 128);
        let y = [128u8; 64 * 64];
        let u = [128u8; 32 * 32];
        let v = [128u8; 32 * 32];
        let pics = [
            InputPicture {
                picture_number: 0,
                y: &y,
                u: &u,
                v: &v,
            },
            InputPicture {
                picture_number: 1,
                y: &y,
                u: &u,
                v: &v,
            },
        ];
        let stream = encode_ld_intra_multi_stream(&seq, &params, &pics);
        // Walk and confirm seq_hdr + 2x 0xC8 + EOS — NEVER 0xCC.
        let mut codes: Vec<u8> = Vec::new();
        let mut off = 0usize;
        while off + 13 <= stream.len() {
            assert_eq!(&stream[off..off + 4], BBCD);
            let pc = stream[off + 4];
            codes.push(pc);
            let next = u32::from_be_bytes([
                stream[off + 5],
                stream[off + 6],
                stream[off + 7],
                stream[off + 8],
            ]);
            if next == 0 {
                break;
            }
            off += next as usize;
        }
        assert_eq!(codes, vec![0x00, 0xC8, 0xC8, 0x10]);
    }

    #[test]
    fn parse_info_header_layout() {
        let mut out = Vec::new();
        write_parse_info(&mut out, 0xE8, 100, 13);
        assert_eq!(&out[..4], BBCD);
        assert_eq!(out[4], 0xE8);
        assert_eq!(&out[5..9], &100u32.to_be_bytes());
        assert_eq!(&out[9..13], &13u32.to_be_bytes());
    }

    #[test]
    fn quantise_identity_at_q0() {
        // At qindex 0 (qf=4), forward quant = x * 4 / 4 = x. The
        // inverse then recovers x exactly (inverse_quant * q=0).
        for &x in &[-17i32, -1, 0, 1, 5, 42, 255] {
            assert_eq!(quantise_coeff(x, 0), x);
            assert_eq!(crate::quant::inverse_quant(quantise_coeff(x, 0), 0), x);
        }
    }

    /// Smoke-test: the emitted stream should start with BBCD + a
    /// sequence header parse code, followed by an HQ intra parse code.
    #[test]
    fn emitted_stream_has_expected_parse_codes() {
        let seq = make_minimal_sequence(64, 64, ChromaFormat::Yuv420);
        let params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
        let (y, u, v) = synthetic_testsrc_64_yuv420();
        let stream = encode_single_hq_intra_stream(&seq, &params, 0, &y, &u, &v);
        // First parse info.
        assert_eq!(&stream[..4], BBCD);
        assert_eq!(stream[4], 0x00);
        let sh_next = u32::from_be_bytes([stream[5], stream[6], stream[7], stream[8]]) as usize;
        assert!(sh_next > 13);
        // Next parse info must be at `sh_next`.
        assert_eq!(&stream[sh_next..sh_next + 4], BBCD);
        assert_eq!(stream[sh_next + 4], 0xE8);
    }

    /// The LD transform_parameters block we emit should parse back
    /// with identical values via the LD-profile reader in `picture.rs`.
    #[test]
    fn ld_transform_parameters_roundtrip() {
        let params = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 128);
        let mut w = BitWriter::new();
        write_ld_transform_parameters(&mut w, &params);
        w.byte_align();
        let bytes = w.finish();

        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_uint(), 1); // LeGall index
        assert_eq!(r.read_uint(), 3); // dwt_depth
        assert_eq!(r.read_uint(), 4); // slices_x
        assert_eq!(r.read_uint(), 4); // slices_y
        assert_eq!(r.read_uint(), 128 * 16); // slice_bytes_numer
        assert_eq!(r.read_uint(), 16); // slice_bytes_denom (slices_x*slices_y)
        assert!(!r.read_bool()); // custom quant flag off
    }

    /// `forward_dc_prediction` subtracts the same neighbour mean the
    /// decoder adds back — so composing the two on any integer band
    /// is the identity.
    #[test]
    fn forward_dc_prediction_inverts_decoder_prediction() {
        let mut band = SubbandData::new(4, 4);
        let mut expected = SubbandData::new(4, 4);
        // Arbitrary test pattern.
        for y in 0..4 {
            for x in 0..4 {
                let v = (y as i32 * 7 + x as i32 * 3 - 5) * (if (x + y) % 2 == 0 { 1 } else { -1 });
                band.set(y, x, v);
                expected.set(y, x, v);
            }
        }
        forward_dc_prediction(&mut band);
        crate::picture::intra_dc_prediction(&mut band);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(
                    band.get(y, x),
                    expected.get(y, x),
                    "round-trip mismatch at ({y}, {x})"
                );
            }
        }
    }

    /// The emitted LD stream should start with a sequence-header parse
    /// code and then a 0xC8 LD non-reference intra picture.
    #[test]
    fn ld_stream_has_expected_parse_codes() {
        let seq = make_minimal_sequence_ld(64, 64, ChromaFormat::Yuv420);
        let params = LdEncoderParams::default_ld(WaveletFilter::LeGall5_3, 3, 4, 4, 128);
        // Use a smooth pattern so the 128-byte budget is sufficient.
        let mut y = [0u8; 64 * 64];
        let u = [128u8; 32 * 32];
        let v = [128u8; 32 * 32];
        for row in 0..64 {
            for col in 0..64 {
                y[row * 64 + col] = ((row + col) * 2) as u8;
            }
        }
        let stream = encode_single_ld_intra_stream(&seq, &params, 0, &y, &u, &v);
        assert_eq!(&stream[..4], BBCD);
        assert_eq!(stream[4], 0x00); // sequence header
        let sh_next = u32::from_be_bytes([stream[5], stream[6], stream[7], stream[8]]) as usize;
        assert!(sh_next > 13);
        assert_eq!(&stream[sh_next..sh_next + 4], BBCD);
        assert_eq!(stream[sh_next + 4], 0xC8); // LD non-ref intra
    }

    /// The transform_parameters block we emit should parse back with
    /// the same values.
    #[test]
    fn transform_parameters_roundtrip() {
        let params = EncoderParams::default_hq(WaveletFilter::LeGall5_3, 3);
        let mut w = BitWriter::new();
        write_transform_parameters(&mut w, &params);
        w.byte_align();
        let bytes = w.finish();

        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_uint(), 1); // LeGall index
        assert_eq!(r.read_uint(), 3); // dwt_depth
        assert_eq!(r.read_uint(), 8); // slices_x
        assert_eq!(r.read_uint(), 8); // slices_y
        assert_eq!(r.read_uint(), 0); // slice_prefix_bytes
        assert_eq!(r.read_uint(), 1); // slice_size_scaler
        assert!(!r.read_bool()); // custom quant flag off
    }
}
