//! Shared helpers for moving DPV-style per-voxel scalars between ODX
//! compact (mask-only) ordering and NIfTI-1 full-grid volumes.
//!
//! The module gathers:
//!
//! 1. **Header construction** — [`nifti_header_for_grid`] builds a
//!    `NiftiHeader` with the ODX affine baked into both `qform` and `sform`
//!    slots (only `qform_code` is active; `sform` data is preserved but
//!    flagged `Unknown` so downstream tools converge on `qform`). See the
//!    detailed sform/qform discussion under the function.
//!
//! 2. **Scatter & gather** — [`dpv_to_volume`] projects a compact
//!    DPV array onto the full `(X, Y, Z)` grid; [`volume_to_dpv`] is its
//!    inverse, picking out mask-indexed voxels in C order. Both are pure
//!    array ops; they don't need a loaded `OdxDataset`, only the
//!    `compact_to_ijk` map and dimensions.
//!
//! 3. **Writers** — [`write_voxel_scalar_nifti_u16`] (and `_u8`, `_u32`,
//!    `_f32`) are end-to-end: take a compact array + grid metadata, do the
//!    scatter, and write a NIfTI to disk with the correct header policy.
//!
//! 4. **Attach helpers** — [`attach_dpv_from_volume`] is the *import* side:
//!    given an ODX path and a full-grid volume, gather to compact order
//!    and append a DPV to the archive or directory in place. Used by the
//!    Python bindings and the `odx attach-dpv` CLI subcommand.
//!
//! Why both qform and sform get the affine, but only qform is marked
//! active:
//!
//! * The previous private helpers populated `sform` only; that worked for
//!   FSLeyes / MRView, but software that prefers `qform` (older FreeSurfer
//!   pipelines, some Talairach-aware tools) got `qform_code = 0` and
//!   silently fell back to identity geometry.
//! * Writing `srow_*` *and* the quaternion + offsets, with
//!   `qform_code = ScannerAnat` and `sform_code = Unknown`, means
//!   - tools that prefer qform see a valid `ScannerAnat` transform;
//!   - tools that prefer sform see `sform_code = 0` and correctly fall
//!     back to the qform, so every reader converges on the same geometry;
//!   - the sform data is still on disk, so a manual `fslorient` /
//!     `nifti_tool` rewrite can promote sform to active later without
//!     re-deriving the matrix.

use std::collections::HashMap;
use std::path::Path;

use bytemuck::Pod;
use nalgebra::Matrix4;
use ndarray::{Array3, ArrayView3};
use nifti::writer::WriterOptions;
use nifti::{NiftiHeader, NiftiType, XForm};
use zip::CompressionMethod;

use crate::data_array::DataArray;
use crate::dtype::DType;
use crate::error::{OdxError, Result};
use crate::header::Header;
use crate::io::directory::append_dpv_to_directory;
use crate::io::zip::append_dpv_to_zip;

// ----------------------------------------------------------------------
// Header construction
// ----------------------------------------------------------------------

/// Build a `NiftiHeader` for a volume whose voxel→RAS+ mm affine is
/// `voxel_to_ras` (row-major).
///
/// Populates:
/// - `pixdim[1..4]` — spacing magnitudes derived from the affine.
/// - `xyzt_units` — `2` (millimetres for spatial units, no time unit).
/// - `sform` data (`srow_*`) — copied from the affine. The **code is set
///   to `Unknown` (0)**, so readers will skip the sform and fall back to
///   the qform. The data is still on disk so it can be promoted later by
///   a tool like `fslorient -setsform` without re-deriving the matrix.
/// - `qform` (code `ScannerAnat`) — quaternion + offsets derived from the
///   affine. Authoritative slot for downstream software. ODX affines are
///   rigid + zoom, so quaternion round-trip is exact in practice.
///
/// `dim`, `datatype`, and `bitpix` are intentionally NOT set — the
/// [`nifti::writer::WriterOptions`] writer derives them from the array
/// shape and element type, so any value here would be overwritten.
pub fn nifti_header_for_grid(voxel_to_ras: [[f64; 4]; 4]) -> NiftiHeader {
    let mut hdr = NiftiHeader::default();

    let dx = (voxel_to_ras[0][0].powi(2)
        + voxel_to_ras[1][0].powi(2)
        + voxel_to_ras[2][0].powi(2))
    .sqrt() as f32;
    let dy = (voxel_to_ras[0][1].powi(2)
        + voxel_to_ras[1][1].powi(2)
        + voxel_to_ras[2][1].powi(2))
    .sqrt() as f32;
    let dz = (voxel_to_ras[0][2].powi(2)
        + voxel_to_ras[1][2].powi(2)
        + voxel_to_ras[2][2].powi(2))
    .sqrt() as f32;
    hdr.pixdim = [1.0, dx, dy, dz, 0.0, 0.0, 0.0, 0.0];

    // 2 = NIFTI_UNITS_MM (spatial units in millimetres, no temporal flag).
    hdr.xyzt_units = 2;

    let affine_mat = Matrix4::<f64>::new(
        voxel_to_ras[0][0], voxel_to_ras[0][1], voxel_to_ras[0][2], voxel_to_ras[0][3],
        voxel_to_ras[1][0], voxel_to_ras[1][1], voxel_to_ras[1][2], voxel_to_ras[1][3],
        voxel_to_ras[2][0], voxel_to_ras[2][1], voxel_to_ras[2][2], voxel_to_ras[2][3],
        voxel_to_ras[3][0], voxel_to_ras[3][1], voxel_to_ras[3][2], voxel_to_ras[3][3],
    );

    // Populate both slots. Code policy:
    //   qform_code = ScannerAnat (1) — primary, what readers should use.
    //   sform_code = Unknown    (0) — data still on disk, but flagged
    //     inactive so any reader that prefers sform falls back to qform.
    hdr.set_qform(&affine_mat, XForm::ScannerAnat);
    hdr.set_sform(&affine_mat, XForm::Unknown);

    hdr
}

// ----------------------------------------------------------------------
// Scatter / gather between compact (mask-only) and full-grid arrays
// ----------------------------------------------------------------------

/// Scatter a compact-order DPV array onto a full `(X, Y, Z)` grid.
/// Voxels outside the compact set (i.e. outside the mask) receive `fill`.
///
/// `compact_to_ijk` must come from
/// [`OdxDataset::compact_to_ijk`](crate::OdxDataset::compact_to_ijk) — its
/// length defines the DPV row count and its `(i, j, k)` entries are in
/// C-order (i-slowest, k-fastest).
///
/// `values_compact.len() == compact_to_ijk.len()` is required and asserted.
pub fn dpv_to_volume<T: Copy>(
    values_compact: &[T],
    compact_to_ijk: &[[u32; 3]],
    dims: [usize; 3],
    fill: T,
) -> Array3<T> {
    assert_eq!(
        values_compact.len(),
        compact_to_ijk.len(),
        "dpv_to_volume: values and compact_to_ijk length mismatch"
    );
    let mut vol = Array3::<T>::from_elem((dims[0], dims[1], dims[2]), fill);
    for (compact, ijk) in compact_to_ijk.iter().enumerate() {
        vol[[ijk[0] as usize, ijk[1] as usize, ijk[2] as usize]] = values_compact[compact];
    }
    vol
}

/// Gather a full-grid volume into compact (mask-only) order using the
/// provided `compact_to_ijk` map. Values outside the compact set are
/// discarded — by construction, only mask-positive voxels make it into the
/// returned vector.
///
/// The volume must match the ODX grid: `vol.shape() == [nx, ny, nz]`. The
/// returned `Vec` has length `compact_to_ijk.len()`.
pub fn volume_to_dpv<T: Copy>(
    vol: ArrayView3<'_, T>,
    compact_to_ijk: &[[u32; 3]],
) -> Vec<T> {
    let mut out = Vec::with_capacity(compact_to_ijk.len());
    for ijk in compact_to_ijk {
        out.push(vol[[ijk[0] as usize, ijk[1] as usize, ijk[2] as usize]]);
    }
    out
}

// ----------------------------------------------------------------------
// Writers: compact array → NIfTI on disk
// ----------------------------------------------------------------------

/// Project a `u16` compact-order DPV onto the dataset's full
/// `dims[0] × dims[1] × dims[2]` grid and write as a NIfTI-1 volume at
/// `path`. Voxels outside the compact set get 0.
///
/// See module-level docs for the sform/qform policy.
pub fn write_voxel_scalar_nifti_u16(
    path: &Path,
    values_compact: &[u16],
    compact_to_ijk: &[[u32; 3]],
    dims: [usize; 3],
    voxel_to_ras: [[f64; 4]; 4],
) -> Result<()> {
    let vol = dpv_to_volume(values_compact, compact_to_ijk, dims, 0u16);
    write_3d_volume(path, vol.view(), NiftiType::Uint16, voxel_to_ras)
}

pub fn write_voxel_scalar_nifti_u8(
    path: &Path,
    values_compact: &[u8],
    compact_to_ijk: &[[u32; 3]],
    dims: [usize; 3],
    voxel_to_ras: [[f64; 4]; 4],
) -> Result<()> {
    let vol = dpv_to_volume(values_compact, compact_to_ijk, dims, 0u8);
    write_3d_volume(path, vol.view(), NiftiType::Uint8, voxel_to_ras)
}

pub fn write_voxel_scalar_nifti_u32(
    path: &Path,
    values_compact: &[u32],
    compact_to_ijk: &[[u32; 3]],
    dims: [usize; 3],
    voxel_to_ras: [[f64; 4]; 4],
) -> Result<()> {
    let vol = dpv_to_volume(values_compact, compact_to_ijk, dims, 0u32);
    write_3d_volume(path, vol.view(), NiftiType::Uint32, voxel_to_ras)
}

pub fn write_voxel_scalar_nifti_f32(
    path: &Path,
    values_compact: &[f32],
    compact_to_ijk: &[[u32; 3]],
    dims: [usize; 3],
    voxel_to_ras: [[f64; 4]; 4],
) -> Result<()> {
    let vol = dpv_to_volume(values_compact, compact_to_ijk, dims, 0.0f32);
    write_3d_volume(path, vol.view(), NiftiType::Float32, voxel_to_ras)
}

fn write_3d_volume<T>(
    path: &Path,
    vol: ArrayView3<'_, T>,
    datatype: NiftiType,
    voxel_to_ras: [[f64; 4]; 4],
) -> Result<()>
where
    T: Pod + nifti::DataElement,
{
    let hdr = nifti_header_for_grid(voxel_to_ras);
    WriterOptions::new(path)
        .reference_header(&hdr)
        .write_nifti_with_type(&vol, datatype)
        .map_err(|err| {
            OdxError::Format(format!(
                "failed to write NIfTI '{}': {err}",
                path.display()
            ))
        })?;
    Ok(())
}

// ----------------------------------------------------------------------
// Attach a NIfTI volume to an ODX as a DPV (in place)
// ----------------------------------------------------------------------

/// What datatype should the appended DPV have?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DpvDtype {
    /// Pick `u8` / `u16` / `u32` if the source array is integer and fits;
    /// otherwise fall back to `f32`. Matches what most users want.
    Auto,
    UInt8,
    UInt16,
    UInt32,
    Int16,
    Int32,
    Float32,
    Float64,
}

impl DpvDtype {
    fn to_dtype(self) -> DType {
        match self {
            DpvDtype::UInt8 => DType::UInt8,
            DpvDtype::UInt16 => DType::UInt16,
            DpvDtype::UInt32 => DType::UInt32,
            DpvDtype::Int16 => DType::Int16,
            DpvDtype::Int32 => DType::Int32,
            DpvDtype::Float32 => DType::Float32,
            DpvDtype::Float64 => DType::Float64,
            DpvDtype::Auto => DType::Float32, // fallback if Auto leaks through
        }
    }
}

/// Tolerance for comparing the input volume's voxel→RAS+ affine against
/// the ODX's affine when attaching a DPV. Roughly "1 / 1000 of a voxel" —
/// generous enough to absorb the float32↔float64 round-trip errors that
/// happen when NIfTI affines get re-derived from quaternions, strict
/// enough that a different acquisition / resampled grid won't sneak past.
pub const ATTACH_AFFINE_TOLERANCE_MM: f64 = 1e-3;

/// Append a per-voxel scalar (DPV) to an existing ODX at `path` from a
/// full-grid `f64` volume. The caller is responsible for matching grid
/// dimensions and voxel→RAS+ affine; both are validated and a mismatch
/// raises `OdxError::Format` with a clear message.
///
/// `dtype` chooses the on-disk DPV datatype:
/// - `DpvDtype::Auto` picks the narrowest unsigned integer (`u8`, `u16`,
///   `u32`) if the volume contains only nonnegative integral values, else
///   falls back to `f32`.
/// - Any explicit variant forces a cast; values outside the destination
///   range are clamped and a warning emitted via the returned report.
///
/// Used by:
/// - `odx attach-dpv` CLI subcommand
/// - `odx.attach_dpv()` Python entry point
/// - `odx.adapters.nibabel.attach_dpv()` (via the Python entry point)
pub fn attach_dpv_from_volume(
    odx_path: &Path,
    name: &str,
    vol: ArrayView3<'_, f64>,
    vol_affine: [[f64; 4]; 4],
    dtype: DpvDtype,
) -> Result<DpvAttachReport> {
    // 1. Load just the ODX header + mask so we can validate grid and
    //    build the compact-to-ijk map. We deliberately don't open the
    //    full dataset (no need to materialise SH/DPF/etc).
    let (header, mask) = load_header_and_mask(odx_path)?;

    let dims = [
        header.dimensions[0] as usize,
        header.dimensions[1] as usize,
        header.dimensions[2] as usize,
    ];
    let nb_voxels = header.nb_voxels as usize;
    let odx_affine = header.voxel_to_rasmm;

    // 2. Validate grid.
    let vol_shape = vol.shape();
    if vol_shape.len() != 3
        || vol_shape[0] != dims[0]
        || vol_shape[1] != dims[1]
        || vol_shape[2] != dims[2]
    {
        return Err(OdxError::Format(format!(
            "attach DPV '{name}': volume shape {:?} does not match ODX dimensions [{}, {}, {}]",
            vol_shape, dims[0], dims[1], dims[2]
        )));
    }
    if !affines_close(&vol_affine, &odx_affine, ATTACH_AFFINE_TOLERANCE_MM) {
        return Err(OdxError::Format(format!(
            "attach DPV '{name}': volume affine does not match ODX affine \
             (tolerance {ATTACH_AFFINE_TOLERANCE_MM} mm). Resample the volume \
             onto the ODX grid first (e.g. with `odx transform` or an \
             external tool) before attaching."
        )));
    }

    // 3. Gather full-grid → compact-order.
    let compact_to_ijk = compact_to_ijk_from_mask(&mask, header.dimensions);
    debug_assert_eq!(compact_to_ijk.len(), nb_voxels);
    let compact_f64: Vec<f64> = volume_to_dpv(vol, &compact_to_ijk);

    // 4. Choose the storage dtype and pack bytes.
    let resolved_dtype = if matches!(dtype, DpvDtype::Auto) {
        infer_compact_dtype(&compact_f64)
    } else {
        dtype.to_dtype()
    };
    let (bytes, range_warning) = pack_compact_to_bytes(&compact_f64, resolved_dtype);

    // 5. Append via the existing in-place writers.
    let mut map = HashMap::new();
    map.insert(
        name.to_string(),
        DataArray::owned_bytes(bytes, 1, resolved_dtype),
    );
    append_dpv_dispatch(odx_path, &map)?;

    Ok(DpvAttachReport {
        name: name.to_string(),
        dtype: resolved_dtype,
        nb_voxels,
        masked_in_count: compact_f64.iter().filter(|v| **v != 0.0).count(),
        clamped: range_warning,
    })
}

/// Summary returned by [`attach_dpv_from_volume`] — useful for CLI output
/// and Python warnings.
#[derive(Debug, Clone)]
pub struct DpvAttachReport {
    pub name: String,
    pub dtype: DType,
    pub nb_voxels: usize,
    /// How many compact voxels had a nonzero value.
    pub masked_in_count: usize,
    /// `true` iff explicit-dtype packing clamped at least one value.
    pub clamped: bool,
}

// ----------------------------------------------------------------------
// Internal helpers
// ----------------------------------------------------------------------

fn load_header_and_mask(path: &Path) -> Result<(Header, Vec<u8>)> {
    // We avoid `OdxDataset::open` here: it pulls in DPF / SH / directions
    // / etc., which we don't need just to validate grid and gather voxels.
    use crate::OdxDataset;
    let dataset = OdxDataset::open(path)?;
    let header = dataset.header().clone();
    let mask = dataset.mask().to_vec();
    Ok((header, mask))
}

fn compact_to_ijk_from_mask(mask: &[u8], dimensions: [u64; 3]) -> Vec<[u32; 3]> {
    let nx = dimensions[0] as u32;
    let ny = dimensions[1] as u32;
    let nz = dimensions[2] as u32;
    let stride_i = (dimensions[1] * dimensions[2]) as usize;
    let stride_j = dimensions[2] as usize;
    let mut out = Vec::new();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let flat = i as usize * stride_i + j as usize * stride_j + k as usize;
                if mask[flat] != 0 {
                    out.push([i, j, k]);
                }
            }
        }
    }
    out
}

fn affines_close(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4], tol: f64) -> bool {
    for r in 0..4 {
        for c in 0..4 {
            if (a[r][c] - b[r][c]).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Infer the narrowest sensible DPV dtype for a compact `f64` array:
/// - all values integral and in `[0, u8::MAX]` → `UInt8`
/// - all values integral and in `[0, u16::MAX]` → `UInt16`
/// - all values integral and in `[0, u32::MAX]` → `UInt32`
/// - otherwise → `Float32`
fn infer_compact_dtype(values: &[f64]) -> DType {
    let mut all_integral_nonneg = true;
    let mut max = 0.0f64;
    for &v in values {
        if !v.is_finite() || v < 0.0 || v.fract() != 0.0 {
            all_integral_nonneg = false;
            break;
        }
        if v > max {
            max = v;
        }
    }
    if !all_integral_nonneg {
        return DType::Float32;
    }
    if max <= u8::MAX as f64 {
        DType::UInt8
    } else if max <= u16::MAX as f64 {
        DType::UInt16
    } else if max <= u32::MAX as f64 {
        DType::UInt32
    } else {
        DType::Float32
    }
}

fn pack_compact_to_bytes(values: &[f64], dtype: DType) -> (Vec<u8>, bool) {
    let mut clamped = false;
    let bytes = match dtype {
        DType::UInt8 => values
            .iter()
            .flat_map(|&v| {
                let (c, was_clamped) = clamp_to_u8(v);
                clamped |= was_clamped;
                c.to_le_bytes().to_vec()
            })
            .collect(),
        DType::UInt16 => values
            .iter()
            .flat_map(|&v| {
                let (c, was_clamped) = clamp_to_u16(v);
                clamped |= was_clamped;
                c.to_le_bytes().to_vec()
            })
            .collect(),
        DType::UInt32 => values
            .iter()
            .flat_map(|&v| {
                let (c, was_clamped) = clamp_to_u32(v);
                clamped |= was_clamped;
                c.to_le_bytes().to_vec()
            })
            .collect(),
        DType::Int16 => values
            .iter()
            .flat_map(|&v| {
                let (c, was_clamped) = clamp_to_i16(v);
                clamped |= was_clamped;
                c.to_le_bytes().to_vec()
            })
            .collect(),
        DType::Int32 => values
            .iter()
            .flat_map(|&v| {
                let (c, was_clamped) = clamp_to_i32(v);
                clamped |= was_clamped;
                c.to_le_bytes().to_vec()
            })
            .collect(),
        DType::Float32 => values
            .iter()
            .flat_map(|&v| (v as f32).to_le_bytes())
            .collect(),
        DType::Float64 => values.iter().flat_map(|&v| v.to_le_bytes()).collect(),
        other => panic!("pack_compact_to_bytes: unsupported DPV dtype {other:?}"),
    };
    (bytes, clamped)
}

fn clamp_to_u8(v: f64) -> (u8, bool) {
    if v.is_nan() {
        return (0, true);
    }
    let r = v.round();
    if r < 0.0 {
        (0, true)
    } else if r > u8::MAX as f64 {
        (u8::MAX, true)
    } else {
        (r as u8, r != v)
    }
}
fn clamp_to_u16(v: f64) -> (u16, bool) {
    if v.is_nan() {
        return (0, true);
    }
    let r = v.round();
    if r < 0.0 {
        (0, true)
    } else if r > u16::MAX as f64 {
        (u16::MAX, true)
    } else {
        (r as u16, r != v)
    }
}
fn clamp_to_u32(v: f64) -> (u32, bool) {
    if v.is_nan() {
        return (0, true);
    }
    let r = v.round();
    if r < 0.0 {
        (0, true)
    } else if r > u32::MAX as f64 {
        (u32::MAX, true)
    } else {
        (r as u32, r != v)
    }
}
fn clamp_to_i16(v: f64) -> (i16, bool) {
    if v.is_nan() {
        return (0, true);
    }
    let r = v.round();
    if r < i16::MIN as f64 {
        (i16::MIN, true)
    } else if r > i16::MAX as f64 {
        (i16::MAX, true)
    } else {
        (r as i16, r != v)
    }
}
fn clamp_to_i32(v: f64) -> (i32, bool) {
    if v.is_nan() {
        return (0, true);
    }
    let r = v.round();
    if r < i32::MIN as f64 {
        (i32::MIN, true)
    } else if r > i32::MAX as f64 {
        (i32::MAX, true)
    } else {
        (r as i32, r != v)
    }
}

fn append_dpv_dispatch(path: &Path, dpv: &HashMap<String, DataArray>) -> Result<()> {
    if path.is_dir() {
        append_dpv_to_directory(path, dpv, /*overwrite=*/ true)
    } else {
        // Treat anything else as an archive (.odx). The append_dpv_to_zip
        // helper validates the format internally.
        append_dpv_to_zip(path, dpv, CompressionMethod::Deflated, /*overwrite=*/ true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scatter_gather_roundtrip() {
        // 3x2x2 grid, three masked voxels.
        let dims = [3usize, 2, 2];
        let compact_to_ijk: Vec<[u32; 3]> = vec![[0, 0, 0], [1, 1, 1], [2, 0, 1]];
        let compact = vec![10u16, 20, 30];

        let vol = dpv_to_volume(&compact, &compact_to_ijk, dims, 0u16);
        assert_eq!(vol[[0, 0, 0]], 10);
        assert_eq!(vol[[1, 1, 1]], 20);
        assert_eq!(vol[[2, 0, 1]], 30);
        assert_eq!(vol[[0, 0, 1]], 0); // outside compact set

        let back = volume_to_dpv(vol.view(), &compact_to_ijk);
        assert_eq!(back, compact);
    }

    #[test]
    fn infer_dtype_picks_u8_for_small_integers() {
        let v = vec![0.0, 1.0, 200.0];
        assert_eq!(infer_compact_dtype(&v), DType::UInt8);
    }

    #[test]
    fn infer_dtype_picks_u16_for_medium_integers() {
        let v = vec![0.0, 1000.0, 60000.0];
        assert_eq!(infer_compact_dtype(&v), DType::UInt16);
    }

    #[test]
    fn infer_dtype_picks_float32_for_fractions() {
        let v = vec![0.0, 1.5, 2.0];
        assert_eq!(infer_compact_dtype(&v), DType::Float32);
    }

    #[test]
    fn infer_dtype_picks_float32_for_negatives() {
        let v = vec![-1.0, 0.0, 1.0];
        assert_eq!(infer_compact_dtype(&v), DType::Float32);
    }

    #[test]
    fn clamp_negative_to_u8_clamps() {
        let (val, clamped) = clamp_to_u8(-3.0);
        assert_eq!(val, 0);
        assert!(clamped);
    }

    #[test]
    fn clamp_overflow_to_u8_clamps_to_max() {
        let (val, clamped) = clamp_to_u8(300.0);
        assert_eq!(val, u8::MAX);
        assert!(clamped);
    }

    #[test]
    fn affines_close_handles_small_drift() {
        let a = [[1.0, 0.0, 0.0, 0.0]; 4];
        let mut b = a;
        b[0][3] = 1e-6;
        assert!(affines_close(&a, &b, 1e-3));
        b[0][3] = 1e-2;
        assert!(!affines_close(&a, &b, 1e-3));
    }
}
