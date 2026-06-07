//! Shared per-voxel fixel-matching primitives used by both the pairwise
//! [`crate::compare`] and the N-way [`crate::combine`] tools, so the two keep
//! identical matching semantics: greedy `max(|dot|)` over a shared grid, with
//! voxels aligned by full-volume index.

use crate::error::{OdxError, Result};

/// Maps every full-volume flat voxel index to its compact in-mask row index,
/// or `usize::MAX` when the voxel is outside the mask.
///
/// Flat indexing is C-order (`flat = i*ny*nz + j*nz + k`), matching the ODX
/// mask layout and [`crate::odx_file::OdxDataset::compact_to_ijk`]. This is the
/// canonical way to align voxels across datasets whose *masks differ* but whose
/// *grid is identical* — never assume the compact row index is shared.
pub(crate) struct VoxelLookup {
    pub(crate) flat_to_compact: Vec<usize>,
}

/// Full-volume flat → compact-row map for a mask, without any offsets check.
///
/// Use when the compact row order is needed before peak offsets exist (e.g. the
/// combine template, whose fixels are built *after* voxel alignment). `mask`
/// must be `dims`-sized (C-order); panics otherwise in debug.
pub(crate) fn compact_index_map(mask: &[u8], dims: [u64; 3]) -> Vec<usize> {
    let total = (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize);
    debug_assert_eq!(mask.len(), total, "mask len must equal product of dims");
    let mut flat_to_compact = vec![usize::MAX; total];
    let yz = (dims[1] as usize) * (dims[2] as usize);
    let z = dims[2] as usize;
    let mut compact = 0usize;
    for i in 0..dims[0] as usize {
        for j in 0..dims[1] as usize {
            for k in 0..dims[2] as usize {
                let flat = i * yz + j * z + k;
                if mask[flat] != 0 {
                    flat_to_compact[flat] = compact;
                    compact += 1;
                }
            }
        }
    }
    flat_to_compact
}

/// Build a [`VoxelLookup`] from a full-volume mask, validating that the number
/// of in-mask voxels equals `offsets.len() - 1` (the ODX offsets sentinel).
pub(crate) fn build_voxel_lookup(
    mask: &[u8],
    dims: [u64; 3],
    offsets: &[u32],
) -> Result<VoxelLookup> {
    let total = (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize);
    if mask.len() != total {
        return Err(OdxError::Format(format!(
            "mask len {} != product of dims {}",
            mask.len(),
            total
        )));
    }
    let flat_to_compact = compact_index_map(mask, dims);
    let compact = flat_to_compact.iter().filter(|&&c| c != usize::MAX).count();
    if compact + 1 != offsets.len() {
        return Err(OdxError::Format(format!(
            "mask voxel count {} differs from offsets-1 {}",
            compact,
            offsets.len() - 1
        )));
    }
    Ok(VoxelLookup { flat_to_compact })
}

/// True when two 4×4 affines agree within `tol` element-wise.
pub(crate) fn affine_close(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4], tol: f64) -> bool {
    for r in 0..4 {
        for c in 0..4 {
            if (a[r][c] - b[r][c]).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Absolute dot product of two 3-vectors — the antipodal-symmetric cosine used
/// for fixel direction matching (fixel axes are sign-agnostic).
pub(crate) fn abs_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).abs()
}
