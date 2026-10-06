//! Shared per-voxel fixel-matching and grid-alignment primitives used by the
//! pairwise [`crate::compare`], the N-way [`crate::combine`], and the FOD
//! aggregation core in [`crate::template`], so all three keep identical
//! semantics: greedy `max(|dot|)` matching over a shared grid, with voxels
//! aligned by full-volume index.

use std::collections::BTreeSet;

use crate::error::{OdxError, Result};
use crate::odx_file::OdxDataset;
use crate::qc::QC_CLASS_DPF_NAME;

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

/// C-order flat index of a voxel: `i*ny*nz + j*nz + k`.
#[inline]
pub(crate) fn flat_index(ijk: [u32; 3], dims: [u64; 3]) -> usize {
    let (ny, nz) = (dims[1] as usize, dims[2] as usize);
    ijk[0] as usize * ny * nz + ijk[1] as usize * nz + ijk[2] as usize
}

/// The `(i, j, k)` of every in-mask voxel, in compact row order (C-order scan).
///
/// The ijk twin of [`compact_index_map`]: that maps flat → compact, this maps
/// compact → ijk. Matches [`crate::odx_file::OdxDataset::compact_to_ijk`] but
/// works on a bare mask, before any dataset exists.
pub(crate) fn mask_compact_ijk(mask: &[u8], dims: [u64; 3]) -> Vec<[u32; 3]> {
    let (ny, nz) = (dims[1] as usize, dims[2] as usize);
    let mut out = Vec::new();
    for i in 0..dims[0] as u32 {
        for j in 0..dims[1] as u32 {
            for k in 0..dims[2] as u32 {
                let flat = i as usize * ny * nz + j as usize * nz + k as usize;
                if mask[flat] != 0 {
                    out.push([i, j, k]);
                }
            }
        }
    }
    out
}

/// Apply a 4x4 affine (last row `[0,0,0,1]`) to a point.
pub(crate) fn apply_affine_pt(a: &[[f64; 4]; 4], p: [f64; 3]) -> [f64; 3] {
    [
        a[0][0] * p[0] + a[0][1] * p[1] + a[0][2] * p[2] + a[0][3],
        a[1][0] * p[0] + a[1][1] * p[1] + a[1][2] * p[2] + a[1][3],
        a[2][0] * p[0] + a[2][1] * p[1] + a[2][2] * p[2] + a[2][3],
    ]
}

/// Invert a voxel→world affine (last row `[0,0,0,1]`): inverts the 3×3 linear
/// block by cofactors and adjusts the translation. Returns `None` if singular.
pub(crate) fn invert_affine4(a: &[[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
    let m = [
        [a[0][0], a[0][1], a[0][2]],
        [a[1][0], a[1][1], a[1][2]],
        [a[2][0], a[2][1], a[2][2]],
    ];
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-12 {
        return None;
    }
    let id = 1.0 / det;
    let inv = [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * id,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * id,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * id,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * id,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * id,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * id,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * id,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * id,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * id,
        ],
    ];
    let t = [a[0][3], a[1][3], a[2][3]];
    let it = [
        -(inv[0][0] * t[0] + inv[0][1] * t[1] + inv[0][2] * t[2]),
        -(inv[1][0] * t[0] + inv[1][1] * t[1] + inv[1][2] * t[2]),
        -(inv[2][0] * t[0] + inv[2][1] * t[1] + inv[2][2] * t[2]),
    ];
    Some([
        [inv[0][0], inv[0][1], inv[0][2], it[0]],
        [inv[1][0], inv[1][1], inv[1][2], it[1]],
        [inv[2][0], inv[2][1], inv[2][2], it[2]],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

/// If two grids of equal `dims` occupy the same physical voxel lattice — i.e.
/// each A voxel maps, through `A`→world→`B⁻¹`, to an integer in-range B voxel —
/// return the flat-index remap `remap[a_flat] = b_flat` (C order,
/// `i*ny*nz + j*nz + k`). Otherwise return `None` (genuinely different grids).
/// This is exactly the LAS↔RAS / axis-permutation case: same scanner space,
/// different voxel ordering.
pub(crate) fn same_lattice_voxel_remap(
    dims: [u64; 3],
    a: &[[f64; 4]; 4],
    b: &[[f64; 4]; 4],
) -> Option<Vec<usize>> {
    let inv_b = invert_affine4(b)?;
    let (nx, ny, nz) = (dims[0] as usize, dims[1] as usize, dims[2] as usize);
    let mut remap = vec![0usize; nx * ny * nz];
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let world = apply_affine_pt(a, [i as f64, j as f64, k as f64]);
                let vb = apply_affine_pt(&inv_b, world);
                let (ri, rj, rk) = (vb[0].round(), vb[1].round(), vb[2].round());
                if (vb[0] - ri).abs() > 1e-3
                    || (vb[1] - rj).abs() > 1e-3
                    || (vb[2] - rk).abs() > 1e-3
                {
                    return None;
                }
                if ri < 0.0
                    || rj < 0.0
                    || rk < 0.0
                    || ri as usize >= nx
                    || rj as usize >= ny
                    || rk as usize >= nz
                {
                    return None;
                }
                remap[i * ny * nz + j * nz + k] =
                    (ri as usize) * ny * nz + (rj as usize) * nz + (rk as usize);
            }
        }
    }
    Some(remap)
}

/// Map every full-volume flat index **in the reference grid** to this input's
/// compact row (`usize::MAX` outside its mask), tolerating a signed
/// axis-permutation difference in voxel ordering.
///
/// Three cases, in order: identical affine → plain [`compact_index_map`]; same
/// physical lattice with permuted/flipped axes → [`same_lattice_voxel_remap`]
/// composed with the compact map; anything else → `Err`.
///
/// **Why the remap needs no SH rotation.** ODX stores `directions` in world
/// (RAS mm) space, and every spherical-harmonic basis matrix in this crate is
/// built from world directions ([`crate::mrtrix_sh::sh2amp_cart`],
/// [`crate::descoteaux_sh::sh2amp_cart`], `sh_apsf::ApsfBasis`). A signed axis
/// permutation therefore permutes *rows* — which voxel a coefficient vector
/// belongs to — without rotating the coefficient vector itself. If ODX ever
/// stored voxel-frame SH, this function would need an accompanying Wigner-D
/// rotation of every row.
pub(crate) fn align_to_reference_grid(
    ref_dims: [u64; 3],
    ref_affine: &[[f64; 4]; 4],
    dims: [u64; 3],
    affine: &[[f64; 4]; 4],
    mask: &[u8],
    label: &str,
) -> Result<Vec<usize>> {
    if dims != ref_dims {
        return Err(OdxError::Argument(format!(
            "input '{label}' dimensions {dims:?} differ from the reference {ref_dims:?}; \
             resample it onto the reference grid first (`odx transform`)"
        )));
    }
    let compact = compact_index_map(mask, dims);
    if affine_close(affine, ref_affine, 1e-4) {
        return Ok(compact);
    }
    match same_lattice_voxel_remap(ref_dims, ref_affine, affine) {
        Some(remap) => Ok(remap.into_iter().map(|b_flat| compact[b_flat]).collect()),
        None => Err(OdxError::Argument(format!(
            "input '{label}' has the reference dimensions but an affine that is not related \
             to it by a signed axis permutation, so the voxel lattices do not coincide; \
             warp it onto the reference grid first (`odx transform`)"
        ))),
    }
}

/// Which array family [`shared_scalar_keys`] inspects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ArrayKind {
    Dpv,
    Dpf,
}

/// Scalar (`ncols == 1`) float array names present in **every** input, excluding
/// the QC class array. Optionally restricted to `want`.
pub(crate) fn shared_scalar_keys(
    datasets: &[OdxDataset],
    kind: ArrayKind,
    want: Option<&[String]>,
) -> Vec<String> {
    let mut common: Option<BTreeSet<String>> = None;
    for ds in datasets {
        let here: BTreeSet<String> = match kind {
            ArrayKind::Dpf => ds
                .iter_dpf()
                .filter(|(name, info)| {
                    *name != QC_CLASS_DPF_NAME && info.ncols == 1 && info.dtype.is_float()
                })
                .map(|(name, _)| name.to_string())
                .collect(),
            ArrayKind::Dpv => ds
                .iter_dpv()
                .filter(|(_, info)| info.ncols == 1 && info.dtype.is_float())
                .map(|(name, _)| name.to_string())
                .collect(),
        };
        common = Some(match common {
            None => here,
            Some(prev) => prev.intersection(&here).cloned().collect(),
        });
    }
    let mut keys: Vec<String> = common.unwrap_or_default().into_iter().collect();
    if let Some(want) = want {
        let wset: BTreeSet<&String> = want.iter().collect();
        keys.retain(|k| wset.contains(k));
    }
    keys
}

#[cfg(test)]
mod remap_tests {
    use super::*;

    #[test]
    fn identity_affine_not_needed_but_lattice_remap_is_identity() {
        // Same affine → remap is identity.
        let a = [[2.0, 0.0, 0.0, -10.0], [0.0, 2.0, 0.0, -20.0], [0.0, 0.0, 2.0, -5.0], [0.0, 0.0, 0.0, 1.0]];
        let r = same_lattice_voxel_remap([3, 4, 5], &a, &a).unwrap();
        assert!(r.iter().enumerate().all(|(i, &v)| i == v));
    }

    #[test]
    fn las_vs_ras_xy_flip_maps_correctly() {
        // A: native LAS-ish (−x,−y). B: RAS+ (+x,+y). Same physical lattice,
        // dims (3,4,5), 2 mm iso. A voxel (i,j,k) ↔ B voxel (nx-1-i, ny-1-j, k).
        let (nx, ny, nz) = (3usize, 4usize, 5usize);
        // A: world_x = 10 - 2 i ; world_y = 12 - 2 j ; world_z = -4 + 2 k
        let a = [[-2.0, 0.0, 0.0, 10.0], [0.0, -2.0, 0.0, 12.0], [0.0, 0.0, 2.0, -4.0], [0.0, 0.0, 0.0, 1.0]];
        // B RAS+: to hit the same world extent, world_x = 10 - 2*(nx-1) + 2 i' = 6 + 2 i'
        //         world_y = 12 - 2*(ny-1) + 2 j' = 6 + 2 j' ; world_z = -4 + 2 k'
        let b = [[2.0, 0.0, 0.0, 6.0], [0.0, 2.0, 0.0, 6.0], [0.0, 0.0, 2.0, -4.0], [0.0, 0.0, 0.0, 1.0]];
        let r = same_lattice_voxel_remap([nx as u64, ny as u64, nz as u64], &a, &b).unwrap();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let a_flat = i * ny * nz + j * nz + k;
                    let expect = (nx - 1 - i) * ny * nz + (ny - 1 - j) * nz + k;
                    assert_eq!(r[a_flat], expect, "voxel ({i},{j},{k})");
                }
            }
        }
    }

    #[test]
    fn different_resolution_is_rejected() {
        let a = [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let b = [[2.5, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        assert!(same_lattice_voxel_remap([3, 4, 5], &a, &b).is_none());
    }

    #[test]
    fn align_composes_remap_with_the_compact_map() {
        // Two 1×1×2 grids on one lattice, z-flipped relative to each other.
        // Reference keeps both voxels; the input masks only k=0 (in its own
        // ordering), which is the reference's k=1.
        let dims = [1u64, 1, 2];
        let r = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let b = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -2.0, 2.0], [0.0, 0.0, 0.0, 1.0]];
        let mask = vec![1u8, 0];
        let map = align_to_reference_grid(dims, &r, dims, &b, &mask, "b").unwrap();
        assert_eq!(map[0], usize::MAX, "reference k=0 is the input's masked-out k=1");
        assert_eq!(map[1], 0, "reference k=1 is the input's compact row 0");
    }

    #[test]
    fn align_rejects_a_shifted_grid() {
        let dims = [2u64, 2, 2];
        let r = [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let b = [[2.0, 0.0, 0.0, 0.7], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let mask = vec![1u8; 8];
        let err = align_to_reference_grid(dims, &r, dims, &b, &mask, "b").unwrap_err();
        assert!(err.to_string().contains("odx transform"), "got: {err}");
    }
}
