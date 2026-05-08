//! Scatter ODX flat (compact-by-mask) arrays back into the full 3D grid.
//!
//! ODX stores `NB_VOXELS` rows where `NB_VOXELS == mask.sum()` (the number
//! of in-mask voxels), iterating in C order (i-slowest, k-fastest). For
//! interop with PAM5 / dipy / NIfTI, we need to scatter those rows back to
//! `(X, Y, Z, …)` arrays with zeros outside the mask.
//!
//! These routines are the canonical implementation; both `save_pam5` and
//! the Python wrapper depend on them. The mask iteration order matches
//! [`OdxDataset::compact_to_ijk`].

use ndarray::{Array3, Array4, Array5};

use crate::error::Result;
use crate::odx_file::OdxDataset;

/// Maximum peak count across all voxels — the `N_max` dimension of densified
/// peak arrays.
pub fn max_peaks_per_voxel(odx: &OdxDataset) -> usize {
    (0..odx.nb_voxels())
        .map(|i| odx.peaks_per_voxel(i))
        .max()
        .unwrap_or(0)
}

/// Densify peak directions into `(X, Y, Z, N_max, 3)` float32. Voxels with
/// fewer than `N_max` peaks are zero-padded.
pub fn densify_directions(odx: &OdxDataset) -> Array5<f32> {
    let dims = odx.header().dimensions;
    let (xx, yy, zz) = (dims[0] as usize, dims[1] as usize, dims[2] as usize);
    let n_max = max_peaks_per_voxel(odx);
    let mut out = Array5::<f32>::zeros((xx, yy, zz, n_max, 3));
    if n_max == 0 {
        return out;
    }

    let ijk = odx.compact_to_ijk();
    let offsets = odx.offsets();
    let directions = odx.directions();

    for (row, [i, j, k]) in ijk.iter().enumerate().map(|(r, ijk)| (r, *ijk)) {
        let (i, j, k) = (i as usize, j as usize, k as usize);
        let start = offsets[row] as usize;
        let count = (offsets[row + 1] - offsets[row]) as usize;
        for p in 0..count {
            if p >= n_max {
                break;
            }
            let dir = directions[start + p];
            out[[i, j, k, p, 0]] = dir[0];
            out[[i, j, k, p, 1]] = dir[1];
            out[[i, j, k, p, 2]] = dir[2];
        }
    }
    out
}

/// Densify a per-fixel scalar (e.g. `amplitude`, `qa`) into `(X, Y, Z, N_max)`.
pub fn densify_scalar_dpf(odx: &OdxDataset, name: &str) -> Result<Array4<f32>> {
    let dims = odx.header().dimensions;
    let (xx, yy, zz) = (dims[0] as usize, dims[1] as usize, dims[2] as usize);
    let n_max = max_peaks_per_voxel(odx);
    let mut out = Array4::<f32>::zeros((xx, yy, zz, n_max));
    if n_max == 0 {
        return Ok(out);
    }

    let values = odx.scalar_dpf_f32(name)?;
    let ijk = odx.compact_to_ijk();
    let offsets = odx.offsets();

    for (row, [i, j, k]) in ijk.iter().enumerate().map(|(r, ijk)| (r, *ijk)) {
        let (i, j, k) = (i as usize, j as usize, k as usize);
        let start = offsets[row] as usize;
        let count = (offsets[row + 1] - offsets[row]) as usize;
        for p in 0..count {
            if p >= n_max {
                break;
            }
            out[[i, j, k, p]] = values[start + p];
        }
    }
    Ok(out)
}

/// Densify a per-voxel scalar (e.g. `gfa`) into `(X, Y, Z)`.
pub fn densify_scalar_dpv(odx: &OdxDataset, name: &str) -> Result<Array3<f32>> {
    let dims = odx.header().dimensions;
    let (xx, yy, zz) = (dims[0] as usize, dims[1] as usize, dims[2] as usize);
    let mut out = Array3::<f32>::zeros((xx, yy, zz));

    let values = odx.scalar_dpv_f32(name)?;
    let ijk = odx.compact_to_ijk();

    for (row, [i, j, k]) in ijk.iter().enumerate().map(|(r, ijk)| (r, *ijk)) {
        let (i, j, k) = (i as usize, j as usize, k as usize);
        out[[i, j, k]] = values[row];
    }
    Ok(out)
}

/// Densify spherical-harmonic coefficients into `(X, Y, Z, K)`.
/// Uses [`crate::data_array::DataArray::to_f32_vec`] to dequantize quantized
/// uint8 storage on the way through, so callers always see float32.
pub fn densify_sh(odx: &OdxDataset, name: &str) -> Result<Array4<f32>> {
    densify_2d_array(odx, ArrayKind::Sh, name)
}

/// Densify ODF amplitudes into `(X, Y, Z, M)`.
pub fn densify_odf(odx: &OdxDataset, name: &str) -> Result<Array4<f32>> {
    densify_2d_array(odx, ArrayKind::Odf, name)
}

#[derive(Copy, Clone)]
enum ArrayKind {
    Sh,
    Odf,
}

fn densify_2d_array(
    odx: &OdxDataset,
    kind: ArrayKind,
    name: &str,
) -> Result<Array4<f32>> {
    let dims = odx.header().dimensions;
    let (xx, yy, zz) = (dims[0] as usize, dims[1] as usize, dims[2] as usize);

    // Pull a flat (NB_VOXELS, ncols) f32 view via DataArray::to_f32_vec, which
    // handles dtype conversion (e.g. f16/quantized u8 → f32).
    let (values, ncols) = match kind {
        ArrayKind::Sh => {
            let arr = odx
                .sh_arrays()
                .get(name)
                .ok_or_else(|| crate::error::OdxError::Argument(format!("no SH array '{name}'")))?;
            (arr.to_f32_vec()?, arr.ncols())
        }
        ArrayKind::Odf => {
            let arr = odx
                .odf_arrays()
                .get(name)
                .ok_or_else(|| crate::error::OdxError::Argument(format!("no ODF array '{name}'")))?;
            (arr.to_f32_vec()?, arr.ncols())
        }
    };

    let mut out = Array4::<f32>::zeros((xx, yy, zz, ncols));
    let ijk = odx.compact_to_ijk();
    for (row, [i, j, k]) in ijk.iter().enumerate().map(|(r, ijk)| (r, *ijk)) {
        let (i, j, k) = (i as usize, j as usize, k as usize);
        let start = row * ncols;
        for c in 0..ncols {
            out[[i, j, k, c]] = values[start + c];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::header::Header;
    use crate::mmap_backing::{vec_to_bytes, MmapBacking};
    use crate::odx_file::{OdxDataset, OdxParts};
    use std::collections::HashMap;

    fn make_simple_dataset() -> OdxDataset {
        // 2x1x2 grid, mask covering 3 of 4 voxels in i-slowest order.
        // mask layout (flat C order, dims [2,1,2]):
        //   (0,0,0)=1  (0,0,1)=0  (1,0,0)=1  (1,0,1)=1
        let mask = vec![1u8, 0, 1, 1];
        // 2 peaks for voxel 0, 1 peak for voxel 1, 0 peaks for voxel 2
        let offsets: Vec<u32> = vec![0, 2, 3, 3];
        let directions: Vec<[f32; 3]> =
            vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let directions_bytes: Vec<u8> = vec_to_bytes(directions);

        let header = Header {
            voxel_to_rasmm: Header::identity_affine(),
            dimensions: [2, 1, 2],
            nb_voxels: 3,
            nb_peaks: 3,
            nb_sphere_vertices: None,
            nb_sphere_faces: None,
            sh_order: None,
            sh_basis: None,
            sh_full_basis: None,
            sh_legacy: None,
            canonical_dense_representation: None,
            sphere_id: None,
            odf_sample_domain: None,
            array_quantization: HashMap::new(),
            pam_metadata: None,
            extra: HashMap::new(),
        };

        let parts = OdxParts {
            header,
            mask_backing: MmapBacking::Owned(mask),
            offsets_backing: MmapBacking::Owned(vec_to_bytes(offsets)),
            directions_backing: MmapBacking::Owned(directions_bytes),
            sphere_vertices: None,
            sphere_faces: None,
            odf: HashMap::new(),
            sh: HashMap::new(),
            dpv: HashMap::new(),
            dpf: HashMap::new(),
            groups: HashMap::new(),
            dpg: HashMap::new(),
            tempdir: None,
        };
        OdxDataset::from_parts(parts)
    }

    #[test]
    fn dense_directions_match_compact() {
        let odx = make_simple_dataset();
        let dense = densify_directions(&odx);
        // Shape: (2, 1, 2, 2, 3) — n_max = 2.
        assert_eq!(dense.shape(), &[2, 1, 2, 2, 3]);
        // Voxel (0,0,0) row 0 → first ODX peak [1,0,0].
        assert_eq!(dense[[0, 0, 0, 0, 0]], 1.0);
        assert_eq!(dense[[0, 0, 0, 0, 1]], 0.0);
        // Voxel (0,0,0) row 1 → second ODX peak [0,1,0].
        assert_eq!(dense[[0, 0, 0, 1, 1]], 1.0);
        // Voxel (0,0,1) is unmasked → all zeros.
        assert_eq!(dense[[0, 0, 1, 0, 0]], 0.0);
        // Voxel (1,0,0) row 0 → third ODX peak [0,0,1]. Second slot zero.
        assert_eq!(dense[[1, 0, 0, 0, 2]], 1.0);
        assert_eq!(dense[[1, 0, 0, 1, 2]], 0.0);
        // Voxel (1,0,1) had 0 peaks → all zeros.
        assert_eq!(dense[[1, 0, 1, 0, 0]], 0.0);
    }

    #[test]
    fn dense_dpv_scatters_per_voxel() {
        let mut odx = make_simple_dataset();
        // Attach a per-voxel scalar via the public insert path? There isn't
        // one for DPV — build via from_parts so we have to re-create.
        let _ = &mut odx;
        let mut parts = make_simple_dataset_parts();
        parts.dpv.insert(
            "gfa".into(),
            crate::data_array::DataArray::owned_bytes(
                vec_to_bytes(vec![0.1_f32, 0.2, 0.3]),
                1,
                DType::Float32,
            ),
        );
        let odx = OdxDataset::from_parts(parts);
        let dense = densify_scalar_dpv(&odx, "gfa").unwrap();
        assert_eq!(dense.shape(), &[2, 1, 2]);
        assert!((dense[[0, 0, 0]] - 0.1).abs() < 1e-6);
        assert!((dense[[0, 0, 1]] - 0.0).abs() < 1e-6);
        assert!((dense[[1, 0, 0]] - 0.2).abs() < 1e-6);
        assert!((dense[[1, 0, 1]] - 0.3).abs() < 1e-6);
    }

    fn make_simple_dataset_parts() -> OdxParts {
        let mask = vec![1u8, 0, 1, 1];
        let offsets: Vec<u32> = vec![0, 2, 3, 3];
        let directions: Vec<[f32; 3]> =
            vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let header = Header {
            voxel_to_rasmm: Header::identity_affine(),
            dimensions: [2, 1, 2],
            nb_voxels: 3,
            nb_peaks: 3,
            nb_sphere_vertices: None,
            nb_sphere_faces: None,
            sh_order: None,
            sh_basis: None,
            sh_full_basis: None,
            sh_legacy: None,
            canonical_dense_representation: None,
            sphere_id: None,
            odf_sample_domain: None,
            array_quantization: HashMap::new(),
            pam_metadata: None,
            extra: HashMap::new(),
        };
        OdxParts {
            header,
            mask_backing: MmapBacking::Owned(mask),
            offsets_backing: MmapBacking::Owned(vec_to_bytes(offsets)),
            directions_backing: MmapBacking::Owned(vec_to_bytes(directions)),
            sphere_vertices: None,
            sphere_faces: None,
            odf: HashMap::new(),
            sh: HashMap::new(),
            dpv: HashMap::new(),
            dpf: HashMap::new(),
            groups: HashMap::new(),
            dpg: HashMap::new(),
            tempdir: None,
        }
    }
}
