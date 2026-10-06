//! Shared helpers for integration tests. Kept dependency-free so fixtures
//! can be generated in-process (no files under `../test_data` are needed).
#![allow(dead_code)]

use std::path::Path;

/// Write a minimal NIfTI-1 float32 image (sform only, `n+1` magic).
///
/// `data_c` is laid out C-order over `dims` (last axis fastest); the file is
/// written in NIfTI's Fortran order (first axis fastest).
pub fn write_nifti1_f32(path: &Path, dims: &[usize], affine: [[f64; 4]; 4], data_c: &[f32]) {
    let total: usize = dims.iter().product();
    assert_eq!(data_c.len(), total, "data length must match dims");
    assert!(dims.len() <= 7);

    let mut hdr = vec![0u8; 352];
    hdr[0..4].copy_from_slice(&348i32.to_le_bytes());
    hdr[40..42].copy_from_slice(&(dims.len() as i16).to_le_bytes());
    for i in 0..7 {
        let d = dims.get(i).copied().unwrap_or(1) as i16;
        hdr[42 + 2 * i..44 + 2 * i].copy_from_slice(&d.to_le_bytes());
    }
    hdr[70..72].copy_from_slice(&16i16.to_le_bytes()); // NIFTI_TYPE_FLOAT32
    hdr[72..74].copy_from_slice(&32i16.to_le_bytes());
    let mut pixdim = [1.0f32; 8];
    for a in 0..3 {
        pixdim[a + 1] =
            (affine[0][a].powi(2) + affine[1][a].powi(2) + affine[2][a].powi(2)).sqrt() as f32;
    }
    for (i, v) in pixdim.iter().enumerate() {
        hdr[76 + 4 * i..80 + 4 * i].copy_from_slice(&v.to_le_bytes());
    }
    hdr[108..112].copy_from_slice(&352.0f32.to_le_bytes()); // vox_offset
    hdr[112..116].copy_from_slice(&1.0f32.to_le_bytes()); // scl_slope
    hdr[252..254].copy_from_slice(&0i16.to_le_bytes()); // qform_code
    hdr[254..256].copy_from_slice(&1i16.to_le_bytes()); // sform_code
    for (r, row) in affine.iter().take(3).enumerate() {
        for (c, &v) in row.iter().enumerate() {
            let off = 280 + 16 * r + 4 * c;
            hdr[off..off + 4].copy_from_slice(&(v as f32).to_le_bytes());
        }
    }
    hdr[344..348].copy_from_slice(b"n+1\0");

    let mut c_strides = vec![1usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        c_strides[i] = c_strides[i + 1] * dims[i + 1];
    }
    let mut idx = vec![0usize; dims.len()];
    let mut out = Vec::with_capacity(total * 4);
    for _ in 0..total {
        let c: usize = idx.iter().zip(&c_strides).map(|(i, s)| i * s).sum();
        out.extend_from_slice(&data_c[c].to_le_bytes());
        for a in 0..dims.len() {
            idx[a] += 1;
            if idx[a] < dims[a] {
                break;
            }
            idx[a] = 0;
        }
    }
    hdr.extend_from_slice(&out);
    std::fs::write(path, hdr).unwrap();
}

/// A uniform 9-volume gradient-deviation NIfTI on a `dims` grid storing the
/// row-major matrix `t` at every voxel.
pub fn write_uniform_graddev_nifti(
    path: &Path,
    dims: [usize; 3],
    affine: [[f64; 4]; 4],
    t: [[f64; 3]; 3],
) {
    let nvox = dims[0] * dims[1] * dims[2];
    let mut data = Vec::with_capacity(nvox * 9);
    for _ in 0..nvox {
        for row in &t {
            for &v in row {
                data.push(v as f32);
            }
        }
    }
    write_nifti1_f32(path, &[dims[0], dims[1], dims[2], 9], affine, &data);
}
