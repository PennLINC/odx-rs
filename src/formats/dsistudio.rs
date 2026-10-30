//! DSI Studio (.fib.gz / .fz) format adapter.
//!
//! ---------------------------------------------------------------------------
//! Third-party attribution: nibabel
//!
//! The voxel-grid reorientation in [`build_dsistudio_export_geometry`] is
//! computed by calling through to [`crate::nifti_canon`], whose
//! `nibabel_io_orientation` and `nibabel_ornt_transform` helpers are ports of
//! routines from the `nibabel.orientations` module of nibabel
//! (<https://nipy.org/nibabel/>), copyright (c) 2009-2024 the nibabel
//! developers. Those portions are made available under the MIT License; see
//! `odx-rs/LICENSE-NIBABEL` for the full notice.

use std::collections::HashMap;
use std::path::Path;

use nalgebra::{Matrix3, Vector3};

use crate::data_array::DataArray;
use crate::dtype::DType;
use crate::error::{OdxError, Result};
use crate::formats::dsistudio_odf8;
use crate::formats::mat4::{
    self, float_record, int16_record, uint8_record, MatCatalog, MatRecord, OwnedMatRecord,
};
use crate::header::CanonicalDenseRepresentation;
use crate::mmap_backing::{vec_to_bytes, MmapBacking};
use crate::nifti_canon::{affine_column_norms, nibabel_io_orientation, nibabel_ornt_transform};
use crate::odx_file::{LoadOptions, OdxDataset, OdxParts};

/// Header-extra flag telling the writer to emit exact float32 `dir{p}`
/// records next to the sphere-quantized `index{p}` ones. DSI Studio's tracker
/// prefers `dir{p}` whenever present (`fib_data.cpp`, `get_fib`), in both
/// `.fib.gz` and `.fz` containers.
pub const WRITE_DIR_RECORDS_KEY: &str = "_ODX_DSISTUDIO_WRITE_DIR_RECORDS";

/// DSI Studio stores `odf_vertices`, `dir{p}` (and so the directions behind
/// `index{p}`) in its voxel frame: the axes of its LPS-reordered voxel grid,
/// not world LPS. The column-normalized linear part of `trans` maps that
/// frame to world RAS. For a non-oblique image this is `diag(-1, -1, 1)`; on
/// an oblique one it also carries the acquisition rotation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct VoxelFrame {
    to_ras: Matrix3<f64>,
    from_ras: Matrix3<f64>,
}

impl VoxelFrame {
    pub(crate) fn from_affine(aff: &[[f64; 4]; 4]) -> Result<Self> {
        let mut to_ras = Matrix3::zeros();
        for col in 0..3 {
            let norm = (0..3).map(|row| aff[row][col].powi(2)).sum::<f64>().sqrt();
            if norm <= 0.0 {
                return Err(OdxError::Format(format!(
                    "DSI Studio affine has a zero-length voxel axis {col}"
                )));
            }
            for row in 0..3 {
                to_ras[(row, col)] = aff[row][col] / norm;
            }
        }
        let from_ras = to_ras
            .try_inverse()
            .ok_or_else(|| OdxError::Format("DSI Studio affine is singular".into()))?;
        Ok(Self { to_ras, from_ras })
    }

    /// DSI Studio voxel-frame direction → world RAS.
    pub(crate) fn to_ras(&self, dir: [f32; 3]) -> [f32; 3] {
        map_direction(&self.to_ras, dir)
    }

    /// World RAS direction → DSI Studio voxel frame.
    pub(crate) fn from_ras(&self, dir: [f32; 3]) -> [f32; 3] {
        map_direction(&self.from_ras, dir)
    }
}

/// `mat · dir`, rescaled to `dir`'s length so that the identity and axis
/// flips stay bit-exact and a sheared affine does not change vector lengths.
fn map_direction(mat: &Matrix3<f64>, dir: [f32; 3]) -> [f32; 3] {
    let v = Vector3::new(f64::from(dir[0]), f64::from(dir[1]), f64::from(dir[2]));
    let out = mat * v;
    let out_norm = out.norm();
    if out_norm == 0.0 {
        return [0.0; 3];
    }
    let out = out * (v.norm() / out_norm);
    [out[0] as f32, out[1] as f32, out[2] as f32]
}

/// The built-in `odf8` sphere as DSI Studio stores it, in its voxel frame.
pub(crate) fn builtin_voxel_frame_vertices() -> Vec<[f32; 3]> {
    dsistudio_odf8::full_vertices_ras()
        .iter()
        // `full_vertices_ras` is the asset with x and y negated; undo that.
        .map(|v| [-v[0], -v[1], v[2]])
        .collect()
}

pub fn load_fibgz(path: &Path, affine: Option<[[f64; 4]; 4]>) -> Result<OdxDataset> {
    load_dsistudio_mat(path, affine, &LoadOptions::default())
}

pub fn load_fz(path: &Path, affine: Option<[[f64; 4]; 4]>) -> Result<OdxDataset> {
    load_dsistudio_mat(path, affine, &LoadOptions::default())
}

/// `load_fibgz` / `load_fz` reading only what `options` needs. With
/// `skip_odf`, the dense `odfN` records are decompressed and discarded
/// without being stored.
pub fn load_dsistudio_with(
    path: &Path,
    affine: Option<[[f64; 4]; 4]>,
    options: &LoadOptions,
) -> Result<OdxDataset> {
    load_dsistudio_mat(path, affine, options)
}

/// DSI Studio stores dense ODFs as `odf0`, `odf1`, ... (not `odf_vertices`
/// or `odf_faces`).
fn is_odf_chunk(name: &str) -> bool {
    name.strip_prefix("odf")
        .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

pub fn save_fibgz(odx: &OdxDataset, path: &Path) -> Result<()> {
    let records = build_dsistudio_records(odx, false)?;
    mat4::write_mat4_gz(path, &records)
}

pub fn save_fz(odx: &OdxDataset, path: &Path) -> Result<()> {
    let records = build_dsistudio_records(odx, true)?;
    mat4::write_mat4_gz(path, &records)
}

fn get_required<'a>(mat: &'a MatCatalog, key: &str) -> Result<MatRecord<'a>> {
    mat.get(key)
        .ok_or_else(|| OdxError::Format(format!("missing required key '{key}' in dsistudio file")))
}

fn load_dsistudio_mat(
    path: &Path,
    affine: Option<[[f64; 4]; 4]>,
    options: &LoadOptions,
) -> Result<OdxDataset> {
    let skip_odf = options.skip_odf;
    let mat = mat4::read_mat4_gz_filtered(path, |name| !(skip_odf && is_odf_chunk(name)))?;

    let dim_arr = get_required(&mat, "dimension")?;
    let dim_vals = dim_arr.as_i32_vec();
    if dim_vals.len() < 3 {
        return Err(OdxError::Format("dimension must have 3 elements".into()));
    }
    let dimensions = [dim_vals[0] as u64, dim_vals[1] as u64, dim_vals[2] as u64];
    let nvoxels_total = dimensions[0] as usize * dimensions[1] as usize * dimensions[2] as usize;

    let vox_arr = get_required(&mat, "voxel_size")?;
    let _vox = vox_arr.as_f32_vec();

    let voxel_to_rasmm = if let Some(trans) = mat.get("trans") {
        dsistudio_trans_to_nifti_affine(trans.as_f32_vec())
    } else if let Some(a) = affine {
        // DSI Studio reorders the voxels of the DWI it reconstructs to LPS, so
        // the DWI's own affine must be reordered the same way to match.
        reference_affine_to_dsistudio_grid(
            a,
            [
                dimensions[0] as usize,
                dimensions[1] as usize,
                dimensions[2] as usize,
            ],
        )
    } else {
        return Err(OdxError::Format(
            "DSI Studio file has no spatial affine ('trans' field). \
             Convert it first with: odx convert --reference-affine <dwi.nii.gz> <file> \
             <output.odx>, passing the DWI NIfTI it was reconstructed from; its affine is \
             reordered to DSI Studio's LPS voxel order automatically"
                .into(),
        ));
    };
    let frame = VoxelFrame::from_affine(&voxel_to_rasmm)?;

    let mask_source = if let Some(mask) = mat.get("mask") {
        if let Some(values) = mask.as_u8_slice() {
            values.to_vec()
        } else {
            mask.as_f32_vec()
                .into_iter()
                .map(|v| u8::from(v > 0.0))
                .collect()
        }
    } else {
        let source = mat
            .get("fa0")
            .or_else(|| mat.get("image0"))
            .or_else(|| mat.get("image"))
            .ok_or_else(|| {
                OdxError::Format("cannot build mask: no mask/fa0/image0/image".into())
            })?;
        let values = source.as_f32_vec();
        if values.len() != nvoxels_total {
            return Err(OdxError::Format(format!(
                "mask source length {} != volume size {}",
                values.len(),
                nvoxels_total
            )));
        }
        values.into_iter().map(|v| u8::from(v > 0.0)).collect()
    };

    let (di, dj, dk) = (
        dimensions[0] as usize,
        dimensions[1] as usize,
        dimensions[2] as usize,
    );
    let mut fortran_to_c = vec![0usize; nvoxels_total];
    for i in 0..di {
        for j in 0..dj {
            for k in 0..dk {
                let f_idx = i + j * di + k * di * dj;
                let c_idx = i * dj * dk + j * dk + k;
                fortran_to_c[f_idx] = c_idx;
            }
        }
    }
    let mut c_to_f = vec![0usize; nvoxels_total];
    for f_idx in 0..nvoxels_total {
        c_to_f[fortran_to_c[f_idx]] = f_idx;
    }

    let mut mask_c = vec![0u8; nvoxels_total];
    for f_idx in 0..nvoxels_total {
        if mask_source[f_idx] != 0 {
            mask_c[fortran_to_c[f_idx]] = 1;
        }
    }
    let nb_masked = mask_c.iter().filter(|&&v| v != 0).count();
    let mut masked_f_indices = Vec::with_capacity(nb_masked);
    for c_idx in 0..nvoxels_total {
        if mask_c[c_idx] != 0 {
            masked_f_indices.push(c_to_f[c_idx]);
        }
    }
    let mut sparse_row_by_f = vec![usize::MAX; nvoxels_total];
    let mut row = 0usize;
    for f_idx in 0..nvoxels_total {
        if mask_source[f_idx] != 0 {
            sparse_row_by_f[f_idx] = row;
            row += 1;
        }
    }
    let sparse_count = row;

    // DSI Studio stores sphere geometry in its voxel frame. ODX keeps
    // directions and spheres in world RAS, so the sphere is rotated once here
    // and `index{p}` lookups then land on RAS vertices.
    let sphere_vertices = if let Some(record) = mat.get("odf_vertices") {
        let verts_flat = record.as_f32_vec();
        let mut out = Vec::with_capacity(record.ncols());
        for i in 0..record.ncols() {
            out.push(frame.to_ras([
                verts_flat[i * record.mrows()],
                verts_flat[i * record.mrows() + 1],
                verts_flat[i * record.mrows() + 2],
            ]));
        }
        Some(out)
    } else {
        Some(
            builtin_voxel_frame_vertices()
                .into_iter()
                .map(|v| frame.to_ras(v))
                .collect(),
        )
    };

    let sphere_faces = if let Some(record) = mat.get("odf_faces") {
        let flat = record.as_i32_vec();
        let mut out = Vec::with_capacity(record.ncols());
        for i in 0..record.ncols() {
            out.push([
                flat[i * record.mrows()] as u32,
                flat[i * record.mrows() + 1] as u32,
                flat[i * record.mrows() + 2] as u32,
            ]);
        }
        Some(out)
    } else {
        Some(dsistudio_odf8::faces().to_vec())
    };

    let max_peaks = count_peak_fields(&mat);
    let mut fa_cache = Vec::with_capacity(max_peaks);
    let mut index_cache: Vec<Option<Vec<i32>>> = Vec::with_capacity(max_peaks);
    // DSI Studio's tracker takes exact float `dir{p}` records over the
    // sphere-quantized `index{p}` whenever both exist; so does this loader.
    let mut dir_cache: Vec<Option<Vec<f32>>> = Vec::with_capacity(max_peaks);
    for p in 0..max_peaks {
        fa_cache.push(load_volume_f32(
            get_required(&mat, &format!("fa{p}"))?,
            nvoxels_total,
            &sparse_row_by_f,
            sparse_count,
        ));
        dir_cache.push(mat.get(&format!("dir{p}")).map(|record| {
            load_volume_vec3(record, nvoxels_total, &sparse_row_by_f, sparse_count)
        }));
        index_cache.push(mat.get(&format!("index{p}")).map(|record| {
            load_volume_i32(record, nvoxels_total, &sparse_row_by_f, sparse_count)
        }));
    }

    let mut offsets: Vec<u32> = Vec::with_capacity(nb_masked + 1);
    let mut directions: Vec<[f32; 3]> = Vec::new();
    let mut amplitudes: Vec<f32> = Vec::new();
    offsets.push(0);

    for &f_idx in &masked_f_indices {
        let mut n_peaks_this_voxel = 0u32;
        for p in 0..max_peaks {
            let fa_val = fa_cache[p][f_idx];
            if fa_val <= 0.0 {
                break;
            }
            let direction = if let Some(dirs) = dir_cache[p].as_ref() {
                let d = &dirs[f_idx * 3..f_idx * 3 + 3];
                let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                // DSI Studio stores directions in its voxel frame; ODX keeps RAS.
                (n > 0.0).then(|| frame.to_ras([d[0] / n, d[1] / n, d[2] / n]))
            } else {
                match (index_cache[p].as_ref(), sphere_vertices.as_ref()) {
                    (Some(indices), Some(verts)) => {
                        let dir_idx = indices[f_idx] as usize;
                        (dir_idx < verts.len()).then(|| verts[dir_idx])
                    }
                    _ => None,
                }
            };
            if let Some(direction) = direction {
                directions.push(direction);
                amplitudes.push(fa_val);
                n_peaks_this_voxel += 1;
            }
        }
        offsets.push(offsets.last().unwrap() + n_peaks_this_voxel);
    }

    let nb_peaks = directions.len();
    let odf_chunks = collect_odf_chunks(&mat);
    let odf_data = if !odf_chunks.is_empty() {
        let n_odf_dirs = odf_chunks[0].mrows();
        let total_sparse_cols: usize = odf_chunks.iter().map(|r| r.ncols()).sum();
        let mut all_odf = vec![0.0f32; n_odf_dirs * total_sparse_cols];
        let mut cursor = 0usize;
        for chunk in &odf_chunks {
            let data = chunk.as_f32_vec();
            let len = data.len();
            all_odf[cursor..cursor + len].copy_from_slice(&data);
            cursor += len;
        }
        let mut odf_out = vec![0.0f32; nb_masked * n_odf_dirs];
        for (odx_row, &f_idx) in masked_f_indices.iter().enumerate() {
            let odf_row = sparse_row_by_f[f_idx];
            if odf_row == usize::MAX {
                continue;
            }
            for d in 0..n_odf_dirs {
                odf_out[odx_row * n_odf_dirs + d] = all_odf[d + odf_row * n_odf_dirs];
            }
        }
        Some((odf_out, n_odf_dirs))
    } else {
        None
    };

    let mut dpv: HashMap<String, DataArray> = HashMap::new();
    let scalar_keys = [
        "dti_fa", "gfa", "md", "ad", "rd", "iso", "rdi", "ha", "rd1", "rd2",
    ];
    for &key in &scalar_keys {
        if let Some(record) = mat.get(key) {
            let vals = load_volume_f32(record, nvoxels_total, &sparse_row_by_f, sparse_count);
            let mut masked_vals = Vec::with_capacity(nb_masked);
            for &f_idx in &masked_f_indices {
                masked_vals.push(vals[f_idx]);
            }
            dpv.insert(
                key.to_string(),
                DataArray::owned_bytes(vec_to_bytes(masked_vals), 1, DType::Float32),
            );
        }
    }

    let mut dpf: HashMap<String, DataArray> = HashMap::new();
    dpf.insert(
        "amplitude".to_string(),
        DataArray::owned_bytes(vec_to_bytes(amplitudes), 1, DType::Float32),
    );

    for prefix in collect_peak_scalar_prefixes(&mat, max_peaks, nvoxels_total, sparse_count) {
        let mut all_vals = Vec::new();
        let mut caches = Vec::new();
        for p in 0..max_peaks {
            if let Some(record) = mat.get(&format!("{prefix}{p}")) {
                caches.push(Some(load_volume_f32(
                    record,
                    nvoxels_total,
                    &sparse_row_by_f,
                    sparse_count,
                )));
            } else {
                caches.push(None);
            }
        }
        for &f_idx in &masked_f_indices {
            for p in 0..max_peaks {
                if fa_cache[p][f_idx] <= 0.0 {
                    break;
                }
                if let Some(values) = &caches[p] {
                    all_vals.push(values[f_idx]);
                }
            }
        }
        if all_vals.len() == nb_peaks {
            dpf.insert(
                prefix.to_string(),
                DataArray::owned_bytes(vec_to_bytes(all_vals), 1, DType::Float32),
            );
        }
    }

    let mut odf = HashMap::new();
    if let Some((odf_vals, ncols)) = odf_data {
        odf.insert(
            "amplitudes".to_string(),
            DataArray::owned_bytes(vec_to_bytes(odf_vals), ncols, DType::Float32),
        );
    }

    let mut extra = HashMap::new();
    if let Some(z0) = mat.get("z0").and_then(|r| r.scalar_f32().ok()) {
        extra.insert(
            "Z0".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(z0 as f64).unwrap_or(serde_json::Number::from(0)),
            ),
        );
    }
    let header = crate::header::Header {
        voxel_to_rasmm,
        dimensions,
        nb_voxels: nb_masked as u64,
        nb_peaks: nb_peaks as u64,
        nb_sphere_vertices: sphere_vertices.as_ref().map(|v| v.len() as u64),
        nb_sphere_faces: sphere_faces.as_ref().map(|f| f.len() as u64),
        sh_order: None,
        sh_basis: None,
        sh_full_basis: None,
        sh_legacy: None,
        canonical_dense_representation: if odf.is_empty() {
            None
        } else {
            Some(CanonicalDenseRepresentation::Odf)
        },
        sphere_id: sphere_vertices
            .as_ref()
            .zip(sphere_faces.as_ref())
            .and_then(|(verts, faces)| {
                dsistudio_odf8::matches_builtin(verts, faces).then_some("dsistudio_odf8".into())
            }),
        // DSI stores dense `odfN` payloads on the first hemisphere of the full
        // sphere, even when `odf_vertices` contains the entire tessellation.
        odf_sample_domain: (!odf.is_empty()).then_some("hemisphere".into()),
        array_quantization: HashMap::new(),
        pam_metadata: None,
        extra,
    };

    Ok(OdxDataset::from_parts(OdxParts {
        header,
        mask_backing: MmapBacking::Owned(mask_c),
        offsets_backing: MmapBacking::Owned(vec_to_bytes(offsets)),
        directions_backing: MmapBacking::Owned(vec_to_bytes(directions)),
        sphere_vertices: sphere_vertices.map(|v| MmapBacking::Owned(vec_to_bytes(v))),
        sphere_faces: sphere_faces.map(|f| MmapBacking::Owned(vec_to_bytes(f))),
        odf,
        sh: HashMap::new(),
        dpv,
        dpf,
        groups: HashMap::new(),
        dpg: HashMap::new(),
        tempdir: None,
    }))
}

fn load_volume_f32(
    record: MatRecord<'_>,
    nvoxels_total: usize,
    sparse_row_by_f: &[usize],
    sparse_count: usize,
) -> Vec<f32> {
    let values = record.as_f32_vec();
    let is_sparse = sparse_count > 0
        && values.len() == sparse_count
        && (record.ncols() == sparse_count || record.mrows() == sparse_count);
    if is_sparse {
        let mut out = vec![0.0f32; nvoxels_total];
        for f_idx in 0..nvoxels_total {
            let row = sparse_row_by_f[f_idx];
            if row != usize::MAX && row < values.len() {
                out[f_idx] = values[row];
            }
        }
        out
    } else if values.len() == nvoxels_total {
        values
    } else {
        vec![0.0f32; nvoxels_total]
    }
}

/// Per-voxel 3-vectors (`dir{p}`): DSI Studio interleaves xyz per voxel in
/// Fortran voxel order, either dense (`3 × nvoxels`) or masked
/// (`3 × sparse_count`, expanded here through the mask).
fn load_volume_vec3(
    record: MatRecord<'_>,
    nvoxels_total: usize,
    sparse_row_by_f: &[usize],
    sparse_count: usize,
) -> Vec<f32> {
    let values = record.as_f32_vec();
    if values.len() == nvoxels_total * 3 {
        return values;
    }
    let mut out = vec![0.0f32; nvoxels_total * 3];
    if sparse_count > 0 && values.len() == sparse_count * 3 {
        for f_idx in 0..nvoxels_total {
            let row = sparse_row_by_f[f_idx];
            if row != usize::MAX {
                out[f_idx * 3..f_idx * 3 + 3].copy_from_slice(&values[row * 3..row * 3 + 3]);
            }
        }
    }
    out
}

fn load_volume_i32(
    record: MatRecord<'_>,
    nvoxels_total: usize,
    sparse_row_by_f: &[usize],
    sparse_count: usize,
) -> Vec<i32> {
    let values = record.as_i32_vec();
    let is_sparse = sparse_count > 0
        && values.len() == sparse_count
        && (record.ncols() == sparse_count || record.mrows() == sparse_count);
    if is_sparse {
        let mut out = vec![0i32; nvoxels_total];
        for f_idx in 0..nvoxels_total {
            let row = sparse_row_by_f[f_idx];
            if row != usize::MAX && row < values.len() {
                out[f_idx] = values[row];
            }
        }
        out
    } else if values.len() == nvoxels_total {
        values
    } else {
        vec![0i32; nvoxels_total]
    }
}

fn collect_odf_chunks(mat: &MatCatalog) -> Vec<MatRecord<'_>> {
    let mut chunks = Vec::new();
    let mut n = 0usize;
    while let Some(arr) = mat.get(&format!("odf{n}")) {
        chunks.push(arr);
        n += 1;
    }
    chunks
}

fn count_peak_fields(mat: &MatCatalog) -> usize {
    let mut n = 0usize;
    while mat.has(&format!("fa{n}"))
        && (mat.has(&format!("index{n}")) || mat.has(&format!("dir{n}")))
    {
        n += 1;
    }
    n
}

fn collect_peak_scalar_prefixes(
    mat: &MatCatalog,
    max_peaks: usize,
    nvoxels_total: usize,
    sparse_count: usize,
) -> Vec<String> {
    let mut prefixes = Vec::new();
    for record in mat.iter() {
        let Some((prefix, peak_idx)) = split_trailing_index(record.name()) else {
            continue;
        };
        if peak_idx >= max_peaks {
            continue;
        }
        if matches!(prefix, "fa" | "index" | "odf" | "dir") {
            continue;
        }
        if !mat.has(&format!("{prefix}0")) {
            continue;
        }
        if record.ncols() != nvoxels_total && record.ncols() != sparse_count {
            continue;
        }
        if prefixes.iter().any(|known| known == prefix) {
            continue;
        }
        prefixes.push(prefix.to_string());
    }
    prefixes.sort();
    prefixes
}

fn split_trailing_index(name: &str) -> Option<(&str, usize)> {
    let suffix_start = name
        .rfind(|ch: char| !ch.is_ascii_digit())
        .map(|idx| idx + 1)
        .unwrap_or(0);
    if suffix_start == name.len() {
        return None;
    }
    let prefix = &name[..suffix_start];
    let suffix = &name[suffix_start..];
    if prefix.is_empty() || !suffix.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    Some((prefix, suffix.parse().ok()?))
}

// DSI Studio `trans` stores the full voxel→RAS+ world affine directly.
// The voxel index ordering in DSI Studio files is still reoriented into LPS+
// array order, but the affine output space itself is RAS+.
fn dsistudio_trans_to_nifti_affine(flat: Vec<f32>) -> [[f64; 4]; 4] {
    let mut ras = [[0.0f64; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            let idx = r * 4 + c;
            if idx < flat.len() {
                ras[r][c] = flat[idx] as f64;
            }
        }
    }
    ras
}

// DSI Studio `trans` stores the full voxel→RAS+ world affine directly.
fn nifti_affine_to_dsistudio_trans(aff: [[f64; 4]; 4]) -> Vec<f32> {
    aff.into_iter()
        .flat_map(|row| row.into_iter().map(|v| v as f32))
        .collect()
}

fn build_reoriented_dsistudio_c_to_f(
    src_dims: [usize; 3],
    dst_dims: [usize; 3],
    src_axis_for_dst: [usize; 3],
    flip_dst_axis: [bool; 3],
) -> Vec<usize> {
    let [si, sj, sk] = src_dims;
    let [di, dj, dk] = dst_dims;
    let mut c_to_f = vec![0usize; si * sj * sk];
    for i in 0..si {
        for j in 0..sj {
            for k in 0..sk {
                let src = [i, j, k];
                let c_idx = i * sj * sk + j * sk + k;
                let mut dst = [
                    src[src_axis_for_dst[0]],
                    src[src_axis_for_dst[1]],
                    src[src_axis_for_dst[2]],
                ];
                for axis in 0..3 {
                    if flip_dst_axis[axis] {
                        dst[axis] = dst_dims[axis] - 1 - dst[axis];
                    }
                }
                let f_idx = dst[0] + dst[1] * di + dst[2] * di * dj;
                debug_assert!(dst[0] < di && dst[1] < dj && dst[2] < dk);
                c_to_f[c_idx] = f_idx;
            }
        }
    }
    c_to_f
}

fn apply_reorientation_to_affine(
    aff: [[f64; 4]; 4],
    src_axis_for_dst: [usize; 3],
    flip_dst_axis: [bool; 3],
    src_dims: [usize; 3],
) -> [[f64; 4]; 4] {
    // Build the index transform: old_voxel = R * new_voxel + offset
    // where R permutes and/or flips axes.
    let mut r = [[0.0f64; 3]; 3];
    let mut offset = [0.0f64; 3];
    for dst in 0..3 {
        let src = src_axis_for_dst[dst];
        if flip_dst_axis[dst] {
            r[src][dst] = -1.0;
            offset[src] = src_dims[src] as f64 - 1.0;
        } else {
            r[src][dst] = 1.0;
        }
    }
    // new_affine = aff_linear * R  |  aff_linear * offset + aff_translation
    let mut out = [[0.0f64; 4]; 4];
    for row in 0..3 {
        for dst_col in 0..3 {
            for src_col in 0..3 {
                out[row][dst_col] += aff[row][src_col] * r[src_col][dst_col];
            }
        }
        let mut trans = aff[row][3];
        for src_col in 0..3 {
            trans += aff[row][src_col] * offset[src_col];
        }
        out[row][3] = trans;
    }
    out[3][3] = 1.0;
    out
}

/// The axis permutation and flips that take a grid with affine `aff` to DSI
/// Studio's native LPS+ voxel order, as `(src_axis_for_dst, flip_dst_axis)`.
/// This matches qsiprep's nibabel-based reorientation model
/// (`io_orientation` + `ornt_transform`) rather than assuming an x/y mirror.
fn lps_reorientation(aff: [[f64; 4]; 4]) -> ([usize; 3], [bool; 3]) {
    let start = nibabel_io_orientation(aff);
    let end = [[0, -1], [1, -1], [2, 1]];
    let xform = nibabel_ornt_transform(start, end);
    let mut src_axis_for_dst = [0usize, 1, 2];
    let mut flip_dst_axis = [false, false, false];
    for (src_axis, [dst_axis, flip]) in xform.into_iter().enumerate() {
        let dst_axis = dst_axis as usize;
        src_axis_for_dst[dst_axis] = src_axis;
        flip_dst_axis[dst_axis] = flip == -1;
    }
    (src_axis_for_dst, flip_dst_axis)
}

/// Reorder a reference affine (typically the DWI NIfTI a `trans`-less
/// `.fib.gz` was reconstructed from) to the LPS voxel order DSI Studio stored
/// that DWI in, like nibabel's `io_orientation` + `ornt_transform` +
/// `inv_ornt_aff`. `dsi_dims` is the DSI Studio grid; the reference grid's
/// shape is recovered from it. A no-op for an affine already in LPS order.
pub fn reference_affine_to_dsistudio_grid(
    aff: [[f64; 4]; 4],
    dsi_dims: [usize; 3],
) -> [[f64; 4]; 4] {
    let (src_axis_for_dst, flip_dst_axis) = lps_reorientation(aff);
    let mut src_dims = [0usize; 3];
    for dst in 0..3 {
        src_dims[src_axis_for_dst[dst]] = dsi_dims[dst];
    }
    apply_reorientation_to_affine(aff, src_axis_for_dst, flip_dst_axis, src_dims)
}

fn reorients_to_lps(header: &crate::header::Header) -> bool {
    matches!(
        header
            .extra
            .get("_ODX_DSISTUDIO_VOXEL_POLICY")
            .and_then(|v| v.as_str()),
        Some("reorient_to_lps_fortran") | Some("flip_xy_to_lps_fortran")
    )
}

/// The `trans` a DSI Studio export of a dataset with this header gets, and
/// so the affine whose [`VoxelFrame`] its directions are written in.
pub(crate) fn dsistudio_export_affine(header: &crate::header::Header) -> [[f64; 4]; 4] {
    let aff = header.voxel_to_rasmm;
    if !reorients_to_lps(header) {
        return aff;
    }
    let src_dims = header.dimensions.map(|d| d as usize);
    let (src_axis_for_dst, flip_dst_axis) = lps_reorientation(aff);
    apply_reorientation_to_affine(aff, src_axis_for_dst, flip_dst_axis, src_dims)
}

fn build_dsistudio_export_geometry(
    aff: [[f64; 4]; 4],
    src_dims: [usize; 3],
    reorient_to_lps: bool,
) -> ([usize; 3], [f32; 3], Vec<usize>, [[f64; 4]; 4]) {
    let src_vox = affine_column_norms(aff);
    if !reorient_to_lps {
        return (
            src_dims,
            src_vox,
            build_reoriented_dsistudio_c_to_f(src_dims, src_dims, [0, 1, 2], [false, false, false]),
            aff,
        );
    }

    // Reorient the voxel grid into DSI Studio's native LPS+ array order before
    // the final Fortran flattening step.
    let (src_axis_for_dst, flip_dst_axis) = lps_reorientation(aff);
    let dst_dims = [
        src_dims[src_axis_for_dst[0]],
        src_dims[src_axis_for_dst[1]],
        src_dims[src_axis_for_dst[2]],
    ];
    let dst_vox = [
        src_vox[src_axis_for_dst[0]],
        src_vox[src_axis_for_dst[1]],
        src_vox[src_axis_for_dst[2]],
    ];
    let c_to_f =
        build_reoriented_dsistudio_c_to_f(src_dims, dst_dims, src_axis_for_dst, flip_dst_axis);
    let reoriented_aff =
        apply_reorientation_to_affine(aff, src_axis_for_dst, flip_dst_axis, src_dims);
    (dst_dims, dst_vox, c_to_f, reoriented_aff)
}

fn build_dsistudio_records(odx: &OdxDataset, masked_sloped: bool) -> Result<Vec<OwnedMatRecord>> {
    let header = odx.header();
    let src_dims = header.dimensions;
    let (si, sj, sk) = (
        src_dims[0] as usize,
        src_dims[1] as usize,
        src_dims[2] as usize,
    );
    let nvoxels_total = si * sj * sk;

    let aff = &header.voxel_to_rasmm;
    let (dst_dims_usize, vox_size, c_to_f, reoriented_aff) =
        build_dsistudio_export_geometry(*aff, [si, sj, sk], reorients_to_lps(header));
    // Sphere and peak directions go out in the voxel frame of the written grid.
    let frame = VoxelFrame::from_affine(&reoriented_aff)?;
    let (di, dj, dk) = (dst_dims_usize[0], dst_dims_usize[1], dst_dims_usize[2]);
    // `trans` stores the voxel→RAS+ mm affine directly, even though the DSI
    // Studio voxel arrays themselves are reoriented into LPS+ index space.
    // The reoriented affine already reflects the index-space transform.
    // DSI Studio uses this full 4×4 for NIfTI I/O and tract I/O;
    // `initial_LPS_nifti_srow` is only a fallback when `trans` is absent
    // (fib_data.cpp:752-754). `apply_trans`/`apply_inverse_trans` are diagonal-only
    // helpers used only in atlas-registration paths and do not affect this write path.
    let trans = nifti_affine_to_dsistudio_trans(reoriented_aff);

    let mask = odx.mask();
    let mut masked_c_indices = Vec::with_capacity(odx.nb_voxels());
    for c_idx in 0..nvoxels_total {
        if mask[c_idx] != 0 {
            masked_c_indices.push(c_idx);
        }
    }
    let mut mask_f = vec![0u8; nvoxels_total];
    for &c_idx in &masked_c_indices {
        mask_f[c_to_f[c_idx]] = 1;
    }
    let sparse_indices: Vec<usize> = (0..nvoxels_total).filter(|&idx| mask_f[idx] != 0).collect();

    let voxel_frame_verts: Vec<[f32; 3]> = match odx.sphere_vertices() {
        Some(verts) => verts.iter().map(|&v| frame.from_ras(v)).collect(),
        // A dense ODF without a `sphere/` payload was sampled on the RAS
        // `dsistudio_odf8` sphere that its SPHERE_ID names.
        None if !odx.odf_arrays().is_empty() => dsistudio_odf8::full_vertices_ras()
            .iter()
            .map(|&v| frame.from_ras(v))
            .collect(),
        // DSI Studio can fall back to its built-in `odf8` sphere when explicit
        // sphere records are absent, so the writer does the same; that sphere
        // is defined in the voxel frame and needs no rotation.
        None => builtin_voxel_frame_vertices(),
    };
    let sphere_faces = odx.sphere_faces().unwrap_or(dsistudio_odf8::faces());

    let odf_verts_data: Vec<f32> = voxel_frame_verts.iter().flatten().copied().collect();
    let odf_faces_data: Vec<i16> = sphere_faces
        .iter()
        .flat_map(|f| [f[0] as i16, f[1] as i16, f[2] as i16])
        .collect();

    let offsets = odx.offsets();
    let directions = odx.directions();
    let amplitudes = odx
        .scalar_dpf_f32("amplitude")
        .unwrap_or_else(|_| vec![0.0; odx.nb_peaks()]);

    let max_peaks = (0..odx.nb_voxels())
        .map(|i| odx.peaks_per_voxel(i))
        .max()
        .unwrap_or(0);

    let mut fa_arrays: Vec<Vec<f32>> = (0..max_peaks)
        .map(|_| vec![0.0f32; nvoxels_total])
        .collect();
    let mut idx_arrays: Vec<Vec<i16>> = (0..max_peaks).map(|_| vec![0i16; nvoxels_total]).collect();
    let write_dir_records = header
        .extra
        .get(WRITE_DIR_RECORDS_KEY)
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let mut dir_arrays: Vec<Vec<f32>> = if write_dir_records {
        (0..max_peaks).map(|_| vec![0.0f32; nvoxels_total * 3]).collect()
    } else {
        Vec::new()
    };
    let mut extra_peak_arrays = Vec::new();
    for (name, info) in odx.iter_dpf() {
        if info.ncols != 1 || name == "amplitude" || name == "pam_peak_index" {
            continue;
        }
        if let Ok(values) = odx.scalar_dpf_f32(name) {
            extra_peak_arrays.push((
                name.to_string(),
                values,
                (0..max_peaks)
                    .map(|_| vec![0.0f32; nvoxels_total])
                    .collect::<Vec<_>>(),
            ));
        }
    }

    for (voxel_row, &c_idx) in masked_c_indices.iter().enumerate() {
        let f_idx = c_to_f[c_idx];
        let start = offsets[voxel_row] as usize;
        let end = offsets[voxel_row + 1] as usize;
        for (p, peak_idx) in (start..end).enumerate() {
            if p >= max_peaks {
                break;
            }
            fa_arrays[p][f_idx] = amplitudes[peak_idx];
            // DSI Studio peak indices refer to the voxel-frame sphere, so
            // internal RAS directions are mapped back before the lookup.
            let voxel_dir = frame.from_ras(directions[peak_idx]);
            idx_arrays[p][f_idx] = closest_vertex(&voxel_frame_verts, &voxel_dir) as i16;
            if write_dir_records {
                dir_arrays[p][f_idx * 3..f_idx * 3 + 3].copy_from_slice(&voxel_dir);
            }
            for (_, values, per_peak) in &mut extra_peak_arrays {
                if peak_idx < values.len() {
                    per_peak[p][f_idx] = values[peak_idx];
                }
            }
        }
    }

    let mut records = Vec::new();
    records.push(OwnedMatRecord {
        name: "dimension".into(),
        mrows: 1,
        ncols: 3,
        type_flag: 20,
        data: vec_to_bytes(vec![di as i32, dj as i32, dk as i32]),
        subrecords: Vec::new(),
    });
    records.push(float_record("voxel_size", vox_size.to_vec(), 1, 3));
    records.push(float_record("trans", trans, 4, 4));
    records.push(float_record(
        "odf_vertices",
        odf_verts_data,
        3,
        voxel_frame_verts.len(),
    ));
    records.push(int16_record(
        "odf_faces",
        odf_faces_data,
        3,
        sphere_faces.len(),
    ));
    records.push(uint8_record("mask", mask_f.clone(), di * dj, dk));

    for p in 0..max_peaks {
        if masked_sloped {
            records.push(masked_float_record(
                format!("fa{p}"),
                &fa_arrays[p],
                &sparse_indices,
            ));
            records.push(masked_i16_record(
                format!("index{p}"),
                &idx_arrays[p],
                &sparse_indices,
            ));
        } else {
            records.push(float_record(
                format!("fa{p}"),
                fa_arrays[p].clone(),
                1,
                nvoxels_total,
            ));
            records.push(int16_record(
                format!("index{p}"),
                idx_arrays[p].clone(),
                1,
                nvoxels_total,
            ));
        }
    }
    for (p, values) in dir_arrays.into_iter().enumerate() {
        if masked_sloped {
            records.push(masked_vec3_record(
                format!("dir{p}"),
                &values,
                &sparse_indices,
            ));
        } else {
            records.push(float_record(format!("dir{p}"), values, 3, nvoxels_total));
        }
    }
    for (name, _, per_peak) in extra_peak_arrays {
        for (p, values) in per_peak.into_iter().enumerate() {
            if masked_sloped {
                records.push(masked_float_record(
                    format!("{name}{p}"),
                    &values,
                    &sparse_indices,
                ));
            } else {
                records.push(float_record(format!("{name}{p}"), values, 1, nvoxels_total));
            }
        }
    }

    if let Some(z0_val) = header.extra.get("Z0").and_then(|v| v.as_f64()) {
        records.push(float_record("z0", vec![z0_val as f32], 1, 1));
    }

    if let Ok(odf_view) = odx.odf::<f32>("amplitudes") {
        let n_odf_dirs = odf_view.ncols();
        // DSI dense ODF matrices are hemisphere-sampled. The first `n_odf_dirs`
        // rows of the full sphere define the payload order.
        let mut f_order_rows: Vec<usize> = masked_c_indices
            .iter()
            .enumerate()
            .map(|(odx_row, &c_idx)| (c_to_f[c_idx], odx_row))
            .collect::<Vec<_>>()
            .into_iter()
            .sorted_by_key(|(f_idx, _)| *f_idx)
            .map(|(_, odx_row)| odx_row)
            .collect();
        let chunk_size = 20000usize;
        let mut chunk_idx = 0usize;
        for chunk_start in (0..f_order_rows.len()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(f_order_rows.len());
            let chunk_len = chunk_end - chunk_start;
            let mut chunk_data = vec![0.0f32; n_odf_dirs * chunk_len];
            for (col, &odx_row) in f_order_rows[chunk_start..chunk_end].iter().enumerate() {
                let row_data = odf_view.row(odx_row);
                for (dir, &val) in row_data.iter().enumerate() {
                    chunk_data[dir + col * n_odf_dirs] = val;
                }
            }
            if masked_sloped {
                records.push(sloped_record(
                    format!("odf{chunk_idx}"),
                    chunk_data,
                    n_odf_dirs,
                    chunk_len,
                ));
            } else {
                records.push(float_record(
                    format!("odf{chunk_idx}"),
                    chunk_data,
                    n_odf_dirs,
                    chunk_len,
                ));
            }
            chunk_idx += 1;
        }
        f_order_rows.clear();
    }

    let dpv_keys_to_fib = [
        ("dti_fa", "dti_fa"),
        ("gfa", "gfa"),
        ("md", "md"),
        ("ad", "ad"),
        ("rd", "rd"),
        ("iso", "iso"),
        ("rdi", "rdi"),
        ("ha", "ha"),
        ("rd1", "rd1"),
        ("rd2", "rd2"),
    ];
    for &(odx_key, fib_key) in &dpv_keys_to_fib {
        if let Ok(vals) = odx.scalar_dpv_f32(odx_key) {
            let mut flat = vec![0.0f32; nvoxels_total];
            for (row, &c_idx) in masked_c_indices.iter().enumerate() {
                flat[c_to_f[c_idx]] = vals[row];
            }
            if masked_sloped {
                records.push(masked_sloped_record(
                    fib_key.to_string(),
                    &flat,
                    &sparse_indices,
                ));
            } else {
                records.push(float_record(fib_key.to_string(), flat, di * dj, dk));
            }
        }
    }

    Ok(records)
}

fn sloped_record(
    name: impl Into<String>,
    values: Vec<f32>,
    mrows: usize,
    ncols: usize,
) -> OwnedMatRecord {
    let (data, slope, intercept) = quantize_to_u8(&values);
    let name = name.into();
    let mut record = uint8_record(name.clone(), data, mrows, ncols);
    record.subrecords = vec![
        float_record(format!("{name}.slope"), vec![slope], 1, 1),
        float_record(format!("{name}.inter"), vec![intercept], 1, 1),
    ];
    record
}

fn masked_sloped_record(
    name: impl Into<String>,
    values: &[f32],
    sparse_indices: &[usize],
) -> OwnedMatRecord {
    let sparse: Vec<f32> = sparse_indices.iter().map(|&idx| values[idx]).collect();
    sloped_record(name, sparse, 1, sparse_indices.len())
}

fn masked_float_record(
    name: impl Into<String>,
    values: &[f32],
    sparse_indices: &[usize],
) -> OwnedMatRecord {
    let sparse: Vec<f32> = sparse_indices.iter().map(|&idx| values[idx]).collect();
    float_record(name, sparse, 1, sparse_indices.len())
}

/// Masked `3 × sparse_count` float record (xyz per masked voxel), the layout
/// DSI Studio's `.fz` reader expands for `dir{p}`.
fn masked_vec3_record(
    name: impl Into<String>,
    values: &[f32],
    sparse_indices: &[usize],
) -> OwnedMatRecord {
    let mut sparse = Vec::with_capacity(sparse_indices.len() * 3);
    for &idx in sparse_indices {
        sparse.extend_from_slice(&values[idx * 3..idx * 3 + 3]);
    }
    float_record(name, sparse, 3, sparse_indices.len())
}

fn masked_i16_record(
    name: impl Into<String>,
    values: &[i16],
    sparse_indices: &[usize],
) -> OwnedMatRecord {
    let sparse: Vec<i16> = sparse_indices.iter().map(|&idx| values[idx]).collect();
    int16_record(name, sparse, 1, sparse_indices.len())
}

fn quantize_to_u8(values: &[f32]) -> (Vec<u8>, f32, f32) {
    let mut min_v = values.first().copied().unwrap_or(0.0);
    let mut max_v = min_v;
    for &v in values.iter().skip(1) {
        if v < min_v {
            min_v = v;
        }
        if v > max_v {
            max_v = v;
        }
    }
    let slope = ((max_v - min_v) / 255.99f32).max(1e-12);
    let inv = 1.0f32 / slope;
    let data = values
        .iter()
        .map(|&v| ((v - min_v) * inv).clamp(0.0, 255.0) as u8)
        .collect();
    (data, slope, min_v)
}

fn closest_vertex(vertices: &[[f32; 3]], dir: &[f32; 3]) -> usize {
    let mut best_idx = 0usize;
    let mut best_dot = f32::NEG_INFINITY;
    for (i, v) in vertices.iter().enumerate() {
        let dot = v[0] * dir[0] + v[1] * dir[1] + v[2] * dir[2];
        if dot > best_dot {
            best_dot = dot;
            best_idx = i;
        }
    }
    best_idx
}

trait SortedByKeyExt<T>: Iterator<Item = T> + Sized {
    fn sorted_by_key<K: Ord, F: FnMut(&T) -> K>(self, mut f: F) -> std::vec::IntoIter<T> {
        let mut items: Vec<T> = self.collect();
        items.sort_by_key(|item| f(item));
        items.into_iter()
    }
}

impl<I, T> SortedByKeyExt<T> for I where I: Iterator<Item = T> {}

#[cfg(test)]
mod tests {
    use nalgebra::Rotation3;
    use tempfile::tempdir;

    use super::*;

    /// DSI Studio's LPS voxel axes rotated by (12, -7, 5) degrees about the RAS
    /// axes, with anisotropic voxels: what DSI Studio writes for an oblique scan.
    fn oblique_trans() -> [[f64; 4]; 4] {
        let rot =
            Rotation3::from_euler_angles(12f64.to_radians(), -7f64.to_radians(), 5f64.to_radians());
        let axes = rot.matrix() * Matrix3::from_diagonal(&Vector3::new(-1.0, -1.0, 1.0));
        let vox = [1.5, 1.5, 2.0];
        let mut aff = [[0.0; 4]; 4];
        for r in 0..3 {
            for c in 0..3 {
                aff[r][c] = axes[(r, c)] * vox[c];
            }
        }
        aff[0][3] = 70.0;
        aff[1][3] = 90.0;
        aff[2][3] = -10.0;
        aff[3][3] = 1.0;
        aff
    }

    fn lps_trans() -> [[f64; 4]; 4] {
        [
            [-1.5, 0.0, 0.0, 70.0],
            [0.0, -1.5, 0.0, 90.0],
            [0.0, 0.0, 2.0, -10.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    fn mat_vec(aff: &[[f64; 4]; 4], v: [f32; 3]) -> [f32; 3] {
        let col_norm = |c: usize| (0..3).map(|r| aff[r][c] * aff[r][c]).sum::<f64>().sqrt();
        std::array::from_fn(|r| {
            (0..3)
                .map(|c| aff[r][c] / col_norm(c) * f64::from(v[c]))
                .sum::<f64>() as f32
        })
    }

    fn normalized(v: [f32; 3]) -> [f32; 3] {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / n, v[1] / n, v[2] / n]
    }

    fn assert_close(a: [f32; 3], b: [f32; 3], tol: f32) {
        for k in 0..3 {
            assert!((a[k] - b[k]).abs() <= tol, "{a:?} != {b:?}");
        }
    }

    // Two voxels along j (dims 1x2x1). Voxel 0: peak 0 at sphere vertex 10,
    // peak 1 given by an exact `dir1` record. Voxel 1: peak 0 at vertex 200.
    const DIR1: [f32; 3] = [0.2, 0.3, 0.9];

    fn write_synthetic_fib(path: &Path, trans: Option<[[f64; 4]; 4]>) {
        let verts = builtin_voxel_frame_vertices();
        let faces = dsistudio_odf8::faces();
        let mut records = vec![
            OwnedMatRecord {
                name: "dimension".into(),
                mrows: 1,
                ncols: 3,
                type_flag: 20,
                data: vec_to_bytes(vec![1i32, 2, 1]),
                subrecords: Vec::new(),
            },
            float_record("voxel_size", vec![1.5, 1.5, 2.0], 1, 3),
        ];
        if let Some(trans) = trans {
            records.push(float_record(
                "trans",
                nifti_affine_to_dsistudio_trans(trans),
                4,
                4,
            ));
        }
        records.extend([
            float_record(
                "odf_vertices",
                verts.iter().flatten().copied().collect(),
                3,
                verts.len(),
            ),
            int16_record(
                "odf_faces",
                faces
                    .iter()
                    .flat_map(|f| f.map(|i| i as i16))
                    .collect::<Vec<_>>(),
                3,
                faces.len(),
            ),
            uint8_record("mask", vec![1, 1], 2, 1),
            float_record("fa0", vec![0.5, 0.4], 1, 2),
            int16_record("index0", vec![10, 200], 1, 2),
            float_record("fa1", vec![0.3, 0.0], 1, 2),
            int16_record("index1", vec![0, 0], 1, 2),
            float_record(
                "dir1",
                [DIR1, [0.0; 3]].into_iter().flatten().collect(),
                3,
                2,
            ),
        ]);
        mat4::write_mat4_gz(path, &records).unwrap();
    }

    #[test]
    fn oblique_directions_are_rotated_by_trans() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("oblique.fib.gz");
        let trans = oblique_trans();
        write_synthetic_fib(&path, Some(trans));

        let odx = load_fibgz(&path, None).unwrap();
        let builtin = builtin_voxel_frame_vertices();
        let verts = odx.sphere_vertices().unwrap();
        for (got, &v) in verts.iter().zip(&builtin) {
            assert_close(*got, mat_vec(&trans, v), 1e-6);
        }
        // A rotated sphere is no longer the canonical RAS `dsistudio_odf8`.
        assert_eq!(odx.header().sphere_id, None);

        let dirs = odx.directions();
        assert_eq!(odx.offsets(), &[0, 2, 3]);
        // `index{p}` resolves through the rotated sphere ...
        assert_close(dirs[0], mat_vec(&trans, builtin[10]), 1e-6);
        assert_close(dirs[2], mat_vec(&trans, builtin[200]), 1e-6);
        assert_close(dirs[0], verts[10], 0.0);
        // ... and `dir{p}` gets the same rotation.
        assert_close(dirs[1], mat_vec(&trans, normalized(DIR1)), 1e-6);
        // The rotation is not just the non-oblique x/y flip.
        let flipped = [-builtin[10][0], -builtin[10][1], builtin[10][2]];
        let dot: f32 = (0..3).map(|k| dirs[0][k] * flipped[k]).sum();
        assert!(dot < 0.99, "oblique rotation was dropped");
    }

    #[test]
    fn non_oblique_trans_keeps_the_xy_flip_and_builtin_sphere() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("lps.fib.gz");
        write_synthetic_fib(&path, Some(lps_trans()));

        let odx = load_fibgz(&path, None).unwrap();
        assert_eq!(
            odx.sphere_vertices().unwrap(),
            dsistudio_odf8::full_vertices_ras()
        );
        assert_eq!(odx.header().sphere_id.as_deref(), Some("dsistudio_odf8"));
        let d = normalized(DIR1);
        assert_eq!(odx.directions()[1], [-d[0], -d[1], d[2]]);
    }

    #[test]
    fn oblique_export_writes_voxel_frame_sphere_and_peaks() {
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("oblique.fib.gz");
        let trans = oblique_trans();
        write_synthetic_fib(&src, Some(trans));
        let odx = load_fibgz(&src, None).unwrap();
        let mut parts = odx.clone_owned_parts();
        parts
            .header
            .extra
            .insert(WRITE_DIR_RECORDS_KEY.into(), serde_json::Value::Bool(true));
        let odx = OdxDataset::from_parts(parts);

        let builtin = builtin_voxel_frame_vertices();
        let voxel_dir = |m: &MatCatalog, name: &str, voxel: usize| -> [f32; 3] {
            let v = m.get(name).unwrap().as_f32_vec();
            [v[voxel * 3], v[voxel * 3 + 1], v[voxel * 3 + 2]]
        };
        for (name, masked) in [("out.fib.gz", false), ("out.fz", true)] {
            let out = tmp.path().join(name);
            if masked {
                save_fz(&odx, &out).unwrap();
            } else {
                save_fibgz(&odx, &out).unwrap();
            }
            let mat = mat4::read_mat4_gz(&out).unwrap();
            let written: Vec<f32> = nifti_affine_to_dsistudio_trans(trans);
            assert_eq!(mat.get("trans").unwrap().as_f32_vec(), written);
            let verts = mat.get("odf_vertices").unwrap().as_f32_vec();
            for (i, v) in builtin.iter().enumerate() {
                assert_close([verts[3 * i], verts[3 * i + 1], verts[3 * i + 2]], *v, 1e-6);
            }
            assert_eq!(mat.get("index0").unwrap().as_i32_vec(), vec![10, 200]);
            assert_close(voxel_dir(&mat, "dir0", 0), builtin[10], 1e-6);
            assert_close(voxel_dir(&mat, "dir1", 0), normalized(DIR1), 1e-6);
            assert_close(voxel_dir(&mat, "dir0", 1), builtin[200], 1e-6);

            let back = load_dsistudio_mat(&out, None, &LoadOptions::default()).unwrap();
            for (a, b) in back.directions().iter().zip(odx.directions()) {
                assert_close(*a, *b, 1e-6);
            }
            for (a, b) in back
                .sphere_vertices()
                .unwrap()
                .iter()
                .zip(odx.sphere_vertices().unwrap())
            {
                assert_close(*a, *b, 1e-6);
            }
        }
    }

    #[test]
    fn reference_affine_is_reordered_to_dsistudio_lps_grid() {
        // An oblique 'ASR' grid of shape (4, 3, 5) and the LPS-reordered affine
        // nibabel gives for it (io_orientation + ornt_transform +
        // inv_ornt_aff); DSI Studio stores that grid as (5, 4, 3).
        let asr = [
            [
                -0.16573923439829594,
                -0.2012637848259822,
                1.2359615173455631,
                10.5,
            ],
            [
                1.4583256443817185,
                -0.4350200589506861,
                0.10813262132203647,
                -20.25,
            ],
            [
                0.30954292290361113,
                1.941713273691062,
                0.15233667925643435,
                30.0,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let lps = [
            [
                -1.2359615173455631,
                0.16573923439829594,
                -0.2012637848259822,
                14.946628366187365,
            ],
            [
                -0.10813262132203647,
                -1.4583256443817185,
                -0.4350200589506861,
                -15.442492581566698,
            ],
            [
                -0.15233667925643435,
                -0.30954292290361113,
                1.941713273691062,
                31.53797548573657,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let close = |a: [[f64; 4]; 4], b: [[f64; 4]; 4]| {
            for r in 0..4 {
                for c in 0..4 {
                    assert!((a[r][c] - b[r][c]).abs() < 1e-9, "{a:?} != {b:?}");
                }
            }
        };
        close(reference_affine_to_dsistudio_grid(asr, [5, 4, 3]), lps);
        // Already in DSI Studio's order: nothing to do.
        close(reference_affine_to_dsistudio_grid(lps, [5, 4, 3]), lps);
    }

    #[test]
    fn las_reference_affine_for_fib_without_trans_is_reordered() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("chen.fib.gz");
        write_synthetic_fib(&path, None);
        // The LAS DWI the file came from: DSI Studio flipped its j axis, so
        // voxel j of the DWI is voxel (2 - 1) - j of the file.
        let trans = oblique_trans();
        let mut las = trans;
        for r in 0..3 {
            las[r][1] = -trans[r][1];
            las[r][3] = trans[r][3] + trans[r][1];
        }
        let odx = load_fibgz(&path, Some(las)).unwrap();
        let aff = odx.header().voxel_to_rasmm;
        for r in 0..4 {
            for c in 0..4 {
                assert!((aff[r][c] - trans[r][c]).abs() < 1e-9, "{aff:?}");
            }
        }
        let builtin = builtin_voxel_frame_vertices();
        assert_close(odx.directions()[0], mat_vec(&trans, builtin[10]), 1e-6);

        let err = load_fibgz(&path, None).unwrap_err().to_string();
        assert!(
            err.contains("reordered to DSI Studio's LPS voxel order"),
            "{err}"
        );
    }
}
