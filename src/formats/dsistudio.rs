use std::collections::HashMap;
use std::path::Path;

use nalgebra::Matrix3;

use crate::data_array::DataArray;
use crate::dtype::DType;
use crate::error::{OdxError, Result};
use crate::formats::dsistudio_odf8;
use crate::formats::mat4::{
    self, float_record, int16_record, uint8_record, MatCatalog, MatRecord, OwnedMatRecord,
};
use crate::header::CanonicalDenseRepresentation;
use crate::mmap_backing::{vec_to_bytes, MmapBacking};
use crate::odx_file::{OdxDataset, OdxParts};

pub fn load_fibgz(path: &Path, affine: Option<[[f64; 4]; 4]>) -> Result<OdxDataset> {
    load_dsistudio_mat(path, affine)
}

pub fn load_fz(path: &Path, affine: Option<[[f64; 4]; 4]>) -> Result<OdxDataset> {
    load_dsistudio_mat(path, affine)
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
) -> Result<OdxDataset> {
    let mat = mat4::read_mat4_gz(path)?;

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
        a
    } else {
        return Err(OdxError::Format(
            "DSI Studio file has no spatial affine ('trans' field). \
             Convert it first with: odx convert --reference-affine <nifti> <file> <output.odx>"
                .into(),
        ));
    };

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

    let sphere_vertices = if let Some(record) = mat.get("odf_vertices") {
        let verts_flat = record.as_f32_vec();
        let mut out = Vec::with_capacity(record.ncols());
        for i in 0..record.ncols() {
            out.push([
                // DSI Studio stores sphere geometry in LPS. ODX keeps directions
                // and spheres in RAS, so imported geometry is flipped once here.
                -verts_flat[i * record.mrows()],
                -verts_flat[i * record.mrows() + 1],
                verts_flat[i * record.mrows() + 2],
            ]);
        }
        Some(out)
    } else {
        Some(dsistudio_odf8::full_vertices_ras().to_vec())
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
    let mut index_cache = Vec::with_capacity(max_peaks);
    for p in 0..max_peaks {
        fa_cache.push(load_volume_f32(
            get_required(&mat, &format!("fa{p}"))?,
            nvoxels_total,
            &sparse_row_by_f,
            sparse_count,
        ));
        index_cache.push(load_volume_i32(
            get_required(&mat, &format!("index{p}"))?,
            nvoxels_total,
            &sparse_row_by_f,
            sparse_count,
        ));
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
            let dir_idx = index_cache[p][f_idx] as usize;
            if let Some(verts) = sphere_vertices.as_ref() {
                if dir_idx < verts.len() {
                    directions.push(verts[dir_idx]);
                    amplitudes.push(fa_val);
                    n_peaks_this_voxel += 1;
                }
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
    while mat.has(&format!("fa{n}")) && mat.has(&format!("index{n}")) {
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
        if matches!(prefix, "fa" | "index" | "odf") {
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

// NIfTI affine (voxel→RAS+ mm) = diag(-1,-1,1,1) · DSI Studio `trans` (voxel→LPS+ mm).
fn dsistudio_trans_to_nifti_affine(flat: Vec<f32>) -> [[f64; 4]; 4] {
    let mut lps = [[0.0f64; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            let idx = r * 4 + c;
            if idx < flat.len() {
                lps[r][c] = flat[idx] as f64;
            }
        }
    }
    [
        [-lps[0][0], -lps[0][1], -lps[0][2], -lps[0][3]],
        [-lps[1][0], -lps[1][1], -lps[1][2], -lps[1][3]],
        [lps[2][0], lps[2][1], lps[2][2], lps[2][3]],
        [lps[3][0], lps[3][1], lps[3][2], lps[3][3]],
    ]
}

// DSI Studio `trans` (voxel→LPS+ mm) = diag(-1,-1,1,1) · NIfTI affine (voxel→RAS+ mm).
fn nifti_affine_to_dsistudio_trans(aff: [[f64; 4]; 4]) -> Vec<f32> {
    let lps = [
        [-aff[0][0], -aff[0][1], -aff[0][2], -aff[0][3]],
        [-aff[1][0], -aff[1][1], -aff[1][2], -aff[1][3]],
        [aff[2][0], aff[2][1], aff[2][2], aff[2][3]],
        [aff[3][0], aff[3][1], aff[3][2], aff[3][3]],
    ];
    lps.into_iter()
        .flat_map(|row| row.into_iter().map(|v| v as f32))
        .collect()
}

fn affine_column_norms(aff: [[f64; 4]; 4]) -> [f32; 3] {
    [
        (aff[0][0] * aff[0][0] + aff[1][0] * aff[1][0] + aff[2][0] * aff[2][0]).sqrt() as f32,
        (aff[0][1] * aff[0][1] + aff[1][1] * aff[1][1] + aff[2][1] * aff[2][1]).sqrt() as f32,
        (aff[0][2] * aff[0][2] + aff[1][2] * aff[1][2] + aff[2][2] * aff[2][2]).sqrt() as f32,
    ]
}

fn nibabel_io_orientation(aff: [[f64; 4]; 4]) -> [[i8; 2]; 3] {
    // This mirrors nibabel.orientations.io_orientation(): derive the closest
    // voxel-axis -> world-axis mapping from the affine, rather than assuming
    // voxel axes already correspond to x/y/z in anatomical order.
    let rzs = Matrix3::new(
        aff[0][0], aff[0][1], aff[0][2], aff[1][0], aff[1][1], aff[1][2], aff[2][0], aff[2][1],
        aff[2][2],
    );
    let zooms = affine_column_norms(aff);
    let rs = Matrix3::new(
        rzs[(0, 0)] / f64::from(zooms[0].max(1e-12)),
        rzs[(0, 1)] / f64::from(zooms[1].max(1e-12)),
        rzs[(0, 2)] / f64::from(zooms[2].max(1e-12)),
        rzs[(1, 0)] / f64::from(zooms[0].max(1e-12)),
        rzs[(1, 1)] / f64::from(zooms[1].max(1e-12)),
        rzs[(1, 2)] / f64::from(zooms[2].max(1e-12)),
        rzs[(2, 0)] / f64::from(zooms[0].max(1e-12)),
        rzs[(2, 1)] / f64::from(zooms[1].max(1e-12)),
        rzs[(2, 2)] / f64::from(zooms[2].max(1e-12)),
    );
    let svd = rs.svd(true, true);
    let u = svd.u.expect("requested U from SVD");
    let vt = svd.v_t.expect("requested V^T from SVD");
    let s = svd.singular_values;
    let tol = s.max() * 3.0 * f64::EPSILON;
    let mut r = Matrix3::<f64>::zeros();
    for idx in 0..3 {
        if s[idx] > tol {
            let ui = u.column(idx);
            let vti = vt.row(idx);
            r += ui * vti;
        }
    }

    let mut in_axes = [0usize, 1, 2];
    in_axes.sort_by(|&a, &b| {
        let sa = (0..3).map(|row| r[(row, a)].powi(2)).fold(0.0, f64::max);
        let sb = (0..3).map(|row| r[(row, b)].powi(2)).fold(0.0, f64::max);
        sb.total_cmp(&sa)
    });

    let mut ornt = [[-1i8, 1i8]; 3];
    let mut work = r;
    for in_ax in in_axes {
        let mut out_ax = 0usize;
        let mut best = 0.0f64;
        for row in 0..3 {
            let value = work[(row, in_ax)].abs();
            if value > best {
                best = value;
                out_ax = row;
            }
        }
        ornt[in_ax][0] = out_ax as i8;
        ornt[in_ax][1] = if work[(out_ax, in_ax)] < 0.0 { -1 } else { 1 };
        for col in 0..3 {
            work[(out_ax, col)] = 0.0;
        }
    }
    ornt
}

fn nibabel_ornt_transform(start_ornt: [[i8; 2]; 3], end_ornt: [[i8; 2]; 3]) -> [[i8; 2]; 3] {
    let mut result = [[0i8, 1i8]; 3];
    for (end_in_idx, [end_out_idx, end_flip]) in end_ornt.into_iter().enumerate() {
        for (start_in_idx, [start_out_idx, start_flip]) in start_ornt.into_iter().enumerate() {
            if end_out_idx == start_out_idx {
                result[start_in_idx] = [
                    end_in_idx as i8,
                    if start_flip == end_flip { 1 } else { -1 },
                ];
                break;
            }
        }
    }
    result
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

    let start = nibabel_io_orientation(aff);
    let end = [[0, -1], [1, -1], [2, 1]];
    // Reorient the voxel grid into DSI Studio's native LPS+ array order before
    // the final Fortran flattening step, matching qsiprep's nibabel-based
    // reorientation model rather than assuming a simple x/y mirror.
    let xform = nibabel_ornt_transform(start, end);
    let mut src_axis_for_dst = [0usize, 1, 2];
    let mut flip_dst_axis = [false, false, false];
    for (src_axis, [dst_axis, flip]) in xform.into_iter().enumerate() {
        let dst_axis = dst_axis as usize;
        src_axis_for_dst[dst_axis] = src_axis;
        flip_dst_axis[dst_axis] = flip == -1;
    }
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
    let reorient_to_lps = matches!(
        header
            .extra
            .get("_ODX_DSISTUDIO_VOXEL_POLICY")
            .and_then(|v| v.as_str()),
        Some("reorient_to_lps_fortran") | Some("flip_xy_to_lps_fortran")
    );
    let (dst_dims_usize, vox_size, c_to_f, reoriented_aff) =
        build_dsistudio_export_geometry(*aff, [si, sj, sk], reorient_to_lps);
    let (di, dj, dk) = (dst_dims_usize[0], dst_dims_usize[1], dst_dims_usize[2]);
    // `trans` stores the voxel→LPS+ mm affine. We derive it from the reoriented
    // NIfTI affine (voxel→RAS+ mm) by flipping the RAS+ output space to LPS+ mm.
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

    let sphere_verts = odx
        .sphere_vertices()
        // DSI Studio can fall back to its built-in `odf8` sphere when explicit
        // sphere records are absent, so the writer does the same.
        .unwrap_or(dsistudio_odf8::full_vertices_ras());
    let sphere_faces = odx.sphere_faces().unwrap_or(dsistudio_odf8::faces());

    let odf_verts_data: Vec<f32> = sphere_verts
        .iter()
        // Export back to DSI Studio's LPS sphere convention.
        .flat_map(|v| [-v[0], -v[1], v[2]])
        .collect();
    let odf_faces_data: Vec<i16> = sphere_faces
        .iter()
        .flat_map(|f| [f[0] as i16, f[1] as i16, f[2] as i16])
        .collect();

    let offsets = odx.offsets();
    let directions = odx.directions();
    let amplitudes = odx
        .scalar_dpf_f32("amplitude")
        .unwrap_or_else(|_| vec![0.0; odx.nb_peaks()]);

    let lps_verts: Vec<[f32; 3]> = sphere_verts.iter().map(|v| [-v[0], -v[1], v[2]]).collect();

    let max_peaks = (0..odx.nb_voxels())
        .map(|i| odx.peaks_per_voxel(i))
        .max()
        .unwrap_or(0);

    let mut fa_arrays: Vec<Vec<f32>> = (0..max_peaks)
        .map(|_| vec![0.0f32; nvoxels_total])
        .collect();
    let mut idx_arrays: Vec<Vec<i16>> = (0..max_peaks).map(|_| vec![0i16; nvoxels_total]).collect();
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
            let dir = directions[peak_idx];
            // DSI Studio peak indices refer to the LPS sphere, so internal RAS
            // directions are flipped back before nearest-vertex lookup.
            let lps_dir = [-dir[0], -dir[1], dir[2]];
            idx_arrays[p][f_idx] = closest_vertex(&lps_verts, &lps_dir) as i16;
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
        sphere_verts.len(),
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
