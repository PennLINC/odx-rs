use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::str::FromStr;

use crate::data_array::DataArray;
use crate::dtype::DType;
use crate::error::{OdxError, Result};
use crate::header::CanonicalDenseRepresentation;
use crate::mmap_backing::{vec_to_bytes, MmapBacking};
use crate::odx_file::{OdxDataset, OdxParts};

const PAM_VERSION: &str = "0.0.1";
const PAM_BASIS_ASSUMED: &str = "_ODX_PAM_SH_BASIS_ASSUMED";

#[derive(Debug, Clone, Default)]
pub struct PamWriteOptions;

#[cfg(not(feature = "pam5"))]
pub fn load_pam5(_path: &Path) -> Result<OdxDataset> {
    Err(OdxError::Argument(
        "PAM5 support is disabled; rebuild with the 'pam5' feature".into(),
    ))
}

#[cfg(not(feature = "pam5"))]
pub fn save_pam5(_odx: &OdxDataset, _path: &Path, _options: &PamWriteOptions) -> Result<()> {
    Err(OdxError::Argument(
        "PAM5 support is disabled; rebuild with the 'pam5' feature".into(),
    ))
}

#[cfg(feature = "pam5")]
pub fn load_pam5(path: &Path) -> Result<OdxDataset> {
    use hdf5_metno::types::VarLenUnicode;
    use hdf5_metno::File;
    use serde_json::{Number, Value};

    let file = File::open(path)?;
    let version = file
        .attr("version")
        .map_err(|_| OdxError::Format("missing PAM5 file attribute 'version'".into()))?
        .read_scalar::<VarLenUnicode>()?;
    if version.as_str() != PAM_VERSION {
        return Err(OdxError::Format(format!(
            "unsupported PAM5 version '{}'",
            version.as_str()
        )));
    }

    let pam = file
        .group("pam")
        .map_err(|_| OdxError::Format("missing HDF5 group 'pam'".into()))?;

    let peak_dirs_ds = pam
        .dataset("peak_dirs")
        .map_err(|_| OdxError::Format("missing PAM dataset 'peak_dirs'".into()))?;
    let peak_values_ds = pam
        .dataset("peak_values")
        .map_err(|_| OdxError::Format("missing PAM dataset 'peak_values'".into()))?;
    let peak_indices_ds = pam
        .dataset("peak_indices")
        .map_err(|_| OdxError::Format("missing PAM dataset 'peak_indices'".into()))?;
    let sphere_vertices_ds = pam
        .dataset("sphere_vertices")
        .map_err(|_| OdxError::Format("missing PAM dataset 'sphere_vertices'".into()))?;

    let peak_dirs_shape = peak_dirs_ds.shape();
    let peak_values_shape = peak_values_ds.shape();
    let peak_indices_shape = peak_indices_ds.shape();
    if peak_dirs_shape.len() != 5 || peak_dirs_shape[4] != 3 {
        return Err(OdxError::Format(format!(
            "PAM peak_dirs must have shape (X,Y,Z,N,3), got {:?}",
            peak_dirs_shape
        )));
    }
    if peak_values_shape.len() != 4 {
        return Err(OdxError::Format(format!(
            "PAM peak_values must have shape (X,Y,Z,N), got {:?}",
            peak_values_shape
        )));
    }
    if peak_indices_shape != peak_values_shape {
        return Err(OdxError::Format(format!(
            "PAM peak_indices shape {:?} does not match peak_values {:?}",
            peak_indices_shape, peak_values_shape
        )));
    }
    if peak_dirs_shape[..4] != peak_values_shape[..] {
        return Err(OdxError::Format(format!(
            "PAM peak_dirs shape {:?} does not match peak_values {:?}",
            peak_dirs_shape, peak_values_shape
        )));
    }

    let dims = [
        peak_values_shape[0] as u64,
        peak_values_shape[1] as u64,
        peak_values_shape[2] as u64,
    ];
    let npeaks = peak_values_shape[3];
    let nvoxels_total = peak_values_shape[0] * peak_values_shape[1] * peak_values_shape[2];

    let affine = if pam.link_exists("affine") {
        let vals = pam.dataset("affine")?.read_raw::<f64>()?;
        if vals.len() != 16 {
            return Err(OdxError::Format(format!(
                "PAM affine dataset must contain 16 elements, got {}",
                vals.len()
            )));
        }
        [
            [vals[0], vals[1], vals[2], vals[3]],
            [vals[4], vals[5], vals[6], vals[7]],
            [vals[8], vals[9], vals[10], vals[11]],
            [vals[12], vals[13], vals[14], vals[15]],
        ]
    } else {
        crate::header::Header::identity_affine()
    };

    let peak_dirs = peak_dirs_ds.read_raw::<f64>()?;
    let peak_values = peak_values_ds.read_raw::<f64>()?;
    let peak_indices = peak_indices_ds.read_raw::<i32>()?;

    let sphere_flat = sphere_vertices_ds.read_raw::<f64>()?;
    let sphere_shape = sphere_vertices_ds.shape();
    if sphere_shape.len() != 2 || sphere_shape[1] != 3 {
        return Err(OdxError::Format(format!(
            "PAM sphere_vertices must have shape (M,3), got {:?}",
            sphere_shape
        )));
    }
    let sphere_vertices = sphere_flat
        .chunks_exact(3)
        .map(|row| [row[0] as f32, row[1] as f32, row[2] as f32])
        .collect::<Vec<_>>();

    let qa = if pam.link_exists("qa") {
        let ds = pam.dataset("qa")?;
        if ds.shape() != peak_values_shape {
            return Err(OdxError::Format(format!(
                "PAM qa shape {:?} does not match peak_values {:?}",
                ds.shape(),
                peak_values_shape
            )));
        }
        Some(ds.read_raw::<f64>()?)
    } else {
        None
    };

    let gfa = if pam.link_exists("gfa") {
        let ds = pam.dataset("gfa")?;
        let shape = ds.shape();
        if shape != peak_values_shape[..3] {
            return Err(OdxError::Format(format!(
                "PAM gfa shape {:?} does not match spatial shape {:?}",
                shape,
                &peak_values_shape[..3]
            )));
        }
        Some(ds.read_raw::<f64>()?)
    } else {
        None
    };

    let shm_coeff = if pam.link_exists("shm_coeff") {
        let ds = pam.dataset("shm_coeff")?;
        let shape = ds.shape();
        if shape.len() != 4 || shape[..3] != peak_values_shape[..3] {
            return Err(OdxError::Format(format!(
                "PAM shm_coeff must have shape (X,Y,Z,K), got {:?}",
                shape
            )));
        }
        Some((ds.read_raw::<f64>()?, shape[3]))
    } else {
        None
    };

    let odf = if pam.link_exists("odf") {
        let ds = pam.dataset("odf")?;
        let shape = ds.shape();
        if shape.len() != 4 || shape[..3] != peak_values_shape[..3] {
            return Err(OdxError::Format(format!(
                "PAM odf must have shape (X,Y,Z,M), got {:?}",
                shape
            )));
        }
        Some((ds.read_raw::<f64>()?, shape[3]))
    } else {
        None
    };

    let mut generic_dpf = Vec::new();
    let mut generic_dpv = Vec::new();
    let reserved: HashSet<&str> = [
        "peak_dirs",
        "peak_values",
        "peak_indices",
        "sphere_vertices",
        "affine",
        "shm_coeff",
        "B",
        "gfa",
        "qa",
        "odf",
        "total_weight",
        "ang_thr",
    ]
    .into_iter()
    .collect();
    for name in pam.member_names()? {
        if reserved.contains(name.as_str()) {
            continue;
        }
        let ds = pam.dataset(&name)?;
        let shape = ds.shape();
        if shape == peak_values_shape {
            generic_dpf.push((name, ds.read_raw::<f64>()?));
        } else if shape == peak_values_shape[..3] {
            generic_dpv.push((name, ds.read_raw::<f64>()?));
        }
    }

    let mut mask = vec![0u8; nvoxels_total];
    let mut offsets = vec![0u32];
    let mut directions = Vec::<[f32; 3]>::new();
    let mut amplitudes = Vec::<f32>::new();
    let mut pam_peak_indices = Vec::<i32>::new();
    let mut qa_sparse = qa.as_ref().map(|_| Vec::<f32>::new());
    let mut generic_dpf_sparse = generic_dpf
        .iter()
        .map(|(name, _)| (name.clone(), Vec::<f32>::new()))
        .collect::<Vec<_>>();

    for voxel in 0..nvoxels_total {
        let mut count = 0u32;
        for peak in 0..npeaks {
            let value_idx = voxel * npeaks + peak;
            let value = peak_values[value_idx] as f32;
            if value <= 0.0 {
                continue;
            }
            if count == 0 {
                mask[voxel] = 1;
            }
            let dir_base = value_idx * 3;
            let pam_dir = [
                peak_dirs[dir_base] as f32,
                peak_dirs[dir_base + 1] as f32,
                peak_dirs[dir_base + 2] as f32,
            ];
            directions.push(rotate_dir_pam_to_ras(pam_dir, &affine));
            amplitudes.push(value);
            pam_peak_indices.push(peak_indices[value_idx]);
            if let Some(ref mut qa_out) = qa_sparse {
                qa_out.push(qa.as_ref().unwrap()[value_idx] as f32);
            }
            for (metric_idx, (_, values)) in generic_dpf.iter().enumerate() {
                generic_dpf_sparse[metric_idx]
                    .1
                    .push(values[value_idx] as f32);
            }
            count += 1;
        }
        if count > 0 {
            let next = offsets.last().copied().unwrap_or(0) + count;
            offsets.push(next);
        }
    }

    let nb_voxels = mask.iter().filter(|&&v| v != 0).count();
    let nb_peaks = directions.len();

    let mut dpv = HashMap::new();
    if let Some(gfa_values) = gfa {
        let masked = sparse_from_dense_scalar(&gfa_values, &mask)
            .into_iter()
            .map(|v| v as f32)
            .collect::<Vec<_>>();
        dpv.insert(
            "gfa".into(),
            DataArray::owned_bytes(vec_to_bytes(masked), 1, DType::Float32),
        );
    }
    for (name, values) in generic_dpv {
        let masked = sparse_from_dense_scalar(&values, &mask)
            .into_iter()
            .map(|v| v as f32)
            .collect::<Vec<_>>();
        dpv.insert(
            name,
            DataArray::owned_bytes(vec_to_bytes(masked), 1, DType::Float32),
        );
    }

    let mut dpf = HashMap::new();
    dpf.insert(
        "amplitude".into(),
        DataArray::owned_bytes(vec_to_bytes(amplitudes), 1, DType::Float32),
    );
    dpf.insert(
        "pam_peak_index".into(),
        DataArray::owned_bytes(vec_to_bytes(pam_peak_indices), 1, DType::Int32),
    );
    if let Some(values) = qa_sparse {
        dpf.insert(
            "qa".into(),
            DataArray::owned_bytes(vec_to_bytes(values), 1, DType::Float32),
        );
    }
    for (name, values) in generic_dpf_sparse {
        dpf.insert(
            name,
            DataArray::owned_bytes(vec_to_bytes(values), 1, DType::Float32),
        );
    }

    let mut sh = HashMap::new();
    let mut sh_order = None;
    let mut sh_basis = None;
    let mut canonical_dense = None;
    let mut extra = HashMap::new();

    if let Some((coeffs, ncols)) = shm_coeff {
        let masked = sparse_from_dense_rows(&coeffs, &mask, ncols)
            .into_iter()
            .map(|v| v as f32)
            .collect::<Vec<_>>();
        sh.insert(
            "coefficients".into(),
            DataArray::owned_bytes(vec_to_bytes(masked), ncols, DType::Float32),
        );
        sh_order = infer_sh_order(ncols);
        sh_basis = Some("descoteaux07".into());
        canonical_dense = Some(CanonicalDenseRepresentation::Sh);
        extra.insert(
            PAM_BASIS_ASSUMED.into(),
            Value::String("descoteaux07".into()),
        );
    }

    let mut odf_arrays = HashMap::new();
    let mut odf_sample_domain = None;
    if let Some((values, ncols)) = odf {
        let masked = sparse_from_dense_rows(&values, &mask, ncols)
            .into_iter()
            .map(|v| v as f32)
            .collect::<Vec<_>>();
        odf_arrays.insert(
            "amplitudes".into(),
            DataArray::owned_bytes(vec_to_bytes(masked), ncols, DType::Float32),
        );
        odf_sample_domain = Some("hemisphere".into());
        canonical_dense.get_or_insert(CanonicalDenseRepresentation::Odf);
    }

    if pam.link_exists("total_weight") {
        let weight = pam.dataset("total_weight")?.read_raw::<f64>()?;
        if let Some(value) = weight.first() {
            extra.insert(
                "_ODX_PAM_TOTAL_WEIGHT".into(),
                Value::Number(Number::from_f64(*value).unwrap_or_else(|| Number::from(0))),
            );
        }
    }
    if pam.link_exists("ang_thr") {
        let ang = pam.dataset("ang_thr")?.read_raw::<f64>()?;
        if let Some(value) = ang.first() {
            extra.insert(
                "_ODX_PAM_ANG_THR".into(),
                Value::Number(Number::from_f64(*value).unwrap_or_else(|| Number::from(0))),
            );
        }
    }
    extra.insert("_ODX_PAM_VERSION".into(), Value::String(PAM_VERSION.into()));

    let header = crate::header::Header {
        voxel_to_rasmm: affine,
        dimensions: dims,
        nb_voxels: nb_voxels as u64,
        nb_peaks: nb_peaks as u64,
        nb_sphere_vertices: Some(sphere_vertices.len() as u64),
        nb_sphere_faces: None,
        sh_order,
        sh_basis,
        canonical_dense_representation: canonical_dense,
        sphere_id: None,
        odf_sample_domain,
        array_quantization: HashMap::new(),
        extra,
    };

    Ok(OdxDataset::from_parts(OdxParts {
        header,
        mask_backing: MmapBacking::Owned(mask),
        offsets_backing: MmapBacking::Owned(vec_to_bytes(offsets)),
        directions_backing: MmapBacking::Owned(vec_to_bytes(directions)),
        sphere_vertices: Some(MmapBacking::Owned(vec_to_bytes(sphere_vertices))),
        sphere_faces: None,
        odf: odf_arrays,
        sh,
        dpv,
        dpf,
        groups: HashMap::new(),
        dpg: HashMap::new(),
        tempdir: None,
    }))
}

#[cfg(feature = "pam5")]
pub fn save_pam5(odx: &OdxDataset, path: &Path, _options: &PamWriteOptions) -> Result<()> {
    use hdf5_metno::types::VarLenUnicode;
    use hdf5_metno::File;

    let file = File::create(path)?;
    let version = VarLenUnicode::from_str(PAM_VERSION)
        .map_err(|err| OdxError::Format(format!("invalid PAM version string: {err}")))?;
    file.new_attr::<VarLenUnicode>()
        .shape(())
        .create("version")?
        .write_scalar(&version)?;
    let pam = file.create_group("pam")?;

    let dims = [
        odx.header().dimensions[0] as usize,
        odx.header().dimensions[1] as usize,
        odx.header().dimensions[2] as usize,
    ];
    let nvoxels_total = dims[0] * dims[1] * dims[2];
    let sparse_indices = sparse_voxel_indices(odx.mask());
    if sparse_indices.len() != odx.nb_voxels() {
        return Err(OdxError::Format(format!(
            "mask contains {} voxels but ODX reports NB_VOXELS={}",
            sparse_indices.len(),
            odx.nb_voxels()
        )));
    }
    let max_peaks = (0..odx.nb_voxels())
        .map(|idx| odx.peaks_per_voxel(idx))
        .max()
        .unwrap_or(0);

    let sphere_vertices = export_sphere_vertices(odx)?;
    write_dataset_f32_2d(
        &pam,
        "sphere_vertices",
        sphere_vertices.as_flat_slice(),
        sphere_vertices.nrows(),
        3,
    )?;
    if let Some(affine) = odx.header().extra.get("_ODX_PAM_VERSION") {
        let _ = affine;
    }
    write_dataset_f64_2d(
        &pam,
        "affine",
        &flatten_affine(odx.header().voxel_to_rasmm),
        4,
        4,
    )?;

    let mut peak_dirs = vec![0.0f32; nvoxels_total * max_peaks * 3];
    let mut peak_values = vec![0.0f32; nvoxels_total * max_peaks];
    let mut peak_indices = vec![-1i32; nvoxels_total * max_peaks];

    let preserved_indices = reusable_peak_indices(odx, &sphere_vertices)?;
    let mut generic_dpf_dense = HashMap::<String, Vec<f32>>::new();
    for (name, info) in odx.iter_dpf() {
        if info.ncols != 1 || matches!(name, "amplitude" | "qa" | "pam_peak_index") {
            continue;
        }
        generic_dpf_dense.insert(name.to_string(), vec![0.0; nvoxels_total * max_peaks]);
    }
    let mut qa_dense = odx
        .scalar_dpf_f32("qa")
        .ok()
        .map(|_| vec![0.0f32; nvoxels_total * max_peaks]);
    let generic_dpf_sparse = generic_dpf_dense
        .keys()
        .map(|name| (name.clone(), odx.scalar_dpf_f32(name)))
        .collect::<HashMap<_, _>>();

    let amplitude = odx.scalar_dpf_f32("amplitude")?;
    let orientation = orientation_matrix(&odx.header().voxel_to_rasmm);
    let mut peak_cursor = 0usize;
    for (row, &full_idx) in sparse_indices.iter().enumerate() {
        let voxel_start = peak_cursor;
        let voxel_count = odx.peaks_per_voxel(row);
        for local_peak in 0..voxel_count {
            let sparse_peak = voxel_start + local_peak;
            let dense_idx = full_idx * max_peaks + local_peak;
            peak_values[dense_idx] = amplitude[sparse_peak];

            let pam_dir = rotate_dir_ras_to_pam(odx.directions()[sparse_peak], &orientation);
            let pam_peak_index = preserved_indices[sparse_peak].unwrap_or_else(|| {
                let (index, quantized) =
                    quantize_to_sphere(pam_dir, sphere_vertices.as_flat_slice());
                let base = dense_idx * 3;
                peak_dirs[base..base + 3].copy_from_slice(&quantized);
                index as i32
            });
            let base = dense_idx * 3;
            if peak_dirs[base..base + 3] == [0.0, 0.0, 0.0] {
                peak_dirs[base..base + 3].copy_from_slice(&pam_dir);
            }
            peak_indices[dense_idx] = pam_peak_index;

            if let Some(ref mut qa_out) = qa_dense {
                let qa = odx.scalar_dpf_f32("qa")?;
                qa_out[dense_idx] = qa[sparse_peak];
            }
            for (name, dense) in &mut generic_dpf_dense {
                let sparse = generic_dpf_sparse
                    .get(name)
                    .and_then(|values| values.as_ref().ok())
                    .ok_or_else(|| {
                        OdxError::Format(format!("missing sparse DPF data for '{name}'"))
                    })?;
                dense[dense_idx] = sparse[sparse_peak];
            }
        }
        peak_cursor += voxel_count;
    }

    write_dataset_f32_5d(
        &pam,
        "peak_dirs",
        &peak_dirs,
        dims[0],
        dims[1],
        dims[2],
        max_peaks,
        3,
    )?;
    write_dataset_f32_4d(
        &pam,
        "peak_values",
        &peak_values,
        dims[0],
        dims[1],
        dims[2],
        max_peaks,
    )?;
    write_dataset_i32_4d(
        &pam,
        "peak_indices",
        &peak_indices,
        dims[0],
        dims[1],
        dims[2],
        max_peaks,
    )?;

    if let Some(values) = qa_dense {
        write_dataset_f32_4d(&pam, "qa", &values, dims[0], dims[1], dims[2], max_peaks)?;
    }
    for (name, values) in generic_dpf_dense {
        write_dataset_f32_4d(&pam, &name, &values, dims[0], dims[1], dims[2], max_peaks)?;
    }

    let mut dense_dpv = HashMap::<String, Vec<f32>>::new();
    for (name, info) in odx.iter_dpv() {
        if info.ncols != 1 {
            continue;
        }
        let sparse = odx.scalar_dpv_f32(name)?;
        let mut dense = vec![0.0f32; nvoxels_total];
        for (row, &full_idx) in sparse_indices.iter().enumerate() {
            dense[full_idx] = sparse[row];
        }
        dense_dpv.insert(name.to_string(), dense);
    }
    if let Some(gfa) = dense_dpv.remove("gfa") {
        write_dataset_f32_3d(&pam, "gfa", &gfa, dims[0], dims[1], dims[2])?;
    }
    for (name, values) in dense_dpv {
        write_dataset_f32_3d(&pam, &name, &values, dims[0], dims[1], dims[2])?;
    }

    if should_export_sh(odx) {
        if let Ok(sh) = odx.sh::<f32>("coefficients") {
            let mut dense = vec![0.0f32; nvoxels_total * sh.ncols()];
            for (row, &full_idx) in sparse_indices.iter().enumerate() {
                let src = sh.row(row);
                let start = full_idx * sh.ncols();
                dense[start..start + sh.ncols()].copy_from_slice(src);
            }
            write_dataset_f32_4d(
                &pam,
                "shm_coeff",
                &dense,
                dims[0],
                dims[1],
                dims[2],
                sh.ncols(),
            )?;
        }
    }

    if let Ok(odf) = odx.odf::<f32>("amplitudes") {
        if odf.ncols() == sphere_vertices.nrows() {
            let mut dense = vec![0.0f32; nvoxels_total * odf.ncols()];
            for (row, &full_idx) in sparse_indices.iter().enumerate() {
                let src = odf.row(row);
                let start = full_idx * odf.ncols();
                dense[start..start + odf.ncols()].copy_from_slice(src);
            }
            write_dataset_f32_4d(&pam, "odf", &dense, dims[0], dims[1], dims[2], odf.ncols())?;
        }
    }

    if let Some(value) = odx
        .header()
        .extra
        .get("_ODX_PAM_TOTAL_WEIGHT")
        .and_then(|value| value.as_f64())
    {
        write_dataset_f64_1d(&pam, "total_weight", &[value])?;
    }
    if let Some(value) = odx
        .header()
        .extra
        .get("_ODX_PAM_ANG_THR")
        .and_then(|value| value.as_f64())
    {
        write_dataset_f64_1d(&pam, "ang_thr", &[value])?;
    }

    Ok(())
}

#[cfg(feature = "pam5")]
fn sparse_from_dense_rows(values: &[f64], mask: &[u8], ncols: usize) -> Vec<f64> {
    let mut out = Vec::new();
    for (idx, &flag) in mask.iter().enumerate() {
        if flag == 0 {
            continue;
        }
        let start = idx * ncols;
        out.extend_from_slice(&values[start..start + ncols]);
    }
    out
}

#[cfg(feature = "pam5")]
fn sparse_from_dense_scalar(values: &[f64], mask: &[u8]) -> Vec<f64> {
    let mut out = Vec::new();
    for (idx, &flag) in mask.iter().enumerate() {
        if flag != 0 {
            out.push(values[idx]);
        }
    }
    out
}

#[cfg(feature = "pam5")]
fn sparse_voxel_indices(mask: &[u8]) -> Vec<usize> {
    mask.iter()
        .enumerate()
        .filter_map(|(idx, &value)| (value != 0).then_some(idx))
        .collect()
}

#[cfg(feature = "pam5")]
fn infer_sh_order(ncols: usize) -> Option<u64> {
    let mut order = 0u64;
    loop {
        let expected = ((order + 1) * (order + 2) / 2) as usize;
        if expected == ncols {
            return Some(order);
        }
        if expected > ncols {
            return None;
        }
        order += 1;
    }
}

#[cfg(feature = "pam5")]
fn orientation_matrix(affine: &[[f64; 4]; 4]) -> [[f32; 3]; 3] {
    let mut out = [[0.0f32; 3]; 3];
    for col in 0..3 {
        let norm = (affine[0][col] * affine[0][col]
            + affine[1][col] * affine[1][col]
            + affine[2][col] * affine[2][col])
            .sqrt();
        if norm > 0.0 {
            out[0][col] = (affine[0][col] / norm) as f32;
            out[1][col] = (affine[1][col] / norm) as f32;
            out[2][col] = (affine[2][col] / norm) as f32;
        }
    }
    out
}

#[cfg(feature = "pam5")]
fn rotate_dir_pam_to_ras(dir: [f32; 3], affine: &[[f64; 4]; 4]) -> [f32; 3] {
    normalize_dir(apply_mat3(&orientation_matrix(affine), dir))
}

#[cfg(feature = "pam5")]
fn rotate_dir_ras_to_pam(dir: [f32; 3], orientation: &[[f32; 3]; 3]) -> [f32; 3] {
    let transposed = [
        [orientation[0][0], orientation[1][0], orientation[2][0]],
        [orientation[0][1], orientation[1][1], orientation[2][1]],
        [orientation[0][2], orientation[1][2], orientation[2][2]],
    ];
    normalize_dir(apply_mat3(&transposed, dir))
}

#[cfg(feature = "pam5")]
fn apply_mat3(mat: &[[f32; 3]; 3], dir: [f32; 3]) -> [f32; 3] {
    [
        mat[0][0] * dir[0] + mat[0][1] * dir[1] + mat[0][2] * dir[2],
        mat[1][0] * dir[0] + mat[1][1] * dir[1] + mat[1][2] * dir[2],
        mat[2][0] * dir[0] + mat[2][1] * dir[1] + mat[2][2] * dir[2],
    ]
}

#[cfg(feature = "pam5")]
fn normalize_dir(dir: [f32; 3]) -> [f32; 3] {
    let norm = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    if norm <= f32::EPSILON {
        [0.0, 0.0, 0.0]
    } else {
        [dir[0] / norm, dir[1] / norm, dir[2] / norm]
    }
}

#[cfg(feature = "pam5")]
fn reusable_peak_indices(odx: &OdxDataset, sphere: &SphereRows<'_>) -> Result<Vec<Option<i32>>> {
    let indices = match odx.dpf::<i32>("pam_peak_index") {
        Ok(values) => values,
        Err(_) => return Ok(vec![None; odx.nb_peaks()]),
    };
    let original = if let Some(vertices) = odx.sphere_vertices() {
        vertices
    } else {
        return Ok(vec![None; odx.nb_peaks()]);
    };
    if original.len() != sphere.nrows() {
        return Ok(vec![None; odx.nb_peaks()]);
    }
    let matches = original
        .iter()
        .zip(sphere.rows())
        .all(|(left, right)| approx_dir_eq(*left, right));
    if !matches {
        return Ok(vec![None; odx.nb_peaks()]);
    }
    Ok(indices.as_flat_slice().iter().copied().map(Some).collect())
}

#[cfg(feature = "pam5")]
fn approx_dir_eq(left: [f32; 3], right: [f32; 3]) -> bool {
    (left[0] - right[0]).abs() <= 1e-5
        && (left[1] - right[1]).abs() <= 1e-5
        && (left[2] - right[2]).abs() <= 1e-5
}

#[cfg(feature = "pam5")]
fn should_export_sh(odx: &OdxDataset) -> bool {
    match odx.header().sh_basis.as_deref() {
        Some("descoteaux07") => true,
        _ => {
            odx.header()
                .extra
                .get(PAM_BASIS_ASSUMED)
                .and_then(|value| value.as_str())
                == Some("descoteaux07")
        }
    }
}

#[cfg(feature = "pam5")]
fn quantize_to_sphere(dir: [f32; 3], sphere: &[[f32; 3]]) -> (usize, [f32; 3]) {
    let mut best_idx = 0usize;
    let mut best_abs_dot = f32::NEG_INFINITY;
    let mut best_sign = 1.0f32;
    for (idx, &candidate) in sphere.iter().enumerate() {
        let dot = dir[0] * candidate[0] + dir[1] * candidate[1] + dir[2] * candidate[2];
        let abs_dot = dot.abs();
        if abs_dot > best_abs_dot {
            best_idx = idx;
            best_abs_dot = abs_dot;
            best_sign = if dot < 0.0 { -1.0 } else { 1.0 };
        }
    }
    let base = sphere[best_idx];
    (
        best_idx,
        [
            base[0] * best_sign,
            base[1] * best_sign,
            base[2] * best_sign,
        ],
    )
}

#[cfg(feature = "pam5")]
fn export_sphere_vertices<'a>(odx: &'a OdxDataset) -> Result<SphereRows<'a>> {
    if let Some(vertices) = odx.sphere_vertices() {
        return Ok(SphereRows::Borrowed(vertices));
    }
    if odx.header().sphere_id.as_deref() == Some("dsistudio_odf8") {
        return Ok(SphereRows::Borrowed(
            crate::formats::dsistudio_odf8::hemisphere_vertices_ras(),
        ));
    }
    Ok(SphereRows::Owned(load_repulsion724_hemisphere()))
}

#[cfg(feature = "pam5")]
fn flatten_affine(affine: [[f64; 4]; 4]) -> Vec<f64> {
    affine.into_iter().flat_map(|row| row.into_iter()).collect()
}

#[cfg(feature = "pam5")]
fn load_repulsion724_hemisphere() -> Vec<[f32; 3]> {
    include_bytes!("pam_repulsion724_hemisphere_f32.bin")
        .chunks_exact(12)
        .map(|chunk| {
            let x = f32::from_le_bytes(chunk[0..4].try_into().unwrap());
            let y = f32::from_le_bytes(chunk[4..8].try_into().unwrap());
            let z = f32::from_le_bytes(chunk[8..12].try_into().unwrap());
            [x, y, z]
        })
        .collect()
}

#[cfg(feature = "pam5")]
enum SphereRows<'a> {
    Borrowed(&'a [[f32; 3]]),
    Owned(Vec<[f32; 3]>),
}

#[cfg(feature = "pam5")]
impl<'a> SphereRows<'a> {
    fn as_flat_slice(&self) -> &[[f32; 3]] {
        match self {
            Self::Borrowed(values) => values,
            Self::Owned(values) => values,
        }
    }

    fn rows(&self) -> impl Iterator<Item = [f32; 3]> + '_ {
        self.as_flat_slice().iter().copied()
    }

    fn nrows(&self) -> usize {
        self.as_flat_slice().len()
    }
}

#[cfg(feature = "pam5")]
fn write_dataset_f32_5d(
    group: &hdf5_metno::Group,
    name: &str,
    data: &[f32],
    d0: usize,
    d1: usize,
    d2: usize,
    d3: usize,
    d4: usize,
) -> Result<()> {
    let ds = group
        .new_dataset::<f32>()
        .shape([d0, d1, d2, d3, d4])
        .create(name)?;
    ds.write_raw(data)?;
    Ok(())
}

#[cfg(feature = "pam5")]
fn write_dataset_f32_4d(
    group: &hdf5_metno::Group,
    name: &str,
    data: &[f32],
    d0: usize,
    d1: usize,
    d2: usize,
    d3: usize,
) -> Result<()> {
    let ds = group
        .new_dataset::<f32>()
        .shape([d0, d1, d2, d3])
        .create(name)?;
    ds.write_raw(data)?;
    Ok(())
}

#[cfg(feature = "pam5")]
fn write_dataset_i32_4d(
    group: &hdf5_metno::Group,
    name: &str,
    data: &[i32],
    d0: usize,
    d1: usize,
    d2: usize,
    d3: usize,
) -> Result<()> {
    let ds = group
        .new_dataset::<i32>()
        .shape([d0, d1, d2, d3])
        .create(name)?;
    ds.write_raw(data)?;
    Ok(())
}

#[cfg(feature = "pam5")]
fn write_dataset_f32_3d(
    group: &hdf5_metno::Group,
    name: &str,
    data: &[f32],
    d0: usize,
    d1: usize,
    d2: usize,
) -> Result<()> {
    let ds = group
        .new_dataset::<f32>()
        .shape([d0, d1, d2])
        .create(name)?;
    ds.write_raw(data)?;
    Ok(())
}

#[cfg(feature = "pam5")]
fn write_dataset_f64_2d(
    group: &hdf5_metno::Group,
    name: &str,
    data: &[f64],
    d0: usize,
    d1: usize,
) -> Result<()> {
    let ds = group.new_dataset::<f64>().shape([d0, d1]).create(name)?;
    ds.write_raw(data)?;
    Ok(())
}

#[cfg(feature = "pam5")]
fn write_dataset_f32_2d(
    group: &hdf5_metno::Group,
    name: &str,
    data: &[[f32; 3]],
    d0: usize,
    d1: usize,
) -> Result<()> {
    let flat = data
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    let ds = group.new_dataset::<f32>().shape([d0, d1]).create(name)?;
    ds.write_raw(&flat)?;
    Ok(())
}

#[cfg(feature = "pam5")]
fn write_dataset_f64_1d(group: &hdf5_metno::Group, name: &str, data: &[f64]) -> Result<()> {
    let ds = group
        .new_dataset::<f64>()
        .shape([data.len()])
        .create(name)?;
    ds.write_raw(data)?;
    Ok(())
}
