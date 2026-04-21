use std::collections::HashMap;
use std::path::Path;

use nalgebra::{Matrix3, SymmetricEigen, Vector3};
use serde_json::{Number, Value};
use statrs::function::gamma::gamma;

use crate::data_array::DataArray;
use crate::dtype::DType;
use crate::error::{OdxError, Result};
use crate::formats::{dsistudio_odf8, mrtrix};
use crate::header::{CanonicalDenseRepresentation, Header};
use crate::mmap_backing::{vec_into_bytes, MmapBacking};
use crate::mrtrix_sh;
use crate::odx_file::{OdxDataset, OdxParts};

const TORTOISE_SOURCE_FORMAT: &str = "tortoise_mapmri";
const TORTOISE_PROJECTION_SPHERE_ID: &str = "dsistudio_odf8_hemisphere";
const DEFAULT_SH_LMAX: usize = 8;
const TORTOISE_ODF_S: usize = 2;
const TORTOISE_DISPLAY_SCALE_PERCENTILE: f32 = 0.99;
const TORTOISE_DISPLAY_SCALE_CONVENTION: &str = "p99_voxel_max_odf_to_1";

const TORTOISE_NNMAX: [usize; 6] = [0, 6, 21, 49, 94, 160];
const TORTOISE_N1A: [usize; 161] = [
    0, 2, 0, 0, 1, 1, 0, 4, 0, 0, 3, 3, 1, 1, 0, 0, 2, 2, 0, 2, 1, 1, 6, 0, 0, 5, 5, 1, 1, 0, 0, 4,
    4, 2, 2, 0, 0, 4, 1, 1, 3, 3, 0, 3, 3, 2, 2, 1, 1, 2, 8, 0, 0, 7, 7, 1, 1, 0, 0, 6, 6, 2, 2, 0,
    0, 6, 1, 1, 5, 5, 3, 3, 0, 0, 5, 5, 2, 2, 1, 1, 4, 4, 0, 4, 4, 3, 3, 1, 1, 4, 2, 2, 3, 3, 2,
    10, 0, 0, 9, 9, 1, 1, 0, 0, 8, 8, 2, 2, 0, 0, 8, 1, 1, 7, 7, 3, 3, 0, 0, 7, 7, 2, 2, 1, 1, 6,
    6, 4, 4, 0, 0, 6, 6, 3, 3, 1, 1, 6, 2, 2, 5, 5, 0, 5, 5, 4, 4, 1, 1, 5, 5, 3, 3, 2, 2, 4, 4, 2,
    4, 3, 3,
];
const TORTOISE_N2A: [usize; 161] = [
    0, 0, 2, 0, 1, 0, 1, 0, 4, 0, 1, 0, 3, 0, 3, 1, 2, 0, 2, 1, 2, 1, 0, 6, 0, 1, 0, 5, 0, 5, 1, 2,
    0, 4, 0, 4, 2, 1, 4, 1, 3, 0, 3, 2, 1, 3, 1, 3, 2, 2, 0, 8, 0, 1, 0, 7, 0, 7, 1, 2, 0, 6, 0, 6,
    2, 1, 6, 1, 3, 0, 5, 0, 5, 3, 2, 1, 5, 1, 5, 2, 4, 0, 4, 3, 1, 4, 1, 4, 3, 2, 4, 2, 3, 2, 3, 0,
    10, 0, 1, 0, 9, 0, 9, 1, 2, 0, 8, 0, 8, 2, 1, 8, 1, 3, 0, 7, 0, 7, 3, 2, 1, 7, 1, 7, 2, 4, 0,
    6, 0, 6, 4, 3, 1, 6, 1, 6, 3, 2, 6, 2, 5, 0, 5, 4, 1, 5, 1, 5, 4, 3, 2, 5, 2, 5, 3, 4, 2, 4, 3,
    4, 3,
];
const TORTOISE_N3A: [usize; 161] = [
    0, 0, 0, 2, 0, 1, 1, 0, 0, 4, 0, 1, 0, 3, 1, 3, 0, 2, 2, 1, 1, 2, 0, 0, 6, 0, 1, 0, 5, 1, 5, 0,
    2, 0, 4, 2, 4, 1, 1, 4, 0, 3, 3, 1, 2, 1, 3, 2, 3, 2, 0, 0, 8, 0, 1, 0, 7, 1, 7, 0, 2, 0, 6, 2,
    6, 1, 1, 6, 0, 3, 0, 5, 3, 5, 1, 2, 1, 5, 2, 5, 0, 4, 4, 1, 3, 1, 4, 3, 4, 2, 2, 4, 2, 3, 3, 0,
    0, 10, 0, 1, 0, 9, 1, 9, 0, 2, 0, 8, 2, 8, 1, 1, 8, 0, 3, 0, 7, 3, 7, 1, 2, 1, 7, 2, 7, 0, 4,
    0, 6, 4, 6, 1, 3, 1, 6, 3, 6, 2, 2, 6, 0, 5, 5, 1, 4, 1, 5, 4, 5, 2, 3, 2, 5, 3, 5, 2, 4, 4, 3,
    3, 4,
];

#[derive(Debug, Clone)]
struct TortoiseOdfTerm {
    alpha_pow: usize,
    beta_pow: usize,
    gamma_pow: usize,
    scale: f64,
}

#[derive(Debug, Clone)]
struct TortoiseCoeffPlan {
    scale: f64,
    terms: Vec<TortoiseOdfTerm>,
}

#[derive(Debug, Clone)]
struct TortoiseOdfPlan {
    coeffs: Vec<TortoiseCoeffPlan>,
    max_power: usize,
    s: usize,
}

pub fn load_tortoise_mapmri(
    coeff_path: &Path,
    tensor_path: &Path,
    uvec_path: &Path,
) -> Result<OdxDataset> {
    let (coeff_dims, coeff_affine, coeff_data) = mrtrix::load_nifti_f32_volume(coeff_path)?;
    let (tensor_dims, tensor_affine, tensor_data) = mrtrix::load_nifti_f32_volume(tensor_path)?;
    let (uvec_dims, uvec_affine, uvec_data) = mrtrix::load_nifti_f32_volume(uvec_path)?;

    let (dims3, ncoeffs) = normalize_trailing_channels(
        &coeff_dims,
        coeff_path,
        &[22, 50, 95, 161],
        "TORTOISE MAPMRI coefficients",
    )?;
    let (_, ntensor) =
        normalize_trailing_channels(&tensor_dims, tensor_path, &[6], "TORTOISE tensor image")?;
    let (_, nuvec) =
        normalize_trailing_channels(&uvec_dims, uvec_path, &[3], "TORTOISE uvec image")?;
    if ntensor != 6 {
        return Err(OdxError::Format(format!(
            "TORTOISE tensor image '{}' must have 6 channels, found {ntensor}",
            tensor_path.display()
        )));
    }
    if nuvec != 3 {
        return Err(OdxError::Format(format!(
            "TORTOISE uvec image '{}' must have 3 channels, found {nuvec}",
            uvec_path.display()
        )));
    }
    ensure_affine_match(&coeff_affine, &tensor_affine, tensor_path)?;
    ensure_affine_match(&coeff_affine, &uvec_affine, uvec_path)?;

    let radial_order = tortoise_order_from_coeffs(ncoeffs)?;
    let sample_dirs = dsistudio_odf8::hemisphere_vertices_ras();
    let sh_lmax = mrtrix_sh::resolve_lmax_for_directions(sample_dirs, None, DEFAULT_SH_LMAX);
    let sh_ncoeffs = mrtrix_sh::ncoeffs_for_lmax(sh_lmax);
    let sh_fit_plan = mrtrix_sh::RowFitPlan::for_amplitudes(sample_dirs, sh_lmax)?;
    let odf_plan = tortoise_odf_plan(radial_order, TORTOISE_ODF_S)?;
    let voxel_count = dims3[0] * dims3[1] * dims3[2];

    let coeff_rows = collapse_rows(&coeff_data, voxel_count, ncoeffs)?;
    let tensor_rows = collapse_rows(&tensor_data, voxel_count, 6)?;
    let uvec_rows = collapse_rows(&uvec_data, voxel_count, 3)?;

    let candidate_voxels = coeff_rows.iter().filter(|row| row[0] != 0.0).count();
    let mut mask = vec![0u8; voxel_count];
    let mut coeffs = Vec::with_capacity(candidate_voxels * sh_ncoeffs);
    let mut anisotropic_power = Vec::with_capacity(candidate_voxels);
    let mut kept_voxels = 0usize;
    let mut excluded_zero_uvec = 0usize;
    let mut excluded_zero_tensor = 0usize;
    let mut voxel_maxima = Vec::with_capacity(candidate_voxels);
    let mut odf_row = vec![0.0f32; sample_dirs.len()];
    let mut sh_row = vec![0.0f32; sh_ncoeffs];

    for voxel in 0..voxel_count {
        let coeff_row = &coeff_rows[voxel];
        if coeff_row[0] == 0.0 {
            continue;
        }
        let uvec_row = &uvec_rows[voxel];
        if uvec_row.iter().all(|value| *value == 0.0) {
            excluded_zero_uvec += 1;
            continue;
        }
        let tensor_row = &tensor_rows[voxel];
        if tensor_row[0] + tensor_row[1] + tensor_row[2] == 0.0 {
            excluded_zero_tensor += 1;
            continue;
        }

        let rotation = tortoise_rotation_from_tensor(tensor_row)?;
        let mu = tortoise_mu_from_uvec(uvec_row)?;
        tortoise_odf_samples_into(
            &odf_plan,
            coeff_row,
            mu,
            &rotation,
            sample_dirs,
            &mut odf_row,
        )?;
        sh_fit_plan.apply_row_into(&odf_row, &mut sh_row);
        mask[voxel] = 1;
        kept_voxels += 1;
        voxel_maxima.push(odf_row.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        anisotropic_power.push(sh_row.iter().map(|value| value * value).sum::<f32>());
        coeffs.extend_from_slice(&sh_row);
    }

    let display_scale =
        robust_positive_percentile(&voxel_maxima, TORTOISE_DISPLAY_SCALE_PERCENTILE);
    if display_scale != 1.0 {
        for value in &mut coeffs {
            *value /= display_scale;
        }
        let power_scale = display_scale * display_scale;
        for value in &mut anisotropic_power {
            *value /= power_scale;
        }
    }

    let mut extra = HashMap::new();
    extra.insert(
        "_ODX_TORTOISE_SOURCE_FORMAT".into(),
        Value::String(TORTOISE_SOURCE_FORMAT.into()),
    );
    extra.insert(
        "_ODX_TORTOISE_UVEC_PATH".into(),
        Value::String(uvec_path.display().to_string()),
    );
    extra.insert(
        "_ODX_TORTOISE_TENSOR_PATH".into(),
        Value::String(tensor_path.display().to_string()),
    );
    extra.insert(
        "_ODX_TORTOISE_RADIAL_ORDER".into(),
        Value::Number(Number::from(radial_order as u64)),
    );
    extra.insert(
        "_ODX_TORTOISE_PROJECTION_SPHERE".into(),
        Value::String(TORTOISE_PROJECTION_SPHERE_ID.into()),
    );
    extra.insert(
        "_ODX_TORTOISE_SH_LMAX".into(),
        Value::Number(Number::from(sh_lmax as u64)),
    );
    extra.insert(
        "_ODX_TORTOISE_CANDIDATE_VOXELS".into(),
        Value::Number(Number::from(candidate_voxels as u64)),
    );
    extra.insert(
        "_ODX_TORTOISE_EXCLUDED_ZERO_UVEC_VOXELS".into(),
        Value::Number(Number::from(excluded_zero_uvec as u64)),
    );
    extra.insert(
        "_ODX_TORTOISE_EXCLUDED_ZERO_TENSOR_VOXELS".into(),
        Value::Number(Number::from(excluded_zero_tensor as u64)),
    );
    extra.insert(
        "_ODX_TORTOISE_GLOBAL_SCALE".into(),
        Value::Number(
            Number::from_f64(display_scale as f64)
                .expect("TORTOISE display scale should be finite"),
        ),
    );
    extra.insert(
        "_ODX_TORTOISE_SCALE_CONVENTION".into(),
        Value::String(TORTOISE_DISPLAY_SCALE_CONVENTION.into()),
    );
    Ok(OdxDataset::from_parts(OdxParts {
        header: Header {
            voxel_to_rasmm: coeff_affine,
            dimensions: [dims3[0] as u64, dims3[1] as u64, dims3[2] as u64],
            nb_voxels: kept_voxels as u64,
            nb_peaks: 0,
            nb_sphere_vertices: None,
            nb_sphere_faces: None,
            sh_order: Some(sh_lmax as u64),
            sh_basis: Some("tournier07".into()),
            canonical_dense_representation: Some(CanonicalDenseRepresentation::Sh),
            sphere_id: None,
            odf_sample_domain: None,
            array_quantization: HashMap::new(),
            extra,
        },
        mask_backing: MmapBacking::Owned(mask),
        offsets_backing: MmapBacking::Owned(vec_into_bytes(vec![0u32; kept_voxels + 1])),
        directions_backing: MmapBacking::Owned(vec_into_bytes(Vec::<[f32; 3]>::new())),
        sphere_vertices: None,
        sphere_faces: None,
        odf: HashMap::new(),
        sh: HashMap::from([(
            "coefficients".into(),
            DataArray::owned_bytes(vec_into_bytes(coeffs), sh_ncoeffs, DType::Float32),
        )]),
        dpv: HashMap::from([(
            "anisotropic_power".into(),
            DataArray::owned_bytes(vec_into_bytes(anisotropic_power), 1, DType::Float32),
        )]),
        dpf: HashMap::new(),
        groups: HashMap::new(),
        dpg: HashMap::new(),
        tempdir: None,
    }))
}

fn normalize_trailing_channels(
    dims: &[usize],
    path: &Path,
    allowed_channels: &[usize],
    label: &str,
) -> Result<([usize; 3], usize)> {
    match dims {
        [x, y, z, channels] if allowed_channels.contains(channels) => Ok(([*x, *y, *z], *channels)),
        [x, y, z, one, channels] if *one == 1 && allowed_channels.contains(channels) => {
            Ok(([*x, *y, *z], *channels))
        }
        _ => Err(OdxError::Format(format!(
            "{label} '{}' has unsupported shape {:?}",
            path.display(),
            dims
        ))),
    }
}

fn collapse_rows(data: &[f32], voxel_count: usize, ncols: usize) -> Result<Vec<&[f32]>> {
    if data.len() != voxel_count * ncols {
        return Err(OdxError::Format(format!(
            "dense payload length {} does not match {} voxels x {} columns",
            data.len(),
            voxel_count,
            ncols
        )));
    }
    Ok((0..voxel_count)
        .map(|voxel| &data[voxel * ncols..(voxel + 1) * ncols])
        .collect())
}

fn ensure_affine_match(
    expected: &[[f64; 4]; 4],
    actual: &[[f64; 4]; 4],
    path: &Path,
) -> Result<()> {
    const TOL: f64 = 1e-4;
    for row in 0..4 {
        for col in 0..4 {
            if (expected[row][col] - actual[row][col]).abs() > TOL {
                return Err(OdxError::Format(format!(
                    "affine mismatch for '{}'",
                    path.display()
                )));
            }
        }
    }
    Ok(())
}

fn tortoise_order_from_coeffs(ncoeffs: usize) -> Result<usize> {
    for order in (2..=10).step_by(2) {
        let nc = ((order / 2 + 1) * (order / 2 + 2) * (2 * order + 3)) / 6;
        if nc == ncoeffs {
            return Ok(order);
        }
    }
    Err(OdxError::Format(format!(
        "unable to infer TORTOISE MAPMRI radial order from {ncoeffs} coefficients"
    )))
}

fn tortoise_rotation_from_tensor(tensor: &[f32]) -> Result<Matrix3<f64>> {
    let tensor = Matrix3::new(
        tensor[0] as f64,
        tensor[3] as f64,
        tensor[4] as f64,
        tensor[3] as f64,
        tensor[1] as f64,
        tensor[5] as f64,
        tensor[4] as f64,
        tensor[5] as f64,
        tensor[2] as f64,
    );
    let eig = SymmetricEigen::new(tensor);
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));
    let mut v = Matrix3::zeros();
    for (dst, &src) in order.iter().enumerate() {
        v.set_column(dst, &eig.eigenvectors.column(src));
    }
    if v.determinant() < 0.0 {
        let col = -v.column(2).into_owned();
        v.set_column(2, &col);
    }
    if !v.iter().all(|value| value.is_finite()) {
        return Err(OdxError::Format(
            "non-finite tensor eigendecomposition while reading TORTOISE tensor image".into(),
        ));
    }
    Ok(v.transpose())
}

fn tortoise_mu_from_uvec(uvec: &[f32]) -> Result<[f64; 3]> {
    let norm =
        ((uvec[0] as f64).powi(2) + (uvec[1] as f64).powi(2) + (uvec[2] as f64).powi(2)).sqrt();
    if norm == 0.0 {
        return Err(OdxError::Format(
            "degenerate zero uvec encountered while evaluating TORTOISE MAPMRI".into(),
        ));
    }
    Ok([
        uvec[0] as f64 / (2.0 * norm),
        uvec[1] as f64 / (2.0 * norm),
        uvec[2] as f64 / (2.0 * norm),
    ])
}

fn tortoise_odf_samples_into(
    plan: &TortoiseOdfPlan,
    coeffs: &[f32],
    mu: [f64; 3],
    rotation: &Matrix3<f64>,
    dirs_ras: &[[f32; 3]],
    out: &mut [f32],
) -> Result<()> {
    if coeffs.len() != plan.coeffs.len() {
        return Err(OdxError::Format(format!(
            "TORTOISE coefficient row has {} values, expected {}",
            coeffs.len(),
            plan.coeffs.len()
        )));
    }
    if dirs_ras.len() != out.len() {
        return Err(OdxError::Format(format!(
            "TORTOISE ODF output row has {} values, expected {}",
            out.len(),
            dirs_ras.len()
        )));
    }

    let mux = mu[0];
    let muy = mu[1];
    let muz = mu[2];
    let mu_scale = (2f64.powf((2 - plan.s) as f64)
        * std::f64::consts::PI.powi(3)
        * (mux * mux * muy * muy * muz * muz))
        .sqrt();
    let mut alpha_pows = vec![1.0f64; plan.max_power + 1];
    let mut beta_pows = vec![1.0f64; plan.max_power + 1];
    let mut gamma_pows = vec![1.0f64; plan.max_power + 1];

    for (row, dir) in dirs_ras.iter().enumerate() {
        let local = rotate_row_vector(*dir, rotation);
        let rho = 1.0
            / ((local[0] / mux).powi(2) + (local[1] / muy).powi(2) + (local[2] / muz).powi(2))
                .sqrt();
        let alpha = 2.0 * rho * (local[0] / mux);
        let beta = 2.0 * rho * (local[1] / muy);
        let gamma_xyz = 2.0 * rho * (local[2] / muz);
        fill_powers(alpha, &mut alpha_pows);
        fill_powers(beta, &mut beta_pows);
        fill_powers(gamma_xyz, &mut gamma_pows);
        let const_term = rho.powf((3 + plan.s) as f64) / mu_scale;

        let mut acc = 0.0f64;
        for (coeff_idx, coeff_plan) in plan.coeffs.iter().enumerate() {
            let mut basis = 0.0f64;
            for term in &coeff_plan.terms {
                basis += term.scale
                    * alpha_pows[term.alpha_pow]
                    * beta_pows[term.beta_pow]
                    * gamma_pows[term.gamma_pow];
            }
            acc += const_term * coeff_plan.scale * basis * coeffs[coeff_idx] as f64;
        }
        out[row] = acc as f32;
    }
    Ok(())
}

fn tortoise_odf_plan(radial_order: usize, s: usize) -> Result<TortoiseOdfPlan> {
    let ind_mat = tortoise_index_matrix(radial_order)?;
    let mut coeffs = Vec::with_capacity(ind_mat.len());
    let mut max_power = 0usize;

    for [n1, n2, n3] in ind_mat {
        max_power = max_power.max(n1.max(n2).max(n3));
        let scale = (factorial(n1) * factorial(n2) * factorial(n3)).sqrt();
        let mut terms = Vec::new();
        for i in (0..=n1).step_by(2) {
            for j in (0..=n2).step_by(2) {
                for k in (0..=n3).step_by(2) {
                    let nn = n1 + n2 + n3 - i - j - k;
                    let sign = if ((i + j + k) / 2) % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    let denom = factorial(n1 - i)
                        * factorial(n2 - j)
                        * factorial(n3 - k)
                        * double_factorial(i)
                        * double_factorial(j)
                        * double_factorial(k);
                    terms.push(TortoiseOdfTerm {
                        alpha_pow: n1 - i,
                        beta_pow: n2 - j,
                        gamma_pow: n3 - k,
                        scale: sign * gamma((3.0 + s as f64 + nn as f64) / 2.0) / denom,
                    });
                }
            }
        }
        coeffs.push(TortoiseCoeffPlan { scale, terms });
    }

    Ok(TortoiseOdfPlan {
        coeffs,
        max_power,
        s,
    })
}

fn rotate_row_vector(dir: [f32; 3], rotation: &Matrix3<f64>) -> [f64; 3] {
    let row = Vector3::new(dir[0] as f64, dir[1] as f64, dir[2] as f64);
    let out = rotation.transpose() * row;
    [out[0], out[1], out[2]]
}

fn tortoise_index_matrix(radial_order: usize) -> Result<Vec<[usize; 3]>> {
    if radial_order % 2 != 0 {
        return Err(OdxError::Format(
            "TORTOISE MAPMRI radial order must be even".into(),
        ));
    }
    let half = radial_order / 2;
    if half >= TORTOISE_NNMAX.len() {
        return Err(OdxError::Format(format!(
            "TORTOISE MAPMRI radial order {radial_order} exceeds supported lookup table"
        )));
    }
    let lim = TORTOISE_NNMAX[half] + 1;
    Ok((0..lim)
        .map(|idx| [TORTOISE_N1A[idx], TORTOISE_N2A[idx], TORTOISE_N3A[idx]])
        .collect())
}

fn factorial(n: usize) -> f64 {
    if n <= 1 {
        1.0
    } else {
        (2..=n).fold(1.0, |acc, value| acc * value as f64)
    }
}

fn double_factorial(n: usize) -> f64 {
    if n <= 1 {
        1.0
    } else {
        (1..=n)
            .rev()
            .step_by(2)
            .fold(1.0, |acc, value| acc * value as f64)
    }
}

fn fill_powers(base: f64, out: &mut [f64]) {
    if out.is_empty() {
        return;
    }
    out[0] = 1.0;
    for idx in 1..out.len() {
        out[idx] = out[idx - 1] * base;
    }
}

fn robust_positive_percentile(values: &[f32], percentile: f32) -> f32 {
    let mut positives = values
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    if positives.is_empty() {
        return 1.0;
    }
    let rank = ((positives.len() - 1) as f32 * percentile.clamp(0.0, 1.0)).round() as usize;
    let (_, value, _) = positives.select_nth_unstable_by(rank, |a, b| a.total_cmp(b));
    *value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infers_tortoise_radial_order_four_from_fixture_coeff_count() {
        assert_eq!(tortoise_order_from_coeffs(22).unwrap(), 4);
    }

    #[test]
    fn real_fixture_selected_voxel_projects_to_finite_sh() {
        let (coeff_dims, _, coeff_data) =
            mrtrix::load_nifti_f32_volume(Path::new("../test_data/xx_mapmri.nii")).unwrap();
        let (tensor_dims, _, tensor_data) =
            mrtrix::load_nifti_f32_volume(Path::new("../test_data/xx_L1_DT.nii")).unwrap();
        let (uvec_dims, _, uvec_data) =
            mrtrix::load_nifti_f32_volume(Path::new("../test_data/xx_uvec.nii")).unwrap();
        let (dims3, ncoeffs) = normalize_trailing_channels(
            &coeff_dims,
            Path::new("xx_mapmri.nii"),
            &[22, 50, 95, 161],
            "coeffs",
        )
        .unwrap();
        let (_, ntensor) =
            normalize_trailing_channels(&tensor_dims, Path::new("xx_L1_DT.nii"), &[6], "tensor")
                .unwrap();
        let (_, nuvec) =
            normalize_trailing_channels(&uvec_dims, Path::new("xx_uvec.nii"), &[3], "uvec")
                .unwrap();
        let voxel_count = dims3[0] * dims3[1] * dims3[2];
        let coeff_rows = collapse_rows(&coeff_data, voxel_count, ncoeffs).unwrap();
        let tensor_rows = collapse_rows(&tensor_data, voxel_count, ntensor).unwrap();
        let uvec_rows = collapse_rows(&uvec_data, voxel_count, nuvec).unwrap();
        let voxel = coeff_rows
            .iter()
            .zip(tensor_rows.iter())
            .zip(uvec_rows.iter())
            .position(|((coeff, tensor), uvec)| {
                coeff[0] != 0.0
                    && !uvec.iter().all(|value| *value == 0.0)
                    && tensor[0] + tensor[1] + tensor[2] != 0.0
            })
            .unwrap();
        let rotation = tortoise_rotation_from_tensor(tensor_rows[voxel]).unwrap();
        let mu = tortoise_mu_from_uvec(uvec_rows[voxel]).unwrap();
        let dirs = dsistudio_odf8::hemisphere_vertices_ras();
        let plan = tortoise_odf_plan(4, TORTOISE_ODF_S).unwrap();
        let mut odf = vec![0.0f32; dirs.len()];
        tortoise_odf_samples_into(&plan, coeff_rows[voxel], mu, &rotation, dirs, &mut odf).unwrap();
        let lmax = mrtrix_sh::resolve_lmax_for_directions(dirs, None, DEFAULT_SH_LMAX);
        let sh = mrtrix_sh::fit_rows_from_amplitudes(&odf, 1, dirs, lmax).unwrap();
        assert_eq!(sh.len(), mrtrix_sh::ncoeffs_for_lmax(lmax));
        assert!(sh.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn real_fixture_has_two_coeff_supported_zero_uvec_voxels() {
        let (coeff_dims, _, coeff_data) =
            mrtrix::load_nifti_f32_volume(Path::new("../test_data/xx_mapmri.nii")).unwrap();
        let (uvec_dims, _, uvec_data) =
            mrtrix::load_nifti_f32_volume(Path::new("../test_data/xx_uvec.nii")).unwrap();
        let (dims3, ncoeffs) = normalize_trailing_channels(
            &coeff_dims,
            Path::new("xx_mapmri.nii"),
            &[22, 50, 95, 161],
            "coeffs",
        )
        .unwrap();
        let (_, nuvec) =
            normalize_trailing_channels(&uvec_dims, Path::new("xx_uvec.nii"), &[3], "uvec")
                .unwrap();
        let voxel_count = dims3[0] * dims3[1] * dims3[2];
        let coeff_rows = collapse_rows(&coeff_data, voxel_count, ncoeffs).unwrap();
        let uvec_rows = collapse_rows(&uvec_data, voxel_count, nuvec).unwrap();
        let mismatches = coeff_rows
            .iter()
            .zip(uvec_rows.iter())
            .filter(|(coeff, uvec)| coeff[0] != 0.0 && uvec.iter().all(|value| *value == 0.0))
            .count();
        assert_eq!(mismatches, 2);
    }

    #[test]
    fn robust_positive_percentile_ignores_nonpositive_values() {
        let values = vec![0.0, -1.0, 2.0, 5.0, 3.0];
        assert_eq!(robust_positive_percentile(&values, 0.99), 5.0);
        assert_eq!(robust_positive_percentile(&values, 0.5), 3.0);
    }

    #[test]
    fn fixture_tensor_export_order_produces_positive_eigenvalues() {
        let (coeff_dims, _, coeff_data) =
            mrtrix::load_nifti_f32_volume(Path::new("../test_data/xx_mapmri.nii")).unwrap();
        let (tensor_dims, _, tensor_data) =
            mrtrix::load_nifti_f32_volume(Path::new("../test_data/xx_L1_DT.nii")).unwrap();
        let (uvec_dims, _, uvec_data) =
            mrtrix::load_nifti_f32_volume(Path::new("../test_data/xx_uvec.nii")).unwrap();
        let (dims3, ncoeffs) = normalize_trailing_channels(
            &coeff_dims,
            Path::new("xx_mapmri.nii"),
            &[22, 50, 95, 161],
            "coeffs",
        )
        .unwrap();
        let (_, ntensor) =
            normalize_trailing_channels(&tensor_dims, Path::new("xx_L1_DT.nii"), &[6], "tensor")
                .unwrap();
        let (_, nuvec) =
            normalize_trailing_channels(&uvec_dims, Path::new("xx_uvec.nii"), &[3], "uvec")
                .unwrap();
        let voxel_count = dims3[0] * dims3[1] * dims3[2];
        let coeff_rows = collapse_rows(&coeff_data, voxel_count, ncoeffs).unwrap();
        let tensor_rows = collapse_rows(&tensor_data, voxel_count, ntensor).unwrap();
        let uvec_rows = collapse_rows(&uvec_data, voxel_count, nuvec).unwrap();
        let voxel = coeff_rows
            .iter()
            .zip(tensor_rows.iter())
            .zip(uvec_rows.iter())
            .position(|((coeff, tensor), uvec)| {
                coeff[0] != 0.0
                    && !uvec.iter().all(|value| *value == 0.0)
                    && tensor[0] + tensor[1] + tensor[2] != 0.0
            })
            .unwrap();
        let tensor = tensor_rows[voxel];
        let matrix = Matrix3::new(
            tensor[0] as f64,
            tensor[3] as f64,
            tensor[4] as f64,
            tensor[3] as f64,
            tensor[1] as f64,
            tensor[5] as f64,
            tensor[4] as f64,
            tensor[5] as f64,
            tensor[2] as f64,
        );
        let eig = SymmetricEigen::new(matrix);
        assert!(eig.eigenvalues.iter().all(|value| *value > 0.0));
    }
}
