//! Gradient-nonlinearity ("graddev") correction of reconstructed data.
//!
//! A gradient-deviation field describes, per voxel, how the diffusion
//! gradient that was *actually* applied differs from the nominal one. A
//! reconstruction fitted with a single global b-table (every ODF/FOD, every
//! peak) is therefore expressed against a locally distorted q-space. This
//! module undoes that distortion on the fitted quantities.
//!
//! # File convention
//!
//! The field is a 4-D NIfTI with 9 volumes in the HCP/FSL `grad_dev` layout.
//! Reading the 9 values of one voxel row-major into a 3×3 matrix `T`
//! (`T[i][j] = vol[3*i + j]`, i.e. numpy `reshape(3, 3)` in C order) gives
//!
//! ```text
//!   g_eff = Tᵀ · g          (components in the field image's voxel frame)
//! ```
//!
//! This is exactly FSL's `correct_bvals_bvecs` (`xfibres.cc`), which fills its
//! matrix column-major from the 9 volumes and applies `(I + L)·g`, and it is
//! what TORTOISE's `CreateGradientNonlinearityBMatrix` produces after its
//! "convert to HCP format ordering" transpose. HCP files store `T − I`
//! (background ≈ 0); TORTOISE/qsiprep files store `T` (diagonal ≈ 1). See
//! [`IdentityPolicy`].
//!
//! The 9 components are expressed in the voxel (i, j, k) axes of the field
//! image. ODX directions live in RAS+ world space, so the field is rotated
//! once with the column-normalized rotation `P` of the field's affine:
//! `T_ras = P · T · P⁻¹`. The field is never axis-canonicalized on load —
//! permuting the grid without re-expressing the components would silently
//! change their meaning.
//!
//! # What is corrected
//!
//! With `q_eff = Tᵀ q` the reconstruction sees `E_est(q) = E_true(Tᵀ q)`;
//! Fourier duality gives `P_est(r) = P_true(T⁻¹ r) / |det T|`, so on the sphere
//!
//! ```text
//!   ψ_true(u)  = ψ_est( normalize(T_ras · u) )        (ODF / FOD amplitudes)
//!   u_true     = normalize( T_ras⁻¹ · u_est )         (peak / fixel directions)
//! ```
//!
//! SH arrays are rotated with the aPSF reorienter, dense ODF arrays are
//! refit to SH and re-evaluated at the rotated sphere directions, and fixel
//! directions are mapped through `T_ras⁻¹`. The per-voxel b-value deviation
//! `|Tᵀ g|²` cannot be undone on an already-fitted ODF and is only reported
//! (`max_abs_log_det`). Nothing is modulated.

use std::collections::HashMap;
use std::path::Path;

use nalgebra::{Matrix3, Matrix4, Vector3, Vector4};
use serde::Serialize;

use crate::data_array::DataArray;
use crate::dtype::DType;
use crate::error::{OdxError, Result};
use crate::formats::mrtrix;
use crate::interop::dsistudio_sampling_dirs;
use crate::mmap_backing::{vec_to_bytes, MmapBacking};
use crate::mrtrix_sh;
use crate::odx_file::OdxDataset;
use crate::transform::sh_apsf::{ApsfBasis, ShReorienter};

/// Header-extra key recording that a gradient-deviation correction was applied.
pub const GRADDEV_APPLIED_KEY: &str = "GRADDEV_APPLIED";

/// DPF written by the PAM5 loader that caches dipy's sphere indices; it is
/// stale once directions rotate and must not be reused on export.
const PAM_PEAK_INDEX_DPF: &str = "pam_peak_index";

/// Default lmax for the SH round trip of dense ODF arrays (capped by what the
/// sampling sphere supports, see [`mrtrix_sh::resolve_lmax_for_directions`]).
const DEFAULT_ODF_LMAX: usize = 12;

/// Median of `|T00| + |T11| + |T22|` below which an auto-detected field is
/// taken to store `T − I` (HCP/FSL style) rather than `T`.
const IDENTITY_AUTO_TRACE_THRESHOLD: f64 = 1.5;

/// Frobenius distance from identity below which a voxel is left untouched.
const NEAR_IDENTITY_FROBENIUS: f64 = 1e-6;

/// |det T| below which a voxel's matrix is treated as unusable.
const SINGULAR_DET: f64 = 1e-9;

/// Whether the field already contains the identity on its diagonal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IdentityPolicy {
    /// Decide from the data: the median diagonal magnitude over nonzero
    /// voxels is compared against [`IDENTITY_AUTO_TRACE_THRESHOLD`].
    #[default]
    Auto,
    /// The file stores `T` (diagonal ≈ 1; TORTOISE/qsiprep output).
    Included,
    /// The file stores `T − I` (diagonal ≈ 0; HCP/FSL `grad_dev`).
    Absent,
}

/// A loaded 9-component gradient-deviation field.
#[derive(Debug, Clone)]
pub struct GradDevField {
    dims: [usize; 3],
    affine: [[f64; 4]; 4],
    inv_affine: Matrix4<f64>,
    rotation: Matrix3<f64>,
    rotation_inv: Matrix3<f64>,
    /// C-order `(i, j, k, 9)`: the 9 components of a voxel are contiguous.
    data: Vec<f32>,
    identity_added: bool,
    source: Option<String>,
}

impl GradDevField {
    /// Load a 9-volume NIfTI (`.nii`/`.nii.gz`, float32). The grid is kept in
    /// its on-disk voxel order.
    pub fn load_nifti(path: &Path, policy: IdentityPolicy) -> Result<Self> {
        let (dims, affine, data) = mrtrix::load_nifti_f32_volume_native(path)?;
        if dims.len() < 4 {
            return Err(OdxError::Format(format!(
                "gradient deviation image '{}' must be 4-D with 9 volumes, found dims {:?}",
                path.display(),
                dims
            )));
        }
        let trailing: usize = dims[3..].iter().product();
        if trailing != 9 {
            return Err(OdxError::Format(format!(
                "gradient deviation image '{}' must carry 9 components per voxel, found {} (dims {:?})",
                path.display(),
                trailing,
                dims
            )));
        }
        let mut field = Self::from_parts([dims[0], dims[1], dims[2]], affine, data, policy)?;
        field.source = Some(path.display().to_string());
        Ok(field)
    }

    /// Build a field from an in-memory `(i, j, k, 9)` C-order array whose
    /// voxel→RAS+ affine is `affine`.
    pub fn from_parts(
        dims: [usize; 3],
        affine: [[f64; 4]; 4],
        data: Vec<f32>,
        policy: IdentityPolicy,
    ) -> Result<Self> {
        let nvox = dims[0] * dims[1] * dims[2];
        if data.len() != nvox * 9 {
            return Err(OdxError::Format(format!(
                "gradient deviation data has {} values, expected {} (= {:?} voxels × 9)",
                data.len(),
                nvox * 9,
                dims
            )));
        }

        let identity_added = match policy {
            IdentityPolicy::Included => false,
            IdentityPolicy::Absent => true,
            IdentityPolicy::Auto => identity_is_absent(&data),
        };

        let affine_mat = Matrix4::from_fn(|r, c| affine[r][c]);
        let inv_affine = affine_mat.try_inverse().ok_or_else(|| {
            OdxError::Format("gradient deviation image affine is singular".into())
        })?;

        let mut rotation = Matrix3::zeros();
        #[allow(clippy::needless_range_loop)]
        for c in 0..3 {
            let col = Vector3::new(affine[0][c], affine[1][c], affine[2][c]);
            let n = col.norm();
            if !(n.is_finite()) || n < 1e-12 {
                return Err(OdxError::Format(format!(
                    "gradient deviation image affine has a degenerate column {c}"
                )));
            }
            rotation.set_column(c, &(col / n));
        }
        let rotation_inv = rotation.try_inverse().ok_or_else(|| {
            OdxError::Format("gradient deviation image affine rotation is singular".into())
        })?;

        Ok(Self {
            dims,
            affine,
            inv_affine,
            rotation,
            rotation_inv,
            data,
            identity_added,
            source: None,
        })
    }

    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    pub fn affine(&self) -> [[f64; 4]; 4] {
        self.affine
    }

    /// `true` when the identity was added to the stored diagonal.
    pub fn identity_added(&self) -> bool {
        self.identity_added
    }

    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }

    /// Row-major `T` at a voxel index, in the field's own (i, j, k) frame,
    /// with the identity added if the file stores `T − I`.
    pub fn matrix_ijk_at(&self, i: usize, j: usize, k: usize) -> Matrix3<f64> {
        let base = ((i * self.dims[1] + j) * self.dims[2] + k) * 9;
        let v = &self.data[base..base + 9];
        let mut t = Matrix3::from_fn(|r, c| v[3 * r + c] as f64);
        if self.identity_added {
            t += Matrix3::identity();
        }
        t
    }

    /// `T` expressed in RAS+ world axes at the voxel containing `p_ras`
    /// (nearest-neighbour), or `None` when the point falls outside the field.
    pub fn matrix_ras_at(&self, p_ras: [f64; 3]) -> Option<Matrix3<f64>> {
        let idx = self.inv_affine * Vector4::new(p_ras[0], p_ras[1], p_ras[2], 1.0);
        let mut ijk = [0usize; 3];
        for a in 0..3 {
            let r = idx[a].round();
            if !r.is_finite() || r < 0.0 || r >= self.dims[a] as f64 {
                return None;
            }
            ijk[a] = r as usize;
        }
        let t = self.matrix_ijk_at(ijk[0], ijk[1], ijk[2]);
        Some(self.rotation * t * self.rotation_inv)
    }
}

/// Auto-detect whether a field stores `T − I`: median diagonal magnitude
/// over voxels that carry finite, not-all-zero values.
fn identity_is_absent(data: &[f32]) -> bool {
    let mut traces: Vec<f64> = data
        .chunks_exact(9)
        .filter(|v| v.iter().all(|x| x.is_finite()) && v.iter().any(|&x| x != 0.0))
        .map(|v| v[0].abs() as f64 + v[4].abs() as f64 + v[8].abs() as f64)
        .collect();
    if traces.is_empty() {
        return false;
    }
    traces.sort_by(|a, b| a.total_cmp(b));
    let n = traces.len();
    let median = if n % 2 == 1 {
        traces[n / 2]
    } else {
        0.5 * (traces[n / 2 - 1] + traces[n / 2])
    };
    median < IDENTITY_AUTO_TRACE_THRESHOLD
}

/// Options for [`apply_graddev`].
#[derive(Debug, Clone, Default)]
pub struct GradDevOptions {
    /// Fibonacci-sphere directions for the aPSF SH reorientation. Defaults to
    /// `max(80, 3 × ncoeffs)` of the SH array being rotated.
    pub apsf_dirs: Option<usize>,
    /// lmax for the SH round trip of dense ODF arrays (default: up to 12,
    /// capped by the sampling sphere).
    pub odf_lmax: Option<usize>,
    /// Debug escape hatch: apply `Tᵀ` instead of `T`. Only useful to prove a
    /// convention error; never for real data.
    pub transpose: bool,
}

/// What [`apply_graddev`] did.
#[derive(Debug, Clone, Serialize)]
pub struct GradDevReport {
    pub nb_voxels: usize,
    /// Voxels whose matrix differed from identity and were corrected.
    pub nb_corrected: usize,
    /// Mask voxels whose centre fell outside the field (left untouched).
    pub nb_outside_field: usize,
    /// Voxels with a non-finite or singular matrix (left untouched).
    pub nb_singular: usize,
    pub identity_added: bool,
    pub transpose: bool,
    /// Per-voxel rotation proxy: max over the three axes of the angle between
    /// `e_i` and `normalize(T⁻¹ e_i)`, summarised over corrected voxels.
    pub median_rotation_deg: f32,
    pub max_rotation_deg: f32,
    /// `max |ln det T|` over corrected voxels — the uncorrectable b-value
    /// deviation, for information only.
    pub max_abs_log_det: f32,
    pub sh_arrays: Vec<String>,
    pub odf_arrays: Vec<String>,
    pub odf_lmax: Option<usize>,
    pub apsf_dirs: Option<usize>,
    pub nb_peaks: usize,
    pub dropped_dpf: Vec<String>,
}

/// Apply a gradient-deviation field to every direction-bearing array of a
/// dataset. Returns the corrected dataset (same grid, mask, peak cardinality
/// and scalar arrays) and a report.
pub fn apply_graddev(
    odx: &OdxDataset,
    field: &GradDevField,
    opts: &GradDevOptions,
) -> Result<(OdxDataset, GradDevReport)> {
    let header = odx.header();
    let nb_voxels = odx.nb_voxels();

    // ---- Per-voxel matrices (None = leave the voxel untouched).
    let mut mats: Vec<Option<Matrix3<f64>>> = Vec::with_capacity(nb_voxels);
    let mut nb_outside_field = 0usize;
    let mut nb_singular = 0usize;
    let mut nb_corrected = 0usize;
    let mut angles: Vec<f32> = Vec::new();
    let mut max_abs_log_det = 0f64;
    let identity = Matrix3::<f64>::identity();
    for c in odx.mask_voxel_centers_ras() {
        let Some(t) = field.matrix_ras_at([c[0] as f64, c[1] as f64, c[2] as f64]) else {
            nb_outside_field += 1;
            mats.push(None);
            continue;
        };
        let t = if opts.transpose { t.transpose() } else { t };
        let det = t.determinant();
        if !t.iter().all(|v| v.is_finite()) || !det.is_finite() || det.abs() < SINGULAR_DET {
            nb_singular += 1;
            mats.push(None);
            continue;
        }
        if (t - identity).norm() < NEAR_IDENTITY_FROBENIUS {
            mats.push(None);
            continue;
        }
        max_abs_log_det = max_abs_log_det.max(det.abs().ln().abs());
        angles.push(rotation_proxy_deg(&t));
        nb_corrected += 1;
        mats.push(Some(t));
    }

    // ---- SH arrays: ψ_true(u) = ψ_est(normalize(T u)) via the aPSF reorienter.
    let mut new_sh: HashMap<String, DataArray> = HashMap::new();
    let mut sh_arrays = Vec::new();
    let mut apsf_dirs_used = None;
    let mut sh_names: Vec<&str> = odx.sh_names();
    sh_names.sort_unstable();
    for name in sh_names {
        let arr = odx
            .get_sh(name)
            .ok_or_else(|| OdxError::Argument(format!("missing SH array '{name}'")))?;
        let ncols = arr.ncols();
        let data = arr.to_f32_vec()?;
        if data.len() != nb_voxels * ncols {
            return Err(OdxError::Format(format!(
                "SH array '{name}' has {} values, expected {} rows × {ncols}",
                data.len(),
                nb_voxels
            )));
        }
        let basis = ApsfBasis::from_header(header, ncols)?;
        let n_dirs = opts.apsf_dirs.unwrap_or_else(|| default_apsf_dirs(ncols));
        let reorienter = ShReorienter::new(basis, n_dirs)?;
        let mut out = data.clone();
        for (v, m) in mats.iter().enumerate() {
            let Some(m) = m else { continue };
            let (src, dst) = (
                &data[v * ncols..(v + 1) * ncols],
                &mut out[v * ncols..(v + 1) * ncols],
            );
            reorienter.reorient_into(src, m, false, dst)?;
        }
        new_sh.insert(
            name.to_string(),
            DataArray::owned_bytes(vec_to_bytes(out), ncols, DType::Float32),
        );
        sh_arrays.push(name.to_string());
        apsf_dirs_used = Some(n_dirs);
    }

    // ---- Fixel directions: u_true = normalize(T⁻¹ u_est).
    let offsets = odx.offsets();
    let dirs_in = odx.directions();
    let mut dirs_out = dirs_in.to_vec();
    for (v, m) in mats.iter().enumerate() {
        let Some(m) = m else { continue };
        let inv = m.try_inverse().unwrap_or(identity);
        for f in offsets[v] as usize..offsets[v + 1] as usize {
            if let Some(d) = map_unit(&inv, dirs_in[f]) {
                dirs_out[f] = d;
            }
        }
    }

    // ---- Dense ODF arrays: refit to SH, evaluate at normalize(T v_k).
    let mut new_odf: HashMap<String, DataArray> = HashMap::new();
    let mut odf_arrays = Vec::new();
    let mut odf_lmax_used = None;
    let mut odf_names: Vec<&str> = odx.odf_names();
    odf_names.sort_unstable();
    for name in odf_names {
        let arr = odx
            .get_odf(name)
            .ok_or_else(|| OdxError::Argument(format!("missing ODF array '{name}'")))?;
        let ncols = arr.ncols();
        let data = arr.to_f32_vec()?;
        if data.len() != nb_voxels * ncols {
            return Err(OdxError::Format(format!(
                "ODF array '{name}' has {} values, expected {} rows × {ncols}",
                data.len(),
                nb_voxels
            )));
        }
        let dirs = dsistudio_sampling_dirs(odx, ncols)?;
        let lmax = mrtrix_sh::resolve_lmax_for_directions(&dirs, opts.odf_lmax, DEFAULT_ODF_LMAX);
        let fit = mrtrix_sh::RowFitPlan::for_amplitudes(&dirs, lmax)?;
        let ncoeffs = fit.target_coeff_count();
        let mut out = data.clone();
        let mut sh = vec![0.0f32; ncoeffs];
        let mut rotated = vec![[0.0f32; 3]; ncols];
        for (v, m) in mats.iter().enumerate() {
            let Some(m) = m else { continue };
            fit.apply_row_into(&data[v * ncols..(v + 1) * ncols], &mut sh);
            for (k, d) in dirs.iter().enumerate() {
                rotated[k] = map_unit(m, *d).unwrap_or(*d);
            }
            let basis = mrtrix_sh::sh2amp_cart(&rotated, lmax);
            let basis = basis.as_slice().expect("contiguous");
            let row_out = &mut out[v * ncols..(v + 1) * ncols];
            for k in 0..ncols {
                let row = &basis[k * ncoeffs..(k + 1) * ncoeffs];
                let mut acc = 0.0f32;
                for c in 0..ncoeffs {
                    acc += row[c] * sh[c];
                }
                // Matching `sh2amp -nonnegative`: clamp after evaluation.
                row_out[k] = acc.max(0.0);
            }
        }
        new_odf.insert(
            name.to_string(),
            DataArray::owned_bytes(vec_to_bytes(out), ncols, DType::Float32),
        );
        odf_arrays.push(name.to_string());
        odf_lmax_used = Some(lmax);
    }

    // ---- Assemble.
    let mut parts = odx.clone_owned_parts();
    parts.directions_backing = MmapBacking::Owned(vec_to_bytes(dirs_out));
    for (name, arr) in new_sh {
        parts.sh.insert(name, arr);
    }
    for (name, arr) in new_odf {
        parts.odf.insert(name, arr);
    }
    let mut dropped_dpf = Vec::new();
    if parts.dpf.remove(PAM_PEAK_INDEX_DPF).is_some() {
        dropped_dpf.push(PAM_PEAK_INDEX_DPF.to_string());
    }

    angles.sort_by(|a, b| a.total_cmp(b));
    let median_rotation_deg = if angles.is_empty() {
        0.0
    } else if angles.len() % 2 == 1 {
        angles[angles.len() / 2]
    } else {
        0.5 * (angles[angles.len() / 2 - 1] + angles[angles.len() / 2])
    };
    let max_rotation_deg = angles.last().copied().unwrap_or(0.0);

    let report = GradDevReport {
        nb_voxels,
        nb_corrected,
        nb_outside_field,
        nb_singular,
        identity_added: field.identity_added,
        transpose: opts.transpose,
        median_rotation_deg,
        max_rotation_deg,
        max_abs_log_det: max_abs_log_det as f32,
        sh_arrays,
        odf_arrays,
        odf_lmax: odf_lmax_used,
        apsf_dirs: apsf_dirs_used,
        nb_peaks: odx.nb_peaks(),
        dropped_dpf,
    };

    parts.header.extra.insert(
        GRADDEV_APPLIED_KEY.into(),
        serde_json::json!({
            "source": field.source,
            "identity_added": field.identity_added,
            "transpose": opts.transpose,
            "convention": "row-major T from 9 volumes; g_eff = T^T g in the field's voxel frame",
            "nb_corrected": nb_corrected,
            "nb_outside_field": nb_outside_field,
            "median_rotation_deg": median_rotation_deg,
            "max_rotation_deg": max_rotation_deg,
        }),
    );

    Ok((OdxDataset::from_parts(parts), report))
}

fn default_apsf_dirs(ncoeffs: usize) -> usize {
    (3 * ncoeffs).max(80)
}

/// `normalize(M · d)`, or `None` if the image is degenerate.
fn map_unit(m: &Matrix3<f64>, d: [f32; 3]) -> Option<[f32; 3]> {
    let v = m * Vector3::new(d[0] as f64, d[1] as f64, d[2] as f64);
    let n = v.norm();
    if !n.is_finite() || n < 1e-12 {
        return None;
    }
    Some([(v[0] / n) as f32, (v[1] / n) as f32, (v[2] / n) as f32])
}

/// Max over the three axes of the angle between `e_i` and `normalize(T⁻¹ e_i)`.
fn rotation_proxy_deg(t: &Matrix3<f64>) -> f32 {
    let inv = t.try_inverse().unwrap_or_else(Matrix3::identity);
    let mut worst = 0.0f64;
    for a in 0..3 {
        let mut e = [0.0f32; 3];
        e[a] = 1.0;
        if let Some(v) = map_unit(&inv, e) {
            let dot = (v[a] as f64).clamp(-1.0, 1.0);
            worst = worst.max(dot.acos().to_degrees());
        }
    }
    worst as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_field_data(nvox: usize, diag: f32) -> Vec<f32> {
        let mut v = vec![0.0f32; nvox * 9];
        for c in v.chunks_exact_mut(9) {
            c[0] = diag;
            c[4] = diag;
            c[8] = diag;
        }
        v
    }

    #[test]
    fn auto_identity_detection_uses_diagonal_median() {
        let dims = [2, 2, 2];
        let with_i = GradDevField::from_parts(
            dims,
            crate::Header::identity_affine(),
            identity_field_data(8, 1.0),
            IdentityPolicy::Auto,
        )
        .unwrap();
        assert!(!with_i.identity_added());

        let mut without = identity_field_data(8, 0.0);
        // Small off-diagonal deviation so the voxels are not all-zero.
        for c in without.chunks_exact_mut(9) {
            c[1] = 0.01;
        }
        let f = GradDevField::from_parts(
            dims,
            crate::Header::identity_affine(),
            without,
            IdentityPolicy::Auto,
        )
        .unwrap();
        assert!(f.identity_added());
        let t = f.matrix_ijk_at(0, 0, 0);
        assert!((t[(0, 0)] - 1.0).abs() < 1e-6);
        assert!((t[(0, 1)] - 0.01).abs() < 1e-6);

        let forced = GradDevField::from_parts(
            dims,
            crate::Header::identity_affine(),
            identity_field_data(8, 0.0),
            IdentityPolicy::Included,
        )
        .unwrap();
        assert!(!forced.identity_added());
    }

    #[test]
    fn ras_conversion_conjugates_by_the_affine_rotation() {
        // A matrix with a single off-diagonal xz term in an LPS+ voxel frame.
        let dims = [1, 1, 1];
        let mut data = identity_field_data(1, 1.0);
        data[2] = 0.05; // T[0][2]
        let mut affine = crate::Header::identity_affine();
        affine[0][0] = -1.0;
        affine[1][1] = -1.0;
        let f = GradDevField::from_parts(dims, affine, data, IdentityPolicy::Included).unwrap();
        let t = f.matrix_ras_at([0.0, 0.0, 0.0]).unwrap();
        // P = diag(-1,-1,1): xz entry flips sign, diagonal untouched.
        assert!((t[(0, 2)] + 0.05).abs() < 1e-9);
        assert!((t[(0, 0)] - 1.0).abs() < 1e-9);
        assert!(f.matrix_ras_at([5.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn rotation_proxy_reports_pure_rotation_angle() {
        let theta = 10.0f64.to_radians();
        let r = Matrix3::new(
            theta.cos(),
            -theta.sin(),
            0.0,
            theta.sin(),
            theta.cos(),
            0.0,
            0.0,
            0.0,
            1.0,
        );
        assert!((rotation_proxy_deg(&r) - 10.0).abs() < 1e-3);
    }
}
