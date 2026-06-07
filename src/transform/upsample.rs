//! Spatial upsampling of ODX datasets onto a finer isotropic voxel grid.
//!
//! Resamples SH coefficients and DPV arrays using trilinear interpolation
//! (with renormalization at mask boundaries). Fixels are recomputed from the
//! interpolated SH via peak finding. Dense ODF data is not supported.

use std::collections::HashMap;

use nalgebra::{Matrix4, Vector4};

use crate::dtype::DType;
use crate::error::{OdxError, Result};
use crate::mmap_backing::vec_into_bytes;
use crate::odx_file::OdxDataset;
use crate::peak_finder::PeakFinderConfig;
use crate::stream::OdxBuilder;
use crate::transform::source_volume::SourceLookup;

/// Options for [`upsample`].
#[derive(Clone, Debug)]
pub struct UpsampleOptions {
    pub peak_config: PeakFinderConfig,
}

impl Default for UpsampleOptions {
    fn default() -> Self {
        Self {
            peak_config: PeakFinderConfig::default(),
        }
    }
}

/// New grid geometry produced by [`compute_upsampled_grid`].
pub struct UpsampledGrid {
    pub affine: [[f64; 4]; 4],
    pub dims: [u64; 3],
}

/// Compute a new affine and dimensions covering the same physical extent as
/// `header` but with isotropic voxel spacing of `spacing_mm`.
///
/// The origin (affine column 3) is unchanged. Each axis vector is rescaled to
/// the requested spacing, and new dimensions are chosen so that
/// `new_dim * spacing ≥ old_dim * old_spacing` on every axis.
pub fn compute_upsampled_grid(
    affine: &[[f64; 4]; 4],
    dims: [u64; 3],
    spacing_mm: f64,
) -> UpsampledGrid {
    let mut new_affine = *affine;
    let mut new_dims = [1u64; 3];

    for axis in 0..3 {
        // Column `axis` of the affine is the axis vector in mm.
        let vx = affine[0][axis];
        let vy = affine[1][axis];
        let vz = affine[2][axis];
        let old_spacing = (vx * vx + vy * vy + vz * vz).sqrt();
        if old_spacing < 1e-12 {
            new_dims[axis] = dims[axis];
            continue;
        }
        let scale = spacing_mm / old_spacing;
        new_affine[0][axis] = vx * scale;
        new_affine[1][axis] = vy * scale;
        new_affine[2][axis] = vz * scale;
        // ceil(old_dim * old_spacing / new_spacing) to cover same extent
        new_dims[axis] = ((dims[axis] as f64 * old_spacing / spacing_mm).ceil() as u64).max(1);
    }

    UpsampledGrid {
        affine: new_affine,
        dims: new_dims,
    }
}

/// Upsample `input` onto a finer isotropic grid of `spacing_mm` mm.
///
/// - SH arrays and DPV arrays are trilinearly interpolated. Boundary voxels
///   (where some of the 8 source neighbours are outside the mask) use
///   renormalized weights so signal levels are preserved.
/// - Fixels are recomputed from the upsampled SH via [`OdxBuilder::compute_peaks`].
/// - DPF arrays other than `amplitude` cannot be remapped to recomputed fixels
///   and are dropped with a warning.
/// - Dense ODF data (`odf/`) is not supported and returns an error.
pub fn upsample(
    input: &OdxDataset,
    spacing_mm: f64,
    opts: &UpsampleOptions,
) -> Result<OdxDataset> {
    // ---- Phase 0: validate ----
    if !input.odf_names().is_empty() {
        return Err(OdxError::Argument(
            "Dense ODF upsampling is not supported; run peak extraction first \
             to produce SH + fixels, then upsample."
                .into(),
        ));
    }
    for name in input.dpf_names() {
        if name != "amplitude" {
            eprintln!(
                "odx upsample: warning: dropping DPF '{name}' \
                 (cannot remap to recomputed fixels)"
            );
        }
    }
    if spacing_mm <= 0.0 {
        return Err(OdxError::Argument(format!(
            "voxel spacing must be positive, got {spacing_mm}"
        )));
    }

    let in_header = input.header();

    // ---- Phase 1: compute target grid ----
    let grid = compute_upsampled_grid(&in_header.voxel_to_rasmm, in_header.dimensions, spacing_mm);
    let total_target = (grid.dims[0] * grid.dims[1] * grid.dims[2]) as usize;

    // ---- Phase 2: build source lookup ----
    let lookup = SourceLookup::new(in_header, input.mask());

    // ---- Phase 3: pre-decode source arrays as f32 ----
    let mut sh_in: HashMap<String, (Vec<f32>, usize)> = HashMap::new();
    for name in input.sh_names() {
        let arr = input
            .sh_arrays_get(name)
            .ok_or_else(|| OdxError::Argument(format!("missing SH array '{name}'")))?;
        sh_in.insert(name.to_string(), (arr.to_f32_vec()?, arr.ncols()));
    }
    let mut dpv_in: HashMap<String, (Vec<f32>, usize)> = HashMap::new();
    for name in input.dpv_names() {
        let arr = input
            .dpv_arrays_get(name)
            .ok_or_else(|| OdxError::Argument(format!("missing DPV '{name}'")))?;
        dpv_in.insert(name.to_string(), (arr.to_f32_vec()?, arr.ncols()));
    }

    let sh_ncoeffs: HashMap<String, usize> =
        sh_in.iter().map(|(n, (_, c))| (n.clone(), *c)).collect();
    let mut dpv_ncols: HashMap<String, usize> =
        dpv_in.iter().map(|(n, (_, c))| (n.clone(), *c)).collect();

    // ---- Phase 4: build target affine matrix for world-coord conversion ----
    let target_mat = Matrix4::from_row_slice(&[
        grid.affine[0][0], grid.affine[0][1], grid.affine[0][2], grid.affine[0][3],
        grid.affine[1][0], grid.affine[1][1], grid.affine[1][2], grid.affine[1][3],
        grid.affine[2][0], grid.affine[2][1], grid.affine[2][2], grid.affine[2][3],
        grid.affine[3][0], grid.affine[3][1], grid.affine[3][2], grid.affine[3][3],
    ]);

    // ---- Phase 5a: compute output mask via nearest-neighbor ----
    let mut out_mask = vec![0u8; total_target];
    for i in 0..grid.dims[0] as i64 {
        for j in 0..grid.dims[1] as i64 {
            for k in 0..grid.dims[2] as i64 {
                let flat = voxel_flat(i, j, k, &grid.dims);
                let world = voxel_world(&target_mat, i, j, k);
                if lookup.nearest_compact(world).is_some() {
                    out_mask[flat] = 1;
                }
            }
        }
    }

    // ---- Phase 5b: trilinear interpolation of SH and DPV ----
    let max_ncoeffs = sh_ncoeffs.values().copied().max().unwrap_or(0);
    let mut sh_scratch = vec![0.0f32; max_ncoeffs];

    let mut sh_out: HashMap<String, Vec<f32>> =
        sh_ncoeffs.keys().map(|n| (n.clone(), Vec::new())).collect();
    let mut dpv_out: HashMap<String, Vec<f32>> =
        dpv_ncols.keys().map(|n| (n.clone(), Vec::new())).collect();

    for i in 0..grid.dims[0] as i64 {
        for j in 0..grid.dims[1] as i64 {
            for k in 0..grid.dims[2] as i64 {
                let flat = voxel_flat(i, j, k, &grid.dims);
                if out_mask[flat] == 0 {
                    continue;
                }
                let world = voxel_world(&target_mat, i, j, k);
                let weights = lookup.trilinear_weights(world);
                let total_w = weights.total_weight;

                // SH: trilinear + renormalize at boundaries (no reorientation)
                for (name, (data, ncols)) in &sh_in {
                    let nc = *ncols;
                    sh_scratch[..nc].iter_mut().for_each(|v| *v = 0.0);
                    weights.accumulate_row(data, nc, &mut sh_scratch[..nc], |v| *v);
                    let out = sh_out.get_mut(name).unwrap();
                    if total_w > 0.0 && total_w < 1.0 - 1e-6 {
                        let inv = 1.0 / total_w;
                        out.extend(sh_scratch[..nc].iter().map(|v| v * inv));
                    } else {
                        out.extend_from_slice(&sh_scratch[..nc]);
                    }
                }

                // DPV: same, renormalize at boundaries
                for (name, (data, ncols)) in &dpv_in {
                    let nc = *ncols;
                    let out = dpv_out.get_mut(name).unwrap();
                    let base = out.len();
                    out.resize(base + nc, 0.0);
                    weights.accumulate_row(data, nc, &mut out[base..], |v| *v);
                    if total_w > 0.0 && total_w < 1.0 - 1e-6 {
                        let inv = 1.0 / total_w;
                        for v in &mut out[base..] {
                            *v *= inv;
                        }
                    }
                }
            }
        }
    }

    // Auto-compute anisotropic_power from interpolated SH (mirrors odx convert).
    // Only added when SH coefficients are present, lmax > 0, and the input
    // didn't already carry its own anisotropic_power DPV.
    if let Some((_, ncoeffs)) = sh_in.get("coefficients") {
        let lmax = in_header.sh_order.unwrap_or(0) as usize;
        if lmax > 0 && !dpv_in.contains_key("anisotropic_power") {
            let nc = *ncoeffs;
            let sh_rows = sh_out.get("coefficients").unwrap();
            let nb_vox_out = sh_rows.len() / nc;
            let ap: Vec<f32> = (0..nb_vox_out)
                .map(|v| {
                    crate::mrtrix_sh::anisotropic_power(
                        &sh_rows[v * nc..(v + 1) * nc],
                        lmax,
                        crate::mrtrix_sh::ANISOTROPIC_POWER_NORM_FACTOR,
                    )
                })
                .collect();
            dpv_out.insert("anisotropic_power".to_string(), ap);
            dpv_ncols.insert("anisotropic_power".to_string(), 1);
        }
    }

    // ---- Phase 6: assemble output with OdxBuilder ----
    let mut builder = OdxBuilder::new(grid.affine, grid.dims, out_mask);

    // Copy SH metadata from input header.
    if let (Some(order), Some(basis)) = (in_header.sh_order, in_header.sh_basis.as_deref()) {
        builder.set_sh_info(order, basis.to_string());
    }
    if let Some(full) = in_header.sh_full_basis {
        builder.set_sh_full_basis(full);
    }
    if let Some(legacy) = in_header.sh_legacy {
        builder.set_sh_legacy(legacy);
    }
    if let Some(ref id) = in_header.sphere_id {
        builder.set_sphere_id(id.clone());
    }
    if let Some(ref domain) = in_header.odf_sample_domain {
        builder.set_odf_sample_domain(domain.clone());
    }
    if let Some(rep) = in_header.canonical_dense_representation.clone() {
        builder.set_canonical_dense_representation(rep);
    }
    if let (Some(verts), Some(faces)) = (input.sphere_vertices(), input.sphere_faces()) {
        builder.set_sphere(verts.to_vec(), faces.to_vec());
    }
    for (k, v) in &in_header.extra {
        builder.set_extra_value(k.clone(), v.clone());
    }

    // Attach interpolated SH.
    for (name, data) in sh_out {
        let ncols = sh_ncoeffs[&name];
        builder.set_sh_data(&name, vec_into_bytes(data), ncols, DType::Float32);
    }

    // Attach interpolated DPV.
    for (name, data) in dpv_out {
        let ncols = dpv_ncols[&name];
        builder.set_dpv_data(&name, vec_into_bytes(data), ncols, DType::Float32);
    }

    // Compute peaks from interpolated SH (also attaches dpf/amplitude).
    if !sh_in.is_empty() {
        builder.skip_all_peaks();
        builder.compute_peaks(None, opts.peak_config.clone())?;
    } else {
        builder.skip_all_peaks();
    }

    builder.finalize()
}

#[inline]
fn voxel_flat(i: i64, j: i64, k: i64, dims: &[u64; 3]) -> usize {
    (i as u64 * dims[1] * dims[2] + j as u64 * dims[2] + k as u64) as usize
}

#[inline]
fn voxel_world(mat: &Matrix4<f64>, i: i64, j: i64, k: i64) -> [f64; 3] {
    let v = mat * Vector4::new(i as f64, j as f64, k as f64, 1.0);
    [v[0], v[1], v[2]]
}
