//! Resampling pipeline for ODX datasets under spatial transforms.
//!
//! Two pipelines, dispatched on whether `TransformOptions::fixel_chain` is set:
//!
//! - **Pull-only** (default, `fixel_chain = None`): SH, DPV, **and fixels** are
//!   all pull-resampled via the same `chain` (which maps `target → source`
//!   coords, e.g. an ANTs `from-A_to-B.h5` storing the chain that pulls from
//!   A into B). Matches the conventional `mrtransform` + `fixeltransform`
//!   semantics, where every output voxel pulls fixel content from the nearest
//!   source voxel.
//!
//! - **Mixed pull/push** (`fixel_chain = Some(...)`): SH and DPV are pulled
//!   via `chain`; **fixels are pushed** via `fixel_chain`, which must map
//!   *source* coords → *target* coords. This is the semantically appropriate
//!   handling for sparse coordinate-bearing fixel data: each source fixel
//!   maps to exactly one target voxel (or is dropped if it lands outside the
//!   target mask). For paired ANTs h5 files
//!   (`from-A_to-B.h5` + `from-B_to-A.h5`), set `chain` to the former and
//!   `fixel_chain` to the latter (whose stored chain is `A → B` for points).
//!
//! In both cases the output mask is the pull mask (target voxels whose pull
//! source lands in the input mask). Pushed fixels that fall outside the pull
//! mask are dropped — this preserves "SH coverage drives the output region"
//! semantics and keeps SH/fixel masks consistent.

use std::collections::HashMap;

use itk_transforms_rs::{TargetGrid, TransformChain};
use nalgebra::{Matrix3, Vector4};

use crate::dtype::DType;
use crate::error::{OdxError, Result};
use crate::odx_file::OdxDataset;
use crate::stream::OdxBuilder;
use crate::transform::sh_apsf::{ApsfBasis, ShReorienter};
use crate::transform::source_volume::SourceLookup;

/// Knobs for [`super::apply_transform`].
///
/// Modulation is split into two independent flags. The CLI's `--modulate`
/// only flips `modulate_sh` (matching `mrtransform -modulate fod` semantics).
/// `modulate_fixel` is exposed for library users who want it but is rarely
/// useful: under push semantics fixels already preserve cardinality and
/// total AFD, and under pull semantics mrtrix3's `fixeltransform` itself
/// does not apply modulation.
#[derive(Clone, Debug)]
pub struct TransformOptions {
    /// Apply mrtrix-style per-direction modulation to SH coefficients
    /// (`‖J·d‖/det(J)`). Default: off.
    pub modulate_sh: bool,
    /// Apply `det(J)`-based modulation to amplitude-like DPF fields named
    /// in `modulated_dpf_names`. Default: off. Library use only — the CLI
    /// keeps fixels unmodulated to match mrtrix3's conventions.
    pub modulate_fixel: bool,
    /// Number of fibonacci-spiral reference directions for aPSF SH
    /// reorientation. ~80 covers lmax 8 reliably; ~300 for lmax 12.
    pub apsf_dirs: usize,
    /// DPF arrays whose values are amplitude-like and should be modulated
    /// when `modulate_fixel` is on. Default includes `amplitude`, `afd`,
    /// `fd`, `fc`, `fdc`. Other DPFs are passed through unchanged.
    pub modulated_dpf_names: Vec<String>,
    /// If `Some`, fixels are *pushed* via this chain (which must map source
    /// coords → target coords). When `None`, fixels are pulled via the same
    /// `chain` as SH/DPV (default).
    pub fixel_chain: Option<TransformChain>,
}

impl Default for TransformOptions {
    fn default() -> Self {
        Self {
            modulate_sh: false,
            modulate_fixel: false,
            apsf_dirs: 80,
            modulated_dpf_names: vec![
                "amplitude".to_string(),
                "amplitudes".to_string(),
                "afd".to_string(),
                "fd".to_string(),
                "fc".to_string(),
                "fdc".to_string(),
            ],
            fixel_chain: None,
        }
    }
}

pub fn run(
    input: &OdxDataset,
    chain: &TransformChain,
    target_grid: &TargetGrid,
    opts: &TransformOptions,
) -> Result<OdxDataset> {
    let in_header = input.header();
    let lookup = SourceLookup::new(in_header, input.mask());

    // ---- Pre-decode source dense arrays as f32.
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
    let mut dpf_in: HashMap<String, (Vec<f32>, usize)> = HashMap::new();
    for name in input.dpf_names() {
        let arr = input
            .dpf_arrays_get(name)
            .ok_or_else(|| OdxError::Argument(format!("missing DPF '{name}'")))?;
        dpf_in.insert(name.to_string(), (arr.to_f32_vec()?, arr.ncols()));
    }

    let directions_in: &[[f32; 3]] = input.directions();
    let offsets_in: &[u32] = input.offsets();

    let sh_reorienter = if !sh_in.is_empty() {
        let any_arr = sh_in.values().next().unwrap();
        let basis = ApsfBasis::from_header(in_header, any_arr.1)?;
        Some(ShReorienter::new(basis, opts.apsf_dirs)?)
    } else {
        None
    };

    let dims = target_grid.dims;
    let total_voxels = (dims[0] * dims[1] * dims[2]) as usize;
    let target_affine = grid_affine_matrix(target_grid);
    let voxel_sizes = target_grid.voxel_sizes();
    let min_size = voxel_sizes[0].min(voxel_sizes[1]).min(voxel_sizes[2]).max(1e-9);
    let fd_step = 0.25 * min_size;

    let sh_ncoeffs: HashMap<String, usize> =
        sh_in.iter().map(|(n, (_, c))| (n.clone(), *c)).collect();
    let dpv_ncols: HashMap<String, usize> =
        dpv_in.iter().map(|(n, (_, c))| (n.clone(), *c)).collect();
    let dpf_ncols: HashMap<String, usize> =
        dpf_in.iter().map(|(n, (_, c))| (n.clone(), *c)).collect();

    // ---- Phase 1: compute pull mask (cheap walk; no SH reorient yet).
    let mut pull_mask = vec![0u8; total_voxels];
    for i in 0..dims[0] as i64 {
        for j in 0..dims[1] as i64 {
            for k in 0..dims[2] as i64 {
                let flat = (i as u64 * dims[1] * dims[2] + j as u64 * dims[2] + k as u64) as usize;
                let p_fix = target_voxel_world(&target_affine, i, j, k);
                let p_mov = chain.map_point(p_fix);
                if lookup.nearest_compact(p_mov).is_some() {
                    pull_mask[flat] = 1;
                }
            }
        }
    }

    // ---- Phase 2: if fixel_chain, push source fixels into per-target buckets.
    //                (Buckets are filtered against pull_mask so SH/fixel
    //                 coverage stay consistent.)
    let push_buckets = if let Some(fchain) = &opts.fixel_chain {
        Some(push_source_fixels(
            input,
            fchain,
            target_grid,
            &target_affine,
            &pull_mask,
            directions_in,
            offsets_in,
            &dpf_in,
            &dpf_ncols,
            opts,
            fd_step,
        )?)
    } else {
        None
    };

    // ---- Phase 3: walk target voxels in C-order, emit final outputs.
    let mut new_mask = vec![0u8; total_voxels];

    let mut sh_out_compact: HashMap<String, Vec<f32>> = sh_in
        .keys()
        .map(|name| (name.clone(), Vec::new()))
        .collect();
    let mut dpv_out_compact: HashMap<String, Vec<f32>> = dpv_in
        .keys()
        .map(|name| (name.clone(), Vec::new()))
        .collect();
    let mut out_dirs: Vec<[f32; 3]> = Vec::new();
    let mut out_offsets: Vec<u32> = vec![0];
    let mut dpf_out_compact: HashMap<String, Vec<f32>> = dpf_in
        .keys()
        .map(|name| (name.clone(), Vec::new()))
        .collect();

    let mut sh_scratch_in: Vec<f32> =
        vec![0.0; sh_in.values().next().map(|(_, c)| *c).unwrap_or(0)];
    let mut sh_scratch_out: Vec<f32> = sh_scratch_in.clone();

    for i in 0..dims[0] as i64 {
        for j in 0..dims[1] as i64 {
            for k in 0..dims[2] as i64 {
                let flat = (i as u64 * dims[1] * dims[2] + j as u64 * dims[2] + k as u64) as usize;
                if pull_mask[flat] == 0 {
                    continue;
                }
                new_mask[flat] = 1;

                let p_fix = target_voxel_world(&target_affine, i, j, k);
                let p_mov = chain.map_point(p_fix);
                let nn_compact = lookup
                    .nearest_compact(p_mov)
                    .expect("pull_mask invariant: nearest is in mask");

                let j_chain = if chain.is_empty() {
                    Matrix3::<f64>::identity()
                } else {
                    chain.jacobian_at(p_fix, fd_step)
                };
                let det_j_chain = j_chain.determinant();

                // SH pull + reorient.
                if let Some(ref reor) = sh_reorienter {
                    for (name, (data, ncols)) in &sh_in {
                        sh_scratch_in.resize(*ncols, 0.0);
                        sh_scratch_out.resize(*ncols, 0.0);
                        for v in sh_scratch_in.iter_mut() {
                            *v = 0.0;
                        }
                        let weights = lookup.trilinear_weights(p_mov);
                        weights.accumulate_row(data, *ncols, &mut sh_scratch_in, |v| *v);
                        reor.reorient_into(
                            &sh_scratch_in,
                            &j_chain,
                            opts.modulate_sh,
                            &mut sh_scratch_out,
                        )?;
                        sh_out_compact
                            .get_mut(name)
                            .unwrap()
                            .extend_from_slice(&sh_scratch_out);
                    }
                }

                // DPV pull (no rotation, no modulation).
                for (name, (data, ncols)) in &dpv_in {
                    let weights = lookup.trilinear_weights(p_mov);
                    let out = dpv_out_compact.get_mut(name).unwrap();
                    let base = out.len();
                    out.resize(base + ncols, 0.0);
                    weights.accumulate_row(data, *ncols, &mut out[base..], |v| *v);
                }

                // Fixels: dispatch on push vs pull.
                if let Some(ref buckets) = push_buckets {
                    let dirs = &buckets.dirs[flat];
                    out_dirs.extend_from_slice(dirs);
                    out_offsets.push(out_dirs.len() as u32);
                    for (name, (_, ncols)) in &dpf_in {
                        let bucket = buckets.dpf.get(name).unwrap();
                        let row = &bucket[flat];
                        debug_assert_eq!(row.len(), dirs.len() * ncols);
                        dpf_out_compact.get_mut(name).unwrap().extend_from_slice(row);
                    }
                } else {
                    // Pull-fixel: nearest source voxel.
                    let src_row = nn_compact as usize;
                    let src_start = offsets_in[src_row] as usize;
                    let src_end = offsets_in[src_row + 1] as usize;

                    let j_fwd = j_chain.try_inverse().unwrap_or_else(Matrix3::identity);
                    for f in src_start..src_end {
                        let d = directions_in[f];
                        let v = j_fwd
                            * nalgebra::Vector3::new(
                                d[0] as f64,
                                d[1] as f64,
                                d[2] as f64,
                            );
                        let n = v.norm().max(1e-12);
                        out_dirs.push([
                            (v[0] / n) as f32,
                            (v[1] / n) as f32,
                            (v[2] / n) as f32,
                        ]);
                    }
                    out_offsets.push(out_dirs.len() as u32);

                    let amp_factor = if opts.modulate_fixel
                        && det_j_chain.is_finite()
                        && det_j_chain.abs() > 1e-12
                    {
                        det_j_chain as f32
                    } else {
                        1.0
                    };
                    for (name, (data, ncols)) in &dpf_in {
                        let out = dpf_out_compact.get_mut(name).unwrap();
                        let factor = if is_modulated_dpf(name, &opts.modulated_dpf_names) {
                            amp_factor
                        } else {
                            1.0
                        };
                        for f in src_start..src_end {
                            let row_start = f * ncols;
                            for c in 0..*ncols {
                                out.push(data[row_start + c] * factor);
                            }
                        }
                    }
                }
            }
        }
    }

    assemble_output(
        target_grid,
        new_mask,
        out_dirs,
        out_offsets,
        sh_out_compact,
        dpv_out_compact,
        dpf_out_compact,
        sh_ncoeffs,
        dpv_ncols,
        dpf_ncols,
        input,
    )
}

// ---------- helpers ----------

#[inline]
fn target_voxel_world(target_affine: &nalgebra::Matrix4<f64>, i: i64, j: i64, k: i64) -> [f64; 3] {
    let v = target_affine * Vector4::new(i as f64, j as f64, k as f64, 1.0);
    [v[0], v[1], v[2]]
}

struct PushFixelBuckets {
    /// Per-target-voxel directions list. `dirs[flat] = []` if no fixels pushed there.
    dirs: Vec<Vec<[f32; 3]>>,
    /// Per-DPF, per-target-voxel concatenated row data: `dpf[name][flat]` has
    /// length `nb_fixels[flat] * ncols`.
    dpf: HashMap<String, Vec<Vec<f32>>>,
}

#[allow(clippy::too_many_arguments)]
fn push_source_fixels(
    input: &OdxDataset,
    fixel_chain: &TransformChain,
    target_grid: &TargetGrid,
    target_affine: &nalgebra::Matrix4<f64>,
    pull_mask: &[u8],
    directions_in: &[[f32; 3]],
    offsets_in: &[u32],
    dpf_in: &HashMap<String, (Vec<f32>, usize)>,
    dpf_ncols: &HashMap<String, usize>,
    opts: &TransformOptions,
    fd_step: f64,
) -> Result<PushFixelBuckets> {
    let dims = target_grid.dims;
    let total_voxels = (dims[0] * dims[1] * dims[2]) as usize;
    let target_inv = target_affine
        .try_inverse()
        .ok_or_else(|| OdxError::Format("target grid affine is singular".into()))?;

    let mut dirs: Vec<Vec<[f32; 3]>> = (0..total_voxels).map(|_| Vec::new()).collect();
    let mut dpf: HashMap<String, Vec<Vec<f32>>> = dpf_in
        .keys()
        .map(|name| (name.clone(), (0..total_voxels).map(|_| Vec::new()).collect()))
        .collect();

    let in_header = input.header();
    let source_affine = nalgebra::Matrix4::from_row_slice(&[
        in_header.voxel_to_rasmm[0][0], in_header.voxel_to_rasmm[0][1], in_header.voxel_to_rasmm[0][2], in_header.voxel_to_rasmm[0][3],
        in_header.voxel_to_rasmm[1][0], in_header.voxel_to_rasmm[1][1], in_header.voxel_to_rasmm[1][2], in_header.voxel_to_rasmm[1][3],
        in_header.voxel_to_rasmm[2][0], in_header.voxel_to_rasmm[2][1], in_header.voxel_to_rasmm[2][2], in_header.voxel_to_rasmm[2][3],
        in_header.voxel_to_rasmm[3][0], in_header.voxel_to_rasmm[3][1], in_header.voxel_to_rasmm[3][2], in_header.voxel_to_rasmm[3][3],
    ]);
    let source_ijk = input.compact_to_ijk();

    for (compact_row, &ijk) in source_ijk.iter().enumerate() {
        // Source voxel center in source RAS+ mm.
        let p_src_v = source_affine
            * Vector4::new(ijk[0] as f64, ijk[1] as f64, ijk[2] as f64, 1.0);
        let p_src = [p_src_v[0], p_src_v[1], p_src_v[2]];
        // Push to target: fixel_chain maps source → target.
        let p_target = fixel_chain.map_point(p_src);
        // Invert target affine to find target voxel index.
        let v_t = target_inv * Vector4::new(p_target[0], p_target[1], p_target[2], 1.0);
        let ti = v_t[0].round() as i64;
        let tj = v_t[1].round() as i64;
        let tk = v_t[2].round() as i64;
        if ti < 0 || tj < 0 || tk < 0 {
            continue;
        }
        let (ti, tj, tk) = (ti as u64, tj as u64, tk as u64);
        if ti >= dims[0] || tj >= dims[1] || tk >= dims[2] {
            continue;
        }
        let target_flat = (ti * dims[1] * dims[2] + tj * dims[2] + tk) as usize;
        if pull_mask[target_flat] == 0 {
            continue;
        }

        // Reorientation: J of fixel_chain at p_src maps source d → target d directly.
        let j_chain_fixel = if fixel_chain.is_empty() {
            Matrix3::<f64>::identity()
        } else {
            fixel_chain.jacobian_at(p_src, fd_step)
        };
        let det_j = j_chain_fixel.determinant();
        // Modulation factor for fixel amplitudes: amp_new = amp_old / det(J_fwd).
        // For push, J_fwd = j_chain_fixel directly (chain already source→target).
        let amp_factor = if opts.modulate_fixel && det_j.is_finite() && det_j.abs() > 1e-12 {
            (1.0 / det_j) as f32
        } else {
            1.0
        };

        let f_start = offsets_in[compact_row] as usize;
        let f_end = offsets_in[compact_row + 1] as usize;
        for f in f_start..f_end {
            let d = directions_in[f];
            let v = j_chain_fixel
                * nalgebra::Vector3::new(d[0] as f64, d[1] as f64, d[2] as f64);
            let n = v.norm().max(1e-12);
            dirs[target_flat].push([(v[0] / n) as f32, (v[1] / n) as f32, (v[2] / n) as f32]);
            for (name, (data, ncols)) in dpf_in {
                let factor = if is_modulated_dpf(name, &opts.modulated_dpf_names) {
                    amp_factor
                } else {
                    1.0
                };
                let row_start = f * ncols;
                let bucket = dpf.get_mut(name).unwrap();
                let target_bucket = &mut bucket[target_flat];
                for c in 0..*ncols {
                    target_bucket.push(data[row_start + c] * factor);
                }
            }
        }
        // (Avoid unused-binding warning on dpf_ncols; map names line up.)
        let _ = dpf_ncols;
    }

    Ok(PushFixelBuckets { dirs, dpf })
}

#[allow(clippy::too_many_arguments)]
fn assemble_output(
    target_grid: &TargetGrid,
    new_mask: Vec<u8>,
    out_dirs: Vec<[f32; 3]>,
    out_offsets: Vec<u32>,
    sh_out_compact: HashMap<String, Vec<f32>>,
    dpv_out_compact: HashMap<String, Vec<f32>>,
    dpf_out_compact: HashMap<String, Vec<f32>>,
    sh_ncoeffs: HashMap<String, usize>,
    dpv_ncols: HashMap<String, usize>,
    dpf_ncols: HashMap<String, usize>,
    input: &OdxDataset,
) -> Result<OdxDataset> {
    let in_header = input.header();
    let mut affine_arr = [[0.0; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            affine_arr[r][c] = target_grid.affine[r][c];
        }
    }
    let mut builder = OdxBuilder::new(affine_arr, target_grid.dims, new_mask);
    let nb_voxels_out = out_offsets.len() - 1;
    for v in 0..nb_voxels_out {
        let s = out_offsets[v] as usize;
        let e = out_offsets[v + 1] as usize;
        builder.push_voxel_peaks(&out_dirs[s..e]);
    }
    if let (Some(order), Some(basis_name)) = (in_header.sh_order, in_header.sh_basis.as_deref()) {
        builder.set_sh_info(order, basis_name.to_string());
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
    for (name, data) in sh_out_compact {
        let ncols = *sh_ncoeffs.get(&name).unwrap();
        builder.set_sh_data(&name, vec_f32_to_bytes(data), ncols, DType::Float32);
    }
    for (name, data) in dpv_out_compact {
        let ncols = *dpv_ncols.get(&name).unwrap();
        builder.set_dpv_data(&name, vec_f32_to_bytes(data), ncols, DType::Float32);
    }
    for (name, data) in dpf_out_compact {
        let ncols = *dpf_ncols.get(&name).unwrap();
        builder.set_dpf_data(&name, vec_f32_to_bytes(data), ncols, DType::Float32);
    }
    for (k, v) in &in_header.extra {
        builder.set_extra_value(k.clone(), v.clone());
    }
    builder.finalize()
}

fn is_modulated_dpf(name: &str, list: &[String]) -> bool {
    let lc = name.to_ascii_lowercase();
    list.iter().any(|n| n.eq_ignore_ascii_case(&lc))
}

fn vec_f32_to_bytes(v: Vec<f32>) -> Vec<u8> {
    crate::mmap_backing::vec_into_bytes(v)
}

fn grid_affine_matrix(g: &TargetGrid) -> nalgebra::Matrix4<f64> {
    nalgebra::Matrix4::from_row_slice(&[
        g.affine[0][0], g.affine[0][1], g.affine[0][2], g.affine[0][3],
        g.affine[1][0], g.affine[1][1], g.affine[1][2], g.affine[1][3],
        g.affine[2][0], g.affine[2][1], g.affine[2][2], g.affine[2][3],
        g.affine[3][0], g.affine[3][1], g.affine[3][2], g.affine[3][3],
    ])
}

// Read-only crate-level helpers (kept private to the transform module).
impl OdxDataset {
    pub(crate) fn sh_arrays_get(&self, name: &str) -> Option<&crate::data_array::DataArray> {
        self.sh_arrays().get(name)
    }
    pub(crate) fn dpv_arrays_get(&self, name: &str) -> Option<&crate::data_array::DataArray> {
        self.dpv_arrays().get(name)
    }
    pub(crate) fn dpf_arrays_get(&self, name: &str) -> Option<&crate::data_array::DataArray> {
        self.dpf_arrays().get(name)
    }
}
