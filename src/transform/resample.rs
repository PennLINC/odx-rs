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
    /// Emit a fibre cross-section (FC) DPF under this name. Default: off.
    ///
    /// FC is the morphological companion to AFD in the fixel-based analysis
    /// framework (Raffelt et al. 2017): the change in fibre-bundle
    /// cross-sectional area, in the plane perpendicular to the fixel, implied
    /// by the warp. It derives *entirely* from the deformation — no diffusion
    /// signal enters it.
    ///
    /// Matches `warp2metric -fc`:
    ///
    /// ```text
    ///   FC = det(J) / ‖J · v‖
    /// ```
    ///
    /// where `J = d(subject)/d(template)` and `v` is the **template-space**
    /// (i.e. output) fixel direction. `det(J)` is the volume ratio and `‖J·v‖`
    /// the length ratio along the fibre, so their quotient is the ratio of
    /// areas perpendicular to it.
    ///
    /// Note the orientation: `d(subject)/d(template)` is the Jacobian of the
    /// map that *pulls* template coordinates back to subject coordinates. In
    /// mrtrix3 that is the Jacobian of `subject2template_warp.mif`, which
    /// despite its name is defined on the template grid and stores subject
    /// coordinates. The pull path's `chain` has exactly this orientation and
    /// is used directly; the push path's `fixel_chain` runs the other way and
    /// is inverted first.
    ///
    /// Like mrtrix3's, this FC is *relative to the template*: interpretable
    /// only within a study sharing one template, and usually log-transformed
    /// for statistics.
    pub fc_dpf_name: Option<String>,
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
            fc_dpf_name: None,
        }
    }
}

/// Fibre cross-section from a warp Jacobian and a template-space direction.
///
/// `j_t2s` must be `d(subject)/d(template)`; `v` the template-space fixel
/// direction (need not be unit — it is normalized here, matching
/// `warp2metric`'s explicit `fixel_direction.normalize()`).
///
/// Returns `None` when the result would not be a usable positive ratio: a
/// singular or near-singular Jacobian, a direction that collapses to zero
/// length under `J`, or any non-finite input. Callers substitute NaN rather
/// than silently emitting a garbage finite value — a spurious `FC = 0` or
/// `FC = 1e12` would survive a log-transform and poison downstream statistics,
/// whereas NaN is caught by ModelArray's finite-value thresholds.
///
/// mrtrix3 performs no such guarding; this is a deliberate robustification.
///
/// ATTRIBUTION: the `det(J) / ‖J·v‖` formulation and the explicit direction
/// normalisation are taken from MRtrix3's `cmd/warp2metric.cpp` (`-fc`),
/// copyright (c) 2008-2026 the MRtrix3 contributors, which implements the FC
/// metric of Raffelt et al. 2017 (NeuroImage 144:58-73). This function is a
/// derivative work of that code and is made available under the terms of the
/// Mozilla Public License, v. 2.0 (see `LICENSE-MRTRIX`); the remainder of this
/// file is under odx-rs's own terms.
#[inline]
fn fibre_cross_section(j_t2s: &Matrix3<f64>, v: [f32; 3]) -> Option<f32> {
    let det = j_t2s.determinant();
    if !det.is_finite() || det <= 0.0 {
        // det <= 0 means the warp folded (non-diffeomorphic) at this point.
        return None;
    }
    let d = nalgebra::Vector3::new(v[0] as f64, v[1] as f64, v[2] as f64);
    let dn = d.norm();
    if !dn.is_finite() || dn < 1e-12 {
        return None;
    }
    let stretched = j_t2s * (d / dn);
    let len = stretched.norm();
    if !len.is_finite() || len < 1e-12 {
        return None;
    }
    let fc = det / len;
    if fc.is_finite() && fc > 0.0 { Some(fc as f32) } else { None }
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
    // FC is *derived* here rather than carried from the input, so it gets its
    // own accumulator and is merged into the DPF maps at assembly time. If the
    // input already has a DPF under this name it is overwritten, since a
    // carried-through FC would refer to a previous warp, not this one.
    let mut fc_out_compact: Vec<f32> = Vec::new();

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
                    if opts.fc_dpf_name.is_some() {
                        let row = &buckets.fc[flat];
                        debug_assert_eq!(row.len(), dirs.len());
                        fc_out_compact.extend_from_slice(row);
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
                        let dir_out =
                            [(v[0] / n) as f32, (v[1] / n) as f32, (v[2] / n) as f32];
                        out_dirs.push(dir_out);
                        if opts.fc_dpf_name.is_some() {
                            // `j_chain` is target→source, i.e. d(subject)/d(template):
                            // exactly warp2metric's Jacobian. Use it directly.
                            fc_out_compact.push(
                                fibre_cross_section(&j_chain, dir_out).unwrap_or(f32::NAN),
                            );
                        }
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

    // Merge the derived FC in alongside the carried-through DPFs. Done here so
    // both the push and pull paths converge on one place, and so an FC already
    // present on the input (from a previous warp) is replaced rather than kept.
    let mut dpf_ncols = dpf_ncols;
    if let Some(name) = &opts.fc_dpf_name {
        debug_assert_eq!(fc_out_compact.len(), out_dirs.len());
        dpf_out_compact.insert(name.clone(), fc_out_compact);
        dpf_ncols.insert(name.clone(), 1);
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
    /// Per-target-voxel FC values, parallel to `dirs`. Empty when FC is off.
    fc: Vec<Vec<f32>>,
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
    let want_fc = opts.fc_dpf_name.is_some();
    let mut fc: Vec<Vec<f32>> = if want_fc {
        (0..total_voxels).map(|_| Vec::new()).collect()
    } else {
        Vec::new()
    };

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
            let dir_out = [(v[0] / n) as f32, (v[1] / n) as f32, (v[2] / n) as f32];
            dirs[target_flat].push(dir_out);
            if want_fc {
                // `fixel_chain` maps source→target, i.e. d(template)/d(subject).
                // warp2metric wants the other orientation, so invert. A singular
                // Jacobian yields NaN rather than a fabricated value.
                let val = j_chain_fixel
                    .try_inverse()
                    .and_then(|j_t2s| fibre_cross_section(&j_t2s, dir_out))
                    .unwrap_or(f32::NAN);
                fc[target_flat].push(val);
            }
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

    Ok(PushFixelBuckets { dirs, dpf, fc })
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

#[cfg(test)]
mod fc_tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * b.abs().max(1.0)
    }

    /// Identity warp deforms nothing, so every fixel has FC exactly 1.
    #[test]
    fn identity_jacobian_gives_unit_fc() {
        let j = Matrix3::<f64>::identity();
        for v in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.577, 0.577, 0.577]] {
            let fc = fibre_cross_section(&j, v).expect("finite");
            assert!(approx(fc, 1.0, 1e-6), "expected 1.0, got {fc}");
        }
    }

    /// Pure isotropic scaling by s: volume scales s^3, length along any fixel
    /// scales s, so the perpendicular area scales s^2.
    #[test]
    fn isotropic_scaling_gives_s_squared() {
        for s in [0.5_f64, 1.0, 2.0, 3.0] {
            let j = Matrix3::<f64>::from_diagonal(&nalgebra::Vector3::new(s, s, s));
            let fc = fibre_cross_section(&j, [0.0, 0.0, 1.0]).expect("finite");
            assert!(
                approx(fc, (s * s) as f32, 1e-5),
                "s={s}: expected {}, got {fc}",
                s * s
            );
        }
    }

    /// Anisotropic case — the discriminating test. Stretch x by 2 and y by 3,
    /// leaving z alone: det = 6. A fixel along z has |J·z| = 1, so FC = 6 (the
    /// full xy-area change). A fixel along x has |J·x| = 2, so FC = 3 (only the
    /// yz-plane change). This is what distinguishes FC from the Jacobian
    /// determinant, and it fails if the direction is dropped or mis-normalized.
    #[test]
    fn anisotropic_scaling_is_direction_dependent() {
        let j = Matrix3::<f64>::from_diagonal(&nalgebra::Vector3::new(2.0, 3.0, 1.0));
        assert!(approx(fibre_cross_section(&j, [0.0, 0.0, 1.0]).unwrap(), 6.0, 1e-5));
        assert!(approx(fibre_cross_section(&j, [1.0, 0.0, 0.0]).unwrap(), 3.0, 1e-5));
        assert!(approx(fibre_cross_section(&j, [0.0, 1.0, 0.0]).unwrap(), 2.0, 1e-5));
    }

    /// Direction need not arrive unit-length; warp2metric normalizes explicitly
    /// and so do we. A non-unit input must not scale the answer.
    #[test]
    fn direction_is_normalized_before_use() {
        let j = Matrix3::<f64>::from_diagonal(&nalgebra::Vector3::new(2.0, 3.0, 1.0));
        let unit = fibre_cross_section(&j, [0.0, 0.0, 1.0]).unwrap();
        let long = fibre_cross_section(&j, [0.0, 0.0, 17.0]).unwrap();
        assert!(approx(unit, long, 1e-6), "{unit} vs {long}");
    }

    /// Robustification beyond mrtrix3: degenerate inputs yield None (written as
    /// NaN) instead of Inf/0/negative values that would survive a log and
    /// corrupt group statistics.
    #[test]
    fn degenerate_inputs_return_none() {
        let singular = Matrix3::<f64>::from_diagonal(&nalgebra::Vector3::new(1.0, 1.0, 0.0));
        assert!(fibre_cross_section(&singular, [0.0, 0.0, 1.0]).is_none(), "singular J");

        let folded = Matrix3::<f64>::from_diagonal(&nalgebra::Vector3::new(-1.0, 1.0, 1.0));
        assert!(fibre_cross_section(&folded, [0.0, 0.0, 1.0]).is_none(), "det < 0 (folded warp)");

        let ok = Matrix3::<f64>::identity();
        assert!(fibre_cross_section(&ok, [0.0, 0.0, 0.0]).is_none(), "zero-length direction");

        let nan_j = Matrix3::<f64>::from_diagonal(&nalgebra::Vector3::new(f64::NAN, 1.0, 1.0));
        assert!(fibre_cross_section(&nan_j, [0.0, 0.0, 1.0]).is_none(), "non-finite J");
    }

    /// A shear has det = 1 but still changes cross-section for fixels not
    /// aligned with the shear-invariant direction. Guards against anyone
    /// "simplifying" FC to det(J).
    #[test]
    fn shear_is_not_the_jacobian_determinant() {
        let mut j = Matrix3::<f64>::identity();
        j[(0, 1)] = 1.0; // x += y
        assert!((j.determinant() - 1.0).abs() < 1e-12);
        let along_z = fibre_cross_section(&j, [0.0, 0.0, 1.0]).unwrap();
        let along_y = fibre_cross_section(&j, [0.0, 1.0, 0.0]).unwrap();
        assert!(approx(along_z, 1.0, 1e-6), "z unaffected by x+=y shear: {along_z}");
        // |J·y| = |(1,1,0)| = sqrt(2), so FC = 1/sqrt(2).
        assert!(approx(along_y, std::f32::consts::FRAC_1_SQRT_2, 1e-5), "{along_y}");
    }
}
