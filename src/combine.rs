//! N-way template fixel correspondence + angular variance across many ODX
//! files warped into one template space.
//!
//! The N-way generalization of [`crate::compare`]: build a shared set of
//! **group fixels** (reference directions per voxel), match every subject's
//! fixels onto them, and summarize the **angular spread of subject directions
//! around each group fixel**. Per-subject angles (and matched scalars) are
//! stored as **multi-column DPF** `(n_fixels × n_subjects)` in a group ODX
//! alongside single-column
//! summary DPFs (for trxviz) and a cohort CSV for ModelArrayIO/ModelArray.
//!
//! Group fixels come from one of two methods (selectable, to compare
//! robustness): `cluster` pools subject directions per voxel and clusters them
//! by the dyadic-tensor mean (amplitude-agnostic); `mean-fod` averages the SH
//! coefficients and peak-finds the mean FOD. A `--template` override adopts an
//! existing fixel set verbatim.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use nalgebra::{Matrix3, SymmetricEigen};
use serde::Serialize;
use serde_json::json;

use crate::dtype::DType;
use crate::error::{OdxError, Result};
use crate::fixel_match::{
    abs_dot, affine_close, align_to_reference_grid, flat_index, mask_compact_ijk,
    shared_scalar_keys, ArrayKind,
};
use crate::mmap_backing::vec_into_bytes;
use crate::nifti_export::{write_voxel_scalar_nifti_f32, write_voxel_scalar_nifti_u8};
use crate::odx_file::OdxDataset;
use crate::peak_finder::PeakFinderConfig;
use crate::stream::OdxBuilder;
use crate::template::{
    aggregate_fod, average_dpvs, aggregate_anisotropic_power, coverage_mask, flag_outliers, fod_qc,
    peaks_from_aggregate, prepare_inputs, reference_header, AggregateOptions, AggregatedFod,
    LmaxPolicy, LooMode, PreparedInput, ScaleMode, ShTarget,
};

/// Per-subject FOD rescaling for `mean-fod`, applied *before* averaging.
/// Re-exported from [`crate::template`] so the existing `--normalize-fod`
/// spelling keeps working.
pub use crate::template::ScaleMode as NormalizeFod;

/// How the group/template fixels are established.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TemplateMethod {
    /// Pool subject fixel directions per voxel and cluster by dyadic-tensor
    /// mean. Amplitude-agnostic — robust to per-subject SH scaling.
    Cluster,
    /// Average `sh/coefficients` across inputs and peak-find the mean FOD.
    MeanFod,
}

/// How to combine the per-subject masks into the template voxel set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskCombine {
    /// Keep voxels covered by ≥1 input.
    Union,
    /// Keep voxels covered by every input.
    Intersection,
}

/// One subject input: its ODX path, a stable key (DPF column / cohort row
/// identity), and the categorical row joined from the design table (in design
/// header order; empty if none).
pub struct CombineInput {
    pub path: PathBuf,
    pub key: String,
    pub categorical: Vec<(String, String)>,
    /// True for scans in the reference cohort (define the scaffold + its
    /// support); used to compute `scaffold_support` from reference scans only.
    pub is_reference: bool,
    /// Optional method label (the design column identifying the processing
    /// method) used for `n_methods_detecting`.
    pub method: Option<String>,
}

#[derive(Clone, Debug)]
pub struct CombineOptions {
    pub method: TemplateMethod,
    /// Adopt this ODX's fixels/geometry as the template, skipping the build.
    pub template_override: Option<PathBuf>,
    pub mask_combine: MaskCombine,
    pub match_angle_deg: f32,
    pub peak_config: PeakFinderConfig,
    pub normalize_fod: ScaleMode,
    /// Keep a voxel when at least this fraction of inputs cover it. Overrides
    /// `mask_combine` when set (`0` ≡ union, `1` ≡ intersection).
    pub min_coverage: Option<f32>,
    /// How the cohort's SH order is reconciled when inputs disagree.
    pub lmax: LmaxPolicy,
    /// Header/grid/basis reference; defaults to the first input.
    pub reference: Option<PathBuf>,
    /// Compute the FOD reproducibility block under methods other than
    /// `mean-fod` (it is always on for `mean-fod` unless `no_fod_qc`).
    pub fod_qc: bool,
    /// Suppress the FOD reproducibility block entirely.
    pub no_fod_qc: bool,
    pub loo: LooMode,
    /// Lowest SH band included in the angular correlation coefficient.
    pub acc_lmin: usize,
    /// Scalar DPVs to average onto the template. `None` averages every shared
    /// scalar float DPV; `Some(vec![])` disables DPV averaging.
    pub average_dpv: Option<Vec<String>>,
    /// Also emit `<name>_sd` beside each averaged DPV.
    pub dpv_sd: bool,
    /// `cluster`: minimum distinct subjects supporting a group fixel.
    pub min_subjects_per_group_fixel: usize,
    /// Restrict carried scalars; `None` carries every shared scalar float DPF.
    pub matched_scalars: Option<Vec<String>>,
    /// Extra match-angle thresholds (degrees) at which to emit `matched_at_<deg>`
    /// detection planes, so a verdict's threshold sensitivity is a reported axis.
    pub match_angle_sweep: Vec<f32>,
}

impl Default for CombineOptions {
    fn default() -> Self {
        Self {
            method: TemplateMethod::Cluster,
            template_override: None,
            mask_combine: MaskCombine::Union,
            match_angle_deg: 30.0,
            peak_config: PeakFinderConfig::default(),
            normalize_fod: ScaleMode::None,
            min_coverage: None,
            lmax: LmaxPolicy::Min,
            reference: None,
            fod_qc: false,
            no_fod_qc: false,
            loo: LooMode::Auto,
            acc_lmin: 2,
            average_dpv: None,
            dpv_sd: false,
            min_subjects_per_group_fixel: 2,
            matched_scalars: None,
            match_angle_sweep: Vec::new(),
        }
    }
}

/// Output sinks; each `Some` path is written.
#[derive(Default)]
pub struct CombineOutputs {
    pub out_odx: Option<PathBuf>,
    pub out_cohort: Option<PathBuf>,
    pub out_mask: Option<PathBuf>,
    pub per_subject_odx_dir: Option<PathBuf>,
    pub out_table: Option<PathBuf>,
    pub out_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CombineReport {
    pub method: String,
    pub n_inputs: usize,
    pub mask_combine: String,
    pub match_angle_deg: f32,
    pub normalize_fod: String,
    pub dims: [u64; 3],
    pub n_template_voxels: u64,
    pub n_template_fixels: u64,
    pub mean_subjects_per_fixel: f64,
    pub mean_angle_deg: Option<f64>,
    pub mean_unmatched_per_scan: f64,
    pub n_reference_scans: usize,
    pub matched_scalar_keys: Vec<String>,
    pub design_columns: Vec<String>,
    /// Voxel-inclusion threshold actually applied (`--min-coverage`, or the
    /// `--mask-combine` equivalent).
    pub min_coverage: f32,
    pub lmax_policy: String,
    /// Resolved SH metadata of the aggregate; `None` when no FOD was averaged.
    pub sh_order: Option<u64>,
    pub sh_basis: Option<String>,
    /// `"on"`, `"off"`, or `"unavailable"` when no FOD block ran.
    pub loo: String,
    pub acc_lmin: usize,
    pub mean_acc: Option<f64>,
    pub mean_acc_loo: Option<f64>,
    /// Template voxels whose aggregate carries no anisotropic energy, so ACC is
    /// undefined there and they are excluded from every ACC summary. Large in
    /// multi-tissue cohorts, where the WM compartment is identically zero over
    /// much of the brain mask — read the ACC numbers against this count.
    pub n_voxels_without_orientation: u64,
    pub averaged_dpv: Vec<String>,
    pub subjects: Vec<CombineSubjectRow>,
    /// Keys of the subjects flagged by the outlier rule.
    pub outliers: Vec<String>,
    pub written_paths: Vec<String>,
}

/// Per-subject QC, so a scan that does not belong in the template is visible
/// rather than silently averaged in.
#[derive(Debug, Clone, Serialize)]
pub struct CombineSubjectRow {
    pub key: String,
    pub path: String,
    /// Template voxels this input covers.
    pub n_voxels: u64,
    pub coverage_frac: f64,
    pub n_fixels: u64,
    /// Mean angular correlation against the template, over covered voxels.
    /// `NaN` when the FOD block did not run.
    pub mean_acc: f32,
    /// Mean leave-one-out angular correlation. `NaN` when LOO is off.
    pub mean_acc_loo: f32,
    /// This input's SH had to be rewritten into the reference basis.
    pub basis_converted: bool,
    /// Set when the input's own lmax exceeded the resolved cohort lmax.
    pub lmax_truncated_from: Option<u64>,
    pub is_outlier: bool,
    pub outlier_reasons: Vec<String>,
}

/// Group fixels in template compact-voxel order, plus per-fixel metadata.
struct Template {
    mask: Vec<u8>,
    ijk: Vec<[u32; 3]>,
    offsets: Vec<u32>,
    dirs: Vec<[f32; 3]>,
    rank: Vec<u32>,
    is_primary: Vec<u8>,
    strength: Vec<f32>,
    /// The aggregated FOD field for `mean-fod`, carrying the resolved SH
    /// metadata the writer needs.
    mean_sh: Option<AggregatedFod>,
}

/// Combine many template-space ODX files into a group ODX with per-subject
/// angular-distance DPF + summary statistics, and the ModelArray cohort.
pub fn combine_odx(
    inputs: &[CombineInput],
    opts: &CombineOptions,
    out: &CombineOutputs,
) -> Result<CombineReport> {
    if inputs.is_empty() {
        return Err(OdxError::Argument(
            "combine requires at least one input ODX".into(),
        ));
    }
    if !opts.match_angle_deg.is_finite() || opts.match_angle_deg <= 0.0 || opts.match_angle_deg >= 90.0
    {
        return Err(OdxError::Argument(format!(
            "match_angle_deg must be in (0, 90), got {}",
            opts.match_angle_deg
        )));
    }
    // Validate the detection-sweep thresholds: same (0, 90) domain as the main
    // angle, and no two values may round to the same integer degree (their
    // `matched_at_<deg>` planes would silently collide).
    {
        let mut rounded = BTreeSet::new();
        for &deg in &opts.match_angle_sweep {
            if !deg.is_finite() || deg <= 0.0 || deg >= 90.0 {
                return Err(OdxError::Argument(format!(
                    "match_angle_sweep value must be in (0, 90), got {deg}"
                )));
            }
            if !rounded.insert(deg.round() as i32) {
                return Err(OdxError::Argument(format!(
                    "match_angle_sweep values round to the same integer degree ({}); \
                     the matched_at_<deg> planes would collide",
                    deg.round() as i32
                )));
            }
        }
    }

    let datasets: Vec<OdxDataset> = inputs
        .iter()
        .map(|i| {
            OdxDataset::open(&i.path).map_err(|e| {
                OdxError::Format(format!("failed to open input '{}': {e}", i.path.display()))
            })
        })
        .collect::<Result<_>>()?;
    let n_inputs = datasets.len();

    let dims = datasets[0].header().dimensions;
    let affine = datasets[0].header().voxel_to_rasmm;

    let (ny, nz) = (dims[1] as usize, dims[2] as usize);

    // Per-input **reference-grid** flat → that input's compact row. Inputs whose
    // voxels are ordered differently (LAS vs RAS+, any signed axis permutation)
    // but sit on the same physical lattice are reindexed here, so every
    // downstream `lookup[ref_flat]` is directly comparable across inputs.
    let input_lookup: Vec<Vec<usize>> = datasets
        .iter()
        .zip(inputs)
        .map(|(ds, inp)| {
            let h = ds.header();
            align_to_reference_grid(
                dims,
                &affine,
                h.dimensions,
                &h.voxel_to_rasmm,
                ds.mask(),
                &inp.path.display().to_string(),
            )
        })
        .collect::<Result<_>>()?;
    let input_offsets: Vec<&[u32]> = datasets.iter().map(|ds| ds.offsets()).collect();

    let cos_thresh = opts.match_angle_deg.to_radians().cos();

    // ── Prepare inputs for FOD aggregation ────────────────────────────────
    // Needed by `mean-fod`, and by the QC block under any method. Skipped when
    // no SH is available, so `--method cluster` still works on peak-only files.
    let labels: Vec<String> = inputs.iter().map(|i| i.path.display().to_string()).collect();
    let want_fod = matches!(opts.method, TemplateMethod::MeanFod) || opts.fod_qc;
    let has_sh = datasets.iter().all(|d| d.get_sh("coefficients").is_some());
    let agg_opts = AggregateOptions {
        scale: opts.normalize_fod,
        min_coverage: opts.min_coverage.unwrap_or(match opts.mask_combine {
            MaskCombine::Union => 0.0,
            MaskCombine::Intersection => 1.0,
        }),
        lmax: opts.lmax,
        loo: opts.loo,
        acc_lmin: opts.acc_lmin,
    };
    let fod_prepared: Option<(Vec<PreparedInput>, ShTarget)> = if want_fod {
        if !has_sh && !matches!(opts.method, TemplateMethod::MeanFod) {
            eprintln!(
                "odx combine: warning: --fod-qc needs sh/coefficients on every input; \
                 skipping the FOD QC block"
            );
            None
        } else {
            let (ref_target, _, _) =
                reference_header(&datasets, &labels, opts.reference.as_deref())?;
            Some(prepare_inputs(
                &datasets,
                &labels,
                dims,
                &affine,
                &ref_target,
                opts.lmax,
            )?)
        }
    } else {
        None
    };

    // ── Build the template ────────────────────────────────────────────────
    let mut template = if let Some(tpath) = opts.template_override.as_ref() {
        build_template_override(tpath, dims, &affine)?
    } else {
        // Coverage counts → template mask. `--min-coverage` generalizes
        // `--mask-combine`: 0 is a union, 1 an intersection.
        let mask = coverage_mask(&input_lookup, dims, agg_opts.min_coverage);
        match opts.method {
            TemplateMethod::Cluster => build_template_cluster(
                &datasets,
                &input_lookup,
                &input_offsets,
                &mask,
                dims,
                cos_thresh,
                opts.min_subjects_per_group_fixel,
            )?,
            TemplateMethod::MeanFod => {
                let (prepared, target) = fod_prepared
                    .as_ref()
                    .expect("mean-fod always prepares inputs");
                build_template_mean_fod(
                    &datasets,
                    prepared,
                    &mask,
                    dims,
                    target,
                    &agg_opts,
                    &opts.peak_config,
                )?
            }
        }
    };

    // ── Aggregate the FOD onto the template voxel set ─────────────────────
    // `mean-fod` already did this (its fixels *are* the aggregate's peaks); the
    // other methods aggregate onto the template they just built, so the QC maps
    // line up with the template's voxels either way.
    let aggregate: Option<AggregatedFod> = if opts.no_fod_qc {
        template.mean_sh.take()
    } else if let Some(agg) = template.mean_sh.take() {
        Some(agg)
    } else if let Some((prepared, target)) = fod_prepared.as_ref() {
        Some(aggregate_fod(
            &datasets,
            prepared,
            dims,
            &template.mask,
            target,
            &agg_opts,
        )?)
    } else {
        None
    };
    let qc = match (opts.no_fod_qc, aggregate.as_ref(), fod_prepared.as_ref()) {
        (false, Some(agg), Some((prepared, _))) => {
            Some(fod_qc(&datasets, prepared, dims, agg, &agg_opts)?)
        }
        _ => None,
    };
    // Scalar DPVs shared by every input, averaged with the same per-voxel
    // contributor divisor as the FOD.
    let dpv_means: Vec<(String, Vec<f32>, Vec<f32>)> =
        match (aggregate.as_ref(), fod_prepared.as_ref()) {
            (Some(agg), Some((prepared, _))) if opts.average_dpv != Some(Vec::new()) => {
                let keys = shared_scalar_keys(
                    &datasets,
                    ArrayKind::Dpv,
                    opts.average_dpv.as_deref(),
                );
                average_dpvs(&datasets, prepared, dims, agg, &keys)?
            }
            _ => Vec::new(),
        };

    // Sign-canonicalize reference directions so the tangent frame + signed
    // residual are deterministic (fixels are undirected). For 'cluster' these
    // are already canonical; this also covers mean-fod and --template.
    for d in template.dirs.iter_mut() {
        *d = sign_canon(*d);
    }
    let n_vox = template.ijk.len();
    let n_fixels = template.dirs.len();
    // Deterministic, method-independent per-fixel tangent frame (depends only on
    // the reference direction) — the frame against which signed tilt is measured.
    let frames: Vec<([f32; 3], [f32; 3])> =
        template.dirs.iter().map(|&r| tangent_basis(r)).collect();

    // ── Discover carried scalar DPF keys (shared, scalar float) ───────────
    let scalar_keys = shared_scalar_keys(&datasets, ArrayKind::Dpf, opts.matched_scalars.as_deref());
    // [key][input] -> per-fixel scalar values
    let scalar_vals: Vec<Vec<Vec<f32>>> = scalar_keys
        .iter()
        .map(|k| {
            datasets
                .iter()
                .map(|ds| ds.scalar_dpf_f32(k))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<_>>()?;

    // ── Per-subject (n_fixels × n_inputs) buffers + summary accumulators ──
    let nf_ns = n_fixels * n_inputs;
    let mut angle = vec![f32::NAN; nf_ns];
    // detection per (fixel × scan): NaN = scan had no coverage (FOV) at this
    // voxel, 0.0 = covered but the method recovered no corresponding fixel,
    // 1.0 = matched. NaN (not 0) for no-coverage so the detection model can't
    // mistake out-of-FOV for a detection miss.
    let mut matched = vec![f32::NAN; nf_ns];
    let mut scalar_buf: Vec<Vec<f32>> = scalar_keys.iter().map(|_| vec![f32::NAN; nf_ns]).collect();

    let mut n_sub_matched = vec![0u32; n_fixels];
    let mut sum_angle = vec![0.0f64; n_fixels];
    let mut sum_angle_sq = vec![0.0f64; n_fixels];
    let mut tensor = vec![[0.0f64; 6]; n_fixels]; // xx,yy,zz,xy,xz,yz
    let mut scalar_sum: Vec<Vec<f64>> = scalar_keys.iter().map(|_| vec![0.0f64; n_fixels]).collect();
    let mut scalar_cnt: Vec<Vec<u32>> = scalar_keys.iter().map(|_| vec![0u32; n_fixels]).collect();

    let mut total_matched: u64 = 0;
    let mut total_angle: f64 = 0.0;

    // Method-comparison metrics (per fixel × scan unless noted).
    let mut best_ungated = vec![f32::NAN; nf_ns]; // nearest scaffold-fixel angle, no cone
    let mut runner_gap = vec![f32::NAN; nf_ns]; // angular gap best vs 2nd-best (fragility)
    let mut mutual_dp = vec![f32::NAN; nf_ns]; // |dot| of the mutual match (drives sweep)
    let mut tangent_a = vec![f32::NAN; nf_ns]; // signed tangent residual e1
    let mut tangent_b = vec![f32::NAN; nf_ns]; // signed tangent residual e2
    let mut n_unmatched = vec![0u32; n_inputs]; // per scan: fixels matching no scaffold fixel
    let mut scaffold_support = vec![0u32; n_fixels]; // reference-cohort matches per fixel

    // ── N-way matching ────────────────────────────────────────────────────
    for (s, ds) in datasets.iter().enumerate() {
        let dirs_s = ds.directions();
        let off_s = input_offsets[s];
        let lk_s = &input_lookup[s];
        let is_ref = inputs[s].is_reference;
        for (t, ijk) in template.ijk.iter().enumerate() {
            let flat = ijk[0] as usize * ny * nz + ijk[1] as usize * nz + ijk[2] as usize;
            let cs = lk_s[flat];
            if cs == usize::MAX {
                continue; // subject does not cover this voxel
            }
            let gstart = template.offsets[t] as usize;
            let gend = template.offsets[t + 1] as usize;
            let m = gend - gstart;
            // Covered voxel → every group fixel here is a detection candidate
            // (0.0 = covered-but-missed). `matched` stays NaN only where the scan
            // lacked coverage (cs==MAX above), so an out-of-FOV fixel is never
            // counted as a detection miss.
            for gf in gstart..gend {
                matched[gf * n_inputs + s] = 0.0;
            }
            if m == 0 {
                continue;
            }
            let sstart = off_s[cs] as usize;
            let send = off_s[cs + 1] as usize;
            let n = send - sstart;
            if n == 0 {
                continue;
            }
            let group = &template.dirs[gstart..gend];
            let subj = &dirs_s[sstart..send];

            let mut best_g_for_s = vec![(usize::MAX, -1.0f32); n];
            let mut best_s_for_g = vec![(usize::MAX, -1.0f32, -1.0f32); m]; // (si, dp1, dp2)
            for (si, &sd) in subj.iter().enumerate() {
                for (gi, &gd) in group.iter().enumerate() {
                    let dp = abs_dot(sd, gd);
                    if dp > best_g_for_s[si].1 {
                        best_g_for_s[si] = (gi, dp);
                    }
                    let bg = &mut best_s_for_g[gi];
                    if dp > bg.1 {
                        bg.2 = bg.1; // demote old best to runner-up
                        bg.0 = si;
                        bg.1 = dp;
                    } else if dp > bg.2 {
                        bg.2 = dp;
                    }
                }
            }
            for (gi, &(si, dp1, dp2)) in best_s_for_g.iter().enumerate() {
                if si == usize::MAX {
                    continue;
                }
                let gf = gstart + gi;
                let idx = gf * n_inputs + s;
                let ang1 = dp1.clamp(-1.0, 1.0).acos().to_degrees();
                best_ungated[idx] = ang1;
                if dp2 >= 0.0 {
                    runner_gap[idx] = dp2.clamp(-1.0, 1.0).acos().to_degrees() - ang1;
                }
                if best_g_for_s[si].0 != gi {
                    continue; // not mutual
                }
                mutual_dp[idx] = dp1;
                if dp1 < cos_thresh {
                    continue; // mutual but outside the cone
                }
                angle[idx] = ang1;
                matched[idx] = 1.0;
                let sfx = sstart + si;
                for (k, vals) in scalar_vals.iter().enumerate() {
                    let v = vals[s][sfx];
                    scalar_buf[k][idx] = v;
                    if v.is_finite() {
                        scalar_sum[k][gf] += v as f64;
                        scalar_cnt[k][gf] += 1;
                    }
                }
                let (ta, tb) = tangent_residual(subj[si], group[gi], frames[gf].0, frames[gf].1);
                tangent_a[idx] = ta;
                tangent_b[idx] = tb;
                n_sub_matched[gf] += 1;
                sum_angle[gf] += ang1 as f64;
                sum_angle_sq[gf] += (ang1 as f64) * (ang1 as f64);
                accumulate_tensor(&mut tensor[gf], subj[si]);
                if is_ref {
                    scaffold_support[gf] += 1;
                }
                total_matched += 1;
                total_angle += ang1 as f64;
            }
            // per scan: how many of this voxel's subject fixels matched no group fixel
            let mut accepted = 0usize;
            for (si, &(gi, dp)) in best_g_for_s.iter().enumerate() {
                if gi != usize::MAX && best_s_for_g[gi].0 == si && dp >= cos_thresh {
                    accepted += 1;
                }
            }
            n_unmatched[s] += (n - accepted) as u32;
        }
    }

    // ── Summary per-fixel DPF ─────────────────────────────────────────────
    let mut mean_angle = vec![f32::NAN; n_fixels];
    let mut sd_angle = vec![f32::NAN; n_fixels];
    let mut dispersion = vec![f32::NAN; n_fixels];
    for gf in 0..n_fixels {
        let n = n_sub_matched[gf] as f64;
        if n >= 1.0 {
            mean_angle[gf] = (sum_angle[gf] / n) as f32;
            dispersion[gf] = dyadic_dispersion(&tensor[gf], n);
        }
        if n >= 2.0 {
            let mean = sum_angle[gf] / n;
            let var = (sum_angle_sq[gf] - n * mean * mean) / (n - 1.0);
            sd_angle[gf] = var.max(0.0).sqrt() as f32;
        }
    }
    let mut mean_scalar: Vec<Vec<f32>> = scalar_keys
        .iter()
        .enumerate()
        .map(|(k, _)| {
            (0..n_fixels)
                .map(|gf| {
                    if scalar_cnt[k][gf] > 0 {
                        (scalar_sum[k][gf] / scalar_cnt[k][gf] as f64) as f32
                    } else {
                        f32::NAN
                    }
                })
                .collect()
        })
        .collect();

    // Reference-cohort support: matches among reference scans only; fall back to
    // all-scan support when no input is flagged reference.
    if !inputs.iter().any(|i| i.is_reference) {
        scaffold_support.copy_from_slice(&n_sub_matched);
    }
    // Crossing complexity per fixel (group fixels sharing its voxel).
    let mut ref_n_fixels_in_voxel = vec![0u32; n_fixels];
    for t in 0..n_vox {
        let st = template.offsets[t] as usize;
        let en = template.offsets[t + 1] as usize;
        let c = (en - st) as u32;
        for v in ref_n_fixels_in_voxel[st..en].iter_mut() {
            *v = c;
        }
    }
    // Distinct methods detecting each fixel (only when a method label is given).
    let n_methods_detecting: Option<Vec<u32>> = if inputs.iter().any(|i| i.method.is_some()) {
        let mut out = vec![0u32; n_fixels];
        let mut seen: Vec<&str> = Vec::new();
        for (gf, slot) in out.iter_mut().enumerate() {
            seen.clear();
            for (s, inp) in inputs.iter().enumerate() {
                if matched[gf * n_inputs + s] > 0.5 {
                    if let Some(meth) = inp.method.as_deref() {
                        if !seen.contains(&meth) {
                            seen.push(meth);
                        }
                    }
                }
            }
            *slot = seen.len() as u32;
        }
        Some(out)
    } else {
        None
    };

    // ── Assemble outputs ──────────────────────────────────────────────────
    let subjects: Vec<String> = inputs.iter().map(|i| i.key.clone()).collect();
    let design_columns = collect_design_columns(inputs);
    let mut written: Vec<String> = Vec::new();

    if let Some(p) = out.out_odx.as_ref() {
        let mut b = OdxBuilder::new(affine, dims, template.mask.clone());
        for t in 0..n_vox {
            let s = template.offsets[t] as usize;
            let e = template.offsets[t + 1] as usize;
            b.push_voxel_peaks(&template.dirs[s..e]);
        }
        // per-subject multi-column DPF
        b.set_dpf_data("angle_deg", vec_into_bytes(angle.clone()), n_inputs, DType::Float32);
        b.set_dpf_data("matched", vec_into_bytes(matched.clone()), n_inputs, DType::Float32);
        for (k, key) in scalar_keys.iter().enumerate() {
            b.set_dpf_data(
                &format!("subj_{key}"),
                vec_into_bytes(scalar_buf[k].clone()),
                n_inputs,
                DType::Float32,
            );
        }
        // single-column summary DPF
        b.set_dpf_data("mean_angle_deg", vec_into_bytes(mean_angle.clone()), 1, DType::Float32);
        b.set_dpf_data("sd_angle_deg", vec_into_bytes(sd_angle.clone()), 1, DType::Float32);
        b.set_dpf_data("dispersion", vec_into_bytes(dispersion.clone()), 1, DType::Float32);
        b.set_dpf_data("within_voxel_rank", vec_into_bytes(template.rank.clone()), 1, DType::UInt32);
        b.set_dpf_data("is_primary", template.is_primary.clone(), 1, DType::UInt8);
        b.set_dpf_data("n_subjects_matched", vec_into_bytes(n_sub_matched.clone()), 1, DType::UInt32);
        b.set_dpf_data("strength", vec_into_bytes(template.strength.clone()), 1, DType::Float32);
        // Under `mean-fod` the fixel strength IS the aggregate's peak amplitude,
        // so also publish it under the canonical name. That makes the template a
        // first-class ODX: `odx compare` and `odx qc` resolve their primary
        // metric as amplitude → afd → qa and would otherwise refuse it. Not
        // emitted for `cluster`, whose strength is a dyadic eigenvalue rather
        // than an amplitude.
        if aggregate.is_some() && matches!(opts.method, TemplateMethod::MeanFod) {
            b.set_dpf_data(
                "amplitude",
                vec_into_bytes(template.strength.clone()),
                1,
                DType::Float32,
            );
        }
        for (k, key) in scalar_keys.iter().enumerate() {
            b.set_dpf_data(
                &format!("mean_{key}"),
                vec_into_bytes(std::mem::take(&mut mean_scalar[k])),
                1,
                DType::Float32,
            );
        }
        // method-comparison per-scan arrays (detection / orientation / tilt)
        b.set_dpf_data(
            "best_angle_deg_ungated",
            vec_into_bytes(best_ungated.clone()),
            n_inputs,
            DType::Float32,
        );
        b.set_dpf_data(
            "runner_up_gap_deg",
            vec_into_bytes(runner_gap.clone()),
            n_inputs,
            DType::Float32,
        );
        b.set_dpf_data("tangent_a", vec_into_bytes(tangent_a.clone()), n_inputs, DType::Float32);
        b.set_dpf_data("tangent_b", vec_into_bytes(tangent_b.clone()), n_inputs, DType::Float32);
        // match-angle sweep detection planes (threshold sensitivity as a reported axis)
        for &deg in &opts.match_angle_sweep {
            let c = deg.to_radians().cos();
            let plane: Vec<u8> = mutual_dp
                .iter()
                .map(|&dp| u8::from(dp.is_finite() && dp >= c))
                .collect();
            b.set_dpf_data(
                &format!("matched_at_{}", deg.round() as i32),
                plane,
                n_inputs,
                DType::UInt8,
            );
        }
        // method-comparison summary
        b.set_dpf_data(
            "scaffold_support",
            vec_into_bytes(scaffold_support.clone()),
            1,
            DType::UInt32,
        );
        b.set_dpf_data(
            "ref_n_fixels_in_voxel",
            vec_into_bytes(ref_n_fixels_in_voxel.clone()),
            1,
            DType::UInt32,
        );
        if let Some(nmd) = n_methods_detecting.as_ref() {
            b.set_dpf_data("n_methods_detecting", vec_into_bytes(nmd.clone()), 1, DType::UInt32);
        }
        // the method-independent tangent frame, for reproducible re-projection
        let e1_flat: Vec<f32> = frames.iter().flat_map(|f| f.0).collect();
        let e2_flat: Vec<f32> = frames.iter().flat_map(|f| f.1).collect();
        b.set_dpf_data("tangent_e1", vec_into_bytes(e1_flat), 3, DType::Float32);
        b.set_dpf_data("tangent_e2", vec_into_bytes(e2_flat), 3, DType::Float32);
        // ── FOD reproducibility block (per-voxel) ─────────────────────────
        for (name, data) in voxel_qc_arrays(&aggregate, &qc, &dpv_means, opts.dpv_sd) {
            b.set_dpv_data(&name, vec_into_bytes(data), 1, DType::Float32);
        }
        if let Some(agg) = aggregate.as_ref() {
            b.set_dpv_data(
                "n_subjects",
                vec_into_bytes(agg.counts.clone()),
                1,
                DType::UInt32,
            );
        }
        // the aggregated FOD, for inspectability / re-peaking
        if let Some(agg) = aggregate.as_ref() {
            // All four fields together define the basis: `compute_peaks` and every
            // downstream reader resolve it from sh_basis + sh_order +
            // sh_full_basis + sh_legacy, so dropping any one of them silently
            // re-evaluates the template in the wrong basis.
            b.set_sh_info(agg.target.order, agg.target.basis_name.clone());
            b.set_sh_full_basis(agg.target.full_basis);
            b.set_sh_legacy(agg.target.legacy);
            b.set_sh_data(
                "coefficients",
                vec_into_bytes(agg.sh.clone()),
                agg.target.ncoeffs,
                DType::Float32,
            );
        }
        // metadata: subject order + design + provenance
        b.set_extra_value(
            "combine",
            json!({
                "subjects": subjects,
                "design_columns": design_columns,
                "design": subject_design_rows(inputs, &design_columns),
                "method": method_str(opts.method),
                "match_angle_deg": opts.match_angle_deg,
                "match_angle_sweep": opts.match_angle_sweep,
                "matched_scalars": scalar_keys,
                "n_inputs": n_inputs,
                "reference_scans": inputs.iter().filter(|i| i.is_reference).map(|i| i.key.clone()).collect::<Vec<_>>(),
                "n_unmatched_per_subject": n_unmatched,
                "has_tangent_frame": true,
                "min_coverage": agg_opts.min_coverage,
                "divisor": "per-voxel-contributors",
                "lmax_policy": opts.lmax.label(),
                "sh_order": aggregate.as_ref().map(|a| a.target.order),
                "sh_basis": aggregate.as_ref().map(|a| a.target.basis_name.clone()),
                "sh_full_basis": aggregate.as_ref().map(|a| a.target.full_basis),
                "sh_legacy": aggregate.as_ref().map(|a| a.target.legacy),
                "loo": qc.as_ref().map(|q| q.loo_enabled),
                "acc_lmin": opts.acc_lmin,
                "averaged_dpv": dpv_means.iter().map(|(k, _, _)| k.clone()).collect::<Vec<_>>(),
            }),
        );
        let dataset = b.finalize()?;
        dataset.save(p)?;
        written.push(p.display().to_string());
    }

    if let Some(p) = out.out_mask.as_ref() {
        let ones = vec![1u8; n_vox];
        let dims_us = [dims[0] as usize, dims[1] as usize, dims[2] as usize];
        write_voxel_scalar_nifti_u8(p, &ones, &template.ijk, dims_us, affine)?;
        written.push(p.display().to_string());
    }

    if let Some(p) = out.out_cohort.as_ref() {
        let odx_ref = out
            .out_odx
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "group.odx".to_string());
        let mask_ref = out.out_mask.as_ref().map(|p| p.display().to_string()).unwrap_or_default();
        write_cohort(
            p,
            &subjects,
            inputs,
            &design_columns,
            &scalar_keys,
            &odx_ref,
            &mask_ref,
            out.per_subject_odx_dir.as_deref(),
        )?;
        written.push(p.display().to_string());
    }

    if let Some(dir) = out.per_subject_odx_dir.as_ref() {
        std::fs::create_dir_all(dir)?;
        for (s, subj) in subjects.iter().enumerate() {
            let mut b = OdxBuilder::new(affine, dims, template.mask.clone());
            for t in 0..n_vox {
                let st = template.offsets[t] as usize;
                let en = template.offsets[t + 1] as usize;
                b.push_voxel_peaks(&template.dirs[st..en]);
            }
            let col: Vec<f32> = (0..n_fixels).map(|gf| angle[gf * n_inputs + s]).collect();
            b.set_dpf_data("angle_deg", vec_into_bytes(col), 1, DType::Float32);
            // detection + signed tilt + ungated angle, one column for this scan,
            // so the odx→ModelArray path can build each as its own scalar.
            let mcol: Vec<f32> = (0..n_fixels).map(|gf| matched[gf * n_inputs + s]).collect();
            b.set_dpf_data("matched", vec_into_bytes(mcol), 1, DType::Float32);
            let ta: Vec<f32> = (0..n_fixels).map(|gf| tangent_a[gf * n_inputs + s]).collect();
            b.set_dpf_data("tangent_a", vec_into_bytes(ta), 1, DType::Float32);
            let tb: Vec<f32> = (0..n_fixels).map(|gf| tangent_b[gf * n_inputs + s]).collect();
            b.set_dpf_data("tangent_b", vec_into_bytes(tb), 1, DType::Float32);
            let bu: Vec<f32> = (0..n_fixels).map(|gf| best_ungated[gf * n_inputs + s]).collect();
            b.set_dpf_data("best_angle_deg_ungated", vec_into_bytes(bu), 1, DType::Float32);
            for (k, key) in scalar_keys.iter().enumerate() {
                let c: Vec<f32> = (0..n_fixels).map(|gf| scalar_buf[k][gf * n_inputs + s]).collect();
                b.set_dpf_data(key, vec_into_bytes(c), 1, DType::Float32);
            }
            let path = dir.join(format!("{subj}.odx"));
            b.finalize()?.save(&path)?;
            written.push(path.display().to_string());
        }
    }

    if let Some(p) = out.out_table.as_ref() {
        write_tidy_table(
            p, &template, &subjects, inputs, &design_columns, &scalar_keys, &angle, &matched,
            &scalar_buf, n_inputs, affine,
        )?;
        written.push(p.display().to_string());
    }

    if let Some(dir) = out.out_dir.as_ref() {
        std::fs::create_dir_all(dir)?;
        let dims_us = [dims[0] as usize, dims[1] as usize, dims[2] as usize];
        // per-voxel summaries
        let mut n_group = vec![0.0f32; n_vox];
        let mut n_cov = vec![0.0f32; n_vox];
        let mut vox_angle = vec![f32::NAN; n_vox];
        let mut prim_disp = vec![f32::NAN; n_vox];
        for t in 0..n_vox {
            let st = template.offsets[t] as usize;
            let en = template.offsets[t + 1] as usize;
            n_group[t] = (en - st) as f32;
            let flat = template.ijk[t][0] as usize * ny * nz
                + template.ijk[t][1] as usize * nz
                + template.ijk[t][2] as usize;
            n_cov[t] = input_lookup.iter().filter(|lk| lk[flat] != usize::MAX).count() as f32;
            let angles: Vec<f32> = (st..en).filter_map(|gf| {
                let v = mean_angle[gf];
                if v.is_finite() { Some(v) } else { None }
            }).collect();
            if !angles.is_empty() {
                vox_angle[t] = angles.iter().sum::<f32>() / angles.len() as f32;
            }
            if let Some(&gf0) = (st..en).find(|&gf| template.is_primary[gf] == 1).as_ref() {
                prim_disp[t] = dispersion[gf0];
            }
        }
        for (name, data) in [
            ("n_group_fixels", &n_group),
            ("n_subjects_covering", &n_cov),
            ("mean_voxel_angle_deg", &vox_angle),
            ("primary_dispersion", &prim_disp),
        ] {
            let path = dir.join(format!("{name}.nii.gz"));
            write_voxel_scalar_nifti_f32(&path, data, &template.ijk, dims_us, affine)?;
            written.push(path.display().to_string());
        }
        // the FOD reproducibility block, same maps as the group ODX carries
        for (name, data) in voxel_qc_arrays(&aggregate, &qc, &dpv_means, opts.dpv_sd) {
            let path = dir.join(format!("{name}.nii.gz"));
            write_voxel_scalar_nifti_f32(&path, &data, &template.ijk, dims_us, affine)?;
            written.push(path.display().to_string());
        }
        if let Some(agg) = aggregate.as_ref() {
            let counts: Vec<f32> = agg.counts.iter().map(|&c| c as f32).collect();
            let path = dir.join("n_subjects.nii.gz");
            write_voxel_scalar_nifti_f32(&path, &counts, &template.ijk, dims_us, affine)?;
            written.push(path.display().to_string());
        }
    }

    let mean_angle_deg = if total_matched > 0 {
        Some(total_angle / total_matched as f64)
    } else {
        None
    };
    let mean_subjects_per_fixel = if n_fixels > 0 {
        n_sub_matched.iter().map(|&v| v as f64).sum::<f64>() / n_fixels as f64
    } else {
        0.0
    };
    let mean_unmatched_per_scan = if n_inputs > 0 {
        n_unmatched.iter().map(|&v| v as f64).sum::<f64>() / n_inputs as f64
    } else {
        0.0
    };
    let n_reference_scans = inputs.iter().filter(|i| i.is_reference).count();

    // ── Per-subject QC rows + outlier flagging ────────────────────────────
    // The QC pass already counted these; without it, count directly off the
    // lookups so coverage stays meaningful under every method.
    let subject_voxels: Vec<u64> = qc.as_ref().map(|q| q.subject_voxels.clone()).unwrap_or_else(
        || {
            input_lookup
                .iter()
                .map(|lk| {
                    template
                        .ijk
                        .iter()
                        .filter(|v| lk[flat_index(**v, dims)] != usize::MAX)
                        .count() as u64
                })
                .collect()
        },
    );
    let coverage: Vec<f64> = subject_voxels
        .iter()
        .map(|&v| if n_vox > 0 { v as f64 / n_vox as f64 } else { 0.0 })
        .collect();
    let subj_acc: Vec<f32> = qc
        .as_ref()
        .map(|q| q.subject_acc.clone())
        .unwrap_or_else(|| vec![f32::NAN; n_inputs]);
    let subj_acc_loo: Vec<f32> = qc
        .as_ref()
        .map(|q| q.subject_acc_loo.clone())
        .unwrap_or_else(|| vec![f32::NAN; n_inputs]);
    // With no FOD block there is nothing to flag on but coverage, which is
    // itself derived from the FOD pass — so skip flagging entirely.
    let reasons = if qc.is_some() {
        flag_outliers(&subj_acc_loo, &coverage)
    } else {
        vec![Vec::new(); n_inputs]
    };
    let subject_rows: Vec<CombineSubjectRow> = (0..n_inputs)
        .map(|s| CombineSubjectRow {
            key: subjects[s].clone(),
            path: inputs[s].path.display().to_string(),
            n_voxels: subject_voxels[s],
            coverage_frac: coverage[s],
            n_fixels: input_offsets[s].last().copied().unwrap_or(0) as u64,
            mean_acc: subj_acc[s],
            mean_acc_loo: subj_acc_loo[s],
            basis_converted: fod_prepared
                .as_ref()
                .map(|(p, _)| p[s].basis_converted)
                .unwrap_or(false),
            lmax_truncated_from: fod_prepared
                .as_ref()
                .and_then(|(p, _)| p[s].lmax_truncated_from),
            is_outlier: !reasons[s].is_empty(),
            outlier_reasons: reasons[s].clone(),
        })
        .collect();
    let outliers: Vec<String> = subject_rows
        .iter()
        .filter(|r| r.is_outlier)
        .map(|r| r.key.clone())
        .collect();
    for r in subject_rows.iter().filter(|r| r.is_outlier) {
        eprintln!(
            "odx combine: warning: '{}' looks like an outlier: {}",
            r.key,
            r.outlier_reasons.join("; ")
        );
    }
    let finite_mean = |v: &[f32]| -> Option<f64> {
        let f: Vec<f64> = v.iter().filter(|x| x.is_finite()).map(|&x| x as f64).collect();
        (!f.is_empty()).then(|| f.iter().sum::<f64>() / f.len() as f64)
    };

    Ok(CombineReport {
        method: method_str(opts.method).to_string(),
        n_inputs,
        mask_combine: match opts.mask_combine {
            MaskCombine::Union => "union",
            MaskCombine::Intersection => "intersection",
        }
        .to_string(),
        match_angle_deg: opts.match_angle_deg,
        normalize_fod: opts.normalize_fod.label().to_string(),
        min_coverage: agg_opts.min_coverage,
        lmax_policy: opts.lmax.label(),
        sh_order: aggregate.as_ref().map(|a| a.target.order),
        sh_basis: aggregate.as_ref().map(|a| a.target.basis_name.clone()),
        loo: match qc.as_ref() {
            None => "unavailable".to_string(),
            Some(q) if q.loo_enabled => "on".to_string(),
            Some(_) => "off".to_string(),
        },
        acc_lmin: opts.acc_lmin,
        n_voxels_without_orientation: qc
            .as_ref()
            .map(|q| q.n_voxels_without_orientation)
            .unwrap_or(0),
        mean_acc: finite_mean(&subj_acc),
        mean_acc_loo: finite_mean(&subj_acc_loo),
        averaged_dpv: dpv_means.iter().map(|(k, _, _)| k.clone()).collect(),
        subjects: subject_rows,
        outliers,
        dims,
        n_template_voxels: n_vox as u64,
        n_template_fixels: n_fixels as u64,
        mean_subjects_per_fixel,
        mean_angle_deg,
        mean_unmatched_per_scan,
        n_reference_scans,
        matched_scalar_keys: scalar_keys,
        design_columns,
        written_paths: written,
    })
}

// ── Template builders ─────────────────────────────────────────────────────

fn build_template_override(path: &Path, dims: [u64; 3], affine: &[[f64; 4]; 4]) -> Result<Template> {
    let ds = OdxDataset::open(path)?;
    if ds.header().dimensions != dims || !affine_close(&ds.header().voxel_to_rasmm, affine, 1e-4) {
        return Err(OdxError::Argument(format!(
            "template '{}' grid differs from the inputs",
            path.display()
        )));
    }
    let mask = ds.mask().to_vec();
    let ijk = ds.compact_to_ijk();
    let offsets = ds.offsets().to_vec();
    let dirs = ds.directions().to_vec();
    let strength = ds.scalar_dpf_f32("amplitude").or_else(|_| ds.scalar_dpf_f32("afd")).unwrap_or_else(|_| vec![f32::NAN; dirs.len()]);
    let (rank, is_primary) = ranks_from_offsets(&offsets);
    Ok(Template { mask, ijk, offsets, dirs, rank, is_primary, strength, mean_sh: None })
}

#[allow(clippy::too_many_arguments)]
fn build_template_cluster(
    datasets: &[OdxDataset],
    input_lookup: &[Vec<usize>],
    input_offsets: &[&[u32]],
    mask: &[u8],
    dims: [u64; 3],
    cos_thresh: f32,
    min_subjects: usize,
) -> Result<Template> {
    let (ny, nz) = (dims[1] as usize, dims[2] as usize);
    let ijk = mask_compact_ijk(mask, dims);

    // Per-input seed scalar (amplitude → afd → 1.0) for deterministic ordering.
    let seed: Vec<Option<Vec<f32>>> = datasets
        .iter()
        .map(|ds| ds.scalar_dpf_f32("amplitude").or_else(|_| ds.scalar_dpf_f32("afd")).ok())
        .collect();

    let mut offsets = vec![0u32];
    let mut dirs: Vec<[f32; 3]> = Vec::new();
    let mut rank: Vec<u32> = Vec::new();
    let mut is_primary: Vec<u8> = Vec::new();
    let mut strength: Vec<f32> = Vec::new();

    for vox in &ijk {
        let flat = vox[0] as usize * ny * nz + vox[1] as usize * nz + vox[2] as usize;
        // pool directions at this voxel
        let mut pooled: Vec<PooledDir> = Vec::new();
        for (s, ds) in datasets.iter().enumerate() {
            let cs = input_lookup[s][flat];
            if cs == usize::MAX {
                continue;
            }
            let start = input_offsets[s][cs] as usize;
            let vox_dirs = ds.voxel_directions(cs);
            for (li, &d) in vox_dirs.iter().enumerate() {
                let w = seed[s].as_ref().map(|v| v[start + li]).unwrap_or(1.0);
                pooled.push(PooledDir { subject: s, dir: d, weight: w });
            }
        }
        // deterministic seed order: weight desc, tie (subject, original idx)
        pooled.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.subject.cmp(&b.subject))
        });
        let mut clusters = cluster_voxel(&pooled, cos_thresh);
        // keep clusters with enough distinct subjects, order by λ1 desc
        clusters.retain(|c| c.subjects >= min_subjects);
        clusters.sort_by(|a, b| b.lambda1.partial_cmp(&a.lambda1).unwrap_or(std::cmp::Ordering::Equal));
        for (r, c) in clusters.iter().enumerate() {
            dirs.push(c.dir);
            rank.push(r as u32);
            is_primary.push(if r == 0 { 1 } else { 0 });
            strength.push(c.lambda1 as f32);
        }
        offsets.push(dirs.len() as u32);
    }

    Ok(Template { mask: mask.to_vec(), ijk, offsets, dirs, rank, is_primary, strength, mean_sh: None })
}

/// Group fixels from the mean FOD: delegate the aggregation to
/// [`crate::template`], then peak-find the aggregate **in its own SH basis**.
///
/// The basis matters: assuming tournier07 here silently mis-orients every peak
/// of a descoteaux07 cohort, and dropping the `legacy` bit flips the sign of the
/// m<0 coefficients.
fn build_template_mean_fod(
    datasets: &[OdxDataset],
    prepared: &[PreparedInput],
    mask: &[u8],
    dims: [u64; 3],
    target: &ShTarget,
    agg_opts: &AggregateOptions,
    peak_config: &PeakFinderConfig,
) -> Result<Template> {
    let agg = aggregate_fod(datasets, prepared, dims, mask, target, agg_opts)?;
    let (offsets, dirs, amps) = peaks_from_aggregate(&agg, peak_config, None)?;
    let ijk = mask_compact_ijk(mask, dims);
    let (rank, is_primary) = ranks_from_offsets(&offsets);
    Ok(Template {
        mask: mask.to_vec(),
        ijk,
        offsets,
        dirs,
        rank,
        is_primary,
        strength: amps,
        mean_sh: Some(agg),
    })
}

// ── Clustering ─────────────────────────────────────────────────────────────

struct PooledDir {
    subject: usize,
    dir: [f32; 3],
    weight: f32,
}

struct ClusterOut {
    dir: [f32; 3],
    lambda1: f64,
    subjects: usize,
}

/// Greedy dyadic clustering + Lloyd refinement (fixed cluster count) on the
/// pooled directions of one voxel. `pooled` must already be in seed order.
fn cluster_voxel(pooled: &[PooledDir], cos_thresh: f32) -> Vec<ClusterOut> {
    if pooled.is_empty() {
        return Vec::new();
    }
    // 1) greedy seeding: assignment + cluster count. `assign` starts at a
    // sentinel so `mean_dir` only averages already-seeded points (not the
    // not-yet-processed ones, which would all alias to cluster 0).
    let mut assign = vec![usize::MAX; pooled.len()];
    let mut means: Vec<[f32; 3]> = Vec::new();
    for (i, p) in pooled.iter().enumerate() {
        let mut best = usize::MAX;
        let mut best_dp = -1.0f32;
        for (ci, m) in means.iter().enumerate() {
            let dp = abs_dot(p.dir, *m);
            if dp > best_dp {
                best_dp = dp;
                best = ci;
            }
        }
        if best != usize::MAX && best_dp >= cos_thresh {
            assign[i] = best;
        } else {
            assign[i] = means.len();
            means.push(p.dir);
        }
        means[assign[i]] = mean_dir(pooled, &assign, assign[i]);
    }
    // 2) Lloyd refinement (fixed cluster count)
    for _ in 0..5 {
        let mut changed = false;
        for (i, p) in pooled.iter().enumerate() {
            let mut best = assign[i];
            let mut best_dp = abs_dot(p.dir, means[best]);
            for (ci, m) in means.iter().enumerate() {
                let dp = abs_dot(p.dir, *m);
                if dp > best_dp {
                    best_dp = dp;
                    best = ci;
                }
            }
            if best != assign[i] {
                assign[i] = best;
                changed = true;
            }
        }
        for (ci, m) in means.iter_mut().enumerate() {
            *m = mean_dir(pooled, &assign, ci);
        }
        if !changed {
            break;
        }
    }
    // 3) finalize per cluster
    let k = means.len();
    let mut out = Vec::with_capacity(k);
    for ci in 0..k {
        let mut t = [0.0f64; 6];
        let mut subs = BTreeSet::new();
        let mut any = false;
        for (i, p) in pooled.iter().enumerate() {
            if assign[i] == ci {
                accumulate_tensor(&mut t, p.dir);
                subs.insert(p.subject);
                any = true;
            }
        }
        if !any {
            continue;
        }
        let (dir, evals) = principal_axis(&t);
        out.push(ClusterOut { dir, lambda1: evals[0], subjects: subs.len() });
    }
    out
}

/// Principal eigenvector of Σ d·dᵀ over the members of cluster `ci`.
fn mean_dir(pooled: &[PooledDir], assign: &[usize], ci: usize) -> [f32; 3] {
    let mut t = [0.0f64; 6];
    for (i, p) in pooled.iter().enumerate() {
        if assign[i] == ci {
            accumulate_tensor(&mut t, p.dir);
        }
    }
    principal_axis(&t).0
}

// ── Dyadic-tensor helpers ──────────────────────────────────────────────────

/// Add `d·dᵀ` into a packed symmetric 3×3 tensor `[xx,yy,zz,xy,xz,yz]`.
fn accumulate_tensor(t: &mut [f64; 6], d: [f32; 3]) {
    let (x, y, z) = (d[0] as f64, d[1] as f64, d[2] as f64);
    t[0] += x * x;
    t[1] += y * y;
    t[2] += z * z;
    t[3] += x * y;
    t[4] += x * z;
    t[5] += y * z;
}

fn matrix_from_packed(t: &[f64; 6]) -> Matrix3<f64> {
    Matrix3::new(t[0], t[3], t[4], t[3], t[1], t[5], t[4], t[5], t[2])
}

/// Principal (largest-eigenvalue) unit axis of a packed symmetric tensor, with
/// eigenvalues sorted descending. The axis sign is canonicalized so its
/// largest-magnitude component is positive (cosmetic, deterministic).
fn principal_axis(t: &[f64; 6]) -> ([f32; 3], [f64; 3]) {
    let m = matrix_from_packed(t);
    let eig = SymmetricEigen::new(m);
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));
    let v = eig.eigenvectors.column(order[0]);
    let mut dir = [v[0] as f32, v[1] as f32, v[2] as f32];
    let n = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    if n > 0.0 {
        dir = [dir[0] / n, dir[1] / n, dir[2] / n];
    }
    let imax = (0..3).max_by(|&a, &b| dir[a].abs().total_cmp(&dir[b].abs())).unwrap();
    if dir[imax] < 0.0 {
        dir = [-dir[0], -dir[1], -dir[2]];
    }
    let evals = [
        eig.eigenvalues[order[0]],
        eig.eigenvalues[order[1]],
        eig.eigenvalues[order[2]],
    ];
    (dir, evals)
}

/// Dyadic dispersion `1 − λ1` of the *normalized* mean tensor (trace 1): 0 when
/// all matched directions are collinear, up to 2/3 for fully isotropic spread.
fn dyadic_dispersion(t: &[f64; 6], n: f64) -> f32 {
    if n <= 0.0 {
        return f32::NAN;
    }
    let norm = [t[0] / n, t[1] / n, t[2] / n, t[3] / n, t[4] / n, t[5] / n];
    let (_, evals) = principal_axis(&norm);
    (1.0 - evals[0]).max(0.0) as f32
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Canonicalize a fixel axis (sign-ambiguous) so its largest-magnitude component
/// is positive — a deterministic, reproducible representative for the reference
/// direction so the tangent frame and signed residual are well-defined.
fn sign_canon(v: [f32; 3]) -> [f32; 3] {
    let i = (0..3).max_by(|&a, &b| v[a].abs().total_cmp(&v[b].abs())).unwrap_or(0);
    if v[i] < 0.0 {
        [-v[0], -v[1], -v[2]]
    } else {
        v
    }
}

/// Deterministic orthonormal tangent basis `(e1, e2)` of the plane perpendicular
/// to a (unit, sign-canonical) reference axis `r`. Built only from `r` plus a
/// fixed gravity rule (z, falling back to x near the poles), so it is wholly
/// method-independent — no contestant scan influences the frame. The signed
/// residual of a scan direction `d` is `(d'·e1, d'·e2)` with `d'` flipped into
/// `r`'s hemisphere; a consistent per-method offset shows up as a nonzero mean
/// residual (systematic tilt), distinct from random scatter.
fn tangent_basis(r: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let g = if r[2].abs() <= 0.9 {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let gr = dot3(g, r);
    let mut e1 = [g[0] - gr * r[0], g[1] - gr * r[1], g[2] - gr * r[2]];
    let n = (e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]).sqrt();
    if n > 0.0 {
        e1 = [e1[0] / n, e1[1] / n, e1[2] / n];
    }
    // e2 = r × e1 (right-handed, unit since r ⟂ e1 are unit)
    let e2 = [
        r[1] * e1[2] - r[2] * e1[1],
        r[2] * e1[0] - r[0] * e1[2],
        r[0] * e1[1] - r[1] * e1[0],
    ];
    (e1, e2)
}

/// Signed tangent-plane residual `(a, b)` of scan direction `d` relative to
/// reference `r` with frame `(e1, e2)`. `d` is first flipped into `r`'s
/// hemisphere (fixels are undirected). `sqrt(a²+b²) = sin(angle)`.
fn tangent_residual(d: [f32; 3], r: [f32; 3], e1: [f32; 3], e2: [f32; 3]) -> (f32, f32) {
    let d = if dot3(d, r) < 0.0 {
        [-d[0], -d[1], -d[2]]
    } else {
        d
    };
    (dot3(d, e1), dot3(d, e2))
}

// ── Misc helpers ───────────────────────────────────────────────────────────

fn ranks_from_offsets(offsets: &[u32]) -> (Vec<u32>, Vec<u8>) {
    let n = *offsets.last().unwrap_or(&0) as usize;
    let mut rank = vec![0u32; n];
    let mut is_primary = vec![0u8; n];
    for w in offsets.windows(2) {
        let (s, e) = (w[0] as usize, w[1] as usize);
        for (r, gf) in (s..e).enumerate() {
            rank[gf] = r as u32;
            is_primary[gf] = if r == 0 { 1 } else { 0 };
        }
    }
    (rank, is_primary)
}

/// The scalar per-voxel arrays of the FOD reproducibility block, in a stable
/// order, as `(name, values)` in template compact-voxel order.
///
/// `anisotropic_power` is recomputed from the aggregate rather than averaged
/// from the inputs, because the anisotropic power of a mean is not the mean of
/// the anisotropic powers.
fn voxel_qc_arrays(
    aggregate: &Option<AggregatedFod>,
    qc: &Option<crate::template::FodQc>,
    dpv_means: &[(String, Vec<f32>, Vec<f32>)],
    dpv_sd: bool,
) -> Vec<(String, Vec<f32>)> {
    let mut out: Vec<(String, Vec<f32>)> = Vec::new();
    if let Some(q) = qc {
        out.push(("coverage_frac".into(), q.coverage_frac.clone()));
        out.push(("l0_mean".into(), q.l0_mean.clone()));
        out.push(("l0_sd".into(), q.l0_sd.clone()));
        out.push(("l0_cv".into(), q.l0_cv.clone()));
        out.push(("acc_mean".into(), q.acc_mean.clone()));
        out.push(("acc_sd".into(), q.acc_sd.clone()));
        out.push(("acc_min".into(), q.acc_min.clone()));
        if q.loo_enabled {
            out.push(("acc_loo_mean".into(), q.acc_loo_mean.clone()));
            out.push(("acc_loo_min".into(), q.acc_loo_min.clone()));
        }
    }
    if let Some(agg) = aggregate {
        out.push((
            "anisotropic_power".into(),
            aggregate_anisotropic_power(agg),
        ));
    }
    for (name, mean, sd) in dpv_means {
        out.push((name.clone(), mean.clone()));
        if dpv_sd {
            out.push((format!("{name}_sd"), sd.clone()));
        }
    }
    out
}

fn method_str(m: TemplateMethod) -> &'static str {
    match m {
        TemplateMethod::Cluster => "cluster",
        TemplateMethod::MeanFod => "mean-fod",
    }
}

fn collect_design_columns(inputs: &[CombineInput]) -> Vec<String> {
    let mut cols: Vec<String> = Vec::new();
    for inp in inputs {
        for (k, _) in &inp.categorical {
            if !cols.iter().any(|c| c == k) {
                cols.push(k.clone());
            }
        }
    }
    cols
}

fn subject_design_rows(inputs: &[CombineInput], cols: &[String]) -> Vec<Vec<String>> {
    inputs
        .iter()
        .map(|inp| {
            let map: BTreeMap<&str, &str> =
                inp.categorical.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
            cols.iter().map(|c| map.get(c.as_str()).copied().unwrap_or("").to_string()).collect()
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn write_cohort(
    path: &Path,
    subjects: &[String],
    inputs: &[CombineInput],
    design_columns: &[String],
    scalar_keys: &[String],
    odx_ref: &str,
    mask_ref: &str,
    per_subject_dir: Option<&Path>,
) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut header = vec![
        "scalar_name".to_string(),
        "source_file".to_string(),
        "source_mask_file".to_string(),
        "subject".to_string(),
        "is_reference".to_string(),
    ];
    header.extend(design_columns.iter().cloned());
    writeln!(f, "{}", header.iter().map(|s| csv_field(s)).collect::<Vec<_>>().join(","))?;

    // detection / orientation / signed-tilt scalars + each matched scalar.
    let mut scalars: Vec<String> = [
        "angle_deg",
        "matched",
        "tangent_a",
        "tangent_b",
        "best_angle_deg_ungated",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    scalars.extend(scalar_keys.iter().cloned());

    for scalar in &scalars {
        for (s, subj) in subjects.iter().enumerate() {
            let source = match per_subject_dir {
                Some(dir) => dir.join(format!("{subj}.odx")).display().to_string(),
                None => odx_ref.to_string(),
            };
            let mut row = vec![
                scalar.clone(),
                source,
                mask_ref.to_string(),
                subj.clone(),
                if inputs[s].is_reference { "1" } else { "0" }.to_string(),
            ];
            let map: BTreeMap<&str, &str> =
                inputs[s].categorical.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
            for c in design_columns {
                row.push(map.get(c.as_str()).copied().unwrap_or("").to_string());
            }
            writeln!(f, "{}", row.iter().map(|s| csv_field(s)).collect::<Vec<_>>().join(","))?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_tidy_table(
    path: &Path,
    template: &Template,
    subjects: &[String],
    inputs: &[CombineInput],
    design_columns: &[String],
    scalar_keys: &[String],
    angle: &[f32],
    matched: &[f32],
    scalar_buf: &[Vec<f32>],
    n_inputs: usize,
    affine: [[f64; 4]; 4],
) -> Result<()> {
    let tab = path.extension().and_then(|e| e.to_str()) == Some("tsv");
    let sep = if tab { '\t' } else { ',' };
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);

    let mut header = vec![
        "fixel_id", "i", "j", "k", "x_ras", "y_ras", "z_ras", "within_voxel_rank", "is_primary",
        "group_dir_x", "group_dir_y", "group_dir_z", "subject_key", "matched", "angle_deg",
    ]
    .into_iter()
    .map(|s| s.to_string())
    .collect::<Vec<_>>();
    for k in scalar_keys {
        header.push(format!("{k}_value"));
    }
    header.extend(design_columns.iter().cloned());
    writeln!(f, "{}", header.iter().map(|s| field(s, sep)).collect::<Vec<_>>().join(&sep.to_string()))?;

    for t in 0..template.ijk.len() {
        let ijk = template.ijk[t];
        let ras = apply_affine(&affine, ijk);
        let st = template.offsets[t] as usize;
        let en = template.offsets[t + 1] as usize;
        for gf in st..en {
            let d = template.dirs[gf];
            for (s, subj) in subjects.iter().enumerate() {
                let mtc = matched[gf * n_inputs + s];
                let a = angle[gf * n_inputs + s];
                let mut row = vec![
                    gf.to_string(),
                    ijk[0].to_string(),
                    ijk[1].to_string(),
                    ijk[2].to_string(),
                    fmt_f(ras[0]),
                    fmt_f(ras[1]),
                    fmt_f(ras[2]),
                    template.rank[gf].to_string(),
                    template.is_primary[gf].to_string(),
                    fmt_f(d[0]),
                    fmt_f(d[1]),
                    fmt_f(d[2]),
                    subj.clone(),
                    if mtc.is_finite() { (mtc as i32).to_string() } else { String::new() },
                    if a.is_finite() { fmt_f(a) } else { String::new() },
                ];
                for buf in scalar_buf {
                    let v = buf[gf * n_inputs + s];
                    row.push(if v.is_finite() { fmt_f(v) } else { String::new() });
                }
                let map: BTreeMap<&str, &str> =
                    inputs[s].categorical.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
                for c in design_columns {
                    row.push(map.get(c.as_str()).copied().unwrap_or("").to_string());
                }
                writeln!(f, "{}", row.iter().map(|s| field(s, sep)).collect::<Vec<_>>().join(&sep.to_string()))?;
            }
        }
    }
    Ok(())
}

fn apply_affine(a: &[[f64; 4]; 4], ijk: [u32; 3]) -> [f32; 3] {
    let v = [ijk[0] as f64, ijk[1] as f64, ijk[2] as f64];
    [
        (a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2] + a[0][3]) as f32,
        (a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2] + a[1][3]) as f32,
        (a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2] + a[2][3]) as f32,
    ]
}

fn fmt_f(v: f32) -> String {
    format!("{v:.6}")
}

fn csv_field(s: &str) -> String {
    field(s, ',')
}

fn field(s: &str, sep: char) -> String {
    if s.contains(sep) || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        combine_odx, dot3, sign_canon, tangent_basis, tangent_residual, CombineInput,
        CombineOptions, CombineOutputs,
    };
    use crate::dtype::DType;
    use crate::mmap_backing::vec_into_bytes;
    use crate::odx_file::OdxDataset;
    use crate::stream::OdxBuilder;
    use std::path::{Path, PathBuf};
    use tempfile::TempDir;

    fn identity_affine() -> [[f64; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    fn unit(v: [f32; 3]) -> [f32; 3] {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / n, v[1] / n, v[2] / n]
    }

    fn adot(a: [f32; 3], b: [f32; 3]) -> f32 {
        (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).abs()
    }

    /// Write a tiny single-grid ODX directory with the given peaks per in-mask
    /// voxel and an `amplitude` DPF; returns its path.
    fn build_input(
        dir: &Path,
        name: &str,
        dims: [u64; 3],
        mask: Vec<u8>,
        peaks: Vec<Vec<[f32; 3]>>,
    ) -> PathBuf {
        build_input_affine(dir, name, identity_affine(), dims, mask, peaks)
    }

    fn build_input_affine(
        dir: &Path,
        name: &str,
        affine: [[f64; 4]; 4],
        dims: [u64; 3],
        mask: Vec<u8>,
        peaks: Vec<Vec<[f32; 3]>>,
    ) -> PathBuf {
        let mut b = OdxBuilder::new(affine, dims, mask);
        let mut amps: Vec<f32> = Vec::new();
        for vox in &peaks {
            b.push_voxel_peaks(vox);
            amps.resize(amps.len() + vox.len(), 1.0);
        }
        if !amps.is_empty() {
            b.set_dpf_data("amplitude", vec_into_bytes(amps), 1, DType::Float32);
        }
        let path = dir.join(format!("{name}.odx"));
        b.finalize().unwrap().save_directory(&path).unwrap();
        path
    }

    fn ci(p: &Path, key: &str) -> CombineInput {
        CombineInput {
            path: p.to_path_buf(),
            key: key.to_string(),
            categorical: vec![],
            is_reference: false,
            method: None,
        }
    }

    fn out_only_odx(p: PathBuf) -> CombineOutputs {
        CombineOutputs {
            out_odx: Some(p),
            ..Default::default()
        }
    }

    #[test]
    fn cluster_recovers_known_direction() {
        let tmp = TempDir::new().unwrap();
        let a = build_input(tmp.path(), "a", [1, 1, 1], vec![1], vec![vec![[0.0, 0.0, 1.0]]]);
        let b = build_input(tmp.path(), "b", [1, 1, 1], vec![1], vec![vec![unit([0.05, 0.0, 0.998])]]);
        let c = build_input(tmp.path(), "c", [1, 1, 1], vec![1], vec![vec![unit([-0.05, 0.05, 0.997])]]);
        let inputs = vec![ci(&a, "a"), ci(&b, "b"), ci(&c, "c")];
        let opts = CombineOptions { min_subjects_per_group_fixel: 2, ..Default::default() };
        let out = tmp.path().join("out.odx");
        let rep = combine_odx(&inputs, &opts, &out_only_odx(out.clone())).unwrap();
        assert_eq!(rep.n_template_fixels, 1);

        let ds = OdxDataset::open(&out).unwrap();
        assert_eq!(ds.nb_peaks(), 1);
        assert!(adot(ds.directions()[0], [0.0, 0.0, 1.0]) > 0.99, "{:?}", ds.directions()[0]);
        assert_eq!(ds.scalar_dpf_f32("n_subjects_matched").unwrap()[0], 3.0);

        let angle = ds.get_dpf("angle_deg").unwrap();
        assert_eq!(angle.ncols(), 3, "angle_deg must be (n_fixels × n_subjects)");
        assert!(angle.to_f32_vec().unwrap().iter().all(|&a| a < 8.0));

        // metadata carries the subject order matching the DPF columns
        let combine = ds.header().extra.get("combine").unwrap();
        let subs: Vec<&str> = combine["subjects"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        assert_eq!(subs, ["a", "b", "c"]);
    }

    /// An input stored on the same physical lattice but with a flipped voxel
    /// axis (LAS vs RAS+) must be reindexed onto the reference ordering, not
    /// rejected. Fixel directions live in world space, so only the voxel
    /// indexing changes — the recovered template must match the unflipped case.
    #[test]
    fn same_lattice_flipped_input_is_reindexed_not_rejected() {
        let tmp = TempDir::new().unwrap();
        let dims = [1u64, 1, 2];
        // Reference: +z voxel spacing 2 mm, k = 0 at world z = 0.
        // b: same lattice, z axis reversed, so b's k=0 is the reference's k=1.
        let flipped = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -2.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let reference = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        // Physical truth: world z=0 has a +z fixel, world z=2 has a +x fixel.
        let a = build_input_affine(
            tmp.path(), "a", reference, dims, vec![1, 1],
            vec![vec![[0.0, 0.0, 1.0]], vec![[1.0, 0.0, 0.0]]],
        );
        // b lists the same two physical voxels in reversed order.
        let b = build_input_affine(
            tmp.path(), "b", flipped, dims, vec![1, 1],
            vec![vec![[1.0, 0.0, 0.0]], vec![[0.0, 0.0, 1.0]]],
        );
        let inputs = vec![ci(&a, "a"), ci(&b, "b")];
        let out = tmp.path().join("out.odx");
        combine_odx(&inputs, &CombineOptions::default(), &out_only_odx(out.clone())).unwrap();

        let ds = OdxDataset::open(&out).unwrap();
        // Both subjects agree at every voxel once reindexed → all angles ~0.
        let angle = ds.get_dpf("angle_deg").unwrap().to_f32_vec().unwrap();
        assert!(
            angle.iter().all(|&x| x < 1.0),
            "flipped input must align to the reference voxel order, got angles {angle:?}"
        );
        assert_eq!(ds.nb_peaks(), 2, "one group fixel per voxel");
    }

    #[test]
    fn cluster_separates_crossing_fibers() {
        let tmp = TempDir::new().unwrap();
        let mk = |n: &str| {
            build_input(tmp.path(), n, [1, 1, 1], vec![1], vec![vec![[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]])
        };
        let (a, b, c) = (mk("a"), mk("b"), mk("c"));
        let inputs = vec![ci(&a, "a"), ci(&b, "b"), ci(&c, "c")];
        let out = tmp.path().join("out.odx");
        let rep = combine_odx(&inputs, &CombineOptions::default(), &out_only_odx(out.clone())).unwrap();
        assert_eq!(rep.n_template_fixels, 2);

        let ds = OdxDataset::open(&out).unwrap();
        let dirs = ds.directions();
        let has_z = dirs.iter().any(|d| adot(*d, [0.0, 0.0, 1.0]) > 0.99);
        let has_x = dirs.iter().any(|d| adot(*d, [1.0, 0.0, 0.0]) > 0.99);
        assert!(has_z && has_x, "both crossing axes recovered: {dirs:?}");
        // exactly one primary fixel in the single voxel
        let prim: u32 = ds.scalar_dpf_f32("is_primary").unwrap().iter().map(|&v| v as u32).sum();
        assert_eq!(prim, 1);
    }

    #[test]
    fn template_override_self_matches_at_zero() {
        let tmp = TempDir::new().unwrap();
        let a = build_input(tmp.path(), "a", [1, 1, 1], vec![1], vec![vec![[0.0, 0.0, 1.0]]]);
        let b = build_input(tmp.path(), "b", [1, 1, 1], vec![1], vec![vec![unit([0.1736, 0.0, 0.9848])]]); // ~10°
        let inputs = vec![ci(&a, "a"), ci(&b, "b")];
        let opts = CombineOptions { template_override: Some(a.clone()), ..Default::default() };
        let out = tmp.path().join("out.odx");
        combine_odx(&inputs, &opts, &out_only_odx(out.clone())).unwrap();

        let ds = OdxDataset::open(&out).unwrap();
        let angle = ds.get_dpf("angle_deg").unwrap().to_f32_vec().unwrap();
        assert_eq!(angle.len(), 2); // 1 fixel × 2 subjects
        assert!(angle[0] < 0.5, "template subject matches itself at ~0°, got {}", angle[0]);
        assert!((angle[1] - 10.0).abs() < 1.0, "second subject ~10°, got {}", angle[1]);
    }

    #[test]
    fn aligns_voxels_by_ijk_not_compact_index() {
        // a covers both voxels (+z @ v0, +x @ v1); b covers ONLY v1 (+x).
        // b's compact row 0 is v1 — a naive compact-index match would put it at
        // v0. Correct ijk alignment gives v1 two subjects, v0 one.
        let tmp = TempDir::new().unwrap();
        let a = build_input(tmp.path(), "a", [2, 1, 1], vec![1, 1], vec![vec![[0.0, 0.0, 1.0]], vec![[1.0, 0.0, 0.0]]]);
        let b = build_input(tmp.path(), "b", [2, 1, 1], vec![0, 1], vec![vec![[1.0, 0.0, 0.0]]]);
        let inputs = vec![ci(&a, "a"), ci(&b, "b")];
        let opts = CombineOptions { min_subjects_per_group_fixel: 1, ..Default::default() };
        let out = tmp.path().join("out.odx");
        combine_odx(&inputs, &opts, &out_only_odx(out.clone())).unwrap();

        let ds = OdxDataset::open(&out).unwrap();
        assert_eq!(ds.nb_peaks(), 2);
        let nsub = ds.scalar_dpf_f32("n_subjects_matched").unwrap();
        // fixel 0 = voxel0 (+z, only a); fixel 1 = voxel1 (+x, a and b)
        assert_eq!(nsub[0], 1.0);
        assert_eq!(nsub[1], 2.0);
    }

    #[test]
    fn min_subjects_drops_singletons() {
        let tmp = TempDir::new().unwrap();
        let a = build_input(tmp.path(), "a", [1, 1, 1], vec![1], vec![vec![[0.0, 0.0, 1.0]]]);
        let b = build_input(tmp.path(), "b", [1, 1, 1], vec![1], vec![vec![[1.0, 0.0, 0.0]]]);
        let inputs = vec![ci(&a, "a"), ci(&b, "b")];
        let out = tmp.path().join("out.odx");

        let drop = CombineOptions { min_subjects_per_group_fixel: 2, ..Default::default() };
        let rep = combine_odx(&inputs, &drop, &out_only_odx(out.clone())).unwrap();
        assert_eq!(rep.n_template_fixels, 0, "two orthogonal singletons should be dropped");

        let keep = CombineOptions { min_subjects_per_group_fixel: 1, ..Default::default() };
        let out2 = tmp.path().join("out2.odx");
        let rep2 = combine_odx(&inputs, &keep, &out_only_odx(out2)).unwrap();
        assert_eq!(rep2.n_template_fixels, 2);
    }

    #[test]
    fn rejects_grid_mismatch() {
        let tmp = TempDir::new().unwrap();
        let a = build_input(tmp.path(), "a", [1, 1, 1], vec![1], vec![vec![[0.0, 0.0, 1.0]]]);
        let b = build_input(tmp.path(), "b", [2, 1, 1], vec![1, 0], vec![vec![[0.0, 0.0, 1.0]]]);
        let inputs = vec![ci(&a, "a"), ci(&b, "b")];
        let out = tmp.path().join("out.odx");
        assert!(combine_odx(&inputs, &CombineOptions::default(), &out_only_odx(out)).is_err());
    }

    #[test]
    fn antipodal_directions_match_at_zero() {
        // subject fixel at -z vs a +z template fixel → angle 0 (sign-agnostic).
        let tmp = TempDir::new().unwrap();
        let a = build_input(tmp.path(), "a", [1, 1, 1], vec![1], vec![vec![[0.0, 0.0, 1.0]]]);
        let b = build_input(tmp.path(), "b", [1, 1, 1], vec![1], vec![vec![[0.0, 0.0, -1.0]]]);
        let inputs = vec![ci(&a, "a"), ci(&b, "b")];
        let opts = CombineOptions { template_override: Some(a.clone()), ..Default::default() };
        let out = tmp.path().join("out.odx");
        combine_odx(&inputs, &opts, &out_only_odx(out.clone())).unwrap();
        let ds = OdxDataset::open(&out).unwrap();
        let angle = ds.get_dpf("angle_deg").unwrap().to_f32_vec().unwrap();
        assert!(angle[1] < 0.5, "antipodal subject should match at ~0°, got {}", angle[1]);
    }

    #[test]
    fn tangent_frame_is_orthonormal_and_signed() {
        let r = sign_canon(unit([0.2, 0.1, 0.97]));
        let (e1, e2) = tangent_basis(r);
        // orthonormal basis of the plane perpendicular to r
        assert!(dot3(e1, r).abs() < 1e-5);
        assert!(dot3(e2, r).abs() < 1e-5);
        assert!(dot3(e1, e2).abs() < 1e-5);
        assert!((dot3(e1, e1) - 1.0).abs() < 1e-5);
        assert!((dot3(e2, e2) - 1.0).abs() < 1e-5);
        // tilt r by 10° toward +e1 → residual ≈ (sin10, 0)
        let ang = 10f32.to_radians();
        let d = [
            r[0] * ang.cos() + e1[0] * ang.sin(),
            r[1] * ang.cos() + e1[1] * ang.sin(),
            r[2] * ang.cos() + e1[2] * ang.sin(),
        ];
        let (a, b) = tangent_residual(d, r, e1, e2);
        assert!((a - ang.sin()).abs() < 1e-4, "a={a} vs {}", ang.sin());
        assert!(b.abs() < 1e-4, "b={b}");
        // antipodal d gives the same residual (axis is sign-agnostic)
        let (a2, b2) = tangent_residual([-d[0], -d[1], -d[2]], r, e1, e2);
        assert!((a2 - a).abs() < 1e-5 && (b2 - b).abs() < 1e-5);
    }

    #[test]
    fn template_method_metrics_detection_tilt_sweep() {
        let tmp = TempDir::new().unwrap();
        // reference scaffold: one +z fixel
        let tmpl = build_input(tmp.path(), "ref", [1, 1, 1], vec![1], vec![vec![[0.0, 0.0, 1.0]]]);
        let a = build_input(tmp.path(), "a", [1, 1, 1], vec![1], vec![vec![[0.0, 0.0, 1.0]]]);
        let b = build_input(tmp.path(), "b", [1, 1, 1], vec![1], vec![vec![unit([0.1736, 0.0, 0.9848])]]); // ~10° toward +x
        let c = build_input(tmp.path(), "c", [1, 1, 1], vec![1], vec![vec![unit([0.643, 0.0, 0.766])]]); // ~40° toward +x
        let inputs = vec![ci(&a, "a"), ci(&b, "b"), ci(&c, "c")];
        let opts = CombineOptions {
            template_override: Some(tmpl),
            match_angle_deg: 30.0,
            match_angle_sweep: vec![20.0, 45.0],
            ..Default::default()
        };
        let out = tmp.path().join("g.odx");
        combine_odx(&inputs, &opts, &out_only_odx(out.clone())).unwrap();
        let ds = OdxDataset::open(&out).unwrap();
        assert_eq!(ds.nb_peaks(), 1);

        // detection: a,b within the 30° cone; c (40°) is not
        let matched = ds.get_dpf("matched").unwrap().to_f32_vec().unwrap();
        assert_eq!(matched, vec![1.0, 1.0, 0.0]);
        // ungated angle is recorded even when unmatched
        let bu = ds.get_dpf("best_angle_deg_ungated").unwrap().to_f32_vec().unwrap();
        assert!((bu[2] - 40.0).abs() < 1.0, "c ungated angle {}", bu[2]);
        // signed tilt: a ≈ 0; b consistently positive on e1 (=+x for a +z fixel); c NaN
        let ta = ds.get_dpf("tangent_a").unwrap().to_f32_vec().unwrap();
        assert!(ta[0].abs() < 1e-3, "a tangent_a {}", ta[0]);
        assert!(ta[1] > 0.10, "b tangent_a {} (expect ~sin10=0.17)", ta[1]);
        assert!(ta[2].is_nan());
        // threshold sweep: c at 40° is detected at 45° but not at 20°
        let m20 = ds.get_dpf("matched_at_20").unwrap().to_f32_vec().unwrap();
        let m45 = ds.get_dpf("matched_at_45").unwrap().to_f32_vec().unwrap();
        assert_eq!(m20[2], 0.0);
        assert_eq!(m45[2], 1.0);
    }

    #[test]
    fn scaffold_support_counts_reference_only() {
        let tmp = TempDir::new().unwrap();
        let tmpl = build_input(tmp.path(), "ref", [1, 1, 1], vec![1], vec![vec![[0.0, 0.0, 1.0]]]);
        let mk = |n: &str| build_input(tmp.path(), n, [1, 1, 1], vec![1], vec![vec![[0.0, 0.0, 1.0]]]);
        let (a, b, c) = (mk("a"), mk("b"), mk("c"));
        let mut inputs = vec![ci(&a, "a"), ci(&b, "b"), ci(&c, "c")];
        inputs[0].is_reference = true; // only 'a' defines the reference cohort
        let opts = CombineOptions { template_override: Some(tmpl), ..Default::default() };
        let out = tmp.path().join("g.odx");
        combine_odx(&inputs, &opts, &out_only_odx(out.clone())).unwrap();
        let ds = OdxDataset::open(&out).unwrap();
        assert_eq!(ds.scalar_dpf_f32("scaffold_support").unwrap()[0], 1.0);
        assert_eq!(ds.scalar_dpf_f32("n_subjects_matched").unwrap()[0], 3.0);
    }
}
