//! Group FOD aggregation: build an average ODF template from N pre-aligned ODX
//! files, plus the reproducibility maps that say how much to trust it.
//!
//! This is the N-file analogue of [`crate::compare`], and the engine behind
//! `odx combine --method mean-fod`. It follows MRtrix3's `population_template`
//! aggregation step (a per-coefficient mean of the SH image, which is exactly
//! the FOD of the mean because the FOD is linear in its coefficients) and DSI
//! Studio's `odf_average` (per-voxel contributor divisor, peaks re-derived from
//! the average). Registration is out of scope: inputs must already share one
//! space, which is what `odx transform` produces.
//!
//! Two design points worth knowing before reading the code:
//!
//! * **The divisor is the per-voxel contributor count, never N.** Dividing by N
//!   would attenuate the FOD wherever some subjects fall outside their mask and
//!   manufacture a rim of low apparent fibre density at the mask boundary that
//!   downstream group tests read as a real effect. Both reference
//!   implementations agree here (`mrmath mean` skips non-finite values; DSI
//!   Studio divides by its per-voxel `odf_count`).
//! * **lmax truncation is a coefficient prefix.** Both SH bases in this crate
//!   are band-ordered ascending, so dropping to a lower lmax is a prefix slice.
//!   Truncating to the cohort minimum is preferred over zero-padding to the
//!   maximum, because padding would make the template's effective sharpness
//!   depend on which subjects happened to cover which voxel.

use std::path::Path;

use crate::descoteaux_sh;
use crate::error::{OdxError, Result};
use crate::fixel_match::{align_to_reference_grid, mask_compact_ijk};
use crate::header::Header;
use crate::interop::{convert_sh_basis, ShBasisTarget};
use crate::mrtrix_sh;
use crate::odx_file::OdxDataset;
use crate::peak_finder::{peaks_from_sh_rows_with_basis, PeakFinderConfig, SpherePeakFinder};
use crate::sh_basis_evaluator::{basis_kind_from_dipy_name, ShBasisKind};

/// How the cohort's SH order is reconciled when inputs disagree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LmaxPolicy {
    /// Truncate every input to the smallest lmax present (default).
    Min,
    /// Zero-pad every input to the largest lmax present.
    Max,
    /// Force an explicit even lmax.
    Fixed(u64),
}

impl LmaxPolicy {
    /// Parse the CLI spelling: `min`, `max`, or an even integer.
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "min" => Ok(LmaxPolicy::Min),
            "max" => Ok(LmaxPolicy::Max),
            other => match other.parse::<u64>() {
                Ok(n) if n % 2 == 0 => Ok(LmaxPolicy::Fixed(n)),
                _ => Err(OdxError::Argument(format!(
                    "--lmax must be 'min', 'max', or an even integer, got '{s}'"
                ))),
            },
        }
    }

    pub fn label(&self) -> String {
        match self {
            LmaxPolicy::Min => "min".into(),
            LmaxPolicy::Max => "max".into(),
            LmaxPolicy::Fixed(n) => n.to_string(),
        }
    }
}

/// Whether to compute the leave-one-out angular correlation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LooMode {
    /// On for 3..=20 inputs. MRtrix's `population_template` uses `2 < n < 15`;
    /// the analytic form here is free, so the window is wider.
    Auto,
    /// On whenever it is defined, which includes `n = 2`: there the leave-one-out
    /// template reduces to the *other* input, so `acc_loo` becomes the direct
    /// pairwise agreement — exactly the number a two-session test-retest wants.
    /// The `_sd` companions are degenerate at `n = 2` (one value per voxel).
    On,
    Off,
}

impl LooMode {
    fn resolve(self, n: usize) -> bool {
        match self {
            LooMode::Auto => (3..=20).contains(&n),
            LooMode::On => n >= 2,
            LooMode::Off => false,
        }
    }
}

/// Per-subject FOD rescaling applied *before* aggregation.
///
/// Post-mean normalization would be a no-op on peak directions, so this only
/// matters for amplitude-carrying outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleMode {
    /// Multiply by 1. Correct for quantitative reconstructions (`consh
    /// --quantitative`, `mtnormalise`d MRtrix FODs) whose amplitudes already
    /// share a unit.
    None,
    /// Per voxel, divide by the ℓ=0 (DC) coefficient. Shape-only; annihilates
    /// apparent-fibre-density contrast.
    L0,
    /// Per voxel, scale so the FOD integrates to 1. Differs from [`ScaleMode::L0`]
    /// by a constant, so peak *directions* are identical.
    Integral,
}

impl ScaleMode {
    pub fn label(&self) -> &'static str {
        match self {
            ScaleMode::None => "none",
            ScaleMode::L0 => "l0",
            ScaleMode::Integral => "integral",
        }
    }

    /// The multiplier for one SH row, given its ℓ=0 coefficient.
    #[inline]
    fn row_scale(&self, c0: f32) -> f32 {
        match self {
            ScaleMode::None => 1.0,
            ScaleMode::L0 => {
                if c0.abs() > 1e-8 {
                    1.0 / c0
                } else {
                    1.0
                }
            }
            ScaleMode::Integral => {
                if c0.abs() > 1e-8 {
                    1.0 / (c0 * 2.0 * std::f32::consts::PI.sqrt())
                } else {
                    1.0
                }
            }
        }
    }
}

/// Knobs for [`aggregate_fod`] and [`fod_qc`].
#[derive(Clone, Debug)]
pub struct AggregateOptions {
    pub scale: ScaleMode,
    /// Keep a voxel when at least this fraction of inputs cover it. `0` behaves
    /// as a mask union, `1` as an intersection.
    pub min_coverage: f32,
    pub lmax: LmaxPolicy,
    pub loo: LooMode,
    /// Lowest SH band included in the angular correlation coefficient. `2`
    /// excludes the isotropic term, which otherwise drives ACC to ~1 everywhere.
    pub acc_lmin: usize,
}

impl Default for AggregateOptions {
    fn default() -> Self {
        Self {
            scale: ScaleMode::None,
            min_coverage: 0.0,
            lmax: LmaxPolicy::Min,
            loo: LooMode::Auto,
            acc_lmin: 2,
        }
    }
}

/// The template voxel set: voxels whose contributor fraction reaches
/// `min_coverage`. `0` behaves as a mask union, `1` as an intersection.
///
/// DSI Studio's `odf_average` *documents* a ">half the population" rule but its
/// code keeps any voxel with at least one contributor; this implements the
/// documented rule.
pub(crate) fn coverage_mask(lookups: &[Vec<usize>], dims: [u64; 3], min_coverage: f32) -> Vec<u8> {
    let total = (dims[0] * dims[1] * dims[2]) as usize;
    let n = lookups.len();
    let mut present = vec![0u32; total];
    for lk in lookups {
        for (flat, &c) in lk.iter().enumerate() {
            if c != usize::MAX {
                present[flat] += 1;
            }
        }
    }
    let frac = min_coverage.clamp(0.0, 1.0) as f64;
    present
        .iter()
        .map(|&c| u8::from(c > 0 && (c as f64) / (n as f64) >= frac - 1e-9))
        .collect()
}

/// One input, opened and aligned to the reference grid.
pub(crate) struct PreparedInput {
    /// Owned only when a basis conversion was needed; otherwise the caller's.
    converted: Option<OdxDataset>,
    /// Reference-grid flat index → this input's compact row (`usize::MAX` out).
    pub(crate) lookup: Vec<usize>,
    /// True when this input's SH had to be rewritten into the reference basis.
    pub(crate) basis_converted: bool,
    /// The input's own lmax, when it exceeded the resolved cohort lmax.
    pub(crate) lmax_truncated_from: Option<u64>,
}

/// The resolved SH metadata every input was brought onto.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShTarget {
    pub order: u64,
    pub ncoeffs: usize,
    pub basis_name: String,
    pub full_basis: bool,
    pub legacy: bool,
}

impl ShTarget {
    /// Read the SH block of a header, erroring with an actionable message when
    /// it is absent or unrecognized.
    pub fn from_header(header: &Header, label: &str) -> Result<Self> {
        let order = header.sh_order.ok_or_else(|| {
            OdxError::Argument(format!(
                "'{label}' has SH coefficients but no SH_ORDER in its header"
            ))
        })?;
        let basis_name = header.sh_basis.clone().ok_or_else(|| {
            OdxError::Argument(format!(
                "'{label}' has SH coefficients but no SH_BASIS in its header"
            ))
        })?;
        let full_basis = header.sh_full_basis.unwrap_or(false);
        let legacy = header.sh_legacy.unwrap_or(false);
        let ncoeffs = if full_basis {
            descoteaux_sh::ncoeffs_for(order as usize, true)
        } else {
            mrtrix_sh::ncoeffs_for_lmax(order as usize)
        };
        Ok(Self {
            order,
            ncoeffs,
            basis_name,
            full_basis,
            legacy,
        })
    }

    /// Coefficient count for a (possibly lower) order in this basis.
    fn ncoeffs_for_order(&self, order: u64) -> usize {
        if self.full_basis {
            descoteaux_sh::ncoeffs_for(order as usize, true)
        } else {
            mrtrix_sh::ncoeffs_for_lmax(order as usize)
        }
    }

    /// The peak-finding / evaluation basis. Resolved from all four header
    /// fields together — `sh_basis` alone is not enough, since a descoteaux
    /// dataset evaluated without its `legacy` bit gets the wrong sign on m<0.
    pub fn basis_kind(&self) -> Result<ShBasisKind> {
        let dipy = match self.basis_name.to_ascii_lowercase().as_str() {
            "tournier07" | "mrtrix" | "mrtrix3" => "tournier07",
            "descoteaux07" | "dipy" => {
                if self.legacy {
                    "descoteaux07_legacy"
                } else {
                    "descoteaux07"
                }
            }
            other => {
                return Err(OdxError::Format(format!("unrecognized SH basis '{other}'")))
            }
        };
        basis_kind_from_dipy_name(dipy, self.order as usize, self.full_basis)
    }

    fn conversion_target(&self) -> Result<ShBasisTarget> {
        match self.basis_name.to_ascii_lowercase().as_str() {
            "tournier07" | "mrtrix" | "mrtrix3" => Ok(ShBasisTarget::Tournier07),
            "descoteaux07" | "dipy" => Ok(ShBasisTarget::Descoteaux07 {
                legacy: self.legacy,
            }),
            other => Err(OdxError::Format(format!("unrecognized SH basis '{other}'"))),
        }
    }

    /// True when `other` needs rewriting to land in this basis.
    fn differs_from(&self, other: &ShTarget) -> bool {
        !self.basis_name.eq_ignore_ascii_case(&other.basis_name)
            || self.legacy != other.legacy
            || self.full_basis != other.full_basis
    }
}

/// The aggregated FOD field plus everything needed to write and evaluate it.
pub struct AggregatedFod {
    /// `n_vox × ncoeffs`, row-major, in template compact-voxel order.
    pub sh: Vec<f32>,
    /// Contributing inputs per template voxel.
    pub counts: Vec<u32>,
    /// Full-volume template mask (C-order).
    pub mask: Vec<u8>,
    /// `(i, j, k)` of each template voxel, compact order.
    pub ijk: Vec<[u32; 3]>,
    pub target: ShTarget,
    /// Number of inputs that went into the aggregate.
    pub n_inputs: usize,
}

impl AggregatedFod {
    pub fn n_voxels(&self) -> usize {
        self.counts.len()
    }

    pub fn ncoeffs(&self) -> usize {
        self.target.ncoeffs
    }

    pub fn basis_kind(&self) -> Result<ShBasisKind> {
        self.target.basis_kind()
    }

    /// The ℓ=0 coefficient of every template voxel.
    pub fn l0(&self) -> Vec<f32> {
        let c = self.target.ncoeffs;
        (0..self.n_voxels()).map(|v| self.sh[v * c]).collect()
    }
}

/// Open every input, align it to the reference grid, and bring its SH onto the
/// reference basis. Returns the prepared inputs and the resolved SH target.
///
/// `datasets` are borrowed for the lifetime of the returned [`PreparedInput`]s
/// unless a basis conversion was needed, in which case the converted copy is
/// owned by the [`PreparedInput`].
pub(crate) fn prepare_inputs(
    datasets: &[OdxDataset],
    labels: &[String],
    ref_dims: [u64; 3],
    ref_affine: &[[f64; 4]; 4],
    ref_target: &ShTarget,
    policy: LmaxPolicy,
) -> Result<(Vec<PreparedInput>, ShTarget)> {
    // Resolve the cohort order first: every input contributes its own lmax.
    let mut orders = Vec::with_capacity(datasets.len());
    for (ds, label) in datasets.iter().zip(labels) {
        if ds.get_sh("coefficients").is_none() {
            return Err(OdxError::Argument(format!(
                "input '{label}' has no sh/coefficients; FOD averaging needs SH on every \
                 input (use `--method cluster` for peak-only files)"
            )));
        }
        orders.push(ShTarget::from_header(ds.header(), label)?.order);
    }
    let resolved_order = match policy {
        LmaxPolicy::Min => *orders.iter().min().expect("non-empty"),
        LmaxPolicy::Max => *orders.iter().max().expect("non-empty"),
        LmaxPolicy::Fixed(n) => n,
    };
    if matches!(policy, LmaxPolicy::Max) && orders.iter().any(|&o| o < resolved_order) {
        eprintln!(
            "odx: warning: --lmax max zero-pads lower-order inputs, so the template's \
             effective sharpness varies with which subjects cover which voxel; --lmax min \
             is the uniform choice"
        );
    }
    let mut target = ref_target.clone();
    target.order = resolved_order;
    target.ncoeffs = ref_target.ncoeffs_for_order(resolved_order);

    let mut prepared = Vec::with_capacity(datasets.len());
    for (ds, label) in datasets.iter().zip(labels) {
        let own = ShTarget::from_header(ds.header(), label)?;
        if target.full_basis && !own.full_basis {
            return Err(OdxError::Argument(format!(
                "reference uses the full (asymmetric) SH basis but input '{label}' is \
                 symmetric; averaging would silently zero the odd bands — pick a symmetric \
                 reference with --reference"
            )));
        }
        let (converted, basis_converted) = if target.differs_from(&own) {
            let out = convert_sh_basis(ds, target.conversion_target()?, None).map_err(|e| {
                OdxError::Argument(format!(
                    "input '{label}' is in SH basis '{}' and could not be converted to the \
                     reference basis '{}': {e}",
                    own.basis_name, target.basis_name
                ))
            })?;
            (Some(out), true)
        } else {
            (None, false)
        };
        let src = converted.as_ref().unwrap_or(ds);
        let h = src.header();
        let lookup = align_to_reference_grid(
            ref_dims,
            ref_affine,
            h.dimensions,
            &h.voxel_to_rasmm,
            src.mask(),
            label,
        )?;
        prepared.push(PreparedInput {
            converted,
            lookup,
            basis_converted,
            lmax_truncated_from: (own.order > resolved_order).then_some(own.order),
        });
    }
    Ok((prepared, target))
}

/// Borrow one prepared input's SH array, preferring the converted copy.
fn sh_rows<'a>(prep: &'a PreparedInput, fallback: &'a OdxDataset) -> &'a crate::data_array::DataArray {
    prep.converted
        .as_ref()
        .unwrap_or(fallback)
        .get_sh("coefficients")
        .expect("prepare_inputs verified sh/coefficients exists")
}

/// Read one SH row into `dst`, truncating or zero-padding to `dst.len()`.
///
/// The f32 case borrows straight off the mmap; other storage dtypes go through
/// a per-element convert. Truncation is a prefix slice because both bases are
/// band-ordered ascending — see the module docs.
fn read_sh_row(arr: &crate::data_array::DataArray, row: usize, dst: &mut [f32]) -> Result<()> {
    let src_cols = arr.ncols();
    let take = src_cols.min(dst.len());
    match arr.dtype() {
        crate::dtype::DType::Float32 => {
            let view = arr.typed_view::<f32>();
            dst[..take].copy_from_slice(&view.row(row)[..take]);
        }
        crate::dtype::DType::Float64 => {
            let view = arr.typed_view::<f64>();
            for (d, &s) in dst[..take].iter_mut().zip(&view.row(row)[..take]) {
                *d = s as f32;
            }
        }
        crate::dtype::DType::Float16 => {
            let view = arr.typed_view::<half::f16>();
            for (d, &s) in dst[..take].iter_mut().zip(&view.row(row)[..take]) {
                *d = s.to_f32();
            }
        }
        other => {
            return Err(OdxError::DType(format!(
                "sh/coefficients has dtype {other}; expected a float type"
            )))
        }
    }
    dst[take..].fill(0.0);
    Ok(())
}

/// Per-coefficient mean of the inputs' SH, accumulated in f64.
///
/// The template mask is the set of voxels whose contributor fraction reaches
/// `opts.min_coverage`; every kept voxel is divided by its own contributor
/// count. f64 accumulation because an f32 running sum visibly loses bits at the
/// tail over hundreds of subjects (DSI Studio accumulates in double for the
/// same reason).
pub(crate) fn aggregate_fod(
    datasets: &[OdxDataset],
    prepared: &[PreparedInput],
    dims: [u64; 3],
    mask: &[u8],
    target: &ShTarget,
    opts: &AggregateOptions,
) -> Result<AggregatedFod> {
    let n = datasets.len();
    let total = (dims[0] * dims[1] * dims[2]) as usize;
    let ncoeffs = target.ncoeffs;

    let ijk = mask_compact_ijk(mask, dims);
    let n_vox = ijk.len();
    if n_vox == 0 {
        return Err(OdxError::Argument(
            "the template mask is empty; lower --min-coverage or check that the inputs \
             overlap"
                .into(),
        ));
    }
    let mask = mask.to_vec();
    // Reference-flat → template compact row.
    let mut flat_to_t = vec![usize::MAX; total];
    for (t, v) in ijk.iter().enumerate() {
        flat_to_t[crate::fixel_match::flat_index(*v, dims)] = t;
    }

    let mut sum = vec![0.0f64; n_vox * ncoeffs];
    let mut counts = vec![0u32; n_vox];
    let mut row = vec![0.0f32; ncoeffs];

    // PARALLEL SEAM: the per-voxel accumulation writes disjoint
    // `sum[t*ncoeffs..]` slices, so inverting to voxel-outer/subject-inner makes
    // this a `par_chunks_mut` away from being embarrassingly parallel.
    for (s, prep) in prepared.iter().enumerate() {
        let arr = sh_rows(prep, &datasets[s]);
        if arr.ncols() < ncoeffs && !matches!(opts.lmax, LmaxPolicy::Max) {
            return Err(OdxError::Argument(format!(
                "input {s} has {} SH coefficients, fewer than the resolved {ncoeffs}; \
                 use --lmax min (the default) to truncate the cohort instead",
                arr.ncols()
            )));
        }
        for (flat, &c) in prep.lookup.iter().enumerate() {
            if c == usize::MAX {
                continue;
            }
            let t = flat_to_t[flat];
            if t == usize::MAX {
                continue;
            }
            read_sh_row(arr, c, &mut row)?;
            let scale = opts.scale.row_scale(row[0]);
            let dst = &mut sum[t * ncoeffs..(t + 1) * ncoeffs];
            for (d, &r) in dst.iter_mut().zip(row.iter()) {
                *d += (r * scale) as f64;
            }
            counts[t] += 1;
        }
    }

    let mut sh = vec![0.0f32; n_vox * ncoeffs];
    for t in 0..n_vox {
        let div = counts[t].max(1) as f64;
        for (o, &acc) in sh[t * ncoeffs..(t + 1) * ncoeffs]
            .iter_mut()
            .zip(&sum[t * ncoeffs..(t + 1) * ncoeffs])
        {
            *o = (acc / div) as f32;
        }
    }

    Ok(AggregatedFod {
        sh,
        counts,
        mask,
        ijk,
        target: target.clone(),
        n_inputs: n,
    })
}

/// Peak-find the aggregate in **its own** SH basis.
///
/// The basis comes from [`AggregatedFod::basis_kind`], which reads all four of
/// `sh_basis`/`sh_order`/`sh_full_basis`/`sh_legacy`. Assuming tournier here
/// silently mis-orients descoteaux cohorts.
pub(crate) fn peaks_from_aggregate(
    agg: &AggregatedFod,
    cfg: &PeakFinderConfig,
    sphere: Option<(&[[f32; 3]], &[[u32; 3]])>,
) -> Result<(Vec<u32>, Vec<[f32; 3]>, Vec<f32>)> {
    let finder = match sphere {
        Some((v, f)) => SpherePeakFinder::new(v, f, cfg.clone()),
        None => SpherePeakFinder::for_dsistudio_odf8(cfg.clone()),
    };
    // PARALLEL SEAM: row-independent.
    peaks_from_sh_rows_with_basis(&agg.sh, agg.n_voxels(), &finder, agg.basis_kind()?)
}

/// Per-voxel reproducibility maps, and the per-subject summaries that expose
/// outliers. All vectors are in template compact-voxel order.
pub struct FodQc {
    pub coverage_frac: Vec<f32>,
    pub l0_mean: Vec<f32>,
    pub l0_sd: Vec<f32>,
    pub l0_cv: Vec<f32>,
    pub acc_mean: Vec<f32>,
    pub acc_sd: Vec<f32>,
    pub acc_min: Vec<f32>,
    /// Empty when leave-one-out is off or n < 3.
    pub acc_loo_mean: Vec<f32>,
    pub acc_loo_min: Vec<f32>,
    /// Per subject: mean ACC over the template voxels it covers.
    pub subject_acc: Vec<f32>,
    /// Per subject: mean leave-one-out ACC. All `NaN` when LOO is off.
    pub subject_acc_loo: Vec<f32>,
    /// Per subject: template voxels covered.
    pub subject_voxels: Vec<u64>,
    pub loo_enabled: bool,
    /// Template voxels whose aggregate carries no anisotropic energy, so ACC is
    /// undefined there. Large in multi-tissue cohorts, where the WM compartment
    /// is identically zero over much of the brain mask.
    pub n_voxels_without_orientation: u64,
}

/// Index of the first coefficient with ℓ >= `lmin`, i.e. how many leading
/// coefficients the ACC skips.
///
/// In a symmetric basis only even bands are stored, so an odd `lmin` resolves
/// to the next even band — `lmin = 3` must skip all of ℓ=2, not land in the
/// middle of it.
fn acc_skip(target: &ShTarget, lmin: usize) -> usize {
    if lmin == 0 {
        return 0;
    }
    if target.full_basis {
        // Full basis packs every m of every ℓ: lmin² coefficients precede ℓ=lmin.
        (lmin * lmin).min(target.ncoeffs)
    } else {
        // Symmetric basis: round up to the next even band, then count what
        // precedes it. ℓ=2 starts after ncoeffs(0)=1, ℓ=4 after ncoeffs(2)=6.
        let even = lmin + (lmin % 2);
        mrtrix_sh::ncoeffs_for_lmax(even - 2).min(target.ncoeffs)
    }
}

/// Angular correlation coefficient of two SH rows over the bands at or above
/// `skip`.
///
/// `NaN` when either side's anisotropic energy is at or below `floor`, which
/// means the voxel carries no orientation information and any correlation
/// between the two rows is arithmetic on rounding noise. This is not a corner
/// case: multi-tissue deconvolution leaves the WM compartment identically zero
/// across a large interior region of a brain mask, and without the floor those
/// voxels return ~1/√n — a *finite* number that then drags every whole-brain
/// summary toward it.
#[inline]
fn acc(u: &[f32], v: &[f32], skip: usize, floor: f64) -> f32 {
    let (mut num, mut nu, mut nv) = (0.0f64, 0.0f64, 0.0f64);
    for k in skip..u.len() {
        let (a, b) = (u[k] as f64, v[k] as f64);
        num += a * b;
        nu += a * a;
        nv += b * b;
    }
    let f2 = floor * floor;
    if nu <= f2 || nv <= f2 {
        return f32::NAN;
    }
    (num / (nu.sqrt() * nv.sqrt())) as f32
}

/// The anisotropic-energy floor below which a row carries no orientation.
///
/// Scale-free by construction: `ANISOTROPY_FLOOR_FRACTION` of the median ℓ≥2
/// norm over template voxels that have any. That adapts to whatever units the
/// cohort is in (absolute AFD, S/S0, ℓ=0-normalized) instead of hardcoding a
/// threshold that only suits one of them.
const ANISOTROPY_FLOOR_FRACTION: f64 = 1e-6;

fn anisotropy_floor(agg: &AggregatedFod, skip: usize) -> f64 {
    let c = agg.ncoeffs();
    let mut norms: Vec<f64> = (0..agg.n_voxels())
        .map(|v| {
            agg.sh[v * c + skip..(v + 1) * c]
                .iter()
                .map(|&x| (x as f64) * (x as f64))
                .sum::<f64>()
                .sqrt()
        })
        .filter(|n| *n > 0.0)
        .collect();
    if norms.is_empty() {
        return 0.0;
    }
    let mid = norms.len() / 2;
    norms.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    norms[mid] * ANISOTROPY_FLOOR_FRACTION
}

/// Running mean/sd/min accumulator that ignores `NaN`.
#[derive(Clone, Copy, Default)]
struct NanStats {
    n: u32,
    sum: f64,
    sumsq: f64,
    min: f32,
}

impl NanStats {
    fn new() -> Self {
        Self { n: 0, sum: 0.0, sumsq: 0.0, min: f32::INFINITY }
    }
    #[inline]
    fn push(&mut self, v: f32) {
        if v.is_nan() {
            return;
        }
        self.n += 1;
        self.sum += v as f64;
        self.sumsq += (v as f64) * (v as f64);
        if v < self.min {
            self.min = v;
        }
    }
    fn mean(&self) -> f32 {
        if self.n == 0 { f32::NAN } else { (self.sum / self.n as f64) as f32 }
    }
    /// Sample standard deviation (`n-1` divisor); `NaN` below two values.
    fn sd(&self) -> f32 {
        if self.n < 2 {
            return f32::NAN;
        }
        let n = self.n as f64;
        let var = (self.sumsq - self.sum * self.sum / n) / (n - 1.0);
        (var.max(0.0)).sqrt() as f32
    }
    fn min_or_nan(&self) -> f32 {
        if self.n == 0 { f32::NAN } else { self.min }
    }
}

/// Second pass over the subject SH: ACC against the template, the analytic
/// leave-one-out ACC, and the ℓ=0 spread.
///
/// The leave-one-out template is `(S_v − x_iv) / (c_v − 1)` where `S_v` is the
/// contributor sum already folded into `agg.sh` and `c_v` the contributor count
/// — one vector subtract per (subject, voxel), no re-aggregation. It removes
/// each subject's own contribution to the reference it is scored against, which
/// otherwise inflates ACC by ~1/n.
pub(crate) fn fod_qc(
    datasets: &[OdxDataset],
    prepared: &[PreparedInput],
    dims: [u64; 3],
    agg: &AggregatedFod,
    opts: &AggregateOptions,
) -> Result<FodQc> {
    let n = datasets.len();
    let n_vox = agg.n_voxels();
    let ncoeffs = agg.ncoeffs();
    let skip = acc_skip(&agg.target, opts.acc_lmin);
    let floor = anisotropy_floor(agg, skip);
    let loo_enabled = opts.loo.resolve(n);

    let total = (dims[0] * dims[1] * dims[2]) as usize;
    let mut flat_to_t = vec![usize::MAX; total];
    for (t, v) in agg.ijk.iter().enumerate() {
        flat_to_t[crate::fixel_match::flat_index(*v, dims)] = t;
    }

    let mut l0 = vec![NanStats::new(); n_vox];
    let mut acc_stats = vec![NanStats::new(); n_vox];
    let mut acc_loo_stats = vec![NanStats::new(); n_vox];
    let mut subject_acc = vec![NanStats::new(); n];
    let mut subject_acc_loo = vec![NanStats::new(); n];
    let mut subject_voxels = vec![0u64; n];

    let mut row = vec![0.0f32; ncoeffs];
    let mut loo_row = vec![0.0f32; ncoeffs];

    // PARALLEL SEAM: per-voxel, no cross-voxel state.
    for (s, prep) in prepared.iter().enumerate() {
        let arr = sh_rows(prep, &datasets[s]);
        for (flat, &c) in prep.lookup.iter().enumerate() {
            if c == usize::MAX {
                continue;
            }
            let t = flat_to_t[flat];
            if t == usize::MAX {
                continue;
            }
            read_sh_row(arr, c, &mut row)?;
            let scale = opts.scale.row_scale(row[0]);
            // Record the ℓ=0 term BEFORE scaling. Under `--normalize-fod l0`
            // the post-scale value is exactly 1.0 by construction, which would
            // make l0_mean/sd/cv degenerate; the pre-scale value is the one
            // that carries the apparent-fibre-density information.
            let l0_raw = row[0];
            if scale != 1.0 {
                for x in row.iter_mut() {
                    *x *= scale;
                }
            }
            subject_voxels[s] += 1;
            l0[t].push(l0_raw);

            let tmpl = &agg.sh[t * ncoeffs..(t + 1) * ncoeffs];
            let a = acc(&row, tmpl, skip, floor);
            acc_stats[t].push(a);
            subject_acc[s].push(a);

            if loo_enabled && agg.counts[t] >= 2 {
                let c_v = agg.counts[t] as f32;
                for (d, (&m, &x)) in loo_row.iter_mut().zip(tmpl.iter().zip(row.iter())) {
                    // mean_{-i} = (c*mean - x) / (c - 1)
                    *d = (c_v * m - x) / (c_v - 1.0);
                }
                let al = acc(&row, &loo_row, skip, floor);
                acc_loo_stats[t].push(al);
                subject_acc_loo[s].push(al);
            }
        }
    }

    let coverage_frac = agg.counts.iter().map(|&c| c as f32 / n as f32).collect();
    let l0_mean: Vec<f32> = l0.iter().map(|s| s.mean()).collect();
    let l0_sd: Vec<f32> = l0.iter().map(|s| s.sd()).collect();
    let l0_cv = l0_mean
        .iter()
        .zip(&l0_sd)
        .map(|(&m, &s)| if m.abs() > 1e-12 { s / m } else { f32::NAN })
        .collect();

    let n_voxels_without_orientation = (0..n_vox)
        .filter(|&t| {
            let row = &agg.sh[t * ncoeffs + skip..(t + 1) * ncoeffs];
            row.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt() <= floor
        })
        .count() as u64;

    Ok(FodQc {
        n_voxels_without_orientation,
        coverage_frac,
        l0_mean,
        l0_sd,
        l0_cv,
        acc_mean: acc_stats.iter().map(|s| s.mean()).collect(),
        acc_sd: acc_stats.iter().map(|s| s.sd()).collect(),
        acc_min: acc_stats.iter().map(|s| s.min_or_nan()).collect(),
        acc_loo_mean: if loo_enabled {
            acc_loo_stats.iter().map(|s| s.mean()).collect()
        } else {
            Vec::new()
        },
        acc_loo_min: if loo_enabled {
            acc_loo_stats.iter().map(|s| s.min_or_nan()).collect()
        } else {
            Vec::new()
        },
        subject_acc: subject_acc.iter().map(|s| s.mean()).collect(),
        subject_acc_loo: subject_acc_loo.iter().map(|s| s.mean()).collect(),
        subject_voxels,
        loo_enabled,
    })
}

/// Mean of the shared scalar DPVs across inputs, on the template voxel set.
///
/// Same per-voxel contributor divisor as the FOD. Returns `(mean, sd)` per key
/// in `keys` order; `sd` uses the `n-1` divisor and is `NaN` below two
/// contributors.
pub(crate) fn average_dpvs(
    datasets: &[OdxDataset],
    prepared: &[PreparedInput],
    dims: [u64; 3],
    agg: &AggregatedFod,
    keys: &[String],
) -> Result<Vec<(String, Vec<f32>, Vec<f32>)>> {
    let n_vox = agg.n_voxels();
    let total = (dims[0] * dims[1] * dims[2]) as usize;
    let mut flat_to_t = vec![usize::MAX; total];
    for (t, v) in agg.ijk.iter().enumerate() {
        flat_to_t[crate::fixel_match::flat_index(*v, dims)] = t;
    }

    let mut out = Vec::with_capacity(keys.len());
    for key in keys {
        let mut stats = vec![NanStats::new(); n_vox];
        for (s, prep) in prepared.iter().enumerate() {
            // DPVs are read from the *original* dataset: a basis conversion
            // rewrites SH only, and may not carry unrelated arrays.
            let Ok(vals) = datasets[s].scalar_dpv_f32(key) else {
                continue;
            };
            for (flat, &c) in prep.lookup.iter().enumerate() {
                if c == usize::MAX {
                    continue;
                }
                let t = flat_to_t[flat];
                if t == usize::MAX || c >= vals.len() {
                    continue;
                }
                stats[t].push(vals[c]);
            }
        }
        out.push((
            key.clone(),
            stats.iter().map(|s| s.mean()).collect(),
            stats.iter().map(|s| s.sd()).collect(),
        ));
    }
    Ok(out)
}

/// Anisotropic power of the aggregate, per template voxel.
pub(crate) fn aggregate_anisotropic_power(agg: &AggregatedFod) -> Vec<f32> {
    let c = agg.ncoeffs();
    let lmax = agg.target.order as usize;
    (0..agg.n_voxels())
        .map(|v| {
            let row = &agg.sh[v * c..(v + 1) * c];
            if agg.target.full_basis {
                descoteaux_sh::anisotropic_power_full_basis(
                    row,
                    lmax,
                    mrtrix_sh::ANISOTROPIC_POWER_NORM_FACTOR,
                )
            } else {
                mrtrix_sh::anisotropic_power(row, lmax, mrtrix_sh::ANISOTROPIC_POWER_NORM_FACTOR)
            }
        })
        .collect()
}

/// Flag subjects whose leave-one-out ACC or coverage sets them apart.
///
/// Robust-z on the median absolute deviation, because with a small cohort one
/// bad scan inflates the standard deviation enough to hide itself. The absolute
/// ACC floor is what stops a tight test-retest cohort — where the MAD is
/// vanishing — from flagging a perfectly good session on noise.
pub fn flag_outliers(acc_loo: &[f32], coverage: &[f64]) -> Vec<Vec<String>> {
    let n = acc_loo.len();
    let mut reasons: Vec<Vec<String>> = vec![Vec::new(); n];

    let finite: Vec<f32> = acc_loo.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.len() >= 3 {
        let med = median(&finite);
        let devs: Vec<f32> = finite.iter().map(|v| (v - med).abs()).collect();
        let mad = median(&devs) * 1.4826;
        let cut = med - 3.0 * mad;
        for (i, &v) in acc_loo.iter().enumerate() {
            if v.is_finite() && v < cut && v < 0.9 {
                reasons[i].push(format!(
                    "low_acc_loo (acc_loo={v:.3} < median {med:.3} - 3*MAD, and < 0.90)"
                ));
            }
        }
    }
    for (i, &cov) in coverage.iter().enumerate() {
        if cov < 0.9 {
            reasons[i].push(format!("low_coverage ({:.1}% of template voxels)", cov * 100.0));
        }
    }
    reasons
}

fn median(v: &[f32]) -> f32 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n == 0 {
        f32::NAN
    } else if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

/// Open a set of ODX paths, erroring with the offending path.
#[cfg(test)]
pub(crate) fn open_all(paths: &[std::path::PathBuf]) -> Result<Vec<OdxDataset>> {
    paths
        .iter()
        .map(|p: &std::path::PathBuf| {
            OdxDataset::open(p).map_err(|e| {
                OdxError::Format(format!("failed to open input '{}': {e}", p.display()))
            })
        })
        .collect()
}

/// The reference header for a cohort: an explicit `--reference` ODX, else the
/// first input.
pub(crate) fn reference_header(
    datasets: &[OdxDataset],
    labels: &[String],
    reference: Option<&Path>,
) -> Result<(ShTarget, [u64; 3], [[f64; 4]; 4])> {
    if let Some(p) = reference {
        let ds = OdxDataset::open(p).map_err(|e| {
            OdxError::Format(format!("failed to open --reference '{}': {e}", p.display()))
        })?;
        let h = ds.header();
        let target = ShTarget::from_header(h, &p.display().to_string())?;
        return Ok((target, h.dimensions, h.voxel_to_rasmm));
    }
    let h = datasets[0].header();
    Ok((
        ShTarget::from_header(h, &labels[0])?,
        h.dimensions,
        h.voxel_to_rasmm,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::mmap_backing::vec_into_bytes;
    use crate::stream::OdxBuilder;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn identity_affine() -> [[f64; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    /// Build an SH-only ODX. `rows` is one coefficient row per in-mask voxel.
    fn build_sh_input(
        dir: &std::path::Path,
        name: &str,
        affine: [[f64; 4]; 4],
        dims: [u64; 3],
        mask: Vec<u8>,
        rows: Vec<Vec<f32>>,
        order: u64,
        basis: &str,
        legacy: bool,
    ) -> PathBuf {
        let ncoeffs = rows[0].len();
        let flat: Vec<f32> = rows.into_iter().flatten().collect();
        let mut b = OdxBuilder::new(affine, dims, mask);
        b.set_sh_info(order, basis.to_string());
        b.set_sh_full_basis(false);
        b.set_sh_legacy(legacy);
        b.set_sh_data("coefficients", vec_into_bytes(flat), ncoeffs, DType::Float32);
        b.skip_all_peaks();
        let path = dir.join(format!("{name}.odx"));
        b.finalize().unwrap().save_directory(&path).unwrap();
        path
    }

    /// A lmax-8 tournier row for a narrow lobe along `dir`, fitted from
    /// amplitudes so the peak is a genuine maximum of the reconstruction.
    fn lobe_sh(dir: [f32; 3], lmax: usize, power: f32) -> Vec<f32> {
        let sphere = crate::formats::dsistudio_odf8::hemisphere_vertices_ras();
        let amps: Vec<f32> = sphere
            .iter()
            .map(|v| {
                let d = (v[0] * dir[0] + v[1] * dir[1] + v[2] * dir[2]).abs();
                d.powf(power)
            })
            .collect();
        crate::mrtrix_sh::fit_from_amplitudes(&amps, &sphere, lmax).unwrap()
    }

    fn prep(
        datasets: &[OdxDataset],
        labels: &[String],
        dims: [u64; 3],
        affine: &[[f64; 4]; 4],
        policy: LmaxPolicy,
    ) -> (Vec<PreparedInput>, ShTarget) {
        let ref_t = ShTarget::from_header(datasets[0].header(), &labels[0]).unwrap();
        prepare_inputs(datasets, labels, dims, affine, &ref_t, policy).unwrap()
    }

    fn lookups_of(prepared: &[PreparedInput]) -> Vec<Vec<usize>> {
        prepared.iter().map(|p| p.lookup.clone()).collect()
    }

    fn labels(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn mean_of_identical_inputs_is_the_input() {
        let tmp = TempDir::new().unwrap();
        let row = lobe_sh([0.0, 0.0, 1.0], 8, 8.0);
        let paths: Vec<PathBuf> = ["a", "b", "c"]
            .iter()
            .map(|n| {
                build_sh_input(
                    tmp.path(), n, identity_affine(), [1, 1, 1], vec![1],
                    vec![row.clone()], 8, "tournier07", false,
                )
            })
            .collect();
        let ds = open_all(&paths).unwrap();
        let l = labels(&["a", "b", "c"]);
        let (prepared, target) = prep(&ds, &l, [1, 1, 1], &identity_affine(), LmaxPolicy::Min);
        let opts = AggregateOptions { loo: LooMode::On, ..Default::default() };
        let agg = aggregate_fod(&ds, &prepared, [1, 1, 1], &coverage_mask(&lookups_of(&prepared), [1, 1, 1], 0.0), &target, &opts).unwrap();

        for (got, want) in agg.sh.iter().zip(&row) {
            assert!((got - want).abs() < 1e-6, "mean of identical rows must be the row");
        }
        let qc = fod_qc(&ds, &prepared, [1, 1, 1], &agg, &opts).unwrap();
        assert!((qc.acc_mean[0] - 1.0).abs() < 1e-5, "acc {}", qc.acc_mean[0]);
        assert!(qc.l0_sd[0].abs() < 1e-6, "l0_sd {}", qc.l0_sd[0]);
        assert!(qc.loo_enabled);
        assert!((qc.acc_loo_mean[0] - 1.0).abs() < 1e-4, "loo acc {}", qc.acc_loo_mean[0]);
    }

    #[test]
    fn mean_recovers_analytic_mean() {
        let tmp = TempDir::new().unwrap();
        let mk = |n: &str, c0: f32| {
            let mut r = vec![0.0f32; 45];
            r[0] = c0;
            build_sh_input(
                tmp.path(), n, identity_affine(), [1, 1, 1], vec![1], vec![r], 8,
                "tournier07", false,
            )
        };
        let paths = vec![mk("a", 1.0), mk("b", 3.0)];
        let ds = open_all(&paths).unwrap();
        let l = labels(&["a", "b"]);
        let (prepared, target) = prep(&ds, &l, [1, 1, 1], &identity_affine(), LmaxPolicy::Min);
        let agg =
            aggregate_fod(&ds, &prepared, [1, 1, 1], &coverage_mask(&lookups_of(&prepared), [1, 1, 1], 0.0), &target, &AggregateOptions::default()).unwrap();
        assert!((agg.sh[0] - 2.0).abs() < 1e-6, "got {}", agg.sh[0]);
    }

    #[test]
    fn partial_coverage_divides_by_contributors() {
        // a covers both voxels, b only the second. v0 must keep a's value.
        let tmp = TempDir::new().unwrap();
        let dims = [1u64, 1, 2];
        let mut r1 = vec![0.0f32; 45];
        r1[0] = 4.0;
        let a = build_sh_input(
            tmp.path(), "a", identity_affine(), dims, vec![1, 1],
            vec![r1.clone(), r1.clone()], 8, "tournier07", false,
        );
        let b = build_sh_input(
            tmp.path(), "b", identity_affine(), dims, vec![0, 1],
            vec![r1.clone()], 8, "tournier07", false,
        );
        let ds = open_all(&[a, b]).unwrap();
        let l = labels(&["a", "b"]);
        let (prepared, target) = prep(&ds, &l, dims, &identity_affine(), LmaxPolicy::Min);
        let agg = aggregate_fod(&ds, &prepared, dims, &coverage_mask(&lookups_of(&prepared), dims, 0.0), &target, &AggregateOptions::default()).unwrap();
        assert_eq!(agg.n_voxels(), 2);
        assert_eq!(agg.counts, vec![1, 2]);
        assert!((agg.sh[0] - 4.0).abs() < 1e-6, "single-contributor voxel must not be halved");
        assert!((agg.sh[45] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn min_coverage_thresholds_voxels() {
        let tmp = TempDir::new().unwrap();
        let dims = [1u64, 1, 2];
        let mut r = vec![0.0f32; 45];
        r[0] = 1.0;
        let a = build_sh_input(
            tmp.path(), "a", identity_affine(), dims, vec![1, 1],
            vec![r.clone(), r.clone()], 8, "tournier07", false,
        );
        let b = build_sh_input(
            tmp.path(), "b", identity_affine(), dims, vec![0, 1], vec![r.clone()], 8,
            "tournier07", false,
        );
        let ds = open_all(&[a, b]).unwrap();
        let l = labels(&["a", "b"]);
        let (prepared, target) = prep(&ds, &l, dims, &identity_affine(), LmaxPolicy::Min);
        let n_at = |frac: f32| {
            let o = AggregateOptions { min_coverage: frac, ..Default::default() };
            let m = coverage_mask(&lookups_of(&prepared), dims, frac);
            aggregate_fod(&ds, &prepared, dims, &m, &target, &o).unwrap().n_voxels()
        };
        assert_eq!(n_at(0.0), 2, "union");
        assert_eq!(n_at(0.5), 2, "1/2 coverage meets 0.5 inclusively");
        assert_eq!(n_at(1.0), 1, "intersection");
    }

    /// Both bases are band-ordered ascending, so lowering lmax is a prefix
    /// slice. The whole `--lmax min` policy rests on this.
    #[test]
    fn lmax_truncation_is_a_prefix_in_both_bases() {
        let dirs = crate::formats::dsistudio_odf8::hemisphere_vertices_ras();
        // tournier
        let full = crate::mrtrix_sh::sh2amp_cart(&dirs, 8);
        let low = crate::mrtrix_sh::sh2amp_cart(&dirs, 4);
        for r in 0..dirs.len() {
            for c in 0..15 {
                assert!(
                    (full[[r, c]] - low[[r, c]]).abs() < 1e-6,
                    "tournier basis column {c} must be identical at lmax 4 and 8"
                );
            }
        }
        // descoteaux, symmetric, modern
        let full = crate::descoteaux_sh::sh2amp_cart(&dirs, 8, false, false);
        let low = crate::descoteaux_sh::sh2amp_cart(&dirs, 4, false, false);
        for r in 0..dirs.len() {
            for c in 0..15 {
                assert!(
                    (full[[r, c]] - low[[r, c]]).abs() < 1e-6,
                    "descoteaux basis column {c} must be identical at lmax 4 and 8"
                );
            }
        }
    }

    #[test]
    fn lmax_min_policy_truncates_and_sets_header_order() {
        let tmp = TempDir::new().unwrap();
        let hi = lobe_sh([0.0, 0.0, 1.0], 8, 8.0);
        let lo: Vec<f32> = hi[..15].to_vec();
        let a = build_sh_input(
            tmp.path(), "a", identity_affine(), [1, 1, 1], vec![1], vec![hi.clone()], 8,
            "tournier07", false,
        );
        let b = build_sh_input(
            tmp.path(), "b", identity_affine(), [1, 1, 1], vec![1], vec![lo.clone()], 4,
            "tournier07", false,
        );
        let ds = open_all(&[a, b]).unwrap();
        let l = labels(&["a", "b"]);
        let (prepared, target) = prep(&ds, &l, [1, 1, 1], &identity_affine(), LmaxPolicy::Min);
        assert_eq!(target.order, 4);
        assert_eq!(target.ncoeffs, 15);
        assert_eq!(prepared[0].lmax_truncated_from, Some(8));
        assert_eq!(prepared[1].lmax_truncated_from, None);

        let agg =
            aggregate_fod(&ds, &prepared, [1, 1, 1], &coverage_mask(&lookups_of(&prepared), [1, 1, 1], 0.0), &target, &AggregateOptions::default()).unwrap();
        assert_eq!(agg.sh.len(), 15);
        // a truncated == b, so the mean is that row exactly.
        for (got, want) in agg.sh.iter().zip(&lo) {
            assert!((got - want).abs() < 1e-6);
        }
    }

    /// The regression test for the hardcoded-tournier peak finder: a descoteaux
    /// lobe must peak-find along its true axis, and the legacy bit must survive.
    #[test]
    fn descoteaux_inputs_keep_their_basis_and_legacy_bit() {
        for legacy in [false, true] {
            let tmp = TempDir::new().unwrap();
            let dirs = crate::formats::dsistudio_odf8::hemisphere_vertices_ras();
            let amps: Vec<f32> = dirs.iter().map(|v| v[2].abs().powf(8.0)).collect();
            let row = crate::descoteaux_sh::fit_rows_from_amplitudes(
                &amps, 1, &dirs, 8, false, legacy,
            )
            .unwrap();
            let p = build_sh_input(
                tmp.path(), "a", identity_affine(), [1, 1, 1], vec![1], vec![row], 8,
                "descoteaux07", legacy,
            );
            let ds = open_all(&[p]).unwrap();
            let l = labels(&["a"]);
            let (prepared, target) = prep(&ds, &l, [1, 1, 1], &identity_affine(), LmaxPolicy::Min);
            assert_eq!(target.basis_name, "descoteaux07");
            assert_eq!(target.legacy, legacy);
            let agg = aggregate_fod(&ds, &prepared, [1, 1, 1], &coverage_mask(&lookups_of(&prepared), [1, 1, 1], 0.0), &target, &AggregateOptions::default())
            .unwrap();
            let (_, peak_dirs, _) =
                peaks_from_aggregate(&agg, &PeakFinderConfig::default(), None).unwrap();
            assert!(!peak_dirs.is_empty(), "legacy={legacy}: no peak found");
            let d = peak_dirs[0];
            let dp = d[2].abs();
            assert!(
                dp > 0.996,
                "legacy={legacy}: peak {d:?} must lie within 5 deg of +z (|cos|={dp})"
            );
        }
    }

    /// ACC must ignore the isotropic term: two rows differing only in c0 are
    /// the same shape.
    #[test]
    fn acc_excludes_l0() {
        let tmp = TempDir::new().unwrap();
        let base = lobe_sh([0.0, 0.0, 1.0], 8, 8.0);
        let mut hi = base.clone();
        hi[0] += 5.0;
        let a = build_sh_input(
            tmp.path(), "a", identity_affine(), [1, 1, 1], vec![1], vec![base], 8,
            "tournier07", false,
        );
        let b = build_sh_input(
            tmp.path(), "b", identity_affine(), [1, 1, 1], vec![1], vec![hi], 8,
            "tournier07", false,
        );
        let ds = open_all(&[a, b]).unwrap();
        let l = labels(&["a", "b"]);
        let (prepared, target) = prep(&ds, &l, [1, 1, 1], &identity_affine(), LmaxPolicy::Min);
        let opts = AggregateOptions::default();
        let agg = aggregate_fod(&ds, &prepared, [1, 1, 1], &coverage_mask(&lookups_of(&prepared), [1, 1, 1], 0.0), &target, &opts).unwrap();
        let qc = fod_qc(&ds, &prepared, [1, 1, 1], &agg, &opts).unwrap();
        assert!((qc.acc_mean[0] - 1.0).abs() < 1e-5, "acc {}", qc.acc_mean[0]);
        assert!(qc.l0_sd[0] > 1.0, "l0 SD must still see the DC difference");
    }

    /// The analytic leave-one-out identity, checked against literally rebuilding
    /// the template without each subject.
    #[test]
    fn loo_analytic_matches_bruteforce() {
        let tmp = TempDir::new().unwrap();
        let axes = [
            [0.0f32, 0.0, 1.0],
            [0.15, 0.0, 0.988],
            [0.0, 0.2, 0.979],
            [-0.1, 0.1, 0.99],
        ];
        let paths: Vec<PathBuf> = axes
            .iter()
            .enumerate()
            .map(|(i, d)| {
                build_sh_input(
                    tmp.path(), &format!("s{i}"), identity_affine(), [1, 1, 1], vec![1],
                    vec![lobe_sh(*d, 8, 8.0)], 8, "tournier07", false,
                )
            })
            .collect();
        let ds = open_all(&paths).unwrap();
        let l = labels(&["s0", "s1", "s2", "s3"]);
        let (prepared, target) = prep(&ds, &l, [1, 1, 1], &identity_affine(), LmaxPolicy::Min);
        let opts = AggregateOptions { loo: LooMode::On, ..Default::default() };
        let agg = aggregate_fod(&ds, &prepared, [1, 1, 1], &coverage_mask(&lookups_of(&prepared), [1, 1, 1], 0.0), &target, &opts).unwrap();
        let qc = fod_qc(&ds, &prepared, [1, 1, 1], &agg, &opts).unwrap();

        // Brute force: rebuild the template from the other three, per subject.
        let rows: Vec<Vec<f32>> = axes.iter().map(|d| lobe_sh(*d, 8, 8.0)).collect();
        let skip = acc_skip(&target, 2);
        for i in 0..4 {
            let mut mean = vec![0.0f32; target.ncoeffs];
            for (j, r) in rows.iter().enumerate() {
                if j == i {
                    continue;
                }
                for (m, &x) in mean.iter_mut().zip(r) {
                    *m += x / 3.0;
                }
            }
            let want = acc(&rows[i], &mean, skip, 0.0);
            let got = qc.subject_acc_loo[i];
            assert!(
                (want - got).abs() < 1e-5,
                "subject {i}: analytic LOO {got} vs brute force {want}"
            );
        }
    }

    /// At n = 2 the leave-one-out template is algebraically the other input, so
    /// acc_loo is the direct session-to-session agreement.
    #[test]
    fn loo_at_two_inputs_is_the_pairwise_agreement() {
        let tmp = TempDir::new().unwrap();
        let a_row = lobe_sh([0.0, 0.0, 1.0], 8, 8.0);
        let b_row = lobe_sh([0.12, 0.0, 0.993], 8, 8.0);
        let a = build_sh_input(
            tmp.path(), "a", identity_affine(), [1, 1, 1], vec![1], vec![a_row.clone()], 8,
            "tournier07", false,
        );
        let b = build_sh_input(
            tmp.path(), "b", identity_affine(), [1, 1, 1], vec![1], vec![b_row.clone()], 8,
            "tournier07", false,
        );
        let ds = open_all(&[a, b]).unwrap();
        let l = labels(&["a", "b"]);
        let (prepared, target) = prep(&ds, &l, [1, 1, 1], &identity_affine(), LmaxPolicy::Min);
        let opts = AggregateOptions { loo: LooMode::On, ..Default::default() };
        let agg = aggregate_fod(
            &ds, &prepared, [1, 1, 1],
            &coverage_mask(&lookups_of(&prepared), [1, 1, 1], 0.0), &target, &opts,
        )
        .unwrap();
        let qc = fod_qc(&ds, &prepared, [1, 1, 1], &agg, &opts).unwrap();
        assert!(qc.loo_enabled, "n=2 leave-one-out is well defined");

        let skip = acc_skip(&target, 2);
        let pairwise = acc(&a_row, &b_row, skip, 0.0);
        assert!(
            (qc.subject_acc_loo[0] - pairwise).abs() < 1e-5,
            "n=2 acc_loo must equal the direct a-vs-b agreement: {} vs {pairwise}",
            qc.subject_acc_loo[0]
        );
        assert!(
            (qc.subject_acc_loo[1] - pairwise).abs() < 1e-5,
            "and symmetrically for b"
        );
        // Auto still keeps its 3..=20 window, so the default is unchanged.
        let auto = AggregateOptions { loo: LooMode::Auto, ..Default::default() };
        let qc_auto = fod_qc(&ds, &prepared, [1, 1, 1], &agg, &auto).unwrap();
        assert!(!qc_auto.loo_enabled, "--loo auto must stay off at n=2");
    }

    #[test]
    fn loo_off_leaves_the_maps_empty() {
        let tmp = TempDir::new().unwrap();
        let row = lobe_sh([0.0, 0.0, 1.0], 8, 8.0);
        let paths: Vec<PathBuf> = ["a", "b", "c"]
            .iter()
            .map(|n| {
                build_sh_input(
                    tmp.path(), n, identity_affine(), [1, 1, 1], vec![1], vec![row.clone()], 8,
                    "tournier07", false,
                )
            })
            .collect();
        let ds = open_all(&paths).unwrap();
        let l = labels(&["a", "b", "c"]);
        let (prepared, target) = prep(&ds, &l, [1, 1, 1], &identity_affine(), LmaxPolicy::Min);
        let opts = AggregateOptions { loo: LooMode::Off, ..Default::default() };
        let agg = aggregate_fod(&ds, &prepared, [1, 1, 1], &coverage_mask(&lookups_of(&prepared), [1, 1, 1], 0.0), &target, &opts).unwrap();
        let qc = fod_qc(&ds, &prepared, [1, 1, 1], &agg, &opts).unwrap();
        assert!(!qc.loo_enabled);
        assert!(qc.acc_loo_mean.is_empty());
    }

    /// A same-lattice input whose voxels are stored in the opposite z order must
    /// be reindexed, giving the same template as if it had matched.
    #[test]
    fn same_lattice_flipped_input_averages_correctly() {
        let tmp = TempDir::new().unwrap();
        let dims = [1u64, 1, 2];
        let reference = [
            [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        ];
        let flipped = [
            [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -2.0, 2.0], [0.0, 0.0, 0.0, 1.0],
        ];
        let z = lobe_sh([0.0, 0.0, 1.0], 8, 8.0);
        let x = lobe_sh([1.0, 0.0, 0.0], 8, 8.0);
        // physical truth: world z=0 → z-lobe, world z=2 → x-lobe
        let a = build_sh_input(
            tmp.path(), "a", reference, dims, vec![1, 1], vec![z.clone(), x.clone()], 8,
            "tournier07", false,
        );
        let b = build_sh_input(
            tmp.path(), "b", flipped, dims, vec![1, 1], vec![x.clone(), z.clone()], 8,
            "tournier07", false,
        );
        let ds = open_all(&[a, b]).unwrap();
        let l = labels(&["a", "b"]);
        let (prepared, target) = prep(&ds, &l, dims, &reference, LmaxPolicy::Min);
        let opts = AggregateOptions::default();
        let agg = aggregate_fod(&ds, &prepared, dims, &coverage_mask(&lookups_of(&prepared), dims, 0.0), &target, &opts).unwrap();
        // Perfect agreement once reindexed → the mean equals each input.
        for (got, want) in agg.sh[..target.ncoeffs].iter().zip(&z) {
            assert!((got - want).abs() < 1e-5, "reference voxel 0 must be the z-lobe");
        }
        let qc = fod_qc(&ds, &prepared, dims, &agg, &opts).unwrap();
        assert!(qc.acc_mean.iter().all(|&a| a > 0.999), "acc {:?}", qc.acc_mean);
    }

    #[test]
    fn rejects_inputs_without_sh() {
        let tmp = TempDir::new().unwrap();
        let mut b = OdxBuilder::new(identity_affine(), [1, 1, 1], vec![1]);
        b.push_voxel_peaks(&[[0.0, 0.0, 1.0]]);
        let p = tmp.path().join("peaks.odx");
        b.finalize().unwrap().save_directory(&p).unwrap();
        let ds = open_all(&[p]).unwrap();
        let l = labels(&["peaks"]);
        let err = ShTarget::from_header(ds[0].header(), &l[0]).unwrap_err();
        assert!(err.to_string().contains("SH_ORDER"), "{err}");
    }

    #[test]
    fn outlier_rule_flags_the_bad_subject_only() {
        // Three tight scans plus one clearly worse.
        let acc = [0.97f32, 0.96, 0.975, 0.40];
        let cov = [1.0f64, 1.0, 1.0, 1.0];
        let r = flag_outliers(&acc, &cov);
        assert!(r[0].is_empty() && r[1].is_empty() && r[2].is_empty(), "{r:?}");
        assert_eq!(r[3].len(), 1, "{r:?}");
        assert!(r[3][0].starts_with("low_acc_loo"));
    }

    /// The absolute floor is what stops a vanishing MAD from flagging good
    /// sessions: eight near-identical scans must produce zero outliers.
    #[test]
    fn outlier_rule_does_not_flag_a_tight_cohort() {
        let acc = [0.981f32, 0.982, 0.980, 0.983, 0.979, 0.981, 0.982, 0.972];
        let cov = [1.0f64; 8];
        let r = flag_outliers(&acc, &cov);
        assert!(r.iter().all(|x| x.is_empty()), "{r:?}");
    }

    #[test]
    fn low_coverage_is_flagged() {
        let acc = [0.98f32, 0.98];
        let cov = [1.0f64, 0.5];
        let r = flag_outliers(&acc, &cov);
        assert!(r[0].is_empty());
        assert_eq!(r[1].len(), 1);
        assert!(r[1][0].starts_with("low_coverage"));
    }

    /// A voxel with no anisotropic energy carries no orientation, so ACC there
    /// is arithmetic on rounding noise. Without a floor it returns ~1/sqrt(n) —
    /// a *finite* value that drags every whole-brain summary toward it.
    #[test]
    fn zero_signal_voxels_give_nan_acc_not_one_over_sqrt_n() {
        let tmp = TempDir::new().unwrap();
        let dims = [1u64, 1, 2];
        // v0: a real lobe. v1: pure denormal noise, as multi-tissue CSD leaves
        // wherever the WM compartment went entirely to GM/CSF.
        let lobe = lobe_sh([0.0, 0.0, 1.0], 8, 8.0);
        let paths: Vec<PathBuf> = (0..8)
            .map(|i| {
                let mut dead = vec![0.0f32; 45];
                // deterministic per-subject noise at the f32 rounding floor
                for (k, x) in dead.iter_mut().enumerate() {
                    *x = ((i * 7 + k * 13) % 17) as f32 * 1e-20;
                }
                build_sh_input(
                    tmp.path(), &format!("s{i}"), identity_affine(), dims, vec![1, 1],
                    vec![lobe.clone(), dead], 8, "tournier07", false,
                )
            })
            .collect();
        let ds = open_all(&paths).unwrap();
        let l: Vec<String> = (0..8).map(|i| format!("s{i}")).collect();
        let (prepared, target) = prep(&ds, &l, dims, &identity_affine(), LmaxPolicy::Min);
        let opts = AggregateOptions { loo: LooMode::On, ..Default::default() };
        let agg = aggregate_fod(
            &ds, &prepared, dims, &coverage_mask(&lookups_of(&prepared), dims, 0.0), &target, &opts,
        )
        .unwrap();
        let qc = fod_qc(&ds, &prepared, dims, &agg, &opts).unwrap();

        assert!((qc.acc_mean[0] - 1.0).abs() < 1e-5, "the real lobe still scores 1");
        assert!(
            qc.acc_mean[1].is_nan(),
            "the zero-signal voxel must be NaN, got {} (1/sqrt(8) = {})",
            qc.acc_mean[1],
            1.0 / 8f32.sqrt()
        );
        assert_eq!(qc.n_voxels_without_orientation, 1, "the count must be reported");
    }

    /// Under `--normalize-fod l0` every scaled row has c0 == 1 by construction,
    /// so the l0 maps must record the PRE-scale value or they are degenerate.
    #[test]
    fn l0_stats_record_the_pre_scale_coefficient() {
        let tmp = TempDir::new().unwrap();
        let mk = |n: &str, c0: f32| {
            let mut r = lobe_sh([0.0, 0.0, 1.0], 8, 8.0);
            r[0] = c0;
            build_sh_input(
                tmp.path(), n, identity_affine(), [1, 1, 1], vec![1], vec![r], 8,
                "tournier07", false,
            )
        };
        let paths = vec![mk("a", 2.0), mk("b", 4.0), mk("c", 6.0)];
        let ds = open_all(&paths).unwrap();
        let l = labels(&["a", "b", "c"]);
        let (prepared, target) = prep(&ds, &l, [1, 1, 1], &identity_affine(), LmaxPolicy::Min);
        let opts = AggregateOptions { scale: ScaleMode::L0, ..Default::default() };
        let agg = aggregate_fod(
            &ds, &prepared, [1, 1, 1],
            &coverage_mask(&lookups_of(&prepared), [1, 1, 1], 0.0), &target, &opts,
        )
        .unwrap();
        let qc = fod_qc(&ds, &prepared, [1, 1, 1], &agg, &opts).unwrap();
        assert!(
            (qc.l0_mean[0] - 4.0).abs() < 1e-5,
            "l0_mean must be the mean of 2/4/6 = 4, got {} (1.0 means the \
             post-scale value leaked in)",
            qc.l0_mean[0]
        );
        assert!(qc.l0_sd[0] > 1.0, "l0_sd must see the spread, got {}", qc.l0_sd[0]);
    }

    /// A symmetric basis stores even bands only, so an odd `acc_lmin` must
    /// resolve to the next even band rather than landing inside a block.
    #[test]
    fn acc_skip_rounds_odd_lmin_up_to_the_next_even_band() {
        let sym = ShTarget {
            order: 8, ncoeffs: 45, basis_name: "tournier07".into(),
            full_basis: false, legacy: false,
        };
        // tournier lmax 8 layout: l=0 at [0], l=2 at [1..6), l=4 at [6..15),
        // l=6 at [15..28), l=8 at [28..45)
        assert_eq!(acc_skip(&sym, 0), 0);
        assert_eq!(acc_skip(&sym, 1), 1, "l>=1 in a symmetric basis means l>=2");
        assert_eq!(acc_skip(&sym, 2), 1);
        assert_eq!(acc_skip(&sym, 3), 6, "l>=3 means l>=4, not the middle of l=2");
        assert_eq!(acc_skip(&sym, 4), 6);
        assert_eq!(acc_skip(&sym, 6), 15);

        let full = ShTarget {
            order: 8, ncoeffs: 81, basis_name: "descoteaux07".into(),
            full_basis: true, legacy: false,
        };
        assert_eq!(acc_skip(&full, 1), 1);
        assert_eq!(acc_skip(&full, 2), 4, "full basis: l=2 starts after 1+3 = 4");
        assert_eq!(acc_skip(&full, 3), 9);
    }

    #[test]
    fn lmax_policy_parses() {
        assert_eq!(LmaxPolicy::parse("min").unwrap(), LmaxPolicy::Min);
        assert_eq!(LmaxPolicy::parse("MAX").unwrap(), LmaxPolicy::Max);
        assert_eq!(LmaxPolicy::parse("6").unwrap(), LmaxPolicy::Fixed(6));
        assert!(LmaxPolicy::parse("7").is_err(), "odd lmax must be rejected");
        assert!(LmaxPolicy::parse("banana").is_err());
    }
}
