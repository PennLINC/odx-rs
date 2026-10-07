//! Fixel-level coherence quality control.
//!
//! The QC pass classifies each stored fixel as thresholded out, disconnected,
//! or connected using only the sparse ODX representation:
//!
//! - pick a scalar primary metric, either explicitly or by trying
//!   `amplitude`, `afd`, then `qa`
//! - threshold that metric with Otsu, positive-only, all-fixels, or a numeric
//!   override
//! - for each remaining fixel, scan the 13 undirected voxel-neighbor offsets
//! - require the source direction to align with the inter-voxel trajectory and
//!   require at least one neighbor fixel to align with the source direction
//!
//! The resulting summary report exposes DSI-Studio style coherence and incoherence
//! indices along with connected/disconnected counts and per-scalar-DPF
//! partition summaries. The full per-fixel class map can also be written back
//! to ODX as `dpf/qc_class.uint8`.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use nalgebra::{Matrix3, Vector3};
use serde::Serialize;

use crate::{DType, DataArray, OdxDataset, OdxError, Result};

pub const QC_CLASS_DPF_NAME: &str = "qc_class";

/// Threshold mode for selecting which fixels participate in QC.
#[derive(Debug, Clone, PartialEq)]
pub enum ThresholdMode {
    Otsu,
    Positive,
    All,
    Value(f32),
    /// Drop the lowest `q` fraction of values (by count) and keep every value
    /// at or above the `q`-quantile. Unlike Otsu, the share evaluated is the
    /// same for every dataset, which keeps coherence comparable across
    /// subjects and grids. Primary mode takes the quantile over voxels that
    /// have a fixel, never over empty grid voxels.
    Quantile(f32),
}

/// `ThresholdMode::Quantile`'s recommended `q` for coherence QC: evaluate the
/// top 90%. Like DSI Studio's 0.6 x whole-grid Otsu, this keeps the cut in
/// the sparse low tail where coherence barely depends on it, but it does not
/// move with the amount of empty field of view.
pub const DEFAULT_QC_QUANTILE: f32 = 0.1;

/// `ThresholdMode::Quantile`'s recommended `q` for `check_btable`: score only
/// the top 10%, the fixels whose directions are reliable.
pub const DEFAULT_BTABLE_QUANTILE: f32 = 0.9;

/// Which sample set to use when computing a fixel-level Otsu threshold.
///
/// The default, `AllFixels`, mirrors `compute_fixel_qc`'s existing
/// behavior: the histogram covers every stored per-fixel value, so
/// smaller-but-real secondary peaks still influence the split. The
/// `PrimaryPeak` mode matches DSI-Studio's `fa_otsu` convention
/// (`fib_data.cpp::set_tracking_index`): project the DPF to one value
/// per voxel by taking the primary peak (`dpf[offsets[voxel]]`) and
/// Otsu that vector instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OtsuScope {
    AllFixels,
    PrimaryPeak,
}

impl Default for OtsuScope {
    fn default() -> Self {
        Self::AllFixels
    }
}

/// Result of `compute_fixel_otsu`: the resolved metric name, scope used,
/// the Otsu threshold in the metric's native units, and the sample
/// count the histogram was built from.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FixelOtsu {
    pub metric_name: String,
    pub scope: OtsuScope,
    pub threshold: f32,
    pub n_values: usize,
}

/// Compute an Otsu threshold over per-fixel scalar values.
///
/// When `metric` is `None`, falls back to the same priority used by
/// `compute_fixel_qc`: `amplitude` → `afd` → `qa`. Errors cleanly if
/// none of those (or the user-named metric) resolve.
///
/// `scope` controls whether the histogram is built from every fixel
/// (`AllFixels`, catches secondary peaks) or only from the primary
/// peak per voxel (`PrimaryPeak`, DSI-Studio parity).
pub fn compute_fixel_otsu(
    odx: &OdxDataset,
    metric: Option<&str>,
    scope: OtsuScope,
) -> Result<FixelOtsu> {
    let primary = resolve_primary_metric(odx, metric)?;
    let values: Vec<f32> = match scope {
        OtsuScope::AllFixels => primary.values,
        OtsuScope::PrimaryPeak => primary_peak_projection(odx, &primary.values)?,
    };
    if values.is_empty() {
        return Err(OdxError::Argument(format!(
            "primary DPF '{}' yielded no samples under scope {:?}",
            primary.name, scope
        )));
    }
    let threshold = otsu_threshold(&values);
    Ok(FixelOtsu {
        metric_name: primary.name,
        scope,
        threshold,
        n_values: values.len(),
    })
}

/// Project a per-fixel DPF vector to a per-voxel vector by taking the
/// primary peak's value at each masked voxel with at least one fixel.
/// Voxels with zero fixels are skipped (no zero-padding bias).
fn primary_peak_projection(odx: &OdxDataset, dpf_values: &[f32]) -> Result<Vec<f32>> {
    primary_peak_projection_from_offsets(odx.offsets(), dpf_values)
}

/// Pure-slice variant of `primary_peak_projection` so we can unit-test
/// the projection logic without constructing an `OdxDataset`.
fn primary_peak_projection_from_offsets(offsets: &[u32], dpf_values: &[f32]) -> Result<Vec<f32>> {
    if offsets.len() <= 1 {
        return Ok(Vec::new());
    }
    let expected = *offsets.last().expect("offsets has at least one element") as usize;
    if dpf_values.len() != expected {
        return Err(OdxError::Argument(format!(
            "DPF vector length {} does not match offsets[last]={}",
            dpf_values.len(),
            expected
        )));
    }
    let mut out = Vec::with_capacity(offsets.len() - 1);
    for window in offsets.windows(2) {
        let (start, end) = (window[0] as usize, window[1] as usize);
        if end > start {
            out.push(dpf_values[start]);
        }
    }
    Ok(out)
}

/// Options controlling fixel coherence QC.
///
/// `primary_metric` must resolve to a scalar nonnegative DPF. When it is not
/// provided, QC tries `amplitude`, `afd`, and `qa` in that order.
///
/// `angle_degrees` is used both for trajectory gating against the voxel-neighbor
/// offset and for direction matching against neighbor fixels.
#[derive(Debug, Clone, PartialEq)]
pub struct FixelQcOptions {
    pub primary_metric: Option<String>,
    pub threshold: ThresholdMode,
    pub angle_degrees: f32,
}

impl Default for FixelQcOptions {
    fn default() -> Self {
        Self {
            primary_metric: None,
            threshold: ThresholdMode::Otsu,
            angle_degrees: 15.0,
        }
    }
}

/// Summary statistics for one side of a connected/disconnected partition.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PartitionValueStats {
    pub count: usize,
    pub mean: Option<f64>,
    pub median: Option<f32>,
}

/// Connected/disconnected summary statistics for one scalar DPF.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PartitionStats {
    pub connected: PartitionValueStats,
    pub disconnected: PartitionValueStats,
}

/// Aggregate fixel QC report.
///
/// `coherence_index` and `incoherence_index` are weighted by the primary metric
/// over evaluated fixels only. `per_dpf` is computed for scalar DPFs other than
/// the reserved `qc_class` output field.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FixelQcReport {
    pub total_fixels: usize,
    pub evaluated_fixels: usize,
    pub excluded_fixels: usize,
    pub connected_fixels: usize,
    pub disconnected_fixels: usize,
    pub connected_to_disconnected_ratio: Option<f64>,
    pub coherence_index: Option<f64>,
    pub incoherence_index: Option<f64>,
    pub primary_metric: String,
    pub threshold_value: Option<f32>,
    pub per_dpf: BTreeMap<String, PartitionStats>,
    pub skipped_dpf: Vec<String>,
}

/// Per-fixel QC class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FixelQcClass {
    ThresholdedOut = 0,
    Disconnected = 1,
    Connected = 2,
}

/// Full QC result: summary report plus one class per fixel.
#[derive(Debug, Clone, PartialEq)]
pub struct FixelQcComputation {
    pub report: FixelQcReport,
    pub classes: Vec<FixelQcClass>,
}

impl FixelQcComputation {
    /// Encode the per-fixel classes as `0/1/2` bytes for on-disk storage.
    pub fn encode_classes_u8(&self) -> Vec<u8> {
        encode_classes_u8(&self.classes)
    }

    /// Build the `qc_class` scalar DPF as an ODX `uint8` array.
    pub fn qc_class_dpf(&self) -> DataArray {
        qc_class_dpf_from_classes(&self.classes)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FixelState {
    Excluded,
    Disconnected,
    Connected,
}

struct PrimaryMetric {
    name: String,
    values: Vec<f32>,
}

/// Compute sparse fixel coherence QC for an `OdxDataset`.
///
/// Theory:
///
/// - only fixels passing the primary-metric threshold are evaluated
/// - a fixel is considered connected when its direction aligns with the
///   trajectory to at least one 13-neighborhood voxel and at least one fixel in
///   that neighbor voxel aligns with the source direction
/// - the angular gate is symmetric and uses `abs(dot(..)) >= cos(angle)`
///
/// Practice:
///
/// - `report` contains headline coherence/incoherence, connected/disconnected
///   counts, and per-scalar-DPF summaries
/// - `classes` contains one `FixelQcClass` per stored fixel
pub fn compute_fixel_qc(odx: &OdxDataset, options: &FixelQcOptions) -> Result<FixelQcComputation> {
    validate_angle(options.angle_degrees)?;

    let primary = resolve_primary_metric(odx, options.primary_metric.as_deref())?;
    let threshold_value = resolve_threshold_value(&primary.values, &options.threshold)?;
    let angular_threshold = options.angle_degrees.to_radians().cos();
    let voxel_lookup = build_voxel_lookup(odx)?;
    let frame = GridFrame::from_header(odx.header())?;

    let mut states = vec![FixelState::Excluded; odx.nb_peaks()];
    for (fixel_idx, &value) in primary.values.iter().enumerate() {
        if should_evaluate(value, &options.threshold, threshold_value) {
            states[fixel_idx] = FixelState::Disconnected;
        }
    }

    classify_fixels(
        odx.directions(),
        odx.offsets(),
        &voxel_lookup,
        &frame,
        angular_threshold,
        &mut states,
    );

    let mut connected_fixels = 0usize;
    let mut disconnected_fixels = 0usize;
    let mut connected_weight = 0.0f64;
    let mut disconnected_weight = 0.0f64;

    for (idx, state) in states.iter().enumerate() {
        match state {
            FixelState::Connected => {
                connected_fixels += 1;
                connected_weight += primary.values[idx] as f64;
            }
            FixelState::Disconnected => {
                disconnected_fixels += 1;
                disconnected_weight += primary.values[idx] as f64;
            }
            FixelState::Excluded => {}
        }
    }

    let evaluated_fixels = connected_fixels + disconnected_fixels;
    let excluded_fixels = states.len() - evaluated_fixels;
    let total_weight = connected_weight + disconnected_weight;
    let (coherence_index, incoherence_index) = if total_weight > 0.0 {
        (
            Some(connected_weight / total_weight),
            Some(disconnected_weight / total_weight),
        )
    } else {
        (None, None)
    };

    let connected_to_disconnected_ratio = if disconnected_fixels > 0 {
        Some(connected_fixels as f64 / disconnected_fixels as f64)
    } else {
        None
    };

    let (per_dpf, skipped_dpf) = summarize_scalar_dpf_partitions(odx, &states)?;

    let classes = states.iter().copied().map(FixelQcClass::from).collect();
    let report = FixelQcReport {
        total_fixels: odx.nb_peaks(),
        evaluated_fixels,
        excluded_fixels,
        connected_fixels,
        disconnected_fixels,
        connected_to_disconnected_ratio,
        coherence_index,
        incoherence_index,
        primary_metric: primary.name,
        threshold_value,
        per_dpf,
        skipped_dpf,
    };

    Ok(FixelQcComputation { report, classes })
}

/// Append or replace `dpf/qc_class.uint8` in an existing ODX directory or
/// `.odx` archive.
///
/// The class vector length must match `NB_PEAKS`. On disk the values are stored
/// as:
///
/// - `0` = thresholded out
/// - `1` = disconnected
/// - `2` = connected
pub fn write_qc_class_dpf(path: &Path, classes: &[FixelQcClass], overwrite: bool) -> Result<()> {
    let dpf = HashMap::from([(
        QC_CLASS_DPF_NAME.to_string(),
        qc_class_dpf_from_classes(classes),
    )]);
    crate::io::append_dpf(path, &dpf, overwrite)
}

/// Primary-fibre coherence index (DSI Studio's `evaluate_fib`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PrimaryCoherenceReport {
    /// Weight of connected voxels over the weight of evaluated voxels.
    pub coherence_index: Option<f64>,
    pub evaluated_voxels: usize,
    pub connected_voxels: usize,
    pub voxels_with_fixels: usize,
    pub primary_metric: String,
    pub threshold_value: Option<f32>,
    pub angle_degrees: f32,
}

/// Compute the primary-fibre coherence index.
///
/// Each voxel contributes only its strongest fixel (by the primary metric).
/// That fixel's direction is taken into voxel-index space and rounded to the
/// lattice step it points along; the voxel is connected when the strongest
/// fixel one step forward or back lies within `angle_degrees` of it. The
/// neighbour need not pass the threshold. The index is the primary-metric
/// weight of connected voxels over that of evaluated voxels.
///
/// This is DSI Studio's fib-QC "coherence index". As in DSI Studio, an Otsu
/// threshold is taken over the whole grid, with voxels that have no fixel
/// counting as zero. Unlike `compute_fixel_qc`, every direction has a
/// neighbour to test, so oblique fibres are not disconnected by construction.
pub fn compute_primary_coherence(
    odx: &OdxDataset,
    options: &FixelQcOptions,
) -> Result<PrimaryCoherenceReport> {
    let prepared = PreparedPrimary::new(odx, options)?;
    let score = prepared.score(&Matrix3::identity());
    Ok(prepared.report(score))
}

/// How strongly the coherence index depends on where the threshold fell.
///
/// Recomputes coherence (by `mode`) at 0.8x and 1.25x the threshold that
/// `options` resolves to and returns d ln(coherence) / d ln(threshold). Near
/// 0 the index is insensitive to the cut; about 0.2 means a 10% threshold
/// shift moves coherence by 2%. `None` when there is no positive threshold or
/// either recomputation has nothing to score.
pub fn coherence_threshold_elasticity(
    odx: &OdxDataset,
    options: &FixelQcOptions,
    mode: CoherenceMode,
) -> Result<Option<f64>> {
    let at = |threshold: ThresholdMode| -> Result<(Option<f32>, Option<f64>)> {
        let options = FixelQcOptions {
            threshold,
            ..options.clone()
        };
        Ok(match mode {
            CoherenceMode::Primary => {
                let r = compute_primary_coherence(odx, &options)?;
                (r.threshold_value, r.coherence_index)
            }
            CoherenceMode::Fixel => {
                let r = compute_fixel_qc(odx, &options)?.report;
                (r.threshold_value, r.coherence_index)
            }
            CoherenceMode::Chain => {
                let r = compute_fixel_chains(odx, &options)?;
                (r.threshold_value, r.weighted_mean_length_mm)
            }
        })
    };
    let Some(t) = at(options.threshold.clone())?.0.filter(|t| *t > 0.0) else {
        return Ok(None);
    };
    let (lo_factor, hi_factor) = (0.8f32, 1.25f32);
    let lo = at(ThresholdMode::Value(t * lo_factor))?.1;
    let hi = at(ThresholdMode::Value(t * hi_factor))?.1;
    Ok(match (lo, hi) {
        (Some(lo), Some(hi)) if lo > 0.0 && hi > 0.0 => {
            Some((hi / lo).ln() / ((hi_factor / lo_factor) as f64).ln())
        }
        _ => None,
    })
}

/// One candidate gradient-table correction and the coherence it yields.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BTableCandidate {
    pub label: String,
    pub coherence_index: Option<f64>,
}

/// Result of `check_btable`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BTableCheck {
    /// The coherence rule the candidates were scored with.
    pub scoring: CoherenceMode,
    /// All 24 axis permutation/flip candidates, the identity (`012`) first.
    pub candidates: Vec<BTableCandidate>,
    /// Label of the most coherent candidate; ties go to the earlier one.
    pub best: String,
    pub current_coherence_index: Option<f64>,
    pub best_coherence_index: Option<f64>,
    /// True when no candidate beats the gradient table as it was used.
    pub current_is_best: bool,
}

/// DSI Studio's 24 b-table candidates. `120fx` means: new (x, y, z) =
/// old (y, z, x), then negate the new x. Two flips equal the third flip
/// under antipodal symmetry, so these cover every axis permutation and flip.
pub const BTABLE_CANDIDATE_LABELS: [&str; 24] = [
    "012", "012fx", "012fy", "012fz", "021", "021fx", "021fy", "021fz", "102", "102fx", "102fy",
    "102fz", "120", "120fx", "120fy", "120fz", "210", "210fx", "210fy", "210fz", "201", "201fx",
    "201fy", "201fz",
];

/// Find the gradient-table permutation/flip that makes the fit most coherent.
///
/// Permuting or flipping the gradient axes of a rotation-equivariant fit (a
/// tensor exactly; SH/SHORE fits with rotation-invariant regularisation
/// nearly so) permutes or flips the fitted directions the same way. So each
/// candidate is scored by transforming the fitted directions instead of
/// refitting. Transforms act on the voxel axes, the frame of FSL/dipy bvec
/// files; a label is the correction to apply to such a file.
///
/// Use `CoherenceMode::Chain` (the CLI and Python default): a wrong table
/// breaks chains almost at once, which separates the candidates far more
/// than coherence does. `CoherenceMode::Fixel` also works, through its
/// trajectory gate (a fixel must point at the neighbour it is compared with).
/// Avoid `Primary`: it only asks whether neighbouring fibres are parallel,
/// and a permuted smooth field is still smooth, so its scores sit close
/// together.
///
/// Threshold strictly, e.g. `ThresholdMode::Quantile(DEFAULT_BTABLE_QUANTILE)`:
/// low-anisotropy fixels point almost at random under every candidate and
/// dilute the difference between the true table and the rest.
pub fn check_btable(
    odx: &OdxDataset,
    options: &FixelQcOptions,
    scoring: CoherenceMode,
) -> Result<BTableCheck> {
    let frame = GridFrame::from_header(odx.header())?;
    let (fixel, primary, chains) = match scoring {
        CoherenceMode::Fixel => (Some(PreparedFixel::new(odx, options)?), None, None),
        CoherenceMode::Primary => (None, Some(PreparedPrimary::new(odx, options)?), None),
        CoherenceMode::Chain => (None, None, Some(PreparedChains::new(odx, options)?)),
    };

    let mut candidates = Vec::with_capacity(BTABLE_CANDIDATE_LABELS.len());
    let mut best: Option<(usize, f64)> = None;
    for (idx, label) in BTABLE_CANDIDATE_LABELS.iter().enumerate() {
        let world = frame.axes * btable_candidate_matrix(label) * frame.axes_inv;
        let index = match (&fixel, &primary, &chains) {
            (Some(fixel), _, _) => fixel.coherence_index(odx, &frame, &world),
            (_, Some(primary), _) => primary.score(&world).coherence_index(),
            (_, _, Some(chains)) => {
                let dirs = transform_directions(odx.directions(), &world);
                chains.run(odx.offsets(), &dirs).weighted_mean_length_mm
            }
            _ => unreachable!(),
        };
        if let Some(value) = index {
            if best.is_none_or(|(_, b)| value > b) {
                best = Some((idx, value));
            }
        }
        candidates.push(BTableCandidate {
            label: (*label).to_string(),
            coherence_index: index,
        });
    }

    let best_idx = best.map_or(0, |(idx, _)| idx);
    Ok(BTableCheck {
        scoring,
        best: BTABLE_CANDIDATE_LABELS[best_idx].to_string(),
        current_coherence_index: candidates[0].coherence_index,
        best_coherence_index: candidates[best_idx].coherence_index,
        current_is_best: best_idx == 0,
        candidates,
    })
}

/// Which coherence rule scores the b-table candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CoherenceMode {
    Fixel,
    Primary,
    /// Fibre-weighted mean chain length (`compute_fixel_chains`).
    Chain,
}

/// Fixel-mode state that does not depend on the directions.
struct PreparedFixel {
    weights: Vec<f32>,
    lookup: VoxelLookup,
    initial: Vec<FixelState>,
    angular_threshold: f32,
}

impl PreparedFixel {
    fn new(odx: &OdxDataset, options: &FixelQcOptions) -> Result<Self> {
        validate_angle(options.angle_degrees)?;
        let primary = resolve_primary_metric(odx, options.primary_metric.as_deref())?;
        let threshold_value = resolve_threshold_value(&primary.values, &options.threshold)?;
        let initial = primary
            .values
            .iter()
            .map(|&v| {
                if should_evaluate(v, &options.threshold, threshold_value) {
                    FixelState::Disconnected
                } else {
                    FixelState::Excluded
                }
            })
            .collect();
        Ok(Self {
            weights: primary.values,
            lookup: build_voxel_lookup(odx)?,
            initial,
            angular_threshold: options.angle_degrees.to_radians().cos(),
        })
    }

    fn coherence_index(
        &self,
        odx: &OdxDataset,
        frame: &GridFrame,
        transform: &Matrix3<f64>,
    ) -> Option<f64> {
        let directions = transform_directions(odx.directions(), transform);
        let mut states = self.initial.clone();
        classify_fixels(
            &directions,
            odx.offsets(),
            &self.lookup,
            frame,
            self.angular_threshold,
            &mut states,
        );
        let (mut connected, mut total) = (0.0f64, 0.0f64);
        for (state, &w) in states.iter().zip(&self.weights) {
            match state {
                FixelState::Connected => {
                    connected += w as f64;
                    total += w as f64;
                }
                FixelState::Disconnected => total += w as f64,
                FixelState::Excluded => {}
            }
        }
        (total > 0.0).then(|| connected / total)
    }
}

/// The voxel-axis matrix for a `BTABLE_CANDIDATE_LABELS` entry.
fn btable_candidate_matrix(label: &str) -> Matrix3<f64> {
    let bytes = label.as_bytes();
    let mut m = Matrix3::zeros();
    for row in 0..3 {
        m[(row, (bytes[row] - b'0') as usize)] = 1.0;
    }
    let flipped = match &label[3..] {
        "fx" => Some(0),
        "fy" => Some(1),
        "fz" => Some(2),
        _ => None,
    };
    if let Some(row) = flipped {
        m.set_row(row, &(-m.row(row)));
    }
    m
}

/// The grid geometry QC needs from `VOXEL_TO_RASMM`.
struct GridFrame {
    /// Voxel index -> world (mm), and its inverse.
    linear: Matrix3<f64>,
    linear_inv: Matrix3<f64>,
    /// Unit voxel axes in world space (affine columns, normalised), and its
    /// inverse: the frame of FSL/dipy bvecs.
    axes: Matrix3<f64>,
    axes_inv: Matrix3<f64>,
    /// Unit world direction of each `neighbor_offsets()` entry.
    offset_units: [[f32; 3]; 13],
}

impl GridFrame {
    fn from_header(header: &crate::Header) -> Result<Self> {
        let a = header.voxel_to_rasmm;
        let linear = Matrix3::new(
            a[0][0], a[0][1], a[0][2], a[1][0], a[1][1], a[1][2], a[2][0], a[2][1], a[2][2],
        );
        let singular = || OdxError::Format("VOXEL_TO_RASMM has a singular linear part".into());
        let linear_inv = linear.try_inverse().ok_or_else(singular)?;
        let mut axes = linear;
        for mut col in axes.column_iter_mut() {
            let norm = col.norm();
            col /= norm;
        }
        let axes_inv = axes.try_inverse().ok_or_else(singular)?;

        let mut offset_units = [[0.0f32; 3]; 13];
        for (unit, [dx, dy, dz]) in offset_units.iter_mut().zip(neighbor_offsets()) {
            let step = (linear * Vector3::new(dx as f64, dy as f64, dz as f64)).normalize();
            *unit = [step.x as f32, step.y as f32, step.z as f32];
        }
        Ok(Self {
            linear,
            linear_inv,
            axes,
            axes_inv,
            offset_units,
        })
    }
}

/// Every grid voxel's strongest fixel, ready to score under any transform.
struct PreparedPrimary {
    frame: GridFrame,
    dims: [usize; 3],
    /// Primary-metric value of the strongest fixel; 0 where there is none.
    weight: Vec<f32>,
    /// World direction of that fixel; zero where there is none.
    dir: Vec<Vector3<f64>>,
    voxels_with_fixels: usize,
    metric_name: String,
    threshold: ThresholdMode,
    threshold_value: Option<f32>,
    angle_degrees: f32,
    cos_threshold: f64,
}

#[derive(Default)]
struct PrimaryScore {
    connected_weight: f64,
    total_weight: f64,
    evaluated: usize,
    connected: usize,
}

impl PrimaryScore {
    fn coherence_index(&self) -> Option<f64> {
        (self.total_weight > 0.0).then(|| self.connected_weight / self.total_weight)
    }
}

impl PreparedPrimary {
    fn new(odx: &OdxDataset, options: &FixelQcOptions) -> Result<Self> {
        validate_angle(options.angle_degrees)?;
        let primary = resolve_primary_metric(odx, options.primary_metric.as_deref())?;
        let frame = GridFrame::from_header(odx.header())?;
        let lookup = build_voxel_lookup(odx)?;
        let dims = lookup.dims;
        let n_grid = dims[0] * dims[1] * dims[2];
        let directions = odx.directions();
        let offsets = odx.offsets();

        let mut weight = vec![0.0f32; n_grid];
        let mut dir = vec![Vector3::zeros(); n_grid];
        let mut voxels_with_fixels = 0usize;
        for (voxel, &[x, y, z]) in lookup.masked_coords.iter().enumerate() {
            let (start, end) = (offsets[voxel] as usize, offsets[voxel + 1] as usize);
            let Some(strongest) = (start..end).reduce(|best, idx| {
                if primary.values[idx] > primary.values[best] {
                    idx
                } else {
                    best
                }
            }) else {
                continue;
            };
            let flat = (x as usize * dims[1] + y as usize) * dims[2] + z as usize;
            let d = directions[strongest];
            weight[flat] = primary.values[strongest];
            dir[flat] = Vector3::new(d[0] as f64, d[1] as f64, d[2] as f64);
            voxels_with_fixels += 1;
        }

        let threshold_value = match options.threshold {
            // Over the whole grid, empty voxels included, as DSI Studio does.
            ThresholdMode::Otsu => Some(otsu_threshold(&weight)),
            ThresholdMode::Quantile(_) => {
                let occupied: Vec<f32> = weight
                    .iter()
                    .zip(&dir)
                    .filter(|(_, d)| d.norm_squared() > 0.0)
                    .map(|(w, _)| *w)
                    .collect();
                resolve_threshold_value(&occupied, &options.threshold)?
            }
            ref other => resolve_threshold_value(&weight, other)?,
        };
        Ok(Self {
            frame,
            dims,
            weight,
            dir,
            voxels_with_fixels,
            metric_name: primary.name,
            threshold: options.threshold.clone(),
            threshold_value,
            angle_degrees: options.angle_degrees,
            cos_threshold: (options.angle_degrees as f64).to_radians().cos(),
        })
    }

    /// Score with every direction mapped through the world-frame `transform`.
    fn score(&self, transform: &Matrix3<f64>) -> PrimaryScore {
        let dir: Vec<Vector3<f64>> = self.dir.iter().map(|d| transform * d).collect();
        let [nx, ny, nz] = self.dims.map(|n| n as i64);
        let mut score = PrimaryScore::default();
        for (flat, d) in dir.iter().enumerate() {
            let w = self.weight[flat];
            if d.norm_squared() == 0.0 || !should_evaluate(w, &self.threshold, self.threshold_value)
            {
                continue;
            }
            let step = (self.frame.linear_inv * d).normalize().map(f64::round);
            let (x, y, z) = (
                flat as i64 / (ny * nz),
                (flat as i64 / nz) % ny,
                flat as i64 % nz,
            );
            let connected = [1.0, -1.0].iter().any(|sign| {
                let (qx, qy, qz) = (
                    x + (sign * step.x) as i64,
                    y + (sign * step.y) as i64,
                    z + (sign * step.z) as i64,
                );
                if qx < 0 || qy < 0 || qz < 0 || qx >= nx || qy >= ny || qz >= nz {
                    return false;
                }
                let other = &dir[((qx * ny + qy) * nz + qz) as usize];
                let norms = d.norm() * other.norm();
                norms > 0.0 && d.dot(other).abs() >= self.cos_threshold * norms
            });
            score.evaluated += 1;
            score.total_weight += w as f64;
            if connected {
                score.connected += 1;
                score.connected_weight += w as f64;
            }
        }
        score
    }

    fn report(&self, score: PrimaryScore) -> PrimaryCoherenceReport {
        PrimaryCoherenceReport {
            coherence_index: score.coherence_index(),
            evaluated_voxels: score.evaluated,
            connected_voxels: score.connected,
            voxels_with_fixels: self.voxels_with_fixels,
            primary_metric: self.metric_name.clone(),
            threshold_value: self.threshold_value,
            angle_degrees: self.angle_degrees,
        }
    }
}

fn transform_directions(directions: &[[f32; 3]], transform: &Matrix3<f64>) -> Vec<[f32; 3]> {
    directions
        .iter()
        .map(|d| {
            let v = transform * Vector3::new(d[0] as f64, d[1] as f64, d[2] as f64);
            [v.x as f32, v.y as f32, v.z as f32]
        })
        .collect()
}

/// Fixel chain-length statistics (`compute_fixel_chains`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FixelChainReport {
    pub evaluated_fixels: usize,
    pub chains: usize,
    /// Chains that close on themselves (every fixel has two links).
    pub loops: usize,
    /// Length of the chain each fixel belongs to, averaged with the primary
    /// metric as weight: the length a typical stretch of fibre sits in.
    /// (Medians are not reported: most chains are a fixel or two long, so
    /// they sit at 0-2 mm whatever the data.)
    pub weighted_mean_length_mm: Option<f64>,
    pub max_chain_length_mm: Option<f64>,
    /// Share of evaluated weight in chains at least 20 / 40 mm long.
    pub weight_in_chains_over_20mm: Option<f64>,
    pub weight_in_chains_over_40mm: Option<f64>,
    pub primary_metric: String,
    pub threshold_value: Option<f32>,
    pub angle_degrees: f32,
}

/// Link fixels into chains and report their lengths.
///
/// Each evaluated fixel looks forward and backward along its own direction
/// into the neighbouring voxels whose step lies within 35 degrees of it (the
/// 26 lattice steps leave no gap that wide, so every direction has one). On
/// each side it picks the single fixel within `angle_degrees` of it that best
/// combines alignment with it and with the step. A link is kept only when it
/// is mutual, so every fixel has at most one link per side and the fixels
/// split into simple chains and loops. Lengths are summed inter-voxel
/// distances in mm through the affine.
///
/// It is deterministic tractography without seeds, steps or interpolation:
/// a wrong gradient table, misregistration or noise breaks chains, and one
/// bad step breaks a whole chain, so it is a longer-range check than the
/// immediate-neighbour coherence index. Chains are much shorter than
/// streamlines; compare them with each other, not with tractography.
pub fn compute_fixel_chains(
    odx: &OdxDataset,
    options: &FixelQcOptions,
) -> Result<FixelChainReport> {
    let prepared = PreparedChains::new(odx, options)?;
    Ok(prepared.run(odx.offsets(), odx.directions()))
}

/// Half-angle of the cone of lattice steps a fixel may continue into.
const CHAIN_STEP_CONE_DEG: f64 = 35.0;

struct PreparedChains {
    lookup: VoxelLookup,
    /// The 26 lattice steps: index offset, unit world direction, length (mm).
    steps: Vec<([i32; 3], Vector3<f64>, f64)>,
    weights: Vec<f32>,
    evaluated: Vec<bool>,
    cos_match: f64,
    cos_cone: f64,
    metric_name: String,
    threshold_value: Option<f32>,
    angle_degrees: f32,
}

/// Link from one side of a fixel: (other fixel, the other fixel's side).
type ChainLink = Option<(usize, usize)>;

impl PreparedChains {
    fn new(odx: &OdxDataset, options: &FixelQcOptions) -> Result<Self> {
        validate_angle(options.angle_degrees)?;
        let primary = resolve_primary_metric(odx, options.primary_metric.as_deref())?;
        let threshold_value = resolve_threshold_value(&primary.values, &options.threshold)?;
        let evaluated = primary
            .values
            .iter()
            .map(|&v| should_evaluate(v, &options.threshold, threshold_value))
            .collect();
        let frame = GridFrame::from_header(odx.header())?;
        let mut steps = Vec::with_capacity(26);
        for dx in -1..=1i32 {
            for dy in -1..=1i32 {
                for dz in -1..=1i32 {
                    if (dx, dy, dz) == (0, 0, 0) {
                        continue;
                    }
                    let w = frame.linear * Vector3::new(dx as f64, dy as f64, dz as f64);
                    steps.push(([dx, dy, dz], w.normalize(), w.norm()));
                }
            }
        }
        Ok(Self {
            lookup: build_voxel_lookup(odx)?,
            steps,
            weights: primary.values,
            evaluated,
            cos_match: (options.angle_degrees as f64).to_radians().cos(),
            cos_cone: CHAIN_STEP_CONE_DEG.to_radians().cos(),
            metric_name: primary.name,
            threshold_value,
            angle_degrees: options.angle_degrees,
        })
    }

    /// Fixel -> voxel index (into `lookup.masked_coords`).
    fn fixel_voxels(offsets: &[u32]) -> Vec<usize> {
        let mut out = vec![0usize; *offsets.last().unwrap_or(&0) as usize];
        for (voxel, w) in offsets.windows(2).enumerate() {
            out[w[0] as usize..w[1] as usize].fill(voxel);
        }
        out
    }

    fn unit(d: [f32; 3]) -> Vector3<f64> {
        let v = Vector3::new(d[0] as f64, d[1] as f64, d[2] as f64);
        let n = v.norm();
        if n > 0.0 {
            v / n
        } else {
            v
        }
    }

    /// Best continuation from `fixel` on `side` (0: along its direction, 1: against).
    fn best_link(
        &self,
        offsets: &[u32],
        directions: &[[f32; 3]],
        voxel: usize,
        fixel: usize,
        side: usize,
    ) -> (ChainLink, f64) {
        let d = Self::unit(directions[fixel]);
        let travel = if side == 0 { d } else { -d };
        let [x, y, z] = self.lookup.masked_coords[voxel];
        let dims = self.lookup.dims;
        let mut best: (ChainLink, f64, f64) = (None, f64::NEG_INFINITY, 0.0);
        for (step, unit, length) in &self.steps {
            let along = unit.dot(&travel);
            if along < self.cos_cone {
                continue;
            }
            let (qx, qy, qz) = (x + step[0], y + step[1], z + step[2]);
            if qx < 0 || qy < 0 || qz < 0 {
                continue;
            }
            let (qx, qy, qz) = (qx as usize, qy as usize, qz as usize);
            if qx >= dims[0] || qy >= dims[1] || qz >= dims[2] {
                continue;
            }
            let other_voxel = self.lookup.full_to_masked[(qx * dims[1] + qy) * dims[2] + qz];
            if other_voxel == usize::MAX {
                continue;
            }
            for g in offsets[other_voxel] as usize..offsets[other_voxel + 1] as usize {
                if !self.evaluated[g] {
                    continue;
                }
                let c = travel.dot(&Self::unit(directions[g]));
                if c.abs() < self.cos_match {
                    continue;
                }
                let score = c.abs() * along;
                if score > best.1 {
                    // g continues along sign(c) * g, so the side facing back
                    // towards this fixel is the opposite one.
                    let side_g = if c > 0.0 { 1 } else { 0 };
                    best = (Some((g, side_g)), score, *length);
                }
            }
        }
        (best.0, best.2)
    }

    fn run(&self, offsets: &[u32], directions: &[[f32; 3]]) -> FixelChainReport {
        let n = directions.len();
        let voxel_of = Self::fixel_voxels(offsets);
        let mut wants: Vec<[(ChainLink, f64); 2]> = vec![[(None, 0.0); 2]; n];
        for f in 0..n {
            if self.evaluated[f] {
                for side in 0..2 {
                    wants[f][side] = self.best_link(offsets, directions, voxel_of[f], f, side);
                }
            }
        }
        // Keep mutual links: links[f][side] = (other fixel, length mm).
        let mut links: Vec<[Option<(usize, f64)>; 2]> = vec![[None; 2]; n];
        for f in 0..n {
            for side in 0..2 {
                if let (Some((g, side_g)), length) = wants[f][side] {
                    if wants[g][side_g].0 == Some((f, side)) {
                        links[f][side] = Some((g, length));
                    }
                }
            }
        }

        // Walk components (max degree 2: paths and loops).
        let mut component_of = vec![usize::MAX; n];
        let mut lengths: Vec<f64> = Vec::new();
        let mut loops = 0usize;
        let mut stack = Vec::new();
        for start in 0..n {
            if !self.evaluated[start] || component_of[start] != usize::MAX {
                continue;
            }
            let id = lengths.len();
            let (mut half_length, mut all_two) = (0.0f64, true);
            stack.push(start);
            component_of[start] = id;
            while let Some(f) = stack.pop() {
                let mut degree = 0;
                for (g, length) in links[f].iter().flatten() {
                    degree += 1;
                    half_length += length;
                    if component_of[*g] == usize::MAX {
                        component_of[*g] = id;
                        stack.push(*g);
                    }
                }
                all_two &= degree == 2;
            }
            if all_two {
                loops += 1;
            }
            // Each link was counted from both of its ends.
            lengths.push(half_length / 2.0);
        }

        let per_fixel: Vec<(f64, f64)> = (0..n)
            .filter(|&f| self.evaluated[f])
            .map(|f| (lengths[component_of[f]], self.weights[f] as f64))
            .collect();
        let total_weight: f64 = per_fixel.iter().map(|(_, w)| w).sum();
        let weighted = |pred: &dyn Fn(f64) -> bool| -> Option<f64> {
            (total_weight > 0.0).then(|| {
                // Fold from +0.0: an empty f64 `sum()` is -0.0.
                per_fixel
                    .iter()
                    .filter(|(l, _)| pred(*l))
                    .fold(0.0, |acc, (_, w)| acc + w)
                    / total_weight
            })
        };
        let weighted_mean = (total_weight > 0.0)
            .then(|| per_fixel.iter().map(|(l, w)| l * w).sum::<f64>() / total_weight);
        let over_20 = weighted(&|l| l >= 20.0);
        let over_40 = weighted(&|l| l >= 40.0);
        FixelChainReport {
            evaluated_fixels: per_fixel.len(),
            chains: lengths.len(),
            loops,
            weighted_mean_length_mm: weighted_mean,
            max_chain_length_mm: lengths.iter().copied().reduce(f64::max),
            weight_in_chains_over_20mm: over_20,
            weight_in_chains_over_40mm: over_40,
            primary_metric: self.metric_name.clone(),
            threshold_value: self.threshold_value,
            angle_degrees: self.angle_degrees,
        }
    }
}

fn validate_angle(angle_degrees: f32) -> Result<()> {
    if !angle_degrees.is_finite() || !(0.0..=90.0).contains(&angle_degrees) {
        return Err(OdxError::Argument(format!(
            "angle_degrees must be finite and within [0, 90], found {angle_degrees}"
        )));
    }
    Ok(())
}

fn resolve_primary_metric(odx: &OdxDataset, requested: Option<&str>) -> Result<PrimaryMetric> {
    match requested {
        Some(QC_CLASS_DPF_NAME) => Err(OdxError::Argument(format!(
            "'{QC_CLASS_DPF_NAME}' is a reserved QC classification DPF and cannot be used as the primary metric"
        ))),
        Some(name) => load_primary_metric(odx, name)?.ok_or_else(|| {
            OdxError::Argument(format!("requested primary DPF '{name}' does not exist"))
        }),
        None => {
            let mut reasons = Vec::new();
            for candidate in ["amplitude", "afd", "qa"] {
                match load_primary_metric(odx, candidate) {
                    Ok(Some(metric)) => return Ok(metric),
                    Ok(None) => {}
                    Err(err) => reasons.push(format!("{candidate}: {err}")),
                }
            }

            let mut message =
                "no usable primary DPF metric found; tried amplitude, afd, qa".to_string();
            if !reasons.is_empty() {
                message.push_str(" (");
                message.push_str(&reasons.join("; "));
                message.push(')');
            }
            Err(OdxError::Argument(message))
        }
    }
}

fn load_primary_metric(odx: &OdxDataset, name: &str) -> Result<Option<PrimaryMetric>> {
    let Some(arr) = odx.dpf_arrays().get(name) else {
        return Ok(None);
    };
    if arr.ncols() != 1 {
        return Err(OdxError::Argument(format!(
            "primary DPF '{name}' has {} columns; expected a scalar field",
            arr.ncols()
        )));
    }

    let values = arr.to_f32_vec().map_err(|err| match err {
        OdxError::DType(_) => OdxError::DType(format!(
            "primary DPF '{name}' uses unsupported scalar dtype {}",
            arr.dtype()
        )),
        other => other,
    })?;

    for (idx, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(OdxError::Argument(format!(
                "primary DPF '{name}' contains a non-finite value at row {idx}"
            )));
        }
        if *value < 0.0 {
            return Err(OdxError::Argument(format!(
                "primary DPF '{name}' contains a negative value at row {idx}"
            )));
        }
    }

    Ok(Some(PrimaryMetric {
        name: name.to_string(),
        values,
    }))
}

fn resolve_threshold_value(values: &[f32], threshold: &ThresholdMode) -> Result<Option<f32>> {
    match threshold {
        ThresholdMode::All => Ok(None),
        ThresholdMode::Positive => Ok(Some(0.0)),
        ThresholdMode::Otsu => Ok(Some(otsu_threshold(values))),
        ThresholdMode::Value(value) => {
            if !value.is_finite() {
                return Err(OdxError::Argument(
                    "numeric threshold override must be finite".into(),
                ));
            }
            Ok(Some(*value))
        }
        ThresholdMode::Quantile(q) => {
            if !(0.0..1.0).contains(q) {
                return Err(OdxError::Argument(format!(
                    "threshold quantile must be within [0, 1), found {q}"
                )));
            }
            if values.is_empty() {
                return Ok(None);
            }
            let k = ((*q as f64) * values.len() as f64).floor() as usize;
            let mut sorted = values.to_vec();
            let (_, kth, _) = sorted.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
            Ok(Some(*kth))
        }
    }
}

fn should_evaluate(value: f32, threshold: &ThresholdMode, threshold_value: Option<f32>) -> bool {
    match threshold {
        ThresholdMode::All => true,
        ThresholdMode::Positive | ThresholdMode::Otsu | ThresholdMode::Value(_) => {
            value > threshold_value.unwrap_or(0.0)
        }
        ThresholdMode::Quantile(_) => threshold_value.is_some_and(|t| value >= t),
    }
}

struct VoxelLookup {
    dims: [usize; 3],
    masked_coords: Vec<[i32; 3]>,
    full_to_masked: Vec<usize>,
}

fn build_voxel_lookup(odx: &OdxDataset) -> Result<VoxelLookup> {
    let dims = [
        usize::try_from(odx.header().dimensions[0]).map_err(|_| {
            OdxError::Format("x dimension does not fit into usize for QC lookup".into())
        })?,
        usize::try_from(odx.header().dimensions[1]).map_err(|_| {
            OdxError::Format("y dimension does not fit into usize for QC lookup".into())
        })?,
        usize::try_from(odx.header().dimensions[2]).map_err(|_| {
            OdxError::Format("z dimension does not fit into usize for QC lookup".into())
        })?,
    ];

    let yz = dims[1] * dims[2];
    let mut masked_coords = Vec::with_capacity(odx.nb_voxels());
    let mut full_to_masked = vec![usize::MAX; odx.mask().len()];
    let mut masked_index = 0usize;

    for (flat_idx, &mask_value) in odx.mask().iter().enumerate() {
        if mask_value == 0 {
            continue;
        }
        let x = flat_idx / yz;
        let yz_offset = flat_idx % yz;
        let y = yz_offset / dims[2];
        let z = yz_offset % dims[2];
        masked_coords.push([x as i32, y as i32, z as i32]);
        full_to_masked[flat_idx] = masked_index;
        masked_index += 1;
    }

    if masked_coords.len() != odx.nb_voxels() {
        return Err(OdxError::Format(format!(
            "mask contains {} voxels but NB_VOXELS is {}",
            masked_coords.len(),
            odx.nb_voxels()
        )));
    }

    Ok(VoxelLookup {
        dims,
        masked_coords,
        full_to_masked,
    })
}

fn classify_fixels(
    directions: &[[f32; 3]],
    offsets: &[u32],
    voxel_lookup: &VoxelLookup,
    frame: &GridFrame,
    angular_threshold: f32,
    states: &mut [FixelState],
) {
    for (src_voxel, &src_xyz) in voxel_lookup.masked_coords.iter().enumerate() {
        let src_start = offsets[src_voxel] as usize;
        let src_end = offsets[src_voxel + 1] as usize;
        if src_start == src_end {
            continue;
        }

        for (offset_idx, [dx, dy, dz]) in neighbor_offsets().into_iter().enumerate() {
            let nx = src_xyz[0] + dx;
            let ny = src_xyz[1] + dy;
            let nz = src_xyz[2] + dz;
            if nx < 0
                || ny < 0
                || nz < 0
                || nx >= voxel_lookup.dims[0] as i32
                || ny >= voxel_lookup.dims[1] as i32
                || nz >= voxel_lookup.dims[2] as i32
            {
                continue;
            }

            let dst_flat = (nx as usize * voxel_lookup.dims[1] * voxel_lookup.dims[2])
                + (ny as usize * voxel_lookup.dims[2])
                + nz as usize;
            let dst_voxel = voxel_lookup.full_to_masked[dst_flat];
            if dst_voxel == usize::MAX {
                continue;
            }

            let dst_start = offsets[dst_voxel] as usize;
            let dst_end = offsets[dst_voxel + 1] as usize;
            if dst_start == dst_end {
                continue;
            }

            // Directions are world RAS, so the trajectory must be too: on an
            // LPS, oblique or anisotropic grid the index offset is not the
            // physical step.
            let offset_unit = frame.offset_units[offset_idx];

            connect_range_to_neighbor(
                directions,
                states,
                src_start..src_end,
                dst_start..dst_end,
                offset_unit,
                angular_threshold,
            );
            connect_range_to_neighbor(
                directions,
                states,
                dst_start..dst_end,
                src_start..src_end,
                offset_unit,
                angular_threshold,
            );
        }
    }
}

fn connect_range_to_neighbor(
    directions: &[[f32; 3]],
    states: &mut [FixelState],
    source_range: std::ops::Range<usize>,
    neighbor_range: std::ops::Range<usize>,
    offset_unit: [f32; 3],
    angular_threshold: f32,
) {
    for source_idx in source_range {
        if states[source_idx] != FixelState::Disconnected {
            continue;
        }

        let source_dir = directions[source_idx];
        if abs_dot(source_dir, offset_unit) < angular_threshold {
            continue;
        }

        let mut matched = false;
        for neighbor_idx in neighbor_range.clone() {
            if states[neighbor_idx] == FixelState::Excluded {
                continue;
            }

            if abs_dot(source_dir, directions[neighbor_idx]) >= angular_threshold {
                matched = true;
                break;
            }
        }

        if matched {
            states[source_idx] = FixelState::Connected;
        }
    }
}

fn summarize_scalar_dpf_partitions(
    odx: &OdxDataset,
    states: &[FixelState],
) -> Result<(BTreeMap<String, PartitionStats>, Vec<String>)> {
    let mut per_dpf = BTreeMap::new();
    let mut skipped_dpf = Vec::new();

    let mut names = odx.dpf_names();
    names.sort_unstable();

    for name in names {
        if name == QC_CLASS_DPF_NAME {
            continue;
        }

        let arr = odx
            .dpf_arrays()
            .get(name)
            .ok_or_else(|| OdxError::Argument(format!("no DPF named '{name}'")))?;
        if arr.ncols() != 1 {
            skipped_dpf.push(name.to_string());
            continue;
        }

        let values = arr.to_f32_vec().map_err(|err| match err {
            OdxError::DType(_) => OdxError::DType(format!(
                "DPF '{name}' uses unsupported scalar dtype {}",
                arr.dtype()
            )),
            other => other,
        })?;

        let mut connected = Vec::new();
        let mut disconnected = Vec::new();
        let mut connected_sum = 0.0f64;
        let mut disconnected_sum = 0.0f64;

        for (idx, value) in values.iter().enumerate() {
            // NaN in an *auxiliary* DPF means "undefined for this fixel", which
            // is normal — `odx compare` writes NaN for unmatched fixels and
            // `odx combine` for group fixels with too few contributors. Skip
            // those rather than refusing the file; only the primary metric,
            // which drives thresholding, must be finite throughout.
            if !value.is_finite() {
                continue;
            }

            match states[idx] {
                FixelState::Connected => {
                    connected.push(*value);
                    connected_sum += *value as f64;
                }
                FixelState::Disconnected => {
                    disconnected.push(*value);
                    disconnected_sum += *value as f64;
                }
                FixelState::Excluded => {}
            }
        }

        per_dpf.insert(
            name.to_string(),
            PartitionStats {
                connected: build_partition_value_stats(connected, connected_sum),
                disconnected: build_partition_value_stats(disconnected, disconnected_sum),
            },
        );
    }

    Ok((per_dpf, skipped_dpf))
}

fn encode_classes_u8(classes: &[FixelQcClass]) -> Vec<u8> {
    classes.iter().map(|class| *class as u8).collect()
}

fn qc_class_dpf_from_classes(classes: &[FixelQcClass]) -> DataArray {
    DataArray::owned_bytes(encode_classes_u8(classes), 1, DType::UInt8)
}

impl From<FixelState> for FixelQcClass {
    fn from(value: FixelState) -> Self {
        match value {
            FixelState::Excluded => Self::ThresholdedOut,
            FixelState::Disconnected => Self::Disconnected,
            FixelState::Connected => Self::Connected,
        }
    }
}

fn build_partition_value_stats(values: Vec<f32>, sum: f64) -> PartitionValueStats {
    let count = values.len();
    PartitionValueStats {
        count,
        mean: if count > 0 {
            Some(sum / count as f64)
        } else {
            None
        },
        median: median(values),
    }
}

fn median(mut values: Vec<f32>) -> Option<f32> {
    if values.is_empty() {
        return None;
    }

    let len = values.len();
    let mid = len / 2;
    let upper = {
        let (_, upper, _) = values.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
        *upper
    };
    if len % 2 == 1 {
        return Some(upper);
    }

    let lower = values[..mid]
        .iter()
        .copied()
        .max_by(|a, b| a.total_cmp(b))
        .unwrap();
    Some((lower + upper) * 0.5)
}

fn abs_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).abs()
}

fn neighbor_offsets() -> [[i32; 3]; 13] {
    let mut offsets = [[0i32; 3]; 13];
    let mut write = 0usize;
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                if dx > 0 || (dx == 0 && dy > 0) || (dx == 0 && dy == 0 && dz > 0) {
                    offsets[write] = [dx, dy, dz];
                    write += 1;
                }
            }
        }
    }
    offsets
}

/// Classical Otsu (max between-class variance) threshold over a sample
/// vector. 256-bin histogram, robust to degenerate / empty input (returns
/// `0.0` for empty, `min_value.max(0.0)` for all-equal). Public so other
/// crates can reuse the same implementation (e.g. trxviz-core drives it
/// through `compute_fixel_otsu`).
pub fn otsu_threshold(values: &[f32]) -> f32 {
    const BINS: usize = 256;

    if values.is_empty() {
        return 0.0;
    }

    let mut min_value = f32::INFINITY;
    let mut max_value = f32::NEG_INFINITY;
    for &value in values {
        min_value = min_value.min(value);
        max_value = max_value.max(value);
    }

    if !min_value.is_finite() || !max_value.is_finite() || min_value >= max_value {
        return min_value.max(0.0);
    }

    let range = max_value - min_value;
    let mut hist = [0usize; BINS];
    for &value in values {
        let scaled = ((value - min_value) / range * (BINS as f32 - 1.0)).round();
        let idx = scaled.clamp(0.0, BINS as f32 - 1.0) as usize;
        hist[idx] += 1;
    }

    let total = values.len() as f64;
    let mut sum_total = 0.0f64;
    for (idx, &count) in hist.iter().enumerate() {
        sum_total += idx as f64 * count as f64;
    }

    let mut sum_background = 0.0f64;
    let mut weight_background = 0.0f64;
    let mut best_bin = 0usize;
    let mut best_score = f64::NEG_INFINITY;

    for (idx, &count) in hist.iter().enumerate() {
        weight_background += count as f64;
        if weight_background == 0.0 {
            continue;
        }

        let weight_foreground = total - weight_background;
        if weight_foreground == 0.0 {
            break;
        }

        sum_background += idx as f64 * count as f64;
        let mean_background = sum_background / weight_background;
        let mean_foreground = (sum_total - sum_background) / weight_foreground;
        let score =
            weight_background * weight_foreground * (mean_background - mean_foreground).powi(2);
        if score > best_score {
            best_score = score;
            best_bin = idx;
        }
    }

    min_value + range * (best_bin as f32 / (BINS as f32 - 1.0))
}

#[cfg(test)]
mod tests {
    use super::neighbor_offsets;
    use super::otsu_threshold;

    #[test]
    fn otsu_returns_zero_for_empty_input() {
        assert_eq!(otsu_threshold(&[]), 0.0);
    }

    #[test]
    fn otsu_handles_degenerate_input() {
        assert_eq!(otsu_threshold(&[2.5, 2.5, 2.5]), 2.5);
    }

    #[test]
    fn primary_peak_projection_keeps_first_fixel_per_voxel() {
        // offsets=[0, 2, 2, 5] → voxel 0: fixels 0..2, voxel 1: empty, voxel 2: fixels 2..5.
        // Expected primary-peak vector: [dpf[0], dpf[2]] (voxel 1 skipped).
        use super::primary_peak_projection_from_offsets;
        let offsets = vec![0u32, 2, 2, 5];
        let dpf = vec![0.9f32, 0.1, 0.3, 0.2, 0.1];
        let projection = primary_peak_projection_from_offsets(&offsets, &dpf).unwrap();
        assert_eq!(projection, vec![0.9, 0.3]);
    }

    #[test]
    fn primary_peak_projection_rejects_mismatched_length() {
        use super::primary_peak_projection_from_offsets;
        let offsets = vec![0u32, 2, 5];
        let dpf = vec![0.9f32, 0.1]; // last offset says 5, got 2
        assert!(primary_peak_projection_from_offsets(&offsets, &dpf).is_err());
    }

    #[test]
    fn otsu_splits_bimodal_values_between_modes() {
        let threshold = otsu_threshold(&[0.0, 0.0, 0.1, 0.1, 1.0, 1.0, 1.1, 1.1]);
        assert!(
            threshold > 0.05,
            "threshold {threshold} should move above the low-valued cluster"
        );
        assert!(
            threshold < 1.0,
            "threshold {threshold} should stay below the high mode"
        );
    }

    #[test]
    fn neighbor_offset_set_contains_thirteen_unique_offsets() {
        let offsets = neighbor_offsets();
        assert_eq!(offsets.len(), 13);
        for offset in offsets {
            assert_ne!(offset, [0, 0, 0]);
        }
    }
}
