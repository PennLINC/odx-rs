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
    if !options.angle_degrees.is_finite()
        || options.angle_degrees < 0.0
        || options.angle_degrees > 90.0
    {
        return Err(OdxError::Argument(format!(
            "angle_degrees must be finite and within [0, 90], found {}",
            options.angle_degrees
        )));
    }

    let primary = resolve_primary_metric(odx, options.primary_metric.as_deref())?;
    let threshold_value = resolve_threshold_value(&primary.values, &options.threshold)?;
    let angular_threshold = options.angle_degrees.to_radians().cos();
    let voxel_lookup = build_voxel_lookup(odx)?;

    let mut states = vec![FixelState::Excluded; odx.nb_peaks()];
    for (fixel_idx, &value) in primary.values.iter().enumerate() {
        if should_evaluate(value, &options.threshold, threshold_value) {
            states[fixel_idx] = FixelState::Disconnected;
        }
    }

    classify_fixels(odx, &voxel_lookup, angular_threshold, &mut states);

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
    }
}

fn should_evaluate(value: f32, threshold: &ThresholdMode, threshold_value: Option<f32>) -> bool {
    match threshold {
        ThresholdMode::All => true,
        ThresholdMode::Positive | ThresholdMode::Otsu | ThresholdMode::Value(_) => {
            value > threshold_value.unwrap_or(0.0)
        }
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
    odx: &OdxDataset,
    voxel_lookup: &VoxelLookup,
    angular_threshold: f32,
    states: &mut [FixelState],
) {
    let directions = odx.directions();
    let offsets = odx.offsets();

    for (src_voxel, &src_xyz) in voxel_lookup.masked_coords.iter().enumerate() {
        let src_start = offsets[src_voxel] as usize;
        let src_end = offsets[src_voxel + 1] as usize;
        if src_start == src_end {
            continue;
        }

        for [dx, dy, dz] in neighbor_offsets() {
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

            let norm = ((dx * dx + dy * dy + dz * dz) as f32).sqrt();
            let offset_unit = [dx as f32 / norm, dy as f32 / norm, dz as f32 / norm];

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
            if !value.is_finite() {
                return Err(OdxError::Argument(format!(
                    "DPF '{name}' contains a non-finite value at row {idx}"
                )));
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

fn otsu_threshold(values: &[f32]) -> f32 {
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
