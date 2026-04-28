//! pyAFQ aodf importer.
//!
//! Reads `*_param-aodf_dwimap.nii.gz` files written by pyAFQ's
//! `unified_filtering` (see `pyAFQ/AFQ/models/asym_filtering.py`) and
//! produces an ODX dataset with the SH coefficients stored as full-basis
//! `descoteaux07` SH and per-voxel asymmetric peaks precomputed at write
//! time. The optional `*.json` sidecar's provenance fields are preserved
//! in the ODX header's `extra` map for traceability.
//!
//! ## What we expect on disk
//! - 4D NIfTI of shape `(X, Y, Z, ncoeffs)` where
//!   `ncoeffs = (lmax + 1)²`. For pyAFQ at `sh_order_max = 8` this is 81.
//! - Sidecar (optional) with `OrientationEncoding.AntipodalSymmetry =
//!   false` and `Type = "odf"`. The sidecar's `Type` field labels the
//!   *interpretation* (an asymmetric ODF) rather than the storage layout
//!   — the actual array is full-basis SH, not pre-sampled amplitudes.
//!
//! ## What we write into ODX
//! - `sh/aodf.{ncoeffs}.f32` — the full-basis SH, masked-row layout.
//! - `directions.3.f32` + `offsets.uint32` — peaks detected with
//!   [`crate::peak_finder`] driven by the descoteaux full-basis evaluator
//!   (sphere sampling on a full sphere → discrete local maxima → Newton
//!   refinement against the SH analytic value).
//! - `dpf/peak_amp.f32` — refined peak amplitudes.
//! - Header: `SH_BASIS=descoteaux07`, `SH_ORDER=lmax`,
//!   `SH_FULL_BASIS=true`, `SH_LEGACY=false`,
//!   `CANONICAL_DENSE_REPRESENTATION=sh`. Sidecar JSON is folded into
//!   `extra._PYAFQ_PROVENANCE` if present.

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::descoteaux_sh;
use crate::dtype::DType;
use crate::error::{OdxError, Result};
use crate::formats::dsistudio_odf8;
use crate::formats::mrtrix::load_nifti_f32_volume;
use crate::odx_file::OdxDataset;
use crate::peak_finder::{
    peaks_from_sh_rows_with_basis, PeakFinderConfig, SpherePeakFinder,
};
use crate::sh_basis_evaluator::ShBasisKind;
use crate::stream::OdxBuilder;

/// Defaults used when the caller doesn't pass a custom config. These
/// mirror dipy's `peak_directions(...)` defaults — relative threshold 0.5
/// and minimum separation 25° — so files round-tripped through this
/// converter are comparable with reference dipy peak counts.
pub const DEFAULT_RELATIVE_PEAK_THRESHOLD: f32 = 0.5;
pub const DEFAULT_MIN_SEPARATION_DEG: f32 = 25.0;
pub const DEFAULT_MAX_PEAKS: usize = 5;

/// Knobs for the converter. Defaults match dipy.
#[derive(Debug, Clone)]
pub struct ImportOptions {
    /// Optional sidecar; if `None`, looked up by replacing `.nii.gz`/`.nii`
    /// with `.json` next to the input.
    pub sidecar_path: Option<PathBuf>,
    /// `legacy=False` per pyAFQ's default at `asym_filtering.py:50`. Pass
    /// `Some(true)` if you have an older file that was filtered with
    /// `is_legacy=True`.
    pub legacy_basis: Option<bool>,
    /// Drives [`PeakFinderConfig`].
    pub relative_peak_threshold: f32,
    pub min_separation_deg: f32,
    pub max_peaks_per_voxel: usize,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            sidecar_path: None,
            legacy_basis: None,
            relative_peak_threshold: DEFAULT_RELATIVE_PEAK_THRESHOLD,
            min_separation_deg: DEFAULT_MIN_SEPARATION_DEG,
            max_peaks_per_voxel: DEFAULT_MAX_PEAKS,
        }
    }
}

pub fn load_pyafq_aodf(nifti_path: &Path) -> Result<OdxDataset> {
    load_pyafq_aodf_with(nifti_path, ImportOptions::default())
}

pub fn load_pyafq_aodf_with(nifti_path: &Path, options: ImportOptions) -> Result<OdxDataset> {
    // 1. NIfTI: 4D float32, RAS-canonicalized, C-order
    //    (((x*Y + y)*Z + z)*ncoeffs + c).
    let (dims, affine, data) = load_nifti_f32_volume(nifti_path)?;
    if dims.len() != 4 {
        return Err(OdxError::Format(format!(
            "expected 4D aodf NIfTI '{}', got dims {:?}",
            nifti_path.display(),
            dims
        )));
    }
    let nx = dims[0];
    let ny = dims[1];
    let nz = dims[2];
    let ncoeffs = dims[3];
    let lmax = descoteaux_sh::lmax_for_ncoeffs(ncoeffs, true).map_err(|err| {
        OdxError::Format(format!(
            "aodf NIfTI has {ncoeffs} coefficients per voxel but no full-basis lmax matches: {err}"
        ))
    })?;

    // 2. Mask: any voxel with at least one non-zero SH coefficient. The
    //    aodf filter writes zeros outside the brain (and outside the
    //    pyAFQ-internal mask) so this is faithful even without a separate
    //    mask file.
    let nb_dense = nx * ny * nz;
    let mut mask = vec![0u8; nb_dense];
    let mut sh_masked: Vec<f32> = Vec::new();
    let mut nb_voxels = 0usize;
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let voxel_offset = ((x * ny + y) * nz + z) * ncoeffs;
                let row = &data[voxel_offset..voxel_offset + ncoeffs];
                let any_nonzero = row.iter().any(|&v| v != 0.0);
                if any_nonzero {
                    let flat = (x * ny + y) * nz + z;
                    mask[flat] = 1;
                    sh_masked.extend_from_slice(row);
                    nb_voxels += 1;
                }
            }
        }
    }

    // 3. Peaks: full sphere from the bundled odf8 asset, descoteaux full
    //    basis, relative threshold + min-sep matching dipy.
    let sphere_vertices = dsistudio_odf8::full_vertices_ras().to_vec();
    let sphere_faces = dsistudio_odf8::faces().to_vec();
    let finder = SpherePeakFinder::new(
        &sphere_vertices,
        &sphere_faces,
        PeakFinderConfig {
            npeaks: options.max_peaks_per_voxel,
            relative_peak_threshold: options.relative_peak_threshold,
            min_separation_angle_deg: options.min_separation_deg,
        },
    );
    let legacy = options.legacy_basis.unwrap_or(false);
    let basis = ShBasisKind::Descoteaux {
        lmax,
        full_basis: true,
        legacy,
    };
    let (offsets, directions, amplitudes) =
        peaks_from_sh_rows_with_basis(&sh_masked, nb_voxels, &finder, basis)?;

    // 4. Assemble the ODX. OdxBuilder is the easy-button writer; we feed
    //    peaks per voxel in masked order.
    let mut builder = OdxBuilder::new(
        affine,
        [nx as u64, ny as u64, nz as u64],
        mask,
    );
    for v in 0..nb_voxels {
        let start = offsets[v] as usize;
        let end = offsets[v + 1] as usize;
        builder.push_voxel_peaks(&directions[start..end]);
    }
    builder.set_sh_info(lmax as u64, "descoteaux07".into());
    builder.set_sh_full_basis(true);
    builder.set_sh_legacy(legacy);

    // Per-voxel scalars derived from the SH coefs. Both are documented in
    // the asymmetric ODF literature and give the slice viewer something
    // useful to background-color by:
    //   * `anisotropic_power` — full-basis generalisation of MRtrix AP;
    //     skips ℓ=0 and sums band-mean of c_ℓm² across all ℓ ≥ 1, then
    //     `log(AP) − log(norm_factor)`. WM lights up, GM/CSF goes dark
    //     just like the symmetric AP map.
    //   * `asymmetry_index` — Cetin Karayumak et al. 2018; ratio of
    //     odd-ℓ to total energy mapped through `√(1 − r²)`. Highlights
    //     voxels where the asymmetric filtering actually moved the FOD.
    // Compute *before* moving `sh_masked` into the SH array.
    let mut ap = Vec::with_capacity(nb_voxels);
    let mut asi = Vec::with_capacity(nb_voxels);
    for v in 0..nb_voxels {
        let row = &sh_masked[v * ncoeffs..(v + 1) * ncoeffs];
        ap.push(descoteaux_sh::anisotropic_power_full_basis(
            row,
            lmax,
            descoteaux_sh::ANISOTROPIC_POWER_NORM_FACTOR,
        ));
        asi.push(descoteaux_sh::asymmetry_index(row, lmax));
    }
    builder.set_dpv_data(
        "anisotropic_power",
        bytes_from_f32_vec(ap),
        1,
        DType::Float32,
    );
    builder.set_dpv_data(
        "asymmetry_index",
        bytes_from_f32_vec(asi),
        1,
        DType::Float32,
    );

    // Convention: SH arrays in ODX are named "coefficients" (matches the
    // PAM and MRtrix importers). TRXViz's `resolve_sh_source` and other
    // downstream code keys on this name; renaming would silently disable
    // glyph rendering.
    builder.set_sh_data(
        "coefficients",
        bytes_from_f32_vec(sh_masked),
        ncoeffs,
        DType::Float32,
    );
    // ODX/TRXViz convention: the per-fixel peak strength must be named
    // "amplitude" so QC and the renderer's Otsu fall-back find it (see
    // `odx-rs/src/qc.rs:365` — DPF resolver tries
    // amplitude → afd → qa). Calling it anything else makes the GUI
    // render zero fixels because no primary metric is selected.
    builder.set_dpf_data(
        "amplitude",
        bytes_from_f32_vec(amplitudes),
        1,
        DType::Float32,
    );

    // Sidecar provenance is optional and lives in `extra` — opaque JSON
    // so we don't have to pin a schema.
    if let Some(provenance) = read_sidecar(nifti_path, options.sidecar_path.as_deref()) {
        builder.set_extra_value("_PYAFQ_PROVENANCE", provenance);
    }

    builder.finalize()
}

fn bytes_from_f32_vec(v: Vec<f32>) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for value in v {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn read_sidecar(nifti_path: &Path, override_path: Option<&Path>) -> Option<Value> {
    let path = match override_path {
        Some(p) => p.to_path_buf(),
        None => {
            let s = nifti_path.to_string_lossy();
            let trimmed = if let Some(stem) = s.strip_suffix(".nii.gz") {
                stem.to_string()
            } else if let Some(stem) = s.strip_suffix(".nii") {
                stem.to_string()
            } else {
                return None;
            };
            PathBuf::from(format!("{trimmed}.json"))
        }
    };
    let bytes = std::fs::read(&path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::CanonicalDenseRepresentation;
    use std::path::PathBuf;

    fn test_aodf_path() -> Option<PathBuf> {
        let p = PathBuf::from(
            "/Users/mcieslak/projects/odx/test_data/afq/sub-NDARAA948VFH/ses-HBNsiteRU/dwi/models/sub-NDARAA948VFH_ses-HBNsiteRU_acq-64dir_model-csd_param-aodf_dwimap.nii.gz",
        );
        p.exists().then_some(p)
    }

    // Slow: processes the real 108×129×109 aodf and refines peaks with
    // finite-difference Newton at ~1M voxels. Run with
    // `cargo test --release --lib pyafq_aodf -- --ignored`.
    #[test]
    #[ignore = "slow real-data integration; run with --release"]
    fn imports_test_aodf_to_full_basis_descoteaux_sh() {
        let Some(path) = test_aodf_path() else {
            eprintln!("test data not found, skipping");
            return;
        };
        let dataset = load_pyafq_aodf(&path).unwrap();
        let h = dataset.header();
        assert_eq!(h.sh_basis.as_deref(), Some("descoteaux07"));
        assert_eq!(h.sh_order, Some(8));
        assert_eq!(h.sh_full_basis, Some(true));
        assert_eq!(h.sh_legacy, Some(false));
        assert_eq!(
            h.canonical_dense_representation,
            Some(CanonicalDenseRepresentation::Sh)
        );
        assert!(h.nb_voxels > 1_000, "expected non-trivial mask");
        assert!(h.nb_peaks > 0);
        // Sidecar should land in extra.
        assert!(h.extra.contains_key("_PYAFQ_PROVENANCE"));
    }
}
