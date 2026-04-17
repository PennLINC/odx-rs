use serde::Serialize;

use crate::formats::dsistudio_odf8;
use crate::header::CanonicalDenseRepresentation;
use crate::{OdxDataset, OdxError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationSeverity {
    Warning,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ValidationIssue {
    pub severity: ValidationSeverity,
    pub code: &'static str,
    pub message: String,
}

impl ValidationIssue {
    fn warning(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            severity: ValidationSeverity::Warning,
            code,
            message: message.into(),
        }
    }

    fn error(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            severity: ValidationSeverity::Error,
            code,
            message: message.into(),
        }
    }
}

pub fn validate_dataset(odx: &OdxDataset) -> Result<()> {
    let issues = validate_dataset_detailed(odx);
    let errors = issues
        .iter()
        .filter(|issue| issue.severity == ValidationSeverity::Error)
        .collect::<Vec<_>>();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(OdxError::Format(
            errors
                .into_iter()
                .map(|issue| format!("{}: {}", issue.code, issue.message))
                .collect::<Vec<_>>()
                .join("; "),
        ))
    }
}

pub fn validate_dataset_detailed(odx: &OdxDataset) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    let header = odx.header();
    let mask_volume_size = header.mask_volume_size();

    if odx.mask().len() != mask_volume_size {
        issues.push(ValidationIssue::error(
            "mask_length",
            format!(
                "mask has {} entries but dimensions imply {} voxels",
                odx.mask().len(),
                mask_volume_size
            ),
        ));
    }

    let mask_count = odx.mask().iter().filter(|&&value| value != 0).count();
    if mask_count != odx.nb_voxels() {
        issues.push(ValidationIssue::error(
            "mask_voxel_count",
            format!(
                "mask has {} nonzero voxels but NB_VOXELS is {}",
                mask_count,
                odx.nb_voxels()
            ),
        ));
    }

    if odx.offsets().len() != odx.nb_voxels() + 1 {
        issues.push(ValidationIssue::error(
            "offset_count",
            format!(
                "offsets has {} entries but NB_VOXELS + 1 is {}",
                odx.offsets().len(),
                odx.nb_voxels() + 1
            ),
        ));
    }

    if odx.offsets().last().copied().unwrap_or(0) as usize != odx.nb_peaks() {
        issues.push(ValidationIssue::error(
            "offset_sentinel",
            format!(
                "offset sentinel {} does not match NB_PEAKS {}",
                odx.offsets().last().copied().unwrap_or(0),
                odx.nb_peaks()
            ),
        ));
    }

    if odx.directions().len() != odx.nb_peaks() {
        issues.push(ValidationIssue::error(
            "direction_count",
            format!(
                "directions has {} rows but NB_PEAKS is {}",
                odx.directions().len(),
                odx.nb_peaks()
            ),
        ));
    }

    for (name, info) in odx.iter_dpv() {
        if info.nrows != odx.nb_voxels() {
            issues.push(ValidationIssue::error(
                "dpv_rows",
                format!(
                    "DPV '{}' has {} rows but NB_VOXELS is {}",
                    name,
                    info.nrows,
                    odx.nb_voxels()
                ),
            ));
        }
    }

    for (name, info) in odx.iter_dpf() {
        if info.nrows != odx.nb_peaks() {
            issues.push(ValidationIssue::error(
                "dpf_rows",
                format!(
                    "DPF '{}' has {} rows but NB_PEAKS is {}",
                    name,
                    info.nrows,
                    odx.nb_peaks()
                ),
            ));
        }
    }

    let odf_names = odx.odf_names();
    if !odf_names.is_empty() {
        let has_explicit_sphere = odx.sphere_vertices().is_some();
        if !has_explicit_sphere && header.sphere_id.is_none() {
            issues.push(ValidationIssue::error(
                "odf_sphere_missing",
                "ODF arrays are present but neither sphere vertices nor SPHERE_ID are set",
            ));
        }

        if header.odf_sample_domain.is_none() {
            issues.push(ValidationIssue::warning(
                "odf_sample_domain_missing",
                "ODF arrays are present but ODF_SAMPLE_DOMAIN is not set",
            ));
        }

        if header.canonical_dense_representation.is_none() {
            issues.push(ValidationIssue::warning(
                "canonical_dense_missing",
                "dense arrays are present but CANONICAL_DENSE_REPRESENTATION is not set",
            ));
        }

        for &name in &odf_names {
            match odx.odf::<f32>(name) {
                Ok(view) => {
                    if view.nrows() != odx.nb_voxels() {
                        issues.push(ValidationIssue::error(
                            "odf_rows",
                            format!(
                                "ODF '{}' has {} rows but NB_VOXELS is {}",
                                name,
                                view.nrows(),
                                odx.nb_voxels()
                            ),
                        ));
                    }
                    if header.odf_sample_domain.as_deref() == Some("hemisphere") {
                        let expected = if let Some(vertices) = odx.sphere_vertices() {
                            Some(vertices.len() / 2)
                        } else if header.sphere_id.as_deref() == Some("dsistudio_odf8") {
                            Some(dsistudio_odf8::hemisphere_vertices_ras().len())
                        } else {
                            None
                        };
                        if let Some(expected_cols) = expected {
                            if view.ncols() != expected_cols {
                                issues.push(ValidationIssue::error(
                                    "odf_hemisphere_columns",
                                    format!(
                                        "ODF '{}' has {} columns but hemisphere sampling expects {}",
                                        name,
                                        view.ncols(),
                                        expected_cols
                                    ),
                                ));
                            }
                        }
                    }
                }
                Err(err) => issues.push(ValidationIssue::error(
                    "odf_read",
                    format!("ODF '{}' could not be read as float32: {err}", name),
                )),
            }
        }
    }

    let sh_names = odx.sh_names();
    if !sh_names.is_empty() {
        if header.canonical_dense_representation.is_none() {
            issues.push(ValidationIssue::warning(
                "canonical_dense_missing",
                "dense arrays are present but CANONICAL_DENSE_REPRESENTATION is not set",
            ));
        }
        if let Some(order) = header.sh_order {
            let expected_cols = ((order + 1) * (order + 2) / 2) as usize;
            for &name in &sh_names {
                match odx.sh::<f32>(name) {
                    Ok(view) => {
                        if view.nrows() != odx.nb_voxels() {
                            issues.push(ValidationIssue::error(
                                "sh_rows",
                                format!(
                                    "SH '{}' has {} rows but NB_VOXELS is {}",
                                    name,
                                    view.nrows(),
                                    odx.nb_voxels()
                                ),
                            ));
                        }
                        if view.ncols() != expected_cols {
                            issues.push(ValidationIssue::error(
                                "sh_columns",
                                format!(
                                    "SH '{}' has {} columns but SH_ORDER={} expects {}",
                                    name,
                                    view.ncols(),
                                    order,
                                    expected_cols
                                ),
                            ));
                        }
                    }
                    Err(err) => issues.push(ValidationIssue::error(
                        "sh_read",
                        format!("SH '{}' could not be read as float32: {err}", name),
                    )),
                }
            }
        } else {
            issues.push(ValidationIssue::error(
                "sh_order_missing",
                "SH arrays are present but SH_ORDER is not set",
            ));
        }
    }

    if header.canonical_dense_representation == Some(CanonicalDenseRepresentation::Odf)
        && odf_names.is_empty()
    {
        issues.push(ValidationIssue::warning(
            "canonical_dense_inconsistent",
            "CANONICAL_DENSE_REPRESENTATION is 'odf' but no ODF arrays are present",
        ));
    }

    if header.canonical_dense_representation == Some(CanonicalDenseRepresentation::Sh)
        && sh_names.is_empty()
    {
        issues.push(ValidationIssue::warning(
            "canonical_dense_inconsistent",
            "CANONICAL_DENSE_REPRESENTATION is 'sh' but no SH arrays are present",
        ));
    }

    issues
}
