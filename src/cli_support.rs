use std::fs;
use std::path::Path;

use serde::Serialize;

use crate::error::{OdxError, Result};
use crate::formats::{dsistudio, mrtrix, pam, tortoise_mapmri};
use crate::header::CanonicalDenseRepresentation;
use crate::reference_affine::read_reference_affine;
use crate::{validate_dataset_detailed, OdxDataset, ValidationIssue, ValidationSeverity};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DetectedFormat {
    OdxDirectory,
    OdxArchive,
    DsistudioFibGz,
    DsistudioFz,
    DipyPam5,
    TortoiseMapmriNifti,
    MrtrixShImage,
    MrtrixFixelDir,
}

impl DetectedFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OdxDirectory => "odx_directory",
            Self::OdxArchive => "odx_archive",
            Self::DsistudioFibGz => "dsistudio_fibgz",
            Self::DsistudioFz => "dsistudio_fz",
            Self::DipyPam5 => "dipy_pam5",
            Self::TortoiseMapmriNifti => "tortoise_mapmri_nifti",
            Self::MrtrixShImage => "mrtrix_sh_image",
            Self::MrtrixFixelDir => "mrtrix_fixel_dir",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct LoadDatasetOptions<'a> {
    pub sh_path: Option<&'a Path>,
    pub fixel_dir: Option<&'a Path>,
    pub reference_affine: Option<&'a Path>,
    pub mapmri_tensor_path: Option<&'a Path>,
    pub mapmri_uvec_path: Option<&'a Path>,
    /// MRtrix-only: when true, NIfTI inputs keep their on-disk affine and
    /// (i,j,k) ordering instead of being canonicalized to RAS+. Set this when
    /// comparing against ODXs produced by Python pipelines that ingest via
    /// nibabel without reorienting (e.g. cs-odf), since nibabel reads the
    /// sform/qform untouched.
    pub preserve_nifti_affine: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArraySummary {
    pub name: String,
    pub nrows: usize,
    pub ncols: usize,
    pub dtype: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct DatasetSummary {
    pub detected_format: String,
    pub dimensions: [u64; 3],
    pub nb_voxels: u64,
    pub nb_peaks: u64,
    pub voxel_to_rasmm: [[f64; 4]; 4],
    pub sh_basis: Option<String>,
    pub sh_order: Option<u64>,
    pub sh_full_basis: Option<bool>,
    pub sh_legacy: Option<bool>,
    pub canonical_dense_representation: Option<String>,
    pub odf_sample_domain: Option<String>,
    pub sphere_id: Option<String>,
    pub nb_sphere_vertices: Option<u64>,
    pub nb_sphere_faces: Option<u64>,
    pub odf_arrays: Vec<ArraySummary>,
    pub sh_arrays: Vec<ArraySummary>,
    pub dpv_arrays: Vec<ArraySummary>,
    pub dpf_arrays: Vec<ArraySummary>,
    pub quantized_arrays: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationReport {
    pub ok: bool,
    pub strict_ok: bool,
    pub warnings: usize,
    pub errors: usize,
    pub issues: Vec<ValidationIssue>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConversionSummary {
    pub input_format: String,
    pub output_format: String,
    pub output_path: String,
    pub out_sh_path: Option<String>,
    pub nb_voxels: u64,
    pub nb_peaks: u64,
}

pub fn detect_existing_input_format(path: &Path) -> Result<DetectedFormat> {
    if !path.exists() {
        return Err(OdxError::FileNotFound(path.to_path_buf()));
    }
    detect_path_format(path, true)
}

pub fn detect_target_format(path: &Path) -> Result<DetectedFormat> {
    detect_path_format(path, false)
}

fn detect_path_format(path: &Path, existing_only: bool) -> Result<DetectedFormat> {
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    if path.is_dir() {
        if !existing_only && (file_name.ends_with(".odx") || file_name.ends_with(".odxd")) {
            return Ok(DetectedFormat::OdxDirectory);
        }
        if path.join("header.json").exists() {
            return Ok(DetectedFormat::OdxDirectory);
        }
        if has_fixel_index(path) && has_fixel_directions(path) {
            return Ok(DetectedFormat::MrtrixFixelDir);
        }
        return Err(OdxError::Format(format!(
            "cannot detect directory format for '{}': expected ODX header.json or MRtrix index/directions",
            path.display()
        )));
    }

    if file_name.ends_with(".fib.gz") {
        return Ok(DetectedFormat::DsistudioFibGz);
    }
    if file_name.ends_with(".fz") {
        return Ok(DetectedFormat::DsistudioFz);
    }
    if file_name.ends_with(".pam5") {
        return Ok(DetectedFormat::DipyPam5);
    }
    if file_name.ends_with(".odx") {
        return Ok(DetectedFormat::OdxArchive);
    }
    if file_name.ends_with(".odxd") {
        return Ok(DetectedFormat::OdxDirectory);
    }

    let lower = file_name.to_ascii_lowercase();
    if lower.ends_with(".mif")
        || lower.ends_with(".mif.gz")
        || lower.ends_with(".nii")
        || lower.ends_with(".nii.gz")
    {
        return Ok(DetectedFormat::MrtrixShImage);
    }

    if !existing_only && path.extension().is_none() {
        return Ok(DetectedFormat::MrtrixFixelDir);
    }

    Err(OdxError::Format(format!(
        "cannot detect format for '{}'; use --input-format/--output-format",
        path.display()
    )))
}

pub fn load_dataset(
    path: &Path,
    options: LoadDatasetOptions<'_>,
) -> Result<(OdxDataset, DetectedFormat)> {
    let detected = detect_existing_input_format(path)?;
    let dataset = load_dataset_with_format(path, detected, options)?;
    Ok((dataset, detected))
}

pub fn load_dataset_with_format(
    path: &Path,
    detected: DetectedFormat,
    options: LoadDatasetOptions<'_>,
) -> Result<OdxDataset> {
    let dataset = match detected {
        DetectedFormat::OdxDirectory | DetectedFormat::OdxArchive => {
            reject_companion_inputs(
                detected,
                options.sh_path,
                options.fixel_dir,
                options.reference_affine,
                options.mapmri_tensor_path,
                options.mapmri_uvec_path,
            )?;
            OdxDataset::load(path)?
        }
        DetectedFormat::DsistudioFibGz | DetectedFormat::DsistudioFz => {
            reject_mrtrix_companions(
                detected,
                options.sh_path,
                options.fixel_dir,
                options.mapmri_tensor_path,
                options.mapmri_uvec_path,
            )?;
            let affine = options
                .reference_affine
                .map(read_reference_affine)
                .transpose()?;
            match detected {
                DetectedFormat::DsistudioFibGz => dsistudio::load_fibgz(path, affine)?,
                DetectedFormat::DsistudioFz => dsistudio::load_fz(path, affine)?,
                _ => unreachable!(),
            }
        }
        DetectedFormat::DipyPam5 => {
            reject_companion_inputs(
                detected,
                options.sh_path,
                options.fixel_dir,
                options.reference_affine,
                options.mapmri_tensor_path,
                options.mapmri_uvec_path,
            )?;
            pam::load_pam5(path)?
        }
        DetectedFormat::TortoiseMapmriNifti => {
            reject_companion_inputs(
                detected,
                options.sh_path,
                options.fixel_dir,
                options.reference_affine,
                None,
                None,
            )?;
            let tensor_path = options.mapmri_tensor_path.ok_or_else(|| {
                OdxError::Argument(
                    "--mapmri-tensor is required for tortoise_mapmri_nifti inputs".into(),
                )
            })?;
            let uvec_path = options.mapmri_uvec_path.ok_or_else(|| {
                OdxError::Argument(
                    "--mapmri-uvec is required for tortoise_mapmri_nifti inputs".into(),
                )
            })?;
            tortoise_mapmri::load_tortoise_mapmri(path, tensor_path, uvec_path)?
        }
        DetectedFormat::MrtrixShImage => {
            if options.mapmri_tensor_path.is_some() {
                return Err(OdxError::Argument(
                    "--mapmri-tensor is only valid for TORTOISE MAPMRI inputs".into(),
                ));
            }
            if options.mapmri_uvec_path.is_some() {
                return Err(OdxError::Argument(
                    "--mapmri-uvec is only valid for TORTOISE MAPMRI inputs".into(),
                ));
            }
            if options.sh_path.is_some() {
                return Err(OdxError::Argument(
                    "--sh is only valid when the primary input is a MRtrix fixel directory".into(),
                ));
            }
            if options.reference_affine.is_some() {
                return Err(OdxError::Argument(
                    "--reference-affine is only valid for DSI Studio inputs".into(),
                ));
            }
            mrtrix::load_mrtrix_dataset_with_options(
                Some(path),
                options.fixel_dir,
                &mrtrix::MrtrixDatasetLoadOptions {
                    preserve_nifti_affine: options.preserve_nifti_affine,
                    ..Default::default()
                },
            )?
        }
        DetectedFormat::MrtrixFixelDir => {
            if options.mapmri_tensor_path.is_some() {
                return Err(OdxError::Argument(
                    "--mapmri-tensor is only valid for TORTOISE MAPMRI inputs".into(),
                ));
            }
            if options.mapmri_uvec_path.is_some() {
                return Err(OdxError::Argument(
                    "--mapmri-uvec is only valid for TORTOISE MAPMRI inputs".into(),
                ));
            }
            if options.fixel_dir.is_some() {
                return Err(OdxError::Argument(
                    "--fixel-dir is only valid when the primary input is an SH image".into(),
                ));
            }
            if options.reference_affine.is_some() {
                return Err(OdxError::Argument(
                    "--reference-affine is only valid for DSI Studio inputs".into(),
                ));
            }
            mrtrix::load_mrtrix_dataset_with_options(
                options.sh_path,
                Some(path),
                &mrtrix::MrtrixDatasetLoadOptions {
                    preserve_nifti_affine: options.preserve_nifti_affine,
                    ..Default::default()
                },
            )?
        }
    };
    Ok(dataset)
}

pub fn summarize_dataset(odx: &OdxDataset, detected_format: DetectedFormat) -> DatasetSummary {
    let mut odf_arrays = odx
        .odf_names()
        .into_iter()
        .filter_map(|name| odx.odf_arrays().get(name).map(|arr| (name, arr)))
        .map(array_summary)
        .collect::<Vec<_>>();
    odf_arrays.sort_by(|a, b| a.name.cmp(&b.name));

    let mut sh_arrays = odx
        .sh_names()
        .into_iter()
        .filter_map(|name| odx.sh_arrays().get(name).map(|arr| (name, arr)))
        .map(array_summary)
        .collect::<Vec<_>>();
    sh_arrays.sort_by(|a, b| a.name.cmp(&b.name));

    let mut dpv_arrays = odx
        .iter_dpv()
        .map(|(name, info)| ArraySummary {
            name: name.to_string(),
            nrows: info.nrows,
            ncols: info.ncols,
            dtype: info.dtype.to_string(),
        })
        .collect::<Vec<_>>();
    dpv_arrays.sort_by(|a, b| a.name.cmp(&b.name));

    let mut dpf_arrays = odx
        .iter_dpf()
        .map(|(name, info)| ArraySummary {
            name: name.to_string(),
            nrows: info.nrows,
            ncols: info.ncols,
            dtype: info.dtype.to_string(),
        })
        .collect::<Vec<_>>();
    dpf_arrays.sort_by(|a, b| a.name.cmp(&b.name));

    let mut quantized_arrays = odx
        .header()
        .array_quantization
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    quantized_arrays.sort();

    DatasetSummary {
        detected_format: detected_format.as_str().into(),
        dimensions: odx.header().dimensions,
        nb_voxels: odx.header().nb_voxels,
        nb_peaks: odx.header().nb_peaks,
        voxel_to_rasmm: odx.header().voxel_to_rasmm,
        sh_basis: odx.header().sh_basis.clone(),
        sh_order: odx.header().sh_order,
        sh_full_basis: odx.header().sh_full_basis,
        sh_legacy: odx.header().sh_legacy,
        canonical_dense_representation: odx
            .header()
            .canonical_dense_representation
            .as_ref()
            .map(canonical_dense_representation_name),
        odf_sample_domain: odx.header().odf_sample_domain.clone(),
        sphere_id: odx.header().sphere_id.clone(),
        nb_sphere_vertices: odx.header().nb_sphere_vertices,
        nb_sphere_faces: odx.header().nb_sphere_faces,
        odf_arrays,
        sh_arrays,
        dpv_arrays,
        dpf_arrays,
        quantized_arrays,
    }
}

pub fn render_summary(summary: &DatasetSummary) -> String {
    let mut out = String::new();
    out.push_str(&format!("format: {}\n", summary.detected_format));
    out.push_str(&format!(
        "dimensions: {} x {} x {}\n",
        summary.dimensions[0], summary.dimensions[1], summary.dimensions[2]
    ));
    out.push_str(&format!("voxels: {}\n", summary.nb_voxels));
    out.push_str(&format!("peaks: {}\n", summary.nb_peaks));
    out.push_str(&format!(
        "affine: [{:.4}, {:.4}, {:.4}, {:.4}] [{:.4}, {:.4}, {:.4}, {:.4}] [{:.4}, {:.4}, {:.4}, {:.4}]\n",
        summary.voxel_to_rasmm[0][0],
        summary.voxel_to_rasmm[0][1],
        summary.voxel_to_rasmm[0][2],
        summary.voxel_to_rasmm[0][3],
        summary.voxel_to_rasmm[1][0],
        summary.voxel_to_rasmm[1][1],
        summary.voxel_to_rasmm[1][2],
        summary.voxel_to_rasmm[1][3],
        summary.voxel_to_rasmm[2][0],
        summary.voxel_to_rasmm[2][1],
        summary.voxel_to_rasmm[2][2],
        summary.voxel_to_rasmm[2][3],
    ));

    if let Some(order) = summary.sh_order {
        let mut line = format!(
            "sh: basis={} order={}",
            summary.sh_basis.as_deref().unwrap_or("unknown"),
            order
        );
        if let Some(full) = summary.sh_full_basis {
            line.push_str(&format!(" full_basis={full}"));
        }
        if let Some(legacy) = summary.sh_legacy {
            line.push_str(&format!(" legacy={legacy}"));
        }
        line.push('\n');
        out.push_str(&line);
    }
    if let Some(rep) = summary.canonical_dense_representation.as_deref() {
        out.push_str(&format!("canonical_dense_representation: {rep}\n"));
    }
    if let Some(domain) = summary.odf_sample_domain.as_deref() {
        out.push_str(&format!("odf_sample_domain: {domain}\n"));
    }
    if summary.nb_sphere_vertices.is_some() || summary.sphere_id.is_some() {
        out.push_str(&format!(
            "sphere: id={} vertices={} faces={}\n",
            summary.sphere_id.as_deref().unwrap_or("none"),
            summary
                .nb_sphere_vertices
                .map(|v| v.to_string())
                .unwrap_or_else(|| "none".into()),
            summary
                .nb_sphere_faces
                .map(|v| v.to_string())
                .unwrap_or_else(|| "none".into())
        ));
    }
    if !summary.odf_arrays.is_empty() {
        out.push_str(&format!(
            "odf arrays: {}\n",
            render_arrays(&summary.odf_arrays)
        ));
    }
    if !summary.sh_arrays.is_empty() {
        out.push_str(&format!(
            "sh arrays: {}\n",
            render_arrays(&summary.sh_arrays)
        ));
    }
    if !summary.dpv_arrays.is_empty() {
        out.push_str(&format!(
            "dpv arrays: {}\n",
            render_arrays(&summary.dpv_arrays)
        ));
    }
    if !summary.dpf_arrays.is_empty() {
        out.push_str(&format!(
            "dpf arrays: {}\n",
            render_arrays(&summary.dpf_arrays)
        ));
    }
    if !summary.quantized_arrays.is_empty() {
        out.push_str(&format!(
            "quantized arrays: {}\n",
            summary.quantized_arrays.join(", ")
        ));
    }
    out
}

pub fn validation_report(odx: &OdxDataset) -> ValidationReport {
    let issues = validate_dataset_detailed(odx);
    let warnings = issues
        .iter()
        .filter(|issue| issue.severity == ValidationSeverity::Warning)
        .count();
    let errors = issues
        .iter()
        .filter(|issue| issue.severity == ValidationSeverity::Error)
        .count();
    ValidationReport {
        ok: errors == 0,
        strict_ok: errors == 0 && warnings == 0,
        warnings,
        errors,
        issues,
    }
}

pub fn render_validation(report: &ValidationReport) -> String {
    let mut out = format!(
        "validation: {} ({} errors, {} warnings)\n",
        if report.ok { "ok" } else { "failed" },
        report.errors,
        report.warnings
    );
    for issue in &report.issues {
        let severity = match issue.severity {
            ValidationSeverity::Warning => "warning",
            ValidationSeverity::Error => "error",
        };
        out.push_str(&format!("{severity}: {}\n", issue.message));
    }
    out
}

pub fn ensure_output_path(path: &Path, overwrite: bool) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    if !overwrite {
        return Err(OdxError::Argument(format!(
            "output path '{}' already exists; pass --overwrite to replace it",
            path.display()
        )));
    }
    if path.is_dir() {
        fs::remove_dir_all(path)?;
    } else {
        fs::remove_file(path)?;
    }
    Ok(())
}

fn has_fixel_index(path: &Path) -> bool {
    path.join("index.mif").exists()
        || path.join("index.nii").exists()
        || path.join("index.nii.gz").exists()
}

fn has_fixel_directions(path: &Path) -> bool {
    path.join("directions.mif").exists()
        || path.join("directions.nii").exists()
        || path.join("directions.nii.gz").exists()
}

fn reject_companion_inputs(
    format: DetectedFormat,
    sh_path: Option<&Path>,
    fixel_dir: Option<&Path>,
    reference_affine: Option<&Path>,
    mapmri_tensor_path: Option<&Path>,
    mapmri_uvec_path: Option<&Path>,
) -> Result<()> {
    reject_mrtrix_companions(
        format,
        sh_path,
        fixel_dir,
        mapmri_tensor_path,
        mapmri_uvec_path,
    )?;
    if reference_affine.is_some() {
        return Err(OdxError::Argument(
            "--reference-affine is only valid for DSI Studio inputs".into(),
        ));
    }
    Ok(())
}

fn reject_mrtrix_companions(
    format: DetectedFormat,
    sh_path: Option<&Path>,
    fixel_dir: Option<&Path>,
    mapmri_tensor_path: Option<&Path>,
    mapmri_uvec_path: Option<&Path>,
) -> Result<()> {
    if sh_path.is_some()
        || fixel_dir.is_some()
        || mapmri_tensor_path.is_some()
        || mapmri_uvec_path.is_some()
    {
        return Err(OdxError::Argument(format!(
            "{} inputs do not accept companion input flags",
            format.as_str()
        )));
    }
    Ok(())
}

fn array_summary((name, arr): (&str, &crate::data_array::DataArray)) -> ArraySummary {
    ArraySummary {
        name: name.to_string(),
        nrows: arr.nrows(),
        ncols: arr.ncols(),
        dtype: arr.dtype().to_string(),
    }
}

fn render_arrays(arrays: &[ArraySummary]) -> String {
    arrays
        .iter()
        .map(|arr| format!("{}({}x{}, {})", arr.name, arr.nrows, arr.ncols, arr.dtype))
        .collect::<Vec<_>>()
        .join(", ")
}

fn canonical_dense_representation_name(rep: &CanonicalDenseRepresentation) -> String {
    match rep {
        CanonicalDenseRepresentation::Sh => "sh".into(),
        CanonicalDenseRepresentation::Odf => "odf".into(),
    }
}
