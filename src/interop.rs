use std::collections::HashSet;
use std::path::Path;

use serde_json::Value;

use crate::data_array::DataArray;
use crate::dtype::DType;
use crate::error::{OdxError, Result};
use crate::formats::dsistudio;
use crate::formats::dsistudio_odf8;
use crate::formats::mrtrix;
use crate::header::CanonicalDenseRepresentation;
use crate::mmap_backing::{vec_to_bytes, MmapBacking};
use crate::mrtrix_sh;
use crate::odx_file::{OdxDataset, OdxParts};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DsistudioFormat {
    FibGz,
    Fz,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseOdfMode {
    Off,
    FromSh,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeakSource {
    Fixels,
    SampledOdf,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Z0Policy {
    Auto,
    Never,
    Always,
}

#[derive(Debug, Clone)]
pub struct DsistudioToMrtrixOptions {
    pub reference_affine: Option<[[f64; 4]; 4]>,
    pub write_sh: bool,
    pub sh_lmax: Option<u32>,
    pub fixel_container: mrtrix::MrtrixFixelContainer,
    pub fixel_amplitude_name: String,
}

impl Default for DsistudioToMrtrixOptions {
    fn default() -> Self {
        Self {
            reference_affine: None,
            write_sh: true,
            sh_lmax: None,
            fixel_container: mrtrix::MrtrixFixelContainer::Nifti,
            fixel_amplitude_name: "amplitude".into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MrtrixToDsistudioOptions {
    pub output_format: DsistudioFormat,
    pub dense_odf_mode: DenseOdfMode,
    pub peak_source: PeakSource,
    pub amplitude_key: Option<String>,
    pub write_z0: Z0Policy,
}

impl Default for MrtrixToDsistudioOptions {
    fn default() -> Self {
        Self {
            output_format: DsistudioFormat::Fz,
            dense_odf_mode: DenseOdfMode::FromSh,
            peak_source: PeakSource::Fixels,
            amplitude_key: None,
            write_z0: Z0Policy::Auto,
        }
    }
}

pub fn dsistudio_to_mrtrix(
    input_ds_path: &Path,
    out_fixels_dir: &Path,
    out_sh_path: Option<&Path>,
    options: &DsistudioToMrtrixOptions,
) -> Result<()> {
    // These wrappers are intentionally thin: all semantic loading/writing still
    // goes through OdxDataset so we don't create a second pairwise converter core.
    let odx = load_dsistudio_dataset(input_ds_path, options.reference_affine)?;
    let fixel_dataset = if options.fixel_amplitude_name == "amplitude" {
        odx.clone_owned_parts()
    } else {
        let mut parts = odx.clone_owned_parts();
        if let Some(amplitude) = parts.dpf.remove("amplitude") {
            parts
                .dpf
                .insert(options.fixel_amplitude_name.clone(), amplitude);
        }
        parts
    };

    mrtrix::save_mrtrix_fixels(
        &OdxDataset::from_parts(fixel_dataset),
        out_fixels_dir,
        &mrtrix::MrtrixFixelWriteOptions {
            container: options.fixel_container,
            include_dpf: true,
            include_dpv: false,
        },
    )?;

    if options.write_sh {
        if let Some(path) = out_sh_path {
            if let Some(sh_dataset) = add_sh_from_dsistudio_odf(&odx, options.sh_lmax)? {
                mrtrix::save_mrtrix_sh(
                    &sh_dataset,
                    path,
                    &mrtrix::MrtrixShWriteOptions::default(),
                )?;
            }
        }
    }

    Ok(())
}

pub fn fit_mrtrix_sh_from_odf(
    odx: &OdxDataset,
    requested_lmax: Option<u32>,
) -> Result<Option<OdxDataset>> {
    add_sh_from_dsistudio_odf(odx, requested_lmax)
}

pub fn save_dsistudio_from_odx(
    odx: &OdxDataset,
    out_ds_path: &Path,
    options: &MrtrixToDsistudioOptions,
) -> Result<()> {
    let prepared = prepare_dsistudio_dataset(odx, options)?;
    match options.output_format {
        DsistudioFormat::FibGz => dsistudio::save_fibgz(&prepared, out_ds_path),
        DsistudioFormat::Fz => dsistudio::save_fz(&prepared, out_ds_path),
    }
}

pub fn mrtrix_to_dsistudio(
    fixel_dir: &Path,
    sh_path: Option<&Path>,
    out_ds_path: &Path,
    options: &MrtrixToDsistudioOptions,
) -> Result<()> {
    let odx = mrtrix::load_mrtrix_dataset(sh_path, Some(fixel_dir))?;
    save_dsistudio_from_odx(&odx, out_ds_path, options)
}

fn load_dsistudio_dataset(path: &Path, affine: Option<[[f64; 4]; 4]>) -> Result<OdxDataset> {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    if name.ends_with(".fz") {
        dsistudio::load_fz(path, affine)
    } else {
        dsistudio::load_fibgz(path, affine)
    }
}

fn add_sh_from_dsistudio_odf(
    odx: &OdxDataset,
    requested_lmax: Option<u32>,
) -> Result<Option<OdxDataset>> {
    let odf = match odx.odf::<f32>("amplitudes") {
        Ok(view) => view,
        Err(_) => return Ok(None),
    };
    let sample_dirs = dsistudio_sampling_dirs(odx, odf.ncols())?;
    let lmax = mrtrix_sh::resolve_lmax_for_directions(
        &sample_dirs,
        requested_lmax.map(|value| value as usize),
        8,
    );
    let coeffs =
        mrtrix_sh::fit_rows_from_amplitudes(odf.as_flat_slice(), odf.nrows(), &sample_dirs, lmax)?;

    let mut parts = odx.clone_owned_parts();
    parts.header.sh_order = Some(lmax as u64);
    parts.header.sh_basis = Some("tournier07".into());
    parts.header.canonical_dense_representation = Some(CanonicalDenseRepresentation::Sh);
    parts.sh.insert(
        "coefficients".into(),
        DataArray::owned_bytes(
            vec_to_bytes(coeffs),
            mrtrix_sh::ncoeffs_for_lmax(lmax),
            DType::Float32,
        ),
    );
    Ok(Some(OdxDataset::from_parts(parts)))
}

fn prepare_dsistudio_dataset(
    odx: &OdxDataset,
    options: &MrtrixToDsistudioOptions,
) -> Result<OdxDataset> {
    let mut parts = odx.clone_owned_parts();
    parts.header.extra.insert(
        "_ODX_DSISTUDIO_VOXEL_POLICY".into(),
        serde_json::Value::String("reorient_to_lps_fortran".into()),
    );
    install_builtin_dsistudio_sphere(&mut parts);

    let amplitude = resolve_amplitude_values(odx, options.amplitude_key.as_deref())?;
    let sampled_odf = if options.dense_odf_mode == DenseOdfMode::FromSh {
        sample_dense_odf_from_sh(odx)?
    } else {
        None
    };

    if let Some(ref dense) = sampled_odf {
        parts.odf.insert(
            "amplitudes".into(),
            DataArray::owned_bytes(
                vec_to_bytes(dense.clone()),
                dsistudio_odf8::hemisphere_vertices_ras().len(),
                DType::Float32,
            ),
        );
        parts.header.canonical_dense_representation = Some(CanonicalDenseRepresentation::Odf);
        parts.header.odf_sample_domain = Some("hemisphere".into());
    } else {
        parts.odf.remove("amplitudes");
        parts.header.odf_sample_domain = None;
    }

    let (offsets, directions, amplitudes) = match options.peak_source {
        PeakSource::Fixels => sort_fixels_by_amplitude(odx, &amplitude)?,
        PeakSource::SampledOdf => {
            let dense = sampled_odf.as_ref().ok_or_else(|| {
                OdxError::Argument(
                    "PeakSource::SampledOdf requires SH-driven dense ODF export".into(),
                )
            })?;
            peaks_from_sampled_odf(odx, dense)
        }
    };

    replace_peaks(&mut parts, offsets, directions, amplitudes);
    apply_z0_policy(&mut parts, options.write_z0, options.output_format);

    Ok(OdxDataset::from_parts(parts))
}

fn dsistudio_sampling_dirs(odx: &OdxDataset, ncols: usize) -> Result<Vec<[f32; 3]>> {
    if let Some(vertices) = odx.sphere_vertices() {
        if vertices.len() >= ncols {
            return Ok(vertices[..ncols].to_vec());
        }
    }
    let builtin = dsistudio_odf8::hemisphere_vertices_ras();
    if builtin.len() == ncols {
        return Ok(builtin.to_vec());
    }
    Err(OdxError::Format(format!(
        "cannot resolve DSI Studio sampling directions for {ncols} ODF columns"
    )))
}

fn sample_dense_odf_from_sh(odx: &OdxDataset) -> Result<Option<Vec<f32>>> {
    let sh = match odx.sh::<f32>("coefficients") {
        Ok(view) => view,
        Err(_) => return Ok(None),
    };
    Ok(Some(mrtrix_sh::sample_rows_nonnegative(
        sh.as_flat_slice(),
        sh.nrows(),
        dsistudio_odf8::hemisphere_vertices_ras(),
        sh.ncols(),
    )?))
}

fn install_builtin_dsistudio_sphere(parts: &mut OdxParts) {
    let vertices = dsistudio_odf8::full_vertices_ras().to_vec();
    let faces = dsistudio_odf8::faces().to_vec();
    parts.header.nb_sphere_vertices = Some(vertices.len() as u64);
    parts.header.nb_sphere_faces = Some(faces.len() as u64);
    parts.header.sphere_id = Some("dsistudio_odf8".into());
    parts.sphere_vertices = Some(MmapBacking::Owned(vec_to_bytes(vertices)));
    parts.sphere_faces = Some(MmapBacking::Owned(vec_to_bytes(faces)));
}

fn resolve_amplitude_values(odx: &OdxDataset, explicit_key: Option<&str>) -> Result<Vec<f32>> {
    if let Some(key) = explicit_key {
        return odx.scalar_dpf_f32(key);
    }
    if let Ok(values) = odx.scalar_dpf_f32("amplitude") {
        return Ok(values);
    }
    if let Ok(values) = odx.scalar_dpf_f32("afd") {
        return Ok(values);
    }

    let mut single_column = odx
        .iter_dpf()
        .filter_map(|(name, info)| (info.ncols == 1).then_some(name.to_string()))
        .collect::<Vec<_>>();
    single_column.sort();
    if let Some(name) = single_column.first() {
        return odx.scalar_dpf_f32(name);
    }
    Ok(vec![1.0; odx.nb_peaks()])
}

fn sort_fixels_by_amplitude(
    odx: &OdxDataset,
    amplitudes: &[f32],
) -> Result<(Vec<u32>, Vec<[f32; 3]>, Vec<f32>)> {
    let mut offsets = Vec::with_capacity(odx.nb_voxels() + 1);
    let mut directions = Vec::with_capacity(odx.nb_peaks());
    let mut out_amplitude = Vec::with_capacity(odx.nb_peaks());
    offsets.push(0);

    for voxel in 0..odx.nb_voxels() {
        let start = odx.offsets()[voxel] as usize;
        let end = odx.offsets()[voxel + 1] as usize;
        let mut indices = (start..end).collect::<Vec<_>>();
        // DSI `fa0` is expected to be the dominant peak, so fixel-derived
        // exports sort each voxel by the chosen amplitude before writing.
        indices.sort_by(|&a, &b| amplitudes[b].total_cmp(&amplitudes[a]));
        for idx in indices {
            directions.push(odx.directions()[idx]);
            out_amplitude.push(amplitudes[idx]);
        }
        offsets.push(directions.len() as u32);
    }

    Ok((offsets, directions, out_amplitude))
}

fn peaks_from_sampled_odf(
    odx: &OdxDataset,
    odf_rows: &[f32],
) -> (Vec<u32>, Vec<[f32; 3]>, Vec<f32>) {
    let neighbors = hemisphere_neighbors();
    let hemisphere = dsistudio_odf8::hemisphere_vertices_ras();
    let ncols = hemisphere.len();
    let mut offsets = Vec::with_capacity(odx.nb_voxels() + 1);
    let mut directions = Vec::new();
    let mut amplitudes = Vec::new();
    offsets.push(0);

    for voxel in 0..odx.nb_voxels() {
        let row = &odf_rows[voxel * ncols..(voxel + 1) * ncols];
        let mut maxima = Vec::new();
        for vertex in 0..ncols {
            let value = row[vertex];
            if value <= 0.0 {
                continue;
            }
            if neighbors[vertex].iter().all(|&nbr| value >= row[nbr]) {
                maxima.push((value, vertex));
            }
        }
        maxima.sort_by(|a, b| b.0.total_cmp(&a.0));
        if maxima.is_empty() {
            let (best_idx, best_val) = row
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap_or((0, 0.0));
            maxima.push((best_val.max(1e-6), best_idx));
        }
        for (value, idx) in maxima.into_iter().take(5) {
            directions.push(hemisphere[idx]);
            amplitudes.push(value.max(1e-6));
        }
        offsets.push(directions.len() as u32);
    }
    (offsets, directions, amplitudes)
}

fn hemisphere_neighbors() -> &'static [Vec<usize>] {
    static NEIGHBORS: std::sync::OnceLock<Vec<Vec<usize>>> = std::sync::OnceLock::new();
    NEIGHBORS.get_or_init(|| {
        let hemisphere_len = dsistudio_odf8::hemisphere_vertices_ras().len();
        let mut neighbors = vec![HashSet::new(); hemisphere_len];
        for face in dsistudio_odf8::faces() {
            let ids = [face[0] as usize, face[1] as usize, face[2] as usize];
            if ids.iter().all(|&idx| idx < hemisphere_len) {
                for i in 0..3 {
                    let a = ids[i];
                    let b = ids[(i + 1) % 3];
                    neighbors[a].insert(b);
                    neighbors[b].insert(a);
                }
            }
        }
        neighbors
            .into_iter()
            .map(|set| set.into_iter().collect::<Vec<_>>())
            .collect()
    })
}

fn replace_peaks(
    parts: &mut OdxParts,
    offsets: Vec<u32>,
    directions: Vec<[f32; 3]>,
    amplitudes: Vec<f32>,
) {
    parts.header.nb_peaks = directions.len() as u64;
    parts.offsets_backing = MmapBacking::Owned(vec_to_bytes(offsets));
    parts.directions_backing = MmapBacking::Owned(vec_to_bytes(directions));
    parts.dpf.insert(
        "amplitude".into(),
        DataArray::owned_bytes(vec_to_bytes(amplitudes), 1, DType::Float32),
    );
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;
    use crate::formats::mat4;
    use crate::stream::OdxBuilder;

    #[test]
    fn mrtrix_to_dsistudio_export_flips_xy_before_fortran_flattening() {
        let dims = [2u64, 2, 1];
        let mask = vec![1u8; 4];
        let mut builder = OdxBuilder::new(crate::Header::identity_affine(), dims, mask);
        for _ in 0..4 {
            builder.push_voxel_peaks(&[[1.0, 0.0, 0.0]]);
        }
        builder.set_dpf_data(
            "amplitude",
            bytemuck::cast_slice(&[10.0f32, 20.0, 30.0, 40.0]).to_vec(),
            1,
            DType::Float32,
        );
        let odx = builder.finalize().unwrap();

        let prepared = prepare_dsistudio_dataset(
            &odx,
            &MrtrixToDsistudioOptions {
                output_format: DsistudioFormat::Fz,
                dense_odf_mode: DenseOdfMode::Off,
                peak_source: PeakSource::Fixels,
                ..Default::default()
            },
        )
        .unwrap();

        let tmp = tempdir().unwrap();
        let out = tmp.path().join("xy_flip.fz");
        dsistudio::save_fz(&prepared, &out).unwrap();

        let mat = mat4::read_mat4_gz(&out).unwrap();
        assert_eq!(
            mat.get("fa0").unwrap().as_f32_vec(),
            vec![40.0, 20.0, 30.0, 10.0]
        );
        // identity affine + x/y flip from the voxel reorientation
        // → reoriented RAS [[-1,0,0,1],[0,-1,0,1],[0,0,1,0]]
        assert_eq!(
            mat.get("trans").unwrap().as_f32_vec(),
            vec![-1.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,]
        );
    }
}

fn apply_z0_policy(parts: &mut OdxParts, policy: Z0Policy, format: DsistudioFormat) {
    match policy {
        Z0Policy::Never => {
            parts.header.extra.remove("Z0");
        }
        Z0Policy::Always => {
            if let Some(z0) = compute_dsi_z0(parts) {
                let (key, value) = number_value("Z0", z0);
                parts.header.extra.insert(key, value);
            }
        }
        Z0Policy::Auto => match format {
            DsistudioFormat::Fz => {
                // DSI Studio's own `.fz` conversion path often omits `z0`, so we
                // treat it as optional compatibility metadata rather than a
                // required dense-ODF scaling field.
                parts.header.extra.remove("Z0");
            }
            DsistudioFormat::FibGz => {
                if let Some(z0) = compute_dsi_z0(parts) {
                    let (key, value) = number_value("Z0", z0);
                    parts.header.extra.insert(key, value);
                } else {
                    parts.header.extra.remove("Z0");
                }
            }
        },
    }
}

fn compute_dsi_z0(parts: &OdxParts) -> Option<f32> {
    let amp = parts.dpf.get("amplitude")?.to_f32_vec().ok()?;
    let offsets: &[u32] = parts.offsets_backing.cast_slice();
    let mut max_fa0 = 0.0f32;
    for voxel in 0..parts.header.nb_voxels as usize {
        let start = offsets[voxel] as usize;
        if start < offsets[voxel + 1] as usize {
            max_fa0 = max_fa0.max(amp[start]);
        }
    }
    (max_fa0 > 0.0).then_some(1.0 / max_fa0)
}

fn number_value(key: &str, value: f32) -> (String, Value) {
    (
        key.to_string(),
        Value::Number(
            serde_json::Number::from_f64(value as f64)
                .unwrap_or_else(|| serde_json::Number::from(0)),
        ),
    )
}
