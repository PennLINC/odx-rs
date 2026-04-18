use std::collections::HashMap;
use std::ffi::OsStr;
use std::io::Read;
use std::io::Write;
use std::path::{Path, PathBuf};

use nalgebra::Matrix3;
use ndarray::{Array, IxDyn};
use nifti::{NiftiHeader, NiftiType};

use crate::data_array::DataArray;
use crate::dtype::DType;
use crate::error::{OdxError, Result};
use crate::formats::{dsistudio_odf8, mif};
use crate::header::{CanonicalDenseRepresentation, Header};
use crate::mmap_backing::{vec_into_bytes, MmapBacking};
use crate::mrtrix_sh;
use crate::odx_file::OdxParts;
use crate::stream::OdxBuilder;
use crate::OdxDataset;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MrtrixMaskPolicy {
    FixelSupport,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MrtrixShContainer {
    Mif,
    Nifti1,
    Nifti2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MrtrixFixelContainer {
    Mif,
    Nifti,
}

#[derive(Debug, Clone)]
pub struct MrtrixDatasetLoadOptions {
    pub mask_policy: MrtrixMaskPolicy,
}

impl Default for MrtrixDatasetLoadOptions {
    fn default() -> Self {
        Self {
            mask_policy: MrtrixMaskPolicy::FixelSupport,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MrtrixShWriteOptions {
    pub array_name: String,
    pub container: MrtrixShContainer,
    pub gzip: bool,
}

impl Default for MrtrixShWriteOptions {
    fn default() -> Self {
        Self {
            array_name: "coefficients".into(),
            container: MrtrixShContainer::Mif,
            gzip: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MrtrixFixelWriteOptions {
    pub container: MrtrixFixelContainer,
    pub include_dpf: bool,
    pub include_dpv: bool,
}

impl Default for MrtrixFixelWriteOptions {
    fn default() -> Self {
        Self {
            container: MrtrixFixelContainer::Nifti,
            include_dpf: true,
            include_dpv: false,
        }
    }
}

#[derive(Debug, Clone)]
struct LoadedF32Image {
    dims: Vec<usize>,
    affine: [[f64; 4]; 4],
    data: Vec<f32>,
}

#[derive(Debug, Clone)]
struct LoadedU32Image {
    dims: Vec<usize>,
    affine: [[f64; 4]; 4],
    data: Vec<u32>,
}

#[derive(Debug)]
struct CanonicalFixels {
    affine: [[f64; 4]; 4],
    dims: [u64; 3],
    mask: Vec<u8>,
    counts: Vec<usize>,
    directions: Vec<[f32; 3]>,
    dpf: HashMap<String, (Vec<f32>, usize)>,
    dpv: HashMap<String, (Vec<f32>, usize)>,
}

pub fn load_mrtrix(path_or_dir: &Path) -> Result<OdxDataset> {
    if path_or_dir.is_dir() {
        load_mrtrix_fixels(path_or_dir)
    } else {
        load_mrtrix_sh(path_or_dir)
    }
}

pub fn load_mrtrix_sh(path: &Path) -> Result<OdxDataset> {
    load_mrtrix_dataset(Some(path), None)
}

pub fn load_mrtrix_fixels(dir: &Path) -> Result<OdxDataset> {
    load_mrtrix_dataset(None, Some(dir))
}

pub fn load_mrtrix_dataset(sh_path: Option<&Path>, fixel_dir: Option<&Path>) -> Result<OdxDataset> {
    load_mrtrix_dataset_with_options(sh_path, fixel_dir, &MrtrixDatasetLoadOptions::default())
}

pub fn load_mrtrix_dataset_with_options(
    sh_path: Option<&Path>,
    fixel_dir: Option<&Path>,
    options: &MrtrixDatasetLoadOptions,
) -> Result<OdxDataset> {
    let _ = options;
    let fixels = if let Some(dir) = fixel_dir {
        Some(load_canonical_fixels(dir)?)
    } else {
        None
    };
    let sh = if let Some(path) = sh_path {
        Some(load_f32_image(path)?)
    } else {
        None
    };

    match (sh, fixels) {
        (None, None) => Err(OdxError::Argument(
            "load_mrtrix_dataset requires at least one MRtrix input".into(),
        )),
        (Some(sh), None) => build_sh_only_dataset(sh),
        (None, Some(fixels)) => build_fixels_dataset(fixels, None),
        (Some(sh), Some(fixels)) => {
            let sh_dims3 = image_dims3(&sh.dims)?;
            if sh_dims3 != fixels.dims {
                return Err(OdxError::Format(format!(
                    "MRtrix SH dimensions {:?} do not match fixel dimensions {:?}",
                    sh_dims3, fixels.dims
                )));
            }
            ensure_affines_match(&sh.affine, &fixels.affine, 1e-3)?;
            build_fixels_dataset(fixels, Some(sh))
        }
    }
}

pub fn save_mrtrix_sh(odx: &OdxDataset, path: &Path, options: &MrtrixShWriteOptions) -> Result<()> {
    let sh = odx
        .sh_arrays()
        .get(&options.array_name)
        .ok_or_else(|| OdxError::Argument(format!("no SH array named '{}'", options.array_name)))?;
    let basis = odx.header().sh_basis.as_deref().unwrap_or("");
    if basis != "tournier07" {
        return Err(OdxError::Argument(format!(
            "MRtrix SH export requires SH_BASIS=tournier07, found '{basis}'"
        )));
    }

    let dims3 = odx.header().dimensions.map(|d| d as usize);
    let ncoeffs = sh.ncols();
    let nb_voxels = odx.nb_voxels();
    if sh.nrows() != nb_voxels {
        return Err(OdxError::Format(format!(
            "SH array '{}' has {} rows but dataset has {} voxels",
            options.array_name,
            sh.nrows(),
            nb_voxels
        )));
    }

    let mut full = vec![0.0f32; dims3[0] * dims3[1] * dims3[2] * ncoeffs];
    let mask = odx.mask();
    let sh_values = sh.to_f32_vec()?;
    let mut voxel_row = 0usize;
    for (flat_idx, &m) in mask.iter().enumerate() {
        if m == 0 {
            continue;
        }
        let dst = flat_idx * ncoeffs;
        let src = voxel_row * ncoeffs;
        full[dst..dst + ncoeffs].copy_from_slice(&sh_values[src..src + ncoeffs]);
        voxel_row += 1;
    }

    let dims = vec![dims3[0], dims3[1], dims3[2], ncoeffs];
    match options.container {
        MrtrixShContainer::Mif => {
            let out = if path.extension() == Some(OsStr::new("gz")) || options.gzip {
                ensure_ext(path, "mif.gz")
            } else {
                ensure_ext(path, "mif")
            };
            write_mif_f32(&out, &dims, &odx.header().voxel_to_rasmm, &full)
        }
        MrtrixShContainer::Nifti1 => {
            let out = if path.extension() == Some(OsStr::new("gz")) || options.gzip {
                ensure_ext(path, "nii.gz")
            } else {
                ensure_ext(path, "nii")
            };
            write_mrtrix_nifti1_f32(&out, &dims, &odx.header().voxel_to_rasmm, &full)
        }
        MrtrixShContainer::Nifti2 => {
            let out = if path.extension() == Some(OsStr::new("gz")) || options.gzip {
                ensure_ext(path, "nii.gz")
            } else {
                ensure_ext(path, "nii")
            };
            write_mrtrix_nifti2_f32(&out, &dims, &odx.header().voxel_to_rasmm, &full)
        }
    }
}

pub fn save_mrtrix_fixels(
    odx: &OdxDataset,
    dir: &Path,
    options: &MrtrixFixelWriteOptions,
) -> Result<()> {
    std::fs::create_dir_all(dir)?;

    let dims3 = odx.header().dimensions.map(|d| d as usize);
    let mask = odx.mask();
    let offsets = odx.offsets();
    let mut index = vec![0u32; dims3[0] * dims3[1] * dims3[2] * 2];
    let mut voxel_row = 0usize;
    let mut running_offset = 0u32;
    for (flat_idx, &m) in mask.iter().enumerate() {
        let base = flat_idx * 2;
        if m != 0 {
            let count = (offsets[voxel_row + 1] - offsets[voxel_row]) as u32;
            index[base] = count;
            index[base + 1] = running_offset;
            running_offset += count;
            voxel_row += 1;
        }
    }

    let dirs_flat: Vec<f32> = odx
        .directions()
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();

    match options.container {
        MrtrixFixelContainer::Mif => {
            write_mif_u32(
                &dir.join("index.mif"),
                &[dims3[0], dims3[1], dims3[2], 2],
                &odx.header().voxel_to_rasmm,
                &index,
            )?;
            write_mif_f32(
                &dir.join("directions.mif"),
                &[odx.nb_peaks(), 3, 1],
                &odx.header().voxel_to_rasmm,
                &dirs_flat,
            )?;
        }
        MrtrixFixelContainer::Nifti => {
            write_mrtrix_nifti2_u32(
                &dir.join("index.nii"),
                &[dims3[0], dims3[1], dims3[2], 2],
                &odx.header().voxel_to_rasmm,
                &index,
            )?;
            write_mrtrix_nifti2_f32(
                &dir.join("directions.nii"),
                &[odx.nb_peaks(), 3, 1],
                &odx.header().voxel_to_rasmm,
                &dirs_flat,
            )?;
        }
    }

    if options.include_dpf {
        for (name, info) in odx.iter_dpf() {
            let arr = odx
                .dpf_arrays()
                .get(name)
                .ok_or_else(|| OdxError::Argument(format!("missing DPF array '{name}'")))?;
            let values = arr.to_f32_vec()?;
            let dims = [arr.nrows(), info.ncols, 1];
            match options.container {
                MrtrixFixelContainer::Mif => {
                    write_mif_f32(
                        &dir.join(format!("{name}.mif")),
                        &dims,
                        &odx.header().voxel_to_rasmm,
                        &values,
                    )?;
                }
                MrtrixFixelContainer::Nifti => {
                    write_mrtrix_nifti2_f32(
                        &dir.join(format!("{name}.nii")),
                        &dims,
                        &odx.header().voxel_to_rasmm,
                        &values,
                    )?;
                }
            }
        }
    }

    if options.include_dpv {
        for (name, info) in odx.iter_dpv() {
            let arr = odx
                .dpv_arrays()
                .get(name)
                .ok_or_else(|| OdxError::Argument(format!("missing DPV array '{name}'")))?;
            let values = arr.to_f32_vec()?;
            let mut full = vec![0.0f32; dims3[0] * dims3[1] * dims3[2] * info.ncols];
            let mut row = 0usize;
            for (flat_idx, &m) in mask.iter().enumerate() {
                if m == 0 {
                    continue;
                }
                let dst = flat_idx * info.ncols;
                let src = row * info.ncols;
                full[dst..dst + info.ncols].copy_from_slice(&values[src..src + info.ncols]);
                row += 1;
            }
            let dims = if info.ncols == 1 {
                vec![dims3[0], dims3[1], dims3[2]]
            } else {
                vec![dims3[0], dims3[1], dims3[2], info.ncols]
            };
            match options.container {
                MrtrixFixelContainer::Mif => {
                    write_mif_f32(
                        &dir.join(format!("{name}.mif")),
                        &dims,
                        &odx.header().voxel_to_rasmm,
                        &full,
                    )?;
                }
                MrtrixFixelContainer::Nifti => {
                    write_mrtrix_nifti2_f32(
                        &dir.join(format!("{name}.nii")),
                        &dims,
                        &odx.header().voxel_to_rasmm,
                        &full,
                    )?;
                }
            }
        }
    }

    Ok(())
}

fn build_sh_only_dataset(sh: LoadedF32Image) -> Result<OdxDataset> {
    let dims3 = image_dims3(&sh.dims)?;
    let voxel_count = dims3[0] as usize * dims3[1] as usize * dims3[2] as usize;
    let ncoeffs = *sh
        .dims
        .get(3)
        .ok_or_else(|| OdxError::Format("MRtrix SH image must be 4D".into()))?;
    let order = infer_sh_order(ncoeffs)?;
    let sample_plan =
        mrtrix_sh::RowSamplePlan::for_sh_rows_nonnegative(dsistudio_odf8::hemisphere_vertices_ras(), ncoeffs)?;
    let mut sampled = vec![0.0f32; sample_plan.ndir()];
    let mut mask = vec![0u8; voxel_count];
    let mut masked = Vec::with_capacity(sh.data.len());
    for voxel in 0..voxel_count {
        let start = voxel * ncoeffs;
        let end = start + ncoeffs;
        let row = &sh.data[start..end];
        sample_plan.apply_row_into(row, &mut sampled);
        let amplitude_sum: f32 = sampled.iter().copied().sum();
        if amplitude_sum > 1e-6 {
            mask[voxel] = 1;
            masked.extend_from_slice(row);
        }
    }
    let masked_voxel_count = mask.iter().filter(|&&m| m != 0).count();
    let mut sh_arrays = HashMap::new();
    sh_arrays.insert(
        "coefficients".to_string(),
        DataArray::owned_bytes(vec_into_bytes(masked), ncoeffs, DType::Float32),
    );

    Ok(OdxDataset::from_parts(OdxParts {
        header: Header {
            voxel_to_rasmm: sh.affine,
            dimensions: dims3,
            nb_voxels: masked_voxel_count as u64,
            nb_peaks: 0,
            nb_sphere_vertices: None,
            nb_sphere_faces: None,
            sh_order: Some(order),
            sh_basis: Some("tournier07".into()),
            canonical_dense_representation: Some(CanonicalDenseRepresentation::Sh),
            sphere_id: None,
            odf_sample_domain: None,
            array_quantization: HashMap::new(),
            extra: HashMap::new(),
        },
        mask_backing: MmapBacking::Owned(mask),
        offsets_backing: MmapBacking::Owned(vec_into_bytes(vec![0u32; masked_voxel_count + 1])),
        directions_backing: MmapBacking::Owned(vec_into_bytes(Vec::<[f32; 3]>::new())),
        sphere_vertices: None,
        sphere_faces: None,
        odf: HashMap::new(),
        sh: sh_arrays,
        dpv: HashMap::new(),
        dpf: HashMap::new(),
        groups: HashMap::new(),
        dpg: HashMap::new(),
        tempdir: None,
    }))
}

fn build_fixels_dataset(fixels: CanonicalFixels, sh: Option<LoadedF32Image>) -> Result<OdxDataset> {
    let CanonicalFixels {
        affine,
        dims,
        mask,
        counts,
        directions,
        dpf,
        dpv,
    } = fixels;

    let mut builder = OdxBuilder::new(affine, dims, mask.clone());
    let mut peak_cursor = 0usize;
    for &count in &counts {
        let next = peak_cursor + count;
        builder.push_voxel_peaks(&directions[peak_cursor..next]);
        peak_cursor = next;
    }

    for (name, (values, ncols)) in dpf {
        builder.set_dpf_data(&name, vec_into_bytes(values), ncols, DType::Float32);
    }
    for (name, (values, ncols)) in dpv {
        builder.set_dpv_data(&name, vec_into_bytes(values), ncols, DType::Float32);
    }

    if let Some(sh_image) = sh {
        let ncoeffs = *sh_image
            .dims
            .get(3)
            .ok_or_else(|| OdxError::Format("MRtrix SH image must be 4D".into()))?;
        let order = infer_sh_order(ncoeffs)?;
        let masked_rows = mask.iter().filter(|&&m| m != 0).count();
        let mut masked = vec![0.0f32; masked_rows * ncoeffs];
        let voxels = (dims[0] * dims[1] * dims[2]) as usize;
        let mut masked_row = 0usize;
        for flat_idx in 0..voxels {
            if mask[flat_idx] == 0 {
                continue;
            }
            let start = flat_idx * ncoeffs;
            let dst = masked_row * ncoeffs;
            masked[dst..dst + ncoeffs].copy_from_slice(&sh_image.data[start..start + ncoeffs]);
            masked_row += 1;
        }
        builder.set_sh_info(order, "tournier07".into());
        builder.set_sh_data(
            "coefficients",
            vec_into_bytes(masked),
            ncoeffs,
            DType::Float32,
        );
        builder.set_canonical_dense_representation(CanonicalDenseRepresentation::Sh);
    }

    builder.finalize()
}

fn load_canonical_fixels(dir: &Path) -> Result<CanonicalFixels> {
    let index_path = find_required_file(dir, "index")?;
    let directions_path = find_required_file(dir, "directions")?;

    let index = load_u32_image(&index_path)?;
    if index.dims.len() != 4 || index.dims[3] != 2 {
        return Err(OdxError::Format(format!(
            "MRtrix fixel index image must have shape (x,y,z,2), found {:?}",
            index.dims
        )));
    }
    let dims3 = image_dims3(&index.dims)?;
    let voxel_count = dims3[0] as usize * dims3[1] as usize * dims3[2] as usize;
    let mut mask = vec![0u8; voxel_count];
    let mut counts = Vec::new();
    let mut firsts = Vec::new();
    let mut total_fixels = 0usize;
    for voxel in 0..voxel_count {
        let count = index.data[voxel * 2] as usize;
        let first = index.data[voxel * 2 + 1] as usize;
        if count > 0 {
            mask[voxel] = 1;
            counts.push(count);
            firsts.push(first);
            total_fixels = total_fixels.max(first + count);
        }
    }

    let dirs = load_f32_image(&directions_path)?;
    if dirs.dims.len() != 3 || dirs.dims[1] != 3 || dirs.dims[2] != 1 {
        return Err(OdxError::Format(format!(
            "MRtrix directions file must have shape (n,3,1), found {:?}",
            dirs.dims
        )));
    }
    if dirs.dims[0] < total_fixels {
        return Err(OdxError::Format(format!(
            "MRtrix directions file has {} fixels but index references {}",
            dirs.dims[0], total_fixels
        )));
    }

    let mut directions = Vec::with_capacity(total_fixels);
    for (&count, &first) in counts.iter().zip(firsts.iter()) {
        for row in first..first + count {
            let base = row * 3;
            directions.push([dirs.data[base], dirs.data[base + 1], dirs.data[base + 2]]);
        }
    }

    let mut dpf = HashMap::new();
    let mut dpv = HashMap::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path == index_path || path == directions_path || path.is_dir() {
            continue;
        }
        if !is_mif_path(&path) && !is_nifti_path(&path) {
            continue;
        }
        let stem = path_stem_without_image_ext(&path)?;
        let image = load_f32_image(&path)?;
        if image.dims.len() >= 3 && image.dims[0] == total_fixels && image.dims[2] == 1 {
            let ncols = image.dims[1];
            let mut values = Vec::with_capacity(total_fixels * ncols);
            // MRtrix fixel directories are semantically ordered by
            // index[...,0]=count and index[...,1]=first_index, not by the raw
            // global row order of each payload file. Rebuilding the arrays via
            // `index` is what makes MIF and NIfTI fixel directories agree.
            for (&count, &first) in counts.iter().zip(firsts.iter()) {
                let start = first * ncols;
                let end = (first + count) * ncols;
                values.extend_from_slice(&image.data[start..end]);
            }
            dpf.insert(stem, (values, ncols));
        } else if image.dims.len() >= 3
            && image.dims[0] == dims3[0] as usize
            && image.dims[1] == dims3[1] as usize
            && image.dims[2] == dims3[2] as usize
        {
            let ncols = image.dims[3..].iter().product::<usize>().max(1);
            let mut values = Vec::new();
            for voxel in 0..voxel_count {
                if mask[voxel] == 0 {
                    continue;
                }
                let start = voxel * ncols;
                values.extend_from_slice(&image.data[start..start + ncols]);
            }
            dpv.insert(stem, (values, ncols));
        }
    }

    Ok(CanonicalFixels {
        affine: index.affine,
        dims: dims3,
        mask,
        counts,
        directions,
        dpf,
        dpv,
    })
}

fn load_f32_image(path: &Path) -> Result<LoadedF32Image> {
    if is_mif_path(path) {
        let mut loaded = load_mif_f32_image(path)?;
        canonicalize_spatial_axes_to_ras_f32(&mut loaded);
        Ok(loaded)
    } else if is_nifti_path(path) {
        load_nifti_f32(path)
    } else {
        Err(OdxError::Argument(format!(
            "unsupported MRtrix image path '{}'",
            path.display()
        )))
    }
}

fn load_u32_image(path: &Path) -> Result<LoadedU32Image> {
    if is_mif_path(path) {
        let mut loaded = load_mif_u32_image(path)?;
        canonicalize_spatial_axes_to_ras_u32(&mut loaded);
        Ok(loaded)
    } else if is_nifti_path(path) {
        load_nifti_u32(path)
    } else {
        Err(OdxError::Argument(format!(
            "unsupported MRtrix image path '{}'",
            path.display()
        )))
    }
}

fn load_mif_f32_image(path: &Path) -> Result<LoadedF32Image> {
    let bytes = mif::read_mif_bytes(path)?;
    let header = mif::parse_mif_header(&bytes)?;
    let payload = bytes.get(header.data_offset..).ok_or_else(|| {
        OdxError::Format(format!(
            "MIF data offset {} exceeds file length in '{}'",
            header.data_offset,
            path.display()
        ))
    })?;
    Ok(LoadedF32Image {
        dims: header.dimensions.clone(),
        affine: header.affine_4x4(),
        data: decode_mif_real_to_logical_f32(payload, &header)?,
    })
}

fn load_mif_u32_image(path: &Path) -> Result<LoadedU32Image> {
    let bytes = mif::read_mif_bytes(path)?;
    let header = mif::parse_mif_header(&bytes)?;
    let payload = bytes.get(header.data_offset..).ok_or_else(|| {
        OdxError::Format(format!(
            "MIF data offset {} exceeds file length in '{}'",
            header.data_offset,
            path.display()
        ))
    })?;
    Ok(LoadedU32Image {
        dims: header.dimensions.clone(),
        affine: header.affine_4x4(),
        data: decode_mif_u32_to_logical(payload, &header)?,
    })
}

fn decode_mif_real_to_logical_f32(payload: &[u8], header: &mif::MifHeader) -> Result<Vec<f32>> {
    if !(header.datatype.starts_with("Float32") || header.datatype.starts_with("Float64")) {
        return Err(OdxError::Format(format!(
            "MRtrix float image requires Float32 or Float64 MIF data, found {}",
            header.datatype
        )));
    }

    let total: usize = header.dimensions.iter().product();
    let elem_size = header.element_size();
    let expected_bytes = total
        .checked_mul(elem_size)
        .ok_or_else(|| OdxError::Format("MIF payload byte size overflow".into()))?;
    if payload.len() < expected_bytes {
        return Err(OdxError::Format(format!(
            "MIF payload shorter than expected: {} < {}",
            payload.len(),
            expected_bytes
        )));
    }

    let strides = header.compute_strides();
    if header.is_native_endian()
        && header.datatype.starts_with("Float32")
        && strides_match_c_order(&header.dimensions, &strides)
    {
        return Ok(payload[..expected_bytes]
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect());
    }

    decode_mif_logical_values(&header.dimensions, &strides, elem_size, |offset| {
        read_mif_float_as_f32(payload, offset, &header.datatype)
    })
}

fn decode_mif_u32_to_logical(payload: &[u8], header: &mif::MifHeader) -> Result<Vec<u32>> {
    if !header.datatype.starts_with("UInt32") {
        return Err(OdxError::Format(format!(
            "MRtrix integer image requires UInt32 MIF data, found {}",
            header.datatype
        )));
    }

    let total: usize = header.dimensions.iter().product();
    let elem_size = header.element_size();
    let expected_bytes = total
        .checked_mul(elem_size)
        .ok_or_else(|| OdxError::Format("MIF payload byte size overflow".into()))?;
    if payload.len() < expected_bytes {
        return Err(OdxError::Format(format!(
            "MIF payload shorter than expected: {} < {}",
            payload.len(),
            expected_bytes
        )));
    }

    let strides = header.compute_strides();
    if header.is_native_endian() && strides_match_c_order(&header.dimensions, &strides) {
        return Ok(payload[..expected_bytes]
            .chunks_exact(4)
            .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect());
    }

    decode_mif_logical_values(&header.dimensions, &strides, elem_size, |offset| {
        read_mif_u32(payload, offset, &header.datatype)
    })
}

fn decode_mif_logical_values<T, F>(
    dims: &[usize],
    strides: &[isize],
    elem_size: usize,
    mut decode_at: F,
) -> Result<Vec<T>>
where
    F: FnMut(usize) -> Result<T>,
{
    let total: usize = dims.iter().product();
    if total == 0 {
        return Ok(Vec::new());
    }

    let base_offset: isize = dims
        .iter()
        .zip(strides.iter())
        .map(|(&dim, &stride)| {
            if stride < 0 {
                (dim as isize - 1) * (-stride)
            } else {
                0
            }
        })
        .sum();

    let mut coords = vec![0usize; dims.len()];
    let mut logical = Vec::with_capacity(total);
    for _ in 0..total {
        let mut raw_index = base_offset;
        for (&coord, &stride) in coords.iter().zip(strides.iter()) {
            raw_index += coord as isize * stride;
        }
        logical.push(decode_at(raw_index as usize * elem_size)?);
        for axis in (0..dims.len()).rev() {
            coords[axis] += 1;
            if coords[axis] < dims[axis] {
                break;
            }
            coords[axis] = 0;
        }
    }
    Ok(logical)
}

fn strides_match_c_order(dims: &[usize], strides: &[isize]) -> bool {
    if dims.len() != strides.len() {
        return false;
    }
    let mut expected = vec![0isize; dims.len()];
    let mut stride = 1isize;
    for axis in (0..dims.len()).rev() {
        expected[axis] = stride;
        stride *= dims[axis] as isize;
    }
    expected == strides
}

fn read_mif_float_as_f32(payload: &[u8], offset: usize, datatype: &str) -> Result<f32> {
    match datatype {
        "Float32LE" | "Float32" => Ok(f32::from_le_bytes(read_4(payload, offset)?)),
        "Float32BE" => Ok(f32::from_be_bytes(read_4(payload, offset)?)),
        "Float64LE" | "Float64" => Ok(f64::from_le_bytes(read_8(payload, offset)?) as f32),
        "Float64BE" => Ok(f64::from_be_bytes(read_8(payload, offset)?) as f32),
        _ => Err(OdxError::Format(format!(
            "unsupported MIF floating datatype {}",
            datatype
        ))),
    }
}

fn read_mif_u32(payload: &[u8], offset: usize, datatype: &str) -> Result<u32> {
    match datatype {
        "UInt32LE" | "UInt32" => Ok(u32::from_le_bytes(read_4(payload, offset)?)),
        "UInt32BE" => Ok(u32::from_be_bytes(read_4(payload, offset)?)),
        _ => Err(OdxError::Format(format!(
            "unsupported MIF u32 datatype {}",
            datatype
        ))),
    }
}

fn read_4(payload: &[u8], offset: usize) -> Result<[u8; 4]> {
    payload
        .get(offset..offset + 4)
        .ok_or_else(|| OdxError::Format("short MIF payload while reading 4-byte value".into()))?
        .try_into()
        .map_err(|_| OdxError::Format("short MIF payload while reading 4-byte value".into()))
}

fn read_8(payload: &[u8], offset: usize) -> Result<[u8; 8]> {
    payload
        .get(offset..offset + 8)
        .ok_or_else(|| OdxError::Format("short MIF payload while reading 8-byte value".into()))?
        .try_into()
        .map_err(|_| OdxError::Format("short MIF payload while reading 8-byte value".into()))
}

fn load_nifti_f32(path: &Path) -> Result<LoadedF32Image> {
    let image = read_nifti(path, ExpectedNiftiType::Float32)?;
    let mut loaded = LoadedF32Image {
        dims: image.dims,
        affine: image.affine,
        data: image.f32_data.ok_or_else(|| {
            OdxError::Format(format!("expected float32 NIfTI '{}'", path.display()))
        })?,
    };
    canonicalize_spatial_axes_to_ras_f32(&mut loaded);
    Ok(loaded)
}

fn load_nifti_u32(path: &Path) -> Result<LoadedU32Image> {
    let image = read_nifti(path, ExpectedNiftiType::UInt32)?;
    let mut loaded = LoadedU32Image {
        dims: image.dims,
        affine: image.affine,
        data: image.u32_data.ok_or_else(|| {
            OdxError::Format(format!("expected uint32 NIfTI '{}'", path.display()))
        })?,
    };
    canonicalize_spatial_axes_to_ras_u32(&mut loaded);
    Ok(loaded)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpectedNiftiType {
    Float32,
    UInt32,
}

#[derive(Debug)]
struct ParsedNifti {
    dims: Vec<usize>,
    affine: [[f64; 4]; 4],
    f32_data: Option<Vec<f32>>,
    u32_data: Option<Vec<u32>>,
}

fn read_nifti(path: &Path, expected: ExpectedNiftiType) -> Result<ParsedNifti> {
    let bytes = read_image_bytes(path)?;
    let sizeof_hdr = i32::from_le_bytes(bytes[0..4].try_into().unwrap());
    match sizeof_hdr {
        348 => parse_nifti1(&bytes, path, expected),
        540 => parse_nifti2(&bytes, path, expected),
        _ => Err(OdxError::Format(format!(
            "unsupported NIfTI header size {} in '{}'",
            sizeof_hdr,
            path.display()
        ))),
    }
}

fn parse_nifti1(
    path_bytes: &[u8],
    path: &Path,
    expected: ExpectedNiftiType,
) -> Result<ParsedNifti> {
    let datatype = read_i16_le(path_bytes, 70)?;
    let ndim = read_i16_le(path_bytes, 40)? as usize;
    let mut dims = Vec::with_capacity(ndim);
    for i in 0..ndim {
        dims.push(read_i16_le(path_bytes, 42 + i * 2)? as usize);
    }
    let vox_offset = read_f32_le(path_bytes, 108)? as usize;
    let affine = if read_i16_le(path_bytes, 254)? > 0 {
        [
            [
                read_f32_le(path_bytes, 280)? as f64,
                read_f32_le(path_bytes, 284)? as f64,
                read_f32_le(path_bytes, 288)? as f64,
                read_f32_le(path_bytes, 292)? as f64,
            ],
            [
                read_f32_le(path_bytes, 296)? as f64,
                read_f32_le(path_bytes, 300)? as f64,
                read_f32_le(path_bytes, 304)? as f64,
                read_f32_le(path_bytes, 308)? as f64,
            ],
            [
                read_f32_le(path_bytes, 312)? as f64,
                read_f32_le(path_bytes, 316)? as f64,
                read_f32_le(path_bytes, 320)? as f64,
                read_f32_le(path_bytes, 324)? as f64,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    } else {
        let px = read_f32_le(path_bytes, 80)? as f64;
        let py = read_f32_le(path_bytes, 84)? as f64;
        let pz = read_f32_le(path_bytes, 88)? as f64;
        [
            [px, 0.0, 0.0, 0.0],
            [0.0, py, 0.0, 0.0],
            [0.0, 0.0, pz, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    };

    parse_nifti_payload(
        path_bytes, path, expected, datatype, vox_offset, dims, affine,
    )
}

fn parse_nifti2(
    path_bytes: &[u8],
    path: &Path,
    expected: ExpectedNiftiType,
) -> Result<ParsedNifti> {
    let datatype = read_i16_le(path_bytes, 12)?;
    let ndim = read_i64_le(path_bytes, 16)? as usize;
    let mut dims = Vec::with_capacity(ndim);
    for i in 0..ndim {
        dims.push(read_i64_le(path_bytes, 24 + i * 8)? as usize);
    }
    let vox_offset = read_i64_le(path_bytes, 168)? as usize;
    let affine = if read_i32_le(path_bytes, 348)? > 0 {
        [
            [
                read_f64_le(path_bytes, 400)?,
                read_f64_le(path_bytes, 408)?,
                read_f64_le(path_bytes, 416)?,
                read_f64_le(path_bytes, 424)?,
            ],
            [
                read_f64_le(path_bytes, 432)?,
                read_f64_le(path_bytes, 440)?,
                read_f64_le(path_bytes, 448)?,
                read_f64_le(path_bytes, 456)?,
            ],
            [
                read_f64_le(path_bytes, 464)?,
                read_f64_le(path_bytes, 472)?,
                read_f64_le(path_bytes, 480)?,
                read_f64_le(path_bytes, 488)?,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    } else {
        let px = read_f64_le(path_bytes, 112)?;
        let py = read_f64_le(path_bytes, 120)?;
        let pz = read_f64_le(path_bytes, 128)?;
        [
            [px, 0.0, 0.0, 0.0],
            [0.0, py, 0.0, 0.0],
            [0.0, 0.0, pz, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    };

    parse_nifti_payload(
        path_bytes, path, expected, datatype, vox_offset, dims, affine,
    )
}

fn parse_nifti_payload(
    bytes: &[u8],
    path: &Path,
    expected: ExpectedNiftiType,
    datatype: i16,
    vox_offset: usize,
    dims: Vec<usize>,
    affine: [[f64; 4]; 4],
) -> Result<ParsedNifti> {
    let total: usize = dims.iter().product();
    match (expected, datatype) {
        (ExpectedNiftiType::Float32, 16) => {
            let raw = read_vec_f32(bytes, vox_offset, total)?;
            let data = reorder_from_fortran(raw, &dims);
            Ok(ParsedNifti {
                dims,
                affine,
                f32_data: Some(data),
                u32_data: None,
            })
        }
        (ExpectedNiftiType::UInt32, 768) => {
            let raw = read_vec_u32(bytes, vox_offset, total)?;
            let data = reorder_from_fortran(raw, &dims);
            Ok(ParsedNifti {
                dims,
                affine,
                f32_data: None,
                u32_data: Some(data),
            })
        }
        _ => Err(OdxError::Format(format!(
            "unsupported NIfTI datatype {} in '{}'",
            datatype,
            path.display()
        ))),
    }
}

fn canonicalize_spatial_axes_to_ras_f32(image: &mut LoadedF32Image) {
    let xform = spatial_ornt_transform_to_ras(image.affine);
    if orientation_is_identity(xform) {
        return;
    }
    let original_dims = image.dims.clone();
    image.data = reorient_spatial_axes(&image.data, &original_dims, xform);
    image.affine = compose_affines(
        image.affine,
        nibabel_inv_ornt_aff(xform, &original_dims[..3]),
    );
    image.dims = reoriented_dims(&original_dims, xform);
}

fn canonicalize_spatial_axes_to_ras_u32(image: &mut LoadedU32Image) {
    let xform = spatial_ornt_transform_to_ras(image.affine);
    if orientation_is_identity(xform) {
        return;
    }
    let original_dims = image.dims.clone();
    image.data = reorient_spatial_axes(&image.data, &original_dims, xform);
    image.affine = compose_affines(
        image.affine,
        nibabel_inv_ornt_aff(xform, &original_dims[..3]),
    );
    image.dims = reoriented_dims(&original_dims, xform);
}

fn mrtrix_nifti_axis_flips(dims: &[usize]) -> [bool; 3] {
    // MRtrix NIfTI fixel payloads arrive mirrored relative to MIF unless the
    // first two spatial axes are flipped back during import/export. This is a
    // container quirk of MRtrix fixel directories rather than a generic NIfTI
    // rule, so keep the correction local to the MRtrix backend.
    [
        dims.first().copied().unwrap_or(1) > 1,
        dims.get(1).copied().unwrap_or(1) > 1,
        false,
    ]
}

fn flip_spatial_axes_in_place<T: Copy + Default>(
    data: &mut Vec<T>,
    dims: &[usize],
    flips: [bool; 3],
) {
    if dims.len() < 3 {
        return;
    }
    let xdim = dims[0];
    let ydim = dims[1];
    let zdim = dims[2];
    let ncols = dims[3..].iter().product::<usize>().max(1);
    let mut out = vec![T::default(); data.len()];
    for x in 0..xdim {
        let sx = if flips[0] { xdim - 1 - x } else { x };
        for y in 0..ydim {
            let sy = if flips[1] { ydim - 1 - y } else { y };
            for z in 0..zdim {
                let sz = if flips[2] { zdim - 1 - z } else { z };
                let dst = ((x * ydim + y) * zdim + z) * ncols;
                let src = ((sx * ydim + sy) * zdim + sz) * ncols;
                out[dst..dst + ncols].copy_from_slice(&data[src..src + ncols]);
            }
        }
    }
    *data = out;
}

fn flip_spatial_axes_in_affine(
    mut affine: [[f64; 4]; 4],
    dims: &[usize],
    flips: [bool; 3],
) -> [[f64; 4]; 4] {
    for axis in 0..3 {
        if !flips[axis] {
            continue;
        }
        let shift = dims.get(axis).copied().unwrap_or(1).saturating_sub(1) as f64;
        let col = [affine[0][axis], affine[1][axis], affine[2][axis]];
        for row in 0..3 {
            affine[row][axis] = -col[row];
            affine[row][3] += col[row] * shift;
        }
    }
    affine
}

fn spatial_ornt_transform_to_ras(affine: [[f64; 4]; 4]) -> [[i8; 2]; 3] {
    // MRtrix realigns spatial axes to a near-RAS canonical image ordering
    // after decoding storage strides. Mirror that behavior here so MIF layout
    // handling and anatomical orientation stay as separate concerns.
    let start = nibabel_io_orientation(affine);
    let end = [[0, 1], [1, 1], [2, 1]];
    nibabel_ornt_transform(start, end)
}

fn orientation_is_identity(ornt: [[i8; 2]; 3]) -> bool {
    ornt == [[0, 1], [1, 1], [2, 1]]
}

fn reoriented_dims(dims: &[usize], ornt: [[i8; 2]; 3]) -> Vec<usize> {
    let mut out = dims.to_vec();
    for (src_axis, [dst_axis, _]) in ornt.into_iter().enumerate() {
        out[dst_axis as usize] = dims[src_axis];
    }
    out
}

fn reorient_spatial_axes<T: Copy + Default>(
    data: &[T],
    dims: &[usize],
    ornt: [[i8; 2]; 3],
) -> Vec<T> {
    if dims.len() < 3 {
        return data.to_vec();
    }
    let new_dims = reoriented_dims(dims, ornt);
    let old_spatial = [dims[0], dims[1], dims[2]];
    let new_spatial = [new_dims[0], new_dims[1], new_dims[2]];
    let ncols = dims[3..].iter().product::<usize>().max(1);
    let mut out = vec![T::default(); data.len()];

    for x in 0..old_spatial[0] {
        for y in 0..old_spatial[1] {
            for z in 0..old_spatial[2] {
                let old = [x, y, z];
                let mut new = [0usize; 3];
                for src_axis in 0..3 {
                    let mut coord = old[src_axis];
                    if ornt[src_axis][1] == -1 {
                        coord = old_spatial[src_axis] - 1 - coord;
                    }
                    new[ornt[src_axis][0] as usize] = coord;
                }
                let src = ((x * old_spatial[1] + y) * old_spatial[2] + z) * ncols;
                let dst = ((new[0] * new_spatial[1] + new[1]) * new_spatial[2] + new[2]) * ncols;
                out[dst..dst + ncols].copy_from_slice(&data[src..src + ncols]);
            }
        }
    }
    out
}

fn compose_affines(a: [[f64; 4]; 4], b: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut out = [[0.0f64; 4]; 4];
    for row in 0..4 {
        for col in 0..4 {
            out[row][col] = (0..4).map(|k| a[row][k] * b[k][col]).sum();
        }
    }
    out
}

fn nibabel_inv_ornt_aff(ornt: [[i8; 2]; 3], shape: &[usize]) -> [[f64; 4]; 4] {
    let shape = [shape[0] as f64, shape[1] as f64, shape[2] as f64];
    let center = [
        -(shape[0] - 1.0) / 2.0,
        -(shape[1] - 1.0) / 2.0,
        -(shape[2] - 1.0) / 2.0,
    ];

    let mut undo_reorder = [[0.0f64; 4]; 4];
    for row in 0..3 {
        undo_reorder[row][ornt[row][0] as usize] = 1.0;
    }
    undo_reorder[3][3] = 1.0;

    let mut undo_flip = [[0.0f64; 4]; 4];
    for axis in 0..3 {
        let flip = f64::from(ornt[axis][1]);
        undo_flip[axis][axis] = flip;
        undo_flip[axis][3] = (flip * center[axis]) - center[axis];
    }
    undo_flip[3][3] = 1.0;
    compose_affines(undo_flip, undo_reorder)
}

fn nibabel_io_orientation(aff: [[f64; 4]; 4]) -> [[i8; 2]; 3] {
    let rzs = Matrix3::new(
        aff[0][0], aff[0][1], aff[0][2], aff[1][0], aff[1][1], aff[1][2], aff[2][0], aff[2][1],
        aff[2][2],
    );
    let zooms = affine_column_norms(aff);
    let rs = Matrix3::new(
        rzs[(0, 0)] / f64::from(zooms[0].max(1e-12)),
        rzs[(0, 1)] / f64::from(zooms[1].max(1e-12)),
        rzs[(0, 2)] / f64::from(zooms[2].max(1e-12)),
        rzs[(1, 0)] / f64::from(zooms[0].max(1e-12)),
        rzs[(1, 1)] / f64::from(zooms[1].max(1e-12)),
        rzs[(1, 2)] / f64::from(zooms[2].max(1e-12)),
        rzs[(2, 0)] / f64::from(zooms[0].max(1e-12)),
        rzs[(2, 1)] / f64::from(zooms[1].max(1e-12)),
        rzs[(2, 2)] / f64::from(zooms[2].max(1e-12)),
    );
    let svd = rs.svd(true, true);
    let u = svd.u.expect("requested U from SVD");
    let vt = svd.v_t.expect("requested V^T from SVD");
    let s = svd.singular_values;
    let tol = s.max() * 3.0 * f64::EPSILON;
    let mut r = Matrix3::<f64>::zeros();
    for idx in 0..3 {
        if s[idx] > tol {
            let ui = u.column(idx);
            let vti = vt.row(idx);
            r += ui * vti;
        }
    }

    let mut in_axes = [0usize, 1, 2];
    in_axes.sort_by(|&a, &b| {
        let sa = (0..3).map(|row| r[(row, a)].powi(2)).fold(0.0, f64::max);
        let sb = (0..3).map(|row| r[(row, b)].powi(2)).fold(0.0, f64::max);
        sb.total_cmp(&sa)
    });

    let mut ornt = [[-1i8, 1i8]; 3];
    let mut work = r;
    for in_ax in in_axes {
        let mut out_ax = 0usize;
        let mut best = 0.0f64;
        for row in 0..3 {
            let value = work[(row, in_ax)].abs();
            if value > best {
                best = value;
                out_ax = row;
            }
        }
        ornt[in_ax][0] = out_ax as i8;
        ornt[in_ax][1] = if work[(out_ax, in_ax)] < 0.0 { -1 } else { 1 };
        for col in 0..3 {
            work[(out_ax, col)] = 0.0;
        }
    }
    ornt
}

fn nibabel_ornt_transform(start_ornt: [[i8; 2]; 3], end_ornt: [[i8; 2]; 3]) -> [[i8; 2]; 3] {
    let mut result = [[0i8, 1i8]; 3];
    for (end_in_idx, [end_out_idx, end_flip]) in end_ornt.into_iter().enumerate() {
        for (start_in_idx, [start_out_idx, start_flip]) in start_ornt.into_iter().enumerate() {
            if end_out_idx == start_out_idx {
                result[start_in_idx] = [
                    end_in_idx as i8,
                    if start_flip == end_flip { 1 } else { -1 },
                ];
                break;
            }
        }
    }
    result
}

fn affine_column_norms(aff: [[f64; 4]; 4]) -> [f32; 3] {
    [
        (aff[0][0] * aff[0][0] + aff[1][0] * aff[1][0] + aff[2][0] * aff[2][0]).sqrt() as f32,
        (aff[0][1] * aff[0][1] + aff[1][1] * aff[1][1] + aff[2][1] * aff[2][1]).sqrt() as f32,
        (aff[0][2] * aff[0][2] + aff[1][2] * aff[1][2] + aff[2][2] * aff[2][2]).sqrt() as f32,
    ]
}

fn infer_sh_order(ncoeffs: usize) -> Result<u64> {
    for order in (0..=20).step_by(2) {
        let coeffs = (order + 1) * (order + 2) / 2;
        if coeffs == ncoeffs {
            return Ok(order as u64);
        }
    }
    Err(OdxError::Format(format!(
        "cannot infer MRtrix SH order from {ncoeffs} coefficients"
    )))
}

fn image_dims3(dims: &[usize]) -> Result<[u64; 3]> {
    if dims.len() < 3 {
        return Err(OdxError::Format(format!(
            "MRtrix image must have at least 3 dimensions, found {:?}",
            dims
        )));
    }
    Ok([dims[0] as u64, dims[1] as u64, dims[2] as u64])
}

fn read_image_bytes(path: &Path) -> Result<Vec<u8>> {
    if path.extension() == Some(OsStr::new("gz")) {
        let file = std::fs::File::open(path)?;
        let mut decoder = flate2::read::MultiGzDecoder::new(file);
        let mut bytes = Vec::new();
        decoder.read_to_end(&mut bytes)?;
        Ok(bytes)
    } else {
        Ok(std::fs::read(path)?)
    }
}

fn read_i16_le(bytes: &[u8], offset: usize) -> Result<i16> {
    Ok(i16::from_le_bytes(
        bytes
            .get(offset..offset + 2)
            .ok_or_else(|| OdxError::Format("NIfTI header truncated".into()))?
            .try_into()
            .unwrap(),
    ))
}

fn read_i32_le(bytes: &[u8], offset: usize) -> Result<i32> {
    Ok(i32::from_le_bytes(
        bytes
            .get(offset..offset + 4)
            .ok_or_else(|| OdxError::Format("NIfTI header truncated".into()))?
            .try_into()
            .unwrap(),
    ))
}

fn read_i64_le(bytes: &[u8], offset: usize) -> Result<i64> {
    Ok(i64::from_le_bytes(
        bytes
            .get(offset..offset + 8)
            .ok_or_else(|| OdxError::Format("NIfTI header truncated".into()))?
            .try_into()
            .unwrap(),
    ))
}

fn read_f32_le(bytes: &[u8], offset: usize) -> Result<f32> {
    Ok(f32::from_le_bytes(
        bytes
            .get(offset..offset + 4)
            .ok_or_else(|| OdxError::Format("NIfTI header truncated".into()))?
            .try_into()
            .unwrap(),
    ))
}

fn read_f64_le(bytes: &[u8], offset: usize) -> Result<f64> {
    Ok(f64::from_le_bytes(
        bytes
            .get(offset..offset + 8)
            .ok_or_else(|| OdxError::Format("NIfTI header truncated".into()))?
            .try_into()
            .unwrap(),
    ))
}

fn read_vec_f32(bytes: &[u8], offset: usize, len: usize) -> Result<Vec<f32>> {
    let needed = offset + len * 4;
    let slice = bytes
        .get(offset..needed)
        .ok_or_else(|| OdxError::Format("NIfTI data truncated".into()))?;
    Ok(slice
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_vec_u32(bytes: &[u8], offset: usize, len: usize) -> Result<Vec<u32>> {
    let needed = offset + len * 4;
    let slice = bytes
        .get(offset..needed)
        .ok_or_else(|| OdxError::Format("NIfTI data truncated".into()))?;
    Ok(slice
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn ensure_affines_match(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4], tol: f64) -> Result<()> {
    let mut max_diff = 0.0f64;
    for row in 0..4 {
        for col in 0..4 {
            max_diff = max_diff.max((a[row][col] - b[row][col]).abs());
        }
    }
    if max_diff > tol {
        return Err(OdxError::Format(format!(
            "MRtrix inputs have mismatched affines (max abs diff {max_diff})"
        )));
    }
    Ok(())
}

fn find_required_file(dir: &Path, stem: &str) -> Result<PathBuf> {
    let mut candidates = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            continue;
        }
        if path_stem_without_image_ext(&path).ok().as_deref() == Some(stem) {
            candidates.push(path);
        }
    }
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .ok_or_else(|| OdxError::FileNotFound(dir.join(format!("{stem}.*"))))
}

fn path_stem_without_image_ext(path: &Path) -> Result<String> {
    let name = path
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| OdxError::Argument(format!("invalid path '{}'", path.display())))?;
    for ext in [".nii.gz", ".nii", ".mif.gz", ".mif"] {
        if let Some(stem) = name.strip_suffix(ext) {
            return Ok(stem.to_string());
        }
    }
    Err(OdxError::Argument(format!(
        "unsupported MRtrix image filename '{}'",
        path.display()
    )))
}

fn is_mif_path(path: &Path) -> bool {
    let name = path.file_name().and_then(OsStr::to_str).unwrap_or("");
    name.ends_with(".mif") || name.ends_with(".mif.gz")
}

fn is_nifti_path(path: &Path) -> bool {
    let name = path.file_name().and_then(OsStr::to_str).unwrap_or("");
    name.ends_with(".nii") || name.ends_with(".nii.gz")
}

fn ensure_ext(path: &Path, ext: &str) -> PathBuf {
    let name = path.file_name().and_then(OsStr::to_str).unwrap_or("");
    if name.ends_with(ext) {
        path.to_path_buf()
    } else {
        let mut out = path.to_path_buf();
        out.set_extension("");
        PathBuf::from(format!("{}.{ext}", out.display()))
    }
}

fn write_mif_f32(path: &Path, dims: &[usize], affine: &[[f64; 4]; 4], data: &[f32]) -> Result<()> {
    write_mif_bytes(path, dims, affine, "Float32LE", &f32_to_bytes(data))
}

fn write_mif_u32(path: &Path, dims: &[usize], affine: &[[f64; 4]; 4], data: &[u32]) -> Result<()> {
    write_mif_bytes(path, dims, affine, "UInt32LE", &u32_to_bytes(data))
}

fn write_mif_bytes(
    path: &Path,
    dims: &[usize],
    affine: &[[f64; 4]; 4],
    datatype: &str,
    data: &[u8],
) -> Result<()> {
    let voxel_sizes = voxel_sizes_from_affine(affine);
    let transform = mif_transform_from_affine(affine, &voxel_sizes);
    let layout = (0..dims.len())
        .map(|axis| format!("+{}", dims.len() - 1 - axis))
        .collect::<Vec<_>>()
        .join(",");

    let mut offset = 0usize;
    let header = loop {
        let mut candidate = String::new();
        candidate.push_str("mrtrix image\n");
        candidate.push_str(&format!(
            "dim: {}\n",
            dims.iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",")
        ));
        candidate.push_str(&format!(
            "vox: {}\n",
            (0..dims.len())
                .map(|axis| {
                    if axis < 3 {
                        voxel_sizes[axis].to_string()
                    } else {
                        "1".to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join(",")
        ));
        candidate.push_str(&format!("layout: {layout}\n"));
        candidate.push_str(&format!("datatype: {datatype}\n"));
        for row in transform {
            candidate.push_str(&format!(
                "transform: {}, {}, {}, {}\n",
                row[0], row[1], row[2], row[3]
            ));
        }
        candidate.push_str(&format!("file: . {offset}\nEND\n"));
        let new_offset = candidate.len();
        if new_offset == offset {
            break candidate;
        }
        offset = new_offset;
    };

    if path.extension() == Some(OsStr::new("gz")) {
        let file = std::fs::File::create(path)?;
        let mut writer = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        writer.write_all(header.as_bytes())?;
        writer.write_all(data)?;
        writer.finish()?;
    } else {
        let mut file = std::fs::File::create(path)?;
        file.write_all(header.as_bytes())?;
        file.write_all(data)?;
    }
    Ok(())
}

fn write_nifti1_f32_raw(
    path: &Path,
    dims: &[usize],
    affine: &[[f64; 4]; 4],
    data: &[f32],
) -> Result<()> {
    let array = Array::from_shape_vec(IxDyn(dims), data.to_vec())
        .map_err(|err| OdxError::Format(format!("failed to shape NIfTI-1 data: {err}")))?;
    let header = nifti1_header(affine);
    nifti::writer::WriterOptions::new(path)
        .reference_header(&header)
        .write_nifti(&array)
        .map_err(|err| {
            OdxError::Format(format!(
                "failed to write NIfTI-1 '{}': {err}",
                path.display()
            ))
        })
}

fn write_mrtrix_nifti1_f32(
    path: &Path,
    dims: &[usize],
    affine: &[[f64; 4]; 4],
    data: &[f32],
) -> Result<()> {
    let mut storage = data.to_vec();
    let flips = mrtrix_nifti_axis_flips(dims);
    let mut storage_affine = *affine;
    if flips.iter().any(|&flip| flip) {
        flip_spatial_axes_in_place(&mut storage, dims, flips);
        storage_affine = flip_spatial_axes_in_affine(storage_affine, dims, flips);
    }
    write_nifti1_f32_raw(path, dims, &storage_affine, &storage)
}

fn write_nifti2_f32_raw(
    path: &Path,
    dims: &[usize],
    affine: &[[f64; 4]; 4],
    data: &[f32],
) -> Result<()> {
    write_nifti2_bytes(
        path,
        dims,
        affine,
        NiftiType::Float32,
        &f32_to_fortran_bytes(data, dims),
    )
}

fn write_mrtrix_nifti2_f32(
    path: &Path,
    dims: &[usize],
    affine: &[[f64; 4]; 4],
    data: &[f32],
) -> Result<()> {
    let mut storage = data.to_vec();
    let flips = mrtrix_nifti_axis_flips(dims);
    let mut storage_affine = *affine;
    if flips.iter().any(|&flip| flip) {
        flip_spatial_axes_in_place(&mut storage, dims, flips);
        storage_affine = flip_spatial_axes_in_affine(storage_affine, dims, flips);
    }
    write_nifti2_f32_raw(path, dims, &storage_affine, &storage)
}

fn write_nifti2_u32_raw(
    path: &Path,
    dims: &[usize],
    affine: &[[f64; 4]; 4],
    data: &[u32],
) -> Result<()> {
    write_nifti2_bytes(
        path,
        dims,
        affine,
        NiftiType::Uint32,
        &u32_to_fortran_bytes(data, dims),
    )
}

fn write_mrtrix_nifti2_u32(
    path: &Path,
    dims: &[usize],
    affine: &[[f64; 4]; 4],
    data: &[u32],
) -> Result<()> {
    let mut storage = data.to_vec();
    let flips = mrtrix_nifti_axis_flips(dims);
    let mut storage_affine = *affine;
    if flips.iter().any(|&flip| flip) {
        flip_spatial_axes_in_place(&mut storage, dims, flips);
        storage_affine = flip_spatial_axes_in_affine(storage_affine, dims, flips);
    }
    write_nifti2_u32_raw(path, dims, &storage_affine, &storage)
}

fn write_nifti2_bytes(
    path: &Path,
    dims: &[usize],
    affine: &[[f64; 4]; 4],
    datatype: NiftiType,
    data: &[u8],
) -> Result<()> {
    let mut header = vec![0u8; 540];
    write_i32(&mut header, 0, 540);
    header[4..8].copy_from_slice(b"n+2\0");
    header[8..12].copy_from_slice(&[13, 10, 26, 10]);
    write_i16(&mut header, 12, datatype as i16);
    write_i16(&mut header, 14, (datatype.size_of() * 8) as i16);

    write_i64(&mut header, 16, dims.len() as i64);
    for (i, &dim) in dims.iter().enumerate() {
        write_i64(&mut header, 16 + 8 * (i + 1), dim as i64);
    }
    for i in dims.len() + 1..8 {
        write_i64(&mut header, 16 + 8 * i, 1);
    }

    let voxel_sizes = voxel_sizes_from_affine(affine);
    write_f64(&mut header, 104 + 8, voxel_sizes[0]);
    write_f64(&mut header, 104 + 16, voxel_sizes[1]);
    write_f64(&mut header, 104 + 24, voxel_sizes[2]);
    for i in 4..8 {
        write_f64(&mut header, 104 + 8 * i, 1.0);
    }

    write_i64(&mut header, 168, 544);
    write_f64(&mut header, 176, 1.0);
    write_i32(&mut header, 348, 1);
    write_f64_row(&mut header, 400, &affine[0]);
    write_f64_row(&mut header, 432, &affine[1]);
    write_f64_row(&mut header, 464, &affine[2]);
    write_i32(&mut header, 500, 2); // mm

    let mut writer: Box<dyn Write> = if path.extension() == Some(OsStr::new("gz")) {
        let file = std::fs::File::create(path)?;
        Box::new(flate2::write::GzEncoder::new(
            file,
            flate2::Compression::fast(),
        ))
    } else {
        Box::new(std::fs::File::create(path)?)
    };
    writer.write_all(&header)?;
    writer.write_all(&[0u8; 4])?;
    writer.write_all(data)?;
    Ok(())
}

fn nifti1_header(affine: &[[f64; 4]; 4]) -> NiftiHeader {
    let voxel_sizes = voxel_sizes_from_affine(affine);
    let mut header = NiftiHeader::default();
    header.sform_code = 1;
    header.qform_code = 0;
    header.pixdim[1] = voxel_sizes[0] as f32;
    header.pixdim[2] = voxel_sizes[1] as f32;
    header.pixdim[3] = voxel_sizes[2] as f32;
    header.xyzt_units = 2;
    header.srow_x = [
        affine[0][0] as f32,
        affine[0][1] as f32,
        affine[0][2] as f32,
        affine[0][3] as f32,
    ];
    header.srow_y = [
        affine[1][0] as f32,
        affine[1][1] as f32,
        affine[1][2] as f32,
        affine[1][3] as f32,
    ];
    header.srow_z = [
        affine[2][0] as f32,
        affine[2][1] as f32,
        affine[2][2] as f32,
        affine[2][3] as f32,
    ];
    header
}

fn voxel_sizes_from_affine(affine: &[[f64; 4]; 4]) -> [f64; 3] {
    [
        (affine[0][0].powi(2) + affine[1][0].powi(2) + affine[2][0].powi(2)).sqrt(),
        (affine[0][1].powi(2) + affine[1][1].powi(2) + affine[2][1].powi(2)).sqrt(),
        (affine[0][2].powi(2) + affine[1][2].powi(2) + affine[2][2].powi(2)).sqrt(),
    ]
}

fn mif_transform_from_affine(affine: &[[f64; 4]; 4], voxel_sizes: &[f64; 3]) -> [[f64; 4]; 3] {
    let mut out = [[0.0; 4]; 3];
    for row in 0..3 {
        for col in 0..3 {
            out[row][col] = affine[row][col] / voxel_sizes[col].max(f64::EPSILON);
        }
        out[row][3] = affine[row][3];
    }
    out
}

fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for &value in data {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn u32_to_bytes(data: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for &value in data {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn f32_to_fortran_bytes(data: &[f32], dims: &[usize]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for idx in fortran_indices(dims) {
        out.extend_from_slice(&data[idx].to_le_bytes());
    }
    out
}

fn u32_to_fortran_bytes(data: &[u32], dims: &[usize]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for idx in fortran_indices(dims) {
        out.extend_from_slice(&data[idx].to_le_bytes());
    }
    out
}

fn reorder_from_fortran<T: Copy + Default>(raw: Vec<T>, dims: &[usize]) -> Vec<T> {
    let mut out = vec![T::default(); raw.len()];
    for (src, dst) in fortran_indices(dims).into_iter().enumerate() {
        out[dst] = raw[src];
    }
    out
}

fn fortran_indices(dims: &[usize]) -> Vec<usize> {
    if dims.is_empty() {
        return Vec::new();
    }
    let total: usize = dims.iter().product();
    let mut coords = vec![0usize; dims.len()];
    let mut indices = Vec::with_capacity(total);
    for _ in 0..total {
        indices.push(c_index(&coords, dims));
        for axis in 0..dims.len() {
            coords[axis] += 1;
            if coords[axis] < dims[axis] {
                break;
            }
            coords[axis] = 0;
        }
    }
    indices
}

fn c_index(coords: &[usize], dims: &[usize]) -> usize {
    let mut index = 0usize;
    let mut stride = 1usize;
    for axis in (0..dims.len()).rev() {
        index += coords[axis] * stride;
        stride *= dims[axis];
    }
    index
}

fn write_i16(buf: &mut [u8], offset: usize, value: i16) {
    buf[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn write_i32(buf: &mut [u8], offset: usize, value: i32) {
    buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_i64(buf: &mut [u8], offset: usize, value: i64) {
    buf[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn write_f64(buf: &mut [u8], offset: usize, value: f64) {
    buf[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn write_f64_row(buf: &mut [u8], offset: usize, row: &[f64; 4]) {
    for (i, &value) in row.iter().enumerate() {
        write_f64(buf, offset + i * 8, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sh_only_dataset_derives_mask_from_sampled_amplitudes() {
        let dims = vec![2usize, 1, 1, 6];
        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let sh = LoadedF32Image {
            dims,
            affine,
            data: vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // positive isotropic voxel
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // empty voxel
            ],
        };

        let odx = build_sh_only_dataset(sh).unwrap();
        assert_eq!(odx.header().dimensions, [2, 1, 1]);
        assert_eq!(odx.nb_voxels(), 1);
        assert_eq!(odx.mask(), &[1, 0]);

        let coeffs = odx.sh::<f32>("coefficients").unwrap();
        assert_eq!(coeffs.nrows(), 1);
        assert_eq!(coeffs.row(0), &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn fortran_reorder_round_trip_u32() {
        let dims = [2usize, 3, 4, 2];
        let total: usize = dims.iter().product();
        let data: Vec<u32> = (0..total as u32).collect();
        let bytes = u32_to_fortran_bytes(&data, &dims);
        let raw: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let reordered = reorder_from_fortran(raw, &dims);
        assert_eq!(reordered, data);
    }

    #[test]
    fn nifti_index_counts_match_mif_counts() {
        let nifti = Path::new("../test_data/fixels_nii/index.nii");
        let mif_path = Path::new("../test_data/fixels_mif/index.mif");
        if !nifti.exists() || !mif_path.exists() {
            return;
        }

        let nii = load_u32_image(nifti).unwrap();
        let mif = load_u32_image(mif_path).unwrap();
        assert_eq!(nii.dims, mif.dims);
        let voxel_count = nii.dims[0] * nii.dims[1] * nii.dims[2];
        for voxel in 0..voxel_count {
            assert_eq!(nii.data[voxel * 2], mif.data[voxel * 2], "voxel {voxel}");
        }
    }

    #[test]
    fn raw_nifti2_u32_file_round_trip() {
        let dims = [4usize, 3, 2, 2];
        let total: usize = dims.iter().product();
        let data: Vec<u32> = (0..total as u32).map(|v| v * 7 + 3).collect();
        let affine = [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 2.0, 0.0, 20.0],
            [0.0, 0.0, 2.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let path = std::env::temp_dir().join("odx_mrtrix_nifti2_u32_test.nii");
        write_nifti2_u32_raw(&path, &dims, &affine, &data).unwrap();
        let parsed = load_u32_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(parsed.dims, dims);
        assert_eq!(parsed.data, data);
    }

    #[test]
    fn exported_real_index_nifti_round_trip() {
        let fixels = Path::new("../test_data/fixels_mif");
        if !fixels.exists() {
            return;
        }

        let odx = load_mrtrix_fixels(fixels).unwrap();
        let dims3 = odx.header().dimensions.map(|d| d as usize);
        let mask = odx.mask();
        let offsets = odx.offsets();
        let mut index = vec![0u32; dims3[0] * dims3[1] * dims3[2] * 2];
        let mut voxel_row = 0usize;
        let mut running_offset = 0u32;
        for (flat_idx, &m) in mask.iter().enumerate() {
            let base = flat_idx * 2;
            if m != 0 {
                let count = (offsets[voxel_row + 1] - offsets[voxel_row]) as u32;
                index[base] = count;
                index[base + 1] = running_offset;
                running_offset += count;
                voxel_row += 1;
            }
        }

        let path = std::env::temp_dir().join("odx_mrtrix_real_index_test.nii");
        write_mrtrix_nifti2_u32(
            &path,
            &[dims3[0], dims3[1], dims3[2], 2],
            &odx.header().voxel_to_rasmm,
            &index,
        )
        .unwrap();
        let parsed = load_u32_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(parsed.dims, vec![dims3[0], dims3[1], dims3[2], 2]);
        assert_eq!(parsed.data, index);
    }
}
