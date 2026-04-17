use bytemuck::cast_slice;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::data_array::{DataArray, DataPerGroup};
use crate::dtype::DType;
use crate::error::{OdxError, Result};
use crate::header::Header;
use crate::io::filename::OdxFilename;
use crate::io::{dequantize_array, maybe_quantize_array, normalize_float_array};
use crate::mmap_backing::{vec_to_bytes, MmapBacking};
use crate::odx_file::{OdxDataset, OdxParts, OdxWritePolicy};

fn mmap_file(path: &Path) -> Result<Mmap> {
    let file = fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    Ok(mmap)
}

fn load_float_data_dir(
    dir: &Path,
    header: &Header,
    prefix: &str,
) -> Result<HashMap<String, DataArray>> {
    let mut map = HashMap::new();
    if !dir.exists() {
        return Ok(map);
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| OdxError::Format(format!("invalid filename: {}", path.display())))?;
        let parsed = OdxFilename::parse(file_name)?;
        let mmap = mmap_file(&path)?;
        let raw = DataArray::from_backing(MmapBacking::ReadOnly(mmap), parsed.ncols, parsed.dtype);
        let quant_key = format!("{prefix}/{}", parsed.name);
        let arr = if let Some(spec) = header.array_quantization.get(&quant_key) {
            dequantize_array(&raw, spec)
        } else {
            normalize_float_array(&raw)?
        };
        map.insert(parsed.name.clone(), arr);
    }
    Ok(map)
}

fn load_raw_data_dir(dir: &Path) -> Result<HashMap<String, DataArray>> {
    let mut map = HashMap::new();
    if !dir.exists() {
        return Ok(map);
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| OdxError::Format(format!("invalid filename: {}", path.display())))?;
        let parsed = OdxFilename::parse(file_name)?;
        let mmap = mmap_file(&path)?;
        map.insert(
            parsed.name.clone(),
            DataArray::from_backing(MmapBacking::ReadOnly(mmap), parsed.ncols, parsed.dtype),
        );
    }
    Ok(map)
}

fn load_dpg_dir(dir: &Path, header: &Header) -> Result<DataPerGroup> {
    let mut out = HashMap::new();
    if !dir.exists() {
        return Ok(out);
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let group_name = entry.file_name().to_string_lossy().to_string();
        let data = load_float_data_dir(&path, header, &format!("dpg/{group_name}"))?;
        if !data.is_empty() {
            out.insert(group_name, data);
        }
    }
    Ok(out)
}

fn find_file_with_prefix(dir: &Path, prefix: &str) -> Result<std::path::PathBuf> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with(prefix) && name_str.chars().nth(prefix.len()) == Some('.') {
            return Ok(entry.path());
        }
    }
    Err(OdxError::FileNotFound(dir.join(prefix)))
}

fn convert_offsets_to_u32(
    mmap: &Mmap,
    dtype: DType,
    nb_voxels: usize,
    nb_peaks: usize,
) -> Result<MmapBacking> {
    match dtype {
        DType::UInt64 => {
            let values: &[u64] = cast_slice(mmap.as_ref());
            let mut owned: Vec<u32> = values
                .iter()
                .copied()
                .map(|v| {
                    u32::try_from(v)
                        .map_err(|_| OdxError::Format(format!("offset {v} exceeds uint32 range")))
                })
                .collect::<Result<_>>()?;
            if owned.len() == nb_voxels {
                owned.push(nb_peaks as u32);
            }
            Ok(MmapBacking::Owned(vec_to_bytes(owned)))
        }
        DType::UInt32 => {
            let values: &[u32] = cast_slice(mmap.as_ref());
            let mut out = values.to_vec();
            if out.len() == nb_voxels {
                out.push(nb_peaks as u32);
            }
            Ok(MmapBacking::Owned(vec_to_bytes(out)))
        }
        other => Err(OdxError::DType(format!(
            "offsets must be uint32 or uint64, got {other}"
        ))),
    }
}

pub fn open_directory(dir: &Path, tempdir: Option<tempfile::TempDir>) -> Result<OdxDataset> {
    if !dir.is_dir() {
        return Err(OdxError::FileNotFound(dir.to_path_buf()));
    }

    let header = Header::from_file(&dir.join("header.json"))?;

    let mask_path = find_file_with_prefix(dir, "mask")?;
    let mask_backing = MmapBacking::ReadOnly(mmap_file(&mask_path)?);
    let expected_mask_len = header.mask_volume_size();
    if mask_backing.len() != expected_mask_len {
        return Err(OdxError::Format(format!(
            "mask length {} does not match volume size {}",
            mask_backing.len(),
            expected_mask_len
        )));
    }

    let off_path = find_file_with_prefix(dir, "offsets")?;
    let off_fname = off_path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| OdxError::Format("invalid offsets filename".into()))?;
    let off_parsed = OdxFilename::parse(off_fname)?;
    let offsets_mmap = mmap_file(&off_path)?;
    let offsets_backing = convert_offsets_to_u32(
        &offsets_mmap,
        off_parsed.dtype,
        header.nb_voxels as usize,
        header.nb_peaks as usize,
    )?;

    let dir_path = find_file_with_prefix(dir, "directions")?;
    let dir_fname = dir_path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| OdxError::Format("invalid directions filename".into()))?;
    let dir_parsed = OdxFilename::parse(dir_fname)?;
    if dir_parsed.dtype != DType::Float32 || dir_parsed.ncols != 3 {
        return Err(OdxError::Format(format!(
            "directions must be stored as directions.3.float32, got {dir_fname}"
        )));
    }
    let directions_backing = MmapBacking::ReadOnly(mmap_file(&dir_path)?);

    let sphere_dir = dir.join("sphere");
    let sphere_vertices = if sphere_dir.exists() {
        let verts_path = find_file_with_prefix(&sphere_dir, "vertices")?;
        let verts_fname = verts_path.file_name().and_then(|n| n.to_str()).unwrap();
        let verts_parsed = OdxFilename::parse(verts_fname)?;
        if verts_parsed.dtype != DType::Float32 || verts_parsed.ncols != 3 {
            return Err(OdxError::Format(format!(
                "sphere vertices must be stored as vertices.3.float32, got {verts_fname}"
            )));
        }
        Some(MmapBacking::ReadOnly(mmap_file(&verts_path)?))
    } else {
        None
    };

    let sphere_faces = if sphere_dir.exists() {
        let faces_path = find_file_with_prefix(&sphere_dir, "faces")?;
        Some(MmapBacking::ReadOnly(mmap_file(&faces_path)?))
    } else {
        None
    };

    let odf = load_float_data_dir(&dir.join("odf"), &header, "odf")?;
    let sh = load_float_data_dir(&dir.join("sh"), &header, "sh")?;
    let dpv = load_float_data_dir(&dir.join("dpv"), &header, "dpv")?;
    let dpf = load_float_data_dir(&dir.join("dpf"), &header, "dpf")?;
    let groups = load_raw_data_dir(&dir.join("groups"))?;
    let dpg = load_dpg_dir(&dir.join("dpg"), &header)?;

    Ok(OdxDataset::from_parts(OdxParts {
        header,
        mask_backing,
        offsets_backing,
        directions_backing,
        sphere_vertices,
        sphere_faces,
        odf,
        sh,
        dpv,
        dpf,
        groups,
        dpg,
        tempdir,
    }))
}

fn offsets_as_u32_bytes(offsets: &[u32]) -> Vec<u8> {
    vec_to_bytes(offsets.to_vec())
}

pub fn save_directory(odx: &OdxDataset, dir: &Path, policy: OdxWritePolicy) -> Result<()> {
    fs::create_dir_all(dir)?;

    let mut header = odx.header().clone();
    header.array_quantization.clear();

    fs::write(dir.join("mask.uint8"), odx.mask())?;
    fs::write(
        dir.join("offsets.uint32"),
        offsets_as_u32_bytes(odx.offsets()),
    )?;
    fs::write(dir.join("directions.3.float32"), odx.directions_bytes())?;

    if let Some(verts_bytes) = odx.sphere_vertices_bytes() {
        let sphere_dir = dir.join("sphere");
        fs::create_dir_all(&sphere_dir)?;
        fs::write(sphere_dir.join("vertices.3.float32"), verts_bytes)?;
        if let Some(faces_bytes) = odx.sphere_faces_bytes() {
            fs::write(sphere_dir.join("faces.3.uint32"), faces_bytes)?;
        }
    }

    save_data_dir(
        odx.odf_arrays(),
        &dir.join("odf"),
        "odf",
        policy,
        true,
        &mut header,
    )?;
    save_data_dir(
        odx.sh_arrays(),
        &dir.join("sh"),
        "sh",
        policy,
        true,
        &mut header,
    )?;
    save_data_dir(
        odx.dpv_arrays(),
        &dir.join("dpv"),
        "dpv",
        policy,
        true,
        &mut header,
    )?;
    save_data_dir(
        odx.dpf_arrays(),
        &dir.join("dpf"),
        "dpf",
        policy,
        false,
        &mut header,
    )?;
    save_data_dir(
        odx.group_arrays(),
        &dir.join("groups"),
        "groups",
        policy,
        false,
        &mut header,
    )?;
    save_dpg_dir(odx.dpg_arrays(), &dir.join("dpg"), policy, &mut header)?;

    header.write_to(&dir.join("header.json"))?;
    Ok(())
}

pub fn append_dpf_to_directory(
    dir: &Path,
    dpf: &HashMap<String, DataArray>,
    overwrite: bool,
) -> Result<()> {
    let mut header = Header::from_file(&dir.join("header.json"))?;
    validate_row_count("DPF", dpf, header.nb_peaks as usize)?;
    let dpf_dir = dir.join("dpf");
    fs::create_dir_all(&dpf_dir)?;
    let mut header_dirty = false;
    for (name, arr) in dpf {
        if !overwrite {
            if find_named_array_file(&dpf_dir, name)?.is_some() {
                continue;
            }
        } else if let Some(existing) = find_named_array_file(&dpf_dir, name)? {
            let target = dpf_dir.join(filename_for_array(name, arr));
            if existing != target && existing.exists() {
                fs::remove_file(existing)?;
            }
            header_dirty |= header
                .array_quantization
                .remove(&format!("dpf/{name}"))
                .is_some();
        }
        fs::write(dpf_dir.join(filename_for_array(name, arr)), arr.as_bytes())?;
    }
    if header_dirty {
        header.write_to(&dir.join("header.json"))?;
    }
    Ok(())
}

pub fn append_dpv_to_directory(
    dir: &Path,
    dpv: &HashMap<String, DataArray>,
    overwrite: bool,
) -> Result<()> {
    let mut header = Header::from_file(&dir.join("header.json"))?;
    validate_row_count("DPV", dpv, header.nb_voxels as usize)?;
    let dpv_dir = dir.join("dpv");
    fs::create_dir_all(&dpv_dir)?;
    let mut header_dirty = false;
    for (name, arr) in dpv {
        if !overwrite {
            if find_named_array_file(&dpv_dir, name)?.is_some() {
                continue;
            }
        } else if let Some(existing) = find_named_array_file(&dpv_dir, name)? {
            let target = dpv_dir.join(filename_for_array(name, arr));
            if existing != target && existing.exists() {
                fs::remove_file(existing)?;
            }
            header_dirty |= header
                .array_quantization
                .remove(&format!("dpv/{name}"))
                .is_some();
        }
        fs::write(dpv_dir.join(filename_for_array(name, arr)), arr.as_bytes())?;
    }
    if header_dirty {
        header.write_to(&dir.join("header.json"))?;
    }
    Ok(())
}

pub fn append_groups_to_directory(
    dir: &Path,
    groups: &HashMap<String, DataArray>,
    overwrite: bool,
) -> Result<()> {
    let groups_dir = dir.join("groups");
    fs::create_dir_all(&groups_dir)?;
    for (name, arr) in groups {
        let target = groups_dir.join(filename_for_array(name, arr));
        if !overwrite {
            if find_named_array_file(&groups_dir, name)?.is_some() {
                continue;
            }
        } else if let Some(existing) = find_named_array_file(&groups_dir, name)? {
            if existing != target && existing.exists() {
                fs::remove_file(existing)?;
            }
        }
        fs::write(target, arr.as_bytes())?;
    }
    Ok(())
}

pub fn append_dpg_to_directory(dir: &Path, dpg: &DataPerGroup, overwrite: bool) -> Result<()> {
    let mut header = Header::from_file(&dir.join("header.json"))?;
    let groups_dir = dir.join("groups");
    let dpg_root = dir.join("dpg");
    let mut header_dirty = false;
    for (group, arrays) in dpg {
        if find_named_array_file(&groups_dir, group)?.is_none() {
            return Err(OdxError::Argument(format!(
                "cannot add DPG entries for missing group '{group}'"
            )));
        }
        let group_dir = dpg_root.join(group);
        fs::create_dir_all(&group_dir)?;
        for (name, arr) in arrays {
            let target = group_dir.join(filename_for_array(name, arr));
            if !overwrite {
                if find_named_array_file(&group_dir, name)?.is_some() {
                    continue;
                }
            } else if let Some(existing) = find_named_array_file(&group_dir, name)? {
                if existing != target && existing.exists() {
                    fs::remove_file(existing)?;
                }
                header_dirty |= header
                    .array_quantization
                    .remove(&format!("dpg/{group}/{name}"))
                    .is_some();
            }
            fs::write(target, arr.as_bytes())?;
        }
    }
    if header_dirty {
        header.write_to(&dir.join("header.json"))?;
    }
    Ok(())
}

pub fn delete_dpf_from_directory(dir: &Path, names: &[&str]) -> Result<()> {
    let mut header = Header::from_file(&dir.join("header.json"))?;
    let dpf_dir = dir.join("dpf");
    let mut header_dirty = false;
    for name in names {
        if let Some(path) = find_named_array_file(&dpf_dir, name)? {
            if path.exists() {
                fs::remove_file(path)?;
            }
            header_dirty |= header
                .array_quantization
                .remove(&format!("dpf/{name}"))
                .is_some();
        }
    }
    if header_dirty {
        header.write_to(&dir.join("header.json"))?;
    }
    Ok(())
}

pub fn delete_dpv_from_directory(dir: &Path, names: &[&str]) -> Result<()> {
    let mut header = Header::from_file(&dir.join("header.json"))?;
    let dpv_dir = dir.join("dpv");
    let mut header_dirty = false;
    for name in names {
        if let Some(path) = find_named_array_file(&dpv_dir, name)? {
            if path.exists() {
                fs::remove_file(path)?;
            }
            header_dirty |= header
                .array_quantization
                .remove(&format!("dpv/{name}"))
                .is_some();
        }
    }
    if header_dirty {
        header.write_to(&dir.join("header.json"))?;
    }
    Ok(())
}

pub fn delete_groups_from_directory(dir: &Path, names: &[&str]) -> Result<()> {
    let mut header = Header::from_file(&dir.join("header.json"))?;
    let groups_dir = dir.join("groups");
    let mut header_dirty = false;
    for name in names {
        if let Some(path) = find_named_array_file(&groups_dir, name)? {
            if path.exists() {
                fs::remove_file(path)?;
            }
        }
        let dpg_dir = dir.join("dpg").join(name);
        if dpg_dir.exists() {
            fs::remove_dir_all(dpg_dir)?;
        }
        let prefix = format!("dpg/{name}/");
        let before = header.array_quantization.len();
        header
            .array_quantization
            .retain(|key, _| !key.starts_with(&prefix));
        header_dirty |= header.array_quantization.len() != before;
    }
    if header_dirty {
        header.write_to(&dir.join("header.json"))?;
    }
    Ok(())
}

pub fn delete_dpg_from_directory(dir: &Path, group: &str, names: Option<&[&str]>) -> Result<()> {
    let mut header = Header::from_file(&dir.join("header.json"))?;
    let group_dir = dir.join("dpg").join(group);
    let mut header_dirty = false;
    match names {
        None | Some([]) => {
            if group_dir.exists() {
                fs::remove_dir_all(group_dir)?;
            }
            let prefix = format!("dpg/{group}/");
            let before = header.array_quantization.len();
            header
                .array_quantization
                .retain(|key, _| !key.starts_with(&prefix));
            header_dirty |= header.array_quantization.len() != before;
        }
        Some(names) => {
            for name in names {
                if let Some(path) = find_named_array_file(&group_dir, name)? {
                    if path.exists() {
                        fs::remove_file(path)?;
                    }
                    header_dirty |= header
                        .array_quantization
                        .remove(&format!("dpg/{group}/{name}"))
                        .is_some();
                }
            }
        }
    }
    if header_dirty {
        header.write_to(&dir.join("header.json"))?;
    }
    Ok(())
}

fn save_data_dir(
    arrays: &HashMap<String, DataArray>,
    dir: &Path,
    prefix: &str,
    policy: OdxWritePolicy,
    allow_quantization: bool,
    header: &mut Header,
) -> Result<()> {
    if arrays.is_empty() {
        return Ok(());
    }
    fs::create_dir_all(dir)?;
    for (name, arr) in arrays {
        let key = format!("{prefix}/{name}");
        let (stored, quant) = maybe_quantize_array(&key, arr, policy, allow_quantization);
        if let Some(spec) = quant {
            header.array_quantization.insert(key.clone(), spec);
        }
        let filename = filename_for_array(name, &stored);
        fs::write(dir.join(filename), stored.as_bytes())?;
    }
    Ok(())
}

fn validate_row_count(
    kind: &str,
    arrays: &HashMap<String, DataArray>,
    expected_rows: usize,
) -> Result<()> {
    for (name, arr) in arrays {
        if arr.nrows() != expected_rows {
            return Err(OdxError::Format(format!(
                "{kind} '{name}' has {} rows, expected {expected_rows}",
                arr.nrows()
            )));
        }
    }
    Ok(())
}

fn find_named_array_file(dir: &Path, name: &str) -> Result<Option<std::path::PathBuf>> {
    if !dir.exists() {
        return Ok(None);
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| OdxError::Format(format!("invalid filename: {}", path.display())))?;
        let parsed = OdxFilename::parse(file_name)?;
        if parsed.name == name {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

fn filename_for_array(name: &str, arr: &DataArray) -> String {
    OdxFilename {
        name: name.to_string(),
        ncols: arr.ncols(),
        dtype: arr.dtype(),
    }
    .to_filename()
}

fn save_dpg_dir(
    groups: &DataPerGroup,
    dir: &Path,
    policy: OdxWritePolicy,
    header: &mut Header,
) -> Result<()> {
    if groups.is_empty() {
        return Ok(());
    }
    fs::create_dir_all(dir)?;
    for (group, arrays) in groups {
        save_data_dir(
            arrays,
            &dir.join(group),
            &format!("dpg/{group}"),
            policy,
            true,
            header,
        )?;
    }
    Ok(())
}
