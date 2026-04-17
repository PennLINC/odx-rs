use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;
use zip::write::SimpleFileOptions;

use super::archive_edit::{self, ArchiveOp};
use crate::data_array::{DataArray, DataPerGroup};
use crate::error::{OdxError, Result};
use crate::header::Header;
use crate::io::filename::OdxFilename;
use crate::odx_file::{OdxDataset, OdxWritePolicy};

#[derive(Debug, Default)]
struct OdxArchiveIndex {
    dpf: HashMap<String, String>,
    dpv: HashMap<String, String>,
    groups: HashMap<String, String>,
    dpg: HashMap<String, HashMap<String, String>>,
}

pub fn open_archive(path: &Path) -> Result<OdxDataset> {
    let file = fs::File::open(path)?;
    let mut archive = zip::ZipArchive::new(file)?;

    let tempdir = tempfile::TempDir::new()?;
    let temp_path = tempdir.path().to_path_buf();

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let entry_path = temp_path.join(entry.name());

        if entry.is_dir() {
            fs::create_dir_all(&entry_path)?;
        } else {
            if let Some(parent) = entry_path.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut out_file = fs::File::create(&entry_path)?;
            std::io::copy(&mut entry, &mut out_file)?;
        }
    }

    crate::io::directory::open_directory(&temp_path, Some(tempdir))
}

pub fn save_archive(odx: &OdxDataset, path: &Path, policy: OdxWritePolicy) -> Result<()> {
    save_archive_with(odx, path, zip::CompressionMethod::Deflated, policy)
}

pub fn save_archive_with(
    odx: &OdxDataset,
    path: &Path,
    compression: zip::CompressionMethod,
    policy: OdxWritePolicy,
) -> Result<()> {
    let tempdir = tempfile::TempDir::new()?;
    let dir = tempdir.path().join("archive.odxd");
    crate::io::directory::save_directory(odx, &dir, policy)?;
    let file = fs::File::create(path)?;
    let mut zip = zip::ZipWriter::new(file);
    let options = SimpleFileOptions::default()
        .compression_method(compression)
        .large_file(true);
    write_dir_to_zip(&dir, &mut zip, options, "")?;
    zip.finish()?;
    Ok(())
}

pub fn append_dpf_to_zip(
    path: &Path,
    dpf: &HashMap<String, DataArray>,
    compression: zip::CompressionMethod,
    overwrite: bool,
) -> Result<()> {
    let mut header = read_header_from_zip(path)?;
    validate_row_count("DPF", dpf, header.nb_peaks as usize)?;
    let index = build_archive_index(path)?;
    let mut ops = Vec::new();
    let mut header_dirty = false;
    for (name, arr) in dpf {
        if overwrite && index.dpf.contains_key(name) {
            header_dirty |= header
                .array_quantization
                .remove(&format!("dpf/{name}"))
                .is_some();
        }
        let target = data_entry_path("dpf", name, arr);
        plan_data_write(
            &index.dpf,
            name,
            target,
            arr,
            overwrite,
            compression,
            &mut ops,
        )?;
    }
    maybe_push_header_replace(&mut ops, &header, header_dirty)?;
    archive_edit::apply_archive_ops(path, ops)
}

pub fn append_dpv_to_zip(
    path: &Path,
    dpv: &HashMap<String, DataArray>,
    compression: zip::CompressionMethod,
    overwrite: bool,
) -> Result<()> {
    let mut header = read_header_from_zip(path)?;
    validate_row_count("DPV", dpv, header.nb_voxels as usize)?;
    let index = build_archive_index(path)?;
    let mut ops = Vec::new();
    let mut header_dirty = false;
    for (name, arr) in dpv {
        if overwrite && index.dpv.contains_key(name) {
            header_dirty |= header
                .array_quantization
                .remove(&format!("dpv/{name}"))
                .is_some();
        }
        let target = data_entry_path("dpv", name, arr);
        plan_data_write(
            &index.dpv,
            name,
            target,
            arr,
            overwrite,
            compression,
            &mut ops,
        )?;
    }
    maybe_push_header_replace(&mut ops, &header, header_dirty)?;
    archive_edit::apply_archive_ops(path, ops)
}

pub fn append_groups_to_zip(
    path: &Path,
    groups: &HashMap<String, DataArray>,
    compression: zip::CompressionMethod,
    overwrite: bool,
) -> Result<()> {
    let index = build_archive_index(path)?;
    let mut ops = Vec::new();
    for (name, arr) in groups {
        let target = data_entry_path("groups", name, arr);
        plan_data_write(
            &index.groups,
            name,
            target,
            arr,
            overwrite,
            compression,
            &mut ops,
        )?;
    }
    archive_edit::apply_archive_ops(path, ops)
}

pub fn append_dpg_to_zip(
    path: &Path,
    dpg: &DataPerGroup,
    compression: zip::CompressionMethod,
    overwrite: bool,
) -> Result<()> {
    let mut header = read_header_from_zip(path)?;
    let index = build_archive_index(path)?;
    let mut ops = Vec::new();
    let mut header_dirty = false;
    for (group, arrays) in dpg {
        if !index.groups.contains_key(group) {
            return Err(OdxError::Argument(format!(
                "cannot add DPG entries for missing group '{group}'"
            )));
        }
        let existing = index.dpg.get(group);
        for (name, arr) in arrays {
            if overwrite && existing.and_then(|entries| entries.get(name)).is_some() {
                header_dirty |= header
                    .array_quantization
                    .remove(&format!("dpg/{group}/{name}"))
                    .is_some();
            }
            let target = format!("dpg/{group}/{}", filename_for_array(name, arr));
            let existing_path = existing.and_then(|entries| entries.get(name));
            plan_bytes_write(
                existing_path,
                target,
                arr.as_bytes().to_vec(),
                overwrite,
                compression,
                &mut ops,
            )?;
        }
    }
    maybe_push_header_replace(&mut ops, &header, header_dirty)?;
    archive_edit::apply_archive_ops(path, ops)
}

pub fn delete_dpf_from_zip(path: &Path, names: &[&str]) -> Result<()> {
    let mut header = read_header_from_zip(path)?;
    let index = build_archive_index(path)?;
    let mut ops = Vec::new();
    let mut header_dirty = false;
    for name in names {
        if let Some(entry_path) = index.dpf.get(*name) {
            ops.push(ArchiveOp::Delete {
                path: entry_path.clone(),
            });
            header_dirty |= header
                .array_quantization
                .remove(&format!("dpf/{name}"))
                .is_some();
        }
    }
    maybe_push_header_replace(&mut ops, &header, header_dirty)?;
    archive_edit::apply_archive_ops(path, ops)
}

pub fn delete_dpv_from_zip(path: &Path, names: &[&str]) -> Result<()> {
    let mut header = read_header_from_zip(path)?;
    let index = build_archive_index(path)?;
    let mut ops = Vec::new();
    let mut header_dirty = false;
    for name in names {
        if let Some(entry_path) = index.dpv.get(*name) {
            ops.push(ArchiveOp::Delete {
                path: entry_path.clone(),
            });
            header_dirty |= header
                .array_quantization
                .remove(&format!("dpv/{name}"))
                .is_some();
        }
    }
    maybe_push_header_replace(&mut ops, &header, header_dirty)?;
    archive_edit::apply_archive_ops(path, ops)
}

pub fn delete_groups_from_zip(path: &Path, names: &[&str]) -> Result<()> {
    let mut header = read_header_from_zip(path)?;
    let index = build_archive_index(path)?;
    let mut ops = Vec::new();
    let mut header_dirty = false;
    for name in names {
        if let Some(entry_path) = index.groups.get(*name) {
            ops.push(ArchiveOp::Delete {
                path: entry_path.clone(),
            });
        }
        ops.push(ArchiveOp::DeletePrefix {
            prefix: format!("dpg/{name}"),
        });
        let prefix = format!("dpg/{name}/");
        let before = header.array_quantization.len();
        header
            .array_quantization
            .retain(|key, _| !key.starts_with(&prefix));
        header_dirty |= header.array_quantization.len() != before;
    }
    maybe_push_header_replace(&mut ops, &header, header_dirty)?;
    archive_edit::apply_archive_ops(path, ops)
}

pub fn delete_dpg_from_zip(path: &Path, group: &str, names: Option<&[&str]>) -> Result<()> {
    let mut header = read_header_from_zip(path)?;
    let index = build_archive_index(path)?;
    let mut ops = Vec::new();
    let mut header_dirty = false;
    match names {
        None | Some([]) => {
            ops.push(ArchiveOp::DeletePrefix {
                prefix: format!("dpg/{group}"),
            });
            let prefix = format!("dpg/{group}/");
            let before = header.array_quantization.len();
            header
                .array_quantization
                .retain(|key, _| !key.starts_with(&prefix));
            header_dirty |= header.array_quantization.len() != before;
        }
        Some(names) => {
            if let Some(entries) = index.dpg.get(group) {
                for name in names {
                    if let Some(entry_path) = entries.get(*name) {
                        ops.push(ArchiveOp::Delete {
                            path: entry_path.clone(),
                        });
                        header_dirty |= header
                            .array_quantization
                            .remove(&format!("dpg/{group}/{name}"))
                            .is_some();
                    }
                }
            }
        }
    }
    maybe_push_header_replace(&mut ops, &header, header_dirty)?;
    archive_edit::apply_archive_ops(path, ops)
}

fn read_header_from_zip(path: &Path) -> Result<Header> {
    let bytes = archive_edit::read_archive_entry(path, "header.json")?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn build_archive_index(path: &Path) -> Result<OdxArchiveIndex> {
    let entries = archive_edit::archive_entry_names(path)?;
    let mut index = OdxArchiveIndex::default();
    for entry in entries {
        if let Some(rest) = entry.strip_prefix("dpf/") {
            index_entry(&mut index.dpf, &entry, rest)?;
        } else if let Some(rest) = entry.strip_prefix("dpv/") {
            index_entry(&mut index.dpv, &entry, rest)?;
        } else if let Some(rest) = entry.strip_prefix("groups/") {
            index_entry(&mut index.groups, &entry, rest)?;
        } else if let Some(rest) = entry.strip_prefix("dpg/") {
            if let Some((group, file_name)) = rest.split_once('/') {
                let parsed = OdxFilename::parse(file_name)?;
                let group_entries = index.dpg.entry(group.to_string()).or_default();
                if group_entries.insert(parsed.name, entry.clone()).is_some() {
                    return Err(OdxError::Format(format!(
                        "duplicate DPG entry path for group '{group}'"
                    )));
                }
            }
        }
    }
    Ok(index)
}

fn index_entry(
    index: &mut HashMap<String, String>,
    full_path: &str,
    file_name: &str,
) -> Result<()> {
    if file_name.ends_with('/') {
        return Ok(());
    }
    let parsed = OdxFilename::parse(file_name)?;
    if index.insert(parsed.name, full_path.to_string()).is_some() {
        return Err(OdxError::Format(format!(
            "duplicate archive entry for '{full_path}'"
        )));
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

fn data_entry_path(prefix: &str, name: &str, arr: &DataArray) -> String {
    format!("{prefix}/{}", filename_for_array(name, arr))
}

fn filename_for_array(name: &str, arr: &DataArray) -> String {
    OdxFilename {
        name: name.to_string(),
        ncols: arr.ncols(),
        dtype: arr.dtype(),
    }
    .to_filename()
}

fn plan_data_write(
    existing: &HashMap<String, String>,
    logical_name: &str,
    target_path: String,
    arr: &DataArray,
    overwrite: bool,
    compression: zip::CompressionMethod,
    ops: &mut Vec<ArchiveOp>,
) -> Result<()> {
    plan_bytes_write(
        existing.get(logical_name),
        target_path,
        arr.as_bytes().to_vec(),
        overwrite,
        compression,
        ops,
    )
}

fn plan_bytes_write(
    existing_path: Option<&String>,
    target_path: String,
    bytes: Vec<u8>,
    overwrite: bool,
    compression: zip::CompressionMethod,
    ops: &mut Vec<ArchiveOp>,
) -> Result<()> {
    match existing_path {
        None => ops.push(ArchiveOp::Add {
            path: target_path,
            bytes,
            compression,
        }),
        Some(_) if !overwrite => {}
        Some(existing) if existing == &target_path => ops.push(ArchiveOp::Replace {
            path: target_path,
            bytes,
            compression,
        }),
        Some(existing) => {
            ops.push(ArchiveOp::Delete {
                path: existing.clone(),
            });
            ops.push(ArchiveOp::Add {
                path: target_path,
                bytes,
                compression,
            });
        }
    }
    Ok(())
}

fn maybe_push_header_replace(ops: &mut Vec<ArchiveOp>, header: &Header, dirty: bool) -> Result<()> {
    if dirty {
        ops.push(ArchiveOp::Replace {
            path: "header.json".into(),
            bytes: header.to_json()?.into_bytes(),
            compression: zip::CompressionMethod::Deflated,
        });
    }
    Ok(())
}

fn write_dir_to_zip<W: Write + std::io::Seek>(
    dir: &Path,
    zip: &mut zip::ZipWriter<W>,
    options: SimpleFileOptions,
    prefix: &str,
) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = if prefix.is_empty() {
            entry.file_name().to_string_lossy().to_string()
        } else {
            format!("{prefix}/{}", entry.file_name().to_string_lossy())
        };
        if path.is_dir() {
            write_dir_to_zip(&path, zip, options, &name)?;
        } else {
            zip.start_file(&name, options)?;
            zip.write_all(&fs::read(&path)?)?;
        }
    }
    Ok(())
}
