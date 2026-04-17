use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

use tempfile::NamedTempFile;
use zip::write::SimpleFileOptions;

use crate::error::{OdxError, Result};

#[derive(Debug, Clone)]
pub(crate) enum ArchiveOp {
    Add {
        path: String,
        bytes: Vec<u8>,
        compression: zip::CompressionMethod,
    },
    Replace {
        path: String,
        bytes: Vec<u8>,
        compression: zip::CompressionMethod,
    },
    Delete {
        path: String,
    },
    DeletePrefix {
        prefix: String,
    },
}

#[derive(Debug)]
struct PendingEntry {
    bytes: Vec<u8>,
    compression: zip::CompressionMethod,
}

#[derive(Debug, Default)]
struct NormalizedOps {
    adds: BTreeMap<String, PendingEntry>,
    replaces: BTreeMap<String, PendingEntry>,
    deletes: BTreeSet<String>,
    delete_prefixes: Vec<String>,
}

impl NormalizedOps {
    fn is_append_only(&self) -> bool {
        self.replaces.is_empty() && self.deletes.is_empty() && self.delete_prefixes.is_empty()
    }

    fn should_skip(&self, path: &str) -> bool {
        self.deletes.contains(path)
            || self.replaces.contains_key(path)
            || self
                .delete_prefixes
                .iter()
                .any(|prefix| path_matches_prefix(path, prefix))
    }
}

pub(crate) fn archive_entry_names(path: &Path) -> Result<BTreeSet<String>> {
    let file = File::open(path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    let mut names = BTreeSet::new();
    for index in 0..archive.len() {
        let entry = archive.by_index(index)?;
        names.insert(entry.name().to_string());
    }
    Ok(names)
}

pub(crate) fn read_archive_entry(path: &Path, entry_name: &str) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    let mut entry = archive.by_name(entry_name)?;
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes)?;
    Ok(bytes)
}

pub(crate) fn apply_archive_ops(path: &Path, ops: Vec<ArchiveOp>) -> Result<()> {
    if ops.is_empty() {
        return Ok(());
    }

    let normalized = normalize_ops(ops)?;
    let existing = archive_entry_names(path)?;

    if normalized.is_append_only() {
        if let Some(duplicate) = normalized
            .adds
            .keys()
            .find(|target| existing.contains(target.as_str()))
        {
            return Err(OdxError::Argument(format!(
                "archive already contains '{duplicate}'"
            )));
        }
        append_fast_path(path, &normalized.adds)
    } else {
        rewrite_path(path, &normalized)
    }
}

fn normalize_ops(ops: Vec<ArchiveOp>) -> Result<NormalizedOps> {
    let mut normalized = NormalizedOps::default();

    for op in ops {
        match op {
            ArchiveOp::Add {
                path,
                bytes,
                compression,
            } => insert_pending(
                &mut normalized.adds,
                &normalized.replaces,
                path,
                bytes,
                compression,
            )?,
            ArchiveOp::Replace {
                path,
                bytes,
                compression,
            } => insert_pending(
                &mut normalized.replaces,
                &normalized.adds,
                path,
                bytes,
                compression,
            )?,
            ArchiveOp::Delete { path } => {
                normalized.deletes.insert(path);
            }
            ArchiveOp::DeletePrefix { prefix } => {
                normalized.delete_prefixes.push(normalize_prefix(prefix));
            }
        }
    }

    Ok(normalized)
}

fn insert_pending(
    target: &mut BTreeMap<String, PendingEntry>,
    other: &BTreeMap<String, PendingEntry>,
    path: String,
    bytes: Vec<u8>,
    compression: zip::CompressionMethod,
) -> Result<()> {
    if other.contains_key(&path) || target.contains_key(&path) {
        return Err(OdxError::Argument(format!(
            "duplicate archive operation target '{path}'"
        )));
    }
    target.insert(path, PendingEntry { bytes, compression });
    Ok(())
}

fn append_fast_path(path: &Path, adds: &BTreeMap<String, PendingEntry>) -> Result<()> {
    let file = OpenOptions::new().read(true).write(true).open(path)?;
    let mut writer = zip::ZipWriter::new_append(file)?;
    for (entry_path, pending) in adds {
        write_entry(&mut writer, entry_path, pending)?;
    }
    writer.finish()?;
    Ok(())
}

fn rewrite_path(path: &Path, normalized: &NormalizedOps) -> Result<()> {
    let source_file = File::open(path)?;
    let mut source = zip::ZipArchive::new(source_file)?;
    let comment = source.comment().to_vec();
    let zip64_comment = source.zip64_comment().map(|bytes| bytes.to_vec());

    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let temp = NamedTempFile::new_in(parent)?;
    let out_file = temp.reopen()?;
    let mut writer = zip::ZipWriter::new(out_file);
    writer.set_raw_comment(comment.into_boxed_slice());
    writer.set_raw_zip64_comment(zip64_comment.map(Vec::into_boxed_slice));

    for index in 0..source.len() {
        let file = source.by_index_raw(index)?;
        let name = file.name().to_string();
        if normalized.should_skip(&name) {
            continue;
        }
        writer.raw_copy_file(file)?;
    }

    for (entry_path, pending) in &normalized.replaces {
        write_entry(&mut writer, entry_path, pending)?;
    }
    for (entry_path, pending) in &normalized.adds {
        write_entry(&mut writer, entry_path, pending)?;
    }

    writer.finish()?;
    replace_archive(temp, path)
}

fn write_entry<W: Write + std::io::Seek>(
    writer: &mut zip::ZipWriter<W>,
    path: &str,
    pending: &PendingEntry,
) -> Result<()> {
    let options = SimpleFileOptions::default()
        .compression_method(pending.compression)
        .large_file(true);
    writer.start_file(path, options)?;
    writer.write_all(&pending.bytes)?;
    Ok(())
}

fn replace_archive(temp: NamedTempFile, path: &Path) -> Result<()> {
    match temp.persist(path) {
        Ok(_) => Ok(()),
        Err(err) => {
            let tempfile::PersistError { error: _, file } = err;
            if path.exists() {
                fs::remove_file(path)?;
                file.persist(path)
                    .map_err(|persist_err| persist_err.error)?;
                Ok(())
            } else {
                Err(OdxError::Io(err.error))
            }
        }
    }
}

fn normalize_prefix(prefix: String) -> String {
    prefix.trim_end_matches('/').to_string()
}

fn path_matches_prefix(path: &str, prefix: &str) -> bool {
    path == prefix
        || path
            .strip_prefix(prefix)
            .is_some_and(|suffix| suffix.starts_with('/'))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_archive(path: &Path, comment: &[u8], entries: &[(&str, &[u8])]) {
        let file = File::create(path).unwrap();
        let mut writer = zip::ZipWriter::new(file);
        writer.set_raw_comment(comment.to_vec().into_boxed_slice());
        let options = SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored)
            .large_file(true);
        for (name, bytes) in entries {
            writer.start_file(name, options).unwrap();
            writer.write_all(bytes).unwrap();
        }
        writer.finish().unwrap();
    }

    #[test]
    fn append_only_adds_new_entries() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("append.odx");
        create_archive(&path, b"comment", &[("header.json", b"{}")]);

        apply_archive_ops(
            &path,
            vec![ArchiveOp::Add {
                path: "dpf/labels.uint8".into(),
                bytes: vec![1, 2, 3],
                compression: zip::CompressionMethod::Stored,
            }],
        )
        .unwrap();

        let entries = archive_entry_names(&path).unwrap();
        assert!(entries.contains("header.json"));
        assert!(entries.contains("dpf/labels.uint8"));
        assert_eq!(
            read_archive_entry(&path, "dpf/labels.uint8").unwrap(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn rewrite_replaces_and_deletes_entries() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("rewrite.odx");
        create_archive(
            &path,
            b"archive-comment",
            &[
                ("header.json", b"{}"),
                ("dpf/keep.float32", b"keep"),
                ("dpf/replace.float32", b"old"),
                ("dpv/drop.float32", b"drop"),
            ],
        );

        apply_archive_ops(
            &path,
            vec![
                ArchiveOp::Replace {
                    path: "dpf/replace.float32".into(),
                    bytes: b"new".to_vec(),
                    compression: zip::CompressionMethod::Stored,
                },
                ArchiveOp::Delete {
                    path: "dpv/drop.float32".into(),
                },
            ],
        )
        .unwrap();

        let file = File::open(&path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        assert_eq!(archive.comment(), b"archive-comment");
        assert_eq!(
            read_archive_entry(&path, "dpf/keep.float32").unwrap(),
            b"keep".to_vec()
        );
        assert_eq!(
            read_archive_entry(&path, "dpf/replace.float32").unwrap(),
            b"new".to_vec()
        );
        assert!(archive.by_name("dpv/drop.float32").is_err());
    }

    #[test]
    fn delete_prefix_removes_matching_paths_only() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("prefix.odx");
        create_archive(
            &path,
            b"",
            &[
                ("dpg/group_a/color.uint8", b"1"),
                ("dpg/group_a/size.uint8", b"2"),
                ("dpg/group_b/color.uint8", b"3"),
            ],
        );

        apply_archive_ops(
            &path,
            vec![ArchiveOp::DeletePrefix {
                prefix: "dpg/group_a".into(),
            }],
        )
        .unwrap();

        let entries = archive_entry_names(&path).unwrap();
        assert!(!entries.contains("dpg/group_a/color.uint8"));
        assert!(!entries.contains("dpg/group_a/size.uint8"));
        assert!(entries.contains("dpg/group_b/color.uint8"));
    }
}
