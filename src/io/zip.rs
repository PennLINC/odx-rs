use std::fs;
use std::io::Write;
use std::path::Path;
use zip::write::SimpleFileOptions;

use crate::error::Result;
use crate::odx_file::{OdxDataset, OdxWritePolicy};

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
