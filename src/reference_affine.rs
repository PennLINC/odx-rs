use std::io::Read;
use std::path::Path;

use crate::error::{OdxError, Result};
use crate::formats::mif;

pub fn read_reference_affine(path: &Path) -> Result<[[f64; 4]; 4]> {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    if name.ends_with(".mif") || name.ends_with(".mif.gz") {
        return Ok(mif::read_mif(path)?.affine_4x4());
    }
    if name.ends_with(".nii") || name.ends_with(".nii.gz") {
        return read_nifti_affine(path);
    }
    Err(OdxError::Argument(format!(
        "unsupported reference affine path '{}'; expected .mif/.mif.gz/.nii/.nii.gz",
        path.display()
    )))
}

fn read_nifti_affine(path: &Path) -> Result<[[f64; 4]; 4]> {
    let bytes = read_image_bytes(path)?;
    if bytes.len() < 540 {
        return Err(OdxError::Format(format!(
            "NIfTI file '{}' is too small to contain a valid header",
            path.display()
        )));
    }
    match i32::from_le_bytes(bytes[0..4].try_into().unwrap()) {
        348 => parse_nifti1_affine(&bytes),
        540 => parse_nifti2_affine(&bytes),
        other => Err(OdxError::Format(format!(
            "unsupported NIfTI header size {other} in '{}'",
            path.display()
        ))),
    }
}

fn read_image_bytes(path: &Path) -> Result<Vec<u8>> {
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext == "gz")
    {
        let file = std::fs::File::open(path)?;
        let mut decoder = flate2::read::MultiGzDecoder::new(file);
        let mut bytes = Vec::new();
        decoder.read_to_end(&mut bytes)?;
        Ok(bytes)
    } else {
        Ok(std::fs::read(path)?)
    }
}

fn parse_nifti1_affine(bytes: &[u8]) -> Result<[[f64; 4]; 4]> {
    if read_i16_le(bytes, 254)? > 0 {
        Ok([
            [
                read_f32_le(bytes, 280)? as f64,
                read_f32_le(bytes, 284)? as f64,
                read_f32_le(bytes, 288)? as f64,
                read_f32_le(bytes, 292)? as f64,
            ],
            [
                read_f32_le(bytes, 296)? as f64,
                read_f32_le(bytes, 300)? as f64,
                read_f32_le(bytes, 304)? as f64,
                read_f32_le(bytes, 308)? as f64,
            ],
            [
                read_f32_le(bytes, 312)? as f64,
                read_f32_le(bytes, 316)? as f64,
                read_f32_le(bytes, 320)? as f64,
                read_f32_le(bytes, 324)? as f64,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ])
    } else {
        Ok([
            [read_f32_le(bytes, 80)? as f64, 0.0, 0.0, 0.0],
            [0.0, read_f32_le(bytes, 84)? as f64, 0.0, 0.0],
            [0.0, 0.0, read_f32_le(bytes, 88)? as f64, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }
}

fn parse_nifti2_affine(bytes: &[u8]) -> Result<[[f64; 4]; 4]> {
    if read_i32_le(bytes, 348)? > 0 {
        Ok([
            [
                read_f64_le(bytes, 400)?,
                read_f64_le(bytes, 408)?,
                read_f64_le(bytes, 416)?,
                read_f64_le(bytes, 424)?,
            ],
            [
                read_f64_le(bytes, 432)?,
                read_f64_le(bytes, 440)?,
                read_f64_le(bytes, 448)?,
                read_f64_le(bytes, 456)?,
            ],
            [
                read_f64_le(bytes, 464)?,
                read_f64_le(bytes, 472)?,
                read_f64_le(bytes, 480)?,
                read_f64_le(bytes, 488)?,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ])
    } else {
        Ok([
            [read_f64_le(bytes, 112)?, 0.0, 0.0, 0.0],
            [0.0, read_f64_le(bytes, 120)?, 0.0, 0.0],
            [0.0, 0.0, read_f64_le(bytes, 128)?, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }
}

fn read_i16_le(bytes: &[u8], offset: usize) -> Result<i16> {
    let end = offset + 2;
    bytes
        .get(offset..end)
        .ok_or_else(|| OdxError::Format("unexpected EOF while reading NIfTI header".into()))
        .map(|slice| i16::from_le_bytes(slice.try_into().unwrap()))
}

fn read_i32_le(bytes: &[u8], offset: usize) -> Result<i32> {
    let end = offset + 4;
    bytes
        .get(offset..end)
        .ok_or_else(|| OdxError::Format("unexpected EOF while reading NIfTI header".into()))
        .map(|slice| i32::from_le_bytes(slice.try_into().unwrap()))
}

fn read_f32_le(bytes: &[u8], offset: usize) -> Result<f32> {
    let end = offset + 4;
    bytes
        .get(offset..end)
        .ok_or_else(|| OdxError::Format("unexpected EOF while reading NIfTI header".into()))
        .map(|slice| f32::from_le_bytes(slice.try_into().unwrap()))
}

fn read_f64_le(bytes: &[u8], offset: usize) -> Result<f64> {
    let end = offset + 8;
    bytes
        .get(offset..end)
        .ok_or_else(|| OdxError::Format("unexpected EOF while reading NIfTI header".into()))
        .map(|slice| f64::from_le_bytes(slice.try_into().unwrap()))
}
