use std::collections::HashMap;
use std::io::Read;
use std::ops::Range;
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{OdxError, Result};
use crate::mmap_backing::vec_to_bytes;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatStorageMode {
    Regular,
    SlopedU8,
    Masked,
    MaskedSlopedU8,
}

#[derive(Debug, Clone)]
struct MatRecordMeta {
    name: String,
    mrows: usize,
    ncols: usize,
    type_flag: u32,
    data_range: Range<usize>,
}

#[derive(Debug, Clone)]
pub struct MatCatalog {
    bytes: Arc<[u8]>,
    records: Vec<MatRecordMeta>,
    name_to_index: HashMap<String, usize>,
}

#[derive(Clone, Copy)]
pub struct MatRecord<'a> {
    catalog: &'a MatCatalog,
    meta: &'a MatRecordMeta,
}

impl MatCatalog {
    pub fn get(&self, name: &str) -> Option<MatRecord<'_>> {
        self.name_to_index.get(name).map(|&idx| MatRecord {
            catalog: self,
            meta: &self.records[idx],
        })
    }

    pub fn has(&self, name: &str) -> bool {
        self.name_to_index.contains_key(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = MatRecord<'_>> {
        self.records.iter().map(|meta| MatRecord {
            catalog: self,
            meta,
        })
    }
}

impl<'a> MatRecord<'a> {
    fn elem_size(type_flag: u32) -> Result<usize> {
        let dtype_code = (type_flag % 100) / 10;
        match dtype_code {
            0 => Ok(8),
            1 => Ok(4),
            2 => Ok(4),
            3 => Ok(2),
            4 => Ok(2),
            5 => Ok(1),
            _ => Err(OdxError::Format(format!(
                "unsupported MAT4 type code {type_flag}"
            ))),
        }
    }

    pub fn name(&self) -> &str {
        &self.meta.name
    }

    pub fn mrows(&self) -> usize {
        self.meta.mrows
    }

    pub fn ncols(&self) -> usize {
        self.meta.ncols
    }

    pub fn type_flag(&self) -> u32 {
        self.meta.type_flag
    }

    pub fn dtype(&self) -> Result<DType> {
        let dtype_code = (self.meta.type_flag % 100) / 10;
        match dtype_code {
            0 => Ok(DType::Float64),
            1 => Ok(DType::Float32),
            2 => Ok(DType::Int32),
            3 => Ok(DType::Int16),
            4 => Ok(DType::UInt16),
            5 => Ok(DType::UInt8),
            _ => Err(OdxError::Format(format!(
                "unsupported MAT4 type code {}",
                self.meta.type_flag
            ))),
        }
    }

    pub fn data(&self) -> &'a [u8] {
        &self.catalog.bytes[self.meta.data_range.clone()]
    }

    pub fn storage_mode(&self) -> MatStorageMode {
        let has_slope = self.catalog.has(&format!("{}.slope", self.name()));
        let is_masked = self.dtype().ok() != Some(DType::UInt8)
            && self.catalog.has("mask")
            && self.catalog.get("mask").map(|m| m.count()).unwrap_or(0) == self.ncols();
        match (is_masked, has_slope, self.dtype().ok()) {
            (true, true, Some(DType::UInt8)) => MatStorageMode::MaskedSlopedU8,
            (false, true, Some(DType::UInt8)) => MatStorageMode::SlopedU8,
            (true, _, _) => MatStorageMode::Masked,
            _ => MatStorageMode::Regular,
        }
    }

    pub fn count(&self) -> usize {
        self.meta.mrows * self.meta.ncols
    }

    pub fn as_f32_vec(&self) -> Vec<f32> {
        match self.dtype() {
            Ok(DType::Float64) => self
                .data()
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            Ok(DType::Float32) => self
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            Ok(DType::Int32) => self
                .data()
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            Ok(DType::Int16) => self
                .data()
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            Ok(DType::UInt16) => self
                .data()
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            Ok(DType::UInt8) => {
                let base: Vec<f32> = self.data().iter().map(|&v| v as f32).collect();
                if let (Some(slope), Some(intercept)) = (
                    self.catalog
                        .get(&format!("{}.slope", self.name()))
                        .and_then(|r| r.scalar_f32().ok()),
                    self.catalog
                        .get(&format!("{}.inter", self.name()))
                        .and_then(|r| r.scalar_f32().ok()),
                ) {
                    base.into_iter().map(|v| v * slope + intercept).collect()
                } else {
                    base
                }
            }
            _ => Vec::new(),
        }
    }

    pub fn as_i32_vec(&self) -> Vec<i32> {
        match self.dtype() {
            Ok(DType::Float64) => self
                .data()
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as i32)
                .collect(),
            Ok(DType::Float32) => self
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as i32)
                .collect(),
            Ok(DType::Int32) => self
                .data()
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            Ok(DType::Int16) => self
                .data()
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes(c.try_into().unwrap()) as i32)
                .collect(),
            Ok(DType::UInt16) => self
                .data()
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes(c.try_into().unwrap()) as i32)
                .collect(),
            Ok(DType::UInt8) => self.data().iter().map(|&v| v as i32).collect(),
            _ => Vec::new(),
        }
    }

    pub fn as_u8_slice(&self) -> Option<&'a [u8]> {
        (self.dtype().ok() == Some(DType::UInt8)).then_some(self.data())
    }

    pub fn scalar_f32(&self) -> Result<f32> {
        self.as_f32_vec()
            .into_iter()
            .next()
            .ok_or_else(|| OdxError::Format(format!("record '{}' is empty", self.name())))
    }
}

pub fn read_mat4_gz(path: &std::path::Path) -> Result<MatCatalog> {
    let file = std::fs::File::open(path)?;
    let mut decoder = flate2::read::MultiGzDecoder::new(file);
    let mut bytes = Vec::new();
    decoder.read_to_end(&mut bytes)?;
    read_mat4(&bytes)
}

pub fn read_mat4(bytes: &[u8]) -> Result<MatCatalog> {
    let bytes: Arc<[u8]> = bytes.to_vec().into();
    let mut records = Vec::new();
    let mut name_to_index = HashMap::new();
    let mut cursor = 0usize;

    while cursor < bytes.len() {
        if cursor + 20 > bytes.len() {
            break;
        }

        let type_flag = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
        let mrows = u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap()) as usize;
        let ncols = u32::from_le_bytes(bytes[cursor + 8..cursor + 12].try_into().unwrap()) as usize;
        let _imagf = u32::from_le_bytes(bytes[cursor + 12..cursor + 16].try_into().unwrap());
        let namlen =
            u32::from_le_bytes(bytes[cursor + 16..cursor + 20].try_into().unwrap()) as usize;
        cursor += 20;

        if cursor + namlen > bytes.len() {
            return Err(OdxError::Format("truncated MAT4 record name".into()));
        }
        let name_bytes = &bytes[cursor..cursor + namlen];
        cursor += namlen;
        let name = name_bytes.split(|b| *b == 0).next().unwrap_or_default();
        let name = std::str::from_utf8(name)
            .map_err(|_| OdxError::Format("MAT4 record name is not valid UTF-8".into()))?
            .to_string();

        let elem_size = MatRecord::elem_size(type_flag)?;
        let payload_len = mrows
            .checked_mul(ncols)
            .and_then(|n| n.checked_mul(elem_size))
            .ok_or_else(|| OdxError::Format(format!("MAT4 record '{name}' overflow")))?;

        if cursor + payload_len > bytes.len() {
            return Err(OdxError::Format(format!(
                "truncated MAT4 payload for '{name}'"
            )));
        }
        let data_range = cursor..cursor + payload_len;
        cursor += payload_len;

        let idx = records.len();
        records.push(MatRecordMeta {
            name: name.clone(),
            mrows,
            ncols,
            type_flag,
            data_range,
        });
        name_to_index.insert(name, idx);
    }

    Ok(MatCatalog {
        bytes,
        records,
        name_to_index,
    })
}

#[derive(Debug, Clone)]
pub struct OwnedMatRecord {
    pub name: String,
    pub mrows: usize,
    pub ncols: usize,
    pub type_flag: u32,
    pub data: Vec<u8>,
    pub subrecords: Vec<OwnedMatRecord>,
}

pub fn write_mat4_gz(path: &std::path::Path, records: &[OwnedMatRecord]) -> Result<()> {
    let out = std::fs::File::create(path)?;
    let mut encoder = flate2::write::GzEncoder::new(out, flate2::Compression::default());
    write_mat4(&mut encoder, records)?;
    encoder.finish()?;
    Ok(())
}

pub fn write_mat4<W: std::io::Write>(writer: &mut W, records: &[OwnedMatRecord]) -> Result<()> {
    for record in records {
        write_owned_record(writer, record)?;
    }
    Ok(())
}

fn write_owned_record<W: std::io::Write>(writer: &mut W, record: &OwnedMatRecord) -> Result<()> {
    writer.write_all(&record.type_flag.to_le_bytes())?;
    writer.write_all(&(record.mrows as u32).to_le_bytes())?;
    writer.write_all(&(record.ncols as u32).to_le_bytes())?;
    writer.write_all(&0u32.to_le_bytes())?;
    let namlen = record.name.len() + 1;
    writer.write_all(&(namlen as u32).to_le_bytes())?;
    writer.write_all(record.name.as_bytes())?;
    writer.write_all(&[0u8])?;
    writer.write_all(&record.data)?;
    for subrecord in &record.subrecords {
        write_owned_record(writer, subrecord)?;
    }
    Ok(())
}

pub fn float_record(
    name: impl Into<String>,
    values: Vec<f32>,
    mrows: usize,
    ncols: usize,
) -> OwnedMatRecord {
    OwnedMatRecord {
        name: name.into(),
        mrows,
        ncols,
        type_flag: 10,
        data: vec_to_bytes(values),
        subrecords: Vec::new(),
    }
}

pub fn int16_record(
    name: impl Into<String>,
    values: Vec<i16>,
    mrows: usize,
    ncols: usize,
) -> OwnedMatRecord {
    OwnedMatRecord {
        name: name.into(),
        mrows,
        ncols,
        type_flag: 30,
        data: vec_to_bytes(values),
        subrecords: Vec::new(),
    }
}

pub fn uint8_record(
    name: impl Into<String>,
    values: Vec<u8>,
    mrows: usize,
    ncols: usize,
) -> OwnedMatRecord {
    OwnedMatRecord {
        name: name.into(),
        mrows,
        ncols,
        type_flag: 50,
        data: values,
        subrecords: Vec::new(),
    }
}
