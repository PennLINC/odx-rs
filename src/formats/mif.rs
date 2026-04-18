use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use crate::error::{OdxError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MifLayoutAxis {
    pub rank: usize,
    pub negative: bool,
}

impl MifLayoutAxis {
    pub fn parse(token: &str) -> Result<Self> {
        let token = token.trim();
        if token.is_empty() {
            return Err(OdxError::Format("empty MIF layout token".into()));
        }

        let (negative, digits) = match token.as_bytes()[0] {
            b'+' => (false, &token[1..]),
            b'-' => (true, &token[1..]),
            b'0'..=b'9' => (false, token),
            _ => return Err(OdxError::Format(format!("bad MIF layout token: '{token}'"))),
        };
        let rank = digits
            .parse::<usize>()
            .map_err(|_| OdxError::Format(format!("bad MIF layout token: '{token}'")))?;
        Ok(Self { rank, negative })
    }
}

#[derive(Debug, Clone)]
pub struct MifHeader {
    pub dimensions: Vec<usize>,
    pub voxel_sizes: Vec<f64>,
    pub datatype: String,
    pub layout: Vec<MifLayoutAxis>,
    pub transform: Option<[[f64; 4]; 3]>,
    pub data_offset: usize,
    pub extra: HashMap<String, String>,
}

impl MifHeader {
    pub fn element_size(&self) -> usize {
        match self.datatype.as_str() {
            "Float64LE" | "Float64BE" | "Float64" => 8,
            "Float32LE" | "Float32BE" | "Float32" => 4,
            "UInt32LE" | "UInt32BE" | "UInt32" | "Int32LE" | "Int32BE" | "Int32" => 4,
            "Int16LE" | "Int16BE" | "Int16" | "UInt16LE" | "UInt16BE" | "UInt16" => 2,
            "UInt8" | "Int8" => 1,
            _ => 4,
        }
    }

    pub fn is_native_endian(&self) -> bool {
        if cfg!(target_endian = "little") {
            !self.datatype.ends_with("BE")
        } else {
            !self.datatype.ends_with("LE")
        }
    }

    pub fn compute_strides(&self) -> Vec<isize> {
        let dims = &self.dimensions;
        let mut axis_rank: Vec<(usize, usize, bool)> = self
            .layout
            .iter()
            .enumerate()
            .map(|(ax, axis)| (ax, axis.rank, axis.negative))
            .collect();
        axis_rank.sort_by_key(|&(_, rank, _)| rank);

        let mut strides = vec![0isize; dims.len()];
        let mut stride = 1isize;
        for &(ax, _, negative) in &axis_rank {
            strides[ax] = if negative { -stride } else { stride };
            stride *= dims[ax] as isize;
        }
        strides
    }

    pub fn affine_4x4(&self) -> [[f64; 4]; 4] {
        if let Some(ref t) = self.transform {
            let vx = self.voxel_sizes.first().copied().unwrap_or(1.0);
            let vy = self.voxel_sizes.get(1).copied().unwrap_or(1.0);
            let vz = self.voxel_sizes.get(2).copied().unwrap_or(1.0);
            [
                [t[0][0] * vx, t[0][1] * vy, t[0][2] * vz, t[0][3]],
                [t[1][0] * vx, t[1][1] * vy, t[1][2] * vz, t[1][3]],
                [t[2][0] * vx, t[2][1] * vy, t[2][2] * vz, t[2][3]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        } else {
            let vs = &self.voxel_sizes;
            let vx = if !vs.is_empty() { vs[0] } else { 1.0 };
            let vy = if vs.len() > 1 { vs[1] } else { 1.0 };
            let vz = if vs.len() > 2 { vs[2] } else { 1.0 };
            [
                [vx, 0.0, 0.0, 0.0],
                [0.0, vy, 0.0, 0.0],
                [0.0, 0.0, vz, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        }
    }
}

#[derive(Debug)]
pub struct MifImage {
    pub header: MifHeader,
    pub data: Vec<u8>,
}

impl MifImage {
    pub fn ndim(&self) -> usize {
        self.header.dimensions.len()
    }

    pub fn nvoxels(&self) -> usize {
        self.header.dimensions.iter().product()
    }

    pub fn element_size(&self) -> usize {
        self.header.element_size()
    }

    pub fn is_native_endian(&self) -> bool {
        self.header.is_native_endian()
    }

    pub fn as_f32_vec(&self) -> Vec<f32> {
        let dt = &self.header.datatype;
        if dt.starts_with("Float32") {
            self.data
                .chunks_exact(4)
                .map(|c| match dt.as_str() {
                    "Float32BE" => f32::from_be_bytes([c[0], c[1], c[2], c[3]]),
                    _ => f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                })
                .collect()
        } else if dt.starts_with("Float64") {
            self.data
                .chunks_exact(8)
                .map(|c| {
                    let bytes = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
                    (match dt.as_str() {
                        "Float64BE" => f64::from_be_bytes(bytes),
                        _ => f64::from_le_bytes(bytes),
                    }) as f32
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    pub fn as_u32_vec(&self) -> Vec<u32> {
        let dt = &self.header.datatype;
        if !dt.starts_with("UInt32") {
            return Vec::new();
        }
        self.data
            .chunks_exact(4)
            .map(|c| match dt.as_str() {
                "UInt32BE" => u32::from_be_bytes([c[0], c[1], c[2], c[3]]),
                _ => u32::from_le_bytes([c[0], c[1], c[2], c[3]]),
            })
            .collect()
    }

    /// Compute signed physical strides in element units for each logical axis.
    pub fn compute_strides(&self) -> Vec<isize> {
        self.header.compute_strides()
    }

    pub fn logical_f32_vec(&self) -> Result<Vec<f32>> {
        if !self.header.datatype.starts_with("Float32") {
            return Err(OdxError::Format(format!(
                "logical_f32_vec requires Float32 MIF data, found {}",
                self.header.datatype
            )));
        }
        Ok(reorder_logical(
            &self.as_f32_vec(),
            &self.header.dimensions,
            &self.compute_strides(),
        ))
    }

    pub fn logical_u32_vec(&self) -> Result<Vec<u32>> {
        if !self.header.datatype.starts_with("UInt32") {
            return Err(OdxError::Format(format!(
                "logical_u32_vec requires UInt32 MIF data, found {}",
                self.header.datatype
            )));
        }
        Ok(reorder_logical(
            &self.as_u32_vec(),
            &self.header.dimensions,
            &self.compute_strides(),
        ))
    }

    pub fn affine_4x4(&self) -> [[f64; 4]; 4] {
        self.header.affine_4x4()
    }
}

fn reorder_logical<T: Copy>(raw: &[T], dims: &[usize], strides: &[isize]) -> Vec<T> {
    let total: usize = dims.iter().product();
    if total == 0 {
        return Vec::new();
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
        logical.push(raw[raw_index as usize]);

        for axis in (0..dims.len()).rev() {
            coords[axis] += 1;
            if coords[axis] < dims[axis] {
                break;
            }
            coords[axis] = 0;
        }
    }

    logical
}

pub fn read_mif(path: &Path) -> Result<MifImage> {
    let raw_bytes = read_mif_bytes(path)?;

    parse_mif(&raw_bytes)
}

pub(crate) fn read_mif_bytes(path: &Path) -> Result<Vec<u8>> {
    if path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e == "gz")
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

pub(crate) fn parse_mif_header(bytes: &[u8]) -> Result<MifHeader> {
    let header_str = find_header_end(bytes)?;
    parse_header(header_str)
}

fn parse_mif(bytes: &[u8]) -> Result<MifImage> {
    let header_str = find_header_end(bytes)?;
    let header = parse_header(header_str)?;
    let data = bytes[header.data_offset..].to_vec();
    Ok(MifImage { header, data })
}

fn find_header_end(bytes: &[u8]) -> Result<&str> {
    let search_limit = bytes.len().min(1024 * 1024);
    for i in 0..search_limit.saturating_sub(4) {
        if bytes[i] == b'E' && bytes[i + 1] == b'N' && bytes[i + 2] == b'D' && bytes[i + 3] == b'\n'
        {
            if i == 0 || bytes[i - 1] == b'\n' {
                let hdr_end = i + 4;
                return Ok(std::str::from_utf8(&bytes[..hdr_end])
                    .map_err(|_| OdxError::Format("MIF header not valid UTF-8".into()))?);
            }
        }
    }

    Err(OdxError::Format("MIF END marker not found".into()))
}

fn parse_header(text: &str) -> Result<MifHeader> {
    let mut dimensions = Vec::new();
    let mut voxel_sizes = Vec::new();
    let mut datatype = String::new();
    let mut layout = Vec::new();
    let mut transform_rows: Vec<[f64; 4]> = Vec::new();
    let mut data_offset = 0usize;
    let mut extra = HashMap::new();

    let mut first_line = true;
    for line in text.lines() {
        if first_line {
            first_line = false;
            if !line.starts_with("mrtrix image") {
                return Err(OdxError::Format(format!(
                    "MIF does not start with 'mrtrix image': '{line}'"
                )));
            }
            continue;
        }

        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() || line == "END" {
            continue;
        }

        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim();

        match key {
            "dim" => {
                dimensions = value
                    .split(',')
                    .map(|s| {
                        s.trim()
                            .parse::<usize>()
                            .map_err(|_| OdxError::Format(format!("bad MIF dim: '{s}'")))
                    })
                    .collect::<Result<_>>()?;
            }
            "vox" => {
                voxel_sizes = value
                    .split(',')
                    .map(|s| {
                        s.trim()
                            .parse::<f64>()
                            .map_err(|_| OdxError::Format(format!("bad MIF vox: '{s}'")))
                    })
                    .collect::<Result<_>>()?;
            }
            "datatype" => {
                datatype = value.to_string();
            }
            "layout" => {
                layout = value
                    .split(',')
                    .map(MifLayoutAxis::parse)
                    .collect::<Result<_>>()?;
            }
            "transform" => {
                let vals: Vec<f64> = value
                    .split(',')
                    .map(|s| {
                        s.trim()
                            .parse::<f64>()
                            .map_err(|_| OdxError::Format(format!("bad MIF transform: '{s}'")))
                    })
                    .collect::<Result<_>>()?;
                if vals.len() >= 4 {
                    transform_rows.push([vals[0], vals[1], vals[2], vals[3]]);
                }
            }
            "file" => {
                let parts: Vec<&str> = value.split_whitespace().collect();
                if parts.len() >= 2 && parts[0] == "." {
                    data_offset = parts[1]
                        .parse::<usize>()
                        .map_err(|_| OdxError::Format(format!("bad MIF file offset: '{value}'")))?;
                }
            }
            _ => {
                extra.insert(key.to_string(), value.to_string());
            }
        }
    }

    if dimensions.is_empty() {
        return Err(OdxError::Format("MIF missing 'dim' field".into()));
    }
    if datatype.is_empty() {
        return Err(OdxError::Format("MIF missing 'datatype' field".into()));
    }

    let transform = if transform_rows.len() >= 3 {
        Some([transform_rows[0], transform_rows[1], transform_rows[2]])
    } else {
        None
    };

    Ok(MifHeader {
        dimensions,
        voxel_sizes,
        datatype,
        layout,
        transform,
        data_offset,
        extra,
    })
}
