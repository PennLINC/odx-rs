mod archive_edit;
pub mod directory;
pub mod filename;
pub mod zip;

use std::collections::HashMap;
use std::path::Path;

use crate::data_array::DataArray;
use crate::dtype::DType;
use crate::error::Result;
use crate::header::QuantizationSpec;
use crate::mmap_backing::vec_to_bytes;
use crate::odx_file::{OdxDataset, OdxWritePolicy};

pub fn load(path: &Path) -> Result<OdxDataset> {
    if path.is_dir() {
        directory::open_directory(path, None)
    } else if path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e == "odx")
    {
        zip::open_archive(path)
    } else {
        Err(crate::error::OdxError::Format(format!(
            "unrecognized ODX path: {}",
            path.display()
        )))
    }
}

pub(crate) fn append_dpf(
    path: &Path,
    dpf: &HashMap<String, DataArray>,
    overwrite: bool,
) -> Result<()> {
    if path.is_dir() {
        directory::append_dpf_to_directory(path, dpf, overwrite)
    } else if path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e == "odx")
    {
        zip::append_dpf_to_zip(path, dpf, ::zip::CompressionMethod::Deflated, overwrite)
    } else {
        Err(crate::error::OdxError::Format(format!(
            "qc_class writing requires an ODX directory or .odx archive: {}",
            path.display()
        )))
    }
}

pub(crate) fn maybe_quantize_array(
    _array_key: &str,
    arr: &DataArray,
    policy: OdxWritePolicy,
    allow_quantization: bool,
) -> (DataArray, Option<QuantizationSpec>) {
    if !policy.quantize_dense || !allow_quantization || arr.dtype() != DType::Float32 {
        return (arr.clone_owned(), None);
    }
    let values = arr.cast_slice::<f32>();
    if values.len() < policy.quantize_min_len || values.is_empty() {
        return (arr.clone_owned(), None);
    }

    let mut min_v = values[0];
    let mut max_v = values[0];
    for &v in values.iter().skip(1) {
        if v < min_v {
            min_v = v;
        }
        if v > max_v {
            max_v = v;
        }
    }
    let slope = (max_v - min_v) / 255.99f32;
    if !slope.is_finite() || slope <= 0.0 {
        return (arr.clone_owned(), None);
    }
    let inv_slope = 1.0f32 / slope;
    let quantized: Vec<u8> = values
        .iter()
        .map(|&v| ((v - min_v) * inv_slope).clamp(0.0, 255.0) as u8)
        .collect();
    let spec = QuantizationSpec {
        slope,
        intercept: min_v,
        stored_dtype: "uint8".into(),
    };
    (
        DataArray::owned_bytes(quantized, arr.ncols(), DType::UInt8),
        Some(spec),
    )
}

pub(crate) fn dequantize_array(arr: &DataArray, spec: &QuantizationSpec) -> DataArray {
    let src: &[u8] = arr.cast_slice();
    let decoded: Vec<f32> = src
        .iter()
        .map(|&v| v as f32 * spec.slope + spec.intercept)
        .collect();
    DataArray::owned_bytes(vec_to_bytes(decoded), arr.ncols(), DType::Float32)
}

pub(crate) fn normalize_float_array(arr: &DataArray) -> Result<DataArray> {
    if arr.dtype() == DType::Float32 {
        return Ok(arr.clone_owned());
    }
    let values = arr.to_f32_vec()?;
    Ok(DataArray::owned_bytes(
        vec_to_bytes(values),
        arr.ncols(),
        DType::Float32,
    ))
}
