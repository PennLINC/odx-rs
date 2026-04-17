use bytemuck::{cast_slice, Pod};
use std::collections::HashMap;

use crate::dtype::DType;
use crate::error::{OdxError, Result};
use crate::mmap_backing::MmapBacking;
use crate::typed_view::TypedView2D;

#[derive(Debug)]
pub struct DataArray {
    backing: MmapBacking,
    ncols: usize,
    dtype: DType,
}

impl DataArray {
    pub fn owned_bytes(backing: Vec<u8>, ncols: usize, dtype: DType) -> Self {
        Self {
            backing: MmapBacking::Owned(backing),
            ncols,
            dtype,
        }
    }

    pub(crate) fn from_backing(backing: MmapBacking, ncols: usize, dtype: DType) -> Self {
        Self {
            backing,
            ncols,
            dtype,
        }
    }

    pub fn clone_owned(&self) -> Self {
        Self::owned_bytes(self.backing.as_bytes().to_vec(), self.ncols, self.dtype)
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn len_bytes(&self) -> usize {
        self.backing.len()
    }

    pub fn nrows(&self) -> usize {
        let row_bytes = self.ncols * self.dtype.size_of();
        if row_bytes == 0 {
            0
        } else {
            self.len_bytes() / row_bytes
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.backing.as_bytes()
    }

    pub fn cast_slice<T: Pod>(&self) -> &[T] {
        self.backing.cast_slice()
    }

    pub fn typed_view<T: Pod>(&self) -> TypedView2D<'_, T> {
        let data: &[T] = cast_slice(self.as_bytes());
        TypedView2D::new(data, self.ncols)
    }

    pub fn info(&self) -> DataArrayInfo {
        DataArrayInfo {
            ncols: self.ncols,
            nrows: self.nrows(),
            dtype: self.dtype,
        }
    }

    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        Ok(match self.dtype() {
            DType::Float16 => self
                .cast_slice::<half::f16>()
                .iter()
                .map(|v| v.to_f32())
                .collect(),
            DType::Float32 => self.cast_slice::<f32>().to_vec(),
            DType::Float64 => self.cast_slice::<f64>().iter().map(|&v| v as f32).collect(),
            DType::Int8 => self.cast_slice::<i8>().iter().map(|&v| v as f32).collect(),
            DType::Int16 => self.cast_slice::<i16>().iter().map(|&v| v as f32).collect(),
            DType::Int32 => self.cast_slice::<i32>().iter().map(|&v| v as f32).collect(),
            DType::UInt8 => self.cast_slice::<u8>().iter().map(|&v| v as f32).collect(),
            DType::UInt16 => self.cast_slice::<u16>().iter().map(|&v| v as f32).collect(),
            DType::UInt32 => self.cast_slice::<u32>().iter().map(|&v| v as f32).collect(),
            other => {
                return Err(OdxError::DType(format!(
                    "cannot convert array with dtype {other} to float32"
                )))
            }
        })
    }
}

pub type DataPerGroup = HashMap<String, HashMap<String, DataArray>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataArrayInfo {
    pub ncols: usize,
    pub nrows: usize,
    pub dtype: DType,
}

pub(crate) fn read_scalar_array_as_f32(
    arr: &DataArray,
    kind: &str,
    name: &str,
) -> Result<Vec<f32>> {
    if arr.ncols() != 1 {
        return Err(OdxError::Argument(format!(
            "{kind} '{name}' has {} columns; expected a scalar field",
            arr.ncols()
        )));
    }

    let values = arr.to_f32_vec().map_err(|err| match err {
        OdxError::DType(_) => OdxError::DType(format!(
            "{kind} '{name}' uses unsupported scalar dtype {}",
            arr.dtype()
        )),
        other => other,
    })?;

    Ok(values)
}
