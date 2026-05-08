use bytemuck::{Pod, Zeroable};
use half::f16;
use std::fmt;

use crate::error::{OdxError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
}

impl DType {
    pub fn size_of(self) -> usize {
        match self {
            DType::Int8 | DType::UInt8 => 1,
            DType::Float16 | DType::Int16 | DType::UInt16 => 2,
            DType::Float32 | DType::Int32 | DType::UInt32 => 4,
            DType::Float64 | DType::Int64 | DType::UInt64 => 8,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            DType::Float16 => "float16",
            DType::Float32 => "float32",
            DType::Float64 => "float64",
            DType::Int8 => "int8",
            DType::Int16 => "int16",
            DType::Int32 => "int32",
            DType::Int64 => "int64",
            DType::UInt8 => "uint8",
            DType::UInt16 => "uint16",
            DType::UInt32 => "uint32",
            DType::UInt64 => "uint64",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "float16" => Ok(DType::Float16),
            "float32" => Ok(DType::Float32),
            "float64" => Ok(DType::Float64),
            "int8" => Ok(DType::Int8),
            "int16" => Ok(DType::Int16),
            "int32" => Ok(DType::Int32),
            "int64" => Ok(DType::Int64),
            "uint8" => Ok(DType::UInt8),
            "uint16" => Ok(DType::UInt16),
            "uint32" => Ok(DType::UInt32),
            "uint64" => Ok(DType::UInt64),
            _ => Err(OdxError::DType(s.to_string())),
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, DType::Float16 | DType::Float32 | DType::Float64)
    }

    /// Map a numpy-style dtype descriptor (e.g. `"float32"`, `"<f4"`, `"uint8"`,
    /// `"|u1"`) to a `DType`. Used by the Python wrapper when accepting numpy
    /// arrays via the builder.
    pub fn from_numpy_str(s: &str) -> Result<Self> {
        if let Ok(d) = Self::parse(s) {
            return Ok(d);
        }
        let trimmed = s.trim_start_matches(|c| c == '<' || c == '>' || c == '=' || c == '|');
        let mapped = match trimmed {
            "f2" => "float16",
            "f4" => "float32",
            "f8" => "float64",
            "i1" => "int8",
            "i2" => "int16",
            "i4" => "int32",
            "i8" => "int64",
            "u1" => "uint8",
            "u2" => "uint16",
            "u4" => "uint32",
            "u8" => "uint64",
            other => other,
        };
        Self::parse(mapped)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

pub trait OdxScalar: Pod + Zeroable + Copy + 'static + fmt::Debug {
    const DTYPE: DType;
    fn to_f32(self) -> f32;
    fn to_f64(self) -> f64;
}

impl OdxScalar for f32 {
    const DTYPE: DType = DType::Float32;
    fn to_f32(self) -> f32 {
        self
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl OdxScalar for f64 {
    const DTYPE: DType = DType::Float64;
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn to_f64(self) -> f64 {
        self
    }
}

impl OdxScalar for f16 {
    const DTYPE: DType = DType::Float16;
    fn to_f32(self) -> f32 {
        f16::to_f32(self)
    }
    fn to_f64(self) -> f64 {
        f16::to_f64(self)
    }
}

macro_rules! impl_odx_scalar_int {
    ($($t:ty => $d:ident),+ $(,)?) => {
        $(
            impl OdxScalar for $t {
                const DTYPE: DType = DType::$d;
                fn to_f32(self) -> f32 { self as f32 }
                fn to_f64(self) -> f64 { self as f64 }
            }
        )+
    };
}

impl_odx_scalar_int!(
    i8 => Int8, i16 => Int16, i32 => Int32, i64 => Int64,
    u8 => UInt8, u16 => UInt16, u32 => UInt32, u64 => UInt64,
);
