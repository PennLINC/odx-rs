use crate::dtype::DType;
use crate::error::{OdxError, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OdxFilename {
    pub name: String,
    pub ncols: usize,
    pub dtype: DType,
}

impl OdxFilename {
    pub fn parse(stem: &str) -> Result<Self> {
        let parts3: Vec<&str> = stem.rsplitn(3, '.').collect();

        if parts3.len() == 3 {
            if let Ok(ncols) = parts3[1].parse::<usize>() {
                if let Ok(dtype) = DType::parse(parts3[0]) {
                    return Ok(OdxFilename {
                        name: parts3[2].to_string(),
                        ncols,
                        dtype,
                    });
                }
            }
        }

        let parts2: Vec<&str> = stem.rsplitn(2, '.').collect();
        if parts2.len() == 2 {
            if let Ok(dtype) = DType::parse(parts2[0]) {
                return Ok(OdxFilename {
                    name: parts2[1].to_string(),
                    ncols: 1,
                    dtype,
                });
            }
        }

        Err(OdxError::Format(format!(
            "cannot parse ODX filename '{stem}'"
        )))
    }

    pub fn to_filename(&self) -> String {
        if self.ncols == 1 {
            format!("{}.{}", self.name, self.dtype.name())
        } else {
            format!("{}.{}.{}", self.name, self.ncols, self.dtype.name())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_directions() {
        let f = OdxFilename::parse("directions.3.float32").unwrap();
        assert_eq!(f.name, "directions");
        assert_eq!(f.ncols, 3);
        assert_eq!(f.dtype, DType::Float32);
    }

    #[test]
    fn parse_scalar() {
        let f = OdxFilename::parse("amplitude.float32").unwrap();
        assert_eq!(f.name, "amplitude");
        assert_eq!(f.ncols, 1);
        assert_eq!(f.dtype, DType::Float32);
    }

    #[test]
    fn round_trip() {
        let f = OdxFilename {
            name: "gfa".into(),
            ncols: 1,
            dtype: DType::Float32,
        };
        assert_eq!(f.to_filename(), "gfa.float32");
        assert_eq!(OdxFilename::parse(&f.to_filename()).unwrap(), f);
    }

    #[test]
    fn parse_high_ncols() {
        let f = OdxFilename::parse("amplitudes.642.float32").unwrap();
        assert_eq!(f.name, "amplitudes");
        assert_eq!(f.ncols, 642);
        assert_eq!(f.dtype, DType::Float32);
    }
}
