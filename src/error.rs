use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum OdxError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[cfg(feature = "pam5")]
    #[error("HDF5 error: {0}")]
    Hdf5(#[from] hdf5_metno::Error),

    #[error("ZIP error: {0}")]
    Zip(#[from] zip::result::ZipError),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("format error: {0}")]
    Format(String),

    #[error("unsupported dtype: {0}")]
    DType(String),

    #[error("invalid argument: {0}")]
    Argument(String),

    #[error("file not found: {}", .0.display())]
    FileNotFound(PathBuf),
}

pub type Result<T> = std::result::Result<T, OdxError>;
