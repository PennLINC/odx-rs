#[doc(hidden)]
pub mod cli_support;
pub mod data_array;
pub mod dtype;
pub mod error;
pub mod formats;
pub mod header;
pub mod interop;
pub mod io;
pub mod mmap_backing;
pub mod mrtrix_sh;
pub mod odx_file;
pub mod stream;
pub mod typed_view;
pub mod validate;

pub use data_array::{DataArray, DataArrayInfo};
pub use dtype::{DType, OdxScalar};
pub use error::{OdxError, Result};
pub use formats::dsistudio;
pub use formats::mif;
pub use formats::mrtrix;
pub use formats::pam;
pub use header::{CanonicalDenseRepresentation, Header, QuantizationSpec};
pub use interop::{
    dsistudio_to_mrtrix, mrtrix_to_dsistudio, DenseOdfMode, DsistudioFormat,
    DsistudioToMrtrixOptions, MrtrixToDsistudioOptions, PeakSource, Z0Policy,
};
pub use mmap_backing::MmapBacking;
pub use odx_file::{OdxDataset, OdxFile, OdxWritePolicy};
pub use stream::{OdxBuilder, OdxStream};
pub use typed_view::TypedView2D;
pub use validate::{
    validate_dataset, validate_dataset_detailed, ValidationIssue, ValidationSeverity,
};
