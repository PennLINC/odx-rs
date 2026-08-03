#[doc(hidden)]
pub mod cli_support;
pub mod combine;
pub mod compare;
pub mod data_array;
pub mod descoteaux_sh;
pub mod dtype;
pub mod error;
pub(crate) mod fixel_match;
pub mod formats;
pub mod header;
pub mod interop;
pub mod io;
pub mod mmap_backing;
pub mod mrtrix_sh;
pub mod nifti_canon;
pub mod nifti_export;
pub mod odx_file;
pub mod peak_finder;
pub mod qc;
pub mod reference_affine;
pub mod densify;
pub mod sh_basis_evaluator;
pub mod sphere_lookup;
pub mod stream;
pub mod template;
pub mod transform;
pub mod typed_view;
pub mod validate;

pub use data_array::{DataArray, DataArrayInfo};
pub use dtype::{DType, OdxScalar};
pub use error::{OdxError, Result};
pub use combine::{
    combine_odx, CombineInput, CombineOptions, CombineOutputs, CombineReport, CombineSubjectRow,
    MaskCombine, NormalizeFod, TemplateMethod,
};
pub use template::{AggregateOptions, LmaxPolicy, LooMode, ScaleMode, ShTarget};
pub use compare::{compare_odx, CompareOptions, CompareReport};
pub use formats::dsistudio;
pub use formats::mif;
pub use formats::mrtrix;
pub use formats::pam;
pub use formats::tortoise_mapmri;
pub use header::{CanonicalDenseRepresentation, Header, QuantizationSpec};
pub use interop::{
    convert_sh_basis, dsistudio_to_mrtrix, mrtrix_to_dsistudio, DenseOdfMode, DsistudioFormat,
    DsistudioToMrtrixOptions, MrtrixToDsistudioOptions, PeakSource, ShBasisTarget, Z0Policy,
};
pub use mmap_backing::MmapBacking;
pub use nifti_canon::CanonTransform;
pub use nifti_export::{
    attach_dpv_from_volume, dpv_to_volume, nifti_header_for_grid, volume_to_dpv,
    write_voxel_scalar_nifti_f32, write_voxel_scalar_nifti_u16, write_voxel_scalar_nifti_u32,
    write_voxel_scalar_nifti_u8, DpvAttachReport, DpvDtype, ATTACH_AFFINE_TOLERANCE_MM,
};
pub use odx_file::{OdxDataset, OdxFile, OdxWritePolicy};
pub use qc::{
    compute_fixel_qc, write_qc_class_dpf, FixelQcClass, FixelQcComputation, FixelQcOptions,
    FixelQcReport, PartitionStats, PartitionValueStats, ThresholdMode, QC_CLASS_DPF_NAME,
};
pub use peak_finder::{peaks_from_sh_rows, PeakFinderConfig, SpherePeakFinder};
pub use reference_affine::read_reference_affine;
pub use sh_basis_evaluator::ShBasisEvaluator;
pub use stream::{OdxBuilder, OdxStream};
pub use typed_view::TypedView2D;
pub use validate::{
    validate_dataset, validate_dataset_detailed, ValidationIssue, ValidationSeverity,
};
pub use transform::{
    apply_transform, apply_transform_h5, TransformMode, TransformOptions,
};
pub use transform::upsample::{upsample, UpsampleOptions};
