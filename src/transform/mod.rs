//! Apply ITK-Composite spatial transforms (ANTs `*Composite.h5`) to ODX
//! datasets. Reads the transform via the [`itk_transforms_rs`] crate and
//! resamples the input ODX onto a user-specified target grid in RAS+ mm.
//!
//! Two workflow modes:
//!
//! - [`TransformMode::Mrtrix`] (default): pull-based for SH, DPV, and fixels.
//!   Single forward h5 (chain maps target→source). Matches the conventions of
//!   mrtrix3 `mrtransform` and `fixeltransform`. Modulation off by default;
//!   opt in via `modulate_sh` for `mrtransform -modulate fod` semantics.
//!
//! - [`TransformMode::Ants`]: SH/DPV pulled via the forward h5; fixels
//!   *pushed* via a second "inverse" h5 (chain maps source→target). Each
//!   source fixel maps to exactly one target voxel (cardinality preserved).
//!   For ANTs paired output, pass `from-A_to-B.h5` as forward and
//!   `from-B_to-A.h5` as inverse.

mod resample;
mod sh_apsf;
pub mod source_volume;
pub mod upsample;

pub use resample::TransformOptions;

use std::path::Path;

use itk_transforms_rs::{read_itk, TargetGrid, TransformChain};

use crate::error::{OdxError, Result};
use crate::odx_file::OdxDataset;

/// Which workflow convention to apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransformMode {
    /// Pull SH/DPV/fixels via a single forward h5 (matches mrtrix3
    /// `mrtransform` + `fixeltransform`). The forward h5 stores a chain
    /// that maps target coords → source coords.
    Mrtrix,
    /// Pull SH/DPV via the forward h5; push fixels via an inverse h5
    /// (matches the semantically natural treatment of fixels as
    /// coordinate-bearing entities). The inverse h5 stores a chain that
    /// maps source coords → target coords.
    Ants,
}

impl Default for TransformMode {
    fn default() -> Self {
        Self::Mrtrix
    }
}

/// High-level entry point: read the h5 transform(s), resolve the target grid,
/// and resample.
///
/// - `forward_transform` — required for both modes.
///     - In [`TransformMode::Mrtrix`]: pull chain for everything.
///     - In [`TransformMode::Ants`]: pull chain for SH/DPV only.
/// - `inverse_transform` — required only for [`TransformMode::Ants`]
///   (push chain for fixels). Must be `None` for `Mrtrix`.
/// - `reference_nifti` — optional NIfTI in target space; required if the
///   forward h5 contains no displacement field (affine-only chain).
/// - `invert` — swap moving↔fixed direction. Only valid with affine-only
///   chains (warps cannot be numerically inverted in v1).
pub fn apply_transform_h5(
    input: &OdxDataset,
    mode: TransformMode,
    forward_transform: &Path,
    inverse_transform: Option<&Path>,
    reference_nifti: Option<&Path>,
    invert: bool,
    opts: &TransformOptions,
) -> Result<OdxDataset> {
    // ---- Mode/argument validation up front, fail-fast.
    match mode {
        TransformMode::Mrtrix => {
            if inverse_transform.is_some() {
                return Err(OdxError::Argument(
                    "--mode mrtrix does not accept --transform-inverse \
                     (mrtrix mode pulls everything via the single forward h5). \
                     Use --mode ants if you want push-based fixel handling."
                        .into(),
                ));
            }
        }
        TransformMode::Ants => {
            if inverse_transform.is_none() {
                return Err(OdxError::Argument(
                    "--mode ants requires --transform-inverse (the second h5 \
                     storing the source→target chain, e.g. from-MNI_to-ACPC.h5 \
                     when --transform is from-ACPC_to-MNI.h5)."
                        .into(),
                ));
            }
        }
    }

    let mut chain = read_itk(forward_transform).map_err(|e| {
        OdxError::Format(format!("reading {}: {e}", forward_transform.display()))
    })?;

    if invert {
        chain = chain
            .invert()
            .map_err(|e| OdxError::Format(format!("--invert with non-invertible chain: {e}")))?;
    }

    let mut effective_opts = opts.clone();
    if let Some(p) = inverse_transform {
        let fchain = read_itk(p).map_err(|e| {
            OdxError::Format(format!("reading inverse transform {}: {e}", p.display()))
        })?;
        effective_opts.fixel_chain = Some(fchain);
    }

    let target_grid = resolve_target_grid(&chain, reference_nifti)?;
    apply_transform(input, &chain, &target_grid, &effective_opts)
}

/// Lower-level entry point: caller has already built the chain and grid.
pub fn apply_transform(
    input: &OdxDataset,
    chain: &TransformChain,
    target_grid: &TargetGrid,
    opts: &TransformOptions,
) -> Result<OdxDataset> {
    resample::run(input, chain, target_grid, opts)
}

fn resolve_target_grid(
    chain: &TransformChain,
    reference_nifti: Option<&Path>,
) -> Result<TargetGrid> {
    if let Some(p) = reference_nifti {
        return TargetGrid::from_nifti(p)
            .map_err(|e| OdxError::Format(format!("reference NIfTI {}: {e}", p.display())));
    }
    if let Some(ref grid) = chain.default_target_grid {
        return Ok(grid.clone());
    }
    Err(OdxError::Argument(
        "transform has no default target grid (affine-only h5); pass --reference <NIfTI>".into(),
    ))
}
