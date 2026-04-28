//! Basis-aware SH→amplitudes evaluator.
//!
//! Earlier code routed every SH array through
//! [`crate::mrtrix_sh::RowSamplePlan`], which is hardcoded to MRtrix
//! `tournier07` even-only SH. Files using dipy `descoteaux07` SH (PAM,
//! pyAFQ aodf) were therefore evaluated with the wrong basis. This enum
//! is the single dispatch point: pick the per-basis evaluator from the
//! ODX header and downstream code stays the same.

use crate::descoteaux_sh;
use crate::error::{OdxError, Result};
use crate::header::Header;
use crate::mrtrix_sh;

/// Lightweight description of which SH basis a coefficient row lives in,
/// stripped of any precomputed sphere data. Useful when you need to call
/// the basis-specific derivative function but don't have an evaluator
/// pinned to a particular sphere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShBasisKind {
    /// MRtrix `tournier07`, even ℓ only.
    MrtrixTournier { lmax: usize },
    /// dipy `descoteaux07`. `full_basis = true` means odd ℓ are included
    /// (asymmetric ODFs).
    Descoteaux {
        lmax: usize,
        full_basis: bool,
        legacy: bool,
    },
}

impl ShBasisKind {
    /// Highest ℓ band represented by this basis.
    pub fn lmax(&self) -> usize {
        match self {
            Self::MrtrixTournier { lmax } => *lmax,
            Self::Descoteaux { lmax, .. } => *lmax,
        }
    }

    /// Coefficient-count this basis carries at its lmax.
    pub fn ncoeffs(&self) -> usize {
        match self {
            Self::MrtrixTournier { lmax } => mrtrix_sh::ncoeffs_for_lmax(*lmax),
            Self::Descoteaux {
                lmax, full_basis, ..
            } => descoteaux_sh::ncoeffs_for(*lmax, *full_basis),
        }
    }

    /// First and second partial derivatives of the SH series at
    /// `(elevation, azimuth)`, returning
    /// `(amplitude, ∂el, ∂az, ∂²el², ∂²el∂az, ∂²az²)`. Used by the
    /// Newton refinement in [`crate::peak_finder`].
    pub fn derivatives(
        &self,
        sh: &[f32],
        el: f64,
        az: f64,
    ) -> (f64, f64, f64, f64, f64, f64) {
        match self {
            Self::MrtrixTournier { lmax } => mrtrix_sh::sh_derivatives(sh, *lmax, el, az),
            Self::Descoteaux {
                lmax,
                full_basis,
                legacy,
            } => descoteaux_sh::sh_derivatives(sh, *lmax, *full_basis, *legacy, el, az),
        }
    }
}

/// Wraps a precomputed SH→amplitudes transform for a fixed sphere. The
/// variant identifies which SH basis the coefficients are in. Apply with
/// [`ShBasisEvaluator::apply_row_into`].
#[derive(Debug, Clone)]
pub enum ShBasisEvaluator {
    /// MRtrix `tournier07`, even ℓ only. Default for files written by
    /// MRtrix and DSI Studio. Always clamps amplitudes to non-negative.
    MrtrixTournier(mrtrix_sh::RowSamplePlan),
    /// dipy `descoteaux07`. Carries the `full_basis` flag (true → odd ℓ
    /// included, asymmetric ODFs) and `legacy` flag (true → dipy ≤ 1.x
    /// default convention with |m| in the m<0 branch).
    Descoteaux(descoteaux_sh::RowSamplePlan),
}

impl ShBasisEvaluator {
    /// Pick the right evaluator from the ODX header + the SH array's
    /// column count.
    ///
    /// - `SH_BASIS = "tournier07"` → [`Self::MrtrixTournier`].
    /// - `SH_BASIS = "descoteaux07"` (or absent, with a legitimate
    ///   descoteaux ncoeffs) → [`Self::Descoteaux`] using
    ///   `SH_FULL_BASIS` (default `false`) and `SH_LEGACY` (default
    ///   `false`, matching modern dipy ≥ 1.x recommendation).
    /// - Anything else → `Err`.
    pub fn from_header(
        header: &Header,
        dirs_ras: &[[f32; 3]],
        ncoeffs: usize,
    ) -> Result<Self> {
        let basis = header
            .sh_basis
            .as_deref()
            .map(str::to_ascii_lowercase)
            .unwrap_or_else(|| "descoteaux07".into());
        match basis.as_str() {
            "tournier07" | "mrtrix" | "mrtrix3" => {
                let plan = mrtrix_sh::RowSamplePlan::for_sh_rows_nonnegative(dirs_ras, ncoeffs)?;
                Ok(Self::MrtrixTournier(plan))
            }
            "descoteaux07" | "dipy" => {
                let full = header.sh_full_basis.unwrap_or(false);
                // Modern dipy emits descoteaux07 with `legacy=False` for
                // newly-derived data (pyAFQ aodf, MAPMRI). Default to that
                // unless the file explicitly opts into the legacy basis.
                let legacy = header.sh_legacy.unwrap_or(false);
                let plan =
                    descoteaux_sh::RowSamplePlan::for_sh_rows(dirs_ras, ncoeffs, full, legacy)?;
                Ok(Self::Descoteaux(plan))
            }
            other => Err(OdxError::Format(format!(
                "unsupported SH basis '{other}': expected 'tournier07' or 'descoteaux07'"
            ))),
        }
    }

    pub fn ncoeffs(&self) -> usize {
        match self {
            Self::MrtrixTournier(p) => p.ncoeffs(),
            Self::Descoteaux(p) => p.ncoeffs(),
        }
    }

    pub fn ndir(&self) -> usize {
        match self {
            Self::MrtrixTournier(p) => p.ndir(),
            Self::Descoteaux(p) => p.ndir(),
        }
    }

    /// Row-major `(ndir × ncoeffs)` transform, ready for GPU upload.
    pub fn transform_flat(&self) -> &[f32] {
        match self {
            Self::MrtrixTournier(p) => p.transform_flat(),
            Self::Descoteaux(p) => p.transform_flat(),
        }
    }

    pub fn apply_row_into(&self, src: &[f32], dst: &mut [f32]) {
        match self {
            Self::MrtrixTournier(p) => p.apply_row_into(src, dst),
            Self::Descoteaux(p) => p.apply_row_into(src, dst),
        }
    }

    pub fn apply_row(&self, src: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0_f32; self.ndir()];
        self.apply_row_into(src, &mut out);
        out
    }

    /// Whether amplitudes coming out of [`Self::apply_row_into`] are
    /// guaranteed non-negative (true for symmetric ODF reconstructions
    /// where negatives are fit noise; false for asymmetric ODFs where
    /// signed amplitudes are intentional).
    pub fn is_nonnegative(&self) -> bool {
        match self {
            Self::MrtrixTournier(_) => true,
            Self::Descoteaux(p) => p.clamp_nonnegative(),
        }
    }

    /// Basis identity for callers that don't need the precomputed
    /// transform (e.g. derivative-only paths).
    pub fn kind(&self) -> Result<ShBasisKind> {
        match self {
            Self::MrtrixTournier(p) => Ok(ShBasisKind::MrtrixTournier {
                lmax: mrtrix_sh::lmax_for_ncoeffs(p.ncoeffs())?,
            }),
            Self::Descoteaux(p) => Ok(ShBasisKind::Descoteaux {
                lmax: descoteaux_sh::lmax_for_ncoeffs(p.ncoeffs(), p.full_basis())?,
                full_basis: p.full_basis(),
                legacy: p.legacy(),
            }),
        }
    }

    /// Convenience: forward to [`ShBasisKind::derivatives`].
    pub fn derivatives(
        &self,
        sh: &[f32],
        el: f64,
        az: f64,
    ) -> Result<(f64, f64, f64, f64, f64, f64)> {
        Ok(self.kind()?.derivatives(sh, el, az))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::Header;
    use std::collections::HashMap;

    fn header_with_basis(basis: &str, full_basis: Option<bool>, legacy: Option<bool>) -> Header {
        Header {
            voxel_to_rasmm: Header::identity_affine(),
            dimensions: [1, 1, 1],
            nb_voxels: 1,
            nb_peaks: 0,
            nb_sphere_vertices: None,
            nb_sphere_faces: None,
            sh_order: None,
            sh_basis: Some(basis.into()),
            sh_full_basis: full_basis,
            sh_legacy: legacy,
            canonical_dense_representation: None,
            sphere_id: None,
            odf_sample_domain: None,
            array_quantization: HashMap::new(),
            extra: HashMap::new(),
        }
    }

    #[test]
    fn dispatch_picks_descoteaux_full() {
        let dirs = vec![[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let h = header_with_basis("descoteaux07", Some(true), Some(false));
        let ev = ShBasisEvaluator::from_header(&h, &dirs, 25).unwrap();
        assert!(matches!(ev, ShBasisEvaluator::Descoteaux(_)));
        assert_eq!(ev.ncoeffs(), 25);
        assert_eq!(ev.ndir(), 2);
        // Default clamps to nonneg even for full basis; see the long
        // comment on `descoteaux_sh::RowSamplePlan::for_sh_rows` for why
        // (signed amps break glyph rendering and are unused by tracking).
        assert!(ev.is_nonnegative());
    }

    #[test]
    fn dispatch_picks_tournier() {
        let dirs = vec![[1.0_f32, 0.0, 0.0]];
        let h = header_with_basis("tournier07", None, None);
        let ev = ShBasisEvaluator::from_header(&h, &dirs, 45).unwrap();
        assert!(matches!(ev, ShBasisEvaluator::MrtrixTournier(_)));
        assert!(ev.is_nonnegative());
    }

    #[test]
    fn dispatch_descoteaux_default_legacy_false() {
        let dirs = vec![[1.0_f32, 0.0, 0.0]];
        let h = header_with_basis("descoteaux07", Some(false), None);
        let ev = ShBasisEvaluator::from_header(&h, &dirs, 15).unwrap();
        if let ShBasisEvaluator::Descoteaux(plan) = ev {
            assert!(!plan.legacy());
            assert!(!plan.full_basis());
        } else {
            panic!("expected descoteaux variant");
        }
    }

    #[test]
    fn unknown_basis_errors() {
        let dirs = vec![[1.0_f32, 0.0, 0.0]];
        let h = header_with_basis("not_a_real_basis", None, None);
        assert!(ShBasisEvaluator::from_header(&h, &dirs, 6).is_err());
    }
}
