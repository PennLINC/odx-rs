//! aPSF (apodised point-spread function) SH reorientation, mrtrix3-style.
//!
//! For an SH series in a *moving* frame and a Jacobian `J_chain` from the
//! transform chain (`fixed → moving`), produce the SH series describing the
//! same FOD in the *fixed* frame. Math (per voxel; constants computed once
//! over a fixed reference sphere `dirs_ref`):
//!
//! ```text
//!   dirs_mov   = J_chain · dirs_ref      (per-direction; magnitude kept)
//!   norms_i    = ‖dirs_mov_i‖
//!   det_J      = det(J_chain)
//!   mod_i      = norms_i / det_J          (if modulate; else 1)
//!   dirs_unit  = dirs_mov / norms
//!   amp_mov    = B(dirs_unit) · sh_in     (signed amplitudes)
//!   amp_target = mod ⊙ amp_mov
//!   sh_out     = pinv(B(dirs_ref)) · amp_target
//! ```
//!
//! `B(dirs)` is the basis-aware SH-to-amplitudes matrix, `(ndir × ncoeffs)`.
//! `pinv(B(dirs_ref))` is computed once via SVD; the per-voxel cost is one
//! `B(dirs_unit)` build (ndir basis rows) plus two matrix-vector multiplies.
//!
//! The reference sphere is built via a Fibonacci-spiral distribution over the
//! unit sphere — bias-free for any rotation, dense enough at `n_dirs ≈ 100`
//! for lmax ≤ 8 (mrtrix3 uses ~300 for lmax = 12).

use nalgebra::{DMatrix, Matrix3};
use ndarray::Array2;

use crate::descoteaux_sh;
use crate::error::{OdxError, Result};
use crate::header::Header;
use crate::mrtrix_sh;

/// Which SH basis a coefficient row belongs to (mirror of
/// [`crate::sh_basis_evaluator::ShBasisKind`] but the only ops we need here
/// are direction-set basis builds).
#[derive(Debug, Clone, Copy)]
pub enum ApsfBasis {
    Tournier { lmax: usize },
    Descoteaux { lmax: usize, full_basis: bool, legacy: bool },
}

impl ApsfBasis {
    pub fn from_header(header: &Header, ncoeffs: usize) -> Result<Self> {
        let raw = header
            .sh_basis
            .as_deref()
            .map(str::to_ascii_lowercase)
            .unwrap_or_else(|| "descoteaux07".into());
        match raw.as_str() {
            "tournier07" | "mrtrix" | "mrtrix3" => {
                let lmax = mrtrix_sh::lmax_for_ncoeffs(ncoeffs)?;
                Ok(Self::Tournier { lmax })
            }
            "descoteaux07" | "dipy" => {
                let full_basis = header.sh_full_basis.unwrap_or(false);
                let legacy = header.sh_legacy.unwrap_or(false);
                let lmax = descoteaux_sh::lmax_for_ncoeffs(ncoeffs, full_basis)?;
                Ok(Self::Descoteaux { lmax, full_basis, legacy })
            }
            other => Err(OdxError::Format(format!(
                "unsupported SH basis '{other}' for transform"
            ))),
        }
    }

    pub fn ncoeffs(&self) -> usize {
        match self {
            Self::Tournier { lmax } => mrtrix_sh::ncoeffs_for_lmax(*lmax),
            Self::Descoteaux { lmax, full_basis, .. } => {
                descoteaux_sh::ncoeffs_for(*lmax, *full_basis)
            }
        }
    }

    /// Build the SH-to-amplitudes basis matrix for the given directions.
    /// Shape: `(ndir, ncoeffs)`; row-major. *No* clamping — we want signed
    /// amplitudes for the aPSF round-trip.
    pub fn build_basis(&self, dirs_unit: &[[f32; 3]]) -> Array2<f32> {
        match self {
            Self::Tournier { lmax } => mrtrix_sh::sh2amp_cart(dirs_unit, *lmax),
            Self::Descoteaux { lmax, full_basis, legacy } => {
                descoteaux_sh::sh2amp_cart(dirs_unit, *lmax, *full_basis, *legacy)
            }
        }
    }
}

/// Per-dataset aPSF reorienter. Constructed once; reused across all voxels.
pub struct ShReorienter {
    basis: ApsfBasis,
    /// Reference sphere directions (in the *fixed* frame). Length = ndir.
    dirs_ref: Vec<[f32; 3]>,
    /// `pinv(B(dirs_ref))` flattened row-major as `(ncoeffs × ndir)`.
    pinv_b_orig: Array2<f32>,
    ncoeffs: usize,
    ndir: usize,
}

impl ShReorienter {
    pub fn new(basis: ApsfBasis, n_dirs: usize) -> Result<Self> {
        let ncoeffs = basis.ncoeffs();
        let dirs_ref = fibonacci_sphere(n_dirs);
        let b_orig = basis.build_basis(&dirs_ref);
        let pinv_b_orig = pseudo_inverse_f32(&b_orig)?;

        Ok(Self {
            basis,
            dirs_ref,
            pinv_b_orig,
            ncoeffs,
            ndir: n_dirs,
        })
    }

    #[allow(dead_code)]
    pub fn ncoeffs(&self) -> usize {
        self.ncoeffs
    }

    /// Reorient a moving-frame SH row to a fixed-frame SH row.
    ///
    /// `j_chain` is `∂chain/∂p_fix`, the fixed→moving Jacobian. Pass an
    /// arbitrary 3×3 matrix; non-orientation-preserving (negative-det)
    /// matrices are handled correctly via the modulation factor.
    pub fn reorient_into(
        &self,
        sh_in: &[f32],
        j_chain: &Matrix3<f64>,
        modulate: bool,
        sh_out: &mut [f32],
    ) -> Result<()> {
        if sh_in.len() != self.ncoeffs {
            return Err(OdxError::Format(format!(
                "SH input has {} coeffs, reorienter expects {}",
                sh_in.len(),
                self.ncoeffs
            )));
        }
        if sh_out.len() != self.ncoeffs {
            return Err(OdxError::Format(format!(
                "SH output has {} coeffs, reorienter expects {}",
                sh_out.len(),
                self.ncoeffs
            )));
        }

        let det_j = j_chain.determinant();
        // Singular Jacobian (e.g. fold-over voxels): pass through unchanged.
        if !det_j.is_finite() || det_j.abs() < 1e-12 {
            sh_out.copy_from_slice(sh_in);
            return Ok(());
        }

        // 1. Rotate reference sphere through J_chain; keep magnitudes for
        //    modulation. Build unit directions for the basis evaluator.
        let mut dirs_unit = Vec::with_capacity(self.ndir);
        let mut norms = Vec::with_capacity(self.ndir);
        for d in &self.dirs_ref {
            let v = j_chain
                * nalgebra::Vector3::new(d[0] as f64, d[1] as f64, d[2] as f64);
            let n = v.norm();
            if !(n.is_finite()) || n < 1e-12 {
                // Degenerate after rotation — punt to identity for this voxel.
                sh_out.copy_from_slice(sh_in);
                return Ok(());
            }
            norms.push(n);
            dirs_unit.push([
                (v[0] / n) as f32,
                (v[1] / n) as f32,
                (v[2] / n) as f32,
            ]);
        }

        // 2. Build B_rot (ndir × ncoeffs) for these moving directions.
        let b_rot = self.basis.build_basis(&dirs_unit);
        let b_rot_slice = b_rot.as_slice().expect("contiguous");

        // 3. Sample amplitudes at moving dirs: amp_mov = B_rot · sh_in.
        let mut amp = vec![0.0_f32; self.ndir];
        for d in 0..self.ndir {
            let row_start = d * self.ncoeffs;
            let mut acc = 0.0_f32;
            for c in 0..self.ncoeffs {
                acc += b_rot_slice[row_start + c] * sh_in[c];
            }
            amp[d] = acc;
        }

        // 4. Modulate by direction-dependent volume factor.
        if modulate {
            let inv_det = 1.0 / det_j;
            for d in 0..self.ndir {
                amp[d] = (amp[d] as f64 * norms[d] * inv_det) as f32;
            }
        }

        // 5. Fit back to SH at the fixed reference sphere.
        let pinv_slice = self.pinv_b_orig.as_slice().expect("contiguous");
        for c in 0..self.ncoeffs {
            let row_start = c * self.ndir;
            let mut acc = 0.0_f32;
            for d in 0..self.ndir {
                acc += pinv_slice[row_start + d] * amp[d];
            }
            sh_out[c] = acc;
        }

        Ok(())
    }
}

/// Build a Fibonacci-spiral set of unit vectors. Quasi-uniform, deterministic,
/// no axis bias.
pub fn fibonacci_sphere(n: usize) -> Vec<[f32; 3]> {
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    (0..n)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
            let r = (1.0 - y * y).max(0.0).sqrt();
            let theta = golden * i as f64;
            [
                (r * theta.cos()) as f32,
                y as f32,
                (r * theta.sin()) as f32,
            ]
        })
        .collect()
}

/// Pseudoinverse of an `(m × n)` row-major matrix, returned `(n × m)`.
/// Uses nalgebra SVD.
fn pseudo_inverse_f32(m: &Array2<f32>) -> Result<Array2<f32>> {
    let (rows, cols) = m.dim();
    let mut dm = DMatrix::<f64>::zeros(rows, cols);
    for r in 0..rows {
        for c in 0..cols {
            dm[(r, c)] = m[(r, c)] as f64;
        }
    }
    let svd = dm.svd(true, true);
    let pinv = svd
        .pseudo_inverse(1e-9)
        .map_err(|e| OdxError::Format(format!("SVD pinv failed: {e}")))?;
    let mut out = Array2::<f32>::zeros((cols, rows));
    for r in 0..cols {
        for c in 0..rows {
            out[(r, c)] = pinv[(r, c)] as f32;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fib_sphere_unit_norms() {
        for d in fibonacci_sphere(64) {
            let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            assert!((n - 1.0).abs() < 1e-5, "got {n}");
        }
    }

    #[test]
    fn identity_jacobian_preserves_sh() {
        let basis = ApsfBasis::Tournier { lmax: 4 };
        let r = ShReorienter::new(basis, 80).unwrap();
        let mut sh = vec![0.0_f32; r.ncoeffs];
        sh[0] = 0.282_094_8; // 1/(2*sqrt(pi)) — c_0,0
        sh[3] = 0.5;          // c_2,-1 ish
        sh[5] = -0.3;
        let mut out = vec![0.0_f32; r.ncoeffs];
        r.reorient_into(&sh, &Matrix3::identity(), false, &mut out).unwrap();
        for (i, (s, o)) in sh.iter().zip(out.iter()).enumerate() {
            assert!(
                (s - o).abs() < 1e-3,
                "coeff {i}: in={s}, out={o}"
            );
        }
    }

    #[test]
    fn det_zero_passes_through() {
        let basis = ApsfBasis::Tournier { lmax: 4 };
        let r = ShReorienter::new(basis, 60).unwrap();
        let sh = vec![0.5_f32; r.ncoeffs];
        let mut out = vec![0.0_f32; r.ncoeffs];
        // Singular Jacobian (rank 2): degenerate, should pass through.
        let mut j = Matrix3::<f64>::zeros();
        j[(0, 0)] = 1.0;
        j[(1, 1)] = 1.0;
        // (2,2) row stays zero
        r.reorient_into(&sh, &j, true, &mut out).unwrap();
        for (a, b) in sh.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
