//! Real spherical harmonics in dipy's `descoteaux07` convention.
//!
//! Mirrors the public API of [`crate::mrtrix_sh`] (which handles MRtrix
//! `tournier07` even-only SH) so that downstream code can dispatch through
//! [`crate::sh_basis_evaluator::ShBasisEvaluator`].
//!
//! Conventions follow dipy ≥ 1.7's `real_sh_descoteaux_from_index` (see
//! `dipy/dipy/reconst/shm.py:396`):
//!
//! ```text
//!   m  > 0:  Y_l^m =  √2 · N_l^m · P_l^m(cosθ) · sin(m·φ)
//!   m == 0:  Y_l^0 =       N_l^0 · P_l^0(cosθ)
//!   m  < 0:  Y_l^m =  √2 · s_lm · N_l^|m| · P_l^|m|(cosθ) · cos(|m|·φ)
//! ```
//!
//! where `N_l^m = √((2l+1)/(4π) · (l−|m|)!/(l+|m|)!)` and `P_l^m` is the
//! associated Legendre function (no Condon–Shortley phase). The sign
//! `s_lm` is `1` in the legacy basis (dipy `legacy=True`, the default
//! through dipy 1.x) and `(-1)^|m|` in the modern basis (`legacy=False`),
//! which is what pyAFQ's `unified_filtering` writes (see
//! `pyAFQ/AFQ/models/asym_filtering.py:50`).
//!
//! Coefficients are ordered by ℓ ascending; within each ℓ band, m runs from
//! `-ℓ` through `+ℓ`. With `full_basis = false`, only even ℓ are emitted
//! (ncoeffs = (lmax+1)(lmax+2)/2). With `full_basis = true`, all ℓ
//! 0..=lmax are emitted (ncoeffs = (lmax+1)²) — this is what asymmetric
//! ODFs like `aodf` use.

use ndarray::Array2;

use crate::error::{OdxError, Result};

/// Number of coefficients for a given lmax in either basis variant.
pub fn ncoeffs_for(lmax: usize, full_basis: bool) -> usize {
    if full_basis {
        (lmax + 1) * (lmax + 1)
    } else {
        (lmax + 1) * (lmax + 2) / 2
    }
}

/// Inverse of [`ncoeffs_for`]. Errors if `n` is not a valid descoteaux SH
/// cardinality for the requested basis variant.
pub fn lmax_for_ncoeffs(n: usize, full_basis: bool) -> Result<usize> {
    if full_basis {
        // (lmax + 1)^2 = n
        if n == 0 {
            return Ok(0);
        }
        let lmax_f = (n as f64).sqrt() - 1.0;
        let lmax = lmax_f.round() as usize;
        if ncoeffs_for(lmax, true) == n {
            Ok(lmax)
        } else {
            Err(OdxError::Format(format!(
                "coefficient count {n} is not a feasible full-basis descoteaux07 SH cardinality"
            )))
        }
    } else {
        if n == 0 {
            return Ok(0);
        }
        let lmax = 2 * ((((1 + 8 * n) as f64).sqrt() - 3.0) / 4.0).floor() as usize;
        if ncoeffs_for(lmax, false) == n {
            Ok(lmax)
        } else {
            Err(OdxError::Format(format!(
                "coefficient count {n} is not a feasible descoteaux07 SH cardinality"
            )))
        }
    }
}

/// Default `norm_factor` for [`anisotropic_power_full_basis`]. Matches the
/// MRtrix-symmetric default at [`crate::mrtrix_sh::ANISOTROPIC_POWER_NORM_FACTOR`].
pub const ANISOTROPIC_POWER_NORM_FACTOR: f64 = 1e-5;

/// Anisotropic-power scalar for full-basis descoteaux07 SH.
///
/// Generalizes [`crate::mrtrix_sh::anisotropic_power`] by summing the
/// per-ℓ band-mean of squared coefficients over **all ℓ ≥ 1** rather
/// than even-ℓ only — for symmetric inputs (odd-ℓ coefs all zero) this
/// reproduces the MRtrix value exactly. Useful as a slice-display scalar
/// for asymmetric ODFs because both symmetric and asymmetric anisotropy
/// contribute, so it cleanly separates white matter from grey/CSF the
/// same way standard AP does for CSD ODFs.
///
/// Order of `sh` must match dipy's `sph_harm_ind_list(lmax,
/// full_basis=True)`: ℓ ascending; m = −ℓ..+ℓ within each band.
pub fn anisotropic_power_full_basis(sh: &[f32], lmax: usize, norm_factor: f64) -> f32 {
    debug_assert_eq!(sh.len(), ncoeffs_for(lmax, true));
    if lmax == 0 {
        return 0.0;
    }
    let mut start = 1_usize; // skip ℓ=0 (carries the isotropic baseline)
    let mut ap_raw = 0.0_f64;
    for ell in 1..=lmax {
        let len = 2 * ell + 1;
        let stop = start + len;
        let mut s = 0.0_f64;
        for c in &sh[start..stop] {
            s += (*c as f64) * (*c as f64);
        }
        ap_raw += s / len as f64;
        start = stop;
    }
    if ap_raw <= 0.0 {
        return 0.0;
    }
    let v = ap_raw.ln() - norm_factor.ln();
    if v <= 0.0 { 0.0 } else { v as f32 }
}

/// Asymmetry index (ASI) of a full-basis descoteaux07 SH series.
///
/// Defined in Cetin Karayumak, Özarslan & Unal (2018,
/// <https://doi.org/10.1016/j.mri.2018.03.006>) and implemented by
/// pyAFQ as `compute_asymmetry_index` (asym_filtering.py:552):
///
/// ```text
///     r = (Σ_lm c_lm² · (-1)^ℓ) / Σ_lm c_lm²
///     ASI = √(1 − clip(r, 0, 1)²)
/// ```
///
/// Returns 0 for symmetric inputs (odd ℓ all zero ⇒ r = 1) and
/// approaches 1 as asymmetric energy dominates. The clip handles the
/// pathological case where `r < 0` (asymmetric > symmetric, which
/// shouldn't happen for valid amplitudes but can show up in noise) by
/// capping ASI at 1.
pub fn asymmetry_index(sh: &[f32], lmax: usize) -> f32 {
    debug_assert_eq!(sh.len(), ncoeffs_for(lmax, true));
    let mut idx = 0_usize;
    let mut total = 0.0_f64;
    let mut signed = 0.0_f64;
    for ell in 0..=lmax {
        let len = 2 * ell + 1;
        let sign = if ell % 2 == 0 { 1.0_f64 } else { -1.0 };
        for _ in 0..len {
            let v = sh[idx] as f64;
            let v2 = v * v;
            total += v2;
            signed += v2 * sign;
            idx += 1;
        }
    }
    if total <= 0.0 {
        return 0.0;
    }
    let r = (signed / total).clamp(0.0, 1.0);
    (1.0 - r * r).sqrt() as f32
}

/// Convert a unit RAS direction to spherical `(θ, φ)` with `θ` measured from
/// +z (colatitude, 0..π) and `φ` measured from +x toward +y (azimuth,
/// −π..π). Matches the convention dipy's `cart2sphere` uses.
pub(crate) fn cart_to_sphere(d: [f32; 3]) -> (f64, f64) {
    let x = d[0] as f64;
    let y = d[1] as f64;
    let z = d[2] as f64;
    let r = (x * x + y * y + z * z).sqrt().max(1e-30);
    let theta = (z / r).clamp(-1.0, 1.0).acos();
    let phi = y.atan2(x);
    (theta, phi)
}

/// Fully-normalized associated Legendre values without Condon–Shortley
/// phase, computed by stable recurrence:
///
/// `P̄_l^m(x) = √((2l+1)/(4π) · (l−m)!/(l+m)!) · P_l^m(x)` for m ≥ 0.
///
/// Output is a `(lmax+1) x (lmax+1)` row-major array indexed `[l*(lmax+1)+m]`,
/// with entries above the band (m > l) left zero.
fn normalized_alf(lmax: usize, cos_theta: f64) -> Vec<f64> {
    let stride = lmax + 1;
    let mut p = vec![0.0_f64; stride * stride];
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let four_pi = 4.0 * std::f64::consts::PI;

    // P̄_0^0 = 1/√(4π)
    p[0] = (1.0 / four_pi).sqrt();

    if lmax == 0 {
        return p;
    }

    // Sectoral recurrence: P̄_m^m = √((2m+1)/(2m)) · sinθ · P̄_{m-1}^{m-1}
    // (no CS phase). Matches NIST DLMF 14.30 for fully normalised ALFs.
    for m in 1..=lmax {
        let prev = p[(m - 1) * stride + (m - 1)];
        let factor = ((2 * m + 1) as f64 / (2 * m) as f64).sqrt();
        p[m * stride + m] = factor * sin_theta * prev;
    }

    // Off-band: P̄_l^m for l > m using the standard normalised recurrence:
    //   P̄_{m+1}^m = √(2m+3) · cosθ · P̄_m^m
    //   P̄_l^m = a_lm · cosθ · P̄_{l-1}^m  −  b_lm · P̄_{l-2}^m
    // where
    //   a_lm = √((2l+1)(2l−1) / ((l−m)(l+m)))
    //   b_lm = √((2l+1)(l+m−1)(l−m−1) / ((l−m)(l+m)(2l−3)))
    for m in 0..=lmax {
        if m + 1 > lmax {
            continue;
        }
        let pmm = p[m * stride + m];
        // Seed: P̄_{m+1}^m
        let l = m + 1;
        p[l * stride + m] = (2.0 * l as f64 + 1.0).sqrt() * cos_theta * pmm;
        // Climb to P̄_l^m for l = m+2..=lmax
        for l in (m + 2)..=lmax {
            let lf = l as f64;
            let mf = m as f64;
            let a_lm = ((2.0 * lf + 1.0) * (2.0 * lf - 1.0) / ((lf - mf) * (lf + mf))).sqrt();
            let b_lm = ((2.0 * lf + 1.0)
                * (lf + mf - 1.0)
                * (lf - mf - 1.0)
                / ((lf - mf) * (lf + mf) * (2.0 * lf - 3.0)))
                .sqrt();
            p[l * stride + m] = a_lm * cos_theta * p[(l - 1) * stride + m]
                - b_lm * p[(l - 2) * stride + m];
        }
    }

    p
}

/// Evaluate the descoteaux07 real SH basis at a single `(θ, φ)`.
///
/// Output length is [`ncoeffs_for`]`(lmax, full_basis)`. Coefficient order:
/// ℓ ascending (skipping odd ℓ when `!full_basis`), m running `-ℓ..=+ℓ`
/// within each band.
pub fn sh_row(theta: f64, phi: f64, lmax: usize, full_basis: bool, legacy: bool) -> Vec<f32> {
    let cos_theta = theta.cos();
    let p = normalized_alf(lmax, cos_theta);
    let stride = lmax + 1;
    let sqrt2 = std::f64::consts::SQRT_2;

    let mut out = Vec::with_capacity(ncoeffs_for(lmax, full_basis));
    let l_step = if full_basis { 1 } else { 2 };
    let mut l = 0usize;
    while l <= lmax {
        for m in -(l as i32)..=(l as i32) {
            let abs_m = m.unsigned_abs() as usize;
            let plm = p[l * stride + abs_m];
            let cs = if abs_m % 2 == 1 { -1.0_f64 } else { 1.0 };
            let val = if m > 0 {
                // scipy `sph_harm_y` for positive m carries a (−1)^m
                // Condon–Shortley phase; descoteaux's √2·Im(Y_l^m) inherits it.
                sqrt2 * cs * plm * (m as f64 * phi).sin()
            } else if m == 0 {
                plm
            } else {
                // For m < 0 in non-legacy mode the CS phase factors cancel
                // because `sph_harm_y` with negative m is itself defined
                // via `(−1)^|m| · conj(sph_harm_y(|m|, l, ...))`. In
                // legacy mode dipy feeds |m| to scipy directly, so the
                // CS phase reappears.
                let sign = if legacy { cs } else { 1.0 };
                sqrt2 * sign * plm * (abs_m as f64 * phi).cos()
            };
            out.push(val as f32);
        }
        l += l_step;
    }
    out
}

/// Build the SH-to-amplitudes transform matrix: shape `(ndir, ncoeffs)`,
/// row-major, applied as `amp = SH · coeffs` per direction.
pub fn sh2amp_cart(
    dirs_ras: &[[f32; 3]],
    lmax: usize,
    full_basis: bool,
    legacy: bool,
) -> Array2<f32> {
    let ncoeffs = ncoeffs_for(lmax, full_basis);
    let mut data = Vec::with_capacity(dirs_ras.len() * ncoeffs);
    for &dir in dirs_ras {
        let (theta, phi) = cart_to_sphere(dir);
        data.extend(sh_row(theta, phi, lmax, full_basis, legacy));
    }
    Array2::from_shape_vec((dirs_ras.len(), ncoeffs), data)
        .expect("sh2amp_cart shape consistent with input")
}

/// Mirrors [`crate::mrtrix_sh::RowSamplePlan`] for descoteaux07 SH.
///
/// Stores the precomputed SH→amplitudes transform for a fixed sphere and
/// applies it row-by-row. `clamp_nonnegative` defaults to `false` for the
/// full basis (asymmetric amplitudes are intentionally signed) and `true`
/// for the symmetric basis (matches MRtrix `sh2amp -nonnegative`).
#[derive(Debug, Clone)]
pub struct RowSamplePlan {
    transform: Array2<f32>,
    ncoeffs: usize,
    ndir: usize,
    full_basis: bool,
    legacy: bool,
    clamp_nonnegative: bool,
}

impl RowSamplePlan {
    pub fn for_sh_rows(
        dirs_ras: &[[f32; 3]],
        ncoeffs: usize,
        full_basis: bool,
        legacy: bool,
    ) -> Result<Self> {
        let lmax = lmax_for_ncoeffs(ncoeffs, full_basis)?;
        let transform = sh2amp_cart(dirs_ras, lmax, full_basis, legacy);
        let (ndir, cols) = transform.dim();
        Ok(Self {
            transform,
            ncoeffs: cols,
            ndir,
            full_basis,
            legacy,
            // Default to clamping for both symmetric and full-basis SH.
            //
            // - Symmetric: matches `mrtrix_sh::RowSamplePlan` (sh2amp -nonnegative).
            // - Full basis: counter-intuitively, we still want clamp here.
            //   The full-basis SH carries asymmetry as the *signed* amplitude
            //   contrast between u and -u (high positive at the forward lobe,
            //   low or negative at the antipode). The asymmetry is preserved
            //   under clamping — the forward lobe stays, the antipode just
            //   becomes 0 amplitude rather than negative — which is exactly
            //   what every consumer wants:
            //   * glyph rendering: the shader uses `amp` as a radial
            //     multiplier (`world = center + dir * amp`); negative amp
            //     pushes vertices through the origin and breaks the surface.
            //   * peak detection: `find_peaks` already rejects v <= 0.
            //   * probabilistic tracking PMF: the GPU shader clamps
            //     negatives anyway (see `dg_prob.rs::sample_direction`).
            //
            // If you specifically need the raw signed surface (e.g. to
            // compute an asymmetry index in dipy), call
            // `set_clamp_nonnegative(false)` on the returned plan.
            clamp_nonnegative: true,
        })
    }

    pub fn for_sh_rows_nonnegative(
        dirs_ras: &[[f32; 3]],
        ncoeffs: usize,
        full_basis: bool,
        legacy: bool,
    ) -> Result<Self> {
        let mut plan = Self::for_sh_rows(dirs_ras, ncoeffs, full_basis, legacy)?;
        plan.clamp_nonnegative = true;
        Ok(plan)
    }

    pub fn ncoeffs(&self) -> usize {
        self.ncoeffs
    }

    pub fn ndir(&self) -> usize {
        self.ndir
    }

    pub fn full_basis(&self) -> bool {
        self.full_basis
    }

    pub fn legacy(&self) -> bool {
        self.legacy
    }

    pub fn clamp_nonnegative(&self) -> bool {
        self.clamp_nonnegative
    }

    pub fn set_clamp_nonnegative(&mut self, clamp: bool) {
        self.clamp_nonnegative = clamp;
    }

    pub fn transform_flat(&self) -> &[f32] {
        self.transform
            .as_slice()
            .expect("descoteaux SH transform should be contiguous")
    }

    pub fn apply_row_into(&self, src: &[f32], dst: &mut [f32]) {
        assert_eq!(src.len(), self.ncoeffs);
        assert_eq!(dst.len(), self.ndir);
        for d in 0..self.ndir {
            let row_start = d * self.ncoeffs;
            let mut acc = 0.0_f32;
            for c in 0..self.ncoeffs {
                acc += self.transform.as_slice().unwrap()[row_start + c] * src[c];
            }
            dst[d] = if self.clamp_nonnegative {
                acc.max(0.0)
            } else {
                acc
            };
        }
    }

    pub fn apply_row(&self, src: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.ndir];
        self.apply_row_into(src, &mut out);
        out
    }
}

/// First and second partial derivatives of a descoteaux07 SH series at
/// `(theta, azimuth)`, returning
/// `(amplitude, ∂/∂θ, ∂/∂φ, ∂²/∂θ², ∂²/∂θ∂φ, ∂²/∂φ²)`.
///
/// Matches the signature of [`crate::mrtrix_sh::sh_derivatives`] so the
/// Newton refinement in [`crate::peak_finder`] can dispatch through
/// [`crate::sh_basis_evaluator::ShBasisEvaluator`].
///
/// `theta` here is the *polar angle* (colatitude, 0 at +z, π at −z) —
/// same convention `mrtrix_sh::sh_derivatives` uses (and what the peak
/// finder hands us). The argument is named `el` only to keep parity with
/// the upstream MRtrix port; do not interpret it as latitude.
pub fn sh_derivatives(
    sh: &[f32],
    lmax: usize,
    full_basis: bool,
    legacy: bool,
    el: f64,
    az: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    debug_assert_eq!(sh.len(), ncoeffs_for(lmax, full_basis));
    // Finite differences on a tight stencil: cheap to implement and
    // accurate enough for Newton refinement (the seed is already within
    // one sphere edge of the truth, and a 5-point stencil at h=1e-4 rad
    // gives ≈1e-7 first-derivative / ≈1e-5 second-derivative error). The
    // 9 stencil points only span 3 distinct polar angles (el−h, el, el+h)
    // so we precompute one ALF table per polar angle rather than per point
    // — saves 6× redundant Legendre work.
    let h = 1.0e-4_f64;
    let alf_pe = normalized_alf(lmax, (el + h).cos());
    let alf_e = normalized_alf(lmax, el.cos());
    let alf_me = normalized_alf(lmax, (el - h).cos());

    let eval = |alf: &[f64], az: f64| -> f64 {
        sh_dot_with_alf(sh, alf, az, lmax, full_basis, legacy)
    };

    let f = eval(&alf_e, az);
    let f_pe = eval(&alf_pe, az);
    let f_me = eval(&alf_me, az);
    let f_pa = eval(&alf_e, az + h);
    let f_ma = eval(&alf_e, az - h);
    let f_pe_pa = eval(&alf_pe, az + h);
    let f_pe_ma = eval(&alf_pe, az - h);
    let f_me_pa = eval(&alf_me, az + h);
    let f_me_ma = eval(&alf_me, az - h);

    let d_el = (f_pe - f_me) / (2.0 * h);
    let d_az = (f_pa - f_ma) / (2.0 * h);
    let d2_el2 = (f_pe - 2.0 * f + f_me) / (h * h);
    let d2_az2 = (f_pa - 2.0 * f + f_ma) / (h * h);
    let d2_el_az = (f_pe_pa - f_pe_ma - f_me_pa + f_me_ma) / (4.0 * h * h);
    (f, d_el, d_az, d2_el2, d2_el_az, d2_az2)
}

fn sh_dot_with_alf(
    sh: &[f32],
    alf: &[f64],
    phi: f64,
    lmax: usize,
    full_basis: bool,
    legacy: bool,
) -> f64 {
    let stride = lmax + 1;
    let sqrt2 = std::f64::consts::SQRT_2;
    let mut acc = 0.0_f64;
    let mut idx = 0_usize;
    let l_step = if full_basis { 1 } else { 2 };
    let mut l = 0_usize;
    while l <= lmax {
        for m in -(l as i32)..=(l as i32) {
            let abs_m = m.unsigned_abs() as usize;
            let plm = alf[l * stride + abs_m];
            let cs = if abs_m % 2 == 1 { -1.0_f64 } else { 1.0 };
            let val = if m > 0 {
                sqrt2 * cs * plm * (m as f64 * phi).sin()
            } else if m == 0 {
                plm
            } else {
                let sign = if legacy { cs } else { 1.0 };
                sqrt2 * sign * plm * (abs_m as f64 * phi).cos()
            };
            acc += sh[idx] as f64 * val;
            idx += 1;
        }
        l += l_step;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ncoeffs_round_trip() {
        for lmax in [0, 2, 4, 6, 8, 10] {
            let n = ncoeffs_for(lmax, false);
            assert_eq!(lmax_for_ncoeffs(n, false).unwrap(), lmax);
        }
        for lmax in 0..=10 {
            let n = ncoeffs_for(lmax, true);
            assert_eq!(lmax_for_ncoeffs(n, true).unwrap(), lmax);
        }
    }

    #[test]
    fn ncoeffs_full_vs_symmetric() {
        // Sanity: aodf at lmax=8 has 81 full coefs vs 45 symmetric.
        assert_eq!(ncoeffs_for(8, true), 81);
        assert_eq!(ncoeffs_for(8, false), 45);
    }

    #[test]
    fn unit_amplitude_for_constant_sh() {
        // SH coef vector with only the (l=0, m=0) coefficient set to 1/Y00
        // should produce amplitude 1 in every direction (Y_0^0 = 1/√(4π)).
        let dirs = vec![
            [1.0_f32, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-0.6, 0.8, 0.0],
        ];
        let inv_y00 = (4.0 * std::f64::consts::PI).sqrt() as f32;
        let mut coefs = vec![0.0_f32; ncoeffs_for(2, true)];
        coefs[0] = inv_y00;
        let plan = RowSamplePlan::for_sh_rows(&dirs, coefs.len(), true, false).unwrap();
        for amp in plan.apply_row(&coefs) {
            assert!((amp - 1.0).abs() < 1e-5, "expected 1.0, got {amp}");
        }
    }

    #[test]
    fn antisymmetry_of_odd_l_full_basis() {
        // In the full basis, odd-ℓ contributions flip sign at antipodes.
        // Set a single ℓ=1 coefficient and verify f(d) = −f(−d) on the
        // *raw* signed surface (the default clamp must be turned off).
        let dirs: Vec<[f32; 3]> = vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5_f32.sqrt(), 0.5_f32.sqrt(), 0.0],
        ];
        let mut antipodes = Vec::new();
        for d in &dirs {
            antipodes.push([-d[0], -d[1], -d[2]]);
        }
        // lmax = 1, full basis → 4 coefs: (l=0,m=0), (l=1,m=-1), (l=1,m=0), (l=1,m=1)
        let mut coefs = vec![0.0_f32; ncoeffs_for(1, true)];
        coefs[2] = 1.0; // l=1, m=0 → cosθ-shaped lobe
        let mut p_dir = RowSamplePlan::for_sh_rows(&dirs, coefs.len(), true, false).unwrap();
        let mut p_anti =
            RowSamplePlan::for_sh_rows(&antipodes, coefs.len(), true, false).unwrap();
        // Disable the default clamp so we can observe the signed amplitudes.
        p_dir.set_clamp_nonnegative(false);
        p_anti.set_clamp_nonnegative(false);
        let amp = p_dir.apply_row(&coefs);
        let amp_anti = p_anti.apply_row(&coefs);
        for (a, b) in amp.iter().zip(amp_anti.iter()) {
            assert!((a + b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    /// Eight directions used to generate the dipy golden values below.
    /// Each row is normalised in the test so the constants stay readable.
    fn golden_dirs() -> Vec<[f32; 3]> {
        let mut dirs = vec![
            [1.0_f32, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.5, 0.5, std::f32::consts::FRAC_1_SQRT_2],
            [0.6, -0.8, 0.0],
            [0.0, 0.6, 0.8],
            [-0.3, 0.4, (1.0_f32 - 0.25).sqrt()],
        ];
        for d in &mut dirs {
            let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            d[0] /= n;
            d[1] /= n;
            d[2] /= n;
        }
        dirs
    }

    /// Reference values from dipy 1.12, lmax=4, full_basis=True, legacy=False.
    /// Captured by running `real_sh_descoteaux(4, theta, phi,
    /// full_basis=True, legacy=False)` on `golden_dirs()`. Row-major,
    /// shape (8, 25). Order: ℓ ascending, m -ℓ..+ℓ within band.
    const DIPY_FULL_LMAX4_LEGACY_FALSE: &[f32] = &[
        // dir 0: +x
        0.2820947918, 0.4886025119, 0.0, 0.0,
        0.5462742153, 0.0, -0.3153915653, 0.0, 0.0,
        0.5900435899, 0.0, -0.4570457995, 0.0, 0.0, 0.0, 0.0,
        0.6258357354, 0.0, -0.4730873479, 0.0, 0.3173566407, 0.0, 0.0, 0.0, 0.0,
        // dir 1: +y
        0.2820947918, 0.0, 0.0, -0.4886025119,
        -0.5462742153, 0.0, -0.3153915653, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.4570457995, 0.0, 0.5900435899,
        0.6258357354, 0.0, 0.4730873479, 0.0, 0.3173566407, 0.0, 0.0, 0.0, 0.0,
        // dir 2: +z
        0.2820947918, 0.0, 0.4886025119, 0.0,
        0.0, 0.0, 0.6307831305, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.7463526652, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.8462843753, 0.0, 0.0, 0.0, 0.0,
        // dir 3: -x
        0.2820947918, -0.4886025119, 0.0, 0.0,
        0.5462742153, 0.0, -0.3153915653, 0.0, 0.0,
        -0.5900435899, 0.0, 0.4570457995, 0.0, 0.0, 0.0, 0.0,
        0.6258357354, 0.0, -0.4730873479, 0.0, 0.3173566407, 0.0, 0.0, 0.0, 0.0,
        // dir 4: (0.5, 0.5, √2/2)
        0.2820947918, 0.2443012560, 0.3454941495, -0.2443012560,
        0.0, 0.3862742020, 0.1576957826, -0.3862742020, 0.2731371076,
        -0.1475108975, 0.0, 0.3427843496, -0.1319377577, -0.3427843496,
        0.5109927382, -0.1475108975,
        -0.1564589339, -0.3129178677, 0.0, 0.1182718370, -0.3438030275,
        -0.1182718370, 0.5913591848, -0.3129178677, 0.0,
        // dir 5: (0.6, -0.8, 0)
        0.2820947918, 0.2931615071, 0.0, 0.3908820095,
        -0.1529567803, 0.0, -0.3153915653, 0.0, -0.5244232467,
        -0.5522808002, 0.0, -0.2742274797, 0.0, -0.3656366396, 0.0, 0.2076953437,
        -0.5277046921, 0.0, 0.1324644574, 0.0, 0.3173566407, 0.0, 0.4541638540, 0.0, 0.3364492914,
        // dir 6: (0, 0.6, 0.8)
        0.2820947918, 0.0, 0.3908820095, -0.2931615071,
        -0.1966587175, 0.0, 0.2901602400, -0.5244232467, 0.0,
        0.0, -0.4162480477, 0.0, 0.0597082132, -0.6033004553, 0.0, 0.1274494154,
        0.0811083113, 0.0, -0.5926838294, 0.0, -0.1971842594, -0.4752906645, 0.0, 0.3058785970, 0.0,
        // dir 7: (-0.3, 0.4, √(0.75))
        0.2820947918, -0.1465807536, 0.4231421877, -0.1954410048,
        -0.0382391951, -0.2838524087, 0.3942394566, -0.3784698783, -0.1311058117,
        0.0690351000, -0.0876170030, -0.3770627846, 0.2423851381, -0.5027503794, -0.3004011530, -0.0259619180,
        -0.0329815433, 0.1793584511, -0.1407434860, -0.3911026295, 0.0198347900, -0.5214701727, -0.4825490948, -0.0674510414, 0.0210280807,
    ];

    /// dipy 1.12, lmax=4, full_basis=True, legacy=True. Three directions:
    /// [+x, (0.5, 0.5, √2/2), (-0.3, 0.4, √0.75)]. Row-major shape (3, 25).
    const DIPY_FULL_LMAX4_LEGACY_TRUE: &[f32] = &[
        0.28209479, -0.48860251, 0.0, 0.0,
        0.54627422, 0.0, -0.31539157, 0.0, 0.0,
        -0.59004359, 0.0, 0.45704580, 0.0, 0.0, 0.0, 0.0,
        0.62583574, 0.0, -0.47308735, 0.0, 0.31735664, 0.0, 0.0, 0.0, 0.0,
        0.28209479, -0.24430126, 0.34549415, -0.24430126,
        0.0, -0.38627420, 0.15769578, -0.38627420, 0.27313711,
        0.14751090, 0.0, -0.34278435, -0.13193776, -0.34278435, 0.51099274, -0.14751090,
        -0.15645893, 0.31291787, 0.0, -0.11827184, -0.34380303, -0.11827184, 0.59135918, -0.31291787, 0.0,
        0.28209479, 0.14658075, 0.42314219, -0.19544100,
        -0.03823920, 0.28385241, 0.39423946, -0.37846988, -0.13110581,
        -0.06903510, -0.08761700, 0.37706278, 0.24238514, -0.50275038, -0.30040115, -0.02596192,
        -0.03298154, -0.17935845, -0.14074349, 0.39110263, 0.01983479, -0.52147017, -0.48254909, -0.06745104, 0.02102808,
    ];

    /// dipy 1.12, lmax=4, full_basis=False (even ℓ only), legacy=False.
    /// Same three directions as above; row-major (3, 15).
    const DIPY_SYM_LMAX4_LEGACY_FALSE: &[f32] = &[
        0.28209479, 0.54627422, 0.0, -0.31539157, 0.0, 0.0,
        0.62583574, 0.0, -0.47308735, 0.0, 0.31735664, 0.0, 0.0, 0.0, 0.0,
        0.28209479, 0.0, 0.38627420, 0.15769578, -0.38627420, 0.27313711,
        -0.15645893, -0.31291787, 0.0, 0.11827184, -0.34380303, -0.11827184, 0.59135918, -0.31291787, 0.0,
        0.28209479, -0.03823920, -0.28385241, 0.39423946, -0.37846988, -0.13110581,
        -0.03298154, 0.17935845, -0.14074349, -0.39110263, 0.01983479, -0.52147017, -0.48254909, -0.06745104, 0.02102808,
    ];

    fn three_dirs() -> Vec<[f32; 3]> {
        let mut dirs = vec![
            [1.0_f32, 0.0, 0.0],
            [0.5, 0.5, std::f32::consts::FRAC_1_SQRT_2],
            [-0.3, 0.4, (1.0_f32 - 0.25).sqrt()],
        ];
        for d in &mut dirs {
            let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            d[0] /= n;
            d[1] /= n;
            d[2] /= n;
        }
        dirs
    }

    fn assert_close(got: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(got.len(), expected.len());
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < tol,
                "mismatch at {i}: got {g}, expected {e} (Δ={:.2e})",
                (g - e).abs()
            );
        }
    }

    #[test]
    fn dipy_golden_full_lmax4_legacy_true() {
        let dirs = three_dirs();
        let plan = RowSamplePlan::for_sh_rows(&dirs, 25, true, true).unwrap();
        assert_close(plan.transform_flat(), DIPY_FULL_LMAX4_LEGACY_TRUE, 5e-6);
    }

    #[test]
    fn dipy_golden_sym_lmax4_legacy_false() {
        let dirs = three_dirs();
        let plan = RowSamplePlan::for_sh_rows(&dirs, 15, false, false).unwrap();
        assert_close(plan.transform_flat(), DIPY_SYM_LMAX4_LEGACY_FALSE, 5e-6);
    }

    #[test]
    fn dipy_golden_full_lmax4_legacy_false() {
        let dirs = golden_dirs();
        let plan = RowSamplePlan::for_sh_rows(&dirs, 25, true, false).unwrap();
        let basis = plan.transform_flat();
        assert_eq!(basis.len(), DIPY_FULL_LMAX4_LEGACY_FALSE.len());
        for (i, (got, expect)) in basis
            .iter()
            .zip(DIPY_FULL_LMAX4_LEGACY_FALSE.iter())
            .enumerate()
        {
            let diff = (got - expect).abs();
            assert!(
                diff < 5e-6,
                "mismatch at {i}: got {got}, expected {expect} (Δ={diff:.2e})"
            );
        }
    }

    #[test]
    fn anisotropic_power_full_zero_for_pure_isotropic() {
        // Pure ℓ=0 input: only the DC coefficient is non-zero. AP should
        // be exactly 0 because we skip the ℓ=0 band.
        let mut sh = vec![0.0_f32; ncoeffs_for(4, true)];
        sh[0] = 0.42;
        let ap = anisotropic_power_full_basis(&sh, 4, ANISOTROPIC_POWER_NORM_FACTOR);
        assert_eq!(ap, 0.0);
    }

    #[test]
    fn anisotropic_power_full_includes_odd_l() {
        // Energy in odd-ℓ bands must contribute (this is what
        // distinguishes the full-basis variant from the symmetric AP).
        let mut sh_even = vec![0.0_f32; ncoeffs_for(4, true)];
        let mut sh_odd = vec![0.0_f32; ncoeffs_for(4, true)];
        // Index of (ℓ=1, m=0) = 0 + 1 + 1 = 2 (after the 1 ℓ=0 + 3 ℓ=1)
        // — actually ℓ=0 takes 1 slot, then ℓ=1 occupies 3, so m=0 at ℓ=1
        // is index 2. Index of (ℓ=2, m=0) is 1 + 3 + 2 = 6.
        sh_odd[2] = 0.5;  // ℓ=1, m=0
        sh_even[6] = 0.5; // ℓ=2, m=0
        let ap_odd = anisotropic_power_full_basis(&sh_odd, 4, ANISOTROPIC_POWER_NORM_FACTOR);
        let ap_even = anisotropic_power_full_basis(&sh_even, 4, ANISOTROPIC_POWER_NORM_FACTOR);
        assert!(ap_odd > 0.0, "odd-ℓ energy must contribute");
        assert!(ap_even > 0.0, "even-ℓ energy must contribute");
    }

    #[test]
    fn asymmetry_index_zero_for_symmetric_sh() {
        // Pure even-ℓ energy ⇒ r = 1 ⇒ ASI = 0.
        let mut sh = vec![0.0_f32; ncoeffs_for(4, true)];
        sh[0] = 0.5;  // ℓ=0
        sh[6] = 0.3;  // ℓ=2, m=0
        sh[20] = 0.2; // ℓ=4, m=0 (1+3+5+7+4 = 20)
        let asi = asymmetry_index(&sh, 4);
        assert!(asi.abs() < 1e-6, "expected 0, got {asi}");
    }

    #[test]
    fn asymmetry_index_one_for_pure_asymmetric_sh() {
        // Pure odd-ℓ energy ⇒ r clipped to 0 ⇒ ASI = 1.
        let mut sh = vec![0.0_f32; ncoeffs_for(3, true)];
        sh[2] = 0.5;  // ℓ=1, m=0
        let asi = asymmetry_index(&sh, 3);
        assert!((asi - 1.0).abs() < 1e-6, "expected 1, got {asi}");
    }

    #[test]
    fn sh_derivatives_amp_matches_apply_row() {
        // The amplitude returned by `sh_derivatives` must agree with the
        // matrix-vector evaluation produced by `apply_row` at the same
        // direction. Catches polar-angle vs elevation confusion that
        // produced silently-wrong peak amplitudes during pyAFQ aodf
        // import.
        let sh: Vec<f32> = (0..ncoeffs_for(4, true))
            .map(|i| ((i as f32 * 0.137).sin() * 0.3 + (i as f32 * 0.21).cos() * 0.1))
            .collect();

        let test_dirs = [
            [0.0_f32, 0.0, 1.0],          // +z (north pole, θ=0)
            [0.0, 0.0, -1.0],             // −z (south pole, θ=π)
            [1.0, 0.0, 0.0],              // +x (equator, θ=π/2, φ=0)
            [0.0, 1.0, 0.0],              // +y (equator, θ=π/2, φ=π/2)
            [0.6, 0.8, 0.0],              // arbitrary equatorial
            [-0.4, 0.3, (1.0_f32 - 0.25).sqrt()], // upper hemisphere off-axis
        ];

        for d in test_dirs {
            let mut plan = RowSamplePlan::for_sh_rows(&[d], sh.len(), true, false).unwrap();
            plan.set_clamp_nonnegative(false);
            let amp_apply_row = plan.apply_row(&sh)[0] as f64;

            let theta = (d[0] * d[0] + d[1] * d[1]).sqrt().atan2(d[2]) as f64;
            let phi = (d[1] as f64).atan2(d[0] as f64);
            let (amp_deriv, _, _, _, _, _) =
                sh_derivatives(&sh, 4, true, false, theta, phi);

            assert!(
                (amp_apply_row - amp_deriv).abs() < 1e-4,
                "amp mismatch at {:?}: apply_row={:.6}, derivatives={:.6}, Δ={:.2e}",
                d,
                amp_apply_row,
                amp_deriv,
                (amp_apply_row - amp_deriv).abs()
            );
        }
    }

    #[test]
    fn even_l_symmetric_at_antipodes_full_basis() {
        // ℓ=2 contributions must respect f(d) = f(−d) even in the full basis.
        let dirs: Vec<[f32; 3]> = vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5_f32.sqrt(), 0.5_f32.sqrt(), 0.0],
            [0.6, -0.8, 0.0],
        ];
        let antipodes: Vec<[f32; 3]> = dirs.iter().map(|d| [-d[0], -d[1], -d[2]]).collect();
        let mut coefs = vec![0.0_f32; ncoeffs_for(2, true)];
        coefs[6] = 1.0; // l=2, m=0
        let plan = RowSamplePlan::for_sh_rows(&dirs, coefs.len(), true, false).unwrap();
        let plan_anti = RowSamplePlan::for_sh_rows(&antipodes, coefs.len(), true, false).unwrap();
        for (a, b) in plan
            .apply_row(&coefs)
            .iter()
            .zip(plan_anti.apply_row(&coefs).iter())
        {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
