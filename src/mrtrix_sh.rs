// Portions of this file (notably `sh_derivatives` and its helpers `index_mpos`,
// `nfor_l_mpos`, and `pack_al`) are derivative works ported from MRtrix3
// (https://www.mrtrix.org/), specifically `Math::SH::derivatives` in
// `core/math/SH.h`.
//
// Original copyright: Copyright (c) 2008-2026 the MRtrix3 contributors.
//
// Those portions are made available under the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at http://mozilla.org/MPL/2.0/. A copy of the license is
// also included in the odx-rs source tree at `LICENSE-MRTRIX`.

use nalgebra::DMatrix;
use ndarray::Array2;

use crate::error::{OdxError, Result};

const Y00: f64 = 0.282_094_791_773_878;

pub fn ncoeffs_for_lmax(lmax: usize) -> usize {
    (lmax + 1) * (lmax + 2) / 2
}

pub fn lmax_for_ncoeffs(n: usize) -> Result<usize> {
    if n == 0 {
        return Ok(0);
    }
    let lmax = 2 * ((((1 + 8 * n) as f64).sqrt() - 3.0) / 4.0).floor() as usize;
    if ncoeffs_for_lmax(lmax) == n {
        Ok(lmax)
    } else {
        Err(OdxError::Format(format!(
            "coefficient count {n} is not a feasible MRtrix SH cardinality"
        )))
    }
}

/// Default `norm_factor` used by dipy's `reconst.shm.anisotropic_power`.
pub const ANISOTROPIC_POWER_NORM_FACTOR: f64 = 1e-5;

/// Anisotropic-power scalar (Dell'Acqua 2014; matches dipy
/// `reconst.shm.anisotropic_power`).
///
/// Computes the mean of squared SH coefficients within each even ℓ ≥ 2 band,
/// sums across bands, then returns `max(0, log(AP_raw) − log(norm_factor))`.
/// The ℓ = 0 band is skipped because it carries the isotropic baseline. `sh`
/// must be in even-only Tournier ordering with `(lmax+1)(lmax+2)/2` entries.
pub fn anisotropic_power(sh: &[f32], lmax: usize, norm_factor: f64) -> f32 {
    debug_assert_eq!(sh.len(), ncoeffs_for_lmax(lmax));
    let mut start = 1_usize; // skip ℓ=0
    let mut ap_raw = 0.0_f64;
    let mut ell = 2_usize;
    while ell <= lmax {
        let len = 2 * ell + 1;
        let stop = start + len;
        let mut s = 0.0_f64;
        for c in &sh[start..stop] {
            s += (*c as f64) * (*c as f64);
        }
        ap_raw += s / len as f64;
        start = stop;
        ell += 2;
    }
    if ap_raw <= 0.0 {
        return 0.0;
    }
    let v = ap_raw.ln() - norm_factor.ln();
    if v <= 0.0 { 0.0 } else { v as f32 }
}

pub fn max_lmax_for_direction_count(ndir: usize) -> usize {
    let mut lmax = 0usize;
    loop {
        let next = lmax + 2;
        if ncoeffs_for_lmax(next) > ndir {
            return lmax;
        }
        lmax = next;
    }
}

/// MRtrix stores the real even-order basis using index = l(l+1)/2 + m.
///
/// Keeping the exact indexing rule here avoids drift between the Rust sampling
/// code and MRtrix's own `Math::SH::index()` implementation.
pub fn coefficient_index(l: usize, m: isize) -> usize {
    ((l * (l + 1) / 2) as isize)
        .checked_add(m)
        .expect("invalid spherical harmonic coefficient index") as usize
}

fn plm_sph_helper(x: f64, m: f64) -> f64 {
    if m < 1.0 {
        1.0
    } else {
        x * (m - 1.0) / m * plm_sph_helper(x, m - 2.0)
    }
}

fn plm_sph_array(lmax: usize, m: usize, x: f64) -> Vec<f64> {
    let mut out = vec![0.0f64; lmax + 1];
    let x2 = x * x;
    if m > 0 && x2 >= 1.0 {
        return out;
    }
    out[m] = Y00;
    if m > 0 {
        out[m] *= (((2 * m + 1) as f64) * plm_sph_helper(1.0 - x2, 2.0 * m as f64)).sqrt();
    }
    if (m & 1) == 1 {
        out[m] = -out[m];
    }
    if lmax == m {
        return out;
    }

    let mut f = (2 * m + 3) as f64;
    f = f.sqrt();
    out[m + 1] = x * f * out[m];

    for n in (m + 2)..=lmax {
        out[n] = x * out[n - 1] - out[n - 2] / f;
        let nf = n as f64;
        let mf = m as f64;
        f = ((4.0 * nf * nf - 1.0) / (nf * nf - mf * mf)).sqrt();
        out[n] *= f;
    }
    out
}

/// AL slot count for `index_mpos` packing — `(lmax/2 + 1)²`.
fn nfor_l_mpos(lmax: usize) -> usize {
    let k = lmax / 2 + 1;
    k * k
}

/// MRtrix `index_mpos(l, m) = l*l/4 + m`. Valid for non-negative `m`.
fn index_mpos(l: usize, m: usize) -> usize {
    l * l / 4 + m
}

/// Pack `Plm_sph(l, m, cos_el)` into the MRtrix AL layout for all even `l`
/// in `0..=lmax` and `m` in `0..=l`.
fn pack_al(lmax: usize, cos_el: f64) -> Vec<f64> {
    let mut al = vec![0.0_f64; nfor_l_mpos(lmax)];
    for m in 0..=lmax {
        let buf = plm_sph_array(lmax, m, cos_el);
        let l_start = if m % 2 == 1 { m + 1 } else { m };
        let mut l = l_start;
        while l <= lmax {
            al[index_mpos(l, m)] = buf[l];
            l += 2;
        }
    }
    al
}

/// First and second partial derivatives of an MRtrix SH series at
/// `(elevation, azimuth)`, returning
/// `(amplitude, ∂/∂el, ∂/∂az, ∂²/∂el², ∂²/∂el∂az, ∂²/∂az²)`.
///
/// Mirrors `Math::SH::derivatives` from
/// `trx-mrtrix2/cpp/core/math/SH.h`. Used by the Newton peak refinement.
pub(crate) fn sh_derivatives(
    sh: &[f32],
    lmax: usize,
    el: f64,
    az: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    debug_assert_eq!(sh.len(), ncoeffs_for_lmax(lmax));
    let sin_el = el.sin();
    let cos_el = el.cos();
    let atpole = sin_el < 1e-4;

    let al = pack_al(lmax, cos_el);

    let mut amplitude = sh[coefficient_index(0, 0)] as f64 * al[index_mpos(0, 0)];
    let mut d_sh_del = 0.0_f64;
    let mut d_sh_daz = 0.0_f64;
    let mut d2_sh_del2 = 0.0_f64;
    let mut d2_sh_deldaz = 0.0_f64;
    let mut d2_sh_daz2 = 0.0_f64;

    let mut l = 2_usize;
    while l <= lmax {
        let v = sh[coefficient_index(l, 0)] as f64;
        amplitude += v * al[index_mpos(l, 0)];
        d_sh_del += v * ((l * (l + 1)) as f64).sqrt() * al[index_mpos(l, 1)];
        let term = ((l * (l + 1) * (l - 1) * (l + 2)) as f64).sqrt() * al[index_mpos(l, 2)]
            - (l * (l + 1)) as f64 * al[index_mpos(l, 0)];
        d2_sh_del2 += v * term / 2.0;
        l += 2;
    }

    let sqrt2 = std::f64::consts::SQRT_2;
    for m in 1..=lmax {
        let mf = m as f64;
        let caz = sqrt2 * (mf * az).cos();
        let saz = sqrt2 * (mf * az).sin();
        let l_start = if m % 2 == 1 { m + 1 } else { m };
        let mut l = l_start;
        while l <= lmax {
            let vp = sh[coefficient_index(l, m as isize)] as f64;
            let vm = sh[coefficient_index(l, -(m as isize))] as f64;
            let cs = vp * caz + vm * saz;
            let sc = vm * caz - vp * saz;

            amplitude += cs * al[index_mpos(l, m)];

            let mut tmp = (((l + m) * (l - m + 1)) as f64).sqrt() * al[index_mpos(l, m - 1)];
            if l > m {
                tmp -= (((l - m) * (l + m + 1)) as f64).sqrt() * al[index_mpos(l, m + 1)];
            }
            tmp /= -2.0;
            d_sh_del += cs * tmp;

            let mut tmp2 = -(((l + m) * (l - m + 1) + (l - m) * (l + m + 1)) as f64)
                * al[index_mpos(l, m)];
            if m == 1 {
                tmp2 -= (((l + m) * (l - m + 1) * (l + m - 1) * (l - m + 2)) as f64).sqrt()
                    * al[index_mpos(l, 1)];
            } else {
                tmp2 += (((l + m) * (l - m + 1) * (l + m - 1) * (l - m + 2)) as f64).sqrt()
                    * al[index_mpos(l, m - 2)];
            }
            if l > m + 1 {
                tmp2 += (((l - m) * (l + m + 1) * (l - m - 1) * (l + m + 2)) as f64).sqrt()
                    * al[index_mpos(l, m + 2)];
            }
            tmp2 /= 4.0;
            d2_sh_del2 += cs * tmp2;

            if atpole {
                d_sh_daz += sc * tmp;
            } else {
                d2_sh_deldaz += mf * sc * tmp;
                d_sh_daz += mf * sc * al[index_mpos(l, m)];
                d2_sh_daz2 -= cs * (mf * mf) * al[index_mpos(l, m)];
            }

            l += 2;
        }
    }

    if !atpole {
        d_sh_daz /= sin_el;
        d2_sh_deldaz /= sin_el;
        d2_sh_daz2 /= sin_el * sin_el;
    }

    (
        amplitude,
        d_sh_del,
        d_sh_daz,
        d2_sh_del2,
        d2_sh_deldaz,
        d2_sh_daz2,
    )
}

fn sh_transform_row(dir: [f32; 3], lmax: usize) -> Vec<f32> {
    let x = dir[0] as f64;
    let y = dir[1] as f64;
    let z = dir[2] as f64;
    let rxy = x.hypot(y);
    let cp = if rxy > 0.0 { x / rxy } else { 1.0 };
    let sp = if rxy > 0.0 { y / rxy } else { 0.0 };

    let mut row = vec![0.0f32; ncoeffs_for_lmax(lmax)];
    let al0 = plm_sph_array(lmax, 0, z);
    for l in (0..=lmax).step_by(2) {
        row[coefficient_index(l, 0)] = al0[l] as f32;
    }

    let mut c0 = 1.0f64;
    let mut s0 = 0.0f64;
    for m in 1..=lmax {
        let alm = plm_sph_array(lmax, m, z);
        let c = c0 * cp - s0 * sp;
        let s = s0 * cp + c0 * sp;
        for l in ((if (m & 1) == 1 { m + 1 } else { m })..=lmax).step_by(2) {
            let base = (std::f64::consts::SQRT_2 * alm[l]) as f32;
            row[coefficient_index(l, m as isize)] = base * c as f32;
            row[coefficient_index(l, -(m as isize))] = base * s as f32;
        }
        c0 = c;
        s0 = s;
    }
    row
}

fn array2_from_row_major(rows: usize, cols: usize, data: Vec<f32>) -> Array2<f32> {
    Array2::from_shape_vec((rows, cols), data).expect("shape mismatch while building Array2")
}

fn to_dmatrix(a: &Array2<f32>) -> DMatrix<f64> {
    let (rows, cols) = a.dim();
    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            data.push(a[[r, c]] as f64);
        }
    }
    DMatrix::from_row_slice(rows, cols, &data)
}

fn from_dmatrix(m: &DMatrix<f64>) -> Array2<f32> {
    let mut data = Vec::with_capacity(m.nrows() * m.ncols());
    for r in 0..m.nrows() {
        for c in 0..m.ncols() {
            data.push(m[(r, c)] as f32);
        }
    }
    array2_from_row_major(m.nrows(), m.ncols(), data)
}

fn pseudoinverse(m: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    if m.nrows() >= m.ncols() {
        let lhs = m.transpose() * m;
        let rhs = m.transpose();
        if let Some(chol) = lhs.clone().cholesky() {
            Ok(chol.solve(&rhs))
        } else if let Some(sol) = lhs.lu().solve(&rhs) {
            Ok(sol)
        } else {
            Err(OdxError::Format(
                "failed to solve MRtrix SH normal equations".into(),
            ))
        }
    } else {
        let lhs = m * m.transpose();
        if let Some(chol) = lhs.clone().cholesky() {
            Ok(chol.solve(m).transpose())
        } else if let Some(sol) = lhs.lu().solve(m) {
            Ok(sol.transpose())
        } else {
            Err(OdxError::Format(
                "failed to solve MRtrix SH normal equations".into(),
            ))
        }
    }
}

fn condition_number(m: &DMatrix<f64>) -> f64 {
    let svd = m.clone().svd(false, false);
    let mut min_sv = f64::INFINITY;
    let mut max_sv = 0.0f64;
    for sv in svd.singular_values.iter().copied() {
        if sv > 0.0 {
            min_sv = min_sv.min(sv);
            max_sv = max_sv.max(sv);
        }
    }
    if min_sv.is_finite() && min_sv > 0.0 {
        max_sv / min_sv
    } else {
        f64::INFINITY
    }
}

#[derive(Debug, Clone)]
pub struct RowSamplePlan {
    transform: Array2<f32>,
    ncoeffs: usize,
    ndir: usize,
    clamp_nonnegative: bool,
}

impl RowSamplePlan {
    pub fn for_sh_rows_nonnegative(dirs_ras: &[[f32; 3]], ncoeffs: usize) -> Result<Self> {
        let lmax = lmax_for_ncoeffs(ncoeffs)?;
        let transform = sh2amp_cart(dirs_ras, lmax);
        let (ndir, cols) = transform.dim();
        Ok(Self {
            transform,
            ncoeffs: cols,
            ndir,
            clamp_nonnegative: true,
        })
    }

    pub fn ncoeffs(&self) -> usize {
        self.ncoeffs
    }

    pub fn ndir(&self) -> usize {
        self.ndir
    }

    pub fn apply_row_into(&self, src: &[f32], dst: &mut [f32]) {
        assert_eq!(src.len(), self.ncoeffs);
        assert_eq!(dst.len(), self.ndir);
        apply_transform_row_into(&self.transform, src, dst);
        if self.clamp_nonnegative {
            for value in dst.iter_mut() {
                *value = value.max(0.0);
            }
        }
    }

    pub fn apply_row(&self, src: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.ndir];
        self.apply_row_into(src, &mut out);
        out
    }

    pub fn transform_flat(&self) -> &[f32] {
        self.transform
            .as_slice()
            .expect("MRtrix SH transform should be contiguous")
    }

    pub fn source_dir_count(&self) -> usize {
        self.ndir
    }
}

#[derive(Debug, Clone)]
pub struct RowFitPlan {
    transform: Array2<f32>,
    ndir: usize,
    ncoeffs: usize,
}

impl RowFitPlan {
    pub fn for_amplitudes(dirs_ras: &[[f32; 3]], lmax: usize) -> Result<Self> {
        let transform = amp2sh_cart(dirs_ras, lmax)?;
        let (ncoeffs, ndir) = transform.dim();
        Ok(Self {
            transform,
            ndir,
            ncoeffs,
        })
    }

    pub fn source_dir_count(&self) -> usize {
        self.ndir
    }

    pub fn target_coeff_count(&self) -> usize {
        self.ncoeffs
    }

    pub fn apply_row_into(&self, src: &[f32], dst: &mut [f32]) {
        assert_eq!(src.len(), self.ndir);
        assert_eq!(dst.len(), self.ncoeffs);
        apply_transform_row_into(&self.transform, src, dst);
    }

    pub fn apply_row(&self, src: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.ncoeffs];
        self.apply_row_into(src, &mut out);
        out
    }
}

pub fn sh2amp_cart(dirs_ras: &[[f32; 3]], lmax: usize) -> Array2<f32> {
    let mut data = Vec::with_capacity(dirs_ras.len() * ncoeffs_for_lmax(lmax));
    for &dir in dirs_ras {
        data.extend(sh_transform_row(dir, lmax));
    }
    array2_from_row_major(dirs_ras.len(), ncoeffs_for_lmax(lmax), data)
}

pub fn amp2sh_cart(dirs_ras: &[[f32; 3]], lmax: usize) -> Result<Array2<f32>> {
    let sh2amp = sh2amp_cart(dirs_ras, lmax);
    let pinv = pseudoinverse(&to_dmatrix(&sh2amp))?;
    Ok(from_dmatrix(&pinv))
}

pub fn sample_nonnegative(coeffs: &[f32], dirs_ras: &[[f32; 3]]) -> Result<Vec<f32>> {
    let lmax = lmax_for_ncoeffs(coeffs.len())?;
    let transform = sh2amp_cart(dirs_ras, lmax);
    Ok(apply_transform_row(&transform, coeffs)
        .into_iter()
        // Matching MRtrix `sh2amp -nonnegative`: clamp after evaluation.
        .map(|value| value.max(0.0))
        .collect())
}

pub fn fit_from_amplitudes(
    amplitudes: &[f32],
    dirs_ras: &[[f32; 3]],
    lmax: usize,
) -> Result<Vec<f32>> {
    let transform = amp2sh_cart(dirs_ras, lmax)?;
    Ok(apply_transform_row(&transform, amplitudes))
}

pub fn resolve_lmax_for_directions(
    dirs_ras: &[[f32; 3]],
    requested_lmax: Option<usize>,
    default_lmax: usize,
) -> usize {
    let lmax_from_ndir = max_lmax_for_direction_count(dirs_ras.len());
    let mut lmax = requested_lmax.unwrap_or(lmax_from_ndir.min(default_lmax));
    if lmax > lmax_from_ndir {
        lmax = lmax_from_ndir;
    }
    if lmax % 2 == 1 {
        lmax -= 1;
    }

    let mut current = lmax;
    loop {
        let transform = sh2amp_cart(dirs_ras, current);
        let cond = condition_number(&to_dmatrix(&transform));
        if cond < 100.0 || current < 2 || requested_lmax.is_some() {
            return current;
        }
        current -= 2;
    }
}

pub fn sample_rows_nonnegative(
    coeffs: &[f32],
    nrows: usize,
    dirs_ras: &[[f32; 3]],
    ncoeffs: usize,
) -> Result<Vec<f32>> {
    let plan = RowSamplePlan::for_sh_rows_nonnegative(dirs_ras, ncoeffs)?;
    let ndir = plan.ndir();
    let mut out = vec![0.0f32; nrows * ndir];
    for row in 0..nrows {
        let src = &coeffs[row * ncoeffs..(row + 1) * ncoeffs];
        let dst = &mut out[row * ndir..(row + 1) * ndir];
        plan.apply_row_into(src, dst);
    }
    Ok(out)
}

pub fn fit_rows_from_amplitudes(
    amplitudes: &[f32],
    nrows: usize,
    dirs_ras: &[[f32; 3]],
    lmax: usize,
) -> Result<Vec<f32>> {
    let plan = RowFitPlan::for_amplitudes(dirs_ras, lmax)?;
    let ncoeffs = plan.target_coeff_count();
    let ndir = plan.source_dir_count();
    let mut out = vec![0.0f32; nrows * ncoeffs];
    for row in 0..nrows {
        let src = &amplitudes[row * ndir..(row + 1) * ndir];
        let dst = &mut out[row * ncoeffs..(row + 1) * ncoeffs];
        plan.apply_row_into(src, dst);
    }
    Ok(out)
}

fn apply_transform_row(transform: &Array2<f32>, src: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; transform.dim().0];
    apply_transform_row_into(transform, src, &mut out);
    out
}

fn apply_transform_row_into(transform: &Array2<f32>, src: &[f32], dst: &mut [f32]) {
    let (rows, cols) = transform.dim();
    assert_eq!(src.len(), cols);
    assert_eq!(dst.len(), rows);
    for r in 0..rows {
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += transform[[r, c]] * src[c];
        }
        dst[r] = acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::dsistudio_odf8;

    #[test]
    fn lmax_and_ncoeffs_round_trip() {
        for lmax in (0..=8).step_by(2) {
            let n = ncoeffs_for_lmax(lmax);
            assert_eq!(lmax_for_ncoeffs(n).unwrap(), lmax);
        }
    }

    #[test]
    fn max_lmax_for_direction_count_supports_overdetermined_fits() {
        assert_eq!(max_lmax_for_direction_count(1), 0);
        assert_eq!(max_lmax_for_direction_count(6), 2);
        assert_eq!(max_lmax_for_direction_count(15), 4);
        assert_eq!(max_lmax_for_direction_count(28), 6);
        assert_eq!(max_lmax_for_direction_count(45), 8);
        assert_eq!(max_lmax_for_direction_count(200), 18);
        assert_eq!(max_lmax_for_direction_count(321), 22);
    }

    #[test]
    fn resolve_lmax_uses_direction_capacity_not_exact_cardinality() {
        let dirs = dsistudio_odf8::hemisphere_vertices_ras().to_vec();
        assert_eq!(resolve_lmax_for_directions(&dirs, None, 8), 8);
    }

    #[test]
    fn coefficient_index_matches_mrtrix_formula() {
        assert_eq!(coefficient_index(0, 0), 0);
        assert_eq!(coefficient_index(2, -2), 1);
        assert_eq!(coefficient_index(2, 0), 3);
        assert_eq!(coefficient_index(8, 8), 44);
    }

    #[test]
    fn transform_and_fit_round_trip() {
        let dirs = vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.57735026, 0.57735026, 0.57735026],
            [-0.57735026, 0.57735026, 0.57735026],
            [0.57735026, -0.57735026, 0.57735026],
        ];
        let coeffs = vec![1.0, 0.1, 0.2, 0.3, 0.4, 0.5];
        let amp = sample_nonnegative(&coeffs, &dirs).unwrap();
        let fit = fit_from_amplitudes(&amp, &dirs, 2).unwrap();
        assert_eq!(fit.len(), coeffs.len());
    }

    #[test]
    fn row_sample_plan_matches_batch_sampling() {
        let dirs = vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.57735026, 0.57735026, 0.57735026],
        ];
        let coeffs = vec![
            1.0, 0.1, 0.2, 0.3, 0.4, 0.5, //
            0.8, 0.2, 0.1, 0.0, -0.1, 0.3,
        ];
        let expected = sample_rows_nonnegative(&coeffs, 2, &dirs, 6).unwrap();
        let plan = RowSamplePlan::for_sh_rows_nonnegative(&dirs, 6).unwrap();
        let mut actual = vec![0.0f32; expected.len()];
        for row in 0..2 {
            plan.apply_row_into(
                &coeffs[row * 6..(row + 1) * 6],
                &mut actual[row * plan.ndir()..(row + 1) * plan.ndir()],
            );
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn row_sample_plan_reuses_output_buffer() {
        let dirs = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let coeffs = vec![1.0, 0.1, 0.2, 0.3, 0.4, 0.5];
        let plan = RowSamplePlan::for_sh_rows_nonnegative(&dirs, coeffs.len()).unwrap();
        let mut out = vec![f32::NAN; plan.ndir()];
        plan.apply_row_into(&coeffs, &mut out);
        assert!(out.iter().all(|value| value.is_finite()));
        let second = plan.apply_row(&coeffs);
        assert_eq!(out, second);
    }

    #[test]
    fn anisotropic_power_is_zero_for_pure_ell0() {
        // Pure isotropic SH (only ℓ=0 nonzero) has no anisotropy.
        let sh = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0];
        let ap = anisotropic_power(&sh, 2, ANISOTROPIC_POWER_NORM_FACTOR);
        assert_eq!(ap, 0.0);
    }

    #[test]
    fn anisotropic_power_logs_band_mean_squared() {
        // lmax=2, ncoeffs=6: ℓ=0 → idx 0, ℓ=2 → idx 1..6 (5 coeffs).
        // AP_raw = (0² + 1² + 0² + 0² + 0²)/5 = 0.2; AP = ln(0.2) − ln(1e-5).
        let sh = vec![0.0_f32, 1.0, 0.0, 0.0, 0.0, 0.0];
        let ap = anisotropic_power(&sh, 2, 1e-5);
        let expected = (0.2_f64.ln() - 1e-5_f64.ln()) as f32;
        assert!((ap - expected).abs() < 1e-6, "ap={ap}, expected={expected}");
    }

    #[test]
    fn anisotropic_power_skips_ell0_band() {
        // Identical ℓ=2 content with different ℓ=0 content must produce identical AP.
        let sh_a = vec![0.0_f32, 0.3, 0.4, 0.5, 0.0, 0.0];
        let sh_b = vec![5.0_f32, 0.3, 0.4, 0.5, 0.0, 0.0];
        let ap_a = anisotropic_power(&sh_a, 2, ANISOTROPIC_POWER_NORM_FACTOR);
        let ap_b = anisotropic_power(&sh_b, 2, ANISOTROPIC_POWER_NORM_FACTOR);
        assert_eq!(ap_a, ap_b);
    }

    #[test]
    fn anisotropic_power_clamps_to_zero_below_norm_factor() {
        // tiny ℓ=2 energy below norm_factor: AP_raw ≪ norm_factor → clamp to 0.
        let sh = vec![0.0_f32, 1e-6, 0.0, 0.0, 0.0, 0.0];
        let ap = anisotropic_power(&sh, 2, 1e-5);
        assert_eq!(ap, 0.0);
    }

    #[test]
    fn sh_derivatives_amplitude_matches_basis_dot_product() {
        // Amplitude returned by sh_derivatives must equal the dot product of
        // sh_transform_row(dir) and the SH coefficients.
        let lmax = 8;
        let n = ncoeffs_for_lmax(lmax);
        let mut sh = vec![0.0_f32; n];
        // Arbitrary deterministic SH values.
        for i in 0..n {
            sh[i] = ((i as f32 * 0.137).sin() * 0.5 + 0.1).abs();
        }
        let dirs: [[f32; 3]; 5] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5773502, 0.5773502, 0.5773502],
            [-0.6, 0.5, 0.6244998],
        ];
        for dir in dirs {
            let row = sh_transform_row(dir, lmax);
            let expected: f64 = row
                .iter()
                .zip(sh.iter())
                .map(|(a, b)| (*a as f64) * (*b as f64))
                .sum();
            let z = dir[2] as f64;
            let el = (((dir[0] as f64).powi(2) + (dir[1] as f64).powi(2)).sqrt()).atan2(z);
            let az = (dir[1] as f64).atan2(dir[0] as f64);
            let (amp, _, _, _, _, _) = sh_derivatives(&sh, lmax, el, az);
            assert!(
                (amp - expected).abs() < 1e-3,
                "amplitude mismatch at dir={:?}: derivatives={amp}, basis-dot={expected}",
                dir
            );
        }
    }

    #[test]
    fn sh_derivatives_pure_ell0_is_flat() {
        // Pure ℓ=0 SH is constant across the sphere, so all derivatives must vanish.
        let lmax = 4;
        let n = ncoeffs_for_lmax(lmax);
        let mut sh = vec![0.0_f32; n];
        sh[0] = 1.0;
        let (_, dsh_del, dsh_daz, d2_del, d2_da, d2_az) =
            sh_derivatives(&sh, lmax, 0.7, 1.3);
        assert!(dsh_del.abs() < 1e-12);
        assert!(dsh_daz.abs() < 1e-12);
        assert!(d2_del.abs() < 1e-12);
        assert!(d2_da.abs() < 1e-12);
        assert!(d2_az.abs() < 1e-12);
    }

    #[test]
    fn sh_derivatives_match_finite_differences() {
        // Compare analytical derivatives to centered finite differences for a
        // generic SH at random-ish (el, az), away from the pole.
        let lmax = 6;
        let n = ncoeffs_for_lmax(lmax);
        let mut sh = vec![0.0_f32; n];
        for i in 0..n {
            sh[i] = ((i as f32 * 0.31 + 0.2).cos() * 0.4) as f32;
        }
        let el = 0.9_f64;
        let az = 1.7_f64;
        let h = 1e-4_f64;
        let (_, dsh_del, dsh_daz, d2_del2, d2_deldaz, d2_daz2) =
            sh_derivatives(&sh, lmax, el, az);

        let f = |e: f64, a: f64| -> f64 { sh_derivatives(&sh, lmax, e, a).0 };
        let fd_del = (f(el + h, az) - f(el - h, az)) / (2.0 * h);
        let fd_daz = (f(el, az + h) - f(el, az - h)) / (2.0 * h);
        let fd_d2_del2 = (f(el + h, az) - 2.0 * f(el, az) + f(el - h, az)) / (h * h);
        let fd_d2_daz2 = (f(el, az + h) - 2.0 * f(el, az) + f(el, az - h)) / (h * h);
        let fd_d2_deldaz = (f(el + h, az + h) - f(el + h, az - h) - f(el - h, az + h)
            + f(el - h, az - h))
            / (4.0 * h * h);
        // Note: derivatives() returns ∂/∂az already divided by sin(el) (and
        // ∂²/∂az² by sin²(el)), but the *finite differences* of f with
        // respect to az do NOT include that factor. Undo it before comparing.
        let sin_el = el.sin();
        let dsh_daz_raw = dsh_daz * sin_el;
        let d2_daz2_raw = d2_daz2 * sin_el * sin_el;
        let d2_deldaz_raw = d2_deldaz * sin_el;
        assert!((dsh_del - fd_del).abs() < 1e-3, "dSH_del: {dsh_del} vs {fd_del}");
        assert!((dsh_daz_raw - fd_daz).abs() < 1e-3, "dSH_daz: {dsh_daz_raw} vs {fd_daz}");
        assert!(
            (d2_del2 - fd_d2_del2).abs() < 1e-2,
            "d2_del2: {d2_del2} vs {fd_d2_del2}"
        );
        assert!(
            (d2_daz2_raw - fd_d2_daz2).abs() < 1e-2,
            "d2_daz2: {d2_daz2_raw} vs {fd_d2_daz2}"
        );
        assert!(
            (d2_deldaz_raw - fd_d2_deldaz).abs() < 1e-2,
            "d2_deldaz: {d2_deldaz_raw} vs {fd_d2_deldaz}"
        );
    }

    #[test]
    fn row_fit_plan_matches_batch_fitting() {
        let dirs = vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.57735026, 0.57735026, 0.57735026],
            [-0.57735026, 0.57735026, 0.57735026],
            [0.57735026, -0.57735026, 0.57735026],
        ];
        let amplitudes = vec![
            1.0, 0.8, 0.7, 0.9, 0.6, 0.5, //
            0.5, 0.6, 0.7, 0.4, 0.3, 0.2,
        ];
        let expected = fit_rows_from_amplitudes(&amplitudes, 2, &dirs, 2).unwrap();
        let plan = RowFitPlan::for_amplitudes(&dirs, 2).unwrap();
        let mut actual = vec![0.0f32; expected.len()];
        for row in 0..2 {
            plan.apply_row_into(
                &amplitudes[row * plan.source_dir_count()..(row + 1) * plan.source_dir_count()],
                &mut actual[row * plan.target_coeff_count()..(row + 1) * plan.target_coeff_count()],
            );
        }
        assert_eq!(actual, expected);
    }
}
