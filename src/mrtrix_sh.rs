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
