//! NIfTI-style spatial canonicalization to RAS+.
//!
//! Computes a nibabel-compatible orientation transform from a 4×4 voxel-to-
//! world affine, then applies it to volumetric data to produce arrays in
//! canonical RAS+ voxel order with a corresponding updated affine.
//!
//! Used by the MRtrix loader inside this crate, and by external tooling (e.g.
//! cs-odf) that needs its on-disk ODX files to share voxel ordering with
//! `odx convert`'s output.
//!
//! ---------------------------------------------------------------------------
//! Third-party attribution: nibabel
//!
//! The functions [`nibabel_io_orientation`], [`nibabel_ornt_transform`], and
//! [`nibabel_inv_ornt_aff`] in this file are line-by-line ports of routines
//! from the `nibabel.orientations` module of nibabel
//! (<https://nipy.org/nibabel/>), copyright (c) 2009-2024 the nibabel
//! developers. Those portions are made available under the MIT License; see
//! `odx-rs/LICENSE-NIBABEL` for the full notice.

use nalgebra::Matrix3;

/// A spatial axis remapping derived from an input affine to bring voxels to
/// canonical RAS+ ordering.
///
/// Each entry `ornt[i] = [out_axis, flip]` means input axis `i` maps to
/// output axis `out_axis`, with `flip = -1` indicating the axis direction is
/// reversed. Identity is `[[0,1],[1,1],[2,1]]`.
#[derive(Debug, Clone, Copy)]
pub struct CanonTransform {
    ornt: [[i8; 2]; 3],
}

impl CanonTransform {
    /// Compute the transform that brings the spatial axes of an image with
    /// the given voxel-to-world affine into canonical RAS+ ordering.
    pub fn from_affine(affine: [[f64; 4]; 4]) -> Self {
        let start = nibabel_io_orientation(affine);
        let end = [[0, 1], [1, 1], [2, 1]];
        Self {
            ornt: nibabel_ornt_transform(start, end),
        }
    }

    /// True when applying this transform is a no-op.
    pub fn is_identity(&self) -> bool {
        self.ornt == [[0, 1], [1, 1], [2, 1]]
    }

    /// Raw `[out_axis, flip]` triple. Useful only for tests and debug output.
    pub fn ornt(&self) -> [[i8; 2]; 3] {
        self.ornt
    }

    /// Reorient data + affine to RAS+. Returns the canonical dims, the
    /// canonical voxel-to-world affine, and the reordered data buffer.
    ///
    /// `data` must be in the natural row-major NIfTI/ndarray layout
    /// `[x, y, z, ...]`. Trailing 4D/5D channels are preserved as a single
    /// `ncols = dims[3..].iter().product()` block per voxel.
    pub fn apply<T: Copy + Default>(
        &self,
        dims: &[usize],
        affine: [[f64; 4]; 4],
        data: &[T],
    ) -> (Vec<usize>, [[f64; 4]; 4], Vec<T>) {
        if self.is_identity() {
            return (dims.to_vec(), affine, data.to_vec());
        }
        let new_data = reorient_spatial_axes(data, dims, self.ornt);
        let new_affine = compose_affines(affine, nibabel_inv_ornt_aff(self.ornt, &dims[..3]));
        let new_dims = reoriented_dims(dims, self.ornt);
        (new_dims, new_affine, new_data)
    }

    /// In-place variant: rewrite `dims`, `affine`, `data` to canonical RAS+.
    pub fn apply_in_place<T: Copy + Default>(
        &self,
        dims: &mut Vec<usize>,
        affine: &mut [[f64; 4]; 4],
        data: &mut Vec<T>,
    ) {
        if self.is_identity() {
            return;
        }
        let original_dims = dims.clone();
        *data = reorient_spatial_axes(data, &original_dims, self.ornt);
        *affine = compose_affines(*affine, nibabel_inv_ornt_aff(self.ornt, &original_dims[..3]));
        *dims = reoriented_dims(&original_dims, self.ornt);
    }
}

pub(crate) fn reorient_spatial_axes<T: Copy + Default>(
    data: &[T],
    dims: &[usize],
    ornt: [[i8; 2]; 3],
) -> Vec<T> {
    if dims.len() < 3 {
        return data.to_vec();
    }
    let new_dims = reoriented_dims(dims, ornt);
    let old_spatial = [dims[0], dims[1], dims[2]];
    let new_spatial = [new_dims[0], new_dims[1], new_dims[2]];
    let ncols = dims[3..].iter().product::<usize>().max(1);
    let mut out = vec![T::default(); data.len()];

    for x in 0..old_spatial[0] {
        for y in 0..old_spatial[1] {
            for z in 0..old_spatial[2] {
                let old = [x, y, z];
                let mut new = [0usize; 3];
                for src_axis in 0..3 {
                    let mut coord = old[src_axis];
                    if ornt[src_axis][1] == -1 {
                        coord = old_spatial[src_axis] - 1 - coord;
                    }
                    new[ornt[src_axis][0] as usize] = coord;
                }
                let src = ((x * old_spatial[1] + y) * old_spatial[2] + z) * ncols;
                let dst = ((new[0] * new_spatial[1] + new[1]) * new_spatial[2] + new[2]) * ncols;
                out[dst..dst + ncols].copy_from_slice(&data[src..src + ncols]);
            }
        }
    }
    out
}

pub(crate) fn reoriented_dims(dims: &[usize], ornt: [[i8; 2]; 3]) -> Vec<usize> {
    let mut out = dims.to_vec();
    for (src_axis, [dst_axis, _]) in ornt.into_iter().enumerate() {
        out[dst_axis as usize] = dims[src_axis];
    }
    out
}

pub(crate) fn compose_affines(a: [[f64; 4]; 4], b: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut out = [[0.0f64; 4]; 4];
    for row in 0..4 {
        for col in 0..4 {
            out[row][col] = (0..4).map(|k| a[row][k] * b[k][col]).sum();
        }
    }
    out
}

pub(crate) fn nibabel_inv_ornt_aff(ornt: [[i8; 2]; 3], shape: &[usize]) -> [[f64; 4]; 4] {
    let shape = [shape[0] as f64, shape[1] as f64, shape[2] as f64];
    let center = [
        -(shape[0] - 1.0) / 2.0,
        -(shape[1] - 1.0) / 2.0,
        -(shape[2] - 1.0) / 2.0,
    ];

    let mut undo_reorder = [[0.0f64; 4]; 4];
    for row in 0..3 {
        undo_reorder[row][ornt[row][0] as usize] = 1.0;
    }
    undo_reorder[3][3] = 1.0;

    let mut undo_flip = [[0.0f64; 4]; 4];
    for axis in 0..3 {
        let flip = f64::from(ornt[axis][1]);
        undo_flip[axis][axis] = flip;
        undo_flip[axis][3] = (flip * center[axis]) - center[axis];
    }
    undo_flip[3][3] = 1.0;
    compose_affines(undo_flip, undo_reorder)
}

pub(crate) fn nibabel_io_orientation(aff: [[f64; 4]; 4]) -> [[i8; 2]; 3] {
    let rzs = Matrix3::new(
        aff[0][0], aff[0][1], aff[0][2], aff[1][0], aff[1][1], aff[1][2], aff[2][0], aff[2][1],
        aff[2][2],
    );
    let zooms = affine_column_norms(aff);
    let rs = Matrix3::new(
        rzs[(0, 0)] / f64::from(zooms[0].max(1e-12)),
        rzs[(0, 1)] / f64::from(zooms[1].max(1e-12)),
        rzs[(0, 2)] / f64::from(zooms[2].max(1e-12)),
        rzs[(1, 0)] / f64::from(zooms[0].max(1e-12)),
        rzs[(1, 1)] / f64::from(zooms[1].max(1e-12)),
        rzs[(1, 2)] / f64::from(zooms[2].max(1e-12)),
        rzs[(2, 0)] / f64::from(zooms[0].max(1e-12)),
        rzs[(2, 1)] / f64::from(zooms[1].max(1e-12)),
        rzs[(2, 2)] / f64::from(zooms[2].max(1e-12)),
    );
    let svd = rs.svd(true, true);
    let u = svd.u.expect("requested U from SVD");
    let vt = svd.v_t.expect("requested V^T from SVD");
    let s = svd.singular_values;
    let tol = s.max() * 3.0 * f64::EPSILON;
    let mut r = Matrix3::<f64>::zeros();
    for idx in 0..3 {
        if s[idx] > tol {
            let ui = u.column(idx);
            let vti = vt.row(idx);
            r += ui * vti;
        }
    }

    let mut in_axes = [0usize, 1, 2];
    in_axes.sort_by(|&a, &b| {
        let sa = (0..3).map(|row| r[(row, a)].powi(2)).fold(0.0, f64::max);
        let sb = (0..3).map(|row| r[(row, b)].powi(2)).fold(0.0, f64::max);
        sb.total_cmp(&sa)
    });

    let mut ornt = [[-1i8, 1i8]; 3];
    let mut work = r;
    for in_ax in in_axes {
        let mut out_ax = 0usize;
        let mut best = 0.0f64;
        for row in 0..3 {
            let value = work[(row, in_ax)].abs();
            if value > best {
                best = value;
                out_ax = row;
            }
        }
        ornt[in_ax][0] = out_ax as i8;
        ornt[in_ax][1] = if work[(out_ax, in_ax)] < 0.0 { -1 } else { 1 };
        for col in 0..3 {
            work[(out_ax, col)] = 0.0;
        }
    }
    ornt
}

pub(crate) fn nibabel_ornt_transform(
    start_ornt: [[i8; 2]; 3],
    end_ornt: [[i8; 2]; 3],
) -> [[i8; 2]; 3] {
    let mut result = [[0i8, 1i8]; 3];
    for (end_in_idx, [end_out_idx, end_flip]) in end_ornt.into_iter().enumerate() {
        for (start_in_idx, [start_out_idx, start_flip]) in start_ornt.into_iter().enumerate() {
            if end_out_idx == start_out_idx {
                result[start_in_idx] = [
                    end_in_idx as i8,
                    if start_flip == end_flip { 1 } else { -1 },
                ];
                break;
            }
        }
    }
    result
}

pub(crate) fn affine_column_norms(aff: [[f64; 4]; 4]) -> [f32; 3] {
    [
        (aff[0][0] * aff[0][0] + aff[1][0] * aff[1][0] + aff[2][0] * aff[2][0]).sqrt() as f32,
        (aff[0][1] * aff[0][1] + aff[1][1] * aff[1][1] + aff[2][1] * aff[2][1]).sqrt() as f32,
        (aff[0][2] * aff[0][2] + aff[1][2] * aff[1][2] + aff[2][2] * aff[2][2]).sqrt() as f32,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ras_affine() -> [[f64; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    fn las_affine() -> [[f64; 4]; 4] {
        // Negative x voxel direction (left-anterior-superior input).
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    #[test]
    fn identity_for_ras_input() {
        let xf = CanonTransform::from_affine(ras_affine());
        assert!(xf.is_identity());
    }

    #[test]
    fn apply_is_noop_for_ras_input() {
        let xf = CanonTransform::from_affine(ras_affine());
        let dims = vec![2, 2, 2];
        let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let (out_dims, out_aff, out_data) = xf.apply(&dims, ras_affine(), &data);
        assert_eq!(out_dims, dims);
        assert_eq!(out_aff, ras_affine());
        assert_eq!(out_data, data);
    }

    #[test]
    fn las_input_flips_x_axis() {
        let xf = CanonTransform::from_affine(las_affine());
        assert!(!xf.is_identity());
        let dims = vec![2, 1, 1];
        let data: Vec<f32> = vec![10.0, 20.0];
        let (out_dims, _out_aff, out_data) = xf.apply(&dims, las_affine(), &data);
        assert_eq!(out_dims, dims);
        assert_eq!(out_data, vec![20.0, 10.0]);
    }

    #[test]
    fn las_input_4d_preserves_trailing_channels() {
        let xf = CanonTransform::from_affine(las_affine());
        let dims = vec![2, 1, 1, 3];
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (out_dims, _, out_data) = xf.apply(&dims, las_affine(), &data);
        assert_eq!(out_dims, dims);
        assert_eq!(out_data, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn apply_updates_affine_to_ras_orientation() {
        // LAS input: voxel (0,0,0) -> world (0,0,0), voxel (3,0,0) -> (-3,0,0).
        // After flipping x to RAS, we want voxel (0,0,0) -> (-3,0,0) and
        // voxel (3,0,0) -> (0,0,0). So new affine row 0 = [1, 0, 0, -3].
        let xf = CanonTransform::from_affine(las_affine());
        let dims = vec![4, 1, 1];
        let data: Vec<f32> = vec![0.0; 4];
        let (_, out_aff, _) = xf.apply(&dims, las_affine(), &data);
        assert!((out_aff[0][0] - 1.0).abs() < 1e-12);
        assert!((out_aff[0][3] - (-3.0)).abs() < 1e-12);
    }

    #[test]
    fn apply_in_place_matches_apply() {
        let xf = CanonTransform::from_affine(las_affine());
        let dims_v = vec![2, 1, 1, 2];
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let (exp_dims, exp_aff, exp_data) = xf.apply(&dims_v, las_affine(), &data);

        let mut got_dims = dims_v.clone();
        let mut got_aff = las_affine();
        let mut got_data = data.clone();
        xf.apply_in_place(&mut got_dims, &mut got_aff, &mut got_data);

        assert_eq!(got_dims, exp_dims);
        assert_eq!(got_aff, exp_aff);
        assert_eq!(got_data, exp_data);
    }

    #[test]
    fn permuted_axes_are_canonicalized() {
        // Affine that puts input x into world y (and vice versa).
        let aff: [[f64; 4]; 4] = [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let xf = CanonTransform::from_affine(aff);
        assert!(!xf.is_identity());
        let dims = vec![2, 3, 1];
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let (out_dims, _, out_data) = xf.apply(&dims, aff, &data);
        // x↔y swap: new dims should be [3, 2, 1]; data reordered accordingly.
        assert_eq!(out_dims, vec![3, 2, 1]);
        // Original layout (x,y,z) → flat[x*ydim*zdim + y*zdim + z]:
        //   (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=3, (1,1)=4, (1,2)=5
        // After swap, new (x',y',z')=(y,x,z), flat[x'*ny'*nz' + y'*nz' + z']:
        //   new (0,0,0)=old(0,0)=0
        //   new (0,1,0)=old(1,0)=3
        //   new (1,0,0)=old(0,1)=1
        //   new (1,1,0)=old(1,1)=4
        //   new (2,0,0)=old(0,2)=2
        //   new (2,1,0)=old(1,2)=5
        assert_eq!(out_data, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }
}
