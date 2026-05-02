//! Sparse-aware sampler for the source ODX dataset's per-voxel arrays
//! (SH coefficients and DPV scalars) under a world-coordinate query.
//!
//! ODX stores per-voxel arrays in compact rows indexed by mask order; they're
//! not a dense 3D volume. To trilinearly sample them at a world point, we
//! invert the source affine to get fractional voxel indices, then look up the
//! 8 corner mask voxels and weight by trilinear factors. Voxels outside the
//! mask contribute zero.

use crate::header::Header;

/// O(1) lookup from `(i, j, k)` to compact row index, or `u32::MAX` for
/// out-of-mask voxels.
pub struct SourceLookup {
    pub dims: [u64; 3],
    pub inv_affine: [[f64; 4]; 4],
    pub flat_to_compact: Vec<u32>,
}

impl SourceLookup {
    pub fn new(header: &Header, mask: &[u8]) -> Self {
        let dims = header.dimensions;
        let total = (dims[0] * dims[1] * dims[2]) as usize;
        let mut flat_to_compact = vec![u32::MAX; total];
        let mut next = 0u32;
        // C-order, matching `compact_to_ijk` in odx_file.rs (i slowest, k fastest).
        for i in 0..dims[0] as usize {
            for j in 0..dims[1] as usize {
                for k in 0..dims[2] as usize {
                    let flat = i * (dims[1] as usize) * (dims[2] as usize)
                        + j * (dims[2] as usize)
                        + k;
                    if mask[flat] != 0 {
                        flat_to_compact[flat] = next;
                        next += 1;
                    }
                }
            }
        }
        let inv = invert_affine(&header.voxel_to_rasmm).unwrap_or_else(|| {
            // Affine should always invert for a valid ODX; fall back to
            // identity so we don't panic on malformed files.
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        });
        Self {
            dims,
            inv_affine: inv,
            flat_to_compact,
        }
    }

    /// World point → fractional voxel index `(fi, fj, fk)`.
    #[inline]
    pub fn world_to_voxel(&self, world: [f64; 3]) -> [f64; 3] {
        let m = &self.inv_affine;
        [
            m[0][0] * world[0] + m[0][1] * world[1] + m[0][2] * world[2] + m[0][3],
            m[1][0] * world[0] + m[1][1] * world[1] + m[1][2] * world[2] + m[1][3],
            m[2][0] * world[0] + m[2][1] * world[1] + m[2][2] * world[2] + m[2][3],
        ]
    }

    /// Voxel-grid index → flat C-order index, or `None` if out of bounds.
    #[inline]
    pub fn ijk_to_flat(&self, i: i64, j: i64, k: i64) -> Option<usize> {
        if i < 0 || j < 0 || k < 0 {
            return None;
        }
        let (i, j, k) = (i as u64, j as u64, k as u64);
        if i >= self.dims[0] || j >= self.dims[1] || k >= self.dims[2] {
            return None;
        }
        Some((i * self.dims[1] * self.dims[2] + j * self.dims[2] + k) as usize)
    }

    /// Compact row for `(i, j, k)`, or `None` if out of bounds or out of mask.
    #[inline]
    pub fn compact_row(&self, i: i64, j: i64, k: i64) -> Option<u32> {
        let flat = self.ijk_to_flat(i, j, k)?;
        let row = self.flat_to_compact[flat];
        if row == u32::MAX { None } else { Some(row) }
    }

    /// Nearest-neighbor source compact row for a world point, or `None`
    /// if the rounded voxel is out of bounds or out of mask.
    pub fn nearest_compact(&self, world: [f64; 3]) -> Option<u32> {
        let v = self.world_to_voxel(world);
        let i = v[0].round() as i64;
        let j = v[1].round() as i64;
        let k = v[2].round() as i64;
        self.compact_row(i, j, k)
    }

    /// Trilinear sampling weights for a world point: 8 corner voxels, with
    /// out-of-mask corners getting zero weight. Returns
    /// `(total_weight, [(compact_row, weight); 8])` — total_weight may be
    /// less than 1 when corners fall outside the mask. Weights are
    /// renormalised by `total_weight` if the caller wants pure interpolation;
    /// for dense-output reconstruction (zero-extended outside mask) leave as
    /// is.
    pub fn trilinear_weights(&self, world: [f64; 3]) -> TrilinearWeights {
        let v = self.world_to_voxel(world);
        let i0 = v[0].floor() as i64;
        let j0 = v[1].floor() as i64;
        let k0 = v[2].floor() as i64;
        let dx = v[0] - i0 as f64;
        let dy = v[1] - j0 as f64;
        let dz = v[2] - k0 as f64;

        let mut entries = [TriCorner::default(); 8];
        let mut total = 0.0;
        let mut idx = 0;
        for di in 0..2_i64 {
            for dj in 0..2_i64 {
                for dk in 0..2_i64 {
                    let wx = if di == 0 { 1.0 - dx } else { dx };
                    let wy = if dj == 0 { 1.0 - dy } else { dy };
                    let wz = if dk == 0 { 1.0 - dz } else { dz };
                    let w = (wx * wy * wz) as f32;
                    if let Some(row) = self.compact_row(i0 + di, j0 + dj, k0 + dk) {
                        entries[idx] = TriCorner { row, weight: w };
                        total += w as f64;
                    } else {
                        entries[idx] = TriCorner::EMPTY;
                    }
                    idx += 1;
                }
            }
        }
        TrilinearWeights {
            total_weight: total as f32,
            corners: entries,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TriCorner {
    pub row: u32,
    pub weight: f32,
}

impl TriCorner {
    pub const EMPTY: Self = Self {
        row: u32::MAX,
        weight: 0.0,
    };
    pub fn is_empty(&self) -> bool {
        self.row == u32::MAX
    }
}

impl Default for TriCorner {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// Result of a trilinear weight build.
pub struct TrilinearWeights {
    /// Sum of weights over in-mask corners. `< 1` near the mask edge.
    /// Exposed so callers can renormalise interpolation if desired (we
    /// currently zero-extend instead, matching nibabel/mrtrix conventions).
    #[allow(dead_code)]
    pub total_weight: f32,
    /// 8 corners; some may be EMPTY (out of mask / out of bounds).
    pub corners: [TriCorner; 8],
}

impl TrilinearWeights {
    /// Sum each corner's `data[row * ncols + col]` into `out[col]`, weighted.
    /// `out` must already be zeroed if the caller wants a clean accumulation.
    pub fn accumulate_row<T>(
        &self,
        data: &[T],
        ncols: usize,
        out: &mut [f32],
        get: impl Fn(&T) -> f32,
    ) where
        T: Copy,
    {
        for c in &self.corners {
            if c.is_empty() {
                continue;
            }
            let base = c.row as usize * ncols;
            for col in 0..ncols {
                out[col] += c.weight * get(&data[base + col]);
            }
        }
    }
}

fn invert_affine(a: &[[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
    let m = nalgebra::Matrix4::from_row_slice(&[
        a[0][0], a[0][1], a[0][2], a[0][3],
        a[1][0], a[1][1], a[1][2], a[1][3],
        a[2][0], a[2][1], a[2][2], a[2][3],
        a[3][0], a[3][1], a[3][2], a[3][3],
    ]);
    let inv = m.try_inverse()?;
    let mut out = [[0.0; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            out[r][c] = inv[(r, c)];
        }
    }
    Some(out)
}
