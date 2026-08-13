//! Icosahedral sphere tessellation, ordered hemisphere-first.
//!
//! Recursive subdivision of the regular icosahedron, projecting new vertices
//! onto the unit sphere. Level `n` yields `10·4ⁿ + 2` vertices and `20·4ⁿ`
//! faces:
//!
//! | level | vertices | hemisphere | faces |
//! |-------|----------|------------|-------|
//! | 2     | 162      | 81         | 320   |
//! | 3     | 642      | **321**    | 1280  |
//! | 4     | 2562     | **1281**   | 5120  |
//!
//! Those hemisphere counts are exactly mrtrix3's `tesselation_321` and
//! `tesselation_1281` (`src/dwi/directions/predefined.cpp`), the direction sets
//! `fod2fixel` segments on — 1281 being its default. Generating them here
//! rather than vendoring mrtrix3's azimuth/elevation tables also gives the
//! **face list**, which those tables lack and which [`crate::fmls`] needs for
//! adjacency.
//!
//! Output is ordered so the first half of `vertices` is a hemisphere and the
//! second half holds the corresponding antipodes, the layout
//! [`crate::fmls::hemisphere_adjacency`] and
//! [`crate::formats::dsistudio_odf8`] both assume.
//!
//! # Provenance
//!
//! This module is an **independent implementation** of standard icosahedral
//! subdivision — it is not derived from MRtrix3 source, and deliberately does
//! not vendor MRtrix3's `tesselation_321` / `tesselation_1281` azimuth/
//! elevation tables from `src/dwi/directions/predefined.cpp`. It is noted here
//! only that levels 3 and 4 reproduce the same direction *counts* those tables
//! provide, so results are comparable with `fod2fixel`. No MRtrix3 code or data
//! is copied.

/// Vertices and faces of a subdivided icosahedron, hemisphere-first.
pub struct IcoSphere {
    /// Unit vectors. `[..len()/2]` is a hemisphere; `[len()/2..]` its antipodes,
    /// paired by offset (vertex `i` and `i + len()/2` are antipodal).
    pub vertices: Vec<[f32; 3]>,
    /// Triangles indexing into `vertices` (full sphere).
    pub faces: Vec<[u32; 3]>,
}

impl IcoSphere {
    pub fn hemisphere(&self) -> &[[f32; 3]] {
        &self.vertices[..self.vertices.len() / 2]
    }
}

/// Build the level-`n` icosphere.
///
/// `level` is clamped to 6 (2 621 442 vertices); beyond that the intended use
/// has almost certainly been misunderstood, and the memory cost is severe.
pub fn icosphere(level: usize) -> IcoSphere {
    let level = level.min(6);
    let (mut verts, mut faces) = icosahedron();
    for _ in 0..level {
        (verts, faces) = subdivide(&verts, &faces);
    }
    reorder_hemisphere_first(verts, faces)
}

fn icosahedron() -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let t = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let raw: [[f64; 3]; 12] = [
        [-1.0, t, 0.0], [1.0, t, 0.0], [-1.0, -t, 0.0], [1.0, -t, 0.0],
        [0.0, -1.0, t], [0.0, 1.0, t], [0.0, -1.0, -t], [0.0, 1.0, -t],
        [t, 0.0, -1.0], [t, 0.0, 1.0], [-t, 0.0, -1.0], [-t, 0.0, 1.0],
    ];
    let verts = raw.iter().map(|v| normalize(*v)).collect();
    let faces = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    (verts, faces)
}

#[inline]
fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / n, v[1] / n, v[2] / n]
}

fn subdivide(verts: &[[f64; 3]], faces: &[[u32; 3]]) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    use std::collections::HashMap;
    let mut out_v = verts.to_vec();
    let mut out_f = Vec::with_capacity(faces.len() * 4);
    // Edge midpoints are shared by two triangles; cache them so the vertex
    // count comes out exact (10·4ⁿ+2) rather than with duplicates.
    let mut mid: HashMap<(u32, u32), u32> = HashMap::new();
    let mut midpoint = |a: u32, b: u32, out_v: &mut Vec<[f64; 3]>| -> u32 {
        let key = if a < b { (a, b) } else { (b, a) };
        if let Some(&i) = mid.get(&key) {
            return i;
        }
        let (p, q) = (out_v[a as usize], out_v[b as usize]);
        let m = normalize([
            (p[0] + q[0]) / 2.0,
            (p[1] + q[1]) / 2.0,
            (p[2] + q[2]) / 2.0,
        ]);
        let i = out_v.len() as u32;
        out_v.push(m);
        mid.insert(key, i);
        i
    };
    for f in faces {
        let a = midpoint(f[0], f[1], &mut out_v);
        let b = midpoint(f[1], f[2], &mut out_v);
        let c = midpoint(f[2], f[0], &mut out_v);
        out_f.push([f[0], a, c]);
        out_f.push([f[1], b, a]);
        out_f.push([f[2], c, b]);
        out_f.push([a, b, c]);
    }
    (out_v, out_f)
}

/// Canonical hemisphere test: `z > 0`, falling back to `x` then `y` on the
/// equator so exactly one of each antipodal pair is selected.
#[inline]
fn is_positive(v: [f64; 3]) -> bool {
    const E: f64 = 1e-9;
    if v[2] > E {
        return true;
    }
    if v[2] < -E {
        return false;
    }
    if v[0] > E {
        return true;
    }
    if v[0] < -E {
        return false;
    }
    v[1] > 0.0
}

fn reorder_hemisphere_first(verts: Vec<[f64; 3]>, faces: Vec<[u32; 3]>) -> IcoSphere {
    let n = verts.len();
    debug_assert!(n % 2 == 0, "icosphere vertex count must be even");
    let n_hemi = n / 2;

    // Antipodal partner by quantized-coordinate lookup. Quantizing to 1e-6 is
    // safe because subdivision midpoints are symmetric by construction, so the
    // two members of a pair agree to far better than that.
    use std::collections::HashMap;
    let key = |v: [f64; 3]| -> (i64, i64, i64) {
        (
            (v[0] * 1e6).round() as i64,
            (v[1] * 1e6).round() as i64,
            (v[2] * 1e6).round() as i64,
        )
    };
    let mut lookup: HashMap<(i64, i64, i64), u32> = HashMap::with_capacity(n);
    for (i, v) in verts.iter().enumerate() {
        lookup.insert(key(*v), i as u32);
    }

    let mut hemi: Vec<u32> = Vec::with_capacity(n_hemi);
    for (i, v) in verts.iter().enumerate() {
        if is_positive(*v) {
            hemi.push(i as u32);
        }
    }
    assert_eq!(
        hemi.len(),
        n_hemi,
        "hemisphere selection must split the sphere exactly in half"
    );

    // old index -> new index
    let mut remap = vec![u32::MAX; n];
    let mut out_v: Vec<[f32; 3]> = Vec::with_capacity(n);
    for (h, &old) in hemi.iter().enumerate() {
        remap[old as usize] = h as u32;
        let v = verts[old as usize];
        out_v.push([v[0] as f32, v[1] as f32, v[2] as f32]);
    }
    for (h, &old) in hemi.iter().enumerate() {
        let v = verts[old as usize];
        let anti = *lookup
            .get(&key([-v[0], -v[1], -v[2]]))
            .expect("every vertex must have an antipodal partner");
        remap[anti as usize] = (n_hemi + h) as u32;
        let a = verts[anti as usize];
        out_v.push([a[0] as f32, a[1] as f32, a[2] as f32]);
    }
    debug_assert!(remap.iter().all(|&r| r != u32::MAX));

    let out_f = faces
        .iter()
        .map(|f| {
            [
                remap[f[0] as usize],
                remap[f[1] as usize],
                remap[f[2] as usize],
            ]
        })
        .collect();

    IcoSphere { vertices: out_v, faces: out_f }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Euler's formula for a subdivided icosahedron, and the specific counts
    /// that make levels 3 and 4 match mrtrix3's tesselation_321 / _1281.
    #[test]
    fn vertex_and_face_counts_match_the_formula() {
        for level in 0..=4 {
            let s = icosphere(level);
            let expect_v = 10 * 4usize.pow(level as u32) + 2;
            let expect_f = 20 * 4usize.pow(level as u32);
            assert_eq!(s.vertices.len(), expect_v, "level {level} vertices");
            assert_eq!(s.faces.len(), expect_f, "level {level} faces");
        }
        assert_eq!(icosphere(3).hemisphere().len(), 321, "mrtrix tesselation_321");
        assert_eq!(icosphere(4).hemisphere().len(), 1281, "mrtrix tesselation_1281");
    }

    #[test]
    fn all_vertices_are_unit_length() {
        let s = icosphere(4);
        for v in &s.vertices {
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((n - 1.0).abs() < 1e-5, "non-unit vertex {v:?} (|v| = {n})");
        }
    }

    /// The layout `hemisphere_adjacency` and dsistudio_odf8 both depend on:
    /// vertex `i` and `i + n/2` must be exact antipodes.
    #[test]
    fn second_half_is_the_antipode_of_the_first() {
        let s = icosphere(3);
        let h = s.vertices.len() / 2;
        for i in 0..h {
            let (a, b) = (s.vertices[i], s.vertices[i + h]);
            for c in 0..3 {
                assert!(
                    (a[c] + b[c]).abs() < 1e-5,
                    "vertex {i} and {} are not antipodal: {a:?} vs {b:?}",
                    i + h
                );
            }
        }
    }

    /// Every face must index real vertices, and no face may be degenerate.
    #[test]
    fn faces_are_valid_after_reordering() {
        let s = icosphere(3);
        let n = s.vertices.len() as u32;
        for f in &s.faces {
            assert!(f.iter().all(|&x| x < n), "face out of range: {f:?}");
            assert!(
                f[0] != f[1] && f[1] != f[2] && f[0] != f[2],
                "degenerate face: {f:?}"
            );
        }
    }

    /// Subdivision must not leave duplicate vertices — the midpoint cache is
    /// what keeps the count exact, and a cache miss would silently inflate it.
    #[test]
    fn no_duplicate_vertices() {
        let s = icosphere(3);
        let mut seen = std::collections::HashSet::new();
        for v in &s.vertices {
            let k = (
                (v[0] * 1e5).round() as i64,
                (v[1] * 1e5).round() as i64,
                (v[2] * 1e5).round() as i64,
            );
            assert!(seen.insert(k), "duplicate vertex {v:?}");
        }
    }

    /// The tessellation should be near-uniform: this is what justifies
    /// `IntegrationWeights::uniform` as a reasonable fallback. Check that the
    /// nearest-neighbour spacing does not vary wildly.
    #[test]
    fn tessellation_is_near_uniform() {
        let s = icosphere(3);
        let v = &s.vertices;
        let mut nn = Vec::with_capacity(v.len());
        for (i, a) in v.iter().enumerate() {
            let mut best = f32::INFINITY;
            for (j, b) in v.iter().enumerate() {
                if i == j {
                    continue;
                }
                let d = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
                if d < best {
                    best = d;
                }
            }
            nn.push(best.sqrt());
        }
        let (lo, hi) = nn.iter().fold((f32::MAX, 0.0f32), |(l, h), &x| (l.min(x), h.max(x)));
        assert!(
            hi / lo < 1.35,
            "nearest-neighbour spacing too uneven: {lo} to {hi} (ratio {})",
            hi / lo
        );
    }
}
