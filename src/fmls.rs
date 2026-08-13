// SPDX-License-Identifier: MPL-2.0
//
// This file is a derivative work ported from MRtrix3 (https://www.mrtrix.org/),
// specifically the FOD maxima-and-lobe segmentation (FMLS) algorithm in
// `src/dwi/fmls.h` and `src/dwi/fmls.cpp`, as used by `cmd/fod2fixel.cpp`.
//
// Ported elements include: the descending-|amplitude| watershed over an
// adjacency graph with new-lobe / join / bridge-merge handling and the
// retrospective-assignment pass; the least-squares construction of spherical
// quadrature weights (`IntegrationWeights`); the negative-lobe, integral and
// peak-value rejection criteria; the default thresholds
// (`FMLS_INTEGRAL_THRESHOLD_DEFAULT`, `FMLS_PEAK_VALUE_THRESHOLD_DEFAULT`,
// `FMLS_MERGE_RATIO_BRIDGE_TO_PEAK_DEFAULT`); the antipodal-wrapped adjacency
// of `DWI::Directions::Set::initialise_adjacency`; and the lobe-to-fixel
// conversion of `fod2fixel`'s `Primitive_FOD_lobes` (direction from the
// amplitude-weighted mean, or the maximal peak under `-dirpeak`; `afd` from the
// lobe integral; `amplitude` from its maximal peak; `-maxnum` truncation).
//
// Deviations from the original are documented in the module docs below.
//
// Original copyright: Copyright (c) 2008-2026 the MRtrix3 contributors.
//
// This file is made available under the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/. A copy of the license is also
// included in the odx-rs source tree at `LICENSE-MRTRIX`.

//! FMLS lobe segmentation and apparent fibre density (AFD).
//!
//! A port of mrtrix3's "fibre orientation distribution — maxima and lobe
//! segmentation" (`src/dwi/fmls.{h,cpp}`, used by `fod2fixel`). Given an FOD
//! sampled on a sphere, it partitions the sphere into lobes by watershed from
//! the amplitude maxima and integrates each lobe to yield AFD.
//!
//! # Algorithm (matching mrtrix3)
//!
//! 1. Sample the FOD at every sphere vertex.
//! 2. Visit vertices in order of **descending |amplitude|**.
//! 3. For each, look at already-assigned mesh neighbours of the same sign:
//!    - none → seed a new lobe (this vertex is a maximum);
//!    - one → join it;
//!    - several → this vertex bridges lobes. Merge them if
//!      `|value| / max_peak_of_last_adjacent > merge_ratio`, else defer to a
//!      retrospective pass that assigns it to the *first* adjacent lobe.
//! 4. Replay deferred assignments.
//! 5. Discard negative lobes and lobes whose integral is below
//!    `integral_threshold`; discard lobes whose peak is below
//!    `peak_value_threshold`.
//!
//! AFD for a lobe is `Σ |amplitude · w|` over its vertices, with `w` the
//! quadrature weights from [`IntegrationWeights`].
//!
//! # Deviations from mrtrix3
//!
//! All deliberate, and each is either a robustification or an optimisation
//! that leaves the result unchanged:
//!
//! - **Adjacency comes from the mesh**, via face-derived neighbours, rather
//!   than mrtrix3's `FastLookupSet` angular-threshold construction. Both
//!   describe the same neighbourhoods on a regular sphere tessellation; the
//!   mesh version is exact and needs no tuning parameter. It must be built
//!   with [`hemisphere_adjacency`], which wraps across the antipodal rim as
//!   mrtrix3 does — plain hemisphere mesh neighbours sever any lobe centred on
//!   the boundary, halving its AFD and doubling its fixel count.
//! - **Non-finite amplitudes are dropped** rather than propagated. mrtrix3
//!   guards only `in[0]`; a NaN deeper in the coefficient vector reaches its
//!   sort comparator, where NaN comparisons are inconsistent and can corrupt
//!   the ordering.
//! - **Sorting is by a total order on the bit pattern**, not `<` on floats, so
//!   the traversal is deterministic and cannot be perturbed by NaN.
//! - **Weights are computed once** and shared across all voxels.
//!
//! Newton refinement of peak directions is intentionally *not* performed here:
//! odx's [`crate::peak_finder::SpherePeakFinder::refine_with_sh`] already does
//! it, and keeping it separate lets callers pay for it only when they want
//! refined directions rather than integrals.

use nalgebra::{DMatrix, DVector};

/// Default lower bound on a lobe's integral. mrtrix3 uses 0 (off) — "tough to
/// get a good number" — and so do we.
pub const DEFAULT_INTEGRAL_THRESHOLD: f32 = 0.0;
/// Default lower bound on a lobe's peak amplitude (mrtrix3's default).
pub const DEFAULT_PEAK_VALUE_THRESHOLD: f32 = 0.1;
/// Default bridge-to-peak ratio above which adjacent lobes merge.
///
/// mrtrix3's default of 1.0 means merging effectively never happens: the
/// bridging vertex is visited in descending-amplitude order, so its value can
/// never exceed the peak it would merge into. Lobes containing multiple
/// discrete peaks are therefore kept separate by default.
pub const DEFAULT_MERGE_RATIO: f32 = 1.0;

/// Tunables for [`Fmls::segment`].
#[derive(Clone, Copy, Debug)]
pub struct FmlsConfig {
    pub integral_threshold: f32,
    pub peak_value_threshold: f32,
    pub merge_ratio: f32,
}

impl Default for FmlsConfig {
    fn default() -> Self {
        Self {
            integral_threshold: DEFAULT_INTEGRAL_THRESHOLD,
            peak_value_threshold: DEFAULT_PEAK_VALUE_THRESHOLD,
            merge_ratio: DEFAULT_MERGE_RATIO,
        }
    }
}

/// One segmented lobe.
#[derive(Clone, Debug)]
pub struct Lobe {
    /// Quadrature integral of the FOD over this lobe — the AFD.
    pub integral: f32,
    /// Largest amplitude within the lobe.
    pub peak_value: f32,
    /// Vertex index at which `peak_value` occurs.
    pub peak_index: usize,
    /// Amplitude-weighted mean direction, sign-corrected for antipodal
    /// symmetry and normalized to unit length.
    pub mean_dir: [f32; 3],
    /// Sphere vertex indices belonging to this lobe.
    pub vertices: Vec<usize>,
}

/// Quadrature weights for integrating a band-limited function sampled on a
/// fixed direction set.
///
/// Solves `A w = b`, where `A[i][d] = Y_i(dir_d)` and `b = [√(4π), 0, …]`, so
/// that `Σ_d f(dir_d) · w_d` reproduces `∫ f dΩ` exactly for any `f` band-
/// limited to the calibration order. Equivalently, the weights sum to 4π and
/// annihilate every non-constant basis function.
///
/// This is mrtrix3's `IntegrationWeights`, and it is why AFD is a true
/// spherical integral rather than a vertex-count proxy.
pub struct IntegrationWeights {
    data: Vec<f32>,
}

impl IntegrationWeights {
    /// Build from a direction set and a basis evaluator.
    ///
    /// `basis(dirs) -> (ncoeffs, row-major [ndirs × ncoeffs])` must evaluate
    /// the *same* SH basis the FOD coefficients are expressed in, at the
    /// calibration order. mrtrix3 calibrates at `LforN(ndirs) + 2`.
    pub fn new(ndirs: usize, ncoeffs: usize, sh2a_row_major: &[f32]) -> Option<Self> {
        if ndirs == 0 || ncoeffs == 0 || sh2a_row_major.len() != ndirs * ncoeffs {
            return None;
        }
        // A is (ncoeffs × ndirs): one row per basis function.
        let a = DMatrix::<f64>::from_fn(ncoeffs, ndirs, |i, d| {
            sh2a_row_major[d * ncoeffs + i] as f64
        });
        let mut b = DVector::<f64>::zeros(ncoeffs);
        // Y_00 = 1/(2√π), so Σ_d w_d = 4π falls out of this single entry.
        b[0] = 2.0 * std::f64::consts::PI.sqrt();

        // Least-squares via QR on the (usually overdetermined) system.
        let w = a.clone().svd(true, true).solve(&b, 1e-12).ok()?;
        let data: Vec<f32> = w.iter().map(|v| *v as f32).collect();
        if data.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(Self { data })
    }

    /// Uniform weights `4π / ndirs`.
    ///
    /// Correct only for an exactly equal-area direction set. Provided as an
    /// explicit fallback so a caller that cannot build the basis degrades to a
    /// documented approximation rather than to silently wrong integrals.
    pub fn uniform(ndirs: usize) -> Self {
        let w = (4.0 * std::f32::consts::PI) / ndirs.max(1) as f32;
        Self { data: vec![w; ndirs] }
    }

    #[inline]
    pub fn get(&self, i: usize) -> f32 {
        self.data[i]
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Sum of weights; 4π for a correctly calibrated set.
    pub fn total(&self) -> f32 {
        self.data.iter().sum()
    }
}

/// Build hemisphere adjacency that wraps across the antipodal rim.
///
/// An FOD is antipodally symmetric, so it is sampled on a hemisphere — but a
/// lobe near the rim continues onto the far side, and plain mesh adjacency on
/// the hemisphere alone severs it there, splitting one lobe into two. That is
/// not a cosmetic issue: it double-counts fixels and halves their AFD.
///
/// mrtrix3 avoids this by constructing its adjacency *after* "generate
/// antipodal vertices" (`DWI::Directions::Set::initialise_adjacency`), so
/// hemisphere direction `i` neighbours `j` if either `+j` or `−j` adjoins `+i`
/// on the full sphere. This reproduces that: adjacency is taken from the full
/// face list and every full-sphere index is folded onto its hemisphere
/// representative.
///
/// `full_vertices` must contain each hemisphere direction and its antipode;
/// the hemisphere is taken to be the first `full_vertices.len() / 2` entries.
/// Antipodal partners are found by matching `−v` geometrically rather than by
/// assuming an index convention, so this holds for any vertex ordering.
pub fn hemisphere_adjacency(
    full_vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
) -> Vec<Vec<usize>> {
    let n_full = full_vertices.len();
    let n_hemi = n_full / 2;
    if n_hemi == 0 {
        return Vec::new();
    }

    // full index -> hemisphere representative.
    let mut rep = vec![0usize; n_full];
    for i in n_hemi..n_full {
        let v = full_vertices[i];
        let target = [-v[0], -v[1], -v[2]];
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for (h, hv) in full_vertices.iter().enumerate().take(n_hemi) {
            let d = (hv[0] - target[0]).powi(2)
                + (hv[1] - target[1]).powi(2)
                + (hv[2] - target[2]).powi(2);
            if d < best_d {
                best_d = d;
                best = h;
            }
        }
        rep[i] = best;
    }
    for (i, r) in rep.iter_mut().enumerate().take(n_hemi) {
        *r = i;
    }

    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_hemi];
    let link = |a: usize, b: usize, adj: &mut Vec<Vec<usize>>| {
        if a != b && !adj[a].contains(&b) {
            adj[a].push(b);
        }
    };
    for f in faces {
        let t = [f[0] as usize, f[1] as usize, f[2] as usize];
        if t.iter().any(|&x| x >= n_full) {
            continue;
        }
        let r = [rep[t[0]], rep[t[1]], rep[t[2]]];
        for a in 0..3 {
            for b in 0..3 {
                if a != b {
                    link(r[a], r[b], &mut adj);
                }
            }
        }
    }
    adj
}

/// Segmenter bound to a sphere tessellation.
pub struct Fmls<'a> {
    vertices: &'a [[f32; 3]],
    neighbors: &'a [Vec<usize>],
    weights: &'a IntegrationWeights,
    config: FmlsConfig,
    /// Scratch reused across voxels: `(key, index)` traversal order.
    order: Vec<(u32, u32)>,
    /// Scratch: vertex → lobe id, `u32::MAX` when unassigned.
    assign: Vec<u32>,
}

impl<'a> Fmls<'a> {
    pub fn new(
        vertices: &'a [[f32; 3]],
        neighbors: &'a [Vec<usize>],
        weights: &'a IntegrationWeights,
        config: FmlsConfig,
    ) -> Self {
        Self {
            vertices,
            neighbors,
            weights,
            config,
            order: Vec::with_capacity(vertices.len()),
            assign: vec![u32::MAX; vertices.len()],
        }
    }

    /// Segment one FOD, given its amplitudes at every sphere vertex.
    ///
    /// Returns lobes sorted by integral descending. Allocations are reused
    /// between calls, so this is cheap to invoke per voxel.
    pub fn segment(&mut self, amplitudes: &[f32]) -> Vec<Lobe> {
        let n = self.vertices.len();
        if amplitudes.len() != n || n == 0 {
            return Vec::new();
        }

        // Traversal order: descending |amplitude|. Sorting on the bit pattern
        // of |v| gives a total order (floats' `<` is only a partial order), so
        // the result is deterministic and NaN cannot corrupt the sort.
        self.order.clear();
        for (i, &v) in amplitudes.iter().enumerate() {
            if v.is_finite() && v != 0.0 {
                self.order.push((v.abs().to_bits(), i as u32));
            }
        }
        if self.order.is_empty() {
            return Vec::new();
        }
        self.order.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

        self.assign.clear();
        self.assign.resize(n, u32::MAX);

        // Lobe accumulators, parallel arrays indexed by lobe id.
        let mut verts: Vec<Vec<usize>> = Vec::new();
        let mut integral: Vec<f64> = Vec::new();
        let mut peak_val: Vec<f32> = Vec::new();
        let mut peak_idx: Vec<usize> = Vec::new();
        let mut mean_dir: Vec<[f64; 3]> = Vec::new();
        let mut negative: Vec<bool> = Vec::new();
        // Union-find over lobe ids so a merge is O(α(n)) and never needs the
        // index-rewriting mrtrix3 does after erasing from a vector.
        let mut parent: Vec<u32> = Vec::new();
        let mut deferred: Vec<(usize, u32)> = Vec::new();

        fn find(parent: &mut [u32], mut x: u32) -> u32 {
            while parent[x as usize] != x {
                let g = parent[parent[x as usize] as usize];
                parent[x as usize] = g;
                x = g;
            }
            x
        }

        for oi in 0..self.order.len() {
            let idx = self.order[oi].1 as usize;
            let val = amplitudes[idx];
            let neg = val < 0.0;

            // Adjacent, already-assigned lobes of the same sign.
            let mut adj: Vec<u32> = Vec::new();
            for &nb in &self.neighbors[idx] {
                let a = self.assign[nb];
                if a == u32::MAX {
                    continue;
                }
                let root = find(&mut parent, a);
                if negative[root as usize] == neg && !adj.contains(&root) {
                    adj.push(root);
                }
            }

            if adj.is_empty() {
                let id = verts.len() as u32;
                parent.push(id);
                verts.push(Vec::new());
                integral.push(0.0);
                peak_val.push(0.0);
                peak_idx.push(idx);
                mean_dir.push([0.0; 3]);
                negative.push(neg);
                add_vertex(
                    id as usize, idx, val, self.weights.get(idx), self.vertices,
                    &mut verts, &mut integral, &mut peak_val, &mut peak_idx, &mut mean_dir,
                );
                self.assign[idx] = id;
            } else if adj.len() == 1 {
                let id = adj[0];
                add_vertex(
                    id as usize, idx, val, self.weights.get(idx), self.vertices,
                    &mut verts, &mut integral, &mut peak_val, &mut peak_idx, &mut mean_dir,
                );
                self.assign[idx] = id;
            } else {
                // Bridging vertex. mrtrix3 compares against the *last* adjacent
                // lobe's peak; with the default ratio of 1.0 this is never true,
                // because `val` was visited after that peak in descending order.
                let last_peak = peak_val[adj[adj.len() - 1] as usize];
                let bridge = last_peak > 0.0 && (val.abs() / last_peak) > self.config.merge_ratio;
                if bridge {
                    adj.sort_unstable();
                    let keep = adj[0];
                    for &other in &adj[1..] {
                        merge_into(
                            keep as usize, other as usize,
                            &mut verts, &mut integral, &mut peak_val, &mut peak_idx, &mut mean_dir,
                        );
                        parent[other as usize] = keep;
                    }
                    add_vertex(
                        keep as usize, idx, val, self.weights.get(idx), self.vertices,
                        &mut verts, &mut integral, &mut peak_val, &mut peak_idx, &mut mean_dir,
                    );
                    self.assign[idx] = keep;
                } else {
                    deferred.push((idx, adj[0]));
                }
            }
        }

        for (idx, id) in deferred {
            let root = find(&mut parent, id);
            add_vertex(
                root as usize, idx, amplitudes[idx], self.weights.get(idx), self.vertices,
                &mut verts, &mut integral, &mut peak_val, &mut peak_idx, &mut mean_dir,
            );
            self.assign[idx] = root;
        }

        // Emit surviving lobes: positive, above both thresholds, not merged away.
        let mut out = Vec::new();
        for id in 0..verts.len() {
            if parent[id] != id as u32 || negative[id] || verts[id].is_empty() {
                continue;
            }
            let integ = integral[id] as f32;
            if integ < self.config.integral_threshold {
                continue;
            }
            if peak_val[id] < self.config.peak_value_threshold {
                continue;
            }
            let m = mean_dir[id];
            let norm = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt().max(1e-12);
            out.push(Lobe {
                integral: integ,
                peak_value: peak_val[id],
                peak_index: peak_idx[id],
                mean_dir: [
                    (m[0] / norm) as f32,
                    (m[1] / norm) as f32,
                    (m[2] / norm) as f32,
                ],
                vertices: std::mem::take(&mut verts[id]),
            });
        }
        out.sort_unstable_by(|a, b| b.integral.total_cmp(&a.integral));
        out
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn add_vertex(
    id: usize,
    idx: usize,
    val: f32,
    weight: f32,
    vertices: &[[f32; 3]],
    verts: &mut [Vec<usize>],
    integral: &mut [f64],
    peak_val: &mut [f32],
    peak_idx: &mut [usize],
    mean_dir: &mut [[f64; 3]],
) {
    verts[id].push(idx);
    integral[id] += (val as f64 * weight as f64).abs();
    if val.abs() > peak_val[id] {
        peak_val[id] = val.abs();
        peak_idx[id] = idx;
    }
    // Antipodal sign correction: an FOD is symmetric, so a vertex on the
    // opposite side of the lobe must contribute with a flipped sign or the
    // mean direction cancels itself out.
    let d = vertices[idx];
    let m = &mut mean_dir[id];
    let dot = m[0] * d[0] as f64 + m[1] * d[1] as f64 + m[2] * d[2] as f64;
    let s = if dot > 0.0 { 1.0 } else { -1.0 };
    let scale = s * (val as f64).abs() * weight as f64;
    m[0] += d[0] as f64 * scale;
    m[1] += d[1] as f64 * scale;
    m[2] += d[2] as f64 * scale;
}

#[allow(clippy::too_many_arguments)]
fn merge_into(
    keep: usize,
    other: usize,
    verts: &mut [Vec<usize>],
    integral: &mut [f64],
    peak_val: &mut [f32],
    peak_idx: &mut [usize],
    mean_dir: &mut [[f64; 3]],
) {
    let moved = std::mem::take(&mut verts[other]);
    verts[keep].extend_from_slice(&moved);
    integral[keep] += integral[other];
    integral[other] = 0.0;
    if peak_val[other] > peak_val[keep] {
        peak_val[keep] = peak_val[other];
        peak_idx[keep] = peak_idx[other];
    }
    let (a, b) = (mean_dir[keep], mean_dir[other]);
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let s = if dot > 0.0 { 1.0 } else { -1.0 };
    mean_dir[keep] = [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]];
}

// ---------- dataset-level: build a fixel ODX from an FOD ODX ----------

/// Options for [`afd_dataset`].
#[derive(Clone, Debug)]
pub struct AfdOptions {
    /// Icosphere subdivision level. 3 = 321 hemisphere directions
    /// (mrtrix3 `tesselation_321`), 4 = 1281 (`fod2fixel`'s default).
    pub ico_level: usize,
    /// FMLS thresholds.
    pub fmls: FmlsConfig,
    /// Keep at most this many lobes per voxel, largest-AFD first
    /// (`fod2fixel -maxnum`). `None` keeps all.
    pub max_per_voxel: Option<usize>,
    /// Use the lobe's maximal-peak direction rather than its amplitude-weighted
    /// mean direction (`fod2fixel -dirpeak`). Default false, matching mrtrix3.
    pub dir_from_peak: bool,
    /// Carry the input SH through to the output, so the result still renders as
    /// glyphs. `fod2fixel` writes a bare fixel directory with no FOD; keeping
    /// the SH costs space but keeps the output viewable.
    pub keep_sh: bool,
}

impl Default for AfdOptions {
    fn default() -> Self {
        Self {
            ico_level: 4,
            fmls: FmlsConfig::default(),
            max_per_voxel: None,
            dir_from_peak: false,
            keep_sh: true,
        }
    }
}

/// Summary of an [`afd_dataset`] run.
#[derive(Clone, Debug, Default)]
pub struct AfdReport {
    pub n_voxels: usize,
    pub n_voxels_with_fixels: usize,
    pub n_fixels: usize,
    pub n_directions: usize,
}

/// Segment every voxel's FOD and emit a new ODX whose **fixels are the lobes**.
///
/// This follows `fod2fixel`: a fixel is created *from* each surviving lobe,
/// taking its direction, `afd` (the lobe integral) and `amplitude` (its maximal
/// peak) from the same object. It deliberately does **not** attach AFD onto the
/// input's existing fixels — those come from peak finding, whose directions are
/// peak directions rather than lobe mean directions, and whose cardinality is
/// much larger. The result is a derived dataset, exactly as `fod2fixel` writes
/// a new fixel directory rather than annotating one.
pub fn afd_dataset(
    input: &crate::odx_file::OdxDataset,
    opts: &AfdOptions,
) -> crate::error::Result<(crate::odx_file::OdxDataset, AfdReport)> {
    use crate::dtype::DType;
    use crate::error::OdxError;
    use crate::mrtrix_sh::RowSamplePlan;
    use crate::stream::OdxBuilder;

    let sh_name = input
        .sh_names()
        .first()
        .map(|s| s.to_string())
        .ok_or_else(|| OdxError::Argument("input ODX has no SH array; AFD needs an FOD".into()))?;
    let view = input.sh::<f32>(&sh_name)?;
    let ncoeffs = view.ncols();
    let nvox = view.nrows();

    let sphere = crate::icosphere::icosphere(opts.ico_level);
    let verts = sphere.hemisphere().to_vec();
    let adj = hemisphere_adjacency(&sphere.vertices, &sphere.faces);
    let weights = IntegrationWeights::uniform(verts.len());
    let plan = RowSamplePlan::for_sh_rows_nonnegative(&verts, ncoeffs)?;
    let mut fmls = Fmls::new(&verts, &adj, &weights, opts.fmls);

    let header = input.header();
    let mut builder = OdxBuilder::new(
        header.voxel_to_rasmm,
        header.dimensions,
        input.mask().to_vec(),
    );

    let mut afd: Vec<f32> = Vec::new();
    let mut amp: Vec<f32> = Vec::new();
    let mut amps = vec![0.0f32; plan.ndir()];
    let mut dirs_scratch: Vec<[f32; 3]> = Vec::new();
    let mut report = AfdReport { n_voxels: nvox, n_directions: verts.len(), ..Default::default() };

    for v in 0..nvox {
        plan.apply_row_into(view.row(v), &mut amps);
        let mut lobes = fmls.segment(&amps);
        if let Some(n) = opts.max_per_voxel {
            lobes.truncate(n); // already sorted by integral descending
        }
        dirs_scratch.clear();
        for l in &lobes {
            let d = if opts.dir_from_peak { verts[l.peak_index] } else { l.mean_dir };
            dirs_scratch.push(d);
            afd.push(l.integral);
            amp.push(l.peak_value);
        }
        if !dirs_scratch.is_empty() {
            report.n_voxels_with_fixels += 1;
        }
        report.n_fixels += dirs_scratch.len();
        builder.push_voxel_peaks(&dirs_scratch);
    }

    if opts.keep_sh {
        if let (Some(order), Some(basis)) = (header.sh_order, header.sh_basis.as_deref()) {
            builder.set_sh_info(order, basis.to_string());
        }
        if let Some(full) = header.sh_full_basis {
            builder.set_sh_full_basis(full);
        }
        if let Some(legacy) = header.sh_legacy {
            builder.set_sh_legacy(legacy);
        }
        let flat: Vec<f32> = view.as_flat_slice().to_vec();
        builder.set_sh_data(
            &sh_name,
            crate::mmap_backing::vec_into_bytes(flat),
            ncoeffs,
            DType::Float32,
        );
        if let Some(rep) = header.canonical_dense_representation.clone() {
            builder.set_canonical_dense_representation(rep);
        }
    }

    builder.set_dpf_data(
        "afd",
        crate::mmap_backing::vec_into_bytes(afd),
        1,
        DType::Float32,
    );
    builder.set_dpf_data(
        "amplitude",
        crate::mmap_backing::vec_into_bytes(amp),
        1,
        DType::Float32,
    );
    for (k, val) in &header.extra {
        builder.set_extra_value(k, val.clone());
    }

    Ok((builder.finalize()?, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::dsistudio_odf8;
    use crate::peak_finder::{PeakFinderConfig, SpherePeakFinder};

    /// Build the sphere + mesh adjacency the segmenter needs.
    fn sphere() -> (Vec<[f32; 3]>, Vec<Vec<usize>>) {
        let finder = SpherePeakFinder::new(
            dsistudio_odf8::hemisphere_vertices_ras(),
            dsistudio_odf8::faces(),
            PeakFinderConfig::default(),
        );
        let v = finder.vertices().to_vec();
        // Antipodal-wrapped adjacency, not the plain hemisphere mesh — see
        // `hemisphere_adjacency`. Using the latter splits rim-straddling lobes.
        let n = hemisphere_adjacency(dsistudio_odf8::full_vertices_ras(), dsistudio_odf8::faces());
        assert_eq!(n.len(), v.len());
        (v, n)
    }

    /// The load-bearing property: quadrature weights must sum to 4π, the total
    /// solid angle. If they do not, every AFD is off by a constant factor and
    /// nothing downstream would notice.
    #[test]
    fn uniform_weights_sum_to_four_pi() {
        let w = IntegrationWeights::uniform(321);
        assert!(
            (w.total() - 4.0 * std::f32::consts::PI).abs() < 1e-3,
            "got {}",
            w.total()
        );
    }

    /// A constant FOD integrates to 4π × amplitude. This is the calibration
    /// mrtrix3's IntegrationWeights is derived from, so it must hold exactly.
    #[test]
    fn constant_fod_integrates_to_four_pi() {
        let (v, nb) = sphere();
        let w = IntegrationWeights::uniform(v.len());
        let mut f = Fmls::new(&v, &nb, &w, FmlsConfig { peak_value_threshold: 0.0, ..Default::default() });
        let amps = vec![1.0f32; v.len()];
        let lobes = f.segment(&amps);
        // A constant function has no maxima to separate, so it forms one lobe
        // covering the whole (hemi)sphere.
        assert_eq!(lobes.len(), 1, "constant FOD should give exactly one lobe");
        assert_eq!(lobes[0].vertices.len(), v.len());
        let total: f32 = w.total();
        assert!(
            (lobes[0].integral - total).abs() < 1e-2 * total,
            "integral {} vs expected {}",
            lobes[0].integral,
            total
        );
    }

    /// Two well-separated maxima must yield two lobes, and the integrals must
    /// partition the total rather than double-count it.
    #[test]
    fn two_separated_peaks_give_two_lobes() {
        let (v, nb) = sphere();
        let w = IntegrationWeights::uniform(v.len());
        // Two lobes along x and y: amplitude falls off as a power of |cos|.
        let amps: Vec<f32> = v
            .iter()
            .map(|d| {
                let ax = d[0].abs().powi(8);
                let ay = d[1].abs().powi(8);
                ax + ay
            })
            .collect();
        let mut f = Fmls::new(&v, &nb, &w, FmlsConfig { peak_value_threshold: 0.05, ..Default::default() });
        let lobes = f.segment(&amps);
        assert_eq!(lobes.len(), 2, "expected 2 lobes, got {}", lobes.len());
        // Each lobe's mean direction should align with x or y.
        let mut saw_x = false;
        let mut saw_y = false;
        for l in &lobes {
            if l.mean_dir[0].abs() > 0.9 {
                saw_x = true;
            }
            if l.mean_dir[1].abs() > 0.9 {
                saw_y = true;
            }
        }
        assert!(saw_x && saw_y, "mean dirs not aligned to x and y: {lobes:?}");
        // Vertices partition: no vertex in two lobes.
        let mut seen = vec![false; v.len()];
        for l in &lobes {
            for &i in &l.vertices {
                assert!(!seen[i], "vertex {i} assigned to two lobes");
                seen[i] = true;
            }
        }
    }

    /// A single lobe's integral must scale linearly with FOD amplitude — the
    /// property that makes AFD quantitative in the first place.
    #[test]
    fn integral_is_linear_in_amplitude() {
        let (v, nb) = sphere();
        let w = IntegrationWeights::uniform(v.len());
        let base: Vec<f32> = v.iter().map(|d| d[2].abs().powi(8)).collect();
        let cfg = FmlsConfig { peak_value_threshold: 0.0, ..Default::default() };
        let mut f = Fmls::new(&v, &nb, &w, cfg);
        let a = f.segment(&base)[0].integral;
        let scaled: Vec<f32> = base.iter().map(|x| x * 3.0).collect();
        let b = f.segment(&scaled)[0].integral;
        assert!(
            (b - 3.0 * a).abs() < 1e-3 * b.abs().max(1.0),
            "expected 3x scaling: {a} -> {b}"
        );
    }

    /// Peak-value thresholding must remove small lobes, matching mrtrix3's
    /// `peak_value_threshold` (default 0.1).
    #[test]
    fn peak_threshold_prunes_small_lobes() {
        let (v, nb) = sphere();
        let w = IntegrationWeights::uniform(v.len());
        // Strong lobe on z, weak lobe on x.
        let amps: Vec<f32> = v
            .iter()
            .map(|d| d[2].abs().powi(8) + 0.02 * d[0].abs().powi(8))
            .collect();
        let permissive = Fmls::new(&v, &nb, &w, FmlsConfig { peak_value_threshold: 0.0, ..Default::default() })
            .segment(&amps)
            .len();
        let strict = Fmls::new(&v, &nb, &w, FmlsConfig { peak_value_threshold: 0.1, ..Default::default() })
            .segment(&amps)
            .len();
        assert!(permissive >= 2, "permissive should keep the weak lobe, got {permissive}");
        assert_eq!(strict, 1, "0.1 threshold should prune the 0.02 lobe, got {strict}");
    }

    /// Robustification over mrtrix3: NaN amplitudes must not corrupt the sort
    /// or leak into an integral.
    #[test]
    fn non_finite_amplitudes_are_dropped() {
        let (v, nb) = sphere();
        let w = IntegrationWeights::uniform(v.len());
        let mut amps: Vec<f32> = v.iter().map(|d| d[2].abs().powi(8)).collect();
        let clean = Fmls::new(&v, &nb, &w, FmlsConfig { peak_value_threshold: 0.0, ..Default::default() })
            .segment(&amps)[0]
            .integral;
        amps[7] = f32::NAN;
        amps[19] = f32::INFINITY;
        let lobes = Fmls::new(&v, &nb, &w, FmlsConfig { peak_value_threshold: 0.0, ..Default::default() })
            .segment(&amps);
        assert!(!lobes.is_empty(), "should still segment");
        assert!(lobes[0].integral.is_finite(), "integral must stay finite");
        assert!(
            lobes[0].integral <= clean * 1.001,
            "dropping 2 vertices should not increase the integral"
        );
    }

    /// An all-negative FOD yields no lobes (mrtrix3 discards negative lobes).
    #[test]
    fn negative_fod_yields_no_lobes() {
        let (v, nb) = sphere();
        let w = IntegrationWeights::uniform(v.len());
        let amps: Vec<f32> = v.iter().map(|d| -d[2].abs().powi(8)).collect();
        let lobes = Fmls::new(&v, &nb, &w, FmlsConfig { peak_value_threshold: 0.0, ..Default::default() })
            .segment(&amps);
        assert!(lobes.is_empty(), "negative lobes must be discarded, got {}", lobes.len());
    }

    /// Determinism: repeated segmentation of the same input is bit-identical.
    #[test]
    fn segmentation_is_deterministic() {
        let (v, nb) = sphere();
        let w = IntegrationWeights::uniform(v.len());
        let amps: Vec<f32> = v
            .iter()
            .map(|d| d[2].abs().powi(6) + 0.5 * d[0].abs().powi(6))
            .collect();
        let cfg = FmlsConfig::default();
        let a = Fmls::new(&v, &nb, &w, cfg).segment(&amps);
        let b = Fmls::new(&v, &nb, &w, cfg).segment(&amps);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.integral.to_bits(), y.integral.to_bits());
            assert_eq!(x.vertices, y.vertices);
        }
    }

    /// Regression test for the antipodal-rim bug.
    ///
    /// A single fibre along x has its lobe centred on the hemisphere boundary,
    /// so plain hemisphere mesh adjacency severs it into two pieces — halving
    /// the AFD and doubling the fixel count. Antipodal-wrapped adjacency must
    /// keep it whole. This is the concrete failure that plain `SpherePeakFinder`
    /// neighbours produce, and it would be invisible in any test using only a
    /// z-aligned lobe.
    #[test]
    fn rim_straddling_lobe_is_not_split() {
        let v = dsistudio_odf8::hemisphere_vertices_ras().to_vec();
        let w = IntegrationWeights::uniform(v.len());
        let cfg = FmlsConfig { peak_value_threshold: 0.0, ..Default::default() };
        let amps: Vec<f32> = v.iter().map(|d| d[0].abs().powi(8)).collect();

        let wrapped =
            hemisphere_adjacency(dsistudio_odf8::full_vertices_ras(), dsistudio_odf8::faces());
        let got = Fmls::new(&v, &wrapped, &w, cfg).segment(&amps);
        assert_eq!(got.len(), 1, "x-aligned lobe must stay whole, got {}", got.len());

        // And the unwrapped hemisphere mesh must actually exhibit the bug, or
        // this test proves nothing.
        let finder = SpherePeakFinder::new(
            dsistudio_odf8::hemisphere_vertices_ras(),
            dsistudio_odf8::faces(),
            PeakFinderConfig::default(),
        );
        let plain = Fmls::new(&v, finder.neighbors(), &w, cfg).segment(&amps);
        assert!(
            plain.len() > 1,
            "expected the unwrapped mesh to split the rim lobe; if this now \
             passes, the adjacency source changed and the wrap may be redundant"
        );
        // The whole lobe carries the AFD the split pieces divide between them.
        let split_total: f32 = plain.iter().map(|l| l.integral).sum();
        assert!(
            (got[0].integral - split_total).abs() < 1e-3 * split_total,
            "wrapped {} vs split-sum {}",
            got[0].integral,
            split_total
        );
    }
}
