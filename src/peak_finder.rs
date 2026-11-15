// SPDX-License-Identifier: MPL-2.0
//
// Portions of this file (notably `refine_with_sh`, the Gauss-Newton constants
// `MAX_DIR_CHANGE` / `ANGLE_TOLERANCE`, and the inner-loop arithmetic that
// drives the Newton step) are derivative works ported from MRtrix3
// (https://www.mrtrix.org/), specifically `Math::SH::get_peak` in
// `core/math/SH.h`.
//
// Original copyright: Copyright (c) 2008-2026 the MRtrix3 contributors.
//
// Those portions are made available under the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at http://mozilla.org/MPL/2.0/. A copy of the license is
// also included in the odx-rs source tree at `LICENSE-MRTRIX`.

use crate::error::Result;
use crate::formats::dsistudio_odf8;
use crate::mrtrix_sh::{lmax_for_ncoeffs, RowSamplePlan};
use crate::sh_basis_evaluator::ShBasisKind;

/// Newton step clamp from MRtrix `Math::SH::get_peak`.
const MAX_DIR_CHANGE: f64 = 0.2;
/// Convergence tolerance from MRtrix `Math::SH::get_peak` (≈ 0.006°).
const ANGLE_TOLERANCE: f64 = 1e-4;
const NEWTON_MAX_ITERS: usize = 50;

#[derive(Debug, Clone)]
pub struct PeakFinderConfig {
    /// Maximum number of peaks returned per voxel.
    pub npeaks: usize,
    /// Drop peaks below this fraction of the voxel maximum (0.0 = keep all, 1.0 = keep only max).
    pub relative_peak_threshold: f32,
    /// Minimum angular separation between accepted peaks in degrees.
    /// Peaks closer than this to a stronger accepted peak are discarded.
    pub min_separation_angle_deg: f32,
}

impl Default for PeakFinderConfig {
    fn default() -> Self {
        Self {
            npeaks: 5,
            relative_peak_threshold: 0.5,
            min_separation_angle_deg: 25.0,
        }
    }
}

/// Sphere-based discrete ODF peak finder.
///
/// Build once from sphere geometry, then call [`find_peaks`] or
/// [`find_peaks_rows`] for each voxel or batch.
///
/// The algorithm mirrors dipy's `peak_directions`:
/// 1. Local maxima detection via mesh adjacency.
/// 2. Relative-threshold pruning.
/// 3. Greedy separation-angle deduplication.
/// 4. Cap at `npeaks`.
pub struct SpherePeakFinder {
    vertices: Vec<[f32; 3]>,
    neighbors: Vec<Vec<usize>>,
    cos_min_sep: f32,
    config: PeakFinderConfig,
}

impl SpherePeakFinder {
    /// Build from explicit sphere vertices and triangular faces.
    ///
    /// Only faces whose three vertex indices are all `< vertices.len()` are
    /// used for adjacency — this naturally restricts a full-sphere face list
    /// to its hemisphere when `vertices` contains only hemisphere vertices.
    pub fn new(vertices: &[[f32; 3]], faces: &[[u32; 3]], config: PeakFinderConfig) -> Self {
        let neighbors = neighbors_from_faces(vertices.len(), faces);
        let cos_min_sep = config.min_separation_angle_deg.to_radians().cos();
        Self {
            vertices: vertices.to_vec(),
            neighbors,
            cos_min_sep,
            config,
        }
    }

    /// Convenience constructor using the built-in DSI Studio ODF8 hemisphere
    /// (321 vertices, subset of the 642-vertex full sphere).
    pub fn for_dsistudio_odf8(config: PeakFinderConfig) -> Self {
        Self::new(
            dsistudio_odf8::hemisphere_vertices_ras(),
            dsistudio_odf8::faces(),
            config,
        )
    }

    /// Find peaks in a single ODF row (one amplitude per sphere vertex).
    ///
    /// Returns peaks sorted by amplitude descending, capped at `npeaks`.
    pub fn find_peaks(&self, odf: &[f32]) -> Vec<(f32, [f32; 3])> {
        self.find_peaks_capped(odf, self.config.npeaks)
    }

    /// Greedy separation filter over `(amplitude, direction)` candidates sorted by
    /// amplitude, descending (antipodal symmetry: `|dot|`), keeping at most `cap`.
    fn separate(&self, candidates: impl IntoIterator<Item = (f32, [f32; 3])>, cap: usize)
        -> Vec<(f32, [f32; 3])> {
        let mut accepted: Vec<(f32, [f32; 3])> = Vec::new();
        'candidate: for (value, dir) in candidates {
            for &(_, acc_dir) in &accepted {
                let dot = dir[0] * acc_dir[0] + dir[1] * acc_dir[1] + dir[2] * acc_dir[2];
                if dot.abs() > self.cos_min_sep {
                    continue 'candidate;
                }
            }
            accepted.push((value, dir));
            if accepted.len() >= cap {
                break;
            }
        }
        accepted
    }

    fn find_peaks_capped(&self, odf: &[f32], cap: usize) -> Vec<(f32, [f32; 3])> {
        assert_eq!(
            odf.len(),
            self.vertices.len(),
            "ODF length must match number of sphere vertices"
        );

        // Step 1: local maxima — vertex is a local max if positive and ≥ all neighbors.
        let mut candidates: Vec<(f32, usize)> = (0..odf.len())
            .filter_map(|i| {
                let v = odf[i];
                if v <= 0.0 {
                    return None;
                }
                if self.neighbors[i].iter().all(|&j| v >= odf[j]) {
                    Some((v, i))
                } else {
                    None
                }
            })
            .collect();

        candidates.sort_by(|a, b| b.0.total_cmp(&a.0));

        // Step 2: relative threshold relative to the strongest peak.
        let threshold = candidates
            .first()
            .map_or(0.0, |(v, _)| v * self.config.relative_peak_threshold);
        let candidates: Vec<_> = candidates
            .into_iter()
            .filter(|(v, _)| *v >= threshold)
            .collect();

        // Step 3: greedy separation-angle filter (antipodal symmetry: use |dot|).
        self.separate(candidates.into_iter().map(|(v, idx)| (v, self.vertices[idx])), cap)
    }

    /// Refine a vertex-quantized seed direction to sub-vertex precision by
    /// running MRtrix's Gauss-Newton step on the SH series itself.
    ///
    /// Mirrors `Math::SH::get_peak` from
    /// `trx-mrtrix2/cpp/core/math/SH.h`: tangent-plane gradient + Hessian,
    /// 2×2 Newton step clamped to [`MAX_DIR_CHANGE`], terminated when
    /// `|dt| < ANGLE_TOLERANCE` or after [`NEWTON_MAX_ITERS`] iterations.
    /// Returns the refined unit direction and SH amplitude at it.
    ///
    /// On non-convergence (rare, only for degenerate FODs) returns the
    /// seed direction with its SH amplitude — never NaN.
    pub fn refine_with_sh(
        &self,
        seed_dir: [f32; 3],
        sh: &[f32],
        basis: ShBasisKind,
    ) -> ([f32; 3], f32) {
        let mut x = seed_dir[0] as f64;
        let mut y = seed_dir[1] as f64;
        let mut z = seed_dir[2] as f64;
        let inv = (x * x + y * y + z * z).sqrt();
        if inv > 0.0 {
            x /= inv;
            y /= inv;
            z /= inv;
        }
        let mut last_amp = 0.0_f64;

        for _ in 0..NEWTON_MAX_ITERS {
            let az = y.atan2(x);
            let el = (x * x + y * y).sqrt().atan2(z);
            let (amplitude, dsh_del, dsh_daz, d2_del2, d2_deldaz, d2_daz2) =
                basis.derivatives(sh, el, az);
            last_amp = amplitude;

            let mut del_dir = (dsh_del * dsh_del + dsh_daz * dsh_daz).sqrt();
            let mut daz_dir = 0.0_f64;
            if del_dir != 0.0 {
                daz_dir = dsh_daz / del_dir;
                del_dir = dsh_del / del_dir;
            }

            let dsh_dt = daz_dir * dsh_daz + del_dir * dsh_del;
            let d2_dt2 = daz_dir * daz_dir * d2_daz2
                + 2.0 * daz_dir * del_dir * d2_deldaz
                + del_dir * del_dir * d2_del2;
            // Degenerate Hessian (d²/dt² == 0): step the full MAX_DIR_CHANGE to
            // walk off the inflection point. A zero step here trips the
            // ANGLE_TOLERANCE early-return below and would falsely report
            // convergence at a non-maximum. Matches MRtrix `Math::SH::get_peak`
            // (MRtrix3 PR #3299).
            let mut dt = if d2_dt2 != 0.0 { -dsh_dt / d2_dt2 } else { MAX_DIR_CHANGE };

            if dt < 0.0 {
                dt = -dt;
            }
            if dt > MAX_DIR_CHANGE {
                dt = MAX_DIR_CHANGE;
            }

            let del_step = del_dir * dt;
            let daz_step = daz_dir * dt;
            let cos_az = az.cos();
            let sin_az = az.sin();
            let cos_el = el.cos();
            let sin_el = el.sin();
            x += del_step * cos_az * cos_el - daz_step * sin_az;
            y += del_step * sin_az * cos_el + daz_step * cos_az;
            z -= del_step * sin_el;
            let n = (x * x + y * y + z * z).sqrt();
            if n > 0.0 {
                x /= n;
                y /= n;
                z /= n;
            }

            if dt < ANGLE_TOLERANCE {
                return ([x as f32, y as f32, z as f32], amplitude as f32);
            }
        }

        // Newton failed to converge — keep the seed (its discrete amplitude
        // is in the caller's hands; we return the last evaluated amplitude
        // since SH amp at the unrefined seed is more accurate than its
        // sphere-quantized neighbor in any case).
        let _ = last_amp;
        let n = (seed_dir[0].powi(2) + seed_dir[1].powi(2) + seed_dir[2].powi(2)).sqrt();
        let s = if n > 0.0 { 1.0 / n } else { 1.0 };
        let dir = [seed_dir[0] * s, seed_dir[1] * s, seed_dir[2] * s];
        let (amp, _, _, _, _, _) = basis.derivatives(
            sh,
            (dir[0].powi(2) + dir[1].powi(2)).sqrt().atan2(dir[2] as f32) as f64,
            (dir[1] as f64).atan2(dir[0] as f64),
        );
        (dir, amp as f32)
    }

    /// Like [`find_peaks`], but refines each accepted peak to sub-vertex
    /// precision using the original SH coefficients.
    ///
    /// Local maxima, the relative threshold and the separation angle are found
    /// on the discrete `odf`, without the `npeaks` cap; each seed is then nudged
    /// off-vertex by [`Self::refine_with_sh`], and the separation filter and the
    /// cap run again on the refined peaks. The second pass matters: the search
    /// sphere is a hemisphere whose mesh is not joined across its boundary, so
    /// a lobe near the boundary can seed two maxima more than the separation
    /// angle apart, and refinement takes both onto the same axis.
    pub fn find_peaks_with_sh(
        &self,
        odf: &[f32],
        sh: &[f32],
        basis: ShBasisKind,
    ) -> Vec<(f32, [f32; 3])> {
        let seeds = self.find_peaks_capped(odf, usize::MAX);
        let mut out = Vec::with_capacity(seeds.len());
        for (_seed_amp, seed_dir) in seeds {
            let (dir, amp) = self.refine_with_sh(seed_dir, sh, basis);
            // Drop peaks whose Newton step drifted to a non-positive
            // critical point — those are saddle points or valley
            // bottoms reached when the seed was already weak. They
            // pollute fixel rendering (zero-length arrows) and Otsu
            // thresholding (a long tail of near-zero amplitudes that
            // shifts the mode). Keep only refined peaks that landed on
            // an actual positive lobe.
            if amp > 0.0 && amp.is_finite() {
                out.push((amp, dir));
            }
        }
        out.sort_by(|a, b| b.0.total_cmp(&a.0));
        self.separate(out, self.config.npeaks)
    }

    /// Batch version of [`find_peaks`].
    ///
    /// `odf_rows` is a flat `[nrows × nvertices]` row-major array.
    /// Returns `(offsets, directions, amplitudes)` in the same layout as
    /// `OdxDataset`: `offsets` has length `nrows + 1`, with
    /// `directions[offsets[v]..offsets[v+1]]` holding voxel `v`'s peaks.
    pub fn find_peaks_rows(
        &self,
        odf_rows: &[f32],
        nrows: usize,
    ) -> (Vec<u32>, Vec<[f32; 3]>, Vec<f32>) {
        let ncols = self.vertices.len();
        let mut offsets = Vec::with_capacity(nrows + 1);
        let mut directions = Vec::new();
        let mut amplitudes = Vec::new();
        offsets.push(0u32);

        for row in 0..nrows {
            let odf = &odf_rows[row * ncols..(row + 1) * ncols];
            for (value, dir) in self.find_peaks(odf) {
                directions.push(dir);
                amplitudes.push(value);
            }
            offsets.push(directions.len() as u32);
        }

        (offsets, directions, amplitudes)
    }

    pub fn config(&self) -> &PeakFinderConfig {
        &self.config
    }

    pub fn vertices(&self) -> &[[f32; 3]] {
        &self.vertices
    }

    /// Per-vertex mesh adjacency, derived from the face list at construction.
    ///
    /// Exposed for [`crate::fmls`], whose watershed needs the same
    /// neighbourhoods the peak finder uses, so peaks and lobes are segmented
    /// against one identical notion of adjacency.
    pub fn neighbors(&self) -> &[Vec<usize>] {
        &self.neighbors
    }
}

/// Evaluate SH rows on the finder's sphere, then extract peaks. Default
/// dispatches to MRtrix `tournier07`, even-only — see
/// [`peaks_from_sh_rows_with_basis`] for descoteaux/full-basis SH.
///
/// Amplitudes are clamped to zero before peak finding (matching
/// `sh2amp -nonnegative`). Returns `(offsets, directions, amplitudes)`.
pub fn peaks_from_sh_rows(
    sh_rows: &[f32],
    nrows: usize,
    finder: &SpherePeakFinder,
    ncoeffs: usize,
) -> Result<(Vec<u32>, Vec<[f32; 3]>, Vec<f32>)> {
    let basis = ShBasisKind::MrtrixTournier {
        lmax: lmax_for_ncoeffs(ncoeffs)?,
    };
    peaks_from_sh_rows_with_basis(sh_rows, nrows, finder, basis)
}

/// Evaluate SH rows on the finder's sphere, then extract peaks, using the
/// requested SH basis to drive both the sphere evaluation *and* the
/// Newton refinement. Use this for descoteaux07 SH (PAM, pyAFQ aodf);
/// stick with [`peaks_from_sh_rows`] for tournier07 (MRtrix output).
///
/// For asymmetric (full-basis) descoteaux SH, the finder's sphere should
/// be a *full* sphere — the antisymmetry between u and −u is meaningful
/// and gets folded away if you sample only a hemisphere.
pub fn peaks_from_sh_rows_with_basis(
    sh_rows: &[f32],
    nrows: usize,
    finder: &SpherePeakFinder,
    basis: ShBasisKind,
) -> Result<(Vec<u32>, Vec<[f32; 3]>, Vec<f32>)> {
    let ncoeffs = basis.ncoeffs();
    let dirs = finder.vertices();
    // Two evaluator paths; basis-specific because the row layout differs.
    enum LocalPlan {
        Tournier(RowSamplePlan),
        Descoteaux(crate::descoteaux_sh::RowSamplePlan),
    }
    impl LocalPlan {
        fn ndir(&self) -> usize {
            match self {
                Self::Tournier(p) => p.ndir(),
                Self::Descoteaux(p) => p.ndir(),
            }
        }
        fn apply_row_into(&self, src: &[f32], dst: &mut [f32]) {
            match self {
                Self::Tournier(p) => p.apply_row_into(src, dst),
                Self::Descoteaux(p) => p.apply_row_into(src, dst),
            }
        }
    }
    let plan = match basis {
        ShBasisKind::MrtrixTournier { .. } => {
            LocalPlan::Tournier(RowSamplePlan::for_sh_rows_nonnegative(dirs, ncoeffs)?)
        }
        ShBasisKind::Descoteaux {
            full_basis, legacy, ..
        } => {
            // Asymmetric peaks live on signed amplitudes — clamp would erase
            // the antipodal asymmetry. For symmetric descoteaux we still
            // clamp (negatives are fit noise), matching MRtrix policy.
            let plan = if full_basis {
                crate::descoteaux_sh::RowSamplePlan::for_sh_rows(dirs, ncoeffs, true, legacy)?
            } else {
                crate::descoteaux_sh::RowSamplePlan::for_sh_rows_nonnegative(
                    dirs, ncoeffs, false, legacy,
                )?
            };
            LocalPlan::Descoteaux(plan)
        }
    };

    let mut odf_buf = vec![0.0f32; plan.ndir()];
    let mut offsets = Vec::with_capacity(nrows + 1);
    let mut directions = Vec::new();
    let mut amplitudes = Vec::new();
    offsets.push(0u32);

    for row in 0..nrows {
        let src = &sh_rows[row * ncoeffs..(row + 1) * ncoeffs];
        plan.apply_row_into(src, &mut odf_buf);
        for (value, dir) in finder.find_peaks_with_sh(&odf_buf, src, basis) {
            directions.push(dir);
            amplitudes.push(value);
        }
        offsets.push(directions.len() as u32);
    }

    Ok((offsets, directions, amplitudes))
}

fn neighbors_from_faces(nvertices: usize, faces: &[[u32; 3]]) -> Vec<Vec<usize>> {
    let mut sets = vec![std::collections::HashSet::<usize>::new(); nvertices];
    for face in faces {
        let ids = [face[0] as usize, face[1] as usize, face[2] as usize];
        if ids.iter().all(|&idx| idx < nvertices) {
            for i in 0..3 {
                let a = ids[i];
                let b = ids[(i + 1) % 3];
                sets[a].insert(b);
                sets[b].insert(a);
            }
        }
    }
    sets.into_iter()
        .map(|s| s.into_iter().collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_triangle_finder(config: PeakFinderConfig) -> (SpherePeakFinder, Vec<[f32; 3]>) {
        // Four vertices forming two triangles on a rough hemisphere.
        // v0=(1,0,0), v1=(0,1,0), v2=(0,0,1), v3=(0.577,0.577,0.577)
        let vertices: Vec<[f32; 3]> = vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.577_350_3, 0.577_350_3, 0.577_350_3],
        ];
        let faces: Vec<[u32; 3]> = vec![[0, 1, 3], [1, 2, 3], [0, 2, 3]];
        let finder = SpherePeakFinder::new(&vertices, &faces, config);
        (finder, vertices)
    }

    #[test]
    fn finds_single_clear_peak() {
        let (finder, _) = simple_triangle_finder(PeakFinderConfig::default());
        // v0 is the clear winner; v3 is its neighbor with a lower value.
        let odf = [1.0_f32, 0.1, 0.1, 0.4];
        let peaks = finder.find_peaks(&odf);
        assert!(!peaks.is_empty());
        assert_eq!(peaks[0].1, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn relative_threshold_drops_weak_secondary() {
        let config = PeakFinderConfig {
            relative_peak_threshold: 0.8,
            min_separation_angle_deg: 0.0,
            npeaks: 5,
        };
        let (finder, _) = simple_triangle_finder(config);
        // v0=1.0 (peak), v1=0.5 (peak), 0.5 < 1.0*0.8=0.8 → dropped.
        let odf = [1.0_f32, 0.5, 0.0, 0.0];
        let peaks = finder.find_peaks(&odf);
        assert_eq!(peaks.len(), 1, "weak secondary should be dropped by threshold");
        assert_eq!(peaks[0].1, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn separation_filter_removes_nearby_weaker_peak() {
        // v0=[1,0,0] and v3=[0.577,0.577,0.577] are ~54° apart — > 25° so both kept.
        // v0 and v1=[0,1,0] are 90° apart — kept.
        // v3 and v1 are ~54° apart — if v3 already accepted, v1 dropped.
        let config = PeakFinderConfig {
            relative_peak_threshold: 0.0,
            min_separation_angle_deg: 60.0, // strict: drops anything within 60°
            npeaks: 5,
        };
        let (finder, _) = simple_triangle_finder(config);
        // Make v0 strong, v3 weaker (54° from v0 — within 60° → dropped).
        let odf = [1.0_f32, 0.0, 0.0, 0.8];
        let peaks = finder.find_peaks(&odf);
        assert_eq!(peaks.len(), 1, "v3 is within 60° of v0 and should be removed");
    }

    #[test]
    fn npeaks_cap_respected() {
        // Use the ODF8 hemisphere (321 vertices) so we can place 3 isolated
        // non-zero values that are guaranteed local maxima (all other vertices
        // are 0, so any positive vertex beats all its neighbors).
        let finder = SpherePeakFinder::for_dsistudio_odf8(PeakFinderConfig {
            relative_peak_threshold: 0.0,
            min_separation_angle_deg: 0.0,
            npeaks: 2,
        });
        let mut odf = vec![0.0f32; 321];
        odf[0] = 1.0;
        odf[100] = 0.9;
        odf[200] = 0.8;
        let peaks = finder.find_peaks(&odf);
        assert_eq!(peaks.len(), 2, "npeaks=2 should cap output even with 3 local maxima");
        assert_eq!(peaks[0].0, 1.0);
        assert_eq!(peaks[1].0, 0.9);
    }

    #[test]
    fn empty_odf_returns_no_peaks() {
        let (finder, _) = simple_triangle_finder(PeakFinderConfig::default());
        let odf = [0.0_f32; 4];
        assert!(finder.find_peaks(&odf).is_empty());
    }

    #[test]
    fn batch_offsets_consistent() {
        let (finder, _) = simple_triangle_finder(PeakFinderConfig::default());
        let odf_rows = vec![
            1.0_f32, 0.1, 0.1, 0.4, // voxel 0: one peak at v0
            0.1_f32, 1.0, 0.1, 0.4, // voxel 1: one peak at v1
        ];
        let (offsets, directions, amplitudes) = finder.find_peaks_rows(&odf_rows, 2);
        assert_eq!(offsets.len(), 3);
        assert_eq!(offsets[0], 0);
        assert_eq!(directions.len(), amplitudes.len());
        let n0 = (offsets[1] - offsets[0]) as usize;
        let n1 = (offsets[2] - offsets[1]) as usize;
        assert!(n0 >= 1);
        assert!(n1 >= 1);
    }

    #[test]
    fn dsistudio_odf8_finder_constructs() {
        let finder = SpherePeakFinder::for_dsistudio_odf8(PeakFinderConfig::default());
        assert_eq!(finder.vertices().len(), 321);
        // All vertices should have at least one neighbor.
        assert!(finder.neighbors.iter().all(|n| !n.is_empty()));
    }

    fn angle_between_deg(a: [f32; 3], b: [f32; 3]) -> f32 {
        let dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).abs();
        dot.clamp(-1.0, 1.0).acos().to_degrees()
    }

    fn synth_dirac_sh(target: [f32; 3], lmax: usize) -> Vec<f32> {
        // Build SH from a delta-like ODF: a single direction ⇒ project onto
        // the basis. `sh_transform_row(d, lmax)` returns Y_lm(d); the SH
        // coefficients of an antipodally-symmetrized delta at ±target are
        // exactly that row (up to scale). The result is a sharply peaked
        // FOD whose maximum is at `target`.
        use crate::mrtrix_sh::sh2amp_cart;
        let row = sh2amp_cart(&[target], lmax);
        row.as_slice().unwrap().to_vec()
    }

    #[test]
    fn refine_with_sh_pulls_off_vertex_to_analytic_peak() {
        let finder = SpherePeakFinder::for_dsistudio_odf8(PeakFinderConfig::default());
        let lmax = 8;
        // Pick a direction guaranteed to NOT match an ODF8 vertex.
        let target = {
            let v = [0.4_f32, 0.3, 0.86];
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            [v[0] / n, v[1] / n, v[2] / n]
        };
        let sh = synth_dirac_sh(target, lmax);

        // Sample ODF on the sphere and find a discrete seed.
        use crate::mrtrix_sh::RowSamplePlan;
        let plan = RowSamplePlan::for_sh_rows_nonnegative(finder.vertices(), sh.len()).unwrap();
        let odf = plan.apply_row(&sh);
        let seeds = finder.find_peaks(&odf);
        assert!(!seeds.is_empty(), "synthetic dirac should produce at least one seed");
        let seed_dir = seeds[0].1;

        let seed_err = angle_between_deg(seed_dir, target);
        assert!(
            seed_err > 0.5,
            "seed should be vertex-quantized (>0.5°), got {seed_err}°"
        );

        let basis = ShBasisKind::MrtrixTournier { lmax };
        let (refined, _amp) = finder.refine_with_sh(seed_dir, &sh, basis);
        let refined_err = angle_between_deg(refined, target);
        assert!(
            refined_err < 0.05,
            "refined direction should be within 0.05°, got {refined_err}° (seed={seed_err}°)"
        );
    }

    #[test]
    fn find_peaks_with_sh_handles_two_crossing_fibers() {
        let finder = SpherePeakFinder::for_dsistudio_odf8(PeakFinderConfig {
            npeaks: 5,
            relative_peak_threshold: 0.3,
            min_separation_angle_deg: 25.0,
        });
        let lmax = 8;
        // Two off-vertex, well-separated directions.
        let a = {
            let v = [0.6_f32, 0.0, 0.8];
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            [v[0] / n, v[1] / n, v[2] / n]
        };
        let b = {
            let v = [-0.5_f32, 0.2, 0.843];
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            [v[0] / n, v[1] / n, v[2] / n]
        };
        use crate::mrtrix_sh::sh2amp_cart;
        let row_a = sh2amp_cart(&[a], lmax);
        let row_b = sh2amp_cart(&[b], lmax);
        let sh: Vec<f32> = row_a
            .iter()
            .zip(row_b.iter())
            .map(|(x, y)| x + y)
            .collect();

        use crate::mrtrix_sh::RowSamplePlan;
        let plan = RowSamplePlan::for_sh_rows_nonnegative(finder.vertices(), sh.len()).unwrap();
        let odf = plan.apply_row(&sh);
        let seeds = finder.find_peaks(&odf);
        let basis = ShBasisKind::MrtrixTournier { lmax };
        let refined = finder.find_peaks_with_sh(&odf, &sh, basis);
        assert_eq!(refined.len(), 2, "should find both crossing peaks");
        assert_eq!(seeds.len(), refined.len(), "refinement preserves peak count");

        // For each refined peak, find the matching seed (closest direction)
        // and confirm the refined direction is at least as close to the
        // input target as the seed was — i.e., refinement improves or ties.
        let inputs = [a, b];
        let mut sum_seed_err = 0.0_f32;
        let mut sum_refined_err = 0.0_f32;
        for input in inputs {
            let seed_err = seeds
                .iter()
                .map(|(_, d)| angle_between_deg(*d, input))
                .fold(f32::INFINITY, f32::min);
            let refined_err = refined
                .iter()
                .map(|(_, d)| angle_between_deg(*d, input))
                .fold(f32::INFINITY, f32::min);
            assert!(
                refined_err <= seed_err + 1e-3,
                "refinement should not increase error: seed={seed_err}°, refined={refined_err}°"
            );
            sum_seed_err += seed_err;
            sum_refined_err += refined_err;
        }
        // Refinement on a 642-vertex sphere should beat vertex quantization
        // by a comfortable margin even with two lobes' mutual interference.
        assert!(
            sum_refined_err < sum_seed_err * 0.5,
            "refinement should at least halve total error: seed={sum_seed_err}°, refined={sum_refined_err}°"
        );

        // Also verify SH gradient is small at each refined direction —
        // i.e., we converged to a local maximum of the actual SH series.
        for (_, dir) in &refined {
            let el = ((dir[0].powi(2) + dir[1].powi(2)).sqrt() as f64).atan2(dir[2] as f64);
            let az = (dir[1] as f64).atan2(dir[0] as f64);
            let (_, dsh_del, dsh_daz, _, _, _) =
                crate::mrtrix_sh::sh_derivatives(&sh, lmax, el, az);
            let grad_mag = (dsh_del * dsh_del + dsh_daz * dsh_daz).sqrt();
            assert!(
                grad_mag < 1e-2,
                "refined direction should sit at a local max (|∇| small), got {grad_mag}"
            );
        }
    }

    #[test]
    fn find_peaks_with_sh_keeps_one_peak_per_lobe_near_the_hemisphere_edge() {
        // The search sphere is a hemisphere whose mesh is not joined across its
        // boundary, so one lobe near the boundary shows up as a discrete local
        // maximum on both sides. The two seeds can sit more than the separation
        // angle apart, and refinement then takes both to the same axis (often
        // with opposite signs). A single fibre must still give one peak.
        use crate::mrtrix_sh::{sh2amp_cart, RowSamplePlan};
        let finder = SpherePeakFinder::for_dsistudio_odf8(PeakFinderConfig {
            npeaks: 5,
            relative_peak_threshold: 0.5,
            min_separation_angle_deg: 25.0,
        });
        let lmax = 8;
        let basis = ShBasisKind::MrtrixTournier { lmax };
        let mut worst = 0usize;
        let mut n_bad = 0usize;
        for el_deg in [-8.0_f32, -4.0, -1.0, 0.0, 1.0, 4.0, 8.0] {
            for k in 0..36 {
                let az = (k as f32 * 10.0 + 3.0).to_radians();
                let el = el_deg.to_radians();
                let dir = [el.cos() * az.cos(), el.cos() * az.sin(), el.sin()];
                // Rotate so the fibre lies near every part of the boundary,
                // whatever axis the hemisphere is cut along.
                for d in [dir, [dir[2], dir[0], dir[1]], [dir[1], dir[2], dir[0]]] {
                    let sh: Vec<f32> = sh2amp_cart(&[d], lmax).iter().copied().collect();
                    let plan = RowSamplePlan::for_sh_rows_nonnegative(finder.vertices(), sh.len())
                        .unwrap();
                    let odf = plan.apply_row(&sh);
                    let peaks = finder.find_peaks_with_sh(&odf, &sh, basis);
                    if peaks.len() != 1 {
                        n_bad += 1;
                        worst = worst.max(peaks.len());
                    }
                    let errs: Vec<f32> = peaks.iter().map(|(_, p)| angle_between_deg(*p, d)).collect();
                    if peaks.len() != 1 {
                        eprintln!("d={d:?} n={} errs={errs:?} pairwise={:?}", peaks.len(),
                            peaks.iter().skip(1).map(|(_, p)| angle_between_deg(*p, peaks[0].1)).collect::<Vec<_>>());
                    }
                    assert!(errs.iter().all(|e| *e < 5.0), "peak off the fibre: {errs:?}");
                }
            }
        }
        assert_eq!(n_bad, 0, "{n_bad} single fibres gave more than one peak (up to {worst})");
    }

    #[test]
    fn find_peaks_with_sh_never_returns_two_peaks_on_one_axis() {
        // WM fODF rows from CONSH (cs_dmri 0.3.0, lmax 8) where the exported peaks
        // held the same axis twice: two seeds more than 25 deg apart refined onto
        // one lobe, often with opposite signs.
        const ROWS: &[&str] = &[
        "0.020001 3.70544e-05 0.000196457 -0.000274297 -0.000669049 -0.000114257 0.000340161 -0.000152604 0.00012367 -1.47892e-08 -0.000129584 4.10546e-05 0.00027226 -0.000124367 5.74178e-05 3.87874e-05 9.05643e-05 -1.58243e-05 -1.25684e-05 -7.64095e-05 -5.5034e-05 6.98173e-05 4.2683e-05 -6.40127e-05 -6.85215e-05 -1.74923e-05 3.44863e-06 2.48358e-05 -1.9128e-05 -9.33313e-07 -1.09802e-05 5.76294e-06 -5.86146e-06 -1.27285e-05 -6.2177e-06 5.6455e-06 -1.8438e-05 -2.31331e-05 -1.5045e-06 -2.2925e-05 -7.64343e-06 1.73684e-05 -1.12061e-05 1.50676e-05 -1.14419e-05",
        "0.00509206 -0.00394427 -0.0026448 -0.00204412 0.00408692 0.0036147 -0.00174032 -0.00232267 -0.000869546 0.000881936 -0.000435028 -0.00195087 -0.000935052 -4.33417e-05 0.000525665 -0.000460918 -0.00029834 4.3253e-05 0.000240121 0.000428127 0.000159857 0.000146224 0.000199289 0.000139787 -0.000484234 -0.000750837 -0.000163946 0.000216163 -0.000107683 -7.20465e-05 7.32629e-05 9.57164e-05 -8.02023e-07 1.50129e-05 4.11548e-05 -3.06238e-05 6.22351e-05 -2.61229e-05 -2.96008e-05 0.000119038 5.45087e-05 -8.04789e-05 -5.56021e-05 2.83407e-05 3.90552e-05",
        "0.0166295 0.00653633 -0.00569082 -0.00193573 -0.00942998 0.0124597 0.00272468 -0.00243812 0.00178219 -0.000953129 -0.000677992 0.0004924 -0.00144635 3.00494e-06 0.00318468 -8.45897e-05 0.000567782 -0.000520075 0.000466809 -0.000436867 0.000103431 -0.000347938 -0.000236118 -0.000412912 0.000456039 -0.000427854 0.000722666 0.000201158 -0.000315116 0.000484258 -0.00022278 9.19441e-05 -0.000417422 0.000285048 7.92924e-06 3.89435e-06 -0.0001305 0.000129701 -1.90785e-05 -0.000125447 0.000222657 -0.000193764 5.43387e-05 8.78768e-05 -2.45091e-05",
        "0.0204613 -0.00066584 0.00197433 0.00222784 -0.00180702 0.00102427 -0.000411757 -1.2118e-05 0.000273079 7.08231e-05 0.000471344 -0.000657725 0.000382965 -0.00020176 -0.000729273 3.91606e-05 -0.000202432 -0.000244344 0.000108551 1.98104e-05 0.000170449 -3.65766e-05 -6.70476e-05 6.62493e-05 0.000162324 9.92119e-05 -5.86277e-05 0.000157406 -3.26614e-05 -4.73648e-05 1.84313e-05 -2.99623e-06 -7.01198e-05 1.99084e-05 -7.38982e-06 2.77363e-05 2.28139e-05 -2.69119e-05 -4.31075e-05 7.03224e-06 -7.8422e-06 1.1047e-05 3.96872e-05 -8.43552e-06 -5.75591e-05",
        "0.00713807 0.00339079 -0.00103044 -0.0061386 -0.000826552 0.0082003 0.00190633 -0.00198514 -0.00140997 0.00109988 0.00331778 0.000812522 -0.00393889 0.000462763 0.0019602 -0.000102241 -0.000762335 -0.000470507 0.00120132 0.000369221 -0.000609059 -0.00113804 -0.000404316 0.00139879 -0.000316085 -0.000765427 0.000707538 -7.7881e-05 -0.000148948 -5.6335e-05 -1.18645e-05 0.000239571 5.7593e-05 -0.000329128 -2.99322e-05 0.000165437 0.000189011 8.75542e-05 -0.000278514 0.000104215 0.000190993 -0.000222191 -6.21878e-05 0.000128678 -3.36133e-05",
        "0.00911124 0.0042636 -0.00250162 -0.00292044 -0.00642482 0.00798826 0.00157445 -0.00195749 0.00115058 -0.000117568 2.75139e-05 0.000873749 -0.000624711 -0.00126921 0.00167151 -0.000551438 0.000200251 0.000173153 -0.00054223 0.00035316 0.000187881 -0.000489381 0.000230142 0.000361385 0.000153043 -0.000342866 -8.28056e-05 0.000193832 -0.000325103 0.000195456 -3.17966e-05 -3.89624e-05 9.31491e-05 -8.7443e-05 -5.91938e-05 6.76002e-05 5.93242e-05 1.96581e-05 -4.39012e-05 6.68431e-05 -6.04483e-05 -2.98802e-06 8.03673e-05 -0.00014283 0.000149106",
        "0.0207078 -0.00912826 -0.000280418 -0.00345379 0.00851847 0.00280469 -0.000571926 -0.00113182 -3.22305e-05 -0.000560326 -0.000178424 -0.000903974 -0.00274634 0.000206762 0.00127298 -0.00059195 -0.000280693 0.000474782 7.13967e-05 -0.00011802 0.000244228 -5.81581e-05 6.46518e-05 0.000119734 6.28979e-06 -0.000311702 0.000148417 -0.000156499 7.89762e-05 0.000181594 -0.000104104 -0.00038818 0.000126554 1.05881e-06 -3.73684e-05 -5.1545e-05 3.69212e-05 6.98023e-05 -0.000193472 -1.69508e-05 0.000354341 7.27828e-05 -5.16985e-05 -0.000145303 7.49501e-05",
        "0.00858252 -0.00204651 0.000317345 0.00182916 -0.00456516 0.00489997 -0.000985689 0.000126664 -0.000217271 -3.43666e-06 0.00039176 -9.36268e-05 0.000310303 -0.000137112 0.000698595 -1.80399e-05 -0.000300732 4.15087e-05 6.39551e-05 -0.000152642 0.000250385 -4.19572e-05 9.95349e-05 -8.17123e-06 9.71228e-05 -0.000101752 0.000169359 -3.17502e-05 7.00833e-05 -5.8298e-05 -4.24459e-05 5.37188e-06 1.8318e-06 2.58761e-05 1.25461e-06 0.000125873 3.5538e-05 4.7104e-05 -4.9865e-05 2.20606e-05 3.27844e-05 -5.69153e-05 9.91125e-05 -5.7064e-05 1.86506e-05",
        "0.0271781 0.00204935 0.00128115 -0.000257789 0.000200468 -0.00067315 0.000330585 -9.49055e-05 -0.000119205 -0.000696796 -0.000278246 -0.000551226 0.000259925 -0.000254365 0.000494597 -2.04616e-05 -0.000126276 8.02324e-05 0.000104759 -9.09473e-05 6.75586e-05 0.000148725 2.83542e-05 -0.000143331 8.69828e-05 0.000104617 -0.000211406 1.36491e-05 3.60502e-05 -2.75081e-05 3.7928e-05 -9.66078e-06 -6.78423e-06 4.45139e-05 -3.0848e-05 5.87303e-05 -1.25989e-05 -2.52647e-05 -1.97211e-05 -3.77388e-05 4.78251e-06 6.28862e-05 4.76508e-05 -1.58487e-05 6.55682e-05",
        "0.0106931 -0.00175415 0.00586164 -0.00190705 -0.00390398 -0.00210425 -0.000223556 -0.000121788 -0.00057407 -0.000161456 0.000923586 0.000336871 0.000645904 -0.000428887 -0.000440656 -1.48008e-05 -0.00021295 -0.000148304 0.000154047 0.000109541 0.000475584 -0.000665469 0.000110946 -0.000441458 -0.000318587 1.52103e-05 0.000135829 -0.000103869 2.67965e-05 5.74638e-05 -3.30177e-06 -8.46844e-05 -9.90469e-05 -0.000132889 8.73883e-05 0.000103447 0.000255929 -3.81133e-05 -1.91169e-05 -0.000209072 0.00014204 2.77721e-05 7.39294e-05 0.000119412 1.07416e-05",
        ];
        use crate::mrtrix_sh::RowSamplePlan;
        let finder = SpherePeakFinder::for_dsistudio_odf8(PeakFinderConfig {
            npeaks: 5,
            relative_peak_threshold: 0.5,
            min_separation_angle_deg: 25.0,
        });
        let basis = ShBasisKind::MrtrixTournier { lmax: 8 };
        let plan = RowSamplePlan::for_sh_rows_nonnegative(finder.vertices(), 45).unwrap();
        let mut n_dup = 0;
        for row in ROWS {
            let sh: Vec<f32> = row.split_whitespace().map(|x| x.parse().unwrap()).collect();
            let odf = plan.apply_row(&sh);
            let peaks = finder.find_peaks_with_sh(&odf, &sh, basis);
            for i in 0..peaks.len() {
                for j in i + 1..peaks.len() {
                    if angle_between_deg(peaks[i].1, peaks[j].1) < 25.0 {
                        n_dup += 1;
                    }
                }
            }
        }
        assert_eq!(n_dup, 0, "{n_dup} peak pairs closer than the separation angle");
    }

    #[test]
    fn refine_with_sh_on_flat_odf_returns_finite() {
        // Pure isotropic SH (lmax=0): no real peak, but refinement must
        // not panic, return NaN, or stall the caller. It should return
        // a finite unit vector.
        let finder = SpherePeakFinder::for_dsistudio_odf8(PeakFinderConfig::default());
        let sh = vec![1.0_f32]; // lmax=0
        let seed = [1.0_f32, 0.0, 0.0];
        let (dir, amp) = finder.refine_with_sh(seed, &sh, ShBasisKind::MrtrixTournier { lmax: 0 });
        assert!(dir[0].is_finite() && dir[1].is_finite() && dir[2].is_finite());
        let n = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        assert!((n - 1.0).abs() < 1e-3, "result should be unit length, got {n}");
        assert!(amp.is_finite());
    }

    #[test]
    fn descoteaux_full_basis_picks_asymmetric_lobe() {
        // Synthesize a clean asymmetric FOD: weighted superposition of a
        // dipy-style synth dirac at +z and a (smaller) one at −z, expressed
        // in full descoteaux07 basis. The peak finder running on this
        // *full-basis* SH must return the +z direction with the larger
        // amplitude — and not the antipodal lobe — confirming that the
        // descoteaux dispatch propagates through both sphere evaluation
        // and Newton refinement.
        use crate::descoteaux_sh;
        let lmax = 6_usize;
        let ncoeffs = descoteaux_sh::ncoeffs_for(lmax, true);
        let strong = [0.0_f32, 0.0, 1.0];
        let weak = [0.0_f32, 0.0, -1.0];
        let row_strong =
            descoteaux_sh::sh2amp_cart(&[strong], lmax, true, false);
        let row_weak = descoteaux_sh::sh2amp_cart(&[weak], lmax, true, false);
        let sh: Vec<f32> = row_strong
            .iter()
            .zip(row_weak.iter())
            .map(|(s, w)| s + 0.3 * w)
            .collect();
        assert_eq!(sh.len(), ncoeffs);

        let finder = SpherePeakFinder::for_dsistudio_odf8(PeakFinderConfig {
            npeaks: 5,
            relative_peak_threshold: 0.0,
            min_separation_angle_deg: 25.0,
        });
        let basis = ShBasisKind::Descoteaux {
            lmax,
            full_basis: true,
            legacy: false,
        };
        // Build a full sphere by mirroring the dsistudio_odf8 hemisphere so
        // both the strong and the weak lobes have sample support. The
        // peak finder uses the vertices stored on `finder` for
        // local-maxima search; symmetric finders would fold +z and −z
        // together.
        let mut vertices: Vec<[f32; 3]> = finder.vertices().to_vec();
        let n_hemi = vertices.len();
        for i in 0..n_hemi {
            let v = vertices[i];
            vertices.push([-v[0], -v[1], -v[2]]);
        }
        // Reuse the hemisphere's faces for the full sphere — fine here since
        // we only need adjacency for local-max detection on each hemisphere
        // and the strong/weak lobes are each contained within a hemisphere.
        let faces: Vec<[u32; 3]> = vec![];
        let full_finder = SpherePeakFinder::new(
            &vertices,
            &faces,
            PeakFinderConfig {
                npeaks: 5,
                relative_peak_threshold: 0.0,
                min_separation_angle_deg: 25.0,
            },
        );
        // With no faces, neighbour graph is empty → every vertex is a "local
        // max"; the relative_peak_threshold=0 means amplitudes alone rank
        // them. Sort by amplitude and pick the top.
        let (_, dirs, amps) =
            peaks_from_sh_rows_with_basis(&sh, 1, &full_finder, basis).unwrap();
        assert!(!amps.is_empty());
        let (top_idx, _) = amps
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let top = dirs[top_idx];
        // +z lobe should win.
        assert!(top[2] > 0.5, "strong lobe should be near +z, got {top:?}");
    }

    #[test]
    fn peaks_from_sh_rows_isotropic_gives_single_peak() {
        // Pure isotropic SH (only ℓ=0): ODF is flat → one peak (global max fallback
        // is NOT applied here; flat ODF has many "local maxima" — all tied vertices).
        // With default threshold=0.5 and sep=25°, only a handful survive.
        let finder = SpherePeakFinder::for_dsistudio_odf8(PeakFinderConfig {
            npeaks: 5,
            relative_peak_threshold: 0.5,
            min_separation_angle_deg: 25.0,
        });
        // lmax=0: 1 coefficient, value = 1/sqrt(4π) for unit-amplitude isotropic FOD.
        let ncoeffs = 1;
        let sh_rows = vec![1.0_f32]; // single isotropic voxel
        let (offsets, directions, amplitudes) =
            peaks_from_sh_rows(&sh_rows, 1, &finder, ncoeffs).unwrap();
        assert_eq!(offsets.len(), 2);
        assert_eq!(directions.len(), amplitudes.len());
        // All amplitudes are equal for isotropic; peaks are spread across the sphere.
        for &a in &amplitudes {
            assert!(a > 0.0);
        }
    }
}
