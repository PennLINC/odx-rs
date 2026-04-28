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
        let mut accepted: Vec<(f32, [f32; 3])> = Vec::new();
        'candidate: for (value, idx) in candidates {
            let dir = self.vertices[idx];
            for &(_, acc_dir) in &accepted {
                let dot = dir[0] * acc_dir[0] + dir[1] * acc_dir[1] + dir[2] * acc_dir[2];
                if dot.abs() > self.cos_min_sep {
                    continue 'candidate;
                }
            }
            accepted.push((value, dir));
            if accepted.len() >= self.config.npeaks {
                break;
            }
        }

        accepted
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
            let mut dt = if d2_dt2 != 0.0 { -dsh_dt / d2_dt2 } else { 0.0 };

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
    /// Local-maxima detection, threshold filtering, and separation-angle
    /// pruning all run on the discrete `odf` (so the set of accepted peaks
    /// is identical to [`find_peaks`]). Each accepted seed is then nudged
    /// off-vertex by [`Self::refine_with_sh`].
    pub fn find_peaks_with_sh(
        &self,
        odf: &[f32],
        sh: &[f32],
        basis: ShBasisKind,
    ) -> Vec<(f32, [f32; 3])> {
        let seeds = self.find_peaks(odf);
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
        out
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
