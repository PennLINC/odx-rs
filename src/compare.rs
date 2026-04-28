//! Pairwise fixel-level comparison of two ODX datasets.
//!
//! Compares two ODX files on a shared grid by:
//! 1. computing fixel coherence QC on each side with shared options,
//! 2. matching fixels per voxel with mutual greedy `max(|dot|)` (gated by
//!    `match_angle_deg`),
//! 3. summarizing per-voxel match counts, angles, and DPF disagreements,
//! 4. writing both per-voxel NIfTIs *and* a comparison ODX that mirrors A's
//!    geometry but carries extra per-fixel DPFs and per-voxel DPVs encoding
//!    the comparison.
//!
//! See `docs/compare.md` for the meaning of every emitted field.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ndarray::{Array, IxDyn};
use nifti::{writer::WriterOptions, NiftiHeader};
use serde::Serialize;

use crate::dtype::DType;
use crate::error::{OdxError, Result};
use crate::mmap_backing::vec_into_bytes;
use crate::odx_file::OdxDataset;
use crate::qc::{compute_fixel_qc, FixelQcClass, FixelQcOptions, ThresholdMode, QC_CLASS_DPF_NAME};
use crate::stream::OdxBuilder;

#[derive(Debug, Clone)]
pub struct CompareOptions {
    pub primary_metric: Option<String>,
    pub threshold: ThresholdMode,
    pub coherence_angle_deg: f32,
    pub match_angle_deg: f32,
    pub write_comparison_odx: bool,
}

impl Default for CompareOptions {
    fn default() -> Self {
        Self {
            primary_metric: None,
            threshold: ThresholdMode::Otsu,
            coherence_angle_deg: 15.0,
            match_angle_deg: 30.0,
            write_comparison_odx: true,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CompareReport {
    pub primary_metric: String,
    pub coherence_angle_deg: f32,
    pub match_angle_deg: f32,
    pub n_voxels_a: u64,
    pub n_voxels_b: u64,
    pub n_voxels_intersection: u64,
    pub n_fixels_a: u64,
    pub n_fixels_b: u64,
    pub n_mutual_matches: u64,
    pub n_unmatched_a: u64,
    pub n_unmatched_b: u64,
    pub mean_match_angle_deg: Option<f64>,
    pub coherence_index_a: Option<f64>,
    pub coherence_index_b: Option<f64>,
    pub shared_dpf_keys: Vec<String>,
    pub written_paths: Vec<String>,
}

pub fn compare_odx(
    a: &OdxDataset,
    b: &OdxDataset,
    out_dir: &Path,
    opts: &CompareOptions,
) -> Result<CompareReport> {
    let header_a = a.header();
    let header_b = b.header();
    if header_a.dimensions != header_b.dimensions {
        return Err(OdxError::Argument(format!(
            "ODX dimensions differ: {:?} vs {:?}",
            header_a.dimensions, header_b.dimensions
        )));
    }
    if !affine_close(&header_a.voxel_to_rasmm, &header_b.voxel_to_rasmm, 1e-4) {
        return Err(OdxError::Argument(
            "ODX affines differ; both files must share the same grid".into(),
        ));
    }
    if !opts.match_angle_deg.is_finite()
        || opts.match_angle_deg <= 0.0
        || opts.match_angle_deg >= 90.0
    {
        return Err(OdxError::Argument(format!(
            "match_angle_deg must be in (0, 90), got {}",
            opts.match_angle_deg
        )));
    }

    std::fs::create_dir_all(out_dir)?;

    let dims = header_a.dimensions;
    let total_voxels = (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize);

    let mask_a = a.mask();
    let mask_b = b.mask();
    if mask_a.len() != total_voxels || mask_b.len() != total_voxels {
        return Err(OdxError::Format(
            "mask size does not match dimensions".into(),
        ));
    }

    let qc_opts_a = FixelQcOptions {
        primary_metric: opts.primary_metric.clone(),
        threshold: opts.threshold.clone(),
        angle_degrees: opts.coherence_angle_deg,
    };
    let qc_a = compute_fixel_qc(a, &qc_opts_a)?;
    let primary = qc_a.report.primary_metric.clone();

    let qc_opts_b = FixelQcOptions {
        primary_metric: Some(primary.clone()),
        threshold: opts.threshold.clone(),
        angle_degrees: opts.coherence_angle_deg,
    };
    let qc_b = compute_fixel_qc(b, &qc_opts_b)?;

    let lookup_a = build_voxel_lookup(mask_a, dims, a.offsets())?;
    let lookup_b = build_voxel_lookup(mask_b, dims, b.offsets())?;

    let nb_peaks_a = a.nb_peaks();
    let nb_peaks_b = b.nb_peaks();
    let dirs_a = a.directions();
    let dirs_b = b.directions();
    let offsets_a = a.offsets();
    let offsets_b = b.offsets();

    // Per-voxel scalar volumes (full 3D in C order: flat = i*ny*nz + j*nz + k).
    let mut n_fixels_a_vol = vec![f32::NAN; total_voxels];
    let mut n_fixels_b_vol = vec![f32::NAN; total_voxels];
    let mut n_fixels_diff_vol = vec![f32::NAN; total_voxels];
    let mut n_matched_vol = vec![f32::NAN; total_voxels];
    let mut n_unmatched_a_vol = vec![f32::NAN; total_voxels];
    let mut n_unmatched_b_vol = vec![f32::NAN; total_voxels];
    let mut n_a_collisions_vol = vec![f32::NAN; total_voxels];
    let mut jaccard_vol = vec![f32::NAN; total_voxels];
    let mut dice_vol = vec![f32::NAN; total_voxels];
    let mut mean_angle_vol = vec![f32::NAN; total_voxels];
    let mut max_angle_vol = vec![f32::NAN; total_voxels];
    let mut n_coherent_a_vol = vec![f32::NAN; total_voxels];
    let mut n_coherent_b_vol = vec![f32::NAN; total_voxels];
    let mut n_coherent_mutual_vol = vec![f32::NAN; total_voxels];
    let mut top1_match_vol = vec![f32::NAN; total_voxels];

    for (vox_idx, &compact) in lookup_a.flat_to_compact.iter().enumerate() {
        if compact == usize::MAX {
            continue;
        }
        let count = (offsets_a[compact + 1] - offsets_a[compact]) as f32;
        n_fixels_a_vol[vox_idx] = count;
    }
    for (vox_idx, &compact) in lookup_b.flat_to_compact.iter().enumerate() {
        if compact == usize::MAX {
            continue;
        }
        let count = (offsets_b[compact + 1] - offsets_b[compact]) as f32;
        n_fixels_b_vol[vox_idx] = count;
    }

    // Per-fixel comparison arrays sized to A's nb_peaks (the comparison ODX
    // mirrors A's geometry).
    let mut match_index_b = vec![-1i32; nb_peaks_a];
    let mut match_angle_deg = vec![f32::NAN; nb_peaks_a];
    let mut match_dp = vec![f32::NAN; nb_peaks_a];
    let mut is_mutual = vec![0u8; nb_peaks_a];
    let mut qc_class_b_matched = vec![255u8; nb_peaks_a];

    // Discover shared scalar float DPF keys.
    let shared_keys: Vec<String> = {
        let info_a: BTreeMap<String, _> = a
            .iter_dpf()
            .map(|(name, info)| (name.to_string(), info))
            .collect();
        let info_b: BTreeMap<String, _> = b
            .iter_dpf()
            .map(|(name, info)| (name.to_string(), info))
            .collect();
        info_a
            .iter()
            .filter(|(name, info)| {
                if name.as_str() == QC_CLASS_DPF_NAME {
                    return false;
                }
                if info.ncols != 1 || !info.dtype.is_float() {
                    return false;
                }
                match info_b.get(name.as_str()) {
                    Some(b_info) => b_info.ncols == 1 && b_info.dtype.is_float(),
                    None => false,
                }
            })
            .map(|(name, _)| name.clone())
            .collect()
    };

    let mut a_values: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let mut b_values: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let mut per_key_b_matched: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let mut per_key_diff: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let mut per_key_diff_mean_vol: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let mut per_key_diff_max_abs_vol: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let mut per_key_a_sum_vol: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let mut per_key_b_sum_vol: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let mut per_key_sum_diff_vol: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    for key in &shared_keys {
        a_values.insert(key.clone(), a.scalar_dpf_f32(key)?);
        b_values.insert(key.clone(), b.scalar_dpf_f32(key)?);
        per_key_b_matched.insert(key.clone(), vec![f32::NAN; nb_peaks_a]);
        per_key_diff.insert(key.clone(), vec![f32::NAN; nb_peaks_a]);
        per_key_diff_mean_vol.insert(key.clone(), vec![f32::NAN; total_voxels]);
        per_key_diff_max_abs_vol.insert(key.clone(), vec![f32::NAN; total_voxels]);
        per_key_a_sum_vol.insert(key.clone(), vec![f32::NAN; total_voxels]);
        per_key_b_sum_vol.insert(key.clone(), vec![f32::NAN; total_voxels]);
        per_key_sum_diff_vol.insert(key.clone(), vec![f32::NAN; total_voxels]);
    }

    let primary_a = a.scalar_dpf_f32(&primary).ok();
    let primary_b = b.scalar_dpf_f32(&primary).ok();

    let cos_thresh = opts.match_angle_deg.to_radians().cos();

    let mut total_mutual_matches: u64 = 0;
    let mut total_unmatched_a: u64 = 0;
    let mut total_unmatched_b: u64 = 0;
    let mut sum_match_angle_deg: f64 = 0.0;
    let mut count_match_angle: u64 = 0;
    let mut n_voxels_intersection: u64 = 0;

    let mut diff_sum_per_key = vec![0.0f64; shared_keys.len()];
    let mut diff_max_abs_per_key = vec![0.0f32; shared_keys.len()];

    for flat in 0..total_voxels {
        let ca = lookup_a.flat_to_compact[flat];
        let cb = lookup_b.flat_to_compact[flat];
        if ca == usize::MAX || cb == usize::MAX {
            continue;
        }
        n_voxels_intersection += 1;

        let a_start = offsets_a[ca] as usize;
        let a_end = offsets_a[ca + 1] as usize;
        let b_start = offsets_b[cb] as usize;
        let b_end = offsets_b[cb + 1] as usize;
        let m = a_end - a_start;
        let n = b_end - b_start;

        n_fixels_diff_vol[flat] = m as f32 - n as f32;

        let mut best_b_for_a = vec![(usize::MAX, 0.0f32); m];
        let mut best_a_for_b = vec![(usize::MAX, 0.0f32); n];
        for i in 0..m {
            let a_dir = dirs_a[a_start + i];
            for j in 0..n {
                let b_dir = dirs_b[b_start + j];
                let dp = abs_dot(a_dir, b_dir);
                if dp > best_b_for_a[i].1 {
                    best_b_for_a[i] = (j, dp);
                }
                if dp > best_a_for_b[j].1 {
                    best_a_for_b[j] = (i, dp);
                }
            }
        }

        let mut b_winner_count = vec![0usize; n];
        for &(j, _) in &best_b_for_a {
            if j != usize::MAX {
                b_winner_count[j] += 1;
            }
        }
        let mut collisions = 0usize;
        for &(j, _) in &best_b_for_a {
            if j != usize::MAX && b_winner_count[j] > 1 {
                collisions += 1;
            }
        }
        n_a_collisions_vol[flat] = collisions as f32;

        for v in diff_sum_per_key.iter_mut() {
            *v = 0.0;
        }
        for v in diff_max_abs_per_key.iter_mut() {
            *v = 0.0;
        }

        let mut n_matched = 0usize;
        let mut a_matched = vec![false; m];
        let mut sum_angle = 0.0f64;
        let mut max_angle = 0.0f32;

        for i in 0..m {
            let a_fixel = a_start + i;
            let (best_j, best_dp) = best_b_for_a[i];
            if best_j == usize::MAX {
                continue;
            }
            match_dp[a_fixel] = best_dp;
            let angle_deg = best_dp.clamp(-1.0, 1.0).acos().to_degrees();
            match_angle_deg[a_fixel] = angle_deg;

            let (best_i_for_j, _) = best_a_for_b[best_j];
            let mutual = best_i_for_j == i && best_dp >= cos_thresh;
            if mutual {
                match_index_b[a_fixel] = (b_start + best_j) as i32;
                is_mutual[a_fixel] = 1;
                qc_class_b_matched[a_fixel] = qc_b.classes[b_start + best_j] as u8;
                a_matched[i] = true;
                n_matched += 1;
                sum_angle += angle_deg as f64;
                if angle_deg > max_angle {
                    max_angle = angle_deg;
                }
                for (k_idx, key) in shared_keys.iter().enumerate() {
                    let av = a_values[key][a_fixel];
                    let bv = b_values[key][b_start + best_j];
                    per_key_b_matched.get_mut(key).unwrap()[a_fixel] = bv;
                    let diff = av - bv;
                    per_key_diff.get_mut(key).unwrap()[a_fixel] = diff;
                    diff_sum_per_key[k_idx] += diff as f64;
                    let abs_diff = diff.abs();
                    if abs_diff > diff_max_abs_per_key[k_idx] {
                        diff_max_abs_per_key[k_idx] = abs_diff;
                    }
                }
            }
        }

        let n_unmatched_a = m - n_matched;
        let n_unmatched_b = n - n_matched;
        total_mutual_matches += n_matched as u64;
        total_unmatched_a += n_unmatched_a as u64;
        total_unmatched_b += n_unmatched_b as u64;
        sum_match_angle_deg += sum_angle;
        count_match_angle += n_matched as u64;

        n_matched_vol[flat] = n_matched as f32;
        n_unmatched_a_vol[flat] = n_unmatched_a as f32;
        n_unmatched_b_vol[flat] = n_unmatched_b as f32;
        let union = (m + n).saturating_sub(n_matched);
        jaccard_vol[flat] = if union == 0 {
            f32::NAN
        } else {
            n_matched as f32 / union as f32
        };
        dice_vol[flat] = if (m + n) == 0 {
            f32::NAN
        } else {
            2.0 * n_matched as f32 / (m + n) as f32
        };
        mean_angle_vol[flat] = if n_matched > 0 {
            (sum_angle / n_matched as f64) as f32
        } else {
            f32::NAN
        };
        max_angle_vol[flat] = if n_matched > 0 { max_angle } else { f32::NAN };

        let mut n_coh_a = 0u32;
        for i in 0..m {
            if matches!(qc_a.classes[a_start + i], FixelQcClass::Connected) {
                n_coh_a += 1;
            }
        }
        let mut n_coh_b = 0u32;
        for j in 0..n {
            if matches!(qc_b.classes[b_start + j], FixelQcClass::Connected) {
                n_coh_b += 1;
            }
        }
        let mut n_coh_mut = 0u32;
        for i in 0..m {
            if !a_matched[i] {
                continue;
            }
            if !matches!(qc_a.classes[a_start + i], FixelQcClass::Connected) {
                continue;
            }
            let j = best_b_for_a[i].0;
            if j == usize::MAX {
                continue;
            }
            if matches!(qc_b.classes[b_start + j], FixelQcClass::Connected) {
                n_coh_mut += 1;
            }
        }
        n_coherent_a_vol[flat] = n_coh_a as f32;
        n_coherent_b_vol[flat] = n_coh_b as f32;
        n_coherent_mutual_vol[flat] = n_coh_mut as f32;

        if let (Some(pa), Some(pb)) = (primary_a.as_ref(), primary_b.as_ref()) {
            if m > 0 && n > 0 {
                let top_a = (0..m)
                    .max_by(|&x, &y| {
                        pa[a_start + x]
                            .partial_cmp(&pa[a_start + y])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap();
                let top_b = (0..n)
                    .max_by(|&x, &y| {
                        pb[b_start + x]
                            .partial_cmp(&pb[b_start + y])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap();
                let mutual_top = a_matched[top_a] && best_b_for_a[top_a].0 == top_b;
                top1_match_vol[flat] = if mutual_top { 1.0 } else { 0.0 };
            }
        }

        for (k_idx, key) in shared_keys.iter().enumerate() {
            let a_sum: f64 = (a_start..a_end).map(|i| a_values[key][i] as f64).sum();
            let b_sum: f64 = (b_start..b_end).map(|j| b_values[key][j] as f64).sum();
            per_key_a_sum_vol.get_mut(key).unwrap()[flat] = a_sum as f32;
            per_key_b_sum_vol.get_mut(key).unwrap()[flat] = b_sum as f32;
            per_key_sum_diff_vol.get_mut(key).unwrap()[flat] = (a_sum - b_sum) as f32;
            if n_matched > 0 {
                per_key_diff_mean_vol.get_mut(key).unwrap()[flat] =
                    (diff_sum_per_key[k_idx] / n_matched as f64) as f32;
                per_key_diff_max_abs_vol.get_mut(key).unwrap()[flat] = diff_max_abs_per_key[k_idx];
            }
        }
    }

    let dims_us = [dims[0] as usize, dims[1] as usize, dims[2] as usize];
    let affine = header_a.voxel_to_rasmm;
    let mut written: Vec<String> = Vec::new();
    let mut write_vol = |name: &str, data: &[f32]| -> Result<()> {
        let p: PathBuf = out_dir.join(format!("{name}.nii.gz"));
        write_voxel_scalar_nifti(&dims_us, &affine, data, &p)?;
        written.push(p.display().to_string());
        Ok(())
    };
    write_vol("n_fixels_a", &n_fixels_a_vol)?;
    write_vol("n_fixels_b", &n_fixels_b_vol)?;
    write_vol("n_fixels_diff", &n_fixels_diff_vol)?;
    write_vol("n_matched", &n_matched_vol)?;
    write_vol("n_unmatched_a", &n_unmatched_a_vol)?;
    write_vol("n_unmatched_b", &n_unmatched_b_vol)?;
    write_vol("n_a_collisions", &n_a_collisions_vol)?;
    write_vol("jaccard", &jaccard_vol)?;
    write_vol("dice", &dice_vol)?;
    write_vol("mean_match_angle_deg", &mean_angle_vol)?;
    write_vol("max_match_angle_deg", &max_angle_vol)?;
    write_vol("n_coherent_a", &n_coherent_a_vol)?;
    write_vol("n_coherent_b", &n_coherent_b_vol)?;
    write_vol("n_coherent_mutual", &n_coherent_mutual_vol)?;
    write_vol("top1_match", &top1_match_vol)?;

    let ci_a = qc_a.report.coherence_index;
    let ci_b = qc_b.report.coherence_index;
    let mut coherence_index_diff_vol = vec![f32::NAN; total_voxels];
    if let (Some(a_ci), Some(b_ci)) = (ci_a, ci_b) {
        let diff = (a_ci - b_ci) as f32;
        for flat in 0..total_voxels {
            let ca = lookup_a.flat_to_compact[flat];
            let cb = lookup_b.flat_to_compact[flat];
            if ca != usize::MAX && cb != usize::MAX {
                coherence_index_diff_vol[flat] = diff;
            }
        }
    }
    write_vol("coherence_index_diff", &coherence_index_diff_vol)?;

    for key in &shared_keys {
        write_vol(
            &format!("dpf_{key}_diff_mean"),
            &per_key_diff_mean_vol[key],
        )?;
        write_vol(
            &format!("dpf_{key}_diff_max_abs"),
            &per_key_diff_max_abs_vol[key],
        )?;
        write_vol(&format!("dpf_{key}_a_sum"), &per_key_a_sum_vol[key])?;
        write_vol(&format!("dpf_{key}_b_sum"), &per_key_b_sum_vol[key])?;
        write_vol(&format!("dpf_{key}_sum_diff"), &per_key_sum_diff_vol[key])?;
    }

    if opts.write_comparison_odx {
        let mut builder = OdxBuilder::new(
            header_a.voxel_to_rasmm,
            header_a.dimensions,
            mask_a.to_vec(),
        );
        for peaks in a.voxel_peaks() {
            builder.push_voxel_peaks(peaks);
        }
        if let (Some(verts), Some(faces)) = (a.sphere_vertices(), a.sphere_faces()) {
            builder.set_sphere(verts.to_vec(), faces.to_vec());
        }
        if let Some(sid) = header_a.sphere_id.as_ref() {
            builder.set_sphere_id(sid.clone());
        }

        let dpv_volumes: [(&str, &[f32]); 16] = [
            ("n_fixels_a", &n_fixels_a_vol),
            ("n_fixels_b", &n_fixels_b_vol),
            ("n_fixels_diff", &n_fixels_diff_vol),
            ("n_matched", &n_matched_vol),
            ("n_unmatched_a", &n_unmatched_a_vol),
            ("n_unmatched_b", &n_unmatched_b_vol),
            ("n_a_collisions", &n_a_collisions_vol),
            ("jaccard", &jaccard_vol),
            ("dice", &dice_vol),
            ("mean_match_angle_deg", &mean_angle_vol),
            ("max_match_angle_deg", &max_angle_vol),
            ("n_coherent_a", &n_coherent_a_vol),
            ("n_coherent_b", &n_coherent_b_vol),
            ("n_coherent_mutual", &n_coherent_mutual_vol),
            ("top1_match", &top1_match_vol),
            ("coherence_index_diff", &coherence_index_diff_vol),
        ];
        for (name, vol) in dpv_volumes {
            let compact = compact_volume_to_dpv(vol, mask_a);
            builder.set_dpv_data(name, vec_into_bytes(compact), 1, DType::Float32);
        }
        for key in &shared_keys {
            for (suffix, vol) in [
                ("diff_mean", &per_key_diff_mean_vol[key]),
                ("diff_max_abs", &per_key_diff_max_abs_vol[key]),
                ("a_sum", &per_key_a_sum_vol[key]),
                ("b_sum", &per_key_b_sum_vol[key]),
                ("sum_diff", &per_key_sum_diff_vol[key]),
            ] {
                let name = format!("dpf_{key}_{suffix}");
                let compact = compact_volume_to_dpv(vol, mask_a);
                builder.set_dpv_data(&name, vec_into_bytes(compact), 1, DType::Float32);
            }
        }

        builder.set_dpf_data(
            "match_index_b",
            vec_into_bytes(match_index_b),
            1,
            DType::Int32,
        );
        builder.set_dpf_data(
            "match_angle_deg",
            vec_into_bytes(match_angle_deg),
            1,
            DType::Float32,
        );
        builder.set_dpf_data("match_dp", vec_into_bytes(match_dp), 1, DType::Float32);
        builder.set_dpf_data("is_mutual", is_mutual, 1, DType::UInt8);
        let qc_class_a_bytes: Vec<u8> = qc_a.classes.iter().map(|c| *c as u8).collect();
        builder.set_dpf_data("qc_class_a", qc_class_a_bytes, 1, DType::UInt8);
        builder.set_dpf_data("qc_class_b_matched", qc_class_b_matched, 1, DType::UInt8);
        for key in &shared_keys {
            let a_v = a_values.get(key).cloned().unwrap();
            let b_v = per_key_b_matched.remove(key).unwrap();
            let d_v = per_key_diff.remove(key).unwrap();
            builder.set_dpf_data(
                &format!("{key}_a"),
                vec_into_bytes(a_v),
                1,
                DType::Float32,
            );
            builder.set_dpf_data(
                &format!("{key}_b_matched"),
                vec_into_bytes(b_v),
                1,
                DType::Float32,
            );
            builder.set_dpf_data(
                &format!("{key}_diff"),
                vec_into_bytes(d_v),
                1,
                DType::Float32,
            );
        }
        let dataset = builder.finalize()?;
        let path = out_dir.join("comparison.odx");
        dataset.save_archive(&path)?;
        written.push(path.display().to_string());
    }

    let mean_match_angle_deg = if count_match_angle > 0 {
        Some(sum_match_angle_deg / count_match_angle as f64)
    } else {
        None
    };

    Ok(CompareReport {
        primary_metric: primary,
        coherence_angle_deg: opts.coherence_angle_deg,
        match_angle_deg: opts.match_angle_deg,
        n_voxels_a: a.nb_voxels() as u64,
        n_voxels_b: b.nb_voxels() as u64,
        n_voxels_intersection,
        n_fixels_a: nb_peaks_a as u64,
        n_fixels_b: nb_peaks_b as u64,
        n_mutual_matches: total_mutual_matches,
        n_unmatched_a: total_unmatched_a,
        n_unmatched_b: total_unmatched_b,
        mean_match_angle_deg,
        coherence_index_a: ci_a,
        coherence_index_b: ci_b,
        shared_dpf_keys: shared_keys,
        written_paths: written,
    })
}

struct VoxelLookup {
    flat_to_compact: Vec<usize>,
}

fn build_voxel_lookup(mask: &[u8], dims: [u64; 3], offsets: &[u32]) -> Result<VoxelLookup> {
    let total = (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize);
    if mask.len() != total {
        return Err(OdxError::Format(format!(
            "mask len {} != product of dims {}",
            mask.len(),
            total
        )));
    }
    let mut flat_to_compact = vec![usize::MAX; total];
    let yz = (dims[1] as usize) * (dims[2] as usize);
    let z = dims[2] as usize;
    let mut compact = 0usize;
    for i in 0..dims[0] as usize {
        for j in 0..dims[1] as usize {
            for k in 0..dims[2] as usize {
                let flat = i * yz + j * z + k;
                if mask[flat] != 0 {
                    flat_to_compact[flat] = compact;
                    compact += 1;
                }
            }
        }
    }
    if compact + 1 != offsets.len() {
        return Err(OdxError::Format(format!(
            "mask voxel count {} differs from offsets-1 {}",
            compact,
            offsets.len() - 1
        )));
    }
    Ok(VoxelLookup { flat_to_compact })
}

fn compact_volume_to_dpv(vol: &[f32], mask: &[u8]) -> Vec<f32> {
    let n = mask.iter().filter(|&&v| v != 0).count();
    let mut out = Vec::with_capacity(n);
    for (i, &m) in mask.iter().enumerate() {
        if m != 0 {
            out.push(vol[i]);
        }
    }
    out
}

fn affine_close(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4], tol: f64) -> bool {
    for r in 0..4 {
        for c in 0..4 {
            if (a[r][c] - b[r][c]).abs() > tol {
                return false;
            }
        }
    }
    true
}

fn abs_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).abs()
}

fn write_voxel_scalar_nifti(
    dims: &[usize; 3],
    affine: &[[f64; 4]; 4],
    data: &[f32],
    path: &Path,
) -> Result<()> {
    let array = Array::from_shape_vec(IxDyn(dims), data.to_vec()).map_err(|err| {
        OdxError::Format(format!("failed to shape comparison NIfTI: {err}"))
    })?;
    let header = make_nifti1_header(affine);
    WriterOptions::new(path)
        .reference_header(&header)
        .write_nifti(&array)
        .map_err(|err| {
            OdxError::Format(format!(
                "failed to write NIfTI '{}': {err}",
                path.display()
            ))
        })?;
    Ok(())
}

fn make_nifti1_header(affine: &[[f64; 4]; 4]) -> NiftiHeader {
    let voxel_sizes = [
        (affine[0][0].powi(2) + affine[1][0].powi(2) + affine[2][0].powi(2)).sqrt(),
        (affine[0][1].powi(2) + affine[1][1].powi(2) + affine[2][1].powi(2)).sqrt(),
        (affine[0][2].powi(2) + affine[1][2].powi(2) + affine[2][2].powi(2)).sqrt(),
    ];
    let mut header = NiftiHeader::default();
    header.sform_code = 1;
    header.qform_code = 0;
    header.pixdim[1] = voxel_sizes[0] as f32;
    header.pixdim[2] = voxel_sizes[1] as f32;
    header.pixdim[3] = voxel_sizes[2] as f32;
    header.xyzt_units = 2;
    header.srow_x = [
        affine[0][0] as f32,
        affine[0][1] as f32,
        affine[0][2] as f32,
        affine[0][3] as f32,
    ];
    header.srow_y = [
        affine[1][0] as f32,
        affine[1][1] as f32,
        affine[1][2] as f32,
        affine[1][3] as f32,
    ];
    header.srow_z = [
        affine[2][0] as f32,
        affine[2][1] as f32,
        affine[2][2] as f32,
        affine[2][3] as f32,
    ];
    header
}
