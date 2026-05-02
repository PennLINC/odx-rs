//! Empirical sanity-check that fixels and SH coefficients actually rotate
//! under the user's real ANTs h5 transform. For a handful of source voxels,
//! we compare:
//!
//!   pushed_dir  = (J_chain_fixel · source_dir).normalized()
//!   actual_dir  = direction recorded in the output ODX after `--mode ants`
//!
//! If these match, fixel rotation is happening. We also verify that for
//! a *non-identity* local Jacobian the directions in fact differ from the
//! source (i.e. it's not a no-op).

use std::path::Path;

use itk_transforms_rs::read_itk_h5;
use nalgebra::{Vector3, Vector4};
use odx_rs::OdxDataset;

fn main() {
    let dir = Path::new("/Users/mcieslak/projects/odx/transform_test_data");
    let acpc = OdxDataset::load(&dir.join("acpc.odx")).unwrap();
    let mni = OdxDataset::load(&dir.join("mni_ants.odx")).unwrap();

    let inv_chain = read_itk_h5(
        &dir.join("sub-0874667_ses-V1_from-MNI152NLin2009cAsym_to-ACPC_mode-image_xfm.h5"),
    )
    .unwrap();

    let acpc_affine = nalgebra::Matrix4::from_row_slice(&[
        acpc.header().voxel_to_rasmm[0][0], acpc.header().voxel_to_rasmm[0][1], acpc.header().voxel_to_rasmm[0][2], acpc.header().voxel_to_rasmm[0][3],
        acpc.header().voxel_to_rasmm[1][0], acpc.header().voxel_to_rasmm[1][1], acpc.header().voxel_to_rasmm[1][2], acpc.header().voxel_to_rasmm[1][3],
        acpc.header().voxel_to_rasmm[2][0], acpc.header().voxel_to_rasmm[2][1], acpc.header().voxel_to_rasmm[2][2], acpc.header().voxel_to_rasmm[2][3],
        0.0, 0.0, 0.0, 1.0,
    ]);

    let acpc_ijk = acpc.compact_to_ijk();
    let acpc_dirs = acpc.directions();
    let acpc_offsets = acpc.offsets();

    // Find a voxel whose source fixel direction differs noticeably from any axis,
    // so a rotation is visually evident.
    let mut printed = 0;
    for (compact_row, ijk) in acpc_ijk.iter().enumerate() {
        let f_start = acpc_offsets[compact_row] as usize;
        let f_end = acpc_offsets[compact_row + 1] as usize;
        if f_end == f_start {
            continue;
        }
        let d_acpc = acpc_dirs[f_start];

        let p_src_v = acpc_affine * Vector4::new(ijk[0] as f64, ijk[1] as f64, ijk[2] as f64, 1.0);
        let p_src = [p_src_v[0], p_src_v[1], p_src_v[2]];
        let j_fixel = inv_chain.jacobian_at(p_src, 0.5);

        // Predicted target direction = (J · d).normalized
        let v = j_fixel * Vector3::new(d_acpc[0] as f64, d_acpc[1] as f64, d_acpc[2] as f64);
        let n = v.norm();
        let predicted = [(v[0] / n) as f32, (v[1] / n) as f32, (v[2] / n) as f32];

        let angle_change_deg = {
            let dot = (d_acpc[0] * predicted[0]
                + d_acpc[1] * predicted[1]
                + d_acpc[2] * predicted[2]) as f64;
            dot.clamp(-1.0, 1.0).acos().to_degrees()
        };
        if angle_change_deg < 1.0 {
            // Skip: local Jacobian is near-identity here, nothing to demonstrate.
            continue;
        }

        // Find pushed fixel in MNI: push p_src into MNI grid.
        let p_mni = inv_chain.map_point(p_src);
        let mni_inv_affine = mni_inv(&mni);
        let v_t = mni_inv_affine * Vector4::new(p_mni[0], p_mni[1], p_mni[2], 1.0);
        let (ti, tj, tk) = (v_t[0].round() as i64, v_t[1].round() as i64, v_t[2].round() as i64);

        // Look up that voxel's compact row in MNI.
        let mni_dims = mni.header().dimensions;
        if ti < 0 || tj < 0 || tk < 0 {
            continue;
        }
        let (ti, tj, tk) = (ti as u64, tj as u64, tk as u64);
        if ti >= mni_dims[0] || tj >= mni_dims[1] || tk >= mni_dims[2] {
            continue;
        }
        let mni_ijk = mni.compact_to_ijk();
        let target_ijk = [ti as u32, tj as u32, tk as u32];
        let mni_compact = match mni_ijk.iter().position(|&p| p == target_ijk) {
            Some(c) => c,
            None => continue,
        };
        let mni_offsets = mni.offsets();
        let m_start = mni_offsets[mni_compact] as usize;
        let m_end = mni_offsets[mni_compact + 1] as usize;
        if m_end == m_start {
            continue;
        }
        // Find the closest MNI fixel direction to our predicted one.
        let mni_dirs = mni.directions();
        let mut best_dot = -2.0_f32;
        let mut best = mni_dirs[m_start];
        for f in m_start..m_end {
            let d = mni_dirs[f];
            let dot = (d[0] * predicted[0] + d[1] * predicted[1] + d[2] * predicted[2]).abs();
            if dot > best_dot {
                best_dot = dot;
                best = d;
            }
        }
        let actual_vs_predicted_deg = best_dot.clamp(-1.0, 1.0).acos().to_degrees();

        println!("--- ACPC voxel {ijk:?} ---");
        println!("  ACPC fixel dir:           [{:.3}, {:.3}, {:.3}]", d_acpc[0], d_acpc[1], d_acpc[2]);
        println!("  Local J ⋅ d (predicted):  [{:.3}, {:.3}, {:.3}]", predicted[0], predicted[1], predicted[2]);
        println!("  MNI nearest target dir:   [{:.3}, {:.3}, {:.3}]", best[0], best[1], best[2]);
        println!("  Source-to-target rotation: {angle_change_deg:.2}°");
        println!("  Predicted-vs-actual:       {actual_vs_predicted_deg:.4}°  (should be ~0 if reorientation matches)");
        println!();

        printed += 1;
        if printed >= 5 {
            break;
        }
    }
}

fn mni_inv(d: &OdxDataset) -> nalgebra::Matrix4<f64> {
    let a = d.header().voxel_to_rasmm;
    let m = nalgebra::Matrix4::from_row_slice(&[
        a[0][0], a[0][1], a[0][2], a[0][3],
        a[1][0], a[1][1], a[1][2], a[1][3],
        a[2][0], a[2][1], a[2][2], a[2][3],
        0.0, 0.0, 0.0, 1.0,
    ]);
    m.try_inverse().unwrap()
}
