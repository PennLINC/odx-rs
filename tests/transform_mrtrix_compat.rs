//! Tier-2 verification: end-to-end SH and fixel reorientation against
//! MRtrix3's `mrtransform` and `fixeltransform` reference outputs.
//!
//! These tests **skip** unless the MRtrix3 binary test-data tree is
//! available locally (it lives in a separate git repo and is fetched at
//! CMake build-time by the upstream MRtrix3 testing CI). The path can be
//! pointed at via the `ODX_MRTRIX_TEST_DATA` environment variable; in its
//! absence we look at `/tmp/test_data` (where `cargo test` will find a
//! sparse-clone if one was made earlier in the session).
//!
//! Test data repo (Apache-2.0):
//!   <https://github.com/mattcieslak/test_data>
//!   tag: `e3a85f94bf79b0556d940c9ffde3899eb86d7dd8`
//!
//! Fetch instructions:
//!   git clone --depth 1 https://github.com/mattcieslak/test_data /tmp/test_data
//!   git -C /tmp/test_data fetch --depth 1 origin e3a85f94bf79b0556d940c9ffde3899eb86d7dd8
//!   git -C /tmp/test_data checkout FETCH_HEAD
//!
//! Each test reports the maximum element-wise SH-coefficient difference
//! against MRtrix3's reference; the assertion fails only above
//! `MAX_ABS_DIFF`. Tighten the tolerance as the implementation matures.


use std::fs;
use std::path::{Path, PathBuf};

use itk_transforms_rs::{Affine3, TargetGrid, TransformChain};
use nalgebra::Matrix4;

use odx_rs::transform::{apply_transform, TransformOptions};
use odx_rs::OdxDataset;

/// Resolve the MRtrix3 test-data directory, or `None` if not present.
fn test_data_root() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ODX_MRTRIX_TEST_DATA") {
        let path = PathBuf::from(p);
        if path.is_dir() {
            return Some(path);
        }
    }
    let fallback = PathBuf::from("/tmp/test_data");
    if fallback.is_dir() {
        return Some(fallback);
    }
    None
}

/// Macro: skip the rest of a test (return early with a printed note) if the
/// MRtrix3 test data isn't present.
macro_rules! require_test_data {
    () => {
        match test_data_root() {
            Some(p) => p,
            None => {
                eprintln!(
                    "skipping: MRtrix3 test data not found. Set ODX_MRTRIX_TEST_DATA or \
                     clone https://github.com/mattcieslak/test_data into /tmp/test_data."
                );
                return;
            }
        }
    };
}

/// Parse an MRtrix3 textual affine matrix (4×4 RAS+, whitespace-separated).
fn parse_mrtrix_matrix(path: &Path) -> Matrix4<f64> {
    let body = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let nums: Vec<f64> = body
        .split_whitespace()
        .filter_map(|tok| tok.parse::<f64>().ok())
        .collect();
    assert!(
        nums.len() == 12 || nums.len() == 16,
        "MRtrix matrix needs 12 or 16 floats, got {}",
        nums.len()
    );
    let mut m = Matrix4::identity();
    for r in 0..3 {
        for c in 0..4 {
            m[(r, c)] = nums[r * 4 + c];
        }
    }
    m
}

/// Build a TargetGrid from an [`OdxDataset`]'s header.
fn grid_from_odx(d: &OdxDataset) -> TargetGrid {
    TargetGrid::new(d.header().voxel_to_rasmm, d.header().dimensions)
}

/// Compare two SH-bearing ODX datasets voxel-by-voxel. Returns `(matches,
/// max_abs_diff, mean_abs_diff)` over the voxels that are in-mask in both.
struct ShComparison {
    matched_voxels: usize,
    only_in_ours: usize,
    only_in_reference: usize,
    max_abs_diff: f32,
    mean_abs_diff: f64,
}

fn compare_sh_voxelwise(ours: &OdxDataset, reference: &OdxDataset, sh_name: &str) -> ShComparison {
    let dims = ours.header().dimensions;
    assert_eq!(
        dims, reference.header().dimensions,
        "SH compat test requires same grid dimensions"
    );

    let our_sh = ours.sh::<f32>(sh_name).expect("ours has SH");
    let ref_sh = reference.sh::<f32>(sh_name).expect("reference has SH");
    let our_ncols = our_sh.shape().1;
    let ref_ncols = ref_sh.shape().1;
    assert_eq!(
        our_ncols, ref_ncols,
        "SH coeff count differs: ours={our_ncols}, ref={ref_ncols}"
    );

    let our_ijk = ours.compact_to_ijk();
    let ref_ijk = reference.compact_to_ijk();

    // Build (i,j,k) → row index for both, so we can match efficiently.
    use std::collections::HashMap;
    let our_index: HashMap<[u32; 3], usize> =
        our_ijk.iter().enumerate().map(|(i, k)| (*k, i)).collect();
    let ref_index: HashMap<[u32; 3], usize> =
        ref_ijk.iter().enumerate().map(|(i, k)| (*k, i)).collect();

    let mut max_abs_diff = 0.0_f32;
    let mut sum_abs_diff = 0.0_f64;
    let mut count = 0_usize;
    let mut only_in_ours = 0_usize;
    for (key, our_row) in &our_index {
        if let Some(&ref_row) = ref_index.get(key) {
            let our = our_sh.row(*our_row);
            let r = ref_sh.row(ref_row);
            for c in 0..our_ncols {
                let d = (our[c] - r[c]).abs();
                if d > max_abs_diff {
                    max_abs_diff = d;
                }
                sum_abs_diff += d as f64;
            }
            count += our_ncols;
        } else {
            only_in_ours += 1;
        }
    }
    let only_in_reference = ref_index.keys().filter(|k| !our_index.contains_key(*k)).count();
    let mean_abs_diff = if count > 0 { sum_abs_diff / count as f64 } else { f64::NAN };
    ShComparison {
        matched_voxels: our_index.len() - only_in_ours,
        only_in_ours,
        only_in_reference,
        max_abs_diff,
        mean_abs_diff,
    }
}

/// Loose tolerance for the linear-affine test. MRtrix3 itself uses
/// `testing_diff_image -voxel 0.001`. Our default 80-direction reference
/// sphere is denser than strictly necessary for lmax=8, but small numeric
/// differences in interpolation and reorientation produce a slightly
/// larger element-wise spread. The test prints the max diff so the
/// developer can ratchet the tolerance down as the implementation matures.
const MAX_ABS_DIFF_LINEAR: f32 = 0.05;

#[test]
fn sh_linear_affine_matches_mrtransform_out8() {
    let root = require_test_data!();

    let fod_path = root.join("fod.mif");
    let xfm_path = root.join("rotatez.txt");
    let ref_path = root.join("mrtransform").join("out8.mif.gz");

    if !fod_path.exists() || !xfm_path.exists() || !ref_path.exists() {
        eprintln!("skipping: missing one or more of: {fod_path:?}, {xfm_path:?}, {ref_path:?}");
        return;
    }

    // Source FOD as ODX.
    let input = odx_rs::mrtrix::load_mrtrix_sh(&fod_path).expect("load fod.mif");
    // Reference output as ODX.
    let reference = odx_rs::mrtrix::load_mrtrix_sh(&ref_path).expect("load out8.mif.gz");

    // MRtrix's `-linear M -template T`: M is the fixed→moving mapping
    // applied to template (fixed) coordinates to look up the moving image
    // (matching ANTs and most other registration tools' convention). Our
    // TransformChain is also fixed→moving, so we push M directly without
    // inversion.
    let m = parse_mrtrix_matrix(&xfm_path);
    let mut chain = TransformChain::new();
    chain.push_affine(Affine3::from_matrix(m));

    // Target grid = source grid (per `-template fod.mif`).
    let target = grid_from_odx(&input);

    // mrtrix3 `-reorient_fod yes` does NOT modulate by default — match that.
    let opts = TransformOptions {
        modulate_sh: false,
        modulate_fixel: false,
        apsf_dirs: 300,
        ..TransformOptions::default()
    };

    let ours = apply_transform(&input, &chain, &target, &opts).expect("apply_transform");

    let sh_name = ours
        .sh_names()
        .into_iter()
        .next()
        .expect("output should have SH")
        .to_string();
    let cmp = compare_sh_voxelwise(&ours, &reference, &sh_name);

    eprintln!(
        "SH linear compat:  matched_voxels={}  only_in_ours={}  only_in_ref={}  \
         max_abs_diff={:.6}  mean_abs_diff={:.6}",
        cmp.matched_voxels, cmp.only_in_ours, cmp.only_in_reference, cmp.max_abs_diff, cmp.mean_abs_diff,
    );

    assert!(
        cmp.matched_voxels > 0,
        "no overlapping voxels between ours and reference — likely a grid/affine mismatch"
    );
    assert!(
        cmp.max_abs_diff < MAX_ABS_DIFF_LINEAR,
        "max SH coefficient difference {} exceeds tolerance {} (mean={:.6})",
        cmp.max_abs_diff,
        MAX_ABS_DIFF_LINEAR,
        cmp.mean_abs_diff,
    );
}

// --------------------------------------------------------------------------
// Skeleton: nonlinear (warp) test. Marked #[ignore] until odx-rs gains a
// MRtrix .mif warp loader (the warp file is a 4D MIF storing absolute
// world-coordinate destinations per voxel).
// --------------------------------------------------------------------------

#[test]
#[ignore = "MRtrix .mif warp loader not yet implemented; see TODO in test"]
fn sh_nonlinear_warp_matches_mrtransform_out9() {
    let _root = require_test_data!();
    // TODO:
    //   1. Read /tmp/test_data/rotatez_warp.mif (4D MIF, "deformation field"):
    //      each voxel stores the absolute fixed-space coord that the moving
    //      voxel ends up at. Convert to a `DisplacementField` in our
    //      DisplacementField representation by subtracting voxel-center
    //      world coords.
    //   2. Build TransformChain with that warp.
    //   3. Apply to fod.mif, compare with mrtransform/out9.mif.gz.
}

// --------------------------------------------------------------------------
// Skeleton: fixel (warp) test.
// --------------------------------------------------------------------------

#[test]
#[ignore = "MRtrix .mif warp loader not yet implemented (shared with out9 test)"]
fn fixel_nonlinear_warp_matches_fixeltransform_default() {
    let _root = require_test_data!();
    // TODO: load fixel_image/ as ODX, apply rotatez_warp.mif, compare to
    // fixeltransform/default/. Tolerance per upstream: -abs 1e-5.
}
