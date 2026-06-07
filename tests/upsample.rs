//! Integration tests for `odx_rs::upsample`.

use odx_rs::header::Header;
use odx_rs::transform::upsample::{compute_upsampled_grid, upsample, UpsampleOptions};
use odx_rs::{DType, OdxBuilder, OdxDataset};

/// Build a small synthetic ODX on a 4×4×4 grid at 2 mm isotropic spacing with
/// lmax=2 SH (6 coefficients) and a GFA DPV. Three voxels are in-mask.
fn make_sh_odx_2mm() -> OdxDataset {
    let affine = [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let dims = [4u64, 4, 4];
    let total = (dims[0] * dims[1] * dims[2]) as usize;
    let mut mask = vec![0u8; total];
    mask[0] = 1; // (0,0,0)
    mask[1] = 1; // (0,0,1)
    mask[2] = 1; // (0,0,2)

    let mut b = OdxBuilder::new(affine, dims, mask);
    b.set_sh_info(2, "tournier07".to_string());

    // 6 coefficients for lmax=2 (even-only tournier): l=0 term = 0.5/sqrt(4pi)
    // gives a roughly isotropic ODF with amplitude ~0.5.
    let sh: Vec<f32> = vec![
        0.5, 0.0, 0.0, 0.0, 0.0, 0.0, // voxel 0
        0.4, 0.0, 0.0, 0.0, 0.0, 0.0, // voxel 1
        0.6, 0.0, 0.0, 0.0, 0.0, 0.0, // voxel 2
    ];
    b.set_sh_data(
        "coefficients",
        bytemuck::cast_slice(&sh).to_vec(),
        6,
        DType::Float32,
    );

    let gfa: Vec<f32> = vec![0.3, 0.5, 0.7];
    b.set_dpv_data(
        "gfa",
        bytemuck::cast_slice(&gfa).to_vec(),
        1,
        DType::Float32,
    );

    b.skip_all_peaks();
    b.compute_peaks(None, Default::default()).unwrap();
    b.finalize().unwrap()
}

/// Build a minimal ODX with a dense ODF array to test the rejection path.
fn make_odf_odx() -> OdxDataset {
    let affine = Header::identity_affine();
    let dims = [2u64, 2, 2];
    let mask = vec![1u8; 8];
    let mut b = OdxBuilder::new(affine, dims, mask);
    b.skip_all_peaks();
    // ODF data needs a sphere. Use a single stub vertex and face so finalize
    // succeeds; the rejection must come from upsample(), not finalize().
    b.set_sphere(
        vec![[0.0f32, 0.0, 1.0]],
        vec![[0u32, 0, 0]],
    );
    b.set_odf_data("amplitudes", vec![0u8; 8 * 4], 1, DType::Float32);
    b.finalize().unwrap()
}

// ── compute_upsampled_grid ────────────────────────────────────────────────────

#[test]
fn grid_2mm_to_1mm_doubles_dims() {
    let affine = [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let grid = compute_upsampled_grid(&affine, [10, 12, 8], 1.0);
    assert_eq!(grid.dims, [20, 24, 16]);
    // New axis vectors should have magnitude 1.0.
    for axis in 0..3 {
        let vx = grid.affine[0][axis];
        let vy = grid.affine[1][axis];
        let vz = grid.affine[2][axis];
        let spacing = (vx * vx + vy * vy + vz * vz).sqrt();
        assert!((spacing - 1.0).abs() < 1e-9, "axis {axis} spacing={spacing}");
    }
    // Origin should be unchanged.
    assert_eq!(grid.affine[0][3], 0.0);
    assert_eq!(grid.affine[1][3], 0.0);
    assert_eq!(grid.affine[2][3], 0.0);
}

#[test]
fn grid_preserves_physical_extent() {
    let affine = [
        [1.25, 0.0, 0.0, -90.0],
        [0.0, 1.25, 0.0, -126.0],
        [0.0, 0.0, 1.25, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let dims = [144u64, 172, 120];
    let old_extent = [
        dims[0] as f64 * 1.25,
        dims[1] as f64 * 1.25,
        dims[2] as f64 * 1.25,
    ];

    let grid = compute_upsampled_grid(&affine, dims, 1.0);
    for axis in 0..3 {
        let new_extent = grid.dims[axis] as f64 * 1.0;
        assert!(
            new_extent >= old_extent[axis] - 1e-6,
            "axis {axis}: new extent {new_extent:.2} < old {:.2}",
            old_extent[axis]
        );
        // Should not overshoot by more than one voxel's worth.
        assert!(
            new_extent < old_extent[axis] + 1.0 + 1e-6,
            "axis {axis}: new extent {new_extent:.2} overshoots by more than 1 voxel"
        );
    }
}

// ── upsample ─────────────────────────────────────────────────────────────────

#[test]
fn upsample_2x_increases_voxel_count() {
    let input = make_sh_odx_2mm();
    let out = upsample(&input, 1.0, &UpsampleOptions::default()).unwrap();

    // Output dims should double on each axis.
    assert_eq!(out.header().dimensions, [8, 8, 8]);
    // More in-mask voxels than input.
    assert!(
        out.header().nb_voxels > input.header().nb_voxels,
        "output nb_voxels {} ≤ input {}",
        out.header().nb_voxels,
        input.header().nb_voxels
    );
    // SH metadata preserved.
    assert_eq!(out.sh_names(), input.sh_names());
    assert_eq!(out.header().sh_order, input.header().sh_order);
    assert_eq!(out.header().sh_basis, input.header().sh_basis);
    // Peaks were found.
    assert!(out.header().nb_peaks > 0);
    // DPV preserved.
    assert!(out.dpv_names().contains(&"gfa"));
}

#[test]
fn upsample_dpv_is_interpolated() {
    // Build an ODX with two adjacent in-mask voxels at gfa=0.0 and gfa=1.0.
    let affine = [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let dims = [4u64, 1, 1];
    let mut mask = vec![0u8; 4];
    mask[0] = 1; // voxel at i=0 → world x=0
    mask[1] = 1; // voxel at i=1 → world x=2

    let mut b = OdxBuilder::new(affine, dims, mask);
    b.set_sh_info(2, "tournier07".to_string());
    // Both voxels get identical isotropic SH so the peak finder succeeds.
    let sh: Vec<f32> = vec![0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0];
    b.set_sh_data("coefficients", bytemuck::cast_slice(&sh).to_vec(), 6, DType::Float32);
    let gfa: Vec<f32> = vec![0.0, 1.0];
    b.set_dpv_data("gfa", bytemuck::cast_slice(&gfa).to_vec(), 1, DType::Float32);
    b.skip_all_peaks();
    b.compute_peaks(None, Default::default()).unwrap();
    let input = b.finalize().unwrap();

    let out = upsample(&input, 1.0, &UpsampleOptions::default()).unwrap();

    // Output dims: 4 voxels × 2mm / 1mm = 8 voxels along i.
    assert_eq!(out.header().dimensions[0], 8);

    // The output DPV at the midpoint between the two source voxels (i=0,i=1)
    // should be between 0 and 1 (linear interpolation).
    let gfa_dense = odx_rs::densify::densify_scalar_dpv(&out, "gfa")
        .expect("gfa DPV missing in output");
    let gfa_arr: Vec<f32> = gfa_dense.iter().copied().collect();
    let in_range = gfa_arr.iter().all(|&v| v >= 0.0 && v <= 1.0 + 1e-5);
    assert!(in_range, "output GFA values outside [0,1]: {gfa_arr:?}");
    // At least one value is strictly between 0 and 1 (the interpolated midpoint).
    let has_intermediate = gfa_arr.iter().any(|&v| v > 0.01 && v < 0.99);
    assert!(has_intermediate, "no interpolated GFA value found: {gfa_arr:?}");
}

#[test]
fn upsample_rejects_dense_odf() {
    let input = make_odf_odx();
    let result = upsample(&input, 1.0, &UpsampleOptions::default());
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("Dense ODF"),
        "error should mention 'Dense ODF', got: {msg}"
    );
}
