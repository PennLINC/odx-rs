//! Integration tests for `odx_rs::transform::apply_transform`.
//!
//! These exercise the per-voxel pipeline against synthetic ODX inputs and
//! manually-constructed `TransformChain`s — no h5 files needed.


use itk_transforms_rs::{Affine3, TargetGrid, TransformChain};
use nalgebra::Matrix4;
use odx_rs::transform::{apply_transform, TransformOptions};
use odx_rs::{DType, Header, OdxBuilder, OdxDataset};

/// Build a 3-voxel ODX on a 4×3×2 grid with a sphere, fixels, and DPF
/// amplitudes — same pattern as `tests/round_trip.rs::make_test_odx`.
fn make_test_odx_no_sh() -> OdxDataset {
    let dims = [4u64, 3, 2];
    let total = (dims[0] * dims[1] * dims[2]) as usize;
    let mut mask = vec![0u8; total];
    mask[0 * 3 * 2 + 0 * 2 + 0] = 1; // (0,0,0)
    mask[1 * 3 * 2 + 0 * 2 + 0] = 1; // (1,0,0)
    mask[2 * 3 * 2 + 1 * 2 + 0] = 1; // (2,1,0)

    let affine = Header::identity_affine();
    let mut stream = OdxBuilder::new(affine, dims, mask);
    stream.push_voxel_peaks(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    stream.push_voxel_peaks(&[[0.0, 0.0, 1.0]]);
    stream.push_voxel_peaks(&[[0.577_350_3, 0.577_350_3, 0.577_350_3]]);

    let amplitude: Vec<f32> = vec![0.8, 0.6, 0.9, 0.7];
    stream.set_dpf_data(
        "amplitude",
        bytemuck::cast_slice(&amplitude).to_vec(),
        1,
        DType::Float32,
    );

    let gfa: Vec<f32> = vec![0.3, 0.5, 0.7];
    stream.set_dpv_data(
        "gfa",
        bytemuck::cast_slice(&gfa).to_vec(),
        1,
        DType::Float32,
    );

    stream.finalize().unwrap()
}

fn identity_chain() -> TransformChain {
    let mut c = TransformChain::new();
    c.push_affine(Affine3::identity());
    c
}

fn target_grid_identity(dims: [u64; 3]) -> TargetGrid {
    TargetGrid::from_matrix(Matrix4::identity(), dims)
}

#[test]
fn identity_affine_round_trips_directions_and_dpf() {
    let input = make_test_odx_no_sh();
    let chain = identity_chain();
    let grid = target_grid_identity([4, 3, 2]);
    let opts = TransformOptions {
        modulate_sh: false,
        modulate_fixel: false,
        ..TransformOptions::default()
    };
    let out = apply_transform(&input, &chain, &grid, &opts).unwrap();

    assert_eq!(out.header().nb_voxels, input.header().nb_voxels);
    assert_eq!(out.header().nb_peaks, input.header().nb_peaks);

    // Mask should be byte-exact.
    assert_eq!(out.mask(), input.mask());

    // Directions should be byte-exact (identity reorientation).
    for (a, b) in out.directions().iter().zip(input.directions().iter()) {
        for i in 0..3 {
            assert!(
                (a[i] - b[i]).abs() < 1e-6,
                "direction mismatch: {a:?} vs {b:?}"
            );
        }
    }

    // DPF amplitudes should round-trip without modulation.
    let amp = out.scalar_dpf_f32("amplitude").unwrap();
    assert_eq!(amp, vec![0.8, 0.6, 0.9, 0.7]);
}

#[test]
fn identity_with_modulation_is_unit_factor() {
    let input = make_test_odx_no_sh();
    let chain = identity_chain();
    let grid = target_grid_identity([4, 3, 2]);
    let opts = TransformOptions {
        modulate_sh: true,
        modulate_fixel: true,
        ..TransformOptions::default()
    };
    let out = apply_transform(&input, &chain, &grid, &opts).unwrap();

    // det(I) = 1, so modulation doesn't change anything.
    let amp = out.scalar_dpf_f32("amplitude").unwrap();
    let want = [0.8_f32, 0.6, 0.9, 0.7];
    for (got, want) in amp.iter().zip(want.iter()) {
        assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
    }
}

#[test]
fn translation_shifts_voxels_keeps_directions() {
    // Move one voxel in +x: target voxel (i,j,k) should pull from source
    // (i-1, j, k). Apply to a 4×3×2 grid where source voxels are at
    // i = 0, 1, 2 → target voxels in mask should be at i = 1, 2, 3.
    let input = make_test_odx_no_sh();

    // Composite chain maps fixed → moving, so for "shift output +x by 1
    // voxel" the chain (fixed→moving) is "subtract 1 from x".
    let mut shift = Matrix4::identity();
    shift[(0, 3)] = -1.0;
    let mut c = TransformChain::new();
    c.push_affine(Affine3::from_matrix(shift));
    let grid = target_grid_identity([4, 3, 2]);

    let opts = TransformOptions {
        modulate_sh: false,
        modulate_fixel: false,
        ..TransformOptions::default()
    };
    let out = apply_transform(&input, &c, &grid, &opts).unwrap();

    // Output mask should be in voxels (1,0,0), (2,0,0), (3,1,0).
    assert_eq!(out.header().nb_voxels, 3);
    let mask = out.mask();
    assert_eq!(mask[1 * 3 * 2 + 0 * 2 + 0], 1, "(1,0,0) should be on");
    assert_eq!(mask[2 * 3 * 2 + 0 * 2 + 0], 1, "(2,0,0) should be on");
    assert_eq!(mask[3 * 3 * 2 + 1 * 2 + 0], 1, "(3,1,0) should be on");
    assert_eq!(mask[0 * 3 * 2 + 0 * 2 + 0], 0, "(0,0,0) should be off");

    // Directions reorient by J_fwd = identity.linear()⁻¹ = identity, so
    // they should be byte-exact.
    for (a, b) in out.directions().iter().zip(input.directions().iter()) {
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < 1e-6);
        }
    }
}

#[test]
fn pure_rotation_z_reorients_fixels() {
    // 90° rotation about z. Fixed→moving rotation: a fiber at +x in moving
    // appears at +y in fixed (and vice versa). So the source's +x direction
    // should come out as +y in the output, etc.
    //
    // Composite chain maps fixed → moving. We want target voxels to pull
    // from source voxels rotated by Rz(-90°) in fixed→moving direction (so
    // that resampling at fixed coords lands on source coords). Then
    // J_chain = Rz(-90°), J_fwd = Rz(+90°), and reoriented direction is
    // J_fwd · d.
    //
    // Use a single-voxel ODX so the rotation doesn't move us out of the
    // tiny 4×3×2 grid.
    let dims = [3u64, 3, 3];
    let total = (dims[0] * dims[1] * dims[2]) as usize;
    let mut mask = vec![0u8; total];
    let center_flat = 1 * 3 * 3 + 1 * 3 + 1; // (1,1,1)
    mask[center_flat] = 1;

    // Affine that puts voxel (1,1,1) at the origin (so rotation is in-place):
    // voxel→world: translate by -(1,1,1).
    let mut affine = Header::identity_affine();
    affine[0][3] = -1.0;
    affine[1][3] = -1.0;
    affine[2][3] = -1.0;

    let mut b = OdxBuilder::new(affine, dims, mask);
    b.push_voxel_peaks(&[[1.0, 0.0, 0.0]]);
    b.set_dpf_data(
        "amplitude",
        bytemuck::cast_slice(&[1.0_f32]).to_vec(),
        1,
        DType::Float32,
    );
    let input = b.finalize().unwrap();

    // Build chain = Rz(-90°): fixed→moving rotation by -90° about z.
    let theta = -std::f64::consts::FRAC_PI_2;
    let (s, c) = (theta.sin(), theta.cos());
    let mut m = Matrix4::identity();
    m[(0, 0)] = c;
    m[(0, 1)] = -s;
    m[(1, 0)] = s;
    m[(1, 1)] = c;
    let mut chain = TransformChain::new();
    chain.push_affine(Affine3::from_matrix(m));

    // Target grid: same as input.
    let grid = TargetGrid::from_matrix(
        Matrix4::from_row_slice(&[
            affine[0][0], affine[0][1], affine[0][2], affine[0][3],
            affine[1][0], affine[1][1], affine[1][2], affine[1][3],
            affine[2][0], affine[2][1], affine[2][2], affine[2][3],
            affine[3][0], affine[3][1], affine[3][2], affine[3][3],
        ]),
        dims,
    );

    let opts = TransformOptions {
        modulate_sh: false,
        modulate_fixel: false,
        ..TransformOptions::default()
    };
    let out = apply_transform(&input, &chain, &grid, &opts).unwrap();

    // The center voxel should still be the only mask voxel (rotation at origin).
    assert_eq!(out.header().nb_voxels, 1);
    assert_eq!(out.header().nb_peaks, 1);

    // The fixel direction +x in moving frame should map to +y in fixed frame.
    let d = out.directions()[0];
    assert!(d[0].abs() < 1e-5, "x: got {}", d[0]);
    assert!((d[1] - 1.0).abs() < 1e-5, "y: got {}", d[1]);
    assert!(d[2].abs() < 1e-5, "z: got {}", d[2]);
}

#[test]
fn modulation_scales_amplitudes_under_uniform_scaling() {
    // 2× isotropic scaling moving→fixed. Each fixed-space mm³ comes from
    // 1/8 mm³ of moving-space → AFD-preserving modulation should shrink
    // amplitudes by 1/8.
    //
    // Composite chain (fixed→moving) is 1/2× scaling: J_chain = diag(1/2),
    // det(J_chain) = 1/8. Modulation factor for fixel amps = det(J_chain).
    let input = make_test_odx_no_sh();

    let mut m = Matrix4::identity();
    m[(0, 0)] = 0.5;
    m[(1, 1)] = 0.5;
    m[(2, 2)] = 0.5;
    let mut chain = TransformChain::new();
    chain.push_affine(Affine3::from_matrix(m));

    // Target grid covers ~2× the source range (so the source is interior).
    // Voxel size 1 mm, dims 8×6×4 in target.
    let grid = target_grid_identity([8, 6, 4]);

    let opts = TransformOptions {
        modulate_sh: true,
        modulate_fixel: true,
        ..TransformOptions::default()
    };
    let out = apply_transform(&input, &chain, &grid, &opts).unwrap();

    // NN pull at 2× upsampling can duplicate fixels — multiple target voxels
    // map back to the same source voxel. So we don't assert a 1-to-1 row
    // count; instead we verify every output amplitude lives in the set of
    // {input × 1/8}, i.e. the modulation factor was applied uniformly.
    let valid = [0.8_f32 * 0.125, 0.6 * 0.125, 0.9 * 0.125, 0.7 * 0.125];
    let amp_out = out.scalar_dpf_f32("amplitude").unwrap();
    assert!(amp_out.len() >= 4, "expected >=4 fixel amps, got {}", amp_out.len());
    for &got in &amp_out {
        let matches = valid.iter().any(|&v| (got - v).abs() < 1e-5);
        assert!(matches, "amplitude {got} is not a 1/8-modulated copy of any input amp");
    }
}

#[test]
fn empty_dpf_modulation_preserves_unmodulated_field() {
    // Add a non-amplitude DPF (e.g. "trash") that should *not* be modulated
    // even when modulate_fixel is on.
    let dims = [3u64, 3, 3];
    let total = (dims[0] * dims[1] * dims[2]) as usize;
    let mut mask = vec![0u8; total];
    mask[0] = 1;
    let mut b = OdxBuilder::new(Header::identity_affine(), dims, mask);
    b.push_voxel_peaks(&[[1.0_f32, 0.0, 0.0]]);
    b.set_dpf_data(
        "trash",
        bytemuck::cast_slice(&[1.0_f32]).to_vec(),
        1,
        DType::Float32,
    );
    let input = b.finalize().unwrap();

    // 0.5× scaling chain → det 0.125. With modulation on, "amplitude"
    // would scale; "trash" should not.
    let mut m = Matrix4::identity();
    m[(0, 0)] = 0.5;
    m[(1, 1)] = 0.5;
    m[(2, 2)] = 0.5;
    let mut chain = TransformChain::new();
    chain.push_affine(Affine3::from_matrix(m));
    let grid = target_grid_identity([3, 3, 3]);

    let opts = TransformOptions {
        modulate_sh: true,
        modulate_fixel: true,
        ..TransformOptions::default()
    };
    let out = apply_transform(&input, &chain, &grid, &opts).unwrap();
    let trash = out.scalar_dpf_f32("trash").unwrap();
    assert!(!trash.is_empty());
    for &v in &trash {
        assert!((v - 1.0).abs() < 1e-6, "trash should be unmodulated, got {v}");
    }
}
