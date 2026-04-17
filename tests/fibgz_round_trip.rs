use std::path::Path;

use odx_rs::{dsistudio, read_reference_affine, OdxDataset};

const FIB_PATH: &str =
    "../test_data/sub-NDARAE199TDD_ses-1_acq-64dirVARIANTVar1e_space-ACPC_model-ss3t_dwimap.fib.gz";
const REF_AFFINE_PATH: &str =
    "../test_data/sub-NDARAE199TDD_ses-1_acq-64dirVARIANTVar1e_space-ACPC_model-tensor_param-fa_dwimap.nii.gz";

fn fixture_reference_affine() -> [[f64; 4]; 4] {
    read_reference_affine(Path::new(REF_AFFINE_PATH)).unwrap()
}

#[test]
fn load_fibgz_basic() {
    let path = Path::new(FIB_PATH);
    let reference = Path::new(REF_AFFINE_PATH);
    if !path.exists() || !reference.exists() {
        eprintln!("skipping: test data not found at {FIB_PATH}");
        return;
    }

    let odx = dsistudio::load_fibgz(path, Some(fixture_reference_affine())).unwrap();

    // Dimensions should be 80x98x85
    let hdr = odx.header();
    assert_eq!(hdr.dimensions, [80, 98, 85]);

    // Volume size
    let vol_size = 80 * 98 * 85;
    assert_eq!(odx.mask().len(), vol_size);

    // Mask count should match nb_voxels
    let mask_count = odx.mask().iter().filter(|&&v| v != 0).count();
    assert_eq!(mask_count, odx.nb_voxels());

    // Should have peaks
    assert!(odx.nb_peaks() > 0);
    assert!(odx.nb_voxels() > 0);

    // Offsets sentinel
    let offsets = odx.offsets();
    assert_eq!(offsets.len(), odx.nb_voxels() + 1);
    assert_eq!(*offsets.last().unwrap() as usize, odx.nb_peaks());

    // Directions should all be unit-ish vectors
    for dir in odx.directions() {
        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        assert!(
            (len - 1.0).abs() < 0.05,
            "direction not unit length: {dir:?} len={len}"
        );
    }

    // Sphere
    let verts = odx.sphere_vertices().unwrap();
    assert_eq!(verts.len(), 642);
    let faces = odx.sphere_faces().unwrap();
    assert_eq!(faces.len(), 1280);

    // ODF amplitudes
    let odf = odx.odf::<f32>("amplitudes").unwrap();
    assert_eq!(odf.nrows(), odx.nb_voxels());
    assert_eq!(odf.ncols(), 321); // hemisphere directions

    // DPF amplitude
    let amp = odx.scalar_dpf_f32("amplitude").unwrap();
    assert_eq!(amp.len(), odx.nb_peaks());
    assert!(amp.iter().all(|&v| v > 0.0));

    // DPV scalars
    let gfa = odx.scalar_dpv_f32("gfa").unwrap();
    assert_eq!(gfa.len(), odx.nb_voxels());

    println!(
        "fib.gz loaded: {} voxels, {} peaks, {} ODF dirs",
        odx.nb_voxels(),
        odx.nb_peaks(),
        odf.ncols()
    );
}

#[test]
fn fibgz_round_trip_via_odx() {
    let path = Path::new(FIB_PATH);
    let reference = Path::new(REF_AFFINE_PATH);
    if !path.exists() || !reference.exists() {
        eprintln!("skipping: test data not found at {FIB_PATH}");
        return;
    }

    let odx = dsistudio::load_fibgz(path, Some(fixture_reference_affine())).unwrap();

    // Save to ODX directory
    let tmpdir = tempfile::TempDir::new().unwrap();
    let odx_dir = tmpdir.path().join("test.odx");
    odx.save_directory(&odx_dir).unwrap();

    // Reload
    let reloaded = OdxDataset::load(&odx_dir).unwrap();
    assert_eq!(reloaded.nb_voxels(), odx.nb_voxels());
    assert_eq!(reloaded.nb_peaks(), odx.nb_peaks());
    assert_eq!(reloaded.offsets(), odx.offsets());

    // Compare directions
    let orig_dirs = odx.directions();
    let reload_dirs = reloaded.directions();
    assert_eq!(orig_dirs.len(), reload_dirs.len());
    for (a, b) in orig_dirs.iter().zip(reload_dirs.iter()) {
        for j in 0..3 {
            assert!(
                (a[j] - b[j]).abs() < 1e-6,
                "direction mismatch: {a:?} vs {b:?}"
            );
        }
    }

    // Compare ODF amplitudes
    let orig_odf = odx.odf::<f32>("amplitudes").unwrap();
    let reload_odf = reloaded.odf::<f32>("amplitudes").unwrap();
    assert_eq!(orig_odf.shape(), reload_odf.shape());
    for i in 0..orig_odf.nrows() {
        for (a, b) in orig_odf.row(i).iter().zip(reload_odf.row(i).iter()) {
            assert!((a - b).abs() < 1e-6, "ODF mismatch at row {i}: {a} vs {b}");
        }
    }
}

#[test]
fn fibgz_round_trip_to_fib() {
    let path = Path::new(FIB_PATH);
    let reference = Path::new(REF_AFFINE_PATH);
    if !path.exists() || !reference.exists() {
        eprintln!("skipping: test data not found at {FIB_PATH}");
        return;
    }

    let odx = dsistudio::load_fibgz(path, Some(fixture_reference_affine())).unwrap();

    // Write back to fib.gz
    let tmpdir = tempfile::TempDir::new().unwrap();
    let fib_out = tmpdir.path().join("round_trip.fib.gz");
    dsistudio::save_fibgz(&odx, &fib_out).unwrap();

    // Reload the written fib.gz
    let odx2 = dsistudio::load_fibgz(&fib_out, None).unwrap();

    assert_eq!(odx2.header().dimensions, odx.header().dimensions);
    assert_eq!(odx2.nb_voxels(), odx.nb_voxels());

    // Peak counts may differ slightly due to sphere index discretization,
    // but should be close
    let peak_diff = (odx2.nb_peaks() as isize - odx.nb_peaks() as isize).unsigned_abs();
    let peak_ratio = peak_diff as f64 / odx.nb_peaks() as f64;
    assert!(
        peak_ratio < 0.01,
        "peak count changed too much: {} -> {} (diff {peak_diff})",
        odx.nb_peaks(),
        odx2.nb_peaks()
    );

    // ODF amplitudes should be preserved exactly
    let orig_odf = odx.odf::<f32>("amplitudes").unwrap();
    let rt_odf = odx2.odf::<f32>("amplitudes").unwrap();
    assert_eq!(orig_odf.shape(), rt_odf.shape());

    let mut max_diff = 0.0f32;
    for i in 0..orig_odf.nrows() {
        for (a, b) in orig_odf.row(i).iter().zip(rt_odf.row(i).iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
    }
    assert!(max_diff < 1e-5, "ODF max difference: {max_diff}");

    println!(
        "fib.gz round trip: {} voxels, {} -> {} peaks, ODF max diff {max_diff}",
        odx2.nb_voxels(),
        odx.nb_peaks(),
        odx2.nb_peaks()
    );
}

#[test]
fn fz_round_trip_to_odx() {
    let path = Path::new(FIB_PATH);
    let reference = Path::new(REF_AFFINE_PATH);
    if !path.exists() || !reference.exists() {
        eprintln!("skipping: test data not found at {FIB_PATH}");
        return;
    }

    let odx = dsistudio::load_fibgz(path, Some(fixture_reference_affine())).unwrap();

    let tmpdir = tempfile::TempDir::new().unwrap();
    let fz_out = tmpdir.path().join("round_trip.fz");
    dsistudio::save_fz(&odx, &fz_out).unwrap();

    let odx2 = dsistudio::load_fz(&fz_out, None).unwrap();
    assert_eq!(odx2.header().dimensions, odx.header().dimensions);
    assert_eq!(odx2.nb_voxels(), odx.nb_voxels());
    assert_eq!(odx2.nb_peaks(), odx.nb_peaks());
    assert_eq!(odx2.offsets(), odx.offsets());
    assert_eq!(
        odx2.header().nb_sphere_vertices,
        odx.header().nb_sphere_vertices
    );
    assert_eq!(odx2.header().nb_sphere_faces, odx.header().nb_sphere_faces);

    let orig_odf = odx.odf::<f32>("amplitudes").unwrap();
    let rt_odf = odx2.odf::<f32>("amplitudes").unwrap();
    assert_eq!(orig_odf.shape(), rt_odf.shape());
    let max_diff = orig_odf
        .as_flat_slice()
        .iter()
        .zip(rt_odf.as_flat_slice())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 0.02, "fz ODF max difference: {max_diff}");
}

#[test]
fn fibgz_fixture_without_reference_affine_fails() {
    let path = Path::new(FIB_PATH);
    if !path.exists() {
        eprintln!("skipping: test data not found at {FIB_PATH}");
        return;
    }

    let err = dsistudio::load_fibgz(path, None).unwrap_err();
    assert!(
        err.to_string()
            .contains("DSI Studio file has no spatial affine ('trans' field)"),
        "unexpected error: {err}"
    );
}
