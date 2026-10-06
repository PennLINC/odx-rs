//! Integration tests for `odx_rs::apply_graddev` and the DSI Studio `dir{p}`
//! write-back it relies on. Everything is built in-process.
//!
//! Convention under test: the file stores `T` row-major with
//! `g_eff = Tᵀ g`; so for a physical operator `A` (`g_eff = A g`) the file
//! holds `T = Aᵀ`, and a fixel estimated at `u` truly lies at
//! `normalize(T⁻¹ u) = normalize(A u)`.

mod common;

use odx_rs::formats::dsistudio_odf8;
use odx_rs::formats::mat4;
use odx_rs::interop::{
    save_dsistudio_from_odx, DenseOdfMode, DsistudioFormat, MrtrixToDsistudioOptions, PeakSource,
    Z0Policy,
};
use odx_rs::sphere_lookup::nearest_vertex;
use odx_rs::{
    apply_graddev, dsistudio, mrtrix_sh, DType, GradDevField, GradDevOptions, Header,
    IdentityPolicy, OdxBuilder, OdxDataset,
};

type Mat = [[f64; 3]; 3];

const DIMS: [u64; 3] = [3, 3, 3];
const NVOX: usize = 27;
const IDENTITY: Mat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn rot_x(deg: f64) -> Mat {
    let (s, c) = deg.to_radians().sin_cos();
    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}

fn rot_z(deg: f64) -> Mat {
    let (s, c) = deg.to_radians().sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

fn transpose(m: Mat) -> Mat {
    let mut t = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            t[r][c] = m[c][r];
        }
    }
    t
}

fn mat_mul(a: Mat, b: Mat) -> Mat {
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = (0..3).map(|k| a[r][k] * b[k][c]).sum();
        }
    }
    out
}

fn unit(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / n, v[1] / n, v[2] / n]
}

/// `normalize(M · v)`.
fn apply(m: &Mat, v: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for r in 0..3 {
        out[r] = (0..3).map(|c| m[r][c] as f32 * v[c]).sum();
    }
    unit(out)
}

/// Antipodally symmetric angle in degrees.
fn angle_deg(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
        .abs()
        .clamp(0.0, 1.0);
    dot.acos().to_degrees()
}

fn bytes<T: bytemuck::Pod>(v: &[T]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}

fn uniform_field(t: Mat, affine: [[f64; 4]; 4], policy: IdentityPolicy) -> GradDevField {
    let mut data = Vec::with_capacity(NVOX * 9);
    for _ in 0..NVOX {
        for row in &t {
            for &v in row {
                data.push(v as f32);
            }
        }
    }
    GradDevField::from_parts([3, 3, 3], affine, data, policy).unwrap()
}

/// `|v·dir|^8` on the DSI Studio odf8 hemisphere — a degree-8 polynomial, so
/// it lies exactly in the span of even SH up to lmax 8.
fn fiber_amplitudes(dir: [f32; 3]) -> Vec<f32> {
    dsistudio_odf8::hemisphere_vertices_ras()
        .iter()
        .map(|v| {
            (v[0] * dir[0] + v[1] * dir[1] + v[2] * dir[2])
                .abs()
                .powi(8)
        })
        .collect()
}

fn fiber_sh(dir: [f32; 3]) -> Vec<f32> {
    mrtrix_sh::fit_from_amplitudes(
        &fiber_amplitudes(dir),
        dsistudio_odf8::hemisphere_vertices_ras(),
        8,
    )
    .unwrap()
}

/// 3×3×3 fully-masked dataset on the identity grid; every voxel carries one
/// fixel along `dir` and, optionally, the matching SH row / dense ODF row.
fn make_dataset(dir: [f32; 3], with_sh: bool, with_odf: bool) -> OdxDataset {
    let mut b = OdxBuilder::new(Header::identity_affine(), DIMS, vec![1u8; NVOX]);
    for _ in 0..NVOX {
        b.push_voxel_peaks(&[dir]);
    }
    b.set_dpf_data("amplitude", bytes(&[1.0f32; NVOX]), 1, DType::Float32);
    if with_sh {
        let row = fiber_sh(dir);
        let mut all = Vec::with_capacity(NVOX * row.len());
        for _ in 0..NVOX {
            all.extend_from_slice(&row);
        }
        b.set_sh_info(8, "tournier07".to_string());
        b.set_sh_full_basis(false);
        b.set_sh_legacy(false);
        b.set_sh_data("coefficients", bytes(&all), row.len(), DType::Float32);
    }
    if with_odf {
        let row = fiber_amplitudes(dir);
        let mut all = Vec::with_capacity(NVOX * row.len());
        for _ in 0..NVOX {
            all.extend_from_slice(&row);
        }
        b.set_sphere(
            dsistudio_odf8::full_vertices_ras().to_vec(),
            dsistudio_odf8::faces().to_vec(),
        );
        b.set_sphere_id("dsistudio_odf8");
        b.set_odf_sample_domain("hemisphere");
        b.set_odf_data("amplitudes", bytes(&all), row.len(), DType::Float32);
    }
    b.finalize().unwrap()
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap()
}

#[test]
fn identity_field_is_a_no_op() {
    let odx = make_dataset(unit([0.3, 0.9, 0.2]), true, true);
    let field = uniform_field(
        IDENTITY,
        Header::identity_affine(),
        IdentityPolicy::Included,
    );
    let (out, report) = apply_graddev(&odx, &field, &GradDevOptions::default()).unwrap();

    assert_eq!(report.nb_voxels, NVOX);
    assert_eq!(report.nb_corrected, 0);
    assert_eq!(report.nb_outside_field, 0);
    assert_eq!(report.nb_singular, 0);
    assert_eq!(out.nb_peaks(), odx.nb_peaks());
    assert_eq!(out.directions(), odx.directions());
    assert_eq!(
        out.sh::<f32>("coefficients").unwrap().as_flat_slice(),
        odx.sh::<f32>("coefficients").unwrap().as_flat_slice()
    );
    assert_eq!(
        out.odf::<f32>("amplitudes").unwrap().as_flat_slice(),
        odx.odf::<f32>("amplitudes").unwrap().as_flat_slice()
    );
    assert!(out.header().extra.contains_key(odx_rs::GRADDEV_APPLIED_KEY));
}

#[test]
fn uniform_rotation_maps_fixels_through_t_inverse() {
    let a = rot_z(20.0);
    let t = transpose(a);
    let odx = make_dataset([1.0, 0.0, 0.0], false, false);
    let field = uniform_field(t, Header::identity_affine(), IdentityPolicy::Included);

    let (out, report) = apply_graddev(&odx, &field, &GradDevOptions::default()).unwrap();
    assert_eq!(report.nb_corrected, NVOX);
    assert!((report.median_rotation_deg - 20.0).abs() < 0.01);
    assert!((report.max_rotation_deg - 20.0).abs() < 0.01);
    assert!(report.max_abs_log_det.abs() < 1e-5);

    let expected = apply(&a, [1.0, 0.0, 0.0]);
    for d in out.directions() {
        assert!(
            angle_deg(*d, expected) < 0.01,
            "got {d:?}, want {expected:?}"
        );
    }
    // Amplitudes and cardinality untouched.
    assert_eq!(out.scalar_dpf_f32("amplitude").unwrap(), vec![1.0f32; NVOX]);

    // The transpose escape hatch goes the other way — proving the transpose
    // is load-bearing rather than a symmetric no-op.
    let opts = GradDevOptions {
        transpose: true,
        ..Default::default()
    };
    let (out_t, _) = apply_graddev(&odx, &field, &opts).unwrap();
    let expected_t = apply(&transpose(a), [1.0, 0.0, 0.0]);
    for d in out_t.directions() {
        assert!(angle_deg(*d, expected_t) < 0.01);
    }
    assert!(angle_deg(expected, expected_t) > 39.0);
}

#[test]
fn uniform_rotation_rotates_sh_lobe() {
    let a = rot_z(20.0);
    let t = transpose(a);
    let odx = make_dataset([1.0, 0.0, 0.0], true, false);
    let field = uniform_field(t, Header::identity_affine(), IdentityPolicy::Included);

    let (out, report) = apply_graddev(&odx, &field, &GradDevOptions::default()).unwrap();
    assert_eq!(report.sh_arrays, vec!["coefficients".to_string()]);

    let sh_in = odx.sh::<f32>("coefficients").unwrap();
    let sh_out = out.sh::<f32>("coefficients").unwrap();
    let expected = apply(&a, [1.0, 0.0, 0.0]);

    let amp_in = mrtrix_sh::sample_nonnegative(sh_in.row(0), &[[1.0, 0.0, 0.0]]).unwrap()[0];
    let amp_out =
        mrtrix_sh::sample_nonnegative(sh_out.row(0), &[expected, [1.0, 0.0, 0.0]]).unwrap();
    // The lobe now peaks at A·x̂ with its original height; at x̂ it has
    // dropped to cos(20°)^8 ≈ 0.61 of that.
    assert!(
        (amp_out[0] - amp_in).abs() < 0.02 * amp_in,
        "peak amplitude {} vs {}",
        amp_out[0],
        amp_in
    );
    assert!(amp_out[1] < 0.7 * amp_in);

    let hemi = dsistudio_odf8::hemisphere_vertices_ras();
    let amps = mrtrix_sh::sample_nonnegative(sh_out.row(NVOX - 1), hemi).unwrap();
    let (nearest, _) = nearest_vertex(expected, hemi, true);
    assert_eq!(argmax(&amps), nearest);
}

#[test]
fn uniform_rotation_rotates_dense_odf() {
    let a = rot_z(20.0);
    let t = transpose(a);
    let odx = make_dataset([1.0, 0.0, 0.0], false, true);
    let field = uniform_field(t, Header::identity_affine(), IdentityPolicy::Included);

    let (out, report) = apply_graddev(&odx, &field, &GradDevOptions::default()).unwrap();
    assert_eq!(report.odf_arrays, vec!["amplitudes".to_string()]);
    assert!(report.odf_lmax.unwrap() >= 8);

    let hemi = dsistudio_odf8::hemisphere_vertices_ras();
    let odf = out.odf::<f32>("amplitudes").unwrap();
    let row = odf.row(0);
    // ψ_true(v_k) = ψ_est(normalize(T v_k)); ψ_est = |v·x̂|^8 is bandlimited at
    // lmax 8, so the SH round trip reproduces it essentially exactly.
    for (k, v) in hemi.iter().enumerate() {
        let w = apply(&t, *v);
        let expected = w[0].abs().powi(8);
        assert!(
            (row[k] - expected).abs() < 5e-3,
            "vertex {k}: got {} want {expected}",
            row[k]
        );
    }
    let expected_dir = apply(&a, [1.0, 0.0, 0.0]);
    let (nearest, _) = nearest_vertex(expected_dir, hemi, true);
    assert_eq!(argmax(row), nearest);
    // Every voxel got the same treatment.
    assert_eq!(odf.row(NVOX - 1), row);
}

#[test]
fn field_in_lps_voxel_frame_matches_ras_result() {
    // Physical A_ras = R_x(15°) ⇒ T_ras = R_x(-15°). Expressed in an LPS+
    // voxel frame (P = diag(-1,-1,1)): T_ijk = P⁻¹ T_ras P = R_x(+15°).
    let a_ras = rot_x(15.0);
    let t_ras = transpose(a_ras);
    let p: Mat = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]];
    let t_ijk = mat_mul(mat_mul(p, t_ras), p);
    assert!((t_ijk[1][2] - rot_x(15.0)[1][2]).abs() < 1e-12);

    // LPS+ grid covering the same world points as the identity ODX grid.
    let mut lps = Header::identity_affine();
    lps[0][0] = -1.0;
    lps[1][1] = -1.0;
    lps[0][3] = 2.0;
    lps[1][3] = 2.0;

    let odx = make_dataset([0.0, 1.0, 0.0], true, false);
    let field_lps = uniform_field(t_ijk, lps, IdentityPolicy::Included);
    let field_ras = uniform_field(t_ras, Header::identity_affine(), IdentityPolicy::Included);

    let (out_lps, rep_lps) = apply_graddev(&odx, &field_lps, &GradDevOptions::default()).unwrap();
    let (out_ras, rep_ras) = apply_graddev(&odx, &field_ras, &GradDevOptions::default()).unwrap();
    assert_eq!(rep_lps.nb_outside_field, 0);
    assert_eq!(rep_lps.nb_corrected, rep_ras.nb_corrected);

    let expected = apply(&a_ras, [0.0, 1.0, 0.0]);
    for (d_lps, d_ras) in out_lps.directions().iter().zip(out_ras.directions()) {
        assert!(angle_deg(*d_lps, *d_ras) < 0.02);
        assert!(
            angle_deg(*d_lps, expected) < 0.01,
            "got {d_lps:?}, want {expected:?}"
        );
    }
    let sh_lps = out_lps.sh::<f32>("coefficients").unwrap();
    let sh_ras = out_ras.sh::<f32>("coefficients").unwrap();
    for (x, y) in sh_lps.as_flat_slice().iter().zip(sh_ras.as_flat_slice()) {
        assert!((x - y).abs() < 1e-4, "{x} vs {y}");
    }
}

#[test]
fn nifti_loader_keeps_voxel_frame_and_detects_identity() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("grad_dev.nii");

    // HCP-style (T − I) with an xz term, expressed in an LPS+ voxel frame.
    let mut hcp = [[0.0f64; 3]; 3];
    hcp[0][2] = 0.05;
    let mut lps = Header::identity_affine();
    lps[0][0] = -1.0;
    lps[1][1] = -1.0;
    lps[0][3] = 2.0;
    lps[1][3] = 2.0;
    common::write_uniform_graddev_nifti(&path, [3, 3, 3], lps, hcp);

    let field = GradDevField::load_nifti(&path, IdentityPolicy::Auto).unwrap();
    assert!(field.identity_added());
    assert_eq!(field.dims(), [3, 3, 3]);
    let t = field.matrix_ras_at([1.0, 1.0, 1.0]).unwrap();
    assert!((t[(0, 0)] - 1.0).abs() < 1e-6);
    assert!((t[(1, 1)] - 1.0).abs() < 1e-6);
    // P T Pᵀ with P = diag(-1,-1,1) flips the sign of the xz entry.
    assert!((t[(0, 2)] + 0.05).abs() < 1e-6, "xz = {}", t[(0, 2)]);
    assert!(field.matrix_ras_at([-1.0, 1.0, 1.0]).is_none());

    let forced = GradDevField::load_nifti(&path, IdentityPolicy::Included).unwrap();
    assert!(!forced.identity_added());
    assert!(forced.matrix_ras_at([1.0, 1.0, 1.0]).unwrap()[(0, 0)].abs() < 1e-6);

    // A TORTOISE/qsiprep-style file (identity on the diagonal) is left alone.
    let path2 = tmp.path().join("graddev_c.nii");
    let mut tortoise = IDENTITY;
    tortoise[0][2] = 0.05;
    common::write_uniform_graddev_nifti(&path2, [3, 3, 3], Header::identity_affine(), tortoise);
    let field2 = GradDevField::load_nifti(&path2, IdentityPolicy::Auto).unwrap();
    assert!(!field2.identity_added());
    let t2 = field2.matrix_ras_at([1.0, 1.0, 1.0]).unwrap();
    assert!((t2[(0, 2)] - 0.05).abs() < 1e-6);
}

#[test]
fn voxels_outside_the_field_are_left_untouched() {
    let a = rot_z(20.0);
    let t = transpose(a);
    let odx = make_dataset([1.0, 0.0, 0.0], false, false);
    // Field covering only the k = 0 plane.
    let mut data = Vec::new();
    for _ in 0..9 {
        for row in &t {
            for &v in row {
                data.push(v as f32);
            }
        }
    }
    let field = GradDevField::from_parts(
        [3, 3, 1],
        Header::identity_affine(),
        data,
        IdentityPolicy::Included,
    )
    .unwrap();

    let (out, report) = apply_graddev(&odx, &field, &GradDevOptions::default()).unwrap();
    assert_eq!(report.nb_outside_field, 18);
    assert_eq!(report.nb_corrected, 9);

    let expected = apply(&a, [1.0, 0.0, 0.0]);
    // Compact order is i-slowest, k-fastest.
    for (v, d) in out.directions().iter().enumerate() {
        if v % 3 == 0 {
            assert!(angle_deg(*d, expected) < 0.01);
        } else {
            assert_eq!(*d, [1.0, 0.0, 0.0]);
        }
    }
}

#[test]
fn pam_peak_index_dpf_is_dropped() {
    let mut b = OdxBuilder::new(Header::identity_affine(), DIMS, vec![1u8; NVOX]);
    for _ in 0..NVOX {
        b.push_voxel_peaks(&[[1.0, 0.0, 0.0]]);
    }
    b.set_dpf_data("amplitude", bytes(&[1.0f32; NVOX]), 1, DType::Float32);
    b.set_dpf_data("pam_peak_index", bytes(&[7i32; NVOX]), 1, DType::Int32);
    b.set_dpf_data("qa", bytes(&[0.5f32; NVOX]), 1, DType::Float32);
    let odx = b.finalize().unwrap();

    let field = uniform_field(
        transpose(rot_z(5.0)),
        Header::identity_affine(),
        IdentityPolicy::Included,
    );
    let (out, report) = apply_graddev(&odx, &field, &GradDevOptions::default()).unwrap();
    assert_eq!(report.dropped_dpf, vec!["pam_peak_index".to_string()]);
    assert!(out.get_dpf("pam_peak_index").is_none());
    assert_eq!(out.scalar_dpf_f32("qa").unwrap(), vec![0.5f32; NVOX]);
    assert_eq!(out.scalar_dpf_f32("amplitude").unwrap(), vec![1.0f32; NVOX]);
}

fn dsistudio_options(format: DsistudioFormat, float_dirs: bool) -> MrtrixToDsistudioOptions {
    MrtrixToDsistudioOptions {
        output_format: format,
        dense_odf_mode: DenseOdfMode::Off,
        peak_source: PeakSource::Fixels,
        amplitude_key: None,
        write_z0: Z0Policy::Never,
        write_float_directions: float_dirs,
    }
}

/// A direction halfway between two adjacent odf8 vertices, i.e. as far from
/// the sphere as a peak can be (~4°).
fn off_vertex_direction() -> [f32; 3] {
    let verts = dsistudio_odf8::full_vertices_ras();
    let face = dsistudio_odf8::faces()[0];
    let (a, b) = (verts[face[0] as usize], verts[face[1] as usize]);
    unit([a[0] + b[0], a[1] + b[1], a[2] + b[2]])
}

#[test]
fn dsistudio_float_dir_records_round_trip_exact_directions() {
    let dir = off_vertex_direction();
    let odx = make_dataset(dir, false, false);
    let tmp = tempfile::tempdir().unwrap();

    for (format, name) in [
        (DsistudioFormat::FibGz, "out.fib.gz"),
        (DsistudioFormat::Fz, "out.fz"),
    ] {
        let path = tmp.path().join(name);
        save_dsistudio_from_odx(&odx, &path, &dsistudio_options(format, true)).unwrap();

        let mat = mat4::read_mat4_gz(&path).unwrap();
        let record = mat
            .get("dir0")
            .unwrap_or_else(|| panic!("{name}: no dir0 record"));
        assert_eq!(record.mrows(), 3, "{name}");
        let values = record.as_f32_vec();
        assert_eq!(values.len(), NVOX * 3, "{name}");
        // Stored in DSI Studio's LPS sphere convention.
        assert!((values[0] + dir[0]).abs() < 1e-6, "{name}: x");
        assert!((values[1] + dir[1]).abs() < 1e-6, "{name}: y");
        assert!((values[2] - dir[2]).abs() < 1e-6, "{name}: z");
        assert!(mat.get("index0").is_some(), "{name}: index0 still written");

        let loaded = match format {
            DsistudioFormat::FibGz => dsistudio::load_fibgz(&path, None).unwrap(),
            DsistudioFormat::Fz => dsistudio::load_fz(&path, None).unwrap(),
        };
        assert_eq!(loaded.nb_peaks(), NVOX, "{name}");
        for d in loaded.directions() {
            assert!(angle_deg(*d, dir) < 0.02, "{name}: {d:?} vs {dir:?}");
        }
        assert!(
            loaded.get_dpf("dir").is_none(),
            "{name}: dir must not become a DPF"
        );
    }

    // Without float directions the round trip snaps to the sphere.
    let path = tmp.path().join("quantized.fib.gz");
    save_dsistudio_from_odx(
        &odx,
        &path,
        &dsistudio_options(DsistudioFormat::FibGz, false),
    )
    .unwrap();
    let mat = mat4::read_mat4_gz(&path).unwrap();
    assert!(mat.get("dir0").is_none());
    let loaded = dsistudio::load_fibgz(&path, None).unwrap();
    let err = angle_deg(loaded.directions()[0], dir);
    assert!(err > 1.0, "expected visible quantization, got {err} deg");
}

#[test]
fn graddev_then_dsistudio_export_keeps_rotated_directions() {
    let a = rot_z(3.0); // a realistic gradient-nonlinearity rotation
    let odx = make_dataset([1.0, 0.0, 0.0], false, true);
    let field = uniform_field(
        transpose(a),
        Header::identity_affine(),
        IdentityPolicy::Included,
    );
    let (corrected, _) = apply_graddev(&odx, &field, &GradDevOptions::default()).unwrap();

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("corrected.fib.gz");
    save_dsistudio_from_odx(
        &corrected,
        &path,
        &dsistudio_options(DsistudioFormat::FibGz, true),
    )
    .unwrap();
    let loaded = dsistudio::load_fibgz(&path, None).unwrap();

    let expected = apply(&a, [1.0, 0.0, 0.0]);
    for d in loaded.directions() {
        assert!(angle_deg(*d, expected) < 0.02, "{d:?} vs {expected:?}");
    }
    // The dense ODF rode along too.
    let odf = loaded.odf::<f32>("amplitudes").unwrap();
    assert_eq!(odf.ncols(), 321);
    let hemi = dsistudio_odf8::hemisphere_vertices_ras();
    let (nearest, _) = nearest_vertex(expected, hemi, true);
    assert_eq!(argmax(odf.row(0)), nearest);
}
