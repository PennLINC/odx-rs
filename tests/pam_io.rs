use std::path::{Path, PathBuf};

use hdf5_metno::types::VarLenUnicode;
use hdf5_metno::File;
use odx_rs::{pam, sh_basis_evaluator, DType, Header, OdxBuilder};

const PAM_FIXTURE: &str = "../test_data/pam_fixture.pam5";

fn fixture_path(rel: &str) -> PathBuf {
    Path::new(rel).to_path_buf()
}

#[test]
fn load_pam_fixture_imports_sparse_peaks_and_metrics() {
    let fixture = fixture_path(PAM_FIXTURE);
    if !fixture.exists() {
        eprintln!("skipping missing fixture {}", fixture.display());
        return;
    }
    let odx = pam::load_pam5(&fixture).unwrap();

    assert_eq!(odx.header().dimensions, [2, 2, 1]);
    assert_eq!(odx.nb_voxels(), 3);
    assert_eq!(odx.nb_peaks(), 6);
    assert_eq!(odx.mask(), &[1, 1, 0, 1]);
    assert_eq!(odx.offsets(), &[0, 2, 3, 6]);

    assert_eq!(odx.directions()[0], [-1.0, 0.0, 0.0]);
    assert_eq!(odx.directions()[1], [0.0, 1.0, 0.0]);
    assert_eq!(odx.directions()[2], [0.0, 0.0, 1.0]);

    assert_eq!(
        odx.scalar_dpf_f32("amplitude").unwrap(),
        vec![0.9, 0.4, 0.8, 0.7, 0.6, 0.5]
    );
    assert_eq!(
        odx.dpf::<i32>("pam_peak_index").unwrap().as_flat_slice(),
        &[0, 1, 2, 3, 1, 2]
    );
    assert_eq!(
        odx.scalar_dpf_f32("qa").unwrap(),
        vec![0.6, 0.2, 0.5, 0.4, 0.3, 0.2]
    );
    assert_eq!(
        odx.scalar_dpf_f32("custom_metric").unwrap(),
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    );
    assert_eq!(odx.scalar_dpv_f32("gfa").unwrap(), vec![0.7, 0.8, 0.9]);
    assert_eq!(
        odx.scalar_dpv_f32("extra_scalar").unwrap(),
        vec![1.5, 2.5, 3.5]
    );

    let sh = odx.sh::<f32>("coefficients").unwrap();
    assert_eq!(sh.shape(), (3, 6));
    assert_eq!(sh.row(0), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(sh.row(1), &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    assert_eq!(sh.row(2), &[13.0, 14.0, 15.0, 16.0, 17.0, 18.0]);

    assert_eq!(odx.header().sh_order, Some(2));
    assert_eq!(odx.header().sh_basis.as_deref(), Some("descoteaux07"));
    assert_eq!(
        odx.header().extra.get("_ODX_PAM_SH_BASIS_ASSUMED").unwrap(),
        "descoteaux07"
    );
    assert_eq!(odx.header().extra.get("_ODX_PAM_VERSION").unwrap(), "0.0.1");
    assert_eq!(
        odx.header()
            .extra
            .get("_ODX_PAM_TOTAL_WEIGHT")
            .and_then(|value| value.as_f64()),
        Some(0.5)
    );
    assert_eq!(
        odx.header()
            .extra
            .get("_ODX_PAM_ANG_THR")
            .and_then(|value| value.as_f64()),
        Some(60.0)
    );
}

#[test]
fn round_trip_pam_preserves_standard_and_generic_metrics() {
    let fixture = fixture_path(PAM_FIXTURE);
    if !fixture.exists() {
        eprintln!("skipping missing fixture {}", fixture.display());
        return;
    }
    let odx = pam::load_pam5(&fixture).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("roundtrip.pam5");

    pam::save_pam5(&odx, &out, &pam::PamWriteOptions::default()).unwrap();

    let file = File::open(&out).unwrap();
    let version: VarLenUnicode = file.attr("version").unwrap().read_scalar().unwrap();
    assert_eq!(version.as_str(), "0.0.1");

    let group = file.group("pam").unwrap();
    assert!(group.link_exists("peak_dirs"));
    assert!(group.link_exists("peak_values"));
    assert!(group.link_exists("peak_indices"));
    assert!(group.link_exists("qa"));
    assert!(group.link_exists("gfa"));
    assert!(group.link_exists("custom_metric"));
    assert!(group.link_exists("extra_scalar"));
    assert!(group.link_exists("shm_coeff"));

    let peak_values = group
        .dataset("peak_values")
        .unwrap()
        .read_raw::<f32>()
        .unwrap();
    assert_eq!(peak_values.len(), 12);
    assert_eq!(peak_values[0..3], [0.9, 0.4, 0.0]);
    assert_eq!(peak_values[3..6], [0.8, 0.0, 0.0]);
    assert_eq!(peak_values[9..12], [0.7, 0.6, 0.5]);

    let custom_metric = group
        .dataset("custom_metric")
        .unwrap()
        .read_raw::<f32>()
        .unwrap();
    assert_eq!(custom_metric[0..3], [10.0, 20.0, 0.0]);
    assert_eq!(custom_metric[3..6], [30.0, 0.0, 0.0]);
    assert_eq!(custom_metric[9..12], [40.0, 50.0, 60.0]);

    let peak_dirs = group
        .dataset("peak_dirs")
        .unwrap()
        .read_raw::<f32>()
        .unwrap();
    let first = &peak_dirs[0..3];
    assert!((first[0] - 1.0).abs() < 1e-5);
    assert!(first[1].abs() < 1e-5);
    assert!(first[2].abs() < 1e-5);

    let peak_indices = group
        .dataset("peak_indices")
        .unwrap()
        .read_raw::<i32>()
        .unwrap();
    assert_eq!(peak_indices[0..3], [0, 1, -1]);
    assert_eq!(peak_indices[3..6], [2, -1, -1]);
    assert_eq!(peak_indices[9..12], [3, 1, 2]);
}

/// Tournier lmax-4 SH on one voxel of a grid rotated 30° about z with a
/// flipped x axis, so both the basis change and the frame change matter.
fn make_oblique_tournier_odx() -> (odx_rs::OdxDataset, Vec<f32>, [[f64; 3]; 3]) {
    let (c, s) = (30f64.to_radians().cos(), 30f64.to_radians().sin());
    let rot = [[-c, -s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]];
    let mut affine = Header::identity_affine();
    for r in 0..3 {
        for k in 0..3 {
            affine[r][k] = 2.0 * rot[r][k];
        }
    }
    let coeffs: Vec<f32> = (0..15)
        .map(|i| ((i * 7 % 11) as f32 - 5.0) / 10.0)
        .collect();
    let mut builder = OdxBuilder::new(affine, [1, 1, 1], vec![1u8]);
    builder.push_voxel_peaks(&[[1.0, 0.0, 0.0]]);
    builder.set_sh_info(4, "tournier07".into());
    builder.set_sh_data(
        "coefficients",
        bytemuck::cast_slice(&coeffs).to_vec(),
        15,
        DType::Float32,
    );
    builder.set_dpf_data(
        "amplitude",
        bytemuck::cast_slice(&[0.8f32]).to_vec(),
        1,
        DType::Float32,
    );
    (builder.finalize().unwrap(), coeffs, rot)
}

fn eval_sh(coeffs: &[f32], dirs: &[[f32; 3]], basis: &str) -> Vec<f32> {
    let n = coeffs.len();
    let b = sh_basis_evaluator::compute_b_matrix(dirs, 4, basis, false).unwrap();
    b.chunks_exact(n)
        .map(|row| row.iter().zip(coeffs).map(|(a, c)| a * c).sum())
        .collect()
}

fn test_dirs() -> Vec<[f32; 3]> {
    let mut dirs = Vec::new();
    for i in 0..40 {
        let z = 1.0 - 2.0 * (i as f32 + 0.5) / 40.0;
        let r = (1.0 - z * z).sqrt();
        let phi = 2.399_963 * i as f32;
        dirs.push([r * phi.cos(), r * phi.sin(), z]);
    }
    dirs
}

#[test]
fn tournier_sh_is_converted_and_reoriented_for_pam() {
    let (odx, coeffs, rot) = make_oblique_tournier_odx();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("converted.pam5");
    pam::save_pam5(&odx, &out, &pam::PamWriteOptions::default()).unwrap();

    let file = File::open(&out).unwrap();
    let group = file.group("pam").unwrap();
    let shm = group
        .dataset("shm_coeff")
        .unwrap()
        .read_raw::<f32>()
        .unwrap();
    assert_eq!(shm.len(), 15);
    // PAM is in the voxel frame: f_pam(v) = f_ras(R v).
    let v = test_dirs();
    let u: Vec<[f32; 3]> = v
        .iter()
        .map(|d| {
            std::array::from_fn(|r| {
                (rot[r][0] * d[0] as f64 + rot[r][1] * d[1] as f64 + rot[r][2] * d[2] as f64) as f32
            })
        })
        .collect();
    let expected = eval_sh(&coeffs, &u, "tournier07");
    let got = eval_sh(&shm, &v, "descoteaux07_legacy");
    for (a, b) in expected.iter().zip(&got) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }
    // dipy's load_pam requires these.
    let tw = group
        .dataset("total_weight")
        .unwrap()
        .read_raw::<f64>()
        .unwrap();
    let ang = group.dataset("ang_thr").unwrap().read_raw::<f64>().unwrap();
    assert_eq!((tw[0], ang[0]), (0.5, 60.0));

    // Loading returns RAS-frame descoteaux SH describing the same function.
    let back = pam::load_pam5(&out).unwrap();
    assert_eq!(back.header().dipy_basis_name(), Some("descoteaux07_legacy"));
    let sh = back.sh::<f32>("coefficients").unwrap();
    let got = eval_sh(sh.row(0), &u, "descoteaux07_legacy");
    let expected = eval_sh(&coeffs, &u, "tournier07");
    for (a, b) in expected.iter().zip(&got) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }
}

#[test]
fn non_legacy_pam_basis_is_honored() {
    let (odx, coeffs, _) = make_oblique_tournier_odx();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("modern.pam5");
    let basis = pam::PamShBasis::Descoteaux07;
    pam::save_pam5(&odx, &out, &pam::PamWriteOptions { sh_basis: basis }).unwrap();
    let back = pam::load_pam5_with_options(&out, &pam::PamReadOptions { sh_basis: basis }).unwrap();
    assert_eq!(back.header().dipy_basis_name(), Some("descoteaux07"));
    let u = test_dirs();
    let got = eval_sh(
        back.sh::<f32>("coefficients").unwrap().row(0),
        &u,
        "descoteaux07",
    );
    let expected = eval_sh(&coeffs, &u, "tournier07");
    for (a, b) in expected.iter().zip(&got) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }
}
