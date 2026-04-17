use std::path::{Path, PathBuf};

use hdf5_metno::types::VarLenUnicode;
use hdf5_metno::File;
use odx_rs::{pam, CanonicalDenseRepresentation, DType, Header, OdxBuilder};

const PAM_FIXTURE: &str = "../test_data/pam_fixture.pam5";

fn fixture_path(rel: &str) -> PathBuf {
    Path::new(rel).to_path_buf()
}

fn make_incompatible_sh_odx() -> odx_rs::OdxDataset {
    let dims = [1u64, 1, 1];
    let mask = vec![1u8];
    let mut builder = OdxBuilder::new(Header::identity_affine(), dims, mask);
    builder.push_voxel_peaks(&[[1.0, 0.0, 0.0]]);
    builder.set_sphere(vec![[1.0, 0.0, 0.0]], vec![]);
    builder.set_sh_info(2, "tournier07".into());
    builder.set_canonical_dense_representation(CanonicalDenseRepresentation::Sh);
    builder.set_sh_data(
        "coefficients",
        bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).to_vec(),
        6,
        DType::Float32,
    );
    builder.set_dpf_data(
        "amplitude",
        bytemuck::cast_slice(&[0.8f32]).to_vec(),
        1,
        DType::Float32,
    );
    builder.finalize().unwrap()
}

#[test]
fn load_pam_fixture_imports_sparse_peaks_and_metrics() {
    let fixture = fixture_path(PAM_FIXTURE);
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
    let odx = pam::load_pam5(&fixture).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("roundtrip.pam5");

    pam::save_pam5(&odx, &out, &pam::PamWriteOptions).unwrap();

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

#[test]
fn incompatible_sh_basis_is_not_written_to_pam() {
    let odx = make_incompatible_sh_odx();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("no_sh.pam5");

    pam::save_pam5(&odx, &out, &pam::PamWriteOptions).unwrap();

    let file = File::open(&out).unwrap();
    let group = file.group("pam").unwrap();
    assert!(!group.link_exists("shm_coeff"));
}
