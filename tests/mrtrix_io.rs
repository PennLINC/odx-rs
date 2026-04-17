use std::path::{Path, PathBuf};

use tempfile::tempdir;

use odx_rs::mrtrix::{
    self, MrtrixFixelContainer, MrtrixFixelWriteOptions, MrtrixShContainer, MrtrixShWriteOptions,
};

const SH_MIF: &str =
    "../test_data/sub-NDARAE199TDD_ses-1_acq-64dirVARIANTVar1e_space-ACPC_model-ss3t_param-fod_label-WM_dwimap.mif.gz";
const FIXELS_NII: &str = "../test_data/fixels_nii";
const FIXELS_MIF: &str = "../test_data/fixels_mif";

fn fixture_path(rel: &str) -> PathBuf {
    Path::new(rel).to_path_buf()
}

fn assert_close_slice(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len());
    let mut max_diff = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        max_diff = max_diff.max((x - y).abs());
    }
    assert!(
        max_diff <= tol,
        "max absolute difference {max_diff} exceeds tolerance {tol}"
    );
}

#[test]
fn load_mrtrix_sh_fixture() {
    let path = fixture_path(SH_MIF);
    if !path.exists() {
        eprintln!("skipping missing fixture {}", path.display());
        return;
    }

    let odx = mrtrix::load_mrtrix_sh(&path).unwrap();
    assert_eq!(odx.header().dimensions, [80, 98, 85]);
    assert_eq!(odx.header().sh_order, Some(8));
    assert_eq!(odx.header().sh_basis.as_deref(), Some("tournier07"));
    let sh = odx.sh::<f32>("coefficients").unwrap();
    assert_eq!(sh.ncols(), 45);
    assert_eq!(sh.nrows(), odx.nb_voxels());
}

#[test]
fn mif_layout_preserves_negative_zero() {
    let path = fixture_path("../test_data/fixels_mif/directions.mif");
    if !path.exists() {
        eprintln!("skipping missing fixture {}", path.display());
        return;
    }

    let img = odx_rs::mif::read_mif(&path).unwrap();
    assert!(img.header.layout[0].negative);
    assert_eq!(img.header.layout[0].rank, 0);
}

#[test]
fn load_mrtrix_fixels_mif_and_nii_match() {
    let nii = fixture_path(FIXELS_NII);
    let mif = fixture_path(FIXELS_MIF);
    if !nii.exists() || !mif.exists() {
        eprintln!("skipping missing fixel fixtures");
        return;
    }

    let odx_nii = mrtrix::load_mrtrix_fixels(&nii).unwrap();
    let odx_mif = mrtrix::load_mrtrix_fixels(&mif).unwrap();

    assert_eq!(odx_nii.header().dimensions, odx_mif.header().dimensions);
    assert_eq!(odx_nii.nb_peaks(), 261_492);
    assert_eq!(
        odx_nii.header().voxel_to_rasmm,
        odx_mif.header().voxel_to_rasmm
    );
    assert_eq!(odx_nii.mask(), odx_mif.mask());
    assert_eq!(odx_nii.offsets(), odx_mif.offsets());
    assert_eq!(odx_nii.directions(), odx_mif.directions());

    let afd_nii = odx_nii.scalar_dpf_f32("afd").unwrap();
    let afd_mif = odx_mif.scalar_dpf_f32("afd").unwrap();
    let disp_nii = odx_nii.scalar_dpf_f32("disp").unwrap();
    let disp_mif = odx_mif.scalar_dpf_f32("disp").unwrap();
    assert_close_slice(&afd_nii, &afd_mif, 1e-6);
    assert_close_slice(&disp_nii, &disp_mif, 1e-6);
}

#[test]
fn mrtrix_fixel_names_are_literal_stems() {
    let path = fixture_path(FIXELS_MIF);
    if !path.exists() {
        eprintln!("skipping missing fixture {}", path.display());
        return;
    }

    let odx = mrtrix::load_mrtrix_fixels(&path).unwrap();
    let mut names = odx.dpf_names();
    names.sort_unstable();
    assert_eq!(names, vec!["afd", "disp"]);
}

#[test]
fn load_combined_mrtrix_dataset() {
    let sh = fixture_path(SH_MIF);
    let fixels = fixture_path(FIXELS_MIF);
    if !sh.exists() || !fixels.exists() {
        eprintln!("skipping missing MRtrix fixtures");
        return;
    }

    let odx = mrtrix::load_mrtrix_dataset(Some(&sh), Some(&fixels)).unwrap();
    assert_eq!(odx.header().dimensions, [80, 98, 85]);
    assert_eq!(odx.header().sh_basis.as_deref(), Some("tournier07"));
    assert_eq!(odx.nb_peaks(), 261_492);
    assert!(odx.dpf_names().contains(&"afd"));
    assert!(odx.dpf_names().contains(&"disp"));
    assert_eq!(
        odx.sh::<f32>("coefficients").unwrap().nrows(),
        odx.nb_voxels()
    );
}

#[test]
fn round_trip_mrtrix_sh_mif_and_nifti1() {
    let sh = fixture_path(SH_MIF);
    if !sh.exists() {
        eprintln!("skipping missing fixture {}", sh.display());
        return;
    }

    let odx = mrtrix::load_mrtrix_sh(&sh).unwrap();
    let tmp = tempdir().unwrap();

    let out_mif = tmp.path().join("roundtrip_sh.mif.gz");
    mrtrix::save_mrtrix_sh(
        &odx,
        &out_mif,
        &MrtrixShWriteOptions {
            container: MrtrixShContainer::Mif,
            ..Default::default()
        },
    )
    .unwrap();
    let reloaded_mif = mrtrix::load_mrtrix_sh(&out_mif).unwrap();
    assert_eq!(reloaded_mif.header().dimensions, odx.header().dimensions);
    assert_eq!(reloaded_mif.header().sh_order, odx.header().sh_order);

    let out_nii = tmp.path().join("roundtrip_sh.nii.gz");
    mrtrix::save_mrtrix_sh(
        &odx,
        &out_nii,
        &MrtrixShWriteOptions {
            container: MrtrixShContainer::Nifti1,
            gzip: true,
            ..Default::default()
        },
    )
    .unwrap();
    let reloaded_nii = mrtrix::load_mrtrix_sh(&out_nii).unwrap();
    assert_eq!(reloaded_nii.header().dimensions, odx.header().dimensions);
    assert_eq!(reloaded_nii.header().sh_order, odx.header().sh_order);
}

#[test]
fn round_trip_mrtrix_fixels_mif_and_nifti() {
    let fixels = fixture_path(FIXELS_MIF);
    if !fixels.exists() {
        eprintln!("skipping missing fixture {}", fixels.display());
        return;
    }

    let odx = mrtrix::load_mrtrix_fixels(&fixels).unwrap();
    let tmp = tempdir().unwrap();

    let out_mif = tmp.path().join("fixels_mif");
    mrtrix::save_mrtrix_fixels(
        &odx,
        &out_mif,
        &MrtrixFixelWriteOptions {
            container: MrtrixFixelContainer::Mif,
            include_dpf: true,
            include_dpv: false,
        },
    )
    .unwrap();
    let reloaded_mif = mrtrix::load_mrtrix_fixels(&out_mif).unwrap();
    assert_eq!(reloaded_mif.offsets(), odx.offsets());
    assert_eq!(reloaded_mif.directions(), odx.directions());
    assert_close_slice(
        &reloaded_mif.scalar_dpf_f32("afd").unwrap(),
        &odx.scalar_dpf_f32("afd").unwrap(),
        1e-6,
    );

    let out_nii = tmp.path().join("fixels_nii");
    mrtrix::save_mrtrix_fixels(
        &odx,
        &out_nii,
        &MrtrixFixelWriteOptions {
            container: MrtrixFixelContainer::Nifti,
            include_dpf: true,
            include_dpv: false,
        },
    )
    .unwrap();
    let reloaded_nii = mrtrix::load_mrtrix_fixels(&out_nii).unwrap();
    assert_eq!(reloaded_nii.offsets(), odx.offsets());
    assert_eq!(reloaded_nii.directions(), odx.directions());
    assert_close_slice(
        &reloaded_nii.scalar_dpf_f32("disp").unwrap(),
        &odx.scalar_dpf_f32("disp").unwrap(),
        1e-6,
    );
}
