use std::path::{Path, PathBuf};
use std::process::Command;

use odx_rs::formats::dsistudio_odf8;
use odx_rs::interop::{
    dsistudio_to_mrtrix, mrtrix_to_dsistudio, save_dsistudio_from_odx, DenseOdfMode,
    DsistudioFormat, DsistudioToMrtrixOptions, MrtrixToDsistudioOptions, PeakSource,
};
use odx_rs::{dsistudio, mif, mrtrix, mrtrix_sh, read_reference_affine};

const SH_MIF: &str =
    "../test_data/sub-NDARAE199TDD_ses-1_acq-64dirVARIANTVar1e_space-ACPC_model-ss3t_param-fod_label-WM_dwimap.mif.gz";
const FIXELS_NII: &str = "../test_data/fixels_nii";
const FIXELS_MIF: &str = "../test_data/fixels_mif";
const FIB_PATH: &str =
    "../test_data/sub-NDARAE199TDD_ses-1_acq-64dirVARIANTVar1e_space-ACPC_model-ss3t_dwimap.fib.gz";
const REF_AFFINE_PATH: &str =
    "../test_data/sub-NDARAE199TDD_ses-1_acq-64dirVARIANTVar1e_space-ACPC_model-tensor_param-fa_dwimap.nii.gz";

fn fixture_path(rel: &str) -> PathBuf {
    Path::new(rel).to_path_buf()
}

fn fixture_reference_affine() -> [[f64; 4]; 4] {
    read_reference_affine(Path::new(REF_AFFINE_PATH)).unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn mrtrix_to_dsistudio_fixels_only_writes_peaks_only_fz() {
    let fixels = fixture_path(FIXELS_MIF);
    if !fixels.exists() {
        eprintln!("skipping missing fixture {}", fixels.display());
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("peaks_only.fz");
    mrtrix_to_dsistudio(
        &fixels,
        None,
        &out,
        &MrtrixToDsistudioOptions {
            output_format: DsistudioFormat::Fz,
            dense_odf_mode: DenseOdfMode::Off,
            peak_source: PeakSource::Fixels,
            ..Default::default()
        },
    )
    .unwrap();

    let odx = dsistudio::load_fz(&out, None).unwrap();
    assert_eq!(odx.header().dimensions, [80, 98, 85]);
    assert!(odx.nb_peaks() > 0);
    assert!(odx.odf::<f32>("amplitudes").is_err());
    assert_eq!(odx.sphere_vertices().unwrap().len(), 642);
}

#[test]
fn mrtrix_to_dsistudio_sh_and_fixels_writes_dense_odf_fz() {
    let sh = fixture_path(SH_MIF);
    let fixels = fixture_path(FIXELS_MIF);
    if !sh.exists() || !fixels.exists() {
        eprintln!("skipping missing fixtures");
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("dense_with_peaks.fz");
    mrtrix_to_dsistudio(
        &fixels,
        Some(&sh),
        &out,
        &MrtrixToDsistudioOptions::default(),
    )
    .unwrap();

    let odx = dsistudio::load_fz(&out, None).unwrap();
    let odf = odx.odf::<f32>("amplitudes").unwrap();
    assert_eq!(odf.ncols(), dsistudio_odf8::hemisphere_vertices_ras().len());
    assert_eq!(
        odx.header().odf_sample_domain.as_deref(),
        Some("hemisphere")
    );
    assert_eq!(odx.header().sphere_id.as_deref(), Some("dsistudio_odf8"));
}

#[test]
fn dsistudio_to_mrtrix_exports_fixels_and_sh_when_dense_odf_exists() {
    let fib = fixture_path(FIB_PATH);
    let reference = fixture_path(REF_AFFINE_PATH);
    if !fib.exists() || !reference.exists() {
        eprintln!("skipping missing fixture {}", fib.display());
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let out_fixels = tmp.path().join("fixels_out");
    let out_sh = tmp.path().join("fod.mif.gz");
    dsistudio_to_mrtrix(
        &fib,
        &out_fixels,
        Some(&out_sh),
        &DsistudioToMrtrixOptions {
            reference_affine: Some(fixture_reference_affine()),
            ..Default::default()
        },
    )
    .unwrap();

    let reloaded_fixels = mrtrix::load_mrtrix_fixels(&out_fixels).unwrap();
    let reloaded_sh = mrtrix::load_mrtrix_sh(&out_sh).unwrap();
    assert!(reloaded_fixels.nb_peaks() > 0);
    assert_eq!(reloaded_sh.header().sh_basis.as_deref(), Some("tournier07"));
}

#[test]
fn dsistudio_fib_to_fz_preserves_existing_dense_odf() {
    let fib = fixture_path(FIB_PATH);
    let reference = fixture_path(REF_AFFINE_PATH);
    if !fib.exists() || !reference.exists() {
        eprintln!("skipping missing fixture {}", fib.display());
        return;
    }

    let odx = dsistudio::load_fibgz(&fib, Some(fixture_reference_affine())).unwrap();
    let orig_odf = odx.odf::<f32>("amplitudes").unwrap();

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("preserved_dense_odf.fz");
    save_dsistudio_from_odx(
        &odx,
        &out,
        &MrtrixToDsistudioOptions {
            output_format: DsistudioFormat::Fz,
            dense_odf_mode: DenseOdfMode::Off,
            peak_source: PeakSource::Fixels,
            ..Default::default()
        },
    )
    .unwrap();

    let reloaded = dsistudio::load_fz(&out, None).unwrap();
    let rt_odf = reloaded.odf::<f32>("amplitudes").unwrap();
    assert_eq!(orig_odf.shape(), rt_odf.shape());

    let max_diff = max_abs_diff(orig_odf.as_flat_slice(), rt_odf.as_flat_slice());
    assert!(max_diff < 0.02, "preserved fz ODF max diff {max_diff}");
}

#[test]
fn fixel_nii_and_mif_produce_matching_dense_dsi_exports() {
    let sh = fixture_path(SH_MIF);
    let fixels_nii = fixture_path(FIXELS_NII);
    let fixels_mif = fixture_path(FIXELS_MIF);
    if !sh.exists() || !fixels_nii.exists() || !fixels_mif.exists() {
        eprintln!("skipping missing fixtures");
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let out_nii = tmp.path().join("from_nii.fz");
    let out_mif = tmp.path().join("from_mif.fz");

    mrtrix_to_dsistudio(
        &fixels_nii,
        Some(&sh),
        &out_nii,
        &MrtrixToDsistudioOptions::default(),
    )
    .unwrap();
    mrtrix_to_dsistudio(
        &fixels_mif,
        Some(&sh),
        &out_mif,
        &MrtrixToDsistudioOptions::default(),
    )
    .unwrap();

    let nii = dsistudio::load_fz(&out_nii, None).unwrap();
    let mif = dsistudio::load_fz(&out_mif, None).unwrap();
    assert_eq!(nii.offsets(), mif.offsets());
    assert_eq!(nii.directions(), mif.directions());
    assert_eq!(
        nii.odf::<f32>("amplitudes").unwrap().shape(),
        mif.odf::<f32>("amplitudes").unwrap().shape()
    );
}

#[test]
fn rust_sh_sampling_matches_external_sh2amp_when_available() {
    if Command::new("sh2amp").arg("-version").output().is_err() {
        eprintln!("skipping: sh2amp not available");
        return;
    }

    let sh = fixture_path(SH_MIF);
    if !sh.exists() {
        eprintln!("skipping missing fixture {}", sh.display());
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let dirs_txt = tmp.path().join("dirs.txt");
    let out_mif = tmp.path().join("amp.mif");
    let dir_text = dsistudio_odf8::hemisphere_vertices_ras()
        .iter()
        .map(|dir| format!("{} {} {}", dir[0], dir[1], dir[2]))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&dirs_txt, dir_text).unwrap();

    let status = Command::new("sh2amp")
        .arg("-quiet")
        .arg("-nonnegative")
        .arg(&sh)
        .arg(&dirs_txt)
        .arg(&out_mif)
        .status()
        .unwrap();
    assert!(status.success(), "external sh2amp failed");

    let odx = mrtrix::load_mrtrix_sh(&sh).unwrap();
    let sh_view = odx.sh::<f32>("coefficients").unwrap();
    let sampled = mrtrix_sh::sample_rows_nonnegative(
        sh_view.as_flat_slice(),
        sh_view.nrows(),
        dsistudio_odf8::hemisphere_vertices_ras(),
        sh_view.ncols(),
    )
    .unwrap();

    let external = mif::read_mif(&out_mif).unwrap().logical_f32_vec().unwrap();
    let diff = max_abs_diff(&sampled, &external);
    assert!(diff <= 1e-5, "sh2amp oracle max diff {diff}");
}
