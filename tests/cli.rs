use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const SH_MIF: &str =
    "../test_data/sub-NDARAE199TDD_ses-1_acq-64dirVARIANTVar1e_space-ACPC_model-ss3t_param-fod_label-WM_dwimap.mif.gz";
const FIXELS_MIF: &str = "../test_data/fixels_mif";
const FIB_PATH: &str =
    "../test_data/sub-NDARAE199TDD_ses-1_acq-64dirVARIANTVar1e_space-ACPC_model-ss3t_dwimap.fib.gz";
const PAM_PATH: &str = "../test_data/pam_fixture.pam5";

fn fixture_path(rel: &str) -> PathBuf {
    Path::new(rel).to_path_buf()
}

fn create_invalid_odx_dir(path: &Path) {
    fs::create_dir_all(path).unwrap();
    let header = serde_json::json!({
        "VOXEL_TO_RASMM": [[1.0, 0.0, 0.0, 0.0],[0.0, 1.0, 0.0, 0.0],[0.0, 0.0, 1.0, 0.0],[0.0, 0.0, 0.0, 1.0]],
        "DIMENSIONS": [2, 2, 1],
        "NB_VOXELS": 3,
        "NB_PEAKS": 2
    });
    fs::write(
        path.join("header.json"),
        serde_json::to_vec_pretty(&header).unwrap(),
    )
    .unwrap();
    fs::write(path.join("mask.uint8"), [1u8, 0, 1, 0]).unwrap();
    fs::write(
        path.join("offsets.uint32"),
        bytemuck::cast_slice(&[0u32, 1u32, 2u32, 3u32]),
    )
    .unwrap();
    fs::write(
        path.join("directions.3.float32"),
        bytemuck::cast_slice(&[[1.0f32, 0.0, 0.0]]),
    )
    .unwrap();
}

#[test]
fn cli_help_prints_top_level_usage() {
    Command::cargo_bin("odx")
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "ODX conversion, inspection, and validation tools",
        ));
}

#[test]
fn info_help_mentions_json_and_verbose() {
    Command::cargo_bin("odx")
        .unwrap()
        .args(["info", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--json"))
        .stdout(predicate::str::contains("--verbose"));
}

#[test]
fn convert_help_mentions_out_sh_and_dsi_options() {
    Command::cargo_bin("odx")
        .unwrap()
        .args(["convert", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--out-sh"))
        .stdout(predicate::str::contains("--dense-odf"))
        .stdout(predicate::str::contains("--peak-source"));
}

#[test]
fn validate_help_mentions_strict() {
    Command::cargo_bin("odx")
        .unwrap()
        .args(["validate", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--strict"));
}

#[test]
fn info_on_fib_fixture_reports_format_and_dimensions() {
    let fib = fixture_path(FIB_PATH);
    if !fib.exists() {
        eprintln!("skipping missing fixture {}", fib.display());
        return;
    }

    Command::cargo_bin("odx")
        .unwrap()
        .args(["info", fib.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("format: dsistudio_fibgz"))
        .stdout(predicate::str::contains("dimensions: 80 x 98 x 85"));
}

#[test]
fn info_on_pam_fixture_reports_pam_format() {
    let pam = fixture_path(PAM_PATH);
    if !pam.exists() {
        eprintln!("skipping missing fixture {}", pam.display());
        return;
    }

    Command::cargo_bin("odx")
        .unwrap()
        .args(["info", pam.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("format: dipy_pam5"))
        .stdout(predicate::str::contains("dimensions: 2 x 2 x 1"));
}

#[test]
fn info_on_combined_mrtrix_input_reports_sh_and_dpf() {
    let sh = fixture_path(SH_MIF);
    let fixels = fixture_path(FIXELS_MIF);
    if !sh.exists() || !fixels.exists() {
        eprintln!("skipping missing MRtrix fixtures");
        return;
    }

    Command::cargo_bin("odx")
        .unwrap()
        .args([
            "info",
            fixels.to_str().unwrap(),
            "--sh",
            sh.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("format: mrtrix_fixel_dir"))
        .stdout(predicate::str::contains("sh: basis=tournier07 order=8"))
        .stdout(predicate::str::contains("afd"))
        .stdout(predicate::str::contains("disp"));
}

#[test]
fn validate_succeeds_on_real_fixture() {
    let fib = fixture_path(FIB_PATH);
    if !fib.exists() {
        eprintln!("skipping missing fixture {}", fib.display());
        return;
    }

    Command::cargo_bin("odx")
        .unwrap()
        .args(["validate", fib.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("validation: ok"));
}

#[test]
fn validate_fails_on_malformed_odx_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let odx_dir = tmp.path().join("broken.odx");
    create_invalid_odx_dir(&odx_dir);

    Command::cargo_bin("odx")
        .unwrap()
        .args(["validate", odx_dir.to_str().unwrap()])
        .assert()
        .failure()
        .stdout(predicate::str::contains("mask has 2 nonzero voxels"))
        .stderr(predicate::str::contains("validation failed"));
}

#[test]
fn convert_fib_to_odx_directory() {
    let fib = fixture_path(FIB_PATH);
    if !fib.exists() {
        eprintln!("skipping missing fixture {}", fib.display());
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("converted.odx");
    Command::cargo_bin("odx")
        .unwrap()
        .args([
            "convert",
            fib.to_str().unwrap(),
            out.to_str().unwrap(),
            "--odx-layout",
            "directory",
        ])
        .assert()
        .success();

    assert!(out.join("header.json").exists());
}

#[test]
fn convert_mrtrix_fixels_and_sh_to_fz() {
    let sh = fixture_path(SH_MIF);
    let fixels = fixture_path(FIXELS_MIF);
    if !sh.exists() || !fixels.exists() {
        eprintln!("skipping missing MRtrix fixtures");
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("from_mrtrix.fz");
    Command::cargo_bin("odx")
        .unwrap()
        .args([
            "convert",
            fixels.to_str().unwrap(),
            out.to_str().unwrap(),
            "--sh",
            sh.to_str().unwrap(),
        ])
        .assert()
        .success();

    assert!(out.exists());
}

#[test]
fn convert_odx_directory_to_mrtrix_fixels_and_sh() {
    let fib = fixture_path(FIB_PATH);
    if !fib.exists() {
        eprintln!("skipping missing fixture {}", fib.display());
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let odx_dir = tmp.path().join("dataset.odx");
    let out_fixels = tmp.path().join("fixels_out");
    let out_sh = tmp.path().join("fod_out.mif.gz");

    Command::cargo_bin("odx")
        .unwrap()
        .args([
            "convert",
            fib.to_str().unwrap(),
            odx_dir.to_str().unwrap(),
            "--odx-layout",
            "directory",
        ])
        .assert()
        .success();

    Command::cargo_bin("odx")
        .unwrap()
        .args([
            "convert",
            odx_dir.to_str().unwrap(),
            out_fixels.to_str().unwrap(),
            "--out-sh",
            out_sh.to_str().unwrap(),
        ])
        .assert()
        .success();

    assert!(out_fixels.join("index.nii").exists() || out_fixels.join("index.nii.gz").exists());
    assert!(out_sh.exists());
}

#[test]
fn convert_sampled_odf_without_sh_fails() {
    let fixels = fixture_path(FIXELS_MIF);
    if !fixels.exists() {
        eprintln!("skipping missing fixel fixture {}", fixels.display());
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("bad.fz");
    Command::cargo_bin("odx")
        .unwrap()
        .args([
            "convert",
            fixels.to_str().unwrap(),
            out.to_str().unwrap(),
            "--peak-source",
            "sampled-odf",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("PeakSource::SampledOdf"));
}

#[test]
fn convert_refuses_existing_output_without_overwrite() {
    let fib = fixture_path(FIB_PATH);
    if !fib.exists() {
        eprintln!("skipping missing fixture {}", fib.display());
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("existing.odx");
    fs::create_dir_all(&out).unwrap();

    Command::cargo_bin("odx")
        .unwrap()
        .args(["convert", fib.to_str().unwrap(), out.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("already exists"));
}
