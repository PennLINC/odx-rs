use assert_cmd::prelude::*;
use odx_rs::{DType, Header, OdxBuilder, OdxDataset, QC_CLASS_DPF_NAME};
use predicates::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const SH_MIF: &str =
    "../test_data/sub-NDARAE199TDD_ses-1_acq-64dirVARIANTVar1e_space-ACPC_model-ss3t_param-fod_label-WM_dwimap.mif.gz";
const FIXELS_MIF: &str = "../test_data/fixels_mif";
const FIB_PATH: &str =
    "../test_data/sub-NDARAE199TDD_ses-1_acq-64dirVARIANTVar1e_space-ACPC_model-ss3t_dwimap.fib.gz";
const REF_AFFINE_PATH: &str =
    "../test_data/sub-NDARAE199TDD_ses-1_acq-64dirVARIANTVar1e_space-ACPC_model-tensor_param-fa_dwimap.nii.gz";
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

fn create_qc_fixture_odx_dir(path: &Path) {
    let dims = [2u64, 1, 1];
    let mask = vec![1u8, 1u8];
    let mut builder = OdxBuilder::new(Header::identity_affine(), dims, mask);
    builder.push_voxel_peaks(&[[1.0, 0.0, 0.0]]);
    builder.push_voxel_peaks(&[[1.0, 0.0, 0.0]]);
    builder.set_dpf_data(
        "amplitude",
        bytemuck::cast_slice(&[1.0f32, 1.0f32]).to_vec(),
        1,
        DType::Float32,
    );
    builder.set_dpf_data(
        "disp",
        bytemuck::cast_slice(&[0.25f32, 0.75f32]).to_vec(),
        1,
        DType::Float32,
    );
    builder.set_dpf_data(
        "vec2",
        bytemuck::cast_slice(&[1.0f32, 2.0f32, 3.0f32, 4.0f32]).to_vec(),
        2,
        DType::Float32,
    );
    builder.finalize().unwrap().save_directory(path).unwrap();
}

fn create_no_primary_metric_odx_dir(path: &Path) {
    let dims = [1u64, 1, 1];
    let mask = vec![1u8];
    let mut builder = OdxBuilder::new(Header::identity_affine(), dims, mask);
    builder.push_voxel_peaks(&[[1.0, 0.0, 0.0]]);
    builder.set_dpf_data(
        "vec2",
        bytemuck::cast_slice(&[1.0f32, 2.0f32]).to_vec(),
        2,
        DType::Float32,
    );
    builder.finalize().unwrap().save_directory(path).unwrap();
}

/// An SH-carrying input: a narrow lobe along `dir`, lmax 8 tournier07, so the
/// mean-fod path and the FOD reproducibility block have something to chew on.
fn create_sh_input(path: &Path, dir: [f32; 3], dc: f32) {
    let dims = [1u64, 1, 1];
    let sphere = odx_rs::formats::dsistudio_odf8::hemisphere_vertices_ras();
    let amps: Vec<f32> = sphere
        .iter()
        .map(|v| (v[0] * dir[0] + v[1] * dir[1] + v[2] * dir[2]).abs().powi(8))
        .collect();
    let mut row = odx_rs::mrtrix_sh::fit_from_amplitudes(&amps, &sphere, 8).unwrap();
    row[0] += dc;
    let mut builder = OdxBuilder::new(Header::identity_affine(), dims, vec![1u8]);
    builder.set_sh_info(8, "tournier07".to_string());
    builder.set_sh_full_basis(false);
    builder.set_sh_legacy(false);
    builder.set_sh_data(
        "coefficients",
        bytemuck::cast_slice(&row).to_vec(),
        45,
        DType::Float32,
    );
    builder.set_dpv_data(
        "csf",
        bytemuck::cast_slice(&[dc]).to_vec(),
        1,
        DType::Float32,
    );
    builder.skip_all_peaks();
    builder.finalize().unwrap().save_directory(path).unwrap();
}

fn create_combine_input(path: &Path, dir: [f32; 3]) {
    let dims = [1u64, 1, 1];
    let mut builder = OdxBuilder::new(Header::identity_affine(), dims, vec![1u8]);
    builder.push_voxel_peaks(&[dir]);
    builder.set_dpf_data(
        "amplitude",
        bytemuck::cast_slice(&[1.0f32]).to_vec(),
        1,
        DType::Float32,
    );
    builder.finalize().unwrap().save_directory(path).unwrap();
}

#[test]
fn combine_help_lists_methods_and_flags() {
    Command::cargo_bin("odx")
        .unwrap()
        .args(["combine", "--help"])
        .assert()
        .success()
        .stdout(
            predicate::str::contains("--method")
                .and(predicate::str::contains("cluster"))
                .and(predicate::str::contains("mean-fod"))
                .and(predicate::str::contains("--out-cohort")),
        );
}

#[test]
fn combine_runs_and_writes_group_odx_and_cohort() {
    let tmp = tempfile::TempDir::new().unwrap();
    let a = tmp.path().join("a.odx");
    let b = tmp.path().join("b.odx");
    let c = tmp.path().join("c.odx");
    create_combine_input(&a, [0.0, 0.0, 1.0]);
    create_combine_input(&b, [0.0, 0.0, 1.0]);
    create_combine_input(&c, [0.0, 0.087_16, 0.996_19]); // ~5° off +z
    let out_odx = tmp.path().join("group.odx");
    let cohort = tmp.path().join("cohort.csv");
    let persubj = tmp.path().join("persubj");

    // --out-cohort requires --per-subject-odx (cohort rows point at single-column
    // per-subject ODX files, which the ModelArrayIO odx loader consumes).
    Command::cargo_bin("odx")
        .unwrap()
        .arg("combine")
        .arg(&a)
        .arg(&b)
        .arg(&c)
        .args(["--method", "cluster", "--min-subjects", "2", "--out-odx"])
        .arg(&out_odx)
        .arg("--out-cohort")
        .arg(&cohort)
        .arg("--per-subject-odx")
        .arg(&persubj)
        .assert()
        .success();

    assert!(out_odx.exists());
    let ds = OdxDataset::open(&out_odx).unwrap();
    assert_eq!(ds.nb_peaks(), 1);
    // angle_deg is an (n_fixels × n_subjects) matrix — ModelArray's `values` shape
    assert_eq!(ds.get_dpf("angle_deg").unwrap().ncols(), 3);

    let text = fs::read_to_string(&cohort).unwrap();
    assert!(text
        .lines()
        .next()
        .unwrap()
        .starts_with("scalar_name,source_file"));
    assert!(text.contains("angle_deg"));
}

#[test]
fn combine_help_lists_the_template_flags() {
    Command::cargo_bin("odx")
        .unwrap()
        .args(["combine", "--help"])
        .assert()
        .success()
        .stdout(
            predicate::str::contains("--min-coverage")
                .and(predicate::str::contains("--loo"))
                .and(predicate::str::contains("--average-dpv"))
                .and(predicate::str::contains("--lmax"))
                .and(predicate::str::contains("--acc-lmin")),
        );
}

#[test]
fn combine_mean_fod_writes_the_reproducibility_block() {
    let tmp = tempfile::TempDir::new().unwrap();
    let paths: Vec<_> = [
        ([0.0f32, 0.0, 1.0], 0.5f32),
        ([0.0, 0.05, 0.998], 0.6),
        ([0.03, 0.0, 0.999], 0.55),
    ]
    .iter()
    .enumerate()
    .map(|(i, (d, dc))| {
        let p = tmp.path().join(format!("s{i}.odx"));
        create_sh_input(&p, *d, *dc);
        p
    })
    .collect();
    let out_odx = tmp.path().join("group.odx");
    let maps = tmp.path().join("maps");
    let report = tmp.path().join("report.json");

    let out = Command::cargo_bin("odx")
        .unwrap()
        .arg("combine")
        .args(&paths)
        .args(["--method", "mean-fod", "--loo", "on", "--dpv-sd"])
        .arg("--out-odx")
        .arg(&out_odx)
        .arg("--out-dir")
        .arg(&maps)
        .arg("--out-report")
        .arg(&report)
        .arg("--json")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let ds = OdxDataset::open(&out_odx).unwrap();
    // coverage: every input covers the single voxel
    assert_eq!(ds.scalar_dpv_f32("n_subjects").unwrap(), vec![3.0]);
    assert_eq!(ds.scalar_dpv_f32("coverage_frac").unwrap(), vec![1.0]);
    // the three near-identical lobes must correlate almost perfectly
    let acc = ds.scalar_dpv_f32("acc_mean").unwrap()[0];
    assert!(acc > 0.99, "acc_mean {acc}");
    assert!(ds.get_dpv("acc_loo_mean").is_some(), "leave-one-out map missing");
    // csf is the one shared scalar DPV, so it is auto-averaged with its SD
    let csf = ds.scalar_dpv_f32("csf").unwrap()[0];
    assert!((csf - 0.55).abs() < 1e-5, "averaged csf {csf}");
    assert!(ds.get_dpv("csf_sd").is_some(), "--dpv-sd must emit csf_sd");
    // the SH block carries the cohort's real basis metadata
    let h = ds.header();
    assert_eq!(h.sh_basis.as_deref(), Some("tournier07"));
    assert_eq!(h.sh_order, Some(8));
    assert_eq!(h.sh_full_basis, Some(false));
    assert_eq!(h.sh_legacy, Some(false));

    assert!(maps.join("acc_mean.nii.gz").exists());
    assert!(maps.join("coverage_frac.nii.gz").exists());
    assert!(maps.join("l0_cv.nii.gz").exists());

    let json: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(json["subjects"].as_array().unwrap().len(), 3);
    assert_eq!(json["loo"], "on");
    assert!(json["outliers"].as_array().unwrap().is_empty());
    assert!(json["subjects"][0]["mean_acc"].as_f64().unwrap().is_finite());
    assert!(report.exists(), "--out-report must write the same report to disk");
}

#[test]
fn combine_min_coverage_one_is_an_intersection() {
    let tmp = tempfile::TempDir::new().unwrap();
    // Two voxels; b masks only the second.
    let mk = |name: &str, mask: Vec<u8>, n: usize| {
        let p = tmp.path().join(name);
        let mut builder = OdxBuilder::new(Header::identity_affine(), [1, 1, 2], mask);
        let mut row = vec![0.0f32; 45 * n];
        for v in 0..n {
            row[v * 45] = 1.0;
        }
        builder.set_sh_info(8, "tournier07".to_string());
        builder.set_sh_full_basis(false);
        builder.set_sh_legacy(false);
        builder.set_sh_data(
            "coefficients",
            bytemuck::cast_slice(&row).to_vec(),
            45,
            DType::Float32,
        );
        builder.skip_all_peaks();
        builder.finalize().unwrap().save_directory(&p).unwrap();
        p
    };
    let a = mk("a.odx", vec![1, 1], 2);
    let b = mk("b.odx", vec![0, 1], 1);
    let out_odx = tmp.path().join("group.odx");

    Command::cargo_bin("odx")
        .unwrap()
        .arg("combine")
        .arg(&a)
        .arg(&b)
        .args(["--method", "mean-fod", "--min-coverage", "1"])
        .arg("--out-odx")
        .arg(&out_odx)
        .assert()
        .success();

    let ds = OdxDataset::open(&out_odx).unwrap();
    assert_eq!(ds.nb_voxels(), 1, "--min-coverage 1 keeps only the shared voxel");
}

#[test]
fn combine_rejects_out_of_range_min_coverage() {
    let tmp = tempfile::TempDir::new().unwrap();
    let a = tmp.path().join("a.odx");
    create_combine_input(&a, [0.0, 0.0, 1.0]);
    Command::cargo_bin("odx")
        .unwrap()
        .arg("combine")
        .arg(&a)
        .args(["--min-coverage", "1.5"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--min-coverage must be in [0, 1]"));
}

#[test]
fn combine_method_comparison_guards() {
    let tmp = tempfile::TempDir::new().unwrap();
    let a = tmp.path().join("a.odx");
    let b = tmp.path().join("b.odx");
    create_combine_input(&a, [0.0, 0.0, 1.0]);
    create_combine_input(&b, [0.0, 0.0, 1.0]);

    // --out-cohort without --per-subject-odx is rejected
    Command::cargo_bin("odx")
        .unwrap()
        .arg("combine")
        .arg(&a)
        .arg(&b)
        .arg("--out-cohort")
        .arg(tmp.path().join("c.csv"))
        .arg("--out-odx")
        .arg(tmp.path().join("g.odx"))
        .assert()
        .failure()
        .stderr(predicate::str::contains("per-subject-odx"));

    // --reference-method without --method-column is rejected
    Command::cargo_bin("odx")
        .unwrap()
        .arg("combine")
        .arg(&a)
        .arg(&b)
        .args(["--reference-method", "abcd", "--out-odx"])
        .arg(tmp.path().join("g2.odx"))
        .assert()
        .failure()
        .stderr(predicate::str::contains("method-column"));
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
        .stdout(predicate::str::contains("--dense-odf"));
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
fn qc_help_mentions_threshold_and_primary_dpf() {
    Command::cargo_bin("odx")
        .unwrap()
        .args(["qc", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--primary-dpf"))
        .stdout(predicate::str::contains("--threshold"))
        .stdout(predicate::str::contains("--angle-deg"))
        .stdout(predicate::str::contains("--write-qc-class"))
        .stdout(predicate::str::contains("--overwrite-qc-class"));
}

#[test]
fn info_on_fib_fixture_reports_format_and_dimensions() {
    let fib = fixture_path(FIB_PATH);
    let reference = fixture_path(REF_AFFINE_PATH);
    if !fib.exists() || !reference.exists() {
        eprintln!("skipping missing fixture {}", fib.display());
        return;
    }

    Command::cargo_bin("odx")
        .unwrap()
        .args([
            "info",
            fib.to_str().unwrap(),
            "--reference-affine",
            reference.to_str().unwrap(),
        ])
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
    let reference = fixture_path(REF_AFFINE_PATH);
    if !fib.exists() || !reference.exists() {
        eprintln!("skipping missing fixture {}", fib.display());
        return;
    }

    Command::cargo_bin("odx")
        .unwrap()
        .args([
            "validate",
            fib.to_str().unwrap(),
            "--reference-affine",
            reference.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("validation: ok"));
}

#[test]
fn info_on_fib_fixture_requires_reference_affine_when_trans_is_missing() {
    let fib = fixture_path(FIB_PATH);
    if !fib.exists() {
        eprintln!("skipping missing fixture {}", fib.display());
        return;
    }

    Command::cargo_bin("odx")
        .unwrap()
        .args(["info", fib.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "DSI Studio file has no spatial affine ('trans' field)",
        ));
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
fn qc_text_output_reports_headline_metrics() {
    let tmp = tempfile::tempdir().unwrap();
    let odx_dir = tmp.path().join("qc_fixture.odx");
    create_qc_fixture_odx_dir(&odx_dir);

    Command::cargo_bin("odx")
        .unwrap()
        .args([
            "qc",
            odx_dir.to_str().unwrap(),
            "--threshold",
            "all",
            "--primary-dpf",
            "amplitude",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("primary_metric: amplitude"))
        .stdout(predicate::str::contains("connected_fixels: 2"))
        .stdout(predicate::str::contains("coherence_index: 1.000000"))
        .stdout(predicate::str::contains("skipped_dpf: vec2"));
}

#[test]
fn qc_json_output_serializes_report() {
    let tmp = tempfile::tempdir().unwrap();
    let odx_dir = tmp.path().join("qc_fixture.odx");
    create_qc_fixture_odx_dir(&odx_dir);

    let output = Command::cargo_bin("odx")
        .unwrap()
        .args([
            "qc",
            odx_dir.to_str().unwrap(),
            "--threshold",
            "all",
            "--primary-dpf",
            "amplitude",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());

    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json["primary_metric"], "amplitude");
    assert_eq!(json["connected_fixels"], 2);
    assert_eq!(json["per_dpf"]["disp"]["connected"]["count"], 2);
    assert_eq!(json["skipped_dpf"][0], "vec2");
}

#[test]
fn qc_reports_missing_primary_metric_failure() {
    let tmp = tempfile::tempdir().unwrap();
    let odx_dir = tmp.path().join("no_primary.odx");
    create_no_primary_metric_odx_dir(&odx_dir);

    Command::cargo_bin("odx")
        .unwrap()
        .args(["qc", odx_dir.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "no usable primary DPF metric found",
        ));
}

#[test]
fn qc_rejects_non_scalar_requested_primary_metric() {
    let tmp = tempfile::tempdir().unwrap();
    let odx_dir = tmp.path().join("qc_fixture.odx");
    create_qc_fixture_odx_dir(&odx_dir);

    Command::cargo_bin("odx")
        .unwrap()
        .args([
            "qc",
            odx_dir.to_str().unwrap(),
            "--primary-dpf",
            "vec2",
            "--threshold",
            "all",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("expected a scalar field"));
}

#[test]
fn qc_can_write_qc_class_dpf_to_odx_input() {
    let tmp = tempfile::tempdir().unwrap();
    let odx_dir = tmp.path().join("qc_fixture.odx");
    create_qc_fixture_odx_dir(&odx_dir);

    Command::cargo_bin("odx")
        .unwrap()
        .args([
            "qc",
            odx_dir.to_str().unwrap(),
            "--threshold",
            "all",
            "--primary-dpf",
            "amplitude",
            "--write-qc-class",
        ])
        .assert()
        .success();

    assert!(odx_dir.join("dpf").join("qc_class.uint8").exists());
    let reopened = OdxDataset::open(&odx_dir).unwrap();
    assert_eq!(
        reopened.scalar_dpf_f32(QC_CLASS_DPF_NAME).unwrap(),
        vec![2.0, 2.0]
    );
}

#[test]
fn qc_write_qc_class_respects_overwrite_flag() {
    let tmp = tempfile::tempdir().unwrap();
    let odx_dir = tmp.path().join("qc_fixture.odx");
    create_qc_fixture_odx_dir(&odx_dir);
    fs::write(odx_dir.join("dpf").join("qc_class.uint8"), [1u8, 1u8]).unwrap();

    Command::cargo_bin("odx")
        .unwrap()
        .args([
            "qc",
            odx_dir.to_str().unwrap(),
            "--threshold",
            "all",
            "--primary-dpf",
            "amplitude",
            "--write-qc-class",
        ])
        .assert()
        .success();

    let reopened = OdxDataset::open(&odx_dir).unwrap();
    assert_eq!(
        reopened.scalar_dpf_f32(QC_CLASS_DPF_NAME).unwrap(),
        vec![1.0, 1.0]
    );

    Command::cargo_bin("odx")
        .unwrap()
        .args([
            "qc",
            odx_dir.to_str().unwrap(),
            "--threshold",
            "all",
            "--primary-dpf",
            "amplitude",
            "--write-qc-class",
            "--overwrite-qc-class",
        ])
        .assert()
        .success();

    let reopened = OdxDataset::open(&odx_dir).unwrap();
    assert_eq!(
        reopened.scalar_dpf_f32(QC_CLASS_DPF_NAME).unwrap(),
        vec![2.0, 2.0]
    );
}

#[test]
fn qc_write_qc_class_rejects_non_odx_input() {
    let fixels = fixture_path(FIXELS_MIF);
    if !fixels.exists() {
        eprintln!("skipping missing fixture {}", fixels.display());
        return;
    }

    Command::cargo_bin("odx")
        .unwrap()
        .args([
            "qc",
            fixels.to_str().unwrap(),
            "--write-qc-class",
            "--threshold",
            "all",
            "--primary-dpf",
            "afd",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "--write-qc-class requires an ODX directory or .odx archive input",
        ));
}

#[test]
fn convert_fib_to_odx_directory() {
    let fib = fixture_path(FIB_PATH);
    let reference = fixture_path(REF_AFFINE_PATH);
    if !fib.exists() || !reference.exists() {
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
            "--output-format",
            "odx-directory",
            "--reference-affine",
            reference.to_str().unwrap(),
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
    let reference = fixture_path(REF_AFFINE_PATH);
    if !fib.exists() || !reference.exists() {
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
            "--output-format",
            "odx-directory",
            "--reference-affine",
            reference.to_str().unwrap(),
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

#[test]
fn transform_help_lists_subcommand() {
    Command::cargo_bin("odx")
        .unwrap()
        .args(["transform", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Composite.h5"))
        .stdout(predicate::str::contains("--transform"))
        .stdout(predicate::str::contains("--transform-inverse"))
        .stdout(predicate::str::contains("--mode"))
        .stdout(predicate::str::contains("mrtrix"))
        .stdout(predicate::str::contains("ants"))
        .stdout(predicate::str::contains("--reference"))
        .stdout(predicate::str::contains("--modulate"))
        .stdout(predicate::str::contains("--invert"));
}

#[test]
fn transform_affine_only_without_reference_errors_clearly() {
    let fixture = "../nitransforms/nitransforms/tests/data/affine-antsComposite.h5";
    if !Path::new(fixture).exists() {
        eprintln!("skipping: {} not present", fixture);
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let in_odx = tmp.path().join("in.odx");
    create_qc_fixture_odx_dir(&in_odx);

    let out_odx = tmp.path().join("out.odx");

    Command::cargo_bin("odx")
        .unwrap()
        .args([
            "transform",
            in_odx.to_str().unwrap(),
            out_odx.to_str().unwrap(),
            "--transform",
            fixture,
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("affine-only"))
        .stderr(predicate::str::contains("--reference"));
}

#[test]
fn transform_refuses_existing_output_without_overwrite() {
    let fixture = "../nitransforms/nitransforms/tests/data/affine-antsComposite.h5";
    if !Path::new(fixture).exists() {
        eprintln!("skipping: {} not present", fixture);
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let in_odx = tmp.path().join("in.odx");
    create_qc_fixture_odx_dir(&in_odx);

    // Pre-create the output directory.
    let out_odx = tmp.path().join("existing.odx");
    fs::create_dir_all(&out_odx).unwrap();

    Command::cargo_bin("odx")
        .unwrap()
        .args([
            "transform",
            in_odx.to_str().unwrap(),
            out_odx.to_str().unwrap(),
            "--transform",
            fixture,
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("already exists"));
}
