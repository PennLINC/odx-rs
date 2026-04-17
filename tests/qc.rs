use std::path::{Path, PathBuf};

use odx_rs::{
    compute_fixel_qc, mrtrix, write_qc_class_dpf, DType, FixelQcClass, FixelQcOptions, OdxBuilder,
    OdxDataset, ThresholdMode, QC_CLASS_DPF_NAME,
};

const FIXELS_NII: &str = "../test_data/fixels_nii";
const FIXELS_MIF: &str = "../test_data/fixels_mif";

#[derive(Clone)]
struct TestVoxel {
    coord: [usize; 3],
    peaks: Vec<[f32; 3]>,
}

fn fixture_path(rel: &str) -> PathBuf {
    Path::new(rel).to_path_buf()
}

fn normalize(dir: [f32; 3]) -> [f32; 3] {
    let norm = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    [dir[0] / norm, dir[1] / norm, dir[2] / norm]
}

fn build_qc_dataset(
    dims: [u64; 3],
    mut voxels: Vec<TestVoxel>,
    scalar_dpf: Vec<(&str, Vec<f32>)>,
    other_dpf: Vec<(&str, Vec<f32>, usize)>,
) -> OdxDataset {
    let flat = |coord: [usize; 3]| -> usize {
        coord[0] * dims[1] as usize * dims[2] as usize + coord[1] * dims[2] as usize + coord[2]
    };
    voxels.sort_by_key(|voxel| flat(voxel.coord));

    let mut mask = vec![0u8; dims[0] as usize * dims[1] as usize * dims[2] as usize];
    let mut total_fixels = 0usize;
    for voxel in &voxels {
        mask[flat(voxel.coord)] = 1;
        total_fixels += voxel.peaks.len();
    }

    let mut builder = OdxBuilder::new(odx_rs::Header::identity_affine(), dims, mask);
    for voxel in &voxels {
        builder.push_voxel_peaks(&voxel.peaks);
    }

    for (name, values) in scalar_dpf {
        assert_eq!(values.len(), total_fixels);
        builder.set_dpf_data(
            name,
            bytemuck::cast_slice(&values).to_vec(),
            1,
            DType::Float32,
        );
    }

    for (name, values, ncols) in other_dpf {
        assert_eq!(values.len(), total_fixels * ncols);
        builder.set_dpf_data(
            name,
            bytemuck::cast_slice(&values).to_vec(),
            ncols,
            DType::Float32,
        );
    }

    builder.finalize().unwrap()
}

fn assert_report_invariants(report: &odx_rs::FixelQcReport) {
    assert_eq!(
        report.connected_fixels + report.disconnected_fixels + report.excluded_fixels,
        report.total_fixels
    );
    assert_eq!(
        report.connected_fixels + report.disconnected_fixels,
        report.evaluated_fixels
    );
    if let (Some(coherence), Some(incoherence)) = (report.coherence_index, report.incoherence_index)
    {
        let sum = coherence + incoherence;
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "coherence + incoherence should equal 1, found {sum}"
        );
    }
}

fn write_u8_dpf(builder: &mut OdxBuilder, name: &str, values: &[u8]) {
    builder.set_dpf_data(name, values.to_vec(), 1, DType::UInt8);
}

#[test]
fn connected_pair_reports_all_fixels_connected_and_skips_vector_dpf() {
    let odx = build_qc_dataset(
        [2, 1, 1],
        vec![
            TestVoxel {
                coord: [0, 0, 0],
                peaks: vec![[1.0, 0.0, 0.0]],
            },
            TestVoxel {
                coord: [1, 0, 0],
                peaks: vec![[1.0, 0.0, 0.0]],
            },
        ],
        vec![("amplitude", vec![1.0, 1.0]), ("disp", vec![0.25, 0.75])],
        vec![("vec2", vec![1.0, 2.0, 3.0, 4.0], 2)],
    );

    let computation = compute_fixel_qc(
        &odx,
        &FixelQcOptions {
            threshold: ThresholdMode::All,
            ..Default::default()
        },
    )
    .unwrap();
    let report = &computation.report;

    assert_report_invariants(&report);
    assert_eq!(report.total_fixels, 2);
    assert_eq!(report.connected_fixels, 2);
    assert_eq!(report.disconnected_fixels, 0);
    assert_eq!(report.excluded_fixels, 0);
    assert_eq!(report.coherence_index, Some(1.0));
    assert_eq!(report.incoherence_index, Some(0.0));
    assert_eq!(report.connected_to_disconnected_ratio, None);
    assert_eq!(report.skipped_dpf, vec!["vec2"]);

    let disp = report.per_dpf.get("disp").unwrap();
    assert_eq!(disp.connected.count, 2);
    assert_eq!(disp.connected.mean, Some(0.5));
    assert_eq!(disp.connected.median, Some(0.5));
    assert_eq!(disp.disconnected.count, 0);
    assert_eq!(
        computation.classes,
        vec![FixelQcClass::Connected, FixelQcClass::Connected]
    );
}

#[test]
fn disconnected_fixel_is_counted_and_weighted() {
    let odx = build_qc_dataset(
        [3, 1, 1],
        vec![TestVoxel {
            coord: [0, 0, 0],
            peaks: vec![[1.0, 0.0, 0.0]],
        }],
        vec![("amplitude", vec![2.0])],
        vec![],
    );

    let computation = compute_fixel_qc(
        &odx,
        &FixelQcOptions {
            threshold: ThresholdMode::All,
            ..Default::default()
        },
    )
    .unwrap();
    let report = &computation.report;

    assert_report_invariants(&report);
    assert_eq!(report.connected_fixels, 0);
    assert_eq!(report.disconnected_fixels, 1);
    assert_eq!(report.connected_to_disconnected_ratio, Some(0.0));
    assert_eq!(report.coherence_index, Some(0.0));
    assert_eq!(report.incoherence_index, Some(1.0));
    assert_eq!(computation.classes, vec![FixelQcClass::Disconnected]);
}

#[test]
fn diagonal_neighbor_is_accepted_by_kernel() {
    let dir = normalize([1.0, 1.0, 1.0]);
    let odx = build_qc_dataset(
        [2, 2, 2],
        vec![
            TestVoxel {
                coord: [0, 0, 0],
                peaks: vec![dir],
            },
            TestVoxel {
                coord: [1, 1, 1],
                peaks: vec![dir],
            },
        ],
        vec![("amplitude", vec![1.0, 1.0])],
        vec![],
    );

    let computation = compute_fixel_qc(
        &odx,
        &FixelQcOptions {
            threshold: ThresholdMode::All,
            ..Default::default()
        },
    )
    .unwrap();
    let report = &computation.report;

    assert_report_invariants(&report);
    assert_eq!(report.connected_fixels, 2);
    assert_eq!(report.disconnected_fixels, 0);
    assert_eq!(
        computation.classes,
        vec![FixelQcClass::Connected, FixelQcClass::Connected]
    );
}

#[test]
fn best_matching_neighbor_fixel_marks_only_aligned_pair_connected() {
    let odx = build_qc_dataset(
        [2, 1, 1],
        vec![
            TestVoxel {
                coord: [0, 0, 0],
                peaks: vec![[1.0, 0.0, 0.0]],
            },
            TestVoxel {
                coord: [1, 0, 0],
                peaks: vec![[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            },
        ],
        vec![("amplitude", vec![1.0, 2.0, 3.0])],
        vec![],
    );

    let computation = compute_fixel_qc(
        &odx,
        &FixelQcOptions {
            threshold: ThresholdMode::All,
            ..Default::default()
        },
    )
    .unwrap();
    let report = &computation.report;

    assert_report_invariants(&report);
    assert_eq!(report.connected_fixels, 2);
    assert_eq!(report.disconnected_fixels, 1);
    let amplitude = report.per_dpf.get("amplitude").unwrap();
    assert_eq!(amplitude.connected.count, 2);
    assert_eq!(amplitude.connected.mean, Some(2.0));
    assert_eq!(amplitude.connected.median, Some(2.0));
    assert_eq!(amplitude.disconnected.count, 1);
    assert_eq!(amplitude.disconnected.mean, Some(2.0));
    assert_eq!(
        computation.classes,
        vec![
            FixelQcClass::Connected,
            FixelQcClass::Disconnected,
            FixelQcClass::Connected,
        ]
    );
}

#[test]
fn positive_threshold_excludes_nonpositive_fixels() {
    let odx = build_qc_dataset(
        [2, 1, 1],
        vec![
            TestVoxel {
                coord: [0, 0, 0],
                peaks: vec![[1.0, 0.0, 0.0]],
            },
            TestVoxel {
                coord: [1, 0, 0],
                peaks: vec![[1.0, 0.0, 0.0]],
            },
        ],
        vec![("amplitude", vec![0.0, 1.0])],
        vec![],
    );

    let computation = compute_fixel_qc(
        &odx,
        &FixelQcOptions {
            threshold: ThresholdMode::Positive,
            ..Default::default()
        },
    )
    .unwrap();
    let report = &computation.report;

    assert_report_invariants(&report);
    assert_eq!(report.evaluated_fixels, 1);
    assert_eq!(report.excluded_fixels, 1);
    assert_eq!(report.connected_fixels, 0);
    assert_eq!(report.disconnected_fixels, 1);
    assert_eq!(report.threshold_value, Some(0.0));
    assert_eq!(
        computation.classes,
        vec![FixelQcClass::ThresholdedOut, FixelQcClass::Disconnected]
    );
}

#[test]
fn qc_class_is_reserved_and_excluded_from_summaries() {
    let dims = [2u64, 1, 1];
    let mask = vec![1u8, 1u8];
    let mut builder = OdxBuilder::new(odx_rs::Header::identity_affine(), dims, mask);
    builder.push_voxel_peaks(&[[1.0, 0.0, 0.0]]);
    builder.push_voxel_peaks(&[[1.0, 0.0, 0.0]]);
    builder.set_dpf_data(
        "amplitude",
        bytemuck::cast_slice(&[1.0f32, 1.0f32]).to_vec(),
        1,
        DType::Float32,
    );
    write_u8_dpf(&mut builder, QC_CLASS_DPF_NAME, &[1, 2]);
    let odx = builder.finalize().unwrap();

    let computation = compute_fixel_qc(
        &odx,
        &FixelQcOptions {
            threshold: ThresholdMode::All,
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(computation.report.primary_metric, "amplitude");
    assert!(!computation.report.per_dpf.contains_key(QC_CLASS_DPF_NAME));
    assert!(compute_fixel_qc(
        &odx,
        &FixelQcOptions {
            primary_metric: Some(QC_CLASS_DPF_NAME.into()),
            threshold: ThresholdMode::All,
            ..Default::default()
        },
    )
    .is_err());
}

#[test]
fn write_qc_class_dpf_updates_directory_and_archive() {
    let odx = build_qc_dataset(
        [2, 1, 1],
        vec![
            TestVoxel {
                coord: [0, 0, 0],
                peaks: vec![[1.0, 0.0, 0.0]],
            },
            TestVoxel {
                coord: [1, 0, 0],
                peaks: vec![[1.0, 0.0, 0.0]],
            },
        ],
        vec![("amplitude", vec![1.0, 1.0])],
        vec![],
    );
    let computation = compute_fixel_qc(
        &odx,
        &FixelQcOptions {
            threshold: ThresholdMode::All,
            ..Default::default()
        },
    )
    .unwrap();

    let temp = tempfile::tempdir().unwrap();
    let odx_dir = temp.path().join("qc_fixture.odx");
    odx.save_directory(&odx_dir).unwrap();

    write_qc_class_dpf(&odx_dir, &computation.classes, false).unwrap();
    assert!(odx_dir.join("dpf").join("qc_class.uint8").exists());
    let reopened_dir = OdxDataset::open(&odx_dir).unwrap();
    assert_eq!(
        reopened_dir.scalar_dpf_f32(QC_CLASS_DPF_NAME).unwrap(),
        vec![2.0, 2.0]
    );

    let odx_archive = temp.path().join("qc_fixture_archive.odx");
    odx.save_archive(&odx_archive).unwrap();
    write_qc_class_dpf(&odx_archive, &computation.classes, false).unwrap();

    let archive_file = std::fs::File::open(&odx_archive).unwrap();
    let mut archive = zip::ZipArchive::new(archive_file).unwrap();
    assert!(archive.by_name("dpf/qc_class.uint8").is_ok());
    let reopened_archive = OdxDataset::open(&odx_archive).unwrap();
    assert_eq!(
        reopened_archive.scalar_dpf_f32(QC_CLASS_DPF_NAME).unwrap(),
        vec![2.0, 2.0]
    );
}

#[test]
fn write_qc_class_dpf_respects_overwrite_and_row_validation() {
    let odx = build_qc_dataset(
        [2, 1, 1],
        vec![
            TestVoxel {
                coord: [0, 0, 0],
                peaks: vec![[1.0, 0.0, 0.0]],
            },
            TestVoxel {
                coord: [1, 0, 0],
                peaks: vec![[1.0, 0.0, 0.0]],
            },
        ],
        vec![("amplitude", vec![1.0, 1.0])],
        vec![],
    );
    let temp = tempfile::tempdir().unwrap();
    let odx_dir = temp.path().join("qc_fixture.odx");
    odx.save_directory(&odx_dir).unwrap();

    write_qc_class_dpf(
        &odx_dir,
        &[FixelQcClass::Disconnected, FixelQcClass::Disconnected],
        false,
    )
    .unwrap();
    write_qc_class_dpf(
        &odx_dir,
        &[FixelQcClass::Connected, FixelQcClass::Connected],
        false,
    )
    .unwrap();
    let reopened = OdxDataset::open(&odx_dir).unwrap();
    assert_eq!(
        reopened.scalar_dpf_f32(QC_CLASS_DPF_NAME).unwrap(),
        vec![1.0, 1.0]
    );

    write_qc_class_dpf(
        &odx_dir,
        &[FixelQcClass::Connected, FixelQcClass::Connected],
        true,
    )
    .unwrap();
    let reopened = OdxDataset::open(&odx_dir).unwrap();
    assert_eq!(
        reopened.scalar_dpf_f32(QC_CLASS_DPF_NAME).unwrap(),
        vec![2.0, 2.0]
    );

    let err = write_qc_class_dpf(&odx_dir, &[FixelQcClass::Connected], true).unwrap_err();
    assert!(err.to_string().contains("has 1 rows, expected 2"));
}

#[test]
fn mrtrix_mif_and_nii_reports_match() {
    let nii = fixture_path(FIXELS_NII);
    let mif = fixture_path(FIXELS_MIF);
    if !nii.exists() || !mif.exists() {
        eprintln!("skipping missing fixel fixtures");
        return;
    }

    let odx_nii = mrtrix::load_mrtrix_fixels(&nii).unwrap();
    let odx_mif = mrtrix::load_mrtrix_fixels(&mif).unwrap();
    let options = FixelQcOptions {
        primary_metric: Some("afd".into()),
        threshold: ThresholdMode::Otsu,
        ..Default::default()
    };

    let report_nii = compute_fixel_qc(&odx_nii, &options).unwrap();
    let report_mif = compute_fixel_qc(&odx_mif, &options).unwrap();
    assert_eq!(report_nii.report, report_mif.report);
    assert_eq!(report_nii.classes, report_mif.classes);
    assert_report_invariants(&report_nii.report);
}
