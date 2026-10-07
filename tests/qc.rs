use std::path::{Path, PathBuf};

use odx_rs::{
    check_btable, coherence_threshold_elasticity, compute_fixel_chains, compute_fixel_qc,
    compute_primary_coherence, mrtrix, write_qc_class_dpf, CoherenceMode, DType, FixelQcClass,
    FixelQcOptions, OdxBuilder, OdxDataset, ThresholdMode, QC_CLASS_DPF_NAME,
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

/// NaN in an auxiliary DPF means "undefined for this fixel" -- `odx compare`
/// writes it for unmatched fixels and `odx combine` for group fixels with too
/// few contributors. QC must summarize the defined values rather than refuse
/// the file, or the crate cannot read back what it just wrote.
#[test]
fn nonfinite_auxiliary_dpf_values_are_skipped_not_rejected() {
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
        vec![
            ("amplitude", vec![1.0, 1.0]),
            ("dispersion", vec![f32::NAN, 0.5]),
        ],
        vec![],
    );

    let computation = compute_fixel_qc(
        &odx,
        &FixelQcOptions {
            threshold: ThresholdMode::All,
            ..Default::default()
        },
    )
    .expect("a NaN in an auxiliary DPF must not fail the QC pass");
    let disp = computation.report.per_dpf.get("dispersion").unwrap();
    assert_eq!(
        disp.connected.count, 1,
        "only the finite value is summarized"
    );
    assert_eq!(disp.connected.mean, Some(0.5));
}

/// The primary metric drives thresholding, so a NaN there is still fatal.
#[test]
fn nonfinite_primary_dpf_is_still_rejected() {
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
        vec![("amplitude", vec![f32::NAN, 1.0])],
        vec![],
    );

    let err = compute_fixel_qc(
        &odx,
        &FixelQcOptions {
            threshold: ThresholdMode::All,
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("primary DPF"), "{err}");
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

// ---------------------------------------------------------------------------
// Grid geometry, primary-fibre coherence and the b-table check
// ---------------------------------------------------------------------------

const LPS_2MM: [[f64; 4]; 4] = [
    [-2.0, 0.0, 0.0, 30.0],
    [0.0, -2.0, 0.0, 40.0],
    [0.0, 0.0, 2.0, -20.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// Fixels given per voxel as `(world direction, amplitude)` on an arbitrary grid.
fn build_grid_dataset(
    affine: [[f64; 4]; 4],
    dims: [u64; 3],
    voxels: Vec<([usize; 3], Vec<([f32; 3], f32)>)>,
) -> OdxDataset {
    let flat = |c: [usize; 3]| (c[0] * dims[1] as usize + c[1]) * dims[2] as usize + c[2];
    let mut voxels = voxels;
    voxels.sort_by_key(|(coord, _)| flat(*coord));
    let mut mask = vec![0u8; (dims[0] * dims[1] * dims[2]) as usize];
    let mut amplitude = Vec::new();
    for (coord, _) in &voxels {
        assert_eq!(mask[flat(*coord)], 0, "voxel {coord:?} listed twice");
        mask[flat(*coord)] = 1;
    }
    let mut builder = OdxBuilder::new(affine, dims, mask);
    for (_, fixels) in &voxels {
        let dirs: Vec<[f32; 3]> = fixels.iter().map(|(d, _)| normalize(*d)).collect();
        builder.push_voxel_peaks(&dirs);
        amplitude.extend(fixels.iter().map(|(_, a)| *a));
    }
    builder.set_dpf_data(
        "amplitude",
        bytemuck::cast_slice(&amplitude).to_vec(),
        1,
        DType::Float32,
    );
    builder.finalize().unwrap()
}

fn world_dir(affine: &[[f64; 4]; 4], index_dir: [f64; 3]) -> [f32; 3] {
    let v: Vec<f64> = (0..3)
        .map(|r| (0..3).map(|c| affine[r][c] * index_dir[c]).sum())
        .collect();
    normalize([v[0] as f32, v[1] as f32, v[2] as f32])
}

/// A straight one-voxel-wide chain stepping by `step` in index space whose
/// fixels point along the physical direction of that step.
fn index_chain(
    affine: &[[f64; 4]; 4],
    start: [usize; 3],
    step: [i64; 3],
    len: usize,
) -> Vec<([usize; 3], Vec<([f32; 3], f32)>)> {
    let dir = world_dir(affine, step.map(|s| s as f64));
    (0..len as i64)
        .map(|i| {
            let c = [0, 1, 2].map(|a| (start[a] as i64 + i * step[a]) as usize);
            (c, vec![(dir, 1.0)])
        })
        .collect()
}

fn positive_options() -> FixelQcOptions {
    FixelQcOptions {
        threshold: ThresholdMode::Positive,
        ..Default::default()
    }
}

#[test]
fn fixel_qc_steps_through_the_affine_on_an_lps_grid() {
    // Index step (1,0,1) is physical (-1,0,1) on LPS. Comparing world
    // directions with raw index offsets scored this chain fully disconnected.
    let ds = build_grid_dataset(
        LPS_2MM,
        [6, 6, 6],
        index_chain(&LPS_2MM, [1, 2, 1], [1, 0, 1], 4),
    );
    let report = compute_fixel_qc(&ds, &positive_options()).unwrap().report;
    assert_eq!(report.connected_fixels, 4);
    assert_eq!(report.coherence_index, Some(1.0));
}

#[test]
fn fixel_qc_uses_the_physical_trajectory_on_anisotropic_voxels() {
    // 1x1x2 mm voxels: index step (1,0,1) is 63.4° from x physically, not 45°.
    let aniso = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let ds = build_grid_dataset(
        aniso,
        [6, 6, 6],
        index_chain(&aniso, [1, 2, 1], [1, 0, 1], 4),
    );
    let report = compute_fixel_qc(&ds, &positive_options()).unwrap().report;
    assert_eq!(report.coherence_index, Some(1.0));
}

#[test]
fn primary_coherence_connects_fibres_between_lattice_directions() {
    // A slab of fibres 22.5° from x in the xy plane: more than 15° from every
    // lattice direction, so the fixel metric can never connect them, while the
    // primary-fibre metric rounds each to the x step and finds its neighbour.
    let a = 22.5f32.to_radians();
    let dir = [a.cos(), a.sin(), 0.0];
    let voxels = (1..7)
        .flat_map(|x| (1..7).map(move |y| ([x, y, 2], vec![(dir, 1.0)])))
        .collect();
    let ds = build_grid_dataset(odx_rs::Header::identity_affine(), [8, 8, 5], voxels);

    let fixel = compute_fixel_qc(&ds, &positive_options()).unwrap().report;
    assert_eq!(fixel.coherence_index, Some(0.0));

    let primary = compute_primary_coherence(&ds, &FixelQcOptions::default()).unwrap();
    assert_eq!(primary.voxels_with_fixels, 36);
    assert_eq!(primary.evaluated_voxels, 36);
    assert_eq!(primary.coherence_index, Some(1.0));
}

#[test]
fn primary_coherence_scores_only_the_strongest_fixel() {
    let x = [1.0, 0.0, 0.0];
    let y = [0.0, 1.0, 0.0];
    let chain = |strong: [f32; 3], weak: [f32; 3]| {
        (1..5)
            .map(|i| ([i, 3, 3], vec![(weak, 0.2), (strong, 1.0)]))
            .collect::<Vec<_>>()
    };
    let identity = odx_rs::Header::identity_affine();
    let along = build_grid_dataset(identity, [6, 7, 7], chain(x, y));
    let across = build_grid_dataset(identity, [6, 7, 7], chain(y, x));
    let opts = FixelQcOptions::default();
    assert_eq!(
        compute_primary_coherence(&along, &opts)
            .unwrap()
            .coherence_index,
        Some(1.0)
    );
    assert_eq!(
        compute_primary_coherence(&across, &opts)
            .unwrap()
            .coherence_index,
        Some(0.0)
    );
}

#[test]
fn primary_coherence_rejects_bad_angles() {
    let ds = build_grid_dataset(
        odx_rs::Header::identity_affine(),
        [4, 4, 4],
        vec![([1, 1, 1], vec![([1.0, 0.0, 0.0], 1.0)])],
    );
    let opts = FixelQcOptions {
        angle_degrees: 120.0,
        ..Default::default()
    };
    assert!(compute_primary_coherence(&ds, &opts).is_err());
}

/// Signed permutation for a `BTABLE_CANDIDATE_LABELS` entry, written out
/// independently of the library: new[k] = old[label[k]], then the flip.
fn label_matrix(label: &str) -> [[f64; 3]; 3] {
    let mut m = [[0.0; 3]; 3];
    for (row, ch) in label[..3].chars().enumerate() {
        m[row][ch.to_digit(10).unwrap() as usize] = 1.0;
    }
    if let Some(axis) = label
        .strip_prefix(&label[..3])
        .and_then(|f| f.strip_prefix('f'))
    {
        let row = "xyz".find(axis).unwrap();
        m[row] = m[row].map(|v| -v);
    }
    m
}

/// Six separated tubes (three axes, three face diagonals): only the identity
/// keeps every tube's fibres pointing along the tube.
fn tube_phantom(affine: [[f64; 4]; 4], corrupt: [[f64; 3]; 3]) -> OdxDataset {
    let tubes: [([usize; 3], [i64; 3]); 6] = [
        ([2, 4, 4], [1, 0, 0]),
        ([14, 2, 4], [0, 1, 0]),
        ([4, 14, 2], [0, 0, 1]),
        ([12, 12, 14], [1, 1, 0]),
        ([4, 12, 12], [0, 1, 1]),
        ([14, 4, 12], [1, 0, 1]),
    ];
    let mut voxels = Vec::new();
    for (start, step) in tubes {
        // The fit saw a corrupted table, so its directions (in voxel axes)
        // are the corruption applied to the truth.
        let s = step.map(|v| v as f64);
        let bad = [0, 1, 2].map(|r| (0..3).map(|c| corrupt[r][c] * s[c]).sum::<f64>());
        let dir = world_dir(&affine, bad);
        for i in 0..6i64 {
            let c = [0, 1, 2].map(|a| (start[a] as i64 + i * step[a]) as usize);
            voxels.push((c, vec![(dir, 1.0)]));
        }
    }
    build_grid_dataset(affine, [22, 22, 22], voxels)
}

#[test]
fn btable_check_finds_the_correction_on_ras_and_lps_grids() {
    let scorings = [
        (CoherenceMode::Fixel, positive_options()),
        (CoherenceMode::Primary, FixelQcOptions::default()),
        (CoherenceMode::Chain, positive_options()),
    ];
    for affine in [odx_rs::Header::identity_affine(), LPS_2MM] {
        for label in ["012", "012fx", "102", "021fy", "120", "201fz"] {
            // Corrupt with the inverse (transpose) so `label` is the fix.
            let m = label_matrix(label);
            let inverse = [0, 1, 2].map(|r| [0, 1, 2].map(|c| m[c][r]));
            let ds = tube_phantom(affine, inverse);
            for (scoring, options) in &scorings {
                let check = check_btable(&ds, options, *scoring).unwrap();
                assert_eq!(check.scoring, *scoring);
                assert_eq!(check.candidates.len(), 24);
                assert_eq!(check.candidates[0].label, "012");
                assert_eq!(check.best, label, "affine {affine:?}, {scoring:?}");
                // Coherence scores 1.0 when every tube connects; chain scoring
                // gives the fibre-weighted mean tube length: three axis tubes
                // of 5 steps and three diagonal tubes of 5 diagonal steps.
                let voxel = if affine == LPS_2MM { 2.0 } else { 1.0 };
                let intact = match scoring {
                    CoherenceMode::Chain => voxel * (3.0 * 5.0 + 3.0 * 5.0 * 2f64.sqrt()) / 6.0,
                    _ => 1.0,
                };
                let best = check.best_coherence_index.unwrap();
                assert!(
                    (best - intact).abs() < 1e-6,
                    "{scoring:?}: {best} != {intact}"
                );
                assert_eq!(check.current_is_best, label == "012");
                if label != "012" {
                    assert!(check.current_coherence_index.unwrap() < 0.9 * intact);
                }
            }
        }
    }
}

#[test]
fn quantile_threshold_scores_the_same_share_whatever_the_scale() {
    // A chain with weights 1..=10. q = 0.3 drops the three lowest, and
    // rescaling every weight (as a different QA scale would) changes nothing.
    let x = [1.0, 0.0, 0.0];
    let chain = |scale: f32| {
        (0..10)
            .map(|i| ([i + 1, 3, 3], vec![(x, scale * (i + 1) as f32)]))
            .collect::<Vec<_>>()
    };
    let opts = FixelQcOptions {
        threshold: ThresholdMode::Quantile(0.3),
        ..Default::default()
    };
    for scale in [1.0, 0.01] {
        let ds = build_grid_dataset(odx_rs::Header::identity_affine(), [12, 7, 7], chain(scale));
        let fixel = compute_fixel_qc(&ds, &opts).unwrap().report;
        assert_eq!(fixel.evaluated_fixels, 7);
        assert!((fixel.threshold_value.unwrap() - 4.0 * scale).abs() < 1e-6);
        let primary = compute_primary_coherence(&ds, &opts).unwrap();
        // Empty grid voxels never enter the quantile.
        assert_eq!(primary.evaluated_voxels, 7);
        assert_eq!(primary.coherence_index, Some(1.0));
    }

    for bad in [-0.1, 1.0, f32::NAN] {
        let opts = FixelQcOptions {
            threshold: ThresholdMode::Quantile(bad),
            ..Default::default()
        };
        let ds = build_grid_dataset(odx_rs::Header::identity_affine(), [12, 7, 7], chain(1.0));
        assert!(compute_fixel_qc(&ds, &opts).is_err(), "q = {bad}");
    }
}

#[test]
fn threshold_elasticity_tracks_how_coherence_moves_with_the_cut() {
    // Strong voxels form a coherent chain along x; weak voxels are scattered
    // singletons. Raising the threshold drops singletons, so coherence rises
    // with the threshold and the elasticity is positive.
    let x = [1.0, 0.0, 0.0];
    let mut voxels: Vec<_> = (1..9).map(|i| ([i, 2, 2], vec![(x, 1.0)])).collect();
    for (k, c) in [[1, 6, 6], [4, 6, 6], [7, 6, 6], [1, 6, 9], [4, 6, 9]]
        .into_iter()
        .enumerate()
    {
        voxels.push((c, vec![(x, 0.3 + 0.05 * k as f32)]));
    }
    let ds = build_grid_dataset(odx_rs::Header::identity_affine(), [10, 10, 12], voxels);
    let opts = FixelQcOptions {
        threshold: ThresholdMode::Value(0.35),
        ..Default::default()
    };
    let e = coherence_threshold_elasticity(&ds, &opts, CoherenceMode::Primary)
        .unwrap()
        .unwrap();
    assert!(e > 0.0, "elasticity {e}");

    // Nothing to perturb without a positive threshold.
    let all = FixelQcOptions {
        threshold: ThresholdMode::All,
        ..Default::default()
    };
    assert_eq!(
        coherence_threshold_elasticity(&ds, &all, CoherenceMode::Fixel).unwrap(),
        None
    );
}

fn assert_close(a: Option<f64>, b: f64) {
    let a = a.expect("value");
    assert!((a - b).abs() < 1e-6, "{a} != {b}");
}

#[test]
fn a_straight_chain_is_one_chain_as_long_as_its_steps() {
    // Ten voxels along index x on a 2 mm LPS grid: nine 2 mm steps.
    let ds = build_grid_dataset(
        LPS_2MM,
        [12, 5, 5],
        index_chain(&LPS_2MM, [1, 2, 2], [1, 0, 0], 10),
    );
    let r = compute_fixel_chains(&ds, &positive_options()).unwrap();
    assert_eq!(r.evaluated_fixels, 10);
    assert_eq!(r.chains, 1);
    assert_eq!(r.loops, 0);
    assert_close(r.weighted_mean_length_mm, 18.0);
    assert_close(r.max_chain_length_mm, 18.0);
    assert_close(r.weight_in_chains_over_20mm, 0.0);
    // Oblique steps are measured through the affine too: 9 * 2*sqrt(2).
    let ds = build_grid_dataset(
        LPS_2MM,
        [12, 12, 5],
        index_chain(&LPS_2MM, [1, 1, 2], [1, 1, 0], 10),
    );
    let r = compute_fixel_chains(&ds, &positive_options()).unwrap();
    assert_eq!(r.chains, 1);
    assert_close(r.weighted_mean_length_mm, 18.0 * 2f64.sqrt());
}

#[test]
fn fibres_between_lattice_directions_still_form_long_chains() {
    // 22.5 degrees from x in the xy plane: the fixel metric connects none of
    // these, but every fixel has a continuation within the step cone.
    let a = 22.5f32.to_radians();
    let dir = [a.cos(), a.sin(), 0.0];
    let voxels = (1..13)
        .flat_map(|x| (1..13).map(move |y| ([x, y, 2], vec![(dir, 1.0)])))
        .collect();
    let ds = build_grid_dataset(odx_rs::Header::identity_affine(), [14, 14, 5], voxels);
    let r = compute_fixel_chains(&ds, &positive_options()).unwrap();
    assert_eq!(r.evaluated_fixels, 144);
    assert!(r.weighted_mean_length_mm.unwrap() >= 6.0, "{r:?}");
    assert!(r.chains < 144 / 3, "{r:?}");
}

#[test]
fn one_perpendicular_voxel_splits_a_chain() {
    let x = [1.0, 0.0, 0.0];
    let mut voxels: Vec<_> = (1..12).map(|i| ([i, 3, 3], vec![(x, 1.0)])).collect();
    voxels[5].1 = vec![([0.0, 1.0, 0.0], 1.0)]; // voxel x=6 points along y
    let ds = build_grid_dataset(odx_rs::Header::identity_affine(), [13, 7, 7], voxels);
    let r = compute_fixel_chains(&ds, &positive_options()).unwrap();
    // x = 1..=5 (4 mm), x = 6 alone, x = 7..=11 (4 mm).
    assert_eq!(r.chains, 3);
    assert_close(r.max_chain_length_mm, 4.0);
    // Fibre-weighted: ten fixels in 4 mm chains, one alone.
    assert_close(r.weighted_mean_length_mm, 40.0 / 11.0);
}

#[test]
fn chains_respect_the_threshold_and_link_only_mutually() {
    // A strong chain along x; weak fixels beside it are thresholded out and
    // must not join or split it.
    let x = [1.0, 0.0, 0.0];
    let mut voxels: Vec<_> = (1..9).map(|i| ([i, 3, 3], vec![(x, 1.0)])).collect();
    voxels.push(([4, 4, 3], vec![(x, 0.01)]));
    let ds = build_grid_dataset(odx_rs::Header::identity_affine(), [10, 7, 7], voxels);
    let opts = FixelQcOptions {
        threshold: ThresholdMode::Quantile(0.2),
        ..Default::default()
    };
    let r = compute_fixel_chains(&ds, &opts).unwrap();
    assert_eq!(r.evaluated_fixels, 8);
    assert_eq!(r.chains, 1);
    assert_close(r.weighted_mean_length_mm, 7.0);
}
