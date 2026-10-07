//! `LoadOptions`: arrays a caller does not need are never read.

use odx_rs::formats::mat4;
use odx_rs::interop::{
    save_dsistudio_from_odx, DenseOdfMode, DsistudioFormat, MrtrixToDsistudioOptions, PeakSource,
};
use odx_rs::{dsistudio, DType, Header, LoadOptions, OdxBuilder, OdxDataset};

/// 3x3x3 grid, one x-pointing fixel per voxel, SH (lmax 2), a dense ODF on a
/// tetrahedron, and amplitude/gfa scalars.
fn dense_dataset() -> OdxDataset {
    let n = 27;
    let mut builder = OdxBuilder::new(Header::identity_affine(), [3, 3, 3], vec![1u8; n]);
    for _ in 0..n {
        builder.push_voxel_peaks(&[[1.0, 0.0, 0.0]]);
    }
    let amplitude: Vec<f32> = (0..n).map(|i| 0.1 + i as f32 / n as f32).collect();
    builder.set_dpf_data(
        "amplitude",
        bytemuck::cast_slice(&amplitude).to_vec(),
        1,
        DType::Float32,
    );
    builder.set_dpv_data(
        "gfa",
        bytemuck::cast_slice(&vec![0.5f32; n]).to_vec(),
        1,
        DType::Float32,
    );
    let mut sh = vec![0.0f32; n * 6];
    for row in sh.chunks_mut(6) {
        row[0] = 1.0;
        row[3] = -0.3;
        row[5] = 0.4;
    }
    builder.set_sh_data(
        "coefficients",
        bytemuck::cast_slice(&sh).to_vec(),
        6,
        DType::Float32,
    );
    builder.set_sh_info(2, "tournier07".into());
    builder.set_sh_legacy(false);
    builder.set_sphere(
        vec![
            [0.0, 0.0, 1.0],
            [0.943, 0.0, -0.333],
            [-0.471, 0.816, -0.333],
            [-0.471, -0.816, -0.333],
        ],
        vec![[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]],
    );
    builder.set_odf_data(
        "amplitudes",
        bytemuck::cast_slice(&vec![0.25f32; n * 4]).to_vec(),
        4,
        DType::Float32,
    );
    builder.finalize().unwrap()
}

fn assert_same_fixels(a: &OdxDataset, b: &OdxDataset) {
    assert_eq!(a.nb_peaks(), b.nb_peaks());
    assert_eq!(a.offsets(), b.offsets());
    assert_eq!(a.directions(), b.directions());
    assert_eq!(
        a.scalar_dpf_f32("amplitude").unwrap(),
        b.scalar_dpf_f32("amplitude").unwrap()
    );
}

#[test]
fn odx_directory_and_archive_skip_dense_arrays() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("dense.odxd");
    let archive = tmp.path().join("dense.odx");
    let ds = dense_dataset();
    ds.save_directory(&dir).unwrap();
    ds.save_archive(&archive).unwrap();

    for path in [&dir, &archive] {
        let full = OdxDataset::load(path).unwrap();
        assert!(full.get_odf("amplitudes").is_some() && full.get_sh("coefficients").is_some());

        let fixels = OdxDataset::load_with(path, &LoadOptions::fixels_only()).unwrap();
        assert!(fixels.odf_names().is_empty(), "{}", path.display());
        assert!(fixels.sh_names().is_empty());
        assert_eq!(fixels.header().sh_order, None);
        assert_eq!(fixels.header().sh_basis, None);
        assert_eq!(fixels.header().canonical_dense_representation, None);
        assert!(fixels.get_dpv("gfa").is_some());
        assert_same_fixels(&full, &fixels);

        let no_odf = LoadOptions {
            skip_odf: true,
            skip_sh: false,
        };
        let sh_only = OdxDataset::load_with(path, &no_odf).unwrap();
        assert!(sh_only.odf_names().is_empty());
        assert_eq!(
            sh_only.sh::<f32>("coefficients").unwrap().row(0),
            full.sh::<f32>("coefficients").unwrap().row(0)
        );
        assert_eq!(sh_only.header().sh_order, Some(2));
    }
}

#[test]
fn dsistudio_skip_odf_keeps_fixels_and_drops_dense_odfs() {
    let tmp = tempfile::tempdir().unwrap();
    let fib = tmp.path().join("dense.fib.gz");
    save_dsistudio_from_odx(
        &dense_dataset(),
        &fib,
        &MrtrixToDsistudioOptions {
            output_format: DsistudioFormat::FibGz,
            dense_odf_mode: DenseOdfMode::FromSh,
            peak_source: PeakSource::Fixels,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(mat4::read_mat4_gz(&fib).unwrap().has("odf0"));

    let full = dsistudio::load_fibgz(&fib, None).unwrap();
    assert!(!full.odf_names().is_empty());
    let skip = LoadOptions {
        skip_odf: true,
        skip_sh: false,
    };
    let fixels = dsistudio::load_dsistudio_with(&fib, None, &skip).unwrap();
    assert!(fixels.odf_names().is_empty());
    assert_eq!(fixels.header().odf_sample_domain, None);
    // The sphere is not an ODF chunk and survives.
    assert_eq!(
        fixels.sphere_vertices().map(|v| v.len()),
        full.sphere_vertices().map(|v| v.len())
    );
    assert_same_fixels(&full, &fixels);
}

#[test]
fn mat4_stream_filters_records_and_rejects_truncation() {
    let records = vec![
        mat4::float_record("dimension", vec![3.0, 3.0, 3.0], 1, 3),
        mat4::float_record("odf0", vec![1.0; 12], 4, 3),
        mat4::float_record("odf_vertices", vec![0.0; 6], 3, 2),
        mat4::float_record("fa0", vec![0.5; 4], 1, 4),
    ];
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("records.mat.gz");
    mat4::write_mat4_gz(&path, &records).unwrap();

    let kept = mat4::read_mat4_gz_filtered(&path, |n| n != "odf0").unwrap();
    assert!(!kept.has("odf0"));
    assert!(kept.has("odf_vertices"));
    assert_eq!(kept.get("fa0").unwrap().as_f32_vec(), vec![0.5; 4]);
    assert_eq!(kept.get("dimension").unwrap().as_f32_vec(), vec![3.0; 3]);

    let mut bytes = Vec::new();
    mat4::write_mat4(&mut bytes, &records).unwrap();
    // A short tail after the last record is ignored, as before.
    let mut padded = bytes.clone();
    padded.extend_from_slice(&[0u8; 7]);
    assert!(mat4::read_mat4(&padded).unwrap().has("fa0"));
    // A record cut short is an error.
    assert!(mat4::read_mat4(&bytes[..bytes.len() - 3]).is_err());
}

#[test]
fn reference_affine_reads_only_the_header() {
    use std::io::Write;
    let affine = [
        [-1.5, 0.0, 0.0, 90.0],
        [0.0, 1.5, 0.0, -126.0],
        [0.0, 0.0, 2.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let tmp = tempfile::tempdir().unwrap();
    let plain = tmp.path().join("ref.nii");
    let dims = [40usize, 40, 40];
    let ijk: Vec<[u32; 3]> = vec![[0, 0, 0]];
    odx_rs::write_voxel_scalar_nifti_f32(&plain, &[1.0], &ijk, dims, affine).unwrap();

    // Gzip the file and corrupt the end of the compressed stream: reading
    // the voxel data now fails, so only a header-only read can succeed.
    let raw = std::fs::read(&plain).unwrap();
    let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    gz.write_all(&raw).unwrap();
    let mut compressed = gz.finish().unwrap();
    let n = compressed.len();
    for b in &mut compressed[n - 16..] {
        *b ^= 0xff;
    }
    let broken = tmp.path().join("ref.nii.gz");
    std::fs::write(&broken, &compressed).unwrap();

    let got = odx_rs::read_reference_affine(&broken).unwrap();
    for r in 0..4 {
        for c in 0..4 {
            assert!((got[r][c] - affine[r][c]).abs() < 1e-5, "{got:?}");
        }
    }
}
