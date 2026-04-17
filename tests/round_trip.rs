use odx_rs::{CanonicalDenseRepresentation, DType, Header, OdxBuilder, OdxDataset, OdxWritePolicy};
use std::collections::HashMap;

fn make_test_odx() -> OdxDataset {
    let dims = [4u64, 3, 2];
    let total = (dims[0] * dims[1] * dims[2]) as usize;

    // C-order mask: last axis varies fastest
    let mut mask = vec![0u8; total];
    // Mark voxels (0,0,0), (1,0,0), (2,1,0) as brain
    // C-order flat index for (i,j,k) with dims [4,3,2] = i*3*2 + j*2 + k
    mask[0 * 3 * 2 + 0 * 2 + 0] = 1; // (0,0,0)
    mask[1 * 3 * 2 + 0 * 2 + 0] = 1; // (1,0,0)
    mask[2 * 3 * 2 + 1 * 2 + 0] = 1; // (2,1,0)

    let affine = Header::identity_affine();
    let mut stream = OdxBuilder::new(affine, dims, mask);

    // Voxel 0: 2 peaks
    stream.push_voxel_peaks(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    // Voxel 1: 1 peak
    stream.push_voxel_peaks(&[[0.0, 0.0, 1.0]]);
    // Voxel 2: 3 peaks
    stream.push_voxel_peaks(&[[0.577, 0.577, 0.577], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);

    // Sphere: a tetrahedron (4 vertices, 4 faces)
    let sphere_verts: Vec<[f32; 3]> = vec![
        [0.0, 0.0, 1.0],
        [0.943, 0.0, -0.333],
        [-0.471, 0.816, -0.333],
        [-0.471, -0.816, -0.333],
    ];
    let sphere_faces: Vec<[u32; 3]> = vec![[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]];
    stream.set_sphere(sphere_verts, sphere_faces);

    // ODF amplitudes: 3 voxels × 4 sphere directions
    let odf_data: Vec<f32> = vec![
        0.1, 0.2, 0.3, 0.4, // voxel 0
        0.5, 0.6, 0.7, 0.8, // voxel 1
        0.9, 1.0, 1.1, 1.2, // voxel 2
    ];
    stream.set_odf_data(
        "amplitudes",
        bytemuck::cast_slice(&odf_data).to_vec(),
        4,
        DType::Float32,
    );

    // DPV: GFA
    let gfa: Vec<f32> = vec![0.3, 0.5, 0.7];
    stream.set_dpv_data(
        "gfa",
        bytemuck::cast_slice(&gfa).to_vec(),
        1,
        DType::Float32,
    );

    // DPF: amplitude (6 total peaks)
    let amplitude: Vec<f32> = vec![0.8, 0.6, 0.9, 0.4, 0.7, 0.5];
    stream.set_dpf_data(
        "amplitude",
        bytemuck::cast_slice(&amplitude).to_vec(),
        1,
        DType::Float32,
    );

    stream.finalize().unwrap()
}

#[test]
fn round_trip_directory() {
    let odx = make_test_odx();
    let dir = tempfile::TempDir::new().unwrap();
    let out_dir = dir.path().join("test.odxd");

    odx.save_directory(&out_dir).unwrap();

    let loaded = OdxDataset::load(&out_dir).unwrap();

    assert_eq!(loaded.nb_voxels(), 3);
    assert_eq!(loaded.nb_peaks(), 6);

    // Check mask
    assert_eq!(loaded.mask().len(), 4 * 3 * 2);
    assert_eq!(loaded.mask()[0], 1);
    assert_eq!(loaded.mask()[1], 0);

    // Check offsets
    assert_eq!(loaded.offsets(), &[0, 2, 3, 6]);

    // Check directions
    assert_eq!(loaded.directions().len(), 6);
    assert_eq!(loaded.directions()[0], [1.0, 0.0, 0.0]);
    assert_eq!(loaded.directions()[2], [0.0, 0.0, 1.0]);

    // Check voxel peak access
    assert_eq!(loaded.voxel_directions(0).len(), 2);
    assert_eq!(loaded.voxel_directions(1).len(), 1);
    assert_eq!(loaded.voxel_directions(2).len(), 3);

    // Check sphere
    let verts = loaded.sphere_vertices().unwrap();
    assert_eq!(verts.len(), 4);
    let faces = loaded.sphere_faces().unwrap();
    assert_eq!(faces.len(), 4);

    // Check ODF
    let odf = loaded.odf::<f32>("amplitudes").unwrap();
    assert_eq!(odf.shape(), (3, 4));
    assert_eq!(odf.row(0), &[0.1, 0.2, 0.3, 0.4]);
    assert_eq!(odf.row(2), &[0.9, 1.0, 1.1, 1.2]);

    // Check DPV
    let gfa = loaded.scalar_dpv_f32("gfa").unwrap();
    assert_eq!(gfa, vec![0.3, 0.5, 0.7]);

    // Check DPF
    let amp = loaded.scalar_dpf_f32("amplitude").unwrap();
    assert_eq!(amp, vec![0.8, 0.6, 0.9, 0.4, 0.7, 0.5]);

    // Check header
    assert_eq!(loaded.header().nb_sphere_vertices, Some(4));
    assert_eq!(loaded.header().nb_sphere_faces, Some(4));
}

#[test]
fn round_trip_zip() {
    let odx = make_test_odx();
    let dir = tempfile::TempDir::new().unwrap();
    let zip_path = dir.path().join("test.odx");

    odx.save_archive(&zip_path).unwrap();

    let loaded = OdxDataset::load(&zip_path).unwrap();

    assert_eq!(loaded.nb_voxels(), 3);
    assert_eq!(loaded.nb_peaks(), 6);
    assert_eq!(loaded.offsets(), &[0, 2, 3, 6]);
    assert_eq!(loaded.directions()[0], [1.0, 0.0, 0.0]);

    let odf = loaded.odf::<f32>("amplitudes").unwrap();
    assert_eq!(odf.shape(), (3, 4));
}

#[test]
fn voxel_peak_iterator() {
    let odx = make_test_odx();
    let peaks: Vec<&[[f32; 3]]> = odx.voxel_peaks().collect();
    assert_eq!(peaks.len(), 3);
    assert_eq!(peaks[0].len(), 2);
    assert_eq!(peaks[1].len(), 1);
    assert_eq!(peaks[2].len(), 3);
}

#[test]
fn header_round_trip() {
    let header = Header {
        voxel_to_rasmm: [
            [1.25, 0.0, 0.0, -90.0],
            [0.0, 1.25, 0.0, -126.0],
            [0.0, 0.0, 1.25, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dimensions: [145, 174, 145],
        nb_voxels: 72534,
        nb_peaks: 198421,
        nb_sphere_vertices: Some(642),
        nb_sphere_faces: Some(1280),
        sh_order: Some(8),
        sh_basis: Some("descoteaux07".into()),
        canonical_dense_representation: None,
        sphere_id: None,
        odf_sample_domain: None,
        array_quantization: HashMap::new(),
        extra: HashMap::new(),
    };

    let json = header.to_json().unwrap();
    let parsed: Header = serde_json::from_str(&json).unwrap();
    assert_eq!(header, parsed);
}

#[test]
fn quantized_directory_round_trip() {
    let odx = make_test_odx();
    let dir = tempfile::TempDir::new().unwrap();
    let out_dir = dir.path().join("quantized.odxd");

    odx.save_directory_with_policy(
        &out_dir,
        OdxWritePolicy {
            quantize_dense: true,
            quantize_min_len: 1,
        },
    )
    .unwrap();

    let loaded = OdxDataset::load(&out_dir).unwrap();
    assert!(!loaded.header().array_quantization.is_empty());

    let orig_odf = odx.odf::<f32>("amplitudes").unwrap();
    let roundtrip_odf = loaded.odf::<f32>("amplitudes").unwrap();
    assert_eq!(orig_odf.shape(), roundtrip_odf.shape());
    let max_diff = orig_odf
        .as_flat_slice()
        .iter()
        .zip(roundtrip_odf.as_flat_slice())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 0.01, "quantized ODF max diff {max_diff}");
}

#[test]
fn sphere_id_allows_odf_without_explicit_sphere() {
    let dims = [1u64, 1, 1];
    let mask = vec![1u8];
    let mut builder = OdxBuilder::new(Header::identity_affine(), dims, mask);
    builder.push_voxel_peaks(&[[1.0, 0.0, 0.0]]);
    builder.set_sphere_id("odf8");
    builder.set_canonical_dense_representation(CanonicalDenseRepresentation::Odf);
    builder.set_odf_data(
        "amplitudes",
        bytemuck::cast_slice(&[0.25f32, 0.5, 0.75]).to_vec(),
        3,
        DType::Float32,
    );
    let odx = builder.finalize().unwrap();

    let dir = tempfile::TempDir::new().unwrap();
    let out_dir = dir.path().join("sphere_id.odxd");
    odx.save_directory(&out_dir).unwrap();

    let loaded = OdxDataset::load(&out_dir).unwrap();
    assert!(loaded.sphere_vertices().is_none());
    assert_eq!(loaded.header().sphere_id.as_deref(), Some("odf8"));
    assert_eq!(
        loaded.header().canonical_dense_representation,
        Some(CanonicalDenseRepresentation::Odf)
    );
    let odf = loaded.odf::<f32>("amplitudes").unwrap();
    assert_eq!(odf.shape(), (1, 3));
}
