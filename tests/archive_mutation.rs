use std::collections::HashMap;
use std::io::Read;

use odx_rs::io::{directory, zip as odx_zip};
use odx_rs::{DType, DataArray, Header, OdxBuilder, OdxDataset, OdxWritePolicy};

fn base_odx() -> OdxDataset {
    let dims = [1u64, 1, 2];
    let mask = vec![1u8, 1u8];
    let mut builder = OdxBuilder::new(Header::identity_affine(), dims, mask);
    builder.push_voxel_peaks(&[[1.0, 0.0, 0.0]]);
    builder.push_voxel_peaks(&[[0.0, 1.0, 0.0]]);
    builder.set_dpv_data(
        "gfa",
        bytemuck::cast_slice(&[0.2f32, 0.8f32]).to_vec(),
        1,
        DType::Float32,
    );
    builder.set_dpf_data(
        "amplitude",
        bytemuck::cast_slice(&[0.5f32, 0.9f32]).to_vec(),
        1,
        DType::Float32,
    );
    builder.finalize().unwrap()
}

fn read_archive_header(path: &std::path::Path) -> serde_json::Value {
    let file = std::fs::File::open(path).unwrap();
    let mut archive = ::zip::ZipArchive::new(file).unwrap();
    let mut entry = archive.by_name("header.json").unwrap();
    let mut text = String::new();
    entry.read_to_string(&mut text).unwrap();
    serde_json::from_str(&text).unwrap()
}

#[test]
fn zip_archive_mutation_supports_compact_arrays_and_header_updates() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("sample.odx");
    base_odx()
        .save_archive_with_policy(
            &path,
            OdxWritePolicy {
                quantize_dense: true,
                quantize_min_len: 1,
            },
        )
        .unwrap();

    let initial_header = read_archive_header(&path);
    assert!(initial_header["ARRAY_QUANTIZATION"]["dpv/gfa"].is_object());

    let dpf = HashMap::from([(
        "qc".to_string(),
        DataArray::owned_bytes(vec![1u8, 2u8], 1, DType::UInt8),
    )]);
    let groups = HashMap::from([(
        "bundle".to_string(),
        DataArray::owned_bytes(bytemuck::cast_slice(&[0u32]).to_vec(), 1, DType::UInt32),
    )]);
    let dpg = HashMap::from([(
        "bundle".to_string(),
        HashMap::from([(
            "quality".to_string(),
            DataArray::owned_bytes(vec![7u8, 8u8], 2, DType::UInt8),
        )]),
    )]);

    odx_zip::append_dpf_to_zip(&path, &dpf, ::zip::CompressionMethod::Stored, false).unwrap();
    odx_zip::append_groups_to_zip(&path, &groups, ::zip::CompressionMethod::Stored, false).unwrap();
    odx_zip::append_dpg_to_zip(&path, &dpg, ::zip::CompressionMethod::Stored, false).unwrap();

    let replacement_dpv = HashMap::from([(
        "gfa".to_string(),
        DataArray::owned_bytes(
            bytemuck::cast_slice(&[0.9f32, 1.1f32]).to_vec(),
            1,
            DType::Float32,
        ),
    )]);
    odx_zip::append_dpv_to_zip(
        &path,
        &replacement_dpv,
        ::zip::CompressionMethod::Stored,
        true,
    )
    .unwrap();
    odx_zip::delete_groups_from_zip(&path, &["bundle"]).unwrap();

    let loaded = OdxDataset::load(&path).unwrap();
    assert_eq!(loaded.scalar_dpf_f32("qc").unwrap(), vec![1.0, 2.0]);
    assert_eq!(loaded.scalar_dpv_f32("gfa").unwrap(), vec![0.9, 1.1]);
    assert_eq!(loaded.group_names(), Vec::<&str>::new());
    assert!(loaded.dpg::<f32>("bundle", "quality").is_err());

    let header = read_archive_header(&path);
    assert!(header["ARRAY_QUANTIZATION"]["dpv/gfa"].is_null());
}

#[test]
fn directory_mutation_matches_zip_semantics() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("sample.odxd");
    base_odx()
        .save_directory_with_policy(
            &path,
            OdxWritePolicy {
                quantize_dense: true,
                quantize_min_len: 1,
            },
        )
        .unwrap();

    let dpf = HashMap::from([(
        "qc".to_string(),
        DataArray::owned_bytes(vec![3u8, 4u8], 1, DType::UInt8),
    )]);
    let groups = HashMap::from([(
        "bundle".to_string(),
        DataArray::owned_bytes(bytemuck::cast_slice(&[0u32]).to_vec(), 1, DType::UInt32),
    )]);
    let dpg = HashMap::from([(
        "bundle".to_string(),
        HashMap::from([(
            "quality".to_string(),
            DataArray::owned_bytes(vec![5u8], 1, DType::UInt8),
        )]),
    )]);

    directory::append_dpf_to_directory(&path, &dpf, false).unwrap();
    directory::append_groups_to_directory(&path, &groups, false).unwrap();
    directory::append_dpg_to_directory(&path, &dpg, false).unwrap();
    directory::delete_groups_from_directory(&path, &["bundle"]).unwrap();
    directory::delete_dpv_from_directory(&path, &["gfa"]).unwrap();

    let loaded = OdxDataset::load(&path).unwrap();
    assert_eq!(loaded.scalar_dpf_f32("qc").unwrap(), vec![3.0, 4.0]);
    assert_eq!(loaded.group_names(), Vec::<&str>::new());
    assert!(loaded.scalar_dpv_f32("gfa").is_err());

    let header: odx_rs::Header =
        serde_json::from_str(&std::fs::read_to_string(path.join("header.json")).unwrap()).unwrap();
    assert!(!header.array_quantization.contains_key("dpv/gfa"));
}

#[test]
fn zip_append_rejects_invalid_row_count() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("invalid.odx");
    base_odx().save_archive(&path).unwrap();

    let invalid = HashMap::from([(
        "bad".to_string(),
        DataArray::owned_bytes(vec![1u8], 1, DType::UInt8),
    )]);
    let err = odx_zip::append_dpf_to_zip(&path, &invalid, ::zip::CompressionMethod::Stored, false)
        .unwrap_err();

    assert!(err.to_string().contains("expected 2"));
}
