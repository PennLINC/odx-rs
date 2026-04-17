use std::path::Path;

use odx_rs::mif;

const MIF_PATH: &str =
    "../test_data/sub-NDARAE199TDD_ses-1_acq-64dirVARIANTVar1e_space-ACPC_model-ss3t_param-fod_label-WM_dwimap.mif.gz";

#[test]
fn load_mif_gz_header() {
    let path = Path::new(MIF_PATH);
    if !path.exists() {
        eprintln!("skipping: test data not found at {MIF_PATH}");
        return;
    }

    let img = mif::read_mif(path).unwrap();

    println!("MIF dimensions: {:?}", img.header.dimensions);
    println!("MIF voxel sizes: {:?}", img.header.voxel_sizes);
    println!("MIF datatype: {}", img.header.datatype);
    println!("MIF layout: {:?}", img.header.layout);
    println!("MIF ndim: {}", img.ndim());
    println!("MIF nvoxels: {}", img.nvoxels());
    println!("MIF data bytes: {}", img.data.len());
    println!("MIF element size: {}", img.element_size());
    println!("MIF expected bytes: {}", img.nvoxels() * img.element_size());

    if let Some(ref t) = img.header.transform {
        println!("MIF transform row 0: {:?}", t[0]);
        println!("MIF transform row 1: {:?}", t[1]);
        println!("MIF transform row 2: {:?}", t[2]);
    }

    // Basic sanity
    assert!(img.ndim() >= 3);
    assert!(img.header.dimensions[0] > 0);
    assert!(!img.header.datatype.is_empty());

    // Data should be non-empty
    assert!(!img.data.is_empty());

    // For SH data, 4th dimension is the number of SH coefficients
    if img.ndim() == 4 {
        let n_sh = img.header.dimensions[3];
        println!("SH coefficients per voxel: {n_sh}");

        // Common SH orders: 4→15, 6→28, 8→45
        let expected_orders = [(6, 15), (8, 28), (10, 45), (12, 66), (14, 91)];
        for &(order, ncoeffs) in &expected_orders {
            if n_sh == ncoeffs {
                println!("Detected SH order: {order}");
            }
        }
    }

    // Check strides
    let strides = img.compute_strides();
    println!("Computed strides: {:?}", strides);

    // Get as f32
    let vals = img.as_f32_vec();
    println!("Total float values: {}", vals.len());
    assert_eq!(vals.len(), img.nvoxels());

    // Check affine
    let aff = img.affine_4x4();
    println!(
        "Affine:\n  {:?}\n  {:?}\n  {:?}\n  {:?}",
        aff[0], aff[1], aff[2], aff[3]
    );
    assert_eq!(aff[0], [2.0, 0.0, 0.0, -79.0]);
    assert_eq!(aff[1], [0.0, 2.0, 0.0, -114.0]);
    assert_eq!(aff[2], [0.0, 0.0, 2.0, -79.0]);
}
