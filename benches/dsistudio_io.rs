use std::path::Path;

use criterion::{criterion_group, criterion_main, Criterion};
use odx_rs::{dsistudio, OdxBuilder, OdxWritePolicy};

const FIB_PATH: &str =
    "../test_data/sub-NDARAE199TDD_ses-1_acq-64dirVARIANTVar1e_space-ACPC_model-ss3t_dwimap.fib.gz";

fn benchmark_dsistudio_load(c: &mut Criterion) {
    let path = Path::new(FIB_PATH);
    if !path.exists() {
        return;
    }
    c.bench_function("dsistudio fib.gz load", |b| {
        b.iter(|| dsistudio::load_fibgz(path, None).unwrap())
    });
}

fn benchmark_quantized_directory_round_trip(c: &mut Criterion) {
    let mut builder = OdxBuilder::new(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [4, 4, 4],
        vec![1; 64],
    );
    for _ in 0..64 {
        builder.push_voxel_peaks(&[[1.0, 0.0, 0.0]]);
    }
    builder.set_sphere(
        vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        vec![[0, 1, 2]],
    );
    let mut dense = Vec::with_capacity(64 * 45);
    for i in 0..(64 * 45) {
        dense.push((i as f32).sin());
    }
    builder.set_sh_info(8, "descoteaux07".into());
    builder.set_sh_data(
        "coefficients",
        bytemuck::cast_slice(&dense).to_vec(),
        45,
        odx_rs::DType::Float32,
    );
    let dataset = builder.finalize().unwrap();

    c.bench_function("odx quantized directory save", |b| {
        b.iter(|| {
            let tmp = tempfile::TempDir::new().unwrap();
            dataset
                .save_directory_with_policy(
                    tmp.path(),
                    OdxWritePolicy {
                        quantize_dense: true,
                        quantize_min_len: 1,
                    },
                )
                .unwrap();
        })
    });
}

criterion_group!(
    benches,
    benchmark_dsistudio_load,
    benchmark_quantized_directory_round_trip
);
criterion_main!(benches);
