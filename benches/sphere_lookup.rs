//! Sphere lookup benchmark: confirms the KD-tree fast path beats brute
//! force at HCP-scale and validates the threshold in `sphere_lookup.rs`.
//!
//! Run with: `cargo bench --bench sphere_lookup`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use odx_rs::sphere_lookup::{nearest_vertex, nearest_vertex_indices};

fn dsi_studio_hemisphere() -> Vec<[f32; 3]> {
    odx_rs::formats::dsistudio_odf8::hemisphere_vertices_ras().to_vec()
}

fn random_unit_directions(n: usize) -> Vec<[f32; 3]> {
    (0..n)
        .map(|i| {
            let f = i as f32;
            // Pseudo-random spherical coords from a few sin/cos terms.
            let phi = (f * 0.0173).sin() * std::f32::consts::PI * 2.0;
            let cos_theta = ((f * 0.0241).cos()).clamp(-1.0, 1.0);
            let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
            [phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta]
        })
        .collect()
}

fn bench_sphere_lookup(c: &mut Criterion) {
    let sphere = dsi_studio_hemisphere(); // 321 vertices

    // Several scales: small (sub-threshold) → medium → HCP-ish.
    let scales = [1_000, 10_000, 100_000, 500_000];
    let mut group = c.benchmark_group("sphere_lookup");

    for &p in &scales {
        let dirs = random_unit_directions(p);

        // Brute force per-direction.
        group.bench_with_input(BenchmarkId::new("brute_force", p), &p, |b, _| {
            b.iter(|| {
                let mut out = Vec::with_capacity(dirs.len());
                for &d in &dirs {
                    out.push(nearest_vertex(d, &sphere, true).0 as i32);
                }
                black_box(out)
            });
        });

        // Dispatched (auto-picks brute or KD-tree based on size).
        group.bench_with_input(BenchmarkId::new("dispatched", p), &p, |b, _| {
            b.iter(|| {
                let out = nearest_vertex_indices(&dirs, &sphere, true);
                black_box(out)
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_sphere_lookup);
criterion_main!(benches);
