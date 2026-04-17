# Fixel QC

This document describes how `odx-rs` computes fixel coherence QC in theory and
how to use it in practice from the library and the CLI.

## Theory

The QC pass operates directly on the sparse ODX representation:

- fixel directions from `directions`
- voxel-to-fixel grouping from `offsets`
- optional scalar fixel metrics from `dpf/`

It does not require dense ODFs or SH coefficients.

### Primary metric

QC needs one scalar nonnegative DPF as the primary weighting and thresholding
metric.

- If you pass `primary_metric`, that field is used.
- Otherwise `odx-rs` tries `amplitude`, then `afd`, then `qa`.
- `qc_class` is reserved output and is never eligible as the primary metric.

### Thresholding

Only fixels that pass the primary-metric threshold are evaluated.

Supported threshold modes:

- `Otsu`: choose a histogram split on the primary metric
- `Positive`: include values `> 0`
- `All`: include every fixel
- `Value(x)`: include values `> x`

Fixels below threshold are excluded from connectivity and become class `0`.

### Connectivity rule

For each evaluated fixel, `odx-rs` scans the 13 undirected voxel-neighbor
offsets:

- face neighbors
- edge neighbors
- corner neighbors

This can be confusing at first: "13-neighbor" here does **not** mean the metric
only sees half of the local neighborhood.

Instead, it is an efficient representation of the full immediate voxel
adjacency:

- a 3x3x3 neighborhood has 26 non-self neighbors
- each voxel pair can be written twice as a directed offset:
  - voxel A -> voxel B
  - voxel B -> voxel A
- `odx-rs` stores only the 13 unique undirected offsets and then evaluates each
  voxel pair in both directions

So the effective spatial neighborhood is still the standard full 26-neighbor
voxel neighborhood, just without duplicate work.

A fixel is marked connected if both conditions hold for at least one neighbor
voxel:

1. The source fixel direction aligns with the inter-voxel trajectory.
2. At least one fixel in the neighbor voxel aligns with the source fixel
   direction.

Both angular tests use the same threshold:

```text
abs(dot(a, b)) >= cos(angle_degrees)
```

The default angle is `15` degrees. This matches DSI Studio's default.

In implementation terms, for a source fixel with direction `source_dir` and a
neighbor voxel offset `offset_unit`, `odx-rs` checks:

```text
trajectory gate:
abs(dot(source_dir, offset_unit)) >= cos(angle_degrees)

neighbor-direction gate:
abs(dot(source_dir, neighbor_dir)) >= cos(angle_degrees)
```

If both pass for at least one fixel in that neighbor voxel, the source fixel is
classified as connected. Otherwise, if it passed thresholding but found no such
neighbor, it remains disconnected.

What this means geometrically:

- the source fixel must point roughly toward the neighboring voxel
- and a fixel in that neighbor voxel must point roughly the same way

This makes the QC measure purely local and spatial:

- it does use adjacent voxels
- it does not use same-voxel fixel matching
- it does not build connected components or multi-step paths
- it does not run tractography

This means the method favors coherent fixel chains that both point along the
neighbor trajectory and remain directionally consistent across voxels.

### Reported measures

The summary report contains:

- `total_fixels`
- `evaluated_fixels`
- `excluded_fixels`
- `connected_fixels`
- `disconnected_fixels`
- `connected_to_disconnected_ratio`
- `coherence_index`
- `incoherence_index`
- per-scalar-DPF connected/disconnected mean and median

`coherence_index` and `incoherence_index` are weighted by the primary metric
over evaluated fixels only:

```text
coherence_index = connected_weight / (connected_weight + disconnected_weight)
incoherence_index = disconnected_weight / (connected_weight + disconnected_weight)
```

Scalar DPF partition summaries are computed for every scalar DPF except
`qc_class`. Vector DPFs are skipped and listed in `skipped_dpf`.

## Practice

### Library API

The main entry point is:

```rust
use odx_rs::{compute_fixel_qc, FixelQcOptions, ThresholdMode};

let computation = compute_fixel_qc(
    &odx,
    &FixelQcOptions {
        primary_metric: Some("afd".into()),
        threshold: ThresholdMode::Otsu,
        angle_degrees: 15.0,
    },
)?;

println!("{:#?}", computation.report);
```

The result contains:

- `computation.report`: aggregate QC report
- `computation.classes`: one `FixelQcClass` per fixel

The in-memory class enum is:

- `FixelQcClass::ThresholdedOut`
- `FixelQcClass::Disconnected`
- `FixelQcClass::Connected`

### Writing `qc_class`

You can write the class map back into an existing ODX dataset:

```rust
use odx_rs::write_qc_class_dpf;

write_qc_class_dpf(path, &computation.classes, true)?;
```

This appends or replaces:

```text
dpf/qc_class.uint8
```

with fixed on-disk encoding:

- `0 = thresholded out`
- `1 = disconnected`
- `2 = connected`

The name `qc_class` is reserved for this purpose.

### CLI

Compute QC and print a text report:

```bash
odx qc input.odx
```

Choose the primary metric and include all fixels:

```bash
odx qc input.odx --primary-dpf afd --threshold all
```

Override the angle and emit JSON:

```bash
odx qc input.odx --angle-deg 20 --json
```

Write the per-fixel class map back into the input ODX dataset:

```bash
odx qc input.odx --write-qc-class
```

Replace an existing `qc_class` field:

```bash
odx qc input.odx --write-qc-class --overwrite-qc-class
```

`--write-qc-class` only works for ODX directory or `.odx` archive inputs.

### Reading `qc_class` later

On disk, `qc_class` is stored as `uint8`. The current ODX loader may normalize
scalar DPFs to `f32` in memory, so downstream consumers may read it back as
`0.0`, `1.0`, and `2.0`. The class semantics remain the same.
