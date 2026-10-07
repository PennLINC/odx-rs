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
- `Quantile(q)`: drop the lowest `q` fraction of values by count and keep
  every value at or above the `q`-quantile

Fixels below threshold are excluded from connectivity and become class `0`.

### Thresholds for QC

Otsu assumes the metric splits into two groups. QA, FA and FOD amplitude in
brain tissue are usually one skewed distribution, so Otsu's cut is close to
arbitrary and the coherence index inherits it: near an Otsu cut, moving the
threshold 10% can move coherence by about 2%.

`odx qc` and `Odx.coherence` therefore default to `Quantile(0.1)`
(`DEFAULT_QC_QUANTILE`; `FixelQcOptions::default()` keeps Otsu): every dataset is scored on the top 90% of its own
values, whatever the metric's scale, field of view or age group. The cut sits
in the sparse low tail, where coherence barely depends on it. DSI Studio gets
a similar effect from 0.6 × a whole-grid Otsu, but that moves with the amount
of empty field of view around the head.

`coherence_threshold_elasticity` reports d ln(coherence) / d ln(threshold),
measured between 0.8× and 1.25× the resolved threshold. Values near 0 mean the
index does not depend on where the cut fell; report it next to the index.

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

The default angle is `15` degrees, the same angle DSI Studio uses. The rule
itself is not DSI Studio's; for that, see
[Primary-fibre coherence](#primary-fibre-coherence) below.

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

`offset_unit` is the physical direction of the step: the voxel offset mapped
through the linear part of `VOXEL_TO_RASMM` and normalised. Fixel directions
are world RAS, so on LPS, oblique or anisotropic grids the raw index offset is
not the step a fibre takes. (Before this was fixed, an LPS grid scored a fibre
along index step `(1,0,1)` as perpendicular to that step.)

What this means geometrically:

- the source fixel must point roughly toward the neighboring voxel
- and a fixel in that neighbor voxel must point roughly the same way

This makes the QC measure purely local and spatial:

- it does use adjacent voxels
- it does not use same-voxel fixel matching
- it does not build connected components or multi-step paths
- it does not run tractography

This means the method favors coherent fixel chains that both point along the
neighbor trajectory and remain directionally consistent across voxels. A fibre
more than `angle_degrees` from all 13 lattice directions (for example 22.5°
from x in the xy plane at 15°) can never be connected, however coherent the
tissue.

### Primary-fibre coherence

`compute_primary_coherence` (`odx qc --mode primary`) is the fib-QC
coherence index of current DSI Studio (`evaluate_fib`, April 2026 source):

- each voxel contributes only its strongest fixel by the primary metric
- that fixel's direction is taken into voxel-index space, normalised and
  rounded component-wise, giving one lattice step
- the voxel is connected when the strongest fixel one step forward or back is
  within `angle_degrees` of it; the neighbour need not pass the threshold
- with `ThresholdMode::Otsu` the threshold is taken over the whole grid, with
  voxels that have no fixel counting as zero, as DSI Studio does

```text
coherence_index = weight(connected voxels) / weight(evaluated voxels)
```

Every direction rounds to some step, so unlike the fixel metric no fibre is
disconnected by its orientation alone. On grids with isotropic voxels this is
DSI Studio's rule exactly; on anisotropic grids the step is the voxel-index
direction of the fibre. The Otsu histogram differs in detail from DSI Studio's,
so thresholds can differ slightly.

DSI Studio has changed this metric repeatedly, so qsiprep's historical
`coherence_index` is none of the above:

| DSI Studio | `coherence_index` |
|---|---|
| `03476a6` (qsiprep's pin) | raw sum of QA over connection events (unnormalised); 0.6×Otsu; all fibres; both fibres within ~10° of a lattice step |
| Jan 2026 | that sum divided by the total QA of all fibres |
| Apr 2026 | the primary-fibre fraction above |

The primary rule measures how parallel neighbouring fibres are, not whether
fibres point along their path, so it is mostly a smoothness measure.

### Fixel chains

`compute_fixel_chains` (`odx qc --mode chain`) links fixels into chains and
reports how long they are: deterministic tractography with no seeds, step size
or interpolation, in the spirit of `dwigradcheck`'s mean streamline length.

- each evaluated fixel looks forward and backward along its own direction into
  the neighbouring voxels whose lattice step lies within 35° of it (the 26
  steps leave no wider gap, so every direction has a continuation)
- on each side it picks the single fixel within `angle_degrees` that best
  combines alignment with it and with the step
- a link is kept only if it is mutual, so every fixel has at most one link per
  side and the fixels split into simple chains (and, rarely, loops); finding
  each fixel's longest path through the full neighbour graph would instead be
  the NP-hard longest-path problem
- lengths are summed inter-voxel distances in mm through the affine

Reported: chain count, loops, the primary-metric-weighted mean length of the
chain each fixel sits in, the weighted share in chains of at least 20 and
40 mm, and the longest chain. Medians are not reported: most chains are a
fixel or two long, so medians sit at 0–2 mm whatever the data. Lengths are not
normalised by brain size; in practice the weighted mean barely depends on it,
and dividing by the cube root of the brain volume overcorrects.

One bad step breaks a whole chain, so chains respond to problems over a longer
range than the immediate-neighbour coherence index. They are much shorter than
streamlines (a strict per-step angle and no interpolation); compare them with
each other, not with tractography.

The b-table check scores candidates by weighted mean chain length by default
(`--btable-scoring chain`): a wrong table breaks chains almost at once, so the
true table wins by a far wider margin than with coherence scoring.

### B-table check

`check_btable` (`odx qc --check-btable`) replaces DSI Studio's
`check_btable` without refitting. Permuting or flipping the gradient axes of a
rotation-equivariant fit permutes or flips the fitted directions the same way:
exactly for a tensor, nearly so for SH or SHORE fits with rotation-invariant
regularisation. So each of the 24 axis permutations/flips is scored by applying
it to the fitted directions and recomputing coherence.

Score with chain length (`CoherenceMode::Chain`, the CLI and Python default)
or with the fixel rule (`CoherenceMode::Fixel`), whose trajectory gate (a
fixel must point at the neighbour it is compared with) is also what a wrong
table breaks; chain scoring separates the candidates more widely. The primary rule only asks whether neighbouring
fibres are parallel, and a permuted smooth field is still smooth, so its 24
scores sit close together and it can prefer a wrong flip.

Score only the strongest fixels: `DEFAULT_BTABLE_QUANTILE` (0.9) keeps the top
10% of the primary metric. Low-anisotropy fixels point almost at random under
every candidate, so a permissive cut dilutes the difference between the true
table and the rest until noise decides the winner. This is the opposite of
the coherence index, which wants the permissive cut (see
[Thresholds for QC](#thresholds-for-qc)).

- transforms act in the voxel axes (affine columns, normalised), the frame of
  FSL/dipy bvec files
- labels follow DSI Studio: `120fx` means new `(x, y, z)` = old `(y, z, x)`,
  then negate the new x; the best label is the correction to apply to the bvecs
- the identity `012` is listed first and wins ties, so `current_is_best` is
  true unless some candidate is strictly more coherent

A real anatomy with a correct table should score well above every other
candidate; compare `current_coherence_index` with `best_coherence_index`
rather than trusting `best` alone on noisy or tiny inputs.

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

Primary-fibre coherence and the b-table check take the same options:

```rust
use odx_rs::{check_btable, compute_primary_coherence, CoherenceMode, FixelQcOptions};

let options = FixelQcOptions::default(); // amplitude/afd/qa, Otsu, 15°
let report = compute_primary_coherence(&odx, &options)?;
let check = check_btable(&odx, &options, CoherenceMode::Chain)?;
if !check.current_is_best {
    eprintln!("b-table looks wrong; best fix is {}", check.best);
}
```

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

DSI Studio's coherence index, plus the b-table check:

```bash
odx qc input.odx --mode primary --check-btable --json
```

`--check-btable` works with any mode and scores candidates with
`--btable-scoring` (default `chain`) on the fixels at or above
`--btable-quantile` (default 0.9), whatever `--threshold` is. JSON output carries a `mode` key and, with
`--check-btable`, a `btable` object.

### Python

```python
ds = odx.load("sub-01_dwimap.odx", skip_odf=True, skip_sh=True)  # fixels only
report = ds.coherence()                       # primary mode, quantile, 15°
report = ds.coherence("fixel", threshold="positive", metric="afd")
report = odx.coherence(ds, check_btable=True)
report["btable"]["best"], report["btable"]["current_is_best"]
```

`threshold` is `"quantile"` (default, with `quantile=0.1`), `"otsu"`,
`"positive"`, `"all"` or a number. The b-table check uses `btable_quantile`
(default 0.9). Reports include `threshold_elasticity`.

Write the per-fixel class map back into the input ODX dataset:

```bash
odx qc input.odx --write-qc-class
```

Replace an existing `qc_class` field:

```bash
odx qc input.odx --write-qc-class --overwrite-qc-class
```

`--write-qc-class` only works for ODX directory or `.odx` archive inputs.

`odx qc` loads fixels, scalars and the grid only (`LoadOptions::fixels_only()`):
dense ODF and SH arrays are not extracted from archives or directories, and a
DSI Studio file's `odfN` records are decompressed and discarded without being
stored. QC memory is therefore set by the fixel count, not by the ODFs.

### Reading `qc_class` later

On disk, `qc_class` is stored as `uint8`. The current ODX loader may normalize
scalar DPFs to `f32` in memory, so downstream consumers may read it back as
`0.0`, `1.0`, and `2.0`. The class semantics remain the same.
