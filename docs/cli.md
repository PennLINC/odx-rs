# `odx` CLI

The `odx` binary is the command-line front end for `odx-rs`. It is library-driven: it detects the input format, normalizes it into an `OdxDataset`, and then writes the requested output through the existing ODX, DSI Studio, MRtrix, or interop APIs.

For DSI Studio and MRtrix convention details, see [docs/dsistudio_mrtrix_conversion_workflows.md](./dsistudio_mrtrix_conversion_workflows.md).
For fixel coherence QC details, see [docs/fixel_qc.md](./fixel_qc.md).

## Quick Start

Inspect a DSI Studio file:

```bash
odx info sub-01.fib.gz
```

Convert DSI Studio to ODX directory:

```bash
odx convert input.fib.gz output.odx --output-format odx-directory
```

Convert DSI Studio to MRtrix fixels plus SH:

```bash
odx convert input.fib.gz out_fixels --out-sh fod.mif.gz
```

Convert MRtrix fixels plus SH to DSI Studio `.fz`:

```bash
odx convert fixels_mif out.fz --sh fod.mif.gz
```

Convert MRtrix SH plus fixels to ODX:

```bash
odx convert fod.mif.gz output.odx --fixel-dir fixels_nii --output-format odx-directory
```

Validate a dataset:

```bash
odx validate output.odx
```

Compute fixel coherence QC:

```bash
odx qc output.odx
```

## Commands

### `odx info`

```bash
odx info <input>
```

Prints:

- detected format
- dimensions
- voxel and peak counts
- affine summary
- SH basis and order
- ODF, SH, DPV, and DPF array listings
- sphere metadata
- ODF sampling domain
- quantization metadata

Useful options:

- `--sh <path>`
- `--fixel-dir <path>`
- `--mapmri-tensor <path>` / `--mapmri-uvec <path>` (TORTOISE MAP-MRI input)
- `--reference-affine <path>`
- `--json`
- `--verbose`

### `odx convert`

```bash
odx convert <input> <output> [options]
```

Supported families:

- DSI Studio ↔ ODX
- MRtrix ↔ ODX
- DSI Studio ↔ MRtrix

The CLI uses path-based detection by default. Use `--input-format` or `--output-format` only when detection is ambiguous.
For ODX specifically, existing `.odx` paths are distinguished by filesystem type: directories load as ODX directories and files load as ODX ZIP archives. When creating a new `.odx` directory path, pass `--output-format odx-directory` because the target does not exist yet.

Shared input options:

- `--sh <path>`
- `--fixel-dir <path>`
- `--mapmri-tensor <path>` / `--mapmri-uvec <path>` (TORTOISE MAP-MRI input)
- `--reference-affine <path>`
- `--input-format odx-directory|odx-archive|dsistudio-fibgz|dsistudio-fz|dipy-pam5|tortoise-mapmri-nifti|mrtrix-sh-image|mrtrix-fixel-dir`

General output options:

- `--overwrite`
- `--quiet`
- `--json`

Output format selection:

- `--output-format odx-directory|odx-archive|dsistudio-fibgz|dsistudio-fz|dipy-pam5|mrtrix-sh-image|mrtrix-fixel-dir`

ODX options:

- `--quantize-dense`

MRtrix options:

- `--out-sh <path>` (write SH alongside a fixel-dir output)
- `--fixel-container mif|nifti` (default `nifti`)
- `--nifti2` (force NIfTI-2 when writing SH to `.nii`/`.nii.gz`)
- `--sh-lmax <even-int>`

The MRtrix SH container (mif vs nifti) and gzip compression are inferred from the output filename extension (`.mif`, `.mif.gz`, `.nii`, `.nii.gz`).

DSI Studio options:

- `--dense-odf off|from-sh`

DIPY PAM5:

PAM5 files do not record their SH basis. `shm_coeff` is read and written in
dipy's default basis, `descoteaux07` with `legacy=True`; SH in any other basis
(including MRtrix `tournier07`) is converted on output. Peak directions and SH
are stored in the voxel frame of the PAM affine, as dipy expects, and converted
to RAS on input. `total_weight` and `ang_thr` are written with dipy's EuDX
defaults (0.5 and 60) unless the dataset carries its own values.

### `odx validate`

```bash
odx validate <input>
```

Validation is performed on the normalized `OdxDataset`, even for foreign formats.

Checks include:

- mask cardinality vs `NB_VOXELS`
- offsets count and sentinel vs `NB_PEAKS`
- direction row count
- DPV and DPF row counts
- ODF row and hemisphere-column consistency
- SH coefficient count vs `SH_ORDER`
- required sphere metadata
- canonical dense representation consistency

Useful options:

- `--sh <path>`
- `--fixel-dir <path>`
- `--mapmri-tensor <path>` / `--mapmri-uvec <path>` (TORTOISE MAP-MRI input)
- `--reference-affine <path>`
- `--json`
- `--strict`

### `odx qc`

```bash
odx qc <input>
```

Computes sparse fixel coherence QC on the normalized `OdxDataset`.

In brief:

- choose a scalar primary metric: explicit `--primary-dpf`, otherwise `amplitude`,
  `afd`, then `qa`
- threshold fixels with `otsu`, `positive`, `all`, or a numeric override
- classify each evaluated fixel as connected or disconnected by scanning the 13
  undirected voxel-neighbor offsets, which is the efficient implementation of
  full immediate 26-neighbor voxel connectivity, and requiring directional
  agreement
- report weighted coherence/incoherence plus connected/disconnected counts and
  per-scalar-DPF summaries

`--mode primary` instead reports current DSI Studio's fib-QC coherence index
(each voxel's strongest fixel against the voxel one rounded lattice step along
it). `--check-btable` adds coherence under all 24 gradient-table axis
permutations/flips and names the best; `--btable-scoring` (default `chain`,
scoring fibre-weighted mean chain length) picks the rule. `--mode chain` links fixels into chains by mutual
best continuation and reports chain lengths in mm. Reports include
`threshold_elasticity`, how much the
index moves with the threshold. See [fixel_qc.md](./fixel_qc.md#thresholds-for-qc)
for why the coherence index and the b-table check use different cuts. Both modes step through the
affine, so LPS, oblique and anisotropic grids are scored physically.

Useful options:

- `--sh <path>`
- `--fixel-dir <path>`
- `--mapmri-tensor <path>` / `--mapmri-uvec <path>` (TORTOISE MAP-MRI input)
- `--reference-affine <path>`
- `--primary-dpf <name>`
- `--threshold quantile|otsu|positive|all|value` (default `quantile`)
- `--threshold-value <f32>`
- `--threshold-quantile <f32>` (default 0.1: score the top 90%)
- `--angle-deg <f32>`
- `--mode fixel|primary|chain`
- `--check-btable`
- `--btable-scoring fixel|primary|chain`
- `--btable-quantile <f32>` (default 0.9: the check scores the top 10%)
- `--write-qc-class` (fixel mode only)
- `--overwrite-qc-class`
- `--json`

When `--write-qc-class` is used on an ODX input, the CLI writes:

- `dpf/qc_class.uint8`

with the fixed encoding:

- `0 = thresholded out`
- `1 = disconnected`
- `2 = connected`

`--write-qc-class` is only valid for existing ODX directory or `.odx` archive
inputs.

### `odx compare`

```bash
odx compare --a <a.odx> --b <b.odx> --out-dir <dir> [options]
```

Pairwise fixel comparison between two ODX files: mutually matches fixels across
A and B, diffs the primary DPF metric, and writes per-voxel NIfTIs plus a
`comparison.odx` archive into `--out-dir`.

Useful options:

- `--a <a.odx>`, `--b <b.odx>` (both required)
- `--out-dir <dir>` (required)
- `--primary-dpf <name>` (default: `amplitude` → `afd` → `qa`)
- `--threshold otsu|positive|all|value`
- `--threshold-value <f32>`
- `--coherence-angle-deg <f32>` (default `15.0`)
- `--match-angle-deg <f32>` (default `30.0`)
- `--no-comparison-odx` (write NIfTIs only)
- `--json`

### `odx combine`

```bash
odx combine <odx>... [--input <odx>] [options]
```

The N-way generalization of `compare`: builds a shared set of group fixels,
matches every subject onto them, and writes a group ODX whose `angle_deg` DPF
is an (n_fixels × n_subjects) matrix, plus a cohort CSV. Inputs must share the
grid; a same-lattice input in a different voxel ordering (LAS vs RAS+) is
reindexed rather than rejected.

With `--method mean-fod` it also builds a group **ODF template** — the average
FOD, plus coverage, ℓ=0 spread, and angular-correlation reproducibility maps.
See [template.md](template.md) for that half of the command.

Useful options:

- `--method cluster|mean-fod` (default `cluster`): `cluster` pools subject
  directions; `mean-fod` averages the SH and peak-finds the mean FOD
- `--template <odx>` (adopt this ODX's fixels/geometry as the template)
- `--mask-combine union|intersection` (default `union`)
- `--min-coverage <frac>` (generalizes `--mask-combine`; `0` = union,
  `1` = intersection, `0.5` recommended for templates)
- `--match-angle-deg <f32>` (default `30.0`)
- `--normalize-fod none|l0|integral` (default `none`, `mean-fod` only) —
  `none` is right for quantitative FODs; `l0`/`integral` destroy AFD contrast
- `--lmax min|max|<N>` (default `min`), `--reference <odx>`
- `--fod-qc` / `--no-fod-qc`, `--loo auto|on|off`, `--acc-lmin <N>` (default `2`)
- `--average-dpv <name>` (repeatable), `--no-average-dpv`, `--dpv-sd`
- `--out-report <json>`, `--fail-on-outlier`
- `--npeaks`, `--peak-threshold`, `--min-separation-angle` (`mean-fod` peak finding)
- `--min-subjects <N>` (default `2`, `cluster` only)
- `--scalar <name>` (repeatable; restrict carried DPF scalars)
- `--design <tsv|csv>`, `--design-key-column <col>`, `--input-key stem|path`
- `--out-odx <path>`, `--out-cohort <csv>`, `--out-mask <nifti>`,
  `--per-subject-odx <dir>`, `--out-table <csv|tsv>`, `--out-dir <dir>`
- `--json`

**Units trap in the outputs.** `dispersion` / `primary_dispersion` are
*dimensionless* dyadic-tensor spreads in `[0, 2/3]`, **not** degrees, and they
grow as the **square** of the angle. A typical white-matter value of `0.01`
means about **6°**, not 1° and not 1%. Convert with
`angle = degrees(asin(sqrt(dispersion)))`, or just read `sd_angle_deg` /
`primary_sd_angle_deg.nii.gz`, which carry the same information in degrees.
Anything named `*_deg` is in degrees; anything not is not.

### `odx import-aodf`

```bash
odx import-aodf <input.nii.gz> <output.odx> [options]
```

Converts a pyAFQ asymmetric ODF (`*_param-aodf_dwimap.nii.gz`) into ODX. Stores
full-basis descoteaux07 SH and precomputes per-voxel asymmetric peaks.

Useful options:

- `--sidecar <json>` (defaults to a JSON beside the NIfTI)
- `--legacy-basis` (use legacy descoteaux SH; default is non-legacy)
- `--relative-peak-threshold <f32>` (default `0.5`)
- `--min-separation-deg <f32>` (default `25.0`)
- `--max-peaks <N>` (default `5`)
- `--odx-layout directory|archive` (default `directory`)
- `--overwrite`
- `--json`

### `odx upsample`

```bash
odx upsample <input.odx> <output.odx> --voxel-spacing <mm> [options]
```

Spatially upsamples an ODX onto a finer isotropic voxel grid. SH and DPV arrays
are trilinearly interpolated; fixels are recomputed from the interpolated SH by
peak finding. DPF arrays other than `amplitude` are dropped, and dense ODF data
is not supported.

Useful options:

- `--voxel-spacing <mm>` (required; target isotropic spacing)
- `--npeaks <N>` (default `5`)
- `--peak-threshold <f32>` (default `0.5`)
- `--min-separation-angle <f32>` (default `25.0`)
- `--odx-layout directory|archive` (default `directory`)
- `--overwrite`
- `--json`

### `odx transform`

```bash
odx transform <input.odx> <output.odx> --transform <h5> [options]
```

Warps an ODX dataset (SH coefficients, per-voxel scalars, fixels) onto a new
spatial grid using an ANTs/ITK Composite `.h5` (with embedded warp + affines)
an Insight Transform File V1.0 (`.txt`, affine-only), or an ITK MATLAB
v4 binary (`.mat`, affine-only — what ANTs writes for `*0GenericAffine.mat`).

#### Direction convention (cartoon BIDS)

`odx transform` resamples a sampled grid, so it follows the **same-direction**
h5 convention as `antsApplyTransforms` for an image — **not** the opposite-named
convention used by `trxrs`/`giftirs` for points.

| You have                                       | You want                              | Pass to `--transform`                            |
| ---------------------------------------------- | ------------------------------------- | ------------------------------------------------ |
| `sub-01_space-ACPC_dwimap.odx`                 | ODX in `MNI152NLin2009cAsym`          | `sub-01_from-ACPC_to-MNI152NLin2009cAsym_xfm.h5` |
| `sub-01_space-MNI152NLin2009cAsym_dwimap.odx`  | ODX in `ACPC`                         | `sub-01_from-MNI152NLin2009cAsym_to-ACPC_xfm.h5` |

Pull-based grid resampling means: at each output (target) voxel, the chain
inside `from-X_to-Y_xfm.h5` returns the source X coordinate to sample from.
That's also what `antsApplyTransforms` does for an image of the same space,
so the file you pass is the same.

For the same subject, contrast with `trxrs`:

```text
sub-01_space-ACPC_tracts.trx       → MNI:  trxrs       --transform sub-01_from-MNI...to-ACPC_xfm.h5
sub-01_space-ACPC_dwimap.odx       → MNI:  odx transform --transform sub-01_from-ACPC_to-MNI..._xfm.h5
                                                                       ^^^^^^ opposite naming
```

#### Worked example (mrtrix mode, default)

```bash
odx transform \
    sub-01_space-ACPC_dwimap.odx \
    sub-01_space-MNI152NLin2009cAsym_dwimap.odx \
    --transform sub-01_from-ACPC_to-MNI152NLin2009cAsym_xfm.h5
```

The output grid (dimensions + voxel-to-world affine) is taken from the warp's
displacement field embedded in the h5. For an affine-only h5, supply
`--reference <target.nii.gz>` to specify the output grid explicitly.

#### Modes: pull-everything (`mrtrix`) vs split (`ants`)

- `--mode mrtrix` (default): SH, DPV, AND fixels are pulled via the single
  `--transform` h5 (target → source). Matches `mrtransform` +
  `fixeltransform` semantics. Fixels may be duplicated or dropped at
  non-uniform warp regions (no fixel-correspondence guarantees).
- `--mode ants`: SH and DPV are pulled via `--transform`, but fixels are
  **pushed** via `--transform-inverse` (source → target). Each source fixel
  maps to exactly one target voxel, preserving cardinality. Use when you
  have an ANTs-style paired h5 set.

```bash
odx transform \
    sub-01_space-ACPC_dwimap.odx \
    sub-01_space-MNI152NLin2009cAsym_dwimap.odx \
    --mode ants \
    --transform         sub-01_from-ACPC_to-MNI152NLin2009cAsym_xfm.h5 \
    --transform-inverse sub-01_from-MNI152NLin2009cAsym_to-ACPC_xfm.h5
```

#### Useful options

- `--reference <target.nii.gz>`: required for affine-only chains; ignored when
  the h5 contains a displacement field (the warp's grid wins).
- `--invert`: numerically invert the chain. Affine-only chains only.
- `--modulate`: mrtrix-style SH modulation (`‖J·d‖/det(J)`, equivalent to
  `mrtransform -modulate fod`). Off by default. Fixels are never modulated.
- `--apsf-dirs <N>`: number of fibonacci-spiral reference directions for
  aPSF SH reorientation. Default 80 (covers lmax 8 reliably); use 300 for
  lmax 12.

### `odx graddev`

```bash
odx graddev <input> <output> --graddev <graddev.nii.gz> [options]
```

Applies a gradient-nonlinearity deviation field to a reconstruction and writes
it back in its native format. A reconstruction fitted with a single global
b-table (every ODF/FOD and every peak) is expressed against a locally distorted
q-space wherever the scanner gradients deviate from nominal; this command
rotates the fitted quantities per voxel so they describe the tissue instead.

Inputs are the same as `odx convert` (DSI Studio `.fib.gz`/`.fz`, MRtrix SH
image with optional `--fixel-dir`, MRtrix fixel directory, DIPY `.pam5`, ODX);
outputs are any of those except `.pam5`. The output grid, mask, scalar arrays
and peak count are unchanged.

What is corrected, per voxel with matrix `T` (see below):

- SH / FOD coefficients: `ψ_true(u) = ψ_est(normalize(T·u))`, via the same aPSF
  reorienter as `odx transform` (no modulation).
- Dense ODF amplitudes (DSI Studio `odf` records, PAM `odf`): refit to SH
  (`--odf-lmax`, default up to 12) and re-evaluated at the rotated sphere
  directions.
- Fixel / peak directions: `normalize(T⁻¹·u)`.

Not corrected: the per-voxel b-value deviation `|Tᵀ·g|²`. It cannot be undone
on an already-fitted ODF and is only reported (`max_abs_log_det`). DSI Studio's
own `grad_dev.nii.gz` handling at `.src` creation time is the complementary
magnitude-only correction; it never rotates ODFs or peaks.

#### File convention

The field is a 4-D NIfTI with 9 volumes in the HCP/FSL `grad_dev` layout.
Reading one voxel's 9 values row-major into `T` (`T[i][j] = vol[3*i + j]`,
numpy `reshape(3, 3)` in C order) the effective gradient is `Tᵀ · g` in the
image's own voxel axes. This is FSL's `correct_bvals_bvecs` (which fills its
matrix column-major and applies `(I + L)·g`) and TORTOISE's
`CreateGradientNonlinearityBMatrix` output (qsiprep `*_space-ACPC_graddev.nii.gz`).
The field is rotated into RAS with the affine's rotation part and sampled at
each ODX voxel centre (nearest neighbour); it is never axis-canonicalized.

`--identity auto|included|absent` says whether the diagonal already holds the
identity: TORTOISE/qsiprep files do (`included`, diagonal ≈ 1); HCP/FSL
`grad_dev` files store `T − I` (`absent`). `auto` inspects the median diagonal.

#### DSI Studio output

`index{p}` snaps every peak to the 642-vertex ODF sphere (~8° apart), which
would swallow a typical 1–4° correction. DSI Studio's tracker prefers exact
float `dir{p}` records whenever they exist, in both `.fib.gz` and `.fz`, so
`odx graddev` always writes them (and `odx convert --dsistudio-float-dirs`
can). The loader also reads them back in preference to `index{p}`.

Useful options:

- `--reference-affine <nifti>`: required for a native-space `.fib.gz`/`.fz`
  without a `trans` record (e.g. the preprocessed DWI it was built from).
- `--odf-lmax <even-int>`, `--apsf-dirs <N>`: accuracy knobs for the dense-ODF
  and SH paths.
- `--json`: machine-readable summary including the correction report
  (`nb_corrected`, `nb_outside_field`, `median_rotation_deg`, `max_rotation_deg`,
  `max_abs_log_det`, arrays touched, dropped DPFs).

### `odx attach-dpv`

```bash
odx attach-dpv <odx> <nifti> --name <name> [options]
```

Attaches a NIfTI volume to an existing ODX as a per-voxel scalar (DPV), in
place. The NIfTI grid must match the ODX (dimensions and affine within 1e-3 mm);
voxels outside the ODX mask are silently dropped. Overwrites the DPV if a field
of the same name already exists.

Useful options:

- `--name <name>` (required; DPV name to register under, e.g. `fa`)
- `--dtype auto|u8|u16|u32|i16|i32|f32|f64` (default `auto`: narrowest unsigned
  int that fits non-negative integer data, else `float32`)
- `--quiet`

## Input Model

Some formats are composite:

- MRtrix fixel directories may be accompanied by an SH image.
- DSI Studio `fib.gz` inputs may need `--reference-affine` when the file does not contain a usable transform.
- ODX may hold sparse fixels, dense ODFs, and SH together in one dataset.

The CLI therefore treats `<input>` as the primary object and accepts companion inputs as flags instead of requiring a manifest format.

## Output and Exit Codes

- exit code `0`: success
- nonzero exit code: conversion, validation, or parsing failure

When `--json` is used:

- `info` prints a dataset summary object
- `validate` prints a validation report object
- `convert` prints a short conversion summary object
- `qc` prints the QC report object
