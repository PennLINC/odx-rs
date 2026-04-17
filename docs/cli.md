# `odx` CLI

The `odx` binary is the command-line front end for `odx-rs`. It is library-driven: it detects the input format, normalizes it into an `OdxDataset`, and then writes the requested output through the existing ODX, DSI Studio, MRtrix, or interop APIs.

For DSI Studio and MRtrix convention details, see [docs/dsistudio_mrtrix_conversion_workflows.md](/Users/mcieslak/projects/odx/odx-rs/docs/dsistudio_mrtrix_conversion_workflows.md).

## Quick Start

Inspect a DSI Studio file:

```bash
odx info sub-01.fib.gz
```

Convert DSI Studio to ODX directory:

```bash
odx convert input.fib.gz output.odx --odx-layout directory
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
odx convert fod.mif.gz output.odx --fixel-dir fixels_nii --odx-layout directory
```

Validate a dataset:

```bash
odx validate output.odx
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
For ODX specifically, existing `.odx` paths are distinguished by filesystem type: directories load as ODX directories and files load as ODX ZIP archives. When creating a new `.odx` directory path, pass `--odx-layout directory` because the target does not exist yet.

Shared input options:

- `--sh <path>`
- `--fixel-dir <path>`
- `--reference-affine <path>`

General output options:

- `--overwrite`
- `--quiet`
- `--json`

ODX options:

- `--odx-layout directory|archive`
- `--quantize-dense`
- `--quantize-min-len <usize>`

MRtrix options:

- `--out-sh <path>`
- `--mrtrix-fixel-container mif|nifti`
- `--mrtrix-sh-container mif|nifti1|nifti2`
- `--mrtrix-sh-gzip`
- `--sh-lmax <even-int>`

DSI Studio options:

- `--dsi-format fibgz|fz`
- `--dense-odf off|from-sh`
- `--peak-source fixels|sampled-odf`
- `--amplitude-key <name>`
- `--z0 auto|never|always`

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
- `--reference-affine <path>`
- `--json`
- `--strict`

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
