# ODX File Format Specification

Version 0.1.0

## Overview

ODX is a file format for storing orientation density functions (ODFs), peaks/fixels,
and associated scalar data from diffusion MRI. It is designed as a companion to the
[TRX](https://github.com/tee-ar-ex/trx-spec) tractography format and follows the
same design principles:

- Directory or ZIP archive layout
- Memory-mappable flat binary arrays
- Data type and column count encoded in filenames
- RAS+mm coordinate convention throughout
- Language-agnostic (no Python pickle, no MATLAB)

By convention, ODX treats sparse `peaks` plus spherical harmonic (`sh/`)
coefficients as the preferred compact representation. Sampled ODF amplitudes in
`odf/` are optional and primarily retained for dsistudio-style interchange,
visualization, and QA.

## Motivation

Each existing format has limitations that ODX addresses:

ODX is intended to be a format with high-fidelity forward and backward conversions to

- DSI Studio `fib.gz`
- DSI Studio `fz`
- MRtrix3 fixels directories
- MRtrix3 sh mif/nifti files
- DIPY Peaks And Metrics `pam5`/`dpy` files

Conversions to/from each of these to ODX means that we can convert pairwise between 
any of them - which is really nice.

## File Layout

An ODX dataset is either a directory (conventionally suffixed `.odx`) or a ZIP
archive (suffixed `.odx`).

```
data.odx/
  header.json                       # spatial metadata and counts
  mask.uint8                        # brain mask (full volume, C-order)
  offsets.uint32                    # peak offset indices (NB_VOXELS + 1)
  directions.3.{dtype}              # peak unit vectors (NB_PEAKS x 3)

  sphere/                           # ODF sampling sphere (optional)
    vertices.3.{dtype}              #   unit directions (NB_SPHERE_VERTICES x 3)
    faces.3.uint32                  #   triangle indices (NB_SPHERE_FACES x 3)

  odf/                              # ODF amplitudes (optional)
    amplitudes.{ndirs}.{dtype}      #   (NB_VOXELS x NB_SPHERE_VERTICES)

  sh/                               # spherical harmonic coefficients (optional)
    coefficients.{ncoeffs}.{dtype}  #   (NB_VOXELS x ncoeffs)

  dpv/                              # data per voxel (optional)
    {name}.{dtype}                  #   scalar: (NB_VOXELS,)
    {name}.{ncols}.{dtype}          #   vector: (NB_VOXELS x ncols)

  dpf/                              # data per fixel/peak (optional)
    {name}.{dtype}                  #   scalar: (NB_PEAKS,)
    {name}.{ncols}.{dtype}          #   vector: (NB_PEAKS x ncols)

  groups/                           # voxel groups (optional)
    {name}.uint32                   #   array of voxel indices

  dpg/                              # data per group (optional)
    {group_name}/
      {field}.{ncols}.{dtype}
```

## Header (`header.json`)

```json
{
  "VOXEL_TO_RASMM": [[1.25, 0, 0, -90], [0, 1.25, 0, -126], [0, 0, 1.25, -72], [0, 0, 0, 1]],
  "DIMENSIONS": [145, 174, 145],
  "NB_VOXELS": 72534,
  "NB_PEAKS": 198421,
  "NB_SPHERE_VERTICES": 642,
  "NB_SPHERE_FACES": 1280,
  "SH_ORDER": 8,
  "SH_BASIS": "descoteaux07",
  "CANONICAL_DENSE_REPRESENTATION": "sh",
  "SPHERE_ID": "odf8"
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `VOXEL_TO_RASMM` | `[[f64; 4]; 4]` | 4x4 affine matrix mapping voxel indices to RAS+mm world coordinates. Row-major. |
| `DIMENSIONS` | `[u64; 3]` | Volume grid size `[x, y, z]`. |
| `NB_VOXELS` | `u64` | Number of masked voxels. Determines the row count for all per-voxel arrays (dpv/, odf/, sh/) and the length of the offsets array. |
| `NB_PEAKS` | `u64` | Total number of peaks/fixels across all voxels. Determines the row count for directions and all per-fixel arrays (dpf/). |

### Conditional Fields

| Field | Present when | Type | Description |
|-------|-------------|------|-------------|
| `NB_SPHERE_VERTICES` | `sphere/` exists | `u64` | Number of ODF sampling directions. |
| `NB_SPHERE_FACES` | `sphere/` exists | `u64` | Number of triangles in the sphere mesh. |
| `SH_ORDER` | `sh/` exists | `u64` | Maximum spherical harmonic order. |
| `SH_BASIS` | `sh/` exists | `string` | SH basis convention: `"descoteaux07"` (Dipy) or `"tournier07"` (MRtrix). |
| `CANONICAL_DENSE_REPRESENTATION` | optional | `string` | Preferred dense representation: `"sh"` or `"odf"`. |
| `SPHERE_ID` | optional | `string` | Canonical sphere identifier when `odf/` uses a standard sphere and explicit `sphere/` payloads can be omitted. |
| `ARRAY_QUANTIZATION` | optional | `object` | Per-array linear quantization metadata for ODX-native `uint8 + slope/intercept` storage. |

Additional fields are preserved through round-trips via a catch-all map.

## Core Data Arrays

### Mask (`mask.uint8`)

A flat `uint8` array of length `DIMENSIONS[0] * DIMENSIONS[1] * DIMENSIONS[2]`.

**Storage order:** C order (row-major). The flat index for voxel `(i, j, k)` is:

```
index = i * DIMENSIONS[1] * DIMENSIONS[2] + j * DIMENSIONS[2] + k
```

Values are 0 (background) or 1 (brain). The number of nonzero entries must
equal `NB_VOXELS`.

**Voxel ordering convention:** The n-th nonzero entry in the flat mask (scanning
in C order) corresponds to row n in all per-voxel arrays (`dpv/`, `odf/`, `sh/`)
and to position n in the offsets array.

Reconstructing a 3D volume from per-voxel data:

```python
import numpy as np

mask_3d = mask.reshape(DIMENSIONS, order='C')
flat_indices = np.flatnonzero(mask_3d.ravel(order='C'))

volume_flat = np.zeros(np.prod(DIMENSIONS), dtype=np.float32)
volume_flat[flat_indices] = dpv_values
volume_3d = volume_flat.reshape(DIMENSIONS, order='C')
```

### Offsets (`offsets.uint32`)

Array of `NB_VOXELS + 1` uint32 values. Exactly analogous to TRX's streamline
offsets.

- `offsets[i]` is the start index (into the directions/dpf arrays) of voxel i's peaks.
- `offsets[i+1] - offsets[i]` is the number of peaks for voxel i.
- `offsets[NB_VOXELS]` equals `NB_PEAKS` (sentinel value).
- A voxel with zero peaks has `offsets[i] == offsets[i+1]`.

### Directions (`directions.3.{dtype}`)

Flat array of shape `(NB_PEAKS, 3)`. Each row is a unit vector in RAS+mm
coordinates giving the orientation of one peak/fixel.

Directions are stored as continuous 3D unit vectors, not as indices into the
ODF sphere. This preserves full angular precision and decouples peak data from
any particular sphere tessellation.

Supported dtypes for directions: `float16`, `float32`, `float64`.

## ODF Sphere (`sphere/`)

Defines the discrete sphere on which ODF amplitudes are sampled.

### `sphere/vertices.3.{dtype}`

Shape `(NB_SPHERE_VERTICES, 3)`. Unit vectors on the sampling sphere, in
RAS+mm orientation.

### `sphere/faces.3.uint32`

Shape `(NB_SPHERE_FACES, 3)`. Triangle indices into the vertex array, for
mesh visualization and interpolation.

The sphere directory is required when `odf/` data is present on a noncanonical
sphere. It may be omitted when `SPHERE_ID` identifies a standard sphere. It is
not required when only SH coefficients or peaks are stored.

## ODF Amplitudes (`odf/`)

### `odf/amplitudes.{ndirs}.{dtype}`

Shape `(NB_VOXELS, NB_SPHERE_VERTICES)`. Row i contains the ODF amplitude
values for masked voxel i, evaluated at each sphere vertex. The column count
(`ndirs`) equals `NB_SPHERE_VERTICES` and is encoded in the filename (e.g.,
`amplitudes.642.float32`).

Additional named ODF arrays may be placed in this directory (e.g.,
`odf/normalized.642.float32`).

## Spherical Harmonic Coefficients (`sh/`)

### `sh/coefficients.{ncoeffs}.{dtype}`

Shape `(NB_VOXELS, ncoeffs)`. Row i contains the SH coefficients for masked
voxel i.

The number of coefficients for even-order symmetric SH of maximum order L is:

```
ncoeffs = (L + 1)(L + 2) / 2
```

For example: order 8 has 45 coefficients.

The `SH_ORDER` and `SH_BASIS` header fields define interpretation. Supported
bases:

- `"descoteaux07"` — Descoteaux et al. 2007 real symmetric basis (Dipy convention)
- `"tournier07"` — Tournier et al. 2007 basis (MRtrix convention)

An ODX file may contain both `odf/` and `sh/` representations. When both are
present, they should be consistent. `sh/` is the preferred compact dense
representation; `odf/` remains optional for sampled-ODF workflows and
dsistudio-style interchange.

## Per-Voxel Data (`dpv/`)

One row per masked voxel (`NB_VOXELS` rows). Exactly analogous to TRX's
data-per-streamline (`dps/`).

Examples:

| File | Shape | Description |
|------|-------|-------------|
| `dpv/gfa.float32` | `(NB_VOXELS,)` | Generalized fractional anisotropy |
| `dpv/fa.float32` | `(NB_VOXELS,)` | Fractional anisotropy |
| `dpv/md.float32` | `(NB_VOXELS,)` | Mean diffusivity |
| `dpv/tensor.6.float32` | `(NB_VOXELS, 6)` | Unique diffusion tensor elements |
| `dpv/isovf.float32` | `(NB_VOXELS,)` | Isotropic volume fraction |

Any name is valid. Scalar (1 column) and multi-column arrays are both supported.

## Per-Fixel/Peak Data (`dpf/`)

One row per peak (`NB_PEAKS` rows total). Indexed via offsets: the data for
voxel i's peaks spans `dpf[offsets[i]..offsets[i+1]]`. Exactly analogous to
TRX's data-per-vertex (`dpv/`).

Examples:

| File | Shape | Description |
|------|-------|-------------|
| `dpf/amplitude.float32` | `(NB_PEAKS,)` | Peak amplitude (equiv. to DSI Studio fa0..faN) |
| `dpf/afd.float32` | `(NB_PEAKS,)` | Apparent fiber density |
| `dpf/dispersion.float32` | `(NB_PEAKS,)` | Orientation dispersion |
| `dpf/icvf.float32` | `(NB_PEAKS,)` | Intra-cellular volume fraction per fixel |
| `dpf/qc_class.uint8` | `(NB_PEAKS,)` | Optional QC class map: `0=thresholded out`, `1=disconnected`, `2=connected` |

### QC Convention: `dpf/qc_class.uint8`

`odx-rs` can store the output of sparse fixel coherence QC as a reserved scalar
DPF:

```text
dpf/qc_class.uint8
```

The value semantics are fixed:

- `0` — fixel was excluded by the QC threshold
- `1` — fixel was evaluated and found disconnected
- `2` — fixel was evaluated and found connected

This field is optional. It is intended as a compact categorical annotation of
existing fixels rather than a new source metric. QC implementations should not
use `qc_class` itself as the primary weighting metric when recomputing QC.
For `.odx` archives, `odx-rs` can append or replace this field efficiently
without rebuilding the entire archive; see [ZIP Archive Support](#zip-archive-support).

## Groups and Per-Group Data (`groups/`, `dpg/`)

Identical to TRX.

- `groups/{name}.uint32` — flat array of voxel indices (0-based into the
  masked voxel ordering) belonging to the group.
- `dpg/{group_name}/{field}.{ncols}.{dtype}` — metadata arrays for each group.

Use cases: atlas parcellations, tissue segmentation labels, hemisphere masks.

## Filename Convention

Identical to TRX. Each binary data file is named:

- `{name}.{ncols}.{dtype}` for multi-column arrays (e.g., `directions.3.float32`)
- `{name}.{dtype}` for single-column arrays (e.g., `amplitude.float32`, ncols=1 implied)

Parsing is right-to-left, allowing dots in the name portion
(e.g., `my.metric.1.float64` parses as name=`my.metric`, ncols=1, dtype=float64).

### Supported Data Types

| Name | Size | Directions | DPV/DPF | Offsets |
|------|------|------------|---------|---------|
| `float16` | 2 bytes | yes | yes | no |
| `float32` | 4 bytes | yes | yes | no |
| `float64` | 8 bytes | yes | yes | no |
| `int8` | 1 byte | no | yes | no |
| `int16` | 2 bytes | no | yes | no |
| `int32` | 4 bytes | no | yes | no |
| `int64` | 8 bytes | no | yes | no |
| `uint8` | 1 byte | no | yes | no |
| `uint16` | 2 bytes | no | yes | no |
| `uint32` | 4 bytes | no | yes | yes |
| `uint64` | 8 bytes | no | yes | yes |

## Coordinate Conventions

All spatial data uses **RAS+mm**:

- **R**ight, **A**nterior, **S**uperior positive
- Units: millimeters

This applies to:

- `VOXEL_TO_RASMM` affine matrix
- `sphere/vertices` (unit vectors in RAS orientation)
- `directions` (peak unit vectors in RAS orientation)

### Conversion from other conventions

| Source | Transform |
|--------|-----------|
| DSI Studio (LPS+) | Negate x and y components: `x_ras = -x_lps`, `y_ras = -y_lps`, `z_ras = z_lps` |
| MRtrix (RAS) | No change needed for directions. Affine from NIfTI header. |
| Dipy (RAS) | No change needed. Affine from NIfTI header. |

## Memory Mapping

All binary arrays are raw flat binary with no embedded headers. The dtype and
shape are fully determined by the filename (ncols, dtype) and the header counts
(`NB_VOXELS`, `NB_PEAKS`, `NB_SPHERE_VERTICES`). This allows zero-copy
memory-mapped access: readers can mmap any file and cast the bytes directly to
typed arrays.

## ZIP Archive Support

A `.odx` file is a ZIP archive containing the directory layout above. Entries
may use deflate compression. For memory-mapped access, the ZIP is extracted to a
temporary directory at load time (same pattern as trx-rs). For streaming or
write-once use, entries can be read/written directly from/to the ZIP.

`odx-rs` also supports archive mutation for a safe subset of payload entries.

### Efficient archive edits

For existing `.odx` archives, `odx-rs` can add, replace, and delete entries in:

- `dpf/`
- `dpv/`
- `groups/`
- `dpg/`

The implementation uses two paths:

- **append fast path** for pure additions of new entry names
- **selective rewrite path** for replacements, deletions, prefix deletions, or
  coordinated `header.json` updates

Append-only edits use ZIP append mode directly. Rewrite edits copy unchanged ZIP
members forward without decompressing and recompressing every file, skip deleted
or replaced members, then write the new payloads and replace the original
archive.

This means archive mutation is usually much cheaper than extracting the full
archive to a directory tree and rebuilding it from scratch.

### Replacement and deletion semantics

- Archive entries are treated as unique by logical path.
- Replacing an entry means rewriting the archive with the old path removed and
  the new path written once.
- Deletion can target a single entry path or a whole prefix such as
  `dpg/<group>/`.
- Deleting a group also removes its `dpg/<group>/` subtree.
- `odx-rs` does not rely on duplicate ZIP member names or shadowed entries.

### Header and metadata updates

Most payload additions and deletions do not require header changes beyond row
count validation against the existing `NB_VOXELS` or `NB_PEAKS`.

When an edited array has an `ARRAY_QUANTIZATION` entry, `odx-rs` updates
`header.json` in the same transaction as the ZIP mutation so that stale
quantization metadata is not left behind.

Archive comments are preserved across selective rewrites.

## Structural Analogy to TRX

| TRX Concept | ODX Analog |
|-------------|------------|
| Streamlines (variable-length vertex sequences) | Voxels (variable-count peaks) |
| `offsets.uint32` → positions | `offsets.uint32` → directions |
| `positions.3.float32` (all vertices) | `directions.3.float32` (all peak vectors) |
| `dps/` (data per streamline) | `dpv/` (data per voxel) |
| `dpv/` (data per vertex) | `dpf/` (data per fixel/peak) |
| `groups/` | `groups/` (identical) |
| `dpg/` | `dpg/` (identical) |
| — | `mask.uint8` (defines the sparse voxel set) |
| — | `sphere/` (ODF sampling geometry) |
| — | `odf/`, `sh/` (dense per-voxel distributions) |
