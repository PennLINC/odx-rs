# DSI Studio ↔ MRtrix Conversion Workflows

This document explains the conversion choices implemented in `odx-rs` for:

- DSI Studio `fib.gz` / `.fz` ↔ `OdxDataset` ↔ MRtrix fixels + optional SH image

The key architectural rule is that `OdxDataset` is the only semantic intermediate. The interop helpers in `odx-rs` are orchestration wrappers, not a second pairwise converter implementation. That keeps the file-format logic in the existing backends:

- DSI Studio I/O in [src/formats/dsistudio.rs](../src/formats/dsistudio.rs)
- MRtrix I/O in [src/formats/mrtrix.rs](../src/formats/mrtrix.rs)
- shared MRtrix-authoritative SH math in [src/mrtrix_sh.rs](../src/mrtrix_sh.rs)

## Voxel Coordinates vs. Physical Coordinates

Despite physical brains existing in the world in continuous coordinates, at the end of the day we have to work with brain measures that are discrete. The voxels measured by MRIs are discrete, and algorithms in computer programs often use voxel indices to find a voxel's neighbor. Since most computations have to happen in voxel space, many software packages started to make assumptions about the direction of the voxel indices. In particular, DSI Studio hard-codes the voxel indexes to increase as the voxels are spatially moving to the left, posterior and anterior. This is an "LPS+ voxel orientation". In TRXViz and nifti, tck and trx file formats, coordinates are in a physical coordinate system where values increase numerically as they move to the right, anterior and superior directions.

Physical coordinate space will _always_ be RAS+. There is no other allowable convention for describing physical space in this software ecosystem.

Voxels can have many different orientations in nifti format, but the nifti affine _always_ maps the voxel indices to their coordinate centroids in RAS+ physical space. DSI Studio enforces an LPS+ voxel orientation in fib.gz and fz files, but other software does not care.

`OdxDataset` does not require any particular voxel orientation. `VOXEL_TO_RASMM` is an arbitrary affine that maps whatever voxel ordering the dataset uses to RAS+ physical mm. Only the DSI Studio write path reorders voxels to LPS+ orientation, because DSI Studio's reader enforces it.

## Conventions

| Concept | ODX internal convention | DSI Studio convention | MRtrix convention | Source backing |
| --- | --- | --- | --- | --- |
| Affine | `VOXEL_TO_RASMM`: voxel indices → RAS+ physical mm (voxel orientation is unconstrained) | `trans` / `trans_to_mni`: LPS+-indexed voxels → RAS+ physical mm | NIfTI/MIF affine: voxel indices → RAS+ physical mm | [src/formats/dsistudio.rs](../src/formats/dsistudio.rs), [src/formats/mrtrix.rs](../src/formats/mrtrix.rs) |
| Directions / sphere vertices | RAS | LPS sphere, flipped with `[-x, -y, z]` on import/export | copied as loaded into RAS-style ODX | [src/formats/dsistudio.rs](../src/formats/dsistudio.rs) |
| Sparse peak/fixel order | masked C-order | voxelwise `faN/indexN` tables | `index[...,0]=count`, `index[...,1]=first_index` | [src/formats/mrtrix.rs](../src/formats/mrtrix.rs), [trx-mrtrix2/cpp/core/fixel/helpers.h](../../trx-mrtrix2/cpp/core/fixel/helpers.h) |
| Dense ODF domain | explicit `ODF_SAMPLE_DOMAIN` | hemisphere amplitudes on a full sphere geometry | SH sampled onto chosen directions | [qsirecon/qsirecon/interfaces/converters.py](../../qsirecon/qsirecon/interfaces/converters.py), [src/formats/dsistudio.rs](../src/formats/dsistudio.rs) |

## Why MRtrix Is Authoritative For `tournier07`

The SH basis, coefficient indexing, and transform matrices in `odx-rs` are derived from MRtrix source, not from qsirecon:

- coefficient indexing and transform construction:
  [trx-mrtrix2/cpp/core/math/SH.h](../../trx-mrtrix2/cpp/core/math/SH.h)
- `sh2amp` command behavior and `-nonnegative` clamping:
  [trx-mrtrix2/cpp/cmd/sh2amp.cpp](../../trx-mrtrix2/cpp/cmd/sh2amp.cpp)
- `amp2sh` least-squares fitting:
  [trx-mrtrix2/cpp/cmd/amp2sh.cpp](../../trx-mrtrix2/cpp/cmd/amp2sh.cpp)

`qsirecon` is still useful as a workflow reference, especially for how real data has been moved between DSI Studio and MRtrix in practice, but it is not treated as the basis authority.

## Why MRtrix Fixels Are Rebuilt From `index`

MRtrix fixel directories are not defined by the raw row order of `directions`, `afd`, `disp`, or any other fixelwise file. The semantic mapping comes from:

- `index[...,0]` = count
- `index[...,1]` = first fixel row

That is explicit in:

- [trx-mrtrix2/cpp/core/fixel/helpers.h](../../trx-mrtrix2/cpp/core/fixel/helpers.h)
- [trx-mrtrix2/cpp/gui/mrview/tool/fixel/directory.cpp](../../trx-mrtrix2/cpp/gui/mrview/tool/fixel/directory.cpp)

`odx-rs` therefore canonicalizes fixels by scanning voxels in masked C-order and rebuilding sparse rows from `index`. This is also what makes the MIF and NIfTI fixel fixtures agree after import.

## DSI `trans` / `trans_to_mni`

Despite the name, DSI Studio uses `trans_to_mni` as its general voxel-to-world
affine field for both images and tracts.

Evidence from DSI Studio source:

- NIfTI images are loaded with `vs >> trans_to_mni >> image` in
  [DSI-Studio/libs/tracking/fib_data.cpp](../../DSI-Studio/libs/tracking/fib_data.cpp)
- NIfTI images and tract density maps are written with
  `vs << trans_to_mni << is_mni << image` in
  [DSI-Studio/libs/tracking/fib_data.cpp](../../DSI-Studio/libs/tracking/fib_data.cpp)
  and [DSI-Studio/libs/tracking/tract_model.cpp](../../DSI-Studio/libs/tracking/tract_model.cpp)
- TinyTrack `.tt.gz` stores `trans_to_mni` explicitly in
  [DSI-Studio/libs/tracking/tract_model.cpp](../../DSI-Studio/libs/tracking/tract_model.cpp)
- TrackVis `.trk` uses the same matrix as `vox_to_ras` in
  [DSI-Studio/libs/tracking/tract_model.cpp](../../DSI-Studio/libs/tracking/tract_model.cpp)
- `.tck` export applies `trans_to_mni` directly to streamline voxel
  coordinates before writing world-space points in
  [DSI-Studio/libs/tracking/tract_model.cpp](../../DSI-Studio/libs/tracking/tract_model.cpp)

Two important caveats apply:

- `initial_LPS_nifti_srow()` is **only a fallback** synthesized when `trans`
  is absent from the fib file (fib_data.cpp:752-754). The name is misleading:
  the matrix it constructs numerically matches the usual voxel-to-RAS+ affine
  sign pattern (`[-x, -y, +z]` with positive x/y offsets for LPS+-indexed
  voxel arrays). It is not evidence that stored `trans` matrices use an
  LPS-world output convention. DSI Studio reads and uses whatever 4×4 affine is
  stored in `trans`. The function is also used when constructing new
  `fib_data` objects during reconstruction from raw diffusion data, and in
  `manual_alignment.cpp` for initializing alignment.
- `apply_trans()` / `apply_inverse_trans()` in fib_data.cpp only use the
  diagonal scale and translation terms (elements [0], [5], [10], [3], [7],
  [11]). These helpers are used narrowly in atlas-registration paths
  (`sub2mni` / `mni2sub`) and MNI warp initialisation. Off-diagonal elements
  are silently dropped, so oblique affines are approximated there. All other
  coordinate operations — tractography loading/saving, NIfTI I/O,
  inter-space transforms — use the full 4×4 via `tipl::from_space(...).to(...)`.

`odx-rs` therefore keeps two ideas separate:

- voxel arrays are reordered into DSI Studio's required LPS+ index space
- `trans` is stored and read back as the direct voxel-to-RAS+ affine for that
  reoriented voxel space

Only if `trans` is absent does the loader fall back to an external reference
affine.

## DSI Studio → ODX → MRtrix

1. Load DSI Studio with [src/formats/dsistudio.rs](../src/formats/dsistudio.rs).
2. If `trans` is present, use it directly as the voxel-to-RAS affine for the
   DSI voxel ordering.
3. Convert DSI LPS sphere/directions into internal RAS.
4. Preserve sparse peaks directly as ODX fixels:
   - `faN` → `dpf/amplitude`
   - `indexN` + sphere → `directions`
5. If dense `odfN` chunks exist, treat them as hemisphere amplitudes, not full-sphere amplitudes.
6. If MRtrix SH output is requested, fit SH from those hemisphere amplitudes using the shared MRtrix-authoritative `amp2sh` implementation.
7. Export MRtrix fixels using the canonicalized sparse fixel representation.
8. Export MRtrix SH only if dense ODFs were present; peaks alone are not upgraded into SH in this pass.

This matches the existing real-world qsirecon workflow where DSI `odf8` hemisphere amplitudes are converted through `amp2sh`, but `odx-rs` now performs the fitting natively.

## MRtrix → ODX → DSI Studio

1. Load MRtrix fixels and optional SH image with [src/formats/mrtrix.rs](../src/formats/mrtrix.rs).
2. Canonicalize the fixel directory through `index`.
3. If SH is present and dense DSI export is requested, sample SH onto the built-in DSI `odf8` hemisphere directions using the shared `sh2amp` implementation with nonnegative clamping.
4. Always install the built-in DSI `odf8` full sphere before writing DSI output.
5. By default, export DSI `faN/indexN` from fixels, not from sampled ODF peaks.
6. If `PeakSource::SampledOdf` is explicitly chosen, derive peaks from the sampled dense ODF instead.
7. If no SH file is present, write a peaks-only DSI dataset. This is intentionally supported because peaks-only DSI files are common and useful even without dense ODF visualization data.

During the write path, voxel arrays are still reoriented into LPS+ index order
before Fortran flattening. The affine written into `trans` is the corresponding
reoriented voxel-to-RAS+ affine, not a second RAS→LPS world-space conversion.

## Hemisphere vs Full Sphere

DSI Studio stores:

- full sphere geometry in `odf_vertices` / `odf_faces`
- hemisphere amplitudes in `odfN`

This is reflected in working qsirecon conversion code:

- [qsirecon/qsirecon/interfaces/converters.py](../../qsirecon/qsirecon/interfaces/converters.py)

and in `odx-rs` fixture behavior, where the DSI sphere has 642 vertices but the ODF matrices have 321 columns.

`odx-rs` therefore records:

- `SPHERE_ID = "dsistudio_odf8"` when the built-in sphere is used
- `ODF_SAMPLE_DOMAIN = "hemisphere"` when dense DSI ODF payloads are present

## `z0`

`z0` is not treated as required for correctness.

Why:

- DSI Studio’s own `.fz` save path skips `z0` in many conversion flows:
  [DSI-Studio/libs/tracking/fib_data.cpp](../../DSI-Studio/libs/tracking/fib_data.cpp)
- DSI Studio reconstruction code scales `z0` from the QA / `fa0` path rather than obviously from raw dense ODF maxima:
  [DSI-Studio/libs/dsi/odf_process.hpp](../../DSI-Studio/libs/dsi/odf_process.hpp)

As a result:

- `.fz` exports omit `z0` by default
- `.fib.gz` exports may write compatibility `z0 = 1 / max(fa0)` when available
- `z0` is treated as optional display-scaling metadata, not a field that defines the semantic content of the dense ODF data
