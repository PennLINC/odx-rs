# odx — Python bindings for the ODX format

Python bindings for [`odx-rs`](https://github.com/PennLINC/odx-rs), a TRX-style
mmap-friendly container for orientation density functions (ODFs), peaks, and
spherical-harmonic coefficients from diffusion MRI.

## Install

```bash
pip install odx           # core package (numpy + nibabel)
pip install odx[dipy]     # with the dipy adapter for PeaksAndMetrics interop
```

## Quickstart

Compute CSD in dipy, dump to a peaked ODX in one call:

```python
import odx
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.direction.peaks import peaks_from_model

# ... fit a model, get csd_fit.shm_coeff (X, Y, Z, K)

peaked = odx.from_sh_coefficients(
    csd_fit.shm_coeff, mask=mask, affine=affine,
    basis="descoteaux07", sh_order=8, legacy=True,
)
peaked.save("csd.odx")              # native, mmap-friendly
peaked.save_fz("autotrack.fz")      # → DSI Studio fz
peaked.save_mrtrix("mrtrix_out/")   # → MRtrix mif (auto-converts basis)

# Round-trip back to dipy if you want
pam = odx.to_peaks_and_metrics(peaked)
```

## Features

- **Native ODX I/O** — read/write `.odx` archives or directories, mmap-backed
  for zero-copy NumPy views. `odx.load(path, skip_odf=True, skip_sh=True)`
  (and `skip_odf=` on `from_fibgz` / `from_fz`) never reads dense arrays you
  don't need.
- **Peak finder** — sphere-mesh local-maxima detection with MRtrix-style
  Gauss-Newton sub-vertex refinement on the SH series itself.
- **Foreign format converters** — load and save DSI Studio (`.fz`, `.fib.gz`),
  MRtrix (fixel directories + `.mif`), Tortoise MAP-MRI, pyAFQ asymmetric ODFs.
- **nibabel adapter** — DPVs to/from `Nifti1Image`; nibabel is a required
  dependency so NIfTI orientation and affines are always handled the same way.
- **dipy adapter** (optional) — bidirectional conversion between
  `odx.Odx` and dipy's `PeaksAndMetrics`. Lazy-imported so dipy isn't required
  for the core package.
- **Coherence QC** — `ds.coherence()` gives current DSI Studio's fib-QC
  coherence index (or `"fixel"` mode's per-fixel summaries);
  `check_btable=True` scores all 24 gradient-table permutations/flips without
  refitting.
- **SH basis conversion** — descoteaux07 ↔ tournier07 round-trips via
  amplitudes, including legacy/modern variants.
- **Gradient-nonlinearity correction** — `odx.apply_graddev` (or
  `Odx.apply_graddev`) reorients SH/FODs, dense ODFs and fixels with a
  per-voxel deviation field, as the `odx graddev` CLI does:

  ```python
  img = nib.load("sub-01_space-ACPC_graddev.nii.gz")
  corrected, report = ds.apply_graddev(img)          # brings its own affine
  # also accepted: a path, or an (X,Y,Z,9) array with affine=
  ```

  Never reorient the field first (e.g. `nib.as_closest_canonical`): its
  components live in its own voxel axes.

  The field holds row-major `T` per voxel with `g_eff = Tᵀ g` (qsiprep /
  TORTOISE, `identity="included"`; HCP/FSL `grad_dev` stores `T − I`,
  `identity="absent"`; `"auto"` decides). This is a post-hoc correction of
  fitted data: it fixes orientation, not the per-voxel b-value change. On a
  Prisma-class whole-body coil the orientation effect is a fraction of a
  degree.

## License

The bulk of `odx-rs` (and the Python bindings) is dual-licensed under the
[MIT](https://github.com/PennLINC/odx-rs/blob/main/LICENSE-MIT) and
[Apache 2.0](https://github.com/PennLINC/odx-rs/blob/main/LICENSE-APACHE)
licenses, the typical Rust ecosystem dual-license. Pick whichever you prefer.

Two source files in the underlying Rust crate (`src/peak_finder.rs` and
`src/mrtrix_sh.rs`) port code from
[MRtrix3](https://github.com/MRtrix3/mrtrix3) and remain governed by the
[Mozilla Public License v. 2.0](https://github.com/PennLINC/odx-rs/blob/main/LICENSE-MRTRIX).
That's a file-scoped (weak) copyleft — modifications to those specific files
must remain MPL and source-available, but the license has no effect on
projects that simply *use* `odx` as a dependency.

SPDX summary: `(MIT OR Apache-2.0) AND MPL-2.0`.
