# odx — Python bindings for the ODX format

Python bindings for [`odx-rs`](https://github.com/PennLINC/odx-rs), a TRX-style
mmap-friendly container for orientation density functions (ODFs), peaks, and
spherical-harmonic coefficients from diffusion MRI.

## Install

```bash
pip install odx           # core package (numpy only)
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
  for zero-copy NumPy views.
- **Peak finder** — sphere-mesh local-maxima detection with MRtrix-style
  Gauss-Newton sub-vertex refinement on the SH series itself.
- **Foreign format converters** — load and save DSI Studio (`.fz`, `.fib.gz`),
  MRtrix (fixel directories + `.mif`), Tortoise MAP-MRI, pyAFQ asymmetric ODFs.
- **dipy adapter** (optional) — bidirectional conversion between
  `odx.Odx` and dipy's `PeaksAndMetrics`. Lazy-imported so dipy isn't required
  for the core package.
- **SH basis conversion** — descoteaux07 ↔ tournier07 round-trips via
  amplitudes, including legacy/modern variants.

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
