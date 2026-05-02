# ODX Next Steps

## Current Priority

The Rust core now covers DSI Studio `.fz` / `.fib.gz`, MRtrix SH + fixel,
Dipy PAM5, and Tortoise MAP-MRI, plus fixel-level QC. Remaining work
focuses on performance (MAT v4 / fib.gz hot paths), Python bindings,
and runtime dtype dispatch.

## Deferred: Python Bindings (`python/`)

Still intentionally deferred until the Rust-side performance work stabilizes:

- [ ] Scaffold `python/` workspace member with maturin + PyO3
- [ ] `OdxDataset` Python class wrapping `AnyOdxDataset` (runtime dtype dispatch)
- [ ] Zero-copy NumPy views for `mask`, `offsets`, `directions`, `sphere_vertices`, `sphere_faces`
- [ ] Typed NumPy views for `odf/`, `sh/`, `dpv/`, `dpf/` arrays (via `get_odf()`, `get_sh()`, etc.)
- [ ] Property accessors: `nb_voxels`, `nb_peaks`, `dtype`, `header` dict
- [ ] Key listing: `odf_keys()`, `sh_keys()`, `dpv_keys()`, `dpf_keys()`, `group_keys()`
- [ ] `load(path)` module-level function
- [ ] Read-only NumPy arrays backed by mmap (owner lifecycle via `PyClassRef`)

Reference: `trx-rs/python/src/lib.rs` — binding pattern to revisit later

## Runtime Dtype Dispatch (`AnyOdxFile`)

trx-rs has `AnyTrxFile` for loading files without knowing the dtype at compile
time. ODX still needs the same:

- [ ] `AnyOdxFile` enum over `OdxFile<f16>`, `OdxFile<f32>`, `OdxFile<f64>`
- [ ] `AnyOdxFile::load(path)` with auto-detection from `directions` filename
- [ ] `with_typed()` visitor pattern for generic code

Reference: `trx-rs/src/any_trx_file.rs` — runtime dtype dispatch pattern

## Format Converters

### dsistudio fib.gz → ODX (done)

Implemented in `src/formats/dsistudio.rs`. Handles:
- [x] MAT v4 catalog reader (`src/formats/mat4.rs`)
- [x] Build mask from `fa0 > 0`
- [x] Read `odf_vertices` and `odf_faces`, flip x/y for LPS→RAS
- [x] Concatenate `odf0..odfN` chunks, transpose to (voxels, ndirs)
- [x] Convert `index0..indexN` + `fa0..faN` to 3D directions via sphere vertices
- [x] Map per-voxel scalars (gfa, dti_fa, md, etc.) to `dpv/`
- [x] Map per-fixel scalars (icvf, isovf, od) to `dpf/`
- [x] Handle `z0` normalization factor
- [x] Fortran→C order reindexing

References:
- `qsirecon/qsirecon/interfaces/converters.py:618-647` — `fast_load_fibgz()`: how fib.gz is loaded (gzip → MAT v4)
- `qsirecon/qsirecon/interfaces/converters.py:337-435` — `amplitudes_to_fibgz()`: fib matrix layout, ODF chunking (ODF_COLS=20000), Fortran order
- `qsirecon/qsirecon/interfaces/converters.py:657-700` — `fib2amps()`: reading ODFs back, mask from `fa0 > 0`, odf chunk reassembly
- `qsirecon/qsirecon/interfaces/converters.py:37` — `ODF_COLS = 20000` chunk size constant
- `trx-rs/src/formats/tt/mat.rs` — existing MAT v4 reader in trx-rs (used as reference for our mat4.rs)

### ODX → dsistudio fib.gz (done)

Implemented in `save_fibgz()`. Handles:
- [x] Reconstruct `fa0..faN`, `index0..indexN` from directions + dpf/amplitude
- [x] Convert sphere vertices RAS→LPS
- [x] ODF chunks in 20000-voxel splits, column-major
- [x] C→Fortran order reindexing
- [x] Preserve DSI Studio ODFs when round-tripping `fib.gz → .fz` without recomputation

### dsistudio .fz format (core done)

The .fz format is the same MAT v4 + gzip as fib.gz, but with additional
optimizations from DSI Studio (post-v202504). Implemented in
`src/formats/dsistudio.rs::{load_fz, save_fz}` and `src/formats/mat4.rs`.
Handles:

- [x] `mat4.rs` preserves raw storage mode per matrix (`Regular`, `SlopedU8`,
      `Masked`, `MaskedSlopedU8`) and materializes lazily
- [x] **Sparse/masked storage**: `si2vi` reconstruction for masked arrays
      without expanding to full-volume dense vectors first
- [x] **Slope+intercept compression**: decode `{name}.slope` /
      `{name}.intercept` and expose `original = value * slope + intercept`
- [x] **Trans matrix**: read `trans` as the authoritative voxel→RAS+ affine
      when present; fall back only when absent
- [x] `.fz` round-trip tests against real DSI-generated sub-20124 fixture
      (`tests/fibgz_round_trip.rs`)

Open ODX design questions:

- [ ] Should ODX preserve sparse voxel ordering metadata when importing `.fz`,
      or always normalize to dense `mask + offsets` on write?
- [ ] Should ODX add optional linear-quantized arrays
      (`quant_scale`/`quant_offset` metadata or filename-level encoding) for
      large float matrices such as ODFs and SH coefficients?
- [ ] Should ODX support masked-on-disk arrays for `dpv/`, `odf/`, or `sh/`
      beyond the existing global `mask`, or is that complexity better left to
      converters and archive-level compression?
- [ ] Benchmark whether gzip-level savings from `.fz`-style quantization remain
      meaningful once ODX arrays are already zipped as `.odx`.

References:
- `DSI-Studio/tipl/io/mat.hpp:772-853` — sparse/masked storage format, `si2vi` mapping, storage type enum
- `DSI-Studio/tipl/io/mat.hpp:794-823` — slope+intercept compression
- `DSI-Studio/libs/tracking/fib_data.cpp` — .fz loading, `trans` matrix handling

### fib.gz / .fz loading performance

A criterion bench harness exists at `benches/dsistudio_io.rs`; remaining
optimizations we could still adopt from DSI Studio:

- [ ] Profile the current Rust path with a real HCP-scale `.fib.gz` and record
      time spent in: gzip inflate, MAT record parsing, `Vec<u8>` copies,
      `as_f32_vec()` conversions, ODF chunk concatenation, and `save_fibgz()`
      sphere vertex lookup.
- [ ] Refactor `Mat4Array` accessors to avoid repeated full allocations inside
      `dsistudio.rs`; today `fa{n}`, `index{n}`, scalar maps, and ODF chunks are
      repeatedly converted via `as_f32_vec()` / `as_i32_vec()`.
- [ ] Add typed borrowed views over MAT payloads (`&[f32]`, `&[i16]`, etc.)
      for native storage modes before adding any more format features.
- [ ] **Selective matrix loading**: API to load only specific matrices by name,
      skipping the rest.
- [ ] **Lazy/delayed loading**: skip matrices above a configurable threshold
      during initial parse and materialize them only on demand.
- [ ] **Access point indexing (.idx files)**: DSI Studio creates `.fib.gz.idx`
      files with gzip access points every ~8MB, enabling faster seeking and
      parallel decompression. Investigate generating and consuming these.
- [ ] **Multi-threaded decompression**: with access points, decompress separate
      regions of large files in parallel using independent inflate states.
- [ ] Speed up `save_fibgz()` peak encoding by replacing the current
      O(`NB_PEAKS * NB_SPHERE_VERTICES`) nearest-vertex search with either
      cached index reuse, a KD-tree, or a DSI-compatible direct index path when
      directions already originated from a known sphere.
- [ ] Consider a lower-level writer that streams MAT records directly into the
      gzip encoder instead of first building large temporary matrices.

References:
- `DSI-Studio/tipl/io/mat.hpp:335-339` — delayed read for matrices >16MB (`delay_read` mode)
- `DSI-Studio/tipl/io/gz_stream.hpp:287-317` — `.idx` access point file creation (SPAN=8MB)
- `DSI-Studio/tipl/io/gz_stream.hpp:377-425` — multi-threaded decompression using access points
- `DSI-Studio/tipl/io/gz_stream.hpp:255-473` — `prepare_idx()`: index sampling for files ≥128MB
- `DSI-Studio/libs/dsi/image_model.cpp:427-450` — `save_idx()` call during fib creation

### Immediate Rust Refactors

These are worth doing because they reduce the same bottlenecks
`.fz` work builds on top of:

- [ ] Split `mat4.rs` into record parsing, typed views, and gzip transport so
      format-specific features do not stay coupled to "read whole file into `Vec<u8>`".
- [ ] Preserve original MAT numeric types (`f32`, `f64`, `i16`, `u8`, etc.)
      through parsing rather than normalizing them eagerly at call sites.
- [ ] Rework `fibgz_to_odx()` so peak and scalar extraction convert each source
      matrix at most once.
- [ ] Rework ODF chunk handling to avoid concatenating every `odfN` chunk into
      one large temporary buffer before reordering.
- [x] Add targeted microbenchmarks for `load_fibgz()` and `save_fibgz()` so
      optimization work has a measurable baseline (`benches/dsistudio_io.rs`).

### MRtrix SH + Fixel I/O (done)

Implemented in `src/formats/mif.rs` and `src/formats/mrtrix.rs`. Handles:
- [x] Parse text header (key: value pairs until END marker)
- [x] dim, vox, datatype, layout, transform, file offset
- [x] .mif.gz via flate2
- [x] Stride computation from layout field
- [x] Affine extraction
- [x] Preserve `-0` in MIF layout parsing
- [x] Load MRtrix SH `.mif` / `.mif.gz` into `sh/coefficients`
- [x] Set `SH_BASIS: "tournier07"` and infer `SH_ORDER`
- [x] Load MRtrix fixel directories from MIF or NIfTI containers into
      `mask + offsets + directions + dpf/*`
- [x] Combine SH image and fixel directory into one `OdxDataset`
- [x] Treat fixel data filenames by literal stem only (`afd`, `disp`, etc. are not special-cased)
- [x] Canonicalize fixel ordering through `index[...,0:2]`, not raw file row order
- [x] Handle MRtrix NIfTI negative-stride conventions for real
      `index`, `directions`, `afd`, and `disp` fixtures
- [x] Export SH to `.mif(.gz)` and NIfTI-1 / NIfTI-2
- [x] Export fixel directories to MIF and NIfTI and round-trip them through ODX
- [x] Compute `anisotropic_power` dpv from SH on load (masked and unmasked paths)

Still needed:
- [ ] Detect SH basis from MIF `command_history` or user specification instead of
      assuming MRtrix `"tournier07"`
- [ ] Add MRtrix benchmarks for SH load, fixel load, combined load, and export
- [ ] Decide whether MRtrix NIfTI export should support NIfTI-1 fixel directories
      in addition to the current NIfTI-2 default

References:
- `trx-mrtrix2/cpp/core/formats/mrtrix_utils.cpp` — MIF header parser
- `trx-mrtrix2/cpp/core/formats/mrtrix_utils.cpp:27-90` — `parse_axes()`: layout string parsing
- `trx-mrtrix2/cpp/core/formats/mrtrix_utils.cpp:125-157` — `get_mrtrix_file_path()`
- `trx-mrtrix2/cpp/core/formats/mrtrix_gz.cpp:27-60` — `MRtrix_GZ::read()`: .mif.gz loading
- `trx-mrtrix2/cpp/core/image_io/gz.cpp:27-68` — `ImageIO::GZ::load()`
- `trx-mrtrix2/cpp/core/stride.h` — stride concept
- `trx-mrtrix2/cpp/core/datatype.h`, `datatype.cpp:78-175` — supported MIF datatypes
- `trx-mrtrix2/cpp/core/file/key_value.cpp` — `KeyValue::Reader`

### Dipy PAM → ODX (done, feature-gated)

Implemented in `src/formats/pam.rs` behind the `pam5` cargo feature
(requires `hdf5-metno`). Provides `load_pam5()` and `save_pam5()`:

- [x] Read PAM5 HDF5 files (v0.0.1)
- [x] Extract `peak_dirs` (x,y,z,npeaks,3), flatten variable peaks with offsets
- [x] `peak_values` → `dpf/amplitude`
- [x] `shm_coeff` → `sh/coefficients` (basis assumed / recorded via
      `_ODX_PAM_SH_BASIS_ASSUMED`)
- [x] `gfa` → `dpv/gfa`
- [x] Round-trip tests in `tests/pam_io.rs`

Still needed:
- [ ] Resolve the PAM5 SH basis convention properly instead of relying on the
      `_ODX_PAM_SH_BASIS_ASSUMED` marker

References:
- `qsirecon/qsirecon/cli/recon_plot.py` — `peaks_from_odfs()`
- `qsirecon/qsirecon/interfaces/dipy.py` — Dipy interface wrappers
- `qsirecon/qsirecon/utils/shm.py` — SH basis functions

### Tortoise MAP-MRI → ODX (done)

Implemented in `src/formats/tortoise_mapmri.rs`. Reads Tortoise MAP-MRI
coefficient + u-vector NIfTI pairs, projects onto the DSI-Studio ODF8
hemisphere sphere, fits MRtrix SH coefficients, and emits:

- [x] `odf/` on `dsistudio_odf8_hemisphere`
- [x] `sh/coefficients` (tournier07, lmax 8 by default)
- [x] `dpv/anisotropic_power`
- [x] Display-scale convention `p99_voxel_max_odf_to_1`
- [x] CLI wiring via `--mapmri-tensor` / `--mapmri-uvec`

## Fixel Coherence QC (done)

Implemented in `src/qc.rs` with CLI subcommand `odx qc`. Supports Otsu /
positive / all / numeric thresholds, primary-peak vs all-fixel Otsu scope,
connected / disconnected classification across 13 neighbor offsets, and
optional `dpf/qc_class.uint8` write-back. Documented in `docs/fixel_qc.md`.

## CLI Tool (`src/bin/odx.rs`)

- [x] `odx info <file>` — print header, array listing, voxel/peak counts
- [x] `odx convert <input> <output>` — convert between fz / fib.gz / fixel / PAM / ODX / Tortoise MAP-MRI
- [x] `odx validate <file>` — check internal consistency
- [x] `odx qc` — fixel coherence QC
- [x] `odx completions <shell>` — shell completion generator

Reference: `trx-rs/src/bin/trx.rs`

## Validation and Consistency Checks (done)

- [x] Verify `mask.nonzero().count() == NB_VOXELS`
- [x] Verify `offsets.len() == NB_VOXELS + 1` and `offsets[NB_VOXELS] == NB_PEAKS`
- [x] Verify `directions` byte length matches `NB_PEAKS * 3 * dtype.size_of()`
- [x] Verify `dpv/` arrays have `NB_VOXELS` rows
- [x] Verify `dpf/` arrays have `NB_PEAKS` rows
- [x] Verify `odf/` arrays have `NB_VOXELS` rows and correct column count
- [x] Verify `sh/` coefficient count matches `(SH_ORDER+1)(SH_ORDER+2)/2`

## Tests

- [x] DSI Studio round-trip: `tests/fibgz_round_trip.rs` (fib.gz ↔ ODX,
      `.fz` ↔ ODX, `.fz`/fib.gz affine parity against real sub-20124 fixture)
- [x] MRtrix parity: load MIF and NIfTI fixel directories with real `afd` /
      `disp` fixtures and confirm canonicalized arrays match
- [x] Round-trip: MRtrix SH MIF/NIfTI1 and fixel MIF/NIfTI through ODX
- [x] PAM5 round-trip (`tests/pam_io.rs`, feature-gated)
- [x] QC integration (`tests/qc.rs`)
- [x] Archive-level mutation / CLI tests
- [ ] Large-file benchmark: memory usage and load time for HCP-scale data
- [ ] Python parity: validate Rust and Python APIs return identical arrays

## Documentation

- [x] README with usage examples (CLI-focused for now)
- [x] `SPECIFICATION.md` (lives at repo root)
- [x] `docs/cli.md`, `docs/fixel_qc.md`,
      `docs/dsistudio_mrtrix_conversion_workflows.md`
- [ ] Publish SPECIFICATION.md as a standalone spec repo (like trx-spec)
