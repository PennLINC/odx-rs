# odx-rs

`odx-rs` is a Rust library and CLI for working with ODX datasets and converting between ODX, DSI Studio, and MRtrix representations.

## CLI

The repository ships an `odx` binary for inspection, conversion, and validation.

Examples:

```bash
odx info input.fib.gz
odx convert input.fib.gz output.odx --output-format odx-directory
odx convert input.fib.gz out_fixels --out-sh fod.mif.gz
odx convert fixels_mif output.fz --sh fod.mif.gz
odx validate output.odx
```

Some formats are composite:

- MRtrix fixel directories may be paired with `--sh <path>`
- MRtrix SH images may be paired with `--fixel-dir <path>`
- DSI Studio `fib.gz` files may use `--reference-affine <mif-or-nifti>`

## Applying ANTs transforms to ODX

`odx transform` warps an ODX dataset (SH coefficients, per-voxel scalars, and
fixels) onto a new spatial grid using an ANTs/ITK Composite `.h5`. See
[`docs/cli.md`](docs/cli.md#odx-transform) for the full reference.

### The "same-direction h5" rule (cartoon BIDS)

Unlike `trxrs` and `giftirs` (which warp **points** and follow ANTs'
opposite-named convention), `odx transform` resamples a sampled grid — and
grid resampling is pull-based, just like `antsApplyTransforms` for an
image. So the h5 you pass is the **same-direction** file as for image
warping. With paired BIDS h5 files for subject `sub-01`:

| You have                                       | You want                              | Pass to `--transform`                            |
| ---------------------------------------------- | ------------------------------------- | ------------------------------------------------ |
| `sub-01_space-ACPC_dwimap.odx`                 | ODX in `MNI152NLin2009cAsym`          | `sub-01_from-ACPC_to-MNI152NLin2009cAsym_xfm.h5` |
| `sub-01_space-MNI152NLin2009cAsym_dwimap.odx`  | ODX in `ACPC`                         | `sub-01_from-MNI152NLin2009cAsym_to-ACPC_xfm.h5` |

> **Heads-up:** for the *same subject*, the h5 you pass to `odx` is the
> **other** member of the BIDS pair than the one you'd pass to `trxrs`/`giftirs`.

```text
sub-01_space-ACPC_tracts.trx       → MNI:  trxrs       --transform sub-01_from-MNI...to-ACPC_xfm.h5
sub-01_space-ACPC_dwimap.odx       → MNI:  odx transform --transform sub-01_from-ACPC_to-MNI..._xfm.h5
                                                                       ^^^^^^ opposite naming
```

### Worked example (warp ACPC ODX into MNI, mrtrix mode)

```bash
odx transform \
    sub-01_space-ACPC_dwimap.odx \
    sub-01_space-MNI152NLin2009cAsym_dwimap.odx \
    --transform sub-01_from-ACPC_to-MNI152NLin2009cAsym_xfm.h5
```

### Two modes: `mrtrix` (default) vs `ants`

- `--mode mrtrix` (default): pull SH, DPV, and fixels via the single forward
  h5. Matches `mrtransform` + `fixeltransform` semantics. Simple; fixels may
  be duplicated or dropped at non-uniform warp regions.
- `--mode ants`: pull SH/DPV via `--transform`, **push** fixels via
  `--transform-inverse` (the paired h5). Each source fixel maps to exactly
  one target voxel, preserving cardinality.

```bash
odx transform \
    sub-01_space-ACPC_dwimap.odx \
    sub-01_space-MNI152NLin2009cAsym_dwimap.odx \
    --mode ants \
    --transform         sub-01_from-ACPC_to-MNI152NLin2009cAsym_xfm.h5 \
    --transform-inverse sub-01_from-MNI152NLin2009cAsym_to-ACPC_xfm.h5
```

For affine-only transforms, supply a `--reference` NIfTI in the target
space (the affine has no embedded grid for `odx` to resample onto).
