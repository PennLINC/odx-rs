# odx-rs

`odx-rs` is a Rust library and CLI for working with ODX datasets and converting between ODX, DSI Studio, and MRtrix representations.

## CLI

The repository ships an `odx` binary for inspection, conversion, and validation.

Examples:

```bash
odx info input.fib.gz
odx convert input.fib.gz output.odx --odx-layout directory
odx convert input.fib.gz out_fixels --out-sh fod.mif.gz
odx convert fixels_mif output.fz --sh fod.mif.gz
odx validate output.odx
```

Some formats are composite:

- MRtrix fixel directories may be paired with `--sh <path>`
- MRtrix SH images may be paired with `--fixel-dir <path>`
- DSI Studio `fib.gz` files may use `--reference-affine <mif-or-nifti>`
