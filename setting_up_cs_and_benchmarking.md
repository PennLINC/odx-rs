# Setting up cs-dmri benchmarking against an MRtrix gold standard

This document captures the architecture and the three nontrivial gotchas we hit
while wiring up `cs-dmri/scripts/gold_standard.sh` so future-us doesn't redebug
the same problems.

## What the harness does

For each (HASC bundle, ABCD bundle) pair × cs-odf config, the script produces:

1. **`abcd_gold/gold.odx`** — qsirecon's `mrtrix_multishell_msmt_noACT` recon
   run on the real ABCD DWI:
   `mrconvert → dwi2response dhollander → dwi2fod msmt_csd → mtnormalise →
   fod2fixel → odx convert`.
2. **`predicted_gold/<config>/gold.odx`** — the same recon, but on the
   cs-predicted ABCD signal (cs-fit on HASC, cs-synth onto the ABCD gradient
   table).
3. Two `odx compare` runs per cell:
   - `cs_direct_vs_gold` — cs-odf `coeffs.odx` vs `abcd_gold/gold.odx`
     ("do cs's HASC-fit fixels match what MRtrix produces from real ABCD?").
   - `cs_roundtrip_vs_gold` — `predicted_gold/.../gold.odx` vs
     `abcd_gold/gold.odx` ("does cs-predict + MRtrix recover gold?").

Run it:

```bash
bash cs-dmri/scripts/gold_standard.sh \
  --bundles-root /Users/mcieslak/data/csdsi/qsiprep \
  --bench-root ~/cs-bench-csdsi/focused
```

Defaults to all paired bundles × 5 focused configs. Optional flags:
`--filter-bundles <substr>`, `--configs <csv>`, `--force`, `--threads N`.

Prerequisites:
- MRtrix3 on PATH (`dwi2fod`, `fod2fixel`, `mtnormalise`, `mrconvert`,
  `dwi2response`).
- Each DWI **must** have a sibling `.b` file (MRtrix grad). The script fails
  loudly if it's missing rather than synthesizing one from `bval/bvec`. FSL
  bvec interpretation can silently rotate the gradient frame and corrupt
  the FOD; the `.b` is in scanner/world coords with no conversion ambiguity.
- `coeffs.odx` for the relevant pairs must already exist under `--bench-root`
  (produced by `bench_fit_options.py`).

---

## Three gotchas, in order

### 1. NIfTI affine canonicalization clobbers the qform

**Symptom.** `odx compare` rejected the (`coeffs.odx`, `gold.odx`) pair with
`ODX affines differ`, even though both were derived from the same ABCD volume.
One side carried `[-1.7, 0, 0, 95.9; …]`, the other `[+1.7, 0, 0, -97.9; …]`.

**Root cause.** The MRtrix backend in `odx-rs` was unconditionally calling
`canonicalize_spatial_axes_to_ras_f32` on every NIfTI input — reordering
voxels and rewriting the affine to a positive-diagonal RAS+ form. cs-odf
preserves the on-disk qform exactly (nibabel default). For this DWI:
`qform_code=1, sform_code=0`, qform =
`[-1.7, 0, 0, 95.9; 0, -1.7, 0, 102.25; 0, 0, 1.7, -85.75]`. nibabel returns
this verbatim; cs-odf wrote it into `coeffs.odx`; odx-rs was overriding it.

**Why canonicalizing NIfTI is wrong.** NIfTI affines already encode the
voxel→RAS+ mapping. The qform IS the spatial truth — that's the whole point of
the field. Canonicalizing to positive diagonals just invents a different
coordinate frame. (Contrast MIF: MRtrix uses on-disk strides as a separate
canonical view, so MIF inputs *do* need canonicalization. NIfTI doesn't.)

This matches TRX's affine policy: prefer qform, SVD-fit sform→qform when only
sform is set, and **never reorder data** as a side effect. Only the affine is
recomputed.

**Fix.** Added `--preserve-affine` to `odx convert`. With it, NIfTI reads skip
canonicalization and keep the on-disk affine + voxel order. MIF reads still
canonicalize (strides need it).

Touched: `odx-rs/src/bin/odx.rs`, `odx-rs/src/cli_support.rs`,
`odx-rs/src/formats/mrtrix.rs` (`MrtrixDatasetLoadOptions::preserve_nifti_affine`).

**Open follow-up.** `odx-rs/src/reference_affine.rs:58` prefers sform over
qform — the opposite of the TRX policy. Harmless for this DWI (sform_code=0
forces the qform path) but should be flipped for files with both codes set.

### 2. DPF naming mismatch between cs-odf and MRtrix

**Symptom.** `odx compare --primary-dpf afd` errored:
`requested primary DPF 'afd' does not exist`. Auto-detect (chain
`amplitude → afd → qa`) couldn't help either — A would resolve `qa`, B would
resolve `afd`, and the comparator forces both sides to one shared name.

**Root cause.** Same physical metric (per-fixel peak amplitude), different
names per pipeline:
- cs-odf wrote `peak_amplitude_raw`.
- MRtrix's `fod2fixel` writes `peak_amp.{mif,nii.gz}`.
- The ecosystem (and odx-rs's own tests) wants `amplitude`.

**Fix.** Normalize both sides to the canonical `amplitude`:
- `cs-dmri/src/bin/cs-odf.rs`: `set_dpf_data("amplitude", …)` instead of
  `peak_amplitude_raw`.
- `odx-rs/src/formats/mrtrix.rs`: `canonical_fixel_dpf_name("peak_amp")
  → "amplitude"` on the read side. The inverse
  `mrtrix_fixel_dpf_filename_stem("amplitude") → "peak_amp"` keeps the
  write/round-trip clean so MRtrix tooling still recognizes the file.

`odx compare`'s auto-detect now picks `amplitude` on both sides without
needing `--primary-dpf` at all.

### 3. Scrambled fixel positions from `mrconvert -strides $dwi`

**Symptom (the user spotted it visually).** `gold.odx`'s ODF glyphs looked
geometrically correct, but per-voxel fixel directions were scrambled across
the brain. Comparator numbers told the same story: gold-side
`coherence_index = 0.04` (essentially random),
`mean_match_angle = 18.8°` direct / 18.9° round-trip, only 142k mutual matches.

**Root cause.** After fix #1, the SH NIfTI (`wmfod_norm.nii.gz`) was being
written with `mrconvert -strides "$dwi"` so it would inherit the DWI's
on-disk stride layout, matching `coeffs.odx` under `--preserve-affine`. To
keep the *fixel* files in the same frame, the original script extended the
same `mrconvert -strides "$dwi"` to every fixel file:

```bash
# Wrong — applies DWI's first 3 strides to (n_fixels, 3, 1) axes.
for f in index directions afd peak_amp dispersion; do
    mrconvert -strides "$dwi" "$f.mif" "$f.nii.gz"
done
```

But `directions.mif`, `afd.mif`, `peak_amp.mif`, `dispersion.mif` have shape
`(n_fixels, 3, 1)` or `(n_fixels, 1, 1)` — those axes are **not spatial**.
`mrconvert -strides $dwi` happily inherits the DWI's first three strides onto
them. If the DWI has any negative stride (a perfectly normal MRtrix
"preferred" layout, e.g. `[-1, -2, 3, 4]`):

- the n_fixels axis gets flipped — global fixel row order is reversed,
- the 3-component axis gets flipped — every direction vector becomes negated.

Meanwhile `index.nii.gz`'s `first_index` values still encode the *original*
row positions, so each voxel pulls its directions from the wrong rows of
`directions.nii.gz`. Voxels in early scan order get directions that belong to
voxels in late scan order — the "scrambled" pattern.

**Fix.** Only convert the genuinely-spatial fixel file (`index.mif`, shape
`(x,y,z,2)`) to NIfTI. Leave `directions.mif` / `afd.mif` / `peak_amp.mif` /
`dispersion.mif` as MIF; odx-rs's MIF loader canonicalizes to its own RAS+
view, but for non-spatial fixel files (identity-ish affine) that's a no-op,
so the global row order is preserved. `index`'s `first_index` values now line
up with the rows of `directions` again.

**Result.**

| Metric | Before fix | After fix |
|---|---|---|
| cs_direct_vs_gold mean angle | 18.8° | **13.4°** |
| cs_direct_vs_gold mutual matches | 142k | **292k** (+105%) |
| cs_roundtrip_vs_gold mean angle | 18.9° | **12.1°** |
| cs_roundtrip_vs_gold mutual matches | 84k | **270k** (+222%) |
| gold `coherence_index` | 0.04 | **0.27** |

Gold's coherence index jumping from 0.04 (random) to 0.27 (clean-WM typical)
is the smoking gun: fixels now form coherent local neighborhoods.

---

## TL;DR architecture

- cs-odf produces `coeffs.odx` with the on-disk qform, `amplitude` DPF.
- The harness runs MRtrix MSMT-CSD on real (and cs-predicted) ABCD DWI, then
  converts SH + fixels into a single `gold.odx` carrying the same on-disk
  qform via `--preserve-affine`, with `peak_amp` rewritten to `amplitude`.
- **Only `index.mif` is converted to NIfTI to share the DWI's strides; the
  non-spatial fixel payloads stay as MIF.** This is non-obvious but critical.
- `odx compare`'s auto-detect chain (`amplitude → afd → qa`) picks `amplitude`
  on both sides — no `--primary-dpf` needed.
- One `cs_direct_vs_gold` and one `cs_roundtrip_vs_gold` per pair × config.

## Out of scope (v1)

- Tractography on gold FODs — the fixel-level comparison is the headline.
- 5TT/ACT — qsirecon's `noACT` recipe doesn't need it; T1 isn't always
  available.
- Single-shell SS3T fallback for HASC — fail loudly rather than silently
  switching pipelines.
- Aggregate CSV/plotting on top of the per-cell `comparison.odx` — small
  follow-up that mirrors `bench_plots.R`.
