# Building an ODF template from many scans

`odx combine --method mean-fod` builds a group/average ODF template from N
pre-aligned ODX files and, alongside it, the maps that say how much to trust
that template.

This is the N-file analogue of [`odx compare`](compare.md), and the direct
counterpart of DSI Studio's `odf_average` (which writes a `*.mean.odf.fz`) and
of the aggregation step inside MRtrix3's `population_template`.

**Registration is out of scope.** Every input must already be in one space on
one grid. `odx transform` is what puts them there — it warps SH with apodised-PSF
reorientation, so the FODs arrive correctly rotated rather than merely
resampled. This is the same contract `odf_average` imposes (QSDR-space inputs
only).

```bash
odx combine ses-*.odx \
  --method mean-fod --normalize-fod none --min-coverage 0.5 --lmax min \
  --average-dpv gm --average-dpv csf --dpv-sd --loo on \
  --out-odx group.odx --out-dir maps/ --out-report report.json
```

## How the average is formed

The FOD is *linear* in its spherical-harmonic coefficients, so the mean
coefficient vector is exactly the FOD of the mean — no projection error, no
resampling. Averaging therefore happens coefficient-by-coefficient, accumulated
in `f64`.

Three choices are worth stating explicitly, because they are the ones that go
wrong quietly.

### The divisor is the per-voxel contributor count, never N

A voxel covered by 5 of 8 inputs is divided by 5. Dividing by 8 would attenuate
the FOD wherever some subjects fall outside their mask, producing a rim of low
apparent fibre density at the mask boundary — which every downstream group test
would read as a real effect. Both reference implementations agree here:
`mrmath mean` skips non-finite values, and DSI Studio divides by its per-voxel
`odf_count`.

The cost is heteroscedasticity near the edge, which is why `dpv/n_subjects` and
`dpv/coverage_frac` ship with the template and why `--min-coverage` exists.

### `--min-coverage` selects the voxel set

`--min-coverage FRAC` keeps a voxel when at least that fraction of inputs cover
it. `0` behaves as a mask union, `1` as an intersection, and `0.5` is the
recommended template setting. It generalizes and overrides `--mask-combine`,
whose `union` default is retained for backwards compatibility.

Note that DSI Studio's `odf_average` *documents* a ">half the population" rule
but its code (`odf_count[i] > n/2 || !odfs[i].empty()`) keeps any voxel with at
least one contributor. We implement the documented rule.

### `--lmax min` truncates rather than pads

Both SH bases in this crate are band-ordered ascending, so dropping to a lower
lmax is a prefix slice of the coefficient row. When inputs disagree on lmax,
the default truncates everyone to the cohort minimum.

Zero-padding to the maximum (`--lmax max`) is available but warned about: it
asserts that a lower-order subject's FOD is band-limited when it is not, so the
template's effective sharpness would vary with *which* subjects covered *which*
voxel — a spatially-varying bias. Truncation is uniform.

### Per-subject scaling

`--normalize-fod` is applied *before* averaging.

| value | what it does | when it is right |
|---|---|---|
| `none` (default) | multiply by 1 | quantitative reconstructions whose amplitudes already share a unit: `consh --quantitative`, `mtnormalise`d MRtrix FODs |
| `l0` | per voxel, divide by the ℓ=0 coefficient | shape-only template; **annihilates AFD contrast** |
| `integral` | per voxel, scale to unit integral | differs from `l0` by a constant, so peak directions are identical |

`l0` and `integral` are per-voxel operations. They are useful for asking "do the
fibre *orientations* agree", and wrong for anything about fibre density.

### Heterogeneous inputs

- **Different SH basis** → converted to the reference basis via
  `interop::convert_sh_basis`. The reference is `--reference <odx>` if given,
  else the first input. A symmetric reference with a full-basis input is
  rejected (asymmetry cannot survive), and so is the reverse (the odd bands
  would be silently zeroed).
- **Different but same-lattice grid** → reindexed. An input stored LAS against a
  RAS+ reference is a signed axis permutation of the same physical lattice, so
  it is remapped rather than rejected. No SH rotation is involved: ODX stores
  directions in world (RAS mm) space and every basis matrix in the crate is
  built from world directions, so permuting voxel order permutes *rows* without
  rotating the coefficient vectors.
- **Genuinely different grid** → hard error naming `odx transform`.
- **The ODF sphere is irrelevant.** We never resample amplitudes, so unlike
  `odf_average` — which enforces an exact vertex match — the inputs' spheres may
  differ or be absent.

## What comes out

### Fixels

`directions` and `dpf/amplitude` are peak-found **from the aggregate, in the
cohort's own SH basis**, resolved from all four of `SH_BASIS`, `SH_ORDER`,
`SH_FULL_BASIS` and `SH_LEGACY` together. Reading only `SH_BASIS` is not enough:
a descoteaux dataset evaluated without its legacy bit gets the wrong sign on the
`m < 0` coefficients.

### Per-voxel maps

Every array below is written as a DPV in the group ODX and, with `--out-dir`, as
a `<name>.nii.gz` sidecar carrying the same numbers.

| array | meaning |
|---|---|
| `n_subjects` (u32) | contributing inputs at this voxel |
| `coverage_frac` | `n_subjects / N` |
| `l0_mean`, `l0_sd`, `l0_cv` | ℓ=0 spread across contributors (`n−1` divisor, `NaN` below 2). **`l0_cv` is the test–retest AFD reproducibility map.** |
| `acc_mean`, `acc_sd`, `acc_min` | angular correlation of each subject against the template |
| `acc_loo_mean`, `acc_loo_min` | leave-one-out versions |
| `anisotropic_power` | recomputed *from the aggregate* |
| `<name>`, `<name>_sd` | averaged shared scalar DPVs |

`anisotropic_power` is deliberately recomputed rather than averaged from the
inputs, because the anisotropic power of a mean is not the mean of the
anisotropic powers.

### The angular correlation coefficient

With `L = --acc-lmin` (default 2):

```
ACC(u, v) = Σ_{ℓ≥L} u_k v_k / sqrt( Σ_{ℓ≥L} u_k² · Σ_{ℓ≥L} v_k² )
```

Excluding ℓ=0 is the whole point. The isotropic DC term dominates the inner
product and would drive ACC to ~1 everywhere, including CSF. The value is `NaN`
in a voxel with no anisotropic energy, and the accumulators skip `NaN` so those
voxels do not poison the map.

### Leave-one-out

A subject scored against a template it helped build is scored partly against
itself, which inflates ACC by roughly `1/n`. The leave-one-out template removes
that contribution:

```
T_{-i,v} = (S_v − x_{i,v}) / (c_v − 1)
```

where `S_v` is the contributor sum already folded into the template and `c_v`
the contributor count. That is one vector subtract per (subject, voxel) — it
costs nothing and needs no re-aggregation. MRtrix's `population_template` uses
the same identity, auto-enabling it for `2 < n < 15`; `--loo auto` here is on
for `3 ≤ n ≤ 20`.

### The report

`--json` (or `--out-report <path>`) includes one row per subject:

```json
{ "key": "ses-3", "coverage_frac": 0.998, "n_fixels": 51203,
  "mean_acc": 0.941, "mean_acc_loo": 0.928,
  "basis_converted": false, "lmax_truncated_from": null,
  "is_outlier": false, "outlier_reasons": [] }
```

A subject is flagged when its leave-one-out ACC falls below
`median − 3·1.4826·MAD` **and** below an absolute floor of 0.90, or when it
covers under 90% of the template. The MAD makes the rule robust when one bad
scan would otherwise inflate the SD enough to hide itself; the absolute floor is
what stops a tight test–retest cohort — where the MAD nearly vanishes — from
flagging a perfectly good session on noise.

Flagging warns loudly on stderr and never drops a subject. `--fail-on-outlier`
makes the command exit nonzero instead. This is the analogue of DSI Studio's R²
outlier *warning* in `cmd/atl.cpp` — neither tool weights or excludes
automatically.

## Comparison with the reference implementations

| | DSI Studio `odf_average` | MRtrix3 `population_template` | `odx combine --method mean-fod` |
|---|---|---|---|
| What is averaged | half-sphere ODF amplitudes | SH coefficients | SH coefficients |
| Accumulator | `double` | `default_type` | `f64` |
| Divisor | per-voxel contributor count | NaN-skipping mean (equivalent) | per-voxel contributor count |
| Sphere must match | yes, exact vertex match | n/a | n/a — never resampled |
| lmax mismatch | n/a | not handled | truncate to cohort min |
| Basis mismatch | n/a | n/a | converted to the reference |
| Reorientation | none (QSDR did it) | aPSF, always, before aggregating | none — do it with `odx transform` |
| Per-subject intensity | `z0 = 1/max(QA)` at reconstruction | `mtnormalise` beforehand | `--normalize-fod`, or nothing if already quantitative |
| Voxel inclusion | ≥1 contributor (despite the docs) | full FOV; NaN-mask to exclude | `--min-coverage` |
| Registration | none | rigid → affine → nonlinear, re-averaging each level | none |
| QC | none in `odf_average` | leave-one-out; transform plausibility | ACC, leave-one-out ACC, ℓ=0 CV, coverage, outlier flagging |

## Quantitative FODs with CONSH

The cleanest input for a template is one where the amplitudes already mean the
same thing in every scan, so `--normalize-fod none` is correct and no
information is thrown away. `consh` (cs-dmri) produces that directly:

```bash
# Pass 1 — per-session responses
consh --dwi $d.nii.gz --bval $d.bval --bvec $d.bvec --mask $m \
      --step 50 --lmax-wm 8 --write-responses-to resp/$s --odx scratch/$s.odx

# Pass 2 — element-wise mean of the {wm,gm,csf}_response_denseb.txt matrices
#          into resp/group (CONSH responses are estimated from S/S0 with the
#          b=0 row pinned to sqrt(4pi), so they pool with no prior intensity
#          normalisation; --step, bmax and --lmax-wm must match)

# Pass 3 — refit everyone with the common kernel, in absolute units
consh --dwi $d.nii.gz --bval $d.bval --bvec $d.bvec --mask $m \
      --step 50 --lmax-wm 8 --read-responses-from resp/group \
      --quantitative --peak-min-amplitude-frac 0.15 --odx odx/$s.odx
```

`--quantitative` multiplies each voxel's fitted coefficients by its own `S0`,
undoing the per-voxel `S/S0` normalisation that makes CONSH's fibre density an
intra-axonal *fraction* rather than a volume. It also makes `mtnormalise`
load-bearing rather than a no-op, so the per-subject `DWI_ref` scale is absorbed
by the bias field while the AFD contrast survives. The pooled response makes
`AFD_ref` common across scans by construction.

Then average with `--normalize-fod none`.

## Division of labour with the rest of `odx combine`

The template block is one half of the command. The other half — per-subject
fixel correspondence (`angle_deg`, `matched`, the match-angle sweep, tangent
residuals), the design join, and the ModelArray cohort CSV — is unchanged and
orthogonal. A useful two-step pattern is to build and inspect the template
first, then adopt it:

```bash
odx combine ses-*.odx --method mean-fod --out-odx template.odx --out-dir maps/
odx combine ses-*.odx --template template.odx \
    --out-odx group.odx --out-cohort cohort.csv --per-subject-odx persubj/
```

Under `--method cluster` or `--template`, the FOD reproducibility block is off
by default; `--fod-qc` turns it on (it needs `sh/coefficients` on every input).
Under `--method mean-fod` it is on by default; `--no-fod-qc` turns it off.
