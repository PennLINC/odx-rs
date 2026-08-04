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
| `acc_mean`, `acc_sd`, `acc_min` | angular correlation of each subject against the template — a coefficient correlation, **not** an angle; see the calibration below |
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
product and would drive ACC to ~1 everywhere, including CSF.

The value is `NaN` where the aggregate carries no anisotropic energy, and the
accumulators skip `NaN`. That floor is load-bearing rather than defensive:
multi-tissue deconvolution leaves the WM compartment identically zero across a
large interior region of a brain mask (16.2% of voxels in the cohort above), and
a naive "norm > 0" test lets those rows through at ~1e-16, where ACC evaluates
to `1/√n` on rounding noise — a *finite* value that drags every whole-brain
summary toward it. The threshold is 1e-6 of the median ℓ≥2 norm over template
voxels that have one, so it adapts to the cohort's units. The excluded count is
reported as `n_voxels_without_orientation`; read the ACC numbers against it.

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

At **n = 2** the identity collapses to `T_{-i} = x_j`, so `acc_loo` becomes the
direct pairwise agreement between the two scans — the number a two-session
test-retest actually wants. `--loo on` enables it there; `auto` does not, so the
default is unchanged. Note that a two-input "template" carries no more
information than `odx compare` between the two scans: `acc_loo`, and the angles,
are pairwise either way. Its value is the shared fixel geometry and the map
layout, not extra statistical power. `l0_sd` has one degree of freedom and every
`_sd` companion is degenerate.

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

## Reference numbers

Baselines from 8 test–retest sessions of one subject (`sub-0001a`, QSIPrep
intramodal-template aligned, 2 mm ACPC, 279 directions, bmax ~4985), each
reconstructed with `consh --quantitative --read-responses-from <pooled>`:

| check | value | note |
|---|---|---|
| template | 207 243 voxels, 376 091 fixels | 9 s wall clock for all 8 subjects |
| peaks/voxel vs sessions | **0.934×** | matched peak finder; averaging suppresses noise lobes |
| ACC median, WM | **0.985** (leave-one-out 0.980) | WM = top tercile of anisotropic power |
| ACC median, whole brain | 0.956 | over the 83.8% of voxels where ACC is defined |
| leave-one-out self-bias | 0.070 | `acc_mean − acc_loo_mean` at n=8 |
| voxels with no orientation | 33 543 (16.2%) | WM compartment identically zero; excluded from every ACC summary |
| coverage | 2.2% of voxels below full | consistent with mask Dice 0.98–0.995 |
| `l0_cv` median in WM | **7.5%** | the test–retest AFD reproducibility number |
| pooled response spread | ≤0.7% (r₀), ≤2.8% (r₂) | across the 8 sessions, before pooling |

Two cross-checks worth repeating on new data:

- **`odx compare` agreement.** Ranking the sessions by
  `mean_match_angle_deg` against the template agreed with ranking them by
  `mean_acc_loo` (rank correlation 0.69, same best session by both). If those
  two disagree, one of the metrics is measuring the wrong thing.
- **Is the cohort really on a common scale?** Rebuilding with
  `--normalize-fod l0` changed mean ACC by 0.004 (0.688 → 0.692) and the fixel
  count by 2.4%. Per-voxel ℓ=0 normalization barely moving the template is the
  direct evidence that `--quantitative` plus a pooled response did put the
  sessions on one scale. Had the two templates diverged, the quantitative
  normalization would not be holding — worth knowing independently of the
  template.

**Watch the peaks-per-voxel ratio.** If the template has *more* fixels per voxel
than its inputs, the inputs are not aligned and the average is smearing distinct
orientations into spurious crossings. Two traps in reading it:

- Compare against a *matched* peak finder. A session ODX peak-found with
  `--peak-min-amplitude-frac` is not comparable to a template peak-found without
  one. A one-input `odx combine --method mean-fod --no-fod-qc` run re-peaks a
  session through the identical path.
- Restrict to voxels where every side has signal. In a multi-tissue cohort the
  template's zero set is the *intersection* of the sessions' zero sets, so an
  unrestricted ratio partly measures dead-voxel fractions rather than sharpness.
  On matched support the ratio above is 0.99; in WM it is 0.97.

## ACC is not an angle

`acc_*` is a correlation of SH coefficient vectors, not an angular difference,
and it is considerably more forgiving than one. Measured on this cohort in WM
(Spearman −0.90 against the per-voxel mean fixel angle):

| ACC | median fixel angle |
|---|---|
| ≥0.995 | 1.3° |
| 0.990–0.995 | 1.9° |
| 0.985–0.990 | 2.5° |
| 0.980–0.985 | **3.0°** |
| 0.970–0.980 | 3.6° |
| 0.950–0.970 | 4.5° |
| 0.900–0.950 | 5.8° |

Near ACC = 1 the relationship follows roughly `θ ≈ 29·√(1 − ACC)` degrees, so
ACC 0.99 is still about 2°, and you need ~0.998 before the angle drops below 1°.
**Quote ACC with this calibration attached, or quote the angle instead.** The
band powers in the denominator are cohort-specific, so recalibrate on your own
data rather than reusing this table.

### Report angles on a stratum, or the number means nothing

A whole-brain fixel angle is dominated by low-amplitude fixels in grey matter and
CSF, where the orientation is barely determined. The same cohort, same
`mean_angle_deg` array, stratified:

| stratum | fixels | median angle |
|---|---|---|
| all fixels where the angle is defined | 200 892 | 6.25° |
| + matched by ≥4 of 8 sessions | 162 256 | 5.33° |
| + white matter (multi-tissue WM fraction > 0.7) | 53 140 | 2.99° |
| + above the Otsu amplitude threshold | 43 398 | **2.54°** |
| + primary fixel only | 39 491 | 2.41° |
| Otsu threshold alone, any tissue | 68 252 | 2.87° |

The white-matter definition comes from the multi-tissue ℓ=0 terms —
`wm/(wm+gm+csf)`, all three on the mtnormalise scale — which is more principled
than an anisotropy proxy when the reconstruction is multi-tissue. The Otsu
threshold is whatever `odx qc` resolves on the primary metric.

Both cutoffs are gentle: the WM-fraction cutoff moves the median from 2.73° at
0.5 to 2.34° at 0.9, and the amplitude cutoff from 3.15° unthresholded to 1.83°
at twice Otsu. So the stratum choice is worth ~1°, not an order of magnitude —
but quote which one you used.

**Selection is part of the answer.** Thresholding on amplitude keeps the fixels
whose orientation is best determined, so a lower angle there is partly better
data and partly selection. That is the right stratum to quote for a
fixel-based analysis, because it is the one you would actually analyze — but it
is not "the scan's accuracy".

### In-sample versus held-out

`mean_angle_deg` is computed against a template the subject helped build, and
unlike `acc_loo_*` it has no leave-one-out variant. Measured on this cohort by
retraining eight 7-session templates and matching the held-out session, on the
same statistic (one session's angle, WM + that template's own Otsu):

| | median angle |
|---|---|
| in-sample | 2.23° |
| held out | **2.56°** (range 2.28–2.92 across sessions) |

so about **1.15×** optimistic in this stratum. Compare the two on the *same*
statistic — a mean over N subjects against a single subject's angle is a
smoother quantity and makes the optimism look like 1.0×.

For wider context on the same data: whole-brain per-voxel median 4.8°, and plain
session-to-session `odx compare` with no template at all gives 8.2–9.2°.

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
