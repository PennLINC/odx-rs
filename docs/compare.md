# Pairwise ODX comparison

`odx compare` compares two ODX files (`A` and `B`) on a shared grid and emits:

1. a directory of per-voxel scalar NIfTIs (3D, single-volume, float32),
2. a `comparison.odx` archive whose geometry mirrors `A`'s, with extra
   per-fixel DPFs and per-voxel DPVs that encode the comparison.

The same numeric fields appear under three views:

- as 3D NIfTIs in `<out-dir>/*.nii.gz` (load in any neuroimaging viewer),
- as DPVs inside `<out-dir>/comparison.odx` (color voxel hulls in TRXViz),
- as DPFs inside `<out-dir>/comparison.odx` (color individual fixels in
  TRXViz).

DPVs and the matching NIfTIs carry the *same numbers* — DPVs are the masked
voxels' values pulled out of the full 3D volume in compact mask order, so
loading either one paints the same picture; the NIfTIs are what other tools
(itksnap, fsleyes, dsi-studio) consume.

## Conventions

- **Mask intersection**: most metrics are only defined on `mask_a ∩ mask_b`
  (voxels both files cover). Outside the intersection a NIfTI voxel reads
  `NaN` so viewers render it transparent. The same `NaN` shows up in the
  comparison ODX's DPV — for voxels that are in `mask_a` but not in
  `mask_b`, the DPV row is `NaN`.
- **Counts** (`n_fixels_*`, `n_matched`, `n_coherent_*`) use `0` for "zero
  fixels" and `NaN` only for "outside the relevant mask".
- **Angles** are in degrees.
- **Booleans** (`is_mutual`, `top1_match`) are stored as `u8` for DPF and
  `f32` for DPV/NIfTI; they take values `0` (false), `1` (true), and `NaN`
  for "undefined / outside intersection".
- **The "primary metric"** is the scalar nonnegative DPF used for fixel QC
  and for "top-1 by primary" logic. It is resolved with the same priority
  list as `odx qc`: `--primary-dpf <name>` if given, else `amplitude` →
  `afd` → `qa`. Both files are forced to use the *same* metric — `B` is
  re-QC'd with `A`'s resolved name if the auto-pick differs, so the two
  coherence indices are directly comparable.

## Per-voxel fields (NIfTIs and DPVs)

Each field below appears as `<out-dir>/<name>.nii.gz` *and* as
`comparison.odx → dpv/<name>.float32`.

### Fixel counts and set membership

- `n_fixels_a`, `n_fixels_b` — fixel count in each file. `NaN` outside that
  file's own mask. Compare to see "where does B find more peaks than A?"
- `n_fixels_diff` — `n_fixels_a − n_fixels_b`. Defined only on the
  intersection. Positive = A has more fixels in this voxel than B.
- `n_matched` — number of mutually matched fixels in this voxel (see
  matching algorithm below).
- `n_unmatched_a`, `n_unmatched_b` — fixels in A (resp. B) that did **not**
  participate in a mutual match. Sums to `n_fixels_* − n_matched`.
- `jaccard` — `n_matched / (n_a + n_b − n_matched)` ∈ [0, 1]. The
  Tanimoto/IoU coefficient between A's and B's fixel sets in this voxel.
  `NaN` when both are zero.
- `dice` — `2·n_matched / (n_a + n_b)` ∈ [0, 1]. The Sørensen–Dice
  coefficient. Always ≥ Jaccard; weights matched fixels twice. `NaN` when
  both are zero.

### Match quality

- `mean_match_angle_deg` — mean angular error (degrees) over mutually
  matched fixels in this voxel. `NaN` if `n_matched == 0`.
- `max_match_angle_deg` — worst angular error among matches. Useful for
  flagging voxels where one match drags the rest.
- `top1_match` — `1` if A's primary-largest fixel is mutually matched to
  B's primary-largest fixel, else `0`. `NaN` outside the intersection or
  when either side has zero fixels. **Use this for "did the dominant
  direction agree?"** — it's the cheapest single bit of agreement and
  often what tracking really cares about.
- `n_a_collisions` — number of A fixels whose best-B is shared with another
  A fixel before mutual filtering. Counts "merge events" — voxels where
  several A fixels collapse onto a single B direction. High values flag
  fanning/crossing voxels where one method resolved more peaks than the
  other.

### Coherence (compared on equal footing)

Coherence here uses `compute_fixel_qc` with both files threaded through
the same `--primary-dpf`, `--threshold`, and `--coherence-angle-deg`. A
fixel is "Connected" iff its direction aligns with the trajectory to a
neighboring voxel and that neighbor has at least one fixel of similar
direction. See `docs/fixel_qc.md` for the full definition.

- `n_coherent_a`, `n_coherent_b` — count of `Connected`-class fixels in
  each file.
- `n_coherent_mutual` — count of fixels that are both (a) `Connected` in
  A, (b) mutually matched to a fixel in B that is also `Connected`. The
  conservative "agreement" measure: it isn't enough that both methods
  found a peak here — both also have to think the peak fits its
  neighborhood.
- `coherence_index_diff` — A's whole-volume coherence index minus B's,
  broadcast across the intersection. (Same scalar repeated everywhere
  inside the intersection. Useful as a single colorbar number; less
  useful as a 3D map.)

### Per-shared-DPF (one set per scalar f32 DPF in both files)

For each scalar nonnegative DPF that exists in *both* A and B with
`ncols=1` and a float dtype (e.g. `qa`, `amplitude`, `afd`),
`compare` emits five voxel fields named
`dpf_<key>_*`:

- `dpf_<key>_a_sum`, `dpf_<key>_b_sum` — total mass in this voxel
  (`Σ value`). Often interpreted as voxel-wise integrated ODF mass.
- `dpf_<key>_sum_diff` — `a_sum − b_sum`. Signed disagreement at the
  voxel level. Independent of matching.
- `dpf_<key>_diff_mean` — mean of `(A − B)` over mutually matched fixels
  in this voxel. `NaN` when `n_matched == 0`. Tells you "where matched
  fixels agree on direction but A reports systematically higher / lower
  qa than B."
- `dpf_<key>_diff_max_abs` — `max(|A − B|)` over matched fixels. Spots
  outlier voxels where one match has a large DPF disagreement.

`qc_class` (the QC output DPF) is reserved and never appears here.

## Per-fixel fields (DPFs in `comparison.odx`)

DPFs have one row per A-fixel in the same order as `A.directions`. Use
them in TRXViz to color individual peaks rather than voxel hulls.

- `match_index_b` (`int32`) — global B-fixel index (offset into
  `B.directions`) that this A-fixel mutually matched, or `−1` if no
  mutual match. Convenient if you want to cross-reference into B
  programmatically.
- `match_angle_deg` (`f32`) — angle (degrees) between this A-fixel and
  its **best B candidate**, regardless of mutuality. `NaN` if the voxel
  has no B fixels (`n_fixels_b == 0`). Lets you visualize "where do A's
  peaks point compared to the closest thing B has?" even for fixels that
  failed the mutual / threshold gates.
- `match_dp` (`f32`) — `|A·B|` (cosine magnitude) at the best B match.
  Equivalent to `cos(match_angle_deg)`. `NaN` under the same conditions.
- `is_mutual` (`u8`, 0/1) — `1` iff this A-fixel and its best B are each
  other's best AND `match_dp ≥ cos(match_angle_deg threshold)`. The
  flag that gates everything in the per-voxel mutual statistics.
- `qc_class_a` (`u8`) — A's coherence class for this fixel:
  `0 = ThresholdedOut`, `1 = Disconnected`, `2 = Connected`.
- `qc_class_b_matched` (`u8`) — the matched B-fixel's class, with `255`
  as "no mutual match." Lets you spot fixels that are coherent in A but
  matched to a disconnected B (or vice-versa).
- `<key>_a` (`f32`) — A's value of shared scalar DPF `<key>`, copied for
  viewer convenience. Same numbers as A's original `dpf/<key>`.
- `<key>_b_matched` (`f32`) — the matched B-fixel's `<key>` value, or
  `NaN` if no mutual match.
- `<key>_diff` (`f32`) — `A − B` for `<key>` on mutual matches, `NaN`
  otherwise. Useful for "qa is systematically higher in A than B" or
  "amplitude disagrees by direction."

## Matching algorithm

For each voxel `v ∈ mask_a ∩ mask_b`, with `m = n_fixels_a(v)` and
`n = n_fixels_b(v)`:

1. Build the `m × n` matrix `dp[i,j] = |A[i] · B[j]|`. Absolute value
   absorbs the antipodal ambiguity natural to fixel directions.
2. `best_b[i] = argmax_j dp[i,j]`, `best_a[j] = argmax_i dp[i,j]`.
3. A fixel pair `(i, j)` is **mutual** iff `best_b[i] = j` AND
   `best_a[j] = i` AND `dp[i,j] ≥ cos(match_angle_deg)`.

Greedy + mutual is `O(m·n)` per voxel. `m` and `n` are typically ≤ 5, so
the matrix is tiny. The greedy collision count (`n_a_collisions`) is the
number of A-fixels whose best-B is also another A's best-B *before*
mutuality is enforced — high values surface "merge events" where multiple
A peaks collapse onto one B direction.

This is a bidirectional generalization of MRtrix's
[`fixelcorrespondence`](../trx-mrtrix2/cpp/cmd/fixelcorrespondence.cpp),
which only does best-B-for-each-A on a fixed template. Mutuality is what
makes the `n_matched` count symmetric: `n_unmatched_a + n_matched`
equals `m` and `n_unmatched_b + n_matched` equals `n`.

### Gates

- `--match-angle-deg` (default `30`) — the maximum angular disagreement
  for a pair to count as mutual. More permissive than MRtrix's `45`
  cross-subject default since `compare` assumes both files are on the
  same anatomy. Tighten to `15`–`20` if you only want clearly-aligned
  matches.
- `--coherence-angle-deg` (default `15`) — passed straight through to
  the `compute_fixel_qc` call run on each side. Controls coherence,
  *not* matching. Drives `n_coherent_*` and `qc_class_*`.
- `--threshold` / `--threshold-value` — same modes as `odx qc`
  (`otsu` / `positive` / `all` / `value`). Determines which fixels are
  evaluated by the QC pass. Does **not** filter the matching pass —
  matching always considers every stored fixel.

## Reading the result

Common patterns:

- **"Where do the two methods disagree on direction?"**
  Color by `mean_match_angle_deg` (DPV) or `match_angle_deg` (DPF).
  Hot-spot voxels at GM/CSF boundaries are normal; deep WM hot spots are
  worth investigating.
- **"Did the dominant peak agree?"**
  Color by `top1_match` (DPV). Voxels where it's `0` while
  `n_matched > 0` are "secondary peaks agreed but the primary did not."
- **"Where did one method find peaks the other missed?"**
  `n_unmatched_a` (DPV) — A peaks B doesn't see; `n_unmatched_b` — vice
  versa. Combine with `n_a_collisions` to spot merge/split events.
- **"How does qa compare on matched fixels?"**
  `dpf_qa_diff_mean` (DPV) for voxel-level summary, `qa_diff` (DPF) for
  per-fixel inspection.
- **"How robust is each fit?"**
  `n_coherent_a` − `n_coherent_b` (DPV-level subtraction in your viewer)
  approximates "which method produces neighborhood-coherent peaks here."
  `n_coherent_mutual` is the conservative agreement count.

## Self-comparison

`odx compare --a x.odx --b x.odx` is a sanity check:

- `n_fixels_diff ≡ 0`, `jaccard ≡ 1`, `dice ≡ 1`, `top1_match ≡ 1`,
- `is_mutual ≡ 1` for every fixel,
- `match_angle_deg` and `mean_match_angle_deg` are at the f32 floor
  (small but nonzero — `acos(1−ε)` is a few mdeg even when the input
  vectors are bit-identical, because `|A·B|` is computed in f32),
- every `*_diff*` field is identically `0`.
