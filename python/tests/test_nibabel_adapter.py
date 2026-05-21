"""Round-trip tests for ``odx.adapters.nibabel`` and the underlying
``odx.attach_dpv*`` Python entry points.

These exercise the full export → write → import → read cycle:

1. Build a minimal ODX in-memory via the SH builder.
2. Use ``odx.adapters.nibabel.to_nifti1_image`` to export a DPV as a
   ``Nifti1Image``, write it to disk.
3. Re-read that NIfTI, attach it back to a fresh ODX with
   ``odx.adapters.nibabel.attach_dpv``.
4. Reload the ODX, confirm the round-tripped DPV matches the original
   within tolerance.

Header policy is checked separately: the exported NIfTI must have
``qform_code = 1``, ``sform_code = 0``, and ``xyzt_units = 2``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import odx

nib = pytest.importorskip("nibabel")
from odx.adapters import nibabel as odx_nib  # noqa: E402


# ─── shared fixture ────────────────────────────────────────────────────────


def _build_minimal_odx(*, dpv_name: str, dpv_values: np.ndarray) -> odx.Odx:
    """Build a tiny ODX with a known DPV attached.

    2x1x2 grid, 3-voxel mask in C-order. Masked voxels iterated in C
    order (i-slowest, k-fastest) are: (0,0,0), (1,0,0), (1,0,1). The DPV
    rows align with that order, so ``dpv_values[0]`` lands at (0,0,0),
    ``dpv_values[1]`` at (1,0,0), and ``dpv_values[2]`` at (1,0,1).
    """
    dims = (2, 1, 2)
    # NB: `mask[i,j,k] = ...`. Keep an explicit 3-D form for readability,
    # then ravel before passing to OdxBuilder (which wants flat uint8).
    mask_3d = np.zeros(dims, dtype=np.uint8)
    mask_3d[0, 0, 0] = 1
    mask_3d[1, 0, 0] = 1
    mask_3d[1, 0, 1] = 1
    mask_flat = np.ascontiguousarray(mask_3d.ravel())

    affine = np.diag([2.0, 2.0, 2.0, 1.0]).astype(np.float64)
    affine[:3, 3] = [-1.0, 0.5, 0.25]

    sh_order = 2
    ncoeffs = (sh_order + 1) * (sh_order + 2) // 2
    sh_4d = np.zeros(dims + (ncoeffs,), dtype=np.float32)
    rng = np.random.default_rng(7)
    for (i, j, k) in np.argwhere(mask_3d):
        sh_4d[i, j, k, 0] = 1.0
        sh_4d[i, j, k, 1:] = rng.standard_normal(ncoeffs - 1).astype(np.float32) * 0.2

    builder = odx.OdxBuilder(np.ascontiguousarray(affine), dims, mask_flat)
    builder.set_sh_coefficients(sh_4d, basis="descoteaux07", sh_order=sh_order, legacy=True)
    builder.compute_peaks()

    arr = np.asarray(dpv_values, dtype=np.float32).reshape(-1, 1)
    nb_voxels = int(mask_3d.sum())
    assert arr.shape[0] == nb_voxels, (
        f"DPV row count {arr.shape[0]} != mask voxels {nb_voxels}"
    )
    builder.set_dpv(dpv_name, arr)
    return builder.finalize()


# ─── tests ─────────────────────────────────────────────────────────────────


def test_to_nifti1_image_scatters_compact_to_grid():
    """Exported NIfTI is `(X, Y, Z)`, masked-out voxels are 0, in-mask
    voxels carry the DPV value."""
    compact = np.array([3.5, 7.5, 9.0], dtype=np.float32)
    o = _build_minimal_odx(dpv_name="fa", dpv_values=compact)

    img = odx_nib.to_nifti1_image(o, "fa")

    assert img.shape == (2, 1, 2)
    arr = np.asarray(img.dataobj)
    # Mask is (0,0,0), (1,0,0), (1,0,1) in C-order — that's the same C-
    # order iteration used by compact_to_ijk, so compact[i] lands at the
    # i-th masked voxel.
    assert arr[0, 0, 0] == pytest.approx(3.5)
    assert arr[1, 0, 0] == pytest.approx(7.5)
    assert arr[1, 0, 1] == pytest.approx(9.0)
    # Masked-out voxel:
    assert arr[0, 0, 1] == 0.0


def test_to_nifti1_image_uses_correct_qform_sform_policy():
    """qform_code = 1, sform_code = 0, xyzt_units = 2; both slots carry
    the ODX affine."""
    o = _build_minimal_odx(dpv_name="x", dpv_values=np.arange(3, dtype=np.float32))

    img = odx_nib.to_nifti1_image(o, "x")
    h = img.header

    assert int(h["qform_code"]) == 1, "qform must be ScannerAnat (active)"
    assert int(h["sform_code"]) == 0, "sform must be Unknown (data preserved, inactive)"
    assert int(h["xyzt_units"]) == 2, "spatial units must be mm"

    # The affine readers see (img.affine) must match the ODX affine.
    np.testing.assert_allclose(img.affine, o.affine, atol=1e-4)
    # sform data is still on disk even though the code is 0:
    np.testing.assert_allclose(h.get_sform(), o.affine, atol=1e-4)
    # qform is the authoritative one:
    np.testing.assert_allclose(h.get_qform(), o.affine, atol=1e-4)


def test_save_dpv_nifti_writes_loadable_file(tmp_path: Path):
    compact = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    o = _build_minimal_odx(dpv_name="counts", dpv_values=compact)

    out = tmp_path / "counts.nii.gz"
    odx_nib.save_dpv_nifti(o, "counts", out)

    reread = nib.load(str(out))
    arr = np.asarray(reread.dataobj)
    assert arr.shape == (2, 1, 2)
    assert int(reread.header["qform_code"]) == 1
    assert int(reread.header["sform_code"]) == 0


def test_attach_dpv_roundtrip(tmp_path: Path):
    """Export a DPV → write to NIfTI → reload it → attach to a fresh
    copy of the ODX → reload and verify values match."""
    compact = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    o = _build_minimal_odx(dpv_name="orig", dpv_values=compact)

    # Save ODX with the original DPV
    odx_path = tmp_path / "subject.odx"
    o.save(str(odx_path))

    # Export DPV → NIfTI
    nii_path = tmp_path / "orig.nii.gz"
    odx_nib.save_dpv_nifti(o, "orig", nii_path)

    # Attach it back under a new name (so we can compare the two DPVs)
    report = odx_nib.attach_dpv(odx_path, "reattached", nii_path)
    assert report["name"] == "reattached"
    assert report["nb_voxels"] == 3

    # Reload and compare
    o2 = odx.load(str(odx_path))
    orig = np.asarray(o2.dpv("orig")).ravel()
    re = np.asarray(o2.dpv("reattached")).ravel()
    np.testing.assert_allclose(re, orig, atol=1e-4)


def test_attach_dpv_rejects_mismatched_dims(tmp_path: Path):
    """A NIfTI on the wrong grid shape must be rejected, not silently
    truncated."""
    compact = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    o = _build_minimal_odx(dpv_name="x", dpv_values=compact)
    odx_path = tmp_path / "subject.odx"
    o.save(str(odx_path))

    # Build a wrong-shape NIfTI: 3x1x2 instead of 2x1x2.
    bad = nib.Nifti1Image(
        np.zeros((3, 1, 2), dtype=np.float32),
        o.affine,
    )
    with pytest.raises(Exception) as excinfo:
        odx_nib.attach_dpv(odx_path, "bad", bad)
    assert "does not match" in str(excinfo.value).lower() \
        or "dimensions" in str(excinfo.value).lower() \
        or "shape" in str(excinfo.value).lower()


def test_attach_dpv_rejects_mismatched_affine(tmp_path: Path):
    """A NIfTI on a translated grid must be rejected (1e-3 mm tolerance
    is much tighter than 1 mm)."""
    compact = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    o = _build_minimal_odx(dpv_name="x", dpv_values=compact)
    odx_path = tmp_path / "subject.odx"
    o.save(str(odx_path))

    bad_aff = o.affine.copy()
    bad_aff[0, 3] += 1.0  # 1 mm shift
    bad = nib.Nifti1Image(np.zeros((2, 1, 2), dtype=np.float32), bad_aff)
    with pytest.raises(Exception) as excinfo:
        odx_nib.attach_dpv(odx_path, "bad", bad)
    assert "affine" in str(excinfo.value).lower()


def test_attach_dpv_dtype_auto_picks_u8_for_small_ints(tmp_path: Path):
    """``dtype='auto'`` should pick the narrowest unsigned int that fits."""
    compact = np.array([0.0, 5.0, 200.0], dtype=np.float32)
    o = _build_minimal_odx(dpv_name="orig", dpv_values=compact)
    odx_path = tmp_path / "subject.odx"
    o.save(str(odx_path))

    nii_path = tmp_path / "ints.nii.gz"
    odx_nib.save_dpv_nifti(o, "orig", nii_path)

    report = odx_nib.attach_dpv(odx_path, "as_u8", nii_path, dtype="auto")
    assert report["dtype"] == "uint8"
    assert not report["clamped"]


def test_attach_dpv_explicit_dtype_clamping(tmp_path: Path):
    """Forcing ``dtype='u8'`` on out-of-range values should clamp and
    report it."""
    compact = np.array([0.0, 500.0, 100000.0], dtype=np.float32)
    o = _build_minimal_odx(dpv_name="orig", dpv_values=compact)
    odx_path = tmp_path / "subject.odx"
    o.save(str(odx_path))

    nii_path = tmp_path / "big.nii.gz"
    odx_nib.save_dpv_nifti(o, "orig", nii_path)

    report = odx_nib.attach_dpv(odx_path, "as_u8", nii_path, dtype="u8")
    assert report["dtype"] == "uint8"
    assert report["clamped"]


def test_compact_to_ijk_matches_dpv_row_order():
    """The new ``compact_to_ijk`` property must agree with the DPV row
    order — that's the contract that makes manual scatter/gather safe."""
    compact = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    o = _build_minimal_odx(dpv_name="x", dpv_values=compact)

    ijk = o.compact_to_ijk
    assert ijk.shape == (3, 3)
    assert ijk.dtype == np.uint32

    # Scatter manually using the new accessor:
    vol = np.zeros(o.dimensions, dtype=np.float32)
    vol[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = compact

    # Must match what densify_dpv produces:
    auto = np.asarray(o.densify_dpv("x"))
    np.testing.assert_array_equal(auto, vol)


def test_attach_dpv_accepts_nibabel_image_directly(tmp_path: Path):
    """``attach_dpv`` should also accept an in-memory Nifti1Image, not
    just a path — that's the natural API when chaining with nilearn."""
    compact = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    o = _build_minimal_odx(dpv_name="orig", dpv_values=compact)
    odx_path = tmp_path / "subject.odx"
    o.save(str(odx_path))

    img = odx_nib.to_nifti1_image(o, "orig")
    report = odx_nib.attach_dpv(odx_path, "via_image", img)
    assert report["name"] == "via_image"

    o2 = odx.load(str(odx_path))
    re = np.asarray(o2.dpv("via_image")).ravel()
    np.testing.assert_allclose(re, compact, atol=1e-4)
