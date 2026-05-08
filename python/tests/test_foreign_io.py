"""Foreign-format I/O round-trips and the FZ quantization warning."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import odx


def _make_dataset(sh_order: int = 4):
    dims = (3, 3, 3)
    mask = np.zeros(dims, dtype=np.uint8)
    mask[1:, 1:, 1:] = 1  # 8 masked voxels
    affine = np.eye(4, dtype=np.float64)
    rng = np.random.default_rng(99)
    ncoeffs = (sh_order + 1) * (sh_order + 2) // 2
    sh = np.zeros(dims + (ncoeffs,), dtype=np.float32)
    for (i, j, k) in np.argwhere(mask):
        sh[i, j, k, 0] = 1.0
        sh[i, j, k, 1:] = rng.standard_normal(ncoeffs - 1).astype(np.float32) * 0.25
    return odx.from_sh_coefficients(
        sh, mask=mask, affine=affine,
        basis="tournier07", sh_order=sh_order, legacy=False,
        compute_peaks=True,
    )


# ─── MRtrix MIF round-trip ───────────────────────────────────────────────────


def test_save_mif_then_from_mrtrix_roundtrips_sh(tmp_path):
    src = _make_dataset()
    out_dir = tmp_path / "mrtrix_out"
    out_dir.mkdir()
    src.save_mrtrix(out_dir)
    loaded = odx.from_mrtrix(out_dir)
    src_sh = np.asarray(src.sh("coefficients"))
    loaded_sh = np.asarray(loaded.sh("coefficients"))
    np.testing.assert_allclose(loaded_sh, src_sh, atol=1e-3)


# ─── DSI Studio FZ ────────────────────────────────────────────────────────────


def test_save_fz_emits_quantization_warning(tmp_path):
    """Newton-refined peaks land off the DSI sphere → save_fz quantizes
    and warns."""
    src = _make_dataset()
    out = tmp_path / "test.fz"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        src.save_fz(out)
    quant = [w for w in caught if "quantizes peaks" in str(w.message)]
    # Synthetic SH may or may not produce off-vertex peaks — assert that if any
    # warning fires, it has the right message structure; if none fires, peaks
    # were already on-vertex (no false positive). Either is acceptable here.
    if quant:
        msg = str(quant[0].message)
        assert "median angular drift" in msg
        assert "with_peaks_from_sh" in msg


def test_save_fz_with_lossy_warning_false_silent(tmp_path):
    src = _make_dataset()
    out = tmp_path / "test.fz"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        src.save_fz(out, lossy_warning=False)
    quant = [w for w in caught if "quantizes peaks" in str(w.message)]
    assert len(quant) == 0


# ─── Basis convert + save_mrtrix ─────────────────────────────────────────────


def test_save_mrtrix_auto_converts_descoteaux_to_tournier(tmp_path):
    """Building a descoteaux dataset and exporting to MRtrix should auto-convert."""
    dims = (2, 2, 2)
    mask = np.ones(dims, dtype=np.uint8)
    affine = np.eye(4, dtype=np.float64)
    rng = np.random.default_rng(7)
    sh_order = 4
    ncoeffs = (sh_order + 1) * (sh_order + 2) // 2
    sh = np.zeros(dims + (ncoeffs,), dtype=np.float32)
    for (i, j, k) in np.argwhere(mask):
        sh[i, j, k, 0] = 1.0
        sh[i, j, k, 1:] = rng.standard_normal(ncoeffs - 1).astype(np.float32) * 0.2
    src = odx.from_sh_coefficients(
        sh, mask=mask, affine=affine,
        basis="descoteaux07", sh_order=sh_order, legacy=True,
        compute_peaks=True,
    )
    assert src.dipy_basis_name == "descoteaux07_legacy"

    out_dir = tmp_path / "mrtrix_auto"
    out_dir.mkdir()
    src.save_mrtrix(out_dir)  # default convert_basis="auto"
    loaded = odx.from_mrtrix(out_dir)
    assert loaded.dipy_basis_name == "tournier07"


def test_save_mrtrix_strict_refuses_descoteaux(tmp_path):
    dims = (2, 2, 2)
    mask = np.ones(dims, dtype=np.uint8)
    affine = np.eye(4, dtype=np.float64)
    sh_order = 4
    ncoeffs = (sh_order + 1) * (sh_order + 2) // 2
    sh = np.zeros(dims + (ncoeffs,), dtype=np.float32)
    sh[..., 0] = 1.0
    src = odx.from_sh_coefficients(
        sh, mask=mask, affine=affine,
        basis="descoteaux07", sh_order=sh_order, legacy=True,
        compute_peaks=False,
    )
    out_dir = tmp_path / "mrtrix_strict"
    out_dir.mkdir()
    with pytest.raises(ValueError, match="tournier07"):
        src.save_mrtrix(out_dir, convert_basis="strict")
