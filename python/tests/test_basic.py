"""Basic smoke tests for the `odx` Python module."""

from __future__ import annotations

import numpy as np
import pytest

import odx


def _identity_affine() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def _make_minimal_dataset(n_voxels: int = 4, sh_order: int = 2) -> odx.Odx:
    """SH-only ODX for `from_sh_coefficients` paths.

    Creates a 2x1x2 grid with a 3-voxel mask. SH coefficients are arbitrary
    but non-zero so the peak finder has something to bite on.
    """
    dims = (2, 1, 2)
    mask = np.array([[[1, 0]], [[1, 1]]], dtype=np.uint8)
    affine = _identity_affine()
    rng = np.random.default_rng(0)
    ncoeffs = (sh_order + 1) * (sh_order + 2) // 2
    sh_4d = np.zeros(dims + (ncoeffs,), dtype=np.float32)
    # Fill masked voxels with structured SH so the peak finder finds peaks.
    for (i, j, k) in np.argwhere(mask):
        sh_4d[i, j, k, 0] = 1.0  # constant DC term
        sh_4d[i, j, k, 1:] = rng.standard_normal(ncoeffs - 1).astype(np.float32) * 0.3
    return odx.from_sh_coefficients(
        sh_4d,
        mask=mask.astype(np.uint8),
        affine=affine,
        basis="descoteaux07",
        sh_order=sh_order,
        legacy=True,
        compute_peaks=True,
    )


# ─── PeakFinderConfig + SpherePeakFinder ─────────────────────────────────────


def test_peak_finder_config_defaults():
    cfg = odx.PeakFinderConfig()
    assert cfg.npeaks == 5
    assert cfg.relative_peak_threshold == pytest.approx(0.5)
    assert cfg.min_separation_angle_deg == pytest.approx(25.0)


def test_sphere_peak_finder_for_dsistudio_odf8_returns_peaks():
    finder = odx.SpherePeakFinder.for_dsistudio_odf8()
    # Synthetic ODF: one strong vertex, rest zero.
    odf = np.zeros(321, dtype=np.float32)
    odf[0] = 1.0
    amps, dirs = finder.find_peaks(odf)
    assert amps.shape == (1,)
    assert dirs.shape == (1, 3)
    assert amps[0] == pytest.approx(1.0)


# ─── from_sh_coefficients + builder + Odx properties ─────────────────────────


def test_from_sh_coefficients_creates_peaked_dataset():
    odx_obj = _make_minimal_dataset()
    assert odx_obj.nb_voxels == 3
    assert odx_obj.nb_peaks > 0
    assert "amplitude" in odx_obj.dpf_names()
    assert "coefficients" in odx_obj.sh_names()
    assert odx_obj.dipy_basis_name == "descoteaux07_legacy"


def test_odx_properties_have_expected_shapes():
    odx_obj = _make_minimal_dataset()
    aff = odx_obj.affine
    assert aff.shape == (4, 4)
    assert np.allclose(aff, np.eye(4))

    mask = odx_obj.mask
    assert mask.shape == (2, 1, 2)
    assert mask.sum() == 3

    offsets = odx_obj.offsets
    assert offsets.shape == (odx_obj.nb_voxels + 1,)
    assert offsets[0] == 0
    assert offsets[-1] == odx_obj.nb_peaks

    dirs = odx_obj.directions
    assert dirs.shape == (odx_obj.nb_peaks, 3)


def test_densify_directions_shape_and_zero_padding():
    odx_obj = _make_minimal_dataset()
    dense = odx_obj.densify_directions()
    n_max = odx_obj.max_peaks_per_voxel()
    assert dense.shape == (2, 1, 2, n_max, 3)
    # Unmasked voxel (0,0,1) must be all zeros.
    assert np.allclose(dense[0, 0, 1], 0.0)


def test_densify_dpf_amplitude():
    odx_obj = _make_minimal_dataset()
    dense = odx_obj.densify_dpf("amplitude")
    n_max = odx_obj.max_peaks_per_voxel()
    assert dense.shape == (2, 1, 2, n_max)
    # Amplitude > 0 wherever directions are non-zero.
    dirs = odx_obj.densify_directions()
    nonzero_dirs = (dirs != 0).any(axis=-1)
    nonzero_amps = dense > 0
    np.testing.assert_array_equal(nonzero_dirs, nonzero_amps)


def test_densify_sh_shape():
    odx_obj = _make_minimal_dataset(sh_order=2)
    sh = odx_obj.densify_sh("coefficients")
    assert sh.shape == (2, 1, 2, 6)  # (l+1)(l+2)/2 = 6 for lmax=2
    # Unmasked voxel zero.
    assert np.allclose(sh[0, 0, 1], 0.0)


# ─── native ODX round-trip ───────────────────────────────────────────────────


def test_native_odx_roundtrip_directory(tmp_path):
    src = _make_minimal_dataset()
    out = tmp_path / "test.odx_dir"
    src.to_directory(out)
    loaded = odx.load(out)
    np.testing.assert_array_equal(np.asarray(loaded.mask), np.asarray(src.mask))
    np.testing.assert_array_equal(np.asarray(loaded.offsets), np.asarray(src.offsets))
    np.testing.assert_allclose(
        np.asarray(loaded.directions), np.asarray(src.directions), atol=1e-6
    )


def test_native_odx_roundtrip_archive(tmp_path):
    src = _make_minimal_dataset()
    out = tmp_path / "test.odx"
    src.to_archive(out)
    loaded = odx.load(out)
    assert loaded.nb_peaks == src.nb_peaks
    np.testing.assert_allclose(
        np.asarray(loaded.directions), np.asarray(src.directions), atol=1e-6
    )


# ─── peak finder Python API ──────────────────────────────────────────────────


def test_peaks_from_sh_returns_offsets_directions_amplitudes():
    rng = np.random.default_rng(42)
    sh_order = 4
    ncoeffs = (sh_order + 1) * (sh_order + 2) // 2
    nrows = 5
    sh_rows = rng.standard_normal((nrows, ncoeffs)).astype(np.float32) * 0.2
    sh_rows[:, 0] = 1.0  # DC term
    # Use the built-in DSI Studio sphere.
    finder = odx.SpherePeakFinder.for_dsistudio_odf8()
    sphere_v = np.asarray(_dsi_hemisphere_vertices(), dtype=np.float32)
    sphere_f = np.asarray(_dsi_hemisphere_faces(), dtype=np.uint32)
    offsets, dirs, amps = odx.peaks_from_sh(
        sh_rows,
        sphere_v,
        sphere_f,
        basis="tournier07",
        sh_order=sh_order,
    )
    assert offsets.shape == (nrows + 1,)
    assert offsets[0] == 0
    assert offsets[-1] == dirs.shape[0]
    assert dirs.ndim == 2 and dirs.shape[1] == 3
    assert amps.shape[0] == dirs.shape[0]


# ─── basis conversion ────────────────────────────────────────────────────────


def test_convert_sh_basis_descoteaux_to_tournier():
    src = _make_minimal_dataset(sh_order=4)
    converted = src.to_tournier()
    assert converted.dipy_basis_name == "tournier07"
    assert "coefficients" in converted.sh_names()
    # Same nb_voxels and ncoeffs.
    assert converted.sh("coefficients").shape == src.sh("coefficients").shape


def test_convert_sh_basis_round_trip_tight():
    """Round-tripping via amplitudes should preserve the dense ODF
    closely (RMS error below a few percent at typical lmax)."""
    src = _make_minimal_dataset(sh_order=4)
    via = src.to_tournier().to_descoteaux(legacy=True)
    # Sample both ODFs on the DSI sphere and compare amplitudes.
    sphere_v = np.asarray(_dsi_hemisphere_vertices(), dtype=np.float32)
    sphere_f = np.asarray(_dsi_hemisphere_faces(), dtype=np.uint32)
    finder = odx.SpherePeakFinder.for_dsistudio_odf8()
    # Compare SH coefficients directly is fragile — sample on sphere instead.
    src_sh = np.asarray(src.sh("coefficients"))
    via_sh = np.asarray(via.sh("coefficients"))
    # SH coefficients should be roughly equal (legacy → modern → legacy is a no-op
    # in basis terms; only amplitude path adds noise).
    rms = np.sqrt(np.mean((src_sh - via_sh) ** 2))
    src_norm = np.sqrt(np.mean(src_sh ** 2)) + 1e-9
    assert rms / src_norm < 0.05, f"basis round-trip RMS {rms / src_norm:.3f} too high"


# ─── helpers (avoid pulling in odx-rs Rust internals from Python) ────────────


def _dsi_hemisphere_vertices() -> np.ndarray:
    v, _ = odx.spheres.dsistudio_odf8()
    return v


def _dsi_hemisphere_faces() -> np.ndarray:
    _, f = odx.spheres.dsistudio_odf8()
    return f
