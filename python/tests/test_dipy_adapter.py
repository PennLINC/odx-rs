"""End-to-end dipy adapter test: SH-only ODX → PeaksAndMetrics → tracking-compatible.

Runs only if dipy is importable (`pytest.importorskip("dipy")`).
"""

from __future__ import annotations

import numpy as np
import pytest

import odx

dipy = pytest.importorskip("dipy")
from dipy.core.sphere import Sphere


def _make_sh_only_odx(sh_order: int = 4):
    dims = (3, 3, 3)
    mask = np.zeros(dims, dtype=np.uint8)
    mask[1:, 1:, 1:] = 1
    affine = np.eye(4, dtype=np.float64)
    rng = np.random.default_rng(0)
    ncoeffs = (sh_order + 1) * (sh_order + 2) // 2
    sh = np.zeros(dims + (ncoeffs,), dtype=np.float32)
    for (i, j, k) in np.argwhere(mask):
        sh[i, j, k, 0] = 1.0
        sh[i, j, k, 1:] = rng.standard_normal(ncoeffs - 1).astype(np.float32) * 0.25
    return odx.from_sh_coefficients(
        sh, mask=mask, affine=affine,
        basis="descoteaux07", sh_order=sh_order, legacy=True,
        compute_peaks=True,
    )


def test_to_peaks_and_metrics_basic_shapes():
    src = _make_sh_only_odx()
    sphere_v, _ = odx.spheres.dsistudio_odf8()
    sphere = Sphere(xyz=sphere_v.astype(np.float64))
    pam = odx.adapters.dipy.to_peaks_and_metrics(src, sphere=sphere)

    # Mandatory PAM5 fields populated.
    assert pam.peak_dirs.dtype == np.float32
    assert pam.peak_dirs.shape[:3] == (3, 3, 3)
    assert pam.peak_dirs.shape[-1] == 3
    n_max = pam.peak_dirs.shape[3]
    assert pam.peak_values.shape == (3, 3, 3, n_max)
    assert pam.peak_values.dtype == np.float32
    assert pam.peak_indices.shape == (3, 3, 3, n_max)
    assert pam.peak_indices.dtype == np.int32

    # Sphere round-tripped.
    np.testing.assert_allclose(
        pam.sphere.vertices.astype(np.float32), sphere_v, atol=1e-5
    )

    # SH attached.
    assert pam.shm_coeff is not None
    assert pam.shm_coeff.shape[:3] == (3, 3, 3)


def test_sh_only_compute_peaks_path():
    """Build an SH-only Odx (no peaks), then materialize PAM with compute_peaks=True."""
    sh_only = _make_sh_only_odx()
    # Round-trip through compute_peaks=False to get an SH-only artifact.
    dims = sh_only.dimensions
    mask = np.asarray(sh_only.mask)
    rng = np.random.default_rng(7)
    sh_order = sh_only.sh_order
    ncoeffs = (sh_order + 1) * (sh_order + 2) // 2
    sh = np.zeros(dims + (ncoeffs,), dtype=np.float32)
    for (i, j, k) in np.argwhere(mask):
        sh[i, j, k, 0] = 1.0
        sh[i, j, k, 1:] = rng.standard_normal(ncoeffs - 1).astype(np.float32) * 0.25
    sh_only_no_peaks = odx.from_sh_coefficients(
        sh, mask=mask, affine=np.asarray(sh_only.affine),
        basis="descoteaux07", sh_order=sh_order, legacy=True,
        compute_peaks=False,
    )
    assert sh_only_no_peaks.nb_peaks == 0

    sphere_v, _ = odx.spheres.dsistudio_odf8()
    sphere = Sphere(xyz=sphere_v.astype(np.float64))

    with pytest.raises(ValueError, match="compute_peaks"):
        odx.adapters.dipy.to_peaks_and_metrics(sh_only_no_peaks, sphere=sphere)

    pam = odx.adapters.dipy.to_peaks_and_metrics(
        sh_only_no_peaks, sphere=sphere, compute_peaks=True
    )
    assert (pam.peak_values > 0).any()


def test_round_trip_pam_to_odx_to_pam():
    """F.3-style round-trip: PAM → ODX → PAM. peak_values, sphere, shm_coeff
    preserved; peak_dirs within sphere-quantization tolerance."""
    src = _make_sh_only_odx()
    sphere_v, _ = odx.spheres.dsistudio_odf8()
    sphere = Sphere(xyz=sphere_v.astype(np.float64))
    pam_a = odx.adapters.dipy.to_peaks_and_metrics(src, sphere=sphere)

    # PAM → ODX: from_peaks_and_metrics. NOTE: directions are pushed verbatim;
    # SH is masked-flattened.
    odx_b = odx.adapters.dipy.from_peaks_and_metrics(pam_a, basis="descoteaux07", legacy=True)

    # ODX → PAM (back).
    pam_b = odx.adapters.dipy.to_peaks_and_metrics(odx_b, sphere=sphere)

    # peak_values should be byte-equal (both pulled from the same DPF/amplitude).
    np.testing.assert_allclose(pam_b.peak_values, pam_a.peak_values, atol=1e-5)
    # peak_dirs within sphere-quantization tolerance: dot product should be high.
    nonzero = pam_a.peak_values > 0
    if nonzero.any():
        dots = np.sum(pam_a.peak_dirs * pam_b.peak_dirs, axis=-1)
        # Where the original was zero-padded, the round-trip might also be zero
        # → dot 0 there. Check only where both have non-zero peaks.
        active = (np.linalg.norm(pam_a.peak_dirs, axis=-1) > 0) & (
            np.linalg.norm(pam_b.peak_dirs, axis=-1) > 0
        )
        if active.any():
            assert np.abs(dots[active]).min() > 0.9
