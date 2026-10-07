"""Coherence QC (`Odx.coherence`) from Python.

The Rust tests (`tests/qc.rs`) cover the scoring rules; these check the
binding: modes, thresholds, the b-table check and the error paths.
"""

from __future__ import annotations

import numpy as np
import pytest

import odx

LPS = np.array([[-2.0, 0, 0, 30], [0, -2.0, 0, 40], [0, 0, 2.0, -20], [0, 0, 0, 1]])
DIMS = (22, 22, 22)
TUBES = [  # (start, index step): three axes and three face diagonals
    ((2, 4, 4), (1, 0, 0)),
    ((14, 2, 4), (0, 1, 0)),
    ((4, 14, 2), (0, 0, 1)),
    ((12, 12, 14), (1, 1, 0)),
    ((4, 12, 12), (0, 1, 1)),
    ((14, 4, 12), (1, 0, 1)),
]


def _tube_phantom(affine: np.ndarray, corrupt: np.ndarray = np.eye(3)) -> odx.Odx:
    """Six one-voxel-wide tubes whose fibres were fitted with a gradient
    table corrupted by `corrupt` (voxel axes)."""
    axes = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
    dirs = {}
    for start, step in TUBES:
        d = axes @ corrupt @ np.asarray(step, float)
        for i in range(6):
            dirs[tuple(np.add(start, np.multiply(i, step)))] = d / np.linalg.norm(d)
    mask = np.zeros(DIMS, dtype=np.uint8)
    for coord in dirs:
        mask[coord] = 1
    builder = odx.OdxBuilder(affine, DIMS, mask.reshape(-1))
    for coord in sorted(dirs):  # C order, the order of the flattened mask
        builder.push_voxel_peaks(np.asarray([dirs[coord]], dtype=np.float32))
    builder.set_dpf("amplitude", np.ones((len(dirs), 1), dtype=np.float32))
    return builder.finalize()


def test_primary_coherence_of_a_clean_phantom_is_one():
    report = _tube_phantom(LPS).coherence()
    assert report["mode"] == "primary"
    assert report["primary_metric"] == "amplitude"
    assert report["evaluated_voxels"] == 36
    assert report["coherence_index"] == pytest.approx(1.0)
    assert "btable" not in report


def test_fixel_mode_and_module_function():
    ds = _tube_phantom(LPS)
    report = odx.coherence(ds, "fixel", threshold="positive")
    assert report["mode"] == "fixel"
    assert report["connected_fixels"] == 36
    assert report["coherence_index"] == pytest.approx(1.0)


@pytest.mark.parametrize("scoring", ["fixel", "primary"])
@pytest.mark.parametrize("affine", [np.eye(4), LPS], ids=["ras", "lps"])
def test_btable_check_names_the_fix(affine, scoring):
    swap_xy = np.array([[0.0, 1, 0], [1, 0, 0], [0, 0, 1]])  # its own inverse
    report = _tube_phantom(affine, swap_xy).coherence(
        scoring, threshold="positive", check_btable=True, btable_scoring=scoring
    )
    btable = report["btable"]
    assert btable["scoring"] == scoring
    assert [c["label"] for c in btable["candidates"]][:4] == ["012", "012fx", "012fy", "012fz"]
    assert btable["best"] == "102"
    assert btable["current_is_best"] is False
    assert btable["best_coherence_index"] == pytest.approx(1.0)
    assert btable["current_coherence_index"] == pytest.approx(report["coherence_index"])
    assert report["coherence_index"] < 0.9


def test_numeric_threshold_excludes_everything_above_it():
    report = _tube_phantom(LPS).coherence(threshold=2.0)
    assert report["threshold_value"] == pytest.approx(2.0)
    assert report["evaluated_voxels"] == 0
    assert report["coherence_index"] is None


def test_bad_arguments_raise():
    ds = _tube_phantom(LPS)
    with pytest.raises(ValueError, match="mode"):
        ds.coherence("voxel")
    with pytest.raises(ValueError, match="btable_scoring"):
        ds.coherence(check_btable=True, btable_scoring="voxel")
    with pytest.raises(ValueError, match="threshold"):
        ds.coherence(threshold="median")
    with pytest.raises(ValueError, match="angle_degrees"):
        ds.coherence(angle_deg=120.0)
    with pytest.raises(ValueError, match="does not exist"):
        ds.coherence(metric="qa")


def test_quantile_is_the_default_threshold():
    ds = _tube_phantom(LPS)
    default = ds.coherence()
    explicit = ds.coherence(threshold="quantile", quantile=0.1)
    assert default["threshold_value"] == explicit["threshold_value"]
    assert "threshold_elasticity" in default
    # All weights are 1, so the 0.1-quantile is 1 and every voxel is kept.
    assert default["evaluated_voxels"] == 36
    with pytest.raises(ValueError, match="quantile"):
        ds.coherence(quantile=1.5)


def test_load_can_skip_dense_arrays(tmp_path):
    mask = np.ones(8, dtype=np.uint8)
    builder = odx.OdxBuilder(np.eye(4), (2, 2, 2), mask)
    sh = np.zeros((2, 2, 2, 6), dtype=np.float32)
    sh[..., 0] = 1.0
    builder.set_sh_coefficients(sh, basis="tournier07", sh_order=2, legacy=False)
    for _ in range(8):
        builder.push_voxel_peaks(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32))
    builder.set_dpf("amplitude", np.ones((8, 1), dtype=np.float32))
    path = tmp_path / "with_sh.odx"
    builder.finalize().save(str(path))

    full = odx.load(str(path))
    lean = odx.load(str(path), skip_odf=True, skip_sh=True)
    assert "coefficients" in full.sh_names()
    assert lean.sh_names() == []
    np.testing.assert_array_equal(lean.directions, full.directions)
    assert lean.coherence(threshold="all")["evaluated_voxels"] == 8


def test_chain_mode_measures_chain_length_and_scores_the_btable():
    # One straight 10-voxel chain along x on a 2 mm LPS grid: nine 2 mm steps.
    mask = np.zeros((12, 5, 5), dtype=np.uint8)
    mask[1:11, 2, 2] = 1
    builder = odx.OdxBuilder(LPS, mask.shape, mask.reshape(-1))
    for _ in range(10):
        builder.push_voxel_peaks(np.asarray([[-1.0, 0.0, 0.0]], dtype=np.float32))
    builder.set_dpf("amplitude", np.ones((10, 1), dtype=np.float32))
    report = builder.finalize().coherence("chain", threshold="positive")
    assert report["mode"] == "chain"
    assert report["chains"] == 1
    assert report["weighted_mean_length_mm"] == pytest.approx(18.0)

    swap_xy = np.array([[0.0, 1, 0], [1, 0, 0], [0, 0, 1]])
    btable = _tube_phantom(LPS, swap_xy).coherence(
        "chain", threshold="positive", check_btable=True, btable_scoring="chain"
    )["btable"]
    assert btable["scoring"] == "chain"
    assert btable["best"] == "102"
