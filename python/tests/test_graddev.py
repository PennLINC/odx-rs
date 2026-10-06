"""Gradient-nonlinearity correction (`apply_graddev`) from Python.

Convention under test: the field stores `T` row-major with `g_eff = Tᵀ g`, so a
fixel estimated at `u` truly lies at `normalize(T⁻¹ u)` (see the Rust
`tests/graddev.rs`).
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

import odx

DIMS = (2, 2, 2)
U = np.array([1.0, 0.0, 0.0])


def _rot_z(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _fixel_dataset(direction=U) -> odx.Odx:
    """2x2x2 fully-masked grid, one fixel per voxel along `direction`."""
    mask = np.ones(int(np.prod(DIMS)), dtype=np.uint8)
    builder = odx.OdxBuilder(np.eye(4), DIMS, mask)
    for _ in range(mask.size):
        builder.push_voxel_peaks(np.asarray([direction], dtype=np.float32))
    return builder.finalize()


def _sh_dataset() -> odx.Odx:
    """2x2x2 fully-masked grid with an lmax-2 SH lobe along x in every voxel."""
    mask = np.ones(int(np.prod(DIMS)), dtype=np.uint8)
    builder = odx.OdxBuilder(np.eye(4), DIMS, mask)
    sh = np.zeros(DIMS + (6,), dtype=np.float32)
    sh[..., 0] = 1.0
    sh[..., 3] = -0.3  # (l=2, m=0): flattens z, so the lobe lies in the xy-plane
    sh[..., 5] = 0.4  # (l=2, m=2): picks out x over y
    builder.set_sh_coefficients(sh, basis="tournier07", sh_order=2, legacy=False)
    builder.skip_all_peaks()
    return builder.finalize()


def _uniform_field(t: np.ndarray, flat: bool = True) -> np.ndarray:
    field = np.broadcast_to(t.astype(np.float32), DIMS + (3, 3)).copy()
    return field.reshape(DIMS + (9,)) if flat else field


def _angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.abs((a * b).sum(-1)), 0.0, 1.0)))


@pytest.mark.parametrize("flat", [True, False])
def test_uniform_rotation_maps_fixels_through_t_inverse(flat):
    t = _rot_z(20.0)
    out, report = _fixel_dataset().apply_graddev(_uniform_field(t, flat), affine=np.eye(4), identity="included")
    want = np.linalg.inv(t) @ U
    assert out.nb_peaks == 8
    assert np.all(_angle_deg(out.directions, want[None]) < 1e-2)  # float32 directions
    assert report["nb_corrected"] == 8
    assert report["median_rotation_deg"] == pytest.approx(20.0, abs=1e-3)


def test_identity_field_is_a_no_op():
    ds = _fixel_dataset()
    out, report = ds.apply_graddev(_uniform_field(np.eye(3)), affine=np.eye(4), identity="included")
    np.testing.assert_allclose(out.directions, ds.directions, atol=1e-6)
    assert report["nb_corrected"] == 0


def test_t_minus_i_storage_is_detected():
    # HCP/FSL files store T − I; "auto" must add the identity back.
    t = _rot_z(15.0)
    a, _ = _fixel_dataset().apply_graddev(_uniform_field(t), affine=np.eye(4), identity="included")
    b, rep = _fixel_dataset().apply_graddev(_uniform_field(t - np.eye(3)), affine=np.eye(4))
    assert rep["identity_added"] is True
    np.testing.assert_allclose(b.directions, a.directions, atol=1e-5)


def test_sh_coefficients_are_reoriented():
    t = _rot_z(30.0)
    ds = _sh_dataset()
    out, report = odx.apply_graddev(ds, _uniform_field(t), affine=np.eye(4), identity="included")
    assert "coefficients" in report["sh_arrays"]
    before, after = ds.sh("coefficients"), out.sh("coefficients")
    assert after.shape == before.shape
    # Rotation preserves per-degree power, and the l=0 term is untouched.
    np.testing.assert_allclose(after[:, 0], before[:, 0], atol=1e-5)
    np.testing.assert_allclose(
        np.linalg.norm(after[:, 1:], axis=1), np.linalg.norm(before[:, 1:], axis=1), rtol=1e-3
    )
    assert not np.allclose(after, before, atol=1e-3)


LPS = np.array([[-2.0, 0.0, 0.0, 10.0], [0.0, -2.0, 0.0, 12.0], [0.0, 0.0, 2.0, -8.0], [0.0, 0.0, 0.0, 1.0]])


def _varying_field() -> np.ndarray:
    """A different rotation per voxel, so any flip or permutation of the
    voxel order between input routes changes the result."""
    field = np.zeros(DIMS + (3, 3), dtype=np.float32)
    for i, j, k in np.ndindex(DIMS):
        field[i, j, k] = _rot_z(5.0 + 7.0 * i + 3.0 * j + 11.0 * k)
    return field.reshape(DIMS + (9,))


def _lps_fixel_dataset() -> odx.Odx:
    mask = np.ones(int(np.prod(DIMS)), dtype=np.uint8)
    builder = odx.OdxBuilder(LPS, DIMS, mask)
    for _ in range(mask.size):
        builder.push_voxel_peaks(np.asarray([U], dtype=np.float32))
    return builder.finalize()


def test_path_image_and_array_inputs_agree_on_a_non_ras_grid(tmp_path):
    field = _varying_field()
    path = tmp_path / "graddev.nii.gz"
    nib.Nifti1Image(field, LPS).to_filename(path)
    img = nib.load(path)
    by_path, _ = _lps_fixel_dataset().apply_graddev(str(path), identity="included")
    by_image, rep = _lps_fixel_dataset().apply_graddev(img, identity="included")
    by_array, _ = _lps_fixel_dataset().apply_graddev(field, affine=LPS, identity="included")
    np.testing.assert_allclose(by_image.directions, by_path.directions, atol=1e-6)
    np.testing.assert_allclose(by_array.directions, by_path.directions, atol=1e-6)
    assert rep["nb_corrected"] == 8
    # The voxels really do differ, so agreement is not an accident of uniformity.
    assert np.ptp(by_path.directions[:, 1]) > 0.1


def test_nibabel_image_brings_its_own_affine():
    img = nib.Nifti1Image(_uniform_field(np.eye(3)), np.eye(4))
    with pytest.raises(ValueError, match="carries its own"):
        _fixel_dataset().apply_graddev(img, affine=np.eye(4))


def test_float64_arrays_are_accepted():
    t = _rot_z(10.0)
    out, _ = _fixel_dataset().apply_graddev(
        _uniform_field(t).astype(np.float64), affine=np.eye(4), identity="included"
    )
    assert np.all(_angle_deg(out.directions, (np.linalg.inv(t) @ U)[None]) < 1e-2)


def test_bad_inputs_raise():
    ds = _fixel_dataset()
    field = _uniform_field(np.eye(3))
    with pytest.raises(ValueError, match="affine"):
        ds.apply_graddev(field)
    with pytest.raises(ValueError, match="shape"):
        ds.apply_graddev(field.reshape(DIMS + (3, 3))[..., :2], affine=np.eye(4))
    with pytest.raises(ValueError, match="identity"):
        ds.apply_graddev(field, affine=np.eye(4), identity="maybe")
    with pytest.raises(ValueError, match="only for array input"):
        ds.apply_graddev("graddev.nii.gz", affine=np.eye(4))
