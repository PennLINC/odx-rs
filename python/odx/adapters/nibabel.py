"""Adapter between ODX and nibabel.

Round-trip per-voxel scalars (DPVs) between an `odx.Odx` and nibabel's
`Nifti1Image`:

* :func:`to_nifti1_image` — read a DPV, scatter onto the full grid, return
  a `Nifti1Image` with the ODX affine baked into both qform (active) and
  sform (data preserved but flagged Unknown). The data slot is the same
  policy used by the Rust writer, so the on-disk file you'd get from
  ``img.to_filename(...)`` matches what ``odx attach-dpv`` would produce
  in the other direction.

* :func:`save_dpv_nifti` — convenience: ``to_nifti1_image(...).to_filename(path)``.

* :func:`attach_dpv` — *import* side. Takes a path, a `Nifti1Image`, or any
  spatial image with ``.get_fdata()`` + ``.affine`` (e.g. nilearn images),
  validates the grid against the ODX, and appends a DPV in place.

Lazy-imports nibabel so the main `odx` package stays usable without it
installed; for installation, ``pip install odx[nibabel]``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from .. import _odx

if TYPE_CHECKING:
    import nibabel as nib

    NiftiLike = Union[nib.Nifti1Image, nib.Nifti2Image, "os.PathLike[str]", str]
else:
    NiftiLike = object  # purely a type-hint sentinel at runtime


def _require_nibabel():
    try:
        import nibabel as nib  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "odx.adapters.nibabel requires nibabel. "
            "Install with `pip install odx[nibabel]`."
        ) from e
    return nib


# ─── export: DPV → Nifti1Image ─────────────────────────────────────────────


def to_nifti1_image(
    odx: "_odx.Odx",
    dpv_name: str,
    *,
    dtype: Optional[Union[str, np.dtype]] = None,
) -> "nib.Nifti1Image":
    """Scatter a DPV onto the ODX grid and return a `Nifti1Image`.

    Parameters
    ----------
    odx : Odx
        Loaded ODX (from ``odx.load(...)``).
    dpv_name : str
        Name of the DPV to extract (must appear in ``odx.dpv_names``).
    dtype : str or numpy dtype, optional
        If given, the volume is cast to this dtype before being wrapped.
        Default: keep the DPV's promoted ``float32`` dtype.

    Returns
    -------
    nibabel.Nifti1Image
        A 3-D image with `qform_code = 1 (ScannerAnat)` and
        `sform_code = 0 (Unknown)` — see the module docstring for the
        rationale. The affine in both slots is the ODX's ``voxel_to_rasmm``
        unchanged.
    """
    nib = _require_nibabel()

    available = list(odx.dpv_names())
    if dpv_name not in available:
        raise KeyError(
            f"no DPV {dpv_name!r} in ODX; available: {available}"
        )

    # densify_dpv returns (X, Y, Z) float32 with masked-out voxels = 0,
    # already in the orientation the affine expects.
    vol = odx.densify_dpv(dpv_name)
    if dtype is not None:
        vol = vol.astype(dtype, copy=False)

    affine = odx.affine  # (4, 4) float64
    img = nib.Nifti1Image(vol, affine)
    hdr = img.header
    # Mirror the Rust writer's policy exactly:
    #   qform_code = 1 (ScannerAnat) — primary
    #   sform_code = 0 (Unknown)    — data on disk but inactive
    #   xyzt_units = 2              — mm
    hdr.set_qform(affine, code=1)
    hdr.set_sform(affine, code=0)
    hdr["xyzt_units"] = 2
    return img


def save_dpv_nifti(
    odx: "_odx.Odx",
    dpv_name: str,
    path: Union[str, "os.PathLike[str]"],
    *,
    dtype: Optional[Union[str, np.dtype]] = None,
) -> None:
    """Write a DPV to ``path`` as a NIfTI-1 (`.nii` or `.nii.gz`).

    Equivalent to ``to_nifti1_image(odx, dpv_name, dtype=dtype).to_filename(path)``.
    """
    img = to_nifti1_image(odx, dpv_name, dtype=dtype)
    img.to_filename(os.fspath(path))


# ─── import: NIfTI / Nifti1Image → DPV ────────────────────────────────────


def attach_dpv(
    odx_path: Union[str, "os.PathLike[str]"],
    name: str,
    image: NiftiLike,
    *,
    dtype: Optional[str] = None,
) -> dict:
    """Append a DPV to an existing ODX from a NIfTI image or path.

    Parameters
    ----------
    odx_path : path
        Path to an ODX directory or ``.odx`` archive. Edited in place.
    name : str
        Name to register the DPV under (e.g. ``"fa"``). An existing DPV
        with the same name is overwritten.
    image : Nifti1Image, Nifti2Image, path, or path-like
        Source volume. If a path/str is given, it's loaded via
        ``nibabel.load``. Any object with ``.get_fdata()`` and ``.affine``
        is also accepted.
    dtype : str, optional
        On-disk DPV dtype. One of ``"auto"``, ``"u8"``/``"uint8"``,
        ``"u16"``, ``"u32"``, ``"i16"``, ``"i32"``, ``"f32"``, ``"f64"``.
        Default ``"auto"`` — picks the narrowest unsigned int that fits
        non-negative integer data, else ``float32``.

    Returns
    -------
    dict
        ``{name, dtype, nb_voxels, masked_in_count, clamped}`` — same shape
        as the report from :func:`odx.attach_dpv_from_volume`. ``clamped``
        is ``True`` iff an explicit ``dtype`` lost precision or range.

    Raises
    ------
    ValueError
        If the input grid (dimensions or affine) does not match the ODX
        within 1e-3 mm. Resample the input onto the ODX grid first.
    """
    nib = _require_nibabel()

    # Resolve the image: path → nibabel.load(...); image-like → use as-is.
    if isinstance(image, (str, os.PathLike)):
        img = nib.load(os.fspath(image))
    else:
        img = image  # duck-typed: must have .get_fdata() + .affine

    if not hasattr(img, "get_fdata") or not hasattr(img, "affine"):
        raise TypeError(
            f"attach_dpv: 'image' must be a path or a NIfTI-like object "
            f"with .get_fdata() and .affine; got {type(img).__name__}"
        )

    vol = np.asarray(img.get_fdata(dtype=np.float64), dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError(
            f"attach_dpv: expected a 3-D volume, got shape {vol.shape!r}"
        )
    affine = np.asarray(img.affine, dtype=np.float64)
    if affine.shape != (4, 4):
        raise ValueError(
            f"attach_dpv: expected (4, 4) affine, got shape {affine.shape!r}"
        )

    # Ensure C-contiguous f64 array — PyO3 readonly views require it.
    vol_c = np.ascontiguousarray(vol, dtype=np.float64)
    affine_c = np.ascontiguousarray(affine, dtype=np.float64)

    return _odx.attach_dpv_from_volume(
        os.fspath(odx_path), name, vol_c, affine_c, dtype=dtype
    )
