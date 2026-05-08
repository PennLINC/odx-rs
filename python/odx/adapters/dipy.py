"""Adapters between ODX and dipy's PeaksAndMetrics.

Lazy-imports dipy: only loaded when a function here is actually called, so
the main `odx` package stays usable without dipy installed.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from .. import _odx


def _require_dipy():
    try:
        import dipy  # noqa: F401
        from dipy.core.sphere import Sphere
        from dipy.direction.peaks import PeaksAndMetrics
        from dipy.reconst.shm import sh_to_sf_matrix
    except ImportError as e:
        raise ImportError(
            "odx.adapters.dipy requires dipy. Install with `pip install odx[dipy]`."
        ) from e
    return Sphere, PeaksAndMetrics, sh_to_sf_matrix


def to_peaks_and_metrics(
    odx,
    *,
    sphere=None,
    basis: Optional[str] = None,
    compute_peaks: bool = False,
    peak_config=None,
    return_sf_matrix: bool = True,
):
    """Build a dipy ``PeaksAndMetrics`` from an :class:`Odx`.

    Parameters
    ----------
    odx : :class:`odx.Odx`
        Source dataset. Must have peak directions OR (if ``compute_peaks=True``)
        SH coefficients.
    sphere : :class:`dipy.core.sphere.Sphere`, optional
        Discretization sphere for ``peak_indices`` and the SH→SF transform.
        Falls back to a Sphere built from ``odx.sphere_vertices``; required if
        the dataset has no sphere attached.
    basis : str, optional
        Override basis name; defaults to ``odx.dipy_basis_name``.
    compute_peaks : bool, optional
        If ``odx.nb_peaks == 0`` and SH coefficients are present, run the
        Rust peak finder before densifying. Uses Newton-refined sub-vertex
        peaks (more accurate than dipy's discrete ``peak_directions``).
    peak_config : :class:`odx.PeakFinderConfig`, optional
        Config for the peak finder when ``compute_peaks=True``.
    return_sf_matrix : bool, optional
        If True (default) compute and attach ``pam.B``; if False leave it
        None and let downstream code recompute on demand.

    Returns
    -------
    pam : :class:`dipy.direction.peaks.PeaksAndMetrics`
    """
    Sphere, PeaksAndMetrics, sh_to_sf_matrix = _require_dipy()

    # SH-only ODX path: run the Rust peak finder first.
    if odx.nb_peaks == 0:
        if not compute_peaks:
            raise ValueError(
                "Odx has no peaks; pass compute_peaks=True to run the Rust "
                "peak finder on the SH coefficients."
            )
        sph_v = sphere.vertices.astype(np.float32) if sphere is not None else None
        sph_f = (
            getattr(sphere, "faces", None).astype(np.uint32)
            if sphere is not None and hasattr(sphere, "faces")
            else None
        )
        odx = odx.with_peaks_from_sh(
            sphere_vertices=sph_v, sphere_faces=sph_f, config=peak_config
        )

    # Resolve sphere: caller-supplied → odx.sphere_vertices → error.
    if sphere is None:
        sphere_vertices = odx.sphere_vertices
        if sphere_vertices is None:
            raise ValueError(
                "to_peaks_and_metrics: no sphere supplied and Odx has no "
                "sphere_vertices. Pass sphere=... explicitly."
            )
        sphere = Sphere(xyz=sphere_vertices.astype(np.float64))
        if odx.sphere_faces is not None and hasattr(sphere, "faces"):
            # dipy's Sphere typically derives faces from xyz; only set if API allows.
            pass

    pam = PeaksAndMetrics()
    pam.affine = np.asarray(odx.affine, dtype=np.float64)
    pam.peak_dirs = odx.densify_directions().astype(np.float32, copy=False)
    pam.peak_values = odx.densify_dpf("amplitude").astype(np.float32, copy=False)
    pam.sphere = sphere

    # peak_indices: nearest-vertex against the resolved sphere.
    sphere_xyz = np.ascontiguousarray(sphere.vertices, dtype=np.float32)
    indices_flat = odx.peak_indices_for(sphere_xyz, antipodal=True)
    n_max = pam.peak_dirs.shape[3]
    dims = pam.peak_dirs.shape[:3]
    pam.peak_indices = np.zeros(dims + (n_max,), dtype=np.int32)
    offsets = np.asarray(odx.offsets)
    mask = np.asarray(odx.mask)
    # Walk masked voxels in compact order to scatter peak indices.
    if odx.nb_peaks > 0:
        # Build (i,j,k) for each masked voxel in C order.
        ijk = np.argwhere(mask.reshape(dims) > 0)
        for row, (i, j, k) in enumerate(ijk):
            start = int(offsets[row])
            count = int(offsets[row + 1] - offsets[row])
            for p in range(count):
                if p >= n_max:
                    break
                pam.peak_indices[i, j, k, p] = int(indices_flat[start + p])

    # PAM-only metadata that round-trips through ODX header.
    pam_meta = odx.pam_metadata
    if pam_meta is not None:
        if "total_weight" in pam_meta:
            pam.total_weight = float(pam_meta["total_weight"])
        if "ang_thr" in pam_meta:
            pam.ang_thr = float(pam_meta["ang_thr"])

    # Optional fields.
    if "gfa" in odx.dpv_names():
        pam.gfa = odx.densify_dpv("gfa").astype(np.float32, copy=False)
    if "qa" in odx.dpf_names():
        pam.qa = odx.densify_dpf("qa").astype(np.float32, copy=False)
    if "coefficients" in odx.sh_names():
        pam.shm_coeff = odx.densify_sh("coefficients").astype(np.float32, copy=False)
    if "amplitudes" in odx.odf_names():
        pam.odf = odx.densify_odf("amplitudes").astype(np.float32, copy=False)

    # B matrix (SH → SF). Prefer dipy's helper for consistency with other
    # downstream code; fall back to the Rust implementation if dipy's
    # signature has drifted.
    if return_sf_matrix and pam.shm_coeff is not None and odx.sh_order is not None:
        basis_name = basis or odx.dipy_basis_name
        if basis_name is None:
            warnings.warn(
                "to_peaks_and_metrics: SH basis unknown; B matrix not computed."
            )
        else:
            try:
                B, _ = sh_to_sf_matrix(
                    sphere,
                    int(odx.sh_order),
                    basis_type=_strip_legacy(basis_name),
                    legacy=basis_name.endswith("_legacy"),
                )
                pam.B = np.asarray(B, dtype=np.float32)
            except (TypeError, AttributeError):
                # dipy version mismatch or unsupported basis_type — fall
                # back to the Rust transform builder.
                pam.B = _odx.compute_b_matrix(
                    np.ascontiguousarray(sphere.vertices, dtype=np.float32),
                    int(odx.sh_order),
                    basis=basis_name,
                    full_basis=bool(odx.sh_full_basis),
                )

    # Ensure tracking-required dtypes.
    pam.peak_dirs = np.ascontiguousarray(pam.peak_dirs, dtype=np.float32)
    pam.peak_values = np.ascontiguousarray(pam.peak_values, dtype=np.float32)
    pam.peak_indices = np.ascontiguousarray(pam.peak_indices, dtype=np.int32)
    return pam


def from_peaks_and_metrics(pam, *, basis: str = "descoteaux07", legacy: bool = True):
    """Build an :class:`Odx` from a dipy ``PeaksAndMetrics``.

    Notes
    -----
    The mask is derived from ``pam.peak_values.any(axis=-1) > 0``. Voxels
    that have only zero-magnitude peaks in PAM are dropped — this mirrors
    the existing odx-rs PAM5 importer.
    """
    _ = _require_dipy()  # validate dipy importable; error early.

    peak_dirs = np.asarray(pam.peak_dirs, dtype=np.float32)
    peak_values = np.asarray(pam.peak_values, dtype=np.float32)
    if peak_dirs.ndim != 5 or peak_dirs.shape[-1] != 3:
        raise ValueError(
            f"pam.peak_dirs must be (X,Y,Z,N,3); got {peak_dirs.shape}"
        )
    if peak_values.ndim != 4:
        raise ValueError(f"pam.peak_values must be (X,Y,Z,N); got {peak_values.shape}")

    mask_3d = (peak_values > 0).any(axis=-1)
    mask_flat = np.ascontiguousarray(mask_3d.astype(np.uint8))
    dims = mask_3d.shape
    affine = np.asarray(pam.affine, dtype=np.float64)
    if affine.shape != (4, 4):
        raise ValueError("pam.affine must be (4, 4)")

    builder = _odx.OdxBuilder(np.ascontiguousarray(affine), dims, mask_flat.ravel())

    # Push peaks per masked voxel in C order to match compact_to_ijk.
    coords = np.argwhere(mask_3d)
    amplitudes: list[float] = []
    qa_per_peak: list[float] = []
    has_qa = getattr(pam, "qa", None) is not None
    qa_arr = np.asarray(pam.qa) if has_qa else None
    for (i, j, k) in coords:
        valid = peak_values[i, j, k] > 0
        n_valid = int(valid.sum())
        peaks = peak_dirs[i, j, k][:n_valid].astype(np.float32, copy=False)
        builder.push_voxel_peaks(np.ascontiguousarray(peaks))
        amplitudes.extend(peak_values[i, j, k][valid].tolist())
        if has_qa:
            qa_per_peak.extend(qa_arr[i, j, k][valid].tolist())

    # Sphere (vertices only — PAM5 doesn't carry faces).
    if getattr(pam, "sphere", None) is not None:
        v = np.ascontiguousarray(pam.sphere.vertices, dtype=np.float32)
        f = (
            np.ascontiguousarray(pam.sphere.faces, dtype=np.uint32)
            if hasattr(pam.sphere, "faces")
            else np.empty((0, 3), dtype=np.uint32)
        )
        if f.size > 0:
            builder.set_sphere(v, f)
        else:
            warnings.warn(
                "from_peaks_and_metrics: PAM sphere has no faces; sphere "
                "round-trip will lose mesh topology."
            )

    # SH info if present.
    shm = getattr(pam, "shm_coeff", None)
    if shm is not None and shm.size > 0:
        sh4d = np.ascontiguousarray(shm.astype(np.float32, copy=False))
        ncoeffs = sh4d.shape[-1]
        # Determine sh_order from ncoeffs and basis. Symmetric: K = (l+1)(l+2)/2.
        sh_order = _sh_order_from_ncoeffs(ncoeffs)
        builder.set_sh_coefficients(
            sh4d, basis=basis, sh_order=sh_order, legacy=legacy, full_basis=False
        )

    # Per-fixel amplitude (the canonical "amplitude" DPF).
    if amplitudes:
        builder.set_dpf("amplitude", np.asarray(amplitudes, dtype=np.float32))
    if has_qa and qa_per_peak:
        builder.set_dpf("qa", np.asarray(qa_per_peak, dtype=np.float32))

    # PAM-only metadata that must round-trip (total_weight, ang_thr, and the
    # basis we wrote under).
    tw = getattr(pam, "total_weight", None)
    at = getattr(pam, "ang_thr", None)
    basis_name = "descoteaux07_legacy" if (basis == "descoteaux07" and legacy) else basis
    if tw is not None or at is not None or basis_name:
        builder.set_pam_metadata(
            total_weight=float(tw) if tw is not None else None,
            ang_thr=float(at) if at is not None else None,
            basis_assumed=basis_name,
        )

    # Per-voxel scalars.
    gfa = getattr(pam, "gfa", None)
    if gfa is not None:
        # Flatten to compact (NB_VOXELS,) order matching the mask iteration.
        gfa_flat = np.asarray(gfa)[mask_3d].astype(np.float32, copy=False)
        builder.set_dpv("gfa", np.ascontiguousarray(gfa_flat))

    # ODF amplitudes (optional).
    odf = getattr(pam, "odf", None)
    if odf is not None and odf.size > 0:
        odf_flat = np.asarray(odf)[mask_3d].astype(np.float32, copy=False)
        builder.set_odf("amplitudes", np.ascontiguousarray(odf_flat))

    return builder.finalize()


# ─── helpers ─────────────────────────────────────────────────────────────────


def _strip_legacy(name: str) -> str:
    return name[: -len("_legacy")] if name.endswith("_legacy") else name


def _sh_order_from_ncoeffs(ncoeffs: int) -> int:
    """Symmetric-basis ncoeffs → lmax. K = (l+1)(l+2)/2 → l = (-3 + sqrt(8K+1)) / 2."""
    import math

    disc = 8 * ncoeffs + 1
    root = int(math.isqrt(disc))
    if root * root != disc:
        raise ValueError(f"{ncoeffs} is not a valid symmetric-SH coefficient count")
    lmax = (root - 3) // 2
    if (lmax + 1) * (lmax + 2) // 2 != ncoeffs:
        raise ValueError(f"{ncoeffs} is not a valid symmetric-SH coefficient count")
    return lmax
