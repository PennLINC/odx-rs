"""Built-in sphere geometries for peak finding and downstream interop."""

from __future__ import annotations

from . import _odx


def dsistudio_odf8():
    """DSI Studio ODF8 hemisphere: 321 vertices + faces (uint32 triangle list).

    Returns
    -------
    vertices : np.ndarray, shape (321, 3), float32
    faces : np.ndarray, shape (F, 3), uint32
    """
    return _odx.dsistudio_odf8_hemisphere()


def dsistudio_odf8_full():
    """DSI Studio ODF8 full sphere: 642 vertices (float32). No faces — used
    internally for FZ quantization measurements.
    """
    return _odx.dsistudio_odf8_full_sphere()
