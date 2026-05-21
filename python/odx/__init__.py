"""Python bindings for the ODX format (orientation density function format).

See https://github.com/PennLINC/odx-rs for the underlying Rust crate and
SPECIFICATION.md for the file format.
"""

from . import adapters, spheres
from ._odx import (
    Odx,
    OdxBuilder,
    PeakFinderConfig,
    SpherePeakFinder,
    attach_dpv,
    attach_dpv_from_volume,
    compute_b_matrix,
    convert_sh_basis,
    from_fibgz,
    from_fz,
    from_mapmri,
    from_mrtrix,
    from_pyafq_aodf,
    from_sh_coefficients,
    load,
    peaks_from_sh,
    save,
)

__all__ = [
    "Odx",
    "OdxBuilder",
    "PeakFinderConfig",
    "SpherePeakFinder",
    "adapters",
    "attach_dpv",
    "attach_dpv_from_volume",
    "compute_b_matrix",
    "convert_sh_basis",
    "from_fibgz",
    "from_fz",
    "from_mapmri",
    "from_mrtrix",
    "from_pyafq_aodf",
    "from_sh_coefficients",
    "load",
    "peaks_from_sh",
    "save",
    "spheres",
]


def __getattr__(name):
    if name in ("to_peaks_and_metrics", "from_peaks_and_metrics"):
        from .adapters.dipy import to_peaks_and_metrics, from_peaks_and_metrics

        return {"to_peaks_and_metrics": to_peaks_and_metrics,
                "from_peaks_and_metrics": from_peaks_and_metrics}[name]
    raise AttributeError(f"module 'odx' has no attribute {name!r}")
