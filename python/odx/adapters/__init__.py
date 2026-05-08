"""Optional adapters between ODX and external libraries (dipy, …).

Importing `odx.adapters` is cheap; individual adapters lazy-import their
external deps so the package stays usable without them installed.
"""

from . import dipy as dipy  # re-export
