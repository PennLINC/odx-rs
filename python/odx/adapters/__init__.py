"""Optional adapters between ODX and external libraries (dipy, nibabel, …).

Importing `odx.adapters` is cheap; individual adapters lazy-import their
external deps so the package stays usable without them installed.
"""

from . import dipy as dipy  # re-export
from . import nibabel as nibabel  # re-export — module import is cheap;
# the actual `import nibabel` only fires when an adapter function is called.
