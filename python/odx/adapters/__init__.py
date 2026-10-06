"""Adapters between ODX and external libraries.

nibabel is a required dependency of `odx` and imported eagerly; dipy stays
optional (its adapter lazy-imports it).
"""

from . import dipy as dipy  # re-export
from . import nibabel as nibabel  # re-export
