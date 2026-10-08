"""Image-math layer: Set-of-Mark labeling, delta frames, and crop/zoom.

Pure functions over ``np.ndarray`` (BGR). No browser or network imports here,
so this layer is independently testable and reusable.
"""

from __future__ import annotations

from .codec import encode_base64
from .crop import ViewportCropper
from .delta import DeltaEngine, DeltaResult
from .labeler import SetOfMarkLabeler

__all__ = [
    "SetOfMarkLabeler",
    "DeltaEngine",
    "DeltaResult",
    "ViewportCropper",
    "encode_base64",
]
