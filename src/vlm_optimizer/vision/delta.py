"""Optimization Layer 1 — Differential patch-rendering (delta frames).

Compares frame N against frame N-1, finds the region that actually changed
(a dropdown opening, a toast appearing, text filling in), and returns a copy of
frame N where everything *outside* that region is blacked out to pure
``[0, 0, 0]``. A VLM patch encoder spends almost no tokens on uniform black,
so the agent's attention — and your bill — collapses onto just the delta.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import settings
from ..models import BoundingBox


@dataclass(slots=True, frozen=True)
class DeltaResult:
    """Outcome of a frame diff.

    ``changed`` is False when nothing material moved (we then fall back to a full
    frame upstream rather than shipping an all-black image).
    """

    changed: bool
    frame: np.ndarray  # BGR; masked when changed, else the untouched current frame
    region: BoundingBox | None


class DeltaEngine:
    """Stateful one-frame-memory differential renderer."""

    def __init__(
        self,
        *,
        threshold: int | None = None,
        min_area_px: int | None = None,
        dilate_px: int | None = None,
    ) -> None:
        self._threshold = threshold if threshold is not None else settings.delta_threshold
        self._min_area = min_area_px if min_area_px is not None else settings.delta_min_area_px
        self._dilate = dilate_px if dilate_px is not None else settings.delta_dilate_px
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        """Forget the previous frame (e.g. after a full navigation)."""
        self._prev = None

    def prime(self, frame: np.ndarray) -> None:
        """Seed the previous-frame memory without producing a diff."""
        self._prev = frame.copy()

    def process(self, frame: np.ndarray) -> DeltaResult:
        """Diff ``frame`` against the stored previous frame, then store it.

        On the very first frame (or after :meth:`reset`) there is nothing to diff
        against, so we report ``changed=False`` and return the full frame.
        """
        prev = self._prev
        self._prev = frame.copy()

        if prev is None or prev.shape != frame.shape:
            return DeltaResult(changed=False, frame=frame, region=None)

        region = self._changed_region(prev, frame)
        if region is None:
            return DeltaResult(changed=False, frame=frame, region=None)

        masked = self._mask_outside(frame, region)
        return DeltaResult(changed=True, frame=masked, region=region)

    # ------------------------------------------------------------------ #
    # Core CV
    # ------------------------------------------------------------------ #
    def _changed_region(
        self, prev: np.ndarray, curr: np.ndarray
    ) -> BoundingBox | None:
        """Return the tight bbox enclosing all meaningful change, or None."""
        diff = cv2.absdiff(prev, curr)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self._threshold, 255, cv2.THRESH_BINARY)

        if self._dilate > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (self._dilate, self._dilate)
            )
            mask = cv2.dilate(mask, kernel, iterations=1)

        if int(cv2.countNonZero(mask)) < self._min_area:
            return None

        # Union of all change contours => one enclosing box. Robust to multiple
        # disjoint updates (e.g. a field value AND a validation message).
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return None
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())

        w = max(x1 - x0 + 1, 1)
        h = max(y1 - y0 + 1, 1)
        if w * h < self._min_area:
            return None

        return BoundingBox(x=float(x0), y=float(y0), width=float(w), height=float(h))

    @staticmethod
    def _mask_outside(frame: np.ndarray, region: BoundingBox) -> np.ndarray:
        """Black out every pixel outside ``region`` (pure zeros)."""
        x, y, w, h = region.to_xywh_int()
        H, W = frame.shape[:2]
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        x2 = max(0, min(x + w, W))
        y2 = max(0, min(y + h, H))

        out = np.zeros_like(frame)
        out[y:y2, x:x2] = frame[y:y2, x:x2]
        return out
