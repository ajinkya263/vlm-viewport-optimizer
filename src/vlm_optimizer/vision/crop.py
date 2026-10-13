"""Optimization Layer 2 — Hierarchical viewport crop & zoom.

Two-stage coordinate pipeline:

    Stage 1 (macro): downscale the whole labeled viewport to a small, heavily
    compressed overview. Cheap spatial awareness so the agent can pick a target.

    Stage 2 (zoom): given one element's bbox, crop the *original uncompressed*
    frame with a padding margin, then letterbox it into a single 512x512 tile.
    The agent gets maximum fidelity on exactly the sub-region it cares about and
    pays for exactly one vision tile.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import settings
from ..models import BoundingBox


@dataclass(slots=True, frozen=True)
class MacroFrame:
    frame: np.ndarray
    scale: float  # macro_px / original_px (kept so marks could be rescaled if needed)


class ViewportCropper:
    """Produces macro overviews and zoom tiles. Stateless and pure."""

    def __init__(
        self,
        *,
        macro_max_width: int | None = None,
        padding_ratio: float | None = None,
        tile_size: int | None = None,
    ) -> None:
        self._macro_w = macro_max_width if macro_max_width is not None else settings.macro_max_width
        self._padding = padding_ratio if padding_ratio is not None else settings.crop_padding_ratio
        self._tile = tile_size if tile_size is not None else settings.crop_tile_size

    # ------------------------------------------------------------------ #
    # Stage 1 — macro overview
    # ------------------------------------------------------------------ #
    def macro(self, frame: np.ndarray) -> MacroFrame:
        """Downscale ``frame`` to at most ``macro_max_width`` wide (never upscales)."""
        return self.downscale_to(frame, self._macro_w)

    @staticmethod
    def downscale_to(frame: np.ndarray, max_width: int) -> MacroFrame:
        """Downscale ``frame`` so its width is at most ``max_width`` (never upscales).

        Used by both the macro overview and delta frames: fewer pixels is what
        actually lowers vision-token cost under pixel-area / tile pricing.
        """
        h, w = frame.shape[:2]
        if w <= max_width:
            return MacroFrame(frame=frame, scale=1.0)
        scale = max_width / float(w)
        new_size = (max_width, max(1, round(h * scale)))
        small = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        return MacroFrame(frame=small, scale=scale)

    # ------------------------------------------------------------------ #
    # Stage 2 — zoom tile
    # ------------------------------------------------------------------ #
    def zoom(self, frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """Crop ``frame`` around ``bbox`` (+padding) into a ``tile x tile`` square.

        Aspect ratio is preserved by letterboxing onto a black canvas, so the
        agent never sees geometric distortion of the target.
        """
        H, W = frame.shape[:2]
        x, y, w, h = bbox.to_xywh_int()

        pad_x = round(w * self._padding)
        pad_y = round(h * self._padding)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(W, x + w + pad_x)
        y1 = min(H, y + h + pad_y)

        if x1 <= x0 or y1 <= y0:  # Degenerate bbox: fall back to the whole frame.
            x0, y0, x1, y1 = 0, 0, W, H

        crop = frame[y0:y1, x0:x1]
        return self._letterbox_to_tile(crop)

    def _letterbox_to_tile(self, crop: np.ndarray) -> np.ndarray:
        tile = self._tile
        ch, cw = crop.shape[:2]
        if ch == 0 or cw == 0:
            return np.zeros((tile, tile, 3), dtype=np.uint8)

        scale = min(tile / cw, tile / ch)
        new_w = max(1, round(cw * scale))
        new_h = max(1, round(ch * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)

        canvas = np.zeros((tile, tile, 3), dtype=np.uint8)
        ox = (tile - new_w) // 2
        oy = (tile - new_h) // 2
        canvas[oy : oy + new_h, ox : ox + new_w] = resized
        return canvas
