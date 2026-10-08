"""Set-of-Mark labeling engine.

Draws a small, color-coded tag carrying each element's short alphanumeric ID in
the top-left corner of the element's bounding box. The marks are what the VLM
reads to choose a target; the registry (not the pixels) is the authority on where
that target actually is.

Implemented with OpenCV for speed — one in-place pass over the frame, no per-mark
PIL object allocation. Colors are keyed by role so the agent gets a free visual
affordance hint (links vs. inputs vs. buttons).
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import settings
from ..models import ElementRole, InteractableElement

# BGR tag background colors, one per normalized role.
_ROLE_COLORS: dict[ElementRole, tuple[int, int, int]] = {
    ElementRole.BUTTON: (60, 76, 231),  # red-ish
    ElementRole.LINK: (219, 152, 52),  # blue
    ElementRole.INPUT: (96, 174, 39),  # green
    ElementRole.TEXTAREA: (96, 174, 39),
    ElementRole.EDITABLE: (96, 174, 39),
    ElementRole.SELECT: (0, 152, 255),  # orange
    ElementRole.CHECKBOX: (182, 89, 155),  # purple
    ElementRole.RADIO: (182, 89, 155),
    ElementRole.OTHER: (120, 120, 120),  # gray
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_OUTLINE = (20, 20, 20)


class SetOfMarkLabeler:
    """Renders ID tags onto a copy of the frame."""

    def __init__(
        self,
        *,
        font_scale: float | None = None,
        thickness: int | None = None,
        pad: int | None = None,
    ) -> None:
        self._font_scale = font_scale if font_scale is not None else settings.label_font_scale
        self._thickness = thickness if thickness is not None else settings.label_thickness
        self._pad = pad if pad is not None else settings.label_pad_px

    def render(
        self, frame: np.ndarray, elements: list[InteractableElement]
    ) -> np.ndarray:
        """Return a labeled copy of ``frame`` (the input is never mutated)."""
        canvas = frame.copy()
        h, w = canvas.shape[:2]
        for el in elements:
            self._draw_one(canvas, el, w, h)
        return canvas

    def _draw_one(
        self, canvas: np.ndarray, el: InteractableElement, w: int, h: int
    ) -> None:
        x, y, bw, bh = el.bbox.to_xywh_int()
        color = _ROLE_COLORS.get(el.role, _ROLE_COLORS[ElementRole.OTHER])

        # Thin element outline so the agent sees the clickable extent.
        x2 = min(x + bw, w - 1)
        y2 = min(y + bh, h - 1)
        cv2.rectangle(canvas, (x, y), (x2, y2), color, 1, lineType=cv2.LINE_AA)

        # Tag geometry.
        (tw, th), base = cv2.getTextSize(
            el.mark_id, _FONT, self._font_scale, self._thickness
        )
        tag_w = tw + 2 * self._pad
        tag_h = th + base + 2 * self._pad

        # Anchor at the element's top-left, but keep the tag fully on-screen.
        tx = int(np.clip(x, 0, max(w - tag_w, 0)))
        ty = int(np.clip(y, 0, max(h - tag_h, 0)))

        cv2.rectangle(canvas, (tx, ty), (tx + tag_w, ty + tag_h), color, -1)
        text_org = (tx + self._pad, ty + th + self._pad)
        # Dark outline then white fill => legible on any tag color.
        cv2.putText(
            canvas, el.mark_id, text_org, _FONT, self._font_scale, _OUTLINE,
            self._thickness + 2, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, el.mark_id, text_org, _FONT, self._font_scale, _WHITE,
            self._thickness, cv2.LINE_AA,
        )
