"""Shared image encoding helpers (BGR ndarray -> compressed base64)."""

from __future__ import annotations

import base64
from typing import Literal

import cv2
import numpy as np


def encode_base64(
    frame: np.ndarray,
    *,
    fmt: Literal["jpeg", "png"] = "jpeg",
    jpeg_quality: int = 80,
) -> str:
    """Encode a BGR frame to a bare base64 string (no ``data:`` prefix).

    JPEG is the default because VLM vision encoders tokenize by pixel content,
    not file size — but smaller transfers still cut end-to-end latency, and the
    blacked-out delta frames compress to almost nothing as JPEG.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("encode_base64 expects an HxWx3 BGR frame.")
    if fmt == "jpeg":
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
        ok, buf = cv2.imencode(".jpg", frame, params)
    else:
        ok, buf = cv2.imencode(".png", frame)
    if not ok:  # pragma: no cover - encoder failure is effectively unreachable
        raise RuntimeError(f"cv2.imencode failed for format '{fmt}'.")
    return base64.b64encode(buf.tobytes()).decode("ascii")
