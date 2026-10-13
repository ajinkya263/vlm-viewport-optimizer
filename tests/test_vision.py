"""Unit tests for the pure image-math layer (no browser required).

    pip install pytest
    pytest tests/test_vision.py -q
"""

from __future__ import annotations

import numpy as np

from vlm_optimizer.models import (
    BoundingBox,
    ElementRole,
    InteractableElement,
    compact_layout,
)
from vlm_optimizer.registry import MarkRegistry, generate_mark_id
from vlm_optimizer.vision import DeltaEngine, SetOfMarkLabeler, ViewportCropper
from vlm_optimizer.vision.codec import encode_base64
from vlm_optimizer.vision.tokens import estimate_vision_tokens, savings_ratio


def _frame(h: int = 400, w: int = 600, fill: int = 30) -> np.ndarray:
    return np.full((h, w, 3), fill, dtype=np.uint8)


def _element(mark: str, x: int, y: int, w: int = 80, h: int = 24) -> InteractableElement:
    bbox = BoundingBox(x=x, y=y, width=w, height=h)
    return InteractableElement(
        mark_id=mark, role=ElementRole.BUTTON, tag="button",
        bbox=bbox, center=bbox.center,
    )


def test_generate_mark_id_sequence() -> None:
    assert generate_mark_id(0) == "A1"
    assert generate_mark_id(98) == "A99"
    assert generate_mark_id(99) == "B1"
    # Bijective base-26 rollover past Z.
    assert generate_mark_id(99 * 26) == "AA1"


def test_registry_resolves_case_insensitively() -> None:
    reg = MarkRegistry()
    el = _element("A1", 10, 10)
    reg.rebuild([el])
    assert reg.resolve("a1").mark_id == "A1"
    assert reg.midpoint("A1") == el.center
    assert "A1" in reg and "ZZ9" not in reg


def test_labeler_is_nondestructive_and_draws() -> None:
    frame = _frame()
    labeler = SetOfMarkLabeler()
    out = labeler.render(frame, [_element("A1", 50, 50)])
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)  # something was drawn
    assert np.array_equal(frame, _frame())  # input untouched


def test_delta_first_frame_has_no_change() -> None:
    engine = DeltaEngine()
    res = engine.process(_frame())
    assert res.changed is False and res.region is None


def test_delta_isolates_and_masks_change() -> None:
    engine = DeltaEngine(min_area_px=50, dilate_px=4)
    base = _frame()
    engine.process(base)  # prime

    moved = base.copy()
    moved[100:140, 200:260] = 255  # a bright box appears
    res = engine.process(moved)

    assert res.changed is True
    assert res.region is not None
    # Everything outside the change region must be pure black.
    x, y, w, h = res.region.to_xywh_int()
    outside = res.frame.copy()
    outside[y : y + h, x : x + w] = 0
    assert int(outside.sum()) == 0
    # The change itself is preserved inside the region.
    assert int(res.frame[100:140, 200:260].sum()) > 0


def test_cropper_macro_downscales_only() -> None:
    cropper = ViewportCropper(macro_max_width=300)
    macro = cropper.macro(_frame(h=400, w=600))
    assert macro.frame.shape[1] == 300
    assert macro.scale == 0.5
    # Never upscales a frame already smaller than the target.
    small = cropper.macro(_frame(h=100, w=200))
    assert small.frame.shape[1] == 200 and small.scale == 1.0


def test_cropper_zoom_is_tile_sized_square() -> None:
    cropper = ViewportCropper(tile_size=512, padding_ratio=0.15)
    tile = cropper.zoom(_frame(), BoundingBox(x=100, y=100, width=80, height=24))
    assert tile.shape == (512, 512, 3)


def test_codec_roundtrip_is_base64() -> None:
    import base64

    b64 = encode_base64(_frame(), fmt="jpeg", jpeg_quality=50)
    assert isinstance(b64, str)
    assert base64.b64decode(b64)  # decodes without error


def test_token_estimate_scales_with_pixels() -> None:
    # Anthropic: ~pixels/750. A downscaled frame must cost strictly fewer tokens.
    full = estimate_vision_tokens(1280, 800, family="anthropic")
    macro = estimate_vision_tokens(768, 480, family="anthropic")
    assert full == 1366 and macro < full
    # OpenAI tile model: fewer pixels => fewer 512px tiles => fewer tokens.
    assert estimate_vision_tokens(512, 512, family="openai") < estimate_vision_tokens(
        1280, 800, family="openai"
    )


def test_savings_ratio_bounds() -> None:
    assert savings_ratio(500, 1000) == 0.5
    assert savings_ratio(0, 1000) == 1.0
    assert savings_ratio(2000, 1000) == 0.0  # clamped, never negative
    assert savings_ratio(10, 0) == 0.0  # no baseline => no claimed savings


def test_compact_layout_is_terse_and_complete() -> None:
    els = [
        _element("A1", 60, 100),
        InteractableElement(
            mark_id="A2", role=ElementRole.LINK, tag="a", name="Next",
            enabled=False, bbox=BoundingBox(x=10, y=10, width=40, height=20),
            center=(30, 20),
        ),
    ]
    text = compact_layout(els)
    lines = text.splitlines()
    assert len(lines) == 2
    assert "A1" in lines[0] and "@" in lines[0]
    assert "[disabled]" in lines[1]  # disabled flag surfaced
    # Far fewer characters than the equivalent JSON dump.
    assert len(text) < len(str([e.model_dump() for e in els]))
