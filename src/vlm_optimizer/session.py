"""Session orchestration — the seam where all three layers meet.

A :class:`Session` owns one browser page plus the per-session vision state
(previous frame for the delta engine) and the isolated mark registry. Every
public coroutine is guarded by a single ``asyncio.Lock`` so overlapping HTTP
requests for the same session can never interleave a capture with an action
and corrupt the frame memory or the registry (the core race condition).

The heavy image math runs in a worker thread via ``asyncio.to_thread`` so it
never blocks the event loop that Playwright and FastAPI share.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import numpy as np

from .browser import BrowserController
from .config import settings
from .models import (
    ActionRequest,
    ActionType,
    InteractableElement,
    ViewMode,
    ViewState,
)
from .registry import MarkRegistry
from .vision import DeltaEngine, SetOfMarkLabeler, ViewportCropper
from .vision.codec import encode_base64


class Session:
    """A single agent browsing session."""

    def __init__(self, *, headless: bool | None = None) -> None:
        self.session_id: str = uuid.uuid4().hex
        self._controller = BrowserController(headless=headless)
        self._registry = MarkRegistry()
        self._delta = DeltaEngine()
        self._labeler = SetOfMarkLabeler()
        self._cropper = ViewportCropper()
        self._lock = asyncio.Lock()
        self._last_used: float = time.monotonic()
        self._started = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def start(self, start_url: str) -> str:
        async with self._lock:
            url = await self._controller.start(start_url)
            self._started = True
            self._touch()
            return url

    async def close(self) -> None:
        async with self._lock:
            await self._controller.close()
            self._delta.reset()
            self._started = False

    @property
    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_used

    @property
    def current_url(self) -> str:
        return self._controller.page.url

    def _touch(self) -> None:
        self._last_used = time.monotonic()

    # ------------------------------------------------------------------ #
    # View
    # ------------------------------------------------------------------ #
    async def view(self, mode: ViewMode = ViewMode.MACRO) -> ViewState:
        """Capture the current page and return it in the requested representation."""
        async with self._lock:
            self._touch()
            return await self._capture_and_render(mode)

    async def zoom(self, mark_id: str) -> ViewState:
        """Return a single 512x512 zoom tile centered on one element.

        Resolves the mark against the registry captured on the *last* view, so a
        zoom never forces a full re-capture (and never shifts coordinates).
        """
        async with self._lock:
            self._touch()
            element = self._registry.resolve(mark_id)
            frame, elements, url, title = await self._controller.capture()
            # Keep the delta memory coherent even though we're returning a zoom.
            await asyncio.to_thread(self._delta.prime, frame)
            self._registry.rebuild(elements)

            tile = await asyncio.to_thread(self._cropper.zoom, frame, element.bbox)
            b64 = await asyncio.to_thread(
                encode_base64, tile, fmt="jpeg", jpeg_quality=settings.crop_jpeg_quality
            )
            h, w = tile.shape[:2]
            return ViewState(
                session_id=self.session_id,
                mode=ViewMode.ZOOM,
                url=url,
                title=title,
                image_base64=b64,
                image_width=w,
                image_height=h,
                elements=[element],
                note=f"Zoom tile for mark '{element.mark_id}'.",
            )

    # ------------------------------------------------------------------ #
    # Action
    # ------------------------------------------------------------------ #
    async def act(self, request: ActionRequest) -> ViewState:
        """Dispatch one action, then return the next optimized state."""
        async with self._lock:
            self._touch()

            center: tuple[float, float] | None = None
            if request.mark_id is not None:
                center = self._registry.resolve(request.mark_id).center

            # A navigation invalidates the entire previous frame; drop delta memory.
            if request.action in (ActionType.GOTO,):
                self._delta.reset()

            await self._controller.dispatch(
                action=request.action.value,
                center=center,
                text=request.text,
                key=request.key,
                url=request.url,
                dx=request.dx,
                dy=request.dy,
                clear_first=request.clear_first,
            )
            return await self._capture_and_render(request.return_mode)

    # ------------------------------------------------------------------ #
    # Internal capture+render pipeline (lock already held)
    # ------------------------------------------------------------------ #
    async def _capture_and_render(self, mode: ViewMode) -> ViewState:
        frame, elements, url, title = await self._controller.capture()
        self._registry.rebuild(elements)

        # Diff on RAW frames (labels are redrawn each frame and would be noise),
        # but render marks onto whatever we ultimately return.
        delta = await asyncio.to_thread(self._delta.process, frame)
        labeled = await asyncio.to_thread(self._labeler.render, frame, elements)

        if mode is ViewMode.DELTA and delta.changed and delta.region is not None:
            return await self._render_delta(labeled, delta.region, elements, url, title)
        if mode is ViewMode.DELTA:
            # Nothing changed (or first frame) — degrade to a full macro overview.
            return await self._render_macro(
                labeled, elements, url, title,
                note="No significant change detected; returning full macro frame.",
            )
        # MACRO (and ZOOM is handled by .zoom()).
        return await self._render_macro(labeled, elements, url, title)

    async def _render_macro(
        self,
        labeled: np.ndarray,
        elements: list[InteractableElement],
        url: str,
        title: str,
        note: str = "",
    ) -> ViewState:
        macro = await asyncio.to_thread(self._cropper.macro, labeled)
        b64 = await asyncio.to_thread(
            encode_base64, macro.frame, fmt="jpeg", jpeg_quality=settings.macro_jpeg_quality
        )
        h, w = macro.frame.shape[:2]
        return ViewState(
            session_id=self.session_id,
            mode=ViewMode.MACRO,
            url=url,
            title=title,
            image_base64=b64,
            image_width=w,
            image_height=h,
            elements=elements,
            note=note,
        )

    async def _render_delta(
        self,
        labeled: np.ndarray,
        region,
        elements: list[InteractableElement],
        url: str,
        title: str,
    ) -> ViewState:
        masked = await asyncio.to_thread(self._delta._mask_outside, labeled, region)
        b64 = await asyncio.to_thread(
            encode_base64, masked, fmt="jpeg", jpeg_quality=settings.crop_jpeg_quality
        )
        h, w = masked.shape[:2]
        # Only report elements whose center falls inside the changed region — the
        # agent shouldn't try to click something it (deliberately) cannot see.
        rx, ry, rw, rh = region.to_xywh_int()
        visible = [
            el
            for el in elements
            if rx <= el.center[0] <= rx + rw and ry <= el.center[1] <= ry + rh
        ]
        return ViewState(
            session_id=self.session_id,
            mode=ViewMode.DELTA,
            url=url,
            title=title,
            image_base64=b64,
            image_width=w,
            image_height=h,
            elements=visible,
            change_region=region,
            note=f"Delta frame: {len(visible)} marks inside the changed region.",
        )
