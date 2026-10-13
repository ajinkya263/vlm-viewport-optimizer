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
    BoundingBox,
    InteractableElement,
    SessionStats,
    ViewMode,
    ViewState,
    compact_layout,
)
from .registry import MarkRegistry
from .vision import (
    DeltaEngine,
    SetOfMarkLabeler,
    ViewportCropper,
    estimate_vision_tokens,
    savings_ratio,
)
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
        # Cumulative token-savings accounting (the headline optimization metric).
        self._frames_served = 0
        self._optimized_tokens = 0
        self._naive_tokens = 0

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
    async def view(
        self, mode: ViewMode = ViewMode.MACRO, *, include_layout_text: bool = True
    ) -> ViewState:
        """Capture the current page and return it in the requested representation."""
        async with self._lock:
            self._touch()
            return await self._capture_and_render(mode, include_layout_text)

    async def zoom(self, mark_id: str, *, include_layout_text: bool = True) -> ViewState:
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
            return await self._build_state(
                frame=tile,
                mode=ViewMode.ZOOM,
                elements=[element],
                url=url,
                title=title,
                jpeg_quality=settings.crop_jpeg_quality,
                include_layout_text=include_layout_text,
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
            return await self._capture_and_render(request.return_mode, include_layout_text=True)

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #
    def stats(self) -> SessionStats:
        """Return the cumulative token-savings scorecard for this session."""
        saved = max(0, self._naive_tokens - self._optimized_tokens)
        return SessionStats(
            session_id=self.session_id,
            frames_served=self._frames_served,
            optimized_tokens_total=self._optimized_tokens,
            naive_tokens_total=self._naive_tokens,
            tokens_saved=saved,
            savings_ratio=savings_ratio(self._optimized_tokens, self._naive_tokens),
            model_family=settings.token_model_family,
        )

    # ------------------------------------------------------------------ #
    # Internal capture+render pipeline (lock already held)
    # ------------------------------------------------------------------ #
    async def _capture_and_render(
        self, mode: ViewMode, include_layout_text: bool
    ) -> ViewState:
        frame, elements, url, title = await self._controller.capture()
        self._registry.rebuild(elements)

        # Diff on RAW frames (labels are redrawn each frame and would be noise),
        # but render marks onto whatever we ultimately return.
        delta = await asyncio.to_thread(self._delta.process, frame)
        labeled = await asyncio.to_thread(self._labeler.render, frame, elements)

        if mode is ViewMode.DELTA and delta.changed and delta.region is not None:
            return await self._render_delta(
                labeled, delta.region, elements, url, title, include_layout_text
            )
        if mode is ViewMode.DELTA:
            # Nothing changed (or first frame) — degrade to a full macro overview.
            return await self._render_macro(
                labeled, elements, url, title, include_layout_text,
                note="No significant change detected; returning full macro frame.",
            )
        # MACRO (and ZOOM is handled by .zoom()).
        return await self._render_macro(labeled, elements, url, title, include_layout_text)

    async def _render_macro(
        self,
        labeled: np.ndarray,
        elements: list[InteractableElement],
        url: str,
        title: str,
        include_layout_text: bool,
        note: str = "",
    ) -> ViewState:
        macro = await asyncio.to_thread(self._cropper.macro, labeled)
        return await self._build_state(
            frame=macro.frame,
            mode=ViewMode.MACRO,
            elements=elements,
            url=url,
            title=title,
            jpeg_quality=settings.macro_jpeg_quality,
            include_layout_text=include_layout_text,
            note=note,
        )

    async def _render_delta(
        self,
        labeled: np.ndarray,
        region: BoundingBox,
        elements: list[InteractableElement],
        url: str,
        title: str,
        include_layout_text: bool,
    ) -> ViewState:
        masked = await asyncio.to_thread(self._delta._mask_outside, labeled, region)
        # Downscale the masked frame: this is what makes a delta cheaper in TOKENS
        # (fewer pixels => fewer tiles), not merely cheaper in bytes.
        shrunk = await asyncio.to_thread(
            self._cropper.downscale_to, masked, settings.delta_max_width
        )
        # Only report elements whose center falls inside the changed region — the
        # agent shouldn't try to click something it (deliberately) cannot see.
        rx, ry, rw, rh = region.to_xywh_int()
        visible = [
            el
            for el in elements
            if rx <= el.center[0] <= rx + rw and ry <= el.center[1] <= ry + rh
        ]
        return await self._build_state(
            frame=shrunk.frame,
            mode=ViewMode.DELTA,
            elements=visible,
            url=url,
            title=title,
            jpeg_quality=settings.crop_jpeg_quality,
            include_layout_text=include_layout_text,
            change_region=region,
            note=f"Delta frame: {len(visible)} marks inside the changed region.",
        )

    # ------------------------------------------------------------------ #
    # Shared finalizer: encode, estimate tokens, accumulate savings
    # ------------------------------------------------------------------ #
    async def _build_state(
        self,
        *,
        frame: np.ndarray,
        mode: ViewMode,
        elements: list[InteractableElement],
        url: str,
        title: str,
        jpeg_quality: int,
        include_layout_text: bool,
        change_region: BoundingBox | None = None,
        note: str = "",
    ) -> ViewState:
        b64 = await asyncio.to_thread(
            encode_base64, frame, fmt="jpeg", jpeg_quality=jpeg_quality
        )
        h, w = frame.shape[:2]
        family = settings.token_model_family

        optimized = estimate_vision_tokens(w, h, family=family)
        naive = estimate_vision_tokens(
            settings.viewport_width, settings.viewport_height, family=family
        )
        # Accumulate the session scorecard: every frame we serve is one frame a
        # naive full-resolution loop would also have served.
        self._frames_served += 1
        self._optimized_tokens += optimized
        self._naive_tokens += naive

        return ViewState(
            session_id=self.session_id,
            mode=mode,
            url=url,
            title=title,
            image_base64=b64,
            image_width=w,
            image_height=h,
            elements=elements,
            layout_text=compact_layout(elements) if include_layout_text else None,
            change_region=change_region,
            estimated_vision_tokens=optimized,
            estimated_full_frame_tokens=naive,
            token_savings_ratio=savings_ratio(optimized, naive),
            note=note,
        )
