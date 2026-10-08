"""Async Playwright backend.

Owns the Chromium lifecycle for a single page and exposes the two primitives the
rest of the framework needs:

    * :meth:`capture`      -- screenshot (as a BGR ``np.ndarray``) + extracted DOM
    * :meth:`dispatch`     -- real click / type / scroll / navigation events

All public coroutines are non-blocking and funnel Playwright's own errors into
our typed :mod:`.errors` hierarchy so the API layer can map them to HTTP codes.
Playwright already runs the browser out-of-process, so CPU-bound image math done
by the caller never blocks this controller's event loop.
"""

from __future__ import annotations

import numpy as np
from playwright.async_api import (
    Browser,
    BrowserContext,
    Error as PlaywrightError,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

from ..config import settings
from ..errors import (
    BrowserNotStartedError,
    BrowserTimeoutError,
    InvalidActionError,
    StaleElementError,
)
from ..models import BoundingBox, ElementRole, InteractableElement
from ..registry import generate_mark_id
from .dom_script import EXTRACT_INTERACTABLES_JS

try:  # OpenCV is optional only for the type checker; required at runtime.
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("opencv-python-headless is required by the browser controller") from exc


class BrowserController:
    """Thin async wrapper around one Playwright page."""

    def __init__(self, *, headless: bool | None = None) -> None:
        self._headless = settings.headless if headless is None else headless
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def start(self, start_url: str = "about:blank") -> str:
        """Launch Chromium and open the first page. Returns the resolved URL."""
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=self._headless,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        self._context = await self._browser.new_context(
            viewport={"width": settings.viewport_width, "height": settings.viewport_height},
            device_scale_factor=settings.device_scale_factor,
            user_agent=settings.user_agent,
        )
        self._context.set_default_timeout(settings.action_timeout_ms)
        self._context.set_default_navigation_timeout(settings.nav_timeout_ms)
        self._page = await self._context.new_page()
        if start_url and start_url != "about:blank":
            await self.goto(start_url)
        return self._page.url

    async def close(self) -> None:
        """Best-effort teardown; never raises."""
        for closer in (
            getattr(self._context, "close", None),
            getattr(self._browser, "close", None),
            getattr(self._pw, "stop", None),
        ):
            if closer is None:
                continue
            try:
                await closer()
            except Exception:  # noqa: BLE001 - teardown must not mask the original flow
                pass
        self._page = self._context = self._browser = self._pw = None

    @property
    def page(self) -> Page:
        if self._page is None:
            raise BrowserNotStartedError("Browser session has not been started.")
        return self._page

    # ------------------------------------------------------------------ #
    # Capture
    # ------------------------------------------------------------------ #
    async def capture(self) -> tuple[np.ndarray, list[InteractableElement], str, str]:
        """Return ``(bgr_frame, elements, url, title)`` for the current page.

        Screenshot bytes are decoded straight into an OpenCV BGR array so the
        vision layer needs no further conversion.
        """
        page = self.page
        await self._settle()
        try:
            png_bytes = await page.screenshot(type="png", full_page=False)
        except PlaywrightTimeoutError as exc:
            raise BrowserTimeoutError("Screenshot timed out.") from exc
        except PlaywrightError as exc:
            raise StaleElementError(f"Screenshot failed: {exc}") from exc

        frame = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:  # pragma: no cover - would require a corrupt encoder
            raise BrowserTimeoutError("Failed to decode screenshot bytes.")

        elements = await self._extract_elements()
        return frame, elements, page.url, await self._safe_title()

    async def _extract_elements(self) -> list[InteractableElement]:
        """Run the DOM script and normalize raw dicts into typed elements."""
        try:
            raw: list[dict] = await self.page.evaluate(EXTRACT_INTERACTABLES_JS)
        except PlaywrightTimeoutError as exc:
            raise BrowserTimeoutError("DOM extraction timed out.") from exc
        except PlaywrightError as exc:
            # A navigation mid-evaluate detaches the execution context; treat as transient.
            raise StaleElementError(f"DOM extraction failed (page changed): {exc}") from exc

        elements: list[InteractableElement] = []
        for index, item in enumerate(raw):
            try:
                bbox = BoundingBox(
                    x=max(item["x"], 0.0),
                    y=max(item["y"], 0.0),
                    width=max(item["width"], 1.0),
                    height=max(item["height"], 1.0),
                )
            except (KeyError, ValueError):
                continue  # Skip malformed rows rather than failing the whole capture.
            role = _coerce_role(item.get("role", "other"))
            elements.append(
                InteractableElement(
                    mark_id=generate_mark_id(index),
                    role=role,
                    tag=str(item.get("tag", "")),
                    name=str(item.get("name", "")),
                    value=str(item.get("value", "")),
                    bbox=bbox,
                    center=bbox.center,
                    enabled=bool(item.get("enabled", True)),
                )
            )
        return elements

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #
    async def dispatch(
        self,
        *,
        action: str,
        center: tuple[float, float] | None = None,
        text: str | None = None,
        key: str | None = None,
        url: str | None = None,
        dx: int = 0,
        dy: int = 0,
        clear_first: bool = True,
    ) -> None:
        """Execute a physical interaction at absolute viewport coordinates.

        We deliberately act by coordinate (not by selector) so behavior matches
        exactly what the VLM saw and clicked on the labeled frame.
        """
        page = self.page
        try:
            match action:
                case "goto":
                    if not url:
                        raise InvalidActionError("GOTO requires a 'url'.")
                    await self.goto(url)
                case "scroll":
                    await page.mouse.wheel(dx, dy)
                case "click" | "double_click" | "hover" | "type" | "press":
                    if center is None:
                        raise InvalidActionError(f"'{action}' requires a target mark_id.")
                    x, y = center
                    await self._act_at_point(page, action, x, y, text, key, clear_first)
                case _:
                    raise InvalidActionError(f"Unsupported action '{action}'.")
        except PlaywrightTimeoutError as exc:
            raise BrowserTimeoutError(f"Action '{action}' timed out.") from exc
        except PlaywrightError as exc:
            raise StaleElementError(f"Action '{action}' failed on a stale/blocked node: {exc}") from exc

    async def _act_at_point(
        self,
        page: Page,
        action: str,
        x: float,
        y: float,
        text: str | None,
        key: str | None,
        clear_first: bool,
    ) -> None:
        await page.mouse.move(x, y)
        match action:
            case "hover":
                return
            case "click":
                await page.mouse.click(x, y)
            case "double_click":
                await page.mouse.dblclick(x, y)
            case "press":
                if not key:
                    raise InvalidActionError("PRESS requires a 'key'.")
                await page.mouse.click(x, y)
                await page.keyboard.press(key)
            case "type":
                if text is None:
                    raise InvalidActionError("TYPE requires 'text'.")
                await page.mouse.click(x, y)
                if clear_first:
                    await page.keyboard.press("ControlOrMeta+A")
                    await page.keyboard.press("Delete")
                await page.keyboard.type(text, delay=12)

    async def goto(self, url: str) -> None:
        try:
            await self.page.goto(url, wait_until="domcontentloaded")
        except PlaywrightTimeoutError as exc:
            raise BrowserTimeoutError(f"Navigation to {url} timed out.") from exc
        except PlaywrightError as exc:
            raise InvalidActionError(f"Navigation to {url} failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    async def _settle(self) -> None:
        """Give the page a bounded grace period to quiesce after an action.

        We never hard-fail on this: a page that keeps a long-lived connection
        open (websockets, polling) would otherwise block every capture forever.
        """
        if settings.settle_timeout_ms <= 0:
            return
        try:
            await self.page.wait_for_load_state(
                "networkidle", timeout=settings.settle_timeout_ms
            )
        except PlaywrightTimeoutError:
            pass  # Expected for chatty pages — capture what we have.
        except PlaywrightError:
            pass

    async def _safe_title(self) -> str:
        try:
            return await self.page.title()
        except PlaywrightError:
            return ""


def _coerce_role(raw: str) -> ElementRole:
    try:
        return ElementRole(raw)
    except ValueError:
        return ElementRole.OTHER
