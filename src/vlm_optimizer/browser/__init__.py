"""Automation layer: async Playwright backend + DOM extraction."""

from __future__ import annotations

from .controller import BrowserController
from .dom_script import EXTRACT_INTERACTABLES_JS

__all__ = ["BrowserController", "EXTRACT_INTERACTABLES_JS"]
