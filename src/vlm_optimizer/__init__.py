"""VLM-Viewport-Optimizer.

A production-grade browser-automation framework that lets a Vision-Language-Model
agent navigate the web via screenshots while aggressively minimizing vision input
tokens, API cost, and execution latency.

Public pipeline stages:
    * BrowserController  -- async Playwright backend + DOM extraction
    * SetOfMarkLabeler   -- draws compact color-coded ID tags onto elements
    * DeltaEngine        -- differential (delta-frame) patch rendering
    * ViewportCropper    -- hierarchical macro/zoom coordinate pipeline
    * Session            -- orchestrates all stages behind an async lock
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
