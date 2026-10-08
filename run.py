#!/usr/bin/env python3
"""Dev entrypoint: ``python run.py`` (or ``uvicorn vlm_optimizer.api.app:app``).

Adds ``src`` to the path so the package is importable without an editable install.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import uvicorn  # noqa: E402

from vlm_optimizer.config import settings  # noqa: E402


def main() -> None:
    host = os.environ.get("VLM_HOST", "127.0.0.1")
    port = int(os.environ.get("VLM_PORT", "8000"))
    reload = os.environ.get("VLM_RELOAD", "0") == "1"
    print(
        f"→ VLM-Viewport-Optimizer on http://{host}:{port}  "
        f"(headless={settings.headless}, viewport={settings.viewport_width}x{settings.viewport_height})"
    )
    uvicorn.run(
        "vlm_optimizer.api.app:app",
        host=host,
        port=port,
        reload=reload,
        app_dir="src",
    )


if __name__ == "__main__":
    main()
