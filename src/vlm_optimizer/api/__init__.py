"""Networking layer: the FastAPI application.

Only ``create_app`` is re-exported here. We deliberately do NOT re-export the
module-level ``app`` instance, because binding the name ``app`` on this package
would shadow the ``app`` submodule and make the ASGI target
``vlm_optimizer.api.app:app`` ambiguous.
"""

from __future__ import annotations

from .app import create_app

__all__ = ["create_app"]
