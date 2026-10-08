"""FastAPI surface for the VLM-Viewport-Optimizer.

This module is *only* networking: it validates requests, delegates to the
session layer, and maps our typed errors onto HTTP status codes. No image math
or Playwright calls live here.

Endpoints
---------
POST /session/start    -> initialize a browser session
POST /session/view     -> capture the current optimized state (macro/delta/zoom)
POST /session/action   -> dispatch an action, return the next optimized state
POST /session/zoom     -> return a 512x512 zoom tile for one mark
DELETE /session/{id}   -> tear a session down
GET  /healthz          -> liveness + active session count
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Path, Query, Request
from fastapi.responses import JSONResponse

from ..errors import VLMOptimizerError
from ..manager import SessionManager
from ..models import (
    ActionRequest,
    StartSessionRequest,
    StartSessionResponse,
    ViewMode,
    ViewState,
    ZoomRequest,
)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Create the session manager at startup; close every browser at shutdown."""
    manager = SessionManager()
    manager.start_reaper()
    app.state.manager = manager
    try:
        yield
    finally:
        await manager.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(
        title="VLM-Viewport-Optimizer",
        version="0.1.0",
        summary="Token- and latency-optimized screenshot navigation for VLM agents.",
        lifespan=_lifespan,
    )

    @app.exception_handler(VLMOptimizerError)
    async def _domain_error_handler(_: Request, exc: VLMOptimizerError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content={"detail": str(exc), "code": exc.code},
        )

    def _manager(request: Request) -> SessionManager:
        return request.app.state.manager

    # ------------------------------------------------------------------ #
    @app.get("/healthz", tags=["ops"])
    async def healthz(request: Request) -> dict[str, object]:
        return {"status": "ok", "active_sessions": _manager(request).count}

    # ------------------------------------------------------------------ #
    @app.post("/session/start", response_model=StartSessionResponse, tags=["session"])
    async def start_session(
        body: StartSessionRequest, request: Request
    ) -> StartSessionResponse:
        session = await _manager(request).create(
            start_url=body.start_url, headless=body.headless
        )
        return StartSessionResponse(session_id=session.session_id, url=session.current_url)

    # ------------------------------------------------------------------ #
    @app.post("/session/view", response_model=ViewState, tags=["session"])
    async def view_session(
        request: Request,
        session_id: str = Query(..., description="Session returned by /session/start."),
        mode: ViewMode = Query(
            ViewMode.MACRO, description="macro = full overview, delta = changed region only."
        ),
    ) -> ViewState:
        session = await _manager(request).get(session_id)
        return await session.view(mode)

    # ------------------------------------------------------------------ #
    @app.post("/session/action", response_model=ViewState, tags=["session"])
    async def action_session(
        body: ActionRequest,
        request: Request,
        session_id: str = Query(..., description="Session returned by /session/start."),
    ) -> ViewState:
        session = await _manager(request).get(session_id)
        return await session.act(body)

    # ------------------------------------------------------------------ #
    @app.post("/session/zoom", response_model=ViewState, tags=["session"])
    async def zoom_session(
        body: ZoomRequest,
        request: Request,
        session_id: str = Query(..., description="Session returned by /session/start."),
    ) -> ViewState:
        session = await _manager(request).get(session_id)
        return await session.zoom(body.mark_id)

    # ------------------------------------------------------------------ #
    @app.delete("/session/{session_id}", tags=["session"])
    async def delete_session(
        request: Request,
        session_id: str = Path(..., description="Session to tear down."),
    ) -> dict[str, str]:
        await _manager(request).destroy(session_id)
        return {"status": "closed", "session_id": session_id}

    return app


app = create_app()
"""Module-level ASGI app so ``uvicorn vlm_optimizer.api.app:app`` just works."""
