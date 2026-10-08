"""Process-wide registry of live sessions, with idle reaping.

Kept separate from the FastAPI module so the session store can be unit-tested
and reused outside of HTTP (e.g. a CLI or notebook driver).
"""

from __future__ import annotations

import asyncio

from .config import settings
from .errors import SessionNotFoundError
from .session import Session


class SessionManager:
    """Owns every :class:`Session` and guarantees clean shutdown."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self._reaper: asyncio.Task | None = None

    async def create(self, *, start_url: str, headless: bool | None) -> Session:
        session = Session(headless=headless)
        await session.start(start_url)
        async with self._lock:
            self._sessions[session.session_id] = session
        return session

    async def get(self, session_id: str) -> Session:
        async with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"No active session '{session_id}'.")
        return session

    async def destroy(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            raise SessionNotFoundError(f"No active session '{session_id}'.")
        await session.close()

    async def shutdown(self) -> None:
        """Close every browser on app teardown."""
        if self._reaper is not None:
            self._reaper.cancel()
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        await asyncio.gather(*(s.close() for s in sessions), return_exceptions=True)

    # ------------------------------------------------------------------ #
    # Background idle reaper
    # ------------------------------------------------------------------ #
    def start_reaper(self) -> None:
        if self._reaper is None:
            self._reaper = asyncio.create_task(self._reap_loop())

    async def _reap_loop(self) -> None:
        interval = max(30, settings.session_idle_ttl_s // 3)
        try:
            while True:
                await asyncio.sleep(interval)
                await self._reap_once()
        except asyncio.CancelledError:  # Clean shutdown path.
            raise

    async def _reap_once(self) -> None:
        async with self._lock:
            expired = [
                s
                for s in self._sessions.values()
                if s.idle_seconds > settings.session_idle_ttl_s
            ]
            for session in expired:
                self._sessions.pop(session.session_id, None)
        # Close outside the lock so a slow teardown can't stall new requests.
        for session in expired:
            await session.close()

    @property
    def count(self) -> int:
        return len(self._sessions)
