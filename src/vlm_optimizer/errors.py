"""Typed exception hierarchy mapped to HTTP status codes at the API boundary."""

from __future__ import annotations


class VLMOptimizerError(Exception):
    """Base class for all framework errors."""

    http_status: int = 500
    code: str = "internal_error"


class SessionNotFoundError(VLMOptimizerError):
    http_status = 404
    code = "session_not_found"


class UnknownMarkError(VLMOptimizerError):
    """The agent referenced a mark_id that is not in the current registry."""

    http_status = 422
    code = "unknown_mark"


class StaleElementError(VLMOptimizerError):
    """The targeted DOM node vanished or detached between view and action."""

    http_status = 409
    code = "stale_element"


class BrowserTimeoutError(VLMOptimizerError):
    http_status = 504
    code = "browser_timeout"


class BrowserNotStartedError(VLMOptimizerError):
    http_status = 409
    code = "browser_not_started"


class InvalidActionError(VLMOptimizerError):
    http_status = 422
    code = "invalid_action"
