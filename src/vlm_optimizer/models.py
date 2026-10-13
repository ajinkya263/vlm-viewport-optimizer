"""Pydantic contracts shared across every layer and exposed over the API.

These models are the single source of truth for the shapes that cross module
boundaries (DOM -> vision -> registry -> HTTP), which keeps the layers decoupled.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    """Axis-aligned box in viewport CSS pixels (origin = top-left)."""

    model_config = ConfigDict(frozen=True)

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    def to_xywh_int(self) -> tuple[int, int, int, int]:
        return (round(self.x), round(self.y), round(self.width), round(self.height))


class ElementRole(StrEnum):
    """Normalized, VLM-friendly action categories."""

    BUTTON = "button"
    LINK = "link"
    INPUT = "input"
    TEXTAREA = "textarea"
    SELECT = "select"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    EDITABLE = "editable"
    OTHER = "other"


class InteractableElement(BaseModel):
    """A single actionable element the agent can target by ``mark_id``."""

    mark_id: str = Field(description="Short alphanumeric tag drawn on the frame, e.g. 'A1'.")
    role: ElementRole
    tag: str = Field(description="Raw lowercase HTML tag name.")
    name: str = Field(default="", description="Accessible name / visible label (truncated).")
    value: str = Field(default="", description="Current value for form fields (truncated).")
    bbox: BoundingBox
    center: tuple[float, float]
    enabled: bool = True


def compact_layout(elements: list["InteractableElement"]) -> str:
    """Serialize elements to a terse, token-cheap layout block for the LLM.

    One line per element, e.g.::

        A1  link      "Learn more" @67,105
        A2  input     "Search box" @249,105
        A3  button    "Open" @423,105 [disabled]

    This is markedly cheaper in tokens than the full JSON (no keys, braces, or
    quotes repeated per field) while preserving everything an agent needs to act:
    the mark id, the role, the accessible name, and the click midpoint.
    """
    lines: list[str] = []
    for el in elements:
        cx, cy = (round(c) for c in el.center)
        name = f'"{el.name}"' if el.name else '""'
        flag = "" if el.enabled else " [disabled]"
        lines.append(f"{el.mark_id:<4} {el.role:<9} {name} @{cx},{cy}{flag}")
    return "\n".join(lines)


class ViewMode(StrEnum):
    MACRO = "macro"
    """Whole-viewport, heavily compressed, labeled overview."""
    DELTA = "delta"
    """Only the changed region is visible; everything else blacked out."""
    ZOOM = "zoom"
    """A single padded 512x512 crop around one element."""


class ViewState(BaseModel):
    """What ``/session/view`` and ``/session/action`` return to the agent."""

    session_id: str
    mode: ViewMode
    url: str
    title: str = ""
    image_format: Literal["jpeg", "png"] = "jpeg"
    image_base64: str = Field(description="Base64 of the processed frame (no data: prefix).")
    image_width: int
    image_height: int
    elements: list[InteractableElement] = Field(default_factory=list)
    layout_text: str | None = Field(
        default=None,
        description=(
            "Token-efficient one-line-per-element layout the LLM can read instead of "
            "the verbose `elements` JSON. Omitted unless requested."
        ),
    )
    change_region: BoundingBox | None = Field(
        default=None, description="Populated in DELTA mode: the box that changed."
    )
    estimated_vision_tokens: int = Field(
        default=0, description="Estimated vision tokens THIS frame costs the model."
    )
    estimated_full_frame_tokens: int = Field(
        default=0,
        description="What a naive full-resolution viewport screenshot would have cost.",
    )
    token_savings_ratio: float = Field(
        default=0.0, description="Fraction of vision tokens saved vs. the naive full frame."
    )
    note: str = Field(default="", description="Human-readable diagnostics (fallbacks, etc).")


class SessionStats(BaseModel):
    """Cumulative optimization scorecard for a session — the headline metric."""

    # 'model_family' would otherwise collide with Pydantic's protected 'model_' namespace.
    model_config = ConfigDict(protected_namespaces=())

    session_id: str
    frames_served: int
    optimized_tokens_total: int = Field(description="Vision tokens actually spent.")
    naive_tokens_total: int = Field(description="Vision tokens a full-frame loop would spend.")
    tokens_saved: int
    savings_ratio: float = Field(description="Overall fraction of vision tokens saved.")
    model_family: str


# --------------------------------------------------------------------------
# Request bodies
# --------------------------------------------------------------------------
class StartSessionRequest(BaseModel):
    start_url: str = Field(default="about:blank")
    headless: bool | None = Field(default=None, description="Per-session headless override.")


class StartSessionResponse(BaseModel):
    session_id: str
    url: str


class ActionType(StrEnum):
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    TYPE = "type"
    PRESS = "press"
    HOVER = "hover"
    SCROLL = "scroll"
    GOTO = "goto"


class ActionRequest(BaseModel):
    """One agent action. ``mark_id`` resolves against the session registry."""

    action: ActionType
    mark_id: str | None = Field(default=None, description="Target element tag (click/type/...).")
    text: str | None = Field(default=None, description="Text payload for TYPE.")
    key: str | None = Field(default=None, description="Key name for PRESS, e.g. 'Enter'.")
    url: str | None = Field(default=None, description="Destination for GOTO.")
    dx: int = Field(default=0, description="Horizontal scroll delta (px) for SCROLL.")
    dy: int = Field(default=0, description="Vertical scroll delta (px) for SCROLL.")
    clear_first: bool = Field(default=True, description="Clear field before TYPE.")
    return_mode: ViewMode = Field(
        default=ViewMode.DELTA,
        description="Which optimized representation to return after the action.",
    )


class ZoomRequest(BaseModel):
    mark_id: str


class ErrorResponse(BaseModel):
    detail: str
    code: str = "error"
