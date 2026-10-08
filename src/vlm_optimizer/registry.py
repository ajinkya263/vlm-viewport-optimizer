"""Strictly-isolated, in-memory mark registry.

Each :class:`Session` owns exactly one :class:`MarkRegistry`. It maps the short
alphanumeric tags we draw onto a frame back to (a) the element metadata and
(b) the precise absolute ``(x, y)`` midpoint used to dispatch real clicks.

The registry is rebuilt from scratch on every capture so a stale tag from a
previous frame can never silently resolve to the wrong coordinate.
"""

from __future__ import annotations

from .errors import UnknownMarkError
from .models import InteractableElement


def generate_mark_id(index: int) -> str:
    """Deterministic, compact, VLM-legible IDs: A1..A99, B1..B99, ... Z99, AA1, ...

    Kept short on purpose — every extra glyph is extra pixels the vision encoder
    must tokenize across potentially dozens of marks.
    """
    if index < 0:
        raise ValueError("index must be non-negative")
    letters_index, number = divmod(index, 99)
    prefix_chars: list[str] = []
    letters_index += 1  # 1-based so we get A, B, ... Z, AA (bijective base-26)
    while letters_index > 0:
        letters_index, rem = divmod(letters_index - 1, 26)
        prefix_chars.append(chr(ord("A") + rem))
    prefix = "".join(reversed(prefix_chars))
    return f"{prefix}{number + 1}"


class MarkRegistry:
    """Isolated per-session mapping of ``mark_id -> element / midpoint``."""

    __slots__ = ("_by_id",)

    def __init__(self) -> None:
        self._by_id: dict[str, InteractableElement] = {}

    def rebuild(self, elements: list[InteractableElement]) -> None:
        """Atomically swap in a fresh set of marks (clears all previous tags)."""
        self._by_id = {el.mark_id: el for el in elements}

    def resolve(self, mark_id: str) -> InteractableElement:
        """Return the element for ``mark_id`` or raise :class:`UnknownMarkError`."""
        key = mark_id.strip().upper()
        element = self._by_id.get(key)
        if element is None:
            raise UnknownMarkError(
                f"Mark '{mark_id}' is not in the current view. "
                f"Known marks: {sorted(self._by_id)[:20]}"
            )
        return element

    def midpoint(self, mark_id: str) -> tuple[float, float]:
        return self.resolve(mark_id).center

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, mark_id: str) -> bool:
        return mark_id.strip().upper() in self._by_id
