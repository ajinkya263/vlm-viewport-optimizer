"""Drop-in agent wiring: a function-calling tool schema + a system prompt.

The whole point of the optimizer is to be trivially *usable by an LLM*. This
module hands you the two artifacts you need to point Claude or GPT at the server
with zero guesswork:

* :data:`ACTION_TOOL_SCHEMA` — a JSON-Schema tool definition (the shape both the
  Anthropic ``tools`` API and the OpenAI function-calling API accept) describing
  every action the agent can take.
* :data:`SYSTEM_PROMPT` — concise instructions that teach the model the
  perceive → decide → act loop and, crucially, *when to spend tokens*: read the
  cheap ``layout_text``, zoom only when detail is required, act by ``mark_id``.

Exposed over HTTP at ``GET /agent/spec`` so any client can fetch them at runtime.
"""

from __future__ import annotations

from typing import Any

# A single "browser" tool with an action discriminator keeps the model's choice
# space small and its outputs easy to validate — one tool, a few named actions.
ACTION_TOOL_SCHEMA: dict[str, Any] = {
    "name": "browser",
    "description": (
        "Control a web browser by referring to on-screen elements by their "
        "Set-of-Mark id (e.g. 'A1'). Use the layout_text/elements from the latest "
        "view to choose a mark. Prefer 'zoom' to inspect before committing to a "
        "'click' when a target is small or ambiguous."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "click",
                    "double_click",
                    "type",
                    "press",
                    "hover",
                    "scroll",
                    "goto",
                    "zoom",
                    "stop",
                ],
                "description": "The interaction to perform.",
            },
            "mark_id": {
                "type": "string",
                "description": "Target element id from the layout (click/type/press/hover/zoom).",
            },
            "text": {"type": "string", "description": "Text to type (action='type')."},
            "key": {"type": "string", "description": "Key name, e.g. 'Enter' (action='press')."},
            "url": {"type": "string", "description": "Destination URL (action='goto')."},
            "dx": {"type": "integer", "description": "Horizontal scroll px (action='scroll')."},
            "dy": {"type": "integer", "description": "Vertical scroll px (action='scroll')."},
            "reasoning": {
                "type": "string",
                "description": "One sentence: why this action advances the goal.",
            },
        },
        "required": ["action"],
    },
}

SYSTEM_PROMPT: str = """\
You are a web-browsing agent operating a real browser through an optimized visual \
interface. Each turn you receive a screenshot plus a layout of interactable \
elements, each tagged with a short id like "A1".

How the interface saves you work (use it deliberately):
- The screenshot is already optimized. A "macro" frame is a downscaled overview; \
a "delta" frame shows only what changed since your last action; a "zoom" tile is \
a 512x512 close-up of one element.
- `layout_text` lists every actionable element as `ID role "name" @x,y`. READ THIS \
FIRST — it is far cheaper than studying pixels and usually enough to decide.
- Always act by `mark_id`, never by raw coordinates. The server resolves the id to \
the exact click point for you, so you cannot miss.

Loop each turn:
1. Read `layout_text` (and the image only if you need visual confirmation).
2. If a target is small, ambiguous, or you must read fine print, emit `zoom` on it \
first, then decide next turn.
3. Otherwise emit exactly one action (`click`, `type`, `scroll`, `goto`, ...) via \
the `browser` tool with a one-sentence `reasoning`.
4. When the goal is achieved or no action can advance it, emit `action: "stop"`.

Be decisive. Prefer the fewest actions that accomplish the goal.
"""


def agent_spec() -> dict[str, Any]:
    """Return both artifacts as a single JSON-serializable payload."""
    return {
        "tool": ACTION_TOOL_SCHEMA,
        "system_prompt": SYSTEM_PROMPT,
        "notes": (
            "POST actions to /session/action (return_mode=macro|delta). "
            "Use /session/zoom for the 'zoom' action. Resolve ids from the latest view."
        ),
    }
