#!/usr/bin/env python3
"""Reference VLM-agent loop — a mock multimodal client driving the optimizer.

This script plays the role a real Vision-Language-Model (Claude Computer Use,
GPT-4o, …) would play in production. It does NOT call any paid API; instead it
contains a tiny, deterministic ``MockVisionModel`` that "looks at" the exact
payload a real VLM would receive — the Base64 frame plus the Set-of-Mark layout
map — and returns the same JSON action shape a real model would emit.

Swapping in a real model is a one-method change: replace
``MockVisionModel.decide`` with a call to your provider, passing
``state.image_base64`` as the image block and ``state.elements`` (the layout
map) as text. Everything else — the perception→decision→action loop, the
macro/delta/zoom optimization, coordinate resolution — stays identical.

Run
---
    # terminal 1
    python run.py
    # terminal 2
    pip install httpx
    python examples/vlm_agent_loop.py --goal "find more information" --url https://example.com

Each frame the "model" sees is written to ``dumps/agent/`` so you can inspect
precisely what pixels were spent on every step.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import os
import sys
import textwrap
from typing import Any

try:
    import httpx
except ImportError:  # pragma: no cover
    sys.exit("This example needs httpx:  pip install httpx")


# --------------------------------------------------------------------------- #
# Transport: a thin, typed wrapper over the three optimizer endpoints.
# --------------------------------------------------------------------------- #
class OptimizerClient:
    """HTTP client for the VLM-Viewport-Optimizer server."""

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self._http = httpx.Client(base_url=base_url, timeout=timeout)

    def start(self, start_url: str) -> str:
        r = self._http.post("/session/start", json={"start_url": start_url})
        r.raise_for_status()
        return r.json()["session_id"]

    def view(self, session_id: str, mode: str = "macro") -> dict[str, Any]:
        # NOTE: the capture endpoint is /session/view (mode = macro | delta).
        r = self._http.post("/session/view", params={"session_id": session_id, "mode": mode})
        r.raise_for_status()
        return r.json()

    def zoom(self, session_id: str, mark_id: str) -> dict[str, Any]:
        r = self._http.post(
            "/session/zoom", params={"session_id": session_id}, json={"mark_id": mark_id}
        )
        r.raise_for_status()
        return r.json()

    def action(self, session_id: str, action: dict[str, Any]) -> dict[str, Any]:
        r = self._http.post(
            "/session/action", params={"session_id": session_id}, json=action
        )
        r.raise_for_status()
        return r.json()

    def close(self, session_id: str) -> None:
        try:
            self._http.delete(f"/session/{session_id}")
        finally:
            self._http.close()


# --------------------------------------------------------------------------- #
# The "model". In production this is one API call; here it's a readable policy
# so the example is fully runnable with zero credentials.
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class Decision:
    """The structured action a (real or mock) VLM returns each turn."""

    tool: str  # "zoom" | "click" | "type" | "scroll" | "stop"
    mark_id: str | None = None
    text: str | None = None
    reasoning: str = ""


class MockVisionModel:
    """A stand-in for a real multimodal model.

    It receives exactly what a real VLM would receive — the decoded frame bytes
    and the layout map — and returns the same action contract. The policy is a
    simple, legible heuristic so you can watch the optimization loop work without
    burning API tokens:

        1. If the goal looks like a query and a text field exists, type into it.
        2. Otherwise, zoom the most goal-relevant element once for a close look.
        3. Then click the best-matching actionable element.
        4. If nothing matches, stop.
    """

    def __init__(self, goal: str) -> None:
        self.goal = goal.lower()
        self._typed = False
        self._zoomed: set[str] = set()

    def decide(self, frame_png: bytes, elements: list[dict[str, Any]]) -> Decision:
        """Perception → decision. ``frame_png`` is the image a real VLM would see."""
        # A real model consumes the pixels; our mock just proves they arrived
        # intact and reasons over the (far cheaper) structured layout map.
        _ = frame_png  # decoded and handed to the "model" — see run loop.

        if not elements:
            return Decision(tool="stop", reasoning="No interactable elements in view.")

        text_fields = [e for e in elements if e["role"] in ("input", "textarea", "editable")]
        clickables = [e for e in elements if e["role"] in ("button", "link", "checkbox", "radio")]

        # 1) Fill a search/query field first if the goal reads like a query.
        if text_fields and not self._typed and self._looks_like_query(self.goal):
            field = self._best_match(text_fields) or text_fields[0]
            self._typed = True
            return Decision(
                tool="type",
                mark_id=field["mark_id"],
                text=self.goal,
                reasoning=f"Goal looks like a query; typing it into {field['mark_id']}.",
            )

        # 2) Zoom the best candidate once to 'read the fine print' before acting.
        target = self._best_match(clickables) or (clickables[0] if clickables else None)
        if target and target["mark_id"] not in self._zoomed:
            self._zoomed.add(target["mark_id"])
            return Decision(
                tool="zoom",
                mark_id=target["mark_id"],
                reasoning=f"Inspecting {target['mark_id']} ({target['name']!r}) up close.",
            )

        # 3) Commit: click the chosen element.
        if target:
            return Decision(
                tool="click",
                mark_id=target["mark_id"],
                reasoning=f"Clicking {target['mark_id']} ({target['name']!r}) — best goal match.",
            )

        return Decision(tool="stop", reasoning="Nothing left that advances the goal.")

    # -- tiny relevance heuristics ----------------------------------------- #
    def _best_match(self, elements: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Pick the element whose accessible name best overlaps the goal."""
        goal_words = {w for w in self.goal.split() if len(w) > 2}
        best, best_score = None, 0
        for el in elements:
            name = (el.get("name") or "").lower()
            score = sum(1 for w in goal_words if w in name)
            if score > best_score:
                best, best_score = el, score
        return best

    @staticmethod
    def _looks_like_query(goal: str) -> bool:
        return any(k in goal for k in ("search", "find", "look up", "query", "who", "what"))


# --------------------------------------------------------------------------- #
# The perception → decision → action loop.
# --------------------------------------------------------------------------- #
def run_agent(base_url: str, start_url: str, goal: str, max_steps: int) -> None:
    client = OptimizerClient(base_url)
    model = MockVisionModel(goal)
    dump_dir = os.path.join(os.path.dirname(__file__), "..", "dumps", "agent")
    os.makedirs(dump_dir, exist_ok=True)

    print(f"🎯 Goal: {goal!r}\n🌐 Start: {start_url}\n")
    session_id = client.start(start_url)
    print(f"▶  session {session_id}\n")

    try:
        for step in range(1, max_steps + 1):
            # --- PERCEIVE: pull the cheap macro overview + layout map --------
            state = client.view(session_id, mode="macro")
            frame = _decode_and_dump(state, dump_dir, f"step{step:02d}_perceive")
            _print_perception(step, state)

            # --- DECIDE: hand the exact VLM payload to the model ------------
            decision = model.decide(frame, state["elements"])
            print(f"    🧠 decision: {decision.tool} "
                  f"{decision.mark_id or ''} — {decision.reasoning}")

            if decision.tool == "stop":
                print("    ⏹  model chose to stop.")
                break

            # --- ACT -------------------------------------------------------
            if decision.tool == "zoom":
                zoom_state = client.zoom(session_id, decision.mark_id or "")
                _decode_and_dump(zoom_state, dump_dir, f"step{step:02d}_zoom")
                print(f"    🔍 zoomed {decision.mark_id}: "
                      f"{zoom_state['image_width']}x{zoom_state['image_height']} tile")
                continue  # Re-perceive next turn with the closer look in mind.

            action = _decision_to_action(decision)
            result = client.action(session_id, action)
            _decode_and_dump(result, dump_dir, f"step{step:02d}_result")
            region = result.get("change_region")
            print(f"    ✅ {decision.tool} → returned a '{result['mode']}' frame"
                  + (f", change_region={_fmt_region(region)}" if region else ""))
            print()
        else:
            print("    ⏹  reached max steps.")
    finally:
        client.close(session_id)
        print(f"\n🧹 session closed. Frames the model saw are in {os.path.normpath(dump_dir)}")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _decision_to_action(decision: Decision) -> dict[str, Any]:
    """Translate a model Decision into an /session/action request body.

    We request a ``delta`` return so each post-action frame carries only the
    pixels that changed — the whole point of the optimizer.
    """
    action: dict[str, Any] = {"action": decision.tool, "return_mode": "delta"}
    if decision.mark_id:
        action["mark_id"] = decision.mark_id
    if decision.text is not None:
        action["text"] = decision.text
    return action


def _decode_and_dump(state: dict[str, Any], dump_dir: str, name: str) -> bytes:
    """Decode the Base64 frame (as a real VLM would) and persist it for review."""
    raw = base64.b64decode(state["image_base64"])
    ext = state.get("image_format", "jpeg").replace("jpeg", "jpg")
    with open(os.path.join(dump_dir, f"{name}.{ext}"), "wb") as fh:
        fh.write(raw)
    return raw


def _print_perception(step: int, state: dict[str, Any]) -> None:
    kb = len(state["image_base64"]) * 3 // 4 // 1024
    print(
        f"── step {step} ──  mode={state['mode']}  "
        f"{state['image_width']}x{state['image_height']}  ~{kb}KB  "
        f"url={state['url']}"
    )
    for el in state["elements"][:8]:
        name = textwrap.shorten(el.get("name", ""), width=38, placeholder="…")
        print(f"    {el['mark_id']:>4}  {el['role']:<9} {name!r}")
    if len(state["elements"]) > 8:
        print(f"    … (+{len(state['elements']) - 8} more marks)")


def _fmt_region(region: dict[str, Any] | None) -> str:
    if not region:
        return "none"
    return f"({region['x']:.0f},{region['y']:.0f} {region['width']:.0f}x{region['height']:.0f})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference VLM-agent loop (mock model).")
    parser.add_argument("--base-url", default=os.environ.get("VLM_BASE", "http://127.0.0.1:8000"))
    parser.add_argument("--url", default="https://example.com", help="Page to start on.")
    parser.add_argument("--goal", default="find more information", help="Natural-language task.")
    parser.add_argument("--max-steps", type=int, default=5)
    args = parser.parse_args()
    run_agent(args.base_url, args.url, args.goal, args.max_steps)


if __name__ == "__main__":
    main()
