#!/usr/bin/env python3
"""End-to-end smoke test / usage example against a running server.

    # terminal 1
    python run.py
    # terminal 2
    python examples/demo_client.py

Walks the full optimized loop: start -> macro view -> type a query -> delta view
of the change -> zoom into one result. Saves each returned frame to ``dumps/``
so you can eyeball exactly what the VLM would receive.
"""

from __future__ import annotations

import base64
import os

import httpx  # type: ignore[import-not-found]

BASE = os.environ.get("VLM_BASE", "http://127.0.0.1:8000")
OUT = os.path.join(os.path.dirname(__file__), "..", "dumps")


def _save(name: str, b64: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "wb") as fh:
        fh.write(base64.b64decode(b64))
    print(f"   saved {path}")


def main() -> None:
    with httpx.Client(base_url=BASE, timeout=60.0) as client:
        print("1. start")
        r = client.post("/session/start", json={"start_url": "https://example.com"})
        r.raise_for_status()
        sid = r.json()["session_id"]
        print(f"   session={sid} url={r.json()['url']}")

        print("2. macro view")
        r = client.post("/session/view", params={"session_id": sid, "mode": "macro"})
        r.raise_for_status()
        state = r.json()
        print(f"   {len(state['elements'])} interactable marks, {state['image_width']}x{state['image_height']}")
        _save("01_macro.jpg", state["image_base64"])
        for el in state["elements"][:10]:
            print(f"     {el['mark_id']:>4}  {el['role']:<8} {el['name'][:40]!r}")

        if state["elements"]:
            target = state["elements"][0]["mark_id"]
            print(f"3. zoom into {target}")
            r = client.post("/session/zoom", params={"session_id": sid}, json={"mark_id": target})
            r.raise_for_status()
            _save("02_zoom.jpg", r.json()["image_base64"])

            print(f"4. click {target} (delta return)")
            r = client.post(
                "/session/action",
                params={"session_id": sid},
                json={"action": "click", "mark_id": target, "return_mode": "delta"},
            )
            r.raise_for_status()
            dstate = r.json()
            print(f"   mode={dstate['mode']} change_region={dstate['change_region']}")
            _save("03_delta.jpg", dstate["image_base64"])

        print("5. close")
        client.delete(f"/session/{sid}")
        print("   done.")


if __name__ == "__main__":
    main()
