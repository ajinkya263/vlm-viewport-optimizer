<div align="center">

# 🪟 VLM-Viewport-Optimizer

**Token- and latency-optimized browser automation for Vision-Language-Model agents.**

Let a visual agent — Claude Computer Use, GPT-4o — navigate the web by screenshot,
while sending it the *smallest sufficient pixels* instead of a full frame every turn.

</div>

---

## The problem

A VLM web agent takes a screenshot, decides, acts, and repeats. The naive loop
re-sends a full **1280×800** screenshot on *every* step. A frame that size costs
on the order of **~1,100 vision tokens** per turn with current patch encoders — and
a single task can be 20–40 turns. Most of those pixels are **unchanged** between
turns (the same nav bar, the same footer), so the model pays, over and over, to
re-encode a page it already saw.

This framework attacks that waste on three fronts.

| Technique | What it sends | Why it's cheap |
|-----------|---------------|----------------|
| **Set-of-Mark labels** | Elements tagged `A1`,`A2`,… | Agent references IDs, not pixel coords → no coordinate hallucination, fewer retries |
| **Differential patch-rendering (delta frames)** | Only the region that *changed*, rest blacked to `[0,0,0]` | A patch encoder tokenizes uniform black for almost nothing |
| **Macro → Zoom hierarchy** | A tiny compressed overview, then one 512×512 tile on demand | Full fidelity only where the agent looks; one vision tile instead of a whole page |

The server keeps an **isolated per-session registry** mapping each mark back to a
precise click midpoint — so the model never has to reason about raw coordinates,
and the coordinates it never sees can never be wrong.

---

## Why vision + DOM, not HTML scanning?

There are two schools of web agents. **DOM/HTML-scanning** agents flatten the page
to text (raw HTML or an accessibility tree) and let the model read that.
**Vision** agents send a screenshot. This framework is a deliberate **hybrid** — it
uses the DOM to *find* elements but renders them visually — and that combination
beats pure HTML scanning on real-world pages:

| Pure HTML/DOM scanning | This hybrid (vision + DOM) |
|------------------------|----------------------------|
| A real page's markup is **50k–500k+ tokens** of nested divs, inline styles, and tracking junk — often *more* than an image | The DOM is distilled to a few dozen **visible, actionable** elements (~hundreds of tokens) |
| HTML **lies about visibility** — hidden, off-screen, or overlay-covered nodes still exist in the markup, so agents "click" invisible things | Extraction runs real `getComputedStyle` + `getBoundingClientRect` + an `elementFromPoint` **occlusion test** — covered elements are dropped |
| **Canvas / WebGL / video** apps (Maps, Figma, charts, games) have no meaningful HTML — DOM scanning is blind | Vision sees the actual rendered pixels |
| Flattened markup **loses spatial layout** (which label goes with which field) | The screenshot preserves real geometry |
| Obfuscated / client-rendered markup is **fragile** | Acts on what's rendered, not on brittle selectors |

**The synthesis:** the server uses the DOM to know *what* is actionable and *exactly
where* (the registry's precise click midpoints), then shows the model only the
minimal pixels needed to decide — **structure *and* sight**, at a fraction of the
token cost of either extreme. Pure text/accessibility-tree agents still win on
static, form-heavy, text-only pages; this approach wins on visual, dynamic,
canvas-based, or scrape-hostile pages — and whenever you're driving a vision model.

---

## How differential patch-rendering works

This is the core optimization. After an action, we don't ask "what does the page
look like?" — we ask "**what changed?**" and send only that.

```
 Frame N-1 (prev)         Frame N (curr)                Delta output → to VLM
┌────────────────┐      ┌────────────────┐            ┌────────────────┐
│ ▜ navbar       │      │ ▜ navbar       │            │                │   everything
│   search:[   ] │  →   │   search:[cat] │     ⇒      │        [cat]   │ ← unchanged is
│                │      │   ┌──────────┐ │            │      ┌──────────┐│   pure black
│ ▟ footer       │      │   │ dropdown │ │            │      │ dropdown ││   = ~free tokens
└────────────────┘      │   └──────────┘ │            │      └──────────┘│
                        │ ▟ footer       │            │                │
                        └────────────────┘            └────────────────┘
```

The pipeline, in [`vision/delta.py`](src/vlm_optimizer/vision/delta.py):

1. **`cv2.absdiff(prev, curr)`** — per-pixel difference of the two raw frames.
2. **Grayscale + threshold** (`VLM_DELTA_THRESHOLD`, default 25) — ignore sub-perceptual
   noise (anti-aliasing, cursor shimmer).
3. **Morphological dilation** — merge nearby changed clusters (a field value *and*
   its validation message) into one coherent region.
4. **Area gate** (`VLM_DELTA_MIN_AREA_PX`) — if nothing meaningful moved, we *don't*
   ship an all-black frame; we transparently fall back to a full macro overview.
5. **Mask-outside** — copy the changed bounding box onto a `np.zeros_like` canvas.
   Every pixel outside the change is exactly `[0,0,0]`.

The diff runs on **raw** frames (the Set-of-Mark labels are redrawn every turn and
would register as spurious change), but the returned delta is rendered from the
**labeled** frame — so the agent still sees the `A1`/`A2` tags *inside* the changed
region and can act immediately.

> **Honest note on where the tokens actually go.** Under both the Anthropic
> (pixels ÷ 750) and OpenAI (512px tiles) cost models, *blacking out pixels at the
> same resolution does not reduce the token bill* — it reduces bytes and sharpens
> model attention, but a mostly-black full-res image still costs like a full frame.
> So the delta is also **downscaled** (`VLM_DELTA_MAX_WIDTH`) before it's returned:
> fewer pixels → fewer tiles → a *real* token reduction. The optimizer never claims
> savings it can't measure — see the scorecard below.

> **Verified behavior** — clicking a button that injects new DOM content returns a
> frame where the entire viewport is black except the mutated region plus its marks.
> Run `examples/demo_client.py` and open `dumps/03_delta.jpg` to see it for yourself.

---

## System architecture

Three cleanly separated layers. **Networking never touches pixels; image math
never touches Playwright.** Every CPU-bound OpenCV call runs in a worker thread
(`asyncio.to_thread`) so the event loop shared by Playwright and FastAPI never
blocks, and a per-session `asyncio.Lock` serializes captures vs. actions to kill
the capture/action race.

```
src/vlm_optimizer/
├── config.py              # all tunables (env-overridable, VLM_* prefix)
├── models.py              # Pydantic contracts shared across every layer
├── errors.py              # typed errors → HTTP status codes
├── registry.py            # isolated in-memory  mark_id → (element, midpoint)
├── session.py             # orchestrator — asyncio.Lock serializes view/action/zoom
├── manager.py             # session store + background idle reaper
├── agent_spec.py          # function-calling tool schema + system prompt for LLMs
│
├── browser/               # ══ AUTOMATION LAYER — Playwright ══
│   ├── controller.py      #    async Chromium lifecycle · capture · dispatch
│   └── dom_script.py      #    in-page JS: visible + interactable extraction
│
├── vision/                # ══ IMAGE-MATH LAYER — OpenCV / NumPy / PIL ══
│   ├── labeler.py         #    Set-of-Mark rendering (color-coded by role)
│   ├── delta.py           #    cv2.absdiff differential engine
│   ├── crop.py            #    macro / delta downscale + 512px zoom tile
│   ├── tokens.py          #    vision-token cost estimation (Anthropic / OpenAI)
│   └── codec.py           #    BGR ndarray → compressed base64
│
└── api/
    └── app.py             # ══ NETWORKING LAYER — FastAPI ══
```

### Request flow

```
                         ┌─────────────────── Session (asyncio.Lock) ───────────────────┐
 VLM Agent ── HTTP ──►   │                                                                │
   │   /session/view     │   BrowserController.capture()                                  │
   │                     │        │  screenshot (PNG) ─► cv2.imdecode ─► BGR ndarray      │
   │                     │        │  page.evaluate(DOM script) ─► interactable elements   │
   │                     │        ▼                                                       │
   │                     │   MarkRegistry.rebuild()   SetOfMarkLabeler.render()           │
   │                     │        │                        │                              │
   │                     │        │                   DeltaEngine.process()  (vs frame-1) │
   │   ◄── ViewState ────┤        └──────────► ViewportCropper (macro | zoom) ─► base64   │
   │   (image + layout)  │                                                                │
   │   /session/action   │   registry.resolve(mark_id) ─► midpoint ─► mouse/keyboard ─────┤
   └─────────────────────┘                                                                │
                         └────────────────────────────────────────────────────────────────┘
```

---

## Quickstart

```bash
cd vlm-viewport-optimizer
python3.11 -m venv .venv && source .venv/bin/activate     # 3.11+ required
pip install -r requirements.txt
playwright install chromium                                # one-time browser download
```

**Run the server:**

```bash
python run.py                       # http://127.0.0.1:8000  (Swagger UI at /docs)
# or
uvicorn vlm_optimizer.api.app:app --app-dir src --reload
```

**Watch the optimization loop with the reference agent** (a mock VLM — zero API keys):

```bash
pip install httpx
python examples/vlm_agent_loop.py --goal "find more information" --url https://example.com
```

```text
── step 1 ──  mode=macro  768x480  ~16KB  url=https://example.com/
      A1  link      'Learn more'
    🧠 decision: zoom A1 — Inspecting A1 ('Learn more') up close.
    🔍 zoomed A1: 512x512 tile
── step 2 ──  mode=macro  768x480  ~16KB
    🧠 decision: click A1 — best goal match.
    ✅ click → returned a 'delta' frame, change_region=(88,41 1099x727)
```

Every frame the "model" sees is written to `dumps/agent/` so you can inspect the
exact pixels spent per step.

**Unit-test the pure vision layer** (no browser needed):

```bash
pip install pytest && pytest -q        # 8 passing
```

---

## Run in Docker

The image builds on Microsoft's official Playwright Python base, which ships a
pinned Chromium **and** every system library headless Chromium needs — the base
tag is matched exactly to our `playwright==1.48.0` pin so browser and client can
never drift. A multi-stage build keeps compilers and pip caches out of the final
image, and it runs as the non-root `pwuser`.

```bash
docker build -t vlm-viewport-optimizer .
docker run --rm -p 8000:8000 \
    -e VLM_HEADLESS=true \
    -e VLM_VIEWPORT_WIDTH=1280 -e VLM_VIEWPORT_HEIGHT=800 \
    vlm-viewport-optimizer
# → http://localhost:8000/docs   (built-in HEALTHCHECK hits /healthz)
```

---

## API

| Method | Path              | Purpose                                                         |
|--------|-------------------|-----------------------------------------------------------------|
| POST   | `/session/start`  | Launch a headless Chromium session · `{ "start_url": "..." }`   |
| POST   | `/session/view`   | Capture current state · `?session_id=&mode=macro\|delta`        |
| POST   | `/session/action` | Dispatch click/type/scroll/goto by `mark_id`, return next state |
| POST   | `/session/zoom`   | Return a 512×512 tile for one `mark_id`                          |
| GET    | `/session/stats`  | Cumulative vision-token savings scorecard for the session       |
| DELETE | `/session/{id}`   | Tear the session down                                           |
| GET    | `/agent/spec`     | Ready-to-use function-calling tool schema + system prompt       |
| GET    | `/healthz`        | Liveness + active session count                                 |

### Typical agent loop

```jsonc
// 1. start
POST /session/start            { "start_url": "https://example.com" }
// → { "session_id": "abc", "url": "https://example.com/" }

// 2. cheap labeled overview + the layout map the model reasons over
POST /session/view?session_id=abc&mode=macro
// → { "mode":"macro", "image_base64":"…",
//     "elements":[ { "mark_id":"A1", "role":"link", "name":"Learn more", "center":[67,105] } ] }

// 3. act by referring to a mark; get back ONLY what changed
POST /session/action?session_id=abc
     { "action":"click", "mark_id":"A1", "return_mode":"delta" }
// → { "mode":"delta", "change_region":{…}, "image_base64":"<mostly black>" }

// 4. need detail? one 512×512 tile, full fidelity
POST /session/zoom?session_id=abc    { "mark_id":"A3" }
// → { "mode":"zoom", "image_width":512, "image_height":512, "image_base64":"…" }
```

**Supported actions:** `click`, `double_click`, `type` (with `clear_first`),
`press` (a key), `hover`, `scroll` (`dx`/`dy`), `goto` (`url`). Every
element-targeted action fires at the exact viewport midpoint stored for the
referenced `mark_id`, so behavior matches what the model saw on the labeled frame.

---

## Measuring the savings (it keeps its own scorecard)

Every returned frame carries an honest token estimate, and each session tracks the
cumulative total versus a naive full-resolution loop:

```jsonc
// every ViewState includes:
"image_width": 768, "image_height": 480,
"estimated_vision_tokens": 492,          // what THIS frame costs
"estimated_full_frame_tokens": 1366,     // what a naive 1280×800 screenshot costs
"token_savings_ratio": 0.64              // 64% fewer vision tokens, this frame

// GET /session/stats → the running headline metric:
{ "frames_served": 12, "optimized_tokens_total": 5904,
  "naive_tokens_total": 16392, "tokens_saved": 10488, "savings_ratio": 0.64 }
```

Estimates come from [`vision/tokens.py`](src/vlm_optimizer/vision/tokens.py), which
models both the Anthropic pixel-area and OpenAI tile schemes (set
`VLM_TOKEN_MODEL_FAMILY=openai` to switch). A typical macro+delta loop runs **~60%+
cheaper** in vision tokens than re-sending full frames — and because it's measured,
not asserted, you can prove it in a demo.

## Plug it into a real LLM

`GET /agent/spec` returns a ready-to-use **function-calling tool schema** and a
**system prompt** — the exact artifacts Claude's `tools` API or OpenAI's function
calling expect:

```python
spec = httpx.get("http://localhost:8000/agent/spec").json()
# → spec["tool"]          # pass straight into the model's tools=[...]
# → spec["system_prompt"] # teaches the perceive→decide→act loop
```

And every view can return a **token-cheap `layout_text`** instead of making the
model parse verbose JSON — one line per element, everything it needs to act:

```
A1   link      "Learn more" @640,749
A2   input     "Search box" @249,105
A3   button    "Open" @423,105 [disabled]
```

The reference loop in [`examples/vlm_agent_loop.py`](examples/vlm_agent_loop.py)
shows the full perceive→decide→act cycle; swapping the mock model for a real one is
a single-method change.

---

## Production robustness

| Failure mode | Handling |
|--------------|----------|
| **Stale / detached DOM node** | `409 stale_element` when the execution context detaches or a node vanishes mid-action — the agent re-views and retries |
| **Unknown mark** | `422 unknown_mark`, returned *with* the list of currently-valid marks |
| **Capture/action race** | per-session `asyncio.Lock` serializes every view/action/zoom — a capture can never interleave with an action |
| **Slow / chatty pages** | bounded `networkidle` settle window; capture proceeds on timeout instead of hanging forever |
| **Nav / action / screenshot stalls** | `504 browser_timeout` instead of an indefinite hang |
| **Leaked sessions** | background reaper closes idle browsers past `VLM_SESSION_IDLE_TTL_S` |
| **Blocking the event loop** | all OpenCV/NumPy math dispatched via `asyncio.to_thread` |

---

## Configuration (env `VLM_*`)

| Var | Default | Meaning |
|-----|---------|---------|
| `VLM_HEADLESS` | `true` | Run Chromium headless |
| `VLM_VIEWPORT_WIDTH` / `_HEIGHT` | `1280` / `800` | Viewport size |
| `VLM_DELTA_THRESHOLD` | `25` | Per-pixel absdiff intensity counted as "changed" |
| `VLM_DELTA_MIN_AREA_PX` | `256` | Ignore change regions smaller than this |
| `VLM_DELTA_DILATE_PX` | `6` | Merge nearby change clusters into one box |
| `VLM_MACRO_MAX_WIDTH` | `768` | Macro overview downscale width |
| `VLM_DELTA_MAX_WIDTH` | `768` | Delta downscale width — the *real* delta token saver |
| `VLM_MACRO_JPEG_QUALITY` | `40` | Macro compression (deliberately low) |
| `VLM_TOKEN_MODEL_FAMILY` | `anthropic` | Token cost model: `anthropic` or `openai` |
| `VLM_CROP_TILE_SIZE` | `512` | Zoom tile size |
| `VLM_CROP_PADDING_RATIO` | `0.15` | Zoom padding around the element |
| `VLM_SESSION_IDLE_TTL_S` | `900` | Idle seconds before a session is reaped |

Full list with validation in [`config.py`](src/vlm_optimizer/config.py).

---

## What's in the box

```
vlm-viewport-optimizer/
├── src/vlm_optimizer/        # the framework (see architecture above)
├── examples/
│   ├── vlm_agent_loop.py     # reference VLM-agent loop (mock model, fully runnable)
│   └── demo_client.py        # minimal start→view→zoom→delta smoke test
├── tests/test_vision.py      # 8 unit tests for the pure image-math layer
├── Dockerfile                # multi-stage, Playwright base, non-root
├── run.py                    # dev entrypoint
└── requirements.txt
```

## Swapping in a real VLM

The reference loop's `MockVisionModel.decide(frame_png, elements)` receives exactly
what a production model would: the decoded frame bytes and the layout map. To go
live, replace that one method with a provider call — pass `image_base64` as the
image block and `elements` as text — and keep the rest of the loop untouched.

> For current Claude model IDs, pricing, and the Computer Use / tool-use request
> shape, consult the official Anthropic API docs rather than hard-coding values.

---

## Tech stack

**Python 3.11+** · **Playwright** (async Chromium) · **FastAPI** + **Uvicorn** ·
**OpenCV** (headless) · **NumPy** · **Pillow** · **Pydantic v2**
