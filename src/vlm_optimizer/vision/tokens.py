"""Vision-token cost estimation.

Lets the framework *quantify* its own savings: how many vision tokens a returned
frame costs versus a naive full-resolution screenshot of the same viewport.

We model the two dominant cost schemes honestly:

* **Anthropic (Claude)** — tokens are approximately ``(width * height) / 750``.
  Cost scales with total pixel area, so **downscaling and cropping are what save
  tokens.** Blacking out pixels at the same resolution does NOT reduce this count
  (it reduces *bytes* and helps model attention, but not the encoder's token bill).

* **OpenAI (GPT-4o, high detail)** — the image is tiled into 512px squares and
  each tile is a fixed cost: ``85 + 170 * num_tiles``. Again, fewer pixels =>
  fewer tiles => fewer tokens; a mostly-black full-res image has the same tile
  count as a busy one.

This is deliberately the honest story. It is *why* the optimizer downscales the
macro frame, downscales delta frames, and crops zoom tiles rather than relying on
black-masking alone to cut cost.
"""

from __future__ import annotations

import math
from typing import Literal

ModelFamily = Literal["anthropic", "openai"]

_ANTHROPIC_DIVISOR = 750  # tokens ~= pixels / 750  (documented approximation)
_OPENAI_TILE_PX = 512
_OPENAI_BASE_TOKENS = 85
_OPENAI_TOKENS_PER_TILE = 170


def estimate_vision_tokens(
    width: int, height: int, *, family: ModelFamily | str = "anthropic"
) -> int:
    """Estimate the vision-input tokens an image of ``width x height`` costs."""
    width = max(int(width), 1)
    height = max(int(height), 1)
    if family == "openai":
        tiles = math.ceil(width / _OPENAI_TILE_PX) * math.ceil(height / _OPENAI_TILE_PX)
        return _OPENAI_BASE_TOKENS + _OPENAI_TOKENS_PER_TILE * tiles
    # Default: Anthropic pixel-area approximation.
    return math.ceil((width * height) / _ANTHROPIC_DIVISOR)


def savings_ratio(optimized_tokens: int, baseline_tokens: int) -> float:
    """Fraction of tokens saved vs. the baseline, clamped to ``[0.0, 1.0]``."""
    if baseline_tokens <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - optimized_tokens / baseline_tokens))
