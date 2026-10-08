"""Centralized, environment-overridable configuration.

Every tunable lives here so the networking, automation, and image-math layers
never hard-code magic numbers. Override any field with an environment variable
prefixed ``VLM_`` (e.g. ``VLM_HEADLESS=false``) or a local ``.env`` file.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed runtime settings for the whole framework."""

    model_config = SettingsConfigDict(env_prefix="VLM_", env_file=".env", extra="ignore")

    # ---- Browser ---------------------------------------------------------
    headless: bool = Field(default=True, description="Run Chromium headless.")
    viewport_width: int = Field(default=1280, ge=320, le=3840)
    viewport_height: int = Field(default=800, ge=240, le=2160)
    device_scale_factor: float = Field(
        default=1.0,
        ge=1.0,
        le=3.0,
        description="Keep at 1.0 so screenshot pixels map 1:1 to DOM CSS coordinates.",
    )
    user_agent: str | None = Field(default=None, description="Optional UA override.")

    # ---- Timeouts (ms) ---------------------------------------------------
    nav_timeout_ms: int = Field(default=30_000, ge=1_000)
    action_timeout_ms: int = Field(default=10_000, ge=500)
    settle_timeout_ms: int = Field(
        default=1_500,
        ge=0,
        description="Grace window to let the DOM settle after an action before capture.",
    )

    # ---- Set-of-Mark labeling -------------------------------------------
    label_font_scale: float = Field(default=0.4, gt=0)
    label_thickness: int = Field(default=1, ge=1)
    label_pad_px: int = Field(default=2, ge=0)

    # ---- Delta / differential engine ------------------------------------
    delta_threshold: int = Field(
        default=25,
        ge=0,
        le=255,
        description="Per-pixel absdiff intensity above which a pixel counts as 'changed'.",
    )
    delta_min_area_px: int = Field(
        default=256,
        ge=1,
        description="Ignore change regions smaller than this (anti-aliasing / cursor noise).",
    )
    delta_dilate_px: int = Field(
        default=6,
        ge=0,
        description="Morphological dilation to merge nearby change clusters into one bbox.",
    )

    # ---- Hierarchical crop / zoom ---------------------------------------
    macro_max_width: int = Field(
        default=768,
        ge=128,
        description="Width the macro (whole-viewport) frame is downscaled to.",
    )
    macro_jpeg_quality: int = Field(default=40, ge=1, le=100)
    crop_padding_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
    crop_tile_size: int = Field(
        default=512,
        ge=64,
        description="Output square size for zoom tiles (OpenAI/Anthropic 512px tile grid).",
    )
    crop_jpeg_quality: int = Field(default=80, ge=1, le=100)

    # ---- Session lifecycle ----------------------------------------------
    session_idle_ttl_s: int = Field(
        default=900,
        ge=30,
        description="Seconds a session may stay idle before it is eligible for reaping.",
    )


settings = Settings()
"""Process-wide singleton. Import this, not the class."""
