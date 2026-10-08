# syntax=docker/dockerfile:1.7
#
# Production image for the VLM-Viewport-Optimizer.
#
# We build on Microsoft's official Playwright Python image, which already ships:
#   * a pinned Chromium build under /ms-playwright, AND
#   * every system library that headless Chromium needs on Debian/Ubuntu
#     (libnss3, libatk, libgbm, fonts, etc.) — the stuff that otherwise makes
#     "it works on my mac, not in the container" a day-long debugging session.
#
# The base tag (v1.48.0) is pinned to EXACTLY our playwright==1.48.0 dependency
# so the browser binary and the client library can never drift apart.
#
# Stage 1 compiles all Python deps into an isolated virtualenv; stage 2 copies
# only that venv (no build toolchain, no pip cache) onto a clean copy of the
# same base, keeping the final image lean and free of compilers.

ARG PLAYWRIGHT_VERSION=v1.48.0-jammy

# --------------------------------------------------------------------------- #
# Stage 1 — builder: resolve and compile Python dependencies into a venv
# --------------------------------------------------------------------------- #
FROM mcr.microsoft.com/playwright/python:${PLAYWRIGHT_VERSION} AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Create the venv first so dependency layers cache independently of app code.
RUN python -m venv "$VIRTUAL_ENV" && pip install --upgrade pip

# opencv-python-headless needs no GUI libs, but libGL/libglib are cheap insurance
# for the handful of cv2 codecs that still dlopen them at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

# --------------------------------------------------------------------------- #
# Stage 2 — runtime: clean base + the prebuilt venv + application source
# --------------------------------------------------------------------------- #
FROM mcr.microsoft.com/playwright/python:${PLAYWRIGHT_VERSION} AS runtime

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    # App defaults (override at `docker run` time with -e VLM_*).
    VLM_HEADLESS=true \
    VLM_HOST=0.0.0.0 \
    VLM_PORT=8000

# Runtime copies of the few shared libs installed in the builder.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Bring over the fully-resolved virtualenv (no pip/build tooling leaks in).
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY src ./src
COPY run.py pyproject.toml README.md ./
COPY examples ./examples

# The Playwright base image provides the non-root `pwuser`; run as it and give
# it ownership of the app so the browser sandbox and app files are least-priv.
RUN chown -R pwuser:pwuser /app
USER pwuser

EXPOSE 8000

# Lightweight liveness probe against the dedicated health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request,sys; \
url=f\"http://127.0.0.1:{os.environ.get('VLM_PORT','8000')}/healthz\"; \
sys.exit(0 if urllib.request.urlopen(url, timeout=3).status==200 else 1)"

# Chromium is already installed at /ms-playwright by the base image, so there is
# no `playwright install` step — the binary and the pinned client always match.
CMD ["uvicorn", "vlm_optimizer.api.app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
