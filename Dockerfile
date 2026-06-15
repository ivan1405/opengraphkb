# ── Builder stage ─────────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first for better cache utilization
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev --no-cache --no-install-project

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ /app/src/

# Create data directory
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["python", "-m", "main"]
