FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# 1. Copy metadata AND the README (required by Hatchling/pyproject.toml)
COPY pyproject.toml uv.lock README.md ./

# 2. Install dependencies (this creates the .venv)
# Using --no-install-project here prevents it from failing if source isn't there yet
RUN uv sync --frozen --no-dev --no-install-project

# 3. Copy your application source code
COPY trading_cli/ ./trading_cli/

# 4. Final sync to install the local 'trading-cli' package into the venv
RUN uv sync --frozen --no-dev

# Environment variables
ENV TOKENIZERS_PARALLELISM=false \
    TRANSFORMERS_VERBOSITY=error \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TQDM_DISABLE=1 \
    PYTHONUNBUFFERED=1

# Ensure the venv is on the PATH so 'uv run' isn't always strictly necessary
ENV PATH="/app/.venv/bin:$PATH"

CMD ["uv", "run", "trading-cli"]
