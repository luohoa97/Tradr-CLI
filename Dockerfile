FROM ghcr.io/astral-sh/uv:0.5-python3.11

WORKDIR /app

# Copy project metadata and dependencies
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Copy application code
COPY trading_cli/ ./trading_cli/

# Re-sync to ensure local package is installed
RUN uv sync --frozen --no-dev

# Set environment variables for non-interactive ML
ENV TOKENIZERS_PARALLELISM=false
ENV TRANSFORMERS_VERBOSITY=error
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TQDM_DISABLE=1
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["uv", "run", "trading-cli"]
