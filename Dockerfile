FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# 1. Install dependencies FIRST, from the lockfile only.
#    This layer is cached and only rebuilds when deps change — not on code edits.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# 2. Copy application code + trained weights.
COPY src/ ./src/
COPY models/model.pth ./models/model.pth

# 3. Install the project itself into the environment.
RUN uv sync --frozen --no-dev

EXPOSE 8000

# --host 0.0.0.0 is REQUIRED in a container (not 127.0.0.1) to be reachable.
CMD ["uv", "run", "uvicorn", "fashion_mnist.serve:app", "--host", "0.0.0.0", "--port", "8000"]
