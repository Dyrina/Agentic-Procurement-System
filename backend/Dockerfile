FROM python:3.14-slim

# Install uv 0.8.x
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /bin/

WORKDIR /app

# Copy dependency manifests first (layer cache optimisation)
COPY pyproject.toml uv.lock ./

# Let uv create .venv and install dependencies directly in the container
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source and tests
COPY src/ ./src/
COPY tests/ ./tests/

# Non-root user for security
RUN groupadd --system app && useradd --system --gid app app \
    && chown -R app:app /app
USER app

# All deps are pre-installed in .venv — skip uv cache writes
ENV UV_NO_CACHE=1

EXPOSE 8000

ENTRYPOINT ["uv", "run"]
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
