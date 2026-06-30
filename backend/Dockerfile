FROM python:3.14-slim

# Install uv 0.8.*
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency manifest first for layer caching
COPY pyproject.toml ./

# Install dependencies via uv
RUN uv sync --no-dev --no-install-project

# Copy source code
COPY src/ ./src/

# Expose the API port
EXPOSE 8000

# Run the FastAPI server
ENTRYPOINT ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
