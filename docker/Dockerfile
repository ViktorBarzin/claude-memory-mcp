FROM python:3.12-slim AS base

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir ".[api]"

RUN useradd -r -u 1000 app
USER app

EXPOSE 8000

CMD ["uvicorn", "claude_memory.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
