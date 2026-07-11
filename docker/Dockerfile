FROM python:3.12-slim AS base

WORKDIR /app

COPY pyproject.toml README.md alembic.ini ./
COPY src/ src/
COPY migrations/ migrations/

RUN pip install --no-cache-dir ".[api,embeddings]" --extra-index-url https://download.pytorch.org/whl/cpu

# Bake the dense-leg model into the image (build-time download on GHA, per
# ADR-0002 off-infra builds). Rationale: the runtime user is a system account
# with no writable $HOME, and a runtime HuggingFace download would re-pull
# ~1.3GB on every pod restart and add an external dependency to startup.
# HF_HOME points at the baked cache; a+rwX because sentence-transformers
# takes .locks in the cache dir even for read-path loads.
ENV HF_HOME=/opt/hf-cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')" \
    && chmod -R a+rwX /opt/hf-cache

RUN useradd -r -u 1000 app
USER app

EXPOSE 8000

CMD ["uvicorn", "claude_memory.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
