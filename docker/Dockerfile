# ── stage 1: export the dense-leg model to ONNX, then throw the toolchain away ──
# torch + optimum are ~2.5 GB and are needed only to trace and quantise the graph. They
# stay in this stage so the runtime image carries onnxruntime and a tokenizer instead of
# a training framework. Runs on GitHub Actions per ADR-0002 (build compute is off-infra).
FROM python:3.12-slim AS exporter

RUN pip install --no-cache-dir \
      "optimum[onnxruntime]>=1.24" "transformers>=4.45" "torch>=2.4" \
      --extra-index-url https://download.pytorch.org/whl/cpu

COPY docker/export_onnx.py /export_onnx.py
ENV OUT_DIR=/export
RUN python /export_onnx.py


# ── stage 2: runtime ────────────────────────────────────────────────────────────
FROM python:3.12-slim AS base

WORKDIR /app

COPY pyproject.toml README.md alembic.ini ./
COPY src/ src/
COPY migrations/ migrations/

# onnxruntime-gpu carries the CUDA and cuDNN userspace libraries via its extras; the
# NVIDIA container runtime injects only the driver. It also serves the CPU provider, so
# one wheel covers both the GPU path and its fallback. No torch, no sentence-transformers.
RUN pip install --no-cache-dir ".[api]" \
    && pip install --no-cache-dir "onnxruntime-gpu[cuda,cudnn]==1.29.0" "tokenizers>=0.20" "numpy>=1.26"

# The exported graph + tokenizer, baked in: a pod start needs no network and no PVC, and
# the model version is pinned by the image tag rather than by whatever the Hub serves.
COPY --from=exporter /export /opt/onnx-embed
RUN chmod -R a+rX /opt/onnx-embed
ENV MEMORY_ONNX_MODEL_DIR=/opt/onnx-embed

# No Hugging Face access at runtime: nothing here loads from the Hub, and an accidental
# reintroduction should fail loudly rather than add a startup dependency on huggingface.co.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN useradd -r -u 1000 app
USER app

EXPOSE 8000

CMD ["uvicorn", "claude_memory.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
