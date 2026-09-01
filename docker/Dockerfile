# ── stage 1: export the model to ONNX, gate its fidelity, throw the toolchain away ──
# torch, optimum and sentence-transformers are ~3 GB and are needed only to trace the
# graph and to produce the reference vectors the gate compares against. They stay in this
# stage so the runtime image carries onnxruntime and a tokenizer instead of a training
# framework. Runs on GitHub Actions per ADR-0002 (build compute is off-infra).
#
# The gate is not optional decoration: the first version of this image shipped a graph
# whose vectors scored cosine 0.16-0.37 against the reference while passing every cheap
# check (dim, unit norm, conventions, latency). export_onnx.py now fails the build unless
# the exported graph reproduces sentence-transformers to within 0.99, checked through the
# production inference path rather than a reimplementation of it.
FROM python:3.12-slim AS exporter

RUN pip install --no-cache-dir \
      "optimum[onnxruntime]>=1.24" "transformers>=4.45" "torch>=2.4" \
      "sentence-transformers>=3.0" "tokenizers>=0.20" \
      --extra-index-url https://download.pytorch.org/whl/cpu

# The gate runs the REAL embedder, so the package has to be importable here.
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .

COPY docker/export_onnx.py /export_onnx.py
ENV OUT_DIR=/export
RUN python /export_onnx.py


# ── stage 2: runtime ────────────────────────────────────────────────────────────
FROM python:3.12-slim AS base

WORKDIR /app

COPY pyproject.toml README.md alembic.ini ./
COPY src/ src/
COPY migrations/ migrations/

# nvidia-cublas-cu12 is listed EXPLICITLY because onnxruntime does not depend on it.
# Its [cuda,cudnn] extras pull nvrtc, cuda-runtime, cufft, curand and cudnn — and not
# cuBLAS, which the CUDA execution provider nonetheless links. Measured twice: the
# provider failed to load with "libcublasLt.so.13: cannot open shared object file" on
# ORT 1.29 and again with "libcublasLt.so.12" on 1.26, and in both cases onnxruntime
# silently served every embed on the CPU rather than erroring.
#
# onnxruntime-gpu is PINNED to 1.26.x on purpose. From 1.27 the GPU wheel is built
# against CUDA 13, and k8s-node1 runs driver 570 / CUDA 12.8 — 1.29 installed cleanly and
# then failed at runtime with "libcublasLt.so.13: cannot open shared object file",
# silently serving every embed on the CPU. 1.26.x is the last release built for CUDA 12.8.
# The [cuda,cudnn] extras carry the CUDA and cuDNN userspace libraries; the NVIDIA
# container runtime injects only the driver. One wheel serves both the GPU path and its
# CPU fallback. No torch, no sentence-transformers.
RUN pip install --no-cache-dir ".[api]" \
    && pip install --no-cache-dir "onnxruntime-gpu[cuda,cudnn]~=1.26.0" \
         "nvidia-cublas-cu12~=12.0" "tokenizers>=0.20" "numpy>=1.26"

# The gated graph + tokenizer, baked in: a pod start needs no network and no PVC, and the
# model version is pinned by the image tag rather than by whatever the Hub serves.
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
