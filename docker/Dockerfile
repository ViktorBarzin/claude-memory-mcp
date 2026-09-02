# ── stage 1: export the model to ONNX, throw the toolchain away ──────────────────
# torch, optimum and sentence-transformers are ~3 GB and are needed only to trace the
# graph and to produce the reference vectors the gate compares against. They stay in this
# stage so the runtime image carries onnxruntime and a tokenizer instead of a training
# framework. Runs on GitHub Actions per ADR-0002 (build compute is off-infra).
#
# THIS STAGE MUST NOT DEPEND ON src/. The export measured 1,686 s, and while the gate
# lived in the same stage a one-line change under src/ re-ran all of it. The gate moved to
# its own stage below; this one sees only docker/export_onnx.py.
FROM python:3.12-slim AS exporter

RUN pip install --no-cache-dir \
      "optimum[onnxruntime]>=1.24" "transformers>=4.45" "torch>=2.4" \
      "sentence-transformers>=3.0" "tokenizers>=0.20" \
      --extra-index-url https://download.pytorch.org/whl/cpu

ENV OUT_DIR=/export \
    GATE_REFERENCE=/gate-reference.json

# `touch -d @0` is load-bearing, not tidiness. A layer is a tar and tar headers carry
# mtime, so the 1,076 MB `COPY --from=exporter` below gets a fresh digest whenever this
# step re-runs — even when it produces identical bytes. CI caches with
# `type=gha,mode=max` into a 10 GB per-repo LRU that these intermediates already exceed,
# so this step IS evicted and re-run periodically. Fixing the mtimes makes the copied
# layer reproducible across those re-runs. (Whether the export itself is byte-reproducible
# is untested — that decides whether this is sufficient as well as necessary.)
COPY docker/export_onnx.py /export_onnx.py
RUN python /export_onnx.py export \
    && find /export /gate-reference.json -exec touch -d @0 {} +


# ── stage 1b: the fidelity gate ─────────────────────────────────────────────────
# Scores the graph stage 1 wrote, through the PRODUCTION inference path, and fails the
# build if it drifted from the sentence-transformers reference. Reasons it is its own
# stage: it needs `claude_memory` importable and therefore src/, which would otherwise
# drag the export down below a source-dependent layer; and it now scores the exact bytes
# the runtime copies rather than a staging directory assembled beside them.
#
# The gate is not optional decoration: the first version of this image shipped a graph
# whose vectors scored cosine 0.16-0.37 against the reference while passing every cheap
# check (dim, unit norm, conventions, latency).
FROM exporter AS gate

WORKDIR /gate
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .
RUN python /export_onnx.py gate \
    && touch -d @0 /gate-passed


# ── stage 2: runtime ────────────────────────────────────────────────────────────
# LAYER ORDER IS LOAD-BEARING. Every source-independent layer sits above every
# source-dependent one, because BuildKit re-executes every layer above a cache miss and
# re-running a COPY re-tars its content with a fresh mtime. With `COPY src/` above them,
# 3,023.8 MB of this image's 3,067 MB re-shipped on every commit and only 44.1 MB of
# layers were shared; each cold pull of it writes 9.2-12.3 GB to k8s-node1's root
# filesystem, which is what filled that disk and cost 130 cached images to kubelet's image
# GC on 2026-09-01. Anything added below the marker near the bottom re-ships per commit.
FROM python:3.12-slim AS base

WORKDIR /app

COPY pyproject.toml README.md alembic.ini ./

# The third-party dependency set, read out of pyproject.toml so that list keeps one home.
# Installing ".[api]" here instead would need src/ present, because hatchling builds the
# wheel from it — which is what put the 2,094.6 MB CUDA install below the source layers.
RUN python -c "import tomllib; p = tomllib.load(open('pyproject.toml', 'rb'))['project']; print(*(list(p.get('dependencies', [])) + p['optional-dependencies']['api']), sep=chr(10))" > /tmp/requirements-api.txt \
    && cat /tmp/requirements-api.txt \
    && pip install --no-cache-dir -r /tmp/requirements-api.txt

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
RUN pip install --no-cache-dir "onnxruntime-gpu[cuda,cudnn]~=1.26.0" \
         "nvidia-cublas-cu12~=12.0" "tokenizers>=0.20" "numpy>=1.26"

# Both CPU-fallback regressions above were SILENT: onnxruntime does not raise when a
# requested execution provider is unavailable, it logs to stderr and hands back a working
# CPU session. So the install is checked here, at build time, where a failure is loud.
# `ort.preload_dlls()` first is not optional — the nvidia pip wheels install their .so
# files under site-packages/nvidia/<lib>/lib/, which is off the default loader path, so a
# bare CDLL by soname fails on an image that is in fact correct (embeddings.py calls the
# same helper for the same reason). `pip check` catches the version conflicts that produce
# the mismatched soname in the first place.
RUN pip check \
    && python -c "import onnxruntime as ort; ort.preload_dlls(); import ctypes; ctypes.CDLL('libcublasLt.so.12'); ctypes.CDLL('libcudnn.so.9'); print('[cuda-guard] onnxruntime', ort.__version__, 'providers', ort.get_available_providers())"

# The gated graph + tokenizer, baked in: a pod start needs no network and no PVC, and the
# model version is pinned by the image tag rather than by whatever the Hub serves. This is
# 1,076.0 MB and depends on no source file, so it belongs above src/ — below it, this copy
# re-tarred with fresh mtimes on every commit even when the exporter stage was CACHED.
COPY --from=exporter /export /opt/onnx-embed
RUN chmod -R a+rX /opt/onnx-embed
ENV MEMORY_ONNX_MODEL_DIR=/opt/onnx-embed

# No Hugging Face access at runtime: nothing here loads from the Hub, and an accidental
# reintroduction should fail loudly rather than add a startup dependency on huggingface.co.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# ── everything below this line depends on src/ and re-ships on every source commit ──
COPY src/ src/
COPY migrations/ migrations/
RUN pip install --no-cache-dir --no-deps . \
    && pip check

# The gate ran in its own stage, against the exact bytes copied into /opt/onnx-embed.
# Copying its marker is what puts that stage in this image's dependency graph: BuildKit
# prunes stages nothing references, so without this line the gate would not run at all and
# a graph that reproduces nothing could ship. The marker is empty and mtime-fixed, so the
# layer is stable, and it sits here because it depends on src/ through the gate stage.
COPY --from=gate /gate-passed /gate-passed

RUN useradd -r -u 1000 app
USER app

EXPOSE 8000

CMD ["uvicorn", "claude_memory.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
