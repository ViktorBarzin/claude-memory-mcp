"""Build-time export of the dense-leg model to ONNX, with a fidelity gate.

Runs in the Dockerfile's throwaway `exporter` stage on GitHub Actions (ADR-0002 keeps
build compute off the homelab). torch, optimum and sentence-transformers are needed to
trace the graph and to produce the reference vectors; keeping them in a stage that is
discarded is why the runtime image carries onnxruntime and a tokenizer instead.

THE GATE, and why it exists
---------------------------
The first attempt at this shipped an int8 graph that produced vectors with cosine
0.16-0.37 against the sentence-transformers reference for identical text, where a
faithful export scores >0.99. Nothing cheap caught it: the dimension was 1024, the L2
norm was exactly 1.0000, the query instruction was applied, the sensitive gate held, and
the latency was plausible. Only comparing against a known-good reference exposed it, and
by then the graph was already in a deployed image.

So the export now embeds a fixed probe set two ways — through sentence-transformers in
fp32, and through the exported graph using THE PRODUCTION INFERENCE PATH
(``claude_memory.embeddings.OnnxEmbedder``, not a reimplementation) — and fails the
build unless every probe clears MIN_COSINE. A graph that would rank badly can no longer
reach a registry.

Output, into $OUT_DIR:
  model.onnx            the gated graph (plus model.onnx_data when split)
  tokenizer.json        the fast tokenizer, loaded directly by `tokenizers`
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

MODEL = os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
OUT_DIR = Path(os.environ.get("OUT_DIR", "/export"))
WORK = Path("/tmp/onnx-export")

#: A faithful export reproduces the reference almost exactly. 0.99 is loose enough for
#: fp16 rounding and tight enough that the 0.16-0.37 int8 collapse cannot pass.
MIN_COSINE = 0.99

#: Probes span what recall actually sees: a short English query, a long English passage,
#: and Bulgarian — the content the multilingual swap is for, and the case an
#: English-centric mistake would break first.
PROBES = {
    "query_en": ("query", "which DNS server do we use at home"),
    "query_bg": ("query", "кой DNS сървър използваме вкъщи"),
    "doc_en": ("document", "Technitium DNS runs at 10.0.20.201 and is the first place to check."),
    "doc_bg": ("document", "Технитиум DNS е на 10.0.20.201 и е първото място за проверка."),
    "doc_long": ("document", "The gpu-vram-watchdog recycles the biggest over-budget GPU tenant, "
                             "but only when free VRAM drops below the floor and something else is "
                             "actually blocked on the card. " * 4),
}


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb)


def _reference() -> dict[str, list[float]]:
    """Ground truth: the model as sentence-transformers runs it, fp32 on CPU."""
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(MODEL, device="cpu")
    out: dict[str, list[float]] = {}
    for name, (kind, text) in PROBES.items():
        kwargs = {"prompt_name": "query"} if kind == "query" else {}
        out[name] = st.encode([text], normalize_embeddings=True, **kwargs)[0].tolist()
    return out


def _check(model_dir: Path, label: str, reference: dict[str, list[float]]) -> None:
    """Score the exported graph through the PRODUCTION embedder and fail if it drifted."""
    os.environ["MEMORY_ONNX_MODEL_DIR"] = str(model_dir)
    from claude_memory.embeddings import OnnxEmbedder

    embedder = OnnxEmbedder(model_dir=str(model_dir), providers=["CPUExecutionProvider"])
    worst = 1.0
    print(f"[gate] {label}: cosine vs sentence-transformers reference", flush=True)
    for name, (kind, text) in PROBES.items():
        vec = embedder.embed_query(text) if kind == "query" else embedder.embed_document(text, is_sensitive=False)
        assert vec is not None
        score = _cosine(vec, reference[name])
        worst = min(worst, score)
        flag = "ok" if score >= MIN_COSINE else "FAIL"
        print(f"[gate]   {name:10} {score:.5f}  {flag}", flush=True)
    if worst < MIN_COSINE:
        raise SystemExit(
            f"[gate] {label} REJECTED: worst cosine {worst:.5f} < {MIN_COSINE}. The graph does not "
            "reproduce the reference model, so it would rank badly while looking healthy. "
            "Refusing to bake it into the image."
        )
    print(f"[gate] {label} accepted (worst {worst:.5f})", flush=True)


def _only_onnx(directory: Path) -> Path:
    graphs = sorted(p for p in directory.iterdir() if p.suffix == ".onnx")
    if len(graphs) != 1:
        raise SystemExit(f"expected exactly one .onnx in {directory}, found {[p.name for p in graphs]}")
    return graphs[0]


def _stage(src_dir: Path) -> Path:
    """Copy the exported graph and a tokenizer into a directory the embedder can load.

    Names are preserved exactly. A large export is split into ``model.onnx`` plus
    ``model.onnx_data``, and that reference is stored INSIDE the graph by filename, so
    renaming the graph leaves it pointing at a data file that is not there. Production
    loads ``model.onnx`` for the same reason: no renaming anywhere in the pipeline.
    """
    staged = WORK / "staged"
    staged.mkdir(parents=True, exist_ok=True)
    for original in src_dir.glob("*.onnx*"):
        shutil.copyfile(original, staged / original.name)
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(MODEL).save_pretrained(staged)
    return staged


def main() -> int:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    fp32_dir = WORK / "fp32"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[export] computing reference vectors with sentence-transformers ({MODEL})", flush=True)
    reference = _reference()

    print(f"[export] tracing {MODEL} to ONNX", flush=True)
    ORTModelForFeatureExtraction.from_pretrained(MODEL, export=True).save_pretrained(fp32_dir)

    # fp32 is what ships. fp16 was attempted and abandoned on 2026-09-01: converting this
    # graph with onnxconverter_common fails both ways round. With shape inference on it
    # cannot serialise, because the fp32 model is ~2.4 GB and protobuf refuses past 2 GB;
    # with inference off the converter cannot place casts and emits a type-inconsistent
    # graph that onnxruntime rejects at load ("Type parameter (T) of Optype (Add) bound to
    # different types"). A T4 runs fp32 on a 0.6B model comfortably, and this artifact is
    # the one the gate has actually accepted, so it is worth the larger image.
    _check(_stage(fp32_dir), "fp32 export", reference)

    for produced in fp32_dir.glob("*.onnx*"):
        shutil.copyfile(produced, OUT_DIR / produced.name)
    AutoTokenizer.from_pretrained(MODEL).save_pretrained(OUT_DIR)
    if not (OUT_DIR / "tokenizer.json").exists():
        raise SystemExit(f"{MODEL} produced no fast tokenizer.json; the runtime cannot tokenise")

    for path in sorted(OUT_DIR.iterdir()):
        print(f"[export] {path.name:28} {path.stat().st_size / 1e6:8.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
