"""Build-time export of the dense-leg model to ONNX, run on GitHub Actions.

Runs in the Dockerfile's throwaway `exporter` stage, never at runtime and never in the
cluster (ADR-0002 keeps build compute off the homelab). torch and optimum are needed to
trace the graph and are ~2.5 GB of dependencies; keeping them in a stage that is thrown
away is why the runtime image carries onnxruntime and a tokenizer instead.

Output, into $OUT_DIR:
  model_int8.onnx        dynamically-quantised graph, the one both providers load today
  model_int8.onnx_data   external tensor data, when the exporter splits it out
  tokenizer.json         the fast tokenizer, loaded directly by `tokenizers`

int8 is a CPU optimisation. Whether the CUDA provider serves it well on our Turing T4 is
unmeasured, so a follow-up may add model_fp16.onnx and point CUDA at it — that costs
~1.2 GB of image, so it is not paid until a measurement asks for it.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

MODEL = os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
OUT_DIR = Path(os.environ.get("OUT_DIR", "/export"))
WORK = Path("/tmp/onnx-export")


def _only_onnx(directory: Path) -> Path:
    graphs = sorted(p for p in directory.iterdir() if p.suffix == ".onnx")
    if len(graphs) != 1:
        raise SystemExit(f"expected exactly one .onnx in {directory}, found {[p.name for p in graphs]}")
    return graphs[0]


def main() -> int:
    from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    fp32_dir, int8_dir = WORK / "fp32", WORK / "int8"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[export] tracing {MODEL} to ONNX", flush=True)
    ORTModelForFeatureExtraction.from_pretrained(MODEL, export=True).save_pretrained(fp32_dir)

    print("[export] quantising to int8 (avx512_vnni, dynamic, per-channel)", flush=True)
    ORTQuantizer.from_pretrained(fp32_dir).quantize(
        save_dir=int8_dir,
        quantization_config=AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True),
    )

    graph = _only_onnx(int8_dir)
    shutil.copyfile(graph, OUT_DIR / "model_int8.onnx")
    for extra in int8_dir.glob("*.onnx_data"):
        shutil.copyfile(extra, OUT_DIR / "model_int8.onnx_data")

    # `tokenizers` loads tokenizer.json directly, so transformers is not needed at runtime.
    AutoTokenizer.from_pretrained(MODEL).save_pretrained(OUT_DIR)
    if not (OUT_DIR / "tokenizer.json").exists():
        raise SystemExit(f"{MODEL} did not produce a fast tokenizer.json; the runtime cannot tokenise without it")

    for path in sorted(OUT_DIR.iterdir()):
        print(f"[export] {path.name:28} {path.stat().st_size / 1e6:8.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
