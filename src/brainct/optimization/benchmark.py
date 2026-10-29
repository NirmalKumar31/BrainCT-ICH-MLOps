"""CPU latency of PyTorch FP32 vs ONNX Runtime FP32 vs INT8, per slice and per whole series.

    uv run python -m brainct.optimization.benchmark --run artifacts/runs/cnn25d_resnet18 \
        --work artifacts/bundle.work --out docs/benchmark.json

Series latency is the deployed chain (CNN over all slices in batches of 16, then the series
model), on a 32-slice series (median RSNA series length here is 34).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from brainct.constants import IMAGE_SIZE
from brainct.serving.model import contexts
from brainct.training.model import SliceNet


def timed(fn, warmup: int, reps: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    a = np.asarray(times)
    return {"p50_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95))}


def session(path: Path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("artifacts/bundle.work"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("docs/benchmark.json"))
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)

    ck = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
    net = SliceNet(ck["config"]["backbone"], ck["config"]["context"], pretrained=False).eval()
    net.load_state_dict(ck["model"])
    k = 2 * net.context + 1
    rng = np.random.default_rng(0)
    vol = rng.integers(-1000, 1500, (32, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    xs = contexts(vol, net.context)
    x1 = xs[:1]
    seq = session(args.work / "seq.onnx", args.threads)

    results: dict[str, dict] = {}
    with torch.inference_mode():
        results["pytorch_fp32"] = {
            "slice": timed(lambda: net(torch.from_numpy(x1)), 3, args.reps),
            "series_cnn_only": timed(
                lambda: [net(torch.from_numpy(xs[i : i + 16])) for i in (0, 16)],
                2, max(5, args.reps // 3),
            ),
        }  # fmt: skip
    for precision in ("fp32", "int8"):
        cnn = session(args.work / f"cnn.{precision}.onnx", args.threads)

        def chain(cnn=cnn):
            x = contexts(vol, net.context)
            outs = [cnn.run(["logits", "embedding"], {"hu": x[i : i + 16]}) for i in (0, 16)]
            lg = np.concatenate([o[0] for o in outs])[None]
            emb = np.concatenate([o[1] for o in outs])[None]
            seq.run(["refined"], {"embedding": emb, "logits": lg})

        results[f"onnx_{precision}"] = {
            "slice": timed(lambda cnn=cnn: cnn.run(["logits"], {"hu": x1}), 5, args.reps),
            "series_chain": timed(chain, 2, max(5, args.reps // 3)),
            "size_mb": (args.work / f"cnn.{precision}.onnx").stat().st_size / 1e6,
        }
    report = {
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "cpu_count": os.cpu_count(), "threads": args.threads},
        "input": {"slice": list(x1.shape), "series_slices": 32, "context_channels": k},
        "results": results,
    }  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
