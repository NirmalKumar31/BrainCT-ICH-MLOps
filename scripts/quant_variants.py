"""Compare INT8 quantization schemes against FP32 on this machine's CPU.

    uv run python scripts/quant_variants.py build artifacts/runs/cnn2d_resnet18 artifacts/qv \
        [artifacts/bundle.work]   # export work dir of that run (holds cnn.prep.onnx)
    python scripts/quant_variants.py check artifacts/qv        # any machine with onnxruntime

`build` quantizes the prepared FP32 CNN four ways (calibrated on the same training slices).
`check` runs each variant and FP32 on the self-test phantom (and, when present, real
validation contexts saved by `build`) and prints max / mean probability error per variant.
Run `check` on every CPU family you deploy to: INT8 kernels differ between ARM, x86 with VNNI
and x86 without it (AVX2 only), and that difference broke serving on GKE (AMD EPYC Milan).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

VARIANTS = {
    "s8_full": {"weight": "QInt8", "reduce_range": False},
    "s8_reduce_range": {"weight": "QInt8", "reduce_range": True},
    "u8_full": {"weight": "QUInt8", "reduce_range": False},
    "u8_reduce_range": {"weight": "QUInt8", "reduce_range": True},
}


def build(run: Path, out: Path) -> None:
    import torch  # noqa: F401  (training extra; build runs where the checkpoint lives)
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    from brainct.optimization.export import (
        SliceReader,
        calibration_batches,
        selftest_volume,
        windowing_nodes,
    )
    from brainct.serving.model import contexts
    from brainct.training.data import load_series, read_volume

    out.mkdir(parents=True, exist_ok=True)
    work = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("artifacts/bundle.work")
    prep = work / "cnn.prep.onnx"
    cfg = json.loads((run / "config.json").read_text())
    train = load_series(Path(cfg["data_dir"]), "train")
    batches = calibration_batches(train, cfg["context"], 192, seed=7)
    excluded = windowing_nodes(prep)
    for name, v in VARIANTS.items():
        t0 = time.time()
        quantize_static(
            str(prep),
            str(out / f"{name}.onnx"),
            SliceReader(batches),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=getattr(QuantType, v["weight"]),
            activation_type=QuantType.QUInt8,
            calibrate_method=CalibrationMethod.Percentile,
            nodes_to_exclude=excluded,
            reduce_range=v["reduce_range"],
            extra_options={"CalibPercentile": 99.99},
        )
        print(f"{name}: {time.time() - t0:.0f} s", flush=True)
    (out / "fp32.onnx").write_bytes((work / "cnn.fp32.onnx").read_bytes())
    vol = selftest_volume()
    np.save(out / "phantom.npy", contexts(vol, cfg["context"]))
    val = load_series(Path(cfg["data_dir"]), "val")[:40]
    real = np.concatenate([contexts(read_volume(s), cfg["context"]) for s in val])
    np.save(out / "real_val.npy", real)  # local only: never copied off this machine


def run_model(path: Path, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return np.concatenate(
        [sess.run(["logits"], {"hu": x[i : i + 16]})[0] for i in range(0, len(x), 16)]
    )


def check(out: Path) -> None:
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    report = {}
    for data in ("phantom", "real_val"):
        f = out / f"{data}.npy"
        if not f.exists():
            continue
        x = np.load(f).astype(np.float32)
        ref = sig(run_model(out / "fp32.onnx", x))
        for name in VARIANTS:
            err = np.abs(sig(run_model(out / f"{name}.onnx", x)) - ref)
            report.setdefault(name, {})[data] = {
                "max": round(float(err.max()), 4),
                "mean": round(float(err.mean()), 5),
            }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    {
        "build": lambda: build(Path(sys.argv[2]), Path(sys.argv[3])),
        "check": lambda: check(Path(sys.argv[2])),
    }[sys.argv[1]]()
