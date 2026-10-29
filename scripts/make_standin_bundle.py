"""Build a tiny stand-in model bundle with the real bundle's interface (CI, Docker, Kubernetes).

    python scripts/make_standin_bundle.py artifacts/standin

Built with onnx.helper (no PyTorch, no trained weights): the real model is derived from RSNA
data and is not distributed. Outputs are deterministic functions of the input HU, so API tests
can assert on structure and ordering, not on clinical meaning.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

LABELS = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
IMAGE_SIZE, CONTEXT, DIM = 224, 1, 8


def _init(name: str, arr: np.ndarray):
    return numpy_helper.from_array(arr.astype(np.float32), name)


def cnn_model() -> onnx.ModelProto:
    k = 2 * CONTEXT + 1
    rng = np.random.default_rng(0)
    nodes = [
        helper.make_node("Mul", ["hu", "scale"], ["x"]),
        helper.make_node("AveragePool", ["x"], ["pooled"], kernel_shape=[32, 32], strides=[32, 32]),
        helper.make_node("Conv", ["pooled", "w_conv"], ["fmap"]),
        helper.make_node("GlobalAveragePool", ["fmap"], ["gap"]),
        helper.make_node("Flatten", ["gap"], ["embedding"]),
        helper.make_node("Gemm", ["embedding", "w_fc", "b_fc"], ["logits"], transB=1),
    ]
    inits = [
        _init("scale", np.array(0.01)),
        _init("w_conv", rng.normal(0, 0.5, (DIM, k, 1, 1))),
        _init("w_fc", rng.normal(0, 0.5, (len(LABELS), DIM))),
        _init("b_fc", np.full(len(LABELS), -1.0)),
    ]
    graph = helper.make_graph(
        nodes,
        "standin_cnn",
        [helper.make_tensor_value_info("hu", TensorProto.FLOAT, ["n", k, IMAGE_SIZE, IMAGE_SIZE])],
        [
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["n", len(LABELS)]),
            helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["n", DIM]),
            helper.make_tensor_value_info("fmap", TensorProto.FLOAT, ["n", DIM, 7, 7]),
        ],
        inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)


def seq_model() -> onnx.ModelProto:
    w = np.random.default_rng(1).normal(0, 0.1, (DIM, len(LABELS)))
    nodes = [
        helper.make_node("MatMul", ["embedding", "w"], ["delta"]),
        helper.make_node("Add", ["logits", "delta"], ["refined"]),
    ]
    graph = helper.make_graph(
        nodes,
        "standin_seq",
        [
            helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, "t", DIM]),
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "t", len(LABELS)]),
        ],
        [helper.make_tensor_value_info("refined", TensorProto.FLOAT, [1, "t", len(LABELS)])],
        [_init("w", w)],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)


def build(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    onnx.save(cnn_model(), out / "cnn.onnx")
    onnx.save(seq_model(), out / "seq.onnx")
    np.save(out / "cam_weights.npy", np.random.default_rng(2).normal(0, 1, (len(LABELS), DIM)))
    identity = {k: [1.0, 0.0] for k in LABELS}
    thresholds = {k: 0.5 for k in LABELS} | {"screening_any": 0.3, "study_any": 0.5}
    uniform = [0.1] * 10
    card = {
        "name": "brainct-standin",
        "version": "0.0.0-standin",
        "contract": {"labels": LABELS, "context": CONTEXT, "image_size": IMAGE_SIZE, "batch": 16},
        "calibration": {"cnn": identity, "seq": identity},
        "thresholds": {"cnn": thresholds, "seq": thresholds},
        "drift_reference": {
            "tissue_mean_hu": {"low": -20.0, "high": 80.0, "proportions": uniform, "mean": 30.0},
            "air_fraction": {"low": 0.0, "high": 1.0, "proportions": uniform, "mean": 0.5},
            "p_any": {"low": 0.0, "high": 1.0, "proportions": uniform, "mean": 0.5},
        },
        "intended_use": "Stand-in for tests. Not a trained model.",
        "files": {},
    }
    for name, path in (
        ("cnn", "cnn.onnx"),
        ("seq", "seq.onnx"),
        ("cam_weights", "cam_weights.npy"),
    ):
        card["files"][name] = {
            "path": path,
            "sha256": hashlib.sha256((out / path).read_bytes()).hexdigest(),
        }
    (out / "model_card.json").write_text(json.dumps(card, indent=2))
    return out


if __name__ == "__main__":
    print(build(Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/standin")))
