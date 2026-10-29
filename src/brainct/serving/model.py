"""Torch-free inference over an exported model bundle (ONNX Runtime + numpy).

A bundle directory holds:
    model_card.json   contract (labels, context, image size), calibration, thresholds,
                      test metrics, drift reference, file checksums
    cnn.onnx          2.5D slice CNN: HU [N, 2C+1, H, W] -> logits, embedding, fmap
    seq.onnx          series model: embedding [1, T, D], logits [1, T, 6] -> refined logits
    cam_weights.npy   CNN head weights [6, D] for class activation maps

The offline evaluation of the deployed chain (optimization/export.py) runs this same class.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from brainct.metrics import sigmoid


class BundleError(RuntimeError):
    pass


@dataclass
class RawOutput:
    cnn_logits: np.ndarray  # [T, 6]
    seq_logits: np.ndarray | None  # [T, 6]
    fmap: np.ndarray | None  # [T, D, h, w] when requested


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def contexts(volume: np.ndarray, context: int) -> np.ndarray:
    """[T, H, W] -> [T, 2C+1, H, W] float32, edges repeated (same rule as training)."""
    t = volume.shape[0]
    idx = np.clip(np.arange(t)[:, None] + np.arange(-context, context + 1)[None, :], 0, t - 1)
    return volume[idx].astype(np.float32)


class InferenceModel:
    def __init__(self, bundle: str | Path, threads: int = 0, verify: bool = True):
        self.dir = Path(bundle)
        card_path = self.dir / "model_card.json"
        if not card_path.exists():
            raise BundleError(f"no model_card.json in {self.dir}")
        self.card = json.loads(card_path.read_text())
        files = self.card["files"]
        if verify:
            for name, meta in files.items():
                digest = sha256(self.dir / meta["path"])
                if digest != meta["sha256"]:
                    raise BundleError(f"{name}: checksum mismatch ({digest[:12]})")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        self.cnn = ort.InferenceSession(
            str(self.dir / files["cnn"]["path"]), opts, providers=providers
        )
        self.seq = (
            ort.InferenceSession(str(self.dir / files["seq"]["path"]), opts, providers=providers)
            if "seq" in files
            else None
        )
        cam = files.get("cam_weights")
        self.cam_weights = np.load(self.dir / cam["path"]) if cam else None
        contract = self.card["contract"]
        self.labels: list[str] = contract["labels"]
        self.context: int = contract["context"]
        self.image_size: int = contract["image_size"]
        self.batch: int = int(contract.get("batch", 16))

    # -------------------------------------------------------------- identity
    @property
    def name(self) -> str:
        return self.card["name"]

    @property
    def version(self) -> str:
        return self.card["version"]

    # -------------------------------------------------------------- raw model calls
    def run(self, volume: np.ndarray, with_seq: bool = True, with_fmap: bool = False) -> RawOutput:
        """Raw logits for a [T, H, W] HU series (already resized to image_size)."""
        if volume.ndim != 3 or volume.shape[1:] != (self.image_size, self.image_size):
            raise ValueError(
                f"expected [T, {self.image_size}, {self.image_size}], got {volume.shape}"
            )
        x = contexts(volume, self.context)
        logits, embs, fmaps = [], [], []
        outputs = ["logits", "embedding"] + (["fmap"] if with_fmap else [])
        for i in range(0, len(x), self.batch):
            out = self.cnn.run(outputs, {"hu": x[i : i + self.batch]})
            logits.append(out[0])
            embs.append(out[1])
            if with_fmap:
                fmaps.append(out[2])
        cnn_logits = np.concatenate(logits)
        seq_logits = None
        if with_seq and self.seq is not None:
            emb = np.concatenate(embs)[None].astype(np.float32)
            seq_logits = self.seq.run(["refined"], {"embedding": emb, "logits": cnn_logits[None]})[
                0
            ][0]
        return RawOutput(cnn_logits, seq_logits, np.concatenate(fmaps) if with_fmap else None)

    # -------------------------------------------------------------- calibrated outputs
    def calibrate(self, logits: np.ndarray, head: str) -> np.ndarray:
        """head = "seq" (series endpoint) or "cnn" (lone slice). Platt params from validation."""
        cal = self.card["calibration"][head]
        a = np.array([cal[k][0] for k in self.labels])
        b = np.array([cal[k][1] for k in self.labels])
        return sigmoid(logits * a + b).astype(np.float32)

    def thresholds(self, head: str) -> dict[str, float]:
        return self.card["thresholds"][head]

    def cam(self, fmap: np.ndarray, label: int) -> np.ndarray:
        """Class activation map for one slice: [D, h, w] -> [H, W] in [0, 1]."""
        if self.cam_weights is None:
            raise BundleError("bundle has no CAM weights")
        m = np.tensordot(self.cam_weights[label], fmap, axes=(0, 0))
        m = np.maximum(m, 0)
        m = cv2.resize(
            m.astype(np.float32),
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        peak = float(m.max())
        return m / peak if peak > 0 else m
