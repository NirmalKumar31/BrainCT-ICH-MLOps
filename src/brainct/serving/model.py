"""Torch-free inference over an exported model bundle (ONNX Runtime + numpy).

A bundle directory holds:
    model_card.json   contract (labels, context, image size), calibration, thresholds,
                      test metrics, drift reference, file checksums
    model_card.sha256 checksum of the card itself (it holds calibration and thresholds)
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
    def __init__(
        self, bundle: str | Path, threads: int = 0, verify: bool = True, precision: str = "auto"
    ):
        """precision: "auto" (INT8 unless the startup self-test fails here, then FP32),
        "int8" or "fp32"."""
        self.dir = Path(bundle)
        card_path = self.dir / "model_card.json"
        if not card_path.exists():
            raise BundleError("no model_card.json in the bundle directory")
        self.card = json.loads(card_path.read_text())
        files = self.card["files"]
        if verify:
            card_sum = self.dir / "model_card.sha256"
            if card_sum.exists() and card_sum.read_text().split()[0] != sha256(card_path):
                raise BundleError("model_card.json: checksum mismatch")
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
        self._opts, self._providers = opts, providers
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
        self.precision = "int8" if "cnn_fp32" in files else "bundle"
        self._fits = {k: self.card[k] for k in ("calibration", "thresholds")}
        self.selftest: dict = {"status": "absent"}
        if precision not in ("auto", "int8", "fp32"):
            raise BundleError(f"unknown precision {precision!r}")
        if precision == "fp32":
            if "cnn_fp32" not in files:
                raise BundleError("precision=fp32 requested but the bundle has no FP32 model")
            self._use_fp32()
        elif precision == "auto" and "selftest" in files:
            self.selftest = self.self_test()
            if not self.selftest["passed"]:
                if "cnn_fp32" not in files:
                    # Serving numbers that were never certified is worse than not serving.
                    raise BundleError(
                        f"INT8 self-test failed on this CPU (max |dp| "
                        f"{self.selftest['max_prob_diff']:.3f}) and there is no FP32 fallback"
                    )
                self._use_fp32()

    def _use_fp32(self) -> None:
        path = self.dir / self.card["files"]["cnn_fp32"]["path"]
        self.cnn = ort.InferenceSession(str(path), self._opts, providers=self._providers)
        self.precision = "fp32"
        if "fp32" in self.card:  # fits of the FP32 chain (older bundles: the INT8 ones)
            self._fits = self.card["fp32"]

    def self_test(self) -> dict:
        """Reproduce the certified INT8 logits on a synthetic phantom, on this CPU."""
        with np.load(self.dir / self.card["files"]["selftest"]["path"]) as z:
            vol, golden = z["volume"], z["certified_logits"]
        x = contexts(vol, self.context)
        parts = [
            self.cnn.run(["logits"], {"hu": x[i : i + self.batch]})[0]
            for i in range(0, len(x), self.batch)
        ]
        diff = float(np.abs(sigmoid(np.concatenate(parts)) - sigmoid(golden)).max())
        limit = float(self.card.get("selftest", {}).get("max_prob_diff", 0.02))
        return {"status": "run", "max_prob_diff": diff, "limit": limit, "passed": diff <= limit}

    # -------------------------------------------------------------- identity
    @property
    def name(self) -> str:
        return self.card["name"]

    @property
    def version(self) -> str:
        return self.card["version"]

    @property
    def has_seq(self) -> bool:
        return self.seq is not None

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
        seq_logits = self.seq_forward(np.concatenate(embs), cnn_logits) if with_seq else None
        return RawOutput(cnn_logits, seq_logits, np.concatenate(fmaps) if with_fmap else None)

    def cnn_forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(logits, embedding) for prepared CNN input [N, 2C+1, H, W] (the micro-batcher's call)."""
        out = self.cnn.run(["logits", "embedding"], {"hu": x.astype(np.float32, copy=False)})
        return out[0], out[1]

    def seq_forward(self, emb: np.ndarray, cnn_logits: np.ndarray) -> np.ndarray | None:
        if self.seq is None:
            return None
        feeds = {"embedding": emb[None].astype(np.float32), "logits": cnn_logits[None]}
        return self.seq.run(["refined"], feeds)[0][0]

    # -------------------------------------------------------------- calibrated outputs
    def calibrate(self, logits: np.ndarray, head: str) -> np.ndarray:
        """head = "seq" (series endpoint) or "cnn" (lone slice). Platt params from validation."""
        cal = self._fits["calibration"][head]
        a = np.array([cal[k][0] for k in self.labels])
        b = np.array([cal[k][1] for k in self.labels])
        return sigmoid(logits * a + b).astype(np.float32)

    def thresholds(self, head: str) -> dict[str, float]:
        return self._fits["thresholds"][head]

    def study_thresholds(self, head: str) -> dict[str, float]:
        """Per-label thresholds for the series max, fit at study level. Bundles certified
        before those existed fall back to the per-slice ones (and "study_any")."""
        thr = self.thresholds(head)
        out = {k: thr.get(f"study_{k}", thr[k]) for k in self.labels}
        out["any"] = thr["study_any"]
        out["screening_any"] = thr.get("study_screening_any", thr["screening_any"])
        return out

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
