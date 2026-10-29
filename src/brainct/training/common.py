"""Device, seeding, GPU-side augmentation, MLflow and prediction I/O shared by the stages."""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from brainct.constants import HU_MIN, LABEL_WEIGHTS

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_dtype(dev: torch.device) -> torch.dtype | None:
    # fp16 measured faster than bf16 on Apple M4 (55 vs 34 img/s, ResNet-18 at 224).
    return {"cuda": torch.bfloat16, "mps": torch.float16}.get(dev.type)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds libraries that use the global state
    torch.manual_seed(seed)


def augment(hu: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
    """Random flip + affine (rotate +-12 deg, scale 0.9-1.1, shift +-6%) on [N, K, H, W] HU.

    All K slices of a sample get the same transform so the 2.5D context stays aligned.
    Out-of-image pixels are filled with air.
    """
    n = hu.shape[0]
    dev = hu.device

    def u(lo: float, hi: float) -> torch.Tensor:
        return torch.rand(n, generator=gen).to(dev) * (hi - lo) + lo

    angle = u(-12, 12) * math.pi / 180
    scale = u(0.9, 1.1)
    flip = torch.where(torch.rand(n, generator=gen).to(dev) < 0.5, -1.0, 1.0)
    tx, ty = u(-0.06, 0.06), u(-0.06, 0.06)
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack(
        [torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1
    )
    grid = F.affine_grid(theta, list(hu.shape), align_corners=False)
    shifted = hu - HU_MIN  # air -> 0, so zero padding fills with air
    out = F.grid_sample(shifted, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out + HU_MIN


def weighted_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    w = torch.tensor(LABEL_WEIGHTS, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    return (loss * w).sum(-1).mean() / w.sum()


def cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * progress))


def save_predictions(path: Path, series, logits: list[np.ndarray], emb=None) -> None:
    """One npz per split: rows are slices, `offsets` delimits series."""
    lengths = [len(x) for x in logits]
    arrays = {
        "series_id": np.asarray([s.series_id for s in series]),
        "patient_id": np.asarray([s.patient_id for s in series]),
        "offsets": np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        "logits": np.concatenate(logits).astype(np.float32),
        "labels": np.concatenate([s.labels for s in series]).astype(np.uint8),
    }
    if emb is not None:
        arrays["emb"] = np.concatenate(emb).astype(np.float16)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
