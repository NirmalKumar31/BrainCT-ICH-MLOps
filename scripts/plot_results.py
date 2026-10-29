"""Figures for docs/results.md from the certified (deployed-chain) predictions.

uv run python scripts/plot_results.py artifacts/bundle.work docs/figures
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from brainct.constants import LABELS
from brainct.evaluation import calibrate, fit_on_validation, load
from brainct.metrics import roc_auc

COLORS = ["#1f2937", "#d97706", "#2563eb", "#059669", "#7c3aed", "#dc2626"]


def roc_points(y: np.ndarray, p: np.ndarray):
    order = np.argsort(-p)
    y = y[order].astype(bool)
    tpr = np.r_[0, np.cumsum(y) / max(1, y.sum())]
    fpr = np.r_[0, np.cumsum(~y) / max(1, (~y).sum())]
    return fpr, tpr


def probs(work: Path, precision: str, head: str):
    val = load(work / f"chain_{precision}_{head}_val.npz")
    test = load(work / f"chain_{precision}_{head}_test.npz")
    fitted = fit_on_validation(val)
    cal = [tuple(fitted["calibration"][k]) for k in LABELS]
    return calibrate(test["logits"], cal), test["labels"]


def main(work: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    p, y = probs(work, "int8", "seq")

    fig, ax = plt.subplots(figsize=(5.2, 5))
    for j, k in enumerate(LABELS):
        fpr, tpr = roc_points(y[:, j], p[:, j])
        ax.plot(fpr, tpr, color=COLORS[j], lw=1.6, label=f"{k} ({roc_auc(y[:, j], p[:, j]):.3f})")
    ax.plot([0, 1], [0, 1], color="#9ca3af", lw=0.8, ls="--")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="Test ROC, deployed INT8 chain (slice level)")  # fmt: skip
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "roc_test.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5.2, 5))
    edges = np.linspace(0, 1, 11)
    for j in (0, 5, 4):
        idx = np.clip(np.digitize(p[:, j], edges[1:-1]), 0, 9)
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.sum() >= 30:
                xs.append(p[m, j].mean())
                ys.append(y[m, j].mean())
        ax.plot(xs, ys, "o-", color=COLORS[j], lw=1.4, ms=4, label=LABELS[j])
    ax.plot([0, 1], [0, 1], color="#9ca3af", lw=0.8, ls="--")
    ax.set(xlabel="Predicted probability", ylabel="Observed frequency",
           title="Test calibration (Platt fit on validation)")  # fmt: skip
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "calibration_test.png", dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
