"""Full-data training curves from the Modal run logs (docs/evidence/modal/*.log).

uv run python scripts/plot_modal_curves.py docs/evidence/modal docs/evidence

The logs, not the MLflow store, are the source here: the first full-data run shared one
SQLite store between two parallel trainings and the copy-back kept only one of them (fixed in
cloud/modal_pipeline.py with one store per config).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CNN = re.compile(r"^epoch (\d+): (\{.*\})")
SEQ = re.compile(r"^epoch (\d+): train [\d.]+ val wlogloss ([\d.]+)")


def main(logs: Path, out: Path) -> None:
    rows, fig = {}, plt.figure(figsize=(10, 4))
    ax_cnn, ax_seq = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
    for name, label in (("2d", "ResNet-50, single slice"), ("25d", "ResNet-50, 2.5D")):
        cnn = [
            (int(m[1]), json.loads(m[2])["auc_mean"])
            for line in (logs / f"train_{name}.log").read_text().splitlines()
            if (m := CNN.match(line))
        ]
        seq = [
            (int(m[1]), float(m[2]))
            for line in (logs / f"finish_{name}.log").read_text().splitlines()
            if (m := SEQ.match(line))
        ]
        ax_cnn.plot(*zip(*cnn, strict=True), "o-", label=label)
        ax_seq.plot(*zip(*seq, strict=True), label=f"{label} + series model")
        rows[name] = {"cnn_val_auc_mean": dict(cnn), "seq_val_wlogloss": dict(seq)}
    ax_cnn.set(title="CNN: validation mean AUC (A100, all of RSNA)", xlabel="epoch")
    ax_seq.set(title="Series model: validation weighted log loss", xlabel="epoch")
    for ax in (ax_cnn, ax_seq):
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "modal_full_data_training_curves.png", dpi=140)
    (out / "modal_full_data_curves.json").write_text(json.dumps(rows, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
