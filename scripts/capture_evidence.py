"""Write proof artifacts into docs/evidence from a running API and the local MLflow store.

    uv run python scripts/capture_evidence.py http://127.0.0.1:8011

Inputs are synthetic phantoms only (no patient images leave the machine).
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/evidence"
sys.path.insert(0, str(ROOT / "tests"))
from dicom_factory import make_series  # noqa: E402


def api_examples(url: str) -> None:
    blobs = make_series(24, size=512, bleed_at=(10, 11, 12))
    files = [("files", (f"{i}.dcm", b, "application/dicom")) for i, b in enumerate(blobs)]
    r = requests.post(f"{url}/v1/predict/series?explain=true", files=files, timeout=60)
    r.raise_for_status()
    body = r.json()
    (OUT / "heatmap_synthetic_phantom.png").write_bytes(
        base64.b64decode(body["heatmap"]["png_base64"])
    )
    body["heatmap"]["png_base64"] = "<base64 PNG: see heatmap_synthetic_phantom.png>"
    body["slices"] = body["slices"][9:14]  # keep the example short
    body["_note"] = "synthetic phantom series (24 slices, bright lesion on slices 10-12); excerpt"
    (OUT / "example_series_response.json").write_text(json.dumps(body, indent=2))
    for path in ("/v1/model", "/v1/drift"):
        name = path.strip("/").replace("/", "_")
        (OUT / f"{name}.json").write_text(json.dumps(requests.get(url + path).json(), indent=2))
    errors = {}
    for name, payload in {"not_dicom": b"junk", "empty": b""}.items():
        resp = requests.post(f"{url}/v1/predict/slice", files={"file": ("x.dcm", payload)})
        errors[name] = {"status": resp.status_code, "body": resp.json()}
    (OUT / "example_errors.json").write_text(json.dumps(errors, indent=2))


def training_curves() -> None:
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{ROOT}/mlruns/mlflow.db")
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name("brainct")
    runs = client.search_runs([exp.experiment_id], order_by=["start_time ASC"])
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for run in runs:
        name = run.info.run_name
        rows.append(
            {
                "run": name,
                "status": run.info.status,
                "id": run.info.run_id[:8],
                **{
                    k: round(v, 4)
                    for k, v in run.data.metrics.items()
                    if k in ("val_auc_mean", "best_epoch", "seq_best_val_wlogloss")
                },
            }
        )
        if name.endswith("_seq"):
            hist = client.get_metric_history(run.info.run_id, "seq_val_wlogloss")
            axes[1].plot([m.step for m in hist], [m.value for m in hist], label=name)
        else:
            hist = client.get_metric_history(run.info.run_id, "val_auc_mean")
            axes[0].plot([m.step for m in hist], [m.value for m in hist], "o-", label=name)
    axes[0].set(title="CNN: validation mean AUC", xlabel="epoch")
    axes[1].set(title="Series model: validation weighted log loss", xlabel="epoch")
    for ax in axes:
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "mlflow_training_curves.png", dpi=140)
    (OUT / "mlflow_runs.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    api_examples(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000")
    training_curves()
    print("wrote", sorted(p.name for p in OUT.iterdir()))
