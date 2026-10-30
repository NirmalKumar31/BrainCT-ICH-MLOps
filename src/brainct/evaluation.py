"""Calibrate on validation, freeze thresholds, report test with patient-bootstrap CIs.

    uv run python -m brainct.evaluation --run artifacts/runs/cnn25d_resnet18 --model seq
    uv run python -m brainct.evaluation --compare artifacts/runs/cnn2d_resnet18:cnn \
        artifacts/runs/cnn25d_resnet18:cnn artifacts/runs/cnn25d_resnet18:seq

Everything fit (Platt calibration, operating thresholds) uses validation only; test is
scored once with those frozen values. Study level = the max slice probability of a series;
its thresholds are fit at study level, because that is the score the API compares them with.
Macro AUC is reported over all six labels ("auc_macro") and over the five subtypes
("auc_macro_subtypes", the usual definition in RSNA work).
"""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from brainct.constants import LABELS
from brainct.metrics import (
    average_precision,
    best_f1_threshold,
    binary_report,
    bootstrap_ci,
    expected_calibration_error,
    fit_platt,
    log_loss,
    roc_auc,
    sigmoid,
    threshold_for_sensitivity,
    weighted_log_loss,
)

SCREENING_SENSITIVITY = 0.95


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files if k != "emb"}


def calibrate(logits: np.ndarray, calib: list[tuple[float, float]]) -> np.ndarray:
    a = np.array([c[0] for c in calib])
    b = np.array([c[1] for c in calib])
    return sigmoid(logits * a + b)


def series_max(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return np.array([values[a:b].max(axis=0) for a, b in pairwise(offsets)])


def slice_groups(pred: dict[str, np.ndarray]) -> np.ndarray:
    lengths = np.diff(pred["offsets"])
    return np.repeat(pred["patient_id"], lengths)


def _on(fn, *arrays):
    """Metric over a row subset, for bootstrap_ci."""
    return lambda i: fn(*(a[i] for a in arrays))


def fit_on_validation(val: dict[str, np.ndarray]) -> dict:
    calib = [fit_platt(val["logits"][:, j], val["labels"][:, j]) for j in range(len(LABELS))]
    p = calibrate(val["logits"], calib)
    y = val["labels"]
    thresholds = {k: best_f1_threshold(y[:, j], p[:, j])[0] for j, k in enumerate(LABELS)}
    study_p = series_max(p, val["offsets"])
    study_y = series_max(y, val["offsets"])
    study = {k: best_f1_threshold(study_y[:, j], study_p[:, j])[0] for j, k in enumerate(LABELS)}
    return {
        "calibration": {k: list(c) for k, c in zip(LABELS, calib, strict=True)},
        "thresholds_f1": thresholds,
        "threshold_screening_any": threshold_for_sensitivity(
            y[:, 0], p[:, 0], SCREENING_SENSITIVITY
        ),
        "threshold_study_any": study["any"],
        "thresholds_study": study,
        "threshold_study_screening_any": threshold_for_sensitivity(
            study_y[:, 0], study_p[:, 0], SCREENING_SENSITIVITY
        ),
    }


def score(test: dict[str, np.ndarray], fitted: dict, n_boot: int = 1000) -> dict:
    calib = [tuple(fitted["calibration"][k]) for k in LABELS]
    p = calibrate(test["logits"], calib)
    y = test["labels"].astype(np.float64)
    groups = slice_groups(test)
    out: dict = {
        "slices": len(y),
        "series": len(test["series_id"]),
        "patients": len(np.unique(test["patient_id"])),
        "labels": {},
    }

    for j, k in enumerate(LABELS):
        yj, pj = y[:, j], p[:, j]
        thr = fitted["thresholds_f1"][k]
        out["labels"][k] = {
            "positives": int(yj.sum()),
            "auc": roc_auc(yj, pj),
            "auc_ci": bootstrap_ci(_on(roc_auc, yj, pj), groups, n_boot),
            "ap": average_precision(yj, pj),
            "ap_ci": bootstrap_ci(_on(average_precision, yj, pj), groups, n_boot),
            "log_loss": log_loss(yj, pj),
            "ece": expected_calibration_error(yj, pj),
            "threshold": thr,
            **binary_report(yj, pj, thr),
        }

    def macro(cols: list[int]):
        return lambda i: float(np.mean([roc_auc(y[i, j], p[i, j]) for j in cols]))

    for name, cols in (
        ("auc_macro", list(range(len(LABELS)))),
        ("auc_macro_subtypes", [1, 2, 3, 4, 5]),
    ):
        out[name] = float(np.mean([out["labels"][LABELS[j]]["auc"] for j in cols]))
        out[name + "_ci"] = bootstrap_ci(macro(cols), groups, n_boot)
    out["weighted_log_loss"] = weighted_log_loss(y, p)
    out["weighted_log_loss_ci"] = bootstrap_ci(
        lambda i: weighted_log_loss(y[i], p[i]), groups, n_boot
    )
    out["screening_any"] = {
        "target_sensitivity": SCREENING_SENSITIVITY,
        "threshold": fitted["threshold_screening_any"],
        **binary_report(y[:, 0], p[:, 0], fitted["threshold_screening_any"]),
    }
    sp_all = series_max(p, test["offsets"])
    sy_all = series_max(y, test["offsets"])
    series_patients = test["patient_id"]
    sp, sy = sp_all[:, 0], sy_all[:, 0]
    out["study_any"] = {
        "positives": int(sy.sum()),
        "auc": roc_auc(sy, sp),
        "auc_ci": bootstrap_ci(lambda i: roc_auc(sy[i], sp[i]), series_patients, n_boot),
        "threshold": fitted["threshold_study_any"],
        **binary_report(sy, sp, fitted["threshold_study_any"]),
    }
    # The operating points the API reports per study (older fits lack them: skipped).
    if "thresholds_study" in fitted:
        out["study_labels"] = {
            k: {
                "positives": int(sy_all[:, j].sum()),
                "auc": roc_auc(sy_all[:, j], sp_all[:, j]),
                "threshold": fitted["thresholds_study"][k],
                **binary_report(sy_all[:, j], sp_all[:, j], fitted["thresholds_study"][k]),
            }
            for j, k in enumerate(LABELS)
        }
        thr = fitted["threshold_study_screening_any"]
        out["study_screening_any"] = {
            "target_sensitivity": SCREENING_SENSITIVITY,
            "threshold": thr,
            **binary_report(sy, sp, thr),
        }
    return out


def evaluate_run(run: Path, model: str, n_boot: int = 1000) -> dict:
    pred_dir = run / "predictions"
    val, test = load(pred_dir / f"{model}_val.npz"), load(pred_dir / f"{model}_test.npz")
    fitted = fit_on_validation(val)
    report = {
        "run": run.name,
        "model": model,
        "fitted_on_val": fitted,
        "test": score(test, fitted, n_boot),
    }
    (run / f"report_{model}.json").write_text(json.dumps(report, indent=2))
    return report


def _fmt(v: float, ci: list[float] | tuple[float, float] | None = None, d: int = 3) -> str:
    s = f"{v:.{d}f}"
    return f"{s} ({ci[0]:.{d}f}–{ci[1]:.{d}f})" if ci else s


def markdown_table(reports: list[dict]) -> str:
    head = ["Model", "Macro AUC (95% CI)", "Weighted log loss", "Study AUC (any)"]
    head += [f"AUC {k}" for k in LABELS[1:]]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in reports:
        t = r["test"]
        row = [
            f"{r['run']} / {r['model']}",
            _fmt(t["auc_macro"], t["auc_macro_ci"]),
            _fmt(t["weighted_log_loss"], t["weighted_log_loss_ci"]),
            _fmt(t["study_any"]["auc"], t["study_any"]["auc_ci"]),
        ] + [_fmt(t["labels"][k]["auc"]) for k in LABELS[1:]]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", type=Path)
    ap.add_argument("--model", default="seq", choices=["cnn", "seq"])
    ap.add_argument("--compare", nargs="*", metavar="RUN:MODEL")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args(argv)
    if args.compare:
        reports = []
        for item in args.compare:
            run, _, model = item.partition(":")
            reports.append(evaluate_run(Path(run), model, args.boot))
        print(markdown_table(reports))
    else:
        r = evaluate_run(args.run, args.model, args.boot)
        print(markdown_table([r]))


if __name__ == "__main__":
    main()
