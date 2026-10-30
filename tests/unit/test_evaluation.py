from __future__ import annotations

import json

import numpy as np
import pytest

from brainct.constants import LABELS
from brainct.evaluation import (
    calibrate,
    evaluate_run,
    fit_on_validation,
    main,
    markdown_table,
    score,
    series_max,
)
from brainct.metrics import sigmoid


def fake_predictions(seed: int, n_series: int = 40, shift: float = 1.5) -> dict[str, np.ndarray]:
    """Series of 20-40 slices; positives get logits shifted up, like an imperfect model."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(20, 40, n_series)
    labels, logits = [], []
    for t in lengths:
        y = np.zeros((t, len(LABELS)), np.uint8)
        if rng.random() < 0.4:
            a = rng.integers(0, t - 5)
            sub = rng.integers(1, len(LABELS))
            y[a : a + 5, sub] = 1
            y[a : a + 5, 0] = 1
        labels.append(y)
        logits.append(rng.normal(-2, 1, y.shape) + shift * y + 1.0)  # +1: prior shift
    return {
        "series_id": np.array([f"S{seed}_{i}" for i in range(n_series)]),
        "patient_id": np.array([f"P{seed}_{i // 2}" for i in range(n_series)]),
        "offsets": np.concatenate([[0], np.cumsum(lengths)]),
        "labels": np.concatenate(labels),
        "logits": np.concatenate(logits).astype(np.float32),
    }


def test_series_max():
    v = np.array([[1, 0], [3, 1], [2, 5], [0, 0]])
    np.testing.assert_array_equal(series_max(v, np.array([0, 2, 4])), [[3, 1], [2, 5]])


def test_calibration_fixes_prior_shift_and_improves_log_loss():
    val, test = fake_predictions(0), fake_predictions(1)
    fitted = fit_on_validation(val)
    assert set(fitted["calibration"]) == set(LABELS)
    assert fitted["calibration"]["any"][1] < 0  # learned to undo the +1 logit shift
    raw = score(test, {**fitted, "calibration": {k: [1.0, 0.0] for k in LABELS}}, n_boot=20)
    cal = score(test, fitted, n_boot=20)
    assert cal["weighted_log_loss"] < raw["weighted_log_loss"]
    assert cal["auc_macro"] == pytest.approx(raw["auc_macro"])  # monotone: ranking unchanged


def test_score_report_shape():
    val, test = fake_predictions(2), fake_predictions(3)
    r = score(test, fit_on_validation(val), n_boot=30)
    assert r["series"] == 40 and r["patients"] == 20
    lo, hi = r["auc_macro_ci"]
    assert lo <= r["auc_macro"] <= hi
    assert 0.5 < r["study_any"]["auc"] <= 1.0
    assert r["screening_any"]["target_sensitivity"] == 0.95
    for k in LABELS:
        assert {"auc", "ap", "ece", "threshold", "sensitivity"} <= set(r["labels"][k])


def test_calibrate_applies_per_label_params():
    z = np.zeros((2, len(LABELS)))
    params = [(1.0, 0.0)] * len(LABELS)
    params[0] = (1.0, 2.0)
    p = calibrate(z, params)
    assert p[0, 0] == pytest.approx(sigmoid(2.0))
    assert p[0, 1] == pytest.approx(0.5)


def test_evaluate_run_and_cli(tmp_path, capsys):
    for run in ("a", "b"):
        d = tmp_path / run / "predictions"
        d.mkdir(parents=True)
        for split, seed in (("val", 4), ("test", 5)):
            np.savez(d / f"cnn_{split}.npz", **fake_predictions(seed))
    r = evaluate_run(tmp_path / "a", "cnn", n_boot=10)
    saved = json.loads((tmp_path / "a" / "report_cnn.json").read_text())
    assert saved["test"]["auc_macro"] == pytest.approx(r["test"]["auc_macro"])
    table = markdown_table([r])
    assert table.count("\n") == 2 and "a / cnn" in table
    main(["--compare", f"{tmp_path / 'a'}:cnn", f"{tmp_path / 'b'}:cnn", "--boot", "10"])
    assert "b / cnn" in capsys.readouterr().out
    main(["--run", str(tmp_path / "b"), "--model", "cnn", "--boot", "10"])


def test_study_level_operating_points_are_fit_and_scored():
    val, test = fake_predictions(1), fake_predictions(2)
    fitted = fit_on_validation(val)
    assert set(fitted["thresholds_study"]) == set(LABELS)
    assert fitted["threshold_study_any"] == fitted["thresholds_study"]["any"]
    r = score(test, fitted, n_boot=20)
    assert set(r["study_labels"]) == set(LABELS)
    assert r["study_screening_any"]["threshold"] == fitted["threshold_study_screening_any"]
    subtypes = np.mean([r["labels"][k]["auc"] for k in LABELS[1:]])
    assert r["auc_macro_subtypes"] == pytest.approx(subtypes)
