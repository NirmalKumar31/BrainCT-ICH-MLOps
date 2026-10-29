from __future__ import annotations

import numpy as np
import pytest

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


def brute_auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    return np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])


@pytest.mark.parametrize("seed", range(5))
def test_auc_matches_pairwise_definition(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 200)
    p = np.round(rng.random(200), 1)  # many ties
    assert roc_auc(y, p) == pytest.approx(brute_auc(y, p))


def test_auc_edge_cases():
    assert roc_auc(np.array([1, 1, 0, 0]), np.array([0.9, 0.8, 0.2, 0.1])) == 1.0
    assert roc_auc(np.array([1, 1, 0, 0]), np.array([0.1, 0.2, 0.8, 0.9])) == 0.0
    assert np.isnan(roc_auc(np.ones(5), np.linspace(0, 1, 5)))


def test_average_precision_known_values():
    # ranks: P N P N -> precision at positives 1/1 and 2/3
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.1])
    assert average_precision(y, p) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision(np.array([1, 1, 0]), np.array([0.9, 0.8, 0.1])) == pytest.approx(1.0)


def test_log_losses():
    y = np.array([[1, 0], [0, 1]], dtype=float)
    p = np.full((2, 2), 0.5)
    assert log_loss(y[:, 0], p[:, 0]) == pytest.approx(np.log(2))
    assert weighted_log_loss(y, p, weights=(2, 1)) == pytest.approx(np.log(2))


def test_platt_recovers_parameters():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, 20000)
    y = (rng.random(20000) < sigmoid(0.5 * z - 1.0)).astype(float)
    a, b = fit_platt(z, y)
    assert a == pytest.approx(0.5, abs=0.05)
    assert b == pytest.approx(-1.0, abs=0.07)


def test_calibration_error_zero_for_perfect_calibration():
    rng = np.random.default_rng(1)
    p = rng.random(50000)
    y = (rng.random(50000) < p).astype(float)
    assert expected_calibration_error(y, p) < 0.01
    assert expected_calibration_error(y, np.clip(p + 0.3, 0, 1)) > 0.2


def test_thresholds():
    y = np.array([0, 0, 1, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9])
    thr, f1 = best_f1_threshold(y, p)
    assert binary_report(y, p, thr)["f1"] == pytest.approx(f1)
    t = threshold_for_sensitivity(y, p, 0.75)
    assert binary_report(y, p, t)["sensitivity"] >= 0.75


def test_binary_report_counts():
    r = binary_report(np.array([1, 1, 0, 0]), np.array([0.9, 0.1, 0.8, 0.2]), 0.5)
    assert r == {"sensitivity": 0.5, "specificity": 0.5, "ppv": 0.5, "f1": 0.5}


def test_bootstrap_resamples_groups_not_rows():
    # Two groups with opposite outcomes: a group bootstrap sees all-one-group resamples,
    # so its CI is wide; a naive row bootstrap would be narrow.
    groups = np.array(["a"] * 50 + ["b"] * 50)
    values = np.array([1.0] * 50 + [0.0] * 50)
    lo, hi = bootstrap_ci(lambda i: values[i].mean(), groups, n=400, seed=0)
    assert lo == 0.0 and hi == 1.0
