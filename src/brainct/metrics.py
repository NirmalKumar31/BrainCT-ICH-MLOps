"""Classification metrics in plain numpy (no sklearn), so tests and the service can use them."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from brainct.constants import LABEL_WEIGHTS

EPS = 1e-7


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Area under the ROC curve via the rank statistic, ties counted half. NaN if one class."""
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    sorted_p = p[order]
    # average ranks over ties
    _, first, counts = np.unique(sorted_p, return_index=True, return_counts=True)
    avg = first + (counts + 1) / 2.0
    ranks[order] = np.repeat(avg, counts)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """Step-wise AP (sum of precision at each positive), as sklearn computes it."""
    y = np.asarray(y).astype(bool)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-np.asarray(p, dtype=np.float64), kind="mergesort")
    y_sorted = y[order]
    p_sorted = np.asarray(p)[order]
    tp = np.cumsum(y_sorted)
    # evaluate only at the last index of each distinct score
    last = np.r_[np.nonzero(np.diff(p_sorted))[0], len(p_sorted) - 1]
    precision = tp[last] / (last + 1)
    recall = tp[last] / y.sum()
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def weighted_log_loss(y: np.ndarray, p: np.ndarray, weights: Sequence[float] = LABEL_WEIGHTS):
    """RSNA 2019 metric: per-label log loss averaged with weights ("any" counts 2x)."""
    w = np.asarray(weights, dtype=np.float64)
    losses = np.array([log_loss(y[:, j], p[:, j]) for j in range(y.shape[1])])
    return float(np.sum(w * losses) / w.sum())


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """(threshold, F1) maximising F1. Used on validation only; frozen for test."""
    y = np.asarray(y).astype(bool)
    if y.sum() == 0:
        return 0.5, float("nan")
    order = np.argsort(-p, kind="mergesort")
    ps, ys = np.asarray(p)[order], y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(~ys)
    f1 = 2 * tp / (tp + fp + y.sum())
    i = int(np.argmax(f1))
    return float(ps[i]), float(f1[i])


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, sensitivity: float) -> float:
    """Highest threshold whose sensitivity is at least `sensitivity`."""
    pos = np.sort(np.asarray(p)[np.asarray(y).astype(bool)])
    if len(pos) == 0:
        return 0.5
    k = int(np.floor((1 - sensitivity) * len(pos)))
    return float(pos[min(k, len(pos) - 1)])


def binary_report(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y).astype(bool)
    pred = np.asarray(p) >= threshold
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    return {"sensitivity": sens, "specificity": spec, "ppv": ppv, "f1": f1}


def bootstrap_ci(
    metric: Callable[[np.ndarray], float],
    groups: np.ndarray,
    n: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI resampling whole groups (patients), not rows.

    `metric` receives an index array into the original rows. Slices of one patient are
    correlated, so row-level resampling would give CIs that are too narrow.
    """
    rng = np.random.default_rng(seed)
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [np.nonzero(inverse == g)[0] for g in range(len(uniq))]
    stats = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        v = metric(idx)
        if not np.isnan(v):
            stats.append(v)
    if not stats:
        return float("nan"), float("nan")
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _platt_loss(z: np.ndarray, y: np.ndarray, a: float, b: float, l2: float) -> float:
    t = a * z + b
    # log(1 + e^t) - y*t, overflow-safe
    return float(np.mean(np.logaddexp(0.0, t) - y * t) + 0.5 * l2 * (a - 1) ** 2)


def fit_platt(
    z: np.ndarray, y: np.ndarray, iters: int = 100, l2: float = 1e-3, tol: float = 1e-9
) -> tuple[float, float]:
    """Platt scaling p = sigmoid(a*z + b), fit by Newton's method on log loss.

    Needed because balanced sampling trains the CNN on ~44% positives while only ~14% of slices
    are positive: ranking (AUC) is unaffected, raw probabilities are not.

    Plain Newton from (1, 0) overshoots when the logits are badly over-confident (a << 1) and
    can run away to a ~ 600, b ~ -1e10; every step here is halved until the loss decreases,
    and a fit that does not converge raises instead of returning garbage.
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    a, b = 1.0, 0.0
    loss = _platt_loss(z, y, a, b, l2)
    for _ in range(iters):
        p = sigmoid(a * z + b)
        r = p - y
        w = np.maximum(p * (1 - p), 1e-12)
        g = np.array([np.mean(r * z) + l2 * (a - 1), np.mean(r)])
        h = np.array(
            [[np.mean(w * z * z) + l2, np.mean(w * z)], [np.mean(w * z), np.mean(w) + 1e-12]]
        )
        step = np.linalg.solve(h, g)
        t = 1.0
        while t > 1e-10:
            na, nb = a - t * step[0], b - t * step[1]
            new = _platt_loss(z, y, na, nb, l2)
            if np.isfinite(new) and new <= loss + 1e-12:
                break
            t /= 2
        else:
            break  # no descent possible: at the optimum to numerical precision
        a, b, prev, loss = na, nb, loss, new
        if prev - loss < tol and np.abs(t * step).max() < 1e-6:
            break
    gnorm = np.abs(np.array([np.mean((sigmoid(a * z + b) - y) * z) + l2 * (a - 1),
                             np.mean(sigmoid(a * z + b) - y)])).max()  # fmt: skip
    if not (np.isfinite(a) and np.isfinite(b)) or gnorm > 1e-4:
        raise ValueError(f"Platt fit did not converge (a={a:.3g}, b={b:.3g}, |grad|={gnorm:.2g})")
    return float(a), float(b)
