"""Input and prediction drift: cheap per-slice features, a training reference, and PSI.

The export step computes histograms of these features over training slices and stores them in
the model card. The service keeps a sliding window of the same features from live traffic and
exports the Population Stability Index per feature to Prometheus. PSI > 0.2 is the common
"significant shift" rule of thumb; the alert in monitoring/ fires on it.

Features catch the failures that matter for CT in practice: a different scanner/kernel or a
wrong rescale (tissue_mean_hu), non-head or cropped images (air/brain/bone fractions), and a
change in case mix or model behaviour (p_any).
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Mapping

import numpy as np

FEATURES: dict[str, tuple[float, float]] = {
    # name: (low, high) range of the reference histogram; values outside go to the end bins
    "tissue_mean_hu": (-20.0, 80.0),
    "air_fraction": (0.0, 1.0),
    "brain_fraction": (0.0, 0.6),
    "bone_fraction": (0.0, 0.4),
    "p_any": (0.0, 1.0),
}
BINS = 10
PSI_EPS = 1e-4


def slice_features(hu: np.ndarray) -> dict[str, float]:
    """Features of one resized HU slice (int16 or float, [H, W])."""
    hu = np.asarray(hu, dtype=np.float32)
    tissue = hu[(hu > -100) & (hu < 200)]
    return {
        "tissue_mean_hu": float(tissue.mean()) if tissue.size else 0.0,
        "air_fraction": float(np.mean(hu < -900)),
        "brain_fraction": float(np.mean((hu >= 0) & (hu <= 80))),
        "bone_fraction": float(np.mean(hu > 300)),
    }


def histogram(values: np.ndarray, low: float, high: float, bins: int = BINS) -> list[float]:
    edges = np.linspace(low, high, bins + 1)
    idx = np.clip(np.digitize(np.asarray(values, dtype=np.float64), edges[1:-1]), 0, bins - 1)
    counts = np.bincount(idx, minlength=bins).astype(np.float64)
    return (counts / max(1.0, counts.sum())).tolist()


def build_reference(samples: Mapping[str, np.ndarray]) -> dict[str, dict]:
    """{feature: {"low", "high", "proportions"}} from arrays of training feature values."""
    ref = {}
    for name, (low, high) in FEATURES.items():
        if name in samples:
            ref[name] = {
                "low": low,
                "high": high,
                "proportions": histogram(samples[name], low, high),
                "mean": float(np.mean(samples[name])),
            }
    return ref


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, dtype=np.float64), PSI_EPS, None)
    a = np.clip(np.asarray(actual, dtype=np.float64), PSI_EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


class DriftMonitor:
    """Thread-safe sliding window of live feature values, scored against the reference."""

    def __init__(self, reference: Mapping[str, Mapping], window: int = 1000, min_count: int = 100):
        self.reference = dict(reference)
        self.window = window
        self.min_count = min_count
        self._values: dict[str, deque[float]] = {k: deque(maxlen=window) for k in self.reference}
        self._lock = threading.Lock()

    def observe(self, features: Mapping[str, float]) -> None:
        with self._lock:
            for k, v in features.items():
                if k in self._values:
                    self._values[k].append(float(v))

    def snapshot(self) -> dict[str, dict]:
        """Per feature: count, live mean, reference mean, PSI (None until min_count values)."""
        with self._lock:
            values = {k: np.asarray(v) for k, v in self._values.items()}
        out = {}
        for k, ref in self.reference.items():
            v = values[k]
            score = None
            if len(v) >= self.min_count:
                live = histogram(v, ref["low"], ref["high"], len(ref["proportions"]))
                score = psi(np.asarray(ref["proportions"]), np.asarray(live))
            out[k] = {
                "count": len(v),
                "live_mean": float(v.mean()) if len(v) else None,
                "reference_mean": ref.get("mean"),
                "psi": score,
            }
        return out
