"""Online model-performance monitoring from delayed ground truth.

Drift says the inputs changed; it cannot say the model got worse. When reads are later
confirmed (radiologist report, follow-up), the client posts the labels to /v1/feedback with
the request id it got. The service keeps each prediction for a while (bounded LRU), joins the
feedback, and maintains a sliding window of (label, probability) pairs at study level, from
which it exports live AUC, Brier score and calibration error per label.
"""

from __future__ import annotations

import threading
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence

import numpy as np

from brainct.metrics import expected_calibration_error, roc_auc


class PerformanceMonitor:
    def __init__(self, labels: Sequence[str], keep: int = 20000, window: int = 2000):
        self.labels = list(labels)
        self.keep = keep
        self._pending: OrderedDict[str, np.ndarray] = OrderedDict()
        self._pairs: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=window)
        self._lock = threading.Lock()

    def record(self, request_id: str, study_probs: np.ndarray) -> None:
        with self._lock:
            self._pending[request_id] = np.asarray(study_probs, dtype=np.float64)
            self._pending.move_to_end(request_id)
            while len(self._pending) > self.keep:
                self._pending.popitem(last=False)

    def feedback(self, request_id: str, labels: Mapping[str, int]) -> bool:
        """Join ground truth to a stored prediction. False if the id is unknown or expired."""
        unknown = set(labels) - set(self.labels)
        if unknown:
            raise ValueError(f"unknown labels {sorted(unknown)}")
        with self._lock:
            probs = self._pending.pop(request_id, None)
            if probs is None:
                return False
            y = np.full(len(self.labels), np.nan)
            for k, v in labels.items():
                y[self.labels.index(k)] = float(bool(v))
            self._pairs.append((y, probs))
            return True

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            pairs = list(self._pairs)
        out = {}
        for j, name in enumerate(self.labels):
            ys = np.array([y[j] for y, _ in pairs])
            ps = np.array([p[j] for _, p in pairs])
            known = ~np.isnan(ys) if len(ys) else np.zeros(0, bool)
            y, p = ys[known], ps[known]
            auc = roc_auc(y, p) if len(y) else float("nan")
            out[name] = {
                "count": len(y),
                "positives": int(y.sum()) if len(y) else 0,
                "auc": None if np.isnan(auc) else auc,
                "brier": float(np.mean((p - y) ** 2)) if len(y) else None,
                "ece": expected_calibration_error(y, p) if len(y) else None,
            }
        return out
