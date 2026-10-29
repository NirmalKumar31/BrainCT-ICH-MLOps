"""Service settings from BRAINCT_* environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class Settings:
    bundle: str = "artifacts/bundle"
    threads: int = 0  # ONNX Runtime intra-op threads; 0 = runtime default (all cores)
    max_concurrency: int = 2  # model calls in flight; more just oversubscribes the CPU
    max_file_mb: float = 4.0  # one DICOM slice
    max_total_mb: float = 200.0  # one series upload (files or zip, uncompressed)
    max_slices: int = 256
    drift_window: int = 2000
    drift_min_count: int = 200
    warmup: bool = True
    verify_bundle: bool = True
    precision: str = "auto"  # auto | int8 | fp32 (auto: INT8 unless the self-test fails)
    batching: bool = True  # cross-request micro-batching of CNN calls
    max_batch: int = 32
    max_wait_ms: float = 4.0
    feedback_window: int = 2000  # study-level (label, probability) pairs for live metrics
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, prefix: str = "BRAINCT_") -> Settings:
        values: dict[str, Any] = {}
        for f in fields(cls):
            raw = os.environ.get(prefix + f.name.upper())
            if raw is None:
                continue
            if f.type in ("bool", bool):
                values[f.name] = raw.strip().lower() in ("1", "true", "yes", "on")
            elif f.type in ("int", int):
                values[f.name] = int(raw)
            elif f.type in ("float", float):
                values[f.name] = float(raw)
            else:
                values[f.name] = raw
        return cls(**values)
