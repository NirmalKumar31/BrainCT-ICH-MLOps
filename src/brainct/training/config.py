"""Run configuration: a YAML file, overridable with --set key=value."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


def _coerce(value, annotation, key: str):
    """YAML reads "1e-3" as a string (YAML 1.1 needs a dot: 1.0e-3); match the field's type."""
    kind = str(annotation)
    try:
        if kind == "float" and isinstance(value, str | int):
            return float(value)
        if kind == "int" and isinstance(value, str):
            return int(value)
    except ValueError as exc:
        raise ValueError(f"{key}: expected {kind}, got {value!r}") from exc
    return value


@dataclass
class TrainConfig:
    run_name: str = "cnn25d_resnet18"
    data_dir: str = "data/rsna"
    out_dir: str = "artifacts/runs"
    backbone: str = "resnet18"
    pretrained: bool = True
    context: int = 1  # 0 = single slice (reference baseline), 1 = 2.5D (i-1, i, i+1)
    epochs: int = 5
    batch_size: int = 48
    lr: float = 6e-4
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    neg_ratio: float = 1.25  # negatives per positive slice, resampled each epoch
    context_dropout: float = 0.1
    num_workers: int = 6
    seed: int = 2026
    max_train_items: int | None = None  # smoke runs only
    max_val_series: int | None = None
    # sequence model
    seq_epochs: int = 40
    seq_lr: float = 1e-3
    seq_hidden: int = 128
    seq_feature_noise: float = 0.5

    @classmethod
    def load(cls, path: str | Path | None, overrides: list[str] | None = None) -> TrainConfig:
        raw = yaml.safe_load(Path(path).read_text()) if path else {}
        types = {f.name: f.type for f in fields(cls)}
        for item in overrides or []:
            key, _, value = item.partition("=")
            if key not in types:
                raise KeyError(f"unknown config key {key!r}")
            raw[key] = _coerce(yaml.safe_load(value), types[key], key)
        unknown = set(raw) - set(types)
        if unknown:
            raise KeyError(f"unknown config keys {sorted(unknown)}")
        return cls(**raw)

    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.run_name

    def to_dict(self) -> dict:
        return asdict(self)
