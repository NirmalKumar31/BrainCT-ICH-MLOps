"""Series store over the int16 HU cache, the 2.5D slice dataset, and the balanced sampler."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from brainct.constants import IMAGE_SIZE, LABELS


@dataclass(frozen=True)
class Series:
    series_id: str
    patient_id: str
    path: Path
    offset: int  # byte offset of the array data inside the .npy file
    labels: np.ndarray  # [T, 6] uint8

    @property
    def length(self) -> int:
        return len(self.labels)


def _data_offset(path: Path) -> int:
    with open(path, "rb") as f:
        major, _ = np.lib.format.read_magic(f)
        read_header = {
            1: np.lib.format.read_array_header_1_0,
            2: np.lib.format.read_array_header_2_0,
        }[major]
        shape, _, dtype = read_header(f)
        if dtype != np.int16 or shape[1:] != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"{path}: unexpected cache array {dtype} {shape}")
        return f.tell()


def load_series(data_dir: Path, split: str | None = None) -> list[Series]:
    """All cached series (optionally of one split), in a fixed order."""
    splits = {}
    with open(data_dir / "splits.csv", newline="") as f:
        for r in csv.DictReader(f):
            splits[r["series_id"]] = (r["patient_id"], r["split"])
    rows: dict[str, list[tuple[int, list[int]]]] = {}
    with open(data_dir / "slices.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["series_id"], []).append(
                (int(r["index"]), [int(r[k]) for k in LABELS])
            )
    out = []
    for sid in sorted(rows):
        patient, name = splits[sid]
        if split is not None and name != split:
            continue
        items = sorted(rows[sid])
        path = data_dir / "series" / f"{sid}.npy"
        labels = np.asarray([lab for _, lab in items], dtype=np.uint8)
        out.append(Series(sid, patient, path, _data_offset(path), labels))
    return out


def read_context(series: Series, index: int, context: int) -> np.ndarray:
    """Slices index-context..index+context as int16 [2C+1, H, W], edges repeated.

    Reads only the needed bytes (no mmap: one open file per series would hit macOS's 256
    file-descriptor default with thousands of series).
    """
    t = series.length
    lo, hi = max(0, index - context), min(t - 1, index + context)
    frame = IMAGE_SIZE * IMAGE_SIZE
    with open(series.path, "rb") as f:
        f.seek(series.offset + lo * frame * 2)
        block = np.fromfile(f, dtype=np.int16, count=(hi - lo + 1) * frame)
    block = block.reshape(hi - lo + 1, IMAGE_SIZE, IMAGE_SIZE)
    idx = np.clip(np.arange(index - context, index + context + 1), lo, hi) - lo
    return block[idx]


def read_volume(series: Series) -> np.ndarray:
    with open(series.path, "rb") as f:
        f.seek(series.offset)
        vol = np.fromfile(f, dtype=np.int16, count=series.length * IMAGE_SIZE * IMAGE_SIZE)
    return vol.reshape(series.length, IMAGE_SIZE, IMAGE_SIZE)


class SliceDataset(Dataset):
    """(series, slice) items -> (HU context [2C+1, H, W] float32, labels [6] float32)."""

    def __init__(
        self,
        series: list[Series],
        context: int,
        context_dropout: float = 0.0,
        seed: int = 0,
    ):
        self.series = series
        self.context = context
        self.context_dropout = context_dropout
        self.items = np.asarray(
            [(s, i) for s in range(len(series)) for i in range(series[s].length)], dtype=np.int64
        )
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.items)

    def labels(self) -> np.ndarray:
        return np.concatenate([s.labels for s in self.series]).astype(np.float32)

    def __getitem__(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        s, i = self.items[k]
        series = self.series[s]
        x = read_context(series, int(i), self.context)
        # Simulate a lone uploaded slice (neighbours = the slice itself), so the single-slice
        # endpoint stays usable with the 2.5D model.
        if self.context and self.context_dropout and self.rng.random() < self.context_dropout:
            x = np.repeat(x[self.context : self.context + 1], len(x), axis=0)
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(
            series.labels[i].astype(np.float32)
        )


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: workers fork the dataset with the parent's numpy RNG, so
    every worker (and every epoch, with non-persistent workers) would draw the same context
    dropout. Reseed each from its torch seed, which differs per worker and per epoch."""
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    ds = info.dataset
    ds = getattr(ds, "dataset", ds)  # unwrap Subset
    if hasattr(ds, "rng"):
        ds.rng = np.random.default_rng(info.seed % 2**32)


def balanced_epoch(labels_any: np.ndarray, neg_ratio: float, rng: np.random.Generator):
    """All positive slices + neg_ratio x as many random negatives, shuffled."""
    pos = np.nonzero(labels_any > 0)[0]
    neg = np.nonzero(labels_any == 0)[0]
    k = min(len(neg), round(neg_ratio * len(pos)))
    idx = np.concatenate([pos, rng.choice(neg, size=k, replace=False)])
    rng.shuffle(idx)
    return idx
