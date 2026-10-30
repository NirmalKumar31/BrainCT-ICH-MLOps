"""Patient-level train/val/test split, stratified by each patient's rarest hemorrhage subtype.

Every slice of a patient lands in one split, so no patient's anatomy is seen in training and
then scored in test (slice-level splits leak and inflate AUC). Stratifying on "any" alone left
validation with no epidural patient at all (0.5% of slices), so thresholds and calibration for
that subtype could not be fit; strata are therefore the rarest subtype a patient has.

    uv run python -m brainct.data.split --data data/rsna
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from brainct.constants import LABELS

DEFAULT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


# Rarest first (slice prevalence in RSNA): a patient is stratified by the first one they have.
RARITY_ORDER = ("epidural", "intraventricular", "subarachnoid", "intraparenchymal", "subdural")


def stratum(labels: set[str]) -> str:
    for name in RARITY_ORDER:
        if name in labels:
            return name
    return "any" if "any" in labels else "negative"


def _no_empty_split(bounds: list[int], n: int) -> list[int]:
    """Per-stratum rounding can leave a split empty (n=5: 4/0/1; n=3: 2/1/0), which is exactly
    what stratifying is meant to prevent. Move one patient into each empty split from the
    largest; strata that already fill every split keep their boundaries (and assignments)."""
    counts = [b - a for a, b in zip([0, *bounds[:-1]], bounds, strict=True)]
    if n >= len(counts):
        for i, c in enumerate(counts):
            if c == 0:
                counts[counts.index(max(counts))] -= 1
                counts[i] = 1
    out, acc = [], 0
    for c in counts:
        acc += c
        out.append(acc)
    return out


def split_patients(
    patient_stratum: dict[str, str],
    fractions: dict[str, float] = DEFAULT_FRACTIONS,
    seed: int = 2026,
) -> dict[str, str]:
    """patient -> split name. Deterministic for a given seed and patient set."""
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    rng = random.Random(seed)
    out: dict[str, str] = {}
    for key in sorted(set(patient_stratum.values())):
        group = sorted(p for p, v in patient_stratum.items() if v == key)
        rng.shuffle(group)
        names = list(fractions)
        bounds, acc = [], 0.0
        for name in names:
            acc += fractions[name]
            bounds.append(round(acc * len(group)))
        bounds = _no_empty_split(bounds, len(group))
        start = 0
        for name, end in zip(names, bounds, strict=True):
            for p in group[start:end]:
                out[p] = name
            start = end
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/rsna"))
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)

    present: dict[str, set[str]] = defaultdict(set)
    series_patient: dict[str, str] = {}
    with open(args.data / "slices.csv", newline="") as f:
        for r in csv.DictReader(f):
            present[r["patient_id"]] |= {k for k in LABELS if r[k] == "1"}
            series_patient[r["series_id"]] = r["patient_id"]
    strata = {p: stratum(v) for p, v in present.items()}
    assignment = split_patients(strata, seed=args.seed)

    with open(args.data / "splits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series_id", "patient_id", "split"])
        for sid, pid in sorted(series_patient.items()):
            w.writerow([sid, pid, assignment[pid]])

    summary: dict[str, dict[str, int]] = {}
    for pid, name in assignment.items():
        s = summary.setdefault(name, {"patients": 0, **dict.fromkeys(LABELS, 0)})
        s["patients"] += 1
        for k in present[pid]:
            s[k] += 1
    print(json.dumps({"seed": args.seed, **summary}, indent=2))


if __name__ == "__main__":
    main()
