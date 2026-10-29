"""End-to-end check on real held-out series through a running API (local only: needs the cache).

    uv run python scripts/e2e_real_series.py http://127.0.0.1:8080 --series 60

Each cached test series (HU, 224 px) is encoded as one DICOM per slice, uploaded in shuffled
order, and the API's study-level p(any) is compared with the offline certified predictions
(artifacts/bundle.work/chain_int8_seq_test.npz + the bundle's calibration). Any difference
means the serving path (DICOM decode, ordering, resize, batching) diverges from evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from dicom_factory import make_ct  # noqa: E402

from brainct.constants import LABELS  # noqa: E402
from brainct.evaluation import calibrate  # noqa: E402
from brainct.metrics import roc_auc  # noqa: E402
from brainct.training.data import load_series, read_volume  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--series", type=int, default=60)
    args = ap.parse_args()

    card = json.loads((ROOT / "artifacts/bundle/model_card.json").read_text())
    cal = [tuple(card["calibration"]["seq"][k]) for k in LABELS]
    with np.load(ROOT / "artifacts/bundle.work/chain_int8_seq_test.npz") as z:
        offline = {k: z[k] for k in z.files}
    p_off = calibrate(offline["logits"], cal)
    index = {sid: i for i, sid in enumerate(offline["series_id"])}

    test = load_series(ROOT / "data/rsna", "test")
    rng = np.random.default_rng(0)
    pos = [s for s in test if s.labels[:, 0].any()]
    neg = [s for s in test if not s.labels[:, 0].any()]
    n = args.series // 2
    picks = [pos[i] for i in rng.choice(len(pos), n, replace=False)]
    picks += [neg[i] for i in rng.choice(len(neg), n, replace=False)]

    api_p, off_p, labels, ms = [], [], [], []
    max_diff = 0.0
    for s in picks:
        vol = read_volume(s)
        uid = f"1.2.826.0.1.3680043.8.498.{abs(hash(s.series_id)) % 10**12}"
        order = rng.permutation(len(vol))
        files = [
            ("files", (f"{i}.dcm", make_ct(vol[i].astype(float), z=2.5 * i, instance=int(i) + 1,
                                           series_uid=uid), "application/dicom"))
            for i in order
        ]  # fmt: skip
        t0 = time.perf_counter()
        r = requests.post(f"{args.url}/v1/predict/series", files=files, timeout=120)
        ms.append((time.perf_counter() - t0) * 1000)
        r.raise_for_status()
        body = r.json()
        a, b = offline["offsets"][index[s.series_id]], offline["offsets"][index[s.series_id] + 1]
        served = np.array([[sl["probabilities"][k] for k in LABELS] for sl in body["slices"]])
        max_diff = max(max_diff, float(np.abs(served - p_off[a:b]).max()))
        api_p.append(body["study"]["probabilities"]["any"])
        off_p.append(float(p_off[a:b, 0].max()))
        labels.append(int(s.labels[:, 0].any()))

    y = np.array(labels)
    print(json.dumps({
        "series": len(picks), "positive": int(y.sum()),
        "study_auc_api": roc_auc(y, np.array(api_p)),
        "study_auc_offline_same_series": roc_auc(y, np.array(off_p)),
        "max_abs_prob_diff_api_vs_offline": max_diff,
        "latency_ms_p50": float(np.median(ms)), "latency_ms_p95": float(np.percentile(ms, 95)),
    }, indent=2))  # fmt: skip
    assert max_diff < 1e-3, "serving path diverges from the certified evaluation"


if __name__ == "__main__":
    main()
