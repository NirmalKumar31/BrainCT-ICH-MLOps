"""Black-box smoke test of a running API: readiness, a series and a slice prediction, metrics.

    python scripts/smoke_test.py http://127.0.0.1:8000 [--wait 60]

Needs only requests + pydicom (+ numpy); builds a synthetic CT series, never real data.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from dicom_factory import make_ct, make_series, phantom


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--wait", type=float, default=0)
    args = ap.parse_args()
    base = args.url.rstrip("/")

    deadline = time.time() + args.wait
    while True:
        try:
            r = requests.get(f"{base}/ready", timeout=3)
            if r.status_code == 200:
                break
        except requests.RequestException:
            pass
        if time.time() > deadline:
            sys.exit(f"not ready after {args.wait} s")
        time.sleep(1)
    print("ready:", r.json())

    blobs = make_series(16, size=512, bleed_at=(8,))
    files = [("files", (f"{i}.dcm", b, "application/dicom")) for i, b in enumerate(blobs)]
    r = requests.post(f"{base}/v1/predict/series?explain=true", files=files, timeout=60)
    r.raise_for_status()
    body = r.json()
    assert len(body["slices"]) == 16, body
    assert body["ordering"] == "position"
    assert body["heatmap"]["png_base64"]
    print("series:", body["study"], f"{body['timing']['total_ms']:.0f} ms")

    r = requests.post(
        f"{base}/v1/predict/slice", files={"file": ("a.dcm", make_ct(phantom(512)))}, timeout=30
    )
    r.raise_for_status()
    print("slice:", r.json()["prediction"]["probabilities"])

    r = requests.post(f"{base}/v1/predict/slice", files={"file": ("x.dcm", b"junk")}, timeout=30)
    assert r.status_code == 422, r.status_code

    metrics = requests.get(f"{base}/metrics", timeout=5).text
    assert 'brainct_requests_total{endpoint="/v1/predict/series",status="200"}' in metrics
    print("smoke test passed")


if __name__ == "__main__":
    main()
