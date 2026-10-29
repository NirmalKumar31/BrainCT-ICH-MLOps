"""In-cluster step load through the Service (port-forward pins one pod, so it cannot test
load balancing). Runs as a Job in the API image: stdlib + numpy + pydicom only.

    kubectl create configmap brainct-load --from-file=load_tests/incluster.py \
        --from-file=tests/dicom_factory.py
    kubectl apply -f load_tests/job.yaml && kubectl logs -f job/brainct-load
"""

from __future__ import annotations

import os
import statistics
import sys
import threading
import time
import urllib.request
import uuid

sys.path.insert(0, "/load")
from dicom_factory import make_series

URL = os.environ.get("TARGET", "http://brainct-brainct/v1/predict/series")
STEPS = [int(x) for x in os.environ.get("STEPS", "1,2,4,6,8").split(",")]
STEP_SECONDS = float(os.environ.get("STEP_SECONDS", "60"))


def multipart(blobs: list[bytes]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for i, b in enumerate(blobs):
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{i}.dcm"\r\n'
            "Content-Type: application/dicom\r\n\r\n".encode()
            + b
            + b"\r\n"
        )
    return b"".join(parts) + f"--{boundary}--\r\n".encode(), boundary


BODY, BOUNDARY = multipart(make_series(32, size=512, bleed_at=(10, 11)))


def worker(stop: threading.Event, out: list[tuple[float, bool]]) -> None:
    while not stop.is_set():
        req = urllib.request.Request(URL, data=BODY, method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={BOUNDARY}")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                ok = r.status == 200
                r.read()
        except Exception:
            ok = False
        out.append((time.perf_counter() - t0, ok))


def main() -> None:
    print("| users | req/s | p50 ms | p95 ms | errors |\n|---|---|---|---|---|", flush=True)
    for users in STEPS:
        stop, results = threading.Event(), []
        threads = [threading.Thread(target=worker, args=(stop, results)) for _ in range(users)]
        for t in threads:
            t.start()
        time.sleep(STEP_SECONDS / 3)  # ramp-in, not measured
        results.clear()
        time.sleep(STEP_SECONDS * 2 / 3)
        window = list(results)
        stop.set()
        for t in threads:
            t.join()
        lat = sorted(x for x, _ in window)
        p95 = lat[int(0.95 * (len(lat) - 1))] * 1000 if lat else float("nan")
        p50 = statistics.median(lat) * 1000 if lat else float("nan")
        rate = len(window) / (STEP_SECONDS * 2 / 3)
        errors = sum(not ok for _, ok in window)
        print(f"| {users} | {rate:.2f} | {p50:.0f} | {p95:.0f} | {errors} |", flush=True)


if __name__ == "__main__":
    main()
