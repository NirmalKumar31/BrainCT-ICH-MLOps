"""Load test: each user uploads whole synthetic CT series (32 slices, 512x512) back to back.

    uv run --group loadtest locust -f load_tests/locustfile.py --host http://127.0.0.1:8080 \
        --headless -u 8 -r 1 -t 4m --csv load_tests/results/run --html load_tests/results/run.html

Step load (1 -> 8 users over the first minutes) via LoadTestShape below; payloads are built
once at start so the client measures the server, not DICOM encoding.
"""

from __future__ import annotations

import sys
from pathlib import Path

from locust import HttpUser, LoadTestShape, between, task

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from dicom_factory import make_series

SERIES = [make_series(32, size=512, bleed_at=(10, 11), series_uid=f"1.2.3.{k}") for k in range(4)]


class Radiology(HttpUser):
    wait_time = between(0.0, 0.2)

    @task
    def series(self) -> None:
        blobs = SERIES[self.environment.runner.user_count % len(SERIES)]
        files = [("files", (f"{i}.dcm", b, "application/dicom")) for i, b in enumerate(blobs)]
        with self.client.post(
            "/v1/predict/series", files=files, name="/v1/predict/series", catch_response=True
        ) as r:
            if r.status_code != 200 or len(r.json().get("slices", [])) != 32:
                r.failure(f"status {r.status_code}")


class Steps(LoadTestShape):
    """Users: 1, 2, 4, 6, 8 for 45 s each."""

    steps = (1, 2, 4, 6, 8)
    step_seconds = 45

    def tick(self):
        k = int(self.get_run_time() // self.step_seconds)
        if k >= len(self.steps):
            return None
        return self.steps[k], 8
