from __future__ import annotations

import asyncio

import numpy as np
import pytest
from dicom_factory import make_series

from brainct.monitoring.performance import PerformanceMonitor
from brainct.serving.batching import MicroBatcher


def fake_forward(calls: list[int]):
    def forward(x: np.ndarray):
        calls.append(len(x))
        flat = x.reshape(len(x), -1)
        return flat[:, :6] * 2.0, flat[:, :3] + 1.0

    return forward


def test_batcher_merges_concurrent_requests_and_splits_results():
    calls: list[int] = []
    sizes: list[int] = []

    async def main():
        b = MicroBatcher(fake_forward(calls), max_batch=32, max_wait_ms=20, on_batch=sizes.append)
        await b.start()
        inputs = [np.full((n, 1, 2, 3), float(i)) for i, n in enumerate((3, 5, 1, 7))]
        outs = await asyncio.gather(*(b.submit(x) for x in inputs))
        await b.stop()
        return inputs, outs

    inputs, outs = asyncio.run(main())
    assert calls == [16]  # four requests, one model call
    assert sizes == [16]
    for x, (logits, emb) in zip(inputs, outs, strict=True):
        assert logits.shape == (len(x), 6)
        np.testing.assert_array_equal(logits, x.reshape(len(x), -1)[:, :6] * 2.0)
        np.testing.assert_array_equal(emb, x.reshape(len(x), -1)[:, :3] + 1.0)


def test_batcher_respects_max_batch_and_splits_large_inputs():
    calls: list[int] = []

    async def main():
        b = MicroBatcher(fake_forward(calls), max_batch=8, max_wait_ms=10)
        await b.start()
        logits, _ = await b.submit(np.zeros((20, 1, 2, 3)))
        await b.stop()
        return logits

    logits = asyncio.run(main())
    assert len(logits) == 20
    assert max(calls) <= 8 and sum(calls) == 20


def test_batcher_propagates_errors():
    def boom(x):
        raise RuntimeError("model failed")

    async def main():
        b = MicroBatcher(boom, max_batch=8, max_wait_ms=1)
        await b.start()
        try:
            await b.submit(np.zeros((2, 1, 2, 3)))
        finally:
            await b.stop()

    with pytest.raises(RuntimeError, match="model failed"):
        asyncio.run(main())


def test_performance_monitor_joins_feedback():
    labels = ["any", "subdural"]
    mon = PerformanceMonitor(labels, keep=100, window=50)
    rng = np.random.default_rng(0)
    for i in range(40):
        y = i % 2
        mon.record(f"r{i}", np.array([0.8 if y else 0.2, rng.random()]))
        assert mon.feedback(f"r{i}", {"any": y})  # subdural unknown: excluded from its metrics
    assert not mon.feedback("r0", {"any": 1})  # already joined
    assert not mon.feedback("never-seen", {"any": 1})
    mon.record("x", np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="unknown labels"):
        mon.feedback("x", {"epidural": 1})
    snap = mon.snapshot()
    assert snap["any"]["count"] == 40 and snap["any"]["auc"] == 1.0
    assert snap["any"]["brier"] == pytest.approx(0.04)
    assert snap["subdural"]["count"] == 0 and snap["subdural"]["auc"] is None


def test_performance_monitor_evicts_old_predictions():
    mon = PerformanceMonitor(["any"], keep=2)
    for i in range(3):
        mon.record(f"r{i}", np.array([0.5]))
    assert not mon.feedback("r0", {"any": 1})
    assert mon.feedback("r2", {"any": 1})


def _files(blobs):
    return [("files", (f"{i}.dcm", b, "application/dicom")) for i, b in enumerate(blobs)]


def test_api_batching_gives_identical_results(make_client):
    blobs = make_series(10, size=64)
    on = make_client(batching=True).post("/v1/predict/series", files=_files(blobs)).json()
    off = make_client(batching=False).post("/v1/predict/series", files=_files(blobs)).json()
    assert on["slices"] == off["slices"]
    assert on["study"] == off["study"]


def test_api_feedback_loop(make_client):
    client = make_client()
    r = client.post("/v1/predict/series", files=_files(make_series(6, size=64)))
    rid = r.headers["x-request-id"]
    assert client.post("/v1/feedback", json={"request_id": rid, "labels": {"any": 1}}).json() == {
        "status": "recorded"
    }
    again = client.post("/v1/feedback", json={"request_id": rid, "labels": {"any": 1}})
    assert again.status_code == 404
    bad = client.post("/v1/feedback", json={"request_id": "x", "labels": {"nope": 1}})
    assert bad.status_code in (404, 422)
    perf = client.get("/v1/performance").json()
    assert perf["labels"]["any"]["count"] == 1
    metrics = client.get("/metrics").text
    assert 'brainct_feedback_total{result="joined"} 1.0' in metrics
    assert 'brainct_live_metric{label="any",metric="count"} 1.0' in metrics
    assert "brainct_batch_slices_bucket" in metrics
