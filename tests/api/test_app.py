from __future__ import annotations

import base64
import io
import json
import zipfile

import cv2
import numpy as np
import pytest
from dicom_factory import make_ct, make_series, phantom

from brainct.constants import LABELS


def files(blobs, name="s{}.dcm"):
    return [("files", (name.format(i), b, "application/dicom")) for i, b in enumerate(blobs)]


@pytest.fixture
def client(make_client):
    return make_client()


def test_health_ready_model(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready").json()
    assert r["status"] == "ready" and r["version"] == "0.0.0-standin"
    card = client.get("/v1/model").json()
    assert card["contract"]["labels"] == list(LABELS)
    assert "thresholds" in card


def test_series_prediction_structure_and_ordering(client):
    blobs = make_series(12, bleed_at=(5,))
    r = client.post("/v1/predict/series", files=files(blobs))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ordering"] == "position"
    assert [s["z"] for s in body["slices"]] == [5.0 * k for k in range(12)]
    assert body["model"]["head"] == "seq"
    study = body["study"]
    assert set(study["probabilities"]) == set(LABELS)
    for label in LABELS:
        assert study["probabilities"][label] == pytest.approx(
            max(s["probabilities"][label] for s in body["slices"]), abs=1e-5
        )
    assert 0 <= study["key_slice"] < 12
    assert body["heatmap"] is None
    assert "Research use only" in body["disclaimer"]
    assert r.headers["x-request-id"]


def test_series_is_order_invariant(client):
    blobs = make_series(10)
    a = client.post("/v1/predict/series", files=files(blobs)).json()
    b = client.post("/v1/predict/series", files=files(blobs[::-1])).json()
    assert a["slices"] == b["slices"]


def test_series_from_zip_with_heatmap(client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i, b in enumerate(make_series(9)):
            zf.writestr(f"series/{i}.dcm", b)
    r = client.post(
        "/v1/predict/series?explain=true",
        files=[("files", ("scan.zip", buf.getvalue(), "application/zip"))],
    )
    assert r.status_code == 200, r.text
    hm = r.json()["heatmap"]
    img = cv2.imdecode(
        np.frombuffer(base64.b64decode(hm["png_base64"]), np.uint8), cv2.IMREAD_COLOR
    )
    assert img.shape == (224, 224, 3)
    assert hm["slice_index"] == r.json()["study"]["key_slice"]


def test_short_series_warns(client):
    r = client.post("/v1/predict/series", files=files(make_series(3)))
    assert any("only 3 slices" in w for w in r.json()["warnings"])


def test_slice_endpoint(client):
    blob = make_ct(phantom(512), z=0.0)
    r = client.post("/v1/predict/slice?explain=true", files={"file": ("a.dcm", blob)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"]["head"] == "cnn"
    assert set(body["prediction"]["probabilities"]) == set(LABELS)
    assert body["heatmap"]["slice_index"] == 0
    assert any("single slice" in w for w in body["warnings"])


@pytest.mark.parametrize(
    ("blob", "status", "fragment"),
    [
        (b"", 400, "empty"),
        (b"definitely not dicom", 422, "DICOM"),
        (make_ct(phantom(64), modality="MR"), 422, "expected CT"),
    ],
)
def test_slice_errors(client, blob, status, fragment):
    r = client.post("/v1/predict/slice", files={"file": ("a.dcm", blob)})
    assert r.status_code == status
    assert fragment in r.json()["detail"]


def test_series_errors(make_client):
    client = make_client(max_slices=4, max_file_mb=0.2)
    mixed = [make_ct(phantom(64), series_uid="1.2.3"), make_ct(phantom(64), series_uid="1.2.4")]
    r = client.post("/v1/predict/series", files=files(mixed))
    assert r.status_code == 422 and "mixes 2 series" in r.json()["detail"]
    r = client.post("/v1/predict/series", files=files(make_series(5, size=64)))
    assert r.status_code == 413
    big = make_ct(phantom(512))  # 512 KB > 0.2 MB
    r = client.post("/v1/predict/series", files=files([big]))
    assert r.status_code == 413
    r = client.post("/v1/predict/series", files=files([b""]))
    assert r.status_code == 400


def test_metrics_and_drift(client):
    for _ in range(2):
        client.post("/v1/predict/series", files=files(make_series(8, size=64)))
    drift = client.get("/v1/drift").json()
    assert drift["features"]["tissue_mean_hu"]["count"] == 16
    assert drift["features"]["tissue_mean_hu"]["psi"] is not None  # min_count=5 in tests
    text = client.get("/metrics").text
    assert 'brainct_requests_total{endpoint="/v1/predict/series",status="200"} 2.0' in text
    assert 'brainct_drift_psi{feature="tissue_mean_hu"}' in text
    assert "brainct_slices_per_request_bucket" in text
    assert 'brainct_model_info{name="brainct-standin",version="0.0.0-standin"} 1.0' in text
    client.get("/no/such/path")
    assert 'endpoint="other"' in client.get("/metrics").text


def test_missing_bundle_is_not_ready(make_client, tmp_path):
    client = make_client(bundle=str(tmp_path))
    assert client.get("/health").status_code == 200
    r = client.get("/ready")
    assert r.status_code == 503 and "model_card.json" in r.json()["detail"]
    r = client.post("/v1/predict/series", files=files(make_series(2, size=64)))
    assert r.status_code == 503
    assert "brainct_model_loaded 0.0" in client.get("/metrics").text


def test_tampered_bundle_is_rejected(make_client, standin_bundle, tmp_path):
    import shutil

    copy = tmp_path / "b"
    shutil.copytree(standin_bundle, copy)
    card = json.loads((copy / "model_card.json").read_text())
    card["files"]["cnn"]["sha256"] = "0" * 64
    (copy / "model_card.json").write_text(json.dumps(card))
    r = make_client(bundle=str(copy)).get("/ready")
    assert r.status_code == 503 and "checksum" in r.json()["detail"]
