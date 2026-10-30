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
    _write_card(copy, card)
    r = make_client(bundle=str(copy)).get("/ready")
    assert r.status_code == 503 and "checksum" in r.json()["detail"]


def _bundle_with_selftest(src, dst, golden_shift: float):
    """Stand-in + FP32 fallback (same graph) + golden logits shifted by `golden_shift`."""
    import hashlib
    import shutil

    from brainct.serving.model import InferenceModel, contexts

    shutil.copytree(src, dst)
    shutil.copy(dst / "cnn.onnx", dst / "cnn_fp32.onnx")
    vol = np.zeros((4, 224, 224), np.int16)
    m = InferenceModel(dst, verify=False)
    golden = m.cnn.run(["logits"], {"hu": contexts(vol, m.context)})[0] + golden_shift
    np.savez(dst / "selftest.npz", volume=vol, certified_logits=golden)
    card = json.loads((dst / "model_card.json").read_text())
    for name, path in (("cnn_fp32", "cnn_fp32.onnx"), ("selftest", "selftest.npz")):
        digest = hashlib.sha256((dst / path).read_bytes()).hexdigest()
        card["files"][name] = {"path": path, "sha256": digest}
    card["selftest"] = {"max_prob_diff": 0.02}
    _write_card(dst, card)
    return dst


def _write_card(bundle, card) -> None:
    """Rewrite a test bundle's card together with its checksum."""
    import hashlib

    (bundle / "model_card.json").write_text(json.dumps(card))
    digest = hashlib.sha256((bundle / "model_card.json").read_bytes()).hexdigest()
    (bundle / "model_card.sha256").write_text(digest)


def test_selftest_passes_and_serves_int8(make_client, standin_bundle, tmp_path):
    b = _bundle_with_selftest(standin_bundle, tmp_path / "ok", golden_shift=0.0)
    client = make_client(bundle=str(b))
    assert client.get("/ready").json()["precision"] == "int8"
    card = client.get("/v1/model").json()
    assert card["selftest"]["passed"] is True
    assert 'brainct_precision{precision="int8"} 1.0' in client.get("/metrics").text


def test_selftest_failure_falls_back_to_fp32(make_client, standin_bundle, tmp_path):
    """The GKE incident: INT8 kernels on a non-VNNI x86 CPU diverged from the certified model."""
    b = _bundle_with_selftest(standin_bundle, tmp_path / "bad", golden_shift=3.0)
    client = make_client(bundle=str(b))
    assert client.get("/ready").json()["precision"] == "fp32"
    st = client.get("/v1/model").json()["selftest"]
    assert st["passed"] is False and st["max_prob_diff"] > st["limit"]
    forced = make_client(bundle=str(b), precision="int8")
    assert forced.get("/ready").json()["precision"] == "int8"


# ---- regressions from the 2026-10 audit (docs/decisions.md, "Audit")


def _edit(blob: bytes, **tags) -> bytes:
    from pydicom import dcmread

    ds = dcmread(io.BytesIO(blob))
    for k, v in tags.items():
        setattr(ds, k, v)
    buf = io.BytesIO()
    ds.save_as(buf)
    return buf.getvalue()


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_malformed_dicoms_are_client_errors(client):
    tiny = make_ct(np.zeros((8, 8)))  # used to reach resize_hu and crash with a 500
    huge = _edit(make_ct(phantom(64)), Rows=4096, Columns=4096)  # declared size, checked first
    for blob in (tiny, huge):
        r = client.post("/v1/predict/series", files=files([blob]))
        assert r.status_code == 422, r.text
    empty = _edit(make_ct(phantom(64)), RescaleSlope="", InstanceNumber="")  # empty = default
    assert client.post("/v1/predict/series", files=files([empty])).status_code == 200


def test_unhandled_errors_are_counted_and_tagged(make_client, monkeypatch):
    from brainct.serving.model import InferenceModel

    client = make_client()
    monkeypatch.setattr(InferenceModel, "calibrate", lambda *a: 1 / 0)
    r = client.post("/v1/predict/series", files=files(make_series(4)))
    assert r.status_code == 500 and r.json()["request_id"] == r.headers["x-request-id"]
    text = client.get("/metrics").text
    assert 'brainct_requests_total{endpoint="/v1/predict/series",status="500"} 1.0' in text


def test_total_limit_counts_uncompressed_bytes(make_client):
    client = make_client(max_total_mb=1.0, max_file_mb=4.0)
    blob = make_ct(phantom(512))  # ~0.5 MB uncompressed, a few KB zipped
    zips = [("files", (f"z{i}.zip", _zip({"a.dcm": blob}), "application/zip")) for i in range(3)]
    r = client.post("/v1/predict/series", files=zips)
    assert r.status_code == 413, r.text


def test_request_body_is_capped_while_streaming(make_client):
    client = make_client(max_request_mb=0.05)
    r = client.post("/v1/predict/series", files=files(make_series(4, size=128)))
    assert r.status_code == 413 and "request body" in r.json()["detail"]
    text = client.get("/metrics").text
    assert 'endpoint="/v1/predict/series",status="413"' in text


def test_request_ids_are_server_generated(client):
    blobs = make_series(4)
    hdr = {"x-request-id": "same"}
    a = client.post("/v1/predict/series", files=files(blobs), headers=hdr)
    b = client.post("/v1/predict/series", files=files(blobs), headers=hdr)
    ids = {a.headers["x-request-id"], b.headers["x-request-id"]}
    assert len(ids) == 2 and "same" not in ids
    for rid in ids:  # both predictions can receive their own ground truth
        r = client.post("/v1/feedback", json={"request_id": rid, "labels": {"any": 0}})
        assert r.status_code == 200


def test_feedback_token(make_client):
    client = make_client(feedback_token="s3cret")
    rid = client.post("/v1/predict/series", files=files(make_series(4))).headers["x-request-id"]
    body = {"request_id": rid, "labels": {"any": 1}}
    assert client.post("/v1/feedback", json=body).status_code == 401
    ok = client.post("/v1/feedback", json=body, headers={"x-feedback-token": "s3cret"})
    assert ok.status_code == 200


def test_bundle_without_series_model_serves_cnn_head(make_client, standin_bundle, tmp_path):
    import shutil

    b = tmp_path / "noseq"
    shutil.copytree(standin_bundle, b)
    card = json.loads((b / "model_card.json").read_text())
    del card["files"]["seq"]
    _write_card(b, card)
    client = make_client(bundle=str(b))
    r = client.post("/v1/predict/series", files=files(make_series(6)))
    assert r.status_code == 200 and r.json()["model"]["head"] == "cnn"


def test_selftest_failure_without_fp32_is_not_ready(make_client, standin_bundle, tmp_path):
    b = _bundle_with_selftest(standin_bundle, tmp_path / "bad", golden_shift=3.0)
    card = json.loads((b / "model_card.json").read_text())
    del card["files"]["cnn_fp32"]
    _write_card(b, card)
    r = make_client(bundle=str(b)).get("/ready")
    assert r.status_code == 503 and "self-test failed" in r.json()["detail"]
    r = make_client(bundle=str(b), precision="fp32").get("/ready")
    assert r.status_code == 503 and "no FP32 model" in r.json()["detail"]


def test_tampered_card_is_rejected(make_client, standin_bundle, tmp_path):
    import shutil

    b = tmp_path / "card"
    shutil.copytree(standin_bundle, b)
    assert (b / "model_card.sha256").exists()
    card = json.loads((b / "model_card.json").read_text())
    card["thresholds"]["seq"]["study_any"] = 0.0  # every study positive
    (b / "model_card.json").write_text(json.dumps(card))
    r = make_client(bundle=str(b)).get("/ready")
    assert r.status_code == 503 and "model_card.json: checksum" in r.json()["detail"]


def test_series_assembly_rejects_partial_uids_and_drops_duplicates(client):
    blobs = make_series(8)
    no_uid = _edit(blobs[0], SeriesInstanceUID="")
    r = client.post("/v1/predict/series", files=files([no_uid, *blobs[1:]]))
    assert r.status_code == 422 and "SeriesInstanceUID" in r.json()["detail"]
    r = client.post("/v1/predict/series", files=files([*blobs, blobs[3]]))
    assert r.status_code == 200
    body = r.json()
    assert len(body["slices"]) == 8 and any("duplicate" in w for w in body["warnings"])
