from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest
from dicom_factory import make_ct, make_series, phantom

from brainct.monitoring.drift import DriftMonitor, build_reference, histogram, psi, slice_features
from brainct.serving.series import UploadError, assemble, decode_one, unpack_zip


# ----------------------------------------------------------------------- drift
def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    ref = histogram(rng.normal(30, 5, 5000), -20, 80)
    same = histogram(rng.normal(30, 5, 5000), -20, 80)
    shifted = histogram(rng.normal(55, 5, 5000), -20, 80)
    assert psi(np.array(ref), np.array(same)) < 0.02
    assert psi(np.array(ref), np.array(shifted)) > 1.0


def test_histogram_clips_out_of_range_into_end_bins():
    h = histogram(np.array([-100.0, 500.0]), 0, 10, bins=5)
    assert h == [0.5, 0, 0, 0, 0.5]


def test_slice_features_on_phantom():
    f = slice_features(phantom(224))
    assert 25 < f["tissue_mean_hu"] < 45
    assert 0.3 < f["air_fraction"] < 0.7
    assert f["bone_fraction"] > 0.02


def test_monitor_waits_for_min_count_then_flags_drift():
    rng = np.random.default_rng(1)
    ref = build_reference({"tissue_mean_hu": rng.normal(30, 5, 4000)})
    mon = DriftMonitor(ref, window=500, min_count=100)
    for v in rng.normal(30, 5, 50):
        mon.observe({"tissue_mean_hu": v, "ignored": 1.0})
    assert mon.snapshot()["tissue_mean_hu"]["psi"] is None
    for v in rng.normal(30, 5, 200):
        mon.observe({"tissue_mean_hu": v})
    assert mon.snapshot()["tissue_mean_hu"]["psi"] < 0.1
    for v in rng.normal(60, 5, 500):  # e.g. a wrong rescale intercept upstream
        mon.observe({"tissue_mean_hu": v})
    snap = mon.snapshot()["tissue_mean_hu"]
    assert snap["psi"] > 0.2 and snap["count"] == 500


# ----------------------------------------------------------------------- series assembly
def _decode_all(blobs):
    return [decode_one(b, 224) for b in blobs]


def test_assemble_orders_by_position():
    vol = assemble(_decode_all(make_series(10)))
    assert vol.order == "position"
    assert [i.z for i in vol.infos] == [5.0 * k for k in range(10)]
    assert vol.hu.shape == (10, 224, 224)


def test_assemble_falls_back_to_instance_number_then_upload_order():
    blobs = [make_ct(phantom(64), z=None, instance=n) for n in (3, 1, 2)]
    vol = assemble(_decode_all(blobs))
    assert vol.order == "instance_number"
    assert [i.instance_number for i in vol.infos] == [1, 2, 3]
    blobs = [make_ct(phantom(64), z=None, instance=None) for _ in range(2)]
    assert assemble(_decode_all(blobs)).order == "upload"


def test_assemble_rejects_mixed_series_and_empty():
    blobs = [make_ct(phantom(64), series_uid="1.2.3"), make_ct(phantom(64), series_uid="1.2.4")]
    with pytest.raises(UploadError) as e:
        assemble(_decode_all(blobs))
    assert e.value.status == 422
    with pytest.raises(UploadError):
        assemble([])


def test_decode_rejects_non_ct():
    with pytest.raises(UploadError, match="expected CT"):
        decode_one(make_ct(phantom(64), modality="MR"), 224)


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_unpack_zip_limits():
    z = _zip({"a.dcm": b"x" * 10, "b.dcm": b"y" * 10, "__MACOSX/._a": b"junk", "dir/": b""})
    assert unpack_zip(z, 10, 1000) == [b"x" * 10, b"y" * 10]
    with pytest.raises(UploadError, match="limit"):
        unpack_zip(z, 1, 1000)
    bomb = _zip({"big.dcm": b"\0" * 5_000_000})  # compresses to a few KB
    assert len(bomb) < 50_000
    with pytest.raises(UploadError, match="exceed"):
        unpack_zip(bomb, 10, 1_000_000)
    with pytest.raises(UploadError, match="zip"):
        unpack_zip(b"PK not really", 10, 1000)


def test_settings_from_env(monkeypatch):
    from brainct.serving.settings import Settings

    monkeypatch.setenv("BRAINCT_THREADS", "3")
    monkeypatch.setenv("BRAINCT_WARMUP", "false")
    monkeypatch.setenv("BRAINCT_MAX_FILE_MB", "1.5")
    monkeypatch.setenv("BRAINCT_BUNDLE", "/models/x")
    s = Settings.from_env()
    assert (s.threads, s.warmup, s.max_file_mb, s.bundle) == (3, False, 1.5, "/models/x")
    assert s.max_slices == Settings().max_slices
