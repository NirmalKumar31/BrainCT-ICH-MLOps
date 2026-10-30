from __future__ import annotations

import io
import json
import tarfile

import cv2
import numpy as np
import pytest

from brainct.constants import LABELS
from brainct.data import ingest
from brainct.data.split import split_patients, stratum


# ----------------------------------------------------------------------- split
def test_split_is_patient_disjoint_deterministic_and_proportional():
    strata = {f"p{i}": ("negative" if i % 3 else "subdural") for i in range(300)}
    a = split_patients(strata, seed=1)
    assert a == split_patients(strata, seed=1)
    assert a != split_patients(strata, seed=2)
    assert set(a) == set(strata)
    counts = {s: list(a.values()).count(s) for s in ("train", "val", "test")}
    assert counts == {"train": 210, "val": 45, "test": 45}
    for s in ("train", "val", "test"):  # every stratum reaches every split
        assert any(a[p] == s and strata[p] == "subdural" for p in strata)


def test_stratum_prefers_rarest_subtype():
    assert stratum({"any", "subdural", "epidural"}) == "epidural"
    assert stratum({"any", "subdural"}) == "subdural"
    assert stratum(set()) == "negative"


def test_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        split_patients({"p": "negative"}, fractions={"train": 0.5, "test": 0.4})


# ----------------------------------------------------------------------- tar stream
def _png(value: int) -> bytes:
    ok, buf = cv2.imencode(".png", np.full((64, 64), value, np.uint16))
    assert ok
    return buf.tobytes()


def make_tar(series: dict[str, list[str]]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for sid, names in series.items():
            for n in names:
                data = _png(2048)
                info = tarfile.TarInfo(f"images/P_{sid}/S_{sid}/{sid}/{n}")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class FakeStream(ingest.ByteStream):
    def __init__(self, blob: bytes, start: int):  # no network
        self.blob, self.pos = blob, start

    def read(self, n: int) -> bytes:
        out = self.blob[self.pos : self.pos + n]
        self.pos += len(out)
        return out


def test_parse_header_validates_checksum():
    blob = make_tar({"A": ["IM1.png"]})
    name, size, flag = ingest.parse_header(blob[:512])
    assert name.endswith("A/IM1.png") and size > 0 and flag == b"0"
    corrupted = bytearray(blob[:512])
    corrupted[10] ^= 0xFF
    assert ingest.parse_header(bytes(corrupted)) is None


def test_iter_members_resyncs_from_the_middle_of_a_tar():
    blob = make_tar({"A": ["IM1.png", "IM2.png"], "B": ["IM1.png", "IM2.png"]})
    full = list(ingest.iter_members(FakeStream(blob, 0), limit=len(blob)))
    assert len(full) == 4
    # start inside the first member's data, at a 512-aligned offset
    mid = list(ingest.iter_members(FakeStream(blob, 1024), limit=len(blob)))
    assert [p.split("/")[-2:] for p, _ in mid][-2:] == [["B", "IM1.png"], ["B", "IM2.png"]]
    assert len(mid) < 4


def test_writer_keeps_complete_series_and_drops_partial(tmp_path):
    names = ["IM000001_a.png", "IM000002_b.png"]
    labels = np.array([[0] * 6, [1, 0, 0, 0, 0, 1]], dtype=np.uint8)
    tables = ingest.Tables(
        series={"S1": ("P1", "T1", names, labels), "S2": ("P2", "T2", names, labels)},
        rescale={"S1": (1.0, -2048.0), "S2": (1.0, -2048.0)},
    )
    w = ingest.Writer(tmp_path, tables)
    assert w.write("S1", {n: _png(2048 + 40) for n in names})
    assert not w.write("S2", {names[0]: _png(2048)})  # cut by a region boundary
    assert not w.write("S1", {n: _png(0) for n in names})  # already cached
    vol = np.load(tmp_path / "series" / "S1.npy")
    assert vol.shape == (2, 224, 224) and vol.dtype == np.int16
    assert (vol == 40).all()  # 2048 + 40 stored, intercept -2048 -> 40 HU
    rows = (tmp_path / "slices.csv").read_text().strip().splitlines()
    assert rows[0].split(",")[5:] == list(LABELS)
    assert rows[2].endswith("1,0,0,0,0,1")
    assert (w.kept, w.dropped) == (1, 1)


# ----------------------------------------------------------------------- network layer
class FakeResponse:
    def __init__(self, blob: bytes, start: int, fail_after: int | None):
        self.blob, self.start, self.fail_after = blob, start, fail_after
        self.status_code = 206

    def raise_for_status(self) -> None:
        pass

    def iter_content(self, chunk_size: int):
        sent = 0
        for i in range(self.start, len(self.blob), 100):
            if self.fail_after is not None and sent >= self.fail_after:
                raise ingest.requests.ConnectionError("connection reset")
            chunk = self.blob[i : i + 100]
            sent += len(chunk)
            yield chunk


class FakeSession:
    """Serves byte ranges of `blob`; the first connection dies after 250 bytes."""

    def __init__(self, blob: bytes):
        self.blob, self.starts = blob, []

    def get(self, url, headers, stream, timeout):
        start = int(headers["Range"].removeprefix("bytes=").rstrip("-"))
        self.starts.append(start)
        return FakeResponse(self.blob, start, 250 if len(self.starts) == 1 else None)


def test_byte_stream_resumes_at_the_right_offset(monkeypatch):
    monkeypatch.setattr(ingest.time, "sleep", lambda s: None)
    blob = bytes(range(256)) * 8
    session = FakeSession(blob)
    stream = ingest.ByteStream("http://x", 16, session)
    assert stream.read(100) == blob[16:116]
    assert stream.read(400) == blob[116:516]  # crosses the dropped connection
    assert session.starts == [16, 316]  # reconnected after the 300 bytes already buffered
    assert stream.pos == 516


def test_stream_region_and_load_tables(tmp_path, monkeypatch):
    names = ["IM000001_a.png", "IM000002_b.png"]
    blob = make_tar({"A": names, "B": names, "C": names})
    (tmp_path / "tables").mkdir()
    header = "patient_ID,study_ID,series_ID,filename," + ",".join(LABELS) + ",extra\n"
    rows = "".join(
        f"P_{s},S_{s},{s},{n},{','.join(['0'] * 6)},0\n" for s in "ABC" for n in reversed(names)
    )
    (tmp_path / "tables" / "slice_labels.csv").write_text(header + rows)
    (tmp_path / "tables" / "rescale_values.csv").write_text(
        "patient_id,study_id,series_id,plane,rescale_slope,rescale_intercept\n"
        + "".join(f"P_{s},S_{s},{s},Axial,1.0,-2048.0\n" for s in "ABC")
    )
    tables = ingest.load_tables(tmp_path / "tables")
    assert tables.series["A"][2] == names  # sorted into instance order
    monkeypatch.setattr(ingest, "ByteStream", lambda url, start, session: FakeStream(blob, start))
    writer = ingest.Writer(tmp_path / "out", tables)
    result = ingest.stream_region("http://x", 0, budget=1, writer=writer, max_series=None)
    # budget exhausted after the first complete series: stop at the next boundary
    assert result["kept"] == 1 and writer.done("A") and not writer.done("C")
    result = ingest.stream_region("http://x", 0, budget=len(blob), writer=writer, max_series=None)
    assert writer.done("B") and writer.done("C")


def test_split_cli(tmp_path, capsys):
    from brainct.data import split

    lines = ["patient_id,study_id,series_id,index,filename," + ",".join(LABELS)]
    for p in range(40):
        for i in range(3):
            pos = "1,0,0,0,0,1" if p % 4 == 0 and i == 1 else "0,0,0,0,0,0"
            lines.append(f"P{p},T{p},S{p},{i},f{i}.png,{pos}")
    (tmp_path / "slices.csv").write_text("\n".join(lines) + "\n")
    split.main(["--data", str(tmp_path)])
    out = json.loads(capsys.readouterr().out)
    assert out["train"]["patients"] + out["val"]["patients"] + out["test"]["patients"] == 40
    assert all(out[s]["subdural"] > 0 for s in ("train", "val", "test"))
    rows = (tmp_path / "splits.csv").read_text().splitlines()
    assert rows[0] == "series_id,patient_id,split" and len(rows) == 41


def test_split_never_leaves_a_split_empty():
    from collections import Counter

    from brainct.data.split import split_patients

    for n in (3, 4, 5, 7):
        counts = Counter(split_patients({f"p{i}": "epidural" for i in range(n)}).values())
        assert set(counts) == {"train", "val", "test"}, (n, counts)
