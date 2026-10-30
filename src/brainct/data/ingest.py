"""Stream a reproducible sample of RSNA ICH series straight from the 144 GB tar into an HU cache.

Source: the RSNA 2019 Intracranial Hemorrhage dataset as lossless 16-bit PNGs
(huggingface.co/datasets/ianpan/rsna-intracranial-hemorrhage-16bit-png, pinned by commit).
The tar holds one directory per series, and series appear in hash order, so any contiguous
byte range is a pseudo-random sample of series. We open `regions` evenly spaced HTTP range
requests in parallel, resynchronise on the first tar header in each, and keep only series whose
slice count matches the label table (series cut at a region start are dropped).

Nothing raw touches the disk: each complete series is decoded, converted to HU, resized and
written as one int16 array [T, H, W] in instance order, plus a row per slice in `slices.csv`.

    uv run python -m brainct.data.ingest --out data/rsna --regions 8 --gib-per-region 1.6
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import requests

from brainct.constants import IMAGE_SIZE, LABELS
from brainct.preprocessing.transforms import hu_from_stored, resize_hu

REPO = "ianpan/rsna-intracranial-hemorrhage-16bit-png"
REVISION = "c40c72b1c01479e2ef08cfd2bf68014960dbadc9"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}"
BLOCK = 512
log = logging.getLogger("brainct.ingest")


# ---------------------------------------------------------------------------- tar parsing
def parse_header(block: bytes) -> tuple[str, int, bytes] | None:
    """(name, size, typeflag) for a valid ustar header block, else None."""
    if len(block) != BLOCK or block[257:262] != b"ustar":
        return None
    stored = block[148:156].split(b"\0")[0].strip()
    try:
        checksum = int(stored or b"0", 8)
        size = int(block[124:136].split(b"\0")[0].strip() or b"0", 8)
    except ValueError:
        return None
    if checksum != sum(block[:148]) + 8 * 32 + sum(block[156:]):
        return None
    name = block[0:100].split(b"\0")[0].decode("utf-8", "replace")
    prefix = block[345:500].split(b"\0")[0].decode("utf-8", "replace")
    return (f"{prefix}/{name}" if prefix else name), size, block[156:157]


class ByteStream:
    """Sequential reader over an HTTP range that reconnects (and resumes) on network errors."""

    def __init__(self, url: str, start: int, session: requests.Session, retries: int = 8):
        # `pos` is always the tar offset of the next byte `read` returns (= offset of _buf[0]).
        self.url, self.pos, self.session, self.retries = url, start, session, retries
        self._iter: Iterator[bytes] | None = None
        self._buf = bytearray()

    def _connect(self) -> None:
        offset = self.pos + len(self._buf)  # resume after what is already buffered
        resp = self.session.get(
            self.url, headers={"Range": f"bytes={offset}-"}, stream=True, timeout=60
        )
        resp.raise_for_status()
        if resp.status_code != 206:
            raise RuntimeError("server ignored the Range header")
        self._iter = resp.iter_content(chunk_size=1 << 20)

    def read(self, n: int) -> bytes:
        attempt = 0
        while len(self._buf) < n:
            try:
                if self._iter is None:
                    self._connect()
                assert self._iter is not None
                self._buf += next(self._iter)
            except StopIteration:
                break
            except (requests.RequestException, ConnectionError, RuntimeError) as exc:
                attempt += 1
                if attempt > self.retries:
                    raise
                log.warning("stream at %d failed (%s); retry %d", self.pos, exc, attempt)
                self._iter = None
                time.sleep(min(2**attempt, 30))
        out = bytes(self._buf[:n])
        del self._buf[:n]
        self.pos += len(out)
        return out


def iter_members(stream: ByteStream, limit: int) -> Iterator[tuple[str, bytes]]:
    """Yield (path, data) of regular files from a stream that may start mid-tar.

    Stops at the first header past `limit` bytes. Resynchronises on a 512-byte boundary first.
    """
    long_name: str | None = None
    while True:
        block = stream.read(BLOCK)
        if len(block) < BLOCK or block == b"\0" * BLOCK:
            return
        header = parse_header(block)
        if header is None:  # still inside the data of a member cut by the range start
            continue
        if stream.pos > limit:
            return
        name, size, flag = header
        data = stream.read(size + (-size % BLOCK))[:size]
        if flag == b"L":  # GNU long name for the next member
            long_name = data.rstrip(b"\0").decode("utf-8", "replace")
            continue
        if flag in (b"0", b"\0"):
            yield (long_name or name), data
        long_name = None


# ---------------------------------------------------------------------------- tables
@dataclass
class Tables:
    # series_id -> (patient, study, sorted filenames, label matrix [T, 6])
    series: dict[str, tuple[str, str, list[str], np.ndarray]]
    rescale: dict[str, tuple[float, float]]


def load_tables(cache_dir: Path) -> Tables:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in ("slice_labels.csv", "rescale_values.csv"):
        path = cache_dir / name
        if not path.exists():
            log.info("downloading %s", name)
            resp = requests.get(f"{BASE}/{name}", timeout=120)
            resp.raise_for_status()
            path.write_bytes(resp.content)

    rows: dict[str, list[tuple[str, str, str, list[int]]]] = {}
    with open(cache_dir / "slice_labels.csv", newline="") as f:
        for r in csv.DictReader(f):
            labels = [int(r[k]) for k in LABELS]
            rows.setdefault(r["series_ID"], []).append(
                (r["patient_ID"], r["study_ID"], r["filename"], labels)
            )
    series = {}
    for sid, items in rows.items():
        items.sort(key=lambda t: t[2])  # IM000001_..., IM000002_...: instance order
        series[sid] = (
            items[0][0],
            items[0][1],
            [t[2] for t in items],
            np.asarray([t[3] for t in items], dtype=np.uint8),
        )
    rescale = {}
    with open(cache_dir / "rescale_values.csv", newline="") as f:
        for r in csv.DictReader(f):
            rescale[r["series_id"]] = (float(r["rescale_slope"]), float(r["rescale_intercept"]))
    return Tables(series, rescale)


# ---------------------------------------------------------------------------- writer
@dataclass
class Writer:
    out: Path
    tables: Tables
    lock: threading.Lock = field(default_factory=threading.Lock)
    kept: int = 0
    dropped: int = 0
    slices: int = 0

    def __post_init__(self) -> None:
        (self.out / "series").mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out / "slices.csv"
        if not self.csv_path.exists():
            self.csv_path.write_text(
                "patient_id,study_id,series_id,index,filename," + ",".join(LABELS) + "\n"
            )
        # A series counts as done once its rows are in slices.csv (written last). An .npy
        # without rows (crash between the two) is redone and overwritten, not skipped forever.
        with open(self.csv_path) as f:
            next(f)
            self._written = {line.split(",", 3)[2] for line in f if line.strip()}
        self._claimed: set[str] = set()

    def done(self, sid: str) -> bool:
        with self.lock:
            return sid in self._written

    def _claim(self, sid: str) -> bool:
        """Overlapping regions can both see a series: only one of them writes it."""
        with self.lock:
            if sid in self._written or sid in self._claimed:
                return False
            self._claimed.add(sid)
            return True

    def write(self, sid: str, files: dict[str, bytes]) -> bool:
        meta = self.tables.series.get(sid)
        if meta is None or sid not in self.tables.rescale or not self._claim(sid):
            return False
        try:
            return self._write(sid, meta, files)
        finally:
            with self.lock:
                self._claimed.discard(sid)

    def _write(self, sid: str, meta, files: dict[str, bytes]) -> bool:
        patient, study, names, labels = meta
        if set(files) != set(names):  # cut by a region boundary
            with self.lock:
                self.dropped += 1
            return False
        slope, intercept = self.tables.rescale[sid]
        vol = np.empty((len(names), IMAGE_SIZE, IMAGE_SIZE), dtype=np.int16)
        for i, name in enumerate(names):
            png = cv2.imdecode(np.frombuffer(files[name], np.uint8), cv2.IMREAD_UNCHANGED)
            if png is None or png.ndim != 2:
                with self.lock:
                    self.dropped += 1
                return False
            vol[i] = resize_hu(hu_from_stored(png, slope, intercept))
        tmp = self.out / "series" / f".{sid}.tmp.npy"
        np.save(tmp, vol)
        tmp.rename(self.out / "series" / f"{sid}.npy")
        with self.lock, open(self.csv_path, "a") as f:
            for i, name in enumerate(names):
                f.write(f"{patient},{study},{sid},{i},{name},{','.join(map(str, labels[i]))}\n")
            self.kept += 1
            self.slices += len(names)
            self._written.add(sid)
        return True


def stream_region(
    url: str, start: int, budget: int, writer: Writer, max_series: int | None
) -> dict[str, int]:
    session = requests.Session()
    stream = ByteStream(url, start, session)
    current: str | None = None
    files: dict[str, bytes] = {}
    kept = 0
    # Past the budget we stop at the next series boundary, so the last series is complete.
    for path, data in iter_members(stream, limit=start + budget + (1 << 30)):
        parts = path.strip("/").split("/")
        if len(parts) != 5 or not parts[-1].endswith(".png"):
            continue
        sid = parts[3]
        if sid != current:
            if current is not None:
                kept += writer.write(current, files)
                if stream.pos > start + budget or (max_series and kept >= max_series):
                    current = None
                    break
            current, files = sid, {}
        files[parts[4]] = data
    if current is not None and writer.write(current, files):
        kept += 1
    return {"start": start, "end": stream.pos, "kept": kept}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/rsna"))
    ap.add_argument("--regions", type=int, default=8)
    ap.add_argument("--gib-per-region", type=float, default=1.6)
    ap.add_argument("--max-series-per-region", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tables = load_tables(args.out / "tables")
    url = f"{BASE}/images.tar"
    size = int(requests.head(url, allow_redirects=True, timeout=60).headers["Content-Length"])
    budget = int(args.gib_per_region * (1 << 30))
    starts = [(size * k // args.regions) // BLOCK * BLOCK for k in range(args.regions)]
    writer = Writer(args.out, tables)
    t0 = time.time()

    def report() -> None:
        while not stop.wait(30):
            log.info(
                "kept %d series / %d slices, dropped %d, %.0f s",
                writer.kept,
                writer.slices,
                writer.dropped,
                time.time() - t0,
            )

    stop = threading.Event()
    threading.Thread(target=report, daemon=True).start()
    with ThreadPoolExecutor(args.regions) as pool:
        futures = [
            pool.submit(stream_region, url, s, budget, writer, args.max_series_per_region)
            for s in starts
        ]
        regions = [f.result() for f in futures]
    stop.set()

    manifest = {
        "source": REPO,
        "revision": REVISION,
        "tar_bytes": size,
        "regions": regions,
        "series": writer.kept,
        "slices": writer.slices,
        "dropped_partial_series": writer.dropped,
        "image_size": IMAGE_SIZE,
        "seconds": round(time.time() - t0, 1),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("done: %s", json.dumps({k: v for k, v in manifest.items() if k != "regions"}))


if __name__ == "__main__":
    main()
