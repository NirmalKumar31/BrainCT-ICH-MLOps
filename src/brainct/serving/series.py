"""Uploaded DICOM files (or a zip of them) -> one ordered, resized HU volume."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field

import numpy as np

from brainct.preprocessing import DicomError, SliceInfo, read_dicom, resize_hu


class UploadError(ValueError):
    """Client error with an HTTP status."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


@dataclass
class Volume:
    hu: np.ndarray  # [T, S, S] int16, ordered inferior -> superior when geometry allows
    infos: list[SliceInfo]
    order: str  # "position", "instance_number" or "upload"
    warnings: list[str] = field(default_factory=list)


def unpack_zip(data: bytes, max_files: int, max_total: int) -> list[bytes]:
    """Members of a zip, with limits on count and *uncompressed* size (zip bombs)."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise UploadError(422, "not a valid zip archive") from exc
    members = [m for m in zf.infolist() if not m.is_dir() and not m.filename.startswith("__MACOSX")]
    if len(members) > max_files:
        raise UploadError(413, f"zip holds {len(members)} files; limit is {max_files}")
    if sum(m.file_size for m in members) > max_total:
        raise UploadError(413, "zip contents exceed the upload size limit")
    out = []
    for m in sorted(members, key=lambda m: m.filename):
        with zf.open(m) as f:
            blob = f.read(max_total + 1)
        if len(blob) != m.file_size:  # header lied about the size
            raise UploadError(413, "zip member larger than declared")
        out.append(blob)
    return out


def decode_one(blob: bytes, image_size: int) -> tuple[np.ndarray, SliceInfo]:
    try:
        hu, info = read_dicom(blob)
    except DicomError as exc:
        raise UploadError(422, str(exc)) from exc
    if info.modality != "CT":
        raise UploadError(422, f"expected CT, got Modality={info.modality or 'missing'}")
    return resize_hu(hu, image_size), info


def assemble(slices: list[tuple[np.ndarray, SliceInfo]]) -> Volume:
    """Order slices and check they form one series."""
    if not slices:
        raise UploadError(400, "no slices uploaded")
    infos = [s[1] for s in slices]
    warnings = []
    uids = {i.series_uid for i in infos if i.series_uid}
    if len(uids) > 1:
        raise UploadError(422, f"upload mixes {len(uids)} series; send one series per request")

    zs = [i.z for i in infos]
    numbers = [i.instance_number for i in infos]
    if all(z is not None for z in zs):
        order = np.argsort(np.asarray(zs, dtype=np.float64), kind="stable")
        how = "position"
        if len({round(z, 3) for z in zs if z is not None}) < len(zs):
            warnings.append("duplicate slice positions; context between them is unreliable")
    elif all(n is not None for n in numbers):
        order = np.argsort(np.asarray(numbers), kind="stable")
        how = "instance_number"
        warnings.append("ImagePositionPatient missing; ordered by InstanceNumber")
    else:
        order = np.arange(len(slices))
        how = "upload"
        warnings.append("no slice geometry; kept upload order")
    if len(slices) < 8:
        warnings.append(
            f"only {len(slices)} slices: the series model is trained on whole scans (24-60 slices)"
        )
    if any(i.sign_fixed for i in infos):
        warnings.append("corrected 12-bit sign encoding on some slices")
    hu = np.stack([slices[k][0] for k in order])
    return Volume(hu, [infos[k] for k in order], how, warnings)
