"""DICOM bytes -> HU slice + the geometry needed to order a series."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

from brainct.preprocessing.transforms import hu_from_stored

# Checked before the pixel data is decoded: a compressed file of a few hundred KB can declare
# a 65535 x 65535 image. Head CT is 512 x 512; 1024 leaves room for high-resolution scanners.
MAX_DIM = 1024
MIN_DIM = 64


class DicomError(ValueError):
    """Upload is not a usable single-frame CT slice."""


def _number(ds: pydicom.Dataset, name: str, default: float | None) -> float | None:
    """A numeric tag; absent or empty -> default; malformed -> DicomError."""
    value = getattr(ds, name, None)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise DicomError(f"malformed {name}") from exc


@dataclass(frozen=True)
class SliceInfo:
    modality: str
    rows: int
    columns: int
    series_uid: str | None
    instance_number: int | None
    # Position along the slice normal (mm). Orders a series robustly even when
    # InstanceNumber is missing or counts the wrong way.
    z: float | None
    sign_fixed: bool


def fix_12bit(pixels: np.ndarray, ds: pydicom.Dataset) -> tuple[np.ndarray, float, bool]:
    """Known RSNA defect: 12-bit data tagged unsigned with an intercept near 0.

    Those slices decode with negative values wrapped to 2048..4095. Detected the same way as the
    widely used competition fix; returns (pixels, intercept, fixed?).
    """
    intercept = _number(ds, "RescaleIntercept", 0.0) or 0.0
    bits = int(_number(ds, "BitsStored", 16) or 16)
    signed = int(_number(ds, "PixelRepresentation", 0) or 0)
    if bits == 12 and signed == 0 and intercept > -100:
        x = pixels.astype(np.int32) + 1000
        x[x >= 4096] -= 4096
        return x, -1000.0, True
    return pixels, intercept, False


def _z_position(ds: pydicom.Dataset) -> float | None:
    pos = getattr(ds, "ImagePositionPatient", None)
    ori = getattr(ds, "ImageOrientationPatient", None)
    try:
        if pos is None or len(pos) != 3:
            return None
        if ori is not None and len(ori) == 6:
            normal = np.cross(np.asarray(ori[:3], float), np.asarray(ori[3:], float))
            z = float(np.dot(normal, np.asarray(pos, float)))
        else:
            z = float(pos[2])
    except (TypeError, ValueError):
        return None  # malformed geometry: the series falls back to InstanceNumber
    return z if np.isfinite(z) else None


def read_dicom(data: bytes) -> tuple[np.ndarray, SliceInfo]:
    """Decode one slice. Raises DicomError for anything that is not a 2D single-frame image."""
    try:
        ds = pydicom.dcmread(io.BytesIO(data), force=False)
    except (InvalidDicomError, AttributeError, ValueError, TypeError, KeyError, OSError) as exc:
        raise DicomError("not a readable DICOM file") from exc
    except Exception as exc:  # malformed files raise a zoo of types
        raise DicomError("not a readable DICOM file") from exc
    if "PixelData" not in ds:
        raise DicomError("DICOM file has no image (PixelData)")
    rows, cols = _number(ds, "Rows", None), _number(ds, "Columns", None)
    if rows is None or cols is None:
        raise DicomError("DICOM image without Rows/Columns")
    if not (MIN_DIM <= rows <= MAX_DIM and MIN_DIM <= cols <= MAX_DIM):
        raise DicomError(f"image is {rows:.0f}x{cols:.0f}; supported {MIN_DIM}-{MAX_DIM} px")
    if (_number(ds, "NumberOfFrames", 1) or 1) != 1 or (
        _number(ds, "SamplesPerPixel", 1) or 1
    ) != 1:
        raise DicomError("expected a single-frame grayscale image")
    try:
        pixels = ds.pixel_array
    except Exception as exc:  # pixel codecs raise a zoo of types for corrupt data
        raise DicomError(f"could not decode pixel data ({type(exc).__name__})") from exc
    if pixels.ndim != 2:
        raise DicomError(f"expected a single 2D frame, got pixel array shape {pixels.shape}")

    pixels, intercept, fixed = fix_12bit(pixels, ds)
    slope = _number(ds, "RescaleSlope", 1.0) or 1.0
    hu = hu_from_stored(pixels, slope, intercept)
    number = _number(ds, "InstanceNumber", None)
    info = SliceInfo(
        modality=str(getattr(ds, "Modality", "")),
        rows=int(pixels.shape[0]),
        columns=int(pixels.shape[1]),
        series_uid=str(ds.SeriesInstanceUID) if ds.get("SeriesInstanceUID") else None,
        instance_number=int(number) if number is not None else None,
        z=_z_position(ds),
        sign_fixed=fixed,
    )
    return hu, info
