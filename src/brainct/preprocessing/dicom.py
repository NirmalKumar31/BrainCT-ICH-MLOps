"""DICOM bytes -> HU slice + the geometry needed to order a series."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

from brainct.preprocessing.transforms import hu_from_stored


class DicomError(ValueError):
    """Upload is not a usable single-frame CT slice."""


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
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    bits = int(getattr(ds, "BitsStored", 16))
    if bits == 12 and int(getattr(ds, "PixelRepresentation", 0)) == 0 and intercept > -100:
        x = pixels.astype(np.int32) + 1000
        x[x >= 4096] -= 4096
        return x, -1000.0, True
    return pixels, intercept, False


def _z_position(ds: pydicom.Dataset) -> float | None:
    pos = getattr(ds, "ImagePositionPatient", None)
    ori = getattr(ds, "ImageOrientationPatient", None)
    if pos is None or len(pos) != 3:
        return None
    if ori is not None and len(ori) == 6:
        normal = np.cross(np.asarray(ori[:3], float), np.asarray(ori[3:], float))
        return float(np.dot(normal, np.asarray(pos, float)))
    return float(pos[2])


def read_dicom(data: bytes) -> tuple[np.ndarray, SliceInfo]:
    """Decode one slice. Raises DicomError for anything that is not a 2D single-frame image."""
    try:
        ds = pydicom.dcmread(io.BytesIO(data), force=False)
        pixels = ds.pixel_array
    except (InvalidDicomError, AttributeError, ValueError, TypeError, KeyError, OSError) as exc:
        raise DicomError(f"not a readable DICOM image: {exc}") from exc
    except Exception as exc:  # pixel codecs raise a zoo of types for corrupt data
        raise DicomError(f"could not decode pixel data: {type(exc).__name__}") from exc
    if pixels.ndim != 2:
        raise DicomError(f"expected a single 2D frame, got pixel array shape {pixels.shape}")

    pixels, intercept, fixed = fix_12bit(pixels, ds)
    hu = hu_from_stored(pixels, float(getattr(ds, "RescaleSlope", 1.0)), intercept)
    number = getattr(ds, "InstanceNumber", None)
    info = SliceInfo(
        modality=str(getattr(ds, "Modality", "")),
        rows=int(pixels.shape[0]),
        columns=int(pixels.shape[1]),
        series_uid=str(ds.SeriesInstanceUID) if "SeriesInstanceUID" in ds else None,
        instance_number=int(number) if number not in (None, "") else None,
        z=_z_position(ds),
        sign_fixed=fixed,
    )
    return hu, info
