"""Synthetic CT DICOMs for tests (no patient data anywhere in the test suite)."""

from __future__ import annotations

import io

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid


def phantom(size: int = 512, seed: int = 0, bleed: bool = False) -> np.ndarray:
    """Head-like HU image: air, a skull ring (~1000 HU), brain (~35 HU), optional bleed (~70)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(yy - size / 2, xx - size / 2) / (size / 2)
    hu = np.full((size, size), -1000.0)
    hu[r < 0.80] = 1000.0
    hu[r < 0.72] = 35.0
    if bleed:
        hu[(np.hypot(yy - size * 0.4, xx - size * 0.6) < size * 0.06)] = 70.0
    return hu + rng.normal(0, 4, hu.shape) * (r < 0.8)


def make_ct(
    hu: np.ndarray,
    *,
    slope: float = 1.0,
    intercept: float = -1024.0,
    z: float | None = 0.0,
    instance: int | None = 1,
    series_uid: str = "1.2.826.0.1.3680043.8.498.1",
    modality: str = "CT",
    buggy_12bit: bool = False,
) -> bytes:
    """Encode HU as a CT slice. `buggy_12bit` writes the RSNA defect (signed 12-bit data tagged
    unsigned, intercept 0) that read_dicom must undo."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = modality
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = "1.2.826.0.1.3680043.8.498.2"
    ds.PatientName = "SYNTHETIC^PHANTOM"
    if instance is not None:
        ds.InstanceNumber = instance
    if z is not None:
        ds.ImagePositionPatient = [-125.0, -125.0, float(z)]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.Rows, ds.Columns = hu.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    if buggy_12bit:
        # The defect: HU stored directly (intercept 0) as 12-bit two's complement but tagged
        # unsigned, so HU in [-1000, 0) appears as 3096..4095. Representable HU: [-1000, 3096).
        stored = np.rint(np.clip(hu, -1000, 3095)).astype(np.int32)
        stored[stored < 0] += 4096
        ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 12, 11, 0
        ds.RescaleSlope, ds.RescaleIntercept = 1, 0
        pixels = stored.astype(np.uint16)
    else:
        stored = np.rint((hu - intercept) / slope)
        ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 15, 1
        ds.RescaleSlope, ds.RescaleIntercept = slope, intercept
        pixels = stored.astype(np.int16)
    ds.PixelData = pixels.tobytes()
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def make_series(n: int = 12, size: int = 256, bleed_at: tuple[int, ...] = (), **kw) -> list[bytes]:
    """n slices, 5 mm apart, returned in *shuffled* order to exercise sorting."""
    order = np.random.default_rng(1).permutation(n)
    return [
        make_ct(
            phantom(size, seed=int(i), bleed=int(i) in bleed_at),
            z=5.0 * int(i),
            instance=int(i) + 1,
            **kw,
        )
        for i in order
    ]
