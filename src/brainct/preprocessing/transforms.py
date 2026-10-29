"""Pure numpy transforms. The training cache stores the output of `resize_hu`; the service runs
`resize_hu` on each uploaded slice. Everything after that (`window`, `stack_context`,
`model_input`) runs per batch in both places, so train/serve skew is limited to the decoder."""

from __future__ import annotations

import cv2
import numpy as np

from brainct.constants import CONTEXT, HU_MAX, HU_MIN, IMAGE_SIZE, WINDOWS


def hu_from_stored(pixels: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Stored pixel values -> HU (float32) with the DICOM modality LUT."""
    return pixels.astype(np.float32) * np.float32(slope) + np.float32(intercept)


def resize_hu(hu: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Pad to a square (with air), resize with area averaging, clip, round to int16.

    Area interpolation keeps a small bleed's mean intensity instead of aliasing it away, and
    padding keeps the anatomy's aspect ratio for non-square scans.
    """
    if hu.ndim != 2 or min(hu.shape) < 16:
        raise ValueError(f"expected a 2D slice of at least 16x16, got shape {hu.shape}")
    h, w = hu.shape
    side = max(h, w)
    if h != w:
        padded = np.full((side, side), HU_MIN, dtype=np.float32)
        top, left = (side - h) // 2, (side - w) // 2
        padded[top : top + h, left : left + w] = hu
        hu = padded
    hu = np.clip(hu.astype(np.float32, copy=False), HU_MIN, HU_MAX)
    if side != size:
        interp = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
        hu = cv2.resize(hu, (size, size), interpolation=interp)
    return np.rint(hu).astype(np.int16)


def window(hu: np.ndarray) -> np.ndarray:
    """[..., H, W] HU -> [..., len(WINDOWS), H, W] float32 in [0, 1]."""
    hu = hu.astype(np.float32, copy=False)
    out = [np.clip((hu - (c - w / 2)) / w, 0.0, 1.0) for c, w in WINDOWS]
    return np.stack(out, axis=-3)


def stack_context(volume: np.ndarray, index: int, context: int = CONTEXT) -> np.ndarray:
    """Slices index-context..index+context of a [T, H, W] series, edges repeated."""
    t = volume.shape[0]
    idx = np.clip(np.arange(index - context, index + context + 1), 0, t - 1)
    return volume[idx]


def model_input(context_hu: np.ndarray) -> np.ndarray:
    """[N, 2C+1, H, W] int16 HU -> [N, 3*(2C+1), H, W] float32 network input.

    Channel order is slice-major: (slice i-1: brain, subdural, bone), (slice i: ...), ...
    """
    n, k, h, w = context_hu.shape
    return window(context_hu).reshape(n, k * len(WINDOWS), h, w)
