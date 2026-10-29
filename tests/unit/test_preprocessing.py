from __future__ import annotations

import numpy as np
import pytest
from dicom_factory import make_ct, phantom

from brainct.constants import HU_MAX, HU_MIN, IMAGE_SIZE, IN_CHANNELS, WINDOWS
from brainct.preprocessing import (
    DicomError,
    hu_from_stored,
    model_input,
    read_dicom,
    resize_hu,
    stack_context,
    window,
)


def test_hu_from_stored_applies_modality_lut():
    px = np.array([[0, 1024, 2048]], dtype=np.uint16)
    np.testing.assert_allclose(hu_from_stored(px, 1.0, -1024.0), [[-1024, 0, 1024]])
    np.testing.assert_allclose(hu_from_stored(px, 0.5, -10.0), [[-10, 502, 1014]])


def test_resize_hu_shape_dtype_and_clip():
    hu = np.full((512, 512), 5000.0)
    hu[:256] = -3000.0
    out = resize_hu(hu)
    assert out.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert out.dtype == np.int16
    assert out.min() == HU_MIN and out.max() == HU_MAX


def test_resize_hu_pads_non_square_with_air_and_keeps_aspect():
    hu = np.full((256, 512), 40.0)  # wide image: padded top and bottom
    out = resize_hu(hu, 128)
    assert out[0, 64] == HU_MIN and out[-1, 64] == HU_MIN
    assert out[64, 0] == 40 and out[64, -1] == 40
    assert out.shape == (128, 128)


def test_resize_hu_area_preserves_mean_of_small_structures():
    hu = np.zeros((512, 512))
    hu[100:108, 100:108] = 80.0  # 8x8 px "bleed" -> ~3.5 px at 224
    out = resize_hu(hu).astype(float)
    assert out.sum() * (512 / 224) ** 2 == pytest.approx(hu.sum(), rel=0.05)


def test_resize_rejects_bad_shapes():
    with pytest.raises(ValueError, match="2D"):
        resize_hu(np.zeros((4, 4)))
    with pytest.raises(ValueError, match="2D"):
        resize_hu(np.zeros((3, 64, 64)))


def test_window_ranges():
    hu = np.array([[-1000.0, 0.0, 40.0, 80.0, 2000.0]])
    w = window(hu)
    assert w.shape == (len(WINDOWS), 1, 5)
    np.testing.assert_allclose(w[0, 0], [0, 0, 0.5, 1, 1])  # brain 40/80
    assert w.min() >= 0 and w.max() <= 1


def test_stack_context_repeats_edges():
    vol = np.arange(5)[:, None, None] * np.ones((1, 2, 2))
    assert stack_context(vol, 0)[:, 0, 0].tolist() == [0, 0, 1]
    assert stack_context(vol, 2)[:, 0, 0].tolist() == [1, 2, 3]
    assert stack_context(vol, 4)[:, 0, 0].tolist() == [3, 4, 4]


def test_model_input_is_slice_major():
    ctx = np.stack([np.full((4, 4), v, np.int16) for v in (0, 40, 80)])[None]
    x = model_input(ctx)
    assert x.shape == (1, IN_CHANNELS, 4, 4)
    # brain window of slices i-1, i, i+1 sit at channels 0, 3, 6
    np.testing.assert_allclose(x[0, [0, 3, 6], 0, 0], [0.0, 0.5, 1.0])


def test_read_dicom_roundtrip_hu():
    hu = phantom(128)
    out, info = read_dicom(make_ct(hu, z=12.5, instance=7))
    np.testing.assert_allclose(out, np.rint(hu + 1024) - 1024, atol=0.5)
    assert info.modality == "CT" and info.instance_number == 7
    assert info.z == pytest.approx(12.5)
    assert not info.sign_fixed


def test_read_dicom_rescale_slope():
    hu = phantom(64)
    out, _ = read_dicom(make_ct(hu, slope=0.5, intercept=-1000.0))
    np.testing.assert_allclose(out, hu, atol=0.3)


def test_read_dicom_fixes_rsna_12bit_sign_bug():
    hu = phantom(128)
    out, info = read_dicom(make_ct(hu, buggy_12bit=True))
    assert info.sign_fixed
    inside = hu >= -1000  # below -1000 HU is not representable in the buggy encoding
    np.testing.assert_allclose(out[inside], np.rint(hu[inside]), atol=0.5)


def test_read_dicom_z_follows_slice_normal():
    import io

    from pydicom import dcmread

    raw = make_ct(phantom(32), z=3.0)
    ds = dcmread(io.BytesIO(raw))
    ds.ImageOrientationPatient = [1, 0, 0, 0, 0, -1]  # coronal: normal is +y
    ds.ImagePositionPatient = [0.0, 42.0, 3.0]
    buf = io.BytesIO()
    ds.save_as(buf)
    _, info = read_dicom(buf.getvalue())
    assert info.z == pytest.approx(42.0)


@pytest.mark.parametrize("blob", [b"", b"not a dicom", b"\0" * 200])
def test_read_dicom_rejects_garbage(blob):
    with pytest.raises(DicomError):
        read_dicom(blob)


def test_png_training_path_equals_dicom_serving_path():
    """Train/serve skew guard: the cache is built from 16-bit PNG + rescale values, the service
    decodes DICOM. Same stored pixels must give byte-identical model input."""
    hu = phantom(512, seed=3, bleed=True)
    stored = np.rint(hu + 1024).astype(np.uint16)  # what a 16-bit PNG would hold
    train = resize_hu(hu_from_stored(stored, 1.0, -1024.0))
    serve_hu, _ = read_dicom(make_ct(hu, intercept=-1024.0))
    np.testing.assert_array_equal(train, resize_hu(serve_hu))
