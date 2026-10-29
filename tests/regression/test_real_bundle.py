"""Release checks for the trained bundle in artifacts/bundle. Skipped when it is absent (CI).

Run before building a release image:  uv run pytest tests/regression -v
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import REAL_BUNDLE
from dicom_factory import make_ct, phantom

from brainct.constants import LABELS
from brainct.preprocessing import read_dicom, resize_hu

pytestmark = pytest.mark.skipif(
    not (REAL_BUNDLE / "model_card.json").exists(), reason="no trained bundle"
)

# Regression floors for the deployed chain, set a margin below the certified v0.2.0 bundle
# (macro AUC 0.920, study AUC 0.948, weighted log loss 0.123). A retrain that falls below
# these does not ship.
GATES = {"auc_macro": 0.90, "study_any_auc": 0.93, "weighted_log_loss_max": 0.135}


@pytest.fixture(scope="module")
def model():
    from brainct.serving.model import InferenceModel

    return InferenceModel(REAL_BUNDLE)


def test_card_contract(model):
    card = model.card
    assert card["contract"]["labels"] == list(LABELS)
    assert set(card["calibration"]) == {"cnn", "seq"}
    assert {"screening_any", "study_any"} <= set(card["thresholds"]["seq"])
    assert card["provenance"]["onnx_parity_max_abs"] < 1e-3


def test_quality_gates(model):
    m = model.card["metrics_test"]["int8"]["seq"]
    assert m["auc_macro"] >= GATES["auc_macro"]
    assert m["study_any_auc"] >= GATES["study_any_auc"]
    assert m["weighted_log_loss"] <= GATES["weighted_log_loss_max"]


def test_int8_close_to_fp32(model):
    fp32 = model.card["metrics_test"]["fp32"]["seq"]
    int8 = model.card["metrics_test"]["int8"]["seq"]
    assert abs(fp32["auc_macro"] - int8["auc_macro"]) < 0.01


def _volume(bleed_slices=()) -> np.ndarray:
    slices = []
    for i in range(24):
        hu, _ = read_dicom(make_ct(phantom(512, seed=i, bleed=i in bleed_slices), z=5.0 * i))
        slices.append(resize_hu(hu))
    return np.stack(slices)


def test_deterministic_and_batch_size_independent(model):
    vol = _volume()
    a = model.run(vol)
    b = model.run(vol)
    np.testing.assert_array_equal(a.seq_logits, b.seq_logits)
    model.batch, saved = 5, model.batch
    try:
        c = model.run(vol)
    finally:
        model.batch = saved
    np.testing.assert_allclose(a.cnn_logits, c.cnn_logits, atol=1e-4)


def test_synthetic_bleed_raises_any(model):
    """Sanity, not validation: a bright lesion inside the phantom brain must raise p(any)."""
    clean = model.calibrate(model.run(_volume()).seq_logits, "seq")[:, 0]
    bleed = model.calibrate(model.run(_volume(bleed_slices=range(8, 14))).seq_logits, "seq")[:, 0]
    assert bleed[8:14].mean() > clean[8:14].mean()
