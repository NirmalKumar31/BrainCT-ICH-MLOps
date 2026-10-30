"""Training-side parity checks. Skipped where PyTorch is not installed (the CI test job)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from brainct.constants import HU_MIN, IMAGE_SIZE, LABELS  # noqa: E402
from brainct.preprocessing import model_input  # noqa: E402
from brainct.serving.model import contexts as np_contexts  # noqa: E402
from brainct.training.common import augment, cosine_lr, weighted_bce  # noqa: E402
from brainct.training.data import Series, SliceDataset, balanced_epoch, read_context  # noqa: E402
from brainct.training.model import HUWindowing, SeqNet  # noqa: E402
from brainct.training.train_cnn import contexts as torch_contexts  # noqa: E402


def test_in_graph_windowing_matches_numpy_reference():
    hu = np.random.default_rng(0).uniform(-1100, 3100, (2, 3, 16, 16)).astype(np.float32)
    ours = HUWindowing()(torch.from_numpy(hu)).numpy()
    ref = (model_input(hu) - 0.5) / 0.25
    np.testing.assert_allclose(ours, ref, atol=1e-5)


def test_torch_and_numpy_context_stacking_agree():
    vol = np.random.default_rng(1).integers(-1000, 1000, (7, 8, 8)).astype(np.float32)
    for c in (0, 1, 2):
        np.testing.assert_array_equal(
            torch_contexts(torch.from_numpy(vol), c).numpy(), np_contexts(vol, c)
        )


def test_augment_fills_with_air_and_keeps_context_aligned():
    x = torch.full((4, 3, 64, 64), 40.0)
    out = augment(x, torch.Generator().manual_seed(0))
    assert out.shape == x.shape
    assert out.min() >= HU_MIN - 1e-3 and out.max() <= 40 + 1e-3
    assert torch.equal(out[:, 0], out[:, 1]) and torch.equal(out[:, 1], out[:, 2])


def test_seqnet_starts_as_identity_on_cnn_logits():
    net = SeqNet(16).eval()
    emb, logits = torch.randn(2, 9, 16), torch.randn(2, 9, len(LABELS))
    torch.testing.assert_close(net(emb, logits), logits)
    lengths = torch.tensor([9, 5])
    assert net(emb, logits, lengths).shape == logits.shape


def test_weighted_bce_counts_any_twice():
    logits = torch.zeros(1, 6)
    y_any = torch.tensor([[1.0, 0, 0, 0, 0, 0]])
    y_sub = torch.tensor([[0.0, 1, 0, 0, 0, 0]])
    assert weighted_bce(logits, y_any) == pytest.approx(weighted_bce(logits, y_sub).item())
    base = torch.nn.functional.binary_cross_entropy_with_logits(torch.zeros(1), torch.ones(1))
    assert weighted_bce(logits, y_any).item() == pytest.approx(base.item())


def test_cosine_schedule():
    assert cosine_lr(0, 100, 10, 1.0) == pytest.approx(0.1)
    assert cosine_lr(10, 100, 10, 1.0) == pytest.approx(1.0)
    assert cosine_lr(100, 100, 10, 1.0) == pytest.approx(0.0, abs=1e-9)


@pytest.fixture
def cached_series(tmp_path):
    vol = np.stack([np.full((IMAGE_SIZE, IMAGE_SIZE), v, np.int16) for v in range(5)])
    path = tmp_path / "S.npy"
    np.save(path, vol)
    from brainct.training.data import _data_offset

    labels = np.zeros((5, 6), np.uint8)
    labels[2] = [1, 0, 0, 0, 0, 1]
    return Series("S", "P", path, _data_offset(path), labels)


def test_read_context_seeks_correct_slices(cached_series):
    assert read_context(cached_series, 0, 1)[:, 0, 0].tolist() == [0, 0, 1]
    assert read_context(cached_series, 4, 1)[:, 0, 0].tolist() == [3, 4, 4]
    assert read_context(cached_series, 2, 2)[:, 0, 0].tolist() == [0, 1, 2, 3, 4]


def test_dataset_and_context_dropout(cached_series):
    ds = SliceDataset([cached_series], context=1, context_dropout=1.0)
    x, y = ds[2]
    assert x[:, 0, 0].tolist() == [2, 2, 2]  # neighbours replaced by the centre slice
    assert y.tolist() == [1, 0, 0, 0, 0, 1]
    assert len(ds) == 5


def test_balanced_epoch():
    labels = np.array([1, 0, 0, 0, 0, 1, 0, 0])
    idx = balanced_epoch(labels, 1.5, np.random.default_rng(0))
    assert sorted(set(idx) & {0, 5}) == [0, 5]
    assert len(idx) == 5 and len(set(idx)) == 5


def test_config_overrides_are_typed():
    from brainct.training.config import TrainConfig

    cfg = TrainConfig.load(None, ["lr=1e-3", "seq_feature_noise=5e-1", "epochs=3"])
    assert cfg.lr == 1e-3 and isinstance(cfg.lr, float)
    assert cfg.seq_feature_noise == 0.5 and cfg.epochs == 3
    with pytest.raises(ValueError, match="lr"):
        TrainConfig.load(None, ["lr=fast"])


def test_save_atomic_leaves_no_partial_file(tmp_path):
    from brainct.training.common import save_atomic

    save_atomic({"a": torch.zeros(3)}, tmp_path / "last.pt")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["last.pt"]
