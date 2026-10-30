"""Stage 2: series model over stage-1 slice embeddings.

    uv run python -m brainct.training.train_seq --config configs/cnn25d_resnet18.yaml

Each series is a sequence of [512-d embedding, 6 CNN logits] per slice; a 2-layer BiGRU refines
every slice's logits with whole-scan context. Trained on CPU in a few minutes.

Caveat (see docs/decisions.md): training embeddings come from the CNN that was trained on those
slices, so they are cleaner than val/test embeddings. Gaussian noise on the input logits and
dropout on the embeddings narrow that gap; the effect is measured on held-out test only.
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow

from brainct.metrics import sigmoid, weighted_log_loss
from brainct.training.common import (
    TRACKING_URI,
    load_predictions,
    save_atomic,
    seed_everything,
    weighted_bce,
)
from brainct.training.config import TrainConfig
from brainct.training.model import SeqNet


def to_sequences(pred: dict[str, np.ndarray]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    off = pred["offsets"]
    return [
        (pred["emb"][a:b].astype(np.float32), pred["logits"][a:b], pred["labels"][a:b])
        for a, b in pairwise(off)
    ]


def batches(seqs, size: int, rng: np.random.Generator | None):
    order = rng.permutation(len(seqs)) if rng is not None else np.arange(len(seqs))
    for i in range(0, len(order), size):
        chunk = [seqs[j] for j in order[i : i + size]]
        lengths = torch.tensor([len(c[0]) for c in chunk])
        t = int(lengths.max())

        def pad(k: int, chunk=chunk, t=t) -> torch.Tensor:
            arr = np.zeros((len(chunk), t, chunk[0][k].shape[1]), dtype=np.float32)
            for n, c in enumerate(chunk):
                arr[n, : len(c[k])] = c[k]
            return torch.from_numpy(arr)

        mask = torch.arange(t)[None, :] < lengths[:, None]
        yield pad(0), pad(1), pad(2), lengths, mask


@torch.no_grad()
def predict(model: SeqNet, seqs) -> list[np.ndarray]:
    model.eval()
    return [
        model(torch.from_numpy(e)[None], torch.from_numpy(lg)[None])[0].numpy() for e, lg, _ in seqs
    ]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    cfg = TrainConfig.load(args.config, args.set)
    seed_everything(cfg.seed)
    torch.set_num_threads(4)
    run_dir = cfg.run_dir()
    pred_dir = run_dir / "predictions"
    preds = {s: load_predictions(pred_dir / f"cnn_{s}.npz") for s in ("train", "val", "test")}
    train, val = to_sequences(preds["train"]), to_sequences(preds["val"])
    embed_dim = train[0][0].shape[1]

    model = SeqNet(embed_dim, hidden=cfg.seq_hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.seq_lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.seq_epochs)
    rng = np.random.default_rng(cfg.seed)
    y_val = np.concatenate([s[2] for s in val])
    best, best_state, patience = float("inf"), None, 0

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("brainct")
    with mlflow.start_run(run_name=f"{cfg.run_name}_seq"):
        mlflow.log_params({k: v for k, v in cfg.to_dict().items() if k.startswith("seq_")})
        mlflow.log_param("stage1_run", cfg.run_name)
        for epoch in range(cfg.seq_epochs):
            model.train()
            total = 0.0
            for emb, lg, y, lengths, mask in batches(train, 16, rng):
                emb = torch.nn.functional.dropout(emb, 0.2)
                lg = lg + cfg.seq_feature_noise * torch.randn_like(lg)
                out = model(emb, lg, lengths)
                loss = weighted_bce(out[mask], y[mask])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += loss.item()
            sched.step()
            p_val = sigmoid(np.concatenate(predict(model, val)))
            wll = weighted_log_loss(y_val, p_val)
            mlflow.log_metrics({"seq_train_loss": total, "seq_val_wlogloss": wll}, step=epoch)
            print(f"epoch {epoch}: train {total:.3f} val wlogloss {wll:.4f}", flush=True)
            if wll < best - 1e-4:
                best, patience = wll, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 8:
                    break
        assert best_state is not None
        model.load_state_dict(best_state)
        mlflow.log_metric("seq_best_val_wlogloss", best)
        save_atomic(
            {"model": best_state, "embed_dim": embed_dim, "hidden": cfg.seq_hidden},
            run_dir / "seq.pt",
        )
        for split, pred in preds.items():
            logits = predict(model, to_sequences(pred))
            out = {k: v for k, v in pred.items() if k != "emb"}
            out["logits"] = np.concatenate(logits).astype(np.float32)
            np.savez(pred_dir / f"seq_{split}.npz", **out)
        print(json.dumps({"best_val_wlogloss": best}))


if __name__ == "__main__":
    main()
