"""Stage 1: train the 2.5D slice CNN, then write its logits + embeddings for every split.

    uv run python -m brainct.training.train_cnn --config configs/cnn25d_resnet18.yaml
    uv run python -m brainct.training.train_cnn --config ... --set epochs=1 max_train_items=512

Resumes from <run>/last.pt if present. Model selection uses validation mean ROC AUC (the
balanced sampler shifts the class prior, so raw log loss is not comparable across epochs;
probabilities are recalibrated on validation afterwards).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from brainct.constants import LABELS
from brainct.metrics import roc_auc, sigmoid, weighted_log_loss
from brainct.training.common import (
    TRACKING_URI,
    amp_dtype,
    augment,
    cosine_lr,
    device,
    save_predictions,
    seed_everything,
    weighted_bce,
)
from brainct.training.config import TrainConfig
from brainct.training.data import Series, SliceDataset, balanced_epoch, load_series, read_volume
from brainct.training.model import SliceNet


class VolumeDataset(Dataset):
    def __init__(self, series: list[Series]):
        self.series = series

    def __len__(self) -> int:
        return len(self.series)

    def __getitem__(self, i: int) -> torch.Tensor:
        return torch.from_numpy(read_volume(self.series[i]).astype(np.float32))


def contexts(volume: torch.Tensor, context: int) -> torch.Tensor:
    """[T, H, W] -> [T, 2C+1, H, W], edges repeated (same rule as read_context)."""
    t = volume.shape[0]
    offsets = torch.arange(-context, context + 1, device=volume.device)
    idx = (torch.arange(t, device=volume.device)[:, None] + offsets).clamp(0, t - 1)
    return volume[idx]


@torch.no_grad()
def infer(model: SliceNet, series: list[Series], dev, chunk: int = 64):
    """Per-series logits and embeddings, no augmentation.

    In-process loading: a 4 MB series read takes ~2 ms, and spawning workers (macOS) cost more
    than it saved (39 vs 105 slices/s measured on 40 series).
    """
    model.eval()
    loader = DataLoader(VolumeDataset(series), batch_size=None, num_workers=0)
    logits, embs = [], []
    dtype = amp_dtype(dev)
    for vol in loader:
        x = contexts(vol.to(dev), model.context)
        out_l, out_e = [], []
        for i in range(0, len(x), chunk):
            with torch.autocast(dev.type, dtype=dtype, enabled=dtype is not None):
                lg, emb, _ = model.forward_all(x[i : i + chunk])
            out_l.append(lg.float().cpu())
            out_e.append(emb.float().cpu())
        logits.append(torch.cat(out_l).numpy())
        embs.append(torch.cat(out_e).numpy())
    return logits, embs


def evaluate(logits: list[np.ndarray], series: list[Series]) -> dict[str, float]:
    p = sigmoid(np.concatenate(logits))
    y = np.concatenate([s.labels for s in series])
    aucs = {f"auc_{k}": roc_auc(y[:, j], p[:, j]) for j, k in enumerate(LABELS)}
    return {
        **aucs,
        "auc_mean": float(np.nanmean(list(aucs.values()))),
        "wlogloss_uncalibrated": weighted_log_loss(y, p),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--skip-train", action="store_true", help="only (re)write predictions")
    args = ap.parse_args(argv)
    cfg = TrainConfig.load(args.config, args.set)
    run_dir = cfg.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    seed_everything(cfg.seed)
    dev = device()
    data_dir = Path(cfg.data_dir)

    train_series = load_series(data_dir, "train")
    val_series = load_series(data_dir, "val")[: cfg.max_val_series]
    train_ds = SliceDataset(train_series, cfg.context, cfg.context_dropout, cfg.seed)
    labels_any = train_ds.labels()[:, 0]

    model = SliceNet(cfg.backbone, cfg.context, pretrained=cfg.pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    dtype = amp_dtype(dev)
    scaler = torch.amp.GradScaler(dev.type, enabled=dtype == torch.float16)

    rng = np.random.default_rng(cfg.seed)
    per_epoch = len(balanced_epoch(labels_any, cfg.neg_ratio, rng))
    if cfg.max_train_items:
        per_epoch = min(per_epoch, cfg.max_train_items)
    steps_per_epoch = per_epoch // cfg.batch_size
    total = steps_per_epoch * cfg.epochs
    warmup = max(1, int(cfg.warmup_frac * total))

    start_epoch, step, best = 0, 0, -1.0
    last = run_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, step, best = ck["epoch"] + 1, ck["step"], ck["best"]
        rng = np.random.default_rng(cfg.seed + start_epoch)
        print(f"resumed from epoch {ck['epoch']}")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("brainct")
    run_id_file = run_dir / "mlflow_run_id"
    run_id = run_id_file.read_text().strip() if run_id_file.exists() else None
    with mlflow.start_run(run_id=run_id, run_name=cfg.run_name) as run:
        run_id_file.write_text(run.info.run_id)
        if run_id is None:
            mlflow.log_params(cfg.to_dict())
            mlflow.log_params(
                {
                    "train_series": len(train_series),
                    "train_slices": len(train_ds),
                    "val_series": len(val_series),
                    "device": dev.type,
                }
            )

        for epoch in range(start_epoch, cfg.epochs if not args.skip_train else 0):
            model.train()
            idx = balanced_epoch(labels_any, cfg.neg_ratio, rng)[:per_epoch]
            loader = DataLoader(
                Subset(train_ds, idx.tolist()),
                batch_size=cfg.batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=cfg.num_workers,
                persistent_workers=False,
                prefetch_factor=4 if cfg.num_workers else None,
            )
            t0, seen, running = time.time(), 0, 0.0
            for i, (x, y) in enumerate(loader):
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                x = augment(x)
                for g in opt.param_groups:
                    g["lr"] = cosine_lr(step, total, warmup, cfg.lr)
                with torch.autocast(dev.type, dtype=dtype, enabled=dtype is not None):
                    loss = weighted_bce(model(x), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
                step += 1
                seen += len(x)
                running = 0.98 * running + 0.02 * loss.item() if i else loss.item()
                if i % 50 == 0:
                    rate = seen / (time.time() - t0)
                    print(
                        f"epoch {epoch} step {i}/{steps_per_epoch} loss {running:.4f} "
                        f"{rate:.0f} img/s",
                        flush=True,
                    )
                    mlflow.log_metric("train_loss", running, step=step)
            train_time = time.time() - t0

            logits, _ = infer(model, val_series, dev)
            metrics = evaluate(logits, val_series)
            metrics |= {"epoch_seconds": train_time, "img_per_s": seen / train_time}
            mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()}, step=epoch)
            print(f"epoch {epoch}: " + json.dumps({k: round(v, 4) for k, v in metrics.items()}))
            if metrics["auc_mean"] > best:
                best = metrics["auc_mean"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": cfg.to_dict(),
                        "epoch": epoch,
                        "val": metrics,
                    },
                    run_dir / "best.pt",
                )
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "step": step,
                    "best": best,
                },
                last,
            )
            rng = np.random.default_rng(cfg.seed + epoch + 1)

        # Predictions of the selected checkpoint for every split (stage 2 + evaluation input).
        ck = torch.load(run_dir / "best.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        mlflow.log_metric("best_epoch", ck["epoch"])
        for split in ("train", "val", "test"):
            series = load_series(data_dir, split)
            if split == "val" or (split == "test" and cfg.max_val_series):
                series = series[: cfg.max_val_series]
            elif split == "train" and cfg.max_train_items:
                series = series[: max(4, cfg.max_train_items // 32)]
            t0 = time.time()
            logits, embs = infer(model, series, dev)
            save_predictions(run_dir / "predictions" / f"cnn_{split}.npz", series, logits, embs)
            print(f"wrote {split} predictions ({len(series)} series, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
