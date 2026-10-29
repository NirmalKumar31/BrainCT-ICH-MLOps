"""Stage 1: train the 2.5D slice CNN, then write its logits + embeddings for every split.

    uv run python -m brainct.training.train_cnn --config configs/cnn25d_resnet18.yaml
    uv run python -m brainct.training.train_cnn --config ... --set epochs=1 max_train_items=512

Multi-GPU: `torchrun --nproc_per_node N -m brainct.training.train_cnn ...` (DDP; NCCL on CUDA,
gloo on CPU). Every rank draws the same balanced epoch and trains on its shard; rank 0 alone
validates, checkpoints, logs to MLflow and writes predictions.

Resumes from <run>/last.pt if present. Model selection uses validation mean ROC AUC (the
balanced sampler shifts the class prior, so raw log loss is not comparable across epochs;
probabilities are recalibrated on validation afterwards).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
def infer(model: SliceNet, series: list[Series], dev, chunk: int | None = None):
    """Per-series logits and embeddings, no augmentation.

    Slices from consecutive series are streamed through fixed-size batches (the last one
    padded): one input shape keeps cuDNN autotuning to a single run (a new shape per series
    length cost ~10x on an A100) and keeps the GPU full. Volumes are prefetched by workers on
    CUDA; on macOS spawning workers cost more than it saved (39 vs 105 slices/s).
    """
    model.eval()
    cuda = dev.type == "cuda"
    chunk = chunk or (256 if cuda else 64)
    fmt = torch.channels_last if cuda else torch.contiguous_format
    loader = DataLoader(
        VolumeDataset(series), batch_size=None, num_workers=6 if cuda else 0, pin_memory=cuda
    )
    dtype = amp_dtype(dev)
    out_l: list[torch.Tensor] = []
    out_e: list[torch.Tensor] = []

    def run(x: torch.Tensor) -> None:
        n = len(x)
        if n < chunk:
            x = torch.cat([x, x[:1].expand(chunk - n, *x.shape[1:])])
        with torch.autocast(dev.type, dtype=dtype, enabled=dtype is not None):
            lg, emb, _ = model.forward_all(x.contiguous(memory_format=fmt))
        out_l.append(lg[:n].float().cpu())
        out_e.append(emb[:n].float().cpu())

    buf: torch.Tensor | None = None
    for vol in loader:
        x = contexts(vol.to(dev, non_blocking=True), model.context)
        buf = x if buf is None else torch.cat([buf, x])
        while len(buf) >= chunk:
            run(buf[:chunk])
            buf = buf[chunk:]
    if buf is not None and len(buf):
        run(buf)
    bounds = np.cumsum([s.length for s in series])[:-1]
    return (
        np.split(torch.cat(out_l).numpy(), bounds),
        np.split(torch.cat(out_e).numpy(), bounds),
    )


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
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    main_rank = rank == 0
    dev = device()
    if world > 1:
        local = int(os.environ.get("LOCAL_RANK", "0"))
        if dev.type == "cuda":
            torch.cuda.set_device(local)
            dev = torch.device("cuda", local)
        else:
            dev = torch.device("cpu")  # gloo; MPS has no collective backend
        dist.init_process_group("nccl" if dev.type == "cuda" else "gloo")
    cuda = dev.type == "cuda"
    if cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    data_dir = Path(cfg.data_dir)

    train_series = load_series(data_dir, "train")
    val_series = load_series(data_dir, "val")[: cfg.max_val_series]
    train_ds = SliceDataset(train_series, cfg.context, cfg.context_dropout, cfg.seed)
    labels_any = train_ds.labels()[:, 0]

    model = SliceNet(cfg.backbone, cfg.context, pretrained=cfg.pretrained).to(dev)
    # NHWC is ~1.3-1.5x faster for convnets on tensor cores; it was slower on Apple MPS.
    memory_format = torch.channels_last if cuda else torch.contiguous_format
    model.backbone.to(memory_format=memory_format)  # conv weights only; the head is 2D
    net = model  # unwrapped: evaluation, inference, checkpoints
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    dtype = amp_dtype(dev)
    scaler = torch.amp.GradScaler(dev.type, enabled=dtype == torch.float16)

    rng = np.random.default_rng(cfg.seed)
    per_epoch = len(balanced_epoch(labels_any, cfg.neg_ratio, rng))
    if cfg.max_train_items:
        per_epoch = min(per_epoch, cfg.max_train_items)
    steps_per_epoch = per_epoch // (cfg.batch_size * world)
    total = steps_per_epoch * cfg.epochs
    warmup = max(1, int(cfg.warmup_frac * total))

    start_epoch, step, best = 0, 0, -1.0
    last = run_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, step, best = ck["epoch"] + 1, ck["step"], ck["best"]
        rng = np.random.default_rng(cfg.seed + start_epoch)
        print(f"resumed from epoch {ck['epoch']}")
    if world > 1:
        model = DDP(net, device_ids=[dev.index] if cuda else None)

    run_id_file = run_dir / "mlflow_run_id"
    run_id = run_id_file.read_text().strip() if run_id_file.exists() else None
    if main_rank:
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment("brainct")
    tracking = mlflow.start_run(run_id=run_id, run_name=cfg.run_name) if main_rank else None
    with tracking or contextlib.nullcontext() as run:
        if main_rank:
            run_id_file.write_text(run.info.run_id)
        if main_rank and run_id is None:
            mlflow.log_params(cfg.to_dict())
            mlflow.log_params(
                {
                    "train_series": len(train_series),
                    "train_slices": len(train_ds),
                    "val_series": len(val_series),
                    "device": dev.type,
                    "world_size": world,
                }
            )

        for epoch in range(start_epoch, cfg.epochs if not args.skip_train else 0):
            model.train()
            idx = balanced_epoch(labels_any, cfg.neg_ratio, rng)[:per_epoch][rank::world]
            loader = DataLoader(
                Subset(train_ds, idx.tolist()),
                batch_size=cfg.batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=cfg.num_workers,
                persistent_workers=False,
                pin_memory=cuda,
                prefetch_factor=4 if cfg.num_workers else None,
            )
            t0, seen, running = time.time(), 0, 0.0
            for i, (x, y) in enumerate(loader):
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                x = augment(x).contiguous(memory_format=memory_format)
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
                    if main_rank:
                        mlflow.log_metric("train_loss", running, step=step)
            train_time = time.time() - t0
            rng = np.random.default_rng(cfg.seed + epoch + 1)
            if not main_rank:
                dist.barrier()  # wait for rank 0 to validate and checkpoint
                continue

            logits, _ = infer(net, val_series, dev)
            metrics = evaluate(logits, val_series)
            metrics |= {"epoch_seconds": train_time, "img_per_s": world * seen / train_time}
            mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()}, step=epoch)
            print(f"epoch {epoch}: " + json.dumps({k: round(v, 4) for k, v in metrics.items()}))
            if metrics["auc_mean"] > best:
                best = metrics["auc_mean"]
                torch.save(
                    {
                        "model": net.state_dict(),
                        "config": cfg.to_dict(),
                        "epoch": epoch,
                        "val": metrics,
                    },
                    run_dir / "best.pt",
                )
            torch.save(
                {
                    "model": net.state_dict(),
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "step": step,
                    "best": best,
                },
                last,
            )
            if world > 1:
                dist.barrier()

        if not main_rank:
            dist.destroy_process_group()
            return

        # Predictions of the selected checkpoint for every split (stage 2 + evaluation input).
        ck = torch.load(run_dir / "best.pt", map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"])
        mlflow.log_metric("best_epoch", ck["epoch"])
        for split in ("train", "val", "test"):
            series = load_series(data_dir, split)
            if split == "val" or (split == "test" and cfg.max_val_series):
                series = series[: cfg.max_val_series]
            elif split == "train" and cfg.max_train_items:
                series = series[: max(4, cfg.max_train_items // 32)]
            t0 = time.time()
            logits, embs = infer(net, series, dev)
            save_predictions(run_dir / "predictions" / f"cnn_{split}.npz", series, logits, embs)
            print(f"wrote {split} predictions ({len(series)} series, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
