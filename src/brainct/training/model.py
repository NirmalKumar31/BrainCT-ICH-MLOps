"""Networks. Windowing lives inside the graph, so the exported ONNX model takes raw HU and the
service cannot window differently from training."""

from __future__ import annotations

import timm
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from brainct.constants import LABELS, WINDOWS


class HUWindowing(nn.Module):
    """[N, K, H, W] HU -> [N, 3K, H, W]: each slice in each window, scaled to ~zero mean."""

    def __init__(self) -> None:
        super().__init__()
        lo = torch.tensor([c - w / 2 for c, w in WINDOWS]).view(1, 1, -1, 1, 1)
        width = torch.tensor([w for _, w in WINDOWS]).view(1, 1, -1, 1, 1)
        self.register_buffer("lo", lo, persistent=False)
        self.register_buffer("width", width, persistent=False)

    def forward(self, hu: torch.Tensor) -> torch.Tensor:
        n, k, h, w = hu.shape
        x = ((hu.unsqueeze(2) - self.lo) / self.width).clamp(0.0, 1.0)
        return ((x - 0.5) / 0.25).reshape(n, k * len(WINDOWS), h, w)


class SliceNet(nn.Module):
    """2.5D CNN: HU context -> per-slice logits, plus embedding and feature map for CAM."""

    def __init__(self, backbone: str, context: int, pretrained: bool = True, drop: float = 0.2):
        super().__init__()
        self.context = context
        self.windowing = HUWindowing()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=len(WINDOWS) * (2 * context + 1),
            num_classes=0,
            global_pool="",
        )
        self.embed_dim = int(self.backbone.num_features)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(self.embed_dim, len(LABELS))

    def forward_all(self, hu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fmap = self.backbone(self.windowing(hu))  # [N, D, h, w]
        emb = fmap.mean(dim=(2, 3))
        return self.head(self.drop(emb)), emb, fmap

    def forward(self, hu: torch.Tensor) -> torch.Tensor:
        return self.forward_all(hu)[0]


class ExportSliceNet(nn.Module):
    """ONNX wrapper: (logits, embedding, feature map)."""

    def __init__(self, net: SliceNet):
        super().__init__()
        self.net = net
        # The wrapper itself must be in eval mode: torch.onnx.export restores the module's
        # original mode afterwards, and a new module defaults to train (dropout + batch BN).
        self.eval()

    def forward(self, hu: torch.Tensor):
        return self.net.forward_all(hu)


class SeqNet(nn.Module):
    """Series model: per-slice [embedding, CNN logits] -> BiGRU -> refined per-slice logits.

    Residual on the CNN logits, so at initialisation it reproduces the CNN.
    """

    def __init__(self, embed_dim: int, hidden: int = 128, layers: int = 2, drop: float = 0.3):
        super().__init__()
        n = len(LABELS)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim + n, 2 * hidden),
            nn.LayerNorm(2 * hidden),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.gru = nn.GRU(
            2 * hidden,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=drop if layers > 1 else 0.0,
        )
        self.out = nn.Linear(2 * hidden, n)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, emb: torch.Tensor, logits: torch.Tensor, lengths: torch.Tensor | None = None):
        x = self.proj(torch.cat([emb, logits], dim=-1))
        if lengths is not None:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            h, _ = self.gru(packed)
            h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])
        else:
            h, _ = self.gru(x)
        return logits + self.out(h)
