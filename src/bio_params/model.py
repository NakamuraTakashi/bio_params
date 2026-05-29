"""MLP architecture for per-target biogeochemical regression."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
from torch import nn


@dataclass(frozen=True)
class MLPConfig:
    in_dim: int
    hidden: int = 128
    n_hidden_layers: int = 3
    out_dim: int = 1
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class MLP(nn.Module):
    """Simple feed-forward network: in_dim -> [hidden x n_hidden_layers] -> out_dim.

    Returns a tensor of shape (batch,) when out_dim == 1, else (batch, out_dim).
    """

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        d = config.in_dim
        for _ in range(config.n_hidden_layers):
            layers.append(nn.Linear(d, config.hidden))
            layers.append(nn.ReLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            d = config.hidden
        layers.append(nn.Linear(d, config.out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.squeeze(-1) if self.config.out_dim == 1 else out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
