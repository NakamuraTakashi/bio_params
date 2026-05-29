"""PyTorch Dataset and feature/target normalization.

The Normalizer is fit on the training fold only (never on val/test) and
its mean/std arrays are persisted alongside the model so inference reproduces
the same input transform. See CLAUDE.md "モデルの保存" section.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Normalizer:
    """Per-feature standardization for inputs and scalar standardization for the target."""

    x_mean: np.ndarray  # shape (n_features,)
    x_std: np.ndarray   # shape (n_features,)
    y_mean: float
    y_std: float

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray) -> "Normalizer":
        return cls(
            x_mean=X.mean(axis=0),
            x_std=X.std(axis=0),
            y_mean=float(y.mean()),
            y_std=float(y.std()),
        )

    def transform_x(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean) / self.x_std

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / self.y_std

    def inverse_transform_y(self, y_norm: np.ndarray) -> np.ndarray:
        return y_norm * self.y_std + self.y_mean

    def to_dict(self) -> dict:
        return {
            "x_mean": self.x_mean.tolist(),
            "x_std": self.x_std.tolist(),
            "y_mean": self.y_mean,
            "y_std": self.y_std,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(
            x_mean=np.asarray(d["x_mean"], dtype=np.float64),
            x_std=np.asarray(d["x_std"], dtype=np.float64),
            y_mean=float(d["y_mean"]),
            y_std=float(d["y_std"]),
        )


class TabularDataset(Dataset):
    """Holds already-normalized X and y as float32 tensors."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
