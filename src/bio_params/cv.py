"""Spatial block cross-validation.

Random train/val splits leak nearby points into both folds and overestimate
generalization. This module groups samples into lat/lon blocks (default 5x5
degrees) and holds out entire blocks per fold so the validation set is
geographically disjoint from the training set. See CLAUDE.md "検証".
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def assign_blocks(
    latitude: np.ndarray,
    longitude: np.ndarray,
    block_deg: float = 5.0,
) -> np.ndarray:
    """Return one integer block ID per row; same ID == same lat/lon block."""
    lat_idx = np.floor(latitude / block_deg).astype(np.int64)
    lon_idx = np.floor(longitude / block_deg).astype(np.int64)
    # Multiplier > number of possible lon bins so (lat_idx, lon_idx) is injective.
    return lat_idx * 100_000 + lon_idx


def fold_assignments(
    latitude: np.ndarray,
    longitude: np.ndarray,
    block_deg: float = 5.0,
    n_folds: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Assign each row to a fold (0..n_folds-1) keeping each block intact."""
    blocks = assign_blocks(latitude, longitude, block_deg)
    unique_blocks = np.unique(blocks)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_blocks)
    block_to_fold = {
        int(b): int(i % n_folds) for i, b in enumerate(unique_blocks)
    }
    return np.fromiter(
        (block_to_fold[int(b)] for b in blocks),
        dtype=np.int64,
        count=len(blocks),
    )


def spatial_block_split(
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    block_deg: float = 5.0,
    n_folds: int = 5,
    seed: int = 42,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (fold_index, train_idx, val_idx) for each of `n_folds` folds."""
    folds = fold_assignments(latitude, longitude, block_deg, n_folds, seed)
    indices = np.arange(len(folds))
    for k in range(n_folds):
        val_mask = folds == k
        yield k, indices[~val_mask], indices[val_mask]
