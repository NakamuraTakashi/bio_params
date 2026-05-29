"""Save and load a trained model artifact in a single .pt file.

The artifact bundles model weights, normalization constants, architecture
config, and feature/target naming so inference can be reproduced offline
without re-fitting the normalizer. See CLAUDE.md "モデルの保存".
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from bio_params.dataset import Normalizer
from bio_params.model import MLP, MLPConfig


def save_artifact(
    path: str | Path,
    *,
    model: MLP,
    normalizer: Normalizer,
    feature_names: list[str],
    target_name: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "model_config": model.config.to_dict(),
        "normalizer": normalizer.to_dict(),
        "feature_names": list(feature_names),
        "target_name": target_name,
        "extra": extra or {},
    }
    torch.save(payload, out)
    return out


def load_artifact(
    path: str | Path,
    map_location: str | torch.device | None = None,
) -> tuple[MLP, Normalizer, dict[str, Any]]:
    """Reconstruct (model, normalizer, metadata) from a saved .pt file."""
    # weights_only=False because the payload contains dict metadata, not just
    # tensors. Safe here because we only load artifacts we produced.
    payload = torch.load(path, map_location=map_location or "cpu", weights_only=False)
    config = MLPConfig(**payload["model_config"])
    model = MLP(config)
    model.load_state_dict(payload["model_state"])
    normalizer = Normalizer.from_dict(payload["normalizer"])
    meta = {
        "feature_names": payload["feature_names"],
        "target_name": payload["target_name"],
        "extra": payload.get("extra", {}),
    }
    return model, normalizer, meta
