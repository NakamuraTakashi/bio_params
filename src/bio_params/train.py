"""Training loop with early stopping for the per-target MLP."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 0.0
    early_stopping_patience: int = 20
    num_workers: int = 0
    device: str | None = None  # None -> "cuda" if available else "cpu"
    log_every: int = 10


@dataclass
class TrainResult:
    best_state: dict[str, torch.Tensor]
    best_val_loss: float
    history: list[dict[str, Any]] = field(default_factory=list)
    n_epochs_run: int = 0


def _resolve_device(name: str | None) -> torch.device:
    if name is not None:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    config: TrainConfig | None = None,
) -> TrainResult:
    """Train `model` on `train_ds`, early-stopping on `val_ds` MSE.

    On return, `model` has had `best_state` re-loaded so it is ready to use
    for inference. `best_state` is also returned for persistence.
    """
    cfg = config or TrainConfig()
    device = _resolve_device(cfg.device)
    model.to(device)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = cfg.early_stopping_patience
    history: list[dict[str, Any]] = []
    epoch = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = _run_epoch(
            model, train_loader, loss_fn, device, optimizer
        )
        val_loss = _run_epoch(model, val_loader, loss_fn, device, optimizer=None)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.early_stopping_patience
        else:
            patience_left -= 1

        if epoch == 1 or epoch % cfg.log_every == 0 or improved is False and patience_left == 0:
            marker = " *" if improved else ""
            print(
                f"  epoch {epoch:4d}  train={train_loss:.5f}  val={val_loss:.5f}  "
                f"best={best_val_loss:.5f}{marker}"
            )

        if patience_left <= 0:
            print(f"  early stopping at epoch {epoch} (patience exhausted)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainResult(
        best_state=best_state if best_state is not None else model.state_dict(),
        best_val_loss=best_val_loss,
        history=history,
        n_epochs_run=epoch,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    """Run one epoch. Training when `optimizer` is provided, else eval."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_n = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if is_train:
                loss.backward()
                optimizer.step()
            n = xb.size(0)
            total_loss += loss.item() * n
            total_n += n
    return total_loss / total_n
