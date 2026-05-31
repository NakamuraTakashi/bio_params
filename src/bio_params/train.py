"""Training loop with early stopping for the per-target MLP.

The MLP is tiny (7 -> 128x3 -> 1), so per-batch CPU->GPU transfer and the
DataLoader/Python overhead dominate, not the GPU math. We therefore move the
whole (already-normalized) dataset onto the device once and iterate with index
slicing. This is ~10x faster than a num_workers=0 DataLoader for datasets of a
few million rows (which still fit in GPU memory: ~8 cols * 4 B * 3e6 ~ 100 MB).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset


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

    # Move the whole dataset onto the device once (see module docstring).
    Xtr, ytr = _device_tensors(train_ds, device)
    Xva, yva = _device_tensors(val_ds, device)

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
            model, Xtr, ytr, loss_fn, cfg.batch_size, optimizer
        )
        val_loss = _run_epoch(model, Xva, yva, loss_fn, cfg.batch_size, optimizer=None)
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


def _device_tensors(ds: Dataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the dataset's (X, y) as contiguous tensors on `device`.

    Fast path for TabularDataset (exposes .X/.y); otherwise stack via indexing.
    """
    X = getattr(ds, "X", None)
    y = getattr(ds, "y", None)
    if X is None or y is None:
        xs, ys = zip(*[ds[i] for i in range(len(ds))])
        X = torch.stack([torch.as_tensor(x) for x in xs])
        y = torch.stack([torch.as_tensor(t) for t in ys])
    return X.to(device), y.to(device)


def _run_epoch(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
    batch_size: int,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    """Run one epoch over device-resident tensors via index slicing.

    Training when `optimizer` is provided, else eval. The running loss is
    accumulated on-device and synced once at the end (no per-batch .item()).
    """
    is_train = optimizer is not None
    model.train(is_train)
    n = X.shape[0]
    order = torch.randperm(n, device=X.device) if is_train else torch.arange(n, device=X.device)
    total_loss = torch.zeros((), device=X.device)
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            xb = X[idx]
            yb = y[idx]
            if is_train:
                optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.detach() * xb.size(0)
    return float((total_loss / n).item())
