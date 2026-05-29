"""Smoke test: train an MLP on TA with a random train/val split.

This is intentionally NOT a proper evaluation -- random splits leak nearby
points into both folds and overestimate generalization. The point here is
to verify the loader / features / dataset / model / train stack works
end-to-end before adding spatial block CV. See CLAUDE.md for the validation
plan.

Usage:
    uv run python scripts/smoke_train_ta.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from bio_params.dataset import Normalizer, TabularDataset
from bio_params.features import build_features, feature_names
from bio_params.loaders.glodap import load_glodap
from bio_params.model import MLP, MLPConfig
from bio_params.train import TrainConfig, train

CSV_PATH = Path("data/glodap/raw/GLODAPv2.2023_Merged_Master_File.csv")
TARGET = "TA"
INCLUDE_SIGMA = False
SEED = 42


def main() -> None:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print(f"Loading target={TARGET} from GLODAP CSV ...")
    df = load_glodap(CSV_PATH, target=TARGET)
    print(f"  rows after coord/flag filtering: {len(df):,}")

    feats = build_features(df, include_sigma_theta=INCLUDE_SIGMA)
    X = feats.to_numpy()
    y = df[TARGET].to_numpy()
    print(f"  feature names: {feature_names(include_sigma_theta=INCLUDE_SIGMA)}")
    print(f"  X shape: {X.shape}, y range: [{y.min():.1f}, {y.max():.1f}]")

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    print(f"  train: {len(X_tr):,}  val: {len(X_va):,}")

    norm = Normalizer.fit(X_tr, y_tr)
    print(f"  y_mean={norm.y_mean:.2f}  y_std={norm.y_std:.2f}")

    train_ds = TabularDataset(norm.transform_x(X_tr), norm.transform_y(y_tr))
    val_ds = TabularDataset(norm.transform_x(X_va), norm.transform_y(y_va))

    config = MLPConfig(in_dim=X.shape[1], hidden=128, n_hidden_layers=3)
    model = MLP(config)
    print(f"  model params: {model.num_parameters():,}")

    cfg = TrainConfig(
        epochs=200,
        batch_size=4096,
        lr=1e-3,
        early_stopping_patience=15,
        log_every=10,
    )
    print()
    result = train(model, train_ds, val_ds, cfg)
    print(f"\n  best val MSE (normalized): {result.best_val_loss:.5f}")
    print(f"  epochs run: {result.n_epochs_run}")

    # Evaluate on the un-normalized scale (umol/kg for TA).
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_va_n = norm.transform_x(X_va)
        pred_n = model(torch.from_numpy(X_va_n.astype(np.float32)).to(device)).cpu().numpy()
    pred = norm.inverse_transform_y(pred_n)
    err = pred - y_va
    rmse = float(np.sqrt((err ** 2).mean()))
    mae = float(np.abs(err).mean())
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y_va - y_va.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    print()
    print(f"  Validation RMSE: {rmse:.2f} umol/kg")
    print(f"  Validation MAE:  {mae:.2f} umol/kg")
    print(f"  Validation R^2:  {r2:.4f}")
    print()
    print("Note: random split overestimates generalization; switch to spatial CV next.")


if __name__ == "__main__":
    main()
