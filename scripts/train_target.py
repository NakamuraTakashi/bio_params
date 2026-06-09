"""Train one MLP per target with spatial block CV and save the final artifact.

Pipeline:
  1. Load <target> from the GLODAP CSV via the project's loader.
  2. Build the feature matrix.
  3. K-fold spatial block CV (5x5 deg blocks by default). For each fold,
     fit normalizer on train rows only and report RMSE / MAE / R^2 in
     un-normalized units.
  4. Save aggregated CV metrics to
       data/glodap/processed/cv_<target>.json
  5. Unless --no-final, train a single final model on ALL data using a
     small random holdout for early stopping (random holdout is OK here
     because honest evaluation is already done via the CV above), and
     save the artifact to
       models/pretrained/glodap_<target>.pt

Usage:
    uv run python scripts/train_target.py --target TA
    uv run python scripts/train_target.py --target NO3 --include-sigma
    uv run python scripts/train_target.py --target TA --no-final
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bio_params.cv import spatial_block_split
from bio_params.dataset import Normalizer, TabularDataset
from bio_params.features import build_features, feature_names
from bio_params.loaders.glodap import available_targets, load_glodap
from bio_params.model import MLP, MLPConfig
from bio_params.persist import save_artifact
from bio_params.train import TrainConfig, train

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "pretrained"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "data" / "glodap" / "processed"


def evaluate(
    model: MLP,
    X_va_n: np.ndarray,
    y_va: np.ndarray,
    normalizer: Normalizer,
) -> dict[str, float]:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        pred_n = model(
            torch.from_numpy(X_va_n.astype(np.float32)).to(device)
        ).cpu().numpy()
    pred = normalizer.inverse_transform_y(pred_n)
    err = pred - y_va
    rmse = float(np.sqrt((err ** 2).mean()))
    mae = float(np.abs(err).mean())
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y_va - y_va.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "n": int(len(y_va))}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target", required=True, choices=available_targets())
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    p.add_argument("--include-sigma", action="store_true",
                   help="Include sigma_theta (potential density) as a feature")
    p.add_argument("--tag", default=None,
                   help="Suffix for artifact/metrics names, e.g. --tag clean -> "
                        "glodap_<target>_clean.pt / cv_<target>_clean.json")
    p.add_argument("--value-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="keep only target values in [LO,HI] (data cleaning), "
                        "e.g. --value-range 30 150 for DOC")
    p.add_argument("--exclude-cruise", type=int, nargs="+", default=None,
                   help="drop these GLODAP cruise numbers (e.g. 4057 = DOC unit error)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--block-deg", type=float, default=5.0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-hidden-layers", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-final", action="store_true",
                   help="Skip the final all-data training run and artifact save")
    p.add_argument("--log-every", type=int, default=25)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    suffix = f"_{args.tag}" if args.tag else ""
    print(f"=== target={args.target}{suffix}  spatial block CV ===")
    print(f"Loading {args.csv} ...")
    df = load_glodap(args.csv, target=args.target,
                     value_range=tuple(args.value_range) if args.value_range else None,
                     exclude_cruises=args.exclude_cruise)
    if args.value_range or args.exclude_cruise:
        print(f"  cleaning: value_range={args.value_range} exclude_cruise={args.exclude_cruise}")
    print(f"  rows after coord/flag/clean filtering: {len(df):,}")

    feats = build_features(df, include_sigma_theta=args.include_sigma)
    fnames = feature_names(include_sigma_theta=args.include_sigma)
    X = feats.to_numpy()
    y = df[args.target].to_numpy()
    lat = df["latitude"].to_numpy()
    lon = df["longitude"].to_numpy()
    print(f"  X shape: {X.shape}  features: {fnames}")
    print(f"  y range: [{y.min():.3g}, {y.max():.3g}]")

    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        early_stopping_patience=args.patience,
        log_every=args.log_every,
    )
    mlp_cfg = MLPConfig(
        in_dim=X.shape[1],
        hidden=args.hidden,
        n_hidden_layers=args.n_hidden_layers,
    )

    fold_metrics: list[dict] = []
    print(f"\n--- {args.folds} folds x {args.block_deg} deg blocks ---")
    for k, train_idx, val_idx in spatial_block_split(
        lat, lon,
        block_deg=args.block_deg,
        n_folds=args.folds,
        seed=args.seed,
    ):
        print(f"\n[fold {k + 1}/{args.folds}] train={len(train_idx):,}  val={len(val_idx):,}")
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        norm = Normalizer.fit(X_tr, y_tr)
        train_ds = TabularDataset(norm.transform_x(X_tr), norm.transform_y(y_tr))
        val_ds = TabularDataset(norm.transform_x(X_va), norm.transform_y(y_va))

        model = MLP(mlp_cfg)
        res = train(model, train_ds, val_ds, train_cfg)

        metrics = evaluate(model, norm.transform_x(X_va), y_va, norm)
        metrics["fold"] = k
        metrics["epochs_run"] = res.n_epochs_run
        metrics["best_val_mse_normalized"] = res.best_val_loss
        fold_metrics.append(metrics)
        print(f"  fold {k} -> RMSE={metrics['rmse']:.3g}  MAE={metrics['mae']:.3g}  "
              f"R^2={metrics['r2']:.4f}  (epochs={res.n_epochs_run})")

    rmses = np.array([m["rmse"] for m in fold_metrics])
    maes = np.array([m["mae"] for m in fold_metrics])
    r2s = np.array([m["r2"] for m in fold_metrics])
    print("\n=== CV summary ===")
    print(f"RMSE: {rmses.mean():.3g} +/- {rmses.std():.3g}")
    print(f"MAE:  {maes.mean():.3g} +/- {maes.std():.3g}")
    print(f"R^2:  {r2s.mean():.4f} +/- {r2s.std():.4f}")

    payload = {
        "target": args.target,
        "include_sigma": args.include_sigma,
        "n_folds": args.folds,
        "block_deg": args.block_deg,
        "feature_names": fnames,
        "n_rows": int(len(X)),
        "rmse_mean": float(rmses.mean()),
        "rmse_std": float(rmses.std()),
        "mae_mean": float(maes.mean()),
        "mae_std": float(maes.std()),
        "r2_mean": float(r2s.mean()),
        "r2_std": float(r2s.std()),
        "folds": fold_metrics,
    }
    payload["value_range"] = args.value_range
    payload["exclude_cruise"] = args.exclude_cruise
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.metrics_dir / f"cv_{args.target}{suffix}.json"
    metrics_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved CV metrics -> {metrics_path}")

    if args.no_final:
        return 0

    print("\n--- Final training on all rows ---")
    # Honest evaluation is already done via CV above. The final model uses
    # all rows, with a small random holdout used only as an early-stopping
    # reference, not a performance estimate.
    norm = Normalizer.fit(X, y)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    cut = int(0.9 * len(perm))
    tr_idx, va_idx = perm[:cut], perm[cut:]
    train_ds = TabularDataset(
        norm.transform_x(X[tr_idx]), norm.transform_y(y[tr_idx])
    )
    val_ds = TabularDataset(
        norm.transform_x(X[va_idx]), norm.transform_y(y[va_idx])
    )
    final_model = MLP(mlp_cfg)
    final_res = train(final_model, train_ds, val_ds, train_cfg)
    print(f"  trained for {final_res.n_epochs_run} epochs")

    artifact_path = args.out_dir / f"glodap_{args.target}{suffix}.pt"
    save_artifact(
        artifact_path,
        model=final_model,
        normalizer=norm,
        feature_names=fnames,
        target_name=args.target,
        extra={
            "source": "glodap",
            "n_rows": int(len(X)),
            "cv_rmse_mean": float(rmses.mean()),
            "cv_rmse_std": float(rmses.std()),
            "cv_r2_mean": float(r2s.mean()),
            "cv_r2_std": float(r2s.std()),
            "cv_block_deg": args.block_deg,
            "cv_n_folds": args.folds,
            "include_sigma": args.include_sigma,
            "value_range": args.value_range,
            "exclude_cruise": args.exclude_cruise,
            "epochs_run_final": final_res.n_epochs_run,
        },
    )
    print(f"Saved model artifact -> {artifact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
