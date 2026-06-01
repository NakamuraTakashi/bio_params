"""Train one MLP per target from BGC-Argo profiles, with spatial block CV.

Mirrors scripts/train_target.py (GLODAP) but loads BGC-Argo _Sprof.nc via the
bgc_argo loader and can add seasonal (day-of-year) features. BGC-Argo carries
a real timestamp per profile, so seasonal encoding is meaningful here (unlike
the static GLODAP models).

It can also add a SOCA-style satellite surface Chl-a feature (--surface-chla),
attached from the GlobColour matchup parquet; rows without a satellite match
are dropped. Use --tag to suffix the artifact name so experiments do not
clobber each other (e.g. --tag satchl -> bgc_argo_Chla_satchl.pt).

Artifacts are written to models/pretrained/bgc_argo_<target>[_<tag>].pt and
CV metrics to data/bgc_argo/processed/cv_<target>[_<tag>].json. The artifact's
extra dict records the feature flags so inference rebuilds the exact feature set.

Usage:
    uv run python scripts/train_bgc_argo_target.py --target Chla --season
    uv run python scripts/train_bgc_argo_target.py --target Chla \
        --box -180 180 -90 90 --surface-chla --tag satchl
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
from bio_params.loaders.bgc_argo import (
    attach_surface_chla, available_targets, load_bgc_argo,
)
from bio_params.model import MLP, MLPConfig
from bio_params.persist import save_artifact
from bio_params.train import TrainConfig, train

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPROF_DIR = PROJECT_ROOT / "data" / "bgc_argo" / "raw" / "floats"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "pretrained"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "data" / "bgc_argo" / "processed"
DEFAULT_MATCHUP = DEFAULT_METRICS_DIR / "satchl_matchup.parquet"

# Kuroshio-covering box used for the BGC-Argo download/selection.
DEFAULT_BOX = (120.0, 180.0, 10.0, 50.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target", required=True, choices=available_targets())
    p.add_argument("--sprof-dir", type=Path, default=DEFAULT_SPROF_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    p.add_argument("--box", type=float, nargs=4, default=list(DEFAULT_BOX),
                   metavar=("LON0", "LON1", "LAT0", "LAT1"))
    p.add_argument("--season", action="store_true",
                   help="Add day-of-year (sin/cos) seasonal features")
    p.add_argument("--surface-chla", action="store_true",
                   help="Add SOCA-style satellite surface Chl-a feature "
                        "(requires the matchup parquet)")
    p.add_argument("--surface-chla-linear", action="store_true",
                   help="Use a linear surface-Chl feature instead of log10 "
                        "(default log10, SOCA; linear can blow up CV folds)")
    p.add_argument("--matchup", type=Path, default=DEFAULT_MATCHUP,
                   help="Satellite surface-Chl matchup parquet")
    p.add_argument("--tag", default=None,
                   help="Suffix for artifact/metrics names, e.g. --tag satchl")
    p.add_argument("--include-sigma", action="store_true")
    p.add_argument("--log-target", action="store_true",
                   help="Train on log10(target); good for Chl-a (log-normal)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--block-deg", type=float, default=5.0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-hidden-layers", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subsample", type=int, default=None,
                   help="Randomly keep at most N rows (for huge targets like "
                        "global O2). Sampling is uniform over all levels, so it "
                        "preserves the spatial/seasonal distribution.")
    p.add_argument("--no-final", action="store_true")
    p.add_argument("--log-every", type=int, default=25)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.tag:
        suffix = f"_{args.tag}"
    elif args.season:
        suffix = "_season"
    else:
        suffix = ""
    print(f"=== BGC-Argo target={args.target}{suffix}  spatial block CV ===")
    print(f"Loading {args.sprof_dir} ...")
    df = load_bgc_argo(args.sprof_dir, target=args.target, box=tuple(args.box))
    print(f"  rows after QC/mode/box filtering: {len(df):,}")
    if len(df) == 0:
        print("ERROR: no rows; download floats first via download_bgc_argo.py")
        return 1

    if args.surface_chla:
        df = attach_surface_chla(df, args.matchup)
        n_before = len(df)
        df = df[np.isfinite(df["surface_chla"])].reset_index(drop=True)
        print(f"  {len(df):,} rows with satellite surface Chl-a "
              f"({n_before - len(df):,} dropped: unmatched/masked/post-MY)")

    if args.subsample is not None and len(df) > args.subsample:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(df), size=args.subsample, replace=False)
        df = df.iloc[np.sort(keep)].reset_index(drop=True)
        print(f"  subsampled to {len(df):,} rows (uniform, seed={args.seed})")

    surface_chla_log = not args.surface_chla_linear
    feats = build_features(
        df, include_sigma_theta=args.include_sigma, include_season=args.season,
        include_surface_chla=args.surface_chla, surface_chla_log=surface_chla_log,
    )
    fnames = feature_names(
        include_sigma_theta=args.include_sigma, include_season=args.season,
        include_surface_chla=args.surface_chla, surface_chla_log=surface_chla_log,
    )
    X = feats.to_numpy()
    y_raw = df[args.target].to_numpy()

    # Optional log transform (Chl-a spans orders of magnitude). Clip tiny/neg
    # values (sensor noise near zero) before log so the transform is defined.
    if args.log_target:
        y = np.log10(np.clip(y_raw, 1e-3, None))
        print(f"  using log10(target); clip floor=1e-3")
    else:
        y = y_raw

    lat = df["latitude"].to_numpy()
    lon = df["longitude"].to_numpy()
    print(f"  X shape: {X.shape}  features: {fnames}")
    print(f"  y range (model space): [{y.min():.3g}, {y.max():.3g}]")

    train_cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        early_stopping_patience=args.patience, log_every=args.log_every,
    )
    mlp_cfg = MLPConfig(
        in_dim=X.shape[1], hidden=args.hidden,
        n_hidden_layers=args.n_hidden_layers,
    )

    def to_native(yv):
        """Map model-space predictions/targets back to native units for metrics."""
        return np.power(10.0, yv) if args.log_target else yv

    fold_metrics: list[dict] = []
    print(f"\n--- {args.folds} folds x {args.block_deg} deg blocks ---")
    for k, train_idx, val_idx in spatial_block_split(
        lat, lon, block_deg=args.block_deg, n_folds=args.folds, seed=args.seed
    ):
        print(f"\n[fold {k + 1}/{args.folds}] train={len(train_idx):,}  val={len(val_idx):,}")
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        norm = Normalizer.fit(X_tr, y_tr)
        train_ds = TabularDataset(norm.transform_x(X_tr), norm.transform_y(y_tr))
        val_ds = TabularDataset(norm.transform_x(X_va), norm.transform_y(y_va))

        model = MLP(mlp_cfg)
        res = train(model, train_ds, val_ds, train_cfg)

        # Metrics in native units so they are comparable to the GLODAP models.
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            pred_n = model(
                torch.from_numpy(norm.transform_x(X_va).astype(np.float32)).to(device)
            ).cpu().numpy()
        pred_native = to_native(norm.inverse_transform_y(pred_n))
        y_va_native = to_native(y_va)
        err = pred_native - y_va_native
        ss_res = float((err ** 2).sum())
        ss_tot = float(((y_va_native - y_va_native.mean()) ** 2).sum())
        metrics = {
            "fold": k,
            "rmse": float(np.sqrt((err ** 2).mean())),
            "mae": float(np.abs(err).mean()),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "n": int(len(y_va)),
            "epochs_run": res.n_epochs_run,
        }
        fold_metrics.append(metrics)
        print(f"  fold {k} -> RMSE={metrics['rmse']:.3g}  MAE={metrics['mae']:.3g}  "
              f"R^2={metrics['r2']:.4f}  (epochs={res.n_epochs_run})")

    rmses = np.array([m["rmse"] for m in fold_metrics])
    maes = np.array([m["mae"] for m in fold_metrics])
    r2s = np.array([m["r2"] for m in fold_metrics])
    print("\n=== CV summary (native units) ===")
    print(f"RMSE: {rmses.mean():.3g} +/- {rmses.std():.3g}")
    print(f"MAE:  {maes.mean():.3g} +/- {maes.std():.3g}")
    print(f"R^2:  {r2s.mean():.4f} +/- {r2s.std():.4f}")

    payload = {
        "target": args.target, "source": "bgc_argo",
        "include_season": args.season, "include_sigma": args.include_sigma,
        "surface_chla": args.surface_chla, "surface_chla_log": surface_chla_log,
        "log_target": args.log_target, "box": list(args.box),
        "n_folds": args.folds, "block_deg": args.block_deg,
        "feature_names": fnames, "n_rows": int(len(X)),
        "rmse_mean": float(rmses.mean()), "rmse_std": float(rmses.std()),
        "mae_mean": float(maes.mean()), "mae_std": float(maes.std()),
        "r2_mean": float(r2s.mean()), "r2_std": float(r2s.std()),
        "folds": fold_metrics,
    }
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.metrics_dir / f"cv_{args.target}{suffix}.json"
    metrics_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved CV metrics -> {metrics_path}")

    if args.no_final:
        return 0

    print("\n--- Final training on all rows ---")
    norm = Normalizer.fit(X, y)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    cut = int(0.9 * len(perm))
    tr_idx, va_idx = perm[:cut], perm[cut:]
    train_ds = TabularDataset(norm.transform_x(X[tr_idx]), norm.transform_y(y[tr_idx]))
    val_ds = TabularDataset(norm.transform_x(X[va_idx]), norm.transform_y(y[va_idx]))
    final_model = MLP(mlp_cfg)
    final_res = train(final_model, train_ds, val_ds, train_cfg)
    print(f"  trained for {final_res.n_epochs_run} epochs")

    artifact_path = args.out_dir / f"bgc_argo_{args.target}{suffix}.pt"
    save_artifact(
        artifact_path, model=final_model, normalizer=norm,
        feature_names=fnames, target_name=args.target,
        extra={
            "source": "bgc_argo", "n_rows": int(len(X)),
            "include_season": args.season, "include_sigma": args.include_sigma,
            "surface_chla": args.surface_chla, "surface_chla_log": surface_chla_log,
            "log_target": args.log_target, "box": list(args.box),
            "cv_rmse_mean": float(rmses.mean()), "cv_rmse_std": float(rmses.std()),
            "cv_r2_mean": float(r2s.mean()), "cv_r2_std": float(r2s.std()),
            "cv_block_deg": args.block_deg, "cv_n_folds": args.folds,
            "epochs_run_final": final_res.n_epochs_run,
        },
    )
    print(f"Saved model artifact -> {artifact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
