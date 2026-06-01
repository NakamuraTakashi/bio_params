"""Train one MLP per target on GLODAP + BGC-Argo combined, with spatial CV.

Same pipeline as train_bgc_argo_target.py but loads both sources via the
combined loader. GLODAP adds deep (>2000 m) and coastal coverage that BGC-Argo
lacks; --per-source-max balances the two contributions (BGC-Argo O2/Chla dwarf
GLODAP otherwise).

Artifacts: models/pretrained/combined_<target>.pt
CV metrics: data/combined/processed/cv_<target>.json

Usage:
    uv run python scripts/train_combined_target.py --target NO3
    uv run python scripts/train_combined_target.py --target O2 --per-source-max 1000000
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
from bio_params.loaders.bgc_argo import attach_surface_chla
from bio_params.loaders.combined import available_targets, load_combined
from bio_params.model import MLP, MLPConfig
from bio_params.persist import save_artifact
from bio_params.train import TrainConfig, train

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
DEFAULT_SPROF = PROJECT_ROOT / "data" / "bgc_argo" / "raw" / "floats"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "pretrained"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "data" / "combined" / "processed"
DEFAULT_MATCHUP = (PROJECT_ROOT / "data" / "bgc_argo" / "processed"
                   / "satchl_matchup_combined.parquet")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target", required=True, choices=available_targets())
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--sprof-dir", type=Path, default=DEFAULT_SPROF)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    p.add_argument("--per-source-max", type=int, default=None,
                   help="Keep at most this many rows from EACH source (balance)")
    p.add_argument("--surface-chla", action="store_true",
                   help="Add SOCA-style satellite surface Chl-a feature")
    p.add_argument("--surface-chla-linear", action="store_true",
                   help="Use a linear surface-Chl feature instead of log10")
    p.add_argument("--matchup", type=Path, default=DEFAULT_MATCHUP,
                   help="Satellite surface-Chl matchup parquet (covers both sources)")
    p.add_argument("--tag", default=None,
                   help="Suffix for artifact/metrics names, e.g. --tag satchl")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--block-deg", type=float, default=5.0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16384)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-hidden-layers", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-final", action="store_true")
    p.add_argument("--log-every", type=int, default=25)
    return p.parse_args()


def evaluate(model, X_va_n, y_va, normalizer) -> dict[str, float]:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        pred_n = model(
            torch.from_numpy(X_va_n.astype(np.float32)).to(device)
        ).cpu().numpy()
    pred = normalizer.inverse_transform_y(pred_n)
    err = pred - y_va
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y_va - y_va.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"rmse": float(np.sqrt((err ** 2).mean())),
            "mae": float(np.abs(err).mean()), "r2": r2, "n": int(len(y_va))}


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    suffix = f"_{args.tag}" if args.tag else ""
    print(f"=== combined (GLODAP+BGC-Argo) target={args.target}{suffix}  spatial block CV ===")
    df = load_combined(args.csv, args.sprof_dir, target=args.target,
                       per_source_max=args.per_source_max, seed=args.seed)

    surface_chla = args.surface_chla
    surface_chla_log = not args.surface_chla_linear
    if surface_chla:
        df = attach_surface_chla(df, args.matchup)
        n_before = len(df)
        df = df[np.isfinite(df["surface_chla"])].reset_index(drop=True)
        print(f"  {len(df):,} rows with satellite surface Chl-a "
              f"({n_before - len(df):,} dropped: unmatched/masked/out-of-range)")

    ng = int((df.source == "glodap").sum())
    na = int((df.source == "bgc_argo").sum())
    print(f"  rows: {len(df):,}  (glodap={ng:,} {100*ng/len(df):.1f}%, bgc_argo={na:,})")

    feats = build_features(df, include_surface_chla=surface_chla,
                           surface_chla_log=surface_chla_log)
    fnames = feature_names(include_surface_chla=surface_chla,
                           surface_chla_log=surface_chla_log)
    X = feats.to_numpy()
    y = df[args.target].to_numpy()
    lat = df["latitude"].to_numpy()
    lon = df["longitude"].to_numpy()
    print(f"  X shape: {X.shape}  y range: [{y.min():.3g}, {y.max():.3g}]")

    train_cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                            early_stopping_patience=args.patience, log_every=args.log_every)
    mlp_cfg = MLPConfig(in_dim=X.shape[1], hidden=args.hidden,
                        n_hidden_layers=args.n_hidden_layers)

    fold_metrics: list[dict] = []
    print(f"\n--- {args.folds} folds x {args.block_deg} deg blocks ---")
    for k, tr, va in spatial_block_split(lat, lon, block_deg=args.block_deg,
                                         n_folds=args.folds, seed=args.seed):
        print(f"\n[fold {k+1}/{args.folds}] train={len(tr):,}  val={len(va):,}")
        norm = Normalizer.fit(X[tr], y[tr])
        train_ds = TabularDataset(norm.transform_x(X[tr]), norm.transform_y(y[tr]))
        val_ds = TabularDataset(norm.transform_x(X[va]), norm.transform_y(y[va]))
        model = MLP(mlp_cfg)
        res = train(model, train_ds, val_ds, train_cfg)
        met = evaluate(model, norm.transform_x(X[va]), y[va], norm)
        met["fold"] = k
        fold_metrics.append(met)
        print(f"  fold {k} -> RMSE={met['rmse']:.3g}  R^2={met['r2']:.4f}  (epochs={res.n_epochs_run})")

    rmses = np.array([m["rmse"] for m in fold_metrics])
    r2s = np.array([m["r2"] for m in fold_metrics])
    print("\n=== CV summary ===")
    print(f"RMSE: {rmses.mean():.3g} +/- {rmses.std():.3g}")
    print(f"R^2:  {r2s.mean():.4f} +/- {r2s.std():.4f}")

    maes = np.array([m["mae"] for m in fold_metrics])
    payload = {"target": args.target, "source": "combined",
               "per_source_max": args.per_source_max, "n_rows": int(len(X)),
               "n_glodap": ng, "n_bgc_argo": na,
               "surface_chla": surface_chla, "surface_chla_log": surface_chla_log,
               "feature_names": fnames, "n_folds": args.folds, "block_deg": args.block_deg,
               "rmse_mean": float(rmses.mean()), "rmse_std": float(rmses.std()),
               "mae_mean": float(maes.mean()), "r2_median": float(np.median(r2s)),
               "r2_mean": float(r2s.mean()), "r2_std": float(r2s.std()),
               "folds": fold_metrics}
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    (args.metrics_dir / f"cv_{args.target}{suffix}.json").write_text(json.dumps(payload, indent=2))
    print(f"\nSaved CV metrics -> {args.metrics_dir / f'cv_{args.target}{suffix}.json'}")

    if args.no_final:
        return 0

    print("\n--- Final training on all rows ---")
    norm = Normalizer.fit(X, y)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    cut = int(0.9 * len(perm))
    train_ds = TabularDataset(norm.transform_x(X[perm[:cut]]), norm.transform_y(y[perm[:cut]]))
    val_ds = TabularDataset(norm.transform_x(X[perm[cut:]]), norm.transform_y(y[perm[cut:]]))
    final_model = MLP(mlp_cfg)
    final_res = train(final_model, train_ds, val_ds, train_cfg)
    print(f"  trained for {final_res.n_epochs_run} epochs")

    artifact_path = args.out_dir / f"combined_{args.target}{suffix}.pt"
    save_artifact(artifact_path, model=final_model, normalizer=norm,
                  feature_names=fnames, target_name=args.target,
                  extra={"source": "combined", "n_rows": int(len(X)),
                         "n_glodap": ng, "n_bgc_argo": na,
                         "per_source_max": args.per_source_max,
                         "surface_chla": surface_chla, "surface_chla_log": surface_chla_log,
                         "log_target": False, "include_season": False,
                         "cv_rmse_mean": float(rmses.mean()), "cv_r2_mean": float(r2s.mean()),
                         "cv_r2_std": float(r2s.std()), "cv_block_deg": args.block_deg,
                         "cv_n_folds": args.folds, "epochs_run_final": final_res.n_epochs_run})
    print(f"Saved model artifact -> {artifact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
