"""Feature/decomposition ablation for GLODAP core variables (CV-only).

For a target, runs spatial-block CV on a common row set under several feature
configs and reports R2, to test whether MLD / sigma_theta help, and (for O2)
whether predicting AOU and reconstructing O2 = O2sat - AOU beats predicting O2
directly.

  base      : 7 base features
  sigma     : + sigma_theta (potential density)
  mld       : + mixed-layer depth
  sigma+mld : + both
  (O2 with --aou) the same configs but the model learns AOU; O2 is then
  reconstructed as O2sat(T,S) - AOU and scored against observed O2.

All configs use the SAME rows (those with a finite MLD) so R2 is comparable.

Usage:
    uv run python scripts/ablation_features.py --target NO3
    uv run python scripts/ablation_features.py --target O2 --aou
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from bio_params.cv import spatial_block_split
from bio_params.dataset import Normalizer, TabularDataset
from bio_params.features import build_features, feature_names
from bio_params.loaders.glodap import load_glodap
from bio_params.model import MLP, MLPConfig
from bio_params.profiles import add_mld
from bio_params.train import TrainConfig, train

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"

CONFIGS = {
    "base": dict(),
    "sigma": dict(include_sigma_theta=True),
    "mld": dict(include_mld=True),
    "sigma+mld": dict(include_sigma_theta=True, include_mld=True),
}


def o2sol(salinity, temperature):
    """O2 saturation concentration (umol/kg) from practical S and temperature."""
    import gsw
    return gsw.O2sol_SP_pt(salinity, temperature)


def _r2(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    ss = float(((pred - obs) ** 2).sum()); tot = float(((obs - obs.mean()) ** 2).sum())
    return 1.0 - ss / tot if tot > 0 else float("nan")


def cv_r2(X, y, lat, lon, *, folds, block_deg, seed, cfg, train_cfg,
          to_obs=None):
    """5-fold spatial-block CV; returns (mean R2, per-fold R2). `to_obs` maps
    model-space target back to the observed quantity for scoring (AOU mode)."""
    r2s = []
    for k, tr, va in spatial_block_split(lat, lon, block_deg=block_deg,
                                         n_folds=folds, seed=seed):
        norm = Normalizer.fit(X[tr], y[tr])
        tr_ds = TabularDataset(norm.transform_x(X[tr]), norm.transform_y(y[tr]))
        va_ds = TabularDataset(norm.transform_x(X[va]), norm.transform_y(y[va]))
        model = MLP(cfg)
        train(model, tr_ds, va_ds, train_cfg)
        dev = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            pn = model(torch.from_numpy(norm.transform_x(X[va]).astype(np.float32)).to(dev)).cpu().numpy()
        pred = norm.inverse_transform_y(pn)
        if to_obs is not None:
            r2s.append(_r2(to_obs(va, y[va]), to_obs(va, pred)))
        else:
            r2s.append(_r2(y[va], pred))
    return float(np.mean(r2s)), [round(x, 4) for x in r2s]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target", required=True)
    p.add_argument("--aou", action="store_true",
                   help="(O2) also test predicting AOU and reconstructing O2")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--block-deg", type=float, default=5.0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"Loading GLODAP {args.target} (with_time) ...", flush=True)
    df = load_glodap(CSV, args.target, with_time=True)
    df = add_mld(df)
    n0 = len(df)
    df = df[np.isfinite(df["mld"])].reset_index(drop=True)
    print(f"  {len(df):,} rows with finite MLD (of {n0:,}); common set for all configs")
    lat = df["latitude"].to_numpy(); lon = df["longitude"].to_numpy()
    train_cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                            early_stopping_patience=args.patience, log_every=10_000)

    o2s = None
    if args.aou:
        o2s = o2sol(df["salinity"].to_numpy(), df["temperature"].to_numpy())
        aou = o2s - df[args.target].to_numpy()
        print(f"  O2sat: median {np.median(o2s):.1f}  AOU: median {np.median(aou):.1f} "
              f"umol/kg (min {aou.min():.1f}, max {aou.max():.1f})")

    print(f"\n=== {args.target}: direct prediction ===")
    results = {}
    for name, opt in CONFIGS.items():
        X = build_features(df, **opt).to_numpy()
        cfg = MLPConfig(in_dim=X.shape[1])
        m, folds = cv_r2(X, df[args.target].to_numpy(), lat, lon, folds=args.folds,
                         block_deg=args.block_deg, seed=args.seed, cfg=cfg, train_cfg=train_cfg)
        results[f"direct/{name}"] = m
        print(f"  {name:10s} ({len(feature_names(**opt))}f): R2={m:.4f}  folds={folds}", flush=True)

    if args.aou:
        print(f"\n=== {args.target}: AOU decomposition (O2 = O2sat - AOU) ===")
        to_o2 = lambda idx, aou_vals: o2s[idx] - aou_vals
        for name, opt in CONFIGS.items():
            X = build_features(df, **opt).to_numpy()
            cfg = MLPConfig(in_dim=X.shape[1])
            m, folds = cv_r2(X, aou, lat, lon, folds=args.folds, block_deg=args.block_deg,
                             seed=args.seed, cfg=cfg, train_cfg=train_cfg, to_obs=to_o2)
            results[f"aou/{name}"] = m
            print(f"  {name:10s} ({len(feature_names(**opt))}f): O2 R2={m:.4f}  folds={folds}", flush=True)

    print(f"\n=== SUMMARY ({args.target}) ===")
    base = results.get("direct/base")
    for k, v in results.items():
        d = f"  (dR2 vs direct/base = {v - base:+.4f})" if base is not None else ""
        print(f"  {k:18s} R2={v:.4f}{d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
