"""Train & save a GLODAP model with MLD added as a feature (base 7 + log_mld).

For O2 (--aou) the model learns AOU = O2sat(T,S) - O2 and O2 is reconstructed at
use time as O2 = O2sat - AOU (O2sat from gsw). The artifact records include_mld
and, for O2, aou_decomposition so inference rebuilds the right quantity.

Saved as models/pretrained/glodap_<target>_<tag>.pt (default tag "mld"), leaving
the base glodap_<target>.pt intact. CV metrics use a common (finite-MLD) row set.

Usage:
    uv run python scripts/train_glodap_mld.py --target NO3
    uv run python scripts/train_glodap_mld.py --target O2 --aou
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
from bio_params.loaders.glodap import load_glodap
from bio_params.model import MLP, MLPConfig
from bio_params.persist import save_artifact
from bio_params.profiles import add_mld
from bio_params.train import TrainConfig, train

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
MODEL_DIR = ROOT / "models" / "pretrained"
METRICS_DIR = ROOT / "data" / "glodap" / "processed"


def o2sol(salinity, temperature):
    import gsw
    return gsw.O2sol_SP_pt(salinity, temperature)


def _r2(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    ss = float(((pred - obs) ** 2).sum()); tot = float(((obs - obs.mean()) ** 2).sum())
    return 1.0 - ss / tot if tot > 0 else float("nan")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target", required=True)
    p.add_argument("--aou", action="store_true",
                   help="(O2) learn AOU = O2sat - O2 and reconstruct O2 = O2sat - AOU")
    p.add_argument("--tag", default="mld")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--block-deg", type=float, default=5.0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-hidden-layers", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== GLODAP {args.target} + MLD"
          + ("  (AOU decomposition)" if args.aou else "") + " ===")
    df = load_glodap(CSV, args.target, with_time=True)
    df = add_mld(df)
    n0 = len(df)
    df = df[np.isfinite(df["mld"])].reset_index(drop=True)
    print(f"  {len(df):,} rows with finite MLD (of {n0:,})")

    X = build_features(df, include_mld=True).to_numpy()
    fnames = feature_names(include_mld=True)
    lat = df["latitude"].to_numpy(); lon = df["longitude"].to_numpy()
    obs = df[args.target].to_numpy()

    if args.aou:
        o2s = o2sol(df["salinity"].to_numpy(), df["temperature"].to_numpy())
        y = o2s - obs                       # learn AOU
        to_native = lambda idx, vals: o2s[idx] - vals   # back to O2
        print(f"  O2sat median {np.median(o2s):.1f}; AOU median {np.median(y):.1f} umol/kg")
    else:
        y = obs
        to_native = None

    train_cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                            early_stopping_patience=args.patience, log_every=10_000)
    mlp_cfg = MLPConfig(in_dim=X.shape[1], hidden=args.hidden, n_hidden_layers=args.n_hidden_layers)

    r2s, rmses = [], []
    print(f"--- {args.folds} folds x {args.block_deg} deg blocks ---")
    for k, tr, va in spatial_block_split(lat, lon, block_deg=args.block_deg,
                                         n_folds=args.folds, seed=args.seed):
        norm = Normalizer.fit(X[tr], y[tr])
        tr_ds = TabularDataset(norm.transform_x(X[tr]), norm.transform_y(y[tr]))
        va_ds = TabularDataset(norm.transform_x(X[va]), norm.transform_y(y[va]))
        model = MLP(mlp_cfg); train(model, tr_ds, va_ds, train_cfg)
        model.eval()
        with torch.no_grad():
            pn = model(torch.from_numpy(norm.transform_x(X[va]).astype(np.float32)).to(device)).cpu().numpy()
        pred = norm.inverse_transform_y(pn)
        o = to_native(va, y[va]) if to_native else y[va]
        p = to_native(va, pred) if to_native else pred
        r2s.append(_r2(o, p)); rmses.append(float(np.sqrt(np.mean((p - o) ** 2))))
        print(f"  fold {k}: R2={r2s[-1]:.4f}  RMSE={rmses[-1]:.3g}")
    r2m, rmm = float(np.mean(r2s)), float(np.mean(rmses))
    print(f"=== CV ({args.target}{'/O2' if args.aou else ''}): R2={r2m:.4f}  RMSE={rmm:.3g} ===")

    print("--- final training on all rows ---")
    norm = Normalizer.fit(X, y)
    rng = np.random.default_rng(args.seed); perm = rng.permutation(len(X)); cut = int(0.9 * len(perm))
    tr_ds = TabularDataset(norm.transform_x(X[perm[:cut]]), norm.transform_y(y[perm[:cut]]))
    va_ds = TabularDataset(norm.transform_x(X[perm[cut:]]), norm.transform_y(y[perm[cut:]]))
    final = MLP(mlp_cfg); res = train(final, tr_ds, va_ds, train_cfg)

    art = MODEL_DIR / f"glodap_{args.target}_{args.tag}.pt"
    save_artifact(art, model=final, normalizer=norm, feature_names=fnames,
                  target_name="AOU" if args.aou else args.target,
                  extra=dict(source="glodap", target=args.target, include_mld=True,
                             aou_decomposition=bool(args.aou), log_target=False,
                             include_season=False, include_sigma=False,
                             cv_r2_mean=r2m, cv_rmse_mean=rmm, n_rows=int(len(X)),
                             epochs_run_final=res.n_epochs_run))
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / f"cv_{args.target}_{args.tag}.json").write_text(json.dumps(
        dict(target=args.target, aou=bool(args.aou), include_mld=True,
             feature_names=fnames, cv_r2_mean=r2m, cv_rmse_mean=rmm,
             folds=[{"fold": k, "r2": r2s[k], "rmse": rmses[k]} for k in range(len(r2s))]),
        indent=2))
    print(f"Saved model -> {art}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
