"""Train a SOCA-style RELATIVE vertical Chl-a profile model.

The target is the surface-normalized profile  rel(z) = Chla(z) / Chla_surface
(clipped to [0, rel_cap]); features are the 7 base features + mixed-layer depth
(MLD). The model learns the SHAPE only; at inference the amplitude is restored
from the satellite surface field:  Chla(z) = rel_pred(z) * satellite_surface.

Why: in-situ Chl-a (esp. BGC-Argo fluorescence) is biased/scattered in absolute
terms vs satellite, but its profile SHAPE is reliable; normalizing by the
profile's own surface cancels the multiplicative calibration bias. MLD predicts
the shape (mixed vs deep-chlorophyll-maximum regime).

CV reports two scores:
  * shape  R^2 : on rel (the learning target)
  * abs    R^2 : on rel_pred * satellite_surface vs in-situ Chla (deploy-relevant;
                 limited by the irreducible satellite-vs-in-situ surface mismatch)

Artifacts: models/pretrained/<source>_Chla_rel[_<tag>].pt
CV metrics: data/<...>/processed/cv_Chla_rel[_<tag>].json

Usage:
    uv run python scripts/train_chla_relative.py --source bgc_argo
    uv run python scripts/train_chla_relative.py --source combined --per-source-max 100000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bio_params.cv import spatial_block_split
from bio_params.dataset import Normalizer, TabularDataset
from bio_params.features import build_features, feature_names
from bio_params.loaders.bgc_argo import attach_surface_chla, load_bgc_argo
from bio_params.loaders.chla_no3 import load_chla_no3
from bio_params.loaders.combined import load_combined
from bio_params.model import MLP, MLPConfig
from bio_params.persist import save_artifact
from bio_params.profiles import add_mld, add_relative_target, kd_from_surface_chl
from bio_params.train import TrainConfig, train

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
DEFAULT_SPROF = ROOT / "data" / "bgc_argo" / "raw" / "floats"
MODEL_DIR = ROOT / "models" / "pretrained"
PROC = ROOT / "data" / "bgc_argo" / "processed"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default="bgc_argo", choices=["bgc_argo", "combined"])
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--sprof-dir", type=Path, default=DEFAULT_SPROF)
    p.add_argument("--matchup", type=Path, default=None,
                   help="Satellite matchup parquet (default per source)")
    p.add_argument("--surface-chla-feature", action="store_true",
                   help="Also feed satellite surface Chl-a as a shape feature "
                        "(requires a satellite match; restricts to matched rows)")
    p.add_argument("--per-source-max", type=int, default=None)
    p.add_argument("--per-source-profiles", type=int, default=None,
                   help="combined: keep at most N WHOLE profiles per source "
                        "(profile-level balance; applied after MLD/relative so "
                        "columns stay intact)")
    p.add_argument("--no-mld", action="store_true",
                   help="ablation: drop the MLD feature (keeps same rows)")
    p.add_argument("--with-no3", action="store_true",
                   help="co-located Chl-a+NO3 data; add raw NO3 as a feature")
    p.add_argument("--no3-light", action="store_true",
                   help="use light-gated NO3 (NO3*rel_light) instead of raw NO3 "
                        "(implies co-located data + Kd); -> 0 in the dark")
    p.add_argument("--rel-light", action="store_true",
                   help="add Beer-Lambert relative-light feature exp(-Kd*z) "
                        "(Kd from surface Chl); lets Chl vanish in the dark")
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--rel-cap", type=float, default=20.0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--block-deg", type=float, default=5.0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-hidden-layers", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-final", action="store_true")
    p.add_argument("--tag", default=None)
    p.add_argument("--log-every", type=int, default=25)
    return p.parse_args()


def _r2(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    ss_res = float(((pred - obs) ** 2).sum())
    ss_tot = float(((obs - obs.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    matchup = args.matchup or (PROC / ("satchl_matchup_combined.parquet"
              if args.source == "combined" else "satchl_matchup.parquet"))
    suffix = f"_{args.tag}" if args.tag else ""

    print(f"=== relative Chla profile  source={args.source}{suffix} ===")
    need_coloc = args.with_no3 or args.no3_light   # need an NO3 column
    use_no3_raw = args.with_no3 and not args.no3_light
    use_no3_lit = args.no3_light
    need_kd = args.rel_light or args.no3_light

    if need_coloc:
        df = load_chla_no3(args.source, glodap_csv=args.csv, sprof_dir=args.sprof_dir)
    elif args.source == "bgc_argo":
        df = load_bgc_argo(args.sprof_dir, "Chla")
    else:
        df = load_combined(args.csv, args.sprof_dir, target="Chla",
                           per_source_max=args.per_source_max, seed=args.seed)
    print(f"  loaded {len(df):,} rows")

    df = add_mld(df)
    df = add_relative_target(df, "Chla", rel_cap=args.rel_cap)
    df = df[np.isfinite(df["mld"])].reset_index(drop=True)
    if need_kd:
        # Per-profile attenuation Kd from the (in-situ) surface Chl.
        df["kd"] = kd_from_surface_chl(df["Chla_surf"].to_numpy())
    if need_coloc:
        df = df[np.isfinite(df["NO3"])].reset_index(drop=True)
    print(f"  {len(df):,} rows with MLD + relative target"
          + (" + NO3" if use_no3_raw else "")
          + (" + NO3*light" if use_no3_lit else "")
          + (" + rel_light" if args.rel_light else ""))

    # Satellite surface, for the absolute-reconstruction metric (and optionally
    # as a shape feature). Left merge keeps unmatched rows (abs metric skips them).
    df = attach_surface_chla(df, matchup)
    use_sat_feat = args.surface_chla_feature
    if use_sat_feat:
        df = df[np.isfinite(df["surface_chla"])].reset_index(drop=True)
        print(f"  {len(df):,} rows with satellite (surface_chla feature ON)")

    # Profile-level per-source balancing (keep whole profiles so MLD/relative,
    # already computed above, stay valid). Used to give GLODAP coastal profiles
    # real weight in the combined model (else BGC-Argo dwarfs them).
    if args.per_source_profiles and "source" in df.columns:
        keys = ["latitude", "longitude", "time"]
        rng = np.random.default_rng(args.seed)
        parts = []
        for src, sub in df.groupby("source"):
            profs = sub[keys].drop_duplicates()
            if len(profs) > args.per_source_profiles:
                profs = profs.iloc[np.sort(rng.choice(
                    len(profs), args.per_source_profiles, replace=False))]
            parts.append(sub.merge(profs, on=keys, how="inner"))
        df = pd.concat(parts, ignore_index=True)
        ng = int((df.source == "glodap").sum()); na = int((df.source == "bgc_argo").sum())
        print(f"  balanced to <= {args.per_source_profiles} profiles/source: "
              f"{len(df):,} rows (glodap={ng:,}, bgc_argo={na:,})")

    if args.subsample and len(df) > args.subsample:
        rng = np.random.default_rng(args.seed)
        df = df.iloc[np.sort(rng.choice(len(df), args.subsample, replace=False))].reset_index(drop=True)
        print(f"  subsampled to {len(df):,}")

    include_mld_feat = not args.no_mld
    feats = build_features(df, include_mld=include_mld_feat,
                           include_surface_chla=use_sat_feat, surface_chla_log=True,
                           include_rel_light=args.rel_light,
                           include_no3=use_no3_raw, include_no3_lit=use_no3_lit)
    fnames = feature_names(include_mld=include_mld_feat, include_surface_chla=use_sat_feat,
                           surface_chla_log=True, include_rel_light=args.rel_light,
                           include_no3=use_no3_raw, include_no3_lit=use_no3_lit)
    X = feats.to_numpy()
    y = df["Chla_rel"].to_numpy()
    chla_abs = df["Chla"].to_numpy()
    sat_surf = df["surface_chla"].to_numpy()
    lat = df["latitude"].to_numpy()
    lon = df["longitude"].to_numpy()
    print(f"  X {X.shape}  features {fnames}")
    print(f"  rel range [{y.min():.3g}, {y.max():.3g}]")

    train_cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                            early_stopping_patience=args.patience, log_every=args.log_every)
    mlp_cfg = MLPConfig(in_dim=X.shape[1], hidden=args.hidden,
                        n_hidden_layers=args.n_hidden_layers)

    folds = []
    print(f"\n--- {args.folds} folds x {args.block_deg} deg blocks ---")
    for k, tr, va in spatial_block_split(lat, lon, block_deg=args.block_deg,
                                         n_folds=args.folds, seed=args.seed):
        norm = Normalizer.fit(X[tr], y[tr])
        train_ds = TabularDataset(norm.transform_x(X[tr]), norm.transform_y(y[tr]))
        val_ds = TabularDataset(norm.transform_x(X[va]), norm.transform_y(y[va]))
        model = MLP(mlp_cfg)
        res = train(model, train_ds, val_ds, train_cfg)
        model.eval()
        dev = next(model.parameters()).device
        with torch.no_grad():
            pn = model(torch.from_numpy(norm.transform_x(X[va]).astype(np.float32)).to(dev)).cpu().numpy()
        rel_pred = np.clip(norm.inverse_transform_y(pn), 0.0, args.rel_cap)
        shape_r2 = _r2(y[va], rel_pred)
        # Absolute reconstruction with the satellite surface (deploy-relevant).
        msk = np.isfinite(sat_surf[va])
        abs_pred = rel_pred[msk] * sat_surf[va][msk]
        abs_obs = chla_abs[va][msk]
        abs_r2 = _r2(abs_obs, abs_pred)
        abs_rmse = float(np.sqrt(np.mean((abs_pred - abs_obs) ** 2))) if msk.any() else float("nan")
        folds.append(dict(fold=k, shape_r2=shape_r2, abs_r2=abs_r2, abs_rmse=abs_rmse,
                          n=int(len(va)), n_abs=int(msk.sum()), epochs=res.n_epochs_run))
        print(f"  fold {k}: shape R2={shape_r2:.4f}  abs R2={abs_r2:.4f}  "
              f"abs RMSE={abs_rmse:.3g}  (n={len(va):,}, epochs={res.n_epochs_run})")

    sr2 = np.array([f["shape_r2"] for f in folds])
    ar2 = np.array([f["abs_r2"] for f in folds])
    arm = np.array([f["abs_rmse"] for f in folds])
    print("\n=== CV summary ===")
    print(f"shape R2:  mean {sr2.mean():.4f}  median {np.median(sr2):.4f}")
    print(f"abs R2:    mean {ar2.mean():.4f}  median {np.median(ar2):.4f}")
    print(f"abs RMSE:  mean {arm.mean():.3g}")

    metrics_dir = (ROOT / "data" / ("combined" if args.source == "combined" else "bgc_argo")
                   / "processed")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(target="Chla", source=args.source, relative_target=True,
                   include_mld=include_mld_feat, include_rel_light=args.rel_light,
                   include_no3=use_no3_raw, include_no3_lit=use_no3_lit,
                   surface_chla_feature=use_sat_feat,
                   rel_cap=args.rel_cap, feature_names=fnames, n_rows=int(len(X)),
                   shape_r2_mean=float(sr2.mean()), shape_r2_median=float(np.median(sr2)),
                   abs_r2_mean=float(ar2.mean()), abs_r2_median=float(np.median(ar2)),
                   abs_rmse_mean=float(arm.mean()), folds=folds,
                   n_folds=args.folds, block_deg=args.block_deg)
    (metrics_dir / f"cv_Chla_rel{suffix}.json").write_text(json.dumps(payload, indent=2))
    print(f"Saved CV metrics -> {metrics_dir / f'cv_Chla_rel{suffix}.json'}")

    if args.no_final:
        return 0

    print("\n--- Final training on all rows ---")
    norm = Normalizer.fit(X, y)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X)); cut = int(0.9 * len(perm))
    train_ds = TabularDataset(norm.transform_x(X[perm[:cut]]), norm.transform_y(y[perm[:cut]]))
    val_ds = TabularDataset(norm.transform_x(X[perm[cut:]]), norm.transform_y(y[perm[cut:]]))
    final = MLP(mlp_cfg)
    fres = train(final, train_ds, val_ds, train_cfg)
    art = MODEL_DIR / f"{args.source}_Chla_rel{suffix}.pt"
    save_artifact(art, model=final, normalizer=norm, feature_names=fnames,
                  target_name="Chla_rel",
                  extra=dict(source=args.source, relative_target=True, include_mld=include_mld_feat,
                             include_rel_light=args.rel_light, include_no3=use_no3_raw,
                             include_no3_lit=use_no3_lit,
                             surface_chla_feature=use_sat_feat, rel_cap=args.rel_cap,
                             log_target=False, include_season=False,
                             surface_chla=use_sat_feat, surface_chla_log=True,
                             cv_shape_r2_mean=float(sr2.mean()),
                             cv_abs_r2_mean=float(ar2.mean()),
                             cv_abs_r2_median=float(np.median(ar2)),
                             n_rows=int(len(X)), epochs_run_final=fres.n_epochs_run))
    print(f"Saved model artifact -> {art}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
