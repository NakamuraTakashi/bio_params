"""Relative Chl-a model = typical base profile x bounded per-layer amplification
(option B). Replaces the light-saturation gate (which wrongly killed subsurface
DCMs) with a data-derived typical shape that already contains the DCM.

    rel_pred(z) = base(z; C_surf) * a,   a = A_max * sigmoid(MLP(features))

`base` is the surface-Chl-binned median rel(z) from build_chla_base_profile.py
(decays to 0 deep), and a in [0, A_max] is the bounded amplification. Bounded a
times a decaying base => rel -> 0 in the deep ocean BY CONSTRUCTION (no light
gate, no fake bottom Chl), while a(z, env) keeps the subsurface free to deviate
from the typical shape (raise/lower the DCM) driven by NO3/MLD/T.

Target: rel = Chla/Chla_surf (relative-only, so the BGC-Argo multiplicative bias
cancels). Amplitude is restored at inference from the satellite surface field.
Custom GPU loop (base is a per-row factor applied before the loss).

Artifacts: models/pretrained/<source>_Chla_baseamp[_<tag>].pt
CV metrics: data/<...>/processed/cv_Chla_baseamp[_<tag>].json

Usage:
    uv run python scripts/build_chla_base_profile.py        # build the base first
    uv run python scripts/train_chla_baseamp.py --source combined --per-source-profiles 10000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from bio_params.base_profile import BaseProfile, build_base_profile
from bio_params.cv import spatial_block_split
from bio_params.dataset import Normalizer
from bio_params.features import build_features, feature_names
from bio_params.loaders.chla_no3 import load_chla_no3
from bio_params.model import MLP, MLPConfig
from bio_params.persist import save_artifact
from bio_params.profiles import add_mld, add_relative_target

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
DEFAULT_SPROF = ROOT / "data" / "bgc_argo" / "raw" / "floats"
MODEL_DIR = ROOT / "models" / "pretrained"
BASE_JSON = ROOT / "data" / "combined" / "processed" / "chla_base_profile.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default="combined", choices=["bgc_argo", "glodap", "combined"])
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--sprof-dir", type=Path, default=DEFAULT_SPROF)
    p.add_argument("--per-source-profiles", type=int, default=None)
    p.add_argument("--rel-cap", type=float, default=20.0)
    p.add_argument("--a-max", type=float, default=5.0, help="upper bound on the amplification a")
    p.add_argument("--surface-chla", action="store_true",
                   help="add log surface Chl-a as a feature (continuous trophic "
                        "state, beyond the discrete base bin). Uses the DAILY "
                        "satellite matchup (consistent with ROMS --satellite daily); "
                        "profiles without a daily match are dropped.")
    p.add_argument("--matchup", type=Path,
                   default=ROOT / "data" / "bgc_argo" / "processed" / "satchl_matchup_daily_combined.parquet",
                   help="daily satellite surface-Chl matchup parquet (--surface-chla / --bin-satellite)")
    p.add_argument("--bin-satellite", action="store_true",
                   help="bin the base profile by the DAILY satellite surf (matched), "
                        "matching ROMS inference; needs a base table built the same way.")
    p.add_argument("--base-json", type=Path, default=None,
                   help="base-profile table JSON (default: chla_base_profile.json)")
    p.add_argument("--strat-features", action="store_true",
                   help="add pycnocline depth (log z_pyc) + stratification strength "
                        "(max dsigma/dz) as features (from T/S column)")
    p.add_argument("--nutricline-features", action="store_true",
                   help="add nutricline depth (log z_nutr) + sharpness (max dNO3/dz) "
                        "as features (from the NO3 column)")
    p.add_argument("--struct-filter-only", action="store_true",
                   help="compute+filter on structure descriptors (all finite) but add "
                        "no features (baseline on the same rows for a fair ablation)")
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
    p.add_argument("--dark-correct", action="store_true",
                   help="subtract the per-float BGC-Argo fluorescence dark offset from CHLA")
    p.add_argument("--box", type=float, nargs=4, default=None,
                   metavar=("LON0", "LON1", "LAT0", "LAT1"),
                   help="restrict to a region, e.g. --box 120 160 20 50 (Japan)")
    p.add_argument("--tag", default=None)
    return p.parse_args()


def _r2(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    ss = float(((pred - obs) ** 2).sum()); tot = float(((obs - obs.mean()) ** 2).sum())
    return 1.0 - ss / tot if tot > 0 else float("nan")


def train_baseamp(cfg, Xn, base_col, y, tr, va, device, *, a_max, epochs, batch, lr, patience):
    """Train rel = base * (a_max * sigmoid(MLP)); MSE on rel."""
    Xt = torch.as_tensor(Xn, dtype=torch.float32, device=device)
    bt = torch.as_tensor(base_col, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    tr_i = torch.as_tensor(tr, dtype=torch.long, device=device)
    va_i = torch.as_tensor(va, dtype=torch.long, device=device)
    model = MLP(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def rel_of(idx):
        a = a_max * torch.sigmoid(model(Xt[idx]).squeeze(-1))
        return bt[idx] * a

    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        perm = tr_i[torch.randperm(len(tr_i), device=device)]
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            loss = F.mse_loss(rel_of(b), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(F.mse_loss(rel_of(va_i), yt[va_i]))
        if vl < best - 1e-7:
            best = vl; bad = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, ep + 1


def predict_rel(model, Xn, base_col, device, a_max, rel_cap, batch=200_000):
    model.eval(); rels = []
    with torch.no_grad():
        for i in range(0, len(Xn), batch):
            xb = torch.as_tensor(Xn[i:i + batch], dtype=torch.float32, device=device)
            a = a_max * torch.sigmoid(model(xb).squeeze(-1)).cpu().numpy()
            rels.append(base_col[i:i + batch] * a)
    return np.clip(np.concatenate(rels), 0.0, rel_cap)


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    suffix = f"_{args.tag}" if args.tag else ""
    print(f"=== base x amplification Chla  source={args.source}{suffix} ===")
    base_json = args.base_json or BASE_JSON
    if not base_json.exists():
        print(f"ERROR: base table missing ({base_json}); run build_chla_base_profile.py")
        return 1
    base = BaseProfile.from_dict(json.loads(base_json.read_text()))

    df = load_chla_no3(args.source, glodap_csv=args.csv, sprof_dir=args.sprof_dir,
                       dark_correct=args.dark_correct,
                       box=tuple(args.box) if args.box else None)
    print(f"  loaded {len(df):,} co-located rows")
    df = add_mld(df)
    df = add_relative_target(df, "Chla", rel_cap=args.rel_cap)
    df = df[np.isfinite(df["mld"]) & np.isfinite(df["NO3"])].reset_index(drop=True)

    if args.surface_chla or args.bin_satellite:
        from bio_params.loaders.bgc_argo import attach_surface_chla
        n0 = len(df)
        df = attach_surface_chla(df, args.matchup)
        df = df[np.isfinite(df["surface_chla"])].reset_index(drop=True)
        print(f"  daily satellite surface Chl attached: {len(df):,}/{n0:,} rows matched"
              + ("  [bin key]" if args.bin_satellite else ""))

    if args.per_source_profiles and "source" in df.columns:
        keys = ["latitude", "longitude", "time"]; rng = np.random.default_rng(args.seed)
        parts = []
        for src, sub in df.groupby("source"):
            pr = sub[keys].drop_duplicates()
            if len(pr) > args.per_source_profiles:
                pr = pr.iloc[np.sort(rng.choice(len(pr), args.per_source_profiles, replace=False))]
            parts.append(sub.merge(pr, on=keys, how="inner"))
        df = pd.concat(parts, ignore_index=True)
        ng = int((df.source == "glodap").sum()); na = int((df.source == "bgc_argo").sum())
        print(f"  balanced <= {args.per_source_profiles} profiles/source: {len(df):,} rows (glodap={ng:,}, bgc={na:,})")

    if args.strat_features or args.nutricline_features or args.struct_filter_only:
        from bio_params.profiles import add_structure_descriptors
        n0 = len(df); df = add_structure_descriptors(df)
        # filter on ALL descriptors (same rows across the ablation configs)
        need = ["z_pyc", "strat_max", "z_nutr", "nutr_max"]
        df = df[np.all([np.isfinite(df[c].to_numpy()) for c in need], axis=0)].reset_index(drop=True)
        print(f"  structure descriptors (all finite): {len(df):,}/{n0:,} rows")

    fnames = feature_names(include_mld=True, include_no3=True,
                           include_surface_chla=args.surface_chla, surface_chla_log=True)
    X = build_features(df, include_mld=True, include_no3=True,
                       include_surface_chla=args.surface_chla, surface_chla_log=True).to_numpy()
    if args.strat_features:
        X = np.column_stack([X, np.log(df["z_pyc"].to_numpy() + 1.0), df["strat_max"].to_numpy()])
        fnames = fnames + ["log_z_pyc", "strat_max"]
    if args.nutricline_features:
        X = np.column_stack([X, np.log(df["z_nutr"].to_numpy() + 1.0), df["nutr_max"].to_numpy()])
        fnames = fnames + ["log_z_nutr", "nutr_max"]
    bin_surf = (df["surface_chla"] if args.bin_satellite else df["Chla_surf"]).to_numpy()
    base_col = base.eval(bin_surf, df["depth"].to_numpy())   # satellite bin if --bin-satellite
    y = df["Chla_rel"].to_numpy()
    surf_insitu = df["Chla_surf"].to_numpy()
    lat = df["latitude"].to_numpy(); lon = df["longitude"].to_numpy()
    print(f"  X {X.shape}  features {fnames}  A_max={args.a_max}  base nodes={len(base.depth_grid)}")

    cfg = MLPConfig(in_dim=X.shape[1], hidden=args.hidden,
                    n_hidden_layers=args.n_hidden_layers, out_dim=1)
    folds = []
    print(f"\n--- {args.folds} folds x {args.block_deg} deg blocks ---")
    for k, tr, va in spatial_block_split(lat, lon, block_deg=args.block_deg, n_folds=args.folds, seed=args.seed):
        norm = Normalizer.fit(X[tr], y[tr])
        norm = Normalizer(x_mean=norm.x_mean, x_std=np.where(norm.x_std == 0, 1.0, norm.x_std),
                          y_mean=0.0, y_std=1.0)  # identity y; base is in native rel space
        Xn = norm.transform_x(X)
        model, eps = train_baseamp(cfg, Xn, base_col, y, tr, va, device, a_max=args.a_max,
                                   epochs=args.epochs, batch=args.batch_size, lr=args.lr,
                                   patience=args.patience)
        rel_pred = predict_rel(model, Xn[va], base_col[va], device, args.a_max, args.rel_cap)
        shape_r2 = _r2(y[va], rel_pred)
        abs_r2 = _r2(df["Chla"].to_numpy()[va], rel_pred * surf_insitu[va])
        folds.append(dict(fold=k, shape_r2=shape_r2, abs_r2=abs_r2, n=int(len(va)), epochs=eps))
        print(f"  fold {k}: shape R2={shape_r2:.4f}  abs R2(insitu surf)={abs_r2:.4f}  (epochs={eps})")

    sr2 = np.array([f["shape_r2"] for f in folds]); ar2 = np.array([f["abs_r2"] for f in folds])
    print(f"\n=== CV ===  shape R2 mean {sr2.mean():.4f} median {np.median(sr2):.4f} | "
          f"abs R2 mean {ar2.mean():.4f}")

    metrics_dir = ROOT / "data" / ("combined" if args.source == "combined" else "bgc_argo") / "processed"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"cv_Chla_baseamp{suffix}.json").write_text(json.dumps(dict(
        target="Chla", source=args.source, base_amp=True, relative_target=True,
        a_max=args.a_max, include_mld=True, include_no3=True,
        surface_chla=args.surface_chla, surface_chla_log=True, bin_satellite=args.bin_satellite,
        feature_names=fnames, rel_cap=args.rel_cap, shape_r2_mean=float(sr2.mean()),
        shape_r2_median=float(np.median(sr2)), abs_r2_mean=float(ar2.mean()), folds=folds), indent=2))
    print(f"Saved CV -> {metrics_dir / f'cv_Chla_baseamp{suffix}.json'}")
    if args.no_final:
        return 0

    print("\n--- Final training on all rows ---")
    norm = Normalizer.fit(X, y)
    norm = Normalizer(x_mean=norm.x_mean, x_std=np.where(norm.x_std == 0, 1.0, norm.x_std),
                      y_mean=0.0, y_std=1.0)
    Xn = norm.transform_x(X)
    rng = np.random.default_rng(args.seed); perm = rng.permutation(len(X)); cut = int(0.9 * len(perm))
    model, eps = train_baseamp(cfg, Xn, base_col, y, perm[:cut], perm[cut:], device, a_max=args.a_max,
                               epochs=args.epochs, batch=args.batch_size, lr=args.lr, patience=args.patience)
    art = MODEL_DIR / f"{args.source}_Chla_baseamp{suffix}.pt"
    save_artifact(art, model=model, normalizer=norm, feature_names=fnames, target_name="Chla_rel",
                  extra=dict(source=args.source, base_amp=True, relative_target=True,
                             a_max=args.a_max, base_profile=base.to_dict(),
                             include_mld=True, include_no3=True,
                             surface_chla=args.surface_chla, surface_chla_log=True,
                             bin_satellite=args.bin_satellite, rel_cap=args.rel_cap,
                             log_target=False, include_season=False,
                             cv_shape_r2_mean=float(sr2.mean()), cv_abs_r2_mean=float(ar2.mean()),
                             n_rows=int(len(X)), epochs_run_final=eps))
    print(f"Saved model -> {art}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
