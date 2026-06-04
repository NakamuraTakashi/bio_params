"""Stratified scoreboard for gated relative Chl-a models, on the GLODAP
co-located rows (unbiased absolute reference; BGC-Argo fluorescence has a
multiplicative bias so absolute eval is GLODAP-only).

For each model tag it reconstructs the absolute profile the training-consistent
way -- pred_abs(z) = rel_pred(z) * Chla_surf(in-situ) with the gate's Kd from the
in-situ surface Chl (Morel, or the model's fixed Ze) and, if the model was
trained with --seasonal-light, the E0(lat,doy) surface-light factor -- then
reports, by depth band:
  * UNDER rate = P(pred<=0.1 | obs>=1)   (the deep-DCM drop-out KPI)
  * log10-space RMSE and bias (median pred/obs)
and the same split into oligotrophic vs productive columns (by surface Chl).

NOTE: this is an in-sample eval (the final models saw these rows); both models
are scored identically, so it is a fair *relative* comparison of the change,
not an out-of-sample skill estimate.

Usage:
    uv run python scripts/eval_chla_scoreboard.py --tags base seaslight
    uv run python scripts/eval_chla_scoreboard.py --tags base seaslight ikhead
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from bio_params.base_profile import BaseProfile
from bio_params.features import build_features, feature_names
from bio_params.loaders.chla_no3 import load_chla_no3
from bio_params.persist import load_artifact
from bio_params.profiles import (add_mld, add_relative_target,
                                  daily_insolation_factor, kd_from_surface_chl)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "pretrained"
DEFAULT_CSV = ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
DAILY_MATCHUP = ROOT / "data" / "bgc_argo" / "processed" / "satchl_matchup_daily_combined.parquet"
DEPTH_BANDS = [(0, 10), (10, 30), (30, 75), (75, 150), (150, 300), (300, 600)]


def _rel_pred(art_path, df, device):
    """Reconstruct rel(z) for one model (gated or base x amplification),
    honoring its extra flags (features built per the model). Returns
    (rel, config_string)."""
    model, norm, meta = load_artifact(art_path, map_location=device)
    model.to(device).eval()
    e = meta["extra"]
    rel_cap = float(e.get("rel_cap", 20.0))
    surf = df["Chla_surf"].to_numpy()
    sfc = bool(e.get("surface_chla", False))
    if sfc:
        # surface-Chl models are trained on the daily satellite matchup, so feed
        # the same here (rows without a daily match -> NaN -> excluded).
        df = df.assign(surface_chla=df["_sat_daily"].to_numpy())
    X = build_features(df, include_mld=bool(e.get("include_mld", True)),
                       include_no3=bool(e.get("include_no3", True)),
                       include_surface_chla=sfc,
                       surface_chla_log=bool(e.get("surface_chla_log", True))).to_numpy()
    Xn = norm.transform_x(X)
    with torch.no_grad():
        out = model(torch.as_tensor(Xn, dtype=torch.float32, device=device)).cpu().numpy()

    if e.get("base_amp"):
        a_max = float(e.get("a_max", 5.0))
        base = BaseProfile.from_dict(e["base_profile"])
        # bin by the daily satellite surf if the model was built that way (matches
        # ROMS inference), else by in-situ surface.
        bin_surf = df["_sat_daily"].to_numpy() if e.get("bin_satellite") else surf
        base_col = base.eval(bin_surf, df["depth"].to_numpy())
        a = a_max * (1.0 / (1.0 + np.exp(-out.ravel())))
        rel = np.clip(base_col * a, 0.0, rel_cap)
        return rel, f"base_amp A_max={a_max}" + (" sat-bin" if e.get("bin_satellite") else "")

    # light-gated model
    ik_head = bool(e.get("ik_head", False))
    gate_ik = float(e.get("gate_ik", 0.005))
    gate_ik_ref = float(e.get("gate_ik_ref", 0.02))
    fixed_ze = e.get("gate_fixed_ze", None)
    seasonal = bool(e.get("seasonal_light", False))
    kd = (np.full(len(df), np.log(100.0) / fixed_ze) if fixed_ze
          else kd_from_surface_chl(surf))
    rl = np.exp(-kd * df["depth"].to_numpy())
    if seasonal:
        rl = daily_insolation_factor(df["latitude"].to_numpy(), df["_doy"].to_numpy()) * rl
    if ik_head:
        g = np.logaddexp(0.0, out[:, 0]); ik = gate_ik_ref * np.exp(np.clip(out[:, 1], -5, 5))
    else:
        g = np.logaddexp(0.0, out.ravel()); ik = gate_ik
    rel = np.clip(g * np.tanh(rl / ik), 0.0, rel_cap)
    cfg = (f"gate seasonal={seasonal}" + (f" fixed_ze={fixed_ze}" if fixed_ze else " Kd=Morel"))
    return rel, cfg


def _log_metrics(obs, pred, floor=0.02):
    m = (obs >= floor) & np.isfinite(pred) & (pred >= 0)
    if m.sum() < 5:
        return np.nan, np.nan, int(m.sum())
    lo = np.log10(obs[m]); lp = np.log10(np.clip(pred[m], 1e-4, None))
    rmse = float(np.sqrt(np.mean((lp - lo) ** 2)))
    bias = float(np.median(pred[m] / obs[m]))
    return rmse, bias, int(m.sum())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tags", nargs="+", default=[],
                   help="gated model tags -> models/pretrained/combined_Chla_gated_<tag>.pt")
    p.add_argument("--models", nargs="+", default=[],
                   help="full model stems -> models/pretrained/<stem>.pt "
                        "(e.g. combined_Chla_baseamp_uitz); auto-detects base_amp vs gated")
    p.add_argument("--rel-cap", type=float, default=20.0)
    p.add_argument("--matched-only", action="store_true",
                   help="restrict eval to rows with a daily satellite match "
                        "(same rows for all models = apples-to-apples)")
    args = p.parse_args()
    entries = [(t, MODEL_DIR / f"combined_Chla_gated_{t}.pt") for t in args.tags]
    entries += [(m, MODEL_DIR / f"{m}.pt") for m in args.models]
    if not entries:
        p.error("give --tags and/or --models")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_chla_no3("glodap", glodap_csv=DEFAULT_CSV)
    df = add_mld(df)
    df = add_relative_target(df, "Chla", rel_cap=args.rel_cap)
    df = df[np.isfinite(df["mld"]) & np.isfinite(df["NO3"])
            & np.isfinite(df["Chla"])].reset_index(drop=True)
    df["_doy"] = df["time"].dt.dayofyear
    # daily satellite surface Chl for surface-Chl models (matched feature, fair eval)
    mday = pd.read_parquet(DAILY_MATCHUP)[["latitude", "longitude", "time", "surface_chla"]]
    mday["time"] = pd.to_datetime(mday["time"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.merge(mday.rename(columns={"surface_chla": "_sat_daily"}),
                  on=["latitude", "longitude", "time"], how="left")
    if args.matched_only:
        df = df[np.isfinite(df["_sat_daily"])].reset_index(drop=True)
    obs = df["Chla"].to_numpy(); surf = df["Chla_surf"].to_numpy()
    depth = df["depth"].to_numpy()
    print(f"GLODAP co-located eval rows: {len(df):,}  (obs>=1: {int((obs>=1).sum()):,})  "
          f"| daily-satellite matched: {int(np.isfinite(df['_sat_daily']).sum()):,}")

    for name, art in entries:
        if not art.exists():
            print(f"\n[{name}] MISSING {art.name}"); continue
        rel, cfg = _rel_pred(art, df, device)
        pred = rel * surf
        tag = name
        rmse, bias, n = _log_metrics(obs, pred)
        print(f"\n=== [{tag}]  {cfg} ===")
        print(f"  global: log10-RMSE {rmse:.3f}  bias(med pred/obs) {bias:.2f}  (n>=0.02 = {n:,})")
        hi = (obs >= 1) & np.isfinite(pred)   # score only where the model produced a value
        und = hi & (pred <= 0.1)
        print(f"  UNDER (obs>=1 & pred<=0.1): {int(und.sum()):,}/{int(hi.sum()):,} "
              f"= {100*und.sum()/max(hi.sum(),1):.1f}%")
        print(f"  {'depth band':>12} | {'n(obs>=1)':>9} | {'UNDER%':>7} | {'logRMSE':>7} | {'bias':>5}")
        for lo, hib in DEPTH_BANDS:
            inb = (depth >= lo) & (depth < hib)
            hib_m = inb & hi
            r, b, _ = _log_metrics(obs[inb], pred[inb])
            ur = 100 * (hib_m & (pred <= 0.1)).sum() / max(hib_m.sum(), 1)
            print(f"  {f'{lo}-{hib}m':>12} | {int(hib_m.sum()):>9} | {ur:>6.1f} | "
                  f"{r:>7.3f} | {b:>5.2f}")
        # oligotrophic vs productive by surface Chl
        for label, msk in [("oligotrophic surf<0.3", surf < 0.3),
                           ("productive  surf>1.0", surf > 1.0)]:
            r, b, nn = _log_metrics(obs[msk], pred[msk])
            hm = msk & hi; ur = 100 * (hm & (pred <= 0.1)).sum() / max(hm.sum(), 1)
            print(f"  {label}: logRMSE {r:.3f}  bias {b:.2f}  UNDER {ur:.1f}%  (n>=0.02={nn:,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
