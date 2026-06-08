"""Observation-vs-model scatter plots at the GLODAP full-set stations, using the
cached FORA-JPN60 T/S profiles (scripts/extract_fora_glodap_matchup.py).

For each GLODAP sample the FORA T/S profile (nearest grid cell, same date) is
interpolated to the sample depth and fed to the biogeochemical MLP; the predicted
value is scattered against the GLODAP observation. T and S panels compare the
FORA reanalysis directly with the GLODAP CTD (the model's input quality). O2/NO3
use the combined model, the rest GLODAP; low-salinity regression is applied to
NO3/PO4/SiO4/TA. Chl-a uses the plain allfeat model (NO3 + nutricline +
pycnocline + MLD from the FORA column, NO3 low-sal corrected, 200 m cutoff).

Usage:
    uv run python scripts/plot_fora_glodap_scatter.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr

from bio_params.features import build_features
from bio_params.persist import load_artifact
from bio_params.profiles import sigma0

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from predict_roms_ini_depths import (                      # noqa: E402
    TRACER_META, SALINITY_REGRESSION, blend_low_salinity, predict_field,
    _grad_peak_field, _nearest_index)
from predict_fora_chla import fora_mld, predict_no3_field   # noqa: E402

PROC = ROOT / "data" / "glodap" / "processed"
MODEL_DIR = ROOT / "models" / "pretrained"
OUT_DIR = ROOT / "figures" / "fora"
SOURCE = {"O2": "combined", "NO3": "combined", "TA": "glodap", "DIC": "glodap",
          "SiO4": "glodap", "PO4": "glodap"}
CORE = ["TA", "DIC", "O2", "NO3", "PO4", "SiO4"]
UNITS = {"T": "degC", "S": "PSU", "TA": "umol/kg", "DIC": "umol/kg", "O2": "umol/kg",
         "NO3": "umol/kg", "PO4": "umol/kg", "SiO4": "umol/kg", "Chla": "mg/m3"}


def interp_profile(depth_axis, prof, target_depths):
    """Linear interp of a FORA column to sample depths; NaN beyond finite range."""
    fin = np.isfinite(prof)
    if fin.sum() < 2:
        return np.full(len(target_depths), np.nan)
    da, pr = depth_axis[fin], prof[fin]
    out = np.interp(target_depths, da, pr, left=pr[0], right=np.nan)
    out[target_depths > da[-1] + 1.0] = np.nan          # below FORA bottom
    return out


def scatter_panel(ax, obs, pred, name, log=False):
    m = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs[m], pred[m]
    if log:
        m2 = (o > 0) & (p > 0); o, p = o[m2], p[m2]
    if len(o) < 2:
        ax.set_title(f"{name}: n<2"); return
    ax.scatter(o, p, s=6, alpha=0.35, edgecolors="none", c="tab:blue")
    lo = float(min(o.min(), p.min())); hi = float(max(o.max(), p.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    r2 = 1.0 - np.sum((p - o) ** 2) / np.sum((o - o.mean()) ** 2)
    rmse = float(np.sqrt(np.mean((p - o) ** 2))); bias = float(np.mean(p - o))
    txt = f"n={len(o)}\nR2={r2:.3f}\nRMSE={rmse:.3g}\nbias={bias:+.3g}"
    if log:
        lr2 = 1.0 - (np.sum((np.log10(p) - np.log10(o)) ** 2)
                     / np.sum((np.log10(o) - np.log10(o).mean()) ** 2))
        txt += f"\nlogR2={lr2:.3f}"
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.set_xlabel(f"GLODAP obs ({UNITS[name]})"); ax.set_ylabel(f"model ({UNITS[name]})")
    ax.set_title(name, fontsize=11)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", "box")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--linear", action="store_true",
                    help="use linear axes for every panel (default: log axes for "
                         "NO3/PO4/SiO4/Chl-a) and write a separate file")
    ap.add_argument("--surface-anchor", action="store_true",
                    help="surface-anchor the Chl-a panel to the DAILY satellite Chl-a "
                         "(per station lat/lon/date): R=clip(sat/model_surf, clip) "
                         "tapered to 1 by --anchor-taper m")
    ap.add_argument("--anchor-taper", type=float, default=100.0)
    ap.add_argument("--anchor-clip", type=float, nargs=2, default=[0.2, 5.0])
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prof = xr.open_dataset(PROC / "fora_glodap_profiles.nc")
    samp = pd.read_parquet(PROC / "fora_glodap_samples.parquet")
    fdepth = prof["fora_depth"].values
    sid_index = {s: k for k, s in enumerate(prof["station"].values.tolist())}

    # FORA T/S interpolated to each sample depth
    samp = samp[samp.station_id.isin(sid_index)].reset_index(drop=True)
    Tf = np.full(len(samp), np.nan); Sf = np.full(len(samp), np.nan)
    Tp = prof["fora_temp"].values; Sp = prof["fora_salt"].values
    for sid, g in samp.groupby("station_id"):
        k = sid_index[sid]; idx = g.index.to_numpy()
        Tf[idx] = interp_profile(fdepth, Tp[k], g.depth.to_numpy())
        Sf[idx] = interp_profile(fdepth, Sp[k], g.depth.to_numpy())
    samp["T_fora"] = Tf; samp["S_fora"] = Sf
    ok = np.isfinite(Tf) & np.isfinite(Sf)
    print(f"matched samples with FORA T/S: {int(ok.sum())}/{len(samp)}", flush=True)

    panels = {}
    panels["T"] = (samp.T_obs.to_numpy(), samp.T_fora.to_numpy(), False)
    panels["S"] = (samp.S_obs.to_numpy(), samp.S_fora.to_numpy(), False)

    base_df = pd.DataFrame({"latitude": samp.obs_lat, "longitude": samp.obs_lon,
                            "depth": samp.depth, "temperature": samp.T_fora,
                            "salinity": samp.S_fora})
    valid = ok.copy()
    for tgt in CORE:
        art = MODEL_DIR / f"{SOURCE[tgt]}_{tgt}.pt"
        model, norm, meta = load_artifact(art, map_location=device); model.to(device)
        log_t = bool(meta["extra"].get("log_target", False))
        pred = np.full(len(samp), np.nan)
        X = build_features(base_df).to_numpy()
        pv = predict_field(model, norm, X[valid], device, log_target=log_t,
                           clip=TRACER_META[tgt]["clip"])
        if tgt in SALINITY_REGRESSION:
            pv, _ = blend_low_salinity(pv, samp.S_fora.to_numpy()[valid], tgt)
        pred[valid] = pv
        panels[tgt] = (samp[tgt].to_numpy(), pred, tgt in ("NO3", "PO4", "SiO4"))
        print(f"  {tgt}: {SOURCE[tgt]} model "
              f"({'low-sal ' if tgt in SALINITY_REGRESSION else ''}n={int(valid.sum())})", flush=True)

    # --- Chl-a: allfeat with structure descriptors from the FORA column ---
    print("  Chla: allfeat (structure from FORA column, NO3 low-sal) ...", flush=True)
    no3_model, no3_norm, _ = load_artifact(MODEL_DIR / "combined_NO3.pt", map_location=device)
    no3_model.to(device)
    chla_model, chla_norm, chla_meta = load_artifact(MODEL_DIR / "combined_Chla_allfeat.pt",
                                                     map_location=device); chla_model.to(device)
    cutoff = float(chla_meta["extra"]["cutoff_depth"])
    nst, nlev = Tp.shape
    d3 = np.broadcast_to(fdepth[:, None, None], (nlev, nst, 1))
    lat3 = np.broadcast_to(prof["obs_lat"].values[None, :, None], (nlev, nst, 1))
    sig = sigma0(Sp.T.ravel(), Tp.T.ravel(), d3.ravel(), lat3.ravel()).reshape(nlev, nst, 1)
    mld = fora_mld(fdepth, sig).ravel()                       # (nst,)
    z_pyc, strat_max = (a.ravel() for a in _grad_peak_field(d3, sig))
    LATc = prof["obs_lat"].values; LONc = prof["obs_lon"].values
    no3_prof = np.stack([predict_no3_field(Tp[:, lv], Sp[:, lv], LATc, LONc, fdepth[lv],
                                           no3_model, no3_norm, device, True)
                         for lv in range(nlev)])              # (nlev, nst)
    z_nutr, nutr_max = (a.ravel() for a in _grad_peak_field(d3, no3_prof[:, :, None]))

    chla_pred = np.full(len(samp), np.nan)
    vc = valid & np.isfinite(samp.Chla.to_numpy())
    if vc.any():
        sub = samp[vc]
        kk = sub.station_id.map(sid_index).to_numpy()
        no3_feat = predict_field(no3_model, no3_norm,
                                 build_features(base_df[vc]).to_numpy(), device, clip=(0.0, 60.0))
        no3_feat, _ = blend_low_salinity(no3_feat, sub.S_fora.to_numpy(), "NO3")
        df2 = pd.DataFrame({"latitude": sub.obs_lat, "longitude": sub.obs_lon,
                            "depth": sub.depth, "temperature": sub.T_fora,
                            "salinity": sub.S_fora, "mld": mld[kk], "NO3": no3_feat})
        X = build_features(df2, include_mld=True, include_no3=True).to_numpy()
        X = np.column_stack([X, np.log(z_nutr[kk] + 1.0), nutr_max[kk],
                             np.log(z_pyc[kk] + 1.0), strat_max[kk]])
        pv = predict_field(chla_model, chla_norm, X, device, clip=TRACER_META["Chla"]["clip"])
        pv = np.where(sub.depth.to_numpy() > cutoff, 0.0, pv)
        chla_pred[np.where(vc)[0]] = pv

    if args.surface_anchor:
        from bio_params.satellite import chla_day_field
        meta_st = pd.DataFrame({"station_id": prof["station"].values,
                                "lat": prof["obs_lat"].values, "lon": prof["obs_lon"].values,
                                "date": pd.to_datetime(prof["date"].values)})
        sat_st = {}
        for d, gst in meta_st.groupby(meta_st.date.dt.normalize()):
            try:
                la, lo, arr, _ = chla_day_field(pd.Timestamp(d).strftime("%Y-%m-%d"))
            except Exception:
                continue
            il = _nearest_index(gst.lat.to_numpy(), la)
            io = _nearest_index(((gst.lon.to_numpy() + 180) % 360) - 180, lo)
            for sid, v in zip(gst.station_id, arr[il, io]):
                sat_st[sid] = float(v)
        lo_c, hi_c = args.anchor_clip
        samp["_cm"] = chla_pred
        anchored = chla_pred.copy()
        for sid, gi in samp.groupby("station_id"):
            ssurf = sat_st.get(sid, np.nan)
            g = gi.sort_values("depth"); mvals = g["_cm"].to_numpy()
            fin = np.isfinite(mvals) & (mvals > 1e-3)
            if not (np.isfinite(ssurf) and fin.any()):
                continue
            msurf = mvals[int(np.argmax(fin))]
            rhat = float(np.clip(ssurf / msurf, lo_c, hi_c))
            z = g.depth.to_numpy()
            reff = 1.0 + (rhat - 1.0) * np.clip((args.anchor_taper - z) / args.anchor_taper, 0.0, 1.0)
            anchored[g.index.to_numpy()] = mvals * reff
        chla_pred = anchored
        print(f"  Chla surface-anchored to DAILY satellite (taper->{args.anchor_taper:.0f}m, "
              f"clip[{lo_c},{hi_c}]); stations matched: {len(sat_st)}", flush=True)
    panels["Chla"] = (samp.Chla.to_numpy(), chla_pred, True)

    # --- figure: 3x3 grid (T, S, 6 core, Chla) ---
    order = ["T", "S", "TA", "DIC", "O2", "NO3", "PO4", "SiO4", "Chla"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for ax, name in zip(axes.ravel(), order):
        obs, pred, log = panels[name]
        scatter_panel(ax, obs, pred, name, log=(log and not args.linear))
    axes_note = "linear axes" if args.linear else "log axes for NO3/PO4/SiO4/Chl-a"
    chla_note = "allfeat + daily-satellite surface anchor" if args.surface_anchor else "plain allfeat"
    fig.suptitle("GLODAP obs vs model on FORA-JPN60 T/S (full-set stations, 1982-2020)\n"
                 "T/S = FORA vs CTD; biogeochem from FORA T/S; low-sal corrected NO3/PO4/SiO4/TA; "
                 f"Chl-a = {chla_note}  [{axes_note}]", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    stem = "fora_glodap_scatter" + ("_linear" if args.linear else "") + (
        "_anchored" if args.surface_anchor else "")
    out = OUT_DIR / f"{stem}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
