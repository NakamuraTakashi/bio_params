"""Per-site observed-vs-model vertical profiles of NO3 and Chl-a at representative
GLODAP full-set stations (one each from the low/mid/high-latitude Pacific, the Sea
of Japan, and the East China Sea), using the cached FORA-JPN60 T/S column as model
input. NO3 = combined model (low-sal corrected); Chl-a = plain allfeat (nutricline
/ pycnocline / MLD from the FORA column, NO3 low-sal corrected, 200 m cutoff).

Usage:
    uv run python scripts/plot_fora_glodap_site_profiles.py
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
    TRACER_META, blend_low_salinity, predict_field, _grad_peak_field)
from predict_fora_chla import fora_mld, predict_no3_field   # noqa: E402

PROC = ROOT / "data" / "glodap" / "processed"
MODEL_DIR = ROOT / "models" / "pretrained"
OUT_DIR = ROOT / "figures" / "fora"
# (region label, station_id) — chosen from the cached full-set stations
SITES = [
    ("Pacific low-lat", "555_910.0"),
    ("Pacific mid-lat", "2065_5332.0"),
    ("Pacific high-lat", "431_3.0"),
    ("Sea of Japan", "2068_5490.0"),
    ("East China Sea", "217_14.0"),
]


def struct_for_column(d, Tcol, Scol, lat0, lon0, no3_model, no3_norm, device):
    """MLD, pycnocline, nutricline scalars from one FORA column (NO3 low-sal)."""
    M = len(d)
    fin = np.isfinite(Tcol) & np.isfinite(Scol)
    d3 = d[:, None, None]
    sig = np.full((M, 1, 1), np.nan)
    sig[fin, 0, 0] = sigma0(Scol[fin], Tcol[fin], d[fin], np.full(fin.sum(), lat0))
    mld = float(fora_mld(d, sig).ravel()[0])
    z_pyc, strat_max = (float(a.ravel()[0]) for a in _grad_peak_field(d3, sig))
    # NO3 profile (per level: depth differs, so predict level by level)
    no3col = np.full(M, np.nan)
    for lv in range(M):
        if fin[lv]:
            no3col[lv:lv + 1] = predict_no3_field(Tcol[lv:lv + 1], Scol[lv:lv + 1],
                                                  np.array([lat0]), np.array([lon0]),
                                                  d[lv], no3_model, no3_norm, device, True)
    z_nutr, nutr_max = (float(a.ravel()[0]) for a in _grad_peak_field(d3, no3col[:, None, None]))
    return mld, z_pyc, strat_max, z_nutr, nutr_max


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full-depth", action="store_true",
                    help="extend the model profile and depth axis to the FORA "
                         "seafloor (deepest finite level) instead of the obs range")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prof = xr.open_dataset(PROC / "fora_glodap_profiles.nc")
    samp = pd.read_parquet(PROC / "fora_glodap_samples.parquet")
    fd = prof["fora_depth"].values
    sids = prof["station"].values.tolist()
    sidx = {s: k for k, s in enumerate(sids)}
    Tp = prof["fora_temp"].values; Sp = prof["fora_salt"].values
    la = prof["obs_lat"].values; lo = prof["obs_lon"].values
    dates = pd.to_datetime(prof["date"].values)

    no3_model, no3_norm, _ = load_artifact(MODEL_DIR / "combined_NO3.pt", map_location=device)
    no3_model.to(device)
    chla_model, chla_norm, cmeta = load_artifact(MODEL_DIR / "combined_Chla_allfeat.pt",
                                                 map_location=device); chla_model.to(device)
    cutoff = float(cmeta["extra"]["cutoff_depth"])

    fig, axes = plt.subplots(len(SITES), 2, figsize=(9, 4.0 * len(SITES)))
    for row, (label, sid) in enumerate(SITES):
        k = sidx[sid]
        obs = samp[samp.station_id == sid].sort_values("depth")
        Tcol, Scol = Tp[k], Sp[k]
        dmax = float(obs.depth.max())
        fin = np.isfinite(Tcol) & np.isfinite(Scol)
        seafloor = float(fd[fin].max()) if fin.any() else dmax
        ymax = seafloor if args.full_depth else dmax + 40
        sel = fin if args.full_depth else (fin & (fd <= dmax + 40))
        dz, Tz, Sz = fd[sel], Tcol[sel], Scol[sel]

        # model NO3 profile (combined, low-sal)
        dfn = pd.DataFrame({"latitude": la[k], "longitude": lo[k], "depth": dz,
                            "temperature": Tz, "salinity": Sz})
        no3m = predict_field(no3_model, no3_norm, build_features(dfn).to_numpy(),
                             device, clip=(0.0, 60.0))
        no3m, _ = blend_low_salinity(no3m, Sz, "NO3")

        # model Chl-a profile (allfeat)
        mld, z_pyc, strat_max, z_nutr, nutr_max = struct_for_column(
            fd, Tcol, Scol, la[k], lo[k], no3_model, no3_norm, device)
        dfc = pd.DataFrame({"latitude": la[k], "longitude": lo[k], "depth": dz,
                            "temperature": Tz, "salinity": Sz, "mld": mld, "NO3": no3m})
        X = build_features(dfc, include_mld=True, include_no3=True).to_numpy()
        X = np.column_stack([X, np.full(len(dz), np.log(z_nutr + 1.0)), np.full(len(dz), nutr_max),
                             np.full(len(dz), np.log(z_pyc + 1.0)), np.full(len(dz), strat_max)])
        chlam = predict_field(chla_model, chla_norm, X, device, clip=TRACER_META["Chla"]["clip"])
        chlam = np.where(dz > cutoff, 0.0, chlam)

        # NO3 panel
        ax = axes[row, 0]
        ax.plot(obs.NO3, obs.depth, "o", color="tab:blue", ms=5, label="GLODAP obs")
        ax.plot(no3m, dz, "-", color="tab:red", lw=1.8, label="model (combined, low-sal)")
        ax.set_ylim(ymax, 0); ax.set_xlabel("NO3 (umol/kg)"); ax.set_ylabel("depth (m)")
        ax.axhline(seafloor, color="0.4", lw=0.8, ls=":")
        ax.set_title(f"{label}  {sid}\n{la[k]:.1f}N {lo[k]:.1f}E  {dates[k]:%Y-%m-%d}"
                     + (f"  (seafloor {seafloor:.0f} m)" if args.full_depth else ""), fontsize=9)
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower right")
        # Chl-a panel
        ax = axes[row, 1]
        ax.plot(obs.Chla, obs.depth, "o", color="tab:green", ms=5, label="GLODAP obs")
        ax.plot(chlam, dz, "-", color="tab:red", lw=1.8, label="model (allfeat)")
        ax.set_ylim(ymax, 0); ax.set_xlabel("Chl-a (mg/m3)")
        ax.axhline(seafloor, color="0.4", lw=0.8, ls=":")
        ax.set_title(f"{label}  Chl-a", fontsize=9); ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Observed (GLODAP) vs model vertical profiles on FORA-JPN60 T/S "
                 "— NO3 & Chl-a" + ("  [to seafloor]" if args.full_depth else ""), fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = OUT_DIR / ("fora_glodap_site_profiles_fulldepth.png" if args.full_depth
                     else "fora_glodap_site_profiles.png")
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
