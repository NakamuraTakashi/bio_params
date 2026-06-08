"""Vertical cross-section of FORA-JPN60 Chl-a (plain allfeat) along a transect,
RAW vs surface-anchored to the daily satellite, to verify the anchor pulls the
upper ~100 m toward the satellite (and stays plausible in low-salinity /
extrapolation regions) while leaving the deeper structure unchanged.

Default transect: 30 N, 122->145 E (East China Sea shelf -> Kuroshio -> open
Pacific), which crosses the low-salinity Changjiang-influenced shelf.

Usage:
    uv run python scripts/plot_fora_chla_section.py --date 2020-06-01
    uv run python scripts/plot_fora_chla_section.py --lat0 30 --lat1 30 --lon0 122 --lon1 145
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
    TRACER_META, blend_low_salinity, predict_field, _grad_peak_field, _nearest_index)
from predict_fora_chla import fora_url, fora_mld, predict_no3_field   # noqa: E402

MODEL_DIR = ROOT / "models" / "pretrained"
OUT_DIR = ROOT / "figures" / "fora"


def extract_section(date, lat0, lat1, lon0, lon1, npts, zmax):
    """FORA T/S along a straight transect: returns d(levels), T/S (nlev,npts), lon/lat (npts).

    Loads the bounding box of the transect with ordinary (orthogonal) slicing
    (robust over OPeNDAP), then samples the transect points locally.
    """
    dt = xr.open_dataset(fora_url("t", date)); ds = xr.open_dataset(fora_url("s", date))
    flat = dt["lat"].values; flon = dt["lon"].values; zc = dt["depth"].values
    sel = np.where(zc <= zmax)[0]
    lon_pts = np.linspace(lon0, lon1, npts); lat_pts = np.linspace(lat0, lat1, npts)
    js = np.where((flat >= lat_pts.min() - 0.1) & (flat <= lat_pts.max() + 0.1))[0]
    is_ = np.where((flon >= lon_pts.min() - 0.1) & (flon <= lon_pts.max() + 0.1))[0]
    j0, j1 = int(js[0]), int(js[-1]) + 1; i0, i1 = int(is_[0]), int(is_[-1]) + 1
    Tb = np.asarray(dt["thetao"].isel(time=0, lat=slice(j0, j1), lon=slice(i0, i1)).load().values)
    Sb = np.asarray(ds["so"].isel(time=0, lat=slice(j0, j1), lon=slice(i0, i1)).load().values)
    dt.close(); ds.close()
    latb = flat[j0:j1]; lonb = flon[i0:i1]
    jj = np.array([int(np.argmin(np.abs(latb - a))) for a in lat_pts])
    ii = np.array([int(np.argmin(np.abs(lonb - a))) for a in lon_pts])
    T = Tb[sel][:, jj, ii]      # (nlev_sel, npts)
    S = Sb[sel][:, jj, ii]
    return zc[sel], T, S, lon_pts, lat_pts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--date", default="2020-06-01")
    p.add_argument("--lat0", type=float, default=30.0); p.add_argument("--lat1", type=float, default=30.0)
    p.add_argument("--lon0", type=float, default=122.0); p.add_argument("--lon1", type=float, default=145.0)
    p.add_argument("--npts", type=int, default=80)
    p.add_argument("--zmax", type=float, default=350.0, help="section depth shown (m)")
    p.add_argument("--anchor-taper", type=float, default=100.0)
    p.add_argument("--anchor-clip", type=float, nargs=2, default=[0.2, 5.0])
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"FORA section {args.date}: ({args.lat0},{args.lon0})->({args.lat1},{args.lon1}) "
          f"{args.npts} pts ...", flush=True)
    d, T, S, lon_pts, lat_pts = extract_section(args.date, args.lat0, args.lat1,
                                                args.lon0, args.lon1, args.npts, args.zmax)
    nlev, npt = T.shape
    no3_model, no3_norm, _ = load_artifact(MODEL_DIR / "combined_NO3.pt", map_location=device)
    no3_model.to(device)
    chla_model, chla_norm, cmeta = load_artifact(MODEL_DIR / "combined_Chla_allfeat.pt",
                                                 map_location=device); chla_model.to(device)
    cutoff = float(cmeta["extra"]["cutoff_depth"])

    # structure descriptors per column
    d3 = np.broadcast_to(d[:, None, None], (nlev, npt, 1))
    lat3 = np.broadcast_to(lat_pts[None, :, None], (nlev, npt, 1))
    sig = sigma0(S.T.ravel(), T.T.ravel(), d3.ravel(), lat3.ravel()).reshape(nlev, npt, 1)
    mld = fora_mld(d, sig).ravel()
    z_pyc, strat_max = (a.ravel() for a in _grad_peak_field(d3, sig))
    no3_prof = np.stack([predict_no3_field(T[lv], S[lv], lat_pts, lon_pts, d[lv],
                                           no3_model, no3_norm, device, True) for lv in range(nlev)])
    z_nutr, nutr_max = (a.ravel() for a in _grad_peak_field(d3, no3_prof[:, :, None]))

    # Chl-a profile at every level
    chla = np.full((nlev, npt), np.nan)
    for lv in range(nlev):
        Tk, Sk = T[lv], S[lv]
        ok = np.isfinite(Tk) & np.isfinite(Sk) & np.isfinite(mld)
        if not ok.any():
            continue
        no3_k = predict_no3_field(Tk, Sk, lat_pts, lon_pts, d[lv], no3_model, no3_norm, device, True)
        df = pd.DataFrame({"latitude": lat_pts[ok], "longitude": lon_pts[ok],
                           "depth": np.full(int(ok.sum()), d[lv]), "temperature": Tk[ok],
                           "salinity": Sk[ok], "mld": mld[ok], "NO3": no3_k[ok]})
        X = build_features(df, include_mld=True, include_no3=True).to_numpy()
        X = np.column_stack([X, np.log(z_nutr[ok] + 1.0), nutr_max[ok],
                             np.log(z_pyc[ok] + 1.0), strat_max[ok]])
        pv = predict_field(chla_model, chla_norm, X, device, clip=TRACER_META["Chla"]["clip"])
        pv = np.where(d[lv] > cutoff, 0.0, pv)
        chla[lv, ok] = pv

    # daily satellite surface along the line + anchor
    from bio_params.satellite import chla_day_field
    la_ax, lo_ax, arr, src = chla_day_field(args.date)
    il = _nearest_index(lat_pts, la_ax); io = _nearest_index(((lon_pts + 180) % 360) - 180, lo_ax)
    sat_surf = arr[il, io]
    lo_c, hi_c = args.anchor_clip
    msurf = chla[0]                                   # model surface (level ~1 m)
    rhat = np.clip(sat_surf / np.maximum(msurf, 1e-3), lo_c, hi_c)
    rhat = np.where(np.isfinite(rhat), rhat, 1.0)
    reff = 1.0 + (rhat[None, :] - 1.0) * np.clip((args.anchor_taper - d[:, None]) / args.anchor_taper, 0.0, 1.0)
    chla_anc = chla * reff
    print(f"  satellite {src}; Rhat median={np.nanmedian(rhat):.2f} "
          f"p10={np.nanpercentile(rhat,10):.2f} p90={np.nanpercentile(rhat,90):.2f}", flush=True)

    # --- plot: surface line + RAW section + anchored section ---
    x = lon_pts
    vmax = float(np.nanpercentile(chla_anc[d <= 200], 98))
    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 2, 2], hspace=0.28)
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(x, sat_surf, "-o", ms=3, color="tab:green", label="daily satellite surface")
    ax0.plot(x, msurf, "-", color="tab:red", label="model surface (RAW)")
    ax0.plot(x, chla_anc[0], "--", color="tab:blue", label="model surface (anchored)")
    ax0.set_ylabel("Chl-a (mg/m3)"); ax0.set_xlim(x.min(), x.max()); ax0.grid(alpha=0.3)
    ax0.legend(fontsize=8); ax0.set_title(f"FORA-JPN60 {args.date}  section "
               f"{args.lat0:.0f}N {args.lon0:.0f}-{args.lon1:.0f}E  (surface Chl-a)", fontsize=10)
    for ax, fld, ttl in [(fig.add_subplot(gs[1]), chla, "RAW allfeat"),
                         (fig.add_subplot(gs[2]), chla_anc, f"surface-anchored (daily sat, taper->{args.anchor_taper:.0f}m)")]:
        pcm = ax.pcolormesh(x, d, np.ma.masked_invalid(fld), cmap="viridis",
                            vmin=0, vmax=vmax, shading="auto")
        ax.plot(x, mld, "w-", lw=1.0, alpha=0.7, label="MLD")
        ax.set_ylim(args.zmax, 0); ax.set_ylabel("depth (m)"); ax.set_title(f"Chl-a section — {ttl}", fontsize=10)
        ax.set_xlabel("Longitude (E)"); ax.legend(fontsize=7, loc="lower right")
        plt.colorbar(pcm, ax=ax, pad=0.02).set_label("Chl-a (mg/m3)", fontsize=8)
    out = OUT_DIR / f"fora_chla_section_{args.date.replace('-', '')}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
