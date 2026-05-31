"""Compare combined vs BGC-Argo-only predictions on the ROMS grid by depth.

For each target and depth, predicts with both the combined_<t>.pt and the
bgc_argo_<t>.pt models on the ROMS ini T/S, and maps their difference
(combined - bgc_argo). The point: GLODAP adds deep (>2000 m) coverage that
BGC-Argo lacks, so the two models should diverge most at depth, where the
combined model is better constrained.

Reuses the depth-interpolation / feature machinery from
predict_roms_ini_depths.py.

Usage:
    uv run python scripts/compare_roms_combined_vs_bgc.py --targets NO3 O2
"""
from __future__ import annotations

import argparse
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

# Reuse constants/helpers from the main inference script.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_pri", Path(__file__).resolve().parent / "predict_roms_ini_depths.py")
_pri = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pri)

INI, GRID, MODEL_DIR = _pri.INI, _pri.GRID, _pri.MODEL_DIR
TARGET_DEPTHS = _pri.TARGET_DEPTHS
compute_z_rho, interp_to_depth, predict_field = (
    _pri.compute_z_rho, _pri.interp_to_depth, _pri.predict_field)
TRACER_META = _pri.TRACER_META
OUT_DIR = Path(__file__).resolve().parent.parent / "figures" / "roms_combined_vs_bgc"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--targets", nargs="+", default=["NO3", "O2", "Chla"],
                   choices=["NO3", "O2", "Chla"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = xr.open_dataset(INI, decode_times=False)
    g = xr.open_dataset(GRID, decode_times=False)
    h = np.where(g["h"].values <= 0, np.nan, g["h"].values).astype(np.float64)
    mask = g["mask_rho"].values
    lon, lat = g["lon_rho"].values, g["lat_rho"].values
    zeta = ds["zeta"].isel(ocean_time=0).values.astype(np.float64)
    s_rho = ds["s_rho"].values.astype(np.float64)
    Cs_r = ds["Cs_r"].values.astype(np.float64)
    hc = float(ds["hc"].values)
    temp = ds["temp"].isel(ocean_time=0).values.astype(np.float64)
    salt = ds["salt"].isel(ocean_time=0).values.astype(np.float64)
    z_rho = compute_z_rho(h, zeta, s_rho, Cs_r, hc)
    land = mask < 0.5
    J, I = h.shape

    per_depth = []
    for depth in TARGET_DEPTHS:
        if depth == 0.0:
            T2, S2 = temp[-1].copy(), salt[-1].copy()
            d2 = np.abs(z_rho[-1]); label = "surface"
        else:
            T2 = interp_to_depth(temp, z_rho, -depth)
            S2 = interp_to_depth(salt, z_rho, -depth)
            d2 = np.full((J, I), depth); label = f"{depth:.0f} m"
        valid = np.isfinite(T2) & np.isfinite(S2) & (~land)
        jj, ii = np.where(valid)
        X = build_features(pd.DataFrame({
            "latitude": lat[jj, ii], "longitude": lon[jj, ii], "depth": d2[jj, ii],
            "temperature": T2[jj, ii], "salinity": S2[jj, ii]})).to_numpy()
        per_depth.append(dict(label=label, valid=valid, jj=jj, ii=ii, X=X))

    for tgt in args.targets:
        a_comb = MODEL_DIR / f"combined_{tgt}.pt"
        a_bgc = MODEL_DIR / f"bgc_argo_{tgt}.pt"
        if not (a_comb.exists() and a_bgc.exists()):
            print(f"skip {tgt}: missing artifact"); continue
        mc, nc, _ = load_artifact(a_comb, map_location=device); mc.to(device)
        mb, nb, metab = load_artifact(a_bgc, map_location=device); mb.to(device)
        logb = bool(metab["extra"].get("log_target", False))
        meta = TRACER_META[tgt]; unit = meta["unit"]

        fig, axes = plt.subplots(1, len(TARGET_DEPTHS),
                                 figsize=(4.6 * len(TARGET_DEPTHS), 5.4),
                                 constrained_layout=True)
        for ax, pd_ in zip(axes, per_depth):
            pc = predict_field(mc, nc, pd_["X"], device, clip=meta.get("clip"))
            pb = predict_field(mb, nb, pd_["X"], device, log_target=logb, clip=meta.get("clip"))
            diff = np.full((J, I), np.nan)
            diff[pd_["jj"], pd_["ii"]] = pc - pb
            fin = diff[np.isfinite(diff)]
            vmax = np.percentile(np.abs(fin), 98) if fin.size else 1.0
            pcm = ax.pcolormesh(lon, lat, np.ma.masked_invalid(diff),
                                cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
            ax.set_facecolor("0.8")
            ax.set_title(pd_["label"], fontsize=10)
            ax.set_xlabel("Longitude")
            if ax is axes[0]:
                ax.set_ylabel("Latitude")
            ax.set_aspect("equal")
            fig.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.08,
                         fraction=0.05).set_label(f"combined - bgc_argo ({unit})", fontsize=9)
        fig.suptitle(f"{meta['long']} ({tgt}): combined minus BGC-Argo-only on ROMS grid\n"
                     f"GLODAP adds deep (>2000m) coverage; expect largest differences at depth",
                     fontsize=12)
        out = OUT_DIR / f"diff_{tgt}_depths.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")

    ds.close(); g.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
