"""Compare the satellite surface Chl-a (the SOCA anchor) with the model's
surface Chl-a prediction on the ROMS grid, for the ini-file month.

Left  = GlobColour surface Chl-a (same month as the ROMS ini), sampled to the
        ROMS grid (the "truth" fed into the satellite-anchored model).
Right = surface (depth=0) prediction of the satellite-anchored model.
Both panels share the SAME colormap and color range so they are directly
comparable; a third panel shows their difference.

Reuses the grid / S-coordinate / satellite-loading machinery from
predict_roms_ini_depths.py.

Usage:
    uv run python scripts/plot_roms_surface_vs_satellite.py --tag satchl
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from bio_params.features import build_features
from bio_params.persist import load_artifact

# Reuse constants/helpers from the main inference script.
_spec = importlib.util.spec_from_file_location(
    "_pri", Path(__file__).resolve().parent / "predict_roms_ini_depths.py")
_pri = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pri)

INI, GRID, MODEL_DIR = _pri.INI, _pri.GRID, _pri.MODEL_DIR
TRACER_META, SAT_DATASET = _pri.TRACER_META, _pri.SAT_DATASET
OUT_DIR = Path(__file__).resolve().parent.parent / "figures" / "roms_surface_vs_satellite"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default="bgc_argo")
    p.add_argument("--target", default="Chla")
    p.add_argument("--tag", default="satchl")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tgt = args.target
    m = TRACER_META[tgt]

    art = MODEL_DIR / f"{args.source}_{tgt}_{args.tag}.pt"
    model, normalizer, meta = load_artifact(art, map_location=device)
    model.to(device)
    extra = meta["extra"]
    if not extra.get("surface_chla", False):
        print(f"ERROR: {art.name} has no surface_chla feature")
        return 1
    log_target = bool(extra.get("log_target", False))
    surface_chla_log = bool(extra.get("surface_chla_log", True))

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
    z_rho = _pri.compute_z_rho(h, zeta, s_rho, Cs_r, hc)
    land = mask < 0.5
    J, I = h.shape

    month = _pri._ini_month(INI)
    print(f"loading satellite surface Chl-a ({month}) ...")
    sat = _pri.load_roms_surface_chla(lat, lon, month)   # filled, ocean+land
    sat_masked = np.where(land, np.nan, sat)

    # Surface (depth=0) model prediction, anchored by the satellite field.
    T2, S2 = temp[-1], salt[-1]
    d2 = np.abs(z_rho[-1])
    valid = np.isfinite(T2) & np.isfinite(S2) & (~land)
    jj, ii = np.where(valid)
    df2 = pd.DataFrame({
        "latitude": lat[jj, ii], "longitude": lon[jj, ii], "depth": d2[jj, ii],
        "temperature": T2[jj, ii], "salinity": S2[jj, ii],
        "surface_chla": sat[jj, ii],
    })
    X = build_features(df2, include_surface_chla=True,
                       surface_chla_log=surface_chla_log).to_numpy()
    pred = _pri.predict_field(model, normalizer, X, device,
                              log_target=log_target, clip=m.get("clip"))
    pred_field = np.full((J, I), np.nan)
    pred_field[jj, ii] = pred

    # Shared color range (2-98 pct of the two fields combined).
    both = np.concatenate([sat_masked[np.isfinite(sat_masked)],
                           pred_field[np.isfinite(pred_field)]])
    vmin, vmax = np.percentile(both, [2, 98])

    # RMSE between satellite and model surface, over ocean.
    ok = np.isfinite(sat_masked) & np.isfinite(pred_field)
    rmse = float(np.sqrt(np.mean((pred_field[ok] - sat_masked[ok]) ** 2)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), constrained_layout=True)
    panels = [
        ("Satellite surface Chl-a\nGlobColour " + month, sat_masked, m["cmap"], vmin, vmax),
        ("Model surface prediction\n(satellite-anchored)", pred_field, m["cmap"], vmin, vmax),
    ]
    for ax, (title, fld, cmap, lo, hi) in zip(axes, panels):
        pcm = ax.pcolormesh(lon, lat, np.ma.masked_invalid(fld), cmap=cmap,
                            vmin=lo, vmax=hi, shading="auto")
        ax.set_facecolor("0.8"); ax.set_aspect("equal")
        ax.set_title(title, fontsize=11); ax.set_xlabel("Longitude")
        fig.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.08,
                     fraction=0.05).set_label(f"{tgt} ({m['unit']})", fontsize=9)
    axes[0].set_ylabel("Latitude")

    diff = pred_field - sat_masked
    dmax = np.percentile(np.abs(diff[np.isfinite(diff)]), 98)
    pcm = axes[2].pcolormesh(lon, lat, np.ma.masked_invalid(diff), cmap="RdBu_r",
                             vmin=-dmax, vmax=dmax, shading="auto")
    axes[2].set_facecolor("0.8"); axes[2].set_aspect("equal")
    axes[2].set_title(f"Model - Satellite\nRMSE={rmse:.3g} {m['unit']}", fontsize=11)
    axes[2].set_xlabel("Longitude")
    fig.colorbar(pcm, ax=axes[2], orientation="horizontal", pad=0.08,
                 fraction=0.05).set_label(f"diff ({m['unit']})", fontsize=9)

    fig.suptitle(
        f"{m['long']} ({tgt}) surface: satellite vs {args.source}_{args.tag} model "
        f"on ROMS grid — {INI.name}", fontsize=12)
    out = OUT_DIR / f"surface_vs_satellite_{tgt}_{args.source}_{args.tag}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}  (shared range {vmin:.3g}-{vmax:.3g} {m['unit']}, "
          f"surface RMSE={rmse:.3g})")
    ds.close(); g.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
