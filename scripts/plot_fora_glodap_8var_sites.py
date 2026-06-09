"""GLODAP observation sites inside the FORA-JPN60 domain, and the subset where ALL
eight variables (DIC, TA, O2, NO3, PO4, SiO4, DOC, Chl-a) are co-located in one
sample. Recreates the all-sites map and overlays the 8-variable full-set sites on
the FORA SST.

Usage:
    uv run python scripts/plot_fora_glodap_8var_sites.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from predict_fora_chla import fora_url   # noqa: E402

CSV = ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
OUT_DIR = ROOT / "figures" / "fora"
BOX = dict(latmin=19.96, latmax=52.02, lonmin=116.94, lonmax=160.03)
PERIOD = (pd.Timestamp("1982-01-01"), pd.Timestamp("2020-12-31"))
VALS = {"DIC": "G2tco2", "TA": "G2talk", "O2": "G2oxygen", "NO3": "G2nitrate",
        "PO4": "G2phosphate", "SiO4": "G2silicate", "DOC": "G2doc", "Chla": "G2chla"}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = (["G2cruise", "G2station", "G2latitude", "G2longitude", "G2year",
             "G2month", "G2day"] + list(VALS.values()))
    df = pd.read_csv(CSV, usecols=cols).replace(-9999, np.nan)
    df = df.dropna(subset=["G2latitude", "G2longitude"])
    df = df[(df.G2latitude >= BOX["latmin"]) & (df.G2latitude <= BOX["latmax"])
            & (df.G2longitude >= BOX["lonmin"]) & (df.G2longitude <= BOX["lonmax"])]
    date = pd.to_datetime(dict(year=df.G2year, month=df.G2month, day=df.G2day), errors="coerce")
    df = df.assign(_date=date)

    def stats(d, label):
        cs = d.drop_duplicates(["G2cruise", "G2station"]).shape[0]
        pos = d.drop_duplicates(["G2latitude", "G2longitude"]).shape[0]
        print(f"{label:42s} samples={len(d):7d}  cruise×station={cs:6d}  uniq pos={pos:6d}  "
              f"cruises={d.G2cruise.nunique()}")
        return d

    print("=== FORA domain (lat 19.96-52.02N, lon 116.94-160.03E) ===")
    stats(df, "all GLODAP (any variable)")
    full = df[df[list(VALS.values())].notna().all(axis=1)]
    stats(full, "8-var full-set (DIC..DOC..Chla present)")
    in_t = (df._date >= PERIOD[0]) & (df._date <= PERIOD[1])
    full_t = full[(full._date >= PERIOD[0]) & (full._date <= PERIOD[1])]
    stats(full_t, "8-var full-set, FORA period 1982-2020")
    # clean DOC: drop cruise 4057 (ugC/L unit error) and DOC outside [30,150]
    full_c = full_t[(full_t.G2cruise != 4057) & (full_t.G2doc >= 30) & (full_t.G2doc <= 150)]
    stats(full_c, "  ... + clean DOC (excl 4057, DOC[30,150])")

    fs = full.drop_duplicates(["G2latitude", "G2longitude"])
    all_pos = df.drop_duplicates(["G2latitude", "G2longitude"])

    # FORA SST background
    print("\nloading FORA SST 2020-06-01 ...", flush=True)
    ds = xr.open_dataset(fora_url("t", "2020-06-01"))
    T = np.asarray(ds["thetao"].isel(time=0, depth=0, lat=slice(None, None, 2),
                                     lon=slice(None, None, 2)).load().values)
    lon = ds["lon"].values[::2]; lat = ds["lat"].values[::2]; ds.close()

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    fig = plt.figure(figsize=(11, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    vmin, vmax = np.nanpercentile(T, [1, 99])
    ax.pcolormesh(lon, lat, T, cmap="turbo", vmin=vmin, vmax=vmax, shading="auto",
                  transform=ccrs.PlateCarree(), alpha=0.85)
    ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=2); ax.coastlines(lw=0.4, zorder=3)
    ax.scatter(all_pos.G2longitude, all_pos.G2latitude, s=3, c="0.25", marker=".",
               linewidths=0, transform=ccrs.PlateCarree(), zorder=4,
               label=f"all GLODAP sites (uniq pos n={len(all_pos):,})")
    ax.scatter(fs.G2longitude, fs.G2latitude, s=40, c="red", marker="o",
               edgecolors="k", linewidths=0.5, transform=ccrs.PlateCarree(), zorder=5,
               label=f"8-var full-set sites (n={len(fs)})")
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, lw=0.2, color="gray", alpha=0.4)
    gl.top_labels = gl.right_labels = False
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_title("GLODAP sites in the FORA-JPN60 domain (on FORA SST 2020-06-01)\n"
                 f"all sites (grey) vs 8-variable full-set (red, DIC/TA/O2/NO3/PO4/SiO4/DOC/Chl-a), "
                 f"n={len(fs)} positions", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "fora_glodap_8var_sites.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
