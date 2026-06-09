"""Labelled map of the 21 GLODAP 8-variable full-set sites that sample to >=1000 m
(the sites whose profiles are in figures/fora/sites21/). Each site is numbered;
coincident sites (e.g. the 137E line) are fanned out with leader lines, and a
table maps number -> cruise_station / lat / lon / date so the map cross-references
the per-site profile figures (site_<cruise>_<station>_*.png).

Usage:
    uv run python scripts/plot_fora_glodap_21site_map.py
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
from predict_fora_chla import fora_url                              # noqa: E402
from plot_fora_glodap_21site_profiles import select_sites, CSV, meta_sub_from_keys  # noqa: E402

OUT = ROOT / "figures" / "fora" / "fora_glodap_21site_map.png"
BAND_COLOR = {"low": "tab:blue", "mid": "tab:orange", "high": "tab:red",
              "ECS": "magenta", "SoJ": "deepskyblue"}
# 6 extra hand-picked 7-var (no DOC) sites: East China Sea shelf + Sea of Japan deep
ECS_SOJ = [(217, 29.0), (217, 20.0), (217, 18.0),
           (2068, 5498.0), (2068, 5490.0), (2068, 5487.0)]
ECS_SOJ_BAND = ["ECS", "ECS", "ECS", "SoJ", "SoJ", "SoJ"]


def band(la):
    return "low" if la < 28 else ("mid" if la < 40 else "high")


def main() -> int:
    meta, _ = select_sites()
    meta = meta.sort_values("lat").reset_index(drop=True)
    meta["idx"] = np.arange(1, len(meta) + 1)
    meta["band"] = meta.lat.apply(band)
    print(f"{len(meta)} sites (low={sum(meta.band=='low')}, mid={sum(meta.band=='mid')}, "
          f"high={sum(meta.band=='high')})")
    # 6 extra ECS / Sea-of-Japan sites (continue the index numbering)
    m6, _ = meta_sub_from_keys(ECS_SOJ)
    m6 = m6.set_index("sid").loc[[f"{c}_{s}" for c, s in ECS_SOJ]].reset_index()
    m6["idx"] = np.arange(len(meta) + 1, len(meta) + 1 + len(m6))
    m6["band"] = ECS_SOJ_BAND
    print(f"+ {len(m6)} ECS/SoJ sites (idx {m6.idx.min()}-{m6.idx.max()})")

    # background: all GLODAP positions in the FORA box
    allp = pd.read_csv(CSV, usecols=["G2latitude", "G2longitude"]).replace(-9999, np.nan).dropna()
    allp = allp[(allp.G2latitude >= 19.96) & (allp.G2latitude <= 52.02)
                & (allp.G2longitude >= 116.94) & (allp.G2longitude <= 160.03)]
    allp = allp.drop_duplicates()

    ds = xr.open_dataset(fora_url("t", "2020-06-01"))
    T = np.asarray(ds["thetao"].isel(time=0, depth=0, lat=slice(None, None, 2),
                                     lon=slice(None, None, 2)).load().values)
    lon = ds["lon"].values[::2]; lat = ds["lat"].values[::2]; ds.close()

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.4], hspace=0.08)
    ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
    vmin, vmax = np.nanpercentile(T, [1, 99])
    ax.pcolormesh(lon, lat, T, cmap="turbo", vmin=vmin, vmax=vmax, shading="auto",
                  transform=ccrs.PlateCarree(), alpha=0.8)
    ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=2); ax.coastlines(lw=0.4, zorder=3)
    ax.scatter(allp.G2longitude, allp.G2latitude, s=2, c="0.3", marker=".", linewidths=0,
               transform=ccrs.PlateCarree(), zorder=3, alpha=0.5)

    # fan out coincident sites within ~1deg clusters; label with the index number
    meta["clat"] = meta.lat.round(0); meta["clon"] = meta.lon.round(0)
    for (cla, clo), grp in meta.groupby(["clat", "clon"]):
        n = len(grp)
        for j, (_, r) in enumerate(grp.iterrows()):
            if n == 1:
                dx = dy = 0.0
            else:
                ang = 2 * np.pi * j / n
                dx, dy = 1.05 * np.cos(ang), 1.05 * np.sin(ang)
            px, py = r.lon + dx, r.lat + dy
            if n > 1:
                ax.plot([r.lon, px], [r.lat, py], "-", color="k", lw=0.4, alpha=0.6,
                        transform=ccrs.PlateCarree(), zorder=4)
            ax.scatter(px, py, s=130, c=BAND_COLOR[r.band], marker="o", edgecolors="k",
                       linewidths=0.6, transform=ccrs.PlateCarree(), zorder=5)
            ax.text(px, py, str(r.idx), ha="center", va="center", fontsize=7,
                    color="white", fontweight="bold", transform=ccrs.PlateCarree(), zorder=6)
    # 6 extra ECS / Sea-of-Japan sites: star markers, same fan-out
    m6["clat"] = m6.lat.round(0); m6["clon"] = m6.lon.round(0)
    for (cla, clo), grp in m6.groupby(["clat", "clon"]):
        n = len(grp)
        for j, (_, r) in enumerate(grp.iterrows()):
            dx, dy = (0.0, 0.0) if n == 1 else (1.05 * np.cos(2 * np.pi * j / n),
                                                1.05 * np.sin(2 * np.pi * j / n))
            px, py = r.lon + dx, r.lat + dy
            if n > 1:
                ax.plot([r.lon, px], [r.lat, py], "-", color="k", lw=0.4, alpha=0.6,
                        transform=ccrs.PlateCarree(), zorder=4)
            ax.scatter(px, py, s=240, c=BAND_COLOR[r.band], marker="*", edgecolors="k",
                       linewidths=0.6, transform=ccrs.PlateCarree(), zorder=5)
            ax.text(px, py, str(r.idx), ha="center", va="center", fontsize=6.5,
                    color="black", fontweight="bold", transform=ccrs.PlateCarree(), zorder=6)
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, lw=0.2, color="gray", alpha=0.4)
    gl.top_labels = gl.right_labels = False
    ax.set_title("GLODAP sites on FORA SST 2020-06-01: 21 deep 8-var (circles, by lat band) "
                 "+ 6 ECS/SoJ 7-var (stars)\n"
                 "numbers = site index (see table); circles blue<28N / orange 28-40N / red>=40N; "
                 "stars magenta=ECS / cyan=Sea of Japan", fontsize=10)

    # table of index -> cruise_station / lat / lon / date / band (3 columns: 21 + 6)
    axt = fig.add_subplot(gs[1]); axt.axis("off")
    allm = pd.concat([meta, m6], ignore_index=True)
    def col_text(rows):
        s = f"{'#':>2}  {'cruise_station':14} {'lat':>6} {'lon':>7}  {'date':10} {'band':4}\n"
        for _, r in rows.iterrows():
            s += (f"{r.idx:>2}  {r.cruise}_{r.station:<10} {r.lat:6.2f} {r.lon:7.2f}  "
                  f"{r.date:%Y-%m-%d} {r.band:4}\n")
        return s
    axt.text(0.01, 0.98, col_text(allm.iloc[:11]), family="monospace", fontsize=8,
             va="top", ha="left", transform=axt.transAxes)
    axt.text(0.36, 0.98, col_text(allm.iloc[11:21]), family="monospace", fontsize=8,
             va="top", ha="left", transform=axt.transAxes)
    axt.text(0.71, 0.98, col_text(allm.iloc[21:]), family="monospace", fontsize=8,
             va="top", ha="left", transform=axt.transAxes)
    axt.text(0.01, 0.0, "profile figs: figures/fora/{sites21,sites_ecs_soj}/"
             "site_<cruise>_<station>_{fulldepth,300m}.png", fontsize=8, style="italic",
             transform=axt.transAxes)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
