"""Map the spatial coverage of GLODAP observations per parameter.

For each target it scatters the locations of good (flag==2) measurements on a
world map with coastlines (30 W origin, Pacific-centered) and on a Japan-coast
focus map (120-160 E, 20-50 N). The CSV is read once with only the needed
columns.

Usage:
    uv run python scripts/plot_glodap_coverage.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bio_params.loaders.glodap import MISSING_SENTINEL, TARGET_COLUMNS

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
except Exception:  # noqa: BLE001
    _HAS_CARTOPY = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV = PROJECT_ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
OUT_DIR = PROJECT_ROOT / "figures" / "glodap_coverage"

JAPAN_EXTENT = (120.0, 160.0, 20.0, 50.0)
GLOBAL_CENTRAL_LON = 150.0  # left edge at 30 W

# Per-target marker color (reuse a qualitative palette).
COLORS = {
    "DIC": "tab:blue", "TA": "tab:orange", "NO3": "tab:green",
    "PO4": "tab:red", "SiO4": "tab:purple", "O2": "tab:brown",
    "DOC": "tab:pink", "Chla": "tab:olive", "TDN": "tab:cyan",
    "TOC": "darkgreen", "DON": "magenta", "C13": "navy",
    "O18": "teal", "C14": "crimson", "H3": "darkorange",
}


def _make_ax(fig, extent=None, central_lon=0.0):
    if _HAS_CARTOPY:
        proj = ccrs.PlateCarree(central_longitude=central_lon if extent is None else 0.0)
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        try:
            ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=0)
            ax.coastlines(resolution="50m", linewidth=0.5, color="0.3", zorder=3)
        except Exception:  # noqa: BLE001
            pass
        gl = ax.gridlines(draw_labels=True, ls=":", alpha=0.4)
        gl.top_labels = gl.right_labels = False
        if extent is not None:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        else:
            ax.set_global()
        return ax, dict(transform=ccrs.PlateCarree())
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    else:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
    ax.set_aspect("equal")
    return ax, {}


def plot_map(lon, lat, *, color, title, out_path, extent=None, point_size=2,
             alpha=0.25) -> None:
    fig = plt.figure(figsize=(12, 6) if extent is None else (8, 7))
    ax, kw = _make_ax(fig, extent=extent, central_lon=GLOBAL_CENTRAL_LON)
    lon = ((np.asarray(lon) + 180) % 360) - 180
    ax.scatter(lon, lat, s=point_size, alpha=alpha, color=color,
               edgecolor="none", zorder=2, **kw)
    ax.set_title(title, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def in_extent(lon, lat, extent):
    lon = ((np.asarray(lon) + 180) % 360) - 180
    lo0, lo1, la0, la1 = extent
    return (lon >= lo0) & (lon <= lo1) & (lat >= la0) & (lat <= la1)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not _HAS_CARTOPY:
        print("WARNING: cartopy unavailable; maps will have no coastlines")

    # Read coordinates + every target value/flag column once.
    usecols = ["G2latitude", "G2longitude"]
    for val, flag in TARGET_COLUMNS.values():
        usecols += [val, flag]
    usecols = [c for c in dict.fromkeys(usecols)]  # dedup, keep order
    print(f"Reading {CSV} ...")
    df = pd.read_csv(CSV, usecols=usecols, low_memory=False).replace(MISSING_SENTINEL, np.nan)
    print(f"  rows: {len(df):,}")

    for tgt, (val, flag) in TARGET_COLUMNS.items():
        good = (df[flag] == 2) & df[val].notna() & df["G2latitude"].notna() & df["G2longitude"].notna()
        lon = df.loc[good, "G2longitude"].to_numpy()
        lat = df.loc[good, "G2latitude"].to_numpy()
        color = COLORS.get(tgt, "tab:blue")
        print(f"{tgt}: {len(lon):,} good points")

        plot_map(
            lon, lat, color=color,
            title=f"GLODAP {tgt} observations (global)  n={len(lon):,}  (flag==2)",
            out_path=OUT_DIR / f"coverage_{tgt}.png",
        )
        m = in_extent(lon, lat, JAPAN_EXTENT)
        plot_map(
            lon[m], lat[m], color=color,
            title=f"GLODAP {tgt} near Japan  n={int(m.sum()):,}  (flag==2)",
            out_path=OUT_DIR / f"coverage_{tgt}_japan.png",
            extent=JAPAN_EXTENT, point_size=6, alpha=0.5,
        )

    print(f"\nSaved maps -> {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
