"""Map the GLODAP sites that reported high DOC (> threshold), reusing the map
helpers from plot_glodap_coverage.py. Highlights cruise 4057 (the ~12x ugC/L unit
error) vs the other high-DOC cruises (sporadic / coastal), over a light grey
background of all DOC sites.

Usage:
    uv run python scripts/plot_doc_outliers_map.py            # threshold 150
    uv run python scripts/plot_doc_outliers_map.py --thr 150
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_glodap_coverage import _make_ax, in_extent, GLOBAL_CENTRAL_LON, JAPAN_EXTENT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
OUT_DIR = ROOT / "figures" / "glodap_coverage"
UNIT_ERROR_CRUISE = 4057


def _scatter_groups(ax, kw, all_lon, all_lat, hi, thr):
    """Background = all DOC sites; overlay 4057 (red) and other high cruises (orange)."""
    bg_lon = ((np.asarray(all_lon) + 180) % 360) - 180
    ax.scatter(bg_lon, all_lat, s=2, alpha=0.12, color="0.6", edgecolor="none",
               zorder=1, label=f"all DOC sites (n={len(all_lon):,})", **kw)
    for cr_mask, color, lbl in [
        (hi.G2cruise == UNIT_ERROR_CRUISE, "red",
         f"cruise {UNIT_ERROR_CRUISE} = unit error ugC/L (n={(hi.G2cruise == UNIT_ERROR_CRUISE).sum()})"),
        (hi.G2cruise != UNIT_ERROR_CRUISE, "darkorange",
         f"other high cruises (n={(hi.G2cruise != UNIT_ERROR_CRUISE).sum()})"),
    ]:
        g = hi[cr_mask]
        lon = ((g.G2longitude.to_numpy() + 180) % 360) - 180
        ax.scatter(lon, g.G2latitude.to_numpy(), s=22, alpha=0.85, color=color,
                   edgecolor="k", linewidths=0.3, zorder=4, label=lbl, **kw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--thr", type=float, default=150.0, help="DOC threshold (umol/kg)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV, usecols=["G2cruise", "G2latitude", "G2longitude", "G2doc"]).replace(-9999, np.nan)
    dd = df.dropna(subset=["G2doc", "G2latitude", "G2longitude"])
    hi = dd[dd.G2doc > args.thr]
    print(f"DOC sites: {len(dd):,}   DOC>{args.thr:.0f}: {len(hi)}  "
          f"(cruise {UNIT_ERROR_CRUISE}: {(hi.G2cruise == UNIT_ERROR_CRUISE).sum()}, "
          f"other: {(hi.G2cruise != UNIT_ERROR_CRUISE).sum()})")

    # global
    fig = plt.figure(figsize=(13, 6.5))
    ax, kw = _make_ax(fig, extent=None, central_lon=GLOBAL_CENTRAL_LON)
    _scatter_groups(ax, kw, dd.G2longitude, dd.G2latitude, hi, args.thr)
    ax.set_title(f"GLODAP sites reporting DOC > {args.thr:.0f} umol/kg (global)", fontsize=12)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9, markerscale=1.2)
    out = OUT_DIR / f"doc_outliers_global_thr{int(args.thr)}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")

    # Japan focus
    m = in_extent(dd.G2longitude, dd.G2latitude, JAPAN_EXTENT)
    mh = in_extent(hi.G2longitude, hi.G2latitude, JAPAN_EXTENT)
    if mh.any():
        fig = plt.figure(figsize=(8, 7))
        ax, kw = _make_ax(fig, extent=JAPAN_EXTENT)
        _scatter_groups(ax, kw, dd.G2longitude[m], dd.G2latitude[m], hi[mh], args.thr)
        ax.set_title(f"DOC > {args.thr:.0f} umol/kg near Japan", fontsize=12)
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
        out = OUT_DIR / f"doc_outliers_japan_thr{int(args.thr)}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"saved {out}")
    else:
        print("no DOC outliers in the Japan box")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
