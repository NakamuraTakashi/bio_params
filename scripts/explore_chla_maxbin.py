"""Explore grouping vertical Chl-a patterns by the PROFILE MAXIMUM Chl-a
(instead of the surface value), to reconsider whether surface Chl is the right
grouping axis.

Each profile (unique lat/lon/time) is assigned a group by its maximum Chl-a:
  <=0.2, 0.2-0.5, 0.5-1.0, 1.0-2.0, 2.0-5.0, >=5.0 mg/m3.
Produces:
  (1) <prefix>_profiles.png : per-group absolute Chl(z) profiles with the observed
      spread (density + 16-84 percentile band + median).
  (2) <prefix>_map.png      : world map of profile locations colored by group.

Uses ALL Chl-a profiles (GLODAP + BGC-Argo), not the NO3 co-located subset, so the
profile maximum is the true column maximum.

Usage:
    uv run python scripts/explore_chla_maxbin.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bio_params.loaders.bgc_argo import load_bgc_argo
from bio_params.loaders.glodap import load_glodap

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
SPROF = ROOT / "data" / "bgc_argo" / "raw" / "floats"
OUT_DIR = ROOT / "figures" / "chla_maxbin"
EDGES = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 1e9]
LABELS = ["<=0.2", "0.2-0.5", "0.5-1.0", "1.0-2.0", "2.0-5.0", ">=5.0"]
DEPTH_GRID = np.array([0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110, 130,
                       150, 175, 200, 250, 300], dtype=float)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--max-depth", type=float, default=300.0)
    p.add_argument("--map-max-points", type=int, default=40000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefix", default=str(OUT_DIR / "chla_maxbin"))
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    g = load_glodap(CSV, "Chla", with_time=True); g["source"] = "glodap"
    a = load_bgc_argo(SPROF, "Chla"); a["source"] = "bgc_argo"
    cols = ["latitude", "longitude", "depth", "Chla", "time", "source"]
    df = pd.concat([g[cols], a[cols]], ignore_index=True)
    df = df[np.isfinite(df["Chla"]) & np.isfinite(df["depth"])].reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])
    keys = ["latitude", "longitude", "time"]

    pmax = df.groupby(keys, sort=False)["Chla"].max().rename("prof_max")
    df = df.merge(pmax, on=keys, how="left")
    df["grp"] = np.clip(np.digitize(df["prof_max"].to_numpy(), EDGES) - 1, 0, len(LABELS) - 1)
    prof = df.drop_duplicates(keys)[keys + ["prof_max", "grp", "source"]].reset_index(drop=True)
    print(f"profiles: {len(prof):,}  (glodap={int((prof.source=='glodap').sum()):,}, "
          f"bgc={int((prof.source=='bgc_argo').sum()):,})")
    for b, lab in enumerate(LABELS):
        print(f"  group {lab:>8}: {int((prof.grp==b).sum()):,} profiles")

    # ---- (1) per-group vertical profiles (absolute Chl) ----
    depth = df["depth"].to_numpy(); chl = df["Chla"].to_numpy(); grp = df["grp"].to_numpy()
    keep = depth <= args.max_depth
    mids = (DEPTH_GRID[:-1] + DEPTH_GRID[1:]) / 2.0
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharey=True)
    axes = axes.ravel()
    for b, lab in enumerate(LABELS):
        ax = axes[b]; m = keep & (grp == b)
        c, d = chl[m], depth[m]
        xhi = max(0.3, float(np.percentile(c, 99.0))) if c.size else 1.0
        if c.size > 500:
            ax.hexbin(c, d, gridsize=40, bins="log", cmap="Blues", mincnt=1,
                      extent=(0, xhi, 0, args.max_depth))
        elif c.size:
            ax.scatter(c, d, s=6, alpha=0.3, color="steelblue", edgecolor="none")
        di = np.digitize(d, mids)
        p50 = np.full(len(DEPTH_GRID), np.nan); p16 = p50.copy(); p84 = p50.copy()
        for k in range(len(DEPTH_GRID)):
            ck = c[di == k]
            if ck.size >= 20:
                p16[k], p50[k], p84[k] = np.percentile(ck, [16, 50, 84])
        ok = np.isfinite(p50)
        ax.fill_betweenx(DEPTH_GRID[ok], p16[ok], p84[ok], color="orange", alpha=0.3,
                         label="obs 16-84%")
        ax.plot(p50[ok], DEPTH_GRID[ok], color="red", lw=2.0, label="median")
        ax.set_xlim(0, xhi); ax.set_ylim(args.max_depth, 0)
        ax.set_title(f"profile max Chl {lab} mg/m3  (n={int((prof.grp==b).sum()):,} prof)",
                     fontsize=10)
        if b % 3 == 0:
            ax.set_ylabel("depth (m)")
        if b >= 3:
            ax.set_xlabel("Chl-a (mg/m3)")
        if b == 0:
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Vertical Chl-a patterns grouped by PROFILE MAX Chl-a", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    f1 = Path(f"{args.prefix}_profiles.png")
    fig.savefig(f1, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved {f1}")

    # ---- (2) map colored by group ----
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    pm = prof.copy()
    if len(pm) > args.map_max_points:
        pm = pm.sample(args.map_max_points, random_state=args.seed)
    lon = ((pm["longitude"].to_numpy() + 180) % 360) - 180
    lat = pm["latitude"].to_numpy()
    cmap = plt.cm.turbo(np.linspace(0.05, 0.95, len(LABELS)))
    fig = plt.figure(figsize=(14, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="0.85"); ax.coastlines(lw=0.4)
    ax.set_global()
    for b, lab in enumerate(LABELS):
        sel = pm["grp"].to_numpy() == b
        ax.scatter(lon[sel], lat[sel], s=5, color=cmap[b], label=f"{lab} (n={int(sel.sum()):,})",
                   transform=ccrs.PlateCarree(), edgecolor="none", alpha=0.6)
    ax.legend(title="profile max Chl-a (mg/m3)", markerscale=3, fontsize=8,
              loc="lower left", framealpha=0.9)
    ax.set_title("Profile locations colored by profile-max Chl-a group", fontsize=12)
    f2 = Path(f"{args.prefix}_map.png")
    fig.savefig(f2, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved {f2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
