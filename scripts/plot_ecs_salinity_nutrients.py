"""Salinity vs NO3 / PO4 in an East China Sea box.

Reports the minimum salinity in the box and fits NO3-salinity and
PO4-salinity relationships by ordinary least squares (with intercept), both
for all depths and for the surface layer only (depth <= SURFACE_MAX_DEPTH).
Low-salinity, high-nutrient water here reflects river/coastal input, so the
surface mixing line is the more meaningful relationship.

Usage:
    uv run python scripts/plot_ecs_salinity_nutrients.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bio_params.loaders.glodap import MISSING_SENTINEL

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV = PROJECT_ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
OUT_DIR = PROJECT_ROOT / "figures" / "ecs_salinity"

BOX = dict(lon0=118.0, lon1=124.0, lat0=25.0, lat1=32.0)
SURFACE_MAX_DEPTH = 20.0  # m; surface layer cut for the surface-only fit


def fit_ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """OLS y = slope*x + intercept; returns (slope, intercept, r2)."""
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def scatter_fit(sal, val, *, nutrient, unit, out_path, scope) -> None:
    if len(sal) < 2:
        print(f"  {nutrient} [{scope}]: too few points (n={len(sal)}); skipped")
        return
    slope, intercept, r2 = fit_ols(sal, val)
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    if len(sal) < 3000:
        ax.scatter(sal, val, s=12, alpha=0.5, edgecolor="none", color="steelblue")
    else:
        hb = ax.hexbin(sal, val, gridsize=60, cmap="viridis", bins="log", mincnt=1)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("log10(count)")
    xs = np.array([sal.min(), sal.max()])
    sign = "+" if intercept >= 0 else "-"
    ax.plot(xs, slope * xs + intercept, "r-", lw=1.8,
            label=f"OLS: {nutrient} = {slope:.3f}*S {sign} {abs(intercept):.2f}  (R2={r2:.3f})")
    ax.set_xlabel("Salinity (PSU)")
    ax.set_ylabel(f"{nutrient} ({unit})")
    ax.set_title(f"GLODAP {nutrient} vs salinity, ECS box ({scope})\n"
                 f"lon[{BOX['lon0']:.0f},{BOX['lon1']:.0f}] "
                 f"lat[{BOX['lat0']:.0f},{BOX['lat1']:.0f}]  n={len(sal):,}",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  {nutrient} [{scope}]: n={len(sal):,}  {nutrient} = {slope:.4f}*S "
          f"{sign} {abs(intercept):.3f}  R2={r2:.4f}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    usecols = ["G2latitude", "G2longitude", "G2depth",
               "G2salinity", "G2salinityf",
               "G2nitrate", "G2nitratef", "G2phosphate", "G2phosphatef"]
    print(f"Reading {CSV} ...")
    df = pd.read_csv(CSV, usecols=usecols, low_memory=False).replace(MISSING_SENTINEL, np.nan)

    inbox = ((df.G2longitude >= BOX["lon0"]) & (df.G2longitude <= BOX["lon1"])
             & (df.G2latitude >= BOX["lat0"]) & (df.G2latitude <= BOX["lat1"]))
    b = df[inbox]
    print(f"  rows in box: {len(b):,}")

    # Minimum salinity (flag==2 only, the trustworthy measurements).
    sal_good = b[b.G2salinityf == 2]["G2salinity"].dropna()
    print(f"\n=== Salinity in box ===")
    print(f"  min (flag==2): {sal_good.min():.3f} PSU   (n={len(sal_good):,})")
    print(f"  salinity range (flag==2): [{sal_good.min():.3f}, {sal_good.max():.3f}]")
    surf = b[(b.G2depth <= SURFACE_MAX_DEPTH) & (b.G2salinityf == 2)]["G2salinity"].dropna()
    if len(surf):
        print(f"  surface (<= {SURFACE_MAX_DEPTH:.0f} m) min: {surf.min():.3f} PSU  (n={len(surf):,})")

    for scope, depth_cut in [("all-depth", None), (f"surface<= {SURFACE_MAX_DEPTH:.0f}m", SURFACE_MAX_DEPTH)]:
        tag = "all" if depth_cut is None else "surface"
        print(f"\n=== OLS fits [{scope}] (salinity vs nutrient, both flag==2) ===")
        depth_ok = b.G2depth <= depth_cut if depth_cut is not None else pd.Series(True, index=b.index)

        m = depth_ok & (b.G2salinityf == 2) & (b.G2nitratef == 2) & b.G2salinity.notna() & b.G2nitrate.notna()
        scatter_fit(b.loc[m, "G2salinity"].to_numpy(), b.loc[m, "G2nitrate"].to_numpy(),
                    nutrient="NO3", unit="umol/kg", scope=scope,
                    out_path=OUT_DIR / f"salinity_no3_{tag}.png")

        m = depth_ok & (b.G2salinityf == 2) & (b.G2phosphatef == 2) & b.G2salinity.notna() & b.G2phosphate.notna()
        scatter_fit(b.loc[m, "G2salinity"].to_numpy(), b.loc[m, "G2phosphate"].to_numpy(),
                    nutrient="PO4", unit="umol/kg", scope=scope,
                    out_path=OUT_DIR / f"salinity_po4_{tag}.png")

    print(f"\nSaved figures -> {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
