"""Sanity-check the SOCA premise: does satellite surface Chl-a match the
in-situ surface Chl-a from BGC-Argo / GLODAP?

For each source, take the shallowest in-situ Chl-a measurement of each profile
(within SURF_MAX_DEPTH m), join it to the matched monthly GlobColour surface
value (from the matchup parquet), and scatter satellite (x) vs in-situ (y) on
log-log axes. Reports n, log-space R^2, RMSE, median ratio and the regression
slope so we can judge how trustworthy the satellite "truth" anchor is.

Usage:
    uv run python scripts/plot_satellite_vs_insitu_surface.py
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV = PROJECT_ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
SPROF = PROJECT_ROOT / "data" / "bgc_argo" / "raw" / "floats"
PROC = PROJECT_ROOT / "data" / "bgc_argo" / "processed"
OUT_DIR = PROJECT_ROOT / "figures" / "satellite_vs_insitu"

SURF_MAX_DEPTH = 10.0  # m; "surface" = shallowest measurement within this depth
CHLA_FLOOR = 1e-3      # mg/m3, for log axes


def surface_insitu(df: pd.DataFrame) -> pd.DataFrame:
    """Shallowest Chla measurement (<= SURF_MAX_DEPTH m) per profile."""
    df = df[df["depth"] <= SURF_MAX_DEPTH].copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("depth")
    surf = (df.groupby(["latitude", "longitude", "time"], as_index=False)
              .first()[["latitude", "longitude", "time", "Chla", "depth"]])
    return surf.rename(columns={"Chla": "insitu", "depth": "surf_depth"})


def load_source(source: str) -> pd.DataFrame:
    if source == "bgc_argo":
        df = load_bgc_argo(SPROF, "Chla")
        matchup = PROC / "satchl_matchup.parquet"
    else:
        df = load_glodap(CSV, "Chla", with_time=True)
        matchup = PROC / "satchl_matchup_glodap.parquet"
    surf = surface_insitu(df)
    m = pd.read_parquet(matchup)
    m["time"] = pd.to_datetime(m["time"])
    j = surf.merge(m[["latitude", "longitude", "time", "surface_chla"]],
                   on=["latitude", "longitude", "time"], how="inner")
    j = j[np.isfinite(j["insitu"]) & np.isfinite(j["surface_chla"])
          & (j["insitu"] > 0) & (j["surface_chla"] > 0)].reset_index(drop=True)
    return j.rename(columns={"surface_chla": "satellite"})


def stats(sat, ins):
    ls, li = np.log10(sat), np.log10(ins)
    ss_res = float(((li - ls) ** 2).sum())
    ss_tot = float(((li - li.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse_log = float(np.sqrt(np.mean((li - ls) ** 2)))
    slope = float(np.polyfit(ls, li, 1)[0])
    med_ratio = float(np.median(ins / sat))
    return dict(n=len(sat), r2_log=r2, rmse_log=rmse_log, slope=slope,
                med_ratio=med_ratio)


def scatter(ax, sat, ins, title, log_axes=True):
    if log_axes:
        lo = max(CHLA_FLOOR, min(sat.min(), ins.min()))
        hi = max(sat.max(), ins.max())
        extent = (np.log10(lo), np.log10(hi), np.log10(lo), np.log10(hi))
        kw = dict(xscale="log", yscale="log")
    else:
        # Linear axes: cap at the 99th percentile so the bulk is visible
        # (a few high-Chl points would otherwise compress everything to 0).
        lo = 0.0
        hi = float(np.percentile(np.concatenate([sat, ins]), 99))
        extent = (lo, hi, lo, hi)
        kw = {}
    if len(sat) < 2000:
        ax.scatter(sat, ins, s=10, alpha=0.4, color="steelblue", edgecolor="none")
    else:
        hb = ax.hexbin(sat, ins, gridsize=70, bins="log", cmap="viridis",
                       mincnt=1, extent=extent, **kw)
        plt.colorbar(hb, ax=ax).set_label("log10(count)")
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="1:1")
    if log_axes:
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("Satellite surface Chl-a (mg/m3)")
    ax.set_ylabel("In-situ surface Chl-a (mg/m3)")
    ax.set_title(title, fontsize=10); ax.legend(loc="upper left", fontsize=8)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sources", nargs="+", default=["bgc_argo", "glodap"])
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load each source once (loading is the expensive part), then draw both
    # log-axis and linear-axis versions.
    data = {}
    for src in args.sources:
        j = load_source(src)
        s = stats(j["satellite"].to_numpy(), j["insitu"].to_numpy())
        data[src] = (j, s)
        print(f"{src}: n={s['n']:,}  log-R2={s['r2_log']:.3f}  "
              f"RMSE(log10)={s['rmse_log']:.3f}  slope={s['slope']:.3f}  "
              f"median(insitu/sat)={s['med_ratio']:.2f}")

    for log_axes, fname in [(True, "satellite_vs_insitu_surface.png"),
                            (False, "satellite_vs_insitu_surface_linear.png")]:
        fig, axes = plt.subplots(1, len(args.sources),
                                 figsize=(7 * len(args.sources), 6.5))
        if len(args.sources) == 1:
            axes = [axes]
        for ax, src in zip(axes, args.sources):
            j, s = data[src]
            axtype = "log-log" if log_axes else "linear"
            scatter(ax, j["satellite"].to_numpy(), j["insitu"].to_numpy(),
                    f"{src}: satellite vs in-situ surface Chl-a "
                    f"(<= {SURF_MAX_DEPTH:.0f} m, {axtype})\n"
                    f"n={s['n']:,}  log-R2={s['r2_log']:.3f}  "
                    f"RMSE(log10)={s['rmse_log']:.3f}  slope={s['slope']:.2f}  "
                    f"med(in/sat)={s['med_ratio']:.2f}", log_axes=log_axes)
        fig.tight_layout()
        fig.savefig(OUT_DIR / fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {OUT_DIR / fname}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
