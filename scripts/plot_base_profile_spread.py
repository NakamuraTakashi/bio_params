"""Plot the per-bin relative Chl-a vertical profiles WITH the observed spread.

For each surface-Chl bin of a base table, scatter every observed
rel(z) = Chl(z)/Chl_surf as a density cloud, overlay the bin's median base
profile (the curve the model uses) and the 16-84 percentile band, so the
within-bin variability of the vertical shape is visible.

Usage:
    uv run python scripts/plot_base_profile_spread.py            # qc base (12 quantile bins)
    uv run python scripts/plot_base_profile_spread.py --base-json data/combined/processed/chla_base_profile.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bio_params.base_profile import BaseProfile
from bio_params.loaders.chla_no3 import load_chla_no3
from bio_params.profiles import add_relative_target

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
DEFAULT_SPROF = ROOT / "data" / "bgc_argo" / "raw" / "floats"
DEFAULT_BASE = ROOT / "data" / "combined" / "processed" / "chla_base_profile_qc.json"
FIG = ROOT / "figures" / "chla_base" / "base_profiles_spread.png"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-json", type=Path, default=DEFAULT_BASE)
    p.add_argument("--per-source-profiles", type=int, default=10000)
    p.add_argument("--rel-cap", type=float, default=20.0)
    p.add_argument("--max-depth", type=float, default=300.0)
    p.add_argument("--absolute", action="store_true",
                   help="x-axis = absolute Chl-a (mg/m3) instead of rel; the base "
                        "curve is scaled by the bin's median surface Chl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=FIG)
    args = p.parse_args()

    base = BaseProfile.from_dict(json.loads(args.base_json.read_text()))
    edges, zg, table = base.surf_edges, base.depth_grid, base.table
    nb = len(edges) - 1

    df = load_chla_no3("combined", glodap_csv=DEFAULT_CSV, sprof_dir=DEFAULT_SPROF)
    df = add_relative_target(df, "Chla", rel_cap=args.rel_cap)
    if args.per_source_profiles and "source" in df.columns:
        keys = ["latitude", "longitude", "time"]; rng = np.random.default_rng(args.seed)
        parts = []
        for _, sub in df.groupby("source"):
            pr = sub[keys].drop_duplicates()
            if len(pr) > args.per_source_profiles:
                pr = pr.iloc[np.sort(rng.choice(len(pr), args.per_source_profiles, replace=False))]
            parts.append(sub.merge(pr, on=keys, how="inner"))
        df = pd.concat(parts, ignore_index=True)

    surf = df["Chla_surf"].to_numpy(); depth = df["depth"].to_numpy()
    rel = (df["Chla"].to_numpy() if args.absolute else df["Chla_rel"].to_numpy())
    sb = np.clip(np.digitize(surf, edges) - 1, 0, nb - 1)
    keep = np.isfinite(rel) & np.isfinite(depth) & (depth <= args.max_depth)
    # per-bin median surface Chl, to scale the (relative) base curve to absolute
    surf_rep = np.array([np.nanmedian(surf[sb == b]) if (sb == b).any() else np.nan
                         for b in range(nb)])
    xlabel = "Chl-a (mg/m3)" if args.absolute else "rel = Chl(z)/Chl_surf"

    ncol = 4; nrow = int(np.ceil(nb / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow),
                             sharey=True)
    axes = np.atleast_1d(axes).ravel()
    # Observed percentiles on an OWN grid extended to max_depth (the base grid only
    # reaches 300 m; we want to see how much Chl persists below where the model = 0).
    og = np.unique(np.concatenate([zg, np.arange(350.0, args.max_depth + 1, 50.0)]))
    og = og[og <= args.max_depth]
    mids = (og[:-1] + og[1:]) / 2.0
    for b in range(nb):
        ax = axes[b]
        m = keep & (sb == b)
        r, d = rel[m], depth[m]
        xhi = max(3.0, float(np.percentile(r, 99.0))) if r.size else 3.0
        if r.size > 500:
            ax.hexbin(r, d, gridsize=40, bins="log", cmap="Blues", mincnt=1,
                      extent=(0, xhi, 0, args.max_depth))
        else:
            ax.scatter(r, d, s=6, alpha=0.3, color="steelblue", edgecolor="none")
        # observed 16/50/84 percentile per depth node (extended grid)
        di = np.digitize(d, mids)
        p16 = np.full(len(og), np.nan); p50 = p16.copy(); p84 = p16.copy()
        for k in range(len(og)):
            rk = r[di == k]
            if rk.size >= 20:
                p16[k], p50[k], p84[k] = np.percentile(rk, [16, 50, 84])
        ok = np.isfinite(p16)
        ax.fill_betweenx(og[ok], p16[ok], p84[ok], color="orange", alpha=0.25,
                         label="obs 16-84%")
        ax.plot(p50[ok], og[ok], color="orange", lw=1.2, ls="--", label="obs median")
        bcurve = table[b] * surf_rep[b] if args.absolute else table[b]
        ax.plot(bcurve, zg, color="red", lw=2.0, label="base (model)")
        if not args.absolute:
            ax.axvline(1.0, color="k", lw=0.5, ls=":")
        lab = f"{edges[b]:.2f}-{edges[b+1]:.2f}" if edges[b + 1] < 100 else f">{edges[b]:.2f}"
        ax.set_title(f"surf Chl {lab} mg/m3  (n={int(m.sum()):,})", fontsize=9)
        ax.set_xlim(0, xhi); ax.set_ylim(args.max_depth, 0)
        if b % ncol == 0:
            ax.set_ylabel("depth (m)")
        if b >= nb - ncol:
            ax.set_xlabel(xlabel)
        if b == 0:
            ax.legend(fontsize=7, loc="lower right")
    for b in range(nb, len(axes)):
        axes[b].axis("off")
    kind = "absolute" if args.absolute else "relative"
    fig.suptitle(f"Per-bin {kind} Chl-a profiles: observed spread vs base "
                 f"({args.base_json.name})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
