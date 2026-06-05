"""Build and visualize the data-derived typical relative Chl-a profile (base)
used by the base x amplification model (option B).

Bins the GLODAP+BGC-Argo co-located rows by surface Chl (trophic state) and
takes the median rel(z)=Chl(z)/Chl_surf per depth node. Saves the base table
(JSON) for training/inference and plots the per-class typical shapes.

Usage:
    uv run python scripts/build_chla_base_profile.py --per-source-profiles 10000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bio_params.base_profile import DEFAULT_DEPTH_GRID, build_base_profile
from bio_params.loaders.chla_no3 import load_chla_no3
from bio_params.profiles import add_relative_target

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
DEFAULT_SPROF = ROOT / "data" / "bgc_argo" / "raw" / "floats"
DAILY_MATCHUP = ROOT / "data" / "bgc_argo" / "processed" / "satchl_matchup_daily_combined.parquet"
OUT_JSON = ROOT / "data" / "combined" / "processed" / "chla_base_profile.json"
FIG = ROOT / "figures" / "chla_base" / "base_profiles.png"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--per-source-profiles", type=int, default=10000)
    p.add_argument("--rel-cap", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bin-satellite", action="store_true",
                   help="bin by the DAILY satellite surface Chl (matched) instead "
                        "of in-situ Chla_surf, to match ROMS inference (which bins "
                        "by satellite). rel target stays in-situ-normalized.")
    p.add_argument("--matchup", type=Path, default=DAILY_MATCHUP)
    p.add_argument("--out", type=Path, default=OUT_JSON)
    p.add_argument("--quantile", action="store_true",
                   help="equal-count (quantile) surface-Chl bins instead of fixed edges")
    p.add_argument("--n-bins", type=int, default=9, help="number of bins for --quantile")
    p.add_argument("--continuous", action="store_true",
                   help="store per-bin surf centers so eval interpolates smoothly "
                        "over surface Chl (no bin jumps)")
    p.add_argument("--depth-max", type=float, default=300.0,
                   help="extend the base depth grid (and hence the hard 0-cutoff) "
                        "to this depth, with 50 m nodes beyond 300 m")
    p.add_argument("--dark-correct", action="store_true",
                   help="subtract the per-float BGC-Argo fluorescence dark offset")
    p.add_argument("--box", type=float, nargs=4, default=None,
                   metavar=("LON0", "LON1", "LAT0", "LAT1"),
                   help="restrict to a region, e.g. --box 120 160 20 50 (Japan)")
    args = p.parse_args()
    if args.depth_max > DEFAULT_DEPTH_GRID[-1]:
        extra = np.arange(DEFAULT_DEPTH_GRID[-1] + 50.0, args.depth_max + 1, 50.0)
        depth_grid = np.concatenate([DEFAULT_DEPTH_GRID, extra])
    else:
        depth_grid = DEFAULT_DEPTH_GRID

    df = load_chla_no3("combined", glodap_csv=DEFAULT_CSV, sprof_dir=DEFAULT_SPROF,
                       dark_correct=args.dark_correct,
                       box=tuple(args.box) if args.box else None)
    df = add_relative_target(df, "Chla", rel_cap=args.rel_cap)
    if args.bin_satellite:
        from bio_params.loaders.bgc_argo import attach_surface_chla
        n0 = len(df)
        df = attach_surface_chla(df, args.matchup)               # daily satellite surf
        df = df[np.isfinite(df["surface_chla"])].reset_index(drop=True)
        print(f"  binning by DAILY satellite surf: {len(df):,}/{n0:,} rows matched")
    bin_key = "surface_chla" if args.bin_satellite else "Chla_surf"
    if args.per_source_profiles and "source" in df.columns:
        keys = ["latitude", "longitude", "time"]; rng = np.random.default_rng(args.seed)
        parts = []
        for src, sub in df.groupby("source"):
            pr = sub[keys].drop_duplicates()
            if len(pr) > args.per_source_profiles:
                pr = pr.iloc[np.sort(rng.choice(len(pr), args.per_source_profiles, replace=False))]
            parts.append(sub.merge(pr, on=keys, how="inner"))
        import pandas as pd
        df = pd.concat(parts, ignore_index=True)
    print(f"rows: {len(df):,}  (glodap={int((df.source=='glodap').sum()):,}, "
          f"bgc={int((df.source=='bgc_argo').sum()):,})")

    base = build_base_profile(df[bin_key].to_numpy(), df["depth"].to_numpy(),
                              df["Chla_rel"].to_numpy(), depth_grid=depth_grid,
                              quantile=args.quantile, n_bins=args.n_bins,
                              continuous=args.continuous)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(base.to_dict(), indent=2))
    print(f"saved base table -> {args.out}  (bin key: {bin_key})")

    edges, zg = base.surf_edges, base.depth_grid
    print(f"\n{'surf-Chl class':>18} | {'profiles':>8} | typical DCM (max rel, depth)")
    for b in range(len(edges) - 1):
        lab = f"{edges[b]:.2f}-{edges[b+1]:.2f}" if edges[b+1] < 100 else f">{edges[b]:.2f}"
        nprof = int(base.counts[b].max())
        imax = int(np.argmax(base.table[b]))
        print(f"  {lab:>16} | {nprof:>8} | rel_max={base.table[b, imax]:.2f} @ {zg[imax]:.0f} m")

    fig, ax = plt.subplots(figsize=(6, 7))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(edges) - 1))
    for b in range(len(edges) - 1):
        lab = f"{edges[b]:.2f}-{edges[b+1]:.2f}" if edges[b+1] < 100 else f">{edges[b]:.2f}"
        ax.plot(base.table[b], zg, "-o", ms=3, color=cmap[b], label=lab)
    ax.axvline(1.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("rel = Chl(z) / Chl_surf (median)"); ax.set_ylabel("depth (m)")
    ax.set_ylim(zg.max(), 0); ax.set_title("Data-derived typical Chl-a profiles by surface Chl")
    ax.legend(title="surface Chl (mg/m3)", fontsize=8, loc="lower right")
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG, dpi=130); print(f"saved figure -> {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
