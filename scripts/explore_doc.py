"""Quick exploration of GLODAP DOC behaviour: correlations of DOC with Chl-a,
DON, and TOC on co-located samples (rows where BOTH are measured). These are all
non-core (un-QC'd) GLODAP variables, so we use value-present (not flag==2) and
report counts/ranges. Pearson (linear) + Spearman (rank) + co-located depth.

Usage:
    uv run python scripts/explore_doc.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
OUT = ROOT / "figures" / "doc_correlations.png"
COLS = {"DOC": "G2doc", "Chla": "G2chla", "DON": "G2don", "TOC": "G2toc"}
UNIT = {"DOC": "umol/kg", "Chla": "mg/m3", "DON": "umol/kg", "TOC": "umol/kg"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--doc-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="restrict DOC to a plausible range (umol/kg), e.g. 30 150, "
                         "to drop implausible outliers; writes a _clean figure")
    args = ap.parse_args()
    use = ["G2depth", "G2latitude", "G2longitude"] + list(COLS.values())
    df = pd.read_csv(CSV, usecols=use).replace(-9999, np.nan)
    n_doc = int(df["G2doc"].notna().sum())
    if args.doc_range:
        lo, hi = args.doc_range
        df = df[(df["G2doc"].isna()) | ((df["G2doc"] >= lo) & (df["G2doc"] <= hi))]
        print(f"DOC restricted to [{lo:.0f},{hi:.0f}] umol/kg")
    print(f"GLODAP rows: {len(df):,}   DOC present: {n_doc:,}")
    print(f"DOC range: {df['G2doc'].min():.1f}..{df['G2doc'].max():.1f} umol/kg  "
          f"(median {df['G2doc'].median():.1f})\n")

    pairs = [("DOC", "Chla"), ("DOC", "DON"), ("DOC", "TOC")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (a, b) in zip(axes, pairs):
        ca, cb = COLS[a], COLS[b]
        m = df[ca].notna() & df[cb].notna()
        x = df.loc[m, ca].to_numpy(); y = df.loc[m, cb].to_numpy()
        n = len(x)
        if n >= 3:
            r, _ = pearsonr(x, y); rho, _ = spearmanr(x, y)
            dep = df.loc[m, "G2depth"]
            print(f"{a} vs {b:4s}: n={n:6d}  Pearson r={r:+.3f}  Spearman rho={rho:+.3f}  "
                  f"| {b} range {y.min():.2f}..{y.max():.2f}  co-loc depth med={dep.median():.0f}m "
                  f"(p90={dep.quantile(.9):.0f}m)")
            sc = ax.scatter(x, y, s=8, alpha=0.35, c=df.loc[m, "G2depth"], cmap="viridis_r")
            txt = f"n={n}\nPearson r={r:+.3f}\nSpearman={rho:+.3f}"
            ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.85))
            if a == "DOC" and b == "TOC":      # 1:1 reference for TOC>=DOC
                lo = float(min(x.min(), y.min())); hi = float(max(x.max(), y.max()))
                ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="1:1")
                ax.legend(fontsize=8, loc="lower right")
            plt.colorbar(sc, ax=ax, pad=0.02).set_label("depth (m)", fontsize=8)
        else:
            print(f"{a} vs {b}: n={n} (too few)")
            ax.text(0.5, 0.5, f"n={n}", ha="center", transform=ax.transAxes)
        ax.set_xlabel(f"{a} ({UNIT[a]})"); ax.set_ylabel(f"{b} ({UNIT[b]})")
        ax.set_title(f"{a} vs {b}", fontsize=11)
    rng_note = f"  DOC in [{args.doc_range[0]:.0f},{args.doc_range[1]:.0f}]" if args.doc_range else ""
    fig.suptitle(f"GLODAP DOC correlations (co-located samples; color = depth){rng_note}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = OUT.with_name("doc_correlations_clean.png") if args.doc_range else OUT
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
