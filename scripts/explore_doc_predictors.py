"""Which variables actually predict GLODAP DOC? Correlate DOC (restricted to a
plausible 30-150 umol/kg range) with AOU, T, S, sigma0, O2, depth and nutrients
on co-located samples, and -- crucially -- the PARTIAL correlation of DOC vs AOU
after removing what the base7 features (log_depth, T, S, lat/lon sin/cos) already
explain. A large partial correlation means AOU carries DOC information beyond the
current model's inputs (i.e. worth adding as a feature / retraining).

Usage:
    uv run python scripts/explore_doc_predictors.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
OUT = ROOT / "figures" / "doc_predictors.png"
CAND = {"AOU": "G2aou", "temperature": "G2temperature", "salinity": "G2salinity",
        "sigma0": "G2sigma0", "O2": "G2oxygen", "depth": "G2depth",
        "NO3": "G2nitrate", "PO4": "G2phosphate"}


def partial_corr(y, x, controls):
    """Pearson corr of y,x after linearly removing `controls` (cols of an (n,k) array)."""
    C = np.column_stack([np.ones(len(y)), controls])
    ry = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
    rx = x - C @ np.linalg.lstsq(C, x, rcond=None)[0]
    return pearsonr(ry, rx)[0]


def main() -> int:
    use = ["G2doc", "G2latitude", "G2longitude"] + list(CAND.values())
    df = pd.read_csv(CSV, usecols=use).replace(-9999, np.nan)
    df = df[(df["G2doc"] >= 30) & (df["G2doc"] <= 150)]
    doc = df["G2doc"]
    print(f"DOC[30,150] present: {doc.notna().sum():,}\n")
    print(f"{'predictor':12s} {'n':>7s} {'Pearson':>8s} {'Spearman':>9s}")
    for name, col in CAND.items():
        m = doc.notna() & df[col].notna()
        x = df.loc[m, col].to_numpy(); y = doc[m].to_numpy()
        print(f"{name:12s} {len(x):7d} {pearsonr(y, x)[0]:+8.3f} {spearmanr(y, x)[0]:+9.3f}")

    # --- partial correlation of DOC vs AOU, controlling for base7-like features ---
    print("\nPartial correlation DOC vs AOU (does AOU add beyond the model's inputs?):")
    base = df.copy()
    base["log_depth"] = np.log(base["G2depth"] + 1.0)
    base["lat_sin"] = np.sin(np.deg2rad(base["G2latitude"]))
    base["lat_cos"] = np.cos(np.deg2rad(base["G2latitude"]))
    base["lon_sin"] = np.sin(np.deg2rad(base["G2longitude"]))
    base["lon_cos"] = np.cos(np.deg2rad(base["G2longitude"]))
    need = ["G2doc", "G2aou", "log_depth", "G2temperature", "G2salinity",
            "lat_sin", "lat_cos", "lon_sin", "lon_cos"]
    b = base.dropna(subset=need)
    y = b["G2doc"].to_numpy(); a = b["G2aou"].to_numpy()
    for label, ctrl in [("depth only", ["log_depth"]),
                        ("depth+T+S", ["log_depth", "G2temperature", "G2salinity"]),
                        ("full base7", ["log_depth", "G2temperature", "G2salinity",
                                        "lat_sin", "lat_cos", "lon_sin", "lon_cos"])]:
        pc = partial_corr(y, a, b[ctrl].to_numpy())
        print(f"  | controlling {label:12s}: partial r = {pc:+.3f}   (raw r = {pearsonr(y, a)[0]:+.3f}, n={len(y):,})")

    # --- scatter panels colored by depth ---
    panels = ["AOU", "temperature", "sigma0", "O2", "NO3", "depth"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, name in zip(axes.ravel(), panels):
        col = CAND[name]
        m = doc.notna() & df[col].notna()
        x = df.loc[m, col].to_numpy(); y = doc[m].to_numpy()
        sc = ax.scatter(x, y, s=7, alpha=0.3, c=df.loc[m, "G2depth"], cmap="viridis_r")
        r = pearsonr(y, x)[0]; rho = spearmanr(y, x)[0]
        ax.text(0.04, 0.96, f"r={r:+.3f}\nrho={rho:+.3f}\nn={len(x)}", transform=ax.transAxes,
                va="top", fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.85))
        ax.set_xlabel(name); ax.set_ylabel("DOC (umol/kg)"); ax.set_title(f"DOC vs {name}", fontsize=11)
        plt.colorbar(sc, ax=ax, pad=0.02).set_label("depth (m)", fontsize=8)
    fig.suptitle("GLODAP DOC[30,150] vs candidate predictors (color = depth)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"\nsaved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
