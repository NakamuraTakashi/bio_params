"""Quantify, per surface-Chl bin, the depth at which Chl-a effectively reaches 0,
to reconsider the model's hard 0-cutoff (currently 300 m, base-grid limit).

For each bin (the qc base surf-Chl edges) it builds the observed median Chl(z)
profile to ~1500 m and reports the depth where the median drops below thresholds
(0.05 / 0.02 / 0.01 mg/m3), plus a conservative p75 crossing. It also prints the
deep (500-1000 m) median Chl BY SOURCE, because BGC-Argo fluorescence can carry a
near-detection dark offset at depth (an apparent, not real, Chl tail) whereas
GLODAP (extracted/HPLC) is trustworthy though sparse.

Uses ALL Chl-a profiles (GLODAP + BGC-Argo), not the NO3 co-located subset.

Usage:
    uv run python scripts/analyze_chla_zero_depth.py
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

from bio_params.loaders.bgc_argo import load_bgc_argo
from bio_params.loaders.glodap import load_glodap
from bio_params.profiles import add_relative_target

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
SPROF = ROOT / "data" / "bgc_argo" / "raw" / "floats"
BASE = ROOT / "data" / "combined" / "processed" / "chla_base_profile_qc.json"
FIG = ROOT / "figures" / "chla_base" / "chla_zero_depth.png"
ZGRID = np.concatenate([np.arange(0, 300, 10.0), np.arange(300, 1000, 50.0),
                        np.arange(1000, 1501, 100.0)])


def _cross_depth(znodes, med, thr):
    """Deepest node with median >= thr (-> below it Chl < thr)."""
    ok = np.isfinite(med) & (med >= thr)
    return float(znodes[ok].max()) if ok.any() else float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rel-cap", type=float, default=20.0)
    p.add_argument("--out", type=Path, default=FIG)
    args = p.parse_args()

    edges = np.asarray(json.loads(BASE.read_text())["surf_edges"], float)
    nb = len(edges) - 1
    g = load_glodap(CSV, "Chla", with_time=True); g["source"] = "glodap"
    a = load_bgc_argo(SPROF, "Chla"); a["source"] = "bgc_argo"
    cols = ["latitude", "longitude", "depth", "Chla", "time", "source"]
    df = pd.concat([g[cols], a[cols]], ignore_index=True)
    df = add_relative_target(df, "Chla", rel_cap=args.rel_cap)
    df = df[np.isfinite(df["Chla"]) & np.isfinite(df["depth"])].reset_index(drop=True)
    surf = df["Chla_surf"].to_numpy(); depth = df["depth"].to_numpy()
    chl = df["Chla"].to_numpy(); src = df["source"].to_numpy()
    sb = np.clip(np.digitize(surf, edges) - 1, 0, nb - 1)
    mids = (ZGRID[:-1] + ZGRID[1:]) / 2.0
    di = np.digitize(depth, mids)

    print(f"{'surf-Chl bin':>13} | {'n':>9} | depth(m) where median Chl < thr "
          f"| deep 500-1000m median (GLO/BGC)")
    print(f"{'':>13} | {'':>9} | {'0.05':>6} {'0.02':>6} {'0.01':>6} {'p75<0.02':>9} |")
    meds = []
    for b in range(nb):
        m = sb == b
        med = np.full(len(ZGRID), np.nan); p75 = med.copy()
        for k in range(len(ZGRID)):
            ck = chl[m & (di == k)]
            if ck.size >= 20:
                med[k] = np.median(ck); p75[k] = np.percentile(ck, 75)
        meds.append(med)
        z05, z02, z01 = (_cross_depth(ZGRID, med, t) for t in (0.05, 0.02, 0.01))
        z75 = _cross_depth(ZGRID, p75, 0.02)
        deep = m & (depth >= 500) & (depth < 1000)
        dg = chl[deep & (src == "glodap")]; dbb = chl[deep & (src == "bgc_argo")]
        dgm = f"{np.median(dg):.3f}" if dg.size >= 10 else "  -  "
        dbm = f"{np.median(dbb):.3f}" if dbb.size >= 10 else "  -  "
        lab = f"{edges[b]:.2f}-{edges[b+1]:.2f}" if edges[b+1] < 100 else f">{edges[b]:.2f}"
        print(f"{lab:>13} | {int(m.sum()):>9,} | {z05:>6.0f} {z02:>6.0f} {z01:>6.0f} "
              f"{z75:>9.0f} | GLO {dgm} / BGC {dbm}")

    # figure: per-bin median Chl(z) on log-x to 1000 m
    fig, ax = plt.subplots(figsize=(7.5, 8))
    cmap = plt.cm.viridis(np.linspace(0, 1, nb))
    for b in range(nb):
        lab = f"{edges[b]:.2f}-{edges[b+1]:.2f}" if edges[b+1] < 100 else f">{edges[b]:.2f}"
        ok = np.isfinite(meds[b])
        ax.plot(np.clip(meds[b][ok], 1e-4, None), ZGRID[ok], "-o", ms=2, color=cmap[b], label=lab)
    for t in (0.05, 0.02, 0.01):
        ax.axvline(t, color="0.6", lw=0.6, ls=":")
    ax.axhline(300, color="red", lw=1.0, ls="--", label="current cutoff 300 m")
    ax.set_xscale("log"); ax.set_ylim(1000, 0)
    ax.set_xlabel("median Chl-a (mg/m3)"); ax.set_ylabel("depth (m)")
    ax.set_title("Per-bin median Chl-a profile to 1000 m (where does it reach ~0?)")
    ax.legend(title="surface Chl (mg/m3)", fontsize=7, loc="lower right")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
