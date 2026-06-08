"""Obs-vs-model Chl-a scatter (global and Japan box) for the plain allfeat model,
RAW vs after the surface-anchor post-processing the user proposed:

  R_hat   = geometric median of obs/model over the 0-30 m samples of each profile
  R_eff(z)= 1 + (R_hat - 1) * max(0, (200 - z) / 200)   # linear -> 1 at 200 m
  A(z)    = model(z) * R_eff(z)

The anchor uses the OBSERVED 0-30 m Chl-a (best case; at real inference it would be
the satellite surface). NO3 / nutricline use observed NO3 (same as the scoreboard).

Usage:
    uv run python scripts/plot_chla_surface_anchor_scatter.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from bio_params.features import build_features
from bio_params.persist import load_artifact
from bio_params.profiles import add_mld, add_structure_descriptors

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from predict_roms_ini_depths import TRACER_META, predict_field   # noqa: E402

CSV = ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
SPROF = ROOT / "data" / "bgc_argo" / "raw" / "floats"
ART = ROOT / "models" / "pretrained" / "combined_Chla_allfeat.pt"
OUT = ROOT / "figures" / "fora" / "chla_surface_anchor_scatter.png"
FLOOR = 1e-3


def surface_anchor(df, taper_end=200.0):
    """Per profile (lat,lon,time): R_hat from 0-30 m obs/model, linear taper to 1 at
    `taper_end` m (the surface correction fades to none by that depth)."""
    out = df["model"].to_numpy().copy()
    for _, g in df.groupby(["latitude", "longitude", "time"]):
        anc = g[(g.depth <= 30) & (g.Chla > 0) & (g.model > FLOOR)]
        if len(anc) < 1:
            continue
        rhat = 10 ** np.median(np.log10(anc.Chla.to_numpy()
                                        / np.maximum(anc.model.to_numpy(), FLOOR)))
        z = g.depth.to_numpy()
        reff = 1.0 + (rhat - 1.0) * np.clip((taper_end - z) / taper_end, 0.0, 1.0)
        out[g.index.to_numpy()] = g.model.to_numpy() * reff
    return out


def metrics(o, p):
    lin_r2 = 1 - np.sum((p - o) ** 2) / np.sum((o - o.mean()) ** 2)
    k = (o > 0) & (p > 0)
    lo, lp = np.log10(o[k]), np.log10(p[k])
    log_r2 = 1 - np.sum((lp - lo) ** 2) / np.sum((lo - lo.mean()) ** 2)
    log_rmse = np.sqrt(np.mean((lp - lo) ** 2))
    return lin_r2, log_r2, log_rmse, int(k.sum())


def panel(ax, o, p, title):
    k = (o > 0) & (p > 0)
    ax.scatter(o[k], p[k], s=5, alpha=0.25, edgecolors="none", c="tab:green")
    ax.plot([5e-3, 50], [5e-3, 50], "k--", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(5e-3, 50); ax.set_ylim(5e-3, 50); ax.set_aspect("equal", "box")
    lin_r2, log_r2, log_rmse, n = metrics(o, p)
    ax.text(0.04, 0.96, f"linR2={lin_r2:.3f}\nlogR2={log_r2:.3f}\nlogRMSE={log_rmse:.3f}\nn={n}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.set_xlabel("GLODAP obs Chl-a (mg/m3)"); ax.set_ylabel("model Chl-a (mg/m3)")
    ax.set_title(title, fontsize=11)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--taper-end", type=float, default=200.0,
                    help="depth (m) where the surface correction fades to 1 (no change)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from bio_params.loaders.chla_no3 import load_chla_no3
    df = load_chla_no3("glodap", glodap_csv=CSV, sprof_dir=SPROF)
    df = add_mld(df)
    df = add_structure_descriptors(df)
    need = ["mld", "NO3", "Chla", "z_nutr", "nutr_max", "z_pyc", "strat_max"]
    df = df[np.all([np.isfinite(df[c]) for c in need], axis=0)].reset_index(drop=True)

    model, norm, meta = load_artifact(ART, map_location=device); model.to(device)
    cutoff = float(meta["extra"]["cutoff_depth"])
    X = build_features(df, include_mld=True, include_no3=True).to_numpy()
    X = np.column_stack([X, np.log(df.z_nutr.to_numpy() + 1.0), df.nutr_max.to_numpy(),
                         np.log(df.z_pyc.to_numpy() + 1.0), df.strat_max.to_numpy()])
    pred = predict_field(model, norm, X, device, clip=TRACER_META["Chla"]["clip"])
    df["model"] = np.where(df.depth.to_numpy() > cutoff, 0.0, pred)
    df["anchored"] = surface_anchor(df, taper_end=args.taper_end)
    te = args.taper_end
    print(f"rows: {len(df):,}  profiles: {df.groupby(['latitude','longitude','time']).ngroups:,}"
          f"  taper_end={te:.0f}m")

    box = (df.longitude >= 120) & (df.longitude <= 160) & (df.latitude >= 20) & (df.latitude <= 50)
    regions = [("GLOBAL", df), ("JAPAN BOX (120-160E,20-50N)", df[box])]
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for r, (rlabel, d) in enumerate(regions):
        o = d.Chla.to_numpy()
        panel(axes[r, 0], o, d.model.to_numpy(), f"{rlabel}\nRAW allfeat")
        panel(axes[r, 1], o, d.anchored.to_numpy(), f"{rlabel}\n0-30m median + taper->{te:.0f}m")
    fig.suptitle("GLODAP Chl-a obs vs allfeat model: RAW vs surface-anchored "
                 f"(obs 0-30m anchor, linear taper to {te:.0f} m)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUT.with_name(f"chla_surface_anchor_scatter_taper{int(te)}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
