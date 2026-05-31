"""Prediction-vs-observation scatter for GLODAP+BGC-Argo combined models.

Loads the combined dataset, predicts with the combined_<target>.pt model, and
draws hexbin obs-vs-pred plots, colored/split so the two sources are visible:
  figures/combined/<target>/scatter_all.png      - all points (global)
  figures/combined/<target>/scatter_glodap.png   - GLODAP rows only
  figures/combined/<target>/scatter_bgc_argo.png - BGC-Argo rows only

The per-source split shows whether the model fits the deep/coastal GLODAP rows
(BGC-Argo's blind spots) as well as the open-ocean Argo rows.

Usage:
    uv run python scripts/plot_combined_predictions.py --target NO3
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from bio_params.features import build_features
from bio_params.loaders.combined import available_targets, load_combined
from bio_params.persist import load_artifact

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
DEFAULT_SPROF = PROJECT_ROOT / "data" / "bgc_argo" / "raw" / "floats"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "pretrained"
DEFAULT_FIG_DIR = PROJECT_ROOT / "figures" / "combined"

UNITS = {"Chla": "mg/m3", "O2": "umol/kg", "NO3": "umol/kg"}


@dataclass
class FitStats:
    n: int
    rmse: float
    r2: float


def compute_stats(obs, pred) -> FitStats:
    err = pred - obs
    ss_res = float((err ** 2).sum())
    ss_tot = float(((obs - obs.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return FitStats(len(obs), float(np.sqrt((err ** 2).mean())), r2)


def predict_all(model, normalizer, X, device, batch=200_000):
    model.eval()
    Xn = normalizer.transform_x(X).astype(np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(Xn), batch):
            out.append(model(torch.from_numpy(Xn[i:i + batch]).to(device)).cpu().numpy())
    return normalizer.inverse_transform_y(np.concatenate(out))


def scatter(obs, pred, *, unit, title, out_path):
    if len(obs) == 0:
        return
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    lo = float(min(obs.min(), pred.min()))
    hi = float(max(obs.max(), pred.max()))
    pad = 0.02 * (hi - lo); lo -= pad; hi += pad
    if len(obs) < 3000:
        ax.scatter(obs, pred, s=8, alpha=0.4, edgecolor="none", color="steelblue")
    else:
        hb = ax.hexbin(obs, pred, gridsize=80, cmap="viridis", bins="log",
                       mincnt=1, extent=(lo, hi, lo, hi))
        fig.colorbar(hb, ax=ax).set_label("log10(count)")
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.0, alpha=0.7, label="1:1")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel(f"Observation ({unit})")
    ax.set_ylabel(f"Model prediction ({unit})")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target", required=True, choices=available_targets())
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--sprof-dir", type=Path, default=DEFAULT_SPROF)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    p.add_argument("--per-source-max", type=int, default=1000000,
                   help="Match the training subsample for a fair in-sample view")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    target = args.target
    unit = UNITS.get(target, "")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    art = args.model_dir / f"combined_{target}.pt"
    if not art.exists():
        print(f"ERROR: artifact not found at {art}")
        return 1
    model, normalizer, meta = load_artifact(art, map_location=device)
    model.to(device)
    cv_r2 = meta["extra"].get("cv_r2_mean")
    print(f"Loading combined {target} ...  CV R2={cv_r2:.4f}")

    df = load_combined(args.csv, args.sprof_dir, target=target,
                       per_source_max=args.per_source_max, seed=42)
    X = build_features(df).to_numpy()
    y = df[target].to_numpy()
    src = df["source"].to_numpy()
    pred = predict_all(model, normalizer, X, device)

    for label, mask in [("all", np.ones(len(df), bool)),
                        ("glodap", src == "glodap"),
                        ("bgc_argo", src == "bgc_argo")]:
        s = compute_stats(y[mask], pred[mask])
        print(f"  {label}: n={s.n:,}  RMSE={s.rmse:.3g}  R2={s.r2:.4f}")
        scatter(y[mask], pred[mask], unit=unit,
                title=(f"{target}: combined model vs obs ({label})\n"
                       f"n={s.n:,}  RMSE={s.rmse:.3g} {unit}  R2={s.r2:.4f}  "
                       f"(honest CV R2={cv_r2:.4f})"),
                out_path=args.fig_dir / target / f"scatter_{label}.png")

    print(f"Saved figures -> {args.fig_dir / target}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
