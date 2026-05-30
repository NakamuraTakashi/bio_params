"""Plot model predictions vs observations for a saved BGC-Argo artifact.

Mirrors scripts/plot_predictions.py (GLODAP) but loads data via the BGC-Argo
loader and honors the artifact's log_target / season flags. Produces two
hexbin scatter plots per target:
  figures/bgc_argo/<target>/scatter_box.png    - all rows in the Kuroshio box
  figures/bgc_argo/<target>/scatter_japan.png  - rows inside the Japan box

NOTE: BGC-Argo here is only downloaded for the Kuroshio box (120-180E,
10-50N), so "box" is that region, not the true global ocean. The scatter is an
in-sample fit; the honest score is the spatial CV in the artifact (annotated).

Usage:
    uv run python scripts/plot_bgc_argo_predictions.py --target Chla
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
from bio_params.loaders.bgc_argo import available_targets, load_bgc_argo
from bio_params.persist import load_artifact

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPROF_DIR = PROJECT_ROOT / "data" / "bgc_argo" / "raw" / "floats"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "pretrained"
DEFAULT_FIG_DIR = PROJECT_ROOT / "figures" / "bgc_argo"

KUROSHIO_BOX = (120.0, 180.0, 10.0, 50.0)
JAPAN_BOX = dict(lat_min=20.0, lat_max=50.0, lon_min=120.0, lon_max=155.0)

UNITS = {"Chla": "mg/m3", "O2": "umol/kg", "NO3": "umol/kg"}


@dataclass
class FitStats:
    n: int
    rmse: float
    mae: float
    r2: float
    bias: float


def compute_stats(obs: np.ndarray, pred: np.ndarray) -> FitStats:
    err = pred - obs
    ss_res = float((err ** 2).sum())
    ss_tot = float(((obs - obs.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return FitStats(
        n=int(len(obs)),
        rmse=float(np.sqrt((err ** 2).mean())),
        mae=float(np.abs(err).mean()),
        r2=r2,
        bias=float(err.mean()),
    )


def predict_all(model, normalizer, X, log_target, batch=200_000) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    X_n = normalizer.transform_x(X).astype(np.float32)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_n), batch):
            xb = torch.from_numpy(X_n[i:i + batch]).to(device)
            preds.append(model(xb).cpu().numpy())
    pred = normalizer.inverse_transform_y(np.concatenate(preds))
    return np.power(10.0, pred) if log_target else pred


def scatter_hexbin(obs, pred, *, unit, title, out_path, log_axes=False) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    if log_axes:
        lo = float(min(obs[obs > 0].min(), pred[pred > 0].min()))
        hi = float(max(obs.max(), pred.max()))
        mask = (obs > 0) & (pred > 0)
        obs, pred = obs[mask], pred[mask]
    else:
        lo = float(min(obs.min(), pred.min()))
        hi = float(max(obs.max(), pred.max()))
        pad = 0.02 * (hi - lo)
        lo -= pad
        hi += pad

    if len(obs) < 3000:
        ax.scatter(obs, pred, s=8, alpha=0.4, edgecolor="none", color="steelblue")
    else:
        kw = dict(gridsize=80, cmap="viridis", bins="log", mincnt=1)
        if log_axes:
            kw["xscale"] = "log"
            kw["yscale"] = "log"
        hb = ax.hexbin(obs, pred, extent=(np.log10(lo), np.log10(hi),
                       np.log10(lo), np.log10(hi)) if log_axes else (lo, hi, lo, hi),
                       **kw)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("log10(count)")

    ax.plot([lo, hi], [lo, hi], "r--", lw=1.0, alpha=0.7, label="1:1")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_aspect("equal")
    ax.set_xlabel(f"Observation ({unit})")
    ax.set_ylabel(f"Model prediction ({unit})")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def stats_text(s: FitStats, unit: str) -> str:
    return (f"n={s.n:,}  RMSE={s.rmse:.3g} {unit}  "
            f"MAE={s.mae:.3g}  bias={s.bias:+.3g}  R2={s.r2:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target", required=True, choices=available_targets())
    p.add_argument("--sprof-dir", type=Path, default=DEFAULT_SPROF_DIR)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    target = args.target
    unit = UNITS.get(target, "")

    artifact_path = args.model_dir / f"bgc_argo_{target}.pt"
    if not artifact_path.exists():
        print(f"ERROR: artifact not found at {artifact_path}")
        return 1

    print(f"Loading model: {artifact_path}")
    model, normalizer, meta = load_artifact(artifact_path)
    extra = meta["extra"]
    log_target = bool(extra.get("log_target", False))
    include_season = bool(extra.get("include_season", False))
    include_sigma = bool(extra.get("include_sigma", False))
    cv_r2 = extra.get("cv_r2_mean")
    cv_rmse = extra.get("cv_rmse_mean")
    print(f"  log_target={log_target}  season={include_season}  CV R2={cv_r2:.4f}")

    print(f"Loading {target} from BGC-Argo ...")
    df = load_bgc_argo(args.sprof_dir, target=target, box=KUROSHIO_BOX)
    feats = build_features(df, include_sigma_theta=include_sigma,
                           include_season=include_season)
    X = feats.to_numpy()
    y = df[target].to_numpy()
    lat = df["latitude"].to_numpy()
    lon = df["longitude"].to_numpy()
    print(f"  rows: {len(df):,}")

    print("Predicting ...")
    pred = predict_all(model, normalizer, X, log_target)
    s_all = compute_stats(y, pred)
    print(f"  Kuroshio-box in-sample: {stats_text(s_all, unit)}")

    title_box = (
        f"{target}: BGC-Argo model vs obs (Kuroshio box 120-180E,10-50N)\n"
        f"{stats_text(s_all, unit)}\n"
        f"Honest spatial CV (5-fold): RMSE={cv_rmse:.3g} {unit}  R2={cv_r2:.4f}"
    )
    scatter_hexbin(y, pred, unit=unit, title=title_box,
                   out_path=args.fig_dir / target / "scatter_box.png",
                   log_axes=log_target)

    mask = ((lat >= JAPAN_BOX["lat_min"]) & (lat <= JAPAN_BOX["lat_max"])
            & (lon >= JAPAN_BOX["lon_min"]) & (lon <= JAPAN_BOX["lon_max"]))
    n_jp = int(mask.sum())
    print(f"\nJapan box: {n_jp:,} rows")
    if n_jp < 10:
        print("  too few points; skipping")
        return 0
    s_jp = compute_stats(y[mask], pred[mask])
    print(f"  japan in-sample: {stats_text(s_jp, unit)}")
    title_jp = (
        f"{target}: BGC-Argo model vs obs (Japan box)\n"
        f"lat[{JAPAN_BOX['lat_min']:.0f},{JAPAN_BOX['lat_max']:.0f}] "
        f"lon[{JAPAN_BOX['lon_min']:.0f},{JAPAN_BOX['lon_max']:.0f}]\n"
        f"{stats_text(s_jp, unit)}"
    )
    scatter_hexbin(y[mask], pred[mask], unit=unit, title=title_jp,
                   out_path=args.fig_dir / target / "scatter_japan.png",
                   log_axes=log_target)

    print(f"\nSaved figures -> {args.fig_dir / target}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
