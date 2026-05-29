"""Plot model predictions vs observations for a saved per-target artifact.

Produces two hexbin scatter plots per target:
  figures/<target>/scatter_global.png   - all rows in GLODAP
  figures/<target>/scatter_japan.png    - rows inside the Japan box

NOTE: predictions are made by the saved final model, which was trained on
all data with only a small random early-stopping holdout. The scatter is
therefore an in-sample fit indicator, not an honest generalization score.
For honest performance, refer to the CV stats embedded in the artifact
(also annotated on the plot title).

Usage:
    uv run python scripts/plot_predictions.py --target TA
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch

from bio_params.features import build_features
from bio_params.loaders.glodap import available_targets, load_glodap
from bio_params.persist import load_artifact

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "pretrained"
DEFAULT_FIG_DIR = PROJECT_ROOT / "figures"

# Bounding box for Japanese waters (Okinawa to northern Hokkaido plus the
# East China Sea, Sea of Japan, and the Pacific east of Honshu).
JAPAN_BOX = dict(lat_min=20.0, lat_max=50.0, lon_min=120.0, lon_max=155.0)


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


def predict_all(model, normalizer, X: np.ndarray, batch_size: int = 100_000) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    X_n = normalizer.transform_x(X).astype(np.float32)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X_n), batch_size):
            xb = torch.from_numpy(X_n[i:i + batch_size]).to(device)
            preds.append(model(xb).cpu().numpy())
    pred_n = np.concatenate(preds)
    return normalizer.inverse_transform_y(pred_n)


def scatter_hexbin(
    obs: np.ndarray,
    pred: np.ndarray,
    *,
    unit: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    lo = float(min(obs.min(), pred.min()))
    hi = float(max(obs.max(), pred.max()))
    pad = 0.02 * (hi - lo)
    lo -= pad
    hi += pad

    use_scatter = len(obs) < 3000
    if use_scatter:
        ax.scatter(obs, pred, s=8, alpha=0.4, edgecolor="none", color="steelblue")
    else:
        hb = ax.hexbin(
            obs, pred,
            gridsize=80, cmap="viridis", bins="log", mincnt=1,
            extent=(lo, hi, lo, hi),
        )
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("log10(count)")

    ax.plot([lo, hi], [lo, hi], "r--", lw=1.0, alpha=0.7, label="1:1")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Observation ({unit})")
    ax.set_ylabel(f"Model prediction ({unit})")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def japan_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return (
        (lat >= JAPAN_BOX["lat_min"]) & (lat <= JAPAN_BOX["lat_max"]) &
        (lon >= JAPAN_BOX["lon_min"]) & (lon <= JAPAN_BOX["lon_max"])
    )


def stats_text(s: FitStats, unit: str) -> str:
    return (
        f"n={s.n:,}  RMSE={s.rmse:.3g} {unit}  "
        f"MAE={s.mae:.3g}  bias={s.bias:+.3g}  R²={s.r2:.4f}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target", required=True, choices=available_targets())
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    p.add_argument("--unit", default="umol/kg",
                   help="Unit label for axes (e.g. umol/kg, umol/L, mg/L)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    target = args.target

    artifact_path = args.model_dir / f"glodap_{target}.pt"
    if not artifact_path.exists():
        print(f"ERROR: artifact not found at {artifact_path}")
        print(f"Run: uv run python scripts/train_target.py --target {target}")
        return 1

    print(f"Loading model: {artifact_path}")
    model, normalizer, meta = load_artifact(artifact_path)
    cv_rmse = meta["extra"].get("cv_rmse_mean")
    cv_r2 = meta["extra"].get("cv_r2_mean")
    include_sigma = bool(meta["extra"].get("include_sigma", False))
    print(f"  CV (honest reference): RMSE={cv_rmse:.3g}  R²={cv_r2:.4f}")

    print(f"Loading {target} from CSV ...")
    df = load_glodap(args.csv, target=target)
    feats = build_features(df, include_sigma_theta=include_sigma)
    X = feats.to_numpy()
    y = df[target].to_numpy()
    lat = df["latitude"].to_numpy()
    lon = df["longitude"].to_numpy()
    print(f"  rows: {len(df):,}")

    print("Predicting ...")
    pred = predict_all(model, normalizer, X)
    s_all = compute_stats(y, pred)
    print(f"  global in-sample: {stats_text(s_all, args.unit)}")

    title_global = (
        f"{target}: model vs observation (global, in-sample)\n"
        f"{stats_text(s_all, args.unit)}\n"
        f"Honest spatial CV (5-fold): RMSE={cv_rmse:.3g} {args.unit}  R²={cv_r2:.4f}"
    )
    scatter_hexbin(
        y, pred,
        unit=args.unit, title=title_global,
        out_path=args.fig_dir / target / "scatter_global.png",
    )

    mask = japan_mask(lat, lon)
    n_jp = int(mask.sum())
    print(f"\nJapan box ({JAPAN_BOX}): {n_jp:,} rows")
    if n_jp < 10:
        print("  too few points for a meaningful plot; skipping")
        return 0

    s_jp = compute_stats(y[mask], pred[mask])
    print(f"  japan in-sample:  {stats_text(s_jp, args.unit)}")
    title_jp = (
        f"{target}: model vs observation (Japan box, in-sample)\n"
        f"lat [{JAPAN_BOX['lat_min']:.0f}, {JAPAN_BOX['lat_max']:.0f}], "
        f"lon [{JAPAN_BOX['lon_min']:.0f}, {JAPAN_BOX['lon_max']:.0f}]\n"
        f"{stats_text(s_jp, args.unit)}"
    )
    scatter_hexbin(
        y[mask], pred[mask],
        unit=args.unit, title=title_jp,
        out_path=args.fig_dir / target / "scatter_japan.png",
    )

    print(f"\nSaved figures -> {args.fig_dir / target}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
