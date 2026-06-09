"""Residual diagnostics for the clean DOC model (base7, cruise 4057 excluded,
DOC in [30,150]). Uses out-of-fold spatial-CV predictions (honest), then shows
where base7 fails: residual vs depth and vs candidate water-mass predictors
(AOU, O2, NO3, sigma0). Residual correlations point to which feature would help.

Usage:
    uv run python scripts/analyze_doc_residuals.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from bio_params.cv import spatial_block_split
from bio_params.dataset import Normalizer, TabularDataset
from bio_params.features import build_features
from bio_params.model import MLP, MLPConfig
from bio_params.train import TrainConfig, train

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
OUT_DIR = ROOT / "figures" / "glodap_coverage"
EXTRA = {"AOU": "G2aou", "O2": "G2oxygen", "NO3": "G2nitrate", "sigma0": "G2sigma0"}


def main() -> int:
    np.random.seed(42); torch.manual_seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["G2cruise", "G2latitude", "G2longitude", "G2depth", "G2temperature",
            "G2salinity", "G2doc"] + list(EXTRA.values())
    df = pd.read_csv(CSV, usecols=cols).replace(-9999, np.nan)
    df = df.rename(columns={"G2latitude": "latitude", "G2longitude": "longitude",
                            "G2depth": "depth", "G2temperature": "temperature",
                            "G2salinity": "salinity", "G2doc": "DOC"})
    df = df.dropna(subset=["latitude", "longitude", "depth", "temperature", "salinity", "DOC"])
    df = df[(df.G2cruise != 4057) & (df.DOC >= 30) & (df.DOC <= 150)].reset_index(drop=True)
    print(f"clean DOC rows: {len(df):,}")

    X = build_features(df).to_numpy()
    y = df["DOC"].to_numpy()
    lat = df["latitude"].to_numpy(); lon = df["longitude"].to_numpy()

    # out-of-fold predictions via spatial block CV
    oof = np.full(len(y), np.nan)
    cfg = TrainConfig(epochs=200, batch_size=4096, early_stopping_patience=15, log_every=999)
    mcfg = MLPConfig(in_dim=X.shape[1], hidden=128, n_hidden_layers=3)
    for k, tr, va in spatial_block_split(lat, lon, block_deg=5.0, n_folds=5, seed=42):
        norm = Normalizer.fit(X[tr], y[tr])
        model = MLP(mcfg)
        train(model, TabularDataset(norm.transform_x(X[tr]), norm.transform_y(y[tr])),
              TabularDataset(norm.transform_x(X[va]), norm.transform_y(y[va])), cfg)
        model.eval().to("cpu")
        with torch.no_grad():
            p = model(torch.tensor(norm.transform_x(X[va]), dtype=torch.float32)).numpy()
        oof[va] = norm.inverse_transform_y(p)
        print(f"  fold {k} done ({len(va):,} val)")
    res = oof - y
    r2 = 1 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"OOF: R2={r2:.3f}  RMSE={np.sqrt(np.mean(res**2)):.2f}  MAE={np.mean(np.abs(res)):.2f}  "
          f"bias={res.mean():+.2f}")

    # residual vs candidate predictors (which feature would help?)
    print("\nresidual (pred-obs) correlation with candidate predictors:")
    for name, col in EXTRA.items():
        m = np.isfinite(df[col].to_numpy())
        if m.sum() > 100:
            r = pearsonr(res[m], df[col].to_numpy()[m])[0]
            rho = spearmanr(res[m], df[col].to_numpy()[m])[0]
            print(f"  resid vs {name:6s}: Pearson={r:+.3f}  Spearman={rho:+.3f}  (n={int(m.sum()):,})")
    for nm, v in [("depth", df.depth.to_numpy()), ("DOC_obs", y)]:
        print(f"  resid vs {nm:6s}: Pearson={pearsonr(res, v)[0]:+.3f}")

    # residual by depth band
    print("\nresidual by depth band:")
    bands = [(0, 50), (50, 200), (200, 1000), (1000, 8000)]
    for lo, hi in bands:
        b = (df.depth >= lo) & (df.depth < hi)
        if b.sum():
            print(f"  {lo:5d}-{hi:5d} m: n={int(b.sum()):6d}  bias={res[b].mean():+.2f}  "
                  f"RMSE={np.sqrt(np.mean(res[b]**2)):.2f}  obs_med={np.median(y[b]):.0f}")

    # --- figure 1: residual scatter panels ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax = axes[0, 0]
    sc = ax.scatter(y, oof, s=6, alpha=0.3, c=df.depth, cmap="viridis_r")
    lo, hi = 30, 150; ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("DOC obs"); ax.set_ylabel("DOC pred (OOF)"); ax.set_title(f"obs vs pred (R2={r2:.3f})")
    plt.colorbar(sc, ax=ax, pad=0.02).set_label("depth (m)")
    panels = [("depth", df.depth.to_numpy(), "depth (m)"), ("AOU", df.G2aou.to_numpy(), "AOU (umol/kg)"),
              ("NO3", df.G2nitrate.to_numpy(), "NO3 (umol/kg)"), ("O2", df.G2oxygen.to_numpy(), "O2 (umol/kg)"),
              ("sigma0", df.G2sigma0.to_numpy(), "sigma0")]
    for ax, (nm, v, xl) in zip(axes.ravel()[1:], panels):
        m = np.isfinite(v)
        ax.scatter(v[m], res[m], s=6, alpha=0.25, c=df.depth.to_numpy()[m], cmap="viridis_r")
        ax.axhline(0, color="k", lw=0.8)
        if nm != "depth":
            r = pearsonr(res[m], v[m])[0]
            ax.text(0.04, 0.96, f"r={r:+.3f}", transform=ax.transAxes, va="top",
                    bbox=dict(boxstyle="round", fc="white", alpha=0.85))
        ax.set_xlabel(xl); ax.set_ylabel("residual (pred-obs)"); ax.set_title(f"residual vs {nm}")
    fig.suptitle("Clean DOC (base7) out-of-fold residuals", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out1 = OUT_DIR / "doc_clean_residuals.png"
    fig.savefig(out1, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"\nsaved {out1}")

    # --- figure 2: residual map ---
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        fig = plt.figure(figsize=(13, 6.5))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=150))
        ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=0); ax.coastlines(lw=0.4, zorder=3)
        lonp = ((lon + 180) % 360) - 180
        dl = float(np.nanpercentile(np.abs(res), 98))
        sc = ax.scatter(lonp, lat, s=5, c=res, cmap="RdBu_r", vmin=-dl, vmax=dl,
                        alpha=0.6, transform=ccrs.PlateCarree(), zorder=2)
        ax.set_global(); plt.colorbar(sc, ax=ax, pad=0.02, shrink=0.7).set_label("residual (pred-obs) umol/kg")
        ax.set_title("Clean DOC base7 residuals (red=over, blue=under)", fontsize=12)
        out2 = OUT_DIR / "doc_clean_residual_map.png"
        fig.savefig(out2, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"saved {out2}")
    except Exception as e:  # noqa: BLE001
        print(f"(skip residual map: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
