"""Scatter NO3 vs PO4 from GLODAP with the Redfield N:P = 16:1 line.

Extracts samples where BOTH nitrate and phosphate are good (flag==2) from the
same GLODAP rows, then plots NO3 against PO4 for the global ocean and for the
Japan box, overlaying the Redfield ratio line (N:P = 16:1) and the fitted
slope through the origin.

Usage:
    uv run python scripts/plot_no3_po4_redfield.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
OUT_DIR = PROJECT_ROOT / "figures" / "redfield"

MISSING_SENTINEL = -9999
REDFIELD_NP = 16.0  # canonical N:P molar ratio

JAPAN_BOX = dict(lat_min=20.0, lat_max=50.0, lon_min=120.0, lon_max=155.0)


def load_no3_po4(csv_path: Path) -> pd.DataFrame:
    """Rows where both nitrate and phosphate are flag==2, with coordinates."""
    cols = [
        "G2latitude", "G2longitude", "G2depth",
        "G2nitrate", "G2nitratef",
        "G2phosphate", "G2phosphatef",
    ]
    df = pd.read_csv(csv_path, usecols=cols, low_memory=False)
    df = df.replace(MISSING_SENTINEL, np.nan)
    good = (df["G2nitratef"] == 2) & (df["G2phosphatef"] == 2)
    df = df[good].dropna(subset=["G2nitrate", "G2phosphate",
                                 "G2latitude", "G2longitude"])
    return df.rename(columns={
        "G2latitude": "latitude", "G2longitude": "longitude",
        "G2depth": "depth", "G2nitrate": "NO3", "G2phosphate": "PO4",
    }).reset_index(drop=True)


def fit_slope_through_origin(po4: np.ndarray, no3: np.ndarray) -> float:
    """Least-squares slope m of NO3 = m * PO4 (forced through origin)."""
    return float((po4 @ no3) / (po4 @ po4))


def fit_ols(po4: np.ndarray, no3: np.ndarray) -> tuple[float, float, float]:
    """Ordinary least squares NO3 = slope * PO4 + intercept.

    Returns (slope, intercept, r2).
    """
    slope, intercept = np.polyfit(po4, no3, 1)
    pred = slope * po4 + intercept
    ss_res = float(((no3 - pred) ** 2).sum())
    ss_tot = float(((no3 - no3.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def scatter(
    po4: np.ndarray,
    no3: np.ndarray,
    *,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 6.6))

    hi_p = float(np.percentile(po4, 99.9))
    hi_n = float(np.percentile(no3, 99.9))
    hi_p = max(hi_p, 1e-6)

    if len(po4) < 3000:
        ax.scatter(po4, no3, s=8, alpha=0.4, edgecolor="none", color="steelblue")
    else:
        hb = ax.hexbin(po4, no3, gridsize=90, cmap="viridis", bins="log",
                       mincnt=1, extent=(0, hi_p, 0, hi_n))
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("log10(count)")

    # Redfield 16:1 line, the slope-through-origin fit, and the OLS fit
    # (intercept allowed).
    xs = np.array([0.0, hi_p])
    ax.plot(xs, REDFIELD_NP * xs, "r-", lw=1.8,
            label=f"Redfield N:P = {REDFIELD_NP:.0f}:1")
    slope0 = fit_slope_through_origin(po4, no3)
    ax.plot(xs, slope0 * xs, "k--", lw=1.5,
            label=f"through origin: NO3 = {slope0:.2f} PO4")
    slope, intercept, r2 = fit_ols(po4, no3)
    sign = "+" if intercept >= 0 else "-"
    ax.plot(xs, slope * xs + intercept, color="darkorange", lw=1.5,
            label=(f"OLS: NO3 = {slope:.2f} PO4 {sign} {abs(intercept):.2f}"
                   f"  (R²={r2:.3f})"))

    ax.set_xlim(0, hi_p)
    ax.set_ylim(0, hi_n)
    ax.set_xlabel("PO4 (umol/kg)")
    ax.set_ylabel("NO3 (umol/kg)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def japan_mask(df: pd.DataFrame) -> np.ndarray:
    return (
        (df.latitude >= JAPAN_BOX["lat_min"]) & (df.latitude <= JAPAN_BOX["lat_max"])
        & (df.longitude >= JAPAN_BOX["lon_min"]) & (df.longitude <= JAPAN_BOX["lon_max"])
    ).to_numpy()


def main() -> int:
    print(f"Loading NO3 & PO4 (both flag==2) from {DEFAULT_CSV} ...")
    df = load_no3_po4(DEFAULT_CSV)
    print(f"  paired samples: {len(df):,}")

    po4_g, no3_g = df.PO4.to_numpy(), df.NO3.to_numpy()
    slope_g = fit_slope_through_origin(po4_g, no3_g)
    sl_g, ic_g, r2_g = fit_ols(po4_g, no3_g)
    print(f"  global through-origin slope: {slope_g:.2f} (Redfield = {REDFIELD_NP:.0f})")
    print(f"  global OLS: NO3 = {sl_g:.3f} PO4 + {ic_g:.3f}  (R²={r2_g:.4f})")
    scatter(
        po4_g, no3_g,
        title=(f"GLODAP NO3 vs PO4 (global)\n"
               f"n={len(df):,}  through-origin N:P={slope_g:.1f}  (Redfield 16:1)"),
        out_path=OUT_DIR / "no3_po4_global.png",
    )

    m = japan_mask(df)
    dj = df[m]
    print(f"\nJapan box: {len(dj):,} paired samples")
    if len(dj) >= 10:
        po4_j, no3_j = dj.PO4.to_numpy(), dj.NO3.to_numpy()
        slope_j = fit_slope_through_origin(po4_j, no3_j)
        sl_j, ic_j, r2_j = fit_ols(po4_j, no3_j)
        print(f"  japan through-origin slope: {slope_j:.2f}")
        print(f"  japan OLS: NO3 = {sl_j:.3f} PO4 + {ic_j:.3f}  (R²={r2_j:.4f})")
        scatter(
            po4_j, no3_j,
            title=(f"GLODAP NO3 vs PO4 (Japan box)\n"
                   f"lat[{JAPAN_BOX['lat_min']:.0f},{JAPAN_BOX['lat_max']:.0f}] "
                   f"lon[{JAPAN_BOX['lon_min']:.0f},{JAPAN_BOX['lon_max']:.0f}]  "
                   f"n={len(dj):,}  through-origin N:P={slope_j:.1f}"),
            out_path=OUT_DIR / "no3_po4_japan.png",
        )

    print(f"\nSaved figures -> {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
