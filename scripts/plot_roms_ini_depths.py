"""Plot ROMS initial-condition tracer fields on fixed depth levels.

Reads a ROMS ini file and its grid, reconstructs the depth of every s_rho
level using the S-coordinate transform (Vtransform=2, Vstretching=4),
linearly interpolates each 3D scalar tracer to a set of fixed depths, and
draws a contour/pcolormesh map per (variable, depth).

Vertical coordinate (Vtransform = 2):
    S(k) = (hc * s_rho(k) + h * Cs_r(k)) / (hc + h)
    z(k) = zeta + (zeta + h) * S(k)            # z negative downward

Usage:
    uv run python scripts/plot_roms_ini_depths.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

INI = Path("/mnt/d/COAWST_DATA/FORP_Kuroshio/Ini/Kuro_Ini_FORP_Nz30_20060102.00.nc")
GRID = Path("/mnt/d/COAWST_DATA/FORP_Kuroshio/Grid/forp-kuroshio_grd_v0.0.nc")
OUT_DIR = Path(__file__).resolve().parent.parent / "figures" / "roms_ini"

# Fixed depths (m, positive down). 0.0 means the surface (top s_rho level).
TARGET_DEPTHS = [0.0, 200.0, 500.0, 1000.0, 3000.0]

# 3D scalar tracers to map, with display metadata.
VARIABLES = {
    "temp": dict(long="Potential temperature", unit="degC", cmap="turbo"),
    "salt": dict(long="Salinity", unit="PSU", cmap="viridis"),
}


def compute_z_rho(h, zeta, s_rho, Cs_r, hc):
    """Depth of each rho level, shape (N, eta, xi); negative downward."""
    h = h[None, :, :]            # (1, J, I)
    zeta = zeta[None, :, :]      # (1, J, I)
    s = s_rho[:, None, None]     # (N, 1, 1)
    C = Cs_r[:, None, None]      # (N, 1, 1)
    S = (hc * s + h * C) / (hc + h)
    return zeta + (zeta + h) * S


def interp_to_depth(data, z, target_z):
    """Linear interp of `data` (N,J,I) along z (N,J,I) to scalar `target_z`.

    z must increase with k (k=0 deepest). Returns (J,I) with NaN where the
    target depth is outside the local water column.
    """
    n = z.shape[0]
    # number of levels at or below the target depth, per column -> in [0, N]
    idx = np.sum(z <= target_z, axis=0)
    idx_lo = np.clip(idx - 1, 0, n - 1)
    idx_hi = np.clip(idx, 0, n - 1)

    def gather(a, k):
        return np.take_along_axis(a, k[None, :, :], axis=0)[0]

    z_lo, z_hi = gather(z, idx_lo), gather(z, idx_hi)
    d_lo, d_hi = gather(data, idx_lo), gather(data, idx_hi)
    denom = z_hi - z_lo
    w = np.where(denom != 0, (target_z - z_lo) / denom, 0.0)
    out = d_lo + w * (d_hi - d_lo)
    valid = (idx >= 1) & (idx <= n - 1)   # target strictly inside the column
    return np.where(valid, out, np.nan)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(INI, decode_times=False)
    g = xr.open_dataset(GRID, decode_times=False)

    h = g["h"].values.astype(np.float64)
    mask = g["mask_rho"].values
    lon = g["lon_rho"].values
    lat = g["lat_rho"].values
    zeta = ds["zeta"].isel(ocean_time=0).values.astype(np.float64)
    s_rho = ds["s_rho"].values.astype(np.float64)
    Cs_r = ds["Cs_r"].values.astype(np.float64)
    hc = float(ds["hc"].values)

    # Guard against bad/negative bathymetry samples on land.
    h = np.where(h <= 0, np.nan, h)
    z_rho = compute_z_rho(h, zeta, s_rho, Cs_r, hc)  # (N, J, I)
    land = mask < 0.5

    for var, meta in VARIABLES.items():
        field = ds[var].isel(ocean_time=0).values.astype(np.float64)  # (N,J,I)

        fig, axes = plt.subplots(
            1, len(TARGET_DEPTHS),
            figsize=(4.6 * len(TARGET_DEPTHS), 5.2),
            constrained_layout=True,
        )
        for ax, depth in zip(axes, TARGET_DEPTHS):
            if depth == 0.0:
                layer = field[-1].copy()          # top s_rho level = surface
                label = "surface"
            else:
                layer = interp_to_depth(field, z_rho, -depth)
                label = f"{depth:.0f} m"
            layer[land] = np.nan

            finite = layer[np.isfinite(layer)]
            if finite.size:
                vmin, vmax = np.percentile(finite, [2, 98])
            else:
                vmin, vmax = None, None

            pcm = ax.pcolormesh(
                lon, lat, np.ma.masked_invalid(layer),
                cmap=meta["cmap"], vmin=vmin, vmax=vmax, shading="auto",
            )
            ax.set_facecolor("0.8")  # land / no-data grey
            ax.set_title(f"{label}", fontsize=11)
            ax.set_xlabel("Longitude")
            if ax is axes[0]:
                ax.set_ylabel("Latitude")
            ax.set_aspect("equal")
            cb = fig.colorbar(pcm, ax=ax, orientation="horizontal",
                              pad=0.08, fraction=0.05)
            cb.set_label(f"{var} ({meta['unit']})", fontsize=9)

        fig.suptitle(
            f"{meta['long']} ({var}) — {INI.name}\n"
            f"ROMS Vtransform=2 Vstretching=4 hc={hc:.0f}  "
            f"(colorbar = 2–98th percentile per panel)",
            fontsize=12,
        )
        out = OUT_DIR / f"ini_{var}_depths.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")

    ds.close()
    g.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
