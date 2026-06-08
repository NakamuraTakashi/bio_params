"""Quick access check / surface map for FORA-JPN60 (JMA/JAMSTEC Japan coastal
ocean reanalysis) over OPeNDAP.

FORA-JPN60 daily-mean 3D files (per variable, per day):
  T: https://www.jamstec.go.jp/jagdas/dodsC/fora/JPN/Daily-mean/Basic-3D/<YYYY>/nc_t.<YYYYMMDD>
  S: .../nc_s.<YYYYMMDD>
The 3D var is (time, depth, lat, lon); surface = depth index 0 (~1 m). Only the
requested (strided) surface slice is streamed via OPeNDAP, not the whole file.

Usage:
    uv run python scripts/plot_fora_surface.py --date 2020-06-01
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "figures" / "fora"
BASE = "https://www.jamstec.go.jp/jagdas/dodsC/fora/JPN/Daily-mean/Basic-3D"


def fora_url(kind: str, date) -> str:
    import pandas as pd
    d = pd.Timestamp(date)
    return f"{BASE}/{d.year}/nc_{kind}.{d:%Y%m%d}"


def surface_slice(url: str, stride: int):
    ds = xr.open_dataset(url)
    var = list(ds.data_vars)[0]                       # single 3D variable
    da = ds[var].isel(time=0, depth=0,
                      lat=slice(None, None, stride), lon=slice(None, None, stride))
    arr = np.asarray(da.load().values)
    lon = ds["lon"].values[::stride]; lat = ds["lat"].values[::stride]
    units = ds[var].attrs.get("units", "")
    ds.close()
    return var, units, lon, lat, arr


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--date", default="2020-06-01")
    p.add_argument("--stride", type=int, default=2, help="lat/lon subsample for plotting")
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    for ax, (kind, cmap, label) in zip(axes, [("t", "turbo", "Sea surface temp"),
                                              ("s", "viridis", "Sea surface salinity")]):
        url = fora_url(kind, args.date)
        print(f"opening {url} ...", flush=True)
        var, units, lon, lat, arr = surface_slice(url, args.stride)
        vmin, vmax = np.nanpercentile(arr, [1, 99])
        pcm = ax.pcolormesh(lon, lat, arr, cmap=cmap, vmin=vmin, vmax=vmax,
                            shading="auto", transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=2)
        ax.coastlines(lw=0.4, zorder=3)
        ax.set_title(f"{label}  [{var}, {units}]  {args.date}", fontsize=10)
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
        plt.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.05, fraction=0.05).set_label(
            f"{var} ({units})")
        print(f"  {var}: range {np.nanmin(arr):.2f}..{np.nanmax(arr):.2f} {units}, "
              f"grid {arr.shape}", flush=True)
    fig.suptitle(f"FORA-JPN60 surface (depth~1 m), {args.date}  (stride={args.stride})",
                 fontsize=13)
    fig.tight_layout()
    out = OUT_DIR / f"fora_surface_{args.date.replace('-', '')}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
