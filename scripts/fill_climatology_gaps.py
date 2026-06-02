"""Spatially fill the systematic gaps in the monthly Chl-a climatology.

The monthly product is not gap-filled, so the climatology (median over years)
still has NaN where a pixel is never observed in that calendar month (polar
winter, sea ice, persistent cloud) -- about 21% of ocean. For ROMS boundary
conditions a complete field is needed, so each month's NaN pixels are filled
with their nearest valid pixel (Euclidean nearest, scipy EDT). Land is filled
too but is masked by the ROMS grid at use time.

Input : data/satellite/climatology/globcolour_chl_monthly_clim.nc (median, gappy)
Output: data/satellite/climatology/globcolour_chl_monthly_clim_filled.nc (gap-free)

Note: high-latitude winter Chl is near zero; nearest-fill there is an
approximation (may pull a higher lower-latitude value) -- acceptable for a
fallback, but flagged.

Usage:
    uv run python scripts/fill_climatology_gaps.py
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import ndimage

from bio_params.satellite import CLIM_FILLED, CLIM_PATH


def main() -> int:
    if not CLIM_PATH.exists():
        print(f"ERROR: {CLIM_PATH} not found; run build_satellite_climatology.py")
        return 1
    ds = xr.open_dataset(CLIM_PATH)
    chl = np.asarray(ds["CHL"].values)            # (12, nlat, nlon)
    filled = chl.copy()
    for m in range(chl.shape[0]):
        nan = ~np.isfinite(chl[m])
        if nan.any() and (~nan).any():
            idx = ndimage.distance_transform_edt(
                nan, return_distances=False, return_indices=True)
            filled[m] = chl[m][tuple(idx)]
        print(f"  month {m + 1:2d}: filled {int(nan.sum()):,} px "
              f"(finite {np.isfinite(chl[m]).mean():.3f} -> {np.isfinite(filled[m]).mean():.3f})",
              flush=True)
    out = xr.Dataset(
        {"CHL": (("month", "latitude", "longitude"), filled.astype("float32"))},
        coords={"month": ds["month"].values, "latitude": ds["latitude"].values,
                "longitude": ds["longitude"].values},
        attrs={**ds.attrs, "postprocess": "nearest-valid spatial gap fill (scipy EDT)"},
    )
    ds.close()
    enc = {"CHL": {"zlib": True, "complevel": 4}}
    out.to_netcdf(CLIM_FILLED, encoding=enc)
    print(f"saved {CLIM_FILLED}  ({CLIM_FILLED.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
