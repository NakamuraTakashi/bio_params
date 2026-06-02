"""Build a monthly Chl-a climatology (12 calendar-month fields) from the local
GlobColour archive, as a fallback for ROMS runs outside the satellite period.

Per pixel and calendar month, the climatology is the median across all years
(robust to blooms/outliers and to cloud NaNs). Output:
data/satellite/climatology/globcolour_chl_monthly_clim.nc with dims
(month=1..12, latitude, longitude), variable CHL.

Usage (after scripts/download_satellite_chla.py):
    uv run python scripts/build_satellite_climatology.py
"""
from __future__ import annotations

import argparse

import numpy as np
import xarray as xr

from bio_params.satellite import CLIM_PATH, open_chla


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stat", choices=["median", "mean"], default="median")
    args = p.parse_args()

    ds, src = open_chla()
    if src != "local":
        print("ERROR: no local archive found (data/satellite/raw/). Download first.")
        return 1
    chl = ds["CHL"]
    months = chl["time"].dt.month.values
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    print(f"local archive: {chl.sizes['time']} months, grid {len(lat)}x{len(lon)}")

    clim = np.full((12, len(lat), len(lon)), np.nan, dtype="float32")
    for m in range(1, 13):
        idx = np.where(months == m)[0]
        arr = np.asarray(chl.isel(time=idx).load().values)  # (k, nlat, nlon)
        with np.errstate(all="ignore"):
            agg = (np.nanmedian(arr, axis=0) if args.stat == "median"
                   else np.nanmean(arr, axis=0))
        clim[m - 1] = agg.astype("float32")
        print(f"  month {m:2d}: {len(idx)} years -> {args.stat}  "
              f"(finite {np.isfinite(agg).mean():.2f}, median {np.nanmedian(agg):.3g})", flush=True)
        del arr
    ds.close()

    out = xr.Dataset(
        {"CHL": (("month", "latitude", "longitude"), clim)},
        coords={"month": np.arange(1, 13, dtype="int16"), "latitude": lat, "longitude": lon},
        attrs={"description": f"GlobColour monthly Chl-a climatology ({args.stat} over years)",
               "source": "cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M"},
    )
    CLIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    enc = {"CHL": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}}
    out.to_netcdf(CLIM_PATH, encoding=enc)
    print(f"saved {CLIM_PATH}  ({CLIM_PATH.stat().st_size/1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
