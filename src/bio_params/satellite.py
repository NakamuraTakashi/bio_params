"""Open the CMEMS GlobColour monthly surface Chl-a field, preferring a local copy.

If NetCDF files exist under data/satellite/raw/ (downloaded by
scripts/download_satellite_chla.py) they are opened with xarray and NO server
access happens; otherwise this falls back to copernicusmarine.open_dataset
(lazy, remote, needs `copernicusmarine login`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

SAT_DATASET = "cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M"
SAT_CHL_VAR = "CHL"
_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SAT_DIR = _ROOT / "data" / "satellite" / "raw"
_CLIM_DIR = _ROOT / "data" / "satellite" / "climatology"
CLIM_PATH = _CLIM_DIR / "globcolour_chl_monthly_clim.nc"            # median, gappy
CLIM_FILLED = _CLIM_DIR / "globcolour_chl_monthly_clim_filled.nc"   # gap-free (preferred)


def open_chla(dataset_id: str = SAT_DATASET, local_dir: Path = LOCAL_SAT_DIR):
    """Return (Dataset, source) where source is 'local' or 'server'.

    Local NetCDFs (e.g. per-year files) are combined along time. The returned
    dataset always has the variable `CHL` and coords latitude/longitude/time,
    so callers can use it identically whether local or remote.
    """
    ncs = sorted(Path(local_dir).glob("*.nc")) if Path(local_dir).exists() else []
    if ncs:
        if len(ncs) > 1:
            ds = xr.open_mfdataset(ncs, combine="by_coords")
        else:
            ds = xr.open_dataset(ncs[0])
        return ds, "local"
    import copernicusmarine as cm
    return cm.open_dataset(dataset_id=dataset_id), "server"


def chla_month_field(month_str: str, dataset_id: str = SAT_DATASET):
    """Global surface CHL field for `month_str` ("YYYY-MM").

    Uses the actual satellite month if it is within the archive's time range,
    otherwise falls back to the monthly climatology (same calendar month) so
    ROMS runs outside the satellite period still get a field. Returns
    (lat_axis, lon_axis, arr2d, source) with source in
    {'local','server','climatology'}.
    """
    ds, src = open_chla(dataset_id=dataset_id)
    req = pd.Period(month_str, "M")
    tmin = pd.Period(pd.Timestamp(np.asarray(ds["time"].values).min()), "M")
    tmax = pd.Period(pd.Timestamp(np.asarray(ds["time"].values).max()), "M")
    if tmin <= req <= tmax:
        arr = np.asarray(ds["CHL"].sel(time=f"{month_str}-01", method="nearest").load().values)
        lat = ds["latitude"].values; lon = ds["longitude"].values
        ds.close()
        return lat, lon, arr, src
    ds.close()
    clim_path = CLIM_FILLED if CLIM_FILLED.exists() else CLIM_PATH
    if not clim_path.exists():
        raise FileNotFoundError(
            f"{month_str} is outside the satellite archive ({tmin}..{tmax}) and no "
            f"climatology at {CLIM_PATH}; run scripts/build_satellite_climatology.py")
    cds = xr.open_dataset(clim_path)
    mon = int(month_str.split("-")[1])
    arr = np.asarray(cds["CHL"].sel(month=mon).values)
    lat = cds["latitude"].values; lon = cds["longitude"].values
    cds.close()
    return lat, lon, arr, "climatology"
