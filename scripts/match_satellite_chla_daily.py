"""Match each combined (GLODAP+BGC-Argo) Chl-a profile to the DAILY gapfree
satellite surface Chl-a at the profile's exact date and position.

Like match_satellite_chla.py but uses the local daily archive
(bio_params.satellite.DAILY_SAT_ROOT, coords lat/lon, var CHL, one file per day)
instead of the monthly product, so the surface-Chl feature is matched on the
actual observation day (consistent with ROMS inference --satellite daily).

Output: data/bgc_argo/processed/satchl_matchup_daily_combined.parquet
  columns: latitude, longitude, time, surface_chla  (one row per matched profile)
Resumable: per-day checkpoints under data/bgc_argo/processed/_satchl_days/.

Usage:
    uv run python scripts/match_satellite_chla_daily.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from bio_params.loaders.chla_no3 import load_chla_no3
from bio_params.satellite import DAILY_SAT_ROOT

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
DEFAULT_SPROF = ROOT / "data" / "bgc_argo" / "raw" / "floats"
PROC = ROOT / "data" / "bgc_argo" / "processed"
OUT = PROC / "satchl_matchup_daily_combined.parquet"


def _nearest_index(coord, axis):
    origin = float(axis[0]); step = float(axis[1] - axis[0])
    return np.clip(np.rint((coord - origin) / step).astype(np.int64), 0, len(axis) - 1)


def _day_file(d: pd.Timestamp, root: Path):
    fdir = Path(root) / f"{d.year:04d}" / f"{d.month:02d}"
    m = sorted(fdir.glob(f"{d.year:04d}{d.month:02d}{d.day:02d}_*_P1D.nc"))
    return m[0] if m else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=DAILY_SAT_ROOT)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()

    df = load_chla_no3("combined", glodap_csv=DEFAULT_CSV, sprof_dir=DEFAULT_SPROF)
    prof = df[["latitude", "longitude", "time"]].drop_duplicates().reset_index(drop=True)
    prof["time"] = pd.to_datetime(prof["time"])
    prof = prof[prof["time"].notna()].reset_index(drop=True)
    prof["day"] = prof["time"].dt.normalize()
    days = sorted(prof["day"].unique())
    print(f"{len(prof):,} unique profiles, {len(days):,} unique days", flush=True)

    ckpt_dir = PROC / "_satchl_days"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    lat_axis = lon_axis = None
    for k, day in enumerate(days, 1):
        d = pd.Timestamp(day)
        ckpt = ckpt_dir / f"{d:%Y%m%d}.parquet"
        if ckpt.exists():
            continue
        grp = prof[prof["day"] == day]
        f = _day_file(d, args.root)
        if f is None:
            grp_out = grp[["latitude", "longitude", "time"]].copy()
            grp_out["surface_chla"] = np.nan
            grp_out.to_parquet(ckpt, index=False)
            continue
        ds = xr.open_dataset(f)
        if lat_axis is None:
            lat_axis = ds["lat"].values; lon_axis = ds["lon"].values
        chl = ds["CHL"]
        arr = np.asarray(chl.isel(time=0).values if "time" in chl.dims else chl.values)
        ds.close()
        ilat = _nearest_index(grp["latitude"].to_numpy(), lat_axis)
        lon_w = ((grp["longitude"].to_numpy() + 180) % 360) - 180   # -> [-180,180]
        ilon = _nearest_index(lon_w, lon_axis)
        grp_out = grp[["latitude", "longitude", "time"]].copy()
        grp_out["surface_chla"] = arr[ilat, ilon]
        grp_out.to_parquet(ckpt, index=False)
        if k % 200 == 0 or k == len(days):
            print(f"  [{k}/{len(days)}] {d:%Y-%m-%d}: {len(grp)} profiles", flush=True)

    parts = [pd.read_parquet(p) for p in sorted(ckpt_dir.glob("*.parquet"))]
    full = pd.concat(parts, ignore_index=True)
    before = len(full)
    full = full[np.isfinite(full["surface_chla"])].reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(args.out, index=False)
    print(f"Saved {args.out}: {len(full):,} matched profiles "
          f"({before - len(full):,} dropped for masked/missing pixel)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
