"""Extract FORA-JPN60 vertical T/S profiles at the GLODAP "full-set" stations
(the 405 unique positions inside the FORA domain & period 1982-2020 where TA,
DIC, O2, NO3, PO4, SiO4 and Chl-a are all present) at the nearest FORA grid cell
and the same (nearest) date, and cache them locally for reuse.

Outputs (data/glodap/processed/):
  fora_glodap_profiles.nc   FORA T/S (station, level) + station meta + depth axis
  fora_glodap_samples.parquet  GLODAP per-depth obs (station_id, depth, T,S, 7 vars)

The expensive OPeNDAP pass (one open per date) runs once here; scatter plots read
the cache. Usage:
    uv run python scripts/extract_fora_glodap_matchup.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from predict_fora_chla import fora_url

CSV = ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
OUT = ROOT / "data" / "glodap" / "processed"
BOX = dict(latmin=19.96, latmax=52.02, lonmin=116.94, lonmax=160.03)
PERIOD = (pd.Timestamp("1982-01-01"), pd.Timestamp("2020-12-31"))
VALS = {"TA": "G2talk", "DIC": "G2tco2", "O2": "G2oxygen", "NO3": "G2nitrate",
        "PO4": "G2phosphate", "SiO4": "G2silicate", "Chla": "G2chla"}


def load_fullset():
    cols = (["G2cruise", "G2station", "G2latitude", "G2longitude", "G2depth",
             "G2year", "G2month", "G2day", "G2temperature", "G2salinity"]
            + list(VALS.values()))
    df = pd.read_csv(CSV, usecols=cols).replace(-9999, np.nan)
    df = df.dropna(subset=["G2latitude", "G2longitude", "G2depth",
                           "G2temperature", "G2salinity"])
    df = df[(df.G2latitude >= BOX["latmin"]) & (df.G2latitude <= BOX["latmax"])
            & (df.G2longitude >= BOX["lonmin"]) & (df.G2longitude <= BOX["lonmax"])]
    date = pd.to_datetime(dict(year=df.G2year, month=df.G2month, day=df.G2day),
                          errors="coerce")
    df = df[(date >= PERIOD[0]) & (date <= PERIOD[1])].copy()
    df["date"] = pd.to_datetime(dict(year=df.G2year, month=df.G2month, day=df.G2day)).dt.normalize()
    full = df[list(VALS.values())].notna().all(axis=1)
    df = df[full.values].copy()
    df["station_id"] = df.G2cruise.astype(int).astype(str) + "_" + df.G2station.astype(str)
    return df


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_fullset()
    samples = df.rename(columns={"G2latitude": "obs_lat", "G2longitude": "obs_lon",
                                 "G2depth": "depth", "G2temperature": "T_obs",
                                 "G2salinity": "S_obs", **{v: k for k, v in VALS.items()}})
    samples = samples[["station_id", "obs_lat", "obs_lon", "date", "depth",
                       "T_obs", "S_obs", *VALS.keys()]]
    samples.to_parquet(OUT / "fora_glodap_samples.parquet")
    # one row per station (profile position + date)
    st = df.drop_duplicates("station_id").set_index("station_id")
    st = st[["G2latitude", "G2longitude", "date"]].rename(
        columns={"G2latitude": "obs_lat", "G2longitude": "obs_lon"})
    print(f"full-set: {len(samples)} samples  {len(st)} stations  "
          f"{st.date.nunique()} dates", flush=True)

    # grid coords (constant) from the first available date
    ref = xr.open_dataset(fora_url("t", st.date.iloc[0]))
    flat = ref["lat"].values; flon = ref["lon"].values; fdepth = ref["depth"].values
    ref.close()
    nlev = len(fdepth)

    sids = list(st.index)
    sidx = {s: k for k, s in enumerate(sids)}
    Tprof = np.full((len(sids), nlev), np.nan)
    Sprof = np.full((len(sids), nlev), np.nan)
    fgrid_lat = np.full(len(sids), np.nan); fgrid_lon = np.full(len(sids), np.nan)

    dates = sorted(st.date.unique())
    for n, d in enumerate(dates, 1):
        rows = st[st.date == d]
        ds = pd.Timestamp(d)
        try:
            dt = xr.open_dataset(fora_url("t", ds)); dsal = xr.open_dataset(fora_url("s", ds))
        except Exception as exc:                                  # missing day -> skip
            print(f"  [{n}/{len(dates)}] {ds:%Y-%m-%d} open failed: {exc}", flush=True)
            continue
        jl = np.array([int(np.argmin(np.abs(flat - la))) for la in rows.obs_lat])
        il = np.array([int(np.argmin(np.abs(flon - lo))) for lo in rows.obs_lon])
        Ti = xr.DataArray(jl, dims="pt"); Ii = xr.DataArray(il, dims="pt")
        T = np.asarray(dt["thetao"].isel(time=0).isel(lat=Ti, lon=Ii).load().values)  # (lev,pt)
        S = np.asarray(dsal["so"].isel(time=0).isel(lat=Ti, lon=Ii).load().values)
        dt.close(); dsal.close()
        for p, sid in enumerate(rows.index):
            k = sidx[sid]
            Tprof[k] = T[:, p]; Sprof[k] = S[:, p]
            fgrid_lat[k] = flat[jl[p]]; fgrid_lon[k] = flon[il[p]]
        if n % 25 == 0 or n == len(dates):
            print(f"  [{n}/{len(dates)}] {ds:%Y-%m-%d} ({len(rows)} stn)", flush=True)

    out = xr.Dataset(
        {"fora_temp": (("station", "level"), Tprof),
         "fora_salt": (("station", "level"), Sprof),
         "obs_lat": ("station", st.obs_lat.values),
         "obs_lon": ("station", st.obs_lon.values),
         "fora_lat": ("station", fgrid_lat),
         "fora_lon": ("station", fgrid_lon),
         "date": ("station", pd.to_datetime(st.date.values))},
        coords={"station": sids, "level": np.arange(nlev), "fora_depth": ("level", fdepth)})
    enc = {v: {"zlib": True, "complevel": 4} for v in ["fora_temp", "fora_salt"]}
    out.to_netcdf(OUT / "fora_glodap_profiles.nc", encoding=enc)
    nval = int(np.isfinite(Tprof[:, 0]).sum())
    print(f"saved profiles for {nval}/{len(sids)} stations -> "
          f"{OUT/'fora_glodap_profiles.nc'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
