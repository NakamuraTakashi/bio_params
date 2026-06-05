"""Load BGC-Argo synthetic profiles (_Sprof.nc) into the project's common schema.

Common schema returned (one row per measurement level):
    latitude, longitude, depth, temperature, salinity,
    <target>, <target>_flag, source, time

`time` (datetime64) is added beyond the GLODAP schema because BGC-Argo's value
for ROMS boundary conditions is its seasonal coverage; downstream feature code
can use it (e.g. day-of-year) or ignore it.

Quality policy (see CLAUDE.md / project memory):
  * Use the *_ADJUSTED fields (delayed/adjusted calibration), not the raw ones.
  * Keep only levels whose ADJUSTED QC flag for the target AND for T, S, P is
    in `qc_ok` (default {'1','2'} = good / probably good).
  * Keep only profiles whose data mode for the target is in `modes`
    (default: CHLA -> delayed-mode only ('D'); O2/NO3 -> ('D','A')).

Units match GLODAP: DOXY/NITRATE in umol/kg, CHLA in mg/m3.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

SOURCE_NAME = "bgc_argo"

# Canonical project target -> Argo synthetic parameter name.
TARGET_PARAM: dict[str, str] = {
    "O2": "DOXY",
    "NO3": "NITRATE",
    "Chla": "CHLA",
}

# Default profile data-mode policy per target.
# BGC-Argo CHLA is rarely in delayed mode (~1% of profiles in the Kuroshio
# box), so delayed-only would discard almost all data; we accept adjusted
# real-time (A) as well. A-mode CHLA is factory-calibrated and auto-adjusted.
DEFAULT_MODES: dict[str, tuple[str, ...]] = {
    "Chla": ("D", "A"),
    "O2": ("D", "A"),
    "NO3": ("D", "A"),
}

# Physical plausibility range (inclusive) per target, in the project's units.
# A few values survive the WOCE QC flag yet are clearly unphysical (e.g. a
# single NO3 of 449.62 umol/kg vs an ocean max ~50). These bounds drop such
# spikes so they do not distort normalization / RMSE. Bounds are generous so
# legitimate deep-water extremes are kept.
VALID_RANGE: dict[str, tuple[float, float]] = {
    "O2": (0.0, 500.0),     # umol/kg; allow surface supersaturation
    "NO3": (0.0, 60.0),     # umol/kg; NW Pacific deep ~45
    "Chla": (0.0, 50.0),    # mg/m3
}

JULD_EPOCH = np.datetime64("1950-01-01T00:00:00")


def available_targets() -> list[str]:
    return list(TARGET_PARAM.keys())


def _decode(arr: np.ndarray) -> np.ndarray:
    """Return a unicode array (decodes bytes / |S dtypes), stripped."""
    a = np.asarray(arr)
    if a.dtype.kind == "S":
        a = np.char.decode(a, "utf-8")
    a = a.astype("U")
    return np.char.strip(a)


def _profile_target_mode(ds: xr.Dataset, param: str) -> np.ndarray:
    """Per-profile data mode (R/A/D) for `param`; '' if the param is absent."""
    n_prof = ds.sizes["N_PROF"]
    sp = _decode(ds["STATION_PARAMETERS"].values)        # (N_PROF, N_PARAM)
    pdm = _decode(ds["PARAMETER_DATA_MODE"].values)       # (N_PROF, N_PARAM)
    match = sp == param
    has = match.any(axis=1)
    idx = match.argmax(axis=1)
    mode = pdm[np.arange(n_prof), idx]
    mode = np.where(has, mode, "")
    return mode


def _level_field(ds: xr.Dataset, name: str) -> np.ndarray:
    """Return an (N_PROF, N_LEVELS) float array for `name` (NaN if absent)."""
    if name in ds:
        return ds[name].values.astype(np.float64)
    return np.full((ds.sizes["N_PROF"], ds.sizes["N_LEVELS"]), np.nan)


def _qc_ok(ds: xr.Dataset, name: str, qc_ok: set[str], shape) -> np.ndarray:
    """Boolean (N_PROF, N_LEVELS): True where the QC flag is acceptable."""
    if name in ds:
        return np.isin(_decode(ds[name].values), list(qc_ok))
    # No QC variable -> do not reject on it.
    return np.ones(shape, dtype=bool)


# Fluorescence dark-offset (residual dark count) correction for CHLA: estimate a
# per-float offset from the deep signal (where real Chl ~ 0) and subtract it.
DARK_DEEP_DBAR = 900.0       # prefer levels deeper than this for the offset
DARK_FALLBACK_DBAR = 300.0   # if the float is shallow, use > this instead
DARK_MIN_LEVELS = 5          # need at least this many deep levels to trust it


def load_sprof_file(
    path: str | Path,
    target: str,
    *,
    modes: tuple[str, ...] | None = None,
    qc_ok: set[str] | None = None,
    dark_correct: bool = False,
) -> pd.DataFrame:
    """Load one float's _Sprof.nc into common-schema rows for `target`.

    `dark_correct` (CHLA only): subtract a per-float fluorescence dark offset,
    estimated as the median CHLA below `DARK_DEEP_DBAR` (or `DARK_FALLBACK_DBAR`
    for shallow floats), then clip to >= 0. Removes the ~0.007 mg/m3 instrumental
    floor that does not cancel in the surface-normalized (relative) target.
    """
    if target not in TARGET_PARAM:
        raise ValueError(f"unknown target {target!r}; available {available_targets()}")
    param = TARGET_PARAM[target]
    modes = modes if modes is not None else DEFAULT_MODES[target]
    qc_ok = qc_ok if qc_ok is not None else {"1", "2"}

    ds = xr.open_dataset(path)
    try:
        val_name = f"{param}_ADJUSTED"
        if val_name not in ds:
            return _empty(target)

        shape = (ds.sizes["N_PROF"], ds.sizes["N_LEVELS"])
        value = _level_field(ds, val_name)
        temp = _level_field(ds, "TEMP_ADJUSTED")
        psal = _level_field(ds, "PSAL_ADJUSTED")
        pres = _level_field(ds, "PRES_ADJUSTED")

        if dark_correct and target == "Chla":
            fv = np.isfinite(value) & np.isfinite(pres)
            deep = fv & (pres > DARK_DEEP_DBAR)
            if int(deep.sum()) < DARK_MIN_LEVELS:
                deep = fv & (pres > DARK_FALLBACK_DBAR)
            off = float(np.nanmedian(value[deep])) if int(deep.sum()) >= DARK_MIN_LEVELS else 0.0
            value = np.clip(value - max(off, 0.0), 0.0, None)

        # Per-profile position; some profiles have missing/failed GPS fixes
        # (NaN LATITUDE/LONGITUDE). Exclude those so downstream features are
        # finite (a single such profile poisons feature/normalizer math).
        lat_col = ds["LATITUDE"].values.astype(np.float64)[:, None]
        lon_col = ds["LONGITUDE"].values.astype(np.float64)[:, None]
        pos_ok = np.isfinite(lat_col) & np.isfinite(lon_col)  # (N_PROF, 1)

        ok = (
            np.isfinite(value) & np.isfinite(temp)
            & np.isfinite(psal) & np.isfinite(pres)
            & pos_ok
            & _qc_ok(ds, f"{val_name}_QC", qc_ok, shape)
            & _qc_ok(ds, "TEMP_ADJUSTED_QC", qc_ok, shape)
            & _qc_ok(ds, "PSAL_ADJUSTED_QC", qc_ok, shape)
            & _qc_ok(ds, "PRES_ADJUSTED_QC", qc_ok, shape)
        )

        # Restrict to profiles whose target data mode is allowed.
        prof_mode = _profile_target_mode(ds, param)            # (N_PROF,)
        prof_ok = np.isin(prof_mode, list(modes))
        ok &= prof_ok[:, None]

        if not ok.any():
            return _empty(target)

        # Per-profile scalars broadcast to levels (reuse lat/lon from above).
        juld = ds["JULD"].values                                # datetime64 or float
        lat_b = np.broadcast_to(lat_col, shape)[ok]
        lon_b = np.broadcast_to(lon_col, shape)[ok]

        # Depth (m, positive down) from pressure via TEOS-10.
        import gsw
        depth = -gsw.z_from_p(pres[ok], lat_b)

        qc_flag = _decode(ds[f"{val_name}_QC"].values)[ok] if f"{val_name}_QC" in ds \
            else np.full(int(ok.sum()), "", dtype="U1")

        time_b = _juld_to_time(juld, shape)[ok]

        out = pd.DataFrame({
            "latitude": lat_b,
            "longitude": lon_b,
            "depth": depth,
            "temperature": temp[ok],
            "salinity": psal[ok],
            target: value[ok],
            f"{target}_flag": qc_flag,
            "source": SOURCE_NAME,
            "time": time_b,
        })
        return out
    finally:
        ds.close()


def _juld_to_time(juld: np.ndarray, shape) -> np.ndarray:
    """Broadcast per-profile JULD to (N_PROF, N_LEVELS) datetime64[ns]."""
    juld = np.asarray(juld)
    if np.issubdtype(juld.dtype, np.datetime64):
        t = juld.astype("datetime64[ns]")
    else:  # days since 1950-01-01
        t = (JULD_EPOCH + (juld * 86400.0).astype("timedelta64[s]")).astype(
            "datetime64[ns]"
        )
    return np.broadcast_to(t[:, None], shape)


def _empty(target: str) -> pd.DataFrame:
    cols = ["latitude", "longitude", "depth", "temperature", "salinity",
            target, f"{target}_flag", "source", "time"]
    return pd.DataFrame({c: [] for c in cols})


def attach_surface_chla(
    df: pd.DataFrame, matchup_parquet: str | Path
) -> pd.DataFrame:
    """Add a `surface_chla` column from the satellite matchup parquet.

    The matchup file (see scripts/match_satellite_chla.py) holds one row per
    profile keyed by (latitude, longitude, time) with the GlobColour surface
    Chl-a. Merging broadcasts that value to every level of the profile. Rows
    whose profile was not matched (e.g. masked satellite pixel, or a date
    beyond the satellite product) get NaN and should be dropped by the caller.
    """
    m = pd.read_parquet(matchup_parquet)
    m["time"] = pd.to_datetime(m["time"])
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    return out.merge(
        m[["latitude", "longitude", "time", "surface_chla"]],
        on=["latitude", "longitude", "time"], how="left",
    )


def load_bgc_argo(
    sprof_dir: str | Path,
    target: str,
    *,
    box: tuple[float, float, float, float] | None = None,
    modes: tuple[str, ...] | None = None,
    qc_ok: set[str] | None = None,
    valid_range: tuple[float, float] | None = None,
    dark_correct: bool = False,
) -> pd.DataFrame:
    """Load all _Sprof.nc in `sprof_dir` into common-schema rows for `target`.

    `box` = (lon0, lon1, lat0, lat1) optionally clips to a region.
    `valid_range` = (lo, hi) drops out-of-range target values; defaults to
    `VALID_RANGE[target]` (a physical plausibility filter for QC survivors).
    Pass `(-inf, inf)` to disable.
    """
    sprof_dir = Path(sprof_dir)
    files = sorted(sprof_dir.glob("*_Sprof.nc"))
    if not files:
        raise FileNotFoundError(f"no *_Sprof.nc in {sprof_dir}")

    frames = [load_sprof_file(f, target, modes=modes, qc_ok=qc_ok,
                              dark_correct=dark_correct) for f in files]
    df = pd.concat(frames, ignore_index=True)
    # Concatenating with empty frames can demote `time` to object dtype, which
    # breaks the .dt accessor used downstream (e.g. day-of-year features).
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    if box is not None:
        lon0, lon1, lat0, lat1 = box
        df = df[(df.latitude >= lat0) & (df.latitude <= lat1)
                & (df.longitude >= lon0) & (df.longitude <= lon1)]

    rng = valid_range if valid_range is not None else VALID_RANGE.get(target)
    if rng is not None:
        lo, hi = rng
        df = df[(df[target] >= lo) & (df[target] <= hi)]

    return df.reset_index(drop=True)
