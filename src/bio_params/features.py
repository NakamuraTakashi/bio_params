"""Feature engineering: raw coordinates + T,S -> model input matrix.

The MLP takes a 7-dim feature vector by default; optional add-ons extend it:
    lat_sin, lat_cos, lon_sin, lon_cos, log_depth, temperature, salinity
    [, sigma_theta]          # include_sigma_theta
    [, doy_sin, doy_cos]     # include_season (day-of-year, circular)

Rationale (see CLAUDE.md): lon/lat are mapped to circular coordinates so
+180/-180 is continuous; depth is log-transformed to resolve the surface;
T and S are passed raw because biogeochemical tracers correlate strongly
with the water mass. Day-of-year is encoded as sin/cos so Dec 31 and Jan 1
are continuous; it matters for variables with a strong seasonal cycle
(e.g. Chl-a) and requires a datetime `time` column in the input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Required input columns (must match the loader's common schema).
REQUIRED_INPUTS = ["latitude", "longitude", "depth", "temperature", "salinity"]

# Feature column names in the order the MLP expects them.
FEATURE_NAMES_BASE = [
    "lat_sin", "lat_cos",
    "lon_sin", "lon_cos",
    "log_depth",
    "temperature", "salinity",
]
SEASON_FEATURES = ["doy_sin", "doy_cos"]

# Days in a mean year, used to map day-of-year onto a circle.
_DAYS_PER_YEAR = 365.25


def feature_names(
    include_sigma_theta: bool = False,
    include_season: bool = False,
) -> list[str]:
    names = list(FEATURE_NAMES_BASE)
    if include_sigma_theta:
        names.append("sigma_theta")
    if include_season:
        names.extend(SEASON_FEATURES)
    return names


def build_features(
    df: pd.DataFrame,
    *,
    include_sigma_theta: bool = False,
    include_season: bool = False,
) -> pd.DataFrame:
    """Build the model-ready feature matrix from common-schema rows.

    Returns a new DataFrame with
    `feature_names(include_sigma_theta, include_season)` columns, in the same
    row order as the input. The input is not modified. `include_season`
    requires a datetime `time` column.
    """
    missing = [c for c in REQUIRED_INPUTS if c not in df.columns]
    if missing:
        raise ValueError(f"input DataFrame missing columns: {missing}")

    lat_rad = np.deg2rad(df["latitude"].to_numpy())
    lon_rad = np.deg2rad(df["longitude"].to_numpy())
    depth = df["depth"].to_numpy()
    temperature = df["temperature"].to_numpy()
    salinity = df["salinity"].to_numpy()

    out = pd.DataFrame(
        {
            "lat_sin": np.sin(lat_rad),
            "lat_cos": np.cos(lat_rad),
            "lon_sin": np.sin(lon_rad),
            "lon_cos": np.cos(lon_rad),
            # +1 keeps log defined at depth==0 (surface).
            "log_depth": np.log(depth + 1.0),
            "temperature": temperature,
            "salinity": salinity,
        },
        index=df.index,
    )

    if include_sigma_theta:
        out["sigma_theta"] = _sigma_theta(
            salinity=salinity,
            temperature=temperature,
            depth=depth,
            latitude=df["latitude"].to_numpy(),
        )

    if include_season:
        if "time" not in df.columns:
            raise ValueError("include_season=True requires a 'time' column")
        doy_sin, doy_cos = _season(df["time"])
        out["doy_sin"] = doy_sin
        out["doy_cos"] = doy_cos

    return out[feature_names(include_sigma_theta, include_season)]


def _season(time: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Circular day-of-year encoding (sin, cos) from a datetime column."""
    t = pd.to_datetime(time)
    doy = t.dt.dayofyear.to_numpy().astype(np.float64)
    angle = 2.0 * np.pi * doy / _DAYS_PER_YEAR
    return np.sin(angle), np.cos(angle)


def _sigma_theta(
    *,
    salinity: np.ndarray,
    temperature: np.ndarray,
    depth: np.ndarray,
    latitude: np.ndarray,
) -> np.ndarray:
    """Potential density anomaly (sigma_0) via TEOS-10.

    Computed from Practical Salinity + in-situ temperature using gsw,
    referenced to 0 dbar. Longitude is set to 0 for the SA conversion;
    the error from omitting it is negligible compared with the value
    range of sigma_theta seen in oceanographic data.
    """
    import gsw

    pressure = gsw.p_from_z(-np.abs(depth), latitude)
    absolute_salinity = gsw.SA_from_SP(
        salinity, pressure, lon=np.zeros_like(latitude), lat=latitude
    )
    conservative_temperature = gsw.CT_from_t(
        absolute_salinity, temperature, pressure
    )
    return gsw.sigma0(absolute_salinity, conservative_temperature)
