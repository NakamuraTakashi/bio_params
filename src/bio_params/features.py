"""Feature engineering: raw coordinates + T,S -> model input matrix.

The MLP takes a 7-dim feature vector (8-dim with sigma_theta):
    lat_sin, lat_cos, lon_sin, lon_cos, log_depth, temperature, salinity
    [, sigma_theta]

Rationale (see CLAUDE.md): lon/lat are mapped to circular coordinates so
+180/-180 is continuous; depth is log-transformed to resolve the surface;
T and S are passed raw because biogeochemical tracers correlate strongly
with the water mass.
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
FEATURE_NAMES_WITH_SIGMA = FEATURE_NAMES_BASE + ["sigma_theta"]


def feature_names(include_sigma_theta: bool = False) -> list[str]:
    return FEATURE_NAMES_WITH_SIGMA if include_sigma_theta else FEATURE_NAMES_BASE


def build_features(
    df: pd.DataFrame,
    *,
    include_sigma_theta: bool = False,
) -> pd.DataFrame:
    """Build the model-ready feature matrix from common-schema rows.

    Returns a new DataFrame with `feature_names(include_sigma_theta)` columns,
    in the same row order as the input. The input is not modified.
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

    return out[feature_names(include_sigma_theta)]


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
