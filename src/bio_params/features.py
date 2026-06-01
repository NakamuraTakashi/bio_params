"""Feature engineering: raw coordinates + T,S -> model input matrix.

The MLP takes a 7-dim feature vector by default; optional add-ons extend it:
    lat_sin, lat_cos, lon_sin, lon_cos, log_depth, temperature, salinity
    [, sigma_theta]          # include_sigma_theta
    [, doy_sin, doy_cos]     # include_season (day-of-year, circular)
    [, surface_chla]         # include_surface_chla (SOCA-style surface anchor)
    [, log_mld]              # include_mld (mixed-layer depth, SOCA Zm input)

Rationale (see CLAUDE.md): lon/lat are mapped to circular coordinates so
+180/-180 is continuous; depth is log-transformed to resolve the surface;
T and S are passed raw because biogeochemical tracers correlate strongly
with the water mass. Day-of-year is encoded as sin/cos so Dec 31 and Jan 1
are continuous; it matters for variables with a strong seasonal cycle
(e.g. Chl-a) and requires a datetime `time` column in the input.

`include_surface_chla` follows Sauzede et al. (2016, SOCA) in spirit: the
satellite surface Chl-a anchors the amplitude of the vertical profile while
T/S/depth carry its shape. It requires a `surface_chla` column (mg/m3) on the
input, e.g. attached from the GlobColour matchup parquet.

`surface_chla_log` selects the transform. log10 (default, SOCA's choice)
compresses the order-of-magnitude range so the feature extrapolates gently;
empirically a LINEAR feature makes one spatial-CV fold blow up (R^2 -3.7)
because the steep learned slope extrapolates badly into an unseen high-Chl
block. The feature column is named `log_surface_chla` or `surface_chla`
accordingly, so the artifact's feature_names records which transform was used.
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
# Floor (mg/m3) added before log10 so the open-ocean minimum stays finite.
_CHLA_FLOOR = 1e-3


def surface_chla_feature_name(log: bool = True) -> str:
    return "log_surface_chla" if log else "surface_chla"


def feature_names(
    include_sigma_theta: bool = False,
    include_season: bool = False,
    include_surface_chla: bool = False,
    surface_chla_log: bool = True,
    include_mld: bool = False,
) -> list[str]:
    names = list(FEATURE_NAMES_BASE)
    if include_sigma_theta:
        names.append("sigma_theta")
    if include_season:
        names.extend(SEASON_FEATURES)
    if include_surface_chla:
        names.append(surface_chla_feature_name(surface_chla_log))
    if include_mld:
        names.append("log_mld")
    return names


def build_features(
    df: pd.DataFrame,
    *,
    include_sigma_theta: bool = False,
    include_season: bool = False,
    include_surface_chla: bool = False,
    surface_chla_log: bool = True,
    include_mld: bool = False,
) -> pd.DataFrame:
    """Build the model-ready feature matrix from common-schema rows.

    Returns a new DataFrame with
    `feature_names(include_sigma_theta, include_season, include_surface_chla)`
    columns, in the same row order as the input. The input is not modified.
    `include_season` requires a datetime `time` column; `include_surface_chla`
    requires a `surface_chla` column (mg/m3).
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

    if include_surface_chla:
        if "surface_chla" not in df.columns:
            raise ValueError(
                "include_surface_chla=True requires a 'surface_chla' column"
            )
        chla = df["surface_chla"].to_numpy(dtype=np.float64)
        name = surface_chla_feature_name(surface_chla_log)
        if surface_chla_log:
            out[name] = np.log10(np.maximum(chla, _CHLA_FLOOR))
        else:
            out[name] = chla

    if include_mld:
        if "mld" not in df.columns:
            raise ValueError("include_mld=True requires an 'mld' column")
        # log compresses the ~3-500 m MLD range, like log_depth.
        out["log_mld"] = np.log(df["mld"].to_numpy(dtype=np.float64) + 1.0)

    return out[feature_names(
        include_sigma_theta, include_season, include_surface_chla,
        surface_chla_log, include_mld)]


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
