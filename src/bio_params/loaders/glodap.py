"""Load GLODAPv2.2023 Merged Master CSV into the project's common schema.

Common schema (one row per water sample):
    latitude, longitude, depth, temperature, salinity,
    <target>, <target>_flag, source

`<target>` is the canonical project name (e.g. 'NO3'), not the GLODAP column.
`source` is set to 'glodap' for downstream provenance tracking.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_NAME = "glodap"

# GLODAP encodes missing values as -9999 across all numeric columns
# (confirmed against GLODAPv2.2023 Merged Master File CSV).
MISSING_SENTINEL = -9999

# Canonical target name -> (value column, flag column).
# All GLODAPv2.2023 variables below carry a WOCE QC flag column (G2<var>f);
# for the non-core variables (DOC, Chla, TDN, TOC, DON, C13, O18, C14, H3)
# every reported value is flagged 2, so flag filtering is a no-op but kept for
# uniformity. Note their units differ: nutrients/carbon are umol/kg, Chla is
# mg/m3 (~ug/L), C13/O18/C14 are isotope ratios in per mil (C14 = Delta-14C),
# and H3 (tritium) is in tritium units (TU). C14 and H3 are transient tracers
# (they decay / reflect bomb input) so a static T-S model is only a rough
# spatial proxy; included for exploratory / future use.
TARGET_COLUMNS: dict[str, tuple[str, str | None]] = {
    "DIC":  ("G2tco2",      "G2tco2f"),
    "TA":   ("G2talk",      "G2talkf"),
    "NO3":  ("G2nitrate",   "G2nitratef"),
    "PO4":  ("G2phosphate", "G2phosphatef"),
    "SiO4": ("G2silicate",  "G2silicatef"),
    "O2":   ("G2oxygen",    "G2oxygenf"),
    "DOC":  ("G2doc",       "G2docf"),
    "Chla": ("G2chla",      "G2chlaf"),
    "TDN":  ("G2tdn",       "G2tdnf"),
    "TOC":  ("G2toc",       "G2tocf"),
    "DON":  ("G2don",       "G2donf"),
    "C13":  ("G2c13",       "G2c13f"),
    "O18":  ("G2o18",       "G2o18f"),
    "C14":  ("G2c14",       "G2c14f"),
    "H3":   ("G2h3",        "G2h3f"),
}

# Common-schema column name -> GLODAP column name.
COORDINATE_COLUMNS: dict[str, str] = {
    "latitude":    "G2latitude",
    "longitude":   "G2longitude",
    "depth":       "G2depth",
    "temperature": "G2temperature",
    "salinity":    "G2salinity",
}


def available_targets() -> list[str]:
    return list(TARGET_COLUMNS.keys())


def load_glodap(
    csv_path: str | Path,
    target: str,
    *,
    require_flag2: bool = True,
    drop_missing_coords: bool = True,
    with_time: bool = False,
) -> pd.DataFrame:
    """Load one target's measurements from the GLODAP CSV.

    Parameters
    ----------
    csv_path
        Path to GLODAPv2.2023_Merged_Master_File.csv.
    target
        Canonical target name; one of `available_targets()`.
    require_flag2
        If True (default), keep only rows where the WOCE flag equals 2
        (good measurement). For non-core variables without a flag column
        (DOC, Chla), this is equivalent to dropping null values.
    drop_missing_coords
        If True (default), drop rows missing any of the coordinate /
        physical columns (latitude, longitude, depth, T, S). Required
        before training since the model needs all of them as input.

    Returns
    -------
    DataFrame with the project's common-schema columns:
    latitude, longitude, depth, temperature, salinity,
    <target>, <target>_flag, source.
    """
    if target not in TARGET_COLUMNS:
        raise ValueError(
            f"unknown target {target!r}; available: {available_targets()}"
        )
    value_col, flag_col = TARGET_COLUMNS[target]

    # Read header alone so we can restrict to columns that actually exist
    # (some non-core columns may be absent in regional CSV variants).
    header = pd.read_csv(csv_path, nrows=0).columns
    needed = list(COORDINATE_COLUMNS.values()) + [value_col]
    if flag_col is not None:
        needed.append(flag_col)
    time_cols = ["G2year", "G2month", "G2day"]
    if with_time:
        needed += [c for c in time_cols if c in header]
    missing = [c for c in needed if c not in header and c not in time_cols]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = pd.read_csv(csv_path, usecols=needed, low_memory=False)
    df = df.replace(MISSING_SENTINEL, np.nan)

    rename_map: dict[str, str] = {gl: common for common, gl in COORDINATE_COLUMNS.items()}
    rename_map[value_col] = target
    if flag_col is not None:
        rename_map[flag_col] = f"{target}_flag"
    df = df.rename(columns=rename_map)

    if require_flag2:
        if flag_col is not None:
            df = df[df[f"{target}_flag"] == 2]
        else:
            df = df[df[target].notna()]
    else:
        df = df[df[target].notna()]

    if drop_missing_coords:
        df = df.dropna(subset=list(COORDINATE_COLUMNS.keys()))

    # Add a flag column for non-core targets so the schema is uniform.
    flag_out = f"{target}_flag"
    if flag_out not in df.columns:
        df[flag_out] = pd.NA

    df = df.copy()
    df["source"] = SOURCE_NAME

    column_order = (
        list(COORDINATE_COLUMNS.keys()) + [target, flag_out, "source"]
    )

    if with_time and all(c in df.columns for c in ("G2year", "G2month")):
        # Build a real timestamp from year/month/day (day clamped to a valid
        # mid-month default when absent), so profiles can be matched to the
        # monthly satellite product. Invalid dates -> NaT.
        day = (df["G2day"] if "G2day" in df.columns else 15)
        parts = pd.DataFrame({
            "year": pd.to_numeric(df["G2year"], errors="coerce"),
            "month": pd.to_numeric(df["G2month"], errors="coerce"),
            "day": pd.to_numeric(day, errors="coerce").fillna(15).clip(1, 28),
        })
        parts.loc[~parts["month"].between(1, 12), "month"] = np.nan
        df["time"] = pd.to_datetime(parts, errors="coerce")
        column_order = column_order + ["time"]

    return df[column_order].reset_index(drop=True)
