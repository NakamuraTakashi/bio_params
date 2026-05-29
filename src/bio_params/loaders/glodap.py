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
# Non-core variables (DOC, Chla) have no WOCE QC flag column.
TARGET_COLUMNS: dict[str, tuple[str, str | None]] = {
    "DIC":  ("G2tco2",      "G2tco2f"),
    "TA":   ("G2talk",      "G2talkf"),
    "NO3":  ("G2nitrate",   "G2nitratef"),
    "PO4":  ("G2phosphate", "G2phosphatef"),
    "SiO4": ("G2silicate",  "G2silicatef"),
    "O2":   ("G2oxygen",    "G2oxygenf"),
    "DOC":  ("G2doc",       None),
    "Chla": ("G2chla",      None),
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
    missing = [c for c in needed if c not in header]
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
    return df[column_order].reset_index(drop=True)
