"""Load co-located Chl-a + NO3 samples (levels where BOTH are measured).

For the light+nutrient Chl-a model: light (from satellite-derived Kd) and NO3
jointly control the vertical phytoplankton distribution, so we train on samples
that have a measured NO3 at the same level as Chl-a. Returns the common schema
plus an `NO3` feature column (umol/kg). NO3 at inference comes from the (high
skill) NO3 model, fed in before the Chl-a model.

Sources: "glodap", "bgc_argo", "combined". Both sources measure both on the
same profile levels, so the co-located subset is large (GLODAP ~50k rows,
BGC-Argo ~1.44M rows).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bio_params.loaders.bgc_argo import load_bgc_argo
from bio_params.loaders.glodap import load_glodap

_KEYS = ["latitude", "longitude", "time"]


def _merge_chla_no3(chla_df: pd.DataFrame, no3_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join Chl-a and NO3 on profile + level (depth rounded to 0.1 m)."""
    a, b = chla_df.copy(), no3_df.copy()
    for d in (a, b):
        d["time"] = pd.to_datetime(d["time"])
        d["_dk"] = d["depth"].round(1)
    m = a.merge(b[_KEYS + ["_dk", "NO3"]], on=_KEYS + ["_dk"], how="inner")
    return m.drop(columns="_dk")


def load_chla_no3(
    source: str,
    *,
    glodap_csv: str | Path | None = None,
    sprof_dir: str | Path | None = None,
    box: tuple[float, float, float, float] | None = None,
) -> pd.DataFrame:
    """Common-schema rows (+ NO3 column) for levels with both Chl-a and NO3."""
    frames = []
    if source in ("glodap", "combined"):
        gc = load_glodap(glodap_csv, "Chla", with_time=True)
        gn = load_glodap(glodap_csv, "NO3", with_time=True)
        frames.append(_merge_chla_no3(gc, gn))
    if source in ("bgc_argo", "combined"):
        ac = load_bgc_argo(sprof_dir, "Chla", box=box)
        an = load_bgc_argo(sprof_dir, "NO3", box=box)
        frames.append(_merge_chla_no3(ac, an))
    if not frames:
        raise ValueError(f"unknown source {source!r}")
    df = pd.concat(frames, ignore_index=True)

    if box is not None:
        lon0, lon1, lat0, lat1 = box
        df = df[(df.longitude >= lon0) & (df.longitude <= lon1)
                & (df.latitude >= lat0) & (df.latitude <= lat1)]
    return df.reset_index(drop=True)
