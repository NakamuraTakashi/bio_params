"""Load a target from GLODAP and BGC-Argo together into one common-schema frame.

Concatenates the two source loaders' output (same canonical target name, same
units: NO3/O2 in umol/kg, Chla in mg/m3). GLODAP has no per-profile time, so a
NaT `time` column is added to keep the schema uniform with BGC-Argo. The
`source` column ("glodap" / "bgc_argo") is preserved so downstream code can
weight, subsample, or split by source.

Optional per-source subsampling balances the two contributions (BGC-Argo O2 and
Chl-a dwarf GLODAP by 1-2 orders of magnitude).

Only targets present in BOTH sources are supported: NO3, O2, Chla.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bio_params.loaders.bgc_argo import load_bgc_argo
from bio_params.loaders.glodap import load_glodap

# Canonical targets available in both sources.
COMMON_TARGETS = ["NO3", "O2", "Chla"]


def available_targets() -> list[str]:
    return list(COMMON_TARGETS)


def load_combined(
    glodap_csv: str | Path,
    bgc_argo_sprof_dir: str | Path,
    target: str,
    *,
    box: tuple[float, float, float, float] | None = None,
    per_source_max: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Load `target` from GLODAP + BGC-Argo into one common-schema DataFrame.

    Parameters
    ----------
    box
        (lon0, lon1, lat0, lat1). Applied to both sources. GLODAP uses -180..180
        longitudes; BGC-Argo is normalized the same way by its loader.
    per_source_max
        If set, randomly keep at most this many rows from EACH source before
        concatenation (balances the contributions). Sampling is uniform.
    """
    if target not in COMMON_TARGETS:
        raise ValueError(f"combined loader supports {COMMON_TARGETS}, got {target!r}")

    g = load_glodap(glodap_csv, target=target, with_time=True)
    if "time" not in g.columns:
        g = g.copy()
        g["time"] = pd.NaT
    if box is not None:
        lon0, lon1, lat0, lat1 = box
        g = g[(g.longitude >= lon0) & (g.longitude <= lon1)
              & (g.latitude >= lat0) & (g.latitude <= lat1)]

    a = load_bgc_argo(bgc_argo_sprof_dir, target=target, box=box)

    # Align columns (GLODAP lacks nothing now; both share the common schema).
    cols = ["latitude", "longitude", "depth", "temperature", "salinity",
            target, f"{target}_flag", "source", "time"]
    g = g[[c for c in cols if c in g.columns]]
    a = a[[c for c in cols if c in a.columns]]

    rng = np.random.default_rng(seed)
    if per_source_max is not None:
        if len(g) > per_source_max:
            g = g.iloc[np.sort(rng.choice(len(g), per_source_max, replace=False))]
        if len(a) > per_source_max:
            a = a.iloc[np.sort(rng.choice(len(a), per_source_max, replace=False))]

    df = pd.concat([g, a], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df
