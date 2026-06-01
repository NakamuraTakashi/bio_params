"""Per-profile derived quantities: mixed-layer depth (MLD) and the
surface-normalized (relative) vertical profile of a target.

These need the whole T/S (and target) column of a profile, unlike the point-wise
features. A "profile" is identified by the key columns (default latitude,
longitude, time). Both helpers broadcast the per-profile scalar back to every
level row so the downstream point-wise feature/training code is unchanged.

Rationale (SOCA, Sauzede et al. 2016):
  * MLD is a strong predictor of the vertical Chl shape (mixed vs DCM regime).
  * Normalizing the profile by its own surface value isolates the SHAPE from the
    amplitude. The amplitude is restored at inference from the satellite surface
    field, so the (multiplicative) in-situ sensor calibration bias cancels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_KEYS = ("latitude", "longitude", "time")


def sigma0(salinity, temperature, depth, latitude) -> np.ndarray:
    """Potential density anomaly sigma_0 (kg/m3) via TEOS-10 (gsw)."""
    import gsw
    p = gsw.p_from_z(-np.abs(depth), latitude)
    SA = gsw.SA_from_SP(salinity, p, lon=np.zeros_like(latitude), lat=latitude)
    CT = gsw.CT_from_t(SA, temperature, p)
    return gsw.sigma0(SA, CT)


def add_mld(
    df: pd.DataFrame,
    *,
    keys=DEFAULT_KEYS,
    threshold: float = 0.03,
    ref_depth: float = 10.0,
) -> pd.DataFrame:
    """Add an `mld` column (m): density-threshold mixed-layer depth per profile.

    MLD = shallowest depth where sigma_0 exceeds its value at the reference
    level (nearest `ref_depth`) by `threshold` kg/m3 (de Boyer Montegut 2004).
    A fully-mixed sampled column gets its deepest sampled depth. Profiles with
    < 2 finite levels get NaN.
    """
    keys = list(keys)
    out = df.copy()
    sig = sigma0(out["salinity"].to_numpy(), out["temperature"].to_numpy(),
                 out["depth"].to_numpy(), out["latitude"].to_numpy())
    out["_sig0"] = sig

    def _mld(g: pd.DataFrame) -> float:
        g = g.sort_values("depth")
        d = g["depth"].to_numpy()
        s = g["_sig0"].to_numpy()
        ok = np.isfinite(d) & np.isfinite(s)
        d, s = d[ok], s[ok]
        if d.size < 2:
            return np.nan
        iref = int(np.argmin(np.abs(d - ref_depth)))
        exceed = np.where((s > s[iref] + threshold) & (np.arange(d.size) > iref))[0]
        return float(d[exceed[0]]) if exceed.size else float(d[-1])

    mld = out.groupby(keys, sort=False).apply(_mld, include_groups=False)
    mld.name = "mld"
    out = out.merge(mld, left_on=keys, right_index=True, how="left")
    return out.drop(columns="_sig0")


def add_relative_target(
    df: pd.DataFrame,
    target: str,
    *,
    keys=DEFAULT_KEYS,
    surf_max_depth: float = 10.0,
    surf_floor: float = 0.01,
    rel_cap: float = 20.0,
) -> pd.DataFrame:
    """Add `<target>_surf` and `<target>_rel` (= value / surface value).

    The surface value is the shallowest measurement within `surf_max_depth` m of
    each profile. Profiles with no such level, or a surface value <= `surf_floor`
    mg/m3, are dropped (normalizing by a near-zero surface blows the ratio up).
    The relative value is clipped to [0, `rel_cap`]: negatives are sensor noise,
    and a subsurface max (DCM) rarely exceeds ~10x the surface, so values beyond
    `rel_cap` are artifacts of a tiny surface value.
    """
    keys = list(keys)
    out = df.copy()
    near = out[out["depth"] <= surf_max_depth].sort_values("depth")
    surf = near.groupby(keys, sort=False)[target].first()
    surf.name = f"{target}_surf"
    out = out.merge(surf, left_on=keys, right_index=True, how="left")
    sv = out[f"{target}_surf"]
    out = out[np.isfinite(sv) & (sv > surf_floor)].reset_index(drop=True)
    rel = out[target] / out[f"{target}_surf"]
    out[f"{target}_rel"] = np.clip(rel.to_numpy(), 0.0, rel_cap)
    return out
