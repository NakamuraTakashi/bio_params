"""Map the global spatial coverage of BGC-Argo profiles per parameter.

For each target (O2/DOXY, Chla/CHLA, NO3/NITRATE) it scans every _Sprof.nc,
collects the profile locations (one lon/lat per profile) where that parameter
is present in an accepted data mode (D/A), and scatters them on a world map
with coastlines. Also produces a Japan-coast focused version per parameter.

Read-only: safe to run while training is using the same files.

Usage:
    uv run python scripts/plot_bgc_argo_coverage.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
except Exception:  # noqa: BLE001
    _HAS_CARTOPY = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPROF_DIR = PROJECT_ROOT / "data" / "bgc_argo" / "raw" / "floats"
OUT_DIR = PROJECT_ROOT / "figures" / "bgc_argo_coverage"

# Canonical target -> (Argo parameter, display, color).
TARGETS = {
    "O2": dict(param="DOXY", color="tab:red"),
    "Chla": dict(param="CHLA", color="tab:green"),
    "NO3": dict(param="NITRATE", color="tab:blue"),
}
ACCEPTED_MODES = ("D", "A")

# Japan-coast focus window (lon0, lon1, lat0, lat1).
JAPAN_EXTENT = (120.0, 160.0, 20.0, 50.0)

# Global map left edge at 30 W -> central longitude 150 E (Pacific-centered).
GLOBAL_CENTRAL_LON = 150.0


def _decode(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype.kind == "S":
        a = np.char.decode(a, "utf-8")
    return np.char.strip(a.astype("U"))


def profile_modes(ds: xr.Dataset, param: str) -> np.ndarray:
    """Per-profile data mode (R/A/D) for `param`; '' if absent."""
    if "STATION_PARAMETERS" not in ds.variables or "PARAMETER_DATA_MODE" not in ds.variables:
        return np.array([], dtype="U1")
    sp = _decode(ds["STATION_PARAMETERS"].values)
    pdm = _decode(ds["PARAMETER_DATA_MODE"].values)
    n = sp.shape[0]
    match = sp == param
    has = match.any(axis=1)
    idx = match.argmax(axis=1)
    mode = pdm[np.arange(n), idx]
    return np.where(has, mode, "")


def collect_locations(param: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (lon, lat, n_floats) of profiles carrying `param` in D/A mode."""
    lons, lats = [], []
    n_floats = 0
    for f in sorted(glob.glob(str(SPROF_DIR / "*_Sprof.nc"))):
        try:
            ds = xr.open_dataset(f)
        except Exception:
            continue
        try:
            modes = profile_modes(ds, param)
            if modes.size == 0:
                continue
            keep = np.isin(modes, ACCEPTED_MODES)
            if not keep.any():
                continue
            lat = ds["LATITUDE"].values
            lon = ds["LONGITUDE"].values
            ok = keep & np.isfinite(lat) & np.isfinite(lon)
            if ok.any():
                lons.append(lon[ok])
                lats.append(lat[ok])
                n_floats += 1
        finally:
            ds.close()
    if lons:
        return np.concatenate(lons), np.concatenate(lats), n_floats
    return np.array([]), np.array([]), 0


def _make_ax(fig, extent=None, central_lon=0.0):
    """Return an axis with coastlines (cartopy if available, else plain).

    For the global map, `central_lon` sets the projection centre (150 E puts
    the left edge at 30 W, i.e. a Pacific-centered view). Regional maps use the
    given `extent` and a 0-centred projection.
    """
    if _HAS_CARTOPY:
        proj = ccrs.PlateCarree(central_longitude=central_lon if extent is None else 0.0)
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        try:
            ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=0)
            ax.coastlines(resolution="50m", linewidth=0.5, color="0.3", zorder=3)
        except Exception:  # noqa: BLE001 - coastline data download may fail
            pass
        gl = ax.gridlines(draw_labels=True, ls=":", alpha=0.4)
        gl.top_labels = gl.right_labels = False
        if extent is not None:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        else:
            ax.set_global()
        return ax, dict(transform=ccrs.PlateCarree())
    # Fallback: plain lon/lat axes, no coastline.
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    else:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
    ax.set_aspect("equal")
    return ax, {}


def plot_map(lon, lat, *, color, title, out_path, extent=None, point_size=2,
             alpha=0.25) -> None:
    fig = plt.figure(figsize=(12, 6) if extent is None else (8, 7))
    ax, kw = _make_ax(fig, extent=extent, central_lon=GLOBAL_CENTRAL_LON)
    lon = ((np.asarray(lon) + 180) % 360) - 180
    ax.scatter(lon, lat, s=point_size, alpha=alpha, color=color,
               edgecolor="none", zorder=2, **kw)
    ax.set_title(title, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def in_extent(lon, lat, extent):
    lon = ((np.asarray(lon) + 180) % 360) - 180
    lo0, lo1, la0, la1 = extent
    m = (lon >= lo0) & (lon <= lo1) & (lat >= la0) & (lat <= la1)
    return m


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not _HAS_CARTOPY:
        print("WARNING: cartopy unavailable; maps will have no coastlines")
    summary = {}
    all_lon, all_lat = {}, {}
    for tgt, meta in TARGETS.items():
        lon, lat, n_floats = collect_locations(meta["param"])
        summary[tgt] = (len(lon), n_floats)
        all_lon[tgt], all_lat[tgt] = lon, lat
        print(f"{tgt} ({meta['param']}): {len(lon):,} profiles from {n_floats} floats")
        # Global map.
        plot_map(
            lon, lat, color=meta["color"],
            title=(f"BGC-Argo {tgt} ({meta['param']}) profile locations  "
                   f"n={len(lon):,} profiles, {n_floats} floats  (D/A mode)"),
            out_path=OUT_DIR / f"coverage_{tgt}.png",
        )
        # Japan-coast focus.
        m = in_extent(lon, lat, JAPAN_EXTENT)
        plot_map(
            lon[m], lat[m], color=meta["color"],
            title=(f"BGC-Argo {tgt} ({meta['param']}) near Japan  "
                   f"n={int(m.sum()):,} profiles  (D/A mode)"),
            out_path=OUT_DIR / f"coverage_{tgt}_japan.png",
            extent=JAPAN_EXTENT, point_size=6, alpha=0.5,
        )

    # Combined overlays (global + Japan).
    for extent, suffix, ps, al in [
        (None, "all", 2, 0.2),
        (JAPAN_EXTENT, "all_japan", 6, 0.5),
    ]:
        fig = plt.figure(figsize=(12, 6) if extent is None else (8, 7))
        ax, kw = _make_ax(fig, extent=extent, central_lon=GLOBAL_CENTRAL_LON)
        for tgt, meta in TARGETS.items():
            lon = ((all_lon[tgt] + 180) % 360) - 180
            lat = all_lat[tgt]
            if extent is not None:
                m = in_extent(all_lon[tgt], lat, extent)
                lon, lat = lon[m], lat[m]
                n = int(m.sum())
            else:
                n = summary[tgt][0]
            ax.scatter(lon, lat, s=ps, alpha=al, color=meta["color"],
                       edgecolor="none", zorder=2, label=f"{tgt} (n={n:,})", **kw)
        scope = "global" if extent is None else "near Japan"
        ax.set_title(f"BGC-Argo profile coverage ({scope}): O2 / Chla / NO3 (D/A mode)",
                     fontsize=11)
        ax.legend(loc="lower left", markerscale=4, fontsize=9)
        fig.savefig(OUT_DIR / f"coverage_{suffix}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"\nSaved maps -> {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
