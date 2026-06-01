"""Does matching to DAILY satellite Chl-a (same date as the cast) agree better
with in-situ surface Chl-a than the MONTHLY mean does?

For a random sample of profiles per source, compares two satellite values
against the same in-situ surface Chl-a:
  - monthly: from the existing matchup parquet (GlobColour L4 monthly)
  - daily:   GlobColour L4 gap-free DAILY, sampled at the cast's exact date
Daily point extraction is lazy (zarr point reads), so it is sampled (~1 s/pt).

Output: figures/satellite_vs_insitu/daily_vs_monthly_surface.png (4 panels)

Usage:
    uv run python scripts/compare_daily_monthly_satellite.py --n 1000
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

_spec = importlib.util.spec_from_file_location(
    "_svi", Path(__file__).resolve().parent / "plot_satellite_vs_insitu_surface.py")
_svi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_svi)

OUT_DIR = _svi.OUT_DIR
DAILY_DATASET = "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D"


def extract_daily(times, lats, lons):
    import copernicusmarine as cm
    ds = cm.open_dataset(dataset_id=DAILY_DATASET)
    lons = ((np.asarray(lons) + 180) % 360) - 180
    sel = ds["CHL"].sel(
        time=xr.DataArray(pd.to_datetime(times), dims="p"),
        latitude=xr.DataArray(np.asarray(lats), dims="p"),
        longitude=xr.DataArray(lons, dims="p"), method="nearest")
    vals = np.asarray(sel.values)
    ds.close()
    return vals


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sources", nargs="+", default=["bgc_argo", "glodap"])
    p.add_argument("--n", type=int, default=1000, help="profiles sampled per source")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    results = {}
    for src in args.sources:
        j = _svi.load_source(src)  # insitu + monthly satellite, surface, >0
        if len(j) > args.n:
            j = j.iloc[np.sort(rng.choice(len(j), args.n, replace=False))].reset_index(drop=True)
        print(f"{src}: extracting daily satellite for {len(j):,} profiles ...", flush=True)
        daily = extract_daily(j["time"].to_numpy(), j["latitude"].to_numpy(),
                              j["longitude"].to_numpy())
        j = j.assign(daily=daily)
        j = j[np.isfinite(j["daily"]) & (j["daily"] > 0)].reset_index(drop=True)
        sm = _svi.stats(j["satellite"].to_numpy(), j["insitu"].to_numpy())
        sd = _svi.stats(j["daily"].to_numpy(), j["insitu"].to_numpy())
        results[src] = (j, sm, sd)
        print(f"  monthly: n={sm['n']:,} log-R2={sm['r2_log']:.3f} "
              f"RMSE(log10)={sm['rmse_log']:.3f} med(in/sat)={sm['med_ratio']:.2f}")
        print(f"  daily:   n={sd['n']:,} log-R2={sd['r2_log']:.3f} "
              f"RMSE(log10)={sd['rmse_log']:.3f} med(in/sat)={sd['med_ratio']:.2f}")

    fig, axes = plt.subplots(len(args.sources), 2,
                             figsize=(13, 6.5 * len(args.sources)))
    if len(args.sources) == 1:
        axes = axes[None, :]
    for row, src in enumerate(args.sources):
        j, sm, sd = results[src]
        _svi.scatter(axes[row, 0], j["satellite"].to_numpy(), j["insitu"].to_numpy(),
                     f"{src} — MONTHLY satellite\n"
                     f"n={sm['n']:,} log-R2={sm['r2_log']:.3f} "
                     f"RMSE(log10)={sm['rmse_log']:.3f} med(in/sat)={sm['med_ratio']:.2f}")
        _svi.scatter(axes[row, 1], j["daily"].to_numpy(), j["insitu"].to_numpy(),
                     f"{src} — DAILY satellite (same date)\n"
                     f"n={sd['n']:,} log-R2={sd['r2_log']:.3f} "
                     f"RMSE(log10)={sd['rmse_log']:.3f} med(in/sat)={sd['med_ratio']:.2f}")
        for c in (0, 1):
            axes[row, c].set_xlabel("Satellite surface Chl-a (mg/m3)")
    fig.suptitle("In-situ surface Chl-a vs satellite: monthly mean vs same-date daily "
                 f"(sample n={args.n}/source)", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "daily_vs_monthly_surface.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
