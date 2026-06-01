"""Match each BGC-Argo profile to a satellite surface chlorophyll value.

For the SOCA-style approach (Sauzede et al. 2016): the satellite surface Chl-a
is the "truth" that anchors the amplitude of the profile, while T/S/depth carry
the vertical shape. This script extracts, for every BGC-Argo profile
(identified by its unique latitude/longitude/time), the surface Chl-a from the
CMEMS GlobColour L4 monthly product at the profile's month and position, and
caches the result so training can attach it as a feature.

Product: cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M
  - variable CHL (mg/m3, same unit as our Chla target)
  - 4 km global, monthly, multi-sensor merged, 1997-09 .. 2026-04 (MY reprocessing)
  - L4 monthly is NOT gap-filled, so cloud-covered pixels are NaN; profiles
    under persistent cloud get no match and are dropped (faithful to real obs).

Output: data/bgc_argo/processed/satchl_matchup.parquet
  columns: latitude, longitude, time, surface_chla
  one row per unique profile (rows with a masked satellite pixel are dropped).

Resumable: per-month results are checkpointed under
data/bgc_argo/processed/_satchl_months/<YYYY-MM>.parquet and skipped on rerun.

Usage (needs `copernicusmarine login` first):
    uv run python scripts/match_satellite_chla.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bio_params.loaders.bgc_argo import load_bgc_argo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPROF = PROJECT_ROOT / "data" / "bgc_argo" / "raw" / "floats"
DEFAULT_CSV = PROJECT_ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
PROC_DIR = PROJECT_ROOT / "data" / "bgc_argo" / "processed"
DEFAULT_DATASET = "cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M"
MY_END = "2026-04"  # last month available in the MY (reprocessed) monthly product
# Satellite MY product starts 1997-09; earlier profiles cannot be matched.
SAT_START = "1997-09"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target", default="Chla")
    p.add_argument("--source", default="bgc_argo", choices=["bgc_argo", "glodap"],
                   help="Which profiles to match (writes a per-source parquet)")
    p.add_argument("--sprof-dir", type=Path, default=DEFAULT_SPROF)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--out", type=Path, default=None,
                   help="Output parquet (default: per-source name under processed/)")
    p.add_argument("--dataset-id", default=DEFAULT_DATASET)
    p.add_argument("--my-end", default=MY_END,
                   help="Drop profiles after this YYYY-MM (satellite MY end)")
    return p.parse_args()


def _load_profiles(args):
    """Load unique (latitude, longitude, time) profiles for the chosen source."""
    if args.source == "bgc_argo":
        from bio_params.loaders.bgc_argo import load_bgc_argo
        df = load_bgc_argo(args.sprof_dir, args.target)
    else:
        from bio_params.loaders.glodap import load_glodap
        df = load_glodap(args.csv, args.target, with_time=True)
    prof = (df[["latitude", "longitude", "time"]]
            .drop_duplicates().reset_index(drop=True))
    prof["time"] = pd.to_datetime(prof["time"])
    return prof[prof["time"].notna()].reset_index(drop=True)


def _nearest_index(coord: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Nearest grid index for `coord` on a uniform (monotone) `axis`."""
    origin = float(axis[0])
    step = float(axis[1] - axis[0])
    idx = np.rint((coord - origin) / step).astype(np.int64)
    return np.clip(idx, 0, len(axis) - 1)


def _open(cm, dataset_id, tries: int = 5):
    """Open the dataset, retrying transient catalog/network failures."""
    for attempt in range(1, tries + 1):
        try:
            return cm.open_dataset(dataset_id=dataset_id)
        except Exception as e:  # noqa: BLE001 - retry any transient open error
            if attempt == tries:
                raise
            print(f"  open attempt {attempt} failed ({type(e).__name__}); "
                  f"retrying ...", flush=True)
            time.sleep(5 * attempt)


def main() -> int:
    args = parse_args()
    out = args.out or (PROC_DIR / (
        "satchl_matchup.parquet" if args.source == "bgc_argo"
        else f"satchl_matchup_{args.source}.parquet"))
    import copernicusmarine as cm

    print(f"Loading {args.source} {args.target} profiles ...", flush=True)
    prof = _load_profiles(args)
    n_all = len(prof)

    start = pd.Period(SAT_START, freq="M")
    end = pd.Period(args.my_end, freq="M")
    prof["ym"] = prof["time"].dt.to_period("M")
    in_range = (prof["ym"] >= start) & (prof["ym"] <= end)
    n_out = int((~in_range).sum())
    prof = prof[in_range].reset_index(drop=True)
    print(f"  {n_all:,} unique profiles; {n_out:,} outside satellite range "
          f"({SAT_START}..{args.my_end}) dropped; {len(prof):,} to match", flush=True)

    ckpt_dir = out.parent / ("_satchl_months" if args.source == "bgc_argo"
                             else f"_satchl_months_{args.source}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ds = _open(cm, args.dataset_id)
    lat_axis = ds["latitude"].values
    lon_axis = ds["longitude"].values

    months = sorted(prof["ym"].unique())
    print(f"Matching {len(months)} months from {args.dataset_id} ...", flush=True)
    for k, ym in enumerate(months, 1):
        ckpt = ckpt_dir / f"{ym}.parquet"
        if ckpt.exists():
            continue
        grp = prof[prof["ym"] == ym]
        stamp = f"{ym}-01"
        for attempt in range(1, 6):  # retry transient per-month read failures
            try:
                arr = np.asarray(
                    ds["CHL"].sel(time=stamp, method="nearest").load().values)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 5:
                    raise
                print(f"  {ym} read attempt {attempt} failed "
                      f"({type(e).__name__}); reopening ...", flush=True)
                time.sleep(5 * attempt)
                ds = _open(cm, args.dataset_id)
        ilat = _nearest_index(grp["latitude"].to_numpy(), lat_axis)
        ilon = _nearest_index(grp["longitude"].to_numpy(), lon_axis)
        sat = arr[ilat, ilon]
        out = grp[["latitude", "longitude", "time"]].copy()
        out["surface_chla"] = sat
        out.to_parquet(ckpt, index=False)
        n_ok = int(np.isfinite(sat).sum())
        print(f"  [{k}/{len(months)}] {ym}: {len(grp):,} profiles, "
              f"{n_ok:,} valid pixels", flush=True)
    ds.close()

    parts = [pd.read_parquet(p) for p in sorted(ckpt_dir.glob("*.parquet"))]
    full = pd.concat(parts, ignore_index=True)
    before = len(full)
    full = full[np.isfinite(full["surface_chla"])].reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out, index=False)
    print(f"Saved {out}: {len(full):,} matched profiles "
          f"({before - len(full):,} dropped for masked pixel)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
