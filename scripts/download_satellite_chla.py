"""Download the full monthly GlobColour surface Chl-a archive to local NetCDF.

Saves one file per year under data/satellite/raw/ (resumable: existing years are
skipped). After this, bio_params.satellite.open_chla() and the ROMS inference
read locally with no server access. Needs `copernicusmarine login`.

Monthly CHL, 4km global: ~149 MB/month uncompressed, ~1.8 GB/year, ~50 GB total
(1997-09..2026-04); the on-server transfer is smaller (compressed).

Usage:
    uv run python scripts/download_satellite_chla.py                 # 1997-2026
    uv run python scripts/download_satellite_chla.py --start-year 2003 --end-year 2005
"""
from __future__ import annotations

import argparse
from pathlib import Path

from bio_params.satellite import LOCAL_SAT_DIR, SAT_DATASET


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset-id", default=SAT_DATASET)
    p.add_argument("--start-year", type=int, default=1997)
    p.add_argument("--end-year", type=int, default=2026)
    p.add_argument("--out-dir", type=Path, default=LOCAL_SAT_DIR)
    p.add_argument("--variables", nargs="+", default=["CHL"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import copernicusmarine as cm
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for year in range(args.start_year, args.end_year + 1):
        fn = f"globcolour_chl_monthly_{year}.nc"
        out = args.out_dir / fn
        if out.exists():
            print(f"skip {fn} (exists)", flush=True)
            continue
        print(f"downloading {year} ({','.join(args.variables)}, global) ...", flush=True)
        cm.subset(
            dataset_id=args.dataset_id,
            variables=args.variables,
            start_datetime=f"{year}-01-01T00:00:00",
            end_datetime=f"{year}-12-31T23:59:59",
            output_directory=str(args.out_dir),
            output_filename=fn,
            overwrite=False,
        )
        sz = out.stat().st_size / 1e6 if out.exists() else 0
        print(f"  saved {fn}  ({sz:.0f} MB)", flush=True)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
