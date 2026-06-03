"""Mirror a CMEMS GlobColour daily dataset to external storage, preserving the
native folder/file structure, and keep it up to date incrementally.

Uses `copernicusmarine get --sync`, which downloads only files that are missing
or newer on the server (compared by last-modified time and size). The same
command serves both the initial bulk download and later incremental updates, so
this script is safe to run from cron. A lock file prevents overlapping runs.

Default target is the long gapfree daily observation series
(`cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D`, ~933 GB,
1997-present), mirrored under /mnt/w/share/Copernicus-GlobColour/gapfree_daily.

Requires `copernicusmarine login` to have been run once (credentials cached).

Examples:
    # initial download + incremental update (same command):
    uv run python scripts/mirror_satellite_daily.py
    # preview only:
    uv run python scripts/mirror_satellite_daily.py --dry-run
    # also delete local files removed on the server (true mirror):
    uv run python scripts/mirror_satellite_daily.py --sync-delete

Cron (weekly, Mondays 03:00; ensure cron/systemd is running under WSL):
    0 3 * * 1 cd /home/nakamulab2/bio_params && \
      /home/nakamulab2/.local/bin/uv run python scripts/mirror_satellite_daily.py \
      >> /mnt/w/share/Copernicus-GlobColour/mirror.log 2>&1
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import subprocess
import sys
from pathlib import Path

DEFAULT_DATASET = "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D"
DEFAULT_OUTDIR = Path("/mnt/w/share/Copernicus-GlobColour/gapfree_daily")
# --sync requires an explicit dataset version. CMEMS publishes a new version as
# a full re-release in a new folder, so pin it here and bump deliberately (a new
# version means re-downloading the whole ~933 GB series), not automatically.
DEFAULT_VERSION = "202603"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset-id", default=DEFAULT_DATASET)
    p.add_argument("--dataset-version", default=DEFAULT_VERSION,
                   help="CMEMS dataset version (required by --sync)")
    p.add_argument("--output-directory", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--sync-delete", action="store_true",
                   help="also remove local files no longer present on the server")
    p.add_argument("--dry-run", action="store_true",
                   help="list what would be downloaded; download nothing")
    args = p.parse_args()

    args.output_directory.mkdir(parents=True, exist_ok=True)

    # single-instance lock so overlapping cron runs do not collide
    lock_path = args.output_directory / ".mirror.lock"
    lock_fh = open(lock_path, "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"[{dt.datetime.now():%F %T}] another mirror run is in progress "
              f"({lock_path}); exiting.", flush=True)
        return 0

    cmd = [
        "copernicusmarine", "get",
        "--dataset-id", args.dataset_id,
        "--dataset-version", args.dataset_version,
        "--output-directory", str(args.output_directory),
        "--sync",
    ]
    if args.sync_delete:
        cmd.append("--sync-delete")
    if args.dry_run:
        cmd.append("--dry-run")

    print(f"[{dt.datetime.now():%F %T}] start: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    print(f"[{dt.datetime.now():%F %T}] done (exit {rc})", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
