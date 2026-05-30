"""Download BGC-Argo per-float synthetic profile files (_Sprof.nc) from the GDAC.

Strategy (lightweight, avoids argopy's slow region fetch):
  1. Read the synthetic-profile index (one row per profile, with lat/lon/date,
     the parameters carried, and each parameter's data mode R/A/D).
  2. Filter to a lon/lat box and to floats carrying a target parameter.
  3. Optionally rank floats by how many profiles have the target in
     delayed mode (D), so a small --limit picks the most useful floats.
  4. Download <wmo>_Sprof.nc for each selected float (skips existing files).

NETWORK: the GDAC is reachable only with the Bash sandbox disabled in this
environment; run this script with that in mind.

Usage:
    uv run python scripts/download_bgc_argo.py --target CHLA --limit 15 --prefer-delayed
    uv run python scripts/download_bgc_argo.py --target DOXY            # all in-box DOXY floats
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "bgc_argo" / "raw"
DEFAULT_INDEX = RAW_DIR / "argo_synthetic-profile_index.txt.gz"
FLOAT_DIR = RAW_DIR / "floats"
GDAC_BASE = "https://data-argo.ifremer.fr/dac"

# Kuroshio-covering box (lon 120-180E, lat 10-50N) for ROMS open-ocean BCs.
DEFAULT_BOX = (120.0, 180.0, 10.0, 50.0)  # lon0, lon1, lat0, lat1


def load_index(index_path: Path) -> pd.DataFrame:
    df = pd.read_csv(index_path, compression="gzip", comment="#", low_memory=False)
    df["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["lon"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["dac"] = df["file"].str.split("/").str[0]
    df["wmo"] = df["file"].str.split("/").str[1]
    return df


def target_mode(parameters: str, modes: str, target: str) -> str:
    """Data mode (R/A/D) of `target` in one profile, or '' if absent.

    `parameters` is a space-separated parameter list; `modes` is a string with
    one mode char per parameter in the same order.
    """
    if not isinstance(parameters, str) or not isinstance(modes, str):
        return ""
    plist = parameters.split()
    if target in plist:
        i = plist.index(target)
        if i < len(modes):
            return modes[i]
    return ""


def select_floats(
    df: pd.DataFrame,
    box: tuple[float, float, float, float],
    target: str,
    limit: int | None,
    prefer_delayed: bool,
) -> list[tuple[str, str]]:
    lon0, lon1, lat0, lat1 = box
    m = (df.lat >= lat0) & (df.lat <= lat1) & (df.lon >= lon0) & (df.lon <= lon1)
    sub = df[m & df["parameters"].fillna("").str.contains(target)].copy()
    sub["tmode"] = [
        target_mode(p, mo, target)
        for p, mo in zip(sub["parameters"], sub["parameter_data_mode"])
    ]
    # Per-float profile counts (total and delayed) for the target.
    grp = sub.groupby(["dac", "wmo"])
    counts = grp.size().rename("n_prof").to_frame()
    counts["n_delayed"] = grp.apply(
        lambda g: int((g["tmode"] == "D").sum()), include_groups=False
    )
    sort_col = "n_delayed" if prefer_delayed else "n_prof"
    counts = counts.sort_values([sort_col, "n_prof"], ascending=False)
    if limit:
        counts = counts.head(limit)
    return list(counts.index)


def download_float(dac: str, wmo: str, out_dir: Path, timeout: int = 180) -> str:
    out = out_dir / f"{wmo}_Sprof.nc"
    if out.exists() and out.stat().st_size > 0:
        return "skip"
    url = f"{GDAC_BASE}/{dac}/{wmo}/{wmo}_Sprof.nc"
    tmp = out.with_suffix(".nc.part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(out)
        return "ok"
    except Exception as e:  # noqa: BLE001 - report and continue
        if tmp.exists():
            tmp.unlink()
        print(f"    FAILED {wmo}: {e}", file=sys.stderr)
        return "fail"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target", default="CHLA",
                   choices=["DOXY", "NITRATE", "CHLA"],
                   help="Download floats carrying this BGC parameter")
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--out-dir", type=Path, default=FLOAT_DIR)
    p.add_argument("--box", type=float, nargs=4, default=list(DEFAULT_BOX),
                   metavar=("LON0", "LON1", "LAT0", "LAT1"))
    p.add_argument("--limit", type=int, default=None,
                   help="Max number of floats (for a pilot run)")
    p.add_argument("--prefer-delayed", action="store_true",
                   help="Rank floats by delayed-mode profile count for the target")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.index.exists():
        print(f"ERROR: index not found at {args.index}")
        print("Download it first:\n  curl -o "
              f"{args.index} https://data-argo.ifremer.fr/argo_synthetic-profile_index.txt.gz")
        return 1

    print(f"Reading index {args.index} ...")
    df = load_index(args.index)
    floats = select_floats(df, tuple(args.box), args.target, args.limit,
                           args.prefer_delayed)
    print(f"Selected {len(floats)} floats carrying {args.target} "
          f"in box {args.box}"
          + (f" (top {args.limit} by delayed-mode count)" if args.limit else ""))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tally = {"ok": 0, "skip": 0, "fail": 0}
    for i, (dac, wmo) in enumerate(floats, 1):
        status = download_float(dac, wmo, args.out_dir)
        tally[status] += 1
        print(f"  [{i}/{len(floats)}] {dac}/{wmo}: {status}")
    print(f"\nDone: {tally}")
    print(f"Files in {args.out_dir}: "
          f"{len(list(args.out_dir.glob('*_Sprof.nc')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
