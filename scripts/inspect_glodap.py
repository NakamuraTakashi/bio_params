"""Inspect GLODAPv2.2023 CSV and report per-target valid-row counts.

For each candidate target variable (DIC, TA, NO3, PO4, SiO4, O2, plus non-core
DOC and Chl-a), report:
  - rows where the value is non-null AND its WOCE flag == 2 (good measurement)
  - value range over the valid rows
  - coordinate coverage (lat/lon/depth bounds) over the valid rows

The result decides which targets are practical to train on. Output is printed
as a table and saved as JSON to data/glodap/processed/inspection_summary.json.

Usage:
    uv run python scripts/inspect_glodap.py
    uv run python scripts/inspect_glodap.py --csv <path>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "glodap" / "processed" / "inspection_summary.json"

# (target_column, flag_column, human_label)
# Non-core variables (DOC, Chl-a) have no WOCE flag column.
TARGETS: list[tuple[str, str | None, str]] = [
    ("G2tco2",       "G2tco2f",       "DIC"),
    ("G2talk",       "G2talkf",       "TA"),
    ("G2nitrate",    "G2nitratef",    "NO3"),
    ("G2phosphate",  "G2phosphatef",  "PO4"),
    ("G2silicate",   "G2silicatef",   "SiO4"),
    ("G2oxygen",     "G2oxygenf",     "O2"),
    ("G2doc",        None,            "DOC (non-core)"),
    ("G2chla",       None,            "Chl-a (non-core)"),
]

COORD_COLS = ["G2latitude", "G2longitude", "G2depth", "G2temperature", "G2salinity"]

# GLODAP encodes missing values as -9999 across all numeric columns
# (confirmed against GLODAPv2.2023 Merged Master File CSV).
MISSING_SENTINEL = -9999


def load_glodap(path: Path) -> pd.DataFrame:
    # Read the header alone so we can request only columns that actually exist.
    # This avoids pandas warnings on absent non-core columns (e.g. G2chla).
    header = pd.read_csv(path, nrows=0).columns
    needed = set(COORD_COLS)
    for target, flag, _ in TARGETS:
        needed.add(target)
        if flag is not None:
            needed.add(flag)
    usecols = [c for c in header if c in needed]

    missing_required = set(COORD_COLS) - set(usecols)
    if missing_required:
        print(f"WARNING: missing required columns: {sorted(missing_required)}",
              file=sys.stderr)

    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    return df.replace(MISSING_SENTINEL, np.nan)


def summarize_target(df: pd.DataFrame, target: str, flag: str | None) -> dict:
    if target not in df.columns:
        return {"present": False, "valid_count": 0, "total_non_null": 0}

    target_series = df[target]
    if flag is not None and flag in df.columns:
        valid_mask = target_series.notna() & (df[flag] == 2)
        flag_used: str | None = flag
    else:
        # Non-core variables: no WOCE flag, fall back to non-null count.
        valid_mask = target_series.notna()
        flag_used = None
    valid = df.loc[valid_mask]

    def safe_pct(series: pd.Series, q: float) -> float | None:
        return float(np.nanpercentile(series, q)) if len(series) else None

    summary = {
        "present": True,
        "flag_column": flag_used,
        "valid_count": int(valid_mask.sum()),
        "total_non_null": int(target_series.notna().sum()),
        "value_min": safe_pct(valid[target], 0),
        "value_p50": safe_pct(valid[target], 50),
        "value_max": safe_pct(valid[target], 100),
    }
    for c in ("G2latitude", "G2longitude", "G2depth"):
        if c in df.columns and len(valid):
            summary[f"{c}_min"] = safe_pct(valid[c], 0)
            summary[f"{c}_max"] = safe_pct(valid[c], 100)
    return summary


def print_table(rows: list[dict]) -> None:
    headers = ["Target", "Column", "Flag", "Valid", "Non-null", "Min", "Median", "Max"]
    widths  = [16,       14,       14,     12,      12,         10,    10,       10]
    sep = "  "
    print(sep.join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-" * (sum(widths) + len(sep) * (len(widths) - 1)))
    for r in rows:
        cells = [
            r["label"],
            r["target"],
            r["flag"] or "-",
            f"{r['valid_count']:,}",
            f"{r['total_non_null']:,}",
            f"{r['value_min']:.3g}" if r.get("value_min") is not None else "-",
            f"{r['value_p50']:.3g}" if r.get("value_p50") is not None else "-",
            f"{r['value_max']:.3g}" if r.get("value_max") is not None else "-",
        ]
        print(sep.join(c[:w].ljust(w) for c, w in zip(cells, widths)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help=f"Path to GLODAP CSV (default: {DEFAULT_CSV})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"Output JSON path (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: CSV not found at {args.csv}", file=sys.stderr)
        print("Run scripts/download_glodap.py first.", file=sys.stderr)
        return 1

    print(f"Loading: {args.csv}")
    df = load_glodap(args.csv)
    print(f"  total rows: {len(df):,}\n")

    rows: list[dict] = []
    for target, flag, label in TARGETS:
        s = summarize_target(df, target, flag)
        s.update({"target": target, "flag": flag, "label": label})
        rows.append(s)
    print_table(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_csv": str(args.csv),
        "total_rows": int(len(df)),
        "targets": rows,
    }
    with args.out.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved summary: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
