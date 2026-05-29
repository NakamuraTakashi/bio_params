"""Download the GLODAPv2.2023 Merged Master File from NOAA NCEI.

Reference:
    https://www.ncei.noaa.gov/access/ocean-carbon-acidification-data-system/oceans/GLODAPv2_2023/
    DOI: https://doi.org/10.25921/zyrq-ht66

Usage:
    uv run python scripts/download_glodap.py
    uv run python scripts/download_glodap.py --force
    uv run python scripts/download_glodap.py --url <custom-url>
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

# Default URL: GLODAPv2.2023 global merged CSV on NCEI OCADS (~853 MB, uncompressed).
# Verify against the landing page above; override via --url if the path changes.
DEFAULT_URL = (
    "https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0283442/"
    "GLODAPv2.2023_Merged_Master_File.csv"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "glodap" / "raw"
EXPECTED_CSV_NAME = "GLODAPv2.2023_Merged_Master_File.csv"


_progress_state = {"last_pct": -1}


def _report_progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    mb = downloaded / (1024 * 1024)
    is_tty = sys.stderr.isatty()
    if total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        total_mb = total_size / (1024 * 1024)
        if is_tty:
            # Carriage-return progress bar; only meaningful on a real terminal.
            sys.stderr.write(f"\r  {pct:5.1f}%  ({mb:.1f} / {total_mb:.1f} MB)")
            sys.stderr.flush()
        else:
            # Non-TTY: emit one line per whole percent to avoid log spam.
            whole_pct = int(pct)
            if whole_pct > _progress_state["last_pct"]:
                _progress_state["last_pct"] = whole_pct
                sys.stderr.write(f"  {whole_pct:3d}%  ({mb:.1f} / {total_mb:.1f} MB)\n")
    else:
        if is_tty:
            sys.stderr.write(f"\r  {mb:.1f} MB")
            sys.stderr.flush()


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {url}")
    print(f"  -> {dest}")
    urllib.request.urlretrieve(url, dest, reporthook=_report_progress)
    sys.stderr.write("\n")
    return dest


def extract_zip(archive: Path, out_dir: Path) -> Path | None:
    """Extract archive and return the path to the expected CSV, if found."""
    print(f"Extracting: {archive.name}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(out_dir)
        names = zf.namelist()
    print(f"  extracted entries: {names}")
    for name in names:
        if Path(name).name == EXPECTED_CSV_NAME:
            return out_dir / name
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=DEFAULT_URL,
                        help="Download URL (default: NCEI v2.2023)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if the target CSV already exists")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    csv_path = out_dir / EXPECTED_CSV_NAME

    if csv_path.exists() and not args.force:
        size_mb = csv_path.stat().st_size / (1024 * 1024)
        print(f"Already present: {csv_path} ({size_mb:.1f} MB)")
        print("Use --force to re-download.")
        return 0

    archive_name = args.url.rsplit("/", 1)[-1]
    archive_path = out_dir / archive_name
    download(args.url, archive_path)

    if archive_path.suffix == ".zip":
        extracted = extract_zip(archive_path, out_dir)
        if extracted is None:
            print(f"ERROR: {EXPECTED_CSV_NAME} not found in archive.",
                  file=sys.stderr)
            return 1
        if extracted != csv_path:
            extracted.rename(csv_path)
    else:
        if archive_path != csv_path:
            archive_path.rename(csv_path)

    size_mb = csv_path.stat().st_size / (1024 * 1024)
    print(f"Ready: {csv_path} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
