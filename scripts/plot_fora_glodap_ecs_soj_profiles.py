"""Same per-site profiles as plot_fora_glodap_21site_profiles.py, for 6 hand-picked
sites where everything EXCEPT DOC is present (7-var: DIC/TA/O2/NO3/PO4/SiO4/Chl-a):
  East China Sea shelf near the Changjiang mouth (cruise 217, 2008): 217_29 / 217_20 / 217_18
  Sea of Japan deep basin (>=1000 m, cruise 2068, 2018):            2068_5498 / 2068_5490 / 2068_5487
DOC obs are absent here, so the DOC panel shows the model line only. Two figures
per site (full depth + 0-300 m) -> figures/fora/sites_ecs_soj/.

Usage:
    uv run python scripts/plot_fora_glodap_ecs_soj_profiles.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from plot_fora_glodap_21site_profiles import (   # noqa: E402
    extract_fora, plot_site_set, meta_sub_from_keys, PROC)

OUT_DIR = ROOT / "figures" / "fora" / "sites_ecs_soj"
CACHE = PROC / "fora_ecs_soj_profiles.nc"
SITES = [
    (217, 29.0), (217, 20.0), (217, 18.0),          # East China Sea shelf (Changjiang)
    (2068, 5498.0), (2068, 5490.0), (2068, 5487.0),  # Sea of Japan deep basin
]


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta, sub = meta_sub_from_keys(SITES)
    meta = meta.set_index("sid").loc[[f"{c}_{s}" for c, s in SITES]].reset_index()
    print(f"ECS/SoJ sites: {len(meta)}")
    for _, r in meta.iterrows():
        print(f"  {r.sid}  {r.lat:.2f}N {r.lon:.2f}E  {r.date:%Y-%m-%d}")
    prof = extract_fora(meta, CACHE)
    plot_site_set(meta, sub, prof, OUT_DIR, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
