"""Infer biogeochemical fields on the FORA-JPN60 reanalysis grid (T/S via OPeNDAP)
and map them at fixed depths. Step toward ROMS biology init/boundary conditions
from FORA-JPN60.

For each target a plain MLP (base features lat/lon/depth/T/S) predicts the value
at the NEAREST FORA z-level to each requested depth (surface, 200, 500, 1000,
3000 m). Low-salinity coastal points are blended with the salinity mixing-line
regression for NO3/PO4/SiO4/TA (the validated set). O2/NO3 use the combined
(GLODAP+BGC-Argo) model; the rest use GLODAP models.

Usage:
    uv run python scripts/predict_fora_biology.py --date 2020-06-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr

from bio_params.features import build_features
from bio_params.persist import load_artifact

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from predict_roms_ini_depths import (                      # noqa: E402
    TRACER_META, SALINITY_REGRESSION, SALT_OPEN_OCEAN_MIN,
    blend_low_salinity, predict_field)

MODEL_DIR = ROOT / "models" / "pretrained"
OUT_DIR = ROOT / "figures" / "fora"
FORA_BASE = "https://www.jamstec.go.jp/jagdas/dodsC/fora/JPN/Daily-mean/Basic-3D"
TARGET_DEPTHS = [0.0, 200.0, 500.0, 1000.0, 3000.0]   # 0 -> surface (shallowest level)
# target -> model source (combined for O2/NO3; GLODAP otherwise)
SOURCE = {"O2": "combined", "NO3": "combined", "TA": "glodap", "DIC": "glodap",
          "SiO4": "glodap", "PO4": "glodap", "C13": "glodap", "C14": "glodap"}
DEFAULT_TARGETS = ["TA", "DIC", "O2", "C13", "C14", "SiO4", "PO4", "NO3"]


def fora_url(kind: str, date) -> str:
    d = pd.Timestamp(date)
    return f"{FORA_BASE}/{d.year}/nc_{kind}.{d:%Y%m%d}"


def load_ts(date, stride):
    """Load FORA T/S at the nearest z-levels to TARGET_DEPTHS (strided)."""
    dt = xr.open_dataset(fora_url("t", date)); dsal = xr.open_dataset(fora_url("s", date))
    zc = dt["depth"].values
    idx = [0] + [int(np.argmin(np.abs(zc - z))) for z in TARGET_DEPTHS[1:]]
    actual = zc[idx]
    sl = dict(lat=slice(None, None, stride), lon=slice(None, None, stride))
    T = np.asarray(dt["thetao"].isel(time=0, depth=idx, **sl).load().values)   # (5,J,I)
    S = np.asarray(dsal["so"].isel(time=0, depth=idx, **sl).load().values)
    lon = dt["lon"].values[::stride]; lat = dt["lat"].values[::stride]
    dt.close(); dsal.close()
    return T, S, lon, lat, actual


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--date", default="2020-06-01")
    p.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--no-low-sal", action="store_true",
                   help="disable the low-salinity regression blend")
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    print(f"FORA-JPN60 {args.date}: loading T/S (stride={args.stride}) ...", flush=True)
    T, S, lon, lat, zlev = load_ts(args.date, args.stride)
    LON, LAT = np.meshgrid(lon, lat)
    print(f"  grid {T.shape[1:]}  z-levels {[f'{z:.0f}' for z in zlev]} m", flush=True)

    for tgt in args.targets:
        src = SOURCE.get(tgt, "glodap")
        art = MODEL_DIR / f"{src}_{tgt}.pt"
        if not art.exists():
            print(f"skip {tgt}: missing {art.name}"); continue
        model, norm, meta = load_artifact(art, map_location=device); model.to(device)
        log_target = bool(meta["extra"].get("log_target", False))
        m = TRACER_META[tgt]
        do_reg = (not args.no_low_sal) and tgt in SALINITY_REGRESSION
        print(f"{tgt}: {src} model  low-sal={do_reg}", flush=True)

        fig, axes = plt.subplots(1, len(TARGET_DEPTHS), figsize=(4.6 * len(TARGET_DEPTHS), 5.2),
                                 subplot_kw={"projection": ccrs.PlateCarree()})
        for ax, k in zip(axes, range(len(TARGET_DEPTHS))):
            Tk, Sk = T[k], S[k]
            field = np.full(Tk.shape, np.nan)
            ok = np.isfinite(Tk) & np.isfinite(Sk)
            if ok.any():
                df = pd.DataFrame({"latitude": LAT[ok], "longitude": LON[ok],
                                   "depth": np.full(int(ok.sum()), zlev[k]),
                                   "temperature": Tk[ok], "salinity": Sk[ok]})
                X = build_features(df).to_numpy()
                pred = predict_field(model, norm, X, device, log_target=log_target,
                                     clip=m.get("clip"))
                if do_reg:
                    pred, _ = blend_low_salinity(pred, Sk[ok], tgt)
                field[ok] = pred
            fin = field[np.isfinite(field)]
            vmin, vmax = (np.percentile(fin, [2, 98]) if fin.size else (None, None))
            pcm = ax.pcolormesh(lon, lat, np.ma.masked_invalid(field), cmap=m["cmap"],
                                vmin=vmin, vmax=vmax, shading="auto", transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=2); ax.coastlines(lw=0.3, zorder=3)
            low = ok & (Sk < SALT_OPEN_OCEAN_MIN)
            if low.any():
                ax.contourf(lon, lat, low.astype(float), levels=[0.5, 1.5], colors="none",
                            hatches=["//" if do_reg else "xx"], transform=ccrs.PlateCarree(), zorder=4)
            lbl = "surface" if k == 0 else f"{TARGET_DEPTHS[k]:.0f} m"
            ax.set_title(f"{lbl}  (FORA z={zlev[k]:.0f} m)", fontsize=9)
            plt.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.04, fraction=0.05).set_label(
                f"{tgt} ({m['unit']})", fontsize=8)
        fig.suptitle(f"FORA-JPN60 {args.date}: {m['long']} ({tgt}) — {src} MLP"
                     + ("  [low-sal regression //]" if do_reg else "")
                     + f"  (hatch S<{SALT_OPEN_OCEAN_MIN:.0f})", fontsize=12)
        fig.tight_layout()
        out = OUT_DIR / f"fora_pred_{tgt}_{args.date.replace('-', '')}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
        print(f"  saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
