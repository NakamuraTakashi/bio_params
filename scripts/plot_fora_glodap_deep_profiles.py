"""Same per-site observed-vs-model NO3 & Chl-a vertical profiles as
plot_fora_glodap_site_profiles.py, but with sites re-selected (except the East
China Sea) to the GLODAP stations with the DEEPEST NO3 sampling in each region
(NO3 to ~6000 m). These stations have an NO3 profile + a shallow Chl-a profile but
are not "full-set", so the full per-depth observations are reloaded from the
GLODAP CSV and their FORA-JPN60 T/S columns are extracted on the fly (and cached).

Usage:
    uv run python scripts/plot_fora_glodap_deep_profiles.py [--full-depth]
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
from predict_roms_ini_depths import TRACER_META, blend_low_salinity, predict_field  # noqa: E402
from predict_fora_chla import fora_url                                              # noqa: E402
from plot_fora_glodap_site_profiles import struct_for_column                        # noqa: E402

CSV = ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
PROC = ROOT / "data" / "glodap" / "processed"
MODEL_DIR = ROOT / "models" / "pretrained"
OUT_DIR = ROOT / "figures" / "fora"
CACHE = PROC / "fora_deep_sites_profiles.nc"
SITES = [
    ("Pacific low-lat", 4066, 5703.0),
    ("Pacific mid-lat", 2065, 5332.0),
    ("Pacific high-lat", 2065, 5351.0),
    ("Sea of Japan", 4067, 5719.0),
    ("East China Sea", 217, 14.0),
]


def load_obs():
    cols = ["G2cruise", "G2station", "G2latitude", "G2longitude", "G2depth",
            "G2year", "G2month", "G2day", "G2temperature", "G2salinity",
            "G2nitrate", "G2nitratef", "G2chla"]
    df = pd.read_csv(CSV, usecols=cols).replace(-9999, np.nan)
    keys = {(c, s) for _, c, s in SITES}
    df = df[[(c, s) in keys for c, s in zip(df.G2cruise, df.G2station)]].copy()
    df["date"] = pd.to_datetime(dict(year=df.G2year, month=df.G2month, day=df.G2day)).dt.normalize()
    return df


def extract_fora(meta):
    """Extract FORA T/S columns (nearest grid, same date) for the chosen stations."""
    if CACHE.exists():
        return xr.open_dataset(CACHE)
    ref = xr.open_dataset(fora_url("t", meta.date.iloc[0]))
    flat = ref["lat"].values; flon = ref["lon"].values; fdepth = ref["depth"].values
    ref.close()
    nlev = len(fdepth)
    Tprof = np.full((len(meta), nlev), np.nan); Sprof = np.full((len(meta), nlev), np.nan)
    for k, (_, r) in enumerate(meta.iterrows()):
        dt = xr.open_dataset(fora_url("t", r.date)); dsal = xr.open_dataset(fora_url("s", r.date))
        j = int(np.argmin(np.abs(flat - r.lat))); i = int(np.argmin(np.abs(flon - r.lon)))
        Tprof[k] = np.asarray(dt["thetao"].isel(time=0, lat=j, lon=i).load().values)
        Sprof[k] = np.asarray(dsal["so"].isel(time=0, lat=j, lon=i).load().values)
        dt.close(); dsal.close()
        print(f"  FORA column {r.sid} ({r.lat:.1f}N {r.lon:.1f}E {r.date:%Y-%m-%d})", flush=True)
    out = xr.Dataset({"fora_temp": (("station", "level"), Tprof),
                      "fora_salt": (("station", "level"), Sprof)},
                     coords={"station": meta.sid.values, "level": np.arange(nlev),
                             "fora_depth": ("level", fdepth)})
    out.to_netcdf(CACHE)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full-depth", action="store_true")
    ap.add_argument("--ymax", type=float, default=None,
                    help="cap the depth axis at this depth (m), e.g. --ymax 300 to "
                         "zoom into the upper ocean")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_obs()
    df["sid"] = df.G2cruise.astype(int).astype(str) + "_" + df.G2station.astype(str)
    meta = (df.sort_values("G2depth").drop_duplicates("sid")
            .set_index("sid").loc[[f"{c}_{s}" for _, c, s in SITES]]
            .reset_index()[["sid", "G2latitude", "G2longitude", "date"]]
            .rename(columns={"G2latitude": "lat", "G2longitude": "lon"}))
    prof = extract_fora(meta)
    fd = prof["fora_depth"].values
    sid_order = prof["station"].values.tolist()

    no3_model, no3_norm, _ = load_artifact(MODEL_DIR / "combined_NO3.pt", map_location=device)
    no3_model.to(device)
    chla_model, chla_norm, cmeta = load_artifact(MODEL_DIR / "combined_Chla_allfeat.pt",
                                                 map_location=device); chla_model.to(device)
    cutoff = float(cmeta["extra"]["cutoff_depth"])

    fig, axes = plt.subplots(len(SITES), 2, figsize=(9, 4.0 * len(SITES)))
    for row, (label, cruise, station) in enumerate(SITES):
        sid = f"{cruise}_{station}"
        k = sid_order.index(sid)
        r = meta[meta.sid == sid].iloc[0]
        sdf = df[df.sid == sid]
        obs_no3 = sdf[(sdf.G2nitratef == 2) & sdf.G2nitrate.notna()].sort_values("G2depth")
        obs_chla = sdf[sdf.G2chla.notna()].sort_values("G2depth")
        Tcol, Scol = prof["fora_temp"].values[k], prof["fora_salt"].values[k]
        fin = np.isfinite(Tcol) & np.isfinite(Scol)
        seafloor = float(fd[fin].max()) if fin.any() else float(obs_no3.G2depth.max())
        obs_dmax = float(max(obs_no3.G2depth.max(), obs_chla.G2depth.max()))
        if args.ymax is not None:
            ymax = args.ymax
        elif args.full_depth:
            ymax = seafloor
        else:
            ymax = obs_dmax + 40
        sel = fin & (fd <= ymax + 1) if not args.full_depth else fin
        dz, Tz, Sz = fd[sel], Tcol[sel], Scol[sel]

        dfn = pd.DataFrame({"latitude": r.lat, "longitude": r.lon, "depth": dz,
                            "temperature": Tz, "salinity": Sz})
        no3m = predict_field(no3_model, no3_norm, build_features(dfn).to_numpy(),
                             device, clip=(0.0, 60.0))
        no3m, _ = blend_low_salinity(no3m, Sz, "NO3")
        mld, z_pyc, strat_max, z_nutr, nutr_max = struct_for_column(
            fd, Tcol, Scol, r.lat, r.lon, no3_model, no3_norm, device)
        dfc = pd.DataFrame({"latitude": r.lat, "longitude": r.lon, "depth": dz,
                            "temperature": Tz, "salinity": Sz, "mld": mld, "NO3": no3m})
        X = build_features(dfc, include_mld=True, include_no3=True).to_numpy()
        X = np.column_stack([X, np.full(len(dz), np.log(z_nutr + 1.0)), np.full(len(dz), nutr_max),
                             np.full(len(dz), np.log(z_pyc + 1.0)), np.full(len(dz), strat_max)])
        chlam = predict_field(chla_model, chla_norm, X, device, clip=TRACER_META["Chla"]["clip"])
        chlam = np.where(dz > cutoff, 0.0, chlam)

        ax = axes[row, 0]
        ax.plot(obs_no3.G2nitrate, obs_no3.G2depth, "o", color="tab:blue", ms=5, label="GLODAP obs")
        ax.plot(no3m, dz, "-", color="tab:red", lw=1.8, label="model (combined, low-sal)")
        ax.set_ylim(ymax, 0); ax.set_xlabel("NO3 (umol/kg)"); ax.set_ylabel("depth (m)")
        ax.axhline(seafloor, color="0.4", lw=0.8, ls=":")
        ax.set_title(f"{label}  {sid}\n{r.lat:.1f}N {r.lon:.1f}E  {r.date:%Y-%m-%d}"
                     f"  (seafloor {seafloor:.0f} m)", fontsize=9)
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower right")

        ax = axes[row, 1]
        ax.plot(obs_chla.G2chla, obs_chla.G2depth, "o", color="tab:green", ms=5, label="GLODAP obs")
        ax.plot(chlam, dz, "-", color="tab:red", lw=1.8, label="model (allfeat)")
        ax.set_ylim(ymax, 0); ax.set_xlabel("Chl-a (mg/m3)")
        ax.axhline(seafloor, color="0.4", lw=0.8, ls=":")
        ax.set_title(f"{label}  Chl-a", fontsize=9); ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")

    note = (f"  [0-{args.ymax:.0f} m zoom]" if args.ymax is not None
            else "  [to seafloor]" if args.full_depth else "")
    fig.suptitle("Observed (GLODAP, deepest-NO3 sites) vs model vertical profiles on "
                 "FORA-JPN60 T/S — NO3 & Chl-a" + note, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    if args.ymax is not None:
        out = OUT_DIR / f"fora_glodap_deep_site_profiles_{int(args.ymax)}m.png"
    elif args.full_depth:
        out = OUT_DIR / "fora_glodap_deep_site_profiles_fulldepth.png"
    else:
        out = OUT_DIR / "fora_glodap_deep_site_profiles.png"
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
