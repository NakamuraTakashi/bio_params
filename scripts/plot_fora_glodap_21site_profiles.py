"""For each of the 21 GLODAP "8-variable full-set" sites that sample to >=1000 m
(DIC/TA/O2/NO3/PO4/SiO4 deep; DOC/Chl-a upper only), overlay GLODAP observed
vertical profiles with: the FORA-JPN60 T/S column, and the model-estimated
profiles of TA, DIC, O2, NO3, PO4, SiO4, DOC, Chl-a (models fed FORA T/S). Two
figures per site (full depth + 0-300 m) = 42 figures. Title carries cruise, lat,
lon, date.

Models: TA/DIC/PO4/SiO4 -> glodap_*; O2/NO3 -> combined_*; DOC -> glodap_DOC
(clean); Chl-a -> combined_Chla_allfeat (+ structure descriptors + 200 m cutoff +
daily-satellite surface anchor). Low-salinity regression on NO3/PO4/SiO4/TA.

Usage:
    uv run python scripts/plot_fora_glodap_21site_profiles.py
"""
from __future__ import annotations

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
    TRACER_META, blend_low_salinity, predict_field, _nearest_index)
from predict_fora_chla import fora_url                                              # noqa: E402
from plot_fora_glodap_site_profiles import struct_for_column                        # noqa: E402

CSV = ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
PROC = ROOT / "data" / "glodap" / "processed"
MODEL_DIR = ROOT / "models" / "pretrained"
OUT_DIR = ROOT / "figures" / "fora" / "sites21"
CACHE = PROC / "fora_21sites_profiles.nc"
BOX = dict(latmin=19.96, latmax=52.02, lonmin=116.94, lonmax=160.03)

# var -> (obs value col, flag col, model stem, low-sal target or None, core?)
VARCFG = {
    "TA":   ("G2talk", "G2talkf", "glodap_TA", "TA", True),
    "DIC":  ("G2tco2", "G2tco2f", "glodap_DIC", None, True),
    "O2":   ("G2oxygen", "G2oxygenf", "combined_O2", None, True),
    "NO3":  ("G2nitrate", "G2nitratef", "combined_NO3", "NO3", True),
    "PO4":  ("G2phosphate", "G2phosphatef", "glodap_PO4", "PO4", True),
    "SiO4": ("G2silicate", "G2silicatef", "glodap_SiO4", "SiO4", True),
    "DOC":  ("G2doc", "G2docf", "glodap_DOC", None, False),
    "Chla": ("G2chla", "G2chlaf", "combined_Chla_allfeat", None, False),
}
PANELS = ["T", "S", "TA", "DIC", "O2", "NO3", "PO4", "SiO4", "DOC", "Chla"]
ALLVALS = {k: v[0] for k, v in VARCFG.items()}


def select_sites():
    core = ["G2tco2", "G2talk", "G2oxygen", "G2nitrate", "G2phosphate", "G2silicate"]
    cols = (["G2cruise", "G2station", "G2latitude", "G2longitude", "G2depth",
             "G2year", "G2month", "G2day", "G2temperature", "G2salinity"]
            + [c for v in VARCFG.values() for c in v[:2]])
    df = pd.read_csv(CSV, usecols=list(dict.fromkeys(cols))).replace(-9999, np.nan)
    df = df.dropna(subset=["G2latitude", "G2longitude", "G2depth"])
    df = df[(df.G2latitude >= BOX["latmin"]) & (df.G2latitude <= BOX["latmax"])
            & (df.G2longitude >= BOX["lonmin"]) & (df.G2longitude <= BOX["lonmax"])]
    full = df[df[list(ALLVALS.values())].notna().all(axis=1)]
    keys = set(zip(full.G2cruise, full.G2station))
    sub = df[[(c, s) in keys for c, s in zip(df.G2cruise, df.G2station)]].copy()
    sites = []
    for (c, s), g in sub.groupby(["G2cruise", "G2station"]):
        dmax = g[g[core].notna().any(axis=1)].G2depth.max()
        if dmax >= 1000:
            r0 = g.iloc[0]
            date = pd.Timestamp(int(r0.G2year), int(r0.G2month), int(r0.G2day))
            sites.append(dict(cruise=int(c), station=s, lat=float(r0.G2latitude),
                              lon=float(r0.G2longitude), date=date))
    meta = pd.DataFrame(sites).sort_values("lat").reset_index(drop=True)
    meta["sid"] = meta.cruise.astype(str) + "_" + meta.station.astype(str)
    return meta, sub


def extract_fora(meta, cache=CACHE):
    if cache.exists():
        return xr.open_dataset(cache)
    ref = xr.open_dataset(fora_url("t", meta.date.iloc[0]))
    flat = ref["lat"].values; flon = ref["lon"].values; fd = ref["depth"].values
    ref.close(); nlev = len(fd)
    Tp = np.full((len(meta), nlev), np.nan); Sp = np.full((len(meta), nlev), np.nan)
    for d, g in meta.groupby(meta.date):
        dt = xr.open_dataset(fora_url("t", d)); dsal = xr.open_dataset(fora_url("s", d))
        for k, r in g.iterrows():
            j = int(np.argmin(np.abs(flat - r.lat))); i = int(np.argmin(np.abs(flon - r.lon)))
            Tp[k] = np.asarray(dt["thetao"].isel(time=0, lat=j, lon=i).load().values)
            Sp[k] = np.asarray(dsal["so"].isel(time=0, lat=j, lon=i).load().values)
        dt.close(); dsal.close()
        print(f"  FORA {d:%Y-%m-%d}: {len(g)} sites", flush=True)
    out = xr.Dataset({"fora_temp": (("station", "level"), Tp),
                      "fora_salt": (("station", "level"), Sp)},
                     coords={"station": meta.sid.values, "level": np.arange(nlev),
                             "fora_depth": ("level", fd)})
    out.to_netcdf(cache)
    return out


def plot_site_set(meta, sub, prof, out_dir, device):
    out_dir.mkdir(parents=True, exist_ok=True)
    fd = prof["fora_depth"].values
    sid_order = prof["station"].values.tolist()

    models = {}
    for name, (_, _, stem, _, _) in VARCFG.items():
        m, n, mt = load_artifact(MODEL_DIR / f"{stem}.pt", map_location=device)
        models[name] = (m.to(device), n, mt)
    no3_model, no3_norm, _ = models["NO3"]
    chla_model, chla_norm, chla_meta = models["Chla"]
    cutoff = float(chla_meta["extra"]["cutoff_depth"])
    from bio_params.satellite import chla_day_field

    for _, site in meta.iterrows():
        sid = site.sid; k = sid_order.index(sid)
        Tcol, Scol = prof["fora_temp"].values[k], prof["fora_salt"].values[k]
        fin = np.isfinite(Tcol) & np.isfinite(Scol)
        dz, Tz, Sz = fd[fin], Tcol[fin], Scol[fin]
        seafloor = float(dz.max()) if dz.size else 1000.0
        g = sub[(sub.G2cruise == site.cruise) & (sub.G2station == site.station)]

        # model profiles at finite FORA levels
        mp = {}
        for v, (_, _, _, lowsal, _) in VARCFG.items():
            if v == "Chla":
                continue
            mdl, nm, mt = models[v]
            dfn = pd.DataFrame({"latitude": site.lat, "longitude": site.lon,
                                "depth": dz, "temperature": Tz, "salinity": Sz})
            pred = predict_field(mdl, nm, build_features(dfn).to_numpy(), device,
                                 log_target=bool(mt["extra"].get("log_target", False)),
                                 clip=TRACER_META[v]["clip"])
            if lowsal:
                pred, _ = blend_low_salinity(pred, Sz, lowsal)
            mp[v] = pred
        # Chl-a: structure descriptors + cutoff + daily-sat anchor
        no3_feat = predict_field(no3_model, no3_norm,
                                 build_features(pd.DataFrame({"latitude": site.lat,
                                 "longitude": site.lon, "depth": dz, "temperature": Tz,
                                 "salinity": Sz})).to_numpy(), device, clip=(0, 60))
        no3_feat, _ = blend_low_salinity(no3_feat, Sz, "NO3")
        mld, z_pyc, strat_max, z_nutr, nutr_max = struct_for_column(
            fd, Tcol, Scol, site.lat, site.lon, no3_model, no3_norm, device)
        Xc = build_features(pd.DataFrame({"latitude": site.lat, "longitude": site.lon,
              "depth": dz, "temperature": Tz, "salinity": Sz, "mld": mld, "NO3": no3_feat}),
              include_mld=True, include_no3=True).to_numpy()
        Xc = np.column_stack([Xc, np.full(len(dz), np.log(z_nutr + 1)), np.full(len(dz), nutr_max),
                              np.full(len(dz), np.log(z_pyc + 1)), np.full(len(dz), strat_max)])
        chla = predict_field(chla_model, chla_norm, Xc, device, clip=TRACER_META["Chla"]["clip"])
        chla = np.where(dz > cutoff, 0.0, chla)
        try:
            la, lo, arr, _ = chla_day_field(site.date.strftime("%Y-%m-%d"))
            il = int(_nearest_index(np.array([site.lat]), la)[0])
            io = int(_nearest_index(((np.array([site.lon]) + 180) % 360) - 180, lo)[0])
            sat = float(arr[il, io]); msurf = chla[0] if len(chla) else np.nan
            if np.isfinite(sat) and np.isfinite(msurf) and msurf > 1e-3:
                rhat = float(np.clip(sat / msurf, 0.2, 5.0))
                chla = chla * (1 + (rhat - 1) * np.clip((100 - dz) / 100, 0, 1))
        except Exception:
            pass
        mp["Chla"] = chla

        # obs depth max for the figure span
        obs_dmax = float(g.G2depth.max())
        for tag, ymax in [("fulldepth", max(seafloor, obs_dmax)), ("300m", 300.0)]:
            fig, axes = plt.subplots(2, 5, figsize=(20, 10))
            for ax, v in zip(axes.ravel(), PANELS):
                if v == "T":
                    ax.plot(g.G2temperature, g.G2depth, "o", ms=4, color="tab:blue", label="GLODAP")
                    ax.plot(Tz, dz, "-", color="tab:red", lw=1.6, label="FORA")
                    ax.set_xlabel("T (degC)")
                elif v == "S":
                    ax.plot(g.G2salinity, g.G2depth, "o", ms=4, color="tab:blue", label="GLODAP")
                    ax.plot(Sz, dz, "-", color="tab:red", lw=1.6, label="FORA")
                    ax.set_xlabel("S (PSU)")
                else:
                    vc, fc, _, _, core = VARCFG[v]
                    if core:
                        o2 = g[(g[fc] == 2) & g[vc].notna()]
                        o0 = g[(g[fc] != 2) & g[vc].notna()]   # flag 0 = calc/interp
                        ax.plot(o2[vc], o2.G2depth, "o", ms=4, color="tab:green",
                                label="GLODAP flag2")
                        if len(o0):
                            ax.plot(o0[vc], o0.G2depth, "o", ms=5, mfc="none",
                                    mec="tab:orange", mew=1.1, label="GLODAP calc/interp")
                    else:
                        o = g[g[vc].notna()]
                        ax.plot(o[vc], o.G2depth, "o", ms=4, color="tab:green", label="GLODAP")
                    ax.plot(mp[v], dz, "-", color="tab:red", lw=1.6, label="model")
                    ax.set_xlabel(f"{v} ({TRACER_META[v]['unit']})")
                ax.set_ylim(ymax, 0); ax.grid(alpha=0.3); ax.set_title(v, fontsize=10)
                if v in ("T", "NO3"):
                    ax.set_ylabel("depth (m)")
                if v in ("T", "TA", "NO3"):
                    ax.legend(fontsize=7, loc="lower right")
            fig.suptitle(f"cruise {site.cruise}  station {site.station}  "
                         f"{site.lat:.2f}N {site.lon:.2f}E  {site.date:%Y-%m-%d}  "
                         f"[{tag}]  — GLODAP obs vs FORA T/S & model (Chl-a daily-sat anchored)\n"
                         f"core vars: filled=flag2 (measured), open orange=calc/interp (flag≠2)",
                         fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            out = out_dir / f"site_{sid}_{tag}.png"
            fig.savefig(out, dpi=100, bbox_inches="tight"); plt.close(fig)
        print(f"  {sid} ({site.lat:.1f}N {site.lon:.1f}E {site.date:%Y-%m-%d}) seafloor {seafloor:.0f}m", flush=True)

    print(f"\nsaved {2*len(meta)} figures -> {out_dir}/", flush=True)


def meta_sub_from_keys(keys):
    """Build (meta, sub) for an explicit list of (cruise, station) keys."""
    cols = (["G2cruise", "G2station", "G2latitude", "G2longitude", "G2depth",
             "G2year", "G2month", "G2day", "G2temperature", "G2salinity"]
            + [c for v in VARCFG.values() for c in v[:2]])
    df = pd.read_csv(CSV, usecols=list(dict.fromkeys(cols))).replace(-9999, np.nan)
    df = df.dropna(subset=["G2latitude", "G2longitude", "G2depth"])
    keyset = set(keys)
    sub = df[[(c, s) in keyset for c, s in zip(df.G2cruise, df.G2station)]].copy()
    rows = []
    for c, s in keys:
        g = sub[(sub.G2cruise == c) & (sub.G2station == s)]
        if not len(g):
            print(f"  WARNING: site {c}_{s} not found"); continue
        r0 = g.iloc[0]
        rows.append(dict(cruise=int(c), station=s, lat=float(r0.G2latitude),
                         lon=float(r0.G2longitude),
                         date=pd.Timestamp(int(r0.G2year), int(r0.G2month), int(r0.G2day))))
    meta = pd.DataFrame(rows)
    meta["sid"] = meta.cruise.astype(str) + "_" + meta.station.astype(str)
    return meta, sub


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta, sub = select_sites()
    print(f"21-site check: {len(meta)} sites (>=1000m core)", flush=True)
    prof = extract_fora(meta, CACHE)
    plot_site_set(meta, sub, prof, OUT_DIR, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
