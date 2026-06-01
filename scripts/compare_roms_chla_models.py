"""Compare Chl-a model variants on the ROMS grid by depth (shared color scale).

Rows = model variants (e.g. direct satellite-anchored absolute vs the relative
profile model); columns = depths. Each column shares a color scale so the
variants are directly comparable. Reuses the inference machinery from
predict_roms_ini_depths.py (z-coord, MLD field, satellite field, euphotic
taper, relative re-anchoring).

Usage:
    uv run python scripts/compare_roms_chla_models.py \
        --models bgc_argo:satchl:"direct (absolute, satellite feature)" \
                 bgc_argo:rel:"relative profile (x satellite, +MLD, cutoff)" \
                 combined:rel_combal:"combined relative (GLODAP+BGC, +MLD, cutoff)"
"""
from __future__ import annotations

import argparse
import importlib.util
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

_spec = importlib.util.spec_from_file_location(
    "_pri", Path(__file__).resolve().parent / "predict_roms_ini_depths.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

OUT_DIR = Path(__file__).resolve().parent.parent / "figures" / "roms_chla_model_compare"
CLIP = P.TRACER_META["Chla"]["clip"]


def reconstruct(art_path, per_depth, sat, mld, zp, device, shape):
    """Chl-a field (J,I) per depth for one model, honoring its extra flags."""
    model, norm, meta = load_artifact(art_path, map_location=device)
    model.to(device)
    e = meta["extra"]
    log_target = bool(e.get("log_target", False))
    surface_chla = bool(e.get("surface_chla", False))
    surface_chla_log = bool(e.get("surface_chla_log", True))
    include_mld = bool(e.get("include_mld", False))
    relative = bool(e.get("relative_target", False))
    rel_cap = float(e.get("rel_cap", 20.0))

    def buildX(pd_):
        if not (surface_chla or include_mld):
            return pd_["X"]
        df = pd_["feat_df"].copy()
        if surface_chla:
            df["surface_chla"] = sat[pd_["jj"], pd_["ii"]]
        if include_mld:
            df["mld"] = mld[pd_["jj"], pd_["ii"]]
        return build_features(df, include_surface_chla=surface_chla,
                              surface_chla_log=surface_chla_log,
                              include_mld=include_mld).to_numpy()

    rel_surf = None
    if relative:
        ps = per_depth[0]
        r0 = np.clip(P.predict_field(model, norm, buildX(ps), device, clip=None), 0, rel_cap)
        rel_surf = np.full(shape, np.nan)
        rel_surf[ps["jj"], ps["ii"]] = np.maximum(r0, 1e-6)

    out = []
    for pd_ in per_depth:
        jj, ii = pd_["jj"], pd_["ii"]
        f = np.full(shape, np.nan)
        if relative:
            if pd_["label"] == "surface":
                pred = sat[jj, ii].copy()
            else:
                rel = np.clip(P.predict_field(model, norm, buildX(pd_), device, clip=None), 0, rel_cap)
                rel = np.clip(rel / rel_surf[jj, ii], 0, rel_cap)
                pred = rel * sat[jj, ii]
            pred = pred * P.productive_taper(pd_["feat_df"]["depth"].to_numpy(), zp[jj, ii])
            pred = np.clip(pred, CLIP[0], CLIP[1])
        else:
            pred = P.predict_field(model, norm, buildX(pd_), device,
                                   log_target=log_target, clip=CLIP)
        f[jj, ii] = pred
        out.append(f)
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", default=[
        "bgc_argo:satchl:direct (absolute, satellite feature)",
        "bgc_argo:rel:relative profile (xsatellite,+MLD,cutoff)",
        "combined:rel_combal:combined relative (GLODAP+BGC,+MLD,cutoff)",
    ], help="each: source:tag:label")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = xr.open_dataset(P.INI, decode_times=False)
    g = xr.open_dataset(P.GRID, decode_times=False)
    h = np.where(g["h"].values <= 0, np.nan, g["h"].values).astype(np.float64)
    mask = g["mask_rho"].values
    lon, lat = g["lon_rho"].values, g["lat_rho"].values
    zeta = ds["zeta"].isel(ocean_time=0).values.astype(np.float64)
    s_rho = ds["s_rho"].values.astype(np.float64)
    Cs_r = ds["Cs_r"].values.astype(np.float64)
    hc = float(ds["hc"].values)
    temp = ds["temp"].isel(ocean_time=0).values.astype(np.float64)
    salt = ds["salt"].isel(ocean_time=0).values.astype(np.float64)
    z = P.compute_z_rho(h, zeta, s_rho, Cs_r, hc)
    land = mask < 0.5
    J, I = h.shape

    mld = P.compute_mld_field(temp, salt, z, lat)
    sat = P.load_roms_surface_chla(lat, lon, P._ini_month(P.INI))
    zp = np.fmax(P.morel_euphotic_depth(sat), mld)

    per_depth = []
    for depth in P.TARGET_DEPTHS:
        if depth == 0.0:
            T2, S2, d2, label = temp[-1], salt[-1], np.abs(z[-1]), "surface"
        else:
            T2 = P.interp_to_depth(temp, z, -depth)
            S2 = P.interp_to_depth(salt, z, -depth)
            d2 = np.full((J, I), depth); label = f"{depth:.0f} m"
        valid = np.isfinite(T2) & np.isfinite(S2) & np.isfinite(mld) & (~land)
        jj, ii = np.where(valid)
        feat_df = pd.DataFrame({"latitude": lat[jj, ii], "longitude": lon[jj, ii],
                                "depth": d2[jj, ii], "temperature": T2[jj, ii],
                                "salinity": S2[jj, ii]})
        X = build_features(feat_df).to_numpy()
        per_depth.append(dict(label=label, jj=jj, ii=ii, X=X, feat_df=feat_df))

    rows = []
    for spec in args.models:
        source, tag, label = spec.split(":", 2)
        art = P.MODEL_DIR / f"{source}_Chla_{tag}.pt"
        if not art.exists():
            print(f"skip {spec}: missing {art.name}"); continue
        print(f"reconstructing {label} ({art.name}) ...")
        fields = reconstruct(art, per_depth, sat, mld, zp, device, (J, I))
        rows.append((label, fields))

    ncol = len(P.TARGET_DEPTHS); nrow = len(rows)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 4.6 * nrow),
                             squeeze=False, constrained_layout=True)
    cmap = P.TRACER_META["Chla"]["cmap"]
    for c in range(ncol):
        vals = np.concatenate([rows[r][1][c][np.isfinite(rows[r][1][c])] for r in range(nrow)])
        vmax = np.percentile(vals, 98) if vals.size else 1.0
        for r in range(nrow):
            ax = axes[r][c]
            pcm = ax.pcolormesh(lon, lat, np.ma.masked_invalid(rows[r][1][c]),
                                cmap=cmap, vmin=0, vmax=vmax, shading="auto")
            ax.set_facecolor("0.8"); ax.set_aspect("equal")
            if r == 0:
                ax.set_title(per_depth[c]["label"], fontsize=11)
            if c == 0:
                ax.set_ylabel(rows[r][0], fontsize=9)
            fig.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.06, fraction=0.05)
    fig.suptitle("Chl-a (mg/m3) on ROMS grid — model variants (rows) x depth (cols), "
                 "shared color scale per column", fontsize=12)
    out = OUT_DIR / "compare_Chla_depths.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    ds.close(); g.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
