"""Infer surface-to-depth Chl-a on the FORA-JPN60 grid with the plain "allfeat"
model (combined_Chla_allfeat: base T/S/lat/lon/depth + NO3 + nutricline +
pycnocline + MLD, 200 m cutoff; satellite-free), then compare the surface field
with the daily GlobColour satellite Chl-a.

The structure descriptors (MLD, nutricline z_nutr/nutr_max, pycnocline
z_pyc/strat_max) need full vertical T/S and an NO3 profile, so the FORA profile is
streamed down to --zmax (default 500 m; the gradient scans only reach 300 m). The
NO3 feature is from the combined model with the low-salinity mixing-line
correction applied (per the user's request).

Usage:
    uv run python scripts/predict_fora_chla.py --date 2020-06-01
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
from bio_params.profiles import sigma0

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from predict_roms_ini_depths import (                      # noqa: E402
    TRACER_META, SALT_OPEN_OCEAN_MIN, blend_low_salinity, predict_field,
    _grad_peak_field, _nearest_index)

MODEL_DIR = ROOT / "models" / "pretrained"
OUT_DIR = ROOT / "figures" / "fora"
FORA_BASE = "https://www.jamstec.go.jp/jagdas/dodsC/fora/JPN/Daily-mean/Basic-3D"
TARGET_DEPTHS = [0.0, 200.0, 500.0, 1000.0, 3000.0]
CHLA_ART = MODEL_DIR / "combined_Chla_allfeat.pt"
NO3_ART = MODEL_DIR / "combined_NO3.pt"


def fora_url(kind, date):
    d = pd.Timestamp(date)
    return f"{FORA_BASE}/{d.year}/nc_{kind}.{d:%Y%m%d}"


def load_fora(date, stride, zmax):
    """Return (d_prof, Tp, Sp, T_tgt, S_tgt, z_tgt, lon, lat).

    d_prof/Tp/Sp: profile levels (depth<=zmax) for MLD/structure descriptors.
    T_tgt/S_tgt: T/S at the 5 nearest levels to TARGET_DEPTHS for the panels.
    """
    dt = xr.open_dataset(fora_url("t", date)); dsal = xr.open_dataset(fora_url("s", date))
    zc = dt["depth"].values
    pidx = np.where(zc <= zmax)[0]
    tidx = [0] + [int(np.argmin(np.abs(zc - z))) for z in TARGET_DEPTHS[1:]]
    sl = dict(lat=slice(None, None, stride), lon=slice(None, None, stride))
    Tp = np.asarray(dt["thetao"].isel(time=0, depth=pidx, **sl).load().values)
    Sp = np.asarray(dsal["so"].isel(time=0, depth=pidx, **sl).load().values)
    Tt = np.asarray(dt["thetao"].isel(time=0, depth=tidx, **sl).load().values)
    St = np.asarray(dsal["so"].isel(time=0, depth=tidx, **sl).load().values)
    lon = dt["lon"].values[::stride]; lat = dt["lat"].values[::stride]
    dt.close(); dsal.close()
    return zc[pidx], Tp, Sp, Tt, St, zc[tidx], lon, lat


def predict_no3_field(Tk, Sk, LAT, LON, depth_val, model, norm, device, low_sal):
    """Combined-model NO3 at one level (J,I); low-sal mixing-line corrected."""
    out = np.full(Tk.shape, np.nan)
    ok = np.isfinite(Tk) & np.isfinite(Sk)
    if not ok.any():
        return out
    df = pd.DataFrame({"latitude": LAT[ok], "longitude": LON[ok],
                       "depth": np.full(int(ok.sum()), depth_val),
                       "temperature": Tk[ok], "salinity": Sk[ok]})
    no3 = predict_field(model, norm, build_features(df).to_numpy(), device, clip=(0.0, 60.0))
    if low_sal:
        no3, _ = blend_low_salinity(no3, Sk[ok], "NO3")
    out[ok] = no3
    return out


def fora_mld(d, sig):
    """MLD (m, 2D) from a surface-first sigma0 profile (d positive, surface-first)."""
    M, J, I = sig.shape
    d3 = np.broadcast_to(d[:, None, None], (M, J, I))
    finite = np.isfinite(sig)
    dd = np.where(finite, np.abs(d3 - 10.0), np.inf)
    iref = np.argmin(dd, axis=0)
    ref_sig = np.take_along_axis(sig, iref[None], axis=0)[0]
    level = np.arange(M)[:, None, None]
    exceeded = finite & (level > iref[None]) & (sig > ref_sig[None] + 0.03)
    any_ex = exceeded.any(axis=0)
    mld = np.take_along_axis(d3, exceeded.argmax(axis=0)[None], axis=0)[0]
    deepest = np.where(finite, d3, -np.inf).max(axis=0)
    mld = np.where(any_ex, mld, deepest)
    return np.where(finite.any(axis=0), mld, np.nan)


def surface_anchor_fields(fields, depths, sat_surf, model_surf, taper_end, clip, floor=1e-3):
    """Rescale each depth field by R_eff(z) = 1 + (Rhat-1)*linear_taper(z) so the
    surface matches the (3x3-median-smoothed) satellite Chl-a. Rhat = clip(sat /
    model_surf, clip) bounds the correction so model extrapolation (e.g. low-salinity
    coastal, model_surf ~ 0) cannot blow up. Returns (anchored_fields, Rhat 2D)."""
    from scipy.ndimage import median_filter
    fin = np.isfinite(sat_surf)
    sm = median_filter(np.where(fin, sat_surf, 0.0), size=3)
    sat = np.where(fin, sm, np.nan)
    rhat = np.clip(sat / np.maximum(model_surf, floor), clip[0], clip[1])
    rhat = np.where(np.isfinite(rhat), rhat, 1.0)
    out = []
    for fld, z in zip(fields, depths):
        reff = 1.0 + (rhat - 1.0) * np.clip((taper_end - z) / taper_end, 0.0, 1.0)
        out.append(fld * reff)
    return out, rhat


def make_panels(ax_row, lon, lat, fields, depths_actual, vmax, cmap, low_masks):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    for ax, fld, zk, low in zip(ax_row, fields, depths_actual, low_masks):
        pcm = ax.pcolormesh(lon, lat, np.ma.masked_invalid(fld), cmap=cmap,
                            vmin=0.0, vmax=vmax, shading="auto", transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=2); ax.coastlines(lw=0.3, zorder=3)
        if low is not None and low.any():
            ax.contourf(lon, lat, low.astype(float), levels=[0.5, 1.5], colors="none",
                        hatches=["//"], transform=ccrs.PlateCarree(), zorder=4)
        plt.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.04, fraction=0.05).set_label(
            "Chl-a (mg/m3)", fontsize=8)
    return pcm


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--date", default="2020-06-01")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--zmax", type=float, default=500.0, help="profile depth for descriptors")
    p.add_argument("--no-low-sal", action="store_true")
    p.add_argument("--surface-anchor", action="store_true",
                   help="rescale the upper profile so the surface matches the satellite "
                        "Chl-a: R=clip(sat/model_surf, clip) tapered linearly to 1 by "
                        "--anchor-taper m (keeps deep/extrapolation results plausible)")
    p.add_argument("--anchor-taper", type=float, default=100.0,
                   help="depth (m) where the surface-anchor correction fades to 1")
    p.add_argument("--anchor-clip", type=float, nargs=2, default=[0.2, 5.0],
                   metavar=("LO", "HI"), help="clip bounds for the surface ratio R")
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    low_sal = not args.no_low_sal
    import cartopy.crs as ccrs

    print(f"FORA-JPN60 {args.date}: loading profile (zmax={args.zmax}m, stride={args.stride}) ...",
          flush=True)
    d_prof, Tp, Sp, Tt, St, z_tgt, lon, lat = load_fora(args.date, args.stride, args.zmax)
    LON, LAT = np.meshgrid(lon, lat)
    M = len(d_prof)
    print(f"  grid {Tt.shape[1:]}  profile {M} levels (to {d_prof[-1]:.0f}m)  "
          f"panel z={[f'{z:.0f}' for z in z_tgt]} m", flush=True)

    chla_model, chla_norm, chla_meta = load_artifact(CHLA_ART, map_location=device)
    chla_model.to(device)
    cutoff = float(chla_meta["extra"]["cutoff_depth"])
    no3_model, no3_norm, _ = load_artifact(NO3_ART, map_location=device); no3_model.to(device)

    # --- structure descriptors from the profile ---
    print("  sigma0 profile + MLD ...", flush=True)
    d3 = np.broadcast_to(d_prof[:, None, None], (M, *Tt.shape[1:]))
    lat3 = np.broadcast_to(LAT[None], (M, *Tt.shape[1:]))
    sig = sigma0(Sp.ravel(), Tp.ravel(), d3.ravel(), lat3.ravel()).reshape(M, *Tt.shape[1:])
    mld = fora_mld(d_prof, sig)
    z_pyc, strat_max = _grad_peak_field(d3, sig)
    print(f"  NO3 profile ({M} levels, combined, low_sal={low_sal}) ...", flush=True)
    no3_prof = np.stack([predict_no3_field(Tp[k], Sp[k], LAT, LON, d_prof[k],
                                           no3_model, no3_norm, device, low_sal)
                         for k in range(M)])
    z_nutr, nutr_max = _grad_peak_field(d3, no3_prof)

    # --- Chl-a at each panel depth ---
    print("  Chl-a allfeat prediction per depth ...", flush=True)
    chla_fields, low_masks = [], []
    for k in range(len(TARGET_DEPTHS)):
        Tk, Sk, zk = Tt[k], St[k], z_tgt[k]
        field = np.full(Tk.shape, np.nan)
        ok = np.isfinite(Tk) & np.isfinite(Sk) & np.isfinite(mld)
        if ok.any():
            no3_k = predict_no3_field(Tk, Sk, LAT, LON, zk, no3_model, no3_norm, device, low_sal)
            df = pd.DataFrame({"latitude": LAT[ok], "longitude": LON[ok],
                               "depth": np.full(int(ok.sum()), zk),
                               "temperature": Tk[ok], "salinity": Sk[ok],
                               "mld": mld[ok], "NO3": no3_k[ok]})
            X = build_features(df, include_mld=True, include_no3=True).to_numpy()
            X = np.column_stack([X,
                                 np.log(z_nutr[ok] + 1.0), nutr_max[ok],
                                 np.log(z_pyc[ok] + 1.0), strat_max[ok]])
            pred = predict_field(chla_model, chla_norm, X, device, clip=TRACER_META["Chla"]["clip"])
            pred = np.where(zk > cutoff, 0.0, pred)         # deep -> 0 cutoff
            field[ok] = pred
        chla_fields.append(field)
        low_masks.append(np.isfinite(Tk) & np.isfinite(Sk) & (Sk < SALT_OPEN_OCEAN_MIN) if low_sal else None)

    # --- satellite surface field (for the anchor and/or the comparison) ---
    from bio_params.satellite import chla_day_field
    lat_ax, lon_ax, arr, src = chla_day_field(args.date)
    print(f"  satellite source: {src}", flush=True)
    ilat = _nearest_index(LAT.ravel(), lat_ax)
    ilon = _nearest_index(((LON.ravel() + 180) % 360) - 180, lon_ax)
    sat = arr[ilat, ilon].reshape(LAT.shape)
    model_surf_raw = chla_fields[0].copy()      # RAW model surface (for the comparison)

    # --- optional surface anchor (rescale upper profile toward the satellite surface) ---
    anchored = False
    if args.surface_anchor:
        chla_fields, rhat = surface_anchor_fields(
            chla_fields, z_tgt, sat, model_surf_raw, args.anchor_taper, tuple(args.anchor_clip))
        anchored = True
        rf = rhat[np.isfinite(model_surf_raw)]
        print(f"  surface anchor: taper->{args.anchor_taper:.0f}m  clip{tuple(args.anchor_clip)}  "
              f"Rhat median={np.median(rf):.2f} p10={np.percentile(rf,10):.2f} "
              f"p90={np.percentile(rf,90):.2f}", flush=True)

    # --- depth-panel map ---
    surf_fin = chla_fields[0][np.isfinite(chla_fields[0])]
    vmax = float(np.percentile(surf_fin, 98)) if surf_fin.size else 1.0
    fig, axes = plt.subplots(1, len(TARGET_DEPTHS), figsize=(4.6 * len(TARGET_DEPTHS), 5.2),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    make_panels(axes, lon, lat, chla_fields, z_tgt, vmax, "viridis", low_masks)
    for ax, k in zip(axes, range(len(TARGET_DEPTHS))):
        lbl = "surface" if k == 0 else f"{TARGET_DEPTHS[k]:.0f} m"
        ax.set_title(f"{lbl}  (FORA z={z_tgt[k]:.0f} m)", fontsize=9)
    anchor_note = (f"  [surface-anchored: taper->{args.anchor_taper:.0f}m]" if anchored else "")
    fig.suptitle(f"FORA-JPN60 {args.date}: Chl-a — plain allfeat MLP "
                 f"(NO3 low-sal corrected, 200 m cutoff; hatch S<{SALT_OPEN_OCEAN_MIN:.0f})"
                 + anchor_note, fontsize=12)
    fig.tight_layout()
    suffix = "_anchored" if anchored else ""
    out1 = OUT_DIR / f"fora_pred_Chla_{args.date.replace('-', '')}{suffix}.png"
    fig.savefig(out1, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out1}", flush=True)

    # --- RAW surface model vs daily satellite + difference (always the raw model) ---
    print("  daily satellite Chl-a + difference ...", flush=True)
    model_surf = model_surf_raw
    sat = np.where(np.isfinite(model_surf), sat, np.nan)   # compare over ocean only
    diff = model_surf - sat

    import cartopy.feature as cfeature
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    vmax = float(np.nanpercentile(np.concatenate([model_surf[np.isfinite(model_surf)],
                                                  sat[np.isfinite(sat)]]), 98))
    dlim = float(np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 98))
    panels = [(model_surf, "viridis", 0, vmax, "Model surface Chl-a (allfeat)", "Chl-a (mg/m3)"),
              (sat, "viridis", 0, vmax, f"Daily satellite Chl-a ({src})", "Chl-a (mg/m3)"),
              (diff, "RdBu_r", -dlim, dlim, "Difference (model - satellite)", "Δ Chl-a (mg/m3)")]
    for ax, (fld, cmap, vmn, vmx, title, clabel) in zip(axes, panels):
        pcm = ax.pcolormesh(lon, lat, np.ma.masked_invalid(fld), cmap=cmap, vmin=vmn, vmax=vmx,
                            shading="auto", transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=2); ax.coastlines(lw=0.3, zorder=3)
        ax.set_title(title, fontsize=10)
        plt.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.04, fraction=0.05).set_label(clabel, fontsize=8)
    md = np.isfinite(diff)
    bias = float(np.nanmean(diff[md])); mae = float(np.nanmean(np.abs(diff[md])))
    fig.suptitle(f"FORA-JPN60 {args.date} surface Chl-a: model vs daily satellite  "
                 f"(bias={bias:+.3f}, MAE={mae:.3f} mg/m3, n={int(md.sum()):,})", fontsize=12)
    fig.tight_layout()
    out2 = OUT_DIR / f"fora_Chla_vs_satellite_{args.date.replace('-', '')}.png"
    fig.savefig(out2, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out2}", flush=True)

    # log-space agreement, matched valid pixels
    both = np.isfinite(model_surf) & np.isfinite(sat) & (model_surf > 0) & (sat > 0)
    if both.sum() > 100:
        lm = np.log10(model_surf[both]); ls = np.log10(sat[both])
        r2 = 1.0 - np.sum((lm - ls) ** 2) / np.sum((ls - ls.mean()) ** 2)
        ratio = float(np.median(model_surf[both] / sat[both]))
        print(f"  surface log-R2(vs sat)={r2:.3f}  median(model/sat)={ratio:.3f}  n={int(both.sum()):,}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
