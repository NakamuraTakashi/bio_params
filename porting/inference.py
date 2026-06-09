"""Self-contained inference for the confirmed biogeochemical MLP models.

Extracted from scripts/predict_roms_ini_depths.py so it depends ONLY on the
portable modules (model.py, dataset.py, persist.py, features.py, profiles.py,
satellite.py) -- no training code, no script globals. Copy this into the new
project and change the `bio_params` imports to the new package name.

What it provides:
  - predict_tracer(...)            base7 tracers (TA/DIC/NO3/O2/SiO4/PO4/C13/C14)
  - structure_descriptors(...)     MLD / pycnocline / nutricline per water column
  - predict_chla_at_depths(...)    plain allfeat Chl-a (13 features, 200 m cutoff)
  - surface_anchor(...)            daily-satellite surface rescale (taper 100 m)
  - blend_low_salinity(...)        low-salinity mixing-line correction
See docs/porting_spec_inference.md for the full algorithm spec.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

# --- portable deps (rename `bio_params` to the new package in the new project) ---
from bio_params.features import build_features
from bio_params.persist import load_artifact
from bio_params.profiles import sigma0

# ---------------------------------------------------------------- constants ----
# model source per target (O2/NO3 = combined; rest = glodap; Chl-a = allfeat)
SOURCE = {"O2": "combined", "NO3": "combined", "TA": "glodap", "DIC": "glodap",
          "SiO4": "glodap", "PO4": "glodap", "C13": "glodap", "C14": "glodap",
          "Chla": "combined"}
CHLA_STEM = "combined_Chla_allfeat"
CLIP = {"TA": (2000, 2600), "DIC": (1800, 2500), "NO3": (0, 60), "PO4": (0, 5),
        "SiO4": (0, 250), "O2": (0, 500), "C13": (-5, 5), "C14": (-300, 250),
        "Chla": (0, 50)}
CHLA_CUTOFF_M = 200.0

SALINITY_REGRESSION = {
    "NO3": dict(slope=-5.315, intercept=185.78),
    "PO4": dict(slope=-0.176, intercept=6.29),
    "SiO4": dict(slope=-5.789, intercept=207.03),
    "TA": dict(slope=8.390, intercept=1983.5),
}
BLEND_S_LO, BLEND_S_HI, REGRESSION_S_FLOOR = 30.8, 34.0, 30.8
ANCHOR_TAPER_M, ANCHOR_CLIP = 100.0, (0.2, 5.0)


# ---------------------------------------------------------------- core MLP ------
def model_path(target: str, model_dir: Path) -> Path:
    if target == "Chla":
        return Path(model_dir) / f"{CHLA_STEM}.pt"
    return Path(model_dir) / f"{SOURCE.get(target, 'glodap')}_{target}.pt"


def load_target(target: str, model_dir: Path, device="cpu"):
    model, norm, meta = load_artifact(model_path(target, model_dir), map_location=device)
    model.to(device).eval()
    return model, norm, meta


def predict_field(model, normalizer, X, device="cpu", log_target=False, clip=None,
                  batch=200_000):
    """Forward pass with stored normalization. X: (N, in_dim). Returns (N,)."""
    Xn = normalizer.transform_x(np.asarray(X, dtype=np.float64)).astype(np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(Xn), batch):
            xb = torch.from_numpy(Xn[i:i + batch]).to(device)
            out.append(np.atleast_1d(model(xb).cpu().numpy()))
    pred = normalizer.inverse_transform_y(np.concatenate(out))
    if log_target:
        if clip is not None:
            lo = -np.inf if clip[0] <= 0 else np.log10(clip[0])
            pred = np.clip(pred, lo, np.log10(clip[1]))
        pred = np.power(10.0, pred)
    elif clip is not None:
        pred = np.clip(pred, clip[0], clip[1])
    return pred


def blend_low_salinity(pred, salinity, target):
    """Blend MLP pred with the salinity mixing-line regression at low S.
    Returns (blended, n_regression). Only NO3/PO4/SiO4/TA have a regression."""
    reg = SALINITY_REGRESSION.get(target)
    if reg is None:
        return np.asarray(pred), 0
    s = np.asarray(salinity, dtype=np.float64)
    reg_val = np.clip(reg["slope"] * np.maximum(s, REGRESSION_S_FLOOR) + reg["intercept"], 0.0, None)
    w = np.clip((BLEND_S_HI - s) / (BLEND_S_HI - BLEND_S_LO), 0.0, 1.0)
    return w * reg_val + (1.0 - w) * pred, int((w > 0).sum())


# ---------------------------------------------------- simple base7 tracers ------
def predict_tracer(target, latitude, longitude, depth, temperature, salinity, *,
                   model_dir, low_sal=False, device="cpu"):
    """Predict a base7 tracer at flat point arrays (all same length). Returns (N,)."""
    model, norm, meta = load_target(target, model_dir, device)
    df = pd.DataFrame(dict(latitude=latitude, longitude=longitude, depth=depth,
                           temperature=temperature, salinity=salinity))
    X = build_features(df).to_numpy()
    pred = predict_field(model, norm, X, device,
                         log_target=bool(meta["extra"].get("log_target", False)),
                         clip=CLIP.get(target))
    if low_sal and target in SALINITY_REGRESSION:
        pred, _ = blend_low_salinity(pred, salinity, target)
    return pred


# ---------------------------------------------- per-column structure ------------
def _grad_peak(d, v, dmin=5.0, dmax=300.0):
    """Depth & magnitude of max dv/dz in [dmin,dmax]. d:(M,) ascending, v:(M,ncol).
    Returns (zpeak, gpeak) each (ncol,), NaN where no valid interval."""
    M, ncol = v.shape
    dz = np.diff(d)[:, None]
    dv = np.diff(v, axis=0)
    mid = np.broadcast_to((0.5 * (d[:-1] + d[1:]))[:, None], (M - 1, ncol))
    with np.errstate(invalid="ignore", divide="ignore"):
        grad = dv / dz
    good = np.isfinite(grad) & (dz > 0) & (mid >= dmin) & (mid <= dmax)
    idx = np.argmax(np.where(good, grad, -np.inf), axis=0)
    anyg = good.any(axis=0)
    zpeak = np.where(anyg, np.take_along_axis(mid, idx[None], 0)[0], np.nan)
    gpeak = np.where(anyg, np.maximum(np.take_along_axis(grad, idx[None], 0)[0], 0.0), np.nan)
    return zpeak, gpeak


def mixed_layer_depth(d, sig, threshold=0.03, ref_depth=10.0):
    """MLD (ncol,) from sigma_t. d:(M,) ascending surface-first, sig:(M,ncol)."""
    M, ncol = sig.shape
    D = np.broadcast_to(d[:, None], (M, ncol))
    finite = np.isfinite(sig)
    iref = np.argmin(np.where(finite, np.abs(D - ref_depth), np.inf), axis=0)
    ref_sig = np.take_along_axis(sig, iref[None], 0)[0]
    level = np.arange(M)[:, None]
    exceeded = finite & (level > iref[None]) & (sig > ref_sig[None] + threshold)
    any_ex = exceeded.any(axis=0)
    mld = np.take_along_axis(D, exceeded.argmax(axis=0)[None], 0)[0]
    deepest = np.where(finite, D, -np.inf).max(axis=0)
    mld = np.where(any_ex, mld, deepest)
    return np.where(finite.any(axis=0), mld, np.nan)


def _no3_profile(d, T, S, lat, lon, no3_model, no3_norm, low_sal, device):
    """Predicted NO3 profile (M,ncol) from combined_NO3 (low-sal optional)."""
    M, ncol = T.shape
    out = np.full((M, ncol), np.nan)
    for lv in range(M):
        ok = np.isfinite(T[lv]) & np.isfinite(S[lv])
        if not ok.any():
            continue
        df = pd.DataFrame(dict(latitude=lat[ok], longitude=lon[ok],
                               depth=np.full(int(ok.sum()), d[lv]),
                               temperature=T[lv][ok], salinity=S[lv][ok]))
        no3 = predict_field(no3_model, no3_norm, build_features(df).to_numpy(),
                            device, clip=CLIP["NO3"])
        if low_sal:
            no3, _ = blend_low_salinity(no3, S[lv][ok], "NO3")
        out[lv, ok] = no3
    return out


def structure_descriptors(d, T, S, lat, lon, *, model_dir, low_sal=False, device="cpu"):
    """Per-column MLD / pycnocline / nutricline for the allfeat Chl-a features.

    d   : (M,) profile depths, ascending (surface-first, positive metres)
    T,S : (M, ncol) ; lat,lon : (ncol,)
    Returns dict of (ncol,) arrays: mld, z_pyc, strat_max, z_nutr, nutr_max.
    """
    M, ncol = T.shape
    D = np.broadcast_to(d[:, None], (M, ncol))
    lat3 = np.broadcast_to(np.asarray(lat)[None], (M, ncol))
    sig = sigma0(S.ravel(), T.ravel(), D.ravel(), lat3.ravel()).reshape(M, ncol)
    mld = mixed_layer_depth(d, sig)
    z_pyc, strat_max = _grad_peak(d, sig)
    no3_model, no3_norm, _ = load_target("NO3", model_dir, device)
    no3p = _no3_profile(d, T, S, np.asarray(lat), np.asarray(lon),
                        no3_model, no3_norm, low_sal, device)
    z_nutr, nutr_max = _grad_peak(d, no3p)
    return dict(mld=mld, z_pyc=z_pyc, strat_max=strat_max, z_nutr=z_nutr, nutr_max=nutr_max)


# ---------------------------------------------- Chl-a (allfeat) -----------------
def predict_chla_at_depths(latitude, longitude, depth, temperature, salinity,
                           mld, no3_feat, z_nutr, nutr_max, z_pyc, strat_max, *,
                           model_dir, cutoff=CHLA_CUTOFF_M, device="cpu"):
    """Plain allfeat Chl-a at flat point arrays. The per-column scalars
    (mld/z_nutr/...) must already be mapped onto each point. `no3_feat` is the
    NO3 point value (combined_NO3, low-sal corrected if desired)."""
    model, norm, _ = load_target("Chla", model_dir, device)
    df = pd.DataFrame(dict(latitude=latitude, longitude=longitude, depth=depth,
                           temperature=temperature, salinity=salinity,
                           mld=mld, NO3=no3_feat))
    X = build_features(df, include_mld=True, include_no3=True).to_numpy()
    X = np.column_stack([X, np.log(np.asarray(z_nutr) + 1.0), nutr_max,
                         np.log(np.asarray(z_pyc) + 1.0), strat_max])
    pred = predict_field(model, norm, X, device, clip=CLIP["Chla"])
    return np.where(np.asarray(depth) > cutoff, 0.0, pred)


def surface_anchor(chla_by_depth, depths, sat_surf, model_surf,
                   taper=ANCHOR_TAPER_M, clip=ANCHOR_CLIP, floor=1e-3, smooth=True):
    """Rescale each per-depth Chl-a field toward the (smoothed) satellite surface.

    chla_by_depth : list of 2D arrays (one per depth in `depths`)
    sat_surf, model_surf : 2D fields (same shape). Returns (anchored_list, Rhat).
    """
    s = np.asarray(sat_surf, dtype=np.float64)
    if smooth:
        from scipy.ndimage import median_filter
        fin = np.isfinite(s)
        s = np.where(fin, median_filter(np.where(fin, s, 0.0), size=3), np.nan)
    rhat = np.clip(s / np.maximum(model_surf, floor), clip[0], clip[1])
    rhat = np.where(np.isfinite(rhat), rhat, 1.0)
    out = []
    for fld, z in zip(chla_by_depth, depths):
        reff = 1.0 + (rhat - 1.0) * np.clip((taper - z) / taper, 0.0, 1.0)
        out.append(fld * reff)
    return out, rhat


# ------------------------------------------------------------------ smoke test --
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="smoke test the inference kit")
    ap.add_argument("--model-dir", default=str(Path(__file__).resolve().parent.parent
                                               / "models" / "pretrained"))
    args = ap.parse_args()
    md = Path(args.model_dir)
    rng = np.random.default_rng(0)
    ncol, M = 4, 24
    d = np.linspace(1, 500, M)                                  # depths
    lat = np.array([20., 30., 40., 28.]); lon = np.array([140., 145., 150., 125.])
    T = np.linspace(25, 4, M)[:, None] + rng.normal(0, 0.2, (M, ncol))
    S = np.linspace(34.8, 34.3, M)[:, None] + rng.normal(0, 0.02, (M, ncol))
    S[:3, 3] = 31.0                                            # a low-salinity column

    # simple tracers (flatten one level)
    no3 = predict_tracer("NO3", lat, lon, np.full(ncol, 100.0), T[10], S[10],
                         model_dir=md, low_sal=True)
    ta = predict_tracer("TA", lat, lon, np.full(ncol, 10.0), T[0], S[0], model_dir=md, low_sal=True)
    print("NO3@100m:", np.round(no3, 2), " TA@10m:", np.round(ta, 1))

    # Chl-a full path
    st = structure_descriptors(d, T, S, lat, lon, model_dir=md, low_sal=True)
    print("MLD:", np.round(st["mld"], 0), " z_nutr:", np.round(st["z_nutr"], 0),
          " z_pyc:", np.round(st["z_pyc"], 0))
    zt = 50.0
    no3_feat = predict_tracer("NO3", lat, lon, np.full(ncol, zt), T[2], S[2],
                              model_dir=md, low_sal=True)
    chla = predict_chla_at_depths(lat, lon, np.full(ncol, zt), T[2], S[2],
                                  st["mld"], no3_feat, st["z_nutr"], st["nutr_max"],
                                  st["z_pyc"], st["strat_max"], model_dir=md)
    print(f"Chl-a@{zt:.0f}m:", np.round(chla, 3))
    # anchor demo (2D 1xN "fields")
    surf = predict_chla_at_depths(lat, lon, np.full(ncol, 1.0), T[0], S[0],
                                  st["mld"], no3_feat, st["z_nutr"], st["nutr_max"],
                                  st["z_pyc"], st["strat_max"], model_dir=md)
    anc, rhat = surface_anchor([surf[None]], [1.0], (surf * 1.5)[None], surf[None], smooth=False)
    print("anchor Rhat:", np.round(rhat.ravel(), 2), " surf->", np.round(anc[0].ravel(), 3))
    print("OK")
