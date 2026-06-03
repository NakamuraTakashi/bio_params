"""Predict GLODAP-trained biogeochemical tracers on the ROMS grid and map
them on fixed depth levels.

This is the ROMS inference template from CLAUDE.md. For each fixed depth it:
  1. interpolates the ini-file temperature & salinity to that depth using the
     S-coordinate transform (Vtransform=2, Vstretching=4),
  2. builds the SAME 7-dim feature vector used in training (lat/lon sin/cos,
     log_depth, T, S) at every valid water-column grid point,
  3. runs each saved per-target model (with its own saved normalizer),
  4. draws one contour/pcolormesh map per (tracer, depth).

Works with either GLODAP or BGC-Argo artifacts via --source; the artifact's
saved log_target flag is honored (predictions back-transformed with 10**).

CAVEAT: the models were trained on open-ocean water (S ~ 33-37). This domain
contains low-salinity coastal / inland-sea water; predictions there are
extrapolation and unreliable until local fine-tuning. Panels are annotated.

Usage:
    uv run python scripts/predict_roms_ini_depths.py
    uv run python scripts/predict_roms_ini_depths.py --targets TA DIC
    uv run python scripts/predict_roms_ini_depths.py --source bgc_argo --targets Chla O2 NO3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from bio_params.features import build_features
from bio_params.persist import load_artifact
from bio_params.base_profile import BaseProfile
from bio_params.profiles import daily_insolation_factor, sigma0

INI = Path("/mnt/d/COAWST_DATA/FORP_Kuroshio/Ini/Kuro_Ini_FORP_Nz30_20060102.00.nc")
GRID = Path("/mnt/d/COAWST_DATA/FORP_Kuroshio/Grid/forp-kuroshio_grd_v0.0.nc")
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "pretrained"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures" / "roms_ini_pred"

TARGET_DEPTHS = [0.0, 200.0, 500.0, 1000.0, 3000.0]

# Satellite surface Chl-a, for SOCA-style models that carry a surface_chla
# feature. We feed the GlobColour monthly CHL of the ini month, sampled to the
# ROMS grid, as the surface "truth" that anchors the predicted profile.
SAT_DATASET = "cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M"

# Display metadata per tracer. Units differ: nutrients/carbon are umol/kg,
# Chla is mg/m3, and the isotope ratios C13/O18 are in per mil.
# `clip` (lo, hi) bounds the prediction to a physically plausible range so
# extrapolation outside the training domain (e.g. low-salinity coastal cells)
# cannot produce absurd values. Especially important for log-target models,
# where 10** turns an extrapolated log into a runaway number.
TRACER_META = {
    "TA": dict(long="Total alkalinity", unit="umol/kg", cmap="viridis", clip=(2000, 2600)),
    "DIC": dict(long="Dissolved inorganic carbon", unit="umol/kg", cmap="viridis", clip=(1800, 2500)),
    "NO3": dict(long="Nitrate", unit="umol/kg", cmap="cividis", clip=(0, 60)),
    "PO4": dict(long="Phosphate", unit="umol/kg", cmap="cividis", clip=(0, 5)),
    "SiO4": dict(long="Silicate", unit="umol/kg", cmap="cividis", clip=(0, 250)),
    "O2": dict(long="Dissolved oxygen", unit="umol/kg", cmap="turbo", clip=(0, 500)),
    "DOC": dict(long="Dissolved organic carbon", unit="umol/kg", cmap="viridis", clip=(0, 200)),
    "Chla": dict(long="Chlorophyll-a", unit="mg/m3", cmap="YlGn", clip=(0, 50)),
    "TDN": dict(long="Total dissolved nitrogen", unit="umol/kg", cmap="cividis", clip=(0, 60)),
    "TOC": dict(long="Total organic carbon", unit="umol/kg", cmap="viridis", clip=(0, 200)),
    "DON": dict(long="Dissolved organic nitrogen", unit="umol/kg", cmap="cividis", clip=(0, 60)),
    "C13": dict(long="d13C of DIC", unit="permil", cmap="coolwarm", clip=(-5, 5)),
    "O18": dict(long="d18O", unit="permil", cmap="coolwarm", clip=(-5, 5)),
    "C14": dict(long="Delta-14C of DIC", unit="permil", cmap="coolwarm", clip=(-300, 250)),
    "H3": dict(long="Tritium", unit="TU", cmap="magma", clip=(0, 80)),
}

# Low-salinity correction (river/coastal mixing line).
# In low-salinity coastal water the MLP extrapolates badly; several tracers
# there follow a near-linear salinity mixing line instead. These OLS fits come
# from GLODAP surface (<=20 m) data in the East China Sea / Changjiang-diluted
# box (118-124E, 25-32N): value = slope * S + intercept.
#   NO3:  -5.315*S + 185.78  (R2=0.96)
#   PO4:  -0.176*S +   6.29  (R2=0.81)
#   SiO4: -5.789*S + 207.03  (R2=0.95)   strong river signal (silicate)
#   TA:   +8.390*S + 1983.5  (R2=0.71)   POSITIVE slope (river TA < ocean TA)
#
# Applied to ALL low-salinity domain grid points as a provisional fix; the
# Changjiang end-member differs from other river mouths, so values outside the
# East China Sea are approximate. Blended with the MLP between S_LO and S_HI.
#
# Validation of the TA freshwater end-member (S=0 intercept): our GLODAP fit
# gives 1983.5 umol/kg, in good agreement (~3.5%) with the published Changjiang
# freshwater end-member of ~2054 umol/kg (Xiong et al., 2019, Earth and Space
# Science, doi:10.1029/2019EA000679). We keep the self-consistent GLODAP fit
# value (1983.5) rather than substituting the literature value.
#
# DIC is intentionally excluded: its S=0 intercept (2466) far exceeds the
# published Changjiang end-member (~1609 umol/kg; Xiong et al., 2019) because
# DIC is non-conservative in the estuary (CO2 degassing / respiration), R2~0.57.
# O2 (R2~0.6) and Chl-a (no salinity relation) are also left to the MLP.
SALINITY_REGRESSION = {
    "NO3": dict(slope=-5.315, intercept=185.78),
    "PO4": dict(slope=-0.176, intercept=6.29),
    "SiO4": dict(slope=-5.789, intercept=207.03),
    "TA": dict(slope=8.390, intercept=1983.5),
}
BLEND_S_LO = 30.8   # at/below this salinity -> pure regression (GLODAP min ~30.8)
BLEND_S_HI = 34.0   # at/above this salinity -> pure MLP
REGRESSION_S_FLOOR = 30.8  # clamp S used in the regression (don't extrapolate below fit range)

# GLODAP open-ocean salinity floor; below this, predictions are extrapolation.
SALT_OPEN_OCEAN_MIN = 33.0


def compute_z_rho(h, zeta, s_rho, Cs_r, hc):
    h = h[None, :, :]
    zeta = zeta[None, :, :]
    s = s_rho[:, None, None]
    C = Cs_r[:, None, None]
    S = (hc * s + h * C) / (hc + h)
    return zeta + (zeta + h) * S


def interp_to_depth(data, z, target_z):
    """Linear interp of data (N,J,I) along z (N,J,I) to scalar target_z."""
    n = z.shape[0]
    idx = np.sum(z <= target_z, axis=0)
    idx_lo = np.clip(idx - 1, 0, n - 1)
    idx_hi = np.clip(idx, 0, n - 1)

    def gather(a, k):
        return np.take_along_axis(a, k[None, :, :], axis=0)[0]

    z_lo, z_hi = gather(z, idx_lo), gather(z, idx_hi)
    d_lo, d_hi = gather(data, idx_lo), gather(data, idx_hi)
    denom = z_hi - z_lo
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(denom != 0, (target_z - z_lo) / denom, 0.0)
    out = d_lo + w * (d_hi - d_lo)
    valid = (idx >= 1) & (idx <= n - 1)
    return np.where(valid, out, np.nan)


def _ini_month(path: Path) -> str:
    """Parse the YYYY-MM month from an ini filename like ..._20060102.00.nc."""
    import re
    m = re.search(r"(\d{4})(\d{2})(\d{2})", path.name)
    if not m:
        raise ValueError(f"cannot parse a date from {path.name}")
    return f"{m.group(1)}-{m.group(2)}"


def _ini_doy(path: Path) -> int:
    """Day-of-year from an ini filename like ..._20060102.00.nc (for E0)."""
    import re
    m = re.search(r"(\d{4})(\d{2})(\d{2})", path.name)
    if not m:
        raise ValueError(f"cannot parse a date from {path.name}")
    return int(pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{m.group(3)}").dayofyear)


def _nearest_index(coord, axis):
    """Nearest grid index for `coord` on a uniform (monotone) `axis`."""
    origin = float(axis[0])
    step = float(axis[1] - axis[0])
    idx = np.rint((coord - origin) / step).astype(np.int64)
    return np.clip(idx, 0, len(axis) - 1)


def load_roms_surface_chla(lat, lon, month, dataset_id=SAT_DATASET):
    """Satellite surface Chl-a (mg/m3) on the ROMS rho grid for `month`.

    Opens the GlobColour monthly product lazily, samples the nearest pixel to
    each (lat, lon) ROMS point, and fills cloud-masked gaps with the nearest
    valid ROMS point so the anchor field is complete (boundary conditions need
    no holes). Returns a 2D array shaped like `lat`/`lon`.

    Uses the local NetCDF archive (data/satellite/raw/) if present, else the
    remote GlobColour product; for months outside the archive range it falls
    back to the monthly climatology (data/satellite/climatology/).
    """
    from bio_params.satellite import chla_month_field
    lat_axis, lon_axis, arr, src = chla_month_field(month, dataset_id=dataset_id)
    print(f"  satellite source: {src}")

    ilat = _nearest_index(lat.ravel(), lat_axis)
    ilon = _nearest_index(((lon.ravel() + 180) % 360) - 180, lon_axis)
    field = arr[ilat, ilon].reshape(lat.shape)

    # Fill cloud-masked NaNs with the nearest valid ROMS pixel (KD-tree).
    bad = ~np.isfinite(field)
    if bad.any() and (~bad).any():
        from scipy.spatial import cKDTree
        good = ~bad
        tree = cKDTree(np.column_stack([lon[good], lat[good]]))
        _, j = tree.query(np.column_stack([lon[bad], lat[bad]]))
        field[bad] = field[good][j]
    return field


def compute_mld_field(temp, salt, z_rho, lat, threshold=0.03, ref_depth=10.0):
    """Mixed-layer depth (m) per ROMS water column from the full T/S profile.

    Mirrors bio_params.profiles.add_mld (density threshold from the level
    nearest ref_depth); returns a 2D (J, I) field, NaN over land/all-invalid.
    """
    N, J, I = temp.shape
    # ROMS orders levels index 0 = deepest .. index -1 = surface. Reverse to
    # surface-first so increasing index == increasing depth (the MLD scan and
    # bio_params.profiles.add_mld both assume surface-first ordering).
    d = (-z_rho)[::-1]                 # positive depth, index 0 = surface
    temp = temp[::-1]
    salt = salt[::-1]
    lat3 = np.broadcast_to(lat[None], (N, J, I))
    sig = sigma0(salt.ravel(), temp.ravel(), d.ravel(), lat3.ravel()).reshape(N, J, I)
    finite = np.isfinite(sig) & np.isfinite(d)
    dd = np.where(finite, np.abs(d - ref_depth), np.inf)
    iref = np.argmin(dd, axis=0)                              # (J, I)
    ref_sig = np.take_along_axis(sig, iref[None], axis=0)[0]  # (J, I)
    level = np.arange(N)[:, None, None]
    exceeded = finite & (level > iref[None]) & (sig > ref_sig[None] + threshold)
    any_ex = exceeded.any(axis=0)
    first = exceeded.argmax(axis=0)                           # 0 if none
    mld = np.take_along_axis(d, first[None], axis=0)[0]
    deepest = np.where(finite, d, -np.inf).max(axis=0)
    mld = np.where(any_ex, mld, deepest)
    return np.where(finite.any(axis=0), mld, np.nan)


def morel_euphotic_depth(chl):
    """Euphotic depth Ze (1% light, m) from surface Chl-a (mg/m3).

    Morel et al. (2007), Case-1 relation: log10(Ze) is a cubic in log10(Chl).
    Ze ~ 12 m at Chl=10, ~33 m at Chl=1, ~85-120 m in oligotrophic water.
    """
    x = np.log10(np.clip(np.asarray(chl, dtype=np.float64), 1e-3, None))
    log_ze = 1.524 - 0.436 * x - 0.0145 * x ** 2 + 0.0186 * x ** 3
    return np.power(10.0, log_ze)


def productive_taper(depth, zp, lo=1.5, hi=3.0):
    """Smooth window (1 -> 0) that confines Chl to the productive layer.

    w = 1 for depth <= lo*zp, a cosine taper to 0 between lo*zp and hi*zp, and 0
    below. `zp` is the productive-layer base = max(euphotic depth, MLD) per point.
    This enforces the physical "Chl -> 0 below the euphotic zone": neither
    BGC-Argo nor GLODAP samples Chl deep enough to constrain it (both are
    photic-zone), so the prior must be imposed, not learned. NaN zp -> w=1.
    """
    depth = np.asarray(depth, dtype=np.float64)
    zp = np.asarray(zp, dtype=np.float64)
    start, end = lo * zp, hi * zp
    w = np.ones_like(depth)
    mid = np.isfinite(zp) & (depth > start) & (depth < end)
    w[mid] = 0.5 * (1.0 + np.cos(np.pi * (depth[mid] - start[mid])
                                 / (end[mid] - start[mid])))
    w[np.isfinite(zp) & (depth >= end)] = 0.0
    return w


def predict_field(model, normalizer, X, device, log_target=False,
                  clip=None, batch=200_000):
    model.eval()
    Xn = normalizer.transform_x(X).astype(np.float32)
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xn), batch):
            xb = torch.from_numpy(Xn[i:i + batch]).to(device)
            preds.append(model(xb).cpu().numpy())
    pred = normalizer.inverse_transform_y(np.concatenate(preds))
    if log_target:
        # Clip in log space first so 10** cannot overflow on extrapolation.
        if clip is not None:
            lo, hi = clip
            log_lo = -np.inf if lo <= 0 else np.log10(lo)
            pred = np.clip(pred, log_lo, np.log10(hi))
        pred = np.power(10.0, pred)
    elif clip is not None:
        pred = np.clip(pred, clip[0], clip[1])
    return pred


def blend_low_salinity(pred, salinity, target):
    """Blend MLP `pred` with the salinity mixing-line regression at low S.

    Returns (blended, n_regression) where the regression dominates for
    S <= BLEND_S_LO, the MLP for S >= BLEND_S_HI, and a linear weight in
    between. Only NO3/PO4 have a regression; others are returned unchanged.
    """
    reg = SALINITY_REGRESSION.get(target)
    if reg is None:
        return pred, 0
    s_clamped = np.maximum(salinity, REGRESSION_S_FLOOR)
    reg_val = reg["slope"] * s_clamped + reg["intercept"]
    reg_val = np.clip(reg_val, 0.0, None)  # nutrients are non-negative
    # Weight w: 1 (pure regression) at S<=LO, 0 (pure MLP) at S>=HI.
    w = (BLEND_S_HI - salinity) / (BLEND_S_HI - BLEND_S_LO)
    w = np.clip(w, 0.0, 1.0)
    blended = w * reg_val + (1.0 - w) * pred
    return blended, int((w > 0).sum())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default="glodap",
                   choices=["glodap", "bgc_argo", "combined"],
                   help="Artifact prefix: glodap_<t>.pt / bgc_argo_<t>.pt / combined_<t>.pt")
    p.add_argument("--targets", nargs="+", default=list(TRACER_META),
                   choices=list(TRACER_META))
    p.add_argument("--low-sal-regression", action="store_true",
                   help="Blend NO3/PO4 with the salinity mixing-line regression "
                        "at low salinity instead of leaving the MLP extrapolation")
    p.add_argument("--tag", default=None,
                   help="Artifact suffix, e.g. --tag satchl loads "
                        "<source>_<target>_satchl.pt (SOCA satellite-anchored)")
    p.add_argument("--no3-model", default="combined",
                   help="NO3 model prefix used to supply the NO3 feature for "
                        "gated Chl-a models (loads <prefix>_NO3.pt)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tag_suffix = f"_{args.tag}" if args.tag else ""
    base_name = f"roms_ini_pred_{args.source}{tag_suffix}" if (args.source != "glodap" or args.tag) else "roms_ini_pred"
    out_dir = OUT_DIR if base_name == "roms_ini_pred" else OUT_DIR.parent / base_name
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  source: {args.source}{tag_suffix}")

    import xarray as xr
    ds = xr.open_dataset(INI, decode_times=False)
    g = xr.open_dataset(GRID, decode_times=False)

    h = g["h"].values.astype(np.float64)
    h = np.where(h <= 0, np.nan, h)
    mask = g["mask_rho"].values
    lon = g["lon_rho"].values
    lat = g["lat_rho"].values
    zeta = ds["zeta"].isel(ocean_time=0).values.astype(np.float64)
    s_rho = ds["s_rho"].values.astype(np.float64)
    Cs_r = ds["Cs_r"].values.astype(np.float64)
    hc = float(ds["hc"].values)
    temp = ds["temp"].isel(ocean_time=0).values.astype(np.float64)
    salt = ds["salt"].isel(ocean_time=0).values.astype(np.float64)

    z_rho = compute_z_rho(h, zeta, s_rho, Cs_r, hc)
    land = mask < 0.5
    J, I = h.shape

    # Pre-build raw features (and salinity) for each depth, shared by all models.
    per_depth = []  # list of dict(label, depth_val, valid 2D bool, X raw, low_sal 2D bool)
    for depth in TARGET_DEPTHS:
        if depth == 0.0:
            T2 = temp[-1].copy()
            S2 = salt[-1].copy()
            d2 = np.abs(z_rho[-1])           # actual surface-cell depth (m)
            label = "surface"
        else:
            T2 = interp_to_depth(temp, z_rho, -depth)
            S2 = interp_to_depth(salt, z_rho, -depth)
            d2 = np.full((J, I), depth)
            label = f"{depth:.0f} m"

        valid = np.isfinite(T2) & np.isfinite(S2) & (~land)
        jj, ii = np.where(valid)
        feat_df = pd.DataFrame({
            "latitude": lat[jj, ii],
            "longitude": lon[jj, ii],
            "depth": d2[jj, ii],
            "temperature": T2[jj, ii],
            "salinity": S2[jj, ii],
        })
        X = build_features(feat_df).to_numpy()
        per_depth.append(dict(
            label=label, valid=valid, jj=jj, ii=ii, X=X, feat_df=feat_df,
            low_sal=(S2 < SALT_OPEN_OCEAN_MIN),
            salinity_pts=S2[jj, ii],
        ))

    # Lazily-loaded satellite surface Chl-a field, MLD field, and NO3 model
    # (the gated Chl-a model takes NO3 as a feature: predict it first, feed it in).
    sat_field = {"loaded": False, "data": None}
    mld_field = {"loaded": False, "data": None}
    no3_holder = {"model": None, "norm": None}

    for tgt in args.targets:
        art = MODEL_DIR / f"{args.source}_{tgt}{tag_suffix}.pt"
        if not art.exists():
            print(f"skip {tgt}: artifact missing ({art})")
            continue
        model, normalizer, meta = load_artifact(art, map_location=device)
        model.to(device)
        extra = meta["extra"]
        cv_r2 = (extra.get("cv_r2_mean") or extra.get("cv_abs_r2_mean")
                 or extra.get("cv_shape_r2_mean") or float("nan"))
        log_target = bool(extra.get("log_target", False))
        if extra.get("include_season"):
            print(f"skip {tgt}: model needs season features (no time on ROMS ini)")
            continue
        surface_chla = bool(extra.get("surface_chla", False))
        surface_chla_log = bool(extra.get("surface_chla_log", True))
        include_mld = bool(extra.get("include_mld", False))
        include_no3 = bool(extra.get("include_no3", False))
        relative_target = bool(extra.get("relative_target", False))
        output_gate = bool(extra.get("output_gate", False))
        gate_ik = float(extra.get("gate_ik", 0.005))
        ik_head = bool(extra.get("ik_head", False))
        gate_ik_ref = float(extra.get("gate_ik_ref", 0.02))
        seasonal_light = bool(extra.get("seasonal_light", False))
        base_amp = bool(extra.get("base_amp", False))
        a_max = float(extra.get("a_max", 5.0))
        base_profile = BaseProfile.from_dict(extra["base_profile"]) if base_amp else None
        # models that force deep->0 by construction (no productive-layer taper,
        # surface == satellite): the light gate and the base x amplification model.
        structural = output_gate or base_amp
        rel_cap = float(extra.get("rel_cap", 20.0))
        # Relative models multiply by the satellite surface field for amplitude.
        need_sat = surface_chla or relative_target
        if need_sat and not sat_field["loaded"]:
            month = _ini_month(INI)
            print(f"  loading satellite surface Chl-a ({SAT_DATASET}, {month}) ...")
            sat_field["data"] = load_roms_surface_chla(lat, lon, month)
            sat_field["loaded"] = True
        if include_mld and not mld_field["loaded"]:
            print("  computing mixed-layer depth field ...")
            mld_field["data"] = compute_mld_field(temp, salt, z_rho, lat)
            mld_field["loaded"] = True
        if include_no3 and no3_holder["model"] is None:
            no3_art = MODEL_DIR / f"{args.no3_model}_NO3.pt"
            print(f"  loading NO3 model for the nutrient feature ({no3_art.name}) ...")
            nm, nn, _ = load_artifact(no3_art, map_location=device)
            no3_holder.update(model=nm.to(device), norm=nn)
        m = TRACER_META[tgt]
        use_reg = args.low_sal_regression and tgt in SALINITY_REGRESSION
        print(f"{tgt}: CV R2={cv_r2:.4f}  log_target={log_target}"
              + ("  [relative profile x satellite]" if relative_target else "")
              + ("  [base x amplification]" if base_amp else "")
              + ("  [satellite surface anchor]" if surface_chla and not relative_target else "")
              + ("  [+MLD]" if include_mld else "")
              + ("  [low-sal regression blend ON]" if use_reg else ""))

        def _no3_for(pd_):
            # NO3 feature: predict it from the base features with the NO3 model
            # (cache per depth). Clip to a physical range.
            if "no3_pred" not in pd_:
                pd_["no3_pred"] = predict_field(no3_holder["model"], no3_holder["norm"],
                                                pd_["X"], device, clip=(0.0, 60.0))
            return pd_["no3_pred"]

        def _build_X(pd_):
            if not (surface_chla or include_mld or include_no3):
                return pd_["X"]
            df2 = pd_["feat_df"].copy()
            if surface_chla:
                df2["surface_chla"] = sat_field["data"][pd_["jj"], pd_["ii"]]
            if include_mld:
                df2["mld"] = mld_field["data"][pd_["jj"], pd_["ii"]]
            if include_no3:
                df2["NO3"] = _no3_for(pd_)
            return build_features(df2, include_surface_chla=surface_chla,
                                  surface_chla_log=surface_chla_log,
                                  include_mld=include_mld,
                                  include_no3=include_no3).to_numpy()

        def _struct_rel(pd_):
            # Relative profile for models that force deep->0 structurally.
            # base_amp: rel = base(z; sat surface Chl) * A_max*sigmoid(MLP), with
            #   the data-derived typical base decaying to 0 (bounded a => deep->0).
            # gate: rel = softplus(g) * tanh(rel_light/Ik); rel_light = exp(-Kd*z),
            #   Kd from satellite surface Chl -> 0 in the dark for any Ik.
            jj, ii = pd_["jj"], pd_["ii"]
            depth = pd_["feat_df"]["depth"].to_numpy()
            out = predict_field(model, normalizer, _build_X(pd_), device, clip=None)
            if base_amp:
                a = a_max * (1.0 / (1.0 + np.exp(-out)))
                base_col = base_profile.eval(sat_field["data"][jj, ii], depth)
                return np.clip(base_col * a, 0.0, rel_cap)
            if ik_head:
                g = np.logaddexp(0.0, out[:, 0])                      # softplus
                ik = gate_ik_ref * np.exp(np.clip(out[:, 1], -5.0, 5.0))
            else:
                g = np.logaddexp(0.0, out); ik = gate_ik
            kd = np.log(100.0) / morel_euphotic_depth(sat_field["data"][jj, ii])
            rl = np.exp(-kd * depth)
            if seasonal_light:
                rl = daily_insolation_factor(lat[jj, ii], _ini_doy(INI)) * rl
            return np.clip(g * np.tanh(rl / ik), 0.0, rel_cap)

        # For relative models, re-anchor the predicted profile to its own
        # predicted surface so the surface maps EXACTLY to the satellite value
        # (a regression does not output rel==1 at the surface by itself).
        rel_surf = None
        zp_field = None
        if relative_target:
            ps = per_depth[0]  # TARGET_DEPTHS[0] == 0.0 -> surface
            r0 = (_struct_rel(ps) if structural
                  else np.clip(predict_field(model, normalizer, _build_X(ps), device,
                                             clip=None), 0.0, rel_cap))
            rel_surf = np.full((J, I), np.nan)
            # Tiny floor only to avoid divide-by-zero; the surface ratio is then
            # r0/r0 == 1 exactly (so surface maps exactly to the satellite value).
            rel_surf[ps["jj"], ps["ii"]] = np.maximum(r0, 1e-6)
            # Plain relative models need the productive-layer taper to kill deep
            # Chl; structural models (gate / base x amp) handle it by construction.
            if not structural:
                zp_field = np.fmax(morel_euphotic_depth(sat_field["data"]),
                                   mld_field["data"])

        fig, axes = plt.subplots(
            1, len(TARGET_DEPTHS),
            figsize=(4.6 * len(TARGET_DEPTHS), 5.4),
            constrained_layout=True,
        )
        for ax, pd_ in zip(axes, per_depth):
            field = np.full((J, I), np.nan)
            jj, ii = pd_["jj"], pd_["ii"]
            X = _build_X(pd_)
            if relative_target:
                if pd_["label"] == "surface":
                    # The satellite IS the surface truth (ratio == 1 by definition).
                    pred = sat_field["data"][jj, ii].copy()
                else:
                    rel = _struct_rel(pd_) if structural else np.clip(
                        predict_field(model, normalizer, X, device, clip=None),
                        0.0, rel_cap)
                    # Re-anchor to the predicted surface; clip the ratio so a tiny
                    # predicted surface cannot blow the deep value up.
                    rel = np.clip(rel / rel_surf[jj, ii], 0.0, rel_cap)
                    pred = rel * sat_field["data"][jj, ii]
                    if not structural:
                        # Productive-layer taper: Chl -> 0 below the euphotic/mixed
                        # layer (the gated model achieves this structurally instead).
                        pred = pred * productive_taper(
                            pd_["feat_df"]["depth"].to_numpy(), zp_field[jj, ii])
                if m.get("clip") is not None:
                    pred = np.clip(pred, m["clip"][0], m["clip"][1])
            else:
                pred = predict_field(model, normalizer, X, device,
                                     log_target=log_target, clip=m.get("clip"))
            if use_reg:
                pred, _ = blend_low_salinity(pred, pd_["salinity_pts"], tgt)
            field[jj, ii] = pred

            finite = field[np.isfinite(field)]
            vmin, vmax = (np.percentile(finite, [2, 98])
                          if finite.size else (None, None))

            pcm = ax.pcolormesh(
                lon, lat, np.ma.masked_invalid(field),
                cmap=m["cmap"], vmin=vmin, vmax=vmax, shading="auto",
            )
            ax.set_facecolor("0.8")
            # Hatch the low-salinity region: extrapolation (xx) by default, or
            # regression-filled (//) when the salinity blend is applied.
            low = pd_["low_sal"] & pd_["valid"]
            if low.any():
                hatch = "//" if use_reg else "xx"
                ax.contourf(lon, lat, low.astype(float), levels=[0.5, 1.5],
                            colors="none", hatches=[hatch])
            n_low = int(low.sum())
            if n_low:
                note = (f"\n(S<{SALT_OPEN_OCEAN_MIN:.0f}: {n_low:,} pts "
                        + ("regression-filled //)" if use_reg else "hatched xx)"))
            else:
                note = ""
            ax.set_title(f"{pd_['label']}{note}", fontsize=10)
            ax.set_xlabel("Longitude")
            if ax is axes[0]:
                ax.set_ylabel("Latitude")
            ax.set_aspect("equal")
            cb = fig.colorbar(pcm, ax=ax, orientation="horizontal",
                              pad=0.08, fraction=0.05)
            cb.set_label(f"{tgt} ({m['unit']})", fontsize=9)

        fig.suptitle(
            f"{m['long']} ({tgt}) — {args.source} MLP prediction on ROMS grid\n"
            f"{INI.name}  |  honest spatial-CV R2={cv_r2:.4f}  "
            f"(hatch = S<{SALT_OPEN_OCEAN_MIN:.0f} PSU: extrapolation)",
            fontsize=12,
        )
        out = out_dir / f"pred_{tgt}_depths.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")

    ds.close()
    g.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
