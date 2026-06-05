"""Prediction-vs-observation scatter for the gated relative Chl-a model.

Reconstructs absolute Chl-a as  rel_pred(z) * Chla_surface(in-situ)  where
rel_pred = softplus(MLP) * tanh(rel_light/Ik), and scatters it against the
observed Chl-a (log-log), for the whole training domain and the Japan box.
Using the in-situ surface isolates the model's SHAPE skill (the satellite-vs-
in-situ surface mismatch is a separate data issue).

Usage:
    uv run python scripts/plot_gated_predictions.py --source combined --tag gated
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
import torch.nn.functional as F

from bio_params.features import build_features
from bio_params.loaders.chla_no3 import load_chla_no3
from bio_params.persist import load_artifact
from bio_params.profiles import add_mld, add_relative_target, kd_from_surface_chl

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
SPROF = ROOT / "data" / "bgc_argo" / "raw" / "floats"
MODEL_DIR = ROOT / "models" / "pretrained"
OUT_DIR = ROOT / "figures" / "gated"
JAPAN = dict(lat0=20.0, lat1=50.0, lon0=120.0, lon1=160.0)


def r2_log(obs, pred):
    lo, lp = np.log10(obs), np.log10(pred)
    ss = float(((lp - lo) ** 2).sum()); tot = float(((lo - lo.mean()) ** 2).sum())
    return 1.0 - ss / tot if tot > 0 else float("nan")


def scatter(ax, obs, pred, title, log_axes=True, lin_max=None, log_range=None):
    if log_axes:
        if log_range is not None:
            lo, hi = float(log_range[0]), float(log_range[1])
        else:
            lo = max(1e-3, float(min(obs.min(), pred.min())))
            hi = float(max(obs.max(), pred.max()))
        extent = (np.log10(lo), np.log10(hi), np.log10(lo), np.log10(hi))
        kw = dict(xscale="log", yscale="log")
    else:
        lo = 0.0
        hi = float(lin_max) if lin_max else float(np.percentile(np.concatenate([obs, pred]), 99))
        extent = (lo, hi, lo, hi); kw = {}
    hb = ax.hexbin(obs, pred, gridsize=70, bins="log", cmap="viridis",
                   mincnt=1, extent=extent, **kw)
    plt.colorbar(hb, ax=ax).set_label("log10(count)")
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="1:1")
    if log_axes:
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("Observed Chl-a (mg/m3)"); ax.set_ylabel("Model Chl-a (mg/m3)")
    ax.set_title(title, fontsize=10); ax.legend(loc="upper left", fontsize=8)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default="combined", help="model source (artifact prefix)")
    p.add_argument("--tag", default="gated")
    p.add_argument("--eval-source", default=None,
                   help="data source to validate against (default = --source). "
                        "Use 'glodap' for absolute validation (GLODAP surface "
                        "matches satellite; BGC-Argo is biased in absolute terms).")
    p.add_argument("--box", type=float, nargs=4, default=None,
                   metavar=("LON0", "LON1", "LAT0", "LAT1"),
                   help="restrict eval data to a region, e.g. --box 120 160 20 50")
    p.add_argument("--cutoff", type=float, default=None,
                   help="override the plain-model hard 0-cutoff depth (m); 0 = none")
    p.add_argument("--log-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="fixed log-log axis range, e.g. --log-range 1e-4 1e2 "
                        "(overview at max range; saved as ..._logrange.png)")
    p.add_argument("--linear-max", type=float, nargs="+", default=None,
                   help="upper limit(s) for the linear-axis scatter; one figure per "
                        "value, named ..._linear_max<N>.png (e.g. --linear-max 20 10 5)")
    args = p.parse_args()
    eval_source = args.eval_source or args.source
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    art = MODEL_DIR / f"{args.source}_Chla_{args.tag}.pt"
    model, norm, meta = load_artifact(art, map_location=device); model.to(device)
    e = meta["extra"]; rel_cap = float(e.get("rel_cap", 20.0))
    base_amp = bool(e.get("base_amp", False))
    is_plain = not (base_amp or e.get("output_gate") or e.get("relative_target"))
    print(f"loaded {art.name}  base_amp={base_amp}  plain={is_plain}")

    print(f"  evaluating against {eval_source} observations")
    df = load_chla_no3(eval_source, glodap_csv=CSV, sprof_dir=SPROF,
                       box=tuple(args.box) if args.box else None)
    df = add_mld(df)
    df = add_relative_target(df, "Chla", rel_cap=rel_cap)
    df = df[np.isfinite(df["mld"]) & np.isfinite(df["NO3"])].reset_index(drop=True)

    if is_plain:                                       # direct-regression model
        Xp = build_features(df, include_sigma_theta=bool(e.get("include_sigma", False)),
                            include_mld=bool(e.get("include_mld", False)),
                            include_no3=bool(e.get("include_no3", False)),
                            include_surface_chla=bool(e.get("surface_chla", False)),
                            surface_chla_log=bool(e.get("surface_chla_log", True))).to_numpy()
        if e.get("nutricline_features") or e.get("strat_features"):
            from bio_params.profiles import add_structure_descriptors
            d2 = add_structure_descriptors(df)
            if e.get("nutricline_features"):
                Xp = np.column_stack([Xp, np.log(d2["z_nutr"].to_numpy() + 1.0),
                                      d2["nutr_max"].to_numpy()])
            if e.get("strat_features"):
                Xp = np.column_stack([Xp, np.log(d2["z_pyc"].to_numpy() + 1.0),
                                      d2["strat_max"].to_numpy()])
        with torch.no_grad():
            o = model(torch.as_tensor(norm.transform_x(Xp).astype(np.float32),
                                      device=device)).cpu().numpy().ravel()
        pred = norm.inverse_transform_y(o)
        if e.get("log_target"):
            pred = 10.0 ** pred
        pred = np.clip(pred, 0.0, None)
        cut = args.cutoff if args.cutoff is not None else e.get("cutoff_depth")
        if cut:
            pred = np.where(df["depth"].to_numpy() > float(cut), 0.0, pred)
    else:
        sfc = bool(e.get("surface_chla", False))
        if sfc:
            # surface-Chl models train on the daily satellite matchup; feed the same.
            md = ROOT / "data" / "bgc_argo" / "processed" / "satchl_matchup_daily_combined.parquet"
            m = pd.read_parquet(md)[["latitude", "longitude", "time", "surface_chla"]]
            m["time"] = pd.to_datetime(m["time"]); df["time"] = pd.to_datetime(df["time"])
            df = df.merge(m, on=["latitude", "longitude", "time"], how="left")
            df = df[np.isfinite(df["surface_chla"])].reset_index(drop=True)
        X = build_features(df, include_mld=True, include_no3=True,
                           include_surface_chla=sfc,
                           surface_chla_log=bool(e.get("surface_chla_log", True))).to_numpy()
        Xn = norm.transform_x(X).astype(np.float32)
        with torch.no_grad():
            out = model(torch.from_numpy(Xn).to(device)).cpu().numpy()
        if base_amp:                                   # base x amplification
            from bio_params.base_profile import BaseProfile
            a_max = float(e.get("a_max", 5.0))
            bp = BaseProfile.from_dict(e["base_profile"])
            base_col = bp.eval(df["Chla_surf"].to_numpy(), df["depth"].to_numpy())
            a = a_max * (1.0 / (1.0 + np.exp(-out.ravel())))
            rel = np.clip(base_col * a, 0.0, rel_cap)
        else:                                          # light-gated
            ik = float(e["gate_ik"]); ik_head = bool(e.get("ik_head", False))
            ik_ref = float(e.get("gate_ik_ref", 0.02)); fixed_ze = e.get("gate_fixed_ze")
            kd = (np.full(len(df), np.log(100.0) / float(fixed_ze)) if fixed_ze
                  else kd_from_surface_chl(df["Chla_surf"].to_numpy()))
            rl = np.exp(-kd * df["depth"].to_numpy())
            if ik_head:
                g = np.logaddexp(0.0, out[:, 0]); ik_local = ik_ref * np.exp(np.clip(out[:, 1], -5, 5))
            else:
                g = np.logaddexp(0.0, out.ravel()); ik_local = ik
            rel = np.clip(g * np.tanh(rl / ik_local), 0.0, rel_cap)
        pred = rel * df["Chla_surf"].to_numpy()        # absolute, in-situ surface
    obs = df["Chla"].to_numpy()
    lat = df["latitude"].to_numpy()
    lon = ((df["longitude"].to_numpy() + 180) % 360) - 180
    keep = np.isfinite(pred) & np.isfinite(obs) & (pred > 0) & (obs > 0)

    jp = keep & (lat >= JAPAN["lat0"]) & (lat <= JAPAN["lat1"]) \
        & (lon >= JAPAN["lon0"]) & (lon <= JAPAN["lon1"])
    # The gated model predicts ~0 below the lit zone (correct), but observed Chl
    # there is tiny-positive noise; log-R2 over those points is meaningless, so
    # the log-R2 is reported on the meaningful Chl band (obs >= PROD_THR). RMSE
    # (linear) is over all points.
    PROD_THR = 0.02
    pred_plot = np.maximum(pred, 1e-5)   # low floor so a wide log range shows the spread
    # With --box the data is already restricted to that region, so a separate
    # "Japan box" panel would be identical; show a single regional panel instead.
    if args.box:
        lab = (f"box {args.box[0]:.0f}-{args.box[1]:.0f}E "
               f"{args.box[2]:.0f}-{args.box[3]:.0f}N")
        region_panels = [(lab, keep)]
    else:
        region_panels = [("global", keep), ("Japan box", jp)]
    stats = {}
    for lbl, mask in region_panels:
        prod = mask & (obs >= PROD_THR)
        r2 = r2_log(obs[prod], pred_plot[prod])
        rmse = float(np.sqrt(np.mean((pred[mask] - obs[mask]) ** 2)))
        stats[lbl] = (mask, r2, rmse)
        print(f"  {lbl}: n={mask.sum():,}  log-R2(obs>={PROD_THR})={r2:.4f}  RMSE={rmse:.3g}")

    evsfx = f"_eval-{eval_source}" if eval_source != args.source else ""
    boxsfx = "_box" if args.box else ""
    base = f"scatter_{args.source}_{args.tag}{evsfx}{boxsfx}"
    # One log-axis figure, then one linear figure per requested upper limit
    # (distinct filename per limit so earlier figures are not overwritten).
    panels = [(True, None, None, f"{base}.png")]
    if args.log_range:
        panels.append((True, None, tuple(args.log_range), f"{base}_logrange.png"))
    for lm in (args.linear_max if args.linear_max else [None]):
        sfx = "_linear" if lm is None else f"_linear_max{int(lm)}"
        panels.append((False, lm, None, f"{base}{sfx}.png"))

    labels = [lbl for lbl, _ in region_panels]
    for log_axes, lin_max, lrange, fname in panels:
        fig, axes = plt.subplots(1, len(labels), figsize=(6.5 * len(labels), 6.2),
                                 squeeze=False)
        axes = axes.ravel()
        for ax, lbl in zip(axes, labels):
            mask, r2, rmse = stats[lbl]
            axt = (f"log-log {lrange[0]:g}-{lrange[1]:g}" if lrange else "log-log") \
                if log_axes else (f"linear 0-{int(lin_max)}" if lin_max else "linear")
            scatter(ax, obs[mask], pred_plot[mask],
                    f"{args.source}_{args.tag} vs {eval_source} obs ({lbl}, {axt})\n"
                    f"n={mask.sum():,}  log-R2(Chl>={PROD_THR})={r2:.3f}  "
                    f"RMSE={rmse:.3g} mg/m3", log_axes=log_axes, lin_max=lin_max,
                    log_range=lrange)
        recon = "direct regression" if is_plain else "rel x in-situ surface"
        fig.suptitle(f"Chl-a ({args.source}_{args.tag}): model ({recon}) "
                     f"vs {eval_source} observation", fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT_DIR / fname, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"saved {OUT_DIR / fname}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
