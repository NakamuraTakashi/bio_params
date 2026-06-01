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


def scatter(ax, obs, pred, title, log_axes=True):
    if log_axes:
        lo = max(1e-3, float(min(obs.min(), pred.min())))
        hi = float(max(obs.max(), pred.max()))
        extent = (np.log10(lo), np.log10(hi), np.log10(lo), np.log10(hi))
        kw = dict(xscale="log", yscale="log")
    else:
        lo = 0.0
        hi = float(np.percentile(np.concatenate([obs, pred]), 99))
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
    p.add_argument("--source", default="combined")
    p.add_argument("--tag", default="gated")
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    art = MODEL_DIR / f"{args.source}_Chla_{args.tag}.pt"
    model, norm, meta = load_artifact(art, map_location=device); model.to(device)
    e = meta["extra"]; ik = float(e["gate_ik"]); rel_cap = float(e.get("rel_cap", 20.0))
    ik_head = bool(e.get("ik_head", False)); ik_ref = float(e.get("gate_ik_ref", 0.02))
    print(f"loaded {art.name}  Ik={ik:.4f}  ik_head={ik_head}  "
          f"shape CV R2={e.get('cv_shape_r2_mean'):.4f}")

    df = load_chla_no3(args.source, glodap_csv=CSV, sprof_dir=SPROF)
    df = add_mld(df)
    df = add_relative_target(df, "Chla", rel_cap=rel_cap)
    df = df[np.isfinite(df["mld"]) & np.isfinite(df["NO3"])].reset_index(drop=True)
    df["kd"] = kd_from_surface_chl(df["Chla_surf"].to_numpy())

    X = build_features(df, include_mld=True, include_no3=True).to_numpy()
    Xn = norm.transform_x(X).astype(np.float32)
    rl = np.exp(-df["kd"].to_numpy() * df["depth"].to_numpy())
    with torch.no_grad():
        out = model(torch.from_numpy(Xn).to(device))
        if ik_head:
            g = F.softplus(out[:, 0]).cpu().numpy()
            ik_local = ik_ref * np.exp(np.clip(out[:, 1].cpu().numpy(), -5.0, 5.0))
        else:
            g = F.softplus(out.squeeze(-1)).cpu().numpy(); ik_local = ik
    rel = np.clip(g * np.tanh(rl / ik_local), 0.0, rel_cap)
    pred = rel * df["Chla_surf"].to_numpy()            # absolute, in-situ surface
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
    pred_plot = np.maximum(pred, 1e-3)
    stats = {}
    for lbl, mask in [("global", keep), ("Japan box", jp)]:
        prod = mask & (obs >= PROD_THR)
        r2 = r2_log(obs[prod], pred_plot[prod])
        rmse = float(np.sqrt(np.mean((pred[mask] - obs[mask]) ** 2)))
        stats[lbl] = (mask, r2, rmse)
        print(f"  {lbl}: n={mask.sum():,}  log-R2(obs>={PROD_THR})={r2:.4f}  RMSE={rmse:.3g}")

    for log_axes, fname in [(True, f"scatter_{args.source}_{args.tag}.png"),
                            (False, f"scatter_{args.source}_{args.tag}_linear.png")]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
        for ax, lbl in zip(axes, ["global", "Japan box"]):
            mask, r2, rmse = stats[lbl]
            axt = "log-log" if log_axes else "linear"
            scatter(ax, obs[mask], pred_plot[mask],
                    f"{args.source}_{args.tag} ({lbl}, {axt})\n"
                    f"n={mask.sum():,}  log-R2(Chl>={PROD_THR})={r2:.3f}  "
                    f"RMSE={rmse:.3g} mg/m3", log_axes=log_axes)
        fig.suptitle("Gated relative Chl-a: model (rel x in-situ surface) vs observation",
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT_DIR / fname, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"saved {OUT_DIR / fname}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
