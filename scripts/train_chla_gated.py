"""Relative Chl-a model with a multiplicative LIGHT OUTPUT GATE (physical).

The model predicts a non-negative shape factor g = softplus(MLP(features)) and
the relative profile is  rel(z) = g(features) * rel_light(z),  with the
Beer-Lambert relative light  rel_light(z) = exp(-Kd*z)  (Kd from surface Chl).
Because rel_light -> 0 in the dark, rel -> 0 below the lit zone *by construction*
- no hard cutoff and no spurious deep Chl, learned end-to-end. Features feeding g
are the 7 base + MLD + NO3 (raw NO3 is fine: the gate zeroes the dark anyway).

Target: surface-normalized rel = Chla / Chla_surface (clipped [0, rel_cap]).
Amplitude is restored at inference from the satellite surface field.

Custom GPU-resident training loop (the gate is applied before the loss, so the
generic Normalizer/train cannot be reused). The saved Normalizer standardizes X
only (identity on y); inference must apply softplus and the rel_light gate.

Artifacts: models/pretrained/<source>_Chla_gated[_<tag>].pt  (extra.output_gate=True)
CV metrics: data/<...>/processed/cv_Chla_gated[_<tag>].json

Usage:
    uv run python scripts/train_chla_gated.py --source combined --per-source-profiles 10000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from bio_params.cv import spatial_block_split
from bio_params.dataset import Normalizer
from bio_params.features import build_features, feature_names
from bio_params.loaders.bgc_argo import attach_surface_chla
from bio_params.loaders.chla_no3 import load_chla_no3
from bio_params.model import MLP, MLPConfig
from bio_params.persist import save_artifact
from bio_params.profiles import add_mld, add_relative_target, kd_from_surface_chl

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "data" / "glodap" / "raw" / "GLODAPv2.2023_Merged_Master_File.csv"
DEFAULT_SPROF = ROOT / "data" / "bgc_argo" / "raw" / "floats"
MODEL_DIR = ROOT / "models" / "pretrained"
PROC = ROOT / "data" / "bgc_argo" / "processed"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default="combined", choices=["bgc_argo", "glodap", "combined"])
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--sprof-dir", type=Path, default=DEFAULT_SPROF)
    p.add_argument("--matchup", type=Path, default=None)
    p.add_argument("--per-source-profiles", type=int, default=None)
    p.add_argument("--rel-cap", type=float, default=20.0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--block-deg", type=float, default=5.0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-hidden-layers", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-final", action="store_true")
    p.add_argument("--ik-head", action="store_true",
                   help="learn a per-sample (environment-dependent) Ik via a 2nd "
                        "MLP output instead of one global Ik")
    p.add_argument("--fixed-ze", type=float, default=None,
                   help="TEST: fix euphotic depth Ze (m) constant (Kd=ln(100)/Ze) "
                        "instead of Morel(surface Chl); opens the gate deeper")
    p.add_argument("--tag", default=None)
    return p.parse_args()


def _r2(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    ss = float(((pred - obs) ** 2).sum()); tot = float(((obs - obs.mean()) ** 2).sum())
    return 1.0 - ss / tot if tot > 0 else float("nan")


IK_INIT = 0.005   # initial light-saturation scale (single-Ik mode)
IK_REF = 0.02     # reference scale for the per-sample Ik head: Ik = IK_REF*exp(clip(h))


def _ik_from_head(h):
    """Per-sample Ik from the 2nd MLP output (torch), bounded around IK_REF."""
    return IK_REF * torch.exp(torch.clamp(h, -5.0, 5.0))


def train_gated(cfg, Xn, rl, y, tr, va, device, *, epochs, batch, lr, patience,
                ik_init=IK_INIT, ik_head=False):
    """Train rel = softplus(g) * tanh(rel_light / Ik); Ik learned.

    The tanh light-saturation gate (Jassby & Platt form) lets the model keep a
    deep-chlorophyll maximum (gate ~1 where light is adequate) while forcing
    rel -> 0 in the dark (rel_light -> 0 => tanh -> 0, for any Ik > 0). With
    `ik_head`, Ik is a per-sample 2nd MLP output (environment/depth-dependent),
    so it can be small in deep-DCM regions; otherwise one global Ik is learned.
    """
    Xt = torch.as_tensor(Xn, dtype=torch.float32, device=device)
    rt = torch.as_tensor(rl, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    tr_i = torch.as_tensor(tr, dtype=torch.long, device=device)
    va_i = torch.as_tensor(va, dtype=torch.long, device=device)
    model = MLP(cfg).to(device)
    params = list(model.parameters())
    log_ik = None
    if not ik_head:
        log_ik = torch.tensor(float(np.log(ik_init)), device=device, requires_grad=True)
        params = params + [log_ik]
    opt = torch.optim.Adam(params, lr=lr)

    def gate_rel(idx):
        out = model(Xt[idx])
        if ik_head:
            g = F.softplus(out[:, 0]); ik = _ik_from_head(out[:, 1])
        else:
            g = F.softplus(out); ik = torch.exp(log_ik)
        return g * torch.tanh(rt[idx] / ik), ik

    best, best_state, best_ik, bad = float("inf"), None, ik_init, 0
    for ep in range(epochs):
        model.train()
        perm = tr_i[torch.randperm(len(tr_i), device=device)]
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            loss = F.mse_loss(gate_rel(b)[0], yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vp, ik = gate_rel(va_i)
            vl = float(F.mse_loss(vp, yt[va_i]))
        if vl < best - 1e-7:
            best = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_ik = float(ik.mean()) if ik_head else float(torch.exp(log_ik)); bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ik, ep + 1


def predict_rel(model, Xn, rl, device, ik, rel_cap, ik_head=False, batch=200_000):
    model.eval()
    rels = []
    with torch.no_grad():
        for i in range(0, len(Xn), batch):
            xb = torch.as_tensor(Xn[i:i + batch], dtype=torch.float32, device=device)
            out = model(xb)
            if ik_head:
                g = F.softplus(out[:, 0]); ikb = _ik_from_head(out[:, 1])
                rel = g * torch.tanh(torch.as_tensor(rl[i:i + batch], dtype=torch.float32, device=device) / ikb)
            else:
                g = F.softplus(out)
                rel = g * torch.tanh(torch.as_tensor(rl[i:i + batch], dtype=torch.float32, device=device) / ik)
            rels.append(rel.cpu().numpy())
    return np.clip(np.concatenate(rels), 0.0, rel_cap)


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    matchup = args.matchup or (PROC / ("satchl_matchup_combined.parquet"
              if args.source == "combined" else "satchl_matchup.parquet"))
    suffix = f"_{args.tag}" if args.tag else ""
    print(f"=== gated relative Chla  source={args.source}{suffix} ===")

    df = load_chla_no3(args.source, glodap_csv=args.csv, sprof_dir=args.sprof_dir)
    print(f"  loaded {len(df):,} co-located rows")
    df = add_mld(df)
    df = add_relative_target(df, "Chla", rel_cap=args.rel_cap)
    df = df[np.isfinite(df["mld"]) & np.isfinite(df["NO3"])].reset_index(drop=True)
    if args.fixed_ze:
        df["kd"] = np.full(len(df), np.log(100.0) / args.fixed_ze)
        print(f"  TEST: fixed Ze={args.fixed_ze:.0f} m -> Kd={np.log(100.0)/args.fixed_ze:.4f} /m (constant)")
    else:
        df["kd"] = kd_from_surface_chl(df["Chla_surf"].to_numpy())

    if args.per_source_profiles and "source" in df.columns:
        keys = ["latitude", "longitude", "time"]; rng = np.random.default_rng(args.seed)
        parts = []
        for src, sub in df.groupby("source"):
            pr = sub[keys].drop_duplicates()
            if len(pr) > args.per_source_profiles:
                pr = pr.iloc[np.sort(rng.choice(len(pr), args.per_source_profiles, replace=False))]
            parts.append(sub.merge(pr, on=keys, how="inner"))
        df = pd.concat(parts, ignore_index=True)
        ng = int((df.source == "glodap").sum()); na = int((df.source == "bgc_argo").sum())
        print(f"  balanced <= {args.per_source_profiles} profiles/source: {len(df):,} rows (glodap={ng:,}, bgc={na:,})")

    df = attach_surface_chla(df, matchup)
    fnames = feature_names(include_mld=True, include_no3=True)
    X = build_features(df, include_mld=True, include_no3=True).to_numpy()
    rl = np.exp(-df["kd"].to_numpy() * df["depth"].to_numpy())   # gate
    y = df["Chla_rel"].to_numpy()
    chla_abs = df["Chla"].to_numpy(); sat_surf = df["surface_chla"].to_numpy()
    lat = df["latitude"].to_numpy(); lon = df["longitude"].to_numpy()
    print(f"  X {X.shape}  features {fnames}  (gate=rel_light, output gate ON)")

    cfg = MLPConfig(in_dim=X.shape[1], hidden=args.hidden,
                    n_hidden_layers=args.n_hidden_layers,
                    out_dim=2 if args.ik_head else 1)
    if args.ik_head:
        print("  Ik head ON: per-sample Ik = IK_REF*exp(MLP_out[1])")
    folds = []
    print(f"\n--- {args.folds} folds x {args.block_deg} deg blocks ---")
    for k, tr, va in spatial_block_split(lat, lon, block_deg=args.block_deg, n_folds=args.folds, seed=args.seed):
        norm = Normalizer.fit(X[tr], y[tr])
        norm = Normalizer(x_mean=norm.x_mean, x_std=np.where(norm.x_std == 0, 1.0, norm.x_std),
                          y_mean=0.0, y_std=1.0)  # identity y; gate is in native rel space
        Xn = norm.transform_x(X)
        model, ik, eps = train_gated(cfg, Xn, rl, y, tr, va, device, epochs=args.epochs,
                                     batch=args.batch_size, lr=args.lr, patience=args.patience,
                                     ik_head=args.ik_head)
        rel_pred = predict_rel(model, Xn[va], rl[va], device, ik, args.rel_cap,
                               ik_head=args.ik_head)
        shape_r2 = _r2(y[va], rel_pred)
        msk = np.isfinite(sat_surf[va])
        abs_r2 = _r2(chla_abs[va][msk], rel_pred[msk] * sat_surf[va][msk]) if msk.any() else float("nan")
        folds.append(dict(fold=k, shape_r2=shape_r2, abs_r2=abs_r2, ik=ik, n=int(len(va)), epochs=eps))
        print(f"  fold {k}: shape R2={shape_r2:.4f}  abs R2={abs_r2:.4f}  Ik={ik:.4f}  (epochs={eps})")

    sr2 = np.array([f["shape_r2"] for f in folds]); ar2 = np.array([f["abs_r2"] for f in folds])
    iks = np.array([f["ik"] for f in folds])
    print(f"\n=== CV ===  shape R2 mean {sr2.mean():.4f} median {np.median(sr2):.4f} | "
          f"abs R2 mean {ar2.mean():.4f} | Ik mean {iks.mean():.4f} (range {iks.min():.4f}-{iks.max():.4f})")

    metrics_dir = ROOT / "data" / ("combined" if args.source == "combined" else "bgc_argo") / "processed"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"cv_Chla_gated{suffix}.json").write_text(json.dumps(dict(
        target="Chla", source=args.source, output_gate=True, relative_target=True,
        include_mld=True, include_no3=True, feature_names=fnames, rel_cap=args.rel_cap,
        gate="tanh(rel_light/Ik)", ik_head=args.ik_head, ik_mean=float(iks.mean()),
        shape_r2_mean=float(sr2.mean()), shape_r2_median=float(np.median(sr2)),
        abs_r2_mean=float(ar2.mean()), folds=folds), indent=2))
    print(f"Saved CV -> {metrics_dir / f'cv_Chla_gated{suffix}.json'}")
    if args.no_final:
        return 0

    print("\n--- Final training on all rows ---")
    norm = Normalizer.fit(X, y)
    norm = Normalizer(x_mean=norm.x_mean, x_std=np.where(norm.x_std == 0, 1.0, norm.x_std),
                      y_mean=0.0, y_std=1.0)
    Xn = norm.transform_x(X)
    rng = np.random.default_rng(args.seed); perm = rng.permutation(len(X)); cut = int(0.9 * len(perm))
    model, ik_final, eps = train_gated(cfg, Xn, rl, y, perm[:cut], perm[cut:], device, epochs=args.epochs,
                                       batch=args.batch_size, lr=args.lr, patience=args.patience,
                                       ik_head=args.ik_head)
    print(f"  learned Ik = {ik_final:.4f}" + (" (mean of head)" if args.ik_head else ""))
    art = MODEL_DIR / f"{args.source}_Chla_gated{suffix}.pt"
    save_artifact(art, model=model, normalizer=norm, feature_names=fnames, target_name="Chla_rel",
                  extra=dict(source=args.source, output_gate=True, relative_target=True,
                             gate="tanh(rel_light/Ik)", gate_ik=float(ik_final),
                             ik_head=args.ik_head, gate_ik_ref=IK_REF,
                             gate_fixed_ze=args.fixed_ze,
                             include_mld=True, include_no3=True, rel_cap=args.rel_cap,
                             log_target=False, include_season=False,
                             cv_shape_r2_mean=float(sr2.mean()), cv_abs_r2_mean=float(ar2.mean()),
                             n_rows=int(len(X)), epochs_run_final=eps))
    print(f"Saved model -> {art}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
