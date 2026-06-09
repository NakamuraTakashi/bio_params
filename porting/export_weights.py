"""Export each confirmed model's weights + normalization to plain files, so the
MLP forward pass can be reimplemented in any language (e.g. pure Fortran) without
PyTorch. Writes, per model, into porting/exported/:

  <stem>.npz   binary arrays (language-agnostic): n_layers, W0,b0,W1,b1,...,
               x_mean, x_std, y_mean, y_std
  <stem>.json  metadata: feature_names, in_dim, hidden, n_hidden_layers, clip,
               cutoff_depth, log_target, extra flags
  FORWARD.md   the exact feature-engineering + normalization + forward formula

Forward (all confirmed models, log_target=False):
  x   = feature_vector (order = feature_names)
  xn  = (x - x_mean) / x_std
  h0  = relu(W0 @ xn + b0); h1 = relu(W1 @ h0 + b1); ...   # n_hidden_layers
  y0  = W_last @ h_last + b_last                            # scalar
  y   = y0 * y_std + y_mean
  y   = clip(y, clip_lo, clip_hi)                           # physical range
(Wk has shape (out, in); rows = next layer.)

Usage:
    uv run python porting/export_weights.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bio_params.persist import load_artifact

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "pretrained"
OUT = Path(__file__).resolve().parent / "exported"
CONFIRMED = ["combined_Chla_allfeat", "combined_NO3", "combined_O2",
             "glodap_TA", "glodap_DIC", "glodap_SiO4", "glodap_PO4",
             "glodap_C13", "glodap_C14"]
CLIP = {"TA": (2000, 2600), "DIC": (1800, 2500), "NO3": (0, 60), "PO4": (0, 5),
        "SiO4": (0, 250), "O2": (0, 500), "C13": (-5, 5), "C14": (-300, 250),
        "Chla": (0, 50)}

FORWARD_MD = __doc__.split("Usage:")[0].split("Writes, per model")[1]


def linear_layers(state_dict):
    """Return Linear (weight,bias) pairs in net order (net.0, net.2, ...)."""
    idx = sorted({int(k.split(".")[1]) for k in state_dict if k.startswith("net.")
                  and k.endswith(".weight")})
    return [(state_dict[f"net.{i}.weight"].cpu().numpy(),
             state_dict[f"net.{i}.bias"].cpu().numpy()) for i in idx]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "FORWARD.md").write_text("# Forward formula (exported weights)\n" + FORWARD_MD)
    for stem in CONFIRMED:
        art = MODEL_DIR / f"{stem}.pt"
        if not art.exists():
            print(f"skip {stem}: missing"); continue
        model, norm, meta = load_artifact(art, map_location="cpu")
        layers = linear_layers(model.state_dict())
        arrays = {"n_layers": np.array(len(layers))}
        for i, (W, b) in enumerate(layers):
            arrays[f"W{i}"] = W.astype(np.float64)
            arrays[f"b{i}"] = b.astype(np.float64)
        arrays.update(x_mean=norm.x_mean.astype(np.float64),
                      x_std=norm.x_std.astype(np.float64),
                      y_mean=np.array(float(norm.y_mean)),
                      y_std=np.array(float(norm.y_std)))
        np.savez(OUT / f"{stem}.npz", **arrays)
        target = meta["target_name"]
        cfg = model.config
        json.dump({
            "stem": stem, "target": target,
            "feature_names": meta["feature_names"],
            "in_dim": cfg.in_dim, "hidden": cfg.hidden,
            "n_hidden_layers": cfg.n_hidden_layers, "out_dim": cfg.out_dim,
            "n_linear_layers": len(layers),
            "layer_shapes": [list(W.shape) for W, _ in layers],
            "clip": list(CLIP.get(target, [None, None])),
            "cutoff_depth": meta["extra"].get("cutoff_depth"),
            "log_target": bool(meta["extra"].get("log_target", False)),
            "include_mld": bool(meta["extra"].get("include_mld", False)),
            "include_no3": bool(meta["extra"].get("include_no3", False)),
            "nutricline_features": bool(meta["extra"].get("nutricline_features", False)),
            "strat_features": bool(meta["extra"].get("strat_features", False)),
        }, open(OUT / f"{stem}.json", "w"), indent=2)
        print(f"exported {stem}: in_dim={cfg.in_dim} layers={len(layers)} "
              f"-> {OUT.name}/{stem}.{{npz,json}}")
    print(f"\nForward formula -> {OUT.name}/FORWARD.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
