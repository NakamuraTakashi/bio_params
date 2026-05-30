#!/usr/bin/env bash
# Full global BGC-Argo pipeline, fully automatic (no prompts):
#   1. train Chla (linear+clip), NO3 (full), O2 (subsampled) on the global box
#   2. prediction-vs-observation scatter plots
#   3. ROMS ini depth-map inference (with physical clip)
#   4. git commit the models + code
#
# Decisions baked in from prior findings:
#   - Chl-a: LINEAR target (log10 explodes under ROMS extrapolation) -> no --log-target
#   - O2:    --subsample 3,000,000 (global ~32M levels is too heavy)
#   - NO3:   full data
#   - box:   global (-180 180 -90 90)
#
# Usage: bash scripts/run_global_bgc_argo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

BOX="-180 180 -90 90"
O2_SUB=3000000
LOG=/tmp/global_pipeline.log

echo "===== [1/4] TRAIN (global) =====" | tee "$LOG"

echo "--- Chla (linear + clip) ---" | tee -a "$LOG"
uv run python scripts/train_bgc_argo_target.py --target Chla --box $BOX 2>&1 | tee -a "$LOG"

echo "--- NO3 (full) ---" | tee -a "$LOG"
uv run python scripts/train_bgc_argo_target.py --target NO3 --box $BOX 2>&1 | tee -a "$LOG"

echo "--- O2 (subsample $O2_SUB) ---" | tee -a "$LOG"
uv run python scripts/train_bgc_argo_target.py --target O2 --box $BOX \
    --subsample "$O2_SUB" 2>&1 | tee -a "$LOG"

echo "===== [2/4] SCATTER PLOTS =====" | tee -a "$LOG"
for t in Chla O2 NO3; do
    uv run python scripts/plot_bgc_argo_predictions.py --target "$t" 2>&1 | tee -a "$LOG"
done

echo "===== [3/4] ROMS DEPTH MAPS =====" | tee -a "$LOG"
uv run python scripts/predict_roms_ini_depths.py --source bgc_argo \
    --targets Chla O2 NO3 2>&1 | tee -a "$LOG"

echo "===== [4/4] GIT COMMIT =====" | tee -a "$LOG"
# Pull headline CV numbers from the metrics JSON for the commit message.
SUMMARY=$(uv run python - <<'PY'
import json
from pathlib import Path
d = Path("data/bgc_argo/processed")
out = []
for t in ["Chla", "NO3", "O2"]:
    p = d / f"cv_{t}.json"
    if p.exists():
        j = json.loads(p.read_text())
        out.append(f"{t} CV R2={j['r2_mean']:.3f} (n={j['n_rows']:,})")
print("; ".join(out))
PY
)
echo "summary: $SUMMARY" | tee -a "$LOG"

git add models/pretrained/bgc_argo_Chla.pt \
        models/pretrained/bgc_argo_NO3.pt \
        models/pretrained/bgc_argo_O2.pt
git commit -q -m "Train global BGC-Argo models: Chla, NO3, O2

Retrain on the global box (-180..180, -90..90) instead of the Kuroshio box.
Chla linear+clip (log explodes under ROMS extrapolation), NO3 full,
O2 subsampled to ${O2_SUB} levels.

${SUMMARY}

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"

echo "===== PIPELINE DONE =====" | tee -a "$LOG"
git --no-pager log --oneline -1
