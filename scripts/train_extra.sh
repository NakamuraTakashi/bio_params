#!/usr/bin/env bash
# Train an MLP and generate prediction-vs-observation plots for the extra
# (non-core) GLODAP targets selected for modelling. Resumable: artifacts that
# already exist are not retrained unless --force is passed.
#
# NOTE: several of these targets are sparse (TOC ~3.6k, DON ~1.6k points) or
# non-core (DOC, Chla, TDN) or isotopes (C13, O18 in per mil). Their spatial
# CV scores are exploratory, not authoritative.
#
# Usage:
#     bash scripts/train_extra.sh
#     bash scripts/train_extra.sh --force
set -euo pipefail

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
fi

# target -> axis unit label for the scatter plots
declare -A UNIT=(
    [DOC]="umol/kg"
    [Chla]="mg/m3"
    [TDN]="umol/kg"
    [TOC]="umol/kg"
    [DON]="umol/kg"
    [C13]="permil"
    [O18]="permil"
)

TARGETS=(DOC Chla TDN TOC DON C13 O18)
MODEL_DIR="models/pretrained"

for tgt in "${TARGETS[@]}"; do
    artifact="${MODEL_DIR}/glodap_${tgt}.pt"

    echo "============================================================"
    echo " Target: ${tgt}  (unit: ${UNIT[$tgt]})"
    echo "============================================================"

    if [[ -f "$artifact" && $FORCE -eq 0 ]]; then
        echo "  artifact exists: $artifact (skip train, re-plot)"
    else
        uv run python scripts/train_target.py --target "$tgt"
    fi

    uv run python scripts/plot_predictions.py --target "$tgt" --unit "${UNIT[$tgt]}"
    echo
done

echo "============================================================"
echo " All extra targets done."
echo "============================================================"
