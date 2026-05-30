#!/usr/bin/env bash
# Train GLODAP MLP models for the transient tracers Delta-14C (C14) and
# tritium (H3), then make prediction-vs-observation scatter plots.
#
# These are transient tracers (bomb-14C / decay), so a static T-S model is
# only a rough spatial proxy. Data are sparse (C14 ~42k, H3 ~29k points);
# treat CV scores as exploratory.
#
# Usage:
#     bash scripts/train_c14_h3.sh
#     bash scripts/train_c14_h3.sh --force
set -euo pipefail

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
fi

# target -> axis unit label for the scatter plots
declare -A UNIT=(
    [C14]="permil"
    [H3]="TU"
)

TARGETS=(C14 H3)
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
echo " C14 / H3 done."
echo "============================================================"
