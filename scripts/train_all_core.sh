#!/usr/bin/env bash
# Train an MLP and generate prediction-vs-observation plots for each core
# GLODAP target. Resumable: artifacts that already exist are not retrained
# unless --force is passed.
#
# Usage:
#     bash scripts/train_all_core.sh
#     bash scripts/train_all_core.sh --force
set -euo pipefail

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
fi

TARGETS=(DIC NO3 PO4 SiO4 O2)
MODEL_DIR="models/pretrained"

for tgt in "${TARGETS[@]}"; do
    artifact="${MODEL_DIR}/glodap_${tgt}.pt"

    echo "============================================================"
    echo " Target: ${tgt}"
    echo "============================================================"

    if [[ -f "$artifact" && $FORCE -eq 0 ]]; then
        echo "  artifact exists: $artifact (skip train, re-plot)"
    else
        uv run python scripts/train_target.py --target "$tgt"
    fi

    uv run python scripts/plot_predictions.py --target "$tgt"
    echo
done

echo "============================================================"
echo " All core targets done."
echo "============================================================"
