#!/bin/sh
# One incremental sync of a CMEMS GlobColour daily dataset into /data, preserving
# the native folder/file structure. Run inside the cmems-mirror container; the
# `copernicusmarine get --sync` only fetches files missing or newer on the
# server, so it is safe to run repeatedly from a scheduler.
#
# Credentials come from the environment (set via --env-file), no interactive
# login needed:
#   COPERNICUSMARINE_SERVICE_USERNAME
#   COPERNICUSMARINE_SERVICE_PASSWORD
#
# Overridable via env (defaults target the long gapfree daily series):
#   DATASET_ID, DATASET_VERSION, OUTDIR
#
# Single-instance is enforced outside the container by `docker run --name
# cmems_mirror` (a second run while one is active fails on the name clash).
set -eu

DATASET_ID="${DATASET_ID:-cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D}"
DATASET_VERSION="${DATASET_VERSION:-202603}"
OUTDIR="${OUTDIR:-/data/gapfree_daily}"

mkdir -p "$OUTDIR"
echo "[$(date '+%F %T')] start sync ${DATASET_ID} v${DATASET_VERSION} -> ${OUTDIR}"
copernicusmarine get \
  --dataset-id "$DATASET_ID" \
  --dataset-version "$DATASET_VERSION" \
  --output-directory "$OUTDIR" \
  --sync
rc=$?
echo "[$(date '+%F %T')] done (exit ${rc})"
exit "$rc"
