#!/bin/bash
# Common helpers sourced by every apptainer/*.sh script.
# Sets VERL_SIF, REPO_ROOT, and exports the bind-mount + env arg arrays so each
# caller just does:  apptainer exec --nv "${BIND_ARGS[@]}" "${ENV_ARGS[@]}" "$VERL_SIF" bash -c "..."

# Resolve the repo root (3 levels up from this file).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Auto-locate the verl image: prefer $VERL_SIF if exported; else look in $WORK/apptainer_images/.
if [ -z "${VERL_SIF:-}" ]; then
    SIF_DIR="${WORK:-$HOME}/apptainer_images"
    candidate="$SIF_DIR/verlai_verl_vllm017.latest.sif"
    if [ -f "$candidate" ]; then
        VERL_SIF="$candidate"
    else
        echo "[apptainer] ERROR: could not find verl .sif image."
        echo "[apptainer]   tried: $candidate"
        echo "[apptainer]   either run training/verl_adapter/apptainer/pull_image.sh"
        echo "[apptainer]   or export VERL_SIF=/path/to/your.sif"
        exit 1
    fi
fi

# Ensure apptainer is on PATH (TACC sometimes hides it behind a module).
if ! command -v apptainer >/dev/null 2>&1; then
    module load tacc-apptainer 2>/dev/null || true
fi
if ! command -v apptainer >/dev/null 2>&1; then
    echo "[apptainer] ERROR: apptainer not found on PATH. Try: module spider apptainer"
    exit 1
fi

# Bind mounts: TACC paths kept identical inside/outside the container so
# absolute $WORK/$SCRATCH paths resolve the same either way.
BIND_ARGS=(
    --bind "/work:/work"
    --bind "/scratch:/scratch"
)

# HF cache redirect: model downloads land in $SCRATCH (not the read-only image).
HF_CACHE="${SCRATCH:-$HOME}/hf_cache"
mkdir -p "$HF_CACHE"

# PYTHONPATH trick: the container has every heavy dep pre-installed except
#   1) verl itself (verl maintainers' design — we expose our submodule), and
#   2) our own packages (models/, data_loader.py, training/ — Phase B+ scripts).
# Both come from PYTHONPATH; no in-container `pip install` (read-only fs).
ENV_ARGS=(
    --env "PYTHONPATH=$REPO_ROOT:$REPO_ROOT/training/verl:${PYTHONPATH:-}"
    --env "HF_HOME=$HF_CACHE"
)

echo "[apptainer] image:      $VERL_SIF"
echo "[apptainer] repo root:  $REPO_ROOT"
echo "[apptainer] HF cache:   $HF_CACHE"
