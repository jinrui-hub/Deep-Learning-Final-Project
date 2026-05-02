#!/bin/bash
# One-time: pull the verl stable Apptainer image to $WORK.
#
# IMPORTANT: TACC blocks apptainer on LOGIN nodes (the binary prints a banner
# and exits 0 without doing anything). Run this from inside an `idev` session
# OR via `sbatch`. All other apptainer commands also need a compute node.
#
# Image: verlai/verl:vllm017.latest
#   - vllm 0.17.0
#   - torch 2.10.0 + CUDA 12.9
#   - flash-attn, apex, TransformerEngine pre-built
#   - matched by verl maintainers; no version-mismatch hell
#
# Cost: ~10-20 min download, ~15 GB on disk. Idempotent (skips if already pulled).

set -euo pipefail

IMAGE_TAG="${VERL_IMAGE_TAG:-verlai/verl:vllm017.latest}"
SIF_NAME="$(echo "$IMAGE_TAG" | sed 's|/|_|g; s|:|_|g').sif"
SIF_DIR="${WORK:-$HOME}/apptainer_images"
SIF_PATH="$SIF_DIR/$SIF_NAME"

mkdir -p "$SIF_DIR"

if [ -f "$SIF_PATH" ]; then
    echo "[apptainer] image already present at $SIF_PATH"
    echo "[apptainer]   $(du -h "$SIF_PATH" | cut -f1)"
    echo "[apptainer]   delete the .sif and re-run if you want a fresh pull"
    exit 0
fi

# Some TACC systems gate apptainer behind a module load
if ! command -v apptainer >/dev/null 2>&1; then
    echo "[apptainer] 'apptainer' not on PATH; trying: module load tacc-apptainer"
    module load tacc-apptainer 2>/dev/null || true
fi

if ! command -v apptainer >/dev/null 2>&1; then
    echo "[apptainer] ERROR: apptainer not found. Try:"
    echo "    module spider apptainer"
    echo "    module load <suggested-name>"
    exit 1
fi

# Detect TACC's login-node block: the apptainer wrapper prints a banner and
# exits 0 there, so a naive `apptainer pull` looks successful but creates no
# file. Warn early instead of silently "succeeding".
if [[ "$(hostname)" == login* ]]; then
    echo "[apptainer] ERROR: you appear to be on a TACC login node (hostname: $(hostname))."
    echo "[apptainer] TACC blocks apptainer on login nodes. Start an idev session first:"
    echo "    idev -p gpu-a100-dev -N 1 -n 1 --gpus-per-node=2 -t 02:00:00 -A ASC25003"
    echo "[apptainer] then re-run this script from inside the compute node shell."
    exit 1
fi

echo "[apptainer] pulling $IMAGE_TAG -> $SIF_PATH"
echo "[apptainer] this takes ~10-20 min and ~15 GB; do not Ctrl-C"
apptainer pull "$SIF_PATH" "docker://$IMAGE_TAG"

# Verify the pull actually produced a file. apptainer can exit 0 even when
# the underlying download was blocked / silently failed.
if [ ! -f "$SIF_PATH" ]; then
    echo "[apptainer] ERROR: apptainer exited 0 but no .sif file at $SIF_PATH"
    echo "[apptainer] common causes:"
    echo "    1. running on a node where apptainer is gated (login node)"
    echo "    2. Docker Hub rate-limit (try: apptainer remote login docker://docker.io)"
    echo "    3. write permission to $SIF_DIR"
    exit 1
fi

echo ""
echo "[apptainer] === done ==="
echo "[apptainer] image: $SIF_PATH ($(du -h "$SIF_PATH" | cut -f1))"
echo "[apptainer] export this in ~/.bashrc to skip re-detection in scripts:"
echo "    export VERL_SIF=$SIF_PATH"
