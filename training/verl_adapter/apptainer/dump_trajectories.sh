#!/bin/bash
# Phase B: dump LLM-as-RNN trajectories to a verl-compatible parquet, inside
# the verl Apptainer container.
#
# Output: $SCRATCH/data/llm_as_rnn/<split>.parquet
#
# Inside an idev session (needs GPU for the 3 vLLM-backed RLN LLMs):
#   module load tacc-apptainer
#   bash training/verl_adapter/apptainer/dump_trajectories.sh \
#       --config configs/experiment1_seperateJudge.yaml \
#       --split train \
#       --max-patients 5      # smoke; drop for full
#
# Pass-through: any extra args go straight to dump_trajectories_to_parquet.py.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_apptainer_common.sh"

# Default output dir; can be overridden by passing --output-dir.
OUTPUT_DIR_DEFAULT="${SCRATCH:-$HOME}/data/llm_as_rnn"
mkdir -p "$OUTPUT_DIR_DEFAULT"

echo "[dump] forwarding to dump_trajectories_to_parquet.py with args: $*"
echo "[dump] (default --output-dir if not passed: $OUTPUT_DIR_DEFAULT)"

apptainer exec --nv \
    "${BIND_ARGS[@]}" \
    "${ENV_ARGS[@]}" \
    "$VERL_SIF" \
    bash -c "cd $REPO_ROOT && python -m training.verl_adapter.dump_trajectories_to_parquet $*"

echo ""
echo "[dump] done. Output should be at: $OUTPUT_DIR_DEFAULT/<split>.parquet"
ls -lh "$OUTPUT_DIR_DEFAULT" 2>/dev/null || true
