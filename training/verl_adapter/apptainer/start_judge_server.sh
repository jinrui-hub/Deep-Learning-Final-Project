#!/bin/bash
# Start a vLLM-served rubric judge for Phase C reward function.
#
# All judge config (model name, GPU index, port, mem util, max_model_len, dtype)
# is read from the rubric YAML — same yaml the reward function uses, so they
# stay in sync automatically.
#
# Default rubric YAML: training/configs/rubric_v1.yaml
#   Override which yaml to read by passing --rubric path/to/other.yaml
#
# Run inside an idev session (needs GPU). Foreground process; wrap with
# tmux if you want to detach.
#
# Usage:
#   module load tacc-apptainer
#   bash training/verl_adapter/apptainer/start_judge_server.sh
#   # or with a different rubric:
#   bash training/verl_adapter/apptainer/start_judge_server.sh --rubric training/configs/rubric_v2.yaml

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_apptainer_common.sh"

# Parse --rubric flag (default to v1)
RUBRIC_YAML="training/configs/rubric_v1.yaml"
if [[ "${1:-}" == "--rubric" ]]; then
    RUBRIC_YAML="$2"
fi

# Extract judge + serve config from the rubric YAML so both the reward
# function and this server use identical settings.
read -r JUDGE_MODEL JUDGE_PORT JUDGE_GPU GPU_MEM_UTIL MAX_MODEL_LEN DTYPE <<<"$(
    python3 -c "
import yaml, sys
y = yaml.safe_load(open('$REPO_ROOT/$RUBRIC_YAML'))
j, s = y['judge'], y['serve']
print(j['model'], s['port'], s['gpu'], s['gpu_memory_utilization'], s['max_model_len'], s['dtype'])
"
)"

echo "[judge] rubric yaml:        $RUBRIC_YAML"
echo "[judge] model:              $JUDGE_MODEL"
echo "[judge] port:               $JUDGE_PORT"
echo "[judge] gpu (in container): $JUDGE_GPU"
echo "[judge] gpu_memory_util:    $GPU_MEM_UTIL"
echo "[judge] max_model_len:      $MAX_MODEL_LEN"
echo "[judge] dtype:              $DTYPE"
echo "[judge] starting vllm serve (foreground; Ctrl-C to stop)"

apptainer exec --nv \
    "${BIND_ARGS[@]}" \
    "${ENV_ARGS[@]}" \
    --env "CUDA_VISIBLE_DEVICES=$JUDGE_GPU" \
    "$VERL_SIF" \
    vllm serve "$JUDGE_MODEL" \
        --host 0.0.0.0 \
        --port "$JUDGE_PORT" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --dtype "$DTYPE"
