#!/bin/bash
# Phase D Phase-1 launcher — GRPO training with rubric judge.
#
# All experiment knobs are at the top of this file. To start a new experiment,
# copy this script (e.g. run_phase1_lr5e6.sh) and edit the variables.
#
# Lifecycle:
#   1. Start the rubric judge server in the background (config from RUBRIC_YAML)
#   2. Poll the endpoint until it's ready
#   3. Run verl GRPO training, plugging in our reward function
#   4. On exit (success OR failure OR Ctrl-C), kill the judge cleanly
#
# Run interactively (idev):
#   module load tacc-apptainer
#   bash training/verl_adapter/apptainer/run_phase1.sh
#
# Or as a SLURM job (uncomment the SBATCH block below first):
#   sbatch training/verl_adapter/apptainer/run_phase1.sh

# ---- (Optional) SLURM headers — uncomment for sbatch use ----
##SBATCH -J llmrnn_phase1
##SBATCH -A ASC25003
##SBATCH -p gpu-a100
##SBATCH -N 1
##SBATCH --gpus-per-node=3
##SBATCH -t 24:00:00
##SBATCH -o logs/phase1_%j.out
##SBATCH -e logs/phase1_%j.err

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_apptainer_common.sh"

# ============================================================================
# EXPERIMENT KNOBS — edit here for a new experiment
# ============================================================================

# --- Identity ---
PROJECT_NAME="llmrnn_grpo"
EXPERIMENT_NAME="phase1_run01"

# --- Data ---
DATA_DIR="${SCRATCH:-$HOME}/data/llm_as_rnn"
TRAIN_PARQUET="$DATA_DIR/train.parquet"
VAL_PARQUET="$DATA_DIR/val.parquet"

# --- Rubric judge (config from this yaml; reward fn + judge server share it) ---
RUBRIC_YAML="training/configs/rubric_v1.yaml"

# --- Policy backbone + LoRA ---
BASE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
LORA_RANK=16
LORA_ALPHA=32

# --- GRPO core hyperparams ---
ALGO="grpo"
LR="3e-6"
KL_COEF=0.001                  # β: KL anchor weight
KL_LOSS_TYPE="low_var_kl"      # k3 estimator (verl default)
ENTROPY_COEF=0
GRAD_CLIP=1.0

# --- Batch shapes ---
TRAIN_BATCH=16                 # prompts per gradient step
PPO_MINI_BATCH=16              # PPO inner mini-batch (== TRAIN_BATCH = 1 mini-batch/step)
PPO_MICRO_BATCH_PER_GPU=4      # forward batch per GPU (lower if OOM)

# --- Sequence shapes ---
MAX_PROMPT_LEN=3072            # memory-update prompts can be ~2-3k tokens
MAX_RESPONSE_LEN=512           # evolving_summary target ~200 tokens, give slack

# --- Rollout sampling (the "G" in GRPO) ---
N_ROLLOUTS=8                   # G: candidates per prompt
ROLLOUT_TEMP=0.8
ROLLOUT_TOP_P=0.95
ROLLOUT_TP_SIZE=2              # vLLM tensor parallel inside actor's GPU group
ROLLOUT_GPU_MEM_UTIL=0.6

# --- Training schedule ---
TOTAL_EPOCHS=20
TEST_FREQ=5
SAVE_FREQ=5

# --- Cluster / placement ---
N_GPUS_PER_NODE=2              # verl uses these (judge runs on the other GPU)
N_NODES=1

# ============================================================================
# Sanity checks before launch
# ============================================================================

[ -f "$REPO_ROOT/$RUBRIC_YAML" ] || { echo "[run] missing rubric yaml: $RUBRIC_YAML"; exit 1; }
[ -f "$TRAIN_PARQUET" ] || { echo "[run] missing train parquet: $TRAIN_PARQUET (did you run dump_trajectories.sh?)"; exit 1; }

# Val is optional — degrade gracefully so smoke runs work without dumping val.
if [ ! -f "$VAL_PARQUET" ]; then
    echo "[run] val parquet missing ($VAL_PARQUET); using train as placeholder + disabling validation"
    VAL_PARQUET="$TRAIN_PARQUET"
    TEST_FREQ=-1
fi

mkdir -p logs

JUDGE_PORT=$(python3 -c "import yaml; print(yaml.safe_load(open('$REPO_ROOT/$RUBRIC_YAML'))['serve']['port'])")
JUDGE_MODEL=$(python3 -c "import yaml; print(yaml.safe_load(open('$REPO_ROOT/$RUBRIC_YAML'))['judge']['model'])")

echo "[run] project    : $PROJECT_NAME / $EXPERIMENT_NAME"
echo "[run] base model : $BASE_MODEL (LoRA r=$LORA_RANK, alpha=$LORA_ALPHA)"
echo "[run] data       : train=$TRAIN_PARQUET, val=$VAL_PARQUET"
echo "[run] rubric yaml: $RUBRIC_YAML"
echo "[run] judge      : $JUDGE_MODEL @ localhost:$JUDGE_PORT"
echo "[run] grpo       : lr=$LR, kl_coef=$KL_COEF, G=$N_ROLLOUTS, batch=$TRAIN_BATCH, epochs=$TOTAL_EPOCHS"

# ============================================================================
# Lifecycle: start judge → wait → run training → cleanup
# ============================================================================

JUDGE_PID=""
cleanup() {
    if [ -n "$JUDGE_PID" ] && kill -0 "$JUDGE_PID" 2>/dev/null; then
        echo "[run] cleanup: killing judge server (PID $JUDGE_PID)"
        kill "$JUDGE_PID" 2>/dev/null || true
        wait "$JUDGE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# 1. Start judge server (background)
echo "[run] starting judge server (background)..."
CUDA_VISIBLE_DEVICES=2 bash "$(dirname "${BASH_SOURCE[0]}")/start_judge_server.sh" --rubric "$RUBRIC_YAML" \
    > logs/judge_${EXPERIMENT_NAME}.log 2>&1 &
JUDGE_PID=$!

# 2. Poll until judge endpoint is alive (timeout: 15 min for 8B+ models)
echo "[run] waiting for judge endpoint http://localhost:$JUDGE_PORT/v1/models ..."
WAITED=0
until curl -s "http://localhost:$JUDGE_PORT/v1/models" > /dev/null 2>&1; do
    if ! kill -0 "$JUDGE_PID" 2>/dev/null; then
        echo "[run] ERROR: judge server died during startup. Check logs/judge_${EXPERIMENT_NAME}.log"
        exit 1
    fi
    if [ $WAITED -ge 900 ]; then
        echo "[run] ERROR: judge server didn't respond within 15 min. Check logs/judge_${EXPERIMENT_NAME}.log"
        exit 1
    fi
    sleep 10
    WAITED=$((WAITED + 10))
    echo "[run]   ... still waiting ($WAITED s)"
done
echo "[run] judge ready after $WAITED s"

# --- Weights & Biases ---
export WANDB_PROJECT="$PROJECT_NAME"
export WANDB_NAME="$EXPERIMENT_NAME"
export WANDB_MODE="online"
export WANDB_API_KEY="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"   

# 3. Run verl GRPO training
echo "[run] launching verl trainer..."
apptainer exec --nv \
    "${BIND_ARGS[@]}" \
    "${ENV_ARGS[@]}" \
    --env "CUDA_VISIBLE_DEVICES=0,1" \
    "$VERL_SIF" \
    python -m verl.trainer.main_ppo \
        algorithm.adv_estimator=$ALGO \
        \
        data.train_files=$TRAIN_PARQUET \
        data.val_files=$VAL_PARQUET \
        data.train_batch_size=$TRAIN_BATCH \
        data.max_prompt_length=$MAX_PROMPT_LEN \
        data.max_response_length=$MAX_RESPONSE_LEN \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        \
        actor_rollout_ref.model.path=$BASE_MODEL \
        actor_rollout_ref.model.lora_rank=$LORA_RANK \
        actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        \
        actor_rollout_ref.actor.optim.lr=$LR \
        actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_PER_GPU \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
        actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
        actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEF \
        actor_rollout_ref.actor.grad_clip=$GRAD_CLIP \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.n=$N_ROLLOUTS \
        actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMP \
        actor_rollout_ref.rollout.top_p=$ROLLOUT_TOP_P \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
        actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL \
        actor_rollout_ref.rollout.load_format=safetensors \
        actor_rollout_ref.rollout.layered_summon=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_PER_GPU \
        \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_PER_GPU \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        \
        reward.custom_reward_function.path=training/verl_adapter/rubric_reward.py \
        reward.custom_reward_function.name=compute_score \
        algorithm.use_kl_in_reward=False \
        \
        trainer.critic_warmup=0 \
        trainer.val_before_train=False \
        trainer.logger='["console", "wandb"]' \
        trainer.project_name=$PROJECT_NAME \
        trainer.experiment_name=$EXPERIMENT_NAME \
        trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
        trainer.nnodes=$N_NODES \
        trainer.total_epochs=$TOTAL_EPOCHS \
        trainer.test_freq=$TEST_FREQ \
        trainer.save_freq=$SAVE_FREQ

echo "[run] verl trainer exited successfully"
# trap cleanup() runs automatically on EXIT and kills the judge
