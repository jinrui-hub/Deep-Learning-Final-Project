#!/usr/bin/env bash

set -euo pipefail
set -x

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}/training/verl:${PROJECT_DIR}:${PYTHONPATH:-}"

: "${TRAIN_FILES:=/path/to/train.parquet}"
: "${VAL_FILES:=/path/to/val.parquet}"
: "${ACTOR_MODEL_PATH:=Qwen/Qwen3-8B}"
: "${JUDGE_MODEL_PATH:=/path/to/your/judge-model}"
: "${PROJECT_NAME:=memory_grpo}"
: "${EXPERIMENT_NAME:=memory_grpo_genrm}"

: "${TRAIN_BATCH_SIZE:=128}"
: "${MAX_PROMPT_LENGTH:=2048}"
: "${MAX_RESPONSE_LENGTH:=1024}"
: "${ROLLOUT_N:=4}"

: "${ACTOR_USE_KL:=True}"
: "${ACTOR_KL_COEF:=0.001}"
: "${ACTOR_KL_TYPE:=low_var_kl}"
: "${USE_REWARD_KL:=True}"
: "${REWARD_KL_PENALTY:=kl}"
: "${REWARD_KL_CTRL_TYPE:=fixed}"
: "${REWARD_KL_COEF:=0.0005}"
: "${REWARD_KL_TARGET:=0.1}"
: "${REWARD_KL_HORIZON:=10000}"

: "${JUDGE_BACKEND:=vllm}"
: "${JUDGE_TP_SIZE:=1}"
: "${JUDGE_GPU_MEM_UTIL:=0.75}"

export MEMORY_JUDGE_MODEL_NAME="${MEMORY_JUDGE_MODEL_NAME:-$JUDGE_MODEL_PATH}"
export MEMORY_REWARD_SCORE_MIN="${MEMORY_REWARD_SCORE_MIN:-1}"
export MEMORY_REWARD_SCORE_MAX="${MEMORY_REWARD_SCORE_MAX:-10}"
export MEMORY_REWARD_PARSE_FAIL_SCORE="${MEMORY_REWARD_PARSE_FAIL_SCORE:-0}"
export MEMORY_REWARD_OUTPUT_SCALE="${MEMORY_REWARD_OUTPUT_SCALE:-raw}"
export MEMORY_REWARD_TEMPERATURE="${MEMORY_REWARD_TEMPERATURE:-0.0}"
export MEMORY_REWARD_MAX_TOKENS="${MEMORY_REWARD_MAX_TOKENS:-256}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$ACTOR_MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss="$ACTOR_USE_KL" \
    actor_rollout_ref.actor.kl_loss_coef="$ACTOR_KL_COEF" \
    actor_rollout_ref.actor.kl_loss_type="$ACTOR_KL_TYPE" \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward="$USE_REWARD_KL" \
    algorithm.kl_penalty="$REWARD_KL_PENALTY" \
    algorithm.kl_ctrl.type="$REWARD_KL_CTRL_TYPE" \
    algorithm.kl_ctrl.kl_coef="$REWARD_KL_COEF" \
    algorithm.kl_ctrl.target_kl="$REWARD_KL_TARGET" \
    algorithm.kl_ctrl.horizon="$REWARD_KL_HORIZON" \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path="$PROJECT_DIR/training/memory_grpo/reward_fn.py" \
    reward.custom_reward_function.name=compute_score \
    reward.reward_model.enable=True \
    reward.reward_model.enable_resource_pool=False \
    reward.reward_model.model_path="$JUDGE_MODEL_PATH" \
    reward.reward_model.rollout.name="$JUDGE_BACKEND" \
    reward.reward_model.rollout.tensor_model_parallel_size="$JUDGE_TP_SIZE" \
    reward.reward_model.rollout.gpu_memory_utilization="$JUDGE_GPU_MEM_UTIL" \
    reward.reward_model.rollout.free_cache_engine=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    "$@"
