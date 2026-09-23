#!/usr/bin/env bash
set -euo pipefail

: "${TRAIN_FILES:?Set TRAIN_FILES to one or more verl-compatible dataset files}"
: "${VAL_FILES:?Set VAL_FILES to one or more verl-compatible validation files}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
ROLLOUT_ENGINE="${ROLLOUT_ENGINE:-vllm}"
TOOL_CONFIG="${TOOL_CONFIG:-configs/verl/search_tool.yaml}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
PROJECT_NAME="${PROJECT_NAME:-agentic-rl-forge}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-search-r1-qwen2.5-3b-grpo}"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
  data.train_batch_size=512 \
  data.max_prompt_length=1024 \
  data.max_response_length=4096 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.name="$ROLLOUT_ENGINE" \
  actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=4 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$GPUS_PER_NODE" \
  trainer.nnodes=1 \
  trainer.save_freq=100 \
  trainer.test_freq=100 \
  trainer.total_training_steps=500 "$@"

