#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-TFBind8-Exact-v0}"
OUT_DIR="${DIBO_OUTPUT_DIR:-outputs/checkpoints}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
DIBO_SEED="${DIBO_SEED:?Set DIBO_SEED before starting training.}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
GRADIENT_CHECKPOINTING_ARGS=(--gradient_checkpointing)
case "${GRADIENT_CHECKPOINTING}" in
  0|false|False|FALSE|no|No|NO)
    GRADIENT_CHECKPOINTING_ARGS=(--no-gradient_checkpointing)
    ;;
esac

case "${TASK}" in
  TFBind8-Exact-v0|TFBind10-Exact-v0)
    DA_LR="2e-5"
    DA_STEPS="1024"
    GRAD_ACCUM_STEPS="16"
    ;;
  AntMorphology-Exact-v0|DKittyMorphology-Exact-v0)
    DA_LR="1e-5"
    DA_STEPS="2048"
    GRAD_ACCUM_STEPS="8"
    ;;
  *)
    echo "No paper hyperparameter preset for task: ${TASK}" >&2
    exit 2
    ;;
esac

python -u main.py \
  --use "${TASK}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --seed "${DIBO_SEED}" \
  --extra_name "dibo" \
  --stages da sft rl \
  --checkpoint_policy last \
  --da_lr "${DA_LR}" \
  --sft_lr 2e-5 \
  --rl_lr 1e-6 \
  --da_steps "${DA_STEPS}" \
  --sft_steps 1024 \
  --rl_steps 128 \
  --batch_size 1 \
  --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
  "${GRADIENT_CHECKPOINTING_ARGS[@]}" \
  --warmup_steps 100 \
  --lr_schedule warmup_constant \
  --output_dir "${OUT_DIR}" \
  --wandb_mode offline
