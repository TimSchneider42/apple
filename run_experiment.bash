#!/bin/bash
set -euo pipefail

CONFIG="${1:-}"
ENV="${2:-}"
EVAL_ENV="${3:-null}"
ARGS=("${@:4}")

: "${CONFIG:?CONFIG (arg 1) is required}"
: "${ENV:?ENV (arg 2) is required}"

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PYTHONPATH="${SCRIPT_DIR}/python/"

if [[ -n "${WANDB_ENTITY:-}" ]]; then
  WANDB_ARGS=(wandb=true wandb_entity="${WANDB_ENTITY}")
else
  WANDB_ARGS=(wandb=false)
fi

python python/main.py \
  "+experiments/${CONFIG}" \
  "${WANDB_ARGS[@]}" \
  wandb_group="${ENV}" \
  env_id="${ENV}" \
  eval_env_id="${EVAL_ENV}" \
  "${ARGS[@]}"
