#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${UC_VLM_LF_ROOT:-${SCRIPT_DIR}/UC-VLM}"
CONFIG_PATH="${PROJECT_ROOT}/examples/train_lora/qwen2_5_vl_label_conditioned_sft.yaml"
DATASET_PATH="${PROJECT_ROOT}/data/only_sdv4_label_conditioned.json"
MERGED_MODEL_PATH="${PROJECT_ROOT}/output/qwen2_5vl_visual_merged"

if ! command -v llamafactory-cli >/dev/null 2>&1; then
  echo "error: llamafactory-cli is not available. Activate the UC-VLM Python 3.11+ environment." >&2
  exit 2
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "error: label-conditioned dataset not found: ${DATASET_PATH}" >&2
  exit 2
fi

if [[ ! -f "${MERGED_MODEL_PATH}/config.json" ]]; then
  echo "error: merged Stage-1 model not found: ${MERGED_MODEL_PATH}" >&2
  exit 2
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "error: Stage-3 training config not found: ${CONFIG_PATH}" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
exec llamafactory-cli train "${CONFIG_PATH}"
