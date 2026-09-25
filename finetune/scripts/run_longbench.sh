#!/usr/bin/env bash
# LongBench 长上下文评测 —— 阶段二的验收手段
#
#   MODEL_PATH=/root/autodl-tmp/output/stage1_dense bash scripts/run_longbench.sh
#   MODEL_PATH=/root/autodl-tmp/output/stage2_sparse SPARSE=1 bash scripts/run_longbench.sh
#
# 两次跑分对比, 就是长序列扩展训练有没有效果的直接证据。
#
# LOADER=project 表示用本项目的 model_utils 加载 (才能控制 attn_impl / sparse);
# LOADER=auto 走 transformers 默认路径, 不带稀疏。

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/CPM8B}"
LONGBENCH_ROOT="${LONGBENCH_ROOT:-/root/autodl-tmp/LongBench/LongBench}"
MODEL_NAME="${MODEL_NAME:-minicpm4-8b}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)}"
MAX_CONTEXT="${MAX_CONTEXT:-32768}"
LIMIT="${LIMIT:-0}"
LOADER="${LOADER:-project}"
DATASETS="${DATASETS:-all}"
LONGBENCH_DATA_DIR="${LONGBENCH_DATA_DIR:-}"

# 稀疏开关: SPARSE=1 启用 InfLLM v2
SPARSE="${SPARSE:-0}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
DENSE_LEN="${DENSE_LEN:-}"

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/root/autodl-tmp/hf_cache/datasets}"

if [[ ! -f "${LONGBENCH_ROOT}/eval.py" ]]; then
  echo "LongBench 仓库未找到: ${LONGBENCH_ROOT}" >&2
  echo "先 clone: git clone https://github.com/THUDM/LongBench.git /root/autodl-tmp/LongBench" >&2
  exit 1
fi

EXTRA_ARGS=()
[[ -n "${LONGBENCH_DATA_DIR}" ]] && EXTRA_ARGS+=(--data-dir "${LONGBENCH_DATA_DIR}")
if [[ "${SPARSE}" == "1" ]]; then
  EXTRA_ARGS+=(--sparse)
  [[ -n "${DENSE_LEN}" ]] && EXTRA_ARGS+=(--dense-len "${DENSE_LEN}")
fi

echo "============================================================"
echo "MiniCPM4-8B LongBench 评测"
echo "Model:       ${MODEL_PATH}"
echo "LongBench:   ${LONGBENCH_ROOT}"
echo "GPUs:        ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} 进程)"
echo "Max context: ${MAX_CONTEXT}"
echo "Datasets:    ${DATASETS}"
echo "Limit/task:  ${LIMIT} (0 = 全量)"
echo "Loader:      ${LOADER}"
echo "Sparse:      $([[ "${SPARSE}" == "1" ]] && echo "开启 (InfLLM v2)" || echo "关闭")"
echo "Attn impl:   ${ATTN_IMPL}"
echo "============================================================"

"${PYTHON}" "${PROJECT_ROOT}/eval/longbench_minicpm.py" \
  --model-path "${MODEL_PATH}" \
  --model-name "${MODEL_NAME}" \
  --longbench-root "${LONGBENCH_ROOT}" \
  --project-root "${PROJECT_ROOT}" \
  --loader "${LOADER}" \
  --attn-impl "${ATTN_IMPL}" \
  --num-gpus "${NUM_GPUS}" \
  --max-context "${MAX_CONTEXT}" \
  --limit "${LIMIT}" \
  --datasets "${DATASETS}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
