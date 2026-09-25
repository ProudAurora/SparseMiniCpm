#!/bin/bash
# 阶段一: 短序列 + 完全稠密 全参数训练
#
#   bash scripts/run_stage1_dense.sh
#   MAX_STEPS=10 bash scripts/run_stage1_dense.sh          # 冒烟测试
#   DATA_PATH=/path/to/small bash scripts/run_stage1_dense.sh
#
# 不传 --sparse, 所以 sparse_config 显式为 None, 每层都是普通 attention。

set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ============ 路径 ============
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/CPM8B}"
DATA_PATH="${DATA_PATH:-/root/autodl-tmp/datasets/sft-final-v2-n1990714}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/models/stage1_dense}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/../log}"

# ============ 训练配置 ============
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-5}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
LOSS_ON="${LOSS_ON:-all_tokens}"
VAL_RATIO="${VAL_RATIO:-0.001}"        # 太大会让每次 eval 拖很久
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-8}"
MAX_STEPS="${MAX_STEPS:--1}"
MAX_SAMPLES="${MAX_SAMPLES:-}"   # 冒烟测试: 只取前 N 条
WARMUP_RATIO=0.05
WEIGHT_DECAY=0.01
SAVE_STEPS="${SAVE_STEPS:-500}"
EVAL_STEPS="${EVAL_STEPS:-500}"
# 完整 checkpoint 实测 107GB (优化器 92G + 权重 15.4G), 保存是"先写新再删旧",
# 峰值 = (LIMIT+1) x 107GB。346G 盘可用 322G, 所以 LIMIT 最大只能是 1。
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
# 1=连优化器状态一起存(72GB/个, 可断点续训); 0=只存权重(16GB/个)
SAVE_OPTIMIZER="${SAVE_OPTIMIZER:-1}"
LOGGING_STEPS=10

# flash_attention_2 更快更省显存; 没装 flash-attn 时改成 eager
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"

# 用 models/minicpm4/ 的本地代码, 启用 varlen 打包 (block-diagonal 注意力,
# 每个文档只能注意到自己)。注意: 只在 ATTN_IMPL=flash_attention_2 时有效,
# sdpa 路径不走 cu_seqlens。
LOCAL_CODE="${LOCAL_CODE:-1}"

# ============ 环境 ============
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/root/autodl-tmp/hf_cache}"
export HF_HUB_OFFLINE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
mkdir -p "$HF_DATASETS_CACHE" "$OUTPUT_DIR" "$LOG_DIR"
LOG_FILE="${LOG_DIR}/stage1_dense_$(date +%Y%m%d_%H%M%S).log"

# conda 自带的 libstdc++ (6.0.29) 缺 GLIBCXX_3.4.30, 而 DeepSpeed 的 cpu_adam 扩展需要它。
# ZeRO-3 offload 会用到 cpu_adam, 不预加载系统库的话会 ImportError。
SYS_LIBSTDCXX=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
if [ -e "$SYS_LIBSTDCXX" ] && ! strings /root/miniconda3/lib/libstdc++.so.6 2>/dev/null | grep -q GLIBCXX_3.4.30; then
    export LD_PRELOAD="${SYS_LIBSTDCXX}${LD_PRELOAD:+:$LD_PRELOAD}"
fi

PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${PROJECT_ROOT}/configs/deepspeed/zero3.json}"

echo "================================================"
echo "阶段一: 短序列 + 稠密 全参数训练"
echo "  Weights:   $MODEL_PATH"
echo "  Data:      $DATA_PATH"
echo "  Output:    $OUTPUT_DIR"
echo "  MaxLen:    $MAX_LENGTH"
echo "  LossOn:    $LOSS_ON"
echo "  Attn:      $ATTN_IMPL  (sparse: 关闭)"
echo "  Code:      $([ "$LOCAL_CODE" = 1 ] && echo "models/minicpm4/ (varlen 打包)" || echo "官方 trust_remote_code")"
echo "  LR:        $LR   Epochs: $EPOCHS"
echo "  Batch:     $BATCH_SIZE x $GRAD_ACCUM x $NUM_GPUS GPU = $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))"
echo "  Save:      每 $SAVE_STEPS 步, 保留 $SAVE_TOTAL_LIMIT 个, 优化器状态: $([ "$SAVE_OPTIMIZER" = 1 ] && echo 存 || echo 不存)"
echo "  Log:       $LOG_FILE"
echo "================================================"

$PYTHON -m torch.distributed.run \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="${MASTER_PORT:-29501}" \
    "${PROJECT_ROOT}/train.py" \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --attn_impl "$ATTN_IMPL" \
    $([ "$LOCAL_CODE" = 1 ] && echo --local_code) \
    --max_length "$MAX_LENGTH" \
    --loss_on "$LOSS_ON" \
    --val_ratio "$VAL_RATIO" \
    --preprocess_workers "$PREPROCESS_WORKERS" \
    --pack \
    --num_train_epochs "$EPOCHS" \
    --max_steps "$MAX_STEPS" \
    ${MAX_SAMPLES:+--max_samples "$MAX_SAMPLES"} \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --warmup_ratio "$WARMUP_RATIO" \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --bf16 \
    --gradient_checkpointing \
    --save_strategy steps \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    $([ "$SAVE_OPTIMIZER" = 1 ] && echo --save_optimizer_state) \
    --eval_strategy steps \
    --eval_steps "$EVAL_STEPS" \
    --logging_steps "$LOGGING_STEPS" \
    --num_workers 8 \
    --deepspeed "$DEEPSPEED_CONFIG" 2>&1 | tee -a "$LOG_FILE"

echo "================================================"
echo "阶段一完成: $OUTPUT_DIR"
echo ""
echo "下一步:"
echo "  1. 快速验证:  python ${PROJECT_ROOT}/eval/inference.py --model_path $OUTPUT_DIR"
echo "  2. 阶段二:    MODEL_PATH=$OUTPUT_DIR bash ${PROJECT_ROOT}/scripts/run_stage2_sparse.sh"
echo "================================================"
