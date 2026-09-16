#!/bin/bash
# 阶段二: 长序列扩展训练 + InfLLM v2 稀疏注意力
#
#   MODEL_PATH=/root/autodl-tmp/output/stage1_dense bash scripts/run_stage2_sparse.sh
#
# 前置条件 (先跑 python check_env.py --sparse 确认):
#   - flash-attn 已安装      —— InfLLMv2Attention 的稠密分支直接调它, 是硬依赖
#   - infllm_v2 已安装       —— third_party/infllmv2_cuda_impl 编译产物
#
# 关于 DENSE_LEN:
#   InfLLMv2Attention 同时承担两条路, 按序列长度分发:
#       kv_seq_len <  dense_len  -> flash attention 稠密分支
#       kv_seq_len >= dense_len  -> infllm_v2 稀疏分支
#   所以 MAX_LENGTH 必须明显大于 DENSE_LEN, 否则稀疏内核一次都不会被调用。
#   设 DENSE_LEN=-1 可强制始终走稀疏。

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ============ 路径 ============
# 默认指向阶段一的产出; 想从原始权重直接开始就显式覆盖 MODEL_PATH
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/output/stage1_dense}"
DATA_PATH="${DATA_PATH:-/root/autodl-tmp/datasets/synthetic_data}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/output/stage2_sparse}"

# ============ 训练配置 ============
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LR="${LR:-5e-6}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
LOSS_ON="${LOSS_ON:-all_tokens}"
MAX_STEPS="${MAX_STEPS:--1}"
MAX_SAMPLES="${MAX_SAMPLES:-}"   # 冒烟测试: 只取前 N 条
WARMUP_RATIO=0.05
WEIGHT_DECAY=0.01
SAVE_STEPS=200
EVAL_STEPS=200
LOGGING_STEPS=10

# ============ 稀疏配置 ============
# 留空则使用官方默认 (kernel_size 32 / block_size 64 / window 2048 / topk 64 / dense_len 8192)
DENSE_LEN="${DENSE_LEN:-8192}"
SPARSE_TOPK="${SPARSE_TOPK:-}"
SPARSE_WINDOW="${SPARSE_WINDOW:-}"
SPARSE_BLOCK_SIZE="${SPARSE_BLOCK_SIZE:-}"

# ============ 环境 ============
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/root/autodl-tmp/hf_cache}"
export HF_HUB_OFFLINE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
mkdir -p "$HF_DATASETS_CACHE" "$OUTPUT_DIR"

# conda 自带的 libstdc++ (6.0.29) 缺 GLIBCXX_3.4.30, 而 DeepSpeed 的 cpu_adam 扩展需要它。
# ZeRO-3 offload 会用到 cpu_adam, 不预加载系统库的话会 ImportError。
SYS_LIBSTDCXX=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
if [ -e "$SYS_LIBSTDCXX" ] && ! strings /root/miniconda3/lib/libstdc++.so.6 2>/dev/null | grep -q GLIBCXX_3.4.30; then
    export LD_PRELOAD="${SYS_LIBSTDCXX}${LD_PRELOAD:+:$LD_PRELOAD}"
fi

PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
# 32K 序列激活值很重, 默认用 offload 版; 显存够就换成 zero3.json
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${PROJECT_ROOT}/configs/deepspeed/zero3_offload.json}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "MODEL_PATH 不存在: $MODEL_PATH" >&2
    echo "先跑阶段一, 或显式指定 MODEL_PATH=<权重目录>" >&2
    exit 1
fi

if [ "$DENSE_LEN" -gt 0 ] 2>/dev/null && [ "$MAX_LENGTH" -le "$DENSE_LEN" ]; then
    echo "警告: MAX_LENGTH=$MAX_LENGTH <= DENSE_LEN=$DENSE_LEN" >&2
    echo "      每个 batch 都会落到稠密分支, 稀疏内核不会被调用。" >&2
    echo "      调大 MAX_LENGTH, 或设 DENSE_LEN=-1 强制始终稀疏。" >&2
fi

# 本地代码含 varlen 打包改造: 32K 序列里会塞进十几个文档,
# 不隔断的话跨文档注意力污染很严重
LOCAL_CODE="${LOCAL_CODE:-1}"

SPARSE_ARGS=(--sparse --dense_len "$DENSE_LEN")
[ "$LOCAL_CODE" = 1 ] && SPARSE_ARGS+=(--local_code)
[ -n "$SPARSE_TOPK" ]       && SPARSE_ARGS+=(--sparse_topk "$SPARSE_TOPK")
[ -n "$SPARSE_WINDOW" ]     && SPARSE_ARGS+=(--sparse_window "$SPARSE_WINDOW")
[ -n "$SPARSE_BLOCK_SIZE" ] && SPARSE_ARGS+=(--sparse_block_size "$SPARSE_BLOCK_SIZE")

echo "================================================"
echo "阶段二: 长序列扩展 + InfLLM v2 稀疏注意力"
echo "  Weights:   $MODEL_PATH"
echo "  Data:      $DATA_PATH"
echo "  Output:    $OUTPUT_DIR"
echo "  MaxLen:    $MAX_LENGTH"
echo "  LossOn:    $LOSS_ON"
echo "  Attn:      flash_attention_2 + InfLLMv2"
echo "  Code:      $([ "$LOCAL_CODE" = 1 ] && echo "models/minicpm4/ (varlen 打包)" || echo "官方 trust_remote_code")"
echo "  dense_len: $DENSE_LEN   (短于此长度的序列仍走稠密分支)"
echo "  LR:        $LR   Epochs: $EPOCHS"
echo "  Batch:     $BATCH_SIZE x $GRAD_ACCUM x $NUM_GPUS GPU = $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))"
echo "  DeepSpeed: $DEEPSPEED_CONFIG"
echo "================================================"

$PYTHON -m torch.distributed.run \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="${MASTER_PORT:-29502}" \
    "${PROJECT_ROOT}/train.py" \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --attn_impl flash_attention_2 \
    "${SPARSE_ARGS[@]}" \
    --max_length "$MAX_LENGTH" \
    --loss_on "$LOSS_ON" \
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
    --save_total_limit 3 \
    --eval_strategy steps \
    --eval_steps "$EVAL_STEPS" \
    --logging_steps "$LOGGING_STEPS" \
    --num_workers 8 \
    --deepspeed "$DEEPSPEED_CONFIG"

echo "================================================"
echo "阶段二完成: $OUTPUT_DIR"
echo ""
echo "下一步 — 长上下文验收:"
echo "  MODEL_PATH=$OUTPUT_DIR SPARSE=1 bash ${PROJECT_ROOT}/scripts/run_longbench.sh"
echo "================================================"
