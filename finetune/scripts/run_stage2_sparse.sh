#!/bin/bash
# 阶段二: 长序列扩展训练 + InfLLM v2 稀疏注意力
#
#   bash scripts/run_stage2_sparse.sh
#   MAX_STEPS=10 bash scripts/run_stage2_sparse.sh        # 冒烟测试
#
# 数据: synthetic_data_chat —— 由 scripts/prepare_longbench_sft.py 从
#   datasets/synthetic_data/*.jsonl (LongBench schema) 转换而来。原始 jsonl
#   既不是 Arrow 目录也没有 messages 列, 训练管线读不了, 必须先跑转换:
#
#       python scripts/prepare_longbench_sft.py --max-length 32768 \
#           --out /root/autodl-tmp/datasets/synthetic_data_chat
#
# 前置条件 (先跑 python check_env.py --sparse 确认):
#   - flash-attn 已安装      —— InfLLMv2Attention 的稠密分支直接调它, 是硬依赖
#   - infllm_v2 已安装       —— third_party/infllmv2_cuda_impl 编译产物
#
# 关于 DENSE_LEN —— 多卡下必须设成 -1, 这不是调参偏好:
#   InfLLMv2Attention 按 kv_seq_len 分发 (modeling_minicpm.py:1166):
#       kv_seq_len <  dense_len  -> flash attention 稠密分支
#       kv_seq_len >= dense_len  -> infllm_v2 稀疏分支
#   而 kv_seq_len = position_ids.max()+1 (同文件 :1118)。打包时 position_ids 在
#   每个文档开头重置为 0, 所以它是"batch 里最长那篇文档"的长度, 不是拼接总长。
#
#   于是各 rank 的 batch 里最长文档跨过 dense_len 这条线时走的分支就不一样,
#   两条路径在 DeepSpeed 参数协调器里记录的子模块序列不同 (注: 与 compress_k
#   有无参数无关 —— InfLLMv2 全程没有可训练参数, 三个权重目录都是 291 个张量),
#   于是 ZeRO-3 在 reset_step() 直接断言失败:
#       RuntimeError: Detected a disagreement on list length between rank0 and rank1:
#        rank0: 1450   rank1: 1322
#   实测 DENSE_LEN=8192 时第 1 步就崩。DENSE_LEN=-1 恒走稀疏, 各 rank 序列一致。
#
#   顺带也更合目的: 8192 时只有 19% 的样本能训到稀疏路径, -1 是 100%。
#   短文档走稀疏分支不会失真 —— window 2048 + topk 64 个 block x 64 = 覆盖 6144
#   token, 短于这个长度的文档等价于全注意力。
#
#   注意 MAX_LENGTH 只是打包长度和截断上限, 和分不分稀疏无关, 别指望调它。

set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ============ 路径 ============
# 本次从官方原版权重起跑 (不接阶段一产物 —— 阶段一在 BBH 上退化了 21 分,
# 接上去会把稀疏化的影响和阶段一的退化混在一起, 分不清归因)
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/CPM8B}"
DATA_PATH="${DATA_PATH:-/root/autodl-tmp/datasets/synthetic_data_chat}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/models/stage2_sparse}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/../log}"

# ============ 训练配置 ============
EPOCHS="${EPOCHS:-1}"
# 32K x bs2 时 lm_head 的 logits (32768 x 73448 x 2B, 反传还要再来一份) 就要 ~20GB,
# 加上激活基本打满; bs=1 配 accum 拉批量
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"   # 1 x 8 x 4卡 x 32768 = 1.05M token/步
LR="${LR:-5e-6}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
# 这批数据 answer 只占总 token 的 1.3% (gov_report 5.8% / repobench-p 0.41%),
# all_tokens 下 ~99% 的梯度会打在 context 上, 等于在学写政府报告而不是学回答
LOSS_ON="${LOSS_ON:-assistant_only}"
MAX_STEPS="${MAX_STEPS:--1}"
MAX_SAMPLES="${MAX_SAMPLES:-}"   # 冒烟测试: 只取前 N 条
WARMUP_RATIO=0.05
WEIGHT_DECAY=0.01
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-100}"
# 权重 16G + 优化器 92G = 107G/个; 682G 可用, 留 2 个 (峰值 3 x 107G)
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
SAVE_OPTIMIZER="${SAVE_OPTIMIZER:-1}"   # 存优化器状态, 长跑可断点续训
VAL_RATIO="${VAL_RATIO:-0.0005}"        # 32K 序列 eval 很贵, 取 ~95 条
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-8}"
LOGGING_STEPS=10

# ============ 稀疏配置 ============
# 留空则使用官方默认 (kernel_size 32 / block_size 64 / window 2048 / topk 64 / dense_len 8192)
DENSE_LEN="${DENSE_LEN:--1}"   # -1 = 恒走稀疏; 多卡下别改 (见顶部说明)
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
mkdir -p "$HF_DATASETS_CACHE" "$OUTPUT_DIR" "$LOG_DIR"
LOG_FILE="${LOG_DIR}/stage2_sparse_$(date +%Y%m%d_%H%M%S).log"

# conda 自带的 libstdc++ (6.0.29) 缺 GLIBCXX_3.4.30, 而 DeepSpeed 的 cpu_adam 扩展需要它。
# ZeRO-3 offload 会用到 cpu_adam, 不预加载系统库的话会 ImportError。
SYS_LIBSTDCXX=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
if [ -e "$SYS_LIBSTDCXX" ] && ! strings /root/miniconda3/lib/libstdc++.so.6 2>/dev/null | grep -q GLIBCXX_3.4.30; then
    export LD_PRELOAD="${SYS_LIBSTDCXX}${LD_PRELOAD:+:$LD_PRELOAD}"
fi

PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
# 实测 32K 序列下稀疏分支比稠密分支多吃 ~35G 显存 (77G vs 42G, 含 faiss 占的 10G)。
# DENSE_LEN=-1 时四张卡全走稀疏, 不 offload 只剩 ~3G 余量, 撑不住最长的文档。
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${PROJECT_ROOT}/configs/deepspeed/zero3_offload.json}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "MODEL_PATH 不存在: $MODEL_PATH" >&2
    echo "显式指定 MODEL_PATH=<权重目录>" >&2
    exit 1
fi

if [ "$DENSE_LEN" -gt 0 ] 2>/dev/null && [ "$NUM_GPUS" -gt 1 ]; then
    echo "警告: DENSE_LEN=$DENSE_LEN > 0 且是多卡。各 rank 可能走不同分支," >&2
    echo "      ZeRO-3 会在 reset_step() 断言失败。建议 DENSE_LEN=-1。" >&2
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
echo "  Log:       $LOG_FILE"
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
echo "阶段二完成: $OUTPUT_DIR"
echo ""
echo "下一步 — 长上下文验收:"
echo "  MODEL_PATH=$OUTPUT_DIR SPARSE=1 bash ${PROJECT_ROOT}/scripts/run_longbench.sh"
echo "================================================"
