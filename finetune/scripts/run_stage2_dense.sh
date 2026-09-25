#!/bin/bash
# 阶段二对照组: 长序列训练 + 稠密注意力 (不开 InfLLM v2)
#
#   bash scripts/run_stage2_dense.sh
#   MAX_STEPS=10 bash scripts/run_stage2_dense.sh        # 冒烟测试
#
# 这是 run_stage2_sparse.sh 的消融对照。除了不传 --sparse, 其余每一项
# (起点权重/数据/max_length/batch/accum/lr/loss_on/调度/DeepSpeed) 都与
# 稀疏版逐字相同, 唯一变量是注意力实现。三方对比才能分离两种影响:
#
#   CPM8B (原版)        -> 长上下文训练的基线
#   stage2_dense (本脚本) -> 只有"长上下文训练"的效果
#   stage2_sparse        -> "长上下文训练 + 稀疏化"的效果
#
#   stage2_sparse - stage2_dense = 稀疏化本身的代价/收益
#
# 数据: synthetic_data_chat, 由 scripts/prepare_longbench_sft.py 从
#   datasets/synthetic_data/*.jsonl (LongBench schema) 转换而来。
#   分词和打包的缓存与稀疏版共用 (两边 max_length/loss_on 相同), 启动很快。
#
# 不传 --sparse 时 model_utils.load_config 会把 config.sparse_config 显式置为
# None, 每层都是普通 MiniCPMFlashAttention2。--local_code 仍要保留 —— varlen
# 打包 (按文档切 cu_seqlens 实现 block-diagonal 注意力) 在本地代码里。

set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ============ 路径 ============
# 本次从官方原版权重起跑 (不接阶段一产物 —— 阶段一在 BBH 上退化了 21 分,
# 接上去会把稀疏化的影响和阶段一的退化混在一起, 分不清归因)
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/CPM8B}"
DATA_PATH="${DATA_PATH:-/root/autodl-tmp/datasets/synthetic_data_chat}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/models/stage2_dense}"
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

# ============ 环境 ============
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/root/autodl-tmp/hf_cache}"
export HF_HUB_OFFLINE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
mkdir -p "$HF_DATASETS_CACHE" "$OUTPUT_DIR" "$LOG_DIR"
LOG_FILE="${LOG_DIR}/stage2_dense_$(date +%Y%m%d_%H%M%S).log"

# conda 自带的 libstdc++ (6.0.29) 缺 GLIBCXX_3.4.30, 而 DeepSpeed 的 cpu_adam 扩展需要它。
# ZeRO-3 offload 会用到 cpu_adam, 不预加载系统库的话会 ImportError。
SYS_LIBSTDCXX=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
if [ -e "$SYS_LIBSTDCXX" ] && ! strings /root/miniconda3/lib/libstdc++.so.6 2>/dev/null | grep -q GLIBCXX_3.4.30; then
    export LD_PRELOAD="${SYS_LIBSTDCXX}${LD_PRELOAD:+:$LD_PRELOAD}"
fi

PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
# 与稀疏版保持一致的并行配置, 保证两次跑分可比。稠密分支比稀疏分支省显存
# (实测 42G vs 77G, 不 offload 时), 所以这里余量更足, 但仍用 offload 以免
# 引入"并行策略不同"这个额外变量。
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${PROJECT_ROOT}/configs/deepspeed/zero3_offload.json}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "MODEL_PATH 不存在: $MODEL_PATH" >&2
    echo "显式指定 MODEL_PATH=<权重目录>" >&2
    exit 1
fi

# 本地代码含 varlen 打包改造: 32K 序列里会塞进十几个文档,
# 不隔断的话跨文档注意力污染很严重
LOCAL_CODE="${LOCAL_CODE:-1}"

CODE_ARGS=()
[ "$LOCAL_CODE" = 1 ] && CODE_ARGS+=(--local_code)

echo "================================================"
echo "阶段二对照组: 长序列扩展 + 稠密注意力"
echo "  Weights:   $MODEL_PATH"
echo "  Data:      $DATA_PATH"
echo "  Output:    $OUTPUT_DIR"
echo "  MaxLen:    $MAX_LENGTH"
echo "  LossOn:    $LOSS_ON"
echo "  Attn:      flash_attention_2  (sparse: 关闭)"
echo "  Code:      $([ "$LOCAL_CODE" = 1 ] && echo "models/minicpm4/ (varlen 打包)" || echo "官方 trust_remote_code")"
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
    "${CODE_ARGS[@]}" \
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
echo "阶段二对照组完成: $OUTPUT_DIR"
echo ""
echo "下一步 — 三方对比 (注意本模型评测时不要开 SPARSE):"
echo "  MODEL_PATH=$OUTPUT_DIR bash ${PROJECT_ROOT}/scripts/run_longbench.sh"
echo "================================================"
