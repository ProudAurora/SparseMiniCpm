# Finetune

MiniCPM4-8B 的两阶段全参数训练：

1. **阶段一** — 短序列 + 完全稠密注意力
2. **阶段二** — 长序列扩展训练 + InfLLM v2 稀疏注意力

模型通过 transformers 的 `trust_remote_code` 从权重目录加载，两个阶段共用一个训练入口，
差别只在命令行参数。

| | 阶段一 | 阶段二 |
|---|---|---|
| 脚本 | `scripts/run_stage1_dense.sh` | `scripts/run_stage2_sparse.sh` |
| `--max_length` | 4096 | 32768 |
| `--sparse` | 关 | 开 |
| attention | `MiniCPMFlashAttention2` | `MiniCPMInfLLMv2Attention` |
| 额外依赖 | 无 | `flash-attn` + `infllm_v2` |
| 起点权重 | `/root/autodl-tmp/models/CPM8B` | 阶段一的产出 |

## 目录

```
finetune/
├── train.py                唯一训练入口，全参数（无 LoRA）
├── data_utils.py           数据管线：chat → input_ids/labels，loss 策略与打包可配
├── data_compat.py          Arrow 兼容层，load_from_disk 失败时回退 pyarrow 直读
├── model_utils.py          模型加载 + sparse_config 注入
├── check_env.py            训练前预检
├── requirements.txt
│
├── configs/deepspeed/
│   ├── zero3.json          主配置，8B 全参在 4×A800 上
│   └── zero3_offload.json  32K 序列激活值撑不下时的退路
│
├── scripts/
│   ├── run_stage1_dense.sh
│   ├── run_stage2_sparse.sh
│   └── run_longbench.sh    长上下文验收
│
├── eval/
│   ├── inference.py        单轮生成，最快的 sanity check
│   └── longbench_minicpm.py  LongBench，复用官方 prompt/metrics，多卡并行，断点续跑
│
└── models/                 ★ 参考用，训练流程不加载（见下）
    └── minicpm4/           configuration + modeling 源码
```

## 安装

```bash
pip install -r requirements.txt
```

阶段二还需要两个 CUDA 库，见 `requirements.txt` 末尾的说明。

## 跑起来

```bash
# 0. 预检
python check_env.py                                # 阶段一
python check_env.py --sparse --max_length 32768    # 阶段二

# 1. 阶段一：短序列稠密
MAX_STEPS=10 bash scripts/run_stage1_dense.sh      # 先冒烟
bash scripts/run_stage1_dense.sh

# 2. 阶段二：长序列 + InfLLM v2
MODEL_PATH=/root/autodl-tmp/output/stage1_dense bash scripts/run_stage2_sparse.sh

# 3. 验收：两次跑分对比
MODEL_PATH=/root/autodl-tmp/output/stage1_dense  bash scripts/run_longbench.sh
MODEL_PATH=/root/autodl-tmp/output/stage2_sparse SPARSE=1 bash scripts/run_longbench.sh
```

所有路径和超参都能用环境变量覆盖，不必改脚本，见各脚本头部。

## 两个必须知道的机制

### InfLLM v2 不是「稀疏替代稠密」，是一个类里的两条路

开了 `sparse_config` 之后，**每一层**都换成 `MiniCPMInfLLMv2Attention`（`MiniCPMDecoderLayer`
是无条件替换，不分层），它内部再按序列长度分发：

```python
if kv_seq_len < self.dense_len:      # 默认 8192
    self._flash_attention_forward_dense(...)   # → flash_attn
else:
    self._sparse_attention_forward(...)        # → infllm_v2
```

两个直接后果：

- **`flash-attn` 是硬依赖**，不是可选加速。`infllm_v2` 包本身只依赖 torch，但 MiniCPM 的这份
  实现里稠密分支直接调 `flash_attn_varlen_func`。
- **`max_length` 必须明显大于 `dense_len`**，否则每个 batch 都落到稠密分支，稀疏内核一次都不会
  被调用——你以为在训稀疏，其实全程在跑 flash attention。`train.py` 和启动脚本都会对这种配置发
  警告，`check_env.py --sparse` 的第 5 项也会明确报出来。想强制始终稀疏就设 `DENSE_LEN=-1`。

### 内核缺失是静默的

`modeling_minicpm.py` 顶部是：

```python
try:
    from flash_attn import ...
    from infllm_v2 import ...
except:
    pass
```

`flash_attn` 在第一行，它一缺整块被跳过——**连带已装好的 `infllm_v2` 符号也不会绑定**，
真正调用时抛的是 `NameError` 而不是清晰的 `ImportError`。`check_env.py` 的第 4 项专门查这个。

## 序列打包与 block-diagonal 注意力

打包把多条短对话拼进一条 `max_length` 序列（实测 4096 下利用率 85%，平均每条装 4.9 个
文档），否则 GPU 算力大量浪费在 padding 上——不打包单 epoch 要 200 小时而非 40 小时。

朴素打包有三个副作用，`--local_code` 一次解决全部三个：

| 问题 | 朴素打包 | `--local_code` |
|---|---|---|
| 跨文档注意力 | 文档 B 能看到文档 A 的全部内容 | 按文档切 `cu_seqlens`，完全隔断 |
| RoPE 位置 | B 的首 token 拿到偏移位置而非 0 | `position_ids` 每文档重置为 0 |
| 边界预测 | A 的末 token 预测 B 的首 token | 仍存在，但占比仅约 0.1% |

实现方式：`pack_dataset` 产出按文档重置的 `position_ids`，模型侧
`_get_unpad_data_packed()` 据此推出文档边界 `cu_seqlens`，交给
`flash_attn_varlen_func` / `infllmv2_attn_varlen_func`。稠密和稀疏两条路径共用
同一个 `_upad_input` 注入点，因此同时生效。

未打包时 `position_ids` 是默认 `arange`，推出的边界正好等于 batch 行边界——与原逻辑
等价，是严格推广。带 KV cache 生成时 `position_ids` 形状不匹配，自动退回原逻辑。

**验证**：同架构模型下，打包序列中每个文档的 logits 与单独跑该文档一致。
稠密路径 B 区域偏差 0.0077 → 0.000061，稀疏路径 0.0020 → 0.000019，均降至 bf16 噪声底。

## `models/` 的定位

阶段一走 `trust_remote_code`（权重目录的官方代码）。阶段二用 `--local_code` 加载
`models/minicpm4/`，因为 varlen 打包改造在那里。

保留这份副本是为了能在 IDE 和 git 里直接读 InfLLM v2 的实现，以及将来要魔改稀疏注意力时有一个
受版本控制的起点。它与权重目录的原版只差两处兼容补丁（Cache API shim、`use_legacy_cache` 判空），
在 transformers 4.56.1 下两处都已非必需。

真要切回本地代码，改 `model_utils.py` 里的 import 即可。

## 数据

默认 `/root/autodl-tmp/datasets/sft-final-v2-n1990714`（Arrow，chat messages 格式）。

`--loss_on` 控制 loss 范围：

- `all_tokens`（默认）— 整段对话都计 loss。一次模板渲染，快
- `assistant_only` — 只训 assistant 回复。每条样本要做 O(轮数) 次模板渲染，明显更慢

阶段一、二都用 `all_tokens`。`assistant_only` 留给之后可能的指令微调。
