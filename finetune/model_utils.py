"""
模型加载模块

训练/评测统一通过 transformers 的 trust_remote_code 从权重目录加载 MiniCPM4,
不使用 models/ 下的本地副本 —— 那份保留作参考和后续魔改, 见 README「models/ 的定位」。

两个阶段的差别全部集中在这里的两个开关:
  阶段一 (稠密): sparse=False
  阶段二 (稀疏): sparse=True, 此时 attn_impl 必须是 flash_attention_2
"""
import os
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# InfLLM v2 官方默认值, 来自 openbmb/MiniCPM4-8B 的 README
DEFAULT_SPARSE_CONFIG = {
    'kernel_size': 32,      # 语义 kernel 大小
    'kernel_stride': 16,    # 相邻 kernel 的步长
    'init_blocks': 1,       # 每个 query 都会注意到的起始 block 数
    'block_size': 64,       # KV block 大小
    'window_size': 2048,    # 局部滑窗
    'topk': 64,             # 每个 token 只与最相关的 topk 个 block 算注意力
    'use_nope': False,      # 选块时是否用 NOPE
    'dense_len': 8192,      # 短于此长度仍走稠密; -1 表示始终稀疏
}


def build_sparse_config(overrides=None):
    """在官方默认值上叠加覆盖项, 例如 {'dense_len': -1}"""
    cfg = dict(DEFAULT_SPARSE_CONFIG)
    if overrides:
        unknown = set(overrides) - set(cfg)
        if unknown:
            raise ValueError(
                f'未知的 sparse_config 字段: {sorted(unknown)}; '
                f'可用字段: {sorted(cfg)}'
            )
        cfg.update(overrides)
    return cfg


def _local_classes():
    """从 models/minicpm4/ 加载本地(可修改)模型代码"""
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from models.minicpm4.configuration_minicpm import MiniCPMConfig
    from models.minicpm4.modeling_minicpm import MiniCPMForCausalLM
    return MiniCPMConfig, MiniCPMForCausalLM


def load_config(model_path, attn_impl='flash_attention_2', sparse=False, sparse_overrides=None,
                local_code=False):
    """加载 config 并按需注入 sparse_config

    注入而不是让用户手改 config.json, 是因为训练产出会被 save_pretrained
    按当前 config 重写, 手改的字段在阶段之间会丢。
    """
    if local_code:
        MiniCPMConfig, _ = _local_classes()
        config = MiniCPMConfig.from_pretrained(model_path)
    else:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    if sparse:
        if attn_impl != 'flash_attention_2':
            raise ValueError(
                f'InfLLMv2 稀疏注意力要求 attn_impl=flash_attention_2, 当前是 {attn_impl!r}。'
                ' (MiniCPMInfLLMv2Attention.__init__ 里有硬断言)'
            )
        config.sparse_config = build_sparse_config(sparse_overrides)
    else:
        # 权重目录的 config.json 里可能残留 sparse_config, 显式置空保证阶段一是纯稠密
        config.sparse_config = None

    return config


def describe_attention(model):
    """返回第一层实际实例化的 attention 类名

    这是确认稀疏有没有真开的唯一可靠手段 —— 光看 config 不够, 因为
    MiniCPMDecoderLayer 还额外要求 torch.cuda.is_available()。
    """
    for name, module in model.named_modules():
        if name.endswith('self_attn'):
            return type(module).__name__
    return '<未找到 self_attn 模块>'


def load_model_and_tokenizer(model_path, dtype=None, attn_impl='flash_attention_2',
                             sparse=False, sparse_overrides=None, local_code=False):
    """加载模型和 tokenizer

    Args:
        model_path: 权重目录 (含 safetensors + 官方 modeling_minicpm.py)
        dtype: torch dtype, 默认 bfloat16
        attn_impl: eager / sdpa / flash_attention_2
        sparse: 是否启用 InfLLM v2 稀疏注意力
        sparse_overrides: 覆盖部分 sparse_config 字段, 如 {'dense_len': -1}
        local_code: True 则用 models/minicpm4/ 下的本地代码 (含 varlen 打包改造),
                    False 用权重目录的官方代码 (trust_remote_code)
    Returns:
        model, tokenizer, config
    """
    config = load_config(model_path, attn_impl, sparse, sparse_overrides, local_code)
    dtype = dtype or torch.bfloat16

    print(f'Loading model: {model_path}')
    print(f'  hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}')
    print(f'  num_heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}')
    print(f'  attn_implementation={attn_impl}')
    print(f'  sparse_config={config.sparse_config}')
    print(f'  code={"models/minicpm4/ (本地)" if local_code else "权重目录 (trust_remote_code)"}')

    if local_code:
        _, MiniCPMForCausalLM = _local_classes()
        model = MiniCPMForCausalLM.from_pretrained(
            model_path, config=config, torch_dtype=dtype, attn_implementation=attn_impl)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )

    # CPM8B 的 tokenizer.json 格式较新, 用 slow tokenizer 兼容
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        eos = tokenizer.eos_token_id
        tokenizer.pad_token_id = eos[0] if isinstance(eos, (list, tuple)) else eos

    total = sum(p.numel() for p in model.parameters())
    print(f'\nModel: {type(model).__name__}, {total / 1e9:.2f}B params')
    print(f'Attention (layer 0): {describe_attention(model)}')

    return model, tokenizer, config
