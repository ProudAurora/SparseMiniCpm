"""
MiniCPM4 (8B) model package
可直接修改本目录下的 modeling_minicpm.py 来调整模型架构

关键可修改的模块:
  - MiniCPMAttention (标准 GQA attention)
  - MiniCPMFlashAttention2 (Flash Attention)
  - MiniCPMInfLLMv2Attention (稀疏注意力 InfLLM v2)
  - MiniCPMSdpaAttention (SDPA attention)
  - MiniCPMDecoderLayer (decoder layer, 选择 attention 实现)
  - MiniCPMMLP (前馈网络)
  - MiniCPMForCausalLM (完整模型)
  - InfLLMv2Cache (稀疏 KV cache)
"""
from .configuration_minicpm import MiniCPMConfig
from .modeling_minicpm import (
    MiniCPMForCausalLM,
    MiniCPMModel,
    MiniCPMPreTrainedModel,
    MiniCPMAttention,
    MiniCPMFlashAttention2,
    MiniCPMInfLLMv2Attention,
    MiniCPMSdpaAttention,
    MiniCPMDecoderLayer,
    MiniCPMMLP,
    MiniCPMRMSNorm,
    InfLLMv2Cache,
)
