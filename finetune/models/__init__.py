"""MiniCPM 模型源码副本 —— 参考用, 训练流程不加载这里

训练和评测走的是 transformers 的 trust_remote_code, 代码来自权重目录
(/root/autodl-tmp/models/CPM8B/modeling_minicpm.py), 见 model_utils.py。

保留这份副本是为了:
  - 在 IDE / git 里直接读 InfLLMv2 的实现, 不用翻到仓库外
  - 将来要魔改稀疏注意力时, 有一个受版本控制的起点

这份副本与权重目录的原版只差两处兼容补丁 (Cache API shim、use_legacy_cache 判空),
在 transformers 4.56.1 下两处都已非必需。

真要启用这份代码, 需要改 model_utils.py 改回直接 import:
    from models.minicpm4.modeling_minicpm import MiniCPMForCausalLM
"""
