"""
训练前预检

按顺序回答五个问题, 任何一项失败都会让训练在更深的地方以更难懂的方式炸掉:

  [1] 基础依赖齐了吗
  [2] 权重目录的 modeling 代码能加载吗
  [3] flash_attn / infllm_v2 这两个 CUDA 库装了吗
  [4] modeling 模块里内核符号真的绑定上了吗   <- 最容易被漏掉的一项
  [5] 开了 --sparse 之后, 实际实例化的是稀疏 attention 吗, 会走到稀疏分支吗

用法:
    python check_env.py                                  # 检查稠密路径 (阶段一)
    python check_env.py --sparse --max_length 32768      # 检查稀疏路径 (阶段二)
"""
import os
import sys
from argparse import ArgumentParser

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_MODEL_PATH = '/root/autodl-tmp/models/CPM8B'

_results = []


def report(ok, name, detail=''):
    mark = 'OK  ' if ok else 'FAIL'
    print(f'  [{mark}] {name}' + (f'  --  {detail}' if detail else ''))
    _results.append((ok, name))
    return ok


def check_basics():
    print('[1] 基础依赖')
    try:
        import torch
        import transformers
        report(True, 'torch', torch.__version__)
        report(True, 'transformers', transformers.__version__)
        report(torch.cuda.is_available(), 'CUDA 可用',
               f'{torch.cuda.device_count()} 卡' if torch.cuda.is_available() else
               '不可用 —— MiniCPMDecoderLayer 只在有 CUDA 时才会选稀疏 attention')
    except ImportError as e:
        report(False, 'torch / transformers', str(e))
        return False

    for mod in ('datasets', 'deepspeed'):
        try:
            __import__(mod)
            report(True, mod)
        except ImportError:
            report(False, mod, 'pip install -r requirements.txt')
    return True


def check_remote_module(model_path):
    """加载权重目录里的 modeling 代码, 不加载权重"""
    print(f'\n[2] 权重目录的 modeling 代码  ({model_path})')
    if not os.path.isdir(model_path):
        report(False, '目录存在', model_path)
        return None
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        cls = get_class_from_dynamic_module(
            'modeling_minicpm.MiniCPMForCausalLM', model_path)
        module = sys.modules[cls.__module__]
        report(True, 'MiniCPMForCausalLM 可加载', cls.__module__)
        report(hasattr(module, 'MiniCPMInfLLMv2Attention'), 'MiniCPMInfLLMv2Attention 存在')
        return module
    except Exception as e:
        report(False, 'modeling 代码加载', f'{type(e).__name__}: {e}')
        return None


def check_kernels():
    print('\n[3] CUDA 内核库')
    ok = True
    for mod, hint in (('flash_attn', 'pip install flash-attn --no-build-isolation'),
                      ('infllm_v2', 'cd third_party/infllmv2_cuda_impl && pip install -e . --no-build-isolation')):
        try:
            m = __import__(mod)
            report(True, mod, getattr(m, '__version__', ''))
        except ImportError:
            ok = report(False, mod, hint) and ok
    return ok


def check_symbol_binding(module):
    """modeling_minicpm.py 顶部是 try: from flash_attn ...; from infllm_v2 ...; except: pass

    flash_attn 在第一行, 它一缺整块被跳过 —— 连带已装好的 infllm_v2 符号也不会绑定,
    真正调用时抛的是 NameError 而不是清晰的 ImportError。这一项就是专门查这个。
    """
    print('\n[4] modeling 模块内的内核符号绑定')
    if module is None:
        report(False, '跳过', '[2] 失败')
        return False
    names = ['flash_attn_func', 'flash_attn_varlen_func', 'infllmv2_attn_stage1',
             'infllmv2_attn_varlen_func', 'max_pooling_1d_varlen']
    missing = [n for n in names if not hasattr(module, n)]
    if missing:
        report(False, f'{len(names) - len(missing)}/{len(names)} 个符号已绑定',
               f'缺: {", ".join(missing)} —— 稀疏路径会抛 NameError')
        return False
    return report(True, f'{len(names)}/{len(names)} 个符号已绑定')


def check_dispatch(model_path, sparse, attn_impl, max_length):
    """[5] 稀疏/稠密分发

    分两步, 且第一步不依赖内核是否装好 —— dense_len 配错是纯配置问题,
    即使 flash_attn 还没装也应该报出来。
    """
    print(f'\n[5] attention 分发  (sparse={sparse}, attn_impl={attn_impl}, max_length={max_length})')

    # --- 5a. 配置层面: max_length 够不够触发稀疏分支 ---
    config = None
    try:
        from model_utils import load_config
        config = load_config(model_path, attn_impl=attn_impl, sparse=sparse)
    except Exception as e:
        report(False, 'config 加载', f'{type(e).__name__}: {e}')

    if config is not None and sparse:
        dense_len = config.sparse_config['dense_len']
        if dense_len is not None and dense_len > 0 and max_length <= dense_len:
            report(False, f'分支判定 (dense_len={dense_len})',
                   f'max_length={max_length} <= dense_len, 每个 batch 都会落到稠密分支, '
                   f'稀疏内核不会被调用 —— 调大 max_length 或用 --dense_len -1')
        else:
            report(True, f'分支判定 (dense_len={dense_len})',
                   f'max_length={max_length} 的序列会走稀疏分支')

    # --- 5b. 实例化层面: 实际挑中的 attention 类 ---
    if config is None:
        return False
    try:
        import torch
        from transformers import AutoModelForCausalLM

        config._attn_implementation = attn_impl
        with torch.device('meta'):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

        attn = None
        for name, module in model.named_modules():
            if name.endswith('self_attn'):
                attn = type(module).__name__
                break

        if sparse:
            return report(attn == 'MiniCPMInfLLMv2Attention', '实例化的 attention 类',
                          attn if attn == 'MiniCPMInfLLMv2Attention'
                          else f'{attn} —— 稀疏未生效, 检查 CUDA 是否可用')
        return report(attn is not None and attn != 'MiniCPMInfLLMv2Attention',
                      '实例化的 attention 类', f'{attn}, 不涉及稀疏内核')
    except Exception as e:
        return report(False, '模型结构实例化', f'{type(e).__name__}: {e}')


def main():
    p = ArgumentParser(description='训练前环境预检')
    p.add_argument('--model_path', default=DEFAULT_MODEL_PATH)
    p.add_argument('--sparse', action='store_true', help='检查阶段二的稀疏路径')
    p.add_argument('--attn_impl', default=None,
                   choices=['eager', 'sdpa', 'flash_attention_2'])
    p.add_argument('--max_length', type=int, default=None)
    args = p.parse_args()

    attn_impl = args.attn_impl or ('flash_attention_2' if args.sparse else 'eager')
    max_length = args.max_length or (32768 if args.sparse else 4096)

    print('=' * 64)
    print(f'训练前预检  ({"阶段二 稀疏" if args.sparse else "阶段一 稠密"})')
    print('=' * 64)

    check_basics()
    module = check_remote_module(args.model_path)
    check_kernels()
    if args.sparse:
        check_symbol_binding(module)
    check_dispatch(args.model_path, args.sparse, attn_impl, max_length)

    failed = [name for ok, name in _results if not ok]
    print('\n' + '=' * 64)
    if failed:
        print(f'{len(failed)} 项未通过: {", ".join(failed)}')
        sys.exit(1)
    print(f'全部 {len(_results)} 项通过')


if __name__ == '__main__':
    main()
