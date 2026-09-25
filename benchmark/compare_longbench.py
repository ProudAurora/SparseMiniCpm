#!/usr/bin/env python3
"""LongBench 训练前后对比

串行跑两个模型的官方 LongBench, 再把 result.json 并排成表。

两边唯一的变量是权重: 同一份数据、同一套官方 prompt 模板、同样的
sparse_config (model_utils.load_config 在 --sparse 时会用 DEFAULT_SPARSE_CONFIG
重建, 所以权重目录 config.json 里残留的 dense_len 不会生效, 两边都是 8192)。

    python benchmark/compare_longbench.py                  # 跑两个模型 + 出表
    python benchmark/compare_longbench.py --report-only    # 只重出表
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

FINETUNE = Path(__file__).resolve().parent.parent / 'finetune'
LB_ROOT = Path('/root/autodl-tmp/LongBench/LongBench')
DATA_DIR = '/root/autodl-tmp/LongBench/_unzip/data'

# (列名, 权重目录, pred 子目录, 评测时是否开 InfLLM v2)
#
# 注意力实现必须和训练时一致, 否则就成了"用稀疏推理跑稠密训练的权重"。
# 基线 CPM8B 两种都跑: 稀疏臂和稠密臂各要一个同口径的起点, 否则
# delta 里会混进"推理实现不同"这个额外变量。
MODELS = [
    ('base_sp',  '/root/autodl-tmp/models/CPM8B',         'lb_before_cpm8b', True),
    ('sparse',   '/root/autodl-tmp/models/stage2_sparse', 'lb_after_stage2', True),
    ('base_de',  '/root/autodl-tmp/models/CPM8B',         'lb_base_dense',   False),
    ('dense',    '/root/autodl-tmp/models/stage2_dense',  'lb_dense_stage2', False),
]


def pred_name(base, limit):
    """不同 limit 的跑分写到不同目录

    官方 eval.py 是按 pred/<model_name>/ 整个目录打分的, 共用目录会让
    上一次中断留下的半截 jsonl 混进来, 也会让小规模和全量的结果互相覆盖。
    """
    return base if limit == 0 else f'{base}_n{limit}'

# LongBench 官方的任务分组, 出表时按组聚合
GROUPS = {
    '单文档 QA':   ['narrativeqa', 'qasper', 'multifieldqa_en', 'multifieldqa_zh'],
    '多文档 QA':   ['hotpotqa', '2wikimqa', 'musique', 'dureader'],
    '摘要':        ['gov_report', 'qmsum', 'multi_news', 'vcsum'],
    '少样本学习':  ['trec', 'triviaqa', 'samsum', 'lsht'],
    '合成任务':    ['passage_count', 'passage_retrieval_en', 'passage_retrieval_zh'],
    '代码补全':    ['lcc', 'repobench-p'],
}
# 阶段二训练数据覆盖到的任务 —— 出表时单独标注, 便于区分"训到的"和"泛化的"
IN_TRAIN = {'gov_report', 'qmsum', 'hotpotqa', 'lcc', 'repobench-p'}


def run_one(tag, model_path, model_name, sparse, args):
    model_name = pred_name(model_name, args.limit)
    out = LB_ROOT / 'pred' / model_name / 'result.json'
    if out.is_file() and not args.overwrite:
        print(f'[{tag}] 已有结果, 跳过: {out}')
        return
    if not Path(model_path).is_dir():
        sys.exit(f'[{tag}] 权重目录不存在: {model_path}')

    env = dict(os.environ)
    env.update({
        'MODEL_PATH': model_path,
        'MODEL_NAME': model_name,
        'SPARSE': '1' if sparse else '0',
        'LIMIT': str(args.limit),
        'DATASETS': args.datasets,
        'NUM_GPUS': str(args.num_gpus),
        'CUDA_VISIBLE_DEVICES': args.gpus,
        'LONGBENCH_DATA_DIR': DATA_DIR,
        'MAX_CONTEXT': str(args.max_context),
    })
    print(f'\n{"="*70}\n[{tag}] {model_path}  sparse={sparse}\n{"="*70}', flush=True)
    t = time.time()
    # 单列失败不应该让后面几列也不跑 —— 长跑里一列崩了还能拿到其余结果
    rc = subprocess.run(['bash', str(FINETUNE / 'scripts' / 'run_longbench.sh')],
                        env=env).returncode
    status = '完成' if rc == 0 else f'失败 (exit {rc})'
    print(f'[{tag}] {status}, 耗时 {(time.time()-t)/3600:.2f} 小时', flush=True)
    return rc == 0


def load(model_name):
    p = LB_ROOT / 'pred' / model_name / 'result.json'
    return json.load(open(p)) if p.is_file() else None


def report(args):
    res = {t: load(pred_name(n, args.limit)) for t, _, n, _ in MODELS}
    have = {t: r for t, r in res.items() if r}
    if len(have) < 2:
        sys.exit(f'结果不足, 已有: {sorted(have)}')
    missing = sorted(set(res) - set(have))
    if missing:
        print(f'警告: 以下列尚无结果, 已跳过: {missing}\n')

    # 只在所有已有列都覆盖到的任务上比较
    common = sorted(set.intersection(*(set(r) for r in have.values())))

    scale = '全量' if args.limit == 0 else f'每任务前 {args.limit} 条'
    L = []
    L.append(f'LongBench 三方对比 ({scale})')
    L.append('')
    L.append('  base_sp = CPM8B 原版        + 稀疏推理     (稀疏臂的起点)')
    L.append('  sparse  = stage2_sparse     + 稀疏推理     (长上下文训练 + InfLLM v2)')
    L.append('  base_de = CPM8B 原版        + 稠密推理     (稠密臂的起点)')
    L.append('  dense   = stage2_dense      + 稠密推理     (只有长上下文训练)')
    L.append('')
    L.append('  每个模型的注意力实现与其训练时一致; 数据/prompt/打分代码完全相同。')
    L.append('')

    cols = [t for t, _, _, _ in MODELS if t in have]
    hdr = f'{"任务":<24}' + ''.join(f'{c:>9}' for c in cols)
    if 'base_sp' in have and 'sparse' in have:
        hdr += f'{"稀疏Δ":>9}'
    if 'base_de' in have and 'dense' in have:
        hdr += f'{"稠密Δ":>9}'
    if 'sparse' in have and 'dense' in have:
        hdr += f'{"sp-de":>9}'
    L.append(hdr + '   训练覆盖')
    L.append('-' * 84)

    def row(name, vals, ind='  '):
        line = f'{ind}{name:<{22 if ind else 24}}' + ''.join(f'{vals[c]:>9.2f}' for c in cols)
        if 'base_sp' in have and 'sparse' in have:
            line += f'{vals["sparse"]-vals["base_sp"]:>+9.2f}'
        if 'base_de' in have and 'dense' in have:
            line += f'{vals["dense"]-vals["base_de"]:>+9.2f}'
        if 'sparse' in have and 'dense' in have:
            line += f'{vals["sparse"]-vals["dense"]:>+9.2f}'
        return line

    for g, tasks in GROUPS.items():
        sel = [t for t in tasks if t in common]
        if not sel:
            continue
        L.append(f'[{g}]')
        for t in sel:
            L.append(row(t, {c: have[c][t] for c in cols}) + f'   {"●" if t in IN_TRAIN else ""}')
        L.append(row('小计', {c: sum(have[c][t] for t in sel)/len(sel) for c in cols}))
        L.append('')

    L.append('=' * 84)
    L.append(row(f'总平均 (n={len(common)})', {c: sum(have[c][t] for t in common)/len(common) for c in cols}, ind=''))
    for label, sub in (('  训练覆盖 (●)', [t for t in common if t in IN_TRAIN]),
                       ('  未覆盖 (泛化)', [t for t in common if t not in IN_TRAIN])):
        if sub:
            L.append(row(label, {c: sum(have[c][t] for t in sub)/len(sub) for c in cols}, ind=''))

    text = '\n'.join(L)
    print('\n' + text)
    suffix = '' if args.limit == 0 else f'_n{args.limit}'
    dst = Path(__file__).resolve().parent / 'results' / f'longbench_compare{suffix}.txt'
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding='utf-8')
    json.dump(have, open(dst.with_suffix('.json'), 'w'), indent=2, ensure_ascii=False)
    print(f'\n已写入 {dst}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='每个任务最多 N 条, 0=全量')
    ap.add_argument('--datasets', default='all')
    ap.add_argument('--num-gpus', type=int, default=4)
    ap.add_argument('--gpus', default='0,1,2,3')
    ap.add_argument('--max-context', type=int, default=32768)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--report-only', action='store_true')
    args = ap.parse_args()

    if not args.report_only:
        failed = []
        for tag, path, name, sparse in MODELS:
            if not run_one(tag, path, name, sparse, args):
                failed.append(tag)
            # 每跑完一列就刷新一次报告, 中途查看也能拿到当前进展
            try:
                report(args)
            except SystemExit as e:
                print(f'(暂不足以出表: {e})', flush=True)
        if failed:
            print(f'\n!! 以下列失败: {failed}', flush=True)
    else:
        report(args)


if __name__ == '__main__':
    main()
