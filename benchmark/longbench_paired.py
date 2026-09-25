#!/usr/bin/env python3
"""LongBench 样本级配对分析

按任务聚合只有 21 个观测, 置信区间很宽。这里复现官方 eval.py 的逐条打分,
在样本级(数千对)做配对 bootstrap, 把"稀疏方法 vs 稠密方法"的差异做成带
置信区间的结论。

配对的前提: 两列跑的是同一批题目、同样顺序。评测脚本按 sample_index 写出,
这里按 sample_index 对齐, 对不上就跳过并报告。

    python benchmark/longbench_paired.py --limit 20
    python benchmark/longbench_paired.py                # 全量
"""
import argparse
import json
import random
import sys
from pathlib import Path

LB = Path('/root/autodl-tmp/LongBench/LongBench')
sys.path.insert(0, str(LB))

GROUPS = {
    '单文档 QA': ['narrativeqa', 'qasper', 'multifieldqa_en', 'multifieldqa_zh'],
    '多文档 QA': ['hotpotqa', '2wikimqa', 'musique', 'dureader'],
    '摘要': ['gov_report', 'qmsum', 'multi_news', 'vcsum'],
    '少样本学习': ['trec', 'triviaqa', 'samsum', 'lsht'],
    '合成任务': ['passage_count', 'passage_retrieval_en', 'passage_retrieval_zh'],
    '代码补全': ['lcc', 'repobench-p'],
}
IN_TRAIN = {'gov_report', 'qmsum', 'hotpotqa', 'lcc', 'repobench-p'}
FIRST_LINE = {'trec', 'triviaqa', 'samsum', 'lsht'}   # 官方 eval.py 的特殊处理


def per_example_scores(pred_dir, dataset, metric):
    """复现 eval.py 的逐条打分, 返回 {sample_index: score}"""
    path = pred_dir / f'{dataset}.jsonl'
    if not path.is_file():
        return {}
    out = {}
    for line in path.open(encoding='utf-8'):
        if not line.strip():
            continue
        r = json.loads(line)
        pred = r['pred']
        if dataset in FIRST_LINE:
            pred = pred.lstrip('\n').split('\n')[0]
        s = 0.0
        for gt in r['answers']:
            s = max(s, metric(pred, gt, all_classes=r['all_classes']))
        out[r.get('sample_index', len(out))] = s
    return out


def bootstrap_ci(diffs, n=10000, alpha=0.05, seed=0):
    rng = random.Random(seed)
    k = len(diffs)
    means = []
    for _ in range(n):
        means.append(sum(diffs[rng.randrange(k)] for _ in range(k)) / k)
    means.sort()
    return means[int(n * alpha / 2)], means[int(n * (1 - alpha / 2))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--a', default='lb_after_stage2', help='稀疏方法的 pred 目录前缀')
    ap.add_argument('--b', default='lb_dense_stage2', help='稠密方法的 pred 目录前缀')
    ap.add_argument('--label-a', default='sparse')
    ap.add_argument('--label-b', default='dense')
    ap.add_argument('--boot', type=int, default=10000)
    args = ap.parse_args()

    from eval import dataset2metric

    suf = '' if args.limit == 0 else f'_n{args.limit}'
    da, db = LB / 'pred' / (args.a + suf), LB / 'pred' / (args.b + suf)
    for d in (da, db):
        if not d.is_dir():
            sys.exit(f'目录不存在: {d}')

    rows, all_diffs, skipped = [], [], []
    for ds, metric in dataset2metric.items():
        A = per_example_scores(da, ds, metric)
        B = per_example_scores(db, ds, metric)
        keys = sorted(set(A) & set(B))
        if not keys:
            continue
        if len(A) != len(B) or len(keys) != len(A):
            skipped.append(f'{ds}({len(A)} vs {len(B)}, 配对 {len(keys)})')
        d = [(A[k] - B[k]) * 100 for k in keys]
        all_diffs += [(ds, x) for x in d]
        rows.append((ds, len(keys),
                     sum(A[k] for k in keys) / len(keys) * 100,
                     sum(B[k] for k in keys) / len(keys) * 100,
                     sum(d) / len(d),
                     sum(1 for x in d if abs(x) < 1e-9) * 100 / len(d)))

    if skipped:
        print(f'注意: 以下任务两列样本数不一致, 只用交集: {", ".join(skipped)}\n')

    print(f'样本级配对分析: {args.label_a} (A) vs {args.label_b} (B)')
    print(f'  A = {da.name}\n  B = {db.name}')
    print(f'  配对样本 {len(all_diffs)} 对, bootstrap {args.boot} 次\n')
    print(f'{"任务":<24}{"n":>6}{"A":>9}{"B":>9}{"A-B":>9}{"逐条相同":>10}')
    print('-' * 68)
    for ds, n, a, b, d, same in sorted(rows, key=lambda r: -abs(r[4])):
        print(f'{ds:<24}{n:>6}{a:>9.2f}{b:>9.2f}{d:>+9.2f}{same:>9.0f}%')

    def summarize(name, sel):
        if not sel:
            return
        d = [x for _, x in sel]
        m = sum(d) / len(d)
        lo, hi = bootstrap_ci(d, args.boot)
        same = sum(1 for x in d if abs(x) < 1e-9) * 100 / len(d)
        print(f'{name:<22}{len(d):>7} 对   A-B = {m:>+6.2f}   95% CI [{lo:>+6.2f}, {hi:>+6.2f}]   逐条相同 {same:.0f}%')

    print('\n' + '=' * 68)
    summarize('全部任务', all_diffs)
    summarize('  训练覆盖 (●)', [(k, v) for k, v in all_diffs if k in IN_TRAIN])
    summarize('  未覆盖 (泛化)', [(k, v) for k, v in all_diffs if k not in IN_TRAIN])
    print()
    for g, tasks in GROUPS.items():
        summarize('  ' + g, [(k, v) for k, v in all_diffs if k in tasks])


if __name__ == '__main__':
    main()
