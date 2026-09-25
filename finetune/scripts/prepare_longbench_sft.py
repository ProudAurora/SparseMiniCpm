#!/usr/bin/env python3
"""把 LongBench 格式的 jsonl 转成训练管线吃的 messages 数据集

synthetic_data/ 里是 LongBench 的原始 schema (context / input / answers / dataset),
而 data_utils.build_dataset 只认 messages 列表, 且 load_dataset_from_disk_compat
只读 Arrow 目录。这个脚本补上中间这一层。

两件事只能在这里做:

  1. prompt 模板。按 LongBench 官方 dataset2prompt 的措辞, 保证训练时的 prompt
     和 eval/longbench_minicpm.py 跑出来的一致。

  2. 中间截断。data_utils 的两个 _encode_* 都是 ids[:max_length] 尾部截断, 而
     这批数据的答案在序列末尾 —— 一条 4 万 token 的 gov_report 被尾截到 32K,
     答案整段没了, assistant_only 下全是 -100, 直接被 filter 丢掉。所以在这里
     先把 context 从中间掏空 (保留头尾), 让答案一定能留在窗口内。

用法:
    python finetune/scripts/prepare_longbench_sft.py \
        --src /root/autodl-tmp/datasets/synthetic_data \
        --out /root/autodl-tmp/datasets/synthetic_data_chat \
        --max-length 32768
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

# LongBench 官方 dataset2prompt 的措辞。key 是 jsonl 里的 dataset 字段。
TEMPLATES = {
    'gov_report':
        'You are given a report by a government agency. Write a one-page summary '
        'of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of '
        'the report.\n\nSummary:',
    'qmsum':
        'You are given a meeting transcript and a query containing a question or '
        'instruction. Answer the query in one or more sentences.\n\nTranscript:\n'
        '{context}\n\nNow, answer the query based on the above meeting transcript '
        'in one or more sentences.\n\nQuery: {input}\nAnswer:',
    # 多文档 QA: hotpotqa / multilexsum 都是"给若干段落, 答一个问题"
    'hotpotqa':
        'Answer the question based on the given passages. Only give me the answer '
        'and do not output any other words.\n\nThe following are given passages.\n'
        '{context}\n\nAnswer the question based on the given passages. Only give me '
        'the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:',
    'multilexsum_multidoc':
        'Answer the question based on the given passages. Only give me the answer '
        'and do not output any other words.\n\nThe following are given passages.\n'
        '{context}\n\nAnswer the question based on the given passages. Only give me '
        'the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:',
    # 单文档 QA: 用 multifieldqa_en 的措辞
    'natural_questions':
        'Read the following text and answer briefly.\n\n{context}\n\nNow, answer the '
        'following question based on the above text, only give me the answer and do '
        'not output any other words.\n\nQuestion: {input}\nAnswer:',
    'wikihow_qa':
        'Read the following text and answer briefly.\n\n{context}\n\nNow, answer the '
        'following question based on the above text, only give me the answer and do '
        'not output any other words.\n\nQuestion: {input}\nAnswer:',
    # 代码补全。lcc 只有 context, repobench-p 的 input 是当前文件前缀 (不可截断)
    'lcc':
        'Please complete the code given below. \n{context}Next line of code:\n',
    'repobench-p':
        'Please complete the code given below. \n{context}{input}Next line of code:\n',
}


def middle_truncate(ids, budget):
    """从中间掏空, 保留头尾。budget<=0 时返回空"""
    if budget <= 0:
        return []
    if len(ids) <= budget:
        return ids
    head = budget // 2
    tail = budget - head
    return ids[:head] + ids[-tail:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/root/autodl-tmp/datasets/synthetic_data')
    ap.add_argument('--out', default='/root/autodl-tmp/datasets/synthetic_data_chat')
    ap.add_argument('--tokenizer', default='/root/autodl-tmp/models/CPM8B')
    ap.add_argument('--max-length', type=int, default=32768)
    # chat template 的特殊 token + 模板里非 context 部分的余量
    ap.add_argument('--reserve', type=int, default=128)
    ap.add_argument('--batch', type=int, default=200)
    ap.add_argument('--limit-per-file', type=int, default=None, help='冒烟测试用')
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from datasets import Dataset

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    files = sorted(glob.glob(os.path.join(args.src, '*.jsonl')))
    if not files:
        sys.exit(f'{args.src} 下没有 jsonl')

    stats = defaultdict(Counter)
    lens = defaultdict(list)

    def gen():
        for path in files:
            name = os.path.basename(path)
            buf = []
            n_read = 0
            with open(path) as fh:
                for line in fh:
                    if args.limit_per_file and n_read >= args.limit_per_file:
                        break
                    buf.append(json.loads(line))
                    n_read += 1
                    if len(buf) >= args.batch:
                        yield from flush(buf, name)
                        buf = []
            if buf:
                yield from flush(buf, name)
            print(f'  {name}: {stats[name]["kept"]} kept / {n_read} read '
                  f'({stats[name]["truncated"]} 截断, {stats[name]["dropped"]} 丢弃)',
                  flush=True)

    def flush(records, fname):
        ds_names = [r.get('dataset', '') for r in records]
        unknown = [d for d in ds_names if d not in TEMPLATES]
        if unknown:
            raise KeyError(f'{fname}: 没有 prompt 模板的 dataset: {set(unknown)}')

        answers = [(r['answers'][0] if r.get('answers') else '') for r in records]
        contexts = [r.get('context', '') or '' for r in records]
        inputs = [r.get('input', '') or '' for r in records]

        ans_ids = tok(answers, add_special_tokens=False)['input_ids']
        ctx_ids = tok(contexts, add_special_tokens=False)['input_ids']
        # 模板里除 context 外的固定部分 (含 input) 的开销
        shells = [TEMPLATES[d].replace('{context}', '').format(input=i, context='')
                  if '{input}' in TEMPLATES[d] else TEMPLATES[d].replace('{context}', '')
                  for d, i in zip(ds_names, inputs)]
        shell_ids = tok(shells, add_special_tokens=False)['input_ids']

        for r, d, a, aid, cid, sid, inp in zip(
                records, ds_names, answers, ans_ids, ctx_ids, shell_ids, inputs):
            if not a.strip():
                stats[fname]['dropped'] += 1
                continue
            budget = args.max_length - args.reserve - len(aid) - len(sid)
            if budget < 128:
                # 答案本身就快撑满窗口, 留不下有意义的 context
                stats[fname]['dropped'] += 1
                continue
            if len(cid) > budget:
                ctx = tok.decode(middle_truncate(cid, budget))
                stats[fname]['truncated'] += 1
            else:
                ctx = r.get('context', '') or ''
            prompt = (TEMPLATES[d].format(context=ctx, input=inp)
                      if '{input}' in TEMPLATES[d] else TEMPLATES[d].format(context=ctx))
            stats[fname]['kept'] += 1
            total = min(len(cid), budget) + len(sid) + len(aid)
            lens[d].append(total)
            stats[fname]['ans_tokens'] += len(aid)
            stats[fname]['total_tokens'] += total
            yield {
                'messages': [{'role': 'user', 'content': prompt},
                             {'role': 'assistant', 'content': a}],
                'dataset': d,
            }

    print(f'源文件 {len(files)} 个, max_length={args.max_length}, reserve={args.reserve}')
    ds = Dataset.from_generator(gen)
    os.makedirs(os.path.dirname(args.out.rstrip('/')) or '.', exist_ok=True)
    ds.save_to_disk(args.out)

    print('\n================ 汇总 ================')
    kept = sum(s['kept'] for s in stats.values())
    ans_t = sum(s['ans_tokens'] for s in stats.values())
    tot_t = sum(s['total_tokens'] for s in stats.values())
    print(f'样本 {kept}  ->  {args.out}')
    print(f'总 token {tot_t/1e6:.1f}M, 其中 answer {ans_t/1e6:.2f}M ({ans_t*100/max(tot_t,1):.2f}%)')
    print(f'{"dataset":<24}{"n":>8}{"p50":>8}{"p90":>8}{"p99":>8}{"max":>8}{">=8K":>7}')
    for d, v in sorted(lens.items(), key=lambda x: -len(x[1])):
        v = sorted(v)
        p = lambda q: v[min(len(v) - 1, int(len(v) * q))]
        print(f'{d:<24}{len(v):>8}{p(.5):>8}{p(.9):>8}{p(.99):>8}{v[-1]:>8}'
              f'{sum(1 for x in v if x >= 8192) * 100 // len(v):>6}%')
    allv = sorted(x for v in lens.values() for x in v)
    p = lambda q: allv[min(len(allv) - 1, int(len(allv) * q))]
    print(f'{"ALL":<24}{len(allv):>8}{p(.5):>8}{p(.9):>8}{p(.99):>8}{allv[-1]:>8}'
          f'{sum(1 for x in allv if x >= 8192) * 100 // len(allv):>6}%')
    print(f'\n打包成 {args.max_length} 的序列约 {sum(allv)/args.max_length:.0f} 条')


if __name__ == '__main__':
    main()
