# Benchmark

对 `../../models/CPM8B`（MiniCPM4-8B）在 `../../eval` 下的标准评测集上跑分，用来在开始稀疏化训练前先建立一个 dense-attention 基线分数，后续接入 InfLLM V2 稀疏注意力训练后可以跟这个基线对比效果变化。

当前接入的评测集（数据来自 `/root/autodl-tmp/eval`，与本仓库同级，不在 git 跟踪范围内）：

- **GSM8K**（`eval/gsm8k_jsonl/main/test.jsonl`）：小学数学应用题，要求模型逐步推理并给出最终数字答案
- **BBH / BIG-Bench Hard**（`eval/bbh_jsonl/*.jsonl`，27 个子任务）：涵盖逻辑推理、日期理解、因果判断等高难度任务

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行

跑单个 benchmark：

```bash
python run_gsm8k.py --limit 20        # 先用 --limit 跑一小部分验证流程通不通
python run_bbh.py --tasks boolean_expressions,date_understanding --limit 20
```

一次性把两个 benchmark 跑完（只加载一次模型，节省重复加载 8B 模型的时间）：

```bash
python run_all.py --tag dense-baseline
```

`--tag` 用来给这次跑分结果打标签，方便以后跟稀疏化训练之后的结果区分（比如 `--tag sparse-v1`）。

常用参数：

- `--model-dir`：模型路径，默认自动定位到 `../../models/CPM8B`
- `--eval-dir`：评测数据根目录，默认自动定位到 `../../eval`
- `--limit`：每个任务/子任务只跑前 N 条，用于快速冒烟测试（BBH 全量 27 个子任务 + GSM8K 全量 1319 条测试样本在 8B 模型上跑一遍会比较久，建议先用 `--limit` 摸清单条耗时再决定要不要跑全量）
- `--max-new-tokens`：生成长度上限，默认 512。设得太小会让模型的推理过程被截断、还没写出最终答案就结束，导致答案抽取拿到一段无意义的中间文本，跑分虚低
- `--device`：默认自动检测 cuda
- `--show-details`：逐条打印详细对比（见下）

## 进度显示与逐条详情

运行时默认就有 tqdm 进度条，实时显示已完成条数、预计剩余时间和**实时正确率**：

```
bbh:  67%|██████▋   | 2/3 [00:03<00:01, 1.94s/ex, acc=0.5000, correct=1/2, task=boolean_expressions]
```

BBH 的进度条覆盖所有子任务的总样本数，`task=` 显示当前正在跑哪个子任务，每个子任务跑完会单独输出一行小结（`[done] boolean_expressions: 2/3 = 0.6667`）。

加上 `--show-details` 会在进度条之外，逐条打印抽取出的答案与标准答案的对比以及是否正确，一条一行：

```bash
python run_gsm8k.py --limit 5 --show-details
python run_bbh.py --tasks date_understanding --limit 5 --show-details
python run_all.py --tag debug --limit 5 --show-details
```

输出形如：

```
[bbh:date_understanding 1/5] correct    gold='(B)'  pred='(B)'
[bbh:date_understanding 2/5] WRONG      gold='(A)'  pred='(C)'
```

题目原文和模型完整回答不在这里打印（太长会淹没进度条），需要时去 `results/*_predictions.jsonl` 里看，每条记录都完整保存了。

这个选项主要用来排查**答案抽取逻辑**是否正常——如果发现跑分异常低，先用它扫一眼 `pred`，就能看出是模型真的答错了，还是回答被截断/格式不符导致抽取抓错了内容。

## 输出

结果写在 `results/`（已加入 `.gitignore`，不会被提交）：

- `<tag>_gsm8k_predictions.jsonl` / `<tag>_bbh_predictions.jsonl`：逐条的模型输出、抽取出的答案、是否正确
- `<tag>_summary.json`：GSM8K 总体准确率 + BBH 每个子任务准确率与 macro-average

## 评分逻辑

- **GSM8K**：提示模型用 `\boxed{答案}` 格式给出最终数字答案（沿用了 `eval/gsm8k/eval.yaml` 里已经定义好的 prompt 模板），抽取 boxed 内容（抽不到则退化为取回答里最后一个数字），跟 gold 答案（`answer` 字段里 `####` 后面的数字）数值比较
- **BBH**：零样本提示模型逐步思考后以 `Answer: <答案>` 结尾，抽取该行内容，跟 `target` 字段做归一化（转小写、去首尾括号/句号）后的精确匹配。这是一个简化版评分逻辑，没有使用 BBH 官方仓库里那套逐任务定制的 few-shot CoT prompt，如果后续要跟公开榜单数值对齐，可以针对性替换 `PROMPT_TEMPLATE`

## 文件结构

```
benchmark/
  common.py        # 模型加载、生成、答案抽取/匹配等共享逻辑
  run_gsm8k.py      # GSM8K 评测，可单独运行
  run_bbh.py        # BBH 评测（27 个子任务），可单独运行
  run_all.py        # 依次跑 GSM8K + BBH，只加载一次模型，汇总成 summary.json
  results/          # 输出目录（gitignore）
```
