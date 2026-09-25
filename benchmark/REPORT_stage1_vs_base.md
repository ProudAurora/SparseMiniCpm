非常v从v地方消防车v先吃饭方差发非常v方差方差发方差发发非常v非常v发从v对付日本官方# 阶段一微调前后评测报告：CPM8B vs stage1_dense

跑分日期 2026-09-17 · 复现命令 `python benchmark/compare_models.py --tag stage1_vs_base --bbh-limit 50`

| | |
|---|---|
| **before** | `/root/autodl-tmp/models/CPM8B`（MiniCPM4-8B 官方 instruct 权重） |
| **after** | `/root/autodl-tmp/models/stage1_dense`（阶段一：短序列 + 稠密注意力全参微调产物） |
| 评测集 | GSM8K 全量 1319 题；BBH 27 个子任务各取前 50 题，共 1350 题 |
| 解码 | greedy（`do_sample=False`），`max_new_tokens=512`，bf16 |
| 硬件/耗时 | 4×A800-80G，before 8.57 GPU·小时 / after 3.90 GPU·小时 |
| 原始结果 | `benchmark/results/stage1_vs_base/`（gitignore，含逐题 `*_predictions.jsonl`） |

两个模型跑的是**完全相同的题目、prompt 模板、解码参数和打分代码**，唯一变量是权重。两边的 chat template 渲染结果逐字符相同。

---

## 1. 总览

```
                                 before       after       delta
  GSM8K (n=1319)                 87.26%      84.91%       -2.35
  BBH macro (27 tasks, n=1350)   66.67%      45.56%      -21.11
  BBH mean response chars           918         163         -756
```

**GSM8K 基本持平（-2.35，1151/1319 → 1120/1319），BBH 大幅下降（-21.11）。**

BBH 27 个子任务中 19 个下降、6 个上升、2 个持平。

---

## 2. BBH 逐子任务（按跌幅排序）

`b_chars` / `a_chars` 为该子任务下模型回答的平均字符数。

| 子任务 | before | after | delta | b_chars | a_chars |
|---|---:|---:|---:|---:|---:|
| multistep_arithmetic_two | 84% | 0% | **-84** | 770 | 12 |
| tracking_shuffled_objects_seven_objects | 88% | 16% | **-72** | 1367 | 127 |
| tracking_shuffled_objects_five_objects | 98% | 30% | **-68** | 876 | 215 |
| logical_deduction_five_objects | 88% | 26% | **-62** | 960 | 415 |
| tracking_shuffled_objects_three_objects | 100% | 38% | **-62** | 473 | 280 |
| penguins_in_a_table | 92% | 50% | -42 | 575 | 98 |
| reasoning_about_colored_objects | 88% | 48% | -40 | 618 | 202 |
| salient_translation_error_detection | 68% | 32% | -36 | 1024 | 73 |
| boolean_expressions | 98% | 64% | -34 | 647 | 13 |
| ruin_names | 68% | 34% | -34 | 1170 | 101 |
| navigate | 72% | 46% | -26 | 770 | 194 |
| logical_deduction_three_objects | 98% | 78% | -20 | 496 | 178 |
| disambiguation_qa | 66% | 50% | -16 | 935 | 77 |
| temporal_sequences | 90% | 78% | -12 | 597 | 214 |
| movie_recommendation | 48% | 38% | -10 | 1654 | 197 |
| object_counting | 70% | 60% | -10 | 307 | 117 |
| causal_judgement | 64% | 58% | -6 | 1021 | 45 |
| date_understanding | 64% | 58% | -6 | 727 | 157 |
| formal_fallacies | 54% | 54% | 0 | 1408 | 145 |
| snarks | 64% | 64% | 0 | 961 | 74 |
| sports_understanding | 56% | 58% | +2 | 808 | 48 |
| geometric_shapes | 24% | 26% | +2 | 947 | 393 |
| word_sorting | 10% | 16% | +6 | 1018 | 180 |
| web_of_lies | 48% | 56% | +8 | 1033 | 76 |
| logical_deduction_seven_objects | 32% | 44% | +12 | 1371 | 361 |
| hyperbaton | 64% | 84% | +20 | 1129 | 133 |
| dyck_languages | 4% | 24% | +20 | 1133 | 270 |

---

## 3. 掉分的机制：CoT 塌缩，且与 prompt 有关

### 3.1 回答长度

| 模型 | benchmark | mean | median | p90 | 回答 <80 字符 | 抽不到答案 |
|---|---|---:|---:|---:|---:|---:|
| before | GSM8K | 748 | 710 | 1097 | 0.0% | 0.4% |
| before | BBH | 918 | 898 | 1505 | 0.1% | 0.0% |
| after | GSM8K | 538 | 486 | 856 | 0.0% | 0.0% |
| after | BBH | **163** | **75** | 290 | **53.0%** | 0.0% |

after 在 BBH 下**有 53% 的回答短于 80 字符**——直接吐 `Answer: X`，没有任何推理过程。但它在 GSM8K 下照常写推理（median 486 字符，没有一条秃答案），分数也只掉 2.35。

两个 benchmark 的 prompt 措辞不同：

- GSM8K（`run_gsm8k.py:29`）："...Think step by step before answering."
- BBH（`run_bbh.py:36`）："First work through your reasoning step by step. Then, on the very last line, write 'Answer: <answer>'..."

**after 只对前者买账。**所以这是**指令遵循 / 输出风格的退化**，不是推理能力整体消失。最直接的证据：`multistep_arithmetic_two`（纯算术，BBH prompt）掉到 0%，而同为算术推理的 GSM8K（另一套 prompt）仍有 84.91%。

### 3.2 实例对照

`multistep_arithmetic_two`，题目 `((-1 + 2 + 9 * 5) - (-2 + -4 + -4 * -7))`，gold = 24：

```
before: "...PEMDAS. Step 1: -1 + 2 + 9*5 = 1 + 45 = 46;  -2 + -4 + -4*-7 = -6 + 28 = 22.
         Step 2: 46 - 22 = 24.  Answer: 24"                                        ✓
after:  "Answer: 40"                                                               ✗
```

`object_counting`（数乐器），gold = 8：

```
before: 逐项列出 8 件乐器后计数 → "Answer: 8"                                       ✓
after:  "...That's 7 musical instruments. I have four stoves and two lamps.
         That's 6 more items. In total, I have 13 items.  Answer: 13"              ✗
```

已用 `--show-details` 与逐题 jsonl 核对过：**答案抽取逻辑工作正常**，这些是模型真的答错了，不是解析失误（after 在 BBH 上抽不到答案的比例是 0.0%）。

### 3.3 掉分集中在 before 本来就强的任务上

| 变量 | 与 per-task delta 的 Pearson r (n=27) |
|---|---:|
| after 的回答长度 | 0.053 |
| before 的回答长度 | 0.256 |
| 长度比 after/before | -0.122 |
| **before 的准确率** | **-0.734** |

- before ≥80% 的 10 个子任务：平均 delta **-49.6**
- before <80% 的 17 个子任务：平均 delta **-4.4**

也就是说，**微调恰恰破坏了基座模型原本有真实能力的地方**；6 个"上升"的子任务（`dyck_languages` 4%→24%、`word_sorting` 10%→16%、`geometric_shapes` 24%→26%）before 本来就接近随机，更像噪声而非真实提升。

> 注意：这个 -0.734 有一部分是机械性的——98% 的任务只能往下走，4% 的任务只能往上走（天花板效应 / 回归均值）。所以它能说明"损失集中在高分任务"，但不能单独作为因果证据。

---

## 4. 这个结果是否意外

不意外。阶段一的配置（`finetune/scripts/run_stage1_dense.sh`）：

```
MODEL_PATH  = models/CPM8B              # 起点是官方已对齐调优的 instruct 权重
DATA_PATH   = datasets/sft-final-v2-n1990714   # ~200 万条通用 SFT
EPOCHS=1  LR=1e-5  MAX_LENGTH=4096  LOSS_ON=all_tokens  # 全参，无 LoRA
```

对一个已经做过对齐/RL 的 instruct checkpoint 再灌一遍通用 SFT 数据，覆盖掉原有输出风格、推理长度退化，是常见结果。而且按 `finetune/README.md`，阶段一的定位本就是阶段二（长序列 + InfLLM v2 稀疏注意力）的起点，不是冲榜。

**但 BBH -21 分不能当作"符合预期"就放过**：阶段二要在这个 checkpoint 上继续训练，如果推理行为在阶段一已经退化，阶段二的稀疏注意力效果会和这层退化混在一起，到时候分不清是稀疏化的锅还是阶段一的锅。

---

## 5. 与公开榜单的关系

**无法直接对齐。** 原因有二：

1. CPM8B 的 model card（`models/CPM8B/README.md:278` 起）只有图片形式的 benchmark 表，仓库内没有可比的数值。
2. 本仓库的 BBH 是**零样本 + `Answer:` 抽取**的简化评分，不是官方那套逐任务 3-shot CoT prompt，绝对值会系统性偏低。

所以本报告里有意义的是 **before → after 的 delta**，不是绝对分数。

---

## 6. 建议的后续验证（本次未做）

1. **BBH 换 few-shot CoT prompt 重跑 after**。如果给了示例就能恢复推理，说明问题在指令遵循而非能力损失，阶段二可以照常推进（但需要在最终评测时统一 prompt 口径）；如果仍不行，得回看阶段一的 SFT 数据配比。这是区分"变笨"和"不听话"的关键实验，成本约 1 小时。
2. **看 `sft-final-v2-n1990714` 的回答长度分布**。如果该数据集本身以短答案为主，那 after 的行为就是被数据带的，和 `LOSS_ON=all_tokens`（loss 同时算在用户 prompt 上）一起构成最可能的解释。
3. **在阶段二产物上复跑本脚本**，与本报告的两列并排，才能把"稀疏化的影响"从"阶段一的影响"里分离出来。

---

## 附：本次跑分过程中修复的问题

| 问题 | 处理 |
|---|---|
| `models/stage1_dense/` 缺 `configuration_minicpm.py` / `modeling_minicpm.py`，`trust_remote_code` 加载直接报错 | 从 `finetune/models/minicpm4/`（训练时所用代码）复制过去，并补 `generation_config.json` |
| 该 checkpoint 的 `config.json` 里 `use_cache: false`（Trainer 把梯度检查点设置持久化了），生成时每步重算整段前缀 | `benchmark/common.py: load_model()` 统一强制 `use_cache=True`，两模型一致 |
| 缺少多模型对比入口 | 新增 `benchmark/compare_models.py`：多卡并行、`O_EXCL` 抢占式领取 job、断点续跑、对比表 |
| GSM8K 无法分片并行 | `run_gsm8k.py` 加 `shard_index` / `num_shards`，并保留原始题号以便合并 |
| 中途出对比表时，两模型完成的子任务集合不同，macro 平均跨集合相减无意义 | 所有跨模型数字只在**两边都已完成**的子任务交集上计算；某模型 GSM8K 无数据时显示 `--` / `n/a` 而不是 0% |

复跑（已完成的 job 会跳过）：

```bash
python benchmark/compare_models.py --tag stage1_vs_base --bbh-limit 50
python benchmark/compare_models.py --tag stage1_vs_base --report-only   # 只重出表
```
