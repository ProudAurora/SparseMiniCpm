"""
数据管线

输入是 HuggingFace Arrow 格式的 chat 数据 (每条样本一个 messages 列表),
输出 input_ids / labels / attention_mask。

两个正交开关:
  loss_on = all_tokens      整段对话都计 loss
          = assistant_only  只对 assistant 回复计 loss
  pack    = True            把短序列拼接到 max_length, 填满 GPU
"""
import torch

from data_compat import load_dataset_from_disk_compat

LOSS_ON_CHOICES = ('all_tokens', 'assistant_only')


def _encode_all_tokens(messages, tokenizer, max_length):
    """整段对话都计 loss —— 一次模板渲染, 快"""
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    ids = ids[:max_length]
    return ids, list(ids)


def _encode_assistant_only(messages, tokenizer, max_length):
    """只对 assistant 回复计 loss

    对每个 assistant 轮次, 用 add_generation_prompt=True 渲染它之前的部分得到前缀长度,
    再用完整渲染得到该轮结束位置, 中间那段就是要计 loss 的区间。
    依赖 chat template 的前缀性质 (MiniCPM 的模板满足, 已验证)。

    注意: 每条样本要做 O(轮数) 次模板渲染, 比 all_tokens 慢数倍。
    """
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    labels = [-100] * len(ids)

    for i, msg in enumerate(messages):
        if msg.get('role') != 'assistant':
            continue
        prefix = tokenizer.apply_chat_template(
            messages[:i], tokenize=True, add_generation_prompt=True)
        upto = tokenizer.apply_chat_template(
            messages[:i + 1], tokenize=True, add_generation_prompt=False)
        for j in range(len(prefix), min(len(upto), len(labels))):
            labels[j] = ids[j]

    return ids[:max_length], labels[:max_length]


def _normalize_messages(sample):
    out = []
    for msg in sample.get('messages', []) or []:
        role, content = msg.get('role', ''), msg.get('content', '')
        if role and content:
            out.append({'role': role, 'content': content})
    return out


def pack_dataset(dataset, max_length, pad_token_id):
    """把短序列拼接成 max_length 的长序列, 避免算力浪费在 padding 上

    阶段二必须开: 对话样本普遍只有几百到几千 token, 不打包的话
    32K 的序列里绝大部分是 padding, 稀疏分支根本喂不饱。

    用生成器逐条 yield 给 Dataset.from_generator, 由 datasets 增量写 Arrow。
    早先的版本先把所有序列攒进 Python list 再 from_dict, 在全量数据上
    (约 39 万条 × 4096) 单进程峰值内存要 ~120GB, 4 个 rank 一起跑会打爆内存。
    """
    from datasets import Dataset

    def gen(ds, max_length, pad_token_id):
        # position_ids 在每个文档开头重置为 0。两个作用:
        #   1) RoPE 位置正确 —— 否则第二个文档的首 token 会拿到偏移后的位置
        #   2) 模型侧据此推出文档边界 cu_seqlens, 实现 block-diagonal 注意力
        cur_ids, cur_labels, cur_pos = [], [], []

        def build(ids, labels, pos):
            pad = max_length - len(ids)
            return {'input_ids': ids + [pad_token_id] * pad,
                    'labels': labels + [-100] * pad,
                    'attention_mask': [1] * len(ids) + [0] * pad,
                    'position_ids': pos + [0] * pad}

        for sample in ds:
            ids, labels = sample['input_ids'], sample['labels']
            if len(cur_ids) + len(ids) <= max_length:
                cur_ids.extend(ids)
                cur_labels.extend(labels)
                cur_pos.extend(range(len(ids)))
            else:
                if cur_ids:
                    yield build(cur_ids, cur_labels, cur_pos)
                cur_ids = list(ids)[:max_length]
                cur_labels = list(labels)[:max_length]
                cur_pos = list(range(len(cur_ids)))
        if cur_ids:
            yield build(cur_ids, cur_labels, cur_pos)

    return Dataset.from_generator(
        gen,
        gen_kwargs={'ds': dataset, 'max_length': max_length, 'pad_token_id': pad_token_id},
    )


def build_dataset(data_path, tokenizer, max_length=4096, val_ratio=0.005,
                  loss_on='all_tokens', pack=True, min_length=64, num_proc=None,
                  max_samples=None):
    """构建训练/验证数据集

    Args:
        data_path: Arrow 数据目录
        tokenizer: 分词器
        max_length: 最大序列长度
        val_ratio: 验证集比例
        loss_on: all_tokens / assistant_only
        pack: 是否做序列打包
        min_length: 短于此长度的样本丢弃
        num_proc: map 的并行进程数
        max_samples: 只取前 N 条 (冒烟测试用, None=全量)
    Returns:
        train_dataset, val_dataset
    """
    if loss_on not in LOSS_ON_CHOICES:
        raise ValueError(f'loss_on 必须是 {LOSS_ON_CHOICES} 之一, 收到 {loss_on!r}')

    raw = load_dataset_from_disk_compat(data_path)
    print(f'Raw dataset: {len(raw)} samples')
    if max_samples is not None and max_samples < len(raw):
        raw = raw.select(range(max_samples))
        print(f'Subsampled to {len(raw)} (max_samples={max_samples})')

    split = raw.train_test_split(test_size=val_ratio, seed=42)
    train_raw, val_raw = split['train'], split['test']
    print(f'Split: train={len(train_raw)}, val={len(val_raw)}')

    encode = _encode_all_tokens if loss_on == 'all_tokens' else _encode_assistant_only

    def preprocess(sample):
        messages = _normalize_messages(sample)
        if not messages:
            return {'input_ids': [], 'labels': [], 'attention_mask': []}
        ids, labels = encode(messages, tokenizer, max_length)
        return {'input_ids': ids, 'labels': labels, 'attention_mask': [1] * len(ids)}

    print(f'Tokenizing (loss_on={loss_on})...')
    train_ds = train_raw.map(preprocess, remove_columns=train_raw.column_names,
                             num_proc=num_proc, desc='Tokenizing train')
    val_ds = val_raw.map(preprocess, remove_columns=val_raw.column_names,
                         num_proc=num_proc, desc='Tokenizing val')

    def keep(sample):
        # 丢弃过短的样本, 以及 assistant_only 下整条都被 mask 掉的样本
        if len(sample['input_ids']) < min_length:
            return False
        return any(x != -100 for x in sample['labels'])

    before = len(train_ds)
    train_ds = train_ds.filter(keep, num_proc=num_proc, desc='Filtering train')
    val_ds = val_ds.filter(keep, num_proc=num_proc, desc='Filtering val')
    print(f'Filtered: train {before} -> {len(train_ds)}, val -> {len(val_ds)}')

    if pack:
        print(f'Packing to {max_length}...')
        pad_id = tokenizer.pad_token_id or 0
        train_ds = pack_dataset(train_ds, max_length, pad_id)
        val_ds = pack_dataset(val_ds, max_length, pad_id)
        print(f'Packed: train={len(train_ds)}, val={len(val_ds)}')

    print(f'Final: train={len(train_ds)}, val={len(val_ds)}')
    return train_ds, val_ds


class DataCollator:
    """动态 padding, 长度对齐到 pad_to_multiple_of 的倍数"""

    def __init__(self, tokenizer, max_length=4096, pad_to_multiple_of=8):
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __call__(self, features):
        max_len = max(len(f['input_ids']) for f in features)
        if self.pad_to_multiple_of > 1:
            mult = self.pad_to_multiple_of
            max_len = ((max_len + mult - 1) // mult) * mult
        max_len = min(max_len, self.max_length)

        # 打包数据带 position_ids (按文档重置); 未打包的现场生成 arange
        has_pos = 'position_ids' in features[0]

        batch = {'input_ids': [], 'labels': [], 'attention_mask': [], 'position_ids': []}
        for f in features:
            ids = f['input_ids'][:max_len]
            labels = f['labels'][:max_len]
            pad_len = max_len - len(ids)
            pos = f['position_ids'][:max_len] if has_pos else list(range(len(ids)))
            # 打包序列内部已经带 padding, 必须沿用样本自带的 attention_mask,
            # 不能按长度重建 —— 否则内部 padding 会被当成有效 token
            am = f['attention_mask'][:max_len] if 'attention_mask' in f else [1] * len(ids)
            batch['input_ids'].append(ids + [self.pad_token_id] * pad_len)
            batch['labels'].append(labels + [-100] * pad_len)
            batch['attention_mask'].append(am + [0] * pad_len)
            batch['position_ids'].append(pos + [0] * pad_len)

        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
