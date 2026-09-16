"""
MiniCPM4-8B 全参数训练入口

两个阶段共用这一个脚本, 区别只在命令行参数:

  阶段一 (短序列 + 稠密):
      python train.py --model_path <base> --data_path <data> \
          --max_length 4096 --attn_impl flash_attention_2

  阶段二 (长序列 + InfLLM v2 稀疏):
      python train.py --model_path <阶段一产出> --data_path <data> \
          --max_length 32768 --attn_impl flash_attention_2 --sparse

实际用启动脚本: scripts/run_stage1_dense.sh / scripts/run_stage2_sparse.sh
"""
import json
import logging
import os
import sys
from argparse import ArgumentParser

import torch
from transformers import Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_utils import LOSS_ON_CHOICES, DataCollator, build_dataset
from model_utils import load_model_and_tokenizer

logger = logging.getLogger(__name__)


def parse_args():
    p = ArgumentParser(description='MiniCPM4-8B full-parameter training')

    # Model
    p.add_argument('--model_path', type=str, required=True,
                   help='权重目录 (阶段二填阶段一的产出)')
    p.add_argument('--local_code', action='store_true', default=False,
                   help='用 models/minicpm4/ 下的本地模型代码 (含 varlen 打包改造: '
                        '按文档切 cu_seqlens, 实现 block-diagonal 注意力)。'
                        '默认用权重目录的官方代码')
    p.add_argument('--attn_impl', type=str, default='flash_attention_2',
                   choices=['eager', 'sdpa', 'flash_attention_2'],
                   help='注意力实现; --sparse 时必须是 flash_attention_2')

    # Sparse attention (阶段二)
    p.add_argument('--sparse', action='store_true',
                   help='启用 InfLLM v2 稀疏注意力')
    p.add_argument('--dense_len', type=int, default=None,
                   help='短于此长度仍走稠密分支 (默认 8192, -1 表示始终稀疏)')
    p.add_argument('--sparse_topk', type=int, default=None, help='每个 token 选取的 block 数 (默认 64)')
    p.add_argument('--sparse_window', type=int, default=None, help='局部滑窗大小 (默认 2048)')
    p.add_argument('--sparse_block_size', type=int, default=None, help='KV block 大小 (默认 64)')

    # Data
    p.add_argument('--data_path', type=str, required=True, help='Arrow 数据目录')
    p.add_argument('--max_length', type=int, default=4096)
    p.add_argument('--val_ratio', type=float, default=0.005)
    p.add_argument('--loss_on', type=str, default='all_tokens', choices=list(LOSS_ON_CHOICES),
                   help='all_tokens=整段计 loss; assistant_only=只训 assistant 回复')
    p.add_argument('--pack', dest='pack', action='store_true', default=True,
                   help='序列打包 (默认开)')
    p.add_argument('--no_pack', dest='pack', action='store_false')
    p.add_argument('--min_length', type=int, default=64)
    p.add_argument('--preprocess_workers', type=int, default=None,
                   help='数据预处理并行进程数')
    p.add_argument('--max_samples', type=int, default=None,
                   help='只用前 N 条样本 (冒烟测试用)')

    # Optimization
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--num_train_epochs', type=float, default=1)
    p.add_argument('--max_steps', type=int, default=-1)
    p.add_argument('--per_device_train_batch_size', type=int, default=1)
    p.add_argument('--per_device_eval_batch_size', type=int, default=1)
    p.add_argument('--gradient_accumulation_steps', type=int, default=16)
    p.add_argument('--learning_rate', type=float, default=1e-5)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_ratio', type=float, default=0.05)
    p.add_argument('--lr_scheduler_type', type=str, default='cosine')
    p.add_argument('--max_grad_norm', type=float, default=1.0)

    # System
    p.add_argument('--bf16', action='store_true', default=True)
    p.add_argument('--fp16', action='store_true', default=False)
    p.add_argument('--gradient_checkpointing', action='store_true', default=True)
    p.add_argument('--no_gradient_checkpointing', dest='gradient_checkpointing',
                   action='store_false')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=8, help='dataloader workers')
    p.add_argument('--save_strategy', type=str, default='steps')
    p.add_argument('--save_steps', type=int, default=500)
    p.add_argument('--save_total_limit', type=int, default=3)
    p.add_argument('--save_only_model', action='store_true', default=True,
                   help='只存模型权重(约15GB), 不存优化器状态(约57GB)。'
                        '本机 96GB 盘装不下完整 checkpoint, 故默认开启; 代价是无法断点续训')
    p.add_argument('--save_optimizer_state', dest='save_only_model', action='store_false',
                   help='连优化器状态一起存, 支持断点续训, 每个 checkpoint 约 72GB')
    p.add_argument('--load_best_model_at_end', action='store_true', default=False,
                   help='训练结束回滚到 eval_loss 最优的 checkpoint。会额外多留一个 '
                        'checkpoint(全量约107GB), 磁盘紧张时别开')
    p.add_argument('--eval_strategy', type=str, default='steps')
    p.add_argument('--eval_steps', type=int, default=500)
    p.add_argument('--logging_steps', type=int, default=10)
    p.add_argument('--deepspeed', type=str, default=None)

    return p.parse_args()


def collect_sparse_overrides(args):
    """把命令行上的稀疏参数收成 overrides dict, 没给的保持官方默认"""
    mapping = {
        'dense_len': args.dense_len,
        'topk': args.sparse_topk,
        'window_size': args.sparse_window,
        'block_size': args.sparse_block_size,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def main():
    args = parse_args()
    set_seed(args.seed)

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
    )
    logger.info('Args: %s', json.dumps(vars(args), indent=2, ensure_ascii=False))

    # ---------------- 1. TrainingArguments ----------------
    # 必须在建模型之前构造! ZeRO-3 靠它注册全局 deepspeed config,
    # from_pretrained 期间 is_deepspeed_zero3_enabled() 才为真, 才会用
    # deepspeed.zero.Init() 分片加载权重。顺序反了会导致权重加载异常
    # (实测初始 loss 从正常的 ~4 变成 ~15.5)。
    # load_best_model_at_end 需要保留优化器状态, 且与 DeepSpeed + save_only_model 互斥
    load_best = (args.load_best_model_at_end
                 and not args.save_only_model
                 and args.eval_strategy != 'no'
                 and args.save_strategy != 'no')
    if not load_best and args.eval_strategy != 'no':
        logger.info('不会自动回滚到 eval_loss 最优的 checkpoint (--load_best_model_at_end 可开启)')

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        dataloader_num_workers=args.num_workers,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to='tensorboard',
        seed=args.seed,
        deepspeed=args.deepspeed,
        load_best_model_at_end=load_best,
        metric_for_best_model='eval_loss' if load_best else None,
        greater_is_better=False if load_best else None,
    )

    # ---------------- 2. 模型 ----------------
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    sparse_overrides = collect_sparse_overrides(args)

    model, tokenizer, config = load_model_and_tokenizer(
        model_path=args.model_path,
        dtype=dtype,
        attn_impl=args.attn_impl,
        sparse=args.sparse,
        sparse_overrides=sparse_overrides,
        local_code=args.local_code,
    )

    # 稀疏分支只在 kv_seq_len >= dense_len 时才会走到, 这里提前提醒
    if args.sparse:
        dense_len = config.sparse_config['dense_len']
        if 0 < dense_len and args.max_length <= dense_len:
            logger.warning(
                'max_length=%d <= dense_len=%d: 每个 batch 都会落到 InfLLMv2Attention '
                '的稠密分支, 稀疏内核一次都不会被调用。把 max_length 调大, '
                '或用 --dense_len -1 强制始终稀疏。',
                args.max_length, dense_len,
            )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False

    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Full-parameter training, trainable = %.2fB', trainable / 1e9)

    # ---------------- 3. 数据 ----------------
    # 只让 rank0 做 tokenize/打包, 其余 rank 等它写完缓存再读, 避免 4 份重复计算
    with training_args.main_process_first(desc='dataset preprocessing'):
        train_ds, val_ds = build_dataset(
            data_path=args.data_path,
            tokenizer=tokenizer,
            max_length=args.max_length,
            val_ratio=args.val_ratio,
            loss_on=args.loss_on,
            pack=args.pack,
            min_length=args.min_length,
            num_proc=args.preprocess_workers,
            max_samples=args.max_samples,
        )
    collator = DataCollator(tokenizer, max_length=args.max_length, pad_to_multiple_of=8)

    # ---------------- 4. Trainer ----------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    # MiniCPM 的 forward 带 **kwargs, Trainer 据此推断它支持 num_items_in_batch
    # (trainer.py: model_accepts_loss_kwargs = 签名里有 VAR_KEYWORD), 但它其实是
    # 把 kwargs 吞掉、照常算 per-device 平均 CE。这个误判会同时造成两个问题:
    #   1) loss *= accelerator.num_processes  -> 多卡时 loss 和梯度被放大 world_size 倍
    #   2) 跳过 loss /= gradient_accumulation_steps -> 梯度累积时再放大 accum 倍
    # 实测 4 卡下初始 loss 从正常的 ~4.2 变成 ~17。置 False 一次修好两处
    # (Trainer.compute_loss 的 docstring 也是这么建议的)。
    trainer.model_accepts_loss_kwargs = False

    # ---------------- 5. 训练 ----------------
    last_checkpoint = None
    if os.path.isdir(args.output_dir):
        last_checkpoint = get_last_checkpoint(args.output_dir)
    if last_checkpoint:
        logger.info('Resuming from %s', last_checkpoint)
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        logger.info('Peak GPU memory: %.1f GB / %.1f GB (%.0f%%)', peak, total, peak / total * 100)

    # ---------------- 6. 保存 ----------------
    logger.info('Saving to %s', args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info('Done.')


if __name__ == '__main__':
    main()
