#!/usr/bin/env python3
"""Run the original LongBench benchmark with a local MiniCPM3 model.

This script intentionally reuses the prompt templates, generation lengths and
metrics from the official THUDM/LongBench repository.  Predictions are written
to the repository's standard ``pred/<model_name>`` layout, so the official
``eval.py`` can score them without conversion.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


LONGBENCH_DATASETS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
]

LONGBENCH_E_DATASETS = [
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]

# This follows the official LongBench pred.py behavior.  These few-shot/code
# tasks are passed as plain completion prompts rather than wrapped in chat.
NO_CHAT_TEMPLATE_DATASETS = {
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "lcc",
    "repobench-p",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a local MiniCPM3 checkpoint on original LongBench."
    )
    parser.add_argument(
        "--model-path",
        default="/root/autodl-tmp/models/CPM3B",
        help="Local MiniCPM3 model/checkpoint directory.",
    )
    parser.add_argument(
        "--model-name",
        default="minicpm3-4b-local",
        help="Name used for pred/<model-name> and result.json.",
    )
    parser.add_argument(
        "--longbench-root",
        default="/root/autodl-tmp/LongBench/LongBench",
        help="Directory containing official config/, metrics.py and eval.py.",
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parent.parent),
        help="Directory containing optional project model_utils.py.",
    )
    parser.add_argument(
        "--loader",
        choices=("auto", "project"),
        default="auto",
        help="auto=Transformers trust_remote_code; project=local model_utils.py.",
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated task names, or 'all'.",
    )
    parser.add_argument(
        "--attn-impl",
        dest="attn_impl",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="flash_attention_2",
        help="Attention implementation (project loader only).",
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="Enable InfLLM v2 sparse attention (project loader only).",
    )
    parser.add_argument(
        "--dense-len",
        dest="dense_len",
        type=int,
        default=None,
        help="Override sparse_config.dense_len; -1 forces sparse for every length.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of visible GPUs. Each process loads one full model.",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=32768,
        help="Total input + generated token budget.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate at most N examples per task; 0 means the full test set.",
    )
    parser.add_argument(
        "--cache-dir",
        default="/root/autodl-tmp/hf_cache",
        help="Hugging Face dataset cache directory.",
    )
    parser.add_argument(
        "--dataset-repo",
        default="zai-org/LongBench",
        help="Hugging Face dataset repository used when --data-dir is omitted.",
    )
    parser.add_argument(
        "--data-dir",
        default="",
        help="Optional local directory containing data/<task>.jsonl or <task>.jsonl.",
    )
    parser.add_argument(
        "--e",
        action="store_true",
        help="Evaluate LongBench-E instead of the original LongBench split.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing task predictions instead of resuming them.",
    )
    parser.add_argument(
        "--skip-score",
        action="store_true",
        help="Only generate predictions; do not run official eval.py.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_environment(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path)
    root = Path(args.longbench_root)
    required = [
        model_path / "config.json",
        root / "config" / "dataset2prompt.json",
        root / "config" / "dataset2maxlen.json",
        root / "metrics.py",
        root / "eval.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required files are missing:\n  " + "\n  ".join(missing))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation script.")
    visible = torch.cuda.device_count()
    if args.num_gpus < 1 or args.num_gpus > visible:
        raise ValueError(
            f"--num-gpus={args.num_gpus}, but only {visible} CUDA device(s) are visible."
        )
    if args.max_context <= 1024:
        raise ValueError("--max-context must be greater than 1024.")


def select_datasets(args: argparse.Namespace) -> List[str]:
    allowed = LONGBENCH_E_DATASETS if args.e else LONGBENCH_DATASETS
    if args.datasets.strip().lower() == "all":
        return list(allowed)
    selected = [item.strip() for item in args.datasets.split(",") if item.strip()]
    invalid = sorted(set(selected) - set(allowed))
    if invalid:
        raise ValueError(f"Unsupported dataset(s): {invalid}; allowed={allowed}")
    if not selected:
        raise ValueError("No datasets selected.")
    return selected


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(
    args: argparse.Namespace, device: torch.device
) -> tuple[Any, Any]:
    dtype = torch.bfloat16

    if args.loader == "project":
        project_root = str(Path(args.project_root).resolve())
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from model_utils import load_model_and_tokenizer as project_loader

        model, tokenizer, _ = project_loader(
            model_path=args.model_path,
            dtype=dtype,
            attn_impl=args.attn_impl,
            sparse=args.sparse,
            sparse_overrides={"dense_len": args.dense_len} if args.dense_len is not None else None,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

    model = model.to(device).eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def chat_input_ids(tokenizer: Any, prompt: str, dataset: str) -> torch.Tensor:
    if dataset in NO_CHAT_TEMPLATE_DATASETS:
        encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
        return encoded["input_ids"]

    messages = [{"role": "user", "content": prompt}]
    try:
        ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        ids = tokenizer(rendered, return_tensors="pt", truncation=False)["input_ids"]

    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    elif isinstance(ids, dict):
        ids = ids["input_ids"]
    return ids


def middle_truncate(input_ids: torch.Tensor, target_length: int) -> torch.Tensor:
    if input_ids.shape[-1] <= target_length:
        return input_ids
    left = (target_length + 1) // 2
    right = target_length - left
    if right == 0:
        return input_ids[:, :left]
    return torch.cat((input_ids[:, :left], input_ids[:, -right:]), dim=-1)


def count_valid_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Corrupt JSONL at {path}:{line_number}; use --overwrite."
                ) from exc
            count += 1
    return count


def load_longbench_data(args: argparse.Namespace, config_name: str) -> Any:
    """Load one task locally when possible, otherwise use the Hub dataset."""
    if args.data_dir:
        base = Path(args.data_dir).expanduser().resolve()
        candidates = [
            base / f"{config_name}.jsonl",
            base / "data" / f"{config_name}.jsonl",
        ]
        data_path = next((path for path in candidates if path.is_file()), None)
        if data_path is None:
            raise FileNotFoundError(
                f"Local LongBench task not found: {config_name}.jsonl; checked "
                + ", ".join(str(path) for path in candidates)
            )
        rows = []
        with data_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid LongBench JSONL at {data_path}:{line_number}"
                    ) from exc
        return rows

    return load_dataset(
        args.dataset_repo,
        config_name,
        split="test",
        cache_dir=args.cache_dir,
    )


def generate_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    dataset: str,
    max_context: int,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    # Reserve output tokens inside MiniCPM3's 32K total context window.
    input_budget = max_context - max_new_tokens
    input_ids = chat_input_ids(tokenizer, prompt, dataset)
    input_ids = middle_truncate(input_ids, input_budget).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    eos_token_id: Any = tokenizer.eos_token_id
    if dataset == "samsum":
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        eos_values = [tokenizer.eos_token_id]
        if newline_ids:
            eos_values.append(newline_ids[-1])
        eos_token_id = list(dict.fromkeys(x for x in eos_values if x is not None))

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output_ids[0, input_ids.shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def worker(
    rank: int,
    datasets: Sequence[str],
    args: argparse.Namespace,
    prompt_formats: Dict[str, str],
    generation_lengths: Dict[str, int],
) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    seed_everything(args.seed + rank)
    model, tokenizer = load_model_and_tokenizer(args, device)

    root = Path(args.longbench_root)
    pred_root = root / ("pred_e" if args.e else "pred") / args.model_name
    pred_root.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        config_name = f"{dataset}_e" if args.e else dataset
        data = load_longbench_data(args, config_name)
        total = len(data) if args.limit <= 0 else min(args.limit, len(data))
        output_path = pred_root / f"{dataset}.jsonl"

        if args.overwrite and output_path.exists():
            output_path.unlink()
        completed = count_valid_jsonl(output_path)
        if completed > total:
            raise RuntimeError(
                f"{output_path} has {completed} rows, expected no more than {total}; "
                "rerun with --overwrite."
            )
        if completed == total:
            print(f"[GPU {rank}] {dataset}: already complete ({total} rows)", flush=True)
            continue

        progress = tqdm(
            range(completed, total),
            initial=completed,
            total=total,
            desc=f"GPU{rank} {dataset}",
            dynamic_ncols=True,
        )
        with output_path.open("a", encoding="utf-8", buffering=1) as output_file:
            for index in progress:
                row = data[index]
                prompt = prompt_formats[dataset].format(**row)
                prediction = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    dataset=dataset,
                    max_context=args.max_context,
                    max_new_tokens=int(generation_lengths[dataset]),
                    device=device,
                )
                record = {
                    "pred": prediction,
                    "answers": row["answers"],
                    "all_classes": row["all_classes"],
                    "length": row["length"],
                    "sample_index": index,
                }
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    del model
    torch.cuda.empty_cache()


def run_official_score(args: argparse.Namespace) -> Path:
    root = Path(args.longbench_root)
    command = [sys.executable, "eval.py", "--model", args.model_name]
    if args.e:
        command.append("--e")
    subprocess.run(command, cwd=str(root), check=True)
    result_path = (
        root
        / ("pred_e" if args.e else "pred")
        / args.model_name
        / "result.json"
    )
    return result_path


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    validate_environment(args)
    datasets = select_datasets(args)

    root = Path(args.longbench_root)
    prompt_formats = read_json(root / "config" / "dataset2prompt.json")
    generation_lengths = read_json(root / "config" / "dataset2maxlen.json")

    print("=" * 72)
    print(f"Model:       {args.model_path}")
    print(f"LongBench:   {root}")
    print(f"Tasks:       {len(datasets)}")
    print(f"Visible GPU: {torch.cuda.device_count()}, using: {args.num_gpus}")
    print(f"Context:     {args.max_context:,}")
    print(f"Per-task N:  {'full' if args.limit <= 0 else args.limit}")
    print(f"Loader:      {args.loader}")
    print(f"Data source: {'local ' + args.data_dir if args.data_dir else args.dataset_repo}")
    print("=" * 72, flush=True)

    assignments = [datasets[rank :: args.num_gpus] for rank in range(args.num_gpus)]
    if args.num_gpus == 1:
        worker(0, assignments[0], args, prompt_formats, generation_lengths)
    else:
        context = mp.get_context("spawn")
        processes = []
        for rank, assigned in enumerate(assignments):
            process = context.Process(
                target=worker,
                args=(rank, assigned, args, prompt_formats, generation_lengths),
            )
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
        failed = [process.pid for process in processes if process.exitcode != 0]
        if failed:
            raise RuntimeError(f"LongBench worker process(es) failed: {failed}")

    if not args.skip_score:
        result_path = run_official_score(args)
        print(f"Official LongBench scores: {result_path}")


if __name__ == "__main__":
    main()
