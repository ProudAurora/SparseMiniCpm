"""Evaluate the model on BBH (BIG-Bench Hard).

Reads every eval/bbh_jsonl/<task>.jsonl file (one per BBH subtask), prompts
the model zero-shot to think step by step and end with an 'Answer: ...'
line, extracts that answer, and scores it against the gold target with
normalized exact match. Reports per-task accuracy plus the macro-average
across tasks, which is the standard way BBH results are reported.
"""

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from common import (
    DEFAULT_EVAL_DIR,
    DEFAULT_MODEL_DIR,
    RESULTS_DIR,
    answers_match,
    extract_final_answer,
    generate_response,
    load_model,
    parse_options,
    print_detail,
    read_jsonl,
    write_jsonl,
)

# Two things this wording has to get right, both learned the hard way:
# 1. The brevity constraint must clearly scope to the text after 'Answer:'.
#    Phrasing it as 'no explanation' up front makes the model skip its chain of
#    thought entirely (median response dropped 573 -> 24 chars), which collapses
#    every multi-step task.
# 2. It must not describe an answer format (option letters, True/False). An 8B
#    model copies the described format onto questions it does not fit, e.g.
#    inventing 'Answer: (E)' for a question that lists no options.
PROMPT_TEMPLATE = (
    "Answer the following question.\n"
    "First work through your reasoning step by step.\n"
    "Then, on the very last line, write 'Answer: <answer>' -- put only the "
    "final answer after 'Answer:', with nothing following it.\n\n{input}"
)


def discover_tasks(bbh_dir: Path):
    return sorted(p.stem for p in bbh_dir.glob("*.jsonl"))


def evaluate_bbh(
    tokenizer,
    model,
    device: str,
    eval_dir: str = str(DEFAULT_EVAL_DIR),
    tasks=None,
    limit: int = None,
    max_new_tokens: int = 512,
    show_details: bool = False,
):
    """Runs BBH end to end and returns (predictions, summary_dict)."""
    bbh_dir = Path(eval_dir) / "bbh_jsonl"
    tasks = tasks or discover_tasks(bbh_dir)

    # Load every subtask up front so the progress bar can span the whole run
    # rather than restarting for each of the 27 subtasks.
    task_examples = {}
    for task in tasks:
        examples = read_jsonl(bbh_dir / f"{task}.jsonl")
        task_examples[task] = examples[:limit] if limit else examples

    predictions = []
    per_task_correct = {task: 0 for task in tasks}
    per_task_total = {task: len(task_examples[task]) for task in tasks}
    done = 0
    correct = 0
    start = time.time()

    progress = tqdm(total=sum(per_task_total.values()), desc="bbh", unit="ex")
    for task in tasks:
        for i, ex in enumerate(task_examples[task]):
            prompt = PROMPT_TEMPLATE.format(input=ex["input"])
            response = generate_response(tokenizer, model, prompt, device, max_new_tokens=max_new_tokens)
            pred = extract_final_answer(response)
            is_correct = answers_match(pred, ex["target"], parse_options(ex["input"]))
            per_task_correct[task] += int(is_correct)
            correct += int(is_correct)
            done += 1
            predictions.append(
                {
                    "task": task,
                    "index": i,
                    "input": ex["input"],
                    "gold": ex["target"],
                    "pred": pred,
                    "correct": is_correct,
                    "response": response,
                }
            )
            if show_details:
                print_detail(
                    f"bbh:{task} {i + 1}/{per_task_total[task]}", ex["target"], pred, is_correct
                )
            progress.update(1)
            progress.set_postfix(
                task=task, acc=f"{correct / done:.4f}", correct=f"{correct}/{done}"
            )

        task_acc = per_task_correct[task] / per_task_total[task] if per_task_total[task] else 0.0
        tqdm.write(f"  [done] {task}: {per_task_correct[task]}/{per_task_total[task]} = {task_acc:.4f}")
    progress.close()

    elapsed = time.time() - start
    per_task_accuracy = {
        task: (per_task_correct[task] / per_task_total[task] if per_task_total[task] else 0.0)
        for task in tasks
    }
    macro_avg = sum(per_task_accuracy.values()) / len(per_task_accuracy) if per_task_accuracy else 0.0
    summary = {
        "task": "bbh",
        "tasks": tasks,
        "num_examples": sum(per_task_total.values()),
        "per_task_accuracy": per_task_accuracy,
        "macro_avg_accuracy": macro_avg,
        "elapsed_seconds": elapsed,
    }
    return predictions, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--eval-dir", type=str, default=str(DEFAULT_EVAL_DIR))
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated subtask names, default: all subtasks found under bbh_jsonl/",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N examples per task")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None, help="Predictions jsonl output path")
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Print the input, full model response, gold vs pred answer and correctness for every example",
    )
    args = parser.parse_args()

    tasks = args.tasks.split(",") if args.tasks else None

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    print(f"Loading model from {args.model_dir} on {args.device} ({dtype}) ...")
    tokenizer, model = load_model(args.model_dir, args.device, dtype)

    predictions, summary = evaluate_bbh(
        tokenizer,
        model,
        args.device,
        eval_dir=args.eval_dir,
        tasks=tasks,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        show_details=args.show_details,
    )

    print(f"\n=== BBH per-task accuracy (in {summary['elapsed_seconds']:.1f}s) ===")
    for task, acc in summary["per_task_accuracy"].items():
        print(f"  {task}: {acc:.4f}")
    print(f"\nBBH macro-average over {len(summary['tasks'])} tasks: {summary['macro_avg_accuracy']:.4f}")

    output_path = args.output or str(RESULTS_DIR / "bbh_predictions.jsonl")
    write_jsonl(output_path, predictions)
    print(f"Wrote predictions to {output_path}")


if __name__ == "__main__":
    main()
