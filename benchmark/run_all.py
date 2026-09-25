"""Run all benchmarks (GSM8K + BBH) against a single loaded model and write
a combined results/summary.json.

Loading the 8B model once and reusing it for every task avoids paying the
model-load cost twice, which matters more once these runs need to be
repeated before/after sparse-attention training for comparison.
"""

import argparse
import json

import torch

from common import DEFAULT_EVAL_DIR, DEFAULT_MODEL_DIR, RESULTS_DIR, load_model, write_jsonl
from run_bbh import evaluate_bbh
from run_gsm8k import evaluate_gsm8k


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--eval-dir", type=str, default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N examples per task/subtask")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", type=str, default="baseline", help="Label for this run, e.g. 'dense-baseline' or 'sparse-v1'")
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Print the input, full model response, gold vs pred answer and correctness for every example",
    )
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    print(f"Loading model from {args.model_dir} on {args.device} ({dtype}) ...")
    tokenizer, model = load_model(args.model_dir, args.device, dtype)

    gsm8k_predictions, gsm8k_summary = evaluate_gsm8k(
        tokenizer,
        model,
        args.device,
        eval_dir=args.eval_dir,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        show_details=args.show_details,
    )
    write_jsonl(RESULTS_DIR / f"{args.tag}_gsm8k_predictions.jsonl", gsm8k_predictions)

    bbh_predictions, bbh_summary = evaluate_bbh(
        tokenizer,
        model,
        args.device,
        eval_dir=args.eval_dir,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        show_details=args.show_details,
    )
    write_jsonl(RESULTS_DIR / f"{args.tag}_bbh_predictions.jsonl", bbh_predictions)

    summary = {"tag": args.tag, "model_dir": args.model_dir, "gsm8k": gsm8k_summary, "bbh": bbh_summary}

    print("\n=== Summary ===")
    print(f"GSM8K accuracy:       {gsm8k_summary['accuracy']:.4f}  ({gsm8k_summary['correct']}/{gsm8k_summary['num_examples']})")
    print(f"BBH macro-avg acc:    {bbh_summary['macro_avg_accuracy']:.4f}  ({bbh_summary['num_examples']} examples across {len(bbh_summary['tasks'])} subtasks)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / f"{args.tag}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nWrote summary to {summary_path}")


if __name__ == "__main__":
    main()
