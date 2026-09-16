"""Evaluate the model on GSM8K (grade-school math word problems).

Reads eval/gsm8k_jsonl/<config>/<split>.jsonl, prompts the model to solve
each problem and box the final answer, extracts the boxed/last number from
the response, and scores it against the gold answer (the number after
'####' in the reference solution).
"""

import argparse
import time

import torch
from tqdm import tqdm

from common import (
    DEFAULT_EVAL_DIR,
    DEFAULT_MODEL_DIR,
    RESULTS_DIR,
    extract_final_number,
    generate_response,
    load_model,
    numbers_match,
    print_detail,
    read_jsonl,
    write_jsonl,
)

PROMPT_TEMPLATE = (
    "Solve the following math problem efficiently and clearly. The last line "
    "of your response should be of the following format: 'Therefore, the "
    "final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without "
    "quotes) where ANSWER is just the final number that solves the problem. "
    "Think step by step before answering.\n\n{question}"
)


def gold_answer(answer_field: str) -> float:
    tail = answer_field.split("####")[-1]
    return float(tail.replace(",", "").strip())


def evaluate_gsm8k(
    tokenizer,
    model,
    device: str,
    eval_dir: str = str(DEFAULT_EVAL_DIR),
    config: str = "main",
    split: str = "test",
    limit: int = None,
    max_new_tokens: int = 512,
    show_details: bool = False,
):
    """Runs GSM8K end to end and returns (predictions, summary_dict)."""
    data_path = f"{eval_dir}/gsm8k_jsonl/{config}/{split}.jsonl"
    examples = read_jsonl(data_path)
    if limit:
        examples = examples[:limit]

    predictions = []
    correct = 0
    start = time.time()
    progress = tqdm(examples, desc="gsm8k", unit="ex")
    for i, ex in enumerate(progress):
        prompt = PROMPT_TEMPLATE.format(question=ex["question"])
        response = generate_response(tokenizer, model, prompt, device, max_new_tokens=max_new_tokens)
        pred = extract_final_number(response)
        gold = gold_answer(ex["answer"])
        is_correct = numbers_match(pred, gold)
        correct += int(is_correct)
        predictions.append(
            {
                "index": i,
                "question": ex["question"],
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "response": response,
            }
        )
        if show_details:
            print_detail(f"gsm8k {i + 1}/{len(examples)}", gold, pred, is_correct)
        progress.set_postfix(acc=f"{correct / (i + 1):.4f}", correct=f"{correct}/{i + 1}")

    elapsed = time.time() - start
    accuracy = correct / len(examples) if examples else 0.0
    summary = {
        "task": "gsm8k",
        "config": config,
        "split": split,
        "num_examples": len(examples),
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_seconds": elapsed,
    }
    return predictions, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--eval-dir", type=str, default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--config", type=str, default="main", choices=["main", "socratic"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N examples")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None, help="Predictions jsonl output path")
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Print the question, full model response, gold vs pred answer and correctness for every example",
    )
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    print(f"Loading model from {args.model_dir} on {args.device} ({dtype}) ...")
    tokenizer, model = load_model(args.model_dir, args.device, dtype)

    predictions, summary = evaluate_gsm8k(
        tokenizer,
        model,
        args.device,
        eval_dir=args.eval_dir,
        config=args.config,
        split=args.split,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        show_details=args.show_details,
    )

    print(
        f"\nGSM8K ({args.config}/{args.split}): {summary['correct']}/{summary['num_examples']} "
        f"= {summary['accuracy']:.4f} in {summary['elapsed_seconds']:.1f}s"
    )

    output_path = args.output or str(RESULTS_DIR / "gsm8k_predictions.jsonl")
    write_jsonl(output_path, predictions)
    print(f"Wrote predictions to {output_path}")


if __name__ == "__main__":
    main()
