"""Shared model loading, generation, and text-matching helpers for the benchmark scripts.

Model weights and eval datasets both live outside this repo (sibling directories
of SparseMiniCpm under autodl-tmp), so paths are resolved relative to this file
rather than assumed to be fixed absolute locations -- same convention as
inference_test/run_inference.py.
"""

import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "CPM8B"
DEFAULT_EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_model(model_dir: Path, device: str, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        dtype=dtype,
    ).to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate_response(
    tokenizer,
    model,
    prompt: str,
    device: str,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 0.8,
    top_p: float = 0.8,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)

    output_ids = model.generate(inputs, **gen_kwargs)
    return tokenizer.decode(
        output_ids[0][inputs.shape[1] :], skip_special_tokens=True
    )


def read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_final_number(text: str):
    """Best-effort extraction of a final numeric answer from free-form text.

    Tries \\boxed{...} first (matches the gsm8k/eval.yaml prompt template),
    then falls back to the last number that appears in the text.
    """
    boxed = re.search(r"\\boxed\{([^}]*)\}", text)
    candidate = boxed.group(1) if boxed else None
    if candidate is None:
        numbers = _NUMBER_RE.findall(text)
        candidate = numbers[-1] if numbers else None
    if candidate is None:
        return None
    candidate = candidate.replace(",", "").strip().rstrip(".")
    try:
        return float(candidate)
    except ValueError:
        return None


def numbers_match(pred, gold) -> bool:
    if pred is None or gold is None:
        return False
    return abs(pred - gold) < 1e-4


# Models end their answer either in the 'Answer: X' form the prompt asks for,
# or in BBH's own few-shot style ('So the answer is X.'); accept both.
_ANSWER_RE = re.compile(r"(?:answer\s*(?:is|:)\s*)(.+)", re.IGNORECASE)
# Requires the closing paren so prose like 'a triangle' is not read as option A.
_OPTION_LETTER_RE = re.compile(r"^\(?([a-z])\)")
_OPTION_ITEM_RE = re.compile(r"^\(([A-Za-z])\)\s*(.+)$", re.MULTILINE)
# yes/no and true/false mean the same thing here, but BBH subtasks disagree on
# which vocabulary the gold label uses, so fold them together before comparing.
_BOOL_ALIASES = {"yes": "true", "no": "false"}


def extract_final_answer(text: str) -> str:
    """Best-effort extraction of a short final answer from free-form text.

    Looks for the last 'Answer: ...' / 'the answer is ...' occurrence, falling
    back to the last non-empty line of the response.
    """
    matches = _ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def normalize_answer(text: str) -> str:
    text = str(text).strip().lower().rstrip(".")
    text = text.strip("()")
    text = text.strip()
    return _BOOL_ALIASES.get(text, text)


def parse_options(text: str) -> dict:
    """Parses a BBH 'Options:' block into {letter: option text}."""
    return {
        m.group(1).lower(): m.group(2).strip()
        for m in _OPTION_ITEM_RE.finditer(text)
    }


def answers_match(pred: str, gold: str, options: dict = None) -> bool:
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    if pred_norm == gold_norm:
        return True

    gold_is_option = len(gold_norm) == 1 and gold_norm.isalpha()
    if not gold_is_option:
        return False

    # Models answer multiple choice either as '(A) 12/02/1986' ...
    option = _OPTION_LETTER_RE.match(pred_norm)
    if option:
        return option.group(1) == gold_norm
    # ... or with the option's text instead of its letter, so look the text up
    # in the question's own options block rather than relying on the model
    # following a format instruction.
    if options and gold_norm in options:
        return normalize_answer(options[gold_norm]) == pred_norm
    return False


def print_detail(label: str, gold, pred, correct: bool):
    """Prints a one-line per-example gold/pred comparison.

    Uses tqdm.write so the output interleaves cleanly with a live progress bar
    instead of tearing it apart. The question text and the full model response
    are deliberately left out -- both are kept in the predictions jsonl.
    """
    mark = "correct" if correct else "WRONG  "
    tqdm.write(f"[{label}] {mark}  gold={gold!r}  pred={pred!r}")
