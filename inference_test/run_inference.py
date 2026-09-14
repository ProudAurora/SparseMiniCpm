"""Quick inference smoke test for the CPM8B (MiniCPM4-8B) model.

Model weights live outside this repo at ../../models/CPM8B (i.e.
<repo-parent>/models/CPM8B), so this script locates them relative to
its own path rather than assuming a fixed absolute location.
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "CPM8B"


def load_model(model_dir: Path, device: str, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        dtype=dtype,
    ).to(device)
    model.eval()
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt", type=str, default="你好，请自我介绍一下。")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    print(f"Loading model from {args.model_dir} on {args.device} ({dtype}) ...")
    tokenizer, model = load_model(args.model_dir, args.device, dtype)

    messages = [{"role": "user", "content": args.prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(args.device)

    with torch.no_grad():
        output_ids = model.generate(
            inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    response = tokenizer.decode(
        output_ids[0][inputs.shape[1]:], skip_special_tokens=True
    )
    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Response ===")
    print(response)


if __name__ == "__main__":
    main()
