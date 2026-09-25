"""
模型推理测试脚本
用于测试微调后的模型
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--input_text", type=str, default="你好，请介绍一下你自己", help="输入文本")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.8, help="温度")
    parser.add_argument("--top_p", type=float, default=0.8, help="top-p")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).cuda()

    messages = [{"role": "user", "content": args.input_text}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"\nInput:\n{text}")

    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    print(f"\nGenerating (max={args.max_new_tokens})...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            eos_token_id=[tokenizer.eos_token_id, 73440],
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nResponse:\n{response}")


if __name__ == "__main__":
    main()
