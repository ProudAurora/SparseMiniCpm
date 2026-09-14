"""Gradio web UI for chatting with the CPM8B (MiniCPM4-8B) model.

Reuses the model-loading logic from run_inference.py. The model is loaded
once at startup and reused across requests; responses stream token by token.
"""

import argparse
import os
import threading

# Local traffic must bypass any HTTP(S)_PROXY set in the shell, otherwise
# gradio's own startup self-check (a request to localhost) gets routed
# through the proxy and fails.
os.environ["NO_PROXY"] = ",".join(
    filter(None, [os.environ.get("NO_PROXY", ""), "localhost,127.0.0.1,0.0.0.0"])
)
os.environ["no_proxy"] = os.environ["NO_PROXY"]

import gradio as gr
import torch
from transformers import TextIteratorStreamer

from run_inference import DEFAULT_MODEL_DIR, load_model

tokenizer = None
model = None
device = None


def build_prompt(message: str, history: list):
    messages = []
    for turn in history:
        if turn["role"] in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": message})
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)


def respond(message, history, max_new_tokens, temperature, top_p):
    input_ids = build_prompt(message, history)
    attention_mask = torch.ones_like(input_ids)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=top_p,
        streamer=streamer,
    )

    thread = threading.Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    partial = ""
    for token in streamer:
        partial += token
        yield partial
    thread.join()


def main():
    global tokenizer, model, device

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    device = args.device
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Loading model from {args.model_dir} on {device} ({dtype}) ...")
    tokenizer, model = load_model(args.model_dir, device, dtype)
    print("Model loaded.")

    demo = gr.ChatInterface(
        fn=respond,
        type="messages",
        title="MiniCPM4-8B (CPM8B) 推理测试",
        description=f"模型路径: {args.model_dir}",
        additional_inputs=[
            gr.Slider(1, 2048, value=512, step=1, label="max_new_tokens"),
            gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="temperature"),
            gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="top_p"),
        ],
    )
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
