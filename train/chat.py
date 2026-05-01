"""Interactive chat with a trained checkpoint (same template as training)."""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.GPTModel import GPTModelStyle
from train.data import load_tokenizer


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def load_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    model = GPTModelStyle(
        vocab_size=cfg["vocab_size"],
        n_embed=cfg["n_embed"],
        block_size=cfg["block_size"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        dropout=0.0,
        device=device,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, ckpt["tokenizer_path"]


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt_ids,
    end_id: int,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    rep_penalty: float = 1.15,
    device: str = "cpu",
):
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated_new = []

    for _ in range(max_new_tokens):
        cond = idx[:, -model.block_size :]
        logits, _ = model(cond)
        logits = logits[:, -1, :]

        if rep_penalty > 1.0 and generated_new:
            for t in set(generated_new[-64:]):
                logits[0, t] /= rep_penalty

        logits = logits / max(temperature, 1e-5)
        if top_k and top_k > 0:
            v, ix = torch.topk(logits, top_k)
            probs = F.softmax(v, dim=-1)
            choice = torch.multinomial(probs, num_samples=1)
            next_token = ix.gather(1, choice).item()
        else:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

        if next_token == end_id:
            break
        generated_new.append(next_token)
        idx = torch.cat(
            [idx, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1
        )

    return tokenizer.decode(generated_new).strip()


def chat_loop(model, tokenizer, cfg, device: str, **gen_kwargs):
    user_tag = tokenizer.token_to_id("<|user|>")
    asst_tag = tokenizer.token_to_id("<|assistant|>")
    end_id = tokenizer.token_to_id("<|end|>")
    if None in (user_tag, asst_tag, end_id):
        raise RuntimeError("Tokenizer is missing required special tokens.")

    print("BuddyGPT ready. Type 'q' to quit.")
    while True:
        try:
            msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if msg.lower() in {"q", "quit", "exit"}:
            break
        if not msg:
            continue

        body = tokenizer.encode(msg).ids
        prompt_ids = (
            [user_tag]
            + tokenizer.encode("\n").ids
            + body
            + tokenizer.encode("\n").ids
            + [asst_tag]
            + tokenizer.encode("\n").ids
        )
        if len(prompt_ids) > cfg["block_size"] - 16:
            prompt_ids = prompt_ids[-(cfg["block_size"] - 16):]

        reply = generate(model, tokenizer, prompt_ids, end_id, device=device, **gen_kwargs)
        print(f"AI: {reply}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/buddyGPT.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--rep-penalty", type=float, default=1.15)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = resolve(args.checkpoint)
    if not os.path.exists(ckpt_path):
        print(
            f"No checkpoint at {ckpt_path}. Train one first with "
            "`python train/train.py`.",
            file=sys.stderr,
        )
        sys.exit(1)

    model, cfg, tokenizer_rel = load_model(ckpt_path, device)
    tokenizer = load_tokenizer(resolve(tokenizer_rel))

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        rep_penalty=args.rep_penalty,
    )
    chat_loop(model, tokenizer, cfg, device, **gen_kwargs)


if __name__ == "__main__":
    main()
