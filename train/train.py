"""Train BuddyGPT on cleaned SFT (optional pretrain). Saves models/buddyGPT.pt."""
import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.GPTModel import GPTModelStyle
from train.config import TrainConfig
from train.data import (
    SPECIAL_TOKENS,
    build_token_stream,
    load_tokenizer,
    make_batch,
    train_val_split,
)


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return 0.1 * cfg.learning_rate + 0.9 * cfg.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, data, cfg: TrainConfig, device: str) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = make_batch(data, cfg.block_size, cfg.batch_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def build_model(cfg: TrainConfig, device: str) -> GPTModelStyle:
    return GPTModelStyle(
        vocab_size=cfg.vocab_size,
        n_embed=cfg.n_embed,
        block_size=cfg.block_size,
        n_head=cfg.n_head,
        n_layer=cfg.n_layer,
        dropout=cfg.dropout,
        device=device,
    ).to(device)


def parse_args(cfg: TrainConfig):
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=cfg.max_steps)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--block-size", type=int, default=cfg.block_size)
    p.add_argument("--n-embed", type=int, default=cfg.n_embed)
    p.add_argument("--n-head", type=int, default=cfg.n_head)
    p.add_argument("--n-layer", type=int, default=cfg.n_layer)
    p.add_argument("--learning-rate", type=float, default=cfg.learning_rate)
    p.add_argument("--dropout", type=float, default=cfg.dropout)
    p.add_argument("--include-pretrain", action="store_true")
    p.add_argument("--checkpoint", default=cfg.checkpoint_path)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    cfg = TrainConfig()
    args = parse_args(cfg)
    cfg.max_steps = args.max_steps
    cfg.batch_size = args.batch_size
    cfg.block_size = args.block_size
    cfg.n_embed = args.n_embed
    cfg.n_head = args.n_head
    cfg.n_layer = args.n_layer
    cfg.learning_rate = args.learning_rate
    cfg.dropout = args.dropout
    cfg.include_pretrain = args.include_pretrain
    cfg.checkpoint_path = args.checkpoint

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    print(f"Device: {device}")
    print(f"Config: {cfg}")

    tokenizer = load_tokenizer(resolve(cfg.tokenizer_path))
    vocab = tokenizer.get_vocab_size()
    cfg.vocab_size = vocab
    print(f"Vocab size: {vocab}")

    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    if bos_id is None or eos_id is None:
        raise RuntimeError(
            f"Tokenizer is missing special tokens. Expected all of: {SPECIAL_TOKENS}"
        )

    paths = [resolve(cfg.sft_path)]
    if cfg.include_pretrain:
        paths.append(resolve(cfg.pretrain_path))
    tokens = build_token_stream(tokenizer, paths, bos_id, eos_id)
    train_data, val_data = train_val_split(tokens, cfg.val_fraction)
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    start = time.time()
    for step in range(1, cfg.max_steps + 1):
        for g in optim.param_groups:
            g["lr"] = cosine_lr(step, cfg)

        x, y = make_batch(train_data, cfg.block_size, cfg.batch_size, device)
        _, loss = model(x, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()

        if step % 50 == 0 or step == 1:
            elapsed = time.time() - start
            tok_per_step = cfg.batch_size * cfg.block_size
            tps = (step * tok_per_step) / max(1e-6, elapsed)
            print(
                f"step {step:>5d} | loss {loss.item():.4f} | "
                f"lr {optim.param_groups[0]['lr']:.2e} | "
                f"{tps:,.0f} tok/s"
            )

        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            val_loss = evaluate(model, val_data, cfg, device)
            print(f"step {step} val_loss {val_loss:.4f}")

    ckpt_path = resolve(cfg.checkpoint_path)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.__dict__,
            "tokenizer_path": cfg.tokenizer_path,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")
    print("Run `python train/chat.py` to talk with it.")


if __name__ == "__main__":
    main()
