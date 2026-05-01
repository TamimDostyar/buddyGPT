"""Load JSONL text streams and build token tensors for training."""
import json
import os
from typing import List

import torch


SPECIAL_TOKENS = ("<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>", "<|end|>")


def load_tokenizer(tokenizer_path: str):
    from tokenizers import Tokenizer
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. "
            "Run `python scripts/trainTokenizer.py` first."
        )
    return Tokenizer.from_file(tokenizer_path)


def _iter_texts(path: str):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text")
            if isinstance(text, str) and text.strip():
                yield text


def build_token_stream(
    tokenizer,
    paths: List[str],
    bos_id: int,
    eos_id: int,
) -> torch.Tensor:
    ids: List[int] = []
    total_docs = 0
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        for text in _iter_texts(path):
            encoded = tokenizer.encode(text).ids
            if not encoded:
                continue
            ids.append(bos_id)
            ids.extend(encoded)
            ids.append(eos_id)
            total_docs += 1

    if not ids:
        raise RuntimeError(
            "No training text found. Run `python scripts/makeJson.py` and "
            "`python scripts/trainTokenizer.py` first."
        )

    print(f"Tokenized {total_docs} documents into {len(ids):,} tokens")
    return torch.tensor(ids, dtype=torch.long)


def train_val_split(tokens: torch.Tensor, val_fraction: float):
    n = len(tokens)
    n_val = max(1, int(n * val_fraction))
    return tokens[:-n_val], tokens[-n_val:]


def make_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
