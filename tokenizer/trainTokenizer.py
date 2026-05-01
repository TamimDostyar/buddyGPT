"""Train BPE on data/clean/*.jsonl; writes tokenizer/tokenizer.json."""
import argparse
import json
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
DATA_DIR = os.path.join(REPO_ROOT, "data")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
TOKENIZER_DIR = os.path.join(REPO_ROOT, "tokenizer")

PRETRAIN_PATH = os.path.join(CLEAN_DIR, "pretrain.jsonl")
SFT_PATH = os.path.join(CLEAN_DIR, "sft.jsonl")

SPECIAL_TOKENS = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>", "<|end|>"]


def iter_texts():
    for path in (PRETRAIN_PATH, SFT_PATH):
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                text = rec.get("text") or ""
                if isinstance(text, str) and text.strip():
                    yield text


def train(vocab_size: int, min_frequency: int):
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    except ImportError:
        print(
            "The `tokenizers` package is required. Install with:\n"
            "    pip install tokenizers",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    texts = list(iter_texts())
    if not texts:
        print(
            "No training text found. Run `python makeJson.py` first to create "
            "data/clean/pretrain.jsonl and data/clean/sft.jsonl.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Training BPE on {len(texts)} text records, target vocab {vocab_size}...")
    tokenizer.train_from_iterator(texts, trainer=trainer)

    tokenizer_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
    tokenizer.save(tokenizer_path)

    try:
        tokenizer.model.save(TOKENIZER_DIR)
    except Exception:
        pass

    print(f"Tokenizer saved to {tokenizer_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    sample = "User: Hello\nAssistant: Hey! BuddyOS here."
    encoded = tokenizer.encode(sample)
    print(f"Sample:  {sample!r}")
    print(f"Tokens:  {encoded.tokens[:30]}{'...' if len(encoded.tokens) > 30 else ''}")
    print(f"IDs:     {encoded.ids[:30]}{'...' if len(encoded.ids) > 30 else ''}")
    print(f"Length:  {len(encoded.ids)} tokens")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8000,
        help="Target vocab size (default 8000; 8k-16k is good for tiny models).",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Min frequency a pair must have to be merged (default 2).",
    )
    args = parser.parse_args()

    train(args.vocab_size, args.min_frequency)


if __name__ == "__main__":
    main()
