"""Build data/clean/pretrain.jsonl and data/clean/sft.jsonl from registered adapters."""
import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from adapters import (
    AlpacaAdapter,
    ShareGPTAdapter,
    MultiTurnAdapter,
    PlainTextAdapter,
)
from adapters.base import SFTRecord, PretrainRecord


DATA_DIR = os.path.join(REPO_ROOT, "data")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")

PRETRAIN_PATH = os.path.join(CLEAN_DIR, "pretrain.jsonl")
SFT_PATH = os.path.join(CLEAN_DIR, "sft.jsonl")
LEGACY_FILES = ("train.json", "train.jsonl")


USER_TAG = "<|user|>"
ASSISTANT_TAG = "<|assistant|>"
END_TAG = "<|end|>"


def _p(name: str) -> str:
    return os.path.join(DATA_DIR, name)


DEFAULT_SOURCES = [
    (ShareGPTAdapter(_p("chatalpaca-20k.json"), "chatalpaca"),),
    (AlpacaAdapter(_p("dailyDialog.json"), "dailyDialog"),),
    (AlpacaAdapter(_p("generated.json"), "generated"),),
    (MultiTurnAdapter(_p("multiTurn.json"), "multiTurn"),),
    (PlainTextAdapter(_p("tinystories.txt"), "tinystories"),),
]

OPTIONAL_SOURCES = {
    "shakespeare": PlainTextAdapter(_p("shakes.txt"), "shakespeare", min_words=5),
}


def render_chat(turns) -> str:
    parts = []
    for t in turns:
        tag = USER_TAG if t.role == "user" else ASSISTANT_TAG
        suffix = END_TAG if t.role == "assistant" else ""
        parts.append(f"{tag}\n{t.content}{suffix}")
    return "\n".join(parts)


def collect(include: List[str]):
    pretrain_records: list = []
    sft_records: list = []

    all_adapters = [a for group in DEFAULT_SOURCES for a in group]
    for opt in include:
        if opt in OPTIONAL_SOURCES:
            all_adapters.append(OPTIONAL_SOURCES[opt])

    for adapter in all_adapters:
        count = 0
        for rec in adapter.records():
            if isinstance(rec, SFTRecord):
                obj = rec.to_dict()
                obj["text"] = render_chat(rec.turns)
                sft_records.append(obj)
            elif isinstance(rec, PretrainRecord):
                pretrain_records.append(rec.to_dict())
            count += 1
        print(f"  {adapter.source:<14} ({adapter.kind:<8}): {count} records")

    return pretrain_records, sft_records


def write_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        choices=list(OPTIONAL_SOURCES.keys()),
        help="Include an optional source (repeatable).",
    )
    args = parser.parse_args()

    pretrain_records, sft_records = collect(args.include)

    write_jsonl(pretrain_records, PRETRAIN_PATH)
    write_jsonl(sft_records, SFT_PATH)

    for legacy in LEGACY_FILES:
        p = os.path.join(CLEAN_DIR, legacy)
        if os.path.exists(p):
            os.remove(p)

    print(f"Pretrain: {len(pretrain_records)} -> {PRETRAIN_PATH}")
    print(f"SFT: {len(sft_records)} -> {SFT_PATH}")


if __name__ == "__main__":
    main()
