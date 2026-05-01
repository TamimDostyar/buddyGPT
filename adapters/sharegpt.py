"""ShareGPT-style JSONL (e.g. data/chatalpaca-20k.json)."""
import json
import os
from typing import Iterator

from .base import Adapter, SFTRecord, Turn, scrub


ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}


class ShareGPTAdapter(Adapter):
    kind = "sft"

    def __iter__(self) -> Iterator[SFTRecord]:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                raw_turns = obj.get("conversations") or obj.get("messages") or []
                turns = []
                for t in raw_turns:
                    if not isinstance(t, dict):
                        continue
                    role_raw = t.get("from") or t.get("role") or ""
                    role = ROLE_MAP.get(role_raw.lower())
                    if role not in ("user", "assistant"):
                        continue
                    content = scrub(t.get("value") or t.get("content") or "")
                    if not content:
                        continue
                    turns.append(Turn(role=role, content=content))

                if len(turns) < 2:
                    continue
                yield SFTRecord(source=self.source, turns=turns)
