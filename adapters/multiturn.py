"""Multi-turn JSON list (e.g. data/multiTurn.json)."""
import json
import os
from typing import Iterator

from .base import Adapter, SFTRecord, Turn, scrub


class MultiTurnAdapter(Adapter):
    kind = "sft"

    def __iter__(self) -> Iterator[SFTRecord]:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for convo in data:
            if not isinstance(convo, dict):
                continue
            raw_turns = convo.get("turns", [])
            turns = []
            for t in raw_turns:
                if not isinstance(t, dict):
                    continue
                role = t.get("role", "")
                if role not in ("user", "assistant"):
                    continue
                content = scrub(t.get("content", ""))
                if not content:
                    continue
                turns.append(Turn(role=role, content=content))
            if len(turns) < 2:
                continue
            yield SFTRecord(source=self.source, turns=turns)
