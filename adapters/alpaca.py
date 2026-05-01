"""Alpaca-style instruction/output JSON arrays."""
import json
import os
from typing import Iterator

from .base import Adapter, SFTRecord, Turn, scrub


class AlpacaAdapter(Adapter):
    kind = "sft"

    def __iter__(self) -> Iterator[SFTRecord]:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for row in data:
            if not isinstance(row, dict):
                continue
            instruction = scrub(row.get("instruction", ""))
            output = scrub(row.get("output", ""))
            if not instruction or not output:
                continue
            yield SFTRecord(
                source=self.source,
                turns=[
                    Turn(role="user", content=instruction),
                    Turn(role="assistant", content=output),
                ],
            )
