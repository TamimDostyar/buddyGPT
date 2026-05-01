"""Adapter for plain `.txt` files used as pretraining corpus."""
import os
from typing import Iterator

from .base import Adapter, PretrainRecord, scrub


class PlainTextAdapter(Adapter):
    kind = "pretrain"

    def __init__(self, path: str, source: str, min_words: int = 4):
        super().__init__(path, source)
        self.min_words = min_words

    def __iter__(self) -> Iterator[PretrainRecord]:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for raw in f:
                line = scrub(raw)
                if not line:
                    continue
                if len(line.split()) < self.min_words:
                    continue
                yield PretrainRecord(source=self.source, text=line)
