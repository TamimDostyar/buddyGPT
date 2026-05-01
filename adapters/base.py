"""Adapter base class and shared types."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, List


URL_RE = re.compile(r"https?://\S+|www\.\S+")
SEPARATOR_RE = re.compile(r"^[\s\-=_*#~]{3,}$")


def scrub(text: str) -> str:
    if text is None:
        return ""
    s = str(text)
    s = URL_RE.sub("", s)
    if SEPARATOR_RE.match(s.strip()):
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    return s


@dataclass
class Turn:
    role: str
    content: str


@dataclass
class SFTRecord:
    source: str
    turns: List[Turn] = field(default_factory=list)

    def is_valid(self) -> bool:
        if len(self.turns) < 2:
            return False
        if not any(t.role == "user" for t in self.turns):
            return False
        if not any(t.role == "assistant" for t in self.turns):
            return False
        return all(t.content for t in self.turns)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "turns": [{"role": t.role, "content": t.content} for t in self.turns],
        }


@dataclass
class PretrainRecord:
    source: str
    text: str

    def is_valid(self) -> bool:
        return bool(self.text)

    def to_dict(self) -> dict:
        return {"source": self.source, "text": self.text}


class Adapter:
    kind: str = "sft"

    def __init__(self, path: str, source: str):
        self.path = path
        self.source = source

    def __iter__(self) -> Iterator:
        raise NotImplementedError

    def records(self) -> Iterable:
        for rec in self:
            if rec is not None and rec.is_valid():
                yield rec
