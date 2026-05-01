"""Dataset adapters yielding canonical SFT or pretrain records."""
from .base import Adapter, SFTRecord, PretrainRecord
from .alpaca import AlpacaAdapter
from .sharegpt import ShareGPTAdapter
from .multiturn import MultiTurnAdapter
from .plaintext import PlainTextAdapter

__all__ = [
    "Adapter",
    "SFTRecord",
    "PretrainRecord",
    "AlpacaAdapter",
    "ShareGPTAdapter",
    "MultiTurnAdapter",
    "PlainTextAdapter",
]
