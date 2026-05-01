"""Default hyperparameters for small GPT training."""
from dataclasses import dataclass


@dataclass
class TrainConfig:
    vocab_size: int = 8000
    block_size: int = 256
    n_embed: int = 192
    n_head: int = 6
    n_layer: int = 4
    dropout: float = 0.1

    batch_size: int = 16
    max_steps: int = 3000
    eval_every: int = 200
    eval_batches: int = 20
    warmup_steps: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    val_fraction: float = 0.02
    seed: int = 1337

    tokenizer_path: str = "tokenizer/tokenizer.json"
    sft_path: str = "data/clean/sft.jsonl"
    pretrain_path: str = "data/clean/pretrain.jsonl"
    checkpoint_path: str = "models/buddyGPT.pt"

    include_pretrain: bool = False
