![PythonImage: ](asset/pythonIcon.png)

# BuddyGPT

A tiny, from-scratch chat model designed to run on slow PCs. The goal is a
model that can take an input and answer it coherently — not a chatbot that
hallucinates when a single-word question comes in. RAG can be layered on
top later.

## Project layout

```
buddyGPT/
├── adapters/          data-format adapters (ShareGPT / Alpaca / multi-turn / plain)
├── data/              raw datasets + data/clean/ cleaned corpus
├── models/            GPTModelStyle (nanoGPT-style tiny transformer)
├── notebook/          end-to-end notebook (systemBuilder.ipynb)
├── scripts/           data pipeline + tokenizer trainer
├── tokenizer/         BPE tokenizer artifacts (written by trainTokenizer.py)
├── train/             trainer + interactive chat
├── testGPT.py         convenience entrypoint -> train/chat.py
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

# 1) Build the clean training corpus
python scripts/makeJson.py

# 2) Train a small BPE tokenizer (8k vocab)
python scripts/trainTokenizer.py --vocab-size 8000

# 3) Train the model (~6M params, CPU-friendly)
python train/train.py --max-steps 3000

# 4) Chat with it
python testGPT.py
```

Or open `notebook/systemBuilder.ipynb` and run top to bottom.

## Data pipeline

Every data source goes through an adapter that normalizes it into one of two
canonical record types:

- **SFT** (`sft.jsonl`): `{"source", "turns": [{"role", "content"}, ...], "text"}`
  — every example, single-turn or multi-turn, is a list of alternating turns.
  `text` is the chat-templated rendering used for training.
- **Pretrain** (`pretrain.jsonl`): `{"source", "text"}` — raw text for
  causal-LM pretraining.

### Registered sources (default)

| Source              | File                         | Adapter              | Kind     |
| ------------------- | ---------------------------- | -------------------- | -------- |
| ChatAlpaca 20k      | `data/chatalpaca-20k.json`   | `ShareGPTAdapter`    | SFT      |
| Daily dialog        | `data/dailyDialog.json`      | `AlpacaAdapter`      | SFT      |
| Generated (BuddyOS) | `data/generated.json`        | `AlpacaAdapter`      | SFT      |
| Multi-turn          | `data/multiTurn.json`        | `MultiTurnAdapter`   | SFT      |
| TinyStories         | `data/tinystories.txt`       | `PlainTextAdapter`   | Pretrain |

Optional (off by default): `--include shakespeare` adds `data/shakes.txt`.

### Adding a new source

1. If it uses a new format, add a file to `adapters/` that subclasses
   `Adapter` and yields `SFTRecord` or `PretrainRecord`.
2. Register it in `DEFAULT_SOURCES` in `scripts/makeJson.py`.
3. Re-run `python scripts/makeJson.py`.

### Getting TinyStories (optional pretraining)

```bash
python scripts/getTinyStories.py --max-mb 50
```

## Chat template

Every SFT record uses this exact template, so prompts at inference time
must be built the same way:

```
<|user|>
{user message}
<|assistant|>
{assistant message}<|end|>
```

For inference, feed a prompt ending in `<|assistant|>\n` and stop on
`<|end|>`. `train/chat.py` handles this for you.

## Architecture (tiny GPT)

`models/GPTModel.py` is a standard nanoGPT-style causal transformer:
token + position embeddings → N transformer blocks (multi-head self-attention
+ GELU MLP, each with LayerNorm + residual) → final LayerNorm → LM head.

Default size (`train/config.py`):

| Param            | Value |
| ---------------- | ----- |
| vocab_size       | 8000  |
| block_size       | 256   |
| n_embed          | 192   |
| n_head           | 6     |
| n_layer          | 4     |
| approx params    | ~6 M  |

All values are CLI-overridable on `train/train.py`.

## Goals

- Grow from a proof-of-concept fine-tuned model into the full inference engine that powers all of buddySystem
- Replace Ollama entirely as the AI backend — no external runtime, no dependency on a host OS
- Run bare-metal inside the buddyOS kernel with a custom tensor allocator, not in user space
- Store model weights as a kernel resource under `/sys/kai/model.bin`, versioned by the OS
- Fine-tune continuously on idle CPU cycles using the user's own interaction logs as training data
- Become a model that is uniquely shaped by its user — sovereign, local, and impossible to rate-limit or deprecate

---

## Architecture (deployment view)

How buddyGPT fits into the system — today with Ollama, and the goal state inside the kernel.

```
  TODAY                                   GOAL
  ─────────────────────────               ─────────────────────────────────────

  User input                              User input
      │                                       │
      ▼                                       │  sys_ask()
  buddyCLI  (TypeScript)                      ▼
      │                               ┌── buddyOS Kernel ─────────────────┐
      │  HTTP  localhost:11434        │                                   │
      ▼                               │  ┌── kai/ AI Subsystem ─────────┐ │
  Ollama  runtime                     │  │                              │ │
      │                               │  │  ┌─────────────────────────┐ │ │
      ▼                               │  │  │       buddyGPT          │ │ │
  downloaded model                    │  │  │                         │ │ │
  (not yours, not local               │  │  │  1. tokenise input      │ │ │
   to the kernel)                     │  │  │  2. read kernel state   │ │ │
                                      │  │  │     · run queue         │ │ │
                                      │  │  │     · page faults       │ │ │
                                      │  │  │     · FS semantic index │ │ │
                                      │  │  │  3. generate response   │ │ │
                                      │  │  │  4. route decision:     │ │ │
                                      │  │  │     │                   │ │ │
                                      │  │  └─────┼───────────────────┘ │ │
                                      │  └────────┼─────────────────────┘ │
                                      │           │                       │
                                      └───────────┼───────────────────────┘
                                                  │
                                   ┌──────────────┼──────────────┐
                                   ▼              ▼              ▼
                                syscall        answer        network
                                (exec,         (text on      (fetch command
                                 open,          buddy>)        from internet,
                                 fork…)                        run it)

  Weights live here:  /sys/kai/model.bin   ← versioned by the OS, never sent anywhere
```

## Note

This model trains from scratch and can produce incorrect or unfiltered
content. If you choose to use or fine-tune it, results are not guaranteed
and you may not claim rights over it later.

## REPOSITORIES:
- [buddyOS](https://github.com/TamimDostyar/buddyOS)
- [buddyCLI](https://github.com/TamimDostyar/buddy)
