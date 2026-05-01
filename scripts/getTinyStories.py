import argparse
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, "data"))
OUTPUT_PATH = os.path.join(DATA_DIR, "tinystories.txt")


def via_datasets(max_bytes: int, min_len: int, max_len: int) -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        return -1

    print("Loading TinyStories via Hugging Face datasets (streaming)...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    written = 0
    count = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in ds:
            story = (row.get("text") or "").strip()
            if not story:
                continue
            if len(story) < min_len or len(story) > max_len:
                continue
            story = " ".join(story.split())
            line = story + "\n"
            encoded = line.encode("utf-8")
            if written + len(encoded) > max_bytes:
                break
            f.write(line)
            written += len(encoded)
            count += 1
            if count % 1000 == 0:
                print(f"  ...{count} stories, {written / (1024 * 1024):.1f} MB")

    return count


def via_direct_download(max_bytes: int, min_len: int, max_len: int) -> int:
    import urllib.request
    import json

    url = (
        "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
        "TinyStoriesV2-GPT4-valid.txt"
    )
    print(f"Downloading (streaming) from {url} ...")
    count = 0
    written = 0
    try:
        with urllib.request.urlopen(url) as resp, open(
            OUTPUT_PATH, "w", encoding="utf-8"
        ) as out:
            buf = []
            for raw in resp:
                line = raw.decode("utf-8", errors="ignore").rstrip("\n")
                if line.strip() == "<|endoftext|>":
                    story = " ".join(buf).strip()
                    buf = []
                    if not story:
                        continue
                    if len(story) < min_len or len(story) > max_len:
                        continue
                    story = " ".join(story.split())
                    encoded = (story + "\n").encode("utf-8")
                    if written + len(encoded) > max_bytes:
                        return count
                    out.write(story + "\n")
                    written += len(encoded)
                    count += 1
                    if count % 1000 == 0:
                        print(
                            f"  ...{count} stories, {written / (1024 * 1024):.1f} MB"
                        )
                else:
                    buf.append(line)
    except Exception as e:
        print(f"Direct download failed: {e}", file=sys.stderr)
        return -1

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-mb",
        type=float,
        default=50.0,
        help="Max MB of text to save (default 50 for small-PC friendliness).",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=200,
        help="Min characters per story.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=2000,
        help="Max characters per story.",
    )
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    max_bytes = int(args.max_mb * 1024 * 1024)

    count = via_datasets(max_bytes, args.min_len, args.max_len)
    if count < 0:
        print("`datasets` not installed; falling back to direct HTTP download.")
        count = via_direct_download(max_bytes, args.min_len, args.max_len)

    if count <= 0:
        print(
            "Could not download TinyStories. Install `datasets` (pip install datasets) "
            "or check your internet connection.",
            file=sys.stderr,
        )
        sys.exit(1)

    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"Wrote {count} stories to {OUTPUT_PATH} ({size_mb:.1f} MB)")
    print("Run `python makeJson.py` to include it in pretrain.jsonl.")


if __name__ == "__main__":
    main()
