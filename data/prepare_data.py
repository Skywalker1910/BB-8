"""
prepare_data.py
---------------
Download and prepare sample datasets for BB8 training.

Available datasets
------------------
tiny_shakespeare
    ~1 MB of Shakespeare's complete works.
    A classic benchmark for character-level language modelling.
    Originally used by Andrej Karpathy in char-rnn.

Usage
-----
    python data/prepare_data.py --dataset tiny_shakespeare
    python data/prepare_data.py --dataset tiny_shakespeare --data-dir data/
"""

import argparse
import hashlib
import json
import os
import random
import sys
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path


DATASETS = {
    "tiny_shakespeare": {
        "url": (
            "https://raw.githubusercontent.com/karpathy/char-rnn"
            "/master/data/tinyshakespeare/input.txt"
        ),
        "filename": "tiny_shakespeare.txt",
        "description": "Complete works of Shakespeare (~1 MB)",
    },
    "dolly_15k": {
        "url": (
            "https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
            "resolve/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/"
            "databricks-dolly-15k.jsonl"
        ),
        "raw_filename": "databricks-dolly-15k.jsonl",
        "filename": "dolly_15k_chat.txt",
        "description": "Databricks Dolly 15K instruction/response corpus",
        "license": "CC BY-SA 3.0",
        "source_revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
    },
}


def download(url: str, dest: str) -> None:
    """Download *url* to *dest* with a simple progress message."""
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    print(f"Downloading {url}")
    print(f"  -> {dest}")
    urllib.request.urlretrieve(url, dest)
    size_mb = os.path.getsize(dest) / (1024 ** 2)
    print(f"  Downloaded {size_mb:.2f} MB")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_training_text(value: object) -> str:
    """Create a compact, deterministic alphabet for the educational BPE."""

    normalized = unicodedata.normalize("NFKD", str(value))
    return normalized.encode("ascii", errors="ignore").decode("ascii").strip()


def format_dolly(
    raw_path: str | Path,
    output_path: str | Path,
    *,
    seed: int = 42,
) -> dict:
    """Convert Dolly JSONL to deterministic USER/BB language-model text."""

    records = []
    categories: Counter[str] = Counter()
    with open(raw_path, "r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            instruction = normalize_training_text(record.get("instruction", ""))
            response = normalize_training_text(record.get("response", ""))
            context = normalize_training_text(record.get("context", ""))
            category = str(record.get("category", "unknown")).strip() or "unknown"
            if not instruction or not response:
                raise ValueError(f"Invalid Dolly record at line {line_number}")
            records.append(
                {
                    "instruction": instruction,
                    "context": context,
                    "response": response,
                    "category": category,
                }
            )
            categories[category] += 1

    random.Random(seed).shuffle(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            user_text = record["instruction"]
            if record["context"]:
                user_text += "\nCONTEXT: " + record["context"]
            fh.write(f"USER: {user_text}\nBB: {record['response']}\n\n")

    return {
        "records": len(records),
        "categories": dict(sorted(categories.items())),
        "shuffle_seed": seed,
        "raw_sha256": sha256_file(raw_path),
        "formatted_sha256": sha256_file(output_path),
        "formatted_characters": output_path.stat().st_size,
    }


def prepare_dolly(info: dict, data_dir: str) -> str:
    raw_path = os.path.join(data_dir, info["raw_filename"])
    output_path = os.path.join(data_dir, info["filename"])
    if not os.path.exists(raw_path):
        download(info["url"], raw_path)

    stats = format_dolly(raw_path, output_path)
    manifest = {
        "name": "databricks-dolly-15k",
        "source_url": info["url"],
        "source_revision": info["source_revision"],
        "license": info["license"],
        "attribution": "Copyright 2023 Databricks, Inc.",
        "format": "USER/BB causal-language-model transcript",
        "normalization": "Unicode NFKD followed by ASCII encoding with unsupported characters removed",
        **stats,
    }
    manifest_path = Path(data_dir) / "manifests" / "dolly_15k.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Prepared {stats['records']:,} records -> {output_path}")
    print(f"Manifest -> {manifest_path}")
    return output_path


def prepare(name: str, data_dir: str = "data") -> str:
    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}"
        )
    info = DATASETS[name]
    if name == "dolly_15k":
        return prepare_dolly(info, data_dir)
    dest = os.path.join(data_dir, info["filename"])
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / (1024 ** 2)
        print(f"Dataset already exists: {dest}  ({size_mb:.2f} MB)")
        return dest
    download(info["url"], dest)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BB8 training data")
    parser.add_argument(
        "--dataset",
        default="tiny_shakespeare",
        choices=list(DATASETS.keys()),
        help="Dataset to download",
    )
    parser.add_argument("--data-dir", default="data", help="Destination directory")
    args = parser.parse_args()

    path = prepare(args.dataset, args.data_dir)

    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()

    print(f"\nDataset summary")
    print(f"  Path       : {path}")
    print(f"  Characters : {len(text):,}")
    print(f"  Lines      : {text.count(chr(10)):,}")
    print(f"  Unique chars: {len(set(text))}")
    console_encoding = sys.stdout.encoding or "utf-8"
    preview = text[:300].encode(console_encoding, errors="replace").decode(console_encoding)
    print(f"\nFirst 300 characters:\n{'-'*40}\n{preview}\n{'-'*40}")


if __name__ == "__main__":
    main()
