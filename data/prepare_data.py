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
import os
import urllib.request


DATASETS = {
    "tiny_shakespeare": {
        "url": (
            "https://raw.githubusercontent.com/karpathy/char-rnn"
            "/master/data/tinyshakespeare/input.txt"
        ),
        "filename": "tiny_shakespeare.txt",
        "description": "Complete works of Shakespeare (~1 MB)",
    },
}


def download(url: str, dest: str) -> None:
    """Download *url* to *dest* with a simple progress message."""
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    print(f"Downloading {url}")
    print(f"  → {dest}")
    urllib.request.urlretrieve(url, dest)
    size_mb = os.path.getsize(dest) / (1024 ** 2)
    print(f"  Downloaded {size_mb:.2f} MB")


def prepare(name: str, data_dir: str = "data") -> str:
    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}"
        )
    info = DATASETS[name]
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
    print(f"\nFirst 300 characters:\n{'-'*40}\n{text[:300]}\n{'-'*40}")


if __name__ == "__main__":
    main()
