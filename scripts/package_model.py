"""Stage one registered BB8 model for container deployment."""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a trained BB8 model")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--tokenizer-type", default="char", choices=["char", "word", "bpe"])
    parser.add_argument("--target", default="deployment/model")
    args = parser.parse_args()

    source_dir = Path("checkpoints") / args.model_name
    target_dir = Path(args.target)
    target_dir.mkdir(parents=True, exist_ok=True)

    if (source_dir / "adapter_config.json").exists():
        copied_files = []
        for source_path in source_dir.iterdir():
            if source_path.is_file():
                target_path = target_dir / source_path.name
                shutil.copy2(source_path, target_path)
                copied_files.append(target_path)

        manifest = {
            "model_name": args.model_name,
            "backend": "hf_lora",
            "packaged_at": datetime.now(timezone.utc).isoformat(),
            "sha256": {
                path.name: sha256_file(path)
                for path in sorted(copied_files, key=lambda item: item.name)
            },
        }
        with (target_dir / "manifest.json").open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

        print(f"Packaged {args.model_name} -> {target_dir}")
        return

    source_checkpoint = source_dir / args.checkpoint
    source_tokenizer = source_dir / "tokenizer.json"
    for path in (source_checkpoint, source_tokenizer):
        if not path.exists():
            raise FileNotFoundError(f"Required model artifact not found: {path}")

    staged_checkpoint = target_dir / "model.pt"
    staged_tokenizer = target_dir / "tokenizer.json"
    shutil.copy2(source_checkpoint, staged_checkpoint)
    shutil.copy2(source_tokenizer, staged_tokenizer)

    source_config = source_dir / "config.yaml"
    if source_config.exists():
        shutil.copy2(source_config, target_dir / "config.yaml")

    manifest = {
        "model_name": args.model_name,
        "tokenizer_type": args.tokenizer_type,
        "checkpoint_source": args.checkpoint,
        "packaged_at": datetime.now(timezone.utc).isoformat(),
        "sha256": {
            "model.pt": sha256_file(staged_checkpoint),
            "tokenizer.json": sha256_file(staged_tokenizer),
        },
    }
    with (target_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Packaged {args.model_name} -> {target_dir}")


if __name__ == "__main__":
    main()
