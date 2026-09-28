"""Small, file-based experiment registry for BB8 training runs."""

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Dict

import torch


REGISTRY_PATH = os.path.join("experiments", "model_registry.jsonl")


def sha256_file(path: str) -> str:
    """Return a stable SHA-256 fingerprint for a dataset or artifact."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    """Return the current Git commit, or 'unknown' outside a Git checkout."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def git_is_dirty() -> bool:
    """Return whether tracked or untracked files differ from the commit."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def runtime_info(device: torch.device) -> Dict:
    """Describe the software and hardware used for a training run."""
    info = {
        "device": str(device),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info["gpu"] = properties.name
        info["gpu_memory_gb"] = round(properties.total_memory / 1024**3, 2)
    return info


def register_run(record: Dict, registry_path: str = REGISTRY_PATH) -> str:
    """Save a detailed run record and append its summary to the registry."""
    output_dir = record["artifacts"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    run_path = os.path.join(output_dir, "run.json")
    with open(run_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)

    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    with open(registry_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, separators=(",", ":")) + "\n")

    print(f"  [ok] run metadata -> {run_path}")
    print(f"  [ok] model registry -> {registry_path}")
    return run_path


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()
