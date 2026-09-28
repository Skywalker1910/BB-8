"""Upload BB8 checkpoints, outputs, and data manifests to Hugging Face Hub."""

import os
import sys
import shutil

import truststore
truststore.inject_into_ssl()

from huggingface_hub import HfApi, create_repo


REPO_ID = "Skywalker1910/BB8"
REPO_TYPE = "model"


def main():
    api = HfApi()
    whoami = api.whoami()
    print(f"Logged in as: {whoami['name']}")

    # Create repo (no-op if exists)
    create_repo(REPO_ID, repo_type=REPO_TYPE, exist_ok=True, private=False)
    print(f"Repo ready: https://huggingface.co/{REPO_ID}")

    # Copy model card as README.md for HF
    model_card_src = os.path.join(os.path.dirname(__file__), "..", "MODEL_CARD.md")
    model_card_dst = os.path.join(os.path.dirname(__file__), "..", "HF_README.md")
    shutil.copy2(model_card_src, model_card_dst)

    # Upload README (model card)
    print("\n--- Uploading model card ---")
    api.upload_file(
        path_or_fileobj=model_card_dst,
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )
    os.remove(model_card_dst)

    # Upload all checkpoints
    checkpoints_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    for run_name in sorted(os.listdir(checkpoints_dir)):
        run_path = os.path.join(checkpoints_dir, run_name)
        if not os.path.isdir(run_path):
            continue
        print(f"\n--- Uploading checkpoint: {run_name} ---")
        api.upload_folder(
            folder_path=run_path,
            path_in_repo=f"checkpoints/{run_name}",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )

    # Upload all outputs (results.json, run.json, etc.)
    outputs_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    for run_name in sorted(os.listdir(outputs_dir)):
        run_path = os.path.join(outputs_dir, run_name)
        if not os.path.isdir(run_path):
            continue
        print(f"\n--- Uploading output: {run_name} ---")
        api.upload_folder(
            folder_path=run_path,
            path_in_repo=f"outputs/{run_name}",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )

    # Upload configs
    configs_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
    print(f"\n--- Uploading configs ---")
    api.upload_folder(
        folder_path=configs_dir,
        path_in_repo="configs",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )

    # Upload data manifests (not the raw data files)
    manifests_dir = os.path.join(os.path.dirname(__file__), "..", "data", "manifests")
    if os.path.isdir(manifests_dir):
        print(f"\n--- Uploading data manifests ---")
        api.upload_folder(
            folder_path=manifests_dir,
            path_in_repo="data/manifests",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )

    # Upload experiment registry
    registry = os.path.join(os.path.dirname(__file__), "..", "experiments", "model_registry.jsonl")
    if os.path.exists(registry):
        print(f"\n--- Uploading experiment registry ---")
        api.upload_file(
            path_or_fileobj=registry,
            path_in_repo="experiments/model_registry.jsonl",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )

    print(f"\n{'='*60}")
    print(f"  Upload complete!")
    print(f"  https://huggingface.co/{REPO_ID}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
