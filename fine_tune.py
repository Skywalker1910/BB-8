"""LoRA fine-tuning for BB8's pretrained-model experiment track.

This script intentionally lives beside the from-scratch training pipeline. It
shows the industry post-training workflow without hiding or replacing BB8's
educational Transformer implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from experiments.tracking import (
    git_commit,
    git_is_dirty,
    register_run,
    runtime_info,
    sha256_file,
    utc_now,
)


PROMPT_TEMPLATE = "### Instruction:\n{instruction}{context}\n\n### Response:\n"
SMOKE_PROMPTS = [
    "Say hello in one short sentence.",
    "What is 2 + 2?",
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
    "Write a Python function that adds two numbers.",
]


def _optional_dependencies():
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.optimization import get_cosine_schedule_with_warmup
    except ImportError as exc:
        raise RuntimeError(
            "Fine-tuning dependencies are missing. Run: "
            "pip install -r requirements-finetune.txt"
        ) from exc
    return (
        LoraConfig,
        TaskType,
        get_peft_model,
        AutoModelForCausalLM,
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
    )


def configure_system_trust_store() -> None:
    """Use the operating-system CA store without weakening TLS verification."""

    try:
        import truststore
    except ImportError as exc:
        raise RuntimeError(
            "The truststore dependency is missing. Run: "
            "pip install -r requirements-finetune.txt"
        ) from exc
    truststore.inject_into_ssl()


def load_records(path: str | Path, seed: int) -> list[dict[str, str]]:
    """Load, validate, and deterministically shuffle Dolly-style JSONL."""

    records: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            instruction = str(raw.get("instruction", "")).strip()
            response = str(raw.get("response", "")).strip()
            if not instruction or not response:
                raise ValueError(f"Invalid record at line {line_number}")
            records.append(
                {
                    "instruction": instruction,
                    "context": str(raw.get("context", "")).strip(),
                    "response": response,
                }
            )
    random.Random(seed).shuffle(records)
    return records


def split_records(
    records: list[dict[str, str]], validation_fraction: float
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    validation_size = max(1, round(len(records) * validation_fraction))
    return records[validation_size:], records[:validation_size]


def format_prompt(record: dict[str, str]) -> str:
    context = record["context"]
    context_block = f"\n\n### Context:\n{context}" if context else ""
    return PROMPT_TEMPLATE.format(
        instruction=record["instruction"],
        context=context_block,
    )


class InstructionDataset(Dataset):
    """Tokenized SFT examples with loss applied only to assistant tokens."""

    def __init__(self, records: list[dict[str, str]], tokenizer: Any, max_length: int,
                 preprocessing: str = "full_example_v2"):
        if max_length < 32:
            raise ValueError("max_length must be at least 32")
        self.examples: list[dict[str, list[int]]] = []
        self.truncated_examples = 0
        self.skipped_ids = []
        if preprocessing not in {"full_example_v2", "legacy_v1"}:
            raise ValueError("Unknown instruction preprocessing version")
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("The base tokenizer must define an EOS token")

        for index, record in enumerate(records):
            prompt_ids = tokenizer.encode(format_prompt(record), add_special_tokens=False)
            response_ids = tokenizer.encode(record["response"], add_special_tokens=False)
            response_ids.append(eos_id)

            original_length = len(prompt_ids) + len(response_ids)
            if preprocessing == "full_example_v2":
                if original_length > max_length:
                    self.skipped_ids.append(index)
                    continue
                self.examples.append({"input_ids": prompt_ids + response_ids,
                                      "attention_mask": [1] * original_length,
                                      "labels": [-100] * len(prompt_ids) + response_ids.copy()})
                continue
            prompt_budget = min(len(prompt_ids), max_length // 2)
            prompt_ids = prompt_ids[:prompt_budget]
            response_ids = response_ids[: max_length - len(prompt_ids)]
            if original_length > max_length:
                self.truncated_examples += 1

            input_ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + response_ids.copy()
            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


class InstructionCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(item["input_ids"]) for item in examples)

        def pad(values: list[int], fill: int) -> list[int]:
            return values + [fill] * (max_length - len(values))

        return {
            "input_ids": torch.tensor(
                [pad(item["input_ids"], self.pad_token_id) for item in examples],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [pad(item["attention_mask"], 0) for item in examples],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [pad(item["labels"], -100) for item in examples],
                dtype=torch.long,
            ),
        }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def precision_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def evaluate(model, loader, device: torch.device, precision: str) -> dict[str, float]:
    model.eval()
    total_nll = 0.0
    supervised_tokens = 0
    for batch in tqdm(loader, desc="Validation", leave=False):
        batch = move_batch(batch, device)
        token_count = int((batch["labels"] != -100).sum().item())
        with precision_context(device, precision):
            loss = model(**batch).loss
        total_nll += float(loss.item()) * token_count
        supervised_tokens += token_count
    average_loss = total_nll / max(supervised_tokens, 1)
    return {
        "loss": average_loss,
        "perplexity": math.exp(min(average_loss, 20.0)),
        "supervised_tokens": supervised_tokens,
    }


@torch.no_grad()
def generate_smoke_samples(model, tokenizer, device: torch.device, config: dict):
    model.eval()
    generation = config["generation"]
    samples: dict[str, str] = {}
    for instruction in SMOKE_PROMPTS:
        prompt = PROMPT_TEMPLATE.format(instruction=instruction, context="")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(
            **inputs,
            max_new_tokens=generation["max_new_tokens"],
            do_sample=True,
            temperature=generation["temperature"],
            top_p=generation["top_p"],
            repetition_penalty=generation["repetition_penalty"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        continuation = output[0, inputs["input_ids"].shape[1] :]
        samples[instruction] = tokenizer.decode(
            continuation, skip_special_tokens=True
        ).strip()
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained BB8 model with LoRA")
    parser.add_argument("--config", default="configs/qwen_lora_v004.yaml")
    parser.add_argument("--name", default="bb8-qwen-lora-v004-dev")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    output_dir = Path("outputs") / args.name
    checkpoint_dir = Path("checkpoints") / args.name
    if output_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError(
            f"Run '{args.name}' already exists. Use a new name; runs are immutable."
        )

    configure_system_trust_store()
    (
        LoraConfig,
        TaskType,
        get_peft_model,
        AutoModelForCausalLM,
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
    ) = _optional_dependencies()

    data_config = config["data"]
    train_config = config["training"]
    model_config = config["base_model"]
    lora_config = config["lora"]
    seed = int(data_config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("v004 LoRA training requires a CUDA GPU")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    chat_data = data_config.get("preprocessing") == "chat_full_v1"
    if chat_data:
        from grounded.training import ChatDataset, read_jsonl
        data_manifest = json.loads(Path(data_config['manifest']).read_text(encoding='utf-8'))
        for filename, digest in data_manifest['files'].items():
            if sha256_file(str(Path(data_config['manifest']).parent / filename)) != digest:
                raise ValueError(f'Frozen grounded dataset changed: {filename}')
        train_records = read_jsonl(data_config['path'])
        validation_records = read_jsonl(data_config['validation_path'])
        if {r['group'] for r in train_records} & {r['group'] for r in validation_records}:
            raise ValueError('Training and validation groups overlap')
    else:
        records = load_records(data_config["path"], seed)
        train_records, validation_records = split_records(records, float(data_config["validation_fraction"]))
    print(
        f"Records: {len(train_records):,} train / "
        f"{len(validation_records):,} validation"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        cache_dir=model_config["cache_dir"],
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_type = ChatDataset if chat_data else InstructionDataset
    dataset_options = {} if chat_data else {'preprocessing':data_config.get('preprocessing','full_example_v2')}
    train_dataset = dataset_type(train_records, tokenizer, int(data_config["max_length"]), **dataset_options)
    validation_dataset = dataset_type(validation_records, tokenizer, int(data_config["max_length"]), **dataset_options)
    if not len(train_dataset) or not len(validation_dataset):
        raise ValueError('No complete examples remain; increase context or filter the data explicitly')
    output_dir.mkdir(parents=True)
    shutil.copy2(args.config, output_dir / 'config.yaml')
    shutil.copy2(data_config['manifest'], output_dir / 'dataset_manifest.json')
    for source in ['fine_tune.py', 'grounded/training.py', 'grounded/core.py', 'grounded/data.py']:
        target = output_dir / 'source_snapshot' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (output_dir / 'preprocessing.json').write_text(json.dumps({
        'version':data_config.get('preprocessing','full_example_v2'),
        'train_retained':len(train_dataset),'validation_retained':len(validation_dataset),
        'train_skipped_ids':train_dataset.skipped_ids,'validation_skipped_ids':validation_dataset.skipped_ids,
    },indent=2),encoding='utf-8')
    print(
        f"Truncated: {train_dataset.truncated_examples:,} train / "
        f"{validation_dataset.truncated_examples:,} validation"
    )

    collator = InstructionCollator(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(train_config["eval_batch_size"]),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    precision = str(train_config["precision"])
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        cache_dir=model_config["cache_dir"],
        dtype=dtype,
    )
    model.config.use_cache = False
    if train_config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    peft_configuration = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_config["rank"]),
        lora_alpha=int(lora_config["alpha"]),
        lora_dropout=float(lora_config["dropout"]),
        target_modules=list(lora_config["target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, peft_configuration).to(device)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters())
    print(
        f"Parameters: {total_parameters:,} total / "
        f"{trainable_parameters:,} trainable "
        f"({trainable_parameters / total_parameters:.2%})"
    )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    accumulation_steps = int(train_config["gradient_accumulation_steps"])
    epochs = int(train_config["epochs"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_updates = updates_per_epoch * epochs
    warmup_steps = round(total_updates * float(train_config["warmup_ratio"]))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_updates
    )

    started_at = utc_now()
    started_timer = time.perf_counter()
    history: list[dict[str, float]] = []
    global_update = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_index, batch in enumerate(progress, start=1):
            batch = move_batch(batch, device)
            with precision_context(device, precision):
                loss = model(**batch).loss
            # Final partial accumulation group must not be underweighted.
            group_start = ((batch_index - 1) // accumulation_steps) * accumulation_steps
            group_size = min(accumulation_steps, len(train_loader) - group_start)
            (loss / group_size).backward()
            running_loss += float(loss.item())

            should_update = (
                batch_index % accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            if should_update:
                clip_grad_norm_(model.parameters(), float(train_config["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1
                if global_update % int(train_config["log_interval"]) == 0:
                    progress.set_postfix(
                        loss=f"{running_loss / batch_index:.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )

        validation = evaluate(model, validation_loader, device, precision)
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(len(train_loader), 1),
            "validation_loss": validation["loss"],
            "validation_perplexity": validation["perplexity"],
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, indent=2))

    duration_seconds = round(time.perf_counter() - started_timer, 2)
    checkpoint_dir.mkdir(parents=True)
    model_metadata = {
        "model_name": args.name,
        "backend": "hf_lora",
        "prompt_style": "chat_template" if chat_data else "instruction",
        "max_context_tokens": int(data_config["max_length"]),
        "base_model": model_config,
        "experiment": config["experiment"],
    }
    (checkpoint_dir / "bb8_model_config.json").write_text(
        json.dumps(model_metadata, indent=2),
        encoding="utf-8",
    )
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_dir)

    smoke_samples = {} if chat_data else generate_smoke_samples(model, tokenizer, device, config)
    for prompt, response in smoke_samples.items():
        print(f"\nQ: {prompt}\nA: {response}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "run_id": args.name,
        "base_model": model_config,
        "lora": lora_config,
        "records": {
            "train": len(train_records),
            "validation": len(validation_records),
        },
        "parameters": {
            "total": total_parameters,
            "trainable": trainable_parameters,
        },
        "history": history,
        "smoke_samples": smoke_samples,
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    record = {
        "run_id": args.name,
        "started_at": started_at,
        "completed_at": utc_now(),
        "duration_seconds": duration_seconds,
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "seed": seed,
        "config_sha256": sha256_file(args.config),
        "source_hashes": {str(p.relative_to(output_dir / 'source_snapshot')): sha256_file(str(p))
                          for p in (output_dir / 'source_snapshot').rglob('*.py')},
        "optimizer_updates": global_update,
        "experiment": config["experiment"],
        "base_model": model_config,
        "dataset": {
            "path": data_config["path"],
            "sha256": sha256_file(data_config["path"]),
            "manifest": data_config["manifest"],
            "train_records": len(train_records),
            "validation_records": len(validation_records),
            "validation_sha256": sha256_file(data_config['validation_path']) if chat_data else None,
            "preprocessing": data_config.get('preprocessing','full_example_v2'),
        },
        "lora": lora_config,
        "training": train_config,
        "parameters": results["parameters"],
        "runtime": runtime_info(device),
        "evaluation": history[-1],
        "artifacts": {
            "adapter_sha256": sha256_file(str(checkpoint_dir / "adapter_model.safetensors")),
            "adapter": str(checkpoint_dir / "adapter_model.safetensors"),
            "checkpoint_dir": str(checkpoint_dir),
            "results": str(results_path),
            "output_dir": str(output_dir),
        },
    }
    register_run(record)
    print(f"\nCompleted {args.name} in {duration_seconds:.1f} seconds")


if __name__ == "__main__":
    main()
