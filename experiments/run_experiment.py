"""
run_experiment.py
-----------------
Config-driven experiment runner for BB8.

Usage
-----
    python experiments/run_experiment.py \\
        --config configs/default.yaml \\
        --data data/tiny_shakespeare.txt \\
        --name experiment_01

The script:
1. Loads a YAML config
2. Initialises the tokenizer and trains it on the data
3. Builds train/val DataLoaders
4. Constructs the BB8LM model
5. Trains the model with the Trainer
6. Runs evaluation and generates sample text
7. Saves a JSON results file to outputs/<name>/

All hyperparameters come from the YAML config so experiments are fully
reproducible.  Override any value by editing the config or duplicating it.
"""

import argparse
import json
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

# Make the parent directory importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.text_dataset import TextDataset, create_train_val_split
from evaluation.evaluator import Evaluator
from inference.generator import TextGenerator
from models.language_model import BB8LM
from tokenizer import BPETokenizer, CharTokenizer, WordTokenizer
from training.trainer import Trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def build_tokenizer(config: dict):
    tok_type = config.get("tokenizer", {}).get("type", "char").lower()
    if tok_type == "char":
        return CharTokenizer()
    if tok_type == "word":
        return WordTokenizer()
    if tok_type == "bpe":
        return BPETokenizer()
    raise ValueError(f"Unknown tokenizer type '{tok_type}'")


def save_results(results: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "results.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  ✓ Results saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run a BB8 training experiment")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--data", required=True, help="Path to training text file")
    parser.add_argument("--name", default="experiment", help="Experiment name (used for output dir)")
    parser.add_argument("--prompt", default="The ", help="Prompt for text generation demo")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  BB8 Experiment: {args.name}")
    print(f"  Config:  {args.config}")
    print(f"  Data:    {args.data}")
    print(f"{'='*60}\n")

    config = load_config(args.config)
    output_dir = os.path.join("outputs", args.name)
    checkpoint_dir = os.path.join("checkpoints", args.name)

    # ------------------------------------------------------------------
    # 1. Load raw text
    # ------------------------------------------------------------------
    with open(args.data, "r", encoding="utf-8") as fh:
        text = fh.read()
    print(f"Corpus loaded  |  {len(text):,} characters")

    # ------------------------------------------------------------------
    # 2. Tokenizer
    # ------------------------------------------------------------------
    tokenizer = build_tokenizer(config)
    vocab_size = config.get("tokenizer", {}).get("vocab_size") or 10_000
    tokenizer.train([text], vocab_size=vocab_size)
    print(tokenizer)

    # ------------------------------------------------------------------
    # 3. Dataset & DataLoaders
    # ------------------------------------------------------------------
    data_cfg = config.get("data", {})
    seq_len = data_cfg.get("seq_len", 256)
    val_frac = data_cfg.get("val_fraction", 0.1)

    dataset = TextDataset(text, tokenizer, seq_len=seq_len)
    train_ds, val_ds = create_train_val_split(dataset, val_frac)

    batch_size = config.get("training", {}).get("batch_size", 32)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0
    )

    # ------------------------------------------------------------------
    # 4. Model
    # ------------------------------------------------------------------
    model_cfg = config.get("model", {})
    model = BB8LM(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=model_cfg.get("d_model", 128),
        num_layers=model_cfg.get("num_layers", 4),
        num_heads=model_cfg.get("num_heads", 4),
        d_ff=model_cfg.get("d_ff", 512),
        max_seq_len=model_cfg.get("max_seq_len", seq_len),
        dropout=model_cfg.get("dropout", 0.1),
        activation=model_cfg.get("activation", "gelu"),
        tie_weights=model_cfg.get("tie_weights", True),
    )

    # ------------------------------------------------------------------
    # 5. Training
    # ------------------------------------------------------------------
    train_cfg = config.get("training", {})
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=train_cfg.get("lr", 3e-4),
        weight_decay=train_cfg.get("weight_decay", 0.1),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        warmup_steps=train_cfg.get("warmup_steps", 100),
        checkpoint_dir=checkpoint_dir,
        log_interval=train_cfg.get("log_interval", 10),
        eval_interval=train_cfg.get("eval_interval", 100),
        save_interval=train_cfg.get("save_interval", 500),
    )

    history = trainer.train(num_epochs=train_cfg.get("num_epochs", 10))

    # ------------------------------------------------------------------
    # 6. Final evaluation
    # ------------------------------------------------------------------
    print("\n--- Final Evaluation ---")
    evaluator = Evaluator(model, tokenizer)
    eval_results = evaluator.full_report(
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # ------------------------------------------------------------------
    # 7. Text generation
    # ------------------------------------------------------------------
    print("\n--- Text Generation ---")
    generator = TextGenerator(model, tokenizer)
    gen_cfg = config.get("generation", {})
    generated = generator.compare_strategies(
        prompt=args.prompt,
        max_new_tokens=gen_cfg.get("max_new_tokens", 150),
        temperature=gen_cfg.get("temperature", 0.8),
        top_k=gen_cfg.get("top_k", 50),
        top_p=gen_cfg.get("top_p", 0.9),
        repetition_penalty=gen_cfg.get("repetition_penalty", 1.1),
    )

    # ------------------------------------------------------------------
    # 8. Save results
    # ------------------------------------------------------------------
    results = {
        "experiment_name": args.name,
        "config": config,
        "tokenizer": {
            "type": config.get("tokenizer", {}).get("type", "char"),
            "vocab_size": tokenizer.get_vocab_size(),
        },
        "model_params": model.count_parameters(),
        "model_config": model.get_config(),
        "training_history": history,
        "evaluation": eval_results,
        "generated_samples": generated,
    }
    save_results(results, output_dir)

    print(f"\n{'='*60}")
    print(f"  Experiment '{args.name}' complete!")
    print(f"  Outputs → {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
