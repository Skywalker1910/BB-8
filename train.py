"""
train.py
--------
Quick-start training script for BB8.

Trains a BB8LM on any plain-text file with a single command.

Usage
-----
    # Minimal — character tokenizer, default small model
    python train.py --data data/tiny_shakespeare.txt

    # BPE tokenizer, medium model, 20 epochs
    python train.py --data data/tiny_shakespeare.txt \\
        --tokenizer bpe --vocab-size 3000 \\
        --d-model 256 --num-layers 6 --num-heads 8 --d-ff 1024 \\
        --seq-len 512 --epochs 20 --batch-size 16

    # Generate with a custom prompt after training
    python train.py --data data/tiny_shakespeare.txt --prompt "HAMLET:"
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from datasets.text_dataset import TextDataset, split_text
from evaluation.evaluator import Evaluator
from inference.generator import TextGenerator
from models.language_model import BB8LM
from tokenizer import BPETokenizer, CharTokenizer, WordTokenizer
from training.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Train BB8 Language Model")
    p.add_argument("--data", required=True, help="Path to training text file")
    p.add_argument(
        "--tokenizer", default="char", choices=["char", "word", "bpe"],
        help="Tokenization strategy"
    )
    p.add_argument("--vocab-size", type=int, default=5000,
                   help="Vocabulary size (for word/bpe tokenizers)")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--stride", type=int, default=1,
                   help="Tokens between adjacent training windows")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--mixed-precision", action="store_true",
                   help="Use float16 mixed precision when training on CUDA")
    p.add_argument("--checkpoint-dir", default="checkpoints/run")
    p.add_argument("--prompt", default="The ", help="Generation prompt")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    banner = "=" * 60
    print(f"\n{banner}")
    print("  BB8 Language Model — Training")
    print(banner)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    with open(args.data, "r", encoding="utf-8") as fh:
        text = fh.read()
    print(f"\nCorpus: {args.data}  |  {len(text):,} characters")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    if args.tokenizer == "char":
        tokenizer = CharTokenizer()
    elif args.tokenizer == "word":
        tokenizer = WordTokenizer()
    else:
        tokenizer = BPETokenizer()

    train_text, val_text = split_text(text, args.val_fraction)
    tokenizer.train([train_text], vocab_size=args.vocab_size)
    vocab_size = tokenizer.get_vocab_size()

    # ------------------------------------------------------------------
    # Dataset & DataLoaders
    # ------------------------------------------------------------------
    train_ds = TextDataset(
        train_text, tokenizer, seq_len=args.seq_len, stride=args.stride
    )
    val_ds = TextDataset(
        val_text, tokenizer, seq_len=args.seq_len, stride=args.stride
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=0, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=0, pin_memory=torch.cuda.is_available()
    )
    print(f"Train batches={len(train_loader)}  Val batches={len(val_loader)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = BB8LM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        max_steps=len(train_loader) * args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        mixed_precision=args.mixed_precision,
    )
    history = trainer.train(num_epochs=args.epochs)

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    print(f"\n{banner}")
    print("  Final Evaluation")
    print(banner)
    evaluator = Evaluator(model, tokenizer)
    evaluator.full_report(val_loader=val_loader)

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------
    print(f"\n{banner}")
    print("  Text Generation Examples")
    print(banner)
    generator = TextGenerator(model, tokenizer)
    generator.compare_strategies(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"\n{banner}")
    print("  Training complete!")
    print(f"  Checkpoints saved in: {args.checkpoint_dir}")
    print(banner)


if __name__ == "__main__":
    main()
